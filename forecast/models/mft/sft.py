"""Accuracy-oriented SFT built from the existing multi-branch Torch components.

The positive distribution is deliberately versioned: 1 + NB, not zero-truncated NB.
``test_sft.py`` checks the baseline, likelihood, feature paths and complete event support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from forecast.models.mft.branches import (
    CalendarEventBranch,
    CommercialBranch,
    DemandHistoryBranch,
    StaticIdentityBranch,
    _binary_mask,
)
from forecast.models.mft.decoder import Decoder
from forecast.models.mft.fusion import GatedFusion
from forecast.models.mft.heads import _validated_series_index
from forecast.models.mft.trunk import Trunk
from torch import Tensor, nn
from torch.nn import functional as F


def level_statistics(
    history: Tensor,
    observed: Tensor,
    windows: tuple[int, ...] = (7, 28, 56),
    available: Tensor | None = None,
) -> Tensor:
    """Twelve origin-known level/shape features; tested by ``test_level_statistics``."""
    if history.ndim != 2 or observed.shape != history.shape or history.shape[1] == 0:
        raise ValueError("history and observation mask must be matching nonempty matrices")
    if not torch.isfinite(history).all() or (history < 0).any():
        raise ValueError("history must contain finite nonnegative counts")
    mask = _binary_mask(observed, history, "history observation mask", error_type=ValueError)
    positions = (
        torch.ones_like(mask)
        if available is None
        else _binary_mask(
            available, history, "history calendar-position mask", error_type=ValueError
        )
    )
    if bool((mask & ~positions).any()) or bool((positions[:, :-1] & ~positions[:, 1:]).any()):
        raise ValueError("history availability must be a left-padded calendar suffix")
    available_count = positions.sum(1)
    if bool((available_count == 0).any()):
        raise ValueError("history needs at least one available calendar position")
    value = torch.where(mask, history, torch.zeros_like(history))
    observed_count = mask.sum(1)
    denominator = observed_count.clamp_min(1)
    mean = value.sum(1) / denominator
    variance = (
        torch.where(mask, (history - mean[:, None]).square(), 0.0).sum(1)
        / denominator
    )
    positive = mask & (history > 0)
    positive_count = positive.sum(1)
    positive_mean = value.sum(1) / positive_count.clamp_min(1)
    index = torch.arange(history.shape[1], device=history.device)
    last = torch.where(positive, index[None, :] + 1, 0).amax(1)
    features = [
        torch.log1p(mean),
        torch.log1p(torch.sqrt(variance + 1e-8)),
        positive_count / denominator,
        torch.log1p(positive_mean),
        observed_count / available_count,
        torch.log1p(positive_count.float()),
        torch.where(
            last > 0,
            (history.shape[1] - last) / available_count,
            torch.ones_like(last),
        ),
    ]
    if any(type(width) is not int or width < 1 for width in windows):
        raise ValueError("statistic windows must be positive integer periods")
    for width in windows:
        recent = value[:, -width:].sum(1) / mask[:, -width:].sum(1).clamp_min(1)
        features.append(torch.log1p(recent) - torch.log1p(mean))
    features.extend([torch.log1p(value[:, -1]), torch.log1p(value.amax(1)) - torch.log1p(mean)])
    return torch.stack(features, dim=-1)


SFT_LOG_K_MIN, SFT_LOG_K_MAX = -4.0, 4.0


@dataclass(frozen=True)
class ShiftedHurdlePrediction:
    """Occurrence logit, positive NB excess mean and dispersion; all [series, horizon]."""

    logit: Tensor
    excess: Tensor
    dispersion: Tensor
    head_kind: str = "shifted_hurdle_negbin"

    def _validate_parameters(self) -> None:
        """One numerical domain for mean/export and likelihood, including standalone callers."""
        if self.head_kind not in {"negbin", "shifted_hurdle_negbin"}:
            raise ValueError("unsupported SFT count distribution")
        if (
            self.excess.shape != self.logit.shape
            or self.dispersion.shape != self.logit.shape
            or not torch.isfinite(self.logit).all()
            or not torch.isfinite(self.excess).all()
            or not torch.isfinite(self.dispersion).all()
            or (self.excess <= 0).any()
            or (self.dispersion <= 0).any()
        ):
            raise ValueError("shifted hurdle needs finite parameters and positive means/dispersion")
        if (
            not self.excess.is_floating_point()
            or self.excess.dtype not in (torch.float32, torch.float64)
            or (self.excess < torch.finfo(self.excess.dtype).tiny ** 0.5).any()
        ):
            raise ValueError("unsafe mean precision; construct float64 means from log predictors")
        if self.dispersion.dtype not in (torch.float32, torch.float64):
            raise ValueError("dispersion must use float32 or float64 precision")
        log_k = self.dispersion.double().log()
        tolerance = 8 * torch.finfo(self.dispersion.dtype).eps
        if ((log_k < SFT_LOG_K_MIN - tolerance) | (log_k > SFT_LOG_K_MAX + tolerance)).any():
            raise ValueError("dispersion is outside the supported SFT log range [-4, 4]")

    def mean(self) -> Tensor:
        """Differentiable float64 mean; log-space product preserves rare/large cancellation."""
        self._validate_parameters()
        if self.head_kind == "negbin":
            return self.excess.double()
        return (F.logsigmoid(self.logit.double()) + self.excess.double().log1p()).exp()

    def nll(self, target: Tensor, observed: Tensor) -> Tensor:
        """Per-cell masked likelihood, using Torch's maintained NB implementation."""
        if target.shape != self.logit.shape or observed.shape != target.shape:
            raise ValueError("target, mask and prediction shapes differ")
        observed = _binary_mask(
            observed, target, "likelihood observation mask", error_type=ValueError
        )
        self._validate_parameters()
        result = self.logit.double() * 0.0
        target = target[observed]
        if (
            not torch.isfinite(target).all()
            or (target < 0).any()
            or (target != target.floor()).any()
        ):
            raise ValueError("shifted hurdle targets must be finite nonnegative integers")
        if not observed.any():
            return result
        positive = target > 0
        # Evaluate only observed labels; missing counts never enter the distribution.
        # Float64 preserves high-count lgamma differences and neural gradients.
        target = target.double()
        excess, dispersion, logit = (
            value[observed].double() for value in (self.excess, self.dispersion, self.logit)
        )
        distribution = torch.distributions.NegativeBinomial(
            total_count=dispersion,
            logits=torch.log(excess) - torch.log(dispersion),
            validate_args=True,
        )
        if self.head_kind == "negbin":
            return result.masked_scatter(observed, -distribution.log_prob(target))
        count_log_prob = distribution.log_prob((target - 1.0).clamp_min(0.0))
        log_prob = torch.where(positive, F.logsigmoid(logit) + count_log_prob, F.logsigmoid(-logit))
        return result.masked_scatter(observed, -log_prob)



class SFTModel(nn.Module):
    """Single event route, seasonal mean anchor, level inputs and aggregate history.

    Counts and identities remain bottom-level. No peer-product or latent-factor claim is made.
    Initial residuals are zero: the encoder starts receiving loss gradients after the output
    weights first move. The explicit excess floor only approximates the positive-mean-one
    boundary; it is recorded in the comparison profile, not hidden in prediction clipping.
    """

    def __init__(
        self,
        *,
        d_model: int,
        encoder_days: int,
        n_attributes: int,
        n_series: int,
        n_products: int,
        n_stores: int,
        n_categories: int,
        n_known: int,
        horizon: int,
        n_events: int,
        event_max_support: int,
        n_heads: int,
        trunk_stages: int,
        decoder_layers: int,
        history_attention_layers: int,
        demand_dropout: float,
        excess_floor: float,
        aggregate_history: bool = True,
        commercial_attention_layers: int = 1,
        commercial_attention_heads: int = 2,
        event_kernel_layers: int = 1,
        patch_size: int = 7,
        statistic_windows: tuple[int, ...] = (7, 28, 56),
        n_commercial: int = 3,
        identity_cardinalities: tuple[int, ...] = (),
        n_history_covariates: int = 0,
        embedding_residual_dim: int = 16,
        head_kind: str = "shifted_hurdle_negbin",
        calendar_availability: bool = False,
    ) -> None:
        super().__init__()
        integers = (
            d_model,
            encoder_days,
            n_series,
            n_known,
            horizon,
            event_max_support,
            n_heads,
            trunk_stages,
            decoder_layers,
            history_attention_layers,
            commercial_attention_layers,
            commercial_attention_heads,
            event_kernel_layers,
            patch_size,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("SFT dimensions must be positive integers")
        nonnegative = (
            n_attributes,
            n_products,
            n_stores,
            n_categories,
            n_commercial,
            n_history_covariates,
            n_events,
            embedding_residual_dim,
        )
        if (
            any(type(n) is not int or n < 0 for n in nonnegative)
            or encoder_days % patch_size
            or d_model % n_heads
            or not 0 <= demand_dropout < 1
            or not 0 < excess_floor <= 1e-3
            or type(aggregate_history) is not bool
            or type(calendar_availability) is not bool
            or head_kind not in {"negbin", "shifted_hurdle_negbin"}
            or any(type(w) is not int or w < 1 for w in statistic_windows)
        ):
            raise ValueError("invalid SFT event, patch, dropout or numerical configuration")
        if n_commercial and d_model % commercial_attention_heads:
            raise ValueError("commercial attention heads must divide d_model")
        self.excess_floor = excess_floor
        self.event_max_support = event_max_support
        self.horizon = horizon
        self.n_categories = n_categories
        self.head_kind = head_kind
        self.statistic_windows = statistic_windows
        self.calendar_availability = calendar_availability
        n_statistics = 9 + len(statistic_windows)
        self.demand = DemandHistoryBranch(
            d_model,
            encoder_days,
            n_heads=n_heads,
            dropout=demand_dropout,
            attention_layers=history_attention_layers,
            weekly_patch=patch_size,
            n_covariates=n_history_covariates,
            position_availability=calendar_availability,
        )
        self.static = StaticIdentityBranch(
            n_attributes,
            d_model,
            n_products=n_products,
            n_stores=n_stores,
            n_series=n_series if not n_products and not identity_cardinalities else 0,
            identity_cardinalities=identity_cardinalities,
            residual_dim=embedding_residual_dim,
        )
        self.commercial = (
            CommercialBranch(
                n_commercial,
                d_model,
                n_heads=commercial_attention_heads,
                attention_layers=commercial_attention_layers,
            )
            if n_commercial
            else None
        )
        self.statistics = nn.Sequential(
            nn.Linear(n_statistics, d_model), nn.SiLU(), nn.LayerNorm(d_model)
        )
        self.aggregate = (
            DemandHistoryBranch(
                d_model,
                encoder_days,
                n_heads=n_heads,
                dropout=demand_dropout,
                attention_layers=1,
                weekly_patch=patch_size,
                position_availability=calendar_availability,
            )
            if aggregate_history
            else None
        )
        self.aggregate_level = nn.Linear(2, d_model) if aggregate_history else None
        self.n_branches = 3 + int(self.commercial is not None) + int(aggregate_history)
        self.fusion = GatedFusion(self.n_branches, d_model, n_heads=n_heads, d_static=n_attributes)
        self.trunk = Trunk(d_model, d_static=n_attributes, n_stages=trunk_stages, n_heads=n_heads)
        self.decoder = Decoder(
            d_model, n_known, n_layers=decoder_layers, n_heads=n_heads, max_horizon=horizon
        )
        self.calendar = (
            CalendarEventBranch(
                n_events,
                2,
                max_support=event_max_support,
                n_series=n_categories,
                kernel_layers=event_kernel_layers,
            )
            if n_events
            else None
        )
        if self.calendar is not None:
            # A never-observed projection stays zero even when other events train the shared
            # kernel. Keep that kernel live so zero projections can receive first-step gradients.
            projection = nn.Linear(n_events, 2, bias=False)
            nn.init.zeros_(projection.weight)
            nn.init.ones_(self.calendar.shared)
            self.calendar.project = projection
        self.occurrence = (
            nn.Linear(d_model + n_statistics, 1) if head_kind == "shifted_hurdle_negbin" else None
        )
        self.size = nn.Linear(d_model + n_statistics, 1)
        self.dispersion = nn.Linear(n_attributes, 1)
        self.dispersion_residual = nn.Embedding(n_series, 1)
        for layer in (self.occurrence, self.size, self.dispersion):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.dispersion_residual.weight)

    def event_residual(self, distance: Tensor, present: Tensor, category: Tensor | None) -> Tensor:
        """Per-day, explicitly supported correction; no event reaches memory or attention."""
        if distance.shape != present.shape or distance.ndim != 3:
            raise ValueError("event distances/presence must be matching [series, horizon, event]")
        if not torch.isfinite(distance).all():
            raise ValueError("event distances must be finite")
        present = _binary_mask(present, distance, "event presence mask", error_type=ValueError)
        if self.calendar is None:
            if category is not None and (
                category.shape != distance.shape[:1]
                or category.dtype not in (torch.int32, torch.int64)
                or category.device != distance.device
                or (category < 0).any()
                or (category >= self.n_categories).any()
            ):
                raise ValueError("category_index must contain one valid integral index per series")
            if distance.shape[-1] != 0:
                raise ValueError("event inputs supplied to an eventless model")
            return distance.new_zeros((*distance.shape[:2], 2))
        if self.calendar.series_deviation is not None:
            if category is None:
                raise ValueError("category_index is required by the declared event pool")
            _validated_series_index(category, self.calendar.series_deviation, distance.sum((1, 2)))
        supported = present & (distance.abs() <= self.event_max_support)
        response = self.calendar(distance, supported, series_index=category)
        return torch.where(supported.any(-1, keepdim=True), response, 0.0)

    def forward(
        self,
        *,
        history: Tensor,
        observed: Tensor,
        attributes: Tensor,
        series_index: Tensor,
        product_index: Tensor | None = None,
        store_index: Tensor | None = None,
        category_index: Tensor | None = None,
        known_future: Tensor,
        commercial: Tensor,
        commercial_mask: Tensor,
        baseline_logit: Tensor,
        baseline_log_size: Tensor,
        event_distance: Tensor,
        event_present: Tensor,
        aggregate_history: Tensor | None = None,
        aggregate_observed: Tensor | None = None,
        identity_indices: Tensor | None = None,
        history_covariates: Tensor | None = None,
        history_covariate_mask: Tensor | None = None,
        history_available: Tensor | None = None,
    ) -> ShiftedHurdlePrediction:
        """Emit the exact factorless mean and a coherent positive-count distribution."""
        if (
            known_future.shape[:2] != (history.shape[0], self.horizon)
            or baseline_logit.shape != known_future.shape[:2]
            or baseline_log_size.shape != baseline_logit.shape
            or event_distance.shape[:2] != known_future.shape[:2]
        ):
            raise ValueError("SFT horizon and per-day baseline shapes disagree")
        tensors = (attributes, known_future, commercial, baseline_logit, baseline_log_size)
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("SFT features must be finite; missingness travels in masks")
        if self.head_kind != "negbin" and (baseline_log_size < 0).any():
            raise ValueError("conditional positive baseline mean must be at least one")
        if self.calendar_availability and history_available is None:
            raise ValueError("global-left-padding model requires history_available")
        if not self.calendar_availability and history_available is not None:
            raise ValueError("history_available requires the global-left-padding model")
        available = (
            torch.ones_like(history, dtype=torch.bool)
            if history_available is None
            else _binary_mask(
                history_available,
                history,
                "history calendar-position mask",
                error_type=ValueError,
            )
        )
        statistics = level_statistics(
            history, observed, self.statistic_windows, available
        )
        # Reuse the legacy head's exact-shape/dtype/device/range contract for every lookup.
        # A singleton embedding row must never broadcast across a larger batch.
        cells = history[:, 0]
        _validated_series_index(series_index, self.dispersion_residual, cells)
        if self.static.residual_product is not None:
            assert self.static.residual_store is not None
            if product_index is None or store_index is None:
                raise ValueError(
                    "product/store indices are required by the declared identity tables"
                )
            _validated_series_index(product_index, self.static.residual_product, cells)
            _validated_series_index(store_index, self.static.residual_store, cells)
        demand_token, sequence = self.demand(
            history,
            observed,
            covariates=history_covariates,
            covariate_mask=history_covariate_mask,
            position_mask=available if self.calendar_availability else None,
        )
        identity = self.static(
            attributes,
            series_index=series_index,
            product_index=product_index,
            store_index=store_index,
            identity_indices=identity_indices,
        )
        tokens = [demand_token, identity]
        if self.commercial is not None:
            commercial_available = _binary_mask(
                commercial_mask, commercial, "commercial availability mask", error_type=ValueError
            )
            safe_mask = commercial_available.clone()
            missing = ~safe_mask.any(-1)
            safe_mask[missing, 0] = True
            safe_commercial = torch.where(commercial_available, commercial, 0.0)
            commercial_token = self.commercial(safe_commercial, safe_mask)
            commercial_token = torch.where(missing[:, None], 0.0, commercial_token)
            tokens.append(commercial_token)
        elif commercial.numel() or commercial_mask.numel():
            raise ValueError("commercial inputs supplied to a disabled branch")
        tokens.append(self.statistics(statistics))
        aggregate_sequence = None
        if self.aggregate is not None:
            assert self.aggregate_level is not None
            if aggregate_history is None or aggregate_observed is None:
                raise ValueError("aggregate branch requires history and its observation mask")
            aggregate_token, aggregate_sequence = self.aggregate(
                aggregate_history,
                aggregate_observed,
                position_mask=available if self.calendar_availability else None,
            )
            group_stats = level_statistics(
                aggregate_history, aggregate_observed, available=available
            )[:, :2]
            tokens.append(aggregate_token + self.aggregate_level(group_stats))
        fused = self.fusion(torch.stack(tokens, dim=1), static=attributes)
        carried = fused[:, :, None, :].expand(-1, -1, sequence.shape[1], -1).clone()
        carried[:, 0] = carried[:, 0] + sequence
        if aggregate_sequence is not None:
            carried[:, -1] = carried[:, -1] + aggregate_sequence
        weekly_available = available.reshape(
            history.shape[0], sequence.shape[1], -1
        ).any(-1)
        memory = self.trunk(
            carried,
            attributes,
            time_mask=weekly_available if self.calendar_availability else None,
        ).flatten(1, 2)
        memory_available = weekly_available[:, None, :].expand(
            -1, self.n_branches, -1
        ).reshape(history.shape[0], -1)
        hidden = self.decoder(
            memory,
            known_future,
            memory_mask=memory_available if self.calendar_availability else None,
        )
        head_context = torch.cat((hidden, statistics[:, None, :].expand(-1, self.horizon, -1)), -1)
        event = self.event_residual(event_distance, event_present, category_index)
        logit = (
            baseline_logit + self.occurrence(head_context).squeeze(-1) + event[..., 0]
            if self.occurrence is not None
            else torch.zeros_like(baseline_logit)
        )
        baseline_excess = (
            baseline_log_size.exp()
            if self.head_kind == "negbin"
            else torch.expm1(baseline_log_size).clamp_min(self.excess_floor)
        )
        log_excess = (
            torch.log(baseline_excess) + self.size(head_context).squeeze(-1) + event[..., 1]
        )
        log_k = self.dispersion(attributes) + self.dispersion_residual(series_index)
        dispersion = log_k.double().clamp(SFT_LOG_K_MIN, SFT_LOG_K_MAX).exp().expand_as(logit)
        excess = log_excess.double().exp()
        if (
            not torch.isfinite(excess).all()
            or not torch.isfinite(logit).all()
            or (excess < torch.finfo(excess.dtype).tiny ** 0.5).any()
        ):
            raise ValueError(
                "nonfinite SFT predictors or numerically unsafe count mean; "
                "refusing to clip a collapsed forecast"
            )
        return ShiftedHurdlePrediction(logit, excess, dispersion, self.head_kind)

    def event_penalty(self) -> Tensor:
        """Partially pooled event-parameter shrinkage, separately weighted by the run profile."""
        if self.calendar is None:
            return self.size.weight.new_zeros(())
        values = [parameter.square().mean() for parameter in self.calendar.parameters()]
        return torch.stack(values).mean()
