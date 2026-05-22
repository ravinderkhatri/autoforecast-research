"""Compact count forecaster with independently supervised aggregate feedback.

Torch-only. SFT supplies the prediction law and local event helpers, not the backbone.
The development recipe and pre-implementation cases are in the model regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from forecast.models.mft.branches import CalendarEventBranch, StaticIdentityBranch
from forecast.models.mft.heads import _validated_series_index
from forecast.models.mft.sft import (
    SFT_LOG_K_MAX,
    SFT_LOG_K_MIN,
    SFTModel,
    ShiftedHurdlePrediction,
)
from torch import Tensor, nn


def tensor_summary(value: Tensor) -> dict:
    """Bounded diagnostic record, with nonfinite refusal; test_hmfnet_diagnostics."""
    value = value.detach()
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite HMFNet stage output")
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        finite=True,
        minimum=float(value.min()) if value.numel() else None,
        maximum=float(value.max()) if value.numel() else None,
        mean=float(value.double().mean()) if value.numel() else None,
    )


class MaskedLocalBlock(nn.Module):
    """Band-local depthwise convolution and token-wise MLP; test_masked_local_block."""

    def __init__(self, width: int) -> None:
        """Ordinary three-offset convolution; test_masked_local_block checks its coefficients."""
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )

    def convolve(self, values: Tensor, mask: Tensor, bands: Tensor, regular: Tensor) -> Tensor:
        """Never pack across missing tokens or band edges; test_masked_local_block."""
        if (
            values.ndim != 3
            or mask.shape != values.shape[:2]
            or mask.dtype != torch.bool
            or bands.shape != values.shape[1:2]
            or regular.shape != bands.shape
            or bands.dtype != torch.int64
            or regular.dtype != torch.bool
            or (bands < 0).any()
            or (bands > 2).any()
            or (bands[1:] < bands[:-1]).any()
        ):
            raise ValueError("invalid local memory masks or band ordering")
        result = torch.zeros_like(values)
        for band in range(3):
            at = torch.nonzero(bands == band, as_tuple=False).flatten()
            if not len(at):
                continue
            active = mask[:, at] & regular[at][None, :]
            safe = torch.where(active[..., None], values[:, at], 0)
            convolved = self.conv(safe.transpose(1, 2)).transpose(1, 2)
            result[:, at] = torch.where(active[..., None], convolved, 0)
        return result

    def forward(self, values: Tensor, mask: Tensor, bands: Tensor, regular: Tensor) -> Tensor:
        """Reapply availability after each residual block; test_masked_local_block."""
        values = torch.where(mask[..., None], values, 0)
        values = values + self.convolve(self.norm(values), mask, bands, regular)
        return torch.where(mask[..., None], values + self.mlp(values), 0)


@dataclass(frozen=True)
class HMFNetPrediction(ShiftedHurdlePrediction):
    """The inherited mean/NLL plus non-published expert diagnostics; test_hmfnet_distribution."""

    expert_mean: Tensor | None = None
    expert_residual: Tensor | None = None
    gates: Tensor | None = None


class HMFNetModel(SFTModel):
    """Reuse SFT's local event helpers only; construct a separate compact backbone.

    No SFT demand branch, branch fusion, trunk or decoder is instantiated. Tests named
    test_hmfnet_forward, test_hmfnet_experts, test_hmfnet_feedback and
    test_hmfnet_event_locality cover it.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        local_blocks: int,
        n_token_features: int,
        n_statistics: int,
        n_attributes: int,
        n_series: int,
        n_known: int,
        n_aggregate_known: int,
        horizon: int,
        identity_cardinalities: tuple[int, ...],
        categorical_cardinalities: tuple[int, ...],
        categorical_width: int,
        n_types: int,
        n_context_groups: int,
        n_expert_groups: int,
        feedback: bool,
        n_events: int,
        n_categories: int,
        event_max_support: int,
        event_kernel_layers: int,
        excess_floor: float,
        head_kind: str,
        embedding_residual_dim: int = 16,
    ) -> None:
        nn.Module.__init__(self)
        required = (
            d_model,
            n_heads,
            local_blocks,
            n_token_features,
            n_statistics,
            n_series,
            n_known,
            n_aggregate_known,
            horizon,
            categorical_width,
            event_max_support,
        )
        if (
            any(type(v) is not int or v < 1 for v in required)
            or d_model % n_heads
            or any(
                type(v) is not int or v < 0
                for v in (
                    n_attributes,
                    n_types,
                    n_context_groups,
                    n_expert_groups,
                    n_events,
                    n_categories,
                )
            )
            or type(feedback) is not bool
            or head_kind not in {"negbin", "shifted_hurdle_negbin"}
            or not 0 < excess_floor <= 1e-3
            or max(n_context_groups, n_expert_groups) > n_types
            or any(type(v) is not int or v < 1 for v in categorical_cardinalities)
        ):
            raise ValueError("invalid HMFNet dimensions, head or feedback configuration")
        self.horizon, self.head_kind, self.excess_floor = horizon, head_kind, excess_floor
        self.n_token_features, self.n_statistics = n_token_features, n_statistics
        self.n_known, self.n_aggregate_known = n_known, n_aggregate_known
        self.n_attributes = n_attributes
        self.n_context_groups, self.n_expert_groups = n_context_groups, n_expert_groups
        self.n_categories, self.event_max_support = n_categories, event_max_support
        self.feedback_enabled = feedback and n_expert_groups > 0
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(n, categorical_width) for n in categorical_cardinalities
        )
        self.token_projection = nn.Linear(
            n_token_features + len(categorical_cardinalities) * categorical_width, d_model
        )
        self.local_blocks = nn.ModuleList(MaskedLocalBlock(d_model) for _ in range(local_blocks))
        self.static = StaticIdentityBranch(
            n_attributes,
            d_model,
            identity_cardinalities=identity_cardinalities,
            residual_dim=embedding_residual_dim,
        )
        self.statistics = nn.Sequential(
            nn.Linear(n_statistics, d_model), nn.SiLU(), nn.LayerNorm(d_model)
        )
        self.known_projection = nn.Linear(n_known, d_model)
        self.context_projection = nn.Linear(d_model + 1, d_model)
        self.step_embedding = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.decoder_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.n_types = n_types
        self.type_embedding = nn.Embedding(n_types, d_model) if n_expert_groups else None
        self.aggregate_known = nn.Linear(n_aggregate_known, d_model) if n_expert_groups else None
        self.expert_size = nn.Linear(d_model, 1) if n_expert_groups else None
        if n_expert_groups:
            self.feedback_phi = nn.Sequential(
                nn.Linear(1 + d_model + 3, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
            )
            self.gate_bottom = nn.Linear(d_model, d_model, bias=False)
            self.gate_context = nn.Linear(d_model, d_model, bias=False)
            self.gate_type = nn.Linear(d_model, d_model, bias=False)
            self.gate_quality = nn.Linear(3, d_model)
            self.gate_score = nn.Linear(d_model, 1, bias=False)
            self.null_gate = nn.Linear(d_model, 1)
        head_width = d_model + n_statistics + (d_model if n_expert_groups else 0)
        self.occurrence = nn.Linear(head_width, 1) if head_kind == "shifted_hurdle_negbin" else None
        self.size = nn.Linear(head_width, 1)
        self.dispersion = nn.Linear(n_attributes, 1)
        self.dispersion_residual = nn.Embedding(n_series, 1)
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
            self.calendar.project = nn.Linear(n_events, 2, bias=False)
            nn.init.zeros_(self.calendar.project.weight)
            nn.init.ones_(self.calendar.shared)
        for layer in (self.occurrence, self.size, self.dispersion, self.expert_size):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.dispersion_residual.weight)
        self.last_diagnostics: dict = {}
        self._aggregate_cache: dict[str, Tensor] | None = None

    def encode(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Shared masked memory and pooled historical context; test_hmfnet_forward."""
        value, mask = inputs["memory_values"], inputs["memory_mask"]
        ids, present = inputs["memory_categorical"], inputs["memory_categorical_mask"]
        if (
            value.ndim != 3
            or value.shape[-1] != self.n_token_features
            or mask.shape != value.shape[:2]
            or mask.dtype != torch.bool
            or ids.shape != (*value.shape[:2], len(self.category_embeddings))
            or present.shape != ids.shape
            or present.dtype != torch.bool
            or ids.dtype != torch.int64
        ):
            raise ValueError("invalid HMFNet token shape or categorical mask")
        parts = [torch.where(mask[..., None], value, 0)]
        for column, embedding in enumerate(self.category_embeddings):
            active = present[..., column] & mask
            index = torch.where(active, ids[..., column], 0)
            if (index < 0).any() or (index >= embedding.num_embeddings).any():
                raise ValueError("historical category exceeds frozen vocabulary")
            parts.append(torch.where(active[..., None], embedding(index), 0))
        safe = torch.cat(parts, -1)
        if not torch.isfinite(safe).all():
            raise ValueError("nonfinite admitted HMFNet memory feature")
        memory = torch.where(mask[..., None], self.token_projection(safe), 0)
        for layer in self.local_blocks:
            memory = layer(memory, mask, inputs["memory_bands"], inputs["memory_regular"])
        pooled = memory.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        return memory, pooled

    def decode(self, query: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        """Direct horizon cross-attention, with an all-missing null slot; test_hmfnet_forward."""
        null = memory.new_zeros((memory.shape[0], 1, memory.shape[-1]))
        safe_memory = torch.cat((memory, null), 1)
        safe_mask = torch.cat((mask, ~mask.any(1, keepdim=True)), 1)
        query = query + self.step_embedding
        crossed = self.cross_attention(
            self.query_norm(query),
            safe_memory,
            safe_memory,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )[0]
        hidden = query + crossed
        return hidden + self.decoder_mlp(hidden)

    def aggregate_forecasts(self, aggregate: dict[str, Tensor]) -> dict[str, Tensor]:
        """Independent aggregate futures from their histories; test_hmfnet_experts."""
        if self._aggregate_cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError("learned aggregate cache is inference-only")
        if not self.n_types:
            raise ValueError("aggregate inputs supplied to a model without aggregate types")
        type_index = aggregate["type_index"]
        if (
            type_index.shape != aggregate["memory_values"].shape[:1]
            or type_index.dtype != torch.int64
            or (type_index < 0).any()
            or (type_index >= self.n_types).any()
        ):
            raise ValueError("invalid aggregate type indices")
        n = len(type_index)
        known, statistics, quality = (
            aggregate["known_future"],
            aggregate["statistics"],
            aggregate["quality"],
        )
        node_ids = aggregate["node_ids"]
        if (
            known.shape != (n, self.horizon, self.n_aggregate_known)
            or statistics.shape != (n, self.n_statistics)
            or quality.shape != (n, 3)
            or node_ids.shape != (n,)
            or node_ids.dtype != torch.int64
            or (node_ids < 0).any()
            or len(torch.unique(node_ids)) != n
        ):
            raise ValueError("invalid aggregate horizons, statistics, quality or node identities")
        if not all(torch.isfinite(value).all() for value in (known, statistics, quality)):
            raise ValueError("nonfinite aggregate input")
        anchor = aggregate["anchor"]
        if (
            anchor.shape != (len(type_index), self.horizon)
            or not torch.isfinite(anchor).all()
            or (anchor <= 0).any()
        ):
            raise ValueError("expert anchor must be positive finite and horizon-aligned")
        if self._aggregate_cache is not None:
            if (node_ids >= len(self._aggregate_cache["mean"])).any():
                raise ValueError("cached aggregate node identities outside the frozen population")
            return {name: value[node_ids] for name, value in self._aggregate_cache.items()}
        memory, context = self.encode(aggregate)
        if self.expert_size is None:
            return dict(
                mean=aggregate["anchor"],
                residual=torch.zeros_like(aggregate["anchor"]),
                context=context,
                typed=torch.zeros_like(context),
                available=aggregate["memory_mask"].any(1),
                quality=aggregate["quality"],
            )
        assert self.type_embedding is not None and self.aggregate_known is not None
        typed = self.type_embedding(type_index)
        query = (
            self.aggregate_known(aggregate["known_future"])
            + typed[:, None]
            + self.statistics(aggregate["statistics"])[:, None]
        )
        hidden = self.decode(query, memory, aggregate["memory_mask"])
        residual = (
            self.expert_size(hidden).squeeze(-1)
            if self.expert_size is not None
            else hidden.new_zeros(hidden.shape[:2])
        )
        mean = (anchor.double().log() + residual.double()).exp()
        if not torch.isfinite(mean).all() or (mean < torch.finfo(torch.float64).tiny ** 0.5).any():
            raise ValueError("expert forecast outside finite safe mean domain")
        return dict(
            mean=mean,
            residual=residual,
            context=context,
            typed=typed,
            available=aggregate["memory_mask"].any(1),
            quality=aggregate["quality"],
        )

    def feedback(
        self,
        hidden: Tensor,
        aggregate: dict[str, Tensor],
        memberships: Tensor,
        *,
        zero_residual: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Horizon-dependent gate plus always-available null; test_hmfnet_feedback."""
        if (
            memberships.shape != (hidden.shape[0], self.n_expert_groups)
            or memberships.dtype != torch.int64
            or (memberships < 0).any()
            or (memberships >= len(aggregate["mean"])).any()
        ):
            raise ValueError("invalid expert membership map")
        context = aggregate["context"][memberships]
        typed = aggregate["typed"][memberships]
        quality = aggregate["quality"][memberships]
        available = aggregate["available"][memberships]
        score = self.gate_score(
            torch.tanh(
                self.gate_bottom(hidden)[:, :, None]
                + self.gate_context(context)[:, None]
                + self.gate_type(typed)[:, None]
                + self.gate_quality(quality)[:, None]
            )
        ).squeeze(-1)
        score = score.masked_fill(~available[:, None], float("-inf"))
        gates = torch.softmax(torch.cat((score, self.null_gate(hidden)), -1), -1)
        residual = aggregate["residual"][memberships].transpose(1, 2)
        if zero_residual:
            residual = torch.zeros_like(residual)
        phi_input = torch.cat(
            (
                residual[..., None],
                typed[:, None].expand(-1, self.horizon, -1, -1),
                quality[:, None].expand(-1, self.horizon, -1, -1),
            ),
            -1,
        )
        signal = self.feedback_phi(phi_input)
        feedback = (gates[..., :-1, None] * signal).sum(2)
        return feedback, gates

    def forward(
        self,
        *,
        memory: dict[str, Tensor],
        attributes: Tensor,
        series_index: Tensor,
        identity_indices: Tensor,
        known_future: Tensor,
        baseline_logit: Tensor,
        baseline_log_size: Tensor,
        event_distance: Tensor,
        event_present: Tensor,
        category_index: Tensor | None = None,
        aggregate: dict[str, Tensor] | None = None,
        context_memberships: Tensor | None = None,
        expert_memberships: Tensor | None = None,
        zero_expert_residual: bool = False,
    ) -> HMFNetPrediction:
        """Feedback enters parameters, never post-hoc means; test_hmfnet_distribution."""
        n = len(series_index)
        if (
            attributes.shape != (n, self.n_attributes)
            or known_future.shape != (n, self.horizon, self.n_known)
            or baseline_logit.shape != (n, self.horizon)
            or baseline_log_size.shape != baseline_logit.shape
            or memory["statistics"].shape != (n, self.n_statistics)
            or event_distance.shape[:2] != (n, self.horizon)
            or self.head_kind != "negbin"
            and (baseline_log_size < 0).any()
        ):
            raise ValueError("HMFNet input horizons, statistics or baselines disagree")
        for value in (
            attributes,
            known_future,
            baseline_logit,
            baseline_log_size,
            memory["statistics"],
        ):
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite HMFNet input")
        _validated_series_index(series_index, self.dispersion_residual, attributes.new_zeros(n))
        encoded, _ = self.encode(memory)
        identity = self.static(attributes, identity_indices=identity_indices)
        if context_memberships is None:
            if self.n_context_groups:
                raise ValueError("configured aggregate context requires its context membership map")
        elif (
            context_memberships.shape != (n, self.n_context_groups)
            or context_memberships.dtype != torch.int64
            or (context_memberships < 0).any()
        ):
            raise ValueError("invalid context membership map")
        if (self.n_context_groups or self.n_expert_groups) and aggregate is None:
            raise ValueError("configured aggregate routes require aggregate inputs")
        experts = self.aggregate_forecasts(aggregate) if aggregate is not None else None
        context = encoded.new_zeros((n, encoded.shape[-1]))
        coverage = encoded.new_zeros((n, 1))
        if self.n_context_groups:
            assert context_memberships is not None
            if (
                experts is None
                or (context_memberships >= len(experts["context"])).any()
            ):
                raise ValueError("invalid context membership map")
            valid = experts["available"][context_memberships]
            context = torch.where(valid[..., None], experts["context"][context_memberships], 0).sum(
                1
            )
            context = context / valid.sum(1).clamp_min(1)[:, None]
            coverage = valid.float().mean(1, keepdim=True)
        query = (
            self.known_projection(known_future)
            + identity[:, None]
            + self.statistics(memory["statistics"])[:, None]
            + self.context_projection(torch.cat((context, coverage), -1))[:, None]
        )
        hidden = self.decode(query, encoded, memory["memory_mask"])
        head = [hidden, memory["statistics"][:, None].expand(-1, self.horizon, -1)]
        gates = hidden.new_ones((n, self.horizon, 1))
        feedback = torch.zeros_like(hidden)
        if self.n_expert_groups:
            if experts is None or expert_memberships is None:
                raise ValueError("configured expert heads require their membership inputs")
            feedback, gates = self.feedback(
                hidden, experts, expert_memberships, zero_residual=zero_expert_residual
            )
            if not self.feedback_enabled:
                feedback = torch.zeros_like(feedback)
            head.append(feedback)
        head_context = torch.cat(head, -1)
        event = self.event_residual(event_distance, event_present, category_index)
        logit = (
            baseline_logit + self.occurrence(head_context).squeeze(-1) + event[..., 0]
            if self.occurrence is not None
            else torch.zeros_like(baseline_logit)
        )
        baseline_excess = (
            baseline_log_size.double().exp()
            if self.head_kind == "negbin"
            else torch.expm1(baseline_log_size.double()).clamp_min(self.excess_floor)
        )
        excess = (
            baseline_excess.log() + self.size(head_context).squeeze(-1).double() + event[..., 1]
        ).exp()
        log_k = self.dispersion(attributes) + self.dispersion_residual(series_index)
        dispersion = log_k.double().clamp(SFT_LOG_K_MIN, SFT_LOG_K_MAX).exp().expand_as(logit)
        prediction = HMFNetPrediction(
            logit,
            excess,
            dispersion,
            self.head_kind,
            experts["mean"] if experts else None,
            experts["residual"] if experts else None,
            gates,
        )
        self.last_diagnostics = {
            name: tensor_summary(value)
            for name, value in (
                (
                    "memory_input",
                    torch.where(memory["memory_mask"][..., None], memory["memory_values"], 0),
                ),
                ("encoded_memory", encoded),
                ("decoded_horizon", hidden),
                ("memory_mask", memory["memory_mask"]),
                ("level_statistics", memory["statistics"]),
                ("feedback", feedback),
                ("gates", gates),
                ("occurrence_logit", logit),
                ("positive_excess_mean", excess),
                ("dispersion", dispersion),
                ("expected_counts", prediction.mean()),
            )
        }
        if experts:
            self.last_diagnostics["expert_means"] = tensor_summary(experts["mean"])
            self.last_diagnostics["expert_log_residuals"] = tensor_summary(experts["residual"])
            self.last_diagnostics["aggregate_historical_context"] = tensor_summary(
                experts["context"]
            )
            self.last_diagnostics["aggregate_available"] = tensor_summary(experts["available"])
        return prediction
