"""Chronological, fixed-epoch/update-budget development training shared by MFT/SFT/HMFNet/PACT.

No model imports: the sole architecture bridge supplies the callbacks. See
``test_multibranch_revisions.py`` and ``test_hierarchy_metric.py`` for chronology,
deterministic refit and full-group gradients.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from autoforecast.hierarchy_metric import HierarchyMetric, TorchHierarchyMetric
from autoforecast.hmfnet_features import HMFNetConfig
from autoforecast.pact_features import PACTConfig
from autoforecast.revision_config import validate_revision_recipe, verify_validation_ledger
from autoforecast.rolling_data import GLOBAL_LEFT_PADDING_CONTRACT, PanelArrays, ValidationWindow
from autoforecast.series_loss_weighting import SERIES_LOSS_WEIGHTING_MODES
from autoforecast.torch_performance import TorchPerformanceProfiler
from autoforecast.training_checkpoint import (
    capture_training_rng,
    configure_training_runtime,
    restore_training_rng,
)
from autoforecast.training_progress import TrainingProgress


@dataclass(frozen=True)
class RollingConfig:
    """Fully serialized development recipe; unknown JSON fields fail at construction."""

    variant: str
    encoder_days: int
    d_model: int = 64
    n_heads: int = 4
    trunk_stages: int = 2
    decoder_layers: int = 2
    history_attention_layers: int = 1
    commercial_attention_layers: int = 1
    commercial_attention_heads: int = 2
    event_kernel_layers: int = 1
    demand_dropout: float = 0.1
    covariate_dropout: float = 0.2
    factor_rank: int = 0
    factor_draws: int = 1
    train_windows: int = 52
    train_stride: int = 28
    max_updates: int = 1200
    batch_size: int = 512
    evaluate_every: int = 100
    validation_every_epochs: int = 1
    validation_backend: str = "torch"
    checkpoint_every_updates: int = 100
    checkpoint_every_seconds: float = 300.0
    checkpoint_keep_last: int = 2
    progress: str = "auto"
    patience: int = 3
    early_stopping_absolute_min_delta: float = 0.0
    early_stopping_relative_min_delta: float = 0.0
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    mean_loss_weight: float = 0.2
    hierarchy_loss_weight: float = 0.1
    hierarchy_every: int = 100
    hierarchy_epsilon: float = 1e-8
    event_penalty_weight: float = 0.0001
    price_penalty_weight: float = 0.0
    event_max_support: int = 14
    excess_floor: float = 1e-6
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 4
    max_array_bytes: int = 1_500_000_000
    aggregate_history: bool = True
    epochs: int | None = None
    head: str = "auto"
    patch_size: int = 7
    statistic_windows: tuple[int, ...] = (7, 28, 56)
    seasonal_window: int = 28
    reference_window: int = 28
    selection_metric: str = "auto"
    mean_weighting: str = "auto"
    series_loss_weighting: str = "uniform"
    hierarchy_weighting: str = "auto"
    identity_columns: tuple[str, ...] = ()
    factor_columns: tuple[str, ...] = ()
    aggregate_columns: tuple[str, ...] = ()
    baseline_columns: tuple[str, ...] = ()
    event_pool_columns: tuple[str, ...] = ()
    calendar_group_column: str | None = None
    observation_contract: str = "masked_observed_counts_v1"
    cycles: tuple[tuple[str, float, int], ...] | None = None
    v3: HMFNetConfig | None = None
    v5: PACTConfig | None = None

    def __post_init__(self) -> None:
        """Reject silent/bool numeric coercions and unsupported mixed-version recipes."""
        if isinstance(self.v3, dict):
            object.__setattr__(self, "v3", HMFNetConfig(**self.v3))
        if isinstance(self.v5, dict):
            object.__setattr__(self, "v5", PACTConfig(**self.v5))
        if (
            (self.variant == "v3") != isinstance(self.v3, HMFNetConfig)
            or self.variant != "v3"
            and self.v3 is not None
        ):
            raise ValueError("only HMFNet requires an explicit HMFNet feature/objective recipe")
        if (
            (self.variant == "v5") != isinstance(self.v5, PACTConfig)
            or self.variant != "v5"
            and self.v5 is not None
        ):
            raise ValueError("only PACT requires an explicit PACT architecture recipe")
        for name in (
            "statistic_windows",
            "identity_columns",
            "factor_columns",
            "aggregate_columns",
            "baseline_columns",
            "event_pool_columns",
        ):
            value = getattr(self, name)
            if not isinstance(value, list | tuple):
                raise ValueError(f"{name} must be a sequence")
            object.__setattr__(self, name, tuple(value))
        if self.cycles is not None:
            if not isinstance(self.cycles, list | tuple) or any(
                not isinstance(value, list | tuple) for value in self.cycles
            ):
                raise ValueError("cycles must be a sequence of seasonal definitions")
            cycles = tuple(tuple(value) for value in self.cycles)
            if any(
                len(value) != 3
                or not isinstance(value[0], str)
                or not value[0].strip()
                or type(value[1]) not in (int, float)
                or not math.isfinite(value[1])
                or value[1] <= 0
                or type(value[2]) is not int
                or value[2] < 1
                for value in cycles
            ) or len({value[0] for value in cycles}) != len(cycles):
                raise ValueError(
                    "cycles require unique non-blank names, positive periods and harmonics"
                )
            object.__setattr__(self, "cycles", cycles)
        if self.calendar_group_column is not None and (
            not isinstance(self.calendar_group_column, str)
            or not self.calendar_group_column.strip()
        ):
            raise ValueError("calendar_group_column must be a non-blank column name or None")
        if self.epochs is not None and (type(self.epochs) is not int or self.epochs < 1):
            raise ValueError("epochs must be a positive integer or None")
        if (
            self.head not in {"auto", "hurdle_negbin", "negbin"}
            or self.selection_metric not in {"auto", "wrmsse", "rmse"}
            or self.mean_weighting not in {"auto", "revenue", "uniform"}
            or self.series_loss_weighting not in SERIES_LOSS_WEIGHTING_MODES
            or self.hierarchy_weighting not in {"auto", "revenue", "uniform"}
            or self.validation_backend not in {"torch", "numpy"}
            or self.progress not in {"auto", "plain", "off"}
            or self.observation_contract
            not in {
                "masked_observed_counts_v1",
                "complete_recorded_sales_prelaunch_zeros_v1",
                GLOBAL_LEFT_PADDING_CONTRACT,
            }
            or any(type(v) is not int or v < 1 for v in self.statistic_windows)
        ):
            raise ValueError("unsupported head, metric, window or observation contract")
        for name in (
            "identity_columns",
            "factor_columns",
            "aggregate_columns",
            "baseline_columns",
            "event_pool_columns",
        ):
            values = getattr(self, name)
            if len(set(values)) != len(values) or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):
                raise ValueError(f"{name} needs unique nonempty column names")
        integers = (
            "encoder_days",
            "d_model",
            "n_heads",
            "trunk_stages",
            "decoder_layers",
            "history_attention_layers",
            "commercial_attention_layers",
            "commercial_attention_heads",
            "event_kernel_layers",
            "factor_draws",
            "train_windows",
            "train_stride",
            "max_updates",
            "batch_size",
            "evaluate_every",
            "validation_every_epochs",
            "checkpoint_every_updates",
            "checkpoint_keep_last",
            "patience",
            "hierarchy_every",
            "event_max_support",
            "torch_threads",
            "max_array_bytes",
            "patch_size",
            "seasonal_window",
            "reference_window",
        )
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in integers):
            raise ValueError("revision counts and budgets must be positive integers")
        if any(type(v) is not int or v < 0 for v in (self.seed, self.factor_rank)):
            raise ValueError("seed and factor_rank must be nonnegative integers")
        numbers = (
            self.demand_dropout,
            self.covariate_dropout,
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip,
            self.mean_loss_weight,
            self.hierarchy_loss_weight,
            self.hierarchy_epsilon,
            self.event_penalty_weight,
            self.excess_floor,
            self.price_penalty_weight,
            self.checkpoint_every_seconds,
            self.early_stopping_absolute_min_delta,
            self.early_stopping_relative_min_delta,
        )
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in numbers):
            raise ValueError("revision numerical controls must be finite nonnegative numbers")
        if (
            self.variant not in {"v1_corrected", "v2", "v3", "v5"}
            or self.device not in {"cpu", "cuda"}
            or type(self.aggregate_history) is not bool
            or self.encoder_days % self.patch_size
            or self.d_model % self.n_heads
            or self.variant == "v5"
            and (self.d_model // self.n_heads) % 2
            or self.demand_dropout >= 1
            or self.covariate_dropout >= 1
            or min(self.learning_rate, self.gradient_clip, self.hierarchy_epsilon) <= 0
            or not 0 < self.excess_floor <= 1e-3
            or self.checkpoint_keep_last < 2
            or self.checkpoint_every_seconds <= 0
        ):
            raise ValueError("unsupported revision topology, device or numerical control")
        if self.variant == "v1_corrected" and (
            self.mean_loss_weight
            or self.hierarchy_loss_weight
            or self.aggregate_history
            or self.event_penalty_weight
            or self.observation_contract == GLOBAL_LEFT_PADDING_CONTRACT
        ):
            raise ValueError(
                "corrected MFT retains the likelihood-only, non-aggregate, unpadded backbone"
            )
        if self.variant in {"v2", "v3", "v5"} and (
            self.factor_rank or self.factor_draws != 1 or self.price_penalty_weight
        ):
            raise ValueError("SFT/HMFNet/PACT are factorless and have no legacy price penalty")
        if self.variant == "v3" and (
            self.v3 is None
            or self.demand_dropout != 0
            or self.decoder_layers != 1
            or self.history_attention_layers != 1
            or self.commercial_attention_layers != 1
            or self.trunk_stages != self.v3.local_blocks
        ):
            raise ValueError(
                "HMFNet requires zero demand dropout and its compact local/direct topology"
            )
        if self.variant == "v5" and (
            self.v5 is None
            or self.demand_dropout != 0
            or self.covariate_dropout != 0
            or self.decoder_layers != 1
            or self.history_attention_layers != 1
            or self.commercial_attention_layers != 1
            or self.trunk_stages != self.v5.n_layers
            or self.aggregate_history != bool(self.v5.context_levels)
            or self.mean_loss_weight != 0
            or self.hierarchy_loss_weight != 0
            or self.event_penalty_weight != 0
            or self.head != ("negbin" if self.v5.head_law == "negbin" else "hurdle_negbin")
            or self.train_windows < 7
            or math.gcd(self.train_stride, 7) != 1
        ):
            raise ValueError(
                "PACT requires its joint likelihood, explicit head law, all cutoff weekdays, "
                "factorless topology and matching registered stream/depth recipe"
            )
        validate_revision_recipe(asdict(self))


@dataclass(frozen=True)
class RollingTrainingContract:
    """Governed optimization mode, separate from the model architecture recipe."""

    mode: str = "rolling_backtest"
    fixed_validation: ValidationWindow | None = None
    fixed_update_budget: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"rolling_backtest", "fixed_validation", "fixed_refit"}:
            raise ValueError("unsupported rolling training mode")
        if self.mode == "fixed_validation":
            if self.fixed_validation is None or self.fixed_update_budget is not None:
                raise ValueError("fixed_validation requires one holdout and no fixed update budget")
            window = self.fixed_validation
            if (
                window.target.ndim != 2
                or window.target.shape != window.observed.shape
                or window.observed.dtype != np.bool_
                or window.objective_mask is not None
                and (
                    window.objective_mask.shape != window.target.shape
                    or window.objective_mask.dtype != np.bool_
                    or np.any(window.objective_mask & ~window.observed)
                )
            ):
                raise ValueError("fixed validation arrays are invalid")
            if window.study is not None and (
                not np.array_equal(window.target, window.study.target, equal_nan=True)
                or not np.array_equal(window.observed, window.study.observed)
            ):
                raise ValueError("fixed validation differs from its frozen study")
        elif self.fixed_validation is not None:
            raise ValueError("only fixed_validation may carry holdout labels")
        if self.mode == "fixed_refit":
            if type(self.fixed_update_budget) is not int or self.fixed_update_budget < 1:
                raise ValueError("fixed_refit requires a positive optimizer-update budget")
        elif self.fixed_update_budget is not None:
            raise ValueError("only fixed_refit may carry an optimizer-update budget")

    def as_record(self) -> dict[str, object]:
        """Serialize decisions without duplicating holdout labels in metadata."""
        return {
            "mode": self.mode,
            "validation_origin": (
                None if self.fixed_validation is None else self.fixed_validation.origin
            ),
            "fixed_update_budget": self.fixed_update_budget,
            **({"study_evaluation": self.fixed_validation.study.as_record()}
               if self.fixed_validation is not None and self.fixed_validation.study is not None
               else {}),
        }


def rolling_origins(data: PanelArrays, end: str, config: RollingConfig) -> tuple[str, ...]:
    """Training targets end by ``end``; refuse insufficient multi-year coverage."""
    stop = data.position(end)
    indices = [stop - data.horizon - j * config.train_stride for j in range(config.train_windows)]
    if min(indices) < 1 or (
        min(indices) < config.encoder_days
        and config.observation_contract != GLOBAL_LEFT_PADDING_CONTRACT
    ):
        raise ValueError("requested rolling windows exceed available encoder/target history")
    return tuple(data.date_at(i - 1).isoformat() for i in indices)


def informative_target_diagnostic(
    data: PanelArrays, origins: tuple[str, ...]
) -> tuple[dict[str, object], np.ndarray]:
    """Describe masked target windows and identify rows usable for likelihood training."""
    rows = np.stack(
        [
            data.observed[:, data.position(day) : data.position(day) + data.horizon].any(1)
            for day in origins
        ]
    )
    inactive = ~rows.any(0)
    missing_by_origin = {day: int((~rows[index]).sum()) for index, day in enumerate(origins)}
    windows = [
        {
            "origin": day,
            "target_start": day,
            "target_end": (
                dt.date.fromisoformat(day) + dt.timedelta(days=data.horizon - 1)
            ).isoformat(),
            "series_without_observed_target": missing_by_origin[day],
        }
        for day in origins
    ]
    diagnostic = {
        "schema": "autoforecast.informative-target-diagnostic.v1",
        "policy": "exclude_uninformative_series_from_likelihood_keep_for_inference_v1",
        "series_count": len(data.series_ids),
        "trainable_series_count": int((~inactive).sum()),
        "excluded_series_count": int(inactive.sum()),
        "excluded_series": [str(data.series_ids[index]) for index in np.flatnonzero(inactive)],
        "windows": windows,
    }
    return diagnostic, ~inactive


def replay_sampler_journal(
    rows: list[dict], *, phase: str, objective_masks: dict[str, np.ndarray],
    blocks: list[np.ndarray], batch_size: int, seed: int, epochs: int | None,
    hierarchy_every: int, hierarchy_only_enabled: bool,
) -> dict[str, object]:
    """Replay the trainer's independent NumPy sampler against committed update rows."""
    origins = tuple(objective_masks)
    if (
        phase not in {"selection", "refit"} or not origins or not blocks
        or type(batch_size) is not int or batch_size < 1
        or type(seed) is not int or seed < 0
        or type(hierarchy_every) is not int or hierarchy_every < 1
    ):
        raise ValueError("journal sampler controls are invalid")
    masks = [np.asarray(objective_masks[origin]) for origin in origins]
    n_series = masks[0].shape[0]
    if any(mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[0] != n_series
           for mask in masks):
        raise ValueError("journal sampler objective masks are invalid")
    informative = np.stack([mask.any(1) for mask in masks])
    trainable = informative.any(0)
    members = np.concatenate(blocks)
    if (
        any(block.dtype.kind not in "iu" or block.ndim != 1 or not len(block)
            for block in blocks)
        or not np.array_equal(np.sort(members), np.flatnonzero(trainable))
        or max(map(len, blocks)) > batch_size
    ):
        raise ValueError("journal sampler blocks differ from the trainable population")
    groups_per_batch = min(len(blocks), max(1, batch_size // max(map(len, blocks))))
    steps_per_epoch = math.ceil(len(blocks) / groups_per_batch)
    rng = np.random.default_rng(seed)
    block_order = rng.permutation(len(blocks))
    block_position = 0
    seen = np.zeros(n_series, dtype=bool)
    visits = np.zeros(n_series, dtype=np.int64)
    proposals = np.zeros(n_series, dtype=np.int64)
    epoch_visits = np.zeros(n_series, dtype=np.int64)
    epoch_proposals = np.zeros(n_series, dtype=np.int64)
    epoch_coverage: list[dict[str, int]] = []
    for update, row in enumerate(rows, 1):
        if type(row) is not dict or row.get("phase") != phase or row.get("update") != update:
            raise ValueError("journal sampler update order is invalid")
        if epochs is None and hierarchy_only_enabled and update % hierarchy_every == 0:
            day, sampled, label = origins[0], [], "hierarchy_only"
            observed = np.empty(0, dtype=bool)
        else:
            if block_position == len(blocks):
                block_order, block_position = rng.permutation(len(blocks)), 0
            chosen = block_order[block_position:block_position + groups_per_batch]
            block_position += len(chosen)
            indices = np.concatenate([blocks[index] for index in chosen])
            eligible = np.flatnonzero(informative[:, indices].any(1))
            missing = ~seen[indices]
            if missing.any():
                coverage = informative[:, indices[missing]].sum(1)
                choices = np.flatnonzero(coverage == coverage.max())
                label = "coverage_curriculum"
            else:
                choices = eligible
                label = "uniform_usable_origins"
            if not len(choices):
                raise ValueError("journal sampler has no informative origin")
            day = origins[int(rng.choice(choices))]
            sampled = indices.tolist()
            observed = objective_masks[day][indices].any(1)
            if not observed.any():
                raise ValueError("journal sampler chose a zero-information batch")
            seen[indices[observed]] = True
            visits[indices[observed]] += 1
            proposals[indices] += 1
            epoch_visits[indices[observed]] += 1
            epoch_proposals[indices] += 1
        if (
            row.get("origin") != day
            or row.get("sampled_series_indices") != sampled
            or row.get("origin_sampling") != label
            or row.get("likelihood_series_in_batch") != len(sampled)
            or row.get("likelihood_informative_series_in_batch") != int(observed.sum())
        ):
            raise ValueError("journal sampler row differs from the governed producer")
        if epochs is not None and update % steps_per_epoch == 0:
            summary = {
                "epoch": update // steps_per_epoch,
                "proposed_series": int((epoch_proposals[trainable] > 0).sum()),
                "minimum_row_proposals": int(epoch_proposals[trainable].min()),
                "maximum_row_proposals": int(epoch_proposals[trainable].max()),
                "informative_series": int((epoch_visits[trainable] > 0).sum()),
                "minimum_informative_visits": int(epoch_visits[trainable].min()),
                "maximum_informative_visits": int(epoch_visits[trainable].max()),
            }
            if row.get("epoch_coverage") != summary:
                raise ValueError("journal sampler epoch coverage differs")
            epoch_coverage.append(summary)
            epoch_visits.fill(0)
            epoch_proposals.fill(0)
        elif "epoch_coverage" in row:
            raise ValueError("journal sampler has an off-cadence epoch summary")
    return {
        "steps_per_epoch": steps_per_epoch,
        "seen": seen,
        "visits": visits,
        "proposals": proposals,
        "epoch_visits": epoch_visits,
        "epoch_proposals": epoch_proposals,
        "epoch_coverage": epoch_coverage,
        "block_order": block_order,
        "block_position": block_position,
        "rng_state": rng.bit_generator.state,
        "phase_summary": {
            "epochs": epoch_coverage,
            "series_seen": int(seen[trainable].sum()),
            "informative_series_visits": int(visits.sum()),
            "minimum_visits_per_series": int(visits[trainable].min()),
            "maximum_visits_per_series": int(visits[trainable].max()),
            "minimum_row_proposals": int(proposals[trainable].min()),
            "maximum_row_proposals": int(proposals[trainable].max()),
        },
    }


def _refit_sampler_can_cover(
    objective_masks: dict[str, np.ndarray], blocks: list[np.ndarray], *,
    batch_size: int, seed: int, budget: int, epochs: int | None,
    hierarchy_every: int, hierarchy_only_enabled: bool,
) -> bool:
    """Replay the seeded origin choices before a sealed refit spends an update."""
    informative = np.stack([mask.any(1) for mask in objective_masks.values()])
    trainable = informative.any(0)
    groups_per_batch = min(len(blocks), max(1, batch_size // max(map(len, blocks))))
    if budget < math.ceil(len(blocks) / groups_per_batch):
        return False
    if epochs is not None and groups_per_batch == 1 and all(len(block) == 1 for block in blocks):
        return True
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    position = 0
    seen = np.zeros(len(trainable), dtype=bool)
    for update in range(1, budget + 1):
        if epochs is None and hierarchy_only_enabled and update % hierarchy_every == 0:
            continue
        if position == len(blocks):
            order, position = rng.permutation(len(blocks)), 0
        chosen = order[position:position + groups_per_batch]
        position += len(chosen)
        rows = np.concatenate([blocks[index] for index in chosen])
        missing = ~seen[rows]
        if missing.any():
            coverage = informative[:, rows[missing]].sum(1)
            choices = np.flatnonzero(coverage == coverage.max())
        else:
            choices = np.flatnonzero(informative[:, rows].any(1))
        if not len(choices):
            return False
        origin_index = int(rng.choice(choices))
        seen[rows] |= informative[origin_index, rows]
        if seen[trainable].all():
            return True
    return False


def verified_validation_summary(
    row: dict[str, object], origins: tuple[str, ...], metric: str,
) -> float:
    """Recompute a journal validation summary from its per-origin measurements."""
    details = row.get("validation")
    summary = row.get("validation_summary")
    if (
        type(details) is not dict or set(details) != set(origins)
        or type(summary) is not dict or not origins
        or summary.get("metric_name") != metric
        or summary.get("update") != row.get("update")
        or summary.get("epoch") != row.get("epoch")
    ):
        raise ValueError("journal validation details or summary are invalid")
    scores = []
    nll = []
    for origin in origins:
        detail = details[origin]
        value = detail.get(metric) if type(detail) is dict else None
        likelihood = detail.get("validation_nll") if type(detail) is dict else None
        if (
            type(value) not in (float, int) or not math.isfinite(value) or value < 0
            or (likelihood is not None and (
                type(likelihood) not in (float, int) or not math.isfinite(likelihood)
            ))
        ):
            raise ValueError("journal validation measurement is invalid")
        scores.append(float(value))
        nll.append(float(likelihood) if likelihood is not None else None)
    if any(value is None for value in nll) and not all(value is None for value in nll):
        raise ValueError("journal validation NLL evidence is inconsistent")
    value = float(np.mean(scores))
    mean_nll = (
        float(np.mean([item for item in nll if item is not None]))
        if nll[0] is not None else None
    )
    if summary.get("metric") != value or summary.get("nll") != mean_nll:
        raise ValueError("journal validation summary differs from measurements")
    return value


def staged_hierarchy_backward(
    model: nn.Module,
    chunks: tuple[np.ndarray, ...],
    forward: Callable[[np.ndarray], Tensor],
    metric: HierarchyMetric,
    target: Tensor,
    *,
    weight: float,
    epsilon: float,
    progress: Callable[[str], None] | None = None,
    backward_scale: Callable[[Tensor], Tensor] | None = None,
) -> float:
    """Exact first-order full-grid gradient with bounded activations, NOT partial totals.

    Both passes use eval mode; the caller steps only after this function returns. SFT's mean
    is deterministic. ``test_staged_gradient_matches_direct`` checks every parameter.
    """
    rows = np.concatenate(chunks) if chunks else np.array([], dtype=int)
    if (
        rows.dtype.kind not in "iu"
        or not np.array_equal(np.sort(rows), np.arange(metric.n_bottom))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("staged hierarchy requires each row exactly once and positive weight")
    flags = [(module, module.training) for module in model.modules()]
    fastpath = torch.backends.mha.get_fastpath_enabled()
    # Torch's no-grad fused attention and grad-enabled attention can differ by ULPs.
    # Force the same maintained implementation in both passes, not a tolerance escape hatch.
    torch.backends.mha.set_fastpath_enabled(False)
    model.eval()
    try:
        with torch.no_grad():
            parts = []
            for index, chunk in enumerate(chunks, 1):
                if progress:
                    progress(f"hierarchy forward {index}/{len(chunks)}")
                parts.append(forward(chunk).detach().cpu())
        packed = torch.cat(parts)
        order = torch.as_tensor(np.argsort(rows), dtype=torch.long)
        leaf = packed[order]
        if backward_scale is not None and leaf.dtype in (torch.float16, torch.bfloat16):
            leaf = leaf.float()
        leaf = leaf.detach().requires_grad_(True)
        loss = weight * metric.loss(leaf, target.to(leaf), epsilon=epsilon)
        (derivative,) = torch.autograd.grad(loss, leaf)
        for index, (chunk, expected) in enumerate(zip(chunks, parts, strict=True), 1):
            if progress:
                progress(f"hierarchy backward {index}/{len(chunks)}")
            prediction = forward(chunk)
            if not torch.equal(prediction.detach().cpu(), expected):
                raise ValueError("hierarchy passes differ: stochastic forward or mutated weights")
            dtype = (
                torch.float32
                if backward_scale is not None and prediction.dtype == torch.float16
                else prediction.dtype
            )
            piece = (
                prediction.to(dtype) * derivative[chunk].to(device=prediction.device, dtype=dtype)
            ).sum()
            (backward_scale(piece) if backward_scale is not None else piece).backward()
        return float(loss.detach())
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        for module, flag in flags:
            module.training = flag


def _weight_schema(values):
    """Shapes/types only, allowing a selection model to differ from its fresh-refit layout."""
    return {name: (value.shape, value.dtype, value.layout) for name, value in values.items()}


def _inputs_to_device(inputs: object, device: str) -> object:
    """Move legacy tensor trees or a typed model batch without knowing the architecture."""
    move = getattr(inputs, "to_device", None)
    if callable(move):
        return move(device)
    from torch.utils._pytree import tree_map

    return tree_map(lambda value: value.to(device) if isinstance(value, Tensor) else value, inputs)


@dataclass
class RollingTrainer:
    """A narrow internal seam, not an extension/plugin authority boundary."""

    data: PanelArrays
    config: RollingConfig
    levels: tuple[tuple[str, ...], ...]
    groups: np.ndarray
    make_model: Callable[[], nn.Module]
    batch: Callable[[str, np.ndarray, str, bool], tuple[object, Tensor | None, Tensor | None]]
    training_loss: Callable[[nn.Module, object, Tensor, Tensor, float, float], Tensor]
    mean: Callable[[nn.Module, object], Tensor]
    optimizer_groups: Callable[[nn.Module], list[dict]] | None = None
    begin_phase: Callable[[tuple[str, ...], str], dict] | None = None
    prediction_context: Callable | None = None
    diagnostics: Callable | None = None
    validation_values: Callable | None = None
    feature_training_state: Callable | None = None
    restore_feature_training_state: Callable | None = None
    objective_mask: Callable[[str], np.ndarray] | None = None
    training_contract: RollingTrainingContract = field(default_factory=RollingTrainingContract)
    performance_profiler: TorchPerformanceProfiler | None = None
    precision: str = "32-true"
    _validation_targets: dict = field(default_factory=dict, init=False, repr=False)

    def _profile_region(self, phase: str, name: str):
        if self.performance_profiler is None:
            return nullcontext()
        return self.performance_profiler.record(phase, name)

    def _objective_mask(self, origin: str) -> np.ndarray:
        """Exact likelihood support, which may be stricter than observed daily labels."""
        stop = self.data.position(origin)
        observed = self.data.observed[:, stop : stop + self.data.horizon]
        if self.objective_mask is None:
            return observed
        mask = np.asarray(self.objective_mask(origin))
        if mask.shape != observed.shape or mask.dtype != np.bool_ or np.any(mask & ~observed):
            raise ValueError("objective mask must be a boolean subset of routed observed labels")
        return mask

    def _improved(self, value: float, best: float) -> bool:
        """Apply the configured absolute/relative validation threshold exactly once."""
        if not math.isfinite(best):
            return True
        required = max(
            self.config.early_stopping_absolute_min_delta,
            abs(best) * self.config.early_stopping_relative_min_delta,
        )
        return value < best - required

    def _check_resume_state(
        self, saved, budget, blocks, model, optimizer, metric_name, *, best_schema=None,
        phase_trainable_counts=None, phase_steps_per_epoch=None,
    ):
        """Refuse internally inconsistent counters/arrays/LRs before restoring native state."""
        c, n = self.config, len(self.data.series_ids)
        trainable = np.zeros(n, dtype=bool)
        trainable[np.concatenate(blocks)] = True
        n_trainable = int(trainable.sum())
        phase_counts = phase_trainable_counts or {
            "selection": n_trainable,
            "refit": n_trainable,
        }
        phase_steps = phase_steps_per_epoch or {
            "selection": saved["steps_per_epoch"],
            "refit": saved["steps_per_epoch"],
        }
        if phase_counts.get(saved.get("phase")) != n_trainable:
            raise ValueError("checkpoint phase trainable population changed")
        update = saved.get("update")
        expected_steps = update
        if (
            c.epochs is not None
            and c.hierarchy_loss_weight
            and self.levels
            and self.precision == "32-true"
            and saved.get("optimizer_step_contract") != "one_step_per_update_v1"
        ):
            raise ValueError("legacy FP32 hierarchy checkpoint has incompatible optimizer steps")
        if (
            type(update) is not int
            or not 0 <= update <= budget
            or saved.get("budget") != budget
            or saved.get("steps_per_epoch") != phase_steps.get(saved.get("phase"))
            or saved.get("optimizer_steps") != expected_steps
            or type(saved.get("phase_done")) is not bool
            or type(saved.get("stale")) is not int
            or not 0 <= saved["stale"] <= len(saved["evaluations"])
            or type(saved.get("selected_updates")) is not int
            or not 0
            <= saved["selected_updates"]
            <= (
                budget
                if self.training_contract.mode == "fixed_refit" and saved.get("phase") == "refit"
                else c.epochs * phase_steps["selection"] if c.epochs else c.max_updates
            )
            or type(saved.get("block_position")) is not int
            or not 0 <= saved["block_position"] <= len(blocks)
            or (
                saved["best_score"] is not None
                and (not math.isfinite(saved["best_score"]) or saved["best_score"] < 0)
            )
            or (saved["best_score"] is None) != (saved["best_state"] is None)
        ):
            raise ValueError("invalid rolling checkpoint update/budget/selection counters")
        if self.precision == "16-mixed":
            amp = saved.get("amp_scaler")
            if (
                type(saved.get("amp_overflow_retries")) is not int
                or saved["amp_overflow_retries"] < 0
                or type(amp) is not dict
                or set(amp) != {
                    "scale", "growth_factor", "backoff_factor", "growth_interval",
                    "_growth_tracker",
                }
                or any(type(amp[key]) not in {int, float}
                       or not math.isfinite(amp[key])
                       for key in ("scale", "growth_factor", "backoff_factor"))
                or amp["scale"] <= 0
                or amp["growth_factor"] <= 1
                or not 0 < amp["backoff_factor"] < 1
                or type(amp["growth_interval"]) is not int
                or amp["growth_interval"] < 1
                or type(amp["_growth_tracker"]) is not int
                or not 0 <= amp["_growth_tracker"] < amp["growth_interval"]
            ):
                raise ValueError("invalid rolling GradScaler checkpoint state")
        order = saved.get("block_order")
        if (
            not isinstance(order, Tensor)
            or order.dtype != torch.int64
            or order.shape != (len(blocks),)
            or not torch.equal(order.sort().values, torch.arange(len(blocks)))
        ):
            raise ValueError("invalid checkpoint block permutation")
        groups_per_batch = min(len(blocks), max(1, c.batch_size // max(map(len, blocks))))
        likelihood_updates = update - (
            update // c.hierarchy_every
            if c.epochs is None and c.hierarchy_loss_weight and self.levels
            else 0
        )
        completed_passes, within_pass = (
            divmod(likelihood_updates - 1, saved["steps_per_epoch"])
            if likelihood_updates
            else (0, -1)
        )
        position = min((within_pass + 1) * groups_per_batch, len(blocks))
        if saved["block_position"] != position:
            raise ValueError("checkpoint sampler position contradicts completed likelihood updates")
        for name in (
            "seen_series",
            "likelihood_visits",
            "proposal_visits",
            "epoch_visits",
            "epoch_proposals",
        ):
            value = saved.get(name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (n,)
                or value.dtype != (torch.bool if name == "seen_series" else torch.int64)
                or (value < 0).any()
                or (value > update).any()
            ):
                raise ValueError("invalid checkpoint coverage arrays")
        if (
            not torch.equal(saved["seen_series"], saved["likelihood_visits"] > 0)
            or (saved["likelihood_visits"] > saved["proposal_visits"]).any()
            or (saved["epoch_visits"] > saved["epoch_proposals"]).any()
            or (saved["epoch_proposals"] > saved["proposal_visits"]).any()
            or any(not math.isfinite(saved[name]) for name in ("loss_sum", "epoch_loss_sum"))
            or saved["epoch_loss_count"]
            != (update % saved["steps_per_epoch"] if c.epochs else update)
        ):
            raise ValueError("inconsistent checkpoint coverage/loss accumulation")
        proposals = torch.zeros(n, dtype=torch.int64)
        proposals[torch.from_numpy(trainable)] = completed_passes
        if position:
            proposed_rows = np.concatenate([blocks[index] for index in order[:position]])
            proposals[torch.tensor(proposed_rows)] += 1
        epoch_proposals = (
            torch.zeros_like(proposals)
            if c.epochs and update % saved["steps_per_epoch"] == 0
            else proposals - completed_passes * torch.from_numpy(trainable)
            if c.epochs
            else proposals
        )
        epochs = saved["epoch_coverage"]
        completed_epochs = update // saved["steps_per_epoch"] if c.epochs else 0
        epoch_keys = {
            "epoch",
            "proposed_series",
            "minimum_row_proposals",
            "maximum_row_proposals",
            "informative_series",
            "minimum_informative_visits",
            "maximum_informative_visits",
        }

        def valid_epochs(rows, count, trainable_count):
            return (
                isinstance(rows, list)
                and len(rows) == count
                and all(
                    isinstance(row, dict)
                    and set(row) == epoch_keys
                    and all(type(value) is int for value in row.values())
                    and row["epoch"] == index
                    and row["proposed_series"] == trainable_count
                    and row["minimum_row_proposals"] == row["maximum_row_proposals"] == 1
                    and 0 <= row["informative_series"] <= trainable_count
                    and row["minimum_informative_visits"]
                    == int(row["informative_series"] == trainable_count)
                    and row["maximum_informative_visits"] == int(row["informative_series"] > 0)
                    for index, row in enumerate(rows, 1)
                )
            )

        if (
            not torch.equal(saved["proposal_visits"], proposals)
            or not torch.equal(saved["epoch_proposals"], epoch_proposals)
            or (saved["epoch_visits"] > saved["likelihood_visits"]).any()
            or not valid_epochs(epochs, completed_epochs, n_trainable)
            or c.epochs is not None
            and int(saved["likelihood_visits"].sum())
            != sum(row["informative_series"] for row in epochs) + int(saved["epoch_visits"].sum())
        ):
            raise ValueError("checkpoint sampler/epoch coverage contradicts completed passes")
        phase_coverage = saved["phase_coverage"]
        completed_phases = (
            {"selection"}
            if saved["phase"] == "refit" and self.training_contract.mode == "rolling_backtest"
            else set()
        )
        if saved["phase_done"]:
            completed_phases.add(saved["phase"])
        coverage_keys = {
            "epochs",
            "series_seen",
            "informative_series_visits",
            "minimum_visits_per_series",
            "maximum_visits_per_series",
            "minimum_row_proposals",
            "maximum_row_proposals",
        }
        prior_selection_epochs = (
            (
                saved["evaluations"][-1].get("update", -1) // phase_steps["selection"]
                if c.epochs is not None
                else 0
            )
            if saved["phase"] == "refit"
            and isinstance(saved.get("evaluations"), list)
            and saved["evaluations"]
            and isinstance(saved["evaluations"][-1], dict)
            else -1
        )
        if (
            not isinstance(phase_coverage, dict)
            or set(phase_coverage) != completed_phases
            or any(
                not isinstance(row, dict)
                or set(row) != coverage_keys
                or not valid_epochs(
                    row["epochs"],
                    completed_epochs if phase_name == saved["phase"] else prior_selection_epochs,
                    phase_counts[phase_name],
                )
                or any(
                    type(value) is not int or value < 0
                    for key, value in row.items()
                    if key != "epochs"
                )
                or row["series_seen"] != phase_counts[phase_name]
                or not 1 <= row["minimum_visits_per_series"] <= row["maximum_visits_per_series"]
                or not 1 <= row["minimum_row_proposals"] <= row["maximum_row_proposals"]
                or not phase_counts[phase_name] * row["minimum_visits_per_series"]
                <= row["informative_series_visits"]
                <= phase_counts[phase_name] * row["maximum_visits_per_series"]
                for phase_name, row in phase_coverage.items()
            )
        ):
            raise ValueError("invalid checkpoint completed-phase coverage")
        denominator = (
            max(c.epochs * saved["steps_per_epoch"], budget)
            if c.epochs is not None and saved["phase"] == "refit"
            else c.epochs * saved["steps_per_epoch"]
            if c.epochs is not None else c.max_updates
        )
        lr = (
            c.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * (update - 1) / denominator)) / 2)
            if update
            else c.learning_rate
        )
        expected_schedule = dict(
            kind="cosine_floor_v1",
            completed_updates=update,
            denominator=denominator,
            base_lr=c.learning_rate,
            group_lrs=[lr] * len(optimizer.param_groups),
        )
        if saved.get("scheduler") != expected_schedule:
            raise ValueError("checkpoint scheduler does not match the completed logical update")
        evaluation_updates, evaluation_scores = [], []
        for row in saved["evaluations"]:
            if (
                set(row) != {"update", f"mean_{metric_name}"}
                or type(row["update"]) is not int
                or row["update"] < 1
                or type(row[f"mean_{metric_name}"]) is not float
                or not math.isfinite(row[f"mean_{metric_name}"])
                or row[f"mean_{metric_name}"] < 0
            ):
                raise ValueError("invalid checkpoint selection evaluation history")
            evaluation_updates.append(row["update"])
            evaluation_scores.append(row[f"mean_{metric_name}"])
        latest = saved["latest_validation"]
        model_schema = _weight_schema(model.state_dict())
        for label, values in (("model", saved["model"]), ("best", saved["best_state"])):
            if label == "best" and values is None:
                continue
            expected_schema = (
                best_schema if label == "best" and best_schema is not None else model_schema
            )
            if (
                not isinstance(values, dict)
                or values.keys() != expected_schema.keys()
                or any(
                    not isinstance(value, Tensor)
                    or (value.shape, value.dtype, value.layout) != expected_schema[name]
                    or not torch.isfinite(value).all()
                    for name, value in values.items()
                )
            ):
                raise ValueError(f"invalid checkpoint {label} weight schema/shape/dtype/value")
        if evaluation_scores:
            expected_best = math.inf
            expected_selected = 0
            expected_selection_stale = 0
            for evaluation_update, evaluation_score in zip(
                evaluation_updates, evaluation_scores, strict=True
            ):
                if self._improved(evaluation_score, expected_best):
                    expected_best = evaluation_score
                    expected_selected = evaluation_update
                    expected_selection_stale = 0
                else:
                    expected_selection_stale += 1
            expected_stale = expected_selection_stale if saved["phase"] == "selection" else 0
            last_evaluation = evaluation_updates[-1]
            selection_ceiling = (
                update
                if saved["phase"] == "selection"
                else (c.epochs * phase_steps["selection"] if c.epochs else c.max_updates)
            )
            if (
                evaluation_updates != sorted(set(evaluation_updates))
                or last_evaluation > selection_ceiling
                or saved["selected_updates"] != expected_selected
                or saved["best_score"] != expected_best
                or saved["stale"] != expected_stale
                or not isinstance(latest, dict)
                or latest.get("update") != last_evaluation
                or latest.get("metric") != evaluation_scores[-1]
                or latest.get("metric_name") != metric_name
                or latest.get("epoch")
                != (math.ceil(last_evaluation / phase_steps["selection"]) if c.epochs else None)
                or not isinstance(saved["best_state"], dict)
                or not saved["best_state"]
                or any(
                    not isinstance(v, Tensor) or not torch.isfinite(v).all()
                    for v in saved["best_state"].values()
                )
            ):
                raise ValueError("inconsistent checkpoint best/selected/stale/evaluation state")
            if (
                set(latest) != {"update", "epoch", "metric", "metric_name", "nll"}
                or self.validation_values is None
                and latest["nll"] is not None
                or self.validation_values is not None
                and (
                    type(latest["nll"]) not in (int, float)
                    or not math.isfinite(latest["nll"])
                    or latest["nll"] < 0
                )
            ):
                raise ValueError("invalid checkpoint retained validation record/NLL")
        else:
            fixed_refit = (
                self.training_contract.mode == "fixed_refit"
                and saved["phase"] == "refit"
                and saved["selected_updates"] == budget
            )
            empty_selection = saved["phase"] == "selection" and saved["selected_updates"] == 0
            if (
                not (fixed_refit or empty_selection)
                or saved["best_score"] is not None
                or saved["best_state"] is not None
                or saved["stale"] != 0
                or latest is not None
            ):
                raise ValueError("checkpoint selection state exists without an evaluation")
        terminal = update == budget or (
            saved["phase"] == "selection" and saved["stale"] >= c.patience
        )
        if saved["phase_done"] and (
            not terminal
            or not saved["seen_series"][torch.from_numpy(trainable)].all()
            or saved["phase"] == "selection"
            and not evaluation_scores
        ):
            raise ValueError("checkpoint marks an unfinished or uncovered phase complete")
        native_groups = optimizer.state_dict()["param_groups"]
        loaded_optimizer = saved["optimizer"]
        if (
            not isinstance(loaded_optimizer, dict)
            or set(loaded_optimizer) != {"state", "param_groups"}
            or not isinstance(loaded_optimizer["state"], dict)
            or any(type(index) is not int for index in loaded_optimizer["state"])
            or not isinstance(loaded_optimizer["param_groups"], list)
            or any(not isinstance(group, dict) for group in loaded_optimizer["param_groups"])
            or [dict(group, lr=None) for group in loaded_optimizer["param_groups"]]
            != [dict(group, lr=None) for group in native_groups]
        ):
            raise ValueError("checkpoint AdamW parameter mapping or configured controls differ")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        parameter_updates = saved.get("parameter_updates")
        if (
            not isinstance(parameter_updates, list)
            or len(parameter_updates) != len(parameters)
            or any(
                type(count) is not int or not 0 <= count <= expected_steps
                for count in parameter_updates
            )
            or set(loaded_optimizer["state"])
            != {index for index, count in enumerate(parameter_updates) if count}
        ):
            raise ValueError("checkpoint AdamW moments differ from per-parameter update coverage")
        from torch.optim.optimizer import _get_scalar_dtype

        step_dtypes = [
            _get_scalar_dtype(is_fused=group["fused"])
            for group in optimizer.param_groups
            for _ in group["params"]
        ]

        def check_moments(states):
            for index, parameter in enumerate(parameters):
                if not parameter_updates[index]:
                    continue
                state = states[index]
                if (
                    not isinstance(state, dict)
                    or set(state) != {"step", "exp_avg", "exp_avg_sq"}
                    or any(not isinstance(value, Tensor) for value in state.values())
                    or state["exp_avg"].shape != parameter.shape
                    or state["exp_avg_sq"].shape != parameter.shape
                    or state["exp_avg"].dtype != parameter.dtype
                    or state["exp_avg_sq"].dtype != parameter.dtype
                    or state["exp_avg"].layout != parameter.layout
                    or state["exp_avg_sq"].layout != parameter.layout
                    or state["step"].ndim != 0
                    or state["step"].dtype != step_dtypes[index]
                    or float(state["step"]) != parameter_updates[index]
                    or not torch.isfinite(state["exp_avg"]).all()
                    or not torch.isfinite(state["exp_avg_sq"]).all()
                    or (state["exp_avg_sq"] < 0).any()
                ):
                    raise ValueError("invalid AdamW checkpoint moments/step counters")

        # Native loading casts moments; malformed serialized dtypes must not be normalized away.
        check_moments(loaded_optimizer["state"])
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if [group["lr"] for group in optimizer.param_groups] != expected_schedule["group_lrs"]:
            raise ValueError("checkpoint optimizer and scheduler learning rates differ")
        check_moments(
            {
                index: optimizer.state[p]
                for index, p in enumerate(parameters)
                if parameter_updates[index]
            }
        )
        if (update > 0 and not optimizer.state) or any(
            not torch.isfinite(v).all() for v in model.state_dict().values()
        ):
            raise ValueError("checkpoint is missing optimizer moments or has nonfinite model state")

    def validation_due(self, update: int, budget: int, steps_per_epoch: int) -> bool:
        """Epoch cadence for fixed epochs; explicit update cadence otherwise; always final."""
        interval = (
            steps_per_epoch * self.config.validation_every_epochs
            if self.config.epochs is not None
            else self.config.evaluate_every
        )
        return update == budget or update % interval == 0

    def validate(self, model, origin, metric, *, progress=None) -> dict:
        """One evaluation forward/batch, no expert-label sampling or full-grid host export.

        The feature callback returns the mean and the already-built distribution's NLL
        callable. Input assembly is deliberately CPU-side so transfer time is measurable.
        """
        started = time.perf_counter()
        device = self.config.device

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        timings = dict(
            feature_seconds=0.0,
            forward_seconds=0.0,
            nll_seconds=0.0,
            transfer_seconds=0.0,
            metric_seconds=0.0,
            prediction_context_seconds=0.0,
            prediction_assembly_seconds=0.0,
            numpy_metric_seconds=None,
        )
        stop = self.data.position(origin)
        sync()
        transfer_started = time.perf_counter()
        key = (origin, device)
        holdout = self.training_contract.fixed_validation
        if key not in self._validation_targets:
            if holdout is not None:
                if holdout.origin != origin:
                    raise ValueError("fixed validation origin differs from the requested window")
                target = holdout.target
                observed = holdout.observed
                objective = (
                    holdout.observed if holdout.objective_mask is None else holdout.objective_mask
                )
            else:
                target = self.data.target[:, stop : stop + self.data.horizon]
                observed = self.data.observed[:, stop : stop + self.data.horizon]
                objective = self._objective_mask(origin)
            self._validation_targets[key] = (
                torch.tensor(target, device=device),
                torch.tensor(observed, device=device),
                torch.tensor(objective, device=device),
            )
        y, mask, nll_mask = self._validation_targets[key]
        sync()
        timings["transfer_seconds"] += time.perf_counter() - transfer_started
        if not mask.any():
            raise ValueError("validation window has no observed targets")
        if self.validation_values is not None and not nll_mask.any():
            raise ValueError("validation window has no labels eligible for the registered NLL")
        flags = [(module, module.training) for module in model.modules()]
        model.eval()
        pieces, nll_sum = [], torch.zeros((), dtype=torch.float64, device=device)
        try:
            context_started = time.perf_counter()
            with (
                torch.no_grad(),
                (
                    self.prediction_context(model, origin)
                    if self.prediction_context
                    else nullcontext()
                ) as context_timings,
            ):
                sync()
                timings["prediction_context_seconds"] = time.perf_counter() - context_started
                if context_timings:
                    for name in ("feature_seconds", "transfer_seconds", "forward_seconds"):
                        timings[name] += context_timings[name]
                if progress:
                    progress.validation_status("running inference")
                for index, rows in enumerate(self.chunks(), 1):
                    if self.performance_profiler is not None:
                        self.performance_profiler.begin_step("validation")
                    assembly_started = time.perf_counter()
                    with self._profile_region("validation", "batch_assembly"):
                        inputs = self.batch(origin, rows, "cpu", False)[0]
                    timings["feature_seconds"] += time.perf_counter() - assembly_started
                    transfer_started = time.perf_counter()
                    with self._profile_region("validation", "device_transfer"):
                        inputs = _inputs_to_device(inputs, device)
                        index_tensor = torch.tensor(rows, device=device)
                    sync()
                    timings["transfer_seconds"] += time.perf_counter() - transfer_started
                    forward_started = time.perf_counter()
                    with self._profile_region("validation", "forward"):
                        if self.validation_values:
                            mean, nll = self.validation_values(model, inputs)
                        else:
                            mean, nll = self.mean(model, inputs), None
                        pieces.append(mean.detach().double())
                    sync()
                    timings["forward_seconds"] += time.perf_counter() - forward_started
                    nll_started = time.perf_counter()
                    with self._profile_region("validation", "negative_log_likelihood"):
                        if nll:
                            nll_sum += nll(y[index_tensor], nll_mask[index_tensor]).double().sum()
                    sync()
                    timings["nll_seconds"] += time.perf_counter() - nll_started
                    if self.performance_profiler is not None:
                        self.performance_profiler.end_step("validation")
                    if progress:
                        progress.validation_batch(index, len(self.chunks()))
                assembly_started = time.perf_counter()
                prediction = torch.cat(pieces)
                if (
                    prediction.shape != y.shape
                    or not torch.isfinite(prediction).all()
                    or (prediction < 0).any()
                    or not torch.isfinite(nll_sum)
                ):
                    raise ValueError("nonfinite/negative validation prediction or NLL")
                sync()
                timings["prediction_assembly_seconds"] = time.perf_counter() - assembly_started
                shared_study = holdout is not None and holdout.study is not None
                backend = "numpy" if shared_study else self.config.validation_backend
                if progress:
                    progress.validation_status(
                        f"scoring with {backend} backend"
                    )
                if holdout is not None and holdout.study is not None:
                    metric_started = time.perf_counter()
                    result = holdout.study.score(prediction.cpu().numpy())
                    if result[holdout.study.metric] is None:
                        raise ValueError("study checkpoint selection metric is unavailable")
                    timings["numpy_metric_seconds"] = time.perf_counter() - metric_started
                elif self.config.validation_backend == "numpy":
                    transfer_started = time.perf_counter()
                    prediction_numpy, target_numpy = prediction.cpu().numpy(), y.cpu().numpy()
                    mask_numpy = mask.cpu().numpy() if metric is None else None
                    sync()
                    timings["transfer_seconds"] += time.perf_counter() - transfer_started
                    metric_started = time.perf_counter()
                    if metric is not None:
                        result = metric.score(prediction_numpy, target_numpy)
                    else:
                        with np.errstate(over="ignore", invalid="ignore"):
                            value = float(
                                np.sqrt(
                                    np.mean(
                                        (prediction_numpy[mask_numpy] - target_numpy[mask_numpy])
                                        ** 2
                                    )
                                )
                            )
                        if not np.isfinite(value):
                            raise ValueError("nonfinite NumPy validation metric")
                        result = dict(rmse=value, observed_cells=int(mask_numpy.sum()))
                    timings["numpy_metric_seconds"] = time.perf_counter() - metric_started
                else:
                    metric_started = time.perf_counter()
                    if metric is not None:
                        result = metric.score(prediction, y)
                    else:
                        value = (prediction[mask] - y[mask].double()).square().mean().sqrt()
                        if not torch.isfinite(value):
                            raise ValueError("nonfinite validation metric")
                        result = dict(rmse=float(value), observed_cells=int(mask.sum()))
                sync()
                timings["metric_seconds"] = time.perf_counter() - metric_started
                result["validation_nll"] = (
                    float(nll_sum / nll_mask.sum()) if self.validation_values else None
                )
                result["nll_observed_cells"] = int(nll_mask.sum()) if self.validation_values else 0
                result["backend"] = backend
        finally:
            for module, flag in flags:
                module.training = flag
        # Include context exit (HMFNet aggregate-cache release) and mode restoration in the pause.
        sync()
        timings["total_seconds"] = time.perf_counter() - started
        timings["other_seconds"] = max(
            0.0,
            timings["total_seconds"]
            - sum(
                timings[name]
                for name in (
                    "feature_seconds",
                    "forward_seconds",
                    "nll_seconds",
                    "transfer_seconds",
                    "metric_seconds",
                    "prediction_assembly_seconds",
                )
            ),
        )
        result["timing"] = timings
        return result

    def chunks(self) -> tuple[np.ndarray, ...]:
        """Deterministic inference chunks, independent of factor-block sampling."""
        return tuple(
            np.arange(i, min(i + self.config.batch_size, len(self.data.series_ids)))
            for i in range(0, len(self.data.series_ids), self.config.batch_size)
        )

    def predict(self, model: nn.Module, origin: str) -> np.ndarray:
        """Full ordered grid; validation never samples a subset of series."""
        flags = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            with (
                torch.no_grad(),
                (
                    self.prediction_context(model, origin)
                    if self.prediction_context
                    else nullcontext()
                ),
            ):
                pieces = [
                    self.mean(model, self.batch(origin, r, self.config.device, False)[0])
                    .detach()
                    .cpu()
                    .double()
                    .numpy()
                    for r in self.chunks()
                ]
            result = np.concatenate(pieces)
            if not np.isfinite(result).all() or np.any(result < 0):
                raise ValueError("model emitted nonfinite or negative forecasts")
            return result
        finally:
            for module, flag in flags:
                module.training = flag

    def fit(
        self, log: Callable[[dict], None], *, run=None, progress=None
    ) -> tuple[nn.Module, dict]:
        """Close the UI on success/failure; the caller owns the durable run lease."""
        progress = progress or TrainingProgress(self.config.progress)
        try:
            try:
                result = self._fit(log, run, progress)
            except BaseException as training_error:
                if self.performance_profiler is not None:
                    try:
                        self.performance_profiler.close()
                    except BaseException as profile_error:
                        training_error.add_note(
                            f"Torch profiler cleanup also failed: {profile_error!r}"
                        )
                raise
            if self.performance_profiler is not None:
                self.performance_profiler.close()
            return result
        finally:
            progress.close()

    def _fit(self, log, run, progress) -> tuple[nn.Module, dict]:
        """Execute the governed research, fixed-validation, or fixed-refit contract."""
        c, data, contract = self.config, self.data, self.training_contract
        if self.precision not in ("32-true", "16-mixed"):
            raise ValueError("rolling precision must be 32-true or 16-mixed")
        if (self.feature_training_state is None) != (
            self.restore_feature_training_state is None
        ):
            raise ValueError("feature-state capture and restore must be supplied together")
        configure_training_runtime(c.device, c.torch_threads)
        progress.stage("setup validation constants", force=True)
        setup_started = time.perf_counter()
        if contract.mode == "rolling_backtest":
            validation = tuple(
                (
                    data.cutoff - dt.timedelta(days=data.horizon * offset * data.step_days)
                ).isoformat()
                for offset in (2, 1)
            )
            train = rolling_origins(data, validation[0], c)
            final_train = rolling_origins(data, data.cutoff.isoformat(), c)
        elif contract.mode == "fixed_validation":
            assert contract.fixed_validation is not None
            validation = (contract.fixed_validation.origin,)
            if validation[0] != data.cutoff.isoformat():
                raise ValueError(
                    "fixed validation must begin immediately after the training cutoff"
                )
            if contract.fixed_validation.target.shape != (
                len(data.series_ids),
                data.horizon,
            ):
                raise ValueError("fixed validation shape differs from the training panel")
            train = rolling_origins(data, data.cutoff.isoformat(), c)
            final_train = ()
        else:
            validation = ()
            train = ()
            final_train = rolling_origins(data, data.cutoff.isoformat(), c)
        metric_name = (
            ("wrmsse" if self.levels and data.price.shape[1] else "rmse")
            if c.selection_metric == "auto"
            else c.selection_metric
        )
        study = (None if contract.fixed_validation is None else contract.fixed_validation.study)
        if study is not None:
            metric_name = study.metric
        use_hierarchy_metric = metric_name == "wrmsse" and study is None
        hierarchy_weighting = (
            ("revenue" if data.price.shape[1] else "uniform")
            if c.hierarchy_weighting == "auto"
            else c.hierarchy_weighting
        )
        if (
            (use_hierarchy_metric or c.hierarchy_loss_weight and self.levels)
            and hierarchy_weighting == "revenue"
            and not data.price.shape[1]
        ):
            raise ValueError("revenue hierarchy weighting requires a declared price input")
        if use_hierarchy_metric and hierarchy_weighting == "uniform":
            metric_name = "uniform_hierarchy_rmsse"
        hierarchy_labels_complete = bool(data.observed.all())
        if data.observation_contract == "complete_recorded_sales_prelaunch_zeros_v1":
            # This contract authenticates every numeric target at load time; ``observed`` is
            # then a likelihood/exposure mask, not an assertion that the metric label is absent.
            hierarchy_labels_complete = True
        elif data.observation_contract == GLOBAL_LEFT_PADDING_CONTRACT:
            # Official M5 scales deliberately exclude leading zero history.  The fixed-length
            # encoder contract may therefore mask per-series prelaunch zeros, but never an
            # internal/history value or any label in a validation horizon.
            leading_zero_masks_only = not (
                np.any(data.target[~data.observed] != 0)
                or np.any(data.observed[:, :-1] & ~data.observed[:, 1:])
            )
            validation_labels_complete = all(
                (
                    contract.fixed_validation.observed.all()
                    if contract.fixed_validation is not None
                    else data.observed[
                        :,
                        data.position(day) : data.position(day) + data.horizon,
                    ].all()
                )
                for day in validation
            )
            hierarchy_labels_complete = leading_zero_masks_only and validation_labels_complete
        if (
            use_hierarchy_metric
            and contract.fixed_validation is not None
            and not contract.fixed_validation.observed.all()
        ):
            hierarchy_labels_complete = False
        if (
            use_hierarchy_metric or c.hierarchy_loss_weight and self.levels
        ) and not hierarchy_labels_complete:
            raise ValueError("complete recorded targets are required for hierarchy scales/labels")
        metrics = (
            [
                HierarchyMetric.build(
                    data.metadata,
                    data.target[:, : data.position(day)],
                    data.price[:, : data.position(day)] if data.price.shape[1] else None,
                    self.levels,
                    weighting=hierarchy_weighting,
                )
                for day in validation
            ]
            if use_hierarchy_metric
            else []
        )
        if c.validation_backend == "torch":
            metrics = [TorchHierarchyMetric(metric, c.device) for metric in metrics]
        if c.device == "cuda":
            torch.cuda.synchronize()
        if run:
            run.event(
                "validation_setup",
                seconds=time.perf_counter() - setup_started,
                origins=validation,
                backend=c.validation_backend,
                training_mode=contract.mode,
            )
        if self.groups.shape != (len(data.series_ids),) or self.groups.dtype.kind not in "iu":
            raise ValueError("factor groups must cover exactly the bottom population")
        all_origins = tuple(dict.fromkeys((*train, *final_train)))
        diagnostic, _all_trainable_series = informative_target_diagnostic(data, all_origins)
        selection_rows = (
            np.stack([self._objective_mask(day).any(1) for day in train])
            if train
            else np.zeros((0, len(data.series_ids)), dtype=bool)
        )
        refit_rows = (
            np.stack([self._objective_mask(day).any(1) for day in final_train])
            if final_train
            else np.zeros((0, len(data.series_ids)), dtype=bool)
        )
        trainable_series = selection_rows.any(0) | refit_rows.any(0)
        selection_excluded = ~selection_rows.any(0)
        refit_excluded = ~refit_rows.any(0)
        def phase_step_count(trainable: np.ndarray) -> int:
            sizes = [
                int(((self.groups == group) & trainable).sum())
                for group in np.unique(self.groups)
            ]
            sizes = [size for size in sizes if size]
            if not sizes:
                return 0
            groups_per_batch = min(len(sizes), max(1, c.batch_size // max(sizes)))
            return math.ceil(len(sizes) / groups_per_batch)

        phase_steps_per_epoch = {
            "selection": phase_step_count(selection_rows.any(0)),
            "refit": phase_step_count(refit_rows.any(0)),
        }
        if (
            contract.mode == "fixed_refit"
            and contract.fixed_update_budget is not None
            and contract.fixed_update_budget < phase_steps_per_epoch["refit"]
        ):
            raise ValueError(
                "[autoforecast:non-retryable] selected refit update budget cannot "
                "cover every trainable series"
            )
        diagnostic.update(
            selection_trainable_series_count=int(selection_rows.any(0).sum()),
            refit_trainable_series_count=int(refit_rows.any(0).sum()),
            selection_excluded_series_count=int(selection_excluded.sum()),
            selection_excluded_series=[
                str(data.series_ids[index]) for index in np.flatnonzero(selection_excluded)
            ],
            refit_excluded_series_count=int(refit_excluded.sum()),
            refit_excluded_series=[
                str(data.series_ids[index]) for index in np.flatnonzero(refit_excluded)
            ],
        )
        if not trainable_series.any():
            raise ValueError(
                "[autoforecast:non-retryable] no informative series have targets "
                "in the rolling windows"
            )
        if run:
            run.event("informative_target_preflight", **diagnostic)
            diagnostic_path = run.root / "preflight" / "informative-target-diagnostic.json"
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostic_path.write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if train and diagnostic["selection_excluded_series_count"]:
            progress.stage(
                "data preflight: "
                f"{diagnostic['selection_trainable_series_count']}/{diagnostic['series_count']} "
                "series trainable; excluded masked-only series",
                force=True,
            )
        blocks: list[np.ndarray] = []
        groups_per_batch = 0
        steps_per_epoch = 0
        likelihood_budget: int | None = None
        selected_updates = contract.fixed_update_budget or 0
        best_score, evaluations = math.inf, []
        validation_slots: list[dict[str, object]] = []
        best_state, latest_validation = None, None
        phase_admission = {}
        phase_coverage = {}
        resumed = run.loaded if run else None
        if resumed:
            if (
                resumed.get("schema") != "rolling_trainer_v1"
                or resumed.get("precision", "32-true") != self.precision
                or resumed.get("phase") not in {"selection", "refit"}
                or resumed.get("training_contract") != contract.as_record()
                or resumed.get("validation_origins") != validation
                or resumed.get("selection_origins") != train
                or resumed.get("refit_origins") != final_train
            ):
                raise ValueError("rolling checkpoint schema/origins/block layout mismatch")
            selected_updates = resumed["selected_updates"]
            best_score = resumed["best_score"] if resumed["best_score"] is not None else math.inf
            best_state = resumed["best_state"]
            evaluations = copy.deepcopy(resumed["evaluations"])
            validation_slots = copy.deepcopy(resumed.get("validation_slots", []))
            if resumed.get("steps_per_epoch") != phase_steps_per_epoch[resumed["phase"]]:
                raise ValueError("rolling checkpoint epoch cadence changed")
            interval = (
                phase_steps_per_epoch[resumed["phase"]] * c.validation_every_epochs
                if c.epochs is not None else c.evaluate_every
            )
            first_due = min(interval, resumed["budget"])
            if (
                contract.mode == "fixed_validation"
                and resumed["update"] >= first_due
                and not validation_slots
            ):
                raise ValueError("rolling checkpoint predates validation-slot evidence")
            latest_validation = copy.deepcopy(resumed["latest_validation"])
            phase_admission = copy.deepcopy(resumed["phase_admission"])
            phase_coverage = copy.deepcopy(resumed["phase_coverage"])

        # Synchronous closure reads the current phase's safe frontier; never escapes fit.
        def checkpoint(reason):
            if run is None:
                return
            checkpoint_started = time.perf_counter()
            progress.stage(f"checkpoint {phase} update {update} ({reason})", force=True)
            run.save(
                dict(
                    schema="rolling_trainer_v1",
                    training_contract=contract.as_record(),
                    phase=phase,
                    update=update,
                    budget=budget,
                    phase_done=phase_done,
                    optimizer_steps=optimizer_steps,
                    optimizer_step_contract="one_step_per_update_v1",
                    parameter_updates=list(parameter_updates),
                    steps_per_epoch=steps_per_epoch,
                    validation_origins=validation,
                    selection_origins=train,
                    refit_origins=final_train,
                    model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(),
                    **(
                        {"precision": self.precision, "amp_scaler": scaler.state_dict(),
                         "amp_overflow_retries": amp_overflow_retries}
                        if scaler is not None else {}
                    ),
                    scheduler=dict(
                        kind="cosine_floor_v1",
                        completed_updates=update,
                        denominator=(
                            max(c.epochs * steps_per_epoch, budget)
                            if c.epochs is not None and phase == "refit"
                            else c.epochs * steps_per_epoch
                            if c.epochs is not None else c.max_updates
                        ),
                        base_lr=c.learning_rate,
                        group_lrs=[group["lr"] for group in optimizer.param_groups],
                    ),
                    selected_updates=selected_updates,
                    best_score=best_score if math.isfinite(best_score) else None,
                    best_state=best_state,
                    stale=stale,
                    evaluations=evaluations,
                    validation_slots=validation_slots,
                    latest_validation=latest_validation,
                    phase_admission=phase_admission,
                    phase_coverage=phase_coverage,
                    block_order=torch.tensor(block_order),
                    block_position=block_position,
                    seen_series=torch.tensor(seen_series),
                    likelihood_visits=torch.tensor(likelihood_visits),
                    proposal_visits=torch.tensor(proposal_visits),
                    epoch_visits=torch.tensor(epoch_visits),
                    epoch_proposals=torch.tensor(epoch_proposals),
                    epoch_coverage=epoch_coverage,
                    loss_sum=loss_sum,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_loss_count=epoch_loss_count,
                    rng=capture_training_rng(rng),
                    feature_state=self.feature_training_state()
                    if self.feature_training_state
                    else {},
                ),
                reason=reason,
            )
            run.event(
                "checkpoint_pause",
                phase=phase,
                update=update,
                reason=reason,
                seconds=time.perf_counter() - checkpoint_started,
                timing_scope="snapshot_capture_serialization_publication_and_retention",
            )

        # Like checkpoint(), this synchronous closure reads only the active phase's state.
        def step_optimizer():
            nonlocal optimizer_steps
            before_calls = actual_optimizer_calls["count"]
            if scaler is not None:
                before_scale = scaler.get_scale()
                if not math.isfinite(before_scale) or before_scale <= 0:
                    raise FloatingPointError("rolling AMP has invalid loss scale")
                scaler.step(optimizer)
                scaler.update()
                after_scale = scaler.get_scale()
                if not math.isfinite(after_scale) or after_scale <= 0:
                    raise FloatingPointError("rolling AMP produced invalid loss scale")
                if actual_optimizer_calls["count"] == before_calls:
                    if after_scale >= before_scale:
                        # MPS can skip a raw step without changing the visible scale.
                        # Back off explicitly, then replay this logical update.
                        backoff_scale = before_scale * 0.5
                        if not math.isfinite(backoff_scale) or backoff_scale <= 0:
                            raise FloatingPointError("rolling AMP cannot back off loss scale")
                        scaler.update(new_scale=backoff_scale)
                        if scaler.get_scale() >= before_scale:
                            raise RuntimeError(
                                "rolling GradScaler skipped an update without backoff"
                            )
                    return False
            else:
                optimizer.step()
            if actual_optimizer_calls["count"] != before_calls + 1:
                raise RuntimeError("rolling optimizer did not complete exactly one update")
            optimizer_steps += 1
            for index, parameter in enumerate(optimizer_parameters):
                if parameter.grad is not None:
                    parameter_updates[index] += 1
            return True

        phases = (
            (("selection", train), ("refit", final_train))
            if contract.mode == "rolling_backtest"
            else (("selection", train),)
            if contract.mode == "fixed_validation"
            else (("refit", final_train),)
        )
        best_schema = None
        for phase, origins in phases:
            objective_masks = {day: self._objective_mask(day) for day in origins}
            informative_rows = np.stack([objective_masks[day].any(1) for day in origins])
            phase_trainable = informative_rows.any(0)
            group_ids = np.unique(self.groups)
            blocks = [
                np.flatnonzero((self.groups == group) & phase_trainable) for group in group_ids
            ]
            blocks = [block for block in blocks if len(block)]
            if not blocks:
                raise ValueError(
                    "[autoforecast:non-retryable] no informative series have targets "
                    f"in the {phase} rolling windows"
                )
            groups_per_batch = min(len(blocks), max(1, c.batch_size // max(map(len, blocks))))
            if max(map(len, blocks)) > c.batch_size:
                raise ValueError("batch_size cannot contain one complete factor block")
            steps_per_epoch = math.ceil(len(blocks) / groups_per_batch)
            likelihood_budget = c.epochs * steps_per_epoch if c.epochs is not None else None
            if resumed and resumed["phase"] == "refit" and phase == "selection":
                # Rebuild only selection structure, on CPU, before refit's own seeded setup.
                # Role-bound category vocabularies may legitimately expand between phases.
                admission = self.begin_phase(origins, phase) if self.begin_phase else {}
                if admission != phase_admission.get(phase):
                    raise ValueError("checkpoint selection feature admission/layout changed")
                best_schema = _weight_schema(self.make_model().state_dict())
                continue
            torch.manual_seed(c.seed)
            random.seed(c.seed)
            np.random.seed(c.seed)
            if c.device == "cuda":
                torch.cuda.manual_seed_all(c.seed)
            rng = np.random.default_rng(c.seed)
            admission = self.begin_phase(origins, phase) if self.begin_phase else {}
            restoring = resumed is not None and resumed["phase"] == phase
            if restoring and admission != phase_admission.get(phase):
                raise ValueError("checkpoint phase feature admission/layout changed")
            phase_admission[phase] = admission
            model = self.make_model().to(c.device)
            if phase == "selection" or best_schema is None:
                best_schema = _weight_schema(model.state_dict())
            parameters = (
                self.optimizer_groups(model)
                if self.optimizer_groups
                else [{"params": model.parameters(), "weight_decay": c.weight_decay}]
            )
            optimizer = torch.optim.AdamW(parameters, lr=c.learning_rate)
            actual_optimizer_calls = {"count": 0}
            optimizer.register_step_post_hook(
                lambda _optimizer, _args, _kwargs, calls=actual_optimizer_calls: calls.__setitem__(
                    "count", calls["count"] + 1
                )
            )
            scaler = (
                torch.amp.GradScaler(c.device)
                if self.precision == "16-mixed" else None
            )
            amp_overflow_retries = 0
            optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
            parameter_updates = [0] * len(optimizer_parameters)

            if phase == "selection":
                budget = likelihood_budget if likelihood_budget is not None else c.max_updates
            else:
                # Validation chooses the optimizer-update budget. The fresh refit must reproduce
                # exactly that budget; maximum_epochs is a selection ceiling, not a second full
                # run after early stopping.
                budget = selected_updates
            assert budget is not None
            if phase == "refit" and budget < steps_per_epoch:
                raise ValueError(
                    "[autoforecast:non-retryable] selected refit update budget cannot "
                    "cover every trainable series"
                )
            if phase == "refit" and not _refit_sampler_can_cover(
                objective_masks, blocks, batch_size=c.batch_size, seed=c.seed,
                budget=budget, epochs=c.epochs, hierarchy_every=c.hierarchy_every,
                hierarchy_only_enabled=bool(c.hierarchy_loss_weight and self.levels),
            ):
                raise ValueError(
                    "[autoforecast:non-retryable] selected refit update budget cannot "
                    "cover every trainable series"
                )
            stale = 0
            block_order = rng.permutation(len(blocks))
            block_position = 0
            seen_series = np.zeros(len(data.series_ids), dtype=bool)
            likelihood_visits = np.zeros(len(data.series_ids), dtype=np.int64)
            proposal_visits = np.zeros(len(data.series_ids), dtype=np.int64)
            epoch_visits = np.zeros(len(data.series_ids), dtype=np.int64)
            epoch_proposals = np.zeros(len(data.series_ids), dtype=np.int64)
            epoch_coverage = []
            update, optimizer_steps, loss_sum, epoch_loss_sum, epoch_loss_count = 0, 0, 0.0, 0.0, 0
            phase_done = False
            informative = {day: float(objective_masks[day].sum()) for day in origins}
            if min(informative.values()) <= 0:
                raise ValueError("rolling origin contains no informative training target")
            hierarchy = None
            if c.hierarchy_loss_weight and self.levels:
                stop = data.position(origins[0])
                hierarchy = HierarchyMetric.build(
                    data.metadata,
                    data.target[:, :stop],
                    data.price[:, :stop] if data.price.shape[1] else None,
                    self.levels,
                    weighting=hierarchy_weighting,
                    training_surrogate=True,
                )
            if restoring:
                self._check_resume_state(
                    resumed, budget, blocks, model, optimizer, metric_name,
                    best_schema=best_schema,
                    phase_trainable_counts={
                        "selection": int(selection_rows.any(0).sum()),
                        "refit": int(refit_rows.any(0).sum()),
                    },
                    phase_steps_per_epoch=phase_steps_per_epoch,
                )
                selection_end = (
                    resumed["evaluations"][-1]["update"]
                    if phase == "refit" and resumed["evaluations"]
                    else 0
                )
                expected_last = (
                    (phase, resumed["update"])
                    if resumed["update"]
                    else ("selection", selection_end)
                    if phase == "refit" and contract.mode == "rolling_backtest"
                    else None
                )
                if (
                    run.records != selection_end + resumed["update"]
                    or run.last_record != expected_last
                ):
                    raise ValueError(
                        "checkpoint phase counters differ from completed journal frontier"
                    )
                journal_epochs = {key: [] for key in resumed["phase_coverage"]}
                journal_visits = dict.fromkeys(journal_epochs, 0)
                journal_slots = []
                journal_evaluations = []
                journal_rows = [json.loads(line) for line in run._pending_recovery[1].splitlines()]
                reconstructed_seen = np.zeros(len(data.series_ids), dtype=bool)
                reconstructed_visits = np.zeros(len(data.series_ids), dtype=np.int64)
                reconstructed_proposals = np.zeros(len(data.series_ids), dtype=np.int64)
                reconstructed_epoch_visits = np.zeros(len(data.series_ids), dtype=np.int64)
                reconstructed_epoch_proposals = np.zeros(len(data.series_ids), dtype=np.int64)
                reconstructed_epochs = []
                for row in journal_rows:
                    if "validation_slot" in row:
                        journal_slots.append(row["validation_slot"])
                    if "validation_summary" in row:
                        verified_metric = verified_validation_summary(
                            row, validation, metric_name
                        )
                        journal_evaluations.append({
                            "update": row["update"],
                            f"mean_{metric_name}": verified_metric,
                        })
                    if row["phase"] in journal_epochs:
                        if "epoch_coverage" in row:
                            journal_epochs[row["phase"]].append(row["epoch_coverage"])
                        journal_visits[row["phase"]] += row[
                            "likelihood_informative_series_in_batch"
                        ]
                    if contract.mode == "fixed_validation":
                        sampled = row.get("sampled_series_indices")
                        day = row.get("origin")
                        if (
                            row.get("phase") != "selection"
                            or type(sampled) is not list
                            or any(type(index) is not int or not 0 <= index < len(data.series_ids)
                                   for index in sampled)
                            or len(set(sampled)) != len(sampled)
                            or type(day) is not str or day not in objective_masks
                            or (sampled and not phase_trainable[sampled].all())
                        ):
                            raise ValueError(
                                "checkpoint journal sampled-series evidence is invalid"
                            )
                        if not sampled and (
                            row.get("origin_sampling") != "hierarchy_only"
                            or day != origins[0]
                            or row.get("likelihood_series_in_batch") != 0
                            or row.get("likelihood_informative_series_in_batch") != 0
                        ):
                            raise ValueError(
                                "checkpoint journal hierarchy-only evidence is invalid"
                            )
                        observed_in_journal = objective_masks[day][sampled].any(1)
                        if (
                            row.get("likelihood_series_in_batch") != len(sampled)
                            or row.get("likelihood_informative_series_in_batch")
                            != int(observed_in_journal.sum())
                        ):
                            raise ValueError("checkpoint journal informative coverage is invalid")
                        sampled_indices = np.asarray(sampled, dtype=np.int64)
                        reconstructed_proposals[sampled_indices] += 1
                        reconstructed_visits[
                            sampled_indices[observed_in_journal]
                        ] += 1
                        reconstructed_epoch_proposals[sampled_indices] += 1
                        reconstructed_epoch_visits[
                            sampled_indices[observed_in_journal]
                        ] += 1
                        reconstructed_seen[
                            sampled_indices[observed_in_journal]
                        ] = True
                        if "epoch_coverage" in row:
                            expected_epoch = dict(
                                epoch=row["update"] // phase_steps_per_epoch["selection"],
                                proposed_series=int((
                                    reconstructed_epoch_proposals[phase_trainable] > 0
                                ).sum()),
                                minimum_row_proposals=int(
                                    reconstructed_epoch_proposals[phase_trainable].min()
                                ),
                                maximum_row_proposals=int(
                                    reconstructed_epoch_proposals[phase_trainable].max()
                                ),
                                informative_series=int((
                                    reconstructed_epoch_visits[phase_trainable] > 0
                                ).sum()),
                                minimum_informative_visits=int(
                                    reconstructed_epoch_visits[phase_trainable].min()
                                ),
                                maximum_informative_visits=int(
                                    reconstructed_epoch_visits[phase_trainable].max()
                                ),
                            )
                            if json.dumps(row["epoch_coverage"], sort_keys=True) != json.dumps(
                                expected_epoch, sort_keys=True
                            ):
                                raise ValueError(
                                    "checkpoint journal epoch coverage differs from sampled rows"
                                )
                            reconstructed_epochs.append(expected_epoch)
                            reconstructed_epoch_visits.fill(0)
                            reconstructed_epoch_proposals.fill(0)
                        slot = row.get("validation_slot")
                        if "validation_slot" in row and (
                            type(slot) is not dict
                            or slot.get("covered_series")
                            != int(reconstructed_seen[phase_trainable].sum())
                            or slot.get("evaluated")
                            != bool(reconstructed_seen[phase_trainable].all())
                        ):
                            raise ValueError(
                                "checkpoint validation coverage differs from objective masks"
                            )
                if any(
                    coverage["epochs"] != journal_epochs[key]
                    or coverage["informative_series_visits"] != journal_visits[key]
                    for key, coverage in resumed["phase_coverage"].items()
                ):
                    raise ValueError(
                        "checkpoint completed-phase coverage differs from committed journal"
                    )
                if (
                    json.dumps(validation_slots, sort_keys=True)
                    != json.dumps(journal_slots, sort_keys=True)
                    or json.dumps(evaluations, sort_keys=True)
                    != json.dumps(journal_evaluations, sort_keys=True)
                ):
                    raise ValueError(
                        "checkpoint validation ledger differs from committed journal"
                    )
                replay = replay_sampler_journal(
                    [row for row in journal_rows if row.get("phase") == phase],
                    phase=phase,
                    objective_masks=objective_masks,
                    blocks=blocks,
                    batch_size=c.batch_size,
                    seed=c.seed,
                    epochs=c.epochs,
                    hierarchy_every=c.hierarchy_every,
                    hierarchy_only_enabled=bool(c.hierarchy_loss_weight and self.levels),
                )
                if (
                    replay["steps_per_epoch"] != steps_per_epoch
                    or not np.array_equal(resumed["block_order"].numpy(), replay["block_order"])
                    or resumed["block_position"] != replay["block_position"]
                    or type(resumed["rng"]) is not dict
                    or resumed["rng"].get("generator") != replay["rng_state"]
                    or not np.array_equal(resumed["seen_series"].numpy(), replay["seen"])
                    or not np.array_equal(resumed["likelihood_visits"].numpy(), replay["visits"])
                    or not np.array_equal(resumed["proposal_visits"].numpy(), replay["proposals"])
                    or not np.array_equal(resumed["epoch_visits"].numpy(), replay["epoch_visits"])
                    or not np.array_equal(
                        resumed["epoch_proposals"].numpy(), replay["epoch_proposals"]
                    )
                    or resumed["epoch_coverage"] != replay["epoch_coverage"]
                ):
                    raise ValueError("checkpoint sampler state differs from committed journal")
                if resumed["phase_done"] and (
                    resumed["phase_coverage"].get(phase) != replay["phase_summary"]
                ):
                    raise ValueError("checkpoint completed-phase coverage differs from sampler")
                if phase == "refit" and contract.mode == "rolling_backtest":
                    selection_masks = {day: self._objective_mask(day) for day in train}
                    selection_trainable = np.stack([
                        mask.any(1) for mask in selection_masks.values()
                    ]).any(0)
                    selection_blocks = [
                        np.flatnonzero((self.groups == group) & selection_trainable)
                        for group in np.unique(self.groups)
                    ]
                    selection_blocks = [block for block in selection_blocks if len(block)]
                    selection_replay = replay_sampler_journal(
                        [row for row in journal_rows if row.get("phase") == "selection"],
                        phase="selection", objective_masks=selection_masks,
                        blocks=selection_blocks, batch_size=c.batch_size, seed=c.seed,
                        epochs=c.epochs, hierarchy_every=c.hierarchy_every,
                        hierarchy_only_enabled=bool(c.hierarchy_loss_weight and self.levels),
                    )
                    if (
                        resumed["phase_coverage"].get("selection")
                        != selection_replay["phase_summary"]
                    ):
                        raise ValueError(
                            "checkpoint selection coverage differs from committed sampler"
                        )
                if contract.mode == "fixed_validation":
                    if resumed["phase_done"]:
                        expected_coverage = dict(
                            epochs=reconstructed_epochs,
                            series_seen=int(reconstructed_seen[phase_trainable].sum()),
                            informative_series_visits=int(reconstructed_visits.sum()),
                            minimum_visits_per_series=int(
                                reconstructed_visits[phase_trainable].min()
                            ),
                            maximum_visits_per_series=int(
                                reconstructed_visits[phase_trainable].max()
                            ),
                            minimum_row_proposals=int(
                                reconstructed_proposals[phase_trainable].min()
                            ),
                            maximum_row_proposals=int(
                                reconstructed_proposals[phase_trainable].max()
                            ),
                        )
                        if json.dumps(
                            resumed["phase_coverage"].get("selection"), sort_keys=True
                        ) != json.dumps(expected_coverage, sort_keys=True):
                            raise ValueError(
                                "checkpoint phase coverage differs from sampled rows"
                            )
                    if (
                        not np.array_equal(resumed["seen_series"].numpy(), reconstructed_seen)
                        or not np.array_equal(
                            resumed["likelihood_visits"].numpy(), reconstructed_visits
                        )
                        or not np.array_equal(
                            resumed["proposal_visits"].numpy(), reconstructed_proposals
                        )
                    ):
                        raise ValueError("checkpoint coverage differs from committed journal")
                    verify_validation_ledger(
                        validation_slots, evaluations,
                        metric=metric_name,
                        validation_interval_updates=(
                            phase_steps_per_epoch["selection"] * c.validation_every_epochs
                            if c.epochs is not None else c.evaluate_every
                        ),
                        budget_updates=budget,
                        last_update=resumed["update"],
                        trainable_series=int(selection_rows.any(0).sum()),
                    )
                update, optimizer_steps = resumed["update"], resumed["optimizer_steps"]
                if scaler is not None:
                    scaler_state = resumed.get("amp_scaler")
                    if not isinstance(scaler_state, dict) or not scaler_state:
                        raise ValueError("mixed-precision checkpoint lacks GradScaler state")
                    scaler.load_state_dict(scaler_state)
                    if not math.isfinite(scaler.get_scale()) or scaler.get_scale() <= 0:
                        raise ValueError("mixed-precision checkpoint has invalid GradScaler scale")
                    amp_overflow_retries = resumed["amp_overflow_retries"]
                parameter_updates = list(resumed["parameter_updates"])
                stale, phase_done = resumed["stale"], resumed["phase_done"]
                block_order = resumed["block_order"].numpy().copy()
                block_position = resumed["block_position"]
                seen_series = resumed["seen_series"].numpy().copy()
                likelihood_visits = resumed["likelihood_visits"].numpy().copy()
                proposal_visits = resumed["proposal_visits"].numpy().copy()
                epoch_visits = resumed["epoch_visits"].numpy().copy()
                epoch_proposals = resumed["epoch_proposals"].numpy().copy()
                epoch_coverage = copy.deepcopy(resumed["epoch_coverage"])
                loss_sum, epoch_loss_sum = resumed["loss_sum"], resumed["epoch_loss_sum"]
                epoch_loss_count = resumed["epoch_loss_count"]
                if self.restore_feature_training_state:
                    self.restore_feature_training_state(resumed["feature_state"])
                restore_training_rng(resumed["rng"], rng, device=c.device)
                run._commit_recovery()

            def amp_context(*, scaler=scaler):
                return (
                    torch.autocast(c.device, dtype=torch.float16, cache_enabled=False)
                    if scaler is not None else nullcontext()
                )

            def complete_update(
                backward, *, region="", diagnostics=None, rng=rng,
                scaler=scaler, optimizer=optimizer, model=model,
            ):
                nonlocal amp_overflow_retries
                replay_rng = capture_training_rng(rng) if scaler is not None else None
                replay_features = (
                    copy.deepcopy(self.feature_training_state())
                    if scaler is not None and self.feature_training_state else None
                )
                for attempt in range(24 if scaler is not None else 1):
                    if attempt:
                        assert replay_rng is not None
                        restore_training_rng(replay_rng, rng, device=c.device)
                        if self.restore_feature_training_state is not None:
                            self.restore_feature_training_state(copy.deepcopy(replay_features))
                    optimizer.zero_grad(set_to_none=True)
                    objective = backward()
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    with self._profile_region("training", f"{region}gradient_clip"):
                        from autoforecast.amp_gradients import clip_gradients

                        gradient_norm, component_overflow = clip_gradients(
                            model.parameters(), c.gradient_clip,
                            allow_amp_overflow=scaler is not None,
                        )
                        if component_overflow:
                            if step_optimizer():
                                raise RuntimeError(
                                    "GradScaler stepped despite nonfinite gradients"
                                )
                            amp_overflow_retries += 1
                            continue
                    diagnostic = diagnostics() if diagnostics is not None else {}
                    with self._profile_region("training", f"{region}optimizer_step"):
                        if step_optimizer():
                            return objective, gradient_norm, diagnostic
                    amp_overflow_retries += 1
                raise FloatingPointError("rolling AMP could not complete an update in 24 tries")

            progress.begin(phase, budget, initial=update)
            if not restoring:
                checkpoint("phase_initial")
            stopped = phase_done or (phase == "selection" and stale >= c.patience)
            first_update = budget + 1 if stopped else update + 1
            for update in range(first_update, budget + 1):
                progress.stage("training")
                if self.performance_profiler is not None:
                    self.performance_profiler.begin_step("training")
                update_started = time.perf_counter()
                likelihood_rows = informative_batch_rows = 0
                origin_sampling = "hierarchy_only"
                # Keep the cosine horizon at or beyond the sealed refit budget; full-data
                # refit may have a different steps-per-epoch cadence from selection.
                lr = c.learning_rate * (
                    0.1
                    + 0.9
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (update - 1)
                            / (
                                max(c.epochs * steps_per_epoch, budget)
                                if c.epochs is not None and phase == "refit"
                                else c.epochs * steps_per_epoch
                                if c.epochs is not None
                                else c.max_updates
                            )
                        )
                    )
                    / 2
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                model.train()
                if hierarchy is not None and update % c.hierarchy_every == 0 and c.epochs is None:
                    day = origins[0]
                    stop = data.position(day)

                    def hierarchy_mean(rows, *, model=model, day=day):
                        with amp_context():
                            return self.mean(model, self.batch(day, rows, c.device, False)[0])

                    def backward(
                        *, model=model, hierarchy=hierarchy, stop=stop, scaler=scaler,
                    ):
                        with self._profile_region("training", "hierarchy_backward"):
                            return staged_hierarchy_backward(
                                model, self.chunks(), hierarchy_mean, hierarchy,
                                torch.tensor(data.target[:, stop : stop + data.horizon]),
                                weight=c.hierarchy_loss_weight,
                                epsilon=c.hierarchy_epsilon,
                                progress=progress.stage,
                                backward_scale=scaler.scale if scaler is not None else None,
                            )
                else:
                    if block_position == len(blocks):
                        block_order, block_position = rng.permutation(len(blocks)), 0
                    chosen = block_order[block_position : block_position + groups_per_batch]
                    block_position += len(chosen)
                    rows = np.concatenate([blocks[i] for i in chosen])
                    eligible_origins = np.flatnonzero(informative_rows[:, rows].any(1))
                    missing = ~seen_series[rows]
                    if missing.any():
                        # Exposure-only coverage curriculum: never count masked pre-launch rows.
                        covered = informative_rows[:, rows[missing]].sum(1)
                        choices = np.flatnonzero(covered == covered.max())
                        importance = 1.0
                        origin_sampling = "coverage_curriculum"
                    else:
                        # Zero-information origin proposals have zero likelihood; condition on
                        # usable proposals and retain their uniform-origin importance factor.
                        choices = eligible_origins
                        importance = len(choices) / len(origins)
                        origin_sampling = "uniform_usable_origins"
                    origin_index = int(rng.choice(choices))
                    day = origins[origin_index]
                    with self._profile_region("training", "batch_assembly_and_transfer"):
                        inputs, y, observed = self.batch(day, rows, c.device, True)
                    assert y is not None and observed is not None
                    if not observed.any():
                        raise ValueError("informative-origin schedule produced an empty batch")
                    observed_rows = observed.any(1).detach().cpu().numpy()
                    seen_series[rows] |= observed_rows
                    likelihood_visits[rows] += observed_rows
                    proposal_visits[rows] += 1
                    epoch_visits[rows] += observed_rows
                    epoch_proposals[rows] += 1
                    likelihood_rows, informative_batch_rows = len(rows), int(observed_rows.sum())
                    denominator = informative[day] * len(chosen) / len(blocks)

                    def backward(
                        *, model=model, inputs=inputs, y=y, observed=observed,
                        denominator=denominator, importance=importance, scaler=scaler,
                    ):
                        with self._profile_region("training", "forward_and_loss"):
                            with amp_context():
                                loss = self.training_loss(
                                    model, inputs, y, observed, denominator, importance
                                )
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite training loss")
                        with self._profile_region("training", "backward"):
                            (scaler.scale(loss) if scaler is not None else loss).backward()
                        return float(loss.detach())

                def collect_diagnostics(
                    *, model=model, update=update, steps_per_epoch=steps_per_epoch,
                ):
                    with self._profile_region("training", "diagnostics"):
                        return (
                            self.diagnostics(model, update == 1 or update % steps_per_epoch == 0)
                            if self.diagnostics else {}
                        )

                combined_hierarchy: dict[str, float] = {}
                if (
                    c.epochs is not None
                    and hierarchy is not None and update % c.hierarchy_every == 0
                ):
                    ordinary_backward = backward
                    hierarchy_day = origins[0]
                    hierarchy_stop = data.position(hierarchy_day)

                    def combined_backward(
                        *, ordinary_backward=ordinary_backward, model=model,
                        hierarchy=hierarchy, hierarchy_day=hierarchy_day,
                        hierarchy_stop=hierarchy_stop, scaler=scaler,
                        combined_hierarchy=combined_hierarchy,
                    ):
                        ordinary_loss = ordinary_backward()

                        def hierarchy_mean(chunk_rows):
                            with amp_context():
                                return self.mean(
                                    model, self.batch(hierarchy_day, chunk_rows, c.device, False)[0]
                                )

                        with self._profile_region("training", "epoch_hierarchy_backward"):
                            hierarchy_started = time.perf_counter()
                            hierarchy_loss = staged_hierarchy_backward(
                                model, self.chunks(), hierarchy_mean, hierarchy,
                                torch.tensor(
                                    data.target[
                                        :, hierarchy_stop : hierarchy_stop + data.horizon
                                    ]
                                ),
                                weight=c.hierarchy_loss_weight,
                                epsilon=c.hierarchy_epsilon,
                                progress=progress.stage,
                                backward_scale=scaler.scale if scaler is not None else None,
                            )
                        combined_hierarchy["loss"] = hierarchy_loss
                        combined_hierarchy["seconds"] = time.perf_counter() - hierarchy_started
                        return ordinary_loss + hierarchy_loss

                    backward = combined_backward

                objective, gradient_norm, diagnostic = complete_update(
                    backward, diagnostics=collect_diagnostics
                )
                loss_sum += objective
                epoch_loss_sum += objective
                epoch_loss_count += 1
                record = {
                    "phase": phase,
                    "update": update,
                    "origin": day,
                    "loss": objective,
                    "learning_rate": lr,
                    "likelihood_series_in_batch": likelihood_rows,
                    "likelihood_informative_series_in_batch": informative_batch_rows,
                    "origin_sampling": origin_sampling,
                    "sampled_series_indices": rows.tolist()
                    if origin_sampling != "hierarchy_only"
                    else [],
                    "training_objective_running_mean": loss_sum / update,
                }
                if scaler is not None:
                    record["precision"] = self.precision
                    record["amp_overflow_retries"] = amp_overflow_retries
                if combined_hierarchy:
                    record["hierarchy_loss"] = combined_hierarchy["loss"]
                    if self.diagnostics:
                        record["hierarchy_seconds"] = combined_hierarchy["seconds"]
                        record["hierarchy_step"] = dict(
                            origin=origins[0],
                            weighted_objective=combined_hierarchy["loss"],
                            weight=c.hierarchy_loss_weight,
                            gradient_norm_before_clip=float(gradient_norm),
                            seconds=combined_hierarchy["seconds"],
                            full_population=len(data.series_ids),
                            optimizer_step=update,
                        )
                if self.diagnostics:
                    record.update(
                        diagnostic,
                        gradient_norm_before_clip=float(gradient_norm),
                        ordinary_update_seconds=time.perf_counter() - update_started,
                    )
                    if origin_sampling == "hierarchy_only":
                        record["loss_components"] = {"weighted_hierarchy": objective}
                if c.epochs is not None:
                    phase_epochs = math.ceil(budget / steps_per_epoch)
                    if budget % steps_per_epoch == 0:
                        record.update(
                            epoch=(update - 1) // steps_per_epoch + 1,
                            epochs=phase_epochs,
                            steps_per_epoch=steps_per_epoch,
                        )
                    if update % steps_per_epoch == 0:
                        coverage = dict(
                            epoch=update // steps_per_epoch,
                            proposed_series=int((epoch_proposals[phase_trainable] > 0).sum()),
                            minimum_row_proposals=int(epoch_proposals[phase_trainable].min()),
                            maximum_row_proposals=int(epoch_proposals[phase_trainable].max()),
                            informative_series=int((epoch_visits[phase_trainable] > 0).sum()),
                            minimum_informative_visits=int(epoch_visits[phase_trainable].min()),
                            maximum_informative_visits=int(epoch_visits[phase_trainable].max()),
                        )
                        epoch_coverage.append(coverage)
                        record["epoch_coverage"] = coverage
                        record["epoch_training_objective"] = epoch_loss_sum / epoch_loss_count
                        epoch_loss_sum, epoch_loss_count = 0.0, 0
                        epoch_visits.fill(0)
                        epoch_proposals.fill(0)
                if self.performance_profiler is not None:
                    self.performance_profiler.end_step("training")
                due_for_validation = (
                    phase == "selection"
                    and self.validation_due(update, budget, steps_per_epoch)
                )
                coverage_ready = bool(seen_series[phase_trainable].all())
                if due_for_validation:
                    slot: dict[str, object] = {
                        "update": update,
                        "covered_series": int(seen_series[phase_trainable].sum()),
                        "trainable_series": int(phase_trainable.sum()),
                        "evaluated": coverage_ready,
                    }
                    validation_slots.append(slot)
                    record["validation_slot"] = slot
                if due_for_validation and coverage_ready:
                    display_loss = loss_sum / update
                    if c.epochs is not None:
                        display_loss = (
                            record["epoch_training_objective"]
                            if "epoch_training_objective" in record
                            else epoch_loss_sum / epoch_loss_count
                        )
                    record["optimizer_steps"] = optimizer_steps
                    progress.update(
                        record,
                        running_mean=display_loss,
                        best=best_score if math.isfinite(best_score) else None,
                        stale=stale,
                        latest_validation=latest_validation,
                    )
                    scores = []
                    if self.performance_profiler is not None:
                        # A Kineto profiler is thread-global; training must be sealed before
                        # the validation profiler can start, including tiny smoke runs whose
                        # training phase has fewer than the normal nine profiled updates.
                        self.performance_profiler.finish_phase("training")
                    for index, day in enumerate(validation):
                        progress.begin_validation(
                            day,
                            index=index + 1,
                            total=len(validation),
                            batches=len(self.chunks()),
                            kind=(
                                "fixed holdout"
                                if contract.mode == "fixed_validation"
                                else "internal early-stopping backtest"
                            ),
                        )
                        try:
                            scores.append(
                                self.validate(
                                    model,
                                    day,
                                    metrics[index] if metrics else None,
                                    progress=progress,
                                )
                            )
                        finally:
                            progress.end_validation()
                    if self.performance_profiler is not None:
                        # Do not let a short validation capture span later training epochs.
                        self.performance_profiler.finish_phase("validation")
                    value = float(np.mean([score[metric_name] for score in scores]))
                    record["validation"] = dict(zip(validation, scores, strict=True))
                    evaluations.append({"update": update, f"mean_{metric_name}": value})
                    if self._improved(value, best_score):
                        selected_updates, best_score, stale = update, value, 0
                        best_state = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }
                    else:
                        stale += 1
                    latest_validation = dict(
                        update=update,
                        epoch=record.get("epoch"),
                        metric=value,
                        metric_name=metric_name,
                        nll=float(np.mean([s["validation_nll"] for s in scores]))
                        if self.validation_values
                        else None,
                    )
                    record["validation_summary"] = dict(
                        latest_validation,
                        best_score=best_score,
                        stale=stale,
                        evaluations=len(evaluations),
                    )
                    progress.validation_summary(
                        latest_validation,
                        train_loss=display_loss,
                        best=best_score,
                        stale=stale,
                        patience=c.patience,
                    )
                else:
                    record["optimizer_steps"] = optimizer_steps
                    display_loss = loss_sum / update
                    if c.epochs is not None:
                        display_loss = (
                            record["epoch_training_objective"]
                            if "epoch_training_objective" in record
                            else epoch_loss_sum / epoch_loss_count
                        )
                    progress.update(
                        record,
                        running_mean=display_loss,
                        best=best_score if math.isfinite(best_score) else None,
                        stale=stale,
                        latest_validation=latest_validation,
                    )
                log(record)
                if "validation" in record or run and run.due(update):
                    checkpoint("validation" if "validation" in record else "periodic")
                if phase == "selection" and stale >= c.patience:
                    break
            if not seen_series[phase_trainable].all() or (
                phase == "selection" and not selected_updates
            ):
                raise ValueError(f"update budget cannot cover all series for {phase}")
            phase_coverage[phase] = dict(
                epochs=epoch_coverage,
                series_seen=int(seen_series[phase_trainable].sum()),
                informative_series_visits=int(likelihood_visits.sum()),
                minimum_visits_per_series=int(likelihood_visits[phase_trainable].min()),
                maximum_visits_per_series=int(likelihood_visits[phase_trainable].max()),
                minimum_row_proposals=int(proposal_visits[phase_trainable].min()),
                maximum_row_proposals=int(proposal_visits[phase_trainable].max()),
            )
            if not phase_done:
                phase_done = True
                checkpoint("phase_complete")
        if contract.mode == "fixed_validation":
            if best_state is None:
                raise ValueError("fixed validation did not select a checkpoint")
            model.load_state_dict(best_state, strict=True)
        refit_optimizer_steps = optimizer_steps if phase == "refit" else None
        if refit_optimizer_steps is not None and refit_optimizer_steps != selected_updates:
            raise ValueError("fresh refit optimizer steps differ from selected_updates")
        return model, {
            "config": asdict(c),
            **({"precision": self.precision} if self.precision != "32-true" else {}),
            "training_contract": contract.as_record(),
            "validation_origins": validation,
            "selection_training_origins": train,
            "refit_training_origins": final_train,
            "selected_updates": selected_updates,
            "refit_optimizer_steps": refit_optimizer_steps,
            "validation": evaluations,
            "validation_slots": validation_slots,
            "refit_series_count": int(refit_rows.any(0).sum()),
            "refit_series_seen_in_likelihood": int(seen_series[refit_rows.any(0)].sum()),
            "excluded_series_count": int((~trainable_series).sum()),
            "selection_metric": metric_name,
            "hierarchy_weighting": hierarchy_weighting,
            "epochs_per_phase": c.epochs,
            "steps_per_epoch": steps_per_epoch,
            "epoch_definition": "one_pass_over_blocks_one_sampled_origin_per_batch",
            "budget_policy": (
                "validation_selected_updates"
                if contract.mode != "fixed_refit"
                else "sealed_fixed_update_budget"
            ),
            "phase_admission": phase_admission,
            "phase_coverage": phase_coverage,
            "sampler": "informative_coverage_then_uniform_usable_origins_v1",
            "optimizer_groups": [
                {
                    "weight_decay": g["weight_decay"],
                    "parameter_count": sum(p.numel() for p in g["params"]),
                }
                for g in optimizer.param_groups
            ],
            "refit_initialization": (
                "not_applicable_selected_checkpoint_restored"
                if contract.mode == "fixed_validation"
                else "fresh_same_seed_not_selected_checkpoint"
            ),
        }
