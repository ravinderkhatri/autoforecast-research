"""The custom model as one entry in the registry. THE ONLY FILE THAT MAY IMPORT `forecast/`.

The adapter connects forecasting models to the common evaluation harness and fixed scoring grids.

WHAT `first_build=True` REFUSES, and this adapter honours every one: the adaptive event kernel,
event interactions, the peer-product branch, the cross-product factor, and the promotion
decomposition. They are Phase 5, each earning its place by an episode-split ablation. Building
them now would confound "the architecture works" with "one of five additions happened to help".

THE DISTRIBUTIONAL OUTPUT IS SAMPLE PATHS. The harness scores a point forecast at this phase, so
the hurdle distribution's count-analytic mean is integrated over the factor by deterministic
quadrature and written without Monte Carlo noise. That factor integral is numerical, not closed
form. The integer paths are persisted separately from the same fitted parameters; Gate 4's
cumulative lead-time calibration cannot be computed from marginal means or quantiles.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import functools
import hashlib
import inspect
import math
import os
import pathlib
import pickle
import shutil
import stat
import sys
import tempfile
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from forecast.features import (
    calendar_events,
    commercial,
    demand_history,
    static_identity,
)
from forecast.models.mft.branches import (
    CalendarEventBranch,
    CommercialBranch,
    DemandHistoryBranch,
    StaticIdentityBranch,
    _binary_mask,
)
from forecast.models.mft.decoder import Decoder
from forecast.models.mft.factor import SharedLatentFactor
from forecast.models.mft.fusion import GatedFusion
from forecast.models.mft.heads import (
    LOG_K_MAX,
    LOG_K_MIN,
    HurdleNegBinHead,
    NegBinHead,
    hurdle_mean,
)
from forecast.models.mft.pact import PACTBatch, PACTLabels, PACTModelInputs
from forecast.models.mft.regularisation import (
    drop_covariates,
    monotonicity_penalty,
    price_violation_rate,
)
from forecast.models.mft.sample import sample_paths, sample_paths_negbin
from forecast.models.mft.seasonality import SeasonalAmplitude
from forecast.models.mft.selection import StaticVariableSelection
from forecast.models.mft.trunk import Trunk
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from spec.resolved import RESOLVED_BUILD_SCHEMA, ResolvedBuild
from torch import nn

from autoforecast import resources
from autoforecast._numeric import is_finite_real
from autoforecast.adapters import _validate_daily_span
from autoforecast.model_names import REVISION_MODELS, model_name
from autoforecast.panel import Forecast, Panel, period_days
from autoforecast.rolling_data import GLOBAL_LEFT_PADDING_CONTRACT
from autoforecast.series_loss_weighting import series_loss_weights, weighted_series_mean
from provenance.lineage import directory_digest

NAME = "mft"
SFT_NAME = "sft"
HMFNET_NAME = "hmfnet"
PACT_NAME = "pact"

EMITTED = "emitted"
DECLINED_NO_WINDOW = "declined_no_encoder_window"
OOV_SCHEME = "pooled-identity-v1"
"""Checkpoint contract for identity rows that had no optimiser evidence.

Every unevidenced series/product residual is exactly zero.  Every unevidenced factor-loading
row is the exact mean of the evidenced loading rows.  The evidence row sets ride the checkpoint,
so replay can validate this claim instead of treating a random, never-trained embedding as a
pooled cold start.
"""

WARM_START_SCHEMA = "autoforecast.warm-start.v1"
"""Checkpoint lineage record for one non-sharded continuation fit."""

_WARM_START_NAMESPACE_FIELDS = (
    "origin",
    "horizon",
    "encoder_days",
    "train_stride",
    "train_windows",
    "batch_size",
    "gradient_microbatch_size",
    "importance_draws",
    "events",
    "event_max_support",
    "event_windows",
    "cycles",
    "seed",
    "resolved_build",
    "resolved_build_hash",
    "training_controls",
    "factor_group_col",
    "factor_enabled",
    "topology",
    "product_col",
    "store_col",
    "category_col",
    "known_names",
    "declared_known_drivers",
    "static_real_stats",
    "vocabularies",
    "oov_rows",
    "oov_scheme",
    "identity_evidence_rows",
    "plan_provenance",
    "panel",
    "panel_parquet_sha256",
    "series_ids",
    "tensors",
)
"""Run fields whose exact row/name/data semantics must survive a warm start."""

SEASONAL_CYCLES: tuple[tuple[str, float, int], ...] = (
    ("weekly", 7.0, 3),
    ("monthly", 30.4375, 2),
    ("annual", 365.25, 3),
)
"""`(name, period in days, harmonics)` for each cyclic coordinate the decoder receives.

**ALL THREE WERE WRONG IN THE SAME WAY, and two of them silently.** The calendar columns carried
`day_of_week = weekday/6` and `day_of_month = (day-1)/30` -- LINEAR RAMPS. Sunday sits at 1.0 and
Monday at 0.0, so the model is told two adjacent days are maximally far apart; the 31st and the
1st say the same thing. A cycle encoded as a ramp has a discontinuity exactly where the cycle
closes, which is where the interesting behaviour usually is (weekend into week, month end into
month start).

The fix for all three is one thing: a sine/cosine PAIR per harmonic, which is continuous across
the wrap by construction.

**HARMONICS ARE RESOLUTION, and that is why there are three cycles here and not six.** Asking for
"month of year" and "week of year" as separate features asks for the same 365.25-day cycle at two
resolutions: annual harmonic 6 resolves roughly a month, harmonic 26 roughly a week. Adding both
as named columns would put the same information in twice, correlated almost perfectly, and the two
sets of weights would trade off against each other with nothing to break the tie. "Week of month"
is likewise the monthly cycle at weekly resolution, which harmonic 2 already carries.

So the tunable thing is the harmonic count per cycle, and it is a per-run parameter precisely so
an ablation can ask whether month-of-year resolution (annual=6) beats quarterly (annual=3) without
adding a redundant column set.

Harmonics per cycle are chosen by how many observations support them:
  * weekly, 3 -- 7 points, and 3 harmonics is 6 basis functions, near-saturating a weekly shape
    the model has seen hundreds of times.
  * monthly, 2 -- the pay-day and month-end shape is broad; more harmonics would fit the noise
    between them. 30.4375 is the mean Gregorian month, so the phase does not drift across years.
  * annual, 3 -- resolves roughly a four-month feature and no finer, because a series has seen
    about five annual cycles and a finer basis would pass exactly through all five.

The linear ramps are KEPT alongside these. They are not harmful once a cyclic coordinate exists --
`is_weekend` in particular is a genuine step change rather than a smooth phase -- and removing
them would change two things at once in an ablation designed to change one.
"""

ANNUAL_HARMONICS = 3
"""Fourier harmonics over the annual cycle, giving the decoder a smooth ANNUAL COORDINATE.

**Without this there is no channel for seasonality nobody named.** `CalendarEventBranch` handles
Christmas and Easter because they arrive as events with dates; it cannot handle back-to-school,
summer drift or a pay-day cycle, because those are not events -- they are a position in the year,
and `known_future` carried day-of-week, day-of-month and weekend but nothing annual.

Three harmonics resolve roughly a four-month feature and no finer. That is deliberate: a series
has seen about five annual cycles, and a basis fine enough to fit a single week of the year would
be fitting five observations with a curve that can pass through all of them.

A Fourier pair rather than a day-of-year index, because the coordinate must be CYCLIC -- 31
December and 1 January are one day apart, and an index makes them 364 apart.
"""


def annual_columns(harmonics: int = ANNUAL_HARMONICS) -> tuple[str, ...]:
    """Column names for the annual Fourier basis, in the order `_known_future` emits them."""
    return cycle_columns("annual", harmonics)


def cycle_columns(name: str, harmonics: int) -> tuple[str, ...]:
    """Column names for one cyclic basis, in emission order."""
    return tuple(f"{name}_{fn}_{k}" for k in range(1, harmonics + 1) for fn in ("sin", "cos"))


def cycle_terms(day: dt.date, period: float, harmonics: int) -> list[float]:
    """Sine/cosine pairs placing `day` on a cycle of `period` days.

    The phase runs off the ORDINAL day number rather than a within-period index, so a cycle whose
    period is not a whole number of days -- the mean month, the mean year -- stays aligned instead
    of drifting a little further out of phase every time it wraps.
    """
    phase = 2.0 * math.pi * (day.toordinal() % period) / period
    out: list[float] = []
    for k in range(1, harmonics + 1):
        out.append(math.sin(k * phase))
        out.append(math.cos(k * phase))
    return out


__all__ = [
    "AGGREGATE_TRAJECTORY_ANCESTOR_COUNT",
    "AGGREGATE_TRAJECTORY_CALIBRATION_ORIGINS",
    "AGGREGATE_TRAJECTORY_EVALUATION_ORIGIN",
    "AGGREGATE_TRAJECTORY_GROUPING_COLS",
    "AGGREGATE_TRAJECTORY_HORIZON",
    "AGGREGATE_TRAJECTORY_MODE",
    "AGGREGATE_TRAJECTORY_PROJECTION_CONTROLS",
    "AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_CONTROLS",
    "AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_SCHEMA",
    "AGGREGATE_TRAJECTORY_PROJECTION_SCHEMA",
    "AGGREGATE_TRAJECTORY_SCHEMA",
    "AggregateTrajectoryCalibrationEpisode",
    "ANNUAL_HARMONICS",
    "COMMERCIAL_BUNDLE",
    "NAME",
    "RESIDUAL_TABLES",
    "SEASONAL_CYCLES",
    "CustomModel",
    "aggregate_trajectory_ancestor_id",
    "annual_columns",
    "cycle_columns",
    "cycle_terms",
    "KnownFuture",
    "OursError",
    "OOV_SCHEME",
    "POINT_FACTOR_DRAWS",
    "PRICE_LADDER_RELATIVE_SCALE_FLOOR",
    "PriceLadderEvaluation",
    "ReferencePriceContrastEvaluation",
    "RNG_DOMAIN_SCHEME",
    "default_factor_draws",
    "point_forecast",
    "mean_over_factor_quadrature",
    "draw_paths_over_factor",
    "evaluate_price_ladder",
    "evaluate_reference_price_contrast",
    "evaluating",
    "continue_aggregate_trajectory_projection",
    "expected_demand_fn",
    "fit_aggregate_trajectory_projection",
    "fit_predict",
    "factor_block_batches",
    "load_checkpoint",
    "mean_over_factor",
    "parameter_groups",
    "pool_panels",
    "predict_from_checkpoint",
    "validate_aggregate_trajectory_artifact",
    "predicted_mean",
    "predicted_paths",
    "shard_plan",
    "verify_price_ladder",
    "save_checkpoint",
    "intervention_bundle",
    "model_diagnostics",
    "split_by_series",
]


class OursError(ValueError):
    """The model was asked for something this build refuses."""


AGGREGATE_TRAJECTORY_SCHEMA = "autoforecast.aggregate-trajectory.v1"
AGGREGATE_TRAJECTORY_MODE = "one_ancestor_uplift_v1"
AGGREGATE_TRAJECTORY_GROUPING_COLS = ("store_id", "dept_id")
AGGREGATE_TRAJECTORY_ANCESTOR_COUNT = 70
AGGREGATE_TRAJECTORY_CALIBRATION_ORIGINS = ("2016-02-28", "2016-03-27")
AGGREGATE_TRAJECTORY_EVALUATION_ORIGIN = "2016-04-24"
AGGREGATE_TRAJECTORY_HORIZON = 28
AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER = "aggregate_trajectory_projection.weight"
"""The single provider-agnostic aggregate feature contract and its only trainable parameter."""

AGGREGATE_TRAJECTORY_PROJECTION_CONTROLS = {
    "epochs": 10,
    "learning_rate": 0.003,
    "batch_size": 2048,
    "gradient_microbatch_size": 128,
    "gradient_clip_norm": 1.0,
    "seed": 0,
    "device": "cpu",
    "optimizer": "AdamW",
    "weight_decay": 0.0,
    "evaluation_schedule": "final_only",
}

AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_CONTROLS = {
    "max_additional_epochs": 5000,
    "patience": 15,
    "learning_rate": 0.0003,
    "batch_size": 2048,
    "gradient_microbatch_size": 128,
    "gradient_clip_norm": 1.0,
    "seed": 0,
    "device": "cpu",
    "absolute_min_delta": 0.000001,
    "relative_min_delta": 0.0001,
    "quiet_epoch_logging": True,
}
"""Registered R9/R10 weights-only projection-continuation controls."""


@dataclass(frozen=True)
class AggregateTrajectoryCalibrationEpisode:
    """One sealed offline feature episode and the labels immediately following it."""

    origin: str
    artifact: str | pathlib.Path


def aggregate_trajectory_ancestor_id(
    grouping: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> str:
    """Collision-safe canonical ID for ordered ``(grouping column, value)`` pairs."""
    if (
        not isinstance(grouping, list | tuple)
        or not grouping
        or any(
            not isinstance(pair, list | tuple)
            or len(pair) != 2
            or any(type(part) is not str or not part for part in pair)
            for pair in grouping
        )
        or len({pair[0] for pair in grouping}) != len(grouping)
    ):
        raise OursError(
            "aggregate trajectory ancestor grouping must be distinct nonempty name/value pairs"
        )
    import json as _json

    return _json.dumps(
        [list(pair) for pair in grouping],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _model_state_stamp(
    model: nn.Module,
) -> tuple[tuple[int, int, str, torch.dtype, int], ...]:
    """Identity, mutation version and storage identity of every model tensor.

    PyTorch's ordinary optimiser/in-place updates advance ``_version``. Device/dtype conversion
    is different: ``Module.to`` can replace a parameter's storage without changing either its
    Python identity or version counter. A cached float32/CPU base is not valid against the same
    module after that conversion, so dtype, full device (including its index), and storage pointer
    are part of the stamp too. Raw ``tensor.data`` content mutation that preserves all of those
    signals is intentionally outside this cache contract: detecting it would require hashing every
    parameter before every cheap factor application and would erase the caching win. Use standard
    optimiser/no-grad mutation APIs, which advance ``_version``.
    """
    return tuple(
        (id(value), int(value._version), str(value.device), value.dtype, int(value.data_ptr()))
        for value in (*model.parameters(), *model.buffers())
    )


def _execution_context(
    model: nn.Module,
) -> tuple[tuple[bool, ...], bool, str, torch.dtype, torch.dtype | None]:
    """All module modes, grad, device, dtype and autocast state of a cached computation.

    Recording only ``model.training`` is insufficient: PyTorch permits one dropout-bearing child
    to be put in train mode while the root remains in eval mode. Such a mixed tree changes the
    encoder's execution semantics without touching a parameter or the root flag.
    """
    reference = next(model.parameters())
    device_type = reference.device.type
    autocast = torch.is_autocast_enabled(device_type)
    autocast_dtype = torch.get_autocast_dtype(device_type) if autocast else None
    module_modes = tuple(module.training for module in model.modules())
    return (
        module_modes,
        torch.is_grad_enabled(),
        str(reference.device),
        reference.dtype,
        autocast_dtype,
    )


@dataclass(frozen=True)
class BasePrediction:
    """The factor-independent state of one forward pass, computed once and reused per draw.

    Holds the DECODER OUTPUT and the deterministic per-cell terms, not merely `hidden`. The
    seasonality offsets are here too: the seasonal response is a property of the series and the
    date, so it is identical under every factor draw and belongs on this side of the split.

    A trap this dataclass exists to prevent: when static variable selection is enabled,
    `CustomModel.forward` REPLACES `attributes` with `self.selection(attributes)` before the
    identity branch, the fusion gate and the head's dispersion read it. Caching `hidden` while
    letting a caller pass the raw attributes back in would silently change the dispersion. The
    SELECTED attributes travel here, so that cannot happen.
    """

    hidden: torch.Tensor
    attributes: torch.Tensor
    baseline_logit: torch.Tensor
    baseline_log_size: torch.Tensor
    series_index: torch.Tensor
    group_index: torch.Tensor
    dense_group_index: torch.Tensor
    group_labels: torch.Tensor
    store_index: torch.Tensor
    horizon: int
    price_depth: torch.Tensor | None = None
    price_response_mask: torch.Tensor | None = None
    aggregate_trajectory: torch.Tensor | None = None
    seasonal: tuple[torch.Tensor, torch.Tensor] | None = None
    selection_weights: torch.Tensor | None = None
    _model_identity: int = 0
    _model_state: tuple[tuple[int, int, str, torch.dtype, int], ...] = ()
    _context: tuple[tuple[bool, ...], bool, str, torch.dtype, torch.dtype | None] = (
        (False,),
        False,
        "cpu",
        torch.float32,
        None,
    )

    @property
    def n_cells(self) -> int:
        return int(self.hidden.shape[0])

    def validate_for(self, model: CustomModel) -> None:
        """Refuse a base whose owner, weights, or execution semantics have changed.

        `BasePrediction` is a cache entry, not a free-standing set of model inputs. The tensors
        already contain the encoder, selected attributes, seasonality and (once memoised) head
        projections from one exact model state. Reusing them with another model, after a standard
        optimiser/no-grad update, or across train/grad/autocast modes returns stale values and can
        silently detach the head from autograd. Those are contract errors, not cache misses. Raw
        ``tensor.data`` content mutation is unsupported for the reason documented by
        `_model_state_stamp`.
        """
        if self._model_identity != id(model):
            raise OursError("BasePrediction belongs to a different CustomModel instance")
        if self._context != _execution_context(model):
            raise OursError(
                "BasePrediction was encoded under a different train/grad/device/dtype/autocast "
                "context; re-encode it in the context where it will be applied"
            )
        if self._model_state != _model_state_stamp(model):
            raise OursError(
                "BasePrediction was encoded before detected model tensor state changed "
                "(identity/version/storage/dtype/device); re-encode it from the current model "
                "state. Raw tensor.data mutation is outside the supported cache contract"
            )

    def predictors(self, model: CustomModel, *, _validated: bool = False):
        """The head's factor-independent linear predictors before factor/seasonality. Memoised.

        `[cells, horizon]` tensors, so adding a `[draws, cells, horizon]` factor broadcasts.

        Seasonality still travels on this base, but separately. The legacy monolithic forward
        formed ``factor_offset + seasonal`` first and then added that sum to these predictors.
        Folding seasonality into the memo changed float32 addition order under a supplied path:
        the seasonal fixture moved occurrence probability by 5.96e-8 and size mean by 2.38e-7.
        `apply_factor` and the quadrature integrands now share the legacy ordering explicitly.

        The separation is also what makes the earlier quadrature omission visible: every factor
        integrand must compose these predictors with `base.seasonal`, rather than relying on a
        side effect of `apply_factor` that quadrature never calls.
        """
        if not _validated:
            self.validate_for(model)
        cached = self.__dict__.get("_predictors")
        if cached is not None:
            return cached
        static = self.attributes.unsqueeze(1).expand(-1, self.horizon, -1)
        series = self.series_index.unsqueeze(-1).expand(-1, self.horizon)
        if model.head_kind == "negbin":
            eta_size, log_k = model.head.linear_predictors(
                self.hidden,
                self.baseline_log_size.unsqueeze(-1).expand(-1, self.horizon),
                static,
                series,
            )
            if self.aggregate_trajectory is not None and bool(
                (self.aggregate_trajectory[..., 1] != 0.0).any()
            ):
                # The plain NB has no occurrence predictor. The aggregate feature changes only
                # its size location; dispersion is deliberately left on the existing static route.
                delta_size = model.aggregate_trajectory_projection(
                    self.aggregate_trajectory
                ).squeeze(-1)
                eta_size = eta_size + delta_size
            if model.price_response is not None:
                if self.price_depth is None or self.price_response_mask is None:
                    raise OursError(
                        "structural price response requires cached discount depth and its "
                        "fixed availability/reference mask"
                    )
                active_depth = torch.where(
                    self.price_response_mask, self.price_depth, torch.zeros_like(self.price_depth)
                )
                size_slope = (
                    torch.sigmoid(model.price_response(self.hidden).squeeze(-1))
                    * model.price_size_slope_max
                )
                eta_size = eta_size + size_slope * active_depth
            value = (eta_size, log_k)
        else:
            eta_occ, eta_size, log_k = model.head.linear_predictors(
                self.hidden,
                self.baseline_logit.unsqueeze(-1).expand(-1, self.horizon),
                self.baseline_log_size.unsqueeze(-1).expand(-1, self.horizon),
                static,
                series,
            )
            if self.aggregate_trajectory is not None and bool(
                (self.aggregate_trajectory[..., 1] != 0.0).any()
            ):
                delta = model.aggregate_trajectory_projection(self.aggregate_trajectory)
                eta_occ = eta_occ + delta[..., 0]
                eta_size = eta_size + delta[..., 1]
            if model.price_response is not None:
                if self.price_depth is None or self.price_response_mask is None:
                    raise OursError(
                        "structural price response requires cached discount depth and its "
                        "fixed availability/reference mask"
                    )
                active_depth = torch.where(
                    self.price_response_mask, self.price_depth, torch.zeros_like(self.price_depth)
                )
                slopes = torch.sigmoid(model.price_response(self.hidden))
                eta_occ = eta_occ + slopes[..., 0] * model.price_occurrence_slope_max * active_depth
                eta_size = eta_size + slopes[..., 1] * model.price_size_slope_max * active_depth
            value = (eta_occ, eta_size, log_k)
        # A grad-enabled predictor owns an autograd graph. Memoising it would either reuse a
        # consumed graph on the second backward or, if the first call happened under no_grad,
        # silently remove the head from training. Inference is the hot path this cache exists for.
        if not torch.is_grad_enabled():
            object.__setattr__(self, "_predictors", value)
        return value


class CustomModel(nn.Module):
    """Branches, gated fusion, trunk, decoder, hurdle head, and an optional shared factor.

    Assembled HERE rather than in `forecast/models/mft/` because assembly is a configuration decision
    -- which branches a client's columns support -- and the model package should not have to know
    what a client has. Each piece is separately tested; this is the wiring.
    """

    def __init__(
        self,
        d_model: int,
        encoder_days: int,
        n_attributes: int,
        n_series: int,
        n_known: int,
        horizon: int,
        *,
        factor_rank: int = 8,
        n_stores: int = 1,
        n_commercial: int = 0,
        n_events: int = 0,
        event_max_support: int = 30,
        annual_index: tuple[int, ...] = (),
        per_series_seasonality: bool = False,
        attribute_widths: tuple[int, ...] = (),
        demand_dropout: float = 0.15,
        price_index: int | None = None,
        price_available_index: int | None = None,
        discount_index: int | None = None,
        discount_available_index: int | None = None,
        promo_index: int | None = None,
        price_response_mode: str = "none",
        price_occurrence_slope_max: float | None = None,
        price_size_slope_max: float | None = None,
        event_distance_index: tuple[int, ...] = (),
        event_present_index: tuple[int, ...] = (),
        n_products: int = 0,
        text_dim: int = 0,
        embedding_residual_dim: int = 16,
        factor_dynamics: str | None = "ar1",
        head_kind: str = "hurdle_negbin",
        n_heads: int = 4,
        patch_size: int = 7,
        trunk_stages: int = 2,
        decoder_layers: int = 2,
        history_attention_layers: int = 1,
        commercial_attention_layers: int = 1,
        event_kernel_layers: int = 1,
        event_windows: tuple[tuple[int, int], ...] | None = None,
        aggregate_trajectory_mode: str = "none",
        identity_cardinalities: tuple[int, ...] = (),
        n_history_covariates: int = 0,
        commercial_attention_heads: int = 2,
        n_event_pools: int | None = None,
    ) -> None:
        """Assemble the branches a client's columns actually support.

        **FOUR branches at first build**, not two. An earlier version instantiated only demand and
        identity while `CommercialBranch` and `CalendarEventBranch` sat built and tested but
        unwired -- so the model that would have trained was a subset, and a loss against the
        shallow LightGBM model (which does use price) would have said nothing about the
        architecture.

        A branch is present when the client HAS the columns for it, absent otherwise, and the
        fusion mask carries the difference. `n_commercial=0` or `n_events=0` therefore means "this
        client has none", not "we did not get round to it".

        The PEER branch is the one legitimately absent: `first_build` refuses it, and it earns its
        place in Phase 5 by an episode-split ablation.
        """
        super().__init__()
        if (
            type(commercial_attention_heads) is not int
            or commercial_attention_heads < 1
            or n_commercial
            and d_model % commercial_attention_heads
        ):
            raise OursError("commercial attention heads must divide d_model")
        if n_event_pools is not None and (type(n_event_pools) is not int or n_event_pools < 1):
            raise OursError("declared event pools must be a positive integer")
        self.event_pool_required = n_event_pools is not None
        for name, value in (
            ("trunk_stages", trunk_stages),
            ("decoder_layers", decoder_layers),
            ("history_attention_layers", history_attention_layers),
            ("commercial_attention_layers", commercial_attention_layers),
            ("event_kernel_layers", event_kernel_layers),
        ):
            if type(value) is not int or value < 1:
                raise OursError(f"{name} must be a positive integer, got {value!r}")
        self.demand = DemandHistoryBranch(
            d_model,
            encoder_days,
            dropout=demand_dropout,
            weekly_patch=patch_size,
            n_heads=n_heads,
            attention_layers=history_attention_layers,
            n_covariates=n_history_covariates,
        )
        # `static_variable_selection`: which attribute this forecast leans on, per series. Absent
        # widths means the caller supplied no group structure, and selecting among groups nobody
        # declared would be selecting among columns -- a bias term wearing a different name.
        self.selection = StaticVariableSelection(attribute_widths) if attribute_widths else None
        # Which known-future column is price. None means this client has no price role.
        self.price_index = price_index
        # Availability is an EXPLICIT layout property, never inferred from tensor width.  The
        # production commercial layout is three values followed by their three flags; a raw
        # decoder-only fixture/checkpoint can legitimately have unrelated columns after price.
        # In that route ``None`` means every supplied price cell is numerically available.
        self.price_available_index = price_available_index
        if price_available_index is None and price_index is not None and n_commercial == 3:
            candidate = price_index + len(commercial.CHANNELS)
            if candidate >= n_known:
                raise OursError(
                    f"commercial price index {price_index} implies availability index "
                    f"{candidate}, outside n_known={n_known}"
                )
            self.price_available_index = candidate
        # Discount depth is DERIVED from price (1 - price/reference), so the price-monotonicity
        # counterfactual must move both together or it describes an impossible world -- the very
        # thing `intervention_bundles` exists to prevent. The availability column guards cells
        # where the depth is genuinely unknown: transforming their zero-filled placeholder would
        # fabricate a discount out of a missing value.
        self.discount_index = discount_index
        self.discount_available_index = discount_available_index
        self.promo_index = promo_index
        allowed_price_modes = {"none", "soft_penalty_v1", "monotone_raw_v1"}
        if price_response_mode not in allowed_price_modes:
            raise OursError(
                f"unsupported price_response_mode {price_response_mode!r}; expected one of "
                f"{sorted(allowed_price_modes)}"
            )
        self.price_response_mode = price_response_mode
        if price_response_mode == "monotone_raw_v1":
            required = {
                "price_index": price_index,
                "price_available_index": self.price_available_index,
                "discount_index": discount_index,
                "discount_available_index": discount_available_index,
            }
            if any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < n_known
                for value in required.values()
            ) or len(set(required.values())) != len(required):
                raise OursError(
                    "monotone_raw_v1 requires four distinct in-range price/depth value and "
                    f"availability indices, got {required} for n_known={n_known}"
                )
            for name, value in (
                ("price_occurrence_slope_max", price_occurrence_slope_max),
                ("price_size_slope_max", price_size_slope_max),
            ):
                if not is_finite_real(value) or value <= 0.0:
                    raise OursError(
                        f"{name} must be finite and positive for monotone_raw_v1, got " f"{value!r}"
                    )
            expected_commercial = 1 if promo_index is not None else 0
            if n_commercial != expected_commercial:
                raise OursError(
                    "monotone_raw_v1 builds only a live promo commercial branch: "
                    f"promo_index={promo_index!r} requires n_commercial={expected_commercial}, "
                    f"got {n_commercial}"
                )
        elif price_occurrence_slope_max is not None or price_size_slope_max is not None:
            raise OursError(
                "price slope caps are executable only for price_response_mode=" "'monotone_raw_v1'"
            )
        self.price_occurrence_slope_max = price_occurrence_slope_max
        self.price_size_slope_max = price_size_slope_max
        self.commercial_known_indices = (
            (None, None, promo_index)
            if price_response_mode == "monotone_raw_v1"
            else (price_index, discount_index, promo_index)
        )
        if bool(event_distance_index) != bool(event_present_index):
            raise OursError("event distance and presence indices must be configured together")
        if event_distance_index and len(event_distance_index) != n_events:
            raise OursError(
                f"{len(event_distance_index)} event distance indices for n_events={n_events}"
            )
        if len(event_present_index) != len(event_distance_index):
            raise OursError("event distance and presence index counts differ")
        self.event_distance_index = tuple(event_distance_index)
        self.event_present_index = tuple(event_present_index)
        # FACTORISED identity where the panel can tell products and stores apart: one table per
        # id, composed, so a pair that has never occurred still has a vector -- the per-series
        # fallback is the joint lookup `allow_joint_embedding=False` forbids on such a panel,
        # and it is kept only for panels whose series IS the product. The text embedding is a
        # frozen INPUT (precomputed upstream), projected here; frozen by construction.
        self.static = StaticIdentityBranch(
            n_attributes,
            d_model,
            text_dim=text_dim,
            residual_dim=embedding_residual_dim,
            n_series=0 if n_products or identity_cardinalities else n_series,
            n_products=n_products,
            n_stores=n_stores if n_products else 0,
            identity_cardinalities=identity_cardinalities,
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
        self.commercial_input_indices = (
            (2,)
            if price_response_mode == "monotone_raw_v1" and n_commercial
            else tuple(range(n_commercial))
        )
        # `event_pooling='partial'`: shared shape + per-event deviation ACROSS EVENTS, and a
        # shrunk per-(series, event) amplitude ACROSS THE CATALOGUE -- the second half was
        # missing, so pooling only ever ran across event types.
        self.calendar = (
            CalendarEventBranch(
                n_events,
                d_model,
                max_support=event_max_support,
                n_series=n_series if n_event_pools is None else n_event_pools,
                event_windows=event_windows,
                kernel_layers=event_kernel_layers,
            )
            if n_events
            else None
        )
        self.n_branches = 2 + int(self.commercial is not None) + int(self.calendar is not None)
        # The third leg of `static_conditioning`, alongside the trunk's FiLM and the head's
        # dispersion.
        self.fusion = GatedFusion(self.n_branches, d_model, d_static=n_attributes, n_heads=n_heads)
        self.trunk = Trunk(d_model, d_static=n_attributes, n_stages=trunk_stages, n_heads=n_heads)
        self.decoder = Decoder(
            d_model,
            n_known=n_known,
            max_horizon=max(horizon, 60),
            n_layers=decoder_layers,
            n_heads=n_heads,
        )
        # Dispersion is pooled from static attributes WITH a per-series residual (v3 spec
        # table: "linear, clamped, partially pooled ... weight-decayed per-series residual").
        # The residual rides the same decay group as the identity residual, for the same
        # reason: a series earns its own dispersion with evidence, and starts pooled.
        # WHICH head is a RESOLVED decision, not an adapter default: `resolve(zero_share=...)`
        # steps hurdle_negbin down to plain negbin where the measured zero rate does not justify
        # an occurrence component, and `ResolvedBuild` is how that decision reaches this
        # constructor. An unknown kind is refused -- a head silently defaulted is a spec field
        # silently ignored.
        if head_kind not in ("hurdle_negbin", "negbin"):
            raise OursError(
                f"unsupported head_kind {head_kind!r}; the resolved spec builds 'hurdle_negbin' "
                "or 'negbin'"
            )
        self.head_kind = head_kind
        self.head = (
            HurdleNegBinHead(d_model, n_static=n_attributes, n_series=n_series)
            if head_kind == "hurdle_negbin"
            else NegBinHead(d_model, n_static=n_attributes, n_series=n_series)
        )
        response_width = 2 if head_kind == "hurdle_negbin" else 1
        if aggregate_trajectory_mode not in ("none", AGGREGATE_TRAJECTORY_MODE):
            raise OursError(
                f"unsupported aggregate_trajectory_mode {aggregate_trajectory_mode!r}; expected "
                f"'none' or {AGGREGATE_TRAJECTORY_MODE!r}"
            )
        self.aggregate_trajectory_mode = aggregate_trajectory_mode
        projection = None
        if aggregate_trajectory_mode == AGGREGATE_TRAJECTORY_MODE:
            # A dormant zero module must not change R6's ordinary initialization relative to R5.
            # ``nn.Linear`` samples an initializer before we zero it, so construct it inside a
            # CPU RNG fork and restore the stream before any later price/factor/seasonal module.
            with torch.random.fork_rng(devices=[]):
                projection = nn.Linear(2, response_width, bias=False)
        self.aggregate_trajectory_projection = projection
        if self.aggregate_trajectory_projection is not None:
            # Rung 6 contains this dormant module. Both hybrid arms load those identical weights
            # and train only this matrix, so merely enabling the boundary cannot move its forecast.
            nn.init.zeros_(self.aggregate_trajectory_projection.weight)
        self.price_response = (
            nn.Linear(d_model, response_width) if price_response_mode == "monotone_raw_v1" else None
        )
        if self.price_response is not None:
            # Small but nonzero contextual weights let the response learn from the first update.
            # A negative initial bias starts close to a flat response without pinning the exact
            # zero boundary of a hard clamp. The v1 mode name binds this initialisation.
            nn.init.normal_(self.price_response.weight, std=1e-3)
            nn.init.constant_(self.price_response.bias, -4.0)
        # ``ForecastModelSpec.resolve`` makes rank 0 the explicit bottom-up degradation when no
        # factor grouping is declared.  Instantiating one rank-8 GLOBAL factor in that case is not
        # a conservative fallback: it couples every series to one shock and turns the complete-
        # block sampler into a full-catalogue batch.  Keep the absence structural instead.  The
        # store/product tables above remain available for factorised IDENTITY; they are unrelated
        # to whether a stochastic dependence factor exists.
        if factor_rank < 0:
            raise OursError(f"factor_rank must be non-negative, got {factor_rank}")
        if factor_rank > 0 and factor_dynamics not in ("ar1", "iid"):
            raise OursError(
                f"factor_dynamics must be 'ar1' or 'iid' for a factor model, got "
                f"{factor_dynamics!r}"
            )
        self.factor = (
            None
            if factor_rank == 0
            else SharedLatentFactor(factor_rank, n_stores, factor_dynamics=factor_dynamics)
        )
        # PER-SERIES ANNUAL AMPLITUDE, off by default. The annual basis reaches the decoder either
        # way; this decides whether its RESPONSE is pooled across series or allowed to vary, and
        # the variation is shrunk because a five-year panel gives about five Januaries per series.
        self.annual_index = annual_index
        self.seasonality = (
            SeasonalAmplitude(len(annual_index), n_series)
            if per_series_seasonality and annual_index
            else None
        )

    def describe(self) -> dict[str, object]:
        """The assembled architecture, component by component, as data.

        **The model describes ITSELF so the experiment record cannot drift from the code.** A
        config block written by hand alongside a run says what someone believed was built; this
        says what was built. They diverge the first time a default changes and nobody edits the
        note, and the divergence is invisible because both are plausible.

        Everything here is read off the constructed modules -- parameter counts from the modules
        themselves, branch presence from whether the attribute is None -- so adding a branch and
        forgetting to document it is not possible.
        """

        def count(module) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        branches = {
            "demand_history": {
                "present": True,
                "encoder_days": self.demand.encoder_days,
                "weekly_patch": self.demand.weekly_patch,
                "attention_layers": self.demand.attention_layers,
                "n_positions": self.demand.encoder_days
                + self.demand.encoder_days // self.demand.weekly_patch,
                "parameters": count(self.demand),
            },
            "static_identity": {
                "present": True,
                "n_attributes": self.static.n_attributes,
                "residual_dim": self.static.residual_dim,
                "residual": (
                    "factorised product+store"
                    if self.static.factorised
                    else ("per_series" if self.static.residual is not None else "none")
                ),
                "text_dim": self.static.text_dim,
                "parameters": count(self.static),
            },
            "commercial": {
                "present": self.commercial is not None,
                "n_channels": getattr(self.commercial, "n_channels", 0),
                "attention_layers": getattr(self.commercial, "attention_layers", None),
                "parameters": count(self.commercial),
            },
            "calendar_events": {
                "present": self.calendar is not None,
                "max_support": getattr(self.calendar, "max_support", None),
                "n_basis": getattr(self.calendar, "n_basis", None),
                "kernel_layers": getattr(self.calendar, "kernel_layers_count", None),
                "shrink": getattr(self.calendar, "shrink", None),
                "parameters": count(self.calendar),
            },
            # Refused by `first_build`; Phase 5 earns it by ablation.
            "peer_products": {"present": False, "why": "first_build refuses it; Phase 5"},
        }
        description = {
            "branches": branches,
            "n_branches_built": self.n_branches,
            "fusion": {"kind": "gated", "parameters": count(self.fusion)},
            "trunk": {"alternating_stages": self.trunk.n_stages, "parameters": count(self.trunk)},
            "decoder": {
                "n_known": self.decoder.known.in_features
                if hasattr(self.decoder, "known")
                else None,
                "max_horizon": self.decoder.max_horizon,
                "layers": len(self.decoder.layers),
                "parameters": count(self.decoder),
            },
            "head": {
                "kind": (
                    "hurdle zero-truncated negative binomial"
                    if self.head_kind == "hurdle_negbin"
                    else "plain negative binomial (resolved step-down)"
                ),
                "parameters": count(self.head),
            },
            "factor": (
                {
                    "present": True,
                    "kind": (
                        "shared latent AR(1)"
                        if self.factor.factor_dynamics == "ar1"
                        else "shared latent IID Gaussian"
                    ),
                    "dynamics": self.factor.factor_dynamics,
                    "topology": "bottom_up_factor",
                    "rank": self.factor.rank,
                    "n_stores": self.factor.n_stores,
                    "parameters": count(self.factor),
                }
                if self.factor is not None
                else {
                    "present": False,
                    "kind": "none",
                    "topology": "bottom_up",
                    "rank": 0,
                    "parameters": 0,
                    "dependence": "independent predictive marginals",
                }
            ),
            "seasonality": (
                {
                    "kind": "per-series annual amplitude, shared + shrunk deviation",
                    "n_terms": self.seasonality.n_terms,
                    "shrink": self.seasonality.shrink,
                    "parameters": count(self.seasonality),
                }
                if self.seasonality is not None
                else {"present": False, "why": "pooled annual response"}
            ),
            "annual_index": list(self.annual_index),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        structural = self.price_response_mode == "monotone_raw_v1"
        description["price_response"] = {
            "schema_version": 2,
            "mode": self.price_response_mode,
            "structurally_monotone": structural,
            "coordinate": "discount_depth_v1" if structural else "raw_price_and_depth_v1",
            "reference": "history_at_or_before_origin_frozen" if structural else None,
            "cross_horizon": "diagonal" if structural else "unrestricted",
            "head_route": self.head_kind,
            "occurrence_slope_max": self.price_occurrence_slope_max,
            "size_slope_max": self.price_size_slope_max,
            "dispersion_price_independent": structural,
            "factor_law_price_independent": structural,
            "parameters": count(self.price_response),
        }
        description["aggregate_trajectory"] = {
            "schema": AGGREGATE_TRAJECTORY_SCHEMA,
            "mode": self.aggregate_trajectory_mode,
            "grouping": "one_ancestor_channel" if self.aggregate_trajectory_projection else None,
            "input": ["mask_times_dimensionless_uplift", "mask"],
            "projection": "bias_free_linear_zero_initialised",
            "head_route": (
                "delta_size_only"
                if self.head_kind == "negbin"
                else "delta_occurrence_and_delta_size"
            ),
            "insertion": "after_linear_predictors_before_structural_price",
            "parameters": count(self.aggregate_trajectory_projection),
        }
        description["branches"]["commercial"]["input_channels"] = (
            ["promo"]
            if structural and self.commercial is not None
            else ([] if structural else list(commercial.CHANNELS))
        )
        return description

    def forward(
        self,
        history: torch.Tensor,
        observed: torch.Tensor,
        attributes: torch.Tensor,
        series_index: torch.Tensor,
        known_future: torch.Tensor,
        baseline_logit: torch.Tensor,
        baseline_log_size: torch.Tensor,
        group_index: torch.Tensor,
        store_index: torch.Tensor,
        n_draws: int,
        commercial: torch.Tensor | None = None,
        commercial_mask: torch.Tensor | None = None,
        event_distance: torch.Tensor | None = None,
        event_present: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        factor_path: torch.Tensor | None = None,
        censored: torch.Tensor | None = None,
        product_index: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        aggregate_trajectory: torch.Tensor | None = None,
        identity_indices: torch.Tensor | None = None,
        history_covariates: torch.Tensor | None = None,
        history_covariate_mask: torch.Tensor | None = None,
        event_pool_index: torch.Tensor | None = None,
        predictor_dtype: torch.dtype | None = None,
    ):
        """Return `(p, mu, k)` per draw and step, factor-conditioned or plain bottom-up.

        COMPOSED FROM THE TWO HALVES, deliberately: `encode_base` does the work that does not
        depend on the draw, `apply_factor` does the work that does. Callers that integrate over
        many realisations should call those two directly and reuse the base -- see
        `encode_base` for the measurement that motivated the split. This method keeps the old
        single-call signature so nothing that does not care has to change, and so there is
        exactly one implementation of each half.
        """
        base = self.encode_base(
            history=history,
            observed=observed,
            attributes=attributes,
            series_index=series_index,
            known_future=known_future,
            baseline_logit=baseline_logit,
            baseline_log_size=baseline_log_size,
            group_index=group_index,
            store_index=store_index,
            commercial=commercial,
            commercial_mask=commercial_mask,
            event_distance=event_distance,
            event_present=event_present,
            censored=censored,
            product_index=product_index,
            text=text,
            aggregate_trajectory=aggregate_trajectory,
            identity_indices=identity_indices,
            history_covariates=history_covariates,
            history_covariate_mask=history_covariate_mask,
            event_pool_index=event_pool_index,
        )
        return self.apply_factor(
            base,
            n_draws=n_draws,
            factor_path=factor_path,
            generator=generator,
            predictor_dtype=predictor_dtype,
        )

    def structural_price_coordinates(
        self,
        known_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and snapshot the sole live price coordinate and its fixed support mask."""
        if self.price_response is None:
            raise OursError("structural_price_coordinates is executable only for monotone_raw_v1")
        if not isinstance(known_future, torch.Tensor) or known_future.dim() != 3:
            raise OursError("structural price response requires rank-3 known_future inputs")
        price = known_future[..., self.price_index]
        depth = known_future[..., self.discount_index]
        price_flag = known_future[..., self.price_available_index]
        depth_flag = known_future[..., self.discount_available_index]
        for name, flag in (("price", price_flag), ("discount depth", depth_flag)):
            if not bool(torch.isfinite(flag).all()) or bool(((flag != 0.0) & (flag != 1.0)).any()):
                raise OursError(
                    f"structural {name} availability must contain exact finite 0/1 flags"
                )
        price_available = price_flag > 0.5
        depth_available = depth_flag > 0.5
        if bool((price_available & (~torch.isfinite(price) | (price < 0.0))).any()):
            raise OursError(
                "structural price values marked available must be finite and non-negative"
            )
        if bool((depth_available & ~torch.isfinite(depth)).any()):
            raise OursError("structural discount depths marked available must be finite")
        response_mask = (price_available & depth_available).clone()
        # ``where`` rather than multiplication: NaN * 0 is NaN, and an unavailable filler must
        # have exactly zero influence regardless of its numeric contents.
        active_depth = torch.where(response_mask, depth, torch.zeros_like(depth)).clone()
        return active_depth, response_mask

    def encode_base(
        self,
        history: torch.Tensor,
        observed: torch.Tensor,
        attributes: torch.Tensor,
        series_index: torch.Tensor,
        known_future: torch.Tensor,
        baseline_logit: torch.Tensor,
        baseline_log_size: torch.Tensor,
        group_index: torch.Tensor,
        store_index: torch.Tensor,
        commercial: torch.Tensor | None = None,
        commercial_mask: torch.Tensor | None = None,
        event_distance: torch.Tensor | None = None,
        event_present: torch.Tensor | None = None,
        censored: torch.Tensor | None = None,
        product_index: torch.Tensor | None = None,
        text: torch.Tensor | None = None,
        aggregate_trajectory: torch.Tensor | None = None,
        identity_indices: torch.Tensor | None = None,
        history_covariates: torch.Tensor | None = None,
        history_covariate_mask: torch.Tensor | None = None,
        event_pool_index: torch.Tensor | None = None,
    ) -> BasePrediction:
        """Everything that does not depend on the factor draw, computed ONCE.

        WHY THIS EXISTS. The 2026-08-25 review measured what the integration loop was actually
        doing: at 1,000 factor realisations `mean_over_factor` ran this entire stack -- four
        branch encoders, gated fusion, the two-stage trunk, the two-layer decoder -- 40 times,
        and `draw_paths_over_factor` ran it 1,000 times, to produce values IDENTICAL in every
        one of them. Measured on the killed rung at the real 30,490-cell shape: the
        deterministic stack costs 12.81s and one factor draw costs 0.385s -- 33 to 1.

        The split preserves the model algebra exactly, and the current `forward` and split entry
        points are bit-identical because `forward` composes them. Relative to the pre-split
        implementation, the production one-draw path call is also bit-identical. Do not broaden
        that into a universal historical bit claim: the old head projected tensors after expanding
        the draw axis, and its own GEMM can move by roughly one float32 ulp when that batch axis
        changes. Caching deliberately removes that repeated projection. Inference runs under
        `evaluating(model)`, so dropout is off and this half is deterministic in its inputs.

        In the historical soft mode, the commercial and calendar branches are evaluated at the
        origin and horizon price also reaches the generic decoder. In ``monotone_raw_v1``, raw
        price and discount depth are excluded from both unrestricted routes: a promo-only branch
        may remain, while the price-free decoder context supplies bounded non-negative slopes to
        the head predictors and each day receives only its own discount-depth coordinate.
        """
        selection_weights = None
        aggregate_trajectory = self._validated_aggregate_trajectory(
            aggregate_trajectory, known_future
        )
        if self.selection is not None:
            attributes, selection_weights = self.selection(attributes)
        price_depth = None
        price_response_mask = None
        if self.price_response is not None:
            price_depth, price_response_mask = self.structural_price_coordinates(known_future)
        demand_token, demand_sequence = self.demand(
            history,
            observed,
            censored=censored,
            covariates=history_covariates,
            covariate_mask=history_covariate_mask,
        )
        branch_tokens = [
            demand_token,
            self.static(
                attributes,
                text=text,
                series_index=series_index,
                product_index=product_index,
                store_index=store_index,
                identity_indices=identity_indices,
            ),
        ]
        if self.commercial is not None:
            if commercial is None:
                raise OursError(
                    "a commercial branch was built but no commercial features were passed. A "
                    "zero tensor would teach 'this product is free' rather than 'we do not know'"
                )
            if commercial_mask is not None and commercial_mask.shape != commercial.shape:
                raise OursError(
                    f"commercial_mask {tuple(commercial_mask.shape)} does not match "
                    f"commercial values {tuple(commercial.shape)}"
                )
            if commercial_mask is not None:
                commercial_mask = _binary_mask(
                    commercial_mask,
                    commercial,
                    "commercial availability mask",
                    error_type=OursError,
                )
            # Keep the semantic availability mask true. Multihead attention cannot consume an
            # all-masked row, so use a temporary safe key only for that operation and then replace
            # its output with the explicit all-missing TOKEN (zero before fusion adds branch
            # identity). This is not a zero price: a genuinely available free-price row goes
            # through CommercialBranch and produces its learned channel token. It also keeps the
            # real mask available to scenario routing, which must never update a placeholder.
            branch_commercial = commercial[..., list(self.commercial_input_indices)]
            branch_mask = (
                None
                if commercial_mask is None
                else commercial_mask[..., list(self.commercial_input_indices)]
            )
            all_missing = None
            attention_mask = branch_mask
            if commercial_mask is not None:
                all_missing = ~branch_mask.any(dim=-1)
                if bool(all_missing.any()):
                    attention_mask = branch_mask.clone()
                    attention_mask[all_missing, 0] = True
            commercial_token = self.commercial(branch_commercial, attention_mask)
            if all_missing is not None and bool(all_missing.any()):
                commercial_token = torch.where(
                    all_missing.unsqueeze(-1), torch.zeros_like(commercial_token), commercial_token
                )
            branch_tokens.append(commercial_token)
        calendar_sequence = None
        if self.calendar is not None:
            # In the assembled model, event drivers remain named known-future inputs so scenario
            # callers can set them, but they reach the prediction only through the learned
            # distributed-lag kernel. The generic decoder receives zero in those raw columns
            # below, preventing it from learning an unconstrained distance effect in parallel.
            if self.event_distance_index:
                event_distance = known_future[..., list(self.event_distance_index)]
                event_present = known_future[..., list(self.event_present_index)] > 0.5
            if event_distance is None:
                raise OursError(
                    "a calendar branch was built but no event distances were passed. Zero is "
                    "'the event is today', not 'there is no event'"
                )
            if self.event_pool_required and event_pool_index is None:
                raise OursError("event_pool_index is required by the declared event pool")
            if not self.event_pool_required and event_pool_index is not None:
                raise OursError("event_pool_index supplied without a declared event pool")
            calendar_encoded = self.calendar(
                event_distance,
                event_present,
                series_index=event_pool_index if self.event_pool_required else series_index,
            )
            if calendar_encoded.dim() == 3:
                calendar_sequence = calendar_encoded
                branch_tokens.append(calendar_encoded[:, 0])
            else:
                branch_tokens.append(calendar_encoded)
        tokens = torch.stack(branch_tokens, dim=1)
        fused = self.fusion(tokens, static=attributes)  # [cells, n_branches, d]
        horizon = known_future.shape[1]
        # THE TRUNK'S TIME AXIS IS THE ENCODER'S WEEKLY PATCH SEQUENCE, not the horizon. An
        # earlier version tiled the fused tokens across the horizon, and attention over
        # identical copies is a fixed average -- verified: the trunk's output was bit-identical
        # at every step, so the "time" half of every stage was arithmetic that could not affect
        # the output, and no time structure of any kind passed through the trunk. Now the
        # demand row carries its own weeks (the v3 table's "52 timesteps"), the other branches
        # are carried across them, the time stages relate week to week, and the group stages
        # relate price and identity to each week -- which is the interleaving the design says
        # the trunk is FOR. The decoder's memory inherits one informative position per
        # (branch, week) instead of horizon-many copies of one vector.
        t_enc = demand_sequence.shape[1]
        carried = fused.unsqueeze(2).expand(-1, -1, t_enc, -1).contiguous()
        carried[:, 0] = carried[:, 0] + demand_sequence
        trunked = self.trunk(carried, attributes)
        memory = trunked.reshape(trunked.shape[0], -1, trunked.shape[-1])
        decoder_excluded = list(self.event_distance_index) + list(self.event_present_index)
        if self.price_response is not None:
            decoder_excluded.extend((self.price_index, self.discount_index))
        decoder_known = known_future
        if decoder_excluded:
            decoder_known = known_future.clone()
            decoder_known[..., decoder_excluded] = 0.0
        hidden = self.decoder(memory, decoder_known)  # [cells, horizon, d]
        if calendar_sequence is not None:
            hidden = hidden + calendar_sequence

        # THE DETERMINISTIC BASE ENDS HERE. Nothing above this line depends on `n_draws` or on a
        # factor realisation, which is why `encode_base` can compute it once and `apply_factor`
        # can be called many times against it. `forward` composes the two so there is exactly one
        # code path; see `encode_base` for what that changed and why.
        horizon = known_future.shape[1]
        seasonal = None
        if self.seasonality is not None:
            annual = known_future[..., list(self.annual_index)]  # [cells, horizon, n_terms]
            seasonal = self.seasonality(annual, series_index)
        # Snapshot every caller-owned tensor that the factor/head half still reads. A frozen
        # dataclass prevents field reassignment, not mutation through an aliased input tensor.
        # Without these clones, changing a scenario/input dictionary after encoding silently
        # changed the supposedly cached computation (baselines, selected/static attributes and
        # even which loading row a cell used). Computed tensors such as ``hidden`` and
        # ``seasonal`` already own their storage.
        dense_group_index, group_labels = _dense_group_index(group_index)
        return BasePrediction(
            hidden=hidden,
            attributes=attributes.clone(),
            baseline_logit=baseline_logit.clone(),
            baseline_log_size=baseline_log_size.clone(),
            series_index=series_index.clone(),
            group_index=group_index.clone(),
            dense_group_index=dense_group_index.clone(),
            group_labels=group_labels.clone(),
            store_index=store_index.clone(),
            horizon=horizon,
            price_depth=price_depth,
            price_response_mask=price_response_mask,
            aggregate_trajectory=aggregate_trajectory,
            seasonal=seasonal,
            selection_weights=selection_weights,
            _model_identity=id(self),
            _model_state=_model_state_stamp(self),
            _context=_execution_context(self),
        )

    def _validated_aggregate_trajectory(
        self,
        value: torch.Tensor | None,
        known_future: torch.Tensor,
    ) -> torch.Tensor | None:
        """Validate and snapshot ``[mask * uplift, mask]`` without any price dependency."""
        if value is None:
            return None
        if self.aggregate_trajectory_projection is None:
            raise OursError(
                "aggregate_trajectory was supplied but this model has no aggregate projection"
            )
        expected = (*known_future.shape[:2], 2)
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or tuple(value.shape) != expected
        ):
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise OursError(
                f"aggregate_trajectory must be a floating tensor shaped {expected}, got {shape}"
            )
        if value.device != known_future.device or value.dtype != known_future.dtype:
            raise OursError(
                "aggregate_trajectory must already share known_future device and dtype; implicit "
                "conversion would weaken the sealed feature contract"
            )
        if not bool(torch.isfinite(value).all()):
            raise OursError("aggregate_trajectory contains a non-finite uplift or mask")
        mask = value[..., 1]
        if bool(((mask != 0.0) & (mask != 1.0)).any()):
            raise OursError("aggregate_trajectory mask must contain exact 0/1 values")
        if bool((value[..., 0][mask == 0.0] != 0.0).any()):
            raise OursError(
                "aggregate_trajectory first channel must be exactly zero wherever mask is zero"
            )
        return value.clone()

    def apply_factor(
        self,
        base: BasePrediction,
        *,
        n_draws: int,
        factor_path: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        predictor_dtype: torch.dtype | None = None,
    ):
        """Draw (or accept) a factor path and turn a cached base into `(p, mu, k)`.

        The per-draw half. Everything here is elementwise on `[draws, cells, horizon]` or a
        gather on the small loading tables -- no encoder, no attention, no projection of
        `d_model`-wide tensors.
        """
        if not isinstance(base, BasePrediction):
            raise OursError(f"base must be BasePrediction, got {type(base).__name__}")
        base.validate_for(self)
        group_index, store_index = base.dense_group_index, base.store_index
        horizon = base.horizon
        if self.factor is None:
            # There is no latent variable to integrate over.  Preserve the leading draw axis for
            # callers shared with the factor model, but every slice contains the same marginal
            # parameters and the count sampler supplies independent uniforms per cell.  A
            # canonical empty path lets the scenario API reuse one object in both arms without
            # pretending a shared random effect exists.
            if factor_path is not None and tuple(factor_path.shape) != (n_draws, 0, horizon, 0):
                raise OursError(
                    f"factor_path is {tuple(factor_path.shape)} but this is a factorless "
                    f"bottom_up model; expected the empty sentinel [{n_draws}, 0, {horizon}, 0]"
                )
            occ = base.hidden.new_zeros((n_draws, base.n_cells, horizon))
            size = base.hidden.new_zeros((n_draws, base.n_cells, horizon))
        else:
            # Group labels are identifiers, not array offsets. A subset can legitimately carry
            # global ids such as [0, 100000]; allocating every absent path between them turns a
            # small query into a 100001-group draw. Dense-remap locally while preserving equality.
            # The sorted original labels solely align an explicitly supplied global path.
            group_labels = base.group_labels
            n_groups = len(group_labels)
            if factor_path is None:
                path = self.factor.draw(n_groups, horizon, n_draws, generator=generator)
            else:
                # A SCENARIO CONTRAST MUST REUSE THE FACTUAL RUN'S DRAWS. Re-drawing per scenario
                # injects Monte Carlo noise into a difference of two means, and the difference is
                # usually smaller than the noise -- so the contrast measures the sampler. Passing
                # the path in makes that structural rather than dependent on seeding both runs.
                if (
                    factor_path.ndim != 4
                    or factor_path.shape[0] != n_draws
                    or factor_path.shape[2] != horizon
                    or factor_path.shape[3] != self.factor.rank
                ):
                    raise OursError(
                        f"factor_path is {tuple(factor_path.shape)}, expected "
                        f"[{n_draws}, groups, {horizon}, {self.factor.rank}] for this "
                        f"rank-{self.factor.rank} factor"
                    )
                if factor_path.shape[1] == n_groups:
                    # Already local, in sorted-label order. This is what scenario queries and the
                    # monotonicity counterfactual pass.
                    path = factor_path
                elif int(group_labels.max()) < factor_path.shape[1]:
                    # A path over the caller's global id space. Select only labels this query owns;
                    # the contracted local index below is then [0, n_present).
                    path = factor_path.index_select(1, group_labels)
                else:
                    raise OursError(
                        f"factor_path is {tuple(factor_path.shape)} but group labels are "
                        f"{group_labels.tolist()}; pass one path per present group in sorted-label "
                        "order, or a global path covering the largest label"
                    )
            occ, size = self.factor.contract(path, group_index, store_index)
        # Preserve the monolithic forward's exact float32 order: it formed factor + seasonality,
        # then handed that combined offset to the head. Moving the deterministic seasonal term
        # onto the cached eta first is algebraically equivalent but not bit-identical. Quadrature
        # performs the same composition at every integration node.
        if base.seasonal is not None:
            occ = occ + base.seasonal[0].unsqueeze(0)
            size = size + base.seasonal[1].unsqueeze(0)

        # THE DETERMINISTIC PREDICTORS ARE COMPUTED ONCE, at [cells, horizon], and only the
        # ELEMENTWISE factor addition is repeated per draw. The previous code expanded `hidden`
        # to [draws, cells, horizon, d_model] and re-ran the head's occurrence, size and
        # dispersion projections over all of it -- a thousand identical linear maps at 1,000
        # draws, which is the other half of what made integration expensive.
        eta = base.predictors(self, _validated=True)
        if predictor_dtype is not None:
            # Promotion must precede exp: promoting a float32 mean afterwards can leave
            # an infinite intermediate mean gradient before exp cancels it mathematically.
            eta = tuple(value.to(predictor_dtype) for value in eta)
            occ, size = occ.to(predictor_dtype), size.to(predictor_dtype)
        if self.head_kind == "negbin":
            # No occurrence component: the factor's size loading and the size baseline are the
            # whole linear predictor, and `p` is None by construction rather than a fake 1.0 --
            # a probability that exists but is ignored is how heads drift apart silently.
            eta_size, log_k = eta
            mu, k = self.head.from_predictors(eta_size.unsqueeze(0), log_k.unsqueeze(0), size)
            return None, mu, k.expand_as(mu)
        eta_occ, eta_size, log_k = eta
        p_out, mu, k = self.head.from_predictors(
            eta_occ.unsqueeze(0), eta_size.unsqueeze(0), log_k.unsqueeze(0), occ, size
        )
        return p_out, mu, k.expand_as(mu)

    def draw_factor_paths(
        self, n_groups: int, horizon: int, n_draws: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Draw the shared path, or return the canonical empty path for bottom-up marginals.

        Scenario code needs one object it can pass unchanged through its factual and
        counterfactual arms.  For a factorless model that object has shape
        ``[draws, 0, horizon, 0]``: it carries no random coordinates and cannot induce common
        movement between cells.
        """
        if min(n_groups, horizon, n_draws) < 1:
            raise OursError(
                f"n_groups={n_groups}, horizon={horizon}, n_draws={n_draws}: all must be >= 1"
            )
        if self.factor is not None:
            return self.factor.draw(n_groups, horizon, n_draws, generator=generator)
        reference = next(self.parameters())
        return torch.empty((n_draws, 0, horizon, 0), device=reference.device, dtype=reference.dtype)

    def expected_demand(self, **kw) -> torch.Tensor:
        """`E[Y]` per cell and step, averaged over the drawn factor paths. For the price
        constraint, which needs a differentiable output rather than sample paths."""
        p, mu, k = self.forward(**kw)
        return predicted_mean(p, mu, k).mean(dim=0)

    def price_penalty(self, kw: dict, *, bump: float = 0.1) -> tuple[torch.Tensor, float]:
        """Penalty/rate over finite positive available-price cells. `(0, 0.0)` with no role.

        The counterfactual raises the price by `bump` as a FRACTION of its own value and holds
        everything else -- including the drawn factor path AND the dropout state -- fixed, so
        the difference is the price's effect and not a re-draw. Prices differ by three orders
        of magnitude across a catalogue, which is why the bump is relative rather than absolute.

        Both arms run in EVAL mode (gradients still flow; only dropout is switched off). In
        train mode the two forward passes draw independent dropout masks, so the hinge measured
        price plus dropout noise -- at bump=0 the two arms already disagreed, and that noise
        floor is exactly the small systematic violation the penalty exists to price.

        The discount-depth column moves WITH the price: depth is 1 - price/reference, so a
        raised price means depth' = (1+b)*depth - b. Holding depth fixed while price rises
        describes a world that cannot exist, which is the off-manifold evaluation the
        intervention-bundle rule forbids -- and it let the model satisfy monotonicity through
        the depth channel while the raw-price channel misbehaved. Cells whose depth is
        unavailable keep their zero placeholder: transforming it would fabricate a discount
        out of a missing value.
        """
        if self.price_index is None:
            return next(self.parameters()).new_zeros(()), 0.0
        eligible = _price_intervention_mask(kw, self.price_index, self.price_available_index)
        if not bool(eligible.any()):
            # No numerical constraint was observed. NaN suppresses the progress-line percentage;
            # reporting 0% would claim a successful audit of zero cells.
            return next(self.parameters()).new_zeros(()), float("nan")
        # The penalty normally receives a minibatch cut from global tensors. Densify before
        # drawing its shared path so a sparse global label cannot allocate absent groups, and so
        # both counterfactual arms use exactly the same local label ordering.
        kw = dict(kw)
        kw["group_index"], group_labels = _dense_group_index(kw["group_index"])
        known = kw["known_future"]
        # One factor path, shared by both arms: re-drawing would let sampler noise dominate a
        # difference that is usually smaller than it -- the same trap the scenario contrast avoids.
        n_groups = len(group_labels)
        path = self.draw_factor_paths(n_groups, known.shape[1], kw["n_draws"])
        raised = _raise_price_inputs(
            kw,
            bump,
            self.price_index,
            self.discount_index,
            self.discount_available_index,
            self.price_available_index,
        )
        # Preserve every child-specific train/eval flag. Remembering only the root and restoring
        # with ``self.train(was_training)`` recursively flattened a mixed tree, just like the
        # inference context-manager bug fixed below.
        with evaluating(self):
            low = self.expected_demand(**{**kw, "factor_path": path})
            high = self.expected_demand(**{**raised, "factor_path": path})
        return (
            monotonicity_penalty(low[eligible], high[eligible]),
            price_violation_rate(low[eligible], high[eligible]),
        )

    def loss(
        self,
        y: torch.Tensor,
        state: torch.Tensor,
        *,
        normalization_count: float | None = None,
        state_probs: torch.Tensor | None = None,
        likelihood_dtype: torch.dtype | None = None,
        **kw,
    ) -> torch.Tensor:
        """Negative IWAE bound per factor block, or exact marginal NLL when factorless.

        The factor is latent and the likelihood is not conjugate to it, so the objective is a
        BOUND, and which bound is part of the contract (`likelihood.iwae_bound`):

            L = sum_blocks log( (1/S) * sum_s exp( sum_{cells in block} log p(y | f^(s)) ) )

        A BLOCK is the set of cells sharing one drawn path -- one `group_index` value, which is
        one `(factor_group, product)` combination. The contract is explicit that the inner sum
        runs over exactly that unit: smaller treats stores as independent and throws away the
        dependence the factor exists to create; larger couples groups that share no factor.

        An earlier version summed the WHOLE BATCH inside the exponential. Still a valid bound
        over the joint latent, but with ~57k cells per batch the per-draw totals differ by
        hundreds of nats, `logsumexp` degenerates to `max`, and the objective becomes "the best
        of S joint draws" -- a much looser bound whose gradient flows through one draw. Blocks
        are independent given their own paths, so the tight form factorises per block, and that
        is what is computed here.

        Computed with `torch.logsumexp`/`index_add_` rather than the contract's pure-Python
        version because this one has to be DIFFERENTIABLE -- the contract's takes floats and is
        the reference this form is tested against, not something to call in a training loop.

        By default this is divided by the batch's informative-cell count. Training may instead
        pass `normalization_count=global_informative/n_batches`: factor blocks are indivisible,
        so packed batches have unequal sizes, and locally normalising every one would give a
        one-cell batch the same optimiser influence as a ten-cell batch. The fixed epoch
        denominator preserves the declared full-cell objective while retaining complete blocks.

        With `factor_rank=0` there is no latent integral and therefore no IWAE approximation:
        group labels are ignored and the ordinary sum of per-cell log likelihoods is returned.
        """
        from forecast.models.mft.heads import (
            hurdle_log_likelihood,
            hurdle_log_likelihood_soft,
            informative_mask,
            negbin_log_likelihood,
            negbin_log_likelihood_soft,
            soft_informative_weight,
        )

        p, mu, k = (
            self.forward(**kw)
            if likelihood_dtype is None
            else self.forward(**kw, predictor_dtype=likelihood_dtype)
        )
        if likelihood_dtype is not None:
            # Revisions use float64 log-gamma differences across their accepted count range.
            # Only likelihood tensors are promoted; the neural backbone and its gradients
            # retain their parameter dtype. The legacy default remains unchanged.
            y, mu, k = y.to(likelihood_dtype), mu.to(likelihood_dtype), k.to(likelihood_dtype)
            p = p.to(likelihood_dtype) if p is not None else None
            if (mu < torch.finfo(mu.dtype).tiny ** 0.5).any():
                raise OursError("mean is below the numerically supported likelihood range")
        y_b = y.unsqueeze(0).expand_as(mu) if y.dim() == 2 else y
        s_b = state.unsqueeze(0).expand_as(mu) if state.dim() == 2 else state
        if state_probs is None:
            ll = (
                negbin_log_likelihood(y_b, mu, k, s_b)
                if p is None
                else hurdle_log_likelihood(y_b, p, mu, k, s_b)
            )
            mask = informative_mask(y_b, s_b).to(ll.dtype)
            weighted_ll = ll * mask
            informative = float(mask[0].sum())
        else:
            # SOFT STATES: the likelihood marginalises over the state distribution, so the
            # information content is already INSIDE the mixture -- a fully-unobserved cell
            # contributes log(1) = 0 by construction, and multiplying by a mask again would
            # double-count. The mask generalises to the expected informative weight, used for
            # the denominator only.
            probs_b = (
                state_probs.unsqueeze(0).expand(*mu.shape, 3)
                if state_probs.dim() == 3
                else state_probs
            )
            weighted_ll = (
                negbin_log_likelihood_soft(y_b, mu, k, probs_b)
                if p is None
                else hurdle_log_likelihood_soft(y_b, p, mu, k, probs_b)
            )
            informative = float(soft_informative_weight(y_b, probs_b)[0].sum())
        if informative == 0:
            raise OursError("no informative cell in this batch; the bound would be over nothing")
        denominator = float(informative if normalization_count is None else normalization_count)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise OursError(
                f"normalization_count must be finite and positive, got {normalization_count!r}"
            )
        n_draws = mu.shape[0]
        if self.factor is None:
            # All draw slices are identical because no latent variable exists.  Score the plain
            # product of independent marginals directly; group labels are identity/batching
            # metadata only in this topology and must not alter the likelihood.
            return -weighted_ll[0].sum() / denominator
        group_index, _ = _dense_group_index(kw["group_index"])
        per_cell = weighted_ll.sum(dim=-1)  # [S, cells] over the horizon
        n_blocks = int(group_index.max()) + 1
        block_sums = per_cell.new_zeros(n_draws, n_blocks)
        block_sums.index_add_(1, group_index, per_cell)
        present = torch.zeros(n_blocks, dtype=torch.bool, device=per_cell.device)
        present[group_index] = True
        bound = (torch.logsumexp(block_sums[:, present], dim=0) - math.log(n_draws)).sum()
        return -bound / denominator


def revision_data_spec(panel, horizon, config):
    """Translate declared Panel roles, not dataset names; tested by ``test_generic_panel_read``."""
    from spec.forecast_spec import ForecastDataSpec, ObservationSpec

    identities = config.identity_columns or (
        (panel.product_col, panel.store_col)
        if panel.product_col and panel.store_col
        else (panel.series_col,)
    )
    return ForecastDataSpec(
        time_idx=panel.time_col,
        target=panel.target_col,
        group_ids=identities,
        target_estimand=panel.target_estimand,
        freq=panel.frequency,
        max_encoder_length=config.encoder_days,
        max_prediction_length=horizon,
        observation=ObservationSpec(status_col=panel.exposure_col),
        price_col=panel.price_col,
        promo_col=panel.promo_col,
        factor_group_col=panel.factor_group
        or (config.factor_columns[0] if config.factor_columns else None),
        hierarchy=config.factor_columns or ((panel.factor_group,) if panel.factor_group else ()),
        static_categoricals=panel.static,
        static_reals=panel.static_reals,
        time_varying_known_reals=panel.known_future,
    )


def _revision_calendar_contract(table):
    """Strictly decode the versioned calendar metadata shared by discovery and assembly."""
    from autoforecast._exact_json import loads_exact_json

    raw = (table.schema.metadata or {}).get(b"revision_calendar_v1")
    if raw is None:
        return None
    contract = loads_exact_json(raw)
    if not isinstance(contract, dict):
        raise ValueError("revision calendar contract must be a JSON object")
    return contract


def _revision_event_records(rows, contract, variant, selection_end):
    """Freeze event channels, not later occurrences; tests cover as-of and disabled roles."""
    selected = contract["variants"].get(variant, {}).get("events")
    records = {}
    for row in rows:
        day = row[contract["time_col"]]
        scope = row[contract["group_col"]] if contract["group_col"] else ""
        known = row[contract["known_at_col"]] if contract["known_at_col"] else dt.date.min
        if (
            type(day) is not dt.date
            or type(known) is not dt.date
            or not isinstance(scope, str)
            or not scope.strip()
            and contract["group_col"]
        ):
            raise ValueError("calendar dates/knowledge dates and scopes must be typed")
        for name_col, type_col in contract["event_pairs"]:
            name, kind = row[name_col], row[type_col] if type_col else None
            if (type_col and ((name is None) != (kind is None))) or any(
                v is not None and (not isinstance(v, str) or not v.strip()) for v in (name, kind)
            ):
                raise ValueError("calendar event identity and type must be paired")
            if name:
                episode = (
                    row[contract["episode_col"]] if contract["episode_col"] else day.isoformat()
                )
                if not isinstance(episode, str) or not episode.strip():
                    raise ValueError("event episodes must be nonempty identifiers")
                for label in [f"name:{name}", *([f"type:{kind}"] if kind else [])]:
                    if selected is None or label in selected:
                        records.setdefault(label, set()).add((day, scope, known, episode))
    records = {
        name: occurrences
        for name, occurrences in records.items()
        if any(known <= selection_end for _, _, known, _ in occurrences)
    }
    if selected is not None and set(records) != set(selected):
        raise ValueError("calendar does not cover the explicitly selected events")
    return records


class RevisionFeatures:
    """Resolved, origin-bounded inputs for both variants; see ``test_revision_portability``.

    Calendar column roles live in the input's versioned metadata. A disabled input family
    has no module or upstream data requirement; per-cell missingness is a separate mask.
    """

    def __init__(self, panel, data, config, calendar_path=None, data_spec=None) -> None:
        import dataclasses

        import numpy as np
        import pyarrow.parquet as pq
        from spec.forecast_spec import ForecastDataSpec, ForecastModelSpec, ObservationSpec

        from autoforecast.rolling_training import rolling_origins

        self.panel, self.data, self.config = panel, data, config
        self.spec = dataclasses.replace(
            data_spec or revision_data_spec(panel, data.horizon, config)
        )
        s, c = self.spec, config
        defaults = ForecastDataSpec(s.time_idx, s.target, s.group_ids)
        declared_calendar_roles = bool(
            s.events or s.event_known_at_col or s.event_episode_col or s.events_preannounced
        )
        unsupported = (
            "exposure",
            "lead_time_plus_review",
            "allow_joint_embedding",
            "aggregate_level",
            "event_windows",
        )
        if any(getattr(s, name) != getattr(defaults, name) for name in unsupported):
            raise ValueError(
                "revision does not implement exposure, joint identity, lead-time, "
                "aggregate-level or per-event-window spec overrides"
            )
        observation_defaults = ObservationSpec(status_col=s.observation.status_col)
        if (
            s.observation.quality_gate != observation_defaults.quality_gate
            or s.observation.on_gate_failure != observation_defaults.on_gate_failure
        ):
            raise ValueError("revision does not implement observation quality-policy overrides")
        if (
            s.time_idx,
            s.target,
            s.freq,
            s.max_encoder_length,
            s.max_prediction_length,
            s.price_col,
            s.promo_col,
            s.target_estimand,
        ) != (
            panel.time_col,
            panel.target_col,
            panel.frequency,
            c.encoder_days,
            data.horizon,
            panel.price_col,
            panel.promo_col,
            panel.target_estimand,
        ):
            raise ValueError("revision data spec disagrees with Panel roles or temporal dimensions")
        if (
            s.text_features
            or s.peer_graph
            or s.branches
            or s.oos_col
            or s.observation.status_prob_cols
            or s.observation.status_col != panel.exposure_col
        ):
            raise ValueError(
                "revision does not implement text, peers, custom branches or soft states"
            )
        if (
            set(s.time_varying_known_reals) | set(s.time_varying_known_categoricals)
            != set(panel.known_future)
            or tuple(s.static_categoricals) != panel.static
            or tuple(s.static_reals) != panel.static_reals
        ):
            raise ValueError("Panel and data spec must declare the same static/known-future roles")
        self.cache, self.cache_origin = None, None
        self.numeric_stats, self.numeric_stats_origin = {}, None
        self.adaptation_reasons = {}
        self.vocab = {name: sorted(set(values)) for name, values in data.metadata.items()}
        self.indices = {}
        for name, values in data.metadata.items():
            mapping = {value: i for i, value in enumerate(self.vocab[name])}
            self.indices[name] = np.array([mapping[value] for value in values], np.int64)
        # Static attributes are already available on the first panel date; targets are not used.
        parts, widths = [], []
        for name in s.static_categoricals:
            width = len(self.vocab[name])
            if width > MAX_ONE_HOT:
                raise ValueError("static categorical vocabulary exceeds MAX_ONE_HOT")
            parts.append(np.eye(width, dtype=np.float32)[self.indices[name]])
            widths.append(width)
        self.static_stats = {}
        for name in s.static_reals:
            value = data.static_reals[name]
            mean, scale = float(value.mean()), max(float(value.std()), 1e-6)
            parts.append(((value - mean) / scale).astype(np.float32)[:, None])
            widths.append(1)
            self.static_stats[name] = (mean, scale)
        self.attributes = (
            np.concatenate(parts, 1) if parts else np.empty((len(data.series_ids), 0), np.float32)
        )
        self.attribute_widths = tuple(widths)
        self.identity_columns = tuple(c.identity_columns or s.group_ids)
        if not self.identity_columns or len(set(self.identity_columns)) != len(
            self.identity_columns
        ):
            raise ValueError("revision needs unique declared series identity columns")
        self.identity_indices = np.stack([self._indices(name) for name in self.identity_columns], 1)
        self.identity_cardinalities = tuple(
            int(self.identity_indices[:, j].max()) + 1 for j in range(len(self.identity_columns))
        )
        self.baseline_columns = c.baseline_columns or (
            (panel.category_col,) if panel.category_col else ()
        )
        self.baseline_index = self._groups(self.baseline_columns, absent=-1)
        self.aggregate_columns = c.aggregate_columns if c.aggregate_history else ()
        self.aggregate_index = (
            self._groups(self.aggregate_columns) if self.aggregate_columns else None
        )
        if c.aggregate_history and not self.aggregate_columns:
            self.adaptation_reasons["aggregate_history"] = "disabled: no aggregate_columns declared"
        self.event_pool_columns = ()
        self.event_pool_index = None
        self.calendar_contract, self.calendar_rows, self.calendar_features = {}, {}, ()
        self.calendar_scopes, self.scope_index = ("",), np.zeros(len(data.series_ids), np.int64)
        self.events, self.event_records = {}, {}
        selection_end = data.date_at(data.target.shape[1] - 2 * data.horizon - 1).isoformat()
        training_origins = rolling_origins(data, selection_end, c)
        if calendar_path is not None:
            table = pq.read_table(calendar_path)
            contract = _revision_calendar_contract(table)
            if contract is None:
                raise ValueError(
                    "revision calendar requires a versioned column/availability contract"
                )
            self.calendar_contract = contract
            required_keys = {
                "time_col",
                "group_col",
                "group_column",
                "known_at_col",
                "preannounced",
                "complete_grid",
                "event_pairs",
                "known_future",
                "variants",
                "binary_columns",
                "episode_col",
            }
            if not isinstance(contract, dict) or set(contract) != required_keys:
                raise ValueError("unexpected revision calendar contract")
            if declared_calendar_roles and (
                s.events
                and pathlib.Path(s.events).resolve() != pathlib.Path(calendar_path).resolve()
                or s.event_known_at_col != contract["known_at_col"]
                or s.events_preannounced != contract["preannounced"]
                or s.event_episode_col != contract["episode_col"]
            ):
                raise ValueError("data spec calendar roles disagree with the calendar contract")
            if (
                type(contract["preannounced"]) is not bool
                or type(contract["complete_grid"]) is not bool
            ):
                raise ValueError("calendar availability/coverage flags must be boolean")
            if not contract["preannounced"] and not contract["known_at_col"]:
                raise ValueError("calendar needs known_at or an explicit preannounced assumption")
            if not set(contract["binary_columns"]) <= set(contract["known_future"]):
                raise ValueError("binary calendar columns must be declared known-future")
            columns = {contract["time_col"], *contract["known_future"]}
            columns.update(v for pair in contract["event_pairs"] for v in pair if v)
            columns.update(
                v
                for v in (contract["group_col"], contract["known_at_col"], contract["episode_col"])
                if v
            )
            if not columns <= set(table.column_names):
                raise ValueError("unexpected revision calendar schema")
            rows = table.to_pylist()
            self.event_records = _revision_event_records(
                rows, contract, c.variant, dt.date.fromisoformat(selection_end)
            )
            variant = contract["variants"].get(c.variant, {})
            self.calendar_features = tuple(variant.get("known_future", contract["known_future"]))
            if not set(self.calendar_features) <= set(contract["known_future"]):
                raise ValueError("calendar variant names an undeclared future covariate")
            calendar_active = bool(self.event_records or self.calendar_features)
            group_column = c.calendar_group_column or contract["group_column"]
            if contract["group_col"] and calendar_active:
                if not group_column:
                    raise ValueError("calendar scope needs an explicitly mapped identity column")
                scopes = self._values(group_column)
                self.calendar_scopes = tuple(sorted(set(scopes)))
                self.scope_index = np.array(
                    [self.calendar_scopes.index(v) for v in scopes], np.int64
                )
            for row in rows:
                day = row[contract["time_col"]]
                scope = row[contract["group_col"]] if contract["group_col"] else ""
                known = row[contract["known_at_col"]] if contract["known_at_col"] else dt.date.min
                if (
                    type(day) is not dt.date
                    or type(known) is not dt.date
                    or not isinstance(scope, str)
                    or not scope.strip()
                    and contract["group_col"]
                ):
                    raise ValueError("calendar dates/knowledge dates and scopes must be typed")
                key = (day, scope)
                if key in self.calendar_rows:
                    raise ValueError("duplicate calendar date/group")
                for name in contract["known_future"]:
                    value = row[name]
                    if value is not None and (
                        not isinstance(value, int | float) or not math.isfinite(value)
                    ):
                        raise ValueError("calendar known-future values must be missing or finite")
                    if name in contract["binary_columns"] and value not in (None, 0, 1):
                        raise ValueError("binary calendar covariates must be missing or 0/1")
                self.calendar_rows[key] = (known, row)
            if contract["complete_grid"] and calendar_active:
                margin = c.event_max_support if self.event_records else 0
                coverage_step = 1 if self.event_records else data.step_days
                first = dt.date.fromisoformat(min(training_origins)) + dt.timedelta(
                    days=data.step_days - margin
                )
                last = data.cutoff + dt.timedelta(days=data.step_days * data.horizon + margin)
                dates = [
                    first + dt.timedelta(days=i * coverage_step)
                    for i in range((last - first).days // coverage_step + 1)
                ]
                if any(
                    (day, group) not in self.calendar_rows
                    for day in dates
                    for group in self.calendar_scopes
                ):
                    raise ValueError(
                        "incomplete calendar date/group grid including event-support margins"
                    )
            self.events = {
                name: tuple(sorted({r[0] for r in rows}))
                for name, rows in self.event_records.items()
            }
            s.events = str(calendar_path) if self.events else None
            s.event_known_at_col = contract["known_at_col"]
            s.events_preannounced = contract["preannounced"]
            s.event_episode_col = contract["episode_col"]
        elif declared_calendar_roles:
            raise ValueError("calendar roles declared without a calendar input")
        self.event_names = tuple(sorted(self.events))
        if self.event_names and c.event_pool_columns:
            self.event_pool_columns = c.event_pool_columns
            self.event_pool_index = self._groups(self.event_pool_columns)
        self.active_events = np.ones(len(self.event_names), bool)
        model_spec = ForecastModelSpec(
            d_model=c.d_model,
            heads=c.n_heads,
            patch_size=c.patch_size,
            branch_layers=c.history_attention_layers,
            trunk_layers=c.trunk_stages,
            decoder_layers=c.decoder_layers,
            head="negbin" if c.head == "negbin" else "hurdle_negbin",
            topology="bottom_up_factor" if c.factor_rank else "bottom_up",
            factor_rank=c.factor_rank,
            demand_branch_dropout=c.demand_dropout,
            covariate_dropout=c.covariate_dropout,
            event_max_support=(-c.event_max_support, c.event_max_support),
            static_variable_selection=c.variant == "v1_corrected" and bool(self.attribute_widths),
            price_response_mode="soft_penalty_v1" if c.price_penalty_weight else "none",
            attribution=None,
            emit="parameters",
            first_build=True,
        )
        stop = data.position(selection_end)
        eligible = data.observed[:, :stop]
        if not eligible.any():
            raise ValueError("selection training history has no observed counts")
        zero_share = (
            float((data.target[:, :stop][eligible] == 0).mean()) if c.head == "auto" else None
        )
        self.resolved = ResolvedBuild.for_revision(
            s,
            model_spec,
            variant=c.variant,
            zero_share=zero_share,
            head_law=c.v5.head_law if c.variant == "v5" and c.v5 is not None else None,
        )
        factor_columns = c.factor_columns or tuple(
            v for v in (s.factor_group_col, panel.product_col) if v
        )
        self.groups = (
            self._groups(factor_columns)
            if self.resolved.factor_rank
            else np.arange(len(data.series_ids), dtype=np.int64)
        )
        self.factor_columns = factor_columns if self.resolved.factor_rank else ()
        self.future_columns = tuple(
            v
            for v in (*s.time_varying_known_reals, *s.time_varying_known_categoricals)
            if v not in {panel.price_col, panel.promo_col}
        )
        self.history_columns = tuple(
            (*s.time_varying_unknown_reals, *s.time_varying_unknown_categoricals)
        )
        self.categorical_columns = set(
            (*s.time_varying_known_categoricals, *s.time_varying_unknown_categoricals)
        )
        self.dynamic_vocab = {}
        self.cycles = (
            tuple(c.cycles)
            if c.cycles is not None
            else (("seasonal", float(panel.season_length * data.step_days), 1),)
        )
        if any(
            len(v) != 3
            or not isinstance(v[0], str)
            or not v[0].strip()
            or type(v[1]) not in (float, int)
            or not math.isfinite(v[1])
            or v[1] <= 0
            or type(v[2]) is not int
            or v[2] < 1
            for v in self.cycles
        ) or len({v[0] for v in self.cycles}) != len(self.cycles):
            raise ValueError("cycles require unique names, positive periods and integer harmonics")
        self.has_commercial = bool(panel.price_col or panel.promo_col)
        if (
            c.variant not in {"v3", "v5"}
            and self.has_commercial
            and c.d_model % c.commercial_attention_heads
        ):
            raise ValueError("commercial attention heads must divide d_model")
        self._fit_layout(min(training_origins))

    def _values(self, name):
        """Role-bound identities; tested by ``test_generic_revision_features``."""
        if name == self.panel.series_col:
            return self.data.series_ids
        if name not in self.data.metadata:
            raise ValueError(f"group/identity column was not admitted: {name}")
        return self.data.metadata[name]

    def _indices(self, name):
        """Dense indices for one declared role; tested by ``test_generic_revision_features``."""
        import numpy as np

        self._values(name)  # Fail before assembly if a configured identity was not admitted.
        return (
            np.arange(len(self.data.series_ids), dtype=np.int64)
            if name == self.panel.series_col
            else self.indices[name]
        )

    def _groups(self, columns, absent=0):
        """Compose only a declared grouping; tested by ``test_generic_revision_features``."""
        import numpy as np

        if not columns:
            return np.full(len(self.data.series_ids), absent, np.int64)
        pairs = list(zip(*(self._values(name) for name in columns), strict=True))
        mapping = {pair: i for i, pair in enumerate(sorted(set(pairs)))}
        return np.array([mapping[pair] for pair in pairs], np.int64)

    def _fit_layout(self, origin):
        """Freeze categories from history before the earliest fitted origin, including an OOV row."""
        stop = self.data.position(origin)
        self.dynamic_vocab = {}
        for name in self.categorical_columns:
            values = self.data.features[name][:, :stop].reshape(-1)
            vocabulary = tuple(sorted({v for v in values if v is not None}))
            if len(vocabulary) > MAX_ONE_HOT:
                raise ValueError("dynamic categorical vocabulary exceeds MAX_ONE_HOT")
            self.dynamic_vocab[name] = vocabulary
        self.history_width = sum(
            len(self.dynamic_vocab[name]) + 1 if name in self.dynamic_vocab else 1
            for name in self.history_columns
        )
        shape = (1, self.data.horizon, 3)
        event_shape = (1, self.data.horizon, len(self.event_names))
        events = (torch.zeros(event_shape), torch.zeros(event_shape))
        layout = _known_future(
            self.data.cutoff.isoformat(),
            self.data.horizon,
            1,
            (torch.zeros(shape), torch.zeros(shape)) if self.has_commercial else None,
            events if self.config.variant == "v1_corrected" and self.event_names else None,
            self.event_names,
            cycles=self.cycles,
            step_days=self.data.step_days,
        )
        names = list(layout.names)
        for name in self.future_columns:
            width = len(self.dynamic_vocab[name]) + 1 if name in self.dynamic_vocab else 1
            names.extend(f"future:{name}:{j}" for j in range(width))
            names.extend(f"future:{name}:{j}:available" for j in range(width))
        for name in self.calendar_features:
            names.extend((name, f"{name}_available"))
        if len(set(names)) != len(names):
            raise ValueError("duplicate known-future feature names")
        self.known_names = tuple(names)

    def _calendar_at(self, origin, dates):
        """Scope and knowledge-cutoff aware event tensors; tested by ``test_revision_origin_integrity``."""
        import numpy as np

        shape = (len(self.calendar_scopes), len(dates), len(self.event_names))
        distance, present = np.zeros(shape, np.float32), np.zeros(shape, bool)
        episodes = np.empty(shape, object)
        episodes.fill(None)
        cutoff = dt.date.fromisoformat(origin)
        for j, name in enumerate(self.event_names):
            for group, scope in enumerate(self.calendar_scopes):
                candidates = sorted(
                    {
                        (day, episode)
                        for day, scoped, known, episode in self.event_records[name]
                        if scoped == scope and known <= cutoff
                    }
                )
                if not candidates:
                    continue
                for step, day in enumerate(dates):
                    nearest, episode = min(
                        candidates, key=lambda event: (abs((day - event[0]).days), event)
                    )
                    value = (day - nearest).days
                    if abs(value) <= self.config.event_max_support:
                        distance[group, step, j], present[group, step, j] = value, True
                        episodes[group, step, j] = episode
        return distance, present, episodes

    def begin_phase(self, origins, phase):
        """Freeze phase categories and estimable event channels using training windows only."""
        import numpy as np

        self._fit_layout(min(origins))
        event_cells = np.zeros(len(self.event_names), np.int64)
        nonevent_cells = np.zeros_like(event_cells)
        episodes = [set() for _ in self.event_names]
        for origin in origins:
            stop = self.data.position(origin)
            if stop + self.data.horizon > self.data.target.shape[1]:
                raise ValueError("event admission needs origin-bounded training target windows")
            if not self.event_names:
                continue
            dates = [self.data.date_at(i) for i in range(stop, stop + self.data.horizon)]
            _, present, identity = self._calendar_at(origin, dates)
            observed = self.data.observed[:, stop : stop + self.data.horizon]
            for group in range(len(self.calendar_scopes)):
                counts = observed[self.scope_index == group].sum(0)
                event_cells += (present[group] * counts[:, None]).sum(0)
                nonevent_cells += (~present[group] * counts[:, None]).sum(0)
                for j in range(len(self.event_names)):
                    episodes[j].update(identity[group, :, j][present[group, :, j] & (counts > 0)])
        active = [
            len(ids) >= 2 and yes > 0 and no > 0
            for ids, yes, no in zip(episodes, event_cells, nonevent_cells, strict=True)
        ]
        self.active_events = np.array(active, bool)
        self.cache, self.cache_origin = None, None
        coverage = {
            name: dict(
                enabled=bool(active[j]),
                informative_event_cells=int(event_cells[j]),
                informative_non_event_cells=int(nonevent_cells[j]),
                distinct_occurrences=sorted(episodes[j]),
            )
            for j, name in enumerate(self.event_names)
        }
        return dict(
            phase=phase,
            policy="two_occurrences_plus_informative_contrast_v1",
            coverage_basis="declared_training_windows_not_sampled_counts",
            events=coverage,
            dynamic_vocab=self.dynamic_vocab,
            known_names=self.known_names,
            static_normalization=self.static_stats,
            layout_origin=min(origins),
        )

    def _origin(self, origin) -> dict:
        """Cache population priors and only enabled auxiliary computations at one origin."""
        import numpy as np

        if self.cache_origin == origin and self.cache is not None:
            return self.cache
        data, c = self.data, self.config
        stop = data.position(origin)
        if c.variant == "v1_corrected" or self.aggregate_index is not None:
            window = data.window(
                origin,
                np.arange(len(data.series_ids)),
                c.encoder_days,
                targets=False,
                left_pad=c.observation_contract == GLOBAL_LEFT_PADDING_CONTRACT,
            )
            history = torch.tensor(window["history"])
            observed = torch.tensor(window["observed"])
        category = torch.tensor(self.baseline_index)
        if c.variant == "v1_corrected":
            logit, log_size, _ = _shrunk_baselines(history, observed, category)
        else:
            logits, sizes = [], []
            positions = np.arange(max(0, stop - c.seasonal_window), stop)
            for day in range(data.horizon):
                selected = positions[
                    positions % self.panel.season_length == (stop + day) % self.panel.season_length
                ]
                if not len(selected):
                    selected = positions
                if c.variant == "v3" and not data.observed[:, selected].any():
                    selected = positions
                    if not data.observed[:, selected].any():
                        raise ValueError("HMFNet baseline has no origin-available group/global donors")
                p, size, _ = _shrunk_baselines(
                    torch.tensor(data.target[:, selected]),
                    torch.tensor(data.observed[:, selected]),
                    category,
                )
                logits.append(p)
                sizes.append(size)
            logit, log_size = torch.stack(logits, 1), torch.stack(sizes, 1)
        if self.resolved.head == "negbin":
            log_size = torch.nn.functional.logsigmoid(logit) + log_size
        elif c.variant == "v1_corrected":
            log_size = torch.log(torch.expm1(log_size).clamp_min(c.excess_floor))
        dates = [data.date_at(i) for i in range(stop, stop + data.horizon)]
        distance, present, _ = self._calendar_at(origin, dates)
        present &= self.active_events[None, None, :]
        result = dict(
            logit=logit,
            log_size=log_size,
            dates=dates,
            distance=np.where(present, distance, 0),
            present=present,
        )
        if self.aggregate_index is not None and c.variant != "v3":
            index = self.aggregate_index
            aggregate = np.zeros((int(index.max()) + 1, c.encoder_days), np.float32)
            count = np.zeros_like(aggregate, np.int32)
            np.add.at(aggregate, index, np.where(observed.numpy(), history.numpy(), 0))
            np.add.at(count, index, observed.numpy().astype(np.int32))
            complete = (
                count > 0
                if data.observation_contract.startswith("complete_recorded")
                else count == np.bincount(index)[:, None]
            )
            result.update(aggregate=np.where(complete, aggregate, 0), aggregate_observed=complete)
        if (c.variant in {"v2", "v3"} and c.mean_loss_weight) or (
            c.series_loss_weighting == "revenue"
        ):
            sold = data.target[:, max(0, stop - c.reference_window) : stop]
            mask = data.observed[:, max(0, stop - c.reference_window) : stop]
            mean_weighting = (
                ("revenue" if data.price.shape[1] else "uniform")
                if c.mean_weighting == "auto"
                else c.mean_weighting
            )
            revenue = None
            if mean_weighting == "revenue" or c.series_loss_weighting == "revenue":
                if not data.price.shape[1] or (
                    not data.observation_contract.startswith("complete_recorded") and not mask.all()
                ):
                    raise ValueError("revenue weights require known recorded quantities and prices")
                price = data.price[:, max(0, stop - c.reference_window) : stop]
                if np.any((sold > 0) & ~np.isfinite(price)):
                    raise ValueError("positive sales lack weight prices")
                revenue = (sold * np.nan_to_num(price, nan=0.0)).sum(1, dtype=np.float64)
                if (
                    not np.isfinite(revenue).all()
                    or not np.isfinite(revenue.sum())
                    or revenue.sum() <= 0
                ):
                    raise ValueError("origin has undefined/nonfinite mean-loss weights")
            if c.variant in {"v2", "v3"} and c.mean_loss_weight:
                mean_weight = (
                    revenue
                    if mean_weighting == "revenue"
                    else np.ones(len(data.series_ids), np.float64)
                )
                recent = (sold * mask).sum(1) / np.maximum(mask.sum(1), 1)
                result["mean_weight"] = mean_weight / mean_weight.sum() / (1 + recent) ** 2
            if c.series_loss_weighting == "revenue":
                assert revenue is not None
                result["series_loss_weight"] = revenue / revenue.mean()
        self.cache_origin, self.cache = origin, result
        return result

    def _covariates(self, names, rows, start, stop, origin_stop):
        """Named covariates normalized using origin history; tested by ``test_history_covariates``."""
        import numpy as np

        if self.numeric_stats_origin != origin_stop:
            self.numeric_stats, self.numeric_stats_origin = {}, origin_stop
        values, masks = [], []
        for name in names:
            source = self.data.features[name]
            real_start = max(0, start)
            raw = source[rows, real_start:stop]
            if start < 0:
                width = stop - start
                if name in self.dynamic_vocab:
                    padded = np.empty((len(rows), width), dtype=object)
                    padded.fill(None)
                else:
                    padded = np.full((len(rows), width), np.nan, dtype=np.float64)
                padded[:, -raw.shape[1] :] = raw
                raw = padded
            if name in self.dynamic_vocab:
                vocabulary = self.dynamic_vocab[name]
                mapping = {v: i for i, v in enumerate(vocabulary)}
                index = np.array([mapping.get(v, len(vocabulary)) for v in raw.reshape(-1)])
                value = np.eye(len(vocabulary) + 1, dtype=np.float32)[index].reshape(*raw.shape, -1)
                mask = np.repeat((raw != None)[..., None], value.shape[-1], axis=-1)  # noqa: E711
            else:
                if name not in self.numeric_stats:
                    past = self.data.features[name][:, :origin_stop]
                    finite = past[np.isfinite(past)]
                    self.numeric_stats[name] = (
                        float(finite.mean()) if len(finite) else 0.0,
                        max(float(finite.std()), 1e-6) if len(finite) else 1.0,
                    )
                mean, scale = self.numeric_stats[name]
                mask = np.isfinite(raw)[..., None]
                value = ((np.where(np.isfinite(raw), raw, mean) - mean) / scale)[..., None]
            values.append(np.where(mask, value, 0).astype(np.float32))
            masks.append(mask)
        shape = (len(rows), stop - start, 0)
        return (
            torch.tensor(np.concatenate(values, -1) if values else np.empty(shape, np.float32)),
            torch.tensor(np.concatenate(masks, -1) if masks else np.empty(shape, bool)),
        )

    def batch(self, origin, rows, device, targets):
        """Construct one minibatch with explicit, ordered feature masks and no future targets."""
        import numpy as np

        data, c, panel = self.data, self.config, self.panel
        stop = data.position(origin)
        padded = c.observation_contract == GLOBAL_LEFT_PADDING_CONTRACT
        values = data.window(
            origin,
            rows,
            c.encoder_days,
            targets=targets,
            reference_window=c.reference_window,
            left_pad=padded,
        )
        cached = self._origin(origin)
        commercial_values = commercial_mask = None
        if self.has_commercial:
            price = (
                values["price_future"]
                if panel.price_col
                else np.full((len(rows), data.horizon), np.nan)
            )
            reference = np.array(
                [
                    np.quantile(p[np.isfinite(p)], 0.5, method="lower")
                    if np.isfinite(p).any()
                    else np.nan
                    for p in values["price_history"]
                ]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = 1 - price / reference[:, None]
            promo = (
                data.features[panel.promo_col][rows, stop : stop + data.horizon]
                if panel.promo_col
                else np.full_like(price, np.nan)
            )
            raw = np.stack((price, depth, promo), -1)
            available = np.isfinite(raw)
            available[..., 1] &= reference[:, None] > 0
            commercial_values = torch.tensor(np.where(available, raw, 0), dtype=torch.float32)
            commercial_mask = torch.tensor(available)
        distance = torch.tensor(cached["distance"][self.scope_index[rows]])
        present = torch.tensor(cached["present"][self.scope_index[rows]])
        known = _known_future(
            origin,
            data.horizon,
            len(rows),
            (commercial_values, commercial_mask.float())
            if commercial_values is not None and commercial_mask is not None
            else None,
            (distance, present.float())
            if c.variant == "v1_corrected" and self.event_names
            else None,
            self.event_names,
            cycles=self.cycles,
            step_days=data.step_days,
        )
        future = [known.values]
        if self.future_columns:
            x, mask = self._covariates(self.future_columns, rows, stop, stop + data.horizon, stop)
            # _covariates packs all values then masks, matching the layout per variable below.
            offset = 0
            for name in self.future_columns:
                width = len(self.dynamic_vocab[name]) + 1 if name in self.dynamic_vocab else 1
                future.extend(
                    (x[..., offset : offset + width], mask[..., offset : offset + width].float())
                )
                offset += width
        for name in self.calendar_features:
            value = np.zeros((len(rows), data.horizon), np.float32)
            mask = np.zeros_like(value, bool)
            for i, row in enumerate(rows):
                scope = self.calendar_scopes[self.scope_index[row]]
                for step, day in enumerate(cached["dates"]):
                    if (day, scope) not in self.calendar_rows:
                        raise ValueError("incomplete calendar known-future date/group grid")
                    known_at, record = self.calendar_rows[(day, scope)]
                    if known_at <= dt.date.fromisoformat(origin) and record[name] is not None:
                        value[i, step], mask[i, step] = record[name], True
            future.extend((torch.tensor(value)[..., None], torch.tensor(mask)[..., None].float()))
        n = len(rows)
        inputs = dict(
            history=torch.tensor(values["history"]),
            observed=torch.tensor(values["observed"]),
            attributes=torch.tensor(self.attributes[rows]),
            series_index=torch.tensor(rows),
            identity_indices=torch.tensor(self.identity_indices[rows]),
            known_future=torch.cat(future, -1),
            commercial=commercial_values[:, 0]
            if commercial_values is not None
            else torch.empty(n, 0),
            commercial_mask=commercial_mask[:, 0]
            if commercial_mask is not None
            else torch.empty(n, 0, dtype=torch.bool),
            baseline_logit=cached["logit"][rows],
            baseline_log_size=cached["log_size"][rows],
            event_distance=distance,
            event_present=present,
        )
        if inputs["known_future"].shape[-1] != len(self.known_names):
            raise ValueError("known-future layout changed across origins")
        if self.history_width and c.variant not in {"v3", "v5"}:
            x, mask = self._covariates(
                self.history_columns, rows, stop - c.encoder_days, stop, stop
            )
            inputs.update(history_covariates=x, history_covariate_mask=mask)
        if c.variant == "v1_corrected":
            inputs.update(
                group_index=torch.tensor(self.groups[rows]),
                n_draws=c.factor_draws,
                store_index=torch.tensor(
                    self.indices[panel.store_col][rows] if panel.store_col else rows
                ),
            )
            if self.event_pool_index is not None:
                inputs["event_pool_index"] = torch.tensor(self.event_pool_index[rows])
        else:
            if c.variant == "v2" and padded:
                inputs["history_available"] = torch.tensor(values["available"])
            if self.event_pool_index is not None:
                inputs["category_index"] = torch.tensor(self.event_pool_index[rows])
            if self.aggregate_index is not None and c.variant != "v3":
                group = self.aggregate_index[rows]
                inputs.update(
                    aggregate_history=torch.tensor(cached["aggregate"][group]),
                    aggregate_observed=torch.tensor(cached["aggregate_observed"][group]),
                )
            if c.mean_loss_weight:
                inputs.update(
                    mean_weight=torch.tensor(cached["mean_weight"][rows], dtype=torch.float32),
                    sampling_fraction=torch.tensor(
                        len(np.unique(self.groups[rows])) / len(np.unique(self.groups))
                    ),
                )
            if c.series_loss_weighting == "revenue":
                inputs["series_loss_weight"] = torch.tensor(
                    cached["series_loss_weight"][rows], dtype=torch.float32
                )
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        y = torch.tensor(values["target"], device=device) if targets else None
        mask = torch.tensor(values["target_observed"], device=device) if targets else None
        return inputs, y, mask

    def make_model(self):
        """Build the exact resolved head, optional branches and feature dimensions."""
        from forecast.models.mft.sft import SFTModel

        c, r, data = self.config, self.resolved, self.data
        common: dict = dict(
            d_model=r.d_model,
            encoder_days=c.encoder_days,
            n_attributes=self.attributes.shape[1],
            n_series=len(data.series_ids),
            n_known=len(self.known_names),
            horizon=data.horizon,
            n_products=0,
            n_stores=0,
            n_events=len(self.event_names),
            event_max_support=c.event_max_support,
            n_heads=r.attention_heads,
            trunk_stages=c.trunk_stages,
            decoder_layers=c.decoder_layers,
            history_attention_layers=c.history_attention_layers,
            commercial_attention_layers=c.commercial_attention_layers,
            commercial_attention_heads=c.commercial_attention_heads,
            event_kernel_layers=c.event_kernel_layers,
            demand_dropout=r.demand_branch_dropout,
            patch_size=r.patch_size,
            n_commercial=3 if self.has_commercial else 0,
            embedding_residual_dim=r.embedding_residual_dim,
            identity_cardinalities=self.identity_cardinalities,
            n_history_covariates=self.history_width,
        )
        if c.variant == "v2":
            return SFTModel(
                **common,
                n_categories=int(self.event_pool_index.max()) + 1
                if self.event_pool_index is not None
                else 0,
                excess_floor=c.excess_floor,
                aggregate_history=self.aggregate_index is not None,
                statistic_windows=c.statistic_windows,
                head_kind=r.head,
                calendar_availability=(c.observation_contract == GLOBAL_LEFT_PADDING_CONTRACT),
            )
        common["n_series"] += 1
        # Loading identity is independent of feature identity; without a store role use the
        # explicitly modeled series, not a fabricated shared store. M5 keeps its store table.
        common["n_stores"] = (
            (
                len(self.vocab[self.panel.store_col])
                if self.panel.store_col
                else len(data.series_ids)
            )
            + 1
            if r.factor_rank
            else 0
        )

        def index(name):
            return self.known_names.index(name) if name in self.known_names else None

        model = CustomModel(
            **common,
            factor_rank=r.factor_rank,
            head_kind=r.head,
            n_event_pools=int(self.event_pool_index.max()) + 1
            if self.event_pool_index is not None
            else None,
            attribute_widths=self.attribute_widths if r.static_variable_selection else (),
            price_index=index("price"),
            price_available_index=index("price_available"),
            discount_index=index("discount_depth"),
            discount_available_index=index("discount_depth_available"),
            promo_index=index("promo"),
            price_response_mode=r.price_response_mode,
            event_distance_index=tuple(
                self.known_names.index(f"{n}_distance") for n in self.event_names
            ),
            event_present_index=tuple(
                self.known_names.index(f"{n}_present") for n in self.event_names
            ),
        )
        layers = (model.head.size, model.head.dispersion)
        if isinstance(model.head, HurdleNegBinHead):
            layers = (model.head.occurrence, *layers)
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        return model

    def mean(self, model, inputs):
        """Deterministic factor integration for MFT, exact resolved count mean for SFT."""
        if self.config.variant == "v1_corrected":
            return point_forecast(model, inputs)
        return model(**self._model_inputs(inputs)).mean()

    def _model_inputs(self, inputs):
        """SFT's forward does not accept loss-only weighting fields; HMFNet overrides this seam."""
        return {
            k: v
            for k, v in inputs.items()
            if k not in {"mean_weight", "sampling_fraction", "series_loss_weight"}
        }

    def validation_values(self, model, inputs):
        """One deterministic SFT/HMFNet forward; return mean and the same distribution's NLL."""
        prediction = model(**self._model_inputs(inputs))
        return prediction.mean(), prediction.nll

    def feature_training_state(self):
        """SFT has deterministic rebuildable feature caches, no independent sampler."""
        return {"schema": "revision_feature_state_v1"}

    def restore_feature_training_state(self, state):
        """Strictly restore after begin_phase has reproduced the saved feature layout."""
        if state != {"schema": "revision_feature_state_v1"}:
            raise ValueError("invalid SFT feature training state")
        self.cache, self.cache_origin = None, None

    def diagnostics(self, model, detailed):
        """SFT loss components for display; no additional training/evaluation forward."""
        return {"loss_components": dict(getattr(self, "last_components", {}))}

    def optimizer_groups(self, model):
        """Preserve residual-only decay, including generic identity tables and SFT dispersion."""
        groups, _ = parameter_groups(model, self.config.weight_decay)
        extra = {id(p) for p in model.static.residual_identities.parameters()}
        if self.config.variant in {"v2", "v3"}:
            extra.add(id(model.dispersion_residual.weight))
        moved = [p for p in groups[0]["params"] if id(p) in extra]
        groups[0]["params"] = [p for p in groups[0]["params"] if id(p) not in extra]
        groups[1]["params"].extend(moved)
        return groups

    def regularized_inputs(self, inputs):
        """Drop commercial availability consistently in branch and decoder inputs."""
        if not self.config.covariate_dropout or not self.has_commercial:
            return inputs
        result, before = dict(inputs), inputs["commercial_mask"]
        result["commercial"], result["commercial_mask"] = drop_covariates(
            inputs["commercial"], before, self.config.covariate_dropout, allow_empty=True
        )
        columns = [
            (self.known_names.index(name), self.known_names.index(f"{name}_available"))
            for name in commercial.CHANNELS
        ]
        result["known_future"] = _hide_known_channels(
            inputs["known_future"], before & ~result["commercial_mask"], columns
        )
        return result

    def training_loss(self, model, inputs, y, observed, denominator, importance=1.0):
        """Count likelihood and only enabled auxiliary objectives; no prediction clipping."""
        c = self.config
        inputs = self.regularized_inputs(inputs)
        if c.variant == "v1_corrected":
            likelihood = model.loss(
                y,
                torch.where(observed, 0, 2),
                normalization_count=denominator,
                likelihood_dtype=torch.float64,
                **inputs,
            )
            penalty = (
                model.price_penalty(inputs)[0]
                if c.price_penalty_weight and self.panel.price_col
                else 0.0
            )
            return importance * likelihood + c.price_penalty_weight * penalty
        prediction = model(**self._model_inputs(inputs))
        per_series_nll = prediction.nll(y, observed).sum(1)
        weights = series_loss_weights(
            c.series_loss_weighting,
            per_series_nll,
            revenue_weight=inputs.get("series_loss_weight"),
            distribution_outputs=(prediction.logit, prediction.excess, prediction.dispersion),
        )
        likelihood = weighted_series_mean(per_series_nll, denominator, weights)
        mean_loss = 0.0
        if c.mean_loss_weight:
            mean_loss = (
                ((prediction.mean() - y).square() * observed).mean(1) * inputs["mean_weight"]
            ).sum() / inputs["sampling_fraction"]
        event_loss = c.event_penalty_weight * model.event_penalty()
        result = importance * (likelihood + c.mean_loss_weight * mean_loss) + event_loss
        self.last_components = dict(
            **weights.diagnostics(),
            nll=float(likelihood.detach()),
            weighted_nll=float((importance * likelihood).detach()),
            weighted_mean=float((importance * c.mean_loss_weight * mean_loss).detach())
            if isinstance(mean_loss, torch.Tensor)
            else 0.0,
            weighted_event=float(event_loss.detach()),
        )
        return result


class HMFNetRevisionFeatures(RevisionFeatures):
    """Data-agnostic HMFNet assembly at the existing sole harness/model edge.

    test_hmfnet_revision.py covers origin cutoffs, typed features, objectives and replay.
    """

    def __init__(self, panel, data, config, calendar_path=None, data_spec=None):
        """Freeze group operators and feature dimensions; test_hmfnet_revision_features."""
        from autoforecast.hmfnet_features import AggregateHistory

        super().__init__(panel, data, config, calendar_path, data_spec)
        self.policy = config.v3
        if self.policy is None:
            raise ValueError("HMFNet assembly requires an explicit HMFNet recipe")
        if config.aggregate_history != bool(self.policy.context_levels):
            raise ValueError("HMFNet aggregate_history must agree with its context grouping set")
        if (
            config.aggregate_columns
            and tuple(sorted(config.aggregate_columns)) not in self.policy.context_levels
        ):
            raise ValueError("HMFNet aggregate_columns must name a declared context grouping")
        self.adaptation_reasons.pop("aggregate_history", None)
        self.aggregate_data = AggregateHistory(data, self.policy)
        self.numeric_history = tuple(
            name for name in self.history_columns if name not in self.categorical_columns
        )
        self.categorical_history = tuple(
            name for name in self.history_columns if name in self.categorical_columns
        )
        self.memory_width = (
            11
            + 4 * sum(h for _, h in self.policy.phase_cycles)
            + 4 * len(self.numeric_history)
            + 3 * len(self.categorical_history)
        )
        self.aggregate_known_names = _known_future(
            data.cutoff.isoformat(),
            data.horizon,
            1,
            None,
            None,
            (),
            cycles=self.cycles,
            step_days=data.step_days,
        ).names
        self.hmfnet_cache, self.hmfnet_cache_origin = None, None
        self.expert_rng = np.random.default_rng(self.policy.expert_seed)
        self.last_components = {}

    def begin_phase(self, origins, phase):
        """Reset phase vocabulary, origin caches and the separate RNG; test_hmfnet_begin_phase."""
        record = super().begin_phase(origins, phase)
        self.hmfnet_cache, self.hmfnet_cache_origin = None, None
        self.expert_rng = np.random.default_rng(self.policy.expert_seed)
        record.update(v3_feature_layout=self.feature_layout(), expert_seed=self.policy.expert_seed)
        return record

    def feature_layout(self):
        """Persist semantic dimensions with the checkpoint; test_hmfnet_resolution."""
        return dict(
            policy=self.policy.feature_policy,
            numeric_history=self.numeric_history,
            categorical_history=self.categorical_history,
            token_numeric_width=self.memory_width,
            statistic_width=11 + len(self.config.statistic_windows),
            aggregate_covariates=self.policy.aggregate_covariates,
            context_levels=self.policy.context_levels,
            expert_levels=self.policy.expert_levels,
            aggregate_nodes=len(self.aggregate_data.target),
            aggregate_node_keys=self.aggregate_data.node_keys,
            aggregate_known_names=self.aggregate_known_names,
            scale_policy="max_raw_scale_one_complete_history_v1",
        )

    def _covariates(self, names, rows, start, stop, origin_stop):
        """Bound future-covariate normalization to the declared memory span; test_hmfnet_revision_features."""
        if self.numeric_stats_origin != origin_stop:
            self.numeric_stats, self.numeric_stats_origin = {}, origin_stop
        low = max(0, origin_stop - self.policy.span) if self.policy.span else 0
        for name in names:
            if name in self.dynamic_vocab or name in self.numeric_stats:
                continue
            raw = self.data.features[name][:, low:origin_stop]
            finite = raw[np.isfinite(raw)]
            self.numeric_stats[name] = (
                float(finite.mean()) if len(finite) else 0.0,
                max(float(finite.std()), self.policy.numeric_scale_floor) if len(finite) else 1.0,
            )
        return super()._covariates(names, rows, start, stop, origin_stop)

    def _memory(self, target, observed, stop, *, rows=None):
        """Build deterministic origin-only tokens; test_hmfnet_revision_features and cutoff tests."""
        from autoforecast.hmfnet_features import summarize_history

        low = max(0, stop - self.policy.span) if self.policy.span else 0
        shape = (len(target), stop - low)
        numeric, categorical, categorical_raw = [], [], []
        for name in self.numeric_history:
            numeric.append(
                self.data.features[name][rows, low:stop]
                if rows is not None
                else np.full(shape, np.nan)
            )
        for name in self.categorical_history:
            if rows is None:
                categorical.append(np.full(shape, -1, np.int64))
                categorical_raw.append(np.full(shape, None, object))
                continue
            raw = self.data.features[name][rows, low:stop]
            categorical_raw.append(raw)
            vocabulary = self.dynamic_vocab[name]
            mapping = {v: i for i, v in enumerate(vocabulary)}
            ids = np.fromiter(
                (mapping.get(v, len(vocabulary)) if v is not None else -1 for v in raw.reshape(-1)),
                np.int64,
            ).reshape(raw.shape)
            categorical.append(ids)
        return summarize_history(
            target[:, low:stop],
            observed[:, low:stop],
            np.arange(low, stop),
            origin_stop=stop,
            first_ordinal=self.data.start.toordinal(),
            step_days=self.data.step_days,
            policy=self.policy,
            statistic_windows=self.config.statistic_windows,
            numeric=tuple(numeric),
            categorical=tuple(categorical),
            categorical_raw=tuple(categorical_raw),
        )

    def _hmfnet_origin(self, origin):
        """Cache data summaries for one origin, never learned outputs; test_hmfnet_begin_phase."""
        if self.hmfnet_cache_origin == origin and self.hmfnet_cache is not None:
            return self.hmfnet_cache
        base = self._origin(origin)
        stop = self.data.position(origin)
        aggregate = self.aggregate_data
        if not len(aggregate.target):
            result = dict(aggregate=None)
        else:
            memory = self._memory(aggregate.target, aggregate.observed, stop)
            if self.resolved.head == "negbin":
                bottom = base["log_size"].double().exp()
            else:
                excess = base["log_size"].double().expm1().clamp_min(self.config.excess_floor)
                bottom = (
                    torch.nn.functional.logsigmoid(base["logit"].double()) + excess.log1p()
                ).exp()
            memory.update(
                anchor=aggregate.anchors(bottom.numpy()),
                type_index=aggregate.type_index,
                node_ids=np.arange(len(aggregate.target)),
                known_future=_known_future(
                    origin,
                    self.data.horizon,
                    len(aggregate.target),
                    None,
                    None,
                    (),
                    cycles=self.cycles,
                    step_days=self.data.step_days,
                ).values.numpy(),
            )
            result = dict(aggregate=memory)
        self.hmfnet_cache_origin, self.hmfnet_cache = origin, result
        return result

    def _aggregate_tensors(self, cached, nodes, device):
        """Keep non-row band descriptors distinct when gathering nodes; test_hmfnet_revision_features."""
        return {
            name: torch.as_tensor(
                value if name in {"memory_bands", "memory_regular"} else value[nodes], device=device
            )
            for name, value in cached.items()
        }

    def batch(self, origin, rows, device, targets):
        """Gather context/feedback nodes and independently sampled targets; test_hmfnet_revision_features."""
        started = time.perf_counter()
        ordinary, target, observed = super().batch(origin, rows, device, targets)
        stop, groups = self.data.position(origin), self.aggregate_data
        inputs = {
            name: value
            for name, value in ordinary.items()
            if name
            not in {
                "history",
                "observed",
                "history_covariates",
                "history_covariate_mask",
                "aggregate_history",
                "aggregate_observed",
            }
        }
        raw_memory = self._memory(
            self.data.target[rows, :stop], self.data.observed[rows, :stop], stop, rows=rows
        )
        inputs["memory"] = {
            name: torch.as_tensor(value, device=device) for name, value in raw_memory.items()
        }
        cached = self._hmfnet_origin(origin)
        sampled, coefficients = np.empty(0, np.int64), np.empty(0, np.float64)
        supervision = None
        if targets and self.policy.expert_levels and self.policy.expert_loss_weight:
            weighting = self.config.hierarchy_weighting
            if weighting == "auto":
                weighting = "revenue" if self.data.price.shape[1] else "uniform"
            supervision = groups.supervision(stop, weighting, self.config.reference_window)
            sampled, coefficients = groups.sample_experts(supervision, self.expert_rng)
        if cached["aggregate"] is not None:
            nodes = np.unique(
                np.concatenate(
                    (groups.context[rows].reshape(-1), groups.experts[rows].reshape(-1), sampled)
                )
            )
            inputs["aggregate"] = self._aggregate_tensors(cached["aggregate"], nodes, device)
            inputs["context_memberships"] = torch.tensor(
                np.searchsorted(nodes, groups.context[rows]), device=device
            )
            inputs["expert_memberships"] = torch.tensor(
                np.searchsorted(nodes, groups.experts[rows]), device=device
            )
            if supervision is not None:
                group_records = []
                for group in self.policy.expert_levels:
                    block = groups.slices[group]
                    selected = (sampled >= block.start) & (sampled < block.stop)
                    group_records.append(
                        dict(
                            columns=list(group),
                            nodes=block.stop - block.start,
                            eligible=int(supervision["eligible"][block].sum()),
                            sampled=int(selected.sum()),
                            sampled_node_ids=sampled[selected].tolist(),
                            sampled_coefficients=coefficients[selected].tolist(),
                            sampled_normalized_weights=supervision["weight"][
                                sampled[selected]
                            ].tolist(),
                            sampled_scales=supervision["scale"][sampled[selected]].tolist(),
                            weight_mass=float(supervision["weight"][block].sum()),
                            incomplete_history=int((~supervision["history_complete"][block]).sum()),
                            incomplete_target=int((~supervision["target_complete"][block]).sum()),
                            invalid_price=int((~supervision["price_valid"][block]).sum()),
                        )
                    )
                inputs["expert_supervision"] = dict(
                    groups=group_records,
                    rows=torch.tensor(np.searchsorted(nodes, sampled), device=device),
                    target=torch.tensor(supervision["target"][sampled], device=device),
                    scale=torch.tensor(supervision["scale"][sampled], device=device),
                    coefficient=torch.tensor(coefficients, device=device),
                    eligible=int(
                        sum(
                            supervision["eligible"][groups.slices[g]].sum()
                            for g in self.policy.expert_levels
                        )
                    ),
                    active_groups=int(
                        sum(
                            supervision["weight"][groups.slices[g]].sum() > 0
                            for g in self.policy.expert_levels
                        )
                    ),
                )
        self.last_batch_seconds = time.perf_counter() - started
        return inputs, target, observed

    def feature_training_state(self):
        """Expert proposals have a separate RNG; it is part of exact training continuation."""
        return dict(schema="v3_feature_state_v1", expert_rng=self.expert_rng.bit_generator.state)

    def restore_feature_training_state(self, state):
        """Restore expert RNG only after deterministic phase feature reconstruction."""
        if (
            not isinstance(state, dict)
            or set(state) != {"schema", "expert_rng"}
            or state["schema"] != "v3_feature_state_v1"
        ):
            raise ValueError("invalid HMFNet feature training state")
        self.expert_rng.bit_generator.state = state["expert_rng"]
        self.cache, self.cache_origin = None, None
        self.hmfnet_cache, self.hmfnet_cache_origin = None, None

    def make_model(self):
        """Build only the HMFNet backbone and resolved count/event components; test_hmfnet_resolution."""
        from forecast.models.mft.hmfnet import HMFNetModel

        c, r = self.config, self.resolved
        return HMFNetModel(
            d_model=c.d_model,
            n_heads=c.n_heads,
            local_blocks=self.policy.local_blocks,
            n_token_features=self.memory_width,
            n_statistics=11 + len(c.statistic_windows),
            n_attributes=self.attributes.shape[1],
            n_series=len(self.data.series_ids),
            n_known=len(self.known_names),
            n_aggregate_known=len(self.aggregate_known_names),
            horizon=self.data.horizon,
            identity_cardinalities=self.identity_cardinalities,
            categorical_cardinalities=tuple(
                len(self.dynamic_vocab[n]) + 1 for n in self.categorical_history
            ),
            categorical_width=self.policy.categorical_width,
            n_types=len(self.aggregate_data.levels),
            n_context_groups=len(self.policy.context_levels),
            n_expert_groups=len(self.policy.expert_levels),
            feedback=self.policy.feedback,
            n_events=len(self.event_names),
            n_categories=int(self.event_pool_index.max()) + 1
            if self.event_pool_index is not None
            else 0,
            event_max_support=c.event_max_support,
            event_kernel_layers=c.event_kernel_layers,
            excess_floor=c.excess_floor,
            head_kind=r.head,
            embedding_residual_dim=r.embedding_residual_dim,
        )

    def _model_inputs(self, inputs):
        """Exclude loss-only targets and commercial dropout bookkeeping; test_hmfnet_training_loss."""
        excluded = {
            "commercial",
            "commercial_mask",
            "mean_weight",
            "sampling_fraction",
            "expert_supervision",
            "series_loss_weight",
        }
        return {name: value for name, value in inputs.items() if name not in excluded}

    def mean(self, model, inputs):
        """Same parameters drive training, export and replay; test_hmfnet_inference_cache."""
        return model(**self._model_inputs(inputs)).mean()

    def training_loss(self, model, inputs, y, observed, denominator, importance=1.0):
        """One leaf loss plus one sampled expert objective; test_hmfnet_training_loss."""
        inputs = self.regularized_inputs(inputs)
        prediction = model(**self._model_inputs(inputs))
        per_series_nll = prediction.nll(y, observed).sum(1)
        weights = series_loss_weights(
            self.config.series_loss_weighting,
            per_series_nll,
            revenue_weight=inputs.get("series_loss_weight"),
            distribution_outputs=(prediction.logit, prediction.excess, prediction.dispersion),
        )
        likelihood = weighted_series_mean(per_series_nll, denominator, weights)
        mean_loss = likelihood.new_zeros(())
        if self.config.mean_loss_weight:
            mean_loss = (
                ((prediction.mean() - y).square() * observed).mean(1) * inputs["mean_weight"]
            ).sum() / inputs["sampling_fraction"]
        expert_loss = likelihood.new_zeros(())
        expert_contributions = expert_loss.new_empty((0,))
        supervision = inputs.get("expert_supervision")
        if supervision is not None and len(supervision["rows"]):
            difference = prediction.expert_mean[supervision["rows"]] - supervision["target"]
            expert_contributions = (
                difference.square().mean(1) / supervision["scale"] * supervision["coefficient"]
            )
            expert_loss = expert_contributions.sum()
        event_loss = model.event_penalty()
        loss = (
            importance * (likelihood + self.config.mean_loss_weight * mean_loss)
            + self.policy.expert_loss_weight * expert_loss
            + self.config.event_penalty_weight * event_loss
        )
        self.last_components = dict(
            **weights.diagnostics(),
            importance=float(importance),
            total_objective=float(loss.detach()),
            weighted_nll=float((importance * likelihood).detach()),
            weighted_bottom_mean=float(
                (importance * self.config.mean_loss_weight * mean_loss).detach()
            ),
            weighted_expert=float((self.policy.expert_loss_weight * expert_loss).detach()),
            weighted_event=float((self.config.event_penalty_weight * event_loss).detach()),
            expert_groups=supervision["groups"] if supervision else [],
            expert_sample_contributions=expert_contributions.detach().tolist(),
            nll=float(likelihood.detach()),
            bottom_mean_loss=float(mean_loss.detach()),
            expert_loss=float(expert_loss.detach()),
            event_penalty=float(event_loss.detach()),
            expert_samples=len(supervision["rows"]) if supervision else 0,
            eligible_experts=supervision["eligible"] if supervision else 0,
            active_expert_groups=supervision["active_groups"] if supervision else 0,
            feature_assembly_seconds=self.last_batch_seconds,
        )
        return loss

    @contextlib.contextmanager
    def prediction_context(self, model, origin):
        """Cache each aggregate once only inside no-grad prediction; test_hmfnet_inference_cache."""
        from torch.utils._pytree import tree_map

        if model.training or torch.is_grad_enabled() or model._aggregate_cache is not None:
            raise ValueError("HMFNet prediction cache requires an uncached eval/no-grad context")

        def sync():
            if self.config.device == "cuda":
                torch.cuda.synchronize()

        sync()
        started = time.perf_counter()
        cached = self._hmfnet_origin(origin)["aggregate"]
        timings = dict(feature_seconds=0.0, transfer_seconds=0.0, forward_seconds=0.0)
        try:
            if cached is not None:
                all_nodes = np.arange(len(self.aggregate_data.target))
                inputs = self._aggregate_tensors(cached, all_nodes, "cpu")
                timings["feature_seconds"] = time.perf_counter() - started
                transfer_started = time.perf_counter()
                inputs = tree_map(
                    lambda v: v.to(self.config.device) if isinstance(v, torch.Tensor) else v, inputs
                )
                sync()
                timings["transfer_seconds"] = time.perf_counter() - transfer_started
                forward_started = time.perf_counter()
                model._aggregate_cache = model.aggregate_forecasts(inputs)
                sync()
                timings["forward_seconds"] = time.perf_counter() - forward_started
            else:
                timings["feature_seconds"] = time.perf_counter() - started
            yield timings
        finally:
            model._aggregate_cache = None

    def diagnostics(self, model, detailed):
        """Bounded stage and gradient records; test_hmfnet_diagnostics checks persisted fields."""
        result = dict(
            loss_components=dict(self.last_components), peak_rss_bytes=_process_peak_rss_bytes()
        )
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            family = name.split(".")[0]
            gradients[family] = gradients.get(family, 0.0) + float(
                parameter.grad.detach().double().square().sum()
            )
        result["gradient_active_by_module"] = {k: v > 0 for k, v in gradients.items()}
        if detailed:
            result.update(
                stages=model.last_diagnostics,
                gradient_l2_by_module={k: math.sqrt(v) for k, v in gradients.items()},
            )
        return result


class PACTRevisionFeatures(RevisionFeatures):
    """PACT decoder token assembly at the existing governed model boundary."""

    def __init__(self, panel, data, config, calendar_path=None, data_spec=None):
        from autoforecast.hmfnet_features import AggregateHistory

        super().__init__(panel, data, config, calendar_path, data_spec)
        self.policy = config.v5
        if self.policy is None:
            raise ValueError("PACT assembly requires an explicit PACT recipe")
        if config.aggregate_history != bool(self.policy.context_levels):
            raise ValueError("PACT aggregate_history must agree with its context streams")
        self.adaptation_reasons.pop("aggregate_history", None)
        self.aggregate_data = AggregateHistory(data, self.policy.history_policy())
        self.numeric_history = tuple(
            name for name in self.history_columns if name not in self.categorical_columns
        )
        self.categorical_history = tuple(
            name for name in self.history_columns if name in self.categorical_columns
        )
        self.pact_history_width = (
            11 + 4 * 6 + 4 * len(self.numeric_history) + 3 * len(self.categorical_history) + 17
        )
        self.future_width = 7 * len(self.known_names) + 34
        self.pact_cache, self.pact_cache_origin = None, None
        self.last_components = {}

    def begin_phase(self, origins, phase):
        """Rebuild only deterministic origin caches when a governed phase changes."""
        record = super().begin_phase(origins, phase)
        self.pact_cache, self.pact_cache_origin = None, None
        record.update(v5_feature_layout=self.feature_layout())
        return record

    def feature_layout(self):
        return dict(
            token_policy=self.policy.token_policy,
            mask_policy=self.policy.mask_policy,
            position_mode=self.policy.position_mode,
            rope_theta=float(self.policy.rope_theta),
            head_law=self.policy.head_law,
            history_width=self.pact_history_width,
            future_width=self.future_width,
            statistic_width=11 + len(self.config.statistic_windows),
            context_levels=self.policy.context_levels,
            max_streams=self.policy.max_streams,
            aggregate_nodes=len(self.aggregate_data.target),
            aggregate_node_keys=self.aggregate_data.node_keys,
            target_estimand=self.panel.target_estimand,
            routing="recorded_lower_bound_to_exact_unobserved_preserved_v1",
        )

    def _memory(self, target, observed, stop, *, physical_state=None, rows=None):
        from autoforecast.pact_features import build_history_tokens

        low = max(0, stop - self.policy.span) if self.policy.span else 0
        shape = (len(target), stop - low)
        numeric, categorical, categorical_raw = [], [], []
        for name in self.numeric_history:
            numeric.append(
                self.data.features[name][rows, low:stop]
                if rows is not None
                else np.full(shape, np.nan)
            )
        for name in self.categorical_history:
            if rows is None:
                categorical.append(np.full(shape, -1, np.int64))
                categorical_raw.append(np.full(shape, None, object))
                continue
            raw = self.data.features[name][rows, low:stop]
            categorical_raw.append(raw)
            vocabulary = self.dynamic_vocab[name]
            mapping = {value: index for index, value in enumerate(vocabulary)}
            ids = np.fromiter(
                (
                    mapping.get(value, len(vocabulary)) if value is not None else -1
                    for value in raw.reshape(-1)
                ),
                np.int64,
            ).reshape(raw.shape)
            categorical.append(ids)
        return build_history_tokens(
            target[:, low:stop],
            observed[:, low:stop],
            np.arange(low, stop),
            origin_stop=stop,
            first_ordinal=self.data.start.toordinal(),
            step_days=self.data.step_days,
            config=self.policy,
            statistic_windows=self.config.statistic_windows,
            physical_state=None if physical_state is None else physical_state[:, low:stop],
            numeric=tuple(numeric),
            categorical=tuple(categorical),
            categorical_raw=tuple(categorical_raw),
        )

    def _pact_origin(self, origin):
        if self.pact_cache_origin == origin and self.pact_cache is not None:
            return self.pact_cache
        stop = self.data.position(origin)
        aggregate = self.aggregate_data
        if len(aggregate.target):
            state = np.where(aggregate.observed, 0, 2).astype(np.int8)
            memory = self._memory(
                aggregate.target,
                aggregate.observed,
                stop,
                physical_state=state,
            )
        else:
            memory = None
        self.pact_cache_origin = origin
        self.pact_cache = {"aggregate": memory}
        return self.pact_cache

    def _leading_prefix(self, origin, rows):
        day = dt.date.fromisoformat(origin)
        sunday = day - dt.timedelta(days=(day.weekday() + 1) % 7)
        values = np.zeros((len(rows), 7), np.float32)
        mask = np.zeros((len(rows), 7), bool)
        for slot in range(7):
            current = sunday + dt.timedelta(days=slot)
            if current > day:
                continue
            offset = (current - self.data.start).days
            if offset < 0 or offset % self.data.step_days:
                continue
            position = offset // self.data.step_days
            if position >= self.data.target.shape[1]:
                continue
            exact = self.data.observed[rows, position]
            values[:, slot] = np.where(exact, self.data.target[rows, position], 0)
            mask[:, slot] = exact
        return values, mask

    def objective_mask(self, origin):
        from autoforecast.pact_features import build_future_segments

        stop = self.data.position(origin)
        states = (
            self.data.physical_state[:, stop : stop + self.data.horizon]
            if self.data.physical_state is not None
            else np.where(self.data.observed[:, stop : stop + self.data.horizon], 0, 2).astype(
                np.int8
            )
        )
        result = build_future_segments(
            np.empty((len(self.data.series_ids), self.data.horizon, 0), np.float32),
            origin=dt.date.fromisoformat(origin),
            target=self.data.target[:, stop : stop + self.data.horizon],
            physical_state=states,
            target_estimand=self.panel.target_estimand,
            position_mode=self.policy.position_mode,
        )
        return result["objective_day_mask"]

    def batch(self, origin, rows, device, targets):
        from autoforecast.pact_features import build_future_segments

        ordinary, daily_target, _daily_observed = super().batch(origin, rows, "cpu", targets)
        stop = self.data.position(origin)
        physical = (
            self.data.physical_state[rows, :stop]
            if self.data.physical_state is not None
            else np.where(self.data.observed[rows, :stop], 0, 2).astype(np.int8)
        )
        focal = self._memory(
            self.data.target[rows, :stop],
            self.data.observed[rows, :stop],
            stop,
            physical_state=physical,
            rows=rows,
        )
        stream_features = [focal["history_features"]]
        stream_exists = [focal["history_exists"]]
        cached = self._pact_origin(origin)["aggregate"]
        if self.policy.context_levels:
            if cached is None:
                raise ValueError("PACT context streams were registered without aggregate history")
            for column in range(len(self.policy.context_levels)):
                nodes = self.aggregate_data.context[rows, column]
                stream_features.append(cached["history_features"][nodes])
                stream_exists.append(cached["history_exists"][nodes])
            for name in ("history_positions", "history_types", "history_widths"):
                if not np.array_equal(focal[name], cached[name]):
                    raise ValueError("PACT focal and aggregate history descriptors are misaligned")
        prefix, prefix_mask = self._leading_prefix(origin, rows)
        known = ordinary["known_future"].numpy()
        if targets:
            states = (
                self.data.physical_state[rows, stop : stop + self.data.horizon]
                if self.data.physical_state is not None
                else np.where(
                    self.data.observed[rows, stop : stop + self.data.horizon], 0, 2
                ).astype(np.int8)
            )
            future = build_future_segments(
                known,
                origin=dt.date.fromisoformat(origin),
                target=daily_target.numpy(),
                physical_state=states,
                target_estimand=self.panel.target_estimand,
                leading_prefix=prefix,
                leading_prefix_mask=prefix_mask,
                position_mode=self.policy.position_mode,
            )
        else:
            future = build_future_segments(
                known,
                origin=dt.date.fromisoformat(origin),
                leading_prefix=prefix,
                leading_prefix_mask=prefix_mask,
                position_mode=self.policy.position_mode,
            )
        model_inputs = PACTModelInputs(
            history_features=torch.tensor(np.stack(stream_features, axis=1)),
            history_exists=torch.tensor(np.stack(stream_exists, axis=1)),
            history_positions=torch.tensor(focal["history_positions"]),
            history_types=torch.tensor(focal["history_types"]),
            future_features=torch.tensor(future["future_features"]),
            future_exists=torch.tensor(future["future_exists"]),
            future_positions=torch.tensor(future["future_positions"]),
            future_types=torch.tensor(future["future_types"]),
            future_active=torch.tensor(future["future_active"]),
            future_day_index=torch.tensor(future["future_day_index"]),
            statistics=torch.tensor(focal["statistics"]),
            attributes=torch.tensor(self.attributes[rows]),
            identity_indices=torch.tensor(self.identity_indices[rows]),
            horizon=self.data.horizon,
        )
        labels = (
            PACTLabels(
                daily_target=daily_target,
                objective_day_mask=torch.tensor(future["objective_day_mask"]),
                target_counts=torch.tensor(future["target_counts"]),
                segment_eligible=torch.tensor(future["segment_eligible"]),
                series_loss_weight=ordinary.get("series_loss_weight"),
            )
            if targets
            else None
        )
        batch = PACTBatch(model_inputs, ordinary["known_future"], labels).to_device(device)
        return (
            batch,
            batch.labels.daily_target if batch.labels is not None else None,
            batch.labels.objective_day_mask if batch.labels is not None else None,
        )

    def scenario_inputs(self, origin, rows, device, overrides, *, known_at):
        """Rebuild all future-token descendants after named campaign-plan edits."""
        from autoforecast.pact_features import build_future_segments

        if not isinstance(overrides, dict) or not overrides:
            raise ValueError("PACT scenario needs at least one named known-future override")
        try:
            knowledge_date = dt.date.fromisoformat(known_at)
        except (TypeError, ValueError) as error:
            raise ValueError("PACT scenario known_at must be a canonical ISO date") from error
        if knowledge_date.isoformat() != known_at or knowledge_date > dt.date.fromisoformat(origin):
            raise ValueError("PACT scenario plan must be known no later than the forecast origin")
        unknown = sorted(set(overrides) - set(self.known_names))
        if unknown:
            raise ValueError(f"PACT scenario names unknown future columns: {unknown}")
        batch, _target, _objective = self.batch(origin, rows, "cpu", False)
        known = batch.known_future.numpy().copy()
        for name, raw in overrides.items():
            value = np.asarray(raw, dtype=np.float32)
            if value.shape != known.shape[:2] or not np.isfinite(value).all():
                raise ValueError(f"PACT scenario override {name!r} must be finite [series,horizon]")
            known[..., self.known_names.index(name)] = value
        for name in overrides:
            if name.endswith("available"):
                value = known[..., self.known_names.index(name)]
                if not np.isin(value, (0, 1)).all():
                    raise ValueError("PACT scenario availability overrides must be binary")
                continue
            available = (
                f"{name}:available"
                if f"{name}:available" in self.known_names
                else f"{name}_available"
                if f"{name}_available" in self.known_names
                else None
            )
            if available is not None and not np.all(
                known[..., self.known_names.index(available)] == 1
            ):
                raise ValueError(
                    f"PACT scenario field {name!r} is not available for every edited cell"
                )
        prefix, prefix_mask = self._leading_prefix(origin, rows)
        future = build_future_segments(
            known,
            origin=dt.date.fromisoformat(origin),
            leading_prefix=prefix,
            leading_prefix_mask=prefix_mask,
            position_mode=self.policy.position_mode,
        )
        rebuilt_layout = {
            "future_exists": torch.tensor(future["future_exists"]),
            "future_positions": torch.tensor(future["future_positions"]),
            "future_types": torch.tensor(future["future_types"]),
            "future_active": torch.tensor(future["future_active"]),
            "future_day_index": torch.tensor(future["future_day_index"]),
        }
        for name, value in rebuilt_layout.items():
            if not torch.equal(value, getattr(batch.inputs, name)):
                raise ValueError(f"PACT scenario unexpectedly changed immutable layout field {name}")
        return replace(
            batch.inputs,
            future_features=torch.tensor(future["future_features"]),
        ).to_device(device)

    def make_model(self):
        from forecast.models.mft.pact import PACTModel

        return PACTModel(
            history_width=self.pact_history_width,
            future_width=self.future_width,
            statistic_width=11 + len(self.config.statistic_windows),
            attribute_width=self.attributes.shape[1],
            identity_cardinalities=self.identity_cardinalities,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.policy.n_layers,
            ff_multiplier=self.policy.ff_multiplier,
            rope_theta=float(self.policy.rope_theta),
            position_mode=self.policy.position_mode,
            max_streams=self.policy.max_streams,
            head_law=self.policy.head_law,
        )

    def mean(self, model, inputs):
        if not isinstance(inputs, PACTBatch):
            raise TypeError("PACT mean expects a PACTBatch")
        return model(**inputs.inputs.as_kwargs()).mean()

    def validation_values(self, model, inputs):
        if not isinstance(inputs, PACTBatch):
            raise TypeError("PACT validation expects a PACTBatch")
        prediction = model(**inputs.inputs.as_kwargs())
        return prediction.mean(), prediction.nll

    def training_loss(self, model, inputs, y, observed, denominator, importance=1.0):
        del y, observed
        if not isinstance(inputs, PACTBatch) or inputs.labels is None:
            raise TypeError("PACT training expects a labeled PACTBatch")
        prediction = model(**inputs.inputs.as_kwargs())
        joint = prediction.joint_nll(inputs.labels.target_counts, inputs.labels.segment_eligible)
        per_series_nll = joint.sum(1)
        outputs = (prediction.mu, prediction.k, prediction.allocation_log)
        if prediction.p_positive is not None:
            outputs += (prediction.p_positive,)
        weights = series_loss_weights(
            self.config.series_loss_weighting,
            per_series_nll,
            revenue_weight=inputs.labels.series_loss_weight,
            distribution_outputs=outputs,
        )
        likelihood = weighted_series_mean(per_series_nll, denominator, weights)
        loss = importance * likelihood
        self.last_components = dict(
            **weights.diagnostics(),
            importance=float(importance),
            objective_days=int(
                (
                    inputs.labels.segment_eligible[..., None]
                    & inputs.inputs.future_active[None, :, :]
                ).sum()
            ),
            joint_nll=float(joint.sum().detach()),
            weighted_nll=float(loss.detach()),
        )
        return loss

    def optimizer_groups(self, model):
        decay, rest = [], []
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and "embedding" not in name and "token_type" not in name:
                decay.append(parameter)
            else:
                rest.append(parameter)
        return [
            {"params": rest, "weight_decay": 0.0},
            {"params": decay, "weight_decay": self.config.weight_decay},
        ]

    def feature_training_state(self):
        return {"schema": "v5_feature_state_v1"}

    def restore_feature_training_state(self, state):
        if state != {"schema": "v5_feature_state_v1"}:
            raise ValueError("invalid PACT feature training state")
        self.cache, self.cache_origin = None, None
        self.pact_cache, self.pact_cache_origin = None, None

    def diagnostics(self, model, detailed):
        del detailed
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                family = name.split(".")[0]
                gradients[family] = gradients.get(family, False) or bool(
                    parameter.grad.detach().abs().sum() > 0
                )
        return {
            "loss_components": dict(self.last_components),
            "gradient_active_by_module": gradients,
            "peak_rss_bytes": _process_peak_rss_bytes(),
        }


def fit_predict_revision(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    calendar_path: str | None = None,
    config,
    levels: tuple[tuple[str, ...], ...] = (),
    out_dir: str,
    replay_checkpoint: str | None = None,
    resume_training_checkpoint: str | None = None,
    data_spec=None,
    training_mode: str = "rolling_backtest",
    validation_panel: Panel | None = None,
    fixed_update_budget: int | None = None,
    evaluation_contract_path: str | None = None,
    evaluation_contract_sha256: str | None = None,
    precision: str = "32-true",
    performance_profile: dict[str, object] | None = None,
    performance_profile_dir: str | None = None,
) -> Forecast:
    """Development-only fresh MFT/SFT/HMFNet/PACT fit or exact-input checkpoint replay.

    No legacy checkpoint can cross this schema. No final protocol gate or candidate registry
    is modified. ``test_revision_fit_replay`` checks immutable publication and exact replay.
    """
    import dataclasses
    import json
    import os
    from contextlib import closing

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast._exact_json import load_exact_json
    from autoforecast.hierarchy_metric import read_forecast_grid
    from autoforecast.protocol import _atomic_write_text
    from autoforecast.rolling_data import PanelArrays, ValidationWindow
    from autoforecast.rolling_training import (
        RollingConfig,
        RollingTrainer,
        RollingTrainingContract,
    )
    from autoforecast.torch_performance import TorchPerformanceProfiler
    from autoforecast.torch_profile_contract import TorchProfileConfig
    from autoforecast.training_checkpoint import TrainingRun, configure_training_runtime
    from autoforecast.training_progress import PipelineProgress, TrainingProgress
    from provenance.lineage import directory_digest

    if not isinstance(config, RollingConfig):
        raise ValueError("revision configuration must be validated before fitting")
    if precision not in ("32-true", "16-mixed"):
        raise ValueError("revision precision must be 32-true or 16-mixed")
    if (evaluation_contract_path is None) != (evaluation_contract_sha256 is None):
        raise ValueError("study evaluation path and digest must be supplied together")
    if evaluation_contract_path is not None and (
        training_mode != "fixed_validation" or validation_panel is None
    ):
        raise ValueError("study checkpoint selection requires a separate fixed validation panel")
    try:
        parsed_origin = dt.date.fromisoformat(origin)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"revision origin must be canonical ISO YYYY-MM-DD, got {origin!r}"
        ) from exc
    if origin != parsed_origin.isoformat():
        raise ValueError(f"revision origin must be canonical ISO YYYY-MM-DD, got {origin!r}")
    if type(horizon) is not int or horizon < 1:
        raise ValueError(f"revision horizon must be a positive integer, got {horizon!r}")
    if replay_checkpoint and resume_training_checkpoint:
        raise ValueError("inference replay and training resume are mutually exclusive")
    profile_config = (
        None
        if performance_profile is None
        else TorchProfileConfig.from_record(performance_profile)
    )
    if (profile_config is None) != (performance_profile_dir is None):
        raise ValueError("Torch profiler config and output directory must be supplied together")
    if profile_config is not None and replay_checkpoint is not None:
        raise ValueError("Torch performance profiling is only defined for model training")
    runtime = configure_training_runtime(config.device, config.torch_threads)
    data_spec = data_spec or revision_data_spec(panel, horizon, config)
    destination = pathlib.Path(out_dir).resolve()
    if destination.exists() and not resume_training_checkpoint:
        raise ValueError("revision output already exists; use a fresh directory")
    root = pathlib.Path(__file__).resolve().parents[2]
    from autoforecast.executable_source import (
        SOURCE_MANIFEST_FILENAME,
        validate_executable_source_snapshot,
    )
    from autoforecast.project import installed_library_identity

    build_manifest = root / "autoforecast/_build.json"
    library_identity = installed_library_identity().as_record() if build_manifest.is_file() else None
    snapshot_manifest = root / SOURCE_MANIFEST_FILENAME
    if snapshot_manifest.is_file():
        snapshot = validate_executable_source_snapshot(load_exact_json(snapshot_manifest))
        if snapshot["root"] != str(root):
            raise ValueError("revision source snapshot belongs to a different root")
        if snapshot["manifest"]["schema"] == "autoforecast.installed-executable-source.v1":
            library_identity = installed_library_identity().as_record()
            if snapshot["manifest"]["library"] != library_identity:
                raise ValueError("revision source snapshot differs from the installed library")
            build_manifest = snapshot_manifest
    code_directories = [root / name for name in ("autoforecast", "forecast", "spec", "provenance")]
    sources = [
        pathlib.Path(panel.path).resolve(),
        *([pathlib.Path(validation_panel.path).resolve()] if validation_panel is not None else []),
        *([pathlib.Path(calendar_path).resolve()] if calendar_path else []),
        *code_directories,
        root / "scripts/run_multibranch_revision.py",
        *([build_manifest] if library_identity is not None
          else [root / "pyproject.toml", root / "uv.lock"]),
    ]
    if any(
        destination == p or p in destination.parents or destination in p.parents for p in sources
    ):
        raise ValueError("revision output overlaps an input/source")
    if replay_checkpoint:
        replay_directory = pathlib.Path(replay_checkpoint).resolve().parent
        if (replay_directory / "COMPLETE.json").is_file() and destination.is_relative_to(
            replay_directory
        ):
            raise ValueError("replay output overlaps the completed source directory")
    before = {
        str(p): directory_digest(p, suffixes=(".py",) if p in code_directories else ())
        for p in sources
    }
    calendar_contract, calendar_events, calendar_features = {}, {}, ()
    if calendar_path:
        calendar_table = pq.read_table(calendar_path)
        parsed_calendar_contract = _revision_calendar_contract(calendar_table)
        if parsed_calendar_contract is not None:
            calendar_contract = parsed_calendar_contract
            calendar_events = _revision_event_records(
                calendar_table.to_pylist(),
                calendar_contract,
                config.variant,
                dt.date.fromisoformat(origin)
                - dt.timedelta(days=2 * horizon * period_days(panel.frequency)),
            )
            calendar_features = (
                calendar_contract["variants"]
                .get(config.variant, {})
                .get("known_future", calendar_contract["known_future"])
            )
    calendar_group = (
        (config.calendar_group_column or calendar_contract.get("group_column"))
        if calendar_contract.get("group_col") and (calendar_events or calendar_features)
        else None
    )
    factor_columns = (
        (config.factor_columns or (data_spec.factor_group_col,))
        if config.factor_rank and data_spec.hierarchy and data_spec.factor_group_col
        else ()
    )
    metadata_columns = tuple(
        dict.fromkeys(
            (
                *config.identity_columns,
                *factor_columns,
                *(config.aggregate_columns if config.aggregate_history else ()),
                *config.baseline_columns,
                *(config.event_pool_columns if calendar_events else ()),
                *((calendar_group,) if calendar_group else ()),
                *(name for level in levels for name in level),
                *(
                    name
                    for level in (
                        (*config.v3.context_levels, *config.v3.expert_levels)
                        if config.v3
                        else config.v5.context_levels
                        if config.v5
                        else ()
                    )
                    for name in level
                ),
            )
        )
    )
    preparation = PipelineProgress("Model data preparation", 2, phase_key="data")
    with preparation.step("Load and validate the governed panel arrays"):
        data = PanelArrays.read(
            panel,
            origin,
            horizon,
            max_bytes=config.max_array_bytes,
            data_spec=data_spec,
            metadata_columns=metadata_columns,
            observation_contract=config.observation_contract,
            route_lower_bound_to_exact=config.variant == "v5",
        )
    feature_type = (
        PACTRevisionFeatures
        if config.variant == "v5"
        else HMFNetRevisionFeatures
        if config.variant == "v3"
        else RevisionFeatures
    )
    with preparation.step(f"Assemble the {model_name(config.variant)} model features"):
        features = feature_type(panel, data, config, calendar_path, data_spec)
    validation_window = None
    if validation_panel is not None:
        if evaluation_contract_path is not None:
            from autoforecast.study_validation import StudyValidation

            assert evaluation_contract_sha256 is not None
            study = StudyValidation.read(
                evaluation_contract_path, evaluation_contract_sha256,
                panel=validation_panel, origin=origin, horizon=horizon, series_ids=data.series_ids,
            )
            validation_window = ValidationWindow.from_study(
                origin, study, route_lower_bound_to_exact=config.variant == "v5",
            )
        else:
            validation_window = ValidationWindow.read(
                validation_panel,
                origin,
                horizon,
                data.series_ids,
                route_lower_bound_to_exact=config.variant == "v5",
            )
        if config.variant == "v5":
            from autoforecast.pact_features import build_future_segments

            assert validation_window.physical_state is not None and config.v5 is not None
            segments = build_future_segments(
                np.empty((len(data.series_ids), horizon, 0), np.float32),
                origin=dt.date.fromisoformat(origin),
                target=validation_window.target,
                physical_state=validation_window.physical_state,
                target_estimand=panel.target_estimand,
                position_mode=config.v5.position_mode,
            )
            validation_window = dataclasses.replace(
                validation_window,
                objective_mask=segments["objective_day_mask"],
            )
    training_contract = RollingTrainingContract(
        mode=training_mode,
        fixed_validation=validation_window,
        fixed_update_budget=fixed_update_budget,
    )
    trainer = RollingTrainer(
        data,
        config,
        levels,
        features.groups,
        features.make_model,
        features.batch,
        features.training_loss,
        features.mean,
        features.optimizer_groups,
        features.begin_phase,
        features.prediction_context if config.variant == "v3" else None,
        features.diagnostics if config.variant in {"v2", "v3", "v5"} else None,
        validation_values=(
            features.validation_values if config.variant in {"v2", "v3", "v5"} else None
        ),
        feature_training_state=features.feature_training_state,
        restore_feature_training_state=features.restore_feature_training_state,
        objective_mask=features.objective_mask if config.variant == "v5" else None,
        training_contract=training_contract,
        precision=precision,
    )
    binding = dict(
        schema=(
            "multibranch_revision_v5"
            if config.variant == "v5"
            else "multibranch_revision_v3"
            if config.variant == "v3"
            else "multibranch_revision_v2"
        ),
        variant=config.variant,
        origin=origin,
        horizon=horizon,
        config=dataclasses.asdict(config),
        training_contract=training_contract.as_record(),
        **({"precision": precision} if precision != "32-true" else {}),
        inputs=before,
        **({"library": library_identity} if library_identity is not None else {}),
        panel=dataclasses.asdict(panel),
        series_ids=list(data.series_ids),
        levels=[list(level) for level in levels],
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        pyarrow_version=pa.__version__,
        training_runtime=runtime,
        data_spec=dataclasses.asdict(data_spec),
        effective_data_spec=dataclasses.asdict(features.spec),
        resolved_build=features.resolved.as_record(),
        build_hash=features.resolved.build_hash,
        effective_assembly=dict(
            identity_columns=features.identity_columns,
            adaptation_reasons=features.adaptation_reasons,
            factor_columns=features.factor_columns,
            aggregate_columns=features.aggregate_columns,
            event_pool_columns=features.event_pool_columns,
            baseline_columns=features.baseline_columns,
            commercial=features.has_commercial,
            calendar=features.calendar_contract,
            **(
                {f"{config.variant}_layout": features.feature_layout()}
                if config.variant in {"v3", "v5"}
                else {}
            ),
        ),
        sales_observation_contract=data.observation_contract,
        evidence_role="retrospective_development_only",
    )
    binding = json.loads(json.dumps(binding, allow_nan=False))  # One JSON/Torch representation.
    replay_binding: tuple[pathlib.Path, pathlib.Path, dict] | None = None

    def source_check():
        current = {
            str(p): directory_digest(p, suffixes=(".py",) if p in code_directories else ())
            for p in sources
        }
        if current != before or (
            library_identity is not None
            and installed_library_identity().as_record() != library_identity
        ):
            raise ValueError("input/source changed during run; refusing checkpoint/publication")

    with (
        TrainingRun(
            destination,
            binding,
            device=config.device,
            resume=resume_training_checkpoint,
            source_check=source_check,
            every_updates=config.checkpoint_every_updates,
            every_seconds=config.checkpoint_every_seconds,
            keep_last=config.checkpoint_keep_last,
        ) as run,
        closing(TrainingProgress(config.progress)) as progress,
    ):
        if replay_checkpoint is None:
            if profile_config is not None:
                assert performance_profile_dir is not None
                trainer.performance_profiler = TorchPerformanceProfiler(
                    profile_config,
                    performance_profile_dir,
                    device=config.device,
                )
                progress.stage(
                    "Torch performance profile enabled "
                    f"({config.device}; {profile_config.capture_steps} sampled steps per phase)",
                    force=True,
                )
                run.event(
                    "torch_performance_profile_enabled",
                    device=config.device,
                    activities=[
                        activity.name.lower()
                        for activity in trainer.performance_profiler.activities
                    ],
                    config=profile_config.as_record(),
                )
            model, training = trainer.fit(run.log, run=run, progress=progress)
        else:
            progress.stage("replay loading completed inference checkpoint", force=True)
            source_checkpoint = pathlib.Path(replay_checkpoint).resolve()
            source_directory = source_checkpoint.parent
            completion_path = source_directory / "COMPLETE.json"
            if not completion_path.is_file() or source_checkpoint.name != "model.pt":
                raise ValueError("replay requires model.pt from a completed revision directory")
            source_record = load_exact_json(completion_path)
            if (
                not isinstance(source_record, dict)
                or not _warm_start_values_equal_exact(source_record.get("binding"), binding)
                or source_record.get("checkpoint_sha256") != directory_digest(source_checkpoint)
                or source_record.get("forecast_sha256")
                != directory_digest(source_directory / "forecast.parquet")
            ):
                raise ValueError("completed checkpoint/forecast digest or binding mismatch")
            saved = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
            if set(saved) != {
                "binding",
                "training",
                "state_dict",
            } or not _warm_start_values_equal_exact(saved["binding"], binding):
                raise ValueError("checkpoint schema/config/input/source binding mismatch")
            torch.set_num_threads(config.torch_threads)
            restored_admission = features.begin_phase(
                tuple(saved["training"]["refit_training_origins"]), "refit"
            )
            if not _warm_start_values_equal_exact(
                restored_admission, saved["training"]["phase_admission"]["refit"]
            ):
                raise ValueError(
                    "checkpoint event admission differs from the bound training windows"
                )
            model = features.make_model().to(config.device)
            model.load_state_dict(saved["state_dict"], strict=True)
            training = saved["training"]
            replay_binding = (source_directory, source_checkpoint, source_record)
        publication = PipelineProgress("Forecast publication", 3, phase_key="forecast")
        with publication.step("Generate the complete-population forecast"):
            prediction_started = time.perf_counter()
            prediction = trainer.predict(model, origin)
            prediction_seconds = time.perf_counter() - prediction_started
        if prediction.shape != (len(data.series_ids), horizon):
            raise ValueError("revision did not emit the complete forecast grid")
        with publication.step("Verify forecast shape, replay, and immutable sources"):
            if replay_binding is not None:
                source_directory, source_checkpoint, source_record = replay_binding
                expected = read_forecast_grid(
                    str(source_directory / "forecast.parquet"),
                    data.series_ids,
                    origin,
                    horizon,
                    series_col="series_id",
                    time_col="date",
                    value_col="yhat",
                    frequency=panel.frequency,
                )
                if (
                    not np.array_equal(prediction, expected)
                    or source_record["checkpoint_sha256"] != directory_digest(source_checkpoint)
                    or source_record["forecast_sha256"]
                    != directory_digest(source_directory / "forecast.parquet")
                ):
                    raise ValueError("replayed forecast differs from the completed source")
            after = {
                str(p): directory_digest(p, suffixes=(".py",) if p in code_directories else ())
                for p in sources
            }
            if before != after:
                raise ValueError("input/source changed during run; refusing complete publication")
        with publication.step("Write and seal the inference checkpoint and forecast"):
            torch.save(
                dict(
                    binding=binding,
                    training=training,
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                ),
                run._path("model.pt"),
            )
            dates = [data.date_at(data.target.shape[1] + i) for i in range(horizon)]
            table = pa.table(
                {
                    "series_id": np.repeat(data.series_ids, horizon),
                    "date": dates * len(data.series_ids),
                    "yhat": prediction.reshape(-1),
                }
            )
            pq.write_table(table, run._path("forecast.parquet"))
            for artifact in (destination / "model.pt", destination / "forecast.parquet"):
                with artifact.open("rb") as stream:
                    os.fsync(stream.fileno())
            forecast = Forecast(
                adapter=REVISION_MODELS[config.variant],
                path=str(destination / "forecast.parquet"),
                origin=origin,
                horizon=horizon,
                n_series=len(data.series_ids),
                disposition={"emitted": len(data.series_ids) * horizon},
                frequency=panel.frequency,
                target_estimand=panel.target_estimand,
            ).validate()
            record = dict(
                binding=binding,
                training=training,
                known_names=features.known_names,
                events=features.event_names,
                calendar_availability=features.calendar_contract,
                checkpoint_sha256=directory_digest(destination / "model.pt"),
                forecast_sha256=directory_digest(destination / "forecast.parquet"),
                replay_source=replay_checkpoint,
            )
            record["execution_checks"] = dict(
                prediction_seconds=prediction_seconds,
                forecast_shape=list(prediction.shape),
                forecast_cells=int(prediction.size),
                minimum=float(prediction.min()),
                maximum=float(prediction.max()),
                mean=float(prediction.mean()),
                finite=bool(np.isfinite(prediction).all()),
                nonnegative=bool((prediction >= 0).all()),
                replay_exactly_equal=True if replay_binding is not None else None,
                last_prediction_chunk_stages=model.last_diagnostics
                if config.variant == "v3"
                else {},
                performance_profile=(
                    None if profile_config is None else profile_config.as_record()
                ),
            )
            source_check()
            _atomic_write_text(
                run._path("COMPLETE.json"),
                json.dumps(record, indent=2, allow_nan=False) + "\n",
            )
            run.event(
                "complete",
                forecast_sha256=record["forecast_sha256"],
                replay=bool(replay_checkpoint),
                resumed=bool(resume_training_checkpoint),
            )
    return forecast


def fit_predict_revision_adapter(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession | None,
    out_dir: str | pathlib.Path,
    rolling_config: dict[str, object],
    expected_variant: str,
    checkpoint_out: str | pathlib.Path,
    training_metrics_path: str | pathlib.Path,
    training_log_dir: str | pathlib.Path | None = None,
    performance_profile: dict[str, object] | None = None,
    events: tuple[tuple[str, str, str | None], ...] = (),
    event_max_support: int = 14,
    device: str | None = None,
    batch_size: int = 512,
    gradient_microbatch_size: int | None = None,
    patience: int = 3,
    early_stopping_absolute_min_delta: float = 0.0,
    early_stopping_relative_min_delta: float = 0.0,
    quiet_epoch_logging: bool = False,
    training_mode: str = "rolling_backtest",
    validation_data_path: str | None = None,
    fixed_update_budget: int | None = None,
    validation_metric: str | None = None,
    precision: str = "32-true",
    evaluation_contract_path: str | None = None,
    evaluation_contract_sha256: str | None = None,
) -> Forecast:
    """Run a registered revision while retaining native evidence in the run bundle.

    The revision trainer owns feature assembly after the governed panel is materialized; Spark is
    present only because all registry adapters share one call protocol. Generic calendar-event
    tuples cannot be losslessly converted to the revision calendar contract, so an event-bearing
    request is rejected instead of silently disabling that branch. RMSE and WRMSSE validation
    remain shared trainer capabilities rather than architecture-specific scoring code.
    """
    del quiet_epoch_logging
    from autoforecast.rolling_training import RollingConfig
    from autoforecast.torch_profile_contract import TorchProfileConfig

    if expected_variant not in {"v2", "v3", "v5"}:
        raise ValueError(f"unsupported registered revision {expected_variant!r}")
    if type(rolling_config) is not dict:
        raise ValueError("registered revision requires one resolved rolling_config mapping")
    if rolling_config.get("variant") != expected_variant:
        raise ValueError("registered revision model and rolling configuration disagree")
    if events:
        raise ValueError(
            "revision SFT/HMFNet/PACT requires a versioned calendar input; generic event tuples would "
            "silently lose scope, known-at and episode metadata"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("registered revisions currently support device cpu or cuda")
    if validation_metric not in {None, "rmse", "wrmsse"}:
        raise ValueError("registered revisions support validation metric rmse or wrmsse")
    profile_config = (
        None
        if performance_profile is None
        else TorchProfileConfig.from_record(performance_profile)
    )
    if profile_config is not None and training_log_dir is None:
        raise ValueError("Torch profiling requires the attempt training_log_dir")
    if gradient_microbatch_size not in {None, batch_size}:
        raise ValueError(
            "registered revisions have no gradient accumulation; set microbatch equal "
            "to batch_size or null"
        )
    if any(
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in (early_stopping_absolute_min_delta, early_stopping_relative_min_delta)
    ):
        raise ValueError("revision early-stopping deltas must be finite non-negative numbers")
    config = RollingConfig(
        **{
            **rolling_config,
            "device": device,
            "batch_size": batch_size,
            "patience": patience,
            "event_max_support": event_max_support,
            "early_stopping_absolute_min_delta": float(early_stopping_absolute_min_delta),
            "early_stopping_relative_min_delta": float(early_stopping_relative_min_delta),
            **({} if validation_metric is None else {"selection_metric": validation_metric}),
        }
    )
    if os.environ.get("AUTOFORECAST_ISOLATED_WORKER") == "1":
        if spark is not None:
            _release_spark_for_torch_training(spark)
    elif spark is None:
        raise ValueError("direct revision adapter calls require a caller-owned Spark session")
    forecast_path = pathlib.Path(out_dir)
    evidence_dir = forecast_path.with_name(f"{forecast_path.name}-revision-evidence")
    validation_panel = (
        None
        if validation_data_path is None
        else replace(panel, path=str(pathlib.Path(validation_data_path).resolve()))
    )
    if validation_metric == "wrmsse" and evaluation_contract_path is None:
        from autoforecast.benchmarks.m5 import M5_LEVELS

        validation_levels = M5_LEVELS
    else:
        validation_levels = ()
    forecast = fit_predict_revision(
        panel,
        origin,
        horizon,
        config=config,
        levels=validation_levels,
        out_dir=str(evidence_dir),
        training_mode=training_mode,
        validation_panel=validation_panel,
        fixed_update_budget=fixed_update_budget,
        evaluation_contract_path=evaluation_contract_path,
        evaluation_contract_sha256=evaluation_contract_sha256,
        precision=precision,
        performance_profile=(None if profile_config is None else profile_config.as_record()),
        performance_profile_dir=(
            None
            if profile_config is None
            else str(pathlib.Path(training_log_dir).resolve() / "performance-profile")
        ),
    )
    forecast_path.mkdir(parents=True, exist_ok=False)
    checkpoint_path = pathlib.Path(checkpoint_out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = pathlib.Path(training_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(evidence_dir / "forecast.parquet", forecast_path / "part-00000.parquet")
    shutil.copyfile(evidence_dir / "model.pt", checkpoint_path)
    shutil.copyfile(evidence_dir / "training.jsonl", metrics_path)
    return replace(forecast, path=str(forecast_path)).validate()


def _ordered_levels(vocab_map: dict[str, int]) -> list[str]:
    """A vocabulary map's levels in embedding-row order -- the list form the checkpoint stores."""
    return [level for level, _ in sorted(vocab_map.items(), key=lambda kv: kv[1])]


def _factor_dynamics_for_build(
    resolved: ResolvedBuild | None,
    factor_enabled: bool,
) -> str | None:
    """Translate the resolved semantic choice into the implemented factor parameterisation."""
    if not factor_enabled:
        return None
    if resolved is None:
        return "ar1"
    # The spec's `None` means independent daily factor draws. That is precisely the implemented
    # IID boundary, not an absent cross-series factor (topology/rank decide absence).
    if resolved.factor_dynamics is None:
        return "iid"
    if resolved.factor_dynamics not in ("ar1", "iid"):
        raise OursError(
            f"resolved factor_dynamics={resolved.factor_dynamics!r} is not implemented; "
            "expected 'ar1', 'iid', or None (the IID semantic alias)"
        )
    return resolved.factor_dynamics


def _validate_resolved_topology(
    resolved: ResolvedBuild | None,
    grouping: str | None,
) -> None:
    """Refuse a build hash whose factor topology disagrees with the runtime Panel roles."""
    if resolved is None:
        return
    if resolved.topology not in ("bottom_up", "bottom_up_factor"):
        raise OursError(
            f"resolved topology={resolved.topology!r} is not implemented by this adapter"
        )
    resolved_factor = resolved.topology == "bottom_up_factor"
    role_factor = grouping is not None
    if resolved_factor != role_factor:
        raise OursError(
            f"resolved topology={resolved.topology!r} but runtime factor_group_col is "
            f"{grouping!r}; the build hash and constructed dependence topology would disagree"
        )
    if resolved_factor and resolved.factor_rank < 1:
        raise OursError(f"resolved bottom_up_factor has invalid rank {resolved.factor_rank}")
    if not resolved_factor and resolved.factor_rank != 0:
        raise OursError(
            f"resolved bottom_up records factor_rank={resolved.factor_rank}, expected 0"
        )
    _factor_dynamics_for_build(resolved, resolved_factor)


def _open_training_log(
    path: str | pathlib.Path | None,
    metrics_path: str | pathlib.Path | None = None,
    epoch_observer: Callable[[int, Mapping[str, float]], None] | None = None,
):
    """Open optional, per-epoch telemetry without making it part of model numerics."""
    if path is None and metrics_path is None and epoch_observer is None:
        return None
    from autoforecast.observability import open_training_log

    return open_training_log(path, metrics_path=metrics_path, epoch_observer=epoch_observer)


def _process_peak_rss_bytes() -> int:
    """Return this training process's sampled peak RSS in bytes on macOS and Linux."""
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _fit_sharded(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark,
    out_dir,
    predicates: list[str | None],
    encoder_days,
    d_model,
    n_draws,
    epochs,
    n_paths,
    paths_sample,
    seed,
    events,
    event_max_support,
    event_windows,
    train_origins,
    target_origins,
    train_stride,
    train_windows,
    cycles,
    per_series_seasonality,
    demand_dropout,
    covariate_dropout,
    price_penalty_weight,
    price_response_mode,
    price_occurrence_slope_max,
    price_size_slope_max,
    aggregate_trajectory_mode,
    inference_factor_draws,
    point_factor_draws,
    point_factor_method,
    grouping,
    product,
    store,
    category,
    resolved,
    head_kind,
    n_heads,
    patch_size,
    trunk_stages,
    decoder_layers,
    history_attention_layers,
    commercial_attention_layers,
    event_kernel_layers,
    static_selection,
    text_embeddings,
    checkpoint_out,
    device,
    precision,
    batch_size,
    gradient_microbatch_size,
    val_fraction,
    patience,
    lr,
    work_dir: pathlib.Path,
    n_total: int,
    max_series_per_shard: int,
    training_log_dir: str | pathlib.Path | None,
    training_metrics_path: str | pathlib.Path | None,
    epoch_observer: Callable[[int, Mapping[str, float]], None] | None,
) -> Forecast:
    """The bounded-memory path: assemble per shard, stream shard files through training,
    forecast and write per shard. One shard at a time is ever resident.

    What is global, deliberately: the series vocabulary (embedding rows are global by
    definition -- an O(n)-strings structure, categorically below the tensor blow-up the shards
    bound), the level vocabularies (small), the category priors (measured in Spark so a cold
    start inherits its category, not its shard), and the optimiser. Everything sized
    [series x days x channels] lives one shard at a time, on disk between uses.

    Recorded limitations: the checkpoint stores vocabularies, predicates and the shard manifest
    instead of full per-series conditioning tensors. `predict_from_checkpoint` can serve it, but
    currently rebuilds that serving panel in one assembly; bounded FIT memory therefore does not
    imply bounded checkpoint-SERVING memory. Optional path, diagnostics/architecture, and text
    outputs are refused by the dispatcher before this function starts; and the peak-memory claim
    at the TARGET catalogue still requires measurement -- this makes the mechanism real, not the
    number.
    """
    series_vocab = _series_vocabulary(panel, spark)
    position = {sid: i for i, sid in enumerate(series_vocab)}
    n_series = len(series_vocab)

    vocab_maps: dict[str, dict[str, int]] = {}
    counts: dict[str, int] = {}
    for role, column in (("grouping", grouping), ("product", product), ("store", store)):
        if column is None:
            vocab_maps[role] = {}
            counts[role] = 1
        else:
            levels = _vocabulary_spark(panel, spark, column)
            vocab_maps[role] = {level: i for i, level in enumerate(levels)}
            counts[role] = len(levels)

    # Always measure the fallback prior over the whole panel. With no category role, passing
    # None made `_assemble` recompute a different "panel-wide" prior inside every shard; sharding
    # then changed both the fitted model and checkpoint replay rather than only bounding memory.
    level_priors = _level_priors(panel, spark, origin, encoder_days, category)
    train_level_priors = [
        _level_priors(panel, spark, train_origin, encoder_days, category)
        for train_origin in train_origins
    ]

    identity_meta: dict = {
        "static": panel.static,
        "category_col": category,
        "grouping": grouping,
        "product": product,
        "store": store,
    }
    shard_expr = None
    if len(predicates) > 1 and predicates[0]:
        shard_expr = predicates[0].rsplit(" = ", 1)[0]
    if panel.static:
        identity_path = static_identity.build(
            panel.path,
            spark,
            str(work_dir / "_identity_global"),
            series_col=panel.series_col,
            attribute_cols=panel.static,
            shard_expr=shard_expr,
        )
        vocabularies = {
            c: static_identity.vocabulary(identity_path, c, spark) for c in panel.static
        }
        for column, vocabulary in vocabularies.items():
            if len(vocabulary) > MAX_ONE_HOT:
                raise OursError(
                    f"static attribute {column!r} has {len(vocabulary)} levels, above "
                    f"MAX_ONE_HOT={MAX_ONE_HOT}"
                )
        one_hot = [f"{c}={level}" for c in panel.static for level in vocabularies[c]]
        identity_meta.update(
            identity_path=str(identity_path),
            vocabularies=vocabularies,
            one_hot_names=tuple(one_hot),
        )
        n_attributes = len(one_hot)
        attribute_widths = tuple(len(vocabularies[c]) for c in panel.static)
    else:
        identity_meta.update(identity_path=None, vocabularies={}, one_hot_names=("_no_static",))
        n_attributes = 1
        attribute_widths = ()
    if panel.static_reals:
        # Static reals are properties of SERIES, not observations.  Averaging panel rows makes a
        # long-history series count more than a short-history series and therefore makes forced
        # sharding change the standardisation used by the single route.  Collapse to the same
        # one-value-per-series population first.
        per_series_reals = (
            spark.read.parquet(panel.path)
            .groupBy(panel.series_col)
            .agg(*[F.first(F.col(c).cast("double")).alias(c) for c in panel.static_reals])
        )
        stats_frame = per_series_reals.agg(
            *[F.avg(F.col(c).cast("double")).alias(f"_m_{c}") for c in panel.static_reals],
            *[F.stddev_pop(F.col(c).cast("double")).alias(f"_s_{c}") for c in panel.static_reals],
        ).collect()[0]
        identity_meta["real_stats"] = {
            c: (float(stats_frame[f"_m_{c}"] or 0.0), float(stats_frame[f"_s_{c}"] or 0.0))
            for c in panel.static_reals
        }
        n_attributes += len(panel.static_reals)
        if attribute_widths:
            attribute_widths = (*attribute_widths, *(1 for _ in panel.static_reals))
    if not static_selection:
        attribute_widths = ()

    shard_dir = work_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifests, example_series_all, informative_by_example_all, n_examples = [], [], [], 0
    layout = None
    declared_drivers: frozenset[str] | None = None
    max_shard_series = 0
    from forecast.models.mft.heads import (
        informative_mask,
        soft_informative_weight,
    )

    for k, predicate in enumerate(predicates):
        bundle = _shard_universe(
            panel,
            spark,
            origin,
            horizon,
            encoder_days=encoder_days,
            events=events,
            event_max_support=event_max_support,
            event_windows=event_windows,
            cycles=cycles,
            work_dir=work_dir,
            shard_id=k,
            predicate=predicate,
            level_priors=level_priors,
            train_level_priors=train_level_priors,
            series_vocab_position=position,
            vocab_maps=vocab_maps,
            attribute_meta=identity_meta,
            text_embeddings=text_embeddings,
            n_draws=n_draws,
            train_origins=train_origins,
            target_origins=target_origins,
            target_estimand=panel.target_estimand,
            status_prob_cols=panel.status_prob_cols,
        )
        if bundle.get("empty"):
            manifests.append(
                {"shard_id": k, "empty": True, "series_ids": bundle.get("series_ids", [])}
            )
            continue
        if layout is None:
            layout = bundle["known_names"]
            declared_drivers = frozenset(bundle["declared_known_drivers"])
        elif tuple(bundle["known_names"]) != tuple(layout):
            raise OursError(
                f"shard {k} produced known-future columns {bundle['known_names']} against "
                f"{layout}; shards of one panel must share one layout"
            )
        elif frozenset(bundle["declared_known_drivers"]) != declared_drivers:
            raise OursError(
                f"shard {k} declared known-future drivers "
                f"{bundle['declared_known_drivers']} against {sorted(declared_drivers)}; "
                "shards of one panel must share one role contract"
            )
        if bundle["target_probs"] is None:
            weights = informative_mask(bundle["target"], bundle["target_state"]).float()
        else:
            weights = soft_informative_weight(bundle["target"], bundle["target_probs"])
        informative_by_example_all.append(weights.reshape(weights.shape[0], -1).sum(dim=1))
        n_examples += int(bundle["target"].shape[0])
        example_series_all.append(bundle["example_series"])
        max_shard_series = max(max_shard_series, len(bundle["series_ids"]))
        file = shard_dir / f"shard_{k}.pt"
        torch.save(bundle, str(file))
        manifests.append(
            {
                "shard_id": k,
                "file": str(file),
                "n_series": len(bundle["series_ids"]),
                "n_examples": int(bundle["target"].shape[0]),
                "plan_provenance": bundle["plan_provenance"],
            }
        )
    live = [m for m in manifests if not m.get("empty")]
    if not live:
        raise OursError(f"no shard has a full {encoder_days}-day encoder window at {origin}")
    if layout is None or declared_drivers is None:
        raise OursError("live shards produced no known-future role layout")
    print(
        f"  sharded: {len(live)} live shard(s) of {len(predicates)}, "
        f"largest {max_shard_series:,} series, {n_examples:,} examples",
        flush=True,
    )

    torch.manual_seed(seed)
    commercial_indices = _commercial_indices_from_roles(tuple(layout), declared_drivers)
    branch_commercial = (
        int(commercial_indices["promo_index"] is not None)
        if price_response_mode == "monotone_raw_v1"
        else (3 if (panel.price_col or panel.promo_col or panel.plan) else 0)
    )
    model_config = dict(
        d_model=d_model,
        encoder_days=encoder_days,
        n_attributes=n_attributes,
        n_series=n_series + 1,
        n_known=len(layout),
        horizon=horizon,
        n_commercial=branch_commercial,
        n_events=len(dict.fromkeys(name for name, _, _ in events)),
        n_stores=counts["store"] + 1,
        factor_rank=(
            (resolved.factor_rank if resolved is not None else 8) if grouping is not None else 0
        ),
        factor_dynamics=_factor_dynamics_for_build(resolved, grouping is not None),
        embedding_residual_dim=(resolved.embedding_residual_dim if resolved is not None else 16),
        event_max_support=event_max_support,
        event_windows=(
            None
            if event_windows is None
            else tuple(
                tuple(event_windows.get(name, (-event_max_support, event_max_support)))
                for name in dict.fromkeys(name for name, _, _ in events)
            )
        ),
        head_kind=head_kind,
        n_heads=n_heads,
        patch_size=patch_size,
        trunk_stages=trunk_stages,
        decoder_layers=decoder_layers,
        history_attention_layers=history_attention_layers,
        commercial_attention_layers=commercial_attention_layers,
        event_kernel_layers=event_kernel_layers,
        annual_index=tuple(
            layout.index(c)
            for c in cycle_columns("annual", next((h for n, _, h in cycles if n == "annual"), 0))
        )
        if any(n == "annual" for n, _, _ in cycles)
        else (),
        per_series_seasonality=per_series_seasonality,
        attribute_widths=attribute_widths,
        demand_dropout=demand_dropout,
        price_response_mode=price_response_mode,
        price_occurrence_slope_max=price_occurrence_slope_max,
        price_size_slope_max=price_size_slope_max,
        aggregate_trajectory_mode=aggregate_trajectory_mode,
        **commercial_indices,
        event_distance_index=tuple(
            layout.index(f"{name}_distance")
            for name in dict.fromkeys(name for name, _, _ in events)
        ),
        event_present_index=tuple(
            layout.index(f"{name}_present") for name in dict.fromkeys(name for name, _, _ in events)
        ),
        n_products=(counts["product"] + 1 if product is not None else 0),
        text_dim=0,
    )
    model = CustomModel(**model_config)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    print(
        f"  sharded training on {dev} ({epochs} epochs max, logical batch {batch_size}, "
        f"gradient microbatch {gradient_microbatch_size})",
        flush=True,
    )
    groups, decayed = parameter_groups(model)
    print(f"  weight decay on the residual tables only ({len(decayed)})", flush=True)
    opt = torch.optim.AdamW(groups, lr=lr)
    scaler = torch.amp.GradScaler(dev.type) if precision == "16-mixed" else None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    training_log = _open_training_log(
        training_log_dir,
        training_metrics_path,
        epoch_observer,
    )

    generator = torch.Generator().manual_seed(seed)
    all_examples = torch.cat(example_series_all)
    informative_by_example = torch.cat(informative_by_example_all)
    train_rows_global, val_rows_global = split_by_series(all_examples, val_fraction, generator)
    train_set = set(train_rows_global.tolist())
    # Identity evidence means a row can affect the OPTIMISED objective, not merely that it was
    # assigned to the training split. An all-unobserved target contributes exactly zero
    # likelihood; preserving its random residual as "trained" would defeat pooling. A row with
    # a perturbable price can still affect the explicit monotonicity term and is conservatively
    # counted as evidence.
    evidence_series: set[int] = set()
    evidence_product: set[int] = set()
    evidence_store: set[int] = set()
    train_price_eligible_total = 0
    train_price_reference_total = 0
    evidence_offset = 0
    for manifest in manifests:
        offset = evidence_offset
        evidence_offset += int(manifest.get("n_examples", 0))
        if manifest.get("empty"):
            continue
        bundle = torch.load(manifest["file"], map_location="cpu", weights_only=True)
        local_train = torch.tensor(
            [i for i in range(int(bundle["target"].shape[0])) if i + offset in train_set],
            dtype=torch.long,
        )
        if len(local_train):
            local_evidence_mask = informative_by_example[offset + local_train] > 0
            if model.price_response is not None:
                local_known = bundle["tensors"]["known_future"][local_train]
                reference_mask = (local_known[..., model.price_available_index] > 0.5) & (
                    local_known[..., model.discount_available_index] > 0.5
                )
                train_price_reference_total += int(reference_mask.sum())
            price_mask = None
            if price_penalty_weight and model.price_index is not None:
                price_mask = _price_intervention_mask(
                    _subset(bundle["tensors"], local_train),
                    model.price_index,
                    model.price_available_index,
                )
                train_price_eligible_total += int(price_mask.sum())
                local_evidence_mask |= price_mask.reshape(len(local_train), -1).any(dim=1)
            local_evidence = local_train[local_evidence_mask]
            evidence_series.update(bundle["example_series"][local_evidence].tolist())
            if model.static.factorised:
                evidence_product.update(
                    row
                    for row in bundle["tensors"]["product_index"][local_evidence].tolist()
                    if row < counts["product"]
                )
            if model.static.factorised or model.factor is not None:
                evidence_store.update(
                    row
                    for row in bundle["tensors"]["store_index"][local_evidence].tolist()
                    if row < counts["store"]
                )
    train_informative_total = float(informative_by_example[train_rows_global].sum())
    val_informative_total = float(informative_by_example[val_rows_global].sum())
    if train_informative_total <= 0.0:
        raise OursError("sharded training split has no informative cells")
    if val_informative_total <= 0.0:
        raise OursError("sharded validation split has no informative cells")
    if price_penalty_weight and model.price_index is not None and train_price_eligible_total == 0:
        raise OursError(
            "price_penalty_weight is positive but the sharded training split has no finite "
            "positive price cell; no relative monotonicity intervention could execute"
        )
    if model.price_response is not None and train_price_reference_total == 0:
        raise OursError(
            "monotone_raw_v1 training split has no cell with both planned price and a valid "
            "frozen historical reference; a structural price-response claim would be vacuous"
        )
    training_identity_evidence = {
        "series": evidence_series,
        "product": evidence_product,
        "store": evidence_store,
    }
    training_oov_rows = {
        "series": n_series,
        "product": counts["product"] if product is not None else 0,
        "store": counts["store"],
    }
    _project_training_oov_parameters(model, training_identity_evidence, training_oov_rows)
    # Per-shard row masks against the CONCATENATED example axis.
    offsets, at = [], 0
    for m in manifests:
        offsets.append(at)
        at += int(m.get("n_examples", 0))

    channel_columns = (
        [(layout.index(c), layout.index(f"{c}_available")) for c in commercial.CHANNELS]
        if "price_available" in layout
        else []
    )
    best, best_state, waited = float("inf"), None, 0
    epoch_normalizer_num = train_informative_total
    optimizer_steps_total = 0
    training_examples_seen = 0
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        learning_rate = float(opt.param_groups[0]["lr"])
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        model.train()
        train_objective_total = 0.0
        gradient_norm_total = 0.0
        optimizer_steps = 0
        shard_order = torch.randperm(len(live), generator=generator).tolist()
        val_loss_total = 0.0
        n_batches_epoch = 0
        step_plan = []
        for oi in shard_order:
            m = live[oi]
            bundle = torch.load(m["file"], map_location="cpu", weights_only=True)
            offset = offsets[m["shard_id"]]
            local_train = torch.tensor(
                [i for i in range(int(bundle["target"].shape[0])) if i + offset in train_set],
                dtype=torch.long,
            )
            if len(local_train):
                batches = factor_block_batches(
                    local_train, bundle["tensors"]["group_index"], batch_size, generator
                )
                step_plan.append((m, batches))
                n_batches_epoch += len(batches)
        for m, batches in step_plan:
            bundle = torch.load(m["file"], map_location="cpu", weights_only=True)
            tensors = {
                k2: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                for k2, v in bundle["tensors"].items()
            }
            target = bundle["target"].to(dev)
            state = bundle["target_state"].to(dev)
            probs = None if bundle["target_probs"] is None else bundle["target_probs"].to(dev)
            for idx in batches:
                _price_violation, batch_objective, gradient_norm, _retries = (
                    _complete_legacy_optimizer_batch(
                        model, opt,
                        functools.partial(
                            _backward_logical_batch, model, tensors, target, state, idx,
                            target_probs=probs,
                            normalization_count=epoch_normalizer_num / n_batches_epoch,
                            gradient_microbatch_size=gradient_microbatch_size,
                            covariate_dropout=covariate_dropout,
                            channel_columns=channel_columns,
                            price_penalty_weight=price_penalty_weight,
                            n_optimizer_batches=n_batches_epoch,
                            total_price_eligible=train_price_eligible_total,
                            scaler=scaler,
                        ),
                        scaler=scaler,
                        zero_oov_gradients=functools.partial(
                            _zero_training_oov_gradients, model, training_oov_rows
                        ),
                    )
                )
                gradient_norm_total += gradient_norm
                train_objective_total += batch_objective
                optimizer_steps += 1
                _project_training_oov_parameters(
                    model, training_identity_evidence, training_oov_rows
                )

            del tensors, target, state, probs

        # Score ONE coherent end-of-epoch model over EVERY held-out series. Validation used to be
        # nested inside `step_plan`, which contains only shards with at least one TRAINING row. A
        # shard made entirely of held-out series was therefore never scored; worse, mixed shards
        # were scored at different intermediate model states as later shards were still updating
        # the weights. Reloading one shard at a time keeps the memory bound while making the sum a
        # validation loss of one actual checkpoint.
        model.eval()
        with torch.no_grad():
            for m in live:
                bundle = torch.load(m["file"], map_location="cpu", weights_only=True)
                tensors = {
                    k2: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                    for k2, v in bundle["tensors"].items()
                }
                target = bundle["target"].to(dev)
                state = bundle["target_state"].to(dev)
                probs = None if bundle["target_probs"] is None else bundle["target_probs"].to(dev)
                offset = offsets[m["shard_id"]]
                local_val = torch.tensor(
                    [
                        i
                        for i in range(int(bundle["target"].shape[0]))
                        if i + offset not in train_set
                    ],
                    dtype=torch.long,
                )
                if len(local_val):
                    val_loss_total += _microbatched_validation_loss(
                        model,
                        tensors,
                        target,
                        state,
                        local_val,
                        target_probs=probs,
                        normalization_count=val_informative_total,
                        gradient_microbatch_size=gradient_microbatch_size,
                        training_identity_evidence=training_identity_evidence,
                        training_oov_rows=training_oov_rows,
                    )
                del tensors, target, state, probs
        model.train()
        sched.step()
        optimizer_steps_total += optimizer_steps
        training_examples_seen += len(train_rows_global)
        if training_log is not None:
            training_log.log_epoch(
                epoch + 1,
                train_objective=train_objective_total,
                validation_objective=val_loss_total,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm_total / optimizer_steps,
                optimizer_steps=optimizer_steps_total,
                training_examples_seen=training_examples_seen,
                seconds=time.perf_counter() - epoch_started,
                process_peak_rss_bytes=_process_peak_rss_bytes(),
                cuda_peak_memory_bytes=(
                    int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
                ),
            )
        print(
            f"    epoch {epoch + 1:3}/{epochs}  val {val_loss_total:.5f}"
            f"{'  *' if val_loss_total < best else ''}",
            flush=True,
        )
        if val_loss_total < best - 1e-6:
            best, waited = val_loss_total, 0
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"    stopped at epoch {epoch + 1}", flush=True)
                break
    if training_log is not None:
        training_log.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    identity_evidence_rows = _finalize_oov_parameters(model, training_identity_evidence)
    training_device = str(dev)
    artifact_device = torch.device("cpu")
    model = model.to(artifact_device)
    model.eval()
    print(
        f"  claim artifacts on {artifact_device} (training device {training_device})",
        flush=True,
    )

    d0 = dt.date.fromisoformat(origin)
    emitted = 0
    for j, m in enumerate(live):
        bundle = torch.load(m["file"], map_location="cpu", weights_only=True)
        predict = {
            k2: (v.to(artifact_device) if isinstance(v, torch.Tensor) else v)
            for k2, v in bundle["inference"].items()
        }
        yhat = point_forecast(
            model, predict, method=point_factor_method, factor_draws=point_factor_draws, seed=seed
        )
        records = [
            (bundle["series_ids"][i], d0 + dt.timedelta(days=step + 1), float(yhat[i, step]))
            for i in range(len(bundle["series_ids"]))
            for step in range(horizon)
        ]
        frame = spark.createDataFrame(records, "series_id string, date date, yhat double")
        frame.write.mode("overwrite" if j == 0 else "append").parquet(str(out_dir))
        emitted += len(records)
        print(f"  shard {m['shard_id']}: wrote {len(records):,} cells", flush=True)
        del bundle, predict, yhat

    if checkpoint_out is not None:
        save_checkpoint(
            model,
            checkpoint_out,
            model_config=model_config,
            run=dict(
                origin=origin,
                horizon=horizon,
                encoder_days=encoder_days,
                aggregate_trajectory_projection=(_dormant_aggregate_projection_record(model)),
                training_device=training_device,
                artifact_device=str(artifact_device),
                train_stride=train_stride,
                train_windows=train_windows,
                epochs=epochs,
                batch_size=batch_size,
                gradient_microbatch_size=gradient_microbatch_size,
                **({"precision": precision} if precision != "32-true" else {}),
                patience=patience,
                importance_draws=n_draws,
                events=[list(e) for e in events],
                event_max_support=event_max_support,
                event_windows=(
                    None
                    if event_windows is None
                    else {k2: list(v) for k2, v in event_windows.items()}
                ),
                cycles=[list(c) for c in cycles],
                seed=seed,
                # BOTH factor counts, separately. A run record that says only "1,000"
                # cannot tell a reader whether that was the point integration or the
                # systemic scenario count -- different claims about different numbers.
                point_factor_method=point_factor_method,
                point_factor_draws=(
                    None
                    if point_factor_method == "gauss_hermite"
                    else (
                        point_factor_draws if point_factor_draws is not None else POINT_FACTOR_DRAWS
                    )
                ),
                path_factor_draws=(
                    inference_factor_draws
                    if inference_factor_draws is not None
                    else default_factor_draws(n_paths)
                ),
                n_paths=n_paths,
                paths_materialized=False,
                rng_domain_scheme=RNG_DOMAIN_SCHEME,
                factor_group_col=grouping,
                factor_enabled=model.factor is not None,
                topology=("bottom_up_factor" if model.factor is not None else "bottom_up"),
                product_col=product,
                store_col=store,
                category_col=category,
                known_names=list(layout),
                declared_known_drivers=sorted(declared_drivers),
                resolved_build=(None if resolved is None else resolved.canonical()),
                resolved_build_hash=(None if resolved is None else resolved.build_hash),
                training_controls=dict(
                    demand_branch_dropout=demand_dropout,
                    covariate_dropout=covariate_dropout,
                    gradient_microbatch_size=gradient_microbatch_size,
                    price_penalty_weight=price_penalty_weight,
                    price_penalty_eligible_cells=train_price_eligible_total,
                    price_response_mode=price_response_mode,
                    price_occurrence_slope_max=price_occurrence_slope_max,
                    price_size_slope_max=price_size_slope_max,
                    price_response_valid_reference_cells=train_price_reference_total,
                    static_variable_selection=static_selection,
                ),
                sharded=dict(
                    n_shards=len(predicates),
                    max_shard_series=max_shard_series,
                    max_series_per_shard=max_series_per_shard,
                    predicates=list(predicates),
                    manifest=[{k2: v for k2, v in m.items() if k2 != "file"} for m in manifests],
                ),
                plan_provenance=(
                    None
                    if not any(m.get("plan_provenance") is not None for m in live)
                    else {
                        "shards": [
                            {"shard_id": m["shard_id"], **m["plan_provenance"]} for m in live
                        ]
                    }
                ),
                # The dispatcher refuses priced sharded report/require runs because a
                # first-shard sample is not a global certificate. Unpriced runs and an
                # explicitly skipped priced run therefore have no ladder evidence.
                price_ladder=None,
                # The SAME identity namespace the single-shard checkpoint stores -- the
                # docstring above promised it and predict_from_checkpoint requires it;
                # without these a sharded fit could never serve a new client.
                static_real_stats=identity_meta.get("real_stats"),
                vocabularies=dict(
                    attribute_names=[*identity_meta["one_hot_names"], *panel.static_reals],
                    product_levels=_ordered_levels(vocab_maps["product"]),
                    store_levels=_ordered_levels(vocab_maps["store"]),
                    factor_levels=_ordered_levels(vocab_maps["grouping"]),
                ),
                oov_rows=dict(
                    series=n_series,
                    product=counts["product"] if product is not None else 0,
                    store=counts["store"],
                ),
                oov_scheme=OOV_SCHEME,
                identity_evidence_rows=identity_evidence_rows,
                paths_sample=paths_sample,
                paths_sample_rule="smallest sha256(seed\\0series_id), without replacement",
                series_ids=list(series_vocab),
                tensors=None,
                panel=asdict(panel),
                panel_parquet_sha256=directory_digest(panel.path, suffixes=(".parquet",)),
            ),
        )
        print(
            f"  checkpoint: {checkpoint_out} (sharded fit: per-series tensors omitted; "
            "checkpoint serving currently rebuilds the requested panel as one assembly)",
            flush=True,
        )

    forecast_series = sum(m.get("n_series", 0) for m in live)
    disposition = {EMITTED: emitted}
    declined = (n_total - forecast_series) * horizon
    if declined:
        disposition[DECLINED_NO_WINDOW] = declined
    return Forecast(
        adapter=NAME,
        path=str(out_dir),
        origin=origin,
        horizon=horizon,
        n_series=n_total,
        disposition=disposition,
        target_estimand=panel.target_estimand,
    ).validate()


def _shard_universe(
    panel: Panel,
    spark: SparkSession,
    origin: str,
    horizon: int,
    *,
    encoder_days: int,
    events,
    event_max_support: int,
    event_windows,
    cycles,
    work_dir: pathlib.Path,
    shard_id: int,
    predicate: str | None,
    level_priors,
    train_level_priors=None,
    series_vocab_position: dict[str, int],
    vocab_maps: dict[str, dict[str, int]],
    attribute_meta: dict,
    text_embeddings,
    n_draws: int,
    train_origins: list[str],
    target_origins: list[str],
    target_estimand: str,
    status_prob_cols,
) -> dict:
    """Assemble ONE shard's complete training-and-inference bundle, bounded by the shard.

    The single-shard `fit_predict` body is the reference implementation of these steps;
    `test_shard_assembly_matches_the_single_shard_universe` pins the two row-for-row, because
    an orchestration written twice is exactly how implementations drift.
    """
    category_col = attribute_meta["category_col"]
    category_of = (
        _series_levels(panel, spark, category_col, series_predicate=predicate)
        if category_col
        else None
    )
    tag = f"shard{shard_id}"
    infer = _assemble(
        panel,
        spark,
        origin,
        horizon,
        encoder_days,
        events,
        event_max_support,
        work_dir / f"infer_{tag}",
        cycles,
        category_of=category_of,
        event_windows=event_windows,
        series_predicate=predicate,
        level_priors=level_priors,
    )
    if not infer.series_ids:
        return {"empty": True, "shard_id": shard_id}
    train = [
        _assemble(
            panel,
            spark,
            o,
            horizon,
            encoder_days,
            events,
            event_max_support,
            work_dir / f"train{i}_{tag}",
            cycles,
            category_of=category_of,
            event_windows=event_windows,
            series_predicate=predicate,
            level_priors=(level_priors if train_level_priors is None else train_level_priors[i]),
        )
        for i, o in enumerate(train_origins)
    ]
    target_rows = []
    for i, o in enumerate(target_origins):
        path = demand_history.build(
            panel.path,
            o,
            horizon,
            spark,
            str(work_dir / f"_target{i}_{tag}"),
            series_col=panel.series_col,
            time_col=panel.time_col,
            target_col=panel.target_col,
            exposure_col=panel.exposure_col,
            oos_col=panel.oos_col,
            status_prob_cols=status_prob_cols,
            series_predicate=predicate,
            negative_target_policy=panel.negative_target_policy,
        )
        target_rows.append({r["series_id"]: r for r in spark.read.parquet(str(path)).collect()})

    series_ids = infer.series_ids
    rows_global = torch.tensor([series_vocab_position[sid] for sid in series_ids], dtype=torch.long)
    local_of = {sid: i for i, sid in enumerate(series_ids)}

    # Identity attributes from the GLOBAL vocabularies, rows restricted to this shard.
    attributes = _identity_rows_shard(attribute_meta, spark, series_ids, shard_id)
    if panel.static_reals:
        reals = _static_reals_shard(
            panel, spark, series_ids, attribute_meta["real_stats"], predicate
        )
        attributes = torch.cat([attributes, reals], dim=1)
    text = None
    if text_embeddings is not None:
        text, _, _ = _text_tensors(text_embeddings, spark, series_ids)

    def index_from(
        role: str,
    ) -> tuple[torch.Tensor, dict[str, str | None] | None]:
        column = attribute_meta[role]
        if column is None:
            return torch.zeros(len(series_ids), dtype=torch.long), None
        vocab = vocab_maps[role]
        levels = _series_levels(panel, spark, column, series_predicate=predicate)
        # A declared-but-null identity is unknown, not the first sorted level.  Parameter lookup
        # uses the reserved final OOV row; trajectory equality below stays on the raw semantics.
        out = torch.full((len(series_ids),), len(vocab), dtype=torch.long)
        for i, sid in enumerate(series_ids):
            level = levels.get(sid)
            if level is not None:
                out[i] = vocab[level]
        return out, levels

    product_index, product_values = index_from("product")
    store_index, _ = index_from("store")
    factor_enabled = attribute_meta["grouping"] is not None
    if factor_enabled:
        _, factor_values = index_from("grouping")
        assert factor_values is not None
        group_index, _ = _query_composite_groups(series_ids, factor_values, product_values)
    else:
        # Match the single route's rank-zero degradation exactly.  Product identity remains an
        # input embedding, but without a declared factor group it must not become a stochastic
        # dependence block (nor one oversized likelihood batch).
        group_index = _independent_group_index(len(series_ids))

    def window_tensors(a: _Assembly, take: torch.Tensor, rows: torch.Tensor) -> dict:
        return dict(
            history=a.history[take],
            observed=a.observed[take],
            censored=a.censored[take],
            attributes=attributes[rows],
            series_index=rows_global[rows],
            text=None if text is None else text[rows],
            known_future=a.known.values[take],
            baseline_logit=a.baseline_logit[take],
            baseline_log_size=a.baseline_log_size[take],
            group_index=group_index[rows],
            store_index=store_index[rows],
            product_index=product_index[rows],
            n_draws=n_draws if factor_enabled else 1,
            commercial=None if a.commercial is None else a.commercial[take],
            commercial_mask=(None if a.commercial_mask is None else a.commercial_mask[take]),
            event_distance=(None if a.event_distance is None else a.event_distance[take]),
            event_present=(None if a.event_present is None else a.event_present[take]),
        )

    pieces, targets, states, series_rows, probs_pieces = [], [], [], [], []
    next_factor_block = 0
    for i, assembly in enumerate(train):
        usable = [sid for sid in assembly.series_ids if sid in local_of and sid in target_rows[i]]
        if not usable:
            continue
        at = {sid: j for j, sid in enumerate(assembly.series_ids)}
        take = torch.tensor([at[sid] for sid in usable], dtype=torch.long)
        rows = torch.tensor([local_of[sid] for sid in usable], dtype=torch.long)
        piece = window_tensors(assembly, take, rows)
        piece["group_index"], next_factor_block = _offset_dense_blocks(
            piece["group_index"], next_factor_block
        )
        pieces.append(piece)
        targets.append(
            torch.tensor([target_rows[i][sid]["history"] for sid in usable], dtype=torch.float32)
        )
        raw_state = torch.tensor([target_rows[i][sid]["state"] for sid in usable], dtype=torch.long)
        states.append(_states_for_estimand(raw_state, target_estimand))
        if status_prob_cols is not None:
            window_probs = torch.stack(
                [
                    torch.tensor([target_rows[i][sid][col] for sid in usable], dtype=torch.float32)
                    for col in ("p_exact", "p_lower_bound", "p_unobserved")
                ],
                dim=-1,
            )
            probs_pieces.append(_probs_for_estimand(window_probs, target_estimand))
        series_rows.append(rows_global[rows])
    if not pieces:
        return {"empty": True, "shard_id": shard_id, "series_ids": series_ids}
    train_inputs = {
        k: (
            torch.cat([piece[k] for piece in pieces])
            if isinstance(pieces[0][k], torch.Tensor)
            else pieces[0][k]
        )
        for k in pieces[0]
    }
    inference = window_tensors(infer, torch.arange(len(series_ids)), torch.arange(len(series_ids)))
    return dict(
        shard_id=shard_id,
        series_ids=series_ids,
        tensors=train_inputs,
        target=torch.cat(targets),
        target_state=torch.cat(states),
        target_probs=torch.cat(probs_pieces) if probs_pieces else None,
        example_series=torch.cat(series_rows),
        inference=inference,
        known_names=tuple(infer.known.names),
        declared_known_drivers=tuple(sorted(infer.known.declared_drivers)),
        n_commercial=infer.n_commercial,
        n_events=infer.n_events,
        plan_provenance=(
            None
            if infer.plan is None
            else {
                "inference": infer.plan.as_record(),
                "training": [None if a.plan is None else a.plan.as_record() for a in train],
            }
        ),
    )


def _identity_rows_shard(
    attribute_meta: dict, spark: SparkSession, series_ids: list[str], shard_id: int
) -> torch.Tensor:
    """One shard's one-hot rows from the PREBUILT identity table and GLOBAL vocabularies."""
    if not attribute_meta["static"]:
        return torch.zeros(len(series_ids), 1)
    frame = spark.read.parquet(attribute_meta["identity_path"])
    if "_shard" in frame.columns:
        frame = frame.filter(F.col("_shard") == shard_id)
    rows = {r["series_id"]: r for r in frame.collect()}
    vocabularies = attribute_meta["vocabularies"]
    names = attribute_meta["one_hot_names"]
    out = torch.zeros(len(series_ids), len(names))
    offset = 0
    for column in attribute_meta["static"]:
        at = {level: j for j, level in enumerate(vocabularies[column])}
        for i, sid in enumerate(series_ids):
            row = rows.get(sid)
            if row is not None and row[column] is not None:
                j = at.get(str(row[column]))
                if j is not None:
                    out[i, offset + j] = 1.0
        offset += len(vocabularies[column])
    return out


def _static_reals_shard(
    panel: Panel,
    spark: SparkSession,
    series_ids: list[str],
    stats: dict[str, tuple[float, float]],
    predicate: str | None,
) -> torch.Tensor:
    """One shard's standardised static reals, using GLOBAL (mean, std) so every shard sees the
    same transformation -- per-shard statistics would make one physical weight mean different
    kilograms in different shards."""
    source = spark.read.parquet(panel.path)
    if predicate is not None:
        source = source.filter(F.expr(predicate))
    frame = source.select(
        F.col(panel.series_col).alias("series_id"),
        *[F.col(c).cast("double").alias(c) for c in panel.static_reals],
    )
    per_series = {
        r["series_id"]: r
        for r in frame.groupBy("series_id")
        .agg(*[F.first(c).alias(c) for c in panel.static_reals])
        .collect()
    }
    out = torch.zeros(len(series_ids), len(panel.static_reals))
    for j, column in enumerate(panel.static_reals):
        mean, std = stats[column]
        for i, sid in enumerate(series_ids):
            row = per_series.get(sid)
            value = None if row is None else row[column]
            if value is None:
                raise OursError(f"static real {column!r} is missing for series {sid!r}")
            out[i, j] = 0.0 if std <= 1e-12 else (float(value) - mean) / std
    return out


def _private_scratch_directory(purpose: str, origin: str) -> pathlib.Path:
    """Create one private local feature namespace for a single invocation."""
    safe_origin = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(origin))
    return pathlib.Path(tempfile.mkdtemp(prefix=f"forecast-{purpose}-{safe_origin}-"))


def _fit_scratch_directory(origin: str) -> pathlib.Path:
    """Create isolated fit scratch, including for concurrent calls at the same origin."""
    return _private_scratch_directory("fit", origin)


def _resolved_price_penalty_weight(
    resolved: ResolvedBuild,
    panel: Panel,
    value: object,
) -> float:
    """Return the exact float training control implied by a resolved build."""
    mode = resolved.price_response_mode
    if mode == "none":
        return 0.0
    if "price" not in _declared_commercial_drivers(panel):
        raise OursError(
            f"resolved price_response_mode={mode!r} but the runtime Panel/PlanSource declares "
            "no price driver; the build hash cannot constrain an absent input"
        )
    if mode == "monotone_raw_v1":
        if not is_finite_real(value) or float(value) != 0.0:
            raise OursError(
                "resolved price_response_mode='monotone_raw_v1' requires "
                f"price_penalty_weight=0.0, got {value!r}; the structural route trains only "
                "through the likelihood and is audited after all epochs"
            )
        return 0.0
    if mode != "soft_penalty_v1":
        raise OursError(f"unsupported resolved price_response_mode={mode!r}")
    if not is_finite_real(value) or value <= 0.0:
        raise OursError(
            "resolved price_response_mode='soft_penalty_v1' requires a finite positive "
            f"price_penalty_weight, got {value!r}"
        )
    return float(value)


def _with_isolated_feature_scratch(function):
    """Give one public fit call private scratch and remove it on every Python exit path."""
    signature = inspect.signature(function)
    private_name = "_feature_scratch"
    if private_name not in signature.parameters:
        raise TypeError(f"{function.__name__} must accept {private_name}")
    public_signature = signature.replace(
        parameters=[
            parameter for name, parameter in signature.parameters.items() if name != private_name
        ]
    )

    @functools.wraps(function)
    def scoped(*args, **kwargs):
        bound = public_signature.bind(*args, **kwargs)
        _validate_daily_span(
            bound.arguments["panel"],
            bound.arguments["origin"],
            bound.arguments["horizon"],
            adapter=NAME,
        )
        scratch = _fit_scratch_directory(str(bound.arguments["origin"]))
        try:
            return function(*args, **kwargs, _feature_scratch=scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    scoped.__signature__ = public_signature
    return scoped


def _snapshot_warm_start_checkpoint(
    source: str | pathlib.Path,
    expected_sha256: str,
    scratch: pathlib.Path,
) -> pathlib.Path:
    """Copy one hash-bound source checkpoint into the invocation's private namespace."""
    if not _sha256_is_canonical(expected_sha256):
        raise OursError("warm_start_checkpoint_sha256 must be a canonical lowercase SHA-256 digest")
    source_path = pathlib.Path(source)
    if not source_path.is_file():
        raise OursError(
            f"warm_start_checkpoint must name one readable non-sharded checkpoint file, got "
            f"{source_path}"
        )
    try:
        before = _file_sha256(source_path)
    except OSError as exc:
        raise OursError(f"cannot hash warm-start checkpoint {source_path}: {exc}") from exc
    if before != expected_sha256:
        raise OursError(
            "warm-start source checkpoint SHA-256 does not match the caller's expected digest: "
            f"expected={expected_sha256}, observed={before}"
        )

    snapshot = scratch / "warm-start-source" / "checkpoint.pt"
    _copy_replay_input(source_path, snapshot)
    after = _file_sha256(source_path)
    copied = _file_sha256(snapshot)
    if after != expected_sha256 or copied != expected_sha256:
        raise OursError(
            "warm-start source changed while it was snapshotted; refusing mixed checkpoint "
            f"bytes (before={before}, after={after}, snapshot={copied})"
        )
    return snapshot


def _private_snapshot_state(
    root: str | pathlib.Path,
    *,
    include_parent: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """Pin inode/ctime topology for one private copy across its complete consumption.

    Content hashes bind what was copied.  This identity record additionally detects a private
    A -> B -> A replacement: restoring the original bytes cannot restore inode/ctime metadata.
    """
    path = pathlib.Path(root)
    candidates: list[tuple[str, pathlib.Path]] = []
    if include_parent:
        candidates.append(("<parent>", path.parent))
    candidates.append((".", path))
    if path.is_dir():
        candidates.extend(
            (candidate.relative_to(path).as_posix(), candidate)
            for candidate in sorted(path.rglob("*"))
        )
    state: list[tuple[object, ...]] = []
    for relative, candidate in candidates:
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise OursError(f"private input snapshot disappeared: {candidate}") from exc
        if not (candidate.is_file() or candidate.is_dir()) or candidate.is_symlink():
            raise OursError(f"private input snapshot contains a non-regular node: {candidate}")
        state.append(
            (
                relative,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        )
    return tuple(state)


def _assert_private_snapshot_state(
    root: str | pathlib.Path,
    expected: tuple[tuple[object, ...], ...],
    *,
    include_parent: bool = False,
    label: str,
) -> None:
    if _private_snapshot_state(root, include_parent=include_parent) != expected:
        raise OursError(f"{label} private snapshot changed while it was consumed")


def _set_private_snapshot_read_only(root: str | pathlib.Path, read_only: bool) -> None:
    """Seal/unseal one owned temporary tree without following links."""
    path = pathlib.Path(root)
    if not path.exists():
        return
    entries = [path, *path.rglob("*")]
    for candidate in entries:
        if candidate.is_symlink():
            raise OursError(f"private input snapshot contains a symbolic link: {candidate}")
    if read_only:
        for candidate in reversed(entries):
            candidate.chmod(0o555 if candidate.is_dir() else 0o444)
    else:
        for candidate in entries:
            candidate.chmod(0o700 if candidate.is_dir() else 0o600)


@dataclass(frozen=True)
class _PrivateReplayInputSnapshot:
    checkpoint: pathlib.Path
    panel: Panel
    text_embeddings: pathlib.Path | None
    source_digests: dict[str, str]
    root: pathlib.Path
    state: tuple[tuple[object, ...], ...]

    def __iter__(self):
        # Preserve the established private helper's four-value unpacking contract.
        yield self.checkpoint
        yield self.panel
        yield self.text_embeddings
        yield self.source_digests

    def _assert_unchanged(self) -> None:
        _assert_private_snapshot_state(
            self.root,
            self.state,
            include_parent=True,
            label="replay input",
        )


@dataclass(frozen=True)
class _PrivateAggregateTrajectorySnapshot:
    artifact: pathlib.Path
    source_digest: str
    root: pathlib.Path
    state: tuple[tuple[object, ...], ...]

    def __iter__(self):
        # Preserve the established private helper's two-value unpacking contract.
        yield self.artifact
        yield self.source_digest

    def _assert_unchanged(self) -> None:
        _assert_private_snapshot_state(
            self.root,
            self.state,
            include_parent=True,
            label="aggregate trajectory",
        )


def _remove_owned_private_namespace(owner: pathlib.Path) -> None:
    """Delete an owned temporary namespace without chmod/following untrusted leaves.

    A malicious consumer may insert a hard link to an unrelated file or replace the snapshot
    root with a symlink.  Teardown therefore changes permissions only through directory file
    descriptors and unlinks every non-directory node without touching its target metadata.
    """

    def clear(directory_descriptor: int) -> None:
        os.fchmod(directory_descriptor, 0o700)
        for entry in list(os.scandir(directory_descriptor)):
            metadata = os.stat(
                entry.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_descriptor,
                )
                try:
                    clear(child)
                finally:
                    os.close(child)
                os.rmdir(entry.name, dir_fd=directory_descriptor)
            else:
                os.unlink(entry.name, dir_fd=directory_descriptor)

    try:
        owner_descriptor = os.open(
            owner,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise OursError(f"cannot open owned private namespace for teardown: {owner}") from exc
    try:
        clear(owner_descriptor)
    finally:
        os.close(owner_descriptor)
    try:
        owner.rmdir()
    except OSError as exc:
        raise OursError(f"cannot remove owned private namespace: {owner}") from exc


def _assert_warm_start_source_unchanged(
    source: str | pathlib.Path,
    expected_sha256: str,
) -> None:
    """Refuse publication if the caller's source checkpoint changed during continuation."""
    try:
        observed = _file_sha256(pathlib.Path(source))
    except OSError as exc:
        raise OursError(f"cannot re-hash warm-start checkpoint {source}: {exc}") from exc
    if observed != expected_sha256:
        raise OursError(
            "warm-start source checkpoint changed during continuation; refusing publication: "
            f"expected={expected_sha256}, observed={observed}"
        )


def _warm_start_values_equal_exact(observed: object, expected: object) -> bool:
    """Exact recursive equality for namespace metadata, including tensor values and dtypes."""
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, torch.Tensor):
        return (
            observed.dtype == expected.dtype
            and observed.shape == expected.shape
            and torch.equal(observed.detach().cpu(), expected.detach().cpu())
        )
    if type(expected) is dict:
        return observed.keys() == expected.keys() and all(
            _warm_start_values_equal_exact(observed[key], value) for key, value in expected.items()
        )
    if type(expected) in (list, tuple):
        return len(observed) == len(expected) and all(
            _warm_start_values_equal_exact(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _validate_warm_start_namespace(
    source_record: dict,
    expected_model_config: dict,
    expected_run: dict,
) -> None:
    """Bind source weights to the exact architecture, data roles and row namespaces in use."""
    source_run = source_record.get("run")
    if type(source_run) is not dict:
        raise OursError("warm-start source checkpoint has no run namespace")
    if source_run.get("sharded") is not None:
        raise OursError("sharded checkpoints cannot be used as warm-start sources")
    if not _warm_start_values_equal_exact(source_record.get("model_config"), expected_model_config):
        raise OursError("warm-start source model_config differs from the continuation architecture")
    missing = [
        field
        for field in _WARM_START_NAMESPACE_FIELDS
        if field not in source_run or field not in expected_run
    ]
    if missing:
        raise OursError(
            f"warm-start source/current namespace is incomplete: {sorted(set(missing))}"
        )
    mismatched = [
        field
        for field in _WARM_START_NAMESPACE_FIELDS
        if not _warm_start_values_equal_exact(source_run[field], expected_run[field])
    ]
    if mismatched:
        raise OursError(
            "warm-start source namespace differs from the continuation build: " f"{mismatched}"
        )


def _meaningful_validation_improvement(
    candidate: float,
    best: float,
    absolute_min_delta: float,
    relative_min_delta: float,
) -> bool:
    """Apply the preregistered absolute-or-relative validation improvement threshold."""
    if not math.isfinite(candidate):
        return False
    if best == float("inf"):
        return True
    if not math.isfinite(best):
        return False
    threshold = max(absolute_min_delta, relative_min_delta * abs(best))
    return candidate < best - threshold


def _print_quiet_training_completion(
    label: str,
    completed: int,
    maximum: int,
    stopped_early: bool,
) -> None:
    """Report completion without exposing an epoch-associated validation metric."""
    print(
        f"    {label} complete: {completed}/{maximum} epoch(s)"
        f"{' (early stop)' if stopped_early else ''}",
        flush=True,
    )


def _require_finite_validation_loss(value: float, *, epoch: int) -> float:
    """Fail an arm immediately; nonfinite validation is never a patience observation."""
    if not math.isfinite(value):
        raise OursError(f"validation loss is not finite at epoch {epoch}: {value!r}")
    return value


def _validate_warm_start_checkpoint_record(run: dict) -> None:
    """Validate the complete, self-consistent continuation summary when one is present."""
    record = run.get("warm_start")
    if record is None:
        return
    fields = {
        "schema",
        "source_checkpoint_sha256",
        "source_checkpoint_schema",
        "source_is_epoch_zero",
        "optimizer_state_restored",
        "scheduler_state_restored",
        "optimizer",
        "scheduler",
        "learning_rate",
        "max_additional_epochs",
        "additional_epochs_completed",
        "patience",
        "validation_fraction",
        "validation_seed",
        "validation_evaluated_every_epoch",
        "common_validation_random_numbers",
        "absolute_min_delta",
        "relative_min_delta",
        "source_validation_loss",
        "best_validation_loss",
        "best_epoch",
        "epochs_without_improvement",
        "stopped_early",
        "stop_reason",
        "best_weights_restored",
        "quiet_epoch_logging",
        "validation_trace",
    }
    if type(record) is not dict or set(record) != fields:
        observed = set(record) if isinstance(record, dict) else set()
        raise OursError(
            "warm-start checkpoint summary is not the exact current schema: "
            f"missing={sorted(fields - observed)}, extra={sorted(observed - fields)}"
        )
    if (
        record["schema"] != WARM_START_SCHEMA
        or record["source_checkpoint_schema"] != CHECKPOINT_SCHEMA
        or not _sha256_is_canonical(record["source_checkpoint_sha256"])
    ):
        raise OursError("warm-start checkpoint lineage schema or source SHA-256 is invalid")
    required_true = (
        "source_is_epoch_zero",
        "validation_evaluated_every_epoch",
        "common_validation_random_numbers",
        "best_weights_restored",
    )
    required_false = ("optimizer_state_restored", "scheduler_state_restored")
    if (
        any(record[field] is not True for field in required_true)
        or any(record[field] is not False for field in required_false)
        or record["optimizer"] != "AdamW"
        or record["scheduler"] != "CosineAnnealingLR"
    ):
        raise OursError("warm-start checkpoint does not record the required fresh training state")
    for field in ("stopped_early", "quiet_epoch_logging"):
        if type(record[field]) is not bool:
            raise OursError(f"warm-start {field} must be boolean")
    for field in ("max_additional_epochs", "patience"):
        value = record[field]
        if type(value) is not int or value < 1:
            raise OursError(f"warm-start {field} must be a positive integer")
    for field in (
        "additional_epochs_completed",
        "best_epoch",
        "epochs_without_improvement",
        "validation_seed",
    ):
        value = record[field]
        if type(value) is not int or value < 0:
            raise OursError(f"warm-start {field} must be a nonnegative integer")
    maximum = record["max_additional_epochs"]
    completed = record["additional_epochs_completed"]
    best_epoch = record["best_epoch"]
    stale = record["epochs_without_improvement"]
    if not (
        1 <= completed <= maximum
        and 0 <= best_epoch <= completed
        and stale == completed - best_epoch
    ):
        raise OursError("warm-start epoch/best/staleness summary is inconsistent")
    for field in (
        "learning_rate",
        "validation_fraction",
        "absolute_min_delta",
        "relative_min_delta",
        "source_validation_loss",
        "best_validation_loss",
    ):
        if type(record[field]) is not float or not math.isfinite(record[field]):
            raise OursError(f"warm-start {field} must be a finite float")
    if (
        record["learning_rate"] <= 0.0
        or not 0.0 < record["validation_fraction"] < 1.0
        or record["absolute_min_delta"] < 0.0
        or record["relative_min_delta"] < 0.0
        or record["best_validation_loss"] > record["source_validation_loss"]
    ):
        raise OursError("warm-start numeric controls or validation losses are inconsistent")
    early = record["stopped_early"]
    reason = record["stop_reason"]
    if early:
        if reason != "validation_patience" or stale != record["patience"]:
            raise OursError("warm-start early-stop reason contradicts its patience counter")
    elif reason != "max_additional_epochs" or completed != maximum or stale >= record["patience"]:
        raise OursError("warm-start max-epoch stop reason contradicts completed epochs")

    trace = record["validation_trace"]
    trace_fields = {
        "epoch",
        "validation_loss",
        "effective_min_delta",
        "meaningful_improvement",
        "best_validation_loss",
        "epochs_without_improvement",
    }
    if type(trace) is not list or len(trace) != completed + 1:
        raise OursError(
            "warm-start validation_trace must contain epoch zero and every completed epoch"
        )
    previous_best = record["source_validation_loss"]
    previous_stale = 0
    for expected_epoch, entry in enumerate(trace):
        if type(entry) is not dict or set(entry) != trace_fields:
            raise OursError("warm-start validation_trace entry is not the exact current schema")
        if type(entry["epoch"]) is not int or entry["epoch"] != expected_epoch:
            raise OursError("warm-start validation_trace epochs are not contiguous from zero")
        for field in ("validation_loss", "effective_min_delta", "best_validation_loss"):
            if type(entry[field]) is not float or not math.isfinite(entry[field]):
                raise OursError(f"warm-start validation_trace {field} must be finite float")
        if (
            entry["effective_min_delta"] < 0.0
            or type(entry["meaningful_improvement"]) is not bool
            or type(entry["epochs_without_improvement"]) is not int
            or entry["epochs_without_improvement"] < 0
        ):
            raise OursError("warm-start validation_trace decision fields are malformed")
        expected_delta = max(
            record["absolute_min_delta"],
            record["relative_min_delta"] * abs(previous_best),
        )
        if entry["effective_min_delta"] != expected_delta:
            raise OursError("warm-start validation_trace effective threshold is inconsistent")
        if expected_epoch == 0:
            if (
                entry["validation_loss"] != previous_best
                or entry["meaningful_improvement"] is not True
                or entry["best_validation_loss"] != previous_best
                or entry["epochs_without_improvement"] != 0
            ):
                raise OursError("warm-start validation_trace does not initialise source epoch zero")
            continue
        improved = entry["validation_loss"] < previous_best - expected_delta
        if entry["meaningful_improvement"] is not improved:
            raise OursError("warm-start validation_trace improvement decision is inconsistent")
        expected_best = entry["validation_loss"] if improved else previous_best
        expected_stale = 0 if improved else previous_stale + 1
        if (
            entry["best_validation_loss"] != expected_best
            or entry["epochs_without_improvement"] != expected_stale
        ):
            raise OursError("warm-start validation_trace best/staleness state is inconsistent")
        if expected_stale >= record["patience"] and expected_epoch != completed:
            raise OursError(
                "warm-start validation_trace continued after validation patience elapsed"
            )
        previous_best, previous_stale = expected_best, expected_stale
    if (
        previous_best != record["best_validation_loss"]
        or previous_stale != record["epochs_without_improvement"]
    ):
        raise OursError("warm-start validation_trace final state contradicts its summary")


def _validate_event_recipe(events: object, *, label: str) -> tuple[str, ...]:
    """Validate one event recipe without touching Spark or panel data."""
    if not isinstance(events, list | tuple):
        raise OursError(f"{label} are missing or malformed")
    seen_occurrences: set[tuple[str, str]] = set()
    names: list[str] = []
    for event in events:
        if not isinstance(event, list | tuple) or len(event) != 3:
            raise OursError(f"{label} are missing or malformed")
        name, when, known_at = event
        try:
            event_day = dt.date.fromisoformat(when) if type(when) is str else None
            known_day = (
                None
                if known_at is None
                else dt.date.fromisoformat(known_at)
                if type(known_at) is str
                else None
            )
        except ValueError as exc:
            raise OursError(f"{label} are missing or malformed") from exc
        if (
            type(name) is not str
            or name != name.strip()
            or not name
            or event_day is None
            or event_day.isoformat() != when
            or (known_at is not None and (known_day is None or known_day.isoformat() != known_at))
        ):
            raise OursError(f"{label} are missing or malformed")
        occurrence = (name, when)
        if occurrence in seen_occurrences:
            raise OursError(f"{label} repeat one named occurrence")
        seen_occurrences.add(occurrence)
        if name not in names:
            names.append(name)
    return tuple(names)


@_with_isolated_feature_scratch
def fit_predict(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    encoder_days: int = 28,
    d_model: int = 32,
    n_draws: int = 8,
    epochs: int = 3,
    n_paths: int = 1000,
    seed: int = 0,
    first_build: bool = True,
    paths_out_dir: str | pathlib.Path | None = None,
    diagnostics_out: str | pathlib.Path | None = None,
    paths_sample: int = 2000,
    events: tuple[tuple[str, str, str | None], ...] = (),
    event_max_support: int = 30,
    event_windows: dict[str, tuple[int, int]] | None = None,
    train_windows: int = 1,
    train_stride: int = 7,
    cycles: tuple[tuple[str, float, int], ...] = SEASONAL_CYCLES,
    per_series_seasonality: bool = False,
    demand_dropout: float = 0.15,
    covariate_dropout: float = 0.2,
    price_penalty_weight: float = 0.1,
    price_response_mode: str = "soft_penalty_v1",
    price_occurrence_slope_max: float | None = None,
    price_size_slope_max: float | None = None,
    aggregate_trajectory_mode: str = "none",
    price_ladder: str = "report",
    ladder_sample: int = 2000,
    ladder_bumps: tuple[float, ...] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2),
    ladder_tolerance: float = 1e-6,
    trunk_stages: int = 2,
    decoder_layers: int = 2,
    history_attention_layers: int = 1,
    commercial_attention_layers: int = 1,
    event_kernel_layers: int = 1,
    inference_factor_draws: int | None = None,
    point_factor_draws: int | None = None,
    point_factor_method: str = "gauss_hermite",
    factor_group_col: str | None = None,
    product_col: str | None = None,
    store_col: str | None = None,
    category_col: str | None = None,
    text_embeddings: str | pathlib.Path | None = None,
    checkpoint_out: str | pathlib.Path | None = None,
    training_log_dir: str | pathlib.Path | None = None,
    training_metrics_path: str | pathlib.Path | None = None,
    epoch_observer: Callable[[int, Mapping[str, float]], None] | None = None,
    resolved: ResolvedBuild | None = None,
    max_series_per_shard: int | None = None,
    device: str | None = None,
    precision: str = "32-true",
    batch_size: int = 4096,
    gradient_microbatch_size: int | None = None,
    val_fraction: float = 0.15,
    patience: int = 5,
    lr: float = 3e-3,
    warm_start_checkpoint: str | pathlib.Path | None = None,
    warm_start_checkpoint_sha256: str | None = None,
    recorded_panel_path: str | pathlib.Path | None = None,
    early_stopping_absolute_min_delta: float = 1e-6,
    early_stopping_relative_min_delta: float = 0.0,
    validation_seed: int | None = None,
    quiet_epoch_logging: bool = False,
    _feature_scratch: pathlib.Path,
) -> Forecast:
    """Train the custom model on eligible history to `origin` and forecast `horizon` days.

    Args:
        panel: Data.
        origin: Last day the model may see.
        horizon: Days to forecast.
        spark: Active session, for the feature build and for writing the result.
        out_dir: Parquet destination.
        encoder_days: Encoder window.
        d_model: Width. Deliberately small in the first build: Gate 4 asks whether the
            ARCHITECTURE beats the comparators, and a capacity advantage would confound that.
        trunk_stages: Alternating group/time transformer stages. This is executable model
            topology, not an implicit code default, and is saved with the checkpoint. The
            direct-API default preserves legacy checkpoints; governed runs receive their value
            from the typed model YAML.
        decoder_layers: Future decoder layers. This is executable model topology, not an
            implicit code default, and is saved with the checkpoint. The direct-API default
            preserves legacy checkpoints; governed runs receive their value from model YAML.
        history_attention_layers: Attention-pooling blocks over demand-history time tokens.
        commercial_attention_layers: Attention-pooling blocks over price/discount/promo tokens.
        event_kernel_layers: Residual spline-basis refinements in the calendar-event kernel.
            This is not attention and runs on compact kernel weights before panel-sized event
            features are formed.  The direct-API default preserves legacy checkpoints; governed
            runs receive their value from model YAML.
        n_draws: Factor draws per training step, for the IWAE bound. Ignored by the resolved
            rank-zero bottom-up model, whose marginal likelihood is exact and evaluated once.
        epochs: Passes over the training cells.
        n_paths: Sample paths at inference. 1,000 is the architecture's figure.
        seed: Torch seed, so a gate result is reproducible.
        device: `"cuda"`, `"cpu"`, or None to detect. **Without this the model trains on CPU on a
            GPU machine and nothing says so** -- the modules take their device from their inputs,
            so moving the inputs is the whole of it.
        batch_size: Soft cap on series rows per optimiser step. Shared-factor blocks are
            indivisible, so one unusually large block may exceed it; splitting that block would
            optimise a different IWAE objective. Full-batch training at 30,490 series takes one
            step per epoch, which is not training -- it is a smoke test that happens to converge
            on a constant.
        gradient_microbatch_size: Physical row cap used while accumulating one logical
            ``batch_size`` update. Shared-factor blocks remain indivisible, gradients are clipped
            and the optimiser steps only after the whole logical batch, and None uses
            ``batch_size``. This is an execution-memory control, not a second optimiser batch.
        val_fraction: Series held out to stop on. **Series, not days**: holding out days would
            let a series appear in both halves, and the residual embedding would memorise it.
        patience: Epochs without validation improvement before stopping. Early stopping is what
            makes `epochs` an upper bound rather than a guess.
        lr: Initial learning rate; cosine-annealed over the run.
        training_log_dir: Empty directory where per-epoch TensorBoard events are written. A
            retry must use a fresh directory; appending epochs from different model generations
            would create a false training curve.
        training_metrics_path: Empty JSONL destination for the same durable per-epoch records.
            The caller normally uses ``<attempt>/metrics/epochs.jsonl``.
        warm_start_checkpoint: Non-sharded source checkpoint whose weights become epoch zero.
            The file is hash-bound and copied into private fit scratch before it is loaded.
        warm_start_checkpoint_sha256: Required expected SHA-256 for a warm-start source.
        recorded_panel_path: Optional inert provenance path used in the checkpoint's Panel
            contract while every data read continues to use ``panel.path``. This supports a
            private authenticated data snapshot without changing the historical logical
            namespace named by a warm-start checkpoint.
        early_stopping_absolute_min_delta: Smallest absolute validation improvement.
        early_stopping_relative_min_delta: Smallest validation improvement relative to the
            absolute best loss. The effective threshold is the larger of the two deltas.
        validation_seed: Fixed validation Monte Carlo seed. Required for warm starts so the
            source and every candidate epoch use common random numbers.
        quiet_epoch_logging: Suppress per-epoch lines while retaining one final training summary.
        diagnostics_out: Where to write `model_diagnostics` as JSON, or None to discard them.
            **A run that records only its score cannot say why it got it**, and the model is not
            checkpointed -- so a diagnostic not written here is one the run has to be repeated to
            recover.
        paths_out_dir: Where to persist the drawn SAMPLE PATHS, or None to discard them. Gate 4's
            cumulative lead-time coverage cannot be computed from a point forecast, so a run that
            discards the paths cannot be gated.
        events: `(name, date, known_at)` triples for the calendar branch. Empty means this
            client supplied no event calendar, and the branch is then genuinely absent rather
            than fed zeros -- zero distance means "the event is today".
        event_max_support: Days either side of an event beyond which it is marked absent. Shared
            by the feature builder and the branch's kernel, so the window the features describe
            is the window the kernel integrates over.
        point_factor_method: How the point forecast integrates the factor. 'gauss_hermite'
            (default) is deterministic, convergence-checked and exact for a plain NB head; 'mc'
            samples. The sampled route was the only one until 2026-08-25, which is why a point
            estimand used to move with the seed.
        point_factor_draws: Factor realisations the POINT forecast integrates over under 'mc',
            ignored under 'gauss_hermite', or None for
            `POINT_FACTOR_DRAWS`. SEPARATE from `inference_factor_draws` because the two answer
            different questions: a mean needs integration accuracy (error ~ 1/sqrt(S)), a tail
            needs distinct systemic scenarios. Until 2026-08-25 one setting served both and was
            derived from `n_paths`, so asking for a larger path artifact silently moved the point
            estimate, and the catalogue-wide consumer paid the tail's sample count.
        inference_factor_draws: Distinct factor realisations the sampled PATHS integrate over,
            or None for one independent realisation per emitted path. **Not 1.** At 1 every path
            shares one demand shock, the factor contributes nothing to the spread, and the
            cumulative lead-time interval -- exactly the quantity the factor's cross-day
            dependence widens -- comes out too narrow. In rank-zero bottom-up mode this setting is
            inert because no factor exists; parameters are evaluated once and count draws remain
            independent per cell and step.
        demand_dropout: Dropout on the demand branch's patch tokens
            (`ForecastModelSpec.demand_branch_dropout`). Applied to TOKENS, never to raw days: a
            deleted day is indistinguishable from a day that sold nothing, which is the confusion
            the observation contract exists to prevent.
        covariate_dropout: Share of available commercial channels hidden per cell during training
            (`ForecastModelSpec.covariate_dropout`). It hides the channel's AVAILABILITY, not its
            value -- zeroing a price teaches "this product is free" rather than "the price is
            unknown here". Channel dropout supports datasets with different feature availability.
        price_penalty_weight: Weight on the hinge penalising demand that RISES with price
            under ``soft_penalty_v1``. ``monotone_raw_v1`` instead requires this to be exactly
            zero: its exclusive diagonal depth route supplies the shape guarantee by construction.
        price_response_mode: ``soft_penalty_v1`` (historical unrestricted paths plus hinge),
            ``monotone_raw_v1`` (exclusive structural depth route), or ``none``.
        price_occurrence_slope_max: Required finite positive occurrence-logit cap for the
            structural mode; otherwise None.
        price_size_slope_max: Required finite positive log-size cap for the structural mode;
            otherwise None.
        ladder_bumps: Relative price changes used by the post-fit monotonicity diagnostic. The
            configured values are persisted in the run record.
        ladder_tolerance: Largest permitted increase in the emitted point forecast between
            adjacent price bumps, checked independently for every series/day cell.
        per_series_seasonality: Let each series have its own annual AMPLITUDE, as a shared
            harmonic weight plus a shrunk per-series deviation -- the construction the event
            kernel already uses, for the same reason. **Off by default**: it is an addition on top
            of the pooled annual response, and `first_build` exists so that "the architecture
            works" is not confounded with "one of several additions helped". Turn it on in an
            ablation, which is what Phase 5 is.
        cycles: `(name, period_days, harmonics)` per cyclic coordinate, or `()` for none. See
            `SEASONAL_CYCLES`: the calendar columns encode week and month as LINEAR RAMPS, so
            Sunday sits maximally far from Monday and the 31st from the 1st. Harmonics are
            RESOLUTION -- "month of year" and "week of year" are the same 365.25-day cycle at
            different harmonic counts, not separate features.
        train_windows: How many rolling training windows each series contributes. **1 gives each
            series exactly ONE supervised example**, which is the minimum that is not zero and is
            not enough: the per-series residual embedding has one target to fit and can simply
            memorise it. It is the default only because it is the cheapest thing that is correct,
            and every real run should raise it. Measured on M5 at 1 window: validation was still
            improving when the budget ran out, and the lead-time intervals were too narrow.
        train_stride: Days between consecutive training windows. 7 by default so the windows step
            a whole week and each one starts on the same weekday -- a stride that is not a
            multiple of the weekly cycle makes consecutive windows differ by day-of-week phase as
            well as by level, which is a harder task than the one being forecast.
        factor_group_col: WHICH column the shared latent factor is indexed by, overriding
            `panel.factor_group`. **There is no defensible default**: chain, region and banner are
            all plausible and give different dependence. When neither location declares one, the
            model follows `ForecastModelSpec.resolve` and builds plain `bottom_up` independent
            marginals (rank 0). It never invents one global shock for the whole catalogue.
        product_col: WHICH column identifies the product, overriding `panel.product_col`. The
            contract indexes the factor path by `(factor_group, product, step)`; without a
            product column the path degrades to one per group, every product in the group rides
            one shock, and the run output says so rather than passing over it.
        store_col: Column selecting each series' factor LOADING row. Defaults to
            `panel.store_col`, then to the grouping column. Distinct from the group in
            principle: the group says which shock a series follows, the loading says how hard
            it is hit.
        category_col: WHICH level anchors the cold-start priors, overriding
            `panel.category_col`. The frequency baseline is Beta-shrunk toward this level's
            selling rate and the size baseline toward its mean positive size -- for a new
            product the only evidence there is. None falls back to panel-wide priors, reported
            rather than silent.
        text_embeddings: Parquet of `series_id · embedding (array<double>)` -- a FROZEN
            sentence embedding of each product's description, precomputed upstream. This is
            identity source 1 of the architecture's three: a newly listed product lands near
            similar ones BEFORE it has sold a unit. Frozen by construction here -- it enters as
            an input tensor and only its projection trains, so a description cannot memorise
            its series' demand level. None means the client has no descriptions (M5 does not),
            and the identity branch carries attributes and the residual alone.
        resolved: The `spec.resolved.ResolvedBuild` this run constructs from, or None for
            explicit runtime parameters. When given, its architecture fields OVERRIDE the
            corresponding keywords (head, d_model, attention heads, patch size, factor rank,
            dropouts, event ceiling, path count, price-monotone on/off), unsupported resolved
            values are refused before training, and the build hash is printed and recorded in
            the checkpoint -- which architecture trained becomes a fact, not a reconstruction.
        paths_sample: Series to persist paths for. The full set is `n_paths x n_series x horizon`
            -- 853 million values on M5 -- so a deterministic SAMPLE is written instead. 2,000
            series is ample for a coverage curve. Selection is the smallest stable
            SHA256(seed, series_id) ranks, a uniform sample without replacement that remains
            reproducible without clustering on a catalogue's lexical prefix.
        first_build: Must be True. See below.

    Returns:
        A validated `Forecast` whose `yhat` integrates the analytic conditional count mean over
        the factor by the declared point method (deterministic quadrature by default). Persisted
        integer paths are drawn from those same fitted parameters for calibration, not averaged
        into a noisier point estimate.

    Raises:
        OursError: `first_build=False`, or no series has a full encoder window.
    """
    if not first_build:
        raise OursError(
            "first_build=False refuses nothing, and Phase 4 is defined as the FIRST build: the "
            "adaptive event kernel, event interactions, the peer graph, the cross-product factor "
            "and the promotion decomposition are Phase 5, each earning its place by an "
            "episode-split ablation. Enabling them here would confound 'the architecture works' "
            "with 'one of five additions happened to help'"
        )
    if precision not in ("32-true", "16-mixed"):
        raise OursError("precision must be 32-true or 16-mixed")
    for name, value in (
        ("trunk_stages", trunk_stages),
        ("decoder_layers", decoder_layers),
        ("history_attention_layers", history_attention_layers),
        ("commercial_attention_layers", commercial_attention_layers),
        ("event_kernel_layers", event_kernel_layers),
    ):
        if type(value) is not int or value < 1:
            raise OursError(f"{name} must be a positive integer, got {value!r}")
    if aggregate_trajectory_mode not in ("none", AGGREGATE_TRAJECTORY_MODE):
        raise OursError(
            f"aggregate_trajectory_mode must be 'none' or {AGGREGATE_TRAJECTORY_MODE!r}, "
            f"got {aggregate_trajectory_mode!r}"
        )
    _validate_event_recipe(events, label="runtime events")
    if resolved is not None:
        # Validate the exact canonical producer record before any Spark or panel data read. A
        # checkpoint loader cannot be stricter than the producer that created its evidence.
        _validate_resolved_build_json_types(resolved.canonical())
    warm_start_enabled = warm_start_checkpoint is not None
    if warm_start_enabled and precision == "16-mixed":
        raise OursError("mixed precision requires a fresh MFT run, not historical warm start")
    if warm_start_enabled != (warm_start_checkpoint_sha256 is not None):
        raise OursError(
            "warm_start_checkpoint and warm_start_checkpoint_sha256 must be supplied together"
        )
    if type(quiet_epoch_logging) is not bool:
        raise OursError(f"quiet_epoch_logging must be boolean, got {quiet_epoch_logging!r}")
    recorded_panel = asdict(panel)
    if recorded_panel_path is not None:
        provenance_path = pathlib.Path(recorded_panel_path)
        if not provenance_path.is_absolute():
            raise OursError("recorded_panel_path must be absolute when supplied")
        recorded_panel["path"] = str(provenance_path)
    for name, value in (
        ("early_stopping_absolute_min_delta", early_stopping_absolute_min_delta),
        ("early_stopping_relative_min_delta", early_stopping_relative_min_delta),
    ):
        if isinstance(value, bool) or not is_finite_real(value) or float(value) < 0.0:
            raise OursError(f"{name} must be finite and nonnegative, got {value!r}")
    early_stopping_absolute_min_delta = float(early_stopping_absolute_min_delta)
    early_stopping_relative_min_delta = float(early_stopping_relative_min_delta)
    if validation_seed is not None and (type(validation_seed) is not int or validation_seed < 0):
        raise OursError(
            f"validation_seed must be a nonnegative integer or None, got {validation_seed!r}"
        )
    warm_start_source_path = None
    warm_start_snapshot = None
    warm_start_snapshot_state = None
    if warm_start_enabled:
        if max_series_per_shard is not None:
            raise OursError(
                "warm-start continuation is explicitly non-sharded; "
                "max_series_per_shard must be None"
            )
        if type(epochs) is not int or isinstance(epochs, bool) or epochs < 1:
            raise OursError(
                f"warm-start epochs must be a positive additional-epoch cap, got {epochs!r}"
            )
        if type(patience) is not int or isinstance(patience, bool) or patience < 1:
            raise OursError(f"warm-start patience must be a positive integer, got {patience!r}")
        if isinstance(lr, bool) or not is_finite_real(lr) or float(lr) <= 0.0:
            raise OursError(f"warm-start lr must be finite and positive, got {lr!r}")
        if validation_seed is None:
            raise OursError(
                "warm-start continuation requires validation_seed for common random numbers"
            )
        if checkpoint_out is None:
            raise OursError(
                "warm-start continuation requires checkpoint_out for lineage and stopping trace"
            )
        warm_start_source_path = pathlib.Path(warm_start_checkpoint)
        output_paths = {
            "out_dir": out_dir,
            "paths_out_dir": paths_out_dir,
            "diagnostics_out": diagnostics_out,
            "checkpoint_out": checkpoint_out,
        }
        for output_name, output_path in output_paths.items():
            if output_path is None:
                continue
            destination = pathlib.Path(output_path)
            same_inode = False
            if warm_start_source_path.exists() and destination.exists():
                try:
                    same_inode = os.path.samefile(warm_start_source_path, destination)
                except OSError:
                    same_inode = False
            if same_inode or _paths_overlap(warm_start_source_path, destination):
                raise OursError(
                    f"warm-start {output_name} must use a separate namespace from the source"
                )
        warm_start_snapshot = _snapshot_warm_start_checkpoint(
            warm_start_source_path, warm_start_checkpoint_sha256, _feature_scratch
        )
        warm_start_snapshot_state = _private_snapshot_state(
            warm_start_snapshot, include_parent=True
        )
    torch.manual_seed(seed)

    # EVERY DECLARED ROLE IS CONSUMED OR REFUSED -- never silently ignored. The generic data
    # contract lets a caller declare known-future covariates by name; this adapter consumes the
    # DERIVED known-future set (calendar, cycles, commercial channels with availability, event
    # distances) plus the declared price and promotion columns, and nothing else. A declared
    # role the model never reads is a capability the caller believes they bought: refusing it
    # here costs one error message, honouring it silently costs a client a covariate they
    # think is in the model.
    consumed_known = {c for c in (panel.price_col, panel.promo_col) if c is not None}
    unconsumed = [c for c in panel.known_future if c not in consumed_known]
    if unconsumed:
        raise OursError(
            f"panel declares known-future column(s) {unconsumed} this adapter does not consume. "
            "It reads price_col and promo_col through the commercial builder and derives the "
            "rest (calendar, cycles, event distances); an arbitrary known-future role has no "
            "path into the model, and continuing would silently drop a covariate the caller "
            "declared. Remove the declaration, or map the column onto a supported role"
        )

    # THE RESOLVED BUILD, when given, is the architecture: `ResolvedBuild.from_specs` has
    # already run the resolver and refused any combination `validate()` blocks, so by the time
    # a build hash is printed here, "reject unsupported combinations before training" has
    # happened. Its fields override the corresponding keyword defaults -- a caller who resolves
    # a spec and then hand-tunes past it is running an architecture the hash does not describe.
    head_kind = "hurdle_negbin"
    n_heads, patch_size = 4, 7
    static_selection = True
    if resolved is not None:
        if resolved.input_normalisation != "revin":
            raise OursError(
                f"resolved input_normalisation={resolved.input_normalisation!r} is not "
                "implemented by this adapter; RevIN is the one built normalisation"
            )
        low_support, high_support = resolved.event_max_support
        if -low_support != high_support:
            raise OursError(
                f"resolved event_max_support={resolved.event_max_support} is asymmetric; this "
                "adapter's kernel takes one symmetric ceiling"
            )
        if resolved.emit != "sample_paths" and paths_out_dir is not None:
            raise OursError(
                f"resolved emit={resolved.emit!r} but paths_out_dir is set: the resolver stepped "
                "emit down because coherent paths are not meaningful for this configuration, and "
                "persisting them anyway would publish the thing the degradation withdrew"
            )
        expected_event_kernel = "fixed_window" if events else None
        if resolved.event_kernel != expected_event_kernel:
            raise OursError(
                f"resolved event_kernel={resolved.event_kernel!r} but runtime events imply "
                f"{expected_event_kernel!r}; the build hash and constructed calendar branch "
                "would disagree"
            )
        head_kind = resolved.head
        d_model = resolved.d_model
        n_heads = resolved.attention_heads
        patch_size = resolved.patch_size
        event_max_support = high_support
        demand_dropout = resolved.demand_branch_dropout
        covariate_dropout = resolved.covariate_dropout
        static_selection = resolved.static_variable_selection
        n_paths = resolved.n_sample_paths
        price_response_mode = resolved.price_response_mode
        price_occurrence_slope_max = resolved.price_occurrence_slope_max
        price_size_slope_max = resolved.price_size_slope_max
        price_penalty_weight = _resolved_price_penalty_weight(resolved, panel, price_penalty_weight)
        print(
            f"  resolved build {resolved.build_hash[:12]}: head={head_kind}, "
            f"d_model={d_model}, factor_rank={resolved.factor_rank}, "
            f"zero_share={resolved.zero_share}",
            flush=True,
        )

    declared_commercial = _declared_commercial_drivers(panel)
    if "price" not in declared_commercial:
        if price_response_mode == "monotone_raw_v1":
            raise OursError(
                "price_response_mode='monotone_raw_v1' requires a declared horizon price"
            )
        price_response_mode = "none"
        price_occurrence_slope_max = None
        price_size_slope_max = None
        price_penalty_weight = 0.0
    if price_response_mode == "monotone_raw_v1":
        if "discount_depth" not in declared_commercial:
            raise OursError(
                "price_response_mode='monotone_raw_v1' requires historical price at or before "
                "the origin so the planned price has a frozen-reference discount_depth"
            )
        for name, value in (
            ("price_occurrence_slope_max", price_occurrence_slope_max),
            ("price_size_slope_max", price_size_slope_max),
        ):
            if isinstance(value, bool) or not is_finite_real(value) or float(value) <= 0.0:
                raise OursError(
                    f"{name} must be finite and positive for monotone_raw_v1, got {value!r}"
                )
        price_occurrence_slope_max = float(price_occurrence_slope_max)
        price_size_slope_max = float(price_size_slope_max)
        if not is_finite_real(price_penalty_weight) or float(price_penalty_weight) != 0.0:
            raise OursError(
                "price_response_mode='monotone_raw_v1' requires "
                f"price_penalty_weight=0.0, got {price_penalty_weight!r}"
            )
    elif price_response_mode == "none":
        if not is_finite_real(price_penalty_weight) or float(price_penalty_weight) != 0.0:
            raise OursError("price_response_mode='none' requires price_penalty_weight=0.0")
    elif price_response_mode != "soft_penalty_v1":
        raise OursError(f"unsupported price_response_mode={price_response_mode!r}")
    if price_response_mode != "monotone_raw_v1" and (
        price_occurrence_slope_max is not None or price_size_slope_max is not None
    ):
        raise OursError(
            "price slope caps are executable only for price_response_mode=" "'monotone_raw_v1'"
        )

    # These controls are consumed only after optimisation, so validate them here or a malformed
    # artifact request can waste the whole training run before failing (and sharded runs used to
    # accept unknown price-ladder modes outright).
    if point_factor_method not in ("gauss_hermite", "mc"):
        raise OursError(
            f"point_factor_method must be gauss_hermite or mc, got {point_factor_method!r}"
        )
    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 1:
        raise OursError(f"n_paths must be a positive integer, got {n_paths!r}")
    if (
        point_factor_method == "mc"
        and point_factor_draws is not None
        and (
            not isinstance(point_factor_draws, int)
            or isinstance(point_factor_draws, bool)
            or point_factor_draws < 1
        )
    ):
        raise OursError(
            f"point_factor_draws must be a positive integer under mc, got "
            f"{point_factor_draws!r}"
        )
    if inference_factor_draws is not None:
        if (
            not isinstance(inference_factor_draws, int)
            or isinstance(inference_factor_draws, bool)
            or not 1 <= inference_factor_draws <= n_paths
        ):
            raise OursError(
                f"inference_factor_draws must lie in [1, n_paths={n_paths}], got "
                f"{inference_factor_draws!r}"
            )
        if n_paths % inference_factor_draws:
            raise OursError(
                f"n_paths={n_paths} is not divisible by "
                f"inference_factor_draws={inference_factor_draws}"
            )
    if not isinstance(paths_sample, int) or isinstance(paths_sample, bool) or paths_sample < 1:
        raise OursError(f"paths_sample must be a positive integer, got {paths_sample!r}")
    if price_ladder not in ("report", "require", "skip"):
        raise OursError(f"price_ladder must be report, require or skip, got {price_ladder!r}")
    if isinstance(ladder_sample, bool) or not isinstance(ladder_sample, int) or ladder_sample < 1:
        raise OursError(f"ladder_sample must be a positive integer, got {ladder_sample!r}")
    _validate_price_ladder_controls(ladder_bumps, ladder_tolerance)

    # Encoder inputs end before each training target window. Targets end at or before origin,
    # so training learns future prediction without observing its own target.
    if train_windows < 1:
        raise OursError(f"train_windows must be >= 1, got {train_windows}")
    if train_stride < 1:
        raise OursError(f"train_stride must be >= 1, got {train_stride}")
    if batch_size < 1:
        raise OursError(f"batch_size must be >= 1, got {batch_size}")
    if gradient_microbatch_size is None:
        gradient_microbatch_size = batch_size
    if (
        isinstance(gradient_microbatch_size, bool)
        or not isinstance(gradient_microbatch_size, int)
        or gradient_microbatch_size < 1
    ):
        raise OursError(
            "gradient_microbatch_size must be a positive integer or None, got "
            f"{gradient_microbatch_size!r}"
        )
    origin_day = dt.date.fromisoformat(origin)
    # Window i: the encoder ends `horizon + i*stride` days back and its target is the `horizon`
    # days following it. Every one of them lies wholly in the past of `origin`, so adding windows
    # adds supervision without moving the leak boundary.
    train_origins = [
        (origin_day - dt.timedelta(days=horizon + i * train_stride)).isoformat()
        for i in range(train_windows)
    ]
    target_origins = [
        (origin_day - dt.timedelta(days=i * train_stride)).isoformat() for i in range(train_windows)
    ]
    # Every builder writes with Spark's overwrite mode. The namespace must therefore be private
    # to this invocation, not merely to the origin: equal-origin fits are ordinary in parallel
    # experiments, and otherwise one can replace another's plausible-looking prices between the
    # build and read. The public wrapper removes this tree on success and exception.
    work_dir = _feature_scratch

    # THE BOUNDED-MEMORY PATH. A shard is a set of whole factor blocks selected by a hash of
    # the block key; when the cap yields more than one, assembly, training tensors and
    # inference all proceed one shard at a time (see `_fit_sharded`). One shard is exactly
    # today's path, so the cap being absent changes nothing.
    _grouping_col = factor_group_col if factor_group_col is not None else panel.factor_group
    _product_col = product_col if product_col is not None else panel.product_col
    _store_col = store_col if store_col is not None else (panel.store_col or _grouping_col)
    _category_col = category_col if category_col is not None else panel.category_col
    if _store_col is not None and _product_col is None and _grouping_col is None:
        raise OursError(
            f"store_col={_store_col!r} has no executable role without product_col or "
            "factor_group_col; refusing instead of silently ignoring a declared identity"
        )
    _validate_resolved_topology(resolved, _grouping_col)
    _validate_series_role_constancy(
        panel,
        spark,
        (
            _grouping_col,
            _product_col,
            _store_col,
            _category_col,
            *panel.static,
            *panel.static_reals,
        ),
    )
    n_total = spark.read.parquet(panel.path).select(panel.series_col).distinct().count()
    shard_predicates = shard_plan(
        panel, spark, grouping=_grouping_col, product=_product_col, max_series=max_series_per_shard
    )
    if len(shard_predicates) > 1:
        unsupported = []
        if paths_out_dir is not None:
            unsupported.append("paths_out_dir (sample paths)")
        if diagnostics_out is not None:
            unsupported.append("diagnostics_out (diagnostics and architecture.json)")
        if text_embeddings is not None:
            unsupported.append("text_embeddings (text identity)")
        if price_ladder != "skip" and "price" in _declared_commercial_drivers(panel):
            unsupported.append(
                f"price_ladder={price_ladder!r} (global sampled verification is not "
                "implemented for sharded fit)"
            )
        if unsupported:
            raise OursError(
                "sharded fit cannot emit or consume the requested output(s): "
                f"{', '.join(unsupported)}. Refusing before training rather than completing a "
                "multi-hour fit with missing artifacts or a silently disabled branch"
            )
        return _fit_sharded(
            panel,
            origin,
            horizon,
            spark=spark,
            out_dir=out_dir,
            predicates=shard_predicates,
            encoder_days=encoder_days,
            d_model=d_model,
            n_draws=n_draws,
            epochs=epochs,
            n_paths=n_paths,
            paths_sample=paths_sample,
            seed=seed,
            events=events,
            event_max_support=event_max_support,
            event_windows=event_windows,
            train_origins=train_origins,
            target_origins=target_origins,
            train_stride=train_stride,
            train_windows=train_windows,
            cycles=cycles,
            per_series_seasonality=per_series_seasonality,
            demand_dropout=demand_dropout,
            covariate_dropout=covariate_dropout,
            price_penalty_weight=price_penalty_weight,
            price_response_mode=price_response_mode,
            price_occurrence_slope_max=price_occurrence_slope_max,
            price_size_slope_max=price_size_slope_max,
            aggregate_trajectory_mode=aggregate_trajectory_mode,
            inference_factor_draws=inference_factor_draws,
            point_factor_draws=point_factor_draws,
            point_factor_method=point_factor_method,
            grouping=_grouping_col,
            product=_product_col,
            store=_store_col,
            category=_category_col,
            resolved=resolved,
            head_kind=head_kind,
            n_heads=n_heads,
            patch_size=patch_size,
            trunk_stages=trunk_stages,
            decoder_layers=decoder_layers,
            history_attention_layers=history_attention_layers,
            commercial_attention_layers=commercial_attention_layers,
            event_kernel_layers=event_kernel_layers,
            static_selection=static_selection,
            text_embeddings=text_embeddings,
            checkpoint_out=checkpoint_out,
            device=device,
            precision=precision,
            batch_size=batch_size,
            gradient_microbatch_size=gradient_microbatch_size,
            val_fraction=val_fraction,
            patience=patience,
            lr=lr,
            work_dir=work_dir,
            n_total=n_total,
            max_series_per_shard=max_series_per_shard,
            training_log_dir=training_log_dir,
            training_metrics_path=training_metrics_path,
            epoch_observer=epoch_observer,
        )
    # WHICH level anchors the cold-start priors. The spec shrinks the frequency baseline toward
    # "a category rate" and the size baseline toward "the CATEGORY's mean positive size"; with
    # no level declared both fall back to panel-wide figures, and the run says so, because on a
    # mixed panel that pulls every launch toward the fast-moving category's numbers.
    category = category_col if category_col is not None else panel.category_col
    category_of = _series_levels(panel, spark, category) if category else None
    if category is None:
        print("  baseline priors: PANEL-WIDE -- no category_col declared", flush=True)
    else:
        n_levels = len({v for v in category_of.values() if v is not None})
        print(f"  baseline priors per {category} ({n_levels} levels)", flush=True)
    infer = _assemble(
        panel,
        spark,
        origin,
        horizon,
        encoder_days,
        events,
        event_max_support,
        work_dir / "infer",
        cycles,
        category_of=category_of,
        event_windows=event_windows,
    )
    if not infer.series_ids:
        raise OursError(f"no series has a full {encoder_days}-day encoder window at {origin}")
    if infer.plan is not None:
        print(
            f"  plan provenance: {infer.plan.n_rows:,} as-of rows, "
            f"selection {infer.plan.selection_hash[:12]} (known_at <= {origin})",
            flush=True,
        )

    train = [
        _assemble(
            panel,
            spark,
            o,
            horizon,
            encoder_days,
            events,
            event_max_support,
            work_dir / f"train{i}",
            cycles,
            category_of=category_of,
            event_windows=event_windows,
        )
        for i, o in enumerate(train_origins)
    ]

    # The supervised target for each window: the `horizon` days ending at that window's target
    # origin, which are exactly the days following its encoder window. Built by the SAME builder
    # as the encoder, so "the days after the window" and "the days the target covers" cannot
    # drift apart.
    target_rows = []
    for i, o in enumerate(target_origins):
        path = demand_history.build(
            panel.path,
            o,
            horizon,
            spark,
            str(work_dir / f"_target{i}"),
            series_col=panel.series_col,
            time_col=panel.time_col,
            target_col=panel.target_col,
            exposure_col=panel.exposure_col,
            oos_col=panel.oos_col,
            status_prob_cols=panel.status_prob_cols,
            negative_target_policy=panel.negative_target_policy,
        )
        target_rows.append({r["series_id"]: r for r in spark.read.parquet(str(path)).collect()})

    series_ids = infer.series_ids  # what gets FORECAST, and the index space
    n_series = len(series_ids)
    position = {sid: i for i, sid in enumerate(series_ids)}

    # Static identity and the factor grouping are properties of the SERIES, not of an origin, so
    # they are built once over the forecast index space and shared by both assemblies.
    attributes, attribute_names = _identity_tensors(panel, spark, series_ids, work_dir)
    if panel.static_reals:
        reals, real_names, real_stats = _static_real_tensors(panel, spark, series_ids)
        attributes = torch.cat([attributes, reals], dim=1)
        attribute_names = (*attribute_names, *real_names)
        print(f"  static reals: {real_names} (standardised on this panel)", flush=True)
    else:
        real_stats = None
    text, text_dim, text_missing = _text_tensors(text_embeddings, spark, series_ids)
    if text is not None:
        print(
            f"  text identity: {text_dim}-wide frozen embeddings"
            f"{f' ({text_missing:,} series without one, zero-filled)' if text_missing else ''}",
            flush=True,
        )
    # Columns per attribute GROUP, so selection weights an attribute rather than a level of one.
    attribute_widths = (
        tuple(sum(1 for n in attribute_names if n.startswith(f"{c}=")) for c in panel.static)
        if panel.static
        else ()
    )
    if panel.static_reals and attribute_widths:
        # Each real is its OWN selection group: what is selected is the attribute, and a
        # standardised scalar is already one attribute wide.
        attribute_widths = (*attribute_widths, *(1 for _ in panel.static_reals))
    # THE FACTOR'S TWO INDEX SETS, composed as the likelihood contract states them: the PATH is
    # drawn per (factor_group, product, step), the LOADINGS are per store. An earlier version
    # indexed the path by the bare factor_group -- and the runs then passed store_id, so every
    # product in a store shared one realisation (with identical loadings, so identical factor
    # terms) while STORES were mutually independent. That is a store-day random effect, not the
    # cross-store dependence the mechanism exists to supply, and the aggregate uncertainty the
    # design motivates was uncovered at every level above a single store.
    grouping = factor_group_col if factor_group_col is not None else panel.factor_group
    product = product_col if product_col is not None else panel.product_col
    product_index, n_products, _, product_levels, product_values = _grouping(
        panel, spark, series_ids, product
    )
    n_products_identity = n_products if product is not None else 0
    factor_enabled = grouping is not None
    if factor_enabled:
        _, _, group_note, factor_levels, factor_values = _grouping(
            panel, spark, series_ids, grouping
        )
        if factor_values is None:
            raise OursError("declared factor grouping produced no semantic values")
        group_index, n_groups = _query_composite_groups(series_ids, factor_values, product_values)
        if product is None:
            group_note += (
                " -- NO product_col: one path per group is shared by EVERY product in "
                "it, so all products ride one shock. Declare product_col for the "
                "contract's (factor_group, product) indexing"
            )
        else:
            group_note = f"factor paths per ({grouping} x {product}): " f"{n_groups:,} blocks"
    else:
        # Plain bottom-up: one likelihood/batching block per marginal series.  In particular, do
        # not reuse `_grouping(None)`, whose single fallback level is useful for optional identity
        # indices but would recreate the full-catalogue block this degradation must avoid.
        group_index = _independent_group_index(n_series)
        n_groups = n_series
        factor_levels = []
        group_note = (
            "shared factor DISABLED: no factor_group_col declared; topology bottom_up "
            f"with {n_groups:,} independent marginal blocks"
        )
    loading_col = store_col if store_col is not None else (panel.store_col or grouping)
    store_index, n_stores, store_note, store_levels, _ = _grouping(
        panel, spark, series_ids, loading_col
    )

    annual_h = next((h for n, _, h in cycles if n == "annual"), 0)
    annual_index = (
        tuple(infer.known.index(c) for c in cycle_columns("annual", annual_h)) if annual_h else ()
    )
    # ONE config dict builds the model AND rides the checkpoint, so a reload constructs exactly
    # what trained -- a checkpoint holding weights without their architecture is a puzzle, not
    # a record.
    commercial_indices = _commercial_indices_from_roles(
        tuple(infer.known.names), infer.known.declared_drivers
    )
    branch_commercial = (
        int(commercial_indices["promo_index"] is not None)
        if price_response_mode == "monotone_raw_v1"
        else infer.n_commercial
    )
    model_config = dict(
        d_model=d_model,
        encoder_days=encoder_days,
        n_attributes=attributes.shape[1],
        # ONE EXTRA ROW per identity table: the OOV bucket. After the best weights are restored,
        # `_finalize_oov_parameters` makes every unevidenced deviation exactly zero and every
        # unevidenced factor loading the evidenced-row mean. A random initial residual is not a
        # pooled state, even when its variance is small.
        n_series=n_series + 1,
        n_known=infer.known.values.shape[-1],
        horizon=horizon,
        n_commercial=branch_commercial,
        n_events=infer.n_events,
        n_stores=n_stores + 1,
        factor_rank=(
            (resolved.factor_rank if resolved is not None else 8) if factor_enabled else 0
        ),
        factor_dynamics=_factor_dynamics_for_build(resolved, factor_enabled),
        embedding_residual_dim=(resolved.embedding_residual_dim if resolved is not None else 16),
        event_max_support=event_max_support,
        event_windows=(
            None
            if event_windows is None
            else tuple(
                tuple(event_windows.get(name, (-event_max_support, event_max_support)))
                for name in dict.fromkeys(name for name, _, _ in events)
            )
        ),
        head_kind=head_kind,
        n_heads=n_heads,
        patch_size=patch_size,
        trunk_stages=trunk_stages,
        decoder_layers=decoder_layers,
        history_attention_layers=history_attention_layers,
        commercial_attention_layers=commercial_attention_layers,
        event_kernel_layers=event_kernel_layers,
        annual_index=annual_index,
        per_series_seasonality=per_series_seasonality,
        attribute_widths=attribute_widths,
        demand_dropout=demand_dropout,
        price_response_mode=price_response_mode,
        price_occurrence_slope_max=price_occurrence_slope_max,
        price_size_slope_max=price_size_slope_max,
        aggregate_trajectory_mode=aggregate_trajectory_mode,
        **commercial_indices,
        event_distance_index=tuple(
            infer.known.index(f"{name}_distance")
            for name in dict.fromkeys(name for name, _, _ in events)
        ),
        event_present_index=tuple(
            infer.known.index(f"{name}_present")
            for name in dict.fromkeys(name for name, _, _ in events)
        ),
        n_products=(n_products_identity + 1 if product is not None else 0),
        text_dim=text_dim,
    )
    if not static_selection:
        # `static_variable_selection=False` in the resolved spec: the identity branch receives
        # the attributes unweighted, and the interpretable selection weights simply do not
        # exist -- absent, not zeroed.
        model_config["attribute_widths"] = ()
    model = CustomModel(**model_config)
    # The known-future columns each commercial channel owns, for decoder-side dropout: hiding a
    # channel must hide it on BOTH paths, and these indices are resolved once, by name, so a
    # layout change fails loudly here rather than zeroing someone else's column.
    channel_columns = (
        [(infer.known.index(c), infer.known.index(f"{c}_available")) for c in commercial.CHANNELS]
        if infer.n_commercial
        else []
    )
    print(
        f"  branches: demand, identity"
        f"{', commercial' if model.commercial is not None else ''}"
        f"{', calendar' if infer.n_events else ''}"
        f"  ({model.n_branches} of 5; peer is Phase 5)",
        flush=True,
    )
    print(
        f"  identity: {len(attribute_names)} attribute columns"
        f"{' (NO STATIC FEATURES -- constant missing-static input)' if not panel.static else ''}"
        + (
            f", factorised residual ({n_products_identity:,} products + {n_stores:,} stores)"
            if n_products_identity
            else ", per-series residual"
        ),
        flush=True,
    )
    print(f"  known-future columns: {infer.known.names}", flush=True)
    print(
        f"  encoder {encoder_days}d, cycles {[(n, h) for n, _, h in cycles]}"
        f"{', per-series amplitude' if model.seasonality is not None else ' (pooled)'}",
        flush=True,
    )
    print(f"  {group_note}", flush=True)
    if factor_enabled and store_note != group_note:
        print(f"  factor loadings from {store_note}", flush=True)

    def _window_tensors(a: _Assembly, take: torch.Tensor, rows: torch.Tensor) -> dict:
        """Aligned inputs for a set of series, indexed explicitly on BOTH axes.

        A training assembly has its own series axis (whichever series had a full window at
        that origin), while attributes, the factor indices and the residual embedding live on
        the FORECAST axis. The two need different index vectors, applied here, once. The
        earlier version routed both through a `_subset` that guessed which tensors were
        already aligned from their first dimension — and that guess is wrong exactly when the
        sizes coincide, which is the case where nothing else fails either.

        Args:
            a: The assembly the encoder-side tensors come from.
            take: Positions in `a.series_ids` of the series wanted, in order.
            rows: The SAME series' positions in the forecast index space.
        """
        return dict(
            history=a.history[take],
            observed=a.observed[take],
            censored=a.censored[take],
            attributes=attributes[rows],
            series_index=rows,
            text=None if text is None else text[rows],
            known_future=a.known.values[take],
            baseline_logit=a.baseline_logit[take],
            baseline_log_size=a.baseline_log_size[take],
            group_index=group_index[rows],
            store_index=store_index[rows],
            product_index=product_index[rows],
            # An IWAE draw axis is meaningful only with a latent factor. Plain bottom-up
            # has exact marginal likelihood, so one identical slice is sufficient.
            n_draws=n_draws if factor_enabled else 1,
            commercial=None if a.commercial is None else a.commercial[take],
            commercial_mask=(None if a.commercial_mask is None else a.commercial_mask[take]),
            event_distance=(None if a.event_distance is None else a.event_distance[take]),
            event_present=(None if a.event_present is None else a.event_present[take]),
        )

    def _inputs(a: _Assembly, rows: list[int]) -> dict:
        """The model's inputs for an assembly whose series ARE the forecast index space.

        Valid only for the inference assembly: its tensors are already aligned with
        `series_ids`, so one index vector selects the same series everywhere. Training
        windows go through `_window_tensors`, whose two axes need different indices.
        """
        if a.series_ids != series_ids:
            raise OursError(
                "_inputs was given an assembly whose series axis differs from the forecast "
                "index; use _window_tensors, which indexes the two axes separately"
            )
        keep = torch.tensor(rows, dtype=torch.long)
        return _window_tensors(a, keep, keep)

    # A series can be forecast at `origin` and still lack a training window further back -- a
    # product launched inside that gap has one but not the other. Those series are FORECAST but
    # not TRAINED ON, which is the cold-start path the identity branch exists to serve. Each
    # window is filtered independently, so a series contributes however many windows it has.
    pieces, targets, states, series_rows, probs_pieces = [], [], [], [], []
    next_factor_block = 0
    for i, assembly in enumerate(train):
        usable = [sid for sid in assembly.series_ids if sid in position and sid in target_rows[i]]
        if not usable:
            print(
                f"    window {i} ({train_origins[i]} -> {target_origins[i]}): no series has "
                "both an encoder window and a target; skipped",
                flush=True,
            )
            continue
        at = {sid: j for j, sid in enumerate(assembly.series_ids)}
        take = torch.tensor([at[sid] for sid in usable], dtype=torch.long)
        rows = torch.tensor([position[sid] for sid in usable], dtype=torch.long)
        piece = _window_tensors(assembly, take, rows)
        # A latent path belongs to ONE forecast episode. The same (group, product) observed at
        # two rolling origins is two independent factor blocks, not two observations of one
        # frozen future shock. Offset each window's dense ids before concatenation so validation
        # and any full-window objective cannot accidentally contract them together.
        piece["group_index"], next_factor_block = _offset_dense_blocks(
            piece["group_index"], next_factor_block
        )
        pieces.append(piece)
        targets.append(
            torch.tensor([target_rows[i][sid]["history"] for sid in usable], dtype=torch.float32)
        )
        # THE STATE CODES, straight through. This line used to rebuild the state from the
        # `observed` BOOLEAN -- eligible became `exact`, everything else `unobserved` -- which
        # made `lower_bound` unreachable from training on any panel that has it: a stockout
        # that sold 5 before the shelf emptied was scored as exactly 5 (the bias censoring
        # exists to remove), and an all-day stockout as an observed zero (a cell the contract
        # says carries nothing). The likelihood, the head and the parity tests all handled
        # state 1; this mapping was the one link that could not express it.
        raw_target_state = torch.tensor(
            [target_rows[i][sid]["state"] for sid in usable], dtype=torch.long
        )
        states.append(_states_for_estimand(raw_target_state, panel.target_estimand))
        if panel.status_prob_cols is not None:
            window_probs = torch.stack(
                [
                    torch.tensor([target_rows[i][sid][col] for sid in usable], dtype=torch.float32)
                    for col in ("p_exact", "p_lower_bound", "p_unobserved")
                ],
                dim=-1,
            )
            probs_pieces.append(_probs_for_estimand(window_probs, panel.target_estimand))
        series_rows.append(rows)
    if not pieces:
        raise OursError(
            f"no series has both a {encoder_days}-day encoder window and {horizon} target days "
            f"in any of the {train_windows} training window(s) ending {target_origins}. "
            "Training would have no supervised signal"
        )

    # Windows stack along the BATCH axis: one row is one (series, window) pair, and a series
    # appears once per window it has. The per-series embedding is shared across its own windows,
    # which is the point -- that is what turns one example per series into several.
    target = torch.cat(targets)
    target_state = torch.cat(states)
    target_probs = torch.cat(probs_pieces) if probs_pieces else None
    train_inputs = {
        k: (
            torch.cat([p[k] for p in pieces])
            if isinstance(pieces[0][k], torch.Tensor)
            else pieces[0][k]
        )
        for k in pieces[0]
    }
    example_series = torch.cat(series_rows)
    print(
        f"  training on {len(target):,} examples from {len(set(example_series.tolist())):,} "
        f"of {n_series:,} series ({len(pieces)} window(s), encoder ends "
        f"{train_origins[0]}, target ends {target_origins[0]})",
        flush=True,
    )

    # Feature assembly is complete at this point.  Keep the complete, reproducible M5 tensor
    # dataset on host memory; an epoch transfers only its current physical microbatch to the
    # accelerator.  Moving every 30k-series input to MPS here used unified memory for both the
    # whole dataset and the active autograd graph, which made ``gradient_microbatch_size`` unable
    # to provide the memory bound its name promises.
    # A direct adapter caller owns its session and may use it after this call.  The governed
    # isolated worker owns its session exclusively, so only that boundary may release and later
    # reconstruct it around Torch training.
    output_spark = (
        _release_spark_for_torch_training(spark)
        if os.environ.get("AUTOFORECAST_ISOLATED_WORKER") == "1"
        else None
    )
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"  training on {dev} ({len(target):,} examples, {epochs} epochs max, "
        f"logical batch {batch_size}, gradient microbatch {gradient_microbatch_size})",
        flush=True,
    )
    model = model.to(dev)
    tensors = train_inputs
    if target_probs is not None:
        print(
            "  observation states are SOFT: the likelihood marginalises over "
            "status_prob_cols; the encoder conditions on the modal reading",
            flush=True,
        )

    # Split by SERIES, not by day. A day-level split puts the same series on both sides, and the
    # per-series residual embedding would memorise it -- validation would then measure recall.
    generator = torch.Generator().manual_seed(seed)
    train_rows, val_rows = split_by_series(example_series, val_fraction, generator)
    val_idx = val_rows
    from forecast.models.mft.heads import (
        informative_mask,
        soft_informative_weight,
    )

    if target_probs is None:
        informative_by_example = informative_mask(target, target_state).float()
    else:
        informative_by_example = soft_informative_weight(target, target_probs)
    informative_by_example = informative_by_example.reshape(len(target), -1).sum(dim=1).cpu()
    evidence_mask = informative_by_example[train_rows] > 0
    train_price_eligible_total = 0
    train_price_reference_total = 0
    if model.price_response is not None:
        local_known = train_inputs["known_future"][train_rows]
        reference_mask = (local_known[..., model.price_available_index] > 0.5) & (
            local_known[..., model.discount_available_index] > 0.5
        )
        train_price_reference_total = int(reference_mask.sum())
        if train_price_reference_total == 0:
            raise OursError(
                "monotone_raw_v1 training split has no cell with both planned price and a "
                "valid frozen historical reference; a structural price-response claim would "
                "be vacuous"
            )
    if price_penalty_weight and model.price_index is not None:
        price_mask = _price_intervention_mask(
            {"known_future": train_inputs["known_future"][train_rows]},
            model.price_index,
            model.price_available_index,
        )
        train_price_eligible_total = int(price_mask.sum())
        evidence_mask |= price_mask.reshape(len(train_rows), -1).any(dim=1)
        if train_price_eligible_total == 0:
            raise OursError(
                "price_penalty_weight is positive but the training split has no finite positive "
                "price cell; no relative monotonicity intervention could execute"
            )

    evidence_rows = train_rows[evidence_mask]
    training_identity_evidence = {
        "series": set(example_series[evidence_rows].tolist()),
        "product": (
            {
                row
                for row in train_inputs["product_index"][evidence_rows].tolist()
                if row < n_products_identity
            }
            if model.static.factorised
            else set()
        ),
        "store": (
            {row for row in train_inputs["store_index"][evidence_rows].tolist() if row < n_stores}
            if model.static.factorised or model.factor is not None
            else set()
        ),
    }
    training_oov_rows = {
        "series": n_series,
        "product": n_products_identity,
        "store": n_stores,
    }

    source_validation_loss = None
    if warm_start_enabled:
        current_identity_evidence = _canonical_identity_evidence(model, training_identity_evidence)
        warm_start_expected_run = dict(
            origin=origin,
            horizon=horizon,
            encoder_days=encoder_days,
            train_stride=train_stride,
            train_windows=train_windows,
            batch_size=batch_size,
            gradient_microbatch_size=gradient_microbatch_size,
            importance_draws=n_draws,
            events=[list(event) for event in events],
            event_max_support=event_max_support,
            event_windows=(
                None
                if event_windows is None
                else {name: list(window) for name, window in event_windows.items()}
            ),
            cycles=[list(cycle) for cycle in cycles],
            seed=seed,
            resolved_build=(None if resolved is None else resolved.canonical()),
            resolved_build_hash=(None if resolved is None else resolved.build_hash),
            training_controls=dict(
                demand_branch_dropout=demand_dropout,
                covariate_dropout=covariate_dropout,
                gradient_microbatch_size=gradient_microbatch_size,
                price_penalty_weight=price_penalty_weight,
                price_penalty_eligible_cells=train_price_eligible_total,
                price_response_mode=price_response_mode,
                price_occurrence_slope_max=price_occurrence_slope_max,
                price_size_slope_max=price_size_slope_max,
                price_response_valid_reference_cells=train_price_reference_total,
                static_variable_selection=static_selection,
            ),
            factor_group_col=grouping,
            factor_enabled=model.factor is not None,
            topology="bottom_up_factor" if model.factor is not None else "bottom_up",
            product_col=product,
            store_col=loading_col,
            category_col=category,
            known_names=list(infer.known.names),
            declared_known_drivers=sorted(infer.known.declared_drivers),
            static_real_stats=real_stats,
            vocabularies=dict(
                attribute_names=list(attribute_names),
                product_levels=list(product_levels),
                store_levels=list(store_levels),
                factor_levels=list(factor_levels),
            ),
            oov_rows=dict(series=n_series, product=n_products_identity, store=n_stores),
            oov_scheme=OOV_SCHEME,
            identity_evidence_rows=current_identity_evidence,
            plan_provenance=(
                None
                if infer.plan is None
                else {
                    "inference": infer.plan.as_record(),
                    "training": [
                        None if assembly.plan is None else assembly.plan.as_record()
                        for assembly in train
                    ],
                }
            ),
            panel=recorded_panel,
            panel_parquet_sha256=directory_digest(panel.path, suffixes=(".parquet",)),
            series_ids=list(series_ids),
            tensors=dict(
                attributes=attributes.cpu(),
                text=None if text is None else text.cpu(),
                group_index=group_index.cpu(),
                store_index=store_index.cpu(),
                product_index=product_index.cpu(),
            ),
        )
        source_model, source_record = load_checkpoint(warm_start_snapshot)
        _assert_private_snapshot_state(
            warm_start_snapshot,
            warm_start_snapshot_state,
            include_parent=True,
            label="warm-start checkpoint",
        )
        _validate_warm_start_namespace(source_record, model_config, warm_start_expected_run)
        model.load_state_dict(source_model.state_dict())
        _assert_warm_start_source_unchanged(warm_start_source_path, warm_start_checkpoint_sha256)
        # Source loading constructs a throwaway model and therefore consumes global RNG. The
        # continuation training stream is bound only to the recorded seed, not loader internals.
        torch.manual_seed(seed)
    _project_training_oov_parameters(model, training_identity_evidence, training_oov_rows)

    def validation_loss() -> float:
        """Score the same validation cells with the same factor draws at every checkpoint."""
        model.eval()
        fixed_generator = None
        if validation_seed is not None:
            fixed_generator = torch.Generator(device=dev).manual_seed(validation_seed)
        with torch.no_grad():
            val_informative = _informative_mass(
                target[val_idx],
                target_state[val_idx],
                None if target_probs is None else target_probs[val_idx],
            )
            return _microbatched_validation_loss(
                model,
                tensors,
                target,
                target_state,
                val_rows,
                target_probs=target_probs,
                normalization_count=val_informative,
                gradient_microbatch_size=gradient_microbatch_size,
                training_identity_evidence=training_identity_evidence,
                training_oov_rows=training_oov_rows,
                generator=fixed_generator,
            )

    if warm_start_enabled:
        source_validation_loss = _require_finite_validation_loss(validation_loss(), epoch=0)
        best = source_validation_loss
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_epoch = 0
        validation_trace = [
            {
                "epoch": 0,
                "validation_loss": float(source_validation_loss),
                "effective_min_delta": max(
                    early_stopping_absolute_min_delta,
                    early_stopping_relative_min_delta * abs(source_validation_loss),
                ),
                "meaningful_improvement": True,
                "best_validation_loss": float(source_validation_loss),
                "epochs_without_improvement": 0,
            }
        ]
        if quiet_epoch_logging:
            print(
                f"  warm start: authenticated source "
                f"{warm_start_checkpoint_sha256[:12]} loaded",
                flush=True,
            )
        else:
            print(
                f"  warm start: source {warm_start_checkpoint_sha256[:12]} is epoch 0, "
                f"validation {source_validation_loss:.5f}",
                flush=True,
            )
    else:
        best, best_state, best_epoch = float("inf"), None, None
        validation_trace = []

    groups, decayed = parameter_groups(model)
    print(f"  weight decay on the residual tables only ({len(decayed)}): {decayed}", flush=True)
    opt = torch.optim.AdamW(groups, lr=lr)
    scaler = torch.amp.GradScaler(dev.type) if precision == "16-mixed" else None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    training_log = _open_training_log(
        training_log_dir,
        training_metrics_path,
        epoch_observer,
    )
    waited, last_violation = 0, float("nan")
    epochs_completed = 0
    stopped_early = False
    optimizer_steps_total = 0
    training_examples_seen = 0
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        learning_rate = float(opt.param_groups[0]["lr"])
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        model.train()
        train_objective_total = 0.0
        gradient_norm_total = 0.0
        optimizer_steps = 0
        # Shuffle BLOCKS, not rows. The IWAE inner sum is defined over every cell sharing a
        # latent path; splitting one block across optimiser steps optimises a different bound
        # (and lets each fragment redraw what is supposed to be one common shock). `batch_size`
        # is therefore a soft cap: an unusually large block is emitted whole.
        batches = factor_block_batches(
            train_rows, train_inputs["group_index"], batch_size, generator
        )
        train_idx = train_rows
        # The soft likelihood's denominator is expected informative MASS, not the modal
        # hard-state count.  The sharded route and ``CustomModel.loss`` already use this
        # quantity; using the modal count here changed the objective and the relative price
        # penalty merely by turning sharding on.
        train_informative = _informative_mass(
            target[train_idx],
            target_state[train_idx],
            None if target_probs is None else target_probs[train_idx],
        )
        if train_informative == 0:
            raise OursError("training split has no informative cells")
        epoch_normalizer = train_informative / len(batches)
        for batch_rows in batches:
            last_violation, batch_objective, gradient_norm, _retries = (
                _complete_legacy_optimizer_batch(
                    model, opt,
                    functools.partial(
                        _backward_logical_batch, model, tensors, target, target_state,
                        batch_rows, target_probs=target_probs,
                        normalization_count=epoch_normalizer,
                        gradient_microbatch_size=gradient_microbatch_size,
                        covariate_dropout=covariate_dropout,
                        channel_columns=channel_columns,
                        price_penalty_weight=price_penalty_weight,
                        n_optimizer_batches=len(batches),
                        total_price_eligible=train_price_eligible_total,
                        scaler=scaler,
                    ),
                    scaler=scaler,
                    zero_oov_gradients=functools.partial(
                        _zero_training_oov_gradients, model, training_oov_rows
                    ),
                )
            )
            gradient_norm_total += gradient_norm
            train_objective_total += batch_objective
            optimizer_steps += 1
            _project_training_oov_parameters(model, training_identity_evidence, training_oov_rows)
        sched.step()

        val = _require_finite_validation_loss(validation_loss(), epoch=epoch + 1)
        epochs_completed = epoch + 1
        optimizer_steps_total += optimizer_steps
        training_examples_seen += len(train_rows)
        if training_log is not None:
            training_log.log_epoch(
                epoch + 1,
                train_objective=train_objective_total,
                validation_objective=val,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm_total / optimizer_steps,
                optimizer_steps=optimizer_steps_total,
                training_examples_seen=training_examples_seen,
                seconds=time.perf_counter() - epoch_started,
                process_peak_rss_bytes=_process_peak_rss_bytes(),
                cuda_peak_memory_bytes=(
                    int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
                ),
            )
        effective_min_delta = (
            max(
                early_stopping_absolute_min_delta,
                early_stopping_relative_min_delta * abs(best),
            )
            if math.isfinite(best)
            else early_stopping_absolute_min_delta
        )
        improved = _meaningful_validation_improvement(
            val, best, early_stopping_absolute_min_delta, early_stopping_relative_min_delta
        )
        if not quiet_epoch_logging:
            print(
                f"    epoch {epoch + 1:3}/{epochs}  val {val:.5f}"
                f"{'' if last_violation != last_violation else f'  price↑demand {last_violation:.1%}'}"
                f"{'  *' if improved else ''}",
                flush=True,
            )
        if improved:
            best, waited = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            waited += 1
        if warm_start_enabled:
            validation_trace.append(
                {
                    "epoch": epoch + 1,
                    "validation_loss": float(val),
                    "effective_min_delta": float(effective_min_delta),
                    "meaningful_improvement": improved,
                    "best_validation_loss": float(best),
                    "epochs_without_improvement": waited,
                }
            )
        if not improved and waited >= patience:
            stopped_early = True
            if not quiet_epoch_logging:
                print(
                    f"    stopped at epoch {epoch + 1}: " f"{patience} epochs without improvement",
                    flush=True,
                )
            break
    if training_log is not None:
        training_log.close()
    if quiet_epoch_logging:
        _print_quiet_training_completion(
            "training",
            epochs_completed,
            epochs,
            stopped_early,
        )
    if best_state is not None:
        # Restore the BEST weights, not the last. Without this, early stopping detects
        # overfitting and then forecasts with the overfitted model anyway.
        model.load_state_dict(best_state)
    warm_start_record = None
    if warm_start_enabled:
        _assert_warm_start_source_unchanged(warm_start_source_path, warm_start_checkpoint_sha256)
        warm_start_record = {
            "schema": WARM_START_SCHEMA,
            "source_checkpoint_sha256": warm_start_checkpoint_sha256,
            "source_checkpoint_schema": CHECKPOINT_SCHEMA,
            "source_is_epoch_zero": True,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "learning_rate": float(lr),
            "max_additional_epochs": epochs,
            "additional_epochs_completed": epochs_completed,
            "patience": patience,
            "validation_fraction": float(val_fraction),
            "validation_seed": validation_seed,
            "validation_evaluated_every_epoch": True,
            "common_validation_random_numbers": True,
            "absolute_min_delta": early_stopping_absolute_min_delta,
            "relative_min_delta": early_stopping_relative_min_delta,
            "source_validation_loss": float(source_validation_loss),
            "best_validation_loss": float(best),
            "best_epoch": best_epoch,
            "epochs_without_improvement": waited,
            "stopped_early": stopped_early,
            "stop_reason": ("validation_patience" if stopped_early else "max_additional_epochs"),
            "best_weights_restored": True,
            "quiet_epoch_logging": quiet_epoch_logging,
            "validation_trace": validation_trace,
        }
        _validate_warm_start_checkpoint_record({"warm_start": warm_start_record})
    identity_evidence_rows = _finalize_oov_parameters(model, training_identity_evidence)
    training_device = str(dev)
    artifact_device = torch.device("cpu")
    model = model.to(artifact_device)
    model.eval()
    print(
        f"  claim artifacts on {artifact_device} (training device {training_device})",
        flush=True,
    )

    # THE PRICE LADDER, verified rather than assumed. The hinge is a monitored preference; the
    # spec's own caveat says to verify monotonicity numerically on the ladder. 'report' records
    # the measurement, 'require' REFUSES a fitted model whose verified cells violate it -- the
    # guarantee the audit asked for, scoped to a declared deterministic sample and recorded as
    # such -- and 'skip' skips, visibly.
    ladder_report = None
    if price_ladder != "skip" and model.price_index is not None:
        keep = _sample_series_indices(series_ids, min(ladder_sample, n_series), seed)
        ladder_inputs = {
            k: (v.to(artifact_device) if isinstance(v, torch.Tensor) else v)
            for k, v in _inputs(infer, keep).items()
        }
        ladder_report = verify_price_ladder(
            model,
            ladder_inputs,
            bumps=ladder_bumps,
            tolerance=ladder_tolerance,
            point_factor_method=point_factor_method,
            point_factor_draws=point_factor_draws,
            seed=seed,
            rng_domain_scheme=RNG_DOMAIN_SCHEME,
        )
        print(
            f"  price ladder ({ladder_report.n_cell_steps:,} cell-steps): "
            f"violations {ladder_report.violation_share:.2%}, "
            f"worst step +{ladder_report.worst_increase:.2%}"
            f"{'  PASSED' if ladder_report.passed else ''}",
            flush=True,
        )
        if price_ladder == "require" and not ladder_report.passed:
            remedy = (
                "The structural route or its audit wiring is broken; do not train further."
                if model.price_response is not None
                else "Retrain only under a prospectively registered soft-penalty change, or ship "
                "under 'report' with the violation rate on the record."
            )
            raise OursError(
                f"price_ladder='require': {ladder_report.violation_share:.2%} of verified "
                f"cells show demand RISING with price (worst step "
                f"+{ladder_report.worst_increase:.2%}). The model is refused, as the guarantee "
                f"demands -- a monotone claim over cells nobody verified is not one. {remedy}"
            )

    if warm_start_enabled:
        _assert_warm_start_source_unchanged(warm_start_source_path, warm_start_checkpoint_sha256)
    if checkpoint_out is not None:
        saved = save_checkpoint(
            model,
            checkpoint_out,
            model_config=model_config,
            run=dict(
                origin=origin,
                horizon=horizon,
                encoder_days=encoder_days,
                aggregate_trajectory_projection=(_dormant_aggregate_projection_record(model)),
                training_device=training_device,
                artifact_device=str(artifact_device),
                train_stride=train_stride,
                train_windows=train_windows,
                epochs=epochs,
                batch_size=batch_size,
                gradient_microbatch_size=gradient_microbatch_size,
                **({"precision": precision} if precision != "32-true" else {}),
                patience=patience,
                importance_draws=n_draws,
                events=[list(e) for e in events],
                event_max_support=event_max_support,
                event_windows=(
                    None
                    if event_windows is None
                    else {k: list(v) for k, v in event_windows.items()}
                ),
                cycles=[list(c) for c in cycles],
                seed=seed,
                # The RESOLVED BUILD identity: which architecture this run believed it was
                # constructing, as a recorded fact. None means the caller passed explicit
                # runtime parameters instead of a resolved spec, and the record says so
                # rather than inventing a hash for a resolution that never happened.
                resolved_build=(None if resolved is None else resolved.canonical()),
                resolved_build_hash=(None if resolved is None else resolved.build_hash),
                resolved_notes=(() if resolved is None else resolved.notes),
                training_controls=dict(
                    demand_branch_dropout=demand_dropout,
                    covariate_dropout=covariate_dropout,
                    gradient_microbatch_size=gradient_microbatch_size,
                    price_penalty_weight=price_penalty_weight,
                    price_penalty_eligible_cells=train_price_eligible_total,
                    price_response_mode=price_response_mode,
                    price_occurrence_slope_max=price_occurrence_slope_max,
                    price_size_slope_max=price_size_slope_max,
                    price_response_valid_reference_cells=train_price_reference_total,
                    static_variable_selection=static_selection,
                ),
                # BOTH factor counts, separately. A run record that says only "1,000"
                # cannot tell a reader whether that was the point integration or the
                # systemic scenario count -- different claims about different numbers.
                point_factor_method=point_factor_method,
                point_factor_draws=(
                    None
                    if point_factor_method == "gauss_hermite"
                    else (
                        point_factor_draws if point_factor_draws is not None else POINT_FACTOR_DRAWS
                    )
                ),
                path_factor_draws=(
                    inference_factor_draws
                    if inference_factor_draws is not None
                    else default_factor_draws(n_paths)
                ),
                n_paths=n_paths,
                paths_materialized=paths_out_dir is not None,
                rng_domain_scheme=RNG_DOMAIN_SCHEME,
                factor_group_col=grouping,
                factor_enabled=model.factor is not None,
                topology="bottom_up_factor" if model.factor is not None else "bottom_up",
                product_col=product,
                store_col=loading_col,
                category_col=category,
                known_names=list(infer.known.names),
                declared_known_drivers=sorted(infer.known.declared_drivers),
                static_real_stats=real_stats,
                # THE STABLE IDENTITY NAMESPACE: which level got which embedding row, and
                # therefore what a NEW panel's identities map onto -- unseen levels go to
                # the reserved OOV row. Without this stored, a checkpoint's embeddings are
                # rows with no names, and no second client can ever be encoded.
                vocabularies=dict(
                    attribute_names=list(attribute_names),
                    product_levels=list(product_levels),
                    store_levels=list(store_levels),
                    factor_levels=list(factor_levels),
                ),
                oov_rows=dict(series=n_series, product=n_products_identity, store=n_stores),
                oov_scheme=OOV_SCHEME,
                identity_evidence_rows=identity_evidence_rows,
                # PLAN PROVENANCE: which as-of selections this run's horizon commercial
                # values came from, content-hashed -- one for inference, one per training
                # window. None means the panel's own rows were used, which the Panel
                # docstring records as backtest convenience, not proof.
                price_ladder=(None if ladder_report is None else ladder_report.as_record()),
                plan_provenance=(
                    None
                    if infer.plan is None
                    else {
                        "inference": infer.plan.as_record(),
                        "training": [None if a.plan is None else a.plan.as_record() for a in train],
                    }
                ),
                # The checkpoint must carry the data-role contract that produced its
                # conditioning tensors. Scenario inference may point at a refreshed panel,
                # but silently changing (for example) the OOS or exposure role rebuilds a
                # different encoder under the same weights.
                panel=recorded_panel,
                panel_parquet_sha256=directory_digest(panel.path, suffixes=(".parquet",)),
                paths_sample=paths_sample,
                paths_sample_rule="smallest sha256(seed\\0series_id), without replacement",
                series_ids=list(series_ids),
                tensors=dict(
                    attributes=attributes.cpu(),
                    text=None if text is None else text.cpu(),
                    group_index=group_index.cpu(),
                    store_index=store_index.cpu(),
                    product_index=product_index.cpu(),
                ),
                **({} if warm_start_record is None else {"warm_start": warm_start_record}),
            ),
        )
        print(f"  checkpoint: {saved}", flush=True)

    # FORECAST FROM THE INFERENCE ASSEMBLY, whose encoder window ends at `origin` and whose
    # known-future covers the days after it. This is the step the training arrangement above
    # exists to make meaningful: the model is now being asked for the same mapping it was taught.
    predict = {
        k: (v.to(artifact_device) if isinstance(v, torch.Tensor) else v)
        for k, v in _inputs(infer, list(range(n_series))).items()
    }
    # The point forecast integrates the COUNT distribution analytically (`hurdle_mean`) and the
    # FACTOR numerically. Drawing 1,000 paths per series and averaging them away would sample
    # both, which is slower and noisier. Note what this is NOT: with `point_factor_method='mc'`
    # the factor half is still a finite-sample estimate that moves with the seed. Only
    # `'gauss_hermite'` makes the whole thing deterministic.
    yhat = point_forecast(
        model, predict, method=point_factor_method, factor_draws=point_factor_draws, seed=seed
    )  # [series, horizon]

    # Spark was deliberately absent for the complete Torch epoch loop in an isolated governed
    # run.  Direct callers retain their session, whose lifecycle remains their responsibility.
    writer_spark = (
        spark if output_spark is None else _resume_spark_after_torch_training(output_spark)
    )
    try:
        if paths_out_dir is not None:
            # Paths are drawn ONLY for the series that get persisted. They exist for the spread --
            # cumulative lead-time coverage -- and that is computed over the persisted sample, so
            # drawing the other 28,000 series' paths is work whose result is discarded. A lexical
            # prefix clusters product/category/store identifiers and can make calibration evidence
            # systematically unrepresentative. Stable hash sampling is uniform without replacement,
            # reproducible from the recorded seed, and independent of catalogue naming conventions.
            keep = _sample_series_indices(series_ids, paths_sample, seed)
            index = torch.tensor(keep, dtype=torch.long)
            subset = _subset(predict, index.to(artifact_device))
            paths = draw_paths_over_factor(
                model, subset, n_paths, factor_draws=inference_factor_draws, seed=seed
            )
            _write_paths(
                paths,
                [series_ids[i] for i in keep],
                origin,
                horizon,
                len(keep),
                writer_spark,
                paths_out_dir,
            )

        if diagnostics_out is not None:
            import json as _json

            path = pathlib.Path(diagnostics_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(model_diagnostics(model))
            if ladder_report is not None:
                record["price_ladder"] = ladder_report.as_record()
            if events:
                record["training_event_episodes"] = _event_episodes(
                    list(events), target_origins, horizon, event_max_support
                )
            path.write_text(_json.dumps(record, indent=2))
            # The ARCHITECTURE goes beside the diagnostics: what was built, not only what it learned.
            # Read off the constructed modules, so it cannot disagree with the code that ran.
            (path.parent / "architecture.json").write_text(_json.dumps(model.describe(), indent=2))
            print(
                f"  architecture: {model.describe()['n_branches_built']} branches, "
                f"{model.describe()['total_parameters']:,} parameters",
                flush=True,
            )
            print(f"  diagnostics: {model_diagnostics(model)}", flush=True)

        d0 = dt.date.fromisoformat(origin)
        records = [
            (series_ids[i], d0 + dt.timedelta(days=step + 1), float(yhat[i, step]))
            for i in range(n_series)
            for step in range(horizon)
        ]
        writer_spark.createDataFrame(
            records, "series_id string, date date, yhat double"
        ).write.mode("overwrite").parquet(str(out_dir))
    finally:
        if output_spark is not None:
            _stop_reopened_spark(writer_spark)

    disposition = {EMITTED: len(records)}
    declined = (n_total - n_series) * horizon
    if declined:
        disposition[DECLINED_NO_WINDOW] = declined
    if warm_start_enabled:
        _assert_warm_start_source_unchanged(warm_start_source_path, warm_start_checkpoint_sha256)
    return Forecast(
        adapter=NAME,
        path=str(out_dir),
        origin=origin,
        horizon=horizon,
        n_series=n_total,
        disposition=disposition,
        target_estimand=panel.target_estimand,
    ).validate()


# ------------------------------------------------------------------ feature assembly
#
# These CALL the Spark builders in `forecast/features/`. An earlier version reimplemented two of
# them inline, and because the reimplementations produced tensors of the right rank nothing
# failed -- they were simply thinner:
#
#   · `discount_depth` was never computed, so the commercial branch saw a raw price with no level
#     to compare it to. Prices differ by three orders of magnitude across a catalogue, so without
#     a reference the branch has to learn each product's normal price before it can tell a
#     discount from a cheap item -- and that is the elasticity signal, not a refinement of it.
#   · event distance was measured from the ORIGIN and held constant across the horizon, so a
#     28-day horizon spanning Christmas told the decoder "Christmas is 12 days away" on all 28
#     days. The peak lands on whichever day the optimiser finds least costly, which is not a day.
#   · the identity branch was fed two DEMAND statistics rather than identity, so it had no
#     cold-start path at all -- and every launch goes through that path.

CALENDAR_COLUMNS = ("day_of_week", "day_of_month", "is_weekend")
"""Known-future columns needing no client data: derivable for any date, years ahead."""

MAX_ONE_HOT = 64
"""Largest vocabulary a static attribute may one-hot into.

An attribute with thousands of levels is an IDENTIFIER, not an attribute, and one-hot encoding it
builds a lookup table with no cold-start path -- which is the thing `StaticIdentityBranch`'s
weight-decayed per-series residual already carries, and carries better because it is shrunk.
"""


@dataclass(frozen=True)
class KnownFuture:
    """The known-future tensor together with WHAT EACH COLUMN IS.

    `scenario.contrast.DriverBundle` addresses columns by INDEX. A layout that lives only in the
    order the columns happened to be concatenated makes every attribution silently wrong the
    moment a client without a promotion column shifts the event columns left by one -- the
    contrast still runs, still returns a number, and intervenes on the wrong driver.

    Carrying the names with the values lets a bundle be resolved by NAME.  Names alone are not a
    declaration, however: the commercial tensor has a fixed width and therefore names price,
    discount and promotion placeholders even when a client supplies only one of them.
    ``declared_drivers`` is the separate role-liveness contract that makes such placeholders
    unavailable to scenario APIs and model-level audits.
    """

    values: torch.Tensor
    names: tuple[str, ...]
    declared_drivers: frozenset[str]

    def __post_init__(self) -> None:
        """Raises: OursError: the layout or its declared-driver metadata is inconsistent."""
        if self.values.shape[-1] != len(self.names):
            raise OursError(
                f"known-future tensor is {self.values.shape[-1]} wide but {len(self.names)} "
                f"columns were named: {self.names}"
            )
        if not isinstance(self.declared_drivers, frozenset):
            raise OursError(
                "known-future declared_drivers must be an immutable frozenset, got "
                f"{type(self.declared_drivers).__name__}"
            )
        unknown = sorted(self.declared_drivers.difference(self.names))
        if unknown:
            raise OursError(
                f"known-future declares drivers absent from its tensor layout: {unknown}"
            )

    def index(self, name: str) -> int:
        """Raw tensor-column index of `name`, including a fixed commercial placeholder.

        Raises:
            OursError: the named tensor column does not exist. Scenario code must use
                :meth:`driver_index`, which additionally checks the client's role declaration.
        """
        if name not in self.names:
            raise OursError(
                f"{name!r} is not a known-future tensor column; available: {self.names}"
            )
        return self.names.index(name)

    def driver_index(self, name: str) -> int:
        """Column index of a role the client actually declared, never of a fixed placeholder."""
        if name not in self.declared_drivers:
            raise OursError(
                f"{name!r} is not a declared known-future driver for this client; declared: "
                f"{tuple(sorted(self.declared_drivers))}. A fixed-width placeholder cannot be "
                "intervened on or used to certify a constraint"
            )
        return self.index(name)


def _pack_by_step(
    path,
    spark: SparkSession,
    columns: tuple[str, ...],
    series_ids: list[str],
    origin: str,
    horizon: int,
) -> torch.Tensor:
    """Read a per-`(series_id, date)` feature table into `[n_series, horizon, len(columns)]`.

    Step 1 is the first day AFTER the origin, matching what the builders emit. A series or day the
    table does not cover stays zero, and every caller pairs the values with an availability mask
    so a zero is never read as a measurement.
    """
    step = F.datediff(F.col("date"), F.to_date(F.lit(origin)))
    frame = (
        spark.read.parquet(str(path))
        .withColumn("_step", step)
        .filter((F.col("_step") >= 1) & (F.col("_step") <= horizon))
    )
    # One row per series holding its days as an ordered array -- the same shape `demand_history`
    # collects, and the reason is the same: 30,490 x 28 individual rows through the driver is a
    # collect the driver does not need to do.
    packed = frame.groupBy("series_id").agg(
        F.sort_array(
            F.collect_list(
                F.struct(F.col("_step"), *[F.col(c).cast("double").alias(c) for c in columns])
            )
        ).alias("cells")
    )

    position = {sid: i for i, sid in enumerate(series_ids)}
    out = torch.zeros(len(series_ids), horizon, len(columns))
    for row in packed.collect():
        i = position.get(row["series_id"])
        if i is None:
            continue  # a series with no encoder window; not forecast at all
        for cell in row["cells"]:
            s = cell["_step"] - 1
            if 0 <= s < horizon:
                for j, column in enumerate(columns):
                    value = cell[column]
                    if value is not None:
                        out[i, s, j] = float(value)
    return out


def _commercial_tensors(
    panel: Panel,
    spark: SparkSession,
    origin: str,
    horizon: int,
    series_ids: list[str],
    work_dir: pathlib.Path,
    series_predicate: str | None = None,
):
    """Price, discount depth and promotion over the horizon, via `features/commercial.build`.

    Returns `(at_origin, mask_at_origin, over_horizon, available_over_horizon, n_channels)`, or
    empties when the panel has neither a price nor a promotion column -- in which case the
    branch is not built at all. An all-masked row would make attention over an empty key set
    undefined.

    `available_over_horizon` exists because the DECODER needs the flags too, not only the
    branch: the horizon values are zero-filled where missing, and 19% of M5 price cells are
    missing, so without the flag the decoder reads "the price is $0.00" -- the exact "this
    product is free" confusion this module's own docstring warns about. Events already carried
    a `_present` column into the known-future tensor for the same reason; price did not.

    `at_origin` is the FIRST HORIZON DAY, which is what the branch's "values at the forecast step"
    means for a direct multi-horizon model. The rest of the horizon reaches the model through the
    decoder's known-future tensor, not through the encoder.  Its mask remains the TRUE
    per-channel availability even when a whole row is absent; ``CustomModel.encode_base`` owns
    the attention-safe all-missing token, so no fake available zero price leaks into scenario
    routing.
    """
    # Resolve semantic roles from the panel contract; source column names are configurable.
    price_col, promo_col = panel.price_col, panel.promo_col
    if price_col is None and promo_col is None and panel.plan is None:
        return None, None, None, None, 0, None
    columns = set(spark.read.parquet(panel.path).columns)
    missing = [c for c in (price_col, promo_col) if c is not None and c not in columns]
    if missing:
        raise OursError(
            f"panel declares commercial column(s) {missing} that the Parquet does not have. "
            "Continuing without them would drop the commercial branch on a client that has it"
        )
    not_known = [c for c in (price_col, promo_col) if c is not None and c not in panel.known_future]
    if not_known and panel.plan is None:
        # With a PlanSource the panel's future rows are never read -- the as-of selection is
        # the horizon source and carries its own provenance -- so this declaration guard
        # applies only to the fallback that does read them.
        raise OursError(
            f"commercial column(s) {not_known} are read after origin={origin} but are not "
            "declared in panel.known_future. Future realised prices/promotions are target-time "
            "information and would leak into the forecast; declare a genuinely planned "
            "schedule as known_future, declare a PlanSource with as-of provenance, or unset "
            "the corresponding price_col/promo_col role"
        )

    plan_selection = None
    plan_path = None
    if panel.plan is not None:
        from forecast.features import plans

        value_cols = {}
        if panel.plan.price_col:
            value_cols["price"] = panel.plan.price_col
        if panel.plan.promo_col:
            value_cols["promo"] = panel.plan.promo_col
        plan_selection = plans.build(
            panel.plan.path,
            origin,
            horizon,
            spark,
            str(work_dir / "_plan"),
            series_col=panel.series_col,
            valid_date_col=panel.plan.valid_date_col,
            known_at_col=panel.plan.known_at_col,
            snapshot_col=panel.plan.snapshot_col,
            value_cols=value_cols,
            series_ids=series_ids,
        )
        plan_path = plan_selection.path
    built = commercial.build(
        panel.path,
        origin,
        horizon,
        spark,
        str(work_dir / "_commercial"),
        series_col=panel.series_col,
        time_col=panel.time_col,
        price_col=price_col,
        promo_col=promo_col,
        plan_path=plan_path,
        series_predicate=series_predicate,
    )
    values = _pack_by_step(built, spark, commercial.CHANNELS, series_ids, origin, horizon)
    available = _pack_by_step(
        built,
        spark,
        tuple(f"{c}_available" for c in commercial.CHANNELS),
        series_ids,
        origin,
        horizon,
    )

    at_origin = values[:, 0, :].clone()
    mask = available[:, 0, :] > 0.5
    return at_origin, mask, values, available, len(commercial.CHANNELS), plan_selection


def _declared_commercial_drivers(panel: Panel) -> frozenset[str]:
    """Commercial roles supplied by the horizon source, separate from fixed placeholders.

    A ``PlanSource`` replaces the panel's recorded future rows, so its declarations decide which
    horizon drivers are live.  Without a plan, the panel declarations decide.  Discount depth is
    live only when planned price is live *and* the panel has historical price from which the
    reference level can be derived.
    """
    source = panel.plan if panel.plan is not None else panel
    declared: set[str] = set()
    if source.price_col is not None:
        declared.add("price")
        if panel.price_col is not None:
            declared.add("discount_depth")
    if source.promo_col is not None:
        declared.add("promo")
    return frozenset(declared)


def _event_tensors(
    events,
    origin: str,
    horizon: int,
    series_ids: list[str],
    spark: SparkSession,
    work_dir: pathlib.Path,
    max_support: int,
    event_windows: dict[str, tuple[int, int]] | None = None,
):
    """Signed distance to each announced event, via `features/calendar_events.build`.

    Returns `(distance_over_horizon, present_over_horizon, named_known_future_pair, n_events)`, or
    four empties when the client supplied no calendar. Absent rather than zero-filled: zero
    distance means "the event is today". The full horizon reaches the kernel; reducing the first
    two outputs to day one made the branch's distributed lag a one-day origin summary while the
    raw distances bypassed it through the generic decoder.

    The builder drops an event whose `known_at` falls after the origin, which is the leak this
    feature exists to prevent -- a model trained on unannounced promotions learns to foresee them,
    scores beautifully in backtest, and live cannot do it.
    """
    if not events:
        return None, None, None, 0
    # One COLUMN per event name, not per occurrence: "Easter" across four years is one event
    # whose kernel pools those occurrences, which is the partial-pooling design.
    names = tuple(dict.fromkeys(name for name, _, _ in events))
    built = calendar_events.build(
        list(events),
        origin,
        horizon,
        series_ids,
        spark,
        str(work_dir / "_events"),
        max_support=max_support,
        event_windows=event_windows,
    )
    # The calendar table is one row per series/date.  Collect it once: separately collecting
    # distance and presence duplicated the full M5 driver-side materialisation while preserving
    # no additional information.
    packed = _pack_by_step(
        built,
        spark,
        tuple(f"{n}_{kind}" for kind in ("distance", "present") for n in names),
        series_ids,
        origin,
        horizon,
    )
    distance, present = packed[..., : len(names)], packed[..., len(names) :]
    # BOTH distance and presence reach the decoder. The builder clips an out-of-support event to
    # distance 0, and 0 is also "the event is today" -- so distance alone is ambiguous exactly
    # where it matters most. The encoder's branch has `present` to disambiguate; without carrying
    # it into the known-future tensor the decoder would read every quiet day as a holiday.
    return distance, present > 0.5, (distance, present), len(names)


def _identity_tensors(
    panel: Panel, spark: SparkSession, series_ids: list[str], work_dir: pathlib.Path
):
    """One-hot static attributes, via `features/static_identity.build`.

    ONE-HOT RATHER THAN AN INDEX. `StaticIdentityBranch` projects the attribute vector through a
    `Linear`, and feeding a category index into a linear layer imposes an ORDER on a nominal
    variable: department 3 is not between department 2 and department 4, but the layer is forced
    to treat it that way and the constraint is invisible in the loss.

    Returns `(attributes, names)`. With no declared `panel.static`, one constant zero column says
    only "static attributes unavailable". It must not be replaced by demand summaries: the
    identity tensor is shared by every rolling training origin, so a summary computed at the
    final forecast origin would reveal later target history to earlier windows. The residual is
    still available for established series; the capability report must describe the resulting
    lack of an attribute-based cold-start path.

    Raises:
        OursError: an attribute has more levels than `MAX_ONE_HOT`.
    """
    if not panel.static:
        return torch.zeros(len(series_ids), 1), ("_no_static",)

    built = static_identity.build(
        panel.path,
        spark,
        str(work_dir / "_identity"),
        series_col=panel.series_col,
        attribute_cols=panel.static,
    )
    vocabularies = {c: static_identity.vocabulary(built, c, spark) for c in panel.static}
    for column, vocabulary in vocabularies.items():
        if len(vocabulary) > MAX_ONE_HOT:
            raise OursError(
                f"static attribute {column!r} has {len(vocabulary)} levels, above MAX_ONE_HOT="
                f"{MAX_ONE_HOT}. An attribute with this many levels is an identifier, and "
                "one-hot encoding it builds a lookup table with no cold-start path -- which the "
                "branch's weight-decayed per-series residual already carries, and carries better"
            )

    rows = {r["series_id"]: r for r in spark.read.parquet(str(built)).collect()}
    names: list[str] = []
    for column in panel.static:
        names.extend(f"{column}={level}" for level in vocabularies[column])
    out = torch.zeros(len(series_ids), len(names))
    offset = 0
    for column in panel.static:
        at = {level: j for j, level in enumerate(vocabularies[column])}
        for i, sid in enumerate(series_ids):
            row = rows.get(sid)
            if row is not None and row[column] is not None:
                j = at.get(str(row[column]))
                if j is not None:
                    out[i, offset + j] = 1.0
        offset += len(vocabularies[column])
    return out, tuple(names)


def _shrunk_baselines(
    history: torch.Tensor,
    occurrence_observed: torch.Tensor,
    category: torch.Tensor,
    exact_size_observed: torch.Tensor | None = None,
    prior_override: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beta-shrunk lagged frequency and category-shrunk lagged positive size, per series.

    Returns `(baseline_logit, baseline_log_size, freq)`, where `freq` is the same posterior
    occurrence probability represented by `baseline_logit` (useful as a finite cold-start
    identity summary even when the series has no observed day).

    THE PRIORS ARE PER CATEGORY, falling back to the panel-wide rate for a series with no
    declared level (`category = -1`) or an empty level. The spec is specific about this --
    `baseline_smoothing` shrinks toward "a category rate", `size_smoothing` toward "the
    CATEGORY's mean positive size, which for a new product is the only evidence there is" --
    and the earlier version used the global panel means for both while a variable named
    `category_mean` said otherwise. On a mixed panel that pulls every cold start toward the
    fast-moving category's numbers: a launch in a slow department inherited the panel's
    grocery-dominated rate, which is not evidence about it at all.

    Args:
        history: `[n, days]` demand over the encoder window.
        occurrence_observed: `[n, days]` True where occurrence is known. An exact zero
            qualifies, as does a positive lower bound (it proves a sale occurred).
        category: `[n]` dense level ids, -1 for "no level".
        exact_size_observed: `[n, days]` True where a positive magnitude, if present, is exact.
            Lower bounds must be False: they inform occurrence but are not samples of positive
            demand size. None preserves the historical API by using `occurrence_observed`.
    """
    if exact_size_observed is None:
        exact_size_observed = occurrence_observed
    if history.shape != occurrence_observed.shape or history.shape != exact_size_observed.shape:
        raise OursError(
            "history, occurrence_observed and exact_size_observed must have the same shape; "
            f"got {tuple(history.shape)}, {tuple(occurrence_observed.shape)}, and "
            f"{tuple(exact_size_observed.shape)}"
        )
    occurrence_observed = _binary_mask(
        occurrence_observed, history, "occurrence observation mask", error_type=OursError
    )
    exact_size_observed = _binary_mask(
        exact_size_observed, history, "exact-size observation mask", error_type=OursError
    )

    # Accumulate the empirical-Bayes counts in float64. In float32, `(N + .5)/(N + 1)` rounds
    # to exactly one once N is large enough, recreating an infinite logit despite the proper
    # Beta posterior being strictly interior.
    work_history = history.double()
    occurrence_f = occurrence_observed.to(torch.float64)
    n_obs = occurrence_f.sum(dim=1)  # DO NOT clamp: zero observations is not one observed zero.
    selling = ((history > 0).to(torch.float64) * occurrence_f).sum(dim=1)
    exact_positive = (history > 0) & exact_size_observed
    positive_f = exact_positive.to(torch.float64)
    positive_total = (work_history * positive_f).sum(dim=1)
    positive_count = positive_f.sum(dim=1)

    # A weak proper hyperprior keeps the empirical Bayes anchors strictly inside their domains.
    # This replaces the epsilon cliff: an all-zero panel gets a small finite Beta posterior rate,
    # and a panel with no exact positive size gets the count model's minimum positive size (one)
    # through a one-observation Gamma-style prior rather than log(1e-3).
    total_days = n_obs.sum()
    global_rate = (selling.sum() + 0.5) / (total_days + 1.0)  # Jeffreys Beta(1/2, 1/2)
    global_size = (positive_total.sum() + 1.0) / (positive_count.sum() + 1.0)
    prior_rate = torch.full_like(n_obs, float(global_rate))
    prior_size = torch.full_like(n_obs, float(global_size))
    if prior_override is not None:
        # SHARDED assembly: per-series priors measured GLOBALLY in Spark. In-tensor level
        # statistics inside one shard would hand a cold start its shard's accident instead of
        # its category's rate; in the single-shard case the in-tensor statistics equal the
        # Spark aggregates, so the two paths cannot drift.
        prior_rate = prior_override[0].to(torch.float64)
        prior_size = prior_override[1].to(torch.float64)
    grouped = (
        (category >= 0) if prior_override is None else torch.zeros_like(category, dtype=torch.bool)
    )
    if bool(grouped.any()):
        n_levels = int(category[grouped].max()) + 1
        idx = category[grouped]
        level_sell = work_history.new_zeros(n_levels).index_add_(0, idx, selling[grouped])
        level_days = work_history.new_zeros(n_levels).index_add_(0, idx, n_obs[grouped])
        level_pos_total = work_history.new_zeros(n_levels).index_add_(
            0, idx, positive_total[grouped]
        )
        level_pos_count = work_history.new_zeros(n_levels).index_add_(
            0, idx, positive_count[grouped]
        )
        # CATEGORY priors are themselves partially pooled toward the proper panel prior. Raw
        # category ratios can be exactly 0/1 (or 0/0), merely moving the log cliff up one level.
        level_rate = (level_sell + global_rate * 4.0) / (level_days + 4.0)
        level_size = (level_pos_total + global_size * 2.0) / (level_pos_count + 2.0)
        prior_rate[grouped] = level_rate[idx]
        prior_size[grouped] = level_size[idx]

    shrunk = (selling + prior_rate * 4.0) / (n_obs + 4.0)  # Beta(4p, 4(1-p))
    baseline_logit = torch.log(shrunk) - torch.log1p(-shrunk)
    size = (positive_total + prior_size * 2.0) / (positive_count + 2.0)
    return (
        baseline_logit.to(history.dtype),
        torch.log(size).to(history.dtype),
        shrunk.to(history.dtype),
    )


def _history_observation_masks(
    history: torch.Tensor,
    state: torch.Tensor,
    explicit_oos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate encoder magnitude, OOS state, and exact-size evidence.

    Returns `(magnitude_observed, oos, exact_size_observed)`. These cannot be one mask:
    `lower_bound, y>0` informs occurrence and a magnitude floor but not exact positive size;
    `lower_bound, y=0` informs only OOS state; `exact, y=0` informs occurrence as a real zero.
    """
    if history.shape != state.shape or (
        explicit_oos is not None and explicit_oos.shape != history.shape
    ):
        raise OursError(
            "history, observation state, and explicit OOS (when supplied) must align; got "
            f"{tuple(history.shape)}, {tuple(state.shape)}, and "
            f"{None if explicit_oos is None else tuple(explicit_oos.shape)}"
        )
    lower_bound = state == 1
    # A zero-sale stockout is state=unobserved because demand>=0 contributes no likelihood, but
    # it remains shelf-state evidence. Only the explicit feature preserves that fact; without it
    # we retain positive lower-bound OOS and report the capability degradation.
    oos = lower_bound if explicit_oos is None else explicit_oos.bool()
    magnitude_observed = (state == 0) | (lower_bound & (history > 0))
    exact_size_observed = state == 0
    return magnitude_observed, oos, exact_size_observed


def _probs_for_estimand(probs: torch.Tensor, target_estimand: str) -> torch.Tensor:
    """The soft analogue of `_states_for_estimand`: route state PROBABILITY MASS by estimand.

    Under `sales_given_availability` a stockout day's recorded sales ARE the sales, so the
    lower-bound mass folds into exact; under `unconstrained_demand` it stays censored. Same
    physics, same routing rule, probability-weighted.
    """
    if target_estimand == "unconstrained_demand":
        return probs
    if target_estimand == "sales_given_availability":
        folded = probs.clone()
        folded[..., 0] = probs[..., 0] + probs[..., 1]
        folded[..., 1] = 0.0
        return folded
    raise OursError(f"unsupported target_estimand {target_estimand!r}")


def _states_for_estimand(state: torch.Tensor, target_estimand: str) -> torch.Tensor:
    """Route a physical stockout state into the likelihood for the requested quantity.

    A positive stockout sale is a lower bound on unconstrained demand, but it is an exact record
    of sales given availability. The OOS feature remains separate, so changing the likelihood
    meaning does not erase shelf-state information from the encoder.
    """
    if target_estimand == "unconstrained_demand":
        return state
    if target_estimand == "sales_given_availability":
        return torch.where(state == 1, torch.zeros_like(state), state)
    raise OursError(f"unsupported target_estimand {target_estimand!r}")


def _informative_mass(
    target: torch.Tensor,
    state: torch.Tensor,
    state_probs: torch.Tensor | None,
) -> float:
    """The exact hard/soft information denominator used by ``CustomModel.loss``."""
    from forecast.models.mft.heads import (
        informative_mask,
        soft_informative_weight,
    )

    if state_probs is None:
        return float(informative_mask(target, state).sum())
    return float(soft_informative_weight(target, state_probs).sum())


PRICE_LADDER_RELATIVE_SCALE_FLOOR = 1e-9
"""Denominator floor for relative price-response increases at near-zero expected demand."""


@dataclass(frozen=True)
class LadderReport:
    """The price-ladder verification: what GUARANTEED monotonicity actually means here.

    The training penalty is a monitored preference -- a finite hinge over sampled
    perturbations, and a model can carry a small penalty while violating monotonicity on a
    third of its cells. The spec's own caveat says to "verify monotonicity numerically by
    evaluating the price ladder; do not assume the parameterisation delivered it". This is
    that evaluation: the same declared factor-integration route emitted for scoring, at every
    supplied audit cell-step and relative price level. A violation is any adjacent price increase
    that RAISES that expected cell-step demand beyond tolerance. Horizon totals are not used: a rise
    on one day must not be cancelled by a larger fall on another.

    Attributes:
        bumps: The declared ladder of relative price multipliers, ascending, containing 0.0.
        violation_share: Share of verified series/day cell-steps with an increasing ladder step.
        worst_increase: Largest relative demand increase across any upward price step.
        n_series: Series sampled for verification.
        n_eligible_series: Sampled series with at least one available planned price.
        n_cell_steps: Available-price series/day forecast cells verified. The scope is declared
            and recorded, because "guaranteed" over unavailable placeholders is vacuous.
        tolerance: Relative increase treated as numerical noise rather than violation.
        relative_scale_floor: Denominator floor used when lower-bump expected demand is near zero.
        point_factor_method: The same integration route used for the emitted point forecast.
        point_factor_draws: MC factor count, or None for deterministic quadrature.
        seed: Common-random-number seed used by MC; recorded even though GH is seedless.
        rng_domain_scheme: Versioned MC factor-stream mapping.
    """

    bumps: tuple[float, ...]
    violation_share: float
    worst_increase: float
    n_series: int
    n_eligible_series: int
    n_cell_steps: int
    tolerance: float
    relative_scale_floor: float
    point_factor_method: str
    point_factor_draws: int | None
    seed: int
    rng_domain_scheme: str

    @property
    def passed(self) -> bool:
        return self.violation_share == 0.0

    def as_record(self) -> dict[str, object]:
        return {
            "bumps": list(self.bumps),
            "violation_share": self.violation_share,
            "worst_increase": self.worst_increase,
            "n_series": self.n_series,
            "n_eligible_series": self.n_eligible_series,
            "n_cell_steps": self.n_cell_steps,
            "tolerance": self.tolerance,
            "relative_scale_floor": self.relative_scale_floor,
            "point_factor_method": self.point_factor_method,
            "point_factor_draws": self.point_factor_draws,
            "seed": self.seed,
            "rng_domain_scheme": self.rng_domain_scheme,
            "passed": self.passed,
        }


_PRICE_LADDER_RECORD_FIELDS = frozenset(
    {
        "bumps",
        "violation_share",
        "worst_increase",
        "n_series",
        "n_eligible_series",
        "n_cell_steps",
        "tolerance",
        "relative_scale_floor",
        "point_factor_method",
        "point_factor_draws",
        "seed",
        "rng_domain_scheme",
        "passed",
    }
)


def _checkpoint_final_price_ladder_controls(run: dict) -> dict[str, object]:
    """Return exact reusable controls from a checkpoint's completed training-time audit."""
    stored = run.get("price_ladder")
    if type(stored) is not dict or set(stored) != _PRICE_LADDER_RECORD_FIELDS:
        raise OursError(
            "final checkpoint price ladder requires one exact completed training-time ladder "
            "record; the source checkpoint skipped it or has an unknown schema"
        )
    bumps_value = stored["bumps"]
    if type(bumps_value) is not list or any(
        isinstance(value, bool) or not is_finite_real(value) for value in bumps_value
    ):
        raise OursError("checkpoint price-ladder bumps are not finite numeric values")
    bumps = tuple(float(value) for value in bumps_value)
    tolerance = stored["tolerance"]
    _validate_price_ladder_controls(bumps, tolerance)
    for field in ("violation_share", "worst_increase", "relative_scale_floor"):
        if not is_finite_real(stored[field]):
            raise OursError(f"checkpoint price-ladder {field} is not finite")
    if (
        not 0.0 <= float(stored["violation_share"]) <= 1.0
        or float(stored["worst_increase"]) < 0.0
        or float(stored["relative_scale_floor"]) != PRICE_LADDER_RELATIVE_SCALE_FLOOR
    ):
        raise OursError("checkpoint price-ladder numeric evidence is malformed")
    counts = ("n_series", "n_eligible_series", "n_cell_steps")
    if any(type(stored[field]) is not int or stored[field] < 0 for field in counts):
        raise OursError("checkpoint price-ladder population counts are malformed")
    if (
        stored["n_series"] < 1
        or stored["n_eligible_series"] > stored["n_series"]
        or stored["n_cell_steps"] < stored["n_eligible_series"]
    ):
        raise OursError("checkpoint price-ladder population evidence is contradictory")
    if type(stored["passed"]) is not bool or stored["passed"] is not (
        float(stored["violation_share"]) == 0.0
    ):
        raise OursError("checkpoint price-ladder pass flag contradicts its violation share")
    if type(stored["seed"]) is not int:
        raise OursError("checkpoint price-ladder seed is not an integer")
    if stored["point_factor_method"] not in ("gauss_hermite", "mc"):
        raise OursError("checkpoint price-ladder point-factor method is unsupported")
    draws = stored["point_factor_draws"]
    if (stored["point_factor_method"] == "gauss_hermite" and draws is not None) or (
        stored["point_factor_method"] == "mc" and (type(draws) is not int or draws < 1)
    ):
        raise OursError("checkpoint price-ladder point-factor draw count is malformed")
    if stored["rng_domain_scheme"] not in (RNG_DOMAIN_SCHEME, LEGACY_RNG_DOMAIN_SCHEME):
        raise OursError("checkpoint price-ladder RNG domain scheme is unsupported")
    return {
        "bumps": bumps,
        "tolerance": float(tolerance),
        "n_series": stored["n_series"],
        "point_factor_method": stored["point_factor_method"],
        "point_factor_draws": draws,
        "seed": stored["seed"],
        "rng_domain_scheme": stored["rng_domain_scheme"],
    }


@dataclass(frozen=True)
class PriceLadderEvaluation:
    """Cell-level evidence underlying a :class:`LadderReport`, retained on CPU.

    ``prices`` and ``expected_demand`` are ``[series, horizon, bumps]``. The three change tensors
    and ``violating`` are ``[series, horizon, adjacent bump pairs]``. ``step_change`` is the signed,
    unnormalised change in demand units; ``absolute_step_change`` is its magnitude; and
    ``relative_change`` uses the same lower-rung denominator and floor as the registered report.
    ``eligible`` is ``[series, horizon]`` and remains separate so a post-hoc table can retain an
    unavailable cell without accidentally counting it in the certificate.
    """

    report: LadderReport
    prices: torch.Tensor
    expected_demand: torch.Tensor
    step_change: torch.Tensor
    absolute_step_change: torch.Tensor
    relative_change: torch.Tensor
    violating: torch.Tensor
    eligible: torch.Tensor


@dataclass(frozen=True)
class ReferencePriceContrastEvaluation:
    """Two fitted-model point forecasts under one coherent reference-price intervention.

    All tensors are retained on CPU. ``reference_price`` is the value after conversion to the
    model's known-future dtype, ``identical_input_series`` marks series whose entire changed
    commercial bundle was already at the reference, and the signed shift is reference minus
    factual demand. This is model sensitivity, not counterfactual accuracy.
    """

    reference_price: torch.Tensor
    factual_expected_demand: torch.Tensor
    reference_expected_demand: torch.Tensor
    model_attributed_shift: torch.Tensor
    identical_input_series: torch.Tensor


def _bitwise_element_equal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Elementwise equality of tensor representations, distinguishing signed zero and NaN bits."""
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        raise OursError(
            "bitwise tensor comparison requires equal shape, dtype, and device, got "
            f"{tuple(left.shape)}/{left.dtype}/{left.device} and "
            f"{tuple(right.shape)}/{right.dtype}/{right.device}"
        )
    reinterpret = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }.get(left.dtype)
    if reinterpret is None:
        return left == right
    return left.contiguous().view(reinterpret) == right.contiguous().view(reinterpret)


def evaluate_reference_price_contrast(
    model: CustomModel,
    inputs: dict,
    reference_price: torch.Tensor,
    *,
    point_factor_method: str = "mc",
    point_factor_draws: int | None = 256,
    seed: int = 0,
    rng_domain_scheme: str | None = None,
) -> ReferencePriceContrastEvaluation:
    """Evaluate factual and median-price arms with one coherent commercial input bundle.

    Only raw price, derived discount depth, their two availability flags, and the duplicated
    day-one price/depth branch values and masks change. Promotion and every other caller input
    remain factual. Calling the seeded point route once per arm resets the same factor-domain
    stream, so factor-bearing checkpoints use common random numbers without changing the exact
    factual replay path.
    """
    known = inputs.get("known_future")
    commercial_values = inputs.get("commercial")
    commercial_mask = inputs.get("commercial_mask")
    if rng_domain_scheme is None:
        rng_domain_scheme = RNG_DOMAIN_SCHEME
    if not isinstance(known, torch.Tensor) or known.dim() != 3:
        raise OursError("reference-price contrast requires rank-3 known_future inputs")
    n_series, horizon, width = known.shape
    required_indices = {
        "price": model.price_index,
        "discount_depth": model.discount_index,
        "price_available": model.price_available_index,
        "discount_depth_available": model.discount_available_index,
    }
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width
        for index in required_indices.values()
    ) or len(set(required_indices.values())) != len(required_indices):
        raise OursError(
            "reference-price contrast requires four distinct live price/depth value and "
            f"availability indices, got {required_indices} for width {width}"
        )
    if (
        not isinstance(commercial_values, torch.Tensor)
        or commercial_values.dim() != 2
        or tuple(commercial_values.shape) != (n_series, len(commercial.CHANNELS))
    ):
        shape = (
            None
            if not isinstance(commercial_values, torch.Tensor)
            else tuple(commercial_values.shape)
        )
        raise OursError(
            "reference-price contrast requires the duplicated day-one commercial values, "
            f"got {shape}"
        )
    if (
        not isinstance(commercial_mask, torch.Tensor)
        or commercial_mask.dtype != torch.bool
        or commercial_mask.shape != commercial_values.shape
    ):
        shape = (
            None
            if not isinstance(commercial_mask, torch.Tensor)
            else (tuple(commercial_mask.shape), commercial_mask.dtype)
        )
        raise OursError(
            "reference-price contrast requires one boolean commercial mask aligned to values, "
            f"got {shape}"
        )
    if (
        not isinstance(reference_price, torch.Tensor)
        or reference_price.dim() != 1
        or tuple(reference_price.shape) != (n_series,)
    ):
        shape = (
            None if not isinstance(reference_price, torch.Tensor) else tuple(reference_price.shape)
        )
        raise OursError(
            f"reference_price must have one value per series, expected {(n_series,)}, got "
            f"{shape}"
        )
    if (
        reference_price.device.type != "cpu"
        or known.device.type != "cpu"
        or commercial_values.device.type != "cpu"
        or commercial_mask.device.type != "cpu"
    ):
        raise OursError("registered reference-price contrast requires CPU inputs")
    if (
        not reference_price.is_floating_point()
        or not bool(torch.isfinite(reference_price).all())
        or bool((reference_price <= 0.0).any())
    ):
        raise OursError("reference prices must be finite, floating, and strictly positive")
    if model.price_response_mode == "monotone_raw_v1":
        factual_support = known[..., required_indices["price_available"]] > 0.5
        factual_support &= known[..., required_indices["discount_depth_available"]] > 0.5
        if not bool(factual_support.all()):
            raise OursError(
                "monotone_raw_v1 reference contrast refuses cells without factual frozen-"
                "reference support; setting missing availability flags to one would create "
                "synthetic support rather than a model-backed comparison"
            )

    reference = reference_price.to(dtype=known.dtype, device=known.device)
    if not bool(torch.isfinite(reference).all()) or bool((reference <= 0.0).any()):
        raise OursError("reference prices became invalid in the model's known-future dtype")
    factual_known = known.detach().clone()
    factual_commercial = commercial_values.detach().clone()
    factual_mask = commercial_mask.detach().clone()

    reference_known = known.clone()
    reference_known[..., required_indices["price"]] = reference[:, None]
    reference_known[..., required_indices["discount_depth"]] = 0.0
    reference_known[..., required_indices["price_available"]] = 1.0
    reference_known[..., required_indices["discount_depth_available"]] = 1.0
    reference_commercial = commercial_values.clone()
    reference_commercial[:, 0] = reference
    reference_commercial[:, 1] = 0.0
    reference_mask = commercial_mask.clone()
    reference_mask[:, 0] = True
    reference_mask[:, 1] = True
    expected_reference_known = reference_known.detach().clone()
    expected_reference_commercial = reference_commercial.detach().clone()
    expected_reference_mask = reference_mask.detach().clone()
    reference_inputs = {
        **inputs,
        "known_future": reference_known,
        "commercial": reference_commercial,
        "commercial_mask": reference_mask,
    }

    same_known = torch.ones((n_series, horizon), dtype=torch.bool)
    for index in required_indices.values():
        same_known &= _bitwise_element_equal(known[..., index], reference_known[..., index])
    same_branch = _bitwise_element_equal(commercial_values[:, :2], reference_commercial[:, :2]).all(
        dim=1
    )
    same_branch &= _bitwise_element_equal(commercial_mask[:, :2], reference_mask[:, :2]).all(dim=1)
    identical_input_series = same_known.all(dim=1) & same_branch

    factual = point_forecast(
        model,
        inputs,
        method=point_factor_method,
        factor_draws=point_factor_draws,
        seed=seed,
        rng_domain_scheme=rng_domain_scheme,
    )
    reference_demand = point_forecast(
        model,
        reference_inputs,
        method=point_factor_method,
        factor_draws=point_factor_draws,
        seed=seed,
        rng_domain_scheme=rng_domain_scheme,
    )
    if (
        not bool(_bitwise_element_equal(known, factual_known).all())
        or not bool(_bitwise_element_equal(commercial_values, factual_commercial).all())
        or not bool(_bitwise_element_equal(commercial_mask, factual_mask).all())
    ):
        raise OursError("point evaluation mutated factual commercial inputs")
    if (
        not bool(_bitwise_element_equal(reference_known, expected_reference_known).all())
        or not bool(
            _bitwise_element_equal(reference_commercial, expected_reference_commercial).all()
        )
        or not bool(_bitwise_element_equal(reference_mask, expected_reference_mask).all())
    ):
        raise OursError("point evaluation mutated reference commercial inputs")
    expected_shape = (n_series, horizon)
    for label, value in (("factual", factual), ("reference", reference_demand)):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected_shape
            or value.device.type != "cpu"
            or value.dtype != torch.float64
            or not bool(torch.isfinite(value).all())
        ):
            shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise OursError(
                f"{label} reference-price arm must be finite CPU float64 {expected_shape}, "
                f"got {shape}"
            )
    if bool(identical_input_series.any()) and not bool(
        _bitwise_element_equal(
            factual[identical_input_series], reference_demand[identical_input_series]
        ).all()
    ):
        raise OursError(
            "identical factual/reference inputs did not produce bitwise-equal common-RNG output"
        )
    shift = reference_demand - factual
    if not bool(torch.isfinite(shift).all()):
        raise OursError("reference-price model-attributed shift is non-finite")
    return ReferencePriceContrastEvaluation(
        reference_price=reference.detach().cpu().double(),
        factual_expected_demand=factual.detach().cpu(),
        reference_expected_demand=reference_demand.detach().cpu(),
        model_attributed_shift=shift.detach().cpu(),
        identical_input_series=identical_input_series.detach().cpu(),
    )


def _validate_price_ladder_controls(
    bumps: tuple[float, ...],
    tolerance: float,
) -> None:
    """Validate counterfactual controls before a multi-hour fit or ladder evaluation."""
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, int | float)
        or not math.isfinite(float(tolerance))
        or tolerance <= 0.0
    ):
        raise OursError(f"the ladder tolerance must be finite and positive, got {tolerance!r}")
    if not bumps or any(
        isinstance(bump, bool)
        or not isinstance(bump, int | float)
        or not math.isfinite(float(bump))
        or bump <= -1.0
        for bump in bumps
    ):
        raise OursError(
            f"ladder bumps must be finite numeric fractions greater than -1, got {bumps!r}"
        )
    if tuple(sorted(bumps)) != tuple(bumps) or len(set(bumps)) != len(bumps):
        raise OursError(f"the ladder must be strictly ascending, got {bumps}")
    if 0.0 not in bumps:
        raise OursError(
            "the ladder must contain 0.0: without the factual level, a violation "
            "between two counterfactuals could bracket a well-behaved reality"
        )


def _price_available_mask(
    inputs: dict,
    price_index: int,
    price_available_index: int | None,
) -> torch.Tensor:
    """Cells whose declared price value is numerically available to the model.

    ``price_available_index=None`` is the explicit raw-price contract: every supplied price is
    eligible.  Production models derive the named flag index when they construct the fixed
    commercial layout.  Looking for ``price_index + 3`` merely because the tensor is wide enough
    misread an unrelated fourth future covariate as availability.
    """
    known = inputs.get("known_future")
    if not isinstance(known, torch.Tensor) or known.dim() != 3:
        raise OursError("the price constraint needs rank-3 known_future inputs")
    if (
        isinstance(price_index, bool)
        or not isinstance(price_index, int)
        or not 0 <= price_index < known.shape[-1]
    ):
        raise OursError(
            f"price_index must name a known_future column, got {price_index!r} for width "
            f"{known.shape[-1]}"
        )
    if price_available_index is None:
        return torch.ones(known.shape[:2], dtype=torch.bool, device=known.device)
    if (
        isinstance(price_available_index, bool)
        or not isinstance(price_available_index, int)
        or not 0 <= price_available_index < known.shape[-1]
    ):
        raise OursError(
            f"price_available_index must name a known_future column or be None, got "
            f"{price_available_index!r} for width {known.shape[-1]}"
        )
    return known[..., price_available_index] > 0.5


def _price_intervention_mask(
    inputs: dict,
    price_index: int,
    price_available_index: int | None,
) -> torch.Tensor:
    """Cells a relative price intervention can actually perturb.

    Zero is a valid observed price (for example, a free item), but multiplying it by
    ``1 + bump`` leaves every ladder rung identical.  Such a cell is numerically available to
    the model yet supplies no evidence about a relative price response.  Non-finite raw prices
    are excluded defensively too; production commercial assembly already marks them unavailable.
    """
    available = _price_available_mask(inputs, price_index, price_available_index)
    price = inputs["known_future"][..., price_index]
    return available & torch.isfinite(price) & (price > 0.0)


def _price_penalty_batch_weight(
    inputs: dict,
    price_index: int,
    price_available_index: int | None,
    *,
    n_batches: int,
    total_eligible: int,
) -> float:
    """Scale an eligible-cell local mean back to one fixed epoch-wide objective.

    ``total_eligible`` is measured on the original training split before covariate dropout.
    A hidden price therefore contributes zero for that stochastic step without renormalising the
    surviving cells upward.  With dropout disabled, averaging the weighted minibatch losses is
    exactly the global mean over perturbable price cell-steps, independent of block packing or
    sharding.
    """
    if isinstance(n_batches, bool) or not isinstance(n_batches, int) or n_batches < 1:
        raise OursError(f"n_batches must be a positive integer, got {n_batches!r}")
    if (
        isinstance(total_eligible, bool)
        or not isinstance(total_eligible, int)
        or total_eligible < 0
    ):
        raise OursError(f"total_eligible must be a non-negative integer, got {total_eligible!r}")
    local_eligible = int(_price_intervention_mask(inputs, price_index, price_available_index).sum())
    if total_eligible == 0:
        if local_eligible:
            raise OursError(
                "a price minibatch contains eligible cells but the fixed training denominator "
                "is zero"
            )
        return 0.0
    if local_eligible > total_eligible:
        raise OursError(
            f"a price minibatch has {local_eligible} eligible cells against an epoch total of "
            f"{total_eligible}"
        )
    return local_eligible * n_batches / total_eligible


def evaluate_price_ladder(
    model: CustomModel,
    inputs: dict,
    *,
    bumps: tuple[float, ...] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2),
    tolerance: float = 1e-6,
    point_factor_method: str = "gauss_hermite",
    point_factor_draws: int | None = None,
    seed: int = 0,
    rng_domain_scheme: str | None = None,
) -> PriceLadderEvaluation:
    """Evaluate the declared price ladder and retain its cell-level evidence.

    Raises:
        OursError: no price channel, an unsorted ladder, or a ladder without the factual 0.0.
    """
    if model.price_index is None:
        raise OursError("the model has no price channel; a ladder cannot be evaluated")
    if rng_domain_scheme is None:
        rng_domain_scheme = RNG_DOMAIN_SCHEME
    _validate_price_ladder_controls(bumps, tolerance)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OursError(
            f"the price ladder seed must be an integer for reproducible common-random-number "
            f"comparisons, got {seed!r}"
        )
    if rng_domain_scheme not in (RNG_DOMAIN_SCHEME, LEGACY_RNG_DOMAIN_SCHEME):
        raise OursError(
            f"unknown RNG domain scheme {rng_domain_scheme!r}; expected "
            f"{RNG_DOMAIN_SCHEME!r} or {LEGACY_RNG_DOMAIN_SCHEME!r}"
        )
    if point_factor_method not in ("gauss_hermite", "mc"):
        raise OursError(
            f"point_factor_method must be gauss_hermite or mc, got {point_factor_method!r}"
        )
    if (
        point_factor_method == "mc"
        and point_factor_draws is not None
        and (
            isinstance(point_factor_draws, bool)
            or not isinstance(point_factor_draws, int)
            or point_factor_draws < 1
        )
    ):
        raise OursError(
            f"point_factor_draws must be a positive integer under mc, got "
            f"{point_factor_draws!r}"
        )
    resolved_draws = (
        POINT_FACTOR_DRAWS
        if point_factor_method == "mc" and point_factor_draws is None
        else point_factor_draws
    )
    # `point_forecast` returns CPU tensors; keep the eligibility mask on CPU so the reductions
    # below do not depend on the device the caller moved its inputs to.
    eligible = _price_intervention_mask(
        inputs, model.price_index, model.price_available_index
    ).cpu()
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        raise OursError(
            "the sampled forecast cells contain no finite positive planned price; a relative "
            "price-ladder certificate over unavailable or zero-price cells would be vacuous"
        )
    structural_base = None
    if model.price_response is not None:
        # Raw price/depth are absent from the expensive encoder/decoder route. Encode that
        # price-independent context once and replace only the diagonal depth coordinate at each
        # ladder rung. This is the serving-time scalability benefit of the structural design,
        # not an approximation: the same BasePrediction is what a full forward call constructs.
        with evaluating(model), torch.no_grad():
            structural_base = model.encode_base(**_base_inputs(inputs))
    prices, demands = [], []
    for bump in bumps:
        raised = _raise_price_inputs(
            inputs,
            bump,
            model.price_index,
            model.discount_index,
            model.discount_available_index,
            model.price_available_index,
        )
        prices.append(raised["known_future"][..., model.price_index].detach().cpu().clone())
        # Eligibility must certify the number that is actually ranked. The former diagnostic
        # averaged eight fixed factor scenarios from the training inputs; a measured hurdle
        # model passed that sample while its true factor-marginal demand rose 0.176% at +10%.
        # Another version then hard-coded GH: a valid MC forecast outside the GH envelope could
        # be emitted but not diagnosed, and an in-envelope MC run was certified against a
        # different point number. Re-run the caller's declared route with common random numbers.
        scenario_base = None
        scenario_inputs = raised
        if structural_base is not None:
            depth, response_mask = model.structural_price_coordinates(raised["known_future"])
            scenario_base = replace(
                structural_base, price_depth=depth, price_response_mask=response_mask
            )
            scenario_inputs = None
        demands.append(
            point_forecast(
                model,
                scenario_inputs,
                method=point_factor_method,
                factor_draws=point_factor_draws,
                seed=seed,
                rng_domain_scheme=rng_domain_scheme,
                base=scenario_base,
            )
        )
    ladder = torch.stack(demands, dim=-1)  # [series, horizon, bumps]
    steps = ladder[..., 1:] - ladder[..., :-1]  # higher price must not raise demand
    scale = ladder[..., :-1].abs().clamp_min(PRICE_LADDER_RELATIVE_SCALE_FLOOR)
    relative_increase = steps / scale
    violating = relative_increase > tolerance
    eligible_steps = eligible.unsqueeze(-1).expand_as(relative_increase)
    report = LadderReport(
        bumps=tuple(bumps),
        violation_share=float(violating.any(dim=-1)[eligible].double().mean()),
        worst_increase=float(relative_increase[eligible_steps].max().clamp_min(0.0)),
        n_series=int(ladder.shape[0]),
        n_eligible_series=int(eligible.any(dim=-1).sum()),
        n_cell_steps=n_eligible,
        tolerance=tolerance,
        relative_scale_floor=PRICE_LADDER_RELATIVE_SCALE_FLOOR,
        point_factor_method=point_factor_method,
        point_factor_draws=(resolved_draws if point_factor_method == "mc" else None),
        seed=seed,
        rng_domain_scheme=rng_domain_scheme,
    )
    return PriceLadderEvaluation(
        report=report,
        prices=torch.stack(prices, dim=-1),
        expected_demand=ladder,
        step_change=steps,
        absolute_step_change=steps.abs(),
        relative_change=relative_increase,
        violating=violating,
        eligible=eligible,
    )


def verify_price_ladder(
    model: CustomModel,
    inputs: dict,
    *,
    bumps: tuple[float, ...] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2),
    tolerance: float = 1e-6,
    point_factor_method: str = "gauss_hermite",
    point_factor_draws: int | None = None,
    seed: int = 0,
    rng_domain_scheme: str | None = None,
) -> LadderReport:
    """Evaluate the declared price ladder on the given cells. See `LadderReport`."""
    return evaluate_price_ladder(
        model,
        inputs,
        bumps=bumps,
        tolerance=tolerance,
        point_factor_method=point_factor_method,
        point_factor_draws=point_factor_draws,
        seed=seed,
        rng_domain_scheme=rng_domain_scheme,
    ).report


def _event_episodes(
    events, target_origins: list[str], horizon: int, max_support: int
) -> dict[str, int]:
    """How many OCCURRENCES of each event fall inside the training targets' reach.

    The effective sample size for an event is its EPISODES, not its rows
    (`ForecastDataSpec.event_episode_col`): two Easters are two replications no matter how many
    stores observed them. Validation here splits by SERIES, so both halves see the same
    occurrences through different series -- what bounds any claim about event accuracy is how
    many distinct occurrences training covered at all, and that number belongs in the run
    record rather than being reconstructed from dates later.
    """
    days: set[dt.date] = set()
    for origin in target_origins:
        end = dt.date.fromisoformat(origin)
        days.update(end - dt.timedelta(days=j) for j in range(horizon))
    out: dict[str, int] = {}
    for name, when, _ in events:
        occurrence = dt.date.fromisoformat(when)
        covered = any(abs((occurrence - day).days) <= max_support for day in days)
        out[name] = out.get(name, 0) + int(covered)
    return out


def _raise_price(
    known: torch.Tensor,
    bump: float,
    price_index: int,
    discount_index: int | None,
    discount_available_index: int | None,
    price_available_index: int | None = None,
) -> torch.Tensor:
    """The raised-price counterfactual, with every derived column moved consistently.

    Discount depth is `1 - price/reference`, so a price raised by a fraction `b` implies
    `depth' = (1+b)*depth - b`. Raising the price while holding the depth describes a world
    that cannot exist -- the off-manifold input the intervention-bundle rule forbids -- and it
    let the model satisfy monotonicity through the depth channel while the raw-price channel
    misbehaved. Cells whose depth is UNAVAILABLE keep their zero placeholder: transforming a
    filler would fabricate a discount out of a missing value. A zero-filled missing price is
    not an intervention: neither price nor its derived depth moves. An unavailable positive
    placeholder is likewise held fixed when an explicit availability channel exists.
    """
    raised = known.clone()
    price = raised[..., price_index]
    perturbable = torch.isfinite(price) & (price > 0.0)
    if price_available_index is not None:
        perturbable &= _price_available_mask(
            {"known_future": known}, price_index, price_available_index
        )
    raised[..., price_index] = torch.where(perturbable, price * (1.0 + bump), price)
    if discount_index is not None:
        depth = raised[..., discount_index]
        adjusted = (1.0 + bump) * depth - bump
        if discount_available_index is not None:
            have = raised[..., discount_available_index] > 0.5
            perturbable &= have
        adjusted = torch.where(perturbable, adjusted, depth)
        raised[..., discount_index] = adjusted
    return raised


def _raise_price_inputs(
    inputs: dict,
    bump: float,
    price_index: int,
    discount_index: int | None,
    discount_available_index: int | None,
    price_available_index: int | None = None,
) -> dict:
    """Raise every model representation of price as one coherent counterfactual bundle.

    Horizon price/depth live in ``known_future`` while the commercial branch deliberately receives
    the first horizon day's same three channels as an origin token. Changing only one copy asks two
    branches to evaluate different worlds and can let a commercial-only price violation pass both
    the training penalty and the hard audit. Promotion and availability are held fixed.
    """
    raised_known = _raise_price(
        inputs["known_future"],
        bump,
        price_index,
        discount_index,
        discount_available_index,
        price_available_index,
    )
    return _with_known_future_inputs(inputs, raised_known, (price_index, discount_index, None))


def _with_known_future_inputs(
    inputs: dict,
    known_future: torch.Tensor,
    commercial_known_indices: tuple[int | None, ...],
) -> dict:
    """Replace known future and synchronise its duplicated day-one commercial branch values."""
    routed = {**inputs, "known_future": known_future}
    values = inputs.get("commercial")
    if values is None:
        return routed
    if values.shape[-1] != len(commercial.CHANNELS) or len(commercial_known_indices) != len(
        commercial.CHANNELS
    ):
        raise OursError(
            f"commercial counterfactual expected channels {commercial.CHANNELS} and matching "
            f"known indices, got value width {values.shape[-1]} and "
            f"{len(commercial_known_indices)} indices"
        )
    mask = inputs.get("commercial_mask")
    if mask is not None and mask.shape != values.shape:
        raise OursError(
            f"commercial_mask {tuple(mask.shape)} does not match commercial values "
            f"{tuple(values.shape)}"
        )
    if mask is not None:
        mask = _binary_mask(mask, values, "commercial availability mask", error_type=OursError)
    commercial_values = values.clone()
    for column, known_index in enumerate(commercial_known_indices):
        if known_index is None:
            continue
        replacement = known_future[:, 0, known_index]
        if mask is not None:
            replacement = torch.where(mask[:, column], replacement, commercial_values[:, column])
        commercial_values[:, column] = replacement
    routed["commercial"] = commercial_values
    return routed


def _hide_known_channels(
    known_future: torch.Tensor, hidden: torch.Tensor, columns: list[tuple[int, int]]
) -> torch.Tensor:
    """Zero the known-future VALUE and AVAILABILITY columns of hidden channels, per cell.

    The other half of covariate dropout. `drop_covariates` hides a channel from the encoder's
    branch token -- but the horizon's prices act on the forecast mostly through the DECODER's
    known-future columns, and a dropout that leaves that path intact regularises the wrong
    door: the model stays free to lean on a channel it will sometimes not have. Hiding means
    looking exactly like a genuinely missing cell -- value zero, flag zero -- for every horizon
    day of the dropped channel.

    Args:
        known_future: `[cells, horizon, n_known]`.
        hidden: `[cells, n_channels]` True where a channel was dropped for that cell.
        columns: `(value_column, availability_column)` per channel, in channel order.
    """
    if not bool(hidden.any()):
        return known_future
    out = known_future.clone()
    for j, (value_col, avail_col) in enumerate(columns):
        rows = hidden[:, j]
        if bool(rows.any()):
            out[rows, :, value_col] = 0.0
            out[rows, :, avail_col] = 0.0
    return out


def _composite_groups(
    factor_index: torch.Tensor, product_index: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Dense block ids for `(factor_group, product)` pairs -- the unit that shares a drawn path.

    Dense rather than `factor * n_products + product`, because the sparse form draws paths for
    every ABSENT combination too: 3 states x 3,049 items is fine, but a client with a deep
    hierarchy and a partial assortment would pay memory for pairs no cell belongs to.
    `torch.unique` is sorted, so the same series set gives the same block ids every run.
    """
    n_products = int(product_index.max()) + 1
    combined = factor_index * n_products + product_index
    levels, dense = torch.unique(combined, sorted=True, return_inverse=True)
    return dense, len(levels)


def _independent_group_index(n_rows: int) -> torch.Tensor:
    """One block per predictive marginal for factorless bottom-up training and sampling.

    The block packer keeps equal labels indivisible.  Giving it the former all-zero fallback made
    a missing factor declaration one block as large as the catalogue; unique labels restore an
    ordinary hard batching cap and record the intended product likelihood explicitly.
    """
    if n_rows < 1:
        raise OursError(f"independent bottom-up grouping needs at least one row, got {n_rows}")
    return torch.arange(n_rows, dtype=torch.long)


def _dense_group_index(group_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(dense_ids, original_labels)` without changing group equality.

    Group ids often come from a larger forecast index. A minibatch containing global labels
    `[0, 100000]` has TWO latent blocks, not 100001; using `max + 1` allocates paths for every
    absent label and can exhaust memory on a perfectly small query. `original_labels` is sorted
    and lets an explicitly supplied global factor path be selected consistently.
    """
    if group_index.dim() != 1 or group_index.numel() == 0:
        raise OursError(
            f"group_index must be a non-empty rank-1 tensor, got {tuple(group_index.shape)}"
        )
    labels, dense = torch.unique(group_index.long(), sorted=True, return_inverse=True)
    return dense, labels


def _sample_series_indices(series_ids: list[str], sample_size: int, seed: int) -> list[int]:
    """Stable uniform sample without replacement, independent of lexical catalogue prefixes."""
    if sample_size < 1:
        raise OursError(f"paths_sample must be >= 1, got {sample_size}")
    ranked = sorted(
        range(len(series_ids)),
        key=lambda i: hashlib.sha256(f"{seed}\0{series_ids[i]}".encode()).digest(),
    )
    return ranked[: min(sample_size, len(ranked))]


def _offset_dense_blocks(group_index: torch.Tensor, offset: int) -> tuple[torch.Tensor, int]:
    """Dense-remap one forecast episode and place it after all earlier episodes' blocks."""
    if offset < 0:
        raise OursError(f"factor-block offset must be non-negative, got {offset}")
    dense, labels = _dense_group_index(group_index)
    return dense + offset, offset + len(labels)


def pool_panels(panels: list[Panel], spark: SparkSession, out_dir: str | pathlib.Path) -> Panel:
    """Union several clients into ONE pooled panel -- the loader the one-model claim needs.

    Every client keeps its own rows; what pooling adds is a `client` column and NAMESPACING:
    series ids and the declared product/store/factor levels are prefixed `client|value`,
    because SKU codes are client-scoped (the spec's own words) and an unprefixed union would
    hand two different products one embedding row on a string coincidence. Attribute columns
    are left as declared -- whether two clients' `brand` values mean the same thing is the
    caller's assertion, and prefixing them would silently forbid the transfer that shared
    attributes exist to provide.

    Raises:
        OursError: fewer than two panels, or panels whose roles or estimands disagree --
            pooling those would train one model on two different contracts.
    """
    if len(panels) < 2:
        raise OursError(f"pooling needs at least two panels, got {len(panels)}")
    reference = panels[0]
    role_fields = (
        "series_col",
        "time_col",
        "target_col",
        "exposure_col",
        "price_col",
        "promo_col",
        "static",
        "static_reals",
        "factor_group",
        "product_col",
        "store_col",
        "category_col",
        "target_estimand",
        "season_length",
    )
    for other in panels[1:]:
        mismatched = [f for f in role_fields if getattr(other, f) != getattr(reference, f)]
        if mismatched:
            raise OursError(
                f"panels disagree on role(s) {mismatched}; one pooled model cannot train on "
                "two different contracts"
            )
    prefix_cols = [
        c
        for c in {
            reference.series_col,
            reference.product_col,
            reference.store_col,
            reference.factor_group,
        }
        if c
    ]
    frames = []
    for k, panel in enumerate(panels):
        client = f"c{k}"
        frame = spark.read.parquet(panel.path).withColumn("client", F.lit(client))
        for column in prefix_cols:
            frame = frame.withColumn(
                column, F.concat_ws("|", F.lit(client), F.col(column).cast("string"))
            )
        frames.append(frame)
    pooled = frames[0]
    for frame in frames[1:]:
        pooled = pooled.unionByName(frame)
    pooled.write.mode("overwrite").parquet(str(out_dir))
    import dataclasses as _dc

    return _dc.replace(reference, path=str(out_dir)).validate()


def _validated_vocabulary(levels: object, role: str) -> list[str]:
    """Return one unambiguous stored identity vocabulary or refuse it."""
    if (
        not isinstance(levels, list | tuple)
        or any(not isinstance(level, str) or not level.strip() for level in levels)
        or len(set(levels)) != len(levels)
    ):
        raise OursError(f"checkpoint {role} vocabulary must be a unique list of non-blank strings")
    return list(levels)


def _indices_from_levels(
    levels: list[str],
    values: dict[str, str | None],
    series_ids: list[str],
    oov_row: int,
    *,
    role: str = "identity",
) -> torch.Tensor:
    """Map each series' level through a STORED vocabulary; unseen levels take the OOV row."""
    levels = _validated_vocabulary(levels, role)
    if isinstance(oov_row, bool) or not isinstance(oov_row, int) or oov_row != len(levels):
        raise OursError(
            f"checkpoint {role} OOV row {oov_row!r} must equal vocabulary size " f"{len(levels)}"
        )
    at = {level: i for i, level in enumerate(levels)}
    out = torch.full((len(series_ids),), oov_row, dtype=torch.long)
    for i, sid in enumerate(series_ids):
        level = values.get(sid)
        if level is not None and level in at:
            out[i] = at[level]
    return out


def _query_composite_groups(
    series_ids: list[str],
    factor_values: dict[str, str | None],
    product_values: dict[str, str | None] | None,
) -> tuple[torch.Tensor, int]:
    """Dense replay blocks from RAW `(factor_group, product)` identities.

    Embedding lookup and latent-path identity are different contracts.  Several unseen products
    legitimately share one pooled OOV *parameter* row, but they must not thereby share one random
    factor path: the architecture draws a path per semantic pair, not per embedding bucket.  Raw
    string pairs preserve that equality relation for both known and unseen identities.  Sorting
    makes the dense ids deterministic and matches the training vocabulary order when all levels
    are known.
    """
    if not series_ids:
        raise OursError("composite replay grouping needs at least one series")

    # Missing is not a semantic identity two series can be proved to share. Tag it with the
    # series id so null factor/product roles get independent paths, while repeated KNOWN raw
    # pairs still share exactly one systemic trajectory. An absent product ROLE is different:
    # the declared degradation intentionally shares one path per factor group.
    keys: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for sid in series_ids:
        factor = factor_values.get(sid)
        factor_key = ("missing", sid) if factor is None else ("value", factor)
        if product_values is None:
            product_key = ("role_absent", "")
        else:
            product = product_values.get(sid)
            product_key = ("missing", sid) if product is None else ("value", product)
        keys.append((factor_key, product_key))

    levels = sorted(set(keys))
    at = {key: row for row, key in enumerate(levels)}
    return torch.tensor([at[key] for key in keys], dtype=torch.long), len(levels)


def _contract_value(value):
    """Canonicalise nested dataclass output for exact panel-role comparison."""
    if isinstance(value, dict):
        return tuple(sorted((str(key), _contract_value(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_contract_value(item) for item in value)
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _panel_role_mismatches(panel: Panel, stored: dict | None) -> list[str]:
    """Fields whose declared semantics changed since fit, excluding replaceable data paths."""
    if not isinstance(stored, dict):
        return ["panel"]
    supplied = asdict(panel)
    mismatched = []
    for field in sorted((set(stored) | set(supplied)) - {"path"}):
        old, new = stored.get(field), supplied.get(field)
        if field == "plan":
            # A serving call may point at a refreshed effective-dated feed, but the columns that
            # give that feed its meaning must remain the fitted contract.
            if isinstance(old, dict):
                old = {key: value for key, value in old.items() if key != "path"}
            if isinstance(new, dict):
                new = {key: value for key, value in new.items() if key != "path"}
        if _contract_value(old) != _contract_value(new):
            mismatched.append(field)
    return mismatched


def _checkpoint_point_replay(
    run: dict,
    override_draws: int | None,
    *,
    factorless: bool = False,
) -> tuple[str, int | None, str]:
    """Resolve only point-inference facts a checkpoint actually recorded.

    Checkpoints written before 2026-08-25 record neither the point integration count nor the RNG
    domain scheme. Their source default was 1,000/direct-seed MC, but a caller could override that
    count and the artifact did not persist the override. Silently substituting today's 128-draw
    default therefore changes the forecast while claiming to replay it. Refuse unless the caller
    explicitly chooses a new MC count; that opt-in is a re-evaluation of the old weights, not a
    claim that an unrecorded historical invocation was reconstructed.
    """
    if override_draws is not None and (
        isinstance(override_draws, bool)
        or not isinstance(override_draws, int)
        or override_draws < 1
    ):
        raise OursError(
            f"point_factor_draws must be a positive integer when supplied, got {override_draws!r}"
        )
    method = run.get("point_factor_method")
    if method is None:
        if override_draws is None and not factorless:
            raise OursError(
                "this checkpoint predates point replay metadata and does not record its Monte "
                "Carlo factor-draw count. Exact historical point replay cannot be inferred; pass "
                "point_factor_draws explicitly to request a legacy-seeded MC re-evaluation, or "
                "refit to produce a self-describing checkpoint"
            )
        method = "mc"
    if method not in ("gauss_hermite", "mc"):
        raise OursError(f"checkpoint records unknown point_factor_method {method!r}")
    if method == "gauss_hermite" and override_draws is not None:
        raise OursError(
            "point_factor_draws has no effect on a checkpoint whose point_factor_method is "
            "gauss_hermite; refusing an argument whose result would be silently ignored"
        )

    draws = override_draws
    if draws is None and method == "mc":
        draws = run.get("point_factor_draws")
        if draws is None and factorless:
            # The old MC entry point already evaluated a factorless model exactly once; neither
            # draw count nor seed can change that result, so there is no missing stochastic fact.
            draws = 1
        elif draws is None:
            raise OursError(
                "checkpoint records MC point inference but no point_factor_draws; exact replay "
                "cannot be inferred. Pass point_factor_draws explicitly or refit"
            )
    if draws is not None and (isinstance(draws, bool) or not isinstance(draws, int) or draws < 1):
        raise OursError(f"checkpoint point_factor_draws must be a positive integer, got {draws!r}")
    scheme = run.get("rng_domain_scheme")
    if scheme is None:
        # A transitional checkpoint may record method/count but predate the RNG-map field.  The
        # implementation at that time drew one batched ``factor.draw(S)``; the current canonical
        # route draws S trajectories separately so ``chunk`` cannot change them.  Direct seed +
        # count is therefore still insufficient for exact replay.  An explicit override opts in
        # to a newly evaluated legacy-seeded result; factorless and GH routes have no missing
        # stochastic mapping.
        if method == "mc" and not factorless and override_draws is None:
            raise OursError(
                "this checkpoint records an MC point count but not the RNG trajectory-mapping "
                "scheme. Exact historical point replay cannot be inferred; pass "
                "point_factor_draws explicitly to request a legacy-seeded re-evaluation, or "
                "refit to produce a self-describing checkpoint"
            )
        scheme = LEGACY_RNG_DOMAIN_SCHEME
    # Validate the recorded scheme now rather than after feature assembly.
    _domain_seed(0, "factor", scheme)
    return method, draws, scheme


def _paths_overlap(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Whether either local path contains the other, including macOS spelling aliases.

    ``Path.resolve`` closes symlink aliases, but it deliberately preserves unresolved spelling.
    That is insufficient on the default APFS configuration: ``forecast``/``FORECAST`` and
    NFC/NFD spellings can name one future inode even when neither path exists yet.  Treating the
    normalised spellings as aliases is conservative on a case-sensitive volume, which is the
    safe direction for an overwrite guard.
    """
    left, right = left.resolve(strict=False), right.resolve(strict=False)
    if left.exists() and right.exists():
        try:
            if os.path.samefile(left, right):
                return True
        except OSError:
            pass

    def component_key(parts) -> tuple[str, ...]:
        return tuple(unicodedata.normalize("NFD", part).casefold() for part in parts)

    def existing_ancestor(path: pathlib.Path) -> tuple[pathlib.Path, tuple[str, ...]]:
        tail = []
        cursor = path
        while not cursor.exists() and cursor != cursor.parent:
            tail.append(cursor.name)
            cursor = cursor.parent
        return cursor, component_key(reversed(tail))

    # macOS firmlinks (notably /Users -> /System/Volumes/Data/Users) are inode aliases that
    # pathlib does not resolve lexically. Compare the nearest existing anchors, then only their
    # unresolved tails.
    left_anchor, left_tail = existing_ancestor(left)
    right_anchor, right_tail = existing_ancestor(right)
    try:
        same_anchor = os.path.samefile(left_anchor, right_anchor)
    except OSError:
        same_anchor = False
    if same_anchor and (
        left_tail == right_tail
        or left_tail == right_tail[: len(left_tail)]
        or right_tail == left_tail[: len(right_tail)]
    ):
        return True

    left_key, right_key = component_key(left.parts), component_key(right.parts)
    return (
        left_key == right_key
        or left_key == right_key[: len(left_key)]
        or right_key == left_key[: len(right_key)]
    )


_AGGREGATE_TRAJECTORY_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "artifact_id",
        "origin",
        "horizon",
        "grouping_cols",
        "n_ancestors",
        "ancestor_id_schema",
        "provider_identity",
        "provider_base_artifact_sha256",
        "provider_base_pretraining_cutoff",
        "provider_adaptation_identity",
        "provider_adaptation_artifact_sha256",
        "provider_adaptation_cutoff",
        "provider_context_cutoff",
        "baseline_identity",
        "baseline_artifact_sha256",
        "baseline_training_cutoff",
        "baseline_context_cutoff",
        "scale_identity",
        "scale_artifact_sha256",
        "scale_source_cutoff",
        "output_statistic",
        "input_roles",
        "provider_execution",
        "price_inputs",
        "provider_inference_post_origin_inputs",
        "price_invariant",
        "evidence_role",
        "claim_eligible",
        "values_parquet_sha256",
    }
)
_AGGREGATE_TRAJECTORY_VALUE_FIELDS = frozenset(
    {
        "ancestor_id",
        "date",
        "provider",
        "baseline",
        "scale",
        "available",
    }
)
_AGGREGATE_TRAJECTORY_STABLE_MANIFEST_FIELDS = (
    "grouping_cols",
    "n_ancestors",
    "ancestor_id_schema",
    "provider_identity",
    "provider_base_artifact_sha256",
    "provider_base_pretraining_cutoff",
    "baseline_identity",
    "scale_identity",
    "output_statistic",
    "input_roles",
    "provider_execution",
    "price_inputs",
    "provider_inference_post_origin_inputs",
    "price_invariant",
    "evidence_role",
    "claim_eligible",
)
_M5_L9_STORES = (
    "CA_1",
    "CA_2",
    "CA_3",
    "CA_4",
    "TX_1",
    "TX_2",
    "TX_3",
    "WI_1",
    "WI_2",
    "WI_3",
)
_M5_L9_DEPARTMENTS = (
    "FOODS_1",
    "FOODS_2",
    "FOODS_3",
    "HOBBIES_1",
    "HOBBIES_2",
    "HOUSEHOLD_1",
    "HOUSEHOLD_2",
)


def _m5_l9_ancestor_ids() -> frozenset[str]:
    return frozenset(
        aggregate_trajectory_ancestor_id(
            (
                (AGGREGATE_TRAJECTORY_GROUPING_COLS[0], store),
                (AGGREGATE_TRAJECTORY_GROUPING_COLS[1], department),
            )
        )
        for store in _M5_L9_STORES
        for department in _M5_L9_DEPARTMENTS
    )


def _sha256_is_canonical(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_iso_date(value: object, label: str) -> dt.date:
    try:
        parsed = dt.date.fromisoformat(value) if type(value) is str else None
    except ValueError as exc:
        raise OursError(f"aggregate trajectory {label} is not canonical ISO YYYY-MM-DD") from exc
    if parsed is None or parsed.isoformat() != value:
        raise OursError(f"aggregate trajectory {label} is not canonical ISO YYYY-MM-DD")
    return parsed


def _validate_aggregate_trajectory_manifest_claims(
    manifest: object,
    *,
    origin: str,
    horizon: int,
) -> dict[str, object]:
    """Validate the manifest claims independently of the mutable values tree."""
    if type(manifest) is not dict:
        raise OursError("aggregate trajectory manifest must be a JSON mapping")
    observed = set(manifest)
    if observed != _AGGREGATE_TRAJECTORY_MANIFEST_FIELDS:
        raise OursError(
            "aggregate trajectory manifest is not the exact current schema: "
            f"missing={sorted(_AGGREGATE_TRAJECTORY_MANIFEST_FIELDS - observed)}, "
            f"extra={sorted(observed - _AGGREGATE_TRAJECTORY_MANIFEST_FIELDS)}"
        )
    if manifest["schema"] != AGGREGATE_TRAJECTORY_SCHEMA:
        raise OursError(f"aggregate trajectory schema must be {AGGREGATE_TRAJECTORY_SCHEMA!r}")
    for field in ("artifact_id", "provider_identity", "baseline_identity", "scale_identity"):
        value = manifest[field]
        if type(value) is not str or value != value.strip() or not value:
            raise OursError(f"aggregate trajectory {field} must be a nonblank string")
    grouping_cols = manifest["grouping_cols"]
    if (
        type(grouping_cols) is not list
        or not grouping_cols
        or any(
            type(column) is not str or column != column.strip() or not column
            for column in grouping_cols
        )
        or len(set(grouping_cols)) != len(grouping_cols)
    ):
        raise OursError(
            "aggregate trajectory grouping_cols must be an ordered list of distinct nonblank "
            "column names"
        )
    if grouping_cols != list(AGGREGATE_TRAJECTORY_GROUPING_COLS):
        raise OursError(
            "this registered pilot requires the ordered M5 L9 grouping_cols "
            f"{list(AGGREGATE_TRAJECTORY_GROUPING_COLS)}, got {grouping_cols}"
        )
    if (
        type(manifest["n_ancestors"]) is not int
        or manifest["n_ancestors"] != AGGREGATE_TRAJECTORY_ANCESTOR_COUNT
    ):
        raise OursError(
            "this registered M5 L9 pilot requires exactly "
            f"{AGGREGATE_TRAJECTORY_ANCESTOR_COUNT} aggregate ancestors, got "
            f"{manifest['n_ancestors']!r}"
        )
    if manifest["ancestor_id_schema"] != "canonical_json_name_value_pairs_v1":
        raise OursError(
            "aggregate trajectory ancestor_id_schema must be " "canonical_json_name_value_pairs_v1"
        )
    manifest_origin = _canonical_iso_date(manifest["origin"], "origin")
    requested_origin = _canonical_iso_date(origin, "requested origin")
    if manifest_origin != requested_origin:
        raise OursError(
            f"aggregate trajectory origin {manifest_origin} does not match forecast origin "
            f"{requested_origin}"
        )
    if (
        type(manifest["horizon"]) is not int
        or manifest["horizon"] < 1
        or manifest["horizon"] != horizon
    ):
        raise OursError(
            f"aggregate trajectory horizon {manifest['horizon']!r} does not match {horizon}"
        )
    for field in (
        "provider_base_artifact_sha256",
        "baseline_artifact_sha256",
        "scale_artifact_sha256",
        "values_parquet_sha256",
    ):
        if not _sha256_is_canonical(manifest[field]):
            raise OursError(f"aggregate trajectory {field} is not a canonical SHA-256")
    base_cutoff = _canonical_iso_date(
        manifest["provider_base_pretraining_cutoff"], "provider_base_pretraining_cutoff"
    )
    adaptation_identity = manifest["provider_adaptation_identity"]
    adaptation_hash = manifest["provider_adaptation_artifact_sha256"]
    adaptation_cutoff_value = manifest["provider_adaptation_cutoff"]
    frozen_provider = (
        adaptation_identity is None and adaptation_hash is None and adaptation_cutoff_value is None
    )
    adapted_provider = (
        type(adaptation_identity) is str
        and adaptation_identity.strip() == adaptation_identity
        and bool(adaptation_identity)
        and _sha256_is_canonical(adaptation_hash)
        and type(adaptation_cutoff_value) is str
    )
    if not (frozen_provider or adapted_provider):
        raise OursError(
            "aggregate trajectory adaptation identity/hash/cutoff must be either all null for "
            "the frozen provider or one complete current-origin adapter seal"
        )
    if adapted_provider:
        adaptation_cutoff = _canonical_iso_date(
            adaptation_cutoff_value, "provider_adaptation_cutoff"
        )
        if adaptation_cutoff != manifest_origin:
            raise OursError(
                "aggregate trajectory provider_adaptation_cutoff must equal its episode origin; "
                "a later-trained adapter cannot be backcast and an earlier fold is not this "
                "registered episode"
            )
    for field in (
        "provider_context_cutoff",
        "baseline_training_cutoff",
        "baseline_context_cutoff",
        "scale_source_cutoff",
    ):
        cutoff = _canonical_iso_date(manifest[field], field)
        if cutoff > manifest_origin:
            raise OursError(
                f"aggregate trajectory {field}={cutoff} is after origin {manifest_origin}"
            )
    evidence_role = manifest["evidence_role"]
    claim_eligible = manifest["claim_eligible"]
    if evidence_role == "origin_safe":
        if claim_eligible is not True or base_cutoff > manifest_origin:
            raise OursError(
                "origin_safe aggregate evidence requires claim_eligible=true and a base "
                "pretraining cutoff at or before its origin"
            )
    elif evidence_role == "descriptive_contaminated_probe":
        if claim_eligible is not False:
            raise OursError(
                "descriptive_contaminated_probe aggregate evidence must be claim_eligible=false"
            )
    else:
        raise OursError(
            "aggregate trajectory evidence_role must be exactly 'origin_safe' or "
            "'descriptive_contaminated_probe'"
        )
    if base_cutoff > manifest_origin and evidence_role != "descriptive_contaminated_probe":
        raise OursError(
            "post-origin base pretraining is permitted only for an explicitly ineligible "
            "descriptive contamination probe"
        )
    if manifest["output_statistic"] != "mean":
        raise OursError(
            "aggregate trajectory output_statistic must be 'mean'; a median is not an expected-"
            "demand head offset"
        )
    if manifest["input_roles"] != ["date", "grouping", "target_history"]:
        raise OursError(
            "aggregate trajectory input_roles must be exactly date, grouping, target_history"
        )
    if manifest["provider_execution"] != "offline_before_projection":
        raise OursError("aggregate trajectory provider_execution must be offline_before_projection")
    if (
        manifest["price_inputs"] != []
        or manifest["provider_inference_post_origin_inputs"] is not False
        or manifest["price_invariant"] is not True
    ):
        raise OursError(
            "aggregate trajectory must declare no price inputs, no post-origin runtime/provider "
            "context inputs, and exact price invariance; disclosed base-pretraining contamination "
            "is tracked separately"
        )
    return manifest


def _read_aggregate_trajectory_manifest(
    artifact: str | pathlib.Path,
    *,
    origin: str,
    horizon: int,
) -> tuple[pathlib.Path, dict[str, object]]:
    """Authenticate the strict, price-free, origin-bounded offline artifact envelope."""
    root = pathlib.Path(artifact)
    manifest_path = root / "manifest.json"
    values_path = root / "values"
    if not root.is_dir() or not manifest_path.is_file() or not values_path.is_dir():
        raise OursError(
            "aggregate trajectory must be a directory containing manifest.json and values/"
        )
    import json as _json

    try:
        manifest = _json.loads(manifest_path.read_text())
    except (OSError, UnicodeError, _json.JSONDecodeError) as exc:
        raise OursError(f"aggregate trajectory manifest is unreadable JSON: {exc}") from exc
    manifest = _validate_aggregate_trajectory_manifest_claims(
        manifest, origin=origin, horizon=horizon
    )
    actual_values_hash = _spark_parquet_digest(values_path)
    if actual_values_hash != manifest["values_parquet_sha256"]:
        raise OursError(
            "aggregate trajectory values hash does not match manifest: "
            f"expected {manifest['values_parquet_sha256']}, got {actual_values_hash}"
        )
    return values_path, manifest


def _validated_aggregate_trajectory_values(
    values_path: pathlib.Path,
    manifest: dict[str, object],
) -> tuple[dict[str, dict[dt.date, tuple[float, float]]], int]:
    """Read the sealed Parquet values once and enforce the provider-neutral tensor contract."""
    import pyarrow as _arrow
    import pyarrow.dataset as _arrow_dataset

    try:
        table = _arrow_dataset.dataset(str(values_path), format="parquet").to_table()
    except (OSError, ValueError, TypeError, _arrow.ArrowException) as exc:
        raise OursError(f"aggregate trajectory values are unreadable Parquet: {exc}") from exc
    if (
        len(table.column_names) != len(_AGGREGATE_TRAJECTORY_VALUE_FIELDS)
        or set(table.column_names) != _AGGREGATE_TRAJECTORY_VALUE_FIELDS
    ):
        raise OursError(
            "aggregate trajectory values columns differ from the exact schema: "
            f"expected={sorted(_AGGREGATE_TRAJECTORY_VALUE_FIELDS)}, "
            f"got={table.column_names}"
        )
    origin_day = dt.date.fromisoformat(manifest["origin"])
    horizon = manifest["horizon"]
    expected_dates = tuple(origin_day + dt.timedelta(days=step + 1) for step in range(horizon))
    by_ancestor: dict[str, dict[dt.date, tuple[float, float]]] = {}
    available_cells = 0
    import json as _json

    for row in table.to_pylist():
        ancestor = row["ancestor_id"]
        raw_day = row["date"]
        available = row["available"]
        if type(raw_day) is dt.date:
            day = raw_day
        elif type(raw_day) is str:
            day = _canonical_iso_date(raw_day, "values date")
        else:
            day = None
        if (
            type(ancestor) is not str
            or not ancestor.strip()
            or day is None
            or type(available) is not bool
        ):
            raise OursError("aggregate trajectory row has an invalid ancestor/date/available key")
        try:
            ancestor_grouping = _json.loads(ancestor)
        except _json.JSONDecodeError as exc:
            raise OursError("aggregate trajectory ancestor_id is not canonical JSON") from exc
        if (
            type(ancestor_grouping) is not list
            or len(ancestor_grouping) != len(manifest["grouping_cols"])
            or any(type(pair) is not list or len(pair) != 2 for pair in ancestor_grouping)
            or [pair[0] for pair in ancestor_grouping] != manifest["grouping_cols"]
            or aggregate_trajectory_ancestor_id(ancestor_grouping) != ancestor
        ):
            raise OursError(
                "aggregate trajectory ancestor_id does not match its ordered grouping_cols"
            )
        bucket = by_ancestor.setdefault(ancestor, {})
        if day in bucket:
            raise OursError(f"aggregate trajectory repeats ancestor/date {(ancestor, day)!r}")
        if not available:
            if any(row[name] is not None for name in ("provider", "baseline", "scale")):
                raise OursError(
                    "aggregate trajectory unavailable rows must carry null provider/baseline/scale"
                )
            bucket[day] = (0.0, 0.0)
            continue
        numbers = []
        for name in ("provider", "baseline", "scale"):
            value = row[name]
            if isinstance(value, bool):
                raise OursError(f"aggregate trajectory {name} is boolean")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise OursError(f"aggregate trajectory {name} is not numeric") from exc
            if not math.isfinite(number):
                raise OursError(f"aggregate trajectory {name} is non-finite")
            numbers.append(number)
        provider, baseline, scale = numbers
        if provider < 0.0 or baseline < 0.0 or scale <= 0.0:
            raise OursError(
                "available aggregate provider/baseline must be nonnegative and scale positive"
            )
        uplift = (provider - baseline) / scale
        if not math.isfinite(uplift):
            raise OursError("aggregate trajectory dimensionless uplift is non-finite")
        bucket[day] = (uplift, 1.0)
        available_cells += 1
    if len(by_ancestor) != manifest["n_ancestors"]:
        raise OursError(
            "aggregate trajectory ancestor cardinality differs from its manifest: "
            f"expected {manifest['n_ancestors']}, got {len(by_ancestor)}"
        )
    expected_ancestors = _m5_l9_ancestor_ids()
    if set(by_ancestor) != expected_ancestors:
        missing = sorted(expected_ancestors.difference(by_ancestor))
        extra = sorted(set(by_ancestor).difference(expected_ancestors))
        raise OursError(
            "aggregate trajectory ancestors differ from the exact M5 L9 store/dept grid: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    for ancestor, values in by_ancestor.items():
        if set(values) != set(expected_dates):
            raise OursError(
                f"aggregate trajectory dates for ancestor {ancestor!r} do not exactly cover "
                f"origin+1..origin+{horizon}"
            )
    return by_ancestor, available_cells


def validate_aggregate_trajectory_artifact(
    path: str | pathlib.Path,
    *,
    origin: str,
    horizon: int,
) -> dict[str, object]:
    """Authenticate one offline trajectory artifact without executing its provider or a model."""
    with _aggregate_trajectory_input_snapshot(path) as snapshot:
        copied, artifact_sha256 = snapshot
        assert copied is not None and artifact_sha256 is not None
        values_path, manifest = _read_aggregate_trajectory_manifest(
            copied, origin=origin, horizon=horizon
        )
        _, available_cells = _validated_aggregate_trajectory_values(values_path, manifest)
    return {
        "schema": manifest["schema"],
        "artifact_id": manifest["artifact_id"],
        "artifact_sha256": artifact_sha256,
        "origin": manifest["origin"],
        "horizon": manifest["horizon"],
        "grouping_cols": list(manifest["grouping_cols"]),
        "n_ancestors": manifest["n_ancestors"],
        "available_cells": available_cells,
        "provider_identity": manifest["provider_identity"],
        "provider_base_artifact_sha256": manifest["provider_base_artifact_sha256"],
        "provider_base_pretraining_cutoff": manifest["provider_base_pretraining_cutoff"],
        "provider_adaptation_identity": manifest["provider_adaptation_identity"],
        "provider_adaptation_artifact_sha256": manifest["provider_adaptation_artifact_sha256"],
        "provider_adaptation_cutoff": manifest["provider_adaptation_cutoff"],
        "provider_adapted": manifest["provider_adaptation_identity"] is not None,
        "provider_context_cutoff": manifest["provider_context_cutoff"],
        "baseline_identity": manifest["baseline_identity"],
        "baseline_artifact_sha256": manifest["baseline_artifact_sha256"],
        "scale_identity": manifest["scale_identity"],
        "scale_artifact_sha256": manifest["scale_artifact_sha256"],
        "values_parquet_sha256": manifest["values_parquet_sha256"],
        "price_invariant": True,
        "evidence_role": manifest["evidence_role"],
        "claim_eligible": manifest["claim_eligible"],
    }


def _origin_bounded_series_levels(
    panel: Panel,
    spark: SparkSession,
    column: str,
    origin: str,
    series_ids: list[str],
) -> dict[str, str]:
    """Read one static grouping using rows at or before origin only."""
    raw = spark.read.parquet(panel.path)
    dated = raw.withColumn("_trajectory_date", F.expr(f"try_cast(`{panel.time_col}` as date)"))
    through_origin = dated.filter(F.col("_trajectory_date") <= F.lit(origin))
    wanted = set(series_ids)
    wanted_frame = spark.createDataFrame(
        [(series_id,) for series_id in series_ids], "series_id string"
    )
    rows = (
        through_origin.select(
            F.col(panel.series_col).cast("string").alias("series_id"),
            F.col(column).cast("string").alias("ancestor_id"),
        )
        .join(F.broadcast(wanted_frame), "series_id", "inner")
        .groupBy("series_id")
        .agg(
            F.countDistinct("ancestor_id").alias("n_levels"),
            F.first("ancestor_id", ignorenulls=True).alias("ancestor_id"),
        )
        .collect()
    )
    result = {}
    for row in rows:
        if row["n_levels"] != 1 or row["ancestor_id"] is None or not row["ancestor_id"].strip():
            raise OursError(
                f"aggregate trajectory grouping {column!r} is missing or changes through origin "
                f"for series {row['series_id']!r}"
            )
        result[row["series_id"]] = row["ancestor_id"]
    missing = sorted(wanted.difference(result))
    if missing:
        raise OursError(
            f"aggregate trajectory grouping {column!r} is absent through origin for "
            f"series {missing[:5]}"
        )
    return result


def _origin_bounded_composite_levels(
    panel: Panel,
    spark: SparkSession,
    columns: list[str],
    origin: str,
    series_ids: list[str],
) -> dict[str, str]:
    """Build canonical ordered composite ancestor IDs without reading post-origin rows."""
    raw = spark.read.parquet(panel.path)
    missing_columns = [column for column in columns if column not in raw.columns]
    if missing_columns:
        raise OursError(
            f"aggregate trajectory grouping columns are absent from the panel: {missing_columns}"
        )
    dated = raw.withColumn("_trajectory_date", F.expr(f"try_cast(`{panel.time_col}` as date)"))
    through_origin = dated.filter(F.col("_trajectory_date") <= F.lit(origin))
    wanted = set(series_ids)
    wanted_frame = spark.createDataFrame(
        [(series_id,) for series_id in series_ids], "series_id string"
    )
    level_aliases = [f"_trajectory_level_{index}" for index in range(len(columns))]
    selected = through_origin.select(
        F.col(panel.series_col).cast("string").alias("series_id"),
        *[
            F.col(column).cast("string").alias(alias)
            for column, alias in zip(columns, level_aliases, strict=True)
        ],
    )
    aggregates = []
    for alias in level_aliases:
        aggregates.extend(
            (
                F.countDistinct(alias).alias(f"{alias}_count"),
                F.first(alias, ignorenulls=True).alias(alias),
                F.sum(F.col(alias).isNull().cast("long")).alias(f"{alias}_nulls"),
            )
        )
    rows = (
        selected.join(F.broadcast(wanted_frame), "series_id", "inner")
        .groupBy("series_id")
        .agg(*aggregates)
        .collect()
    )
    result = {}
    for row in rows:
        levels = []
        for column, alias in zip(columns, level_aliases, strict=True):
            level = row[alias]
            if (
                row[f"{alias}_count"] != 1
                or row[f"{alias}_nulls"] != 0
                or level is None
                or not level.strip()
            ):
                raise OursError(
                    f"aggregate trajectory grouping column {column!r} is missing or changes "
                    f"through origin for series {row['series_id']!r}"
                )
            levels.append(level)
        result[row["series_id"]] = aggregate_trajectory_ancestor_id(
            tuple(zip(columns, levels, strict=True))
        )
    missing = sorted(wanted.difference(result))
    if missing:
        raise OursError(
            f"aggregate trajectory grouping is absent through origin for series {missing[:5]}"
        )
    return result


def _aggregate_trajectory_tensor(
    artifact: str | pathlib.Path,
    panel: Panel,
    spark: SparkSession,
    *,
    origin: str,
    horizon: int,
    series_ids: list[str],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Derive and broadcast ``[mask*(A-B)/scale, mask]`` from a sealed ancestor artifact."""
    values_path, manifest = _read_aggregate_trajectory_manifest(
        artifact, origin=origin, horizon=horizon
    )
    by_ancestor, _ = _validated_aggregate_trajectory_values(values_path, manifest)
    origin_day = dt.date.fromisoformat(origin)
    expected_dates = tuple(origin_day + dt.timedelta(days=step + 1) for step in range(horizon))
    grouping = _origin_bounded_composite_levels(
        panel, spark, manifest["grouping_cols"], origin, series_ids
    )
    unknown = sorted(set(grouping.values()).difference(by_ancestor))
    if unknown:
        raise OursError(f"aggregate trajectory has no channel for panel ancestor(s) {unknown[:5]}")
    tensor = torch.empty(len(series_ids), horizon, 2, dtype=dtype)
    for row, series_id in enumerate(series_ids):
        values = by_ancestor[grouping[series_id]]
        for step, day in enumerate(expected_dates):
            tensor[row, step] = torch.tensor(values[day], dtype=dtype)
    if not bool(torch.isfinite(tensor).all()):
        raise OursError("aggregate trajectory overflows the model input dtype")
    return tensor, manifest


@contextlib.contextmanager
def _aggregate_trajectory_input_snapshot(source: str | pathlib.Path | None):
    """Copy the whole manifest/value tree and bind the consumed bytes against an ABA race."""
    if source is None:
        yield None, None
        return
    source = pathlib.Path(source)
    if not source.is_dir():
        raise OursError("aggregate trajectory input must be a directory")
    owner = pathlib.Path(tempfile.mkdtemp(prefix="forecast-aggregate-trajectory-"))
    root = owner / "snapshot"
    root.mkdir()
    try:
        before = directory_digest(source)
        copied = root / "artifact"
        _copy_replay_input(source, copied)
        after = directory_digest(source)
        copied_hash = directory_digest(copied)
        if before != after or copied_hash != before:
            raise OursError("aggregate trajectory changed while its immutable snapshot was copied")
        _set_private_snapshot_read_only(root, True)
        snapshot_state = _private_snapshot_state(root, include_parent=True)
        snapshot = _PrivateAggregateTrajectorySnapshot(
            artifact=copied,
            source_digest=before,
            root=root,
            state=snapshot_state,
        )
        try:
            yield snapshot
        finally:
            snapshot._assert_unchanged()
    finally:
        _remove_owned_private_namespace(owner)


def _validate_replay_paths(
    checkpoint: str | pathlib.Path,
    panel: Panel,
    out_dir: str | pathlib.Path,
    paths_out_dir: str | pathlib.Path | None,
    text_embeddings: str | pathlib.Path | None,
    *,
    scratch: str | pathlib.Path | None = None,
    aggregate_trajectory: str | pathlib.Path | None = None,
) -> None:
    """Refuse input/output aliases and enforce one atomic replay artifact tree.

    When sample paths are requested they must live at ``out_dir/_paths``.  Both Parquet
    datasets and the completion record are then published by one directory rename; two sibling
    directories cannot be committed atomically as a pair.
    """
    inputs = {
        "checkpoint": pathlib.Path(checkpoint),
        "panel": pathlib.Path(panel.path),
    }
    if panel.plan is not None:
        inputs["plan"] = pathlib.Path(panel.plan.path)
    if text_embeddings is not None:
        inputs["text_embeddings"] = pathlib.Path(text_embeddings)
    if aggregate_trajectory is not None:
        inputs["aggregate_trajectory"] = pathlib.Path(aggregate_trajectory)
    point_output = pathlib.Path(out_dir)
    outputs = {"point output": point_output}
    if paths_out_dir is not None:
        path_output = pathlib.Path(paths_out_dir)
        required = point_output / "_paths"
        if path_output.resolve(strict=False) != required.resolve(strict=False):
            raise OursError(
                f"checkpoint replay paths_out_dir must be exactly {required.resolve(strict=False)}; "
                "point forecasts, paths, and their completion record require one atomic tree"
            )
    if scratch is not None:
        outputs["scratch"] = pathlib.Path(scratch)

    output_items = list(outputs.items())
    for left_name, left in output_items:
        for input_name, input_path in inputs.items():
            if _paths_overlap(left, input_path):
                raise OursError(
                    f"replay {left_name} {left.resolve()} overlaps {input_name} input "
                    f"{input_path.resolve()}; overwrite could destroy or mutate evidence"
                )
        if left.exists() and (not left.is_dir() or any(left.iterdir())):
            raise OursError(
                f"replay {left_name} {left.resolve()} already contains an artifact; replay "
                "outputs are immutable, so choose a new destination"
            )


def _replay_input_digests(
    checkpoint: str | pathlib.Path,
    panel: Panel,
    text_embeddings: str | pathlib.Path | None,
) -> dict[str, str]:
    """Hash every mutable artifact replay consumes, before the first read."""
    # Hash the complete input trees, not merely lower-case ``*.parquet`` paths. Spark readers
    # can consume upper-case and extensionless data files; a suffix filter would let those bytes
    # change during assembly without changing the replay guard.
    digests = {
        "checkpoint_sha256": _file_sha256(pathlib.Path(checkpoint)),
        "panel_parquet_sha256": directory_digest(panel.path),
    }
    if panel.plan is not None:
        digests["plan_source_parquet_sha256"] = directory_digest(panel.plan.path)
    if text_embeddings is not None:
        digests["text_embeddings_parquet_sha256"] = directory_digest(text_embeddings)
    return digests


def _assert_replay_inputs_unchanged(
    start: dict[str, str],
    checkpoint: str | pathlib.Path,
    panel: Panel,
    text_embeddings: str | pathlib.Path | None,
) -> None:
    """Reject a mixed-version forecast before publishing its staged artifacts."""
    end = _replay_input_digests(checkpoint, panel, text_embeddings)
    changed = {
        name: (start.get(name), end.get(name))
        for name in set(start) | set(end)
        if start.get(name) != end.get(name)
    }
    if changed:
        raise OursError(
            f"replay input changed while it was being consumed; refusing publication: {changed}"
        )


def _copy_replay_input(source: str | pathlib.Path, destination: pathlib.Path) -> None:
    """Copy one replay input byte-for-byte into a private immutable working tree."""
    source = pathlib.Path(source)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    elif source.is_dir():
        shutil.copytree(source, destination)
    else:
        raise OursError(f"replay input {source} is neither a readable file nor directory")


@contextlib.contextmanager
def _replay_input_snapshot(
    checkpoint: str | pathlib.Path,
    panel: Panel,
    text_embeddings: str | pathlib.Path | None,
):
    """Yield private copies whose bytes exactly match the recorded source digests.

    Start/end hashing alone has an ABA hole: a source can be A at both hashes while Spark reads B
    between them.  Replay consumes only these copies.  A mutation during copying either leaves a
    snapshot whose digest differs from the recorded source or is restored byte-for-byte to A, in
    which case consuming the snapshot is still consuming A.
    """
    owner = pathlib.Path(tempfile.mkdtemp(prefix="forecast-replay-inputs-"))
    root = owner / "snapshot"
    root.mkdir()
    try:
        source_digests = _replay_input_digests(checkpoint, panel, text_embeddings)
        checkpoint_copy = root / "checkpoint.pt"
        panel_source = pathlib.Path(panel.path)
        panel_copy = (
            root / "panel_input" / panel_source.name if panel_source.is_file() else root / "panel"
        )
        _copy_replay_input(checkpoint, checkpoint_copy)
        _copy_replay_input(panel.path, panel_copy)

        plan_copy = None
        copied_plan = panel.plan
        if panel.plan is not None:
            plan_source = pathlib.Path(panel.plan.path)
            plan_copy = (
                root / "plan_input" / plan_source.name if plan_source.is_file() else root / "plan"
            )
            _copy_replay_input(panel.plan.path, plan_copy)
            copied_plan = replace(panel.plan, path=str(plan_copy))
        copied_panel = replace(panel, path=str(panel_copy), plan=copied_plan)

        text_copy = None
        if text_embeddings is not None:
            text_source = pathlib.Path(text_embeddings)
            text_copy = (
                root / "text_input" / text_source.name if text_source.is_file() else root / "text"
            )
            _copy_replay_input(text_embeddings, text_copy)

        # First reject a persistent source change, then independently bind the bytes that will
        # actually be consumed.  The second comparison is what catches A -> B -> A during copy.
        _assert_replay_inputs_unchanged(source_digests, checkpoint, panel, text_embeddings)
        snapshot_digests = _replay_input_digests(checkpoint_copy, copied_panel, text_copy)
        mismatched = {
            name: (source_digests.get(name), snapshot_digests.get(name))
            for name in set(source_digests) | set(snapshot_digests)
            if source_digests.get(name) != snapshot_digests.get(name)
        }
        if mismatched:
            raise OursError(
                "replay input changed while its immutable snapshot was copied; refusing "
                f"mixed-version consumption: {mismatched}"
            )
        _set_private_snapshot_read_only(root, True)
        snapshot_state = _private_snapshot_state(root, include_parent=True)
        snapshot = _PrivateReplayInputSnapshot(
            checkpoint=checkpoint_copy,
            panel=copied_panel,
            text_embeddings=text_copy,
            source_digests=source_digests,
            root=root,
            state=snapshot_state,
        )
        try:
            yield snapshot
        finally:
            snapshot._assert_unchanged()
    finally:
        _remove_owned_private_namespace(owner)


def _staging_directory(destination: str | pathlib.Path) -> pathlib.Path:
    """Create a unique sibling directory so final publication is a local rename."""
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )


def _replay_scratch_directory(origin: str) -> pathlib.Path:
    """One private assembly namespace per serve, even for equal origins and sibling outputs."""
    return _private_scratch_directory("predict", origin)


def _publish_directory(stage: pathlib.Path, destination: str | pathlib.Path) -> None:
    """Publish one immutable staged directory with a same-filesystem rename.

    Replacing a non-empty directory requires first moving the old tree aside, which creates a
    crash window with no authoritative artifact. Replay outputs are evidence, so they are
    immutable: choose a new destination for a new serve. An existing empty directory is safe to
    replace atomically and accommodates callers that use ``tempfile.mkdtemp()``.
    """
    destination = pathlib.Path(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise OursError(
            f"replay output {destination} already contains an artifact; replay outputs are "
            "immutable, so choose a new destination"
        )
    os.replace(stage, destination)


def _write_json_atomic(path: pathlib.Path, record: dict[str, object]) -> None:
    """Publish a JSON completion record with one same-filesystem rename."""
    import json

    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _checkpoint_point_forecast(
    model: CustomModel,
    inputs: dict,
    run: dict,
    panel: Panel,
    spark: SparkSession,
    series_ids: list[str],
    *,
    method: str,
    factor_draws: int | None,
    seed: int,
    rng_domain_scheme: str,
) -> torch.Tensor:
    """Replay the fit-time point batching contract, including recorded hash shards.

    A finite-MC factor mean is deterministic only after its batching map is fixed. Sharded fit
    calls ``point_forecast`` once per hash shard with the same domain seed; serving all series in
    one call assigns different random coordinates to every group after the first. Reapplying the
    checkpoint's disjoint predicates preserves exact replay while the surrounding assembly is
    still (explicitly) full-panel and unbounded.
    """
    sharded = run.get("sharded")
    if sharded is None:
        return point_forecast(
            model,
            inputs,
            method=method,
            factor_draws=factor_draws,
            seed=seed,
            rng_domain_scheme=rng_domain_scheme,
        )
    if not isinstance(sharded, dict):
        raise OursError("checkpoint sharded replay contract must be a mapping")
    predicates = sharded.get("predicates")
    if (
        not isinstance(predicates, list)
        or not predicates
        or any(
            predicate is not None and (not isinstance(predicate, str) or not predicate.strip())
            for predicate in predicates
        )
    ):
        raise OursError("checkpoint sharded replay predicates are missing or malformed")

    position = {sid: row for row, sid in enumerate(series_ids)}
    assigned: set[int] = set()
    result = None
    source = spark.read.parquet(panel.path)
    for predicate in predicates:
        selected = source if predicate is None else source.filter(F.expr(predicate))
        member_ids = [
            str(row[panel.series_col])
            for row in selected.select(panel.series_col).distinct().collect()
        ]
        keep = sorted(position[sid] for sid in member_ids if sid in position)
        overlap = assigned.intersection(keep)
        if overlap:
            raise OursError(
                f"checkpoint shard predicates assign forecast rows more than once: "
                f"{sorted(overlap)[:5]}"
            )
        if not keep:
            continue
        assigned.update(keep)
        index = torch.tensor(keep, dtype=torch.long)
        local = point_forecast(
            model,
            _subset(inputs, index),
            method=method,
            factor_draws=factor_draws,
            seed=seed,
            rng_domain_scheme=rng_domain_scheme,
        )
        if result is None:
            result = local.new_empty((len(series_ids), local.shape[1]))
        result[index] = local
    missing = sorted(set(range(len(series_ids))).difference(assigned))
    if result is None or missing:
        raise OursError(
            f"checkpoint shard predicates do not cover replay forecast rows: {missing[:5]}"
        )
    return result


def _checkpoint_artifact_device(model: CustomModel, run: dict) -> str:
    """Require the portable CPU numeric boundary used by persisted claim artifacts."""
    artifact_device = run.get("artifact_device", "cpu")
    replay_device = str(next(model.parameters()).device)
    if artifact_device != "cpu" or replay_device != "cpu":
        raise OursError(
            "exact checkpoint replay requires CPU claim artifacts and a CPU-loaded model; "
            f"checkpoint artifact_device={artifact_device!r}, replay_device={replay_device!r}"
        )
    return replay_device


def _projection_calibration_data(
    model: CustomModel,
    record: dict,
    panel: Panel,
    spark: SparkSession,
    *,
    calibration_origin: str,
    aggregate_trajectory: str | pathlib.Path,
    maximum_label_date: str,
    scratch: pathlib.Path,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, object], str]:
    """Build one forward calibration episode under a frozen checkpoint namespace."""
    run = record["run"]
    horizon = run["horizon"]
    origin_day = _canonical_iso_date(calibration_origin, "calibration origin")
    label_ceiling = _canonical_iso_date(maximum_label_date, "maximum calibration label date")
    target_end = origin_day + dt.timedelta(days=horizon)
    if not origin_day < target_end <= label_ceiling:
        raise OursError("projection calibration horizon exceeds the explicitly sealed label cutoff")
    expected_panel_hash = run.get("panel_parquet_sha256")
    actual_panel_hash = directory_digest(panel.path, suffixes=(".parquet",))
    if expected_panel_hash != actual_panel_hash:
        raise OursError(
            "projection calibration panel bytes differ from the frozen R6 training panel"
        )
    mismatched_roles = _panel_role_mismatches(panel, run.get("panel"))
    if mismatched_roles:
        raise OursError(f"projection calibration panel roles differ from R6: {mismatched_roles}")
    stored_series_ids = run.get("series_ids")
    tensors = run.get("tensors")
    if (
        not isinstance(stored_series_ids, list)
        or not stored_series_ids
        or not isinstance(tensors, dict)
    ):
        raise OursError(
            "projection calibration requires the non-sharded R6 checkpoint namespace tensors"
        )
    n_stored = len(stored_series_ids)
    required_tensors = ("attributes", "group_index", "store_index", "product_index")
    for name in required_tensors:
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor) or value.shape[0] != n_stored:
            raise OursError(f"R6 checkpoint calibration tensor {name!r} is absent or misaligned")
    category = run.get("category_col")
    category_of = (
        None
        if category is None
        else _origin_bounded_series_levels(
            panel, spark, category, calibration_origin, stored_series_ids
        )
    )
    assembly = _assemble(
        panel,
        spark,
        calibration_origin,
        horizon,
        run["encoder_days"],
        [tuple(event) for event in run["events"]],
        run["event_max_support"],
        scratch / "assembly",
        tuple(tuple(cycle) for cycle in run["cycles"]),
        category_of=category_of,
        event_windows=(
            None
            if run.get("event_windows") is None
            else {name: tuple(window) for name, window in run["event_windows"].items()}
        ),
    )
    if list(assembly.known.names) != list(run["known_names"]):
        raise OursError("projection calibration known-future layout differs from frozen R6")
    stored_position = {series_id: row for row, series_id in enumerate(stored_series_ids)}
    unknown = sorted(set(assembly.series_ids).difference(stored_position))
    if unknown:
        raise OursError(
            f"projection calibration contains identities absent from frozen R6: {unknown[:5]}"
        )
    source_rows = torch.tensor(
        [stored_position[series_id] for series_id in assembly.series_ids], dtype=torch.long
    )
    text = tensors.get("text")
    if text is not None and (not isinstance(text, torch.Tensor) or text.shape[0] != n_stored):
        raise OursError("R6 checkpoint calibration text tensor is misaligned")
    trajectory, manifest = _aggregate_trajectory_tensor(
        aggregate_trajectory,
        panel,
        spark,
        origin=calibration_origin,
        horizon=horizon,
        series_ids=assembly.series_ids,
        dtype=assembly.known.values.dtype,
    )

    target_path = demand_history.build(
        panel.path,
        target_end.isoformat(),
        horizon,
        spark,
        str(scratch / "target"),
        series_col=panel.series_col,
        time_col=panel.time_col,
        target_col=panel.target_col,
        exposure_col=panel.exposure_col,
        oos_col=panel.oos_col,
        status_prob_cols=panel.status_prob_cols,
        negative_target_policy=panel.negative_target_policy,
    )
    target_rows = {row["series_id"]: row for row in spark.read.parquet(str(target_path)).collect()}
    usable_local = [
        row for row, series_id in enumerate(assembly.series_ids) if series_id in target_rows
    ]
    if not usable_local:
        raise OursError("projection calibration has no series with both context and target")
    local = torch.tensor(usable_local, dtype=torch.long)
    source = source_rows[local]
    usable_ids = [assembly.series_ids[row] for row in usable_local]
    inputs = dict(
        history=assembly.history[local],
        observed=assembly.observed[local],
        censored=assembly.censored[local],
        attributes=tensors["attributes"][source],
        series_index=source,
        text=None if text is None else text[source],
        known_future=assembly.known.values[local],
        baseline_logit=assembly.baseline_logit[local],
        baseline_log_size=assembly.baseline_log_size[local],
        group_index=tensors["group_index"][source],
        store_index=tensors["store_index"][source],
        product_index=tensors["product_index"][source],
        n_draws=run.get("importance_draws", 1),
        commercial=None if assembly.commercial is None else assembly.commercial[local],
        commercial_mask=(
            None if assembly.commercial_mask is None else assembly.commercial_mask[local]
        ),
        event_distance=(
            None if assembly.event_distance is None else assembly.event_distance[local]
        ),
        event_present=(None if assembly.event_present is None else assembly.event_present[local]),
        aggregate_trajectory=trajectory[local],
    )
    target = torch.tensor(
        [target_rows[series_id]["history"] for series_id in usable_ids], dtype=torch.float32
    )
    raw_state = torch.tensor(
        [target_rows[series_id]["state"] for series_id in usable_ids], dtype=torch.long
    )
    target_state = _states_for_estimand(raw_state, panel.target_estimand)
    target_probs = None
    if panel.status_prob_cols is not None:
        target_probs = torch.stack(
            [
                torch.tensor(
                    [target_rows[series_id][column] for series_id in usable_ids],
                    dtype=torch.float32,
                )
                for column in ("p_exact", "p_lower_bound", "p_unobserved")
            ],
            dim=-1,
        )
        target_probs = _probs_for_estimand(target_probs, panel.target_estimand)
    return inputs, target, target_state, target_probs, manifest, target_end.isoformat()


def fit_aggregate_trajectory_projection(
    r6_checkpoint: str | pathlib.Path,
    panel: Panel,
    spark: SparkSession,
    calibration_episodes: tuple[AggregateTrajectoryCalibrationEpisode, ...],
    evaluation_origin: str,
    checkpoint_out: str | pathlib.Path,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_microbatch_size: int,
    gradient_clip_norm: float,
    seed: int,
    device: str,
) -> pathlib.Path:
    """Jointly fit the sole projection on two ordered forward episodes from dormant R6.

    Both hybrid arms must call this function with the same ``r6_checkpoint`` bytes. The provider
    artifact may differ, but the model graph, initial base weights, optimizer route, and sole
    trainable parameter cannot.
    """
    controls = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_microbatch_size": gradient_microbatch_size,
        "gradient_clip_norm": gradient_clip_norm,
        "seed": seed,
        "device": device,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "evaluation_schedule": "final_only",
    }
    if not _json_values_equal_exact(controls, AGGREGATE_TRAJECTORY_PROJECTION_CONTROLS):
        raise OursError(
            "projection controls must exactly reuse the registered R6 runner controls: "
            f"expected {AGGREGATE_TRAJECTORY_PROJECTION_CONTROLS}, got {controls}"
        )
    if (
        type(calibration_episodes) is not tuple
        or len(calibration_episodes) != 2
        or any(
            type(episode) is not AggregateTrajectoryCalibrationEpisode
            for episode in calibration_episodes
        )
    ):
        raise OursError(
            "projection calibration_episodes must be exactly two ordered current-schema episodes"
        )
    evaluation_day = _canonical_iso_date(evaluation_origin, "projection evaluation origin")
    episode_days = [
        _canonical_iso_date(episode.origin, f"episode {index} origin")
        for index, episode in enumerate(calibration_episodes)
    ]
    if (
        tuple(episode.origin for episode in calibration_episodes)
        != AGGREGATE_TRAJECTORY_CALIBRATION_ORIGINS
        or evaluation_origin != AGGREGATE_TRAJECTORY_EVALUATION_ORIGIN
    ):
        raise OursError(
            "projection calibration must use the registered forward protocol "
            f"{AGGREGATE_TRAJECTORY_CALIBRATION_ORIGINS} -> "
            f"{AGGREGATE_TRAJECTORY_EVALUATION_ORIGIN}"
        )
    output = pathlib.Path(checkpoint_out)
    if output.exists():
        raise OursError("projection checkpoint output is immutable; choose a new path")
    sources = [("R6 checkpoint", pathlib.Path(r6_checkpoint)), ("panel", pathlib.Path(panel.path))]
    sources.extend(
        (f"episode {index} aggregate trajectory", pathlib.Path(episode.artifact))
        for index, episode in enumerate(calibration_episodes)
    )
    for label, source in sources:
        if _paths_overlap(output, source):
            raise OursError(f"projection checkpoint output overlaps {label} input")

    with contextlib.ExitStack() as stack:
        source_snapshot = stack.enter_context(_replay_input_snapshot(r6_checkpoint, panel, None))
        copied_checkpoint, copied_panel, _, source_digests = source_snapshot
        trajectory_snapshots = [
            stack.enter_context(_aggregate_trajectory_input_snapshot(episode.artifact))
            for episode in calibration_episodes
        ]
        for copied_trajectory, trajectory_digest in trajectory_snapshots:
            assert copied_trajectory is not None and trajectory_digest is not None
        model, record = load_checkpoint(copied_checkpoint)
        horizon = record["run"]["horizon"]
        if horizon != AGGREGATE_TRAJECTORY_HORIZON:
            raise OursError(
                "projection calibration requires the registered 28-day M5 horizon, got "
                f"{horizon!r}"
            )
        projection_record = record["run"].get("aggregate_trajectory_projection")
        if (
            not isinstance(projection_record, dict)
            or projection_record.get("status") != "dormant_r6"
        ):
            raise OursError(
                "projection fitting must start from one dormant exact-zero R6 checkpoint, "
                "not another calibrated hybrid"
            )
        source_origin = _canonical_iso_date(record["run"].get("origin"), "source R6 origin")
        if episode_days[0] != source_origin:
            raise OursError("first projection episode must start exactly at the dormant R6 origin")
        target_ends = [day + dt.timedelta(days=horizon) for day in episode_days]
        if episode_days[1] != target_ends[0] or target_ends[1] != evaluation_day:
            raise OursError(
                "projection episodes must be contiguous O0->O1 and finish at evaluation_origin"
            )

        episode_data = []
        episode_records = []
        for index, (episode, snapshot) in enumerate(
            zip(calibration_episodes, trajectory_snapshots, strict=True)
        ):
            copied_trajectory, trajectory_digest = snapshot
            scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"forecast-projection-fit-{index}-"))
            try:
                data = _projection_calibration_data(
                    model,
                    record,
                    copied_panel,
                    spark,
                    calibration_origin=episode.origin,
                    aggregate_trajectory=copied_trajectory,
                    maximum_label_date=evaluation_origin,
                    scratch=scratch,
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            inputs, target, target_state, target_probs, manifest, target_end = data
            episode_data.append((inputs, target, target_state, target_probs))
            episode_records.append(
                {
                    "origin": episode.origin,
                    "target_end": target_end,
                    "artifact_sha256": trajectory_digest,
                    "manifest": manifest,
                }
            )
        first_manifest = episode_records[0]["manifest"]
        drift = {
            field: (first_manifest[field], episode_records[1]["manifest"][field])
            for field in _AGGREGATE_TRAJECTORY_STABLE_MANIFEST_FIELDS
            if not _json_values_equal_exact(
                first_manifest[field], episode_records[1]["manifest"][field]
            )
        }
        if drift:
            raise OursError(
                f"projection calibration episode manifests change fixed semantics: {drift}"
            )
        if (first_manifest["provider_adaptation_identity"] is None) != (
            episode_records[1]["manifest"]["provider_adaptation_identity"] is None
        ):
            raise OursError(
                "projection calibration cannot switch between frozen and adapted providers "
                "across episodes"
            )

        pieces = []
        next_factor_block = 0
        targets, states, probabilities = [], [], []
        probabilities_present = [data[3] is not None for data in episode_data]
        if len(set(probabilities_present)) != 1:
            raise OursError("projection calibration episodes disagree on soft-state semantics")
        for inputs, target, target_state, target_probs in episode_data:
            piece = dict(inputs)
            piece["group_index"], next_factor_block = _offset_dense_blocks(
                piece["group_index"], next_factor_block
            )
            pieces.append(piece)
            targets.append(target)
            states.append(target_state)
            if target_probs is not None:
                probabilities.append(target_probs)
        inputs = {
            name: (
                torch.cat([piece[name] for piece in pieces])
                if isinstance(pieces[0][name], torch.Tensor)
                else pieces[0][name]
            )
            for name in pieces[0]
        }
        target = torch.cat(targets)
        target_state = torch.cat(states)
        target_probs = torch.cat(probabilities) if probabilities else None
        if not bool((inputs["aggregate_trajectory"][..., 1] != 0.0).any()):
            raise OursError(
                "projection calibration artifacts are all-missing and supply no trainable "
                "trajectory evidence"
            )

        frozen_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name != AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER
        }
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        projection = model.aggregate_trajectory_projection
        assert projection is not None
        projection.weight.requires_grad_(True)
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if trainable != [AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER]:
            raise OursError(
                f"projection-only fit exposed unexpected trainable parameters {trainable}"
            )
        model.eval()
        optimizer = torch.optim.AdamW(
            [model.aggregate_trajectory_projection.weight], lr=learning_rate, weight_decay=0.0
        )
        torch.manual_seed(seed)
        batch_generator = torch.Generator().manual_seed(seed)
        from forecast.models.mft.heads import (
            informative_mask,
            soft_informative_weight,
        )

        informative_weight = (
            informative_mask(target, target_state).float()
            if target_probs is None
            else soft_informative_weight(target, target_probs)
        )
        informative = informative_weight.sum().item()
        if informative <= 0.0:
            raise OursError("projection calibration targets have no informative cells")
        # Rows with no label mass contribute exactly zero but can otherwise form an all-empty
        # physical batch, which ``CustomModel.loss`` correctly refuses. Remove only those rows;
        # no likelihood term or factor-block evidence is lost.
        all_rows = torch.nonzero(
            informative_weight.reshape(len(target), -1).sum(dim=-1) > 0.0, as_tuple=False
        ).flatten()
        for _ in range(epochs):
            batches = factor_block_batches(
                all_rows, inputs["group_index"], batch_size, batch_generator
            )
            normalization = informative / len(batches)
            for batch in batches:
                optimizer.zero_grad()
                logical = _subset(inputs, batch)
                logical_target = target[batch]
                logical_state = target_state[batch]
                logical_probs = None if target_probs is None else target_probs[batch]
                relative = torch.arange(len(batch), dtype=torch.long)
                microbatches = _factor_block_microbatches(
                    relative, logical["group_index"], gradient_microbatch_size
                )
                backward_calls = 0
                for microbatch in microbatches:
                    local_inputs = _subset(logical, microbatch)
                    loss = model.loss(
                        logical_target[microbatch],
                        logical_state[microbatch],
                        normalization_count=normalization,
                        state_probs=(None if logical_probs is None else logical_probs[microbatch]),
                        **local_inputs,
                    )
                    if loss.requires_grad:
                        loss.backward()
                        backward_calls += 1
                if backward_calls:
                    nn.utils.clip_grad_norm_(
                        [model.aggregate_trajectory_projection.weight], gradient_clip_norm
                    )
                    optimizer.step()

        model.eval()
        for name, expected in frozen_state.items():
            actual = model.state_dict()[name].detach().cpu()
            if not torch.equal(actual, expected):
                raise OursError(f"projection-only fit mutated frozen R6 tensor {name!r}")
        run = dict(record["run"])
        calibration_record = {
            "evaluation_origin": evaluation_origin,
            "max_label_date": evaluation_origin,
            "panel_parquet_sha256": source_digests["panel_parquet_sha256"],
            "evidence_role": first_manifest["evidence_role"],
            "claim_eligible": first_manifest["claim_eligible"],
            "episodes": episode_records,
            **controls,
            "final_projection_sha256": _projection_weight_sha256(model),
        }
        run["aggregate_trajectory_projection"] = {
            "schema": AGGREGATE_TRAJECTORY_PROJECTION_SCHEMA,
            "mode": AGGREGATE_TRAJECTORY_MODE,
            "status": "calibrated",
            "source_r6_checkpoint_sha256": source_digests["checkpoint_sha256"],
            "trainable_parameters": [AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER],
            "base_parameters_frozen": True,
            "calibration": calibration_record,
        }
        source_snapshot._assert_unchanged()
        for trajectory_snapshot in trajectory_snapshots:
            trajectory_snapshot._assert_unchanged()
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
        try:
            save_checkpoint(model, staged, model_config=dict(record["model_config"]), run=run)
            try:
                os.link(staged, output)
            except FileExistsError as exc:
                raise OursError(
                    "projection checkpoint output appeared during fitting; refusing to "
                    "overwrite an immutable artifact"
                ) from exc
        finally:
            staged.unlink(missing_ok=True)
        return output


def continue_aggregate_trajectory_projection(
    projection_checkpoint: str | pathlib.Path,
    panel: Panel,
    spark: SparkSession,
    training_episode: AggregateTrajectoryCalibrationEpisode,
    validation_episode: AggregateTrajectoryCalibrationEpisode,
    evaluation_origin: str,
    checkpoint_out: str | pathlib.Path,
    *,
    source_checkpoint_sha256: str,
    max_additional_epochs: int,
    patience: int,
    learning_rate: float,
    batch_size: int,
    gradient_microbatch_size: int,
    gradient_clip_norm: float,
    seed: int,
    device: str,
    absolute_min_delta: float,
    relative_min_delta: float,
    quiet_epoch_logging: bool = True,
) -> pathlib.Path:
    """Continue only a calibrated R9/R10 projection with O0 train and O1 validation.

    The source calibrated projection is epoch zero. The source checkpoint contains no optimizer
    or scheduler state, so both are intentionally fresh. O2 is not accepted by this boundary and
    cannot influence stopping; the caller may replay it only after the best O0/O1 weights have
    been restored and this immutable checkpoint has been published.
    """
    controls = {
        "max_additional_epochs": max_additional_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_microbatch_size": gradient_microbatch_size,
        "gradient_clip_norm": gradient_clip_norm,
        "seed": seed,
        "device": device,
        "absolute_min_delta": absolute_min_delta,
        "relative_min_delta": relative_min_delta,
        "quiet_epoch_logging": quiet_epoch_logging,
    }
    if not _json_values_equal_exact(
        controls, AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_CONTROLS
    ):
        raise OursError(
            "projection continuation controls differ from the registered R9/R10 protocol: "
            f"expected {AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_CONTROLS}, got {controls}"
        )
    if not _sha256_is_canonical(source_checkpoint_sha256):
        raise OursError("source_checkpoint_sha256 must be a canonical lowercase SHA-256 digest")
    if (
        type(training_episode) is not AggregateTrajectoryCalibrationEpisode
        or type(validation_episode) is not AggregateTrajectoryCalibrationEpisode
    ):
        raise OursError(
            "projection continuation requires explicit training and validation episodes"
        )
    if (
        (training_episode.origin, validation_episode.origin)
        != AGGREGATE_TRAJECTORY_CALIBRATION_ORIGINS
        or evaluation_origin != AGGREGATE_TRAJECTORY_EVALUATION_ORIGIN
    ):
        raise OursError(
            "projection continuation must use registered O0 training, O1 validation and O2 "
            "final-evaluation origins"
        )
    evaluation_day = _canonical_iso_date(
        evaluation_origin, "projection continuation evaluation origin"
    )
    episode_days = (
        _canonical_iso_date(training_episode.origin, "projection training origin"),
        _canonical_iso_date(validation_episode.origin, "projection validation origin"),
    )
    output = pathlib.Path(checkpoint_out)
    if output.exists():
        raise OursError("projection continuation output is immutable; choose a new path")
    sources = {
        "source projection checkpoint": pathlib.Path(projection_checkpoint),
        "panel": pathlib.Path(panel.path),
        "O0 training trajectory": pathlib.Path(training_episode.artifact),
        "O1 validation trajectory": pathlib.Path(validation_episode.artifact),
    }
    for label, source in sources.items():
        if _paths_overlap(output, source):
            raise OursError(f"projection continuation output overlaps {label} input")

    with contextlib.ExitStack() as stack:
        source_snapshot = stack.enter_context(
            _replay_input_snapshot(projection_checkpoint, panel, None)
        )
        copied_checkpoint, copied_panel, _, source_digests = source_snapshot
        if source_digests["checkpoint_sha256"] != source_checkpoint_sha256:
            raise OursError(
                "projection continuation source hash differs from its preregistered identity: "
                f"expected={source_checkpoint_sha256}, "
                f"observed={source_digests['checkpoint_sha256']}"
            )
        episodes = (training_episode, validation_episode)
        trajectory_snapshots = [
            stack.enter_context(_aggregate_trajectory_input_snapshot(episode.artifact))
            for episode in episodes
        ]
        model, record = load_checkpoint(copied_checkpoint)
        run = record.get("run")
        projection_record = (
            run.get("aggregate_trajectory_projection") if isinstance(run, dict) else None
        )
        source_calibration = (
            projection_record.get("calibration") if isinstance(projection_record, dict) else None
        )
        if (
            not isinstance(projection_record, dict)
            or projection_record.get("status") != "calibrated"
            or not isinstance(source_calibration, dict)
            or "continuation" in source_calibration
        ):
            raise OursError(
                "projection continuation must start from an original calibrated R9/R10 "
                "checkpoint, not dormant or already-continued weights"
            )
        horizon = run.get("horizon")
        if horizon != AGGREGATE_TRAJECTORY_HORIZON:
            raise OursError("projection continuation requires the registered 28-day horizon")
        if (
            episode_days[0] + dt.timedelta(days=horizon) != episode_days[1]
            or episode_days[1] + dt.timedelta(days=horizon) != evaluation_day
        ):
            raise OursError("projection continuation episodes are not contiguous O0->O1->O2")

        episode_data = []
        episode_records = []
        for index, (episode, snapshot) in enumerate(
            zip(episodes, trajectory_snapshots, strict=True)
        ):
            copied_trajectory, trajectory_digest = snapshot
            assert copied_trajectory is not None and trajectory_digest is not None
            scratch = pathlib.Path(
                tempfile.mkdtemp(prefix=f"forecast-projection-continuation-{index}-")
            )
            try:
                data = _projection_calibration_data(
                    model,
                    record,
                    copied_panel,
                    spark,
                    calibration_origin=episode.origin,
                    aggregate_trajectory=copied_trajectory,
                    maximum_label_date=evaluation_origin,
                    scratch=scratch,
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            inputs, target, target_state, target_probs, manifest, target_end = data
            episode_data.append((inputs, target, target_state, target_probs))
            episode_records.append(
                {
                    "origin": episode.origin,
                    "target_end": target_end,
                    "artifact_sha256": trajectory_digest,
                    "manifest": manifest,
                }
            )
        if not _json_values_equal_exact(source_calibration.get("episodes"), episode_records):
            raise OursError(
                "projection continuation trajectories differ from the source checkpoint's "
                "authenticated O0/O1 calibration episodes"
            )

        train_inputs, train_target, train_state, train_probs = episode_data[0]
        val_inputs, val_target, val_state, val_probs = episode_data[1]
        if (
            (train_probs is None) != (val_probs is None)
            or not bool((train_inputs["aggregate_trajectory"][..., 1] != 0.0).any())
            or not bool((val_inputs["aggregate_trajectory"][..., 1] != 0.0).any())
        ):
            raise OursError(
                "projection continuation episodes disagree on state semantics or contain no "
                "available trajectory evidence"
            )

        projection = model.aggregate_trajectory_projection
        if projection is None:
            raise OursError("projection continuation source has no executable projection")
        frozen_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name != AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER
        }
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        projection.weight.requires_grad_(True)
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if trainable != [AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER]:
            raise OursError(
                f"projection continuation exposed unexpected trainable parameters {trainable}"
            )

        from forecast.models.mft.heads import (
            informative_mask,
            soft_informative_weight,
        )

        def informative_rows(target, state, probabilities):
            weight = (
                informative_mask(target, state).float()
                if probabilities is None
                else soft_informative_weight(target, probabilities)
            )
            mass = float(weight.sum())
            rows = torch.nonzero(
                weight.reshape(len(target), -1).sum(dim=-1) > 0.0,
                as_tuple=False,
            ).flatten()
            if mass <= 0.0 or len(rows) == 0:
                raise OursError("projection continuation episode has no informative cells")
            return rows, mass

        train_rows, train_mass = informative_rows(train_target, train_state, train_probs)
        val_rows, val_mass = informative_rows(val_target, val_state, val_probs)

        def validation_loss(epoch: int) -> float:
            model.eval()
            fixed_generator = torch.Generator(device=device).manual_seed(seed)
            total = 0.0
            with torch.no_grad():
                pieces = _factor_block_microbatches(
                    val_rows, val_inputs["group_index"], gradient_microbatch_size
                )
                for piece in pieces:
                    local = _subset(val_inputs, piece)
                    local["generator"] = fixed_generator
                    probabilities = None if val_probs is None else val_probs[piece]
                    total += float(
                        model.loss(
                            val_target[piece],
                            val_state[piece],
                            normalization_count=val_mass,
                            state_probs=probabilities,
                            **local,
                        )
                    )
            return _require_finite_validation_loss(total, epoch=epoch)

        torch.manual_seed(seed)
        batch_generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW([projection.weight], lr=learning_rate, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_additional_epochs
        )
        source_validation_loss = validation_loss(0)
        best_loss = source_validation_loss
        best_epoch = 0
        best_weight = projection.weight.detach().clone()
        waited = 0
        completed = 0
        stopped_early = False
        validation_trace = [
            {
                "epoch": 0,
                "validation_loss": float(source_validation_loss),
                "effective_min_delta": float(
                    max(absolute_min_delta, relative_min_delta * abs(source_validation_loss))
                ),
                "meaningful_improvement": True,
                "best_validation_loss": float(source_validation_loss),
                "epochs_without_improvement": 0,
            }
        ]
        for additional_epoch in range(1, max_additional_epochs + 1):
            # The base graph is frozen numerically as well as parametrically. In particular,
            # R9/R10 inherit demand dropout=0.15 from R6; switching the root to train mode would
            # inject an unregistered stochastic base representation into a projection-only fit.
            model.eval()
            batches = factor_block_batches(
                train_rows, train_inputs["group_index"], batch_size, batch_generator
            )
            normalization = train_mass / len(batches)
            for batch in batches:
                optimizer.zero_grad()
                logical = _subset(train_inputs, batch)
                logical_target = train_target[batch]
                logical_state = train_state[batch]
                logical_probs = None if train_probs is None else train_probs[batch]
                relative = torch.arange(len(batch), dtype=torch.long)
                for microbatch in _factor_block_microbatches(
                    relative, logical["group_index"], gradient_microbatch_size
                ):
                    micro = _subset(logical, microbatch)
                    probabilities = None if logical_probs is None else logical_probs[microbatch]
                    loss = model.loss(
                        logical_target[microbatch],
                        logical_state[microbatch],
                        normalization_count=normalization,
                        state_probs=probabilities,
                        **micro,
                    )
                    if loss.requires_grad:
                        loss.backward()
                nn.utils.clip_grad_norm_([projection.weight], gradient_clip_norm)
                optimizer.step()
            scheduler.step()
            candidate = validation_loss(additional_epoch)
            completed = additional_epoch
            effective_delta = max(absolute_min_delta, relative_min_delta * abs(best_loss))
            improved = _meaningful_validation_improvement(
                candidate, best_loss, absolute_min_delta, relative_min_delta
            )
            if improved:
                best_loss = candidate
                best_epoch = additional_epoch
                best_weight = projection.weight.detach().clone()
                waited = 0
            else:
                waited += 1
            validation_trace.append(
                {
                    "epoch": additional_epoch,
                    "validation_loss": float(candidate),
                    "effective_min_delta": float(effective_delta),
                    "meaningful_improvement": improved,
                    "best_validation_loss": float(best_loss),
                    "epochs_without_improvement": waited,
                }
            )
            if waited >= patience:
                stopped_early = True
                break
        with torch.no_grad():
            projection.weight.copy_(best_weight)
        if quiet_epoch_logging:
            _print_quiet_training_completion(
                "projection continuation",
                completed,
                max_additional_epochs,
                stopped_early,
            )

        model.eval()
        for name, expected in frozen_state.items():
            if not torch.equal(model.state_dict()[name].detach().cpu(), expected):
                raise OursError(f"projection continuation mutated frozen base tensor {name!r}")
        _assert_replay_inputs_unchanged(source_digests, projection_checkpoint, panel, None)
        for episode, (_, expected_digest) in zip(episodes, trajectory_snapshots, strict=True):
            if directory_digest(episode.artifact) != expected_digest:
                raise OursError(
                    f"projection continuation trajectory {episode.origin} changed during fit"
                )

        continuation_record = {
            "schema": AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_SCHEMA,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_projection_sha256": source_calibration["final_projection_sha256"],
            "source_checkpoint_is_epoch_zero": True,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "max_additional_epochs": max_additional_epochs,
            "additional_epochs_completed": completed,
            "patience": patience,
            "batch_size": batch_size,
            "gradient_microbatch_size": gradient_microbatch_size,
            "gradient_clip_norm": gradient_clip_norm,
            "seed": seed,
            "validation_seed": seed,
            "device": device,
            "absolute_min_delta": absolute_min_delta,
            "relative_min_delta": relative_min_delta,
            "training_episode": episode_records[0],
            "validation_episode": episode_records[1],
            "final_evaluation_origin": evaluation_origin,
            "o2_used_during_stopping": False,
            "accuracy_evaluation_during_training": False,
            "validation_evaluated_every_epoch": True,
            "common_validation_random_numbers": True,
            "source_validation_loss": float(source_validation_loss),
            "best_validation_loss": float(best_loss),
            "best_epoch": best_epoch,
            "epochs_without_improvement": waited,
            "stopped_early": stopped_early,
            "stop_reason": ("validation_patience" if stopped_early else "max_additional_epochs"),
            "best_weights_restored": True,
            "base_parameters_frozen": True,
            "trainable_parameters": [AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER],
            "quiet_epoch_logging": quiet_epoch_logging,
            "validation_trace": validation_trace,
        }
        calibration_record = dict(source_calibration)
        calibration_record["continuation"] = continuation_record
        calibration_record["final_projection_sha256"] = _projection_weight_sha256(model)
        continued_run = dict(run)
        continued_run["aggregate_trajectory_projection"] = {
            **projection_record,
            "calibration": calibration_record,
        }
        source_snapshot._assert_unchanged()
        for trajectory_snapshot in trajectory_snapshots:
            trajectory_snapshot._assert_unchanged()
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
        try:
            save_checkpoint(
                model, staged, model_config=dict(record["model_config"]), run=continued_run
            )
            try:
                os.link(staged, output)
            except FileExistsError as exc:
                raise OursError(
                    "projection continuation output appeared during fitting; refusing to "
                    "overwrite an immutable artifact"
                ) from exc
        finally:
            staged.unlink(missing_ok=True)
        return output


def predict_from_checkpoint(
    checkpoint: str | pathlib.Path,
    panel: Panel,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    origin: str | None = None,
    paths_out_dir: str | pathlib.Path | None = None,
    n_paths: int | None = None,
    inference_factor_draws: int | None = None,
    point_factor_draws: int | None = None,
    text_embeddings: str | pathlib.Path | None = None,
    aggregate_trajectory: str | pathlib.Path | None = None,
    final_price_ladder: str = "skip",
    seed: int | None = None,
) -> Forecast:
    """Forecast from a checkpoint while consuming an immutable snapshot of every input.

    The source paths are validated before any copy, then the checkpoint, panel, optional plan,
    and optional text artifact are copied into a private tree and content-bound to the source
    digests.  All model and Spark reads use that tree, closing the ABA race in a mutable input.
    """
    if paths_out_dir is None and (n_paths is not None or inference_factor_draws is not None):
        raise OursError(
            "n_paths and inference_factor_draws control a sample-path artifact, but "
            "paths_out_dir is not set; refusing arguments whose result cannot be written"
        )
    if final_price_ladder not in ("skip", "require"):
        raise OursError(
            "final_price_ladder must be 'skip' or 'require'; checkpoint replay has no "
            f"per-epoch ladder mode, got {final_price_ladder!r}"
        )
    _validate_replay_paths(
        checkpoint,
        panel,
        out_dir,
        paths_out_dir,
        text_embeddings,
        aggregate_trajectory=aggregate_trajectory,
    )
    plan_source_label = None if panel.plan is None else str(panel.plan.path)
    with _replay_input_snapshot(checkpoint, panel, text_embeddings) as snapshot:
        copied_checkpoint, copied_panel, copied_text, input_digests = snapshot
        with _aggregate_trajectory_input_snapshot(aggregate_trajectory) as trajectory_snapshot:
            copied_trajectory, trajectory_digest = trajectory_snapshot
            if trajectory_digest is not None:
                input_digests = {
                    **input_digests,
                    "aggregate_trajectory_sha256": trajectory_digest,
                }
            return _predict_from_checkpoint_snapshot(
                copied_checkpoint,
                copied_panel,
                spark,
                out_dir,
                origin=origin,
                paths_out_dir=paths_out_dir,
                n_paths=n_paths,
                inference_factor_draws=inference_factor_draws,
                point_factor_draws=point_factor_draws,
                text_embeddings=copied_text,
                aggregate_trajectory=copied_trajectory,
                seed=seed,
                final_price_ladder=final_price_ladder,
                _input_digests=input_digests,
                _plan_source_label=plan_source_label,
                _private_input_snapshots=tuple(
                    guard
                    for guard in (snapshot, trajectory_snapshot)
                    if hasattr(guard, "assert_unchanged")
                ),
            )


def _predict_from_checkpoint_snapshot(
    checkpoint: str | pathlib.Path,
    panel: Panel,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    origin: str | None = None,
    paths_out_dir: str | pathlib.Path | None = None,
    n_paths: int | None = None,
    inference_factor_draws: int | None = None,
    point_factor_draws: int | None = None,
    text_embeddings: str | pathlib.Path | None = None,
    aggregate_trajectory: str | pathlib.Path | None = None,
    final_price_ladder: str = "skip",
    seed: int | None = None,
    _input_digests: dict[str, str] | None = None,
    _plan_source_label: str | None = None,
    _private_input_snapshots: tuple[object, ...] = (),
) -> Forecast:
    """Replay from private input snapshots prepared by :func:`predict_from_checkpoint`.

    The add-a-client-without-retraining route the architecture claims: features are assembled
    from the GIVEN panel at the origin, identities are encoded through the checkpoint's stored
    vocabularies, and everything unseen lands in an explicitly finalised pooled row -- exact-zero
    identity deviations and the evidenced-row mean for factor loadings.  A random untouched
    embedding is not accepted as pooling.  An unseen attribute level one-hots to zeros, and
    baselines anchor to the new panel's OWN measured history. Static reals are standardised with
    the TRAINING stats, because a transformation that changes per client would make one physical
    weight mean different kilograms.

    Optional sample paths are materialised only when `paths_out_dir` is explicit. Without that
    destination, path-count arguments are refused instead of being accepted and ignored. A
    text-enabled checkpoint uses supplied embeddings when given; otherwise it aligns the saved
    training embeddings by stored series id and zero-fills genuinely unseen series.

    A checkpoint produced by bounded sharded FIT can be served here, but serving currently uses
    one full-panel `_assemble`. The checkpoint persists its shard predicates and requested cap so
    a future streaming implementation has the exact partition contract; until that implementation
    also merges one globally indexed path artifact transactionally, this entry point does not claim
    bounded serving memory.

    Raises:
        OursError: the panel role contract or derived known-future layout differs from fit; a
            checkpoint namespace/text tensor is incomplete; or a requested path artifact has no
            destination.
    """
    if paths_out_dir is None and (n_paths is not None or inference_factor_draws is not None):
        raise OursError(
            "n_paths and inference_factor_draws control a sample-path artifact, but "
            "paths_out_dir is not set; refusing arguments whose result cannot be written"
        )
    if final_price_ladder not in ("skip", "require"):
        raise OursError(
            "final_price_ladder must be 'skip' or 'require'; checkpoint replay has no "
            f"per-epoch ladder mode, got {final_price_ladder!r}"
        )
    _validate_replay_paths(
        checkpoint,
        panel,
        out_dir,
        paths_out_dir,
        text_embeddings,
        aggregate_trajectory=aggregate_trajectory,
    )
    input_digests = (
        _replay_input_digests(checkpoint, panel, text_embeddings)
        if _input_digests is None
        else _input_digests
    )
    if _input_digests is None and aggregate_trajectory is not None:
        input_digests["aggregate_trajectory_sha256"] = directory_digest(aggregate_trajectory)
    model, record = load_checkpoint(checkpoint)
    run = record["run"]
    projection_record = run.get("aggregate_trajectory_projection")
    if (
        isinstance(projection_record, dict)
        and projection_record.get("status") == "calibrated"
        and aggregate_trajectory is None
    ):
        raise OursError(
            "a calibrated aggregate-trajectory checkpoint requires an origin-matched sealed "
            "aggregate_trajectory artifact"
        )
    if aggregate_trajectory is not None and model.aggregate_trajectory_projection is None:
        raise OursError(
            "aggregate_trajectory was supplied to a checkpoint whose model has no projection"
        )
    artifact_device = run.get("artifact_device", "cpu")
    replay_device = _checkpoint_artifact_device(model, run)
    if run.get("oov_scheme") != OOV_SCHEME:
        raise OursError(
            "this checkpoint predates proof-carrying pooled identity rows; its reserved and "
            "short-history rows may still contain random initial embeddings. Refit before "
            "checkpoint serving rather than treating an untrained row as a cold-start estimate"
        )
    point_method, replay_point_draws, point_rng_scheme = _checkpoint_point_replay(
        run, point_factor_draws, factorless=model.factor is None
    )
    vocab = run.get("vocabularies")
    if vocab is None:
        raise OursError(
            "this checkpoint predates stored vocabularies; a new panel's identities cannot be "
            "encoded onto its embedding rows. Refit to produce a namespace-carrying checkpoint"
        )
    mismatched_roles = _panel_role_mismatches(panel, run.get("panel"))
    if mismatched_roles:
        raise OursError(
            f"panel role declaration(s) {mismatched_roles} differ from the fitted panel "
            "contract; rebuilding features under the old weights would change their meaning"
        )
    at_origin = origin or run["origin"]
    if isinstance(projection_record, dict) and projection_record.get("status") == "calibrated":
        expected_origin = projection_record["calibration"]["evaluation_origin"]
        if at_origin != expected_origin:
            raise OursError(
                f"calibrated projection is sealed for evaluation origin {expected_origin}, "
                f"not {at_origin}"
            )
        expected_panel_hash = projection_record["calibration"]["panel_parquet_sha256"]
        observed_panel_hash = input_digests.get("panel_parquet_sha256")
        if observed_panel_hash != expected_panel_hash:
            raise OursError(
                "calibrated projection evaluation panel differs from its sealed R6/calibration "
                f"panel: expected {expected_panel_hash}, got {observed_panel_hash}"
            )
    category = run.get("category_col")
    category_of = _series_levels(panel, spark, category) if category else None
    work = _replay_scratch_directory(at_origin)
    try:
        _validate_replay_paths(
            checkpoint,
            panel,
            out_dir,
            paths_out_dir,
            text_embeddings,
            scratch=work,
            aggregate_trajectory=aggregate_trajectory,
        )
        assembly = _assemble(
            panel,
            spark,
            at_origin,
            run["horizon"],
            run["encoder_days"],
            [tuple(e) for e in run["events"]],
            run["event_max_support"],
            work,
            tuple(tuple(c) for c in run["cycles"]),
            category_of=category_of,
            event_windows=(
                None
                if run.get("event_windows") is None
                else {k: tuple(v) for k, v in run["event_windows"].items()}
            ),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if assembly.plan is not None and _plan_source_label is not None:
        assembly = replace(assembly, plan=replace(assembly.plan, source=_plan_source_label))
    if list(assembly.known.names) != list(run["known_names"]):
        raise OursError(
            f"panel yields known-future columns {assembly.known.names} but the model trained "
            f"on {tuple(run['known_names'])}; a shifted layout reads one covariate as another"
        )
    recorded_drivers = run.get("declared_known_drivers")
    if recorded_drivers is not None and set(assembly.known.declared_drivers) != set(
        recorded_drivers
    ):
        raise OursError(
            f"panel declares known-future drivers {sorted(assembly.known.declared_drivers)} but "
            f"the model trained on {sorted(recorded_drivers)}; fixed placeholder columns make "
            "the tensor width match, but the feature-role contract changed"
        )
    series_ids = assembly.series_ids
    if not series_ids:
        raise OursError(
            f"no series has a full {run['encoder_days']}-day encoder window at {at_origin}"
        )
    n = len(series_ids)
    oov = run["oov_rows"]
    stored_series_ids = run.get("series_ids")
    if (
        not isinstance(stored_series_ids, list | tuple)
        or any(not isinstance(sid, str) or not sid.strip() for sid in stored_series_ids)
        or len(set(stored_series_ids)) != len(stored_series_ids)
    ):
        raise OursError(
            "checkpoint series_ids is absent, duplicated, or invalid; fitted identities cannot "
            "be distinguished from genuinely unseen series. Refit this checkpoint"
        )
    stored_series_ids = list(stored_series_ids)
    configured_series = record.get("model_config", {}).get("n_series")
    if (
        oov.get("series") != len(stored_series_ids)
        or configured_series != len(stored_series_ids) + 1
    ):
        raise OursError(
            f"checkpoint series namespace is inconsistent: {len(stored_series_ids)} saved ids, "
            f"OOV row {oov.get('series')!r}, model rows {configured_series!r}"
        )
    stored_series_row = {sid: row for row, sid in enumerate(stored_series_ids)}
    series_index = torch.tensor(
        [stored_series_row.get(sid, oov["series"]) for sid in series_ids], dtype=torch.long
    )

    # Attributes through the STORED one-hot layout; unseen levels one-hot to zeros.
    names = vocab["attribute_names"]
    attributes = torch.zeros(n, len(names))
    static_columns = list(dict.fromkeys(name.split("=", 1)[0] for name in names if "=" in name))
    if static_columns:
        column_levels = {}
        for j, name in enumerate(names):
            if "=" in name:
                column, level = name.split("=", 1)
                column_levels.setdefault(column, {})[level] = j
        for column in static_columns:
            values = _series_levels(panel, spark, column)
            slots = column_levels[column]
            for i, sid in enumerate(series_ids):
                j = slots.get(str(values.get(sid)))
                if j is not None:
                    attributes[i, j] = 1.0
    stats = run.get("static_real_stats")
    if stats:
        for column, (mean, std) in stats.items():
            j = names.index(column)
            values = _series_levels(panel, spark, column)
            for i, sid in enumerate(series_ids):
                value = values.get(sid)
                if value is None:
                    raise OursError(f"static real {column!r} is missing for series {sid!r}")
                attributes[i, j] = 0.0 if std <= 1e-12 else (float(value) - mean) / std

    def role_values(column: str | None) -> dict[str, str | None] | None:
        return None if column is None else _series_levels(panel, spark, column)

    def role_index(
        levels_key: str, values: dict[str, str | None] | None, oov_row: int
    ) -> torch.Tensor:
        if values is None:
            return torch.zeros(n, dtype=torch.long)
        return _indices_from_levels(vocab[levels_key], values, series_ids, oov_row, role=levels_key)

    product_values = role_values(run.get("product_col"))
    store_values = role_values(run.get("store_col"))
    factor_values = role_values(run.get("factor_group_col"))
    product_index = role_index("product_levels", product_values, oov["product"])
    store_index = role_index("store_levels", store_values, oov["store"])

    # A row can be in the stored namespace without ever reaching an optimiser batch (a new launch
    # can have the inference window but no earlier training window; series-held-out validation is
    # another case). Route those rows to the same explicit OOV state instead of preserving an
    # arbitrary physical row number.
    evidence = run["identity_evidence_rows"]

    def pool_unevidenced(
        index: torch.Tensor, role: str, oov_row: int, active: bool
    ) -> torch.Tensor:
        if not active:
            return index
        trained = set(evidence[role])
        return torch.tensor(
            [int(row) if int(row) in trained else oov_row for row in index], dtype=torch.long
        )

    series_index = pool_unevidenced(series_index, "series", oov["series"], True)
    product_index = pool_unevidenced(
        product_index, "product", oov["product"], model.static.factorised
    )
    store_index = pool_unevidenced(
        store_index, "store", oov["store"], model.static.factorised or model.factor is not None
    )

    if model.factor is None:
        group_index = _independent_group_index(n)
    else:
        if factor_values is None:
            raise OursError("factor checkpoint has no factor_group_col role")
        # Use RAW identities for path equality. Parameter OOV pooling must not collapse distinct
        # unseen `(factor_group, product)` pairs onto one systemic trajectory.
        group_index, _ = _query_composite_groups(series_ids, factor_values, product_values)

    text_dim = record.get("model_config", {}).get("text_dim", 0)
    if text_embeddings is not None:
        if text_dim == 0:
            raise OursError(
                "text_embeddings was supplied, but this checkpoint has no text projection"
            )
        try:
            text, supplied_dim, _ = _text_tensors(text_embeddings, spark, series_ids)
        except KeyError as exc:
            raise OursError(
                "supplied text artifact has no non-null embedding vector from which to identify "
                "its width"
            ) from exc
        if supplied_dim != text_dim:
            raise OursError(
                f"supplied text embeddings are {supplied_dim}-wide but the checkpoint text "
                f"projection expects {text_dim}"
            )
        if text is None or not bool(torch.isfinite(text).all()):
            raise OursError(
                "supplied text embeddings contain non-finite values; refusing before the "
                "identity projection matrix multiplication"
            )
    elif text_dim:
        tensors = run.get("tensors")
        saved_text = None if not isinstance(tensors, dict) else tensors.get("text")
        expected = (len(stored_series_ids), text_dim)
        if not isinstance(saved_text, torch.Tensor) or tuple(saved_text.shape) != expected:
            shape = None if not isinstance(saved_text, torch.Tensor) else tuple(saved_text.shape)
            raise OursError(
                f"text-enabled checkpoint has saved text tensor {shape}, expected {expected}; "
                "refusing before the identity projection matrix multiplication"
            )
        if not saved_text.is_floating_point() or not bool(torch.isfinite(saved_text).all()):
            raise OursError(
                "text-enabled checkpoint contains non-floating or non-finite embeddings; "
                "refusing before the identity projection matrix multiplication"
            )
        text = torch.zeros(n, text_dim, dtype=torch.float32)
        for row, sid in enumerate(series_ids):
            stored_row = stored_series_row.get(sid)
            if stored_row is not None:
                text[row] = saved_text[stored_row].float()
    else:
        text = None
    trajectory_tensor = None
    trajectory_manifest = None
    if aggregate_trajectory is not None:
        trajectory_tensor, trajectory_manifest = _aggregate_trajectory_tensor(
            aggregate_trajectory,
            panel,
            spark,
            origin=at_origin,
            horizon=run["horizon"],
            series_ids=series_ids,
            dtype=assembly.known.values.dtype,
        )
        if isinstance(projection_record, dict) and projection_record.get("status") == "calibrated":
            calibration_manifest = projection_record["calibration"]["episodes"][0]["manifest"]
            drift = {
                field: (calibration_manifest[field], trajectory_manifest[field])
                for field in _AGGREGATE_TRAJECTORY_STABLE_MANIFEST_FIELDS
                if not _json_values_equal_exact(
                    calibration_manifest[field], trajectory_manifest[field]
                )
            }
            if drift:
                raise OursError(
                    f"evaluation aggregate trajectory changes calibrated semantics: {drift}"
                )
            if (calibration_manifest["provider_adaptation_identity"] is None) != (
                trajectory_manifest["provider_adaptation_identity"] is None
            ):
                raise OursError(
                    "evaluation aggregate trajectory switches the calibrated provider between "
                    "frozen and adapted execution"
                )
    inputs = dict(
        history=assembly.history,
        observed=assembly.observed,
        censored=assembly.censored,
        attributes=attributes,
        # Known series retain their fitted row. Only a genuinely unseen id takes the untouched
        # OOV residual, which carries the pooled behaviour for a cold start.
        series_index=series_index,
        text=text,
        known_future=assembly.known.values,
        baseline_logit=assembly.baseline_logit,
        baseline_log_size=assembly.baseline_log_size,
        group_index=group_index,
        store_index=store_index,
        product_index=product_index,
        n_draws=8,
        commercial=assembly.commercial,
        commercial_mask=assembly.commercial_mask,
        event_distance=assembly.event_distance,
        event_present=assembly.event_present,
    )
    inputs["aggregate_trajectory"] = trajectory_tensor

    # A calibrated hybrid must be audited on the exact O2 inputs that it serves, including its
    # immutable aggregate trajectory.  This is intentionally inside the public checkpoint
    # replay boundary: a runner that rebuilt private tensors itself could silently audit a
    # different graph.  The controls come from the sealed R6 checkpoint, and the callback is
    # reachable only after projection fitting has returned a final checkpoint.
    served_seed = seed if seed is not None else run["seed"]
    final_ladder_record = None
    if final_price_ladder == "require":
        ladder_controls = _checkpoint_final_price_ladder_controls(run)
        if (
            ladder_controls["point_factor_method"] != point_method
            or ladder_controls["point_factor_draws"] != replay_point_draws
            or ladder_controls["seed"] != served_seed
            or ladder_controls["rng_domain_scheme"] != point_rng_scheme
        ):
            raise OursError(
                "final checkpoint price-ladder controls differ from the emitted point-forecast "
                "route; refusing to certify one numerical forecast with another"
            )
        sample_size = ladder_controls["n_series"]
        if sample_size > n:
            raise OursError(
                f"final checkpoint price ladder requires {sample_size} series but replay has "
                f"only {n}"
            )
        keep = _sample_series_indices(series_ids, sample_size, ladder_controls["seed"])
        ladder_inputs = _subset(inputs, torch.tensor(keep, dtype=torch.long))
        ladder_report = verify_price_ladder(
            model,
            ladder_inputs,
            bumps=ladder_controls["bumps"],
            tolerance=ladder_controls["tolerance"],
            point_factor_method=ladder_controls["point_factor_method"],
            point_factor_draws=ladder_controls["point_factor_draws"],
            seed=ladder_controls["seed"],
            rng_domain_scheme=ladder_controls["rng_domain_scheme"],
        )
        if not ladder_report.passed:
            raise OursError(
                "final_price_ladder='require': calibrated checkpoint replay violates the "
                f"structural price guarantee in {ladder_report.violation_share:.2%} of "
                f"eligible cells (worst increase {ladder_report.worst_increase:.2%})"
            )
        final_ladder_record = {
            **ladder_report.as_record(),
            "mode": "require",
            "evaluation_origin": at_origin,
            "evaluation_timing": "once_after_all_checkpoint_and_projection_epochs",
            "aggregate_trajectory_sha256": input_digests.get("aggregate_trajectory_sha256"),
        }
    path_request = None
    if paths_out_dir is not None:
        path_count = n_paths if n_paths is not None else run.get("n_paths", 1000)
        if not isinstance(path_count, int) or isinstance(path_count, bool) or path_count < 1:
            raise OursError(f"n_paths must be a positive integer, got {path_count!r}")
        path_factor_draws = inference_factor_draws
        if path_factor_draws is None:
            path_factor_draws = run.get("path_factor_draws")
        if path_factor_draws is None:
            path_factor_draws = default_factor_draws(path_count)
        if (
            not isinstance(path_factor_draws, int)
            or isinstance(path_factor_draws, bool)
            or not 1 <= path_factor_draws <= path_count
        ):
            raise OursError(
                f"factor_draws must lie in [1, n_paths={path_count}], got {path_factor_draws!r}"
            )
        if path_count % path_factor_draws:
            raise OursError(
                f"n_paths={path_count} is not divisible by factor_draws={path_factor_draws}. "
                "An uneven split would reweight the predictive distribution"
            )
        sample = run.get("paths_sample", min(2000, n))
        if not isinstance(sample, int) or isinstance(sample, bool) or sample < 1:
            raise OursError(f"checkpoint paths_sample must be a positive integer, got {sample!r}")
        path_request = path_count, path_factor_draws, min(sample, n)

    # THE CHECKPOINT'S OWN METHOD, count and RNG domain -- never this process's defaults. A fit
    # that recorded deterministic quadrature must not be re-served by seeded Monte Carlo. Older
    # checkpoints did use MC, but did not record whether their 1,000-draw default was overridden;
    # `_checkpoint_point_replay` refuses that missing evidence unless the caller opts into a new
    # explicitly counted re-evaluation.
    yhat = _checkpoint_point_forecast(
        model,
        inputs,
        run,
        panel,
        spark,
        series_ids,
        method=point_method,
        factor_draws=replay_point_draws,
        seed=served_seed,
        rng_domain_scheme=point_rng_scheme,
    )

    point_stage = _staging_directory(out_dir)
    paths_stage = point_stage / "_paths" if paths_out_dir is not None else None
    try:
        d0 = dt.date.fromisoformat(at_origin)
        records = [
            (series_ids[i], d0 + dt.timedelta(days=step + 1), float(yhat[i, step]))
            for i in range(n)
            for step in range(run["horizon"])
        ]
        spark.createDataFrame(records, "series_id string, date date, yhat double").write.mode(
            "overwrite"
        ).parquet(str(point_stage))
        # Compute the point digest before adding the hidden path dataset below. Spark ignores
        # underscore-prefixed children when reading the point Parquet root, and the two digests
        # deliberately describe the two datasets separately.
        forecast_digest = _spark_parquet_digest(point_stage)

        path_digest = None
        path_series_ids = None
        path_rng_scheme = None
        if path_request is not None:
            path_count, path_factor_draws, sample = path_request
            keep = _sample_series_indices(series_ids, sample, served_seed)
            path_series_ids = [series_ids[i] for i in keep]
            index = torch.tensor(keep, dtype=torch.long)
            path_inputs = _subset(inputs, index)
            path_rng_scheme = run.get("rng_domain_scheme", LEGACY_RNG_DOMAIN_SCHEME)
            paths = draw_paths_over_factor(
                model,
                path_inputs,
                path_count,
                factor_draws=path_factor_draws,
                seed=served_seed,
                rng_domain_scheme=path_rng_scheme,
            )
            assert paths_stage is not None
            _write_paths(
                paths, path_series_ids, at_origin, run["horizon"], len(keep), spark, paths_stage
            )
            # `_write_paths` persists lexical series order and records `series_index` against
            # that order. Bind the replay record to the stored order, not hash-sampling order.
            path_series_ids = sorted(path_series_ids)
            path_digest = _spark_parquet_digest(paths_stage)

        n_total = spark.read.parquet(panel.path).select(panel.series_col).distinct().count()
        disposition = {EMITTED: len(records)}
        declined = (n_total - n) * run["horizon"]
        if declined:
            disposition[DECLINED_NO_WINDOW] = declined

        # A refreshed PlanSource is allowed, but every mutable input is hashed before its first
        # read and rechecked after its last. The record and Parquet files live in the unpublished
        # stage together, so no apparently complete forecast can precede its provenance marker.
        replay_record = {
            **input_digests,
            "aggregate_trajectory_manifest": trajectory_manifest,
            "checkpoint_resolved_build_hash": run.get("resolved_build_hash"),
            "training_device": run.get("training_device", "cpu"),
            "artifact_device": artifact_device,
            "replay_device": replay_device,
            "origin": at_origin,
            "horizon": run["horizon"],
            "point_factor_method": point_method,
            "point_factor_draws": replay_point_draws,
            "seed": served_seed,
            "rng_domain_scheme": point_rng_scheme,
            "path_n_paths": None if path_request is None else path_count,
            "path_factor_draws": None if path_request is None else path_factor_draws,
            "path_rng_domain_scheme": path_rng_scheme,
            "path_series_ids": path_series_ids,
            "final_price_ladder": final_ladder_record,
            "plan_provenance": None if assembly.plan is None else assembly.plan.as_record(),
            "forecast_parquet_sha256": forecast_digest,
            "paths_parquet_sha256": path_digest,
        }
        _write_json_atomic(point_stage / "_replay_record.json", replay_record)
        _assert_replay_inputs_unchanged(
            {
                name: digest
                for name, digest in input_digests.items()
                if name != "aggregate_trajectory_sha256"
            },
            checkpoint,
            panel,
            text_embeddings,
        )
        if aggregate_trajectory is not None:
            observed_trajectory_digest = directory_digest(aggregate_trajectory)
            expected_trajectory_digest = input_digests.get("aggregate_trajectory_sha256")
            if observed_trajectory_digest != expected_trajectory_digest:
                raise OursError("aggregate trajectory changed while checkpoint replay consumed it")
        for private_snapshot in _private_input_snapshots:
            private_snapshot._assert_unchanged()

        # Point rows, optional paths, and the completion record are one immutable tree. One local
        # rename is the transaction boundary; there is no interval in which an old point artifact
        # names new paths (or vice versa).
        _publish_directory(point_stage, out_dir)
        point_stage = None
        paths_stage = None
    finally:
        if point_stage is not None:
            shutil.rmtree(point_stage, ignore_errors=True)
    return Forecast(
        adapter=NAME,
        path=str(out_dir),
        origin=at_origin,
        horizon=run["horizon"],
        n_series=n_total,
        disposition=disposition,
        target_estimand=panel.target_estimand,
    ).validate()


CHECKPOINT_SCHEMA = "autoforecast.checkpoint.v2"
AGGREGATE_TRAJECTORY_PROJECTION_SCHEMA = "autoforecast.aggregate-trajectory-projection.v1"
AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_SCHEMA = (
    "autoforecast.aggregate-trajectory-projection-continuation.v1"
)
PRICE_RESPONSE_CONFIG_FIELDS = frozenset(
    {
        "price_response_mode",
        "price_occurrence_slope_max",
        "price_size_slope_max",
    }
)


def _projection_weight_sha256(model: CustomModel) -> str | None:
    projection = model.aggregate_trajectory_projection
    if projection is None:
        return None
    weight = projection.weight.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(weight.dtype).encode())
    digest.update(repr(tuple(weight.shape)).encode())
    digest.update(weight.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _dormant_aggregate_projection_record(model: CustomModel) -> dict[str, object] | None:
    if model.aggregate_trajectory_projection is None:
        return None
    return {
        "schema": AGGREGATE_TRAJECTORY_PROJECTION_SCHEMA,
        "mode": AGGREGATE_TRAJECTORY_MODE,
        "status": "dormant_r6",
        "source_r6_checkpoint_sha256": None,
        "trainable_parameters": [],
        "base_parameters_frozen": False,
        "calibration": None,
    }


def _validate_projection_continuation_record(
    continuation: object,
    *,
    episodes: list[dict[str, object]],
    evaluation_origin: str,
    final_projection_sha256: str,
) -> None:
    """Validate the exact stopping trace attached to a continued R9/R10 projection."""
    fields = {
        "schema",
        "source_checkpoint_sha256",
        "source_projection_sha256",
        "source_checkpoint_is_epoch_zero",
        "optimizer_state_restored",
        "scheduler_state_restored",
        "optimizer",
        "scheduler",
        "learning_rate",
        "weight_decay",
        "max_additional_epochs",
        "additional_epochs_completed",
        "patience",
        "batch_size",
        "gradient_microbatch_size",
        "gradient_clip_norm",
        "seed",
        "validation_seed",
        "device",
        "absolute_min_delta",
        "relative_min_delta",
        "training_episode",
        "validation_episode",
        "final_evaluation_origin",
        "o2_used_during_stopping",
        "accuracy_evaluation_during_training",
        "validation_evaluated_every_epoch",
        "common_validation_random_numbers",
        "source_validation_loss",
        "best_validation_loss",
        "best_epoch",
        "epochs_without_improvement",
        "stopped_early",
        "stop_reason",
        "best_weights_restored",
        "base_parameters_frozen",
        "trainable_parameters",
        "quiet_epoch_logging",
        "validation_trace",
    }
    if type(continuation) is not dict or set(continuation) != fields:
        raise OursError("projection continuation record is absent or not the exact current schema")
    if (
        continuation["schema"] != AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_SCHEMA
        or not _sha256_is_canonical(continuation["source_checkpoint_sha256"])
        or not _sha256_is_canonical(continuation["source_projection_sha256"])
    ):
        raise OursError("projection continuation source identity is malformed")
    required_true = {
        "source_checkpoint_is_epoch_zero",
        "validation_evaluated_every_epoch",
        "common_validation_random_numbers",
        "best_weights_restored",
        "base_parameters_frozen",
        "quiet_epoch_logging",
    }
    required_false = {
        "optimizer_state_restored",
        "scheduler_state_restored",
        "o2_used_during_stopping",
        "accuracy_evaluation_during_training",
    }
    if (
        any(continuation[field] is not True for field in required_true)
        or any(continuation[field] is not False for field in required_false)
        or continuation["optimizer"] != "AdamW"
        or continuation["scheduler"] != "CosineAnnealingLR"
        or not _json_values_equal_exact(continuation["weight_decay"], 0.0)
        or continuation["trainable_parameters"] != [AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER]
    ):
        raise OursError(
            "projection continuation does not preserve the registered fresh-state, "
            "projection-only boundary"
        )
    observed_controls = {
        field: continuation[field]
        for field in AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_CONTROLS
    }
    if not _json_values_equal_exact(
        observed_controls, AGGREGATE_TRAJECTORY_PROJECTION_CONTINUATION_CONTROLS
    ):
        raise OursError(
            "projection continuation controls differ from the registered R9/R10 controls"
        )
    if (
        not _json_values_equal_exact(continuation["validation_seed"], continuation["seed"])
        or not _json_values_equal_exact(continuation["training_episode"], episodes[0])
        or not _json_values_equal_exact(continuation["validation_episode"], episodes[1])
        or not _json_values_equal_exact(continuation["final_evaluation_origin"], evaluation_origin)
    ):
        raise OursError("projection continuation episode roles or fixed validation seed changed")
    maximum = continuation["max_additional_epochs"]
    completed = continuation["additional_epochs_completed"]
    best_epoch = continuation["best_epoch"]
    waited = continuation["epochs_without_improvement"]
    if (
        type(completed) is not int
        or type(best_epoch) is not int
        or type(waited) is not int
        or not (1 <= completed <= maximum)
        or not (0 <= best_epoch <= completed)
        or waited != completed - best_epoch
    ):
        raise OursError("projection continuation epoch/best/staleness summary is inconsistent")
    for field in ("source_validation_loss", "best_validation_loss"):
        if type(continuation[field]) is not float or not math.isfinite(continuation[field]):
            raise OursError(f"projection continuation {field} must be a finite float")
    if continuation["best_validation_loss"] > continuation["source_validation_loss"]:
        raise OursError("projection continuation best validation loss is worse than epoch zero")
    if best_epoch == 0 and continuation["source_projection_sha256"] != final_projection_sha256:
        raise OursError(
            "projection continuation claims epoch-zero restoration but its final projection "
            "weight hash differs from the source projection"
        )
    if type(continuation["stopped_early"]) is not bool:
        raise OursError("projection continuation stopped_early must be boolean")
    if continuation["stopped_early"]:
        if (
            continuation["stop_reason"] != "validation_patience"
            or waited != continuation["patience"]
        ):
            raise OursError("projection continuation early-stop summary is inconsistent")
    elif (
        continuation["stop_reason"] != "max_additional_epochs"
        or completed != maximum
        or waited >= continuation["patience"]
    ):
        raise OursError("projection continuation max-epoch summary is inconsistent")

    trace = continuation["validation_trace"]
    trace_fields = {
        "epoch",
        "validation_loss",
        "effective_min_delta",
        "meaningful_improvement",
        "best_validation_loss",
        "epochs_without_improvement",
    }
    if type(trace) is not list or len(trace) != completed + 1:
        raise OursError(
            "projection continuation validation trace must contain epoch zero and every epoch"
        )
    previous_best = None
    previous_stale = 0
    for epoch, entry in enumerate(trace):
        if (
            type(entry) is not dict
            or set(entry) != trace_fields
            or type(entry["epoch"]) is not int
            or entry["epoch"] != epoch
        ):
            raise OursError("projection continuation validation trace schema/order is invalid")
        for field in ("validation_loss", "effective_min_delta", "best_validation_loss"):
            if type(entry[field]) is not float or not math.isfinite(entry[field]):
                raise OursError(f"projection continuation trace {field} must be a finite float")
        if (
            type(entry["meaningful_improvement"]) is not bool
            or type(entry["epochs_without_improvement"]) is not int
            or entry["epochs_without_improvement"] < 0
        ):
            raise OursError("projection continuation trace decision state is malformed")
        if epoch == 0:
            expected_delta = max(
                continuation["absolute_min_delta"],
                continuation["relative_min_delta"] * abs(entry["validation_loss"]),
            )
            if (
                entry["validation_loss"] != continuation["source_validation_loss"]
                or entry["best_validation_loss"] != entry["validation_loss"]
                or entry["effective_min_delta"] != expected_delta
                or entry["meaningful_improvement"] is not True
                or entry["epochs_without_improvement"] != 0
            ):
                raise OursError(
                    "projection continuation trace does not initialise epoch zero exactly"
                )
            previous_best = entry["best_validation_loss"]
            continue
        assert previous_best is not None
        expected_delta = max(
            continuation["absolute_min_delta"],
            continuation["relative_min_delta"] * abs(previous_best),
        )
        improved = entry["validation_loss"] < previous_best - expected_delta
        expected_best = entry["validation_loss"] if improved else previous_best
        expected_stale = 0 if improved else previous_stale + 1
        if (
            entry["effective_min_delta"] != expected_delta
            or entry["meaningful_improvement"] is not improved
            or entry["best_validation_loss"] != expected_best
            or entry["epochs_without_improvement"] != expected_stale
        ):
            raise OursError("projection continuation trace decision is inconsistent")
        if expected_stale >= continuation["patience"] and epoch != completed:
            raise OursError(
                "projection continuation trace continued after validation patience elapsed"
            )
        previous_best = expected_best
        previous_stale = expected_stale
    if previous_best != continuation["best_validation_loss"] or previous_stale != waited:
        raise OursError("projection continuation trace final state contradicts its summary")


def _validate_aggregate_projection_checkpoint(
    model: CustomModel,
    run: dict,
    model_config: dict,
) -> None:
    """Bind the executable projection to either a zero R6 source or one sealed calibration."""
    configured = model_config.get("aggregate_trajectory_mode", "none")
    if configured not in ("none", AGGREGATE_TRAJECTORY_MODE):
        raise OursError(f"checkpoint aggregate_trajectory_mode is unsupported: {configured!r}")
    if configured != model.aggregate_trajectory_mode:
        raise OursError("checkpoint aggregate trajectory constructor contradicts executable model")
    projection = model.aggregate_trajectory_projection
    if (configured == "none") != (projection is None):
        raise OursError(
            "checkpoint aggregate trajectory mode contradicts projection module presence"
        )
    record = run.get("aggregate_trajectory_projection")
    if projection is None:
        if record is not None:
            raise OursError(
                "checkpoint records aggregate projection evidence but has no executable module"
            )
        return
    expected_width = 1 if model.head_kind == "negbin" else 2
    if (
        projection.bias is not None
        or projection.in_features != 2
        or projection.out_features != expected_width
    ):
        raise OursError(
            "checkpoint aggregate projection is not the registered bias-free 2-channel head route"
        )
    if not bool(torch.isfinite(projection.weight.detach()).all()):
        raise OursError("checkpoint aggregate projection contains a non-finite weight")
    outer_fields = {
        "schema",
        "mode",
        "status",
        "source_r6_checkpoint_sha256",
        "trainable_parameters",
        "base_parameters_frozen",
        "calibration",
    }
    if type(record) is not dict or set(record) != outer_fields:
        raise OursError(
            "aggregate projection checkpoint record is absent or not the exact current schema"
        )
    if (
        record["schema"] != AGGREGATE_TRAJECTORY_PROJECTION_SCHEMA
        or record["mode"] != AGGREGATE_TRAJECTORY_MODE
    ):
        raise OursError("aggregate projection checkpoint schema/mode is unsupported")
    status = record["status"]
    if status == "dormant_r6":
        if (
            record["source_r6_checkpoint_sha256"] is not None
            or record["trainable_parameters"] != []
            or record["base_parameters_frozen"] is not False
            or record["calibration"] is not None
            or bool(torch.count_nonzero(projection.weight.detach()))
        ):
            raise OursError(
                "dormant R6 aggregate projection must be exact-zero and carry no calibration"
            )
        return
    if status != "calibrated":
        raise OursError(f"unsupported aggregate projection status {status!r}")
    if (
        not _sha256_is_canonical(record["source_r6_checkpoint_sha256"])
        or record["trainable_parameters"] != [AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER]
        or record["base_parameters_frozen"] is not True
    ):
        raise OursError(
            "calibrated projection does not bind one R6 hash and the sole trainable matrix"
        )
    calibration = record["calibration"]
    calibration_fields = {
        "evaluation_origin",
        "max_label_date",
        "panel_parquet_sha256",
        "evidence_role",
        "claim_eligible",
        "episodes",
        "epochs",
        "learning_rate",
        "batch_size",
        "gradient_microbatch_size",
        "gradient_clip_norm",
        "seed",
        "device",
        "optimizer",
        "weight_decay",
        "final_projection_sha256",
        "evaluation_schedule",
    }
    if type(calibration) is not dict or set(calibration) not in (
        calibration_fields,
        calibration_fields | {"continuation"},
    ):
        raise OursError("aggregate projection calibration record is not the exact current schema")
    evaluation_origin = _canonical_iso_date(
        calibration["evaluation_origin"], "projection evaluation origin"
    )
    max_label_date = _canonical_iso_date(
        calibration["max_label_date"], "projection maximum label date"
    )
    source_origin = _canonical_iso_date(run.get("origin"), "source R6 origin")
    episodes = calibration["episodes"]
    episode_fields = {"origin", "target_end", "artifact_sha256", "manifest"}
    if (
        type(episodes) is not list
        or len(episodes) != 2
        or any(type(episode) is not dict or set(episode) != episode_fields for episode in episodes)
    ):
        raise OursError(
            "aggregate projection must seal exactly two ordered forward calibration episodes"
        )
    if (
        model_config["horizon"] != AGGREGATE_TRAJECTORY_HORIZON
        or tuple(episode["origin"] for episode in episodes)
        != AGGREGATE_TRAJECTORY_CALIBRATION_ORIGINS
        or calibration["evaluation_origin"] != AGGREGATE_TRAJECTORY_EVALUATION_ORIGIN
    ):
        raise OursError(
            "aggregate projection checkpoint differs from the registered d1857/d1885/d1913 "
            "28-day forward protocol"
        )
    previous_end = None
    for index, episode in enumerate(episodes):
        origin = _canonical_iso_date(episode["origin"], f"episode {index} origin")
        target_end = _canonical_iso_date(episode["target_end"], f"episode {index} target_end")
        if target_end != origin + dt.timedelta(days=model_config["horizon"]):
            raise OursError(
                f"aggregate projection episode {index} does not span exactly one horizon"
            )
        if index == 0 and origin != source_origin:
            raise OursError("first projection episode must start at the dormant R6 training origin")
        if previous_end is not None and origin != previous_end:
            raise OursError("projection calibration episodes are not contiguous and forward")
        if not _sha256_is_canonical(episode["artifact_sha256"]):
            raise OursError(f"projection episode {index} artifact hash is not SHA-256")
        _validate_aggregate_trajectory_manifest_claims(
            episode["manifest"], origin=episode["origin"], horizon=model_config["horizon"]
        )
        previous_end = target_end
    first_manifest = episodes[0]["manifest"]
    drift = {
        field: (first_manifest[field], episodes[1]["manifest"][field])
        for field in _AGGREGATE_TRAJECTORY_STABLE_MANIFEST_FIELDS
        if not _json_values_equal_exact(first_manifest[field], episodes[1]["manifest"][field])
    }
    if drift:
        raise OursError(f"aggregate projection checkpoint changes episode semantics: {drift}")
    if (first_manifest["provider_adaptation_identity"] is None) != (
        episodes[1]["manifest"]["provider_adaptation_identity"] is None
    ):
        raise OursError(
            "aggregate projection checkpoint switches between frozen and adapted episodes"
        )
    if (
        calibration["evidence_role"] != first_manifest["evidence_role"]
        or calibration["claim_eligible"] is not first_manifest["claim_eligible"]
    ):
        raise OursError(
            "aggregate projection calibration claim summary contradicts its sealed episodes"
        )
    if previous_end != evaluation_origin or max_label_date != evaluation_origin:
        raise OursError(
            "projection maximum label date and final episode end must equal evaluation_origin"
        )
    for field in ("panel_parquet_sha256", "final_projection_sha256"):
        if not _sha256_is_canonical(calibration[field]):
            raise OursError(f"aggregate projection calibration {field} is not SHA-256")
    if calibration["final_projection_sha256"] != _projection_weight_sha256(model):
        raise OursError("aggregate projection weight hash does not match loaded weights")
    observed_controls = {
        field: calibration[field] for field in AGGREGATE_TRAJECTORY_PROJECTION_CONTROLS
    }
    if not _json_values_equal_exact(observed_controls, AGGREGATE_TRAJECTORY_PROJECTION_CONTROLS):
        raise OursError(
            "aggregate projection calibration controls differ from the registered R6 controls: "
            f"{observed_controls}"
        )
    if "continuation" in calibration:
        _validate_projection_continuation_record(
            calibration["continuation"],
            episodes=episodes,
            evaluation_origin=calibration["evaluation_origin"],
            final_projection_sha256=calibration["final_projection_sha256"],
        )


def _validate_checkpoint_model_config_schema(model_config: object) -> str:
    """Return the explicit price mode for a current-schema checkpoint constructor."""
    if type(model_config) is not dict:
        raise OursError("checkpoint model_config must be a JSON-style mapping")
    missing = PRICE_RESPONSE_CONFIG_FIELDS.difference(model_config)
    if missing:
        raise OursError(
            "checkpoint model_config is unversioned/legacy and cannot be loaded by the v2-only "
            f"runtime; missing current fields {sorted(missing)}"
        )
    mode = model_config["price_response_mode"]
    if type(mode) is not str or mode not in ("none", "soft_penalty_v1", "monotone_raw_v1"):
        raise OursError(f"checkpoint price_response_mode is unsupported: {mode!r}")
    occurrence_cap = model_config["price_occurrence_slope_max"]
    size_cap = model_config["price_size_slope_max"]
    if mode == "monotone_raw_v1":
        for field, value in (
            ("price_occurrence_slope_max", occurrence_cap),
            ("price_size_slope_max", size_cap),
        ):
            if type(value) is not float or not is_finite_real(value) or value <= 0.0:
                raise OursError(
                    f"checkpoint {field} must be a finite positive float under "
                    f"monotone_raw_v1, got {value!r}"
                )
    elif (occurrence_cap, size_cap) != (None, None):
        raise OursError("checkpoint price slope caps must both be null outside monotone_raw_v1")
    return mode


def save_checkpoint(
    model: CustomModel, path: str | pathlib.Path, *, model_config: dict, run: dict
) -> pathlib.Path:
    """Persist the fitted model with its architecture and its per-series conditioning tensors.

    `path_materialisation='on_demand'` is defined as "persist model state and a seed;
    regenerate paths per query" -- and until this existed the model was never persisted, so
    nothing could be regenerated, and the scenario-contrast machinery had no way to run outside
    the training process that happened to hold the weights. What is saved is exactly what a
    later query cannot re-derive from the panel alone: the weights, the constructor config that
    shapes them, and the run record (`run`) holding the assembly recipe plus the per-series
    tensors (attributes, indices, text) aligned to the forecast index. Encoder windows and
    known-future tensors are NOT saved -- they are origin-dependent and `_assemble` rebuilds
    them from the panel at whatever origin the query asks about.
    """
    configured_mode = _validate_checkpoint_model_config_schema(model_config)
    if configured_mode != model.price_response_mode:
        raise OursError(
            "checkpoint model_config price-response mode contradicts the executable model: "
            f"config={configured_mode!r}, model={model.price_response_mode!r}"
        )
    run_record = dict(run)
    support = model_config.get("event_max_support", 30)
    if type(support) is not int or not is_finite_real(support) or support < 1:
        raise OursError(f"checkpoint event_max_support is not a positive integer: {support!r}")
    run_support = run_record.setdefault("event_max_support", support)
    if type(run_support) is not int or run_support != support:
        raise OursError(
            f"checkpoint replay event_max_support={run_support!r} contradicts executable "
            f"model_config support {support!r}"
        )
    _validate_checkpoint_commercial_contract(model_config, run_record)
    _validate_aggregate_projection_checkpoint(model, run_record, dict(model_config))
    _validate_checkpoint_resolved_build(model, {"run": run_record}, dict(model_config))
    _validate_warm_start_checkpoint_record(run_record)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": dict(model_config),
        "run": run_record,
    }
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out))
    return out


def _file_sha256(path: pathlib.Path) -> str:
    """Streaming SHA-256 for a checkpoint bound into a replay record."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _spark_parquet_digest(path: str | pathlib.Path) -> str:
    """Hash every data file visible at a Spark dataset root.

    Hadoop/Spark ignore descendants whose relative directory component begins with ``_`` or
    ``.``. Replay stores sample paths under ``_paths`` inside the atomic point-artifact tree, so
    the point digest must apply that same visibility rule and remain reproducible after publish.
    """
    root = pathlib.Path(path)
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and not any(part.startswith(("_", ".")) for part in candidate.relative_to(root).parts)
    )
    if not files:
        raise OursError(f"{root} contains no Spark-visible data files")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(str(candidate.relative_to(root)).encode())
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_commercial_drivers(run: dict, known_names: tuple[str, ...]) -> frozenset[str]:
    """Validate the explicit current-schema role metadata."""
    recorded = run.get("declared_known_drivers")
    if (
        not isinstance(recorded, list | tuple)
        or any(not isinstance(name, str) or not name.strip() for name in recorded)
        or len(set(recorded)) != len(recorded)
    ):
        raise OursError(
            "current-schema checkpoint declared_known_drivers is absent, duplicated, or "
            "malformed"
        )
    declared = frozenset(recorded)
    unknown = sorted(declared.difference(known_names))
    if unknown:
        raise OursError(
            f"checkpoint declares known-future drivers absent from known_names: {unknown}"
        )
    return declared


def _commercial_indices_from_roles(
    known_names: tuple[str, ...],
    declared: frozenset[str],
) -> dict[str, int | None]:
    """Resolve model audit/synchronisation indices from live roles, not placeholder names."""

    def live_index(name: str) -> int | None:
        return known_names.index(name) if name in declared else None

    price = live_index("price")
    if price is not None:
        available = "price_available"
        if available not in known_names:
            raise OursError("checkpoint declares price but lacks price_available")
        if known_names.index(available) != price + len(commercial.CHANNELS):
            raise OursError("checkpoint price availability is not in the fixed commercial layout")
    discount = live_index("discount_depth")
    available_name = "discount_depth_available"
    if discount is not None and available_name not in known_names:
        raise OursError("checkpoint declares discount_depth but lacks discount_depth_available")
    return {
        "price_index": price,
        "price_available_index": (
            known_names.index("price_available") if price is not None else None
        ),
        "discount_index": discount,
        "discount_available_index": (
            known_names.index(available_name) if discount is not None else None
        ),
        "promo_index": live_index("promo"),
    }


def _validate_checkpoint_commercial_contract(model_config: dict, run: dict) -> None:
    """Bind every live commercial role to explicit current-schema names and indices."""
    role_fields = (
        "price_index",
        "price_available_index",
        "discount_index",
        "discount_available_index",
        "promo_index",
    )
    requires_role_layout = (
        bool(model_config.get("n_commercial", 0))
        or any(model_config.get(name) is not None for name in role_fields)
        or model_config.get("price_response_mode") in ("soft_penalty_v1", "monotone_raw_v1")
    )
    if not requires_role_layout:
        return
    missing_role_fields = set(role_fields).difference(model_config)
    if missing_role_fields:
        raise OursError(
            "current-schema commercial checkpoint omits explicit role indices: "
            f"{sorted(missing_role_fields)}"
        )
    known_names = run.get("known_names")
    n_known = model_config.get("n_known")
    if (
        not isinstance(known_names, list | tuple)
        or len(known_names) != n_known
        or any(not isinstance(name, str) or not name.strip() for name in known_names)
        or len(set(known_names)) != len(known_names)
    ):
        raise OursError(
            "commercial checkpoint does not carry one valid known_names layout; coherent "
            "constraint and scenario replay cannot be reconstructed"
        )
    expected = _commercial_indices_from_roles(
        tuple(known_names), _checkpoint_commercial_drivers(run, tuple(known_names))
    )
    mismatched = {
        name: (model_config[name], value)
        for name, value in expected.items()
        if model_config[name] != value
    }
    if mismatched:
        raise OursError(
            f"checkpoint commercial role indices disagree with its declared drivers: "
            f"{mismatched}"
        )


def _validate_checkpoint_event_contract(
    model: CustomModel,
    run: dict,
    model_config: dict,
) -> int:
    """Bind replay's event recipe to constructor metadata and loaded support buffers."""
    support = model_config.get("event_max_support", 30)
    if type(support) is not int or not is_finite_real(support) or support < 1:
        raise OursError(f"checkpoint event_max_support is not a positive integer: {support!r}")
    run_support = run.get("event_max_support")
    if type(run_support) is not int or run_support != support:
        raise OursError(
            f"checkpoint replay event_max_support={run_support!r} contradicts executable "
            f"model_config support {support!r}"
        )
    if model.calendar is None:
        run_events = run.get("events")
        if run_events not in (None, [], ()):
            raise OursError(
                "checkpoint replay records events but its executable model has no calendar"
            )
        return support

    configured_windows = model_config.get("event_windows")
    if configured_windows is None:
        expected_lows = [-float(support)] * model.calendar.n_events
        expected_highs = [float(support)] * model.calendar.n_events
    else:
        if (
            not isinstance(configured_windows, list | tuple)
            or len(configured_windows) != model.calendar.n_events
            or any(
                not isinstance(window, list | tuple)
                or len(window) != 2
                or any(not is_finite_real(bound) for bound in window)
                for window in configured_windows
            )
        ):
            raise OursError("checkpoint event_windows do not match the executable calendar width")
        expected_lows = [float(window[0]) for window in configured_windows]
        expected_highs = [float(window[1]) for window in configured_windows]
    actual_lows = model.calendar.support_low.detach().cpu().tolist()
    actual_highs = model.calendar.support_high.detach().cpu().tolist()
    if actual_lows != expected_lows or actual_highs != expected_highs:
        raise OursError(
            "checkpoint calendar support buffers contradict model_config: "
            f"loaded={list(zip(actual_lows, actual_highs, strict=True))}, "
            f"configured={list(zip(expected_lows, expected_highs, strict=True))}"
        )

    run_events = run.get("events")
    event_names = list(_validate_event_recipe(run_events, label="checkpoint replay events"))
    if len(event_names) != model.calendar.n_events:
        raise OursError(
            f"checkpoint replay names {event_names} do not match executable calendar width "
            f"{model.calendar.n_events}"
        )
    run_windows = run.get("event_windows")
    if run_windows is None:
        replay_windows = tuple((-support, support) for _ in event_names)
    else:
        if (
            not isinstance(run_windows, dict)
            or any(name not in event_names for name in run_windows)
            or any(
                not isinstance(window, list | tuple)
                or len(window) != 2
                or any(not is_finite_real(bound) for bound in window)
                for window in run_windows.values()
            )
        ):
            raise OursError("checkpoint replay event_windows are malformed or name no event")
        replay_windows = tuple(
            tuple(run_windows.get(name, (-support, support))) for name in event_names
        )
    configured_tuple = tuple(zip(expected_lows, expected_highs, strict=True))
    if replay_windows != configured_tuple:
        raise OursError(
            "checkpoint replay event windows contradict its executable calendar: "
            f"replay={replay_windows}, configured={configured_tuple}"
        )
    return support


def _validate_checkpoint_execution_contract(
    model: CustomModel,
    run: dict,
    model_config: dict,
) -> None:
    """Bind current replay dimensions to the model constructor that loaded the weights."""
    horizon = model_config.get("horizon")
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
        or horizon > model.decoder.max_horizon
    ):
        raise OursError(
            f"checkpoint model horizon {horizon!r} is invalid for decoder ceiling "
            f"{model.decoder.max_horizon}"
        )
    run_horizon = run.get("horizon")
    if type(run_horizon) is not int or run_horizon != horizon:
        raise OursError(
            f"checkpoint replay horizon={run_horizon!r} contradicts executable model "
            f"horizon {horizon}"
        )
    encoder_days = model_config.get("encoder_days")
    run_encoder_days = run.get("encoder_days")
    if (
        type(encoder_days) is not int
        or encoder_days < 1
        or type(run_encoder_days) is not int
        or run_encoder_days != encoder_days
        or model.demand.encoder_days != encoder_days
    ):
        raise OursError(
            f"checkpoint replay/model encoder_days disagree: replay={run_encoder_days!r}, "
            f"config={encoder_days!r}, executable={model.demand.encoder_days!r}"
        )
    n_known = model_config.get("n_known")
    known_names = run.get("known_names")
    if (
        not isinstance(n_known, int)
        or isinstance(n_known, bool)
        or n_known < 1
        or model.decoder.known.in_features != n_known
        or not isinstance(known_names, list | tuple)
        or len(known_names) != n_known
        or any(not isinstance(name, str) or not name.strip() for name in known_names)
        or len(set(known_names)) != len(known_names)
    ):
        raise OursError(
            "checkpoint known-future namespace does not match its executable decoder width"
        )


def _json_values_equal_exact(observed: object, expected: object) -> bool:
    """Recursive value equality without bool/int/float or list/tuple aliasing."""
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return observed.keys() == expected.keys() and all(
            _json_values_equal_exact(observed[key], value) for key, value in expected.items()
        )
    if type(expected) in (list, tuple):
        return len(observed) == len(expected) and all(
            _json_values_equal_exact(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _validate_resolved_build_json_types(resolved: object) -> tuple[int, str]:
    """Validate the exact current resolved schema and return ``(version, price mode)``."""
    if type(resolved) is not dict:
        raise OursError("resolved build must be a canonical JSON mapping")
    if any(type(key) is not str for key in resolved):
        raise OursError("resolved build field names must be JSON strings")
    current_fields = set(ResolvedBuild.__dataclass_fields__).difference({"notes"})
    observed_fields = set(resolved)
    if observed_fields != current_fields:
        raise OursError(
            "resolved build fields differ from the current v2 canonical schema; unversioned "
            "and legacy schemas are unsupported: "
            f"missing={sorted(current_fields.difference(observed_fields))}, "
            f"extra={sorted(observed_fields.difference(current_fields))}"
        )
    version = resolved.get("schema_version")
    if version != 2 or type(version) is not int:
        raise OursError(
            f"resolved build schema_version must be the JSON integer 2, got {version!r}"
        )
    if resolved.get("schema") != RESOLVED_BUILD_SCHEMA:
        raise OursError(
            f"resolved build schema must be {RESOLVED_BUILD_SCHEMA!r}, got "
            f"{resolved.get('schema')!r}"
        )
    price_mode = resolved.get("price_response_mode")
    if price_mode not in ("none", "soft_penalty_v1", "monotone_raw_v1"):
        raise OursError(f"resolved build price_response_mode is unsupported: {price_mode!r}")
    occurrence_cap = resolved.get("price_occurrence_slope_max")
    size_cap = resolved.get("price_size_slope_max")
    if price_mode == "monotone_raw_v1":
        for field, value in (
            ("price_occurrence_slope_max", occurrence_cap),
            ("price_size_slope_max", size_cap),
        ):
            if type(value) is not float or not is_finite_real(value) or value <= 0.0:
                raise OursError(
                    f"resolved build {field} must be a finite positive JSON float under "
                    f"monotone_raw_v1: {value!r}"
                )
    elif (occurrence_cap, size_cap) != (None, None):
        raise OursError("resolved build price slope caps must both be null outside monotone_raw_v1")

    positive_ints = ("d_model", "attention_heads", "patch_size", "n_sample_paths")
    nonnegative_ints = ("factor_rank", "embedding_residual_dim")
    for field in positive_ints:
        value = resolved.get(field)
        if type(value) is not int or not is_finite_real(value) or value < 1:
            raise OursError(f"resolved build {field} must be a positive JSON integer: {value!r}")
    for field in nonnegative_ints:
        value = resolved.get(field)
        if type(value) is not int or not is_finite_real(value) or value < 0:
            raise OursError(f"resolved build {field} must be a nonnegative JSON integer: {value!r}")
    for field in ("static_variable_selection",):
        value = resolved.get(field)
        if type(value) is not bool:
            raise OursError(f"resolved build {field} must be a JSON boolean: {value!r}")
    for field in ("demand_branch_dropout", "covariate_dropout"):
        value = resolved.get(field)
        if type(value) is not float or not is_finite_real(value) or not 0.0 <= value < 1.0:
            raise OursError(
                f"resolved build {field} must be a finite JSON float in [0, 1): {value!r}"
            )
    zero_share = resolved.get("zero_share")
    if zero_share is not None and (
        type(zero_share) is not float
        or not is_finite_real(zero_share)
        or not 0.0 <= zero_share <= 1.0
    ):
        raise OursError(
            "resolved build zero_share must be null or a finite JSON float in [0, 1]: "
            f"{zero_share!r}"
        )
    for field in ("head", "topology", "emit"):
        value = resolved.get(field)
        if type(value) is not str or not value:
            raise OursError(f"resolved build {field} must be a nonempty JSON string: {value!r}")
    for field in ("factor_dynamics", "event_kernel", "input_normalisation"):
        value = resolved.get(field)
        if value is not None and (type(value) is not str or not value):
            raise OursError(
                f"resolved build {field} must be null or a nonempty JSON string: {value!r}"
            )
    event_support = resolved.get("event_max_support")
    if (
        type(event_support) is not list
        or len(event_support) != 2
        or any(type(value) is not int or not is_finite_real(value) for value in event_support)
    ):
        raise OursError(
            "resolved build event_max_support must be exactly two finite JSON integers: "
            f"{event_support!r}"
        )
    return version, price_mode


def _validate_checkpoint_resolved_build(
    model: CustomModel,
    record: dict,
    model_config: dict,
) -> None:
    """Refuse a checkpoint whose declared build hash contradicts its executable model."""
    configured_mode = _validate_checkpoint_model_config_schema(model_config)
    if configured_mode != model.price_response_mode:
        raise OursError(
            "checkpoint model_config price-response mode contradicts the executable model: "
            f"config={configured_mode!r}, model={model.price_response_mode!r}"
        )
    run = record.get("run", {})
    support = _validate_checkpoint_event_contract(model, run, model_config)
    resolved = run.get("resolved_build")
    if resolved is None:
        if model.price_response is not None:
            raise OursError(
                "a monotone_raw_v1 checkpoint requires a canonical v2 resolved build and hash; "
                "structural semantics cannot be inferred from weights or runtime arguments"
            )
        return
    _, price_mode = _validate_resolved_build_json_types(resolved)
    import json as _json

    try:
        canonical_payload = _json.dumps(
            resolved, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise OursError(f"checkpoint resolved_build is not finite canonical JSON: {exc}") from exc
    canonical_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    if run.get("resolved_build_hash") != canonical_hash:
        raise OursError(
            "checkpoint resolved_build_hash does not match its canonical resolved_build"
        )

    if model.price_response_mode != price_mode:
        raise OursError(
            "checkpoint executable price-response mode contradicts its resolved build: "
            f"executable={model.price_response_mode!r}, resolved={price_mode!r}"
        )
    configured_price = {
        "price_response_mode": model_config.get("price_response_mode"),
        "price_occurrence_slope_max": model_config.get("price_occurrence_slope_max"),
        "price_size_slope_max": model_config.get("price_size_slope_max"),
    }
    resolved_price = {
        "price_response_mode": price_mode,
        "price_occurrence_slope_max": resolved.get("price_occurrence_slope_max"),
        "price_size_slope_max": resolved.get("price_size_slope_max"),
    }
    price_config_mismatch = {
        name: (resolved_price[name], configured_price[name])
        for name in resolved_price
        if not _json_values_equal_exact(configured_price[name], resolved_price[name])
    }
    if price_config_mismatch:
        raise OursError(
            "checkpoint price-response constructor contradicts its resolved build: "
            f"{price_config_mismatch}"
        )

    if price_mode == "monotone_raw_v1":
        if model.price_response is None:
            raise OursError(
                "checkpoint resolved monotone_raw_v1 but has no structural response module"
            )
        expected_commercial = 1 if model.promo_index is not None else 0
        actual_commercial = 0 if model.commercial is None else model.commercial.n_channels
        structural_mismatch = {}
        if actual_commercial != expected_commercial:
            structural_mismatch["commercial_channels"] = (expected_commercial, actual_commercial)
        if model.commercial_known_indices != (None, None, model.promo_index):
            structural_mismatch["unrestricted_commercial_indices"] = (
                (None, None, model.promo_index),
                model.commercial_known_indices,
            )
        expected_inputs = (2,) if expected_commercial else ()
        if model.commercial_input_indices != expected_inputs:
            structural_mismatch["commercial_input_indices"] = (
                expected_inputs,
                model.commercial_input_indices,
            )
        for name, observed in (
            ("price_occurrence_slope_max", model.price_occurrence_slope_max),
            ("price_size_slope_max", model.price_size_slope_max),
        ):
            if not _json_values_equal_exact(observed, resolved.get(name)):
                structural_mismatch[name] = (resolved.get(name), observed)
        if structural_mismatch:
            raise OursError(
                "checkpoint structural route contradicts its resolved price contract: "
                f"{structural_mismatch}"
            )
    else:
        if model.price_response is not None:
            raise OursError(
                f"checkpoint resolved price_response_mode={price_mode!r} but contains a "
                "structural response module"
            )
        if price_mode == "soft_penalty_v1" and (
            model.commercial is None
            or model.price_index is None
            or model.price_available_index is None
        ):
            raise OursError(
                "checkpoint resolved soft_penalty_v1 but its executable model has no live "
                "price value/availability path"
            )

    actual_topology = "bottom_up_factor" if model.factor is not None else "bottom_up"
    actual_rank = 0 if model.factor is None else model.factor.rank
    expected_dynamics = resolved.get("factor_dynamics")
    if resolved.get("topology") == "bottom_up_factor" and expected_dynamics is None:
        expected_dynamics = "iid"
    actual_dynamics = None if model.factor is None else model.factor.factor_dynamics
    actual_event_kernel = "fixed_window" if model.calendar is not None else None
    actual_event_support = [-support, support]

    selection_widths = model_config.get("attribute_widths", ())
    if not isinstance(selection_widths, list | tuple):
        raise OursError("checkpoint attribute_widths must be a sequence")
    expected_selection = bool(resolved.get("static_variable_selection") and selection_widths)
    if (model.selection is not None) != expected_selection:
        raise OursError(
            "checkpoint static-variable-selection module contradicts its resolved/configured "
            f"applicability: resolved={resolved.get('static_variable_selection')!r}, "
            f"attribute_widths={selection_widths!r}, built={model.selection is not None}"
        )
    expected = {
        "head": model.head_kind,
        "d_model": model_config.get("d_model"),
        "attention_heads": model_config.get("n_heads", 4),
        "patch_size": model_config.get("patch_size", 7),
        "topology": actual_topology,
        "factor_rank": actual_rank,
        "factor_dynamics": actual_dynamics,
        "embedding_residual_dim": model.static.residual_dim,
        "n_sample_paths": run.get("n_paths"),
        "event_kernel": actual_event_kernel,
        "event_max_support": actual_event_support,
        "input_normalisation": "revin",
        "demand_branch_dropout": model.demand.dropout.p,
    }
    claimed = {
        "head": resolved.get("head"),
        "d_model": resolved.get("d_model"),
        "attention_heads": resolved.get("attention_heads"),
        "patch_size": resolved.get("patch_size"),
        "topology": resolved.get("topology"),
        "factor_rank": resolved.get("factor_rank"),
        "factor_dynamics": expected_dynamics,
        "embedding_residual_dim": resolved.get("embedding_residual_dim"),
        "n_sample_paths": resolved.get("n_sample_paths"),
        "event_kernel": resolved.get("event_kernel"),
        "event_max_support": resolved.get("event_max_support"),
        "input_normalisation": resolved.get("input_normalisation"),
        "demand_branch_dropout": resolved.get("demand_branch_dropout"),
    }
    mismatched = {
        name: (claimed[name], expected[name])
        for name in expected
        if not _json_values_equal_exact(claimed[name], expected[name])
    }
    if mismatched:
        raise OursError(
            f"checkpoint resolved build contradicts its executable model/run: {mismatched}"
        )

    if resolved.get("emit") not in ("sample_paths", "parameters"):
        raise OursError(f"checkpoint resolved emit is unsupported: {resolved.get('emit')!r}")
    paths_materialized = run.get("paths_materialized")
    if not isinstance(paths_materialized, bool):
        raise OursError("resolved checkpoint has no boolean paths_materialized execution evidence")
    if resolved.get("emit") != "sample_paths" and paths_materialized:
        raise OursError(
            f"checkpoint resolved emit={resolved.get('emit')!r} but records materialized paths"
        )

    controls = run.get("training_controls")
    if not isinstance(controls, dict):
        raise OursError(
            "resolved checkpoint has no training_controls evidence; its build hash cannot prove "
            "the resolved regularisation choices were executed"
        )
    control_expected = {
        "demand_branch_dropout": resolved.get("demand_branch_dropout"),
        "covariate_dropout": resolved.get("covariate_dropout"),
        "static_variable_selection": resolved.get("static_variable_selection"),
    }
    control_expected.update(
        {
            "price_response_mode": price_mode,
            "price_occurrence_slope_max": resolved.get("price_occurrence_slope_max"),
            "price_size_slope_max": resolved.get("price_size_slope_max"),
        }
    )
    control_mismatch = {
        name: (control_expected[name], controls.get(name))
        for name in control_expected
        if not _json_values_equal_exact(controls.get(name), control_expected[name])
    }
    price_weight = controls.get("price_penalty_weight")
    valid_price = (
        type(price_weight) is float
        and is_finite_real(price_weight)
        and (
            (price_mode == "soft_penalty_v1" and price_weight > 0.0)
            or (price_mode != "soft_penalty_v1" and price_weight == 0.0)
        )
    )
    if not valid_price:
        control_mismatch["price_penalty_weight"] = (
            ">0.0" if price_mode == "soft_penalty_v1" else "0.0",
            price_weight,
        )
    eligible_prices = controls.get("price_penalty_eligible_cells")
    if eligible_prices is not None and (
        type(eligible_prices) is not int
        or not is_finite_real(eligible_prices)
        or eligible_prices < 0
    ):
        control_mismatch["price_penalty_eligible_cells"] = ("nonnegative integer", eligible_prices)
    if price_mode == "soft_penalty_v1" and (
        type(eligible_prices) is not int or eligible_prices < 1
    ):
        control_mismatch["price_penalty_eligible_cells"] = (">=1", eligible_prices)
    if price_mode != "soft_penalty_v1" and eligible_prices != 0:
        control_mismatch["price_penalty_eligible_cells"] = (0, eligible_prices)
    valid_references = controls.get("price_response_valid_reference_cells")
    expected_references = ">=1" if price_mode == "monotone_raw_v1" else 0
    references_valid = (
        type(valid_references) is int
        and is_finite_real(valid_references)
        and (
            (price_mode == "monotone_raw_v1" and valid_references >= 1)
            or (price_mode != "monotone_raw_v1" and valid_references == 0)
        )
    )
    if not references_valid:
        control_mismatch["price_response_valid_reference_cells"] = (
            expected_references,
            valid_references,
        )
    run_n_paths = run.get("n_paths")
    if type(run_n_paths) is not int or not is_finite_real(run_n_paths) or run_n_paths < 1:
        control_mismatch["n_paths"] = ("positive integer", run_n_paths)
    if control_mismatch:
        raise OursError(
            f"checkpoint resolved build contradicts its recorded training controls: "
            f"{control_mismatch}"
        )


def load_checkpoint(path: str | pathlib.Path) -> tuple[CustomModel, dict]:
    """Rebuild the fitted model from a checkpoint. Returns `(model, record)`, model in eval.

    The model is constructed from the SAVED config, not from current defaults -- a default that
    changed since the run would otherwise rebuild a different architecture under the same
    weights, which either crashes (lucky) or loads and predicts with mismatched semantics.
    """
    try:
        record = torch.load(str(path), map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise OursError(
            "checkpoint cannot be read and is unversioned/legacy or corrupted; "
            f"the v2-only runtime requires schema {CHECKPOINT_SCHEMA!r}"
        ) from exc
    if type(record) is not dict or record.get("schema") != CHECKPOINT_SCHEMA:
        observed = record.get("schema") if isinstance(record, dict) else None
        raise OursError(
            f"checkpoint is unversioned/legacy or unsupported; the v2-only runtime requires "
            f"schema {CHECKPOINT_SCHEMA!r}, got {observed!r}"
        )
    model_config = dict(record["model_config"])
    configured_price_mode = _validate_checkpoint_model_config_schema(model_config)
    run = record.get("run", {})
    _validate_warm_start_checkpoint_record(run)
    resolved = run.get("resolved_build")
    if resolved is not None:
        _, resolved_price_mode = _validate_resolved_build_json_types(resolved)
        if configured_price_mode != resolved_price_mode:
            raise OursError(
                "checkpoint constructor price-response mode contradicts its resolved build: "
                f"config={configured_price_mode!r}, "
                f"resolved={resolved_price_mode!r}"
            )

    _validate_checkpoint_commercial_contract(model_config, run)
    model = CustomModel(**model_config)
    try:
        model.load_state_dict(record["state_dict"])
    except RuntimeError as exc:
        raise OursError(
            "checkpoint weights do not match the constructor bound to its resolved build"
        ) from exc
    model.eval()
    _validate_aggregate_projection_checkpoint(model, run, model_config)
    if run.get("oov_scheme") is not None:
        _validate_checkpoint_execution_contract(model, run, model_config)
    _validate_checkpoint_resolved_build(model, record, model_config)
    if run.get("oov_scheme") is not None:
        _validate_oov_parameters(model, run)
    return model, record


def _text_tensors(
    path: str | pathlib.Path | None, spark: SparkSession, series_ids: list[str]
) -> tuple[torch.Tensor | None, int, int]:
    """Frozen text embeddings aligned to the forecast index. Returns `(tensor, dim, missing)`.

    A series without an embedding gets ZEROS and is counted -- reported, not silent, because a
    zero vector is "no description", and a client whose descriptions mostly failed to embed
    should see that number rather than a quietly weaker identity branch.

    Raises:
        OursError: embeddings of inconsistent width -- concatenation would silently truncate or
            pad, and either way project garbage.
    """
    if path is None:
        return None, 0, 0
    rows = {r["series_id"]: r["embedding"] for r in spark.read.parquet(str(path)).collect()}
    widths = {len(v) for v in rows.values() if v is not None}
    if len(widths) != 1:
        raise OursError(
            f"text embeddings have inconsistent widths {sorted(widths)}; every series must "
            "share one embedding space or the projection is reading mixed coordinates"
        )
    dim = widths.pop()
    out = torch.zeros(len(series_ids), dim)
    missing = 0
    for i, sid in enumerate(series_ids):
        vector = rows.get(sid)
        if vector is None:
            missing += 1
        else:
            out[i] = torch.tensor(vector, dtype=torch.float32)
    return out, dim, missing


RESIDUAL_TABLES = (
    "static.residual",
    "head.dispersion_residual",
    "calendar.series_deviation",
    "seasonality.deviation",
)
"""Parameter prefixes for the per-series residual tables subject to weight decay. The
static.residual prefix also covers factorized product and store residual tables.
"""


def _identity_table_sizes(model: CustomModel) -> dict[str, int]:
    """Actual parameter-table row counts for the three cold-start identity roles."""
    dispersion = model.head.dispersion_residual
    if dispersion is None:
        raise OursError("the configured count head has no per-series dispersion table")
    product_rows = (
        0 if model.static.residual_product is None else model.static.residual_product.num_embeddings
    )
    store_sizes = []
    if model.static.residual_store is not None:
        store_sizes.append(model.static.residual_store.num_embeddings)
    if model.factor is not None:
        store_sizes.append(model.factor.n_stores)
    if len(set(store_sizes)) > 1:
        raise OursError(f"static and factor store namespaces disagree: {sorted(set(store_sizes))}")
    return {
        "series": dispersion.num_embeddings,
        "product": product_rows,
        "store": 0 if not store_sizes else store_sizes[0],
    }


def _canonical_identity_evidence(
    model: CustomModel,
    evidence: dict[str, object],
    *,
    checkpoint: bool = False,
) -> dict[str, list[int]]:
    """Validate and sort the rows that received optimiser evidence."""
    if not isinstance(evidence, dict):
        raise OursError("identity_evidence_rows must be a mapping")
    sizes = _identity_table_sizes(model)
    out: dict[str, list[int]] = {}
    for role, size in sizes.items():
        raw = evidence.get(role)
        if checkpoint and not isinstance(raw, list):
            raise OursError(f"checkpoint identity evidence for {role} must be a list")
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().tolist()
        if not isinstance(raw, list | tuple | set | frozenset):
            raise OursError(f"identity evidence for {role} must be a sequence of row integers")
        rows = list(raw)
        if any(isinstance(row, bool) or not isinstance(row, int) for row in rows):
            raise OursError(f"identity evidence for {role} contains a non-integer row")
        if len(set(rows)) != len(rows):
            raise OursError(f"identity evidence for {role} contains duplicate rows")
        if any(row < 0 or row >= size for row in rows):
            raise OursError(f"identity evidence for {role} has a row outside [0, {size}): {rows}")
        if role == "series" and size and not rows:
            raise OursError("identity role series has parameter rows but no optimiser evidence")
        if not size and rows:
            raise OursError(f"identity role {role} has no parameter table but records rows {rows}")
        out[role] = sorted(rows)
    return out


def _finalize_oov_parameters(
    model: CustomModel,
    evidence: dict[str, object],
) -> dict[str, list[int]]:
    """Turn every unevidenced identity row into an explicit pooled state.

    Random embedding initialisation is not pooling.  After the best checkpoint is restored, all
    per-identity deviation rows without a training update are set to exact zero.  Factor loadings
    are different: zero would remove the common shock, so unevidenced stores receive the exact
    CPU-computed mean of the evidenced rows.  This is performed before both in-process forecast
    and checkpoint save.
    """
    rows = _canonical_identity_evidence(model, evidence)
    sizes = _identity_table_sizes(model)

    def missing(role: str) -> list[int]:
        used = set(rows[role])
        return [row for row in range(sizes[role]) if row not in used]

    def zero_unevidenced(table: nn.Embedding | None, role: str) -> None:
        if table is not None and missing(role):
            table.weight.index_fill_(
                0, torch.tensor(missing(role), device=table.weight.device), 0.0
            )

    with torch.no_grad():
        zero_unevidenced(model.static.residual, "series")
        zero_unevidenced(model.static.residual_product, "product")
        zero_unevidenced(model.static.residual_store, "store")
        zero_unevidenced(model.head.dispersion_residual, "series")
        if model.calendar is not None:
            zero_unevidenced(model.calendar.series_deviation, "series")
        if model.seasonality is not None:
            zero_unevidenced(model.seasonality.deviation_occurrence, "series")
            zero_unevidenced(model.seasonality.deviation_size, "series")
        if model.factor is not None and missing("store"):
            if not rows["store"]:
                raise OursError(
                    "factor store table has no optimiser evidence from which to pool its "
                    "unevidenced loading rows"
                )
            evidence_index = torch.tensor(rows["store"], dtype=torch.long)
            absent_index = torch.tensor(
                missing("store"), dtype=torch.long, device=model.factor.theta.device
            )
            for table in (model.factor.loading_occurrence, model.factor.loading_size):
                # Compute on CPU in sorted-row order so the stored value can be independently
                # recomputed bit-for-bit regardless of the training accelerator.
                pooled = table.weight.detach().cpu().index_select(0, evidence_index).mean(dim=0)
                table.weight[absent_index] = pooled.to(
                    device=table.weight.device, dtype=table.weight.dtype
                )
    return rows


def _project_training_oov_parameters(
    model: CustomModel,
    evidence: dict[str, object],
    oov_rows: dict[str, int],
) -> None:
    """Keep reserved rows at their serving-time pooled values throughout optimisation.

    Declared null identities legitimately index OOV during training. Letting that row learn and
    zeroing it only after the final step makes shared weights optimise against a temporary model
    that is never served. This cheap projection touches only the three reserved rows after each
    optimiser step; other unevidenced namespace rows are finalised once at the end.
    """
    rows = _canonical_identity_evidence(model, evidence)
    sizes = _identity_table_sizes(model)
    if not isinstance(oov_rows, dict) or any(
        isinstance(oov_rows.get(role), bool)
        or not isinstance(oov_rows.get(role), int)
        or (sizes[role] and oov_rows[role] != sizes[role] - 1)
        for role in sizes
    ):
        raise OursError("training OOV rows do not match the executable identity tables")

    def zero_row(table: nn.Embedding | None, role: str) -> None:
        if table is not None:
            table.weight[oov_rows[role]].zero_()

    with torch.no_grad():
        zero_row(model.static.residual, "series")
        zero_row(model.static.residual_product, "product")
        zero_row(model.static.residual_store, "store")
        zero_row(model.head.dispersion_residual, "series")
        if model.calendar is not None:
            zero_row(model.calendar.series_deviation, "series")
        if model.seasonality is not None:
            zero_row(model.seasonality.deviation_occurrence, "series")
            zero_row(model.seasonality.deviation_size, "series")
        if model.factor is not None:
            if not rows["store"]:
                raise OursError("factor training has no evidenced store row from which to pool OOV")
            evidence_index = torch.tensor(rows["store"], dtype=torch.long)
            for table in (model.factor.loading_occurrence, model.factor.loading_size):
                pooled = table.weight.detach().cpu().index_select(0, evidence_index).mean(dim=0)
                table.weight[oov_rows["store"]].copy_(
                    pooled.to(device=table.weight.device, dtype=table.weight.dtype)
                )


def _zero_training_oov_gradients(
    model: CustomModel,
    oov_rows: dict[str, int],
) -> None:
    """Remove gradients for values projected away, before global gradient clipping."""

    def zero_row(table: nn.Embedding | None, role: str) -> None:
        if table is not None and table.weight.grad is not None:
            table.weight.grad[oov_rows[role]].zero_()

    zero_row(model.static.residual, "series")
    zero_row(model.static.residual_product, "product")
    zero_row(model.static.residual_store, "store")
    zero_row(model.head.dispersion_residual, "series")
    if model.calendar is not None:
        zero_row(model.calendar.series_deviation, "series")
    if model.seasonality is not None:
        zero_row(model.seasonality.deviation_occurrence, "series")
        zero_row(model.seasonality.deviation_size, "series")
    if model.factor is not None:
        zero_row(model.factor.loading_occurrence, "store")
        zero_row(model.factor.loading_size, "store")


def _validate_oov_parameters(
    model: CustomModel,
    run: dict,
) -> dict[str, list[int]]:
    """Prove a scheme-bearing checkpoint actually contains the pooled rows it claims."""
    if run.get("oov_scheme") != OOV_SCHEME:
        raise OursError(
            f"unsupported checkpoint oov_scheme {run.get('oov_scheme')!r}; expected "
            f"{OOV_SCHEME!r}"
        )
    actual_factor = model.factor is not None
    if run.get("factor_enabled") is not actual_factor:
        raise OursError(
            f"checkpoint factor_enabled={run.get('factor_enabled')!r} contradicts executable "
            f"factor presence {actual_factor}"
        )
    actual_topology = "bottom_up_factor" if actual_factor else "bottom_up"
    if run.get("topology") != actual_topology:
        raise OursError(
            f"checkpoint topology={run.get('topology')!r} contradicts executable "
            f"{actual_topology!r}"
        )
    factor_group_col = run.get("factor_group_col")
    if actual_factor:
        if not isinstance(factor_group_col, str) or not factor_group_col.strip():
            raise OursError("factor checkpoint has no nonblank factor_group_col replay role")
    elif factor_group_col is not None:
        raise OursError(
            "factorless checkpoint records a factor_group_col that replay cannot execute"
        )
    rows = _canonical_identity_evidence(model, run.get("identity_evidence_rows"), checkpoint=True)
    sizes = _identity_table_sizes(model)
    oov = run.get("oov_rows")
    if not isinstance(oov, dict):
        raise OursError("scheme-bearing checkpoint has no OOV row mapping")
    vocabularies = run.get("vocabularies")
    if not isinstance(vocabularies, dict):
        raise OursError("scheme-bearing checkpoint has no identity vocabularies")
    validated_vocabularies = {}
    for name in ("attribute_names", "product_levels", "store_levels", "factor_levels"):
        if name not in vocabularies:
            raise OursError(f"scheme-bearing checkpoint has no {name} vocabulary")
        validated_vocabularies[name] = _validated_vocabulary(vocabularies[name], name)
    if len(validated_vocabularies["attribute_names"]) != model.static.n_attributes:
        raise OursError(
            f"checkpoint attribute vocabulary has "
            f"{len(validated_vocabularies['attribute_names'])} columns but the executable "
            f"identity branch expects {model.static.n_attributes}"
        )

    series_ids = _validated_vocabulary(run.get("series_ids"), "series_ids")
    if oov.get("series") != len(series_ids):
        raise OursError(
            f"checkpoint series OOV row {oov.get('series')!r} does not follow its "
            f"{len(series_ids)} saved identities"
        )
    for role, size in sizes.items():
        if size and oov.get(role) != size - 1:
            raise OursError(
                f"checkpoint {role} OOV row {oov.get(role)!r} is not final table row {size - 1}"
            )
        if size and oov[role] in rows[role]:
            raise OursError(f"checkpoint {role} OOV row is incorrectly marked as trained")
    product_col = run.get("product_col")
    has_product_role = isinstance(product_col, str) and bool(product_col.strip())
    if product_col is not None and not has_product_role:
        raise OursError("checkpoint product_col must be a nonblank string or None")
    if sizes["product"] and not has_product_role:
        raise OursError(
            "checkpoint has an executable product identity table but no declared product_col; "
            "replay would silently route every series through row 0"
        )
    if not sizes["product"] and has_product_role:
        raise OursError(
            "checkpoint declares product_col but its executable model has no product identity "
            "table; replay would silently ignore the fitted role"
        )
    store_col = run.get("store_col")
    has_store_role = isinstance(store_col, str) and bool(store_col.strip())
    if store_col is not None and not has_store_role:
        raise OursError("checkpoint store_col must be a nonblank string or None")
    anonymous_store_role = (
        bool(sizes["store"])
        and model.static.factorised
        and model.factor is None
        and store_col is None
    )
    if sizes["store"] and not anonymous_store_role and not has_store_role:
        raise OursError(
            "checkpoint has an executable store/loading table but no declared store_col"
        )
    if not sizes["store"] and has_store_role:
        raise OursError(
            "checkpoint declares store_col but its executable model has no store/loading table"
        )
    for role, levels_key in (("product", "product_levels"), ("store", "store_levels")):
        levels = validated_vocabularies[levels_key]
        # A product-only factorised identity still needs the additive store half. With no store
        # role, row 0 is a deliberate anonymous fallback and row 1 is OOV; there is no named
        # store vocabulary entry for row 0. Declared roles have the ordinary levels-then-OOV
        # layout and must account for every preceding row.
        anonymous_store = role == "store" and anonymous_store_role
        if anonymous_store:
            if levels or oov[role] != 1:
                raise OursError(
                    "checkpoint absent-store fallback must have no named levels and use row 1 "
                    "as OOV after its anonymous row 0"
                )
        elif sizes[role] and len(levels) != oov[role]:
            raise OursError(
                f"checkpoint {role} vocabulary has {len(levels)} levels "
                f"but its OOV/table contract requires {oov[role]}"
            )

    def missing(role: str) -> list[int]:
        used = set(rows[role])
        return [row for row in range(sizes[role]) if row not in used]

    def require_zero(table: nn.Embedding | None, role: str, name: str) -> None:
        if table is None or not missing(role):
            return
        values = table.weight.detach().cpu()[missing(role)]
        if bool(torch.count_nonzero(values)):
            raise OursError(f"checkpoint {name} has nonzero unevidenced rows under {OOV_SCHEME}")

    require_zero(model.static.residual, "series", "static.residual")
    require_zero(model.static.residual_product, "product", "static.residual_product")
    require_zero(model.static.residual_store, "store", "static.residual_store")
    require_zero(model.head.dispersion_residual, "series", "head.dispersion_residual")
    if model.calendar is not None:
        require_zero(model.calendar.series_deviation, "series", "calendar.series_deviation")
    if model.seasonality is not None:
        require_zero(
            model.seasonality.deviation_occurrence, "series", "seasonality.deviation_occurrence"
        )
        require_zero(model.seasonality.deviation_size, "series", "seasonality.deviation_size")
    if model.factor is not None and missing("store"):
        if not rows["store"]:
            raise OursError(
                "checkpoint factor store table has no evidenced row from which its pooled "
                "loading could have been derived"
            )
        evidence_index = torch.tensor(rows["store"], dtype=torch.long)
        for name, table in (
            ("loading_occurrence", model.factor.loading_occurrence),
            ("loading_size", model.factor.loading_size),
        ):
            weight = table.weight.detach().cpu()
            expected = weight.index_select(0, evidence_index).mean(dim=0)
            actual = weight[missing("store")]
            if not torch.equal(actual, expected.expand_as(actual)):
                raise OursError(
                    f"checkpoint factor.{name} unevidenced rows are not the exact evidenced "
                    f"row mean required by {OOV_SCHEME}"
                )
    return rows


def parameter_groups(
    model: torch.nn.Module, weight_decay: float = 1e-4
) -> tuple[list[dict], list[str]]:
    """AdamW groups implementing the residual-only decay. Returns `(groups, decayed_names)`.

    The names are returned so the run can PRINT what is being shrunk -- a decay group selected
    by name-prefix drifts silently when a module is renamed, and an empty group would quietly
    reinstate the old uniform behaviour with the opposite sign.
    """
    decay, rest, decayed_names = [], [], []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in RESIDUAL_TABLES):
            decay.append(param)
            decayed_names.append(name)
        else:
            rest.append(param)
    return (
        [{"params": rest, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}],
        decayed_names,
    )


def _static_real_tensors(
    panel: Panel,
    spark: SparkSession,
    series_ids: list[str],
) -> tuple[torch.Tensor, tuple[str, ...], dict[str, tuple[float, float]]]:
    """Declared NUMERIC statics, standardised on this panel. Returns `(values, names, stats)`.

    `ForecastDataSpec.static_reals` had no path at all -- a declared pack weight was silently
    ignored, which the audit recorded as MISSING. Standardised rather than raw because the
    identity projection shares one weight matrix with one-hot columns of unit scale, and a
    shelf capacity in the thousands would drown them; per-column `(mean, std)` is returned so
    a checkpoint consumer can state exactly what transformation the model saw. A CONSTANT
    column standardises to zeros -- carried, not refused, because a constant is a client fact
    ("every series in this extract is the same pack size") and refusing it would make the same
    declaration valid on one extract and invalid on another. A missing value is refused: an
    invented static is an invented product property.
    """
    frame = spark.read.parquet(panel.path).select(
        F.col(panel.series_col).alias("series_id"),
        *[F.col(c).cast("double").alias(c) for c in panel.static_reals],
    )
    per_series = {
        r["series_id"]: r
        for r in frame.groupBy("series_id")
        .agg(*[F.first(c).alias(c) for c in panel.static_reals])
        .collect()
    }
    out = torch.zeros(len(series_ids), len(panel.static_reals))
    for j, column in enumerate(panel.static_reals):
        values = []
        for sid in series_ids:
            row = per_series.get(sid)
            value = None if row is None else row[column]
            if value is None:
                raise OursError(
                    f"static real {column!r} is missing for series {sid!r}; a static the panel "
                    "declares must exist for every series -- inventing one would invent a "
                    "product property"
                )
            values.append(float(value))
        column_values = torch.tensor(values)
        out[:, j] = column_values
    means = out.mean(dim=0)
    stds = out.std(dim=0, unbiased=False).clamp_min(1e-12)
    standardised = (out - means) / stds
    constant = stds <= 1e-12
    if bool(constant.any()):
        standardised[:, constant] = 0.0
    stats = {
        column: (float(means[j]), float(stds[j])) for j, column in enumerate(panel.static_reals)
    }
    return standardised, tuple(panel.static_reals), stats


def _vocabulary_spark(panel: Panel, spark: SparkSession, column: str) -> list[str]:
    """Sorted distinct levels of a panel column -- a GLOBAL vocabulary, small by construction
    (levels, not rows), so every shard maps identities onto the same embedding rows."""
    rows = (
        spark.read.parquet(panel.path)
        .select(F.col(column).cast("string").alias("_v"))
        .distinct()
        .collect()
    )
    return sorted(str(r["_v"]) for r in rows if r["_v"] is not None)


def _series_vocabulary(panel: Panel, spark: SparkSession) -> list[str]:
    """Sorted distinct series ids. O(n) STRINGS -- the one global structure the sharded path
    keeps whole, because the per-series embedding tables are global by definition. Categorically
    below the tensor blow-up the sharding bounds ([n x encoder_days x channels] and friends);
    a truly id-free variant would hash ids to rows and is recorded as future work, not implied."""
    rows = (
        spark.read.parquet(panel.path)
        .select(F.col(panel.series_col).alias("_s"))
        .distinct()
        .collect()
    )
    return sorted(str(r["_s"]) for r in rows)


def shard_plan(
    panel: Panel,
    spark: SparkSession,
    *,
    grouping: str | None,
    product: str | None,
    max_series: int | None,
) -> list[str | None]:
    """Hash-shard predicates that keep every factor block whole. `[None]` means one shard.

    The scale contract is bounded MEMORY, and the bound has to hold at the collect: a shard is
    a set of complete `(factor_group, product)` blocks selected by a hash of the BLOCK key, so
    the per-block bound and the block-batching stay exact -- a block can never straddle two
    shards, because its every row hashes identically. With no grouping declared, blocks are
    single series and the series id is the key. The cap is soft in the same sense as the batch
    cap: hashing balances shards in expectation, not by exact count.
    """
    if max_series is None:
        return [None]
    if isinstance(max_series, bool) or not isinstance(max_series, int) or max_series < 1:
        raise OursError(f"max_series_per_shard must be a positive integer, got {max_series!r}")
    n_series = spark.read.parquet(panel.path).select(panel.series_col).distinct().count()
    n_shards = -(-n_series // max_series)
    if n_shards <= 1:
        return [None]
    # A product is part of the factor-block key only when a factor group exists.  With no
    # grouping the topology is rank-zero bottom-up, so each series is its own independent block
    # exactly as the docstring promises.  Hashing a repeated product here can funnel an entire
    # catalogue into one shard and defeats the memory bound.
    keys = [c for c in (grouping, product) if c is not None] if grouping is not None else []
    if keys:
        # Query-time composite grouping treats a missing semantic key as unknown for THAT
        # series, not as one shared empty-string block. Mirror that identity here: any null
        # component carries the series id into the hash, while complete non-null tuples remain
        # whole. Otherwise an all-null catalogue hashes into one shard and defeats the cap.
        series_sql = f"cast(`{panel.series_col}` as string)"
        key_sql = ", ".join(
            f"case when `{c}` is null "
            f"then concat('__missing_series__:', {series_sql}) "
            f"else concat('__value__:', cast(`{c}` as string)) end"
            for c in keys
        )
    else:
        key_sql = f"cast(`{panel.series_col}` as string)"
    return [f"pmod(xxhash64({key_sql}), {n_shards}) = {k}" for k in range(n_shards)]


def _level_priors(
    panel: Panel, spark: SparkSession, origin: str, encoder_days: int, category_col: str | None
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """Category and fallback priors measured GLOBALLY in Spark for the sharded path.

    In one shard the in-tensor level statistics see only that shard's series, and a cold start
    would inherit its shard's accident rather than its category's rate. These aggregates run
    where the data lives and return two small dicts; in the single-shard case they equal the
    in-tensor statistics exactly, because every series is present either way.
    """
    lo = F.date_sub(F.to_date(F.lit(origin)), encoder_days - 1)
    df = spark.read.parquet(panel.path).filter(
        (F.to_date(F.col(panel.time_col)) <= F.to_date(F.lit(origin)))
        & (F.to_date(F.col(panel.time_col)) >= lo)
    )
    y = F.col(panel.target_col).cast("double")
    if panel.status_prob_cols is not None:
        p_unobs = F.col(panel.status_prob_cols["unobserved"]).cast("double")
        p_lower = F.col(panel.status_prob_cols["lower_bound"]).cast("double")
        p_exact = F.col(panel.status_prob_cols["exact"]).cast("double")
        modal_unobs = (p_unobs >= p_exact) & (p_unobs >= p_lower)
        modal_lower = (~modal_unobs) & (p_lower >= p_exact)
        informative = (~modal_unobs) & (~(modal_lower & (y <= 0)))
    elif panel.exposure_col is not None:
        state = F.col(panel.exposure_col)
        informative = (state == "exact") | ((state == "lower_bound") & (y > 0))
    else:
        informative = F.lit(True)
    if panel.status_prob_cols is not None:
        exact_size = (~modal_unobs) & (~modal_lower)
    elif panel.exposure_col is not None:
        exact_size = F.col(panel.exposure_col) == "exact"
    else:
        exact_size = F.lit(True)
    grouped = (
        df.groupBy(F.col(category_col).cast("string").alias("_level"))
        if category_col is not None
        else df.groupBy()
    )
    stats = grouped.agg(
        F.sum(F.when(informative & (y > 0), 1).otherwise(0)).alias("_sell"),
        F.sum(F.when(informative, 1).otherwise(0)).alias("_days"),
        F.sum(F.when(exact_size & (y > 0), y).otherwise(0.0)).alias("_pos_total"),
        F.sum(F.when(exact_size & (y > 0), 1).otherwise(0)).alias("_pos_count"),
    ).collect()
    # EXACTLY the in-tensor formulas of `_shrunk_baselines`: Jeffreys panel priors, and
    # category priors partially pooled toward them. In the single-shard case both computations
    # see identical inputs, so any difference here would be drift between the two paths -- and
    # the assembly-parity test pins them.
    # A global aggregate over an empty time window is one row of nulls (where a grouped
    # aggregate is zero rows). Treat both shapes as zero evidence so the caller reaches its
    # deliberate "no full encoder window" refusal instead of crashing in int(None).
    total_sell = sum(int(r["_sell"] or 0) for r in stats)
    total_days = sum(int(r["_days"] or 0) for r in stats)
    total_pos = sum(float(r["_pos_total"] or 0.0) for r in stats)
    total_pos_count = sum(int(r["_pos_count"] or 0) for r in stats)
    global_rate = (total_sell + 0.5) / (total_days + 1.0)
    global_size = (total_pos + 1.0) / (total_pos_count + 1.0)
    priors = {}
    if category_col is None:
        return priors, (global_rate, global_size)
    for r in stats:
        if r["_level"] is None:
            continue
        rate = (int(r["_sell"]) + global_rate * 4.0) / (int(r["_days"]) + 4.0)
        size = (float(r["_pos_total"]) + global_size * 2.0) / (int(r["_pos_count"]) + 2.0)
        priors[str(r["_level"])] = (rate, size)
    return priors, (global_rate, global_size)


def _series_levels(
    panel: Panel, spark: SparkSession, column: str, series_predicate: str | None = None
) -> dict[str, str]:
    """Each series' level of ``column``, after proving it is actually static.

    A bare Spark ``first`` is not validation: if a factor/category role changes within a series,
    its answer depends on partition order and can import a future hierarchy correction into an
    earlier origin. Semantic/index roles therefore obey the same one-value contract as declared
    static features. `series_predicate` is the shard pushdown: the collect below is the memory
    step, and the bound has to hold here.
    """
    source = spark.read.parquet(panel.path)
    if series_predicate is not None:
        source = source.filter(F.expr(series_predicate))
    frame = source.select(
        F.col(panel.series_col).alias("series_id"), F.col(column).cast("string").alias("_g")
    )
    grouped = frame.groupBy("series_id").agg(
        F.first("_g", ignorenulls=False).alias("_g"),
        F.countDistinct(F.struct(F.col("_g").isNull(), F.col("_g"))).alias("_n"),
    )
    conflict = grouped.filter(F.col("_n") > 1).select("series_id").limit(1).collect()
    if conflict:
        raise OursError(
            f"semantic/index column {column!r} changes within series "
            f"{conflict[0]['series_id']!r}; assigning it with Spark first() would be "
            "partition-order dependent"
        )
    return {r["series_id"]: r["_g"] for r in grouped.select("series_id", "_g").collect()}


def _validate_series_role_constancy(
    panel: Panel,
    spark: SparkSession,
    columns: tuple[str | None, ...],
) -> None:
    """Prove semantic roles are static globally before role values define shard membership.

    A shard-local check is too late for the columns used in the shard hash: if one series changes
    factor group or product, its rows can enter two predicates and each local view sees one
    apparently constant value.  This aggregate returns at most one row to the driver and therefore
    preserves the bounded-memory contract while closing that split-series path.
    """
    roles = tuple(dict.fromkeys(column for column in columns if column is not None))
    if not roles:
        return
    counts = [
        F.countDistinct(F.struct(F.col(column).isNull(), F.col(column).cast("string"))).alias(
            f"_role_{i}"
        )
        for i, column in enumerate(roles)
    ]
    grouped = (
        spark.read.parquet(panel.path)
        .groupBy(F.col(panel.series_col).alias("series_id"))
        .agg(*counts)
    )
    changed_condition = F.lit(False)
    for i in range(len(roles)):
        changed_condition = changed_condition | (F.col(f"_role_{i}") > 1)
    conflict = grouped.filter(changed_condition).limit(1).collect()
    if conflict:
        row = conflict[0]
        changed = [column for i, column in enumerate(roles) if row[f"_role_{i}"] > 1]
        raise OursError(
            f"semantic/index column(s) {changed} change within series {row['series_id']!r}; "
            "one series cannot be assigned to multiple identity or factor shards"
        )


def _grouping(panel: Panel, spark: SparkSession, series_ids: list[str], column: str | None):
    """Map a static/index role to known rows plus one reserved OOV row.

    Returns ``(index, known_cardinality, note, vocabulary, raw_values)``.  A null value maps to
    ``known_cardinality`` -- never row zero -- while ``raw_values`` lets factor-path grouping keep
    parameter pooling separate from semantic equality.

    This helper is also used for product and store identity. Its one-level `None` result is only a
    placeholder for those optional indices; it does **not** decide factor topology. `fit_predict`
    handles an absent factor-group declaration before calling it and constructs independent row
    blocks with a rank-zero model.
    """
    if column is None:
        return (
            torch.zeros(len(series_ids), dtype=torch.long),
            1,
            "index role NOT DECLARED: one fallback identity level",
            [],
            None,
        )
    per_series = _series_levels(panel, spark, column)
    levels = sorted({v for v in per_series.values() if v is not None})
    at = {level: i for i, level in enumerate(levels)}
    index = torch.full((len(series_ids),), len(levels), dtype=torch.long)
    for i, sid in enumerate(series_ids):
        level = per_series.get(sid)
        if level is not None:
            index[i] = at[level]
    return (
        index,
        len(levels),
        f"factor group: {column} ({len(levels)} known levels)",
        levels,
        per_series,
    )


def _known_future(
    origin: str,
    horizon: int,
    n_series: int,
    commercial_horizon: tuple[torch.Tensor, torch.Tensor] | None,
    event_horizon: tuple[torch.Tensor, torch.Tensor] | None,
    event_names: tuple[str, ...],
    cycles: tuple[tuple[str, float, int], ...] = SEASONAL_CYCLES,
    commercial_drivers: frozenset[str] = frozenset(),
    step_days: int = 1,
) -> KnownFuture:
    """What is KNOWN about each horizon day at forecast time, as a named-column tensor.

    Day of week and day of month are known years ahead. Price and promotion come from the
    commercial builder's horizon rows -- a PLANNED price, which is legitimate known-future when
    the client plans prices. Event distance is per DAY, which is the point: the distance to
    Christmas is not the same on day 1 and day 28.

    What may never enter: demand, a competitor's price, an unannounced promotion. No runtime check
    can catch that, because the tensor looks identical either way -- which is why the contents are
    written down here rather than assembled somewhere convenient.
    """
    d0 = dt.date.fromisoformat(origin)
    calendar = torch.zeros(n_series, horizon, len(CALENDAR_COLUMNS))
    width = sum(2 * h for _, _, h in cycles)
    cyclic = torch.zeros(n_series, horizon, width)
    for step in range(horizon):
        day = d0 + dt.timedelta(days=(step + 1) * step_days)
        calendar[:, step, 0] = day.weekday() / 6.0
        calendar[:, step, 1] = (day.day - 1) / 30.0
        calendar[:, step, 2] = float(day.weekday() >= 5)
        at = 0
        for _, period, harmonics in cycles:
            for value in cycle_terms(day, period, harmonics):
                cyclic[:, step, at] = value
                at += 1

    parts, names = [calendar], list(CALENDAR_COLUMNS)
    # Calendar and event columns exist only when their roles exist. Commercial columns are the
    # exception: their fixed positional branch layout includes unavailable placeholders, which
    # must not become scenario drivers merely because they have names.
    declared_drivers = set(CALENDAR_COLUMNS)
    if width:
        parts.append(cyclic)
        for name, _, harmonics in cycles:
            columns = cycle_columns(name, harmonics)
            names.extend(columns)
            declared_drivers.update(columns)
    if commercial_horizon is not None:
        invalid = sorted(set(commercial_drivers).difference(commercial.CHANNELS))
        if invalid:
            raise OursError(f"unknown declared commercial driver(s): {invalid}")
        values, available = commercial_horizon
        parts.extend((values, available))
        names.extend(commercial.CHANNELS)
        # AVAILABILITY IS A COLUMN PER CHANNEL, exactly as events carry `_present`. The values
        # are zero-filled where missing -- the tensor is dense -- and a zero price without its
        # flag reads as "this product is free". The branch got its mask from day one; the
        # decoder, through which the horizon's prices actually act, never did.
        names.extend(f"{c}_available" for c in commercial.CHANNELS)
        declared_drivers.update(commercial_drivers)
    if event_horizon is not None:
        distance, present = event_horizon
        parts.extend((distance, present))
        event_distance_names = tuple(f"{n}_distance" for n in event_names)
        names.extend(event_distance_names)
        # Presence is a SEPARATE column per event, not a refinement of the distance. See
        # `_event_tensors`: an out-of-support event is written as distance 0, which is also what
        # "the event is today" looks like.
        event_present_names = tuple(f"{n}_present" for n in event_names)
        names.extend(event_present_names)
        declared_drivers.update(event_distance_names)
        declared_drivers.update(event_present_names)
    return KnownFuture(torch.cat(parts, dim=-1), tuple(names), frozenset(declared_drivers))


# Entries of the input dict that are NOT per-series. Everything else is indexed when a minibatch
# is taken, and an unknown key RAISES rather than being passed through -- see `_subset`.
NON_PER_SERIES = frozenset({"n_draws", "generator"})


@dataclass(frozen=True)
class _SparkResumePlan:
    """The bounded local Spark settings needed only to serialize fit artifacts."""

    master: str
    config: tuple[tuple[str, str], ...]


def _release_spark_for_torch_training(spark: SparkSession) -> _SparkResumePlan:
    """Release the feature-engineering JVM before the pure-Torch epoch loop.

    Feature assembly returns ordinary host tensors.  Retaining Spark thereafter needlessly
    competes with MPS for unified memory.  The forecast and optional paths remain Parquet
    artifacts, so their small writer session is reconstructed after CPU inference.
    """
    context = spark.sparkContext
    plan = _SparkResumePlan(
        master=context.master,
        config=tuple(
            sorted(
                (key, value)
                for key, value in context.getConf().getAll()
                if key
                in {
                    "spark.driver.memory",
                    "spark.sql.shuffle.partitions",
                    "spark.python.worker.reuse",
                }
            )
        ),
    )
    spark.stop()
    SparkContext._active_spark_context = None
    print("  feature assembly complete; Spark released for Torch training", flush=True)
    return plan


def _resume_spark_after_torch_training(plan: _SparkResumePlan) -> SparkSession:
    """Open a fresh, bounded local Spark writer after fitting has completed."""
    local_spark_config = resources.configure_local_spark_runtime()
    builder = SparkSession.builder.master(plan.master).appName("forecast-multibranch-output")
    for key, value in plan.config:
        builder = builder.config(key, value)
    for key, value in local_spark_config.items():
        builder = builder.config(key, value)
    spark = builder.config("spark.ui.showConsoleProgress", "false").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _stop_reopened_spark(spark: SparkSession) -> None:
    """Close the writer session even when Parquet serialization raises."""
    with contextlib.suppress(Exception):
        spark.stop()
    SparkContext._active_spark_context = None


def _subset(tensors: dict, idx: torch.Tensor) -> dict:
    """Index every per-series tensor by `idx`. Unconditionally.

    **The default is to INDEX, and an unrecognised key is an error.** An earlier version listed the
    per-series keys instead, so adding the commercial and calendar tensors left them full-size in
    every minibatch -- a [512] batch against a [3049] feature matrix, which happened to fail loudly
    only because the shapes collided inside `torch.stack`. Had the branch pooled instead of
    stacked, it would have paired each series with another series' price and trained perfectly
    happily.

    A second version indexed CONDITIONALLY -- skipping any tensor whose first dimension already
    equalled `len(idx)`, as a convenience for pre-aligned inputs. That guess is wrong exactly
    when a caller's index covers every row: the tensor passes through in storage order while the
    caller's `target[idx]` is permuted, so the model trains on shuffled targets against
    unshuffled features and nothing fails. Alignment now happens where the alignment is KNOWN
    (`_window_tensors`), and this function has one meaning.

    Listing the exceptions rather than the members is what makes the first bug unrepeatable: a
    new per-series tensor is indexed automatically, and a new non-series one has to be declared.

    Raises:
        OursError: a key is neither a tensor nor declared in `NON_PER_SERIES`.
    """
    out = {}
    for key, value in tensors.items():
        if key in NON_PER_SERIES or value is None:
            out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value[idx]
        else:
            raise OursError(
                f"{key!r} is neither a tensor nor declared in NON_PER_SERIES. A value that is "
                "silently passed through a minibatch pairs every series with the wrong row, and "
                "the model trains without complaint"
            )
    if "group_index" in out:
        # A query/minibatch retains global group labels after row selection. Only equality is
        # meaningful locally; densifying prevents `max(label)+1` allocations for absent groups.
        out["group_index"], _ = _dense_group_index(out["group_index"])
    return out


def _to_model_device(tensors: dict, device: torch.device) -> dict:
    """Transfer one already-indexed physical batch, never the full M5 tensor dataset."""
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in tensors.items()
    }


def _model_device(model: nn.Module) -> torch.device:
    """Return the model's sole execution device, refusing a parameter-free model."""
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise OursError("cannot stage a batch for a parameter-free model") from exc


def _pool_unevidenced_indices(
    inputs: dict,
    evidence: dict[str, object],
    oov_rows: dict[str, int],
) -> dict:
    """Route validation-only parameter identities through the model that will be served."""
    out = dict(inputs)
    for key, role in (
        ("series_index", "series"),
        ("product_index", "product"),
        ("store_index", "store"),
    ):
        index = out.get(key)
        if not isinstance(index, torch.Tensor):
            continue
        trained = evidence.get(role)
        if not isinstance(trained, list | tuple | set | frozenset):
            raise OursError(f"identity evidence for {role} must be a sequence")
        pooled = index.clone()
        if trained:
            known = torch.tensor(sorted(trained), dtype=index.dtype, device=index.device)
            pooled[~torch.isin(index, known)] = oov_rows[role]
        else:
            pooled.fill_(oov_rows[role])
        out[key] = pooled
    return out


PATH_ARTIFACT_VERSION = 2
_PATH_COLUMNS = {
    "artifact_version",
    "series_id",
    "series_index",
    "origin",
    "horizon_start",
    "horizon_end",
    "n_paths",
    "n_series",
    "horizon",
    "encoding",
    "payload_sha256",
    "demand_bytes",
}

_PATH_ENCODING = "int64_le_c_order"
_PATH_ROWS_PER_SPARK_WRITE = 128


def _write_paths(paths, series_ids, origin, horizon, sample, spark, out_dir) -> None:
    """Persist one binary ``[n_paths, horizon]`` matrix per series.

    Gate 4 writes 1,000 paths for as many as 2,000 series. The former row-per-series-per-path
    format constructed two million Python rows and 56 million boxed Python floats, then collected
    them all again on read. Version 2 has exactly ``n_series`` rows. Each carries one
    little-endian int64 C-order payload plus a checksum; the loader streams those rows and never
    materialises a Python object per draw.

    Integer storage is part of the contract: these are paths from a count distribution, not
    interpolated pseudo-demand. ``demand_bytes`` for series ``i`` encodes
    ``paths[:, i, :]``; column zero is explicitly ``origin + 1``.
    """
    if paths.dim() != 3:
        raise OursError(f"paths must be [n_paths, n_series, horizon], got {tuple(paths.shape)}")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise OursError(f"horizon must be a positive integer, got {horizon!r}")
    if paths.shape[0] < 1 or paths.shape[2] != horizon:
        raise OursError(
            f"paths shape {tuple(paths.shape)} disagrees with horizon={horizon} or has no draws"
        )
    if paths.shape[1] != len(series_ids):
        raise OursError(
            f"paths contain {paths.shape[1]} series but {len(series_ids)} ids were supplied"
        )
    if (
        not isinstance(sample, int)
        or isinstance(sample, bool)
        or not 1 <= sample <= len(series_ids)
    ):
        raise OursError(f"sample must lie in [1, {len(series_ids)}], got {sample!r}")
    if any(not isinstance(s, str) or not s.strip() for s in series_ids):
        raise OursError("every persisted series id must be a non-blank string")
    if len(set(series_ids)) != len(series_ids):
        raise OursError("cannot persist paths with duplicate series ids")
    try:
        origin_day = dt.date.fromisoformat(origin)
    except (TypeError, ValueError) as exc:
        raise OursError(f"path origin must be an ISO date, got {origin!r}") from exc
    if origin != origin_day.isoformat():
        raise OursError(f"path origin must be the canonical ISO date, got {origin!r}")

    values = paths.detach().cpu()
    if values.dtype == torch.bool or values.is_complex():
        raise OursError("paths must contain numeric integer demand counts, not bool or complex")
    if (
        not bool(torch.isfinite(values).all())
        or bool((values < 0).any())
        or (values.is_floating_point() and not bool((values == values.round()).all()))
    ):
        raise OursError("paths must contain only finite, non-negative integer demand")
    exceeds_int64 = (
        (values >= float(2**63)).any()
        if values.is_floating_point()
        else (values > torch.iinfo(torch.int64).max).any()
    )
    if bool(exceeds_int64):
        raise OursError("a path demand value exceeds the int64 artifact representation")

    # The caller normally passes an already hash-sampled tensor. Sorting here gives the artifact
    # an explicit stable row order and `series_index` makes that order independently checkable.
    order = sorted(range(len(series_ids)), key=lambda i: series_ids[i])[:sample]
    n_paths, n_series = paths.shape[0], len(order)
    horizon_start = (origin_day + dt.timedelta(days=1)).isoformat()
    horizon_end = (origin_day + dt.timedelta(days=horizon)).isoformat()
    rows = []
    for series_index, i in enumerate(order):
        matrix = values[:, i, :].to(torch.int64).contiguous().numpy().astype("<i8", copy=False)
        payload = matrix.tobytes(order="C")
        rows.append(
            {
                "artifact_version": PATH_ARTIFACT_VERSION,
                "series_id": series_ids[i],
                "series_index": series_index,
                "origin": origin,
                "horizon_start": horizon_start,
                "horizon_end": horizon_end,
                "n_paths": n_paths,
                "n_series": n_series,
                "horizon": horizon,
                "encoding": _PATH_ENCODING,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "demand_bytes": payload,
            }
        )
    schema = (
        "artifact_version int, series_id string, series_index int, origin string, "
        "horizon_start string, horizon_end string, n_paths int, n_series int, horizon int, "
        "encoding string, payload_sha256 string, demand_bytes binary"
    )
    # A local Python collection becomes a PythonRDD. At the registered 2,000 x 1,000 x 28
    # geometry that fanned 448 MB through ten Python-worker sockets concurrently and exhausted
    # the host socket buffers after training had completed. Spark 4's native Arrow-table input
    # transfers the same typed rows to the JVM as one Arrow stream, so the Parquet writer remains
    # Spark-native without involving Python workers. The artifact schema and payload bytes are
    # unchanged.
    import pyarrow as pa

    arrow_schema = pa.schema(
        [
            ("artifact_version", pa.int32()),
            ("series_id", pa.string()),
            ("series_index", pa.int32()),
            ("origin", pa.string()),
            ("horizon_start", pa.string()),
            ("horizon_end", pa.string()),
            ("n_paths", pa.int32()),
            ("n_series", pa.int32()),
            ("horizon", pa.int32()),
            ("encoding", pa.string()),
            ("payload_sha256", pa.string()),
            ("demand_bytes", pa.binary()),
        ]
    )
    for offset in range(0, len(rows), _PATH_ROWS_PER_SPARK_WRITE):
        table = pa.Table.from_pylist(
            rows[offset : offset + _PATH_ROWS_PER_SPARK_WRITE],
            schema=arrow_schema,
        )
        frame = spark.createDataFrame(table, schema)
        del table
        frame.write.mode("overwrite" if offset == 0 else "append").parquet(str(out_dir))
    del rows


def load_paths(
    path_dir: str | pathlib.Path,
    spark: SparkSession,
    *,
    expected_origin: str | None = None,
    expected_horizon: int | None = None,
    expected_n_paths: int | None = None,
    expected_series_ids: list[str] | tuple[str, ...] | None = None,
):
    """Read and validate a path artifact as ``[n_paths, n_series, horizon]``.

    The artifact contract is exact, not best-effort: metadata must be uniform; series indices must
    be dense and unique; every payload must have exactly ``n_paths * horizon`` int64 values and a
    matching SHA-256; and the horizon must immediately follow the origin. Rows are streamed with
    ``toLocalIterator`` after allocating the final tensor. A missing series therefore cannot
    masquerade as perfectly certain zero-demand paths.

    Optional expectations bind the self-described artifact to a calling run. Series expectations
    are order-sensitive because the returned tensor uses the artifact's declared series order.
    Returns ``(tensor, series_ids)``.
    """
    frame = spark.read.parquet(str(path_dir))
    first_rows = frame.take(1)
    if not first_rows:
        raise OursError(f"no paths at {path_dir}")
    missing_columns = sorted(_PATH_COLUMNS - set(frame.columns))
    if missing_columns:
        raise OursError(
            f"path artifact at {path_dir} lacks required columns {missing_columns}; legacy "
            "artifacts cannot prove rectangular coverage and must be regenerated"
        )

    first = first_rows[0]

    def positive_int(name: str, value) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise OursError(f"path artifact {name} must be a positive integer, got {value!r}")
        return value

    version = positive_int("artifact_version", first["artifact_version"])
    if version != PATH_ARTIFACT_VERSION:
        raise OursError(
            f"unsupported path artifact version {version}; expected {PATH_ARTIFACT_VERSION}"
        )
    n_paths = positive_int("n_paths", first["n_paths"])
    n_series = positive_int("n_series", first["n_series"])
    horizon = positive_int("horizon", first["horizon"])
    origin = first["origin"]
    try:
        origin_day = dt.date.fromisoformat(origin)
    except (TypeError, ValueError) as exc:
        raise OursError(f"path artifact origin must be an ISO date, got {origin!r}") from exc
    if origin != origin_day.isoformat():
        raise OursError(f"path artifact origin must be the canonical ISO date, got {origin!r}")
    horizon_start = (origin_day + dt.timedelta(days=1)).isoformat()
    horizon_end = (origin_day + dt.timedelta(days=horizon)).isoformat()

    metadata = (
        version,
        origin,
        first["horizon_start"],
        first["horizon_end"],
        n_paths,
        n_series,
        horizon,
        first["encoding"],
    )
    if first["encoding"] != _PATH_ENCODING:
        raise OursError(
            f"unsupported path payload encoding {first['encoding']!r}; expected {_PATH_ENCODING}"
        )
    if first["horizon_start"] != horizon_start or first["horizon_end"] != horizon_end:
        raise OursError(
            f"path horizon [{first['horizon_start']}, {first['horizon_end']}] is not the "
            f"{horizon}-day daily window immediately after origin {origin}; expected "
            f"[{horizon_start}, {horizon_end}]"
        )

    if expected_origin is not None and origin != expected_origin:
        raise OursError(f"path origin {origin} does not match expected origin {expected_origin}")
    if expected_horizon is not None and horizon != expected_horizon:
        raise OursError(
            f"path horizon {horizon} does not match expected horizon {expected_horizon}"
        )
    if expected_n_paths is not None and n_paths != expected_n_paths:
        raise OursError(f"artifact has {n_paths} paths, expected exactly {expected_n_paths}")

    row_count = frame.count()
    if row_count != n_series:
        relation = "missing series row(s)" if row_count < n_series else "excess series row(s)"
        raise OursError(
            f"path artifact has {row_count} rows for {n_series} declared series: {relation}; "
            "artifact is not rectangular"
        )

    # Output is allocated empty, never zero-filled: no value enters until that series row passes
    # its metadata, mapping, checksum, shape and domain validations.
    occupied = bytearray(n_series)
    out = torch.empty(n_paths, n_series, horizon, dtype=torch.float64)
    index_to_series: dict[int, str] = {}
    series_to_index: dict[str, int] = {}
    expected_bytes = n_paths * horizon * np.dtype("<i8").itemsize

    def bounded_rows():
        for start in range(0, n_series, _PATH_ROWS_PER_SPARK_WRITE):
            end = min(start + _PATH_ROWS_PER_SPARK_WRITE, n_series)
            ordered = frame.where(
                (F.col("series_index") >= start) & (F.col("series_index") < end)
            ).orderBy("series_index")
            yield from ordered.toLocalIterator(prefetchPartitions=True)

    for row_number, row in enumerate(bounded_rows(), 1):
        got = (
            row["artifact_version"],
            row["origin"],
            row["horizon_start"],
            row["horizon_end"],
            row["n_paths"],
            row["n_series"],
            row["horizon"],
            row["encoding"],
        )
        if got != metadata:
            raise OursError(
                f"path artifact metadata changes at row {row_number}: {got!r} != {metadata!r}"
            )
        series_id = row["series_id"]
        if not isinstance(series_id, str) or not series_id.strip():
            raise OursError(f"blank or non-string series id at path row {row_number}")
        series_index = row["series_index"]
        if (
            not isinstance(series_index, int)
            or isinstance(series_index, bool)
            or not 0 <= series_index < n_series
        ):
            raise OursError(
                f"series_index {series_index!r} at row {row_number} is outside [0, {n_series})"
            )
        if (series_index in index_to_series and index_to_series[series_index] != series_id) or (
            series_id in series_to_index and series_to_index[series_id] != series_index
        ):
            raise OursError(
                f"series id/index mapping is inconsistent at row {row_number}: "
                f"{series_id!r} -> {series_index}"
            )
        index_to_series[series_index] = series_id
        series_to_index[series_id] = series_index

        if occupied[series_index]:
            raise OursError(f"duplicate series row for {series_id!r} at index {series_index}")
        payload = row["demand_bytes"]
        if payload is None or len(payload) != expected_bytes:
            got_bytes = None if payload is None else len(payload)
            raise OursError(
                f"series {series_id!r} payload has {got_bytes} bytes; expected {expected_bytes} "
                f"for [{n_paths}, {horizon}] int64 demand"
            )
        checksum = row["payload_sha256"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or hashlib.sha256(payload).hexdigest() != checksum
        ):
            raise OursError(f"series {series_id!r} path payload checksum mismatch")
        decoded = np.frombuffer(payload, dtype="<i8")
        if (decoded < 0).any():
            raise OursError(f"series {series_id!r} path payload contains negative demand")
        # Copy via torch.tensor: Spark owns the backing bytes and the iterator may release them.
        out[:, series_index, :] = torch.tensor(
            decoded.reshape(n_paths, horizon), dtype=torch.float64
        )
        occupied[series_index] = 1

    if len(index_to_series) != n_series or set(index_to_series) != set(range(n_series)):
        raise OursError(
            f"path artifact declares {n_series} series but only has dense indices "
            f"{sorted(index_to_series)}"
        )
    if occupied.count(1) != n_series:
        series_index = occupied.index(0)
        series_id = index_to_series.get(series_index, f"index {series_index}")
        raise OursError(f"missing series row for {series_id!r}; artifact is not rectangular")

    series_ids = [index_to_series[i] for i in range(n_series)]
    if expected_series_ids is not None:
        expected = list(expected_series_ids)
        if len(set(expected)) != len(expected) or any(
            not isinstance(s, str) or not s.strip() for s in expected
        ):
            raise OursError("expected_series_ids must be unique non-blank strings")
        if series_ids != expected:
            raise OursError(
                f"artifact series order {series_ids!r} does not match expected {expected!r}"
            )
    return out, series_ids


def series_forecast_paths(
    model: CustomModel,
    inputs: dict,
    n_paths: int,
    seed: int = 0,
    *,
    factor_draws: int | None = None,
) -> torch.Tensor:
    """Draw `n_paths` integer paths without reducing them to a mean.

    Gate 4 requires calibrated cumulative lead-time coverage, and that cannot be computed from a
    point forecast. Summed marginal quantiles can sit above or below the quantile of the sum
    because they omit cross-day dependence. This is the entry point for that check.
    """
    return draw_paths_over_factor(model, inputs, n_paths, factor_draws=factor_draws, seed=seed)


# ------------------------------------------------------------------ scenario attribution
#
# `forecast/scenario/contrast.py` addresses the known-future tensor BY COLUMN INDEX, which is why
# it could not be wired until the layout carried names. With the old 4-column tensor -- day of
# week, day of month, weekend, carried price -- promotion and event were absent from the spec's
# settable driver set (`price`, `promo`, `event`), so those interventions could not be expressed.
# The lagged baseline is inferred encoder state, not a driver: no scenario API can set it.

COMMERCIAL_BUNDLE = ("price", "discount_depth", "promo")
"""The commercial drivers, which move TOGETHER and must be intervened on together.

Turning promotion off while leaving the discounted price and its depth in place describes a world
that has never existed. The model still returns a number, because it is a smooth function
evaluated off-manifold, and that number is extrapolation through a physically impossible input.

`InterventionBundle.columns` is plural for this reason, and a single-column commercial bundle is
nearly always the mistake it exists to prevent.
"""


def intervention_bundle(
    known: KnownFuture,
    name: str,
    columns: tuple[str, ...],
    reference: float = 0.0,
    reference_values: torch.Tensor | None = None,
):
    """Resolve an `InterventionBundle` against a known-future layout BY NAME.

    Args:
        known: The layout the model was given.
        name: What the bundle is.
        columns: Driver column names, e.g. `COMMERCIAL_BUNDLE`.
        reference: Scalar level each column takes in the counterfactual.
        reference_values: Per-cell references, `[..., len(columns)]`, for bundles a scalar
            cannot express -- "no promotion" for a raw price column is each product's own LIST
            price, and a scalar 0 there prices the whole catalogue at free.

    Returns:
        A `forecast.scenario.contrast.InterventionBundle` with resolved indices.

    Raises:
        OursError: a named driver is not an input for this client -- raised rather than skipped,
            because a bundle silently missing its promotion column intervenes on price alone and
            reports the result as the effect of the promotion.
    """
    from forecast.scenario.contrast import InterventionBundle

    return InterventionBundle(
        name=name,
        columns=tuple(known.driver_index(c) for c in columns),
        reference=reference,
        reference_values=reference_values,
    )


def predicted_mean(p: torch.Tensor | None, mu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """`E[Y]` for whichever head produced these parameters.

    The hurdle mean is `p * mu / (1 - P_NB(0))` -- the truncation factor the closed form must
    carry -- and the plain NB mean is `mu` exactly, because nothing was truncated. Routing on
    `p is None` is the ONE place the two heads' means differ, and every consumer goes through
    here so no caller can apply the hurdle correction to a head that has no hurdle.
    """
    if p is None:
        return mu
    return hurdle_mean(p, mu, k)


def _plain_mean_float64(
    model: CustomModel,
    base: BasePrediction,
    *,
    factor_path: torch.Tensor | None,
) -> torch.Tensor:
    """Recover a plain-NB conditional mean from its log predictor without float32 overflow.

    This is an exceptional fallback for the MC route. Normal finite outputs continue through the
    head unchanged so their seeded last bits do not move; only a non-finite float32 ``exp`` is
    recomputed in float64. A value beyond float64 is refused rather than returned as infinity.
    """
    if model.head_kind != "negbin":
        raise OursError("float64 plain-mean recovery requires the plain negative-binomial head")
    eta_size, _ = base.predictors(model)
    if model.factor is None:
        n_draws = 1 if factor_path is None else int(factor_path.shape[0])
        size = eta_size.new_zeros((n_draws, base.n_cells, base.horizon))
    else:
        if factor_path is None:
            raise OursError("factor-bearing plain-mean recovery requires the sampled factor path")
        _, size = model.factor.contract(factor_path, base.dense_group_index, base.store_index)
    if base.seasonal is not None:
        size = size + base.seasonal[1].unsqueeze(0)
    value = torch.exp((eta_size.unsqueeze(0) + size).double())
    if not bool(torch.isfinite(value).all()):
        raise OursError(
            "the plain negative-binomial Monte Carlo point mean exceeds the finite float64 "
            "range; refusing rather than publishing infinity"
        )
    return value


def predicted_paths(
    p: torch.Tensor | None,
    mu: torch.Tensor,
    k: torch.Tensor,
    n_draws: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Integer sample paths for whichever head produced these parameters -- same routing rule
    as `predicted_mean`, for the same reason."""
    if p is None:
        return sample_paths_negbin(mu, k, n_draws, generator=generator)
    return sample_paths(p, mu, k, n_draws, generator=generator)


def expected_demand_fn(model: CustomModel, inputs: dict):
    """Build the `(known_future, factor_path) -> expected demand per cell AND DAY` callable
    `contrast` wants -- `[cells, horizon]`, so the attribution window can integrate a chosen
    sub-window rather than being forced to the whole horizon.

    The callable also synchronises the first-day commercial-branch token from the supplied
    known-future tensor. Price, discount and promotion are duplicated there by construction;
    leaving the token factual while changing the decoder would evaluate two scenarios at once.

    For plain bottom-up checkpoints, ``factor_path`` is the model's canonical empty sentinel; it
    is reused by both arms but contains no stochastic coordinate and creates no dependence.

    **Analytic in the counts, given the factor.** `ForecastModelSpec.attribution_scope` says point
    estimates need no sampling at all, and it matters here specifically: a contrast is a DIFFERENCE
    of two means, usually smaller than the Monte Carlo error on either one, so sampling would let
    the sampler dominate the quantity being reported.

    **The closed form is `heads.hurdle_mean`, NOT `p * mu`.** An earlier contract draft used that
    shortcut, but `mu` is the UNTRUNCATED negative binomial mean while the size component is
    zero-truncated, so its mean is `mu / (1 - P_NB(0))`. Measured against
    4,000 drawn paths at `p=0.4, mu=2.5, k=1.5`, `p * mu` is out by 23% and `hurdle_mean` by 0.02%.
    The error grows with intermittency, which is where this project's data lives, and it does NOT
    cancel in a contrast: the truncation factor depends on `mu`, so it differs between the factual
    and counterfactual runs being subtracted.

    Under ``monotone_raw_v1``, a price-only scenario reuses one cached price-independent base and
    replaces only its validated depth/mask tensors. Any change to promotion, availability, or a
    non-price known feature falls back to a full encode. Thus the fast path has exact context
    parity rather than assuming that an arbitrary intervention was price-only.
    """
    structural_base = None
    factual_known = None
    unrestricted_known_indices: tuple[int, ...] = ()
    if model.price_response is not None:
        known = inputs.get("known_future")
        if not isinstance(known, torch.Tensor) or known.dim() != 3:
            raise OursError(
                "structural scenario contrast requires rank-3 factual known_future inputs"
            )
        factual_known = known.detach().clone()
        price_columns = {model.price_index, model.discount_index}
        unrestricted_known_indices = tuple(
            index for index in range(known.shape[-1]) if index not in price_columns
        )
        with evaluating(model), torch.no_grad():
            structural_base = model.encode_base(**_base_inputs(inputs))

    def fn(known_future: torch.Tensor, factor_path: torch.Tensor) -> torch.Tensor:
        use_structural_base = False
        if structural_base is not None and factual_known is not None:
            use_structural_base = (
                isinstance(known_future, torch.Tensor)
                and known_future.shape == factual_known.shape
                and known_future.dtype == factual_known.dtype
                and known_future.device == factual_known.device
                and bool(
                    _bitwise_element_equal(
                        known_future[..., list(unrestricted_known_indices)],
                        factual_known[..., list(unrestricted_known_indices)],
                    ).all()
                )
            )
        with evaluating(model), torch.no_grad():
            if use_structural_base:
                depth, response_mask = model.structural_price_coordinates(known_future)
                scenario_base = replace(
                    structural_base, price_depth=depth, price_response_mask=response_mask
                )
                p, mu, k = model.apply_factor(
                    scenario_base,
                    n_draws=inputs["n_draws"],
                    factor_path=factor_path,
                    generator=inputs.get("generator"),
                )
            else:
                routed = _with_known_future_inputs(
                    inputs, known_future, model.commercial_known_indices
                )
                p, mu, k = model(**{**routed, "factor_path": factor_path})
        # PER DAY, not per cell: `attribution_window` integrates a chosen sub-window of the
        # horizon, and a fn that pre-sums the days makes every window the whole horizon.
        return predicted_mean(p, mu, k).mean(dim=0)  # [cells, horizon]

    return fn


@dataclass(frozen=True)
class _Assembly:
    """Everything the model needs at ONE origin: the encoder window and the features after it.

    Built by `_assemble` for the training origin and again for the forecast origin. They go
    through the SAME function deliberately -- the defect this structure replaced was a training
    target taken from inside the encoder input, and two separately written paths are exactly how
    that kind of mismatch survives.
    """

    series_ids: list[str]
    history: torch.Tensor
    observed: torch.Tensor
    censored: torch.Tensor
    baseline_logit: torch.Tensor
    baseline_log_size: torch.Tensor
    commercial: torch.Tensor | None
    commercial_mask: torch.Tensor | None
    event_distance: torch.Tensor | None
    event_present: torch.Tensor | None
    known: KnownFuture
    n_commercial: int
    n_events: int
    plan: object | None = None


POINT_FACTOR_DRAWS = 128
"""Factor realisations the POINT forecast integrates over, when it integrates by sampling.

SEPARATE FROM THE PATH COUNT ON PURPOSE. The two consumers of a factor draw want different
things and the same number served both badly:

  * the point forecast is an EXPECTATION. Its Monte Carlo error falls as `sigma/sqrt(S)`, so
    going from 128 to 1,000 buys a ~2.8x smaller error term in a quantity whose dominant error
    is model error, not integration error;
  * a sampled path's job is the TAIL. There the count of DISTINCT systemic scenarios bounds
    what any aggregate quantile can resolve, and 1,000 is a floor rather than a luxury.

Coupling them meant the expensive consumer (the point forecast, over the whole catalogue) paid
the tail's sample count, while the cheap one (paths, over `paths_sample` series) set it. Worse,
because the default was derived from `n_paths`, asking for a larger PATH ARTIFACT silently moved
the POINT ESTIMATE -- a point estimand that changes because a caller wanted more paths stored.
`fit_predict` carries the two separately and records the MC point count and path factor count.
"""


RNG_DOMAIN_SCHEME = "sha256-domain-v1"
LEGACY_RNG_DOMAIN_SCHEME = "shared-seed-v0"
"""Versioned separation between systemic-factor and idiosyncratic-count random streams.

Using the same integer to initialise both generators does not make the streams independent: they
are deterministic transforms of the same underlying sequence. A reproduced rank-one model showed
a -0.346 correlation and moved the occurrence rate by 0.027. New forecasts derive disjoint seeds
for the `factor` and `count` domains. Checkpoints record the scheme. A pre-scheme factor-bearing
MC checkpoint can replay only when its historical draw mapping is also recorded; transitional
checkpoints without that fact refuse an exact-replay claim unless a caller explicitly requests a
fresh draw count. Rank-zero checkpoints have no factor draw mapping to recover and retain their
legacy count stream.
"""


def _domain_seed(seed: int | None, domain: str, scheme: str = RNG_DOMAIN_SCHEME) -> int | None:
    """Derive a stable Torch seed for one stochastic domain, or replay the legacy shared seed."""
    if seed is None:
        return None
    if scheme == LEGACY_RNG_DOMAIN_SCHEME:
        return seed
    if scheme != RNG_DOMAIN_SCHEME:
        raise OursError(
            f"unknown RNG domain scheme {scheme!r}; expected {RNG_DOMAIN_SCHEME!r} or "
            f"{LEGACY_RNG_DOMAIN_SCHEME!r}"
        )
    digest = hashlib.sha256(f"{scheme}\0{seed}\0{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


DRAW_ONLY_KEYS = ("n_draws", "factor_path", "generator", "predictor_dtype")
"""Inputs that belong to a factor draw rather than to the deterministic encoding."""


def _base_inputs(inputs: dict) -> dict:
    """The subset of a forward-input dict that `encode_base` consumes."""
    return {k: v for k, v in inputs.items() if k not in DRAW_ONLY_KEYS}


def default_factor_draws(n_paths: int) -> int:
    """One independent shared-factor realisation per emitted predictive PATH.

    A path is a draw from the full predictive distribution, not merely a conditional count draw.
    Reusing 100 factor trajectories ten times in a 1,000-path artifact leaves only 100 systemic
    tail scenarios and overstates the effective sample size of every aggregate quantile. The
    architecture promises 1,000 coherent paths, so the default is exactly `n_paths`. Callers may
    still request fewer factor draws explicitly as a named approximation, but gate artifacts do
    not do that.
    """
    if n_paths < 1:
        raise OursError(f"n_paths must be >= 1, got {n_paths}")
    return n_paths


def mean_over_factor(
    model: CustomModel,
    inputs: dict | None,
    *,
    factor_draws: int | None = None,
    n_paths: int = 1000,
    chunk: int = 25,
    seed: int | None = None,
    rng_domain_scheme: str = RNG_DOMAIN_SCHEME,
    base: BasePrediction | None = None,
) -> torch.Tensor:
    """`E[Y]` per cell, integrating the factor out (one evaluation when factorless).

    `ForecastModelSpec.attribution_scope` says point estimates need no sampling at all, and at
    catalogue scale that is not a refinement. Drawing 1,000 paths for 30,490 series builds a
    `[cells, horizon, support]` table per factor realisation -- measured: the run was still inside
    it after an hour -- and then averages every one of those paths away again. Integrating the
    count distribution analytically reaches the same quantity directly.

    **What is analytic here and what is not.** `predicted_mean` is exact GIVEN a factor
    realisation; the factor itself is integrated by Monte Carlo in this function, so the result
    carries a finite-sample error that moves with `seed` and `factor_draws`. Calling the whole
    thing "a closed form with no Monte Carlo error" -- as this docstring did until 2026-08-25 --
    was wrong by one integral. `mean_over_factor_quadrature` is the deterministic version.

    The mean is `heads.hurdle_mean`, NOT `p * mu`: the size component is zero-truncated, so its
    mean is `mu / (1 - P_NB(0))`.

    Args:
        model: A fitted `CustomModel`.
        inputs: Its inputs, as `fit_predict` assembles them. Exactly one of ``inputs`` and
            ``base`` must be supplied.
        factor_draws: Realisations to average over, or None for `POINT_FACTOR_DRAWS`. It is
            deliberately NOT derived from `n_paths`: a point estimand must not move because a
            caller asked for a bigger path artifact.
        n_paths: Accepted and ignored, kept so existing callers do not break. It used to set the
            default draw count, which is exactly the coupling this signature removed.
        chunk: Factor realisations per head pass. Several at once is much faster than one at a time
            and costs `chunk x cells x horizon` floats, which is small without the support table
            that sampling needs. It changes batching only, never which seeded factor paths are
            drawn or their accumulation order.
        rng_domain_scheme: Versioned factor-seed derivation. New forecasts use independent
            factor/count domains. Checkpoint replay supplies the scheme recorded by that artifact;
            a pre-scheme checkpoint uses the legacy direct seed.

    Returns:
        `[n_cells, horizon]` expected demand.
    """
    if (inputs is None) == (base is None):
        raise OursError("mean_over_factor requires exactly one of raw inputs or a BasePrediction")
    if factor_draws is None:
        factor_draws = POINT_FACTOR_DRAWS
    if isinstance(factor_draws, bool) or not isinstance(factor_draws, int) or factor_draws < 1:
        raise OursError(f"factor_draws must be >= 1, got {factor_draws}")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise OursError(f"chunk must be a positive integer, got {chunk!r}")
    if model.factor is None:
        # There is nothing to Monte Carlo-integrate.  Calling the same deterministic marginal
        # head `factor_draws` times is both misleading and, at the advertised 1,000 draws,
        # needlessly expensive.  Keep this entry point shared with factor models but evaluate it
        # once.  An explicitly supplied factor_draws remains a harmless compatibility option.
        with (
            _seeded_rng(model, _domain_seed(seed, "factor", rng_domain_scheme)),
            evaluating(model),
            torch.no_grad(),
        ):
            if base is None:
                base = model.encode_base(**_base_inputs(inputs))
            p, mu, k = model.apply_factor(base, n_draws=1, factor_path=None)
        value = predicted_mean(None if p is None else p[0], mu[0], k[0]).cpu().double()
        if not bool(torch.isfinite(value).all()):
            if p is not None:
                raise OursError(
                    "the Monte Carlo point mean became non-finite under the hurdle head; "
                    "refusing rather than publishing an overflowed expectation"
                )
            # ``NegBinHead`` exponentiates in the model dtype (normally float32).  A finite
            # mathematical mean such as exp(100) overflows there, even though the point API
            # returns float64. Recover only this exceptional branch from the cached log-mean so
            # ordinary seeded results retain their historical last bits.
            with evaluating(model), torch.no_grad():
                recovered = _plain_mean_float64(model, base, factor_path=None)[0].cpu()
            value = torch.where(torch.isfinite(value), value, recovered)
        return value
    total, running_mean, overflowed, seen, done = None, None, None, 0, 0
    # ENCODE ONCE. The deterministic stack does not depend on the factor, so evaluating it per
    # chunk was pure waste -- measured at 33x one factor draw on the real 30,490-cell batch.
    # It consumes no randomness in eval mode, so hoisting it out of the seeded block leaves the
    # factor stream exactly where it was.
    if base is None:
        with evaluating(model), torch.no_grad():
            base = model.encode_base(**_base_inputs(inputs))
    # Seeded for the same reason as `draw_paths_over_factor`: the factor realisations being
    # averaged over come from the global RNG, and a point forecast that moves with whatever
    # consumed the stream earlier is a random quantity wearing a deterministic name.
    with _seeded_rng(model, _domain_seed(seed, "factor", rng_domain_scheme)):
        while done < factor_draws:
            take = min(chunk, factor_draws - done)
            # Draw ONE systemic trajectory per RNG call, then batch only the head work. A single
            # ``randn([take, ...])`` does not promise the same seeded number-to-trajectory mapping
            # as ``take`` calls of ``randn([1, ...])`` (and in practice it did not): chunk 1/7/25
            # moved a 25-draw point forecast by up to 0.165. Canonical per-trajectory draws make
            # ``chunk`` a cost knob rather than part of the estimand while retaining bounded path
            # memory. The dense group mapping already stored on ``base`` makes this loop cheap.
            factor_path = torch.cat(
                [model.factor.draw(len(base.group_labels), base.horizon, 1) for _ in range(take)],
                dim=0,
            )
            with evaluating(model), torch.no_grad():
                p, mu, k = model.apply_factor(base, n_draws=take, factor_path=factor_path)
            values = predicted_mean(p, mu, k).double()
            if not bool(torch.isfinite(values).all()):
                if p is not None:
                    raise OursError(
                        "the Monte Carlo point mean became non-finite under the hurdle head; "
                        "refusing rather than publishing an overflowed expectation"
                    )
                with evaluating(model), torch.no_grad():
                    recovered = _plain_mean_float64(model, base, factor_path=factor_path)
                # Preserve every ordinary finite head result bit-for-bit. Recomputing the whole
                # chunk in float64 because one neighbour overflowed made both cells and chunk
                # size part of an otherwise seeded result.
                values = torch.where(torch.isfinite(values), values, recovered)
            # Accumulate in one fixed draw order in float64. A reduction over ``take`` would make
            # floating grouping depend on chunk even after the paths themselves were canonical.
            for value in values:
                value = value.cpu()
                if total is None:
                    total = value
                    running_mean = torch.zeros_like(value)
                    overflowed = torch.zeros_like(value, dtype=torch.bool)
                    seen = 1
                    continue
                next_seen = seen + 1
                candidate = total + value
                newly_overflowed = ~overflowed & ~torch.isfinite(candidate)
                if bool(newly_overflowed.any()):
                    prior_mean = total / seen
                    new_mean = prior_mean + (value - prior_mean) / next_seen
                    running_mean = torch.where(newly_overflowed, new_mean, running_mean)
                if bool(overflowed.any()):
                    updated_mean = running_mean + (value - running_mean) / next_seen
                    running_mean = torch.where(overflowed, updated_mean, running_mean)
                overflowed = overflowed | newly_overflowed
                total = torch.where(overflowed, total, candidate)
                seen = next_seen
            done += take
    ordinary_mean = total / factor_draws
    return torch.where(overflowed, running_mean, ordinary_mean)


GH_ORDERS = (5, 7, 11, 15)
"""Quadrature orders tried in sequence, coarsest first.

Three rules are the minimum stability screen, not an error proof. Finite Gauss-Hermite rules can
agree by aliasing while all remain far from an integrand with a clamp kink. The implementation
therefore accepts only when BOTH finest gaps have collapsed to the float32 head's evaluation-noise
floor, per cell and horizon. Cells that do not settle then go through the independent-scramble
randomized-QMC screen below; a cell-step that fails both routes is refused.
"""

GH_START_ORDERS = (7,)
"""Validated production start: the contracted 7 -> 11 -> 15 convergence sequence.

Starting at 5 used to permit a 5 -> 7 -> 11 screen to settle before the required 15-node rule was
ever evaluated. A measured in-envelope fixture did exactly that while the public 7-node route
reached 15 and correctly refused. Five nodes remain useful as a diagnostic in ``GH_ORDERS`` but
are not a certifying public start.
"""

GH_NEGLIGIBLE = 0.01
"""Fraction of the tolerance below which a gap is indistinguishable for this decision."""

GH_NOISE_ULPS = 64.0
"""Float32 ulps, relative to value scale, used by the conservative stability screen.

The head evaluates in float32, so the noise floor is proportional to the magnitude of what is
being integrated -- which is why an absolute floor cannot work here and a relative one can.
Measured on a converged integrand: successive gaps bottom out around 4.4x float32 eps relative
to scale (1.6e-4, 1.4e-5, 5.2e-7 relative, ratios 0.10 and 0.04). 64 leaves better than an order
of magnitude of headroom.

The floor is not called an error bound: it says only that the available rules no longer resolve a
difference through the float32 head. Both finest gaps must reach it, and both must independently
remain inside the requested absolute tolerance. Forecast magnitude alone is never a failure or a
reason to trigger the expensive fallback."""

GH_NODES = 7
"""Gauss-Hermite nodes per dimension for the point forecast's factor integral.

Measured on the fitted checkpoint against an 80x80 reference: 7x7 gives max relative error
~7e-9 and max absolute error ~4.2e-5 units, while 5x5 leaves ~2.7e-3 -- close enough to the
smallest archived ladder separation (~0.005 MAE) to matter. 7 is the floor, not a guarantee:
`mean_over_factor_quadrature` checks it against a higher-order rule on every call and escalates
or refuses rather than trusting this constant.
"""

GH_MAX_OCCURRENCE_SD = 0.5
GH_MAX_SIZE_SD = 0.2
"""Runtime envelope validated against the fitted checkpoint.

The converged review measured occurrence loading SDs through 0.492 and size SDs through 0.156,
then established the standard-GH error on that envelope. Successive centred rules are not capable
of detecting every out-of-envelope failure: with a very large occurrence loading all 7/11/15
nodes can land on the same +/-14 clamp plateau and agree exactly while missing a material Gaussian
tail. A checkpoint outside these deliberately rounded limits must use a separately validated
tilted/adaptive or Monte Carlo route; it is not silently certified by this implementation.
"""

GH_OUTER_STANDARD_NODE = float(
    np.polynomial.hermite.hermgauss(GH_ORDERS[-1])[0][-1] * math.sqrt(2.0)
)
"""Largest standard-normal coordinate represented by the final centred GH rule."""

NODE_CHUNK = 16
"""Quadrature nodes evaluated per pass. Bounds peak memory at `[chunk, cells, horizon]` exactly
as the sampled loop's `chunk` does, and for the same reason."""

GH_TOLERANCE = 1e-4
"""Max absolute per-cell disagreement, in demand units, between successive quadrature orders.

Tied to the DECISION, not to taste: the smallest ladder separation on record is ~0.005 MAE, so a
1e-4 integration discrepancy is ~2% of the smallest difference the ladder is asked to resolve."""

RQMC_REPLICATIONS = 4
RQMC_COARSE_POINTS = 4096
RQMC_FINE_POINTS = 16384
RQMC_SEED = 271828
RQMC_NODE_CHUNK = 256
RQMC_MAX_BATCH_ELEMENTS = 2_000_000
"""Deterministic replicated randomized-QMC fallback configuration.

Each replication has its own fixed Owen-scrambled Sobol net. The 4,096-point prefix and the full
16,384-point net provide a resolution check, while disagreement among the four independently
scrambled full nets provides a separate replication check. These checks are empirical numerical
stability evidence, not a universal error bound. Fixed scrambles make the point forecast exactly
repeatable; changing these constants changes the numerical method and requires revalidation.

Only cells with at least one unresolved GH cell-step enter this route. The dynamic chunk cap keeps
the largest materialised head tensor below ``RQMC_MAX_BATCH_ELEMENTS`` even if an adversarial
catalogue sends every cell here.
"""


def _loading_geometry(
    model: CustomModel,
    store_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cell offset covariance and the true rank of the two loading rows.

    Returns `([cells, 2, 2], [cells])`. Rank is computed from the loading matrix itself rather
    than inferred from a rounded covariance determinant: zero and collinear loading rows are
    structural properties of the fitted model. No diagonal jitter and no invented variance.

    `torch.linalg.matrix_rank` operates in float64 so its tolerance is applied to the stored
    float32 loading values, not to a second round of float32 covariance arithmetic. Geometry is
    computed once per distinct store and gathered back to cells; loadings do not vary by product
    or horizon.
    """
    stores, inverse = torch.unique(store_index, sorted=True, return_inverse=True)
    occ = model.factor.loading_occurrence(stores)  # [stores, factor_rank]
    size = model.factor.loading_size(stores)
    loading = torch.stack([occ, size], dim=-2)  # [stores, 2, factor_rank]
    rank = torch.linalg.matrix_rank(loading.double())
    var_o = (occ * occ).sum(-1)
    var_s = (size * size).sum(-1)
    cov = (occ * size).sum(-1)
    covariance = torch.stack(
        [torch.stack([var_o, cov], dim=-1), torch.stack([cov, var_s], dim=-1)], dim=-2
    )
    return covariance.index_select(0, inverse), rank.index_select(0, inverse)


def _loading_covariance(model: CustomModel, store_index: torch.Tensor) -> torch.Tensor:
    """Per-cell covariance of `(occurrence, size)` factor offsets. `[cells, 2, 2]`.

    THE FACTOR'S PER-STEP MARGINAL IS EXACTLY `N(0, I)`. `SharedLatentFactor` draws
    `f_0 ~ N(0, I)` and `f_t = phi*f_{t-1} + sqrt(1-phi^2)*eps_t`, whose innovation scale is
    chosen precisely to hold the marginal variance at 1 (see `innovation_scale`, which explains
    that a unit innovation would fan the forecast out with lead time by construction).

    Two consequences that make this whole approach work:

      * `phi` does NOT enter a per-step marginal mean. Persistence is cross-TIME dependence; it
        matters enormously for paths and lead-time sums and not at all here.
      * the offsets are linear projections of a standard Gaussian, so `(occ, size)` is bivariate
        Gaussian with covariance built from the loadings alone -- per STORE, so it is the same
        for every product, group and horizon step that store owns.
    """
    # Keep the plain-NB closed form on the cheap covariance-only path: it does not need a rank
    # decomposition, and `matrix_rank` over every catalogue cell would be pure overhead there.
    occ = model.factor.loading_occurrence(store_index)
    size = model.factor.loading_size(store_index)
    var_o = (occ * occ).sum(-1)
    var_s = (size * size).sum(-1)
    cov = (occ * size).sum(-1)
    return torch.stack(
        [torch.stack([var_o, cov], dim=-1), torch.stack([cov, var_s], dim=-1)], dim=-2
    )


def _gauss_hermite_offsets(cov: torch.Tensor, nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-rank quadrature offsets and weights. `([nodes^2, cells, 2], [nodes^2])`.

    Standard tensor-product Gauss-Hermite in the probabilists' scaling: for `Z ~ N(0, Sigma)`,
    `E[g(Z)] = (1/pi) * sum_ij w_i w_j g(L [sqrt(2) x_i, sqrt(2) x_j])` with `L L^T = Sigma`.

    The caller partitions structural covariance ranks first: rank zero is evaluated once and rank
    one uses `_gauss_hermite_offsets_1d`. This helper therefore owns only the genuine 2-D rule.
    Eigendecomposition still clamps negative eigenvalues from round-off to zero and never adds
    jitter.
    """
    import numpy as np

    x, w = np.polynomial.hermite.hermgauss(nodes)
    x = torch.tensor(x, dtype=cov.dtype, device=cov.device) * (2.0**0.5)
    w = torch.tensor(w, dtype=cov.dtype, device=cov.device) / (torch.pi**0.5)
    grid = torch.stack(torch.meshgrid(x, x, indexing="ij"), dim=-1).reshape(-1, 2)
    weights = (w.unsqueeze(1) * w.unsqueeze(0)).reshape(-1)  # [nodes^2], sums to 1

    values, vectors = torch.linalg.eigh(cov.double())
    scale = vectors * values.clamp_min(0.0).sqrt().unsqueeze(-2)  # L = V sqrt(Lambda)
    offsets = torch.einsum("cij,nj->nci", scale, grid.double())  # [nodes^2, cells, 2]
    return offsets.to(cov.dtype), weights


def _gauss_hermite_offsets_1d(
    cov: torch.Tensor,
    nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-one offsets and weights. `([nodes, cells, 2], [nodes])`.

    A singular bivariate Gaussian lives on one line. Tensoring that line with a dead second
    direction evaluates every useful point `nodes` times and, at rank zero, can even manufacture
    a convergence failure from repeated float32 weights. This uses the sole positive eigenvector
    directly. The caller invokes it only for loading matrices whose structural rank is one.
    """
    import numpy as np

    x, w = np.polynomial.hermite.hermgauss(nodes)
    x = torch.tensor(x, dtype=cov.dtype, device=cov.device) * (2.0**0.5)
    w = torch.tensor(w, dtype=cov.dtype, device=cov.device) / (torch.pi**0.5)
    values, vectors = torch.linalg.eigh(cov.double())
    direction = vectors[..., -1] * values[..., -1].clamp_min(0.0).sqrt().unsqueeze(-1)
    offsets = x.double().view(-1, 1, 1) * direction.unsqueeze(0)
    return offsets.to(cov.dtype), w


def _randomized_qmc_integral(
    model: CustomModel,
    cov: torch.Tensor,
    eta_occ: torch.Tensor,
    eta_size: torch.Tensor,
    log_k: torch.Tensor,
    rank: int,
    seasonal: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicated scrambled-Sobol estimate and two per-cell/horizon stability diagnostics.

    Returns ``(fine_center, coarse_to_fine_shift, independent_replication_spread)``. The latter
    two tensors are deliberately kept separate: agreement of nested resolutions is not
    independence, and agreement of independent scrambles alone does not show that increasing the
    net size stopped moving the answer. The caller accepts only where both are within its declared
    absolute tolerance.

    The covariance is transformed in its structural dimension. Rank one uses only the positive
    eigendirection; rank two uses both. Negative eigenvalues attributable to round-off are clamped
    to zero exactly as in the GH route, with no diagonal jitter or invented variance.
    """
    if rank not in (1, 2):
        raise OursError(f"randomized QMC requires structural rank 1 or 2, got {rank}")
    if (
        RQMC_REPLICATIONS < 2
        or RQMC_COARSE_POINTS < 1
        or RQMC_COARSE_POINTS & (RQMC_COARSE_POINTS - 1)
        or RQMC_FINE_POINTS < RQMC_COARSE_POINTS
        or RQMC_FINE_POINTS & (RQMC_FINE_POINTS - 1)
    ):
        raise OursError(
            "RQMC needs at least two replications and positive nested power-of-two point "
            "counts with fine >= coarse"
        )

    n_cells, horizon = eta_occ.shape
    if RQMC_MAX_BATCH_ELEMENTS < horizon:
        raise OursError(
            f"RQMC_MAX_BATCH_ELEMENTS={RQMC_MAX_BATCH_ELEMENTS} cannot hold even one "
            f"cell/node across horizon={horizon}"
        )

    # A node chunk alone cannot enforce the element cap once ``cells * horizon`` exceeds it:
    # even one node would already be too large. Partition cells first, then choose the largest
    # node chunk each leaf can hold. Every recursive leaf constructs the SAME fixed scrambled
    # Sobol nets, so cell partitioning does not change which integration points a cell sees.
    cells_per_chunk = max(1, RQMC_MAX_BATCH_ELEMENTS // max(horizon * RQMC_NODE_CHUNK, 1))
    if n_cells > cells_per_chunk:
        pieces = [
            _randomized_qmc_integral(
                model,
                cov[start : start + cells_per_chunk],
                eta_occ[start : start + cells_per_chunk],
                eta_size[start : start + cells_per_chunk],
                log_k[start : start + cells_per_chunk],
                rank,
                None
                if seasonal is None
                else (
                    seasonal[0][start : start + cells_per_chunk],
                    seasonal[1][start : start + cells_per_chunk],
                ),
            )
            for start in range(0, n_cells, cells_per_chunk)
        ]
        return (
            torch.cat([piece[0] for piece in pieces], dim=0),
            torch.cat([piece[1] for piece in pieces], dim=0),
            torch.cat([piece[2] for piece in pieces], dim=0),
        )

    eigenvalues, eigenvectors = torch.linalg.eigh(cov.double())
    if rank == 1:
        direction = eigenvectors[..., -1] * eigenvalues[..., -1].clamp_min(0.0).sqrt().unsqueeze(-1)
        scale = None
    else:
        direction = None
        scale = eigenvectors * eigenvalues.clamp_min(0.0).sqrt().unsqueeze(-2)

    chunk = max(
        1,
        min(
            RQMC_NODE_CHUNK,
            RQMC_MAX_BATCH_ELEMENTS // max(n_cells * horizon, 1),
        ),
    )
    coarse_replications: list[torch.Tensor] = []
    fine_replications: list[torch.Tensor] = []
    exponent = int(math.log2(RQMC_FINE_POINTS))
    unit_eps = torch.finfo(torch.float64).eps

    for replication in range(RQMC_REPLICATIONS):
        engine = torch.quasirandom.SobolEngine(
            dimension=rank, scramble=True, seed=RQMC_SEED + replication
        )
        unit = engine.draw_base2(exponent, dtype=torch.float64)
        unit = unit.clamp(min=unit_eps, max=1.0 - unit_eps).to(cov.device)
        normal = math.sqrt(2.0) * torch.erfinv(2.0 * unit - 1.0)
        coarse_total = torch.zeros((n_cells, horizon), dtype=torch.float64, device=cov.device)
        fine_total = torch.zeros_like(coarse_total)

        for start in range(0, RQMC_FINE_POINTS, chunk):
            stop = min(start + chunk, RQMC_FINE_POINTS)
            z = normal[start:stop]
            if rank == 1:
                offsets = z[:, 0].view(-1, 1, 1) * direction.unsqueeze(0)
            else:
                offsets = torch.einsum("cij,nj->nci", scale, z)
            offsets = offsets.to(cov.dtype)
            occurrence_offset = offsets[..., 0].unsqueeze(-1)
            size_offset = offsets[..., 1].unsqueeze(-1)
            if seasonal is not None:
                # Same order as the legacy monolithic forward and `apply_factor`: combine the
                # factor and seasonal offsets first, then add that result to the cached eta.
                occurrence_offset = occurrence_offset + seasonal[0].unsqueeze(0)
                size_offset = size_offset + seasonal[1].unsqueeze(0)
            p, mu, k = model.head.from_predictors(
                eta_occ.unsqueeze(0),
                eta_size.unsqueeze(0),
                log_k.unsqueeze(0),
                occurrence_offset,
                size_offset,
            )
            values = predicted_mean(p, mu, k.expand_as(mu)).double()
            fine_total += values.sum(0)
            if start < RQMC_COARSE_POINTS:
                take = min(stop, RQMC_COARSE_POINTS) - start
                coarse_total += values[:take].sum(0)

        coarse_replications.append(coarse_total / RQMC_COARSE_POINTS)
        fine_replications.append(fine_total / RQMC_FINE_POINTS)

    coarse = torch.stack(coarse_replications)
    fine = torch.stack(fine_replications)
    coarse_center = coarse.mean(0)
    fine_center = fine.mean(0)
    refinement = (fine_center - coarse_center).abs()
    replication_spread = (fine - fine_center.unsqueeze(0)).abs().amax(0)
    return fine_center, refinement, replication_spread


def _unrepresented_tail_bound(
    cov: torch.Tensor,
    eta_occ: torch.Tensor,
    eta_size: torch.Tensor,
    log_k: torch.Tensor,
) -> torch.Tensor:
    """Conservative hurdle-mean mass beyond the final centred rule's loading projections.

    A finite centred rule has no node beyond ``GH_OUTER_STANDARD_NODE`` in either loading
    projection. Rule-to-rule agreement can therefore be vacuous when every available node sees a
    clamp plateau or the same asymptote. This bounds what could live outside those projections.

    For the NB positive component, with ``x = mu/k``::

        E[Y | Y>0] = mu / (1 - (1+x)^(-k)) <= 1 + mu * (1 + 1/k).

    The inequality follows from ``1-exp(-y) >= y/(1+y)`` and
    ``log(1+x) >= x/(1+x)``. The clamped sigmoid also obeys
    ``p <= sigmoid(-14) + exp(eta_occ + occurrence_offset)``. Expanding those two bounds gives
    four Gaussian exponential moments: a constant, size, occurrence, and occurrence+size.
    Exponential tilting gives each moment's one-dimensional tail exactly, including covariance.
    Summing the occurrence and size tail bounds may double-count their intersection, which is
    conservative. It is a safety screen, not an estimate returned to the caller.
    """
    work_cov = cov.double()
    var_occ = work_cov[:, 0, 0].clamp_min(0.0).unsqueeze(-1)
    var_size = work_cov[:, 1, 1].clamp_min(0.0).unsqueeze(-1)
    sd_occ, sd_size = var_occ.sqrt(), var_size.sqrt()
    cov_occ_size = work_cov[:, 0, 1].unsqueeze(-1)
    eta_o, eta_s = eta_occ.double(), eta_size.double()
    clamped_log_k = log_k.double().clamp(LOG_K_MIN, LOG_K_MAX)
    log_size_multiplier = torch.logaddexp(torch.zeros_like(clamped_log_k), -clamped_log_k)
    outer = GH_OUTER_STANDARD_NODE
    log_p_floor = -math.log1p(math.exp(14.0))

    # (log coefficient, Var[X], Cov[X, occurrence], Cov[X, size]) for exp(X).
    occurrence_sensitive_moments = (
        (
            torch.full_like(eta_o, log_p_floor),
            torch.zeros_like(var_size),
            torch.zeros_like(var_size),
            torch.zeros_like(var_size),
        ),
        (log_p_floor + log_size_multiplier + eta_s, var_size, cov_occ_size, var_size),
        (eta_o, var_occ, var_occ, cov_occ_size),
        (
            log_size_multiplier + eta_o + eta_s,
            var_occ + var_size + 2.0 * cov_occ_size,
            var_occ + cov_occ_size,
            var_size + cov_occ_size,
        ),
    )
    probability_one_moments = (
        (
            torch.zeros_like(eta_o),
            torch.zeros_like(var_size),
            torch.zeros_like(var_size),
            torch.zeros_like(var_size),
        ),
        (log_size_multiplier + eta_s, var_size, cov_occ_size, var_size),
    )

    def projection_bound(sd: torch.Tensor, covariance_at: int) -> torch.Tensor:
        active = sd > 0.0
        safe_sd = sd.clamp_min(torch.finfo(torch.float64).tiny)
        alternatives = []
        # Both p<=1 and p<=sigmoid(-14)+exp(eta_occ+offset) hold. Use the tighter complete
        # bound per cell-step: the first is sharp for a saturated positive occurrence predictor;
        # the second is sharp beside the negative clamp where p is tiny.
        for moments in (probability_one_moments, occurrence_sensitive_moments):
            pieces = []
            for log_coefficient, variance, covariance_occ, covariance_size in moments:
                covariance = covariance_occ if covariance_at == 0 else covariance_size
                shift = covariance / safe_sd
                # Q(outer-shift) + Q(outer+shift), evaluated in log space so a huge moment
                # multiplied by an underflowed tail cannot become the indeterminate inf*0.
                log_tail = torch.logaddexp(
                    torch.special.log_ndtr(shift - outer), torch.special.log_ndtr(-shift - outer)
                )
                pieces.append(torch.exp(log_coefficient + 0.5 * variance + log_tail))
            alternatives.append(torch.stack(pieces).sum(0))
        total = torch.minimum(*alternatives)
        return torch.where(active, total, torch.zeros_like(total))

    return projection_bound(sd_occ, 0) + projection_bound(sd_size, 1)


def point_forecast(
    model: CustomModel,
    inputs: dict | None,
    *,
    method: str = "gauss_hermite",
    factor_draws: int | None = None,
    seed: int = 0,
    rng_domain_scheme: str = RNG_DOMAIN_SCHEME,
    base: BasePrediction | None = None,
) -> torch.Tensor:
    """The point forecast, by the declared integration method. `[n_cells, horizon]`.

    One routing point, so the two call sites in `fit_predict` cannot drift apart:

      * `'gauss_hermite'` (default): deterministic, seedless, convergence-checked, and exact for
        a plain NB head. This is what a point ESTIMAND should be -- a function of the fitted
        model rather than of a random draw.
      * `'mc'`: the sampled integration, kept because it is the reference the quadrature is
        validated against, and because a caller outside the validated loading envelope needs a
        route that does not refuse.

    Raises:
        OursError: an unknown method name. There is no silent fallback: choosing an integration
            rule the caller did not ask for is how a number loses its provenance.
    """
    if (inputs is None) == (base is None):
        raise OursError(
            "point_forecast requires exactly one of raw inputs or a BasePrediction; accepting "
            "both would silently ignore inputs the base was not encoded from"
        )
    if method == "gauss_hermite":
        return mean_over_factor_quadrature(model, inputs, base=base)
    if method == "mc":
        return mean_over_factor(
            model,
            inputs,
            factor_draws=factor_draws,
            seed=seed,
            rng_domain_scheme=rng_domain_scheme,
            base=base,
        )
    raise OursError(
        f"unknown point_factor_method {method!r}; this build defines 'gauss_hermite' and 'mc'"
    )


def mean_over_factor_quadrature(
    model: CustomModel,
    inputs: dict | None,
    *,
    nodes: int = GH_NODES,
    tolerance: float = GH_TOLERANCE,
    base: BasePrediction | None = None,
) -> torch.Tensor:
    """`E[Y]` per cell, integrating the factor DETERMINISTICALLY. `[n_cells, horizon]`.

    The point forecast is an expectation over a latent Gaussian, and sampling it was costing
    both time and accuracy. This computes the same quantity by quadrature:

      * for a PLAIN NB head there is no quadrature at all -- `E[Y] = exp(eta_size + var_s/2)` is
        the lognormal mean, exact, and `k` does not enter it;
      * for the HURDLE head the integrand is
        `p(f) * mu(f) / (1 - P_NB(0; mu(f), k))`, whose zero-truncation factor is nonlinear in
        `mu`, so no elementary closed form exists -- but a two-dimensional Gaussian integral is
        precisely what Gauss-Hermite is for.

    Why this is only two dimensions: the point forecast needs the per-cell MARGINAL expectation,
    so the factor's cross-cell and cross-time dependence -- the entire reason the factor exists --
    is irrelevant here. Only the bivariate marginal of `(occ, size)` matters, and that is
    `N(0, Sigma_store)` at every horizon step.

    Compared with `mean_over_factor`, this has no caller-controlled seed or run-to-run sampling
    noise: the result is a deterministic function of the fitted model. Its remaining integration
    error is numerical. Successive GH-rule agreement is not a general mathematical bound for the
    clamped integrand, so unresolved cells are checked with fixed, independently scrambled Sobol
    replications at two nested resolutions. That replicated-QMC screen is also numerical evidence,
    not a proof. Before either screen may certify a result, the loading standard deviations must
    remain inside the envelope validated on the fitted checkpoint and a Gaussian exponential-tail
    bound must show that the final centred rule does not leave material hurdle-mean mass beyond its
    outer projections. The routine refuses any cell-step those independent checks cannot support.

    Raises:
        OursError: the hurdle loading/tail envelope is unsupported, neither GH nor replicated
            randomized QMC became numerically stable, or the exact plain-NB mean exceeds float64.
            The harder integral is refused rather than presented as certified.
    """
    if (
        isinstance(nodes, bool)
        or not isinstance(nodes, int | np.integer)
        or int(nodes) not in GH_START_ORDERS
    ):
        raise OursError(
            f"nodes must be one of the validated starting GH orders {GH_START_ORDERS}; each "
            f"start must leave three convergence rules, got {nodes!r}"
        )
    nodes = int(nodes)
    if isinstance(tolerance, bool):
        raise OursError(f"tolerance must be a finite positive number, got {tolerance!r}")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise OursError(f"tolerance must be a finite positive number, got {tolerance!r}") from exc
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise OursError(f"tolerance must be a finite positive number, got {tolerance!r}")
    if (inputs is None) == (base is None):
        raise OursError(
            "mean_over_factor_quadrature requires exactly one of raw inputs or a "
            "BasePrediction; a supplied base cannot be verified against separate raw inputs"
        )
    if base is None:
        with evaluating(model), torch.no_grad():
            base = model.encode_base(**_base_inputs(inputs))
    with evaluating(model), torch.no_grad():
        eta = base.predictors(model)
        seasonal = getattr(base, "seasonal", None)
        if model.head_kind == "negbin":
            # Exact. E[exp(b + S)] = exp(b + Var(S)/2) -- the lognormal moment generating
            # function. No nodes, no tolerance, nothing to converge.
            #
            # EXPONENTIATE IN DOUBLE. The predictors are float32, so `exp` overflowed to inf and
            # only THEN got cast, returning a float64 infinity where the mathematical answer is
            # finite and representable -- exp(89) is 4.49e38, fine in float64 and past float32's
            # 3.4e38 ceiling. Casting first costs nothing and the result is the advertised dtype.
            eta_size, _ = eta
            if seasonal is not None:
                # Match the ordinary forward at a zero factor offset before applying the exact
                # lognormal variance correction. This addition is intentionally float32: it is
                # the fitted head's implemented linear predictor, then the analytic moment is
                # evaluated in float64 to avoid overflow.
                eta_size = eta_size + seasonal[1]
            if model.factor is None:
                # Rank zero is still part of the exact plain-NB branch. The old factorless
                # shortcut called ``apply_factor`` first, so ``exp(eta_size)`` happened in
                # float32 and could become infinity even when the exact mean (for example
                # exp(100)) is finite in the float64 return type promised by this routine.
                size_variance = torch.zeros(
                    eta_size.shape[0], dtype=torch.float64, device=eta_size.device
                )
            else:
                cov = _loading_covariance(model, base.store_index)
                size_variance = cov[:, 1, 1].double()
            exponent = eta_size.double() + 0.5 * size_variance.unsqueeze(-1)
            limit = math.log(torch.finfo(torch.float64).max)
            invalid = ~torch.isfinite(exponent) | (exponent > limit)
            if bool(invalid.any()):
                finite = exponent[torch.isfinite(exponent)]
                largest = float(finite.max()) if finite.numel() else float("nan")
                raise OursError(
                    f"the exact plain-NB factor mean overflows float64 in "
                    f"{int(invalid.sum())} cell-step(s) (largest finite log-mean "
                    f"{largest:.3f}, float64 limit {limit:.3f}); returning infinity would "
                    "corrupt scoring"
                )
            return torch.exp(exponent).cpu()

        if model.factor is None:
            # Nothing latent to integrate. Keep the implemented float32 hurdle head exactly, but
            # never let a non-finite conditional mean escape as a valid point forecast.
            p, mu, k = model.apply_factor(base, n_draws=1)
            value = predicted_mean(p[0], mu[0], k[0]).cpu().double()
            if not bool(torch.isfinite(value).all()):
                raise OursError(
                    "the factorless hurdle mean is non-finite; returning it would corrupt scoring"
                )
            return value

        cov, covariance_rank = _loading_geometry(model, base.store_index)
        eta_occ, eta_size, log_k = eta

        if bool((~torch.isfinite(cov)).any()):
            raise OursError("factor loading covariance is non-finite; quadrature cannot certify it")
        occurrence_sd = cov[:, 0, 0].clamp_min(0.0).sqrt()
        size_sd = cov[:, 1, 1].clamp_min(0.0).sqrt()
        largest_occurrence_sd = float(occurrence_sd.max())
        largest_size_sd = float(size_sd.max())
        if largest_occurrence_sd > GH_MAX_OCCURRENCE_SD or largest_size_sd > GH_MAX_SIZE_SD:
            raise OursError(
                "factor loadings lie outside the centred-GH envelope validated for this "
                f"implementation: occurrence sd max {largest_occurrence_sd:.6g} "
                f"(limit {GH_MAX_OCCURRENCE_SD}), size sd max {largest_size_sd:.6g} "
                f"(limit {GH_MAX_SIZE_SD}). Successive centred rules can agree vacuously on "
                "clamp plateaus outside this envelope; use a separately validated tilted/QMC "
                "or Monte Carlo point route"
            )

        tail_eta_occ = eta_occ if seasonal is None else eta_occ + seasonal[0]
        tail_eta_size = eta_size if seasonal is None else eta_size + seasonal[1]
        tail_bound = _unrepresented_tail_bound(cov, tail_eta_occ, tail_eta_size, log_k)
        unsafe_tail = ~torch.isfinite(tail_bound) | (tail_bound > tolerance)
        if bool(unsafe_tail.any()):
            finite = tail_bound[torch.isfinite(tail_bound)]
            largest = float(finite.max()) if finite.numel() else float("inf")
            raise OursError(
                f"{int(unsafe_tail.sum())} cell-step(s) have an unrepresented Gaussian-tail "
                f"hurdle-mean bound above tolerance {tolerance:.3e} beyond the final "
                f"+/-{GH_OUTER_STANDARD_NODE:.3f}-sigma centred GH projections "
                f"(largest finite bound {largest:.3e}). Successive rules cannot certify a "
                "clamp plateau or rare lognormal tail they never evaluate; use a separately "
                "validated tilted/QMC or Monte Carlo point route"
            )

        def integrate(order: int, index: torch.Tensor, rank: int) -> torch.Tensor:
            selected_cov = cov.index_select(0, index)
            offsets, weights = (
                _gauss_hermite_offsets_1d(selected_cov, order)
                if rank == 1
                else _gauss_hermite_offsets(selected_cov, order)
            )
            selected_occ = eta_occ.index_select(0, index)
            selected_size = eta_size.index_select(0, index)
            selected_k = log_k.index_select(0, index)
            selected_seasonal = (
                None
                if seasonal is None
                else (
                    seasonal[0].index_select(0, index),
                    seasonal[1].index_select(0, index),
                )
            )
            total = None
            # CHUNKED OVER NODES for the same reason the sampled loop is chunked over draws: a
            # 15x15 rule over the full catalogue would materialise [225, 30490, 28] three times
            # over (p, mu, k), which is several GB and turns a compute-bound integral into a
            # memory-bound one.
            #
            # NOT exactly invariant to the chunk size, and an earlier comment here wrongly said
            # it was. Floating summation is order-sensitive and the head evaluates in float32, so
            # changing NODE_CHUNK moves the result by of order the float32 noise floor -- enough,
            # in one measured case, to change which quadrature order the gate settled on. The
            # accumulator is float64 to keep that as small as possible, but the honest statement
            # is that this knob is part of the computation, not free.
            for start in range(0, offsets.shape[0], NODE_CHUNK):
                stop = min(start + NODE_CHUNK, offsets.shape[0])
                occ = offsets[start:stop, :, 0].unsqueeze(-1)  # [chunk, cells, 1]
                size = offsets[start:stop, :, 1].unsqueeze(-1)
                if selected_seasonal is not None:
                    occ = occ + selected_seasonal[0].unsqueeze(0)
                    size = size + selected_seasonal[1].unsqueeze(0)
                p, mu, k = model.head.from_predictors(
                    selected_occ.unsqueeze(0),
                    selected_size.unsqueeze(0),
                    selected_k.unsqueeze(0),
                    occ,
                    size,
                )
                values = predicted_mean(p, mu, k.expand_as(mu))  # [chunk, cells, horizon]
                part = (values.double() * weights[start:stop].double().view(-1, 1, 1)).sum(0)
                total = part if total is None else total + part
            return total

        result = torch.empty_like(eta_size, dtype=torch.float64)

        # RANK ZERO IS NOT A QUADRATURE PROBLEM. Evaluating repeated zero offsets under several
        # rules used to make a deterministic answer refuse solely because rounded weights summed
        # in a different order. One head evaluation is the exact integral of a point mass.
        zero = torch.nonzero(covariance_rank == 0, as_tuple=False).flatten()
        if zero.numel():
            zero_seasonal = (
                None
                if seasonal is None
                else (
                    seasonal[0].index_select(0, zero),
                    seasonal[1].index_select(0, zero),
                )
            )
            p, mu, k = model.head.from_predictors(
                eta_occ.index_select(0, zero),
                eta_size.index_select(0, zero),
                log_k.index_select(0, zero),
                None if zero_seasonal is None else zero_seasonal[0],
                None if zero_seasonal is None else zero_seasonal[1],
            )
            result.index_copy_(0, zero, predicted_mean(p, mu, k).double())

        orders = [o for o in GH_ORDERS if o >= nodes]
        for rank in (1, 2):
            index = torch.nonzero(covariance_rank == rank, as_tuple=False).flatten()
            if not index.numel():
                continue
            values = [integrate(orders[0], index, rank), integrate(orders[1], index, rank)]
            previous_gap = (values[0] - values[1]).abs()
            settled = torch.zeros_like(previous_gap, dtype=torch.bool)
            rank_result = values[1].clone()
            for order in orders[2:]:
                values.append(integrate(order, index, rank))
                gap = (values[-2] - values[-1]).abs()
                penultimate_gap = previous_gap

                # A finite pair of successive-rule gaps is NOT a mathematical error bound. The
                # previous contraction heuristic false-accepted a measured case 1.46e-4 away
                # from an independent converged integral under a declared 1e-4 tolerance. Keep
                # only the conservative stability screen supported by the fitted catalogue: BOTH
                # finest gaps must have collapsed to the float32 head's evaluation-noise floor,
                # separately for every cell and horizon, while BOTH gaps must also remain inside
                # the requested absolute tolerance. The relative noise estimate controls when a
                # difference is no longer resolved; it is not a maximum forecast scale. Only
                # cells with observed gaps outside those criteria enter randomized QMC below.
                scale = torch.maximum(
                    torch.maximum(values[-3].abs(), values[-2].abs()), values[-1].abs()
                )
                noise = (
                    GH_NOISE_ULPS
                    * float(torch.finfo(torch.float32).eps)
                    * torch.maximum(scale, torch.ones_like(scale))
                )
                floor = torch.maximum(torch.full_like(noise, tolerance * GH_NEGLIGIBLE), noise)
                stable = (
                    (penultimate_gap <= floor)
                    & (gap <= floor)
                    & (penultimate_gap <= tolerance)
                    & (gap <= tolerance)
                )
                newly_settled = stable & ~settled
                rank_result = torch.where(newly_settled, values[-1], rank_result)
                settled |= stable
                if bool(settled.all()):
                    result.index_copy_(0, index, rank_result)
                    break
                previous_gap = gap
            else:
                failed = ~settled
                rule = f"{orders[-1]}-node" if rank == 1 else f"{orders[-1]}x{orders[-1]}"
                # Keep every cell-step at the first order where its own three-rule screen settled;
                # an unrelated hard cell may make the loop compute another order, but cannot change
                # an already-settled value. Evaluate QMC only for CELLS containing a failure, then
                # replace only their unresolved steps.
                result.index_copy_(0, index, rank_result)
                failed_cells = failed.any(dim=-1)
                local_index = torch.nonzero(failed_cells, as_tuple=False).flatten()
                qmc_index = index.index_select(0, local_index)
                qmc_seasonal = (
                    None
                    if seasonal is None
                    else (
                        seasonal[0].index_select(0, qmc_index),
                        seasonal[1].index_select(0, qmc_index),
                    )
                )
                qmc_value, refinement, replication_spread = _randomized_qmc_integral(
                    model,
                    cov.index_select(0, qmc_index),
                    eta_occ.index_select(0, qmc_index),
                    eta_size.index_select(0, qmc_index),
                    log_k.index_select(0, qmc_index),
                    rank,
                    qmc_seasonal,
                )
                qmc_needed = failed.index_select(0, local_index)
                qmc_stable = (
                    torch.isfinite(qmc_value)
                    & torch.isfinite(refinement)
                    & torch.isfinite(replication_spread)
                    & (refinement <= tolerance)
                    & (replication_spread <= tolerance)
                )
                unresolved = qmc_needed & ~qmc_stable
                if bool(unresolved.any()):
                    raise OursError(
                        f"the rank-{rank} factor quadrature did not become numerically stable: "
                        f"{int(failed.sum())} cell-step(s) failed GH rules through {rule}, and "
                        f"{int(unresolved.sum())} remained unsettled under deterministic "
                        f"replicated randomized QMC (largest GH penultimate gap "
                        f"{float(penultimate_gap.max()):.3e}, last gap {float(gap.max()):.3e}; "
                        f"largest QMC coarse/fine shift {float(refinement.max()):.3e}, "
                        f"independent-scramble spread {float(replication_spread.max()):.3e}; "
                        f"tolerance {tolerance:.3e}). Neither successive-rule agreement nor "
                        "replicated-QMC agreement is a general error bound, so this integral is "
                        "refused rather than returned as certified"
                    )
                gh_for_qmc_cells = rank_result.index_select(0, local_index)
                merged = torch.where(qmc_needed, qmc_value, gh_for_qmc_cells)
                result.index_copy_(0, qmc_index, merged)
        return result.cpu()


def draw_paths_over_factor(
    model: CustomModel,
    inputs: dict,
    n_paths: int,
    *,
    factor_draws: int | None = None,
    seed: int = 0,
    rng_domain_scheme: str = RNG_DOMAIN_SCHEME,
) -> torch.Tensor:
    """Sample paths over the factor, or independent predictive marginals when it is absent.

    **This is what makes the interval the model's own predictive distribution.** An earlier
    version called the model with `n_draws=1`, took the single factor realisation it returned, and
    drew all 1,000 paths from it -- `sample_paths` says in its own signature that its parameters
    are "already factor-conditioned". Every path then shared one demand shock, so the factor
    contributed nothing to the spread.

    That is not a small omission and it lands hardest exactly where Gate 4 looks. The factor is
    what makes days within a lead time move TOGETHER; conditioning on one draw removes the
    cross-day dependence from the sampled sum, and the variance of the sum is the whole quantity a
    cumulative lead-time interval is about. Measured on the full panel with one factor draw:
    50% covered 46%, 80% covered 73%, 95% covered 88% -- too narrow at every level.

    Args:
        model: A fitted `CustomModel`.
        inputs: Its inputs, exactly as `fit_predict` assembles them.
        n_paths: Total paths to return.
        factor_draws: Distinct factor realisations to integrate over, or None for one per emitted
            path. Supplying fewer is an explicit Monte Carlo approximation; the default preserves
            the advertised effective path count.
        seed: Reproducibility. A gate result that cannot be reproduced is not one.
        rng_domain_scheme: Versioned derivation of independent factor/count streams. The legacy
            value is accepted only so a checkpoint written before this fix can be replayed.

    Returns:
        `[n_paths, n_cells, horizon]` integer paths.

    Raises:
        OursError: `factor_draws` below 1 or above `n_paths`, or `n_paths` not divisible by it --
            an uneven split would weight some factor realisations more than others, which is a
            silent reweighting of the predictive distribution.
    """
    if factor_draws is None:
        factor_draws = default_factor_draws(n_paths)
    if not 1 <= factor_draws <= n_paths:
        raise OursError(f"factor_draws must lie in [1, n_paths={n_paths}], got {factor_draws}")
    if n_paths % factor_draws:
        raise OursError(
            f"n_paths={n_paths} is not divisible by factor_draws={factor_draws}. An uneven split "
            "would give some factor realisations more sample paths than others, which reweights "
            "the predictive distribution without saying so"
        )
    per_draw = n_paths // factor_draws

    count_seed = _domain_seed(seed, "count", rng_domain_scheme)
    factor_seed = _domain_seed(seed, "factor", rng_domain_scheme)
    generator = torch.Generator().manual_seed(count_seed)
    if model.factor is None:
        # One parameter evaluation, then independent count uniforms for every path/cell/step.
        # This is the product of predictive marginals promised by bottom_up; repeating a shared
        # latent scenario would silently put the removed dependence back through the sampler.
        with _seeded_rng(model, factor_seed), evaluating(model), torch.no_grad():
            p, mu, k = model(**{**inputs, "n_draws": 1, "factor_path": None})
        return predicted_paths(
            None if p is None else p[0].cpu(), mu[0].cpu(), k[0].cpu(), n_paths, generator=generator
        )
    out, row = None, 0
    # ONE factor realisation at a time. The support table inside `sample_paths` is
    # [cells, horizon, support], and holding several draws' worth at once is how a previous
    # version reached 4.3 billion floats and was killed.
    #
    # The factor draws are seeded through `_seeded_rng`, not left on the global stream: before
    # that, "seed=0" reproduced the path UNIFORMS while the factor realisations depended on
    # everything that had consumed the global RNG first -- so the same call in a fresh process
    # returned different paths, and a gate figure that cannot be reproduced is not one.
    # ENCODE ONCE -- this loop was the worse offender of the two: one full evaluation of the
    # deterministic stack per factor realisation, i.e. 1,000 of them at the default, all
    # producing the same numbers.
    with evaluating(model), torch.no_grad():
        path_base = model.encode_base(**_base_inputs(inputs))
    with _seeded_rng(model, factor_seed):
        for _ in range(factor_draws):
            with evaluating(model), torch.no_grad():
                p, mu, k = model.apply_factor(path_base, n_draws=1)
            chunk = predicted_paths(
                None if p is None else p[0].cpu(),
                mu[0].cpu(),
                k[0].cpu(),
                per_draw,
                generator=generator,
            )
            if out is None:
                # Preallocated and filled in place: `torch.cat` over the whole list would hold
                # the parts and the result at once, and the result alone is
                # n_paths x cells x horizon.
                out = torch.empty((n_paths, *chunk.shape[1:]), dtype=chunk.dtype)
            out[row : row + per_draw] = chunk
            row += per_draw
    return out


@contextlib.contextmanager
def _seeded_rng(model: torch.nn.Module, seed: int | None):
    """Deterministic GLOBAL RNG for the block, restored afterwards. No-op when `seed` is None.

    The factor paths inside `forward` are drawn from the global stream, so before this existed
    a "seeded" sampling call was only as reproducible as everything that had consumed the
    stream first -- model construction, training, an unrelated parameter added to a module.
    Adding one embedding table shifted every factor realisation downstream, which is exactly
    the fragility the flaky-test note in `test_ours._factor_model` records. Forking the RNG
    makes the draw a function of the seed alone, and restores the caller's stream so a
    training loop that resumes afterwards is not silently re-seeded.

    CUDA state is forked alongside CPU state when the model lives there. MPS has no
    fork/restore API in this torch version, so on MPS the global MPS stream is left at
    wherever the block moved it -- deterministic draws, imperfect restoration, and the
    trade is stated here rather than made silently.
    """
    if seed is None:
        yield
        return
    device = next(model.parameters()).device
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


@contextlib.contextmanager
def evaluating(model: torch.nn.Module):
    """Put a model in eval mode for the block, then restore whatever it was.

    **Because dropout exists now, and inference must not sample it.** `demand_branch_dropout`
    randomises the branch tokens; left on at inference it makes the same inputs give different
    forecasts and silently widens every sampled interval, which would read as honest uncertainty
    and is instead a training regulariser leaking into the answer.

    Restores rather than forcing eval permanently: the training loop calls these entry points
    between epochs, and leaving the model in eval afterwards would turn the regulariser off for
    the rest of the run. The complete module tree is restored, not merely ``model.training``.
    PyTorch permits a mixed tree such as an eval root with one dropout-bearing child explicitly
    in train mode; calling ``model.train(root_mode)`` on exit destroys that configuration.
    """
    modes = tuple((module, module.training) for module in model.modules())
    model.eval()
    try:
        yield model
    finally:
        # ``Module.train`` recurses and would repeatedly overwrite child-specific modes. Setting
        # the flags captured from the same module tree is exactly the state transition train/eval
        # ultimately performs, without flattening a deliberately mixed configuration.
        for module, training in modes:
            module.training = training


def model_diagnostics(model: CustomModel) -> dict[str, object]:
    """What the fitted model's own parameters say, as numbers rather than as a loss curve.

    **Written because a run that reports only MAE cannot answer why.** The ladder's fourth rung
    added per-series annual amplitude and improved MAE slightly, and there was no way afterwards
    to tell whether the per-series deviations had moved at all -- the model was not checkpointed
    and the measurement had not been recorded. A finding that cannot be recovered from the
    artefacts is a finding that has to be re-run for.

    What each number is for:

      * `seasonality_deviation` -- mean absolute per-series annual deviation, occurrence and size.
        Near zero means the shrinkage held and POOLED seasonality was sufficient on this panel,
        which is a result and not a failure to train.
      * `factor_phi` -- temporal persistence per rank: learned under AR(1), exact zero under IID.
        This is the parameter the cumulative lead-time interval depends on: low persistence means
        days inside a lead time barely move together, so the variance of their SUM is small.
      * `factor_innovation` -- `sqrt(1 - phi^2)`, reported alongside because the marginal variance
        is held at 1 and it is the pair that is interpretable, not either alone.
        Both factor diagnostics are absent, with `factor.present=false`, for resolved bottom-up
        models: reporting zero persistence would falsely imply a white-noise shared factor.
      * `event_kernel` -- magnitude of the calendar kernel's shared shape and its per-event
        deviation, the same shrunk-deviation question as seasonality but for named events.
    """
    out: dict[str, object] = {}
    if model.seasonality is not None:
        occ, size = model.seasonality.deviation_scale()
        out["seasonality_deviation"] = {
            "occurrence": occ,
            "size": size,
            "shrink": model.seasonality.shrink,
        }
    if model.factor is None:
        out["factor"] = {
            "present": False,
            "topology": "bottom_up",
            "dependence": "independent predictive marginals",
        }
    else:
        out["factor"] = {
            "present": True,
            "topology": "bottom_up_factor",
            "rank": model.factor.rank,
            "dynamics": model.factor.factor_dynamics,
        }
        phi = model.factor.phi.detach()
        out["factor_phi"] = {
            "per_rank": [round(float(v), 4) for v in phi],
            "mean": round(float(phi.mean()), 4),
        }
        out["factor_innovation"] = round(float(model.factor.innovation_scale.detach().mean()), 4)
    if model.selection is not None:
        # Uniform means the model found no attribute more useful than another, which is a finding
        # about the panel rather than a failure to train.
        out["static_selection"] = {
            "n_groups": model.selection.n_groups,
            "widths": list(model.selection.widths),
        }
    out["price_constrained"] = model.price_index is not None
    out["price_response"] = {
        "mode": model.price_response_mode,
        "structurally_monotone": model.price_response is not None,
        "coordinate": (
            "discount_depth_v1" if model.price_response is not None else "raw_price_and_depth_v1"
        ),
        "cross_horizon": "diagonal" if model.price_response is not None else "unrestricted",
        "occurrence_slope_max": model.price_occurrence_slope_max,
        "size_slope_max": model.price_size_slope_max,
    }
    projection = model.aggregate_trajectory_projection
    out["aggregate_trajectory"] = {
        "mode": model.aggregate_trajectory_mode,
        "present": projection is not None,
        "trainable_parameter": (
            None if projection is None else AGGREGATE_TRAJECTORY_PROJECTION_PARAMETER
        ),
        "weight_sha256": _projection_weight_sha256(model),
        "weight": (None if projection is None else projection.weight.detach().cpu().tolist()),
    }
    if model.calendar is not None:
        out["event_kernel"] = {
            "shared_magnitude": round(float(model.calendar.shared.detach().abs().mean()), 6),
            "deviation_magnitude": round(float(model.calendar.deviation.detach().abs().mean()), 6),
            "shrink": model.calendar.shrink,
        }
        if model.calendar.series_deviation is not None:
            # Near zero means the shrinkage held and the POOLED event response was sufficient
            # on this panel -- a finding, not a failure to train. Same reading as seasonality.
            out["event_kernel"]["series_deviation"] = round(
                float(model.calendar.series_deviation.weight.detach().abs().mean()), 6
            )
            out["event_kernel"]["series_shrink"] = model.calendar.series_shrink
    return out


def factor_block_batches(
    rows: torch.Tensor, group_index: torch.Tensor, batch_size: int, generator: torch.Generator
) -> list[torch.Tensor]:
    """Pack complete shared-factor blocks into shuffled optimiser batches.

    `CustomModel.loss` sums cell log likelihoods inside each factor block before applying the
    IWAE `logmeanexp`. A row-wise minibatch silently changes that objective whenever two stores
    sharing a path land in different steps: each fragment redraws the common latent shock and is
    scored as an independent block. This sampler treats blocks as indivisible.

    `batch_size` is a soft row cap. A single block larger than it is yielded whole because there
    is no mathematically faithful way to split that block across optimiser steps.
    """
    if batch_size < 1:
        raise OursError(f"batch_size must be >= 1, got {batch_size}")
    if rows.dim() != 1 or group_index.dim() != 1:
        raise OursError(
            f"rows and group_index must be rank 1, got {tuple(rows.shape)} and "
            f"{tuple(group_index.shape)}"
        )
    rows_cpu = rows.detach().cpu().long()
    groups_cpu = group_index.detach().cpu().long()
    if rows_cpu.numel() == 0:
        return []
    if int(rows_cpu.min()) < 0 or int(rows_cpu.max()) >= len(groups_cpu):
        raise OursError(f"batch row indices lie outside group_index with {len(groups_cpu)} rows")

    blocks: dict[int, list[int]] = {}
    for row in rows_cpu.tolist():
        blocks.setdefault(int(groups_cpu[row]), []).append(row)
    labels = list(blocks)
    order = torch.randperm(len(labels), generator=generator).tolist()

    packed: list[torch.Tensor] = []
    current: list[int] = []
    for position in order:
        block = blocks[labels[position]]
        if current and len(current) + len(block) > batch_size:
            packed.append(torch.tensor(current, dtype=torch.long, device=rows.device))
            current = []
        current.extend(block)
        if len(current) >= batch_size:
            packed.append(torch.tensor(current, dtype=torch.long, device=rows.device))
            current = []
    if current:
        packed.append(torch.tensor(current, dtype=torch.long, device=rows.device))
    return packed


def _factor_block_microbatches(
    rows: torch.Tensor,
    group_index: torch.Tensor,
    row_cap: int,
) -> list[torch.Tensor]:
    """Pack whole factor blocks without another shuffle or optimiser-step boundary.

    ``factor_block_batches`` chooses and shuffles the LOGICAL optimiser batches. This second
    partition is purely physical: every returned piece contributes gradients to that same update.
    Keeping the incoming first-seen block order avoids consuming another random stream merely
    because a machine needs smaller activation tensors.
    """
    if isinstance(row_cap, bool) or not isinstance(row_cap, int) or row_cap < 1:
        raise OursError(f"row_cap must be a positive integer, got {row_cap!r}")
    if rows.dim() != 1 or group_index.dim() != 1:
        raise OursError(
            f"rows and group_index must be rank 1, got {tuple(rows.shape)} and "
            f"{tuple(group_index.shape)}"
        )
    rows_cpu = rows.detach().cpu().long()
    groups_cpu = group_index.detach().cpu().long()
    if rows_cpu.numel() == 0:
        return []
    if int(rows_cpu.min()) < 0 or int(rows_cpu.max()) >= len(groups_cpu):
        raise OursError(
            f"microbatch row indices lie outside group_index with {len(groups_cpu)} rows"
        )

    blocks: dict[int, list[int]] = {}
    for row in rows_cpu.tolist():
        blocks.setdefault(int(groups_cpu[row]), []).append(row)
    packed: list[torch.Tensor] = []
    current: list[int] = []
    for block in blocks.values():
        if current and len(current) + len(block) > row_cap:
            packed.append(torch.tensor(current, dtype=torch.long, device=rows.device))
            current = []
        current.extend(block)
        if len(current) >= row_cap:
            packed.append(torch.tensor(current, dtype=torch.long, device=rows.device))
            current = []
    if current:
        packed.append(torch.tensor(current, dtype=torch.long, device=rows.device))
    return packed


def _backward_logical_batch(
    model: CustomModel,
    tensors: dict,
    target: torch.Tensor,
    target_state: torch.Tensor,
    batch_rows: torch.Tensor,
    *,
    target_probs: torch.Tensor | None,
    normalization_count: float,
    gradient_microbatch_size: int,
    covariate_dropout: float,
    channel_columns: list[tuple[int, int]],
    price_penalty_weight: float,
    n_optimizer_batches: int,
    total_price_eligible: int,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Backpropagate one logical update and return price violation plus objective value."""
    # ``tensors`` and targets are the CPU-resident Torch time-series dataset.  A logical batch
    # preserves the shared-factor objective and dropout realization; only its physical pieces
    # cross to the accelerator, so gradient_microbatch_size is a real device-memory control.
    logical_idx = batch_rows.to(target.device)
    logical = _subset(tensors, logical_idx)
    if covariate_dropout and logical.get("commercial_mask") is not None:
        # Sample availability dropout once over the logical batch, then partition that realised
        # training example. Physical memory must not change which covariates the update saw.
        before = logical["commercial_mask"]
        logical["commercial"], logical["commercial_mask"] = drop_covariates(
            logical["commercial"], before, covariate_dropout, generator=None
        )
        logical["known_future"] = _hide_known_channels(
            logical["known_future"], before & ~logical["commercial_mask"], channel_columns
        )

    logical_target = target[logical_idx]
    logical_state = target_state[logical_idx]
    logical_probs = None if target_probs is None else target_probs[logical_idx]
    relative_rows = torch.arange(len(logical_idx), device=logical_idx.device)
    pieces = _factor_block_microbatches(
        relative_rows, logical["group_index"], gradient_microbatch_size
    )
    if not pieces:
        raise OursError("an empty logical optimiser batch cannot produce an update")

    backward_calls = 0
    objective_total = 0.0
    violation_cells = 0.0
    price_cells = 0
    dev = _model_device(model)
    for relative in pieces:
        micro = _to_model_device(_subset(logical, relative), dev)
        micro_target = logical_target[relative].to(dev)
        micro_state = logical_state[relative].to(dev)
        micro_probs = None if logical_probs is None else logical_probs[relative].to(dev)
        with (
            torch.autocast(dev.type, dtype=torch.float16, cache_enabled=False)
            if scaler is not None else contextlib.nullcontext()
        ):
            objective = None
            if _informative_mass(micro_target, micro_state, micro_probs) > 0.0:
                objective = model.loss(
                    micro_target,
                    micro_state,
                    normalization_count=normalization_count,
                    state_probs=micro_probs,
                    **micro,
                )
            if price_penalty_weight and model.price_index is not None:
                local_price_cells = int(
                    _price_intervention_mask(
                        micro, model.price_index, model.price_available_index
                    ).sum()
                )
                batch_weight = _price_penalty_batch_weight(
                    micro,
                    model.price_index,
                    model.price_available_index,
                    n_batches=n_optimizer_batches,
                    total_eligible=total_price_eligible,
                )
                if batch_weight:
                    penalty, rate = model.price_penalty(micro)
                    penalty_term = price_penalty_weight * batch_weight * penalty
                    objective = penalty_term if objective is None else objective + penalty_term
                    violation_cells += rate * local_price_cells
                    price_cells += local_price_cells
        if objective is not None:
            if scaler is not None and not bool(torch.isfinite(objective)):
                raise FloatingPointError("legacy mixed-precision batch has nonfinite loss")
            objective_total += float(objective.detach())
            (scaler.scale(objective) if scaler is not None else objective).backward()
            backward_calls += 1
    if backward_calls == 0:
        raise OursError(
            "a logical optimiser batch has neither informative targets nor eligible prices"
        )
    return (float("nan") if price_cells == 0 else violation_cells / price_cells, objective_total)


def _complete_legacy_optimizer_batch(
    model: CustomModel,
    optimizer: torch.optim.Optimizer,
    backward: Callable[[], tuple[float, float]],
    *,
    scaler: torch.amp.GradScaler | None,
    zero_oov_gradients: Callable[[], None],
) -> tuple[float, float, float, int]:
    """Retry one logical batch at a lower AMP scale without losing its optimizer update."""
    def scaled_step() -> bool:
        assert scaler is not None
        calls = {"count": 0}
        handle = optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: calls.__setitem__("count", calls["count"] + 1)
        )
        try:
            before = scaler.get_scale()
            if not math.isfinite(before) or before <= 0:
                raise FloatingPointError("legacy AMP has invalid loss scale")
            scaler.step(optimizer)
            scaler.update()
            after = scaler.get_scale()
        finally:
            handle.remove()
        if not math.isfinite(after) or after <= 0:
            raise FloatingPointError("legacy AMP produced invalid loss scale")
        if calls["count"] not in (0, 1):
            raise RuntimeError("legacy AMP completed multiple optimizer updates")
        if not calls["count"] and after >= before:
            # MPS can report a skipped raw step while the visible gradients and
            # automatic scale remain unchanged. Back off, then replay this batch.
            scaler.update(new_scale=before * 0.5)
        return bool(calls["count"])

    device = _model_device(model)
    cpu_rng = torch.get_rng_state() if scaler is not None else None
    device_rng = (
        torch.cuda.get_rng_state(device) if scaler is not None and device.type == "cuda"
        else torch.mps.get_rng_state() if scaler is not None and device.type == "mps"
        else None
    )
    for retry in range(24 if scaler is not None else 1):
        if retry:
            assert cpu_rng is not None
            torch.set_rng_state(cpu_rng)
            if device.type == "cuda":
                torch.cuda.set_rng_state(device_rng, device)
            elif device.type == "mps":
                torch.mps.set_rng_state(device_rng)
        optimizer.zero_grad()
        violation, objective = backward()
        zero_oov_gradients()
        if scaler is not None:
            scaler.unscale_(optimizer)
        from autoforecast.amp_gradients import clip_gradients

        norm, component_overflow = clip_gradients(
            model.parameters(), 1.0, allow_amp_overflow=scaler is not None
        )
        if component_overflow:
            if scaled_step():
                raise RuntimeError("GradScaler stepped despite nonfinite gradients")
            continue
        if scaler is None:
            optimizer.step()
        else:
            if not scaled_step():
                continue
        return violation, objective, float(norm), retry
    raise FloatingPointError("legacy AMP could not complete an optimizer update in 24 tries")


def _microbatched_validation_loss(
    model: CustomModel,
    tensors: dict,
    target: torch.Tensor,
    target_state: torch.Tensor,
    rows: torch.Tensor,
    *,
    target_probs: torch.Tensor | None,
    normalization_count: float,
    gradient_microbatch_size: int,
    training_identity_evidence: dict[str, object],
    training_oov_rows: dict[str, int],
    generator: torch.Generator | None = None,
) -> float:
    """Evaluate one coherent validation checkpoint without a catalogue-sized forward."""
    if not math.isfinite(normalization_count) or normalization_count <= 0.0:
        raise OursError(
            f"validation normalization_count must be finite and positive, got "
            f"{normalization_count!r}"
        )
    pieces = _factor_block_microbatches(rows, tensors["group_index"], gradient_microbatch_size)
    total = 0.0
    dev = _model_device(model)
    for piece in pieces:
        idx = piece.to(target.device)
        micro_target = target[idx].to(dev)
        micro_state = target_state[idx].to(dev)
        micro_probs = None if target_probs is None else target_probs[idx].to(dev)
        if _informative_mass(micro_target, micro_state, micro_probs) == 0.0:
            continue
        inputs = _pool_unevidenced_indices(
            _to_model_device(_subset(tensors, idx), dev),
            training_identity_evidence,
            training_oov_rows,
        )
        if generator is not None:
            inputs["generator"] = generator
        total += float(
            model.loss(
                micro_target,
                micro_state,
                normalization_count=normalization_count,
                state_probs=micro_probs,
                **inputs,
            )
        )
    return total


def split_by_series(
    example_series: torch.Tensor, val_fraction: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split training rows into `(train_rows, val_rows)` so a SERIES is wholly on one side.

    **Rows are not the unit; series are.** With rolling windows one series owns several rows, and
    splitting rows directly puts one of its windows in training and another in validation. The
    per-series residual embedding is then fitted on the very series it is scored on, and the
    validation curve measures recall while looking exactly like generalisation.

    The same argument rules out a split by DAY, which is the form this took before rolling windows
    existed: every day of a series would land on both sides at once.

    Args:
        example_series: `[n_rows]` the series index each training row belongs to.
        val_fraction: Share of SERIES to hold out.
        generator: Seeded, so a gate result is reproducible.

    Returns:
        `(train_rows, val_rows)`, indices into the row axis.

    Raises:
        OursError: `val_fraction` outside (0,1), or too few series to hold any out.
    """
    if not 0.0 < val_fraction < 1.0:
        raise OursError(f"val_fraction must lie in (0,1), got {val_fraction}")
    series = sorted(set(int(s) for s in example_series.tolist()))
    if len(series) < 2:
        raise OursError(
            f"{len(series)} series: a validation split needs at least two, or the held-out set is "
            "the training set"
        )
    shuffled = torch.randperm(len(series), generator=generator)
    n_val = max(1, int(len(series) * val_fraction))
    held_out = {series[i] for i in shuffled[:n_val].tolist()}
    is_val = torch.tensor([int(s) in held_out for s in example_series.tolist()])
    return (~is_val).nonzero(as_tuple=True)[0], is_val.nonzero(as_tuple=True)[0]


def _assemble(
    panel: Panel,
    spark: SparkSession,
    origin: str,
    horizon: int,
    encoder_days: int,
    events,
    event_max_support: int,
    work_dir: pathlib.Path,
    cycles: tuple[tuple[str, float, int], ...] = SEASONAL_CYCLES,
    category_of: dict[str, str] | None = None,
    event_windows: dict[str, tuple[int, int]] | None = None,
    series_predicate: str | None = None,
    level_priors: tuple[dict[str, tuple[float, float]], tuple[float, float]] | None = None,
) -> _Assembly:
    """The encoder window ending at `origin`, and the covariates for the `horizon` days after it.

    Args:
        panel: Data and column roles.
        spark: Active session.
        origin: Last day the encoder may see.
        horizon: Days after `origin` the covariates cover.
        encoder_days: Encoder window length.
        events: `(name, date, known_at)` triples.
        event_max_support: Days either side beyond which an event is marked absent.
        work_dir: Scratch for the Spark builders.

    Returns:
        An `_Assembly`. Its `series_ids` are exactly the series with a FULL encoder window --
        a ragged window would make the model's time axis mean different things for different
        series, so a series without one is declined rather than padded.
    """
    window = demand_history.build(
        panel.path,
        origin,
        encoder_days,
        spark,
        str(work_dir / "_window"),
        series_col=panel.series_col,
        time_col=panel.time_col,
        target_col=panel.target_col,
        exposure_col=panel.exposure_col,
        oos_col=panel.oos_col,
        status_prob_cols=panel.status_prob_cols,
        series_predicate=series_predicate,
        negative_target_policy=panel.negative_target_policy,
    )
    rows = spark.read.parquet(str(window)).orderBy("series_id").collect()
    series_ids = [r["series_id"] for r in rows]
    if not rows:
        # Hash sharding is balanced in expectation, so empty buckets are ordinary (and a non-empty
        # bucket may also have no series with a full encoder window at this origin). Return the
        # canonical empty assembly before reductions that require a rank-2 history tensor. Every
        # caller checks `series_ids` before reading the remaining fields: inference refuses when
        # the whole panel is empty, while the sharded and rolling-window paths skip this piece.
        empty_window = torch.empty((0, encoder_days), dtype=torch.float32)
        return _Assembly(
            series_ids=[],
            history=empty_window,
            observed=torch.empty((0, encoder_days), dtype=torch.bool),
            censored=torch.empty((0, encoder_days), dtype=torch.bool),
            baseline_logit=torch.empty(0),
            baseline_log_size=torch.empty(0),
            commercial=None,
            commercial_mask=None,
            event_distance=None,
            event_present=None,
            known=KnownFuture(torch.empty((0, horizon, 0)), (), frozenset()),
            n_commercial=0,
            n_events=0,
            plan=None,
        )
    history = torch.tensor([r["history"] for r in rows], dtype=torch.float32)
    raw_state = torch.tensor([r["state"] for r in rows], dtype=torch.long)
    state = _states_for_estimand(raw_state, panel.target_estimand)
    raw_oos = [value for r in rows for value in r["oos"]]
    if any(value is None for value in raw_oos):
        raise OursError(
            f"oos column {panel.oos_col!r} contains null inside an encoder window; null shelf "
            "state cannot silently mean in-stock"
        )
    explicit_oos = torch.tensor([r["oos"] for r in rows], dtype=torch.bool)
    # Keep MAGNITUDE information and OOS state separate. A positive lower bound tells the
    # encoder at least how many units sold, so its magnitude is usable with a censor flag. A
    # zero lower bound says nothing about demand magnitude or occurrence, but it still says the
    # shelf was unavailable -- a temporal state the demand branch must see. Collapsing these
    # into one boolean made zero-stockout and never-ranged days identical.
    observed, censored, exact_size_observed = _history_observation_masks(
        history, state, explicit_oos
    )

    # Baselines: the Beta-shrunk lagged frequency and the category-shrunk lagged positive size.
    # Computed from the INFORMATIVE days only -- a masked day is not evidence of a zero.
    category = torch.full((len(series_ids),), -1, dtype=torch.long)
    if category_of:
        levels = sorted({v for v in category_of.values() if v is not None})
        at = {level: i for i, level in enumerate(levels)}
        for i, sid in enumerate(series_ids):
            level = category_of.get(sid)
            if level is not None:
                category[i] = at[level]
    # Occurrence and positive SIZE have different evidence under censoring. A positive lower
    # bound proves occurrence but is not an exact draw from the size distribution; using it in
    # the mean-positive baseline biases size downward by construction.
    prior_override = None
    if level_priors is not None:
        levels_map, (g_rate, g_size) = level_priors
        rate_list, size_list = [], []
        for sid in series_ids:
            level = (category_of or {}).get(sid)
            rate, size_prior = (
                levels_map.get(level, (g_rate, g_size)) if level is not None else (g_rate, g_size)
            )
            rate_list.append(rate)
            size_list.append(size_prior)
        prior_override = (
            torch.tensor(rate_list, dtype=torch.float64),
            torch.tensor(size_list, dtype=torch.float64),
        )
    baseline_logit, baseline_log_size, _ = _shrunk_baselines(
        history, observed, category, exact_size_observed, prior_override
    )

    (price, price_mask, price_horizon, price_available, n_commercial, plan_selection) = (
        _commercial_tensors(
            panel, spark, origin, horizon, series_ids, work_dir, series_predicate=series_predicate
        )
    )
    event_distance, event_present, event_horizon, n_events = _event_tensors(
        events,
        origin,
        horizon,
        series_ids,
        spark,
        work_dir,
        event_max_support,
        event_windows=event_windows,
    )
    commercial_pair = None if price_horizon is None else (price_horizon, price_available)
    known = _known_future(
        origin,
        horizon,
        len(series_ids),
        commercial_pair,
        event_horizon,
        tuple(dict.fromkeys(name for name, _, _ in events)),
        cycles,
        commercial_drivers=_declared_commercial_drivers(panel),
    )
    return _Assembly(
        series_ids=series_ids,
        history=history,
        observed=observed,
        censored=censored,
        baseline_logit=baseline_logit,
        baseline_log_size=baseline_log_size,
        commercial=price,
        commercial_mask=price_mask,
        event_distance=event_distance,
        event_present=event_present,
        known=known,
        n_commercial=n_commercial,
        n_events=n_events,
        plan=plan_selection,
    )
