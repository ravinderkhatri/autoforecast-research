"""Prepare and evaluate model-specific readiness checks for registered adapters.

This is the only Spark-aware part of model readiness.  Individual model check functions receive
``ModelCheckInput`` from :mod:`autoforecast.model_api`, never a DataFrame or a storage path.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters import adapter_capabilities, known_adapter_names
from autoforecast.data_admission import get_model_input_contract
from autoforecast.model_api import (
    CheckReport,
    CheckResult,
    DataFacts,
    ModelCheckInput,
    ResolvedRun,
    RunApproval,
)
from autoforecast.panel import ELIGIBLE_STATES, Panel

__all__ = [
    "ModelRegistration",
    "create_resolved_run",
    "model_registration",
    "prepare_model_check_input",
    "run_model_checks",
    "approve_run",
]


@dataclass(frozen=True)
class ModelRegistration:
    """Code-owned readiness semantics for one implemented adapter."""

    adapter: str
    required_history: Callable[[Panel, ResolvedRun], int]
    requires_complete_window: bool = False
    governed_horizon: int | None = None
    sales_only: bool = False
    rejects_soft_availability: bool = False


def _season(panel: Panel, _: ResolvedRun) -> int:
    return panel.season_length


def _two(_: Panel, __: ResolvedRun) -> int:
    return 2


def _classical_season(panel: Panel, _: ResolvedRun) -> int:
    return max(2 * panel.season_length, 12)


def _one(_: Panel, __: ResolvedRun) -> int:
    return 1


def _tft(_: Panel, run: ResolvedRun) -> int:
    return 180 + run.horizon


def _tft_q(panel: Panel, run: ResolvedRun) -> int:
    return get_model_input_contract("tft_q").minimum_history(
        season_length=panel.season_length, horizon=run.horizon
    )


def _timesfm(_: Panel, run: ResolvedRun) -> int:
    return _positive_setting(run, "context_days", 1024)


def _mft(_: Panel, run: ResolvedRun) -> int:
    if run.adapter in {
        "sft",
        "hmfnet",
        "pact",
    }:
        required = run.setting("required_history", None)
        if required is not None:
            return _positive_setting(run, "required_history", 1)
    return _positive_setting(run, "encoder_days", 28) + run.horizon


def _positive_setting(run: ResolvedRun, name: str, default: int) -> int:
    value = run.setting(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            f"{run.adapter} setting {name!r} must be a positive integer, got {value!r}"
        )
    return value


_REGISTRATIONS = {
    "seasonal_naive": ModelRegistration("seasonal_naive", _season, True),
    "croston": ModelRegistration("croston", _two),
    "auto_ets": ModelRegistration("auto_ets", _classical_season),
    "auto_theta": ModelRegistration("auto_theta", _classical_season),
    "shallow_global": ModelRegistration("shallow_global", _one),
    # The adapter explicitly constructs TimeSeriesDataSet with allow_missing_timesteps=True.
    # Its implemented contract is a count of eligible observations, not a gap-free window.
    "tft": ModelRegistration("tft", _tft),
    "tft_q": ModelRegistration(
        "tft_q", _tft_q, requires_complete_window=True, governed_horizon=28
    ),
    "tft_hnb": ModelRegistration(
        "tft_hnb", _tft_q, requires_complete_window=True, governed_horizon=28
    ),
    "autogluon_zoo": ModelRegistration("autogluon_zoo", _one),
    "timesfm_standalone": ModelRegistration(
        "timesfm_standalone",
        _timesfm,
        True,
        governed_horizon=28,
        sales_only=True,
        rejects_soft_availability=True,
    ),
    "mft": ModelRegistration(
        "mft", _mft, True
    ),
    "sft": ModelRegistration(
        "sft", _mft, True
    ),
    "hmfnet": ModelRegistration(
        "hmfnet", _mft, True
    ),
    "pact": ModelRegistration(
        "pact", _mft, True, sales_only=True
    ),
}


def model_registration(adapter: str) -> ModelRegistration:
    """Return the registered semantics or reject an adapter with no readiness contract."""
    from autoforecast.adapter_contracts import descriptor
    definition = descriptor(adapter)
    if adapter not in _REGISTRATIONS:
        contract = get_model_input_contract(adapter)
        def required_history(panel: Panel, run: ResolvedRun) -> int:
            return contract.minimum_history(season_length=panel.season_length, horizon=run.horizon)
        return ModelRegistration(
            adapter, required_history, definition.requires_complete_window,
            sales_only=definition.target_estimands == ("sales_given_availability",),
            rejects_soft_availability=True,
        )
    try:
        return _REGISTRATIONS[adapter]
    except KeyError:
        raise ValueError(f"no model readiness registration for adapter {adapter!r}") from None


def create_resolved_run(
    *,
    model_id: str,
    adapter: str,
    origin: str,
    horizon: int,
    settings: dict[str, object] | None = None,
) -> ResolvedRun:
    """Freeze allowlisted run settings before checking or executing a model."""
    if adapter not in known_adapter_names():
        raise ValueError(f"unknown adapter {adapter!r}; known={list(known_adapter_names())}")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    try:
        origin_day = dt.date.fromisoformat(origin)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"origin must be an ISO date, got {origin!r}") from exc
    if origin_day.isoformat() != origin:
        raise ValueError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}")
    if adapter == "timesfm_standalone":
        allowed = {"context_days"}
    elif adapter == "mft":
        allowed = {"encoder_days"}
    elif adapter in {
        "sft",
        "hmfnet",
        "pact",
    }:
        allowed = {"encoder_days", "required_history"}
    elif adapter in {"tft_q", "tft_hnb"}:
        allowed = {"min_encoder_length"}
    else:
        allowed = set()
    supplied = settings or {}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"settings are not declared by {adapter!r}: {sorted(unknown)}")
    return ResolvedRun(model_id, adapter, origin, horizon, tuple(sorted(supplied.items())))


def prepare_model_check_input(
    panel: Panel,
    spark: SparkSession,
    data: DataFacts,
    run: ResolvedRun,
) -> ModelCheckInput:
    """Measure the exact, bounded evidence required by one registered model checker."""
    registration = model_registration(run.adapter)
    required = registration.required_history(panel, run)
    raw = spark.read.parquet(panel.path).filter(
        F.to_date(F.col(panel.time_col)) <= F.lit(run.origin).cast("date")
    )
    total = raw.select(panel.series_col).distinct().count()
    eligible = raw
    if panel.has_exposure:
        exposure_col = panel.exposure_col
        if exposure_col is None:
            raise ValueError("exposure-backed panel has no exposure column")
        eligible = eligible.filter(F.col(exposure_col).isin(*sorted(ELIGIBLE_STATES)))
    window_start = (dt.date.fromisoformat(run.origin) - dt.timedelta(days=required - 1)).isoformat()
    day = F.to_date(F.col(panel.time_col))
    per_series = eligible.groupBy(panel.series_col).agg(
        F.count(F.lit(1)).alias("history"),
        F.count_distinct(day).alias("dates"),
        F.count_distinct(
            F.when(day >= F.lit(window_start).cast("date"), day)
        ).alias("window_dates"),
        F.max(day).alias("last_date"),
    )
    eligible_series = per_series.count()
    meets_history = per_series.filter(F.col("history") >= required).count()
    irregular = per_series.filter(F.col("history") != F.col("dates")).count()
    complete: int | None = None
    if registration.requires_complete_window:
        complete = per_series.filter(
            (F.col("history") >= required)
            & (F.col("last_date") == F.lit(run.origin).cast("date"))
            & (F.col("window_dates") == required)
        ).count()
    training_examples = None
    positive_training_examples = None
    if run.adapter == "tft":
        training_examples = meets_history
    elif run.adapter in {"tft_q", "tft_hnb"}:
        training_examples = complete
    elif run.adapter in {
        "mft",
        "sft",
        "hmfnet",
        "pact",
    }:
        training_examples = complete
    elif run.adapter == "shallow_global":
        training_examples, positive_training_examples = _shallow_training_evidence(
            eligible, panel, run
        )
    return ModelCheckInput(
        run=run,
        data=data,
        frequency=panel.frequency,
        target_estimand=panel.target_estimand,
        has_probabilistic_availability=panel.status_prob_cols is not None,
        total_series=total,
        eligible_series=eligible_series,
        series_meeting_history=meets_history,
        series_with_complete_window=complete,
        training_examples=training_examples,
        positive_training_examples=positive_training_examples,
        irregular_series=irregular,
        required_history=required,
    )


def _shallow_training_evidence(panel_rows, panel: Panel, run: ResolvedRun) -> tuple[int, int]:
    """Count the exact labelled rows the shallow adapter's default origins can form."""
    origin_day = dt.date.fromisoformat(run.origin)
    origins = tuple(
        (origin_day - dt.timedelta(days=run.horizon * index)).isoformat() for index in range(1, 9)
    )
    total = 0
    positive = 0
    for origin in origins:
        history = (
            panel_rows.filter(F.to_date(F.col(panel.time_col)) <= F.lit(origin).cast("date"))
            .select(panel.series_col)
            .distinct()
        )
        labels = panel_rows.filter(
            (F.to_date(F.col(panel.time_col)) > F.lit(origin).cast("date"))
            & (
                F.to_date(F.col(panel.time_col))
                <= F.date_add(F.lit(origin).cast("date"), run.horizon)
            )
        ).join(history, panel.series_col, "inner")
        row = labels.agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum((F.col(panel.target_col) > 0).cast("long")).alias("positive"),
        ).collect()[0]
        total += int(row["rows"] or 0)
        positive += int(row["positive"] or 0)
    return total, positive


def run_model_checks(input: ModelCheckInput) -> CheckReport:
    """Apply the registered model's checks to prepared evidence only."""
    registration = model_registration(input.run.adapter)
    checks: list[CheckResult] = []
    checks.append(
        CheckResult(
            "model_frequency",
            "Dates match this model",
            "passed" if input.frequency == "D" else "failed",
            "This model accepts daily data."
            if input.frequency == "D"
            else f"This model accepts daily data, not {input.frequency!r} data.",
            scope="model",
        )
    )
    supported = cast(frozenset[str], adapter_capabilities(input.run.adapter)["supported_estimands"])
    checks.append(
        CheckResult(
            "model_forecast_meaning",
            "Forecast meaning fits this model",
            "passed" if input.target_estimand in supported else "failed",
            "This model supports the requested forecast meaning."
            if input.target_estimand in supported
            else "This model does not support the requested forecast meaning.",
            scope="model",
        )
    )
    if registration.sales_only:
        checks.append(
            CheckResult(
                "model_sales_only",
                "Sales forecast required",
                "passed" if input.target_estimand == "sales_given_availability" else "failed",
                "This model forecasts observed sales."
                if input.target_estimand == "sales_given_availability"
                else "This model cannot estimate demand beyond observed sales.",
                scope="model",
            )
        )
    if registration.rejects_soft_availability:
        checks.append(
            CheckResult(
                "model_availability_input",
                "Availability input fits this model",
                "failed" if input.has_probabilistic_availability else "passed",
                "This model can use this availability input."
                if not input.has_probabilistic_availability
                else "This model cannot use probabilistic availability inputs.",
                scope="model",
            )
        )
    if registration.governed_horizon is not None:
        wanted = registration.governed_horizon
        checks.append(
            CheckResult(
                "model_horizon",
                "Forecast length fits this model",
                "passed" if input.run.horizon == wanted else "failed",
                f"This governed model runs with a {wanted}-day forecast."
                if input.run.horizon == wanted
                else f"This governed model requires a {wanted}-day forecast.",
                scope="model",
            )
        )
    enough = input.series_meeting_history
    checks.append(
        CheckResult(
            "model_history",
            "Enough usable history",
            "passed" if enough else "failed",
            f"{enough:,} series meet this model's {input.required_history}-day history requirement."
            if enough
            else f"No series meets this model's {input.required_history}-day history requirement.",
            scope="model",
            numerator=enough,
            denominator=input.total_series,
        )
    )
    if registration.requires_complete_window:
        complete = input.series_with_complete_window or 0
        checks.append(
            CheckResult(
                "model_recent_window",
                "Complete recent history",
                "passed" if complete else "failed",
                f"{complete:,} series have a complete {input.required_history}-day window "
                "ending on the forecast date."
                if complete
                else f"No series has a complete {input.required_history}-day window "
                "ending on the forecast date.",
                scope="model",
                numerator=complete,
                denominator=input.total_series,
            )
        )
    if input.training_examples is not None:
        checks.append(
            CheckResult(
                "model_training_examples",
                "Training examples are available",
                "passed" if input.training_examples else "failed",
                f"{input.training_examples:,} series can supply a complete training example."
                if input.training_examples
                else "No series can supply a complete training example.",
                scope="model",
                numerator=input.training_examples,
                denominator=input.total_series,
            )
        )
    if input.positive_training_examples is not None:
        checks.append(
            CheckResult(
                "model_positive_examples",
                "Positive training examples are available",
                "passed" if input.positive_training_examples else "failed",
                f"{input.positive_training_examples:,} positive training labels are available."
                if input.positive_training_examples
                else "No positive training labels are available for this model's size component.",
                scope="model",
                numerator=input.positive_training_examples,
                denominator=input.training_examples,
            )
        )
    if input.eligible_series < input.total_series and input.eligible_series:
        checks.append(
            CheckResult(
                "model_series_coverage",
                "Series ready to forecast",
                "warning",
                f"{input.total_series - input.eligible_series:,} series have no eligible "
                "history and may be declined.",
                scope="model",
                numerator=input.eligible_series,
                denominator=input.total_series,
            )
        )
    return CheckReport("model", input.data, tuple(checks), model_id=input.run.model_id)


def approve_run(
    data_checks: CheckReport, model_checks: CheckReport, input: ModelCheckInput
) -> RunApproval:
    """Issue a checked authorization only for the immutable request that was evaluated."""
    approved = data_checks.accepted and model_checks.accepted
    reason = (
        "Data checks and model checks passed."
        if approved
        else "Data checks or model checks failed."
    )
    return RunApproval(
        approved,
        input.run.model_id,
        input.run.adapter,
        input.run.config_hash,
        input.data.data_fingerprint,
        reason,
    )
