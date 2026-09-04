"""Autoforecast package initialization."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from autoforecast.integrations.delta import DeltaTable
    from autoforecast.study_registration import (
        DateRange,
        PreparationContext,
        PreparationResult,
        PreparationSettings,
        StageInput,
        StudyConfigValidation,
        StudyPreparation,
        StudyRegistry,
    )


def register_study(
    config: str | pathlib.Path | None = None, *, registry: StudyRegistry | None = None,
    spark: SparkSession | None = None, prepare: StudyPreparation | None = None,
    log_level: str | None = None,
    training_data: StageInput | None = None, testing_data: StageInput | None = None,
    validation_data: StageInput | None = None, series_id: list[str] | None = None,
    date_col: str | None = None, target_col: str | None = None, target_unit: str | None = None,
    frequency: str | None = None, training_date_range: DateRange | None = None,
    testing_date_range: DateRange | None = None, validation_date_range: DateRange | None = None,
    metric: str | None = None, history_missing: str | None = None,
    actual_missing: str | None = None, feature_roles: dict | None = None,
    feature_availability: dict | None = None, references: dict | None = None,
    tuning_windows: list[dict] | None = None, tuning_schedule: dict | None = None,
) -> str:
    """Register data once, then reload the frozen study by its returned ID.

    Choose a YAML/JSON ``config`` path or supply stages directly. Do not mix the
    two modes. ``registry`` sets storage and Spark access; ``spark`` can instead
    provide the session for configured registration. ``log_level`` controls logs.
    ``prepare`` is an optional configured-source hook with validate/prepare methods.

    Direct inputs:
        training_data, testing_data: Required Spark DataFrames (classic or Connect)
            or DeltaTable references. Each input must contain only its declared range.
        validation_data: Optional input. Omit it and its range to skip validation;
            the library will not split your training data.
        series_id: List of columns that identify one series, such as ["product"].
        date_col, target_col: Observation time and numeric value column names.
        target_unit: What the value measures, such as "units".
        frequency: Supported cadence, such as "D", "H", "W", "M" or "2D".
        training_date_range, testing_date_range: Inclusive (start, end) strings.
        validation_date_range: Required exactly when validation_data is supplied.
        metric: Existing scoring metric, such as "mae" or "wrmsse".
        history_missing: Fit-history policy; defaults to "preserve".
        actual_missing: Scoring policy; defaults to "error"; also "zero" or "mask".
        feature_roles: Existing Panel column roles and optional season_length.
            Extra input columns are not automatically used as predictors.
        feature_availability: Existing recipe mapping from a feature to its
            availability-time column. Unknown future values must remain unavailable.
        references: Existing weight/scale settings or reference path for WRMSSE.
        tuning_windows, tuning_schedule: Optional inner windows within training.
            Registering them does not start tuning. A later run must request it.

    Direct stages require a registry and already prepared data. They use a fixed
    training cohort and reject unexpected series. See tutorials 10 and 11 for
    complete examples of stage inputs, configuration and preparation hooks.
    """
    from autoforecast.study_registration import register_study as register

    return register(config,
                    registry=registry,
                    spark=spark,
                    prepare=prepare,
                    log_level=log_level,
                    training_data=training_data,
                    testing_data=testing_data,
                    validation_data=validation_data,
                    series_id=series_id,
                    date_col=date_col,
                    target_col=target_col,
                    target_unit=target_unit,
                    frequency=frequency,
                    training_date_range=training_date_range,
                    testing_date_range=testing_date_range,
                    validation_date_range=validation_date_range,
                    metric=metric,
                    history_missing=history_missing,
                    actual_missing=actual_missing,
                    feature_roles=feature_roles,
                    feature_availability=feature_availability,
                    references=references,
                    tuning_windows=tuning_windows,
                    tuning_schedule=tuning_schedule,
                    )


def validate_study_config(path: str | pathlib.Path) -> StudyConfigValidation:
    """Check local settings and SQL syntax; no source queries or writes are performed."""
    from autoforecast.study_registration import validate_study_config as validate

    return validate(path)


def study_config_schema() -> dict:
    """Return the machine-readable configuration contract as a fresh dictionary."""
    from autoforecast._study_config import study_config_schema as schema

    return schema()


def run_study(study_id, model, **options):
    """Run one model against an existing registered study; see study_workflow."""
    from autoforecast.study_workflow import run_study as run

    return run(study_id, model, **options)


def configure_logging(*, stream=None, progress="auto", interval=10.0):
    """Enable readable library progress; see tutorials/README.md for examples."""
    from autoforecast._execution import configure_logging as configure

    return configure(stream=stream, progress=progress, interval=interval)


__all__ = [
    "validate_study_config",
    "study_config_schema",
    "StudyConfigValidation",
    "register_study",
    "run_study",
    "configure_logging",
    "DeltaTable",
    "StudyRegistry",
    "StudyPreparation",
    "PreparationContext",
    "PreparationResult",
    "PreparationSettings",
]


def __getattr__(name):
    if name == "DeltaTable":
        from autoforecast.integrations.delta import DeltaTable
        return DeltaTable
    if name in {
        "StudyConfigValidation",
        "StudyRegistry",
        "StudyPreparation",
        "PreparationContext",
        "PreparationResult",
        "PreparationSettings",
    }:
        from autoforecast import study_registration
        return getattr(study_registration, name)
    raise AttributeError(name)
