"""Build caller-declared daily WRMSSE references without customer-specific rules."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, StructType

from autoforecast.preparation import _column, prepare_time_series


def prepare_wrmsse_references(
    data: DataFrame,
    *,
    series_id: list[str],
    date_col: str,
    target_col: str,
    weight_col: str,
    origins: Mapping[str, dt.date] | Sequence[tuple[str, dt.date]],
    missing_history: str,
    duplicate_policy: str,
    weight_window_days: int = 28,
    scale_lag: int = 1,
    scale_start: str = "first_positive",
    validate: bool = True,
) -> DataFrame:
    """Return keys/stage/origin/weight/scale rows accepted by study registration.

    Supply additive daily observations. Origins may repeat a stage, as with several
    tuning windows. Only history through each origin contributes. Weights are net
    sums over the inclusive lookback; scales are mean squared lagged changes from
    the first positive target or first historical row. Scaling is not a metric.

    Missing periods follow the explicit na/zero policy. Explicit null values are
    never silently converted to zeros. Invalid scales/weights raise by default;
    validate=False returns them for check_wrmsse_references, without dropping
    or admitting series. Caller-owned exclusions require a separate decision.
    """
    for name, value in (("weight_window_days", weight_window_days), ("scale_lag", scale_lag)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if scale_start not in {"first_positive", "history_start"}:
        raise ValueError("scale_start must be first_positive or history_start")
    if type(validate) is not bool:
        raise ValueError("validate must be boolean")
    windows = _reference_origins(origins)
    if (
        not series_id
        or len(set([*series_id, date_col, target_col, weight_col])) != len(series_id) + 3
    ):
        raise ValueError("series, date, target and weight columns must be distinct")
    keys = [f"key_{i}" for i in range(len(series_id))]
    history = data.select(
        *[_column(name).alias(key) for name, key in zip(series_id, keys, strict=False)],
        _column(date_col).alias("at"),
        _column(target_col).alias("target"),
        _column(weight_col).alias("revenue"),
    )
    series = history.select(*keys).distinct()
    output = None
    for stage, origin in windows:
        prepared = prepare_time_series(
            history.filter(F.col("at") <= F.lit(origin)),
            series_id=keys,
            date_col="at",
            value_cols=["target", "revenue"],
            source_frequency="D",
            frequency="D",
            fill_missing=missing_history,
            duplicate_policy=duplicate_policy,
            series_end_date=origin,
        )
        recent = F.col("at") >= F.lit(origin - dt.timedelta(days=weight_window_days - 1))
        weights = prepared.groupBy(*keys).agg(
            F.when(
                F.sum(F.when(recent & F.col("revenue").isNull(), 1).otherwise(0)) == 0,
                F.sum(F.when(recent, F.col("revenue")).otherwise(0.0)),
            ).alias("weight"),
        )
        starts = prepared.groupBy(*keys).agg(
            F.min(
                F.when(F.col("target") > 0, F.col("at"))
                if scale_start == "first_positive"
                else F.col("at")
            ).alias("start"),
        )
        active = prepared.join(starts, keys).filter(F.col("at") >= F.col("start"))
        order = Window.partitionBy(*keys).orderBy("at")
        changes = active.withColumn("previous", F.lag("target", scale_lag).over(order))
        scales = changes.groupBy(*keys).agg(
            F.when(
                F.count("target") == F.count(F.lit(1)),
                F.avg(F.pow(F.col("target") - F.col("previous"), 2)),
            ).alias("scale"),
        )
        refs = series.join(weights, keys, "left").join(scales, keys, "left")
        frame = refs.select(
            F.struct(
                *[F.col(key).alias(name) for key, name in zip(keys, series_id, strict=False)]
            ).alias("keys"),
            F.lit(stage).alias("stage"),
            F.lit(origin.isoformat()).alias("origin"),
            F.col("weight").cast("double"),
            F.col("scale").cast("double"),
        )
        if validate:
            issue = check_wrmsse_references(frame, origins=[(stage, origin)]).first()
            if issue is not None:
                raise ValueError(f"invalid weight or scale at {stage}/{origin}: {issue.message}")
        output = frame if output is None else output.unionByName(frame)
    assert output is not None
    return output


def _reference_origins(
    origins: Mapping[str, dt.date] | Sequence[tuple[str, dt.date]],
) -> list[tuple[str, dt.date]]:
    """Normalize the explicit, nonempty set of daily reference boundaries."""
    message = "origins must contain distinct stage/date pairs"
    if not isinstance(origins, Mapping | Sequence) or isinstance(origins, str | bytes):
        raise ValueError(message)
    entries = list(origins.items() if isinstance(origins, Mapping) else origins)
    windows = []
    for entry in entries:
        if not isinstance(entry, tuple | list) or len(entry) != 2:
            raise ValueError(message)
        stage, origin = entry
        if (not isinstance(stage, str) or stage not in {"tuning", "validation", "test"}
                or type(origin) is not dt.date):
            raise ValueError(message)
        windows.append((stage, origin))
    if not windows or len(set(windows)) != len(windows):
        raise ValueError(message)
    return windows


def _invalid_reference_value(value: Column, *, positive: bool) -> Column:
    """Identify missing, non-finite or out-of-range strict reference values."""
    return (
        value.isNull() | F.isnan(value) | (F.abs(value) == float("inf"))
        | ((value <= 0) if positive else (value < 0))
    )


def check_wrmsse_references(
    references: DataFrame,
    *,
    origins: Mapping[str, dt.date] | Sequence[tuple[str, dt.date]],
) -> DataFrame:
    """Report every numeric problem without changing or excluding references.

    Input uses keys (a nonempty struct), stage/origin (strings, origin in ISO date
    form), and numeric weight/scale columns. Origins declares every expected
    daily boundary, including ones with no remaining rows. Malformed schemas,
    null keys and undeclared boundaries raise ValueError.

    Returns keys/stage/origin/field/issue_code/observed_value/message, with the
    original key type and double observed values (including null, NaN and Inf).
    A clean report has zero rows with the same schema. INVALID_WEIGHT requires
    finite nonnegative weights; INVALID_SCALE requires finite positive scales.
    INVALID_TOTAL_WEIGHT requires a finite positive sum over rows where BOTH
    values are usable; its keys are null and an empty usable sum is null.

    This strict numeric report does not check duplicate identities, cohort
    coverage, frozen-study schema or artifact integrity. Keep the existing study
    registration checks; some frozen studies allow undefined scales. Applications
    own exclusions and must apply them consistently across their shared cohort.
    """
    windows = _reference_origins(origins)
    required = {"keys", "stage", "origin", "weight", "scale"}
    if (not isinstance(references, DataFrame)
            or not required.issubset(references.columns)
            or len({name.casefold() for name in references.columns}) != len(references.columns)):
        raise ValueError("references require unambiguous keys, stage, origin, weight and scale")
    key_type = references.schema["keys"].dataType
    if (not isinstance(key_type, StructType) or not key_type.fields
            or len({name.casefold() for name in key_type.names}) != len(key_type.names)
            or any(not isinstance(references.schema[name].dataType, StringType)
                   for name in ("stage", "origin"))
            or any(not isinstance(references.schema[name].dataType, NumericType)
                   for name in ("weight", "scale"))):
        raise ValueError("reference key, stage, origin, weight or scale types are invalid")
    frame = references.select(
        "keys", "stage", "origin", F.col("weight").cast("double").alias("weight"),
        F.col("scale").cast("double").alias("scale"),
    )
    expected = references.sparkSession.createDataFrame(
        [(stage, origin.isoformat()) for stage, origin in windows],
        "stage string, origin string",
    )
    invalid_keys = F.col("keys").isNull()
    for key in key_type.names:
        invalid_keys |= F.col("keys")[key].isNull()
    if frame.filter(invalid_keys).limit(1).count():
        raise ValueError("reference keys cannot be null")
    if frame.join(expected, ["stage", "origin"], "left_anti").limit(1).count():
        raise ValueError("reference stage/origin must match an expected boundary")

    invalid_weight = _invalid_reference_value(F.col("weight"), positive=False)
    invalid_scale = _invalid_reference_value(F.col("scale"), positive=True)
    issues = frame.filter(invalid_weight).select(
        "keys", "stage", "origin", F.lit("weight").alias("field"),
        F.lit("INVALID_WEIGHT").alias("issue_code"), F.col("weight").alias("observed_value"),
        F.lit("Weight must be finite and nonnegative.").alias("message"),
    ).unionByName(frame.filter(invalid_scale).select(
        "keys", "stage", "origin", F.lit("scale").alias("field"),
        F.lit("INVALID_SCALE").alias("issue_code"), F.col("scale").alias("observed_value"),
        F.lit("This strict reference check requires a finite positive scale.").alias("message"),
    ))
    totals = frame.filter(~invalid_weight & ~invalid_scale).groupBy("stage", "origin").agg(
        F.sum("weight").alias("observed_value"),
    )
    totals = expected.join(totals, ["stage", "origin"], "left").filter(
        _invalid_reference_value(F.col("observed_value"), positive=True),
    )
    return issues.unionByName(totals.select(
        F.lit(None).cast(key_type).alias("keys"), "stage", "origin",
        F.lit("total_weight").alias("field"), F.lit("INVALID_TOTAL_WEIGHT").alias("issue_code"),
        "observed_value",
        F.lit("Reference weights require a finite positive total over usable rows.").alias(
            "message",
        ),
    ))
