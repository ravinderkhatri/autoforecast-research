"""Build caller-declared daily WRMSSE references without customer-specific rules."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

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
    validate=False returns them for an existing admission report, without dropping
    or admitting series. Caller-owned exclusions require a separate decision.
    """
    for name, value in (("weight_window_days", weight_window_days), ("scale_lag", scale_lag)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if scale_start not in {"first_positive", "history_start"}:
        raise ValueError("scale_start must be first_positive or history_start")
    if type(validate) is not bool:
        raise ValueError("validate must be boolean")
    windows = list(origins.items() if isinstance(origins, Mapping) else origins)
    if (
        not windows
        or any(
            stage not in {"tuning", "validation", "test"} or type(origin) is not dt.date
            for stage, origin in windows
        )
        or len(set(windows)) != len(windows)
    ):
        raise ValueError("origins must contain distinct stage/date pairs")
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
        if validate:
            invalid = F.lit(False)
            for name in ("weight", "scale"):
                value = F.col(name)
                invalid |= value.isNull() | F.isnan(value) | (F.abs(value) == float("inf"))
                invalid |= value < 0 if name == "weight" else value <= 0
            if refs.filter(invalid).limit(1).count():
                raise ValueError(f"invalid weight or scale at {stage}/{origin}")
            total_row = refs.agg(F.sum("weight")).first()
            total = total_row[0] if total_row is not None else None
            if total is None or not 0 < total < float("inf"):
                raise ValueError("reference weights require a finite positive total")
        frame = refs.select(
            F.struct(
                *[F.col(key).alias(name) for key, name in zip(keys, series_id, strict=False)]
            ).alias("keys"),
            F.lit(stage).alias("stage"),
            F.lit(origin.isoformat()).alias("origin"),
            F.col("weight").cast("double"),
            F.col("scale").cast("double"),
        )
        output = frame if output is None else output.unionByName(frame)
    assert output is not None
    return output
