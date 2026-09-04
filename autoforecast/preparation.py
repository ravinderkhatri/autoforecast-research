"""Prepare regular Spark time series without inventing finer observations."""

from __future__ import annotations

from datetime import UTC, date, datetime

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, NumericType, TimestampNTZType, TimestampType

from autoforecast.frequency import parse_frequency, parse_origin

_PERIODS = {
    "hourly": ("hour", "INTERVAL 1 HOUR"),
    "daily": ("day", "INTERVAL 1 DAY"),
    "weekly": ("week", "INTERVAL 7 DAYS"),
    "monthly": ("month", "INTERVAL 1 MONTH"),
    "yearly": ("year", "INTERVAL 1 YEAR"),
}
_CONVERSIONS = {
    "hourly": {"hourly", "daily", "weekly", "monthly", "yearly"},
    "daily": {"daily", "weekly", "monthly", "yearly"},
    "weekly": {"weekly"},
    "monthly": {"monthly", "yearly"},
    "yearly": {"yearly"},
}


def _column(name: str) -> Column:
    """Read a literal column name, including spaces, dots, or backticks."""
    return F.col("`" + name.replace("`", "``") + "`")


def _validate_preparation_policies(*, fill_missing, duplicate_policy, aggregation="sum"):
    if aggregation != "sum" or duplicate_policy not in {"error", "sum"}:
        raise ValueError("aggregation must be sum and duplicate_policy must be error or sum")
    if fill_missing not in {"zero", "na"}:
        raise ValueError("fill_missing must be 'zero' or 'na'")


def prepare_fit_history(frame: DataFrame, *, panel, origin: str, missing: str) -> DataFrame:
    """Prepare only a fit slice; frozen actuals and past-only values stay unchanged.

    The freezer already owns calendar completion and per-series starts. This
    operation never aggregates rows or invents earlier history. Static labels may
    fill inserted cells only when their observed non-null value is invariant in
    this fit slice. Explicit null targets and absent-calendar targets follow the
    same declared history policy. The original-missing marker remains factual.
    """
    from autoforecast.column_names import fresh_internal_names
    from autoforecast.grid import time_literal

    if missing not in {"preserve", "zero"}:
        raise ValueError("fit missing policy must be preserve or zero")
    cadence = parse_frequency(panel.frequency)
    frame = frame.filter(
        F.col(panel.time_col) <= time_literal(parse_origin(origin, cadence), cadence),
    )
    if missing == "zero":
        frame = frame.withColumn(panel.target_col, F.coalesce(
            F.col(panel.target_col), F.lit(0).cast(frame.schema[panel.target_col].dataType),
        ))
    for feature in (*panel.static, *panel.static_reals):
        value, count = fresh_internal_names(frame.columns, ("_static_value", "_static_count"))
        labels = frame.groupBy(panel.series_col).agg(
            F.first(feature, ignorenulls=True).alias(value),
            F.countDistinct(feature).alias(count),
        )
        if labels.filter(F.col(count) > 1).limit(1).count():
            raise ValueError(f"static feature changes within the fit history: {feature}")
        frame = frame.join(labels.select(panel.series_col, value), panel.series_col, "left")
        frame = frame.withColumn(feature, F.coalesce(F.col(feature), F.col(value))).drop(value)
    return frame


def _cutoff(value: str | date | datetime, source_frequency: str) -> datetime | None:
    """Resolve the automatic end or validate a manually supplied period start.

    Naive datetimes are UTC. Aware datetimes are converted to UTC. Daily and
    coarser input also accepts a date. ISO strings are accepted for convenience.
    """
    if value == "max_series_date":
        return None
    if isinstance(value, str):
        try:
            value = (
                datetime.fromisoformat(value)
                if "T" in value or " " in value
                else date.fromisoformat(value)
            )
        except ValueError as exc:
            raise ValueError(
                "series_end_date must be 'max_series_date' or an ISO date/datetime"
            ) from exc
    if not isinstance(value, date):
        raise ValueError("series_end_date must be 'max_series_date' or a date/datetime")
    if source_frequency == "hourly" and not isinstance(value, datetime):
        raise ValueError("hourly source data requires a datetime for series_end_date")
    stamp = value if isinstance(value, datetime) else datetime.combine(value, datetime.min.time())
    stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    aligned = stamp.minute == stamp.second == stamp.microsecond == 0
    if source_frequency != "hourly":
        aligned = aligned and stamp.hour == 0
    if source_frequency == "weekly":
        aligned = aligned and stamp.weekday() == 0
    if source_frequency in {"monthly", "yearly"}:
        aligned = aligned and stamp.day == 1
    if source_frequency == "yearly":
        aligned = aligned and stamp.month == 1
    if not aligned:
        raise ValueError("series_end_date must label the start of a source period")
    return stamp


def prepare_time_series(
    df: DataFrame,
    *,
    series_id: list[str],
    date_col: str,
    value_cols: list[str],
    source_frequency: str,
    frequency: str = "daily",
    fill_missing: str,
    duplicate_policy: str,
    aggregation: str = "sum",
    observed_col: str | None = None,
    series_end_date: str | date | datetime = "max_series_date",
) -> DataFrame:
    """Aggregate additive quantities and complete their calendars with explicit policies.

    Frequencies accept H/D/W/MS/YS or hourly/daily/weekly/monthly/yearly. Input
    labels must be period starts; timestamps require a UTC Spark session. Finer
    output and weekly-to-month/year splitting are rejected. Only sum aggregation
    is supported: callers must not treat prices or stock levels as additive.

    duplicate_policy is error or sum. fill_missing is na or zero and applies only
    to absent output periods; explicit null observations remain null. Partial
    periods sum observed rows, not an assumed complete feed. observed_col, when
    supplied, adds a boolean source-presence marker so filled periods stay visible.

    Series start at their first retained observation. A common cutoff is inclusive;
    otherwise each series ends at its own last observation. Output uses original
    column names and Spark numeric widening. No tables or settings are changed.
    """
    aliases = {"hourly": "H", "daily": "D", "weekly": "W", "monthly": "MS", "yearly": "YS"}
    names = {value: key for key, value in aliases.items()}
    source_code = parse_frequency(aliases.get(source_frequency, source_frequency)).code
    output_code = parse_frequency(aliases.get(frequency, frequency)).code
    if source_code not in names or output_code not in names:
        raise ValueError("preparation supports period-start H, D, W, MS and YS")
    source_frequency, frequency = names[source_code], names[output_code]
    _validate_preparation_policies(fill_missing=fill_missing,
                                   duplicate_policy=duplicate_policy, aggregation=aggregation)
    if observed_col is not None and (
        not isinstance(observed_col, str)
        or not observed_col
        or observed_col.casefold() in {name.casefold() for name in df.columns}
    ):
        raise ValueError("observed_col must be a new nonblank column name")
    if source_frequency not in _PERIODS or frequency not in _PERIODS:
        raise ValueError(f"frequencies must be one of {list(_PERIODS)}")
    if frequency not in _CONVERSIONS[source_frequency]:
        raise ValueError(
            f"cannot convert {source_frequency} to {frequency} without finer source data"
        )
    if (
        not isinstance(series_id, list)
        or not series_id
        or not isinstance(value_cols, list)
        or not value_cols
    ):
        raise ValueError("series_id and value_cols must be nonempty lists")
    required = [*series_id, date_col, *value_cols]
    if any(not isinstance(name, str) or not name for name in required):
        raise ValueError("column names must be nonempty strings")
    if len({name.casefold() for name in required}) != len(required):
        raise ValueError("series, date, and value columns must be distinct")
    if len({name.casefold() for name in df.columns}) != len(df.columns):
        raise ValueError("input has duplicate or ambiguous column names")
    if missing := set(required) - set(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    date_type = df.schema[date_col].dataType
    if not isinstance(date_type, DateType | TimestampType | TimestampNTZType):
        raise ValueError("date_col must be a Spark date or timestamp")
    if source_frequency == "hourly" and isinstance(date_type, DateType):
        raise ValueError("hourly source data requires a timestamp column")
    if (
        not isinstance(date_type, DateType)
        and df.sparkSession.conf.get("spark.sql.session.timeZone") != "UTC"
    ):
        raise ValueError("timestamp input requires spark.sql.session.timeZone=UTC")
    if any(not isinstance(df.schema[name].dataType, NumericType) for name in value_cols):
        raise ValueError("value_cols must contain numeric columns")
    cutoff = _cutoff(series_end_date, source_frequency)

    keys = [f"_key_{i}" for i in range(len(series_id))]
    values = [f"_value_{i}" for i in range(len(value_cols))]
    source = df.select(
        *[_column(name).alias(key) for name, key in zip(series_id, keys, strict=False)],
        _column(date_col).alias("_date"),
        *[_column(name).alias(value) for name, value in zip(value_cols, values, strict=False)],
    )
    invalid = F.col("_date").isNull() | (
        F.col("_date") != F.date_trunc(_PERIODS[source_frequency][0], "_date")
    )
    for key in keys:
        invalid |= F.col(key).isNull()
    for value in values:
        invalid |= F.isnan(value) | (F.abs(F.col(value)) == float("inf"))
    if source.filter(invalid).limit(1).count():
        raise ValueError(
            "input has null IDs/dates, non-finite values, or dates not aligned to source periods"
        )
    cutoff_literal = (
        F.lit(cutoff.date() if isinstance(date_type, DateType) else cutoff)
        if cutoff is not None
        else None
    )
    if cutoff_literal is not None:
        source = source.filter(F.col("_date") <= cutoff_literal)
    if not source.limit(1).count():
        raise ValueError("no input rows remain at the requested cutoff")

    if (
        duplicate_policy == "error"
        and source.groupBy(*keys, "_date").count().filter(F.col("count") > 1).limit(1).count()
    ):
        raise ValueError("duplicate source periods require duplicate_policy=sum")

    unit, step = _PERIODS[frequency]
    output_date_type = "timestamp" if frequency == "hourly" else "date"
    source = source.withColumn("_date", F.date_trunc(unit, "_date").cast(output_date_type))
    totals = source.groupBy(*keys, "_date").agg(
        *[
            F.when(F.count(value) == F.count(F.lit(1)), F.sum(value)).alias(value)
            for value in values
        ]
    )
    bounds = totals.groupBy(*keys).agg(F.min("_date").alias("_start"), F.max("_date").alias("_end"))
    if cutoff_literal is not None:
        bounds = bounds.withColumn(
            "_end", F.date_trunc(unit, cutoff_literal).cast(output_date_type)
        )
    calendar = bounds.select(
        *keys, F.explode(F.sequence("_start", "_end", F.expr(step))).alias("_date")
    )
    result = calendar.join(totals.withColumn("_observed", F.lit(True)), [*keys, "_date"], "left")
    if fill_missing == "zero":
        for value in values:
            result = result.withColumn(
                value,
                F.when(
                    F.col("_observed").isNull(),
                    F.lit(0).cast(totals.schema[value].dataType),
                ).otherwise(F.col(value)),
            )
    return result.select(
        *[F.col(key).alias(name) for name, key in zip(series_id, keys, strict=False)],
        F.col("_date").alias(date_col),
        *[F.col(value).alias(name) for name, value in zip(value_cols, values, strict=False)],
        *(
            [F.coalesce(F.col("_observed"), F.lit(False)).alias(observed_col)]
            if observed_col is not None
            else []
        ),
    )
