"""The eligible scoring grid: the exact set of cells a forecast is made and judged on.

This lives in the model-neutral harness because it governs every competitor and every metric.
The custom model consumes the resulting Parquet path through its adapter; the harness must not
reach through that adapter into ``forecast/`` merely to learn which cells constitute evidence.

WHY THIS IS A MATERIALISED ARTEFACT AND NOT A FILTER APPLIED TWICE. The grid the model trains on
and the grid the metric scores on must be the SAME set of cells. If each derives its own from the
panel, a change to either derivation moves one and not the other, and the resulting comparison is
between a model and a metric that disagree about what exists. Materialising it once and passing
its PATH is what makes that impossible rather than merely unlikely.

IT IS ALSO THE MANIFEST'S GRID. `EvaluationManifest` records `eligible_rule` -- the sentence
describing how this was derived -- and a result citing that manifest is claiming it was scored on
this grid. Rebuilding the grid from a changed rule changes the manifest hash, which invalidates
every result that cited the old one. That is the intended behaviour, not an inconvenience.

THE FULL CROSS-PRODUCT, RESTRICTED. Not an inner join to whatever the fact table happens to
contain. A cell with a forecast and no actual row is precisely the over-forecast-on-a-zero-day
that the deployed metric could not see, and it has to be IN the grid for the metric to catch it.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Iterable

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast.frequency import (
    Cadence,
    FrequencyError,
    TimePoint,
    canonical_origin,
    parse_frequency,
    period_points,
)

__all__ = [
    "GridError",
    "agrees_with",
    "build",
    "canonical_key",
    "content_hash",
    "period_membership",
    "summarise",
    "time_literal",
]


class GridError(ValueError):
    """A grid was requested that would not describe what is actually being scored."""


def build(
    panel_path: str | pathlib.Path,
    origin: str,
    horizon: int,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    series_col: str,
    time_col: str,
    eligible_col: str | None = None,
    frequency: str = "D",
) -> pathlib.Path:
    """Write `series_id · date` for every cell to be forecast and scored. Returns the path.

    Args:
        panel_path: Parquet source, used for the series list and the eligibility flag.
        origin: Canonical forecast origin. The grid covers the next ``horizon`` cadence points.
        horizon: Periods.
        spark: Active session.
        out_dir: Parquet destination.
        series_col: Series identifier.
        time_col: Cadence-dependent time-key column.
        eligible_col: Eligibility flag from `exposure.apply`. When None, every series-period in the
            window enters the grid -- correct only for a dataset with no exposure information at
            all, and the caller is asserting that by omitting it.

    Returns:
        The written path.

    Raises:
        GridError: a non-positive horizon, or an empty grid -- which means nothing is eligible in
            the window and a forecast would be scored on no cells at all.
    """
    if type(horizon) is not int or horizon < 1:
        raise GridError(f"horizon must be >= 1, got {horizon}")
    try:
        cadence = parse_frequency(frequency)
        period_dates = period_points(origin, cadence, horizon)
    except (TypeError, ValueError, OverflowError, FrequencyError) as exc:
        raise GridError(
            "origin/frequency does not define a supported canonical period grid") from exc
    key_type = "timestamp_ntz" if cadence.key_kind == "timestamp" else "date"
    dates = spark.createDataFrame([(day,) for day in period_dates], f"date {key_type}")

    df = spark.read.parquet(str(panel_path))
    if eligible_col is None:
        series = df.select(F.col(series_col).alias("series_id")).distinct()
        grid = series.crossJoin(F.broadcast(dates))
    else:
        if cadence.key_kind == "timestamp" and (
            df.schema[time_col].dataType.typeName() != "timestamp_ntz"
        ):
            raise GridError(
                f"panel time column {time_col!r} must have timestamp_ntz physical type for "
                f"hourly cadence, got {df.schema[time_col].dataType.simpleString()}"
            )
        # Eligibility is EFFECTIVE-DATED, so it is taken from the window itself rather than from
        # a series-level summary. A series eligible in March and delisted in April must not be
        # eligible in April's grid, and a `distinct()` over series would say it was.
        key, valid = canonical_key(time_col, cadence)
        if df.filter(~valid).limit(1).count():
            raise GridError(
                f"panel time column {time_col!r} contains a null, malformed, lossy, or "
                f"off-anchor {cadence.code} key"
            )
        window = df.withColumn("__af_grid_time", key).join(
            F.broadcast(dates),
            F.col("__af_grid_time") == dates["date"],
            "left_semi",
        )
        grid = (window.filter(F.col(eligible_col))
                .select(F.col(series_col).alias("series_id"),
                        F.col("__af_grid_time").alias("date"))
                .distinct())
    n = grid.count()
    if n == 0:
        raise GridError(
            f"no eligible cell in the {horizon}-period grid after {origin}. "
            "A forecast scored on an empty grid is "
            "not a forecast -- check the eligibility rule before the model")
    grid.write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir))


def summarise(grid_path: str | pathlib.Path, spark: SparkSession) -> DataFrame:
    """Summarize the grid size, series count, time span, and density."""
    g = spark.read.parquet(str(grid_path))
    return g.agg(
        F.count(F.lit(1)).alias("n_cells"),
        F.countDistinct("series_id").alias("n_series"),
        F.countDistinct("date").alias("n_days"),
    ).withColumn("density",
                 F.col("n_cells") / (F.col("n_series") * F.col("n_days")))


def content_hash(
    grid_path: str | pathlib.Path,
    spark: SparkSession,
    *,
    frequency: str | None = None,
) -> str:
    """Stable SHA-256 of the grid's actual key content, independent of Parquet layout.

    Spark gives output part files generated names, so hashing the directory bytes makes an
    identical rebuild look like a different grid.  This hashes a globally sorted, unambiguous
    stream of the canonical ``series_id · date`` keys instead. ``toLocalIterator`` streams the
    result; it does not collect an estate-sized grid into driver memory.

    Raises:
        GridError: required columns are absent, keys are invalid, or keys are duplicated.
    """
    raw = spark.read.parquet(str(grid_path))
    missing = sorted({"series_id", "date"} - set(raw.columns))
    if missing:
        raise GridError(f"grid lacks required columns {missing}")
    if frequency is None:
        physical_type = raw.schema["date"].dataType.typeName()
        frequency = "H" if physical_type in ("timestamp", "timestamp_ntz") else "D"
    try:
        cadence = parse_frequency(frequency)
    except FrequencyError as exc:
        raise GridError(str(exc)) from exc
    if cadence.key_kind == "timestamp" and (
        raw.schema["date"].dataType.typeName() != "timestamp_ntz"
    ):
        raise GridError(
            "hourly grid date must have timestamp_ntz physical type, got "
            f"{raw.schema['date'].dataType.simpleString()}"
        )
    date, valid_date = canonical_key("date", cadence)
    keys = raw.select(
        F.col("series_id").cast("string").alias("series_id"), date.alias("date"),
        valid_date.alias("_valid_date"),
    )
    if keys.filter(
            F.col("series_id").isNull()
            | (F.trim("series_id") == F.lit(""))
            | ~F.col("_valid_date")).limit(1).count():
        raise GridError("grid contains a null/blank series_id or invalid time key")
    keys = keys.drop("_valid_date")
    if (keys.groupBy("series_id", "date").count()
            .filter(F.col("count") != 1).limit(1).count()):
        raise GridError("grid contains duplicate (series_id, date) keys")

    digest = hashlib.sha256(b"autoforecast-eligible-grid-v1\0")
    for row in keys.orderBy("series_id", "date").toLocalIterator(prefetchPartitions=True):
        series = row["series_id"].encode("utf-8")
        day = row["date"].isoformat().encode("ascii")
        digest.update(len(series).to_bytes(8, "big"))
        digest.update(series)
        digest.update(day)  # ISO dates are fixed-width, so no second length prefix is needed.
    return digest.hexdigest()


def canonical_key(column: str, cadence: Cadence) -> tuple[Column, Column]:
    """Return a lossless canonical Spark key and the predicate proving it is valid."""
    escaped = column.replace("`", "``")
    timestamp = F.expr(f"try_cast(`{escaped}` as timestamp_ntz)")
    raw = F.col(column)
    source_type = F.expr(f"typeof(`{escaped}`)")
    text = raw.cast("string")
    if cadence.key_kind == "timestamp":
        key = timestamp
        representation = (source_type == F.lit("timestamp_ntz")) | (
            (source_type == F.lit("string"))
            & text.rlike(
                r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:00:00\z"
            )
        )
        valid = (
            raw.isNotNull()
            & timestamp.isNotNull()
            & (F.date_trunc("hour", timestamp) == timestamp)
            & representation
        )
        return key, valid

    key = F.expr(f"try_cast(`{escaped}` as date)")
    representation = source_type.isin("date", "timestamp_ntz") | (
        (source_type == F.lit("string"))
        & text.rlike(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}\z")
    )
    valid = (
        raw.isNotNull()
        & timestamp.isNotNull()
        & key.isNotNull()
        & (F.date_trunc("day", timestamp) == timestamp)
        & representation
    )
    if cadence.unit == "M":
        expected_day = F.lit(1) if cadence.anchor == "start" else F.dayofmonth(F.last_day(key))
        valid = valid & (F.dayofmonth(key) == expected_day)
    elif cadence.unit == "Y":
        expected_month, expected_day = (1, 1) if cadence.anchor == "start" else (12, 31)
        valid = valid & (F.month(key) == expected_month) & (F.dayofmonth(key) == expected_day)
    return key, valid


def time_literal(point: str | TimePoint, cadence: Cadence) -> Column:
    """Build a Spark literal without routing a naive H point through session-zone timestamp."""
    spark_type = "timestamp_ntz" if cadence.key_kind == "timestamp" else "date"
    return F.lit(canonical_origin(point, cadence)).cast(spark_type)


def period_membership(
    column: str | Column,
    points: Iterable[str | TimePoint],
    cadence: Cadence,
) -> Column:
    """Test membership using cadence-typed literals, including timezone-stable H values."""
    key = F.col(column) if isinstance(column, str) else column
    literals = tuple(time_literal(point, cadence) for point in points)
    return key.isin(*literals) if literals else F.lit(False)


def agrees_with(grid_path: str | pathlib.Path, other_path: str | pathlib.Path,
                spark: SparkSession) -> tuple[int, int]:
    """Cells in one grid and not the other, both directions. `(0, 0)` means they are identical.

    The check that makes "the model trained and the metric scored on the same cells" verifiable
    rather than assumed. Two derivations of the same rule drift the moment one is edited, and the
    resulting comparison is between a model and a metric that disagree about what exists.
    """
    a = spark.read.parquet(str(grid_path)).select("series_id", "date")
    b = spark.read.parquet(str(other_path)).select("series_id", "date")
    return a.subtract(b).count(), b.subtract(a).count()
