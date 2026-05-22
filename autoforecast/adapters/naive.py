"""Seasonal naive, as a Spark window over the panel. No forecasting library is involved.

`y_hat(origin + h) = y(origin + h - m * ceil(h / m))` -- repeat the last complete season. For a
daily panel with `m = 7` and a 28-day horizon that is "repeat the last complete week".
"""

from __future__ import annotations

import pathlib

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters import _validate_span
from autoforecast.frequency import (
    advance,
    canonical_origin,
    parse_frequency,
    parse_origin,
    period_points,
)
from autoforecast.grid import canonical_key, period_membership, time_literal
from autoforecast.panel import Forecast, Panel

REUSE_EXEMPTION = (
    "seasonal naive is LAG(y, m) OVER (PARTITION BY series ORDER BY date) -- a definition, not "
    "an algorithm. There is no numerical method to get wrong, statsforecast's own implementation "
    "is 52 lines of array indexing, and adapting it would buy a Python round-trip per series "
    "against large panels."
)

NAME = "seasonal_naive"

EMITTED = "emitted"
EMITTED_CENSORED = "emitted_from_lower_bound"
DECLINED_NO_SOURCE = "declined_no_source"
DECLINED_UNOBSERVED = "declined_source_unobserved"


def seasonal_naive(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
) -> Forecast:
    """Forecast `horizon` periods after `origin` by repeating the last complete season.

    Every requested cell is accounted for. Three cases are NOT forecast-and-hope, because each
    would otherwise contribute a silent zero that reads as a confident prediction of no demand:

    | source cell | disposition | why |
    |---|---|---|
    | present, `exact` | `emitted` | the ordinary path |
    | present, `lower_bound` | `emitted_from_lower_bound` | a stockout day sold before it ran out, so the value is real but biased LOW. Emitted, and counted separately so the bias is visible in the results table rather than buried in the score |
    | present, `unobserved` | `declined_source_unobserved` | never ranged or no data. Copying it forward would manufacture a demand observation out of an assortment decision |
    | absent | `declined_no_source` | the lag reaches before the series started, or into a gap. This is the edge case a library would get wrong for us anyway, since it does not know the exposure contract |

    Args:
        panel: Validated panel. When `panel.has_exposure` is False every source counts as
            `exact` -- which is correct for public benchmarks that carry no exposure column, and
            is why this adapter still registers as consuming exposure.
        origin: Last date the adapter may see, ISO `YYYY-MM-DD`. Exclusive lower bound of the
            forecast window.
        horizon: Periods to forecast. May exceed `season_length`; the source date walks back a
            whole number of seasons, so cycle 2 reads the same week as cycle 1.
        spark: Active session. Passed rather than created -- a library that starts its own
            context cannot be given the caller's cluster config.
        out_dir: Where the forecast Parquet is written. Overwritten.

    Returns:
        A validated `Forecast` pointing at `series_id · date · yhat · source_state`.

    Raises:
        ValueError: `horizon` is not positive.
        ForecastError: the dispositions do not account for every requested cell.
    """
    _validate_span(panel, origin, horizon, adapter=NAME)
    panel.validate()
    m = panel.season_length

    cadence = parse_frequency(panel.frequency)
    time_key, valid_time_key = canonical_key(panel.time_col, cadence)
    raw_source = spark.read.parquet(panel.path)
    if cadence.key_kind == "timestamp" and (
        raw_source.schema[panel.time_col].dataType.typeName() != "timestamp_ntz"
    ):
        raise ValueError(
            f"hourly panel time column {panel.time_col!r} must have timestamp_ntz physical type; "
            "timezone-sensitive timestamp storage is lossy for wall-clock keys"
        )
    raw = raw_source.withColumn(
        "__af_naive_time", time_key
    ).withColumn("__af_naive_valid_time", valid_time_key)
    if raw.filter(~F.col("__af_naive_valid_time")).limit(1).count():
        raise ValueError(
            f"panel contains a malformed, lossy, or off-anchor {cadence.code} time key"
        )
    source = _source_window(raw, panel, origin, m)
    grid = _future_grid(spark, raw, panel, origin, horizon, m)

    joined = grid.join(
        source,
        (grid["series_id"] == source["_sid"]) & (grid["source_date"] == source["_hist_date"]),
        "left",
    )
    state = F.col("_state") if panel.has_exposure else F.lit("exact")
    bucket = (
        F.when(F.col("_y").isNull(), F.lit(DECLINED_NO_SOURCE))
        .when(state == F.lit("unobserved"), F.lit(DECLINED_UNOBSERVED))
        .when(state == F.lit("lower_bound"), F.lit(EMITTED_CENSORED))
        .otherwise(F.lit(EMITTED))
    )
    labelled = joined.withColumn("_bucket", bucket).cache()
    try:
        disposition = {r["_bucket"]: r["count"] for r in
                       labelled.groupBy("_bucket").count().collect()}
        n_series = grid.select("series_id").distinct().count()

        (labelled
         .filter(F.col("_bucket").startswith("emitted"))
         .select(
             F.col("series_id"),
             F.col("ds").alias("date"),
             F.col("_y").alias("yhat"),
             F.col("_bucket").alias("source_state"))
         .write.mode("overwrite").parquet(str(out_dir)))

        return Forecast(
            adapter=NAME,
            path=str(out_dir),
            origin=origin,
            horizon=horizon,
            n_series=n_series,
            disposition=disposition,
            target_estimand=panel.target_estimand,
            frequency=panel.frequency,
        ).validate()
    finally:
        labelled.unpersist()


def _source_window(raw: DataFrame, panel: Panel, origin: str, m: int) -> DataFrame:
    """The only history a seasonal naive can read: the `m` periods ending at `origin`.

    Writing `h = q*m + r` with `0 <= r < m`, the source date is `origin + r - m` when `r > 0` and
    `origin` when `r = 0`. So it lies in `[origin - m + 1, origin]` for EVERY horizon -- a 28-day
    horizon and a 365-day one read the same seven days.

    This is what makes the adapter scale rather than merely run. On M5 it takes the join's right
    side from 59,181,090 rows to 30,490 series x 7 days = 213,430; without it a local run OOMs the
    JVM shuffling the full history to match 853,720 grid rows. On an estate the ratio is worse.

    The upper filter is redundant given the arithmetic above and is kept deliberately: that
    argument spans two functions, and a leak resting on an off-by-one staying correct is the kind
    this project keeps finding.
    """
    cadence = parse_frequency(panel.frequency)
    origin_point = parse_origin(origin, cadence)
    source_points = tuple(
        advance(origin_point, cadence, offset) for offset in range(-(m - 1), 1)
    )
    history_key = (
        F.col("__af_naive_time")
        if "__af_naive_time" in raw.columns
        else canonical_key(panel.time_col, cadence)[0]
    )
    cols = [
        F.col(panel.series_col).alias("_sid"),
        history_key.alias("_hist_date"),
        F.col(panel.target_col).alias("_y"),
    ]
    if panel.has_exposure:
        cols.append(F.col(panel.exposure_col).alias("_state"))
    selected = raw.select(*cols)
    return selected.filter(period_membership("_hist_date", source_points, cadence))


def _future_grid(
    spark: SparkSession, raw: DataFrame, panel: Panel, origin: str, horizon: int, m: int
) -> DataFrame:
    """One row per (series, future date), carrying the source date its value is copied from.

    `horizon` is tens of periods, so the step frame is built on the driver; the series side is
    read from the panel and stays distributed. Crossing them is the only wide step, and it is a
    broadcast join on a frame of `horizon` rows.

    Series come from the WHOLE history at or before `origin`, not from `_source_window`. A series
    that existed but sold nothing in the last `m` days is still a series we were asked to forecast,
    and taking the list from the narrow window would drop it from the grid entirely -- silently
    shrinking `n_series` instead of declining its cells with a reason.
    """
    cadence = parse_frequency(panel.frequency)
    origin_point = parse_origin(origin, cadence)
    targets = period_points(origin_point, cadence, horizon)
    rows = [
        (
            step,
            canonical_origin(target, cadence),
            canonical_origin(
                advance(target, cadence, -m * ((step + m - 1) // m)), cadence
            ),
        )
        for step, target in enumerate(targets, 1)
    ]
    key_type = "timestamp_ntz" if cadence.unit == "H" else "date"
    steps = spark.createDataFrame(
        rows, "step int, ds string, source_date string"
    ).select(
        "step",
        F.col("ds").cast(key_type).alias("ds"),
        F.col("source_date").cast(key_type).alias("source_date"),
    )
    history_key = (
        F.col("__af_naive_time")
        if "__af_naive_time" in raw.columns
        else canonical_key(panel.time_col, cadence)[0]
    )
    series = (raw.filter(history_key <= time_literal(origin_point, cadence))
              .select(F.col(panel.series_col).alias("series_id")).distinct())
    return series.crossJoin(F.broadcast(steps))


__all__ = ["NAME", "REUSE_EXEMPTION", "seasonal_naive"]
