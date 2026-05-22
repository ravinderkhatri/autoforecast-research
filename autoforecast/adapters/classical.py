"""Fitted classical baselines through statsforecast, distributed by Fugue onto Spark.

NAMED `classical`, NOT `statsforecast`: a module named after the package it wraps shadows that
package for anything run from this directory, and `test_naive.py` already runs from here.

Seasonal naive needed no library because it is a definition. These are different: Croston models
inter-arrival times and demand sizes separately, ETS fits level/trend/season by maximum likelihood.
Maintained implementations keep these comparisons independent of custom model code.

Dispatch is automatic in statsforecast 2.x: passing a SPARK DataFrame (rather than pandas) routes
through Fugue onto the cluster. There is no `backend` argument any more.

REQUIRES `PYSPARK_PYTHON` to point at the interpreter that has `fugue` installed. Unlike the naive
adapter -- native Catalyst, no Python in the loop -- these run a Python UDF per series, so the
EXECUTOR's interpreter must have the dependencies. Without it the failure is a
`ModuleNotFoundError: No module named 'fugue'` raised inside a worker, which reads as a Spark
problem rather than an environment one.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta, CrostonClassic

from autoforecast import resources
from autoforecast.adapters import _validate_daily_span
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel


@dataclass(frozen=True)
class ModelSpec:
    """One classical model and the history it needs before it can be fitted at all.

    `min_history` is not tuning. `statsforecast` REFUSES a series it considers too short --
    `AutoETS` raises `NotImplementedError("tiny datasets")` when `n <= npars + 4`, which with a
    damped seasonal form is n <= 11 -- and the refusal surfaces inside a Spark worker and kills
    the job. Declaring the floor turns a crashed run into declined cells with a reason.

    Attributes:
        build: `season_length -> model`. Takes the season length from the PANEL; hardcoding one
            here would be an estate assumption in generic code.
        min_history: `season_length -> minimum eligible observations`.
    """

    build: Callable[[int], object]
    min_history: Callable[[int], int]


MODELS = {
    # Croston is the standard intermittent-demand method and the right classical comparator for
    # this data: M5 is 59.5% zeros on eligible cells, and a method that models inter-arrival time
    # separately from demand size is the one designed for that. It needs at least one non-zero
    # demand to have anything to average, and two points to estimate an interval between them.
    "croston": ModelSpec(build=lambda m: CrostonClassic(), min_history=lambda m: 2),
    # AutoETS selects among error/trend/season forms by AICc. On intermittent series it will often
    # lose to Croston; it is here because losing to it is INFORMATION -- it bounds how much a
    # smooth exponential model can extract from a series that is mostly zero.
    #
    # Floor: statsforecast's own bound is n > npars + 4 (npars <= 7 for a damped seasonal form),
    # and separately n > season_length. Two full cycles is the smallest history on which a
    # seasonal term means anything, so max(2m, 12) clears both.
    "auto_ets": ModelSpec(build=lambda m: AutoETS(season_length=m),
                          min_history=lambda m: max(2 * m, 12)),
    "auto_theta": ModelSpec(build=lambda m: AutoTheta(season_length=m),
                             min_history=lambda m: max(2 * m, 12)),
}

EMITTED = "emitted"
DECLINED_NO_HISTORY = "declined_no_eligible_history"
DECLINED_SHORT_HISTORY = "declined_history_below_model_minimum"

SERIES_PER_PARTITION = resources.spark_profile().series_per_partition
"""How many series one Python worker fits at a time.

Fugue partitions by `unique_id` and inherits the INPUT frame's partitioning, so without this the
adapter's memory per worker is decided by how the panel happens to be laid out on disk. On M5
partitioned by store, that is ~3,000 series and ~5.9M rows in one pandas frame per worker; running
all ten stores crashed the Python workers outright ("Python worker exited unexpectedly",
SocketException: Connection reset -- the OOM signature, since the worker dies before it can raise).

DERIVED FROM THE MACHINE by `resources.spark_profile()`, from the memory left to the Python
workers after the heap, divided by the concurrent workers. A constant here is a ceiling everywhere:
100 was chosen on a 16 GB laptop and, shipped to a 34 GB box, left it running at the laptop's
limits.
The cost is more tasks -- normally the right trade, because a task is cheap and a dead worker
takes the whole job.

IT IS AN ENVIRONMENT VARIABLE because "a task is cheap" stops being true when Spark is also
configured NOT to reuse Python workers: each task then pays a fresh interpreter plus a numpy and
statsforecast import. Measured on a 34 GB box at 100 series with reuse off, AutoETS took about an
hour per origin on 30,490 series while the JVM held 2.3 GB of the 34 available -- the memory
settings were doing nothing except multiplying startup cost. A machine with headroom should say
so rather than inherit a smaller machine's limit.

MEASURED, after 250 was still too many. A standalone AutoETS run over all ten stores succeeded at
250 on an idle 16 GB machine (1,690s, 30,490 series). The same setting crashed inside the full
Phase 2 run, because by the time AutoETS starts the JVM heap has grown through the naive and
Croston stages and there is less left for the Python workers. A per-partition size that only works
when nothing else has run is not a setting, it is a coincidence.
"""
DECLINED_NO_FORECAST = "declined_no_forecast_at_that_date"


def forecast_with(
    model: str,
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
) -> Forecast:
    """Fit `model` per series on eligible history up to `origin` and forecast `horizon` periods.

    Eligibility is honoured by FILTERING: only cells in `ELIGIBLE_STATES` are fitted on. That is
    what `consumes_exposure=True` means for this adapter, and it is the right treatment for the
    dominant case, which is a leading run of never-ranged cells before a product was listed.

    A series with fewer than `MODELS[model].min_history(season_length)` eligible observations is
    NOT fitted and its cells are declined as `declined_history_below_model_minimum`. That is not a
    tuning choice: statsforecast refuses such a series itself, raising inside a Spark worker and
    killing the whole job -- which is how this was found, on the 2016-01-03 M5 origin.

    LIMIT, stated because it bounds every result from this adapter: removing an INTERIOR ineligible
    stretch leaves the surrounding days adjacent, so a model with a seasonal term sees a calendar
    that skips. Croston is insensitive to this -- it works on the sequence of non-zero demands --
    and AutoETS is not. Neither is given a way to express "this cell did not exist", because
    statsforecast has none.

    Args:
        model: A key of `MODELS`.
        panel: Data. `panel.season_length` is passed to seasonal models.
        origin: Last date the model may see, ISO. Exclusive lower bound of the forecast window.
        horizon: Periods to forecast.
        spark: Active session.
        out_dir: Where the forecast Parquet is written. Overwritten.

    Returns:
        A validated `Forecast`.

    Raises:
        ValueError: unknown model, non-positive horizon, `PYSPARK_PYTHON` unset, or no series
            meeting the model's minimum history.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; available: {sorted(MODELS)}")
    _validate_daily_span(panel, origin, horizon, adapter=model)
    if not os.environ.get("PYSPARK_PYTHON"):
        raise ValueError(
            "PYSPARK_PYTHON is not set. These models run a Python UDF per series, so the Spark "
            "EXECUTOR needs the interpreter that has fugue and statsforecast installed. Without "
            "it the failure surfaces as ModuleNotFoundError inside a worker and reads as a Spark "
            "fault rather than an environment one")

    spec = MODELS[model]
    floor = spec.min_history(panel.season_length)
    train = _long_enough(_eligible_history(panel, spark, origin), floor)
    n_fitted = train.select("unique_id").distinct().count()
    if n_fitted == 0:
        raise ValueError(
            f"no series has {floor} eligible observations at or before {origin}, which is "
            f"{model}'s minimum. Nothing can be fitted")

    # Repartition by series so each Python worker fits a bounded number of them. See
    # SERIES_PER_PARTITION -- inheriting the panel's layout is what killed the first full run.
    # Enough tasks to keep every core fed several times over, then narrowed by the per-partition
    # ceiling. The adapter is the only place that knows n_fitted, so the parallelism half of the
    # rule can only be applied here.
    by_parallelism = spark.sparkContext.defaultParallelism * resources.TASKS_PER_CORE
    by_size = -(-n_fitted // SERIES_PER_PARTITION)
    parts = max(spark.sparkContext.defaultParallelism, by_parallelism, by_size)
    raw = StatsForecast(models=[spec.build(panel.season_length)], freq="D").forecast(
        h=horizon, df=train.repartition(parts, "unique_id"))
    grid = _expected_grid(panel, spark, origin, horizon, floor)
    joined = grid.join(
        raw.select(F.col("unique_id").alias("_uid"),
                   F.to_date(F.col("ds")).alias("_d"),
                   F.col(raw.columns[-1]).cast("double").alias("_yhat")),
        (grid["series_id"] == F.col("_uid")) & (grid["date"] == F.col("_d")), "left")

    bucket = (F.when(F.col("_history_days") == 0, F.lit(DECLINED_NO_HISTORY))
              .when(F.col("_history_days") < floor, F.lit(DECLINED_SHORT_HISTORY))
              .when(F.col("_yhat").isNull(), F.lit(DECLINED_NO_FORECAST))
              .otherwise(F.lit(EMITTED)))
    labelled = joined.withColumn("_bucket", bucket).cache()
    try:
        disposition = {r["_bucket"]: r["count"] for r in
                       labelled.groupBy("_bucket").count().collect()}
        total_series = grid.select("series_id").distinct().count()

        # Negative predictions are not demand. AutoETS produces them on near-zero series;
        # clipping is a property of the TARGET (units sold cannot be negative), not a fudge of
        # the model.
        (labelled.filter(F.col("_bucket") == EMITTED)
         .select("series_id", "date",
                 F.greatest(F.col("_yhat"), F.lit(0.0)).alias("yhat"))
         .write.mode("overwrite").parquet(str(out_dir)))

        return Forecast(adapter=model, path=str(out_dir), origin=origin, horizon=horizon,
                        n_series=total_series, disposition=disposition,
                        target_estimand=panel.target_estimand,
                        frequency=panel.frequency).validate()
    finally:
        labelled.unpersist()


def _eligible_history(panel: Panel, spark: SparkSession, origin: str) -> DataFrame:
    """Eligible cells at or before `origin`, renamed to statsforecast's column contract."""
    df = spark.read.parquet(panel.path).filter(
        F.col(panel.time_col) <= F.to_date(F.lit(origin)))
    if panel.has_exposure:
        df = df.filter(F.col(panel.exposure_col).isin(*sorted(ELIGIBLE_STATES)))
    return df.select(F.col(panel.series_col).alias("unique_id"),
                     F.col(panel.time_col).cast("timestamp").alias("ds"),
                     F.col(panel.target_col).cast("double").alias("y"))


def _long_enough(history: DataFrame, floor: int) -> DataFrame:
    """Keep only series with at least `floor` eligible observations.

    Filtering rather than relying on statsforecast's `fallback_model`: a fallback silently
    substitutes a DIFFERENT model for some series, so a column labelled `auto_ets` would hold
    two models' output and nothing downstream could tell which cells came from which.
    """
    w = Window.partitionBy("unique_id")
    return (history.withColumn("_n", F.count(F.lit(1)).over(w))
            .filter(F.col("_n") >= floor).drop("_n"))


def _expected_grid(panel: Panel, spark: SparkSession, origin: str, horizon: int,
                   floor: int) -> DataFrame:
    """Every (series, date) we asked for, flagged with whether the series had eligible history.

    This is what catches the adapter's sharpest edge: statsforecast forecasts forward from each
    SERIES' OWN last observation, not from the origin. A series whose eligible history ended
    months ago comes back with predictions for the wrong dates entirely. Joining onto the expected
    grid turns that into a declined cell with a reason instead of a silently misaligned one.
    """
    d0 = dt.date.fromisoformat(origin)
    dates = [(d0 + dt.timedelta(days=i),) for i in range(1, horizon + 1)]
    date_df = spark.createDataFrame(dates, "date date")
    raw = spark.read.parquet(panel.path)
    all_series = raw.select(F.col(panel.series_col).alias("series_id")).distinct()
    # How MUCH eligible history, not merely whether any: the two decline reasons are different
    # and a run that reports them as one cannot say whether the model was too demanding or the
    # product too new.
    lengths = (_eligible_history(panel, spark, origin)
               .groupBy(F.col("unique_id").alias("_h_sid"))
               .agg(F.count(F.lit(1)).alias("_n")))
    return (all_series.crossJoin(F.broadcast(date_df))
            .join(lengths, F.col("series_id") == F.col("_h_sid"), "left")
            .withColumn("_history_days", F.coalesce(F.col("_n"), F.lit(0)))
            .drop("_h_sid", "_n"))


__all__ = ["MODELS", "forecast_with"]
