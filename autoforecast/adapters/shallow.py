"""A global shallow model: LightGBM, one model over every series, direct multi-horizon.

This is the comparator Gate 2 is really about. If a gradient-boosted model over lagged baselines
and calendar features captures most of the available accuracy, a custom deep model is not
justified -- and finding that out costs days rather than months.

DIRECT, NOT RECURSIVE. Features are computed AS OF THE ORIGIN and the horizon step is itself a
feature, so predicting day 28 never consumes a prediction for day 27. Recursive forecasting would
need lag-7 at day 28, which does not exist at forecast time; feeding a prediction back in is how a
leak-free-looking pipeline starts training on its own output.

WHERE THE PLANE BOUNDARY SITS, precisely. Feature construction runs in SPARK over the whole panel
and writes Parquet -- that is the part that has to survive large panels, and it is the part that
will run unchanged on the estate. TRAINING is single-machine LightGBM on a sampled frame. That is
a deliberate ceiling, not an oversight; see the marker on `fit_predict`.
"""

from __future__ import annotations

import datetime as dt
import gc
import pathlib

import lightgbm as lgb
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters import _validate_daily_span
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel

NAME = "shallow_global"

EMITTED = "emitted"
DECLINED_NO_FEATURES = "declined_no_eligible_history"

_HASH_BUCKETS = 10_000
"""Granularity of the stable sampling hash. 10,000 buckets resolves a fraction to 0.01%."""

LAG_WINDOWS = (7, 28, 91)
"""Trailing windows the features are summarised over, in days.

7 and 28 are the operating cycles retail plans on. 91 is there because a slow mover shows nothing
in 28 days: at M5's eligible zero rate of 59.5%, a 28-day window on a bottom-decile series can be
entirely zero, and a feature that is always zero cannot separate anything.
"""

FEATURES = (
    "step",
    "dow",
    "dom",
    "month",
    *[f"mean_{w}" for w in LAG_WINDOWS],
    *[f"zero_rate_{w}" for w in LAG_WINDOWS],
    *[f"max_{w}" for w in LAG_WINDOWS],
    "dow_mean_28",
    "days_since_last_sale",
    "history_days",
)


def fit_predict(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    train_origins: tuple[str, ...] = (),
    max_train_rows: int = 750_000,
    seed: int = 0,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    min_data_in_leaf: int = 200,
    num_boost_round: int = 300,
    checkpoint_dir: str | pathlib.Path | None = None,
) -> Forecast:
    """Train a global hurdle pair on `train_origins` and forecast `horizon` days after `origin`.

    The head is a HURDLE PAIR, matching the architecture's output rather than a single regressor:
    a binary classifier for "does this cell sell at all" and a regressor for "how much, given it
    sells", multiplied. On a panel that is 59.5% zeros on eligible cells, a single squared-error
    regressor spends its capacity predicting the mean of a mostly-zero distribution and returns a
    flat fraction everywhere -- which scores acceptably and is useless.

    Args:
        panel: Data.
        origin: Forecast origin, ISO. Features are computed at or before it.
        horizon: Days to forecast.
        spark: Active session.
        out_dir: Where the forecast Parquet is written.
        train_origins: Past origins to build training rows from. Must all be strictly before
            `origin`, and the check is enforced -- a training origin at or after the forecast
            origin trains on the period being predicted. Defaults to eight origins stepping back
            by `horizon` from `origin`.
        max_train_rows: Cap on rows pulled to the driver for training. Default measured, not
            guessed: 3,000,000 killed the Phase 2 run on a 16 GB machine, where the JVM heap and
            the driver's pandas frame and LightGBM's own copy of it all come out of the same
            memory. 750,000 rows over M5's 30,490 series is still a global model.
        seed: Sampling and LightGBM seed, so a gate is reproducible.
        num_leaves: Maximum leaves in each LightGBM tree; declared model capacity.
        learning_rate: LightGBM shrinkage rate from the checked model profile.
        min_data_in_leaf: Minimum training rows per leaf, preventing tiny-leaf fits.
        num_boost_round: Fixed count of boosting rounds for each hurdle head.
        checkpoint_dir: Optional empty directory for the two fitted LightGBM model files.

    Returns:
        A validated `Forecast`.

    Raises:
        ValueError: non-positive horizon, or a training origin at or after `origin`.
    """
    _validate_daily_span(panel, origin, horizon, adapter=NAME)
    _validate_training_options(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        num_boost_round=num_boost_round,
    )
    origins = train_origins or _default_train_origins(origin, horizon)
    late = [o for o in origins if o >= origin]
    if late:
        raise ValueError(
            f"training origins at or after the forecast origin {origin}: {sorted(late)}. Those "
            "windows include the period being predicted, which is the leak this argument exists "
            "to make visible rather than convenient"
        )

    raw = spark.read.parquet(panel.path)
    train = _training_frame(raw, panel, origins, horizon, spark)
    predict = _feature_frame(raw, panel, origin, horizon, spark)

    # training is single-machine; the FEATURE BUILD above is distributed and is the part
    # that has to scale. THE TRIGGER HAS ALREADY FIRED ONCE, on M5's ten stores at
    # max_train_rows=3_000_000 -- the process was killed with no traceback. It was bought back by
    # sampling before sorting and a smaller cap, both of which are real fixes, but the ceiling is
    # unchanged: the first estate client, or a panel whose sampled frame exceeds driver memory,
    # moves this to SynapseML's LightGBMRegressor, which is Spark-native and takes the same
    # Parquet; distributed materialization remains a limitation.
    frame = _to_driver(train, max_train_rows, seed)
    models = _fit(
        frame,
        seed,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        num_boost_round=num_boost_round,
    )
    try:
        forecast = _predict(models, predict, panel, origin, horizon, out_dir, spark)
        if checkpoint_dir is not None:
            _write_checkpoint(models, checkpoint_dir)
        return forecast
    finally:
        # LightGBM's Dataset and Booster hold NATIVE memory that Python frees only when the
        # objects are collected, and a generational GC will not do that promptly. One fit at M5's
        # ten stores is comfortable (measured: 392s, 30,474 series, no crash); THREE in one
        # process, which is what a three-origin ranking does, was killed by the OS with no
        # traceback. Releasing at the end of each fit is what makes the sequence survivable.
        del frame, models
        gc.collect()


def _default_train_origins(origin: str, horizon: int) -> tuple[str, ...]:
    """Eight origins stepping back by one horizon each, so training windows do not overlap.

    Overlapping windows would repeat the same actuals across training rows, which inflates the
    effective sample size and makes early stopping optimistic without changing anything visible.
    """
    d0 = dt.date.fromisoformat(origin)
    return tuple((d0 - dt.timedelta(days=horizon * i)).isoformat() for i in range(1, 9))[::-1]


def _feature_frame(
    raw: DataFrame, panel: Panel, origin: str, horizon: int, spark: SparkSession
) -> DataFrame:
    """Features as of `origin`, crossed with the horizon steps. One row per (series, target day).

    Everything here is computed from data at or before `origin`. The horizon step and the TARGET
    day's calendar are the only forward-looking columns, and both are known at forecast time --
    which is what makes this direct rather than recursive.
    """
    d0 = dt.date.fromisoformat(origin)
    hist = raw.filter(F.col(panel.time_col) <= F.to_date(F.lit(origin)))
    if panel.has_exposure:
        hist = hist.filter(F.col(panel.exposure_col).isin(*sorted(ELIGIBLE_STATES)))
    hist = hist.select(
        F.col(panel.series_col).alias("series_id"),
        F.col(panel.time_col).alias("d"),
        F.col(panel.target_col).cast("double").alias("y"),
    )

    aggs = [
        F.count(F.lit(1)).alias("history_days"),
        F.max(F.when(F.col("y") > 0, F.col("d"))).alias("_last_sale"),
    ]
    for w in LAG_WINDOWS:
        lo = F.date_sub(F.to_date(F.lit(origin)), w - 1)
        inside = F.col("d") >= lo
        aggs += [
            F.avg(F.when(inside, F.col("y"))).alias(f"mean_{w}"),
            F.avg(F.when(inside, F.when(F.col("y") == 0, 1.0).otherwise(0.0))).alias(
                f"zero_rate_{w}"
            ),
            F.max(F.when(inside, F.col("y"))).alias(f"max_{w}"),
        ]
    per_series = hist.groupBy("series_id").agg(*aggs)

    # Day-of-week level, the one seasonal feature a shallow model genuinely needs here: retail
    # demand at daily grain is dominated by day of week, and without it the model can only learn
    # a flat rate per series -- which is Croston, already in the comparison.
    lo28 = F.date_sub(F.to_date(F.lit(origin)), 27)
    dow_mean = (
        hist.filter(F.col("d") >= lo28)
        .groupBy("series_id", F.dayofweek("d").alias("dow"))
        .agg(F.avg("y").alias("dow_mean_28"))
    )

    dates = [(d0 + dt.timedelta(days=i), i) for i in range(1, horizon + 1)]
    grid = spark.createDataFrame(dates, "date date, step int")
    out = (
        per_series.crossJoin(F.broadcast(grid))
        .withColumn("dow", F.dayofweek("date"))
        .withColumn("dom", F.dayofmonth("date"))
        .withColumn("month", F.month("date"))
        .withColumn(
            "days_since_last_sale", F.datediff(F.to_date(F.lit(origin)), F.col("_last_sale"))
        )
        .drop("_last_sale")
    )
    return out.join(dow_mean, ["series_id", "dow"], "left")


def _training_frame(
    raw: DataFrame, panel: Panel, origins: tuple[str, ...], horizon: int, spark: SparkSession
) -> DataFrame:
    """Feature rows for several past origins, each labelled with the actual that followed.

    Built with exactly the same function the prediction uses, so a feature that is available at
    training time and not at forecast time cannot exist -- the commonest silent leak in a
    tabular forecasting pipeline is a training feature the serving path cannot compute.
    """
    actuals = raw.select(
        F.col(panel.series_col).alias("_sid"),
        F.col(panel.time_col).alias("_d"),
        F.col(panel.target_col).cast("double").alias("_y"),
        *([F.col(panel.exposure_col).alias("_state")] if panel.has_exposure else []),
    )
    frames = []
    for o in origins:
        f = _feature_frame(raw, panel, o, horizon, spark)
        # Only the following horizon can label this origin. Prune the 1,941-day M5 source
        # before the join: otherwise Spark may broadcast millions of irrelevant actual rows
        # for each training origin and exhaust the local driver's broadcast memory.
        labels = actuals.filter(
            (F.col("_d") > F.to_date(F.lit(o)))
            & (F.col("_d") <= F.date_add(F.to_date(F.lit(o)), horizon))
        )
        j = f.join(labels, (f["series_id"] == F.col("_sid")) & (f["date"] == F.col("_d")), "inner")
        if panel.has_exposure:
            j = j.filter(F.col("_state").isin(*sorted(ELIGIBLE_STATES)))
        # series_id and date are carried only as a deterministic sort key and dropped in
        # `_to_driver`. See there: without a stable order LightGBM's bagging picks different
        # rows and the same data trains a different model.
        frames.append(j.select("series_id", "date", *FEATURES, F.col("_y").alias("y")))
    return _union(frames)


def _union(frames: list[DataFrame]) -> DataFrame:
    """Stack training frames from several origins."""
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out


def _to_driver(train: DataFrame, max_rows: int, seed: int):
    """Sample down to `max_rows` by a stable hash, then sort, then materialise for LightGBM.

    ORDER MATTERS TWICE HERE, in two different ways.

    **The result must be sorted**, and it is not cosmetic. Spark does not promise a row order, so
    the same training rows arrive in a different sequence when the input's partition layout
    differs -- and LightGBM's bagging selects by row POSITION, so an identical dataset trains a
    DIFFERENT model. Caught by a leak test that compared a run against the full panel with one
    against a panel truncated at the origin: the frames were identical as SETS (0 rows in either
    difference) and the forecasts still differed in all 21,343 cells.

    **But the sample must come first.** Sorting the whole frame before sampling means a full
    shuffle-sort of every training row -- 6.8M of them on M5's ten stores -- and that is where the
    Phase 2 run was killed, with no traceback, which is what OS memory pressure looks like.
    Sampling first leaves a sort of `max_rows`.

    `Spark.sample` cannot do that safely: its RNG is seeded per PARTITION, so re-partitioned input
    yields a different sample and the determinism above is lost again. A stable hash of the row's
    identity is partition-independent by construction, which is what this uses.
    """
    n = train.count()
    if n and n > max_rows:
        # Murmur3 over (series, date): the same row is kept or dropped whatever partition it
        # lands in, and whatever order it arrives in.
        cut = int(min(1.0, max_rows / n) * _HASH_BUCKETS)
        train = train.filter(
            F.pmod(F.hash("series_id", "date", F.lit(seed)), F.lit(_HASH_BUCKETS)) < cut
        )
    return train.orderBy("series_id", "date").drop("series_id", "date").toPandas()


def _validate_training_options(
    *,
    num_leaves: int,
    learning_rate: float,
    min_data_in_leaf: int,
    num_boost_round: int,
) -> None:
    """Reject malformed profile/HPO values before the native runtime is reached."""
    if type(num_leaves) is not int or num_leaves < 2:
        raise ValueError("num_leaves must be an integer of at least 2")
    if type(min_data_in_leaf) is not int or min_data_in_leaf < 1:
        raise ValueError("min_data_in_leaf must be a positive integer")
    if type(num_boost_round) is not int or num_boost_round < 1:
        raise ValueError("num_boost_round must be a positive integer")
    if (
        not isinstance(learning_rate, int | float)
        or isinstance(learning_rate, bool)
        or not 0 < float(learning_rate) <= 1
    ):
        raise ValueError("learning_rate must be a finite number in (0, 1]")


def _fit(
    frame,
    seed: int,
    *,
    num_leaves: int,
    learning_rate: float,
    min_data_in_leaf: int,
    num_boost_round: int,
) -> tuple[lgb.Booster, lgb.Booster]:
    """The hurdle pair: P(y > 0), and E[y | y > 0]. Returns (occurrence, size).

    Each booster drops its training Dataset before returning. A Booster keeps a reference to the
    Dataset it was built from, so without this the training matrix stays alive for as long as the
    model does -- which across a multi-origin ranking is every model at once.

    **`num_threads=1`, AND IT IS NOT A PERFORMANCE CHOICE.** This trains on the DRIVER, after
    `.toPandas()`, and the driver goes on to do more Spark work afterwards -- which forks Python
    workers. OpenMP is not fork-safe: a child inheriting an initialised OpenMP runtime crashes
    when its threads reach the fork barrier. Measured on this machine, as a SIGSEGV inside
    `libomp.dylib`:

        __kmp_suspend_initialize_thread  <-  __kmp_fork_barrier  <-  __kmp_launch_worker

    with no Python traceback, because the fault is below the interpreter. Leaving the thread count
    at its default made `shallow_global` unable to complete a single origin.

    One thread keeps LightGBM from creating the pool at all, and it costs almost nothing: measured
    at 800k rows over 300 rounds, 9.2 s against 5.5 s for the default -- against a phase measured
    in hours. Parallelism here comes from Spark across partitions, not from OpenMP inside one
    booster, and nesting the two oversubscribes anyway (4 threads beat 10 on the same benchmark).

    It also makes training MORE reproducible: single-threaded reductions have a fixed summation
    order, so the bagged split scores no longer depend on how the work happened to be divided.
    """
    x = frame[list(FEATURES)].astype("float32")
    y = frame["y"].to_numpy()
    common = dict(
        verbosity=-1,
        seed=seed,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        num_threads=1,
    )
    occurrence = lgb.train(
        {**common, "objective": "binary"},
        lgb.Dataset(x, label=(y > 0).astype("int8")),
        num_boost_round=num_boost_round,
    )
    positive = y > 0
    size = lgb.train(
        {**common, "objective": "tweedie", "tweedie_variance_power": 1.2},
        lgb.Dataset(x[positive], label=y[positive]),
        num_boost_round=num_boost_round,
    )
    occurrence.free_dataset()
    size.free_dataset()
    return occurrence, size


def _write_checkpoint(
    models: tuple[lgb.Booster, lgb.Booster],
    directory: str | pathlib.Path,
) -> None:
    """Persist both hurdle heads as an inference-only LightGBM artifact pair."""
    target = pathlib.Path(directory)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"checkpoint directory must be absent or empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    occurrence, size = models
    occurrence.save_model(str(target / "occurrence.txt"))
    size.save_model(str(target / "positive_size.txt"))


def _predict(
    models: tuple[lgb.Booster, lgb.Booster],
    features: DataFrame,
    panel: Panel,
    origin: str,
    horizon: int,
    out_dir: str | pathlib.Path,
    spark: SparkSession,
) -> Forecast:
    """Score the prediction grid and write `series_id · date · yhat`.

    The grid is small -- series x horizon, not series x history -- so it is scored on the driver.
    That is the one place this adapter's ceiling shows at forecast time, and it is bounded by the
    horizon rather than by the length of the history.
    """
    grid = features.toPandas()
    n_series = int(grid["series_id"].nunique()) if len(grid) else 0
    usable = grid.dropna(subset=[f"mean_{LAG_WINDOWS[0]}"])
    occurrence, size = models
    x = usable[list(FEATURES)].astype("float32")
    yhat = occurrence.predict(x) * size.predict(x)
    out = usable[["series_id", "date"]].copy()
    out["yhat"] = yhat

    disposition = {EMITTED: int(len(out))}
    declined = int(len(grid) - len(out))
    if declined:
        disposition[DECLINED_NO_FEATURES] = declined
    spark.createDataFrame(out).write.mode("overwrite").parquet(str(out_dir))
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


__all__ = ["FEATURES", "LAG_WINDOWS", "NAME", "fit_predict"]
