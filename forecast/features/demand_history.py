"""Build encoder windows in Spark with explicit observation-state masks. An absent observation must
remain distinct from an observed zero when calculating normalization statistics and training
loss.
"""

from __future__ import annotations

import pathlib

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from spec.forecast_spec import ELIGIBLE_STATES, STATE_CODES

__all__ = ["FeatureError", "build"]


class FeatureError(ValueError):
    """A feature window was requested that the panel cannot supply."""


def build(
    panel_path: str | pathlib.Path,
    origin: str,
    encoder_days: int,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    series_col: str,
    time_col: str,
    target_col: str,
    exposure_col: str | None = None,
    oos_col: str | None = None,
    status_prob_cols: dict[str, str] | None = None,
    series_predicate: str | None = None,
    negative_target_policy: str = "refuse",
) -> pathlib.Path:
    """Write `series_id · history · observed · state · oos` for the window ending at `origin`.

    Returns the path. `state` carries the per-day `STATE_CODES` integers; `observed` remains the
    eligibility boolean for callers that only need a mask.

    Args:
        panel_path: Parquet source. A PATH and column names, not a `Panel` -- that type belongs
            to the harness, while model code stays independent of the harness. Inert data crosses
            this boundary, which is the same rule the harness applies to adapters.
        origin: Last day in the window, inclusive.
        encoder_days: Window length. Every series gets the SAME length -- a ragged window would
            make the model's time axis mean different things for different series.
        spark: Active session.
        out_dir: Parquet destination.
        series_col: Series identifier column.
        time_col: Date column.
        target_col: Quantity column.
        exposure_col: Observation-state column, or None when the dataset carries no exposure
            information -- in which case every day counts as observed.
        oos_col: Optional boolean shelf-state input. A zero-sale stockout is likelihood-
            unobserved but remains True here, so the encoder can distinguish it from never-ranged
            and missing days. With no explicit column, positive lower-bound states remain OOS.
        series_predicate: SQL boolean over the panel's columns restricting WHICH series this
            build covers -- the shard pushdown. Memory at catalogue scale is bounded by
            assembling one shard of whole factor blocks at a time, and the bound has to hold
            HERE, in the collect, not after it.
        status_prob_cols: SOFT states -- `{state: probability column}` over exactly the three
            contract states, for feeds where the state is inferred rather than measured
            (`ObservationSpec.status_prob_cols`). Mutually exclusive with `exposure_col`. Emits
            `p_exact / p_lower_bound / p_unobserved` arrays beside the hard columns; the hard
            `state` and `observed` become the MODAL reading, which is what the ENCODER
            conditions on, while the likelihood marginalises over the probabilities.

    Returns:
        The written path. A PATH, not a frame: this is the plane boundary and it is where Spark
        stops.

    Raises:
        FeatureError: `encoder_days` is not positive.
    """
    if encoder_days < 1:
        raise FeatureError(f"encoder_days must be >= 1, got {encoder_days}")
    if status_prob_cols is not None:
        if exposure_col is not None:
            raise FeatureError(
                "status_col OR status_prob_cols, not both -- the observation contract's own "
                "rule, enforced where the tensors are built")
        expected = {"exact", "lower_bound", "unobserved"}
        if set(status_prob_cols) != expected:
            raise FeatureError(
                f"soft states must cover exactly {sorted(expected)}, got "
                f"{sorted(status_prob_cols)}")
    lo = F.date_sub(F.to_date(F.lit(origin)), encoder_days - 1)
    hi = F.to_date(F.lit(origin))
    df = (spark.read.parquet(str(panel_path))
          .withColumn("_history_date", F.to_date(F.col(time_col)))
          .filter((F.col("_history_date") <= hi) & (F.col("_history_date") >= lo)))
    if series_predicate is not None:
        df = df.filter(F.expr(series_predicate))

    # Row count is not a calendar. Before this check, Jan-01 twice plus Jan-03 once satisfied a
    # three-day encoder and was packed as [Jan-01, Jan-01, Jan-03], silently assigning the second
    # Jan-01 observation to Jan-02's model position. Validate at the daily grain the architecture
    # declares; a duplicate is source corruption, while a genuine gap is dropped by the exact
    # length check below.
    duplicate = (df.groupBy(F.col(series_col), F.col("_history_date")).count()
                 .filter(F.col("count") != 1).select(series_col, "_history_date")
                 .limit(1).collect())
    if duplicate:
        row = duplicate[0]
        raise FeatureError(
            f"duplicate encoder key ({row[series_col]!r}, {row['_history_date']}); one daily "
            "position cannot contain two observations")

    if status_prob_cols is not None:
        p_exact = F.col(status_prob_cols["exact"]).cast("double")
        p_lower = F.col(status_prob_cols["lower_bound"]).cast("double")
        p_unobs = F.col(status_prob_cols["unobserved"]).cast("double")
        observed = p_unobs < 0.5
    else:
        observed = (F.col(exposure_col).isin(*sorted(ELIGIBLE_STATES))
                    if exposure_col else F.lit(True))
    # The STATE travels beside the value, as the integer codes the likelihood consumes
    # (`spec.forecast_spec.STATE_CODES`). The boolean alone cannot say which eligible kind a day
    # is, and collapsing `lower_bound` into `observed` here is exactly how the censoring
    # contract became unreachable from training: the trainer downstream could only rebuild
    # `exact`-or-`unobserved` from a bool, so a stockout that sold 5 first was scored as
    # exactly 5 and a vacuous all-day stockout as an observed zero.
    if status_prob_cols is not None:
        # The MODAL state, for the encoder's hard conditioning channels. Ties break toward the
        # more conservative reading: unobserved beats both, lower_bound beats exact.
        state = (F.when((p_unobs >= p_exact) & (p_unobs >= p_lower),
                        F.lit(STATE_CODES["unobserved"]))
                 .when(p_lower >= p_exact, F.lit(STATE_CODES["lower_bound"]))
                 .otherwise(F.lit(STATE_CODES["exact"])))
    elif exposure_col:
        # The fallback is -1, NOT `unobserved`: the contract says a state it does not define
        # must not be given a default. -1 fails the likelihood's state check loudly at training
        # time, where a silently-defaulted unknown state would just quietly drop the cell.
        state = F.coalesce(*[F.when(F.col(exposure_col) == name, F.lit(code))
                             for name, code in STATE_CODES.items()],
                           F.lit(-1))
    else:
        state = F.lit(STATE_CODES["exact"])
    oos = (F.col(oos_col).cast("boolean") if oos_col is not None
           else (state == F.lit(STATE_CODES["lower_bound"])))
    rows = df.select(
        F.col(series_col).alias("series_id"),
        F.col("_history_date").alias("d"),
        # A masked day still needs a NUMBER in the array -- the tensor is dense. Zero is the
        # safe filler precisely because the mask travels beside it: nothing downstream reads the
        # value without reading the mask.
        F.when(observed,
               F.greatest(F.col(target_col).cast("double"), F.lit(0.0))
               if negative_target_policy == "floor"
               else F.col(target_col).cast("double")).otherwise(0.0).alias("y"),
        observed.alias("obs"),
        state.alias("st"),
        oos.alias("oos"),
        *( [p_exact.alias("pe"), p_lower.alias("pl"), p_unobs.alias("pu")]
           if status_prob_cols is not None else []))

    ordered = Window.partitionBy("series_id").orderBy("d")
    out = (rows.withColumn("_i", F.row_number().over(ordered))
           .groupBy("series_id")
           .agg(F.collect_list(F.struct(
               "_i", "y", "obs", "st", "oos",
               *(("pe", "pl", "pu") if status_prob_cols is not None else ()))).alias("_w")))
    out = out.select(
        "series_id",
        F.transform(F.array_sort(F.col("_w"), _by_index()), lambda s: s["y"]).alias("history"),
        F.transform(F.array_sort(F.col("_w"), _by_index()), lambda s: s["obs"]).alias("observed"),
        F.transform(F.array_sort(F.col("_w"), _by_index()), lambda s: s["st"]).alias("state"),
        F.transform(F.array_sort(F.col("_w"), _by_index()), lambda s: s["oos"]).alias("oos"),
        *([F.transform(F.array_sort(F.col("_w"), _by_index()),
                       lambda s: s["pe"]).alias("p_exact"),
           F.transform(F.array_sort(F.col("_w"), _by_index()),
                       lambda s: s["pl"]).alias("p_lower_bound"),
           F.transform(F.array_sort(F.col("_w"), _by_index()),
                       lambda s: s["pu"]).alias("p_unobserved")]
          if status_prob_cols is not None else []))
    # A series with a short window is DROPPED rather than padded. Padding would put invented days
    # at the start of the encoder and the mask would say they were observed.
    out = out.filter(F.size("history") == encoder_days)
    out.write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir))


def _by_index():
    """Comparator ordering the collected structs by their row number.

    `collect_list` does not promise an order -- it is the order tasks happened to finish in. An
    encoder window assembled in task-completion order is a shuffled history, and the model would
    train on it perfectly happily.
    """
    return lambda a, b: F.when(a["_i"] < b["_i"], -1).when(a["_i"] > b["_i"], 1).otherwise(0)


def summarise(path: str | pathlib.Path, spark: SparkSession) -> DataFrame:
    """Coverage of the written window: how much of each history was actually observed.

    Reported rather than assumed, because a window that is 90% masked produces a RevIN statistic
    from a handful of days and the model has no way to say so.
    """
    df = spark.read.parquet(str(path))
    return df.select(
        F.size("history").alias("n"),
        F.aggregate("observed", F.lit(0), lambda acc, x: acc + x.cast("int")).alias("n_observed"),
    ).agg(F.avg(F.col("n_observed") / F.col("n")).alias("mean_coverage"),
          F.min(F.col("n_observed") / F.col("n")).alias("min_coverage"),
          F.count(F.lit(1)).alias("n_series"))
