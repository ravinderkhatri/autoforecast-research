"""Price, discount depth and promotion features with per-channel availability. Missing channels are
masked rather than interpreted as observed zero prices or discounts. Channel order remains fixed
for positional model embeddings.
"""

from __future__ import annotations

import pathlib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

__all__ = ["CHANNELS", "CommercialError", "build"]

CHANNELS = ("price", "discount_depth", "promo")
"""The commercial channels, in a FIXED order.

Fixed because the model's per-channel embeddings are positional: reordering them silently trains
each weight on a different column, and nothing in the loss would notice.
"""


class CommercialError(ValueError):
    """A commercial channel was requested that the panel does not describe."""


def _finite(column):
    """Spark expression for a present finite numeric value (NaN/Inf are unavailable)."""
    value = column.cast("double")
    return value.isNotNull() & ~F.isnan(value) & (F.abs(value) != F.lit(float("inf")))


def build(
    panel_path: str | pathlib.Path,
    origin: str,
    horizon: int,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    series_col: str,
    time_col: str,
    price_col: str | None = None,
    promo_col: str | None = None,
    reference_days: int = 28,
    plan_path: str | pathlib.Path | None = None,
    series_predicate: str | None = None,
) -> pathlib.Path:
    """Write `series_id · date · price · discount_depth · promo · *_available`. Returns the path.

    Discount depth is derived rather than read: `1 - price / reference`, where the reference is
    the series' median price over the preceding `reference_days`. A raw price says nothing without
    a level to compare it to, and the level differs by three orders of magnitude across a
    catalogue -- so the model would have to learn one product's normal price before it could tell
    a discount from a cheap item.

    Args:
        panel_path: Parquet source.
        origin: Forecast origin.
        horizon: Days after the origin to emit. These are KNOWN-FUTURE values -- a planned price
            and a scheduled promotion -- so emitting them for the horizon is legitimate. Emitting
            anything not known at `origin` here is the leak the decoder's docstring describes and
            no runtime check can catch it.
        spark: Active session.
        out_dir: Parquet destination.
        series_col: Series identifier column.
        time_col: Date column.
        price_col: Price column, or None if this client has none.
        promo_col: Promotion flag column, or None.
        reference_days: Window for the price reference.
        plan_path: A prepared AS-OF plan selection (`features/plans.build` output: series_id,
            date, price/promo roles) to use as the HORIZON values instead of the panel's own
            future rows. The reference window still comes from the panel's HISTORY (at or
            before the origin, which is recorded fact either way); only the future stops being
            read from recorded outcomes. Availability then means "the plan covers this cell",
            which is the honest live semantics. A plan may be the only source of price/promo;
            planned raw price is then available, but discount depth is unavailable because the
            panel has no historical price from which to derive an honest reference.

    Returns:
        The written path.

    Raises:
        CommercialError: neither the panel nor the plan supplies price/promotion, so the branch
            has no channels at all.
    """
    if price_col is None and promo_col is None and plan_path is None:
        raise CommercialError(
            "neither a price nor a promotion column was given, so the commercial branch has no "
            "channels. Drop the branch for this client rather than feeding it an all-masked row: "
            "attention over an empty key set is undefined")
    df = spark.read.parquet(str(panel_path))
    if series_predicate is not None:
        df = df.filter(F.expr(series_predicate))
    lo = F.date_sub(F.to_date(F.lit(origin)), reference_days - 1)
    hi = F.date_add(F.to_date(F.lit(origin)), horizon)

    if price_col is not None:
        ref = (df.filter((F.col(time_col) >= lo)
                         & (F.col(time_col) <= F.to_date(F.lit(origin)))
                         & _finite(F.col(price_col)))
               .groupBy(F.col(series_col).alias("_sid"))
               .agg(F.expr(f"percentile_approx({price_col}, 0.5)").alias("_ref")))
    if plan_path is not None:
        plan = spark.read.parquet(str(plan_path))
        if not ({"price", "promo"} & set(plan.columns)):
            raise CommercialError(
                "the plan has neither a standardized price nor promo role, so the commercial "
                "branch has no channels. Drop the branch rather than feeding it an all-masked "
                "row: attention over an empty key set is undefined")
        out = plan.select(
            F.col("series_id"), F.to_date(F.col("date")).alias("date"),
            (F.col("price").cast("double") if "price" in plan.columns
             else F.lit(None).cast("double")).alias("price"),
            (F.col("promo").cast("double") if "promo" in plan.columns
             else F.lit(None).cast("double")).alias("promo"))
        out = out.filter((F.col("date") > F.to_date(F.lit(origin))) & (F.col("date") <= hi))
    else:
        window = df.filter(
            (F.col(time_col) > F.to_date(F.lit(origin))) & (F.col(time_col) <= hi))
        out = window.select(F.col(series_col).alias("series_id"), F.col(time_col).alias("date"),
                            *([F.col(price_col).cast("double").alias("price")] if price_col
                              else [F.lit(None).cast("double").alias("price")]),
                            *([F.col(promo_col).cast("double").alias("promo")] if promo_col
                              else [F.lit(None).cast("double").alias("promo")]))
    if price_col is not None:
        out = (out.join(ref, out["series_id"] == F.col("_sid"), "left")
               # A reference of zero or null gives no depth -- null, not zero, because zero depth
               # is a real value meaning "sold at the usual price".
               .withColumn("discount_depth",
                           F.when(F.col("_ref") > 0, 1.0 - F.col("price") / F.col("_ref")))
               .drop("_sid", "_ref"))
    else:
        out = out.withColumn("discount_depth", F.lit(None).cast("double"))

    for channel in CHANNELS:
        available = _finite(F.col(channel))
        out = out.withColumn(f"{channel}_available", available)
        # The VALUE under an unavailable channel is zero and is never read without its flag. It is
        # filled at all only because the tensor is dense.
        out = out.withColumn(channel, F.when(available, F.col(channel)).otherwise(F.lit(0.0)))
    out.select("series_id", "date", *CHANNELS,
               *[f"{c}_available" for c in CHANNELS]).write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir))
