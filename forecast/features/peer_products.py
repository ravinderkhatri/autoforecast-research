"""Optional peer-product messages from known-future price and promotion inputs. Never use
neighboring future sales. Separate signed channels represent complement and substitution
relationships; candidate edges come from residual co-purchase and taxonomy similarity. This
branch is disabled in the initial model and requires separate ablation evidence.
"""

from __future__ import annotations

import pathlib

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

__all__ = ["PeerError", "build_messages", "complement_edges", "substitute_edges"]


class PeerError(ValueError):
    """A peer graph was requested that would leak or that cannot be built."""


def complement_edges(
    baskets: DataFrame, *, basket_col: str, product_col: str, top_k: int = 10,
    min_support: int = 20,
) -> DataFrame:
    """Degree-corrected co-purchase edges: products bought together more than popularity explains.

    RAW co-occurrence is a popularity ranking. Bananas co-occur with everything, so an
    uncorrected graph gives every product the same neighbours and the branch learns nothing
    product-specific. Dividing by the product of the marginal counts is the correction -- what
    survives is co-purchase ABOVE what independence would produce.

    Args:
        baskets: One row per (basket, product).
        basket_col: Basket identifier.
        product_col: Product identifier.
        top_k: Neighbours kept per product.
        min_support: Co-occurrences below which a pair is dropped as noise. A pair seen twice has
            a spectacular lift and means nothing.

    Returns:
        `product · peer · weight`, weight positive.

    Raises:
        PeerError: `top_k` or `min_support` below 1.
    """
    if top_k < 1 or min_support < 1:
        raise PeerError(f"top_k and min_support must be >= 1, got {top_k} and {min_support}")
    counts = baskets.groupBy(product_col).agg(F.count(F.lit(1)).alias("_n"))
    pairs = (baskets.alias("a").join(baskets.alias("b"), basket_col)
             .filter(F.col(f"a.{product_col}") < F.col(f"b.{product_col}"))
             .groupBy(F.col(f"a.{product_col}").alias("product"),
                      F.col(f"b.{product_col}").alias("peer"))
             .agg(F.count(F.lit(1)).alias("_co"))
             .filter(F.col("_co") >= min_support))
    lifted = (pairs
              .join(counts.select(F.col(product_col).alias("product"),
                                  F.col("_n").alias("_na")), "product")
              .join(counts.select(F.col(product_col).alias("peer"),
                                  F.col("_n").alias("_nb")), "peer")
              .withColumn("weight", F.col("_co") / F.sqrt(F.col("_na") * F.col("_nb"))))
    from pyspark.sql import Window
    ranked = lifted.withColumn(
        "_r", F.row_number().over(Window.partitionBy("product").orderBy(F.desc("weight"))))
    return ranked.filter(F.col("_r") <= top_k).select("product", "peer", "weight")


def substitute_edges(
    catalogue: DataFrame, *, product_col: str, taxonomy_cols: tuple[str, ...], top_k: int = 10,
) -> DataFrame:
    """Taxonomy-similarity edges: products that fill the same role.

    A SEPARATE edge set, not an afterthought. Two brands of the same thing usually never appear in
    one basket, so a co-purchase graph scores them at approximately zero -- a graph built only
    from co-occurrence contains no substitutes at all, and the branch would then be structurally
    incapable of seeing cannibalisation while appearing to work.

    Similarity is agreement on the taxonomy path, weighted toward the FINEST level: two products
    in the same sub-category are far more substitutable than two in the same department.

    Raises:
        PeerError: no taxonomy columns, so there is nothing to compare on.
    """
    if not taxonomy_cols:
        raise PeerError(
            "no taxonomy columns, so substitutes cannot be found. Falling back to co-purchase "
            "alone would silently produce a graph with no substitutes in it")
    a = catalogue.alias("a")
    b = catalogue.alias("b")
    joined = a.join(b, F.col(f"a.{product_col}") != F.col(f"b.{product_col}"))
    # Finest level weighted highest: level i of n contributes 2^i, so agreeing on sub-category
    # outweighs agreeing on department.
    score = F.lit(0.0)
    for i, col in enumerate(taxonomy_cols):
        score = score + F.when(F.col(f"a.{col}") == F.col(f"b.{col}"), float(2 ** i)).otherwise(0.0)
    total = float(sum(2 ** i for i in range(len(taxonomy_cols))))
    scored = joined.select(F.col(f"a.{product_col}").alias("product"),
                           F.col(f"b.{product_col}").alias("peer"),
                           (score / total).alias("weight")).filter(F.col("weight") > 0)
    from pyspark.sql import Window
    ranked = scored.withColumn(
        "_r", F.row_number().over(Window.partitionBy("product").orderBy(F.desc("weight"),
                                                                        F.asc("peer"))))
    return ranked.filter(F.col("_r") <= top_k).select("product", "peer", "weight")


def build_messages(
    known_future_path: str | pathlib.Path,
    complements: DataFrame,
    substitutes: DataFrame,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    product_col: str = "series_id",
    date_col: str = "date",
    signal_cols: tuple[str, ...] = ("discount_depth", "promo"),
) -> pathlib.Path:
    """Aggregate neighbours' KNOWN-FUTURE signals into per-cell messages, in signed channels.

    Emits `<signal>_complement` and `<signal>_substitute` per cell -- separate columns, never
    summed. Summing them here would do exactly what softmax pooling does and for the same reason:
    a large complement effect would hide a substitution before the model saw either.

    Args:
        known_future_path: Output of `features.commercial.build` -- planned price and promotion,
            per product per horizon day. **The target column is not read and must not be present
            in `signal_cols`**; a neighbour's SALES are unknown at forecast time, and in backtest
            they are available, so the leak scores well and cannot be reproduced live.
        complements: From `complement_edges`.
        substitutes: From `substitute_edges`.
        spark: Active session.
        out_dir: Parquet destination.
        product_col: Product identifier in the known-future frame.
        date_col: Date column.
        signal_cols: Which known-future signals to carry.

    Returns:
        The written path.

    Raises:
        PeerError: a signal column is missing, or looks like a target rather than a plan.
    """
    known = spark.read.parquet(str(known_future_path))
    missing = [c for c in signal_cols if c not in known.columns]
    if missing:
        raise PeerError(f"signal columns absent from the known-future frame: {missing}")
    suspect = [c for c in signal_cols if c.lower() in
               ("qty", "sales", "units", "demand", "y", "target", "revenue")]
    if suspect:
        raise PeerError(
            f"{suspect} looks like a TARGET rather than a plan. A neighbour's sales are unknown "
            "at forecast time; using them means needing a forecast of the neighbour first, and in "
            "backtest they are available -- so the leak scores beautifully and cannot be "
            "reproduced live")

    out = None
    for kind, edges in (("complement", complements), ("substitute", substitutes)):
        peer_signal = known.select(
            F.col(product_col).alias("peer"), F.col(date_col).alias("date"),
            *[F.col(c).alias(f"_{c}") for c in signal_cols])
        agg = (edges.join(peer_signal, "peer")
               .groupBy("product", "date")
               .agg(*[(F.sum(F.col(f"_{c}") * F.col("weight")) / F.sum("weight")).alias(
                   f"{c}_{kind}") for c in signal_cols]))
        out = agg if out is None else out.join(agg, ["product", "date"], "outer")
    # A product with no neighbours of one kind gets zero on that channel and a flag, not a null:
    # the tensor is dense, and "no substitutes found" is a real state distinct from "substitutes
    # found and they are doing nothing".
    for kind in ("complement", "substitute"):
        for c in signal_cols:
            col = f"{c}_{kind}"
            out = (out.withColumn(f"{col}_present", F.col(col).isNotNull())
                   .withColumn(col, F.coalesce(F.col(col), F.lit(0.0))))
    out.write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir))
