"""Product and store identity: attributes, a frozen text embedding, and a small learned residual.

THREE SOURCES, IN DECREASING ORDER OF TRUST, and the order is the design:

  1. STRUCTURED ATTRIBUTES -- category, department, pack size. Known for a new product on the day
     it is created, which is why they carry the cold start.
  2. A FROZEN TEXT EMBEDDING of the description. Frozen, not fine-tuned: fine-tuning it on this
     objective lets a description memorise its series' demand level, and the embedding then stops
     generalising to the new product it exists to help.
  3. A SMALL, WEIGHT-DECAYED RESIDUAL per series. Sixteen dimensions, decayed, because its job is
     to carry what the attributes miss -- not to become a lookup table of every series' level. An
     unconstrained per-series embedding will do exactly that, score wonderfully in backtest, and
     have nothing to say about a product launched next week.

`allow_joint_embedding=False` by DEFAULT, and this module honours it: identity is FACTORISED into
product and store rather than looked up on their cross-product. A joint embedding has one row per
(product, store) pair. Sparse cross-products can give each row too little evidence for a useful
estimate, while factorized embeddings share information across observed combinations.
"""

from __future__ import annotations

import pathlib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

__all__ = ["IdentityError", "build", "vocabulary"]


class IdentityError(ValueError):
    """An identity feature was requested that the panel cannot supply."""


def build(
    panel_path: str | pathlib.Path,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    series_col: str,
    attribute_cols: tuple[str, ...],
    product_col: str | None = None,
    store_col: str | None = None,
    allow_joint_embedding: bool = False,
    series_predicate: str | None = None,
    shard_expr: str | None = None,
) -> pathlib.Path:
    """Write one row per series: its attributes and its factorised identity keys.

    Args:
        panel_path: Parquet source.
        spark: Active session.
        out_dir: Parquet destination.
        series_col: Series identifier.
        attribute_cols: Structured attribute columns. Encoding happens in the model; what is
            emitted here is the raw categorical, so the vocabulary is derived once and shared.
        product_col: Product identifier, for the factorised product key.
        store_col: Store identifier, for the factorised store key.
        allow_joint_embedding: Whether a (product, store) cross-product key may be emitted.
            **Refused by default.** See the module docstring: the joint table has one row per pair
            and most pairs are seen a handful of times.
        shard_expr: Optional Spark SQL expression assigning each series to a shard. The assignment
            is retained as ``_shard`` so a sharded caller can push its filter into this table
            instead of collecting every series' identity row for every shard.

    Returns:
        The written path.

    Raises:
        IdentityError: no attributes were given, or a joint embedding was requested without both
            identifier columns.
    """
    if not attribute_cols:
        raise IdentityError(
            "no attribute columns: the static branch would carry only a learned residual, which "
            "is a lookup table with no cold-start path -- and every launch goes through that path")
    if allow_joint_embedding and not (product_col and store_col):
        raise IdentityError("a joint embedding needs both a product and a store column to cross")

    df = spark.read.parquet(str(panel_path))
    if series_predicate is not None:
        df = df.filter(F.expr(series_predicate))
    cols = [F.col(series_col).alias("series_id"), *[F.col(c) for c in attribute_cols]]
    if product_col:
        cols.append(F.col(product_col).alias("product_key"))
    if store_col:
        cols.append(F.col(store_col).alias("store_key"))
    if shard_expr is not None:
        cols.append(F.expr(shard_expr).alias("_shard"))
    # Prove the premise before using ``first``. Spark's first row is partition-order dependent;
    # when a supposedly static attribute changes, silently choosing one makes checkpoint identity
    # nondeterministic and can leak a future catalogue correction into an earlier backtest.
    identity_cols = [*attribute_cols]
    if product_col:
        identity_cols.append("product_key")
    if store_col:
        identity_cols.append("store_key")
    if shard_expr is not None:
        identity_cols.append("_shard")
    selected = df.select(*cols)
    per_series = selected.groupBy("series_id").agg(
        *[F.first(c, ignorenulls=False).alias(c) for c in identity_cols],
        *[F.countDistinct(F.struct(F.col(c).isNull(), F.col(c).cast("string")))
          .alias(f"_distinct_{i}") for i, c in enumerate(identity_cols)])
    conflict_condition = F.lit(False)
    for i in range(len(identity_cols)):
        conflict_condition = conflict_condition | (F.col(f"_distinct_{i}") > 1)
    conflict = per_series.filter(conflict_condition).select("series_id").limit(1).collect()
    if conflict:
        raise IdentityError(
            f"static identity changes within series {conflict[0]['series_id']!r}; a static role "
            "must have exactly one value (including nullness) over the panel")
    per_series = per_series.drop(*[f"_distinct_{i}" for i in range(len(identity_cols))])

    if allow_joint_embedding:
        per_series = per_series.withColumn(
            "joint_key", F.concat_ws("|", "product_key", "store_key"))
    per_series.write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir))


def vocabulary(path: str | pathlib.Path, column: str, spark: SparkSession) -> list[str]:
    """Sorted distinct values of `column`, which becomes that attribute's embedding vocabulary.

    SORTED, so the same data gives the same index every run. A vocabulary built in whatever order
    Spark returned rows would silently permute every embedding row between runs, and a checkpoint
    trained under one ordering would be meaningless under another.
    """
    rows = spark.read.parquet(str(path)).select(column).distinct().collect()
    return sorted(str(r[column]) for r in rows if r[column] is not None)
