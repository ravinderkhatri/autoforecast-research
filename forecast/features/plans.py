"""Effective-dated plan records: what was KNOWN about the horizon, proven, not declared.

`known_future` has been a declaration -- the docstrings said "a planned price is legitimate
known-future when the client plans prices" and then read the horizon's values from the panel's
own future rows, which is backtest convenience wearing a plan's name: nothing proved the value
was known at the origin, and a price corrected after the fact would silently rewrite history.

A PLAN SOURCE fixes that with three columns the declaration never had:

  * `valid_date`  -- the day the value applies to;
  * `known_at`    -- when this record became known. **Filtering on `known_at <= origin` is the
    whole point**: a revision made after the origin does not exist for that origin's forecast,
    however much better it describes what happened;
  * `snapshot`    -- which extract the record came from, so two records for one day rank by
    recency of knowledge, not by file order.

Selection is AS-OF: for each (series, valid_date) in the horizon, the record with the greatest
`(known_at, snapshot)` among those with `known_at <= origin`. The SELECTED rows are content-
hashed with their selected knowledge rank (per-row sha256, sorted, hashed again -- computed
inside Spark, one digest on the driver) and the digest travels with the run record, so which
selected plan knowledge trained this model is a fact rather than a directory someone can rewrite.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

__all__ = ["PlanError", "PlanSelection", "build"]


class PlanError(ValueError):
    """A plan source cannot support the as-of selection as declared."""


@dataclass(frozen=True)
class PlanSelection:
    """The provenance of one as-of selection -- what the manifest binds.

    Attributes:
        path: Where the selected rows were written.
        source: The plan source they came from.
        origin: The knowledge cutoff applied.
        n_rows: Selected rows. Zero is legal (no plan covers this horizon) and visible.
        selection_hash: sha256 over the selected rows' canonical values and knowledge rank. Two
            runs selecting the same knowledge get the same hash; a different selected known_at,
            snapshot or role value gets a different one. An unselected post-origin revision does
            not move it, because that knowledge did not exist at this origin.
    """

    path: str
    source: str
    origin: str
    n_rows: int
    selection_hash: str

    def as_record(self) -> dict[str, object]:
        return {"source": self.source, "origin": self.origin, "n_rows": self.n_rows,
                "selection_hash": self.selection_hash}


def build(
    plan_path: str | pathlib.Path,
    origin: str,
    horizon: int,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    series_col: str,
    valid_date_col: str,
    known_at_col: str,
    snapshot_col: str | None = None,
    value_cols: dict[str, str],
    series_ids: list[str] | tuple[str, ...] | None = None,
) -> PlanSelection:
    """Select the plan as it was knowable at `origin` and write it. Returns the provenance.

    Args:
        plan_path: Parquet of effective-dated records.
        origin: Knowledge cutoff. A record with `known_at > origin` DOES NOT EXIST for this
            selection -- that is the leak this module exists to prevent.
        horizon: Days after the origin to select for.
        spark: Active session.
        out_dir: Parquet destination: `series_id · date · <role> ... · _known_at ·
            _snapshot`. The final two columns are the selected knowledge-rank evidence.
        series_col: Series identifier in the plan.
        valid_date_col: The day each record applies to.
        known_at_col: When the record became known.
        snapshot_col: Extract identifier for tie-breaking within one knowledge day; None means
            `known_at` alone ranks records.
        value_cols: `{output role: plan column}`, e.g. `{"price": "planned_price"}`. The output
            uses the ROLE names, so downstream layout does not depend on a client's naming.
        series_ids: Optional exact series scope. Sharded callers pass the series they can
            consume so selection, writing and hashing do not repeatedly process the full plan.

    Raises:
        PlanError: missing columns, an empty role map, or duplicate records that tie exactly
            (same series, day, known_at and snapshot with different values -- two truths), or
            an invalid series scope.
    """
    if not value_cols:
        raise PlanError("value_cols is empty; a plan source with no values selects nothing")
    df = spark.read.parquet(str(plan_path))
    needed = [series_col, valid_date_col, known_at_col,
              *([snapshot_col] if snapshot_col else []), *value_cols.values()]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise PlanError(f"plan source lacks column(s) {missing}")
    if series_ids is not None:
        if (not isinstance(series_ids, list | tuple)
                or any(not isinstance(sid, str) or not sid.strip() for sid in series_ids)
                or len(set(series_ids)) != len(series_ids)):
            raise PlanError("series_ids must be a unique list/tuple of non-blank strings")
        df = (df.filter(F.col(series_col).isin(*series_ids)) if series_ids
              else df.filter(F.lit(False)))

    lo = F.to_date(F.lit(origin))
    scoped = (df
              .withColumn("_valid", F.to_date(F.col(valid_date_col)))
              .withColumn("_known", F.to_date(F.col(known_at_col)))
              .filter((F.col("_valid") > lo)
                      & (F.col("_valid") <= F.date_add(lo, horizon))
                      & (F.col("_known") <= lo)))
    order = [F.col("_known").desc()]
    if snapshot_col:
        order.append(F.col(snapshot_col).desc())
    ranked = scoped.withColumn(
        "_rank", F.row_number().over(
            Window.partitionBy(F.col(series_col), F.col("_valid")).orderBy(*order)))
    # An exact tie with different values is two truths for one day; refuse rather than let
    # row_number's arbitrary tiebreak decide a price.
    ties = (scoped.groupBy(F.col(series_col), F.col("_valid"), F.col("_known"),
                           *([F.col(snapshot_col)] if snapshot_col else []))
            # countDistinct(value) drops null. Wrap each value in a non-null struct so null is a
            # real alternative: a null and 5.0 at the same knowledge rank are two truths too.
            .agg(*[F.countDistinct(F.struct(F.col(c).isNull(), F.col(c)))
                   .alias(f"_n_{role}")
                   for role, c in value_cols.items()])
            .filter(" or ".join(f"_n_{role} > 1" for role in value_cols)).limit(1).collect())
    if ties:
        raise PlanError(
            f"the plan holds contradictory records at identical knowledge "
            f"({dict(ties[0].asDict())}); two truths for one day cannot be ranked")
    chosen = ranked.filter(F.col("_rank") == 1)
    selected = (chosen
                .select(F.col(series_col).alias("series_id"),
                        F.col("_valid").alias("date"),
                        *[F.col(c).cast("double").alias(role)
                          for role, c in value_cols.items()],
                        F.col("_known").alias("_known_at"),
                        (F.col(snapshot_col).cast("string") if snapshot_col is not None
                         else F.lit(None).cast("string")).alias("_snapshot")))
    selected.write.mode("overwrite").parquet(str(out_dir))

    # Canonical JSON gives every field an explicit name and retains null fields. `concat_ws`
    # skipped nulls, so `(price=null, promo=5)` and `(price=5, promo=null)` had the same bytes.
    # Read the written selection rather than re-evaluating `chosen`: the source may be replaced
    # after the write action, and returned provenance must describe the exact persisted artifact.
    written = spark.read.parquet(str(out_dir))
    canonical = F.to_json(F.struct(
        F.col("series_id").cast("string").alias("series_id"),
        F.col("date").cast("string").alias("valid_date"),
        F.col("_known_at").cast("string").alias("known_at"),
        F.col("_snapshot").cast("string").alias("snapshot"),
        *[F.col(role).cast("double").alias(f"role:{role}")
          for role in sorted(value_cols)]),
        options={"ignoreNullFields": "false"})
    row_hash = F.sha2(canonical, 256)
    digest_row = (written.select(row_hash.alias("_h"))
                  .agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list("_h"))), 256)
                       .alias("_digest"),
                       F.count(F.lit(1)).alias("_n")).collect()[0])
    return PlanSelection(path=str(out_dir), source=str(plan_path), origin=origin,
                         n_rows=int(digest_row["_n"]),
                         selection_hash=str(digest_row["_digest"]))
