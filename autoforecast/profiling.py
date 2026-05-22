"""Measure a supplied panel through an explicit forecast origin.

Eligible zero share, scale range, intermittency, exposure, and feature coverage
help resolve model requirements from the caller's data without future labels."""

from __future__ import annotations

import datetime as dt
import pathlib
from dataclasses import dataclass, field

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from spec.forecast_spec import ELIGIBLE_STATES

__all__ = ["Profile", "ProfileError", "profile"]


class ProfileError(ValueError):
    """A panel was profiled in a way that would produce a misleading number."""


@dataclass(frozen=True)
class Profile:
    """What a panel looks like, and what each number implies.

    Attributes:
        n_cells: Rows in the panel.
        n_series: Distinct series.
        n_days: Distinct dates.
        as_of: Inclusive last date used for every measurement in this profile.
        eligible_share: Share of cells the exposure contract admits.
        zero_rate_naive: Share of ALL cells that are zero.
        zero_rate_eligible: Share of ELIGIBLE cells that are zero. **This is the one that
            matters** -- the naive figure counts never-ranged cells and so measures assortment.
        scale_range: Ratio of the 99th to the 1st percentile of per-series mean demand.
        median_positive_size: Median quantity on days that sold, across the panel.
        column_coverage: Column -> share non-null, for the columns that decide which branches
            a client can support.
        notes: What these numbers imply, in words, so a reader does not have to remember the
            thresholds.
    """

    n_cells: int
    n_series: int
    n_days: int
    as_of: str
    eligible_share: float
    zero_rate_naive: float
    zero_rate_eligible: float
    scale_range: float
    median_positive_size: float
    min_eligible_history: int = 0
    max_eligible_history: int = 0
    column_coverage: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{self.n_cells:,} cells · {self.n_series:,} series × {self.n_days:,} days "
                 f"(through {self.as_of})",
                 f"  eligible          {self.eligible_share:.1%}",
                 f"  zero rate         {self.zero_rate_naive:.1%} naive → "
                 f"{self.zero_rate_eligible:.1%} on eligible cells",
                 f"  scale range       {self.scale_range:.0f}x across series",
                 f"  median positive   {self.median_positive_size:.1f} units",
                 f"  eligible history  {self.min_eligible_history}..{self.max_eligible_history} "
                 "observations per series"]
        for col, share in sorted(self.column_coverage.items()):
            lines.append(f"  {col:17} {share:.1%} populated")
        lines += [f"  → {n}" for n in self.notes]
        return "\n".join(lines)

    def as_record(self) -> dict[str, object]:
        return {"n_cells": self.n_cells, "n_series": self.n_series, "n_days": self.n_days,
                "as_of": self.as_of,
                "eligible_share": self.eligible_share, "zero_rate_naive": self.zero_rate_naive,
                "zero_rate_eligible": self.zero_rate_eligible, "scale_range": self.scale_range,
                "median_positive_size": self.median_positive_size,
                "min_eligible_history": self.min_eligible_history,
                "max_eligible_history": self.max_eligible_history,
                "column_coverage": self.column_coverage, "notes": self.notes}


def _implications(eligible_share: float, zero_eligible: float, scale_range: float,
                  coverage: dict[str, float]) -> list[str]:
    """Turn the measurements into the decisions they support.

    Each threshold is stated with the reason rather than as a magic number, because the RULE is
    what generalises to another estate -- the number was derived from one.
    """
    notes = []
    if eligible_share < 0.95:
        notes.append(
            f"{1 - eligible_share:.0%} of cells are ineligible, so the exposure contract is doing "
            "real work here -- training on the dense grid would teach the occurrence head an "
            "assortment decision")
    else:
        notes.append(
            "almost every cell is eligible, so the exposure contract changes little on this "
            "panel. That is a property of the data, not a reason to drop it")
    if zero_eligible > 0.30:
        notes.append(
            f"{zero_eligible:.0%} of eligible cells are zero: genuine intermittency, and a hurdle "
            "head is justified. A single count likelihood would spend its capacity on the zeros")
    else:
        notes.append(
            f"only {zero_eligible:.0%} of eligible cells are zero, so the hurdle's occurrence "
            "component has little to separate. A plain count head may be enough here")
    if scale_range > 100:
        notes.append(
            f"{scale_range:.0f}x scale range: RevIN is load-bearing, because attention is a dot "
            "product and the fast movers would otherwise dominate the softmax and the gradient")
    else:
        notes.append(f"{scale_range:.0f}x scale range: per-series normalisation matters less here")
    thin = [c for c, s in coverage.items() if s < 0.5]
    if thin:
        notes.append(
            f"thinly populated: {thin}. Per-CHANNEL masking is what keeps these usable -- a "
            "row-level mask would drop cells that have the other channels")
    return notes


def profile(
    panel_path: str | pathlib.Path,
    spark: SparkSession,
    *,
    series_col: str,
    time_col: str,
    target_col: str,
    as_of: str,
    exposure_col: str | None = None,
    coverage_cols: tuple[str, ...] = (),
) -> Profile:
    """Measure a panel and report what its shape implies.

    Args:
        panel_path: Parquet source.
        spark: Active session.
        series_col: Series identifier.
        time_col: Date column.
        target_col: Quantity.
        as_of: Inclusive ISO date cutoff. Rows after this date are excluded from every
            measurement, so a model decision cannot depend on information unavailable at its
            forecast origin.
        exposure_col: Observation-state column. When absent, every cell counts as eligible and the
            two zero rates coincide -- which is itself the finding, and the notes say so rather
            than reporting a single number as if eligibility had been considered.
        coverage_cols: Columns whose population share decides which branches are supported.

    Returns:
        A `Profile`.

    Raises:
        ProfileError: the cutoff is not an ISO date, the panel has no rows on or before it, or a
            named column is absent.
    """
    if not isinstance(as_of, str):
        raise ProfileError("as_of must be an ISO date string (YYYY-MM-DD)")
    try:
        cutoff = dt.date.fromisoformat(as_of)
    except ValueError:
        raise ProfileError(f"as_of must be an ISO date string (YYYY-MM-DD), got {as_of!r}") \
            from None
    # Canonicalise the recorded boundary even if ``date.fromisoformat`` learns to accept another
    # spelling in a future Python release.
    as_of = cutoff.isoformat()

    df = spark.read.parquet(str(panel_path))
    needed = [c for c in (series_col, time_col, target_col, exposure_col, *coverage_cols) if c]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ProfileError(f"columns absent from the panel: {missing}")

    # Compare calendar dates, not timestamps at midnight: every observation ON ``as_of`` belongs
    # to the information set, including a timestamp later that day. This filter must precede both
    # aggregate scans below; bounding only the zero-rate scan would still let future scale data
    # decide whether normalisation is load-bearing.
    df = df.filter(F.to_date(F.col(time_col)) <= F.lit(as_of).cast("date"))

    eligible = (F.col(exposure_col).isin(*sorted(ELIGIBLE_STATES)) if exposure_col
                else F.lit(True))
    agg = df.agg(
        F.count(F.lit(1)).alias("n"),
        F.countDistinct(series_col).alias("n_series"),
        F.countDistinct(time_col).alias("n_days"),
        F.avg(eligible.cast("double")).alias("eligible_share"),
        F.avg((F.col(target_col) == 0).cast("double")).alias("zero_naive"),
        F.avg(F.when(eligible, (F.col(target_col) == 0).cast("double"))).alias("zero_eligible"),
        F.expr(f"percentile_approx(CASE WHEN {target_col} > 0 THEN {target_col} END, 0.5)"
               ).alias("median_positive"),
        *[F.avg(F.col(c).isNotNull().cast("double")).alias(f"cov_{c}") for c in coverage_cols],
    ).collect()[0]

    if not agg["n"]:
        raise ProfileError(
            f"{panel_path} has no rows on or before {as_of}; there is nothing to profile")

    # Scale range from the 1st and 99th percentile of PER-SERIES mean demand, not from the
    # extremes: one miskeyed row would otherwise set the whole figure, and the number is used to
    # decide whether RevIN is load-bearing.
    per_series = (df.filter(eligible).groupBy(series_col)
                  .agg(F.avg(target_col).alias("m")).filter(F.col("m") > 0))
    bounds = per_series.agg(
        F.expr("percentile_approx(m, 0.01)").alias("lo"),
        F.expr("percentile_approx(m, 0.99)").alias("hi")).collect()[0]
    scale_range = float(bounds["hi"] / bounds["lo"]) if bounds["lo"] else float("inf")
    history_bounds = (df.filter(eligible).groupBy(series_col).count().agg(
        F.min("count").alias("minimum"), F.max("count").alias("maximum")).collect()[0])

    coverage = {c: float(agg[f"cov_{c}"] or 0.0) for c in coverage_cols}
    notes = _implications(float(agg["eligible_share"]), float(agg["zero_eligible"] or 0.0),
                          scale_range, coverage)
    if exposure_col is None:
        notes.insert(0, "no exposure column, so the two zero rates are the same number by "
                        "construction -- eligibility has not been considered on this panel")
    return Profile(
        n_cells=int(agg["n"]), n_series=int(agg["n_series"]), n_days=int(agg["n_days"]),
        as_of=as_of,
        eligible_share=float(agg["eligible_share"]),
        zero_rate_naive=float(agg["zero_naive"]),
        zero_rate_eligible=float(agg["zero_eligible"] or 0.0),
        scale_range=scale_range,
        median_positive_size=float(agg["median_positive"] or 0.0),
        min_eligible_history=int(history_bounds["minimum"] or 0),
        max_eligible_history=int(history_bounds["maximum"] or 0),
        column_coverage=coverage, notes=notes)
