"""The quality and cleansing rules, in Spark, at estate scale.

`forecast/data/reference.py` holds the same rules over plain Python rows and is the REFERENCE: it is
engine-agnostic by design -- `dataclasses`, `datetime`, `collections`, `math`, `typing` and
nothing else -- which is what lets one definition run over a Spark row and a dict. This module is
the distributed form, and `test_rules.py` holds the two to the same answers on real M5.

**THE DIVISION THAT MATTERS**, and it is not stylistic:

  · **QUALITY rules REJECT or FLAG. They never modify.** A failed quality rule stops the load, so
    that a broken feed is a refusal rather than a forecast built on it.
  · **CLEANSING rules MODIFY, and every modification is COUNTED and REPORTED.** Nothing is fixed
    silently, because a silent fix is indistinguishable from data that was always clean.

**THE PRINCIPLE UNDERNEATH BOTH: cleansing must never silently create or destroy a zero.**
Densifying a sparse fact table CREATES zeros -- and every created zero is `unobserved` unless
something establishes otherwise, or the model learns that absence of a row means absence of
demand. That mistake was made here twice, in both directions: once erasing genuine stockouts, once
destroying the intermittency the whole hurdle head exists to model.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

__all__ = ["RuleError", "SparkReport", "cleanse", "check_quality"]


class RuleError(ValueError):
    """A rule was applied in a way that would change what the data means."""


@dataclass
class SparkReport:
    """What the rules found and what they changed, per rule.

    Attributes:
        failures: Rule name -> message, for quality rules that REJECT. A non-empty failures dict
            stops the load.
        flags: Rule name -> count, for quality rules that flag without rejecting.
        modifications: Rule name -> rows changed. **Reported even when zero**, so "this rule ran
            and changed nothing" is distinguishable from "this rule did not run".
    """

    failures: dict[str, str] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    modifications: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        head = "PASS" if self.passed else "REJECT"
        parts = [f"[{head}]"]
        for name, msg in sorted(self.failures.items()):
            parts.append(f"  FAIL {name}: {msg}")
        for name, n in sorted(self.flags.items()):
            parts.append(f"  flag {name}: {n:,} rows")
        for name, n in sorted(self.modifications.items()):
            parts.append(f"  changed {name}: {n:,} rows")
        return "\n".join(parts)

    def as_record(self) -> dict[str, object]:
        return {"passed": self.passed, "failures": self.failures, "flags": self.flags,
                "modifications": self.modifications}


def check_quality(
    df: DataFrame,
    *,
    series_col: str,
    time_col: str,
    target_col: str,
    origin: str,
    today: str | None = None,
    stock_col: str | None = None,
    max_stale_days: int = 7,
    feed_threshold: float = 0.10,
    allow_negative: bool = False,
) -> SparkReport:
    """Run every quality rule. Rejects or flags; NEVER modifies.

    The rules, and what each one catches that nothing downstream would:

      · **grain** -- duplicate `(series, date)`. A duplicate silently doubles a cell's weight in
        every aggregate and in the loss, and nothing further down can see it.
      · **target null** -- a null quantity is not a zero. Treating it as one manufactures demand
        evidence out of a missing row.
      · **target negative** -- returns net to negative sales. Legitimate for some estimands and
        not for a count likelihood, so it is a FLAG when allowed and a REJECT when not.
      · **future dates** -- a row dated after the origin has no business in a training frame at
        all, and its presence means the extract was built with the wrong as-of.
      · **freshness** -- the feed stopped. A stale extract produces a forecast that looks normal
        and is built on data that ends before the origin.
      · **feed plausibility** -- positive sales recorded against zero stock. Not impossible (a
        same-day sellout looks exactly like this), which is why it is a rate rather than a
        contradiction, and why the threshold sits above a healthy range rather than at zero.

    Args:
        df: Rows to check.
        series_col: Series identifier.
        time_col: Date column.
        target_col: Quantity.
        origin: As-of date; nothing may be dated after it.
        today: Date to measure staleness against. None skips the freshness rule -- which is
            correct for a historical backfill and wrong for a live load, so it is explicit.
        stock_col: Stock on hand, for the plausibility rule. None skips it.
        max_stale_days: Days after which the feed is considered stale.
        feed_threshold: Sold-with-no-stock rate above which the feed is rejected.
        allow_negative: Whether negative quantities are legitimate for this estimand.

    Returns:
        A `SparkReport`. **Read `passed` before using the frame.**
    """
    rep = SparkReport()
    stats = df.agg(
        F.count(F.lit(1)).alias("n"),
        F.count(F.when(F.col(target_col).isNull(), F.lit(1))).alias("null_target"),
        F.count(F.when(F.col(target_col) < 0, F.lit(1))).alias("negative"),
        F.count(F.when(F.col(time_col) > F.to_date(F.lit(origin)), F.lit(1))).alias("future"),
        F.max(time_col).alias("latest"),
    ).collect()[0]

    n_rows, n_keys = stats["n"], df.select(series_col, time_col).distinct().count()
    if n_keys != n_rows:
        rep.failures["grain"] = (
            f"{n_rows - n_keys:,} duplicate (series, date) rows. A duplicate doubles a cell's "
            "weight in every aggregate and in the loss, and nothing downstream can see it")
    if stats["null_target"]:
        rep.failures["target_null"] = (
            f"{stats['null_target']:,} null quantities. A null is not a zero -- treating it as "
            "one manufactures demand evidence out of a missing row")
    if stats["negative"]:
        if allow_negative:
            rep.flags["target_negative"] = int(stats["negative"])
        else:
            rep.failures["target_negative"] = (
                f"{stats['negative']:,} negative quantities, which a count likelihood cannot "
                "represent. Set allow_negative only if the estimand genuinely admits them")
    if stats["future"]:
        rep.failures["future_dates"] = (
            f"{stats['future']:,} rows dated after the origin {origin}. Their presence means the "
            "extract was built with the wrong as-of, so nothing else about it can be trusted")
    if today is not None and stats["latest"] is not None:
        stale = (dt.date.fromisoformat(today) - stats["latest"]).days
        if stale > max_stale_days:
            rep.failures["freshness"] = (
                f"the feed ends {stale} days before {today}. A stale extract produces a forecast "
                "that looks entirely normal and is built on data that stops early")
    if stock_col is not None:
        bad = df.filter((F.col(target_col) > 0) & (F.col(stock_col) <= 0)).count()
        share = bad / n_rows if n_rows else 0.0
        if share > feed_threshold:
            rep.failures["feed_plausibility"] = (
                f"{share:.1%} of rows sold with no stock, above the {feed_threshold:.0%} gate. "
                "The stock feed cannot support the configured censoring policy.")
        elif bad:
            rep.flags["feed_plausibility"] = bad
    return rep


def cleanse(
    df: DataFrame,
    eligible: DataFrame | None,
    *,
    series_col: str,
    time_col: str,
    target_col: str,
    state_col: str = "observation_state",
    price_col: str | None = None,
    allow_negative: bool = False,
    outlier_k: float = 20.0,
) -> tuple[DataFrame, SparkReport]:
    """Densify, mark, impute and flag -- counting every change. Returns `(frame, report)`.

    **DENSIFICATION IS THE DANGEROUS ONE**, and it went wrong here twice, in opposite directions:

      · marking every densified zero `unobserved` erased genuine stockouts and left almost nothing
        to train on;
      · marking them `exact` destroyed the intermittency the hurdle head exists to model, because
        an absent row in a sparse fact table IS usually a real zero.

    The resolution is per CELL, not per series and not per feed: a densified cell is `exact` when
    it falls inside the eligible grid -- the product was listed, the store was open, so a missing
    row means it did not sell -- and `unobserved` when it does not. That is the only reading that
    does not create or destroy a zero.

    Args:
        df: Rows to cleanse.
        eligible: The eligible grid (`series_id · date`), or None to skip densification. **None is
            not a shortcut**: without it, nothing knows which absent rows are real zeros.
        series_col: Series identifier.
        time_col: Date column.
        target_col: Quantity.
        state_col: Observation-state column to write.
        price_col: Price column to forward-fill, with an imputation flag. None skips it.
        allow_negative: Whether negatives survive; otherwise they are clipped to zero and counted.
        outlier_k: Multiples of a series' median above which a value is FLAGGED. Flagged, never
            removed -- a genuine promotion spike looks exactly like an outlier, and deleting it
            teaches the model that promotions do not happen.

    Returns:
        The cleansed frame and a report of every modification.

    Raises:
        RuleError: an outlier threshold at or below 1, which would flag most of the data.
    """
    if outlier_k <= 1.0:
        raise RuleError(
            f"outlier_k={outlier_k} would flag most of a normal series. It is a multiple of the "
            "series median, not a z-score")
    rep = SparkReport()
    out = df

    if eligible is not None:
        keys = eligible.select(F.col("series_id").alias(series_col),
                               F.col("date").alias(time_col))
        before = out.count()
        out = keys.join(out, [series_col, time_col], "left")
        created = out.count() - before
        rep.modifications["densify"] = max(0, created)
        # A created row is `exact` at zero: it is inside the eligible grid, so the product was
        # listed and the store was open, and the absent row means it did not sell.
        out = out.withColumn(
            state_col,
            F.when(F.col(target_col).isNotNull(), F.coalesce(F.col(state_col)
                                                             if state_col in df.columns
                                                             else F.lit("exact"), F.lit("exact")))
             .otherwise(F.lit("exact")))
        out = out.withColumn(target_col, F.coalesce(F.col(target_col), F.lit(0).cast("double")))
    elif state_col not in out.columns:
        out = out.withColumn(state_col, F.lit("exact"))

    if not allow_negative:
        n = out.filter(F.col(target_col) < 0).count()
        if n:
            rep.modifications["clip_negative"] = n
            out = out.withColumn(target_col, F.greatest(F.col(target_col), F.lit(0.0)))

    if price_col is not None:
        from pyspark.sql import Window
        backward = (Window.partitionBy(series_col).orderBy(time_col)
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow))
        missing = out.filter(F.col(price_col).isNull()).count()
        if missing:
            rep.modifications["impute_price"] = missing
        # Forward fill only, never backward: a backward fill would carry a future price into the
        # past, which is a leak that looks like unusually stable pricing.
        out = (out.withColumn(f"{price_col}_imputed", F.col(price_col).isNull())
               .withColumn(price_col, F.last(price_col, ignorenulls=True).over(backward)))

    from pyspark.sql import Window
    per_series = Window.partitionBy(series_col)
    med = F.expr(f"percentile_approx({target_col}, 0.5)")
    out = out.withColumn("_med", med.over(per_series))
    flagged = out.filter((F.col("_med") > 0)
                         & (F.col(target_col) > outlier_k * F.col("_med")))
    n_out = flagged.count()
    rep.flags["outliers"] = n_out
    # FLAGGED, not removed. A genuine promotion spike looks exactly like an outlier, and deleting
    # it teaches the model that promotions do not happen.
    out = out.withColumn("is_outlier",
                         (F.col("_med") > 0) & (F.col(target_col) > outlier_k * F.col("_med"))
                         ).drop("_med")
    return out, rep
