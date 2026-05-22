"""The four zeros: mapping a client's raw columns onto the observation contract.

THE PRINCIPLE THE WHOLE DATA CONTRACT RESTS ON: **cleansing must never silently create or destroy
a zero.** A zero in a retail warehouse means four different things, and three of them are not
"demand was zero":

| the zero means | state | what the likelihood does with it |
|---|---|---|
| sold nothing | `exact` | trains on it -- this is real evidence of low demand |
| stocked out | `lower_bound` | keeps the occurrence, RIGHT-CENSORS the magnitude |
| never ranged | `unobserved` | excluded entirely |
| no data | `unobserved` | excluded, and the feed is flagged |

THE MODEL IS NEVER TOLD ABOUT STOCK. "Out of stock" is a REASON; what the model needs is the
LIKELIHOOD CONTRACT -- how the recorded number relates to the quantity being predicted. Separating
them is what lets one model serve clients whose stock feeds range from excellent to unusable, and
it is why this module emits a state rather than a stock level.

THE FEED GATE IS NOT OPTIONAL. A client recording positive sales while its own stock feed says
zero stock is not necessarily broken -- that is the signature of a same-day sellout -- but at a
high enough rate it means the feed cannot support the censoring contract. Measure the rate and
choose a threshold appropriate to the source before using its stock evidence.
"""

from __future__ import annotations

import pathlib

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

__all__ = ["ObservationError", "FeedQuality", "ObservationPolicy", "apply_policy",
           "assign", "feed_quality", "oos_expression", "state_expression"]


class ObservationError(ValueError):
    """A client's columns cannot support the observation contract as configured."""


class FeedQuality:
    """The result of the feed gate, with the numbers behind it.

    Attributes:
        sold_with_no_stock: Share of rows recording positive sales while stock was zero.
        threshold: The gate that was applied.
        passed: Whether the censoring contract may be used for this client.
        n_rows: Rows the share was computed over -- a share from a thousand rows is not the same
            evidence as a share from a billion, and the gate result should say which it was.
    """

    def __init__(self, sold_with_no_stock: float, threshold: float, n_rows: int) -> None:
        self.sold_with_no_stock = sold_with_no_stock
        self.threshold = threshold
        self.n_rows = n_rows
        self.passed = sold_with_no_stock <= threshold

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "QUARANTINE"
        return (f"[{verdict}] sold-with-no-stock {self.sold_with_no_stock:.2%} against a "
                f"{self.threshold:.0%} gate, over {self.n_rows:,} rows")

    def as_record(self) -> dict[str, object]:
        return {"sold_with_no_stock": self.sold_with_no_stock, "threshold": self.threshold,
                "n_rows": self.n_rows, "passed": self.passed}


def feed_quality(
    df: DataFrame, *, target_col: str, stock_col: str, threshold: float = 0.10,
) -> FeedQuality:
    """Measure whether a stock feed can support the censoring contract.

    Args:
        df: Rows to measure.
        target_col: Quantity sold.
        stock_col: End-of-day stock on hand.
        threshold: Share above which the feed is quarantined. The default 0.10 is a starting
            point; validate and register a source-specific threshold before relying on it.

    Returns:
        A `FeedQuality`. **A default inherited from another estate is a starting point, not a
        finding** -- the rule is that the threshold sits above your own healthy range and below
        your own broken one, and the profiler measures both.
    """
    stats = df.agg(
        F.count(F.lit(1)).alias("n"),
        F.count(F.when((F.col(target_col) > 0) & (F.col(stock_col) <= 0), F.lit(1))).alias("bad"),
    ).collect()[0]
    n = int(stats["n"])
    share = (int(stats["bad"]) / n) if n else 0.0
    return FeedQuality(share, threshold, n)


def state_expression(
    *, target_col: str, ranged_col: str | None, stock_col: str | None,
    has_data_col: str | None = None,
) -> Column:
    """The `exact` / `lower_bound` / `unobserved` expression, in priority order.

    ORDER MATTERS AND IT IS NOT ARBITRARY:

      1. **no data** wins over everything. A row the feed did not deliver tells us nothing about
         ranging or stock either, and calling it "not ranged" would assert something we cannot see.
      2. **not ranged** wins over stock. A product that was never listed cannot be out of stock in
         any meaningful sense, and a warehouse will happily record zero stock for it -- which
         would turn every never-ranged cell into a censored observation and destroy the contract.
      3. **stocked out with a positive sale** is `lower_bound`: it sold, then the shelf emptied,
         so demand was AT LEAST what sold.
      4. **stocked out with no sale** is `unobserved`, not `lower_bound`. "Demand >= 0" is
         vacuous, and the occurrence is unknown too -- there was nothing to buy all day.
      5. everything else is `exact`.
    """
    if has_data_col is not None:
        no_data = ~F.col(has_data_col)
    else:
        no_data = F.lit(False)
    not_ranged = F.lit(False) if ranged_col is None else ~F.col(ranged_col)
    out_of_stock = F.lit(False) if stock_col is None else (F.col(stock_col) <= 0)
    sold = F.col(target_col) > 0
    return (F.when(no_data, F.lit("unobserved"))
            .when(not_ranged, F.lit("unobserved"))
            .when(out_of_stock & sold, F.lit("lower_bound"))
            .when(out_of_stock & ~sold, F.lit("unobserved"))
            .otherwise(F.lit("exact")))


def oos_expression(*, ranged_col: str | None, stock_col: str | None,
                   has_data_col: str | None = None) -> Column:
    """Boolean shelf-state feature, separate from likelihood observation state.

    A ranged, delivered cell with stock <= 0 is OOS even when it sold zero. That zero-sale case is
    likelihood-unobserved because its demand lower bound is vacuous, but erasing the physical
    state makes it indistinguishable from a never-ranged or missing day in the encoder.
    """
    if stock_col is None:
        return F.lit(False)
    delivered = F.lit(True) if has_data_col is None else F.col(has_data_col)
    ranged = F.lit(True) if ranged_col is None else F.col(ranged_col)
    return delivered & ranged & (F.col(stock_col) <= 0)


class ObservationPolicy:
    """ONE atomic decision about a stock feed: censoring and stock-derived ranging, together.

    `ObservationSpec.on_gate_failure` has always specified this orchestration -- "a tripped gate
    must disable BOTH censoring inference AND any range inference that depends on the same stock
    feed, otherwise the broken signal re-enters through the exposure table" -- and until this
    object existed no code executed it: `assign` hard-raised regardless of the declared policy,
    so 'disable', the spec's own DEFAULT, was unreachable. The two switches live on one object
    precisely so no caller can disable censoring while leaving stock-derived ranging on.

    Attributes:
        censoring_enabled: Whether `lower_bound` states may be derived from this feed.
        stock_range_inference_enabled: Whether ranging derived FROM THE STOCK FEED may be used.
            Ranging from an independent source (an assortment table) is unaffected -- the gate
            measured the stock feed, not the assortment system.
        action: The policy that produced this decision.
        reason: One sentence a run log can print.
        quality: The measurement behind the decision, or None when no stock feed exists.
    """

    def __init__(self, *, censoring_enabled: bool, stock_range_inference_enabled: bool,
                 action: str, reason: str, quality: FeedQuality | None) -> None:
        self.censoring_enabled = censoring_enabled
        self.stock_range_inference_enabled = stock_range_inference_enabled
        self.action = action
        self.reason = reason
        self.quality = quality

    def __str__(self) -> str:
        return (f"[{self.action}] censoring={'on' if self.censoring_enabled else 'OFF'}, "
                f"stock-derived ranging="
                f"{'on' if self.stock_range_inference_enabled else 'OFF'} -- {self.reason}")

    def as_record(self) -> dict[str, object]:
        return {"action": self.action, "censoring_enabled": self.censoring_enabled,
                "stock_range_inference_enabled": self.stock_range_inference_enabled,
                "reason": self.reason,
                "quality": None if self.quality is None else self.quality.as_record()}


def apply_policy(quality: FeedQuality | None, *,
                 on_gate_failure: str = "disable") -> ObservationPolicy:
    """Turn a feed measurement into the atomic decision. Pure, so it is testable without Spark.

    Args:
        quality: The gate measurement, or None when the client has no stock feed at all --
            which is not a failure; censoring is simply not available.
        on_gate_failure: `ObservationSpec.on_gate_failure`. 'disable' keeps forecasting without
            censoring (the spec's default: a broken stock feed is a DATA problem, not a
            modelling one); 'raise' stops the run; 'infer' is REFUSED -- the spec calls it the
            most dangerous option because it produces a plausible number from a feed already
            known to disagree with itself, and refusing it loudly beats implementing it badly.

    Raises:
        ObservationError: the gate tripped under 'raise', or 'infer' was requested, or the
            policy name is not one the spec defines.
    """
    if on_gate_failure not in ("raise", "disable", "infer"):
        raise ObservationError(
            f"unknown on_gate_failure policy {on_gate_failure!r}; the spec defines raise, "
            "disable and infer, and a policy it does not define must not be given a default")
    if quality is None:
        return ObservationPolicy(
            censoring_enabled=False, stock_range_inference_enabled=False,
            action="no_stock_feed", quality=None,
            reason="no stock feed: censoring and stock-derived ranging are not available, "
                   "which is a property of the data rather than a failure")
    if quality.passed:
        return ObservationPolicy(
            censoring_enabled=True, stock_range_inference_enabled=True,
            action="pass", quality=quality, reason=str(quality))
    if on_gate_failure == "raise":
        raise ObservationError(
            f"feed gate failed: {quality}. on_gate_failure='raise' stops the run, which is "
            "right in a batch where silent degradation would go unnoticed")
    if on_gate_failure == "infer":
        raise ObservationError(
            "on_gate_failure='infer' is declared by the spec and deliberately NOT implemented: "
            "recovering states probabilistically from a feed already known to disagree with "
            "itself produces a plausible number with nothing behind it. Refused loudly rather "
            "than guessed quietly; use 'disable' to keep forecasting without censoring")
    return ObservationPolicy(
        censoring_enabled=False, stock_range_inference_enabled=False,
        action="disable", quality=quality,
        reason=f"feed gate failed ({quality}); forecasting continues WITHOUT censoring, and "
               "ranging derived from the same stock feed is disabled with it -- the broken "
               "signal must not re-enter through the exposure table")


def assign(
    panel_path: str | pathlib.Path,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    target_col: str,
    ranged_col: str | None = None,
    stock_col: str | None = None,
    has_data_col: str | None = None,
    state_col: str = "observation_state",
    oos_col: str | None = "is_oos",
    feed_threshold: float = 0.10,
    on_gate_failure: str = "disable",
    ranged_from_stock: bool = False,
) -> tuple[pathlib.Path, FeedQuality | None, ObservationPolicy]:
    """Write the panel with an observation-state column. Returns `(path, feed_quality)`.

    Args:
        panel_path: Parquet source.
        spark: Active session.
        out_dir: Parquet destination.
        target_col: Quantity sold.
        ranged_col: Boolean "this product was listed in this store on this day". Never-ranged
            cells describe assortment decisions and must not become observed demand zeros.
        stock_col: End-of-day stock. None disables censoring entirely, which is correct for a
            client with no stock feed and is not a degraded mode -- most public benchmarks are
            like this.
        has_data_col: Boolean "the feed delivered this row". None means every row was delivered.
        state_col: Output column name.
        oos_col: Output boolean shelf-state feature name, or None not to write it. Keep this
            separate from `state_col`: zero-sale OOS is unobserved in the likelihood but True in
            the encoder feature.
        feed_threshold: Passed to `feed_quality`.

    Args (policy):
        on_gate_failure: `ObservationSpec.on_gate_failure`, EXECUTED here rather than described.
            An earlier version hard-raised regardless of the declared policy, so 'disable' --
            the spec's own default -- was unreachable and a client with a broken stock feed
            could not be forecast at all.
        ranged_from_stock: Whether `ranged_col` is DERIVED from this same stock feed. When the
            gate trips under 'disable', such ranging is switched off ATOMICALLY with censoring;
            ranging from an independent assortment source is left alone, because the gate
            measured the stock feed and asserts nothing about the assortment system.

    Returns:
        `(path, feed_quality, policy)`. The policy is returned even on a pass, so the caller's
        run record can state the decision that was taken rather than implying one.

    Raises:
        ObservationError: the gate tripped under 'raise', or 'infer' was requested.
    """
    df = spark.read.parquet(str(panel_path))
    missing = [c for c in (target_col, ranged_col, stock_col, has_data_col)
               if c is not None and c not in df.columns]
    if missing:
        raise ObservationError(f"columns absent from the panel: {missing}")

    quality = None
    if stock_col is not None:
        quality = feed_quality(df, target_col=target_col, stock_col=stock_col,
                               threshold=feed_threshold)
    policy = apply_policy(quality, on_gate_failure=on_gate_failure)

    effective_stock = stock_col if policy.censoring_enabled else None
    effective_ranged = ranged_col
    if ranged_from_stock and not policy.stock_range_inference_enabled:
        effective_ranged = None
    out = df.withColumn(state_col, state_expression(
        target_col=target_col, ranged_col=effective_ranged, stock_col=effective_stock,
        has_data_col=has_data_col))
    if oos_col is not None:
        # The shelf-state FEATURE follows the same decision: an OOS flag derived from a feed
        # the gate just quarantined is the broken signal wearing feature clothes.
        out = out.withColumn(oos_col, oos_expression(
            ranged_col=effective_ranged, stock_col=effective_stock, has_data_col=has_data_col))
    out.write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir)), quality, policy
