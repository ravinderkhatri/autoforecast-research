"""Bare-minimum data quality and cleansing rules for the demand forecaster.

THE ONE PRINCIPLE EVERYTHING ELSE FOLLOWS FROM
----------------------------------------------
**Cleansing must never silently create or destroy a zero.**

The whole architecture turns on what a zero means. A zero is one of four different facts:

    sold nothing          demand happened to be zero          -> train on it
    stocked out           demand was censored                 -> mask the magnitude
    never ranged          the shop does not stock this        -> exclude entirely
    no data               the feed failed                     -> exclude, and alert

Standard cleansing advice -- "fill missing values with zero" -- destroys that distinction in one
line, and it is the single most damaging thing that can be done to this data. Every rule below
exists to keep the four apart.

DIVISION OF LABOUR
------------------
QUALITY rules REJECT or FLAG. They never modify. A failed quality rule stops the load.
CLEANSING rules MODIFY, and every modification is counted and reported. Nothing is fixed silently.

Pure Python, no dependencies. Rows are dicts so this is testable without a warehouse.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

Row = dict
Key = tuple[str, date]


@dataclass
class RuleResult:
    rule: str
    severity: str                     # "reject" | "flag" | "fixed"
    n_affected: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity:6}] {self.rule:26} {self.n_affected:>10,}  {self.detail}"


@dataclass
class Report:
    results: list[RuleResult] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return any(r.severity == "reject" for r in self.results)

    def add(self, *a):
        self.results.append(RuleResult(*a))

    def __str__(self) -> str:
        head = "LOAD REJECTED" if self.rejected else "LOAD ACCEPTED"
        return "\n".join([head, "-" * 78] + [str(r) for r in self.results])


# =====================================================================================
# QUALITY -- reject or flag. Never modify.
# =====================================================================================

REQUIRED = ("series_id", "date", "qty")


def q_schema(rows: list[Row], rep: Report) -> None:
    """Q1. The three mandatory columns exist on every row. Nothing else is required."""
    if not rows:
        rep.add("schema", "reject", 0,
                "empty feed. No rows is 'no data', not a clean load with zero demand")
        return
    missing = [c for c in REQUIRED if any(c not in r for r in rows)]
    if missing:
        rep.add("schema", "reject", len(rows), f"missing mandatory column(s): {missing}")


def q_grain(rows: list[Row], rep: Report) -> None:
    """Q2. Exactly one row per (series, date).

    Duplicates are NOT summed automatically. A duplicate means either a double-loaded file or a
    finer grain than declared, and those need opposite treatment -- summing a double-loaded file
    doubles demand silently.
    """
    dup = [k for k, n in Counter((r["series_id"], r["date"]) for r in rows).items() if n > 1]
    if dup:
        rep.add("grain", "reject", len(dup),
                f"duplicate (series, date) keys, e.g. {dup[0]} -- double load, or grain is finer "
                "than declared. Do not sum blindly")


def q_target_null(rows: list[Row], rep: Report) -> None:
    """Q3. The target is never null. A null quantity is not a zero -- it is an unknown."""
    n = sum(1 for r in rows if r.get("qty") is None)
    if n:
        rep.add("target_null", "reject", n,
                "null quantity. A null is 'unknown', a zero is 'sold none' -- they must not be "
                "conflated, and only the loader knows which this is")


def q_target_negative(rows: list[Row], rep: Report) -> None:
    """Q4. Negative quantities can represent returns; flag them for an explicit policy."""
    n = sum(1 for r in rows if (r.get("qty") or 0) < 0)
    if n:
        rep.add("target_negative", "flag", n,
                "negative quantities (returns). These NaN a negative-binomial likelihood through "
                "log-gamma; set target_allows_negative or route them to their own series")


def q_future_dates(rows: list[Row], rep: Report, origin: date) -> None:
    """Q5. No actuals dated after the forecast origin. This is the leak that flatters everything."""
    n = sum(1 for r in rows if r["date"] > origin)
    if n:
        rep.add("future_dates", "reject", n,
                f"rows dated after the origin {origin}. Training on these leaks the future and "
                "makes every backtest optimistic")


def q_freshness(rows: list[Row], rep: Report, today: date, max_stale_days: int = 7) -> None:
    """Q6. The feed is current.

    A stopped feed can have valid individual files while providing no recent observations.
    No data arriving is not a failed load, so nothing else catches this.
    """
    if not rows:
        return
    last = max(r["date"] for r in rows)
    stale = (today - last).days
    if stale > max_stale_days:
        rep.add("freshness", "reject", stale,
                f"days stale -- last data {last}. Every file may have loaded correctly; a feed "
                "that stopped is invisible to per-file checks")


def q_feed_plausibility(rows: list[Row], rep: Report, threshold: float = 0.10) -> None:
    """Q7. Sales with no stock, which is a feed-agreement check rather than an impossibility.

    Positive sales at zero END-OF-DAY stock is a same-day sellout and is normal. A high rate means
    the stock and sales feeds may disagree. Calibrate the threshold against your own
    feed-quality evidence before using this check.
    """
    sellers = [r for r in rows if (r.get("qty") or 0) > 0 and "stock" in r]
    if not sellers:
        return
    bad = sum(1 for r in sellers if (r.get("stock") or 0) <= 0)
    rate = bad / len(sellers)
    if rate > threshold:
        rep.add("feed_plausibility", "reject", bad,
                f"{rate:.1%} of selling rows show no stock. The feeds disagree, so censoring "
                "CANNOT be inferred -- and range inference from the same feed is also void")


def q_calendar_gaps(rows: list[Row], rep: Report, max_gap_days: int = 3) -> None:
    """Q8. Flag runs of absent dates inside a series' active life.

    A gap is ambiguous by construction: it is either genuine zeros the source did not emit, or
    missing data. Densification (C1) resolves it, but only after somebody has been told.
    """
    by = {}
    for r in rows:
        by.setdefault(r["series_id"], []).append(r["date"])
    gappy = 0
    for _sid, ds in by.items():
        ds = sorted(ds)
        if any((b - a).days > max_gap_days for a, b in zip(ds, ds[1:], strict=False)):
            gappy += 1
    if gappy:
        rep.add("calendar_gaps", "flag", gappy,
                f"series with gaps over {max_gap_days} days -- either unemitted zeros or missing "
                "data, and the two need opposite handling")


# =====================================================================================
# CLEANSING -- modify, and report every modification.
# =====================================================================================

def c_densify(rows: list[Row], eligible: set[Key], rep: Report, feed_ok: bool = True,
              stock_by_cell: dict[Key, float | None] | None = None,
              complete_sales_cells: set[Key] | None = None) -> list[Row]:
    """C1. Emit rows for absent DAILY eligible cells, without guessing their observation state.

    This is the rule that most often gets written wrong. Densifying the FULL grid manufactures
    zeros for shop-product pairs that never stocked the item, so the occurrence head
    learns assortment rather than demand.

    A single `(start, end)` interval is also insufficient: range can stop and restart, a shop can
    close, and a feed can fail inside the interval. `eligible` is therefore the exact set of daily
    cells emitted by the exposure contract.

    An absent sales row becomes trainable only when `complete_sales_cells` proves that the sales
    source arrived for that cell/date AND `stock_by_cell[(series, date)]` supplies trusted positive
    availability. A live inventory feed does not prove the sales feed arrived, and evidence that
    the SERIES had stock on another day says nothing about this day. Without both daily facts the
    row is retained as `unobserved`.
    """
    if not isinstance(eligible, set) or any(
            not isinstance(k, tuple) or len(k) != 2 or not isinstance(k[0], str)
            or not isinstance(k[1], date) for k in eligible):
        raise ValueError("eligible must be the set of (series_id, date) cells from exposure_daily")
    stock_by_cell = stock_by_cell or {}
    complete_sales_cells = complete_sales_cells or set()
    have = {(r["series_id"], r["date"]) for r in rows}
    out = list(rows)
    for r in out:
        # The passed exposure grid is authoritative. A fail-open `eligible=True` default recreated
        # never-ranged demand whenever a caller forgot to attach the field to a source row.
        r["eligible"] = (r["series_id"], r["date"]) in eligible
    added = 0
    for sid, d in sorted(eligible):
        key = (sid, d)
        if key not in have:
            row = {"series_id": sid, "date": d, "qty": 0.0, "densified": True,
                   "eligible": True, "source_complete": key in complete_sales_cells}
            if feed_ok and key in complete_sales_cells and key in stock_by_cell:
                row["stock"] = stock_by_cell[key]
            out.append(row)
            added += 1
    if added:
        n_obs = sum(1 for r in out if r.get("densified") and r.get("stock") is not None
                    and r["stock"] > 0)
        rep.add("densify", "fixed", added,
                f"daily eligible cells added ({n_obs} with positive same-cell stock evidence -> "
                f"trainable zeros; {added - n_obs} without -> unobserved)")
    return out


def c_mark_observation(rows: list[Row], rep: Report) -> list[Row]:
    """C2. Label every row with which of the four zeros it is. Nothing is dropped.

    The likelihood reads this label:
      exact       -> full contribution
      lower_bound -> occurrence kept, magnitude masked (it sold, then ran out)
      unobserved  -> excluded entirely
    """
    counts = Counter()
    # NOTE: a row reaches 'exact' only with a POSITIVE stock reading. Absence of stock data is
    # never treated as evidence of availability.
    for r in rows:
        qty, stock = r.get("qty") or 0, r.get("stock")
        if not isinstance(r.get("eligible"), bool):
            raise ValueError("every row needs explicit daily exposure eligibility")
        if not r["eligible"]:
            r["observation"] = "unobserved"
        elif r.get("returned"):
            # Flooring a return to zero is a tensor-compatibility repair, not evidence that demand
            # was zero today. Keep the row and provenance, but never train the occurrence head on it.
            r["observation"] = "unobserved"
        elif stock is not None and stock <= 0:
            # A positive sale proves occurrence and end-of-day stock zero censors magnitude. With
            # zero sales, however, an empty shelf does NOT prove occurrence was zero; exclude it.
            r["observation"] = "lower_bound" if qty > 0 else "unobserved"
        elif stock is None and qty > 0:
            # A POSITIVE SALE is an observation whatever the stock feed says -- the units left the
            # shelf. Availability is unknown only for how much MORE could have sold, which is
            # exactly a lower bound. Marking this `unobserved` would discard a real occurrence.
            r["observation"] = "lower_bound"
        elif stock is None:
            # A zero on a day the feed reported, with no stock reading: genuinely unknown whether
            # demand was zero or the shelf was empty.
            r["observation"] = "unobserved"
        else:
            r["observation"] = "exact"
        counts[r["observation"]] += 1
    rep.add("mark_observation", "fixed", len(rows),
            " · ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))
    return rows


def c_impute_price(rows: list[Row], rep: Report) -> list[Row]:
    """C3. Forward-fill price within a series, and ALWAYS set a provenance flag.

    Imputed prices that are indistinguishable
    from measured ones read as genuine price stability, which destroys elasticity -- the model
    concludes price never moves and therefore never matters.
    """
    by = {}
    for r in rows:
        by.setdefault(r["series_id"], []).append(r)
    filled = unfilled = 0
    for series in by.values():
        last = None
        for r in sorted(series, key=lambda x: x["date"]):
            if r.get("price") is not None:
                last = r["price"]
                r["price_imputed"] = False
            elif last is not None:
                r["price"], r["price_imputed"] = last, True
                filled += 1
            else:
                # leading gap: no earlier price exists to carry forward. Leave price absent but
                # FLAGGED, and record it so the caller must decide -- torch cannot consume None,
                # so a silent pass-through here becomes a crash mid-training.
                r["price_imputed"], r["price_missing_leading"] = True, True
                unfilled += 1
    if unfilled:
        rep.add("price_missing_leading", "flag", unfilled,
                "rows before a series' first observed price -- nothing to carry forward. Flagged "
                "price_missing_leading; the caller must impute or exclude, because None cannot "
                "enter a tensor")
    if filled:
        rep.add("impute_price", "fixed", filled,
                "forward-filled, each flagged price_imputed -- unflagged imputation reads as "
                "price stability and destroys elasticity")
    return rows


def c_clip_negative(rows: list[Row], rep: Report, allow: bool = False) -> list[Row]:
    """C4. Floor returns at zero unless the target explicitly permits negatives."""
    if allow:
        return rows
    n = 0
    for r in rows:
        if (r.get("qty") or 0) < 0:
            r["qty"], r["returned"] = 0.0, True
            n += 1
    if n:
        rep.add("clip_negative", "fixed", n,
                "returns floored to 0 and flagged; they are NOT deleted, because a return is "
                "evidence a sale happened earlier")
    return rows


def c_flag_outliers(rows: list[Row], rep: Report, k: float = 20.0) -> list[Row]:
    """C5. Flag extreme values. Never remove them.

    A 50x spike is usually a real event -- a bulk order, a viral product, a heatwave. Removing it
    teaches the model that the largest demand it will ever face cannot happen. Flag it, let the
    model see it, and let the dispersion parameter absorb it.
    """
    by = {}
    for r in rows:
        by.setdefault(r["series_id"], []).append(r)
    n = 0
    for series in by.values():
        pos = [r["qty"] for r in series if (r.get("qty") or 0) > 0]
        if len(pos) < 10:
            continue
        med = sorted(pos)[len(pos) // 2]
        for r in series:
            if med > 0 and (r.get("qty") or 0) > k * med:
                r["outlier"] = True
                n += 1
    if n:
        rep.add("flag_outliers", "flag", n,
                f"values over {k:g}x the series median, FLAGGED not removed -- removing them "
                "teaches the model its largest real demand is impossible")
    return rows


# =====================================================================================

def run_quality(rows, *, origin, today, **kw) -> Report:
    rep = Report()
    q_schema(rows, rep)
    if rep.rejected:
        return rep
    q_grain(rows, rep)
    q_target_null(rows, rep)
    q_target_negative(rows, rep)
    q_future_dates(rows, rep, origin)
    q_freshness(rows, rep, today, kw.get("max_stale_days", 7))
    q_feed_plausibility(rows, rep)
    q_calendar_gaps(rows, rep)
    return rep


def run_cleansing(rows, *, eligible: set[Key], allow_negative=False,
                  feed_ok: bool = True,
                  stock_by_cell: dict[Key, float | None] | None = None,
                  complete_sales_cells: set[Key] | None = None) -> tuple[list[Row], Report]:
    """Clean observations using explicit eligibility and feed-quality evidence. Densified zeros
    from an untrusted stock feed cannot be treated as observed targets.
    """
    rep = Report()
    rows = c_clip_negative(rows, rep, allow=allow_negative)
    rows = c_densify(rows, eligible, rep, feed_ok=feed_ok, stock_by_cell=stock_by_cell,
                     complete_sales_cells=complete_sales_cells)
    rows = c_mark_observation(rows, rep)
    rows = c_impute_price(rows, rep)
    rows = c_flag_outliers(rows, rep)
    return rows, rep
