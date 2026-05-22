"""Tests for the data rules, including the failure they exist to prevent."""
from datetime import date, timedelta

from forecast.data.reference import run_cleansing, run_quality

TODAY, ORIGIN = date(2026, 8, 21), date(2026, 8, 20)
def mk(sid, d, qty, **kw): return {"series_id": sid, "date": d, "qty": qty, **kw}
def cells(rows): return {(r["series_id"], r["date"]) for r in rows}


def test_duplicate_rows_are_rejected_not_summed():
    rows = [mk("a", ORIGIN, 5), mk("a", ORIGIN, 5)]
    rep = run_quality(rows, origin=ORIGIN, today=TODAY)
    assert rep.rejected and any(r.rule == "grain" for r in rep.results)


def test_null_target_is_rejected_not_zero_filled():
    rows = [mk("a", ORIGIN, None)]
    rep = run_quality(rows, origin=ORIGIN, today=TODAY)
    assert rep.rejected


def test_empty_feed_is_rejected_as_no_data():
    """An empty source used to report LOAD ACCEPTED, erasing the fourth zero kind entirely."""
    rep = run_quality([], origin=ORIGIN, today=TODAY)
    assert rep.rejected
    assert any(r.rule == "schema" and "empty feed" in r.detail for r in rep.results)


def test_future_dated_actuals_are_rejected():
    rows = [mk("a", ORIGIN + timedelta(days=3), 5)]
    rep = run_quality(rows, origin=ORIGIN, today=TODAY)
    assert any(r.rule == "future_dates" and r.severity == "reject" for r in rep.results)


def test_stale_feed_is_caught_though_every_row_is_valid():
    """A stale feed must fail even when all rows are individually valid."""
    old = date(2026, 2, 13)
    rows = [mk("a", old - timedelta(days=i), 3) for i in range(20)]
    rep = run_quality(rows, origin=ORIGIN, today=TODAY)
    assert any(r.rule == "freshness" and r.severity == "reject" for r in rep.results)
    print(f"    stale feed caught: {[str(r) for r in rep.results if r.rule=='freshness'][0][:64]}...")


def test_broken_stock_feed_is_rejected():
    rows = [mk("a", ORIGIN - timedelta(days=i), 5, stock=0) for i in range(50)]
    rep = run_quality(rows, origin=ORIGIN, today=TODAY)
    assert any(r.rule == "feed_plausibility" for r in rep.results)


def test_same_day_sellout_alone_does_not_trip_the_feed_gate():
    """Positive sales at zero end-of-day stock is normal; only a HIGH rate is a fault."""
    rows = [mk("a", ORIGIN - timedelta(days=i), 5, stock=(0 if i < 2 else 9)) for i in range(50)]
    rep = run_quality(rows, origin=ORIGIN, today=TODAY)
    assert not any(r.rule == "feed_plausibility" for r in rep.results)


def test_densify_never_invents_zeros_outside_the_eligible_window():
    """Unstocked cells must never be treated as observed demand zeros."""
    rows = [mk("a", date(2026, 8, 10), 4)]
    eligible = {("a", date(2026, 8, d)) for d in range(8, 13)}   # 5 daily cells
    out, rep = run_cleansing(rows, eligible=eligible)
    assert len(out) == 5, "exactly the eligible window, not the full calendar"
    assert all(r["date"] >= date(2026, 8, 8) and r["date"] <= date(2026, 8, 12) for r in out)
    print(f"    densify: 1 row -> {len(out)} inside a 5-day window, none outside")


def test_the_four_zeros_stay_distinct():
    rows = [
        mk("a", date(2026, 8, 10), 0, stock=7),    # sold nothing -> exact
        mk("a", date(2026, 8, 11), 3, stock=0),    # stocked out after sale -> lower_bound
        mk("a", date(2026, 8, 12), 0, stock=0),    # never ranged -> unobserved
        mk("a", date(2026, 8, 13), 0),             # no stock/feed evidence -> unobserved
    ]
    eligible = {("a", date(2026, 8, 10)), ("a", date(2026, 8, 11)),
                ("a", date(2026, 8, 13))}
    out, rep = run_cleansing(rows, eligible=eligible)
    got = [r["observation"] for r in sorted(out, key=lambda x: x["date"])]
    assert got == ["exact", "lower_bound", "unobserved", "unobserved"], got
    assert next(r for r in out if r["date"].day == 12)["eligible"] is False
    assert "stock" not in next(r for r in out if r["date"].day == 13)
    print(f"    four-zeros: {got}")


def test_imputed_prices_are_always_flagged():
    rows = [mk("a", date(2026, 8, 1) + timedelta(days=i), 2,
               price=(1.5 if i == 0 else None)) for i in range(4)]
    out, _ = run_cleansing(rows, eligible=cells(rows))
    imputed = [r for r in out if r.get("price_imputed")]
    assert len(imputed) == 3 and all(r["price"] == 1.5 for r in imputed)


def test_outliers_are_flagged_not_removed():
    rows = [mk("a", date(2026, 8, 1) + timedelta(days=i), 2.0) for i in range(20)]
    rows.append(mk("a", date(2026, 9, 1), 500.0))
    out, _ = run_cleansing(rows, eligible=cells(rows))
    assert len(out) == 21, "nothing removed"
    assert sum(1 for r in out if r.get("outlier")) == 1


def test_returns_are_floored_flagged_and_not_trained_as_demand_zeros():
    rows = [mk("a", ORIGIN, -4)]
    out, _ = run_cleansing(rows, eligible=cells(rows))
    assert out[0]["qty"] == 0.0 and out[0]["returned"] is True
    assert out[0]["observation"] == "unobserved"


def test_healthy_load_passes_and_reports():
    rows = [mk("a", TODAY - timedelta(days=i), float(i % 4), stock=5, price=2.0)
            for i in range(30)]
    rep = run_quality(rows, origin=TODAY, today=TODAY)
    assert not rep.rejected
    out, crep = run_cleansing(rows, eligible=cells(rows))
    print("\n" + str(crep) + "\n")


def test_densified_zeros_are_trainable_with_same_cell_stock_evidence():
    """Daily stock evidence preserves genuine intermittent zeros without guessing across days."""
    rows = [mk("a", date(2026, 8, 1), 3.0, stock=9),
            mk("a", date(2026, 8, 5), 2.0, stock=7)]
    eligible = {("a", date(2026, 8, d)) for d in range(1, 11)}
    stock_by_cell = {key: 5.0 for key in eligible - cells(rows)}
    out, _ = run_cleansing(rows, eligible=eligible, stock_by_cell=stock_by_cell,
                           complete_sales_cells=eligible)
    train = [r for r in out if r["observation"] != "unobserved"]
    assert len(train) == 10, "a series with a stock feed must train on its whole eligible window"
    assert sum(1 for r in train if r["qty"] == 0) == 8, "the zeros ARE the intermittency"


def test_a_tripped_feed_gate_refuses_to_fabricate_observations():
    rows = [mk("a", date(2026, 8, 1), 3.0, stock=9)]
    eligible = {("a", date(2026, 8, d)) for d in range(1, 6)}
    out, _ = run_cleansing(rows, eligible=eligible,
                           stock_by_cell={key: 5.0 for key in eligible - cells(rows)},
                           complete_sales_cells=eligible, feed_ok=False)
    assert sum(1 for r in out if r["observation"] == "unobserved") == 4


def test_a_positive_sale_is_never_unobserved():
    """Units left the shelf. Availability is unknown only for how much MORE could have sold."""
    rows = [mk("b", date(2026, 8, 1), 3.0)]
    out, _ = run_cleansing(rows, eligible=cells(rows))
    assert out[0]["observation"] == "lower_bound"


def test_one_stock_reading_does_not_make_every_missing_day_an_exact_zero():
    """The old per-series guard converted outages/stockouts into zero demand across the interval."""
    rows = [mk("a", date(2026, 8, 1), 3.0, stock=9)]
    eligible = {("a", date(2026, 8, d)) for d in range(1, 4)}
    out, _ = run_cleansing(rows, eligible=eligible)
    gaps = [r for r in out if r.get("densified")]
    assert len(gaps) == 2
    assert all(r["observation"] == "unobserved" for r in gaps)


def test_live_stock_feed_does_not_hide_a_missing_sales_feed():
    """Positive daily inventory alone cannot prove that an absent sale row means zero sales."""
    rows = [mk("a", date(2026, 8, 1), 3.0, stock=9)]
    missing = ("a", date(2026, 8, 2))
    out, _ = run_cleansing(rows, eligible=cells(rows) | {missing},
                           stock_by_cell={missing: 5.0}, complete_sales_cells=set())
    added = next(r for r in out if r.get("densified"))
    assert added["source_complete"] is False
    assert added["observation"] == "unobserved"


def test_daily_eligibility_preserves_an_internal_range_gap():
    """A single start/end interval used to manufacture demand zeros through closures/delistings."""
    rows = [mk("a", date(2026, 8, 1), 2.0, stock=5),
            mk("a", date(2026, 8, 3), 2.0, stock=5)]
    out, _ = run_cleansing(rows, eligible=cells(rows))
    assert {r["date"] for r in out} == {date(2026, 8, 1), date(2026, 8, 3)}


def test_zero_sales_at_zero_stock_does_not_claim_a_known_occurrence():
    rows = [mk("a", date(2026, 8, 1), 0.0, stock=0)]
    out, _ = run_cleansing(rows, eligible=cells(rows))
    assert out[0]["observation"] == "unobserved"


# --------------------------------------------------------------- A1: each rule named directly
# These reach each rule FUNCTION rather than the pipeline. A composition test proves the pipeline
# refuses a bad row; it does not prove WHICH rule refused it, so a rule that silently stopped
# working could be masked by another catching the same row.

def _rep():
    from forecast.data.reference import Report
    return Report()


def test_q_schema_names_the_missing_mandatory_column():
    from forecast.data.reference import q_schema
    rep = _rep()
    q_schema([{"series_id": "a", "date": ORIGIN}], rep)      # no `qty`
    assert rep.rejected and "qty" in str(rep.results[0])


def test_q_grain_refuses_rather_than_sums_duplicates():
    from forecast.data.reference import q_grain
    rep = _rep()
    q_grain([mk("a", ORIGIN, 5), mk("a", ORIGIN, 5)], rep)
    assert rep.rejected
    assert "double load" in str(rep.results[0]), "the message must name why summing is wrong"


def test_q_target_null_separates_unknown_from_zero():
    from forecast.data.reference import q_target_null
    rep = _rep()
    q_target_null([mk("a", ORIGIN, None)], rep)
    assert rep.rejected


def test_q_target_negative_flags_returns_without_rejecting():
    from forecast.data.reference import q_target_negative
    rep = _rep()
    q_target_negative([mk("a", ORIGIN, -4)], rep)
    assert not rep.rejected and rep.results[0].severity == "flag"


def test_q_future_dates_refuses_actuals_after_the_origin():
    from forecast.data.reference import q_future_dates
    rep = _rep()
    q_future_dates([mk("a", ORIGIN + timedelta(days=3), 5)], rep, ORIGIN)
    assert rep.rejected


def test_q_freshness_measures_staleness_in_days():
    from forecast.data.reference import q_freshness
    rep = _rep()
    old = date(2026, 2, 13)
    q_freshness([mk("a", old, 3)], rep, TODAY)
    assert rep.rejected and rep.results[0].n_affected == (TODAY - old).days


def test_q_feed_plausibility_allows_a_sellout_and_refuses_a_broken_feed():
    from forecast.data.reference import q_feed_plausibility
    ok = _rep()
    q_feed_plausibility([mk("a", ORIGIN, 5, stock=(0 if i < 2 else 9)) for i in range(50)], ok)
    assert not ok.rejected
    bad = _rep()
    q_feed_plausibility([mk("a", ORIGIN, 5, stock=0) for _ in range(50)], bad)
    assert bad.rejected


def test_q_calendar_gaps_flags_a_run_of_absent_dates():
    from forecast.data.reference import q_calendar_gaps
    rep = _rep()
    q_calendar_gaps([mk("a", date(2026, 8, 1), 1), mk("a", date(2026, 8, 20), 1)], rep)
    assert rep.results and rep.results[0].severity == "flag"


def test_c_densify_adds_only_inside_the_eligible_cell_set():
    """Eligibility is the exact set of daily (series, date) cells -- a coarse start/end interval
    cannot represent closures, delistings or feed outages inside it."""
    from forecast.data.reference import c_densify
    cells = {("a", date(2026, 8, 8) + timedelta(days=i)) for i in range(5)}
    out = c_densify([mk("a", date(2026, 8, 10), 4, stock=5)], cells, _rep(),
                    stock_by_cell={c: 5 for c in cells}, complete_sales_cells=cells)
    assert len(out) == 5, "one row per eligible cell, and nothing outside the set"


def test_c_mark_observation_separates_the_four_zeros():
    from forecast.data.reference import c_mark_observation
    rows = [mk("a", date(2026, 8, 10), 0, stock=7, eligible=True),
            mk("a", date(2026, 8, 11), 3, stock=0, eligible=True),
            mk("a", date(2026, 8, 12), 0, stock=0, eligible=False)]
    out = c_mark_observation(rows, _rep())
    assert [r["observation"] for r in out] == ["exact", "lower_bound", "unobserved"]


def test_c_impute_price_always_flags_what_it_filled():
    from forecast.data.reference import c_impute_price
    rows = [mk("a", date(2026, 8, 1) + timedelta(days=i), 2,
               price=(1.5 if i == 0 else None)) for i in range(3)]
    out = c_impute_price(rows, _rep())
    assert sum(1 for r in out if r.get("price_imputed")) == 2


def test_c_clip_negative_floors_and_flags_rather_than_deleting():
    from forecast.data.reference import c_clip_negative
    out = c_clip_negative([mk("a", ORIGIN, -4)], _rep())
    assert out[0]["qty"] == 0.0 and out[0]["returned"] is True


def test_c_flag_outliers_marks_and_keeps():
    from forecast.data.reference import c_flag_outliers
    rows = [mk("a", date(2026, 8, 1) + timedelta(days=i), 2.0) for i in range(20)]
    rows.append(mk("a", date(2026, 9, 1), 500.0))
    out = c_flag_outliers(rows, _rep())
    assert len(out) == 21 and sum(1 for r in out if r.get("outlier")) == 1


if __name__ == "__main__":
    import sys
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
