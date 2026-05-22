"""Executable demonstration that the deployed metric cannot see two whole classes of error.

Run: python3 autoforecast/test_score.py
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.score import Cell, Score, compare, naive_seasonal_mae, score, score_the_old_way


def C(store, product, day, forecast, actual, *, eligible=True, actual_state="observed",
      client="client", origin="2026-08-01"):
    """Small structural fixture; real-data findings must never use values from these tests."""
    return Cell(client, origin, store, product, day, forecast, actual, eligible, actual_state)


def test_aggregation_hides_offsetting_store_errors():
    """Ten stores, every one wrong by 5 units. Aggregated, they cancel to nothing."""
    cells = [C(f"s{i}", "p", "d", forecast=10 + (5 if i % 2 else -5), actual=10)
             for i in range(10)]
    old, new = score_the_old_way(cells), score(cells)
    assert old.mae == 0.0, "the deployed metric reports a PERFECT forecast"
    assert new.mae == 5.0, "every single store was wrong by 5"
    print(f"    offsetting errors:  old MAE={old.mae:.1f}  honest MAE={new.mae:.1f}")


def test_over_forecast_on_zero_days_is_invisible_to_the_inner_join():
    """SYNTHETIC — demonstrates the mechanism; the printed magnitude is NOT EVIDENCE.

    The failure that matters most: forecasting stock into shops that sold nothing. The cells are
    constructed so the deployed metric's inner join drops exactly the over-forecast days, which
    is a structural property of that metric and holds for any data. The +100% figure it prints is
    a consequence of the fixture, not a measurement of anything.
    """
    cells = ([C(f"s{i}", "p", "d1", forecast=8.0, actual=8.0) for i in range(5)]
             + [C(f"s{i}", "p", "d2", forecast=8.0, actual=None,
                  actual_state="sparse_zero") for i in range(5)])
    old, new = score_the_old_way(cells), score(cells)
    assert old.bias == 0.0, "the deployed metric sees no bias at all"
    assert new.bias == 4.0, "half the cells were forecast 8 and sold 0"
    print(f"    zero-day over-forecast:  old bias={old.bias:+.1f}  honest bias={new.bias:+.1f}"
          f"  ({new.bias_pct_of_volume:+.0%} of volume)")


def test_ineligible_cells_are_excluded_not_scored_as_errors():
    """A forecast for a product a shop does not stock is a range error, not a forecast error."""
    cells = ([C("s1", "p", "d", forecast=3.0, actual=3.0)]
             + [C("s2", "p", "d", forecast=3.0, actual=None, eligible=False,
                  actual_state="sparse_zero")])
    s = score(cells)
    assert s.n_scored == 1 and s.n_dropped_ineligible == 1
    assert s.mae == 0.0


def test_metric_is_defined_when_everything_is_zero():
    """36%-77% of cells are zero; a percentage error divides by zero on most of them.

    r19: this returned inf, which is honest per-cell and silently poisons any downstream mean or
    median. It is now None, with the absolute bias still reported and the raw sums retained so a
    caller can aggregate correctly.
    """
    cells = [C(f"s{i}", "p", "d", forecast=0.4, actual=0.0) for i in range(4)]
    s = score(cells)
    assert s.mae == 0.4 and s.bias == 0.4
    assert s.bias_pct_of_volume is None
    assert s.sum_actual == 0.0 and s.sum_error == 4 * 0.4


def test_mase_denominator_rejects_too_little_history():
    try:
        naive_seasonal_mae([1.0] * 5, season=7)
    except ValueError:
        return
    raise AssertionError("must refuse to build a seasonal baseline it has no data for")


def test_combined_realistic_case():
    """Synthetic demonstration of systematic over-forecasting on zero-sales days. The chosen values
    test the metric mechanism; their magnitude is not empirical evidence about demand.
    """
    cells = []
    for shop in range(20):
        for day in range(7):
            sells = (shop + day) % 3 != 0                     # ~two thirds of cells sell
            actual = 6.0 if sells else None
            forecast = (6.0 + (1.5 if shop % 2 else -1.5)) if sells else 5.0
            cells.append(C(f"s{shop}", "p", f"d{day}", forecast, actual,
                           actual_state="sparse_zero" if actual is None else "observed"))
    old, new = score_the_old_way(cells), score(cells)
    print("    realistic week:")
    print(f"      deployed metric : {old}")
    print(f"      honest metric   : {new}")
    gap = compare(old, new)
    assert (gap["honest_absolute_error_pct_of_volume"]
            > gap["deployed_absolute_error_pct_of_volume"])
    assert gap["honest_bias_pct_of_volume"] > gap["deployed_bias_pct_of_volume"]


def test_the_two_metrics_are_different_units_and_must_not_be_compared_by_mae():
    """r19: Gate 1 mandates reporting both, and their MAEs are not the same quantity.

    score_the_old_way measures error on a SUM ACROSS STORES, so its MAE grows with store count
    while the cell-grain MAE does not. Comparing them measures how many stores there are.
    """
    maes = {}
    for n in (5, 100):
        cells = [C(f"s{i}", "p", "d", forecast=10.0, actual=8.0) for i in range(n)]
        maes[n] = (score_the_old_way(cells).mae, score(cells).mae)
    assert maes[100][0] == 20 * maes[5][0], "aggregated MAE scales with store count"
    assert maes[100][1] == maes[5][1], "cell MAE does not"
    c = compare(score_the_old_way([C("s", "p", "d", 10.0, 8.0)]),
                score([C("s", "p", "d", 10.0, 8.0)]))
    assert "mae" not in " ".join(c).lower(), "compare() must not expose a cross-grain MAE ratio"


def test_combine_recovers_the_global_formula_across_partitions():
    """honest_metric must reduce under Spark; averaging per-partition MAEs is wrong."""
    a = [C(f"s{i}", "p", "d1", 10.0, 8.0) for i in range(3)]
    b = [C(f"s{i}", "p", "d2", 4.0, 9.0) for i in range(7)]
    assert math.isclose(Score.combine([score(a), score(b)]).mae, score(a + b).mae, rel_tol=1e-12)


def test_combine_keeps_audit_counts_from_zero_scored_partitions():
    """A partition may contain only ineligible/no-data rows; those counts are still evidence."""
    empty = Score(
        n_scored=0, n_dropped_ineligible=5, n_dropped_no_data=7, n_zero_filled=0,
        mae=0.0, rmse=0.0, bias=0.0, bias_pct_of_volume=None, mase=None,
        sum_abs_error=0.0, sum_squared_error=0.0, sum_error=0.0, sum_actual=0.0,
        sum_naive_abs_error=0.0,
    )
    populated = Score(
        n_scored=1, n_dropped_ineligible=2, n_dropped_no_data=3, n_zero_filled=0,
        mae=1.0, rmse=1.0, bias=1.0, bias_pct_of_volume=1.0, mase=None,
        sum_abs_error=1.0, sum_squared_error=1.0, sum_error=1.0, sum_actual=1.0,
        sum_naive_abs_error=0.0,
    )
    combined = Score.combine([empty, populated])
    assert combined.n_dropped_ineligible == 7
    assert combined.n_dropped_no_data == 10
    assert combined.rmse == 1.0


def test_combine_refuses_to_mix_grains():
    a = [C("s", "p", "d", 10.0, 8.0)]
    try:
        Score.combine([score(a), score_the_old_way(a)])
    except ValueError:
        return
    raise AssertionError("combining a cell score with an aggregated one must raise")


def test_mase_survives_distributed_reduction():
    """r20: combine() dropped MASE, so a global MASE was unobtainable under Spark."""
    hist = [1.0, 0, 2, 0, 1, 0, 3, 1, 0, 2, 0, 1, 4, 0, 1, 0, 2, 1]
    nm = naive_seasonal_mae(hist)
    a = [C(f"s{i}", "p", "d1", 3.0, 1.0) for i in range(4)]
    b = [C(f"s{i}", "p", "d2", 1.0, 2.0) for i in range(6)]
    # Tolerance, not equality: reduction is exact in arithmetic but summation ORDER differs, so
    # the last bit can. The earlier MAE reduction test used == and passed by luck.
    assert math.isclose(Score.combine([score(a, nm), score(b, nm)]).mase,
                        score(a + b, nm).mase, rel_tol=1e-12)


def test_comparison_exposes_aggregation_cancellation_with_common_units():
    """Signed bias is zero on both sides; absolute error / volume still exposes the defect."""
    cells = [C("s1", "p", "d", 15.0, 10.0), C("s2", "p", "d", 5.0, 10.0)]
    result = compare(score_the_old_way(cells), score(cells))
    assert result["deployed_bias_pct_of_volume"] == 0.0
    assert result["honest_bias_pct_of_volume"] == 0.0
    assert result["deployed_absolute_error_pct_of_volume"] == 0.0
    assert result["honest_absolute_error_pct_of_volume"] == 0.5
    assert result["honest_sum_abs_error"] == 10.0


def test_old_and_honest_metrics_use_the_same_eligible_population():
    """Ineligible actual-backed cells used to manufacture a large but meaningless metric gap."""
    cells = [C("s1", "p", "d", 3.0, 3.0),
             C("s2", "p", "d", 101.0, 1.0, eligible=False)]
    old, new = score_the_old_way(cells), score(cells)
    result = compare(old, new)
    assert old.n_dropped_ineligible == new.n_dropped_ineligible == 1
    assert old.sum_actual == new.sum_actual == 3.0
    assert result["deployed_bias_pct_of_volume"] == result["honest_bias_pct_of_volume"] == 0.0


def test_overlapping_origins_never_cancel_each_other_in_the_old_metric():
    """The same target date can be forecast at several origins; each is a separate decision."""
    cells = [C("s", "p", "2026-08-10", 15.0, 10.0, origin="2026-08-01"),
             C("s", "p", "2026-08-10", 5.0, 10.0, origin="2026-08-02")]
    assert score_the_old_way(cells).mae == 5.0
    assert score(cells).mae == 5.0


def test_no_data_is_excluded_and_counted_not_zero_filled():
    """Eligibility is an assortment fact; it does not prove that the actuals feed arrived."""
    cells = [C("s1", "p", "d", 4.0, 4.0),
             C("s2", "p", "d", 0.0, None, actual_state="no_data")]
    result = score(cells)
    assert result.n_scored == 1
    assert result.n_dropped_no_data == 1
    assert result.n_zero_filled == 0


def test_missing_actual_requires_explicit_sparse_zero_or_no_data_provenance():
    try:
        score([C("s", "p", "d", 0.0, None)])
    except ValueError as e:
        assert "actual_state='observed'" in str(e)
        return
    raise AssertionError("an absent source row is not self-evidently a demand zero")


def test_mase_rejects_invalid_and_partial_denominator_coverage():
    cells = [C("s", "p", "d", 2.0, 1.0)]
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            score(cells, bad)
        except ValueError:
            continue
        raise AssertionError(f"invalid naive denominator was accepted: {bad}")
    try:
        Score.combine([score(cells, 1.0), score([C("s", "p", "d2", 2.0, 1.0)])])
    except ValueError as e:
        assert "partial MASE coverage" in str(e)
        return
    raise AssertionError("a global MASE cannot mix partitions with and without denominators")


def test_compare_rejects_reversed_arguments():
    cells = [C("s", "p", "d", 2.0, 1.0)]
    try:
        compare(score(cells), score_the_old_way(cells))
    except ValueError:
        return
    raise AssertionError("argument reversal must not silently relabel honest as deployed")


def test_duplicate_cells_are_rejected_instead_of_reweighted():
    cell = C("s", "p", "d", 2.0, 1.0)
    try:
        score([cell, cell])
    except ValueError as e:
        assert "duplicate" in str(e)
        return
    raise AssertionError("duplicating an easy cell silently changes the published number")


def test_finite_cells_that_overflow_the_metric_are_refused():
    """Finite inputs do not guarantee a representable aggregate."""
    cells = [C("s", str(i), "d", 1e308, 0.0) for i in range(2)]
    try:
        score(cells)
    except ValueError as exc:
        assert "finite" in str(exc)
        return
    raise AssertionError("published a non-finite score from finite cells")


def test_seasonal_naive_mean_stays_finite_when_the_sum_would_overflow() -> None:
    denominator = naive_seasonal_mae([0.0, 1e308, 0.0, 1e308], season=1)
    assert math.isfinite(denominator) and denominator == 1e308


def test_arbitrary_precision_integers_are_refused_as_outside_binary64() -> None:
    huge = 10**400  # valid JSON and a Python int, but not a representable metric value
    calls = (
        lambda: score([C("s", "p", "d", huge, 0.0)]),
        lambda: score([C("s", "p", "d", 0.0, huge)]),
        lambda: score([C("s", "p", "d", 1.0, 0.0)], naive_mae=huge),
        lambda: naive_seasonal_mae([0.0, huge], season=1),
        lambda: Score(
            n_scored=1, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
            mae=huge, rmse=0.0, bias=0.0, bias_pct_of_volume=None, mase=None,
            sum_abs_error=0.0, sum_squared_error=0.0, sum_error=0.0, sum_actual=0.0,
            sum_naive_abs_error=0.0,
        ),
    )
    for call in calls:
        try:
            call()
        except ValueError as exc:
            assert "finite" in str(exc) or "invalid non-negative" in str(exc), exc
            continue
        raise AssertionError("an arbitrary-precision integer entered a binary64 metric")


def test_score_refuses_ratios_that_disagree_with_their_raw_evidence():
    try:
        Score(n_scored=2, n_dropped_ineligible=0, n_dropped_no_data=0,
              n_zero_filled=0, mae=1.0, rmse=1.0, bias=0.0, bias_pct_of_volume=0.0, mase=None,
              sum_abs_error=100.0, sum_squared_error=2.0, sum_error=0.0, sum_actual=1.0,
              sum_naive_abs_error=0.0)
    except ValueError as exc:
        assert "disagrees" in str(exc)
        return
    raise AssertionError("accepted a displayed score inconsistent with its reducible raw sums")


def test_score_requires_every_raw_sufficient_statistic() -> None:
    """A missing SSE must not fabricate a zero RMSE for a non-zero error."""
    try:
        Score(
            n_scored=1, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
            mae=1.0, rmse=0.0, bias=1.0, bias_pct_of_volume=1.0, mase=None,
            sum_abs_error=1.0, sum_error=1.0, sum_actual=1.0,
            sum_naive_abs_error=0.0,
        )
    except TypeError as exc:
        assert "sum_squared_error" in str(exc)
        return
    raise AssertionError("Score fabricated sum_squared_error=0 for a caller that omitted it")


def test_score_refuses_raw_sums_no_residual_vector_can_have() -> None:
    """Ratio-vs-sum agreement is insufficient if the sums contradict one another."""
    impossible = (
        # For non-negative |e_i|, sum(e_i^2) cannot exceed sum(|e_i|)^2.
        dict(sae=1.0, sse=100.0, se=0.0),
        # Triangle inequality: |sum(e_i)| cannot exceed sum(|e_i|).
        dict(sae=1.0, sse=1.0, se=2.0),
        # Cauchy: sum(|e_i|)^2 cannot exceed n * sum(e_i^2).
        dict(sae=2.0, sse=1.0, se=0.0),
    )
    for case in impossible:
        try:
            Score(
                n_scored=2, n_dropped_ineligible=0, n_dropped_no_data=0,
                n_zero_filled=0, mae=case["sae"] / 2,
                rmse=(case["sse"] / 2) ** 0.5, bias=case["se"] / 2,
                bias_pct_of_volume=None, mase=None,
                sum_abs_error=case["sae"], sum_squared_error=case["sse"],
                sum_error=case["se"], sum_actual=0.0, sum_naive_abs_error=0.0,
            )
        except ValueError:
            continue
        raise AssertionError(f"accepted impossible score sufficient statistics: {case}")


def test_score_refuses_raw_sums_that_imply_negative_forecasts() -> None:
    """Residual norms can agree while the implied non-negative forecast domain does not."""
    try:
        Score(
            n_scored=1, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
            mae=2.0, rmse=2.0, bias=-2.0, bias_pct_of_volume=-2.0, mase=None,
            sum_abs_error=2.0, sum_squared_error=4.0, sum_error=-2.0, sum_actual=1.0,
            sum_naive_abs_error=0.0,
        )
    except ValueError as exc:
        assert "negative total forecast" in str(exc)
        return
    raise AssertionError("accepted raw sums whose implied total forecast is negative")


def test_negative_implied_forecast_tolerance_does_not_scale_with_estate_volume() -> None:
    """A relative tolerance must not turn 1e-12 of a huge volume into negative demand."""
    actual = 1e15
    error = -1e15 - 100.0
    try:
        Score(
            n_scored=1, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
            mae=abs(error), rmse=abs(error), bias=error,
            bias_pct_of_volume=error / actual, mase=1.0,
            sum_abs_error=abs(error), sum_squared_error=error * error,
            sum_error=error, sum_actual=actual, sum_naive_abs_error=abs(error),
        )
    except ValueError as exc:
        assert "negative total forecast" in str(exc)
        return
    raise AssertionError("accepted an implied -100 forecast behind a large-volume tolerance")


def test_zero_cell_score_cannot_claim_metrics_without_raw_evidence() -> None:
    try:
        Score(
            n_scored=0, n_dropped_ineligible=1, n_dropped_no_data=2, n_zero_filled=0,
            mae=999.0, rmse=999.0, bias=-999.0, bias_pct_of_volume=None, mase=None,
            sum_abs_error=0.0, sum_squared_error=0.0, sum_error=0.0, sum_actual=0.0,
            sum_naive_abs_error=0.0,
        )
    except ValueError as exc:
        assert "zero-cell score" in str(exc)
        return
    raise AssertionError("a zero-cell Score published metrics unsupported by any raw evidence")


def test_rmse_selects_the_mean_where_mae_selects_the_median() -> None:
    """WHY RMSE EXISTS ON Score. MAE is minimised by the conditional MEDIAN, squared error by the
    conditional MEAN. This model emits a mean and demand is zero-heavy and right-skewed, so the
    two functionals are far apart -- and ranking mean forecasts by MAE prefers whichever one sits
    nearer the median, which is selection pressure toward under-forecasting.

    The fixture is that skew made explicit: 8 zero days and 2 days of 10. The mean is 2.0, the
    median is 0. A forecast at the mean must WIN on RMSE and LOSE on MAE -- which is exactly why
    the ladder cannot be ranked by MAE."""
    actuals = [0.0] * 8 + [10.0, 10.0]
    at_mean = [C("s", "p", f"d{i}", forecast=2.0, actual=a)      # what the model emits
               for i, a in enumerate(actuals)]
    at_median = [C("s", "p", f"d{i}", forecast=0.0, actual=a)    # what MAE would rather have
                 for i, a in enumerate(actuals)]

    mean_score, median_score = score(at_mean), score(at_median)

    assert mean_score.rmse < median_score.rmse, "RMSE must prefer the mean forecast"
    assert median_score.mae < mean_score.mae, \
        "MAE must prefer the median forecast -- if this ever fails the whole argument is wrong"
    assert abs(mean_score.bias) < 1e-12 and median_score.bias < 0, \
        "the MAE-preferred forecast is the biased-low one"


def test_rmse_reduces_from_global_sufficient_statistics() -> None:
    """RMSE must survive distributed reduction through its SUM, never by averaging ratios --
    the same rule the other metrics already follow, and the reason sum_squared_error is stored."""
    cells = [C("s", "p", "d0", forecast=1.0, actual=0.0),
             C("s", "p", "d1", forecast=3.0, actual=1.0),
             C("s", "p", "d2", forecast=0.0, actual=4.0)]
    combined = Score.combine([score(cells[:2]), score(cells[2:])])
    whole = score(cells)
    assert abs(combined.rmse - whole.rmse) < 1e-12, "RMSE did not recover the global formula"
    assert abs(combined.sum_squared_error - whole.sum_squared_error) < 1e-12


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
