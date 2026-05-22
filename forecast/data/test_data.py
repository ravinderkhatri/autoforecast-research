"""Tests for the data contract, on REAL M5 through Spark.

Every check here targets a way the contract can be violated SILENTLY -- a zero created or
destroyed by cleansing, a ranging label that leaks the future into the past, a grid that the model
and the metric derive differently. None of these fail loudly at training time. They produce a
model that trains, converges, and has learned the wrong task.

Run: JAVA_HOME=... python3 forecast/data/test_data.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pytest
from forecast.data import exposure, observation
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"

_SPARK: SparkSession | None = None


def _require_real_m5_store() -> None:
    if not STORE.exists():
        pytest.skip(f"real M5 Parquet fixture is not present: {STORE}")


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_data")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


# ---------------------------------------------------------------- observation states

def test_the_state_priority_puts_no_data_above_everything() -> None:
    """A row the feed did not deliver tells us nothing about ranging or stock either. Calling it
    'not ranged' would assert something we cannot see."""
    df = spark().createDataFrame(
        [(5, False, 0, False), (5, True, 10, True)],
        "qty int, ranged boolean, stock int, delivered boolean")
    out = df.withColumn("s", observation.state_expression(
        target_col="qty", ranged_col="ranged", stock_col="stock", has_data_col="delivered"))
    states = [r["s"] for r in out.collect()]
    assert states == ["unobserved", "exact"]


def test_a_never_ranged_cell_is_not_treated_as_a_stockout() -> None:
    """A warehouse records zero stock for a product it never listed. Without ranging taking
    priority, every never-ranged cell becomes a censored observation and the contract collapses."""
    df = spark().createDataFrame([(0, False, 0)], "qty int, ranged boolean, stock int")
    out = df.withColumn("s", observation.state_expression(
        target_col="qty", ranged_col="ranged", stock_col="stock"))
    assert out.collect()[0]["s"] == "unobserved"


def test_a_stockout_that_sold_is_a_lower_bound_and_one_that_did_not_is_unobserved() -> None:
    """It sold, then the shelf emptied: demand was at least what sold. But out of stock all day
    with no sale is vacuous -- 'demand >= 0' says nothing and the occurrence is unknown too."""
    df = spark().createDataFrame([(3, True, 0), (0, True, 0)],
                                 "qty int, ranged boolean, stock int")
    out = df.withColumn("s", observation.state_expression(
        target_col="qty", ranged_col="ranged", stock_col="stock"))
    assert [r["s"] for r in out.collect()] == ["lower_bound", "unobserved"]


def test_zero_sale_stockout_remains_an_encoder_feature() -> None:
    df = spark().createDataFrame(
        [(3, True, 0, True), (0, True, 0, True), (0, False, 0, True),
         (0, True, 0, False)],
        "qty int, ranged boolean, stock int, delivered boolean")
    rows = (df.withColumn("state", observation.state_expression(
                target_col="qty", ranged_col="ranged", stock_col="stock",
                has_data_col="delivered"))
            .withColumn("oos", observation.oos_expression(
                ranged_col="ranged", stock_col="stock", has_data_col="delivered"))
            .collect())
    assert [r["state"] for r in rows] == [
        "lower_bound", "unobserved", "unobserved", "unobserved"]
    assert [r["oos"] for r in rows] == [True, True, False, False]


def test_a_client_with_no_stock_feed_gets_exact_states_not_a_degraded_mode() -> None:
    """Most public benchmarks have no stock data. That is the absence of censoring, not a
    failure -- and M5 is exactly this case."""
    df = spark().createDataFrame([(0, True), (4, True)], "qty int, ranged boolean")
    out = df.withColumn("s", observation.state_expression(
        target_col="qty", ranged_col="ranged", stock_col=None))
    assert [r["s"] for r in out.collect()] == ["exact", "exact"]


def test_the_feed_gate_measures_and_reports_rather_than_asserting() -> None:
    """Synthetic rates on either side of the configured threshold must produce distinct verdicts."""
    df = spark().createDataFrame([(5, 0)] * 80 + [(5, 10)] * 20, "qty int, stock int")
    broken = observation.feed_quality(df, target_col="qty", stock_col="stock")
    assert not broken.passed and broken.n_rows == 100
    assert "QUARANTINE" in str(broken)
    healthy = spark().createDataFrame([(5, 0)] * 5 + [(5, 10)] * 95, "qty int, stock int")
    assert observation.feed_quality(healthy, target_col="qty", stock_col="stock").passed


def test_apply_policy_is_atomic_and_refuses_infer() -> None:
    """apply_policy is the pure core of the gate decision: censoring and stock-derived ranging
    flip TOGETHER (a broken feed must not re-enter through the exposure table), 'infer' is
    refused as the spec's most dangerous option, and an undefined policy gets no default."""
    from forecast.data.observation import FeedQuality, ObservationError, apply_policy

    failed = FeedQuality(sold_with_no_stock=0.4, threshold=0.05, n_rows=1000)
    healthy = FeedQuality(sold_with_no_stock=0.01, threshold=0.05, n_rows=1000)
    assert not failed.passed and healthy.passed

    disabled = apply_policy(failed, on_gate_failure="disable")
    assert not disabled.censoring_enabled and not disabled.stock_range_inference_enabled, \
        "the atomic pair split: censoring and stock-derived ranging must flip together"
    passed = apply_policy(healthy, on_gate_failure="disable")
    assert passed.censoring_enabled and passed.stock_range_inference_enabled
    absent = apply_policy(None)
    assert absent.action == "no_stock_feed" and not absent.censoring_enabled

    for policy, needle in (("raise", "gate"), ("infer", "refus"), ("repair", "does not define")):
        try:
            apply_policy(failed, on_gate_failure=policy)
        except ObservationError as e:
            assert needle in str(e).lower(), (policy, str(e))
            continue
        raise AssertionError(f"policy {policy!r} did not raise on a failed feed")


def test_the_gate_policy_is_executed_not_described() -> None:
    """`on_gate_failure` was a documented field no code ran: `assign` hard-raised regardless,
    so 'disable' -- the spec's own default, "the model should still forecast that client, just
    without censoring correction" -- was unreachable. The decision is atomic: disabling
    censoring disables stock-derived ranging and the OOS feature with it, because the broken
    signal must not re-enter through the exposure table."""
    d = tempfile.mkdtemp()
    frame = spark().createDataFrame(
        [(5, 0, True)] * 90 + [(5, 9, True)] * 10, "qty int, stock int, is_ranged boolean")
    frame.write.mode("overwrite").parquet(d)

    # 'raise' stops the run, as before.
    try:
        observation.assign(d, spark(), tempfile.mkdtemp(), target_col="qty", stock_col="stock",
                           on_gate_failure="raise")
    except observation.ObservationError as e:
        assert "raise" in str(e)
    else:
        raise AssertionError("a tripped gate under 'raise' did not stop the run")

    # 'infer' is refused as unimplemented rather than guessed.
    try:
        observation.assign(d, spark(), tempfile.mkdtemp(), target_col="qty", stock_col="stock",
                           on_gate_failure="infer")
    except observation.ObservationError as e:
        assert "NOT implemented" in str(e)
    else:
        raise AssertionError("'infer' produced states from a feed known to disagree with itself")

    # 'disable' keeps forecasting: censoring off, stock-derived ranging off WITH it, OOS off.
    out, quality, policy = observation.assign(
        d, spark(), tempfile.mkdtemp(), target_col="qty", stock_col="stock",
        ranged_col="is_ranged", ranged_from_stock=True, on_gate_failure="disable")
    assert quality is not None and not quality.passed
    assert not policy.censoring_enabled and not policy.stock_range_inference_enabled
    rows = spark().read.parquet(str(out)).collect()
    assert all(r["observation_state"] == "exact" for r in rows), \
        "a disabled feed must not produce censored or stock-ranged states"
    assert all(not r["is_oos"] for r in rows), \
        "an OOS feature from a quarantined feed is the broken signal wearing feature clothes"

    # And a HEALTHY feed on the same path keeps everything on, with the decision recorded.
    healthy_dir = tempfile.mkdtemp()
    spark().createDataFrame([(5, 0, True)] * 3 + [(5, 9, True)] * 97,
                            "qty int, stock int, is_ranged boolean") \
        .write.mode("overwrite").parquet(healthy_dir)
    _, quality, policy = observation.assign(
        healthy_dir, spark(), tempfile.mkdtemp(), target_col="qty", stock_col="stock",
        ranged_col="is_ranged", ranged_from_stock=True)
    assert quality.passed and policy.censoring_enabled and policy.stock_range_inference_enabled


# ---------------------------------------------------------------- exposure

def test_the_first_sale_label_finds_m5s_leading_zeros() -> None:
    """20.9% of M5's grid, and the largest single piece of the problem."""
    _require_real_m5_store()
    df = spark().read.parquet(str(STORE))
    out = exposure.from_first_sale(df, series_col="id", time_col="date", target_col="qty")
    share = out.agg(F.avg(F.col("is_ranged").cast("double"))).collect()[0][0]
    assert 0.6 < share < 0.95, f"eligible share {share:.1%} is not M5's shape"


def test_a_status_history_is_forward_filled_and_never_backward() -> None:
    """THE leak this module is built around. A product ranged only in December must not be marked
    ranged in January -- those cells would carry information that did not exist in January."""
    status = spark().createDataFrame(
        [("a", "2016-01-01", "inactive"), ("a", "2016-06-01", "listed")],
        "sid string, d string, st string").withColumn("d", F.to_date("d"))
    out = exposure.from_status_history(status, series_col="sid", time_col="d",
                                       status_col="st", ranged_values=("listed",))
    rows = {str(r["d"]): r["is_ranged"] for r in out.collect()}
    assert rows["2016-01-01"] is False, "June's listing decided January's flag"
    assert rows["2016-06-01"] is True


def test_an_empty_ranged_value_set_is_refused() -> None:
    status = spark().createDataFrame([("a", "2016-01-01", "x")], "sid string, d string, st string")
    try:
        exposure.from_status_history(status, series_col="sid", time_col="d", status_col="st",
                                     ranged_values=())
    except exposure.ExposureError as e:
        assert "every cell would be excluded" in str(e); return
    raise AssertionError("marked everything unranged")


def test_an_absent_exposure_column_contributes_true_rather_than_excluding() -> None:
    """Asserting a cell was not ranged because the client has no ranging feed would discard real
    demand evidence on the strength of a column that does not exist."""
    df = spark().createDataFrame([(True,), (False,)], "open boolean")
    out = df.withColumn("e", exposure.eligible_expression(open_col="open", ranged_col=None))
    assert [r["e"] for r in out.collect()] == [True, False]
    assert [r["e"] for r in df.withColumn("e", exposure.eligible_expression()).collect()] \
        == [True, True]


def test_no_exposure_column_at_all_is_refused_unless_the_fallback_is_asked_for() -> None:
    """Marking every cell eligible would feed never-ranged cells into the loss as demand
    evidence, which is the failure the whole module exists to prevent."""
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        [("a", "2016-01-01", 1.0)], "id string, date string, qty double"
    ).write.mode("overwrite").parquet(src)
    try:
        exposure.apply(src, spark(), "/tmp/never", series_col="id", time_col="date",
                       target_col="qty")
    except exposure.ExposureError as e:
        assert "fall_back_to_first_sale=True" in str(e); return
    raise AssertionError("marked the whole cross-product eligible by default")


def test_the_fallback_produces_a_usable_eligibility_column_on_real_m5() -> None:
    _require_real_m5_store()
    out = spark().read.parquet(str(exposure.apply(
        STORE, spark(), tempfile.mkdtemp(), series_col="id", time_col="date", target_col="qty",
        fall_back_to_first_sale=True)))
    assert "is_eligible" in out.columns
    assert 0 < out.filter(~F.col("is_eligible")).count() < out.count()


def test_coverage_reports_the_share_that_reshaped_this_architecture() -> None:
    _require_real_m5_store()
    df = exposure.from_first_sale(spark().read.parquet(str(STORE)),
                                  series_col="id", time_col="date", target_col="qty")
    stats = exposure.coverage(df).collect()[0]
    assert 0.0 < stats["eligible_share"] < 1.0
    assert stats["n_cells"] > 5_000_000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
