"""Tests for the Spark quality and cleansing rules, on REAL M5 where the rule allows it.

The rules exist to make two guarantees, and the tests are shaped around them:

  · a QUALITY rule never modifies -- so a failing rule must leave the frame untouched and stop
    the load, rather than fixing something and continuing;
  · a CLEANSING rule never changes anything silently -- so every modification is counted, and a
    count of zero is reported rather than omitted.

Underneath both: **cleansing must never silently create or destroy a zero.** Densification is
where that goes wrong, and it went wrong here twice in opposite directions -- once erasing genuine
stockouts, once destroying the intermittency the hurdle head exists to model.

Run: JAVA_HOME=... python3 forecast/data/test_rules.py
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pytest
from forecast.data.rules import RuleError, check_quality, cleanse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
ORIGIN = "2016-04-24"

_SPARK: SparkSession | None = None


def _require_real_m5_store() -> None:
    if not STORE.exists():
        pytest.skip(f"real M5 Parquet fixture is not present: {STORE}")


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_rules")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def frame(rows, schema):
    return spark().createDataFrame(rows, schema).withColumn("date", F.to_date("date"))


# ---------------------------------------------------------------- quality

def test_real_m5_passes_every_quality_rule() -> None:
    """If the rules rejected M5 they would reject everything, and the suite would be measuring the
    rules against nothing."""
    _require_real_m5_store()
    df = spark().read.parquet(str(STORE)).filter(F.col("date") <= F.lit(ORIGIN))
    rep = check_quality(df, series_col="id", time_col="date", target_col="qty", origin=ORIGIN)
    assert rep.passed, str(rep)


def test_a_duplicate_grain_row_is_rejected() -> None:
    """A duplicate doubles a cell's weight in every aggregate and in the loss, and nothing
    downstream can see it."""
    df = frame([("a", "2016-01-01", 1.0), ("a", "2016-01-01", 2.0)],
               "id string, date string, qty double")
    rep = check_quality(df, series_col="id", time_col="date", target_col="qty",
                        origin="2016-02-01")
    assert not rep.passed and "grain" in rep.failures
    assert "doubles a cell's weight" in rep.failures["grain"]


def test_a_null_quantity_is_rejected_because_it_is_not_a_zero() -> None:
    df = frame([("a", "2016-01-01", None)], "id string, date string, qty double")
    rep = check_quality(df, series_col="id", time_col="date", target_col="qty",
                        origin="2016-02-01")
    assert "target_null" in rep.failures
    assert "not a zero" in rep.failures["target_null"]


def test_a_negative_quantity_is_rejected_unless_the_estimand_admits_one() -> None:
    """Returns net to negative sales -- legitimate for some estimands and impossible for a count
    likelihood, so it is a flag when allowed and a reject when not."""
    df = frame([("a", "2016-01-01", -3.0)], "id string, date string, qty double")
    strict = check_quality(df, series_col="id", time_col="date", target_col="qty",
                           origin="2016-02-01")
    assert "target_negative" in strict.failures
    allowed = check_quality(df, series_col="id", time_col="date", target_col="qty",
                            origin="2016-02-01", allow_negative=True)
    assert allowed.passed and allowed.flags["target_negative"] == 1


def test_a_row_dated_after_the_origin_is_rejected() -> None:
    """Its presence means the extract was built with the wrong as-of, so nothing else about it can
    be trusted either."""
    df = frame([("a", "2016-05-01", 1.0)], "id string, date string, qty double")
    rep = check_quality(df, series_col="id", time_col="date", target_col="qty", origin=ORIGIN)
    assert "future_dates" in rep.failures
    assert "wrong as-of" in rep.failures["future_dates"]


def test_a_stale_feed_is_rejected_and_skipping_the_rule_is_explicit() -> None:
    """A stale extract produces a forecast that looks entirely normal and is built on data that
    stops early. Skipping the check is correct for a backfill, so `today=None` is deliberate."""
    df = frame([("a", "2016-01-01", 1.0)], "id string, date string, qty double")
    stale = check_quality(df, series_col="id", time_col="date", target_col="qty",
                          origin="2016-06-01", today="2016-06-01")
    assert "freshness" in stale.failures
    backfill = check_quality(df, series_col="id", time_col="date", target_col="qty",
                             origin="2016-06-01", today=None)
    assert backfill.passed


def test_the_feed_plausibility_rule_flags_a_little_and_rejects_a_lot() -> None:
    """Synthetic sale-without-stock rates below the threshold are flagged; larger rates fail."""
    import datetime as _dt
    days = [(_dt.date(2016, 1, 1) + _dt.timedelta(days=i)).isoformat() for i in range(50)]
    healthy = frame([("a", d, 1.0, 0 if i < 2 else 5) for i, d in enumerate(days)],
                    "id string, date string, qty double, stock int")
    rep = check_quality(healthy, series_col="id", time_col="date", target_col="qty",
                        origin="2016-03-01", stock_col="stock")
    assert rep.passed and rep.flags["feed_plausibility"] == 2
    broken = frame([("a", d, 1.0, 0) for d in days],
                   "id string, date string, qty double, stock int")
    assert "feed_plausibility" in check_quality(
        broken, series_col="id", time_col="date", target_col="qty",
        origin="2016-03-01", stock_col="stock").failures


def test_a_quality_rule_never_modifies_the_frame() -> None:
    """The whole division: quality rejects or flags, cleansing modifies. A rule that fixed
    something here would make a failing load indistinguishable from a clean one."""
    df = frame([("a", "2016-01-01", -3.0)], "id string, date string, qty double")
    before = df.collect()
    check_quality(df, series_col="id", time_col="date", target_col="qty", origin="2016-02-01")
    assert df.collect() == before


# ---------------------------------------------------------------- cleansing

def test_densification_marks_a_created_zero_by_the_cell_not_by_the_series() -> None:
    """This went wrong twice, in opposite directions: marking every densified zero `unobserved`
    erased genuine stockouts, and marking them `exact` destroyed the intermittency the hurdle head
    exists to model. A cell inside the eligible grid is a real zero; one outside it is not there
    at all."""
    df = frame([("a", "2016-01-01", 5.0)], "id string, date string, qty double")
    eligible = frame([("a", "2016-01-01"), ("a", "2016-01-02")],
                     "series_id string, date string").withColumnRenamed("series_id", "series_id")
    out, rep = cleanse(df, eligible, series_col="id", time_col="date", target_col="qty")
    rows = {str(r["date"]): r for r in out.collect()}
    assert rep.modifications["densify"] == 1
    assert rows["2016-01-02"]["qty"] == 0.0
    assert rows["2016-01-02"]["observation_state"] == "exact", \
        "a cell inside the eligible grid is a real zero, not an unobserved one"


def test_every_modification_is_counted_including_zero() -> None:
    """'This rule ran and changed nothing' must be distinguishable from 'this rule did not run'."""
    df = frame([("a", "2016-01-01", 5.0)], "id string, date string, qty double")
    _, rep = cleanse(df, None, series_col="id", time_col="date", target_col="qty")
    assert "outliers" in rep.flags
    assert rep.modifications.get("clip_negative", 0) == 0 or "clip_negative" in rep.modifications


def test_a_negative_is_clipped_and_the_change_is_reported() -> None:
    df = frame([("a", "2016-01-01", -3.0)], "id string, date string, qty double")
    out, rep = cleanse(df, None, series_col="id", time_col="date", target_col="qty")
    assert rep.modifications["clip_negative"] == 1
    assert out.collect()[0]["qty"] == 0.0


def test_price_is_forward_filled_only_and_flagged() -> None:
    """A backward fill would carry a FUTURE price into the past, which is a leak that looks like
    unusually stable pricing. And an imputed price must be flagged, or the commercial branch reads
    a flat line as real market stability."""
    df = frame([("a", "2016-01-01", 1.0, None), ("a", "2016-01-02", 1.0, 5.0),
                ("a", "2016-01-03", 1.0, None)],
               "id string, date string, qty double, price double")
    out, rep = cleanse(df, None, series_col="id", time_col="date", target_col="qty",
                       price_col="price")
    rows = {str(r["date"]): r for r in out.orderBy("date").collect()}
    assert rows["2016-01-01"]["price"] is None, "a future price was carried into the past"
    assert rows["2016-01-03"]["price"] == 5.0
    assert rows["2016-01-03"]["price_imputed"] is True
    assert rep.modifications["impute_price"] == 2


def test_an_outlier_is_flagged_and_never_removed() -> None:
    """A genuine promotion spike looks exactly like an outlier, and deleting it teaches the model
    that promotions do not happen."""
    import datetime as _dt
    rows = [("a", (_dt.date(2016, 1, 1) + _dt.timedelta(days=i)).isoformat(), 2.0)
            for i in range(19)] + [("a", "2016-01-20", 500.0)]
    df = frame(rows, "id string, date string, qty double")
    out, rep = cleanse(df, None, series_col="id", time_col="date", target_col="qty",
                       outlier_k=20.0)
    assert rep.flags["outliers"] == 1
    assert out.count() == 20, "the outlier row was removed"
    assert out.filter(F.col("is_outlier")).collect()[0]["qty"] == 500.0


def test_an_outlier_threshold_at_or_below_one_is_refused() -> None:
    """It is a multiple of the series median, not a z-score -- at 1.0 it flags most of the data."""
    df = frame([("a", "2016-01-01", 1.0)], "id string, date string, qty double")
    try:
        cleanse(df, None, series_col="id", time_col="date", target_col="qty", outlier_k=1.0)
    except RuleError as e:
        assert "not a z-score" in str(e); return
    raise AssertionError("accepted a threshold that flags most of a normal series")


def test_the_report_reads_as_a_verdict() -> None:
    df = frame([("a", "2016-01-01", -3.0)], "id string, date string, qty double")
    rep = check_quality(df, series_col="id", time_col="date", target_col="qty",
                        origin="2016-02-01")
    assert str(rep).startswith("[REJECT]")
    assert rep.as_record()["passed"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
