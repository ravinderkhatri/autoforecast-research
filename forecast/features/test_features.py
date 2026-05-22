"""Tests for the feature builders, on REAL M5 through Spark.

What these check is the small number of ways a feature table can be wrong in a manner the model
cannot detect: an encoder window assembled in the wrong ORDER, a masked day that reads as
observed, an absent channel imputed to a plausible value, an event that was not yet announced,
and a vocabulary whose indices move between runs. Each one trains happily and is wrong.

Run: JAVA_HOME=... python3 forecast/features/test_features.py
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pytest
from forecast.features import calendar_events, commercial, demand_history, static_identity
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
ORIGIN = "2016-04-24"
ENCODER = 28
HORIZON = 14

_SPARK: SparkSession | None = None


def _require_real_m5_store() -> None:
    if not STORE.exists():
        pytest.skip(f"real M5 Parquet fixture is not present: {STORE}")


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_features")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def history(origin: str = ORIGIN, days: int = ENCODER) -> pathlib.Path:
    _require_real_m5_store()
    return demand_history.build(
        STORE, origin, days, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="date", target_col="qty",
        exposure_col="observation_state")


# ---------------------------------------------------------------- demand history

def test_the_encoder_window_is_in_calendar_order_not_task_order() -> None:
    """`collect_list` does not promise an order -- it is whatever order tasks finished in. A
    history assembled that way is a shuffled series, and the model trains on it happily."""
    out = spark().read.parquet(str(history()))
    row = out.limit(1).collect()[0]
    sid = row["series_id"]
    want = [r["qty"] for r in
            spark().read.parquet(str(STORE))
            .filter((F.col("id") == sid) & (F.col("date") <= F.lit(ORIGIN))
                    & (F.col("date") >= F.date_sub(F.to_date(F.lit(ORIGIN)), ENCODER - 1)))
            .orderBy("date").collect()]
    got = [v for v, obs in zip(row["history"], row["observed"], strict=True) if obs]
    want_observed = want[-len(row["history"]):]
    assert len(row["history"]) == ENCODER
    # Every observed value must match the calendar-ordered series at its own position.
    for i, (v, obs) in enumerate(zip(row["history"], row["observed"], strict=True)):
        if obs:
            assert abs(v - want_observed[i]) < 1e-9, f"position {i} is out of calendar order"
    assert got  # the series has at least one observed day, or this proves nothing


def test_a_masked_day_carries_a_value_but_is_flagged_unobserved() -> None:
    """The tensor is dense, so a masked day needs a number -- but nothing may read that number
    without reading the flag. A never-ranged day and a day that sold nothing are both zero in the
    warehouse and mean opposite things."""
    out = spark().read.parquet(str(history(origin="2011-03-01")))
    masked = out.select(F.explode(F.arrays_zip("history", "observed")).alias("z")) \
                .filter(~F.col("z.observed"))
    assert masked.count() > 0, "no masked days at an early origin; M5 is not what we think"
    assert masked.filter(F.col("z.history") != 0.0).count() == 0, \
        "a masked day carries a non-zero value, which something downstream will read as demand"


def test_the_state_codes_travel_beside_the_values() -> None:
    """`lower_bound` must survive the builder AS ITSELF. The trainer used to rebuild states from
    the observed boolean, which is how censoring became unreachable from training. An unknown
    state string maps to -1, so the likelihood refuses it loudly instead of the builder quietly
    defaulting a state the contract does not define."""
    import datetime as dt
    days = [("exact", 2.0, False), ("exact", 0.0, False),
            ("lower_bound", 3.0, True), ("unobserved", 0.0, True),
            ("unobserved", 0.0, False), ("not_a_state", 1.0, False)]
    rows = [("s1", dt.date(2016, 4, 19) + dt.timedelta(days=i), qty, st, oos)
            for i, (st, qty, oos) in enumerate(days)]
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        rows, "id string, date date, qty double, observation_state string, is_oos boolean") \
        .write.mode("overwrite").parquet(src)
    out = demand_history.build(src, "2016-04-24", 6, spark(), tempfile.mkdtemp(),
                               series_col="id", time_col="date", target_col="qty",
                               exposure_col="observation_state", oos_col="is_oos")
    row = spark().read.parquet(str(out)).collect()[0]
    assert row["state"] == [0, 0, 1, 2, 2, -1], f"codes drifted: {row['state']}"
    assert row["observed"] == [True, True, True, False, False, False]
    assert row["history"] == [2.0, 0.0, 3.0, 0.0, 0.0, 0.0], \
        "a censored day must keep its recorded floor; an ineligible day must carry zero"
    assert row["oos"] == [False, False, True, True, False, False], \
        "a zero-sale stockout was collapsed into ordinary unobserved time"


def test_a_series_with_too_short_a_window_is_dropped_not_padded() -> None:
    """Padding would put invented days at the START of the encoder and the mask would call them
    observed -- which is worse than having no window at all."""
    out = spark().read.parquet(str(history()))
    sizes = out.select(F.size("history").alias("n")).distinct().collect()
    assert [r["n"] for r in sizes] == [ENCODER], f"ragged windows present: {[r['n'] for r in sizes]}"


def test_duplicate_days_cannot_impersonate_a_contiguous_encoder_window() -> None:
    """Three rows are not a three-day history when two rows occupy Jan 1 and Jan 2 is absent."""
    import datetime as dt
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s1", dt.date(2024, 1, 1), 1.0),
         ("s1", dt.date(2024, 1, 1), 99.0),
         ("s1", dt.date(2024, 1, 3), 3.0)],
        "id string, date date, qty double",
    ).write.mode("overwrite").parquet(src)
    try:
        demand_history.build(
            src, "2024-01-03", 3, spark(), tempfile.mkdtemp(),
            series_col="id", time_col="date", target_col="qty")
    except demand_history.FeatureError as exc:
        assert "duplicate encoder key" in str(exc)
        return
    raise AssertionError("duplicate Jan 1 was silently packed into Jan 2's encoder position")


def test_an_interior_calendar_gap_is_dropped_even_when_endpoints_match() -> None:
    import datetime as dt
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s1", dt.date(2024, 1, 1), 1.0), ("s1", dt.date(2024, 1, 3), 3.0)],
        "id string, date date, qty double",
    ).write.mode("overwrite").parquet(src)
    out = demand_history.build(
        src, "2024-01-03", 3, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="date", target_col="qty")
    assert spark().read.parquet(str(out)).count() == 0


def test_summarise_reports_how_much_of_each_window_was_observed() -> None:
    """A window that is 90% masked produces a RevIN statistic from a handful of days, and the
    model has no way to say so."""
    stats = demand_history.summarise(history(), spark()).collect()[0]
    assert 0.0 < stats["mean_coverage"] <= 1.0
    assert stats["n_series"] == 3049


def test_a_non_positive_encoder_window_is_refused() -> None:
    try:
        demand_history.build(STORE, ORIGIN, 0, spark(), "/tmp/never", series_col="id",
                             time_col="date", target_col="qty")
    except demand_history.FeatureError as e:
        assert "encoder_days" in str(e); return
    raise AssertionError("built a zero-length encoder window")


# ---------------------------------------------------------------- commercial

def test_an_absent_price_is_flagged_unavailable_rather_than_imputed() -> None:
    """A flat-lined imputed price reads to the commercial branch as real market stability, and
    the elasticity learned from it is an artefact of the imputation."""
    _require_real_m5_store()
    out = spark().read.parquet(str(commercial.build(
        STORE, ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="date", price_col="sell_price")))
    assert out.filter(~F.col("price_available")).count() >= 0
    # Where price is unavailable the value is zero AND the flag is false -- never one without the
    # other, because zero is a legitimate price-derived value.
    assert out.filter((~F.col("price_available")) & (F.col("price") != 0.0)).count() == 0
    assert out.filter(F.col("promo_available")).count() == 0, \
        "no promo column was supplied, so the channel must be unavailable everywhere"


def test_discount_depth_is_relative_to_the_series_own_reference() -> None:
    """A raw price says nothing without a level to compare it to, and levels differ by three
    orders of magnitude across a catalogue."""
    _require_real_m5_store()
    out = spark().read.parquet(str(commercial.build(
        STORE, ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="date", price_col="sell_price")))
    depths = out.filter(F.col("discount_depth_available")).select("discount_depth")
    assert depths.count() > 0
    stats = depths.agg(F.min("discount_depth"), F.max("discount_depth")).collect()[0]
    assert -2.0 < stats[0] and stats[1] < 1.0, f"implausible depth range {stats}"


def test_discount_reference_is_frozen_at_the_forecast_origin() -> None:
    """Post-origin realised prices and the proposed plan must not rewrite the price baseline."""
    origin = "2024-01-03"
    plan = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s1", dt.date(2024, 1, 4), 9.0)],
        "series_id string, date date, price double",
    ).write.mode("overwrite").parquet(plan)

    def build_with_realised_future(realised_price: float):
        source = tempfile.mkdtemp()
        rows = [
            ("s1", dt.date(2024, 1, 1), 8.0),
            ("s1", dt.date(2024, 1, 2), 10.0),
            ("s1", dt.date(2024, 1, 3), 12.0),
            ("s1", dt.date(2024, 1, 4), realised_price),
        ]
        spark().createDataFrame(rows, "id string, date date, price double") \
            .write.mode("overwrite").parquet(source)
        path = commercial.build(
            source, origin, 1, spark(), tempfile.mkdtemp(),
            series_col="id", time_col="date", price_col="price",
            reference_days=3, plan_path=plan,
        )
        return spark().read.parquet(str(path)).collect()[0]

    first = build_with_realised_future(100.0)
    changed_outcome = build_with_realised_future(1000.0)
    assert first.asDict() == changed_outcome.asDict(), \
        "a post-origin realised price leaked into the frozen historical reference"
    assert first["price"] == 9.0 and first["price_available"]
    assert first["discount_depth_available"]
    assert abs(first["discount_depth"] - 0.1) < 1e-12, first["discount_depth"]
    recovered_reference = first["price"] / (1.0 - first["discount_depth"])
    assert abs(recovered_reference - 10.0) < 1e-12, recovered_reference


def test_a_plan_can_be_the_only_source_of_commercial_roles() -> None:
    """A provenance-bound plan is the live horizon source, so it must not require duplicate
    price/promo roles on the historical panel. Without historical price there is no honest
    reference level: raw planned price is available, while derived discount depth is not."""
    panel = tempfile.mkdtemp()
    plan = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s1", dt.date(2016, 4, 24), 3.0)], "id string, date date, qty double"
    ).write.mode("overwrite").parquet(panel)
    spark().createDataFrame(
        [("s1", dt.date(2016, 4, 25), 9.5, 1.0),
         ("s2", dt.date(2016, 4, 25), float("nan"), float("inf")),
         ("s3", dt.date(2016, 4, 25), float("-inf"), float("nan"))],
        "series_id string, date date, price double, promo double",
    ).write.mode("overwrite").parquet(plan)

    out = spark().read.parquet(str(commercial.build(
        panel, "2016-04-24", 1, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="date", plan_path=plan)))
    rows = {row["series_id"]: row for row in out.collect()}
    row = rows["s1"]
    assert row["price"] == 9.5 and row["price_available"]
    assert row["promo"] == 1.0 and row["promo_available"]
    assert row["discount_depth"] == 0.0 and not row["discount_depth_available"], \
        "a planned raw price without a historical reference fabricated a discount depth"
    for sid in ("s2", "s3"):
        assert rows[sid]["price"] == 0.0 and not rows[sid]["price_available"]
        assert rows[sid]["promo"] == 0.0 and not rows[sid]["promo_available"], \
            "a non-finite plan value was marked available or reached the tensor"


def test_a_plan_without_standardized_commercial_roles_is_refused() -> None:
    """The commercial boundary consumes role names emitted by the as-of plan selection. An
    unrelated planned value cannot turn an empty commercial branch into a defined one."""
    panel = tempfile.mkdtemp()
    plan = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s1", dt.date(2016, 4, 24), 3.0)], "id string, date date, qty double"
    ).write.mode("overwrite").parquet(panel)
    spark().createDataFrame(
        [("s1", dt.date(2016, 4, 25), 0.25, dt.date(2016, 4, 20), "snapA")],
        "series_id string, date date, markdown double, _known_at date, _snapshot string",
    ).write.mode("overwrite").parquet(plan)

    try:
        commercial.build(
            panel, "2016-04-24", 1, spark(), tempfile.mkdtemp(),
            series_col="id", time_col="date", plan_path=plan)
    except commercial.CommercialError as e:
        assert "neither a standardized price nor promo role" in str(e); return
    raise AssertionError("accepted a plan with no standardized commercial role")


def test_a_client_with_neither_price_nor_promotion_is_refused() -> None:
    """Attention over an empty key set is undefined. The branch has to be dropped, not fed an
    all-masked row."""
    try:
        commercial.build(STORE, ORIGIN, HORIZON, spark(), "/tmp/never",
                         series_col="id", time_col="date")
    except commercial.CommercialError as e:
        assert "Drop the branch" in str(e); return
    raise AssertionError("built a commercial table with no channels at all")


# ---------------------------------------------------------------- calendar events

@pytest.mark.parametrize("origin,horizon", [("20160320", 5), ("2016-03-20", True)])
def test_calendar_refuses_noncanonical_span_before_spark_io(origin, horizon) -> None:
    with pytest.raises(calendar_events.CalendarError, match="canonical|horizon"):
        calendar_events.build(
            [("easter", "2016-03-27", None)],
            origin,
            horizon,
            ["s1"],
            object(),
            "/not-written",
        )


def test_a_recurring_event_is_one_name_measured_to_the_nearest_occurrence() -> None:
    """'Easter' across years is ONE event -- that identity is what the kernel pools across
    occurrences. The old builder refused duplicate names, so a recurring event had to be named
    per year, splitting the N the pooling design exists to grow, and multi-year training windows
    saw last year's spike with no event feature at all."""
    out = calendar_events.build(
        [("easter", "2015-04-05", None), ("easter", "2016-03-27", None)],
        "2016-03-20", 10, ["s1"], spark(), tempfile.mkdtemp(), max_support=30)
    rows = {r["date"].isoformat(): r for r in spark().read.parquet(str(out)).collect()}
    assert rows["2016-03-27"]["easter_distance"] == 0, "the 2016 occurrence is the nearest"
    assert rows["2016-03-25"]["easter_distance"] == -2 and rows["2016-03-25"]["easter_present"]
    # A year-old occurrence never bleeds in: nearest is 2016 throughout this horizon.
    assert all(abs(r["easter_distance"]) <= 30 for r in rows.values())


def test_the_same_occurrence_listed_twice_is_refused() -> None:
    try:
        calendar_events.build([("easter", "2016-03-27", None), ("easter", "2016-03-27", None)],
                              "2016-03-20", 5, ["s1"], spark(), tempfile.mkdtemp())
    except calendar_events.CalendarError as e:
        assert "twice" in str(e); return
    raise AssertionError("one occurrence counted as two")


def test_an_event_not_yet_announced_is_dropped() -> None:
    """Without this a model trained on historical promotions learns to FORESEE unannounced ones:
    it backtests beautifully and cannot do it live."""
    out = spark().read.parquet(str(calendar_events.build(
        [("xmas", "2016-05-01", None), ("secret_promo", "2016-05-01", "2016-04-30")],
        ORIGIN, HORIZON, ["a", "b"], spark(), tempfile.mkdtemp())))
    assert out.filter(F.col("xmas_present")).count() > 0, "a known event was dropped"
    assert out.filter(F.col("secret_promo_present")).count() == 0, \
        "an event announced AFTER the origin reached the features"


def test_known_at_is_iso_parsed_before_it_is_compared_to_the_origin() -> None:
    """String ordering can make a malformed date look meaningfully before or after an origin.

    `2016-3-9` sorts after `2016-03-10`, so the former implementation silently dropped the event
    instead of revealing that the announcement date was not an ISO date at all.  Calendar
    causality must be decided between dates, never between unchecked strings.
    """
    try:
        calendar_events.build(
            [("promo", "2016-03-11", "2016-3-9")], "2016-03-10", 2, ["a"], spark(),
            tempfile.mkdtemp())
    except calendar_events.CalendarError as exc:
        assert "known_at ISO date" in str(exc) and "2016-3-9" in str(exc)
    else:
        raise AssertionError("malformed known_at was silently ordered as a string")


def test_an_event_announced_on_the_origin_is_available() -> None:
    """The cutoff is inclusive: information published by the origin may enter the forecast."""
    out = spark().read.parquet(str(calendar_events.build(
        [("promo", "2016-04-26", ORIGIN)], ORIGIN, HORIZON, ["a"], spark(),
        tempfile.mkdtemp())))
    assert out.filter(F.col("promo_present")).count() > 0


def test_the_distance_is_signed() -> None:
    """Three days before Christmas and three days after are opposite states of the world; an
    absolute distance makes them identical."""
    out = spark().read.parquet(str(calendar_events.build(
        [("e", "2016-05-01", None)], ORIGIN, HORIZON, ["a"], spark(), tempfile.mkdtemp())))
    values = [r["e_distance"] for r in out.select("e_distance").collect()]
    assert min(values) < 0 < max(values), f"distances are not signed: {sorted(set(values))}"


def test_an_event_column_survives_even_when_the_event_does_not_apply() -> None:
    """The model's event columns are positional, so a client-specific column set would change the
    tensor's width between origins."""
    out = spark().read.parquet(str(calendar_events.build(
        [("never", "2016-05-01", "2099-01-01")], ORIGIN, HORIZON, ["a"], spark(),
        tempfile.mkdtemp())))
    assert "never_distance" in out.columns and "never_present" in out.columns
    assert out.filter(F.col("never_present")).count() == 0


def test_two_occurrences_of_one_event_share_one_column() -> None:
    """This used to be `test_duplicate_event_names_are_refused`, and the refusal was the defect:
    a recurring event HAD to be renamed per year, splitting the pooled identity the kernel's
    partial pooling depends on. Same name + different dates is one event; the refused case is
    the same (name, date) pair twice, covered by its own test above."""
    out = spark().read.parquet(str(calendar_events.build(
        [("e", "2016-05-01", None), ("e", "2016-06-01", None)],
        ORIGIN, HORIZON, ["a"], spark(), tempfile.mkdtemp())))
    assert out.columns.count("e_distance") == 1 and out.columns.count("e_present") == 1, \
        "occurrences of one event must share one column pair, or nothing pools across years"


# ---------------------------------------------------------------- static identity

def identity_path() -> pathlib.Path:
    _require_real_m5_store()
    return static_identity.build(
        STORE, spark(), tempfile.mkdtemp(), series_col="id",
        attribute_cols=("dept_id", "cat_id"), product_col="item_id", store_col="state_id")


def test_identity_is_one_row_per_series_on_real_m5() -> None:
    out = spark().read.parquet(str(identity_path()))
    assert out.count() == 3049
    assert out.select("series_id").distinct().count() == 3049


def test_a_joint_embedding_is_refused_by_default() -> None:
    """One row per (product, store) pair is tens of millions of rows on the estate, most seen a
    handful of times -- each a parameter fitted to almost no data."""
    out = spark().read.parquet(str(identity_path()))
    assert "joint_key" not in out.columns


def test_a_joint_embedding_needs_both_identifier_columns() -> None:
    try:
        static_identity.build(STORE, spark(), "/tmp/never", series_col="id",
                              attribute_cols=("dept_id",), allow_joint_embedding=True)
    except static_identity.IdentityError as e:
        assert "both a product and a store" in str(e); return
    raise AssertionError("crossed identifiers that were not supplied")


def test_identity_with_no_attributes_is_refused() -> None:
    """The branch would carry only a learned residual, which is a lookup table with no cold-start
    path -- and every launch goes through that path."""
    try:
        static_identity.build(STORE, spark(), "/tmp/never", series_col="id", attribute_cols=())
    except static_identity.IdentityError as e:
        assert "cold-start path" in str(e); return
    raise AssertionError("built an identity table with no attributes")


def test_a_declared_static_attribute_must_not_change_within_a_series() -> None:
    """Spark first() is partition-order dependent. A changing value under a static role must
    fail instead of choosing whichever catalogue version one task happened to return."""
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s1", "food"), ("s1", "toys")], "id string, category string"
    ).repartition(2).write.mode("overwrite").parquet(src)
    try:
        static_identity.build(
            src, spark(), tempfile.mkdtemp(), series_col="id",
            attribute_cols=("category",))
    except static_identity.IdentityError as exc:
        assert "changes within series" in str(exc)
        return
    raise AssertionError("a time-varying value was silently accepted as static identity")


def test_identity_retains_the_shard_assignment_for_bounded_collection() -> None:
    """A global identity table is safe only when each shard can filter it before collect()."""
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s0", "food", 0), ("s0", "food", 0),
         ("s1", "toys", 1), ("s1", "toys", 1)],
        "id string, category string, bucket integer",
    ).write.mode("overwrite").parquet(src)
    path = static_identity.build(
        src, spark(), tempfile.mkdtemp(), series_col="id",
        attribute_cols=("category",), shard_expr="bucket")

    out = spark().read.parquet(str(path))
    assert "_shard" in out.columns
    assert {(row["series_id"], row["_shard"]) for row in out.collect()} == {
        ("s0", 0), ("s1", 1)}
    assert [row["series_id"] for row in out.filter(F.col("_shard") == 0).collect()] == ["s0"]


def test_a_shard_expression_must_be_constant_within_series() -> None:
    """A split identity would either be collected twice or disappear from one shard."""
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        [("s1", "food", 0), ("s1", "food", 1)],
        "id string, category string, bucket integer",
    ).write.mode("overwrite").parquet(src)
    try:
        static_identity.build(
            src, spark(), tempfile.mkdtemp(), series_col="id",
            attribute_cols=("category",), shard_expr="bucket")
    except static_identity.IdentityError as exc:
        assert "changes within series" in str(exc)
        return
    raise AssertionError("one series was assigned to multiple identity shards")


def test_the_vocabulary_is_sorted_so_indices_are_stable_between_runs() -> None:
    """An unsorted vocabulary permutes every embedding row between runs, and a checkpoint trained
    under one ordering is meaningless under another."""
    vocab = static_identity.vocabulary(identity_path(), "dept_id", spark())
    assert vocab == sorted(vocab)
    assert len(vocab) == len(set(vocab))
    assert vocab == static_identity.vocabulary(identity_path(), "dept_id", spark())


def test_per_event_windows_gate_each_event_by_its_own_ceiling() -> None:
    """One shared +/-30 window forces a pay-day effect (no build-up) and a holiday (no payback
    tail) to carry each other's dead days -- `ForecastDataSpec.event_windows` is per event, and
    the builder now honours it: outside an event's OWN window is absent, inside another's is
    not enough."""
    out = spark().read.parquet(str(calendar_events.build(
        [("payday", "2016-05-01", None), ("holiday", "2016-05-01", None)],
        "2016-04-24", 14, ["s1"], spark(), tempfile.mkdtemp(), max_support=30,
        event_windows={"payday": (0, 5)})))
    rows = {r["date"].isoformat(): r for r in out.collect()}
    before = rows["2016-04-28"]                      # 3 days BEFORE both events
    assert not before["payday_present"], "a pay-day effect has no build-up; -3 is outside (0,5)"
    assert before["holiday_present"] and before["holiday_distance"] == -3
    after = rows["2016-05-04"]                       # 3 days after
    assert after["payday_present"] and after["payday_distance"] == 3
    try:
        calendar_events.build([("e", "2016-05-01", None)], "2016-04-24", 5, ["s1"], spark(),
                              tempfile.mkdtemp(), event_windows={"e": (2, 5)})
    except calendar_events.CalendarError as e:
        assert "bracket zero" in str(e)
    else:
        raise AssertionError("a window excluding the event's own day was accepted")


def test_plan_selection_is_as_of_the_origin_and_content_bound() -> None:
    """The provenance the blocker named: a revision KNOWN AFTER the origin does not exist for
    that origin's forecast, a later snapshot with earlier knowledge wins over an older one, an
    exact tie with two values is refused as two truths, and the selection hash binds exactly
    the knowledge used -- a post-origin revision leaves it unchanged, a pre-origin one moves it."""
    from forecast.features import plans

    rows = [
        # (series, valid, known_at, snapshot, price)
        ("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 1), "snapA", 5.0),
        ("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 20), "snapB", 6.0),   # later knowledge
        ("s1", dt.date(2016, 5, 2), dt.date(2016, 5, 1), "snapC", 9.99),   # AFTER the origin
        ("s2", dt.date(2016, 5, 3), dt.date(2016, 4, 10), "snapA", 3.0),
    ]
    src = tempfile.mkdtemp()
    spark().createDataFrame(rows, "id string, valid date, known date, snap string, p double") \
        .write.mode("overwrite").parquet(src)

    def select(source):
        return plans.build(source, "2016-04-24", 14, spark(), tempfile.mkdtemp(),
                           series_col="id", valid_date_col="valid", known_at_col="known",
                           snapshot_col="snap", value_cols={"price": "p"})

    sel = select(src)
    got = {(r["series_id"], r["date"].isoformat()): r["price"]
           for r in spark().read.parquet(sel.path).collect()}
    assert got[("s1", "2016-05-02")] == 6.0, \
        "the price known at the origin is 6.0; 9.99 was learned after and must not exist yet"
    assert got[("s2", "2016-05-03")] == 3.0 and sel.n_rows == 2

    # A post-origin revision added later must leave the selection BIT-IDENTICAL.
    revised = tempfile.mkdtemp()
    spark().createDataFrame(rows + [("s1", dt.date(2016, 5, 2), dt.date(2016, 6, 1), "snapD", 7.5)],
                            "id string, valid date, known date, snap string, p double") \
        .write.mode("overwrite").parquet(revised)
    assert select(revised).selection_hash == sel.selection_hash, \
        "knowledge acquired after the origin changed an as-of selection"
    # A PRE-origin revision is different knowledge and must move the hash.
    earlier = tempfile.mkdtemp()
    spark().createDataFrame(rows + [("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 22), "snapD", 7.5)],
                            "id string, valid date, known date, snap string, p double") \
        .write.mode("overwrite").parquet(earlier)
    assert select(earlier).selection_hash != sel.selection_hash

    tie = tempfile.mkdtemp()
    spark().createDataFrame(rows + [("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 20), "snapB", 6.5)],
                            "id string, valid date, known date, snap string, p double") \
        .write.mode("overwrite").parquet(tie)
    try:
        select(tie)
    except plans.PlanError as e:
        assert "two truths" in str(e)
    else:
        raise AssertionError("an exact-tie contradiction was ranked by arbitrary row order")

    null_tie = tempfile.mkdtemp()
    spark().createDataFrame(
        rows + [("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 20), "snapB", None)],
        "id string, valid date, known date, snap string, p double",
    ).write.mode("overwrite").parquet(null_tie)
    try:
        select(null_tie)
    except plans.PlanError as e:
        assert "two truths" in str(e)
    else:
        raise AssertionError("a null-versus-value exact tie was ranked by arbitrary row order")


def test_plan_hash_preserves_null_roles_and_selected_knowledge_rank() -> None:
    """Null has a position, and known_at/snapshot are the evidence behind an as-of value.
    Swapping a value between roles or changing selected provenance must move the digest."""
    from forecast.features import plans

    valid = dt.date(2016, 5, 2)
    known = dt.date(2016, 4, 20)

    def select(price, promo, snapshot="snapA", known_at=known):
        source = tempfile.mkdtemp()
        spark().createDataFrame(
            [("s1", valid, known_at, snapshot, price, promo)],
            "id string, valid date, known date, snap string, p double, m double",
        ).write.mode("overwrite").parquet(source)
        return plans.build(
            source, "2016-04-24", 14, spark(), tempfile.mkdtemp(),
            series_col="id", valid_date_col="valid", known_at_col="known",
            snapshot_col="snap", value_cols={"price": "p", "promo": "m"})

    selected = select(5.0, 1.0, "snapA")
    artifact = spark().read.parquet(selected.path)
    assert {"_known_at", "_snapshot"} <= set(artifact.columns), \
        "the selected knowledge rank was hashed but absent from the persisted evidence"
    persisted = artifact.collect()[0]
    assert persisted["_known_at"] == known and persisted["_snapshot"] == "snapA"

    price_null = select(None, 5.0)
    promo_null = select(5.0, None)
    assert price_null.selection_hash != promo_null.selection_hash, \
        "null role position was erased from the canonical selection hash"
    assert select(5.0, 1.0, "snapA").selection_hash != \
        select(5.0, 1.0, "snapB").selection_hash, \
        "different selected snapshots produced the same provenance hash"
    assert select(5.0, 1.0, known_at=known).selection_hash != select(
        5.0, 1.0, known_at=known - dt.timedelta(days=1)).selection_hash, \
        "different selected known_at values produced the same provenance hash"


def test_plan_hash_and_count_bind_the_written_selection_if_the_source_changes() -> None:
    """Write is one Spark action and digest/count is another. If both evaluate the source, a
    concurrent plan replacement can make the returned provenance describe rows that were not
    persisted. Mutate the source at that exact boundary and require the written artifact to win."""
    from forecast.features import plans
    from pyspark.sql.readwriter import DataFrameWriter

    schema = "id string, valid date, known date, snap string, p double"
    original_rows = [
        ("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 20), "snapA", 5.0)]
    replacement_rows = [
        ("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 21), "snapB", 99.0),
        ("s2", dt.date(2016, 5, 3), dt.date(2016, 4, 20), "snapA", 7.0)]
    source = tempfile.mkdtemp()
    reference_source = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()
    for path in (source, reference_source):
        spark().createDataFrame(original_rows, schema).write.mode("overwrite").parquet(path)

    def select(path: str, destination: str):
        return plans.build(
            path, "2016-04-24", 14, spark(), destination,
            series_col="id", valid_date_col="valid", known_at_col="known",
            snapshot_col="snap", value_cols={"price": "p"})

    reference = select(reference_source, tempfile.mkdtemp())
    original_parquet = DataFrameWriter.parquet
    source_changed = False

    def write_then_replace_source(writer, path, *args, **kwargs):
        nonlocal source_changed
        result = original_parquet(writer, path, *args, **kwargs)
        if str(path) == out_dir and not source_changed:
            source_changed = True
            (spark().createDataFrame(replacement_rows, schema).write.mode("overwrite")
             .parquet(source))
            spark().catalog.refreshByPath(source)
        return result

    with mock.patch.object(DataFrameWriter, "parquet", write_then_replace_source):
        selected = select(source, out_dir)

    written = spark().read.parquet(out_dir)
    assert source_changed and written.count() == selected.n_rows == reference.n_rows == 1
    assert written.collect()[0]["price"] == 5.0, "the test did not preserve the first action"
    assert selected.selection_hash == reference.selection_hash, \
        "the returned hash described a later source evaluation, not the persisted selection"


def test_plan_series_scope_limits_rows_and_hash() -> None:
    """A shard must not select/hash every other shard's plan rows. Changes outside its exact
    series scope are irrelevant; changes inside remain content-bound."""
    from forecast.features import plans

    def source(s2_price: float) -> str:
        path = tempfile.mkdtemp()
        spark().createDataFrame(
            [("s1", dt.date(2016, 5, 2), dt.date(2016, 4, 20), 5.0),
             ("s2", dt.date(2016, 5, 2), dt.date(2016, 4, 20), s2_price)],
            "id string, valid date, known date, p double",
        ).write.mode("overwrite").parquet(path)
        return path

    def select(path: str, ids=None):
        return plans.build(
            path, "2016-04-24", 14, spark(), tempfile.mkdtemp(),
            series_col="id", valid_date_col="valid", known_at_col="known",
            value_cols={"price": "p"}, series_ids=ids)

    original, outside_changed = source(9.0), source(99.0)
    scoped = select(original, ["s1"])
    assert scoped.n_rows == 1
    scoped_row = spark().read.parquet(scoped.path).collect()[0]
    assert scoped_row["series_id"] == "s1"
    assert scoped_row["_known_at"] == dt.date(2016, 4, 20)
    assert scoped_row["_snapshot"] is None, \
        "a source without a snapshot rank persisted invented provenance"
    assert select(outside_changed, ["s1"]).selection_hash == scoped.selection_hash, \
        "a row outside the shard changed its scoped plan hash"
    assert select(outside_changed).selection_hash != select(original).selection_hash, \
        "the same changed row was absent from the unscoped plan hash"

    for invalid in ("s1", ["s1", "s1"], [""]):
        try:
            select(original, invalid)
        except plans.PlanError as e:
            assert "series_ids" in str(e)
        else:
            raise AssertionError(f"invalid plan series scope was accepted: {invalid!r}")


def test_a_non_finite_history_price_does_not_poison_the_discount_reference() -> None:
    import datetime as dt
    import tempfile

    from forecast.features import commercial

    """The reference is the median FINITE price; NaN/Inf in the window must be ignored."""
    src = tempfile.mkdtemp()
    days = [dt.date(2016, 4, 24) - dt.timedelta(days=i) for i in range(0, 5)]
    # Two finite reference prices against three +inf ones: an unfiltered median becomes inf,
    # which makes 1 - price/ref == 1.0 look like a real, available 100% discount.
    rows = [("s1", days[1], 1.0, 10.0), ("s1", days[2], 1.0, 10.0),
            ("s1", days[0], 1.0, float("inf")), ("s1", days[3], 1.0, float("inf")),
            ("s1", days[4], 1.0, float("inf"))]
    rows += [("s1", dt.date(2016, 4, 25), 1.0, 10.0)]          # horizon day, planned price 10
    spark().createDataFrame(rows, "id string, date date, qty double, price double") \
        .write.mode("overwrite").parquet(src)
    out = spark().read.parquet(str(commercial.build(
        src, "2016-04-24", 1, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="date", price_col="price", reference_days=5)))
    row = out.collect()[0]
    assert row["price_available"] and row["price"] == 10.0
    assert row["discount_depth_available"], "a NaN in the reference window erased the depth"
    assert abs(row["discount_depth"]) < 1e-12, f"depth {row['discount_depth']} != 0 at the median"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
