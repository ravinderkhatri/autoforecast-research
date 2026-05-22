"""Tests for the panel profiler, on REAL M5 through Spark.

The profiler's job is to turn a client's data into decisions, so what the tests check is that the
numbers it reports are the ones those decisions actually rest on -- above all the zero rate on
ELIGIBLE cells, which differs from the naive figure by enough to change the answer.

M5's own numbers are the fixture here, because they are measured rather than invented: 68.0% zero
naive against 59.5% on eligible cells, and 20.9% of the grid ineligible.

Run: JAVA_HOME=... python3 autoforecast/test_profiling.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from pyspark.sql import SparkSession

from autoforecast.profiling import Profile, ProfileError, profile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "m5_parquet"
STORE = PANEL / "store_id=CA_1"
PANEL_END = "2016-05-22"

_SPARK: SparkSession | None = None
_P: Profile | None = None


def _require_public_m5(path: pathlib.Path) -> None:
    if not path.is_dir():
        pytest.skip("public M5 Parquet is not installed")


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_profiling")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def m5() -> Profile:
    """One profile of the whole real panel, reused."""
    global _P
    _require_public_m5(PANEL)
    if _P is None:
        _P = profile(PANEL, spark(), series_col="id", time_col="date", target_col="qty",
                     as_of=PANEL_END,
                     exposure_col="observation_state",
                     coverage_cols=("sell_price", "revenue"))
    return _P


def test_the_shape_matches_what_m5_actually_is() -> None:
    p = m5()
    assert p.n_cells == 59_181_090, f"got {p.n_cells:,}"
    assert p.n_series == 30_490
    assert p.n_days == 1941


def test_the_eligible_zero_rate_differs_from_the_naive_one() -> None:
    """THE measurement the hurdle head rests on. The naive figure counts never-ranged cells and so
    measures assortment; the eligible figure measures demand. Measured on M5: 68.0% against
    59.5%, which is not a rounding difference."""
    p = m5()
    assert 0.66 < p.zero_rate_naive < 0.70, p.zero_rate_naive
    assert 0.57 < p.zero_rate_eligible < 0.62, p.zero_rate_eligible
    assert p.zero_rate_naive - p.zero_rate_eligible > 0.05, \
        "the two rates coincide, so eligibility is not being applied"


def test_the_eligible_share_matches_m5s_leading_zeros() -> None:
    """20.9% of the grid, which is the largest single piece of the problem."""
    p = m5()
    assert 0.76 < p.eligible_share < 0.82, p.eligible_share


def test_the_scale_range_is_measured_from_percentiles_not_extremes() -> None:
    """One miskeyed row would otherwise set the whole figure -- and this number decides whether
    RevIN is load-bearing."""
    p = m5()
    assert p.scale_range > 1.0
    assert p.scale_range < 1e6, "an extreme set the range, so a single row can move this decision"


def test_the_notes_state_the_decision_each_number_supports() -> None:
    """A reader should not have to remember the thresholds. The RULE is what generalises to
    another estate; the number was derived from one."""
    p = m5()
    joined = " ".join(p.notes)
    assert "exposure contract" in joined
    assert "hurdle head is justified" in joined or "plain count head" in joined
    assert "RevIN" in joined or "normalisation" in joined


def test_a_panel_with_no_exposure_column_says_so_rather_than_reporting_one_number() -> None:
    """Otherwise the two zero rates coincide by construction and the output looks like a panel
    where eligibility had been considered and made no difference."""
    _require_public_m5(STORE)
    p = profile(STORE, spark(), series_col="id", time_col="date", target_col="qty",
                as_of=PANEL_END)
    assert abs(p.zero_rate_naive - p.zero_rate_eligible) < 1e-9
    assert any("has not been considered" in n for n in p.notes)


def test_column_coverage_is_reported_per_column() -> None:
    """Coverage differs by COLUMN, not by row -- which is what per-channel masking exists for."""
    p = m5()
    assert set(p.column_coverage) == {"sell_price", "revenue"}
    assert all(0.0 <= v <= 1.0 for v in p.column_coverage.values())


def test_an_absent_column_is_refused_rather_than_silently_skipped() -> None:
    source = pathlib.Path(tempfile.mkdtemp()) / "panel"
    spark().createDataFrame(
        [("A", "2024-01-01", 1.0)], "id string, date string, qty double"
    ).write.parquet(str(source))
    try:
        profile(
            source,
            spark(),
            series_col="id",
            time_col="date",
            target_col="not_a_column",
            as_of="2024-01-01",
        )
    except ProfileError as e:
        assert "not_a_column" in str(e); return
    raise AssertionError("profiled a panel using a column it does not have")


def test_a_non_string_as_of_is_refused() -> None:
    try:
        profile(PANEL, spark(), series_col="id", time_col="date", target_col="qty", as_of=123)
    except ProfileError as e:
        assert "as_of" in str(e); return
    raise AssertionError("accepted a non-string profile cutoff")


def test_an_invalid_iso_as_of_is_refused() -> None:
    try:
        profile(PANEL, spark(), series_col="id", time_col="date", target_col="qty",
                as_of="2016-02-30")
    except ProfileError as e:
        assert "ISO date" in str(e); return
    raise AssertionError("accepted a profile cutoff that is not a calendar date")


def test_a_cutoff_before_the_panel_is_refused() -> None:
    src = tempfile.mkdtemp()
    spark().createDataFrame(
        [("A", "2024-01-02", 1.0)], ("id", "date", "qty")
    ).write.mode("overwrite").parquet(src)
    try:
        profile(src, spark(), series_col="id", time_col="date", target_col="qty",
                as_of="2024-01-01")
    except ProfileError as e:
        assert "no rows on or before 2024-01-01" in str(e); return
    raise AssertionError("profiled an empty pre-cutoff panel")


def test_the_record_and_the_string_carry_the_same_numbers() -> None:
    """The string is what a person reads and the record is what a manifest stores. If they could
    disagree, the number quoted in a review would not be the number that was recorded."""
    import json
    p = m5()
    rec = json.loads(json.dumps(p.as_record()))
    assert rec["n_cells"] == p.n_cells
    assert rec["as_of"] == p.as_of == PANEL_END
    assert abs(rec["zero_rate_eligible"] - p.zero_rate_eligible) < 1e-12
    assert f"{p.n_series:,}" in str(p)


def test_post_origin_rows_cannot_change_the_resolved_zero_share_or_head() -> None:
    """Architecture selection is an origin-time decision, not a full-dataset statistic.

    The future contamination is deliberately large enough to cross the resolver's 10% hurdle
    threshold. If the runner ever drops its cutoff again, this changes ``negbin`` to
    ``hurdle_negbin`` and the regression fails for the architectural consequence, not merely for
    a dataframe row count.
    """
    from spec.forecast_spec import ForecastDataSpec, ForecastModelSpec
    from spec.resolved import ResolvedBuild

    origin = "2016-04-24"

    def resolved(path):
        shape = profile(path, spark(), series_col="id", time_col="date", target_col="qty",
                        as_of=origin, exposure_col="observation_state")
        return ResolvedBuild.from_specs(
            ForecastDataSpec(time_idx="date", target="qty", group_ids=("id",)),
            ForecastModelSpec(), zero_share=shape.zero_rate_eligible)

    root = pathlib.Path(tempfile.mkdtemp())
    historical = [
        ("A", "2016-04-23", 2, "exact"),
        ("A", origin, 3, "exact"),  # the inclusive boundary must be retained
    ]
    future = [("A", f"2016-05-{day:02d}", 0, "exact") for day in range(1, 21)]
    before = root / "before"
    contaminated = root / "contaminated"
    spark().createDataFrame(
        historical, ("id", "date", "qty", "observation_state")).write.parquet(str(before))
    spark().createDataFrame(
        historical + future,
        ("id", "date", "qty", "observation_state")).write.parquet(str(contaminated))

    def panel(path: pathlib.Path) -> SimpleNamespace:
        return SimpleNamespace(
            path=str(path), series_col="id", time_col="date", target_col="qty",
            exposure_col="observation_state", price_col="sell_price")

    original = resolved(before)
    with_future = resolved(contaminated)
    unbounded_shape = profile(
        contaminated, spark(), series_col="id", time_col="date", target_col="qty",
        as_of="2016-05-20", exposure_col="observation_state")

    assert unbounded_shape.zero_rate_eligible > 0.10, \
        "the synthetic future rows were not strong enough to exercise the head threshold"
    assert original.zero_share == with_future.zero_share == 0.0
    assert original.head == with_future.head == "negbin"
    assert original.build_hash == with_future.build_hash


def test_scale_range_is_bounded_by_as_of_like_the_zero_rate() -> None:
    import math
    import pathlib
    import tempfile

    from autoforecast.profiling import profile

    """A future fast-mover must not change the origin-time RevIN decision (scale_range)."""
    root = pathlib.Path(tempfile.mkdtemp())
    historical = [(sid, "2016-04-2%d" % d, q, "exact")
                  for sid, q in (("A", 1.0), ("B", 2.0), ("C", 3.0), ("D", 4.0))
                  for d in range(0, 5)]
    future = [("A", "2016-05-%02d" % d, 100000.0, "exact") for d in range(1, 21)]
    before, contaminated = root / "before", root / "contaminated"
    spark().createDataFrame(historical, ("id", "date", "qty", "observation_state")) \
        .write.parquet(str(before))
    spark().createDataFrame(historical + future, ("id", "date", "qty", "observation_state")) \
        .write.parquet(str(contaminated))
    kw = dict(series_col="id", time_col="date", target_col="qty", as_of="2016-04-24",
              exposure_col="observation_state")
    clean = profile(before, spark(), **kw)
    bounded = profile(contaminated, spark(), **kw)
    unbounded = profile(contaminated, spark(), **{**kw, "as_of": "2016-05-20"})
    assert unbounded.scale_range > 100 * clean.scale_range, \
        "the synthetic future rows were not strong enough to move the scale range"
    assert math.isclose(bounded.scale_range, clean.scale_range, rel_tol=1e-12), \
        f"post-origin rows changed scale_range: {bounded.scale_range} vs {clean.scale_range}"
    assert bounded.as_record() == clean.as_record(), "post-origin rows changed the profile record"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
