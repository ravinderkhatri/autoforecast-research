"""Tests for the seasonal-naive adapter, on REAL M5 data through a real Spark session.

Scoped to one real store partition (CA_1: 3,049 series x 1,941 days) rather than all ten, because
the properties under test are per-series and ten stores would multiply runtime without adding a
case. Nothing here is synthetic: M5 supplies the leading zeros, the gaps and the intermittency,
and every expected value is read back out of the same Parquet the adapter read.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from collections.abc import Iterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters.naive import NAME, seasonal_naive
from autoforecast.panel import Forecast, Panel

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# The local test panel can be rebuilt from the pinned raw M5 CSVs when data is supplied.
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"

# `sales_train_evaluation.csv` ends 2016-05-22 (measured). The competition's last 28 days are
# held-out labels in a file we do not have, so this is the LATEST origin with a full 28 days of
# real actuals on the far side -- which is what Phase 1 scores against.
ORIGIN = "2016-04-24"
# Early in M5, where most series had not been listed yet: 71.2% of this origin's source window is
# `unobserved`, against 0.0% at ORIGIN. The eligibility cases can only be exercised here.
EARLY_ORIGIN = "2011-02-01"
HORIZON = 28
SEASON = 7

_SPARK: SparkSession | None = None


def _require_real_m5_store() -> None:
    if not STORE.exists():
        pytest.skip(f"real M5 Parquet fixture is not present: {STORE}")


def spark() -> SparkSession:
    """One local session for the whole file. Starting a JVM per test dominates the runtime."""
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError(
                "JAVA_HOME is not set, so Spark cannot start and these tests would silently "
                "prove nothing. Install a JDK 17 and export it -- set JAVA_HOME to a Java 17 installation."
            )
        _SPARK = (SparkSession.builder.master("local[*]").appName("test_naive")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.ui.enabled", "false")
                  # Spark's console progress bar writes \r without a newline, so it overwrites
                  # the line a PASS is printed on and the run appears to skip a test. It does not.
                  .config("spark.ui.showConsoleProgress", "false")
                  .getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


@pytest.fixture(scope="module", autouse=True)
def _release_spark_after_module() -> Iterator[None]:
    """Do not leak this module's `local[*]` session into another real-Spark suite."""
    yield
    global _SPARK
    if _SPARK is not None:
        _SPARK.stop()
        _SPARK = None


def _panel_contract(**over: object) -> Panel:
    kw: dict[str, object] = dict(path=STORE, series_col="id", time_col="date", target_col="qty",
                                 exposure_col="observation_state", season_length=SEASON)
    kw.update(over)
    return Panel(**kw)  # type: ignore[arg-type]


def panel(**over: object) -> Panel:
    _require_real_m5_store()
    return _panel_contract(**over).validate()


def run(p: Panel, origin: str = ORIGIN, horizon: int = HORIZON) -> tuple[Forecast, object]:
    """Run the adapter into a temp directory and return the Forecast plus the written frame."""
    d = tempfile.mkdtemp()
    fc = seasonal_naive(p, origin, horizon, spark=spark(), out_dir=d)
    return fc, spark().read.parquet(d)


def test_every_requested_cell_on_real_m5_is_accounted_for() -> None:
    fc, _ = run(panel())
    # `validate()` inside the adapter already enforces this; asserting it here makes the property
    # visible at the level a reader checks, and fails loudly if the adapter stops validating.
    assert sum(fc.disposition.values()) == fc.n_series * HORIZON
    assert fc.n_series == 3049, f"CA_1 has 3,049 series in M5; got {fc.n_series}"
    assert fc.adapter == NAME


def test_real_leading_zeros_are_declined_not_forecast_as_zero() -> None:
    # M5's leading zeros are absence of listing, not absence of demand. Copying one forward would
    # manufacture a confident prediction of no demand out of an assortment decision.
    #
    # MEASURED, and it corrected this test's first premise. Ineligible cells are 20.9% of M5
    # overall (12,384,870 of 59,181,090, all ten stores) but they are concentrated entirely in the
    # early history: in the source window for ORIGIN they are 0.0% (0 of 213,430), and in the
    # source window for EARLY_ORIGIN they are 71.2% (86,786 of 121,960). So this case has to be
    # exercised at an early origin -- at a recent one there is nothing to decline, and asserting
    # otherwise tested the test rather than the adapter.
    fc, _ = run(panel(), origin=EARLY_ORIGIN, horizon=7)
    assert fc.disposition.get("declined_source_unobserved", 0) > 0, (
        "no cell declined for an unobserved source, though 71.2% of the source window is "
        f"ineligible -- the exposure column is not being read: {fc.disposition}")
    assert fc.n_emitted > 0, "declined everything; the eligible 28.8% should still forecast"


def test_a_recent_origin_on_m5_has_nothing_to_decline() -> None:
    # The other half of the measurement above, asserted so the bound on what M5 can demonstrate is
    # visible in the suite: at the origin Phase 1 will actually score, M5 exercises the eligibility
    # path not at all. Any claim that the exposure contract helped must come from EARLY_ORIGIN or
    # from estate data -- not from this run.
    fc, _ = run(panel())
    assert fc.disposition == {"emitted": fc.n_series * HORIZON}, (
        f"expected every cell emitted at a recent M5 origin, got {fc.disposition}")


def test_a_horizon_beyond_one_season_reads_the_same_source_day() -> None:
    # h=1 and h=8 both resolve to origin-6, so their forecasts must be identical for every series.
    # This is the seasonal property itself, checked on real series rather than asserted.
    _, out = run(panel(), horizon=14)
    import datetime as dt
    d0 = dt.date.fromisoformat(ORIGIN)
    a = out.filter(F.col("date") == F.lit((d0 + dt.timedelta(days=1)).isoformat()))
    b = out.filter(F.col("date") == F.lit((d0 + dt.timedelta(days=8)).isoformat()))
    joined = a.select(F.col("series_id"), F.col("yhat").alias("y1")).join(
        b.select(F.col("series_id"), F.col("yhat").alias("y8")), "series_id")
    assert joined.count() > 0, "no series had both h=1 and h=8 emitted"
    assert joined.filter(F.col("y1") != F.col("y8")).count() == 0


def test_emitted_values_equal_the_real_history_at_the_source_date() -> None:
    # The forecast is a copy. Verify it against the same Parquet the adapter read, rather than
    # against a reimplementation of the lag, which could only agree with itself.
    import datetime as dt
    _, out = run(panel(), horizon=7)
    hist = spark().read.parquet(str(STORE))
    d0 = dt.date.fromisoformat(ORIGIN)
    for h in (1, 4, 7):
        target = (d0 + dt.timedelta(days=h)).isoformat()
        source = (d0 + dt.timedelta(days=h - SEASON)).isoformat()
        got = out.filter(F.col("date") == F.lit(target)).select("series_id", "yhat")
        want = (hist.filter(F.col("date") == F.lit(source))
                    .select(F.col("id").alias("series_id"), F.col("qty").alias("expected")))
        pair = got.join(want, "series_id")
        assert pair.count() == got.count(), f"h={h}: an emitted cell has no source row"
        assert pair.filter(F.col("yhat") != F.col("expected")).count() == 0, f"h={h} mismatch"


def test_an_origin_before_most_series_start_declines_for_want_of_a_source() -> None:
    # 2011-02-01 is early in M5: most CA_1 series had not been listed yet, so their source cells
    # are unobserved, and series with no history at all yield no source row.
    fc, _ = run(panel(), origin=EARLY_ORIGIN, horizon=7)
    declined = fc.disposition.get("declined_source_unobserved", 0) + \
        fc.disposition.get("declined_no_source", 0)
    assert declined > fc.n_emitted, (
        f"an origin near the start of M5 should decline more than it emits; got {fc.disposition}")


def test_a_panel_with_no_exposure_column_treats_every_source_as_exact() -> None:
    # Public benchmarks routinely carry no exposure information. That is a property of the data,
    # and the adapter must still run -- not silently decline everything.
    fc, _ = run(panel(exposure_col=None))
    assert "declined_source_unobserved" not in fc.disposition
    assert fc.n_emitted > 0
    assert sum(fc.disposition.values()) == fc.n_series * HORIZON


def test_the_exposure_column_changes_the_answer() -> None:
    # If declaring exposure made no difference, the contract would be decorative. It is the
    # difference between these two runs that the whole design rests on.
    with_exp, _ = run(panel(), origin=EARLY_ORIGIN, horizon=7)
    without, _ = run(panel(exposure_col=None), origin=EARLY_ORIGIN, horizon=7)
    assert with_exp.n_emitted < without.n_emitted, (
        "declaring the exposure column emitted just as many cells -- the column is being ignored")


def test_the_source_window_is_exactly_one_season_whatever_the_horizon() -> None:
    """The claim the adapter's scalability rests on: source dates lie in [origin-m+1, origin] for
    EVERY horizon, so a 28-day and a 365-day forecast read the same seven days of M5. Without it
    the join's right side is the full 59,181,090-row history and a local run OOMs the JVM."""
    from autoforecast.adapters.naive import _source_window
    _require_real_m5_store()
    raw = spark().read.parquet(str(STORE))
    for horizon in (7, 28, 90, 365):
        win = _source_window(raw, panel(), ORIGIN, SEASON)
        dates = sorted(r["_hist_date"] for r in win.select("_hist_date").distinct().collect())
        assert len(dates) == SEASON, f"h={horizon}: {len(dates)} source dates, expected {SEASON}"
        assert dates[-1].isoformat() == ORIGIN
        assert (dates[-1] - dates[0]).days == SEASON - 1


def test_a_horizon_far_beyond_one_season_still_forecasts_every_cell() -> None:
    """91 days is thirteen seasons. Every cell must still be accounted for, and the values must
    repeat with period m -- which is what a seasonal naive IS, asserted on real series."""
    fc, out = run(panel(), horizon=91)
    assert sum(fc.disposition.values()) == fc.n_series * 91
    import datetime as dt
    d0 = dt.date.fromisoformat(ORIGIN)
    first = out.filter(F.col("date") == F.lit((d0 + dt.timedelta(days=3)).isoformat())) \
               .select("series_id", F.col("yhat").alias("a"))
    last = out.filter(F.col("date") == F.lit((d0 + dt.timedelta(days=3 + 12 * SEASON)).isoformat())) \
              .select("series_id", F.col("yhat").alias("b"))
    pair = first.join(last, "series_id")
    assert pair.count() > 0
    assert pair.filter(F.col("a") != F.col("b")).count() == 0, \
        "day 3 and day 3+12 seasons differ, so the forecast is not periodic in m"


def test_a_non_positive_horizon_is_refused() -> None:
    for bad in (0, -1):
        try:
            seasonal_naive(_panel_contract(), ORIGIN, bad, spark=spark(), out_dir="/tmp/never")
        except ValueError as e:
            assert "horizon" in str(e); continue
        raise AssertionError(f"accepted horizon={bad}")


def test_daily_adapter_refuses_invalid_span_before_spark_io() -> None:
    class ReadSentinel:
        @property
        def read(self):
            raise AssertionError("invalid daily span reached Spark")

    class IntAlias(int):
        pass

    daily = _panel_contract()
    cases = (
        (_panel_contract(frequency="W"), ORIGIN, HORIZON, "daily"),
        (daily, "20160424", HORIZON, "canonical"),
        (daily, ORIGIN, True, "positive integer"),
        (daily, ORIGIN, 1.0, "positive integer"),
        (daily, ORIGIN, IntAlias(1), "positive integer"),
        (daily, ORIGIN, 10**400, "representable"),
    )
    for described, origin, horizon, message in cases:
        with pytest.raises(ValueError, match=message):
            seasonal_naive(
                described, origin, horizon, spark=ReadSentinel(), out_dir="/not-written"
            )


def test_write_failure_releases_cached_labelled_frame(tmp_path, monkeypatch) -> None:
    """An output failure must not leave executor memory owned by an aborted adapter call."""
    from pyspark.sql import DataFrame
    source = tmp_path / "panel"
    rows = [
        ("a", f"2024-01-{day:02d}", float(day), "exact")
        for day in range(1, 8)
    ]
    (spark().createDataFrame(rows, "id string, date string, qty double, state string")
     .selectExpr("id", "to_date(date) as date", "qty", "state")
     .write.parquet(str(source)))
    described = Panel(source, "id", "date", "qty", exposure_col="state", season_length=7)
    cached: list[DataFrame] = []
    frame_type = type(spark().range(1))
    writer_type = type(spark().range(1).write)
    real_cache = frame_type.cache

    def record_cache(frame: DataFrame) -> DataFrame:
        cached.append(real_cache(frame))
        return cached[-1]

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected forecast write failure")

    monkeypatch.setattr(frame_type, "cache", record_cache)
    monkeypatch.setattr(writer_type, "parquet", fail_write)
    with pytest.raises(RuntimeError, match="injected forecast write failure"):
        seasonal_naive(
            described,
            "2024-01-07",
            2,
            spark=spark(),
            out_dir=tmp_path / "forecast",
        )
    assert len(cached) == 1
    assert not cached[0].is_cached, "aborted seasonal-naive call retained its labelled cache"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
