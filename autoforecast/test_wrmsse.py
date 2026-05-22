"""Tests for WRMSSE, on REAL M5 through Spark.

The metric exists so our number can be compared to a published leaderboard score, which means the
ways it can quietly disagree with the published definition matter more than its value: weights
that do not sum to one, a denominator that includes leading zeros, or an aggregate level whose
benchmark is averaged from its parts instead of computed on the aggregate.

Scoped to one real store. The formula is per-level and per-series, so ten stores would multiply
runtime without adding a case -- but the level list is the full twelve, and the levels that
collapse to a single group on one store are exactly where a weight bug would show.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast import registry, wrmsse
from autoforecast.adapters import register_defaults
from autoforecast.benchmarks.m5 import M5_LEVELS
from autoforecast.panel import Forecast, Panel

ROOT = pathlib.Path(__file__).resolve().parent.parent
PANEL_ROOT = ROOT / "data" / "m5_parquet"
_STORE: pathlib.Path | None = None
ORIGIN = "2016-04-24"
HORIZON = 28

_SPARK: SparkSession | None = None
_FC = None


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_wrmsse")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def store() -> pathlib.Path:
    """One real store, written so `store_id` is a COLUMN rather than a partition directory.

    Reading the partitioned root materialises `store_id`; writing that slice out keeps it. The
    metric needs it as a column because levels 3, 8, 9 and 12 group by it, and pointing straight
    at `store_id=CA_1/` would put it back in the directory name where only Spark can see it.
    """
    global _STORE
    if not PANEL_ROOT.is_dir():
        pytest.skip("public M5 Parquet is not installed")
    if _STORE is None:
        d = pathlib.Path(tempfile.mkdtemp()) / "ca1"
        (spark().read.parquet(str(PANEL_ROOT)).filter(F.col("store_id") == "CA_1")
         .write.mode("overwrite").parquet(str(d)))
        _STORE = d
    return _STORE


def panel() -> Panel:
    return Panel(path=store(), series_col="id", time_col="date", target_col="qty",
                 exposure_col="observation_state", season_length=7).validate()


def forecast():
    """One real seasonal-naive forecast, reused across the cases."""
    global _FC
    if _FC is None:
        register_defaults() if "seasonal_naive" not in registry.names() else None
        _FC = registry.run("seasonal_naive", panel(), ORIGIN, HORIZON, spark=spark(),
                           out_dir=tempfile.mkdtemp())
    return _FC


def scored():
    return wrmsse.score(panel(), forecast(), spark(), levels=M5_LEVELS, weight_col="revenue")


def _tiny_case(
    *,
    omit_prediction: tuple[str, str] | None = None,
    extra_prediction: tuple[str, str, float] | None = None,
    omit_actual: tuple[str, str] | None = None,
    values: Mapping[str, Sequence[float]] | None = None,
    revenue_override: float | None = None,
) -> tuple[Panel, Forecast]:
    """Small independent fixture for adversarial grid checks; it does not use an adapter."""
    root = pathlib.Path(tempfile.mkdtemp())
    actual_rows = []
    values = values or {
        # The combined training series must vary too. A previous fixture summed to 3 every day,
        # so its supposedly successful case silently dropped the total level.
        "a": (1.0, 2.0, 1.0, 3.0, 8.0, 1.0),
        "b": (2.0, 1.0, 2.0, 1.0, 1.0, 8.0),
    }
    days = [f"2024-01-{day:02d}" for day in range(1, 7)]
    for sid, ys in values.items():
        for day, y in zip(days, ys, strict=True):
            if omit_actual == (sid, day):
                continue
            revenue = revenue_override if revenue_override is not None else y * 2.0
            actual_rows.append((sid, day, y, revenue, sid))
    panel_path = root / "panel"
    (spark().createDataFrame(actual_rows, "id string, date string, qty double, "
                                          "revenue double, grp string")
     .selectExpr("id", "to_date(date) as date", "qty", "revenue", "grp")
     .write.parquet(str(panel_path)))

    pred_rows = []
    for sid in values:
        for day in days[-2:]:
            if omit_prediction == (sid, day):
                continue
            pred_rows.append((sid, day, 0.0))
    if extra_prediction is not None:
        pred_rows.append(extra_prediction)
    forecast_path = root / "forecast"
    (spark().createDataFrame(pred_rows, "series_id string, date string, yhat double")
     .selectExpr("series_id", "to_date(date) as date", "yhat")
     .write.parquet(str(forecast_path)))
    emitted = len(pred_rows)
    requested = 4
    disposition = {"emitted": emitted}
    if emitted < requested:
        disposition["declined"] = requested - emitted
    return (
        Panel(path=panel_path, series_col="id", time_col="date", target_col="qty").validate(),
        Forecast(adapter="adversarial", path=forecast_path, origin="2024-01-04", horizon=2,
                 n_series=2, disposition=disposition),
    )


def _assert_wrmsse_error(panel_: Panel, forecast_: Forecast, text: str) -> None:
    try:
        wrmsse.score(panel_, forecast_, spark(), levels=((), ("grp",)),
                     weight_col="revenue")
    except wrmsse.WrmsseError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError("WRMSSE accepted malformed or incomplete evidence")


def test_wrmsse_scores_all_twelve_levels_on_real_m5() -> None:
    r = scored()
    assert len(r.per_level) == 12, f"expected 12 levels, got {sorted(r.per_level)}"
    assert r.score > 0
    assert all(v >= 0 for v in r.per_level.values())


def test_the_score_is_the_mean_of_the_levels_and_nothing_else() -> None:
    """WRMSSE is defined as that mean. A weighted-by-something-else average is another metric,
    and the difference is invisible in a single number."""
    r = scored()
    assert abs(r.score - sum(r.per_level.values()) / 12) < 1e-9


def test_wrmsse_record_refuses_a_score_that_is_not_the_mean_of_its_levels() -> None:
    try:
        wrmsse.Wrmsse(score=3.0, per_level={"total": 1.0, "grp": 2.0})
    except wrmsse.WrmsseError as e:
        assert "is not the mean" in str(e); return
    raise AssertionError("accepted a score that is not the mean of its levels")


def test_the_grand_total_level_and_the_store_item_level_disagree() -> None:
    """Level 1 is the grand total, level 12 is our forecasting grain. A forecast can be good at
    12 and bad at 1 -- that gap IS cross-series cancellation, which is the defect the honest
    metric was built to expose. If they were equal the aggregation would not be happening."""
    r = scored()
    assert abs(r.per_level["total"] - r.per_level["item_id+store_id"]) > 1e-6


def test_no_level_silently_drops_an_undefined_series() -> None:
    """Published WRMSSE covers its full hierarchy population; a zero scale must fail closed."""
    r = scored()
    assert set(r.n_series_dropped) == set(r.per_level)
    assert not any(r.n_series_dropped.values()), r.n_series_dropped
    assert r.n_series_scored["item_id+store_id"] == 3049


def test_the_denominator_excludes_leading_zeros() -> None:
    """M5's definition starts each series' benchmark at its first non-zero sale. Including the
    leading zeros makes the denominator smaller -- a long run of zero-to-zero steps -- which
    inflates RMSSE. Checked by recomputing one real series both ways."""
    df = (spark().read.parquet(str(store()))
          .filter(F.col("date") <= F.lit(ORIGIN))
          .select(F.col("id").alias("sid"), "date", F.col("qty").cast("double").alias("y")))
    sid = (df.filter(F.col("y") > 0).groupBy("sid").agg(F.min("date").alias("f"))
           .orderBy(F.col("f").desc()).first())          # the latest first-sale = most leading zeros
    from pyspark.sql import Window
    w = Window.partitionBy("sid").orderBy("date")
    one = df.filter(F.col("sid") == sid["sid"]).withColumn("prev", F.lag("y").over(w))
    sq = one.filter(F.col("prev").isNotNull()).withColumn("s", F.pow(F.col("y") - F.col("prev"), 2))
    with_leading = sq.agg(F.avg("s")).collect()[0][0]
    without = sq.filter(F.col("date") >= F.lit(sid["f"])).agg(F.avg("s")).collect()[0][0]
    assert without > with_leading, (
        "excluding leading zeros did not raise the benchmark, so this series has none and the "
        "test is not exercising the case")


def test_a_panel_without_the_weight_column_is_refused() -> None:
    """WRMSSE weights by dollar sales. A panel with no value column cannot support it, and a
    metric must not invent a hierarchy or a weight."""
    p, fc = _tiny_case()
    try:
        wrmsse.score(p, fc, spark(), levels=((), ("grp",)), weight_col="not_a_column")
    except wrmsse.WrmsseError as e:
        assert "missing columns" in str(e); return
    raise AssertionError("scored without the weights the definition requires")


def test_an_unknown_level_column_is_refused() -> None:
    p, fc = _tiny_case()
    try:
        wrmsse.score(p, fc, spark(), levels=(("region_id",),), weight_col="revenue")
    except wrmsse.WrmsseError as e:
        assert "region_id" in str(e); return
    raise AssertionError("scored a level the panel has no column for")


def test_eligible_only_states_the_convention_rather_than_applying_it() -> None:
    """The published score does not filter to eligible cells, so ours must not either -- a
    filtered number would be better for a reason that has nothing to do with forecasting."""
    text = wrmsse.eligible_only(Panel("unused", "id", "date", "qty"))
    assert "FULL grid" in text and "NOT applied" in text


def test_as_record_is_json_safe() -> None:
    import json
    rec = scored().as_record()
    assert len(json.loads(json.dumps(rec))["per_level"]) == 12


def test_wrmsse_refuses_a_selectively_omitted_prediction() -> None:
    """An inner join used to erase this hard cell and award an artificially lower score."""
    p, fc = _tiny_case(omit_prediction=("a", "2024-01-05"))
    _assert_wrmsse_error(p, fc, "full series-by-horizon grid")


def test_wrmsse_refuses_a_forecast_key_outside_the_window() -> None:
    p, fc = _tiny_case(
        omit_prediction=("a", "2024-01-05"),
        extra_prediction=("a", "2024-01-07", 0.0),
    )
    _assert_wrmsse_error(p, fc, "full series-by-horizon grid")


def test_wrmsse_refuses_a_missing_horizon_actual() -> None:
    p, fc = _tiny_case(omit_actual=("a", "2024-01-05"))
    _assert_wrmsse_error(p, fc, "missing 1 actual cell")


def test_wrmsse_accepts_the_exact_full_grid() -> None:
    p, fc = _tiny_case()
    got = wrmsse.score(p, fc, spark(), levels=((), ("grp",)), weight_col="revenue")
    assert set(got.per_level) == {"total", "grp"}
    assert got.score > 0
    assert got.n_series_dropped == {"total": 0, "grp": 0}


def test_wrmsse_refuses_a_constant_training_scale_instead_of_dropping_it() -> None:
    """The old inner join/drop path turned a catastrophic error on `a` into no contribution."""
    p, fc = _tiny_case(values={
        "a": (1.0, 1.0, 1.0, 1.0, 1_000_000.0, 1_000_000.0),
        "b": (1.0, 2.0, 1.0, 3.0, 1.0, 1.0),
    })
    _assert_wrmsse_error(p, fc, "cannot drop that group")


def test_wrmsse_refuses_a_missing_active_training_day() -> None:
    """A row lag over Jan 2 -> Jan 4 is not a one-day naive benchmark."""
    p, fc = _tiny_case(omit_actual=("a", "2024-01-03"))
    _assert_wrmsse_error(p, fc, "consecutive daily observations")


def test_wrmsse_refuses_finite_inputs_whose_squared_error_overflows() -> None:
    """Every input is finite, but (actual - prediction)^2 is not representable as float64."""
    p, fc = _tiny_case(values={
        "a": (1.0, 2.0, 1.0, 3.0, 9e307, 9e307),
        "b": (2.0, 1.0, 2.0, 1.0, 1.0, 1.0),
    }, revenue_override=1.0)
    _assert_wrmsse_error(p, fc, "non-finite")


def test_wrmsse_record_refuses_non_finite_or_dropped_level_results() -> None:
    for value in (float("nan"), float("inf")):
        try:
            wrmsse.Wrmsse(score=value, per_level={"total": value})
        except wrmsse.WrmsseError as exc:
            assert "finite" in str(exc)
        else:
            raise AssertionError("accepted a non-finite WRMSSE record")
    try:
        wrmsse.Wrmsse(score=1.0, per_level={"total": 1.0},
                      n_series_scored={"total": 1}, n_series_dropped={"total": 1})
    except wrmsse.WrmsseError as exc:
        assert "may not drop" in str(exc)
    else:
        raise AssertionError("accepted a score that silently changed the metric population")


def test_wrmsse_record_refuses_arbitrary_precision_json_integers() -> None:
    huge = 10**400
    for score_value, levels in ((huge, {"total": 1.0}), (1.0, {"total": huge})):
        try:
            wrmsse.Wrmsse(score=score_value, per_level=levels)
        except wrmsse.WrmsseError as exc:
            assert "finite" in str(exc), exc
            continue
        raise AssertionError("an arbitrary-precision integer entered a WRMSSE record")


def test_weight_days_is_an_exact_representable_positive_integer() -> None:
    class ReadSentinel:
        @property
        def read(self):
            raise AssertionError("invalid weight_days reached Spark")

    class IntAlias(int):
        pass

    p = Panel(path="unused", series_col="id", time_col="date", target_col="qty")
    fc = Forecast(
        adapter="test",
        path="unused",
        origin="2024-01-04",
        horizon=1,
        n_series=1,
        disposition={"emitted": 1},
    )
    for invalid in (True, 0, -1, 1.0, "1", IntAlias(1), 10**400):
        try:
            wrmsse.score(
                p,
                fc,
                ReadSentinel(),
                levels=((),),
                weight_col="revenue",
                weight_days=invalid,
            )
        except wrmsse.WrmsseError as exc:
            assert "weight_days" in str(exc)
            continue
        raise AssertionError(f"accepted invalid weight_days {invalid!r}")


def test_published_wrmsse_refuses_non_daily_cadence_before_reading_data() -> None:
    panel = Panel("/not-read", "id", "date", "qty", frequency="W")
    forecast = Forecast(
        "weekly", "/not-read", "2024-01-01", 2, 1, {"emitted": 2}, frequency="W"
    )
    with pytest.raises(wrmsse.WrmsseError, match="daily"):
        wrmsse.score(panel, forecast, object(), levels=((),), weight_col="revenue")


def test_unexpected_cached_action_failure_releases_wrmsse_frames(monkeypatch) -> None:
    """Framework exceptions must release caches just like each named validation error."""
    panel_, forecast_ = _tiny_case()
    cached: list[DataFrame] = []
    frame_type = type(spark().range(1))
    real_count = frame_type.count

    def fail_first_cached_count(frame: DataFrame) -> int:
        if frame.is_cached and not cached:
            cached.append(frame)
            raise RuntimeError("injected cached action failure")
        return real_count(frame)

    monkeypatch.setattr(frame_type, "count", fail_first_cached_count)
    with pytest.raises(RuntimeError, match="injected cached action failure"):
        wrmsse.score(
            panel_, forecast_, spark(), levels=((), ("grp",)), weight_col="revenue"
        )
    assert len(cached) == 1
    assert not cached[0].is_cached, "aborted WRMSSE retained its series cache"


@pytest.mark.parametrize("fail_level", [False, True])
def test_grid_caches_release_before_hierarchy_scoring(monkeypatch, fail_level) -> None:
    """Failure cleanup must not extend grid-cache lifetimes through every hierarchy level."""
    panel_, forecast_ = _tiny_case()
    frames: list[DataFrame] = []
    frame_type = type(spark().range(1))
    real_cache = frame_type.cache
    real_level_score = wrmsse._level_score

    def capture_cache(frame: DataFrame) -> DataFrame:
        frames.append(real_cache(frame))
        return frames[-1]

    def check_lifetimes(*args, **kwargs):
        assert len(frames) == 3
        assert not frames[0].is_cached, "series cache outlived grid validation"
        assert not frames[1].is_cached, "expected-grid cache outlived grid validation"
        assert frames[2].is_cached, "predictions must remain cached for hierarchy scoring"
        if fail_level:
            raise RuntimeError("injected hierarchy scoring failure")
        return real_level_score(*args, **kwargs)

    monkeypatch.setattr(frame_type, "cache", capture_cache)
    monkeypatch.setattr(wrmsse, "_level_score", check_lifetimes)
    if fail_level:
        with pytest.raises(RuntimeError, match="injected hierarchy scoring failure"):
            wrmsse.score(panel_, forecast_, spark(), levels=((), ("grp",)), weight_col="revenue")
    else:
        result = wrmsse.score(
            panel_, forecast_, spark(), levels=((), ("grp",)), weight_col="revenue"
        )
        assert result.n_series_dropped == {"total": 0, "grp": 0}
    assert not any(frame.is_cached for frame in frames)


@pytest.mark.parametrize("origin,horizon,n_series", [
    ("20240101", 2, 1), ("2024-01-01", True, 1),
    ("2024-01-01", 0, 1), ("2024-01-01", -1, 1),
    ("2024-01-01", 1, True), ("2024-01-01", 1, 1.0),
    ("2024-01-01", 1, 0), ("2024-01-01", 1, -1),
])
def test_published_wrmsse_refuses_noncanonical_span_before_reading_data(
    origin, horizon, n_series
) -> None:
    panel = Panel("/not-read", "id", "date", "qty")
    forecast = Forecast(
        "candidate", "/not-read", origin, horizon, n_series, {"emitted": int(horizon)}
    )
    with pytest.raises(wrmsse.WrmsseError, match="canonical|horizon"):
        wrmsse.score(panel, forecast, object(), levels=((),), weight_col="revenue")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
