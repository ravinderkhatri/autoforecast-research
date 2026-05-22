"""Tests for the statsforecast adapters, on REAL M5 through Spark.

The sharp edge these exist for: statsforecast forecasts forward from each SERIES' OWN last
observation, not from the origin we asked about. A series whose eligible history ended months ago
comes back with predictions for the wrong dates, and nothing in statsforecast notices.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters import classical
from autoforecast.adapters.classical import MODELS, forecast_with
from autoforecast.panel import Panel

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
ORIGIN = "2016-04-24"
# Early enough that some real M5 products have only days of eligible history. This is where the
# Phase 2 run died: AutoETS raises NotImplementedError("tiny datasets") inside a Spark worker on
# a series it considers too short, which killed the whole job.
SHORT_HISTORY_ORIGIN = "2016-01-03"
HORIZON = 7

_SPARK: SparkSession | None = None
_SMALL: pathlib.Path | None = None

# MEASURED on this machine: AutoETS fits 3,049 CA_1 series in 120s, against Croston's 16s -- it
# runs a per-series optimisation and there is no way around that. Four AutoETS calls made this
# file take over ten minutes, which is a suite people stop running.
#
# So the AutoETS cases use a 200-SERIES SLICE OF THE SAME REAL DATA. Everything under test here is
# a per-series property -- the date window, the sign, the history floor -- so fewer series tests
# the same thing. Croston and the whole-store cases stay on the full store.
SMALL_SERIES = 200


def _require_real_m5_store() -> None:
    if not STORE.exists():
        pytest.skip(f"real M5 Parquet fixture is not present: {STORE}")


def spark() -> SparkSession:
    """One local session. PYSPARK_PYTHON matters here in a way it does not for the naive adapter:
    these run a Python UDF per series, so the EXECUTOR needs fugue and statsforecast."""
    global _SPARK
    if _SPARK is None:
        for var in ("JAVA_HOME", "PYSPARK_PYTHON"):
            if not os.environ.get(var):
                raise RuntimeError(f"{var} unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_classical")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def small_store() -> pathlib.Path:
    """A real 200-series slice of CA_1, written once and reused across the AutoETS cases."""
    global _SMALL
    _require_real_m5_store()
    if _SMALL is None:
        d = pathlib.Path(tempfile.mkdtemp()) / "small"
        full = spark().read.parquet(str(STORE))
        keep = [r["id"] for r in
                full.select("id").distinct().orderBy("id").limit(SMALL_SERIES).collect()]
        full.filter(F.col("id").isin(*keep)).write.mode("overwrite").parquet(str(d))
        _SMALL = d
    return _SMALL


def _panel_contract(**over: object) -> Panel:
    kw: dict[str, object] = dict(path=STORE, series_col="id", time_col="date", target_col="qty",
                                 exposure_col="observation_state", season_length=7)
    kw.update(over)
    return Panel(**kw)  # type: ignore[arg-type]


def panel(**over: object) -> Panel:
    described = _panel_contract(**over)
    if pathlib.Path(described.path) == STORE:
        _require_real_m5_store()
    return described.validate()


_RUNS: dict[tuple, tuple] = {}


def cached_run(model: str, **kw):
    """`run`, memoised on its arguments.

    Two cases need AutoETS at SHORT_HISTORY_ORIGIN on the full store, which costs ~120s each.
    They stay separate tests -- one asserts the decline reason exists, the other that a lower
    floor declines fewer -- and share the fit rather than repeating it.
    """
    key = (model, tuple(sorted(kw.items(), key=lambda kv: str(kv))))
    if key not in _RUNS:
        _RUNS[key] = run(model, **dict(kw))
    return _RUNS[key]


def run(model: str, **kw):
    d = tempfile.mkdtemp()
    p = panel(path=kw.pop("path", STORE))
    fc = forecast_with(model, p, kw.pop("origin", ORIGIN), kw.pop("horizon", HORIZON),
                       spark=spark(), out_dir=d)
    return fc, spark().read.parquet(d)


def test_forecast_with_croston_accounts_for_every_cell_on_real_m5() -> None:
    fc, _ = run("croston")
    assert fc.adapter == "croston"
    assert fc.n_series == 3049
    assert sum(fc.disposition.values()) == fc.n_series * HORIZON


def test_every_emitted_date_is_inside_the_window_we_asked_for() -> None:
    """The edge the adapter exists to close. statsforecast counts forward from each series' last
    observation; if that is not the origin, the dates come back wrong and nothing complains."""
    import datetime as dt
    fc, out = run("croston")
    d0 = dt.date.fromisoformat(ORIGIN)
    lo, hi = d0 + dt.timedelta(days=1), d0 + dt.timedelta(days=HORIZON)
    bad = out.filter((F.col("date") < F.lit(lo.isoformat()))
                     | (F.col("date") > F.lit(hi.isoformat()))).count()
    assert bad == 0, f"{bad} emitted cells fall outside {lo}..{hi}"
    assert out.select("date").distinct().count() == HORIZON


def test_croston_is_flat_across_the_horizon_which_is_what_croston_is() -> None:
    """Not a quirk to fix: Croston predicts a constant rate, so every horizon day is identical.
    Asserting it means a future change that makes it vary is noticed rather than welcomed."""
    _, out = run("croston")
    spread = out.groupBy("series_id").agg(
        (F.max("yhat") - F.min("yhat")).alias("spread")).agg(F.max("spread")).collect()[0][0]
    assert spread < 1e-6, f"Croston varied by {spread} across the horizon"


def test_no_emitted_forecast_is_negative() -> None:
    """Units sold cannot be negative. AutoETS produces negative point forecasts on near-zero
    series, and a negative demand forecast is not a small error -- it is a nonsense the metric
    would happily average."""
    for model in sorted(MODELS):
        _, out = run(model, path=small_store() if model == "auto_ets" else STORE)
        assert out.filter(F.col("yhat") < 0).count() == 0, f"{model} emitted a negative forecast"


def test_auto_ets_runs_on_real_intermittent_series() -> None:
    fc, _ = run("auto_ets", path=small_store())
    assert fc.adapter == "auto_ets"
    assert fc.n_emitted > 0
    assert fc.n_series == SMALL_SERIES
    assert sum(fc.disposition.values()) == fc.n_series * HORIZON


def test_a_series_too_short_for_the_model_is_declined_rather_than_crashing() -> None:
    """statsforecast REFUSES a series it considers too short, and the refusal surfaces inside a
    Spark worker and kills the job. Each model declares its own floor, so that becomes a declined
    cell with a reason."""
    from autoforecast.adapters.classical import DECLINED_SHORT_HISTORY
    # Horizon 7, not 28: the floor is about HISTORY length, so a longer horizon would only make
    # AutoETS fit the same 3,049 series more slowly. The suite has to stay runnable.
    fc, _ = cached_run("auto_ets", origin=SHORT_HISTORY_ORIGIN, path=STORE)
    assert fc.disposition.get(DECLINED_SHORT_HISTORY, 0) > 0, (
        f"nothing declined for short history at {SHORT_HISTORY_ORIGIN}, where real M5 products "
        f"have days of history: {fc.disposition}")
    assert sum(fc.disposition.values()) == fc.n_series * HORIZON


def test_a_lower_floor_declines_fewer_series() -> None:
    """Croston needs two observations; AutoETS needs two seasonal cycles. The difference must show
    up as fewer declines, or the floors are not being applied per model."""
    from autoforecast.adapters.classical import DECLINED_SHORT_HISTORY
    ets, _ = cached_run("auto_ets", origin=SHORT_HISTORY_ORIGIN, path=STORE)
    cro, _ = cached_run("croston", origin=SHORT_HISTORY_ORIGIN, path=STORE)
    assert cro.disposition.get(DECLINED_SHORT_HISTORY, 0) \
        < ets.disposition.get(DECLINED_SHORT_HISTORY, 0)
    assert cro.n_emitted > ets.n_emitted


def test_the_season_length_comes_from_the_panel_not_from_a_constant() -> None:
    """AutoETS's season_length was hardcoded to 7, which is an estate assumption in generic code
   . A panel declaring a different season must change the model's floor."""
    from autoforecast.adapters.classical import MODELS
    assert MODELS["auto_ets"].min_history(7) != MODELS["auto_ets"].min_history(30)
    built = MODELS["auto_ets"].build(30)
    assert getattr(built, "season_length", None) == 30


def test_an_unknown_model_is_refused_by_name() -> None:
    try:
        forecast_with(
            "prophet", _panel_contract(), ORIGIN, HORIZON, spark=spark(), out_dir="/tmp/never"
        )
    except ValueError as e:
        assert "unknown model 'prophet'" in str(e) and "croston" in str(e); return
    raise AssertionError("accepted a model that is not registered")


def test_a_non_positive_horizon_is_refused() -> None:
    try:
        forecast_with("croston", _panel_contract(), ORIGIN, 0, spark=spark(), out_dir="/tmp/never")
    except ValueError as e:
        assert "horizon" in str(e); return
    raise AssertionError("accepted horizon=0")


def test_daily_adapter_refuses_weekly_panel_before_runtime_io() -> None:
    class ReadSentinel:
        @property
        def read(self):
            raise AssertionError("weekly panel reached StatsForecast/Spark")

    with pytest.raises(ValueError, match="daily"):
        forecast_with(
            "croston",
            _panel_contract(frequency="W"),
            ORIGIN,
            HORIZON,
            spark=ReadSentinel(),
            out_dir="/not-written",
        )


def test_write_failure_releases_cached_labelled_frame(tmp_path, monkeypatch) -> None:
    """A failed output write must release the adapter-owned Spark cache."""
    import datetime as dt

    source = tmp_path / "panel"
    rows = [("a", f"2024-01-{day:02d}", float(day)) for day in range(1, 4)]
    (spark().createDataFrame(rows, "id string, date string, qty double")
     .selectExpr("id", "to_date(date) as date", "qty")
     .write.parquet(str(source)))
    described = Panel(source, "id", "date", "qty", season_length=1)

    class FakeStatsForecast:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def forecast(self, *, h: int, df: DataFrame) -> DataFrame:
            assert h == 2 and df.select("unique_id").distinct().count() == 1
            return spark().createDataFrame(
                [
                    ("a", dt.date(2024, 1, 4), 1.0),
                    ("a", dt.date(2024, 1, 5), 1.0),
                ],
                "unique_id string, ds date, CrostonClassic double",
            )

    cached: list[DataFrame] = []
    frame_type = type(spark().range(1))
    writer_type = type(spark().range(1).write)
    real_cache = frame_type.cache

    def record_cache(frame: DataFrame) -> DataFrame:
        cached.append(real_cache(frame))
        return cached[-1]

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected forecast write failure")

    monkeypatch.setattr(classical, "StatsForecast", FakeStatsForecast)
    monkeypatch.setattr(frame_type, "cache", record_cache)
    monkeypatch.setattr(writer_type, "parquet", fail_write)
    with pytest.raises(RuntimeError, match="injected forecast write failure"):
        forecast_with(
            "croston",
            described,
            "2024-01-03",
            2,
            spark=spark(),
            out_dir=tmp_path / "forecast",
        )
    assert len(cached) == 1
    assert not cached[0].is_cached, "aborted classical call retained its labelled cache"


def test_a_missing_executor_interpreter_is_named_rather_than_left_to_spark() -> None:
    """Without PYSPARK_PYTHON the failure is a ModuleNotFoundError raised inside a Spark worker,
    which reads as a Spark fault. It cost real time once; the adapter now says what is wrong."""
    session = spark()  # start it BEFORE unsetting the variable it needs to start
    saved = os.environ.pop("PYSPARK_PYTHON")
    try:
        forecast_with(
            "croston", _panel_contract(), ORIGIN, HORIZON, spark=session, out_dir="/tmp/never"
        )
    except ValueError as e:
        assert "PYSPARK_PYTHON" in str(e) and "EXECUTOR" in str(e); return
    finally:
        os.environ["PYSPARK_PYTHON"] = saved
    raise AssertionError("ran without naming the executor interpreter requirement")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
