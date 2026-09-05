"""Tests for the global shallow model, on REAL M5 through Spark.

The one that matters most is the leak test: features are built by the same function for training
and for prediction, and the whole design rests on nothing in them seeing past the origin. That is
checked by running the adapter twice -- once against the full panel, once against a panel
physically truncated at the origin -- and requiring identical forecasts.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters.shallow import NAME, _default_train_origins, fit_predict
from autoforecast.panel import Panel

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
ORIGIN = "2016-04-24"
HORIZON = 7

_SPARK: SparkSession | None = None


def _require_real_m5_store() -> None:
    if not STORE.exists():
        pytest.skip(f"real M5 Parquet fixture is not present: {STORE}")


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        for var in ("JAVA_HOME",):
            if not os.environ.get(var):
                raise RuntimeError(f"{var} unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_shallow")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "6g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def _panel_contract(path: pathlib.Path = STORE) -> Panel:
    return Panel(path=path, series_col="id", time_col="date", target_col="qty",
                 exposure_col="observation_state", season_length=7)


def panel(path: pathlib.Path = STORE) -> Panel:
    if path == STORE:
        _require_real_m5_store()
    return _panel_contract(path).validate()


def run(p: Panel | None = None, **kw):
    d = tempfile.mkdtemp()
    fc = fit_predict(p or panel(), kw.pop("origin", ORIGIN), kw.pop("horizon", HORIZON),
                     # 50k forces the SAMPLING path. CA_1 produces ~170k training rows, so the
                     # earlier 400k cap meant no sampling happened and the determinism test below
                     # was checking a code path the real run does not take.
                     spark=spark(), out_dir=d, max_train_rows=50_000, **kw)
    return fc, spark().read.parquet(d)


def test_importing_an_adapter_does_not_drag_in_every_heavy_dependency() -> None:
    """A regression with no assertion and no traceback: adding `ours` (torch) to the adapters
    package __init__ made every LightGBM run inherit torch, and shallow's training died with
    SIGSEGV. Measured -- it completes normally when torch is absent and segfaults when present;
    both ship their own OpenMP runtime and two in one process is a known hazard.

    It also made the optional extras compulsory: `zoo` needs autogluon and `tft` needs
    pytorch-forecasting, and a client using only the naive adapter should need neither."""
    import subprocess
    probe = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r});"
        "import autoforecast.adapters;"
        "heavy = [m for m in ('torch','lightgbm','autogluon','pytorch_forecasting')"
        "         if m in sys.modules];"
        "print(heavy)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", (
        f"importing autoforecast.adapters loaded {out.stdout.strip()}; on this platform torch "
        "alongside lightgbm segfaults LightGBM training, with no assertion to catch it")


def test_fit_predict_accounts_for_every_cell_on_real_m5() -> None:
    fc, _ = run()
    assert fc.adapter == NAME
    assert fc.n_series == 3049
    assert sum(fc.disposition.values()) == fc.n_series * HORIZON


def test_a_training_origin_at_or_after_the_forecast_origin_is_refused() -> None:
    """A training window that includes the period being predicted is the leak this argument
    exists to make visible. Refused loudly rather than left to a reviewer."""
    for bad in (ORIGIN, "2016-05-01"):
        try:
            fit_predict(
                _panel_contract(), ORIGIN, HORIZON, spark=spark(), out_dir="/tmp/never",
                train_origins=("2016-01-03", bad),
            )
        except ValueError as e:
            assert "at or after the forecast origin" in str(e); continue
        raise AssertionError(f"trained on {bad}, which is not before {ORIGIN}")


def test_default_training_origins_are_before_the_origin_and_do_not_overlap() -> None:
    """Overlapping windows repeat the same actuals across training rows, which inflates the
    effective sample size without changing anything visible."""
    origins = _default_train_origins(ORIGIN, HORIZON)
    assert len(origins) == 8
    assert all(o < ORIGIN for o in origins)
    assert origins == tuple(sorted(origins)), "not in chronological order"
    import datetime as dt
    gaps = {(dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days
            for a, b in zip(origins, origins[1:], strict=False)}
    assert gaps == {HORIZON}, f"windows are not one horizon apart: {gaps}"


def test_no_feature_can_see_past_the_origin() -> None:
    """THE leak test. Run against the full panel and against one physically truncated at the
    origin. If any feature reached forward, the two forecasts would differ."""
    _require_real_m5_store()
    with tempfile.TemporaryDirectory() as d:
        truncated = pathlib.Path(d) / "cut"
        (spark().read.parquet(str(STORE))
         .filter(F.col("date") <= F.lit(ORIGIN))
         .write.mode("overwrite").parquet(str(truncated)))
        _, full_out = run(panel(), seed=7)
        _, cut_out = run(panel(truncated), seed=7)
        joined = (full_out.select("series_id", "date", F.col("yhat").alias("a"))
                  .join(cut_out.select("series_id", "date", F.col("yhat").alias("b")),
                        ["series_id", "date"]))
        assert joined.count() == full_out.count(), "the truncated run lost cells"
        differing = joined.filter(F.abs(F.col("a") - F.col("b")) > 1e-9).count()
        assert differing == 0, (
            f"{differing} cells differ when the panel is truncated at the origin, so a feature "
            "is reading data that does not exist at forecast time")
        shutil.rmtree(truncated, ignore_errors=True)


def test_training_is_reproducible_when_the_input_layout_changes() -> None:
    """Spark promises no row order, and LightGBM's bagging selects by row POSITION -- so the same
    training rows in a different sequence train a different model. Found when a leak test compared
    the full panel against one truncated at the origin: the training frames were identical as sets
    and all 21,343 forecasts still differed. The fix is a deterministic sort in `_to_driver`; this
    is the test that keeps it, by repartitioning the panel rather than by changing its content."""
    _require_real_m5_store()
    with tempfile.TemporaryDirectory() as d:
        shuffled = pathlib.Path(d) / "shuffled"
        (spark().read.parquet(str(STORE)).repartition(17, "date")
         .write.mode("overwrite").parquet(str(shuffled)))
        _, a = run(panel(), seed=3)
        _, b = run(panel(shuffled), seed=3)
        joined = (a.select("series_id", "date", F.col("yhat").alias("a"))
                  .join(b.select("series_id", "date", F.col("yhat").alias("b")),
                        ["series_id", "date"]))
        differing = joined.filter(F.abs(F.col("a") - F.col("b")) > 1e-9).count()
        assert differing == 0, (
            f"{differing} cells differ after repartitioning the same data, so the model depends "
            "on row order and a gate result cannot be reproduced")


def test_hurdle_model_emits_nonnegative_values_from_its_positive_parameterization() -> None:
    _, out = run()
    assert out.filter(F.col("yhat") < 0).count() == 0


def test_the_forecast_varies_by_day_of_week_unlike_a_flat_rate() -> None:
    """A shallow model that returned a constant per series would be Croston with more machinery.
    Day-of-week structure is the cheapest thing it should be adding over one."""
    _, out = run()
    spread = out.groupBy("series_id").agg(
        (F.max("yhat") - F.min("yhat")).alias("s")).agg(F.avg("s")).collect()[0][0]
    assert spread > 0.01, f"mean within-series spread is {spread}; the model is flat"


def test_a_non_positive_horizon_is_refused() -> None:
    try:
        fit_predict(_panel_contract(), ORIGIN, 0, spark=spark(), out_dir="/tmp/never")
    except ValueError as e:
        assert "horizon" in str(e); return
    raise AssertionError("accepted horizon=0")


def test_daily_adapter_refuses_weekly_panel_before_runtime_io() -> None:
    class ReadSentinel:
        @property
        def read(self):
            raise AssertionError("weekly panel reached LightGBM/Spark")

    weekly = Panel("/not-read", "id", "date", "qty", frequency="W")
    with pytest.raises(ValueError, match="daily"):
        fit_predict(
            weekly,
            ORIGIN,
            HORIZON,
            spark=ReadSentinel(),
            out_dir="/not-written",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))


def test_shallow_export_keeps_model_product_unchanged(tmp_path):
    import datetime as dt

    import numpy as np
    import pyarrow.parquet as pq

    from autoforecast.adapters.shallow import FEATURES, _predict

    class Predictor:
        def __init__(self, values):
            self.values = values

        def predict(self, frame):
            return np.array(self.values)
    features = spark().createDataFrame([
        {**dict.fromkeys(FEATURES, 1.0), "series_id": "a", "date": dt.date(2024, 1, day)}
        for day in (2, 3)
    ])
    result = _predict((Predictor([1., 1.]), Predictor([-2., -6.7e-15])), features,
                      Panel(tmp_path, "id", "date", "qty"), "2024-01-01", 2,
                      tmp_path / "forecast", spark())
    rows = sorted(pq.read_table(result.path).to_pylist(), key=lambda row: row["date"])
    assert [row["yhat"] for row in rows] == [-2., -6.7e-15]
