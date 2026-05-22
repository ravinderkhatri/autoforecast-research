"""Tests for the headroom estimate, on REAL M5 through Spark.

The estimate decides whether Phase 3 happens at all, so the cases here are mostly about the ways
it could be optimistic -- an oracle that fits the noise it is measuring, or one scored on a
different population from the model it is compared to.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast import headroom, ranking, registry
from autoforecast.adapters import register_defaults
from autoforecast.panel import Panel
from autoforecast.score import Score
from autoforecast.score_spark import score_forecast

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
ORIGIN = "2016-04-24"
HORIZON = 28

_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        for var in ("JAVA_HOME", "PYSPARK_PYTHON"):
            if not os.environ.get(var):
                raise RuntimeError(f"{var} unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_headroom")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def panel() -> Panel:
    if not STORE.is_dir():
        pytest.skip("public M5 Parquet is not installed")
    return Panel(path=STORE, series_col="id", time_col="date", target_col="qty",
                 exposure_col="observation_state", season_length=7).validate()


def fake(mae: float, n: int = 100) -> Score:
    """SYNTHETIC (structure): the share arithmetic and its guards, which no forecast produces."""
    return Score(n_scored=n, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
                 mae=mae, rmse=mae, bias=0.0, bias_pct_of_volume=0.0, mase=None,
                 sum_abs_error=mae * n, sum_squared_error=mae * mae * n,
                 sum_error=0.0, sum_actual=float(n), sum_naive_abs_error=0.0)


def test_the_oracle_is_not_registered_and_cannot_be_run_as_a_competitor() -> None:
    """It sees the held-out actuals. If registry.run could reach it, it would win every
    comparison it appeared in and the number would mean nothing."""
    register_defaults() if "seasonal_naive" not in registry.names() else None
    assert headroom.ORACLE not in registry.names()
    source = Panel(path=STORE, series_col="id", time_col="date", target_col="qty")
    try:
        registry.run(headroom.ORACLE, source, ORIGIN, HORIZON, spark=object(), out_dir="/tmp/n")
    except registry.RegistryError as e:
        assert "unknown adapter" in str(e); return
    raise AssertionError("the oracle is reachable through the registry")


def test_the_oracle_leaves_its_own_cell_out() -> None:
    """An oracle fitted on the cell it is scored on fits the noise it is meant to be measuring,
    which understates irreducible error and so OVERSTATES headroom -- the direction that argues
    for building a deep model. Checked against the real M5 window arithmetic."""
    d = tempfile.mkdtemp()
    fc = headroom.oracle_forecast(panel(), ORIGIN, HORIZON, spark=spark(), out_dir=d)
    pred = spark().read.parquet(fc.path)
    actual = spark().read.parquet(str(STORE)).select(
        F.col("id").alias("series_id"), F.col("date"), F.col("qty").cast("double").alias("y"))
    joined = pred.join(actual, ["series_id", "date"]).withColumn("dow", F.dayofweek("date"))
    # Recompute leave-one-out independently from the joined rows and require agreement.
    from pyspark.sql import Window
    w = Window.partitionBy("series_id", "dow")
    check = (joined.withColumn("_s", F.sum("y").over(w)).withColumn("_n", F.count(F.lit(1)).over(w))
             .withColumn("expected", (F.col("_s") - F.col("y")) / (F.col("_n") - 1)))
    assert check.filter(F.abs(F.col("yhat") - F.col("expected")) > 1e-9).count() == 0
    # And it must NOT equal the plain group mean, which is what including the cell would give.
    plain = joined.withColumn("mean_with_self", F.avg("y").over(w))
    assert plain.filter(F.abs(F.col("yhat") - F.col("mean_with_self")) > 1e-9).count() > 0, \
        "the oracle equals the group mean everywhere, so it is including its own cell"


def test_the_oracle_uses_the_panels_declared_period_grid() -> None:
    import datetime as dt

    root = pathlib.Path(tempfile.mkdtemp())
    panel_path = root / "weekly-panel"
    spark().createDataFrame(
        [
            ("series", dt.date(2020, 1, 8), 2.0),
            ("series", dt.date(2020, 1, 15), 6.0),
        ],
        "id string, date date, qty double",
    ).write.parquet(str(panel_path))
    source = Panel(
        panel_path, "id", "date", "qty", frequency="W", target_estimand="observed_count"
    ).validate()
    forecast = headroom.oracle_forecast(
        source, "2020-01-01", 2, spark=spark(), out_dir=root / "oracle")
    assert forecast.frequency == "W"
    rows = sorted((row["date"], row["yhat"])
                  for row in spark().read.parquet(forecast.path).collect())
    assert rows == [(dt.date(2020, 1, 8), 6.0), (dt.date(2020, 1, 15), 2.0)]


@pytest.mark.parametrize("origin,horizon", [(ORIGIN, True), ("20160424", HORIZON)])
def test_oracle_refuses_noncanonical_span_before_spark_io(origin, horizon) -> None:
    """Malformed identities must fail before the oracle reads held-out actuals."""
    source = Panel("/not-read", "id", "date", "qty")
    with pytest.raises(headroom.HeadroomError, match="horizon|canonical"):
        headroom.oracle_forecast(
            source, origin, horizon, spark=object(), out_dir="/not-written"
        )


def test_estimate_refuses_when_the_two_were_scored_on_different_populations() -> None:
    """A difference of MAE between different cell populations is partly a difference of
    population. The estimate must refuse rather than report it as headroom."""
    import datetime as dt

    out = pathlib.Path(tempfile.mkdtemp())
    origin = "2020-01-01"
    horizon = 8
    panel_path = out / "panel"
    spark().createDataFrame(
        [
            ("series", dt.date(2020, 1, 2) + dt.timedelta(days=day), float(day), "exact")
            for day in range(horizon)
        ],
        "id string, date date, qty double, observation_state string",
    ).write.parquet(str(panel_path))
    source = Panel(
        panel_path,
        "id",
        "date",
        "qty",
        exposure_col="observation_state",
    ).validate()
    try:
        headroom.estimate(
            source,
            "candidate",
            fake(1.0, n=7),
            (origin,),
            horizon,
            spark(),
            out / "runs",
            missing_actual="no_data",
        )
    except headroom.HeadroomError as e:
        assert "same cells" in str(e) or "compare two populations" in str(e); return
    raise AssertionError("compared an oracle and a model scored on different cells")


def test_headroom_on_real_m5_above_the_seasonal_naive_baseline() -> None:
    register_defaults() if "seasonal_naive" not in registry.names() else None
    out = pathlib.Path(tempfile.mkdtemp())
    fc = registry.run("seasonal_naive", panel(), ORIGIN, HORIZON, spark=spark(),
                      out_dir=str(out / "naive"))
    naive_score = score_forecast(panel(), fc, spark(), missing_actual="no_data")
    oracle_fc = headroom.oracle_forecast(panel(), ORIGIN, HORIZON, spark=spark(),
                                         out_dir=str(out / "oracle"))
    # Score both on the oracle's cells, which is the smaller population here.
    common = str(out / "common")
    spark().read.parquet(oracle_fc.path).select("series_id", "date").write.mode(
        "overwrite").parquet(common)
    h = headroom.estimate(panel(), "seasonal_naive",
                          score_forecast(panel(), fc, spark(), missing_actual="no_data",
                                         restrict_to=common),
                          (ORIGIN,), HORIZON, spark(), out, missing_actual="no_data",
                          restrict={ORIGIN: common})
    assert 0.0 <= h.reducible_share <= 1.0
    assert abs(h.reducible_share + h.irreducible_share - 1.0) < 1e-12
    assert h.n_scored > 0 and naive_score.n_scored > 0


def test_estimate_applies_zero_policy_to_oracle_omissions_on_the_full_grid() -> None:
    """A young series has no same-weekday oracle partner; Phase 2 must still remain runnable."""
    import datetime as dt

    root = pathlib.Path(tempfile.mkdtemp())
    origin = "2020-01-01"
    horizon = 8
    rows = [
        ("long", dt.date(2020, 1, 2) + dt.timedelta(days=i), float(i + 1), "exact")
        for i in range(horizon)
    ] + [("cold", dt.date(2020, 1, 2), 3.0, "exact")]
    panel_path = root / "panel"
    spark().createDataFrame(
        rows, "id string, date date, qty double, observation_state string"
    ).write.parquet(str(panel_path))
    source = Panel(panel_path, "id", "date", "qty", exposure_col="observation_state").validate()
    grid_path = root / "grid"
    spark().createDataFrame(
        [(sid, day) for sid, day, _qty, _state in rows],
        "series_id string, date date").write.parquet(str(grid_path))
    grid_hash = ranking.eligible_grid_content_hash(grid_path, spark())

    result = headroom.estimate(
        source, "candidate", fake(10.0, n=len(rows)), (origin,), horizon, spark(), root / "runs",
        missing_actual="no_data", restrict={origin: str(grid_path)},
        eligible_grids={origin: str(grid_path)}, eligible_grid_hashes={origin: grid_hash},
        missing_forecast_policy="zero")
    assert result.n_scored == len(rows)
    finalized = spark().read.parquet(
        str(root / "runs" / headroom.ORACLE / f"{origin}.canonical"))
    assert finalized.count() == len(rows)
    assert finalized.filter(F.col("forecast_source") == "fallback_zero").count() > 0


def test_a_beaten_oracle_reports_zero_headroom_rather_than_a_negative() -> None:
    """A model that uses covariates the oracle cannot see can beat it. That is not evidence that
    error is reducible, so it is reported as zero -- and `oracle_lost` says it happened."""
    h = headroom.Headroom(best_adapter="m", best_mae=1.0, oracle_mae=1.5, n_scored=10)
    assert h.reducible_share == 0.0 and h.irreducible_share == 1.0 and h.oracle_lost


def test_headroom_refuses_a_zero_mae() -> None:
    try:
        headroom.Headroom(best_adapter="m", best_mae=0.0, oracle_mae=0.0, n_scored=10)
    except headroom.HeadroomError as e:
        assert "is a leak, not a result" in str(e); return
    raise AssertionError("accepted a zero MAE and divided by it")


def test_headroom_refuses_an_empty_population() -> None:
    try:
        headroom.Headroom(best_adapter="m", best_mae=1.0, oracle_mae=0.5, n_scored=0)
    except headroom.HeadroomError as e:
        assert "no cells were scored" in str(e); return
    raise AssertionError("estimated headroom from nothing")


def test_headroom_refuses_non_finite_or_boolean_evidence() -> None:
    bad = (
        dict(best_mae=float("nan"), oracle_mae=0.5, n_scored=10),
        dict(best_mae=10**400, oracle_mae=0.5, n_scored=10),
        dict(best_mae=1.0, oracle_mae=float("inf"), n_scored=10),
        dict(best_mae=1.0, oracle_mae=0.5, n_scored=True),
    )
    for fields in bad:
        try:
            headroom.Headroom(best_adapter="m", **fields)
        except headroom.HeadroomError:
            continue
        raise AssertionError(f"accepted malformed headroom evidence: {fields!r}")


def test_as_record_is_json_safe() -> None:
    import json
    r = headroom.Headroom("m", 1.0, 0.6, 10).as_record()
    assert abs(json.loads(json.dumps(r))["reducible_share"] - 0.4) < 1e-12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
