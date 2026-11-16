"""Classical model contracts on real classic Spark and local Spark Connect."""

import datetime as dt
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession
from statsforecast import StatsForecast

from autoforecast import resources, run_study
from autoforecast.adapters.classical import (
    DECLINED_NO_FORECAST,
    DECLINED_NO_HISTORY,
    DECLINED_SHORT_HISTORY,
    EMITTED,
    MODELS,
    forecast_with,
)
from autoforecast.panel import Panel
from autoforecast.study_registration import StudyRegistry, register_study
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_builtins import daily_study


@pytest.fixture(scope="module", params=["classic", "connect"])
def session(request):
    """Use real local servers; never silently skip a missing Connect dependency."""
    with pytest.MonkeyPatch.context() as env:
        # The local Connect server changes its worker cwd. Exercise this checkout's source,
        # not an older installed wheel or a relative PYTHONPATH interpreted in that worker cwd.
        root = str(pathlib.Path(__file__).resolve().parents[2])
        env.setenv("PYTHONPATH", os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
        env.setenv("PYSPARK_PYTHON", sys.executable)
        env.setenv("SPARK_LOCAL_IP", "127.0.0.1")
        builder = SparkSession.builder
        builder = (builder.remote("local[2]") if request.param == "connect"
                   else builder.master("local[2]"))
        spark = (builder.config("spark.sql.shuffle.partitions", "2")
                 .config("spark.sql.session.timeZone", "UTC")
                 .config("spark.driver.memory", "1g")
                 .getOrCreate())
        try:
            assert ("connect" in type(spark).__module__) == (request.param == "connect")
            if request.param == "connect":
                with pytest.raises(Exception, match="JVM_ATTRIBUTE_NOT_SUPPORTED"):
                    _ = spark.sparkContext
            # Workers were configured when Spark started. A managed client need not export this.
            env.delenv("PYSPARK_PYTHON")
            yield spark
        finally:
            spark.stop()


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("numeric_ids", [False, True])
def test_native_classical_forecast_matches_pandas_and_accounts_for_declines(
    session, tmp_path, monkeypatch, model, numeric_ids,
):
    """Unordered, zero, gapped and stale histories retain their exact date-key semantics."""
    start = dt.date(2024, 1, 1)
    origin = start + dt.timedelta(days=41)
    horizon = 5
    rows = []
    for series in range(6):
        for day in range(47):
            state = "exact" if day % 6 else "lower_bound"
            if series == 2 and day % 5 == 0:
                state = "unobserved"
            if series == 3 and day > 39:  # last eligible date is two days before origin
                state = "unobserved"
            if series == 4 and day != 41:
                state = "unobserved"
            if series == 5:
                state = "unobserved"
            value = (0.0 if series == 1 else float(3 + day % 7) if day % 4 else 0.0)
            if state == "unobserved" or day > 41:
                value = 1e8  # any leakage changes predictions conspicuously
            rows.append({"id": series if numeric_ids else f"S{series}",
                         "date": start + dt.timedelta(days=day), "qty": value,
                         "observation_state": state})
    path = tmp_path / "panel.parquet"
    pq.write_table(pa.Table.from_pylist(rows[::-1]), path)
    panel = Panel(path, "id", "date", "qty", exposure_col="observation_state", season_length=7)
    # Remote execution must not estimate worker resources from the client's machine.
    monkeypatch.setattr(resources, "detect", lambda: pytest.fail("client host was sized"))
    result = forecast_with(model, panel, origin.isoformat(), horizon,
                           spark=session, out_dir=tmp_path / "out")
    assert result.n_series == 6
    assert result.disposition == {EMITTED: 18, DECLINED_NO_FORECAST: 2,
                                  DECLINED_SHORT_HISTORY: 5, DECLINED_NO_HISTORY: 5}
    actual = pq.read_table(result.path).to_pandas().sort_values(["series_id", "date"])
    assert pd.api.types.is_integer_dtype(actual.series_id) == numeric_ids
    # Independent reference: fit the full eligible pandas panel with StatsForecast itself.
    eligible = pd.DataFrame([r for r in rows if r["date"] <= origin
                             and r["observation_state"] in {"exact", "lower_bound"}
                             and r["id"] not in ([4, 5] if numeric_ids else ["S4", "S5"])])
    reference_input = eligible.rename(columns={"id": "unique_id", "date": "ds", "qty": "y"})
    reference_input["ds"] = pd.to_datetime(reference_input.ds)
    fitted = MODELS[model].build(7)
    expected = StatsForecast(models=[fitted], freq="D", n_jobs=1).forecast(
        h=horizon, df=reference_input[["unique_id", "ds", "y"]])
    expected = expected[expected.ds > pd.Timestamp(origin)].sort_values(["unique_id", "ds"])
    assert actual.series_id.tolist() == expected.unique_id.tolist()
    assert pd.to_datetime(actual.date).tolist() == expected.ds.tolist()
    np.testing.assert_allclose(actual.yhat, expected[str(fitted)], rtol=0, atol=1e-12)
    assert np.isfinite(actual.yhat).all()


def test_no_eligible_series_refuses_without_output(session, tmp_path):
    path = tmp_path / "short.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id": "a", "date": dt.date(2024, 1, 1), "qty": 1.0}
    ]), path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="no series has 2 eligible observations"):
        forecast_with("croston", Panel(path, "id", "date", "qty", season_length=1),
                      "2024-01-01", 2, spark=session, out_dir=output)
    assert not output.exists()


@pytest.fixture(scope="module")
def registered(session, tmp_path_factory):
    config = daily_study(tmp_path_factory.mktemp("classical-study"))
    registry = StudyRegistry.from_config(config, spark=session)
    return registry, register_study(config, registry=registry)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_named_classical_study_retains_model_and_reloads(registered, model):
    registry, study_id = registered
    result = run_study(study_info={"study_id": study_id, "create_study": False,
                                  "publish_results": False}, model=model, registry=registry)
    assert result.status == "completed"
    assert result.validation["score"]["population"]["forecasted_series"] == 3
    assert result.testing["score"]["population"]["forecasted_series"] == 3
    model_path = result.path / result.model_links["test"]["path"]
    record = json.loads((model_path / "panel.json").read_text())
    assert (record["model"], record["season_length"], record["origin"]) == (
        model, 7, "2024-04-23")
    assert (model_path / "history").is_dir()
    assert load_forecast_workflow(result.path).testing_metrics == result.testing_metrics
