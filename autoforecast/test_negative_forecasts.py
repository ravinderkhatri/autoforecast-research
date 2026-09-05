"""Finite signed predictions use the same Spark grid, metrics and ranking paths."""

import datetime as dt
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoforecast.hierarchy_metric import HierarchyMetric
from autoforecast.panel import Forecast, Panel
from autoforecast.ranking import _validated_source_frame, common_cells
from autoforecast.score import Cell, score
from autoforecast.score_spark import score_forecast
from autoforecast.test_study_tuning import spark as _spark
from autoforecast.wrmsse import score as wrmsse_score
from provenance.lineage import directory_digest


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def test_signed_forecasts_match_python_spark_hierarchy_and_ranking(spark, tmp_path):
    import numpy as np

    history = np.array([[0., 1., 3., 2.], [1., 2., 2., 4.]])
    target = np.array([[0., 1.], [2., 3.]])
    predictions = np.array([[-2., -6.7e-15], [0., 3.]])
    panel_path, forecast_path = tmp_path / "panel.parquet", tmp_path / "forecasts"
    forecast_path.mkdir()
    actual = np.concatenate((history, target), axis=1)
    pq.write_table(pa.Table.from_pylist([
        {"id": sid, "date": dt.date(2024, 1, step + 1), "qty": float(value),
         "revenue": float(value)}
        for row, sid in enumerate(("a", "b")) for step, value in enumerate(actual[row])
    ]), panel_path)
    pq.write_table(pa.Table.from_pylist([
        {"series_id": sid, "date": dt.date(2024, 1, step + 5), "yhat": float(value)}
        for row, sid in enumerate(("a", "b")) for step, value in enumerate(predictions[row])
    ]), forecast_path / "part.parquet")
    panel = Panel(panel_path, "id", "date", "qty", season_length=1)
    forecast = Forecast("signed", str(forecast_path), "2024-01-04", 2, 2, {"emitted": 4})
    before = directory_digest(forecast_path)
    forecast.validate()
    scored = score_forecast(panel, forecast, spark, missing_actual="no_data")
    reference = score([
        Cell("generic", "2024-01-04", "store", sid, str(step), float(value),
             float(target[row, step]), True)
        for row, sid in enumerate(("a", "b")) for step, value in enumerate(predictions[row])
    ])
    assert scored == reference
    metric = HierarchyMetric.build({"id": ("a", "b")}, history, np.ones_like(history),
                                   ((), ("id",)), weight_days=2)
    hierarchy = wrmsse_score(panel, forecast, spark, levels=((), ("id",)),
                             weight_col="revenue", weight_days=2)
    assert hierarchy.score == pytest.approx(metric.score(predictions, target)["wrmsse"])
    bottom = wrmsse_score(panel, forecast, spark, levels=(("id",),),
                          weight_col="revenue", weight_days=2)
    expected = ((5 / 11) * math.sqrt(2.5 / 2.5) + (6 / 11) * math.sqrt(2 / (5 / 3)))
    assert bottom.score == pytest.approx(expected)
    frame = _validated_source_frame(forecast, spark)
    assert sorted(row.yhat for row in frame.collect()) == sorted(predictions.flatten())
    _, count = common_cells([forecast, forecast], spark, tmp_path / "common")
    assert count == 4
    assert directory_digest(forecast_path) == before
