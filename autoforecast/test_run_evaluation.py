"""Functional check for the run-local candidate-comparison artifact."""

from __future__ import annotations

import json
import os
import pathlib

import pytest
from pyspark.sql import SparkSession

from autoforecast.panel import Forecast, Panel
from autoforecast.run_evaluation import EvaluationError, _value, compare_run_forecasts
from autoforecast.score import Cell
from autoforecast.score import score as score_cells
from autoforecast.wrmsse import Wrmsse


def test_value_keeps_cell_and_hierarchy_metric_families_distinct():
    """Structural metric routing: a hierarchy record cannot masquerade as point-error evidence."""
    point = score_cells([Cell("c", "2024-01-01", "s", "p", "2024-01-02", 3, 2, True)])
    hierarchy = Wrmsse(0.5, {"total": 0.5})
    assert _value(point, "rmse") == 1.0
    assert _value(point, "mae") == 1.0
    assert _value(point, "wape") == 0.5
    assert _value(hierarchy, "wrmsse") == 0.5
    for metric in ("rmse", "mae", "wape"):
        with pytest.raises(EvaluationError, match="hierarchy score"):
            _value(hierarchy, metric)
    with pytest.raises(EvaluationError, match="cell score"):
        _value(point, "wrmsse")
    zero_volume = score_cells([
        Cell("c", "2024-01-01", "s", "p", "2024-01-02", 1, 0, True)
    ])
    assert _value(zero_volume, "wape") is None


@pytest.mark.skipif(not os.environ.get("JAVA_HOME"), reason="Spark requires JAVA_HOME")
def test_completed_forecasts_get_an_honest_common_cell_comparison(tmp_path: pathlib.Path) -> None:
    spark = (SparkSession.builder.master("local[1]").appName("test_run_evaluation")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    try:
        panel_path = tmp_path / "panel"
        spark.createDataFrame([
            ("a", "2024-01-01", 5.0), ("a", "2024-01-02", 7.0),
            ("a", "2024-01-03", 9.0), ("b", "2024-01-01", 1.0),
            ("b", "2024-01-02", 2.0), ("b", "2024-01-03", 3.0),
        ], ["id", "date", "qty"]).write.parquet(str(panel_path))
        forecasts: dict[str, Forecast] = {}
        for name, values in {
            "accurate": [("a", "2024-01-02", 7.0), ("a", "2024-01-03", 9.0),
                         ("b", "2024-01-02", 2.0), ("b", "2024-01-03", 3.0)],
            "inaccurate": [("a", "2024-01-02", 0.0), ("a", "2024-01-03", 0.0),
                           ("b", "2024-01-02", 0.0), ("b", "2024-01-03", 0.0)],
        }.items():
            path = tmp_path / name
            spark.createDataFrame(values, ["series_id", "date", "yhat"]).write.parquet(str(path))
            forecasts[name] = Forecast(name, str(path), "2024-01-01", 2, 2, {"emitted": 4})
        artifact = compare_run_forecasts(
            Panel(str(panel_path), "id", "date", "qty"), forecasts, spark,
            missing_actual="no_data", metric="rmse", output_dir=tmp_path / "evaluation",
        )
    finally:
        spark.stop()
    record = json.loads(artifact.read_text(encoding="utf-8"))
    assert record["ranking"]["order"] == ["accurate", "inaccurate"]
    assert record["ranking"]["n_common_cells"] == 4
    assert artifact.with_suffix(".html").is_file()


@pytest.mark.skipif(not os.environ.get("JAVA_HOME"), reason="Spark requires JAVA_HOME")
def test_m5_wrmsse_comparison_requires_and_scores_the_complete_hierarchy(
    tmp_path: pathlib.Path,
) -> None:
    """A governed M5 replay publishes full-grid WRMSSE rather than a point-metric proxy."""
    spark = (SparkSession.builder.master("local[1]").appName("test_run_evaluation_wrmsse")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    try:
        panel_path = tmp_path / "panel"
        rows = []
        values = {
            "a": ("CA", "CA_1", "FOODS", "FOODS_1", "FOODS_1_001", (1, 2, 4, 3, 5, 7)),
            "b": ("TX", "TX_1", "HOBBIES", "HOBBIES_1", "HOBBIES_1_001", (2, 3, 2, 4, 8, 4)),
        }
        for series_id, (state, store, category, department, item, history) in values.items():
            for day, quantity in enumerate(history, start=1):
                rows.append((
                    series_id, f"2024-01-{day:02d}", float(quantity), float(quantity * 2),
                    state, store, category, department, item,
                ))
        spark.createDataFrame(
            rows,
            ["id", "date", "qty", "revenue", "state_id", "store_id", "cat_id", "dept_id", "item_id"],
        ).write.parquet(str(panel_path))
        forecast_path = tmp_path / "forecast"
        spark.createDataFrame(
            [("a", "2024-01-05", 5.0), ("a", "2024-01-06", 7.0),
             ("b", "2024-01-05", 8.0), ("b", "2024-01-06", 4.0)],
            ["series_id", "date", "yhat"],
        ).write.parquet(str(forecast_path))
        forecast = Forecast("perfect", str(forecast_path), "2024-01-04", 2, 2, {"emitted": 4})
        artifact = compare_run_forecasts(
            Panel(str(panel_path), "id", "date", "qty"),
            {"perfect": forecast},
            spark,
            missing_actual="no_data",
            metric="wrmsse",
            output_dir=tmp_path / "evaluation",
        )
    finally:
        spark.stop()
    record = json.loads(artifact.read_text(encoding="utf-8"))
    assert record["comparison_kind"] == "m5_full_grid_wrmsse"
    assert record["candidates"][0]["score"]["wrmsse"] == 0.0
    assert record["ranking"]["n_common_cells"] == 4
    assert "WRMSSE" in artifact.with_suffix(".html").read_text(encoding="utf-8")
