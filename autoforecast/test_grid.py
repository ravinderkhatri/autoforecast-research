"""Adversarial tests for the model-neutral evaluation-grid contract.

The grid governs every adapter and metric, so these cases use synthetic panel rows rather than
the custom model's exposure implementation. Run with the repository's Java/Python environment:

    JAVA_HOME=... PYSPARK_PYTHON=... python3 autoforecast/test_grid.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast import grid

ORIGIN = "2024-01-03"
HORIZON = 3

_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[2]").appName("test_grid")
                  .config("spark.sql.shuffle.partitions", "2")
                  .config("spark.driver.memory", "2g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def source() -> pathlib.Path:
    path = pathlib.Path(tempfile.mkdtemp()) / "panel"
    rows = [
        ("a", "2024-01-01", False),
        ("a", "2024-01-04", True),
        ("a", "2024-01-05", False),
        ("a", "2024-01-06", True),
        ("b", "2024-01-01", False),
        ("b", "2024-01-04", False),
        ("b", "2024-01-05", True),
        ("b", "2024-01-06", False),
    ]
    (spark().createDataFrame(rows, "id string, day string, eligible boolean")
     .withColumn("day", F.to_date("day"))
     .write.parquet(str(path)))
    return path


def test_build_without_eligibility_materialises_the_full_cross_product() -> None:
    path = grid.build(source(), ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
                      series_col="id", time_col="day")
    keys = {(row["series_id"], str(row["date"]))
            for row in spark().read.parquet(str(path)).collect()}
    assert len(keys) == 2 * HORIZON
    assert ("a", "2024-01-04") in keys and ("b", "2024-01-06") in keys


def test_build_applies_effective_dated_eligibility_per_cell() -> None:
    path = grid.build(source(), ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
                      series_col="id", time_col="day", eligible_col="eligible")
    keys = {(row["series_id"], str(row["date"]))
            for row in spark().read.parquet(str(path)).collect()}
    assert keys == {
        ("a", "2024-01-04"), ("a", "2024-01-06"), ("b", "2024-01-05")}


def test_build_materialises_the_declared_integral_day_cadence() -> None:
    path = grid.build(
        source(), ORIGIN, 2, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="day", frequency="2D",
    )
    dates = {str(row["date"]) for row in spark().read.parquet(str(path)).collect()}
    assert dates == {"2024-01-05", "2024-01-07"}


def test_summarise_reports_density_of_a_ragged_grid() -> None:
    path = grid.build(source(), ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
                      series_col="id", time_col="day", eligible_col="eligible")
    stats = grid.summarise(path, spark()).collect()[0]
    assert (stats["n_cells"], stats["n_series"], stats["n_days"]) == (3, 2, 3)
    assert stats["density"] == 0.5


def test_content_hash_is_logical_not_a_parquet_layout_hash() -> None:
    first = grid.build(source(), ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
                       series_col="id", time_col="day", eligible_col="eligible")
    second = pathlib.Path(tempfile.mkdtemp()) / "rewritten"
    spark().read.parquet(str(first)).repartition(2).write.parquet(str(second))
    assert grid.content_hash(first, spark()) == grid.content_hash(second, spark())


def test_content_hash_refuses_duplicate_or_invalid_keys() -> None:
    duplicate = pathlib.Path(tempfile.mkdtemp()) / "duplicate"
    spark().createDataFrame(
        [("a", "2024-01-04"), ("a", "2024-01-04")],
        "series_id string, date string",
    ).write.parquet(str(duplicate))
    try:
        grid.content_hash(duplicate, spark())
    except grid.GridError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("hashed a grid with duplicate keys")


def test_agrees_with_detects_key_set_drift_in_both_directions() -> None:
    first = grid.build(source(), ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
                       series_col="id", time_col="day", eligible_col="eligible")
    changed = pathlib.Path(tempfile.mkdtemp()) / "changed"
    rows = [("a", "2024-01-04"), ("b", "2024-01-05"), ("x", "2024-01-06")]
    (spark().createDataFrame(rows, "series_id string, date string")
     .withColumn("date", F.to_date("date")).write.parquet(str(changed)))
    assert grid.agrees_with(first, changed, spark()) == (1, 1)


def test_build_refuses_an_empty_grid_and_non_positive_horizon() -> None:
    empty = pathlib.Path(tempfile.mkdtemp()) / "empty"
    (spark().createDataFrame([("a", "2024-01-04", False)],
                             "id string, day string, eligible boolean")
     .withColumn("day", F.to_date("day")).write.parquet(str(empty)))
    for call, expected in (
        (lambda: grid.build(empty, ORIGIN, HORIZON, spark(), tempfile.mkdtemp(),
                            series_col="id", time_col="day", eligible_col="eligible"),
         "no eligible cell"),
        (lambda: grid.build(empty, ORIGIN, 0, spark(), tempfile.mkdtemp(),
                            series_col="id", time_col="day"), "horizon"),
    ):
        try:
            call()
        except grid.GridError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("built a grid from an invalid request")


def test_build_refuses_noncanonical_span_before_spark_io() -> None:
    """Identity-invalid requests must not reach an expensive or stateful Spark boundary."""
    for origin, horizon in ((ORIGIN, True), ("20240103", HORIZON)):
        try:
            grid.build(
                "/not-read",
                origin,
                horizon,
                object(),
                "/not-written",
                series_col="id",
                time_col="day",
            )
        except grid.GridError as exc:
            assert "horizon" in str(exc) or "canonical" in str(exc)
        else:
            raise AssertionError(f"accepted origin={origin!r}, horizon={horizon!r}")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
    if _SPARK is not None:
        _SPARK.stop()
