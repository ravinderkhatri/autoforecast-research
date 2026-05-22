"""Adversarial tests for the model-neutral evaluation-grid contract.

The grid governs every adapter and metric, so these cases use synthetic panel rows rather than
the custom model's exposure implementation. Run with the repository's Java/Python environment:

    JAVA_HOME=... PYSPARK_PYTHON=... python3 autoforecast/test_grid.py
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast import grid
from autoforecast.frequency import parse_frequency

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
                  .config("spark.sql.session.timeZone", "Asia/Kolkata")
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


def hourly_source() -> pathlib.Path:
    path = pathlib.Path(tempfile.mkdtemp()) / "hourly-panel"
    rows = [
        ("a", dt.datetime(2024, 3, 1, 22), False),
        ("a", dt.datetime(2024, 3, 1, 23), True),
        ("a", dt.datetime(2024, 3, 2, 0), True),
        ("b", dt.datetime(2024, 3, 1, 22), False),
        ("b", dt.datetime(2024, 3, 1, 23), False),
        ("b", dt.datetime(2024, 3, 2, 0), True),
    ]
    spark().createDataFrame(
        rows, "id string, time timestamp_ntz, eligible boolean",
    ).write.parquet(str(path))
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


def test_build_preserves_two_hourly_keys_on_one_calendar_day() -> None:
    path = grid.build(
        hourly_source(), "2024-03-01T22:00:00", 2, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="time", frequency="H",
    )
    rows = spark().read.parquet(str(path)).collect()
    assert spark().conf.get("spark.sql.session.timeZone") == "Asia/Kolkata"
    assert spark().read.parquet(str(path)).schema["date"].dataType.typeName() == "timestamp_ntz"
    assert {(row["series_id"], row["date"]) for row in rows} == {
        ("a", dt.datetime(2024, 3, 1, 23)),
        ("a", dt.datetime(2024, 3, 2, 0)),
        ("b", dt.datetime(2024, 3, 1, 23)),
        ("b", dt.datetime(2024, 3, 2, 0)),
    }


def test_hourly_full_cross_product_only_requires_series_ids() -> None:
    source_path = pathlib.Path(tempfile.mkdtemp()) / "series"
    spark().createDataFrame([("a",), ("b",)], "id string").write.parquet(str(source_path))
    path = grid.build(
        source_path, "2024-03-01T22:00:00", 2, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="unused", frequency="H",
    )
    assert {(row["series_id"], row["date"])
            for row in spark().read.parquet(str(path)).collect()} == {
        ("a", dt.datetime(2024, 3, 1, 23)),
        ("a", dt.datetime(2024, 3, 2, 0)),
        ("b", dt.datetime(2024, 3, 1, 23)),
        ("b", dt.datetime(2024, 3, 2, 0)),
    }


def test_hourly_literals_and_membership_ignore_the_non_utc_session_zone() -> None:
    cadence = parse_frequency("H")
    frame = spark().createDataFrame(
        [(dt.datetime(2024, 3, 1, 22),), (dt.datetime(2024, 3, 1, 23),)],
        "date timestamp_ntz",
    )
    literal = spark().range(1).select(
        grid.time_literal(dt.datetime(2024, 3, 1, 22), cadence).alias("date"),
    ).collect()[0]["date"]
    assert literal == dt.datetime(2024, 3, 1, 22)
    assert [row["date"] for row in frame.filter(
        grid.period_membership("date", [dt.datetime(2024, 3, 1, 23)], cadence)
    ).collect()] == [dt.datetime(2024, 3, 1, 23)]


def test_period_membership_handles_a_large_hourly_history_as_one_native_predicate() -> None:
    cadence = parse_frequency("H")
    start = dt.datetime(2024, 1, 1)
    history = tuple(start + dt.timedelta(hours=index) for index in range(1024))
    frame = spark().createDataFrame(
        [(point,) for point in (*history, start + dt.timedelta(hours=1024))],
        "date timestamp_ntz",
    )

    predicate = grid.period_membership("date", history, cadence)
    assert str(predicate).startswith("Column<'in(")
    selected = frame.filter(predicate)

    assert selected.count() == 1024
    assert selected.agg(F.min("date"), F.max("date")).first() == (
        history[0],
        history[-1],
    )
    assert frame.filter(grid.period_membership("date", (), cadence)).count() == 0


def test_hourly_effective_eligibility_uses_the_exact_timestamp_key() -> None:
    path = grid.build(
        hourly_source(), "2024-03-01T22:00:00", 2, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="time", eligible_col="eligible", frequency="H",
    )
    assert {(row["series_id"], row["date"])
            for row in spark().read.parquet(str(path)).collect()} == {
        ("a", dt.datetime(2024, 3, 1, 23)),
        ("a", dt.datetime(2024, 3, 2, 0)),
        ("b", dt.datetime(2024, 3, 2, 0)),
    }


def test_build_uses_anchored_calendar_points() -> None:
    monthly = grid.build(
        source(), "2024-01-31", 3, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="day", frequency="M",
    )
    assert {row["date"] for row in spark().read.parquet(str(monthly)).collect()} == {
        dt.date(2024, 2, 29), dt.date(2024, 3, 31), dt.date(2024, 4, 30),
    }
    month_start = grid.build(
        source(), "2024-01-01", 2, spark(), tempfile.mkdtemp(),
        series_col="id", time_col="day", frequency="MS",
    )
    assert {row["date"] for row in spark().read.parquet(str(month_start)).collect()} == {
        dt.date(2024, 2, 1), dt.date(2024, 3, 1),
    }


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


def test_content_hash_preserves_hourly_keys_instead_of_collapsing_to_date() -> None:
    one = pathlib.Path(tempfile.mkdtemp()) / "one"
    two = pathlib.Path(tempfile.mkdtemp()) / "two"
    spark().createDataFrame(
        [("a", dt.datetime(2024, 3, 1, 22))], "series_id string, date timestamp_ntz",
    ).write.parquet(str(one))
    spark().createDataFrame(
        [("a", dt.datetime(2024, 3, 1, 23))], "series_id string, date timestamp_ntz",
    ).write.parquet(str(two))
    assert grid.content_hash(one, spark(), frequency="H") != grid.content_hash(
        two, spark(), frequency="H",
    )


def test_grid_key_validation_refuses_lossy_and_off_anchor_values() -> None:
    lossy = pathlib.Path(tempfile.mkdtemp()) / "lossy"
    spark().createDataFrame(
        [("a", dt.datetime(2024, 1, 31, 1), True)],
        "id string, time timestamp_ntz, eligible boolean",
    ).write.parquet(str(lossy))
    with pytest.raises(grid.GridError, match="lossy"):
        grid.build(
            lossy, "2024-01-30", 1, spark(), tempfile.mkdtemp(),
            series_col="id", time_col="time", eligible_col="eligible", frequency="D",
        )

    off_anchor = pathlib.Path(tempfile.mkdtemp()) / "off-anchor"
    spark().createDataFrame(
        [("a", dt.date(2024, 2, 28), True)],
        "id string, time date, eligible boolean",
    ).write.parquet(str(off_anchor))
    with pytest.raises(grid.GridError, match="off-anchor"):
        grid.build(
            off_anchor, "2024-01-31", 1, spark(), tempfile.mkdtemp(),
            series_col="id", time_col="time", eligible_col="eligible", frequency="M",
        )


def test_hourly_grid_boundaries_require_timestamp_physical_type() -> None:
    source_path = pathlib.Path(tempfile.mkdtemp()) / "date-panel"
    spark().createDataFrame(
        [("a", dt.date(2024, 3, 1), True)], "id string, time date, eligible boolean",
    ).write.parquet(str(source_path))
    with pytest.raises(grid.GridError, match="timestamp_ntz physical type"):
        grid.build(
            source_path, "2024-03-01T22:00:00", 1, spark(), tempfile.mkdtemp(),
            series_col="id", time_col="time", eligible_col="eligible", frequency="H",
        )

    grid_path = pathlib.Path(tempfile.mkdtemp()) / "date-grid"
    spark().createDataFrame(
        [("a", dt.date(2024, 3, 1))], "series_id string, date date",
    ).write.parquet(str(grid_path))
    with pytest.raises(grid.GridError, match="timestamp_ntz physical type"):
        grid.content_hash(grid_path, spark(), frequency="H")


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
