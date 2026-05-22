"""Real Parquet writer/reader regressions for timezone-naive legacy time keys."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from autoforecast import grid, plots
from autoforecast.adapters.naive import seasonal_naive
from autoforecast.data_observability import inspect_panel
from autoforecast.frequency import parse_frequency
from autoforecast.panel import Forecast, ForecastError, Panel, PanelError
from autoforecast.score_spark import score_forecast


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]").appName("test_time_parquet")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false").getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    return session


def _write(path, values, *, unit="us", timezone=None, store_schema=True, int96=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"when": pa.array(values, type=pa.timestamp(unit, tz=timezone))}),
        path, use_deprecated_int96_timestamps=int96, store_schema=store_schema,
    )


@pytest.mark.parametrize("zone", ["UTC", "Asia/Kolkata"])
@pytest.mark.parametrize("frequency", ["D", "H"])
def test_naive_int96_preserves_exact_wall_clock_keys(spark, tmp_path, zone, frequency):
    points = [dt.datetime(2024, 3, 1, 23), dt.datetime(2024, 3, 2)] if frequency == "H" else [
        dt.datetime(2024, 3, 1), dt.datetime(2024, 3, 2),
    ]
    path = tmp_path / "data.parquet"
    _write(path, points)
    previous = spark.conf.get("spark.sql.session.timeZone")
    try:
        spark.conf.set("spark.sql.session.timeZone", zone)
        frame = grid.read_time_parquet(spark, path, "when")
        key, valid = grid.canonical_key("when", parse_frequency(frequency))
        rows = frame.select(key.alias("key"), valid.alias("valid")).collect()
        expected = points if frequency == "H" else [value.date() for value in points]
        assert [row.key for row in rows] == expected
        assert all(row.valid for row in rows)
        assert frame.schema["when"].dataType.typeName() == "timestamp_ntz"
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous)


@pytest.mark.parametrize("timezone", ["UTC", "America/New_York"])
@pytest.mark.parametrize("int96", [True, False])
def test_timezone_bearing_parquet_is_never_reinterpreted_as_naive(
    spark, tmp_path, timezone, int96,
):
    path = tmp_path / "data.parquet"
    _write(path, [dt.datetime(2024, 1, 1)], timezone=timezone, int96=int96)
    # INT96's reconstructed Arrow schema loses timezone information. The embedded writer
    # schema is the evidence that must prevent an apparently-naive reinterpretation.
    if int96:
        assert pq.read_schema(path).field("when").type.tz is None
    with pytest.raises(grid.GridError, match="timezone-bearing"):
        grid.read_time_parquet(spark, path, "when")


@pytest.mark.parametrize("second_fragment", ["unproven", "aware"])
def test_every_visible_int96_fragment_requires_naive_writer_evidence(
    spark, tmp_path, second_fragment,
):
    _write(tmp_path / "a.parquet", [dt.datetime(2024, 1, 1)], int96=False)
    _write(
        tmp_path / "b.parquet", [dt.datetime(2024, 1, 2)],
        store_schema=second_fragment != "unproven",
        timezone="UTC" if second_fragment == "aware" else None,
    )
    expected = "ARROW:schema" if second_fragment == "unproven" else "timezone-bearing"
    with pytest.raises(grid.GridError, match=expected):
        grid.read_time_parquet(spark, tmp_path, "when")


def test_arrow_contracts_reject_timezone_bearing_int96(tmp_path):
    panel_path = tmp_path / "panel"
    panel_path.mkdir()
    pq.write_table(
        pa.table({
            "series_id": ["a"],
            "when": pa.array(
                [dt.datetime(2024, 1, 1)],
                type=pa.timestamp("us", tz="Asia/Kathmandu"),
            ),
            "value": [1.0],
        }),
        panel_path / "part.parquet",
        use_deprecated_int96_timestamps=True,
    )
    assert pq.read_schema(panel_path / "part.parquet").field("when").type.tz is None
    with pytest.raises(PanelError, match="timezone-bearing"):
        Panel(panel_path, "series_id", "when", "value", frequency="H").validate()

    forecast_path = tmp_path / "forecast"
    forecast_path.mkdir()
    pq.write_table(
        pa.table({
            "series_id": ["a"],
            "date": pa.array(
                [dt.datetime(2024, 1, 1, 1)],
                type=pa.timestamp("us", tz="Asia/Kathmandu"),
            ),
            "yhat": [1.0],
        }),
        forecast_path / "part.parquet",
        use_deprecated_int96_timestamps=True,
    )
    assert pq.read_schema(forecast_path / "part.parquet").field("date").type.tz is None
    with pytest.raises(ForecastError, match="timezone-bearing"):
        Forecast(
            "external", forecast_path, "2024-01-01T00:00:00", 1, 1,
            {"emitted": 1}, frequency="H",
        ).validate()


def test_raw_grid_consumers_reject_timezone_bearing_int96(spark, tmp_path):
    invalid = tmp_path / "aware-grid"
    invalid.mkdir()
    pq.write_table(
        pa.table({
            "series_id": ["a"],
            "date": pa.array(
                [dt.datetime(2024, 1, 1, 1)],
                type=pa.timestamp("us", tz="Asia/Kathmandu"),
            ),
        }),
        invalid / "part.parquet",
        use_deprecated_int96_timestamps=True,
    )

    with pytest.raises(grid.GridError, match="timezone-bearing"):
        grid.summarise(invalid, spark, frequency="H")
    with pytest.raises(grid.GridError, match="timezone-bearing"):
        grid.agrees_with(invalid, invalid, spark, frequency="H")


def test_hive_partitioned_date_forecast_remains_valid_through_consumers(
    spark, tmp_path,
):
    forecast_path = tmp_path / "forecast"
    pq.write_to_dataset(
        pa.table({
            "series_id": ["a"],
            "date": pa.array([dt.date(2024, 1, 2)], type=pa.date32()),
            "yhat": [2.0],
        }),
        root_path=str(forecast_path),
        partition_cols=["date"],
    )
    forecast = Forecast(
        "external", forecast_path, "2024-01-01", 1, 1, {"emitted": 1}
    ).validate()

    panel_path = tmp_path / "panel.parquet"
    pq.write_table(
        pa.table({
            "series_id": ["a"],
            "date": pa.array([dt.date(2024, 1, 2)], type=pa.date32()),
            "value": [3.0],
        }),
        panel_path,
    )
    panel = Panel(panel_path, "series_id", "date", "value").validate()

    frame = grid.read_time_parquet(spark, forecast_path, "date")
    assert frame.schema["date"].dataType.typeName() == "date"
    assert frame.first()["date"] == dt.date(2024, 1, 2)
    assert score_forecast(panel, forecast, spark, missing_actual="no_data").n_scored == 1
    assert plots._collect_forecast(forecast, spark, ["a"]) == {  # noqa: SLF001
        "a": [(dt.date(2024, 1, 2), 2.0)]
    }


def test_int96_submicrosecond_keys_are_refused_before_spark_truncates_them(spark, tmp_path):
    path = tmp_path / "data.parquet"
    _write(path, [1_704_067_200_000_000_001], unit="ns")
    with pytest.raises(grid.GridError, match="sub-microsecond"):
        grid.read_time_parquet(spark, path, "when")


def test_int96_microsecond_keys_keep_dates_outside_nanosecond_range(spark, tmp_path):
    path = tmp_path / "data.parquet"
    points = [dt.datetime(2500, 1, 1)]
    _write(path, points)
    assert [row.when for row in grid.read_time_parquet(spark, path, "when").collect()] == points


@pytest.mark.parametrize("frequency,point", [
    ("D", dt.datetime(2024, 1, 1, 1)),
    ("H", dt.datetime(2024, 1, 1, 1, 1)),
])
def test_int96_read_does_not_relax_cadence_alignment(spark, tmp_path, frequency, point):
    path = tmp_path / "data.parquet"
    _write(path, [point])
    frame = grid.read_time_parquet(spark, path, "when")
    _, valid = grid.canonical_key("when", parse_frequency(frequency))
    assert not frame.select(valid.alias("valid")).first().valid


@pytest.mark.parametrize("frequency", ["D", "H"])
def test_int96_panel_and_forecast_work_through_inspection_baseline_and_scoring(
    spark, tmp_path, frequency,
):
    # Full synthetic population: one series, six observed periods, no exclusions. The last
    # two periods are held out. Their actuals differ from the repeated last season by 2 each.
    start = dt.datetime(2024, 1, 1)
    step = dt.timedelta(hours=1) if frequency == "H" else dt.timedelta(days=1)
    points = [start + step * index for index in range(6)]
    source = tmp_path / "panel"
    source.mkdir()
    pq.write_table(pa.table({
        "product": ["a"] * 6, "when": pa.array(points, type=pa.timestamp("us")),
        "value": [1., 2., 3., 4., 5., 6.],
    }), source / "data.parquet", use_deprecated_int96_timestamps=True)
    panel = Panel(str(source), "product", "when", "value", frequency=frequency,
                  season_length=2).validate()
    origin = points[3].isoformat() if frequency == "H" else points[3].date().isoformat()
    report = inspect_panel(panel, spark, as_of=origin)
    assert report.accepted and report.facts.n_series == 1 and report.facts.n_rows == 4
    forecast = seasonal_naive(panel, origin, 2, spark=spark, out_dir=tmp_path / "forecast")
    result = score_forecast(panel, forecast, spark, missing_actual="no_data")
    assert result.n_scored == 2 and result.sum_abs_error == 4.0
    eligible = grid.build(source, origin, 2, spark, tmp_path / "grid",
                          series_col="product", time_col="when", frequency=frequency)
    assert grid.summarise(eligible, spark).first()["n_cells"] == 2
    # An externally supplied forecast/grid can use the same legacy representation.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    pq.write_table(pa.table({
        "series_id": ["a", "a"], "date": pa.array(points[4:], type=pa.timestamp("us")),
        "yhat": [3., 4.],
    }), legacy / "data.parquet", use_deprecated_int96_timestamps=True)
    external = Forecast("external", str(legacy), origin, 2, 1, {"emitted": 2},
                        frequency=frequency).validate()
    result = score_forecast(panel, external, spark, missing_actual="no_data",
                            restrict_to=str(legacy))
    assert result.n_scored == 2 and result.sum_abs_error == 4.0
    summary = grid.summarise(legacy, spark, frequency=frequency).first()
    assert (summary["n_cells"], summary["n_series"], summary["n_days"]) == (2, 1, 2)
    assert grid.agrees_with(
        legacy, eligible, spark, frequency=frequency
    ) == (0, 0)
    assert grid.content_hash(legacy, spark, frequency=frequency) == grid.content_hash(
        Path(eligible), spark, frequency=frequency,
    )
