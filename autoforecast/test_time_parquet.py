"""Real Parquet writer/reader regressions for timezone-naive legacy time keys."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from autoforecast import forecast_campaign, grid, plots, ranking
from autoforecast.adapters.naive import seasonal_naive
from autoforecast.data_observability import inspect_panel
from autoforecast.frequency import parse_frequency
from autoforecast.panel import Forecast, ForecastError, Panel, PanelError
from autoforecast.parquet_time import ParquetTimeError, validate_parquet_time_storage
from autoforecast.score_spark import ScoreError, score_forecast


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


@pytest.fixture
def corrupt_ns_int96(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "part.parquet"
    values = pa.array(
        [1_704_067_200_000_000_000 + 1_000 * index for index in range(10_000)],
        type=pa.timestamp("ns"),
    )
    pq.write_table(
        pa.table({"series_id": ["a"] * len(values), "date": values,
                  "yhat": [1.0] * len(values)}),
        path, use_deprecated_int96_timestamps=True, compression="gzip",
        write_page_checksum=True,
    )
    with pq.ParquetFile(path) as part:
        column = part.metadata.row_group(0).column(1)
        start = column.dictionary_page_offset
        if start is None or start < 0:
            start = column.data_page_offset
        position = start + column.total_compressed_size // 2
    damaged = bytearray(path.read_bytes())
    damaged[position] ^= 0xFF
    path.write_bytes(damaged)
    # The footer remains readable; the real compressed time page fails while scanning.
    with pq.ParquetFile(path) as part:
        assert part.schema.column(1).physical_type == "INT96"
        with pytest.raises(OSError):
            list(part.iter_batches(columns=["date"]))
    panel = Panel(source, "series_id", "date", "yhat", frequency="H")
    forecast = Forecast(
        "external", source, "2024-01-01T00:00:00", len(values), 1,
        {"emitted": len(values)}, frequency="H",
    )
    return panel, forecast


def test_corrupt_ns_int96_page_uses_public_storage_error_contracts(
    spark, tmp_path, corrupt_ns_int96,
):
    panel, forecast = corrupt_ns_int96
    expected = r"could not read INT96 time column 'date'"
    with pytest.raises(ParquetTimeError, match=expected):
        validate_parquet_time_storage(panel.path, "date")
    with pytest.raises(PanelError, match=expected):
        panel.validate()
    with pytest.raises(ForecastError, match=expected):
        forecast.validate()
    with pytest.raises(grid.GridError, match=expected):
        grid.read_time_parquet(spark, panel.path, "date")
    with pytest.raises(grid.GridError, match=expected):
        grid.summarise(panel.path, spark, frequency="H")
    with pytest.raises(grid.GridError, match=expected):
        grid.agrees_with(panel.path, panel.path, spark, frequency="H")
    common = tmp_path / "common"
    with pytest.raises(ranking.SelectionError, match=expected):
        ranking.common_cells([forecast], spark, common)
    assert not common.exists()

    contract = tmp_path / "hourly.yaml"
    contract.write_text(
        "schema_version: 1\ndataset:\n  id: corrupt-int96\n  panel:\n"
        "    series_col: series_id\n    time_col: date\n    target_col: yhat\n"
        "    frequency: H\n    season_length: 2\n",
        encoding="utf-8",
    )
    source = Path(panel.path)
    digest = forecast_campaign.directory_digest(source, suffixes=(".parquet",))
    with pytest.raises(PanelError, match=expected):
        forecast_campaign._panel_range_facts(source, contract_path=contract)  # noqa: SLF001
    training = tmp_path / "training"
    with pytest.raises(PanelError, match=expected):
        forecast_campaign._materialize_training_panel(  # noqa: SLF001
            source, training, contract_path=contract, origin=forecast.origin, horizon=1,
            training_start=None, expected_source_sha256=digest,
        )
    assert not training.exists()
    evaluation = tmp_path / "evaluation"
    with pytest.raises(PanelError, match=expected):
        forecast_campaign._materialize_panel_range(  # noqa: SLF001
            source, evaluation, contract_path=contract, start=None, end=forecast.origin,
            expected_source_sha256=digest,
        )
    assert not evaluation.exists()


def test_scorer_wraps_int96_precision_scan_io_failure_as_score_error(
    spark, corrupt_ns_int96,
):
    panel, forecast = corrupt_ns_int96
    with pytest.raises(ScoreError, match=r"could not read INT96 time column 'date'"):
        score_forecast(panel, forecast, spark, missing_actual="no_data")


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


@pytest.mark.parametrize("zone", ["UTC", "Asia/Kolkata"])
@pytest.mark.parametrize("frequency", ["D", "H"])
@pytest.mark.parametrize("legacy_first", [True, False])
@pytest.mark.parametrize("native_unit", ["ms", "us"])
def test_mixed_naive_timestamp_fragments_keep_exact_keys(
    spark, tmp_path, zone, frequency, legacy_first, native_unit,
):
    points = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)] if frequency == "D" else [
        dt.datetime(2024, 1, 1, 23), dt.datetime(2024, 1, 2),
    ]
    for index, point in enumerate(points):
        legacy = (index == 0) == legacy_first
        _write(tmp_path / f"{index}.parquet", [point], int96=legacy,
               unit="us" if legacy else native_unit)
    previous = spark.conf.get("spark.sql.session.timeZone")
    try:
        spark.conf.set("spark.sql.session.timeZone", zone)
        frame = grid.read_time_parquet(spark, tmp_path, "when").orderBy("when")
        assert frame.schema["when"].dataType.typeName() == "timestamp_ntz"
        key, valid = grid.canonical_key("when", parse_frequency(frequency))
        rows = frame.select(key.alias("key"), valid.alias("valid")).collect()
        expected = points if frequency == "H" else [point.date() for point in points]
        assert [row.key for row in rows] == expected
        assert all(row.valid for row in rows)
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous)


@pytest.mark.parametrize("bad_first", [True, False])
@pytest.mark.parametrize("fault", ["aware_native", "aware_legacy", "unproven", "lossy"])
def test_mixed_timestamp_fragments_still_require_lossless_naive_storage(
    spark, tmp_path, bad_first, fault,
):
    _write(tmp_path / ("1.parquet" if bad_first else "0.parquet"),
           [dt.datetime(2024, 1, 1)], int96=fault == "aware_native")
    _write(
        tmp_path / ("0.parquet" if bad_first else "1.parquet"),
        [1_704_067_200_000_000_001] if fault == "lossy" else [dt.datetime(2024, 1, 2)],
        unit="ns" if fault == "lossy" else "us",
        timezone="Asia/Kathmandu" if fault.startswith("aware") else None,
        store_schema=fault != "unproven", int96=fault != "aware_native",
    )
    with pytest.raises(grid.GridError, match="timezone-bearing|ARROW:schema|sub-microsecond"):
        grid.read_time_parquet(spark, tmp_path, "when")


@pytest.mark.parametrize("legacy_first", [True, False])
def test_mixed_int96_and_date_fields_require_a_uniform_schema(spark, tmp_path, legacy_first):
    _write(tmp_path / ("0.parquet" if legacy_first else "1.parquet"),
           [dt.datetime(2024, 1, 1)])
    pq.write_table(pa.table({"when": [dt.date(2024, 1, 2)]}),
                   tmp_path / ("1.parquet" if legacy_first else "0.parquet"))
    with pytest.raises(grid.GridError, match="uniform time schema"):
        grid.read_time_parquet(spark, tmp_path, "when")


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
@pytest.mark.parametrize("mixed", [False, True])
def test_int96_panel_and_forecast_work_through_inspection_baseline_and_scoring(
    spark, tmp_path, frequency, mixed,
):
    # Full synthetic population: one series, six observed periods, no exclusions. The last
    # two periods are held out. Their actuals differ from the repeated last season by 2 each.
    start = dt.datetime(2024, 1, 1)
    step = dt.timedelta(hours=1) if frequency == "H" else dt.timedelta(days=1)
    points = [start + step * index for index in range(6)]
    source = tmp_path / "panel"
    source.mkdir()
    table = pa.table({
        "product": ["a"] * 6, "when": pa.array(points, type=pa.timestamp("us")),
        "value": [1., 2., 3., 4., 5., 6.],
    })
    pq.write_table(table.slice(0, 3) if mixed else table,
                   source / "a.parquet", use_deprecated_int96_timestamps=True)
    if mixed:
        pq.write_table(table.slice(3), source / "b.parquet")
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
    table = pa.table({
        "series_id": ["a", "a"], "date": pa.array(points[4:], type=pa.timestamp("us")),
        "yhat": [3., 4.],
    })
    pq.write_table(table.slice(0, 1) if mixed else table,
                   legacy / "a.parquet", use_deprecated_int96_timestamps=True)
    if mixed:
        pq.write_table(table.slice(1), legacy / "b.parquet")
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
