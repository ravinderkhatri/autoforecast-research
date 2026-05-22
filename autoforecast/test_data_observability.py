"""Functional Spark checks for authoritative data profiling and temporal integrity reporting."""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoforecast.data_admission import validate_model_compatibility
from autoforecast.data_observability import (
    CheckReport,
    inspect_panel,
    run_incident_from_error,
    write_data_observability_report,
)
from autoforecast.frequency import canonical_origin, parse_origin, period_points
from autoforecast.panel import Panel


def _spark():
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Spark functional test needs JAVA_HOME")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("test_data_observability")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _panel(path: pathlib.Path) -> Panel:
    return Panel(
        path=str(path),
        series_col="id",
        time_col="date",
        target_col="qty",
        static=("category",),
        known_future=("promo",),
        season_length=7,
    ).validate()


def test_run_incident_from_error_uses_the_first_nonblank_error_line() -> None:
    incident = run_incident_from_error(
        RuntimeError("\n  parquet footer is unreadable  \nlater diagnostic detail")
    )

    assert incident.id == "data_read_failed"
    assert incident.summary == "Could not read the requested data. Data checks were not run."
    assert incident.technical_detail == "RuntimeError: parquet footer is unreadable"
    assert "later diagnostic detail" not in incident.technical_detail


def test_data_checks_write_structured_results_and_report_daily_gaps_as_warnings() -> None:
    spark = _spark()
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        source = root / "panel"
        start = dt.date(2024, 1, 1)
        rows = [
            ("a", start, 1.0, "x", 0.0),
            ("a", start + dt.timedelta(days=2), 2.0, "x", 1.0),
            ("b", start, 0.0, "y", 0.0),
            ("b", start + dt.timedelta(days=1), 1.0, "y", 0.0),
        ]
        spark.createDataFrame(
            rows, "id string, date date, qty double, category string, promo double"
        ).write.parquet(str(source))
        report = inspect_panel(_panel(source), spark, as_of="2024-01-03")
        assert isinstance(report, CheckReport)
        assert report.accepted
        assert report.facts.missing_daily_timestamps == 1
        time_grain = next(check for check in report.checks if check.id == "time_grain")
        assert time_grain.status == "warning"
        assert time_grain.numerator == 1
        output = write_data_observability_report(report, root / "out")
        assert (output / "data_profile.json").exists()
        assert (output / "data_validation_report.json").exists()
        page = (output / "report.html").read_text(encoding="utf-8")
        assert "Data check passed" in page
        assert "Every check is shown with its result and measurement." in page
        assert "What this data suggests" in page
        assert "0 missing required values of 4 rows" in page


def test_duplicate_index_and_changing_static_values_block_data_admission() -> None:
    spark = _spark()
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        source = root / "panel"
        day = dt.date(2024, 1, 1)
        rows = [
            ("a", day, 1.0, "x", 0.0),
            ("a", day, 2.0, "y", 0.0),
        ]
        spark.createDataFrame(
            rows, "id string, date date, qty double, category string, promo double"
        ).write.parquet(str(source))
        report = inspect_panel(_panel(source), spark, as_of="2024-01-01")
        assert isinstance(report, CheckReport)
        assert not report.accepted
        assert report.facts.duplicate_series_timestamps == 1
        assert report.facts.inconsistent_static_series == {"category": 1}
        duplicate = next(check for check in report.checks if check.id == "one_value_per_date")
        static = next(check for check in report.checks if check.id == "static_information")
        assert duplicate.status == "failed" and duplicate.numerator == 1
        assert static.status == "failed" and static.numerator == 1
        assert report.facts.profile is not None
        admission = validate_model_compatibility(
            "mft",
            _panel(source),
            report.facts.profile,
            horizon=1,
            data_checks=report,
        )
        assert admission.status == "ineligible"
        assert admission.reasons == ("Data checks failed. Model checks were not run.",)


@pytest.mark.parametrize(
    "frequency,anchor",
    [
        ("H", "2024-03-01T20:00:00"),
        ("D", "2024-03-01"),
        ("W", "2024-02-23"),
        ("M", "2023-12-31"),
        ("MS", "2024-01-01"),
        ("Y", "2021-12-31"),
        ("YS", "2022-01-01"),
    ],
)
def test_every_cadence_reports_a_missing_internal_period(
    tmp_path, frequency: str, anchor: str
) -> None:
    spark = _spark()
    points = period_points(anchor, frequency, 3)
    source = tmp_path / f"gap-{frequency}"
    key_type = "timestamp_ntz" if frequency == "H" else "date"
    spark.createDataFrame(
        [("a", points[0], 1.0), ("a", points[2], 3.0)],
        f"id string, date {key_type}, qty double",
    ).write.parquet(str(source))
    panel = Panel(source, "id", "date", "qty", frequency=frequency)

    report = inspect_panel(
        panel, spark, as_of=canonical_origin(points[-1], frequency)
    )

    assert isinstance(report, CheckReport)
    assert report.accepted
    assert report.facts.missing_timestamps == (None if frequency == "D" else 1)
    assert report.facts.missing_daily_timestamps == (1 if frequency == "D" else None)
    check = next(item for item in report.checks if item.id == "time_grain")
    assert check.status == "warning" and check.numerator == 1


def test_hourly_inspection_excludes_later_hours_on_the_same_day(tmp_path) -> None:
    spark = _spark()
    source = tmp_path / "hourly"
    rows = [
        ("a", parse_origin("2024-03-01T08:00:00", "H"), 2.0),
        ("a", parse_origin("2024-03-01T09:00:00", "H"), 4.0),
        ("a", parse_origin("2024-03-01T10:00:00", "H"), 0.0),
    ]
    spark.createDataFrame(rows, "id string, date timestamp_ntz, qty double").write.parquet(
        str(source)
    )

    report = inspect_panel(
        Panel(source, "id", "date", "qty", frequency="H"),
        spark,
        as_of="2024-03-01T09:00:00",
    )

    assert isinstance(report, CheckReport) and report.accepted
    assert report.facts.n_rows == 2
    assert report.facts.data_range[-1] == "2024-03-01T09:00:00"
    assert report.facts.profile is not None
    assert report.facts.profile.n_cells == 2


@pytest.mark.parametrize(
    "frequency,as_of,bad_key",
    [
        ("H", "2024-03-01T10:00:00", dt.datetime(2024, 3, 1, 9, 30)),
        ("D", "2024-03-01", dt.datetime(2024, 3, 1, 12)),
        ("M", "2024-03-31", dt.datetime(2024, 3, 30)),
    ],
)
def test_inspection_rejects_lossy_and_off_anchor_time_keys(
    tmp_path, frequency: str, as_of: str, bad_key: dt.datetime
) -> None:
    spark = _spark()
    source = tmp_path / f"off-grid-{frequency}"
    spark.createDataFrame(
        [("a", bad_key, 1.0)], "id string, date timestamp, qty double"
    ).write.parquet(str(source))

    report = inspect_panel(
        Panel(source, "id", "date", "qty", frequency=frequency), spark, as_of=as_of
    )

    assert isinstance(report, CheckReport) and not report.accepted
    check = next(item for item in report.checks if item.id == "time_grain")
    assert check.status == "failed" and check.numerator == 1


def test_inspection_does_not_drop_a_malformed_time_key_during_cutoff_filtering(
    tmp_path,
) -> None:
    spark = _spark()
    source = tmp_path / "malformed-hourly-key"
    spark.createDataFrame(
        [("a", dt.datetime(2024, 3, 1, 9), 1.0), ("a", None, 2.0)],
        "id string, date timestamp_ntz, qty double",
    ).write.parquet(str(source))

    report = inspect_panel(
        Panel(source, "id", "date", "qty", frequency="H"),
        spark,
        as_of="2024-03-01T10:00:00",
    )

    assert isinstance(report, CheckReport) and not report.accepted
    check = next(item for item in report.checks if item.id == "time_grain")
    assert check.status == "failed" and check.numerator == 1


def test_hourly_inspection_rejects_timezone_sensitive_spark_timestamp_storage(
    tmp_path,
) -> None:
    spark = _spark()
    source = tmp_path / "hourly-ltz"
    spark.createDataFrame(
        [("a", dt.datetime(2024, 3, 1, 9), 1.0)],
        "id string, date timestamp, qty double",
    ).write.parquet(str(source))

    report = inspect_panel(
        Panel(source, "id", "date", "qty", frequency="H"),
        spark,
        as_of="2024-03-01T10:00:00",
    )

    assert isinstance(report, CheckReport) and not report.accepted
    check = next(item for item in report.checks if item.id == "time_grain")
    assert check.status == "failed" and check.numerator == 1


def test_preflight_cli_writes_an_accepted_snapshot_report() -> None:
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Spark functional test needs JAVA_HOME")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        source = root / "panel.parquet"
        pq.write_table(
            pa.table(
                {
                    "item": ["a", "a"],
                    "day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
                    "units": [1.0, 2.0],
                }
            ),
            source,
        )
        contract = root / "contract.yaml"
        contract.write_text(
            "schema_version: 1\ndataset:\n  id: cli-daily\n  panel:\n"
            "    series_col: item\n    time_col: day\n    target_col: units\n    frequency: D\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_data_preflight.py",
                str(contract),
                str(source),
                "--as-of",
                "2024-01-02",
                "--output",
                str(root / "report"),
            ],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "Data check report:" in result.stdout
        assert (root / "report" / "report.html").exists()


def test_preflight_cli_keeps_a_read_failure_out_of_the_data_check_report() -> None:
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Spark functional test needs JAVA_HOME")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        source = root / "broken.parquet"
        source.write_text("not parquet", encoding="utf-8")
        contract = root / "contract.yaml"
        contract.write_text(
            "schema_version: 1\ndataset:\n  id: broken-daily\n  panel:\n"
            "    series_col: item\n    time_col: day\n    target_col: units\n    frequency: D\n",
            encoding="utf-8",
        )
        output = root / "report"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_data_preflight.py",
                str(contract),
                str(source),
                "--as-of",
                "2024-01-02",
                "--output",
                str(output),
            ],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "Run incident report:" in result.stdout
        record = __import__("json").loads((output / "data_validation_report.json").read_text())
        assert record["kind"] == "run_incident"
        assert record["status"] == "not_checked"
        page = (output / "report.html").read_text(encoding="utf-8")
        assert "Could not read the requested data" in page
        assert "Spark inspection failed" not in page
