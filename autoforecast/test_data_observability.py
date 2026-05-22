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
