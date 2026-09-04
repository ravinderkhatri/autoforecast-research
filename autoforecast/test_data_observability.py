"""Functional Spark checks for authoritative data profiling and temporal integrity reporting."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F

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


@pytest.mark.parametrize(
    "series_col,time_col,target_col",
    [
        ("sid", "day", "_time_key"),
        ("_time_key", "day", "value"),
        ("sid", "day", "_valid_time_key"),
        ("COUNT", "day", "value"),
        ("LEVELS", "day", "value"),
        ("FIRST", "day", "value"),
        ("M", "__AF_PROFILE_TIME", "__AF_PROFILE_VALID_TIME"),
    ],
)
def test_inspection_temporary_columns_cannot_collide_with_mapped_columns(
    tmp_path, series_col: str, time_col: str, target_col: str
) -> None:
    spark = _spark()
    source = tmp_path / f"collision-{series_col}-{time_col}-{target_col}"
    rows = []
    for offset, value in enumerate((1.0, 2.0)):
        rows.append({
            series_col: "a",
            time_col: dt.date(2024, 1, 1) + dt.timedelta(days=offset),
            target_col: value,
            "feature": "x",
        })
    spark.createDataFrame(rows).write.parquet(str(source))
    panel = Panel(
        source,
        series_col,
        time_col,
        target_col,
        static=("feature",),
    )

    report = inspect_panel(panel, spark, as_of="2024-01-02")

    assert isinstance(report, CheckReport)
    assert report.accepted
    assert report.facts.n_rows == 2
    assert report.facts.n_series == 1
    assert report.facts.n_dates == 2
    assert report.facts.profile is not None
    assert report.facts.profile.n_cells == 2
    assert report.facts.profile.n_series == 1


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


@pytest.mark.parametrize(
    "frequency,origin,aligned,shifted",
    [
        (
            "W",
            "2024-01-10",
            (dt.date(2023, 12, 27), dt.date(2024, 1, 3), dt.date(2024, 1, 10)),
            (dt.date(2023, 12, 21), dt.date(2023, 12, 28), dt.date(2024, 1, 4)),
        ),
        (
            "3D",
            "2024-01-10",
            (dt.date(2024, 1, 4), dt.date(2024, 1, 7), dt.date(2024, 1, 10)),
            (dt.date(2024, 1, 3), dt.date(2024, 1, 6), dt.date(2024, 1, 9)),
        ),
    ],
)
def test_generic_preflight_rejects_consistent_wrong_origin_phase_for_weekly_and_strided_daily(
    tmp_path, frequency: str, origin: str,
    aligned: tuple[dt.date, ...], shifted: tuple[dt.date, ...],
) -> None:
    spark = _spark()
    populations = {
        "aligned": [("aligned", point, 1.0) for point in aligned],
        "mixed": (
            [("aligned", point, 1.0) for point in aligned]
            + [("shifted", point, 1.0) for point in shifted]
        ),
        "all-shifted": [("shifted", point, 1.0) for point in shifted],
    }

    for population, rows in populations.items():
        source = tmp_path / f"{frequency}-{population}"
        spark.createDataFrame(rows, "id string, date date, qty double").write.parquet(
            str(source)
        )
        panel = Panel(
            source, "id", "date", "qty", frequency=frequency, season_length=3,
        )
        report = inspect_panel(panel, spark, as_of=origin)
        assert isinstance(report, CheckReport)
        time_grain = next(item for item in report.checks if item.id == "time_grain")

        if population == "aligned":
            assert report.accepted
            assert time_grain.status == "passed" and time_grain.numerator == 0
            continue

        assert not report.accepted
        assert time_grain.status == "failed"
        assert time_grain.numerator == len(shifted)
        admission = validate_model_compatibility(
            "seasonal_naive",
            panel,
            report.facts.profile,
            horizon=1,
            data_checks=report,
        )
        assert admission.status == "ineligible"
        assert admission.reasons == ("Data checks failed. Model checks were not run.",)


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
        [("a", bad_key, 1.0)], "id string, date timestamp_ntz, qty double"
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
    check = next(item for item in report.checks if item.id == "data_definition")
    assert check.status == "failed" and check.numerator is None
    assert "ARROW:schema" in check.technical_detail
    assert "timezone-naive" in check.technical_detail
    assert report.facts.n_rows == report.facts.n_series == report.facts.n_dates == 0
    assert report.facts.data_range == ("", "")
    assert report.facts.profile is None and report.facts.spark_schema == {}


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


def test_preflight_cli_rejects_a_complete_weekly_lattice_on_the_wrong_as_of_phase() -> None:
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Spark functional test needs JAVA_HOME")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        source = root / "panel.parquet"
        pq.write_table(
            pa.table(
                {
                    "item": ["shifted"] * 3,
                    "day": [
                        dt.date(2023, 12, 21),
                        dt.date(2023, 12, 28),
                        dt.date(2024, 1, 4),
                    ],
                    "units": [1.0, 2.0, 3.0],
                }
            ),
            source,
        )
        contract = root / "contract.yaml"
        contract.write_text(
            "schema_version: 1\ndataset:\n  id: cli-weekly\n  panel:\n"
            "    series_col: item\n    time_col: day\n    target_col: units\n"
            "    frequency: W\n    season_length: 3\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_data_preflight.py",
                str(contract),
                str(source),
                "--as-of",
                "2024-01-10",
                "--output",
                str(root / "report"),
            ],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, result.stdout + result.stderr
        record = json.loads((root / "report" / "data_validation_report.json").read_text())
        assert not record["accepted"]
        time_grain = next(check for check in record["checks"] if check["id"] == "time_grain")
        assert time_grain["status"] == "failed" and time_grain["numerator"] == 3


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


def _source_study_context():
    from autoforecast.data_admission import DatasetContract

    contract = DatasetContract("synthetic-source", {
        "series_col": "item", "time_col": "day", "target_col": "units",
        "frequency": "D", "target_estimand": "observed_nonnegative",
    }, pathlib.Path("synthetic.yaml"))
    recipe = {
        "schema_version": 2, "source_keys": ["item", "store"],
        "training_start": "2024-01-01", "validation_origins": ["2024-01-05"],
        "validation_horizon": 2, "final_train_cutoff": "2024-01-07", "final_test_horizon": 2,
        "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
        "feature_availability": {}, "metric": "rmse", "actual_missing": "mask",
        "series_starts": {"rule": "declared", "values": [
            {"keys": {"item": "A", "store": 1}, "start": "2024-01-01"},
            {"keys": {"item": "B", "store": 2}, "start": "2024-01-04"},
        ]},
    }
    return contract, recipe


def test_source_findings_are_factual_and_keep_all_windows_and_original_keys():
    from autoforecast.data_observability import report_source_issues, summarize_source_issues

    spark = _spark()
    contract, recipe = _source_study_context()
    recipe["tuning_windows"] = [
        {"origin": "2024-01-01", "horizon": 2},
        {"origin": "2024-01-02", "horizon": 1},
    ]
    rows = [
        ("A", 1, dt.date(2024, 1, 1), -2.0),
        ("A", 1, dt.date(2024, 1, 1), -2.0),
        ("A", 1, dt.date(2024, 1, 2), None),
        ("A", 1, dt.date(2024, 1, 4), float("inf")),
        ("A", 1, dt.date(2024, 1, 5), float("nan")),
        ("A", 1, dt.date(2024, 1, 6), 0.0),
        ("A", 1, dt.date(2024, 1, 10), 3.0),
        (None, 1, dt.date(2024, 1, 1), 2.0),
        (" \t", 1, dt.date(2024, 1, 1), 2.0),
        ("A", 1, None, 2.0),
    ]
    frame = spark.createDataFrame(rows, "item string,store int,day date,units double")
    before = sorted(frame.toJSON().collect())
    issues = report_source_issues(frame, contract=contract, recipe=recipe).cache()
    try:
        records = [row.asDict(recursive=True) for row in issues.collect()]
        assert {row["issue_code"] for row in records} == {
            "NULL_ID", "BLANK_ID", "NULL_DATE", "OUTSIDE_DATE_RANGE", "NONFINITE_VALUE",
            "DUPLICATE_DATE", "MISSING_HISTORY", "MISSING_ACTUAL",
        }
        nonfinite = [row for row in records if row["issue_code"] == "NONFINITE_VALUE"]
        assert {row["observed_value"] for row in nonfinite} == {"Infinity", "NaN"}
        duplicate, = (row for row in records if row["issue_code"] == "DUPLICATE_DATE")
        assert (duplicate["observed_value"], duplicate["affected_rows"]) == ("2", 1)
        assert duplicate["keys"] == {"item": "A", "store": 1}
        missing = [row for row in records if row["issue_code"].startswith("MISSING_")]
        assert all(row["status"] == "warning" for row in missing)
        assert not any(row["keys"]["item"] == "B" and row["time"] < "2024-01-04"
                       and row["issue_code"] == "MISSING_HISTORY" for row in missing)
        assert {row["origin"] for row in missing if row["keys"]["item"] == "A"
                and row["time"] == "2024-01-03" and row["issue_code"] == "MISSING_ACTUAL"} == {
                    "2024-01-01", "2024-01-02",
                }
        assert {row["time"] for row in missing if row["keys"]["item"] == "B"
                and row["stage"] == "test" and row["issue_code"] == "MISSING_ACTUAL"} == {
                    "2024-01-08", "2024-01-09",
                }
        assert not any(row["time"] == "2024-01-06" and row["keys"]["item"] == "A"
                       and row["issue_code"] == "MISSING_ACTUAL" for row in records)
        summaries = summarize_source_issues(issues).collect()
        assert sum(row.findings for row in summaries) == len(records)
        assert sum(row.affected_rows for row in summaries) == sum(
            row["affected_rows"] for row in records)
        assert sorted(frame.toJSON().collect()) == before
    finally:
        issues.unpersist()


def test_source_report_empty_schema_and_unequal_inferred_starts():
    from autoforecast.data_observability import report_source_issues

    spark = _spark()
    contract, recipe = _source_study_context()
    rows = [(item, store, dt.date(2024, 1, day), 0.0 if day == 6 else -1.0)
            for item, store, first in (("A", 1, 1), ("B", 2, 4)) for day in range(first, 10)]
    frame = spark.createDataFrame(rows, "item string,store int,day date,units double")
    clean = report_source_issues(frame, contract=contract, recipe=recipe)
    assert clean.count() == 0
    recipe.pop("series_starts")
    inferred = report_source_issues(frame, contract=contract, recipe=recipe)
    assert inferred.count() == 0
    assert inferred.schema == clean.schema
    broken = report_source_issues(frame.filter("day < date'2024-01-09'"),
                                  contract=contract, recipe=recipe)
    assert broken.schema == clean.schema
    missing = broken.collect()
    assert len(missing) == 2
    assert {row.keys["item"] for row in missing} == {"A", "B"}
    assert all(row.issue_code == "MISSING_ACTUAL" and row.time == "2024-01-09" for row in missing)


@pytest.mark.parametrize("case", ["missing", "nonnumeric", "cadence", "timezone", "lossy", "integer"])
def test_source_report_rejects_invalid_schema_or_time_without_changing_input(case):
    import dataclasses

    from autoforecast.data_observability import DataCheckError, report_source_issues

    spark = _spark()
    contract, recipe = _source_study_context()
    frame = spark.createDataFrame([("A", 1, dt.date(2024, 1, 1), 1.0)],
                                  "item string,store int,day date,units double")
    if case == "missing":
        frame = frame.drop("units")
    elif case == "nonnumeric":
        frame = frame.withColumn("units", F.col("units").cast("string"))
    elif case == "cadence":
        contract = dataclasses.replace(contract, panel_roles={**contract.panel_roles, "frequency": "H"})
    elif case == "timezone":
        frame = frame.withColumn("day", F.col("day").cast("timestamp"))
    elif case == "lossy":
        frame = frame.withColumn("day", F.col("day").cast("timestamp_ntz") + F.expr("INTERVAL 1 HOUR"))
    else:
        frame = frame.withColumn("day", F.lit(1))
    before = (frame.schema.json(), frame.toJSON().collect())
    with pytest.raises(DataCheckError):
        report_source_issues(frame, contract=contract, recipe=recipe)
    assert (frame.schema.json(), frame.toJSON().collect()) == before


@pytest.mark.parametrize("frequency,storage", [("D", "timestamp_ntz"), ("H", "timestamp_ntz"),
                                                ("MS", "date")])
def test_source_report_preserves_cadence_and_declared_starts(frequency, storage):
    import dataclasses

    from autoforecast.data_observability import DataCheckError, report_source_issues
    from autoforecast.frequency import advance, parse_origin

    spark = _spark()
    contract, recipe = _source_study_context()
    contract = dataclasses.replace(contract, panel_roles={**contract.panel_roles,
                                                         "frequency": frequency})
    first = parse_origin("2024-01-01T00:00:00" if frequency == "H" else "2024-01-01", frequency)
    points = [advance(first, frequency, index) for index in range(9)]
    recipe.update(frequency=frequency, training_start=points[0].isoformat(),
                  validation_origins=[points[4].isoformat()], final_train_cutoff=points[6].isoformat())
    recipe["series_starts"]["values"][0]["start"] = points[0].isoformat()
    recipe["series_starts"]["values"][1]["start"] = points[3].isoformat()
    rows = [(item, store, point, 1.0) for item, store, start in (("A", 1, 0), ("B", 2, 3))
            for point in points[start:]]
    source_type = "timestamp_ntz" if frequency == "H" else "date"
    frame = spark.createDataFrame(rows, f"item string,store int,day {source_type},units double")
    frame = frame.withColumn("day", F.col("day").cast(storage))
    assert report_source_issues(frame, contract=contract, recipe=recipe).count() == 0
    shortened = frame.filter(~((F.col("item") == "B") & (F.col("day") == F.lit(points[-1]))))
    missing = report_source_issues(shortened, contract=contract, recipe=recipe).collect()
    assert len(missing) == 1 and missing[0].issue_code == "MISSING_ACTUAL"
    assert missing[0].keys["item"] == "B"
    if frequency == "MS":
        with pytest.raises(DataCheckError, match="off-grid"):
            report_source_issues(frame.withColumn("day", F.date_add("day", 1)),
                                 contract=contract, recipe=recipe)


def test_source_report_accepts_boolean_past_covariates():
    import dataclasses

    from autoforecast.data_observability import report_source_issues

    spark = _spark()
    contract, recipe = _source_study_context()
    contract = dataclasses.replace(contract, panel_roles={**contract.panel_roles,
                                                         "past_only": ("flag",)})
    frame = spark.createDataFrame([
        (item, store, dt.date(2024, 1, day), 1.0, day % 2 == 0)
        for item, store, first in (("A", 1, 1), ("B", 2, 4)) for day in range(first, 10)
    ], "item string,store int,day date,units double,flag boolean")
    assert report_source_issues(frame, contract=contract, recipe=recipe).count() == 0
