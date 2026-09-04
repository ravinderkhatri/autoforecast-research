"""Lifecycle evidence from the real study, scoring and export boundaries."""

import dataclasses
import json
import pathlib

import pytest
from scripts.study_example import example_forecast

from autoforecast import _execution as events
from autoforecast import benchmark
from autoforecast.run_mlflow import RunMLflowError, export_forecast_workflow
from autoforecast.study import load_study
from autoforecast.study_preflight import StudyPreflightError
from autoforecast.study_scoring import score_study
from autoforecast.test_study_tuning import open_study
from autoforecast.test_study_workflow import profile

pytest_plugins = ["autoforecast.test_study_workflow"]


def assert_lifecycle(audit, name, outcomes):
    """Check paired lifecycle events and their durable parent for each real call."""
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    calls = [row for row in rows if row["stage"] == name]
    assert [row["status"] for row in calls] == [
        status for result, _ in outcomes for status in ("START", result)
    ]
    for index, (_, error_type) in enumerate(outcomes):
        start, finish = calls[2 * index:2 * index + 2]
        assert start["operation_id"] == finish["operation_id"]
        assert start["parent_operation"] == finish["parent_operation"]
        assert start["parent_audit"] == finish["parent_audit"]
        assert start["attempt_id"] == finish["attempt_id"] == "coverage-attempt"
        assert finish.get("error_type") == error_type
        parents = [json.loads(line) for line in
                   pathlib.Path(start["parent_audit"]).read_text().splitlines()]
        assert any(row.get("operation_id") == start["parent_operation"] for row in parents)
    return rows


def test_study_preflight_and_scoring_report_real_success_and_failure(complete, spark, tmp_path):
    audit = tmp_path / "audit.jsonl"
    with events.Execution(audit, ["host"], attempt_id="coverage-attempt"):
        with events._stage(audit, "host"):
            with pytest.raises(FileNotFoundError, match="Unknown study_id"):
                load_study("0" * 64, registry_root=complete.parent.parent, spark=spark)
            with open_study(complete, spark) as study:
                config = profile(tmp_path, selection={"origins": "all"})
                assert study.preflight(config=config, output_dir=tmp_path / "preflight-ok").report[
                    "status"
                ] == "passed"
                config = profile(tmp_path, selection={"origins": "all"}, minimum_history=100)
                with pytest.raises(StudyPreflightError):
                    study.preflight(config=config, output_dir=tmp_path / "preflight-failed")

                forecast = example_forecast(study, spark, tmp_path / "forecast", "validation")
                contract = benchmark.evaluation_contract(
                    complete, stage="validation", origin=forecast.origin,
                    horizon=forecast.horizon, metric="mae",
                )
                panel = dataclasses.replace(study.panel, path=contract["actuals"]["path"])
                result = score_study(panel, forecast, spark, contract=contract,
                                     output_dir=tmp_path / "score")
                assert result["coverage"]["scored"] == 4
                with pytest.raises(ValueError, match="forecast_accuracy_breakdown"):
                    score_study(None, None, None, contract=None, output_dir=tmp_path / "invalid",
                                forecast_accuracy_breakdown="invalid")

    assert_lifecycle(audit, "Verify study definition and frozen artifacts", [
        ("FAIL", "FileNotFoundError"), ("PASS", None),
    ])
    assert_lifecycle(audit, "Check window coverage and cohort", [
        ("PASS", None), ("FAIL", "StudyPreflightError"),
    ])
    assert_lifecycle(audit, "Calculate requested scores and coverage", [
        ("PASS", None), ("FAIL", "ValueError"),
    ])


def test_workflow_export_reports_verified_upload_and_remote_conflict(
    workflow_evidence, workflow_tracking, tmp_path,
):
    completed = workflow_evidence[1]
    client, experiment = workflow_tracking
    audit = tmp_path / "export-audit.jsonl"
    with events.Execution(audit, ["host"], attempt_id="coverage-attempt"):
        with events._stage(audit, "host"):
            options = {"tracking_uri": client.tracking_uri, "experiment_id": experiment}
            links = export_forecast_workflow(completed.path, **options)
            assert set(links) == {"validation", "test"}
            client.delete_run(links["validation"].run_id)
            with pytest.raises(RunMLflowError):
                export_forecast_workflow(completed.path, **options)
    assert_lifecycle(audit, "Export completed workflow to MLflow", [
        ("PASS", None), ("FAIL", "RunMLflowError"),
    ])
