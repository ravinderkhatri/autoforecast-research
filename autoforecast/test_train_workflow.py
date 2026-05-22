"""Functional flow checks for the approval-gated training coordinator."""

from __future__ import annotations

import pathlib
from typing import Literal

import yaml

from autoforecast.model_api import (
    CheckReport,
    CheckResult,
    DataFacts,
    ModelCheckInput,
    ResolvedRun,
    RunApproval,
    RunIncident,
)
from autoforecast.train_workflow import (
    ModelOutcome,
    RunRequest,
    RunSummary,
    load_run_request,
    run_training_workflow,
)


def _input(model_id: str) -> ModelCheckInput:
    facts = DataFacts(
        "2026-09-10", "a" * 64, ("2026-01-01", "2026-09-10"), 2, 1, 2, 0, 0, {}, {}, None, {}
    )
    run = ResolvedRun(model_id, "seasonal_naive", "2026-09-10", 1)
    return ModelCheckInput(
        run, facts, "D", "sales_given_availability", False, 1, 1, 1, 1, None, None, 0, 1
    )


def _checks(model_id: str, passed: bool = True) -> tuple[ModelCheckInput, CheckReport, RunApproval]:
    input = _input(model_id)
    report = CheckReport(
        "model",
        input.data,
        (CheckResult("ready", "Ready", "passed" if passed else "failed", "ok"),),
        model_id,
    )
    return (
        input,
        report,
        RunApproval(
            passed,
            model_id,
            input.run.adapter,
            input.run.config_hash,
            input.data.data_fingerprint,
            "ok",
        ),
    )


def _request(models: tuple[str, ...] | Literal["all"] = "all") -> RunRequest:
    return RunRequest("functional-run", "dataset.yaml", "models", models, "2026-09-10", 1)


def test_run_summary_trained_models_returns_completed_models_in_outcome_order(
    tmp_path: pathlib.Path,
) -> None:
    summary = RunSummary(
        run_id="run-1",
        path=tmp_path,
        outcomes=(
            ModelOutcome("first", "completed", "trained"),
            ModelOutcome("second", "failed", "worker failed"),
            ModelOutcome("third", "completed", "trained"),
            ModelOutcome("fourth", "not_ready", "not eligible"),
        ),
        data_outcome=RunIncident("data-read", "Data read", "available"),
        notification_error=None,
    )

    assert summary.trained_models == ("first", "third")


def test_request_yaml_accepts_all_models_without_a_runtime_data_path(
    tmp_path: pathlib.Path,
) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text(
        """schema_version: 1
run:
  name: functional-run
  dataset_contract: configs/datasets/m5_daily.yaml
  model_catalog: configs/models
  models: all
  origin: '2026-09-10'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
""",
        encoding="utf-8",
    )
    assert load_run_request(request_path).models == "all"


def test_hmfnet_request_marks_new_precision_policy_without_changing_evaluation_contract(
    tmp_path: pathlib.Path,
) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text(
        """schema_version: 3
run:
  name: new-torch-run
  dataset_contract: configs/datasets/m5_daily.yaml
  model_catalog: configs/models
  models: [sft]
  origin: '2016-04-24'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
  evaluation:
    missing_actual: no_data
    metric: wrmsse
    input_mode: post_training_score_only
    evidence_role: rolling_validation
""",
        encoding="utf-8",
    )
    request = load_run_request(request_path)
    assert request.schema_version == 3
    assert request.evaluation is not None
    assert request.evaluation.input_mode == "post_training_score_only"


def test_run_request_refuses_duplicate_yaml_keys_and_boolean_version(
    tmp_path: pathlib.Path,
) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text("schema_version: 1\nschema_version: 1\nrun: {}\n")
    try:
        load_run_request(request_path)
    except ValueError as error:
        assert "duplicate YAML mapping key" in str(error)
    else:
        raise AssertionError("run request accepted an ambiguous YAML mapping")

    request_path.write_text("schema_version: true\nrun: {}\n")
    try:
        load_run_request(request_path)
    except ValueError as error:
        assert "schema_version 1, 2, or 3" in str(error)
    else:
        raise AssertionError("run request accepted bool as an integer schema version")


def test_run_request_refuses_coerced_paths_and_noncanonical_origin(
    tmp_path: pathlib.Path,
) -> None:
    template = """schema_version: 1
run:
  name: functional-run
  dataset_contract: VALUE
  model_catalog: configs/models
  models: all
  origin: 'ORIGIN'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
"""
    request_path = tmp_path / "run.yaml"
    for value, origin, message in (
        ("[configs/datasets/m5_daily.yaml]", "2026-09-10", "path string"),
        ("configs/datasets/m5_daily.yaml", "2026-9-10", "ISO date"),
    ):
        request_path.write_text(template.replace("VALUE", value).replace("ORIGIN", origin))
        try:
            load_run_request(request_path)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("run request accepted a coerced path or noncanonical origin")


def test_sft_request_makes_evaluation_semantics_explicit(tmp_path: pathlib.Path) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text(
        """schema_version: 2
run:
  name: functional-run
  dataset_contract: configs/datasets/m5_daily.yaml
  model_catalog: configs/models
  models: [seasonal_naive]
  origin: '2026-09-10'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
  evaluation:
    missing_actual: no_data
    metric: rmse
""",
        encoding="utf-8",
    )
    evaluation = load_run_request(request_path).evaluation
    assert evaluation is not None
    assert evaluation.missing_actual == "no_data" and evaluation.metric == "rmse"
    assert evaluation.input_mode == "training_panel"


def test_sft_request_accepts_the_explicit_m5_wrmsse_metric(tmp_path: pathlib.Path) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text(
        """schema_version: 2
run:
  name: m5-private-replay
  dataset_contract: configs/datasets/m5_daily.yaml
  model_catalog: configs/models
  models: [seasonal_naive]
  origin: '2016-05-22'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
  evaluation:
    missing_actual: no_data
    metric: wrmsse
""",
        encoding="utf-8",
    )
    evaluation = load_run_request(request_path).evaluation
    assert evaluation is not None and evaluation.metric == "wrmsse"


def test_sft_request_can_require_a_separate_post_training_score_input(
    tmp_path: pathlib.Path,
) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text(
        """schema_version: 2
run:
  name: m5-private-replay
  dataset_contract: configs/datasets/m5_daily.yaml
  model_catalog: configs/models
  models: [seasonal_naive]
  origin: '2016-05-22'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
  evaluation:
    missing_actual: no_data
    metric: wrmsse
    input_mode: post_training_score_only
    evidence_role: archived_private_replay_not_blind
""",
        encoding="utf-8",
    )
    evaluation = load_run_request(request_path).evaluation
    assert evaluation is not None
    assert evaluation.input_mode == "post_training_score_only"
    assert evaluation.evidence_role == "archived_private_replay_not_blind"


def test_m5_sft_score_only_requests_do_not_rewrite_frozen_mft_requests() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    for stem in ("m5_public_holdout", "m5_private_evaluation"):
        frozen = yaml.safe_load(
            (root / "configs/autoforecast" / f"{stem}_deep_v1.yaml").read_text()
        )
        current = yaml.safe_load(
            (root / "configs/autoforecast" / f"{stem}_deep_v2.yaml").read_text()
        )
        assert set(frozen["run"]["evaluation"]) == {"missing_actual", "metric"}
        assert current["run"]["evaluation"]["input_mode"] == "post_training_score_only"


def test_sft_request_binds_a_source_controlled_calendar_event_recipe(tmp_path: pathlib.Path) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text(
        """schema_version: 2
run:
  name: calendar-functional-run
  dataset_contract: configs/datasets/m5_daily.yaml
  model_catalog: configs/models
  models: [mft]
  origin: '2016-05-22'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
  features:
    calendar_events:
      max_support: 30
      events: [[Memorial, '2016-05-30', null]]
  evaluation:
    missing_actual: no_data
    metric: wrmsse
""",
        encoding="utf-8",
    )
    request = load_run_request(request_path)
    assert request.event_max_support == 30
    assert request.calendar_events == (("Memorial", "2016-05-30", None),)


def test_mft_request_refuses_sft_calendar_features(tmp_path: pathlib.Path) -> None:
    request_path = tmp_path / "run.yaml"
    request_path.write_text(
        """schema_version: 1
run:
  name: calendar-functional-run
  dataset_contract: configs/datasets/m5_daily.yaml
  model_catalog: configs/models
  models: [seasonal_naive]
  origin: '2016-05-22'
  horizon: 28
  tracking:
    mlflow: {enabled: false}
  features:
    calendar_events:
      max_support: 30
      events: []
""",
        encoding="utf-8",
    )

    try:
        load_run_request(request_path)
    except ValueError as error:
        assert "exact run mapping" in str(error)
    else:
        raise AssertionError("schema version 1 accepted version 2 calendar features")


def test_data_failure_blocks_every_model_and_publishes_a_report(tmp_path: pathlib.Path) -> None:
    data = CheckReport("data", _input("a").data, (CheckResult("bad", "Bad", "failed", "bad"),))
    ran: list[str] = []
    summary = run_training_workflow(
        _request(),
        output_root=tmp_path,
        data_outcome=data,
        checked_models={"a": _checks("a")},
        execute_isolated=lambda context: ran.append(context.model_id),
    )
    assert not ran and summary.exit_code == 2 and summary.outcomes[0].status == "blocked"
    assert (summary.path / "report.html").is_file()


def test_all_models_are_reported_when_data_fails_before_model_checks(
    tmp_path: pathlib.Path,
) -> None:
    data = CheckReport("data", _input("a").data, (CheckResult("bad", "Bad", "failed", "bad"),))
    summary = run_training_workflow(
        _request(),
        output_root=tmp_path,
        data_outcome=data,
        checked_models={},
        execute_isolated=lambda context: None,
        resolved_model_ids=("first", "second"),
    )
    assert [item.model_id for item in summary.outcomes] == ["first", "second"]
    assert all(item.status == "blocked" for item in summary.outcomes)


def test_missing_model_runtime_is_reported_without_running_that_model(
    tmp_path: pathlib.Path,
) -> None:
    ran: list[str] = []
    data = CheckReport("data", _input("a").data, (CheckResult("ok", "OK", "passed", "ok"),))
    summary = run_training_workflow(
        _request(("ready", "missing_runtime")),
        output_root=tmp_path,
        data_outcome=data,
        checked_models={"ready": _checks("ready")},
        execute_isolated=lambda context: ran.append(context.model_id),
        execution_blockers={"missing_runtime": "Runtime 'classical' is unavailable."},
    )
    assert ran == ["ready"]
    assert [item.status for item in summary.outcomes] == ["completed", "not_ready"]
    assert "Runtime 'classical'" in summary.outcomes[1].summary


def test_ready_model_runs_while_unready_and_failed_models_are_reported(
    tmp_path: pathlib.Path,
) -> None:
    ran: list[str] = []
    summary = run_training_workflow(
        _request(),
        output_root=tmp_path,
        data_outcome=CheckReport(
            "data", _input("a").data, (CheckResult("ok", "OK", "passed", "ok"),)
        ),
        checked_models={
            "ready": _checks("ready"),
            "unready": _checks("unready", False),
            "broken": _checks("broken"),
        },
        execute_isolated=lambda context: (_ for _ in ()).throw(RuntimeError("worker exited"))
        if context.model_id == "broken"
        else ran.append(context.model_id),
    )
    assert ran == ["ready"]
    assert [item.status for item in summary.outcomes] == ["completed", "not_ready", "failed"]
    assert summary.exit_code == 0


def test_technical_data_incident_blocks_without_becoming_a_quality_failure(
    tmp_path: pathlib.Path,
) -> None:
    summary = run_training_workflow(
        _request(("a",)),
        output_root=tmp_path,
        data_outcome=RunIncident("spark_read", "Data could not be inspected", "trace"),
        checked_models={"a": _checks("a")},
        execute_isolated=lambda context: None,
    )
    assert summary.outcomes[0].status == "blocked" and summary.exit_code == 2


def test_notification_failure_does_not_change_a_completed_run(tmp_path: pathlib.Path) -> None:
    data = CheckReport("data", _input("a").data, (CheckResult("ok", "OK", "passed", "ok"),))
    summary = run_training_workflow(
        _request(("a",)),
        output_root=tmp_path,
        data_outcome=data,
        checked_models={"a": _checks("a")},
        execute_isolated=lambda context: None,
        notify=lambda _: (_ for _ in ()).throw(RuntimeError("mail unavailable")),
    )
    assert summary.outcomes[0].status == "completed"
    assert summary.exit_code == 0
    assert summary.notification_error == "mail unavailable"
