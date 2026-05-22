"""One small coordinator for an approval-gated AutoForecast training request.

It owns state, immutable run-group publication, the user-facing report, and terminal notices.
Data inspection, model readiness, and model execution remain separate responsibilities.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import pathlib
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast.model_api import CheckReport, ModelCheckInput, RunApproval, RunIncident

__all__ = [
    "CheckedModelRun",
    "ModelOutcome",
    "RunRequest",
    "RunSummary",
    "TrainWorkflowError",
    "load_run_request",
    "run_training_workflow",
]

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
ModelStatus = Literal["blocked", "not_ready", "failed", "completed"]
EvaluationMetric = Literal["rmse", "mae", "wape", "wrmsse"]
EvaluationInputMode = Literal["training_panel", "post_training_score_only"]


class TrainWorkflowError(ValueError):
    """A train request or its explicit dependencies are unsafe."""


@dataclass(frozen=True)
class EvaluationRequest:
    """Explicit scoring semantics and the data boundary for one completed run."""

    missing_actual: Literal["sparse_zero", "no_data"]
    metric: EvaluationMetric
    input_mode: EvaluationInputMode = "training_panel"
    evidence_role: str = "run_local_exploratory"
    selection_lock: str | None = None


@dataclass(frozen=True)
class RunRequest:
    """Source-controlled intent; runtime data and output locations stay outside this record."""

    name: str
    dataset_contract: str
    model_catalog: str
    models: tuple[str, ...] | Literal["all"]
    origin: str
    horizon: int
    mlflow_enabled: bool = False
    evaluation: EvaluationRequest | None = None
    calendar_events: tuple[tuple[str, str, str | None], ...] = ()
    event_max_support: int = 30
    schema_version: int = 3
    precision: str = "16-mixed"


@dataclass(frozen=True)
class CheckedModelRun:
    """The only execution input after data and model checks have passed."""

    model_id: str
    approval: RunApproval
    input: ModelCheckInput
    attempt_dir: pathlib.Path


@dataclass(frozen=True)
class ModelOutcome:
    model_id: str
    status: ModelStatus
    summary: str
    incident: RunIncident | None = None


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    path: pathlib.Path
    outcomes: tuple[ModelOutcome, ...]
    data_outcome: CheckReport | RunIncident
    notification_error: str | None

    @property
    def trained_models(self) -> tuple[str, ...]:
        return tuple(item.model_id for item in self.outcomes if item.status == "completed")

    @property
    def exit_code(self) -> int:
        return 0 if self.trained_models else 2


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise TrainWorkflowError(f"{label} must match {_ID.pattern}")
    return value


def _canonical_time_text(value: object, label: str) -> str:
    """Accept a canonical date or timezone-naive exact-hour timestamp without choosing cadence."""
    if type(value) is not str:
        raise TrainWorkflowError(f"{label} must be a canonical ISO date or exact-hour timestamp")
    try:
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            parsed: dt.date | dt.datetime = dt.date.fromisoformat(value)
            canonical = parsed.isoformat()
        elif re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:00:00", value):
            parsed = dt.datetime.fromisoformat(value)
            canonical = parsed.isoformat(timespec="seconds")
        else:
            raise ValueError
    except ValueError as exc:
        raise TrainWorkflowError(
            f"{label} must be a canonical ISO date or exact-hour timestamp"
        ) from exc
    if canonical != value:
        raise TrainWorkflowError(
            f"{label} must be a canonical ISO date or exact-hour timestamp"
        )
    return value


def load_run_request(path: str | pathlib.Path) -> RunRequest:
    """Load the deliberately small request YAML; it has no arbitrary executor kwargs."""
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise TrainWorkflowError(f"run request must be a regular file: {source}")
    try:
        value = loads_exact_yaml(source.read_bytes())
    except ExactYAMLError as exc:
        raise TrainWorkflowError(f"invalid YAML in {source}: {exc}") from exc
    if type(value) is not dict or set(value) != {"schema_version", "run"}:
        raise TrainWorkflowError("run request requires exactly schema_version and run")
    run = value.get("run")
    expected = {
        "name",
        "dataset_contract",
        "model_catalog",
        "models",
        "origin",
        "horizon",
        "tracking",
    }
    version = value.get("schema_version")
    optional = {"features", "evaluation", "precision"}
    if (
        type(version) is not int
        or version != 3
        or type(run) is not dict
        or not expected <= set(run)
        or set(run) - expected - optional
    ):
        raise TrainWorkflowError(
            "run request must be schema_version 3 with an exact run mapping"
        )
    precision = run.get("precision", "16-mixed")
    if precision not in ("32-true", "16-mixed"):
        raise TrainWorkflowError("run.precision must be 32-true or 16-mixed")
    models = run["models"]
    if models != "all":
        if type(models) is not list or not models:
            raise TrainWorkflowError("run.models must be 'all' or a non-empty model-id list")
        model_ids = tuple(_id(item, "run.models item") for item in models)
        if len(set(model_ids)) != len(model_ids):
            raise TrainWorkflowError("run.models must not contain duplicates")
    else:
        model_ids = "all"
    origin = _canonical_time_text(run["origin"], "run.origin")
    for field in ("dataset_contract", "model_catalog"):
        if type(run[field]) is not str or not run[field].strip():
            raise TrainWorkflowError(f"run.{field} must be a non-blank path string")
    horizon = run["horizon"]
    if type(horizon) is not int or horizon < 1:
        raise TrainWorkflowError("run.horizon must be a positive integer")
    tracking = run["tracking"]
    if type(tracking) is not dict or set(tracking) != {"mlflow"}:
        raise TrainWorkflowError("run.tracking requires exactly mlflow")
    mlflow = tracking["mlflow"]
    if (
        type(mlflow) is not dict
        or set(mlflow) != {"enabled"}
        or type(mlflow["enabled"]) is not bool
    ):
        raise TrainWorkflowError("run.tracking.mlflow requires exactly boolean enabled")
    evaluation = None
    if "evaluation" in run:
        raw_evaluation = run["evaluation"]
        base_evaluation = {"missing_actual", "metric"}
        extended_evaluation = base_evaluation | {"input_mode", "evidence_role"}
        locked_evaluation = extended_evaluation | {"selection_lock"}
        if type(raw_evaluation) is not dict or set(raw_evaluation) not in (
            base_evaluation,
            extended_evaluation,
            locked_evaluation,
        ):
            raise TrainWorkflowError(
                "run.evaluation requires missing_actual and metric, with optional "
                "input_mode/evidence_role, plus an optional selection_lock"
            )
        missing_actual = raw_evaluation["missing_actual"]
        metric = raw_evaluation["metric"]
        if (type(missing_actual) is not str or type(metric) is not str
                or missing_actual not in ("sparse_zero", "no_data") or metric not in (
            "rmse",
            "mae",
            "wape",
            "wrmsse",
        )):
            raise TrainWorkflowError("run.evaluation has unsupported missing_actual or metric")
        input_mode: EvaluationInputMode = "training_panel"
        evidence_role = "run_local_exploratory"
        selection_lock: str | None = None
        if set(raw_evaluation) in (extended_evaluation, locked_evaluation):
            mode = raw_evaluation["input_mode"]
            if type(mode) is not str or mode not in ("training_panel", "post_training_score_only"):
                raise TrainWorkflowError("run.evaluation.input_mode is unsupported")
            input_mode = mode
            evidence_role = _id(raw_evaluation["evidence_role"], "run.evaluation.evidence_role")
            if set(raw_evaluation) == locked_evaluation:
                raw_lock = raw_evaluation["selection_lock"]
                if not isinstance(raw_lock, str) or not raw_lock.strip():
                    raise TrainWorkflowError(
                        "run.evaluation.selection_lock must be a non-empty path"
                    )
                selection_lock = raw_lock
        evaluation = EvaluationRequest(
            missing_actual, metric, input_mode, evidence_role, selection_lock
        )
    events: tuple[tuple[str, str, str | None], ...] = ()
    event_max_support = 30
    if "features" in run:
        features = run["features"]
        if type(features) is not dict or set(features) != {"calendar_events"}:
            raise TrainWorkflowError("run.features requires exactly calendar_events")
        calendar_events = features["calendar_events"]
        if type(calendar_events) is not dict or set(calendar_events) != {"max_support", "events"}:
            raise TrainWorkflowError("calendar_events requires exactly max_support and events")
        event_max_support, raw_events = calendar_events["max_support"], calendar_events["events"]
        if (
            type(event_max_support) is not int
            or event_max_support < 1
            or type(raw_events) is not list
        ):
            raise TrainWorkflowError(
                "calendar_events needs positive max_support and an events list"
            )
        parsed: list[tuple[str, str, str | None]] = []
        for item in raw_events:
            if type(item) is not list or len(item) != 3:
                raise TrainWorkflowError("each calendar event must be [name, date, known_at]")
            name, when, known_at = item
            if not isinstance(name, str) or not name or not isinstance(when, str):
                raise TrainWorkflowError("calendar event name and date must be non-empty strings")
            _canonical_time_text(when, "calendar event time")
            if known_at is not None:
                _canonical_time_text(known_at, "calendar event known_at")
            parsed.append((name, when, known_at))
        events = tuple(parsed)
    return RunRequest(
        _id(run["name"], "run.name"),
        run["dataset_contract"],
        run["model_catalog"],
        model_ids,
        origin,
        horizon,
        mlflow["enabled"],
        evaluation,
        events,
        event_max_support,
        version,
        precision,
    )


def _new_group(root: pathlib.Path, name: str) -> tuple[str, pathlib.Path]:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{name}-{stamp}-{uuid.uuid4().hex[:8]}"
    path = root / run_id
    path.mkdir(parents=True)
    return run_id, path


def _publish(summary: RunSummary) -> None:
    record = {
        "schema": "autoforecast.train-run.v1",
        "run_id": summary.run_id,
        "exit_code": summary.exit_code,
        "data": summary.data_outcome.as_record(),
        "models": [
            {
                "model_id": item.model_id,
                "status": item.status,
                "summary": item.summary,
                "incident": None if item.incident is None else item.incident.as_record(),
            }
            for item in summary.outcomes
        ],
        "notification_error": summary.notification_error,
    }
    (summary.path / "run-summary.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    rows = (
        "".join(
            f"<tr><td>{html.escape(item.model_id)}</td><td>{html.escape(item.status)}</td>"
            f"<td>{html.escape(item.summary)}</td></tr>"
            for item in summary.outcomes
        )
        or "<tr><td colspan='3'>No model was run.</td></tr>"
    )
    data_status = (
        "passed"
        if isinstance(summary.data_outcome, CheckReport) and summary.data_outcome.accepted
        else "blocked or not checked"
    )
    (summary.path / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>AutoForecast run</title>"
        f"<h1>AutoForecast run: {html.escape(summary.run_id)}</h1>"
        f"<p>Data checks: {data_status}</p>"
        "<table><tr><th>Model</th><th>Status</th><th>Explanation</th></tr>" + rows + "</table>",
        encoding="utf-8",
    )


def run_training_workflow(
    request: RunRequest,
    *,
    output_root: str | pathlib.Path,
    data_outcome: CheckReport | RunIncident,
    checked_models: Mapping[str, tuple[ModelCheckInput, CheckReport, RunApproval]],
    execute_isolated: Callable[[CheckedModelRun], None],
    notify: Callable[[RunSummary], None] | None = None,
    resolved_model_ids: tuple[str, ...] | None = None,
    execution_blockers: Mapping[str, str] | None = None,
) -> RunSummary:
    """Run only approved/runtime-ready models; a model error cannot reclassify data quality."""
    run_id, path = _new_group(pathlib.Path(output_root), request.name)
    if request.models == "all":
        requested = tuple(checked_models) if resolved_model_ids is None else resolved_model_ids
        if not requested:
            raise TrainWorkflowError("an all-model request resolved no catalog candidates")
    else:
        requested = request.models
    blockers = dict(execution_blockers or {})
    unknown_blockers = sorted(set(blockers) - set(requested))
    if unknown_blockers:
        raise TrainWorkflowError(
            f"execution blockers reference models absent from this request: {unknown_blockers}"
        )
    if isinstance(data_outcome, RunIncident) or not data_outcome.accepted:
        outcomes = tuple(
            ModelOutcome(model_id, "blocked", "Shared data checks did not pass.")
            for model_id in requested
        )
    else:
        results: list[ModelOutcome] = []
        for model_id in requested:
            if reason := blockers.get(model_id):
                results.append(ModelOutcome(model_id, "not_ready", reason))
                continue
            try:
                input, checks, approval = checked_models[model_id]
            except KeyError:
                results.append(
                    ModelOutcome(model_id, "not_ready", "No model readiness result was supplied.")
                )
                continue
            if not checks.accepted or not approval.approved:
                results.append(
                    ModelOutcome(
                        model_id, "not_ready", "This model's readiness checks did not pass."
                    )
                )
                continue
            context = CheckedModelRun(
                model_id, approval, input, path / "attempts" / model_id / uuid.uuid4().hex
            )
            try:
                execute_isolated(context)
            except BaseException as exc:
                incident = RunIncident("model_execution", "Model training failed", str(exc))
                results.append(ModelOutcome(model_id, "failed", incident.summary, incident))
            else:
                results.append(ModelOutcome(model_id, "completed", "Training completed."))
        outcomes = tuple(results)
    provisional = RunSummary(run_id, path, outcomes, data_outcome, None)
    _publish(provisional)
    notification_error = None
    if notify is not None:
        try:
            notify(provisional)
        except Exception as exc:  # notification is a terminal, non-governing side effect
            notification_error = str(exc)
    summary = RunSummary(run_id, path, outcomes, data_outcome, notification_error)
    _publish(summary)
    return summary
