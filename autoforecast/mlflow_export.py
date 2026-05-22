"""Optional one-way MLflow export for completed AutoForecast Run Bundles.

MLflow is a searchable presentation sink. It is intentionally unable to influence data admission,
model readiness, trial selection, held-out approval, or benchmark ranking.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from mlops._safe_files import SafeFileError, snapshot_regular_files
from mlops.provenance import ProvenanceError, validate_data_reference

from autoforecast._exact_json import load_exact_json
from autoforecast.experiments import ExperimentError, load_completed_attempt

__all__ = [
    "MLflowExportError",
    "MLflowModel",
    "MLflowRun",
    "MLflowTrainingRun",
    "export_completed_bundle",
    "publish_model_from_completed_bundle",
    "start_training_tracking",
]


class MLflowExportError(ValueError):
    """A completed bundle cannot safely be exported to MLflow."""


def _safe_source_bundle_uri(value: str | None) -> str | None:
    """Validate retained bundle references before they reach remote tags or artifacts."""
    if value is None:
        return None
    try:
        return validate_data_reference(value)
    except ProvenanceError as error:
        raise MLflowExportError("source_bundle_uri is not a safe retained reference") from error


@dataclass(frozen=True)
class MLflowRun:
    """The remote run created by a one-way export."""

    run_id: str
    experiment_id: str


@dataclass(frozen=True)
class MLflowModel:
    """A candidate model logged and versioned in the MLflow registry."""

    model_id: str
    registered_name: str
    registered_version: str


@dataclass
class MLflowTrainingRun:
    """Live MLflow record for one running attempt.

    The local Run Bundle remains authoritative.  This object exists only so MLflow can begin
    observing machine health while training is actually running, rather than after the result is
    already known.  Call :meth:`complete` only after the attempt has been finalized successfully.
    """

    run_id: str
    experiment_id: str
    _mlflow: Any
    _ended: bool = False
    _live_epochs: set[int] = field(default_factory=set)
    _live_epoch_error: str | None = None

    def log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Mirror one durable optimization epoch without making MLflow a training dependency.

        The caller writes its JSONL record before calling this method. If MLflow is temporarily
        unavailable, model training continues and completion backfills the authoritative JSONL.
        """
        if self._ended:
            raise MLflowExportError("cannot log an epoch to a closed MLflow training run")
        if type(epoch) is not int or epoch < 1:
            raise MLflowExportError("epoch must be a positive integer")
        values = {
            name: float(value)
            for name, value in metrics.items()
            if isinstance(name, str) and isinstance(value, int | float)
            and not isinstance(value, bool) and math.isfinite(float(value))
        }
        if len(values) != len(metrics):
            raise MLflowExportError("live epoch metrics must be named finite numbers")
        try:
            self._mlflow.log_metrics(values, step=epoch)
            self._live_epochs.add(epoch)
        except Exception as exc:  # MLflow is a sink, never a reason to discard a training run.
            self._live_epoch_error = type(exc).__name__

    def complete(
        self, attempt_path: str | pathlib.Path, *, source_bundle_uri: str | None = None,
    ) -> None:
        """Upload the verified, allow-listed result and finish this live run."""
        if self._ended:
            raise MLflowExportError("MLflow training run is already closed")
        try:
            with _completed_bundle_snapshot(
                attempt_path, source_bundle_uri=source_bundle_uri
            ) as (path, metadata, completion, metrics):
                publication = _log_completed_contents(
                    self._mlflow, path, metadata, completion, metrics,
                    source_bundle_uri=source_bundle_uri, include_parameters=False,
                    skipped_epochs=self._live_epochs,
                )
            self._mlflow.set_tags({
                "autoforecast.optimization_metrics": "verified_completed_bundle",
                "autoforecast.live_epoch_tracking": (
                    "degraded" if self._live_epoch_error else "recorded"
                ),
                "autoforecast.model_status": (
                    "logged_candidate_registered" if publication is not None
                    else "checkpoint_artifact_not_logged"
                ),
            })
            if self._live_epoch_error:
                self._mlflow.set_tag(
                    "autoforecast.live_epoch_tracking_error_class", self._live_epoch_error,
                )
            self._mlflow.end_run(status="FINISHED")
            self._ended = True
        except Exception:
            self._mlflow.end_run(status="FAILED")
            self._ended = True
            raise

    def fail(self) -> None:
        """Mark a live record failed when the corresponding training attempt fails."""
        if not self._ended:
            self._mlflow.end_run(status="FAILED")
            self._ended = True


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MLflowExportError("MLflow export requires: pip install '.[mlflow-tracking]'") from exc
    return mlflow


def _parameters(metadata: Mapping[str, Any]) -> dict[str, str]:
    hyperparameters = metadata.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise MLflowExportError("completed bundle has no hyperparameters")
    values: dict[str, str] = {}
    for name, value in hyperparameters.items():
        if not isinstance(name, str) or not name:
            raise MLflowExportError("hyperparameter names must be non-blank strings")
        try:
            values[name] = json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise MLflowExportError(f"hyperparameter {name!r} is not finite JSON") from exc
    return values


def _metrics(attempt: pathlib.Path) -> dict[str, float]:
    records = (
        ("evaluation/validation.json", "autoforecast.hpo-validation.v1", "evaluation/objective"),
        ("evaluation/heldout.json", "autoforecast.heldout-evaluation.v1", "heldout/objective"),
    )
    values: dict[str, float] = {}
    for relative, schema, objective in records:
        path = attempt / relative
        if not path.is_file():
            continue
        record = load_exact_json(path)
        if not isinstance(record, dict) or record.get("schema") != schema:
            raise MLflowExportError(f"invalid {relative} in completed bundle")
        value = record.get("value")
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise MLflowExportError(f"{relative} has no finite objective")
        values[objective] = float(value)
        extra = record.get("metrics", {})
        if not isinstance(extra, dict):
            raise MLflowExportError(f"{relative} metrics must be a mapping")
        for name, metric in extra.items():
            if (not isinstance(name, str) or not name
                    or not isinstance(metric, int | float) or isinstance(metric, bool)):
                raise MLflowExportError(f"{relative} metrics must be named finite numbers")
            values[name] = float(metric)
    if not values:
        raise MLflowExportError("completed bundle has neither validation nor held-out result")
    return values


def _evidence(
    metadata: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    metrics: Mapping[str, float],
    source_bundle_uri: str | None,
) -> dict[str, object]:
    source_bundle_uri = _safe_source_bundle_uri(source_bundle_uri)
    data = metadata.get("data")
    model = metadata.get("model")
    if not isinstance(data, dict):
        raise MLflowExportError("completed bundle has no data identity")
    safe_data = {
        key: data.get(key)
        for key in (
            "fingerprint",
            "observed_date_range",
            "training_date_range",
            "validation_date_range",
            "test_date_range",
            "validation_target_date_range",
            "test_target_date_range",
            "training_series",
            "validation_series",
        )
    }
    return {
        "schema": "autoforecast.mlflow-evidence.v1",
        "bundle": {
            "study_id": metadata.get("study_id"),
            "trial_id": metadata.get("trial_id"),
            "attempt_id": metadata.get("attempt_id"),
            "completion_status": completion.get("status"),
            "artifact_manifest_sha256": completion.get("artifact_manifest_sha256"),
            "source_bundle_uri": source_bundle_uri,
        },
        "model": model if isinstance(model, dict) else {},
        "comparability_fingerprint": metadata.get("comparability_fingerprint"),
        "data": safe_data,
        "metrics": dict(metrics),
    }


def _completed_bundle(
    attempt_path: str | pathlib.Path, *, source_bundle_uri: str | None,
) -> tuple[pathlib.Path, dict[str, Any], dict[str, Any], dict[str, float]]:
    _safe_source_bundle_uri(source_bundle_uri)
    path = pathlib.Path(attempt_path)
    try:
        metadata, completion = load_completed_attempt(path)
    except ExperimentError as exc:
        raise MLflowExportError(f"bundle is not a verified completed attempt: {path}") from exc
    if completion.get("status") != "completed":
        raise MLflowExportError("only completed bundles may be exported to MLflow")
    return path, metadata, completion, _metrics(path)


@contextmanager
def _completed_bundle_snapshot(
    attempt_path: str | pathlib.Path,
    *,
    source_bundle_uri: str | None,
) -> Iterator[tuple[pathlib.Path, dict[str, Any], dict[str, Any], dict[str, float]]]:
    """Yield a verified private copy so remote upload never races the authoritative bundle."""
    source, source_metadata, source_completion, _metric_values = _completed_bundle(
        attempt_path, source_bundle_uri=source_bundle_uri
    )
    manifest = load_exact_json(source / "artifact_manifest.json")
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, list):
        raise MLflowExportError("completed bundle has no artifact manifest")
    relative_paths = ["artifact_manifest.json", "completion.json"]
    relative_paths.extend(
        entry["path"]
        for entry in artifacts
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    )
    if len(relative_paths) != len(artifacts) + 2:
        raise MLflowExportError("completed bundle artifact manifest is invalid")
    with tempfile.TemporaryDirectory(prefix="autoforecast-mlflow-snapshot-") as temporary:
        snapshot = pathlib.Path(temporary) / source.name
        try:
            snapshot_regular_files(source, snapshot, relative_paths)
        except SafeFileError as error:
            raise MLflowExportError("could not snapshot completed bundle") from error
        copied = _completed_bundle(snapshot, source_bundle_uri=source_bundle_uri)
        if copied[1] != source_metadata or copied[2] != source_completion:
            raise MLflowExportError("completed bundle identity changed while it was snapshotted")
        yield copied


def _tags(path: pathlib.Path, metadata: Mapping[str, Any], completion: Mapping[str, Any], *,
          source_bundle_uri: str | None) -> dict[str, str]:
    source_bundle_uri = _safe_source_bundle_uri(source_bundle_uri)
    tags = {
        "autoforecast.export_schema": "v1",
        "autoforecast.study_id": str(metadata["study_id"]),
        "autoforecast.trial_id": str(metadata["trial_id"]),
        "autoforecast.attempt_id": str(metadata["attempt_id"]),
        "autoforecast.comparability_fingerprint": str(metadata["comparability_fingerprint"]),
        "autoforecast.data_fingerprint": str(metadata["data"]["fingerprint"]),
        "autoforecast.artifact_manifest_sha256": str(completion["artifact_manifest_sha256"]),
        "autoforecast.trace_status": (
            "recorded" if any((path / "tensorboard").glob("events.out.tfevents.*"))
            else "not_recorded"
        ),
    }
    if source_bundle_uri:
        tags["autoforecast.source_bundle_uri"] = source_bundle_uri
    return tags


def _log_epoch_history(mlflow: Any, path: pathlib.Path, *, skipped_epochs: set[int]) -> None:
    """Backfill durable epochs that were not already mirrored while training was live."""
    records = path / "metrics" / "epochs.jsonl"
    if not records.is_file():
        return
    for line_number, line in enumerate(records.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MLflowExportError(f"invalid metrics/epochs.jsonl line {line_number}") from exc
        if not isinstance(record, dict):
            raise MLflowExportError(f"metrics/epochs.jsonl line {line_number} is not an object")
        epoch = record.get("epoch")
        if type(epoch) is not int or epoch < 1:
            raise MLflowExportError(
                f"metrics/epochs.jsonl line {line_number} has no positive epoch")
        if epoch in skipped_epochs:
            continue
        for name, value in record.items():
            if name == "epoch":
                continue
            if (isinstance(name, str) and isinstance(value, int | float)
                    and not isinstance(value, bool)):
                mlflow.log_metric(name, float(value), step=epoch)


def _log_safe_artifacts(mlflow: Any, path: pathlib.Path) -> None:
    """Upload reproducibility artifacts, never a panel, raw predictions, or feature scratch."""
    descendants = tuple(path.rglob("*"))
    linked = next((item for item in descendants if item.is_symlink()), None)
    if path.is_symlink() or linked is not None:
        raise MLflowExportError(
            f"refusing to upload an attempt that contains a symlink: {linked or path}"
        )
    files = (
        ("metadata/attempt.json", "autoforecast/metadata"),
        ("artifact_manifest.json", "autoforecast/metadata"),
        ("completion.json", "autoforecast/metadata"),
        ("evaluation/validation.json", "autoforecast/evaluation"),
        ("evaluation/heldout.json", "autoforecast/evaluation"),
        ("evaluation/result.json", "autoforecast/evaluation"),
        ("metrics/epochs.jsonl", "autoforecast/metrics"),
        ("diagnostics.json", "autoforecast/diagnostics"),
        ("tracking/mlflow.json", "autoforecast/tracking"),
    )
    for relative, destination in files:
        artifact = path / relative
        if artifact.is_file():
            mlflow.log_artifact(str(artifact), destination)
    for relative, destination in (("tensorboard", "tensorboard"), ("model", "model"),
                                  ("checkpoints", "checkpoints")):
        directory = path / relative
        if directory.is_dir() and any(
            child.is_file() for child in descendants if directory in child.parents
        ):
            mlflow.log_artifacts(str(directory), destination)


def _registry_name(metadata: Mapping[str, Any]) -> str | None:
    """Name one registry family per AutoForecast model ID, never per experiment run."""
    model = metadata.get("model")
    if not isinstance(model, Mapping) or model.get("adapter") != "mft":
        return None
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise MLflowExportError(
            "mft bundle has no model ID for its registry family"
        )
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_.-")
    if not normalized:
        raise MLflowExportError("model ID cannot form a safe MLflow registry name")
    return f"autoforecast__{normalized}"


def _model_code_paths() -> list[str]:
    """Package every local runtime namespace imported by the checkpoint's model module."""
    root = pathlib.Path(__file__).resolve().parent.parent
    return [str(root / name) for name in ("autoforecast", "forecast", "provenance", "spec")]


def _log_candidate_model(
    mlflow: Any, path: pathlib.Path, metadata: Mapping[str, Any],
) -> MLflowModel | None:
    """Log and register a real PyTorch model for an adapter that has a serving checkpoint.

    A raw checkpoint is kept as immutable evidence regardless.  This additional MLflow Model is
    deliberately registered as a *candidate*: it has no deployment alias and no claim of held-out
    approval.  Its PyTorch forward interface consumes encoded tensors, not an invented tabular
    serving schema.
    """
    registry_name = _registry_name(metadata)
    checkpoint = path / "model" / "model.pt"
    if registry_name is None or not checkpoint.is_file():
        return None
    try:
        import mlflow.pytorch

        from autoforecast.adapters.mft import load_checkpoint
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise MLflowExportError(
            "logging a mft checkpoint requires PyTorch MLflow support") from exc
    model, _record = load_checkpoint(checkpoint)
    model_info = mlflow.pytorch.log_model(
        model,
        name="forecast_checkpoint",
        code_paths=_model_code_paths(),
        model_type="forecasting",
        tags={
            "autoforecast.lifecycle": "candidate",
            "autoforecast.interface": "pytorch_internal_tensor_contract",
            "autoforecast.heldout_status": "not_evaluated",
        },
    )
    registered = mlflow.register_model(model_info.model_uri, registry_name)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_tag(registry_name, "autoforecast.family", "forecasting")
    client.set_registered_model_tag(registry_name, "autoforecast.model_id", str(
        metadata["model"]["id"],
    ))
    version = str(registered.version)
    for key, value in {
        "autoforecast.lifecycle": "candidate",
        "autoforecast.deployment_eligible": "false",
        "autoforecast.heldout_status": "not_evaluated",
        "autoforecast.interface": "pytorch_internal_tensor_contract",
    }.items():
        client.set_model_version_tag(registry_name, version, key, value)
    return MLflowModel(model_info.model_id, registry_name, version)


def _log_completed_contents(
    mlflow: Any,
    path: pathlib.Path,
    metadata: Mapping[str, Any],
    completion: Mapping[str, Any],
    metrics: Mapping[str, float],
    *,
    source_bundle_uri: str | None,
    include_parameters: bool,
    skipped_epochs: set[int] | None = None,
) -> MLflowModel | None:
    mlflow.set_tags(_tags(path, metadata, completion, source_bundle_uri=source_bundle_uri))
    if include_parameters:
        mlflow.log_params(_parameters(metadata))
    _log_epoch_history(mlflow, path, skipped_epochs=skipped_epochs or set())
    mlflow.log_metrics(metrics)
    mlflow.log_dict(
        _evidence(metadata, completion, metrics=metrics, source_bundle_uri=source_bundle_uri),
        "autoforecast/evidence.json",
    )
    _log_safe_artifacts(mlflow, path)
    model = _log_candidate_model(mlflow, path, metadata)
    try:
        load_completed_attempt(path)
    except ExperimentError as error:
        raise MLflowExportError(
            f"completed attempt changed while it was exported: {path}"
        ) from error
    return model


def start_training_tracking(
    *,
    metadata: Mapping[str, Any],
    tracking_uri: str,
    experiment_name: str,
    source_bundle_uri: str | None = None,
) -> MLflowTrainingRun:
    """Begin MLflow tracking *before* a Bundle's model code runs.

    System metrics need an active run.  Metadata is supplied by the caller's fresh LocalAttempt;
    the final evidence and artifacts are accepted only from a subsequently verified completion.
    """
    if not isinstance(tracking_uri, str) or not tracking_uri:
        raise MLflowExportError("tracking_uri must be a non-blank string")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise MLflowExportError("experiment_name must be a non-blank string")
    _safe_source_bundle_uri(source_bundle_uri)
    required = ("study_id", "trial_id", "attempt_id", "comparability_fingerprint", "data")
    if (any(key not in metadata for key in required)
            or not isinstance(metadata.get("data"), Mapping)):
        raise MLflowExportError("training metadata is missing required attempt identity")
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    run_name = f"{metadata['study_id']}/{metadata['trial_id']}/{metadata['attempt_id']}"
    active = mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=run_name, log_system_metrics=True,
    )
    try:
        mlflow.set_tags({
            "autoforecast.export_schema": "v1",
            "autoforecast.study_id": str(metadata["study_id"]),
            "autoforecast.trial_id": str(metadata["trial_id"]),
            "autoforecast.attempt_id": str(metadata["attempt_id"]),
            "autoforecast.comparability_fingerprint": str(metadata["comparability_fingerprint"]),
            "autoforecast.data_fingerprint": str(metadata["data"].get("fingerprint", "")),
            "autoforecast.tracking_phase": "training",
            "autoforecast.run_kind": (
                "hpo_candidate"
                if metadata.get("purpose") == "hpo_validation_selection" else "training"
            ),
            "autoforecast.optimization_metrics": "provisional_live",
            "autoforecast.model_status": "checkpoint_artifact_pending",
        })
        mlflow.log_params(_parameters(metadata))
        return MLflowTrainingRun(
            run_id=active.info.run_id, experiment_id=experiment.experiment_id, _mlflow=mlflow,
        )
    except Exception:
        mlflow.end_run(status="FAILED")
        raise


def export_completed_bundle(
    attempt_path: str | pathlib.Path,
    *,
    tracking_uri: str,
    experiment_name: str,
    source_bundle_uri: str | None = None,
) -> MLflowRun:
    """Export one verified bundle without reading MLflow state for a decision."""
    if not isinstance(tracking_uri, str) or not tracking_uri:
        raise MLflowExportError("tracking_uri must be a non-blank string")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise MLflowExportError("experiment_name must be a non-blank string")
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    with _completed_bundle_snapshot(
        attempt_path, source_bundle_uri=source_bundle_uri
    ) as (path, metadata, completion, metrics):
        run_name = f"{metadata['study_id']}/{metadata['trial_id']}/{metadata['attempt_id']}"
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as active:
            publication = _log_completed_contents(
                mlflow, path, metadata, completion, metrics,
                source_bundle_uri=source_bundle_uri, include_parameters=True,
                skipped_epochs=set(),
            )
            mlflow.set_tag(
                "autoforecast.model_status",
                "logged_candidate_registered" if publication is not None
                else "checkpoint_artifact_not_logged",
            )
            return MLflowRun(run_id=active.info.run_id, experiment_id=experiment.experiment_id)


def publish_model_from_completed_bundle(
    attempt_path: str | pathlib.Path,
    *,
    tracking_uri: str,
) -> MLflowModel:
    """Backfill a logged candidate model into the existing MLflow run for a completed bundle.

    This changes only the non-authoritative MLflow presentation.  The completed local Bundle and
    its artifact manifest remain untouched, which permits safe migration of older runs.
    """
    with _completed_bundle_snapshot(
        attempt_path, source_bundle_uri=None
    ) as (path, metadata, _completion, _metrics_value):
        link = path / "tracking" / "mlflow.json"
        if not link.is_file():
            raise MLflowExportError("completed bundle has no MLflow tracking link to backfill")
        record = load_exact_json(link)
        if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
            raise MLflowExportError("MLflow tracking link has no run ID")
        mlflow = _mlflow()
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri)
        run = client.get_run(record["run_id"])
        tags = run.data.tags
        expected = {
            "autoforecast.study_id": str(metadata["study_id"]),
            "autoforecast.trial_id": str(metadata["trial_id"]),
            "autoforecast.attempt_id": str(metadata["attempt_id"]),
        }
        if any(tags.get(key) != value for key, value in expected.items()):
            raise MLflowExportError("MLflow run identity does not match the completed bundle")
        existing = client.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"source_run_id = '{run.info.run_id}'",
        )
        if existing:
            raise MLflowExportError(
                "MLflow run already has a logged model; refusing duplicate backfill")
        with mlflow.start_run(run_id=run.info.run_id):
            publication = _log_candidate_model(mlflow, path, metadata)
            if publication is None:
                raise MLflowExportError(
                    "completed bundle has no supported model checkpoint to log"
                )
            mlflow.set_tag("autoforecast.model_status", "logged_candidate_registered")
            return publication
