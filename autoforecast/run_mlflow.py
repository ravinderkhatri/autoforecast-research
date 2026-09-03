"""One-way MLflow export for a sealed generic run bundle.

The run bundle stays authoritative.  This exporter verifies it first, uploads only safe evidence
and model artifacts, and never registers or aliases a model: promotion remains a separate gate.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from mlops import ProvenanceError, load_completed_run_bundle, sanitize_web_url
from mlops._safe_files import SafeFileError, snapshot_regular_files

__all__ = [
    "LiveRunMLflow",
    "RunMLflowError",
    "RunMLflowLink",
    "export_run_bundle",
    "export_forecast_workflow",
    "start_live_run_bundle_tracking",
]


class RunMLflowError(ValueError):
    """A sealed bundle cannot be mirrored safely to MLflow."""


@contextmanager
def _completed_run_snapshot(
    bundle_path: str | pathlib.Path,
) -> Iterator[tuple[pathlib.Path, dict[str, Any]]]:
    """Yield a verified private copy so remote upload cannot race mutable local paths."""
    source = pathlib.Path(bundle_path)
    try:
        source_completion = load_completed_run_bundle(source)
    except ProvenanceError as error:
        raise RunMLflowError(f"bundle is not verified and completed: {source}") from error
    manifest = _load(source / "artifact_manifest.json")
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, list):
        raise RunMLflowError("completed bundle has no artifact manifest")
    relative_paths = ["artifact_manifest.json", "completion.json"]
    relative_paths.extend(
        entry["path"]
        for entry in artifacts
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    )
    if len(relative_paths) != len(artifacts) + 2:
        raise RunMLflowError("completed bundle artifact manifest is invalid")
    with tempfile.TemporaryDirectory(prefix="autoforecast-run-mlflow-snapshot-") as temporary:
        snapshot = pathlib.Path(temporary) / source.name
        try:
            snapshot_regular_files(source, snapshot, relative_paths)
        except SafeFileError as error:
            raise RunMLflowError("could not snapshot completed run bundle") from error
        try:
            completion = load_completed_run_bundle(snapshot)
        except ProvenanceError as error:
            raise RunMLflowError("completed run bundle changed while it was snapshotted") from error
        if completion != source_completion:
            raise RunMLflowError("completed run bundle identity changed while it was snapshotted")
        yield snapshot, completion


@dataclass(frozen=True)
class RunMLflowLink:
    tracking_uri: str
    experiment_id: str
    run_id: str

    def as_record(self) -> dict[str, str]:
        record = {
            "schema": "autoforecast.generic-mlflow-link.v2",
            "tracking_uri_sha256": hashlib.sha256(self.tracking_uri.encode()).hexdigest(),
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
        }
        web_url = sanitize_web_url(self.tracking_uri)
        if web_url is not None:
            record["tracking_web_url"] = web_url
        return record


@dataclass
class LiveRunMLflow:
    """One parent-owned MLflow run for an isolated generic training attempt.

    The coordinator starts this before a worker begins.  A Lightning worker may attach using
    :attr:`worker_context`, but only the coordinator verifies the sealed bundle and closes the
    run.  This prevents the old "worker run plus exporter run" duplication.
    """

    link: RunMLflowLink
    _mlflow: Any
    _ended: bool = False
    _failed: bool = False

    @property
    def worker_context(self) -> dict[str, str]:
        return {
            "tracking_uri": self.link.tracking_uri,
            "run_id": self.link.run_id,
            "experiment_id": self.link.experiment_id,
        }

    def complete(
        self,
        bundle_path: str | pathlib.Path,
        *,
        comparison_path: str | pathlib.Path | None = None,
    ) -> None:
        """Verify, mirror, and close the exact run that observed training live."""
        if self._ended:
            raise RunMLflowError("live MLflow run is already closed")
        try:
            with _completed_run_snapshot(bundle_path) as (root, completion):
                if completion.get("status") != "completed":
                    raise RunMLflowError("only completed bundles may finish a live MLflow run")
                resolved = _load(root / "metadata" / "resolved-run.json")
                plan = _load(root / "metadata" / "execution-plan.json")
                model_id = resolved.get("model_id")
                if not isinstance(model_id, str) or not model_id:
                    raise RunMLflowError("resolved run has no model ID")
                comparison = (
                    None if comparison_path is None else _load(pathlib.Path(comparison_path))
                )
                parameters = _parameters(plan)
                if parameters:
                    self._mlflow.log_params(parameters)
                for epoch, metrics in _epoch_metrics(root).items():
                    self._mlflow.log_metrics(metrics, step=epoch)
                metrics = _comparison_metrics(comparison, model_id)
                if metrics:
                    self._mlflow.log_metrics(metrics)
                _log_safe_artifacts(self._mlflow, root)
            self._mlflow.set_tag("autoforecast.tracking_phase", "verified_completed_bundle")
            self._mlflow.end_run(status="FINISHED")
            self._ended = True
        except Exception as error:
            try:
                self._mlflow.end_run(status="FAILED")
            except Exception:
                pass
            self._ended = True
            self._failed = True
            if isinstance(error, RunMLflowError):
                raise
            raise RunMLflowError(
                f"could not complete live MLflow projection: {type(error).__name__}"
            ) from error

    def project_comparison(
        self,
        comparison_path: str | pathlib.Path,
        *,
        model_id: str,
    ) -> None:
        """Backfill comparison metrics after all sequential model runs have closed."""
        if not self._ended or self._failed:
            raise RunMLflowError("comparison requires a successfully completed live MLflow run")
        comparison = _load(pathlib.Path(comparison_path))
        metrics = _comparison_metrics(comparison, model_id)
        try:
            self._mlflow.set_tracking_uri(self.link.tracking_uri)
            with self._mlflow.start_run(run_id=self.link.run_id):
                if metrics:
                    self._mlflow.log_metrics(metrics)
                self._mlflow.set_tag(
                    "autoforecast.comparison_projection", "verified_run_comparison"
                )
        except Exception as error:
            raise RunMLflowError(
                f"could not project run comparison to MLflow: {type(error).__name__}"
            ) from error

    def fail(self) -> None:
        """Close the coordinator-owned live run after a worker failure."""
        if not self._ended:
            try:
                self._mlflow.end_run(status="FAILED")
            except Exception as error:
                raise RunMLflowError(
                    f"could not close failed MLflow projection: {type(error).__name__}"
                ) from error
            finally:
                self._ended = True
                self._failed = True


def _load(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunMLflowError(f"cannot read required bundle artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise RunMLflowError(f"bundle artifact must be a JSON object: {path.name}")
    return value


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - optional runtime boundary
        raise RunMLflowError(
            "MLflow tracking requires the standard AutoForecast installation with dependencies"
        ) from error
    return mlflow


def _parameters(plan: Mapping[str, object]) -> dict[str, str]:
    effective = plan.get("effective_parameters")
    if not isinstance(effective, Mapping):
        return {}
    values: dict[str, str] = {}
    for name, value in effective.items():
        if not isinstance(name, str) or not name:
            raise RunMLflowError("execution plan has an invalid effective parameter")
        values[name] = json.dumps(value, sort_keys=True, allow_nan=False)
    return values


def _parameter_source_tags(root: pathlib.Path) -> dict[str, str]:
    path = root / "metadata" / "parameter-resolution.json"
    if not path.is_file():
        return {}
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunMLflowError("cannot read parameter-resolution.json") from error
    if not isinstance(record, dict):
        raise RunMLflowError("parameter resolution must be a JSON object")
    source = record.get("source")
    kind = source.get("kind") if isinstance(source, dict) else None
    if not isinstance(kind, str) or not kind:
        raise RunMLflowError("parameter resolution has no source kind")
    return {
        "autoforecast.parameter_source": kind,
        "autoforecast.parameter_resolution_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _comparison_metrics(comparison: Mapping[str, object] | None, model_id: str) -> dict[str, float]:
    if comparison is None:
        return {}
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list):
        return {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("model_id") != model_id:
            continue
        score = candidate.get("score")
        if not isinstance(score, dict):
            return {}
        return {
            f"evaluation.{name}": float(value)
            for name, value in score.items()
            if isinstance(name, str)
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        }
    return {}


def _epoch_metrics(root: pathlib.Path) -> dict[int, dict[str, float]]:
    """Load durable epoch metrics without treating a dashboard event as evidence."""
    path = root / "metrics" / "epochs.jsonl"
    if not path.is_file():
        return {}
    values: dict[int, dict[str, float]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunMLflowError(f"invalid metrics/epochs.jsonl line {line_number}") from error
        if not isinstance(record, dict) or type(record.get("epoch")) is not int:
            raise RunMLflowError(f"metrics/epochs.jsonl line {line_number} has no integer epoch")
        metrics = {
            name: float(value)
            for name, value in record.items()
            if name != "epoch"
            and isinstance(name, str)
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        }
        values[record["epoch"]] = metrics
    return values


def _log_safe_artifacts(mlflow: Any, root: pathlib.Path) -> None:
    descendants = tuple(root.rglob("*"))
    linked = next((item for item in descendants if item.is_symlink()), None)
    if root.is_symlink() or linked is not None:
        raise RunMLflowError(
            f"refusing to upload a run bundle that contains a symlink: {linked or root}"
        )
    files = (
        "artifact_manifest.json",
        "completion.json",
        "checks/data.json",
        "checks/model.json",
        "evaluation/forecast.json",
        "metadata/execution-plan.json",
        "metadata/resolved-run.json",
        "metadata/parameter-resolution.json",
        "metadata/runtime-check.json",
        "provenance/source.json",
        "provenance/runtime.json",
        "provenance/data.json",
        "provenance/manifest.json",
        "provenance/checkpoint.json",
        "provenance/parameter-source/capsule.json",
        "provenance/parameter-source/study-config.yaml",
        "provenance/parameter-source/study/selection.json",
        "metrics/epochs.jsonl",
    )
    for relative in files:
        artifact = root / relative
        if artifact.is_file():
            mlflow.log_artifact(str(artifact), "autoforecast")
    for relative, destination in (("model", "model"), ("tensorboard", "tensorboard")):
        directory = root / relative
        if directory.is_dir() and any(
            item.is_file() for item in descendants if directory in item.parents
        ):
            mlflow.log_artifacts(str(directory), destination)


def start_live_run_bundle_tracking(
    bundle_path: str | pathlib.Path,
    *,
    tracking_uri: str,
    experiment_name: str,
) -> LiveRunMLflow:
    """Open one MLflow run before an isolated generic worker starts.

    Only immutable identities already written into the unsealed bundle are logged at this stage.
    Metrics and artifacts are accepted later only after bundle verification.
    """
    if not isinstance(tracking_uri, str) or not tracking_uri:
        raise RunMLflowError("tracking URI must be a non-blank string")
    root = pathlib.Path(bundle_path)
    resolved = _load(root / "metadata" / "resolved-run.json")
    source = _load(root / "provenance" / "source.json")
    data = _load(root / "provenance" / "data.json")
    runtime = _load(root / "provenance" / "runtime.json")
    plan = _load(root / "metadata" / "execution-plan.json")
    model_id = resolved.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RunMLflowError("resolved run has no model ID")
    mlflow = _mlflow()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.set_experiment(experiment_name)
        active = mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=model_id, log_system_metrics=True
        )
        mlflow.set_tags(
            {
                "autoforecast.export_schema": "generic-live-run-v1",
                "autoforecast.model_id": model_id,
                "autoforecast.adapter": str(resolved.get("adapter", "")),
                "autoforecast.source_commit": str(source.get("commit", "")),
                "autoforecast.data_fingerprint": str(data.get("fingerprint", "")),
                "autoforecast.runtime_lock_sha256": str(runtime.get("lockfile_sha256", "")),
                "autoforecast.lifecycle": "run_artifact_only",
                "autoforecast.model_registry": "not_registered_pending_promotion",
                "autoforecast.tracking_phase": "training",
                **_parameter_source_tags(root),
            }
        )
        parameters = _parameters(plan)
        if parameters:
            mlflow.log_params(parameters)
        return LiveRunMLflow(
            RunMLflowLink(tracking_uri, experiment.experiment_id, active.info.run_id),
            mlflow,
        )
    except Exception as error:
        try:
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        if isinstance(error, RunMLflowError):
            raise
        raise RunMLflowError(
            f"could not start live MLflow tracking: {type(error).__name__}"
        ) from error


def export_run_bundle(
    bundle_path: str | pathlib.Path,
    *,
    tracking_uri: str,
    experiment_name: str,
    comparison_path: str | pathlib.Path | None = None,
) -> RunMLflowLink:
    """Mirror one verified completed bundle; no external state can alter its evidence."""
    if not isinstance(tracking_uri, str) or not tracking_uri:
        raise RunMLflowError("tracking URI must be a non-blank string")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise RunMLflowError("MLflow experiment name must be a non-blank string")
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    with _completed_run_snapshot(bundle_path) as (root, completion):
        if completion.get("status") != "completed":
            raise RunMLflowError("only completed bundles may be exported")
        resolved = _load(root / "metadata" / "resolved-run.json")
        source = _load(root / "provenance" / "source.json")
        data = _load(root / "provenance" / "data.json")
        runtime = _load(root / "provenance" / "runtime.json")
        plan = _load(root / "metadata" / "execution-plan.json")
        model_id = resolved.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise RunMLflowError("resolved run has no model ID")
        comparison = None if comparison_path is None else _load(pathlib.Path(comparison_path))
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_id) as active:
            mlflow.set_tags(
                {
                    "autoforecast.export_schema": "generic-run-v1",
                    "autoforecast.run_id": str(completion.get("run_id", "")),
                    "autoforecast.model_id": model_id,
                    "autoforecast.adapter": str(resolved.get("adapter", "")),
                    "autoforecast.source_commit": str(source.get("commit", "")),
                    "autoforecast.data_fingerprint": str(data.get("fingerprint", "")),
                    "autoforecast.runtime_lock_sha256": str(runtime.get("lockfile_sha256", "")),
                    "autoforecast.lifecycle": "run_artifact_only",
                    "autoforecast.model_registry": "not_registered_pending_promotion",
                    **_parameter_source_tags(root),
                }
            )
            params = _parameters(plan)
            if params:
                mlflow.log_params(params)
            metrics = _comparison_metrics(comparison, model_id)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.log_dict(
                {
                    "schema": "autoforecast.generic-mlflow-evidence.v1",
                    "completion": completion,
                    "resolved_run": resolved,
                    "source": source,
                    "data": data,
                    "runtime": runtime,
                    "execution_plan": plan,
                },
                "autoforecast/evidence.json",
            )
            _log_safe_artifacts(mlflow, root)
            return RunMLflowLink(tracking_uri, experiment.experiment_id, active.info.run_id)


def export_forecast_workflow(
    bundle_path: str | pathlib.Path,
    *,
    tracking_uri: str,
    experiment_id: str,
) -> dict[str, RunMLflowLink]:
    """Export verified validation/test evidence to an existing experiment, retryably.

    The immutable workflow remains authoritative. A single local publisher holds
    a lock outside it; remote identity tags recover an upload interrupted after run
    creation. Retries reuse the same two run IDs and never fit or score. This is a
    completed-workflow exporter, not live tracking of partially fitted workflows.
    """
    import math

    from autoforecast.campaign import _open_lock
    from autoforecast.study_workflow import load_forecast_workflow

    if not all(isinstance(value, str) and value.strip() for value in (tracking_uri, experiment_id)):
        raise RunMLflowError("tracking_uri and experiment_id must be nonblank strings")
    # Authenticate and snapshot before constructing a remote client or writing there.
    with _completed_run_snapshot(bundle_path) as (root, completion):
        if completion.get("task") != "forecast-workflow" or completion.get("status") != "completed":
            raise RunMLflowError("only completed forecast workflows may be exported")
        workflow = load_forecast_workflow(root)
        if workflow.testing is None:
            raise RunMLflowError("validation checkpoints cannot be exported as completed workflows")
        manifest_sha = hashlib.sha256((root / "artifact_manifest.json").read_bytes()).hexdigest()
        identity = hashlib.sha256(
            json.dumps([manifest_sha, tracking_uri, experiment_id], separators=(",", ":")).encode()
        ).hexdigest()
        source = pathlib.Path(bundle_path).absolute()
        state_root = source.parent / ".workflow-exports"
        if state_root.is_symlink():
            raise RunMLflowError("workflow export directory cannot be a symlink")
        state_root.mkdir(exist_ok=True)
        import fcntl

        with _open_lock(state_root / (identity + ".lock")) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            mlflow = _mlflow()
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment(experiment_id)
            if experiment is None or experiment.lifecycle_stage != "active":
                raise RunMLflowError("supply an existing active MLflow experiment")
            links = {}
            for stage, evidence in (
                ("validation", workflow.validation),
                ("test", workflow.testing),
            ):
                key = identity + "-" + stage
                matches = client.search_runs(
                    [experiment_id],
                    filter_string=f"tags.`autoforecast.export_key` = '{key}'",
                    max_results=2,
                    run_view_type=mlflow.entities.ViewType.ALL,
                )
                if len(matches) > 1:
                    raise RunMLflowError("multiple remote runs claim the same workflow export")
                tags = {
                    "autoforecast.export_key": key,
                    "autoforecast.workflow_id": source.name,
                    "autoforecast.manifest_sha256": manifest_sha,
                    "autoforecast.stage": stage,
                    "autoforecast.study_sha256": workflow.provenance["study"]["sha256"],
                    "autoforecast.export_schema": "forecast-workflow-v1",
                }
                remote = (
                    matches[0]
                    if matches
                    else client.create_run(
                        experiment_id,
                        tags=tags,
                        run_name=f"{source.name}-{stage}",
                    )
                )
                if any(remote.data.tags.get(name) != value for name, value in tags.items()):
                    raise RunMLflowError("MLflow run identity differs from the verified workflow")
                run_id = remote.info.run_id
                if remote.info.lifecycle_stage != "active":
                    raise RunMLflowError("the recorded workflow export run has been deleted")
                links[stage] = RunMLflowLink(tracking_uri, experiment_id, run_id)
                if remote.data.tags.get("autoforecast.export_complete") == manifest_sha:
                    _verify_workflow_export(client, run_id, evidence)
                    if remote.info.status != "FINISHED":
                        client.set_terminated(run_id, status="FINISHED")
                    continue
                try:
                    client.log_dict(run_id, evidence["score"], "checked_score.json")
                    client.log_dict(run_id, workflow.parameters, "model_config.json")
                    client.log_dict(run_id, workflow.provenance, "workflow_provenance.json")
                    client.log_dict(
                        run_id, evidence["evaluation_contract"], "evaluation_contract.json"
                    )
                    client.log_artifacts(run_id, str(evidence["model_artifacts"]), "models")
                    client.log_artifacts(run_id, str(root / stage / "forecasts"), "forecasts")
                    client.log_artifacts(run_id, str(root / stage / "scores"), "scores")
                    score = evidence["score"]
                    rows = [("", score, 0)]
                    breakdown = score.get("forecast_accuracy_breakdown") or {}
                    for kind in ("per_step", "cumulative"):
                        rows.extend(
                            (kind + "_", point, point["end_lead"])
                            for point in breakdown.get(kind, [])
                        )
                    for prefix, point, step in rows:
                        values = {
                            name: point.get(name)
                            for name in ("mae", "rmse", "wape", "bias", "wrmsse")
                        }
                        values.update(
                            {
                                "coverage_" + name: value
                                for name, value in point.get("coverage", {}).items()
                            }
                        )
                        for name, value in values.items():
                            if type(value) in (int, float) and math.isfinite(value):
                                client.log_metric(
                                    run_id, prefix + name, float(value), timestamp=0, step=step
                                )
                    _verify_workflow_export(client, run_id, evidence)
                    client.set_tag(run_id, "autoforecast.export_complete", manifest_sha)
                    client.set_terminated(run_id, status="FINISHED")
                except Exception:
                    client.set_terminated(run_id, status="FAILED")
                    raise
            return links


def _verify_workflow_export(client: Any, run_id: str, evidence: dict) -> None:
    """Check remote model bytes and the full score, including unavailable values."""
    from provenance.lineage import directory_digest

    with tempfile.TemporaryDirectory(prefix="autoforecast-export-check-") as folder:
        models = client.download_artifacts(run_id, "models", folder)
        if directory_digest(models) != evidence["model_sha256"]:
            raise RunMLflowError("remote model artifacts differ from the verified workflow")
        score_path = client.download_artifacts(run_id, "checked_score.json", folder)
        from autoforecast._exact_json import load_exact_json

        try:
            remote_score = json.dumps(
                load_exact_json(pathlib.Path(score_path)), sort_keys=True, allow_nan=False
            )
        except (OSError, ValueError) as error:
            raise RunMLflowError("cannot verify remote scores") from error
        if remote_score != json.dumps(evidence["score"], sort_keys=True, allow_nan=False):
            raise RunMLflowError("remote scores differ from the verified workflow")
