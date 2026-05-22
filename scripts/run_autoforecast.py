"""Run selected AutoForecast candidates through one approval-gated, reproducible path.

The common ``mlops`` package owns provenance/bundles.  This script is deliberately forecast
specific: it resolves a panel contract, applies forecasting data/readiness checks, and invokes
registered forecast adapters in isolated workers.  Another repository reuses ``mlops`` but owns
its own task executor.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from typing import Any, cast

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoforecast import resources
from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast.adapters import isolated, validate_execution_controls
from autoforecast.benchmarks.policy import is_m5_daily
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.data_observability import CheckReport, inspect_panel, run_incident_from_error
from autoforecast.executable_source import materialize_executable_source_snapshot
from autoforecast.experiments import ExperimentError, load_completed_attempt
from autoforecast.model_api import ModelCheckInput, RunApproval
from autoforecast.model_catalog import Candidate, discover_candidates
from autoforecast.model_execution import ExecutionPlan, ExecutionPlanError, resolve_execution_plan
from autoforecast.model_readiness import (
    approve_run,
    create_resolved_run,
    prepare_model_check_input,
    run_model_checks,
)
from autoforecast.model_runtime import check_runtime
from autoforecast.notifications import notifier_from_environment
from autoforecast.panel import Forecast, Panel
from autoforecast.parameter_sources import (
    ParameterSourceError,
    load_parameter_resolution,
    parameter_resolution_sha256,
)
from autoforecast.precision_policy import TRAINABLE_TORCH_ADAPTERS, effective_precision
from autoforecast.project import (
    LIBRARY_ROOT,
    installed_library_identity,
    installed_mode,
    project_entrypoint,
    project_root,
)
from autoforecast.run_evaluation import EvaluationError, compare_run_forecasts
from autoforecast.run_mlflow import (
    LiveRunMLflow,
    RunMLflowError,
    export_run_bundle,
    start_live_run_bundle_tracking,
)
from autoforecast.train_workflow import (
    CheckedModelRun,
    RunRequest,
    RunSummary,
    TrainWorkflowError,
    load_run_request,
    run_training_workflow,
)
from autoforecast.training_lease import TrainingLeaseError, with_training_lease
from autoforecast.training_progress import PipelineProgress
from mlops import (
    CheckpointCapability,
    DataSnapshot,
    ProvenanceError,
    capture_runtime_identity,
    capture_source_identity,
    open_run_bundle,
    write_run_provenance,
)
from pyspark import SparkContext
from pyspark.sql import SparkSession

from provenance.lineage import LineageError, directory_content_digest, directory_digest

_ARCHIVED_M5_REPLAY_ROLE = "archived_private_replay_not_blind"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _root() -> pathlib.Path:
    return project_root(ROOT)


def _preflight_spark_profiles(
    selected: tuple[Candidate, ...],
) -> tuple[resources.SparkProfile, resources.SparkProfile]:
    """Split parent and worker resource budgets."""
    return resources.isolated_spark_profiles()


def _shutdown_owned_preflight_runtime(
    spark: SparkSession,
    *,
    session_type=SparkSession,
    context_type=SparkContext,
) -> None:
    """Release an owned preflight JVM before isolated Torch training begins.

    ``SparkContext.stop`` ends the context but leaves PySpark's process-global gateway JVM alive.
    On unified-memory macOS hosts that retained heap overlaps the isolated adapter's feature
    assembly and Torch tensors, even though no parent Spark work remains. Resetting the gateway
    is safe here: this runner owns the preflight runtime exclusively, and any later scoring
    session deliberately creates a fresh JVM.
    """
    try:
        spark.stop()
    except Exception:
        # A Spark JVM failure can make stop itself raise a Py4J transport error. Cleanup must
        # still sever the stale process-global handles before the isolated worker starts.
        pass
    finally:
        context_type._active_spark_context = None
        session_type._activeSession = None
        session_type._instantiatedSession = None
        gateway = getattr(context_type, "_gateway", None)
        try:
            if gateway is not None:
                gateway.shutdown()
        except Exception:
            pass
        finally:
            context_type._gateway = None
            context_type._jvm = None


def _require_expected_candidate(
    selected: tuple[Candidate, ...],
    expected: Candidate,
) -> None:
    """Bind an in-process HPO handoff to the catalog bytes its caller authenticated."""
    if len(selected) != 1:
        raise TrainWorkflowError("a frozen HPO candidate requires exactly one selected model")
    actual = selected[0]
    if (
        actual.id != expected.id
        or actual.adapter != expected.adapter
        or actual.runtime != expected.runtime
        or actual.source_path != expected.source_path
        or actual.source_sha256 != expected.source_sha256
        or actual.implementation_config_path != expected.implementation_config_path
        or actual.implementation_config_sha256 != expected.implementation_config_sha256
    ):
        raise TrainWorkflowError(
            "catalog candidate changed after the HPO caller authenticated its configuration"
        )


def _freeze_parameter_resolution(
    directory: pathlib.Path,
    record: dict[str, object],
    expected_sha256: str,
) -> pathlib.Path:
    """Materialize validated parameter bytes in private scratch for later provenance copying."""
    if parameter_resolution_sha256(record) != expected_sha256:
        raise TrainWorkflowError("validated parameter resolution differs from its supplied SHA-256")
    path = directory / "parameter-resolution.json"
    payload = (json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    try:
        path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise TrainWorkflowError("private parameter-resolution snapshot failed verification")
        path.chmod(0o400)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _copy_parameter_source_capsule(
    bundle: Any,
    record: dict[str, object],
    expected_sha256: str,
) -> None:
    """Retain a content-addressed HPO evidence tree inside a governed run bundle."""
    source = record.get("source")
    if type(source) is not dict or source.get("kind") not in {"tune", "reuse_selection"}:
        return
    selection = pathlib.Path(cast(str, source["selection_path"]))
    attempt = pathlib.Path(cast(str, source["selected_attempt_path"]))
    study_config = _root() / cast(str, source["study_config_path"])
    study_root = selection.parent
    for path, label in (
        (selection, "HPO selection"),
        (attempt, "HPO selected attempt"),
        (study_config, "HPO study config"),
    ):
        if path.is_symlink() or not path.exists():
            raise TrainWorkflowError(f"{label} is missing or linked while creating its capsule")
    try:
        attempt_relative = attempt.relative_to(study_root).as_posix()
    except ValueError as error:
        raise TrainWorkflowError(
            "selected HPO attempt is outside its study evidence root"
        ) from error
    try:
        _metadata, completion = load_completed_attempt(attempt)
    except ExperimentError as error:
        raise TrainWorkflowError(f"selected HPO attempt is not complete: {error}") from error
    selection_sha = hashlib.sha256(selection.read_bytes()).hexdigest()
    config_sha = hashlib.sha256(study_config.read_bytes()).hexdigest()
    if (
        selection_sha != source.get("selection_sha256")
        or config_sha != source.get("study_config_sha256")
        or completion.get("artifact_manifest_sha256")
        != source.get("selected_attempt_manifest_sha256")
    ):
        raise TrainWorkflowError("HPO evidence changed after parameter resolution")
    manifest_path = attempt / "artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = manifest["artifacts"]
        declared_paths = [cast(str, item["path"]) for item in declared]
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise TrainWorkflowError("selected HPO attempt manifest is invalid") from error
    source_files = (
        attempt / "completion.json",
        manifest_path,
        *(attempt / pathlib.PurePosixPath(relative) for relative in declared_paths),
    )
    if len(set(source_files)) != len(source_files):
        raise TrainWorkflowError("selected HPO attempt manifest contains duplicate paths")
    artifacts: list[dict[str, object]] = []

    def copy_one(source_path: pathlib.Path, relative: str) -> None:
        before = hashlib.sha256(source_path.read_bytes()).hexdigest()
        size = source_path.stat().st_size
        copied = bundle.copy_file(relative, source_path)
        after = hashlib.sha256(source_path.read_bytes()).hexdigest()
        copied_sha = hashlib.sha256(copied.read_bytes()).hexdigest()
        if before != after or before != copied_sha or copied.stat().st_size != size:
            raise TrainWorkflowError(f"HPO evidence changed while copying {source_path}")
        artifacts.append({"path": relative, "sha256": copied_sha, "bytes": size})

    config_relative = "provenance/parameter-source/study-config.yaml"
    copy_one(study_config, config_relative)
    selection_relative = "provenance/parameter-source/study/selection.json"
    copy_one(selection, selection_relative)
    attempt_capsule = f"provenance/parameter-source/study/{attempt_relative}"
    for source_path in source_files:
        relative = source_path.relative_to(attempt).as_posix()
        copy_one(source_path, f"{attempt_capsule}/{relative}")
    if (
        hashlib.sha256(selection.read_bytes()).hexdigest() != selection_sha
        or hashlib.sha256(study_config.read_bytes()).hexdigest() != config_sha
    ):
        raise TrainWorkflowError("HPO source evidence changed while creating its capsule")
    bundle.write_json(
        "provenance/parameter-source/capsule.json",
        {
            "schema": "autoforecast.parameter-source-capsule.v2",
            "parameter_resolution_sha256": expected_sha256,
            "study_id": source["study_id"],
            "study_config_sha256": config_sha,
            "study_config": config_relative,
            "selection": selection_relative,
            "selection_sha256": selection_sha,
            "selected_trial_id": source["selected_trial_id"],
            "selected_attempt_id": source["selected_attempt_id"],
            "selected_attempt_manifest_sha256": source["selected_attempt_manifest_sha256"],
            "selected_attempt": attempt_capsule,
            "artifacts": artifacts,
        },
    )


def _inside_repository(path: pathlib.Path, label: str) -> pathlib.Path:
    if path.is_symlink():
        raise TrainWorkflowError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve()
    repository = _root().resolve()
    external_request = installed_mode() and label == "run request"
    if not resolved.is_relative_to(repository) and not external_request:
        raise TrainWorkflowError(f"{label} must be source-controlled below {repository}")
    if not resolved.is_file():
        raise TrainWorkflowError(f"{label} must be a regular file: {resolved}")
    return resolved


def _runtime_data_path(path: pathlib.Path) -> pathlib.Path:
    """Freeze a caller's local input location before an isolated worker changes directory."""
    if path.is_symlink():
        raise TrainWorkflowError("runtime data path must be a regular non-symlink path")
    return path.resolve()


def _sealed_m5_selection_lock(
    request: RunRequest,
    selected: tuple[Any, ...],
    *,
    request_path: pathlib.Path,
    epochs: int,
    seed: int,
    training_controls: dict[str, object],
) -> tuple[pathlib.Path, str, dict[str, object]] | None:
    """Validate the source-controlled final-selection decision before archived replay.

    The private M5 archive is already observable in this workspace.  A final replay therefore
    needs an immutable decision made from the public holdout before this function permits its
    worker to see d_1941 training data or its scorer to open organizer labels.
    """
    settings = request.evaluation
    requested_contract = _inside_repository(
        _root() / request.dataset_contract, "requested dataset contract"
    )
    is_archived_m5_final = (
        is_m5_daily(requested_contract)
        and request.origin == "2016-05-22"
        and request.horizon == 28
    )
    if settings is not None and settings.evidence_role == _ARCHIVED_M5_REPLAY_ROLE:
        if not is_archived_m5_final:
            raise TrainWorkflowError(
                "archived M5 replay evidence requires the canonical M5 contract content and "
                "the d_1941 final window"
            )
    if not is_archived_m5_final:
        return None
    # A training-only final refit cannot observe the held-out labels: the panel presented to the
    # adapter is physically truncated at ``origin``.  Campaign orchestration supplies and verifies
    # its own dynamic selection lock before invoking this path.  Keep the checked-in lock mandatory
    # for the standalone operation that actually opens the archived organizer labels.
    if settings is None:
        return None
    if (
        settings.input_mode != "post_training_score_only"
        or settings.evidence_role != _ARCHIVED_M5_REPLAY_ROLE
    ):
        raise TrainWorkflowError(
            "the archived M5 final window requires post-training scoring with "
            f"evidence_role={_ARCHIVED_M5_REPLAY_ROLE!r} and a sealed public selection"
        )
    if settings.selection_lock is None:
        raise TrainWorkflowError("archived M5 replay requires run.evaluation.selection_lock")
    lock_path = _inside_repository(_root() / settings.selection_lock, "M5 selection lock")
    raw = lock_path.read_bytes()
    try:
        value = loads_exact_yaml(raw)
    except ExactYAMLError as exc:
        raise TrainWorkflowError(f"invalid YAML in M5 selection lock: {exc}") from exc
    if type(value) is not dict or set(value) != {"schema_version", "selection_lock"}:
        raise TrainWorkflowError(
            "M5 selection lock requires exactly schema_version and selection_lock"
        )
    lock = value["selection_lock"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(lock) is not dict
    ):
        raise TrainWorkflowError("M5 selection lock must use schema_version 1")
    if lock.get("status") != "locked":
        raise TrainWorkflowError(
            "M5 selection lock is pending: complete and commit the public-holdout selection first"
        )
    expected = {"id", "status", "protocol", "public_holdout", "final_refit"}
    if set(lock) != expected or lock["protocol"] != "m5_accuracy_archived_replay_v1":
        raise TrainWorkflowError("sealed M5 selection lock has an unsupported schema or protocol")
    public = lock["public_holdout"]
    final = lock["final_refit"]
    if type(public) is not dict or set(public) != {
        "request",
        "origin",
        "horizon",
        "candidate_comparison",
        "candidate_comparison_sha256",
    }:
        raise TrainWorkflowError("sealed M5 selection lock has an invalid public_holdout record")
    final_keys = {
        "request",
        "origin",
        "horizon",
        "model_id",
        "model_profile_sha256",
        "run_request_sha256",
        "seed",
        "maximum_epochs",
        "patience",
        "device",
        "batch_size",
        "gradient_microbatch_size",
        "early_stopping_absolute_min_delta",
        "early_stopping_relative_min_delta",
    }
    if type(final) is not dict or set(final) != final_keys:
        raise TrainWorkflowError("sealed M5 selection lock has an invalid final_refit record")
    if (
        public["request"] != "configs/autoforecast/m5_public_holdout_deep_v2.yaml"
        or public["origin"] != "2016-04-24"
        or public["horizon"] != 28
        or type(public["candidate_comparison"]) is not str
        or not isinstance(public["candidate_comparison_sha256"], str)
        or _SHA256.fullmatch(public["candidate_comparison_sha256"]) is None
    ):
        raise TrainWorkflowError("sealed M5 selection lock must bind the d_1913 public holdout")
    comparison_path = _inside_repository(
        _root() / cast(str, public["candidate_comparison"]),
        "M5 public candidate comparison",
    )
    if (
        hashlib.sha256(comparison_path.read_bytes()).hexdigest()
        != public["candidate_comparison_sha256"]
    ):
        raise TrainWorkflowError("sealed M5 public candidate comparison bytes changed")
    if len(selected) != 1:
        raise TrainWorkflowError("sealed M5 final replay requires exactly one selected model")
    candidate = selected[0]
    expected_final = {
        "request": request_path.relative_to(_root()).as_posix(),
        "origin": request.origin,
        "horizon": request.horizon,
        "model_id": candidate.id,
        "model_profile_sha256": candidate.source_sha256,
        "run_request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "seed": seed,
        "maximum_epochs": epochs,
        **training_controls,
    }
    if final != expected_final:
        raise TrainWorkflowError(
            "sealed M5 selection lock does not match this final request, model profile, or "
            "training controls"
        )
    return lock_path, hashlib.sha256(raw).hexdigest(), cast(dict[str, object], lock)


def _verified_parent_checkpoint(
    checkpoint: pathlib.Path | None,
    expected_sha256: str | None,
) -> tuple[pathlib.Path, str] | None:
    """Accept one explicit local warm-start input without treating it as exact resume state."""
    if (checkpoint is None) != (expected_sha256 is None):
        raise TrainWorkflowError(
            "--parent-checkpoint and --parent-checkpoint-sha256 must be supplied together"
        )
    if checkpoint is None:
        return None
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise TrainWorkflowError(
            "parent checkpoint SHA-256 must be 64 lowercase hexadecimal characters"
        )
    if checkpoint.is_symlink():
        raise TrainWorkflowError("parent checkpoint must be a regular local file")
    path = checkpoint.resolve()
    if not path.is_file():
        raise TrainWorkflowError("parent checkpoint must be a regular local file")
    import hashlib

    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise TrainWorkflowError(
            f"parent checkpoint digest differs: expected={expected_sha256}, actual={actual}"
        )
    return path, actual


def _worker_kwargs(
    context: CheckedModelRun,
    plan: ExecutionPlan,
    *,
    parent_checkpoint: tuple[pathlib.Path, str] | None,
    training_controls: dict[str, object],
    calendar_events: tuple[tuple[str, str, str | None], ...],
    event_max_support: int,
    training_mode: str,
    validation_data_path: pathlib.Path | None,
    fixed_update_budget: int | None,
    validation_metric: str | None,
) -> dict[str, object]:
    """Add common output paths to adapter-owned typed execution inputs."""
    out = context.attempt_dir / "evaluation" / "forecast"
    serialized = plan.as_record()["adapter_kwargs"]
    if type(serialized) is not dict:  # pragma: no cover - ExecutionPlan owns this invariant
        raise TrainWorkflowError("execution plan adapter inputs are not a mapping")
    kwargs: dict[str, object] = {**serialized, "out_dir": str(out)}
    if context.input.run.adapter in {
        "mft",
        "sft",
        "hmfnet",
        "pact",
    }:
        from autoforecast.torch_profile_contract import profile_request_from_environment

        try:
            performance_profile = profile_request_from_environment(os.environ)
        except ValueError as error:
            raise TrainWorkflowError(str(error)) from error
        kwargs.update(
            {
                "checkpoint_out": str(context.attempt_dir / "model" / "model.pt"),
                "training_metrics_path": str(context.attempt_dir / "metrics" / "epochs.jsonl"),
                "training_log_dir": str(context.attempt_dir / "tensorboard"),
                "events": calendar_events,
                "event_max_support": event_max_support,
                **training_controls,
            }
        )
        if performance_profile is not None:
            if context.input.run.adapter not in {
                "sft",
                "hmfnet",
                "pact",
            }:
                raise TrainWorkflowError(
                    "Torch profiling is currently supported by registered rolling revisions"
                )
            kwargs["performance_profile"] = performance_profile
        if parent_checkpoint is not None:
            if context.input.run.adapter != "mft":
                raise TrainWorkflowError(
                    "registered revisions do not expose weights-only continuation"
                )
            seed = kwargs.get("seed")
            if type(seed) is not int:
                raise TrainWorkflowError("custom execution plan must provide an integer seed")
            kwargs.update(
                {
                    "warm_start_checkpoint": str(parent_checkpoint[0]),
                    "warm_start_checkpoint_sha256": parent_checkpoint[1],
                    "validation_seed": seed + 1,
                }
            )
        if context.input.run.adapter in {
            "sft",
            "hmfnet",
            "pact",
        }:
            kwargs.update(
                {
                    "training_mode": training_mode,
                    "validation_data_path": (
                        None if validation_data_path is None else str(validation_data_path)
                    ),
                    "fixed_update_budget": fixed_update_budget,
                    "validation_metric": validation_metric,
                }
            )
    elif context.input.run.adapter == "shallow_global":
        kwargs["checkpoint_dir"] = str(context.attempt_dir / "model")
    elif context.input.run.adapter in {"tft", "tft_q", "tft_hnb", }:
        kwargs.update(
            {
                "checkpoint_out": str(
                    context.attempt_dir
                    / "model"
                    / ("model.pt" if context.input.run.adapter != "tft" else "tft-state.pt")
                ),
                "batch_size": training_controls["batch_size"],
            }
        )
        if context.input.run.adapter == "tft":
            kwargs["device"] = training_controls["device"]
        if training_controls.get("precision") == "16-mixed":
            kwargs["precision"] = "16-mixed"
        if context.input.run.adapter in {"tft_q", "tft_hnb", }:
            kwargs.update(
                {
                    "device": training_controls["device"] or (
                        "auto" if training_controls.get("precision") == "16-mixed" else "cpu"
                    ),
                    "early_stopping_patience": training_controls["patience"],
                    "early_stopping_min_delta": training_controls[
                        "early_stopping_absolute_min_delta"
                    ],
                    "training_mode": training_mode,
                    "fixed_update_budget": fixed_update_budget,
                }
            )
    return kwargs


def _checkpoint_capability(bundle_path: pathlib.Path) -> CheckpointCapability | None:
    checkpoint = bundle_path / "model" / "model.pt"
    if not checkpoint.is_file():
        return None
    import hashlib

    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return CheckpointCapability(
        kind="inference_only",
        checkpoint_path="model/model.pt",
        checkpoint_sha256=digest,
        missing_resume_state=(
            "optimizer_state",
            "scheduler_state",
            "amp_scaler_state",
            "rng_state",
            "sampler_state",
            "completed_step",
        ),
    )


def _model_artifact_manifest(bundle_path: pathlib.Path) -> dict[str, object] | None:
    """Hash every saved model artifact without pretending every format can resume training."""
    root = bundle_path / "model"
    if not root.is_dir():
        return None
    import hashlib

    files: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TrainWorkflowError(f"model artifact may not be a symlink: {path}")
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path": path.relative_to(bundle_path).as_posix(), "sha256": digest})
    if not files:
        return None
    return {"schema": "autoforecast.model-artifacts.v1", "files": files}


def _append_run_link(report: pathlib.Path, relative_target: str, label: str) -> None:
    """Keep the coordinator's plain static report as the single entry point for run evidence."""
    text = report.read_text(encoding="utf-8")
    href = relative_target.replace("'", "%27")
    report.write_text(text + f"<p><a href='{href}'>{label}</a></p>", encoding="utf-8")


def _evaluate_completed_forecasts(
    *,
    request: RunRequest,
    training_panel: Panel,
    forecasts: dict[str, Forecast],
    run_path: pathlib.Path,
    score_panel: Panel | None = None,
    score_data_snapshot_id: str | None = None,
    score_data_reference: str | None = None,
    fixed_contract: dict | None = None,
) -> bool:
    """Publish declared post-run evidence without changing completed training outcomes."""
    # These attributes are intentionally checked here rather than broadening the generic
    # coordinator: evaluation is forecast-specific while the coordinator remains task-neutral.
    settings = request.evaluation
    if settings is None or not forecasts:
        return True
    post_training = settings.input_mode == "post_training_score_only"
    if post_training and (
        score_panel is None or score_data_snapshot_id is None or score_data_reference is None
    ):
        raise TrainWorkflowError(
            "post-training evaluation requires a separate score panel, snapshot id, and reference"
        )
    panel = training_panel if score_panel is None else score_panel
    score_manifest = run_path / "evaluation" / "post-training-evaluation.json"
    if post_training and score_manifest.exists():
        raise TrainWorkflowError("post-training evaluation record already exists for this run")
    score_snapshot: DataSnapshot | None = None
    if post_training:
        try:
            score_snapshot = DataSnapshot(
                cast(str, score_data_snapshot_id),
                directory_digest(panel.path, suffixes=(".parquet",)),
                cast(str, score_data_reference),
            )
        except (LineageError, ProvenanceError) as error:
            raise TrainWorkflowError(
                f"cannot identify post-training score input: {error}"
            ) from error
    profile = resources.forecast_scoring_profile(
        settings.metric, tuple(forecast.adapter for forecast in forecasts.values())
    )
    score_progress = PipelineProgress("Forecast scoring", 2, phase_key="score")
    spark = None
    with score_progress.step("Start the bounded Spark scoring runtime"):
        spark = resources.acquire_local_runtime(
            SparkSession, SparkContext, profile=profile, app_name="autoforecast_run_evaluation"
        )
        with contextlib.suppress(Exception):
            spark.sparkContext.setLogLevel("ERROR")
    try:
        try:
            with score_progress.step("Calculate and publish the declared metric", spark=spark):
                score_progress.operation(
                    "Join forecasts to observed labels and calculate the declared metric"
                )
                artifact = compare_run_forecasts(
                    panel,
                    forecasts,
                    spark,
                    missing_actual=settings.missing_actual,
                    metric=settings.metric,
                    output_dir=run_path / "evaluation",
                    **({"fixed_contract": fixed_contract} if fixed_contract else {}),
                )
        except (EvaluationError, ValueError) as error:
            target = run_path / "evaluation"
            target.mkdir(parents=True, exist_ok=True)
            (target / "candidate-comparison-error.json").write_text(
                json.dumps(
                    {
                        "schema": "autoforecast.run-evaluation-error.v1",
                        "error": str(error),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if post_training:
                score_manifest.write_text(
                    json.dumps(
                        {
                            "schema": "autoforecast.post-training-evaluation.v1",
                            "status": "failed",
                            "evidence_role": settings.evidence_role,
                            "score_input": score_snapshot.as_record() if score_snapshot else None,
                            "error": str(error),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return False
            return True
        else:
            if post_training and score_snapshot is not None:
                observed_score_digest = directory_digest(panel.path, suffixes=(".parquet",))
                if observed_score_digest != score_snapshot.fingerprint:
                    raise TrainWorkflowError(
                        "post-training score data changed while evaluation was running"
                    )
            if post_training:
                comparison_record = json.loads(artifact.read_text(encoding="utf-8"))
                unscored = [
                    str(candidate.get("model_id", "unknown"))
                    for candidate in comparison_record.get("candidates", [])
                    if not isinstance(candidate, dict) or candidate.get("status") != "scored"
                ]
                if unscored or comparison_record.get("ranking") is None:
                    detail = (
                        "post-training score did not produce a complete comparable result"
                        f"; unscored candidates={unscored}"
                    )
                    score_manifest.write_text(
                        json.dumps(
                            {
                                "schema": "autoforecast.post-training-evaluation.v1",
                                "status": "failed",
                                "evidence_role": settings.evidence_role,
                                "score_input": (
                                    score_snapshot.as_record() if score_snapshot else None
                                ),
                                "comparison": artifact.relative_to(run_path).as_posix(),
                                "error": detail,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    return False
                score_manifest.write_text(
                    json.dumps(
                        {
                            "schema": "autoforecast.post-training-evaluation.v1",
                            "status": "scored",
                            "evidence_role": settings.evidence_role,
                            "score_input": score_snapshot.as_record() if score_snapshot else None,
                            "forecast_artifacts": {
                                model_id: directory_digest(forecast.path, suffixes=(".parquet",))
                                for model_id, forecast in sorted(forecasts.items())
                            },
                            "comparison": artifact.relative_to(run_path).as_posix(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            _append_run_link(
                run_path / "report.html",
                artifact.with_suffix(".html").relative_to(run_path).as_posix(),
                "Open candidate comparison",
            )
            if post_training:
                _append_run_link(
                    run_path / "report.html",
                    score_manifest.relative_to(run_path).as_posix(),
                    "Open post-training evaluation provenance",
                )
            return True
    finally:
        with contextlib.suppress(Exception):
            if spark is not None:
                spark.stop()


def _export_mlflow_if_requested(
    request: RunRequest,
    summary: RunSummary,
    *,
    skip_model_ids: set[str] | None = None,
) -> None:
    """Mirror sealed artifacts to MLflow after the run; never let a sink change its outcome."""
    if not request.mlflow_enabled:
        return
    run_path = summary.path
    tracking = run_path / "tracking"
    tracking.mkdir(exist_ok=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment = os.environ.get("AUTOFORECAST_MLFLOW_EXPERIMENT", "AutoForecast governed runs")
    comparison = run_path / "evaluation" / "candidate-comparison.json"
    entries: list[dict[str, object]] = []
    if not tracking_uri:
        entries.append(
            {
                "status": "not_exported",
                "detail": "MLflow was requested but MLFLOW_TRACKING_URI is not set.",
            }
        )
    else:
        for outcome in summary.outcomes:
            if outcome.status != "completed":
                continue
            if outcome.model_id in (skip_model_ids or set()):
                continue
            bundles = sorted((run_path / "attempts" / outcome.model_id).glob("*/completion.json"))
            if len(bundles) != 1:
                entries.append(
                    {
                        "model_id": outcome.model_id,
                        "status": "not_exported",
                        "detail": "Expected exactly one completed sealed model bundle.",
                    }
                )
                continue
            try:
                link = export_run_bundle(
                    bundles[0].parent,
                    tracking_uri=tracking_uri,
                    experiment_name=experiment,
                    comparison_path=comparison if comparison.is_file() else None,
                )
            except RunMLflowError as error:
                entries.append(
                    {
                        "model_id": outcome.model_id,
                        "status": "not_exported",
                        "detail": str(error),
                    }
                )
            else:
                entries.append(
                    {
                        "model_id": outcome.model_id,
                        "status": "exported",
                        **link.as_record(),
                    }
                )
    (tracking / "mlflow.json").write_text(
        json.dumps(
            {
                "schema": "autoforecast.run-mlflow-export.v1",
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for entry in entries:
        uri = entry.get("tracking_web_url")
        experiment_id = entry.get("experiment_id")
        run_id = entry.get("run_id")
        if (
            isinstance(uri, str)
            and uri.startswith(("http://", "https://"))
            and isinstance(experiment_id, str)
            and isinstance(run_id, str)
        ):
            target = f"{uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
            link = (
                "<a target='_blank' rel='noopener noreferrer' href="
                f"'{html.escape(target, quote=True)}'>Open MLflow run</a>"
            )
        else:
            link = "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry.get('model_id', 'run')))}</td>"
            f"<td>{html.escape(str(entry['status']))}</td><td>{link}</td>"
            f"<td>{html.escape(str(entry.get('detail', '')))}</td></tr>"
        )
    (tracking / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>MLflow export</title>"
        "<style>body{font:14px system-ui;margin:32px}table{border-collapse:collapse}"
        "th,td{border:1px solid #d7dce5;padding:8px;text-align:left}</style>"
        "<h1>MLflow export</h1><p>MLflow is a mirror of sealed evidence; it cannot approve or "
        "promote a model.</p><table><tr><th>Model</th><th>Status</th><th>Run</th><th>Detail</th>"
        "</tr>" + "".join(rows) + "</table>",
        encoding="utf-8",
    )
    _append_run_link(run_path / "report.html", "tracking/report.html", "Open MLflow export status")


def _complete_live_mlflow(
    summary: RunSummary,
    live_runs: dict[str, LiveRunMLflow],
) -> None:
    """Backfill comparison metrics after every sequential live run has been closed."""
    if not live_runs:
        return
    comparison = summary.path / "evaluation" / "candidate-comparison.json"
    entries: list[dict[str, object]] = []
    for model_id, live in live_runs.items():
        if not comparison.is_file():
            entries.append(
                {
                    "model_id": model_id,
                    "status": "not_exported",
                    "detail": "Completed run comparison is unavailable.",
                }
            )
            continue
        try:
            live.project_comparison(comparison, model_id=model_id)
        except RunMLflowError as error:
            entries.append({"model_id": model_id, "status": "not_exported", "detail": str(error)})
        else:
            entries.append({"model_id": model_id, "status": "exported", **live.link.as_record()})
    tracking = summary.path / "tracking"
    tracking.mkdir(exist_ok=True)
    (tracking / "live-mlflow.json").write_text(
        json.dumps({"schema": "autoforecast.generic-live-mlflow.v1", "entries": entries}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_observability_landing(summary: RunSummary) -> None:
    """Publish one lightweight, file-based entry point for one governed run group."""
    root = summary.path
    tensorboard_url = os.environ.get("AUTOFORECAST_TENSORBOARD_URL")
    rows = []
    for outcome in summary.outcomes:
        completions = sorted((root / "attempts" / outcome.model_id).glob("*/completion.json"))
        if len(completions) != 1:
            rows.append(
                "<tr>"
                f"<td>{html.escape(outcome.model_id)}</td><td>{html.escape(outcome.status)}</td>"
                "<td>—</td><td>—</td><td>—</td><td>—</td></tr>"
            )
            continue
        bundle = completions[0].parent
        relative = bundle.relative_to(root).as_posix()

        def link(relative_path: str, target: str, label: str) -> str:
            return f"<a href='{html.escape(relative_path + '/' + target, quote=True)}'>{label}</a>"

        events = list((bundle / "tensorboard").glob("events.out.tfevents.*"))
        tensorboard = (
            f"<a target='_blank' rel='noopener noreferrer' href="
            f"'{html.escape(tensorboard_url, quote=True)}'>Open TensorBoard</a>"
            if events and tensorboard_url
            else (
                "Event files recorded; set AUTOFORECAST_TENSORBOARD_URL to link a running server."
                if events
                else "Not applicable"
            )
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(outcome.model_id)}</td><td>{html.escape(outcome.status)}</td>"
            f"<td>{link(relative, 'checks/data.json', 'Data checks')}</td>"
            f"<td>{link(relative, 'checks/model.json', 'Model checks')}</td>"
            f"<td>{link(relative, 'metadata/execution-plan.json', 'Run configuration')}</td>"
            f"<td>{tensorboard}</td></tr>"
        )
    auxiliary = []
    for target, label in (
        ("evaluation/candidate-comparison.html", "Candidate comparison"),
        ("tracking/report.html", "MLflow export status"),
    ):
        if (root / target).is_file():
            auxiliary.append(f"<li><a href='{target}'>{label}</a></li>")
    (root / "observability.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>AutoForecast run observability</title>"
        "<style>body{font:14px system-ui;margin:32px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d7dce5;padding:8px;text-align:left;vertical-align:top}"
        "th{background:#f4f7fb}</style>"
        f"<h1>Run observability: {html.escape(summary.run_id)}</h1>"
        "<p>Run artifacts are authoritative. MLflow and TensorBoard are linked views, not gates. "
        "A TensorBoard link is shown only when a model emitted event files.</p>"
        "<table><tr><th>Model</th><th>Status</th><th>Data</th><th>Readiness</th>"
        "<th>Configuration</th><th>Training graph</th></tr>" + "".join(rows) + "</table>"
        "<h2>Run reports</h2><ul>" + "".join(auxiliary) + "</ul>",
        encoding="utf-8",
    )
    _append_run_link(root / "report.html", "observability.html", "Open run observability")


def _durable_worker_scratch_parent(output_root: pathlib.Path) -> pathlib.Path:
    """Create the private runtime-scratch parent beside governed run artifacts.

    macOS may reclaim ``/tmp`` entries during a multi-hour training run.  The
    authenticated executable-source snapshot must survive every checkpoint, so
    its short-lived parent belongs below the caller's durable output root, not
    the operating system's temporary directory.
    """
    if output_root.is_symlink():
        raise TrainWorkflowError("run output root must not be a symbolic link")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent = root / ".autoforecast-worker-scratch"
    if parent.is_symlink():
        raise TrainWorkflowError("worker scratch parent must not be a symbolic link")
    parent.mkdir(exist_ok=True)
    return parent.resolve()


def _resolved_precision(
    request_schema_version: int, explicit: str | None, *, has_parent: bool
) -> str:
    """Keep legacy and continuation runs FP32; new-schema fresh runs default to FP16 AMP."""
    if explicit not in (None, "32-true", "16-mixed"):
        raise TrainWorkflowError("precision must be 32-true, 16-mixed, or null")
    if has_parent and explicit == "16-mixed":
        raise TrainWorkflowError("mixed precision is unsupported for checkpoint continuations")
    return explicit or (
        "16-mixed" if request_schema_version >= 3 and not has_parent else "32-true"
    )


@with_training_lease
def run_request(
    request_path: pathlib.Path,
    *,
    data_path: pathlib.Path,
    data_snapshot_id: str,
    data_reference: str,
    output_root: pathlib.Path,
    epochs: int,
    timeout_seconds: int,
    seed: int = 0,
    patience: int | None = None,
    device: str | None = None,
    precision: str | None = None,
    batch_size: int = 4096,
    gradient_microbatch_size: int | None = None,
    early_stopping_absolute_min_delta: float = 1e-6,
    early_stopping_relative_min_delta: float = 0.0,
    parent_checkpoint: pathlib.Path | None = None,
    parent_checkpoint_sha256: str | None = None,
    parameter_overrides: dict[str, object] | None = None,
    expected_candidate: Candidate | None = None,
    parameter_resolution_path: pathlib.Path | None = None,
    parameter_resolution_sha256: str | None = None,
    mlflow_enabled: bool | None = None,
    evaluation_data_path: pathlib.Path | None = None,
    evaluation_data_snapshot_id: str | None = None,
    evaluation_data_reference: str | None = None,
    notifications_enabled: bool = True,
    training_mode: str = "rolling_backtest",
    fixed_update_budget: int | None = None,
    evaluation_contract_path: pathlib.Path | None = None,
    evaluation_contract_sha256: str | None = None,
) -> int:
    """Run one checked request.  It is public for functional tests and thin CLIs."""
    fixed_contract = None
    if (evaluation_contract_path is None) != (evaluation_contract_sha256 is None):
        raise TrainWorkflowError("evaluation contract path and SHA-256 must be supplied together")
    if evaluation_contract_path is not None:
        raw_contract = evaluation_contract_path.read_bytes()
        if hashlib.sha256(raw_contract).hexdigest() != evaluation_contract_sha256:
            raise TrainWorkflowError("evaluation contract changed before training")
        from autoforecast._exact_json import loads_exact_json
        parsed_contract = loads_exact_json(raw_contract)
        if type(parsed_contract) is not dict:
            raise TrainWorkflowError("evaluation contract must be an object")
        fixed_contract = parsed_contract
    resources._require_fresh_spark_runtime(SparkSession, SparkContext)
    if type(seed) is not int or seed < 0:
        raise TrainWorkflowError("seed must be a non-negative integer")
    if patience is not None and (type(patience) is not int or patience < 1):
        raise TrainWorkflowError("patience must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise TrainWorkflowError("batch_size must be a positive integer")
    if training_mode not in {"rolling_backtest", "fixed_validation", "fixed_refit"}:
        raise TrainWorkflowError("training_mode is unsupported")
    if training_mode == "fixed_refit":
        if type(fixed_update_budget) is not int or fixed_update_budget < 1:
            raise TrainWorkflowError("fixed_refit requires a positive fixed update budget")
    elif fixed_update_budget is not None:
        raise TrainWorkflowError("only fixed_refit accepts a fixed update budget")
    if gradient_microbatch_size is not None and (
        type(gradient_microbatch_size) is not int or gradient_microbatch_size < 1
    ):
        raise TrainWorkflowError("gradient_microbatch_size must be a positive integer or null")
    if device not in (None, "cpu", "cuda", "mps"):
        raise TrainWorkflowError("device must be one of cpu, cuda, mps, or null")
    if precision not in (None, "32-true", "16-mixed"):
        raise TrainWorkflowError("precision must be 32-true, 16-mixed, or null")
    if (parameter_resolution_path is None) != (parameter_resolution_sha256 is None):
        raise TrainWorkflowError("parameter resolution path and SHA-256 must be supplied together")
    if parameter_overrides is not None and parameter_resolution_path is not None:
        raise TrainWorkflowError(
            "raw HPO overrides and a governed parameter resolution are mutually exclusive"
        )
    if (parameter_overrides is None) != (expected_candidate is None):
        raise TrainWorkflowError(
            "raw HPO overrides and their frozen expected candidate must be supplied together"
        )
    for name, value in (
        ("early_stopping_absolute_min_delta", early_stopping_absolute_min_delta),
        ("early_stopping_relative_min_delta", early_stopping_relative_min_delta),
    ):
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise TrainWorkflowError(f"{name} must be a finite non-negative number")
    data_path = _runtime_data_path(data_path)
    training_controls: dict[str, object] = {
        "patience": patience,
        "device": device,
        "batch_size": batch_size,
        "gradient_microbatch_size": gradient_microbatch_size,
        "early_stopping_absolute_min_delta": early_stopping_absolute_min_delta,
        "early_stopping_relative_min_delta": early_stopping_relative_min_delta,
    }
    request_path = _inside_repository(request_path, "run request")
    request_bytes = request_path.read_bytes()
    request = load_run_request(request_path)
    if request_path.read_bytes() != request_bytes:
        raise TrainWorkflowError("request changed while loading")
    precision = _resolved_precision(
        request.schema_version, precision, has_parent=parent_checkpoint is not None
    )
    score_input_values = (
        evaluation_data_path,
        evaluation_data_snapshot_id,
        evaluation_data_reference,
    )
    if any(value is not None for value in score_input_values) and not all(
        value is not None for value in score_input_values
    ):
        raise TrainWorkflowError(
            "--evaluation-data-path, --evaluation-data-snapshot-id, and "
            "--evaluation-data-reference must be supplied together"
        )
    post_training = (
        request.evaluation is not None
        and request.evaluation.input_mode == "post_training_score_only"
    )
    if post_training != all(value is not None for value in score_input_values):
        raise TrainWorkflowError(
            "a post-training score-only request requires all evaluation-data arguments, and "
            "other requests must not receive them"
        )
    if training_mode == "fixed_validation" and not post_training:
        raise TrainWorkflowError("fixed_validation requires a separate post-training label panel")
    if training_mode == "fixed_refit" and post_training:
        raise TrainWorkflowError("fixed_refit must not receive validation or test labels")
    if evaluation_data_path is not None:
        if evaluation_data_path.resolve() == data_path.resolve():
            raise TrainWorkflowError(
                "post-training score data must be a separate path from training data"
            )
        if evaluation_data_path.is_symlink() or not (
            evaluation_data_path.is_dir() or evaluation_data_path.is_file()
        ):
            raise TrainWorkflowError(
                "evaluation-data-path must be a regular local file or directory"
            )
        evaluation_data_path = _runtime_data_path(evaluation_data_path)
        training_digest = directory_digest(data_path, suffixes=(".parquet",))
        score_digest = directory_digest(evaluation_data_path, suffixes=(".parquet",))
        if score_digest == training_digest or directory_content_digest(
            evaluation_data_path, suffixes=(".parquet",)
        ) == directory_content_digest(data_path, suffixes=(".parquet",)):
            raise TrainWorkflowError(
                "post-training score data must have content distinct from training data"
            )
    notifier = notifier_from_environment() if notifications_enabled else None
    if mlflow_enabled is not None:
        request = RunRequest(
            request.name,
            request.dataset_contract,
            request.model_catalog,
            request.models,
            request.origin,
            request.horizon,
            mlflow_enabled,
            request.evaluation,
            request.calendar_events,
            request.event_max_support,
            request.schema_version,
        )
    source = installed_library_identity() if installed_mode() else capture_source_identity(_root())
    runtime = capture_runtime_identity(
        _root(),
        image_digest=os.environ.get("AUTOFORECAST_RUNTIME_IMAGE_DIGEST"),
        **({"lockfile": None} if installed_mode() else {}),
    )
    contract_path = _inside_repository(_root() / request.dataset_contract, "dataset contract")
    contract_bytes = contract_path.read_bytes()
    catalog_input = _root() / request.model_catalog
    if catalog_input.is_symlink():
        raise TrainWorkflowError("model catalog must be a source-controlled directory")
    catalog_root = catalog_input.resolve()
    if not catalog_root.is_relative_to(_root().resolve()) or not catalog_root.is_dir():
        raise TrainWorkflowError("model catalog must be a source-controlled directory")
    candidates = discover_candidates(catalog_root)
    by_id = {candidate.id: candidate for candidate in candidates}
    selected_ids = tuple(by_id) if request.models == "all" else request.models
    if unknown := sorted(set(selected_ids) - set(by_id)):
        raise TrainWorkflowError(f"requested models are absent from catalog: {unknown}")
    selected = tuple(by_id[model_id] for model_id in selected_ids)
    if (fixed_contract is not None
            and fixed_contract.get("schema") == "autoforecast.fixed-evaluation.v2"):
        from autoforecast.adapter_contracts import descriptor
        for candidate in selected:
            if not descriptor(candidate.adapter).supports_shared_studies:
                raise TrainWorkflowError(
                    f"model {candidate.id!r} does not support shared studies; "
                    "native checkpoint selection requires a legacy strict benchmark"
                )
    if patience is None:
        patience = (
            10 if any(candidate.adapter in {"tft_q", "tft_hnb"} for candidate in selected) else 5
        )
    if any(candidate.adapter in {"tft_q", "tft_hnb"} for candidate in selected) and patience != 10:
        raise TrainWorkflowError("TFT-Q/TFT-HNB M5 requires validation patience 10")
    training_controls["patience"] = patience
    if precision == "16-mixed" and any(
        candidate.adapter in TRAINABLE_TORCH_ADAPTERS for candidate in selected
    ):
        training_controls["precision"] = precision
    if expected_candidate is not None:
        _require_expected_candidate(selected, expected_candidate)
    for candidate in selected:
        try:
            validate_execution_controls(
                candidate.adapter,
                device=device,
                batch_size=batch_size,
                gradient_microbatch_size=gradient_microbatch_size,
                has_generic_events=bool(request.calendar_events),
                precision=precision,
            )
        except ValueError as error:
            raise TrainWorkflowError(
                f"run controls are not executable by {candidate.id!r}: {error}"
            ) from error
    parameter_resolution: dict[str, object] | None = None
    if parameter_resolution_path is not None:
        if len(selected) != 1:
            raise TrainWorkflowError("a parameter resolution requires exactly one selected model")
        try:
            parameter_resolution = load_parameter_resolution(
                parameter_resolution_path,
                cast(str, parameter_resolution_sha256),
                selected[0],
                project_root=_root(),
            )
        except ParameterSourceError as error:
            raise TrainWorkflowError(f"parameter resolution is invalid: {error}") from error
        values = cast(dict[str, object], parameter_resolution["effective_parameters"])
        parameter_overrides = values or None
    selection_lock = _sealed_m5_selection_lock(
        request,
        selected,
        request_path=request_path,
        epochs=epochs,
        seed=seed,
        training_controls=training_controls,
    )
    if parameter_overrides is not None and len(selected) != 1:
        raise TrainWorkflowError("HPO parameter overrides require exactly one selected model")
    parent = _verified_parent_checkpoint(parent_checkpoint, parent_checkpoint_sha256)
    if parent is not None and (
        len(selected) != 1 or selected[0].adapter != "mft"
    ):
        raise TrainWorkflowError(
            "a parent checkpoint is supported only for one selected mft model"
        )
    def authenticate_configuration() -> None:
        if (request_path.read_bytes() != request_bytes
                or contract_path.read_bytes() != contract_bytes):
            raise TrainWorkflowError("request or dataset configuration changed before training")
        if (fixed_contract is not None
                and fixed_contract.get("schema") == "autoforecast.fixed-evaluation.v2"):
            from autoforecast.study_scoring import verify_study_evaluation
            verify_study_evaluation(fixed_contract)
        for item in selected:
            if item.source_path.read_bytes() != item.source_bytes:
                raise TrainWorkflowError("model profile changed before training")

    authenticate_configuration()
    runtime_checks = {candidate.id: check_runtime(candidate.runtime) for candidate in selected}
    execution_blockers = {
        candidate.id: runtime_checks[candidate.id].summary
        for candidate in selected
        if not runtime_checks[candidate.id].available
    }

    # Shut down the preflight JVM before starting the isolated adapter.
    preflight_profile, worker_profile = _preflight_spark_profiles(tuple(selected))
    authenticate_configuration()
    preflight_progress = PipelineProgress(
        "Spark preflight", 2 + len(selected), phase_key="preflight"
    )
    with preflight_progress.step("Start the bounded local Spark runtime"):
        preflight_progress.operation("Initialize the local Spark runtime")
        spark = resources.acquire_local_runtime(
            SparkSession, SparkContext, profile=preflight_profile,
            app_name="autoforecast_governed_preflight",
        )
        spark.sparkContext.setLogLevel("ERROR")
    try:
        dataset_contract = _load_dataset_contract_bytes(contract_bytes, contract_path)
        panel = dataset_contract.panel(data_path)
        with preflight_progress.step("Inspect and validate the governed data panel", spark=spark):
            preflight_progress.operation(
                "Check schema, dates, targets, duplicates, hierarchy, and coverage"
            )
            try:
                data_outcome = inspect_panel(panel, spark, as_of=request.origin)
            except Exception as error:
                data_outcome = run_incident_from_error(error)
        checked: dict[str, tuple[ModelCheckInput, CheckReport, RunApproval]] = {}
        plans: dict[str, ExecutionPlan] = {}
        if isinstance(data_outcome, CheckReport) and data_outcome.accepted:
            for candidate in selected:
                with preflight_progress.step(
                    f"Resolve and check model {candidate.id}", spark=spark
                ):
                    preflight_progress.operation(
                        f"Check data readiness and execution controls for {candidate.id}"
                    )
                    if candidate.id in execution_blockers:
                        continue
                    try:
                        plan = resolve_execution_plan(
                            candidate,
                            panel,
                            data_outcome.facts,
                            horizon=request.horizon,
                            epochs=epochs,
                            seed=seed,
                            calendar_events=request.calendar_events,
                            event_max_support=request.event_max_support,
                            parameter_overrides=parameter_overrides,
                        )
                    except ExecutionPlanError as error:
                        execution_blockers[candidate.id] = (
                            f"Execution configuration is invalid: {error}"
                        )
                        continue
                    resolved = create_resolved_run(
                        model_id=candidate.id,
                        adapter=candidate.adapter,
                        origin=request.origin,
                        horizon=request.horizon,
                        settings=dict(plan.readiness_settings),
                        frequency=panel.frequency,
                    )
                    model_input = prepare_model_check_input(
                        panel, spark, data_outcome.facts, resolved
                    )
                    required_count = dataset_contract.required_series_count
                    if required_count is not None and (
                        model_input.total_series != required_count
                        or model_input.eligible_series != required_count
                        or model_input.series_meeting_history != required_count
                        or model_input.series_with_complete_window not in (None, required_count)
                    ):
                        execution_blockers[candidate.id] = (
                            f"Study requires all {required_count} frozen series before fitting; "
                            f"{model_input.series_meeting_history} meet the model's "
                            f"{model_input.required_history}-period minimum history. "
                            "Short histories or incomplete windows cannot be silently dropped."
                        )
                        continue
                    model_checks = run_model_checks(model_input)
                    checked[candidate.id] = (
                        model_input,
                        model_checks,
                        approve_run(data_outcome, model_checks, model_input),
                    )
                    plans[candidate.id] = plan
    finally:
        _shutdown_owned_preflight_runtime(spark)

    if not isinstance(data_outcome, CheckReport) or not data_outcome.accepted:
        summary = run_training_workflow(
            request,
            output_root=output_root,
            data_outcome=data_outcome,
            checked_models=checked,
            execute_isolated=lambda _context: None,
            resolved_model_ids=selected_ids,
            execution_blockers=execution_blockers,
            notify=notifier,
        )
        return summary.exit_code

    # Worker source is copied into private scratch only.  Long-term recovery remains Git commit
    # plus protected refs/mirror, while each output bundle records that commit and its small inputs.
    # The scratch parent is durable for the lifetime of this multi-hour run; the two children are
    # removed on normal exit.
    worker_scratch_parent = _durable_worker_scratch_parent(output_root)
    with (
        tempfile.TemporaryDirectory(
            prefix="autoforecast-run-source-", dir=worker_scratch_parent
        ) as scratch,
        tempfile.TemporaryDirectory(
            prefix="autoforecast-run-config-", dir=worker_scratch_parent
        ) as config_scratch,
    ):
        # macOS exposes /var as a symlink to /private/var.  The authenticated snapshot boundary
        # intentionally rejects aliases, so canonicalise the temporary parent before publishing.
        source_snapshot = materialize_executable_source_snapshot(
            LIBRARY_ROOT.resolve() if installed_mode() else _root().resolve(),
            pathlib.Path(scratch).resolve(),
            **({"installed": True} if installed_mode() else {}),
        )
        parameter_resolution_snapshot = (
            None
            if parameter_resolution is None
            else _freeze_parameter_resolution(
                pathlib.Path(config_scratch).resolve(),
                parameter_resolution,
                cast(str, parameter_resolution_sha256),
            )
        )
        forecasts: dict[str, Forecast] = {}
        live_runs: dict[str, LiveRunMLflow] = {}
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        tracking_experiment = os.environ.get(
            "AUTOFORECAST_MLFLOW_EXPERIMENT",
            "AutoForecast governed runs",
        )

        def execute(context: CheckedModelRun) -> None:
            authenticate_configuration()
            candidate = by_id[context.model_id]
            bundle = open_run_bundle(
                context.attempt_dir.parent,
                run_id=context.attempt_dir.name,
                task="forecast",
            )
            live: LiveRunMLflow | None = None
            try:
                data = DataSnapshot(
                    data_snapshot_id,
                    context.input.data.data_fingerprint,
                    data_reference,
                )
                bundle.write_json("checks/data.json", data_outcome.as_record())
                _input, model_checks, _approval = checked[context.model_id]
                bundle.write_json("checks/model.json", model_checks.as_record())
                bundle.write_json(
                    "metadata/resolved-run.json",
                    {
                        "schema": "autoforecast.resolved-run.v1",
                        "model_id": context.input.run.model_id,
                        "adapter": context.input.run.adapter,
                        "origin": context.input.run.origin,
                        "horizon": context.input.run.horizon,
                        "settings": list(context.input.run.settings),
                        **(
                            {"frequency": context.input.run.frequency}
                            if context.input.run.frequency != "D"
                            else {}
                        ),
                        "runtime": candidate.runtime,
                    },
                )
                bundle.write_json(
                    "metadata/runtime-check.json",
                    {
                        "schema": "autoforecast.model-runtime-check.v1",
                        "runtime": candidate.runtime,
                        "available": runtime_checks[candidate.id].available,
                        "missing_modules": list(runtime_checks[candidate.id].missing_modules),
                    },
                )
                plan = plans[context.model_id]
                bundle.write_json("metadata/execution-plan.json", plan.as_record())
                if parameter_resolution is not None:
                    bundle.write_json("metadata/parameter-resolution.json", parameter_resolution)
                    _copy_parameter_source_capsule(
                        bundle,
                        parameter_resolution,
                        cast(str, parameter_resolution_sha256),
                    )
                bundle.write_json(
                    "metadata/training-controls.json",
                    {
                        "schema": (
                            "autoforecast.training-controls.v2"
                            if request.schema_version >= 3 or precision == "16-mixed"
                            else "autoforecast.training-controls.v1"
                        ),
                        "maximum_epochs": epochs,
                        **{key: value for key, value in training_controls.items()
                           if key != "precision"},
                        **(
                            {
                                "requested_precision": precision,
                                "effective_precision": effective_precision(
                                    candidate.adapter, precision
                                ),
                            }
                            if request.schema_version >= 3 or precision == "16-mixed" else {}
                        ),
                    },
                )
                if selection_lock is not None:
                    lock_path, lock_sha256, lock_body = selection_lock
                    bundle.write_json(
                        "provenance/m5-selection-lock.json",
                        {
                            "schema": "autoforecast.m5-selection-lock.v1",
                            "path": lock_path.relative_to(_root()).as_posix(),
                            "sha256": lock_sha256,
                            "selection_lock": lock_body,
                        },
                    )
                if parent is not None:
                    bundle.write_json(
                        "provenance/parent-checkpoint.json",
                        {
                            "schema": "autoforecast.child-finetune-parent.v1",
                            "checkpoint_path": str(parent[0]),
                            "checkpoint_sha256": parent[1],
                            "lifecycle": "child_finetune_fresh_optimizer",
                        },
                    )
                configuration_files: dict[str, pathlib.Path] = {}
                configuration_payloads = {
                    "dataset_contract": (contract_path.suffix, contract_bytes),
                    "run_request": (request_path.suffix, request_bytes),
                    "model_profile": (
                        candidate.source_path.suffix or ".txt",
                        candidate.source_bytes,
                    )
                }
                if candidate.implementation_config_path is not None:
                    assert candidate.implementation_config_bytes is not None
                    configuration_payloads["model_implementation_config"] = (
                        candidate.implementation_config_path.suffix or ".txt",
                        candidate.implementation_config_bytes,
                    )
                if parameter_resolution_snapshot is not None:
                    configuration_files["parameter_resolution"] = parameter_resolution_snapshot
                if selection_lock is not None:
                    configuration_files["m5_selection_lock"] = selection_lock[0]
                write_run_provenance(
                    bundle,
                    source=source,
                    runtime=runtime,
                    data=data,
                    configuration_files=configuration_files,
                    configuration_payloads=configuration_payloads,
                    lockfile_source=None if installed_mode() else _root() / "uv.lock",
                )
                if request.mlflow_enabled and tracking_uri:
                    try:
                        live = start_live_run_bundle_tracking(
                            bundle.path,
                            tracking_uri=tracking_uri,
                            experiment_name=tracking_experiment,
                        )
                    except RunMLflowError as error:
                        bundle.write_json(
                            "tracking/mlflow.json",
                            {
                                "schema": "autoforecast.generic-mlflow-link.v1",
                                "status": "not_started",
                                "detail": str(error),
                            },
                        )
                    else:
                        bundle.write_json(
                            "tracking/mlflow.json",
                            {
                                "schema": "autoforecast.generic-mlflow-link.v1",
                                "status": "started_before_worker",
                                **live.link.as_record(),
                            },
                        )
                        live_runs[context.model_id] = live
                forecast = isolated.run(
                    candidate.adapter,
                    panel,
                    request.origin,
                    request.horizon,
                    source_snapshot=source_snapshot,
                    worker_profile=worker_profile,
                    timeout_seconds=timeout_seconds,
                    heartbeat_seconds=max(1, min(60, timeout_seconds // 10)),
                    worker_log_path=bundle.path / "logs" / "worker.log",
                    tracking_context=None if live is None else live.worker_context,
                    **cast(
                        Any,
                        _worker_kwargs(
                            context,
                            plan,
                            parent_checkpoint=parent,
                            training_controls=training_controls,
                            calendar_events=request.calendar_events,
                            event_max_support=request.event_max_support,
                            training_mode=training_mode,
                            validation_data_path=(
                                evaluation_data_path
                                if training_mode == "fixed_validation"
                                else None
                            ),
                            fixed_update_budget=fixed_update_budget,
                            validation_metric=(
                                request.evaluation.metric
                                if request.evaluation is not None
                                and request.evaluation.metric in {"rmse", "wrmsse"}
                                and training_mode != "fixed_refit"
                                else None
                            ),
                        ),
                    ),
                )
                forecasts[context.model_id] = forecast
                bundle.write_json("evaluation/forecast.json", forecast.as_record())
                capability = _checkpoint_capability(bundle.path)
                if capability is not None:
                    bundle.write_json("provenance/checkpoint.json", capability.as_record())
                artifacts = _model_artifact_manifest(bundle.path)
                if artifacts is not None:
                    bundle.write_json("provenance/model-artifacts.json", artifacts)
                bundle.finalize(status="completed", summary={"forecast_path": forecast.path})
                if live is not None:
                    try:
                        # The fluent MLflow API owns one process-global active run. Close this
                        # model before the sequential campaign starts another; comparison metrics
                        # are backfilled by run ID once every model has finished.
                        live.complete(bundle.path)
                    except RunMLflowError:
                        # Local sealed evidence is authoritative. The run-level tracking report
                        # records the optional projection failure after scoring.
                        pass
            except BaseException as error:
                if live is not None:
                    live.fail()
                with contextlib.suppress(Exception):
                    bundle.finalize(status="failed", summary={"error": type(error).__name__})
                raise

        summary = run_training_workflow(
            request,
            output_root=output_root,
            data_outcome=data_outcome,
            checked_models=checked,
            execute_isolated=execute,
            resolved_model_ids=selected_ids,
            execution_blockers=execution_blockers,
            notify=notifier,
        )
        scored = _evaluate_completed_forecasts(
            request=request,
            training_panel=panel,
            forecasts=forecasts,
            run_path=summary.path,
            score_panel=(
                None
                if evaluation_data_path is None
                else dataset_contract.panel(evaluation_data_path)
            ),
            score_data_snapshot_id=evaluation_data_snapshot_id,
            score_data_reference=evaluation_data_reference,
            **({"fixed_contract": fixed_contract} if fixed_contract else {}),
        )
        if not scored:
            failure_path = summary.path / "evaluation" / "post-training-evaluation.json"
            detail = "post-training evaluation did not produce a comparable score"
            with contextlib.suppress(OSError, ValueError, TypeError):
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                if isinstance(failure, dict) and isinstance(failure.get("error"), str):
                    detail = failure["error"]
            print(
                f"[autoforecast:non-retryable] {detail}; details: {failure_path}",
                file=sys.stderr,
                flush=True,
            )
        _complete_live_mlflow(summary, live_runs)
        _export_mlflow_if_requested(
            request,
            summary,
            skip_model_ids=set(live_runs),
        )
        _write_observability_landing(summary)
    return summary.exit_code if scored else 2


@project_entrypoint
def main() -> int:
    parser = argparse.ArgumentParser(description="Run an approval-gated AutoForecast request.")
    parser.add_argument("request", type=pathlib.Path)
    parser.add_argument("--data-path", type=pathlib.Path, required=True)
    parser.add_argument("--data-snapshot-id", required=True)
    parser.add_argument("--data-reference", required=True)
    parser.add_argument("--evaluation-data-path", type=pathlib.Path)
    parser.add_argument("--evaluation-data-snapshot-id")
    parser.add_argument("--evaluation-data-reference")
    parser.add_argument("--output-root", type=pathlib.Path, default=_root() / ".autoforecast/runs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--precision", choices=("16-mixed", "32-true"))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gradient-microbatch-size", type=int)
    parser.add_argument("--early-stopping-absolute-min-delta", type=float, default=1e-6)
    parser.add_argument("--early-stopping-relative-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--training-mode",
        choices=("rolling_backtest", "fixed_validation", "fixed_refit"),
        default="rolling_backtest",
    )
    parser.add_argument("--fixed-update-budget", type=int)
    parser.add_argument("--evaluation-contract", type=pathlib.Path)
    parser.add_argument("--evaluation-contract-sha256")
    parser.add_argument("--parent-checkpoint", type=pathlib.Path)
    parser.add_argument("--parent-checkpoint-sha256")
    parser.add_argument("--parameter-resolution", type=pathlib.Path)
    parser.add_argument("--parameter-resolution-sha256")
    parser.add_argument(
        "--no-notifications",
        action="store_true",
        help="suppress child-run notices when a campaign owns the terminal notification",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.seed < 0
        or args.timeout_seconds < 2
        or args.patience is not None and args.patience < 1
        or args.batch_size < 1
    ):
        parser.error(
            "--epochs, --patience, and --batch-size must be positive; --seed non-negative; "
            "--timeout-seconds must be at least 2"
        )
    if args.gradient_microbatch_size is not None and args.gradient_microbatch_size < 1:
        parser.error("--gradient-microbatch-size must be positive when supplied")
    try:
        return run_request(
            args.request,
            data_path=args.data_path,
            data_snapshot_id=args.data_snapshot_id,
            data_reference=args.data_reference,
            evaluation_data_path=args.evaluation_data_path,
            evaluation_data_snapshot_id=args.evaluation_data_snapshot_id,
            evaluation_data_reference=args.evaluation_data_reference,
            output_root=args.output_root,
            epochs=args.epochs,
            timeout_seconds=args.timeout_seconds,
            seed=args.seed,
            patience=args.patience,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            gradient_microbatch_size=args.gradient_microbatch_size,
            early_stopping_absolute_min_delta=args.early_stopping_absolute_min_delta,
            early_stopping_relative_min_delta=args.early_stopping_relative_min_delta,
            parent_checkpoint=args.parent_checkpoint,
            parent_checkpoint_sha256=args.parent_checkpoint_sha256,
            parameter_resolution_path=args.parameter_resolution,
            parameter_resolution_sha256=args.parameter_resolution_sha256,
            notifications_enabled=not args.no_notifications,
            training_mode=args.training_mode,
            fixed_update_budget=args.fixed_update_budget,
            evaluation_contract_path=args.evaluation_contract,
            evaluation_contract_sha256=args.evaluation_contract_sha256,
        )
    except (ProvenanceError, TrainWorkflowError, TrainingLeaseError) as error:
        parser.error(str(error))
    return 2  # pragma: no cover - parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
