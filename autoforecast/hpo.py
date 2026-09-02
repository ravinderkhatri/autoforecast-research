"""Governed Optuna/Ray execution over the existing model-YAML tuning domains.

The model YAML owns sampled hyperparameters.  A separate immutable study YAML owns the data
fingerprint, split, selection objective and compute budget.  An evaluator is model-specific, but
the execution lifecycle, artifact layout and test-set prohibition are shared.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import re
import sys
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from autoforecast._exact_json import load_exact_json
from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast._numeric import is_finite_real
from autoforecast.experiment_report import render_study_report
from autoforecast.experiments import (
    LocalAttempt,
    load_completed_attempt,
    load_study_selection,
    open_local_attempt,
    open_local_heldout_attempt,
    require_heldout_evaluation_approval,
    seal_study_selection,
)
from autoforecast.frequency import TimePoint, parse_origin
from autoforecast.mlflow_export import MLflowTrainingRun, start_training_tracking
from autoforecast.tuning import (
    PARAMETER_EXECUTION_COMPATIBILITY,
    TuningSpec,
    load_tuning_spec,
    materialize_trial_parameters,
    ray_param_space,
    suggest_optuna_parameters,
)
from provenance.lineage import directory_digest

__all__ = [
    "HPOError",
    "HPOStudySpec",
    "HeldoutOutcome",
    "HeldoutRequest",
    "TrialOutcome",
    "TrialRequest",
    "load_hpo_study",
    "run_heldout_evaluation",
    "run_hpo_study",
    "sync_hpo_selection_to_mlflow",
]

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_SELECTION_METRIC_TOKENS = frozenset({"final", "heldout", "private", "test"})
_ROOT_KEYS = frozenset({"schema_version", "study"})
_STUDY_KEYS = frozenset(
    {
        "id",
        "model_config",
        "backend",
        "direction",
        "objective",
        "max_trials",
        "top_k",
        "seed",
        "comparability_fingerprint",
        "data",
        "evaluator",
    }
)
_DATA_KEYS = frozenset(
    {
        "id",
        "fingerprint",
        "observed_date_range",
        "training_date_range",
        "validation_date_range",
        "test_date_range",
        "validation_target_date_range",
        "test_target_date_range",
        "training_series",
        "validation_series",
        "test",
    }
)


class HPOError(ValueError):
    """An HPO study would be ambiguous, leaky, or lack required trial evidence."""


@contextlib.contextmanager
def _study_lease(output: pathlib.Path, study_id: str) -> Iterator[None]:
    """Hold one stable local lease for a study's full paid-compute lifecycle."""
    output.mkdir(parents=True, exist_ok=True)
    lease_root = output / ".leases"
    if output.is_symlink() or lease_root.is_symlink():
        raise HPOError("HPO output and lease roots may not be symlinks")
    lease_root.mkdir(exist_ok=True)
    path = lease_root / f"{study_id}.lock"
    if path.is_symlink():
        raise HPOError("HPO study lease may not be a symlink")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise HPOError(f"cannot open HPO study lease: {path}") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise HPOError(f"HPO study {study_id!r} is already running") from error
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class HPOStudySpec:
    """The frozen contract shared by every candidate trial in one HPO study."""

    id: str
    backend: str
    direction: str
    objective: str
    max_trials: int
    top_k: int
    seed: int
    comparability_fingerprint: str
    data: dict[str, object]
    evaluator: str
    tuning: TuningSpec
    source_path: pathlib.Path
    source_sha256: str


@dataclass(frozen=True)
class TrialRequest:
    """Inert trial inputs; the evaluator writes artifacts only below ``attempt.path``."""

    study: HPOStudySpec
    trial_id: str
    attempt: LocalAttempt
    model_parameters: dict[str, object]
    training_parameters: dict[str, object]
    epoch_observer: Callable[[int, Mapping[str, float]], None] | None = None


@dataclass(frozen=True)
class TrialOutcome:
    """Validation-only result returned by one model-specific evaluator."""

    validation_objective: float
    metrics: Mapping[str, float] | None = None


@dataclass(frozen=True)
class HeldoutRequest:
    """Inputs for one selected candidate's held-out evaluation.

    The source attempt is completed and immutable. The evaluator may load its saved model or
    checkpoint, but writes all new evidence only beneath evaluation_attempt.path.
    """

    study: HPOStudySpec
    source_attempt: pathlib.Path
    source_selection: Mapping[str, object]
    evaluation_attempt: LocalAttempt
    model_parameters: dict[str, object]
    training_parameters: dict[str, object]


@dataclass(frozen=True)
class HeldoutOutcome:
    """One finite held-out objective and optional additional held-out metrics."""

    heldout_objective: float
    metrics: Mapping[str, float] | None = None


class TrialEvaluator(Protocol):
    def __call__(self, request: TrialRequest) -> TrialOutcome: ...


class HeldoutEvaluator(Protocol):
    def __call__(self, request: HeldoutRequest) -> HeldoutOutcome: ...


def _finite_builtin_number(value: object) -> bool:
    return is_finite_real(value)


def _selection_metric_name(name: object) -> bool:
    if type(name) is not str or re.fullmatch(
        r"(?:validation|evaluation|optimization)/[a-z][a-z0-9_]{0,63}", name
    ) is None:
        return False
    tokens = {token for token in re.split(r"[^a-z0-9]+", name.casefold()) if token}
    return not tokens.intersection(_RESERVED_SELECTION_METRIC_TOKENS)


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise HPOError(f"{label} must be a mapping")
    return value


def _date_range(value: object, label: str) -> tuple[TimePoint, TimePoint]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
    ):
        raise HPOError(f"{label} must be a two-item ISO-8601 date/time range")
    try:
        start, end = (
            parse_origin(item, "H" if "T" in item else "D")
            for item in value
        )
    except ValueError as exc:
        raise HPOError(f"{label} must be a two-item ISO-8601 date/time range") from exc
    if type(start) is not type(end):
        raise HPOError(f"{label} endpoints must use the same temporal key type")
    if start > end:
        raise HPOError(f"{label} start must not be after its end")
    return start, end


def _validate_data(data: dict[str, object]) -> None:
    if set(data) != _DATA_KEYS:
        raise HPOError(f"study.data must contain exactly {sorted(_DATA_KEYS)}")
    if not isinstance(data["id"], str) or not data["id"]:
        raise HPOError("study.data.id must be non-blank")
    if not isinstance(data["fingerprint"], str) or _SHA256.fullmatch(data["fingerprint"]) is None:
        raise HPOError("study.data.fingerprint must be a complete lowercase SHA-256")
    for name in (
        "observed_date_range",
        "training_date_range",
        "validation_date_range",
        "test_date_range",
        "validation_target_date_range",
        "test_target_date_range",
    ):
        _date_range(data[name], f"study.data.{name}")
    ranges = {
        name: _date_range(data[name], f"study.data.{name}")
        for name in (
            "observed_date_range",
            "training_date_range",
            "validation_date_range",
            "test_date_range",
            "validation_target_date_range",
            "test_target_date_range",
        )
    }
    if len({type(point) for bounds in ranges.values() for point in bounds}) != 1:
        raise HPOError("all study data ranges must use the same temporal key type")
    observed_start, observed_end = ranges["observed_date_range"]
    validation_start, validation_end = ranges["validation_target_date_range"]
    test_start, test_end = ranges["test_target_date_range"]
    if validation_end >= test_start:
        raise HPOError(
            "study.data.test_target_date_range must start strictly after "
            "validation_target_date_range"
        )
    if any(start < observed_start or end > observed_end for start, end in ranges.values()):
        raise HPOError("all study data ranges must be contained in observed_date_range")
    for name in ("training_date_range", "validation_date_range"):
        if ranges[name][1] >= validation_start:
            raise HPOError(f"study.data.{name} must end before validation_target_date_range starts")
    test_data_start, test_data_end = ranges["test_date_range"]
    if (
        test_data_start <= validation_end
        or test_start < test_data_start
        or test_end > test_data_end
    ):
        raise HPOError(
            "study.data.test_date_range must begin after validation targets and contain "
            "test_target_date_range"
        )
    for name in ("training_series", "validation_series"):
        count = data[name]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise HPOError(
                "study.data training_series and validation_series must be positive integers"
            )
    if data["test"] != "reserved_not_evaluated":
        raise HPOError("HPO study.data.test must be 'reserved_not_evaluated'")


def load_hpo_study(path: str | pathlib.Path, *, project_root: str | pathlib.Path) -> HPOStudySpec:
    """Load a strict study definition and bind it to the model YAML byte identity."""
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise HPOError(f"study config must be a regular file, not {source}")
    raw = source.read_bytes()
    try:
        root = _mapping(loads_exact_yaml(raw), str(source))
    except HPOError:
        raise
    except ExactYAMLError as exc:
        raise HPOError(f"invalid YAML in {source}: {exc}") from exc
    if set(root) != _ROOT_KEYS or root.get("schema_version") != 1:
        raise HPOError("study config must contain schema_version: 1 and a study mapping")
    study = _mapping(root.get("study"), "study")
    if set(study) != _STUDY_KEYS:
        raise HPOError(f"study must contain exactly {sorted(_STUDY_KEYS)}")
    identifier = study["id"]
    if not isinstance(identifier, str) or _ID.fullmatch(identifier) is None:
        raise HPOError("study.id is invalid")
    backend, direction, objective = study["backend"], study["direction"], study["objective"]
    if (
        not isinstance(backend, str)
        or not isinstance(direction, str)
        or not isinstance(objective, str)
    ):
        raise HPOError("study backend, direction and objective must be strings")
    if backend not in {"optuna", "ray"}:
        raise HPOError("study.backend must be 'optuna' or 'ray'")
    if direction not in {"minimize", "maximize"}:
        raise HPOError("study.direction must be 'minimize' or 'maximize'")
    if objective != "validation/objective":
        raise HPOError("HPO objective must be exactly 'validation/objective'; test is forbidden")
    max_trials, top_k, seed = study["max_trials"], study["top_k"], study["seed"]
    if (
        not isinstance(max_trials, int)
        or isinstance(max_trials, bool)
        or max_trials < 1
        or not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k <= max_trials
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise HPOError(
            "study.max_trials must be positive, top_k must be within its budget, and "
            "study.seed must be an integer in [0, 2**32 - 1]"
        )
    fingerprint = study["comparability_fingerprint"]
    if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
        raise HPOError("study.comparability_fingerprint must be a complete lowercase SHA-256")
    data = _mapping(study["data"], "study.data")
    _validate_data(data)
    evaluator = study["evaluator"]
    if not isinstance(evaluator, str) or (evaluator != "builtin" and ":" not in evaluator):
        raise HPOError(
            "study.evaluator must be 'builtin' or a model-specific module:function bridge"
        )
    model_config = study["model_config"]
    if not isinstance(model_config, str):
        raise HPOError("study.model_config must be a relative source-controlled path")
    model_path = pathlib.PurePosixPath(model_config)
    if model_path.is_absolute() or ".." in model_path.parts:
        raise HPOError("study.model_config must be a relative source-controlled path")
    config_path = pathlib.Path(project_root) / model_config
    try:
        tuning = load_tuning_spec(config_path)
    except Exception as exc:
        raise HPOError(f"invalid HPO model config {model_config!r}: {exc}") from exc
    return HPOStudySpec(
        id=identifier,
        backend=backend,
        direction=direction,
        objective=objective,
        max_trials=max_trials,
        top_k=top_k,
        seed=seed,
        comparability_fingerprint=fingerprint,
        data=data,
        evaluator=evaluator,
        tuning=tuning,
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _trial_metadata(study: HPOStudySpec, parameters: dict[str, object]) -> dict[str, object]:
    data = {"name": study.data["id"], **study.data}
    return {
        "purpose": "hpo_validation_selection",
        "parameter_execution_compatibility": PARAMETER_EXECUTION_COMPATIBILITY,
        "comparability_fingerprint": study.comparability_fingerprint,
        "model": {
            "id": study.tuning.model_id,
            "adapter": study.tuning.adapter,
            "config_sha256": study.tuning.source_sha256,
        },
        "study": {
            "config_sha256": study.source_sha256,
            "objective": study.objective,
            "direction": study.direction,
            "backend": study.backend,
        },
        "hyperparameters": parameters,
        "data": data,
    }


def _require_trial_evidence(
    attempt: LocalAttempt, *, external_fingerprint: str | None = None,
) -> None:
    # Classical/boosted adapters do not have an epoch loop. They must still leave the governed
    # runner record (or external callback evidence); neural adapters additionally retain
    # their epoch metrics when available.
    has_epochs = attempt.metrics_path.is_file() and bool(
        attempt.metrics_path.read_text(encoding="utf-8").strip()
    )
    governed = attempt.path / "evaluation" / "governed-run.json"
    if external_fingerprint is not None:
        try:
            record = load_exact_json(attempt.path / "evaluation" / "external-run.json")
            if (type(record) is not dict
                    or record.get("schema") != "autoforecast.external-trial.v1"
                    or record.get("test_metrics") != "not_evaluated"
                    or type(record.get("context")) is not dict):
                raise HPOError("invalid external trial evidence")
            context_bytes = (json.dumps(
                record["context"], sort_keys=True, allow_nan=False,
            ) + "\n").encode()
            if (hashlib.sha256(context_bytes).hexdigest() != external_fingerprint
                    or record.get("model_sha256") != directory_digest(attempt.path / "model")
                    or record.get("forecast_sha256") != directory_digest(
                        attempt.path / "forecasts", suffixes=(".parquet",),
                    )):
                raise HPOError("external trial evidence differs from frozen inputs or artifacts")
        except (OSError, ValueError) as error:
            raise HPOError("external trial evidence is missing, invalid or changed") from error
    elif not has_epochs and not governed.is_file():
        raise HPOError(
            "completed HPO trial needs epoch metrics or an evaluation/governed-run.json record"
        )
    artifact_directories = (attempt.path / "model", attempt.path / "checkpoints")
    saved_artifacts = (path for directory in artifact_directories for path in directory.rglob("*"))
    if not any(path.is_file() for path in saved_artifacts):
        raise HPOError("completed HPO trial needs a saved model or checkpoint")


def _evaluate(
    study: HPOStudySpec,
    root: pathlib.Path,
    trial_id: str,
    parameters: dict[str, object],
    evaluator: TrialEvaluator,
    *,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str,
) -> float:
    attempt = open_local_attempt(
        root,
        study_id=study.id,
        trial_id=trial_id,
        attempt_id=f"attempt-{uuid.uuid4().hex}",
        metadata=_trial_metadata(study, parameters),
    )
    external_fingerprint = (
        study.comparability_fingerprint
        if (study.tuning.adapter, study.evaluator) == ("external", "external:fit_predict")
        else None
    )
    live: MLflowTrainingRun | None = None
    tracking_record: dict[str, object] | None = None
    try:
        if mlflow_tracking_uri is not None:
            try:
                metadata = json.loads(
                    (attempt.path / "metadata" / "attempt.json").read_text(encoding="utf-8")
                )
                live = start_training_tracking(
                    metadata=metadata,
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=mlflow_experiment,
                )
            except Exception as tracking_error:
                tracking_record = {
                    "schema": "autoforecast.mlflow-tracking-link.v2",
                    "status": "unavailable",
                    "tracking_uri_sha256": hashlib.sha256(
                        mlflow_tracking_uri.encode()
                    ).hexdigest(),
                    "error_type": type(tracking_error).__name__,
                }
            else:
                tracking_record = {
                    "schema": "autoforecast.mlflow-tracking-link.v2",
                    "status": "started",
                    "tracking_uri_sha256": hashlib.sha256(
                        mlflow_tracking_uri.encode()
                    ).hexdigest(),
                    "experiment_name": mlflow_experiment,
                    "experiment_id": live.experiment_id,
                    "run_id": live.run_id,
                    "system_metrics": "enabled_from_training_start",
                    "optimization_metrics": "streamed_live_then_verified_from_jsonl",
                    "run_kind": "hpo_candidate",
                }
        grouped = materialize_trial_parameters(study.tuning, parameters)
        outcome = evaluator(
            TrialRequest(
                study=study,
                trial_id=trial_id,
                attempt=attempt,
                model_parameters=grouped["model"],
                training_parameters=grouped["training"],
                epoch_observer=None if live is None else live.log_epoch,
            )
        )
        if not isinstance(outcome, TrialOutcome) or not _finite_builtin_number(
            outcome.validation_objective
        ):
            raise HPOError("evaluator must return a finite TrialOutcome.validation_objective")
        metrics = dict(outcome.metrics or {})
        if any(not _selection_metric_name(name) for name in metrics) or any(
            not _finite_builtin_number(value) for value in metrics.values()
        ):
            raise HPOError("HPO evaluator metrics must be finite and may not report test metrics")
        if tracking_record is not None:
            # The evaluator is model code and must not be able to substitute a remote run ID.
            # Exclusive publication after it returns makes any forged pre-write fail the trial.
            attempt.write_json("tracking/mlflow.json", tracking_record)
        _require_trial_evidence(attempt, external_fingerprint=external_fingerprint)
        attempt.write_json(
            "evaluation/validation.json",
            {
                "schema": "autoforecast.hpo-validation.v1",
                "objective": study.objective,
                "value": outcome.validation_objective,
                "metrics": metrics,
                "test_metrics": "not_evaluated",
            },
        )
        attempt.finalize(
            status="completed",
            summary={
                "selection_metric": study.objective,
                "validation_objective": outcome.validation_objective,
                "test_metrics": "not_evaluated",
            },
        )
        if live is not None:
            try:
                live.complete(attempt.path)
            except Exception:
                # The immutable local attempt is authoritative. MLflow is an optional view and
                # cannot retroactively turn a completed paid trial into a failed trial.
                pass
        return outcome.validation_objective
    except Exception as exc:
        if not (attempt.path / "completion.json").exists():
            attempt.write_json(
                "evaluation/failure.json",
                {
                    "schema": "autoforecast.hpo-failure.v1",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            attempt.finalize(status="failed", summary={"test_metrics": "not_evaluated"})
        if live is not None:
            try:
                live.fail()
            except Exception:
                pass
        raise


def _run_optuna(
    study: HPOStudySpec,
    root: pathlib.Path,
    evaluator: TrialEvaluator,
    *,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str,
) -> None:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise HPOError("Optuna runner requires: pip install '.[hpo-optuna]'") from exc
    sampler = optuna.samplers.TPESampler(seed=study.seed)
    direction = cast(Literal["minimize", "maximize"], study.direction)
    optuna_study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial: Any) -> float:
        parameters = suggest_optuna_parameters(study.tuning, trial)
        value = _evaluate(
            study,
            root,
            f"trial-{trial.number:04d}",
            parameters,
            evaluator,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
        )
        trial.set_user_attr("validation_objective", value)
        return value

    optuna_study.optimize(objective, n_trials=study.max_trials)


def _run_ray(
    study: HPOStudySpec,
    root: pathlib.Path,
    evaluator: TrialEvaluator,
    *,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str,
) -> None:
    try:
        import ray
        from ray import tune
        from ray.tune.search.basic_variant import BasicVariantGenerator
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise HPOError("Ray runner requires: pip install '.[hpo-ray]'") from exc

    # A session created here is local and owned by this runner.  Keep its workers on the exact
    # interpreter that imported Ray.  Invoking ``uv run`` as the worker command makes uv resolve
    # the packaged Ray working directory as a new project, creates an empty .venv there, and can
    # leave every trial pending because that environment has no Ray installation.
    owns_ray_session = not ray.is_initialized()
    if owns_ray_session:
        environment_root = str(pathlib.Path(sys.executable).parent.parent)
        ray.init(
            runtime_env={
                "py_executable": sys.executable,
                "env_vars": {"UV_PROJECT_ENVIRONMENT": environment_root},
            },
            include_dashboard=False,
        )

    def trainable(parameters: dict[str, object]) -> None:
        trial_id = f"trial-{uuid.uuid4().hex}"
        value = _evaluate(
            study,
            root,
            trial_id,
            parameters,
            evaluator,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
        )
        tune.report({study.objective: value})

    try:
        tuner = tune.Tuner(
            trainable,
            param_space=ray_param_space(study.tuning),
            tune_config=tune.TuneConfig(
                metric=study.objective,
                mode={"minimize": "min", "maximize": "max"}[study.direction],
                num_samples=study.max_trials,
                # The campaign supervisor owns one heavy-stage resource lease. Letting Ray fan out
                # trials inside that stage would bypass its RAM/device admission and can start two
                # Spark or accelerator-backed trainers on the same single-machine resource.
                max_concurrent_trials=1,
                search_alg=BasicVariantGenerator(random_state=study.seed),
            ),
            run_config=tune.RunConfig(name=study.id, storage_path=str(root / "ray")),
        )
        results = tuner.fit()
        if any(result.error for result in results):
            raise HPOError(
                "one or more Ray HPO trials failed; inspect their immutable failure records"
            )
    finally:
        if owns_ray_session:
            ray.shutdown()


def _write_mlflow_sync_record(study_root: pathlib.Path, record: dict[str, object]) -> pathlib.Path:
    path = study_root / "mlflow-selection-sync.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def sync_hpo_selection_to_mlflow(
    study_root: pathlib.Path,
    *,
    tracking_uri: str | None,
) -> pathlib.Path:
    """Add selection tags after the local, immutable study selection is sealed.

    MLflow remains a view: selection is read from ``selection.json`` and is never read back from
    MLflow to decide which candidate may receive held-out evaluation.
    """
    if tracking_uri is None:
        return _write_mlflow_sync_record(
            study_root,
            {
                "schema": "autoforecast.hpo-mlflow-selection-sync.v1",
                "status": "disabled",
                "attempted_run_ids": [],
                "synchronized_run_ids": [],
                "error_type": None,
            },
        )
    attempted: list[str] = []
    synchronized: list[str] = []
    try:
        import mlflow

        selection = load_study_selection(study_root)
        selected = {
            (str(item["trial_id"]), str(item["attempt_id"])): int(item["rank"])
            for item in selection["selected"]
        }
        client = mlflow.tracking.MlflowClient(tracking_uri)  # pyright: ignore[reportPrivateImportUsage]
        for attempt in study_root.glob("trials/*/attempts/*"):
            metadata, completion = load_completed_attempt(attempt)
            if completion.get("status") != "completed":
                continue
            link = attempt / "tracking" / "mlflow.json"
            if not link.is_file():
                continue
            record = load_exact_json(link)
            run_id = record.get("run_id") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("schema") != "autoforecast.mlflow-tracking-link.v2"
                or record.get("status") != "started"
                or not isinstance(record.get("tracking_uri_sha256"), str)
                or record["tracking_uri_sha256"]
                != hashlib.sha256(tracking_uri.encode()).hexdigest()
                or not isinstance(run_id, str)
            ):
                continue
            remote = client.get_run(run_id)
            expected_tags = {
                "autoforecast.study_id": str(metadata.get("study_id")),
                "autoforecast.trial_id": str(metadata.get("trial_id")),
                "autoforecast.attempt_id": str(metadata.get("attempt_id")),
            }
            remote_tags = getattr(getattr(remote, "data", None), "tags", None)
            remote_experiment = getattr(getattr(remote, "info", None), "experiment_id", None)
            if (
                not isinstance(remote_tags, dict)
                or any(remote_tags.get(name) != value for name, value in expected_tags.items())
                or remote_experiment != record.get("experiment_id")
            ):
                raise HPOError("MLflow run identity does not match the sealed HPO attempt")
            attempted.append(run_id)
            key = (attempt.parent.parent.name, attempt.name)
            selected_rank = selected.get(key)
            client.set_tag(
                run_id,
                "autoforecast.hpo_selection",
                "selected_for_heldout" if selected_rank is not None else "not_selected",
            )
            if selected_rank is not None:
                client.set_tag(run_id, "autoforecast.hpo_rank", str(selected_rank))
            synchronized.append(run_id)
    except Exception as error:
        return _write_mlflow_sync_record(
            study_root,
            {
                "schema": "autoforecast.hpo-mlflow-selection-sync.v1",
                "status": "failed",
                "attempted_run_ids": attempted,
                "synchronized_run_ids": synchronized,
                "error_type": type(error).__name__,
            },
        )
    return _write_mlflow_sync_record(
        study_root,
        {
            "schema": "autoforecast.hpo-mlflow-selection-sync.v1",
            "status": "synchronized",
            "attempted_run_ids": attempted,
            "synchronized_run_ids": synchronized,
            "error_type": None,
        },
    )


def run_heldout_evaluation(
    study: HPOStudySpec,
    root: str | pathlib.Path,
    *,
    trial_id: str,
    attempt_id: str,
    evaluator: HeldoutEvaluator,
) -> pathlib.Path:
    """Evaluate one sealed selected candidate on held-out targets in a fresh immutable bundle."""
    output = pathlib.Path(root)
    study_root = output / "studies" / study.id
    selected = require_heldout_evaluation_approval(
        study_root,
        trial_id=trial_id,
        attempt_id=attempt_id,
    )
    source_relative = selected.get("attempt_path")
    if not isinstance(source_relative, str):
        raise HPOError("sealed selection has no source attempt path")
    source_attempt = study_root / source_relative
    metadata, _completion = load_completed_attempt(source_attempt)
    parameters = metadata.get("hyperparameters")
    if not isinstance(parameters, dict):
        raise HPOError("selected attempt has no hyperparameters")
    grouped = materialize_trial_parameters(study.tuning, parameters)
    source_data = metadata.get("data")
    if not isinstance(source_data, dict):
        raise HPOError("selected attempt has no data contract")
    heldout_data = {**source_data, "test": "evaluated"}
    evaluation_id = f"heldout-{uuid.uuid4().hex}"
    attempt = open_local_heldout_attempt(
        output,
        study_id=study.id,
        trial_id=trial_id,
        attempt_id=evaluation_id,
        metadata={
            "purpose": "heldout_evaluation",
            "comparability_fingerprint": study.comparability_fingerprint,
            "model": metadata.get("model"),
            "study": metadata.get("study"),
            "hyperparameters": parameters,
            "data": heldout_data,
            "selected_source": {
                "trial_id": trial_id,
                "attempt_id": attempt_id,
                "artifact_manifest_sha256": selected.get("artifact_manifest_sha256"),
            },
        },
    )
    try:
        outcome = evaluator(
            HeldoutRequest(
                study=study,
                source_attempt=source_attempt,
                source_selection=selected,
                evaluation_attempt=attempt,
                model_parameters=grouped["model"],
                training_parameters=grouped["training"],
            )
        )
        if not isinstance(outcome, HeldoutOutcome) or not _finite_builtin_number(
            outcome.heldout_objective
        ):
            raise HPOError(
                "held-out evaluator must return a finite HeldoutOutcome.heldout_objective"
            )
        metrics = dict(outcome.metrics or {})
        if any(type(name) is not str or not name for name in metrics) or any(
            not _finite_builtin_number(value) for value in metrics.values()
        ):
            raise HPOError("held-out evaluator metrics must be finite and named")
        attempt.write_json(
            "evaluation/heldout.json",
            {
                "schema": "autoforecast.heldout-evaluation.v1",
                "objective": "heldout/objective",
                "value": outcome.heldout_objective,
                "metrics": metrics,
                "selected_source": {
                    "trial_id": trial_id,
                    "attempt_id": attempt_id,
                    "artifact_manifest_sha256": selected.get("artifact_manifest_sha256"),
                },
            },
        )
        return attempt.finalize(
            status="completed",
            summary={
                "evaluation_metric": "heldout/objective",
                "heldout_objective": outcome.heldout_objective,
            },
        )
    except Exception as exc:
        if not (attempt.path / "completion.json").exists():
            attempt.write_json(
                "evaluation/failure.json",
                {
                    "schema": "autoforecast.heldout-failure.v1",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            attempt.finalize(status="failed", summary={"heldout_metrics": "not_evaluated"})
        raise


def run_hpo_study(
    study: HPOStudySpec,
    root: str | pathlib.Path,
    evaluator: TrialEvaluator,
    *,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "AutoForecast HPO",
) -> pathlib.Path:
    """Run one fresh study while excluding duplicate local paid-compute owners."""
    output = pathlib.Path(root)
    with _study_lease(output, study.id):
        return _run_hpo_study_locked(
            study,
            output,
            evaluator,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
        )


def _run_hpo_study_locked(
    study: HPOStudySpec,
    root: str | pathlib.Path,
    evaluator: TrialEvaluator,
    *,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "AutoForecast HPO",
) -> pathlib.Path:
    """Execute a fresh validation-only study beneath an already-held study lease."""
    output = pathlib.Path(root)
    study_root = output / "studies" / study.id
    if study_root.exists():
        raise HPOError(f"study output already exists: {study_root}; use a new frozen study id")
    if mlflow_tracking_uri is not None and (
        not isinstance(mlflow_tracking_uri, str) or not mlflow_tracking_uri
    ):
        raise HPOError("mlflow_tracking_uri must be a non-blank string when supplied")
    if not isinstance(mlflow_experiment, str) or not mlflow_experiment:
        raise HPOError("mlflow_experiment must be a non-blank string")
    if study.backend == "optuna":
        _run_optuna(
            study,
            output,
            evaluator,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
        )
    else:
        _run_ray(
            study,
            output,
            evaluator,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
        )
    seal_study_selection(
        study_root,
        top_k=study.top_k,
        direction=study.direction,
        required_trial_count=study.max_trials,
    )
    sync_hpo_selection_to_mlflow(study_root, tracking_uri=mlflow_tracking_uri)
    return render_study_report(study_root, mlflow_url=mlflow_tracking_uri)
