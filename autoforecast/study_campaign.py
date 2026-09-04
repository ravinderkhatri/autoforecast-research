"""Bridge registered custom models into the existing durable campaign runner.

Campaign stages own phase order/recovery; Ray supplies optional process placement.
The same study workflow fits and scores each model in every execution mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import socket
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, cast

from mlops._safe_files import publish_bytes, read_regular_bytes

from autoforecast._exact_json import load_exact_json
from autoforecast._exact_yaml import loads_exact_yaml
from autoforecast._execution import (
    Execution,
    _console,
    _stage,
    _stage_event,
    configure_logging,
    note,
    record_result,
    resolve_level,
    standalone_attempt,
)
from autoforecast.benchmark import digest, verify_benchmark
from autoforecast.campaign import CampaignStore, CampaignSupervisor, _write_new_json
from autoforecast.implementation import (
    CapturedImplementation,
    resolve_implementation,
    verify_retained,
    worker_implementation,
)
from autoforecast.study_tuning import (
    FitPredictRequest,
    FitPredictResult,
    _config,
    _implementation_record,
    _output_root,
    _resolve_options,
    _validation_controls,
)
from autoforecast.study_workflow import load_forecast_workflow


@dataclasses.dataclass(frozen=True)
class StudyModel:
    """An installed Python entry point plus an existing external parameter profile.

    entrypoint uses module:function, not a notebook closure. fixed_parameters=None
    tunes; {} uses defaults without search. CPU/memory declarations include the
    model's own workers. Every participating host needs the same code and storage.
    """

    entrypoint: str
    implementation: Mapping[str, object] | None = None
    config: str | pathlib.Path | None = None
    fixed_parameters: Mapping[str, object] | None = None
    cpus: int = 1
    memory_gib: float = 4.0
    gpus: int = 0
    source_modules: tuple[str, ...] = ()
    node_labels: Mapping[str, str] | None = None


def _runtime_identity() -> dict:
    """Identify the library implementation and pinned numerical/execution runtime."""
    root = pathlib.Path(__file__).resolve().parent.parent
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for package in ("autoforecast", "forecast", "mlops", "provenance", "spec", "observability")
        for path in (root / package).rglob("*.py")
        if not (path.name.startswith("test_") or path.stem.endswith("_test"))
    }
    return {
        "python": platform.python_version(),
        "source_sha256": digest(files),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "pyspark",
                "pyarrow",
                "numpy",
                "pandas",
                "optuna",
                "ray",
            )
        },
    }


def _module_hash(name: str) -> str:
    module = importlib.import_module(name)
    if not module.__file__:
        raise ValueError("model source module must have a source file")
    return hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()


def _read(path: str | pathlib.Path, sha: str) -> dict:
    payload = pathlib.Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError("campaign input identity changed")
    return json.loads(payload)


def _entrypoint(
    name: str, expected_sha: str | None = None
) -> tuple[
    Callable[[FitPredictRequest], FitPredictResult],
    str,
]:
    parts = name.split(":")
    if len(parts) != 2 or not all(parts) or "<" in name:
        raise ValueError("model entrypoint must be an installed module:function")
    module = importlib.import_module(parts[0])
    function = getattr(module, parts[1])
    if not callable(function) or not module.__file__:
        raise ValueError("entrypoint must name an importable function")
    sha = hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise ValueError("model entrypoint code differs on worker")
    return cast(Callable[[FitPredictRequest], FitPredictResult], function), sha


def run_study_campaign(
    study,
    *,
    models: Sequence[StudyModel],
    output_dir: str | pathlib.Path | None = None,
    name: str = "custom-models",
    execution: str = "sequential",
    max_parallel_models: int = 1,
    final_evaluation: str = "all_models",
    ray_address: str | None = None,
    timeout_seconds: int = 3600,
    max_attempts: int = 1,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Run custom models through durable validation, comparison and final stages.

    Local/cluster execution uses Ray model jobs; trials remain sequential within
    each model. A supplied existing Ray address is required for cluster mode.
    Output storage and study paths must be shared in cluster mode. Calling again
    with the identical request reopens verified completed work; it does not retune.
    """
    from autoforecast.study_preflight import preflight_study

    if execution not in {"sequential", "local", "cluster"}:
        raise ValueError("execution must be sequential, local or cluster")
    if (tracking_uri is None) != (experiment_id is None):
        raise ValueError("tracking_uri and existing experiment_id must be supplied together")
    if final_evaluation not in {"all_models", "validation_winner"}:
        raise ValueError("final_evaluation must be all_models or validation_winner")
    if (execution == "cluster") != (ray_address is not None):
        raise ValueError("only cluster execution requires an explicit ray_address")
    for label, value in (
        ("max_parallel_models", max_parallel_models),
        ("timeout_seconds", timeout_seconds),
        ("max_attempts", max_attempts),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if execution == "sequential" and max_parallel_models != 1:
        raise ValueError("sequential execution requires max_parallel_models=1")
    if not models:
        raise ValueError("supply at least one model")
    resolved, options = [], []
    for model in models:
        if type(model) is not StudyModel:
            raise TypeError("models must contain StudyModel declarations")
        if (
            type(model.cpus) is not int
            or model.cpus < 1
            or type(model.gpus) is not int
            or model.gpus < 0
        ):
            raise ValueError("model CPUs must be positive and GPUs nonnegative integers")
        if (
            type(model.memory_gib) not in (int, float)
            or not math.isfinite(model.memory_gib)
            or model.memory_gib <= 0
        ):
            raise ValueError("model memory_gib must be positive and finite")
        if execution == "sequential" and model.gpus:
            raise ValueError("GPU reservations require local or cluster execution")
        if execution == "sequential" and model.cpus > (os.cpu_count() or 1):
            raise ValueError("model CPU request exceeds this machine's capacity")
        if not isinstance(model.source_modules, tuple) or any(
            not isinstance(module, str) or not module for module in model.source_modules
        ):
            raise ValueError("source_modules must be a tuple of importable module names")
        if model.node_labels is not None and (
            not isinstance(model.node_labels, Mapping)
            or any(
                not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in model.node_labels.items()
            )
        ):
            raise ValueError("node_labels must map nonempty strings to exact label values")
        labels = dict(model.node_labels or {})
        if labels and execution != "cluster":
            raise ValueError("node_labels require an explicitly supplied cluster")
        if model.config is None:
            raise ValueError("StudyModel requires config")
        raw, spec, _, _, controls = _config(
            pathlib.Path(model.config), require_hpo=model.fixed_parameters is None
        )
        callback, code_sha = _entrypoint(model.entrypoint)
        implementation = resolve_implementation(
            model.implementation, json.loads(study._implementation_scope)
            if model.implementation is None and getattr(study, "_implementation_scope", None)
            else None, callback, spec.model_id,
        )
        identity = _implementation_record(implementation)
        source_record = None
        if isinstance(implementation, CapturedImplementation):
            source_root = _output_root(study, output_dir, "study-campaigns") / "sources" / (
                identity["source_sha256"] + "-" + implementation.binding["evidence_sha256"]
            )
            source_root.mkdir(parents=True, exist_ok=True)
            implementation.retain(source_root)
            verify_retained(source_root, identity, implementation.binding)
            source_record = {"path": str(source_root), "identity": identity,
                             "binding": implementation.binding,
                             "scope": json.loads(implementation.scope_bytes)}
        # Fixed candidates participate in the same study-wide admission but do not search.
        selection_controls = controls
        if model.fixed_parameters is not None:
            from autoforecast.tuning import materialize_manual_parameters

            materialize_manual_parameters(spec, dict(model.fixed_parameters))
            selection_controls = _validation_controls(controls)
        options.append(_resolve_options(study, selection_controls, None))
        resolved.append(
            {
                "id": spec.model_id,
                "entrypoint": model.entrypoint,
                "code_sha256": code_sha,
                "implementation": identity,
                "implementation_source": source_record,
                "profile": loads_exact_yaml(raw),
                "original_profile_sha256": hashlib.sha256(raw).hexdigest(),
                "original_profile_text": raw.decode("utf-8"),
                "fixed_parameters": None
                if model.fixed_parameters is None
                else dict(model.fixed_parameters),
                "cpus": model.cpus,
                "memory_gib": model.memory_gib,
                "gpus": model.gpus,
                "source_modules": {name: _module_hash(name) for name in model.source_modules},
                "node_labels": labels,
            }
        )
    if len({model["id"] for model in resolved}) != len(resolved):
        raise ValueError("each candidate needs a distinct profile model ID")
    if (
        len({option["metric"] for option in options}) != 1
        or len({option["series_policy"] for option in options}) != 1
    ):
        raise ValueError("all candidates require the same comparison metric and series policy")
    searches = [
        option
        for model, option in zip(resolved, options, strict=True)
        if model["fixed_parameters"] is None
    ]
    if searches and any(
        option["selection_boundaries"] != searches[0]["selection_boundaries"] for option in searches
    ):
        raise ValueError("tuned candidates must use the same selection windows")
    common = dict(searches[0] if searches else options[0])
    common["minimum_history"] = max(option["minimum_history"] for option in options)
    root = _output_root(study, output_dir, "study-campaigns")
    prepared = preflight_study(study, common, output_dir=root / "preflight")
    for model in resolved:
        model["profile"]["minimum_history"] = common["minimum_history"]
    request = {
        "schema": "autoforecast.study-campaign.v1",
        "name": name,
        "study": {"path": str(prepared.study_path.resolve()), "sha256": prepared.study_sha256},
        "models": resolved,
        "metric": common["metric"],
        "execution": execution,
        "max_parallel_models": max_parallel_models,
        "final_evaluation": final_evaluation,
        "ray_address": ray_address,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "runtime": _runtime_identity(),
    }
    root.mkdir(parents=True, exist_ok=True)
    request_dir = root / ("request-" + digest(request)[:20])
    request_dir.mkdir(exist_ok=True)
    request_path = request_dir / "request.json"
    if request_path.exists():
        if load_exact_json(request_path) != request:
            raise ValueError("existing request differs")
    else:
        _write_new_json(request_path, request)
    request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    resources = {
        "device": "cpu",
        "memory_gib": 2
        if execution == "cluster"
        else sum(
            sorted((model["memory_gib"] for model in resolved), reverse=True)[:max_parallel_models]
        ) + 2,
        "min_free_disk_gib": 0.0,
    }
    stages = []
    for phase, needs in (("validation", []), ("final", ["validation"])):
        stages.append(
            {
                "id": phase,
                "needs": needs,
                "command": [
                    sys.executable,
                    "-m",
                    "autoforecast.study_campaign",
                    str(request_path),
                    request_sha,
                    phase,
                    "{campaign_root}",
                    "{attempt_dir}",
                ],
                "timeout_seconds": timeout_seconds,
                "max_attempts": max_attempts,
                "retry_mode": "restart",
                "required_outputs": ["summary.json"],
                "resources": resources,
            }
        )
    plan = {"schema_version": 1, "campaign": {"name": name, "stages": stages}}
    plan_path = request_dir / "plan.json"
    if plan_path.exists():
        if load_exact_json(plan_path) != plan:
            raise ValueError("campaign execution plan differs from the original request")
    else:
        _write_new_json(plan_path, plan)
    store = CampaignStore.open_or_create(
        plan_path, output_root=root, working_directory=pathlib.Path.cwd()
    )
    profiles = store.root / "profiles"
    if profiles.is_symlink():
        raise ValueError("retained profile directory may not be a symlink")
    profiles.mkdir(exist_ok=True)
    for index, model in enumerate(resolved):
        original = model["original_profile_text"].encode("utf-8")
        source = profiles / f"candidate-{index}.yaml"
        if source.is_symlink():
            raise ValueError("retained original profile may not be a symlink")
        if not source.exists():
            publish_bytes(profiles, source.name, original)
        if read_regular_bytes(profiles, source.name) != original:
            raise ValueError("retained original profile differs from campaign request")
    state = CampaignSupervisor(store).run()
    summary: dict = {"campaign_path": str(store.root), "state": state}
    summary["phases"] = {}
    for phase in ("validation", "final"):
        reports = sorted((store.root / "stages" / phase).glob("*/summary.json"))
        if reports:
            summary["phases"][phase] = load_exact_json(reports[-1])
    final_state = next(row for row in cast(list[dict], state["stages"]) if row["id"] == "final")
    if final_state["status"] == "completed":
        validation_records = _validation_results(store.root, request)
        # Stage outputs were authenticated by CampaignSupervisor before this read.
        attempts = sorted((store.root / "stages/final").glob("*/summary.json"))
        summary["results"] = cast(dict, load_exact_json(attempts[-1]))
        for key, recorded in summary["results"]["results"].items():
            retained = _load_model_result(
                store.root / "models" / f"candidate-{key}" / "final/result.json",
                request,
                resolved[int(key)],
                "final",
                validation_record=validation_records[key],
            )
            if retained != recorded:
                raise ValueError("model result differs from the completed campaign stage")
        if tracking_uri is not None:
            assert experiment_id is not None
            from autoforecast.run_mlflow import export_forecast_workflow

            summary["mlflow_links"] = {
                key: {
                    stage: link.as_record()
                    for stage, link in export_forecast_workflow(
                        value["workflow_path"],
                        tracking_uri=tracking_uri,
                        experiment_id=experiment_id,
                    ).items()
                }
                for key, value in summary["results"]["results"].items()
            }
    from mlops.run_report import run_guide

    from autoforecast.protocol import _atomic_write_text

    _atomic_write_text(store.root / "RUN.md", run_guide(
        identity={"campaign_id": store.root.name, "name": name},
        status=str(state["status"]),
        files=[p.relative_to(store.root).as_posix() for p in store.root.rglob("*") if p.is_file()],
        sealed=False,
    ).decode("utf-8"))
    return summary


def _load_model_result(
    path: pathlib.Path, request: dict, model: dict, phase: str,
    *, validation_record: dict | None = None,
) -> dict:
    record = cast(dict, load_exact_json(path))
    if (
        record["request_sha256"] != digest(request)
        or record["model_id"] != model["id"]
        or record["phase"] != phase
    ):
        raise ValueError("model result belongs to a different campaign task")
    workflow = load_forecast_workflow(record["workflow_path"])
    if (
        workflow.provenance["study"] != request["study"]
        or workflow.provenance["implementation"] != model["implementation"]
    ):
        raise ValueError("model workflow identity differs from campaign")
    if (workflow.testing is None) != (phase == "validation"):
        raise ValueError("model result has the wrong execution phase")
    if phase == "final" and (
        validation_record is None
        or workflow.provenance.get("validation_run") != {
            "path": validation_record["workflow_path"],
            "manifest_sha256": validation_record["workflow_sha256"],
        }
    ):
        raise ValueError("final model differs from the completed validation checkpoint")
    if loads_exact_yaml(workflow.search_config.read_bytes()) != model["profile"]:
        raise ValueError("model workflow uses a different campaign profile")
    if model["fixed_parameters"] is not None:
        from autoforecast.tuning import loads_external_tuning_spec, materialize_manual_parameters

        profile = dict(model["profile"])
        for key in ("hpo", "selection", "reporting", "series_policy", "minimum_history"):
            profile.pop(key, None)
        spec = loads_external_tuning_spec(
            json.dumps(profile).encode(), source_path=workflow.search_config
        )
        grouped = materialize_manual_parameters(spec, model["fixed_parameters"])
        expected = {
            f"{group}.{key}": value
            for group, values in grouped.items()
            for key, value in values.items()
        }
        if workflow.parameters != expected:
            raise ValueError("model workflow uses different fixed parameters")
    if (
        hashlib.sha256((workflow.path / "artifact_manifest.json").read_bytes()).hexdigest()
        != record["workflow_sha256"]
    ):
        raise ValueError("model workflow artifact identity changed")
    return record


def _validation_results(root: pathlib.Path, request: dict) -> dict:
    """Bind every candidate to the completed validation phase, including losers."""
    store = CampaignStore.from_root(root)
    stage = next(
        row for row in cast(list[dict], store.state()["stages"]) if row["id"] == "validation"
    )
    if stage["status"] != "completed":
        raise ValueError("model comparison requires a completed validation phase")
    summary = cast(dict, load_exact_json(
        root / "stages/validation" / f"attempt-{stage['attempts']:04d}" / "summary.json"
    ))
    expected_keys = {str(index) for index in range(len(request["models"]))}
    if (
        summary["status"] != "completed" or summary["failures"]
        or set(summary["results"]) != expected_keys
    ):
        raise ValueError("validation phase does not contain every requested candidate")
    for index, model in enumerate(request["models"]):
        record = _load_model_result(
            root / "models" / f"candidate-{index}" / "validation/result.json",
            request, model, "validation",
        )
        if record != summary["results"][str(index)]:
            raise ValueError("validation record differs from the completed campaign phase")
    return summary["results"]


def _watch_memory(stop: threading.Event, folder: pathlib.Path, limit_gib: float) -> None:
    """Stop a job when its process tree exceeds the limit or cannot be measured."""
    import signal

    from autoforecast.campaign import _process_tree_rss

    while not stop.wait(1):
        try:
            exceeded = _process_tree_rss(os.getpid()) > limit_gib * 1024**3
            reason = "process tree exceeded its memory reservation" if exceeded else None
        except Exception as error:
            reason = f"memory measurement failed: {type(error).__name__}: {error}"
        if reason is not None:
            try:
                _write_new_json(
                    folder / f"memory-limit-{time.time_ns()}.json",
                    {"limit_gib": limit_gib, "pid": os.getpid(), "reason": reason},
                )
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
            return


def _model_job(
    request_path: str, request_sha: str, index: int, phase: str, campaign_root: str,
    validation_record: dict | None, parent_attempt: str | None = None,
) -> dict:
    """Worker boundary: inert references in, authenticated retained result out."""
    request = _read(request_path, request_sha)
    model = request["models"][index]
    if _runtime_identity() != request["runtime"]:
        raise ValueError("worker library or numerical runtime differs from the campaign")
    implementation = (worker_implementation(model["implementation_source"])
                      if model.get("implementation_source") else None)
    if any(_module_hash(name) != expected
           for name, expected in model["source_modules"].items()):
        raise ValueError("model helper code differs on worker")
    callback, _ = _entrypoint(model["entrypoint"], model["code_sha256"])
    implementation = (resolve_implementation(implementation, None, callback, model["id"])
                      if implementation is not None else model["implementation"])
    for package, version in model["implementation"]["dependencies"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"worker dependency differs: {package}")
    path = pathlib.Path(request["study"]["path"])
    verify_benchmark(path, request["study"]["sha256"])
    if _console() is None:
        configure_logging(progress="plain")
    # Workers receive paths and IDs, then establish their own logging context.
    with standalone_attempt(pathlib.Path(campaign_root) / "progress", "Model job",
                            campaign_id=pathlib.Path(campaign_root).name,
                            request_sha256=request_sha, model_variant=model["id"],
                            parent_attempt=parent_attempt,
                            candidate_index=index, phase=phase, separate=True):
        note("Configured model", model_variant=model["id"], entrypoint=model["entrypoint"],
             phase=phase, model_parameters=model["fixed_parameters"])
        folder = pathlib.Path(campaign_root) / "models" / f"candidate-{index}" / phase
        folder.mkdir(parents=True, exist_ok=True)
        completion = folder / "result.json"
        if completion.exists():
            retained = _load_model_result(
                completion, request, model, phase, validation_record=validation_record
            )
            note("Model result reused and verified", workflow_path=retained["workflow_path"])
            record_result(study_id=None, workflow_path=retained["workflow_path"])
            return retained
        import fcntl

        from autoforecast.campaign import _open_lock

        with _open_lock(folder / ".job.lock") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if completion.exists():
                retained = _load_model_result(
                    completion, request, model, phase, validation_record=validation_record
                )
                note("Model result reused and verified", workflow_path=retained["workflow_path"])
                record_result(study_id=None, workflow_path=retained["workflow_path"])
                return retained
            # Each fit owns a new process/session; no caller Spark object crosses here.
            from pyspark.sql import SparkSession

            from autoforecast.study import load_study

            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            stop_watch = threading.Event()

            watcher = threading.Thread(
                target=_watch_memory, args=(stop_watch, folder, model["memory_gib"]), daemon=True
            )

            spark = None
            started = time.time()
            watcher.start()
            try:
                spark = (
                    SparkSession.builder.master(f"local[{model['cpus']}]")
                    .appName("autoforecast-model-job")
                    .config("spark.ui.enabled", "false")
                    .config("spark.sql.shuffle.partitions", str(model["cpus"]))
                    .config("spark.driver.memory", f"{max(512, int(model['memory_gib'] * 512))}m")
                    .getOrCreate()
                )
                config_path = folder / "profile.json"
                if not config_path.exists():
                    _write_new_json(config_path, model["profile"])
                if load_exact_json(config_path) != model["profile"]:
                    raise ValueError("worker profile differs from frozen request")
                with load_study(
                    path.parent.name, registry_root=path.parent.parent, spark=spark
                ) as study:
                    if phase == "validation":
                        run = study.run_forecast_workflow(
                            fit_predict=callback,
                            implementation=implementation,
                            config=config_path,
                            fixed_parameters=model["fixed_parameters"],
                            validation_only=True,
                            output_dir=folder / "runs",
                        )
                    else:
                        validation_path = folder.parent / "validation/result.json"
                        previous = _load_model_result(validation_path, request, model, "validation")
                        if previous != validation_record:
                            raise ValueError(
                                "worker validation input differs from its phase handoff")
                        run = study.run_forecast_workflow(
                            fit_predict=callback,
                            implementation=implementation,
                            validation_run=previous["workflow_path"],
                            output_dir=folder / "runs",
                        )
                record_result(study_id=run.study_id, workflow_id=run.run_id,
                              workflow_path=str(run.path))
                record = {
                    "request_sha256": digest(request),
                    "model_id": model["id"],
                    "phase": phase,
                    "workflow_path": str(run.path),
                    "workflow_sha256": hashlib.sha256(
                        (run.path / "artifact_manifest.json").read_bytes()
                    ).hexdigest(),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started": started,
                    "finished": time.time(),
                }
                _write_new_json(completion, record)
                return _load_model_result(
                    completion, request, model, phase, validation_record=validation_record
                )
            finally:
                stop_watch.set()
                watcher.join(timeout=2)
                if spark is not None:
                    spark.stop()


def _run_phase(
    request_path: str, request_sha: str, phase: str, campaign_root: str, attempt: str
) -> None:
    request = _read(request_path, request_sha)
    candidates = list(range(len(request["models"])))
    selected = None
    validation_records = {}
    if phase == "final":
        validation_records = _validation_results(pathlib.Path(campaign_root), request)
        comparisons = []
        contracts = set()
        for index in candidates:
            record = validation_records[str(index)]
            run = load_forecast_workflow(record["workflow_path"])
            contracts.add(digest(run.validation["evaluation_contract"]))
            value = run.validation["score"].get(request["metric"])
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("candidate validation objective is unavailable")
            comparisons.append((value, request["models"][index]["id"], index))
        if len(contracts) != 1:
            raise ValueError("candidate validation contracts differ; refusing comparison")
        selected = min(comparisons)[2]
        if request["final_evaluation"] == "validation_winner":
            candidates = [selected]
    audit = pathlib.Path(attempt) / "progress.jsonl"
    names = {i: f"Model {i + 1}: {request['models'][i]['id']}" for i in candidates}
    with Execution(audit, [*names.values(), "Verify campaign results"],
                   level=resolve_level(), attempt_id=pathlib.Path(attempt).name,
                   campaign_id=pathlib.Path(campaign_root).name,
                   request_sha256=request_sha, phase=phase):
        results, failures = {}, {}
        for index in candidates:
            note("Queued model", model_variant=request["models"][index]["id"],
                 candidate_index=index, phase=phase)
        if request["execution"] == "sequential":
            for index in candidates:
                _stage_event(audit, names[index], "START")
                try:
                    results[str(index)] = _model_job(
                        request_path, request_sha, index, phase, campaign_root,
                        validation_records.get(str(index)), parent_attempt=attempt,
                    )
                except Exception as error:
                    failures[str(index)] = str(error)
                    # Keep the model failure even if its audit append cannot be verified.
                    with suppress(BaseException):
                        _stage_event(audit, names[index], "FAIL",
                                     error_type=type(error).__name__)
                else:
                    _stage_event(audit, names[index], "PASS")
        else:
            import ray

            if ray.is_initialized():
                raise ValueError("campaign phase must own its Ray connection")
            if request["execution"] == "cluster":
                ray.init(address=request["ray_address"])
            else:
                cpus = min(
                    os.cpu_count() or 1,
                    sum(
                        sorted((m["cpus"] for m in request["models"]), reverse=True)[
                            : request["max_parallel_models"]
                        ]
                    ),
                )
                ray.init(address="local", num_cpus=cpus, include_dashboard=False)
            pending = {}
            waiting = list(candidates)
            deadline = time.monotonic() + request["timeout_seconds"]
            try:
                nodes = [node for node in ray.nodes() if node["Alive"]]
                for index in candidates:
                    model = request["models"][index]
                    if not any(
                        model["cpus"] <= node["Resources"].get("CPU", 0)
                        and model["gpus"] <= node["Resources"].get("GPU", 0)
                        and model["memory_gib"] * 1024**3 <= node["Resources"].get("memory", 0)
                        and all(
                            node.get("Labels", {}).get(key) == value
                            for key, value in model.get("node_labels", {}).items()
                        )
                        for node in nodes
                    ):
                        raise ValueError(
                            "no worker node can satisfy the model resources and exact node labels"
                        )
                # A phase can run as __main__. Import the worker by its canonical name
                # so Ray resolves the installed module on each host instead of copying
                # __main__ globals, including the coordinator's installation path.
                from autoforecast.study_campaign import _model_job as remote_model_job

                # Ray's pinned typing overloads stop at six positional inputs; the native
                # runtime also accepts our optional, inert parent-attempt keyword.
                worker = cast(Any, ray.remote(max_calls=1, max_retries=0)(remote_model_job))
                while waiting or pending:
                    while waiting and len(pending) < request["max_parallel_models"]:
                        index = waiting.pop(0)
                        model = request["models"][index]
                        placement = {}
                        if model.get("node_labels"):
                            from ray.util.scheduling_strategies import (
                                In,
                                NodeLabelSchedulingStrategy,
                            )

                            # Keep caller labels literal; selector strings parse operators.
                            placement["scheduling_strategy"] = NodeLabelSchedulingStrategy(
                                hard={key: In(value) for key, value in model["node_labels"].items()}
                            )
                        ref = worker.options(
                            **placement,
                            num_cpus=model["cpus"],
                            num_gpus=model["gpus"],
                            memory=int(model["memory_gib"] * 1024**3),
                            runtime_env={
                                "env_vars": {
                                    "OMP_NUM_THREADS": "1",
                                    "OPENBLAS_NUM_THREADS": "1",
                                    "AUTOFORECAST_LOG_LEVEL": os.environ.get(
                                        "AUTOFORECAST_LOG_LEVEL", "INFO"),
                                }
                            },
                        ).remote(
                            request_path, request_sha, index, phase, campaign_root,
                            validation_records.get(str(index)), parent_attempt=attempt,
                        )
                        pending[ref] = index
                        _stage_event(audit, names[index], "START")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("campaign model jobs exceeded the phase deadline")
                    ready, _ = ray.wait(list(pending), num_returns=1, timeout=min(remaining, 1))
                    for ref in ready:
                        index = pending.pop(ref)
                        try:
                            results[str(index)] = ray.get(ref)
                        except Exception as error:
                            failures[str(index)] = str(error)
                            with suppress(BaseException):
                                _stage_event(audit, names[index], "FAIL",
                                             error_type=type(error).__name__)
                        else:
                            _stage_event(audit, names[index], "PASS")
            finally:
                for ref in pending:
                    ray.cancel(ref, force=True)
                ray.shutdown()
        with _stage(audit, "Verify campaign results"):
            for key, record in results.items():
                retained = _load_model_result(
                    (pathlib.Path(campaign_root) / "models"
                     / f"candidate-{key}" / phase / "result.json"),
                    request,
                    request["models"][int(key)],
                    phase,
                    validation_record=validation_records.get(key),
                )
                if record != retained:
                    raise ValueError("worker returned a different model result")
            candidate_summary = []
            for index, model in enumerate(request["models"]):
                validation_file = (
                    pathlib.Path(campaign_root) / "models"
                    / f"candidate-{index}" / "validation/result.json"
                )
                row = {
                    "model_id": model["id"],
                    "status": "failed" if str(index) in failures else "not_evaluated",
                }
                if validation_file.exists():
                    record = (
                        validation_records[str(index)] if phase == "final"
                        else _load_model_result(validation_file, request, model, "validation")
                    )
                    validation = load_forecast_workflow(record["workflow_path"])
                    row.update(validation=validation.validation["score"],
                               parameters=validation.parameters)
                    if str(index) not in failures:
                        row["status"] = "validated"
                if phase == "final" and str(index) in results:
                    completed = load_forecast_workflow(results[str(index)]["workflow_path"])
                    assert completed.testing is not None
                    row.update(testing=completed.testing["score"], status="completed")
                candidate_summary.append(row)
            summary = {
                "phase": phase,
                "requested_models": [m["id"] for m in request["models"]],
                "selected_model": (request["models"][selected]["id"]
                                   if selected is not None else None),
                "final_evaluation": request["final_evaluation"],
                "results": results,
                "failures": failures,
                "candidates": candidate_summary,
                "status": "failed" if failures else "completed",
            }
            _write_new_json(pathlib.Path(attempt) / "summary.json", summary)
            if failures:
                raise RuntimeError("one or more campaign models failed; see retained summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("sha256")
    parser.add_argument("phase", choices=("validation", "final"))
    parser.add_argument("campaign_root")
    parser.add_argument("attempt")
    args = parser.parse_args()
    configure_logging(progress="plain")
    import signal

    def cancel(_signal, _frame):
        raise KeyboardInterrupt("campaign phase cancelled")

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        _run_phase(args.request, args.sha256, args.phase, args.campaign_root, args.attempt)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
