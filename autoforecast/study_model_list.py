"""Public model lists using the existing study campaign and verified workflow records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import cast

from mlops._safe_files import publish_bytes, read_regular_bytes

from autoforecast._exact_yaml import loads_exact_yaml
from autoforecast._execution import _stage, _stage_event, record_result
from autoforecast.benchmark import digest
from autoforecast.campaign import CampaignStore
from autoforecast.implementation import CapturedImplementation
from autoforecast.study_builtins import BUILTIN_STUDY_MODELS, builtin_profile, builtin_study_model
from autoforecast.study_campaign import (
    StudyModel,
    _entrypoint,
    _load_model_result,
    _read,
    _select_validation_candidate,
    _validation_results,
    validate_model_resources,
)
from autoforecast.study_tuning import (
    _config,
    _implementation_record,
    _output_root,
    _validate_controls,
    _validation_controls,
)
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.tuning import materialize_manual_parameters


@dataclasses.dataclass(frozen=True)
class StudyCampaignResult:
    """One campaign, with distinct retained workflows and publication state per candidate.

    study_id is the effective comparison study; requested_study_id is the registered input.
    Failed training leaves verified completed peers available in candidates. Publication failures
    raise with campaign_path, so the same request can retry publication without refitting.
    """

    path: pathlib.Path
    requested_study_id: str
    study_id: str
    status: str
    candidates: tuple[dict, ...]
    selected_model: str | None
    final_evaluation: str
    state: dict
    publication: dict = dataclasses.field(default_factory=dict)
    tracking: dict = dataclasses.field(default_factory=dict)
    data_outputs: object = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def campaign_id(self) -> str:
        return self.path.name

    def summary(self) -> dict:
        """Small result summary; forecasts and fitted state remain in their saved artifacts."""
        return {"campaign_id": self.campaign_id, "path": str(self.path),
                "requested_study_id": self.requested_study_id, "study_id": self.study_id,
                "status": self.status, "selected_model": self.selected_model,
                "final_evaluation": self.final_evaluation, "candidates": list(self.candidates),
                "publication": self.publication, "tracking": self.tracking}

    def outputs(self, section="all"):
        """Combine verified candidate training output maps without owning open study resources."""
        from autoforecast.outputs import OutputsMap, workflow_outputs

        result = self.data_outputs if isinstance(self.data_outputs, OutputsMap) else OutputsMap()
        for candidate in self.candidates:
            if candidate.get("workflow_path"):
                result = result.merge(workflow_outputs(
                    load_forecast_workflow(candidate["workflow_path"]),
                ))
        return result.select(section)


def prepare_candidates(models, options, execution) -> list[dict]:
    """Validate declarations and snapshot caller profiles before registration or fitting."""
    if (not isinstance(models, Sequence) or isinstance(models, str | bytes) or not models):
        raise ValueError(
            "models must be a nonempty ordered sequence of names or StudyModel entries")
    if not isinstance(options, Mapping):
        raise ValueError("model_options must be a mapping keyed by built-in names")
    names = {model for model in models if isinstance(model, str)}
    if set(options) - names:
        raise ValueError("model_options contains a name absent from models")
    mode = execution.get("execution", "sequential")
    if mode not in {"sequential", "local", "cluster"}:
        raise ValueError("execution must be sequential, local or cluster")
    if (mode == "cluster") != (execution.get("ray_address") is not None):
        raise ValueError("only cluster execution requires an explicit ray_address")
    if mode == "cluster" and (not isinstance(execution["ray_address"], str)
                              or not execution["ray_address"].strip()):
        raise ValueError("ray_address must be a nonempty string")
    for key, default in (("max_parallel_models", 1), ("max_attempts", 1),
                         ("timeout_seconds", 3600)):
        value = execution.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if mode == "sequential" and execution.get("max_parallel_models", 1) != 1:
        raise ValueError("sequential execution requires max_parallel_models=1")
    if execution.get("final_evaluation", "all_models") not in {"all_models", "validation_winner"}:
        raise ValueError("final_evaluation must be all_models or validation_winner")
    prepared = []
    for model in models:
        if isinstance(model, str):
            if model not in BUILTIN_STUDY_MODELS:
                raise ValueError(f"unsupported study model: {model}")
            settings = options.get(model, {})
            allowed = {"model_config", "fixed_parameters", "cpus", "memory_gib",
                       "gpus", "node_labels"}
            if not isinstance(settings, Mapping) or set(settings) - allowed:
                raise ValueError(f"invalid model_options for {model}")
            settings = dict(settings)
            callback, implementation = builtin_study_model(model)
            config = settings.pop("model_config", None)
            infer_fixed = "fixed_parameters" not in settings
            declaration = StudyModel(
                f"{callback.__module__}:{callback.__name__}", implementation, config, **settings,
            )
            candidate_id = model
        elif type(model) is StudyModel:
            declaration = dataclasses.replace(model)
            _entrypoint(declaration.entrypoint)
            infer_fixed = False
            candidate_id = None
            if declaration.config is None:
                raise ValueError("StudyModel requires config")
        else:
            raise ValueError("models accepts built-in names or importable StudyModel declarations")
        validate_model_resources(declaration, mode)
        raw = None
        fixed = declaration.fixed_parameters
        if declaration.config is not None:
            raw, spec, _, _, controls = _config(
                pathlib.Path(declaration.config), require_hpo=not infer_fixed and fixed is None,
            )
            if candidate_id is not None and spec.model_id != candidate_id:
                raise ValueError("a named built-in profile must use the same model ID")
            candidate_id = spec.model_id
            if infer_fixed:
                fixed = None if "hpo" in cast(dict, loads_exact_yaml(raw)) else {}
            if fixed is not None:
                if not isinstance(fixed, Mapping):
                    raise ValueError("fixed_parameters must be a mapping or None")
                materialize_manual_parameters(spec, dict(fixed))
                controls = _validation_controls(controls)
            _validate_controls(controls)
        elif infer_fixed:
            fixed = {}
        elif fixed is None:
            raise ValueError("parameter search requires an hpo mapping in model_config")
        if fixed is not None:
            if not isinstance(fixed, Mapping):
                raise ValueError("fixed_parameters must be a mapping or None")
            fixed = json.loads(json.dumps(dict(fixed), allow_nan=False))
        implementation = declaration.implementation
        if implementation is not None:
            identity = _implementation_record(implementation)
            if not isinstance(implementation, CapturedImplementation):
                implementation = identity
        declaration = dataclasses.replace(
            declaration, fixed_parameters=fixed,
            node_labels=dict(declaration.node_labels) if declaration.node_labels else None,
            implementation=implementation,
        )
        prepared.append({"id": candidate_id, "builtin": model if isinstance(model, str) else None,
                         "declaration": declaration, "profile": raw})
    if len({candidate["id"] for candidate in prepared}) != len(prepared):
        raise ValueError("each candidate needs a distinct profile model ID")
    return prepared


def _retain_profile(root, raw):
    folder = root / "candidate-profiles"
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(raw).hexdigest() + ".json"
    path = folder / name
    if not path.exists():
        publish_bytes(folder, name, raw)
    if read_regular_bytes(folder, name) != raw:
        raise ValueError("retained candidate profile changed")
    return path


def _campaign_request(store):
    """Read the request pinned by every authenticated phase command; never execute it."""
    references = set()
    for stage in store.plan.stages:
        command = stage.command
        if len(command) != 8 or command[1:3] != ("-m", "autoforecast.study_campaign"):
            raise ValueError("not a study campaign plan")
        references.add((command[3], command[4]))
    if len(references) != 1:
        raise ValueError("campaign phases refer to different requests")
    path, sha = references.pop()
    return _read(path, sha)


def load_study_campaign(path) -> StudyCampaignResult:
    """Reload a campaign and authenticate retained candidate workflows, including failed peers.

    Native inputs need worker connection aliases available in AUTOFORECAST_CONFIG, as for
    load_forecast_workflow. No training or publication takes place during reload.
    """
    from autoforecast.snapshots import input_scope

    with input_scope():
        return _load_campaign(path)


def _load_campaign(path):
    store = CampaignStore.from_root(path)
    request = _campaign_request(store)
    state = store.state()
    stages = {stage["id"]: stage for stage in cast(list[dict], state["stages"])}
    selected = None
    if stages.get("validation", {}).get("status") == "completed":
        selected = _select_validation_candidate(request, _validation_results(store.root, request))
    selected_id = request["models"][selected]["id"] if selected is not None else None
    final = next(stage for stage in cast(list[dict], state["stages"]) if stage["id"] == "final")
    summary = None
    if final["status"] == "completed":
        summary = json.loads(read_regular_bytes(
            store.root / "stages/final" / f"attempt-{final['attempts']:04d}", "summary.json",
        ))
        if summary["selected_model"] != selected_id:
            raise ValueError("campaign winner differs from verified validation")
    candidates = []
    pending_actuals = False
    for index, model in enumerate(request["models"]):
        row = {"model_id": model["id"], "status": "not_evaluated"}
        validation = None
        for phase in ("validation", "final"):
            record_path = store.root / "models" / f"candidate-{index}" / phase / "result.json"
            if not record_path.exists():
                continue
            record = _load_model_result(record_path, request, model, phase,
                                        validation_record=validation)
            if phase == "validation":
                validation = record
            workflow = load_forecast_workflow(record["workflow_path"])
            if phase == "final" and summary is not None:
                if summary["results"].get(str(index)) != record:
                    raise ValueError("final summary differs from retained model result")
            pending_actuals |= workflow.status == "pending_actuals"
            row.update(status="validated" if phase == "validation" else workflow.status,
                       run_id=workflow.run_id, workflow_path=str(workflow.path),
                       validation_metrics=workflow.validation_metrics,
                       testing_metrics=workflow.testing_metrics, coverage=workflow.coverage,
                       model_links=workflow.model_links, host=record["host"], pid=record["pid"])
        if (request["final_evaluation"] == "validation_winner"
                and selected is not None and index != selected):
            row["status"] = "not_final_selected"
        elif state["status"] != "completed" and row["status"] in {"not_evaluated", "validated"}:
            # A terminated phase may never write a model-specific result. Its journal is
            # authoritative; an unsealed summary is only a diagnostic, not completion proof.
            failures = [event for event in store.events() if event["kind"] == "stage_failed"]
            if failures:
                row.update(status="incomplete", error=failures[-1]["reason"])
        candidates.append(row)
    if summary is not None:
        expected = (set(map(str, range(len(candidates))))
                    if request["final_evaluation"] == "all_models"
                    else {str(selected)})
        if set(summary["results"]) != expected:
            raise ValueError("campaign final results omit a required candidate")
    effective = pathlib.Path(request["study"]["path"]).parent.name
    registration = request.get("registration", {})
    publication, tracking = {}, {}
    for index, row in enumerate(candidates):
        folder = store.root / "publication" / str(index)
        if (folder / "result.json").exists():
            receipt = json.loads(read_regular_bytes(folder, "result.json"))
            if (receipt["request_sha256"] != digest(request)
                    or receipt["workflow_path"] != row.get("workflow_path")):
                raise ValueError("campaign publication receipt belongs to a different workflow")
            workflow = load_forecast_workflow(receipt["workflow_path"])
            if receipt["workflow_sha256"] != hashlib.sha256(
                (workflow.path / "artifact_manifest.json").read_bytes(),
            ).hexdigest():
                raise ValueError("campaign publication artifact identity changed")
            publication[row["model_id"]] = receipt["publication"]
            tracking[row["model_id"]] = receipt["tracking"]
    status = str(state["status"])
    if status == "completed":
        public = request.get("public_settings", {})
        expected_receipts = sum(c["status"] != "not_final_selected" for c in candidates)
        if (public.get("publish") or public.get("tracking") == "mlflow") and (
            len(publication) != expected_receipts
        ):
            status = "publication_pending"
        elif pending_actuals:
            status = "pending_actuals"
    return StudyCampaignResult(
        store.root, registration.get("study_id", effective), effective, status,
        tuple(candidates), selected_id,
        request["final_evaluation"], state, publication, tracking,
    )


def run_registered_campaign(study, *, registry, candidates, options, settings, experiment_id,
                            audit, output_dir, data_outputs):
    """Bridge the public registration lifecycle to one existing durable campaign."""
    from mlops._volume_files import is_volume

    from autoforecast.outputs import tracking_records
    from autoforecast.run_mlflow import export_forecast_workflow

    destination = pathlib.Path(output_dir or registry.root / "campaigns")
    if is_volume(destination):
        raise ValueError("study campaigns require POSIX shared storage with cross-process locks; "
                         "/Volumes campaign output is not supported")
    root = _output_root(study, destination, "study-campaigns")
    declarations = []
    for candidate in candidates:
        raw = candidate["profile"]
        if raw is None:
            raw = json.dumps(builtin_profile(candidate["builtin"], study.panel)).encode()
        profile = cast(dict, loads_exact_yaml(raw))
        if candidate["builtin"] in {"croston", "auto_ets", "auto_theta"}:
            from autoforecast.adapters.classical import MODELS

            parameters = candidate["declaration"].fixed_parameters
            season = ((parameters or {}).get("model.season_length")
                      or profile["hyperparameters"]["model.season_length"]["default"])
            # Seasonal searches can require more history than their default setting.
            domain = profile["hyperparameters"]["model.season_length"]["search"]
            if parameters is None:
                season = (max(domain["values"]) if domain["kind"] == "categorical"
                          else domain["high"])
            minimum = MODELS[candidate["builtin"]].min_history(season)
            profile["minimum_history"] = max(profile.get("minimum_history", 1), minimum)
        retained = _retain_profile(root, json.dumps(profile, sort_keys=True).encode())
        declarations.append(dataclasses.replace(candidate["declaration"], config=retained))
    binding = {"study_id": study.study_id, "sha256": study.registration["sha256"]}
    public = {"publish": settings["publish"], "tracking": settings["tracking"],
              "publication_destination": digest(dict(getattr(registry.store, "tables", {}))),
              "tracking_destination": digest(settings.get("tracking_uri", "")),
              "experiment_id": experiment_id,
              "file_checks": settings["file_checks"]}
    with _stage(audit, "workflow", study_id=study.study_id) as outcome:
        campaign = study.run_campaign(
            models=declarations, output_dir=root, _registration=binding,
            _public_settings=public, **options,
        )
        path = pathlib.Path(campaign["campaign_path"])
        result = _load_campaign(path)
        outcome.update(campaign_id=result.campaign_id, campaign_path=str(path),
                       campaign_status=result.status)
        record_result(study_id=result.study_id)
    request = _campaign_request(CampaignStore.from_root(path))
    try:
        if result.study_id != study.study_id:
            registry.retain_derived(study.study_id, pathlib.Path(request["study"]["path"]))
        if result.state["status"] != "completed":
            _stage_event(audit, "run", "FAIL", campaign_path=str(path))
            return result
        for index, candidate in enumerate(result.candidates):
            if candidate["status"] == "not_final_selected":
                continue
            workflow = load_forecast_workflow(candidate["workflow_path"])
            links = None
            if experiment_id is not None:
                with _stage(audit, "tracking_publication", run_id=workflow.run_id):
                    links = export_forecast_workflow(
                        workflow.path, tracking_uri=settings.get("tracking_uri", ""),
                        experiment_id=experiment_id, outputs=data_outputs,
                    )
            publication = {"status": "not_requested"}
            if settings["publish"]:
                with _stage(audit, "table_publication", run_id=workflow.run_id):
                    publication = registry.publish_results(
                        workflow.study_id, workflow.path, mlflow_links=links,
                    )
            receipt = {"request_sha256": digest(request), "workflow_path": str(workflow.path),
                       "workflow_sha256": hashlib.sha256(
                           (workflow.path / "artifact_manifest.json").read_bytes()).hexdigest(),
                       "publication": publication, "tracking": tracking_records(links)}
            folder = path / "publication" / str(index)
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "result.json").exists():
                publish_bytes(folder, "result.json", json.dumps(receipt, sort_keys=True).encode())
            else:
                existing = json.loads(read_regular_bytes(folder, "result.json"))
                # Export/write retries can return additional operational metadata. The frozen
                # workflow binding must remain exact; destination writers own idempotence.
                if any(existing[key] != receipt[key] for key in (
                    "request_sha256", "workflow_path", "workflow_sha256",
                )):
                    raise ValueError("publication receipt differs from completed workflow")
        result = dataclasses.replace(_load_campaign(path), data_outputs=data_outputs)
        if experiment_id is None:
            _stage_event(audit, "tracking_publication", "SKIP", reason="Not requested")
        if not settings["publish"]:
            _stage_event(audit, "table_publication", "SKIP", reason="Not requested")
        _stage_event(audit, "run", "UNAVAILABLE" if result.status == "pending_actuals" else "PASS",
                     campaign_path=str(path))
        return result
    except BaseException as error:
        error.__dict__.update(campaign_path=path, registered_study_id=study.study_id,
                              study_id=result.study_id)
        with suppress(Exception):
            error.__dict__["campaign_result"] = _load_campaign(path)
        raise
