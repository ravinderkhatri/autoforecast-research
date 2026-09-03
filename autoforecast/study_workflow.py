"""One external-model workflow using the existing tuner, scorer and run bundles."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
import tempfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING

from mlops._safe_files import SafeFileError, read_regular_bytes
from mlops.provenance import ProvenanceError
from mlops.run_bundle import load_completed_run_bundle, open_run_bundle
from mlops.run_report import new_run_id

from autoforecast.benchmark import digest
from autoforecast.hpo import HPOError
from autoforecast.hpo_bridge import _copy_run_artifact_directory
from autoforecast.study_preflight import preflight_study
from autoforecast.study_scoring import score_study
from autoforecast.study_tuning import (
    FitPredictRequest,
    FitPredictResult,
    _config,
    _fit_window,
    _implementation_record,
    _json_bytes,
    _output_root,
    _prepare_window,
    _resolve_options,
    _retained_json,
    _retained_manifest,
    _validate_workflow_configuration,
    _validation_controls,
    _window_result,
    load_selected_configuration,
    load_tuning_result,
    tune_study,
)
from autoforecast.tuning import (
    loads_external_tuning_spec,
    materialize_manual_parameters,
    materialize_runtime_parameters,
)
from provenance.lineage import directory_digest, snapshot_parquet_directory

if TYPE_CHECKING:
    from autoforecast.study import Study


@dataclasses.dataclass(frozen=True)
class ForecastWorkflowResult:
    """Retained models and scores; pending actuals are never a completed evaluation."""

    path: pathlib.Path
    status: str
    parameters: dict
    validation: dict
    testing: dict | None
    provenance: dict
    tuning_path: pathlib.Path | None
    selected_config: pathlib.Path
    search_config: pathlib.Path


def load_forecast_workflow(path: str | pathlib.Path) -> ForecastWorkflowResult:
    """Verify every workflow artifact before returning its saved stage results."""
    root = pathlib.Path(path).absolute()
    try:
        manifest_bytes = read_regular_bytes(root, "artifact_manifest.json")
    except SafeFileError as error:
        raise ProvenanceError(f"cannot read completed workflow manifest: {root}") from error
    completion = load_completed_run_bundle(root)
    manifest = _retained_manifest(completion, manifest_bytes)
    if (
        completion["task"] not in {"forecast-workflow", "forecast-validation"}
        or completion["status"] != "completed"
    ):
        raise HPOError("workflow did not finish successfully")
    record = _retained_json(root, "result.json", manifest)
    expected_schema = "autoforecast." + completion["task"] + ".v1"
    if record["schema"] != expected_schema:
        raise HPOError("unsupported forecast workflow result")
    _validate_workflow_configuration(
        root, record, _retained_json(root, "metadata/winner-config.json", manifest), manifest
    )
    stages = {"validation": _window_result(root, record["validation"])}
    testing = _window_result(root, record["testing"]) if "testing" in record else None
    return ForecastWorkflowResult(
        root,
        record["status"],
        record["parameters"],
        stages["validation"],
        testing,
        record["provenance"],
        pathlib.Path(record["tuning_path"]) if record["tuning_path"] else None,
        root / "metadata/winner-config.json",
        root / "metadata/search-config.yaml",
    )


def run_forecast_workflow(
    study: Study,
    *,
    fit_predict: Callable[[FitPredictRequest], FitPredictResult],
    implementation: Mapping[str, object],
    config: str | pathlib.Path | None = None,
    tuning_origin: str | None = None,
    output_dir: str | pathlib.Path | None = None,
    reuse_tuning: str | pathlib.Path | None = None,
    selected_config: str | pathlib.Path | None = None,
    fixed_parameters: Mapping[str, object] | None = None,
    validation_only: bool = False,
    validation_run: str | pathlib.Path | None = None,
) -> ForecastWorkflowResult:
    """Tune (or use saved settings), validate, refit and evaluate the final forecasts.

    ``selected_config`` skips HPO and permits new study dates/data. It requires the
    same declared implementation and valid parameters. ``reuse_tuning`` instead
    reopens an identical completed search and checks its original study identity.
    Caller Python is trusted. No model code is imported from configuration files.
    """
    from autoforecast.study import load_study

    if not callable(fit_predict):
        raise TypeError("fit_predict must be a Python callable")
    if type(validation_only) is not bool:
        raise TypeError("validation_only must be boolean")
    previous = None
    if validation_run is not None:
        if validation_only or any(
            value is not None
            for value in (
                config,
                selected_config,
                reuse_tuning,
                fixed_parameters,
                tuning_origin,
            )
        ):
            raise ValueError(
                "validation_run cannot combine with new configuration or validation_only"
            )
        previous = load_forecast_workflow(validation_run)
        if previous.testing is not None:
            raise HPOError("validation_run must be a validation checkpoint for this exact study")
        selected_config = previous.selected_config
    fixed = fixed_parameters is not None
    if fixed and (
        config is None
        or any(value is not None for value in (selected_config, reuse_tuning, tuning_origin))
    ):
        raise ValueError("fixed_parameters requires config and cannot combine with selection reuse")
    if fixed and not isinstance(fixed_parameters, Mapping):
        raise TypeError("fixed_parameters must be a mapping")
    identity = _implementation_record(implementation)
    if selected_config is not None and (reuse_tuning is not None or tuning_origin is not None):
        raise ValueError("selected_config conflicts with reuse_tuning and tuning_origin")
    saved = load_selected_configuration(selected_config) if selected_config is not None else None
    if saved is not None and saved["implementation"] != identity:
        raise HPOError("saved configuration uses a different model implementation")
    if config is not None:
        raw, spec, budget, seed, controls = _config(
            pathlib.Path(config), require_hpo=not (fixed or saved is not None)
        )
    elif saved is not None:
        assert selected_config is not None
        profile = dict(saved["profile"])
        controls = {
            key: profile.pop(key)
            for key in (
                "selection",
                "reporting",
                "series_policy",
                "minimum_history",
            )
            if key in profile
        }
        profile.pop("hpo", None)
        raw = _json_bytes(profile)
        spec = loads_external_tuning_spec(raw, source_path=pathlib.Path(selected_config))
        seed = saved["seed"]
    else:
        raise ValueError("provide config to tune, or selected_config to skip HPO")
    if fixed:
        assert fixed_parameters is not None
        manual = materialize_manual_parameters(spec, dict(fixed_parameters))
        controls = _validation_controls(controls)
    if saved is not None:
        if spec.model_id != saved["model_id"]:
            raise HPOError("saved configuration uses a different model identity")
        materialize_runtime_parameters(spec, saved["parameters"])
        seed = saved["seed"]
        controls = _validation_controls(controls)
    options = _resolve_options(study, controls, tuning_origin)
    if saved is not None or fixed:
        options["selection_boundaries"] = []
    if previous is not None:
        options = dict(previous.provenance["options"])
    root = _output_root(study, output_dir, "study-workflows")
    task = "forecast-validation" if validation_only else "forecast-workflow"
    bundle = open_run_bundle(root, run_id=new_run_id("workflow"), task=task)
    phase = "preflight"
    tuning_path = None
    try:
        preflight = preflight_study(study, options, output_dir=bundle.path / "preflight")
        if previous is not None and previous.provenance["study"] != {
            "path": str(preflight.study_path.resolve()),
            "sha256": preflight.study_sha256,
        }:
            raise HPOError("validation_run must be a validation checkpoint for this exact study")
        bundle.write_json("metadata/preflight.json", preflight.report)
        retained_config = bundle.write_bytes("metadata/config.yaml", raw)
        trial_fits = (
            0
            if fixed or saved is not None or reuse_tuning is not None
            else budget * len(options["windows"])
        )
        final_fits = 2 if fixed or saved is not None or options["stage"] == "tuning" else 1
        if validation_only or previous is not None:
            final_fits -= 1
        estimate = {
            "trial_callback_fits": trial_fits,
            "post_selection_callback_fits": final_fits,
            "total_callback_fits": trial_fits + final_fits,
        }
        bundle.write_json("metadata/work-estimate.json", estimate)
        logging.getLogger(__name__).warning(
            "Workflow work: %s trial fits + %s later fits = %s callback fits",
            trial_fits,
            final_fits,
            trial_fits + final_fits,
        )
        with ExitStack() as stack:
            if preflight.study_id != study.study_id:
                study = stack.enter_context(
                    load_study(
                        preflight.study_id,
                        registry_root=preflight.study_path.parent.parent,
                        spark=study._spark,
                    )
                )
            selection = None
            phase = "tuning"
            if fixed:
                parameters = {
                    f"{namespace}.{key}": value
                    for namespace, values in manual.items()
                    for key, value in values.items()
                }
                from autoforecast._exact_yaml import loads_exact_yaml

                saved = {
                    "schema": "autoforecast.selected-configuration.v1",
                    "model_id": spec.model_id,
                    "parameters": parameters,
                    "parameter_sha256": digest(parameters),
                    "profile_sha256": hashlib.sha256(raw).hexdigest(),
                    "profile": loads_exact_yaml(raw),
                    "profile_text": raw.decode("utf-8"),
                    "implementation": identity,
                    "seed": seed,
                    "source": {
                        "kind": "fixed",
                        "workflow_id": bundle.path.name,
                        "study": {"path": str(study._path.resolve()), "sha256": study._sha256},
                    },
                }
            elif saved is None:
                assert config is not None
                if reuse_tuning is not None:
                    tuning_path = pathlib.Path(reuse_tuning).absolute()
                selection = tune_study(
                    study,
                    fit_predict=fit_predict,
                    config=retained_config,
                    implementation=identity,
                    tuning_origin=tuning_origin,
                    output_dir=root / "hpo",
                    reuse_tuning=reuse_tuning,
                )
                tuning_path = selection.path
                bundle.write_json("metadata/tuning-reference.json", {"path": str(tuning_path)})
                saved = load_selected_configuration(selection.winner_config)
            parameters = saved["parameters"]
            grouped = materialize_runtime_parameters(spec, parameters)
            bundle.write_json("metadata/winner-config.json", saved)
            bundle.write_bytes("metadata/search-config.yaml", saved["profile_text"].encode("utf-8"))
            provenance = {
                "study": {"path": str(study._path.resolve()), "sha256": study._sha256},
                "implementation": identity,
                "configuration_source": saved["source"],
                "mode": (
                    "fixed_configuration"
                    if fixed
                    else "saved_configuration"
                    if selected_config is not None
                    else "reused_search"
                    if reuse_tuning is not None
                    else "tuned"
                ),
                "options": options,
                "seed": seed,
                "work_estimate": estimate,
            }
            if previous is not None:
                provenance["validation_run"] = {
                    "path": str(previous.path),
                    "manifest_sha256": hashlib.sha256(
                        (previous.path / "artifact_manifest.json").read_bytes()
                    ).hexdigest(),
                }
            bundle.write_json("metadata/provenance.json", provenance)
            temporary = pathlib.Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="af-workflow-"))
            )
            stages = {}
            recipe = study.recipe
            for stage, origin, horizon in (
                ("validation", recipe["validation_origins"][0], recipe["validation_horizon"]),
                ("test", recipe["final_train_cutoff"], recipe["final_test_horizon"]),
            ):
                if stage == "test" and validation_only:
                    continue
                phase = "validation" if stage == "validation" else "final_refit"
                directory = bundle.path / stage
                inputs = _prepare_window(study, stage, origin, horizon, temporary / stage)
                if stage == "validation" and previous is not None:
                    previous = load_forecast_workflow(previous.path)
                    if (
                        previous.validation["evaluation_contract"]
                        != inputs.verify()["evaluation_contract"]
                    ):
                        raise HPOError("validation checkpoint uses a different evaluation contract")
                    from mlops._safe_files import snapshot_regular_files

                    validation_source = previous.path / "validation"
                    snapshot_regular_files(
                        validation_source,
                        directory,
                        [
                            str(path.relative_to(validation_source))
                            for path in validation_source.rglob("*")
                            if path.is_file() and path != validation_source / "result.json"
                        ],
                    )
                    result = json.loads(
                        _json_bytes(
                            {
                                key: value
                                for key, value in previous.validation.items()
                                if key not in {"forecasts", "model_artifacts", "artifacts"}
                            }
                        )
                    )
                    # The retained score is unchanged; point its evidence at the new copy.
                    if "artifacts_sha256" in result and (
                        directory_digest(directory / "artifacts") != result["artifacts_sha256"]
                    ):
                        raise HPOError("validation callback artifacts changed during retention")

                    score = result["score"]
                    references = [score["series_scores"]]
                    if score.get("forecast_accuracy_breakdown"):
                        references.append(score["forecast_accuracy_breakdown"]["series_sums"])
                    for reference in references:
                        relative = pathlib.Path(reference["path"]).relative_to(validation_source)
                        retained_path = directory / relative
                        if (
                            directory_digest(retained_path, suffixes=(".parquet",))
                            != reference["sha256"]
                        ):
                            raise HPOError("validation score evidence changed during retention")
                        reference["path"] = str(retained_path)
                elif (
                    stage == "validation"
                    and selection is not None
                    and options["stage"] == "validation"
                ):
                    # The selected outer-boundary model is already fitted; retain and rescore it.
                    selection = load_tuning_result(selection.path)
                    window = selection.windows[0]
                    _copy_run_artifact_directory(window["model_artifacts"], directory / "model")
                    if window.get("artifacts") is not None:
                        _copy_run_artifact_directory(window["artifacts"], directory / "artifacts")
                        if directory_digest(directory / "artifacts") != window["artifacts_sha256"]:
                            raise HPOError("selected callback artifacts changed during retention")
                    forecast = window["forecasts"]
                    retained = snapshot_parquet_directory(
                        pathlib.Path(forecast.path),
                        directory / "forecasts",
                        expected_sha256=window["forecast_sha256"],
                        label="selected validation forecasts",
                    )
                    forecast = dataclasses.replace(forecast, path=str(retained))
                    fixed = inputs.verify()["evaluation_contract"]
                    if window["evaluation_contract"] != fixed:
                        raise HPOError("selected outer model uses a different validation boundary")
                    score = score_study(
                        dataclasses.replace(inputs.panel, path=fixed["actuals"]["path"]),
                        forecast,
                        study._spark,
                        contract=fixed,
                        output_dir=directory / "scores",
                        metrics=tuple(options["metrics"]),
                        forecast_accuracy_breakdown=options["workflow_breakdown"],
                    )
                    result = {
                        k: v for k, v in window.items()
                        if k not in ("forecasts", "model_artifacts", "artifacts")
                    }
                    result["score"] = score
                else:
                    result = _fit_window(
                        study._spark,
                        fit_predict,
                        inputs,
                        directory,
                        model_id=spec.model_id,
                        model_parameters=grouped["model"],
                        training_parameters=grouped["training"],
                        seed=seed,
                        metrics=options["metrics"],
                        breakdown=options["workflow_breakdown"],
                        purpose="validation" if stage == "validation" else "refit",
                    )
                result["forecast"]["path"] = stage + "/forecasts"
                result["model_path"] = stage + "/model"
                if "artifacts_path" in result:
                    result["artifacts_path"] = stage + "/artifacts"
                stages["testing" if stage == "test" else stage] = result
                bundle.write_json(stage + "/result.json", result)
                inputs.verify()
            study._record(data=True)
            result = {
                "schema": "autoforecast." + task + ".v1",
                "status": stages["validation" if validation_only else "testing"]["score"][
                    "evaluation_status"
                ],
                "parameters": parameters,
                "provenance": provenance,
                "tuning_path": str(selection.path) if selection else None,
                **stages,
            }
            bundle.write_json("result.json", result)
            bundle.finalize(status="completed", summary={"evaluation_status": result["status"]})
    except Exception as error:
        if phase == "tuning":
            tuning_path = getattr(error, "tuning_path", tuning_path)
        if (tuning_path is not None
                and not (bundle.path / "metadata/tuning-reference.json").exists()):
            bundle.write_json("metadata/tuning-reference.json", {"path": str(tuning_path)})
        bundle.write_json(
            "failure.json",
            {
                "phase": phase,
                "error": str(error),
                "validation_report": str(getattr(error, "report_path", "")),
                "tuning_path": str(tuning_path) if tuning_path is not None else None,
            },
        )
        bundle.finalize(status="failed", summary={"phase": phase})
        raise
    return load_forecast_workflow(bundle.path)
