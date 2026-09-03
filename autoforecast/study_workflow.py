"""One external-model workflow using the existing tuner, scorer and run bundles."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, cast

from mlops.run_bundle import load_completed_run_bundle, open_run_bundle

from autoforecast._exact_json import load_exact_json
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
    _validate_workflow_configuration,
    _window_result,
    load_selected_configuration,
    load_tuning_result,
    tune_study,
)
from autoforecast.tuning import loads_external_tuning_spec, materialize_runtime_parameters
from provenance.lineage import snapshot_parquet_directory

if TYPE_CHECKING:
    from autoforecast.study import Study


@dataclasses.dataclass(frozen=True)
class ForecastWorkflowResult:
    """Retained models and scores; pending actuals are never a completed evaluation."""

    path: pathlib.Path
    status: str
    parameters: dict
    validation: dict
    testing: dict
    provenance: dict
    tuning_path: pathlib.Path | None
    selected_config: pathlib.Path
    search_config: pathlib.Path


def load_forecast_workflow(path: str | pathlib.Path) -> ForecastWorkflowResult:
    """Verify every workflow artifact before returning its saved stage results."""
    root = pathlib.Path(path).absolute()
    completion = load_completed_run_bundle(root)
    if completion["task"] != "forecast-workflow" or completion["status"] != "completed":
        raise HPOError("workflow did not finish successfully")
    record = cast(dict, load_exact_json(root / "result.json"))
    if record["schema"] != "autoforecast.forecast-workflow.v1":
        raise HPOError("unsupported forecast workflow result")
    _validate_workflow_configuration(
        root, record, cast(dict, load_exact_json(root / "metadata/winner-config.json"))
    )
    stages = {stage: _window_result(root, record[stage]) for stage in ("validation", "testing")}
    return ForecastWorkflowResult(
        root,
        record["status"],
        record["parameters"],
        stages["validation"],
        stages["testing"],
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
    identity = _implementation_record(implementation)
    if selected_config is not None and (reuse_tuning is not None or tuning_origin is not None):
        raise ValueError("selected_config conflicts with reuse_tuning and tuning_origin")
    saved = load_selected_configuration(selected_config) if selected_config is not None else None
    if saved is not None and saved["implementation"] != identity:
        raise HPOError("saved configuration uses a different model implementation")
    if config is not None:
        raw, spec, budget, seed, controls = _config(pathlib.Path(config))
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
    if saved is not None:
        if spec.model_id != saved["model_id"]:
            raise HPOError("saved configuration uses a different model identity")
        materialize_runtime_parameters(spec, saved["parameters"])
        seed = saved["seed"]
        controls = {**controls, "selection": {"stage": "validation"}}
    options = _resolve_options(study, controls, tuning_origin)
    if saved is not None:
        options["selection_boundaries"] = []
    root = _output_root(study, output_dir, "study-workflows")
    bundle = open_run_bundle(root, run_id="workflow-" + uuid.uuid4().hex, task="forecast-workflow")
    phase = "preflight"
    try:
        preflight = preflight_study(study, options)
        bundle.write_json("metadata/preflight.json", preflight.report)
        retained_config = bundle.write_bytes("metadata/config.yaml", raw)
        trial_fits = (
            0 if saved is not None or reuse_tuning is not None else budget * len(options["windows"])
        )
        final_fits = 2 if saved is not None or options["stage"] == "tuning" else 1
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
            if saved is None:
                assert config is not None
                selection = tune_study(
                    study,
                    fit_predict=fit_predict,
                    config=retained_config,
                    implementation=identity,
                    tuning_origin=tuning_origin,
                    output_dir=root / "hpo",
                    reuse_tuning=reuse_tuning,
                )
                assert selection.winner_config is not None
                saved = load_selected_configuration(selection.winner_config)
            parameters = saved["parameters"]
            grouped = materialize_runtime_parameters(spec, parameters)
            bundle.write_json("metadata/winner-config.json", saved)
            bundle.write_bytes("metadata/search-config.yaml", saved["profile_text"].encode("utf-8"))
            provenance = {
                "study": {"path": str(study._path.resolve()), "sha256": study._sha256},
                "implementation": identity,
                "configuration_source": saved["source"],
                "mode": "saved_configuration"
                if selected_config is not None
                else "reused_search"
                if reuse_tuning is not None
                else "tuned",
                "options": options,
                "seed": seed,
                "work_estimate": estimate,
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
                phase = "validation" if stage == "validation" else "final_refit"
                directory = bundle.path / stage
                inputs = _prepare_window(study, stage, origin, horizon, temporary / stage)
                if (
                    stage == "validation"
                    and selection is not None
                    and options["stage"] == "validation"
                ):
                    # The selected outer-boundary model is already fitted; retain and rescore it.
                    selection = load_tuning_result(selection.path)
                    window = selection.windows[0]
                    _copy_run_artifact_directory(window["model_artifacts"], directory / "model")
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
                        k: v for k, v in window.items() if k not in ("forecasts", "model_artifacts")
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
                    )
                result["forecast"]["path"] = stage + "/forecasts"
                result["model_path"] = stage + "/model"
                stages["testing" if stage == "test" else stage] = result
                bundle.write_json(stage + "/result.json", result)
                inputs.verify()
            study._record(data=True)
            result = {
                "schema": "autoforecast.forecast-workflow.v1",
                "status": stages["testing"]["score"]["evaluation_status"],
                "parameters": parameters,
                "provenance": provenance,
                "tuning_path": str(selection.path) if selection else None,
                **stages,
            }
            bundle.write_json("result.json", result)
            bundle.finalize(status="completed", summary={"evaluation_status": result["status"]})
    except Exception as error:
        bundle.write_json(
            "failure.json",
            {
                "phase": phase,
                "error": str(error),
                "validation_report": str(getattr(error, "report_path", "")),
            },
        )
        bundle.finalize(status="failed", summary={"phase": phase})
        raise
    return load_forecast_workflow(bundle.path)
