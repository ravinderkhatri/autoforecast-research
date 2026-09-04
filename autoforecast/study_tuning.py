"""Caller-supplied models tuned across a frozen study's selected boundaries."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import re
import tempfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack, suppress
from typing import TYPE_CHECKING, cast

from mlops._safe_files import (
    _platform_absolute,
    is_regular_file,
    read_path_bytes,
    read_regular_bytes,
    snapshot_tree_types,
)
from mlops.run_report import new_run_id
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast._exact_json import loads_exact_json
from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast._execution import _stage, callable_name, note, reported
from autoforecast.benchmark import (
    MISSING_COLUMN,
    boundary,
    digest,
    evaluation_contract,
    verify_benchmark,
)
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.experiments import load_completed_attempt, load_study_selection
from autoforecast.frequency import (
    TimePoint,
    canonical_origin,
    period_points,
)
from autoforecast.hpo import HPOError, HPOStudySpec, TrialOutcome, TrialRequest, run_hpo_study
from autoforecast.hpo_bridge import _copy_run_artifact_directory
from autoforecast.implementation import (
    CapturedImplementation,
    resolve_implementation,
    verify_retained,
)
from autoforecast.panel import Forecast, Panel
from autoforecast.project import library_revision
from autoforecast.study_horizon_metrics import (
    ForecastAccuracyBreakdown,
    validate_forecast_accuracy_breakdown,
)
from autoforecast.study_preflight import preflight_study
from autoforecast.study_scoring import SELECTION_METRICS, score_study, validate_study_metrics
from autoforecast.tuning import (
    TuningError,
    TuningSpec,
    loads_external_tuning_spec,
    materialize_runtime_parameters,
)
from provenance.lineage import directory_digest, snapshot_parquet_directory

if TYPE_CHECKING:
    from autoforecast.study import Study


@dataclasses.dataclass(frozen=True)
class FitPredictRequest:
    """Training inputs for one fit, without actuals for its forecast dates.

    ``panel.path`` contains only training rows through ``origin``. Save forecasts
    beneath ``output_dir`` and at least one model file beneath ``output_dir/model``.
    Apply the supplied model/training parameters and seed in the caller's code.
    Optional logs or diagnostics under output_dir/artifacts are retained too.
    Other scratch files are deleted after the callback returns.
    The seed is held constant across trials to compare parameter configurations.
    Dataframes borrow the study's Spark session and are valid during the callback.
    purpose is selection (including outer-validation searches), validation, or refit.
    It is assigned by orchestration, never by model hyperparameters.
    """

    training_data: DataFrame
    series_mapping: DataFrame
    panel: Panel
    origin: str
    forecast_dates: tuple[TimePoint, ...]
    model_id: str
    model_parameters: dict[str, object]
    training_parameters: dict[str, object]
    seed: int
    output_dir: pathlib.Path
    purpose: str = "selection"


@dataclasses.dataclass(frozen=True)
class FitPredictResult:
    """Forecasts and the caller's model format; AutoForecast computes the score.

    Forecast.adapter must equal request.model_id. The model directory is fixed at
    request.output_dir/model. Loading its contents remains the caller's responsibility.
    """

    forecasts: Forecast
    model_format: str


@dataclasses.dataclass(frozen=True)
class TuningResult:
    """Verified selected search result, independent of the live Study session.

    Parameters use the profile's full names (for example ``model.trend``).
    Reload with ``load_tuning_result(path)`` to recheck persisted evidence. This is
    selected search model; inner selection needs a later validation fit and final refit.
    Multi-window results expose separate artifacts in windows; singular artifacts are None.
    No model deserialization or executable import happens when loading this result.
    """

    path: pathlib.Path
    parameters: dict[str, object]
    trial_id: str
    model_id: str
    tuning_score: float
    forecasts: Forecast | None
    model_artifacts: pathlib.Path | None
    model_format: str
    trial_history: tuple[dict, ...]
    provenance: dict
    windows: tuple[dict, ...]
    winner_config: pathlib.Path
    validation_report: pathlib.Path
    search_config: pathlib.Path


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _retained_manifest(completion: dict, payload: bytes) -> dict:
    """Anchor one immutable manifest to the completion verified by the existing loader."""
    if hashlib.sha256(payload).hexdigest() != completion["artifact_manifest_sha256"]:
        raise HPOError("retained manifest differs from verified completion")
    manifest = cast(dict, loads_exact_json(payload))
    terminal = {key: value for key, value in completion.items()
                if key != "artifact_manifest_sha256"}
    terminal_bytes = (
        json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if hashlib.sha256(terminal_bytes).hexdigest() != manifest["terminal_payload_sha256"]:
        raise HPOError("retained completion differs from its sealed terminal payload")
    return manifest


def _retained_bytes(root: pathlib.Path, relative: str, manifest: dict) -> bytes:
    """Read once, then authenticate the exact bytes that the consumer will parse."""
    payload = read_regular_bytes(root, relative)
    entries = [entry for entry in manifest["artifacts"] if entry["path"] == relative]
    if (len(entries) != 1 or len(payload) != entries[0]["bytes"]
            or hashlib.sha256(payload).hexdigest() != entries[0]["sha256"]):
        raise HPOError(f"retained artifact differs from its sealed bytes: {relative}")
    return payload


def _retained_json(root: pathlib.Path, relative: str, manifest: dict) -> dict:
    return cast(dict, loads_exact_json(_retained_bytes(root, relative, manifest)))


def _implementation_record(value: Mapping[str, object]) -> dict:
    """Capture declared code identity; this does not attest arbitrary caller Python."""
    if not isinstance(value, Mapping) or set(value) != {
        "name",
        "version",
        "source_sha256",
        "dependencies",
    }:
        raise ValueError("implementation requires name, version, source_sha256 and dependencies")
    if any(type(item) is not str or not item.strip() for item in (value["name"], value["version"])):
        raise ValueError("implementation name and version must be nonblank strings")
    sha = value["source_sha256"]
    if type(sha) is not str or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("implementation source_sha256 must be a lowercase SHA-256")
    dependencies = value["dependencies"]
    if type(dependencies) is not dict or any(
        type(k) is not str or not k.strip() or type(v) is not str or not v.strip()
        for k, v in dependencies.items()
    ):
        raise ValueError("implementation dependencies must map package names to version strings")
    return json.loads(_json_bytes(dict(value)))


def _config(
    path: pathlib.Path,
    *,
    require_hpo: bool = True,
) -> tuple[bytes, TuningSpec, int, int, dict]:
    """Read once: profile settings and explicit budget; dates come only from the study."""
    if path.is_symlink() or not is_regular_file(path):
        raise TuningError("external tuning config must be a regular file")
    raw = read_path_bytes(path)
    try:
        value = loads_exact_yaml(raw)
    except ExactYAMLError as error:
        raise TuningError(f"invalid YAML in external tuning config: {error}") from error
    if type(value) is not dict or (require_hpo and "hpo" not in value):
        raise TuningError("external tuning config requires an hpo mapping")
    controls = {
        name: value.pop(name)
        for name in (
            "selection",
            "reporting",
            "series_policy",
            "minimum_history",
        )
        if name in value
    }
    options = value.pop("hpo", {"backend": "optuna", "max_trials": 1, "seed": 42})
    if type(options) is not dict or set(options) != {"backend", "max_trials", "seed"}:
        raise TuningError("hpo requires exactly backend, max_trials and seed")
    if options["backend"] != "optuna":
        raise TuningError("Study.tune supports only sequential optuna execution")
    budget, seed = options["max_trials"], options["seed"]
    if type(budget) is not int or budget < 1:
        raise TuningError("hpo.max_trials must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise TuningError("hpo.seed must be an integer in [0, 2**32 - 1]")
    try:
        profile = _json_bytes(value)
    except (TypeError, ValueError) as error:
        raise TuningError("external profile requires finite JSON-compatible values") from error
    spec = loads_external_tuning_spec(profile, source_path=path)
    spec = dataclasses.replace(spec, source_sha256=hashlib.sha256(raw).hexdigest())
    return raw, spec, budget, seed, controls


def _validation_controls(controls: dict) -> dict:
    """Skip search windows while retaining the declared comparison and reporting rules."""
    selection = controls.get("selection", {})
    if type(selection) is not dict:
        raise TuningError("selection must be a mapping")
    return {
        **controls,
        "selection": {
            **{key: value for key, value in selection.items() if key not in {"stage", "origins"}},
            "stage": "validation",
        },
    }


def _resolve_options(study: Study, controls: dict, tuning_origin: str | None) -> dict:
    recipe = study.recipe
    if tuning_origin is not None and (type(tuning_origin) is not str or not tuning_origin.strip()):
        raise ValueError("tuning_origin must be a nonblank frozen origin string")
    selection = controls.get("selection", {})
    if type(selection) is not dict or set(selection) - {
        "stage",
        "origins",
        "metric",
        "aggregation",
        "direction",
    }:
        raise TuningError("selection contains unknown settings")
    stage = selection.get("stage", "tuning")
    metric = selection.get("metric", recipe["metric"])
    if stage not in ("tuning", "validation"):
        raise TuningError("selection.stage must be tuning or validation; testing is reserved")
    if metric not in SELECTION_METRICS:
        raise TuningError("selection.metric must be mae, rmse, wape or wrmsse")
    if (
        selection.get("aggregation", "mean") != "mean"
        or selection.get("direction", "minimize") != "minimize"
    ):
        raise TuningError("selection uses equal-weight mean and the metric's minimize direction")
    if tuning_origin is not None and ("origins" in selection or stage != "tuning"):
        raise TuningError("tuning_origin conflicts with the selection policy")
    if stage == "validation":
        if "origins" in selection:
            raise TuningError("validation selection uses the existing outer boundary")
        windows = [
            {"origin": recipe["validation_origins"][0], "horizon": recipe["validation_horizon"]}
        ]
    else:
        windows = recipe.get("tuning_windows", [])
        if not windows:
            raise HPOError("Study.tune requires tuning_windows declared before freezing the study")
        origins = selection.get("origins")
        if tuning_origin is not None:
            origins = [tuning_origin]
        if origins is None:
            if len(windows) != 1:
                raise HPOError(
                    "choose tuning_origin explicitly when a study has multiple tuning windows, "
                    "or selection.origins=all"
                )
        elif origins != "all":
            if (
                type(origins) is not list
                or not origins
                or any(type(x) is not str for x in origins)
                or len(origins) != len(set(origins))
            ):
                raise TuningError("selection.origins must be all or a nonempty unique list")
            if not set(origins) <= {w["origin"] for w in windows}:
                raise HPOError("tuning_origin is absent from the frozen study")
            windows = [window for window in windows if window["origin"] in origins]
    reporting = controls.get("reporting", {})
    if type(reporting) is not dict or set(reporting) - {"metrics", "forecast_accuracy_breakdown"}:
        raise TuningError("reporting contains unknown settings")
    defaults = ["mae", "rmse", "wape", "bias"] + (
        ["wrmsse"] if recipe["metric"] == "wrmsse" else []
    )
    metrics = validate_study_metrics(reporting.get("metrics", defaults))
    metrics = tuple(dict.fromkeys([*metrics, metric]))
    mode = reporting.get("forecast_accuracy_breakdown", "none")
    validate_forecast_accuracy_breakdown(mode)
    policy = controls.get("series_policy", "error")
    if policy not in ("error", "exclude_unavailable", "use_available"):
        raise TuningError("series_policy must be error, exclude_unavailable or use_available")
    minimum = controls.get("minimum_history", 1)
    if type(minimum) is not int or minimum < 1:
        raise TuningError("minimum_history must be a positive integer declared for the model")
    return {
        "stage": stage,
        "windows": windows,
        "metric": metric,
        "metrics": list(metrics),
        "reported_metrics": list(reporting.get("metrics", defaults)),
        "aggregation": "mean",
        "direction": "minimize",
        "breakdown": mode,
        "workflow_breakdown": reporting.get("forecast_accuracy_breakdown", "per_step"),
        "series_policy": policy,
        "minimum_history": minimum,
        "selection_boundaries": [
            boundary(stage, w["origin"], w["horizon"], recipe["frequency"]) for w in windows
        ],
    }


def _window_result(attempt: pathlib.Path, evidence: dict) -> dict:
    record = dict(evidence)
    if ("artifacts_path" in record) != ("artifacts_sha256" in record):
        raise HPOError("callback artifacts require both path and checksum")
    paths = [record["forecast"]["path"], record["model_path"]]
    if "artifacts_path" in record:
        paths.append(record["artifacts_path"])
    for relative in paths:
        path = pathlib.PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise HPOError("retained window path must stay within its run")
    forecast = Forecast.from_record(record["forecast"])
    forecast = dataclasses.replace(forecast, path=str(attempt / record["forecast"]["path"]))
    forecast.validate()
    contract = record["evaluation_contract"]
    _stage, origin, horizon, frequency = record["boundary"].split("/")
    if (
        contract["boundary"] != record["boundary"]
        or (forecast.origin, forecast.horizon, forecast.frequency)
        != (origin, int(horizon), frequency)
        or forecast.n_declined
        or forecast.n_emitted != contract["grid"]["n_cells"]
        or type(record["model_format"]) is not str
        or not record["model_format"].strip()
    ):
        raise HPOError("retained window forecast or model format differs from its boundary")
    model = attempt / record["model_path"]
    if (
        directory_digest(forecast.path, suffixes=(".parquet",)) != record["forecast_sha256"]
        or directory_digest(model) != record["model_sha256"]
    ):
        raise HPOError("retained window model or forecasts changed")
    extra = {}
    if "artifacts_path" in record:
        artifacts = attempt / record["artifacts_path"]
        if directory_digest(artifacts) != record["artifacts_sha256"]:
            raise HPOError("retained callback artifacts changed")
        extra["artifacts"] = artifacts
    return {**record, "forecasts": forecast, "model_artifacts": model, **extra}


def load_tuning_result(path: str | pathlib.Path) -> TuningResult:
    """Verify all attempts/windows and reload a completed search, without model execution."""
    root = pathlib.Path(path).absolute()
    selection = load_study_selection(root)
    chosen = selection["selected"][0]
    attempt = root / chosen["attempt_path"]
    manifest_bytes = read_regular_bytes(attempt, "artifact_manifest.json")
    _, completion = load_completed_attempt(attempt)
    manifest = _retained_manifest(completion, manifest_bytes)
    if completion["artifact_manifest_sha256"] != chosen["artifact_manifest_sha256"]:
        raise HPOError("selected attempt manifest differs from selection")
    metadata = _retained_json(attempt, "metadata/attempt.json", manifest)
    evidence = _retained_json(attempt, "evaluation/external-run.json", manifest)
    if type(evidence) is not dict or evidence.get("schema") not in {
        "autoforecast.external-trial.v1",
        "autoforecast.external-trial.v2",
    }:
        raise HPOError("selected attempt is not a supported external-model trial")
    context = evidence["context"]
    verify_retained(attempt, context["implementation"], context.get("implementation_evidence"))
    if hashlib.sha256(_json_bytes(context)).hexdigest() != selection["comparability_fingerprint"]:
        raise HPOError("external selection context differs from its frozen identity")
    profile = _retained_bytes(attempt, "metadata/external-profile.yaml", manifest)
    if hashlib.sha256(profile).hexdigest() != context["profile_sha256"]:
        raise HPOError("external profile differs from its frozen identity")
    if (
        context["max_trials"] != len(selection["candidates"])
        or context["model_id"] != metadata["model"]["id"]
    ):
        raise HPOError("external selection budget or model differs from its context")
    study_path, _ = verify_benchmark(
        pathlib.Path(context["study"]["path"]), context["study"]["sha256"]
    )
    dataset_path = study_path.parent / "dataset.yaml"
    dataset = _load_dataset_contract_bytes(read_path_bytes(dataset_path), dataset_path)
    if not isinstance(evidence.get("windows"), list):
        raise HPOError("external trial requires current window evidence; rerun the search")
    windows = tuple(_window_result(attempt, window) for window in evidence["windows"])
    if (
        not windows
        or len(windows) != len(context["windows"])
        or [w["boundary"] for w in windows]
        != [w["evaluation_contract"]["boundary"] for w in context["windows"]]
        or any(w["parameter_sha256"] != digest(metadata["hyperparameters"]) for w in windows)
    ):
        raise HPOError("window scores, parameters or aggregate differ from selection")
    if any(
        window["forecasts"].adapter != context["model_id"]
        or window["forecasts"].target_estimand
        != dataset.panel(window["forecasts"].path).target_estimand
        or type(window["score"]) is not dict
        for window in windows
    ):
        raise HPOError("saved forecast identity or score differs from the selected context")
    scores = [window["score"].get(context["metric"]) for window in windows]
    if (
        any(type(value) not in (int, float) or not math.isfinite(value) for value in scores)
        or math.fsum(scores) / len(scores) != chosen["validation_objective"]
    ):
        raise HPOError("saved window scores differ from the selected objective")
    winner = attempt / "metadata/winner-config.json"
    preflight = attempt / "metadata/preflight.json"
    if not winner.is_file() or not preflight.is_file():
        raise HPOError("current tuning results require winner settings and preflight evidence")
    saved = load_selected_configuration(winner)
    if (
        saved["parameters"] != metadata["hyperparameters"]
        or saved["profile_sha256"] != context["profile_sha256"]
        or saved["implementation"] != context["implementation"]
        or saved["seed"] != context["seed"]
    ):
        raise HPOError("saved winner configuration differs from the selected trial")
    return TuningResult(
        path=root,
        parameters=dict(metadata["hyperparameters"]),
        trial_id=chosen["trial_id"],
        model_id=context["model_id"],
        tuning_score=chosen["validation_objective"],
        forecasts=windows[0]["forecasts"] if len(windows) == 1 else None,
        model_artifacts=windows[0]["model_artifacts"] if len(windows) == 1 else None,
        model_format=windows[0]["model_format"] if len(windows) == 1 else "per_window",
        trial_history=tuple(selection["candidates"]),
        provenance=context,
        windows=windows,
        winner_config=winner,
        search_config=attempt / "metadata/external-profile.yaml",
        validation_report=preflight,
    )


@dataclasses.dataclass(frozen=True)
class _WindowInput:
    panel: Panel
    mapping_path: pathlib.Path
    context_bytes: bytes

    def verify(self) -> dict:
        context = json.loads(self.context_bytes)
        identity = context["evaluation_contract"]["study"]
        verify_benchmark(pathlib.Path(identity["path"]), identity["sha256"])
        for key, path in (
            ("training_sha256", self.panel.path),
            ("mapping_sha256", self.mapping_path),
        ):
            if directory_digest(path, suffixes=(".parquet",)) != context[key]:
                raise HPOError("external tuning input snapshot changed")
        return context


@reported("Prepare training window and verify inputs")
def _prepare_window(
    study: Study, stage: str, origin: str, horizon: int, root: pathlib.Path
) -> _WindowInput:
    fixed = evaluation_contract(
        study._path, stage=stage, origin=origin, horizon=horizon, metric=study.recipe["metric"]
    )
    if fixed["study"]["sha256"] != study._sha256:
        raise HPOError("study changed before fitting")
    panel = study.panel
    frame = study._fit_data(
        "final_training" if stage == "test" else "selection",
        "refit" if stage == "test" else "training", origin,
    )
    mapping = study.series_mapping
    eligible = fixed.get("availability", {}).get("eligible_ids")
    if eligible is not None:
        frame = frame.filter(F.col(panel.series_col).isin(eligible))
        mapping = mapping.filter(F.col(panel.series_col).isin(eligible))
    training, mapping_path = root / "training", root / "series-mapping"
    frame.write.parquet(str(training))
    mapping.write.parquet(str(mapping_path))
    study._record(data=True)
    context = {
        "evaluation_contract": fixed,
        "stage": stage,
        "origin": origin,
        "horizon": horizon,
        "history_zero_filled": (frame.filter(F.col(MISSING_COLUMN)).count()
                                if study.recipe["history_missing"] == "zero" else 0),
        "training_sha256": directory_digest(training, suffixes=(".parquet",)),
        "mapping_sha256": directory_digest(mapping_path, suffixes=(".parquet",)),
    }
    return _WindowInput(
        dataclasses.replace(panel, path=str(training)), mapping_path, _json_bytes(context)
    )


def _retain_callback_artifacts(work, output):
    artifact_source = work / "artifacts"
    if artifact_source.exists() or artifact_source.is_symlink():
        kinds = snapshot_tree_types(artifact_source)
        if (artifact_source.is_symlink() or not artifact_source.is_dir()
                or any(kind not in {"file", "directory"} for kind in kinds.values())):
            raise HPOError("callback artifacts may not contain symlinks or special files")
        if "file" in kinds.values():
            artifact_sha = directory_digest(artifact_source)
            _copy_run_artifact_directory(artifact_source, output / "artifacts")
            if directory_digest(output / "artifacts") != artifact_sha:
                raise HPOError("callback artifacts changed while retaining files")
            return {"artifacts_path": "artifacts", "artifacts_sha256": artifact_sha}
    return {}


def _scoring_outcome(score):
    """Summarize the same checked score for fitted and reused validation models."""
    outcome = {}
    outcome["coverage"] = score["coverage"]
    outcome["evaluation_status"] = score["evaluation_status"]
    outcome["metric_status"] = score["metric_status"]
    outcome["scores"] = {k: v for k, v in score.items()
                         if isinstance(v, int | float) and not isinstance(v, bool)}
    outcome["warnings"] = [
        name for name in ("zero_filled", "excluded") if score["coverage"].get(name, 0)
    ]
    if score["evaluation_status"] == "pending_actuals":
        outcome["status"] = "UNAVAILABLE"
    return outcome


def _fit_window(
    spark,
    callback,
    inputs: _WindowInput,
    output: pathlib.Path,
    *,
    model_id: str,
    model_parameters: dict,
    training_parameters: dict,
    seed: int,
    metrics: list[str],
    breakdown: ForecastAccuracyBreakdown,
    purpose: str = "selection",
    run_id: str | None = None,
    trial_id: str | None = None,
    implementation_capture: CapturedImplementation | None = None,
) -> dict:
    if purpose not in {"selection", "validation", "refit"}:
        raise ValueError("unknown fitting purpose")
    if implementation_capture is not None:
        implementation_capture.verify()
    context = inputs.verify()
    panel = inputs.panel
    origin, horizon = context["origin"], context["horizon"]
    extras = {}
    with tempfile.TemporaryDirectory(
        prefix="autoforecast-fit-", dir=pathlib.Path(inputs.panel.path).parent,
    ) as temporary:
        work = pathlib.Path(temporary)
        try:
            with _stage(output / "events.jsonl", "fit_predict", model_id=model_id,
                        origin=origin, purpose=purpose, window=output.name,
                        forecast_start=str(period_points(origin, panel.frequency, horizon)[0]),
                        forecast_periods=horizon, frequency=panel.frequency,
                        study_id=context["evaluation_contract"]["study_id"],
                        run_id=run_id, trial_id=trial_id,
                        history_zero_filled=context.get("history_zero_filled", 0),
                        warnings=(["history_zero_filled"]
                                  if context.get("history_zero_filled", 0) else [])):
                note("Callback inputs", callback=callable_name(callback),
                     model_parameters=model_parameters, training_parameters=training_parameters,
                     history_through=origin, purpose=purpose)
                result = callback(
                    FitPredictRequest(
                        training_data=spark.read.parquet(panel.path),
                        series_mapping=spark.read.parquet(str(inputs.mapping_path)),
                        panel=Panel.from_record(dataclasses.asdict(panel)),
                        origin=origin,
                        forecast_dates=tuple(period_points(origin, panel.frequency, horizon)),
                        model_id=model_id,
                        model_parameters=json.loads(_json_bytes(model_parameters)),
                        training_parameters=json.loads(_json_bytes(training_parameters)),
                        seed=seed,
                        output_dir=work,
                        purpose=purpose,
                    )
                )
        except BaseException:
            with suppress(Exception):
                _retain_callback_artifacts(work, output)
            raise
        if implementation_capture is not None:
            implementation_capture.verify()
        with _stage(output / "events.jsonl", "artifact_retention", model_id=model_id,
                    origin=origin, run_id=run_id, trial_id=trial_id):
            extras = _retain_callback_artifacts(work, output)
            inputs.verify()
            if (
                type(result) is not FitPredictResult
                or type(result.forecasts) is not Forecast
                or type(result.model_format) is not str
                or not result.model_format.strip()
            ):
                raise HPOError(
                    "fit_predict must return FitPredictResult with a Forecast and model_format"
                )
            forecast = Forecast.from_record(result.forecasts.as_record())
            if (forecast.origin, forecast.horizon, forecast.frequency, forecast.adapter) != (
                origin,
                horizon,
                panel.frequency,
                model_id,
            ):
                raise HPOError("external forecast uses the wrong model or tuning boundary")
            if not pathlib.Path(forecast.path).resolve().is_relative_to(work.resolve()):
                raise HPOError("external forecasts must be saved beneath output_dir")
            model_source = work / "model"
            if not model_source.is_dir() or not any(p.is_file() for p in model_source.rglob("*")):
                raise HPOError("external trial requires saved model files beneath output_dir/model")
            model_sha = directory_digest(model_source)
            _copy_run_artifact_directory(model_source, output / "model")
            if directory_digest(output / "model") != model_sha:
                raise HPOError("external model changed while retaining artifacts")
            forecast_sha = directory_digest(forecast.path, suffixes=(".parquet",))
            retained = snapshot_parquet_directory(
                pathlib.Path(forecast.path),
                output / "forecasts",
                expected_sha256=forecast_sha,
                label="external forecasts",
            )
            forecast = dataclasses.replace(forecast, path=str(retained))
            forecast.validate()
    fixed = context["evaluation_contract"]
    with _stage(output / "events.jsonl", "scoring", model_id=model_id,
                metrics=metrics, breakdown=breakdown,
                origin=origin, purpose=purpose, window=output.name,
                        forecast_start=str(period_points(origin, panel.frequency, horizon)[0]),
                        forecast_periods=horizon, frequency=panel.frequency,
                study_id=context["evaluation_contract"]["study_id"],
                run_id=run_id, trial_id=trial_id) as outcome:
        score = score_study(
            dataclasses.replace(panel, path=fixed["actuals"]["path"]),
            forecast,
            spark,
            contract=fixed,
            output_dir=output / "scores",
            metrics=tuple(metrics),
            forecast_accuracy_breakdown=breakdown,
            scratch_root=pathlib.Path(inputs.panel.path).parent,
        )
        outcome.update(_scoring_outcome(score))
    inputs.verify()
    if implementation_capture is not None:
        implementation_capture.verify()
    record = {
        **extras,
        "boundary": fixed["boundary"],
        "evaluation_contract": fixed,
        "forecast": {**forecast.as_record(), "path": "forecasts"},
        "model_path": "model",
        "forecast_sha256": forecast_sha,
        "model_sha256": model_sha,
        "model_format": result.model_format,
        "score": score,
        "parameter_sha256": digest(
            {
                **{"model." + k: v for k, v in model_parameters.items()},
                **{"training." + k: v for k, v in training_parameters.items()},
            }
        ),
    }
    return record


@dataclasses.dataclass(frozen=True)
class _ExternalTrialBridge:
    spark: SparkSession
    callback: Callable[[FitPredictRequest], FitPredictResult]
    inputs: tuple[_WindowInput, ...]
    context_bytes: bytes
    profile_bytes: bytes
    report_bytes: bytes
    implementation_capture: CapturedImplementation | None = None

    def __call__(self, request: TrialRequest) -> TrialOutcome:
        context = json.loads(self.context_bytes)
        records = []
        if self.implementation_capture is not None:
            self.implementation_capture.retain(request.attempt.path)
        request.attempt.write_bytes("metadata/external-profile.yaml", self.profile_bytes)
        request.attempt.write_bytes("metadata/preflight.json", self.report_bytes)
        def retain_evidence():
            # A failed trial still owns its exact inputs and any windows completed before failure.
            evidence = {
                "schema": "autoforecast.external-trial.v1"
                if len(self.inputs) == 1 else "autoforecast.external-trial.v2",
                "context": context,
                "windows": records,
                "aggregation": "mean_of_window_scores",
                "model_format": records[0]["model_format"]
                if len(self.inputs) == 1 and records else "per_window",
                "training_attestation": "caller_supplied",
                "test_metrics": "not_evaluated",
            }
            if len(self.inputs) == 1 and records:
                evidence.update({
                    "forecast_sha256": records[0]["forecast_sha256"],
                    "model_sha256": records[0]["model_sha256"],
                })
            elif (request.attempt.path / "windows").is_dir():
                evidence["windows_sha256"] = directory_digest(request.attempt.path / "windows")
            request.attempt.write_json("evaluation/external-run.json", evidence)

        try:
            for index, inputs in enumerate(self.inputs):
                note("CV window", window_index=index + 1, window_total=len(self.inputs),
                     search_id=request.study.id, trial_id=request.trial_id)
                prefix = "" if len(self.inputs) == 1 else f"windows/{index:04d}"
                directory = request.attempt.path / prefix
                record = _fit_window(
                    self.spark,
                    self.callback,
                    inputs,
                    directory,
                    model_id=context["model_id"],
                    model_parameters=request.model_parameters,
                    training_parameters=request.training_parameters,
                    seed=context["seed"],
                    metrics=context["options"]["metrics"],
                    breakdown=context["options"]["breakdown"],
                    run_id=request.study.id, trial_id=request.trial_id,
                    implementation_capture=self.implementation_capture,
                )
                value = record["score"].get(context["metric"])
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise HPOError(
                        "external trial requires a finite tuning objective in every window"
                    )
                record["forecast"]["path"] = str(pathlib.PurePosixPath(prefix) / "forecasts")
                record["model_path"] = str(pathlib.PurePosixPath(prefix) / "model")
                if "artifacts_path" in record:
                    record["artifacts_path"] = str(pathlib.PurePosixPath(prefix) / "artifacts")
                records.append(record)
        except BaseException:
            with suppress(Exception):
                retain_evidence()
            raise
        else:
            retain_evidence()
        value = math.fsum(record["score"][context["metric"]] for record in records) / len(records)
        for inputs in self.inputs:
            inputs.verify()
        parameters = {
            **{"model." + k: v for k, v in request.model_parameters.items()},
            **{"training." + k: v for k, v in request.training_parameters.items()},
        }
        request.attempt.write_json(
            "metadata/winner-config.json",
            {
                "schema": "autoforecast.selected-configuration.v1",
                "model_id": context["model_id"],
                "parameters": parameters,
                "parameter_sha256": digest(parameters),
                "profile_sha256": context["profile_sha256"],
                "implementation": context["implementation"],
                "profile": loads_exact_yaml(self.profile_bytes),
                "profile_text": self.profile_bytes.decode("utf-8"),
                "seed": context["seed"],
                "source": {
                    "study": context["study"],
                    "search_id": request.study.id,
                    "trial_id": request.trial_id,
                    "attempt_id": request.attempt.attempt_id,
                },
            },
        )
        if len(records) == 1:
            request.attempt.write_json("evaluation/forecast.json", records[0]["forecast"])
            request.attempt.write_json("evaluation/score.json", records[0]["score"])
        return TrialOutcome(float(value), {"validation/" + context["metric"]: float(value)})


def _output_root(study: Study, supplied: str | pathlib.Path | None, name: str) -> pathlib.Path:
    root = (
        pathlib.Path(supplied).absolute()
        if supplied is not None
        else study._path.parent.parent.parent / name / study.study_id
    )
    root = _platform_absolute(root)
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise HPOError("HPO output root may not traverse a symlink")
    # Run artifacts must never become children of inputs or already sealed evidence.
    record = study._record()
    protected = [
        study._path.parent,
        *(pathlib.Path(source["path"]) for source in record["sources"].values()),
    ]
    if any(root.is_relative_to(path.resolve()) for path in protected) or any(
        any(
            (parent / marker).exists()
            for marker in (
                "benchmark.json",
                "selection.json",
                "artifact_manifest.json",
                "metadata/run.json",
                "metadata/attempt.json",
            )
        )
        for parent in (root, *root.parents)
    ):
        raise HPOError("output root overlaps frozen inputs or retained run artifacts")
    return root


def tune_study(
    study: Study,
    *,
    fit_predict: Callable[[FitPredictRequest], FitPredictResult],
    config: str | pathlib.Path,
    implementation: Mapping[str, object] | None = None,
    tuning_origin: str | None = None,
    output_dir: str | pathlib.Path | None = None,
    reuse_tuning: str | pathlib.Path | None = None,
) -> TuningResult:
    """Select one parameter configuration across frozen windows, or verify a completed search."""
    from autoforecast.study import load_study

    if not callable(fit_predict):
        raise TypeError("fit_predict must be a Python callable")
    if tuning_origin is not None and (type(tuning_origin) is not str or not tuning_origin.strip()):
        raise ValueError("tuning_origin must be a nonblank frozen origin string")
    implementation_record = (
        _implementation_record(implementation) if implementation is not None else None
    )
    raw, tuning, budget, seed, controls = _config(pathlib.Path(config))
    scope_bytes = getattr(study, "_implementation_scope", None) if implementation is None else None
    implementation = resolve_implementation(
        implementation, json.loads(scope_bytes) if scope_bytes else None,
        fit_predict, tuning.model_id,
    )
    if implementation_record is None:
        implementation_record = _implementation_record(implementation)
    capture = implementation if isinstance(implementation, CapturedImplementation) else None
    options = _resolve_options(study, controls, tuning_origin)
    root = _output_root(study, output_dir, "study-tuning")
    preflight = preflight_study(study, options, output_dir=root / "preflight")
    with ExitStack() as stack:
        if preflight.study_id != study.study_id:
            study = stack.enter_context(
                load_study(
                    preflight.study_id,
                    registry_root=preflight.study_path.parent.parent,
                    spark=study._spark,
                    scratch_root=study._snapshots.name,
                )
            )
        record, recipe = study._record(data=True), study.recipe
        if reuse_tuning is not None:
            result = load_tuning_result(reuse_tuning)
            context = result.provenance
            if (
                context["study"]["sha256"] != study._sha256
                or context["profile_sha256"] != tuning.source_sha256
                or context["implementation"] != implementation_record
                or context.get("options") != options
            ):
                raise HPOError(
                    "completed search differs from the requested study, "
                    "configuration or implementation"
                )
            note("Tuning reuse verified", search_path=str(result.path),
                 selected_parameters=result.parameters, objective_score=result.tuning_score)
            return result
        note("Tuning work", trial_budget=budget, windows=len(options["windows"]),
             maximum_callback_fits=budget * len(options["windows"]),
             retained_series=study.n_series, objective=study.recipe["metric"])
        temporary = pathlib.Path(
            stack.enter_context(tempfile.TemporaryDirectory(
                prefix="autoforecast-inputs-", dir=study._snapshots.name,
            ))
        )
        inputs = tuple(
            _prepare_window(study, options["stage"], w["origin"], w["horizon"], temporary / str(i))
            for i, w in enumerate(options["windows"])
        )
        contexts = [json.loads(item.context_bytes) for item in inputs]
        first = contexts[0]
        context = {
            "schema": "autoforecast.external-tuning.v2",
            "library": library_revision(),
            "study": first["evaluation_contract"]["study"],
            "study_id": record["study_id"],
            "model_id": tuning.model_id,
            "implementation": implementation_record,
            "profile_sha256": tuning.source_sha256,
            "metric": options["metric"],
            "options": options,
            "windows": contexts,
            "frequency": study.panel.frequency,
            "seed": seed,
            "max_trials": budget,
            "backend": "optuna",
            **first,
        }
        context["implementation_evidence"] = (
            capture.binding if capture is not None else {"mode": "declared"}
        )
        context_bytes = _json_bytes(context)
        frequency = study.panel.frequency
        test_dates = tuple(
            period_points(recipe["final_train_cutoff"], frequency, recipe["final_test_horizon"])
        )
        first_dates = tuple(period_points(first["origin"], frequency, first["horizon"]))
        last = contexts[-1]
        last_dates = tuple(period_points(last["origin"], frequency, last["horizon"]))
        test_range = [
            canonical_origin(test_dates[0], frequency),
            canonical_origin(test_dates[-1], frequency),
        ]
        train_range = [recipe["training_start"], first["origin"]]
        data = {
            "id": record["study_id"],
            "fingerprint": study._sha256,
            "observed_date_range": [recipe["training_start"], test_range[-1]],
            "training_date_range": train_range,
            "validation_date_range": train_range,
            "validation_target_date_range": [
                canonical_origin(first_dates[0], frequency),
                canonical_origin(last_dates[-1], frequency),
            ],
            "test_date_range": test_range,
            "test_target_date_range": test_range,
            "training_series": study.n_series,
            "validation_series": study.n_series,
            "test": "reserved_not_evaluated",
        }
        identifier = new_run_id("tune")
        spec = HPOStudySpec(
            id=identifier,
            backend="optuna",
            direction="minimize",
            objective="validation/objective",
            max_trials=budget,
            top_k=1,
            seed=seed,
            comparability_fingerprint=hashlib.sha256(context_bytes).hexdigest(),
            data=data,
            evaluator="external:fit_predict",
            tuning=tuning,
            source_path=pathlib.Path(config),
            source_sha256=tuning.source_sha256,
        )
        bridge = _ExternalTrialBridge(
            study._spark, fit_predict, inputs, context_bytes, raw, _json_bytes(preflight.report),
            capture,
        )
        search_path = root / "studies" / identifier
        try:
            run_hpo_study(spec, root, bridge)
        except Exception as error:
            # Preserve the callback's exception type while exposing the library-owned location.
            if getattr(error, "tuning_path", None) == search_path:
                from mlops.run_report import run_guide

                from autoforecast.protocol import _atomic_write_text

                _atomic_write_text(search_path / "RUN.md", run_guide(
                    identity={"search_id": identifier, "study_id": study.study_id},
                    status="failed",
                    files=[p.relative_to(search_path).as_posix()
                           for p in search_path.rglob("*") if p.is_file()],
                    sealed=False,
                ).decode())
            raise
        return load_tuning_result(search_path)


def _validate_workflow_configuration(
    root: pathlib.Path, record: dict, saved: dict, manifest: dict,
) -> None:
    """Require the summary and both fits to refer to the retained winner settings."""
    validation_only = record.get("schema") == "autoforecast.forecast-validation.v1"
    labels = (
        (("validation", "validation"),)
        if validation_only
        else (
            ("validation", "validation"),
            ("testing", "test"),
        )
    )
    if set(record) != {
        "schema",
        "status",
        "parameters",
        "provenance",
        "tuning_path",
        *[label for label, _ in labels],
    } or record["schema"] not in {
        "autoforecast.forecast-workflow.v1",
        "autoforecast.forecast-validation.v1",
    }:
        raise HPOError("unsupported workflow configuration record")
    profile_bytes = _retained_bytes(root, "metadata/search-config.yaml", manifest)
    if (
        profile_bytes != saved["profile_text"].encode("utf-8")
        or hashlib.sha256(profile_bytes).hexdigest() != saved["profile_sha256"]
    ):
        raise HPOError("workflow search configuration differs from its retained profile")
    parameters_sha = digest(record["parameters"])
    provenance = record["provenance"]
    if provenance.get("mode") == "fixed_configuration":
        if (
            saved.get("source")
            != {
                "kind": "fixed",
                "workflow_id": root.name,
                "study": provenance["study"],
            }
            or record["tuning_path"] is not None
        ):
            raise HPOError("fixed configuration differs from its originating workflow")
    if provenance != _retained_json(root, "metadata/provenance.json", manifest):
        raise HPOError("workflow configuration differs from its recorded execution provenance")
    if record["tuning_path"] is not None and (
        pathlib.Path(record["tuning_path"]).name != saved["source"]["search_id"]
    ):
        raise HPOError("workflow configuration names a different source search")
    for label, stage in labels:
        window = record[label]
        contract = window["evaluation_contract"]
        if (
            window != _retained_json(root, stage + "/result.json", manifest)
            or not window["boundary"].startswith(stage + "/")
            or window["boundary"] != contract["boundary"]
            or contract["study"] != provenance["study"]
            or window["forecast"]["adapter"] != saved["model_id"]
            or window["forecast"]["path"] != stage + "/forecasts"
            or window["model_path"] != stage + "/model"
        ):
            raise HPOError("workflow stage configuration differs from its retained evidence")
    if (
        record["status"]
        != record["validation" if validation_only else "testing"]["score"]["evaluation_status"]
    ):
        raise HPOError("workflow configuration status differs from testing coverage")
    if (
        parameters_sha != saved["parameter_sha256"]
        or parameters_sha != digest(saved["parameters"])
        or any(record[stage]["parameter_sha256"] != parameters_sha for stage, _ in labels)
        or provenance["implementation"] != saved["implementation"]
        or type(provenance["seed"]) is not int
        or provenance["seed"] != saved["seed"]
        or provenance["configuration_source"] != saved["source"]
    ):
        raise HPOError("workflow configuration differs from its winner or fitted stages")


def load_selected_configuration(path: str | pathlib.Path) -> dict:
    """Load the inert winner settings from a sealed trial without requiring its old data.

    Keep this file in its retained attempt directory so its artifact manifest can
    authenticate it. Use the returned settings through the workflow's selected_config.
    """
    source = pathlib.Path(path).absolute()
    if source.name != "winner-config.json" or source.parent.name != "metadata":
        raise HPOError("selected_config must name a retained trial's metadata/winner-config.json")
    attempt = source.parent.parent
    manifest_bytes = read_regular_bytes(attempt, "artifact_manifest.json")
    if (attempt / "metadata/run.json").is_file():
        from mlops.run_bundle import load_completed_run_bundle

        completion = load_completed_run_bundle(attempt)
        if completion["status"] != "completed" or completion["task"] not in {
            "forecast-workflow",
            "forecast-validation",
        }:
            raise HPOError("selected configuration requires a successful workflow")
        manifest = _retained_manifest(completion, manifest_bytes)
        saved = _retained_json(attempt, "metadata/winner-config.json", manifest)
        result = _retained_json(attempt, "result.json", manifest)
        expected_schema = (
            "autoforecast.forecast-validation.v1"
            if completion["task"] == "forecast-validation"
            else "autoforecast.forecast-workflow.v1"
        )
        if result.get("schema") != expected_schema:
            raise HPOError("workflow task and configuration schema differ")
        _validate_workflow_configuration(attempt, result, saved, manifest)
        metadata = {
            "hyperparameters": result["parameters"],
            "model": {"id": saved["model_id"]},
        }
    else:
        _, completion = load_completed_attempt(attempt)
        manifest = _retained_manifest(completion, manifest_bytes)
        metadata = _retained_json(attempt, "metadata/attempt.json", manifest)
        saved = _retained_json(attempt, "metadata/winner-config.json", manifest)
        search = attempt.parents[3]
        selection = load_study_selection(search)
        if not any(
            search / item["attempt_path"] == attempt
            and item["artifact_manifest_sha256"] == completion["artifact_manifest_sha256"]
            for item in selection["selected"]
        ):
            raise HPOError("configuration belongs to an unselected trial")
    if (
        not isinstance(saved, dict)
        or saved.get("schema") != "autoforecast.selected-configuration.v1"
        or saved.get("parameters") != metadata["hyperparameters"]
        or saved.get("model_id") != metadata["model"]["id"]
        or saved.get("parameter_sha256") != digest(saved.get("parameters"))
    ):
        raise HPOError("selected configuration differs from its sealed trial")
    if "study_id" in metadata:
        context = _retained_json(attempt, "evaluation/external-run.json", manifest)["context"]
        expected_source = {
            "study": context["study"],
            "search_id": metadata["study_id"],
            "trial_id": metadata["trial_id"],
            "attempt_id": metadata["attempt_id"],
        }
        if (
            saved["source"] != expected_source
            or saved["implementation"] != context["implementation"]
            or saved["seed"] != context["seed"]
            or saved["profile_sha256"] != context["profile_sha256"]
        ):
            raise HPOError("saved configuration source differs from its sealed trial")
    raw_profile = saved["profile_text"].encode("utf-8")
    if (
        hashlib.sha256(raw_profile).hexdigest() != saved["profile_sha256"]
        or loads_exact_yaml(raw_profile) != saved["profile"]
    ):
        raise HPOError("saved search settings differ from their original profile identity")
    profile = dict(saved["profile"])
    for key in ("hpo", "selection", "reporting", "series_policy", "minimum_history"):
        profile.pop(key, None)
    spec = loads_external_tuning_spec(_json_bytes(profile), source_path=source)
    materialize_runtime_parameters(spec, saved["parameters"])
    _implementation_record(saved["implementation"])
    return saved
