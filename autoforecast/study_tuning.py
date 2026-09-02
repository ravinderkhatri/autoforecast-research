"""Caller-supplied models tuned on a frozen study's inner window."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast._exact_json import load_exact_json
from autoforecast._exact_yaml import loads_exact_yaml
from autoforecast.benchmark import evaluation_contract, verify_benchmark
from autoforecast.experiments import load_completed_attempt, load_study_selection
from autoforecast.frequency import (
    TimePoint,
    canonical_origin,
    parse_frequency,
    parse_origin,
    period_points,
)
from autoforecast.grid import time_literal
from autoforecast.hpo import HPOError, HPOStudySpec, TrialOutcome, TrialRequest, run_hpo_study
from autoforecast.hpo_bridge import _copy_run_artifact_directory
from autoforecast.panel import Forecast, Panel
from autoforecast.study_scoring import score_study
from autoforecast.tuning import TuningError, TuningSpec, loads_external_tuning_spec
from provenance.lineage import directory_digest, snapshot_parquet_directory

if TYPE_CHECKING:
    from autoforecast.study import Study


@dataclasses.dataclass(frozen=True)
class FitPredictRequest:
    """Inputs for one trial, with no tuning, validation or test target actuals.

    ``panel.path`` contains only training rows through ``origin``. Save forecasts
    beneath ``output_dir`` and at least one model file beneath ``output_dir/model``.
    Apply the supplied model/training parameters and seed in the caller's code.
    The seed is held constant across trials to compare parameter configurations.
    Dataframes borrow the study's Spark session and are valid during the callback.
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
    """Verified selected inner-window result, independent of the live Study session.

    Parameters use the profile's full names (for example ``model.trend``).
    Reload with ``load_tuning_result(path)`` to recheck persisted evidence. This is
    the inner tuning model; later validation and final test stages need their own fits.
    No model deserialization or executable import happens when loading this result.
    """

    path: pathlib.Path
    parameters: dict[str, object]
    trial_id: str
    model_id: str
    tuning_score: float
    forecasts: Forecast
    model_artifacts: pathlib.Path
    model_format: str
    trial_history: tuple[dict, ...]
    provenance: dict


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _implementation_record(value: Mapping[str, object]) -> dict:
    """Capture declared code identity; this does not attest arbitrary caller Python."""
    if not isinstance(value, Mapping) or set(value) != {
        "name", "version", "source_sha256", "dependencies",
    }:
        raise ValueError("implementation requires name, version, source_sha256 and dependencies")
    if any(type(item) is not str or not item.strip()
           for item in (value["name"], value["version"])):
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


def _config(path: pathlib.Path) -> tuple[bytes, TuningSpec, int, int]:
    """Read once: profile settings and explicit budget; dates come only from the study."""
    if path.is_symlink() or not path.is_file():
        raise TuningError("external tuning config must be a regular file")
    raw = path.read_bytes()
    value = loads_exact_yaml(raw)
    if type(value) is not dict or "hpo" not in value:
        raise TuningError("external tuning config requires an hpo mapping")
    options = value.pop("hpo")
    if type(options) is not dict or set(options) != {"backend", "max_trials", "seed"}:
        raise TuningError("hpo requires exactly backend, max_trials and seed")
    if options["backend"] != "optuna":
        raise TuningError("Study.tune supports only sequential optuna execution")
    budget, seed = options["max_trials"], options["seed"]
    if type(budget) is not int or budget < 1:
        raise TuningError("hpo.max_trials must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise TuningError("hpo.seed must be an integer in [0, 2**32 - 1]")
    spec = loads_external_tuning_spec(_json_bytes(value), source_path=path)
    spec = dataclasses.replace(spec, source_sha256=hashlib.sha256(raw).hexdigest())
    return raw, spec, budget, seed


def load_tuning_result(path: str | pathlib.Path) -> TuningResult:
    """Verify all ranked attempts and reload the selected forecasts and model paths."""
    root = pathlib.Path(path).absolute()
    selection = load_study_selection(root)
    chosen = selection["selected"][0]
    attempt = root / chosen["attempt_path"]
    metadata, _ = load_completed_attempt(attempt)
    evidence = load_exact_json(attempt / "evaluation/external-run.json")
    if (type(evidence) is not dict
            or evidence.get("schema") != "autoforecast.external-trial.v1"):
        raise HPOError("selected attempt is not a supported external-model trial")
    context = evidence["context"]
    if hashlib.sha256(_json_bytes(context)).hexdigest() != selection["comparability_fingerprint"]:
        raise HPOError("external selection context differs from its frozen identity")
    profile = (attempt / "metadata/external-profile.yaml").read_bytes()
    if hashlib.sha256(profile).hexdigest() != context["profile_sha256"]:
        raise HPOError("external profile differs from its frozen identity")
    if (context["max_trials"] != len(selection["candidates"])
            or context["model_id"] != metadata["model"]["id"]):
        raise HPOError("external selection budget or model differs from its context")
    forecast_record = cast(dict, load_exact_json(attempt / "evaluation/forecast.json"))
    forecast = Forecast.from_record(forecast_record)
    forecast = dataclasses.replace(forecast, path=str(attempt / "forecasts"))
    forecast.validate()
    if directory_digest(forecast.path, suffixes=(".parquet",)) != evidence["forecast_sha256"]:
        raise HPOError("selected forecast bytes changed")
    model = attempt / "model"
    if directory_digest(model) != evidence["model_sha256"]:
        raise HPOError("selected model bytes changed")
    return TuningResult(
        path=root, parameters=dict(metadata["hyperparameters"]), trial_id=chosen["trial_id"],
        model_id=context["model_id"], tuning_score=chosen["validation_objective"],
        forecasts=forecast, model_artifacts=model, model_format=evidence["model_format"],
        trial_history=tuple(selection["candidates"]), provenance=context,
    )


@dataclasses.dataclass(frozen=True)
class _ExternalTrialBridge:
    """Own scoring and publication; caller code receives only fitting inputs."""

    spark: SparkSession
    callback: Callable[[FitPredictRequest], FitPredictResult]
    panel: Panel
    mapping_path: pathlib.Path
    context_bytes: bytes
    profile_bytes: bytes

    def _verify_inputs(self, context: dict) -> None:
        identity = context["study"]
        verify_benchmark(pathlib.Path(identity["path"]), identity["sha256"])
        for key, path in (("training_sha256", self.panel.path),
                          ("mapping_sha256", self.mapping_path)):
            if directory_digest(path, suffixes=(".parquet",)) != context[key]:
                raise HPOError("external tuning input snapshot changed")

    def __call__(self, request: TrialRequest) -> TrialOutcome:
        context = json.loads(self.context_bytes)
        self._verify_inputs(context)
        origin, horizon, frequency = context["origin"], context["horizon"], self.panel.frequency
        # The callback gets its own parameter objects and panel, never HPO's live spec/attempt.
        with tempfile.TemporaryDirectory(prefix="autoforecast-fit-") as temporary:
            work = pathlib.Path(temporary)
            result = self.callback(FitPredictRequest(
                training_data=self.spark.read.parquet(self.panel.path),
                series_mapping=self.spark.read.parquet(str(self.mapping_path)),
                panel=Panel.from_record(dataclasses.asdict(self.panel)), origin=origin,
                forecast_dates=tuple(period_points(origin, frequency, horizon)),
                model_id=context["model_id"],
                model_parameters=json.loads(_json_bytes(request.model_parameters)),
                training_parameters=json.loads(_json_bytes(request.training_parameters)),
                seed=context["seed"], output_dir=work,
            ))
            self._verify_inputs(context)
            if type(result) is not FitPredictResult or type(result.model_format) is not str \
                    or not result.model_format.strip():
                raise HPOError("fit_predict must return FitPredictResult with a model_format")
            forecast = Forecast.from_record(result.forecasts.as_record())
            if (forecast.origin, forecast.horizon, forecast.frequency, forecast.adapter) != (
                origin, horizon, frequency, context["model_id"],
            ):
                raise HPOError("external forecast uses the wrong model or tuning boundary")
            if not pathlib.Path(forecast.path).resolve().is_relative_to(work.resolve()):
                raise HPOError("external forecasts must be saved beneath output_dir")
            model_source = work / "model"
            if not model_source.is_dir():
                raise HPOError("external trial requires saved model files beneath output_dir/model")
            model_sha = directory_digest(model_source)
            if not any(p.is_file() for p in model_source.rglob("*")):
                raise HPOError("external trial requires saved model files beneath output_dir/model")
            _copy_run_artifact_directory(model_source, request.attempt.path / "model")
            if directory_digest(request.attempt.path / "model") != model_sha:
                raise HPOError("external model changed while retaining artifacts")
            forecast_sha = directory_digest(forecast.path, suffixes=(".parquet",))
            retained = snapshot_parquet_directory(
                pathlib.Path(forecast.path), request.attempt.path / "forecasts",
                expected_sha256=forecast_sha, label="external forecasts",
            )
            forecast = dataclasses.replace(forecast, path=str(retained))
            forecast.validate()
        fixed = context["evaluation_contract"]
        score = score_study(
            dataclasses.replace(self.panel, path=fixed["actuals"]["path"]), forecast, self.spark,
            contract=fixed, output_dir=request.attempt.path / "scores",
        )
        value = score[context["metric"]]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise HPOError("external trial requires a finite tuning objective")
        self._verify_inputs(context)
        request.attempt.write_bytes("metadata/external-profile.yaml", self.profile_bytes)
        request.attempt.write_json("evaluation/forecast.json", {
            **forecast.as_record(), "path": "forecasts",
        })
        request.attempt.write_json("evaluation/score.json", score)
        request.attempt.write_json("evaluation/external-run.json", {
            "schema": "autoforecast.external-trial.v1", "context": context,
            "forecast_sha256": forecast_sha, "model_sha256": model_sha,
            "model_format": result.model_format, "training_attestation": "caller_supplied",
            "test_metrics": "not_evaluated",
        })
        return TrialOutcome(float(value), {"validation/" + context["metric"]: float(value)})


def tune_study(
    study: Study, *, fit_predict: Callable[[FitPredictRequest], FitPredictResult],
    config: str | pathlib.Path, implementation: Mapping[str, object],
    tuning_origin: str | None = None,
) -> TuningResult:
    """Tune one caller-supplied implementation on one explicitly frozen inner window."""
    if tuning_origin is not None and (type(tuning_origin) is not str or not tuning_origin):
        raise ValueError("tuning_origin must be a nonblank frozen origin string")
    if not callable(fit_predict):
        raise TypeError("fit_predict must be a Python callable")
    implementation_record = _implementation_record(implementation)
    raw, tuning, budget, seed = _config(pathlib.Path(config))
    record = study._record(data=True)
    recipe = record["recipe"]
    windows = recipe.get("tuning_windows", [])
    if not windows:
        raise HPOError("Study.tune requires tuning_windows declared before freezing the study")
    if tuning_origin is None and len(windows) != 1:
        raise HPOError("choose tuning_origin explicitly when a study has multiple tuning windows")
    window = next(
        (w for w in windows if tuning_origin is None or w["origin"] == tuning_origin), None,
    )
    if window is None:
        raise HPOError("tuning_origin is absent from the frozen study")
    origin, horizon = window["origin"], window["horizon"]
    fixed = evaluation_contract(study._path, stage="tuning", origin=origin, horizon=horizon,
                                metric=recipe["metric"])
    if fixed["study"]["sha256"] != study._sha256:
        raise HPOError("study changed before tuning")
    panel = study.panel
    cadence = parse_frequency(panel.frequency)
    dates = tuple(period_points(origin, cadence, horizon))
    identifier = "tune-" + uuid.uuid4().hex
    root = study._path.parent.parent.parent / "study-tuning"
    inputs = root / "inputs" / identifier
    inputs.mkdir(parents=True)
    training = inputs / "training"
    mapping = inputs / "series-mapping"
    # Materialize the slice: a lazy filter alone still points at future target rows.
    study.training_data.filter(F.col(panel.time_col) <= time_literal(
        parse_origin(origin, cadence), cadence,
    )).write.parquet(str(training))
    study.series_mapping.write.parquet(str(mapping))
    study._record(data=True)
    panel = dataclasses.replace(panel, path=str(training))
    context = {
        "schema": "autoforecast.external-tuning.v1", "study": fixed["study"],
        "study_id": record["study_id"], "evaluation_contract": fixed,
        "model_id": tuning.model_id, "implementation": implementation_record,
        "profile_sha256": tuning.source_sha256, "metric": recipe["metric"],
        "origin": origin, "horizon": horizon, "frequency": panel.frequency,
        "seed": seed, "max_trials": budget, "backend": "optuna",
        "training_sha256": directory_digest(training, suffixes=(".parquet",)),
        "mapping_sha256": directory_digest(mapping, suffixes=(".parquet",)),
    }
    context_bytes = _json_bytes(context)
    # Existing HPO metadata calls inner targets "validation". The recorded contract
    # explicitly identifies them as tuning, and reserves the final test dates untouched.
    test_dates = tuple(period_points(recipe["final_train_cutoff"], cadence,
                                    recipe["final_test_horizon"]))
    def iso(point):
        return canonical_origin(point, cadence)
    train_range = [recipe["training_start"], origin]
    test_range = [iso(test_dates[0]), iso(test_dates[-1])]
    data = {
        "id": record["study_id"], "fingerprint": record["artifacts"]["selection"],
        "observed_date_range": [recipe["training_start"], test_range[-1]],
        "training_date_range": train_range, "validation_date_range": train_range,
        "validation_target_date_range": [iso(dates[0]), iso(dates[-1])],
        "test_date_range": test_range, "test_target_date_range": test_range,
        "training_series": record["cohort_series_count"],
        "validation_series": record["cohort_series_count"], "test": "reserved_not_evaluated",
    }
    spec = HPOStudySpec(
        id=identifier, backend="optuna", direction="minimize", objective="validation/objective",
        max_trials=budget, top_k=1, seed=seed,
        comparability_fingerprint=hashlib.sha256(context_bytes).hexdigest(),
        data=data, evaluator="external:fit_predict", tuning=tuning,
        source_path=pathlib.Path(config), source_sha256=tuning.source_sha256,
    )
    bridge = _ExternalTrialBridge(study._spark, fit_predict, panel, mapping, context_bytes, raw)
    run_hpo_study(spec, root, bridge)
    return load_tuning_result(root / "studies" / identifier)
