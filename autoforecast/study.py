"""Load frozen study data and check forecasts from Python."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING

from mlops.run_bundle import open_run_bundle
from mlops.run_report import new_run_id
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast.benchmark import (
    BenchmarkError,
    artifact_path,
    evaluation_contract,
    read_benchmark_manifest,
    resolve_study,
    verify_benchmark,
)
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.frequency import TimePoint, advance, parse_frequency, parse_origin
from autoforecast.grid import time_literal
from autoforecast.panel import Forecast, Panel
from autoforecast.project import project_root
from autoforecast.study_horizon_metrics import (
    ForecastAccuracyBreakdown,
    validate_forecast_accuracy_breakdown,
)
from autoforecast.study_scoring import score_study
from autoforecast.study_tuning import FitPredictRequest, FitPredictResult, TuningResult, tune_study
from provenance.lineage import directory_digest, snapshot_parquet_directory

if TYPE_CHECKING:
    from autoforecast.study_preflight import PreflightResult
    from autoforecast.study_workflow import ForecastWorkflowResult


def load_study(
    study_id: str, *, registry_root: str | pathlib.Path | None = None,
    spark: SparkSession | None = None,
    scratch_root: str | pathlib.Path | None = None,
) -> Study:
    """Load an existing ID from the project's .autoforecast/benchmarks registry.

    Use ``experiment_project`` once for the session, or supply ``registry_root``.
    Spark must be supplied or active. The returned context manager owns its private
    snapshots, never the caller's session. Close it after using its dataframes.
    ``scratch_root`` must be an existing filesystem directory visible to Spark
    workers. It controls operation-owned snapshots without changing global state.
    """
    registry = (pathlib.Path(registry_root) if registry_root is not None else
                project_root(pathlib.Path.cwd()) / ".autoforecast/benchmarks").absolute()
    try:
        path, _ = resolve_study(registry, study_id)
    except BenchmarkError as error:
        if (isinstance(error.__cause__, FileNotFoundError)
                and error.__cause__.filename == str(registry / study_id / "benchmark.json")):
            try:
                (registry / study_id).lstat()
            except FileNotFoundError:
                if registry.is_dir():
                    raise FileNotFoundError(f"Unknown study_id: {study_id}") from error
        raise
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    _, record = verify_benchmark(path, sha)
    if record["study_id"] != study_id:
        raise BenchmarkError("study identity changed while loading")
    if len(record["recipe"]["validation_origins"]) != 1:
        raise BenchmarkError("load_study requires exactly one validation origin")
    session = spark if spark is not None else SparkSession.getActiveSession()
    if session is None:
        raise ValueError("Supply a supported Spark session or activate one before load_study")
    return Study(path, sha, session, tempfile.TemporaryDirectory(
        prefix="autoforecast-study-", dir=scratch_root,
    ))


@dataclasses.dataclass(frozen=True)
class Study:
    """Read-only study settings and stage data; external training itself is not attested."""

    _path: pathlib.Path = dataclasses.field(repr=False)
    _sha256: str = dataclasses.field(repr=False)
    _spark: SparkSession = dataclasses.field(repr=False, compare=False)
    _snapshots: tempfile.TemporaryDirectory = dataclasses.field(repr=False, compare=False)
    _registration: bytes | None = dataclasses.field(default=None, repr=False)
    _implementation_scope: bytes | None = dataclasses.field(default=None, repr=False)
    _closed: bool = dataclasses.field(default=False, init=False, repr=False, compare=False)

    def __enter__(self) -> Study:
        try:
            self._record()
        except BaseException:
            with suppress(Exception):
                self.close()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        if _exc and _exc[0] is not None:
            with suppress(Exception):
                self.close()
        else:
            self.close()

    def close(self) -> None:
        """Release study snapshots; returned lazy dataframes must no longer be used."""
        if not self._closed:
            object.__setattr__(self, "_closed", True)
            self._snapshots.cleanup()

    def _record(self, *, data: bool = False) -> dict:
        if self._closed or not pathlib.Path(self._snapshots.name).is_dir():
            raise ValueError("Study is closed")
        verify = verify_benchmark if data else read_benchmark_manifest
        return verify(self._path, self._sha256)[1]

    @property
    def study_id(self) -> str:
        return self._record()["study_id"]

    @property
    def registration(self) -> dict | None:
        """A fresh view of the registry binding authenticated at load time, if any."""
        return json.loads(self._registration) if self._registration is not None else None

    @property
    def recipe(self) -> dict:
        """A fresh view of frozen settings; modifying this view does not modify the study."""
        return self._record()["recipe"]

    @property
    def policies(self) -> dict:
        return self._record()["policies"]

    @property
    def metric_definition(self) -> dict:
        return self._record()["metric_definition"]

    @property
    def n_series(self) -> int:
        return self._record()["cohort_series_count"]

    @property
    def panel(self) -> Panel:
        """Frozen column, cadence and target settings; use stage properties for data."""
        record = self._record()
        path = self._path.parent / "dataset.yaml"
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["dataset_sha256"]:
            raise BenchmarkError("dataset contract changed before study use")
        return _load_dataset_contract_bytes(payload, path).panel(
            artifact_path(self._path, record, "selection"),
        )

    def _range(self, stage: str) -> tuple[TimePoint, TimePoint]:
        recipe = self.recipe
        frequency = recipe["frequency"]
        origin = (recipe["validation_origins"][0] if stage in {"training", "validation"}
                  else recipe["final_train_cutoff"])
        end = parse_origin(origin, frequency)
        if stage in {"training", "refit"}:
            return parse_origin(recipe["training_start"], frequency), end
        horizon = recipe["validation_horizon" if stage == "validation" else "final_test_horizon"]
        return advance(end, frequency, 1), advance(end, frequency, horizon)

    @property
    def training_date_range(self) -> tuple[TimePoint, TimePoint]:
        return self._range("training")

    @property
    def validation_date_range(self) -> tuple[TimePoint, TimePoint]:
        return self._range("validation")

    @property
    def refit_date_range(self) -> tuple[TimePoint, TimePoint]:
        return self._range("refit")

    @property
    def testing_date_range(self) -> tuple[TimePoint, TimePoint]:
        return self._range("test")

    @property
    def forecasting_date_range(self) -> tuple[TimePoint, TimePoint]:
        return self.testing_date_range

    def _frame(self, role: str, stage: str | None = None) -> DataFrame:
        record = self._record(data=True)
        # Keep operation-owned bytes alive for Spark's deferred reads. Never return a
        # dataframe whose plan points at a mutable registry artifact.
        destination = pathlib.Path(self._snapshots.name) / role
        expected = record["artifacts"][role]
        if destination.exists():
            if directory_digest(destination, suffixes=(".parquet",)) != expected:
                raise BenchmarkError("study snapshot changed")
        else:
            snapshot_parquet_directory(
                artifact_path(self._path, record, role), destination,
                expected_sha256=expected, label=role,
            )
        frame = self._spark.read.parquet(str(destination))
        if "execution_policy" in record and role in {
            "selection", "final_training", "validation_score", "final_score",
        }:
            frame = frame.filter(F.col(self.panel.series_col).isin(
                record["execution_policy"]["retained_ids"],
            ))
        if stage is not None:
            start, end = self._range(stage)
            panel = self.panel
            cadence = parse_frequency(panel.frequency)
            frame = frame.filter(F.col(panel.time_col).between(
                time_literal(start, cadence), time_literal(end, cadence),
            ))
        if directory_digest(destination, suffixes=(".parquet",)) != expected:
            raise BenchmarkError("study snapshot changed")
        self._record(data=True)
        return frame

    @property
    def training_data(self) -> DataFrame:
        return self._fit_data("selection", "training", self.recipe["validation_origins"][0])

    @property
    def validation_data(self) -> DataFrame:
        return self._frame("validation_score", "validation")

    @property
    def refit_data(self) -> DataFrame:
        return self._fit_data("final_training", "refit", self.recipe["final_train_cutoff"])

    def _fit_data(self, role: str, stage: str, origin: str) -> DataFrame:
        from autoforecast.preparation import prepare_fit_history

        return prepare_fit_history(
            self._frame(role, stage), panel=self.panel, origin=origin,
            missing=self.recipe["history_missing"],
        )

    @property
    def testing_data(self) -> DataFrame:
        return self._frame("final_score", "test")

    @property
    def series_mapping(self) -> DataFrame:
        """Exact frozen internal IDs, source keys and declared history starts."""
        return self._frame("source-keys")

    @property
    def series_starts(self) -> DataFrame:
        return self.series_mapping

    @property
    def scoring_references(self) -> DataFrame | None:
        if "scoring_references" not in self._record()["artifacts"]:
            return None
        return self._frame("scoring_references")

    def tune(
        self, *, fit_predict: Callable[[FitPredictRequest], FitPredictResult],
        config: str | pathlib.Path, implementation: Mapping[str, object] | None = None,
        tuning_origin: str | None = None, output_dir: str | pathlib.Path | None = None,
        reuse_tuning: str | pathlib.Path | None = None,
    ) -> TuningResult:
        """Tune a Python model on selected frozen windows using sequential Optuna.

        Config contains an external model profile and explicit HPO budget/seed.
        Implementation records name, version, source_sha256 and dependency versions.
        The callback fits and saves predictions/model files; AutoForecast scores them.
        The sole inner window is the default; configure origins for multiple windows.
        selection.stage=validation uses outer validation; testing never selects settings.
        Callback code is trusted caller code, not sandboxed or attested.
        No final refit is performed here.
        output_dir overrides default storage; reuse_tuning verifies a completed search.
        """
        return tune_study(self, fit_predict=fit_predict, config=config,
                          implementation=implementation, tuning_origin=tuning_origin,
                          output_dir=output_dir, reuse_tuning=reuse_tuning)

    def preflight(
        self, *, config: str | pathlib.Path, tuning_origin: str | None = None,
        output_dir: str | pathlib.Path | None = None,
    ) -> PreflightResult:
        """Check all registered stages and retain a report before invoking any model."""
        from autoforecast.study_preflight import preflight_study
        from autoforecast.study_tuning import _config, _output_root, _resolve_options
        _, _, _, _, controls = _config(pathlib.Path(config))
        return preflight_study(
            self, _resolve_options(self, controls, tuning_origin),
            output_dir=_output_root(self, output_dir, "study-preflight"),
        )

    def run_forecast_workflow(
        self, *, fit_predict: Callable[[FitPredictRequest], FitPredictResult],
        implementation: Mapping[str, object] | None = None,
        config: str | pathlib.Path | None = None,
        tuning_origin: str | None = None, output_dir: str | pathlib.Path | None = None,
        reuse_tuning: str | pathlib.Path | None = None,
        selected_config: str | pathlib.Path | None = None,
        fixed_parameters: Mapping[str, object] | None = None,
        validation_only: bool = False,
        validation_run: str | pathlib.Path | None = None,
    ) -> ForecastWorkflowResult:
        """Tune, validate and refit, or use fixed/reused settings without HPO.

        reuse_tuning reopens an identical search. selected_config applies verified
        winning settings to a new study. Both modes retain provenance and artifacts.
        fixed_parameters={} uses profile defaults; overrides use full parameter names.
        Fixed mode requires config, performs two fits and no search. The profile's
        hpo block is optional in this mode; without it the recorded seed is 42.
        """
        from autoforecast.study_workflow import run_forecast_workflow
        return run_forecast_workflow(self, fit_predict=fit_predict, implementation=implementation,
            config=config, tuning_origin=tuning_origin, output_dir=output_dir,
            reuse_tuning=reuse_tuning, selected_config=selected_config,
            fixed_parameters=fixed_parameters, validation_only=validation_only,
            validation_run=validation_run)

    def run_campaign(self, *, models, output_dir=None, **execution) -> dict:
        """Compare declared external models through the existing campaign runner.

        Supply StudyModel declarations. Sequential is the default; local/cluster
        modes use bounded Ray model jobs with importable entry points.
        """
        from autoforecast.study_campaign import run_study_campaign
        return run_study_campaign(self, models=models, output_dir=output_dir, **execution)

    def evaluate_forecasts(
        self, *, validation_forecasts: Forecast, testing_forecasts: Forecast, model_id: str,
        forecast_accuracy_breakdown: ForecastAccuracyBreakdown = "per_step",
        output_dir: str | pathlib.Path | None = None,
    ) -> dict:
        """Evaluate validation and test forecasts against the frozen study.

        Save forecast snapshots, scores, coverage and input identities in a separate
        run directory. Return the uniquely identified result, including its saved path.

        ``forecast_accuracy_breakdown`` selects per-step leads (default), cumulative
        prefixes, both, or none. Leads follow each forecast's frequency and horizon.
        The full-horizon overall score is always calculated independently.

        Forecast Parquet uses ``series_id``, ``date`` and ``yhat``. IDs must match
        ``series_mapping``; dates use the study cadence. All frozen cells are required.
        This checks predictions and records input identities, not external training code.
        """
        validate_forecast_accuracy_breakdown(forecast_accuracy_breakdown)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-blank string")
        record = self._record(data=True)
        recipe = record["recipe"]
        # Capture both inputs before invoking either scorer or the caller's Spark runtime.
        forecasts = {stage: Forecast.from_record(value.as_record()) for stage, value in (
            ("validation", validation_forecasts), ("test", testing_forecasts),
        )}
        from autoforecast.study_tuning import _output_root

        runs = _output_root(self, output_dir, "study-runs")
        bundle = open_run_bundle(runs, run_id=new_run_id("evaluation"), task="forecast-evaluation")
        output = bundle.path
        try:
            stages = {}
            for stage, forecast in forecasts.items():
                origin = (recipe["validation_origins"][0] if stage == "validation"
                          else recipe["final_train_cutoff"])
                horizon = recipe[
                    "validation_horizon" if stage == "validation" else "final_test_horizon"
                ]
                if (forecast.origin, forecast.horizon, forecast.frequency) != (
                    origin, horizon, recipe["frequency"],
                ):
                    raise ValueError(f"{stage} forecast uses the wrong evaluation boundary")
                fixed = evaluation_contract(
                    self._path, stage=stage, origin=origin, horizon=horizon,
                    metric=recipe["metric"],
                )
                if fixed["study"]["sha256"] != self._sha256:
                    raise BenchmarkError("study changed before scoring")
                directory = output / stage
                directory.mkdir()
                sha = directory_digest(forecast.path, suffixes=(".parquet",))
                retained = snapshot_parquet_directory(
                    pathlib.Path(forecast.path), directory / "forecasts",
                    expected_sha256=sha, label=f"{stage} forecasts",
                )
                forecast = dataclasses.replace(forecast, path=str(retained))
                forecast.validate()
                forecast_bytes = (json.dumps(forecast.as_record(), sort_keys=True) + "\n").encode()
                forecast_path = directory / "forecast.json"
                forecast_path.write_bytes(forecast_bytes)
                forecasts[stage] = forecast
                stages[stage] = {
                    "model_id": model_id, "evaluation_contract": fixed,
                    "forecast": str(forecast_path),
                    "forecast_sha256": hashlib.sha256(forecast_bytes).hexdigest(),
                    "forecast_artifact_sha256": sha,
                }
            for stage, forecast in forecasts.items():
                evidence = stages[stage]
                fixed = evidence["evaluation_contract"]
                if (directory_digest(forecast.path, suffixes=(".parquet",))
                        != evidence["forecast_artifact_sha256"]):
                    raise BenchmarkError("retained study forecasts changed before scoring")
                # The single-forecast scorer snapshots all inputs before its first Spark call.
                # There is no comparison/ranking prepass that can reread mutable predictions.
                score = score_study(
                    dataclasses.replace(self.panel, path=fixed["actuals"]["path"]),
                    forecast, self._spark, contract=fixed, output_dir=output / stage / "scores",
                    forecast_accuracy_breakdown=forecast_accuracy_breakdown,
                    scratch_root=self._snapshots.name,
                )
                scoring = output / stage / "score.json"
                payload = (json.dumps(score, sort_keys=True, allow_nan=False) + "\n").encode()
                scoring.write_bytes(payload)
                evidence.update(
                    score=score,
                    status="unavailable" if score[recipe["metric"]] is None else "scored",
                    scoring=str(scoring), scoring_sha256=hashlib.sha256(payload).hexdigest(),
                )
            self._record(data=True)
            for stage_name, stage in stages.items():
                breakdown = stage["score"].get("forecast_accuracy_breakdown")
                if breakdown is not None:
                    evidence = breakdown["series_sums"]
                    if (directory_digest(evidence["path"], suffixes=(".parquet",))
                            != evidence["sha256"]):
                        raise BenchmarkError("retained study horizon metric evidence changed")
                for name in ("forecast", "scoring"):
                    if (hashlib.sha256(pathlib.Path(stage[name]).read_bytes()).hexdigest()
                            != stage[name + "_sha256"]):
                        raise BenchmarkError("retained study evidence changed during scoring")
                retained = output / stage_name
                series = stage["score"]["series_scores"]
                if (directory_digest(retained / "forecasts", suffixes=(".parquet",))
                        != stage["forecast_artifact_sha256"]
                        or directory_digest(series["path"], suffixes=(".parquet",))
                        != series["sha256"]):
                    raise BenchmarkError("retained study artifacts changed during scoring")
            result = {
                "schema": "autoforecast.study-check.v1", "study_id": record["study_id"],
                "study": {"path": str(self._path), "sha256": self._sha256},
                "model_id": model_id, "run_id": output.name,
                "verified_input_artifacts": record["artifacts"],
                "training_provenance": "unverified_external", "managed_campaign": False,
                **stages, "result_path": str(output / "result.json"),
            }
            (output / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )
            bundle.finalize(status="completed", summary={"study_id": record["study_id"]})
            return result
        except Exception as error:
            bundle.write_json("failure.json", {"error": str(error)})
            bundle.finalize(status="failed", summary={"study_id": record["study_id"]})
            raise
