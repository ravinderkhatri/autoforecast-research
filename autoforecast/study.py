"""Load frozen study data and check forecasts from Python."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import shutil
import tempfile

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
from provenance.lineage import directory_digest, snapshot_parquet_directory


def load_study(
    study_id: str, *, registry_root: str | pathlib.Path | None = None,
    spark: SparkSession | None = None,
) -> Study:
    """Load an existing ID from the project's .autoforecast/benchmarks registry.

    Use ``experiment_project`` once for the session, or supply ``registry_root``.
    Spark must be supplied or active. The returned context manager owns its private
    snapshots, never the caller's session. Close it after using its dataframes.
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
    return Study(path, sha, session, tempfile.TemporaryDirectory(prefix="autoforecast-study-"))


@dataclasses.dataclass(frozen=True)
class Study:
    """Read-only study settings and stage data; external training itself is not attested."""

    _path: pathlib.Path = dataclasses.field(repr=False)
    _sha256: str = dataclasses.field(repr=False)
    _spark: SparkSession = dataclasses.field(repr=False, compare=False)
    _snapshots: tempfile.TemporaryDirectory = dataclasses.field(repr=False, compare=False)
    _closed: bool = dataclasses.field(default=False, init=False, repr=False, compare=False)

    def __enter__(self) -> Study:
        try:
            self._record()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc) -> None:
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
        return self._frame("selection", "training")

    @property
    def validation_data(self) -> DataFrame:
        return self._frame("validation_score", "validation")

    @property
    def refit_data(self) -> DataFrame:
        return self._frame("final_training", "refit")

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

    def evaluate_forecasts(
        self, *, validation_forecasts: Forecast, testing_forecasts: Forecast, model_id: str,
        forecast_accuracy_breakdown: ForecastAccuracyBreakdown = "per_step",
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
        runs = self._path.parent.parent.parent / "study-runs"
        runs.mkdir(parents=True, exist_ok=True)
        output = pathlib.Path(tempfile.mkdtemp(prefix="run-", dir=runs))
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
            return result
        except BaseException:
            shutil.rmtree(output)
            raise
