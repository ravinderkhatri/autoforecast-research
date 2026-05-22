"""Built-in bridge from a frozen HPO study to the governed forecast runner.

The bridge is intentionally narrow: it tunes one profile with a declared typed execution plan and
uses the run-request's declared validation origin/horizon. It never scores the study's reserved
test targets during selection.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import shutil
from dataclasses import dataclass, field
from typing import cast

from mlops import ProvenanceError, load_completed_run_bundle, validate_data_reference

from autoforecast._exact_json import ExactJSONError, load_exact_json
from autoforecast._numeric import is_finite_real
from autoforecast.data_admission import DatasetContractError, load_dataset_contract
from autoforecast.hpo import HPOError, HPOStudySpec, TrialOutcome, TrialRequest
from autoforecast.model_catalog import Candidate
from autoforecast.model_execution import (
    ExecutionPlanError,
    resolve_revision_rolling_config,
    validate_revision_parameter_groups,
)
from autoforecast.panel import PanelError, period_days
from autoforecast.train_workflow import RunRequest, load_run_request
from provenance.lineage import directory_digest

__all__ = ["ForecastTrialBridge"]


def _load_evidence_record(path: pathlib.Path, label: str) -> dict[str, object]:
    """Load one claim-bearing JSON object without accepting ambiguous JSON."""
    try:
        value = load_exact_json(path)
    except ExactJSONError as error:
        raise HPOError(f"{label} is unreadable: {error}") from error
    if type(value) is not dict:
        raise HPOError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _same_int(value: object, expected: int) -> bool:
    """Compare claim-bearing integer evidence without treating booleans as integers."""
    return type(value) is int and value == expected


def _require_exact_validation_population(
    comparison: dict[str, object],
    forecast_record: dict[str, object],
    *,
    model_id: str,
    metric: str,
    validation_series: int,
    horizon: int,
) -> dict[str, object]:
    """Return the score only when every trial is evaluated on the same frozen grid."""
    candidates = comparison.get("candidates")
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and item.get("model_id") == model_id
        ),
        None,
    ) if isinstance(candidates, list) else None
    ranking = comparison.get("ranking")
    if type(candidate) is not dict or type(ranking) is not dict:
        raise HPOError("governed trial score does not cover the frozen validation population")
    candidate_record = cast(dict[str, object], candidate)
    ranking_record = cast(dict[str, object], ranking)
    score_value = candidate_record.get("score")
    if type(score_value) is not dict:
        raise HPOError("governed trial score does not cover the frozen validation population")
    score = cast(dict[str, object], score_value)
    expected_cells = validation_series * horizon
    exact = (
        _same_int(forecast_record.get("n_emitted"), expected_cells)
        and _same_int(forecast_record.get("n_declined"), 0)
        and candidates == [candidate_record]
        and candidate_record.get("status") == "scored"
        and candidate_record.get("detail") == ""
        and ranking_record.get("metric") == metric
        and _same_int(ranking_record.get("n_common_cells"), expected_cells)
        and ranking_record.get("order") == [model_id]
    )
    if metric == "wrmsse":
        scored_by_level = score.get("n_series_scored")
        dropped_by_level = score.get("n_series_dropped")
        exact = exact and (
            comparison.get("comparison_kind") == "m5_full_grid_wrmsse"
            and isinstance(scored_by_level, dict)
            and isinstance(dropped_by_level, dict)
            and all(
                type(count) is int and count > 0
                for count in scored_by_level.values()
            )
            and all(
                type(count) is int and count == 0
                for count in dropped_by_level.values()
            )
        )
    else:
        exact = exact and (
            comparison.get("comparison_kind") == "exploratory_common_cells"
            and _same_int(score.get("n_scored"), expected_cells)
            and _same_int(score.get("n_dropped_ineligible"), 0)
            and _same_int(score.get("n_dropped_no_data"), 0)
            and type(zero_filled := score.get("n_zero_filled")) is int
            and 0 <= zero_filled <= expected_cells
        )
    if not exact:
        raise HPOError("governed trial score does not cover the frozen validation population")
    return score


def _objective_value(score: dict[str, object], metric: str) -> float:
    value = score.get(metric)
    if not is_finite_real(value):
        raise HPOError(f"governed trial did not emit validation {metric}")
    return float(value)


def _published_run_groups(output_root: pathlib.Path) -> list[pathlib.Path]:
    """Return governed runs without the isolated worker's private scratch directory."""
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name != ".autoforecast-worker-scratch"
    )


@dataclass(frozen=True)
class ForecastTrialBridge:
    """Runtime-only inputs for an HPO study's built-in, data-agnostic executor."""

    request_path: pathlib.Path
    data_path: pathlib.Path
    data_snapshot_id: str
    data_reference: str
    epochs: int
    timeout_seconds: int
    project_root: pathlib.Path
    candidate: Candidate | None = None
    patience: int = 5
    device: str | None = None
    precision: str | None = None
    batch_size: int = 4096
    gradient_microbatch_size: int | None = None
    early_stopping_absolute_min_delta: float = 1e-6
    early_stopping_relative_min_delta: float = 0.0
    training_data_fingerprint: str | None = None
    evaluation_data_path: pathlib.Path | None = None
    evaluation_data_snapshot_id: str | None = None
    evaluation_data_reference: str | None = None
    evaluation_data_fingerprint: str | None = None
    source_data_path: pathlib.Path = field(kw_only=True)
    source_data_fingerprint: str = field(kw_only=True)

    def __post_init__(self) -> None:
        for name in ("epochs", "timeout_seconds", "patience", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise HPOError(f"builtin HPO {name} must be a positive integer")
        if self.gradient_microbatch_size is not None and (
            type(self.gradient_microbatch_size) is not int
            or self.gradient_microbatch_size < 1
            or self.gradient_microbatch_size > self.batch_size
        ):
            raise HPOError(
                "builtin HPO gradient_microbatch_size must be positive and no larger "
                "than batch_size"
            )
        if self.device is not None and (
            type(self.device) is not str or self.device not in {"cpu", "cuda", "mps"}
        ):
            raise HPOError("builtin HPO device must be cpu, cuda, mps, or null")
        if self.precision is not None and self.precision not in ("32-true", "16-mixed"):
            raise HPOError("builtin HPO precision must be 32-true, 16-mixed, or null")
        for name in (
            "early_stopping_absolute_min_delta",
            "early_stopping_relative_min_delta",
        ):
            value = getattr(self, name)
            if (
                not is_finite_real(value)
                or float(value) < 0
            ):
                raise HPOError(f"builtin HPO {name} must be a finite non-negative number")
        try:
            validate_data_reference(self.data_reference)
            if self.evaluation_data_reference is not None:
                validate_data_reference(self.evaluation_data_reference)
        except ProvenanceError as error:
            raise HPOError(f"builtin HPO data reference is unsafe: {error}") from error
        score_inputs = (
            self.evaluation_data_path,
            self.evaluation_data_snapshot_id,
            self.evaluation_data_reference,
        )
        if any(value is not None for value in score_inputs) and not all(
            value is not None for value in score_inputs
        ):
            raise HPOError(
                "builtin HPO evaluation data path, snapshot ID, and reference are atomic"
            )
        if (
            not isinstance(self.source_data_path, pathlib.Path)
            or type(self.source_data_fingerprint) is not str
            or len(self.source_data_fingerprint) != 64
            or set(self.source_data_fingerprint) - set("0123456789abcdef")
        ):
            raise HPOError("builtin HPO requires a source data path and fingerprint")

    def _request(self, study: HPOStudySpec) -> RunRequest:
        request = load_run_request(self.request_path)
        if request.models != (study.tuning.model_id,):
            raise HPOError(
                "builtin HPO requires the run request to select exactly the study model "
                f"{study.tuning.model_id!r}"
            )
        if request.evaluation is None:
            raise HPOError("builtin HPO requires explicit run.evaluation semantics")
        target_range = study.data["validation_target_date_range"]
        if not isinstance(target_range, list):  # validated by load_hpo_study
            raise HPOError("study validation target range is invalid")
        start, end = (dt.date.fromisoformat(item) for item in target_range)
        origin = dt.date.fromisoformat(request.origin)
        try:
            contract = load_dataset_contract(self.project_root / request.dataset_contract)
            step_days = period_days(str(contract.panel_roles.get("frequency", "D")))
        except (DatasetContractError, PanelError) as error:
            raise HPOError(f"builtin HPO dataset cadence is invalid: {error}") from error
        if (
            origin + dt.timedelta(days=step_days) != start
            or origin + dt.timedelta(days=step_days * request.horizon) != end
        ):
            raise HPOError(
                "run request origin/horizon must exactly produce study.validation_target_date_range"
            )
        return request

    def __call__(self, request: TrialRequest) -> TrialOutcome:
        run_definition = self._request(request.study)
        candidate = self.candidate
        if candidate is None:
            raise HPOError("builtin HPO execution requires a frozen model candidate")
        if (
            candidate.id != request.study.tuning.model_id
            or candidate.adapter != request.study.tuning.adapter
            or candidate.source_sha256 != request.study.tuning.source_sha256
        ):
            raise HPOError("builtin HPO candidate differs from the frozen tuning profile")
        if candidate.adapter in {
            "sft",
            "hmfnet",
            "pact",
        }:
            try:
                model_parameters, training_parameters = validate_revision_parameter_groups(
                    candidate,
                    request.model_parameters,
                    request.training_parameters,
                )
                resolve_revision_rolling_config(
                    candidate,
                    model_parameters=model_parameters,
                    training_parameters=training_parameters,
                    epochs=self.epochs,
                    seed=0,
                )
            except ExecutionPlanError as error:
                raise HPOError(
                    f"builtin HPO trial cannot form a valid execution recipe: {error}"
                ) from error
        if (
            self.evaluation_data_path is None
            or self.evaluation_data_snapshot_id is None
            or self.evaluation_data_reference is None
        ):
            raise HPOError(
                "builtin HPO execution requires physically separate training and validation data"
            )
        observed_training_fingerprint = directory_digest(self.data_path, suffixes=(".parquet",))
        observed_validation_fingerprint = directory_digest(
            self.evaluation_data_path, suffixes=(".parquet",)
        )
        observed_source_fingerprint = directory_digest(
            self.source_data_path, suffixes=(".parquet",)
        )
        if (
            (
                self.training_data_fingerprint is not None
                and self.training_data_fingerprint != observed_training_fingerprint
            )
            or (
                self.evaluation_data_fingerprint is not None
                and self.evaluation_data_fingerprint != observed_validation_fingerprint
            )
            or self.source_data_fingerprint != observed_source_fingerprint
            or request.study.data.get("fingerprint") != observed_source_fingerprint
        ):
            raise HPOError("builtin HPO materialized data changed before trial execution")
        # Import at execution time so an HPO metadata read does not acquire the Spark runner.
        from scripts.run_autoforecast import run_request

        output_root = request.attempt.path / "source-run"
        exit_code = run_request(
            self.request_path,
            data_path=self.data_path,
            data_snapshot_id=self.data_snapshot_id,
            data_reference=self.data_reference,
            output_root=output_root,
            epochs=self.epochs,
            timeout_seconds=self.timeout_seconds,
            patience=self.patience,
            device=self.device,
            precision=self.precision,
            batch_size=self.batch_size,
            gradient_microbatch_size=self.gradient_microbatch_size,
            early_stopping_absolute_min_delta=self.early_stopping_absolute_min_delta,
            early_stopping_relative_min_delta=self.early_stopping_relative_min_delta,
            parameter_overrides={
                **{f"model.{name}": value for name, value in request.model_parameters.items()},
                **{
                    f"training.{name}": value for name, value in request.training_parameters.items()
                },
            },
            expected_candidate=candidate,
            # The HPO attempt owns the live MLflow record; nested generic exports duplicate it.
            mlflow_enabled=False,
            evaluation_data_path=self.evaluation_data_path,
            evaluation_data_snapshot_id=self.evaluation_data_snapshot_id,
            evaluation_data_reference=self.evaluation_data_reference,
            # A study can execute many trials. Its caller may emit one study-level notice, but
            # a trial must never produce a Slack/email burst by inheriting the ordinary CLI sink.
            notifications_enabled=False,
        )
        runs = _published_run_groups(output_root)
        if exit_code != 0 or len(runs) != 1:
            raise HPOError(
                _incomplete_run_message(
                    output_root, request.study.tuning.model_id, exit_code, len(runs)
                )
            )
        run = runs[0]
        completed = sorted(
            (run / "attempts" / request.study.tuning.model_id).glob("*/completion.json")
        )
        comparison_path = run / "evaluation" / "candidate-comparison.json"
        if len(completed) != 1 or not comparison_path.is_file():
            raise HPOError(
                "governed trial did not emit a completed bundle and validation comparison"
            )
        bundle = completed[0].parent
        try:
            completion = load_completed_run_bundle(bundle)
        except ProvenanceError as error:
            raise HPOError(f"governed trial bundle failed verification: {error}") from error
        if completion.get("status") != "completed":
            raise HPOError("governed trial bundle did not complete successfully")
        data_record = _load_evidence_record(
            bundle / "provenance" / "data.json", "governed trial data identity"
        )
        data_check = _load_evidence_record(
            bundle / "checks" / "data.json", "governed trial data check"
        )
        facts = data_check.get("facts")
        forecast_record = _load_evidence_record(
            bundle / "evaluation" / "forecast.json", "governed trial forecast"
        )
        score_record = _load_evidence_record(
            run / "evaluation" / "post-training-evaluation.json",
            "governed trial score evidence",
        )
        score_input = score_record.get("score_input") if isinstance(score_record, dict) else None
        declared = request.study.data
        if (
            data_record.get("fingerprint") != observed_training_fingerprint
            or not isinstance(facts, dict)
            or facts.get("as_of") != run_definition.origin
            or facts.get("data_range") != declared["training_date_range"]
            or facts.get("data_range") != declared["validation_date_range"]
            or not _same_int(facts.get("n_series"), cast(int, declared["training_series"]))
            or forecast_record.get("origin") != run_definition.origin
            or not _same_int(forecast_record.get("horizon"), run_definition.horizon)
            or not _same_int(
                forecast_record.get("n_series"), cast(int, declared["validation_series"])
            )
            or score_record.get("schema") != "autoforecast.post-training-evaluation.v1"
            or score_record.get("status") != "scored"
            or score_record.get("evidence_role") != "hpo_validation_selection"
            or not isinstance(score_input, dict)
            or score_input.get("fingerprint") != observed_validation_fingerprint
            or score_input.get("id") != self.evaluation_data_snapshot_id
            or self.data_snapshot_id != f"{declared['id']}-hpo-training"
            or self.evaluation_data_snapshot_id != f"{declared['id']}-hpo-validation"
            or observed_source_fingerprint != declared["fingerprint"]
        ):
            raise HPOError("governed trial split/count evidence differs from the frozen HPO study")
        for name in ("model", "metrics", "tensorboard"):
            source = bundle / name
            if source.is_dir():
                _copy_run_artifact_directory(source, request.attempt.path / name)
        comparison = _load_evidence_record(comparison_path, "governed trial comparison")
        evaluation = run_definition.evaluation
        if evaluation is None:  # checked by _request; retain the runtime invariant here.
            raise HPOError("builtin HPO requires explicit run.evaluation semantics")
        metric = evaluation.metric
        validation_series = declared["validation_series"]
        if type(validation_series) is not int:  # validated by load_hpo_study
            raise HPOError("frozen HPO validation series count is invalid")
        score = _require_exact_validation_population(
            comparison,
            forecast_record,
            model_id=request.study.tuning.model_id,
            metric=metric,
            validation_series=validation_series,
            horizon=run_definition.horizon,
        )
        objective = _objective_value(score, metric)
        request.attempt.write_json(
            "evaluation/governed-run.json",
            {
                "schema": "autoforecast.hpo-governed-run.v1",
                "run_id": run.name,
                "source_bundle": str(bundle.relative_to(request.attempt.path)),
                "metric": metric,
                "training_data_fingerprint": observed_training_fingerprint,
                "validation_data_fingerprint": observed_validation_fingerprint,
                "source_data_fingerprint": declared["fingerprint"],
                "test_metrics": "not_evaluated",
            },
        )
        return TrialOutcome(objective, {f"validation/{metric}": objective})


def _copy_run_artifact_directory(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Copy one governed artifact directory without overwriting trial-owned evidence.

    ``open_local_attempt`` creates empty structural directories such as ``model`` and ``metrics``.
    They contain no evidence yet, so replacing exactly those empty directories is safe. Any content
    means a caller or a previous operation has already written trial evidence and the bridge stops.
    """
    descendants = tuple(source.rglob("*")) if source.is_dir() and not source.is_symlink() else ()
    if source.is_symlink() or not source.is_dir() or any(item.is_symlink() for item in descendants):
        raise HPOError(f"governed run artifact source may not contain symlinks: {source}")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise HPOError(f"HPO trial artifact destination is already populated: {destination}")
        destination.rmdir()
    shutil.copytree(source, destination, dirs_exist_ok=False)


def _incomplete_run_message(
    output_root: pathlib.Path,
    model_id: str,
    exit_code: int,
    run_count: int,
) -> str:
    """Return one actionable HPO error without conflating a model incident with data readiness."""
    if run_count != 1:
        return (
            "governed trial did not create exactly one run folder: "
            f"exit_code={exit_code}, run_count={run_count}"
        )
    summary_path = next(output_root.glob("*/run-summary.json"), None)
    if summary_path is None:
        return f"governed trial exited {exit_code} without a run summary"
    try:
        summary = _load_evidence_record(summary_path, "governed trial run summary")
    except HPOError:
        return f"governed trial exited {exit_code}; run summary is unreadable"
    models = summary.get("models")
    if isinstance(models, list):
        candidate = next(
            (
                item
                for item in models
                if isinstance(item, dict) and item.get("model_id") == model_id
            ),
            None,
        )
        if isinstance(candidate, dict) and isinstance(candidate.get("summary"), str):
            return f"governed trial model {model_id!r} failed: {candidate['summary']}"
    return f"governed trial exited {exit_code}; see {summary_path}"
