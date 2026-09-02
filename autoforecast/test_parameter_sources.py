"""Case-table and tamper tests for governed campaign parameter sources."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import subprocess

import pytest
import yaml
from mlops.provenance import capture_runtime_identity, capture_source_identity
from mlops.run_bundle import open_run_bundle

from autoforecast.experiments import open_local_attempt, seal_study_selection
from autoforecast.hpo import _trial_metadata, load_hpo_study
from autoforecast.model_catalog import discover_candidates
from autoforecast.parameter_sources import (
    PARAMETER_EXECUTION_COMPATIBILITY,
    ParameterSourceError,
    _training_controls_are_valid,
    _training_controls_match_candidate,
    _validate_retained_attempt_data_references,
    default_parameter_resolution,
    load_parameter_resolution,
    manual_parameter_resolution,
    parameter_resolution_from_selection,
    parameter_resolution_sha256,
    validate_parameter_resolution,
    validate_parameter_source_capsule,
)


def test_training_controls_precision_schema_preserves_mft_and_rejects_tampering() -> None:
    legacy = {
        "schema": "autoforecast.training-controls.v1",
        "maximum_epochs": 2,
        "patience": 5,
        "device": "cpu",
        "batch_size": 32,
        "gradient_microbatch_size": None,
        "early_stopping_absolute_min_delta": 1e-6,
        "early_stopping_relative_min_delta": 0.0,
    }
    assert not _training_controls_are_valid(legacy)
    mixed = {
        **legacy,
        "schema": "autoforecast.training-controls.v2",
        "requested_precision": "16-mixed",
        "effective_precision": "16-mixed",
    }
    assert _training_controls_are_valid(mixed)
    assert _training_controls_are_valid({**mixed, "effective_precision": "native"})
    assert not _training_controls_are_valid({**legacy, "effective_precision": "16-mixed"})
    assert not _training_controls_are_valid({**mixed, "effective_precision": "32-true"})


@pytest.mark.parametrize("requested", ["16-mixed", "32-true"])
def test_hpo_precision_evidence_matches_candidate_execution_class(requested: str) -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    candidates = {candidate.adapter: candidate for candidate in discover_candidates(root / "configs/models")}
    base = {
        "schema": "autoforecast.training-controls.v2",
        "maximum_epochs": 2, "patience": 5, "device": "cpu", "batch_size": 2,
        "gradient_microbatch_size": None,
        "early_stopping_absolute_min_delta": 1e-6,
        "early_stopping_relative_min_delta": 0.0,
        "requested_precision": requested,
    }
    for adapter in (
        "mft", "sft",
        "hmfnet", "pact", "tft", "tft_q",
        "shallow_global",
    ):
        expected = requested if adapter != "shallow_global" else "native"
        assert _training_controls_match_candidate(
            {**base, "effective_precision": expected}, candidates[adapter]
        )
        wrong = "native" if expected != "native" else requested
        assert not _training_controls_match_candidate(
            {**base, "effective_precision": wrong}, candidates[adapter]
        )


def _revision_candidate_with_extra_parameter(model_id: str, parameter: str):
    root = pathlib.Path(__file__).resolve().parent.parent
    candidate = next(
        item for item in discover_candidates(root / "configs/models") if item.id == model_id
    )
    profile = yaml.safe_load(candidate.source_bytes)
    profile["hyperparameters"][parameter] = {
        "default": 1,
        "valid": {"kind": "int", "low": 1, "high": 2, "step": 1},
        "search": {"kind": "int", "low": 1, "high": 2, "step": 1},
    }
    payload = yaml.safe_dump(profile, sort_keys=False).encode()
    return dataclasses.replace(
        candidate,
        source_bytes=payload,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


@pytest.mark.parametrize("mode", ["defaults", "manual"])
def test_revision_parameter_sources_validate_complete_recipe_before_worker(mode: str) -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    candidate = next(
        item
        for item in discover_candidates(root / "configs/models")
        if item.id == "sft"
    )
    recipe = json.loads(candidate.implementation_config_bytes)
    recipe["gradient_clip"] = 0
    payload = (json.dumps(recipe, allow_nan=False) + "\n").encode()
    invalid = dataclasses.replace(
        candidate,
        implementation_config_bytes=payload,
        implementation_config_sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ParameterSourceError, match="valid execution recipe"):
        if mode == "defaults":
            default_parameter_resolution(invalid)
        else:
            manual_parameter_resolution(invalid, {"model.encoder_days": 28})


@pytest.mark.parametrize("mode", ["defaults", "manual"])
@pytest.mark.parametrize("model_id", ["sft", "hmfnet"])
@pytest.mark.parametrize("parameter", ["model.unused", "training.unused"])
def test_revision_parameter_sources_reject_declared_but_unconsumed_parameters(
    mode: str,
    model_id: str,
    parameter: str,
) -> None:
    candidate = _revision_candidate_with_extra_parameter(model_id, parameter)
    with pytest.raises(ParameterSourceError, match="typed executable settings"):
        if mode == "defaults":
            default_parameter_resolution(candidate)
        else:
            manual_parameter_resolution(candidate, {})


def _git(project: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _project(
    tmp_path: pathlib.Path,
    *,
    compatibility: str = PARAMETER_EXECUTION_COMPATIBILITY,
    purpose: str = "hpo_validation_selection",
    extra_evaluation: bool = False,
    source_commit: str | None = None,
    summary_validation_objective: float = 0.5,
    include_forecast_evidence: bool = True,
    include_runtime_evidence: bool = True,
    metric: str = "mae",
    retained_contract_suffix: bytes = b"",
    retained_request_suffix: bytes = b"",
    data_reference: str = "local://example-data/hpo-training",
    evaluation_data_reference: str = "local://example-data/hpo-validation",
    repository_origin: str | None = None,
    source_origin_url: object | None = None,
    source_relevant_paths: object | None = None,
    malformed_evidence: str | None = None,
    training_series: int = 2,
    zero_filled_count: object = 0,
    legacy_study_metadata: bool = False,
):
    project = tmp_path / "project"
    model_path = project / "configs/models/neural/tft/model.yaml"
    model_path.parent.mkdir(parents=True)
    model_path.write_text(
        """schema_version: 2
model:
  id: example_tft
  adapter: tft
  family: neural
  runtime: tft
  status: candidate
  tags: [development]
hyperparameters:
  model.hidden_size:
    default: 16
    valid: {kind: int, low: 8, high: 64, step: 8}
    search: {kind: categorical, values: [16, 32]}
  model.attention_head_size:
    default: 4
    valid: {kind: int, low: 1, high: 8, step: 1}
    search: {kind: categorical, values: [2, 4]}
  model.dropout:
    default: 0.1
    valid: {kind: float, low: 0.0, high: 0.9, log: false}
    search: {kind: float, low: 0.0, high: 0.3, log: false}
  training.learning_rate:
    default: 0.001
    valid: {kind: float, low: 0.000001, high: 1.0, log: false}
    search: {kind: float, low: 0.0001, high: 0.01, log: true}
""",
        encoding="utf-8",
    )
    study_path = project / "configs/hpo/example.yaml"
    study_path.parent.mkdir(parents=True)
    data_fingerprint = "d" * 64
    study_path.write_text(
        f"""schema_version: 1
study:
  id: example-study
  model_config: configs/models/neural/tft/model.yaml
  backend: optuna
  direction: minimize
  objective: validation/objective
  max_trials: 1
  top_k: 1
  seed: 7
  comparability_fingerprint: {'c' * 64}
  evaluator: builtin
  data:
    id: example-data
    fingerprint: {data_fingerprint}
    observed_date_range: ['2024-01-01', '2024-02-14']
    training_date_range: ['2024-01-01', '2024-02-08']
    validation_date_range: ['2024-01-01', '2024-02-08']
    test_date_range: ['2024-02-11', '2024-02-14']
    validation_target_date_range: ['2024-02-09', '2024-02-10']
    test_target_date_range: ['2024-02-13', '2024-02-14']
    training_series: {training_series}
    validation_series: 2
    test: reserved_not_evaluated
""",
        encoding="utf-8",
    )
    lockfile = project / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    contract_path = project / "configs/datasets/example.yaml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        "schema_version: 1\ndataset:\n  id: example-data\n  panel:\n"
        "    series_col: series_id\n    time_col: date\n    target_col: target\n"
        "    frequency: D\n    season_length: 7\n"
        "    target_estimand: observed_nonnegative\n", encoding="utf-8",
    )
    run_request_path = project / "configs/autoforecast/hpo.yaml"
    run_request_path.parent.mkdir(parents=True)
    run_request_path.write_text(
        "schema_version: 3\nrun:\n  precision: 32-true\n  name: hpo-example\n  models: [example_tft]\n",
        encoding="utf-8",
    )
    candidate = discover_candidates(project / "configs/models")[0]
    study = load_hpo_study(study_path, project_root=project)
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "test@example.invalid")
    _git(project, "config", "user.name", "Parameter Source Test")
    if repository_origin is not None:
        _git(project, "remote", "add", "origin", repository_origin)
    _git(project, "add", ".")
    _git(project, "commit", "-qm", "test fixture")
    root = tmp_path / "hpo"
    parameters = {
        "model.hidden_size": 32,
        "model.attention_head_size": 4,
        "model.dropout": 0.2,
        "training.learning_rate": 0.001,
    }
    metadata = _trial_metadata(study, parameters)
    metadata["purpose"] = purpose
    metadata["parameter_execution_compatibility"] = compatibility
    if legacy_study_metadata:
        metadata["study"].pop("top_k")
    attempt = open_local_attempt(
        root,
        study_id=study.id,
        trial_id="trial-0000",
        attempt_id="attempt-one",
        metadata=metadata,
    )
    attempt.write_json(
        "evaluation/validation.json",
        {
            "schema": "autoforecast.hpo-validation.v1",
            "objective": "validation/objective",
            "value": 0.5,
            "metrics": {f"validation/{metric}": 0.5},
            "test_metrics": "not_evaluated",
        },
    )
    bundle = open_run_bundle(
        attempt.path / "source-run/run/attempts/example_tft",
        run_id="one",
        task="forecast",
    )
    bundle.write_json(
        "metadata/resolved-run.json",
        {
            "schema": "autoforecast.resolved-run.v1",
            "model_id": candidate.id,
            "adapter": candidate.adapter,
            "origin": "2024-02-08",
            "horizon": 2,
            "settings": [],
            "runtime": candidate.runtime,
        },
    )
    bundle.write_json(
        "metadata/execution-plan.json",
        {
            "schema": "autoforecast.execution-plan.v1",
            "readiness_settings": {},
            "adapter_kwargs": {
                "max_epochs": 2,
                "hidden_size": 32,
                "attention_head_size": 4,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "seed": 7,
            },
            "effective_parameters": {
                **parameters,
                "data.horizon": 2,
                "training.epochs": 2,
                "training.seed": 7,
            },
        },
    )
    training_controls: dict[str, object] = {
            "schema": "autoforecast.training-controls.v2",
            "requested_precision": "32-true",
            "effective_precision": "32-true",
            "maximum_epochs": True if malformed_evidence == "boolean_control" else 2,
            "patience": 5,
            "device": "cpu",
            "batch_size": 4096,
            "gradient_microbatch_size": None,
            "early_stopping_absolute_min_delta": 1e-6,
            "early_stopping_relative_min_delta": 0.0,
    }
    bundle.write_json("metadata/training-controls.json", training_controls)
    bundle.write_json(
        "checks/data.json",
        {
            "schema_version": 2,
            "kind": "data_checks",
            "accepted": True,
            "model_id": None,
            "facts": {
                "as_of": "2024-02-08",
                "data_fingerprint": "e" * 64,
                "data_range": ["2024-01-01", "2024-02-08"],
                "n_series": (
                    True if malformed_evidence == "boolean_data_count" else training_series
                ),
            },
            "checks": [],
        },
    )
    source_identity = capture_source_identity(project).as_record()
    runtime_identity = capture_runtime_identity(project).as_record()
    if source_commit is not None:
        source_identity["commit"] = source_commit
    if source_origin_url is not None:
        source_identity["origin_url"] = source_origin_url
    if source_relevant_paths is not None:
        source_identity["relevant_paths"] = source_relevant_paths
    data_identity = {
        "schema": "mlops.data-snapshot.v1",
        "id": "example-data-hpo-training",
        "fingerprint": "e" * 64,
        "reference": data_reference,
        "version": None,
    }
    bundle.write_json("provenance/source.json", source_identity)
    bundle.write_json("provenance/data.json", data_identity)
    if include_runtime_evidence:
        bundle.write_json("provenance/runtime.json", runtime_identity)
        bundle.copy_file("provenance/runtime/uv.lock", lockfile)
    bundle.copy_file("provenance/config/model_profile.yaml", model_path)
    bundle.copy_file("provenance/config/dataset_contract.yaml", contract_path)
    bundle.copy_file("provenance/config/run_request.yaml", run_request_path)
    if retained_contract_suffix:
        retained_contract = bundle.path / "provenance/config/dataset_contract.yaml"
        retained_contract.write_bytes(retained_contract.read_bytes() + retained_contract_suffix)
    if retained_request_suffix:
        retained_request = bundle.path / "provenance/config/run_request.yaml"
        retained_request.write_bytes(retained_request.read_bytes() + retained_request_suffix)
    bundle.write_json(
        "provenance/manifest.json",
        {
            "schema": "mlops.run-provenance.v1",
            "source": source_identity,
            "runtime": runtime_identity if include_runtime_evidence else {},
            "data": data_identity,
            "configuration": {
                "dataset_contract": {
                    "path": "provenance/config/dataset_contract.yaml",
                    "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                },
                "model_profile": {
                    "path": "provenance/config/model_profile.yaml",
                    "sha256": candidate.source_sha256,
                },
                "run_request": {
                    "path": "provenance/config/run_request.yaml",
                    "sha256": hashlib.sha256(run_request_path.read_bytes()).hexdigest(),
                },
            },
        },
    )
    if include_forecast_evidence:
        nested_model = bundle.path / "model/model.bin"
        nested_model.parent.mkdir()
        nested_model.write_bytes(b"model")
        model_sha = hashlib.sha256(b"model").hexdigest()
        bundle.write_json(
            "provenance/model-artifacts.json",
            {
                "schema": "autoforecast.model-artifacts.v1",
                "files": [{"path": "model/model.bin", "sha256": model_sha}],
            },
        )
        forecast_dir = bundle.path / "evaluation/forecast"
        forecast_dir.mkdir(parents=True)
        (forecast_dir / "part-00000.parquet").write_bytes(b"forecast")
        bundle.write_json(
            "evaluation/forecast.json",
            {
                "adapter": candidate.adapter,
                "path": str(forecast_dir),
                "origin": "2024-02-08",
                "horizon": 2,
                "n_series": 2,
                "disposition": {"emitted": 4},
                "target_estimand": "sales_given_availability",
                "frequency": "D",
                "n_emitted": 4,
                "n_declined": False if malformed_evidence == "boolean_forecast_count" else 0,
            },
        )
    bundle.finalize(
        status="completed",
        summary={"forecast_path": str(bundle.path / "evaluation/forecast")}
        if include_forecast_evidence
        else {},
    )
    comparison_dir = attempt.path / "source-run/run/evaluation"
    comparison_dir.mkdir()
    (comparison_dir / "candidate-comparison.json").write_text(
        json.dumps(
            {
                "schema": "autoforecast.run-candidate-comparison.v1",
                "comparison_kind": (
                    "m5_full_grid_wrmsse" if metric == "wrmsse" else "exploratory_common_cells"
                ),
                "metric": metric,
                "missing_actual": "no_data",
                "comparison_error": None,
                "candidates": [
                    {
                        "model_id": candidate.id,
                        "status": "scored",
                        "detail": "",
                        "score": (
                            {
                                "wrmsse": 0.5,
                                "per_level": {
                                    "total": (
                                        10**400
                                        if malformed_evidence == "wrmsse_overflow"
                                        else 0.5
                                    ),
                                    "item": 0.5,
                                },
                                "n_series_scored": {"total": 1, "item": 2},
                                "n_series_dropped": {"total": 0, "item": 0},
                            }
                            if metric == "wrmsse"
                            else {
                                "bias": 0.0,
                                "mae": 0.5,
                                "n_dropped_ineligible": 0,
                                "n_dropped_no_data": (
                                    False
                                    if malformed_evidence == "boolean_score_count"
                                    else 0
                                ),
                                "n_scored": 4,
                                "n_zero_filled": zero_filled_count,
                                "rmse": 0.5,
                                "wape": 0.1,
                            }
                        ),
                    }
                ],
                "ranking": {"metric": metric, "n_common_cells": 4, "order": [candidate.id]},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if include_forecast_evidence:
        from provenance.lineage import directory_digest

        (comparison_dir / "post-training-evaluation.json").write_text(
            json.dumps(
                {
                    "schema": "autoforecast.post-training-evaluation.v1",
                    "status": "scored",
                    "evidence_role": "hpo_validation_selection",
                    "score_input": {
                        "schema": "mlops.data-snapshot.v1",
                        "id": "example-data-hpo-validation",
                        "fingerprint": "f" * 64,
                        "reference": evaluation_data_reference,
                        "version": None,
                    },
                    "forecast_artifacts": {
                        candidate.id: directory_digest(
                            bundle.path / "evaluation/forecast", suffixes=(".parquet",)
                        )
                    },
                    "comparison": "evaluation/candidate-comparison.json",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    attempt.write_json(
        "evaluation/governed-run.json",
        {
            "schema": "autoforecast.hpo-governed-run.v1",
            "run_id": "run",
            "source_bundle": "source-run/run/attempts/example_tft/one",
            "metric": metric,
            "training_data_fingerprint": "e" * 64,
            "validation_data_fingerprint": "f" * 64,
            "source_data_fingerprint": study.data["fingerprint"],
            "test_metrics": "not_evaluated",
        },
    )
    if extra_evaluation:
        attempt.write_json("evaluation/test-results.json", {"leaked": True})
    (attempt.path / "model/model.bin").write_bytes(b"model")
    attempt.finalize(
        status="completed",
        summary={
            "selection_metric": study.objective,
            "validation_objective": summary_validation_objective,
            "test_metrics": "not_evaluated",
        },
    )
    study_root = root / "studies" / study.id
    selection = seal_study_selection(
        study_root, top_k=1, direction="minimize", required_trial_count=1
    )
    if malformed_evidence == "boolean_rank":
        selection_record = json.loads(selection.read_text(encoding="utf-8"))
        selection_record["selected"][0]["rank"] = True
        selection.write_text(
            json.dumps(selection_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if malformed_evidence == "boolean_candidate_rank":
        selection_record = json.loads(selection.read_text(encoding="utf-8"))
        selection_record["candidates"][0]["rank"] = True
        selection.write_text(
            json.dumps(selection_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return project, candidate, study, selection, attempt.path


def test_defaults_and_partial_manual_values_resolve_to_complete_maps(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, _study, _selection, _attempt = _project(tmp_path)
    defaults = default_parameter_resolution(candidate)
    manual = manual_parameter_resolution(candidate, {"model.hidden_size": 40})
    assert validate_parameter_resolution(defaults, candidate) == defaults
    assert defaults["source"] == {"kind": "defaults"}
    assert manual["effective_parameters"]["model.hidden_size"] == 40
    assert manual["effective_parameters"]["model.dropout"] == 0.1
    path = tmp_path / "resolution.json"
    path.write_text(json.dumps(manual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert (
        load_parameter_resolution(
            path,
            parameter_resolution_sha256(manual),
            candidate,
            project_root=project,
        )
        == manual
    )
    with pytest.raises(ParameterSourceError, match="valid domain"):
        manual_parameter_resolution(candidate, {"model.hidden_size": 41})
    invalid = {**defaults, "model_id": "wrong-model"}
    with pytest.raises(ParameterSourceError, match="differs from its model profile"):
        validate_parameter_resolution(invalid, candidate)

    invalid_kind = {**defaults, "source": {"kind": []}}
    with pytest.raises(ParameterSourceError, match="source kind is unsupported"):
        validate_parameter_resolution(invalid_kind, candidate)


@pytest.mark.parametrize(
    ("source", "selection_record", "message"),
    [
        ({"kind": []}, {}, "not an HPO source"),
        (
            {"kind": "tune"},
            {
                "candidates": [{"rank": 1}],
                "selected": [
                    {
                        "rank": True,
                        "trial_id": "trial",
                        "attempt_id": "attempt",
                        "artifact_manifest_sha256": "a" * 64,
                    }
                ]
            },
            "selection and attempt disagree",
        ),
        (
            {"kind": "tune"},
            {
                "candidates": [{"rank": True}],
                "selected": [
                    {
                        "rank": 1,
                        "trial_id": "trial",
                        "attempt_id": "attempt",
                        "artifact_manifest_sha256": "a" * 64,
                    }
                ],
            },
            "selection and attempt disagree",
        ),
        (
            {"kind": "tune"},
            {
                "candidates": [{"rank": 1}],
                "selected": [
                    {
                        "rank": 1,
                        "trial_id": "trial",
                        "attempt_id": "attempt",
                        "artifact_manifest_sha256": "a" * 64,
                    },
                    {"rank": True},
                ],
            },
            "selection and attempt disagree",
        ),
    ],
)
def test_retained_capsule_rejects_unhashable_kind_or_malformed_ranks(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    source: dict[str, object],
    selection_record: dict[str, object],
    message: str,
) -> None:
    root = tmp_path / "bundle"
    capsule_root = root / "provenance/parameter-source"
    selection_payload = (json.dumps(selection_record) + "\n").encode()
    files = {
        "provenance/parameter-source/study-config.yaml": b"schema_version: 1\n",
        "provenance/parameter-source/study/selection.json": selection_payload,
        "provenance/parameter-source/study/attempt/evidence.json": b"{}\n",
    }
    artifacts = []
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    if source == {"kind": "tune"}:
        source = {
            "kind": "tune",
            "study_id": "study",
            "study_config_sha256": hashlib.sha256(
                files["provenance/parameter-source/study-config.yaml"]
            ).hexdigest(),
            "selection_sha256": hashlib.sha256(selection_payload).hexdigest(),
            "selected_trial_id": "trial",
            "selected_attempt_id": "attempt",
            "selected_attempt_manifest_sha256": "a" * 64,
        }
    resolution = {"source": source}
    resolution_sha = parameter_resolution_sha256(resolution)
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "parameter-resolution.json").write_text(
        json.dumps(resolution), encoding="utf-8"
    )
    capsule = {
        "schema": "autoforecast.parameter-source-capsule.v2",
        "parameter_resolution_sha256": resolution_sha,
        "study_id": "study",
        "study_config": "provenance/parameter-source/study-config.yaml",
        "study_config_sha256": hashlib.sha256(
            files["provenance/parameter-source/study-config.yaml"]
        ).hexdigest(),
        "selection": "provenance/parameter-source/study/selection.json",
        "selection_sha256": hashlib.sha256(
            files["provenance/parameter-source/study/selection.json"]
        ).hexdigest(),
        "selected_trial_id": "trial",
        "selected_attempt_id": "attempt",
        "selected_attempt_manifest_sha256": "a" * 64,
        "selected_attempt": "provenance/parameter-source/study/attempt",
        "artifacts": artifacts,
    }
    capsule_root.mkdir(parents=True, exist_ok=True)
    (capsule_root / "capsule.json").write_text(json.dumps(capsule), encoding="utf-8")
    monkeypatch.setattr(
        "autoforecast.parameter_sources.load_completed_run_bundle",
        lambda _path: {"status": "completed"},
    )
    monkeypatch.setattr(
        "autoforecast.parameter_sources.load_completed_attempt",
        lambda _path: (
            {"study_id": "study", "trial_id": "trial", "attempt_id": "attempt"},
            {"artifact_manifest_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        "autoforecast.parameter_sources._validate_retained_attempt_data_references",
        lambda _path: None,
    )

    with pytest.raises(ParameterSourceError, match=message):
        validate_parameter_source_capsule(
            root, expected_parameter_resolution_sha256=resolution_sha
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        (
            {"data_reference": "https://user:secret@example.test/data"},
            "retained source data reference is unsafe",
        ),
        (
            {"evaluation_data_reference": "https://example.test/data?token=secret"},
            "retained source score data reference is unsafe",
        ),
    ],
)
def test_retained_capsule_rechecks_data_references(
    tmp_path: pathlib.Path, fixture_kwargs: dict[str, object], message: str
) -> None:
    _project_root, _candidate, _study, _selection, attempt = _project(
        tmp_path, **fixture_kwargs  # type: ignore[arg-type]
    )
    with pytest.raises(ParameterSourceError, match=message):
        _validate_retained_attempt_data_references(attempt)


@pytest.mark.parametrize("legacy_study_metadata", [False, True])
def test_reused_hpo_selection_replays_manifest_model_study_and_source_identity(
    tmp_path: pathlib.Path, legacy_study_metadata: bool,
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, legacy_study_metadata=legacy_study_metadata,
    )
    record = parameter_resolution_from_selection(
        candidate,
        study,
        selection,
        kind="reuse_selection",
        project_root=project,
    )
    assert record["effective_parameters"]["model.hidden_size"] == 32
    assert record["source"]["selected_source_commit"] == _git(project, "rev-parse", "HEAD")

    path = tmp_path / "resolution.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection.write_text(selection.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ParameterSourceError, match="differs from sealed evidence"):
        load_parameter_resolution(
            path,
            parameter_resolution_sha256(record),
            candidate,
            project_root=project,
        )


def test_reused_hpo_selection_accepts_captured_scp_origin(tmp_path: pathlib.Path) -> None:
    project, candidate, study, selection, attempt = _project(
        tmp_path, repository_origin="alice@git:team/repo.git"
    )
    retained_source = json.loads(
        (
            attempt
            / "source-run/run/attempts/example_tft/one/provenance/source.json"
        ).read_text(encoding="utf-8")
    )

    record = parameter_resolution_from_selection(
        candidate,
        study,
        selection,
        kind="reuse_selection",
        project_root=project,
    )

    assert record["source"]["kind"] == "reuse_selection"
    assert retained_source["origin_url"] == "git:team/repo.git"


def test_reused_hpo_selection_accepts_complete_wrmsse_evidence(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, study, selection, _attempt = _project(tmp_path, metric="wrmsse")

    record = parameter_resolution_from_selection(
        candidate,
        study,
        selection,
        kind="reuse_selection",
        project_root=project,
    )

    assert record["source"]["protocol"]["metric"] == "wrmsse"


def test_reused_hpo_selection_rejects_a_different_campaign_protocol(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, study, selection, _attempt = _project(tmp_path)

    with pytest.raises(ParameterSourceError, match="campaign protocol"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
            expected_protocol={"metric": "rmse"},
        )


def test_reused_selection_refuses_incompatible_execution_contract(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, compatibility="obsolete-contract"
    )
    with pytest.raises(ParameterSourceError, match="incompatible"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


@pytest.mark.parametrize(
    "fixture_change",
    (
        {"retained_contract_suffix": b"# altered retained contract\n"},
        {"retained_request_suffix": b"# altered retained request\n"},
    ),
)
def test_reused_selection_hash_checks_retained_protocol_files(
    tmp_path: pathlib.Path, fixture_change: dict[str, bytes]
) -> None:
    project, candidate, study, selection, _attempt = _project(tmp_path, **fixture_change)
    with pytest.raises(ParameterSourceError, match="configuration bytes are incompatible"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


def test_reused_selection_refuses_a_completed_attempt_with_wrong_semantics(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, purpose="unrelated_experiment"
    )
    with pytest.raises(ParameterSourceError, match="incompatible"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


def test_hpo_resolution_structurally_rejects_malformed_provenance(
    tmp_path: pathlib.Path,
) -> None:
    _project_root, candidate, _study, _selection, _attempt = _project(tmp_path)
    record = default_parameter_resolution(candidate)
    record["source"] = {
        "kind": "tune",
        "study_id": [],
        "study_config_path": "../../escape.yaml",
        "study_config_sha256": "a" * 64,
        "selection_path": "",
        "selection_sha256": "b" * 64,
        "selected_trial_id": {},
        "selected_attempt_id": None,
        "selected_attempt_path": 42,
        "selected_attempt_manifest_sha256": "c" * 64,
        "selected_source_commit": "not-a-commit",
    }
    with pytest.raises(ParameterSourceError, match="provenance is incomplete"):
        validate_parameter_resolution(record, candidate)


def test_parameter_resolution_sha_binds_canonical_file_bytes(tmp_path: pathlib.Path) -> None:
    project, candidate, _study, _selection, _attempt = _project(tmp_path)
    record = default_parameter_resolution(candidate)
    path = tmp_path / "resolution.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n\n", encoding="utf-8")
    with pytest.raises(ParameterSourceError, match="bytes do not match"):
        load_parameter_resolution(
            path,
            parameter_resolution_sha256(record),
            candidate,
            project_root=project,
        )


def test_parameter_resolution_is_parsed_and_hashed_from_one_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, candidate, _study, _selection, _attempt = _project(tmp_path)
    record = default_parameter_resolution(candidate)
    path = tmp_path / "resolution.json"
    canonical = json.dumps(record, indent=2, sort_keys=True).encode() + b"\n"
    path.write_bytes(canonical)
    original = pathlib.Path.read_bytes
    reads = 0

    def adversarial_read(source: pathlib.Path) -> bytes:
        nonlocal reads
        if source == path:
            reads += 1
            return canonical if reads == 1 else b"{}\n"
        return original(source)

    monkeypatch.setattr(pathlib.Path, "read_bytes", adversarial_read)
    assert (
        load_parameter_resolution(
            path,
            parameter_resolution_sha256(record),
            candidate,
            project_root=project,
        )
        == record
    )
    assert reads == 1


def test_reused_selection_requires_retained_source_commit(tmp_path: pathlib.Path) -> None:
    project, candidate, study, selection, _attempt = _project(tmp_path, source_commit="a" * 40)
    with pytest.raises(ParameterSourceError, match="not retained locally"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        (
            {"data_reference": "https://user:secret@example.test/data"},
            "data reference is unsafe",
        ),
        (
            {"evaluation_data_reference": "https://example.test/data?token=secret"},
            "score data reference is unsafe",
        ),
        (
            {"source_origin_url": "https://user:secret@example.test/repo.git"},
            "source identity is incompatible",
        ),
        (
            {"source_origin_url": "https:///user:secret@example.test/repo.git"},
            "source identity is incompatible",
        ),
        (
            {"source_origin_url": "https://exa mple.test/repo.git"},
            "source identity is incompatible",
        ),
        (
            {"source_origin_url": "https://example.test/%ZZ"},
            "source identity is incompatible",
        ),
        (
            {"source_origin_url": "https://exa{mple}.test/repo.git"},
            "source identity is incompatible",
        ),
        (
            {"source_origin_url": "https://example.test/{repo}.git"},
            "source identity is incompatible",
        ),
        (
            {"source_origin_url": ""},
            "source identity is incompatible",
        ),
        (
            {"source_origin_url": "://example.test/repo.git"},
            "source identity is incompatible",
        ),
        (
            {"source_relevant_paths": [["autoforecast"]]},
            "source identity is incompatible",
        ),
        (
            {"malformed_evidence": "boolean_control"},
            "HPO protocol differs",
        ),
        (
            {"malformed_evidence": "boolean_data_count", "training_series": 1},
            "data check differs",
        ),
        (
            {"malformed_evidence": "boolean_forecast_count"},
            "forecast/model evidence is incomplete",
        ),
        (
            {"malformed_evidence": "boolean_score_count"},
            "score differs",
        ),
        (
            {"malformed_evidence": "wrmsse_overflow", "metric": "wrmsse"},
            "score differs",
        ),
        (
            {"malformed_evidence": "boolean_rank"},
            "HPO selection is invalid",
        ),
        (
            {"malformed_evidence": "boolean_candidate_rank"},
            "HPO selection is invalid",
        ),
    ],
)
def test_reused_selection_rejects_unsafe_or_malformed_retained_evidence(
    tmp_path: pathlib.Path, fixture_kwargs: dict[str, object], message: str
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, **fixture_kwargs  # type: ignore[arg-type]
    )
    with pytest.raises(ParameterSourceError, match=message):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


@pytest.mark.parametrize("value", [False, True, -1, 5, 0.0, None, "0", [], {}])
def test_reused_selection_rejects_malformed_zero_filled_count(
    tmp_path: pathlib.Path, value: object
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, zero_filled_count=value
    )

    with pytest.raises(ParameterSourceError, match="score differs"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


def test_reused_selection_rejects_test_evaluation_evidence(tmp_path: pathlib.Path) -> None:
    project, candidate, study, selection, _attempt = _project(tmp_path, extra_evaluation=True)
    with pytest.raises(ParameterSourceError, match="unexpected evaluation evidence"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


def test_reused_selection_rejects_validation_completion_disagreement(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, summary_validation_objective=999.0
    )
    with pytest.raises(ParameterSourceError, match="differs from completed HPO evidence"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


def test_reused_selection_requires_full_governed_forecast_and_model_evidence(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, include_forecast_evidence=False
    )
    with pytest.raises(ParameterSourceError, match="selected source forecast must be"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


def test_reused_selection_requires_runtime_and_lockfile_provenance(
    tmp_path: pathlib.Path,
) -> None:
    project, candidate, study, selection, _attempt = _project(
        tmp_path, include_runtime_evidence=False
    )
    with pytest.raises(ParameterSourceError, match="runtime"):
        parameter_resolution_from_selection(
            candidate,
            study,
            selection,
            kind="reuse_selection",
            project_root=project,
        )


def test_reused_hpo_selection_rejects_a_different_declared_winner_count(tmp_path):
    project, candidate, study, selection, _attempt = _project(tmp_path)
    with pytest.raises(ParameterSourceError, match="metadata is incompatible"):
        parameter_resolution_from_selection(
            candidate, dataclasses.replace(study, top_k=2), selection,
            kind="reuse_selection", project_root=project,
        )
