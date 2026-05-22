"""Functional proof that an Optuna trial reaches the governed shallow-model runner."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from pyspark.sql import SparkSession
from scripts.run_autoforecast import ROOT

import autoforecast.hpo_bridge as hpo_bridge_module
from autoforecast.data_admission import load_dataset_contract
from autoforecast.data_observability import CheckReport, inspect_panel
from autoforecast.hpo import HPOError
from autoforecast.hpo_bridge import (
    ForecastTrialBridge,
    _copy_run_artifact_directory,
    _load_evidence_record,
    _objective_value,
    _published_run_groups,
    _require_exact_validation_population,
    _same_int,
)
from autoforecast.model_catalog import discover_candidates
from autoforecast.tuning import loads_tuning_spec


def _revision_candidate_with_extra_parameter(model_id: str, parameter: str):
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == model_id
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


def _revision_parameter_groups(candidate) -> dict[str, dict[str, object]]:
    return loads_tuning_spec(
        candidate.source_bytes, source_path=candidate.source_path
    ).defaults()


def test_hpo_trial_score_must_cover_the_same_frozen_population() -> None:
    forecast = {"n_emitted": 24, "n_declined": 0}
    comparison = {
        "comparison_kind": "exploratory_common_cells",
        "candidates": [
            {
                "model_id": "model",
                "status": "scored",
                "detail": "",
                "score": {
                    "mae": 1.0,
                    "n_scored": 24,
                    "n_dropped_ineligible": 0,
                    "n_dropped_no_data": 0,
                    "n_zero_filled": 0,
                },
            }
        ],
        "ranking": {"metric": "mae", "n_common_cells": 24, "order": ["model"]},
    }
    assert _require_exact_validation_population(
        comparison,
        forecast,
        model_id="model",
        metric="mae",
        validation_series=12,
        horizon=2,
    )["mae"] == 1.0

    comparison["candidates"][0]["score"]["n_scored"] = 23
    with pytest.raises(HPOError, match="frozen validation population"):
        _require_exact_validation_population(
            comparison,
            forecast,
            model_id="model",
            metric="mae",
            validation_series=12,
            horizon=2,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("forecast", "n_emitted", True),
        ("forecast", "n_declined", False),
        ("ranking", "n_common_cells", True),
        ("score", "n_scored", True),
        ("score", "n_dropped_ineligible", False),
        ("score", "n_dropped_no_data", False),
        ("score", "n_zero_filled", False),
        ("score", "n_zero_filled", True),
        ("score", "n_zero_filled", -1),
        ("score", "n_zero_filled", 2),
        ("score", "n_zero_filled", 0.0),
        ("score", "n_zero_filled", None),
        ("score", "n_zero_filled", "0"),
        ("score", "n_zero_filled", []),
        ("score", "n_zero_filled", {}),
    ],
)
def test_hpo_trial_population_counts_reject_malformed_values(
    section: str, field: str, value: object
) -> None:
    forecast: dict[str, object] = {"n_emitted": 1, "n_declined": 0}
    score: dict[str, object] = {
        "mae": 1.0,
        "n_scored": 1,
        "n_dropped_ineligible": 0,
        "n_dropped_no_data": 0,
        "n_zero_filled": 0,
    }
    ranking: dict[str, object] = {
        "metric": "mae",
        "n_common_cells": 1,
        "order": ["model"],
    }
    comparison: dict[str, object] = {
        "comparison_kind": "exploratory_common_cells",
        "candidates": [
            {"model_id": "model", "status": "scored", "detail": "", "score": score}
        ],
        "ranking": ranking,
    }
    {"forecast": forecast, "ranking": ranking, "score": score}[section][field] = value

    with pytest.raises(HPOError, match="frozen validation population"):
        _require_exact_validation_population(
            comparison,
            forecast,
            model_id="model",
            metric="mae",
            validation_series=1,
            horizon=1,
        )


def test_claim_bearing_integer_comparison_rejects_booleans() -> None:
    assert _same_int(1, 1)
    assert _same_int(0, 0)
    assert not _same_int(True, 1)
    assert not _same_int(False, 0)


def test_bridge_refuses_linked_governed_artifact_sources(tmp_path: pathlib.Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("outside bundle", encoding="utf-8")
    source = tmp_path / "model"
    source.symlink_to(external, target_is_directory=True)
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(HPOError, match="may not contain symlinks"):
        _copy_run_artifact_directory(source, destination)


@pytest.mark.parametrize(
    "frequency,adapter,accepted",
    [("W", "seasonal_naive", True), ("7D", "auto_ets", False)],
)
def test_builtin_bridge_validates_cadence_capability_and_horizon_periods(
    frequency: str,
    adapter: str,
    accepted: bool,
    tmp_path: pathlib.Path,
) -> None:
    project = tmp_path / "project"
    contract = project / "configs/datasets/weekly.yaml"
    contract.parent.mkdir(parents=True)
    contract.write_text(
        f"""schema_version: 1
dataset:
  id: weekly
  panel:
    series_col: series_id
    time_col: date
    target_col: target
    frequency: {frequency}
    season_length: 52
""",
        encoding="utf-8",
    )
    request_path = project / "configs/autoforecast/weekly.yaml"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        """schema_version: 3
run:
  precision: 32-true
  name: weekly-hpo
  dataset_contract: configs/datasets/weekly.yaml
  model_catalog: configs/models
  models: [weekly_model]
  origin: '2024-01-07'
  horizon: 2
  tracking:
    mlflow: {enabled: false}
  evaluation:
    missing_actual: no_data
    metric: mae
    input_mode: post_training_score_only
    evidence_role: hpo_validation_selection
""",
        encoding="utf-8",
    )
    bridge = ForecastTrialBridge(
        request_path=request_path,
        data_path=tmp_path / "unused",
        data_snapshot_id="weekly-v1",
        data_reference="local://weekly-v1",
        epochs=1,
        timeout_seconds=10,
        project_root=project,
        source_data_path=tmp_path / "source",
        source_data_fingerprint="a" * 64,
    )
    study = SimpleNamespace(
        tuning=SimpleNamespace(model_id="weekly_model", adapter=adapter),
        data={"validation_target_date_range": ["2024-01-14", "2024-01-21"]},
    )

    if accepted:
        assert bridge._request(study).horizon == 2
        shared = yaml.safe_load(contract.read_text())
        shared["schema_version"] = 2
        shared["dataset"]["required_series_count"] = 2
        contract.write_text(yaml.safe_dump(shared))
        with pytest.raises(HPOError, match="shared-study inner HPO windows are not frozen"):
            bridge._request(study)
    else:
        with pytest.raises(HPOError, match="does not support 7D cadence"):
            bridge._request(study)


def test_builtin_bridge_requires_source_identity(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="required keyword-only"):
        ForecastTrialBridge(
            request_path=tmp_path / "request.yaml",
            data_path=tmp_path / "training",
            data_snapshot_id="training-v1",
            data_reference="local://training-v1",
            epochs=1,
            timeout_seconds=10,
            project_root=tmp_path,
        )

    with pytest.raises(HPOError, match="finite non-negative number"):
        ForecastTrialBridge(
            request_path=tmp_path / "request.yaml",
            data_path=tmp_path / "training",
            data_snapshot_id="training-v1",
            data_reference="local://training-v1",
            epochs=1,
            timeout_seconds=10,
            project_root=tmp_path,
            source_data_path=tmp_path / "source",
            source_data_fingerprint="a" * 64,
            early_stopping_absolute_min_delta=10**400,
        )

    with pytest.raises(HPOError, match="device must be"):
        ForecastTrialBridge(
            request_path=tmp_path / "request.yaml",
            data_path=tmp_path / "training",
            data_snapshot_id="training-v1",
            data_reference="local://training-v1",
            epochs=1,
            timeout_seconds=10,
            project_root=tmp_path,
            source_data_path=tmp_path / "source",
            source_data_fingerprint="a" * 64,
            device=[],  # type: ignore[arg-type] - public runtime boundary must reject it.
        )


@pytest.mark.parametrize("payload", ['{"value": 1, "value": 2}', '{"value": NaN}'])
def test_governed_trial_evidence_requires_exact_json(
    tmp_path: pathlib.Path, payload: str
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(payload, encoding="utf-8")
    with pytest.raises(HPOError, match="evidence is unreadable"):
        _load_evidence_record(evidence, "evidence")


def test_builtin_bridge_rejects_objectives_outside_binary64() -> None:
    with pytest.raises(HPOError, match="did not emit validation mae"):
        _objective_value({"mae": 10**400}, "mae")


def test_published_run_groups_excludes_private_worker_scratch(
    tmp_path: pathlib.Path,
) -> None:
    run = tmp_path / "governed-run"
    run.mkdir()
    (tmp_path / ".autoforecast-worker-scratch").mkdir()
    (tmp_path / "not-a-run.txt").write_text("ignored", encoding="utf-8")

    assert _published_run_groups(tmp_path) == [run]


def test_builtin_bridge_rejects_wrong_study_source_before_running(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    source = tmp_path / "source"
    observed = {
        training: "1" * 64,
        validation: "2" * 64,
        source: "b" * 64,
    }
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "seasonal_naive"
    )
    bridge = ForecastTrialBridge(
        request_path=tmp_path / "request.yaml",
        data_path=training,
        data_snapshot_id="training-v1",
        data_reference="local://training-v1",
        evaluation_data_path=validation,
        evaluation_data_snapshot_id="validation-v1",
        evaluation_data_reference="local://validation-v1",
        source_data_path=source,
        source_data_fingerprint="b" * 64,
        epochs=1,
        timeout_seconds=10,
        project_root=tmp_path,
        candidate=candidate,
    )
    study = SimpleNamespace(
        tuning=SimpleNamespace(
            model_id=candidate.id,
            adapter=candidate.adapter,
            source_sha256=candidate.source_sha256,
        ),
        data={"fingerprint": "a" * 64},
    )
    monkeypatch.setattr(ForecastTrialBridge, "_request", lambda _self, _study: object())
    monkeypatch.setattr(
        hpo_bridge_module,
        "directory_digest",
        lambda path, *, suffixes: observed[path],
    )
    called = False
    runner_module = ModuleType("scripts.run_autoforecast")

    def run_request(*_args: object, **_kwargs: object) -> int:
        nonlocal called
        called = True
        return 0

    runner_module.run_request = run_request  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts.run_autoforecast", runner_module)

    with pytest.raises(HPOError, match="changed before trial execution"):
        bridge(SimpleNamespace(study=study))  # type: ignore[arg-type]
    assert not called


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "changed-model"),
        ("adapter", "changed-adapter"),
        ("source_sha256", "0" * 64),
    ],
)
def test_builtin_bridge_rejects_changed_candidate_identity_before_data_or_runner(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "seasonal_naive"
    )
    bridge = ForecastTrialBridge(
        request_path=tmp_path / "request.yaml",
        data_path=tmp_path / "training",
        data_snapshot_id="training-v1",
        data_reference="local://training-v1",
        source_data_path=tmp_path / "source",
        source_data_fingerprint="a" * 64,
        epochs=1,
        timeout_seconds=10,
        project_root=tmp_path,
        candidate=candidate,
    )
    tuning = SimpleNamespace(
        model_id=candidate.id,
        adapter=candidate.adapter,
        source_sha256=candidate.source_sha256,
    )
    setattr(tuning, field, value)
    trial = SimpleNamespace(study=SimpleNamespace(tuning=tuning))
    monkeypatch.setattr(ForecastTrialBridge, "_request", lambda _self, _study: object())

    def must_not_run(*_args: object, **_kwargs: object) -> str:
        pytest.fail("candidate mismatch reached data or runner execution")

    monkeypatch.setattr(hpo_bridge_module, "directory_digest", must_not_run)
    runner_module = ModuleType("scripts.run_autoforecast")
    runner_module.run_request = must_not_run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts.run_autoforecast", runner_module)

    with pytest.raises(HPOError, match="differs from the frozen tuning profile"):
        bridge(trial)  # type: ignore[arg-type]


@pytest.mark.parametrize("precision", [None, "16-mixed"])
def test_builtin_bridge_forwards_frozen_candidate_to_governed_runner(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, precision: str | None
) -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "seasonal_naive"
    )
    bridge = ForecastTrialBridge(
        request_path=tmp_path / "request.yaml",
        data_path=tmp_path / "training",
        data_snapshot_id="training-v1",
        data_reference="local://training-v1",
        evaluation_data_path=tmp_path / "validation",
        evaluation_data_snapshot_id="validation-v1",
        evaluation_data_reference="local://validation-v1",
        source_data_path=tmp_path / "source",
        source_data_fingerprint="a" * 64,
        epochs=1,
        timeout_seconds=10,
        project_root=tmp_path,
        candidate=candidate,
        precision=precision,
    )
    trial = SimpleNamespace(
        study=SimpleNamespace(
            tuning=SimpleNamespace(
                model_id=candidate.id,
                adapter=candidate.adapter,
                source_sha256=candidate.source_sha256,
            ),
            data={"fingerprint": "a" * 64},
        ),
        attempt=SimpleNamespace(path=tmp_path / "attempt"),
        model_parameters={},
        training_parameters={},
    )
    monkeypatch.setattr(ForecastTrialBridge, "_request", lambda _self, _study: object())
    monkeypatch.setattr(
        hpo_bridge_module,
        "directory_digest",
        lambda _path, *, suffixes: "a" * 64,
    )

    class ReachedRunner(Exception):
        pass

    runner_module = ModuleType("scripts.run_autoforecast")

    def run_request(*_args: object, **kwargs: object) -> int:
        assert kwargs["expected_candidate"] is candidate
        assert kwargs["precision"] == precision
        raise ReachedRunner

    runner_module.run_request = run_request  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts.run_autoforecast", runner_module)
    with pytest.raises(ReachedRunner):
        bridge(trial)


def test_builtin_bridge_refuses_execution_without_a_frozen_candidate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = ForecastTrialBridge(
        request_path=tmp_path / "request.yaml",
        data_path=tmp_path / "training",
        data_snapshot_id="training-v1",
        data_reference="local://training-v1",
        source_data_path=tmp_path / "source",
        source_data_fingerprint="a" * 64,
        epochs=1,
        timeout_seconds=10,
        project_root=tmp_path,
    )
    monkeypatch.setattr(ForecastTrialBridge, "_request", lambda _self, _study: object())
    with pytest.raises(HPOError, match="requires a frozen model candidate"):
        bridge(SimpleNamespace(study=object()))  # type: ignore[arg-type]


def test_revision_hpo_trial_recipe_is_validated_before_data_or_runner(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "sft"
    )
    bridge = ForecastTrialBridge(
        request_path=tmp_path / "request.yaml",
        data_path=tmp_path / "training",
        data_snapshot_id="training-v1",
        data_reference="local://training-v1",
        source_data_path=tmp_path / "source",
        source_data_fingerprint="a" * 64,
        epochs=2,
        timeout_seconds=10,
        project_root=ROOT,
        candidate=candidate,
    )
    parameters = _revision_parameter_groups(candidate)
    parameters["model"]["encoder_days"] = 29
    trial = SimpleNamespace(
        study=SimpleNamespace(
            tuning=SimpleNamespace(
                model_id=candidate.id,
                adapter=candidate.adapter,
                source_sha256=candidate.source_sha256,
            )
        ),
        model_parameters=parameters["model"],
        training_parameters=parameters["training"],
    )
    monkeypatch.setattr(ForecastTrialBridge, "_request", lambda _self, _study: object())
    with pytest.raises(HPOError, match="valid execution recipe"):
        bridge(trial)


@pytest.mark.parametrize("model_id", ["sft", "hmfnet"])
@pytest.mark.parametrize(
    ("section", "parameter"),
    [("model", "unused"), ("training", "unused")],
)
def test_revision_hpo_trial_rejects_unconsumed_parameter_before_data_or_runner(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    section: str,
    parameter: str,
) -> None:
    candidate = _revision_candidate_with_extra_parameter(model_id, f"{section}.{parameter}")
    bridge = ForecastTrialBridge(
        request_path=tmp_path / "request.yaml",
        data_path=tmp_path / "training",
        data_snapshot_id="training-v1",
        data_reference="local://training-v1",
        evaluation_data_path=tmp_path / "validation",
        evaluation_data_snapshot_id="validation-v1",
        evaluation_data_reference="local://validation-v1",
        source_data_path=tmp_path / "source",
        source_data_fingerprint="a" * 64,
        epochs=2,
        timeout_seconds=10,
        project_root=ROOT,
        candidate=candidate,
    )
    parameters = _revision_parameter_groups(candidate)
    model_parameters = parameters["model"]
    training_parameters = parameters["training"]
    {"model": model_parameters, "training": training_parameters}[section][parameter] = 1
    trial = SimpleNamespace(
        study=SimpleNamespace(
            tuning=SimpleNamespace(
                model_id=candidate.id,
                adapter=candidate.adapter,
                source_sha256=candidate.source_sha256,
            )
        ),
        model_parameters=model_parameters,
        training_parameters=training_parameters,
    )
    monkeypatch.setattr(ForecastTrialBridge, "_request", lambda _self, _study: object())

    def reached_data_digest(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("REACHED_DATA_DIGEST")

    monkeypatch.setattr(hpo_bridge_module, "directory_digest", reached_data_digest)
    with pytest.raises(HPOError, match="typed executable settings"):
        bridge(trial)


@pytest.mark.parametrize("model_id", ["sft", "hmfnet"])
def test_revision_hpo_trial_forwards_every_sampled_parameter_to_governed_runner(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    candidate = next(
        item for item in discover_candidates(ROOT / "configs/models") if item.id == model_id
    )
    parameters = _revision_parameter_groups(candidate)
    parameters["model"].update(
        encoder_days=728,
        d_model=96,
        n_heads=8,
        trunk_stages=2 if model_id.endswith("_v2") else 4,
        covariate_dropout=0.4,
    )
    if model_id.endswith("_v2"):
        parameters["model"].update(
            decoder_layers=2,
            history_attention_layers=2,
            commercial_attention_layers=2,
            commercial_attention_heads=4,
            demand_dropout=0.2,
            patch_size=14,
        )
    parameters["training"].update(
        learning_rate=0.0005,
        train_stride=14,
        train_windows=26,
    )
    bridge = ForecastTrialBridge(
        request_path=tmp_path / "request.yaml",
        data_path=tmp_path / "training",
        data_snapshot_id="training-v1",
        data_reference="local://training-v1",
        evaluation_data_path=tmp_path / "validation",
        evaluation_data_snapshot_id="validation-v1",
        evaluation_data_reference="local://validation-v1",
        source_data_path=tmp_path / "source",
        source_data_fingerprint="a" * 64,
        epochs=2,
        timeout_seconds=10,
        project_root=ROOT,
        candidate=candidate,
    )
    trial = SimpleNamespace(
        study=SimpleNamespace(
            tuning=SimpleNamespace(
                model_id=candidate.id,
                adapter=candidate.adapter,
                source_sha256=candidate.source_sha256,
            ),
            data={"fingerprint": "a" * 64},
        ),
        attempt=SimpleNamespace(path=tmp_path / "attempt"),
        model_parameters=parameters["model"],
        training_parameters=parameters["training"],
    )
    monkeypatch.setattr(ForecastTrialBridge, "_request", lambda _self, _study: object())
    monkeypatch.setattr(
        hpo_bridge_module,
        "directory_digest",
        lambda _path, *, suffixes: "a" * 64,
    )

    class ReachedRunner(Exception):
        pass

    expected = {
        **{f"model.{name}": value for name, value in parameters["model"].items()},
        **{
            f"training.{name}": value
            for name, value in parameters["training"].items()
        },
    }
    runner_module = ModuleType("scripts.run_autoforecast")

    def run_request(*_args: object, **kwargs: object) -> int:
        assert kwargs["expected_candidate"] is candidate
        assert kwargs["parameter_overrides"] == expected
        raise ReachedRunner

    runner_module.run_request = run_request  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scripts.run_autoforecast", runner_module)
    with pytest.raises(ReachedRunner):
        bridge(trial)


@pytest.mark.skipif(not os.environ.get("JAVA_HOME"), reason="Spark requires JAVA_HOME")
@pytest.mark.parametrize(
    ("model_id", "profile_path", "history_days", "expected_model_artifacts"),
    [
        (
            "shallow_global",
            "configs/models/global/shallow_global/model.yaml",
            120,
            ("occurrence.txt", "positive_size.txt"),
        ),
        (
            "tft",
            "configs/models/neural/tft/model.yaml",
            210,
            ("tft-state.pt",),
        ),
    ],
)
def test_builtin_optuna_bridge_runs_one_supported_trial_and_retains_its_model(
    tmp_path: pathlib.Path,
    model_id: str,
    profile_path: str,
    history_days: int,
    expected_model_artifacts: tuple[str, ...],
) -> None:
    pytest.importorskip("optuna")
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("hpo-bridge-functional")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        data_path = tmp_path / "panel"
        training_path = tmp_path / "training-panel"
        validation_path = tmp_path / "validation-panel"
        records = [
            (
                f"series-{series:02d}",
                dt.date(2024, 1, 1) + dt.timedelta(days=offset),
                float(3 + series + (offset % 7)),
            )
            for series in range(12)
            for offset in range(history_days)
        ]
        frame = spark.createDataFrame(records, ["series_id", "date", "target"])
        frame.write.parquet(str(data_path))
        origin = dt.date(2024, 1, 1) + dt.timedelta(days=history_days - 4)
        validation_end = origin + dt.timedelta(days=2)
        frame.where(f"date <= DATE '{origin.isoformat()}'").write.parquet(str(training_path))
        frame.where(
            f"date > DATE '{origin.isoformat()}' AND date <= DATE '{validation_end.isoformat()}'"
        ).write.parquet(str(validation_path))
        panel = load_dataset_contract(ROOT / "configs/datasets/synthetic_daily.yaml").panel(
            data_path
        )
        report = inspect_panel(panel, spark, as_of=origin.isoformat())
        assert isinstance(report, CheckReport) and report.accepted
        fingerprint = report.facts.data_fingerprint
    finally:
        spark.stop()

    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT,
        repository,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            ".autoforecast",
            "data",
            "gates",
            "mlruns",
            "mlartifacts",
            "lightning_logs",
            "__pycache__",
        ),
    )
    request = repository / f"configs/autoforecast/examples/hpo_{model_id}.yaml"
    request.write_text(
        f"""schema_version: 3
run:
  precision: 32-true
  name: hpo-{model_id.replace('_', '-')}
  dataset_contract: configs/datasets/synthetic_daily.yaml
  model_catalog: configs/models
  models: [{model_id}]
  origin: \"{origin.isoformat()}\"
  horizon: 2
  tracking:
    mlflow: {{enabled: false}}
  evaluation:
    missing_actual: no_data
    metric: mae
    input_mode: post_training_score_only
    evidence_role: hpo_validation_selection
""",
        encoding="utf-8",
    )
    study_id = f"hpo-{model_id.replace('_', '-')}-functional"
    validation_end = origin + dt.timedelta(days=2)
    observed_end = origin + dt.timedelta(days=3)
    validation_start = origin + dt.timedelta(days=1)
    study = repository / f"configs/autoforecast/examples/{study_id}.yaml"
    study.write_text(
        f"""schema_version: 1
study:
  id: {study_id}
  model_config: {profile_path}
  backend: optuna
  direction: minimize
  objective: validation/objective
  max_trials: 1
  top_k: 1
  seed: 7
  comparability_fingerprint: {hashlib.sha256(study_id.encode()).hexdigest()}
  evaluator: builtin
  data:
    id: synthetic-{model_id.replace('_', '-')}
    fingerprint: {fingerprint}
    observed_date_range: ['2024-01-01', '{observed_end.isoformat()}']
    training_date_range: ['2024-01-01', '{origin.isoformat()}']
    validation_date_range: ['2024-01-01', '{origin.isoformat()}']
    test_date_range: ['{observed_end.isoformat()}', '{observed_end.isoformat()}']
    validation_target_date_range: ['{validation_start.isoformat()}', '{validation_end.isoformat()}']
    test_target_date_range: ['{observed_end.isoformat()}', '{observed_end.isoformat()}']
    training_series: 12
    validation_series: 12
    test: reserved_not_evaluated
""",
        encoding="utf-8",
    )
    for command in (
        ("git", "init"),
        ("git", "config", "user.email", "test@example.test"),
        ("git", "config", "user.name", "Functional"),
        ("git", "add", "."),
        ("git", "commit", "-m", "functional source"),
    ):
        outcome = subprocess.run(command, cwd=repository, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stderr
    output_root = tmp_path / "hpo"
    program = """
import pathlib
import sys

from autoforecast.hpo import load_hpo_study, run_hpo_study
from autoforecast.hpo_bridge import ForecastTrialBridge
from autoforecast.model_catalog import discover_candidates

project = pathlib.Path.cwd()
study = load_hpo_study(project / sys.argv[1], project_root=project)
candidate = next(
    item
    for item in discover_candidates(project / "configs/models")
    if item.id == study.tuning.model_id
)
bridge = ForecastTrialBridge(
    request_path=project / sys.argv[2],
    data_path=pathlib.Path(sys.argv[3]),
    data_snapshot_id=sys.argv[4],
    data_reference=sys.argv[5],
    evaluation_data_path=pathlib.Path(sys.argv[6]),
    evaluation_data_snapshot_id=sys.argv[7],
    evaluation_data_reference=sys.argv[8],
    source_data_path=pathlib.Path(sys.argv[9]),
    source_data_fingerprint=sys.argv[10],
    epochs=1,
    timeout_seconds=180,
    project_root=project,
    candidate=candidate,
)
run_hpo_study(study, pathlib.Path(sys.argv[11]), bridge)
"""
    outcome = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(study.relative_to(repository)),
            str(request.relative_to(repository)),
            str(training_path),
            f"synthetic-{model_id.replace('_', '-')}-hpo-training",
            f"local://synthetic-{model_id}-snapshot",
            str(validation_path),
            f"synthetic-{model_id.replace('_', '-')}-hpo-validation",
            f"local://synthetic-{model_id}-validation",
            str(data_path),
            fingerprint,
            str(output_root),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=240,
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    attempt = next((output_root / f"studies/{study_id}/trials").glob("*/attempts/*"))
    assert (attempt / "completion.json").is_file()
    assert (attempt / "evaluation/governed-run.json").is_file()
    assert all((attempt / "model" / artifact).is_file() for artifact in expected_model_artifacts)
    selection = json.loads(
        (output_root / f"studies/{study_id}/selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected"][0]["trial_id"] == "trial-0000"
