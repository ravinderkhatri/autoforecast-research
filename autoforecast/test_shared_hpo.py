"""HPO consumes only a declared inner boundary and the complete frozen cohort."""

import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from autoforecast import benchmark
from autoforecast import forecast_campaign as campaign
from autoforecast.hpo import HPOError
from autoforecast.hpo_bridge import ForecastTrialBridge, _require_exact_validation_population
from autoforecast.test_study import freeze

pytest_plugins = ["autoforecast.test_study"]


@pytest.fixture(scope="module")
def inner_study(tmp_path_factory, spark):
    # Three source IDs, one late-series exclusion, two retained IDs, four inner cells.
    path = freeze(
        tmp_path_factory.mktemp("inner-study"), spark, actual_missing="mask",
        mutate=lambda recipe: recipe.update(
            tuning_windows=[{"origin": "2024-01-03", "horizon": 2}],
        ),
    )
    _, record = benchmark.verify_benchmark(path)
    fixed = benchmark.evaluation_contract(
        path, stage="tuning", origin="2024-01-03", horizon=2, metric="rmse",
    )
    return path, record, fixed


def hpo_data():
    return {
        "validation_target_date_range": ["2024-01-04", "2024-01-05"],
        "training_date_range": ["2024-01-01", "2024-01-03"],
        "training_series": 2, "validation_series": 2,
    }


@pytest.mark.parametrize("defect", [None, "undeclared", "outer", "history", "cohort", "metric"])
def test_campaign_hpo_requires_declared_inner_history_population_and_metric(inner_study, defect):
    path, _, expected = inner_study
    data = hpo_data()
    metric = "rmse"
    if defect == "undeclared":
        data["validation_target_date_range"] = ["2024-01-03", "2024-01-04"]
    elif defect == "outer":
        data["validation_target_date_range"] = ["2024-01-06", "2024-01-07"]
    elif defect == "history":
        data["training_date_range"][0] = "2024-01-02"
    elif defect == "cohort":
        data["validation_series"] = 1
    elif defect == "metric":
        metric = "mae"
    args = (SimpleNamespace(data=data), {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, metric)
    if defect:
        with pytest.raises((campaign.ForecastCampaignError, benchmark.BenchmarkError)):
            campaign._hpo_study_evaluation(*args)
    else:
        assert campaign._hpo_study_evaluation(*args) == expected


@pytest.mark.parametrize("defect", [None, "missing", "sha", "outer", "source", "cohort"])
def test_hpo_worker_handoff_requires_authenticated_tuning_context(inner_study, tmp_path, defect):
    path, _, fixed = inner_study
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump({"schema_version": 3, "run": {
        "name": "inner-test", "dataset_contract": str(path.parent / "dataset.yaml"),
        "model_catalog": "models", "models": ["seasonal_naive"],
        "origin": "2024-01-03", "horizon": 2, "precision": "32-true",
        "tracking": {"mlflow": {"enabled": False}}, "evaluation": {
            "missing_actual": "no_data", "metric": "rmse",
            "input_mode": "post_training_score_only", "evidence_role": "hpo_validation_selection",
        },
    }}))
    contract = tmp_path / "evaluation.json"
    if defect == "outer":
        fixed = benchmark.evaluation_contract(
            path, stage="validation", origin="2024-01-05", horizon=2, metric="rmse",
        )
    contract.write_text(json.dumps(fixed))
    bridge = ForecastTrialBridge(
        request_path=request, data_path=tmp_path / "history", data_snapshot_id="inner-history",
        data_reference="local://inner-history", epochs=1, timeout_seconds=30,
        project_root=tmp_path, source_data_path=path.parent / "selection",
        source_data_fingerprint=fixed["actuals"]["sha256"],
        evaluation_contract_path=contract,
        evaluation_contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
    )
    study = SimpleNamespace(
        tuning=SimpleNamespace(model_id="seasonal_naive", adapter="seasonal_naive"),
        data=hpo_data(),
    )
    if defect == "missing":
        bridge = dataclasses.replace(
            bridge, evaluation_contract_path=None, evaluation_contract_sha256=None,
        )
    elif defect == "sha":
        bridge = dataclasses.replace(bridge, evaluation_contract_sha256="0" * 64)
    elif defect == "source":
        bridge = dataclasses.replace(bridge, source_data_fingerprint="0" * 64)
    elif defect == "cohort":
        study.data["training_series"] = 1
    if defect:
        with pytest.raises(HPOError):
            bridge._request(study)
    else:
        assert bridge._request(study).origin == "2024-01-03"


@pytest.mark.parametrize("defect", [None, "forecast", "masked", "coverage", "policy", "boundary"])
def test_hpo_objective_preserves_masked_denominator_and_full_forecasts(inner_study, defect):
    _, _, fixed = inner_study
    fixed = copy.deepcopy(fixed)
    score = {
        "rmse": 1.0, "metric_definition": fixed["metric_policy"],
        "coverage": fixed["coverage"]["counts"], "n_scored": 2,
        "n_dropped_ineligible": 0, "n_dropped_no_data": 2, "n_zero_filled": 0,
    }
    comparison = {
        "comparison_kind": "fixed_benchmark", "evaluation_contract": copy.deepcopy(fixed),
        "candidates": [{"model_id": "sft", "status": "scored", "detail": "", "score": score}],
        "ranking": {"metric": "rmse", "n_common_cells": 4, "order": ["sft"]},
    }
    forecast = {"n_emitted": 4, "n_declined": 0}
    if defect == "forecast":
        forecast.update(n_emitted=2, n_declined=2)
    elif defect == "masked":
        score["n_scored"] = 4
    elif defect == "coverage":
        score["coverage"] = {**score["coverage"], "scored": 4}
    elif defect == "policy":
        score["metric_definition"] = {**score["metric_definition"], "actual_missing": "zero"}
    elif defect == "boundary":
        comparison["evaluation_contract"]["boundary"] = "validation/2024-01-05/2/D"
    kwargs = dict(model_id="sft", metric="rmse", validation_series=2,
                  horizon=2, fixed_contract=fixed)
    if defect:
        with pytest.raises(HPOError, match="frozen validation population"):
            _require_exact_validation_population(comparison, forecast, **kwargs)
    else:
        assert _require_exact_validation_population(comparison, forecast, **kwargs) == score


@pytest.mark.parametrize("field", [None, "boundary", "study_id", "precision", "maximum_epochs"])
def test_tuned_parameter_handoff_still_matches_the_frozen_campaign(tmp_path, monkeypatch, field):
    """Changing a verified resolution in the next handoff must not change the tuning protocol."""
    candidate = SimpleNamespace(id="sft", source_sha256="a" * 64,
                                implementation_config_sha256="b" * 64)
    study = {"id": "inner", "sha256": "c" * 64}
    protocol = {
        "precision": "32-true", "maximum_epochs": 2,
        "study_evaluation": {"study_id": "d" * 64, "boundary": "tuning/2024-01-03/2/D"},
    }
    _, spec = campaign._variant_spec_identity(candidate, "tune", None, study, protocol)
    profile = {"model_id": "sft", "mode": "tune", "tune_stage": "t00",
               "study": study, "variant_spec_sha256": spec}
    altered = copy.deepcopy(protocol)
    if field in {"boundary", "study_id"}:
        altered["study_evaluation"][field] = "changed"
    elif field:
        altered[field] = "16-mixed" if field == "precision" else 3
    record = {"source": {"study_id": "inner", "study_config_sha256": "c" * 64,
                         "protocol": altered}}
    monkeypatch.setattr(campaign, "_parameter_profile", lambda *_: profile)
    monkeypatch.setattr(campaign, "_campaign_candidate", lambda *_: candidate)
    monkeypatch.setattr(campaign, "_completed_output", lambda *_: tmp_path / "resolution.json")
    monkeypatch.setattr(campaign, "_load_completed_json", lambda *_: (record, "e" * 64))
    monkeypatch.setattr(campaign, "_validated_parameter_record", lambda record, *_a, **_k: record)
    if field:
        with pytest.raises(campaign.ForecastCampaignError, match="frozen campaign"):
            campaign._child_parameter_resolution({}, "sft:tuned", tmp_path)
    else:
        assert campaign._child_parameter_resolution({}, "sft:tuned", tmp_path) == record
