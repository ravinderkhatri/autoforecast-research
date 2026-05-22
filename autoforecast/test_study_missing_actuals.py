"""Study scoring requires complete actuals unless masking is explicitly frozen."""

import dataclasses
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark
from autoforecast.data_admission import load_dataset_contract
from autoforecast.panel import Forecast
from autoforecast.score_spark import ScoreError
from autoforecast.study_scoring import _score_study_snapshot, score_study
from autoforecast.study_validation import StudyValidation

pytest_plugins = ["autoforecast.test_study"]


def prepare(root, spark, *, policy=None, missing="complete"):
    """Population: A and B, three dates each, six required cells; no exclusions."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ("A", "B"):
        for day in range(3, 12):
            value = float(day + 5 if day <= 5 else 10)
            absent = sid == "B" and (day == 7 if missing in {"absent", "null"}
                                     else 6 <= day <= 8 if missing == "all" else False)
            if absent and missing != "null":
                continue
            rows.append({"item": sid, "day": dt.date(2026, 10, day),
                         "units": None if absent else value})
    source = root / "observations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    dataset = root / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"schema_version": 1, "dataset": {"id": "coverage", "panel": {
        "series_col": "item", "time_col": "day", "target_col": "units", "frequency": "D",
        "season_length": 1, "target_estimand": "observed_nonnegative",
    }}}))
    recipe = {
        "schema_version": 2, "source_keys": ["item"], "training_start": "2026-10-03",
        "validation_origins": ["2026-10-05"], "validation_horizon": 3,
        "final_train_cutoff": "2026-10-08", "final_test_horizon": 3,
        "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
        "feature_availability": {}, "metric": "wrmsse",
    }
    if policy is not None:
        recipe["actual_missing"] = policy
    refs = root / "references.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"keys": {"item": sid}, "stage": stage, "origin": origin, "weight": weight, "scale": 1.0}
        for sid, weight in (("A", 80.0), ("B", 20.0))
        for stage, origin in (("validation", "2026-10-05"), ("test", "2026-10-08"))
    ]), refs)
    return benchmark.freeze_benchmark(
        contract_path=dataset, sources=dict.fromkeys(
            ("selection", "final_training", "validation_score", "final_score"), source,
        ), recipe=recipe, scoring_references=refs, output_root=root / "frozen", spark=spark,
    )


def scoring_inputs(manifest, root):
    contract = benchmark.evaluation_contract(
        manifest, stage="validation", origin="2026-10-05", horizon=3, metric="wrmsse",
    )
    cells = pq.read_table(contract["grid"]["path"]).to_pylist()
    predictions = root / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist([
        {**row, "yhat": 12.0 if json.loads(row["series_id"])["item"] == "A"
         else 110.0 if row["date"].day == 7 else 14.0} for row in cells
    ]), predictions)
    panel = load_dataset_contract(manifest.parent / "dataset.yaml").panel(contract["actuals"]["path"])
    forecast = Forecast("fixture", str(predictions), "2026-10-05", 3, 2, {"emitted": 6},
                        target_estimand=panel.target_estimand)
    return contract, panel, forecast


@pytest.mark.parametrize("missing", ["absent", "null", "all"])
def test_default_refuses_even_one_missing_actual_at_registration(tmp_path, spark, missing):
    with pytest.raises(benchmark.BenchmarkError, match="requires complete actuals") as error:
        prepare(tmp_path, spark, missing=missing)
    assert "expected=6" in str(error.value)
    assert f"missing={3 if missing == 'all' else 1}" in str(error.value)
    assert not list((tmp_path / "frozen").glob("*/benchmark.json"))


@pytest.mark.parametrize("policy", ["zero", "guess"])
def test_study_rejects_imputation_policy_before_reading_data(tmp_path, policy):
    with pytest.raises(benchmark.BenchmarkError, match="actual_missing must be error or mask"):
        prepare(tmp_path, object(), policy=policy)


@pytest.mark.parametrize("missing,expected", [("complete", 13.165465835840768),
                                               ("absent", 2.4), ("null", 2.4)])
def test_spark_and_checkpoint_scores_preserve_missingness_and_fixed_weights(
    tmp_path, spark, missing, expected,
):
    policy = None if missing == "complete" else "mask"
    manifest = prepare(tmp_path, spark, policy=policy, missing=missing)
    contract, panel, forecast = scoring_inputs(manifest, tmp_path)
    result = score_study(panel, forecast, spark, contract=contract, output_dir=tmp_path / "score")
    assert result["wrmsse"] == pytest.approx(expected)
    assert result["coverage"] == {
        "expected": 6, "originally_missing": int(missing != "complete"), "zero_filled": 0,
        "excluded": int(missing != "complete"), "scored": 6 - int(missing != "complete"),
    }
    assert result["metric_definition"]["actual_missing"] == (policy or "error")
    assert result["metric_definition"]["label"] == ("masked WRMSSE" if policy else "WRMSSE")
    per_series = sorted(pq.read_table(result["series_scores"]["path"]).to_pylist(),
                        key=lambda row: row["series_id"])
    assert [row["normalized_weight"] for row in per_series] == pytest.approx([0.8, 0.2])
    actuals = pq.read_table(panel.path).to_pylist()
    assert all(row["units"] is None for row in actuals if row[benchmark.MISSING_COLUMN])
    file = tmp_path / "evaluation.json"
    file.write_text(json.dumps(contract))
    window = StudyValidation.read(
        file, hashlib.sha256(file.read_bytes()).hexdigest(), panel=panel, origin=forecast.origin,
        horizon=3, series_ids=tuple(row["series_id"] for row in per_series),
    )
    predictions = np.array([[12.0, 12.0, 12.0], [14.0, 110.0, 14.0]])
    assert window.score(predictions)["wrmsse"] == pytest.approx(expected)
    if policy:
        assert math.isnan(window.target[1, 1])
        # Both numerical boundaries must refuse a strict policy even after registration.
        strict = {**contract, "metric_policy": {**contract["metric_policy"], "actual_missing": "error"}}
        with pytest.raises(ValueError, match="requires complete actuals"):
            dataclasses.replace(window, contract_json=json.dumps(strict)).score(predictions)
        snapshots = {key: Path(contract[key]["path"])
                     for key in ("actuals", "grid", "coverage")}
        with pytest.raises(ScoreError, match="requires complete actuals"):
            _score_study_snapshot(panel, forecast, spark, contract=strict, snapshots=snapshots,
                                  output_dir=tmp_path / "strict-score")


def test_mask_refuses_an_entirely_missing_series_without_redistributing_weight(tmp_path, spark):
    manifest = prepare(tmp_path, spark, policy="mask", missing="all")
    contract, panel, forecast = scoring_inputs(manifest, tmp_path)
    with pytest.raises(ScoreError, match="every retained series"):
        score_study(panel, forecast, spark, contract=contract, output_dir=tmp_path / "score")


def test_missing_policy_defaults_and_history_never_impute(tmp_path):
    from autoforecast.test_study import inputs

    _, _, recipe = inputs(tmp_path)
    recipe.pop("actual_missing")
    normalized = benchmark._normalise_recipe(recipe)
    assert normalized["actual_missing"] == "error"
    assert normalized["history_missing"] == "preserve"
    for policy in ("zero", "mask", "error"):
        with pytest.raises(benchmark.BenchmarkError, match="history_missing must be preserve"):
            benchmark._normalise_recipe({**recipe, "history_missing": policy})


def test_previous_manifest_is_not_silently_reinterpreted():
    with pytest.raises(benchmark.BenchmarkError, match="unsupported benchmark schema"):
        benchmark._validate_manifest({"schema": "autoforecast.frozen-benchmark.v2"})


@pytest.mark.parametrize("policy", ["error", "mask"])
def test_checkpoint_scores_observed_zero_and_requires_complete_forecasts(policy):
    window = StudyValidation(
        json.dumps({"metric_policy": {"metric": "wrmsse", "actual_missing": policy}}),
        "a" * 64, ("A", "B"), np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.ones((2, 2), dtype=bool), np.array([0.8, 0.2]), np.ones(2),
    )
    assert window.score(np.array([[2.0, 2.0], [4.0, 4.0]]))["wrmsse"] == pytest.approx(2.4)
    for prediction in (np.array([[2.0, np.nan], [4.0, 4.0]]), np.ones((2, 1))):
        with pytest.raises(ValueError, match="finite nonnegative grid"):
            window.score(prediction)
