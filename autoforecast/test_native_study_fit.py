"""Small native fits prove study selection/refit/replay, not forecasting accuracy."""

import copy
import datetime as dt
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from autoforecast import benchmark
from autoforecast.adapters.mft import fit_predict_revision
from autoforecast.data_admission import load_dataset_contract
from autoforecast.forecast_campaign import _materialize_training_panel
from autoforecast.study_scoring import score_study
from autoforecast.test_multibranch_revisions import full_history_test_config
from autoforecast.test_pact_revision import _config as pact_config

pytest_plugins = ["autoforecast.test_study"]


@pytest.fixture(scope="module")
def native_study(tmp_path_factory, spark):
    """Two retained series, 126 days each, no exclusions; 14 cells per forecast boundary."""
    return freeze_native_study(tmp_path_factory.mktemp("native-shared-study"), spark)


def freeze_native_study(root, spark, *, horizon=7, price=False):
    """Freeze two complete series with two missing validation actuals and optional known price."""
    start = dt.date(2024, 1, 1)
    rows = [{
        "series": series, "date": start + dt.timedelta(days=offset),
        "sales": None if (series, offset) in {("a", 112), ("b", 113)}
        else (offset + index) % 5 + 1,
        "state": "exact",
        **({"price": 1.0, "price_known_at": start} if price else {}),
    } for index, series in enumerate(("a", "b")) for offset in range(112 + 2 * horizon)]
    source = root / "observations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    contract = root / "dataset.yaml"
    contract.write_text(yaml.safe_dump({
        "schema_version": 1, "dataset": {"id": "synthetic-native-study", "panel": {
            "series_col": "series", "time_col": "date", "target_col": "sales",
            "exposure_col": "state", "frequency": "D", "season_length": 7,
            "target_estimand": "sales_given_availability",
            **({"price_col": "price", "known_future": ["price"]} if price else {}),
        }},
    }))
    recipe = {
        "schema_version": 2, "source_keys": ["series"], "training_start": start.isoformat(),
        "validation_origins": [(start + dt.timedelta(days=111)).isoformat()],
        "validation_horizon": horizon,
        "final_train_cutoff": (start + dt.timedelta(days=111 + horizon)).isoformat(),
        "final_test_horizon": horizon, "history_policy": "fixed", "new_series_policy": "error",
        "target_unit": "counts", "feature_availability": {"price": "price_known_at"} if price else {},
        "metric": "rmse",
        "actual_missing": "mask",
    }
    path = benchmark.freeze_benchmark(
        contract_path=contract,
        sources=dict.fromkeys(("selection", "validation_score", "final_training", "final_score"),
                              source),
        recipe=recipe, output_root=root / "studies", spark=spark,
    )
    _, record = benchmark.verify_benchmark(path)
    return path, record


@pytest.mark.parametrize("variant", ["v2", "v3", "v5"])
def test_native_study_selects_scores_refits_and_replays(native_study, tmp_path, spark, variant):
    path, record = native_study
    recipe = record["recipe"]
    origin, horizon = recipe["validation_origins"][0], recipe["validation_horizon"]
    contract_path = path.parent / "dataset.yaml"
    dataset = load_dataset_contract(contract_path)
    config = (pact_config() if variant == "v5" else full_history_test_config(
        variant, encoder_days=28, epochs=1,
    ))
    training = tmp_path / "selection-input"
    _materialize_training_panel(
        benchmark.artifact_path(path, record, "selection"), training,
        contract_path=contract_path, origin=origin, horizon=horizon,
        training_start=recipe["training_start"],
        expected_source_sha256=record["artifacts"]["selection"],
        expected_contract_sha256=record["dataset_sha256"],
    )
    assert max(pq.read_table(training)["date"].to_pylist()).isoformat() == origin
    fixed = benchmark.evaluation_contract(
        path, stage="validation", origin=origin, horizon=horizon, metric="rmse",
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(fixed))
    evaluation_sha = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    validation_panel = dataset.panel(fixed["actuals"]["path"])
    forecast = fit_predict_revision(
        dataset.panel(training), origin, horizon, config=config, out_dir=str(tmp_path / "selection"),
        training_mode="fixed_validation", validation_panel=validation_panel,
        evaluation_contract_path=str(evaluation_path), evaluation_contract_sha256=evaluation_sha,
    )
    assert forecast.n_emitted == 14 and forecast.n_declined == 0
    selected = torch.load(tmp_path / "selection/model.pt", weights_only=True)["training"]
    assert selected["training_contract"]["study_evaluation"]["study_id"] == record["study_id"]
    assert selected["training_contract"]["study_evaluation"]["evaluation_contract_sha256"] == (
        evaluation_sha
    )
    score = score_study(
        validation_panel, forecast, spark, contract=fixed, output_dir=tmp_path / "score",
    )
    assert score["n_scored"] == 12 and score["n_dropped_no_data"] == 2
    assert selected["validation"][-1]["mean_rmse"] == pytest.approx(score["rmse"], rel=1e-6)

    cutoff = recipe["final_train_cutoff"]
    final_input = tmp_path / "refit-input"
    _materialize_training_panel(
        benchmark.artifact_path(path, record, "final_training"), final_input,
        contract_path=contract_path, origin=cutoff, horizon=horizon,
        training_start=recipe["training_start"],
        expected_source_sha256=record["artifacts"]["final_training"],
        expected_contract_sha256=record["dataset_sha256"],
    )
    final = fit_predict_revision(
        dataset.panel(final_input), cutoff, horizon, config=config, out_dir=str(tmp_path / "refit"),
        training_mode="fixed_refit", fixed_update_budget=selected["selected_updates"],
    )
    saved = torch.load(tmp_path / "refit/model.pt", weights_only=True)["training"]
    assert saved["validation"] == []
    assert saved["refit_optimizer_steps"] == selected["selected_updates"]
    assert "study_evaluation" not in saved["training_contract"]
    replay = fit_predict_revision(
        dataset.panel(final_input), cutoff, horizon, config=config, out_dir=str(tmp_path / "replay"),
        training_mode="fixed_refit", fixed_update_budget=selected["selected_updates"],
        replay_checkpoint=str(tmp_path / "refit/model.pt"),
    )
    np.testing.assert_array_equal(
        pq.read_table(final.path)["yhat"].to_numpy(), pq.read_table(replay.path)["yhat"].to_numpy(),
    )


@pytest.fixture(scope="module")
def lightning_study(tmp_path_factory, spark):
    """Two retained series, 168 days each, no exclusions; 56 forecast cells, 54 scored."""
    return freeze_native_study(
        tmp_path_factory.mktemp("lightning-shared-study"), spark, horizon=28, price=True,
    )


@pytest.mark.parametrize("adapter_name,device,precision", [
    ("tft_q", "cpu", "32-true"), ("tft_hnb", "cpu", "32-true"),
    ("tft_q", "mps", "16-mixed"), ("tft_hnb", "mps", "16-mixed"),

])
def test_lightning_study_selection_refit_and_saved_score(
    lightning_study, tmp_path, spark, monkeypatch, adapter_name, device, precision,
):
    from autoforecast.adapters import tft_q
    from autoforecast.forecast_campaign import _selected_update_budget, _verify_fixed_refit_budget

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    path, record = lightning_study
    recipe = record["recipe"]
    origin, horizon = recipe["validation_origins"][0], recipe["validation_horizon"]
    contract_path = path.parent / "dataset.yaml"
    dataset = load_dataset_contract(contract_path)
    fit = tft_q.fit_predict_quantile if adapter_name == "tft_q" else tft_q.fit_predict_hnb
    options = dict(
        hidden_size=8, hidden_continuous_size=4, attention_head_size=2, dropout=0.1,
        learning_rate=0.001, max_encoder_length=30, min_encoder_length=7,
        lag_short_days=2, lag_long_days=14,
    )
    options.update(
        max_epochs=2, batch_size=2, early_stopping_patience=10,
        early_stopping_min_delta=1e-6, device=device, seed=0, precision=precision,
    )
    fixed = benchmark.evaluation_contract(
        path, stage="validation", origin=origin, horizon=horizon, metric="rmse",
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(fixed))
    evaluation_sha = hashlib.sha256((evaluation_path).read_bytes()).hexdigest()

    def input_panel(role, boundary):
        target = tmp_path / (role + "-input")
        _materialize_training_panel(
            benchmark.artifact_path(path, record, role), target, contract_path=contract_path,
            origin=boundary, horizon=horizon, training_start=recipe["training_start"],
            expected_source_sha256=record["artifacts"][role],
            expected_contract_sha256=record["dataset_sha256"],
        )
        # Known-future price rows exist, but no post-origin target reaches the trainer.
        rows = pq.read_table(target).to_pylist()
        assert all(
            row["sales"] == 0 and row["state"] == "unobserved"
            for row in rows if row["date"].isoformat() > boundary
        )
        return dataset.panel(target)

    training_panel = input_panel("selection", origin)
    selection_args = dict(
        panel=training_panel, origin=origin, horizon=horizon, spark=spark,
        out_dir=tmp_path / "selection", checkpoint_out=tmp_path / "selected.pt",
        validation_data_path=fixed["actuals"]["path"], evaluation_contract_path=evaluation_path,
        evaluation_contract_sha256=evaluation_sha, **options,
    )
    forecast = fit(**selection_args)
    saved = torch.load(tmp_path / "selected.pt", map_location="cpu", weights_only=True)
    training = saved["training"]
    assert forecast.n_emitted == 56 and forecast.n_declined == 0
    assert training["selection_metric"] == "rmse"
    assert training["training_contract"]["study_evaluation"] == saved["run_identity"][
        "study_evaluation"
    ]
    assert training["training_contract"]["study_evaluation"]["evaluation_contract_sha256"] == (
        evaluation_sha
    )
    assert all(row["val_loss"] is None for row in training["validation"])
    assert all(row["study_score_dtype"] == "torch.float64" for row in training["validation"])
    score = score_study(
        dataset.panel(fixed["actuals"]["path"]), forecast, spark, contract=fixed,
        output_dir=tmp_path / "scored",
    )
    assert score["n_scored"] == 54 and score["n_dropped_no_data"] == 2
    best = min(training["validation"], key=lambda row: row["study_score"])
    assert training["selected_epoch"] == best["epoch"]
    assert best["study_score"] == pytest.approx(score["rmse"], rel=1e-6)

    def campaign_result(forecast, checkpoint, name):
        record_path = tmp_path / (name + ".json")
        record_path.write_text(json.dumps(forecast.as_record()))
        return {
            "adapter": adapter_name, "origin": forecast.origin,
            "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256((checkpoint).read_bytes()).hexdigest()},
            "forecast_record_path": str(record_path),
            "forecast_record_sha256": hashlib.sha256((record_path).read_bytes()).hexdigest(),
            "budget": {"maximum_epochs": 2, "patience": 10,
                       "early_stopping_absolute_min_delta": 1e-6,
                       "early_stopping_relative_min_delta": 0.0},
        }

    result = campaign_result(forecast, tmp_path / "selected.pt", "selected")
    loss_kind = "quantile" if adapter_name == "tft_q" else "v3_hurdle_nb"
    updates = _selected_update_budget(result, "rmse", 2, expected_loss_kind=loss_kind)
    assert updates == best["updates"]
    for field, value in (("study_score", float("nan")), ("study_score_dtype", "torch.float32"),
                         ("val_loss", 0.0)):
        altered = copy.deepcopy(saved)
        altered["training"]["validation"][0][field] = value
        altered["training"]["epoch_log"][0][field] = value
        checkpoint = tmp_path / (field + ".pt")
        torch.save(altered, checkpoint)
        bad = {**result, "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256((checkpoint).read_bytes()).hexdigest()}}
        with pytest.raises(RuntimeError, match="selection (controls|validation history)"):
            _selected_update_budget(bad, "rmse", 2, expected_loss_kind=loss_kind)
    assert fit(**selection_args) == forecast  # Authenticated completed-artifact adoption.
    final = fit(
        input_panel("final_training", recipe["final_train_cutoff"]),
        recipe["final_train_cutoff"], horizon, spark=spark,
        out_dir=tmp_path / "refit", checkpoint_out=tmp_path / "refit.pt",
        training_mode="fixed_refit", fixed_update_budget=updates, **options,
    )
    refit = torch.load(tmp_path / "refit.pt", map_location="cpu", weights_only=True)["training"]
    assert refit["validation"] == [] and "study_evaluation" not in refit["training_contract"]
    _verify_fixed_refit_budget(
        campaign_result(final, tmp_path / "refit.pt", "refit"), updates, batch_size=2,
    )
