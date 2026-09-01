"""TimesFM-HNB functional contract: real count training/artifacts, fake only frozen source inference."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from autoforecast.adapters import timesfm_hnb as adapter
from autoforecast.panel import Panel, PanelError
from autoforecast.timesfm_hnb_data import features, groups, load_data, missing_profiles
from autoforecast.timesfm_hnb_source import (
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_REVISION,
    SourceRequest,
    TimesFM3Source,
    cached_quantiles,
    file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "2024-02-10"


class FakeSource:
    """Deterministic conditional quantiles; every input channel and its mask can affect output."""

    calls = []

    def __init__(self, device="cpu"):
        self.identity = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION,
            "device": device,
            "weights_sha256": "a" * 64,
            "config_sha256": "b" * 64,
        }

    def predict(self, request):
        self.calls.append(request)
        total = sum(
            float(getattr(request, n).sum())
            for n in ("target", "past_only", "past_future")
            if getattr(request, n) is not None
        )
        values = np.arange(1, 10, dtype=np.float32) + total / 1000
        return np.broadcast_to(values, (len(request.target), request.horizon, 9)).copy()


def panel_fixture(tmp_path, *, business=True, future_target=2, timestamp=False):
    rows = []
    for sid in ("a", "b", "c"):
        for day in pd.date_range("2024-01-01", "2024-02-20"):
            rows.append(
                {
                    "id": sid,
                    "date": day if timestamp else day.date(),
                    "qty": int(day.day % 4) if str(day.date()) <= ORIGIN else future_target,
                    "state": "exact",
                    "price": float(day.day % 3),
                    "competitor": float(day.day % 5),
                    "store": "shop",
                    "cat": "food",
                }
            )
    path = tmp_path / "panel.parquet"
    tmp_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return Panel(
        str(path),
        "id",
        "date",
        "qty",
        exposure_col="state",
        known_future=("price",) if business else (),
        past_only=("competitor",) if business else (),
        price_col="price" if business else None,
        static=("store", "cat"),
        store_col="store",
        category_col="cat",
    )


def _fit(panel, root, **kwargs):
    return adapter.fit_predict(
        panel,
        ORIGIN,
        3,
        out_dir=root / "forecast",
        checkpoint_out=root / "model/model.pt",
        context_days=12,
        group_size=2,
        head_width=8,
        batch_size=2,
        **kwargs,
    )


def _saved(root):
    return torch.load(root / "model/model.pt", weights_only=True)


def _campaign_result(root, forecast):
    record = root / "forecast.json"
    record.write_text(json.dumps(forecast.as_record()))
    return {
        "adapter": "timesfm_hnb",
        "origin": ORIGIN,
        "checkpoint": {
            "path": str(root / "model/model.pt"),
            "sha256": file_sha256(root / "model/model.pt"),
        },
        "forecast_record_path": str(record),
        "forecast_record_sha256": file_sha256(record),
        "budget": {
            "maximum_epochs": 2,
            "patience": 10,
            "early_stopping_absolute_min_delta": 1e-6,
            "early_stopping_relative_min_delta": 0.0,
        },
    }


def test_timesfm_hnb_two_epoch_select_refit_reload(tmp_path, monkeypatch):
    from autoforecast.forecast_campaign import _selected_update_budget, _verify_fixed_refit_budget

    monkeypatch.setattr(adapter, "TimesFM3Source", FakeSource)
    panel = panel_fixture(tmp_path)
    forecast = _fit(panel, tmp_path / "selection", max_epochs=2)
    selected = _saved(tmp_path / "selection")
    assert len(selected["training"]["epoch_log"]) == 2
    assert selected["training"]["n_training_examples"] == 3  # one visit per eligible series
    result = _campaign_result(tmp_path / "selection", forecast)
    updates = _selected_update_budget(result, "wrmsse", 2)
    assert updates in (2, 4)
    expanded = copy.deepcopy(selected)
    expanded["training"]["n_training_examples"] = 4
    expanded_path = tmp_path / "expanded-epoch.pt"
    torch.save(expanded, expanded_path)
    expanded_result = {
        **result,
        "checkpoint": {"path": str(expanded_path), "sha256": file_sha256(expanded_path)},
    }
    with pytest.raises(RuntimeError, match="selection controls"):
        _selected_update_budget(expanded_result, "wrmsse", 2)
    with pytest.raises(RuntimeError, match="selection controls"):
        _selected_update_budget(result, "wrmsse", 3)
    restored = adapter.predict_from_checkpoint(
        panel,
        ORIGIN,
        3,
        checkpoint=tmp_path / "selection/model/model.pt",
        checkpoint_sha256=result["checkpoint"]["sha256"],
        out_dir=tmp_path / "replay",
    )
    pd.testing.assert_frame_equal(pd.read_parquet(forecast.path), pd.read_parquet(restored.path))
    assert _fit(panel, tmp_path / "selection", max_epochs=2).as_record() == forecast.as_record()
    # Partial final epoch is legal: fresh weights, exactly the sealed step frontier.
    final = _fit(
        panel, tmp_path / "refit", max_epochs=2, training_mode="fixed_refit", fixed_update_budget=7
    )
    _verify_fixed_refit_budget(_campaign_result(tmp_path / "refit", final), 7, batch_size=2)
    assert _saved(tmp_path / "refit")["training"]["epoch_log"][-1]["updates"] == 7
    with pytest.raises(ValueError, match="digest"):
        adapter.predict_from_checkpoint(
            panel,
            ORIGIN,
            3,
            checkpoint=tmp_path / "selection/model/model.pt",
            checkpoint_sha256="0" * 64,
            out_dir=tmp_path / "bad",
        )
    changed = dataclasses.replace(panel, known_future=(), past_only=(), price_col=None)
    optional = adapter.predict_from_checkpoint(
        changed,
        ORIGIN,
        3,
        checkpoint=tmp_path / "selection/model/model.pt",
        checkpoint_sha256=result["checkpoint"]["sha256"],
        out_dir=tmp_path / "missing",
    )
    assert optional.n_emitted == 9


def test_training_future_target_invariance(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "TimesFM3Source", FakeSource)
    p1 = panel_fixture(tmp_path / "a", future_target=1)
    p2 = panel_fixture(tmp_path / "b", future_target=99999)
    f1, f2 = _fit(p1, tmp_path / "run-a"), _fit(p2, tmp_path / "run-b")
    pd.testing.assert_frame_equal(pd.read_parquet(f1.path), pd.read_parquet(f2.path))
    a, b = _saved(tmp_path / "run-a"), _saved(tmp_path / "run-b")
    assert a["features"] == b["features"]
    for name in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], rtol=0, atol=0)


@pytest.mark.parametrize("timestamp", [False, True])
def test_optional_covariates_and_past_only_cutoff(tmp_path, timestamp):
    panel = panel_fixture(tmp_path, timestamp=timestamp)
    assert Panel.from_record(dataclasses.asdict(panel)) == panel
    data = load_data(panel, ORIGIN, 3, 12)
    grouping = groups(data, 2, 32)
    assert grouping == ((0, 1), (2,))
    source = FakeSource()
    source.calls = []
    inputs, record = features(data, ORIGIN, 3, 12, grouping, source, tmp_path / "cache")
    assert source.calls[0].target.shape == (2, 12)
    assert source.calls[0].past_only.shape == (2, 12)
    # Values after a window origin in a past-only feed cannot affect that window.
    pos = data.position(ORIGIN)
    data.covariates[:, pos + 1 :, 0] = 123456
    unchanged, same = features(data, ORIGIN, 3, 12, grouping, source, tmp_path / "cache")
    assert same == record
    np.testing.assert_array_equal(inputs["features"], unchanged["features"])
    all_missing, changed = features(
        data, ORIGIN, 3, 12, grouping, source, tmp_path / "cache", absent=data.fields
    )
    assert record["cache"] != changed["cache"]
    assert source.calls[-1].past_future is None and source.calls[-1].past_only is None
    assert not all_missing["features"][..., 11::2].any()
    zero, zr = features(
        data,
        ORIGIN,
        3,
        12,
        grouping,
        source,
        tmp_path / "cache",
        overrides={"price": np.zeros((3, 3), np.float32)},
    )
    absent, ar = features(
        data,
        ORIGIN,
        3,
        12,
        grouping,
        source,
        tmp_path / "cache",
        overrides={"price": np.full((3, 3), np.nan)},
    )
    assert zr["cache"] != ar["cache"]
    assert zero["features"][..., -1].all() and not absent["features"][..., -1].any()
    with pytest.raises(ValueError, match="only declared future"):
        features(
            data,
            ORIGIN,
            3,
            12,
            grouping,
            source,
            tmp_path / "cache",
            overrides={"competitor": np.zeros((3, 3))},
        )
    assert missing_profiles(()) == ((),)
    with pytest.raises(ValueError):
        dataclasses.replace(panel, past_only=("qty",)).validate()


def test_history_only_training_and_invalid_count_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "TimesFM3Source", FakeSource)
    panel = panel_fixture(tmp_path, business=False)
    assert _fit(panel, tmp_path / "run", max_epochs=2).n_emitted == 9
    for mode, budget in (("fixed_refit", None), ("fixed_refit", True), ("fixed_validation", 2)):
        with pytest.raises(ValueError, match="update|budget"):
            _fit(panel, tmp_path / "invalid", training_mode=mode, fixed_update_budget=budget)
    table = pd.read_parquet(panel.path)
    table["qty"] = table["qty"].astype(float)
    table.loc[0, "qty"] = 1.5
    table.to_parquet(panel.path, index=False)
    # This row lies outside the bounded grid; an invalid in-window count is refused.
    table.loc[table.date == dt.date(2024, 2, 1), "qty"] = 1.5
    table.to_parquet(panel.path, index=False)
    with pytest.raises(ValueError, match="nonnegative counts"):
        _fit(panel, tmp_path / "invalid-count")


def test_cache_identity_and_corruption(tmp_path):
    source = FakeSource()
    source.calls = []
    target = np.arange(8, dtype=np.float32).reshape(2, 4)
    mask = np.zeros(target.shape, bool)
    mask[0, 0] = True
    request = SourceRequest(target, mask, 3, ("target:a", "target:b"))
    output, record = cached_quantiles(source, request, tmp_path)
    again, _ = cached_quantiles(source, request, tmp_path)
    assert len(source.calls) == 1
    np.testing.assert_array_equal(output, again)
    target[0, 0] = 999
    assert SourceRequest(target, mask, 3, request.schema).key(source.identity) == record["key"]
    for changed in (
        dataclasses.replace(request, horizon=4),
        dataclasses.replace(request, schema=("target:b", "target:a")),
        dataclasses.replace(request, target_mask=np.zeros(target.shape, bool)),
        dataclasses.replace(request, target=request.target + 1),
    ):
        assert changed.key(source.identity) != record["key"]
    assert request.key({**source.identity, "weights_sha256": "c" * 64}) != record["key"]
    (tmp_path / f"{record['key']}.npz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="invalid TimesFM-HNB source cache"):
        cached_quantiles(source, request, tmp_path)

    def mutate(r):
        source.identity["weights_sha256"] = "changed"
        return output

    source.predict = mutate
    with pytest.raises(ValueError, match="changed during inference"):
        cached_quantiles(source, request, tmp_path / "mutation")


def test_timesfm_hnb_count_head_law():
    from forecast.models.timesfm_hnb import CountHead

    head = CountHead(n_features=2, n_static=0, cardinalities=[], n_series=1, width=4)
    batch = {
        "features": torch.zeros(1, 3, 2),
        "static": torch.zeros(1, 0),
        "categories": torch.zeros(1, 0, dtype=torch.long),
        "series_index": torch.zeros(1, dtype=torch.long),
        "baseline": torch.tensor([[0.0, np.log(2)]], dtype=torch.float32),
    }
    pred = head(batch)
    torch.testing.assert_close(pred.mean(), torch.full((1, 3), 1.5, dtype=torch.float64))
    # P(0)=.5, P(1)=.5/(1+2), P(2)=.5*(2/3)/(1+2).
    expected = -torch.tensor([[0.5, 1 / 6, 1 / 9]], dtype=torch.float64).log()
    torch.testing.assert_close(
        pred.nll(torch.tensor([[0, 1, 2.0]]), torch.ones(1, 3, dtype=torch.bool)), expected
    )


def test_timesfm_hnb_execution_and_registration(tmp_path):
    from scripts.run_autoforecast import _worker_kwargs

    from autoforecast.adapters import known_adapter_names, validate_execution_controls
    from autoforecast.model_catalog import discover_candidates
    from autoforecast.model_execution import resolve_execution_plan
    from autoforecast.model_readiness import create_resolved_run
    from autoforecast.parameter_sources import default_parameter_resolution
    from autoforecast.test_model_execution import _facts

    candidate = next(
        c for c in discover_candidates(ROOT / "configs/models") if c.id == "timesfm_hnb"
    )
    panel = panel_fixture(tmp_path)
    resolution = default_parameter_resolution(candidate)
    plan = resolve_execution_plan(
        candidate,
        panel,
        _facts(),
        horizon=3,
        epochs=2,
        parameter_overrides=resolution["effective_parameters"],
    )
    assert plan.as_record()["adapter_kwargs"]["context_days"] == 371
    assert "timesfm_hnb" in known_adapter_names()
    run = create_resolved_run(
        model_id=candidate.id, adapter=candidate.adapter, origin=ORIGIN, horizon=3
    )
    context = SimpleNamespace(input=SimpleNamespace(run=run), attempt_dir=tmp_path)
    controls = {
        "batch_size": 2,
        "device": "cpu",
        "patience": 10,
        "precision": "32-true",
        "early_stopping_absolute_min_delta": 1e-6,
    }
    kwargs = _worker_kwargs(
        context,
        plan,
        parent_checkpoint=None,
        training_controls=controls,
        calendar_events=(),
        event_max_support=30,
        training_mode="fixed_refit",
        validation_data_path=None,
        fixed_update_budget=7,
        validation_metric="wrmsse",
    )
    assert kwargs["fixed_update_budget"] == 7 and kwargs["training_mode"] == "fixed_refit"
    assert kwargs["checkpoint_out"] == str(tmp_path / "model/model.pt")
    with pytest.raises(ValueError, match="precision"):
        validate_execution_controls(
            "timesfm_hnb",
            precision="16-mixed",
            device="cpu",
            batch_size=2,
            gradient_microbatch_size=None,
            has_generic_events=False,
        )


@pytest.mark.parametrize("context,horizon", [(12, 8), (11, 5)])
def test_source_identity_and_masked_provider(tmp_path, context, horizon):
    """Official tiny model: low-level adapter versus provider and target-only extraction."""
    pytest.importorskip("timesfm3")
    from timesfm3.model import TimesFM3Torch
    from timesfm3.timesfm3_forecaster import TimesFM3Forecaster

    torch.manual_seed(7)
    model = TimesFM3Torch(
        input_patch_len=4,
        output_patch_len=8,
        residual_block_config={
            "hidden_dims": 16,
            "output_dims": 16,
            "use_bias": False,
            "activation": "relu",
        },
        transformer_config={
            "num_layers": 1,
            "transformer": {
                "model_dims": 16,
                "hidden_dims": 16,
                "num_heads": 2,
                "use_rope_var": False,
                "use_rope_seq": True,
                "use_bias": False,
                "attention_norm": "rms",
                "feedforward_norm": "rms",
                "qk_norm": "rms",
                "ff_activation": "relu",
                "deterministic": True,
            },
        },
    )
    model.pre_transformer_resblock.set_input_dims(24)
    model.save_pretrained(tmp_path / "tiny")
    provider = TimesFM3Forecaster.from_pretrained(str(tmp_path / "tiny"), device="cpu")
    source = TimesFM3Source.__new__(TimesFM3Source)
    source.provider = provider
    source.model = provider.model.eval().requires_grad_(False)
    target = np.arange(2 * context, dtype=np.float32).reshape(2, context) % 7
    cov = np.arange(context + horizon, dtype=np.float32)[None, :]
    request = SourceRequest(
        target,
        np.zeros(target.shape, bool),
        horizon,
        ("target:a", "target:b", "future:p"),
        past_future=cov,
        past_future_mask=np.zeros(cov.shape, bool),
    )
    output = source.predict(request)
    official = provider.predict(
        target,
        horizon,
        past_future_covariates=cov,
        return_quantiles=True,
        use_symmetric_averaging=False,
        use_znorm=False,
        padding_mode="none",
    )
    np.testing.assert_allclose(output, official.quantiles, atol=1e-5, rtol=1e-5)
    assert output.shape == (2, horizon, 9)
    mask = np.zeros(target.shape, bool)
    mask[:, 2] = True
    future_mask = np.zeros(cov.shape, bool)
    future_mask[:, 1] = True
    future_mask[:, -1] = True
    missing = dataclasses.replace(request, target_mask=mask, past_future_mask=future_mask)
    masked_output = source.predict(missing)
    # Independent low-level call proves masks reached the official model, not just the key.
    direct = provider.model.decode(
        target=torch.tensor(missing.target).unsqueeze(0),
        horizon=horizon,
        target_mask=torch.tensor(mask).unsqueeze(0),
        past_future_covariates=torch.tensor(missing.past_future).unsqueeze(0),
        past_future_mask=torch.tensor(future_mask).unsqueeze(0),
    )[0, :2].numpy()
    np.testing.assert_allclose(masked_output, np.sort(direct, axis=-1), atol=1e-5, rtol=1e-5)
    zeros = dataclasses.replace(
        missing,
        target_mask=np.zeros(target.shape, bool),
        past_future_mask=np.zeros(cov.shape, bool),
    )
    assert not np.allclose(masked_output, source.predict(zeros), atol=1e-5, rtol=1e-5)
    assert all(p.grad is None and not p.requires_grad for p in source.model.parameters())
    source.model.train()
    with pytest.raises(ValueError, match="frozen"):
        source.predict(request)


def test_plan_asof_and_scenario(tmp_path):
    from pyspark.sql import SparkSession

    from autoforecast.panel import PlanSource

    panel = panel_fixture(tmp_path)
    plan_path = tmp_path / "plans.parquet"
    rows = [
        {
            "id": sid,
            "valid": dt.date(2024, 2, 11),
            "known": dt.date(2024, 2, day),
            "planned_price": value,
        }
        for sid in ("a", "b", "c")
        for day, value in ((9, 7.0), (12, 999.0))
    ]
    pq.write_table(pa.Table.from_pylist(rows), plan_path)
    panel = dataclasses.replace(
        panel, plan=PlanSource(str(plan_path), "valid", "known", price_col="planned_price")
    )
    data = load_data(panel, ORIGIN, 3, 12)
    source = FakeSource()
    source.calls = []
    spark = (
        SparkSession.builder.master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    try:
        inputs, record = features(
            data, ORIGIN, 3, 12, groups(data, 2, 32), source, tmp_path / "cache", spark=spark
        )
        assert record["plan"]["n_rows"] == 3
        assert source.calls[0].past_future[0, 12] == 7
        assert source.calls[0].past_future_mask[0, 13:].all()
        scenarios, changed = features(
            data,
            ORIGIN,
            3,
            12,
            groups(data, 2, 32),
            source,
            tmp_path / "cache",
            spark=spark,
            overrides={"price": np.full((3, 3), 13.0)},
        )
        assert changed["cache"] != record["cache"]
        assert not np.array_equal(inputs["features"], scenarios["features"])
    finally:
        spark.stop()


def test_campaign_binds_timesfm_hnb_executed_recipe(tmp_path, monkeypatch):
    import copy

    from autoforecast import forecast_campaign as campaign

    monkeypatch.setattr(adapter, "TimesFM3Source", FakeSource)
    panel = panel_fixture(tmp_path)
    _fit(panel, tmp_path / "fit")
    saved = _saved(tmp_path / "fit")
    monkeypatch.setattr(
        campaign, "load_dataset_contract", lambda p: SimpleNamespace(panel=lambda p: panel)
    )
    parameters = {
        "effective_parameters": {
            "model.context_days": 12,
            "model.group_size": 2,
            "model.max_variates": 32,
            "model.head_width": 8,
            "training.learning_rate": 0.0003,
        "training.mean_loss_weight": 0.0,
        "training.train_windows": 1, "training.train_stride": 28,
        }
    }
    child = {
        "adapter": "timesfm_hnb",
        "kind": "validation",
        "origin": ORIGIN,
        "horizon": 3,
        "seed": 0,
        "budget": {"maximum_epochs": 2, "patience": 10, "early_stopping_absolute_min_delta": 1e-6},
    }
    resolved = {
        "purpose": "forecast_campaign",
        "compute": {"device": "cpu", "batch_size": 2},
        "selection": {"metric": "wrmsse"},
        "source": {"dataset_contract": "unused.yaml"},
    }
    digest = saved["run_identity"]["panel_sha256"]
    campaign._verify_revision_checkpoint_binding(
        saved, resolved, child, parameters, Path(panel.path), digest
    )
    for field, value in (
        ("learning_rate", 0.1),
        ("training_mode", "fixed_refit"),
        ("panel_sha256", "f" * 64),
        ("batch_size", 999),
        ("epoch_definition", "expanded_windows_and_profiles"),
    ):
        modified = copy.deepcopy(saved)
        modified["run_identity"][field] = value
        with pytest.raises(campaign.ForecastCampaignError, match="recipe"):
            campaign._verify_revision_checkpoint_binding(
                modified, resolved, child, parameters, Path(panel.path), digest
            )
    modified = copy.deepcopy(saved)
    modified["run_identity"]["source"]["model_revision"] = "wrong"
    with pytest.raises(campaign.ForecastCampaignError, match="recipe"):
        campaign._verify_revision_checkpoint_binding(
            modified, resolved, child, parameters, Path(panel.path), digest
        )
    modified = copy.deepcopy(saved)
    modified["training"]["epoch_definition"] = "expanded_windows_and_profiles"
    with pytest.raises(campaign.ForecastCampaignError, match="recipe"):
        campaign._verify_revision_checkpoint_binding(
            modified, resolved, child, parameters, Path(panel.path), digest
        )


@pytest.mark.parametrize("hour", [0, 12])
def test_naive_timestamp_calendar_dates_preserve_last_horizon_day(tmp_path, hour):
    reference = panel_fixture(tmp_path / "date")
    panel = panel_fixture(tmp_path / "time", timestamp=True)
    table = pd.read_parquet(panel.path)
    table["date"] = pd.to_datetime(table["date"]) + pd.Timedelta(hours=hour)
    table.to_parquet(panel.path, index=False)
    outputs = []
    for p in (reference, panel):
        data = load_data(p, ORIGIN, 3, 12)
        src = FakeSource()
        src.calls = []
        inputs, record = features(
            data, ORIGIN, 3, 12, groups(data, 2, 32), src, tmp_path / p.path.split("/")[-2]
        )
        outputs.append((inputs, record))
        assert not src.calls[0].past_future_mask[:, -3:].any()
        assert src.calls[0].past_future[0, -1] == 1.0
        assert src.calls[0].past_future.shape[1] == 15  # excludes following date
    for name in outputs[0][0]:
        np.testing.assert_array_equal(outputs[0][0][name], outputs[1][0][name])
    assert outputs[0][1] == outputs[1][1]


@pytest.mark.parametrize("zone", ["UTC", "Asia/Kolkata", "America/New_York"])
def test_timestamp_calendar_loader_rejects_timezone_bearing_keys(tmp_path, zone):
    panel = panel_fixture(tmp_path, timestamp=True)
    table = pd.read_parquet(panel.path)
    table["date"] = pd.to_datetime(table["date"]).dt.tz_localize(zone)
    table.to_parquet(panel.path, index=False)

    with pytest.raises(PanelError, match="timezone-naive"):
        load_data(panel, ORIGIN, 3, 12)


def test_stockout_replay_schema_add_rename_retain_omit(tmp_path):
    panel = panel_fixture(tmp_path)
    data = load_data(panel, ORIGIN, 3, 12)
    table = pd.read_parquet(panel.path)
    table["stockout"] = True
    table["shelf_missing"] = False
    table.to_parquet(panel.path, index=False)
    with_stock = dataclasses.replace(panel, oos_col="stockout")
    with pytest.raises(ValueError, match="new covariate schema"):
        load_data(with_stock, ORIGIN, 3, 12, schema=data.schema)
    trained = load_data(with_stock, ORIGIN, 3, 12)
    retained = load_data(with_stock, ORIGIN, 3, 12, schema=trained.schema)
    np.testing.assert_array_equal(trained.covariates, retained.covariates)
    renamed = dataclasses.replace(with_stock, oos_col="shelf_missing")
    with pytest.raises(ValueError, match="new covariate schema"):
        load_data(renamed, ORIGIN, 3, 12, schema=trained.schema)
    missing = load_data(panel, ORIGIN, 3, 12, schema=trained.schema)
    assert missing.schema == trained.schema
    assert np.isnan(missing.covariates[..., missing.fields.index("stockout")]).all()


def test_unobserved_labels_and_cold_series_preserve_replay(tmp_path, monkeypatch):
    """All-zero observed series remain admitted; unlabeled examples never divide by zero."""
    monkeypatch.setattr(adapter, "TimesFM3Source", FakeSource)
    panel = panel_fixture(tmp_path)
    table = pd.read_parquet(panel.path)
    table.loc[table.id == "b", "qty"] = 0
    table.loc[table.id == "c", "state"] = "unobserved"
    extra = table[table.id == "a"].copy()
    extra["id"] = "d"
    extra.loc[extra.date.astype(str) >= "2024-02-05", "state"] = "unobserved"
    future_only = table[(table.id == "a") & (table.date.astype(str) > ORIGIN)].copy()
    future_only["id"] = "e"
    future_only["store"] = "future-only-shop"
    future_only["qty"] = 999
    pd.concat([table, extra, future_only]).to_parquet(panel.path, index=False)
    data = load_data(panel, ORIGIN, 3, 12)
    future_row = data.series.index("e")
    assert not data.observed[future_row].any()
    assert not data.target[future_row].any()
    assert data.metadata["store"][future_row] == "<missing>"
    assert not data.categories[future_row].any()
    forecast = _fit(panel, tmp_path / "run")
    checkpoint = tmp_path / "run/model/model.pt"
    replay = adapter.predict_from_checkpoint(
        panel,
        ORIGIN,
        3,
        checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        out_dir=tmp_path / "replay",
    )
    pd.testing.assert_frame_equal(pd.read_parquet(forecast.path), pd.read_parquet(replay.path))
    assert (forecast.n_series, forecast.n_emitted, forecast.n_declined) == (5, 9, 6)
    assert replay.disposition == forecast.disposition
    assert replay.n_series == forecast.n_series
    assert _saved(tmp_path / "run")["training"]["n_training_examples"] == 2
    assert set(pd.read_parquet(forecast.path).series_id) == {"a", "b", "d"}


def test_source_revision_rejected_before_weight_download(monkeypatch):
    pytest.importorskip("timesfm3")
    import importlib.metadata

    import huggingface_hub

    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            version="3.0.0", read_text=lambda name: '{"vcs_info":{"commit_id":"wrong"}}'
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("download reached before source revision validation")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    with pytest.raises(ValueError, match="pinned timesfm3"):
        TimesFM3Source("cpu")
