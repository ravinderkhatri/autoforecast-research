"""Optional feed families, shared calendar, rolling windows and HMFNet-style mean loss."""

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from autoforecast.m5_timesfm_features import calendar_features, enrich_batch, materialize
from autoforecast.test_timesfm_hnb import ORIGIN, FakeSource, _fit, _saved, panel_fixture
from autoforecast.timesfm_hnb_data import features, groups, load_data, missing_profiles


def test_shared_calendar_and_family_absence(tmp_path):
    panel = panel_fixture(tmp_path)
    panel = dataclasses.replace(
        panel, shared_known_future=("price",), covariate_families=(("price", "competitor"),)
    )
    data = load_data(panel, ORIGIN, 3, 12)
    grouping = groups(data, 2, 5)
    assert len(data.local_fields) == 1
    assert missing_profiles(data.fields, panel.covariate_families) == ((), data.fields)
    # Canonical family ordering does not change which fields are removed.
    inputs, _ = features(data, ORIGIN, 3, 12, grouping, FakeSource(), tmp_path / "cache")
    request = FakeSource.calls[-2]
    assert request.past_future.shape[0] == 1
    assert "past_future:shared:price" in request.schema
    omitted, _ = features(
        data, ORIGIN, 3, 12, grouping, FakeSource(), tmp_path / "cache", absent=data.fields
    )
    assert np.all(omitted["features"][..., 10:] == 0)
    assert not np.array_equal(inputs["features"][..., :9], omitted["features"][..., :9])
    frame = pd.read_parquet(panel.path)
    frame.loc[(frame.id == "a") & (frame.date == dt.date.fromisoformat(ORIGIN)), "price"] = 99
    frame.to_parquet(panel.path, index=False)
    with pytest.raises(ValueError, match="shared field"):
        load_data(panel, ORIGIN, 3, 12)


def test_mean_loss_is_scaled_and_masked():
    from forecast.models.timesfm_hnb import CountModule

    model = CountModule(
        {"n_features": 1, "n_static": 0, "cardinalities": [], "n_series": 2, "width": 4},
        0.001,
        0,
        0.2,
    )
    batch = {
        "features": torch.zeros(2, 2, 1),
        "baseline": torch.zeros(2, 2),
        "static": torch.zeros(2, 0),
        "categories": torch.zeros(2, 0, dtype=torch.long),
        "series_index": torch.arange(2),
        "target": torch.tensor([[0.0, 2.0], [4.0, 999.0]]),
        "observed": torch.tensor([[True, True], [True, False]]),
        "mean_scale": torch.tensor([1.0, 4.0]),
    }
    # Baseline logit=0, excess=1 -> hurdle mean=0.5*(1+1)=1.
    prediction = model.head(batch)
    nll = (prediction.nll(batch["target"], batch["observed"]).sum(-1) / torch.tensor([2, 1])).mean()
    # MSE series 1=(1+1)/2=1; series 2=(9+masked)/2/4=1.125.
    torch.testing.assert_close(model._loss(batch), nll + 0.2 * (1 + 1.125) / 2)
    batch["target"][1, 1] = 100000
    torch.testing.assert_close(model._loss(batch), nll + 0.2 * (1 + 1.125) / 2)


def test_rolling_frozen_selection_and_refit(tmp_path, monkeypatch):
    from autoforecast.adapters import timesfm_hnb

    monkeypatch.setattr(timesfm_hnb, "TimesFM3Source", FakeSource)
    panel = panel_fixture(tmp_path)
    _fit(
        panel,
        tmp_path / "selection",
        max_epochs=1,
        train_windows=2,
        train_stride=3,
        mean_loss_weight=0.2,
    )
    saved = _saved(tmp_path / "selection")
    assert saved["training"]["n_training_examples"] == 3
    assert saved["training"]["steps_per_epoch"] == 2
    _fit(
        panel,
        tmp_path / "refit",
        train_windows=2,
        train_stride=3,
        mean_loss_weight=0.2,
        training_mode="fixed_refit",
        fixed_update_budget=3,
    )
    assert _saved(tmp_path / "refit")["training"]["selected_updates"] == 3


def test_calendar_join_preserves_targets_and_uses_both_event_slots(tmp_path):
    path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "event_name_1": ["NewYear", None],
            "event_name_2": ["Other", None],
            "event_type_1": ["National", None],
            "event_type_2": ["Cultural", None],
            "snap_CA": [1, 0],
            "snap_TX": [0, 1],
            "snap_WI": [0, 0],
        }
    ).to_csv(path, index=False)
    calendar, shared = calendar_features(path)
    source = pd.DataFrame(
        {
            "id": ["a", "b"],
            "date": [dt.date(2024, 1, 1)] * 2,
            "qty": [2, 999],
            "state_id": ["CA", "TX"],
            "observation_state": ["exact", "unobserved"],
        }
    )
    result = enrich_batch(source, calendar)
    assert result.qty.tolist() == [2, 999]
    assert result.snap.tolist() == [1, 0]
    assert result.cal_event_name_other.tolist() == [1, 1]
    assert "cal_event_name_newyear" in shared
    pq.write_table(pa.Table.from_pandas(source, preserve_index=False), tmp_path / "source.parquet")
    record = materialize(tmp_path / "source.parquet", tmp_path / "destination", path)
    assert record["rows"] == 2
    assert materialize(tmp_path / "source.parquet", tmp_path / "destination", path) == record
    (tmp_path / "destination/extra.parquet").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="provenance"):
        materialize(tmp_path / "source.parquet", tmp_path / "destination", path)


def test_large_panel_slices_match_single_slice(tmp_path, monkeypatch):
    import autoforecast.timesfm_hnb_data as module
    from autoforecast.panel import Panel

    rows = [
        {
            "id": f"{i:04d}",
            "date": day.date(),
            "qty": i % 4,
            "event": float(day.day % 2),
            "kind": str(i % 3),
        }
        for i in range(2050)
        for day in pd.date_range("2024-02-05", "2024-02-13")
    ]
    path = tmp_path / "large.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    panel = Panel(
        str(path),
        "id",
        "date",
        "qty",
        known_future=("event",),
        shared_known_future=("event",),
        static=("kind",),
    )
    sliced = module.load_data(panel, ORIGIN, 3, 3)
    direct = module._load_data(panel, ORIGIN, 3, 3)
    assert sliced.schema == direct.schema
    for field in ("target", "observed", "covariates", "static", "categories", "shared_values"):
        np.testing.assert_array_equal(getattr(sliced, field), getattr(direct, field))
    replay = module.load_data(panel, ORIGIN, 3, 3, schema=sliced.schema)
    assert replay.schema == sliced.schema
    frame = pd.read_parquet(path)
    frame.loc[frame.id.isin(["2048", "2049"]), "event"] = 12
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="shared fields differ"):
        module.load_data(panel, ORIGIN, 3, 3)


def test_comparison_runs_timesfm_hnb_before_v8_and_stops_on_failure(monkeypatch, tmp_path):
    from scripts import run_m5_timesfm_comparison as runner

    calls = []

    def run(action, request, data_root):
        calls.append((action, request, data_root))
        return {"status": "planned"}

    monkeypatch.setattr(runner, "run_campaign", run)
    runner.run_pair("prepare", tmp_path, device="mps")
    assert "hnb_2epoch_mps" in calls[0][1].name and "v8_2epoch_mps" in calls[1][1].name
    assert calls[0][2] == calls[1][2]
    calls.clear()

    def fail(action, request, data_root):
        calls.append(request)
        raise RuntimeError("TimesFM-HNB failed")

    monkeypatch.setattr(runner, "run_campaign", fail)
    with pytest.raises(RuntimeError, match="TimesFM-HNB failed"):
        runner.run_pair("run", tmp_path, device="mps")
    assert len(calls) == 1
