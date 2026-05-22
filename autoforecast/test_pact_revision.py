"""Registered PACT assembly, campaign scenarios and sealed fit/replay smoke tests."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from forecast.models.mft.pact import PACTBatch, PACTModelInputs
from forecast.scenario.pact_campaign import evaluate_pact_campaign_pair

from autoforecast.adapters.mft import (
    PACTRevisionFeatures,
    fit_predict_revision,
    revision_data_spec,
)
from autoforecast.panel import Panel
from autoforecast.rolling_data import PanelArrays
from autoforecast.rolling_training import RollingConfig, rolling_origins

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _panel(tmp_path) -> tuple[Panel, str, int]:
    start = dt.date(2024, 1, 1)
    history, horizon = 112, 7
    rows = []
    for series_index, series in enumerate(("a", "b")):
        for offset in range(history + horizon):
            historical = offset < history
            rows.append(
                {
                    "series": series,
                    "date": start + dt.timedelta(days=offset),
                    "sales": (offset + series_index) % 5 if historical else None,
                    "state": (
                        "lower_bound" if historical and series == "a" and offset == 50 else "exact"
                    ),
                    "campaign": float(offset % 3 == 0),
                }
            )
    path = tmp_path / "panel.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    origin = (start + dt.timedelta(days=history - 1)).isoformat()
    return (
        Panel(
            str(path),
            "series",
            "date",
            "sales",
            exposure_col="state",
            known_future=("campaign",),
            target_estimand="sales_given_availability",
        ),
        origin,
        horizon,
    )


def _config() -> RollingConfig:
    raw = json.loads(
        (ROOT / "configs/experiments/multibranch_revisions/pact_campaign_default.json").read_text()
    )
    raw.update(
        encoder_days=28,
        d_model=16,
        n_heads=2,
        trunk_stages=2,
        train_windows=7,
        train_stride=1,
        max_updates=2,
        batch_size=2,
        evaluate_every=1,
        epochs=1,
        patience=2,
        torch_threads=1,
        progress="off",
        max_array_bytes=10_000_000,
    )
    raw["v5"] = {**raw["v5"], "n_layers": 2, "span": 56}
    return RollingConfig(**raw)


def test_pact_revision_batch_campaign_and_causal_fallback(tmp_path) -> None:
    panel, origin, horizon = _panel(tmp_path)
    config = _config()
    spec = revision_data_spec(panel, horizon, config)
    data = PanelArrays.read(
        panel,
        origin,
        horizon,
        max_bytes=config.max_array_bytes,
        data_spec=spec,
        route_lower_bound_to_exact=True,
        observation_contract=config.observation_contract,
    )
    assert data.physical_state is not None and (data.physical_state == 1).sum() == 1
    features = PACTRevisionFeatures(panel, data, config, data_spec=spec)
    selection_end = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    origins = rolling_origins(data, selection_end, config)
    features.begin_phase(origins, "selection")
    inputs, target, objective = features.batch(origins[0], np.arange(2), "cpu", True)
    assert isinstance(inputs, PACTBatch) and inputs.labels is not None
    model = features.make_model()
    prediction, nll = features.validation_values(model, inputs)
    assert prediction.shape == target.shape == objective.shape == (2, horizon)
    assert torch.isfinite(nll(target, objective)).all()
    assert torch.isfinite(features.training_loss(model, inputs, target, objective, 14))
    double_inputs = {
        name: value.double() if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for name, value in inputs.inputs.as_kwargs().items()
    }
    double_prediction = features.make_model().double()(**double_inputs)
    assert double_prediction.mu.dtype == torch.float64
    assert double_prediction.allocation.dtype == torch.float64

    campaign = "future:campaign:0"
    reference = features.scenario_inputs(
        origin,
        np.arange(2),
        "cpu",
        {campaign: np.zeros((2, horizon))},
        known_at=origin,
    )
    planned = features.scenario_inputs(
        origin,
        np.arange(2),
        "cpu",
        {campaign: np.ones((2, horizon))},
        known_at=origin,
    )
    assert isinstance(reference, PACTModelInputs)
    assert isinstance(planned, PACTModelInputs)
    record = evaluate_pact_campaign_pair(model, planned, reference)
    assert record["effect_type"] == "prospective_model_response"
    assert record["causal_uplift"]["status"] == "not_identifiable"
    with pytest.raises(ValueError, match="known no later"):
        features.scenario_inputs(
            origin,
            np.arange(2),
            "cpu",
            {campaign: np.ones((2, horizon))},
            known_at=(dt.date.fromisoformat(origin) + dt.timedelta(days=1)).isoformat(),
        )
    with pytest.raises(ValueError, match="not available"):
        features.scenario_inputs(
            origin,
            np.arange(2),
            "cpu",
            {
                campaign: np.ones((2, horizon)),
                f"{campaign}:available": np.zeros((2, horizon)),
            },
            known_at=origin,
        )


def test_pact_revision_objective_mask_routes_physical_states(tmp_path) -> None:
    panel, origin, horizon = _panel(tmp_path)
    config = _config()
    spec = revision_data_spec(panel, horizon, config)
    data = PanelArrays.read(
        panel,
        origin,
        horizon,
        max_bytes=config.max_array_bytes,
        data_spec=spec,
        route_lower_bound_to_exact=True,
        observation_contract=config.observation_contract,
    )
    features = PACTRevisionFeatures(panel, data, config, data_spec=spec)
    selection_end = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    training_origin = rolling_origins(data, selection_end, config)[0]

    mask = features.objective_mask(training_origin)

    assert mask.shape == (2, horizon)
    assert mask.dtype == np.bool_
    assert mask.all()


@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
def test_pact_fresh_fit_and_exact_replay(tmp_path, precision) -> None:
    panel, origin, horizon = _panel(tmp_path)
    config = _config()
    fresh = tmp_path / "fresh"
    replay = tmp_path / "replay"
    first = fit_predict_revision(
        panel, origin, horizon, config=config, out_dir=str(fresh), precision=precision
    )
    second = fit_predict_revision(
        panel,
        origin,
        horizon,
        config=config,
        out_dir=str(replay),
        replay_checkpoint=str(fresh / "model.pt"),
        precision=precision,
    )
    first_values = pq.read_table(first.path)["yhat"].to_numpy()
    second_values = pq.read_table(second.path)["yhat"].to_numpy()
    np.testing.assert_array_equal(first_values, second_values)
    complete = json.loads((replay / "COMPLETE.json").read_text())
    assert complete["binding"]["schema"] == "multibranch_revision_v5"
    if precision == "16-mixed":
        assert complete["binding"]["precision"] == "16-mixed"
    assert complete["binding"]["effective_assembly"]["v5_layout"]["head_law"] == (
        "zero_truncated_hurdle_negbin"
    )
    assert dataclasses.asdict(config)["v5"]["n_layers"] == 2

    assert config.v5 is not None
    changed_rope = dataclasses.replace(
        config,
        v5=dataclasses.replace(config.v5, rope_theta=config.v5.rope_theta * 2),
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        fit_predict_revision(
            panel,
            origin,
            horizon,
            config=changed_rope,
            out_dir=str(tmp_path / "changed-rope-replay"),
            replay_checkpoint=str(fresh / "model.pt"),
        )
