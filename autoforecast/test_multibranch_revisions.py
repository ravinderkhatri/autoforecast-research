"""Real M5 boundary and fit/replay tests for the development revisions.

Tiny fits test wiring only, never estimate forecasting quality. The masked public panel is
the source: these tests do not load the archived private labels or the public scoring panel.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import torch

from autoforecast.adapters.mft import (
    HMFNetRevisionFeatures,
    RevisionFeatures,
    fit_predict_revision,
    revision_data_spec,
)
from autoforecast.benchmarks.m5 import M5_LEVELS, build_revision_calendar
from autoforecast.model_names import REVISION_MODELS
from autoforecast.panel import Panel
from autoforecast.rolling_data import GLOBAL_LEFT_PADDING_CONTRACT, PanelArrays, ValidationWindow
from autoforecast.rolling_training import (
    RollingConfig,
    informative_target_diagnostic,
    rolling_origins,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_validation_window_reads_exact_horizon_in_training_series_order(tmp_path):
    origin = dt.date(2024, 1, 1)
    rows = [
        {"series": series, "date": origin + dt.timedelta(days=offset), "count": count}
        for series, offset, count in (
            ("b", 1, 20),
            ("a", 2, 11),
            ("a", 1, 10),
            ("b", 2, 21),
        )
    ]
    path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    panel = Panel(str(path), "series", "date", "count", target_estimand="observed_count")

    window = ValidationWindow.read(panel, origin.isoformat(), 2, ("a", "b"))

    assert window.origin == origin.isoformat()
    np.testing.assert_array_equal(window.target, [[10, 11], [20, 21]])
    assert window.observed.all()
    assert not window.target.flags.writeable
    assert not window.observed.flags.writeable

    incomplete = tmp_path / "incomplete.parquet"
    pq.write_table(pa.Table.from_pylist(rows[:-1]), incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        ValidationWindow.read(
            dataclasses.replace(panel, path=str(incomplete)), origin.isoformat(), 2, ("a", "b")
        )


@pytest.fixture
def synthetic_revision_data(tmp_path):
    """Small complete panel that keeps the full-history path exercised in clean CI."""
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=offset) for offset in range(84)]
    rows = [
        {
            "series": series,
            "date": day,
            "count": (-999 if series == "a" and offset == 10 else (offset + series_index) % 5),
            "state": "unobserved" if series == "a" and offset == 10 else "exact",
            "group": "all",
            "temperature": float(offset + series_index),
            "weather": "sun" if offset % 2 else "rain",
        }
        for series_index, series in enumerate(("a", "b"))
        for offset, day in enumerate(dates)
    ]
    path = tmp_path / "panel.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    panel = Panel(
        str(path),
        "series",
        "date",
        "count",
        exposure_col="state",
        static=("group",),
        target_estimand="observed_count",
    )
    return panel, dates[-1].isoformat(), 7


def full_history_test_config(variant: str, **changes: Any) -> RollingConfig:
    """Two-epoch, small-width form of the maintained SFT/HMFNet profile."""
    profile = json.loads(
        (
            ROOT / f"configs/experiments/multibranch_revisions/{REVISION_MODELS[variant]}_campaign_default.json"
        ).read_text()
    )
    profile.update(
        encoder_days=1911,
        d_model=8,
        n_heads=2,
        trunk_stages=1,
        decoder_layers=1,
        history_attention_layers=1,
        commercial_attention_layers=1,
        train_windows=3,
        train_stride=7,
        epochs=2,
        batch_size=2,
        torch_threads=1,
        progress="off",
        max_array_bytes=10_000_000,
    )
    if variant == "v3":
        profile["v3"] = {**profile["v3"], "local_blocks": 1}
    profile.update(changes)
    return RollingConfig(**profile)


@pytest.fixture(scope="module")
def revision_data(tmp_path_factory):
    """Twelve real series, complete historical dates, original masked future."""
    source = ROOT / "data/m5_public_training_d1913_v2"
    if not source.exists():
        pytest.skip("real M5 masked panel is not installed")
    out = tmp_path_factory.mktemp("multibranch-revision-real-m5")
    condition = (
        ds.field("store_id").isin(["CA_1", "CA_2", "TX_1"])  # pyright: ignore[reportPrivateImportUsage]
        & ds.field("item_id").isin(
            [
                "FOODS_1_001",
                "FOODS_1_002",  # pyright: ignore[reportPrivateImportUsage]
                "HOBBIES_1_001",
                "HOUSEHOLD_1_001",
            ]
        )
    )
    source_table = ds.dataset(source, format="parquet", partitioning="hive").to_table(
        filter=condition
    )
    pq.write_table(source_table, out / "panel.parquet")
    calendar = build_revision_calendar(out / "calendar.parquet")
    panel = Panel(
        path=str(out / "panel.parquet"),
        series_col="id",
        time_col="date",
        target_col="qty",
        exposure_col="observation_state",
        price_col="sell_price",
        known_future=("sell_price",),
        static=("dept_id", "cat_id", "state_id", "store_id"),
        factor_group="state_id",
        product_col="item_id",
        store_col="store_id",
        category_col="dept_id",
        frequency="D",
        season_length=7,
    )
    data = PanelArrays.read(panel, "2016-04-24", 28, max_bytes=2_000_000)
    return panel, data, calendar


def small_config(variant="v2", **changes):
    """Structural smoke recipe; the checked-in profiles retain the full update budgets."""
    values: dict[str, Any] = dict(
        variant=variant,
        encoder_days=28 if variant == "v1_corrected" else 112,
        d_model=16,
        n_heads=4,
        trunk_stages=1,
        decoder_layers=1,
        demand_dropout=0.1,
        max_updates=6,
        evaluate_every=1,
        patience=2,
        train_windows=3,
        batch_size=6,
        hierarchy_every=2,
        torch_threads=1,
        observation_contract="complete_recorded_sales_prelaunch_zeros_v1",
        aggregate_columns=("store_id", "dept_id"),
        event_pool_columns=("dept_id",),
        head="hurdle_negbin",
    )
    if variant == "v1_corrected":
        values.update(
            aggregate_history=False,
            mean_loss_weight=0.0,
            hierarchy_loss_weight=0.0,
            event_penalty_weight=0.0,
            factor_rank=2,
        )
    values.update(changes)
    return RollingConfig(**values)


@pytest.mark.parametrize("variant", ["v2", "v3", "v5"])
def test_campaign_profile_auto_selects_the_dataset_metric(variant):
    """M5 resolves this governed default to revenue WRMSSE; other datasets may fall back."""
    profile = json.loads(
        (
            ROOT / f"configs/experiments/multibranch_revisions/{REVISION_MODELS[variant]}_campaign_default.json"
        ).read_text()
    )
    assert profile["progress"] == "auto"
    assert profile["selection_metric"] == "auto"
    assert profile["hierarchy_weighting"] == "auto"


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_revision_event_pooling(revision_data, variant):
    """Declared pools control table size AND the kernel's index, not just recorded metadata."""
    panel, data, calendar = revision_data
    config = small_config(variant)
    features = RevisionFeatures(panel, data, config, str(calendar))
    model = features.make_model()
    assert features.event_pool_index is not None and model.calendar is not None
    pools = int(features.event_pool_index.max()) + 1
    assert pools < len(data.series_ids)
    assert model.calendar.series_deviation.weight.shape[0] == pools
    captured = []
    hook = model.calendar.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs["series_index"]), with_kwargs=True
    )
    inputs, _, _ = features.batch(
        data.cutoff.isoformat(), np.arange(len(data.series_ids)), "cpu", False
    )
    features.mean(model, inputs)
    hook.remove()
    torch.testing.assert_close(captured[0], torch.tensor(features.event_pool_index))


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_revision_commercial_heads(revision_data, variant):
    """Real price inputs exercise the separate commercial-head dimension contract."""
    panel, data, _ = revision_data
    for width, heads in ((15, 3), (18, 2)):
        config = small_config(variant, d_model=width, n_heads=3, commercial_attention_heads=heads)
        features = RevisionFeatures(panel, data, config)
        model = features.make_model()
        assert model.commercial.attn.num_heads == heads
        inputs, _, _ = features.batch(data.cutoff.isoformat(), np.arange(3), "cpu", False)
        assert torch.isfinite(features.mean(model, inputs)).all()
    with pytest.raises(ValueError, match="commercial attention"):
        RevisionFeatures(panel, data, small_config(variant, d_model=15, n_heads=3))


def test_read_real_m5(revision_data):
    panel, data, _ = revision_data
    assert len(data.series_ids) == 12
    assert data.target.shape == (12, 1913)
    assert data.price.shape == (12, 1941)
    assert not data.observed.all()
    assert not data.target.flags.writeable
    assert set(data.metadata[panel.store_col]) == {"CA_1", "CA_2", "TX_1"}


def test_unknown_quantities_cannot_enter_hierarchy_supervision(revision_data, tmp_path):
    panel, _, _ = revision_data
    table = pq.read_table(panel.path)
    for fault in ("changed_placeholder", "masked_positive", "internal_missing_zero"):
        rows = table.to_pylist()
        if fault == "changed_placeholder":
            row = next(
                row
                for row in rows
                if row["observation_state"] == "unobserved" and row["date"] < dt.date(2016, 4, 24)
            )
            row["qty"] = 1
        else:
            row = next(row for row in rows if row["date"] == dt.date(2016, 4, 1) and row["qty"] > 0)
            row["observation_state"] = "unobserved"
            if fault == "internal_missing_zero":
                row["qty"] = 0
        path = tmp_path / f"{fault}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        with pytest.raises(ValueError, match="complete recorded sales required"):
            PanelArrays.read(
                dataclasses.replace(panel, path=str(path)), "2016-04-24", 28, max_bytes=2_000_000
            )


def test_position_and_window_has_no_future_target(revision_data):
    _, data, _ = revision_data
    assert data.position("2016-04-24") == 1913
    assert "target" not in data.window("2016-04-24", np.array([0]), 112, targets=False)
    with pytest.raises(ValueError, match="crosses"):
        data.window("2016-04-24", np.array([0]), 112, targets=True)
    for day in ("2011-01-28", "2016-04-25"):
        with pytest.raises(ValueError, match="outside"):
            data.position(day)
    for rows in (np.array([]), np.array([-1]), np.array([12]), np.array([1.5])):
        with pytest.raises(ValueError, match="rows"):
            data.window("2016-04-24", rows, 28, targets=False)
    with pytest.raises(ValueError, match="encoder"):
        data.window("2011-01-29", np.array([0]), 112, targets=False)


def test_global_history_padding_is_automatic_and_keeps_targets_unchanged(revision_data):
    """Encoder length controls prefix padding; target observation remains a separate mask."""
    _, data, _ = revision_data
    rows = np.array([0, 1])
    origin = "2011-01-29"
    window = data.window(origin, rows, 112, targets=True, left_pad=True)
    assert window["history"].shape == (2, 112)
    assert window["available"].shape == (2, 112)
    assert not window["available"][:, :111].any()
    assert window["available"][:, -1].all()
    assert not window["observed"][:, :111].any()
    assert not window["history"][:, :111].any()
    np.testing.assert_array_equal(window["target"], data.target[rows, 1:29])
    np.testing.assert_array_equal(window["target_observed"], data.observed[rows, 1:29])

    config = small_config(
        encoder_days=1911,
        train_windows=52,
        train_stride=28,
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
    )
    assert len(rolling_origins(data, "2016-02-28", config)) == 52
    legacy = dataclasses.replace(config, observation_contract="masked_observed_counts_v1")
    with pytest.raises(ValueError, match="exceed available"):
        rolling_origins(data, "2016-02-28", legacy)


def test_panel_window_padding_contract_is_exact_without_external_data():
    target = np.arange(14, dtype=np.float32).reshape(2, 7)
    observed = np.ones_like(target, dtype=bool)
    target[0, 1] = 0
    observed[0, 1] = False
    data = PanelArrays(
        ("a", "b"),
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 7),
        2,
        target,
        observed,
        np.zeros_like(target),
        {"series": ("a", "b")},
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
    )
    rows = np.array([0, 1])
    padded = data.window("2024-01-02", rows, 4, targets=True, left_pad=True)
    np.testing.assert_array_equal(padded["history"][:, -2:], target[:, :2])
    np.testing.assert_array_equal(padded["observed"][:, -2:], observed[:, :2])
    np.testing.assert_array_equal(padded["target"], target[:, 2:4])
    assert padded["available"].dtype == np.bool_
    assert not padded["available"][:, :2].any() and padded["available"][:, 2:].all()
    assert padded["available"][0, -1] and not padded["observed"][0, -1]
    unpadded = data.window("2024-01-04", rows, 4, targets=False, left_pad=True)
    assert unpadded["available"].all()
    np.testing.assert_array_equal(unpadded["history"], target[:, :4])
    with pytest.raises(ValueError, match="insufficient encoder history"):
        data.window("2024-01-02", rows, 4, targets=False)
    with pytest.raises(ValueError, match="insufficient encoder history"):
        data.window("2024-01-02", rows, 4, targets=False, left_pad=1)

    config = dataclasses.replace(
        full_history_test_config("v2"),
        encoder_days=4,
        patch_size=2,
        train_windows=2,
        train_stride=2,
    )
    assert rolling_origins(data, "2024-01-05", config) == (
        "2024-01-03",
        "2024-01-01",
    )
    with pytest.raises(ValueError, match="exceed available"):
        rolling_origins(
            data,
            "2024-01-05",
            dataclasses.replace(config, observation_contract="masked_observed_counts_v1"),
        )
    with pytest.raises(ValueError, match="exceed available"):
        rolling_origins(data, "2024-01-05", dataclasses.replace(config, train_windows=3))


def test_informative_target_diagnostic_names_excluded_series_by_window():
    start = dt.date(2024, 1, 1)
    observed = np.array(
        [
            [False, False, True, False, False, False],
            [False, False, False, False, False, False],
        ]
    )
    data = PanelArrays(
        ("trainable", "inactive"),
        start,
        start + dt.timedelta(days=5),
        2,
        np.zeros((2, 6), np.float32),
        observed,
        np.empty((2, 0), np.float64),
        {"series": ("trainable", "inactive")},
    )

    diagnostic, trainable = informative_target_diagnostic(
        data, ("2024-01-02", "2024-01-04")
    )

    np.testing.assert_array_equal(trainable, [True, False])
    assert diagnostic["trainable_series_count"] == 1
    assert diagnostic["excluded_series"] == ["inactive"]
    assert [row["series_without_observed_target"] for row in diagnostic["windows"]] == [1, 2]


def test_m5_full_history_schedule_has_all_52_exact_origins():
    start = dt.date(2011, 1, 29)
    data = PanelArrays(
        ("series",),
        start,
        dt.date(2016, 4, 24),
        28,
        np.zeros((1, 1913), np.float32),
        np.ones((1, 1913), bool),
        np.zeros((1, 1913), np.float64),
        {"series": ("series",)},
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
    )
    config = dataclasses.replace(full_history_test_config("v2"), train_windows=52, train_stride=28)
    actual = rolling_origins(data, "2016-02-28", config)
    expected = tuple(
        (dt.date(2016, 1, 31) - dt.timedelta(days=28 * offset)).isoformat() for offset in range(52)
    )
    assert actual == expected


def test_sft_padding_covers_aggregate_and_history_covariates(synthetic_revision_data):
    panel, origin, horizon = synthetic_revision_data
    config = full_history_test_config(
        "v2",
        aggregate_history=True,
        aggregate_columns=("group",),
        mean_loss_weight=0.0,
    )
    spec = revision_data_spec(panel, horizon, config)
    spec.time_varying_unknown_reals = ("temperature",)
    spec.time_varying_unknown_categoricals = ("weather",)
    data = PanelArrays.read(
        panel,
        origin,
        horizon,
        max_bytes=config.max_array_bytes,
        data_spec=spec,
        metadata_columns=("group",),
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
    )
    assert data.target[0, 10] == 0 and not data.observed[0, 10]
    features = RevisionFeatures(panel, data, config, data_spec=spec)
    selection_end = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    earliest = min(rolling_origins(data, selection_end, config))
    inputs, _, _ = features.batch(earliest, np.arange(2), "cpu", True)
    available = inputs["history_available"]
    assert inputs["aggregate_history"].shape == inputs["history"].shape == (2, 1911)
    assert inputs["history_covariates"].shape[:2] == (2, 1911)
    assert not inputs["observed"][~available].any()
    assert not inputs["aggregate_observed"][~available].any()
    assert not inputs["history_covariate_mask"][~available].any()
    stop = data.position(earliest)
    padding = config.encoder_days - stop
    assert not available[:, :padding].any() and available[:, padding:].all()
    np.testing.assert_array_equal(inputs["history"].numpy()[:, padding:], data.target[:, :stop])
    np.testing.assert_array_equal(inputs["observed"].numpy()[:, padding:], data.observed[:, :stop])
    assert available[0, padding + 10] and not inputs["observed"][0, padding + 10]
    assert inputs["aggregate_observed"][:, padding:].any()
    assert inputs["history_covariate_mask"][:, padding:].any()

    model = features.make_model().eval()
    with torch.no_grad():
        assert model.occurrence is not None
        model.occurrence.weight.fill_(0.1)
        model.size.weight.fill_(0.1)

    def run(prefix: float):
        values = {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        for name in ("history", "aggregate_history", "history_covariates"):
            value = values[name]
            mask = ~available if value.ndim == 2 else (~available).unsqueeze(-1).expand_as(value)
            value[mask] = prefix
            value.requires_grad_(True)
        prediction = features.mean(model, values)
        prediction.sum().backward()
        result = prediction.detach().clone()
        gradients = {
            name: values[name].grad.detach().clone()
            for name in ("history", "aggregate_history", "history_covariates")
        }
        model.zero_grad(set_to_none=True)
        return result, gradients

    expected, expected_gradients = run(0.0)
    actual, actual_gradients = run(999.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    for name in expected_gradients:
        torch.testing.assert_close(
            actual_gradients[name], expected_gradients[name], rtol=0, atol=1e-6
        )
        mask = ~available
        if actual_gradients[name].ndim == 3:
            mask = mask.unsqueeze(-1).expand_as(actual_gradients[name])
        assert not actual_gradients[name][mask].any()


def test_sft_1911_day_earliest_origin_runs_through_the_model(revision_data):
    """Exercise the real adapter-to-attention handoff, not only the array slicer."""
    panel, data, _ = revision_data
    data = dataclasses.replace(data, observation_contract=GLOBAL_LEFT_PADDING_CONTRACT)
    config = small_config(
        encoder_days=1911,
        train_windows=52,
        train_stride=28,
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
        aggregate_history=False,
        mean_weighting="uniform",
    )
    features = RevisionFeatures(panel, data, config)
    selection_end = data.date_at(data.target.shape[1] - 2 * data.horizon - 1).isoformat()
    origin = min(rolling_origins(data, selection_end, config))
    inputs, target, target_observed = features.batch(origin, np.array([0, 1]), "cpu", True)
    assert inputs["history"].shape == (2, 1911)
    assert inputs["history_available"].shape == (2, 1911)
    assert not inputs["history_available"][:, :1_400].any()
    assert inputs["history_available"][:, -1].all()
    prediction = features.validation_values(features.make_model().eval(), inputs)[0]
    assert prediction.shape == target.shape == target_observed.shape == (2, 28)
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_synthetic_full_history_fit_refit_forecast_and_replay(
    synthetic_revision_data, tmp_path, monkeypatch, variant
):
    """Always-running two-epoch proof; real-data tests below remain supplemental evidence."""
    from autoforecast.rolling_training import RollingTrainer

    panel, origin, horizon = synthetic_revision_data
    config = full_history_test_config(variant)
    from provenance.lineage import directory_digest

    output = tmp_path / variant
    predict = RollingTrainer.predict
    begin_phase = RevisionFeatures.begin_phase
    predicted_origins = []
    executed_origins = {}

    def record_predict(self, model, day):
        predicted_origins.append(day)
        return predict(self, model, day)

    def record_phase(self, origins, phase):
        executed_origins.setdefault(phase, []).append(tuple(origins))
        return begin_phase(self, origins, phase)

    monkeypatch.setattr(RollingTrainer, "predict", record_predict)
    monkeypatch.setattr(RevisionFeatures, "begin_phase", record_phase)
    forecast = fit_predict_revision(panel, origin, horizon, config=config, out_dir=str(output))
    complete = json.loads((output / "COMPLETE.json").read_text())
    training = complete["training"]
    assert training["epochs_per_phase"] == 2
    assert training["selected_updates"] > 0
    assert set(training["phase_coverage"]) == {"selection", "refit"}
    data = PanelArrays.read(
        panel,
        origin,
        horizon,
        max_bytes=config.max_array_bytes,
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
    )
    validation_start = (dt.date.fromisoformat(origin) - dt.timedelta(days=2 * horizon)).isoformat()
    assert tuple(training["selection_training_origins"]) == rolling_origins(
        data, validation_start, config
    )
    assert tuple(training["refit_training_origins"]) == rolling_origins(data, origin, config)
    assert executed_origins["selection"][0] == rolling_origins(data, validation_start, config)
    assert executed_origins["refit"][0] == rolling_origins(data, origin, config)
    assert complete["binding"]["config"]["observation_contract"] == GLOBAL_LEFT_PADDING_CONTRACT
    assert complete["binding"]["sales_observation_contract"] == GLOBAL_LEFT_PADDING_CONTRACT
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert {row["phase"] for row in rows} == {"selection", "refit"}
    assert {row["epoch"] for row in rows if row["phase"] == "selection"} == {1, 2}
    assert len(training["phase_coverage"]["selection"]["epochs"]) == 2
    selected_epochs = (training["selected_updates"] + training["steps_per_epoch"] - 1) // training[
        "steps_per_epoch"
    ]
    assert {row["epoch"] for row in rows if row["phase"] == "refit"} == set(
        range(1, selected_epochs + 1)
    )
    assert len(training["phase_coverage"]["refit"]["epochs"]) == selected_epochs

    monkeypatch.setattr(
        RollingTrainer,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay retrained")),
    )
    replay_dir = tmp_path / f"{variant}-replay"
    replay = fit_predict_revision(
        panel,
        origin,
        horizon,
        config=config,
        out_dir=str(replay_dir),
        replay_checkpoint=str(output / "model.pt"),
    )
    replay_complete = json.loads((replay_dir / "COMPLETE.json").read_text())
    assert replay_complete["replay_source"] == str(output / "model.pt")
    assert replay_complete["execution_checks"]["replay_exactly_equal"] is True
    assert predicted_origins == [origin, origin]
    assert pq.read_table(forecast.path).equals(pq.read_table(replay.path))

    tampered = tmp_path / f"{variant}-tampered"
    import shutil

    shutil.copytree(output, tampered)
    saved = torch.load(tampered / "model.pt", weights_only=True)
    for parameter in saved["state_dict"].values():
        if parameter.is_floating_point():
            parameter.add_(0.01)
    torch.save(saved, tampered / "model.pt")
    tampered_complete = json.loads((tampered / "COMPLETE.json").read_text())
    tampered_complete["checkpoint_sha256"] = directory_digest(tampered / "model.pt")
    (tampered / "COMPLETE.json").write_text(json.dumps(tampered_complete))
    with pytest.raises(ValueError, match="replayed forecast"):
        fit_predict_revision(
            panel,
            origin,
            horizon,
            config=config,
            out_dir=str(tmp_path / f"{variant}-tampered-replay"),
            replay_checkpoint=str(tampered / "model.pt"),
        )


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_catalog_architecture_trains_two_epochs_with_1911_days(revision_data, tmp_path, variant):
    """Run the maintained catalog architecture through selection, refit and forecast."""
    panel, data, calendar = revision_data
    profile = json.loads(
        (
            ROOT / f"configs/experiments/multibranch_revisions/{REVISION_MODELS[variant]}_campaign_default.json"
        ).read_text()
    )
    profile.update(
        encoder_days=1911,
        d_model=80,
        n_heads=10,
        trunk_stages=10,
        epochs=2,
        batch_size=6,
        device="cpu",
        max_array_bytes=20_000_000,
    )
    if variant == "v2":
        profile.update(
            decoder_layers=10,
            history_attention_layers=10,
            commercial_attention_layers=10,
            commercial_attention_heads=10,
            event_kernel_layers=10,
        )
    else:
        profile["v3"] = {**profile["v3"], "local_blocks": 10}
    config = RollingConfig(**profile)
    feature_type = HMFNetRevisionFeatures if variant == "v3" else RevisionFeatures
    model = feature_type(panel, data, config, str(calendar)).make_model()
    if variant == "v2":
        assert model.trunk.n_stages == 10
        assert len(model.decoder.layers) == 10
        assert model.demand.attention_layers == 10
        assert model.demand.attn.num_heads == 10
        assert model.commercial is not None
        assert model.commercial.attention_layers == 10
        assert model.commercial.attn.num_heads == 10
        assert model.calendar is not None and model.calendar.kernel_layers_count == 10
    else:
        assert len(model.local_blocks) == 10
        assert model.cross_attention.num_heads == 10
    output = tmp_path / variant
    forecast = fit_predict_revision(
        panel,
        "2016-04-24",
        28,
        config=config,
        calendar_path=str(calendar),
        levels=M5_LEVELS,
        out_dir=str(output),
    )
    forecast.validate()
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["training"]["epochs_per_phase"] == 2
    assert complete["binding"]["config"]["encoder_days"] == 1911
    assert complete["binding"]["config"]["d_model"] == 80
    assert complete["binding"]["config"]["n_heads"] == 10
    assert complete["binding"]["config"]["trunk_stages"] == 10
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert set(complete["training"]["phase_coverage"]) == {"selection", "refit"}
    assert {row["epoch"] for row in rows if row["phase"] == "selection"} == {1, 2}
    assert [
        item["epoch"] for item in complete["training"]["phase_coverage"]["selection"]["epochs"]
    ] == [1, 2]
    selected_epochs = (
        complete["training"]["selected_updates"] + complete["training"]["steps_per_epoch"] - 1
    ) // complete["training"]["steps_per_epoch"]
    assert {row["epoch"] for row in rows if row["phase"] == "refit"} == set(
        range(1, selected_epochs + 1)
    )
    assert [
        item["epoch"] for item in complete["training"]["phase_coverage"]["refit"]["epochs"]
    ] == list(range(1, selected_epochs + 1))
    assert complete["execution_checks"]["finite"]
    replay = fit_predict_revision(
        panel,
        "2016-04-24",
        28,
        config=config,
        calendar_path=str(calendar),
        levels=M5_LEVELS,
        out_dir=str(tmp_path / f"{variant}-replay"),
        replay_checkpoint=str(output / "model.pt"),
    )
    replay_complete = json.loads((tmp_path / f"{variant}-replay" / "COMPLETE.json").read_text())
    assert replay_complete["execution_checks"]["replay_exactly_equal"] is True
    assert pq.read_table(forecast.path).equals(pq.read_table(replay.path))


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "missing", "negative", "fraction", "state", "identity", "price", "null_target"],
)
def test_read_rejects_bad_panel_cells(revision_data, tmp_path, fault):
    panel, _, _ = revision_data
    table = pq.read_table(panel.path)
    rows = table.to_pylist()
    if fault == "duplicate":
        rows.append(next(row for row in rows if row["date"] == dt.date(2016, 4, 1)))
    elif fault == "missing":
        rows.pop(40)
    else:
        field, value = {
            "negative": ("qty", -1),
            "fraction": ("qty", 0.5),
            "state": ("observation_state", "lower_bound"),
            "identity": ("dept_id", "changed"),
            "price": ("sell_price", -1),
            "null_target": ("qty", None),
        }[fault]
        historical = next(row for row in rows if row["date"] == dt.date(2016, 4, 1))
        historical[field] = value
    bad = tmp_path / "bad.parquet"
    schema = table.schema
    if fault == "fraction":
        schema = schema.set(schema.get_field_index("qty"), pa.field("qty", pa.float64()))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), bad)
    message = {
        "duplicate": "duplicate series/date",
        "missing": "incomplete panel grid",
        "negative": "integer counts",
        "fraction": "integer counts",
        "state": "censored/unknown",
        "identity": "identity changes",
        "price": "price must",
        "null_target": "integer counts",
    }[fault]
    with pytest.raises(ValueError, match=message):
        PanelArrays.read(
            dataclasses.replace(panel, path=str(bad)), "2016-04-24", 28, max_bytes=2_000_000
        )


def test_read_refuses_memory_and_unsupported_contract(revision_data):
    panel, _, _ = revision_data
    with pytest.raises(ValueError, match="budget"):
        PanelArrays.read(panel, "2016-04-24", 28, max_bytes=1)
    with pytest.raises(ValueError, match="known_future"):
        PanelArrays.read(
            dataclasses.replace(panel, known_future=()), "2016-04-24", 28, max_bytes=2_000_000
        )
    with pytest.raises(ValueError, match="positive integers"):
        PanelArrays.read(panel, "2016-04-24", True, max_bytes=2_000_000)


def test_read_refuses_noncanonical_origin_before_panel_io():
    with pytest.raises(ValueError, match="canonical"):
        PanelArrays.read(
            Panel("/not-read", "id", "date", "qty"),
            "20160424",
            28,
            max_bytes=2_000_000,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"frequency": "W"}, "misaligned"),
    ],
)
def test_read_refuses_role_contracts(revision_data, change, message):
    panel, _, _ = revision_data
    with pytest.raises(ValueError, match=message):
        PanelArrays.read(
            dataclasses.replace(panel, **change), "2016-04-24", 28, max_bytes=2_000_000
        )


@pytest.mark.parametrize(
    "change", [{"exposure_col": None}, {"product_col": None}, {"price_col": None}, {"static": ()}]
)
def test_read_allows_missing_optional_roles(revision_data, change):
    panel, original, _ = revision_data
    data = PanelArrays.read(
        dataclasses.replace(panel, **change), "2016-04-24", 28, max_bytes=2_000_000
    )
    np.testing.assert_array_equal(data.target, original.target)


@pytest.mark.parametrize(
    "fault",
    [
        "column",
        "first_duplicate",
        "first_null",
        "first_blank",
        "metadata_null",
        "metadata_blank",
        "unknown_series",
        "no_history",
    ],
)
def test_read_identity_grid_refusals(revision_data, tmp_path, fault):
    panel, _, _ = revision_data
    table = pq.read_table(panel.path)
    rows = table.to_pylist()
    origin = "2016-04-24"
    if fault == "column":
        table = table.drop(["dept_id"])
    elif fault == "first_duplicate":
        rows.append(rows[0])
    elif fault == "first_null":
        rows[0]["id"] = None
    elif fault == "first_blank":
        first_id = rows[0]["id"]
        for row in rows:
            if row["id"] == first_id:
                row["id"] = " "
    elif fault == "metadata_null":
        rows[0]["dept_id"] = None
    elif fault == "metadata_blank":
        first_id = rows[0]["id"]
        for row in rows:
            if row["id"] == first_id:
                row["dept_id"] = " "
    elif fault == "unknown_series":
        rows[-1]["id"] = "unknown"
    else:
        origin = "2010-01-01"
    if fault != "column":
        table = pa.Table.from_pylist(rows, schema=table.schema)
    path = tmp_path / "bad-grid.parquet"
    pq.write_table(table, path)
    message = {
        "column": "absent from the Parquet schema",
        "first_duplicate": "first date",
        "first_null": "first date",
        "first_blank": "first date",
        "metadata_null": "nonnull strings",
        "metadata_blank": "nonnull strings",
        "unknown_series": "unknown series",
        "no_history": "no date-grain history",
    }[fault]
    with pytest.raises(ValueError, match=message):
        PanelArrays.read(
            dataclasses.replace(panel, path=str(path)), origin, 28, max_bytes=2_000_000
        )


def test_revision_calendar(revision_data):
    _, _, calendar = revision_data
    table = pq.read_table(calendar)
    assert set(table["group"].to_pylist()) == {"CA", "TX", "WI"}
    assert "Easter" in table["event_name_1"].to_pylist()
    assert "Religious" in table["event_type_1"].to_pylist()
    with pytest.raises(FileExistsError):
        build_revision_calendar(calendar)


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
def test_revision_calendar_rejects_duplicate_json_metadata(revision_data, tmp_path, nested):
    """Calendar role/availability metadata has one unambiguous interpretation."""
    panel, data, calendar = revision_data
    table = pq.read_table(calendar)
    metadata = dict(table.schema.metadata or {})
    raw = metadata[b"revision_calendar_v1"]
    if nested:
        raw = raw.replace(b'"variants": {', b'"variants": {"v2": {}, "v2": {}, ', 1)
    else:
        raw = b'{"preannounced":false,' + raw[1:]
    metadata[b"revision_calendar_v1"] = raw
    path = tmp_path / f"duplicate-{'nested' if nested else 'top'}.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        RevisionFeatures(panel, data, small_config(), str(path))


@pytest.mark.parametrize("raw", [b"[]", b"null", b"1", b'"calendar"'])
def test_revision_calendar_contract_requires_a_json_object(raw):
    from autoforecast.adapters.mft import _revision_calendar_contract

    table = pa.table({})
    assert _revision_calendar_contract(table) is None
    assert (
        _revision_calendar_contract(table.replace_schema_metadata({b"revision_calendar_v1": b"{}"}))
        == {}
    )
    with pytest.raises(ValueError, match="revision calendar contract must be a JSON object"):
        _revision_calendar_contract(table.replace_schema_metadata({b"revision_calendar_v1": raw}))


def test_calendar_requires_complete_support_grid_and_consistent_events(revision_data, tmp_path):
    panel, data, calendar = revision_data
    table = pq.read_table(calendar)
    for fault in ("interior", "beyond_horizon", "null_date", "empty_event"):
        rows = table.to_pylist()
        if fault in {"interior", "beyond_horizon"}:
            day = dt.date(2016, 3, 27) if fault == "interior" else dt.date(2016, 5, 30)
            assert any(r["date"] == day and r["event_name_1"] for r in rows)
            rows = [r for r in rows if r["date"] != day]
            message = "incomplete calendar"
        elif fault == "null_date":
            rows[0]["date"] = None
            message = "must be typed"
        else:
            rows[0]["event_name_1"] = rows[0]["event_type_1"] = ""
            message = "must be paired"
        path = tmp_path / f"{fault}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        with pytest.raises(ValueError, match=message):
            RevisionFeatures(panel, data, small_config(), str(path))
    # The release ends June 19: 30 days beyond May 22 would require unavailable June 20/21.
    with pytest.raises(ValueError, match="event-support margins"):
        RevisionFeatures(
            panel, data, small_config("v1_corrected", event_max_support=30), str(calendar)
        )
    RevisionFeatures(panel, data, small_config("v1_corrected", event_max_support=28), str(calendar))


def test_event_admission_disables_unseen_and_always_present_channels(revision_data):
    panel, data, calendar = revision_data
    features = RevisionFeatures(panel, data, small_config(), str(calendar))
    name = next(name for name in features.event_names if name.startswith("name:"))
    index = features.event_names.index(name)
    features.event_records[name] = {
        (dt.date(2016, 5, 10), scope, dt.date.min, "2016-05-10")
        for scope in features.calendar_scopes
    }
    result = features.begin_phase(("2016-01-31",), "selection")
    assert not result["events"][name]["enabled"]
    assert result["events"][name]["informative_event_cells"] == 0
    inputs, _, _ = features.batch("2016-04-24", np.array([0]), "cpu", False)
    assert not inputs["event_present"][..., index].any()
    # Two episodes cover every training date: no non-event contrast, so still not estimable.
    features.event_records[name] = {
        (day, scope, dt.date.min, day.isoformat())
        for scope in features.calendar_scopes
        for day in (dt.date(2016, 2, 1), dt.date(2016, 2, 28))
    }
    result = features.begin_phase(("2016-01-31",), "selection")
    assert len(result["events"][name]["distinct_occurrences"]) == 2
    assert result["events"][name]["informative_non_event_cells"] == 0
    assert not result["events"][name]["enabled"]
    with pytest.raises(ValueError, match="origin-bounded"):
        features.begin_phase(("2016-04-24",), "selection")


def test_rolling_origins_are_chronological(revision_data):
    _, data, _ = revision_data
    origins = rolling_origins(data, "2016-02-28", RollingConfig(variant="v2", encoder_days=112))
    assert len(origins) == 52
    assert data.position(origins[0]) + 28 == data.position("2016-02-28")
    with pytest.raises(ValueError, match="exceed"):
        rolling_origins(data, "2012-01-01", small_config(train_windows=52))


@pytest.mark.parametrize(
    "change",
    [
        {"max_updates": True},
        {"learning_rate": float("nan")},
        {"encoder_days": 31},
        {"device": "mps"},
        {"factor_rank": 8},
        {"excess_floor": 0},
        {"variant": "legacy"},
        {"identity_columns": (" ",)},
        {"calendar_group_column": " "},
        {"cycles": ((" ", 7.0, 1),)},
        {"cycles": (("weekly", 7.0, 1), ("weekly", 14.0, 1))},
    ],
)
def test_config_rejects_invalid_controls(change):
    with pytest.raises(ValueError):
        small_config(**change)


@pytest.mark.parametrize(
    "field",
    [
        "statistic_windows",
        "identity_columns",
        "factor_columns",
        "aggregate_columns",
        "baseline_columns",
        "event_pool_columns",
    ],
)
def test_config_requires_sequence_fields(field):
    for invalid in (1, True, "id"):
        with pytest.raises(ValueError, match=f"{field} must be a sequence"):
            small_config(**{field: invalid})
    assert getattr(small_config(**{field: []}), field) == ()


@pytest.mark.parametrize("invalid", [1, True, "weekly", [1]])
def test_config_requires_nested_seasonal_sequences(invalid):
    with pytest.raises(ValueError, match="cycles must be a sequence of seasonal definitions"):
        small_config(cycles=invalid)
    assert small_config(cycles=[["weekly", 7, 1]]).cycles == (("weekly", 7, 1),)


def test_config_refuses_seed_types_and_mft_unused_mechanisms():
    with pytest.raises(ValueError, match="seed and factor_rank"):
        small_config(seed=True)
    with pytest.raises(ValueError, match="likelihood-only"):
        small_config("v1_corrected", mean_loss_weight=0.1)


@pytest.mark.parametrize(
    "fault",
    ["availability", "schema", "duplicate", "snap", "event_pair", "event_blank", "scope_blank"],
)
def test_calendar_refusals(revision_data, tmp_path, fault):
    panel, data, calendar = revision_data
    table = pq.read_table(calendar)
    if fault == "availability":
        table = table.replace_schema_metadata({})
    elif fault == "schema":
        table = table.drop(["snap"])
    else:
        rows = table.to_pylist()
        if fault == "duplicate":
            rows.append(rows[0])
        elif fault == "snap":
            rows[0]["snap"] = 0.5
        elif fault == "event_pair":
            rows[0]["event_name_1"] = "Unpaired"
            rows[0]["event_type_1"] = None
        elif fault == "event_blank":
            rows[0]["event_name_1"] = " "
        else:
            rows[0]["group"] = " "
        table = pa.Table.from_pylist(rows, schema=table.schema)
    path = tmp_path / "bad-calendar.parquet"
    pq.write_table(table, path)
    with pytest.raises(ValueError):
        RevisionFeatures(panel, data, small_config(), str(path))


@pytest.mark.parametrize("fault", ["cuda", "groups", "block", "informative", "loss", "prediction"])
def test_trainer_refusals(revision_data, monkeypatch, fault):
    from autoforecast.rolling_training import RollingTrainer

    panel, data, calendar = revision_data
    c = small_config()
    if fault == "cuda":
        c = dataclasses.replace(c, device="cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    features = RevisionFeatures(panel, data, c, str(calendar))
    if fault == "informative":
        data = dataclasses.replace(data, observed=np.zeros_like(data.observed))
    groups = features.groups
    if fault == "groups":
        groups = groups.astype(float)
    elif fault == "block":
        groups = np.zeros(len(data.series_ids), dtype=int)
    trainer = RollingTrainer(
        data,
        c,
        M5_LEVELS,
        groups,
        features.make_model,
        features.batch,
        features.training_loss,
        features.mean,
    )
    message = {
        "cuda": "CUDA explicitly",
        "groups": "factor groups",
        "block": "factor block",
        "informative": "no informative",
        "loss": "nonfinite training loss",
        "prediction": "nonfinite",
    }[fault]
    if fault == "loss":
        trainer.training_loss = lambda *args: torch.tensor(float("nan"), requires_grad=True)
    if fault == "prediction":
        trainer.mean = lambda model, inputs: torch.full((len(inputs["history"]), 28), float("nan"))
        with pytest.raises(ValueError, match=message):
            trainer.predict(features.make_model(), data.cutoff.isoformat())
    else:
        with pytest.raises(ValueError, match=message):
            trainer.fit(lambda record: None)


def test_sft_real_forward_backward_and_seasonal_anchor(revision_data):
    from forecast.models.mft.sft import SFTModel

    panel, data, calendar = revision_data
    torch.set_num_threads(1)
    f = RevisionFeatures(panel, data, small_config(), str(calendar))
    inputs, y, mask = f.batch("2016-02-28", np.arange(6), "cpu", True)
    model = f.make_model()
    assert isinstance(model, SFTModel) and model.aggregate is not None
    assert y is not None and mask is not None
    model.eval()
    mean = f.mean(model, inputs)
    anchor = inputs["baseline_logit"].sigmoid() * inputs["baseline_log_size"].exp()
    torch.testing.assert_close(mean, anchor.to(mean), rtol=2e-6, atol=1e-6)
    assert not torch.equal(mean[:, 0], mean[:, 1])
    loss = f.training_loss(model, inputs, y, mask, float(mask.sum()))
    loss.backward()
    assert model.occurrence.weight.grad is not None
    assert model.occurrence.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.occurrence.weight.add_(-0.001 * model.occurrence.weight.grad)
    model.zero_grad(set_to_none=True)
    f.training_loss(model, inputs, y, mask, float(mask.sum())).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.demand.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.aggregate.parameters())
    assert "snap" in f.known_names
    assert not any("event" in name or "_distance" in name for name in f.known_names)
    assert any(name.startswith("type:") for name in f.event_names)


def test_corrected_mft_k1_anchor(revision_data):
    from forecast.models.mft.heads import hurdle_mean

    panel, data, calendar = revision_data
    c = small_config("v1_corrected", factor_rank=0)
    f = RevisionFeatures(panel, data, c, str(calendar))
    inputs, _, _ = f.batch("2016-02-28", np.arange(6), "cpu", True)
    model = f.make_model().eval()
    with torch.no_grad():
        p, mu, k = model(**inputs)
    torch.testing.assert_close(k, torch.ones_like(k))
    expected = inputs["baseline_logit"].sigmoid() * (1 + inputs["baseline_log_size"].exp())
    torch.testing.assert_close(hurdle_mean(p[0], mu[0], k[0]), expected[:, None].expand(-1, 28))
    assert model.factor is None


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_revision_fit_replay(revision_data, tmp_path, variant):
    panel, data, calendar = revision_data
    config = small_config(variant)
    arguments: dict[str, Any] = dict(
        panel=panel,
        origin="2016-04-24",
        horizon=28,
        calendar_path=str(calendar),
        config=config,
        levels=M5_LEVELS,
    )
    forecast = fit_predict_revision(**arguments, out_dir=str(tmp_path / "fit"))
    complete = json.loads((tmp_path / "fit/COMPLETE.json").read_text())
    assert complete["training"]["refit_series_count"] == 12
    assert 1 <= complete["training"]["selected_updates"] <= 6
    assert complete["training"]["refit_series_seen_in_likelihood"] == 12
    assert complete["training"]["validation_origins"] == ["2016-02-28", "2016-03-27"]
    for phase in ("selection", "refit"):
        admission = complete["training"]["phase_admission"][phase]
        assert admission["phase"] == phase
        assert admission["policy"] == "two_occurrences_plus_informative_contrast_v1"
    replay = fit_predict_revision(
        **arguments,
        out_dir=str(tmp_path / "replay"),
        replay_checkpoint=str(tmp_path / "fit/model.pt"),
    )
    assert pq.read_table(forecast.path).equals(pq.read_table(replay.path))
    assert pq.read_table(forecast.path).num_rows == 12 * 28
    with pytest.raises(ValueError, match="already exists"):
        fit_predict_revision(**arguments, out_dir=str(tmp_path / "fit"))
    bad = tmp_path / "legacy.pt"
    torch.save({"run": {}, "state_dict": {}}, bad)
    with pytest.raises(ValueError, match="completed revision"):
        fit_predict_revision(**arguments, out_dir=str(tmp_path / "bad"), replay_checkpoint=str(bad))
    assert not (tmp_path / "bad/COMPLETE.json").exists()


def test_role_alias_cannot_read_future_targets(revision_data, monkeypatch):
    panel, _, _ = revision_data
    bad = dataclasses.replace(panel, price_col=panel.target_col, known_future=(panel.target_col,))

    def forbidden_scan(*args, **kwargs):
        raise AssertionError("data scan happened before the role-collision refusal")

    monkeypatch.setattr(ds, "dataset", forbidden_scan)
    with pytest.raises(ValueError, match="target leaking"):
        PanelArrays.read(bad, "2016-04-24", 28, max_bytes=2_000_000)


def test_null_date_is_not_silently_filtered_out(revision_data, tmp_path):
    panel, _, _ = revision_data
    table = pq.read_table(panel.path)
    rows = table.to_pylist()
    rows.append(dict(rows[0], date=None))
    path = tmp_path / "null-date.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="null panel date"):
        PanelArrays.read(
            dataclasses.replace(panel, path=str(path)), "2016-04-24", 28, max_bytes=2_000_000
        )


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_residual_only_optimizer_and_covariate_dropout(revision_data, variant):
    from forecast.models.mft.heads import HurdleNegBinHead
    from forecast.models.mft.sft import SFTModel

    panel, data, calendar = revision_data
    c = small_config(variant, covariate_dropout=0.8)
    features = RevisionFeatures(panel, data, c, str(calendar))
    model = features.make_model()
    groups = features.optimizer_groups(model)
    regularized = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    assert regularized
    if isinstance(model, SFTModel):
        occurrence, residual = model.occurrence, model.dispersion_residual
    else:
        assert isinstance(model.head, HurdleNegBinHead)
        occurrence, residual = model.head.occurrence, model.head.dispersion_residual
        assert residual is not None
    assert id(occurrence.weight) not in regularized
    assert id(residual.weight) in regularized
    inputs, _, _ = features.batch("2016-03-27", np.arange(12), "cpu", False)
    original = inputs["known_future"].clone()
    torch.manual_seed(0)
    changed = features.regularized_inputs(inputs)
    hidden = inputs["commercial_mask"] & ~changed["commercial_mask"]
    assert hidden.any()
    for j, name in enumerate(("price", "discount_depth", "promo")):
        if hidden[:, j].any():
            assert not changed["known_future"][
                hidden[:, j], :, features.known_names.index(name)
            ].any()
            assert not changed["known_future"][
                hidden[:, j], :, features.known_names.index(f"{name}_available")
            ].any()
    torch.testing.assert_close(inputs["known_future"], original)


def test_refit_excludes_series_without_informative_targets_from_likelihood(revision_data):
    from autoforecast.rolling_training import RollingTrainer

    panel, data, calendar = revision_data
    observed = data.observed.copy()
    # Only the latest selection target episode informs this real series; the quantities remain real.
    observed[0, : data.position("2016-01-31")] = False
    changed = dataclasses.replace(data, observed=observed)
    c = small_config("v1_corrected")
    features = RevisionFeatures(panel, changed, c, str(calendar))
    trainer = RollingTrainer(
        changed,
        c,
        M5_LEVELS,
        features.groups,
        features.make_model,
        features.batch,
        features.training_loss,
        features.mean,
        features.optimizer_groups,
    )
    _, result = trainer.fit(lambda record: None)
    assert result["refit_series_seen_in_likelihood"] == 12
    observed[0] = False
    changed = dataclasses.replace(data, observed=observed)
    features = RevisionFeatures(panel, changed, c, str(calendar))
    trainer.data = changed
    trainer.batch = features.batch
    _, result = trainer.fit(lambda record: None)
    assert result["excluded_series_count"] == 1
    assert result["refit_series_seen_in_likelihood"] == 11


def test_origin_and_batch_information_guards(revision_data):
    from autoforecast.rolling_training import RollingTrainer

    panel, data, calendar = revision_data
    c = small_config()
    observed = data.observed.copy()
    stop = data.position(rolling_origins(data, "2016-02-28", c)[0])
    observed[:, stop : stop + data.horizon] = False
    changed = dataclasses.replace(data, observed=observed)
    features = RevisionFeatures(panel, changed, c, str(calendar))
    trainer = RollingTrainer(
        changed,
        c,
        M5_LEVELS,
        features.groups,
        features.make_model,
        features.batch,
        features.training_loss,
        features.mean,
    )
    with pytest.raises(ValueError, match="origin contains no informative"):
        trainer.fit(lambda record: None)
    features = RevisionFeatures(panel, data, c, str(calendar))

    def bad_mask(*args):
        inputs, y, mask = features.batch(*args)
        return inputs, y, torch.zeros_like(mask) if mask is not None else None

    trainer.data, trainer.batch = data, bad_mask
    with pytest.raises(ValueError, match="schedule produced an empty batch"):
        trainer.fit(lambda record: None)


def test_feature_origin_and_layout_guards(revision_data, tmp_path):
    panel, data, calendar = revision_data
    rows = pq.read_table(calendar).to_pylist()
    for row in rows:
        for slot in (1, 2):
            if row[f"event_name_{slot}"] == "Christmas":
                row[f"event_name_{slot}"] = row[f"event_type_{slot}"] = None
    missing_event = tmp_path / "missing-event.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(calendar)), missing_event)
    with pytest.raises(ValueError, match="explicitly selected events"):
        RevisionFeatures(panel, data, small_config("v1_corrected"), str(missing_event))
    day = "2016-03-27"
    for price, message in (
        (np.full_like(data.price, np.nan), "lack weight prices"),
        (np.zeros_like(data.price), "undefined/nonfinite mean-loss weights"),
    ):
        features = RevisionFeatures(
            panel, dataclasses.replace(data, price=price), small_config(), str(calendar)
        )
        with pytest.raises(ValueError, match=message):
            features._origin(day)
    features = RevisionFeatures(panel, data, small_config(), str(calendar))
    features.known_names = ()
    with pytest.raises(ValueError, match="layout changed"):
        features.batch(day, np.array([0]), "cpu", False)


def test_revision_api_refuses_unsupported_contract_before_fit(revision_data, tmp_path):
    panel, _, calendar = revision_data
    arguments: dict[str, Any] = dict(
        panel=panel,
        origin="2016-04-24",
        horizon=28,
        calendar_path=str(calendar),
        config=small_config(),
        levels=M5_LEVELS,
        out_dir=str(tmp_path / "out"),
    )
    for changes, message in (
        ({"config": {}}, "configuration must be validated"),
        ({"panel": dataclasses.replace(panel, known_future=())}, "declared known_future"),
        ({"out_dir": str(pathlib.Path(panel.path) / "must-not-create")}, "overlaps"),
    ):
        with pytest.raises(ValueError, match=message):
            fit_predict_revision(**dict(arguments, **changes))
    assert not (tmp_path / "out").exists()


def test_revision_api_refuses_noncanonical_origin_before_runtime(
    revision_data, tmp_path, monkeypatch
):
    """A parseable but non-canonical origin must not start an expensive training runtime."""
    from autoforecast import training_checkpoint

    panel, _, calendar = revision_data
    entered_runtime = False

    def unexpected_runtime(*_args, **_kwargs):
        nonlocal entered_runtime
        entered_runtime = True
        raise AssertionError("non-canonical origin reached runtime setup")

    monkeypatch.setattr(training_checkpoint, "configure_training_runtime", unexpected_runtime)
    with pytest.raises(ValueError, match="canonical ISO"):
        fit_predict_revision(
            panel,
            "20160424",
            28,
            calendar_path=str(calendar),
            config=small_config(),
            levels=M5_LEVELS,
            out_dir=str(tmp_path / "noncanonical"),
        )
    assert not entered_runtime
    assert not (tmp_path / "noncanonical").exists()


def test_revision_api_refuses_invalid_horizon_before_runtime(revision_data, tmp_path, monkeypatch):
    from autoforecast import training_checkpoint

    panel, _, calendar = revision_data
    entered_runtime = False

    def unexpected_runtime(*_args, **_kwargs):
        nonlocal entered_runtime
        entered_runtime = True
        raise AssertionError("invalid horizon reached runtime setup")

    monkeypatch.setattr(training_checkpoint, "configure_training_runtime", unexpected_runtime)
    for horizon in (True, 0, -1):
        with pytest.raises(ValueError, match="horizon"):
            fit_predict_revision(
                panel,
                "2016-04-24",
                horizon,
                calendar_path=str(calendar),
                config=small_config(),
                levels=M5_LEVELS,
                out_dir=str(tmp_path / f"invalid-{horizon}"),
            )
    assert not entered_runtime


def test_revision_refuses_wrong_grid_and_source_drift(revision_data, tmp_path, monkeypatch):
    from autoforecast.rolling_training import RollingTrainer
    from provenance import lineage

    panel, _, calendar = revision_data
    arguments: dict[str, Any] = dict(
        panel=panel,
        origin="2016-04-24",
        horizon=28,
        calendar_path=str(calendar),
        config=small_config(),
        levels=M5_LEVELS,
    )
    predict = RollingTrainer.predict
    with monkeypatch.context() as patch:
        patch.setattr(
            RollingTrainer,
            "predict",
            lambda self, model, day: predict(self, model, day)[:1]
            if day == "2016-04-24"
            else predict(self, model, day),
        )
        with pytest.raises(ValueError, match="complete forecast grid"):
            fit_predict_revision(**arguments, out_dir=str(tmp_path / "wrong-grid"))
    digest = lineage.directory_digest
    state = {"after_prediction": False}
    monkeypatch.setattr(
        lineage,
        "directory_digest",
        lambda path, **kw: "changed"
        if state["after_prediction"] and pathlib.Path(path) == ROOT / "forecast"
        else digest(path, **kw),
    )

    def drift_after_prediction(self, model, day):
        value = predict(self, model, day)
        if day == "2016-04-24":
            state["after_prediction"] = True
        return value

    monkeypatch.setattr(RollingTrainer, "predict", drift_after_prediction)
    with pytest.raises(ValueError, match="input/source changed"):
        fit_predict_revision(**arguments, out_dir=str(tmp_path / "source-drift"))
    assert not (tmp_path / "source-drift/COMPLETE.json").exists()


def test_revision_checkpoint_refuses_source_drift_before_saving_update(
    revision_data, tmp_path, monkeypatch
):
    from autoforecast.training_checkpoint import TrainingRun
    from provenance import lineage

    panel, _, calendar = revision_data
    changed = False
    digest, register, save = (
        lineage.directory_digest,
        torch.optim.Optimizer.register_step_post_hook,
        TrainingRun.save,
    )
    saved_updates = []

    def digest_after_step(path, **kwargs):
        return (
            "changed"
            if changed and pathlib.Path(path) == ROOT / "forecast"
            else digest(path, **kwargs)
        )

    def register_drift(optimizer, hook):
        handle = register(optimizer, hook)

        def after_step(_optimizer, _args, _kwargs):
            nonlocal changed
            changed = True

        register(optimizer, after_step)
        return handle

    def record_save(run, state, *, reason):
        result = save(run, state, reason=reason)
        saved_updates.append(state["update"])
        return result

    monkeypatch.setattr(lineage, "directory_digest", digest_after_step)
    monkeypatch.setattr(torch.optim.Optimizer, "register_step_post_hook", register_drift)
    monkeypatch.setattr(TrainingRun, "save", record_save)
    destination = tmp_path / "source-drift-at-checkpoint"
    with pytest.raises(ValueError, match="refusing checkpoint/publication"):
        fit_predict_revision(
            panel,
            "2016-04-24",
            28,
            calendar_path=str(calendar),
            config=small_config(checkpoint_every_updates=1, progress="off"),
            levels=M5_LEVELS,
            out_dir=str(destination),
        )
    assert changed and saved_updates and all(update == 0 for update in saved_updates)
    assert not (destination / "COMPLETE.json").exists()


def test_replay_requires_sealed_weights_and_dependency_binding(
    revision_data, tmp_path, monkeypatch
):
    from provenance import lineage

    panel, _, calendar = revision_data
    config = small_config()
    arguments: dict[str, Any] = dict(
        panel=panel,
        origin="2016-04-24",
        horizon=28,
        calendar_path=str(calendar),
        config=config,
        levels=M5_LEVELS,
    )
    fit_predict_revision(**arguments, out_dir=str(tmp_path / "fit"))
    checkpoint = tmp_path / "fit/model.pt"
    complete = json.loads((tmp_path / "fit/COMPLETE.json").read_text())
    assert str(ROOT / "forecast") in complete["binding"]["inputs"]
    assert str(ROOT / "uv.lock") in complete["binding"]["inputs"]
    original_digest = lineage.directory_digest

    def changed_dependency(path, **kwargs):
        return (
            "f" * 64 if pathlib.Path(path) == ROOT / "forecast" else original_digest(path, **kwargs)
        )

    with monkeypatch.context() as patch:
        patch.setattr(lineage, "directory_digest", changed_dependency)
        with pytest.raises(ValueError, match="binding mismatch"):
            fit_predict_revision(
                **arguments,
                out_dir=str(tmp_path / "changed-code"),
                replay_checkpoint=str(checkpoint),
            )
    saved = torch.load(checkpoint, weights_only=True)
    saved["state_dict"]["occurrence.bias"] += 1.0
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="digest or binding mismatch"):
        fit_predict_revision(
            **arguments,
            out_dir=str(tmp_path / "changed-weights"),
            replay_checkpoint=str(checkpoint),
        )
    assert not (tmp_path / "changed-weights/COMPLETE.json").exists()


def test_replay_completion_schema_and_value_guards(revision_data, tmp_path):
    import shutil

    from provenance.lineage import directory_digest

    panel, _, calendar = revision_data
    arguments: dict[str, Any] = dict(
        panel=panel,
        origin="2016-04-24",
        horizon=28,
        calendar_path=str(calendar),
        config=small_config(),
        levels=M5_LEVELS,
    )
    fit_predict_revision(**arguments, out_dir=str(tmp_path / "fit"))
    checkpoint = tmp_path / "fit/model.pt"
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    shutil.copyfile(checkpoint, standalone / "model.pt")
    with pytest.raises(ValueError, match="completed revision directory"):
        fit_predict_revision(
            **arguments,
            out_dir=str(tmp_path / "incomplete-replay"),
            replay_checkpoint=str(standalone / "model.pt"),
        )
    with pytest.raises(ValueError, match="overlaps the completed source"):
        fit_predict_revision(
            **arguments, out_dir=str(tmp_path / "fit/nested"), replay_checkpoint=str(checkpoint)
        )
    assert not (tmp_path / "fit/nested").exists()
    for fault, message in (
        ("binding", "checkpoint schema"),
        ("weights", "replayed forecast"),
        ("admission", "checkpoint event admission"),
    ):
        source = tmp_path / fault
        shutil.copytree(tmp_path / "fit", source)
        saved = torch.load(source / "model.pt", weights_only=True)
        if fault == "binding":
            saved["binding"]["horizon"] = False
        elif fault == "admission":
            saved["training"]["phase_admission"]["refit"]["policy"] = "unbound-policy"
        else:
            saved["state_dict"]["occurrence.bias"] += 1.0
        torch.save(saved, source / "model.pt")
        complete = json.loads((source / "COMPLETE.json").read_text())
        complete["checkpoint_sha256"] = directory_digest(source / "model.pt")
        (source / "COMPLETE.json").write_text(json.dumps(complete))
        with pytest.raises(ValueError, match=message):
            fit_predict_revision(
                **arguments,
                out_dir=str(tmp_path / f"replay-{fault}"),
                replay_checkpoint=str(source / "model.pt"),
            )


def test_cli_rejects_later_origin_before_calendar_read(tmp_path, monkeypatch):
    from scripts import run_multibranch_revision as cli
    from scripts.run_multibranch_revision import main

    with pytest.raises(ValueError, match="later/archived-final"):
        main(
            [
                "--profile",
                "missing.json",
                "--out",
                str(tmp_path / "no-run"),
                "--origin",
                "2016-05-22",
                "--development-only",
            ]
        )
    with pytest.raises(ValueError, match="canonical ISO"):
        main(
            [
                "--profile",
                "missing.json",
                "--out",
                str(tmp_path / "no-run"),
                "--origin",
                "20160424",
                "--development-only",
            ]
        )
    args = [
        "--profile",
        str(tmp_path / "profile.json"),
        "--out",
        str(tmp_path / "no-run"),
        "--development-only",
    ]

    with pytest.raises(ValueError, match="archived-private input paths"):
        main([*args, "--panel", str(tmp_path / "organizer_private")])
    (tmp_path / "profile.json").write_text("[]")
    with pytest.raises(ValueError, match="profile must be a JSON object"):
        main(args)
    (tmp_path / "profile.json").write_text(json.dumps(dataclasses.asdict(small_config())))
    (tmp_path / "no-run").mkdir()
    with pytest.raises(ValueError, match="output exists"):
        main(args)
    with pytest.raises(ValueError, match="cannot also replay"):
        main([*args, "--score-only", "--replay-checkpoint", "model.pt"])
    with pytest.raises(ValueError, match="distinct actuals"):
        main([*args, "--score-only", "--actuals", str(ROOT / "data/m5_public_training_d1913_v2")])
    (tmp_path / "no-run/COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="complete revision record"):
        main([*args, "--score-only"])
    # Rejected path graphs must not call a builder or contaminate a Parquet input directory.
    from provenance.lineage import directory_digest

    training, actuals, raw = (tmp_path / name for name in ("training", "actuals", "raw"))
    for folder in (training, actuals, raw):
        folder.mkdir()
        (folder / "marker.parquet").write_bytes(b"untouched fixture")

    def unexpected_builder(*args, **kwargs):
        raise AssertionError("unsafe paths reached calendar creation")

    monkeypatch.setattr(cli, "build_revision_calendar", unexpected_builder)
    fresh = tmp_path / "fresh"
    base = [
        "--profile",
        str(tmp_path / "profile.json"),
        "--panel",
        str(training),
        "--actuals",
        str(actuals),
        "--m5-dir",
        str(raw),
        "--development-only",
    ]
    for calendar_path, output in (
        (training / "calendar.parquet", fresh),
        (actuals / "calendar.parquet", fresh),
        (raw / "calendar.parquet", fresh),
        (fresh / "calendar.parquet", fresh),
        (ROOT / "forecast/never-create-calendar.parquet", fresh),
        (tmp_path / "safe-calendar.parquet", training / "new-run"),
        (tmp_path / "same", tmp_path / "same"),
    ):
        before = directory_digest(tmp_path)
        before_paths = set(tmp_path.rglob("*"))
        with pytest.raises(ValueError, match="outputs overlap"):
            main([*base, "--calendar", str(calendar_path), "--out", str(output)])
        assert directory_digest(tmp_path) == before
        assert set(tmp_path.rglob("*")) == before_paths


@pytest.mark.parametrize("variant", ["v2", "v3", "v5"])
@pytest.mark.parametrize("mode", ["revenue", "output_gradient"])
def test_series_loss_weighting_modes_train_for_two_epochs(
    revision_data, tmp_path, variant, mode
):
    """Smoke both opt-in objectives through every supported revision trainer."""
    panel, _data, calendar = revision_data
    profile = json.loads(
        (
            ROOT
            / f"configs/experiments/multibranch_revisions/{REVISION_MODELS[variant]}_campaign_default.json"
        ).read_text()
    )
    profile.update(
        epochs=2,
        max_updates=12,
        progress="off",
        encoder_days=112,
        d_model=16,
        n_heads=4,
        trunk_stages=1,
        decoder_layers=1,
        history_attention_layers=1,
        commercial_attention_layers=1,
        train_windows=7 if variant == "v5" else 3,
        train_stride=1 if variant == "v5" else 7,
        batch_size=6,
        torch_threads=1,
        max_array_bytes=10_000_000,
        hierarchy_loss_weight=0.0,
        event_penalty_weight=0.0,
        series_loss_weighting=mode,
    )
    if variant == "v3":
        profile["v3"] = {
            **profile["v3"],
            "recent": 28,
            "middle_end": 112,
            "old_slots": 2,
            "span": 112,
            "local_blocks": 1,
        }
    elif variant == "v5":
        profile["v5"] = {
            **profile["v5"],
            "recent": 28,
            "middle_end": 112,
            "old_slots": 2,
            "span": 112,
            "n_layers": 1,
            "ff_multiplier": 2,
        }
    config = RollingConfig(**profile)
    output = tmp_path / f"{variant}-{mode}"
    forecast = fit_predict_revision(
        panel,
        "2016-04-24",
        28,
        config=config,
        calendar_path=str(calendar),
        levels=M5_LEVELS,
        out_dir=str(output),
    )

    forecast.validate()
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["binding"]["config"]["series_loss_weighting"] == mode
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    components = [row["loss_components"] for row in rows if "loss_components" in row]
    assert components and {row["series_loss_weighting"] for row in components} == {mode}


def test_profiles_are_explicit_and_share_budget():
    directory = ROOT / "configs/experiments/multibranch_revisions"
    configs = [
        RollingConfig(**json.loads((directory / f"{REVISION_MODELS[variant]}.json").read_text()))
        for variant in ("v1_corrected", "v2")
    ]
    for field in ("max_updates", "seed", "train_windows", "train_stride", "batch_size"):
        assert getattr(configs[0], field) == getattr(configs[1], field)
    assert configs[0].factor_rank == 8 and configs[1].factor_rank == 0
    assert configs[0].trunk_stages == configs[0].decoder_layers == 4
    assert configs[0].commercial_attention_layers == configs[0].event_kernel_layers == 4


def test_corrected_mft_retains_six_events_static_selection_and_price_penalty(revision_data):
    panel, data, calendar = revision_data
    c = small_config("v1_corrected", price_penalty_weight=0.1)
    features = RevisionFeatures(panel, data, c, str(calendar))
    assert len(features.event_names) == 6 and "snap" not in features.known_names
    inputs, y, mask = features.batch("2016-03-27", np.arange(6), "cpu", True)
    assert y is not None and mask is not None
    model = features.make_model()
    assert model.selection is not None
    assert model.price_response_mode == "soft_penalty_v1"
    loss = features.training_loss(model, inputs, y, mask, float(mask.sum()))
    loss.backward()
    assert torch.isfinite(loss)


def test_price_hinge_is_conditional_not_likelihood_importance_scaled(revision_data, monkeypatch):
    panel, data, calendar = revision_data
    config = small_config("v1_corrected", covariate_dropout=0.0, price_penalty_weight=0.1)
    features = RevisionFeatures(panel, data, config, str(calendar))
    inputs, y, observed = features.batch("2016-03-27", np.arange(6), "cpu", True)
    assert y is not None and observed is not None and inputs["commercial_mask"].any()
    model = features.make_model()
    theta = torch.tensor(2.0, requires_grad=True)
    # Isolate the two objective reductions, not forecasting performance on invented data.
    monkeypatch.setattr(model, "loss", lambda target, state, **kwargs: theta * state.eq(0).any())
    monkeypatch.setattr(model, "price_penalty", lambda given: (5 * theta, 0.0))
    # With two origins but only one usable, likelihood gets 1/2; the conditional hinge does not.
    objective = features.training_loss(model, inputs, y, observed, 1.0, importance=0.5)
    torch.testing.assert_close(objective, 0.5 * theta + 0.1 * 5 * theta)
    objective.backward()
    torch.testing.assert_close(theta.grad, torch.tensor(1.0))
    # An excluded origin can have a price signal despite no likelihood-informative targets.
    excluded = features.training_loss(
        model, inputs, torch.zeros_like(y), torch.zeros_like(observed), 1.0, importance=1.0
    )
    torch.testing.assert_close(excluded, 0.1 * 5 * theta)


def test_cli_fit_score_and_sealed_forecast_guard(revision_data, tmp_path, capsys, monkeypatch):
    from scripts import run_multibranch_revision as cli
    from scripts.run_multibranch_revision import main

    panel, _, calendar = revision_data
    actuals = tmp_path / "actuals.parquet"
    pq.write_table(pq.read_table(panel.path), actuals)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(dataclasses.asdict(small_config())))
    args = [
        "--profile",
        str(profile),
        "--panel",
        panel.path,
        "--calendar",
        str(calendar),
        "--out",
        str(tmp_path / "run"),
        "--origin",
        "2016-03-27",
        "--development-only",
    ]
    assert main(args) == 0
    assert main([*args, "--score-only", "--actuals", str(actuals)]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["role"] == "retrospective_development_only"
    assert result["n_series_scored"]["item_id+store_id"] == 12
    assert result["origin"] == "2016-03-27" and result["horizon"] == 28
    assert result["run_directory"] == str((tmp_path / "run").resolve())
    assert str(ROOT / "autoforecast") in result["scoring_binding"]["sources"]
    assert result["scoring_binding"]["runtime"]["torch_version"] == str(torch.__version__)
    calendar_bytes = calendar.read_bytes()
    with monkeypatch.context() as patch:

        def forbidden_calendar_grid(*args, **kwargs):
            raise AssertionError("invalid calendar must refuse before reading forecast/actuals")

        patch.setattr(cli, "read_forecast_grid", forbidden_calendar_grid)
        for fault in ("missing", "corrupt", "substituted"):
            score_args = [*args, "--score-only", "--actuals", str(actuals)]
            if fault == "missing":
                calendar.rename(tmp_path / "saved-calendar.parquet")
            elif fault == "corrupt":
                calendar.write_bytes(b"corrupted calendar")
            else:
                substitute = tmp_path / "substitute-calendar.parquet"
                substitute.write_bytes(calendar_bytes)
                score_args[score_args.index("--calendar") + 1] = str(substitute)
            try:
                with pytest.raises(ValueError, match="calendar"):
                    main(score_args)
            finally:
                if fault == "missing":
                    (tmp_path / "saved-calendar.parquet").rename(calendar)
                elif fault == "corrupt":
                    calendar.write_bytes(calendar_bytes)
    complete_path = tmp_path / "run/COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    checkpoint = tmp_path / "run/model.pt"
    checkpoint_bytes = checkpoint.read_bytes()
    for fault in ("missing", "corrupt"):
        if fault == "missing":
            checkpoint.rename(tmp_path / "saved-model.pt")
        else:
            checkpoint.write_bytes(b"corrupted checkpoint")
        try:
            with monkeypatch.context() as patch:

                def forbidden_grid_read(*args, **kwargs):
                    raise AssertionError(
                        "invalid checkpoint must refuse before reading forecast/actuals"
                    )

                patch.setattr(cli, "read_forecast_grid", forbidden_grid_read)
                with pytest.raises(ValueError, match="checkpoint"):
                    main([*args, "--score-only", "--actuals", str(actuals)])
        finally:
            if fault == "missing":
                (tmp_path / "saved-model.pt").rename(checkpoint)
            else:
                checkpoint.write_bytes(checkpoint_bytes)
    changed_complete = json.loads(complete_path.read_text())
    changed_complete["binding"]["series_ids"] = []
    complete_path.write_text(json.dumps(changed_complete))
    with pytest.raises(ValueError, match="scoring population"):
        main([*args, "--score-only", "--actuals", str(actuals)])
    complete_path.write_text(json.dumps(complete))
    digest = cli.directory_digest
    with monkeypatch.context() as patch:
        patch.setattr(
            cli,
            "directory_digest",
            lambda path, **kwargs: "changed-source"
            if pathlib.Path(path) == cli.CODE_SOURCES[0]
            else digest(path, **kwargs),
        )
        with pytest.raises(ValueError, match="scoring source/runtime differs"):
            main([*args, "--score-only", "--actuals", str(actuals)])
    for target, message in (
        (actuals, "actuals changed"),
        (checkpoint, "checkpoint changed"),
        (calendar, "calendar changed"),
        (tmp_path / "run/forecast.parquet", "forecast changed"),
        (cli.CODE_SOURCES[0], "source/runtime or completion"),
        (complete_path, "source/runtime or completion"),
    ):
        calls = 0

        def changing_digest(path, target=target, **kwargs):
            nonlocal calls
            if pathlib.Path(path).resolve() == target.resolve():
                calls += 1
                if calls > 1:
                    return "changed"
            return digest(path, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(cli, "directory_digest", changing_digest)
            with pytest.raises(ValueError, match=message):
                main([*args, "--score-only", "--actuals", str(actuals)])
    path = tmp_path / "run/forecast.parquet"
    table = pq.read_table(path)
    changed = table.set_column(
        table.column_names.index("yhat"), "yhat", pa.array([0.0] * len(table))
    )
    pq.write_table(changed, path)
    with pytest.raises(ValueError, match="binding differs"):
        main([*args, "--score-only", "--actuals", str(actuals)])


def test_refit_seed_reproducibility_and_coverage_guard(revision_data, tmp_path):
    panel, _, calendar = revision_data
    config = small_config()
    args: dict[str, Any] = dict(
        panel=panel,
        origin="2016-04-24",
        horizon=28,
        calendar_path=str(calendar),
        levels=M5_LEVELS,
        config=config,
    )
    first = fit_predict_revision(**args, out_dir=str(tmp_path / "first"))
    second = fit_predict_revision(**args, out_dir=str(tmp_path / "second"))
    assert pq.read_table(first.path).equals(pq.read_table(second.path))
    args["config"] = dataclasses.replace(config, max_updates=1)
    with pytest.raises(ValueError, match="cover all series"):
        fit_predict_revision(**args, out_dir=str(tmp_path / "insufficient"))


def test_cached_metric_matches_spark_on_real_m5(revision_data, tmp_path, monkeypatch):
    """External-to-new-implementation reference: existing production scorer, all 12 levels."""
    import os
    import sys

    from pyspark.sql import SparkSession

    from autoforecast import wrmsse
    from autoforecast.hierarchy_metric import HierarchyMetric
    from autoforecast.panel import Forecast

    if not os.environ.get("JAVA_HOME"):
        pytest.skip("JAVA_HOME required for independent existing-scorer parity")
    monkeypatch.setenv("PYSPARK_PYTHON", sys.executable)
    monkeypatch.setenv("PYSPARK_DRIVER_PYTHON", sys.executable)
    panel, data, _ = revision_data
    origin = "2016-03-27"
    stop = data.position(origin)
    prediction = np.tile(data.target[:, stop - 7 : stop], (1, 4)).astype(float)
    dates = [dt.date.fromisoformat(origin) + dt.timedelta(days=i + 1) for i in range(28)]
    path = tmp_path / "seasonal.parquet"
    pq.write_table(
        pa.table(
            {
                "series_id": np.repeat(data.series_ids, 28),
                "date": dates * 12,
                "yhat": prediction.reshape(-1),
            }
        ),
        path,
    )
    forecast = Forecast(
        adapter="seasonal_naive",
        path=str(path),
        origin=origin,
        horizon=28,
        n_series=12,
        disposition={"emitted": 336},
    )
    metric = HierarchyMetric.build(
        data.metadata, data.target[:, :stop], data.price[:, :stop], M5_LEVELS
    )
    actual = metric.score(prediction, data.target[:, stop : stop + 28])
    # The existing scorer validates its entire input; omit later masked, unscored dates.
    source = pq.read_table(panel.path)
    source = source.filter(pa.compute.less_equal(source["date"], pa.scalar(data.cutoff)))
    score_panel = tmp_path / "observed-history.parquet"
    pq.write_table(source, score_panel)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("revision-metric-parity")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        expected = wrmsse.score(
            dataclasses.replace(panel, path=str(score_panel)),
            forecast,
            spark,
            levels=M5_LEVELS,
            weight_col="revenue",
        ).as_record()
    finally:
        spark.stop()
    assert actual["wrmsse"] == pytest.approx(expected["wrmsse"], rel=1e-12)
    assert isinstance(expected["per_level"], dict)
    for level in metric.names:
        assert actual["per_level"][level] == pytest.approx(expected["per_level"][level], rel=1e-12)
    assert actual["n_series_scored"] == expected["n_series_scored"]
    assert actual["n_series_dropped"] == expected["n_series_dropped"]
