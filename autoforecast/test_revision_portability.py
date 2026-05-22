"""Real UCI count-data portability, feature-availability and fixed-epoch integration tests.

Structural fault mutations are explicit. Tiny-run scores test execution only, not accuracy.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from decimal import Decimal
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from forecast.models.mft.branches import DemandHistoryBranch
from spec.forecast_spec import ForecastDataSpec, ForecastModelSpec
from spec.resolved import ResolvedBuild, ResolveError

from autoforecast.adapters.mft import (
    RevisionFeatures,
    _revision_event_records,
    fit_predict_revision,
    revision_data_spec,
)
from autoforecast.benchmarks.bike_sharing import build_parquet
from autoforecast.hierarchy_metric import HierarchyMetric, read_forecast_grid
from autoforecast.panel import Forecast, Panel, period_days
from autoforecast.rolling_data import GLOBAL_LEFT_PADDING_CONTRACT, PanelArrays
from autoforecast.rolling_training import RollingConfig, RollingTrainer, rolling_origins

ROOT = pathlib.Path(__file__).resolve().parents[1]


def generic_config(variant, weekly=False, **changes):
    """Small fixed-epoch recipe, with no retail metadata requirement."""
    config = RollingConfig(
        variant=variant,
        encoder_days=16 if weekly else 30,
        patch_size=4 if weekly else 5,
        statistic_windows=(2, 4, 8),
        seasonal_window=52 if weekly else 28,
        reference_window=4 if weekly else 28,
        d_model=16,
        n_heads=4,
        trunk_stages=1,
        decoder_layers=1,
        train_windows=3,
        train_stride=4 if weekly else 14,
        epochs=2,
        batch_size=1,
        max_updates=1,
        demand_dropout=0.0,
        covariate_dropout=0.0,
        torch_threads=1,
        aggregate_history=False,
        mean_loss_weight=0.2 if variant == "v2" else 0.0,
        hierarchy_loss_weight=0.0,
        event_penalty_weight=0.0,
        factor_rank=0,
        observation_contract="masked_observed_counts_v1",
    )
    return dataclasses.replace(config, **changes)


@pytest.fixture(scope="module", params=[False, True], ids=["daily", "weekly"])
def real_counts(request, tmp_path_factory):
    archive = ROOT / "data/bike_sharing/bike-sharing.zip"
    if not archive.exists():
        pytest.skip("download the pinned UCI Bike Sharing archive; see revision runbook")
    weekly = request.param
    path = build_parquet(
        archive, tmp_path_factory.mktemp("real-bike-counts") / "panel.parquet", weekly=weekly
    )
    dates = sorted(set(pq.read_table(path)["timestamp"].to_pylist()))
    horizon = 4 if weekly else 14
    origin = dates[-horizon - 1].isoformat()
    panel = Panel(
        str(path),
        "series",
        "timestamp",
        "rentals",
        target_estimand="observed_count",
        frequency="W" if weekly else "D",
        season_length=52 if weekly else 7,
    )
    return panel, origin, horizon, weekly


def test_period_days_and_grid(real_counts):
    panel, origin, horizon, _ = real_counts
    assert [period_days(v) for v in ("D", "W", "7D", "2D")] == [1, 7, 7, 2]
    for value in ("H", "M", "0D", "-1D", "", None, True):
        with pytest.raises(ValueError):
            period_days(cast(str, value))
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    assert data.date_at(data.position(origin) - 1).isoformat() == origin
    assert data.step_days == period_days(panel.frequency)
    if data.step_days > 1:
        with pytest.raises(ValueError, match="misaligned"):
            data.position((dt.date.fromisoformat(origin) - dt.timedelta(days=1)).isoformat())


def test_bike_build_parquet_refuses_tampered_source_before_output(tmp_path):
    """A synthetic corrupt archive tests refusal, not forecasting performance."""
    archive, output = tmp_path / "tampered.zip", tmp_path / "new" / "panel.parquet"
    archive.write_bytes(b"not the pinned primary-source archive")
    with pytest.raises(ValueError, match="source archive differs"):
        build_parquet(archive, output, weekly=False)
    assert not output.parent.exists()


def test_bike_build_parquet_refuses_nonboolean_weekly_before_output(tmp_path):
    archive = ROOT / "data/bike_sharing/bike-sharing.zip"
    if not archive.is_file():
        pytest.skip("requires the pinned real UCI Bike Sharing archive")
    output = tmp_path / "new" / "panel.parquet"
    for weekly in (0, 1, "false", None):
        with pytest.raises(ValueError, match="weekly must be boolean"):
            build_parquet(archive, output, weekly=weekly)
    assert not output.parent.exists()


def test_data_spec_validate_refuses_nonboolean_event_availability():
    spec = ForecastDataSpec(time_idx="date", target="qty", group_ids=("id",))
    message = "events_preannounced must be boolean"
    for value in (0, 1, "false", None):
        assert message in dataclasses.replace(spec, events_preannounced=value).validate()
    for value in (False, True):
        assert message not in dataclasses.replace(spec, events_preannounced=value).validate()


@pytest.mark.parametrize("fraction", ["1.000000000000000001", "16777216.000000000000000001"])
def test_revision_decimal_counts(real_counts, tmp_path, fraction):
    """Decimal counts are validated before float conversion; unknown fillers are not labels."""
    panel, origin, horizon, _ = real_counts
    table = pq.read_table(panel.path)
    values = [Decimal(v) for v in table[panel.target_col].to_pylist()]
    values[0] = Decimal(2**24)
    path = tmp_path / "decimal.parquet"
    changed = dataclasses.replace(panel, path=str(path), exposure_col="state")
    states = ["exact"] * table.num_rows
    table = table.append_column("state", pa.array(states))

    def write():
        pq.write_table(
            table.set_column(
                table.column_names.index(panel.target_col),
                panel.target_col,
                pa.array(values, type=pa.decimal128(30, 18)),
            ),
            path,
        )

    write()
    data = PanelArrays.read(changed, origin, horizon, max_bytes=3_000_000)
    assert data.target.max() == 2**24
    values[0] = Decimal(fraction)
    write()
    with pytest.raises(ValueError, match="integer counts"):
        PanelArrays.read(changed, origin, horizon, max_bytes=3_000_000)
    states[0] = "unobserved"
    table = table.set_column(table.column_names.index("state"), "state", pa.array(states))
    write()
    with pytest.raises(ValueError, match="integer counts"):
        PanelArrays.read(changed, origin, horizon, max_bytes=3_000_000)
    data = PanelArrays.read(
        changed,
        origin,
        horizon,
        max_bytes=3_000_000,
        observation_contract="masked_observed_counts_v1",
    )
    assert (~data.observed).sum() == 1 and (data.target[~data.observed] == 0).all()


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_revision_single_commercial_dropout(real_counts, tmp_path, variant):
    """A lone promotion channel can drop in revisions, without changing the legacy default."""
    from forecast.models.mft.regularisation import drop_covariates

    panel, origin, horizon, weekly = real_counts
    table = pq.read_table(panel.path)
    table = table.append_column("promotion", pa.array([1.0] * table.num_rows))
    path = tmp_path / "promotions.parquet"
    pq.write_table(table, path)
    panel = dataclasses.replace(
        panel, path=str(path), known_future=("promotion",), promo_col="promotion"
    )
    config = generic_config(variant, weekly, covariate_dropout=0.8)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000)
    features = RevisionFeatures(panel, data, config)
    inputs, y, observed = features.batch(
        data.date_at(data.target.shape[1] - horizon - 1).isoformat(), np.arange(2), "cpu", True
    )
    assert y is not None and observed is not None
    assert inputs["commercial_mask"].tolist() == [[False, False, True]] * 2
    _, legacy = drop_covariates(
        inputs["commercial"],
        inputs["commercial_mask"],
        0.8,
        generator=torch.Generator().manual_seed(0),
    )
    assert legacy[:, 2].all()
    torch.manual_seed(0)
    changed = features.regularized_inputs(inputs)
    assert not changed["commercial_mask"].any()
    for name in ("promo", "promo_available"):
        assert not changed["known_future"][..., features.known_names.index(name)].any()
    model = features.make_model()
    mean = features.mean(model, changed)
    assert torch.isfinite(mean).all()
    loss = features.training_loss(model, changed, y, observed, float(observed.sum()))
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_real_count_portability(real_counts, tmp_path, variant):
    panel, origin, horizon, weekly = real_counts
    config = generic_config(variant, weekly)
    forecast = fit_predict_revision(
        panel, origin, horizon, config=config, out_dir=str(tmp_path / "fit")
    )
    forecast.validate()
    complete = json.loads((tmp_path / "fit/COMPLETE.json").read_text())
    assert complete["binding"]["resolved_build"]["head"] == "negbin"
    assert complete["binding"]["resolved_build"]["factor_rank"] == 0
    assert complete["binding"]["effective_assembly"]["commercial"] is False
    assert complete["events"] == []
    assert complete["training"]["selection_metric"] == "rmse"
    assert complete["training"]["epochs_per_phase"] == 2
    assert complete["training"]["refit_series_seen_in_likelihood"] == 2
    rows = [json.loads(line) for line in (tmp_path / "fit/training.jsonl").read_text().splitlines()]
    steps_per_epoch = complete["training"]["steps_per_epoch"]
    selection_records = [row for row in rows if row["phase"] == "selection"]
    refit_records = [row for row in rows if row["phase"] == "refit"]
    assert len(selection_records) == 2 * steps_per_epoch
    assert len(refit_records) == complete["training"]["selected_updates"]
    for records in (selection_records, refit_records):
        expected_updates = len(records)
        assert [row["epoch"] for row in records] == [
            update // steps_per_epoch + 1 for update in range(expected_updates)
        ]
    replay = fit_predict_revision(
        panel,
        origin,
        horizon,
        config=config,
        out_dir=str(tmp_path / "replay"),
        replay_checkpoint=str(tmp_path / "fit/model.pt"),
    )
    assert pq.read_table(forecast.path).equals(pq.read_table(replay.path))
    predicted = read_forecast_grid(
        forecast.path,
        ("casual", "registered"),
        origin,
        horizon,
        series_col="series_id",
        time_col="date",
        value_col="yhat",
        frequency=panel.frequency,
    )
    actual = read_forecast_grid(
        panel.path,
        ("casual", "registered"),
        origin,
        horizon,
        series_col="series",
        time_col="timestamp",
        value_col="rentals",
        frequency=panel.frequency,
        filter_actuals=True,
    )
    assert predicted.shape == actual.shape == (2, horizon)
    assert np.isfinite(np.sqrt(np.mean((predicted - actual) ** 2)))


def test_resolved_revision_build():
    data = ForecastDataSpec("time", "target", ("series",), target_estimand="observed_count")
    model = ForecastModelSpec(
        factor_rank=0, topology="bottom_up", emit="parameters", attribution=None
    )
    for variant in ("v1_corrected", "v2", "v3"):
        build = ResolvedBuild.for_revision(data, model, variant=variant, zero_share=0.01)
        assert build.head == "negbin" and build.factor_rank == 0
        assert ResolvedBuild.from_record(build.as_record()) == build
        high = ResolvedBuild.for_revision(data, model, variant=variant, zero_share=0.6)
        assert high.head == (
            "hurdle_negbin" if variant == "v1_corrected" else "shifted_hurdle_negbin"
        )
        assert high.build_hash != build.build_hash

        if variant in {"v2", "v3"}:
            with pytest.raises(
                ResolveError, match="SFT/HMFNet do not implement a shared probabilistic factor"
            ):
                factor_model = dataclasses.replace(
                    model, factor_rank=2, topology="bottom_up_factor", factor_dynamics="iid"
                )
                ResolvedBuild.for_revision(data, factor_model, variant=variant)

        for changed in (
            dataclasses.replace(data, target_family="continuous"),
            dataclasses.replace(data, target_allows_negative=True),
        ):
            with pytest.raises(ValueError, match="observed nonnegative counts"):
                ResolvedBuild.for_revision(changed, model, variant=variant)
        with pytest.raises(ValueError, match="count heads"):
            ResolvedBuild.for_revision(
                data, dataclasses.replace(model, head="point"), variant=variant
            )


@pytest.mark.parametrize("series_count", (1, 3))
@pytest.mark.parametrize("arrow_type", (pa.float32(), pa.float64()), ids=("source32", "source64"))
@pytest.mark.parametrize("use_data_spec", (False, True), ids=("panel", "data_spec"))
def test_panel_arrays_read_budget_counts_static_reals(
    tmp_path, series_count, arrow_type, use_data_spec,
):
    """Synthetic columns isolate a byte-accounting failure, not forecasting accuracy."""
    path = tmp_path / "panel.parquet"
    pq.write_table(pa.table({
        "id": [f"series-{i}" for i in range(series_count)],
        "date": [dt.date(2020, 1, 1)] * series_count,
        "qty": [1] * series_count,
        "static_a": pa.array([1.0] * series_count, type=arrow_type),
        "static_b": pa.array([2.0] * series_count, type=arrow_type),
    }), path)
    bare = Panel(str(path), "id", "date", "qty", target_estimand="observed_count")
    # Each series needs 4 target + 1 observed + 2 seen bytes before static roles.
    base_bytes = 7 * series_count
    control = PanelArrays.read(bare, "2020-01-01", 1, max_bytes=base_bytes)
    assert not control.static_reals
    names = ("static_a", "static_b")
    panel = bare if use_data_spec else dataclasses.replace(bare, static_reals=names)
    spec = ForecastDataSpec("date", "qty", ("id",), static_reals=names,
                            target_estimand="observed_count") if use_data_spec else None
    # Both retained static roles are float64, even when their source is float32.
    required_bytes = base_bytes + 16 * series_count
    for budget in (base_bytes, required_bytes - 1):
        with pytest.raises(ValueError, match="panel arrays require .* bytes, budget"):
            PanelArrays.read(panel, "2020-01-01", 1, max_bytes=budget, data_spec=spec)
    arrays = PanelArrays.read(panel, "2020-01-01", 1,
                              max_bytes=required_bytes, data_spec=spec)
    assert sum(value.nbytes for value in arrays.static_reals.values()) == 16 * series_count
    for name, expected in zip(names, (1.0, 2.0), strict=True):
        assert arrays.static_reals[name].dtype == np.float64
        np.testing.assert_array_equal(arrays.static_reals[name], [expected] * series_count)


def test_generic_panel_read(real_counts, tmp_path):
    panel, origin, horizon, weekly = real_counts
    config = generic_config("v2", weekly)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    assert data.observed.all() and data.price.shape == (2, 0) and not data.features
    changed = dataclasses.replace(
        panel,
        known_future=("workdays",),
        static=("membership",),
        static_reals=("registered_member",),
    )
    spec = revision_data_spec(changed, horizon, config)
    spec.time_varying_unknown_reals = ("temperature",)
    spec.time_varying_unknown_categoricals = ("weather",)
    enriched = PanelArrays.read(changed, origin, horizon, max_bytes=3_000_000, data_spec=spec)
    assert enriched.features["temperature"].shape == enriched.target.shape
    assert enriched.features["workdays"].shape[1] == enriched.target.shape[1] + horizon
    assert "registered_member" in enriched.static_reals
    features = RevisionFeatures(changed, enriched, config, data_spec=spec)
    features.begin_phase(
        (enriched.date_at(enriched.target.shape[1] - horizon - 1).isoformat(),), "refit"
    )
    inputs, _, _ = features.batch(origin, np.arange(2), "cpu", False)
    assert inputs["history_covariates"].shape[-1] >= 2
    assert features.make_model().demand.n_covariates == features.history_width

    padded_config = dataclasses.replace(
        config,
        encoder_days=120,
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
    )
    padded_data = dataclasses.replace(
        enriched, observation_contract=GLOBAL_LEFT_PADDING_CONTRACT
    )
    padded_spec = dataclasses.replace(spec, max_encoder_length=120)
    padded_features = RevisionFeatures(
        changed, padded_data, padded_config, data_spec=padded_spec
    )
    padded, _, _ = padded_features.batch(
        padded_data.date_at(0).isoformat(), np.arange(2), "cpu", True
    )
    assert not padded["history_available"][:, :-1].any()
    assert not padded["history_covariate_mask"][:, :-1].any()
    assert padded["history_covariates"].shape[1] == 120
    rows = pq.read_table(panel.path).to_pylist()
    for row in rows:
        row["state"] = "exact"
    row = next(row for row in rows if row["timestamp"].isoformat() == origin)
    row["state"], row["rentals"] = "unobserved", -999
    path = tmp_path / "explicit-unknown.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    masked = dataclasses.replace(panel, path=str(path), exposure_col="state")
    safe = PanelArrays.read(
        masked,
        origin,
        horizon,
        max_bytes=1_000_000,
        observation_contract="masked_observed_counts_v1",
    )
    assert np.all(safe.target[~safe.observed] == 0) and (~safe.observed).sum() == 1
    with pytest.raises(ValueError, match="integer counts"):
        PanelArrays.read(masked, origin, horizon, max_bytes=1_000_000)


def test_history_covariates(real_counts):
    panel, origin, horizon, _ = real_counts
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    history = torch.tensor(data.target[:, -16:])
    observed = torch.ones_like(history, dtype=torch.bool)
    cov = torch.tensor(data.target[:, -17:-1, None] / 1000)
    mask = torch.ones_like(cov, dtype=torch.bool)
    model = DemandHistoryBranch(16, 16, weekly_patch=4, n_covariates=1, dropout=0.0)
    first = model(history, observed, covariates=cov, covariate_mask=mask)
    other = model(history, observed, covariates=cov + 1, covariate_mask=mask)
    assert not torch.equal(first[0], other[0])
    first[0].sum().backward()
    assert model.daily.weight.grad is not None
    assert model.daily.weight.grad[:, 3].abs().sum() > 0
    mask[:, 2] = False
    altered = cov.clone()
    altered[:, 2] = float("nan")
    safe = model(history, observed, covariates=cov, covariate_mask=mask)
    perturbed = model(history, observed, covariates=altered, covariate_mask=mask)
    torch.testing.assert_close(safe[0], perturbed[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="layout"):
        model(history, observed, covariates=cov[:, :-1], covariate_mask=mask)


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_generic_revision_model(real_counts, variant):
    panel, origin, horizon, weekly = real_counts
    config = generic_config(variant, weekly)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    features = RevisionFeatures(panel, data, config)
    model = features.make_model().eval()
    assert model.commercial is None and model.calendar is None
    assert model.demand.weekly_patch == config.patch_size
    inputs, _, _ = features.batch(origin, np.arange(2), "cpu", False)
    assert inputs["attributes"].shape == (2, 0)
    mean = features.mean(model, inputs)
    assert isinstance(inputs["baseline_log_size"], torch.Tensor)
    expected = inputs["baseline_log_size"].exp()
    if expected.ndim == 1:
        expected = expected[:, None].expand_as(mean)
    torch.testing.assert_close(mean, expected.to(mean), rtol=2e-6, atol=1e-5)
    assert "aggregate_history" not in inputs and data.price.size == 0
    if variant == "v2":
        assert model.aggregate is None and model.occurrence is None


def test_revision_frequency_replay(real_counts, tmp_path):
    panel, origin, horizon, _ = real_counts
    dates = [
        dt.date.fromisoformat(origin) + dt.timedelta(days=period_days(panel.frequency) * i)
        for i in range(1, horizon + 1)
    ]
    path = tmp_path / "forecast.parquet"
    # Structural output-schema fixture only; no forecasting-performance finding.
    pq.write_table(
        pa.table(dict(series_id=["one"] * horizon, date=dates, yhat=[1.0] * horizon)), path
    )
    forecast = Forecast(
        "structural",
        str(path),
        origin,
        horizon,
        1,
        {"emitted": horizon},
        target_estimand="observed_count",
        frequency=panel.frequency,
    )
    assert Forecast.from_record(forecast.as_record()).validate() == forecast
    if panel.frequency == "W":
        with pytest.raises(ValueError, match="outside"):
            dataclasses.replace(forecast, frequency="D").validate()


def test_uniform_hierarchy_without_prices(real_counts):
    panel, origin, horizon, _ = real_counts
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    metric = HierarchyMetric.build(
        data.metadata,
        data.target,
        None,
        ((), ("series",)),
        weighting="uniform",
        training_surrogate=True,
    )
    assert sum(metric.weights[-1]) == pytest.approx(1.0)
    prediction = torch.tensor(data.target[:, -horizon:]).requires_grad_(True)
    value = metric.loss(prediction, prediction.detach() + 1, epsilon=1e-8)
    value.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_generic_revision_features(real_counts, tmp_path, variant):
    """Disabled families have no upstream columns; declared generic inputs are trainable."""
    panel, origin, horizon, weekly = real_counts
    disabled = generic_config(
        variant,
        weekly,
        aggregate_columns=("absent_aggregate",),
        factor_columns=("absent_factor",),
        event_pool_columns=("absent_pool",),
        calendar_group_column="absent_scope",
    )
    result = fit_predict_revision(
        panel, origin, horizon, config=disabled, out_dir=str(tmp_path / "disabled")
    )
    assert result.validate() == result
    panel = dataclasses.replace(
        panel,
        known_future=("workdays",),
        static=("membership",),
        static_reals=("registered_member",),
    )
    config = generic_config(variant, weekly, identity_columns=("membership", "series"))
    spec = revision_data_spec(panel, horizon, config)
    spec.time_varying_unknown_reals = ("temperature",)
    spec.time_varying_unknown_categoricals = ("weather",)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000, data_spec=spec)
    features = RevisionFeatures(panel, data, config, data_spec=spec)
    train_origin = rolling_origins(data, origin, config)[0]
    features.begin_phase((train_origin,), "refit")
    model = features.make_model()
    assert len(model.static.residual_identities) == 2
    assert features._groups(("membership", "series")).tolist() == [0, 1]
    with pytest.raises(ValueError, match="not admitted"):
        features._indices("absent_identity")
    inputs, target, observed = features.batch(train_origin, np.arange(2), "cpu", True)
    assert target is not None and observed is not None
    optimizer = torch.optim.AdamW(features.optimizer_groups(model), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = features.training_loss(model, inputs, target, observed, float(observed.sum()), 1.0)
        assert loss.dtype == torch.float64
        loss.backward()
        optimizer.step()
    assert model.demand.daily.weight.grad is not None
    assert model.demand.daily.weight.grad[:, 3:].abs().sum() > 0
    assert torch.isfinite(features.mean(model, inputs)).all()


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_revision_origin_integrity(real_counts, tmp_path, variant):
    """Structural calendar faults use real counts, not fabricated performance evidence."""
    panel, origin, horizon, weekly = real_counts
    panel = dataclasses.replace(panel, known_future=("workdays",), static=("membership",))
    config = generic_config(variant, weekly, head="negbin", event_max_support=1)
    spec = revision_data_spec(panel, horizon, config)
    spec.time_varying_unknown_reals = ("temperature",)
    spec.time_varying_unknown_categoricals = ("weather",)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000, data_spec=spec)
    selection_end = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    origins = rolling_origins(data, selection_end, config)
    early = min(origins)
    stop = data.position(early)
    features = RevisionFeatures(panel, data, config, data_spec=spec)
    admission = features.begin_phase(origins, "selection")
    before, _, _ = features.batch(early, np.arange(2), "cpu", False)
    modified = dataclasses.replace(
        data,
        target=data.target.copy(),
        features={key: value.copy() for key, value in data.features.items()},
    )
    modified.target[:, stop:] += 100
    modified.features["temperature"][:, stop:] = 777
    modified.features["weather"][:, stop:] = "unseen_after_origin"
    other = RevisionFeatures(panel, modified, config, data_spec=spec)
    assert other.begin_phase(origins, "selection") == admission
    after, _, _ = other.batch(early, np.arange(2), "cpu", False)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, after[name], rtol=0, atol=0)

    # A generic locally scoped event, with a separate announcement date and episode ID.
    contract = dict(
        time_col="when",
        group_col="scope",
        group_column="membership",
        known_at_col="announced",
        preannounced=False,
        complete_grid=False,
        event_pairs=[["event", None]],
        known_future=[],
        variants={},
        binary_columns=[],
        episode_col="episode",
    )
    first_day = data.date_at(stop)
    second_day = first_day + dt.timedelta(days=config.train_stride * data.step_days)
    rows = [
        dict(
            when=day,
            scope="casual",
            announced=dt.date(2000, 1, 1),
            event="local_event",
            episode=f"episode-{i}",
        )
        for i, day in enumerate((first_day, second_day))
    ]
    calendar = tmp_path / "local.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"revision_calendar_v1": json.dumps(contract).encode()}
    )
    pq.write_table(table, calendar)
    scoped = RevisionFeatures(panel, data, config, str(calendar), data_spec=spec)
    declared = dataclasses.replace(
        spec, events=str(calendar), event_known_at_col="announced", event_episode_col="episode"
    )
    RevisionFeatures(panel, data, config, str(calendar), data_spec=declared)
    separate = dataclasses.replace(declared, events=None)
    RevisionFeatures(panel, data, config, str(calendar), data_spec=separate)
    with pytest.raises(ValueError, match="calendar input"):
        RevisionFeatures(panel, data, config, data_spec=separate)
    for change in (
        dict(events=str(tmp_path / "wrong.parquet")),
        dict(event_known_at_col="wrong"),
        dict(event_episode_col="wrong"),
        dict(events_preannounced=True),
    ):
        with pytest.raises(ValueError, match="calendar roles disagree"):
            RevisionFeatures(
                panel,
                data,
                config,
                str(calendar),
                data_spec=dataclasses.replace(declared, **change),
            )
        if "events" not in change:
            with pytest.raises(ValueError, match="calendar roles disagree"):
                RevisionFeatures(
                    panel,
                    data,
                    config,
                    str(calendar),
                    data_spec=dataclasses.replace(separate, **change),
                )
    report = scoped.begin_phase(origins, "selection")
    event_report = report["events"]
    assert isinstance(event_report, dict)
    assert event_report["name:local_event"]["enabled"] is True
    _, present, _ = scoped._calendar_at(early, [first_day])
    assert present[scoped.calendar_scopes.index("casual")].any()
    assert not present[scoped.calendar_scopes.index("registered")].any()
    late_rows = [dict(row, announced=first_day) for row in rows]
    late = tmp_path / "late.parquet"
    pq.write_table(pa.Table.from_pylist(late_rows, schema=table.schema), late)
    announced = RevisionFeatures(panel, data, config, str(late), data_spec=spec)
    assert not announced._calendar_at(early, [first_day])[1].any()
    assert announced._calendar_at(first_day.isoformat(), [first_day])[1].any()
    broken = dict(contract, complete_grid=True)
    gap = tmp_path / "gap.parquet"
    pq.write_table(
        table.replace_schema_metadata({b"revision_calendar_v1": json.dumps(broken).encode()}), gap
    )
    with pytest.raises(ValueError, match="incomplete calendar"):
        RevisionFeatures(panel, data, config, str(gap), data_spec=spec)


def test_revision_epochs(real_counts):
    """Selection covers the maximum epochs; refit uses the validation-selected update count."""
    panel, origin, horizon, weekly = real_counts
    config = generic_config(
        "v2", weekly, hierarchy_loss_weight=0.1, hierarchy_every=2, patience=1, evaluate_every=1
    )
    data = PanelArrays.read(
        panel,
        origin,
        horizon,
        max_bytes=1_000_000,
        observation_contract=config.observation_contract,
    )
    features = RevisionFeatures(panel, data, config)
    trainer = RollingTrainer(
        data,
        config,
        ((), ("series",)),
        features.groups,
        features.make_model,
        features.batch,
        features.training_loss,
        features.mean,
        features.optimizer_groups,
        features.begin_phase,
    )
    records = []
    _, report = trainer.fit(records.append)
    selection = [record for record in records if record["phase"] == "selection"]
    assert [record["epoch"] for record in selection] == [1, 1, 2, 2]
    assert sum("hierarchy_loss" in record for record in selection) == 2
    refit = [record for record in records if record["phase"] == "refit"]
    assert len(refit) == report["selected_updates"]
    assert all(record["likelihood_informative_series_in_batch"] == 1 for record in records)
    assert [record["origin_sampling"] for record in selection] == [
        "coverage_curriculum",
        "coverage_curriculum",
        "uniform_usable_origins",
        "uniform_usable_origins",
    ]
    assert report["refit_series_seen_in_likelihood"] == 2
    assert report["selected_updates"] in {2, 4}
    assert report["budget_policy"] == "validation_selected_updates"
    bad = dataclasses.replace(data, observed=data.observed.copy())
    bad.observed[0, -1] = False
    with pytest.raises(ValueError, match="complete recorded targets"):
        dataclasses.replace(trainer, data=bad).fit(records.append)
    with pytest.raises(ValueError, match="positive integer"):
        dataclasses.replace(config, epochs=True)


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
def test_revision_resolved_settings(real_counts, variant):
    """Resolver knobs must reach the executable model, or fail explicitly before training."""
    from spec.forecast_spec import ExposureSpec

    panel, origin, horizon, weekly = real_counts
    bare_config = generic_config(variant, weekly)
    bare_data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    bare = RevisionFeatures(panel, bare_data, bare_config)
    assert bare.resolved.static_variable_selection is False
    assert getattr(bare.make_model(), "selection", None) is None
    panel = dataclasses.replace(panel, static=("membership",))
    config = generic_config(variant, weekly)
    spec = revision_data_spec(panel, horizon, config)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000, data_spec=spec)
    for width in (0, 4, 16):
        selected = dataclasses.replace(spec, embedding_residual_dim=width)
        features = RevisionFeatures(panel, data, config, data_spec=selected)
        model = features.make_model()
        assert features.resolved.embedding_residual_dim == model.static.residual_dim == width
        assert features.resolved.static_variable_selection == (
            getattr(model, "selection", None) is not None
        )
        inputs, _, _ = features.batch(origin, np.arange(2), "cpu", False)
        assert torch.isfinite(features.mean(model, inputs)).all()
    for change in (
        dict(exposure=ExposureSpec(ranged_col="not_consumed")),
        dict(lead_time_plus_review=4),
        dict(allow_joint_embedding=True),
        dict(aggregate_level="not_consumed"),
        dict(event_windows={"holiday": (-1, 1)}),
    ):
        with pytest.raises(ValueError, match="does not implement"):
            RevisionFeatures(panel, data, config, data_spec=dataclasses.replace(spec, **change))
    altered = dataclasses.replace(
        spec, observation=dataclasses.replace(spec.observation, quality_gate=0.2)
    )
    with pytest.raises(ValueError, match="quality-policy"):
        RevisionFeatures(panel, data, config, data_spec=altered)


def revision_calendar_fixture(tmp_path, rows, **changes):
    """Structural calendar metadata over real counts; not accuracy evidence."""
    contract = dict(
        time_col="when",
        group_col=None,
        group_column=None,
        known_at_col="announced",
        preannounced=False,
        complete_grid=False,
        event_pairs=[["event", None]],
        known_future=[],
        variants={},
        binary_columns=[],
        episode_col="episode",
    )
    contract.update(changes)
    path = tmp_path / "calendar.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"revision_calendar_v1": json.dumps(contract).encode()}
    )
    pq.write_table(table, path)
    return str(path)


@pytest.mark.parametrize("field", ["event", "kind", "episode", "scope"])
def test_revision_event_records_refuse_blank_identifiers(field):
    contract = dict(
        time_col="when",
        group_col="scope",
        known_at_col="announced",
        event_pairs=[["event", "kind"]],
        episode_col="episode",
        variants={},
    )
    row = dict(
        when=dt.date(2020, 1, 1),
        announced=dt.date(2019, 1, 1),
        event="holiday",
        kind="national",
        episode="holiday-2020",
        scope="region-a",
    )
    row[field] = " "
    with pytest.raises(ValueError, match="identity|episodes|scopes"):
        _revision_event_records([row], contract, "v2", dt.date(2020, 1, 1))


@pytest.mark.parametrize("field", ["event", "episode", "scope"])
def test_revision_calendar_refuses_blank_identifiers(real_counts, tmp_path, field):
    panel, origin, horizon, weekly = real_counts
    config = generic_config("v2", weekly, event_max_support=1)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    row = dict(
        when=data.cutoff,
        announced=dt.date(2000, 1, 1),
        event="known",
        episode="one",
        scope="casual",
    )
    row[field] = " "
    calendar = revision_calendar_fixture(
        tmp_path,
        [row],
        group_col="scope",
        group_column="membership",
    )
    with pytest.raises(ValueError, match="identity|episodes|scopes"):
        RevisionFeatures(panel, data, config, calendar)


def test_revision_later_event_occurrence(real_counts, tmp_path):
    panel, origin, horizon, weekly = real_counts
    config = generic_config("v2", weekly, event_max_support=1)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    selection_end = data.date_at(data.target.shape[1] - 2 * horizon - 1)
    forecast_day = data.cutoff + dt.timedelta(days=data.step_days)
    rows = [
        dict(
            when=selection_end - dt.timedelta(days=i * 30),
            announced=dt.date(2000, 1, 1),
            event="known_event",
            episode=f"past-{i}",
        )
        for i in (1, 2)
    ]
    rows.extend(
        [
            dict(
                when=forecast_day,
                announced=selection_end + dt.timedelta(days=1),
                event="known_event",
                episode="later-occurrence",
            ),
            dict(
                when=forecast_day + dt.timedelta(days=1),
                announced=selection_end + dt.timedelta(days=1),
                event="new_untrained_event",
                episode="new-channel",
            ),
        ]
    )
    calendar = revision_calendar_fixture(tmp_path, rows)
    features = RevisionFeatures(panel, data, config, calendar)
    assert features.event_names == ("name:known_event",)
    assert not features._calendar_at(selection_end.isoformat(), [forecast_day])[1].any()
    distance, present, episodes = features._calendar_at(origin, [forecast_day])
    assert present.all() and np.all(distance == 0)
    assert episodes[0, 0, 0] == "later-occurrence"


def test_revision_event_support_days(real_counts, tmp_path):
    panel, origin, horizon, weekly = real_counts
    config = generic_config("v2", weekly, event_max_support=1)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    event_day = data.cutoff
    calendar = revision_calendar_fixture(
        tmp_path,
        [dict(when=event_day, announced=dt.date(2000, 1, 1), event="known", episode="one")],
    )
    features = RevisionFeatures(panel, data, config, calendar)
    days = [event_day + dt.timedelta(days=i) for i in (1, 6)]
    distance, present, _ = features._calendar_at(origin, days)
    assert distance[0, 0, 0] == 1 and present[0, 0, 0]
    assert not present[0, 1, 0]
    assert features.resolved.event_max_support == (-1, 1)
    selection_end = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    first = dt.date.fromisoformat(min(rolling_origins(data, selection_end, config))) + dt.timedelta(
        days=data.step_days - config.event_max_support
    )
    last = data.cutoff + dt.timedelta(days=horizon * data.step_days + config.event_max_support)
    rows = [
        dict(
            when=first + dt.timedelta(days=i),
            announced=dt.date(2000, 1, 1),
            event="known" if first + dt.timedelta(days=i) == event_day else None,
            episode="one",
        )
        for i in range((last - first).days + 1)
    ]
    complete = revision_calendar_fixture(tmp_path, rows, complete_grid=True)
    RevisionFeatures(panel, data, config, complete)
    # Even weekly forecasts require daily event coverage when complete_grid is promised.
    gap = revision_calendar_fixture(tmp_path, rows[1:], complete_grid=True)
    with pytest.raises(ValueError, match="incomplete calendar"):
        RevisionFeatures(panel, data, config, gap)


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
@pytest.mark.parametrize("empty", ["variant", "catalog"])
def test_revision_disabled_calendar_dependencies(real_counts, tmp_path, variant, empty):
    panel, origin, horizon, weekly = real_counts
    rows = [
        dict(
            when=dt.date.fromisoformat(origin),
            announced=dt.date(2000, 1, 1),
            event=None if empty == "catalog" else "unused",
            episode="one",
            scope="unused",
        )
    ]
    calendar = revision_calendar_fixture(
        tmp_path,
        rows,
        group_col="scope",
        group_column="absent_scope",
        variants={variant: {"events": []}} if empty == "variant" else {},
    )
    config = generic_config(variant, weekly, event_pool_columns=("absent_pool",))
    result = fit_predict_revision(
        panel,
        origin,
        horizon,
        config=config,
        calendar_path=calendar,
        out_dir=str(tmp_path / "disabled"),
    )
    assert result.validate() == result
    complete = json.loads((tmp_path / "disabled/COMPLETE.json").read_text())
    assert complete["events"] == []
    assert complete["binding"]["effective_assembly"]["event_pool_columns"] == []
    active = revision_calendar_fixture(tmp_path, [dict(rows[0], event="active")])
    with pytest.raises(ValueError, match="absent_pool"):
        fit_predict_revision(
            panel,
            origin,
            horizon,
            config=config,
            calendar_path=active,
            out_dir=str(tmp_path / "active"),
        )


@pytest.mark.parametrize("weighting", ["auto", "uniform"])
def test_revision_uniform_hierarchy_selection(real_counts, tmp_path, weighting):
    panel, origin, horizon, weekly = real_counts
    config = generic_config(
        "v2",
        weekly,
        selection_metric="wrmsse",
        hierarchy_weighting=weighting,
        hierarchy_loss_weight=0.1,
        hierarchy_every=2,
    )
    fit_predict_revision(
        panel,
        origin,
        horizon,
        config=config,
        levels=((), ("series",)),
        out_dir=str(tmp_path / "uniform"),
    )
    complete = json.loads((tmp_path / "uniform/COMPLETE.json").read_text())
    training = complete["training"]
    assert training["selection_metric"] == "uniform_hierarchy_rmsse"
    assert training["hierarchy_weighting"] == "uniform"
    assert len(training["validation"]) == 2
    assert all(np.isfinite(v["mean_uniform_hierarchy_rmsse"]) for v in training["validation"])
    with pytest.raises(ValueError, match="price"):
        fit_predict_revision(
            panel,
            origin,
            horizon,
            config=dataclasses.replace(config, hierarchy_weighting="revenue"),
            levels=((), ("series",)),
            out_dir=str(tmp_path / "revenue"),
        )


def test_revision_aggregate_adaptation(real_counts, tmp_path):
    panel, origin, horizon, weekly = real_counts
    config = generic_config("v2", weekly, aggregate_history=True)
    fit_predict_revision(panel, origin, horizon, config=config, out_dir=str(tmp_path / "auto"))
    complete = json.loads((tmp_path / "auto/COMPLETE.json").read_text())
    assert complete["binding"]["effective_assembly"]["adaptation_reasons"] == {
        "aggregate_history": "disabled: no aggregate_columns declared"
    }
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    disabled = RevisionFeatures(panel, data, dataclasses.replace(config, aggregate_history=False))
    assert disabled.make_model().aggregate is None and disabled.adaptation_reasons == {}


def test_revision_numeric_stats_cache(real_counts):
    """Per-origin statistics are cached across chunks, never across a changed origin."""
    panel, origin, horizon, weekly = real_counts
    config = generic_config("v2", weekly)
    spec = revision_data_spec(panel, horizon, config)
    spec.time_varying_unknown_reals = ("temperature",)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000, data_spec=spec)
    features = RevisionFeatures(panel, data, config, data_spec=spec)
    first = features.batch(origin, np.array([0]), "cpu", False)[0]
    cache = features.numeric_stats
    assert set(cache) == {"temperature"}
    statistics = cache["temperature"]
    second = features.batch(origin, np.array([1]), "cpu", False)[0]
    assert features.numeric_stats is cache and features.numeric_stats["temperature"] is statistics
    combined = features.batch(origin, np.arange(2), "cpu", False)[0]
    torch.testing.assert_close(
        torch.cat([first["history_covariates"], second["history_covariates"]]),
        combined["history_covariates"],
    )
    older = data.date_at(data.position(origin) - 2).isoformat()
    features.batch(older, np.array([0]), "cpu", False)
    assert features.numeric_stats is not cache
    stop = data.position(older)
    assert features.numeric_stats["temperature"][0] == pytest.approx(
        np.nanmean(data.features["temperature"][:, :stop])
    )


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
@pytest.mark.parametrize("hurdle", [False, True])
def test_revision_likelihood_precision(monkeypatch, variant, hurdle):
    """Structural boundary test: k=2 has closed-form log mass and analytic derivatives."""
    from forecast.models.mft.sft import ShiftedHurdlePrediction

    from autoforecast.adapters.mft import CustomModel

    y = torch.tensor([[0, 1, 1000, 1_000_000, 2**24]], dtype=torch.float32)
    mu = (0.75 * (y + 1)).requires_grad_(True)
    k = torch.full_like(y, 2.0, requires_grad=True)
    logit = torch.full_like(y, 0.4)
    if variant == "v2":
        prediction = ShiftedHurdlePrediction(
            logit, mu, k, head_kind="shifted_hurdle_negbin" if hurdle else "negbin"
        )
        loss = prediction.nll(y, torch.ones_like(y, dtype=torch.bool)).sum()
        p = logit.double().sigmoid()
    else:
        p = logit.sigmoid() if hurdle else None
        model = CustomModel(16, 28, 0, 1, 1, 5, factor_rank=0)
        monkeypatch.setattr(
            model,
            "forward",
            lambda **kw: (
                p.unsqueeze(0) if p is not None else None,
                mu.unsqueeze(0),
                k.unsqueeze(0),
            ),
        )
        loss = model.loss(
            y,
            torch.zeros_like(y, dtype=torch.long),
            normalization_count=1,
            likelihood_dtype=torch.float64,
        )
        p = logit.sigmoid().double()
    count = (y.double() - int(variant == "v2" and hurdle)).clamp_min(0)
    mean, dispersion = mu.detach().double(), k.detach().double()
    log_zero = dispersion * (dispersion.log() - (mean + dispersion).log())
    log_nb = count.log1p() + log_zero + count * torch.log1p(-dispersion / (mean + dispersion))
    expected = -log_nb
    grad_mu = dispersion * (mean - count) / (mean * (mean + dispersion))
    grad_k = -(
        torch.special.digamma(count + dispersion)
        - torch.special.digamma(dispersion)
        + dispersion.log()
        - (dispersion + mean).log()
        + 1
        - (count + dispersion) / (mean + dispersion)
    )
    if hurdle:
        expected = -p.log() - log_nb
        if variant == "v1_corrected":
            zero = log_zero.exp()
            expected += torch.log1p(-zero)
            grad_mu += zero * dispersion / ((mean + dispersion) * (1 - zero))
            grad_k -= (
                zero
                / (1 - zero)
                * (
                    dispersion.log()
                    - (mean + dispersion).log()
                    + 1
                    - dispersion / (mean + dispersion)
                )
            )
        expected = torch.where(y > 0, expected, -torch.log1p(-p))
        grad_mu = torch.where(y > 0, grad_mu, 0)
        grad_k = torch.where(y > 0, grad_k, 0)
    torch.testing.assert_close(loss.double(), expected.sum(), rtol=1e-8, atol=5e-7)
    loss.backward()
    assert mu.grad is not None and k.grad is not None
    torch.testing.assert_close(mu.grad.double(), grad_mu, rtol=2e-5, atol=1e-10)
    torch.testing.assert_close(k.grad.double(), grad_k, rtol=2e-5, atol=1e-7)


def test_revision_calendar_covariates_only(real_counts, tmp_path):
    panel, origin, horizon, weekly = real_counts
    config = generic_config("v2", weekly, event_pool_columns=("unused",))
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    selection_end = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    first = data.position(min(rolling_origins(data, selection_end, config)))
    rows = [
        dict(when=data.date_at(i), announced=dt.date(2000, 1, 1), day_signal=1)
        for i in range(first, data.target.shape[1] + horizon)
    ]
    calendar = revision_calendar_fixture(
        tmp_path,
        rows,
        event_pairs=[],
        episode_col=None,
        known_future=["day_signal"],
        complete_grid=True,
    )
    features = RevisionFeatures(panel, data, config, calendar)
    assert features.event_names == () and features.event_pool_index is None
    inputs, _, _ = features.batch(origin, np.arange(2), "cpu", False)
    assert torch.all(inputs["known_future"][..., features.known_names.index("day_signal")] == 1)


@pytest.mark.parametrize("variant", ["v1_corrected", "v2"])
@pytest.mark.parametrize("head", ["negbin", "hurdle_negbin"])
def test_revision_tiny_mean_gradient(real_counts, variant, head):
    from forecast.models.mft.sft import ShiftedHurdlePrediction

    panel, origin, horizon, weekly = real_counts
    config = generic_config(variant, weekly, head=head, mean_loss_weight=0, covariate_dropout=0)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=1_000_000)
    features = RevisionFeatures(panel, data, config)
    model = features.make_model()
    inputs, _, _ = features.batch(origin, np.array([0]), "cpu", False)
    inputs["baseline_log_size"].fill_(
        0 if variant == "v1_corrected" or head == "negbin" else float(np.log(2))
    )
    size = cast(torch.nn.Linear, model.head.size if variant == "v1_corrected" else model.size)
    assert size.bias is not None
    with torch.no_grad():
        size.bias.fill_(-90)
    y = torch.full((1, horizon), 2.0)
    loss = features.training_loss(
        model, inputs, y, torch.ones_like(y, dtype=torch.bool), float(horizon)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert size.bias.grad is not None and torch.isfinite(size.bias.grad).all()
    assert size.bias.grad.item() == pytest.approx(-2 if head == "negbin" else -1, abs=2e-5)
    if variant == "v2":
        assert model(**inputs).excess.dtype == torch.float64
        # A parameter-space float32 input has an unrepresentable derivative at this scale.
        theta = torch.tensor([[-90.0]], requires_grad=True)
        unsafe = ShiftedHurdlePrediction(
            torch.zeros_like(theta), theta.exp(), torch.ones_like(theta)
        )
        with pytest.raises(ValueError, match="log predictors"):
            unsafe.nll(torch.full_like(theta, 2), torch.ones_like(theta, dtype=torch.bool))
