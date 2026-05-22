"""Real M5/UCI HMFNet wiring and explicit fault injection, never tiny-sample skill claims."""

import copy
import dataclasses
import json

import numpy as np
import pytest
import torch

from autoforecast.adapters.mft import (
    HMFNetRevisionFeatures,
    fit_predict_revision,
    revision_data_spec,
)
from autoforecast.benchmarks.m5 import M5_LEVELS
from autoforecast.hierarchy_metric import HierarchyMetric
from autoforecast.hmfnet_features import HMFNetConfig
from autoforecast.rolling_data import GLOBAL_LEFT_PADDING_CONTRACT, PanelArrays
from autoforecast.rolling_training import (
    RollingConfig,
    RollingTrainer,
    rolling_origins,
    staged_hierarchy_backward,
)
from autoforecast.test_multibranch_revisions import small_config
from autoforecast.test_revision_portability import generic_config

pytest_plugins = [
    "autoforecast.test_multibranch_revisions",
    "autoforecast.test_revision_portability",
]


def hmfnet_config(**changes):
    """Two tiny real-population passes, not a forecasting comparison."""
    values = dict(
        epochs=2,
        demand_dropout=0.0,
        covariate_dropout=0.0,
        trunk_stages=2,
        v3=HMFNetConfig(
            context_levels=(("store_id", "dept_id"),),
            expert_levels=(("store_id", "dept_id"),),
            phase_cycles=((7.0, 1), (365.25, 1)),
        ),
    )
    values.update(changes)
    return small_config("v3", **values)


def assembled(revision_data, **changes):
    """Reuse the existing twelve-series real M5 fixture and actual calendar."""
    panel, data, calendar = revision_data
    features = HMFNetRevisionFeatures(panel, data, hmfnet_config(**changes), str(calendar))
    origins = rolling_origins(
        data, data.date_at(data.target.shape[1] - 57).isoformat(), features.config
    )
    features.begin_phase(origins, "selection")
    return features, origins


def assert_tensors_equal(first, second):
    """Recursive equality for input contracts; excludes independently sampled labels."""
    assert set(first) == set(second)
    for name in first:
        if isinstance(first[name], dict):
            assert_tensors_equal(first[name], second[name])
        elif isinstance(first[name], torch.Tensor):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        else:
            assert first[name] == second[name]


def test_hmfnet_resolution(revision_data):
    features, _ = assembled(revision_data)
    assert features.resolved.schema == "autoforecast.resolved-revision.v3"
    assert features.resolved.head == "shifted_hurdle_negbin"
    assert features.feature_layout()["token_numeric_width"] == 19
    model = features.make_model()
    assert len(model.local_blocks) == 2 and model.expert_size is not None
    config = RollingConfig(**json.loads(json.dumps(dataclasses.asdict(features.config))))
    assert config == features.config
    with pytest.raises(ValueError, match="explicit HMFNet"):
        small_config("v3")
    with pytest.raises(ValueError, match="only HMFNet"):
        small_config("v2", v3=HMFNetConfig())
    with pytest.raises(ValueError, match="factorless"):
        hmfnet_config(factor_rank=1)
    with pytest.raises(ValueError, match="compact"):
        hmfnet_config(demand_dropout=0.1)
    panel, data, calendar = revision_data
    with pytest.raises(ValueError, match="explicit HMFNet recipe"):
        HMFNetRevisionFeatures(panel, data, small_config("v2"), str(calendar))
    with pytest.raises(ValueError, match="context grouping set"):
        HMFNetRevisionFeatures(panel, data, hmfnet_config(aggregate_history=False), str(calendar))
    with pytest.raises(ValueError, match="declared context grouping"):
        HMFNetRevisionFeatures(panel, data, hmfnet_config(aggregate_columns=("dept_id",)), str(calendar))
    odd = HMFNetRevisionFeatures(
        panel, data, hmfnet_config(d_model=15, n_heads=3, commercial_attention_heads=2), str(calendar)
    )
    assert odd.has_commercial and odd.make_model().cross_attention.num_heads == 3


def test_hmfnet_1911_day_schedule_keeps_compressed_history_and_all_origins(revision_data):
    """HMFNet accepts the same campaign schedule without pretending to use dense raw attention."""
    panel, data, _ = revision_data
    data = dataclasses.replace(data, observation_contract=GLOBAL_LEFT_PADDING_CONTRACT)
    config = hmfnet_config(
        encoder_days=1911,
        train_windows=52,
        train_stride=28,
        observation_contract=GLOBAL_LEFT_PADDING_CONTRACT,
        mean_weighting="uniform",
    )
    features = HMFNetRevisionFeatures(panel, data, config)
    selection_end = data.date_at(data.target.shape[1] - 2 * data.horizon - 1).isoformat()
    origins = rolling_origins(data, selection_end, config)
    assert len(origins) == 52
    features.begin_phase(origins, "selection")
    inputs, target, target_observed = features.batch(
        min(origins), np.array([0, 1]), "cpu", True
    )
    assert "history" not in inputs and "history_available" not in inputs
    prediction = features.validation_values(features.make_model().eval(), inputs)[0]
    assert prediction.shape == target.shape == target_observed.shape == (2, 28)
    assert torch.isfinite(prediction).all()


def test_hmfnet_cli_binding(revision_data, tmp_path, capsys, monkeypatch):
    """Reuse every existing sealed CLI/source/label/checkpoint fault for the HMFNet schema."""
    from autoforecast import test_multibranch_revisions as existing

    monkeypatch.setattr(existing, "small_config", lambda: hmfnet_config())
    existing.test_cli_fit_score_and_sealed_forecast_guard(
        revision_data, tmp_path, capsys, monkeypatch
    )


def test_hmfnet_typed_history(real_counts):
    """Actual non-retail historical numeric/category fields, frozen OOV, finite span."""
    panel, origin, horizon, weekly = real_counts
    panel = dataclasses.replace(panel, known_future=("workdays",), static=("membership",))
    config = dataclasses.replace(
        generic_config("v2", weekly),
        variant="v3",
        v3=HMFNetConfig(span=28),
        demand_dropout=0.0,
        aggregate_history=False,
        trunk_stages=2,
    )
    spec = revision_data_spec(panel, horizon, config)
    spec.time_varying_unknown_reals = ("temperature",)
    spec.time_varying_unknown_categoricals = ("weather",)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000, data_spec=spec)
    features = HMFNetRevisionFeatures(panel, data, config, data_spec=spec)
    day = data.date_at(data.target.shape[1] - horizon - 1).isoformat()
    features.begin_phase((day,), "selection")
    rows = np.arange(2)
    inputs, y, mask = features.batch(day, rows, "cpu", True)
    assert features.numeric_history == ("temperature",)
    assert features.categorical_history == ("weather",)
    assert inputs["memory"]["memory_categorical_mask"].any()
    assert inputs["memory"]["memory_values"].shape[1] == 28
    model = features.make_model()
    assert torch.isfinite(features.training_loss(model, inputs, y, mask, 2 * horizon))
    stop = data.position(day)
    # Separate calls at one origin must not silently normalize a second field over all history.
    features._covariates(("temperature",), rows, stop - 1, stop, stop)
    raw = data.features["temperature"][:, stop - 28 : stop]
    assert features.numeric_stats["temperature"][0] == pytest.approx(np.nanmean(raw))
    changed_features = dict(data.features)
    raw_category = data.features["weather"].copy()
    raw_category[:, stop - 1] = "not_in_training_vocabulary"
    changed_features["weather"] = raw_category
    features.data = dataclasses.replace(data, features=changed_features)
    altered, _, _ = features.batch(day, rows, "cpu", True)
    assert torch.all(
        altered["memory"]["memory_categorical"][:, 0, 0] == len(features.dynamic_vocab["weather"])
    )
    assert torch.isfinite(features.mean(model, altered)).all()
    features.policy = dataclasses.replace(features.policy, recent=1, middle_end=3, middle_width=2)
    raw_category[:, stop - 3 : stop] = ["unseen_a", "unseen_b", "unseen_b"]
    bucketed, _, _ = features.batch(day, rows, "cpu", True)
    assert torch.all(bucketed["memory"]["memory_values"][:, 1, -2] == 1)


def test_hmfnet_revision_features(revision_data):
    features, origins = assembled(revision_data)
    n = len(features.data.series_ids)
    inputs, _, _ = features.batch(features.data.cutoff.isoformat(), np.arange(n), "cpu", False)
    assert inputs["memory"]["memory_values"].shape == (12, 156, 19)
    assert "expert_supervision" not in inputs
    assert inputs["aggregate"]["anchor"].shape[1] == 28
    model = features.make_model()
    baseline = features.mean(model, inputs)
    totals = np.zeros((len(features.aggregate_data.target), 28))
    np.add.at(totals, features.aggregate_data.experts[:, 0], baseline.detach().numpy())
    np.testing.assert_allclose(totals, inputs["aggregate"]["anchor"], rtol=1e-12, atol=1e-12)
    before, _, _ = features.batch(origins[0], np.arange(n), "cpu", False)
    changed_y = features.data.target.copy()
    changed_y[:, features.data.position(origins[0]) :] += 9  # Explicit future-target fault.
    changed_data = dataclasses.replace(features.data, target=changed_y)
    altered = HMFNetRevisionFeatures(
        features.panel, changed_data, features.config, str(revision_data[2])
    )
    altered.begin_phase(origins, "selection")
    after, _, _ = altered.batch(origins[0], np.arange(n), "cpu", False)
    assert_tensors_equal(before, after)
    # The common baseline is origin-dependent, not fitted through the later endpoint.
    null = features.data.observed.copy()
    null[:, : features.data.position(origins[0])] = False
    no_donors = HMFNetRevisionFeatures(
        features.panel,
        dataclasses.replace(features.data, observed=null),
        features.config,
        str(revision_data[2]),
    )
    with pytest.raises(ValueError, match="donors"):
        no_donors.batch(origins[0], np.arange(n), "cpu", False)


def test_hmfnet_begin_phase(revision_data):
    features, origins = assembled(
        revision_data,
        v3=HMFNetConfig(
            context_levels=(("store_id", "dept_id"),),
            expert_levels=(("store_id", "dept_id"),),
            expert_batch_per_group=2,
        ),
    )
    rows = np.arange(2)
    a, _, _ = features.batch(origins[0], rows, "cpu", True)
    features.batch(origins[1], rows, "cpu", True)
    record = features.begin_phase(origins, "selection")
    b, _, _ = features.batch(origins[0], rows, "cpu", True)
    assert_tensors_equal(a, b)
    assert record["expert_seed"] == 43 and record["v3_feature_layout"]["aggregate_nodes"] > 0


def test_hmfnet_training_loss(revision_data):
    features, origins = assembled(revision_data)
    model = features.make_model()
    inputs, y, observed = features.batch(origins[0], np.arange(6), "cpu", True)
    result = model(**features._model_inputs(inputs))
    supervision = inputs["expert_supervision"]
    expert = (
        (result.expert_mean[supervision["rows"]] - supervision["target"]).square().mean(1)
        / supervision["scale"]
        * supervision["coefficient"]
    ).sum()
    likelihood = result.nll(y, observed).sum() / 168
    point = (
        ((result.mean() - y).square() * observed).mean(1) * inputs["mean_weight"]
    ).sum() / inputs["sampling_fraction"]
    expected = (
        0.3 * (likelihood + 0.2 * point)
        + 0.1 * expert
        + features.config.event_penalty_weight * model.event_penalty()
    )
    actual = features.training_loss(model, inputs, y, observed, 168, 0.3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert model.expert_size.weight.grad.abs().sum() > 0
    diagnostic = features.diagnostics(model, True)
    assert diagnostic["loss_components"]["expert_samples"] == len(supervision["rows"])
    assert diagnostic["gradient_l2_by_module"]["expert_size"] > 0
    assert "stages" not in features.diagnostics(model, False)
    terms = diagnostic["loss_components"]
    assert sum(
        terms[n]
        for n in ("weighted_nll", "weighted_bottom_mean", "weighted_expert", "weighted_event")
    ) == pytest.approx(float(actual.detach()))
    assert terms["importance"] == 0.3
    assert sum(group["sampled"] for group in terms["expert_groups"]) == terms["expert_samples"]
    assert [
        node for group in terms["expert_groups"] for node in group["sampled_node_ids"]
    ] == inputs["aggregate"]["node_ids"][supervision["rows"]].tolist()
    assert [
        v for group in terms["expert_groups"] for v in group["sampled_coefficients"]
    ] == supervision["coefficient"].tolist()
    assert len(features.feature_layout()["aggregate_node_keys"]) == len(
        features.aggregate_data.target
    )
    assert sum(terms["expert_sample_contributions"]) == pytest.approx(terms["expert_loss"])
    for group in terms["expert_groups"]:
        for weight, coefficient in zip(
            group["sampled_normalized_weights"], group["sampled_coefficients"], strict=True
        ):
            assert coefficient == pytest.approx(
                group["eligible"] / group["sampled"] * weight / len(features.policy.expert_levels)
            )
    no_labels = dict(
        inputs, expert_supervision=dict(supervision, target=supervision["target"] + 500)
    )
    torch.testing.assert_close(
        features.mean(model, inputs), features.mean(model, no_labels), rtol=0, atol=0
    )


def test_hmfnet_inference_cache(revision_data):
    features, origins = assembled(revision_data)
    model = features.make_model().eval()
    with torch.no_grad():
        model.expert_size.weight.fill_(0.01)
        model.size.weight.fill_(0.001)
        inputs, _, _ = features.batch(origins[0], np.arange(6), "cpu", False)
        normal = features.mean(model, inputs)
        with features.prediction_context(model, origins[0]):
            assert model._aggregate_cache is not None
            cached = features.mean(model, inputs)
            torch.testing.assert_close(normal, cached, rtol=1e-6, atol=1e-8)
        assert model._aggregate_cache is None
        with pytest.raises(RuntimeError, match="injected"):
            with features.prediction_context(model, origins[0]):
                raise RuntimeError("injected")
        assert model._aggregate_cache is None
    with pytest.raises(ValueError, match="eval/no-grad"):
        with features.prediction_context(model, origins[0]):
            pass


def test_hmfnet_staged_gradient(revision_data):
    features, origins = assembled(revision_data)
    day = origins[0]
    stop = features.data.position(day)
    model = features.make_model().eval()
    with torch.no_grad():
        model.size.weight.fill_(0.0001)
        model.expert_size.weight.fill_(0.0001)
    direct = copy.deepcopy(model)
    data = features.data
    metric = HierarchyMetric.build(
        data.metadata,
        data.target[:, :stop],
        data.price[:, :stop],
        (("store_id",), ("dept_id",)),
        training_surrogate=True,
    )
    chunks = tuple(
        np.arange(i, min(i + 5, len(data.series_ids))) for i in range(0, len(data.series_ids), 5)
    )
    target = torch.tensor(data.target[:, stop : stop + data.horizon])
    staged_hierarchy_backward(
        model,
        chunks,
        lambda rows: features.mean(model, features.batch(day, rows, "cpu", False)[0]),
        metric,
        target,
        weight=0.1,
        epsilon=1e-8,
    )
    full = torch.cat(
        [features.mean(direct, features.batch(day, rows, "cpu", False)[0]) for rows in chunks]
    )
    (0.1 * metric.loss(full, target, epsilon=1e-8)).backward()
    for (name, parameter), (other, reference) in zip(
        model.named_parameters(), direct.named_parameters(), strict=True
    ):
        assert name == other
        assert (parameter.grad is None) == (reference.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, reference.grad, rtol=2e-5, atol=1e-8, msg=name
            )
    assert model.expert_size.weight.grad.abs().sum() > 0


def test_hmfnet_real_m5_replay(revision_data, tmp_path):
    panel, data, calendar = revision_data
    config = hmfnet_config()
    args = dict(
        panel=panel,
        origin=data.cutoff.isoformat(),
        horizon=28,
        config=config,
        calendar_path=str(calendar),
        levels=M5_LEVELS,
    )
    forecast = fit_predict_revision(**args, out_dir=str(tmp_path / "fit"))
    forecast.validate()
    complete = json.loads((tmp_path / "fit/COMPLETE.json").read_text())
    assert complete["binding"]["schema"] == "multibranch_revision_v3"
    assert complete["training"]["epochs_per_phase"] == 2
    assert complete["execution_checks"]["forecast_shape"] == [12, 28]
    assert complete["execution_checks"]["finite"]
    training = complete["training"]
    assert training["phase_coverage"]["selection"]["minimum_visits_per_series"] == 2
    selected_epochs = training["selected_updates"] // training["steps_per_epoch"]
    assert training["phase_coverage"]["refit"]["minimum_visits_per_series"] == selected_epochs
    assert training["refit_optimizer_steps"] == training["selected_updates"]
    assert all(record["series_seen"] == 12 for record in training["phase_coverage"].values())
    replay = fit_predict_revision(
        **args, out_dir=str(tmp_path / "replay"), replay_checkpoint=str(tmp_path / "fit/model.pt")
    )
    assert replay.n_series == 12
    assert json.loads((tmp_path / "replay/COMPLETE.json").read_text())["execution_checks"][
        "replay_exactly_equal"
    ]
    rows = [json.loads(line) for line in (tmp_path / "fit/training.jsonl").read_text().splitlines()]
    assert {r["epoch"] for r in rows} == {1, 2}
    assert any("hierarchy_loss" in r for r in rows)
    assert all(r["gradient_norm_before_clip"] >= 0 for r in rows)
    assert all(r["loss_components"]["expert_samples"] > 0 for r in rows)
    node_ids = {
        node["node_id"]
        for node in complete["binding"]["effective_assembly"]["v3_layout"]["aggregate_node_keys"]
    }
    for row in rows:
        for group in row["loss_components"]["expert_groups"]:
            assert set(group["sampled_node_ids"]) <= node_ids
            assert (
                len(group["sampled_node_ids"])
                == len(group["sampled_coefficients"])
                == group["sampled"]
            )
    for row in rows:
        if "hierarchy_loss" in row:
            assert row["hierarchy_step"]["weighted_objective"] == row["hierarchy_loss"]
            assert row["hierarchy_step"]["gradient_norm_before_clip"] > 0
            assert row["hierarchy_step"]["full_population"] == 12
    for phase, coverage in complete["training"]["phase_coverage"].items():
        assert len(coverage["epochs"]) == (2 if phase == "selection" else selected_epochs)
        assert all(
            epoch["minimum_row_proposals"] == epoch["maximum_row_proposals"] == 1
            for epoch in coverage["epochs"]
        )


def test_hmfnet_epoch_coverage_accounting(real_counts):
    """A late-availability fault must not be mislabeled as two informative visits."""
    panel, origin, horizon, weekly = real_counts
    config = dataclasses.replace(
        generic_config("v2", weekly),
        variant="v3",
        v3=HMFNetConfig(),
        demand_dropout=0.0,
        aggregate_history=False,
        trunk_stages=2,
        train_windows=2,
        batch_size=2,
        hierarchy_loss_weight=0,
    )
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000)
    validation_start = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    origins = rolling_origins(data, validation_start, config)
    mask = data.observed.copy()
    mask[1, : data.position(origins[0])] = False
    data = dataclasses.replace(data, observed=mask)
    features = HMFNetRevisionFeatures(panel, data, config)
    trainer = RollingTrainer(
        data,
        config,
        (),
        features.groups,
        features.make_model,
        features.batch,
        features.training_loss,
        features.mean,
        features.optimizer_groups,
        features.begin_phase,
        features.prediction_context,
        features.diagnostics,
    )
    _, training = trainer.fit(lambda record: None)
    selected_epochs = training["selected_updates"] // training["steps_per_epoch"]
    for phase_name, phase in training["phase_coverage"].items():
        expected_epochs = 2 if phase_name == "selection" else selected_epochs
        assert phase["minimum_row_proposals"] == phase["maximum_row_proposals"] == expected_epochs
        assert all(epoch["minimum_row_proposals"] == 1 for epoch in phase["epochs"])
    epochs = training["phase_coverage"]["selection"]["epochs"]
    assert epochs[0]["informative_series"] == 2
    assert epochs[1]["informative_series"] == 1
    assert epochs[1]["minimum_informative_visits"] == 0


def test_hmfnet_real_count_replay(real_counts, tmp_path):
    panel, origin, horizon, weekly = real_counts
    config = dataclasses.replace(
        generic_config("v2", weekly),
        variant="v3",
        v3=HMFNetConfig(),
        demand_dropout=0.0,
        aggregate_history=False,
        trunk_stages=2,
    )
    args = dict(panel=panel, origin=origin, horizon=horizon, config=config)
    forecast = fit_predict_revision(**args, out_dir=str(tmp_path / "fit"))
    complete = json.loads((tmp_path / "fit/COMPLETE.json").read_text())
    assert complete["binding"]["resolved_build"]["head"] == "negbin"
    assert complete["binding"]["effective_assembly"]["v3_layout"]["aggregate_nodes"] == 0
    replay = fit_predict_revision(
        **args, out_dir=str(tmp_path / "replay"), replay_checkpoint=str(tmp_path / "fit/model.pt")
    )
    assert replay.n_series == forecast.n_series


def test_hmfnet_context_without_experts(revision_data):
    policy = HMFNetConfig(context_levels=(("store_id", "dept_id"),), expert_levels=(), feedback=False)
    features, origins = assembled(revision_data, v3=policy)
    model = features.make_model()
    assert model.expert_size is None
    inputs, _, _ = features.batch(origins[0], np.arange(3), "cpu", False)
    with torch.no_grad():
        model.size.weight.fill_(0.01)
    before = features.mean(model, inputs)
    inputs["aggregate"]["memory_values"] += 0.1
    assert not torch.equal(before, features.mean(model, inputs))
