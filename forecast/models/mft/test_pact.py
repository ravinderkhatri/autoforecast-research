"""Structural and numerical tests for the PACT decoder and coherent output law."""

from __future__ import annotations

import dataclasses
import math

import forecast.models.mft.pact as pact_module
import pytest
import torch
from forecast.models.mft.pact import (
    PACTBatch,
    PACTLabels,
    PACTModel,
    PACTModelInputs,
    PACTPrediction,
    apply_signed_rope,
    full_attention_mask,
    prefix_attention_mask,
)


def _segments() -> tuple[torch.Tensor, torch.Tensor]:
    active = torch.tensor(
        [[True, True, False, False, False, False, False],
         [False, False, True, True, False, False, False]]
    )
    index = torch.tensor(
        [[0, 1, -1, -1, -1, -1, -1], [-1, -1, 2, 3, -1, -1, -1]],
        dtype=torch.int64,
    )
    return active, index


def _prediction(law: str = "zero_truncated_hurdle_negbin") -> PACTPrediction:
    active, index = _segments()
    allocation = torch.tensor(
        [[[0.25, 0.75, 0, 0, 0, 0, 0], [0, 0, 0.6, 0.4, 0, 0, 0]]],
        dtype=torch.float64,
    )
    allocation_log = torch.where(
        active[None], torch.log(allocation.clamp_min(torch.finfo(torch.float64).tiny)), 0
    )
    return PACTPrediction(
        law=law,
        p_positive=torch.tensor([[0.7, 0.8]], dtype=torch.float64)
        if law != "negbin"
        else None,
        mu=torch.tensor([[2.0, 4.0]], dtype=torch.float64),
        k=torch.tensor([[1.5, 2.0]], dtype=torch.float64),
        allocation=allocation,
        allocation_log=allocation_log,
        active=active,
        day_index=index,
        horizon=4,
    )


def _model_inputs(*, streams: int = 1) -> dict[str, torch.Tensor | int]:
    torch.manual_seed(4)
    batch, history, segments = 2, 5, 2
    active, day_index = _segments()
    return {
        "history_features": torch.randn(batch, streams, history, 9),
        "history_exists": torch.ones(batch, streams, history, dtype=torch.bool),
        "history_positions": torch.tensor([-28, -7, -2, -1, 0]),
        "history_types": torch.tensor([2, 1, 0, 0, 0]),
        "future_features": torch.randn(batch, segments, 11),
        "future_exists": torch.ones(batch, segments, dtype=torch.bool),
        "future_positions": torch.tensor([2, 4]),
        "future_types": torch.tensor([4, 5]),
        "future_active": active,
        "future_day_index": day_index,
        "statistics": torch.randn(batch, 3),
        "attributes": torch.randn(batch, 2),
        "identity_indices": torch.tensor([[0], [1]]),
        "horizon": 4,
    }


def _typed_model_inputs(*, streams: int = 1) -> PACTModelInputs:
    return PACTModelInputs(**_model_inputs(streams=streams))


def _model(*, layers: int = 2, streams: int = 1, law: str = "zero_truncated_hurdle_negbin") -> PACTModel:
    torch.manual_seed(7)
    return PACTModel(
        history_width=9,
        future_width=11,
        statistic_width=3,
        attribute_width=2,
        identity_cardinalities=(2,),
        d_model=16,
        n_heads=2,
        n_layers=layers,
        ff_multiplier=2,
        rope_theta=10_000.0,
        position_mode="elapsed_day",
        max_streams=max(streams, 1),
        head_law=law,
    ).eval()


def test_pact_attention_graph() -> None:
    exists = torch.tensor([[True, True, True, True, False]])
    allowed, safe = prefix_attention_mask(exists, 3)
    assert allowed[0, 0].tolist() == [True, False, False, False, False]
    assert allowed[0, 2].tolist() == [True, True, True, False, False]
    assert allowed[0, 3].tolist() == [True, True, True, True, False]
    assert not allowed[0, 0, 3]
    assert safe[0, 4].any() and not allowed[0, 4].any()


def test_full_attention_mask_excludes_padding_and_keeps_safe_rows() -> None:
    exists = torch.tensor([[True, False, True], [True, True, True]])

    allowed, safe = full_attention_mask(exists)

    assert allowed[0, 0].tolist() == [True, False, True]
    assert not allowed[0, 1].any()
    assert safe[0, 1].tolist() == [True, False, False]
    assert allowed[1].all()


def test_pact_model_inputs_as_kwargs_preserves_the_typed_boundary() -> None:
    inputs = _typed_model_inputs()

    values = inputs.as_kwargs()

    assert tuple(values) == tuple(_model_inputs())
    assert values["history_features"] is inputs.history_features
    assert values["horizon"] == inputs.horizon


def test_pact_typed_batch_validates_model_and_label_boundaries() -> None:
    inputs = _typed_model_inputs()
    labels = PACTLabels(
        daily_target=torch.zeros((2, 4)),
        objective_day_mask=torch.ones((2, 4), dtype=torch.bool),
        target_counts=torch.zeros((2, 2, 7)),
        segment_eligible=torch.ones((2, 2), dtype=torch.bool),
        series_loss_weight=torch.ones(2),
    )
    batch = PACTBatch(inputs, torch.zeros((2, 4, 3)), labels)
    moved = batch.to_device("cpu")
    assert isinstance(moved.inputs, PACTModelInputs)
    assert isinstance(moved.labels, PACTLabels)
    assert moved.inputs.horizon == 4
    assert moved.labels.series_loss_weight is not None

    for invalid_weight in (
        torch.ones(1),
        torch.ones(2, dtype=torch.int64),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, -1.0]),
    ):
        with pytest.raises(ValueError, match="series loss weights"):
            dataclasses.replace(labels, series_loss_weight=invalid_weight)

    invalid = _model_inputs()
    invalid["history_positions"] = invalid["history_positions"].float()
    with pytest.raises(ValueError, match="history positions"):
        PACTModelInputs(**invalid)
    with pytest.raises(ValueError, match="batch labels disagree"):
        PACTBatch(
            inputs,
            torch.zeros((2, 4, 3)),
            PACTLabels(
                daily_target=torch.zeros((2, 3)),
                objective_day_mask=torch.ones((2, 3), dtype=torch.bool),
                target_counts=torch.zeros((2, 2, 7)),
                segment_eligible=torch.ones((2, 2), dtype=torch.bool),
            ),
        )
    with pytest.raises(ValueError, match="future segments"):
        PACTBatch(
            inputs,
            torch.zeros((2, 4, 3)),
            PACTLabels(
                daily_target=torch.zeros((2, 4)),
                objective_day_mask=torch.ones((2, 4), dtype=torch.bool),
                target_counts=torch.zeros((2, 3, 7)),
                segment_eligible=torch.ones((2, 3), dtype=torch.bool),
            ),
        )


def test_pact_rope_contract() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 2, 3, 8, dtype=torch.float64)
    key = torch.randn(1, 2, 3, 8, dtype=torch.float64)
    signed = torch.tensor([[-28, -7, 4]])
    offset = signed + 100
    signed_score = apply_signed_rope(query, signed, 10_000.0) @ apply_signed_rope(
        key, signed, 10_000.0
    ).transpose(-1, -2)
    offset_score = apply_signed_rope(query, offset, 10_000.0) @ apply_signed_rope(
        key, offset, 10_000.0
    ).transpose(-1, -2)
    assert torch.allclose(signed_score, offset_score, atol=2e-13, rtol=2e-13)


def test_pact_decoder_forward_and_padding_invariance() -> None:
    model = _model(layers=2)
    inputs = _model_inputs()
    first = model(**inputs)
    assert first.mean().shape == (2, 4)
    assert torch.allclose(first.mean().sum(-1), first.segment_mean().sum(-1), atol=1e-6)

    padded = dict(inputs)
    padded["history_features"] = inputs["history_features"].clone()
    padded["history_exists"] = inputs["history_exists"].clone()
    padded["history_exists"][:, :, 0] = False
    padded["history_features"][:, :, 0] = torch.randn_like(
        padded["history_features"][:, :, 0]
    ) * 1e6
    zeroed = dict(padded)
    zeroed["history_features"] = padded["history_features"].clone()
    zeroed["history_features"][:, :, 0] = 0
    assert torch.allclose(model(**padded).mean(), model(**zeroed).mean(), atol=1e-6, rtol=1e-6)


def test_pact_aggregate_reaches_future_only_from_layer_two() -> None:
    inputs = _model_inputs(streams=2)
    changed = dict(inputs)
    changed["history_features"] = inputs["history_features"].clone()
    changed["history_features"][:, 1] += 3.0
    one = _model(layers=1, streams=2)
    two = _model(layers=2, streams=2)
    assert torch.equal(one(**inputs).mean(), one(**changed).mean())
    assert not torch.allclose(two(**inputs).mean(), two(**changed).mean())


def test_pact_probability_contract_and_shifted_law_difference() -> None:
    prediction = _prediction()
    counts = torch.tensor(
        [[[1.0, 2.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]],
        dtype=torch.float64,
    )
    nll = prediction.joint_nll(counts, torch.ones((1, 2), dtype=torch.bool))
    assert torch.isfinite(nll).all() and bool((nll >= 0).all())
    expected = prediction.p_positive * prediction.mu / (
        1 - torch.pow(prediction.k / (prediction.k + prediction.mu), prediction.k)
    )
    assert torch.allclose(prediction.segment_mean(), expected)
    shifted_mean = prediction.p_positive * (1 + prediction.mu)
    assert not torch.allclose(prediction.segment_mean(), shifted_mean)


def test_pact_daily_marginal_matches_joint_enumeration() -> None:
    prediction = _prediction()
    label = torch.tensor([[1.0, 0.0, 2.0, 0.0]], dtype=torch.float64)
    marginal = prediction.daily_log_prob(label).exp()
    for day in range(4):
        segment, slot = next(
            (segment, slot)
            for segment in range(2)
            for slot in range(7)
            if int(prediction.day_index[segment, slot]) == day
        )
        probability = 0.0
        target = int(label[0, day])
        share = float(prediction.allocation[0, segment, slot])
        p = float(prediction.p_positive[0, segment])
        mu = float(prediction.mu[0, segment])
        k = float(prediction.k[0, segment])
        q0 = (k / (k + mu)) ** k
        for total in range(max(target, 1), 120):
            nb = math.exp(
                float(
                    torch.lgamma(torch.tensor(total + k))
                    - torch.lgamma(torch.tensor(k))
                    - torch.lgamma(torch.tensor(total + 1.0))
                    + k * math.log(k / (k + mu))
                    + total * math.log(mu / (k + mu))
                )
            )
            total_probability = p * nb / (1 - q0)
            conditional = math.comb(total, target) * share**target * (1 - share) ** (total - target)
            probability += total_probability * conditional
        if target == 0:
            probability += 1 - p
        assert abs(float(marginal[0, day]) - probability) < 2e-7


def test_pact_samples_are_integer_and_segment_coherent(monkeypatch) -> None:
    prediction = _prediction("negbin")
    expected_totals = torch.tensor(
        [[[3, 4]], [[1, 2]], [[0, 5]]], dtype=torch.int64
    )
    monkeypatch.setattr(
        pact_module,
        "sample_paths_negbin",
        lambda mu, k, n_draws, generator=None: expected_totals,
    )
    paths = prediction.sample(3, generator=torch.Generator().manual_seed(9))
    assert paths.shape == (3, 1, 4)
    assert paths.dtype == torch.int64 and bool((paths >= 0).all())
    actual_totals = torch.stack((paths[..., :2].sum(-1), paths[..., 2:].sum(-1)), -1)
    assert torch.equal(actual_totals, expected_totals)


def test_pact_extreme_allocation_log_is_finite() -> None:
    active, index = _segments()
    allocation = torch.tensor(
        [[[0.0, 1.0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0.0, 0, 0, 0]]],
        dtype=torch.float64,
    )
    allocation_log = torch.tensor(
        [[[-1000.0, 0.0, 0, 0, 0, 0, 0], [0, 0, 0.0, -1000.0, 0, 0, 0]]],
        dtype=torch.float64,
    )
    prediction = PACTPrediction(
        law="negbin",
        p_positive=None,
        mu=torch.ones((1, 2), dtype=torch.float64),
        k=torch.ones((1, 2), dtype=torch.float64),
        allocation=allocation,
        allocation_log=allocation_log,
        active=active,
        day_index=index,
        horizon=4,
    )
    counts = torch.tensor(
        [[[1.0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0]]],
        dtype=torch.float64,
    )
    assert torch.isfinite(prediction.joint_nll(counts, torch.ones((1, 2), dtype=torch.bool))).all()


@pytest.mark.parametrize("defect", ["duplicate", "negative", "out_of_range", "inactive", "empty"])
def test_pact_mapping_rejects_invalid_day_layouts(defect: str) -> None:
    prediction = _prediction()
    active = prediction.active.clone()
    index = prediction.day_index.clone()
    if defect == "duplicate":
        index[0, 1] = 0
    elif defect == "negative":
        index[0, 1] = -4
    elif defect == "out_of_range":
        index[0, 1] = prediction.horizon
    elif defect == "inactive":
        index[0, 2] = 0
    else:
        active[0] = False
        index[0] = -1
    with pytest.raises(ValueError, match="day mapping|day indices"):
        dataclasses.replace(prediction, active=active, day_index=index)
    inputs = _model_inputs()
    with pytest.raises(ValueError, match="day mapping|day indices"):
        PACTModelInputs(**{**inputs, "future_active": active, "future_day_index": index})


def test_pact_prediction_rejects_incoherent_allocation_and_dtype() -> None:
    prediction = _prediction()
    invalid = prediction.allocation.clone()
    invalid[0, 0, 0], invalid[0, 0, 1] = -1, 2
    with pytest.raises(ValueError, match="allocation probabilities"):
        dataclasses.replace(prediction, allocation=invalid)
    with pytest.raises(ValueError, match="disagree with the simplex"):
        dataclasses.replace(prediction, allocation_log=torch.zeros_like(prediction.allocation_log))
    with pytest.raises(ValueError, match="floating tensors"):
        dataclasses.replace(prediction, mu=prediction.mu.long(), k=prediction.k.long())


def test_pact_plain_nb_zero_share_is_degenerate_at_zero() -> None:
    prediction = _prediction("negbin")
    allocation = prediction.allocation.clone()
    allocation[0, 0, 0], allocation[0, 0, 1] = 0, 1
    allocation_log = prediction.allocation_log.clone()
    allocation_log[0, 0, 0], allocation_log[0, 0, 1] = -1000, 0
    prediction = dataclasses.replace(
        prediction, allocation=allocation, allocation_log=allocation_log
    )
    zero = torch.zeros((1, 4), dtype=torch.float64)
    assert prediction.daily_log_prob(zero)[0, 0] == 0
    positive = zero.clone()
    positive[0, 0] = 1
    assert prediction.daily_log_prob(positive)[0, 0] == -torch.inf


def test_pact_daily_marginal_rejects_fractional_counts() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        _prediction().daily_log_prob(torch.tensor([[0.5, 0, 0, 0]], dtype=torch.float64))
