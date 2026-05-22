"""Structural and numerical SFT checks, not simulated forecasting evidence."""

import math
from typing import Any

import pytest
import torch
from forecast.models.mft.sft import (
    SFTModel,
    ShiftedHurdlePrediction,
    level_statistics,
)


def test_level_statistics():
    history = torch.tensor([[0.0, 2.0, 4.0, 999.0]])
    mask = torch.tensor([[True, True, True, False]])
    values = level_statistics(history, mask)
    assert values.shape == (1, 12)
    assert values[0, 0].item() == pytest.approx(math.log1p(2))
    assert values[0, 2].item() == pytest.approx(2 / 3)
    assert values[0, 3].item() == pytest.approx(math.log1p(3))
    assert values[0, 4].item() == 0.75
    assert torch.isfinite(level_statistics(torch.zeros(1, 28), torch.zeros(1, 28))).all()
    for bad in (
        torch.tensor([1.0]),
        torch.tensor([[-1.0]]),
        torch.tensor([[math.nan]]),
        torch.empty(1, 0),
    ):
        with pytest.raises(ValueError):
            level_statistics(bad, torch.ones_like(bad))
    for windows in ((0,), (-1,), (1.5,), (True,)):
        with pytest.raises(ValueError, match="positive integer periods"):
            level_statistics(history, mask, windows)


def test_level_statistics_excludes_global_padding_from_coverage_and_recency():
    available = torch.tensor([[False, False, True, True]])
    observed = torch.tensor([[False, False, True, True]])
    base = torch.tensor([[0.0, 0.0, 0.0, 2.0]])
    changed = torch.tensor([[999.0, 777.0, 0.0, 2.0]])
    expected = level_statistics(base, observed, available=available)
    actual = level_statistics(changed, observed, available=available)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 4] == 1  # two observed cells / two real calendar positions
    assert actual[0, 6] == 0  # last sale is on the last real calendar position
    no_observations = level_statistics(
        torch.zeros_like(base), torch.zeros_like(observed), available=available
    )
    assert no_observations[0, 4] == 0
    first_day_sale = level_statistics(
        torch.tensor([[0.0, 0.0, 2.0, 0.0]]), observed, available=available
    )
    assert first_day_sale[0, 6] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="left-padded calendar suffix"):
        level_statistics(base, observed, available=torch.tensor([[True, False, True, True]]))
    with pytest.raises(ValueError, match="outside|left-padded"):
        level_statistics(base, observed, available=torch.tensor([[False, False, True, False]]))
    with pytest.raises(ValueError, match="at least one available"):
        level_statistics(base, torch.zeros_like(observed), available=torch.zeros_like(available))
    with pytest.raises(ValueError, match="matching shape"):
        level_statistics(base, observed, available=available[:, :-1])


def test_sft_global_padding_cannot_change_outputs_or_gradients():
    torch.manual_seed(7)
    model = SFTModel(
        d_model=4,
        encoder_days=14,
        n_attributes=0,
        n_series=1,
        n_products=0,
        n_stores=0,
        n_categories=0,
        n_known=1,
        horizon=2,
        n_events=0,
        event_max_support=1,
        n_heads=1,
        trunk_stages=1,
        decoder_layers=1,
        history_attention_layers=1,
        demand_dropout=0.0,
        excess_floor=1e-6,
        aggregate_history=False,
        n_commercial=1,
        commercial_attention_heads=1,
        patch_size=7,
        calendar_availability=True,
    ).eval()
    with torch.no_grad():
        model.occurrence.weight.fill_(0.1)
        model.size.weight.fill_(0.1)
    available = torch.tensor([[False] * 7 + [True] * 7])
    observed = available.clone()

    def arguments(history):
        return dict(
            history=history,
            observed=observed,
            history_available=available,
            attributes=torch.empty(1, 0),
            series_index=torch.tensor([0]),
            known_future=torch.ones(1, 2, 1),
            commercial=torch.ones(1, 1),
            commercial_mask=torch.ones(1, 1, dtype=torch.bool),
            baseline_logit=torch.zeros(1, 2),
            baseline_log_size=torch.ones(1, 2),
            event_distance=torch.empty(1, 2, 0),
            event_present=torch.empty(1, 2, 0, dtype=torch.bool),
        )

    def run(prefix: float):
        history = torch.tensor([[prefix] * 7 + [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 1.0]], requires_grad=True)
        prediction = model(**arguments(history))
        loss = prediction.logit.sum() + prediction.excess.sum()
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        output = (prediction.logit.detach().clone(), prediction.excess.detach().clone())
        history_gradient = history.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        return output, gradients, history_gradient

    expected, expected_gradients, expected_history_gradient = run(0.0)
    actual, actual_gradients, actual_history_gradient = run(999.0)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
    assert expected_gradients.keys() == actual_gradients.keys()
    for name in expected_gradients:
        torch.testing.assert_close(
            expected_gradients[name], actual_gradients[name], rtol=0, atol=1e-6
        )
    assert not expected_history_gradient[:, :7].any()
    assert not actual_history_gradient[:, :7].any()

    missing = arguments(torch.zeros(1, 14))
    del missing["history_available"]
    with pytest.raises(ValueError, match="requires history_available"):
        model(**missing)
    model.calendar_availability = False
    with pytest.raises(ValueError, match="history_available requires"):
        model(**arguments(torch.zeros(1, 14)))


def test_sft_passes_partial_week_masks_to_every_consumer(monkeypatch):
    import forecast.models.mft.sft as subject

    model = SFTModel(
        d_model=4,
        encoder_days=14,
        n_attributes=0,
        n_series=1,
        n_products=0,
        n_stores=0,
        n_categories=0,
        n_known=1,
        horizon=2,
        n_events=0,
        event_max_support=1,
        n_heads=1,
        trunk_stages=4,
        decoder_layers=4,
        history_attention_layers=4,
        demand_dropout=0.0,
        excess_floor=1e-6,
        aggregate_history=True,
        n_commercial=0,
        patch_size=7,
        calendar_availability=True,
    ).eval()
    available = torch.tensor([[False] * 5 + [True] * 9])
    observed = available.clone()
    recorded = {"statistics": [], "demand": [], "aggregate": [], "trunk": [], "decoder": []}
    original_statistics = subject.level_statistics

    def statistics(history, mask, windows=(7, 28, 56), available=None):
        recorded["statistics"].append(available.clone())
        return original_statistics(history, mask, windows, available)

    monkeypatch.setattr(subject, "level_statistics", statistics)
    hooks = [
        model.demand.register_forward_pre_hook(
            lambda _module, _args, kwargs: recorded["demand"].append(
                kwargs["position_mask"].clone()
            ),
            with_kwargs=True,
        ),
        model.aggregate.register_forward_pre_hook(
            lambda _module, _args, kwargs: recorded["aggregate"].append(
                kwargs["position_mask"].clone()
            ),
            with_kwargs=True,
        ),
        model.trunk.register_forward_pre_hook(
            lambda _module, _args, kwargs: recorded["trunk"].append(
                kwargs["time_mask"].clone()
            ),
            with_kwargs=True,
        ),
        model.decoder.register_forward_pre_hook(
            lambda _module, _args, kwargs: recorded["decoder"].append(
                kwargs["memory_mask"].clone()
            ),
            with_kwargs=True,
        ),
    ]
    history = torch.arange(14, dtype=torch.float32).reshape(1, 14)
    model(
        history=history,
        observed=observed,
        history_available=available,
        attributes=torch.empty(1, 0),
        series_index=torch.tensor([0]),
        known_future=torch.ones(1, 2, 1),
        commercial=torch.empty(1, 0),
        commercial_mask=torch.empty(1, 0, dtype=torch.bool),
        baseline_logit=torch.zeros(1, 2),
        baseline_log_size=torch.ones(1, 2),
        event_distance=torch.empty(1, 2, 0),
        event_present=torch.empty(1, 2, 0, dtype=torch.bool),
        aggregate_history=history,
        aggregate_observed=observed,
    )
    for hook in hooks:
        hook.remove()
    weekly = available.reshape(1, 2, 7).any(-1)
    memory = weekly[:, None, :].expand(-1, model.n_branches, -1).reshape(1, -1)
    assert len(recorded["statistics"]) == 2
    assert all(torch.equal(mask, available) for mask in recorded["statistics"])
    assert torch.equal(recorded["demand"][0], available)
    assert torch.equal(recorded["aggregate"][0], available)
    assert torch.equal(recorded["trunk"][0], weekly)
    assert torch.equal(recorded["decoder"][0], memory)


@pytest.mark.parametrize("fault", [0.5, float("nan")])
def test_sft_refuses_nonbinary_masks_at_every_public_mask_boundary(fault):
    history = torch.ones(1, 4)
    invalid = torch.ones_like(history)
    invalid[0, 0] = fault
    with pytest.raises(ValueError, match="binary"):
        level_statistics(history, invalid)

    prediction = ShiftedHurdlePrediction(
        torch.zeros(1, 4), torch.ones(1, 4), torch.ones(1, 4)
    )
    with pytest.raises(ValueError, match="binary"):
        prediction.nll(history, invalid)

    model = SFTModel(
        d_model=4,
        encoder_days=4,
        n_attributes=0,
        n_series=1,
        n_products=0,
        n_stores=0,
        n_categories=0,
        n_known=1,
        horizon=1,
        n_events=1,
        event_max_support=1,
        n_heads=1,
        trunk_stages=1,
        decoder_layers=1,
        history_attention_layers=1,
        demand_dropout=0.0,
        excess_floor=1e-6,
        aggregate_history=False,
        n_commercial=0,
        patch_size=2,
    )
    present = torch.ones(1, 1, 1)
    present[0, 0, 0] = fault
    with pytest.raises(ValueError, match="binary"):
        model.event_residual(torch.zeros_like(present), present, None)


def test_shifted_hurdle_nll_normalizes_and_mean_is_exact():
    # k=1, lambda=2 gives P(Y=y>0)=p*(1/3)*(2/3)**(y-1).
    y = torch.arange(200, dtype=torch.double).reshape(1, -1)
    logit = torch.full_like(y, math.log(0.2 / 0.8), requires_grad=True)
    excess = torch.full_like(y, 2.0, requires_grad=True)
    distribution = ShiftedHurdlePrediction(logit, excess, torch.ones_like(y))
    probability = (-distribution.nll(y, torch.ones_like(y))).exp()
    torch.testing.assert_close(probability.sum(), torch.tensor(1.0, dtype=torch.double))
    torch.testing.assert_close((probability * y).sum(), distribution.mean()[0, 0])
    assert distribution.mean()[0, 0].item() == pytest.approx(0.6)
    distribution.mean().sum().backward()
    assert excess.grad is not None and logit.grad is not None
    assert excess.grad.min() > 0 and logit.grad.min() > 0
    assert distribution.nll(y, torch.zeros_like(y)).sum() == 0


@pytest.mark.parametrize("fault", ["fraction", "negative", "nan", "shape", "excess", "k"])
def test_shifted_hurdle_refuses_invalid_targets_and_parameters(fault):
    target, excess, k = torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2)
    if fault == "fraction":
        target[0, 0] = 0.5
    elif fault == "negative":
        target[0, 0] = -1
    elif fault == "nan":
        target[0, 0] = math.nan
    elif fault == "shape":
        target = torch.ones(1, 1)
    elif fault == "excess":
        excess[0, 0] = 0
    else:
        k[0, 0] = -1
    message = (
        "target, mask and prediction shapes"
        if fault == "shape"
        else "finite parameters"
        if fault in {"excess", "k"}
        else "finite nonnegative integers"
    )
    with pytest.raises(ValueError, match=message):
        ShiftedHurdlePrediction(torch.zeros(1, 2), excess, k).nll(target, torch.ones_like(target))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("head_kind", "unknown", "unsupported SFT count distribution"),
        ("logit", torch.full((1, 2), math.nan), "finite parameters"),
        ("logit", torch.full((1, 2), math.inf), "finite parameters"),
        ("excess", torch.full((1, 2), math.nan), "finite parameters"),
        ("excess", torch.full((1, 2), math.inf), "finite parameters"),
        ("dispersion", torch.full((1, 2), math.nan), "finite parameters"),
        ("dispersion", torch.full((1, 2), math.inf), "finite parameters"),
        ("excess", torch.ones(1, 1), "finite parameters"),
        ("dispersion", torch.ones(1, 1), "finite parameters"),
        ("excess", torch.ones(1, 2, dtype=torch.int64), "unsafe mean precision"),
        ("excess", torch.ones(1, 2, dtype=torch.float16), "unsafe mean precision"),
        (
            "excess",
            torch.full((1, 2), torch.finfo(torch.float64).tiny ** 0.5 / 2, dtype=torch.float64),
            "unsafe mean precision",
        ),
        ("dispersion", torch.ones(1, 2, dtype=torch.int64), "dispersion must use"),
        ("dispersion", torch.ones(1, 2, dtype=torch.float16), "dispersion must use"),
    ],
)
def test_shifted_hurdle_parameter_domain_at_mean_and_likelihood(field, value, message):
    parameters = dict(logit=torch.zeros(1, 2), excess=torch.ones(1, 2), dispersion=torch.ones(1, 2))
    prediction = ShiftedHurdlePrediction(**dict(parameters, **{field: value}))
    target = torch.ones(1, 2)
    for call in (prediction.mean, lambda: prediction.nll(target, torch.ones_like(target))):
        with pytest.raises(ValueError, match=message):
            call()


def test_complete_model_event_support_and_day_locality():
    torch.set_num_threads(1)
    arguments: dict[str, Any] = dict(
        d_model=16,
        encoder_days=28,
        n_attributes=2,
        n_series=2,
        n_products=2,
        n_stores=1,
        n_categories=1,
        n_known=3,
        horizon=7,
        n_events=2,
        event_max_support=14,
        n_heads=4,
        trunk_stages=1,
        decoder_layers=1,
        history_attention_layers=1,
        demand_dropout=0.1,
        excess_floor=1e-6,
    )
    with pytest.raises(ValueError, match="SFT dimensions"):
        SFTModel(**dict(arguments, n_series=0))
    pooled = SFTModel(**dict(arguments, n_categories=0))
    assert pooled.calendar is not None and pooled.calendar.series_deviation is None
    with pytest.raises(ValueError, match="invalid SFT event"):
        SFTModel(**dict(arguments, n_events=-1))
    for invalid in (
        dict(encoder_days=27),
        dict(head_kind="unknown"),
        dict(excess_floor=0),
        dict(excess_floor=0.1),
    ):
        with pytest.raises(ValueError, match="invalid SFT event"):
            SFTModel(**dict(arguments, **invalid))
    model = SFTModel(**arguments).eval()
    assert model.calendar is not None
    inputs = dict(
        history=torch.ones(2, 28),
        observed=torch.ones(2, 28, dtype=torch.bool),
        attributes=torch.eye(2),
        series_index=torch.arange(2),
        product_index=torch.arange(2),
        store_index=torch.zeros(2, dtype=torch.long),
        category_index=torch.zeros(2, dtype=torch.long),
        known_future=torch.ones(2, 7, 3),
        commercial=torch.ones(2, 3),
        commercial_mask=torch.ones(2, 3, dtype=torch.bool),
        baseline_logit=torch.zeros(2, 7),
        baseline_log_size=torch.ones(2, 7),
        event_distance=torch.zeros(2, 7, 2),
        event_present=torch.zeros(2, 7, 2),
        aggregate_history=torch.ones(2, 28),
        aggregate_observed=torch.ones(2, 28),
    )
    with torch.no_grad():
        model.dispersion_residual.weight[:, 0].copy_(torch.tensor([0.0, 1.0]))
    assert not torch.equal(model(**inputs).dispersion[0], model(**inputs).dispersion[1])
    for name in ("product_index", "store_index"):
        with pytest.raises(ValueError, match="product/store indices are required"):
            model(**dict(inputs, **{name: None}))
    for name in ("aggregate_history", "aggregate_observed"):
        with pytest.raises(ValueError, match="aggregate branch requires"):
            model(**dict(inputs, **{name: None}))
    with pytest.raises(ValueError, match="category_index is required"):
        model.event_residual(inputs["event_distance"], inputs["event_present"], None)
    without_commercial = SFTModel(**dict(arguments, n_commercial=0))
    for commercial, commercial_mask in (
        (inputs["commercial"], inputs["commercial_mask"]),
        (torch.empty(2, 0), inputs["commercial_mask"]),
        (inputs["commercial"], torch.empty(2, 0, dtype=torch.bool)),
    ):
        with pytest.raises(ValueError, match="commercial inputs supplied to a disabled branch"):
            without_commercial(
                **dict(inputs, commercial=commercial, commercial_mask=commercial_mask)
            )
    for name in ("series_index", "product_index", "store_index", "category_index"):
        for bad in (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(2),
            torch.full((2,), 99, dtype=torch.long),
            torch.zeros(2, device="meta", dtype=torch.long),
        ):
            with pytest.raises(ValueError, match="series_index"):
                model(**dict(inputs, **{name: bad}))
    # Learning a shared kernel must not activate an event whose projection saw no data.
    optimizer = torch.optim.SGD(model.calendar.parameters(), lr=0.01)
    seen = torch.zeros_like(inputs["event_present"])
    seen[..., 0] = 1
    response = model.event_residual(inputs["event_distance"], seen, inputs["category_index"])
    (-response.sum()).backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    unseen = torch.zeros_like(seen)
    unseen[..., 1] = 1
    with torch.no_grad():
        model.calendar.shared.add_(1.0)
    assert model.calendar.project.weight[:, 0].abs().sum() > 0
    assert not model.calendar.project.weight[:, 1].any()
    assert not model.event_residual(
        inputs["event_distance"], unseen, inputs["category_index"]
    ).any()
    with torch.no_grad():
        # Deliberately nonzero all event parameters: support cannot depend on initialization.
        for parameter in model.calendar.parameters():
            parameter.fill_(0.1)
        model.occurrence.weight.fill_(0.1)
        baseline = model(**inputs).mean()
        unsupported = dict(
            inputs, event_distance=torch.full((2, 7, 2), 15.0), event_present=torch.ones(2, 7, 2)
        )
        torch.testing.assert_close(model(**unsupported).mean(), baseline, rtol=0, atol=0)
        present = inputs["event_present"].clone()
        present[:, 0, 0] = 1
        changed = model(**dict(inputs, event_present=present)).mean()
        assert not torch.equal(changed[:, 0], baseline[:, 0])
        torch.testing.assert_close(changed[:, 1:], baseline[:, 1:], rtol=0, atol=0)
    assert model.event_penalty() > 0
    with pytest.raises(ValueError, match="at least one"):
        model(**dict(inputs, baseline_log_size=-torch.ones(2, 7)))
    with pytest.raises(ValueError, match="shapes"):
        model(**dict(inputs, baseline_logit=torch.zeros(2, 1)))
    with pytest.raises(ValueError, match="features must be finite"):
        model(**dict(inputs, known_future=torch.full((2, 7, 3), float("nan"))))
    with pytest.raises(ValueError, match="matching"):
        model.event_residual(torch.zeros(2, 7), torch.zeros(2, 7), inputs["category_index"])
    with pytest.raises(ValueError, match="distances must be finite"):
        model.event_residual(
            torch.full((2, 7, 2), float("nan")), inputs["event_present"], inputs["category_index"]
        )
    with torch.no_grad():
        model.size.bias.fill_(1000)
    with pytest.raises(ValueError, match="nonfinite SFT predictors"):
        model(**inputs)
    eventless = SFTModel(**dict(arguments, n_events=0))
    with pytest.raises(ValueError, match="eventless model"):
        eventless.event_residual(
            inputs["event_distance"], inputs["event_present"], inputs["category_index"]
        )
    empty = torch.zeros(2, 7, 0)
    assert not eventless.event_residual(empty, empty, None).any()
    with pytest.raises(ValueError, match="category_index"):
        eventless.event_residual(empty, empty, torch.zeros(1, dtype=torch.long))
    assert not eventless.event_residual(empty, empty, inputs["category_index"]).any()
    assert eventless.event_penalty() == 0


@pytest.mark.parametrize("head", ["negbin", "shifted_hurdle_negbin"])
@pytest.mark.parametrize("bias", [-1000.0, 1000.0])
def test_sft_exponential_parameter_boundary(head, bias):
    model = SFTModel(
        d_model=16,
        encoder_days=4,
        n_attributes=0,
        n_series=1,
        n_products=0,
        n_stores=0,
        n_categories=0,
        n_known=1,
        horizon=1,
        n_events=0,
        event_max_support=1,
        n_heads=4,
        trunk_stages=1,
        decoder_layers=1,
        history_attention_layers=1,
        demand_dropout=0.0,
        excess_floor=1e-6,
        aggregate_history=False,
        n_commercial=0,
        patch_size=2,
        head_kind=head,
    ).eval()
    inputs = dict(
        history=torch.ones(1, 4),
        observed=torch.ones(1, 4, dtype=torch.bool),
        attributes=torch.empty(1, 0),
        series_index=torch.tensor([0]),
        known_future=torch.ones(1, 1, 1),
        commercial=torch.empty(1, 0),
        commercial_mask=torch.empty(1, 0, dtype=torch.bool),
        baseline_logit=torch.zeros(1, 1),
        baseline_log_size=torch.ones(1, 1),
        event_distance=torch.empty(1, 1, 0),
        event_present=torch.empty(1, 1, 0, dtype=torch.bool),
    )
    with torch.no_grad():
        model.size.bias.fill_(bias)
    with pytest.raises(ValueError, match="SFT predictors"):
        model(**inputs).mean()


@pytest.mark.parametrize("logit_value,size_value", [(-100.0, 100.0), (-710.0, 709.0)])
def test_sft_rare_occurrence_mean(logit_value, size_value):
    logit = torch.tensor([[logit_value]], dtype=torch.float32, requires_grad=True)
    size = torch.tensor([[size_value]], dtype=torch.float64, requires_grad=True)
    prediction = ShiftedHurdlePrediction(logit, size.exp(), torch.ones_like(size))
    mean = prediction.mean()
    expected = math.exp(logit_value + size_value)
    assert mean.item() == pytest.approx(expected, rel=1e-12)
    mean.sum().backward()
    assert logit.grad is not None and size.grad is not None
    assert logit.grad.item() == pytest.approx(expected, rel=1e-6)
    assert size.grad.item() == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("head", ["negbin", "shifted_hurdle_negbin"])
def test_sft_dispersion_domain(dtype, head):
    target = torch.tensor([[2]], dtype=dtype)
    for value in (math.exp(-90), 1e20):
        k = torch.tensor([[value]], dtype=dtype, requires_grad=True)
        prediction = ShiftedHurdlePrediction(torch.zeros_like(k), torch.ones_like(k), k, head)
        for call in (prediction.mean, lambda p=prediction: p.nll(target, torch.ones_like(target))):
            with pytest.raises(ValueError, match="dispersion"):
                call()
    for log_k in (-4.0, 0.0, 4.0):
        theta = torch.tensor([[log_k]], dtype=torch.float64, requires_grad=True)
        k = theta.exp()
        prediction = ShiftedHurdlePrediction(torch.zeros_like(k), torch.ones_like(k), k, head)
        loss = prediction.nll(target, torch.ones_like(target)).sum()
        loss.backward()
        count = 2 if head == "negbin" else 1
        expected = (
            -(
                torch.special.digamma(k + count)
                - torch.special.digamma(k)
                + k.log()
                - (k + 1).log()
                + 1
                - (count + k) / (1 + k)
            )
            * k
        )
        assert theta.grad is not None
        torch.testing.assert_close(theta.grad, expected, rtol=1e-10, atol=1e-12)
