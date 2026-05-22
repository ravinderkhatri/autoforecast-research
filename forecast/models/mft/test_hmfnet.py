"""Torch-only structural counterexamples. No synthetic forecast score is evidence."""

import copy
import math

import pytest
import torch
from forecast.models.mft.hmfnet import (
    HMFNetModel,
    MaskedLocalBlock,
    tensor_summary,
)


def model_inputs(*, groups=2, events=0, categorical=False, head="shifted_hurdle_negbin"):
    """Fixed shapes make routing/gradient failures observable, not accuracy estimates."""
    torch.set_num_threads(1)
    torch.manual_seed(5)
    config = dict(
        d_model=8,
        n_heads=2,
        local_blocks=2,
        n_token_features=11,
        n_statistics=14,
        n_attributes=2,
        n_series=4,
        n_known=5,
        n_aggregate_known=3,
        horizon=3,
        identity_cardinalities=(4,),
        categorical_cardinalities=(3,) if categorical else (),
        categorical_width=2,
        n_types=2 if groups else 0,
        n_context_groups=groups,
        n_expert_groups=groups,
        feedback=True,
        n_events=events,
        n_categories=1 if events else 0,
        event_max_support=2,
        event_kernel_layers=1,
        excess_floor=1e-6,
        head_kind=head,
    )
    model = HMFNetModel(**config)

    def memory(n):
        return dict(
            memory_values=torch.rand(n, 6, 11),
            memory_mask=torch.ones(n, 6, dtype=torch.bool),
            memory_categorical=torch.zeros(n, 6, int(categorical), dtype=torch.long),
            memory_categorical_mask=torch.ones(n, 6, int(categorical), dtype=torch.bool),
            memory_bands=torch.tensor([0, 0, 1, 1, 2, 2]),
            memory_regular=torch.tensor([True, True, True, True, True, False]),
            statistics=torch.rand(n, 14),
            quality=torch.ones(n, 3),
        )

    inputs = dict(
        memory=memory(4),
        attributes=torch.eye(2).repeat(2, 1),
        series_index=torch.arange(4),
        identity_indices=torch.arange(4)[:, None],
        known_future=torch.rand(4, 3, 5),
        baseline_logit=torch.full((4, 3), math.log(1 / 3)),
        baseline_log_size=torch.full((4, 3), math.log(4)),
        event_distance=torch.zeros(4, 3, events),
        event_present=torch.zeros(4, 3, events, dtype=torch.bool),
    )
    if events:
        inputs["category_index"] = torch.zeros(4, dtype=torch.long)
    if groups:
        inputs["aggregate"] = dict(
            memory(2),
            known_future=torch.rand(2, 3, 3),
            anchor=torch.full((2, 3), 4.0, dtype=torch.double),
            type_index=torch.arange(2),
            node_ids=torch.arange(2),
        )
        inputs["context_memberships"] = torch.tensor([[0, 1]]).expand(4, -1)
        inputs["expert_memberships"] = torch.tensor([[0, 1]]).expand(4, -1)
    return model, inputs, config


def test_masked_local_block():
    layer = MaskedLocalBlock(1)
    with torch.no_grad():
        layer.conv.weight.copy_(torch.tensor([[[1.0, 2.0, 3.0]]]))
        layer.conv.bias.fill_(0.25)
    value = torch.tensor([[[2.0], [999.0], [6.0]]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    bands, regular = torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.bool)
    result = layer.convolve(value, mask, bands, regular)
    torch.testing.assert_close(result.flatten(), torch.tensor([4.25, 0, 12.25]))
    result.sum().backward()
    assert value.grad[0, 1] == 0
    changed = value.detach().clone()
    changed[0, 1] = float("nan")
    torch.testing.assert_close(layer.convolve(changed, mask, bands, regular), result)
    torch.testing.assert_close(
        layer(changed, mask, bands, regular), layer(value, mask, bands, regular)
    )
    assert layer(value, mask, bands, regular)[0, 1] == 0
    regular[2] = False
    assert layer.convolve(value, mask, bands, regular)[0, 2] == 0
    with pytest.raises(ValueError, match="band"):
        layer.convolve(value, mask, torch.tensor([1, 0, 1]), regular)


@pytest.mark.parametrize(
    "field,value",
    [
        ("values", torch.ones(2, 3)),
        ("mask", torch.ones(1, 3, dtype=torch.bool)),
        ("mask", torch.ones(2, 3)),
        ("bands", torch.zeros(2, dtype=torch.int64)),
        ("bands", torch.zeros(3)),
        ("bands", torch.tensor([-1, 0, 0])),
        ("bands", torch.tensor([0, 0, 3])),
        ("regular", torch.ones(2, dtype=torch.bool)),
        ("regular", torch.ones(3)),
    ],
)
def test_masked_local_block_rejects_structural_inputs(field, value):
    inputs = dict(
        values=torch.ones(2, 3, 1),
        mask=torch.ones(2, 3, dtype=torch.bool),
        bands=torch.zeros(3, dtype=torch.int64),
        regular=torch.ones(3, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="invalid local memory masks or band ordering"):
        MaskedLocalBlock(1).convolve(**dict(inputs, **{field: value}))


def test_hmfnet_forward():
    model, inputs, config = model_inputs(categorical=True)
    output = model(**inputs)
    assert output.mean().shape == (4, 3) and output.mean().dtype == torch.float64
    torch.testing.assert_close(output.mean(), torch.ones(4, 3, dtype=torch.double))
    assert not hasattr(model, "trunk") and not hasattr(model, "demand")
    with torch.no_grad():
        model.size.weight.fill_(0.01)
    inputs["memory"]["memory_mask"][:, 1] = False
    reference = model(**inputs).mean()
    inputs["memory"]["memory_values"][:, 1] = float("nan")
    # Placeholder values are masked before the encoder and before diagnostics.
    torch.testing.assert_close(model(**inputs).mean(), reference)
    inputs["memory"]["memory_mask"][:] = False
    assert torch.isfinite(model(**inputs).mean()).all()
    with pytest.raises(ValueError, match="dimensions"):
        HMFNetModel(**dict(config, n_heads=3))
    for invalid in (
        dict(n_attributes=-1),
        dict(feedback=1),
        dict(n_context_groups=config["n_types"] + 1),
        dict(n_expert_groups=config["n_types"] + 1),
        dict(head_kind="unknown"),
        dict(excess_floor=0),
        dict(excess_floor=0.1),
    ):
        with pytest.raises(ValueError, match="invalid HMFNet dimensions"):
            HMFNetModel(**dict(config, **invalid))
    with pytest.raises(ValueError, match="horizons"):
        model(**dict(inputs, baseline_logit=torch.zeros(1, 3)))
    with pytest.raises(ValueError, match="vocabulary"):
        inputs["memory"]["memory_mask"][:] = True
        inputs["memory"]["memory_categorical"][:] = 99
        model(**inputs)


def test_hmfnet_experts():
    model, inputs, _ = model_inputs()
    output = model(**inputs)
    torch.testing.assert_close(
        model.aggregate_forecasts(inputs["aggregate"])["mean"], output.expert_mean, rtol=0, atol=0
    )
    torch.testing.assert_close(output.expert_mean, inputs["aggregate"]["anchor"], rtol=0, atol=0)
    assert not output.expert_residual.any()
    ((output.expert_mean - 8) ** 2).mean().backward()
    assert model.expert_size.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.expert_size.weight.add_(0.01)
    first = model(**inputs).expert_mean.detach().clone()
    inputs["known_future"] += 100
    inputs["memory"]["memory_values"] += 100
    torch.testing.assert_close(model(**inputs).expert_mean, first, rtol=0, atol=0)
    bad = copy.deepcopy(inputs)
    bad["aggregate"]["anchor"][0, 0] = -1
    with pytest.raises(ValueError, match="anchor"):
        model(**bad)


def test_hmfnet_feedback():
    model, inputs, _ = model_inputs()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        output = model(**inputs)
        loss = (output.mean() - 3).square().mean() + (output.expert_mean - 6).square().mean()
        loss.backward()
        optimizer.step()
    for module in (
        model.token_projection,
        model.expert_size,
        model.feedback_phi,
        model.gate_score,
        model.context_projection,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    normal = model(**inputs)
    zeroed = model(**dict(inputs, zero_expert_residual=True))
    torch.testing.assert_close(normal.gates, zeroed.gates, rtol=0, atol=0)
    assert not torch.equal(normal.mean(), zeroed.mean())
    reordered = dict(inputs, expert_memberships=inputs["expert_memberships"].flip(1))
    torch.testing.assert_close(model(**reordered).mean(), normal.mean())
    inputs["aggregate"]["memory_mask"][:] = False
    null = model(**inputs)
    assert (null.gates[..., -1] == 1).all() and not null.gates[..., :-1].any()
    with pytest.raises(ValueError, match="membership"):
        model(**dict(inputs, expert_memberships=torch.zeros(4, 1, dtype=torch.long)))


def test_hmfnet_event_locality():
    model, inputs, _ = model_inputs(events=1)
    with torch.no_grad():
        model.size.weight.fill_(0.01)
        model.calendar.project.weight.fill_(0.2)
    baseline = model(**inputs)
    inputs["event_present"][:, 1] = True
    event = model(**inputs)
    torch.testing.assert_close(event.expert_mean, baseline.expert_mean, rtol=0, atol=0)
    torch.testing.assert_close(event.mean()[:, [0, 2]], baseline.mean()[:, [0, 2]], rtol=0, atol=0)
    assert not torch.equal(event.mean()[:, 1], baseline.mean()[:, 1])
    inputs["event_distance"][:, 1] = 3
    torch.testing.assert_close(model(**inputs).mean(), baseline.mean(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="index"):
        model(**dict(inputs, category_index=torch.tensor([99])))
    assert model.event_penalty() > 0


@pytest.mark.parametrize("head", ["negbin", "shifted_hurdle_negbin"])
def test_hmfnet_distribution(head):
    model, inputs, _ = model_inputs(groups=0, head=head)
    output = model(**inputs)
    if head == "negbin":
        torch.testing.assert_close(output.mean(), output.excess)
    else:
        torch.testing.assert_close(
            output.mean(), output.logit.double().sigmoid() * (1 + output.excess)
        )
    loss = output.nll(torch.ones(4, 3), torch.ones(4, 3, dtype=torch.bool)).mean()
    loss.backward()
    assert torch.isfinite(loss) and model.event_penalty() == 0
    with pytest.raises(ValueError, match="counts|integers"):
        output.nll(-torch.ones(4, 3), torch.ones(4, 3, dtype=torch.bool))


def test_hmfnet_diagnostics():
    model, inputs, _ = model_inputs()
    model(**inputs)
    assert set(model.last_diagnostics) >= {
        "encoded_memory",
        "decoded_horizon",
        "expert_means",
        "feedback",
        "gates",
        "expected_counts",
    }
    assert all(v["finite"] for v in model.last_diagnostics.values())
    assert tensor_summary(torch.empty(0))["minimum"] is None
    with pytest.raises(ValueError, match="nonfinite"):
        tensor_summary(torch.tensor([float("nan")]))


def test_hmfnet_boundary_guards():
    """Exercise each rejection directly, before any redundant downstream check."""
    model, inputs, _ = model_inputs(categorical=True)
    memory = inputs["memory"]
    with pytest.raises(ValueError, match="token shape"):
        model.encode(dict(memory, memory_categorical_mask=torch.ones(4, 6, 1)))
    for invalid in (
        dict(memory_values=torch.ones(4, 6)),
        dict(memory_values=torch.ones(4, 6, 10)),
        dict(memory_mask=torch.ones(1, 6, dtype=torch.bool)),
        dict(memory_mask=torch.ones(4, 6)),
        dict(memory_categorical=torch.zeros(4, 6, 2, dtype=torch.long)),
        dict(memory_categorical=torch.zeros(4, 6, 1)),
        dict(memory_categorical_mask=torch.ones(1, 6, 1, dtype=torch.bool)),
    ):
        with pytest.raises(ValueError, match="invalid HMFNet token shape or categorical mask"):
            model.encode(dict(memory, **invalid))
    bad = copy.deepcopy(memory)
    bad["memory_values"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="admitted HMFNet memory"):
        model.encode(bad)
    aggregate = inputs["aggregate"]
    model._aggregate_cache = model.aggregate_forecasts(aggregate)
    with pytest.raises(ValueError, match="inference-only"):
        model.aggregate_forecasts(aggregate)
    model._aggregate_cache = None
    ungrouped, _, _ = model_inputs(groups=0)
    with pytest.raises(ValueError, match="without aggregate types"):
        ungrouped.aggregate_forecasts(aggregate)
    with pytest.raises(ValueError, match="type indices"):
        model.aggregate_forecasts(dict(aggregate, type_index=torch.tensor([-1, 0])))
    with torch.no_grad():
        model.expert_size.bias.fill_(1000)
    with pytest.raises(ValueError, match="safe mean domain"):
        model.aggregate_forecasts(aggregate)
    with torch.no_grad():
        model.expert_size.bias.zero_()
    with pytest.raises(ValueError, match="nonfinite HMFNet input"):
        model(**dict(inputs, baseline_logit=torch.full((4, 3), float("nan"))))
    for memberships in (torch.zeros(4, dtype=torch.long), torch.full((4, 2), -1)):
        with pytest.raises(ValueError, match="context membership"):
            model(**dict(inputs, context_memberships=memberships))
    with pytest.raises(ValueError, match="context membership"):
        model(**dict(inputs, context_memberships=torch.zeros(4, 1, dtype=torch.long)))
    with pytest.raises(ValueError, match="context membership"):
        model(**dict(inputs, context_memberships=None))
    with pytest.raises(ValueError, match="require their membership"):
        model(**dict(inputs, expert_memberships=None))


def test_hmfnet_aggregate_shape_guards():
    """No singleton broadcasting, including the inference cache path."""
    model, inputs, _ = model_inputs()
    aggregate = inputs["aggregate"]
    faults = [
        dict(known_future=aggregate["known_future"][:, :1]),
        dict(known_future=aggregate["known_future"][:1]),
        dict(statistics=aggregate["statistics"][:1]),
        dict(quality=aggregate["quality"][:1]),
        dict(node_ids=torch.tensor([0, 0])),
    ]
    model.eval()
    with torch.no_grad():
        cached = model.aggregate_forecasts(aggregate)
        for cache in (None, cached):
            model._aggregate_cache = cache
            for fault in faults:
                with pytest.raises(ValueError, match="invalid aggregate horizons"):
                    model.aggregate_forecasts(dict(aggregate, **fault))
            with pytest.raises(ValueError, match="nonfinite aggregate input"):
                model.aggregate_forecasts(dict(aggregate, quality=torch.full((2, 3), float("nan"))))
        with pytest.raises(ValueError, match="outside the frozen population"):
            model.aggregate_forecasts(dict(aggregate, node_ids=torch.tensor([0, 2])))
        model._aggregate_cache = None
