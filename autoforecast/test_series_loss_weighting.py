from __future__ import annotations

import pytest
import torch

from autoforecast.series_loss_weighting import series_loss_weights, weighted_series_mean


def test_uniform_weighting_preserves_the_original_reduction() -> None:
    loss = torch.tensor([2.0, 6.0])
    weights = series_loss_weights("uniform", loss)
    assert weighted_series_mean(loss, 4.0, weights).item() == pytest.approx(2.0)
    assert weights.values.tolist() == [1.0, 1.0]


def test_revenue_weighting_uses_governed_precomputed_series_weights() -> None:
    loss = torch.tensor([2.0, 6.0])
    weights = series_loss_weights("revenue", loss, revenue_weight=torch.tensor([0.5, 1.5]))
    assert weighted_series_mean(loss, 4.0, weights).item() == pytest.approx(2.5)
    assert not weights.values.requires_grad


def test_output_gradient_weighting_is_bounded_detached_and_one_probe_only() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
    output = parameter.square()
    loss = output.square()

    weights = series_loss_weights("output_gradient", loss, distribution_outputs=(output,))

    assert parameter.grad is None
    assert not weights.values.requires_grad
    assert weights.values[1] > weights.values[0]
    assert weights.values.mean().item() == pytest.approx(1.0)
    assert 0.25 <= weights.values.min() <= weights.values.max() <= 4.0
    weighted_series_mean(loss, 2.0, weights).backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"mode": "unknown"},
        {"mode": "revenue"},
        {"mode": "output_gradient"},
    ),
)
def test_weighting_modes_fail_closed_when_required_evidence_is_missing(kwargs) -> None:
    loss = torch.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        series_loss_weights(per_series_loss=loss, **kwargs)


@pytest.mark.parametrize(
    "loss",
    (
        torch.tensor([-1.0]),
        torch.tensor([float("nan")]),
        torch.tensor([[1.0]]),
        torch.tensor([1]),
    ),
)
def test_per_series_loss_rejects_invalid_values_and_shapes(loss) -> None:
    with pytest.raises(ValueError, match="per-series loss"):
        series_loss_weights("uniform", loss)


@pytest.mark.parametrize(
    "revenue_weight",
    (
        torch.tensor([-1.0, 1.0]),
        torch.tensor([float("inf"), 1.0]),
        torch.tensor([1.0]),
        torch.tensor([1, 1]),
    ),
)
def test_revenue_weighting_rejects_invalid_values_and_shapes(revenue_weight) -> None:
    with pytest.raises(ValueError, match="revenue weight"):
        series_loss_weights(
            "revenue",
            torch.tensor([1.0, 2.0]),
            revenue_weight=revenue_weight,
        )


@pytest.mark.parametrize("values", ([], [0.0, 0.0]))
def test_output_gradient_weighting_fails_closed_without_an_informative_row(values) -> None:
    output = torch.tensor(values, requires_grad=True)
    with pytest.raises(ValueError, match="undefined"):
        series_loss_weights(
            "output_gradient",
            output.square(),
            distribution_outputs=(output,),
        )


def test_weighted_series_mean_rejects_misaligned_weights() -> None:
    weights = series_loss_weights("uniform", torch.tensor([1.0]))
    with pytest.raises(ValueError, match="same batch shape"):
        weighted_series_mean(torch.tensor([1.0, 2.0]), 2.0, weights)
