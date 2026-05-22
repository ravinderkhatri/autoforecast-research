"""Tests for covariate dropout and the price monotonicity constraint.

Both exist because the model will fit anything the data supports, and some of what the data
supports is not true of the world it forecasts in.

Run: python3 forecast/models/mft/test_regularisation.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import torch
from forecast.models.mft.regularisation import (
    RegularisationError,
    drop_covariates,
    monotonicity_penalty,
    price_violation_rate,
)


def test_dropout_hides_availability_and_never_touches_the_value() -> None:
    """THE distinction. Zeroing a dropped price teaches "this product is free"; clearing its flag
    teaches "the price is unknown here", which is the lesson intended and the same one the panel
    already makes for the 19% of M5 cells with no price."""
    values = torch.rand(500, 3) * 10 + 1
    mask = torch.ones(500, 3, dtype=torch.bool)
    out_values, out_mask = drop_covariates(
        values, mask, 0.5, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(out_values, values), "a value was rewritten; only the mask may change"
    assert int((~out_mask).sum()) > 0, "nothing was dropped at rate 0.5"
    assert not bool((out_mask & ~mask).any()), "dropout made an unavailable channel available"


def test_a_cell_never_loses_every_channel() -> None:
    """Dropping the last one leaves the commercial branch attending over nothing, which returns
    NaN -- the failure that cost a night of the ladder when the demand branch hit it."""
    mask = torch.ones(400, 3, dtype=torch.bool)
    _, out = drop_covariates(
        torch.rand(400, 3), mask, 0.95, generator=torch.Generator().manual_seed(1)
    )
    assert bool(out.any(dim=-1).all()), "a cell was left with no available channel"


def test_a_cell_that_already_had_nothing_is_left_alone() -> None:
    """The restore path must not invent availability the panel never claimed."""
    mask = torch.zeros(10, 3, dtype=torch.bool)
    _, out = drop_covariates(
        torch.rand(10, 3), mask, 0.5, generator=torch.Generator().manual_seed(2)
    )
    assert not bool(out.any()), "dropout created a channel out of nothing"


def test_rate_zero_is_the_identity() -> None:
    values, mask = torch.rand(20, 3), torch.ones(20, 3, dtype=torch.bool)
    out_v, out_m = drop_covariates(values, mask, 0.0)
    assert torch.equal(out_v, values) and torch.equal(out_m, mask)


def test_a_rate_of_one_is_refused() -> None:
    """It would hide every channel of every cell, and attention over an empty key set is
    undefined."""
    try:
        drop_covariates(torch.rand(4, 2), torch.ones(4, 2, dtype=torch.bool), 1.0)
    except RegularisationError as e:
        assert "empty key set is undefined" in str(e)
        return
    raise AssertionError("accepted a dropout rate of 1.0")


def test_mismatched_shapes_are_refused() -> None:
    try:
        drop_covariates(torch.rand(4, 2), torch.ones(4, 3, dtype=torch.bool), 0.2)
    except RegularisationError as e:
        assert "same channels" in str(e)
        return
    raise AssertionError("dropped a mask that describes different channels")


@pytest.mark.parametrize("fault", [0.5, float("nan")])
def test_drop_covariates_refuses_nonbinary_masks_even_when_rate_is_zero(fault: float) -> None:
    values = torch.ones(1, 2)
    mask = torch.ones_like(values)
    mask[0, 0] = fault
    with pytest.raises(RegularisationError, match="binary"):
        drop_covariates(values, mask, 0.0)


def test_drop_covariates_allow_empty_is_explicit_and_preserves_values(monkeypatch) -> None:
    values = torch.tensor([[5., 7.], [3., 9.]])
    mask = torch.tensor([[True, False], [False, False]])
    # Force every eligible channel to drop using an admitted rate, not forbidden rate 1.
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.zeros(shape))
    kept_values, allowed = drop_covariates(values, mask, 0.5, allow_empty=True)
    assert kept_values is values and not allowed.any()
    kept_values, protected = drop_covariates(values, mask, 0.5)
    assert kept_values is values
    torch.testing.assert_close(protected, mask)
    with pytest.raises(RegularisationError, match="rate must lie"):
        drop_covariates(values, mask, 1.0, allow_empty=True)


# ---------------------------------------------------------------- monotonicity


def test_demand_falling_with_price_costs_nothing() -> None:
    """The constraint is one-sided: a price rise that lowers demand is the expected shape."""
    low = torch.tensor([10.0, 5.0, 2.0])
    high = torch.tensor([8.0, 4.0, 1.0])
    assert float(monotonicity_penalty(low, high)) == 0.0
    assert price_violation_rate(low, high) == 0.0


def test_demand_rising_with_price_is_penalised_by_how_much() -> None:
    low = torch.tensor([10.0, 5.0])
    high = torch.tensor([12.0, 5.0])
    assert abs(float(monotonicity_penalty(low, high)) - 1.0) < 1e-6  # (2 + 0) / 2
    assert price_violation_rate(low, high) == 0.5


def test_the_violation_rate_is_not_the_penalty_and_reports_a_different_thing() -> None:
    """A model can carry a tiny penalty and still violate monotonicity on most of its cells, if
    each violation is small -- and that is a different thing to tell a commercial user."""
    low = torch.zeros(100)
    high = torch.full((100,), 1e-4)
    assert float(monotonicity_penalty(low, high)) < 1e-3, "the penalty is negligible"
    assert price_violation_rate(low, high) == 1.0, "yet every cell violates the constraint"


def test_the_violation_rate_runs_on_mps_without_float64() -> None:
    if not torch.backends.mps.is_available():
        return
    low = torch.tensor([1.0, 2.0], device="mps")
    high = torch.tensor([2.0, 1.0], device="mps")
    assert price_violation_rate(low, high) == 0.5


def test_the_penalty_is_differentiable_where_it_bites() -> None:
    """It is a training signal, so it has to carry gradient on the violating cells."""
    low = torch.tensor([1.0, 5.0], requires_grad=True)
    high = torch.tensor([3.0, 1.0], requires_grad=True)
    monotonicity_penalty(low, high).backward()
    assert float(high.grad[0]) > 0, "no gradient on the violating cell"
    assert float(high.grad[1]) == 0.0, "gradient on a cell that satisfies the constraint"


def test_comparing_different_cells_is_refused() -> None:
    """Otherwise this penalises the difference between two products."""
    for call in (
        lambda: monotonicity_penalty(torch.zeros(3), torch.zeros(4)),
        lambda: price_violation_rate(torch.zeros(3), torch.zeros(4)),
    ):
        try:
            call()
        except RegularisationError:
            continue
        raise AssertionError("compared two different sets of cells")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
