"""Covariate dropout and the price monotonicity penalty -- the two constraints on what the model
is allowed to learn from.

They sit together because both answer the same question: the model will fit anything the data
supports, and some of what the data supports is not true of the world it will forecast in.

COVARIATE DROPOUT DROPS THE MASK, NOT THE VALUE, and that distinction is the whole of it. Setting
a dropped price to zero teaches "this product is free", which is a statement about the world.
Setting its availability flag to False teaches "the price is unknown here", which is the statement
intended -- and it is the same statement the panel already makes for the 19% of M5 cells that
genuinely have no price. A model regularised by zero-filling learns a different lesson from the
one that was meant, and learns it confidently.

Price coverage can vary between training and deployment. A model trained with nearly
complete prices may degrade when those prices are unavailable. Covariate dropout exposes
that missing-feature condition during training.

THE MONOTONICITY PENALTY IS SOFT, and it is worth being honest about why. A hard constraint --
sign-restricted weights on a dedicated price path -- would guarantee the shape, but price reaches
the head through channel attention, the trunk and the decoder's cross-attention, so restricting it
would mean restricting all three and losing the interactions that make the branch worth having.
The penalty instead measures the violation where it matters, on the model's actual output, and
prices it. That yields a model that is monotone almost everywhere rather than provably everywhere,
and `violation_rate` reports which.

**A forecast that says raising the price raises demand is the one output a commercial user will
check by hand**, and the one that ends the conversation about whether the model can be trusted.
"""

from __future__ import annotations

import torch
from forecast.models.mft.branches import _binary_mask
from torch import Tensor

__all__ = ["RegularisationError", "drop_covariates", "monotonicity_penalty", "price_violation_rate"]


class RegularisationError(ValueError):
    """A constraint was requested that cannot be applied to the tensors given."""


def drop_covariates(
    values: Tensor,
    mask: Tensor,
    rate: float,
    *,
    generator: torch.Generator | None = None,
    allow_empty: bool = False,
) -> tuple[Tensor, Tensor]:
    """Mark a random share of AVAILABLE covariate channels unavailable, per cell.

    Args:
        values: `[batch, n_channels]` covariate values.
        mask: `[batch, n_channels]` True where the channel is available.
        rate: Share of available channels to hide, in `[0, 1)`.
        generator: Seeded, for reproducibility.
        allow_empty: Permit hiding the last channel only for callers with a tested missing-row
            representation. Default False keeps one channel for attention that requires an
            observed token.

    Returns:
        `(values, mask)` -- values UNCHANGED, mask with some True entries turned False. The values
        are deliberately left alone: they are never read without their flag, and rewriting them
        would make the dropout visible to anything that forgot to check.

    Raises:
        RegularisationError: a rate outside `[0, 1)`, or shapes that disagree. Rate 1.0 is
            excluded rather than clamped -- it would hide every channel of every cell, and
            attention over an empty key set is undefined.
    """
    if not 0.0 <= rate < 1.0:
        raise RegularisationError(
            f"rate must lie in [0,1), got {rate}. At 1.0 every channel of every cell is hidden "
            "and attention over an empty key set is undefined"
        )
    if values.shape != mask.shape:
        raise RegularisationError(
            f"values {tuple(values.shape)} and mask {tuple(mask.shape)} must describe the same "
            "channels"
        )
    mask = _binary_mask(mask, values, "covariate availability mask", error_type=RegularisationError)
    if rate == 0.0:
        return values, mask

    keep = torch.rand(mask.shape, device=mask.device, generator=generator) >= rate
    dropped = mask & keep
    # Attention requiring an observed token retains a channel; other callers opt into empty rows.
    empty = dropped.any(dim=-1).logical_not() & mask.any(dim=-1)
    if not allow_empty and bool(empty.any()):
        first_available = mask[empty].float().argmax(dim=-1)
        restore = torch.zeros_like(dropped[empty])
        restore[torch.arange(restore.shape[0], device=restore.device), first_available] = True
        dropped[empty] = restore
    return values, dropped


def monotonicity_penalty(
    expected_low: Tensor,
    expected_high: Tensor,
) -> Tensor:
    """Penalise demand that RISES with price. Returns a scalar.

    Args:
        expected_low: `[...]` expected demand at the observed price.
        expected_high: `[...]` expected demand at a raised price, same cells, everything else
            held equal.

    Returns:
        Mean hinge `max(0, high - low)`. Zero when demand is non-increasing in price, which is the
        constraint; positive by exactly the amount it is violated.

    Raises:
        RegularisationError: the two tensors describe different cells, which would compare a
            cell's raised price against another cell's demand and penalise the difference between
            two products.
    """
    if expected_low.shape != expected_high.shape:
        raise RegularisationError(
            f"low {tuple(expected_low.shape)} and high {tuple(expected_high.shape)} must describe "
            "the same cells; otherwise this compares one product against another"
        )
    return torch.clamp(expected_high - expected_low, min=0.0).mean()


def price_violation_rate(expected_low: Tensor, expected_high: Tensor) -> float:
    """Share of cells where demand rose with price. The number to report, not the penalty.

    The penalty is a training signal whose magnitude depends on the demand scale; this is the
    quantity a reader can interpret. **A model can carry a small penalty and still violate
    monotonicity on a third of its cells**, if each violation is tiny -- and that is a different
    thing to describe to a commercial user than a model that is monotone on 99% of them.
    """
    if expected_low.shape != expected_high.shape:
        raise RegularisationError("shapes must match; see monotonicity_penalty")
    # This is a reporting reduction, not a numerically sensitive accumulator.  MPS does not
    # implement float64 tensors, whereas float32 represents every attainable batch fraction
    # closely and can be transferred to the host scalar by ``float``.
    return float((expected_high > expected_low).float().mean())
