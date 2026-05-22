"""Paired PACT campaign inference with an explicit causal-evidence boundary."""

from __future__ import annotations

import numpy as np
import torch
from forecast.models.mft.pact import PACTModelInputs
from forecast.scenario.causal_uplift import CausalUplift, campaign_impact_record
from torch import Tensor, nn

_SCENARIO_FIELDS = frozenset({"future_features"})


def _equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    return left == right


def evaluate_pact_campaign_pair(
    model: nn.Module,
    planned_inputs: PACTModelInputs,
    reference_inputs: PACTModelInputs,
    *,
    causal_uplift: CausalUplift | None = None,
    causal_missing: tuple[str, ...] = (
        "campaign_episode_ids",
        "comparison_population",
        "cross_fitted_nuisance_predictions",
        "declared_confounders",
        "propensity_overlap",
    ),
    require_causal: bool = False,
) -> dict[str, object]:
    """Evaluate rebuilt campaign ON/OFF inputs; never call the result causal by default.

    Scenario descendants must already be rebuilt into ``future_features``.  Every historical,
    static and attention-layout input must remain byte-identical between the two calls.
    """
    if not isinstance(planned_inputs, PACTModelInputs) or not isinstance(
        reference_inputs, PACTModelInputs
    ):
        raise TypeError("PACT campaign evaluation expects two PACTModelInputs values")
    planned_values = planned_inputs.as_kwargs()
    reference_values = reference_inputs.as_kwargs()
    changed = {
        name
        for name in planned_values
        if not _equal(planned_values[name], reference_values[name])
    }
    if not changed or not changed <= _SCENARIO_FIELDS:
        raise ValueError(
            "PACT campaign scenarios must differ only in rebuilt future_features, "
            f"changed={sorted(changed)}"
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            planned = model(**planned_values).mean().detach().cpu().numpy()
            reference = model(**reference_values).mean().detach().cpu().numpy()
    finally:
        model.train(was_training)
    return campaign_impact_record(
        np.asarray(planned),
        np.asarray(reference),
        causal_uplift=causal_uplift,
        causal_missing=causal_missing,
        require_causal=require_causal,
    )


__all__ = ["evaluate_pact_campaign_pair"]
