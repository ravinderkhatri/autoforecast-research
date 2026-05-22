"""Separate causal-uplift gate for campaign forecasts.

The forecasting model's paired ON/OFF response is always reported under its own name.  A causal
number appears only when a registered, cross-fitted identification input passes this boundary.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


class CausalUpliftError(ValueError):
    """A causal result was required but its identification contract was absent or invalid."""


@dataclass(frozen=True)
class CausalDesign:
    """Frozen identification metadata for a cross-fitted AIPW estimate."""

    origin: str
    campaign_start: str
    treatment_unit: str
    comparison_population: str
    confounders: tuple[str, ...]
    overlap_floor: float = 0.05
    estimator: str = "cross_fitted_aipw_v1"
    cross_fitted: bool = True

    def __post_init__(self) -> None:
        try:
            origin = dt.date.fromisoformat(self.origin)
            campaign = dt.date.fromisoformat(self.campaign_start)
        except (TypeError, ValueError) as exc:
            raise CausalUpliftError("causal origin and campaign_start must be ISO dates") from exc
        if origin.isoformat() != self.origin or campaign.isoformat() != self.campaign_start:
            raise CausalUpliftError("causal dates must use canonical YYYY-MM-DD form")
        if origin >= campaign:
            raise CausalUpliftError("causal uplift requires an origin before campaign treatment")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.treatment_unit, self.comparison_population)
        ):
            raise CausalUpliftError("causal treatment and comparison populations must be named")
        if (
            not isinstance(self.confounders, tuple)
            or not self.confounders
            or len(set(self.confounders)) != len(self.confounders)
            or any(not isinstance(value, str) or not value.strip() for value in self.confounders)
        ):
            raise CausalUpliftError("causal design needs unique declared confounders")
        if (
            type(self.overlap_floor) not in (int, float)
            or not math.isfinite(self.overlap_floor)
            or not 0 < self.overlap_floor < 0.5
        ):
            raise CausalUpliftError("causal overlap_floor must lie strictly between 0 and 0.5")
        if self.estimator != "cross_fitted_aipw_v1" or self.cross_fitted is not True:
            raise CausalUpliftError(
                "causal uplift requires the registered cross-fitted AIPW design"
            )


@dataclass(frozen=True)
class CausalUplift:
    status: str
    estimate: float | None
    low: float | None
    high: float | None
    episodes: int
    estimator: str | None
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"identified", "not_identifiable"}:
            raise CausalUpliftError("unknown causal uplift status")
        if self.status == "identified":
            values = (self.estimate, self.low, self.high)
            if (
                any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or self.low > self.high
                or self.episodes < 2
                or self.estimator != "cross_fitted_aipw_v1"
                or self.missing
            ):
                raise CausalUpliftError("identified causal evidence is internally inconsistent")
        elif (
            any(value is not None for value in (self.estimate, self.low, self.high, self.estimator))
            or self.episodes != 0
            or not self.missing
        ):
            raise CausalUpliftError("not-identifiable causal evidence must name missing inputs")

    def as_record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "episodes": self.episodes,
            "estimator": self.estimator,
            "missing": list(self.missing),
        }


def not_identifiable(
    missing: tuple[str, ...] | list[str], *, required: bool = False
) -> CausalUplift:
    """Return an explicit status, or refuse when the caller required a causal answer."""
    fields = tuple(sorted(set(missing)))
    if not fields or any(not isinstance(value, str) or not value.strip() for value in fields):
        raise CausalUpliftError("not-identifiable status must name missing causal inputs")
    if required:
        raise CausalUpliftError("causal uplift is not identifiable; missing: " + ", ".join(fields))
    return CausalUplift("not_identifiable", None, None, None, 0, None, fields)


def estimate_cross_fitted_aipw(
    *,
    outcome: np.ndarray,
    treated: np.ndarray,
    propensity: np.ndarray,
    outcome_if_treated: np.ndarray,
    outcome_if_control: np.ndarray,
    episode_id: np.ndarray,
    design: CausalDesign,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> CausalUplift:
    """Estimate average treatment effect from externally cross-fitted nuisance predictions.

    This does not fit nuisance models inside the evaluation sample.  Their cross-fitted origin is
    a required part of ``design`` so in-sample outcome models cannot manufacture narrow evidence.
    """
    arrays = tuple(
        np.asarray(value)
        for value in (
            outcome,
            treated,
            propensity,
            outcome_if_treated,
            outcome_if_control,
            episode_id,
        )
    )
    y, treatment, e, mu1, mu0, episodes = arrays
    if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
        raise CausalUpliftError("causal arrays must be aligned one-dimensional observations")
    if not len(y):
        raise CausalUpliftError("causal population is empty")
    if treatment.dtype != np.bool_:
        raise CausalUpliftError("treated must be an explicit boolean array")
    if episodes.dtype.kind not in "iu" or np.any(episodes < 0):
        raise CausalUpliftError("episode_id must contain nonnegative integer campaign clusters")
    numeric = (y, e, mu1, mu0)
    if any(
        not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()
        for value in numeric
    ):
        raise CausalUpliftError(
            "causal outcomes, propensities and nuisance predictions must be finite"
        )
    if not treatment.any() or treatment.all():
        raise CausalUpliftError("causal population needs both treated and comparison observations")
    if np.any((e < design.overlap_floor) | (e > 1 - design.overlap_floor)):
        raise CausalUpliftError("propensity overlap falls outside the registered support")
    unique_episodes = np.unique(episodes)
    if len(unique_episodes) < 2:
        raise CausalUpliftError("causal uncertainty needs at least two campaign/control episodes")
    if type(n_bootstrap) is not int or n_bootstrap < 2:
        raise CausalUpliftError("n_bootstrap must be a built-in integer of at least two")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise CausalUpliftError("causal bootstrap seed must be in [0,2**63)")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise CausalUpliftError("causal bootstrap alpha must lie strictly between zero and one")
    score = mu1 - mu0 + treatment * (y - mu1) / e - (~treatment) * (y - mu0) / (1 - e)
    estimate = float(score.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, np.float64)
    by_episode = {value: score[episodes == value] for value in unique_episodes}
    for index in range(n_bootstrap):
        sampled = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        draws[index] = np.concatenate([by_episode[value] for value in sampled]).mean()
    low, high = np.quantile(draws, (alpha / 2, 1 - alpha / 2))
    return CausalUplift(
        "identified",
        estimate,
        float(low),
        float(high),
        len(unique_episodes),
        design.estimator,
    )


def campaign_impact_record(
    planned_mean: np.ndarray,
    reference_mean: np.ndarray,
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
    """Combine useful ON/OFF response with a non-silent causal status."""
    planned, reference = np.asarray(planned_mean), np.asarray(reference_mean)
    if (
        planned.shape != reference.shape
        or planned.ndim != 2
        or not np.isfinite(planned).all()
        or not np.isfinite(reference).all()
        or np.any(planned < 0)
        or np.any(reference < 0)
    ):
        raise ValueError("campaign means must be aligned finite nonnegative [product,day] arrays")
    if causal_uplift is None:
        causal_uplift = not_identifiable(causal_missing, required=require_causal)
    elif require_causal and causal_uplift.status != "identified":
        raise CausalUpliftError(
            "causal uplift was required but supplied evidence is not identifiable"
        )
    if causal_uplift.status == "not_identifiable":
        LOGGER.warning(
            "causal uplift not identifiable; continuing with prospective ON/OFF response; "
            "missing=%s",
            ",".join(causal_uplift.missing),
        )
    response = planned - reference
    return {
        "effect_type": "prospective_model_response",
        "daily": response.tolist(),
        "product_total": response.sum(1).tolist(),
        "total": float(response.sum()),
        "causal_uplift": causal_uplift.as_record(),
    }


__all__ = [
    "CausalDesign",
    "CausalUplift",
    "CausalUpliftError",
    "campaign_impact_record",
    "estimate_cross_fitted_aipw",
    "not_identifiable",
]
