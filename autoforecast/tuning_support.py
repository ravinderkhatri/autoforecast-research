"""Truthful HPO eligibility for every registered forecasting adapter.

An adapter can be executable without offering a legitimate outer-loop search space.  This small
table prevents a UI or study YAML from turning fixed data semantics, provider revisions, or an
internal AutoML budget into pretend hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TuningSupport", "tuning_support"]


@dataclass(frozen=True)
class TuningSupport:
    """Whether an adapter has a reviewed, externally tuneable execution surface."""

    supported: bool
    reason: str


_SUPPORT = {
    "timesfm_hnb": TuningSupport(
        True, "typed TimesFM-HNB development recipe with singleton frozen domains"
    ),
    "timesfm_v8": TuningSupport(True, "typed V8 development recipe with singleton frozen domains"),
    "mft": TuningSupport(True, "typed architecture and optimizer settings"),
    "sft": TuningSupport(True, "typed SFT revision architecture and optimizer settings"),
    "hmfnet": TuningSupport(True, "typed HMFNet revision architecture and optimizer settings"),
    "pact": TuningSupport(True, "typed PACT decoder architecture and optimizer settings"),
    "shallow_global": TuningSupport(True, "typed LightGBM capacity and learning-rate settings"),
    "tft": TuningSupport(True, "typed TFT architecture and optimizer settings"),
    "tft_q": TuningSupport(
        True,
        "typed, identity-preserving daily benchmark TFT profile",
    ),
    "tft_hnb": TuningSupport(True, "typed TFT hurdle-negative-binomial profile"),
    "seasonal_naive": TuningSupport(
        False,
        "season length is a dataset contract, not a model hyperparameter",
    ),
    "croston": TuningSupport(False, "the released CrostonClassic adapter has no outer-loop knob"),
    "auto_ets": TuningSupport(
        False,
        "StatsForecast selects the ETS form internally from the declared panel frequency",
    ),
    "auto_theta": TuningSupport(
        False,
        "StatsForecast selects the Theta fit internally from the declared panel frequency",
    ),
    "autogluon_zoo": TuningSupport(
        False,
        "its search belongs to AutoGluon under a separately frozen time budget",
    ),
    "timesfm_standalone": TuningSupport(
        False,
        "the released foundation checkpoint and provider revision are frozen identity",
    ),
}


def tuning_support(adapter: str) -> TuningSupport:
    """Return one dependency-free HPO eligibility declaration for an adapter."""
    from autoforecast.adapter_contracts import descriptor
    definition = descriptor(adapter)
    if adapter not in _SUPPORT and definition.parameters == "none":
        return TuningSupport(False, "maintained descriptor declares no model parameters")
    try:
        result = _SUPPORT[adapter]
        if result.supported != (definition.parameters == "typed"):
            raise ValueError(f"adapter {adapter!r} has conflicting parameter contracts")
        return result
    except KeyError:
        raise ValueError(
            f"unknown adapter {adapter!r} has no HPO eligibility declaration"
        ) from None
