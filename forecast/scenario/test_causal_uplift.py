"""Tests for the separate PACT campaign-response and causal-uplift boundary."""

from __future__ import annotations

import numpy as np
from forecast.scenario.causal_uplift import (
    CausalDesign,
    CausalUpliftError,
    campaign_impact_record,
    estimate_cross_fitted_aipw,
)


def test_normal_scenario_continues_with_explicit_not_identifiable_status() -> None:
    planned = np.array([[12.0, 13.0], [4.0, 5.0]])
    reference = np.array([[10.0, 10.0], [5.0, 5.0]])
    result = campaign_impact_record(planned, reference)
    assert result["effect_type"] == "prospective_model_response"
    assert result["product_total"] == [5.0, -1.0]
    assert result["causal_uplift"]["status"] == "not_identifiable"
    assert result["causal_uplift"]["estimate"] is None
    assert result["causal_uplift"]["missing"]


def test_causal_required_mode_refuses_missing_identification_inputs() -> None:
    try:
        campaign_impact_record(np.ones((1, 2)), np.ones((1, 2)), require_causal=True)
    except CausalUpliftError as error:
        assert "not identifiable" in str(error)
    else:
        raise AssertionError("causal-required mode silently fell back to model response")


def test_cross_fitted_aipw_recovers_hand_computed_effect() -> None:
    design = CausalDesign(
        origin="2026-01-01",
        campaign_start="2026-02-01",
        treatment_unit="product_store",
        comparison_population="eligible untreated product stores",
        confounders=("pre_sales", "price", "season", "store"),
    )
    treated = np.array([True, False, True, False])
    mu0 = np.array([10.0, 20.0, 30.0, 40.0])
    mu1 = mu0 + 3.0
    outcome = np.where(treated, mu1, mu0)
    result = estimate_cross_fitted_aipw(
        outcome=outcome,
        treated=treated,
        propensity=np.full(4, 0.5),
        outcome_if_treated=mu1,
        outcome_if_control=mu0,
        episode_id=np.array([0, 0, 1, 1]),
        design=design,
        n_bootstrap=40,
        seed=2,
    )
    assert result.status == "identified"
    assert result.estimate == 3.0
    assert result.low == result.high == 3.0


def test_causal_gate_refuses_overlap_and_in_sample_aliases() -> None:
    try:
        CausalDesign(
            origin="2026-01-01",
            campaign_start="2026-02-01",
            treatment_unit="product",
            comparison_population="controls",
            confounders=("pre_sales",),
            cross_fitted=False,
        )
    except CausalUpliftError as error:
        assert "cross-fitted" in str(error)
    else:
        raise AssertionError("in-sample nuisance predictions were accepted as causal evidence")

    design = CausalDesign(
        origin="2026-01-01",
        campaign_start="2026-02-01",
        treatment_unit="product",
        comparison_population="controls",
        confounders=("pre_sales",),
    )
    try:
        estimate_cross_fitted_aipw(
            outcome=np.arange(4.0),
            treated=np.array([True, False, True, False]),
            propensity=np.array([0.01, 0.5, 0.5, 0.5]),
            outcome_if_treated=np.arange(4.0),
            outcome_if_control=np.arange(4.0),
            episode_id=np.array([0, 0, 1, 1]),
            design=design,
            n_bootstrap=10,
        )
    except CausalUpliftError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("unsupported causal extrapolation was accepted")
