"""PACT paired scenario and causal fallback tests."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from forecast.models.mft.test_pact import _model, _typed_model_inputs
from forecast.scenario.causal_uplift import CausalUpliftError
from forecast.scenario.pact_campaign import evaluate_pact_campaign_pair


def test_pact_campaign_pair_keeps_prediction_and_logs_missing_causal(caplog) -> None:
    model = _model()
    reference = _typed_model_inputs()
    future = reference.future_features.clone()
    future[:, :, 0] += 1
    planned = replace(reference, future_features=future)
    with caplog.at_level(logging.WARNING):
        result = evaluate_pact_campaign_pair(model, planned, reference)
    assert result["effect_type"] == "prospective_model_response"
    assert result["causal_uplift"]["status"] == "not_identifiable"
    assert "continuing with prospective ON/OFF response" in caplog.text


def test_pact_campaign_pair_strict_causal_mode_raises() -> None:
    model = _model()
    reference = _typed_model_inputs()
    future = reference.future_features.clone()
    future[:, :, 0] += 1
    planned = replace(reference, future_features=future)
    with pytest.raises(CausalUpliftError, match="not identifiable"):
        evaluate_pact_campaign_pair(model, planned, reference, require_causal=True)


def test_pact_campaign_pair_refuses_history_change() -> None:
    model = _model()
    reference = _typed_model_inputs()
    planned = replace(reference, history_features=reference.history_features.clone() + 1)
    with pytest.raises(ValueError, match="future_features"):
        evaluate_pact_campaign_pair(model, planned, reference)
