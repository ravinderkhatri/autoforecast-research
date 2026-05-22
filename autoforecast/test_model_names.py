"""Naming boundaries: public identity, loss routing, and current Python import paths."""

import importlib

import pytest

from autoforecast.model_names import REVISION_MODELS, REVISION_VARIANTS, model_name


def test_model_names_preserve_recipe_wire_identifiers():
    assert REVISION_VARIANTS == {
        "mft_corrected": "v1_corrected", "sft": "v2", "hmfnet": "v3", "pact": "v5",
    }
    assert {REVISION_MODELS[v] for v in REVISION_VARIANTS.values()} == set(REVISION_VARIANTS)
    assert [model_name(v) for v in ("v2", "v3", "v5", "tft_hnb", )] == [
        "SFT", "HMFNet", "PACT", "TFT-HNB",
    ]
    assert model_name("custom_model") == "custom_model"


@pytest.mark.parametrize(("name", "loss", "width"), [
    ("tft_q", "quantile", 4), ("tft_hnb", "v3_hurdle_nb", 3),
])
def test_tft_named_variants(name, loss, width, monkeypatch):
    from autoforecast import adapters, registry
    from autoforecast.adapters import tft_q

    sentinel = object()
    calls = []

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(tft_q, "fit_predict", fit)
    registry._reset_for_test()
    try:
        adapters.register_defaults(only=("tft_q",))
        entry = registry.get(name)
        assert entry.forecast("panel", "origin", 28, seed=7) is sentinel
        assert calls == [(("panel", "origin", 28), {
            "seed": 7, "loss_kind": loss, "output_size": width,
        })]
        for override in ({"loss_kind": "wrong"}, {"output_size": 99}):
            with pytest.raises(ValueError, match="requires"):
                entry.forecast("panel", "origin", 28, **override)
        assert len(calls) == 1
    finally:
        registry._reset_for_test()


@pytest.mark.parametrize(("module", "model"), [
    ("forecast.models.mft.sft", "SFTModel"),
    ("forecast.models.mft.hmfnet", "HMFNetModel"),
    ("forecast.models.mft.pact", "PACTModel"),
])
def test_current_model_types_import_without_compatibility_modules(module, model):
    assert getattr(importlib.import_module(module), model).__module__ == module
