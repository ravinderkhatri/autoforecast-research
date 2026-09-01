"""Public functional checks for the governed standalone M5 TimesFM contract."""

from __future__ import annotations

import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.acceptance_retirement import canonical_json_sha256  # noqa: E402
from autoforecast.benchmarks.m5_contract import M5_ACCEPTANCE_RETIREMENT  # noqa: E402
from autoforecast.benchmarks.m5_timesfm_contract import (  # noqa: E402
    TIMESFM_1P0_HF_REVISION,
    TIMESFM_1P0_MODEL_ID,
    TIMESFM_1P0_QUALIFIED_LABEL,
    TIMESFM_1P0_WEIGHTS_SHA256,
    TIMESFM_2P5_CONTAMINATION_LABEL,
    TIMESFM_2P5_MODEL_ID,
    TIMESFM_2P5_RETIRED_LABEL,
    TIMESFM_3P0_MODEL_ID,
    TIMESFM_M5_REGISTRATION_BODY_SHA256,
    TIMESFM_M5_REGISTRATION_ID,
    TIMESFM_RUNTIME_CANARIES_FILE_SHA256,
    TIMESFM_RUNTIME_CANARIES_RELATIVE_PATH,
    timesfm_2p5_run_contract,
    timesfm_m5_registration_body,
    timesfm_result_header_contract,
    validate_timesfm_claim_label,
    validate_timesfm_m5_registration,
    validate_timesfm_model_policy,
    validate_timesfm_result_header,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRATION_PATH = ROOT / "gates" / "registered" / "m5_timesfm_standalone_v1.json"
REGISTRATION_ENVELOPE = {
    "registration": timesfm_m5_registration_body(),
    "hash": TIMESFM_M5_REGISTRATION_BODY_SHA256,
}


def test_supplied_timesfm_registration_matches_code_pin() -> None:
    if not REGISTRATION_PATH.exists():
        import pytest
        pytest.skip("historical standalone TimesFM registration is not supplied")
    assert validate_timesfm_m5_registration(
        json.loads(REGISTRATION_PATH.read_text())
    ) == REGISTRATION_ENVELOPE["registration"]


def _refuses(call, message: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted invalid TimesFM evidence: {message}")


def test_timesfm_m5_registration_body_pins_models_origins_outputs_and_separate_routes() -> None:
    body = timesfm_m5_registration_body()
    assert body["registration_id"] == TIMESFM_M5_REGISTRATION_ID
    assert body["acceptance_retirement"] == M5_ACCEPTANCE_RETIREMENT.reference
    assert body["adapter_registry"] == {
        "name": "timesfm_standalone",
        "consumes_exposure": True,
        "handles_right_censoring": False,
        "supported_estimands": ["sales_given_availability"],
        "handicap": (
            "uses exposure for lifecycle trimming but has no right-censored likelihood; "
            "M5 evidence also requires the origin-specific pretraining-overlap block"
        ),
    }
    assert body["runtime_canary_evidence"] == {
        "path": TIMESFM_RUNTIME_CANARIES_RELATIVE_PATH,
        "file_sha256": TIMESFM_RUNTIME_CANARIES_FILE_SHA256,
        "evidence_role": "development_runtime_probe_only",
        "accuracy_measured": False,
    }

    model_1p0 = body["models"]["timesfm_1p0"]
    assert model_1p0["hf_revision"] == TIMESFM_1P0_HF_REVISION \
        == "0581e2c56cb06feb51cfd98fc2b4005b74f7187b"
    assert model_1p0["weights_sha256"] == TIMESFM_1P0_WEIGHTS_SHA256 \
        == "e6f83ebb8f448b66ad49d61b2f73aea0f68ffb20a83f1695c30badb1a5792eeb"
    assert model_1p0["inference"] == {
        "context_length": 512,
        "horizon_length": 28,
        "frequency_indicator": 0,
        "normalize_inputs": True,
        "point_forecast_mode": "mean",
        "post_provider_clipping": "clip_to_zero_and_count",
    }
    assert model_1p0["runtime"] == {
        "python": "3.11.15",
        "timesfm_package": "1.3.0",
        "torch": "2.13.0",
        "backend": "torch_cpu",
        "offline_load_required": True,
        "canary_negative_output_observed": True,
    }
    assert model_1p0["output"] == {
        "statistic": "mean", "tensor": "point_forecast", "point_forecast_mode": "mean",
    }

    model = body["models"]["timesfm_2p5"]
    assert model["model_id"] == TIMESFM_2P5_MODEL_ID
    assert model["hf_revision"] == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    assert model["weights_sha256"] == \
        "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8"
    assert model["source_revision"] == "331c6d33cb1ac2611de3056d0ac7164aab6301eb"
    assert model["inference"] == {
        "context_length": 1024,
        "horizon_length": 28,
        "normalize_inputs": True,
        "per_core_batch_size": 32,
        "infer_is_positive": True,
        "force_flip_invariance": True,
        "fix_quantile_crossing": True,
        "continuous_quantile_head": True,
        "post_provider_clipping": "clip_to_zero_and_count",
    }
    assert model["runtime"] == {
        "python": "3.12.11",
        "timesfm_package": "2.0.2",
        "torch": "2.13.0",
        "backend": "torch_cpu",
        "offline_load_required": True,
        "canary_negative_output_observed": True,
    }
    assert model["outputs"]["primary"] == {
        "statistic": "mean", "tensor": "quantile_forecast", "channel_index": 0,
    }
    assert model["outputs"]["secondary"] == {
        "statistic": "median", "tensor": "point_forecast",
    }

    assert body["origins"]["d1857"]["scoring_target_days"] == [1858, 1885]
    assert body["origins"]["d1857"]["documented_scoring_target_overlap"] is True
    assert body["origins"]["d1857"]["accuracy_admission"] == "refused"
    assert body["origins"]["d1913"]["scoring_target_days"] == [1914, 1941]
    assert body["origins"]["d1913"]["documented_scoring_target_overlap"] is False
    assert body["origins"]["d1913"]["evidence_role"] \
        == "descriptive_retired_origin_comparator"
    assert body["origins"]["d1913"]["origin_status"] \
        == "retired_origin_comparator_only"
    assert body["origins"]["d1857"]["pretraining_overlap"] == {
        "corpus": "Salesforce/GiftEvalPretrain",
        "artifact_sha256":
            "99be76eb30f0a91426beb8df5663bfd49b2deefd090839ca0e3515c00476ae6f",
        "series_overlap": "identical_30490",
        "documented_coverage_days": [1, 1913],
        "origin_day": 1857,
        "scoring_target_days": [1858, 1885],
        "documented_scoring_target_overlap": True,
    }
    assert body["origins"]["d1913"]["pretraining_overlap"] == {
        "corpus": "Salesforce/GiftEvalPretrain",
        "artifact_sha256":
            "99be76eb30f0a91426beb8df5663bfd49b2deefd090839ca0e3515c00476ae6f",
        "series_overlap": "identical_30490",
        "documented_coverage_days": [1, 1913],
        "origin_day": 1913,
        "scoring_target_days": [1914, 1941],
        "documented_scoring_target_overlap": False,
    }

    assert tuple(body["routes"]) == ("bottom_up", "direct")
    assert body["routes"]["bottom_up"]["score_levels"] == list(range(1, 13))
    assert body["routes"]["direct"]["score_levels"] == list(range(1, 10))
    assert body["route_combination_policy"] == "one_result_one_route_never_splice_or_average"


def test_validate_timesfm_m5_registration_authenticates_the_checked_in_envelope() -> None:
    body = validate_timesfm_m5_registration(REGISTRATION_ENVELOPE)
    assert body == timesfm_m5_registration_body()
    assert REGISTRATION_ENVELOPE["hash"] == TIMESFM_M5_REGISTRATION_BODY_SHA256
    assert canonical_json_sha256(body) == TIMESFM_M5_REGISTRATION_BODY_SHA256


def test_validate_timesfm_m5_registration_refuses_resealed_governance_mutations() -> None:
    mutations = (
        lambda body: body["models"]["timesfm_2p5"].__setitem__("hf_revision", "f" * 40),
        lambda body: body["models"]["timesfm_2p5"]["outputs"]["primary"].__setitem__(
            "statistic", "median"),
        lambda body: body["origins"]["d1857"].__setitem__("accuracy_admission", "allowed"),
        lambda body: body["origins"]["d1913"].__setitem__("evidence_role", "acceptance"),
        lambda body: body.__setitem__("route_combination_policy", "combine"),
        lambda body: body["version_policy"]["timesfm_3p0"].__setitem__(
            "weights_use", "allowed"),
    )
    for mutate in mutations:
        envelope = copy.deepcopy(REGISTRATION_ENVELOPE)
        mutate(envelope["registration"])
        envelope["hash"] = canonical_json_sha256(envelope["registration"])
        _refuses(
            lambda envelope=envelope: validate_timesfm_m5_registration(envelope),
            "not the exact code-pinned body",
        )


def test_validate_timesfm_model_policy_allows_only_pinned_1p0_and_2p5_and_refuses_3p0() -> None:
    validate_timesfm_model_policy(TIMESFM_1P0_MODEL_ID)
    validate_timesfm_model_policy(TIMESFM_2P5_MODEL_ID)
    for model_id in (
        TIMESFM_3P0_MODEL_ID,
        "google/timesfm-3.0",
        "google/timesfm-3.0-200m-pytorch",
        "timesfm-3.0",
        "timesfm",
        "",
        True,
    ):
        _refuses(
            lambda model_id=model_id: validate_timesfm_model_policy(model_id),
            "not permitted",
        )


def test_validate_timesfm_claim_label_accepts_only_the_registered_qualified_language() -> None:
    validate_timesfm_claim_label(
        TIMESFM_1P0_MODEL_ID, "2016-04-24", TIMESFM_1P0_QUALIFIED_LABEL)
    validate_timesfm_claim_label(
        TIMESFM_2P5_MODEL_ID, "2016-04-24", TIMESFM_2P5_RETIRED_LABEL)
    validate_timesfm_claim_label(
        TIMESFM_2P5_MODEL_ID, "2016-02-28", TIMESFM_2P5_CONTAMINATION_LABEL)
    for label in (
        "TimesFM 2.5 zero-shot M5",
        "clean M5 comparator",
        "target-leak-free retail transfer",
        TIMESFM_2P5_RETIRED_LABEL + ".",
    ):
        _refuses(
            lambda label=label: validate_timesfm_claim_label(
                TIMESFM_2P5_MODEL_ID, "2016-04-24", label),
            "exact registered evidence label",
        )


def test_timesfm_2p5_run_contract_refuses_d1857_accuracy_but_allows_a_segregated_probe() -> None:
    for purpose in ("accuracy_comparison", "descriptive_comparison", "leaderboard"):
        _refuses(
            lambda purpose=purpose: timesfm_2p5_run_contract(
                "2016-02-28", "bottom_up", purpose),
            "d1858-d1885 targets overlap",
        )
    probe = timesfm_2p5_run_contract("2016-02-28", "bottom_up", "contamination_probe")
    assert probe["origin"]["accuracy_admission"] == "refused"
    assert probe["claim"]["label"] == TIMESFM_2P5_CONTAMINATION_LABEL
    assert probe["claim"]["comparison_table_admission"] is False
    assert probe["claim"]["leaderboard_admission"] is False


def test_timesfm_2p5_run_contract_labels_d1913_as_retired_descriptive_only() -> None:
    contract = timesfm_2p5_run_contract(
        "2016-04-24", "bottom_up", "descriptive_comparison")
    assert contract["origin"]["origin_day"] == 1913
    assert contract["origin"]["scoring_target_days"] == [1914, 1941]
    assert contract["origin"]["documented_scoring_target_overlap"] is False
    assert contract["origin"]["evidence_role"] == "descriptive_retired_origin_comparator"
    assert contract["origin"]["origin_status"] == "retired_origin_comparator_only"
    assert contract["claim"]["label"] == TIMESFM_2P5_RETIRED_LABEL
    assert contract["claim"]["acceptance"] is False
    assert contract["claim"]["holdout"] is False
    assert contract["claim"]["current_v2_superiority"] is False


def test_timesfm_2p5_run_contract_keeps_direct_and_bottom_up_scopes_disjoint() -> None:
    bottom_up = timesfm_2p5_run_contract(
        "2016-04-24", "bottom_up", "descriptive_comparison")
    direct = timesfm_2p5_run_contract(
        "2016-04-24", "direct", "descriptive_comparison")
    assert bottom_up["route"]["result_identity"] == "m5_timesfm_2p5_d1913_bottom_up"
    assert direct["route"]["result_identity"] == "m5_timesfm_2p5_d1913_direct"
    assert bottom_up["route"]["score_levels"] == list(range(1, 13))
    assert direct["route"]["score_levels"] == list(range(1, 10))
    assert bottom_up["route"]["forecast_series"] == 30_490
    assert direct["route"]["forecast_series"] == 154
    for invalid in ("direct+bottom_up", "combined", "all", True):
        _refuses(
            lambda invalid=invalid: timesfm_2p5_run_contract(
                "2016-04-24", invalid, "descriptive_comparison"),
            "route must be exactly",
        )


def test_timesfm_result_header_contract_builds_one_route_only_without_legacy_shape() -> None:
    header = timesfm_result_header_contract(
        "2016-04-24", "direct", "descriptive_comparison")
    assert header["schema"] == "autoforecast.m5-timesfm-result.v1"
    assert header["reported_routes"] == ["direct"]
    assert header["registration"] == {
        "registration_id": TIMESFM_M5_REGISTRATION_ID,
        "body_sha256": TIMESFM_M5_REGISTRATION_BODY_SHA256,
    }
    assert "legacy" not in header
    assert "checkpoint_translation" not in header


def test_validate_timesfm_result_header_refuses_route_claim_and_output_relabelling() -> None:
    header = timesfm_result_header_contract(
        "2016-04-24", "bottom_up", "descriptive_comparison")
    validate_timesfm_result_header(header)
    mutations = (
        lambda value: value.__setitem__("reported_routes", ["bottom_up", "direct"]),
        lambda value: value["route"].__setitem__("score_levels", list(range(1, 10))),
        lambda value: value["claim"].__setitem__("label", "TimesFM 2.5 zero-shot M5"),
        lambda value: value["model"]["outputs"]["primary"].__setitem__(
            "statistic", "median"),
        lambda value: value["origin"].__setitem__("evidence_role", "acceptance"),
    )
    for mutate in mutations:
        changed = copy.deepcopy(header)
        mutate(changed)
        _refuses(
            lambda changed=changed: validate_timesfm_result_header(changed),
            "exact registered TimesFM result header",
        )


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_"):
            function()
            print("PASS " + name)
