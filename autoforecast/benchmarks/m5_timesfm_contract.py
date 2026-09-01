"""Governance boundary for standalone TimesFM measurements on M5.

The public functions in this module classify evidence; they do not run a model or score a
forecast. The contract has no legacy translation path. A run either carries the exact registered
identity below or it is not governed TimesFM evidence.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from autoforecast.acceptance_retirement import canonical_json_sha256
from autoforecast.benchmarks.m5_contract import (
    M5_ACCEPTANCE_RETIREMENT,
    M5_BOTTOM_LEVEL_SERIES,
    M5_EXPECTED_GRID_CELLS,
    M5_HORIZON_DAYS,
    M5_LEVEL_CARDINALITIES,
    json_values_equal_exact,
)

TIMESFM_M5_REGISTRATION_SCHEMA = "autoforecast.m5-timesfm-registration.v1"
TIMESFM_M5_RESULT_SCHEMA = "autoforecast.m5-timesfm-result.v1"
TIMESFM_M5_REGISTRATION_ID = "m5_timesfm_standalone_v1"
TIMESFM_M5_REGISTERED_ON = "2026-08-29"
TIMESFM_M5_REGISTRATION_BODY_SHA256 = (
    "788d78aa0c4bf16e954afbb7b90ce26d0d271b5a804e2418920f9ba8c9e5d41a"
)

TIMESFM_1P0_MODEL_ID = "google/timesfm-1.0-200m-pytorch"
TIMESFM_2P5_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
TIMESFM_3P0_MODEL_ID = "google/timesfm-3.0-pytorch"

TIMESFM_1P0_QUALIFIED_LABEL = "M5 absent from the disclosed TimesFM 1.0 corpus"
TIMESFM_2P5_RETIRED_LABEL = (
    "M5-pretrained; documented scoring targets disjoint at day 1913"
)
TIMESFM_2P5_CONTAMINATION_LABEL = (
    "contamination probe; d1858-d1885 targets overlap TimesFM 2.5 pretraining"
)

TIMESFM_2P5_HF_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
TIMESFM_2P5_WEIGHTS_SHA256 = (
    "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8"
)
TIMESFM_2P5_CONFIG_SHA256 = (
    "cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408"
)
TIMESFM_2P5_SOURCE_REVISION = "331c6d33cb1ac2611de3056d0ac7164aab6301eb"
TIMESFM_2P5_PRETRAIN_ARTIFACT_SHA256 = (
    "99be76eb30f0a91426beb8df5663bfd49b2deefd090839ca0e3515c00476ae6f"
)
TIMESFM_2P5_CONTEXT_DAYS = 1024
TIMESFM_2P5_BATCH_SIZE = 32
TIMESFM_2P5_POSTPROCESS = "clip_to_zero_and_count"

TIMESFM_1P0_HF_REVISION = "0581e2c56cb06feb51cfd98fc2b4005b74f7187b"
TIMESFM_1P0_WEIGHTS_SHA256 = (
    "e6f83ebb8f448b66ad49d61b2f73aea0f68ffb20a83f1695c30badb1a5792eeb"
)
TIMESFM_RUNTIME_CANARIES_RELATIVE_PATH = (
    "gates/derived/timesfm_runtime_canaries_2026-08-29.json"
)
TIMESFM_RUNTIME_CANARIES_FILE_SHA256 = (
    "c6a7b3d1c47a522f0e6433f229efcf9a7e9cf6ed7c8f85d0b205ffda3a427743"
)
_TIMESFM_3P0_REFUSED_ALIASES = (
    TIMESFM_3P0_MODEL_ID,
    "google/timesfm-3.0",
    "google/timesfm-3.0-200m-pytorch",
    "timesfm-3.0",
)
_D1857_ORIGIN = "2016-02-28"
_D1913_ORIGIN = "2016-04-24"
_ROUTES = ("bottom_up", "direct")


def _timesfm_2p5_model_contract() -> dict[str, object]:
    return {
        "model_id": TIMESFM_2P5_MODEL_ID,
        "hf_revision": TIMESFM_2P5_HF_REVISION,
        "weights_sha256": TIMESFM_2P5_WEIGHTS_SHA256,
        "config_sha256": TIMESFM_2P5_CONFIG_SHA256,
        "source_revision": TIMESFM_2P5_SOURCE_REVISION,
        "api_family": "timesfm_2p5",
        "license": "Apache-2.0",
        "inference": {
            "context_length": TIMESFM_2P5_CONTEXT_DAYS,
            "horizon_length": M5_HORIZON_DAYS,
            "normalize_inputs": True,
            "per_core_batch_size": TIMESFM_2P5_BATCH_SIZE,
            "infer_is_positive": True,
            "force_flip_invariance": True,
            "fix_quantile_crossing": True,
            "continuous_quantile_head": True,
            "post_provider_clipping": TIMESFM_2P5_POSTPROCESS,
        },
        "runtime": {
            "python": "3.12.11",
            "timesfm_package": "2.0.2",
            "torch": "2.13.0",
            "backend": "torch_cpu",
            "offline_load_required": True,
            "canary_negative_output_observed": True,
        },
        "outputs": {
            "primary": {
                "statistic": "mean",
                "tensor": "quantile_forecast",
                "channel_index": 0,
            },
            "secondary": {
                "statistic": "median",
                "tensor": "point_forecast",
            },
            "combination": "never_swap_average_or_merge_mean_and_median",
        },
        "inputs": {
            "target": "unit_sales_only",
            "cutoff": "date_less_than_or_equal_to_origin",
            "observed_zero": "preserve_exactly",
            "leading_unobserved": "trim",
            "interior_unobserved": "refuse",
            "all_unobserved": "decline_before_provider",
            "future_covariates": "forbidden",
            "xreg": "forbidden",
        },
    }


def _timesfm_1p0_contract() -> dict[str, object]:
    return {
        "model_id": TIMESFM_1P0_MODEL_ID,
        "hf_revision": TIMESFM_1P0_HF_REVISION,
        "weights_sha256": TIMESFM_1P0_WEIGHTS_SHA256,
        "status": "real_arm64_runtime_canary_passed_qualified_comparison_candidate",
        "inference": {
            "context_length": 512,
            "horizon_length": M5_HORIZON_DAYS,
            "frequency_indicator": 0,
            "normalize_inputs": True,
            "point_forecast_mode": "mean",
            "post_provider_clipping": "clip_to_zero_and_count",
        },
        "runtime": {
            "python": "3.11.15",
            "timesfm_package": "1.3.0",
            "torch": "2.13.0",
            "backend": "torch_cpu",
            "offline_load_required": True,
            "canary_negative_output_observed": True,
        },
        "output": {
            "statistic": "mean",
            "tensor": "point_forecast",
            "point_forecast_mode": "mean",
        },
        "qualified_label": TIMESFM_1P0_QUALIFIED_LABEL,
        "qualification": (
            "disclosed-corpus absence is not a cryptographic audit; other retail data is "
            "disclosed"
        ),
        "remaining_before_governed_run": ["final_registered_runtime_lock"],
    }


def _timesfm_3p0_refusal() -> dict[str, object]:
    return {
        "model_id": TIMESFM_3P0_MODEL_ID,
        "weights_license": "timesfm-non-commercial-license-v1.0",
        "refused_aliases": list(_TIMESFM_3P0_REFUSED_ALIASES),
        "weights_use": "refused",
        "download_authorized": False,
        "gated_terms_acceptance_authorized": False,
        "benchmark_authorized": False,
        "production_authorized": False,
    }


def _overlap_contract(origin_day: int, target_days: list[int]) -> dict[str, object]:
    target_overlap = target_days[0] <= 1913
    return {
        "corpus": "Salesforce/GiftEvalPretrain",
        "artifact_sha256": TIMESFM_2P5_PRETRAIN_ARTIFACT_SHA256,
        "series_overlap": "identical_30490",
        "documented_coverage_days": [1, 1913],
        "origin_day": origin_day,
        "scoring_target_days": list(target_days),
        "documented_scoring_target_overlap": target_overlap,
    }


def _origin_contracts() -> dict[str, object]:
    return {
        "d1857": {
            "date": _D1857_ORIGIN,
            "origin_day": 1857,
            "scoring_target_days": [1858, 1885],
            "documented_scoring_target_overlap": True,
            "evidence_role": "segregated_contamination_probe",
            "origin_status": "documented_pretraining_target_overlap",
            "accuracy_admission": "refused",
            "comparison_table_admission": False,
            "leaderboard_admission": False,
            "pretraining_overlap": _overlap_contract(1857, [1858, 1885]),
        },
        "d1913": {
            "date": _D1913_ORIGIN,
            "origin_day": 1913,
            "scoring_target_days": [1914, 1941],
            "documented_scoring_target_overlap": False,
            "evidence_role": "descriptive_retired_origin_comparator",
            "origin_status": "retired_origin_comparator_only",
            "accuracy_admission": "retrospective_descriptive_only",
            "comparison_table_admission": "retrospective_descriptive_only",
            "leaderboard_admission": False,
            "pretraining_overlap": _overlap_contract(1913, [1914, 1941]),
        },
    }


def _route_contracts() -> dict[str, object]:
    direct_series = sum(list(M5_LEVEL_CARDINALITIES.values())[:9])
    return {
        "bottom_up": {
            "name": "bottom_up",
            "result_identity": "m5_timesfm_2p5_d1913_bottom_up",
            "forecast_grain": "item_id+store_id",
            "forecast_series": M5_BOTTOM_LEVEL_SERIES,
            "forecast_cells": M5_EXPECTED_GRID_CELLS,
            "score_levels": list(range(1, 13)),
            "aggregation": "existing_wrmsse_scorer_from_leaf_forecasts",
        },
        "direct": {
            "name": "direct",
            "result_identity": "m5_timesfm_2p5_d1913_direct",
            "forecast_grain": "registered_non_leaf_hierarchy_nodes",
            "forecast_series": direct_series,
            "forecast_cells": direct_series * M5_HORIZON_DAYS,
            "score_levels": list(range(1, 10)),
            "aggregation": "none_direct_node_forecasts",
        },
    }


def timesfm_m5_registration_body() -> dict[str, object]:
    """Return the exact prospective standalone M5 TimesFM registration body."""
    return {
        "schema": TIMESFM_M5_REGISTRATION_SCHEMA,
        "registration_id": TIMESFM_M5_REGISTRATION_ID,
        "registered_on": TIMESFM_M5_REGISTERED_ON,
        "status": "prospective_before_governed_standalone_run",
        "scope": "standalone_timesfm_only_no_custom_model_integration",
        "runtime_canary_evidence": {
            "path": TIMESFM_RUNTIME_CANARIES_RELATIVE_PATH,
            "file_sha256": TIMESFM_RUNTIME_CANARIES_FILE_SHA256,
            "evidence_role": "development_runtime_probe_only",
            "accuracy_measured": False,
        },
        "adapter_registry": {
            "name": "timesfm_standalone",
            "consumes_exposure": True,
            "handles_right_censoring": False,
            "supported_estimands": ["sales_given_availability"],
            "handicap": (
                "uses exposure for lifecycle trimming but has no right-censored likelihood; "
                "M5 evidence also requires the origin-specific pretraining-overlap block"
            ),
        },
        "models": {
            "timesfm_1p0": _timesfm_1p0_contract(),
            "timesfm_2p5": _timesfm_2p5_model_contract(),
        },
        "version_policy": {
            "timesfm_1p0": "qualified_runtime_canary_passed_optional_context_only",
            "timesfm_2p5": "retired_origin_descriptive_or_segregated_probe_only",
            "timesfm_3p0": _timesfm_3p0_refusal(),
        },
        "origins": _origin_contracts(),
        "routes": _route_contracts(),
        "route_combination_policy": "one_result_one_route_never_splice_or_average",
        "claims": {
            "timesfm_1p0_qualified_label": TIMESFM_1P0_QUALIFIED_LABEL,
            "timesfm_2p5_retired_label": TIMESFM_2P5_RETIRED_LABEL,
            "timesfm_2p5_contamination_label": TIMESFM_2P5_CONTAMINATION_LABEL,
            "forbidden_labels": [
                "zero-shot M5",
                "clean M5",
                "clean comparator",
                "target-leak-free",
                "general retail transfer",
                "current-v2 superiority",
            ],
        },
        "acceptance_retirement": M5_ACCEPTANCE_RETIREMENT.reference,
    }


def validate_timesfm_m5_registration(envelope: object) -> dict[str, object]:
    """Authenticate and return the one code-pinned standalone registration body."""
    if not isinstance(envelope, dict) or set(envelope) != {"registration", "hash"}:
        raise ValueError("TimesFM registration envelope must contain registration and hash")
    registration = envelope.get("registration")
    stored_hash = envelope.get("hash")
    if not isinstance(registration, dict) or not isinstance(stored_hash, str):
        raise ValueError("TimesFM registration body/hash types are invalid")
    if (
        stored_hash != TIMESFM_M5_REGISTRATION_BODY_SHA256
        or canonical_json_sha256(registration) != TIMESFM_M5_REGISTRATION_BODY_SHA256
        or not json_values_equal_exact(registration, timesfm_m5_registration_body())
    ):
        raise ValueError("TimesFM registration is not the exact code-pinned body")
    return copy.deepcopy(registration)


def validate_timesfm_model_policy(model_id: object) -> None:
    """Refuse TimesFM 3.0, floating aliases and every unregistered model identity."""
    if type(model_id) is not str or model_id not in {
        TIMESFM_1P0_MODEL_ID,
        TIMESFM_2P5_MODEL_ID,
    }:
        raise ValueError("TimesFM model identity is not permitted by the standalone contract")


def validate_timesfm_claim_label(model_id: object, origin: object, label: object) -> None:
    """Require the exact qualified evidence label for a registered model and origin."""
    validate_timesfm_model_policy(model_id)
    expected: str | None = None
    if model_id == TIMESFM_1P0_MODEL_ID and origin == _D1913_ORIGIN:
        expected = TIMESFM_1P0_QUALIFIED_LABEL
    elif model_id == TIMESFM_2P5_MODEL_ID and origin == _D1913_ORIGIN:
        expected = TIMESFM_2P5_RETIRED_LABEL
    elif model_id == TIMESFM_2P5_MODEL_ID and origin == _D1857_ORIGIN:
        expected = TIMESFM_2P5_CONTAMINATION_LABEL
    if type(label) is not str or label != expected:
        raise ValueError("claim does not use the exact registered evidence label")


def _claim_contract(origin: str, purpose: str) -> dict[str, object]:
    if origin == _D1857_ORIGIN:
        return {
            "purpose": purpose,
            "label": TIMESFM_2P5_CONTAMINATION_LABEL,
            "comparison_table_admission": False,
            "leaderboard_admission": False,
            "acceptance": False,
            "holdout": False,
            "gate": False,
            "current_v2_superiority": False,
            "production_transfer": False,
        }
    return {
        "purpose": purpose,
        "label": TIMESFM_2P5_RETIRED_LABEL,
        "comparison_table_admission": "retrospective_descriptive_only",
        "leaderboard_admission": False,
        "acceptance": False,
        "holdout": False,
        "gate": False,
        "current_v2_superiority": False,
        "production_transfer": False,
    }


def timesfm_2p5_run_contract(
    origin: object,
    route: object,
    purpose: object,
) -> dict[str, object]:
    """Return one exact route-scoped run contract or refuse inadmissible accuracy evidence."""
    validate_timesfm_model_policy(TIMESFM_2P5_MODEL_ID)
    if type(route) is not str or route not in _ROUTES:
        raise ValueError("TimesFM route must be exactly 'bottom_up' or 'direct'")
    if type(origin) is not str or type(purpose) is not str:
        raise ValueError("TimesFM origin and purpose must be strings")
    if origin == _D1857_ORIGIN:
        if purpose != "contamination_probe":
            raise ValueError(
                "TimesFM 2.5 d1858-d1885 targets overlap documented pretraining; "
                "accuracy evidence is refused"
            )
        origin_key = "d1857"
    elif origin == _D1913_ORIGIN:
        if purpose != "descriptive_comparison":
            raise ValueError(
                "TimesFM 2.5 day-1913 permits only a retired descriptive comparison"
            )
        origin_key = "d1913"
    else:
        raise ValueError("TimesFM 2.5 origin is not registered")

    body = timesfm_m5_registration_body()
    route_contract = copy.deepcopy(body["routes"][route])
    if origin_key == "d1857":
        route_contract["result_identity"] = (
            f"m5_timesfm_2p5_d1857_{route}_contamination_probe"
        )
    return {
        "model": copy.deepcopy(body["models"]["timesfm_2p5"]),
        "origin": copy.deepcopy(body["origins"][origin_key]),
        "route": route_contract,
        "claim": _claim_contract(origin, purpose),
        "pretraining_overlap": copy.deepcopy(
            body["origins"][origin_key]["pretraining_overlap"]
        ),
        "acceptance_retirement": copy.deepcopy(body["acceptance_retirement"]),
    }


def timesfm_result_header_contract(
    origin: object,
    route: object,
    purpose: object,
) -> dict[str, object]:
    """Build the complete governance header for one separately published route result."""
    run_contract = timesfm_2p5_run_contract(origin, route, purpose)
    assert isinstance(route, str)  # narrowed by timesfm_2p5_run_contract
    return {
        "schema": TIMESFM_M5_RESULT_SCHEMA,
        "registration": {
            "registration_id": TIMESFM_M5_REGISTRATION_ID,
            "body_sha256": TIMESFM_M5_REGISTRATION_BODY_SHA256,
        },
        "reported_routes": [route],
        **run_contract,
    }


def validate_timesfm_result_header(record: object) -> None:
    """Require the exact governance header for one TimesFM model, origin and route."""
    if not isinstance(record, Mapping):
        raise ValueError("result lacks the exact registered TimesFM result header")
    origin_record = record.get("origin")
    route_record = record.get("route")
    claim_record = record.get("claim")
    if not all(isinstance(value, Mapping) for value in (
        origin_record, route_record, claim_record,
    )):
        raise ValueError("result lacks the exact registered TimesFM result header")
    origin = origin_record.get("date")
    route = route_record.get("name")
    purpose = claim_record.get("purpose")
    try:
        expected = timesfm_result_header_contract(origin, route, purpose)
    except ValueError as exc:
        raise ValueError("result lacks the exact registered TimesFM result header") from exc
    observed = {field: record.get(field) for field in expected}
    if not json_values_equal_exact(observed, expected):
        raise ValueError("result lacks the exact registered TimesFM result header")


__all__ = [
    "TIMESFM_1P0_HF_REVISION",
    "TIMESFM_1P0_MODEL_ID",
    "TIMESFM_1P0_QUALIFIED_LABEL",
    "TIMESFM_1P0_WEIGHTS_SHA256",
    "TIMESFM_2P5_CONTAMINATION_LABEL",
    "TIMESFM_2P5_CONFIG_SHA256",
    "TIMESFM_2P5_BATCH_SIZE",
    "TIMESFM_2P5_CONTEXT_DAYS",
    "TIMESFM_2P5_HF_REVISION",
    "TIMESFM_2P5_MODEL_ID",
    "TIMESFM_2P5_PRETRAIN_ARTIFACT_SHA256",
    "TIMESFM_2P5_POSTPROCESS",
    "TIMESFM_2P5_RETIRED_LABEL",
    "TIMESFM_2P5_SOURCE_REVISION",
    "TIMESFM_2P5_WEIGHTS_SHA256",
    "TIMESFM_3P0_MODEL_ID",
    "TIMESFM_M5_REGISTERED_ON",
    "TIMESFM_M5_REGISTRATION_BODY_SHA256",
    "TIMESFM_M5_REGISTRATION_ID",
    "TIMESFM_M5_REGISTRATION_SCHEMA",
    "TIMESFM_M5_RESULT_SCHEMA",
    "TIMESFM_RUNTIME_CANARIES_FILE_SHA256",
    "TIMESFM_RUNTIME_CANARIES_RELATIVE_PATH",
    "timesfm_2p5_run_contract",
    "timesfm_m5_registration_body",
    "timesfm_result_header_contract",
    "validate_timesfm_claim_label",
    "validate_timesfm_m5_registration",
    "validate_timesfm_model_policy",
    "validate_timesfm_result_header",
]
