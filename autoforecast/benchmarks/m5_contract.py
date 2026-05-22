"""Shared immutable facts and workflow identities for the governed M5 runs.

These values used to be copied into individual runners. That allowed Phase 2 to register a v3
manifest while Phase 4 still waited forever for v2, and allowed one scorer to call a smaller grid
"full M5". A comparison boundary is only shared if every producer and consumer imports the same
identity and population.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from autoforecast.acceptance_retirement import RetirementExpectation
from autoforecast.protocol import EvaluationManifest

M5_HORIZON_DAYS = 28
M5_BOTTOM_LEVEL_SERIES = 30_490
M5_EXPECTED_GRID_CELLS = M5_BOTTOM_LEVEL_SERIES * M5_HORIZON_DAYS

M5_LEVEL_CARDINALITIES = {
    "total": 1,
    "state_id": 3,
    "store_id": 10,
    "cat_id": 3,
    "dept_id": 7,
    "state_id+cat_id": 9,
    "state_id+dept_id": 21,
    "store_id+cat_id": 30,
    "store_id+dept_id": 70,
    "item_id": 3_049,
    "item_id+state_id": 9_147,
    "item_id+store_id": 30_490,
}
M5_EXPECTED_ZERO_DROPS = {name: 0 for name in M5_LEVEL_CARDINALITIES}

# EvaluationManifest.save() materialises this as
# gates/registered/manifest_<PHASE2_MANIFEST_NAME>.json. Phase 2 and every downstream gate must
# import this exact value; a copied filename silently makes a successful rerun unusable.
PHASE2_MANIFEST_NAME = "m5_phase2_grid_zero_v3"
M5_PHASE2_DEVELOPMENT_ORIGINS = ("2016-01-03", "2016-01-31", "2016-02-28")
M5_PHASE2_ACCEPTANCE_ORIGIN = "2016-04-24"
M5_PUBLIC_VALIDATION_ORIGIN = "2016-04-24"
PHASE3_MANIFEST_NAME = "m5_phase3_public_window_checkpoint_v2"
M5_PHASE3_COMPARATORS = (
    "seasonal_naive", "croston", "shallow_global", "tft", "autogluon_zoo")
M5_PHASE3_REQUIRED_COMPARATORS = M5_PHASE3_COMPARATORS[:-1]
M5_PHASE3_COMPLETION_RULE = (
    "Descriptive completion only, with no performance threshold and no acceptance, holdout, "
    "gate, or independent-confirmation eligibility. Every required comparator must run and "
    "produce exact-grid point and WRMSSE evidence; autogluon_zoo is optional."
)

M5_PHASE3_EVIDENCE_ROLE = "descriptive_public_window_checkpoint"
M5_PHASE3_ORIGIN_STATUS = "retired_from_acceptance"
M5_PHASE3_CLAIM_ELIGIBILITY = MappingProxyType({
    "phase3_checkpoint": True,
    "acceptance": False,
    "holdout": False,
    "gate": False,
    "independent_confirmation": False,
})

M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_EVIDENCE_ROLE = (
    "descriptive_retired_public_window_publication"
)
M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_ORIGIN_STATUS = "retired_from_acceptance"
M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_CLAIM_ELIGIBILITY = MappingProxyType({
    "public_window_measurement": True,
    "acceptance": False,
    "holdout": False,
    "gate": False,
    "independent_confirmation": False,
})

M5_ACCEPTANCE_RETIREMENT_ID = "m5_phase2_acceptance_retirement_2026-08-27"
M5_ACCEPTANCE_RETIREMENT_RELATIVE_PATH = (
    "gates/registered/m5_acceptance_retirement_2026-08-27.json"
)
M5_ACCEPTANCE_RETIREMENT_BODY_CONTENT_HASH = (
    "66d656da57ca4477b3e2448949bfb3bfbe5660dea0a363925e729f219275433b"
)
M5_RETIRED_SOURCE_MANIFEST_RELATIVE_PATH = "gates/registered/manifest_m5_phase2.json"
M5_ACCEPTANCE_RETIREMENT_EFFECT = MappingProxyType({
    "phase2_gate_blocked": True,
    "phase3_acceptance_claim_blocked": True,
    "phase4_blocked": True,
    "reason": (
        "Acceptance labels must not be used to choose or approve a model. The "
        "current M5 Phase 2 manifest cannot authorize a Gate 2 or Gate 4 claim because its sole "
        "acceptance origin had already been scored before the manifest reserved it and was "
        "repeatedly used for development afterward. The Phase 3 checkpoint at that origin "
        "remains historical descriptive evidence, not an acceptance measurement."
    ),
    "unblock_requirement": (
        "Register a new manifest under a new id using acceptance labels from a genuinely "
        "untouched source. Do not move an already observed M5 origin into a replacement "
        "acceptance set."
    ),
})
M5_ACCEPTANCE_RETIREMENT = RetirementExpectation(
    retirement_id=M5_ACCEPTANCE_RETIREMENT_ID,
    retirement_relative_path=M5_ACCEPTANCE_RETIREMENT_RELATIVE_PATH,
    body_content_hash=M5_ACCEPTANCE_RETIREMENT_BODY_CONTENT_HASH,
    source_manifest_relative_path=M5_RETIRED_SOURCE_MANIFEST_RELATIVE_PATH,
    retired_origins=(M5_PUBLIC_VALIDATION_ORIGIN,),
    effect=M5_ACCEPTANCE_RETIREMENT_EFFECT,
)


def phase3_claim_contract() -> dict[str, object]:
    """Return a fresh exact no-claim classification for Phase-3-derived evidence."""
    return {
        "evidence_role": M5_PHASE3_EVIDENCE_ROLE,
        "origin_status": M5_PHASE3_ORIGIN_STATUS,
        "claim_eligibility": dict(M5_PHASE3_CLAIM_ELIGIBILITY),
        "acceptance_retirement": M5_ACCEPTANCE_RETIREMENT.reference,
    }


def retired_public_window_publication_contract() -> dict[str, object]:
    """Return the exact no-claim contract for any publication from the retired M5 window.

    This is intentionally distinct from :func:`phase3_claim_contract`: a Phase 2 score, a mixed
    measurement table, or a top-level publication is not itself a Phase 3 checkpoint merely
    because some of its descriptive evidence came from the same retired public window.
    """
    return {
        "evidence_role": M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_EVIDENCE_ROLE,
        "origin_status": M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_ORIGIN_STATUS,
        "claim_eligibility": dict(M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_CLAIM_ELIGIBILITY),
        "acceptance_retirement": M5_ACCEPTANCE_RETIREMENT.reference,
    }


def json_values_equal_exact(observed: object, expected: object) -> bool:
    """Compare JSON values recursively without Python's bool/int aliasing.

    JSON booleans and numbers are different protocol values even though Python considers
    ``True == 1`` and ``False == 0``.  Evidence validators use this helper whenever a resealed
    artifact is compared with an immutable contract.
    """
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            set(observed) == set(expected)
            and all(json_values_equal_exact(observed[key], value)
                    for key, value in expected.items())
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            json_values_equal_exact(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def phase2_manifest(eligible_grid_hashes: Mapping[str, str]) -> EvaluationManifest:
    """Build the exact governed M5 Phase-2 contract shared by producer and consumers.

    The grid digests are measured at runtime, but every other field is immutable.  Keeping the
    typed value here prevents a downstream gate or publication script from accepting a manifest
    merely because its filename and a few headline fields look familiar.
    """
    return EvaluationManifest(
        name=PHASE2_MANIFEST_NAME,
        target_estimand="sales_given_availability",
        origins=(*M5_PHASE2_DEVELOPMENT_ORIGINS, M5_PHASE2_ACCEPTANCE_ORIGIN),
        horizon_days=M5_HORIZON_DAYS,
        scoring_grain=("store", "product", "day"),
        eligible_rule="m5.exposure_from_first_sale (first sale of the product in the store)",
        zero_fill_rule="M5 is dense; an absent row is a source failure, not a zero",
        source_cutoffs={"m5": "2016-05-22T00:00:00Z"},
        clients=("m5",),
        excluded_clients={},
        encoder_days=365,
        acceptance_origins=(M5_PHASE2_ACCEPTANCE_ORIGIN,),
        eligible_grid_hashes=dict(eligible_grid_hashes),
        missing_forecast_policy="zero",
    )


def phase3_manifest(eligible_grid_sha256: str) -> dict[str, object]:
    """Build the exact descriptive public-window Phase-3 contract used at every boundary."""
    if (not isinstance(eligible_grid_sha256, str) or len(eligible_grid_sha256) != 64
            or any(char not in "0123456789abcdef" for char in eligible_grid_sha256)):
        raise ValueError("eligible_grid_sha256 must be a lowercase SHA-256 digest")
    return {
        "schema": "autoforecast.phase3-checkpoint.v2",
        "name": PHASE3_MANIFEST_NAME,
        **phase3_claim_contract(),
        "target_estimand": "sales_given_availability",
        "origin": M5_PUBLIC_VALIDATION_ORIGIN,
        "horizon_days": M5_HORIZON_DAYS,
        "scoring_grain": ["store", "product", "day"],
        "source_cutoffs": {"m5": "2016-05-22T00:00:00Z"},
        "eligible_rule": (
            "complete M5 bottom-level series x horizon-day rectangle, including pre-range zeros"
        ),
        "eligible_grid_sha256": eligible_grid_sha256,
        "missing_forecast_policy": "zero",
        "point_score": {
            "implementation": "autoforecast.score_spark.score_forecast",
            "metric": "mae",
            "missing_actual": "no_data",
            "expected_cells": M5_EXPECTED_GRID_CELLS,
        },
        "wrmsse": {
            "implementation": "autoforecast.wrmsse.score",
            "levels": [
                [] if name == "total" else name.split("+")
                for name in M5_LEVEL_CARDINALITIES
            ],
            "weight_col": "revenue",
            "weight_days": M5_HORIZON_DAYS,
            "expected_n_series_scored": dict(M5_LEVEL_CARDINALITIES),
            "expected_n_series_dropped": dict(M5_EXPECTED_ZERO_DROPS),
        },
        "comparators": {
            name: {
                "required": name in M5_PHASE3_REQUIRED_COMPARATORS,
                "run_options": (
                    {"max_epochs": 10} if name == "tft" else
                    {"time_limit": 600} if name == "autogluon_zoo" else {}
                ),
            }
            for name in M5_PHASE3_COMPARATORS
        },
    }

__all__ = [
    "M5_BOTTOM_LEVEL_SERIES",
    "M5_EXPECTED_GRID_CELLS",
    "M5_EXPECTED_ZERO_DROPS",
    "M5_HORIZON_DAYS",
    "M5_LEVEL_CARDINALITIES",
    "M5_ACCEPTANCE_RETIREMENT",
    "M5_ACCEPTANCE_RETIREMENT_BODY_CONTENT_HASH",
    "M5_ACCEPTANCE_RETIREMENT_EFFECT",
    "M5_ACCEPTANCE_RETIREMENT_ID",
    "M5_ACCEPTANCE_RETIREMENT_RELATIVE_PATH",
    "M5_PHASE2_ACCEPTANCE_ORIGIN",
    "M5_PHASE2_DEVELOPMENT_ORIGINS",
    "M5_PHASE3_CLAIM_ELIGIBILITY",
    "M5_PHASE3_COMPLETION_RULE",
    "M5_PHASE3_COMPARATORS",
    "M5_PHASE3_EVIDENCE_ROLE",
    "M5_PHASE3_ORIGIN_STATUS",
    "M5_PHASE3_REQUIRED_COMPARATORS",
    "M5_PUBLIC_VALIDATION_ORIGIN",
    "M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_CLAIM_ELIGIBILITY",
    "M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_EVIDENCE_ROLE",
    "M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_ORIGIN_STATUS",
    "M5_RETIRED_SOURCE_MANIFEST_RELATIVE_PATH",
    "PHASE2_MANIFEST_NAME",
    "PHASE3_MANIFEST_NAME",
    "phase2_manifest",
    "phase3_claim_contract",
    "phase3_manifest",
    "json_values_equal_exact",
    "retired_public_window_publication_contract",
]
