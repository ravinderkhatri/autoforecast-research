"""Direct adversarial checks for the shared governed M5 workflow contracts."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.benchmarks.m5_contract import (
    M5_ACCEPTANCE_RETIREMENT,
    M5_EXPECTED_GRID_CELLS,
    M5_EXPECTED_ZERO_DROPS,
    M5_LEVEL_CARDINALITIES,
    M5_PHASE2_ACCEPTANCE_ORIGIN,
    M5_PHASE2_DEVELOPMENT_ORIGINS,
    M5_PHASE3_CLAIM_ELIGIBILITY,
    M5_PHASE3_COMPARATORS,
    M5_PHASE3_EVIDENCE_ROLE,
    M5_PHASE3_ORIGIN_STATUS,
    M5_PHASE3_REQUIRED_COMPARATORS,
    M5_PUBLIC_VALIDATION_ORIGIN,
    M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_CLAIM_ELIGIBILITY,
    M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_EVIDENCE_ROLE,
    M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_ORIGIN_STATUS,
    PHASE2_MANIFEST_NAME,
    PHASE3_MANIFEST_NAME,
    json_values_equal_exact,
    phase2_manifest,
    phase3_manifest,
    retired_public_window_publication_contract,
)


def test_phase2_manifest_freezes_every_origin_and_semantic_field() -> None:
    origins = (*M5_PHASE2_DEVELOPMENT_ORIGINS, M5_PHASE2_ACCEPTANCE_ORIGIN)
    hashes = {origin: f"{index:x}" * 64 for index, origin in enumerate(origins, start=1)}
    manifest = phase2_manifest(hashes)
    assert manifest.name == PHASE2_MANIFEST_NAME
    assert manifest.origins == origins
    assert manifest.acceptance_origins == (M5_PHASE2_ACCEPTANCE_ORIGIN,)
    assert manifest.eligible_grid_hashes == hashes
    assert manifest.missing_forecast_policy == "zero"
    assert manifest.source_cutoffs == {"m5": "2016-05-22T00:00:00Z"}


def test_phase3_manifest_freezes_population_budgets_and_grid_identity() -> None:
    digest = "a" * 64
    manifest = phase3_manifest(digest)
    assert manifest["name"] == PHASE3_MANIFEST_NAME
    assert manifest["schema"] == "autoforecast.phase3-checkpoint.v2"
    assert manifest["origin"] == M5_PUBLIC_VALIDATION_ORIGIN
    assert manifest["evidence_role"] == M5_PHASE3_EVIDENCE_ROLE
    assert manifest["origin_status"] == M5_PHASE3_ORIGIN_STATUS
    assert manifest["claim_eligibility"] == dict(M5_PHASE3_CLAIM_ELIGIBILITY)
    assert manifest["acceptance_retirement"] == M5_ACCEPTANCE_RETIREMENT.reference
    assert manifest["eligible_grid_sha256"] == digest
    assert manifest["point_score"]["expected_cells"] == M5_EXPECTED_GRID_CELLS
    assert manifest["wrmsse"]["expected_n_series_scored"] == M5_LEVEL_CARDINALITIES
    assert manifest["wrmsse"]["expected_n_series_dropped"] == M5_EXPECTED_ZERO_DROPS
    assert tuple(manifest["comparators"]) == M5_PHASE3_COMPARATORS
    required = tuple(
        name for name, record in manifest["comparators"].items() if record["required"])
    assert required == M5_PHASE3_REQUIRED_COMPARATORS


def test_phase3_manifest_refuses_a_truncated_or_non_hex_grid_identity() -> None:
    for malformed in ("a" * 63, "A" * 64, "g" * 64, "", True):
        try:
            phase3_manifest(malformed)
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError(f"accepted malformed Phase-3 grid digest {malformed!r}")


def test_json_values_equal_exact_refuses_bool_int_and_float_aliases() -> None:
    """`True == 1` in Python; a resealed artifact must not use that to pass a boolean bar."""
    same = {"a": [1, 2.5, True, None, "x", {"b": False}]}
    assert json_values_equal_exact(same, json.loads(json.dumps(same)))
    for observed, expected in (
        (1, True), (True, 1), (0, False), (False, 0), (1, 1.0), (1.0, 1), (None, False),
        ("1", 1), ({"a": 1}, {"a": 1, "b": 2}), ({"a": 1, "b": 2}, {"a": 1}),
        ([1, 2], [1]), ([1], [1, 2]), ({"a": [1]}, {"a": [True]}), ([{"a": 1}], [{"a": 1.0}]),
    ):
        assert not json_values_equal_exact(observed, expected), (observed, expected)
    assert not json_values_equal_exact(float("nan"), float("nan"))


def test_retired_publication_contract_is_generic_and_distinct_from_phase3() -> None:
    contract = retired_public_window_publication_contract()
    assert contract == {
        "evidence_role": M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_EVIDENCE_ROLE,
        "origin_status": M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_ORIGIN_STATUS,
        "claim_eligibility": dict(M5_RETIRED_PUBLIC_WINDOW_PUBLICATION_CLAIM_ELIGIBILITY),
        "acceptance_retirement": M5_ACCEPTANCE_RETIREMENT.reference,
    }
    assert "phase3_checkpoint" not in contract["claim_eligibility"]
    assert contract["claim_eligibility"]["public_window_measurement"] is True
    assert all(contract["claim_eligibility"][field] is False for field in (
        "acceptance", "holdout", "gate", "independent_confirmation",
    ))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
