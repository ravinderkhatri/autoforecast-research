"""Standalone model identity admission."""

from autoforecast.benchmarks.m5_timesfm_contract import (
    TIMESFM_1P0_MODEL_ID,
    TIMESFM_2P5_MODEL_ID,
    validate_timesfm_model_policy,
)


def _refuses(call, message: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted invalid TimesFM evidence: {message}")


def test_validate_timesfm_model_policy_allows_only_pinned_identities() -> None:
    validate_timesfm_model_policy(TIMESFM_1P0_MODEL_ID)
    validate_timesfm_model_policy(TIMESFM_2P5_MODEL_ID)
    for model_id in ("timesfm", "", True, None, 1, "google/timesfm-latest"):
        _refuses(
            lambda model_id=model_id: validate_timesfm_model_policy(model_id),
            "not permitted",
        )
