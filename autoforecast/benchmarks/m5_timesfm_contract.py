"""Pinned standalone TimesFM model identities and admission policy."""

TIMESFM_1P0_MODEL_ID = "google/timesfm-1.0-200m-pytorch"

TIMESFM_2P5_MODEL_ID = "google/timesfm-2.5-200m-pytorch"

TIMESFM_2P5_HF_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"

TIMESFM_2P5_CONTEXT_DAYS = 1024

TIMESFM_2P5_BATCH_SIZE = 32

def validate_timesfm_model_policy(model_id: object) -> None:
    """Refuse floating aliases and every unregistered model identity."""
    if type(model_id) is not str or model_id not in {
        TIMESFM_1P0_MODEL_ID,
        TIMESFM_2P5_MODEL_ID,
    }:
        raise ValueError("TimesFM model identity is not permitted by the standalone contract")
