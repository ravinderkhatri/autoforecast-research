"""Architecture names and the retained recipe wire identifiers.

``test_model_names`` covers these mappings; format versions are not model names.
"""

MODEL_NAMES = {
    "mft": "MFT",
    "mft_corrected": "MFT (corrected)",
    "sft": "SFT",
    "hmfnet": "HMFNet",
    "pact": "PACT",
    "tft_q": "TFT-Q",
    "tft_hnb": "TFT-HNB",
    "timesfm_hnb": "TimesFM-HNB",
}

# These values belong to existing recipe/checkpoint formats, not public catalog IDs.
REVISION_VARIANTS = {
    "mft_corrected": "v1_corrected",
    "sft": "v2",
    "hmfnet": "v3",
    "pact": "v5",
}
REVISION_MODELS = {variant: name for name, variant in REVISION_VARIANTS.items()}


def model_name(identifier: str) -> str:
    """Format a public or retained recipe ID; covered by ``test_model_names``."""
    return MODEL_NAMES.get(REVISION_MODELS.get(identifier, identifier), identifier)
