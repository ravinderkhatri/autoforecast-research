"""Dependency-free precision policy shared by runners and research identity checks."""

from __future__ import annotations

TRAINABLE_TORCH_ADAPTERS = frozenset({
    "mft",
    "sft",
    "hmfnet",
    "pact",
    "tft",
    "tft_q",
    "tft_hnb",

})


def effective_precision(adapter: str, requested: str) -> str:
    if requested not in ("32-true", "16-mixed"):
        raise ValueError("requested precision must be 32-true or 16-mixed")
    return requested if adapter in TRAINABLE_TORCH_ADAPTERS else "native"
