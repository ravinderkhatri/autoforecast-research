"""Versioned benchmark recognition, independent of checkout fixture locations."""

import hashlib
import pathlib

# Frozen identity of the published M5 daily contract, not a dependency on a user's files.
M5_DAILY_V1_SHA256 = "aea535b6e9b7e3abd0efd663a49e9d0298fc500000a33f4dc291d75d1379575e"


def is_m5_daily(path: pathlib.Path) -> bool:
    """Recognize the historical contract even when copied into an external project."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() == M5_DAILY_V1_SHA256:
        return True
    from autoforecast._exact_yaml import loads_exact_yaml
    value = loads_exact_yaml(payload)
    return (isinstance(value, dict) and isinstance(value.get("dataset"), dict)
            and value["dataset"].get("id") == "m5_daily")
