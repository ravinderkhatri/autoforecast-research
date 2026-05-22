"""Strict JSON reads for claim-bearing and reusable artifacts."""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Iterable
from typing import Literal


class ExactJSONError(ValueError):
    """A JSON input could not be read or had an ambiguous representation."""

    def __init__(
        self,
        message: str,
        *,
        kind: Literal["read", "decode", "duplicate"],
        cause_type_name: str,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.cause_type_name = cause_type_name


def _object_without_duplicate_keys(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExactJSONError(
                f"duplicate JSON object key {key!r}",
                kind="duplicate",
                cause_type_name="ExactJSONError",
            )
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> object:
    raise ExactJSONError(
        f"non-finite JSON number {value!r}",
        kind="decode",
        cause_type_name="ExactJSONError",
    )


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExactJSONError(
            f"non-finite JSON number {value!r}",
            kind="decode",
            cause_type_name="ExactJSONError",
        )
    return parsed


def loads_exact_json(payload: str | bytes | bytearray) -> object:
    """Parse one JSON value while rejecting duplicate keys at every object depth."""
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
            parse_float=_finite_float,
        )
    except ExactJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactJSONError(str(exc), kind="decode", cause_type_name=type(exc).__name__) from exc


def load_exact_json(path: pathlib.Path) -> object:
    """Read and strictly parse one UTF-encoded JSON file."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ExactJSONError(str(exc), kind="read", cause_type_name=type(exc).__name__) from exc
    return loads_exact_json(payload)


__all__ = ["ExactJSONError", "load_exact_json", "loads_exact_json"]
