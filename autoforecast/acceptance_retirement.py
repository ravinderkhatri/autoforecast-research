"""Fail-closed authentication for an acceptance-origin retirement record.

This module is deliberately stdlib-only so runners can authenticate a retirement before loading
Spark, resource profilers, or model adapters.  A self-consistent JSON hash is not authority: the
expected registered body hash and the exact bytes of the retired source manifest are both bound.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass

from autoforecast._exact_json import ExactJSONError, loads_exact_json


def canonical_json_sha256(value: object) -> str:
    """Return the repository's canonical JSON SHA-256 using only the stdlib."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class RetirementExpectation:
    """Immutable identity and semantics of one registered retirement."""

    retirement_id: str
    retirement_relative_path: str
    body_content_hash: str
    source_manifest_relative_path: str
    retired_origins: tuple[str, ...]
    effect: Mapping[str, object]

    @property
    def reference(self) -> dict[str, str]:
        """The exact reference persisted by descriptive downstream evidence."""
        return {
            "retirement_id": self.retirement_id,
            "path": self.retirement_relative_path,
            "body_content_hash": self.body_content_hash,
        }


def authenticate_retirement(
    retirement_path: pathlib.Path,
    source_manifest_path: pathlib.Path,
    expectation: RetirementExpectation,
) -> dict[str, object]:
    """Authenticate one retirement and its byte-exact retired manifest.

    The caller supplies resolved paths so tests can exercise every branch without touching the
    repository.  The expectation still carries the registered relative paths; substituting a
    different file with a valid-looking envelope cannot change the contract it must satisfy.
    """
    try:
        retirement_bytes = retirement_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"acceptance retirement record is unreadable: {exc}") from exc
    try:
        envelope = loads_exact_json(retirement_bytes)
    except ExactJSONError as exc:
        raise RuntimeError(f"acceptance retirement record is not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"retirement", "hash"}:
        raise RuntimeError(
            "acceptance retirement envelope must contain exactly 'retirement' and 'hash'")
    retirement = envelope["retirement"]
    stored_hash = envelope["hash"]
    if not isinstance(retirement, dict) or not isinstance(stored_hash, str):
        raise RuntimeError("acceptance retirement envelope has invalid body/hash types")
    if canonical_json_sha256(retirement) != stored_hash:
        raise RuntimeError("acceptance retirement body does not match its content hash")
    if retirement.get("retirement_id") != expectation.retirement_id:
        raise RuntimeError("acceptance retirement id is not the registered retirement")

    manifest_record = retirement.get("manifest")
    if not isinstance(manifest_record, dict):
        raise RuntimeError("acceptance retirement has no valid manifest binding")
    if set(manifest_record) != {
        "path", "payload_sha256", "file_sha256", "retired_acceptance_origins",
    }:
        raise RuntimeError("acceptance retirement manifest binding has unexpected fields")
    if manifest_record.get("path") != expectation.source_manifest_relative_path:
        raise RuntimeError(
            "acceptance retirement names a different manifest path than the registered source "
            "manifest")
    retired_origins = manifest_record.get("retired_acceptance_origins")
    if retired_origins != list(expectation.retired_origins):
        raise RuntimeError(
            "acceptance retirement does not retire current exact registered origin set")

    effect = retirement.get("effect")
    expected_effect = dict(expectation.effect)
    if not isinstance(effect, dict):
        raise RuntimeError(
            "acceptance retirement must declare phase2_gate_blocked as true, "
            "phase3_acceptance_claim_blocked as true, and phase4_blocked as true")
    if set(effect) != set(expected_effect):
        raise RuntimeError("acceptance retirement effect has unexpected fields")
    for field_name, expected_value in expected_effect.items():
        if effect.get(field_name) == expected_value:
            continue
        if expected_value is True:
            raise RuntimeError(
                f"acceptance retirement must declare {field_name} as true")
        raise RuntimeError(
            f"acceptance retirement effect.{field_name} does not match the registered contract")
    if stored_hash != expectation.body_content_hash:
        raise RuntimeError("acceptance retirement body hash is not the registered body hash")

    try:
        manifest_bytes = source_manifest_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"retired source manifest is unreadable: {exc}") from exc
    try:
        manifest_envelope = loads_exact_json(manifest_bytes)
    except ExactJSONError as exc:
        raise RuntimeError(f"retired source manifest is not valid JSON: {exc}") from exc
    if (not isinstance(manifest_envelope, dict)
            or set(manifest_envelope) != {"manifest", "hash"}):
        raise RuntimeError(
            "retired source manifest envelope must contain exactly 'manifest' and 'hash'")
    manifest_payload = manifest_envelope["manifest"]
    manifest_stored_hash = manifest_envelope["hash"]
    if not isinstance(manifest_payload, dict) or not isinstance(manifest_stored_hash, str):
        raise RuntimeError("retired source manifest has invalid payload/hash types")
    current_payload_hash = canonical_json_sha256(manifest_payload)
    if manifest_stored_hash != current_payload_hash:
        raise RuntimeError("retired source manifest payload does not match its wrapper hash")
    if manifest_record.get("payload_sha256") != current_payload_hash:
        raise RuntimeError(
            "acceptance retirement payload hash does not match the retired source manifest")
    current_file_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_record.get("file_sha256") != current_file_hash:
        raise RuntimeError(
            "acceptance retirement raw file hash does not match the retired source manifest")
    return retirement


__all__ = [
    "RetirementExpectation",
    "authenticate_retirement",
    "canonical_json_sha256",
]
