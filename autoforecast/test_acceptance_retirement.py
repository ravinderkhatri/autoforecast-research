"""Direct adversarial tests for the shared pre-compute retirement authenticator."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.acceptance_retirement import (  # noqa: E402
    authenticate_retirement,
    canonical_json_sha256,
)
from autoforecast.benchmarks.m5_contract import (  # noqa: E402
    M5_ACCEPTANCE_RETIREMENT,
    M5_ACCEPTANCE_RETIREMENT_RELATIVE_PATH,
    M5_RETIRED_SOURCE_MANIFEST_RELATIVE_PATH,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES_AVAILABLE = all((ROOT / relative).is_file() for relative in (
    M5_ACCEPTANCE_RETIREMENT_RELATIVE_PATH,
    M5_RETIRED_SOURCE_MANIFEST_RELATIVE_PATH,
))
pytestmark = pytest.mark.skipif(
    not FIXTURES_AVAILABLE, reason="historical retirement and source manifest are not supplied"
)
RETIREMENT_BYTES = (
    (ROOT / M5_ACCEPTANCE_RETIREMENT_RELATIVE_PATH).read_bytes() if FIXTURES_AVAILABLE else b""
)
MANIFEST_BYTES = (
    (ROOT / M5_RETIRED_SOURCE_MANIFEST_RELATIVE_PATH).read_bytes() if FIXTURES_AVAILABLE else b""
)


def _fixtures(directory: str) -> tuple[pathlib.Path, pathlib.Path]:
    root = pathlib.Path(directory)
    retirement = root / "retirement.json"
    manifest = root / "manifest.json"
    retirement.write_bytes(RETIREMENT_BYTES)
    manifest.write_bytes(MANIFEST_BYTES)
    return retirement, manifest


def _authenticate(retirement: pathlib.Path, manifest: pathlib.Path) -> dict[str, object]:
    return authenticate_retirement(retirement, manifest, M5_ACCEPTANCE_RETIREMENT)


def _refuses(retirement: pathlib.Path, manifest: pathlib.Path, expected: str) -> None:
    try:
        _authenticate(retirement, manifest)
    except RuntimeError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(f"retirement authenticator accepted mutation: {expected}")


def _rewrite(retirement: pathlib.Path, mutate) -> None:
    envelope = json.loads(RETIREMENT_BYTES)
    mutate(envelope["retirement"])
    envelope["hash"] = canonical_json_sha256(envelope["retirement"])
    retirement.write_text(json.dumps(envelope))


def test_authenticates_registered_body_and_exact_source_manifest_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        retirement, manifest = _fixtures(directory)
        body = _authenticate(retirement, manifest)
    assert body["retirement_id"] == M5_ACCEPTANCE_RETIREMENT.retirement_id
    assert M5_ACCEPTANCE_RETIREMENT.reference["body_content_hash"] \
        == canonical_json_sha256(body)


def test_resealed_identity_origin_and_effect_mutations_are_refused() -> None:
    cases = (
        (lambda body: body.update(retirement_id="other"), "retirement id"),
        (lambda body: body["manifest"].update(path="other.json"), "different manifest path"),
        (lambda body: body["manifest"].update(retired_acceptance_origins=[]),
         "exact registered origin set"),
        (lambda body: body["effect"].update(phase2_gate_blocked=False),
         "phase2_gate_blocked as true"),
        (lambda body: body["effect"].update(phase3_acceptance_claim_blocked=False),
         "phase3_acceptance_claim_blocked as true"),
        (lambda body: body["effect"].update(phase4_blocked=False),
         "phase4_blocked as true"),
        (lambda body: body["effect"].update(reason=""), "effect.reason"),
        (lambda body: body["effect"].update(reason=7), "effect.reason"),
        (lambda body: body["effect"].pop("reason"), "unexpected fields"),
        (lambda body: body["effect"].update(unblock_requirement=""),
         "effect.unblock_requirement"),
        (lambda body: body["effect"].update(unblock_requirement=[]),
         "effect.unblock_requirement"),
        (lambda body: body["effect"].pop("unblock_requirement"), "unexpected fields"),
    )
    with tempfile.TemporaryDirectory() as directory:
        retirement, manifest = _fixtures(directory)
        for mutate, expected in cases:
            _rewrite(retirement, mutate)
            _refuses(retirement, manifest, expected)


def test_self_resealed_unregistered_body_is_not_authoritative() -> None:
    with tempfile.TemporaryDirectory() as directory:
        retirement, manifest = _fixtures(directory)
        _rewrite(retirement, lambda body: body.update(retired_on="2099-01-01"))
        _refuses(retirement, manifest, "registered body hash")


def test_source_manifest_wrapper_payload_and_raw_bytes_are_all_bound() -> None:
    with tempfile.TemporaryDirectory() as directory:
        retirement, manifest = _fixtures(directory)
        wrapper = json.loads(MANIFEST_BYTES)
        wrapper["manifest"]["encoder_days"] += 1
        manifest.write_text(json.dumps(wrapper))
        _refuses(retirement, manifest, "wrapper hash")

        wrapper["hash"] = canonical_json_sha256(wrapper["manifest"])
        manifest.write_text(json.dumps(wrapper))
        _refuses(retirement, manifest, "payload hash")

        manifest.write_bytes(MANIFEST_BYTES + b"\n")
        _refuses(retirement, manifest, "raw file hash")


def test_duplicate_keys_are_refused_in_both_registered_authority_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        retirement, manifest = _fixtures(directory)
        retirement.write_text(
            RETIREMENT_BYTES.decode().replace('"hash":', '"hash":"ignored","hash":', 1)
        )
        _refuses(retirement, manifest, "duplicate JSON object key 'hash'")

        retirement.write_bytes(RETIREMENT_BYTES)
        manifest.write_text(
            MANIFEST_BYTES.decode().replace(
                '"manifest": {', '"manifest": {"name":"ignored",', 1
            )
        )
        _refuses(retirement, manifest, "duplicate JSON object key 'name'")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
