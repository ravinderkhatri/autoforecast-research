"""Local, immutable batch-deployment hand-off records.

This module deliberately does not start an HTTP service.  It converts an approved local promotion
into a reproducible batch-forecast hand-off: retained source ref, held-out evidence, exact small
configuration files, and data identity.  A separate serving owner can consume this descriptor
without treating a mutable model file or dashboard alias as deployment evidence.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from mlops.promotion import (
    PromotionError,
    _governed_bundle_for_heldout_evidence,
    _read_heldout_evidence,
)
from mlops.provenance import ProvenanceError
from mlops.release import verify_retained_run

__all__ = ["Deployment", "DeploymentError", "create_local_batch_deployment"]

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class DeploymentError(ValueError):
    """A local promotion cannot become a reproducible batch deployment hand-off."""


@dataclass(frozen=True)
class Deployment:
    """One immutable local batch deployment descriptor and its frozen configurations."""

    path: pathlib.Path
    model_id: str
    alias: str
    retained_ref: str
    source_commit: str


def _read_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DeploymentError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise DeploymentError(f"{label} must contain a JSON object")
    return value


def _git(repository: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise DeploymentError(result.stderr.strip() or result.stdout.strip() or "Git failed")
    return result.stdout.strip()


def _configurations(bundle: pathlib.Path, destination: pathlib.Path) -> list[dict[str, str]]:
    source = bundle / "provenance" / "config"
    if not source.is_dir():
        raise DeploymentError("completed bundle has no frozen configuration directory")
    records: list[dict[str, str]] = []
    for path in sorted(source.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise DeploymentError(f"frozen configuration is not a regular file: {path}")
        target = destination / path.name
        shutil.copyfile(path, target)
        records.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    if not records:
        raise DeploymentError("completed bundle has no frozen configuration files")
    return records


def _heldout_evidence(promotion: dict[str, Any], bundle: pathlib.Path) -> dict[str, str]:
    """Bind current held-out bytes and the approved metric to the selected training bundle.

    Tested by test_create_local_batch_deployment_rejects_changed_heldout_evidence and the
    direct/HPO hand-off tests. This is not an approval-time digest or a signed attestation.
    """
    raw_path = promotion["heldout_evidence_path"]
    if not isinstance(raw_path, str) or not pathlib.Path(raw_path).is_absolute():
        raise DeploymentError("promotion has an invalid held-out evidence path")
    path = pathlib.Path(raw_path)
    try:
        raw, evidence, metric, value = _read_heldout_evidence(path)
        root = path.parent.parent
        source = _governed_bundle_for_heldout_evidence(root, evidence)
    except PromotionError as error:
        raise DeploymentError(f"cannot validate held-out evidence: {error}") from error
    if promotion["heldout"] != {"metric": metric, "value": value}:
        raise DeploymentError("held-out evidence no longer matches the approved metric")
    if source != bundle:
        raise DeploymentError("held-out evidence belongs to a different training bundle")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def create_local_batch_deployment(
    promotion_path: str | pathlib.Path,
    *,
    deployment_root: str | pathlib.Path,
    repository: str | pathlib.Path,
) -> Deployment:
    """Create one non-overwritable, local batch-deployment descriptor from a promotion.

    The descriptor is intentionally inert.  It makes a reviewed hand-off executable by a future
    batch runner while keeping deployment/serving policy outside the research repository.
    """
    promotion_file = pathlib.Path(promotion_path).resolve()
    promotion = _read_object(promotion_file, "promotion record")
    required = {
        "schema",
        "model_id",
        "alias",
        "approver",
        "retained_ref",
        "source_commit",
        "run_id",
        "bundle",
        "heldout",
        "heldout_evidence_path",
        "lifecycle",
    }
    if set(promotion) != required or promotion.get("schema") != "autoforecast.local-promotion.v1":
        raise DeploymentError("promotion record has an unsupported schema")
    model_id, alias = promotion["model_id"], promotion["alias"]
    if (
        not isinstance(model_id, str)
        or not isinstance(alias, str)
        or not (_ID.fullmatch(model_id) and _ID.fullmatch(alias))
    ):
        raise DeploymentError("promotion record has invalid model_id or alias")
    if promotion.get("lifecycle") != "approved_local_alias_not_deployed":
        raise DeploymentError("only an approved, not-yet-deployed promotion may be handed off")
    bundle_info = promotion["bundle"]
    if not isinstance(bundle_info, dict) or set(bundle_info) != {"path", "completion_sha256"}:
        raise DeploymentError("promotion record has invalid bundle identity")
    raw_bundle_path = bundle_info["path"]
    if (not isinstance(raw_bundle_path, str)
            or not isinstance(bundle_info["completion_sha256"], str)):
        raise DeploymentError("promotion record points to an unavailable local bundle")
    try:
        bundle_path = pathlib.Path(raw_bundle_path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DeploymentError("promotion record points to an unavailable local bundle") from error
    if not bundle_path.is_dir():
        raise DeploymentError("promotion record points to an unavailable local bundle")
    try:
        completion_hash = hashlib.sha256((bundle_path / "completion.json").read_bytes()).hexdigest()
    except OSError as error:
        raise DeploymentError("promotion bundle has no readable completion artifact") from error
    if completion_hash != bundle_info["completion_sha256"]:
        raise DeploymentError("promotion bundle completion artifact no longer matches its approval")
    heldout_evidence = _heldout_evidence(promotion, bundle_path)
    retained_ref, source_commit = promotion["retained_ref"], promotion["source_commit"]
    if not isinstance(retained_ref, str) or not isinstance(source_commit, str):
        raise DeploymentError("promotion record has invalid retained source identity")
    try:
        retained = verify_retained_run(bundle_path, retained_ref=retained_ref)
    except ProvenanceError as error:
        raise DeploymentError(str(error)) from error
    if retained.source_commit != source_commit:
        raise DeploymentError("promotion source commit differs from verified bundle source")
    repo = pathlib.Path(repository).resolve()
    if _git(repo, "rev-parse", f"{retained_ref}^{{commit}}") != source_commit:
        raise DeploymentError("retained source ref no longer resolves to the promoted commit")
    data = _read_object(bundle_path / "provenance" / "data.json", "bundle data identity")
    destination = pathlib.Path(deployment_root) / model_id / alias
    if destination.exists():
        raise DeploymentError(f"local deployment already exists: {destination}")
    temporary = destination.with_name(f".{alias}.tmp")
    temporary.mkdir(parents=True)
    try:
        config = temporary / "config"
        config.mkdir()
        configuration_files = _configurations(bundle_path, config)
        payload = {
            "schema": "autoforecast.local-batch-deployment.v1",
            "mode": "local_batch_forecast",
            "model_id": model_id,
            "alias": alias,
            "promotion": {
                "path": str(promotion_file),
                "sha256": hashlib.sha256(promotion_file.read_bytes()).hexdigest(),
                "approver": promotion["approver"],
            },
            "source": {"retained_ref": retained_ref, "commit": source_commit},
            "heldout": promotion["heldout"],
            "heldout_evidence": heldout_evidence,
            "data_snapshot": data,
            "configuration_files": configuration_files,
            "next_action": "run an approved batch-forecast executor at source.retained_ref",
            "lifecycle": "local_batch_hand_off_not_online_serving",
        }
        (temporary / "deployment.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Deployment(destination, model_id, alias, retained_ref, source_commit)
