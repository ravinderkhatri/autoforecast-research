"""Local, source-controlled model-promotion records.

Promotion is an explicit human approval after held-out evidence.  It does not deploy a model or
ask MLflow which run won; it writes a reviewable record whose inputs are verified first.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from mlops.provenance import ProvenanceError
from mlops.release import RetainedRun, verify_retained_run

__all__ = ["Promotion", "PromotionError", "promote_completed_run"]

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class PromotionError(ValueError):
    """A requested local promotion lacks verifiable held-out evidence."""


@dataclass(frozen=True)
class Promotion:
    """One immutable approval record stored in source control."""

    path: pathlib.Path
    model_id: str
    alias: str
    retained: RetainedRun
    heldout_metric: str
    heldout_value: float


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PromotionError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{path} must contain a JSON object")
    return value


def _canonical_regular_file(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PromotionError(f"{label} is missing or unreadable: {path}") from error
    if path != resolved or not resolved.is_file():
        raise PromotionError(f"{label} must use its canonical regular-file locator")
    return resolved


def _canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PromotionError(f"{label} is missing or unreadable: {path}") from error
    if path != resolved or not resolved.is_dir():
        raise PromotionError(f"{label} must use its canonical directory locator")
    return resolved


def _read_heldout_evidence(
    path: pathlib.Path,
) -> tuple[bytes, dict[str, Any], str, float]:
    """Read canonical held-out bytes and return their validated metric."""
    if path.name != "heldout.json" or path.parent.name != "evaluation":
        raise PromotionError("held-out evidence must use evaluation/heldout.json")
    path = _canonical_regular_file(path, "held-out evidence")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as error:
        raise PromotionError(f"cannot read held-out evidence: {path}") from error
    if not isinstance(value, dict):
        raise PromotionError("held-out evidence must be a JSON object")
    if value.get("schema") != "autoforecast.heldout-evaluation.v1":
        raise PromotionError("promotion requires a governed evaluation/heldout.json artifact")
    metric = value.get("objective")
    raw_metric_value = value.get("value")
    if (not isinstance(metric, str) or not metric
            or not isinstance(raw_metric_value, int | float)
            or isinstance(raw_metric_value, bool)):
        raise PromotionError("held-out evaluation must contain a named finite objective")
    try:
        metric_value = float(raw_metric_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PromotionError("held-out evaluation must contain a named finite objective") from error
    if not math.isfinite(metric_value):
        raise PromotionError("held-out evaluation must contain a named finite objective")
    return raw, value, metric, metric_value


def _approver(repository: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "config", "user.email"],
        capture_output=True, text=True, check=False,
    )
    identity = result.stdout.strip()
    if result.returncode or not identity:
        raise PromotionError("Git user.email must be configured before recording a promotion")
    return identity


def _governed_bundle_for_heldout_evidence(
    root: pathlib.Path, heldout: dict[str, Any]
) -> pathlib.Path:
    """Return the sealed training bundle that produced direct or HPO held-out evidence.

    A normal evaluation stores its held-out result in its training bundle.  HPO deliberately
    stores it in a separate immutable evaluation attempt so selection cannot mutate a validation
    attempt.  The latter records the selected trial ID; that trial's governed-run record then
    names the sealed bundle to verify.
    """
    if "selected_source" not in heldout:
        return root
    selected = heldout["selected_source"]
    if not isinstance(selected, dict):
        raise PromotionError("HPO held-out evidence has malformed selected_source")
    trial_id = selected.get("trial_id")
    attempt_id = selected.get("attempt_id")
    if (not isinstance(trial_id, str) or _ID.fullmatch(trial_id) is None
            or not isinstance(attempt_id, str) or _ID.fullmatch(attempt_id) is None):
        raise PromotionError("HPO selected trial and attempt must use canonical identifiers")
    if len(root.parents) < 6:
        raise PromotionError("HPO held-out evidence is not inside a canonical study hierarchy")
    study_root = root.parents[4]
    if (
        root.parent.name != "attempts"
        or root.parents[1].name != trial_id
        or root.parents[2].name != "trials"
        or root.parents[3].name != "heldout"
        or root.parents[5].name != "studies"
        or _ID.fullmatch(root.name) is None
        or _ID.fullmatch(study_root.name) is None
    ):
        raise PromotionError("HPO held-out evidence is not inside a canonical study hierarchy")
    source_attempt = study_root / "trials" / trial_id / "attempts" / attempt_id
    source_attempt = _canonical_directory(source_attempt, "selected HPO source attempt")
    governed_path = _canonical_regular_file(
        source_attempt / "evaluation" / "governed-run.json", "selected HPO governed-run record",
    )
    governed = _read_json(governed_path)
    if governed.get("schema") != "autoforecast.hpo-governed-run.v1":
        raise PromotionError("selected HPO trial has no governed-run record")
    relative_bundle = governed.get("source_bundle")
    if not isinstance(relative_bundle, str):
        raise PromotionError("selected HPO trial has no source bundle path")
    relative = pathlib.PurePosixPath(relative_bundle)
    if relative.is_absolute() or ".." in relative.parts or relative_bundle in {"", "."}:
        raise PromotionError("selected HPO trial has an unsafe source bundle path")
    bundle = source_attempt.joinpath(*relative.parts)
    return _canonical_directory(bundle, "selected HPO trial source bundle")


def promote_completed_run(
    bundle_path: str | pathlib.Path,
    *,
    retained_ref: str,
    model_id: str,
    alias: str,
    record_root: str | pathlib.Path,
    repository: str | pathlib.Path,
) -> Promotion:
    """Verify a completed bundle and write one non-overwritable approval record.

    Direct training bundles must include an explicit ``evaluation/heldout.json`` artifact.  An HPO
    held-out evaluation attempt may instead point back to its selected trial's sealed source
    bundle.  In either case, validation-only evidence cannot become champion evidence.
    """
    if _ID.fullmatch(model_id) is None or _ID.fullmatch(alias) is None:
        raise PromotionError("model_id and alias must use lowercase letters, digits, _ or -")
    root = pathlib.Path(bundle_path).resolve()
    evidence_path = root / "evaluation" / "heldout.json"
    _, heldout, metric, value = _read_heldout_evidence(evidence_path)
    bundle = _governed_bundle_for_heldout_evidence(root, heldout)
    try:
        retained = verify_retained_run(bundle, retained_ref=retained_ref)
    except ProvenanceError as error:
        raise PromotionError(str(error)) from error
    destination = pathlib.Path(record_root) / model_id / f"{alias}.json"
    if destination.exists():
        raise PromotionError(f"promotion record already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "autoforecast.local-promotion.v1",
        "model_id": model_id,
        "alias": alias,
        "approver": _approver(pathlib.Path(repository)),
        "retained_ref": retained.retained_ref,
        "source_commit": retained.source_commit,
        "run_id": retained.run_id,
        "bundle": {
            "path": str(bundle.resolve()),
            "completion_sha256": hashlib.sha256(
                (bundle / "completion.json").read_bytes()
            ).hexdigest(),
        },
        "heldout_evidence_path": str(evidence_path),
        "heldout": {"metric": metric, "value": value},
        "lifecycle": "approved_local_alias_not_deployed",
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = pathlib.Path(stream.name)
        try:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        # Same-filesystem hard-link publication is atomic and refuses any existing alias.
        # test_promote_completed_run_never_replaces_a_concurrent_approval exercises the race.
        os.link(temporary, destination)
    except FileExistsError as error:
        raise PromotionError(f"promotion record already exists: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return Promotion(destination, model_id, alias, retained, metric, value)
