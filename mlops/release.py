"""Local verification required before a retained run can enter a release workflow."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

from mlops.provenance import ProvenanceError, SourceIdentity, verify_retained_ref
from mlops.run_bundle import load_completed_run_bundle

__all__ = ["RetainedRun", "verify_retained_run"]


@dataclass(frozen=True)
class RetainedRun:
    """A verified completed bundle whose named retained Git ref still resolves to its source."""

    run_id: str
    source_commit: str
    retained_ref: str


def _source(path: pathlib.Path) -> SourceIdentity:
    try:
        record: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError("completed bundle has no readable source identity") from error
    required = {"schema", "repository_root", "commit", "tree", "origin_url", "relevant_paths"}
    if (not isinstance(record, dict) or set(record) != required
            or record["schema"] != "mlops.source-identity.v1"):
        raise ProvenanceError("completed bundle source identity has an invalid schema")
    if not isinstance(record["relevant_paths"], list) or not all(
            isinstance(value, str) for value in record["relevant_paths"]):
        raise ProvenanceError("completed bundle source identity has invalid relevant paths")
    return SourceIdentity(
        repository_root=record["repository_root"], commit=record["commit"], tree=record["tree"],
        origin_url=record["origin_url"], relevant_paths=tuple(record["relevant_paths"]),
    )


def verify_retained_run(
    bundle_path: str | pathlib.Path, *, retained_ref: str,
) -> RetainedRun:
    """Require a completed bundle and its exact local retained-source ref before release CI acts."""
    root = pathlib.Path(bundle_path)
    completion = load_completed_run_bundle(root)
    if completion.get("status") != "completed" or not isinstance(completion.get("run_id"), str):
        raise ProvenanceError("only completed bundles can enter retained-run verification")
    source = _source(root / "provenance" / "source.json")
    verify_retained_ref(source, retained_ref)
    return RetainedRun(completion["run_id"], source.commit, retained_ref)
