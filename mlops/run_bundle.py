"""Small immutable run bundle usable by any ML task.

It deliberately has no model, framework, Spark or forecasting dependency.  Task-specific code
writes its own artifacts, then this class seals their byte manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from mlops._safe_files import (
    SafeFileError,
    _create_fresh_directory,
    _regular_file_identity,
    copy_regular_file,
    publish_bytes,
    read_regular_bytes,
    snapshot_tree_types,
)
from mlops.provenance import ProvenanceError

__all__ = ["RunBundle", "load_completed_run_bundle", "open_run_bundle"]


def _sha256(path: pathlib.Path) -> str:
    return _file_identity(path)[0]


def _file_identity(path: pathlib.Path) -> tuple[str, int]:
    try:
        return _regular_file_identity(path)
    except SafeFileError as exc:
        raise ProvenanceError(f"cannot read regular bundle artifact: {path}") from exc


def _safe(root: pathlib.Path, relative_path: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative_path in {"", "."}:
        raise ProvenanceError("bundle path must be relative and must not contain '..'")
    return root.joinpath(*relative.parts)


def _traverses_symlink(root: pathlib.Path, path: pathlib.Path) -> bool:
    cursor = path
    while cursor != root:
        if cursor.is_symlink() or cursor.parent == cursor:
            return True
        cursor = cursor.parent
    return root.is_symlink()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _strict_json(path: pathlib.Path) -> object:
    """Read one finite JSON value and reject duplicate object keys at every depth."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ProvenanceError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def finite(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ProvenanceError(f"non-finite JSON number {value!r}: {path}")
        return parsed

    try:
        return json.loads(
            read_regular_bytes(path.parent, path.name),
            object_pairs_hook=unique,
            parse_float=finite,
            parse_constant=finite,
        )
    except (SafeFileError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"cannot read completed bundle: {path.parent}") from error


@dataclass
class RunBundle:
    """Fresh bundle with a terminal artifact manifest and completion record."""

    path: pathlib.Path
    run_id: str
    task: str
    _finalized: bool = False

    def _check_writable(self, relative_path: str) -> None:
        if self._finalized:
            raise ProvenanceError("cannot change a completed run bundle")
        _safe(self.path, relative_path)

    def write_json(self, relative_path: str, payload: Mapping[str, object]) -> pathlib.Path:
        self._check_writable(relative_path)
        try:
            return publish_bytes(self.path, relative_path, _json_bytes(payload))
        except SafeFileError as exc:
            raise ProvenanceError(f"could not write bundle artifact: {relative_path}") from exc

    def write_bytes(self, relative_path: str, payload: bytes) -> pathlib.Path:
        """Publish caller-retained immutable bytes inside this bundle."""
        self._check_writable(relative_path)
        try:
            return publish_bytes(self.path, relative_path, payload)
        except SafeFileError as exc:
            raise ProvenanceError(f"could not write bundle artifact: {relative_path}") from exc

    def copy_file(self, relative_path: str, source: str | pathlib.Path) -> pathlib.Path:
        self._check_writable(relative_path)
        source_path = pathlib.Path(source)
        try:
            return copy_regular_file(self.path, relative_path, source_path)
        except SafeFileError as exc:
            raise ProvenanceError(f"could not copy bundle source: {source_path}") from exc

    def finalize(self, *, status: str, summary: Mapping[str, object]) -> pathlib.Path:
        if type(status) is not str or status not in {"completed", "failed"}:
            raise ProvenanceError("bundle status must be completed or failed")
        if self._finalized:
            raise ProvenanceError("run bundle is already complete")
        manifest_path = self.path / "artifact_manifest.json"
        completion_path = self.path / "completion.json"
        descendants = tuple(self.path.rglob("*"))
        linked = [item for item in descendants if item.is_symlink()]
        if linked:
            raise ProvenanceError(f"bundle artifacts may not traverse symlinks: {linked[0]}")
        files = sorted(
            file
            for file in descendants
            if file.is_file() and file not in {manifest_path, completion_path}
        )
        artifacts = []
        for file in files:
            digest, size = _file_identity(file)
            artifacts.append(
                {
                    "path": file.relative_to(self.path).as_posix(),
                    "sha256": digest,
                    "bytes": size,
                }
            )
        terminal_payload = {
            "schema": "mlops.run-bundle.v2",
            "run_id": self.run_id,
            "task": self.task,
            "status": status,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "summary": dict(summary),
        }
        self.write_json(
            "artifact_manifest.json",
            {
                "schema": "mlops.artifact-manifest.v2",
                "artifacts": artifacts,
                "terminal_payload_sha256": hashlib.sha256(
                    _json_bytes(terminal_payload)
                ).hexdigest(),
            },
        )
        self.write_json(
            "completion.json",
            {
                **terminal_payload,
                "artifact_manifest_sha256": _sha256(manifest_path),
            },
        )
        self._finalized = True
        return completion_path


def open_run_bundle(
    root: str | pathlib.Path,
    *,
    run_id: str,
    task: str,
) -> RunBundle:
    """Create a fresh common bundle, with no forecast-specific required fields."""
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in run_id):
        raise ProvenanceError("run id must use lowercase letters, digits, _ or -")
    if not task or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in task):
        raise ProvenanceError("task must use lowercase letters, digits, _ or -")
    path = pathlib.Path(root) / run_id
    try:
        _create_fresh_directory(path)
    except SafeFileError as error:
        if "already exists" in str(error):
            raise ProvenanceError(f"run bundle already exists: {path}") from error
        raise ProvenanceError(f"cannot create run bundle: {path}") from error
    bundle = RunBundle(path, run_id, task)
    bundle.write_json(
        "metadata/run.json",
        {
            "schema": "mlops.run-definition.v1",
            "run_id": run_id,
            "task": task,
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return bundle


def load_completed_run_bundle(path: str | pathlib.Path) -> dict[str, Any]:
    """Verify and return a completed generic bundle.

    A completed bundle is not trusted merely because its completion file exists: its manifest,
    every declared byte, and the set of files on disk must agree.  This is deliberately generic
    so an LLM fine-tune and a forecast run receive the same tamper/corruption detection.
    """
    root = pathlib.Path(path)
    completion_path = root / "completion.json"
    manifest_path = root / "artifact_manifest.json"
    if (
        root.is_symlink()
        or _traverses_symlink(root, completion_path)
        or _traverses_symlink(root, manifest_path)
        or not completion_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ProvenanceError(f"completed bundle is missing terminal records: {root}")
    try:
        initial_types = snapshot_tree_types(root)
    except SafeFileError as error:
        raise ProvenanceError(f"cannot inspect completed bundle: {root}") from error
    linked = next((item for item, kind in initial_types.items() if kind == "symlink"), None)
    if linked is not None:
        raise ProvenanceError(f"completed bundle may not contain symlinks: {linked}")
    special = next((item for item, kind in initial_types.items() if kind == "special"), None)
    if special is not None:
        raise ProvenanceError(f"completed bundle contains an unsupported path type: {special}")
    completion = _strict_json(completion_path)
    manifest = _strict_json(manifest_path)
    if type(completion) is not dict or type(manifest) is not dict:
        raise ProvenanceError(f"completed bundle terminal records are invalid: {root}")
    if (
        completion.get("schema") == "mlops.run-bundle.v1"
        or manifest.get("schema") == "mlops.artifact-manifest.v1"
    ):
        raise ProvenanceError(
            f"legacy run bundle has unsealed terminal metadata and must be rerun: {root}"
        )
    if (
        set(completion)
        != {
            "schema",
            "run_id",
            "task",
            "status",
            "completed_at_utc",
            "artifact_manifest_sha256",
            "summary",
        }
        or set(manifest) != {"schema", "artifacts", "terminal_payload_sha256"}
        or completion.get("schema") != "mlops.run-bundle.v2"
        or manifest.get("schema") != "mlops.artifact-manifest.v2"
        or type(completion.get("run_id")) is not str
        or type(completion.get("task")) is not str
        or type(completion.get("status")) is not str
        or completion.get("status") not in {"completed", "failed"}
        or type(completion.get("completed_at_utc")) is not str
        or type(completion.get("summary")) is not dict
        or completion.get("artifact_manifest_sha256") != _sha256(manifest_path)
    ):
        raise ProvenanceError(f"completed bundle terminal records are invalid: {root}")
    terminal_payload = {
        key: value for key, value in completion.items() if key != "artifact_manifest_sha256"
    }
    if (
        manifest.get("terminal_payload_sha256")
        != hashlib.sha256(_json_bytes(terminal_payload)).hexdigest()
    ):
        raise ProvenanceError(f"completed bundle terminal payload is not sealed: {root}")
    declared = manifest.get("artifacts")
    if not isinstance(declared, list):
        raise ProvenanceError(f"completed bundle manifest is invalid: {root}")
    expected: set[str] = set()
    for entry in declared:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise ProvenanceError(f"completed bundle manifest entry is invalid: {root}")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or relative in expected
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or set(entry["sha256"]) - set("0123456789abcdef")
            or type(entry["bytes"]) is not int
            or entry["bytes"] < 0
        ):
            raise ProvenanceError(f"completed bundle manifest entry is invalid: {root}")
        target = _safe(root, relative)
        digest, size = _file_identity(target)
        if (
            _traverses_symlink(root, target)
            or not target.is_file()
            or digest != entry["sha256"]
            or size != entry["bytes"]
        ):
            raise ProvenanceError(f"completed bundle artifact failed verification: {relative}")
        expected.add(relative)
    actual = {
        item.relative_to(root).as_posix()
        for item, kind in initial_types.items()
        if kind == "file" and item not in {completion_path, manifest_path}
    }
    if actual != expected:
        raise ProvenanceError("completed bundle contains undeclared or missing artifacts")
    definition = _strict_json(root / "metadata" / "run.json")
    if (
        type(definition) is not dict
        or set(definition) != {"schema", "run_id", "task", "started_at_utc"}
        or definition.get("schema") != "mlops.run-definition.v1"
        or type(definition.get("run_id")) is not str
        or type(definition.get("task")) is not str
        or type(definition.get("started_at_utc")) is not str
        or definition.get("run_id") != completion.get("run_id")
        or definition.get("task") != completion.get("task")
    ):
        raise ProvenanceError(f"completed bundle identity differs from its run definition: {root}")
    try:
        final_types = snapshot_tree_types(root)
    except SafeFileError as error:
        raise ProvenanceError(f"cannot inspect completed bundle: {root}") from error
    final_link = next((item for item, kind in final_types.items() if kind == "symlink"), None)
    if final_link is not None:
        raise ProvenanceError(f"completed bundle may not contain symlinks: {final_link}")
    final_special = next((item for item, kind in final_types.items() if kind == "special"), None)
    if final_special is not None:
        raise ProvenanceError(
            f"completed bundle contains an unsupported path type: {final_special}"
        )
    if final_types != initial_types:
        raise ProvenanceError("completed bundle path set or type changed while it was verified")
    if _strict_json(completion_path) != completion or _strict_json(manifest_path) != manifest:
        raise ProvenanceError("completed bundle terminal records changed while verified")
    if completion.get("artifact_manifest_sha256") != _sha256(manifest_path):
        raise ProvenanceError("completed bundle manifest changed while verified")
    for entry in declared:
        target = _safe(root, entry["path"])
        digest, size = _file_identity(target)
        if (
            _traverses_symlink(root, target)
            or not target.is_file()
            or digest != entry["sha256"]
            or size != entry["bytes"]
        ):
            raise ProvenanceError(
                f"completed bundle artifact changed while verified: {entry['path']}"
            )
    try:
        sealed_types = snapshot_tree_types(root)
    except SafeFileError as error:
        raise ProvenanceError(f"cannot inspect completed bundle: {root}") from error
    if sealed_types != final_types:
        raise ProvenanceError("completed bundle path set or type changed while it was verified")
    return completion
