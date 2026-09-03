"""Local, immutable experiment-attempt records.

TensorBoard is the interactive trace viewer.  This module supplies the small amount it cannot:
an attempt identity, durable metadata, byte hashes, and a completion record written last.  The
directory layout deliberately mirrors the later object-store layout, so GCS transport can copy
bytes without redefining experiment semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import uuid
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
from mlops.run_report import _discard_unsealed_guide, run_guide

from autoforecast._exact_json import ExactJSONError, load_exact_json, loads_exact_json
from autoforecast.frequency import FrequencyError, TimePoint, parse_origin

__all__ = [
    "ExperimentError",
    "LocalAttempt",
    "load_completed_attempt",
    "load_study_selection",
    "open_local_heldout_attempt",
    "open_local_attempt",
    "require_heldout_evaluation_approval",
    "seal_study_selection",
]

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_METADATA_SCHEMA = "autoforecast.local-attempt.v1"
_COMPLETION_SCHEMA = "autoforecast.local-attempt-completion.v2"
_SELECTION_SCHEMA = "autoforecast.study-selection.v1"
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class ExperimentError(ValueError):
    """An attempt would be ambiguous, mutable, or incomplete."""


def _require_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ExperimentError(f"{name} must match {_ID.pattern}, got {value!r}")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Render one finite, deterministic JSON record."""
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _same_json_value(left: object, right: object) -> bool:
    """Compare finite JSON values without Python's bool/int equality alias."""
    try:
        return json.dumps(
            left, allow_nan=False, sort_keys=True, separators=(",", ":")
        ) == json.dumps(right, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return False


def _sha256(path: pathlib.Path) -> str:
    return _file_identity(path)[0]


def _file_identity(path: pathlib.Path) -> tuple[str, int]:
    try:
        return _regular_file_identity(path)
    except SafeFileError as exc:
        raise ExperimentError(f"cannot read regular attempt artifact: {path}") from exc


def _load_attempt_json(root: pathlib.Path, relative_path: str) -> object:
    try:
        return loads_exact_json(read_regular_bytes(root, relative_path))
    except (ExactJSONError, SafeFileError) as exc:
        raise ExperimentError(
            f"cannot read completed attempt JSON: {root / relative_path}"
        ) from exc


def _safe_artifact_path(root: pathlib.Path, relative: object) -> pathlib.Path:
    if not isinstance(relative, str):
        raise ExperimentError("artifact manifest path must be a string")
    parts = pathlib.PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or relative in {"", "."}:
        raise ExperimentError(f"unsafe artifact manifest path: {relative!r}")
    return root.joinpath(*parts.parts)


def _traverses_symlink(root: pathlib.Path, path: pathlib.Path) -> bool:
    """Return whether *path* or one of its bundle-relative parents is a link."""
    cursor = path
    while cursor != root:
        if cursor.is_symlink() or cursor.parent == cursor:
            return True
        cursor = cursor.parent
    return root.is_symlink()


def _validate_date_range(value: object, label: str) -> tuple[TimePoint, TimePoint]:
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
    ):
        raise ExperimentError(f"{label} must be a two-point ISO-8601 date/time range")
    try:
        start, end = (
            parse_origin(item, "H" if "T" in item else "D")
            for item in value
        )
    except FrequencyError as exc:
        raise ExperimentError(f"{label} must be a two-point ISO-8601 date/time range") from exc
    if type(start) is not type(end):
        raise ExperimentError(f"{label} endpoints must use the same temporal key type")
    if start > end:
        raise ExperimentError(f"{label} start must not be after its end")
    return start, end


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    fingerprint = metadata.get("comparability_fingerprint")
    if not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(fingerprint) is None:
        raise ExperimentError("comparability_fingerprint must be a complete lowercase SHA-256")
    if type(metadata.get("hyperparameters")) is not dict:
        raise ExperimentError("attempt metadata must include a hyperparameters mapping")
    data = metadata.get("data")
    if type(data) is not dict:
        raise ExperimentError("attempt metadata must include a data mapping")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise ExperimentError("data.name must be a non-empty string")
    data_fingerprint = data.get("fingerprint")
    if not isinstance(data_fingerprint, str) or _FINGERPRINT.fullmatch(data_fingerprint) is None:
        raise ExperimentError("data.fingerprint must be a complete lowercase SHA-256")
    ranges = {
        name: _validate_date_range(data.get(name), f"data.{name}")
        for name in (
            "observed_date_range",
            "training_date_range",
            "validation_date_range",
            "test_date_range",
            "validation_target_date_range",
            "test_target_date_range",
        )
    }
    if len({type(point) for bounds in ranges.values() for point in bounds}) != 1:
        raise ExperimentError("all attempt data ranges must use the same temporal key type")
    if ranges["validation_target_date_range"][1] >= ranges["test_target_date_range"][0]:
        raise ExperimentError(
            "data.test_target_date_range must start strictly after validation_target_date_range"
        )
    for name in ("training_series", "validation_series"):
        if type(data.get(name)) is not int or data[name] < 1:
            raise ExperimentError(f"data.{name} must be a positive integer")
    if type(data.get("test")) is not str or data.get("test") not in {
        "reserved_not_evaluated",
        "evaluated",
    }:
        raise ExperimentError("data.test must be 'reserved_not_evaluated' or 'evaluated'")


@dataclass
class LocalAttempt:
    """One fresh execution in a ``study/trial/attempt`` hierarchy."""

    path: pathlib.Path
    study_id: str
    trial_id: str
    attempt_id: str
    _finalized: bool = False

    @property
    def tensorboard_dir(self) -> pathlib.Path:
        return self.path / "tensorboard"

    @property
    def metrics_path(self) -> pathlib.Path:
        return self.path / "metrics" / "epochs.jsonl"

    def write_json(self, relative_path: str, payload: Mapping[str, Any]) -> pathlib.Path:
        """Publish one small metadata/evaluation JSON record before completion.

        Only a relative path below this fresh attempt is accepted.  Completion is a terminal
        operation: a later write would invalidate its artifact manifest.
        """
        if self._finalized:
            raise ExperimentError("cannot change a finalized attempt")
        relative = pathlib.PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
            raise ExperimentError("attempt JSON path must be a relative .json path without '..'")
        try:
            return publish_bytes(self.path, relative_path, _json_bytes(payload))
        except SafeFileError as exc:
            raise ExperimentError(f"could not write attempt record: {relative_path}") from exc

    def copy_file(self, relative_path: str, source: str | pathlib.Path) -> pathlib.Path:
        """Copy one regular, non-symlink input into this fresh attempt before completion.

        Resolved YAML and dependency locks are small but essential run inputs.  They are copied
        rather than referenced from the mutable checkout, then become part of the attempt's
        terminal artifact manifest.
        """
        if self._finalized:
            raise ExperimentError("cannot change a finalized attempt")
        relative = pathlib.PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative_path in {"", "."}:
            raise ExperimentError("attempt file path must be relative and must not contain '..'")
        source_path = pathlib.Path(source)
        try:
            return copy_regular_file(self.path, relative_path, source_path)
        except SafeFileError as exc:
            raise ExperimentError(f"could not copy attempt source: {source_path}") from exc

    def write_bytes(self, relative_path: str, payload: bytes) -> pathlib.Path:
        """Publish caller-retained immutable bytes before completion."""
        if self._finalized:
            raise ExperimentError("cannot change a finalized attempt")
        relative = pathlib.PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative_path in {"", "."}:
            raise ExperimentError("attempt file path must be relative and must not contain '..'")
        try:
            return publish_bytes(self.path, relative_path, payload)
        except SafeFileError as exc:
            raise ExperimentError(f"could not write attempt artifact: {relative_path}") from exc

    def finalize(self, *, status: str, summary: Mapping[str, Any]) -> pathlib.Path:
        """Hash all prior attempt bytes, then publish terminal completion metadata last."""
        if type(status) is not str or status not in {"completed", "failed"}:
            raise ExperimentError("attempt status must be 'completed' or 'failed'")
        if self._finalized or (self.path / "completion.json").exists():
            raise ExperimentError("attempt already has terminal completion metadata")
        if (self.path / "RUN.md").exists():
            raise ExperimentError("RUN.md is reserved for the generated file guide")
        artifact_manifest = self.path / "artifact_manifest.json"
        descendants = tuple(self.path.rglob("*"))
        linked = [path for path in descendants if path.is_symlink()]
        if linked:
            raise ExperimentError(f"attempt artifacts may not traverse symlinks: {linked[0]}")
        files = sorted(
            path
            for path in descendants
            if path.is_file() and path not in {artifact_manifest, self.path / "completion.json"}
        )
        artifacts = []
        for path in files:
            digest, size = _file_identity(path)
            artifacts.append(
                {
                    "path": path.relative_to(self.path).as_posix(),
                    "sha256": digest,
                    "bytes": size,
                }
            )
        terminal_payload = {
            "schema": _COMPLETION_SCHEMA,
            "study_id": self.study_id,
            "trial_id": self.trial_id,
            "attempt_id": self.attempt_id,
            "status": status,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "summary": dict(summary),
        }
        terminal_bytes = _json_bytes(terminal_payload)
        guide_payload = run_guide(
            identity={"study_id": self.study_id, "trial_id": self.trial_id,
                      "attempt_id": self.attempt_id}, status=status,
            files=[*(item["path"] for item in artifacts),
                   "RUN.md", "artifact_manifest.json", "completion.json"],
        )
        guide = self.write_bytes("RUN.md", guide_payload)
        try:
            guide_digest, guide_size = _file_identity(guide)
            artifacts.append({"path": "RUN.md", "sha256": guide_digest, "bytes": guide_size})
            artifacts.sort(key=lambda item: item["path"])
            self.write_json(
                "artifact_manifest.json",
                {
                    "schema": "autoforecast.artifact-manifest.v2",
                    "artifacts": artifacts,
                    "terminal_payload_sha256": hashlib.sha256(
                        terminal_bytes
                    ).hexdigest(),
                },
            )
            self.write_json(
                "completion.json",
                {
                    **terminal_payload,
                    "artifact_manifest_sha256": _sha256(artifact_manifest),
                },
            )
        except Exception as error:
            _discard_unsealed_guide(self.path, guide_payload, error)
            raise
        self._finalized = True
        return self.path / "completion.json"


def open_local_attempt(
    root: str | pathlib.Path,
    *,
    study_id: str,
    trial_id: str,
    attempt_id: str,
    metadata: Mapping[str, Any],
) -> LocalAttempt:
    """Create one empty local attempt and its immutable initial metadata record.

    ``metadata`` is deliberately caller-owned because only the execution harness knows its exact
    data snapshot, split, candidate YAML hash, and hyperparameters.  It must contain those fields
    before a result can be treated as comparable.
    """
    _require_id("study_id", study_id)
    _require_id("trial_id", trial_id)
    _require_id("attempt_id", attempt_id)
    if not isinstance(metadata, Mapping):
        raise ExperimentError("attempt metadata must be a mapping")
    reserved = {"schema", "study_id", "trial_id", "attempt_id", "started_at_utc"}
    overlap = reserved & set(metadata)
    if overlap:
        raise ExperimentError(
            f"attempt metadata may not override identity fields: {sorted(overlap)}"
        )
    _validate_metadata(metadata)
    path = pathlib.Path(root) / "studies" / study_id / "trials" / trial_id / "attempts" / attempt_id
    return _open_attempt_at(
        path,
        study_id=study_id,
        trial_id=trial_id,
        attempt_id=attempt_id,
        metadata=metadata,
    )


def _open_attempt_at(
    path: pathlib.Path,
    *,
    study_id: str,
    trial_id: str,
    attempt_id: str,
    metadata: Mapping[str, Any],
) -> LocalAttempt:
    try:
        _create_fresh_directory(
            path,
            ("metadata", "metrics", "tensorboard", "checkpoints", "model", "evaluation"),
        )
    except SafeFileError as error:
        if "already exists" in str(error):
            raise ExperimentError(
                f"attempt path already exists: {path}; retries need a new attempt id"
            ) from error
        raise ExperimentError(f"cannot create attempt path: {path}") from error
    attempt = LocalAttempt(path=path, study_id=study_id, trial_id=trial_id, attempt_id=attempt_id)
    attempt.write_json(
        "metadata/attempt.json",
        {
            "schema": _METADATA_SCHEMA,
            "study_id": study_id,
            "trial_id": trial_id,
            "attempt_id": attempt_id,
            "started_at_utc": datetime.now(UTC).isoformat(),
            **dict(metadata),
        },
    )
    return attempt


def open_local_heldout_attempt(
    root: str | pathlib.Path,
    *,
    study_id: str,
    trial_id: str,
    attempt_id: str,
    metadata: Mapping[str, Any],
) -> LocalAttempt:
    """Create one fresh held-out evaluation bundle below its selected validation study.

    Held-out evidence is deliberately a sibling of validation attempts, never a later mutation of
    one.  It has the same manifest/completion contract and can therefore be transported or
    rendered by the same downstream readers.
    """
    _require_id("study_id", study_id)
    _require_id("trial_id", trial_id)
    _require_id("attempt_id", attempt_id)
    if not isinstance(metadata, Mapping):
        raise ExperimentError("held-out metadata must be a mapping")
    reserved = {"schema", "study_id", "trial_id", "attempt_id", "started_at_utc"}
    overlap = reserved & set(metadata)
    if overlap:
        raise ExperimentError(
            f"held-out metadata may not override identity fields: {sorted(overlap)}"
        )
    _validate_metadata(metadata)
    path = (
        pathlib.Path(root)
        / "studies"
        / study_id
        / "heldout"
        / "trials"
        / trial_id
        / "attempts"
        / attempt_id
    )
    return _open_attempt_at(
        path,
        study_id=study_id,
        trial_id=trial_id,
        attempt_id=attempt_id,
        metadata=metadata,
    )


def load_completed_attempt(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a completed attempt for renderers, refusing partial or malformed records."""
    terminal_paths = (
        path / "completion.json",
        path / "metadata" / "attempt.json",
        path / "artifact_manifest.json",
    )
    if path.is_symlink() or any(_traverses_symlink(path, item) for item in terminal_paths):
        raise ExperimentError(f"attempt root may not be a symlink: {path}")
    try:
        initial_types = snapshot_tree_types(path)
    except SafeFileError as error:
        raise ExperimentError(f"cannot inspect completed attempt: {path}") from error
    linked = next((item for item, kind in initial_types.items() if kind == "symlink"), None)
    if linked is not None:
        raise ExperimentError(f"completed attempt may not contain symlinks: {linked}")
    special = next((item for item, kind in initial_types.items() if kind == "special"), None)
    if special is not None:
        raise ExperimentError(f"completed attempt contains an unsupported path type: {special}")
    completion = _load_attempt_json(path, "completion.json")
    metadata = _load_attempt_json(path, "metadata/attempt.json")
    if type(completion) is not dict or type(metadata) is not dict:
        raise ExperimentError(f"attempt records must be JSON objects: {path}")
    manifest_path = path / "artifact_manifest.json"
    manifest = _load_attempt_json(path, "artifact_manifest.json")
    if type(manifest) is not dict:
        raise ExperimentError(f"invalid artifact manifest: {path}")
    if (
        completion.get("schema") == "autoforecast.local-attempt.v1"
        or manifest.get("schema") == "autoforecast.artifact-manifest.v1"
    ):
        raise ExperimentError(
            f"legacy attempt has unsealed terminal metadata and must be rerun: {path}"
        )
    if (
        set(completion)
        != {
            "schema",
            "study_id",
            "trial_id",
            "attempt_id",
            "status",
            "completed_at_utc",
            "artifact_manifest_sha256",
            "summary",
        }
        or type(completion.get("status")) is not str
        or completion.get("status") not in {"completed", "failed"}
        or type(completion.get("completed_at_utc")) is not str
        or type(completion.get("summary")) is not dict
        or completion.get("schema") != _COMPLETION_SCHEMA
        or metadata.get("schema") != _METADATA_SCHEMA
    ):
        raise ExperimentError(f"unsupported attempt schema: {path}")
    for identity in ("study_id", "trial_id", "attempt_id"):
        value = completion.get(identity)
        _require_id(identity, value if isinstance(value, str) else "")
        if metadata.get(identity) != value:
            raise ExperimentError(f"attempt {identity} differs from completion: {path}")
    _validate_metadata(metadata)
    if completion.get("artifact_manifest_sha256") != _sha256(manifest_path):
        raise ExperimentError(
            f"artifact manifest checksum does not match completion record: {path}"
        )
    if (
        set(manifest) != {"schema", "artifacts", "terminal_payload_sha256"}
        or manifest.get("schema") != "autoforecast.artifact-manifest.v2"
    ):
        raise ExperimentError(f"invalid artifact manifest: {path}")
    terminal_payload = {
        key: value for key, value in completion.items() if key != "artifact_manifest_sha256"
    }
    if (
        manifest.get("terminal_payload_sha256")
        != hashlib.sha256(_json_bytes(terminal_payload)).hexdigest()
    ):
        raise ExperimentError(f"attempt terminal payload is not sealed: {path}")
    entries = manifest.get("artifacts")
    if type(entries) is not list:
        raise ExperimentError(f"artifact manifest entries must be a list: {path}")
    expected_paths: set[pathlib.Path] = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"path", "sha256", "bytes"}:
            raise ExperimentError(f"artifact manifest entry must be an object: {path}")
        artifact = _safe_artifact_path(path, entry.get("path"))
        if artifact in expected_paths:
            raise ExperimentError(f"artifact manifest contains a duplicate path: {artifact}")
        expected_paths.add(artifact)
        digest, size = _file_identity(artifact)
        if (
            _traverses_symlink(path, artifact)
            or not artifact.is_file()
            or not isinstance(entry.get("sha256"), str)
            or entry.get("sha256") != digest
            or type(entry.get("bytes")) is not int
            or entry["bytes"] != size
        ):
            raise ExperimentError(f"artifact manifest does not match bytes: {artifact}")
    actual_paths = {
        candidate
        for candidate, kind in initial_types.items()
        if kind == "file" and candidate not in {manifest_path, path / "completion.json"}
    }
    if actual_paths != expected_paths:
        raise ExperimentError(f"attempt bytes changed after completion: {path}")
    try:
        final_types = snapshot_tree_types(path)
    except SafeFileError as error:
        raise ExperimentError(f"cannot inspect completed attempt: {path}") from error
    final_link = next((item for item, kind in final_types.items() if kind == "symlink"), None)
    if final_link is not None:
        raise ExperimentError(f"completed attempt may not contain symlinks: {final_link}")
    final_special = next((item for item, kind in final_types.items() if kind == "special"), None)
    if final_special is not None:
        raise ExperimentError(
            f"completed attempt contains an unsupported path type: {final_special}"
        )
    if final_types != initial_types:
        raise ExperimentError(f"attempt path set or type changed while it was verified: {path}")
    if (
        _load_attempt_json(path, "completion.json") != completion
        or _load_attempt_json(path, "metadata/attempt.json") != metadata
        or _load_attempt_json(path, "artifact_manifest.json") != manifest
        or completion.get("artifact_manifest_sha256") != _sha256(manifest_path)
    ):
        raise ExperimentError(f"attempt terminal records changed while verified: {path}")
    for entry in entries:
        artifact = _safe_artifact_path(path, entry["path"])
        digest, size = _file_identity(artifact)
        if (
            _traverses_symlink(path, artifact)
            or not artifact.is_file()
            or digest != entry["sha256"]
            or size != entry["bytes"]
        ):
            raise ExperimentError(f"attempt artifact changed while verified: {artifact}")
    try:
        sealed_types = snapshot_tree_types(path)
    except SafeFileError as error:
        raise ExperimentError(f"cannot inspect completed attempt: {path}") from error
    if sealed_types != final_types:
        raise ExperimentError(f"attempt path set or type changed while it was verified: {path}")
    return metadata, completion


def _selection_candidates(
    study_root: pathlib.Path,
    *,
    top_k: int,
    direction: str,
    required_trial_count: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Read one completed attempt per trial and make its validation result rankable.

    A retry deliberately gets a new attempt ID.  Choosing between two completed retries would be
    a hidden policy decision, so this first contract refuses it until a study declares one.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for attempt_path in sorted(study_root.glob("trials/*/attempts/*")):
        if not (attempt_path / "completion.json").is_file():
            continue
        metadata, completion = load_completed_attempt(attempt_path)
        if completion.get("status") != "completed":
            continue
        validation = load_exact_json(attempt_path / "evaluation" / "validation.json")
        if not isinstance(validation, dict):
            raise ExperimentError(f"validation result must be an object: {attempt_path}")
        value = validation.get("value")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ExperimentError(f"validation result must have a finite value: {attempt_path}")
        if validation.get("objective") != "validation/objective":
            raise ExperimentError(f"validation result has an unsupported objective: {attempt_path}")
        if validation.get("test_metrics") != "not_evaluated":
            raise ExperimentError(f"study candidate leaked test metrics: {attempt_path}")
        fingerprint = metadata.get("comparability_fingerprint")
        if not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(fingerprint) is None:
            raise ExperimentError(f"candidate has no valid comparison identity: {attempt_path}")
        study = metadata.get("study")
        if type(study) is not dict or study.get("direction") != direction:
            raise ExperimentError(f"candidate direction does not match its study: {attempt_path}")
        if type(study.get("top_k")) is not int or study["top_k"] != top_k:
            raise ExperimentError(f"selection top_k differs from its study: {attempt_path}")
        trial_id = metadata.get("trial_id")
        attempt_id = metadata.get("attempt_id")
        if not isinstance(trial_id, str) or not isinstance(attempt_id, str):
            raise ExperimentError(f"candidate has invalid identity: {attempt_path}")
        grouped.setdefault(trial_id, []).append(
            {
                "trial_id": trial_id,
                "attempt_id": attempt_id,
                "validation_objective": float(value),
                "comparability_fingerprint": fingerprint,
                "attempt_path": attempt_path.relative_to(study_root).as_posix(),
                "artifact_manifest_sha256": completion.get("artifact_manifest_sha256"),
                "parameter_identity": hashlib.sha256(
                    json.dumps(
                        metadata.get("hyperparameters"),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    if not grouped:
        raise ExperimentError(f"study has no completed validation candidates: {study_root}")
    if required_trial_count is not None and len(grouped) != required_trial_count:
        raise ExperimentError(
            f"study needs exactly {required_trial_count} completed trials, found {len(grouped)}"
        )
    if any(len(attempts) != 1 for attempts in grouped.values()):
        raise ExperimentError(
            "study has more than one completed attempt for a trial; declare a retry policy first"
        )
    candidates = [attempts[0] for attempts in grouped.values()]
    fingerprints = {candidate["comparability_fingerprint"] for candidate in candidates}
    if len(fingerprints) != 1:
        raise ExperimentError("study candidates have different comparison identities")
    candidates.sort(
        key=lambda candidate: (
            float(candidate["validation_objective"]) * (1 if direction == "minimize" else -1),
            str(candidate["parameter_identity"]),
            str(candidate["trial_id"]),
            str(candidate["attempt_id"]),
        )
    )
    last_value: float | None = None
    rank = 0
    for index, candidate in enumerate(candidates, start=1):
        value = float(candidate["validation_objective"])
        if last_value is None or value != last_value:
            rank = index
            last_value = value
        candidate["rank"] = rank
    return str(next(iter(fingerprints))), candidates


def _create_json_once(path: pathlib.Path, payload: Mapping[str, Any]) -> pathlib.Path:
    """Atomically publish a small terminal study record without replacing a prior publication."""
    raw = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(raw)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ExperimentError(f"study record already exists and is sealed: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def seal_study_selection(
    study_root: str | pathlib.Path,
    *,
    top_k: int,
    direction: str,
    required_trial_count: int | None = None,
) -> pathlib.Path:
    """Seal validation-only top-k selection once a study has reached its declared budget.

    The exact candidate manifest hashes make this a durable bridge from an HPO result to the later
    held-out evaluator.  The local create-if-absent operation maps to conditional object creation
    when the same layout is transported to GCS.
    """
    root = pathlib.Path(study_root)
    if not root.is_dir():
        raise ExperimentError(f"study root does not exist: {root}")
    if type(top_k) is not int or top_k < 1:
        raise ExperimentError("top_k must be a positive integer")
    if direction not in {"minimize", "maximize"}:
        raise ExperimentError("selection direction must be 'minimize' or 'maximize'")
    if required_trial_count is not None and (
        type(required_trial_count) is not int or required_trial_count < 1
    ):
        raise ExperimentError("required_trial_count must be a positive integer when supplied")
    fingerprint, candidates = _selection_candidates(
        root, top_k=top_k, direction=direction, required_trial_count=required_trial_count
    )
    if top_k > len(candidates):
        raise ExperimentError(f"top_k={top_k} exceeds {len(candidates)} completed trials")
    selection = {
        "schema": _SELECTION_SCHEMA,
        "study_id": root.name,
        "comparability_fingerprint": fingerprint,
        "direction": direction,
        "top_k": top_k,
        "tie_policy": "stable_candidate_identity_at_cutoff",
        "candidates": candidates,
        "selected": candidates[:top_k],
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }
    return _create_json_once(root / "selection.json", selection)


def load_study_selection(study_root: str | pathlib.Path) -> dict[str, Any]:
    """Load the sealed validation selection for a later held-out evaluator or renderer."""
    root = pathlib.Path(study_root)
    record = load_exact_json(root / "selection.json")
    required = {
        "schema",
        "study_id",
        "comparability_fingerprint",
        "direction",
        "top_k",
        "tie_policy",
        "candidates",
        "selected",
        "selected_at_utc",
    }
    if (
        type(record) is not dict
        or set(record) != required
        or record.get("schema") != _SELECTION_SCHEMA
    ):
        raise ExperimentError(f"invalid or unsupported study selection: {root / 'selection.json'}")
    if record.get("study_id") != root.name:
        raise ExperimentError("selection study identity does not match its path")
    if record.get("direction") not in {"minimize", "maximize"}:
        raise ExperimentError("selection direction is invalid")
    top_k = record.get("top_k")
    candidates, selected = record.get("candidates"), record.get("selected")
    if (
        type(top_k) is not int
        or top_k < 1
        or type(candidates) is not list
        or type(selected) is not list
    ):
        raise ExperimentError("selection candidates or top_k are invalid")
    if len(selected) != top_k or len(candidates) < top_k:
        raise ExperimentError("selection does not contain the declared number of candidates")
    if record.get("tie_policy") != "stable_candidate_identity_at_cutoff":
        raise ExperimentError("selection tie policy is invalid")
    selected_at = record.get("selected_at_utc")
    try:
        timestamp = datetime.fromisoformat(selected_at) if isinstance(selected_at, str) else None
    except ValueError as exc:
        raise ExperimentError("selection timestamp is invalid") from exc
    if timestamp is None or timestamp.tzinfo is None:
        raise ExperimentError("selection timestamp is invalid")
    fingerprint, ranked = _selection_candidates(
        root,
        top_k=top_k,
        direction=record["direction"],
        required_trial_count=None,
    )
    if (
        record.get("comparability_fingerprint") != fingerprint
        or not _same_json_value(candidates, ranked)
        or not _same_json_value(selected, ranked[:top_k])
    ):
        raise ExperimentError("selection no longer matches the completed ranked candidates")
    return record


def require_heldout_evaluation_approval(
    study_root: str | pathlib.Path,
    *,
    trial_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    """Return the matching sealed selection entry or refuse held-out evaluation.

    A held-out runner must call this before it reads or scores held-out targets.  The small
    explicit gate avoids a generic workflow engine while mechanically separating selection from
    acceptance evaluation.
    """
    _require_id("trial_id", trial_id)
    _require_id("attempt_id", attempt_id)
    record = load_study_selection(study_root)
    for candidate in record["selected"]:
        if type(candidate) is dict and (
            candidate.get("trial_id") == trial_id and candidate.get("attempt_id") == attempt_id
        ):
            root = pathlib.Path(study_root)
            relative = candidate.get("attempt_path")
            if not isinstance(relative, str):
                raise ExperimentError("selected candidate has no safe source attempt path")
            source = _safe_artifact_path(root, relative)
            metadata, completion = load_completed_attempt(source)
            if completion.get("artifact_manifest_sha256") != candidate.get(
                "artifact_manifest_sha256"
            ):
                raise ExperimentError("selected candidate no longer matches its sealed manifest")
            if metadata.get("comparability_fingerprint") != record.get("comparability_fingerprint"):
                raise ExperimentError(
                    "selected candidate no longer matches its comparison identity"
                )
            return dict(candidate)
    raise ExperimentError(
        f"held-out evaluation is not approved for {trial_id}/{attempt_id}; it was not selected"
    )
