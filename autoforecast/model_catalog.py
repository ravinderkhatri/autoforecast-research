"""Strict, development-only catalog for selecting registered AutoForecast adapters.

The Python registry remains authoritative for callable implementations and capability metadata.
This module supplies the missing *declarative* layer: one ``model.yaml`` per candidate plus one
suite policy that says which candidates to evaluate.  A model file may also contain its finite
hyperparameter contract.  It deliberately does not select a winner from live scores or mutate a
production alias.  A top-K result is evidence, so it must be materialised by the evaluation
harness with its manifest and source hashes, not recomputed from a mutable catalog at inference
time.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import stat
from dataclasses import dataclass

from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast.adapter_contracts import descriptor
from autoforecast.adapters import known_adapter_names
from autoforecast.model_runtime import runtime_ids

__all__ = [
    "Candidate",
    "CatalogError",
    "SelectionPolicy",
    "discover_candidates",
    "load_selection_policy",
    "resolve_candidates",
]

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MODEL_KEYS = frozenset({"schema_version", "model"})
_MODEL_KEYS_WITH_HYPERPARAMETERS = _MODEL_KEYS | {"hyperparameters"}
_MODEL_KEYS_WITH_PARAMETER_CONSTRAINTS = _MODEL_KEYS_WITH_HYPERPARAMETERS | {"constraints"}
_IMPLEMENTATION_KEYS = frozenset({"config", "sha256"})
_MODEL_BODY_KEYS = frozenset({"id", "adapter", "family", "status", "tags", "runtime"})
_POLICY_KEYS = frozenset({"schema_version", "selection"})
_POLICY_BODY_KEYS = frozenset({"id", "candidate_tags", "statuses", "metric", "direction", "top_k"})


class CatalogError(ValueError):
    """A catalog file is unsafe, malformed, or contradicts the registered adapter boundary."""


@dataclass(frozen=True)
class Candidate:
    """One source-controlled model candidate, without executable Python or live artifact state."""

    id: str
    adapter: str
    family: str
    status: str
    tags: tuple[str, ...]
    runtime: str
    source_path: pathlib.Path
    source_sha256: str
    source_bytes: bytes
    implementation_config_path: pathlib.Path | None = None
    implementation_config_sha256: str | None = None
    implementation_config_bytes: bytes | None = None


@dataclass(frozen=True)
class SelectionPolicy:
    """The pre-registered rule for the development candidates a suite will evaluate.

    ``top_k`` is only a requested number of *eligible, ranked* candidates. It is not a promise
    that all K will be promoted, and it must never be used to rank or choose on final holdout data.
    """

    id: str
    candidate_tags: frozenset[str]
    statuses: frozenset[str]
    metric: str
    direction: str
    top_k: int
    source_path: pathlib.Path
    source_sha256: str


def _read_regular_bytes(path: pathlib.Path, *, root: pathlib.Path | None = None) -> bytes:
    """Read one single-link file once, without dereferencing any lexical symlink."""
    lexical = pathlib.Path(os.path.abspath(path))
    if root is not None:
        root = pathlib.Path(os.path.abspath(root))
        if not lexical.is_relative_to(root):
            raise CatalogError(f"catalog file escapes the project root: {path}")
        cursor = lexical.parent
        while cursor != root:
            try:
                metadata = cursor.lstat()
            except OSError as error:
                raise CatalogError(f"catalog path is not readable: {path}") from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise CatalogError(f"catalog path may not traverse an alias: {path}")
            cursor = cursor.parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise CatalogError(f"catalog file must be a regular file, not {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CatalogError(f"catalog file must be a single-link regular file, not {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read()
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise CatalogError(f"catalog file changed while it was read: {path}")
        return raw
    finally:
        os.close(descriptor)


def _load_mapping(path: pathlib.Path) -> tuple[dict[str, object], str, bytes]:
    """Read one regular YAML file with safe parsing and a byte identity for resolved-run records."""
    raw = _read_regular_bytes(path)
    try:
        value = loads_exact_yaml(raw)
    except ExactYAMLError as exc:
        raise CatalogError(f"invalid YAML in {path}: {exc}") from exc
    if type(value) is not dict:
        raise CatalogError(f"catalog file {path} must contain a mapping")
    return value, hashlib.sha256(raw).hexdigest(), raw


def _exact_keys(value: dict[str, object], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise CatalogError(f"{label} must contain exactly {sorted(expected)}, got {sorted(value)}")


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise CatalogError(f"{label} must match {_ID.pattern!r}")
    return value


def _string_set(value: object, label: str) -> frozenset[str]:
    if type(value) is not list or not value:
        raise CatalogError(f"{label} must be a non-empty YAML list")
    if any(type(item) is not str or not item.strip() for item in value):
        raise CatalogError(f"{label} must contain only non-blank strings")
    result = frozenset(value)
    if len(result) != len(value):
        raise CatalogError(f"{label} must not contain duplicates")
    return result


def _candidate(path: pathlib.Path) -> Candidate:
    value, digest, source_bytes = _load_mapping(path)
    base_keys = set(value) - {"implementation"}
    if base_keys not in (
        set(_MODEL_KEYS),
        set(_MODEL_KEYS_WITH_HYPERPARAMETERS),
        set(_MODEL_KEYS_WITH_PARAMETER_CONSTRAINTS),
    ):
        raise CatalogError(
            f"{path} must contain {_MODEL_KEYS} with optional hyperparameters, constraints and "
            "implementation, "
            f"got {sorted(value)}"
        )
    if type(value.get("schema_version")) is not int or value.get("schema_version") not in {1, 2}:
        raise CatalogError(f"{path}: schema_version must be integer 1 or 2")
    body = value.get("model")
    if type(body) is not dict:
        raise CatalogError(f"{path}: model must be a mapping")
    _exact_keys(body, _MODEL_BODY_KEYS, f"{path}: model")
    model_id = _identifier(body.get("id"), f"{path}: model.id")
    adapter = _identifier(body.get("adapter"), f"{path}: model.adapter")
    if adapter not in known_adapter_names():
        raise CatalogError(
            f"{path}: model.adapter {adapter!r} is unknown; known={list(known_adapter_names())}"
        )
    family = _identifier(body.get("family"), f"{path}: model.family")
    status = _identifier(body.get("status"), f"{path}: model.status")
    tags = tuple(sorted(_string_set(body.get("tags"), f"{path}: model.tags")))
    runtime = _identifier(body.get("runtime"), f"{path}: model.runtime")
    if runtime not in runtime_ids():
        raise CatalogError(
            f"{path}: model.runtime {runtime!r} is unknown; known={list(runtime_ids())}"
        )
    if runtime != descriptor(adapter).runtime:
        raise CatalogError(f"{path}: runtime differs from maintained adapter descriptor")
    if descriptor(adapter).parameters == "none" and value.get("hyperparameters"):
        raise CatalogError(f"{path}: adapter has no supported parameters")
    if "hyperparameters" in value and type(value["hyperparameters"]) is not dict:
        raise CatalogError(f"{path}: hyperparameters must be a mapping")
    if "constraints" in value and (
        value.get("schema_version") != 2 or type(value["constraints"]) is not list
    ):
        raise CatalogError(f"{path}: constraints require schema_version 2 and a list")
    implementation_path = None
    implementation_sha256 = None
    if "implementation" in value:
        implementation = value["implementation"]
        if type(implementation) is not dict:
            raise CatalogError(f"{path}: implementation must be a mapping")
        _exact_keys(implementation, _IMPLEMENTATION_KEYS, f"{path}: implementation")
        relative = implementation["config"]
        expected_sha256 = implementation["sha256"]
        if (
            type(relative) is not str
            or not relative.strip()
            or pathlib.PurePath(relative).is_absolute()
            or type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            raise CatalogError(f"{path}: implementation identity is invalid")
        pure = pathlib.PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise CatalogError(f"{path}: implementation identity is invalid")
        implementation_path = _PROJECT_ROOT.joinpath(*pure.parts)
        try:
            implementation_bytes = _read_regular_bytes(
                implementation_path, root=_PROJECT_ROOT
            )
        except CatalogError as error:
            raise CatalogError(f"{path}: implementation config is missing or aliased") from error
        if hashlib.sha256(implementation_bytes).hexdigest() != expected_sha256:
            raise CatalogError(f"{path}: implementation config is missing or has changed")
        implementation_sha256 = expected_sha256
    else:
        implementation_bytes = None
    return Candidate(
        model_id,
        adapter,
        family,
        status,
        tags,
        runtime,
        path,
        digest,
        source_bytes,
        implementation_path,
        implementation_sha256,
        implementation_bytes,
    )


def discover_candidates(root: str | pathlib.Path) -> tuple[Candidate, ...]:
    """Load every ``model.yaml`` beneath *root* in sorted order and refuse duplicate identities."""
    root_path = pathlib.Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise CatalogError(f"model catalog root must be a directory, not {root_path}")
    candidates = tuple(_candidate(path) for path in sorted(root_path.rglob("model.yaml")))
    if not candidates:
        raise CatalogError(f"model catalog {root_path} contains no model.yaml files")
    ids = [candidate.id for candidate in candidates]
    if len(set(ids)) != len(ids):
        duplicates = sorted(model_id for model_id in set(ids) if ids.count(model_id) > 1)
        raise CatalogError(f"model catalog contains duplicate model ids {duplicates}")
    return tuple(sorted(candidates, key=lambda candidate: candidate.id))


def load_selection_policy(path: str | pathlib.Path) -> SelectionPolicy:
    """Load the policy that filters candidates before the harness performs its governed ranking."""
    source = pathlib.Path(path)
    value, digest, _ = _load_mapping(source)
    _exact_keys(value, _POLICY_KEYS, str(source))
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise CatalogError(f"{source}: schema_version must be integer 1")
    body = value.get("selection")
    if type(body) is not dict:
        raise CatalogError(f"{source}: selection must be a mapping")
    _exact_keys(body, _POLICY_BODY_KEYS, f"{source}: selection")
    top_k = body.get("top_k")
    if type(top_k) is not int or type(top_k) is bool or top_k < 1:
        raise CatalogError(f"{source}: selection.top_k must be a positive integer")
    metric = _identifier(body.get("metric"), f"{source}: selection.metric")
    direction = body.get("direction")
    if type(direction) is not str or direction not in {"minimize", "maximize"}:
        raise CatalogError(f"{source}: selection.direction must be minimize or maximize")
    return SelectionPolicy(
        id=_identifier(body.get("id"), f"{source}: selection.id"),
        candidate_tags=_string_set(
            body.get("candidate_tags"), f"{source}: selection.candidate_tags"
        ),
        statuses=_string_set(body.get("statuses"), f"{source}: selection.statuses"),
        metric=metric,
        direction=direction,
        top_k=top_k,
        source_path=source,
        source_sha256=digest,
    )


def resolve_candidates(
    candidates: tuple[Candidate, ...], policy: SelectionPolicy
) -> tuple[Candidate, ...]:
    """Return the deterministic pre-ranking candidate set selected by a policy.

    The harness must score this entire result on its frozen development protocol. It then applies
    eligibility gates before ranking and records the resulting top-K lock as an immutable artifact.
    """
    selected = tuple(
        candidate
        for candidate in candidates
        if candidate.status in policy.statuses
        and policy.candidate_tags <= frozenset(candidate.tags)
    )
    if len(selected) < policy.top_k:
        raise CatalogError(
            f"selection {policy.id!r} resolves {len(selected)} candidates, "
            f"fewer than top_k={policy.top_k}"
        )
    return selected
