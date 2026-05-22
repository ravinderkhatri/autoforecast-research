"""Offline Transformers/PEFT runtime for the TimesFM standalone boundary.

This module adds only the maintained-provider concerns that :mod:`timesfm` does not own:
loading the exact local Transformers conversion, optionally loading one content-bound PEFT
adapter, and producing a final-only LoRA artifact from point-in-time training rows.  Spark
selection and training-row publication happen before the model is loaded.  Forecast Parquet,
exposure dispositions, output validation, clipping, and atomic publication remain in
``autoforecast.adapters.foundation``; ``TimesFMTransformersBackend`` implements its injected backend
protocol.

The Transformers port uses confusing output names.  ``mean_predictions`` is the decoded median
(also ``full_predictions[..., 5]``), while expected demand is
``full_predictions[..., 0]``.  ``forecast`` returns those arrays in the order expected by the
shared TimesFM boundary: ``(median, full_predictions)``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import random
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast._exact_json import ExactJSONError, load_exact_json
from autoforecast.benchmarks.m5_contract import json_values_equal_exact
from autoforecast.panel import Panel
from provenance.lineage import directory_digest

TRANSFORMERS_MODEL_ID = "google/timesfm-2.5-200m-transformers"
TRANSFORMERS_REVISION = "5a9806b9b291fad9233b5249d88263f1846304d3"
TRANSFORMERS_CONFIG_SHA256 = (
    "452ecae918f67b2e7d0f2892ab424d1876939e70d077db3708a4fe8ca03a7de5"
)
TRANSFORMERS_WEIGHTS_SHA256 = (
    "b53f6d52114e2ad786890f3c4637ce05f580b7800d6e24401f88b398b76035ef"
)
M5_SALES_TRAIN_EVALUATION_SHA256 = (
    "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0"
)
M5_PANEL_CONTENT_SHA256 = (
    "3bb0cda1631defcde962e0745186fc5a6fe0506523952b09ab6a5cf6b458f9ae"
)
M5_LORA_SOURCE_PARTITIONS = 1024
GOVERNED_HORIZON = 28

TRAINING_ROWS_SCHEMA = "autoforecast.timesfm-lora-training-rows.v1"
LORA_ARTIFACT_SCHEMA = "autoforecast.timesfm-lora-artifact.v1"
LORA_CONTINUATION_ARTIFACT_SCHEMA = (
    "autoforecast.timesfm-lora-continuation-artifact.v1"
)
TRAINING_ROWS_FILE = "training_rows.parquet"
TRAINING_ROWS_INDEX = "training_rows.jsonl"
TRAINING_ROWS_MANIFEST = "training_rows_manifest.json"
LORA_MANIFEST = "timesfm_lora_manifest.json"
RUNTIME_LOCK_CANONICALIZATION = (
    "lexicographic_normalized_name_equals_equals_version_utf8_lines_final_lf"
)

_REQUIRED_RUNTIME_DISTRIBUTIONS = frozenset(
    {
        "accelerate",
        "huggingface-hub",
        "numpy",
        "peft",
        "py4j",
        "pyarrow",
        "pyspark",
        "safetensors",
        "timesfm",
        "torch",
        "transformers",
    }
)


@dataclass(frozen=True)
class M5LoRACutoffIdentity:
    """Sealed deterministic training-row identity for one forward adapter cutoff."""

    cutoff: str
    cutoff_day: int
    eligible_series: int
    windows: int
    last_start_day: int
    training_rows_sha256: str
    first_start_day: int
    first_context_end_day: int
    first_target_end_day: int


M5_LORA_CUTOFF_IDENTITIES = (
    M5LoRACutoffIdentity(
        cutoff="2016-02-28",
        cutoff_day=1857,
        eligible_series=30099,
        windows=90297,
        last_start_day=1574,
        training_rows_sha256=(
            "0de03c202133e6dea6ea44c229e710e24336015677347ffa4760e4c30e8ab799"
        ),
        first_start_day=1197,
        first_context_end_day=1452,
        first_target_end_day=1480,
    ),
    M5LoRACutoffIdentity(
        cutoff="2016-03-27",
        cutoff_day=1885,
        eligible_series=30328,
        windows=90984,
        last_start_day=1602,
        training_rows_sha256=(
            "3020125d2935754bfc94caaefb31e9468afc1694731b7f58f2568574c7a7f44f"
        ),
        first_start_day=765,
        first_context_end_day=1020,
        first_target_end_day=1048,
    ),
    M5LoRACutoffIdentity(
        cutoff="2016-04-24",
        cutoff_day=1913,
        eligible_series=30347,
        windows=91041,
        last_start_day=1630,
        training_rows_sha256=(
            "e931ef2e1698d44436e4991f2d01501b956ebf0a0956a296d3dcadfc88fb7c51"
        ),
        first_start_day=1141,
        first_context_end_day=1396,
        first_target_end_day=1424,
    ),
)


@dataclass(frozen=True)
class LoRATrainingConfig:
    """Every numerical LoRA degree of freedom; governed callers must pass one explicitly."""

    context_length: int
    horizon_length: int
    epochs: int
    batch_size: int
    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    seed: int
    device: str


@dataclass(frozen=True)
class LoRAContinuationConfig:
    """Numerical controls for a weights-only LoRA warm-start continuation."""

    max_additional_epochs: int
    patience: int
    restart_learning_rate: float
    absolute_min_delta: float = 1e-6
    relative_min_delta: float = 1e-4


PROSPECTIVE_M5_LORA_CONFIG = LoRATrainingConfig(
    context_length=256,
    horizon_length=28,
    epochs=3,
    batch_size=32,
    rank=4,
    alpha=8,
    dropout=0.05,
    learning_rate=1e-4,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
    seed=20260829,
    device="mps",
)


@dataclass(frozen=True)
class TrainingRowsArtifact:
    """Immutable identity of materialised, pre-training TimesFM windows."""

    path: str
    rows_sha256: str
    manifest_path: str
    manifest_sha256: str
    n_series: int
    n_rows: int
    cutoff: str
    panel_content_sha256: str


@dataclass(frozen=True)
class LoRAAdapterArtifact:
    """Final-only PEFT adapter identity returned after atomic publication."""

    path: str
    sha256: str
    manifest_path: str
    n_training_rows: int
    training_rows_sha256: str
    panel_content_sha256: str


@dataclass(frozen=True)
class RuntimeLockIdentity:
    """Authenticated source and canonical full-distribution identity for one worker."""

    source_path: str
    source_sha256: str
    record_path: tuple[str, ...]
    projection_sha256: str
    python: str
    package_versions: tuple[tuple[str, str], ...]


class _TrainingBackend(Protocol):
    """Injectable seam around the maintained torch/PEFT training runtime."""

    def _train_batch(
        self,
        contexts: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float: ...

    def _save_adapter(self, path: pathlib.Path) -> None: ...

    def _identity(self) -> Mapping[str, object]: ...


class _ContinuationBackend(_TrainingBackend, Protocol):
    """Injectable seam for validation and in-memory best-adapter restoration."""

    def _validation_batch(
        self,
        contexts: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float: ...

    def _snapshot_adapter(self) -> object: ...

    def _restore_adapter(self, state: object) -> None: ...


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_node_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return metadata that changes when a source node is replaced or rewritten."""
    return (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_source_node_no_follow(
    path: pathlib.Path,
    *,
    label: str,
    directory: bool,
) -> int:
    """Open one absolute source without following any path component."""
    if not path.is_absolute() or path.anchor != os.sep or ".." in path.parts:
        raise ValueError(f"{label} path must be absolute, canonical, and symlink-free")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    descriptor = os.open(os.sep, directory_flags)
    try:
        parts = path.parts[1:]
        if not parts:
            if not directory:
                raise ValueError(f"{label} must be one real, single-link regular file")
            return descriptor
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            expects_directory = not final or directory
            try:
                named = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"cannot inspect {label} path without following links") from exc
            if stat.S_ISLNK(named.st_mode):
                raise ValueError(f"{label} path must be canonical and symlink-free")
            if expects_directory:
                if not stat.S_ISDIR(named.st_mode):
                    raise ValueError(f"{label} path must contain only real directories")
                flags = directory_flags
            else:
                if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
                    raise ValueError(
                        f"{label} must be one real, single-link regular file"
                    )
                flags = file_flags
            try:
                child = os.open(part, flags, dir_fd=descriptor)
                opened = os.fstat(child)
            except OSError as exc:
                raise ValueError(f"cannot open {label} path without following links") from exc
            if _source_node_identity(named) != _source_node_identity(opened):
                os.close(child)
                raise RuntimeError(f"{label} identity changed while it was opened")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _source_file_identity(path: pathlib.Path, *, label: str) -> tuple[int, ...]:
    descriptor = _open_source_node_no_follow(path, label=label, directory=False)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"{label} must be one real, single-link regular file")
        return _source_node_identity(metadata)
    finally:
        os.close(descriptor)


def _descriptor_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _source_tree_entry_flags(metadata: os.stat_result, *, label: str) -> int:
    """Classify one unfiltered tree entry and reject aliases/special nodes."""
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"{label} tree must not contain symbolic links")
    if stat.S_ISREG(mode):
        if metadata.st_nlink != 1:
            raise ValueError(f"{label} tree must not contain hard-linked files")
        return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    if stat.S_ISDIR(mode):
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    raise ValueError(f"{label} tree must not contain special filesystem nodes")


def _walk_source_tree(
    descriptor: int,
    *,
    label: str,
    relative: pathlib.PurePosixPath | None = None,
    copy_root: pathlib.Path | None = None,
    hash_files: bool = False,
) -> tuple[
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[str, str], ...],
]:
    """Inspect, optionally copy, and optionally hash an unfiltered descriptor tree."""
    relative = relative or pathlib.PurePosixPath(".")
    identities: list[tuple[str, tuple[int, ...]]] = [
        (relative.as_posix(), _source_node_identity(os.fstat(descriptor)))
    ]
    hashes: list[tuple[str, str]] = []
    try:
        names = sorted(entry.name for entry in os.scandir(descriptor))
    except OSError as exc:
        raise ValueError(f"cannot inspect {label} tree") from exc
    if not names and relative != pathlib.PurePosixPath("."):
        raise ValueError(f"{label} tree contains an unauthenticated empty directory")
    for name in names:
        child_relative = relative / name
        try:
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"cannot inspect {label} tree entry") from exc
        mode = named.st_mode
        flags = _source_tree_entry_flags(named, label=label)
        try:
            child = os.open(name, flags, dir_fd=descriptor)
            opened = os.fstat(child)
        except OSError as exc:
            raise ValueError(f"cannot open {label} tree entry without links") from exc
        try:
            opened_identity = _source_node_identity(opened)
            if _source_node_identity(named) != opened_identity:
                raise RuntimeError(f"{label} tree entry changed while it was opened")
            if stat.S_ISDIR(mode):
                if copy_root is not None:
                    (copy_root / child_relative).mkdir(mode=0o700)
                nested_identities, nested_hashes = _walk_source_tree(
                    child,
                    label=label,
                    relative=child_relative,
                    copy_root=copy_root,
                    hash_files=hash_files,
                )
                identities.extend(nested_identities)
                hashes.extend(nested_hashes)
            else:
                digest: str | None = None
                if copy_root is not None:
                    destination = copy_root / child_relative
                    destination_descriptor = os.open(
                        destination,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        0o400,
                    )
                    try:
                        source_digest = hashlib.sha256()
                        os.lseek(child, 0, os.SEEK_SET)
                        while block := os.read(child, 1024 * 1024):
                            source_digest.update(block)
                            view = memoryview(block)
                            while view:
                                written = os.write(destination_descriptor, view)
                                view = view[written:]
                        digest = source_digest.hexdigest()
                    finally:
                        os.close(destination_descriptor)
                elif hash_files:
                    digest = _descriptor_sha256(child)
                if digest is not None:
                    hashes.append((child_relative.as_posix(), digest))
                if _source_node_identity(os.fstat(child)) != opened_identity:
                    raise RuntimeError(f"{label} tree file changed while it was read")
                identities.append((child_relative.as_posix(), opened_identity))
        finally:
            os.close(child)
    return tuple(identities), tuple(hashes)


def _source_tree_identity(
    path: pathlib.Path,
    *,
    label: str,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    descriptor = _open_source_node_no_follow(path, label=label, directory=True)
    try:
        identities, _hashes = _walk_source_tree(descriptor, label=label)
        return identities
    finally:
        os.close(descriptor)


def _source_tree_hashes(
    path: pathlib.Path,
    *,
    label: str,
) -> tuple[tuple[str, str], ...]:
    descriptor = _open_source_node_no_follow(path, label=label, directory=True)
    try:
        before, _ = _walk_source_tree(descriptor, label=label)
        during, hashes = _walk_source_tree(
            descriptor,
            label=label,
            hash_files=True,
        )
        after, _ = _walk_source_tree(descriptor, label=label)
    finally:
        os.close(descriptor)
    if not hashes:
        raise ValueError(f"{label} tree is empty")
    if before != during or during != after or _source_tree_identity(path, label=label) != before:
        raise RuntimeError(f"{label} tree changed while it was authenticated")
    return hashes


def _copy_private_file(source: pathlib.Path, destination: pathlib.Path, *, label: str) -> None:
    """Copy one source through a pinned descriptor and verify both byte identities."""
    descriptor = _open_source_node_no_follow(source, label=label, directory=False)
    try:
        before = _source_node_identity(os.fstat(descriptor))
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
        )
        try:
            source_digest = hashlib.sha256()
            os.lseek(descriptor, 0, os.SEEK_SET)
            for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
                source_digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(destination_descriptor, view)
                    view = view[written:]
        finally:
            os.close(destination_descriptor)
        if _source_node_identity(os.fstat(descriptor)) != before:
            raise RuntimeError(f"{label} changed while its private snapshot was copied")
    finally:
        os.close(descriptor)
    if _source_file_identity(source, label=label) != before:
        raise RuntimeError(f"{label} path changed while its private snapshot was copied")
    snapshot_descriptor = _open_source_node_no_follow(
        destination,
        label=f"private {label} snapshot",
        directory=False,
    )
    try:
        copied_digest = _descriptor_sha256(snapshot_descriptor)
    finally:
        os.close(snapshot_descriptor)
    if copied_digest != source_digest.hexdigest():
        raise RuntimeError(f"private {label} snapshot bytes differ from its source")


def _copy_private_tree(source: pathlib.Path, destination: pathlib.Path, *, label: str) -> None:
    """Copy an unfiltered tree through descriptors, then verify topology and bytes."""
    destination.mkdir(mode=0o700)
    descriptor = _open_source_node_no_follow(source, label=label, directory=True)
    try:
        before, _ = _walk_source_tree(descriptor, label=label)
        during, source_hashes = _walk_source_tree(
            descriptor,
            label=label,
            copy_root=destination,
            hash_files=True,
        )
        after, _ = _walk_source_tree(descriptor, label=label)
    finally:
        os.close(descriptor)
    if before != during or during != after or _source_tree_identity(source, label=label) != before:
        raise RuntimeError(f"{label} tree changed while its private snapshot was copied")
    copied_hashes = _source_tree_hashes(destination, label=f"private {label} snapshot")
    if copied_hashes != source_hashes:
        raise RuntimeError(f"private {label} snapshot differs from its source tree")


def _copy_private_file_or_tree(
    source: pathlib.Path,
    root: pathlib.Path,
    *,
    label: str,
) -> pathlib.Path:
    """Copy one authenticated artifact without guessing its kind through a followed path."""
    try:
        descriptor = _open_source_node_no_follow(source, label=label, directory=True)
    except ValueError:
        file_parent = root / "panel-file"
        file_parent.mkdir(mode=0o700)
        destination = file_parent / source.name
        _copy_private_file(source, destination, label=label)
        return destination
    else:
        os.close(descriptor)
        destination = root / "source-panel"
        _copy_private_tree(source, destination, label=label)
        return destination


def _normalized_distribution_name(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("runtime distribution names must be nonblank strings")
    normalized = re.sub(r"[-_.]+", "-", value.strip()).lower()
    if normalized != value:
        raise ValueError(
            f"runtime distribution name {value!r} is not normalized as {normalized!r}"
        )
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized) is None:
        raise ValueError(f"runtime distribution name {normalized!r} is not a valid projection key")
    return normalized


def _exact_package_map(value: object) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise ValueError("runtime package_versions must be a nonempty JSON object")
    packages: dict[str, str] = {}
    for name, version in value.items():
        normalized = _normalized_distribution_name(name)
        if (
            type(version) is not str
            or not version.strip()
            or version != version.strip()
            or any(character.isspace() for character in version)
            or "==" in version
        ):
            raise ValueError(f"runtime version for {normalized!r} must be a nonblank string")
        packages[normalized] = version
    missing = sorted(_REQUIRED_RUNTIME_DISTRIBUTIONS - set(packages))
    if missing:
        raise ValueError(
            "TimesFM runtime lock omits required worker distributions: "
            f"{missing}"
        )
    return packages


def runtime_lock_projection_sha256(package_versions: Mapping[str, str]) -> str:
    """Hash the registered full distribution map with one unambiguous projection."""
    packages = _exact_package_map(package_versions)
    payload = "".join(
        f"{name}=={packages[name]}\n" for name in sorted(packages)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _installed_package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if type(raw_name) is not str or not raw_name.strip():
            raise ValueError("installed runtime contains a distribution without a Name")
        name = re.sub(r"[-_.]+", "-", raw_name.strip()).lower()
        version = distribution.version
        if type(version) is not str or not version.strip():
            raise ValueError(f"installed runtime distribution {name!r} has no version")
        if name in packages:
            raise ValueError(f"installed runtime repeats distribution {name!r}")
        packages[name] = version
    return packages


def validate_runtime_lock(
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    *,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
) -> RuntimeLockIdentity:
    """Authenticate an exact-JSON lock source and the worker's complete package map.

    ``runtime_lock_record_path`` is a JSON-object key path.  It permits the lock record to live
    either at the root of a dedicated artifact (empty path) or inside an already authenticated
    prospective registration such as ``("registration", "runtime")``.  The projection digest
    is distinct from the source-file digest and hashes every normalized installed distribution.
    """
    expected_source_digest = _require_sha256(
        runtime_lock_file_sha256,
        "runtime_lock_file_sha256",
    )
    expected_projection_digest = _require_sha256(
        runtime_lock_sha256,
        "runtime_lock_sha256",
    )
    source = pathlib.Path(runtime_lock_path)
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise ValueError("runtime_lock_path must be an absolute regular exact-JSON file")
    if isinstance(runtime_lock_record_path, str | bytes) or not isinstance(
        runtime_lock_record_path,
        Sequence,
    ):
        raise TypeError("runtime_lock_record_path must be an explicit sequence of JSON keys")
    record_path = tuple(runtime_lock_record_path)
    if any(type(key) is not str or not key for key in record_path):
        raise ValueError("runtime_lock_record_path contains a blank or non-string key")
    before = _file_sha256(source)
    if before != expected_source_digest:
        raise ValueError(
            "TimesFM runtime-lock source hash mismatch: "
            f"expected {expected_source_digest}, got {before}"
        )
    node: object = _strict_record(source, "TimesFM runtime-lock source")
    for key in record_path:
        if type(node) is not dict or key not in node:
            raise ValueError(f"TimesFM runtime-lock record path is absent at {key!r}")
        node = node[key]
    if type(node) is not dict:
        raise ValueError("TimesFM runtime-lock record must be one JSON object")
    if node.get("runtime_lock_canonicalization") != RUNTIME_LOCK_CANONICALIZATION:
        raise ValueError("TimesFM runtime-lock canonicalization is unknown")
    if node.get("runtime_lock_sha256") != expected_projection_digest:
        raise ValueError("TimesFM runtime-lock record binds a different projection digest")
    packages = _exact_package_map(node.get("package_versions"))
    projected = runtime_lock_projection_sha256(packages)
    if projected != expected_projection_digest:
        raise ValueError(
            "TimesFM runtime package map does not produce its registered digest: "
            f"expected {expected_projection_digest}, got {projected}"
        )
    python = node.get("python")
    observed_python = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if type(python) is not str or python != observed_python:
        raise ValueError(
            f"TimesFM runtime Python differs: expected {python!r}, got {observed_python!r}"
        )
    observed_packages = _installed_package_versions()
    if observed_packages != packages:
        missing = sorted(set(packages) - set(observed_packages))
        extra = sorted(set(observed_packages) - set(packages))
        changed = {
            name: (packages[name], observed_packages[name])
            for name in sorted(set(packages) & set(observed_packages))
            if packages[name] != observed_packages[name]
        }
        raise ValueError(
            "TimesFM installed runtime differs from its complete package lock: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    after = _file_sha256(source)
    if after != before:
        raise RuntimeError("TimesFM runtime-lock source changed while it was validated")
    return RuntimeLockIdentity(
        source_path=str(source),
        source_sha256=before,
        record_path=record_path,
        projection_sha256=projected,
        python=python,
        package_versions=tuple(sorted(packages.items())),
    )


def _runtime_lock_record(identity: RuntimeLockIdentity) -> dict[str, object]:
    return {
        "source_path": identity.source_path,
        "source_sha256": identity.source_sha256,
        "record_path": list(identity.record_path),
        "projection_sha256": identity.projection_sha256,
        "python": identity.python,
        "n_distributions": len(identity.package_versions),
    }


def _validate_snapshotted_runtime_lock(
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    *,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    recorded_identity: RuntimeLockIdentity,
) -> RuntimeLockIdentity:
    """Authenticate copied lock bytes while preserving the registered source-path record."""
    if not isinstance(recorded_identity, RuntimeLockIdentity):
        raise TypeError("recorded runtime-lock identity is malformed")
    copied = validate_runtime_lock(
        runtime_lock_path,
        runtime_lock_file_sha256,
        runtime_lock_record_path=runtime_lock_record_path,
        runtime_lock_sha256=runtime_lock_sha256,
    )
    rebound = dataclasses.replace(copied, source_path=recorded_identity.source_path)
    if rebound != recorded_identity:
        raise RuntimeError("private TimesFM runtime-lock snapshot differs from its source")
    return recorded_identity


def _coerce_runtime_lock_identity(value: object) -> RuntimeLockIdentity | None:
    """Parse the exact inert form accepted across an isolated-worker JSON boundary."""
    if value is None or isinstance(value, RuntimeLockIdentity):
        return value
    expected_fields = {
        "source_path",
        "source_sha256",
        "record_path",
        "projection_sha256",
        "python",
        "package_versions",
    }
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError("recorded_runtime_lock must be one exact runtime-lock identity")
    source_path = value["source_path"]
    record_path = value["record_path"]
    python = value["python"]
    package_versions = value["package_versions"]
    if (
        type(source_path) is not str
        or not pathlib.Path(source_path).is_absolute()
        or isinstance(record_path, str | bytes)
        or not isinstance(record_path, Sequence)
        or any(type(key) is not str or not key for key in record_path)
        or type(python) is not str
        or not python
    ):
        raise ValueError("recorded_runtime_lock contains an invalid path/runtime identity")
    if isinstance(package_versions, Mapping):
        packages = _exact_package_map(package_versions)
    elif isinstance(package_versions, Sequence) and not isinstance(
        package_versions, str | bytes
    ):
        try:
            packages = _exact_package_map(dict(package_versions))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "recorded_runtime_lock package_versions are malformed"
            ) from exc
        if len(packages) != len(package_versions):
            raise ValueError("recorded_runtime_lock repeats a package version")
    else:
        raise ValueError("recorded_runtime_lock package_versions are malformed")
    projection = _require_sha256(
        value["projection_sha256"],
        "recorded runtime projection_sha256",
    )
    if runtime_lock_projection_sha256(packages) != projection:
        raise ValueError("recorded_runtime_lock package projection does not verify")
    return RuntimeLockIdentity(
        source_path=source_path,
        source_sha256=_require_sha256(
            value["source_sha256"],
            "recorded runtime source_sha256",
        ),
        record_path=tuple(record_path),
        projection_sha256=projection,
        python=python,
        package_versions=tuple(sorted(packages.items())),
    )


def _tree_sha256(path: pathlib.Path) -> str:
    """Content-bind a PEFT tree after a complete no-follow descriptor walk."""
    files = _source_tree_hashes(path, label="LoRA adapter")
    digest = hashlib.sha256(b"autoforecast.timesfm-lora-tree.v1\0")
    for relative_value, file_digest in files:
        relative = relative_value.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def lora_adapter_tree_sha256(path: str | pathlib.Path) -> str:
    """Return the one public canonical identity used by governed LoRA artifacts."""
    return _tree_sha256(pathlib.Path(path))


def _adapter_model_file_sha256(path: pathlib.Path) -> str:
    model_file = path / "adapter_model.safetensors"
    if model_file.is_symlink() or not model_file.is_file():
        raise ValueError(
            "TimesFM LoRA adapter must contain a regular adapter_model.safetensors"
        )
    return _file_sha256(model_file)


def _strict_record(path: pathlib.Path, label: str) -> dict[str, object]:
    try:
        record = load_exact_json(path)
    except ExactJSONError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if type(record) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    return record


def _validate_base_checkpoint(
    checkpoint_dir: pathlib.Path,
    model_revision: str,
) -> tuple[str, str]:
    if model_revision != TRANSFORMERS_REVISION:
        raise ValueError(
            "TimesFM Transformers requires the exact immutable checkpoint revision "
            f"{TRANSFORMERS_REVISION}, got {model_revision!r}"
        )
    if not checkpoint_dir.is_absolute():
        raise ValueError(
            "TimesFM Transformers checkpoint_dir must be absolute; PATH or cwd discovery is "
            "not a reproducible offline identity"
        )
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"TimesFM Transformers checkpoint directory is absent: {checkpoint_dir}")
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(
            "TimesFM Transformers checkpoint must contain config.json and model.safetensors"
        )
    config_digest = _file_sha256(config_path)
    weights_digest = _file_sha256(weights_path)
    if config_digest != TRANSFORMERS_CONFIG_SHA256:
        raise ValueError(
            "TimesFM Transformers config hash differs from the exact registered conversion: "
            f"{config_digest}"
        )
    if weights_digest != TRANSFORMERS_WEIGHTS_SHA256:
        raise ValueError(
            "TimesFM Transformers weights hash differs from the exact registered conversion: "
            f"{weights_digest}"
        )
    config = _strict_record(config_path, "TimesFM Transformers config")
    required = {
        "architectures": ["TimesFm2_5ModelForPrediction"],
        "decode_index": 5,
        "dtype": "float32",
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "use_continuous_quantile_head": True,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if not json_values_equal_exact(config.get(key), expected)
    }
    if config.get("horizon_length", 0) < GOVERNED_HORIZON:
        mismatches["horizon_length"] = (config.get("horizon_length"), f">={GOVERNED_HORIZON}")
    if not json_values_equal_exact(
        config.get("quantiles"), [index / 10 for index in range(1, 10)]
    ):
        mismatches["quantiles"] = (config.get("quantiles"), "0.1 through 0.9")
    if mismatches:
        raise ValueError(f"TimesFM Transformers config semantics differ: {mismatches}")
    return config_digest, weights_digest


def _config_record(config: LoRATrainingConfig) -> dict[str, object]:
    return {
        **dataclasses.asdict(config),
        "precision": "float32",
        "external_normalization": "none_timesfm_revin_only",
        "objective": "transformers_native_mse_plus_quantile_loss",
        "optimizer": "torch.optim.AdamW",
        "scheduler": "CosineAnnealingLR_per_batch",
        "target_modules": "all-linear",
        "bias": "none",
        "base_weights_frozen": True,
        "force_flip_invariance": True,
        "evaluation_during_training": False,
        "early_stopping": False,
        "best_epoch_selection": False,
        "checkpoint_publication": "once_after_final_epoch",
    }


def _validate_continuation_config(config: LoRAContinuationConfig) -> None:
    if not isinstance(config, LoRAContinuationConfig):
        raise TypeError("continuation_config must be an explicit LoRAContinuationConfig")
    for field in ("max_additional_epochs", "patience"):
        value = getattr(config, field)
        if type(value) is not int or value < 1:
            raise ValueError(f"LoRA continuation {field} must be a positive integer")
    if config.patience > config.max_additional_epochs:
        raise ValueError(
            "LoRA continuation patience cannot exceed max_additional_epochs"
        )
    for field in (
        "restart_learning_rate",
        "absolute_min_delta",
        "relative_min_delta",
    ):
        value = getattr(config, field)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"LoRA continuation {field} must be finite and positive")


def _continuation_training_record(
    source_config: LoRATrainingConfig,
    continuation_config: LoRAContinuationConfig,
) -> dict[str, object]:
    return {
        "context_length": source_config.context_length,
        "horizon_length": source_config.horizon_length,
        "source_pool_epochs": source_config.epochs,
        "batch_size": source_config.batch_size,
        "rank": source_config.rank,
        "alpha": source_config.alpha,
        "dropout": source_config.dropout,
        "weight_decay": source_config.weight_decay,
        "gradient_clip_norm": source_config.gradient_clip_norm,
        "seed": source_config.seed,
        "device": source_config.device,
        "precision": "float32",
        "objective": "transformers_native_mse_plus_quantile_loss",
        "target_modules": "all-linear",
        "bias": "none",
        "base_weights_frozen": True,
        "adapter_only_trainable": True,
        "force_flip_invariance": True,
        "external_normalization": "none_timesfm_revin_only",
        "max_additional_epochs": continuation_config.max_additional_epochs,
        "patience": continuation_config.patience,
        "restart_learning_rate": continuation_config.restart_learning_rate,
        "optimizer": "torch.optim.AdamW",
        "optimizer_state_restored": False,
        "scheduler": "CosineAnnealingLR_per_batch",
        "scheduler_state_restored": False,
        "source_epoch_cycle": "(additional_epoch-1)%3",
        "accuracy_evaluation_during_training": False,
        "checkpoint_publication": "once_after_best_adapter_restoration",
    }


def _validate_training_config(config: LoRATrainingConfig) -> None:
    if not isinstance(config, LoRATrainingConfig):
        raise TypeError("config must be an explicit LoRATrainingConfig")
    integer_fields = (
        "context_length",
        "horizon_length",
        "epochs",
        "batch_size",
        "rank",
        "alpha",
    )
    for field in integer_fields:
        value = getattr(config, field)
        if type(value) is not int or value < 1:
            raise ValueError(f"LoRA {field} must be a positive integer, got {value!r}")
    if config.context_length % 32:
        raise ValueError("LoRA context_length must be a multiple of TimesFM's 32-day patch")
    if config.horizon_length > 128:
        raise ValueError("LoRA horizon_length exceeds the pinned checkpoint's 128-day head")
    for field in ("learning_rate", "weight_decay", "gradient_clip_norm"):
        value = getattr(config, field)
        if (
            type(value) is not float
            or not math.isfinite(value)
        ):
            raise ValueError(f"LoRA {field} must be finite, got {value!r}")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.gradient_clip_norm <= 0:
        raise ValueError("LoRA learning_rate and gradient_clip_norm must be >0; weight_decay >=0")
    if (
        type(config.dropout) is not float
        or not math.isfinite(config.dropout)
        or not 0 <= config.dropout < 1
    ):
        raise ValueError(f"LoRA dropout must be in [0, 1), got {config.dropout!r}")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError(f"LoRA seed must be a non-negative integer, got {config.seed!r}")
    if config.device not in ("cpu", "mps"):
        raise ValueError("LoRA device must be explicitly 'cpu' or 'mps'")


def _validate_provider_options(
    checkpoint_dir: pathlib.Path,
    model_revision: str,
    context_length: int,
    device: str,
    adapter_dir: pathlib.Path | None,
    adapter_sha256: str | None,
    forecast_origin: str | None,
) -> None:
    if not checkpoint_dir.is_absolute():
        raise ValueError("TimesFM Transformers checkpoint_dir must be absolute")
    if model_revision != TRANSFORMERS_REVISION:
        raise ValueError("TimesFM Transformers model_revision is not the exact pinned revision")
    if type(context_length) is not int or context_length < 32 or context_length % 32:
        raise ValueError("TimesFM Transformers context_length must be a positive multiple of 32")
    if context_length > 16384:
        raise ValueError("TimesFM Transformers context_length exceeds the checkpoint limit")
    if device not in ("cpu", "mps"):
        raise ValueError("TimesFM Transformers device must be explicitly 'cpu' or 'mps'")
    if (adapter_dir is None) != (adapter_sha256 is None):
        raise ValueError("adapter_dir and adapter_sha256 must be supplied together")
    if adapter_dir is not None:
        if not adapter_dir.is_absolute():
            raise ValueError("TimesFM LoRA adapter_dir must be absolute")
        _require_sha256(adapter_sha256, "adapter_sha256")
        try:
            forecast_day = (
                dt.date.fromisoformat(forecast_origin)
                if type(forecast_origin) is str
                else None
            )
        except ValueError as exc:
            raise ValueError(
                "a LoRA provider forecast_origin must be a canonical ISO date"
            ) from exc
        if forecast_day is None or forecast_day.isoformat() != forecast_origin:
            raise ValueError("a LoRA provider requires an explicit canonical forecast_origin")


def _base_model_record() -> dict[str, object]:
    return {
        "model_id": TRANSFORMERS_MODEL_ID,
        "revision": TRANSFORMERS_REVISION,
        "config_sha256": TRANSFORMERS_CONFIG_SHA256,
        "weights_sha256": TRANSFORMERS_WEIGHTS_SHA256,
    }


def _validate_v1_lora_manifest(manifest: dict[str, object]) -> None:
    expected_fields = {
        "schema",
        "base_model",
        "training_rows",
        "training",
        "runtime",
        "parameters",
        "publication",
    }
    if set(manifest) != expected_fields or manifest.get("schema") != LORA_ARTIFACT_SCHEMA:
        raise ValueError("TimesFM LoRA manifest has an unknown schema or field set")
    if not json_values_equal_exact(manifest.get("base_model"), _base_model_record()):
        raise ValueError("TimesFM LoRA manifest was trained from a different base checkpoint")
    training = manifest.get("training")
    training_fields = {
        "context_length",
        "horizon_length",
        "epochs",
        "batch_size",
        "rank",
        "alpha",
        "dropout",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "seed",
        "device",
        "precision",
        "external_normalization",
        "objective",
        "optimizer",
        "scheduler",
        "target_modules",
        "bias",
        "base_weights_frozen",
        "force_flip_invariance",
        "evaluation_during_training",
        "early_stopping",
        "best_epoch_selection",
        "checkpoint_publication",
    }
    if type(training) is not dict or set(training) != training_fields or any(
        not json_values_equal_exact(training.get(key), value)
        for key, value in {
            "precision": "float32",
            "external_normalization": "none_timesfm_revin_only",
            "objective": "transformers_native_mse_plus_quantile_loss",
            "optimizer": "torch.optim.AdamW",
            "scheduler": "CosineAnnealingLR_per_batch",
            "target_modules": "all-linear",
            "bias": "none",
            "base_weights_frozen": True,
            "force_flip_invariance": True,
            "evaluation_during_training": False,
            "early_stopping": False,
            "best_epoch_selection": False,
            "checkpoint_publication": "once_after_final_epoch",
        }.items()
    ):
        raise ValueError("TimesFM LoRA manifest violates the final-only all-linear contract")
    if any(
        type(training.get(field)) is not int or training[field] < 1
        for field in (
            "context_length",
            "horizon_length",
            "epochs",
            "batch_size",
            "rank",
            "alpha",
        )
    ) or type(training.get("seed")) is not int or training["seed"] < 0:
        raise ValueError("TimesFM LoRA manifest has invalid integer training controls")
    float_fields = ("dropout", "learning_rate", "weight_decay", "gradient_clip_norm")
    if any(
        type(training.get(field)) is not float or not math.isfinite(training[field])
        for field in float_fields
    ) or (
        not 0 <= training["dropout"] < 1
        or training["learning_rate"] <= 0
        or training["weight_decay"] < 0
        or training["gradient_clip_norm"] <= 0
        or training.get("device") not in {"cpu", "mps"}
    ):
        raise ValueError("TimesFM LoRA manifest has invalid floating-point training controls")
    training_rows = manifest.get("training_rows")
    runtime = manifest.get("runtime")
    parameters = manifest.get("parameters")
    if (
        type(training_rows) is not dict
        or type(training_rows.get("n_rows")) is not int
        or training_rows["n_rows"] < 1
        or type(runtime) is not dict
        or type(runtime.get("runtime_lock")) is not dict
        or type(runtime["runtime_lock"].get("n_distributions")) is not int
        or runtime["runtime_lock"]["n_distributions"] < 1
        or type(parameters) is not dict
        or set(parameters) != {"trainable", "total"}
        or any(type(value) is not int or value < 1 for value in parameters.values())
        or parameters["trainable"] > parameters["total"]
    ):
        raise ValueError("TimesFM LoRA manifest has invalid produced integer evidence")
    publication = manifest.get("publication")
    if not json_values_equal_exact(publication, {
        "saved_after_all_epochs": True,
        "intermediate_checkpoints": 0,
        "evaluation_runs": 0,
    }):
        raise ValueError("TimesFM LoRA manifest is not a final-only artifact")


def _validate_continuation_lora_manifest(manifest: dict[str, object]) -> None:
    expected_fields = {
        "schema",
        "base_model",
        "source_adapter",
        "training_rows",
        "split",
        "training",
        "stopping",
        "runtime",
        "parameters",
        "publication",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("schema") != LORA_CONTINUATION_ARTIFACT_SCHEMA
    ):
        raise ValueError("TimesFM LoRA continuation manifest has an unknown field set")
    if not json_values_equal_exact(manifest.get("base_model"), _base_model_record()):
        raise ValueError(
            "TimesFM LoRA continuation was trained from a different base checkpoint"
        )
    source = manifest.get("source_adapter")
    source_fields = {
        "path",
        "tree_sha256",
        "adapter_model_file",
        "adapter_model_sha256",
        "manifest_path",
        "manifest_sha256",
        "manifest_schema",
    }
    if (
        type(source) is not dict
        or set(source) != source_fields
        or source.get("manifest_schema") != LORA_ARTIFACT_SCHEMA
        or source.get("adapter_model_file") != "adapter_model.safetensors"
    ):
        raise ValueError("TimesFM LoRA continuation has an invalid source-adapter identity")
    source_path = pathlib.Path(str(source.get("path")))
    source_manifest_path = pathlib.Path(str(source.get("manifest_path")))
    if (
        type(source.get("path")) is not str
        or not source_path.is_absolute()
        or type(source.get("manifest_path")) is not str
        or source_manifest_path != source_path / LORA_MANIFEST
    ):
        raise ValueError("TimesFM LoRA continuation source paths are not canonical")
    for field in ("tree_sha256", "adapter_model_sha256", "manifest_sha256"):
        _require_sha256(source.get(field), f"source adapter {field}")
    rows = manifest.get("training_rows")
    if type(rows) is not dict or set(rows) != {
        "sha256",
        "manifest_sha256",
        "n_rows",
        "cutoff",
        "panel_content_sha256",
    }:
        raise ValueError("TimesFM LoRA continuation has an invalid training-row identity")
    for field in ("sha256", "manifest_sha256", "panel_content_sha256"):
        _require_sha256(rows.get(field), f"continuation training rows {field}")
    if type(rows.get("n_rows")) is not int or rows["n_rows"] < 1:
        raise ValueError("TimesFM LoRA continuation has no positive training-row count")
    try:
        cutoff = dt.date.fromisoformat(rows["cutoff"])
    except (TypeError, ValueError) as exc:
        raise ValueError("TimesFM LoRA continuation cutoff is not a canonical date") from exc
    if cutoff.isoformat() != rows["cutoff"]:
        raise ValueError("TimesFM LoRA continuation cutoff is not a canonical date")
    split = manifest.get("split")
    if type(split) is not dict or set(split) != {
        "schema",
        "algorithm",
        "validation_fraction",
        "validation_rounding",
        "n_series",
        "n_train_series",
        "n_validation_series",
        "assignment_sha256",
        "series_disjoint",
        "validation_source_epochs",
    }:
        raise ValueError("TimesFM LoRA continuation has an invalid series split")
    if (
        split.get("schema") != "sha256_series_train_validation_split_v1"
        or split.get("algorithm")
        != "ascending_sha256(series_id_utf8)_then_series_id_utf8"
        or split.get("validation_fraction") != 0.1
        or split.get("validation_rounding")
        != "first_max(1,floor(0.1*n_series))_with_nonempty_training"
        or split.get("series_disjoint") is not True
    ):
        raise ValueError("TimesFM LoRA continuation changed its fixed split semantics")
    n_series = split.get("n_series")
    n_train = split.get("n_train_series")
    n_validation = split.get("n_validation_series")
    if (
        type(n_series) is not int
        or type(n_train) is not int
        or type(n_validation) is not int
        or min(n_train, n_validation) < 1
        or n_train + n_validation != n_series
    ):
        raise ValueError("TimesFM LoRA continuation split counts are inconsistent")
    _require_sha256(split.get("assignment_sha256"), "continuation split assignment_sha256")
    validation_epochs = split.get("validation_source_epochs")
    if (
        type(validation_epochs) is not list
        or not json_values_equal_exact(validation_epochs, [0, 1, 2])
    ):
        raise ValueError("TimesFM LoRA continuation validation rows are not the fixed 3-row pool")
    training = manifest.get("training")
    training_fields = {
        "context_length",
        "horizon_length",
        "source_pool_epochs",
        "batch_size",
        "rank",
        "alpha",
        "dropout",
        "weight_decay",
        "gradient_clip_norm",
        "seed",
        "device",
        "precision",
        "objective",
        "target_modules",
        "bias",
        "base_weights_frozen",
        "adapter_only_trainable",
        "force_flip_invariance",
        "external_normalization",
        "max_additional_epochs",
        "patience",
        "restart_learning_rate",
        "optimizer",
        "optimizer_state_restored",
        "scheduler",
        "scheduler_state_restored",
        "source_epoch_cycle",
        "accuracy_evaluation_during_training",
        "checkpoint_publication",
    }
    if type(training) is not dict or set(training) != training_fields or any(
        not json_values_equal_exact(training.get(key), value)
        for key, value in {
            "precision": "float32",
            "objective": "transformers_native_mse_plus_quantile_loss",
            "target_modules": "all-linear",
            "bias": "none",
            "base_weights_frozen": True,
            "adapter_only_trainable": True,
            "force_flip_invariance": True,
            "external_normalization": "none_timesfm_revin_only",
            "optimizer": "torch.optim.AdamW",
            "optimizer_state_restored": False,
            "scheduler": "CosineAnnealingLR_per_batch",
            "scheduler_state_restored": False,
            "source_epoch_cycle": "(additional_epoch-1)%3",
            "accuracy_evaluation_during_training": False,
            "checkpoint_publication": "once_after_best_adapter_restoration",
        }.items()
    ):
        raise ValueError("TimesFM LoRA continuation violates the weights-only training contract")
    positive_integer_fields = (
        "context_length",
        "horizon_length",
        "source_pool_epochs",
        "batch_size",
        "rank",
        "alpha",
        "max_additional_epochs",
        "patience",
    )
    if any(
        type(training.get(field)) is not int or training[field] < 1
        for field in positive_integer_fields
    ) or training["patience"] > training["max_additional_epochs"]:
        raise ValueError("TimesFM LoRA continuation has invalid epoch or architecture counts")
    if type(training.get("seed")) is not int or training["seed"] < 0:
        raise ValueError("TimesFM LoRA continuation has an invalid split/training seed")
    finite_training_fields = (
        "dropout",
        "weight_decay",
        "gradient_clip_norm",
        "restart_learning_rate",
    )
    if any(
        type(training.get(field)) is not float
        or not math.isfinite(training[field])
        for field in finite_training_fields
    ):
        raise ValueError("TimesFM LoRA continuation has a non-finite numerical control")
    if (
        not 0 <= training["dropout"] < 1
        or training["weight_decay"] < 0
        or training["gradient_clip_norm"] <= 0
        or training["restart_learning_rate"] <= 0
        or training.get("device") not in {"cpu", "mps"}
    ):
        raise ValueError("TimesFM LoRA continuation has an invalid numerical control")
    if training.get("source_pool_epochs") != len(validation_epochs):
        raise ValueError("TimesFM LoRA continuation split and source pool disagree")
    if (
        rows["n_rows"] != split["n_series"] * training["source_pool_epochs"]
        or split["n_validation_series"]
        != max(1, min(split["n_series"] - 1, math.floor(split["n_series"] * 0.1)))
    ):
        raise ValueError("TimesFM LoRA continuation split is inconsistent with its row pool")
    stopping = manifest.get("stopping")
    if type(stopping) is not dict or set(stopping) != {
        "metric",
        "baseline_epoch",
        "baseline_validation_loss",
        "best_epoch",
        "best_validation_loss",
        "last_training_loss",
        "last_validation_loss",
        "actual_additional_epochs",
        "patience",
        "waited_at_stop",
        "stop_reason",
        "improvement_rule",
        "absolute_min_delta",
        "relative_min_delta",
        "best_adapter_restored",
        "validation_trace",
    }:
        raise ValueError("TimesFM LoRA continuation has malformed early-stopping evidence")
    finite_losses = (
        stopping.get("baseline_validation_loss"),
        stopping.get("best_validation_loss"),
        stopping.get("last_training_loss"),
        stopping.get("last_validation_loss"),
    )
    if any(
        type(value) is not float
        or not math.isfinite(value)
        for value in finite_losses
    ):
        raise ValueError("TimesFM LoRA continuation records a non-finite loss")
    actual_epochs = stopping.get("actual_additional_epochs")
    best_epoch = stopping.get("best_epoch")
    deltas = (
        stopping.get("absolute_min_delta"),
        stopping.get("relative_min_delta"),
    )
    if (
        stopping.get("metric") != "mean_native_loss_on_fixed_validation_rows"
        or not json_values_equal_exact(stopping.get("baseline_epoch"), 0)
        or stopping.get("improvement_rule")
        != (
            "validation_loss < best_loss - "
            "max(absolute_min_delta,relative_min_delta*abs(best_loss))"
        )
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value <= 0
            for value in deltas
        )
        or type(actual_epochs) is not int
        or not 1 <= actual_epochs <= training.get("max_additional_epochs", 0)
        or type(best_epoch) is not int
        or not 0 <= best_epoch <= actual_epochs
        or not json_values_equal_exact(
            stopping.get("patience"), training.get("patience")
        )
        or stopping.get("best_adapter_restored") is not True
        or type(stopping.get("waited_at_stop")) is not int
        or not 0 <= stopping["waited_at_stop"] <= stopping["patience"]
        or stopping.get("best_validation_loss")
        > stopping.get("baseline_validation_loss")
        or stopping.get("stop_reason")
        not in {"early_stopping_patience", "maximum_additional_epochs"}
    ):
        raise ValueError("TimesFM LoRA continuation early-stopping semantics are inconsistent")
    if stopping["stop_reason"] == "early_stopping_patience" and (
        stopping["waited_at_stop"] != stopping["patience"]
    ):
        raise ValueError("TimesFM LoRA continuation stopped before patience was exhausted")
    if stopping["stop_reason"] == "maximum_additional_epochs" and (
        actual_epochs != training["max_additional_epochs"]
    ):
        raise ValueError("TimesFM LoRA continuation did not reach its recorded maximum")
    validation_trace = stopping.get("validation_trace")
    if type(validation_trace) is not list or len(validation_trace) != actual_epochs + 1:
        raise ValueError("TimesFM LoRA continuation validation trace has the wrong length")
    recomputed_best_epoch = 0
    recomputed_best_loss: float | None = None
    recomputed_waited = 0
    for additional_epoch, trace_row in enumerate(validation_trace):
        if additional_epoch > 0 and recomputed_waited >= training["patience"]:
            raise ValueError(
                "TimesFM LoRA continuation trace continues after patience was exhausted"
            )
        if type(trace_row) is not dict or set(trace_row) != {
            "additional_epoch",
            "source_row_epoch",
            "training_loss",
            "validation_loss",
        }:
            raise ValueError("TimesFM LoRA continuation validation trace is malformed")
        expected_source_epoch = None if additional_epoch == 0 else (additional_epoch - 1) % 3
        training_loss = trace_row.get("training_loss")
        validation_loss = trace_row.get("validation_loss")
        if (
            not json_values_equal_exact(
                trace_row.get("additional_epoch"), additional_epoch
            )
            or not json_values_equal_exact(
                trace_row.get("source_row_epoch"), expected_source_epoch
            )
            or (
                additional_epoch == 0
                and training_loss is not None
            )
            or (
                additional_epoch > 0
                and (
                    type(training_loss) is not float
                    or not math.isfinite(training_loss)
                )
            )
            or type(validation_loss) is not float
            or not math.isfinite(validation_loss)
        ):
            raise ValueError("TimesFM LoRA continuation validation trace row is invalid")
        if additional_epoch == 0:
            recomputed_best_loss = float(validation_loss)
            continue
        assert recomputed_best_loss is not None
        threshold = max(deltas[0], deltas[1] * abs(recomputed_best_loss))
        if validation_loss < recomputed_best_loss - threshold:
            recomputed_best_epoch = additional_epoch
            recomputed_best_loss = float(validation_loss)
            recomputed_waited = 0
        else:
            recomputed_waited += 1
    if (
        validation_trace[0]["validation_loss"] != stopping["baseline_validation_loss"]
        or validation_trace[-1]["training_loss"] != stopping["last_training_loss"]
        or validation_trace[-1]["validation_loss"] != stopping["last_validation_loss"]
        or recomputed_best_epoch != best_epoch
        or recomputed_best_loss != stopping["best_validation_loss"]
        or recomputed_waited != stopping["waited_at_stop"]
    ):
        raise ValueError("TimesFM LoRA continuation stopping summary differs from its trace")
    if (
        stopping["stop_reason"] == "maximum_additional_epochs"
        and recomputed_waited >= training["patience"]
    ):
        raise ValueError(
            "TimesFM LoRA continuation maximum-epoch trace already exhausted patience"
        )
    runtime = manifest.get("runtime")
    if type(runtime) is not dict or set(runtime) != {
        "torch",
        "transformers",
        "peft",
        "device",
        "precision",
        "runtime_lock",
    }:
        raise ValueError("TimesFM LoRA continuation runtime identity is malformed")
    if runtime.get("device") != training["device"] or runtime.get("precision") != "float32":
        raise ValueError("TimesFM LoRA continuation runtime differs from its training controls")
    runtime_lock = runtime.get("runtime_lock")
    if type(runtime_lock) is not dict or set(runtime_lock) != {
        "source_path",
        "source_sha256",
        "record_path",
        "projection_sha256",
        "python",
        "n_distributions",
    }:
        raise ValueError("TimesFM LoRA continuation runtime-lock identity is malformed")
    _require_sha256(runtime_lock.get("source_sha256"), "continuation runtime source_sha256")
    _require_sha256(
        runtime_lock.get("projection_sha256"),
        "continuation runtime projection_sha256",
    )
    if (
        type(runtime_lock.get("source_path")) is not str
        or not pathlib.Path(runtime_lock["source_path"]).is_absolute()
        or type(runtime_lock.get("record_path")) is not list
        or any(type(key) is not str or not key for key in runtime_lock["record_path"])
        or type(runtime_lock.get("python")) is not str
        or type(runtime_lock.get("n_distributions")) is not int
        or runtime_lock["n_distributions"] < 1
    ):
        raise ValueError("TimesFM LoRA continuation runtime-lock identity is invalid")
    parameters = manifest.get("parameters")
    if (
        type(parameters) is not dict
        or set(parameters) != {"trainable", "total"}
        or any(type(value) is not int or value < 1 for value in parameters.values())
        or parameters["trainable"] > parameters["total"]
    ):
        raise ValueError("TimesFM LoRA continuation parameter counts are invalid")
    publication = manifest.get("publication")
    if not json_values_equal_exact(publication, {
        "saved_after_best_adapter_restoration": True,
        "intermediate_checkpoints": 0,
        "accuracy_evaluation_runs": 0,
        "validation_runs": actual_epochs + 1,
    }):
        raise ValueError("TimesFM LoRA continuation was not one final-only publication")


def _validate_lora_manifest(path: pathlib.Path) -> dict[str, object]:
    manifest = _strict_record(path / LORA_MANIFEST, "TimesFM LoRA manifest")
    schema = manifest.get("schema")
    if schema == LORA_ARTIFACT_SCHEMA:
        _validate_v1_lora_manifest(manifest)
    elif schema == LORA_CONTINUATION_ARTIFACT_SCHEMA:
        _validate_continuation_lora_manifest(manifest)
        _adapter_model_file_sha256(path)
    else:
        raise ValueError("TimesFM LoRA manifest has an unknown schema")
    return manifest


def validate_lora_adapter(
    adapter_dir: str | pathlib.Path,
    adapter_sha256: str,
    *,
    config: LoRATrainingConfig,
    expected_training_rows_sha256: str,
    expected_training_rows_manifest_sha256: str,
    expected_panel_content_sha256: str,
    expected_training_cutoff: str,
    forecast_origin: str,
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    _recorded_runtime_lock: RuntimeLockIdentity | None = None,
) -> LoRAAdapterArtifact:
    """Authenticate one sealed final adapter before a runner authorizes inference.

    The digest covers every relative filename and file byte in the adapter directory, including
    the governed manifest.  The explicit expectations additionally bind that authenticated tree
    to its training rows, training configuration, cutoff, and forecast origin.  This function is
    read-only and does not import torch, Transformers, or PEFT.
    """
    _validate_training_config(config)
    expected_adapter_digest = _require_sha256(adapter_sha256, "adapter_sha256")
    expected_rows_digest = _require_sha256(
        expected_training_rows_sha256,
        "expected_training_rows_sha256",
    )
    expected_rows_manifest_digest = _require_sha256(
        expected_training_rows_manifest_sha256,
        "expected_training_rows_manifest_sha256",
    )
    expected_panel_digest = _require_sha256(
        expected_panel_content_sha256,
        "expected_panel_content_sha256",
    )
    if _recorded_runtime_lock is None:
        runtime_lock = validate_runtime_lock(
            runtime_lock_path,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
        )
    else:
        runtime_lock = _validate_snapshotted_runtime_lock(
            runtime_lock_path,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_identity=_recorded_runtime_lock,
        )
    path = pathlib.Path(adapter_dir)
    if not path.is_absolute():
        raise ValueError("TimesFM LoRA adapter_dir must be absolute")
    before = _tree_sha256(path)
    if before != expected_adapter_digest:
        raise ValueError(
            "TimesFM LoRA adapter hash mismatch: "
            f"expected {expected_adapter_digest}, got {before}"
        )
    manifest = _validate_lora_manifest(path)
    if manifest.get("schema") != LORA_ARTIFACT_SCHEMA:
        raise ValueError("v1 LoRA validation cannot authenticate a continuation artifact")
    if not json_values_equal_exact(manifest.get("training"), _config_record(config)):
        raise ValueError("TimesFM LoRA adapter was trained with a different config")
    runtime = manifest.get("runtime")
    if type(runtime) is not dict or not json_values_equal_exact(
        runtime.get("runtime_lock"), _runtime_lock_record(runtime_lock)
    ):
        raise ValueError("TimesFM LoRA adapter binds a different worker runtime lock")
    training_rows = manifest.get("training_rows")
    if type(training_rows) is not dict or set(training_rows) != {
        "sha256",
        "manifest_sha256",
        "n_rows",
        "cutoff",
        "panel_content_sha256",
    }:
        raise ValueError("TimesFM LoRA manifest has an invalid training_rows identity")
    if training_rows.get("sha256") != expected_rows_digest:
        raise ValueError("TimesFM LoRA adapter binds different training rows")
    if training_rows.get("manifest_sha256") != expected_rows_manifest_digest:
        raise ValueError("TimesFM LoRA adapter binds a different training-rows manifest")
    if training_rows.get("panel_content_sha256") != expected_panel_digest:
        raise ValueError("TimesFM LoRA adapter binds a different source panel")
    n_rows = training_rows.get("n_rows")
    if type(n_rows) is not int or n_rows < 1:
        raise ValueError("TimesFM LoRA manifest has no positive training-row count")
    cutoff = training_rows.get("cutoff")
    try:
        cutoff_day = dt.date.fromisoformat(cutoff) if type(cutoff) is str else None
        expected_cutoff_day = dt.date.fromisoformat(expected_training_cutoff)
        origin_day = dt.date.fromisoformat(forecast_origin)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TimesFM LoRA cutoff and forecast_origin must be canonical ISO dates"
        ) from exc
    if (
        cutoff_day is None
        or cutoff_day.isoformat() != cutoff
        or expected_cutoff_day.isoformat() != expected_training_cutoff
        or origin_day.isoformat() != forecast_origin
    ):
        raise ValueError("TimesFM LoRA cutoff and forecast_origin must be canonical ISO dates")
    if cutoff != expected_training_cutoff:
        raise ValueError("TimesFM LoRA adapter binds a different training cutoff")
    if cutoff_day > origin_day:
        raise ValueError(
            "TimesFM LoRA adapter training cutoff is after its forecast origin; a later adapter "
            "may not backcast an earlier trajectory"
        )
    after = _tree_sha256(path)
    if after != before:
        raise RuntimeError("TimesFM LoRA adapter changed while it was validated")
    return LoRAAdapterArtifact(
        path=str(path),
        sha256=before,
        manifest_path=str(path / LORA_MANIFEST),
        n_training_rows=n_rows,
        training_rows_sha256=expected_rows_digest,
        panel_content_sha256=expected_panel_digest,
    )


@dataclass
class _InferenceInputSnapshots:
    """Owned private inputs retained until the lazy provider load completes."""

    temporary: tempfile.TemporaryDirectory[str]
    root: pathlib.Path
    checkpoint_dir: pathlib.Path
    adapter_dir: pathlib.Path | None
    runtime_lock_path: pathlib.Path
    recorded_runtime_lock: RuntimeLockIdentity
    recorded_adapter_dir: pathlib.Path | None

    def cleanup(self) -> None:
        self.temporary.cleanup()


def _materialize_inference_input_snapshots(
    *,
    checkpoint_dir: pathlib.Path,
    model_revision: str,
    adapter_dir: pathlib.Path | None,
    adapter_sha256: str | None,
    runtime_lock_path: pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    recorded_runtime_lock: RuntimeLockIdentity | None,
    recorded_adapter_dir: pathlib.Path | None,
) -> _InferenceInputSnapshots:
    """Authenticate/copy/re-authenticate every file the lazy provider may consume."""
    before_base = _validate_base_checkpoint(checkpoint_dir, model_revision)
    if recorded_runtime_lock is None:
        before_runtime = validate_runtime_lock(
            runtime_lock_path,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
        )
    else:
        before_runtime = _validate_snapshotted_runtime_lock(
            runtime_lock_path,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_identity=recorded_runtime_lock,
        )
    before_adapter: str | None = None
    if adapter_dir is not None:
        before_adapter = _tree_sha256(adapter_dir)
        if before_adapter != adapter_sha256:
            raise ValueError(
                "TimesFM LoRA adapter hash mismatch before snapshot: "
                f"expected {adapter_sha256}, got {before_adapter}"
            )

    temporary: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="autoforecast-timesfm-inference."
    )
    root = pathlib.Path(temporary.name).resolve()
    private_base = root / "base"
    private_runtime = root / "runtime-lock.json"
    private_adapter = root / "adapter" if adapter_dir is not None else None
    try:
        _copy_private_tree(
            checkpoint_dir,
            private_base,
            label="TimesFM base checkpoint",
        )
        _copy_private_file(
            runtime_lock_path,
            private_runtime,
            label="TimesFM runtime lock",
        )
        if adapter_dir is not None and private_adapter is not None:
            _copy_private_tree(
                adapter_dir,
                private_adapter,
                label="TimesFM LoRA adapter",
            )

        after_base = _validate_base_checkpoint(checkpoint_dir, model_revision)
        if recorded_runtime_lock is None:
            after_runtime = validate_runtime_lock(
                runtime_lock_path,
                runtime_lock_file_sha256,
                runtime_lock_record_path=runtime_lock_record_path,
                runtime_lock_sha256=runtime_lock_sha256,
            )
        else:
            after_runtime = _validate_snapshotted_runtime_lock(
                runtime_lock_path,
                runtime_lock_file_sha256,
                runtime_lock_record_path=runtime_lock_record_path,
                runtime_lock_sha256=runtime_lock_sha256,
                recorded_identity=recorded_runtime_lock,
            )
        after_adapter = _tree_sha256(adapter_dir) if adapter_dir is not None else None
        if (
            after_base != before_base
            or after_runtime != before_runtime
            or after_adapter != before_adapter
        ):
            raise RuntimeError("TimesFM inference inputs changed while they were snapshotted")

        if _validate_base_checkpoint(private_base, model_revision) != before_base:
            raise RuntimeError("private TimesFM base snapshot differs from its source")
        _validate_snapshotted_runtime_lock(
            private_runtime,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_identity=before_runtime,
        )
        if private_adapter is not None and _tree_sha256(private_adapter) != before_adapter:
            raise RuntimeError("private TimesFM adapter snapshot differs from its source")
        return _InferenceInputSnapshots(
            temporary=temporary,
            root=root,
            checkpoint_dir=private_base,
            adapter_dir=private_adapter,
            runtime_lock_path=private_runtime,
            recorded_runtime_lock=before_runtime,
            recorded_adapter_dir=recorded_adapter_dir,
        )
    except BaseException:
        temporary.cleanup()
        raise


class TimesFMTransformersBackend:
    """Shared frozen/LoRA backend for the existing TimesFM Spark/output adapter."""

    governed_provider = True
    inference_spec = "timesfm_2p5_transformers_full0_expected_mean_predictions_median_v1"

    def __init__(
        self,
        *,
        checkpoint_dir: str | pathlib.Path,
        model_revision: str,
        context_length: int,
        device: str,
        adapter_dir: str | pathlib.Path | None = None,
        adapter_sha256: str | None = None,
        forecast_origin: str | None = None,
        runtime_lock_path: str | pathlib.Path,
        runtime_lock_file_sha256: str,
        runtime_lock_record_path: Sequence[str],
        runtime_lock_sha256: str,
        recorded_runtime_lock: RuntimeLockIdentity | Mapping[str, object] | None = None,
        recorded_adapter_dir: str | pathlib.Path | None = None,
    ) -> None:
        base = pathlib.Path(checkpoint_dir)
        adapter = pathlib.Path(adapter_dir) if adapter_dir is not None else None
        recorded_runtime = _coerce_runtime_lock_identity(recorded_runtime_lock)
        recorded_adapter = (
            pathlib.Path(recorded_adapter_dir)
            if recorded_adapter_dir is not None
            else adapter
        )
        _validate_provider_options(
            base,
            model_revision,
            context_length,
            device,
            adapter,
            adapter_sha256,
            forecast_origin,
        )
        if (adapter is None) != (recorded_adapter is None):
            raise ValueError("recorded_adapter_dir requires one loaded LoRA adapter")
        if recorded_adapter is not None and not recorded_adapter.is_absolute():
            raise ValueError("recorded_adapter_dir must be absolute")
        snapshots = _materialize_inference_input_snapshots(
            checkpoint_dir=base,
            model_revision=model_revision,
            adapter_dir=adapter,
            adapter_sha256=adapter_sha256,
            runtime_lock_path=pathlib.Path(runtime_lock_path),
            runtime_lock_file_sha256=runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_runtime_lock=recorded_runtime,
            recorded_adapter_dir=recorded_adapter,
        )
        runtime_lock = snapshots.recorded_runtime_lock
        self._input_snapshots: _InferenceInputSnapshots | None = snapshots
        self._checkpoint_dir = snapshots.checkpoint_dir
        self._model_revision = model_revision
        self._context_length = context_length
        self._device = device
        self._adapter_dir = snapshots.adapter_dir
        self._adapter_sha256 = adapter_sha256
        self._forecast_origin = forecast_origin
        self._runtime_lock_path = snapshots.runtime_lock_path
        self._runtime_lock_file_sha256 = runtime_lock_file_sha256
        self._runtime_lock_record_path = tuple(runtime_lock_record_path)
        self._runtime_lock_sha256 = runtime_lock_sha256
        self._runtime_lock = runtime_lock
        self._recorded_adapter_dir = snapshots.recorded_adapter_dir
        self._model: object | None = None
        self._torch: object | None = None

    def validate_origin(self, origin: str) -> None:
        """Bind an adapted provider's declared origin to the shared Spark boundary origin."""
        if self._forecast_origin is not None and origin != self._forecast_origin:
            raise ValueError(
                "TimesFM LoRA provider forecast_origin differs from the adapter boundary origin; "
                "a later adapter may not backcast an earlier trajectory"
            )

    def _load_from_snapshots(self) -> tuple[object, object]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        if self._input_snapshots is None:
            raise RuntimeError("TimesFM private inference inputs are no longer available")
        snapshot_identity = _source_tree_identity(
            self._input_snapshots.root,
            label="private TimesFM inference inputs",
        )
        before_runtime = _validate_snapshotted_runtime_lock(
            self._runtime_lock_path,
            self._runtime_lock_file_sha256,
            runtime_lock_record_path=self._runtime_lock_record_path,
            runtime_lock_sha256=self._runtime_lock_sha256,
            recorded_identity=self._runtime_lock,
        )
        if before_runtime != self._runtime_lock:
            raise RuntimeError("TimesFM runtime lock identity changed before model load")
        before_base = _validate_base_checkpoint(self._checkpoint_dir, self._model_revision)
        before_adapter: str | None = None
        if self._adapter_dir is not None:
            before_adapter = _tree_sha256(self._adapter_dir)
            if before_adapter != self._adapter_sha256:
                raise ValueError(
                    "TimesFM LoRA adapter hash mismatch before load: "
                    f"expected {self._adapter_sha256}, got {before_adapter}"
                )
            adapter_manifest = _validate_lora_manifest(self._adapter_dir)
            adapter_runtime = adapter_manifest.get("runtime")
            if (
                type(adapter_runtime) is not dict
                or adapter_runtime.get("runtime_lock")
                != _runtime_lock_record(before_runtime)
            ):
                raise ValueError(
                    "TimesFM LoRA adapter was produced under a different worker runtime lock"
                )
            training_rows = adapter_manifest["training_rows"]
            assert isinstance(training_rows, dict)
            training_cutoff = training_rows.get("cutoff")
            try:
                training_day = (
                    dt.date.fromisoformat(training_cutoff)
                    if type(training_cutoff) is str
                    else None
                )
                forecast_day = (
                    dt.date.fromisoformat(self._forecast_origin)
                    if type(self._forecast_origin) is str
                    else None
                )
            except ValueError as exc:
                raise ValueError("TimesFM LoRA manifest has an invalid training cutoff") from exc
            if (
                training_day is None
                or training_day.isoformat() != training_cutoff
                or forecast_day is None
                or training_day > forecast_day
            ):
                raise ValueError(
                    "TimesFM LoRA adapter training cutoff is after its forecast origin; a later "
                    "adapter may not backcast an earlier trajectory"
                )

        torch = importlib.import_module("torch")
        if self._device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "TimesFM Transformers requested MPS, but torch reports it unavailable"
            )
        transformers = importlib.import_module("transformers")
        model_class = getattr(transformers, "TimesFm2_5ModelForPrediction", None)
        if model_class is None:
            raise ImportError(
                "installed transformers does not expose TimesFm2_5ModelForPrediction"
            )
        model = model_class.from_pretrained(
            str(self._checkpoint_dir),
            local_files_only=True,
            dtype=torch.float32,
        )
        if self._adapter_dir is not None:
            peft = importlib.import_module("peft")
            model = peft.PeftModel.from_pretrained(
                model,
                str(self._adapter_dir),
                is_trainable=False,
                local_files_only=True,
            )
        model = model.to(device=self._device, dtype=torch.float32)
        model.eval()

        after_base = _validate_base_checkpoint(self._checkpoint_dir, self._model_revision)
        if before_base != after_base:
            raise RuntimeError("TimesFM base checkpoint changed while the provider loaded it")
        if self._adapter_dir is not None:
            after_adapter = _tree_sha256(self._adapter_dir)
            if after_adapter != before_adapter:
                raise RuntimeError("TimesFM LoRA adapter changed while the provider loaded it")
        after_runtime = _validate_snapshotted_runtime_lock(
            self._runtime_lock_path,
            self._runtime_lock_file_sha256,
            runtime_lock_record_path=self._runtime_lock_record_path,
            runtime_lock_sha256=self._runtime_lock_sha256,
            recorded_identity=self._runtime_lock,
        )
        if after_runtime != before_runtime:
            raise RuntimeError("TimesFM runtime lock identity changed while the provider loaded")
        if _source_tree_identity(
            self._input_snapshots.root,
            label="private TimesFM inference inputs",
        ) != snapshot_identity:
            raise RuntimeError("private TimesFM inference inputs changed during provider load")
        self._model, self._torch = model, torch
        return model, torch

    def _load(self) -> tuple[object, object]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            return self._load_from_snapshots()
        finally:
            if self._input_snapshots is not None:
                self._input_snapshots.cleanup()
                self._input_snapshots = None

    def close(self) -> None:
        """Remove private lazy-load inputs when an unused provider is discarded."""
        if self._input_snapshots is not None:
            self._input_snapshots.cleanup()
            self._input_snapshots = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown is best effort
            pass

    def forecast(
        self,
        inputs: Sequence[Sequence[float]],
        horizon: int,
    ) -> tuple[object, object]:
        """Return ``(median, full_predictions)`` for the shared TimesFM selector."""
        if horizon != GOVERNED_HORIZON:
            raise ValueError(
                f"governed TimesFM Transformers inference requires horizon={GOVERNED_HORIZON}"
            )
        if not inputs:
            raise ValueError("TimesFM Transformers forecast inputs cannot be empty")
        model, torch = self._load()
        tensors = [
            torch.tensor(values, dtype=torch.float32, device=self._device)
            for values in inputs
        ]
        with torch.no_grad():
            outputs = model(
                past_values=tensors,
                forecast_context_len=self._context_length,
                truncate_negative=False,
                force_flip_invariance=True,
            )
        median = outputs.mean_predictions[:, :horizon]
        full = outputs.full_predictions[:, :horizon, :]
        median_array = median.detach().to(device="cpu", dtype=torch.float32).numpy()
        full_array = full.detach().to(device="cpu", dtype=torch.float32).numpy()
        if median_array.shape != (len(inputs), horizon):
            raise ValueError(
                "TimesFM Transformers mean_predictions has an unexpected shape: "
                f"{median_array.shape}"
            )
        if full_array.shape != (len(inputs), horizon, 10):
            raise ValueError(
                "TimesFM Transformers full_predictions has an unexpected shape: "
                f"{full_array.shape}"
            )
        if not np.isfinite(median_array).all() or not np.isfinite(full_array).all():
            raise ValueError("TimesFM Transformers returned non-finite forecasts")
        if not np.allclose(median_array, full_array[..., 5], rtol=1e-5, atol=1e-5):
            raise ValueError(
                "TimesFM Transformers mean_predictions no longer matches median channel 5"
            )
        # The existing adapter deliberately selects full_array[..., 0] for expected demand.
        return median_array, full_array


def _canonical_training_source(
    panel: Panel,
    spark: SparkSession,
    cutoff: str,
    *,
    source_partitions: int,
) -> DataFrame:
    if (
        type(source_partitions) is not int
        or source_partitions < 1
    ):
        raise ValueError("TimesFM LoRA source_partitions must be a positive integer")
    if panel.status_prob_cols is not None:
        raise ValueError("TimesFM LoRA training does not support soft observation states")
    if panel.target_estimand != "sales_given_availability":
        raise ValueError("TimesFM LoRA trains on raw observed sales only")
    raw = spark.read.parquet(panel.path)
    dated = raw.withColumn("_timesfm_date", F.expr(f"try_cast(`{panel.time_col}` as date)"))
    if dated.filter(F.col("_timesfm_date").isNull()).limit(1).count():
        raise ValueError("TimesFM LoRA input contains a null or invalid calendar date")
    selected = [
        F.col(panel.series_col).cast("string").alias("series_id"),
        F.col("_timesfm_date").alias("date"),
        F.expr(f"try_cast(`{panel.target_col}` as double)").alias("target"),
    ]
    if panel.has_exposure:
        selected.append(F.col(panel.exposure_col).cast("string").alias("state"))
    else:
        selected.append(F.lit("exact").alias("state"))
    return (
        dated.filter(F.col("_timesfm_date") <= F.lit(cutoff))
        .select(*selected)
        .repartitionByRange(source_partitions, "series_id", "date")
        .sortWithinPartitions("series_id", "date")
    )


def _series_rows(frame: DataFrame, staging_dir: pathlib.Path):
    """Stream range-sorted rows through private Parquet, not Spark task-result transport."""
    if staging_dir.exists():
        raise ValueError(f"TimesFM LoRA source staging path already exists: {staging_dir}")
    frame.write.mode("errorifexists").parquet(str(staging_dir))
    parts = sorted(staging_dir.glob("part-*.parquet"))
    if not parts or any(not part.is_file() or part.is_symlink() for part in parts):
        raise RuntimeError("TimesFM LoRA source staging produced no regular Parquet parts")

    current: str | None = None
    rows: list[object] = []
    failed = False
    try:
        for part in parts:
            parquet = pq.ParquetFile(part)
            for batch in parquet.iter_batches(
                batch_size=8192,
                columns=["series_id", "date", "target", "state"],
            ):
                for row in batch.to_pylist():
                    series_id = row["series_id"]
                    if current is not None and series_id != current:
                        yield current, rows
                        rows = []
                    current = series_id
                    rows.append(row)
        if current is not None:
            yield current, rows
    except BaseException:
        failed = True
        raise
    finally:
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            if not failed:
                raise


def _float32_target(
    value: object,
    *,
    series_id: str,
    day: dt.date,
    negative_policy: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"TimesFM LoRA target for {series_id} on {day} is boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"TimesFM LoRA target for {series_id} on {day} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"TimesFM LoRA target for {series_id} on {day} is non-finite")
    if number < 0:
        if negative_policy == "floor":
            number = 0.0
        else:
            raise ValueError(f"TimesFM LoRA target for {series_id} on {day} is negative")
    converted = float(np.float32(number))
    if not math.isfinite(converted):
        raise ValueError(f"TimesFM LoRA target for {series_id} on {day} overflows float32")
    return converted


def _positive_exact_history(
    series_id: str,
    rows: Sequence[object],
    negative_policy: str,
) -> tuple[list[dt.date], list[float]] | None:
    """Return the exact daily history from first positive sale through the cutoff row."""
    if type(series_id) is not str or not series_id.strip():
        raise ValueError("TimesFM LoRA input contains a null or blank series identifier")
    allowed_states = {"exact", "lower_bound", "unobserved"}
    seen: set[dt.date] = set()
    dates: list[dt.date] = []
    values: list[float] = []
    started = False
    for row in rows:
        day, state = row["date"], row["state"]
        if day in seen:
            raise ValueError(f"TimesFM LoRA input has duplicate key {(series_id, day)!r}")
        seen.add(day)
        if state not in allowed_states:
            raise ValueError(
                f"TimesFM LoRA input has unknown observation state {state!r} for {series_id}"
            )
        number: float | None = None
        if state == "exact":
            number = _float32_target(
                row["target"],
                series_id=series_id,
                day=day,
                negative_policy=negative_policy,
            )
        if not started and number is not None and number > 0:
            started = True
        if started:
            if state != "exact" or number is None:
                return None
            if dates and day != dates[-1] + dt.timedelta(days=1):
                return None
            dates.append(day)
            values.append(number)
    return (dates, values) if started else None


def _window_start_day(seed: int, epoch: int, series_id: str, first: int, last: int) -> int:
    encoded = f"{seed}\0{epoch}\0{series_id}".encode()
    draw = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return first + draw % (last - first + 1)


def _training_rows_arrow_schema(config: LoRATrainingConfig) -> pa.Schema:
    return pa.schema(
        [
            pa.field("epoch", pa.int32(), nullable=False),
            pa.field("series_id", pa.string(), nullable=False),
            pa.field("start_day", pa.int32(), nullable=False),
            pa.field("context_end_day", pa.int32(), nullable=False),
            pa.field("target_end_day", pa.int32(), nullable=False),
            pa.field(
                "context",
                pa.list_(pa.float32(), config.context_length),
                nullable=False,
            ),
            pa.field(
                "target",
                pa.list_(pa.float32(), config.horizon_length),
                nullable=False,
            ),
        ]
    )


def _publish_directory(temporary: pathlib.Path, target: pathlib.Path) -> None:
    if target.exists():
        target.rmdir()
    os.replace(temporary, target)


def _validate_empty_target(target: pathlib.Path, label: str) -> None:
    if not target.is_absolute():
        raise ValueError(f"{label} output path must be absolute: {target}")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"{label} output path must be absent or an empty directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)


def prepare_lora_training_rows(
    panel: Panel,
    cutoff: str,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    config: LoRATrainingConfig,
    source_data_path: str | pathlib.Path,
    source_data_sha256: str,
    panel_content_sha256: str,
) -> TrainingRowsArtifact:
    """Materialise one deterministic fully-exact window per series and epoch.

    Series are UTF-8 sorted.  The first positive-sale day starts eligibility.  For zero-based
    epoch ``e``, the first eight SHA-256 bytes of ``seed\0e\0series_id`` select a start between
    that day and the last start ending at ``cutoff``.  The returned canonical JSONL hash exists
    before model construction and is the training-data identity embedded in the final adapter.
    """
    _validate_training_config(config)
    try:
        cutoff_day = dt.date.fromisoformat(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TimesFM LoRA cutoff must be ISO YYYY-MM-DD, got {cutoff!r}") from exc
    if cutoff_day.isoformat() != cutoff:
        raise ValueError(f"TimesFM LoRA cutoff must be canonical ISO date, got {cutoff!r}")
    source_digest = _require_sha256(source_data_sha256, "source_data_sha256")
    source_path = pathlib.Path(source_data_path)
    if not source_path.is_absolute() or not source_path.is_file():
        raise ValueError("source_data_path must be an absolute existing raw-data file")
    actual_source_digest = _file_sha256(source_path)
    if actual_source_digest != source_digest:
        raise ValueError(
            f"raw source data hash mismatch: expected {source_digest}, got {actual_source_digest}"
        )
    expected_panel_digest = _require_sha256(
        panel_content_sha256,
        "panel_content_sha256",
    )
    panel_path = pathlib.Path(panel.path)
    if not panel_path.is_absolute() or not panel_path.exists():
        raise ValueError("panel.path must be an absolute existing Parquet artifact")
    actual_panel_digest = directory_digest(panel_path, suffixes=(".parquet",))
    if actual_panel_digest != expected_panel_digest:
        raise ValueError(
            "TimesFM LoRA source panel hash mismatch: "
            f"expected {expected_panel_digest}, got {actual_panel_digest}"
        )
    if (
        source_digest == M5_SALES_TRAIN_EVALUATION_SHA256
        and config == PROSPECTIVE_M5_LORA_CONFIG
        and expected_panel_digest != M5_PANEL_CONTENT_SHA256
    ):
        raise ValueError(
            "real M5 TimesFM LoRA source panel differs from the registered canonical panel"
        )

    target = pathlib.Path(out_dir)
    _validate_empty_target(target, "TimesFM LoRA training rows")
    temporary = pathlib.Path(
        tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.timesfm-rows.")
    )
    rows_path = temporary / TRAINING_ROWS_FILE
    index_path = temporary / TRAINING_ROWS_INDEX
    schema = _training_rows_arrow_schema(config)
    epoch_paths = [temporary / f".epoch-{epoch:03d}.parquet" for epoch in range(config.epochs)]
    epoch_writers = [
        pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)
        for path in epoch_paths
    ]
    buffers: list[list[dict[str, object]]] = [[] for _ in range(config.epochs)]
    n_series = 0
    n_rows = 0
    window_length = config.context_length + config.horizon_length

    def flush(epoch: int) -> None:
        buffer = buffers[epoch]
        if not buffer:
            return
        epoch_writers[epoch].write_table(pa.Table.from_pylist(buffer, schema=schema))
        buffer.clear()

    try:
        configured_partitions = int(spark.conf.get("spark.sql.shuffle.partitions"))
        source_partitions = (
            M5_LORA_SOURCE_PARTITIONS
            if (
                source_digest == M5_SALES_TRAIN_EVALUATION_SHA256
                and expected_panel_digest == M5_PANEL_CONTENT_SHA256
            )
            else configured_partitions
        )
        frame = _canonical_training_source(
            panel,
            spark,
            cutoff,
            source_partitions=source_partitions,
        )
        day_one_row = frame.agg(F.min("date").alias("day_one")).first()
        day_one = day_one_row["day_one"] if day_one_row is not None else None
        if day_one is None:
            raise ValueError("TimesFM LoRA input has no row at or before the cutoff")
        cutoff_ordinal = (cutoff_day - day_one).days + 1
        last_start_day = cutoff_ordinal - window_length + 1
        previous_series_bytes: bytes | None = None
        for series_id, series_rows in _series_rows(
            frame,
            temporary / ".canonical-source",
        ):
            encoded_series = series_id.encode("utf-8") if type(series_id) is str else b""
            if previous_series_bytes is not None and encoded_series <= previous_series_bytes:
                raise ValueError("TimesFM LoRA source is not strictly UTF-8 series sorted")
            previous_series_bytes = encoded_series
            history = _positive_exact_history(
                series_id,
                series_rows,
                panel.negative_target_policy,
            )
            if history is None:
                continue
            dates, values = history
            if dates[-1] != cutoff_day:
                continue
            first_day = (dates[0] - day_one).days + 1
            if first_day + window_length - 1 > cutoff_ordinal:
                continue
            n_series += 1
            for epoch in range(config.epochs):
                start_day = _window_start_day(
                    config.seed,
                    epoch,
                    series_id,
                    first_day,
                    last_start_day,
                )
                offset = start_day - first_day
                selected = values[offset : offset + window_length]
                if len(selected) != window_length:
                    raise AssertionError("TimesFM selected window exceeds its exact history")
                buffers[epoch].append(
                    {
                        "epoch": epoch,
                        "series_id": series_id,
                        "start_day": start_day,
                        "context_end_day": start_day + config.context_length - 1,
                        "target_end_day": start_day + window_length - 1,
                        "context": selected[: config.context_length],
                        "target": selected[config.context_length :],
                    }
                )
                n_rows += 1
                if len(buffers[epoch]) == 512:
                    flush(epoch)
        for epoch in range(config.epochs):
            flush(epoch)
            epoch_writers[epoch].close()
        if n_series == 0:
            raise ValueError(
                "no series has one fully exact consecutive TimesFM context+target window"
            )
        if n_rows != n_series * config.epochs:
            raise AssertionError("TimesFM stratified sampling did not emit one row per epoch")

        rows_writer = pq.ParquetWriter(
            rows_path,
            schema,
            compression="zstd",
            use_dictionary=True,
        )
        index_digest = hashlib.sha256()
        first_descriptor: dict[str, object] | None = None
        with index_path.open("wb") as index_stream:
            for epoch_path in epoch_paths:
                parquet = pq.ParquetFile(epoch_path)
                for batch in parquet.iter_batches(batch_size=512):
                    table = pa.Table.from_batches([batch], schema=schema)
                    rows_writer.write_table(table)
                    for row in table.select(
                        [
                            "series_id",
                            "epoch",
                            "start_day",
                            "context_end_day",
                            "target_end_day",
                        ]
                    ).to_pylist():
                        encoded = (
                            json.dumps(
                                row,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        ).encode("utf-8")
                        if first_descriptor is None:
                            first_descriptor = row
                        index_stream.write(encoded)
                        index_digest.update(encoded)
        rows_writer.close()
        for epoch_path in epoch_paths:
            epoch_path.unlink()
        rows_digest = index_digest.hexdigest()
        payload_digest = _file_sha256(rows_path)
        panel_digest_after_selection = directory_digest(
            panel_path,
            suffixes=(".parquet",),
        )
        if panel_digest_after_selection != actual_panel_digest:
            raise RuntimeError("TimesFM LoRA source panel changed while rows were selected")
        if (
            source_digest == M5_SALES_TRAIN_EVALUATION_SHA256
            and config == PROSPECTIVE_M5_LORA_CONFIG
        ):
            identity = next(
                (
                    candidate
                    for candidate in M5_LORA_CUTOFF_IDENTITIES
                    if candidate.cutoff == cutoff
                ),
                None,
            )
            if identity is None:
                raise ValueError(
                    "real M5 TimesFM LoRA preflight requires one of the sealed forward cutoffs"
                )
        else:
            identity = None
        if identity is not None:
            expected_first = {
                "series_id": "FOODS_1_001_CA_1_evaluation",
                "epoch": 0,
                "start_day": identity.first_start_day,
                "context_end_day": identity.first_context_end_day,
                "target_end_day": identity.first_target_end_day,
            }
            observed = {
                "day_one": day_one.isoformat(),
                "last_start_day": last_start_day,
                "n_series": n_series,
                "n_rows": n_rows,
                "rows_sha256": rows_digest,
                "first_row": first_descriptor,
            }
            expected = {
                "day_one": "2011-01-29",
                "last_start_day": identity.last_start_day,
                "n_series": identity.eligible_series,
                "n_rows": identity.windows,
                "rows_sha256": identity.training_rows_sha256,
                "first_row": expected_first,
            }
            if observed != expected:
                raise ValueError(
                    "real M5 TimesFM LoRA training-row preflight differs from the sealed "
                    f"identity: observed={observed}, expected={expected}"
                )
        manifest = {
            "schema": TRAINING_ROWS_SCHEMA,
            "cutoff": cutoff,
            "day_one": day_one.isoformat(),
            "source": {
                "name": source_path.name,
                "sha256": source_digest,
            },
            "panel": {
                "path": str(panel_path),
                "content_sha256": expected_panel_digest,
                "series_col": panel.series_col,
                "time_col": panel.time_col,
                "target_col": panel.target_col,
                "exposure_col": panel.exposure_col,
                "negative_target_policy": panel.negative_target_policy,
                "target_estimand": panel.target_estimand,
            },
            "selection": {
                "eligible_state": "exact_only",
                "calendar": "consecutive_daily",
                "first_observed_day": "first_positive_sale",
                "samples_per_eligible_series_per_epoch": 1,
                "start_selector": (
                    "first + int.from_bytes(sha256(seed\\0epoch\\0series_id)[:8],big) "
                    "% (last_start-first+1)"
                ),
                "row_order": "epoch_ascending_then_series_id_utf8_ascending",
                "epoch_indexing": "zero_based",
                "post_cutoff_rows": "forbidden",
                "external_normalization": "none",
            },
            "training": _config_record(config),
            "rows": {
                "file": TRAINING_ROWS_FILE,
                "payload_sha256": payload_digest,
                "index_file": TRAINING_ROWS_INDEX,
                "sha256": rows_digest,
                "n_series": n_series,
                "n_rows": n_rows,
                "last_start_day": last_start_day,
                "first_row": first_descriptor,
            },
            "runtime": {"pyarrow": pa.__version__},
        }
        manifest_path = temporary / TRAINING_ROWS_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        manifest_digest = _file_sha256(manifest_path)
        _publish_directory(temporary, target)
    except BaseException:
        for writer in epoch_writers:
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return TrainingRowsArtifact(
        path=str(target),
        rows_sha256=rows_digest,
        manifest_path=str(target / TRAINING_ROWS_MANIFEST),
        manifest_sha256=manifest_digest,
        n_series=n_series,
        n_rows=n_rows,
        cutoff=cutoff,
        panel_content_sha256=expected_panel_digest,
    )


def _validate_training_rows(
    path: pathlib.Path,
    expected_rows_sha256: str,
    expected_manifest_sha256: str,
    config: LoRATrainingConfig,
) -> tuple[dict[str, object], pathlib.Path, pa.Table]:
    expected_digest = _require_sha256(expected_rows_sha256, "training_rows_sha256")
    expected_manifest_digest = _require_sha256(
        expected_manifest_sha256,
        "training_rows_manifest_sha256",
    )
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("training_rows_dir must be an absolute existing artifact directory")
    manifest_path = path / TRAINING_ROWS_MANIFEST
    actual_manifest_digest = _file_sha256(manifest_path)
    if actual_manifest_digest != expected_manifest_digest:
        raise ValueError(
            "TimesFM training-rows manifest changed: "
            f"expected {expected_manifest_digest}, got {actual_manifest_digest}"
        )
    manifest = _strict_record(manifest_path, "TimesFM training-rows manifest")
    expected_fields = {
        "schema",
        "cutoff",
        "day_one",
        "source",
        "panel",
        "selection",
        "training",
        "rows",
        "runtime",
    }
    if set(manifest) != expected_fields or manifest.get("schema") != TRAINING_ROWS_SCHEMA:
        raise ValueError("TimesFM training-rows manifest has an unknown schema or field set")
    if not json_values_equal_exact(manifest.get("training"), _config_record(config)):
        raise ValueError("TimesFM training rows were prepared with a different LoRA config")
    panel = manifest.get("panel")
    if type(panel) is not dict or set(panel) != {
        "path",
        "content_sha256",
        "series_col",
        "time_col",
        "target_col",
        "exposure_col",
        "negative_target_policy",
        "target_estimand",
    }:
        raise ValueError("TimesFM training-rows manifest has an invalid source panel identity")
    _require_sha256(panel.get("content_sha256"), "panel content_sha256")
    rows = manifest.get("rows")
    if (
        type(rows) is not dict
        or rows.get("file") != TRAINING_ROWS_FILE
        or rows.get("index_file") != TRAINING_ROWS_INDEX
    ):
        raise ValueError("TimesFM training-rows manifest does not identify its Parquet payload")
    if rows.get("sha256") != expected_digest:
        raise ValueError("TimesFM training-rows manifest does not bind the expected rows hash")
    index_path = path / TRAINING_ROWS_INDEX
    actual_digest = _file_sha256(index_path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"TimesFM canonical training-row index changed: expected {expected_digest}, "
            f"got {actual_digest}"
        )
    rows_path = path / TRAINING_ROWS_FILE
    payload_digest = rows.get("payload_sha256")
    if _require_sha256(payload_digest, "training rows payload_sha256") != _file_sha256(rows_path):
        raise ValueError("TimesFM training-row Parquet payload changed")
    table = pq.read_table(rows_path)
    if table.schema != _training_rows_arrow_schema(config):
        raise ValueError("TimesFM training rows do not have the exact registered Arrow schema")
    if table.num_rows != rows.get("n_rows"):
        raise ValueError("TimesFM training rows count differs from its manifest")
    n_series = rows.get("n_series")
    if type(n_series) is not int or n_series < 1:
        raise ValueError("TimesFM training rows manifest has no positive series count")
    if table.num_rows != n_series * config.epochs:
        raise ValueError("TimesFM training rows are not one window per series per epoch")
    day_one = manifest.get("day_one")
    cutoff = manifest.get("cutoff")
    try:
        day_one_date = dt.date.fromisoformat(day_one) if type(day_one) is str else None
        cutoff_day = dt.date.fromisoformat(cutoff) if type(cutoff) is str else None
    except ValueError as exc:
        raise ValueError("TimesFM training rows have an invalid day-one or cutoff") from exc
    if (
        day_one_date is None
        or cutoff_day is None
        or day_one_date.isoformat() != day_one
        or cutoff_day.isoformat() != cutoff
    ):
        raise ValueError("TimesFM training rows have a noncanonical day-one or cutoff")
    cutoff_ordinal = (cutoff_day - day_one_date).days + 1
    if pc.any(pc.greater(table["target_end_day"], pa.scalar(cutoff_ordinal))).as_py():
        raise ValueError("TimesFM training rows contain a post-cutoff target")
    return manifest, manifest_path, table


def validate_training_rows_artifact(
    training_rows_dir: str | pathlib.Path,
    training_rows_sha256: str,
    training_rows_manifest_sha256: str,
    *,
    config: LoRATrainingConfig,
    expected_cutoff: str,
    expected_panel_content_sha256: str,
    panel_validation_path: str | pathlib.Path | None = None,
) -> TrainingRowsArtifact:
    """Authenticate prepared rows before any model or training runtime is loaded."""
    _validate_training_config(config)
    rows_dir = pathlib.Path(training_rows_dir)
    manifest, manifest_path, table = _validate_training_rows(
        rows_dir,
        training_rows_sha256,
        training_rows_manifest_sha256,
        config,
    )
    cutoff = manifest.get("cutoff")
    try:
        expected_day = dt.date.fromisoformat(expected_cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_cutoff must be a canonical ISO date") from exc
    if expected_day.isoformat() != expected_cutoff:
        raise ValueError("expected_cutoff must be a canonical ISO date")
    if cutoff != expected_cutoff:
        raise ValueError("TimesFM training rows bind a different cutoff")
    expected_panel_digest = _require_sha256(
        expected_panel_content_sha256,
        "expected_panel_content_sha256",
    )
    panel = manifest["panel"]
    assert isinstance(panel, dict)
    if panel.get("content_sha256") != expected_panel_digest:
        raise ValueError("TimesFM training rows bind a different source panel")
    recorded_panel_path = pathlib.Path(str(panel.get("path")))
    if not recorded_panel_path.is_absolute():
        raise ValueError("TimesFM training rows bind a non-absolute source panel")
    validation_path = (
        pathlib.Path(panel_validation_path)
        if panel_validation_path is not None
        else recorded_panel_path
    )
    if not validation_path.is_absolute() or not validation_path.exists():
        raise ValueError("TimesFM panel_validation_path must be an absolute existing artifact")
    actual_panel_digest = directory_digest(validation_path, suffixes=(".parquet",))
    if actual_panel_digest != expected_panel_digest:
        raise ValueError(
            "TimesFM training-row source panel changed: "
            f"expected {expected_panel_digest}, got {actual_panel_digest}"
        )
    rows = manifest["rows"]
    assert isinstance(rows, dict)
    n_series = rows["n_series"]
    assert isinstance(n_series, int)
    return TrainingRowsArtifact(
        path=str(rows_dir),
        rows_sha256=training_rows_sha256,
        manifest_path=str(manifest_path),
        manifest_sha256=training_rows_manifest_sha256,
        n_series=n_series,
        n_rows=table.num_rows,
        cutoff=expected_cutoff,
        panel_content_sha256=expected_panel_digest,
    )


def _continuation_series_split(
    table: pa.Table,
    source_config: LoRATrainingConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, object]]:
    series_ids = sorted(set(table["series_id"].to_pylist()), key=lambda value: value.encode())
    if len(series_ids) < 2:
        raise ValueError("LoRA continuation requires at least two series for a 90/10 split")
    ranked = sorted(
        series_ids,
        key=lambda series_id: (
            hashlib.sha256(series_id.encode("utf-8")).digest(),
            series_id.encode("utf-8"),
        ),
    )
    n_validation = max(
        1,
        min(len(series_ids) - 1, math.floor(len(series_ids) * 0.1)),
    )
    validation = tuple(ranked[:n_validation])
    training = tuple(ranked[n_validation:])
    validation_members = set(validation)
    assignment_digest = hashlib.sha256()
    for series_id in ranked:
        assignment_digest.update(
            (
                json.dumps(
                    {
                        "role": "validation" if series_id in validation_members else "train",
                        "series_id": series_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    record = {
        "schema": "sha256_series_train_validation_split_v1",
        "algorithm": "ascending_sha256(series_id_utf8)_then_series_id_utf8",
        "validation_fraction": 0.1,
        "validation_rounding": "first_max(1,floor(0.1*n_series))_with_nonempty_training",
        "n_series": len(series_ids),
        "n_train_series": len(training),
        "n_validation_series": len(validation),
        "assignment_sha256": assignment_digest.hexdigest(),
        "series_disjoint": True,
        "validation_source_epochs": list(range(source_config.epochs)),
    }
    return training, validation, record


def _selected_training_batches(
    table: pa.Table,
    *,
    selected_series: Sequence[str],
    batch_size: int,
    source_epoch: int | None,
):
    ordered_series = tuple(selected_series)
    series_rank = {series_id: rank for rank, series_id in enumerate(ordered_series)}
    if len(series_rank) != len(ordered_series):
        raise ValueError("LoRA continuation selection contains a duplicate series")
    epochs = table["epoch"].to_pylist()
    series_ids = table["series_id"].to_pylist()
    indices = [
        index
        for index, series_id in enumerate(series_ids)
        if series_id in series_rank
        and (source_epoch is None or epochs[index] == source_epoch)
    ]
    indices.sort(key=lambda index: (epochs[index], series_rank[series_ids[index]]))
    expected_per_series = 1 if source_epoch is not None else len(set(epochs))
    if len(indices) != len(ordered_series) * expected_per_series:
        raise ValueError("LoRA continuation selection repeats or omits a series window")
    for start in range(0, len(indices), batch_size):
        selected = pa.array(indices[start : start + batch_size], type=pa.int64())
        batch = table.take(selected)
        contexts = batch["context"].to_pylist()
        targets = batch["target"].to_pylist()
        if not np.isfinite(np.asarray(contexts, dtype=np.float32)).all():
            raise ValueError("TimesFM LoRA continuation context contains a non-finite value")
        if not np.isfinite(np.asarray(targets, dtype=np.float32)).all():
            raise ValueError("TimesFM LoRA continuation target contains a non-finite value")
        yield contexts, targets


class _PeftTrainingRuntime:
    def __init__(
        self,
        checkpoint_dir: pathlib.Path,
        model_revision: str,
        config: LoRATrainingConfig,
        total_steps: int,
    ) -> None:
        before_base = _validate_base_checkpoint(checkpoint_dir, model_revision)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch = importlib.import_module("torch")
        torch.manual_seed(config.seed)
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("TimesFM LoRA requested MPS, but torch reports it unavailable")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        model_class = getattr(transformers, "TimesFm2_5ModelForPrediction", None)
        if model_class is None:
            raise ImportError(
                "installed transformers does not expose TimesFm2_5ModelForPrediction"
            )
        base = model_class.from_pretrained(
            str(checkpoint_dir),
            local_files_only=True,
            dtype=torch.float32,
        )
        after_base = _validate_base_checkpoint(checkpoint_dir, model_revision)
        if before_base != after_base:
            raise RuntimeError("TimesFM base checkpoint changed while LoRA loaded it")
        base = base.to(device=config.device, dtype=torch.float32)
        lora = peft.LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            target_modules="all-linear",
            lora_dropout=config.dropout,
            bias="none",
        )
        model = peft.get_peft_model(base, lora)
        unexpected = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if unexpected or not trainable:
            raise RuntimeError(
                "PEFT did not freeze the TimesFM base or create only LoRA parameters: "
                f"unexpected={unexpected[:5]}, trainable={len(trainable)}"
            )
        self._torch = torch
        self._model = model
        self._optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=total_steps,
        )
        self._config = config
        self._runtime_identity = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "device": config.device,
            "precision": "float32",
        }
        self._parameters = {
            "trainable": sum(parameter.numel() for parameter in trainable),
            "total": sum(parameter.numel() for parameter in model.parameters()),
        }

    def _train_batch(
        self,
        contexts: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float:
        torch = self._torch
        self._model.train()
        context = torch.tensor(contexts, dtype=torch.float32, device=self._config.device)
        target = torch.tensor(targets, dtype=torch.float32, device=self._config.device)
        self._optimizer.zero_grad(set_to_none=True)
        outputs = self._model(
            past_values=context,
            future_values=target,
            forecast_context_len=self._config.context_length,
            truncate_negative=False,
            force_flip_invariance=True,
        )
        loss = outputs.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("TimesFM LoRA produced a non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            max_norm=self._config.gradient_clip_norm,
        )
        self._optimizer.step()
        self._scheduler.step()
        return float(loss.detach().to(device="cpu", dtype=torch.float32).item())

    def _save_adapter(self, path: pathlib.Path) -> None:
        self._model.save_pretrained(path, safe_serialization=True)

    def _identity(self) -> Mapping[str, object]:
        return {
            "runtime": dict(self._runtime_identity),
            "parameters": dict(self._parameters),
        }


class _PeftContinuationRuntime:
    """PEFT runtime that loads one exact adapter and resets only optimizer state."""

    def __init__(
        self,
        checkpoint_dir: pathlib.Path,
        model_revision: str,
        source_adapter_dir: pathlib.Path,
        source_adapter_sha256: str,
        source_adapter_model_sha256: str,
        source_config: LoRATrainingConfig,
        continuation_config: LoRAContinuationConfig,
        total_steps: int,
        expected_parameters: Mapping[str, object],
    ) -> None:
        before_base = _validate_base_checkpoint(checkpoint_dir, model_revision)
        before_source = _tree_sha256(source_adapter_dir)
        before_model = _adapter_model_file_sha256(source_adapter_dir)
        if before_source != source_adapter_sha256 or before_model != source_adapter_model_sha256:
            raise ValueError("TimesFM LoRA warm-start source changed before model load")
        random.seed(source_config.seed)
        np.random.seed(source_config.seed)
        torch = importlib.import_module("torch")
        torch.manual_seed(source_config.seed)
        if source_config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("TimesFM LoRA continuation requested unavailable MPS")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        model_class = getattr(transformers, "TimesFm2_5ModelForPrediction", None)
        if model_class is None:
            raise ImportError(
                "installed transformers does not expose TimesFm2_5ModelForPrediction"
            )
        base = model_class.from_pretrained(
            str(checkpoint_dir),
            local_files_only=True,
            dtype=torch.float32,
        )
        base = base.to(device=source_config.device, dtype=torch.float32)
        model = peft.PeftModel.from_pretrained(
            base,
            str(source_adapter_dir),
            is_trainable=True,
            local_files_only=True,
        )
        model = model.to(device=source_config.device, dtype=torch.float32)
        after_base = _validate_base_checkpoint(checkpoint_dir, model_revision)
        after_source = _tree_sha256(source_adapter_dir)
        after_model = _adapter_model_file_sha256(source_adapter_dir)
        if before_base != after_base:
            raise RuntimeError("TimesFM base checkpoint changed during warm-start load")
        if before_source != after_source or before_model != after_model:
            raise RuntimeError("TimesFM LoRA warm-start source changed during model load")
        trainable_named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        unexpected = [name for name, _parameter in trainable_named if "lora_" not in name]
        if unexpected or not trainable_named:
            raise RuntimeError(
                "PEFT warm start exposed non-LoRA trainable parameters: "
                f"unexpected={unexpected[:5]}, trainable={len(trainable_named)}"
            )
        parameters = {
            "trainable": sum(parameter.numel() for _name, parameter in trainable_named),
            "total": sum(parameter.numel() for parameter in model.parameters()),
        }
        if not json_values_equal_exact(dict(expected_parameters), parameters):
            raise RuntimeError(
                "PEFT warm-start parameter counts differ from the source manifest: "
                f"expected={dict(expected_parameters)}, observed={parameters}"
            )
        trainable = [parameter for _name, parameter in trainable_named]
        self._optimizer = torch.optim.AdamW(
            trainable,
            lr=continuation_config.restart_learning_rate,
            weight_decay=source_config.weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=total_steps,
        )
        self._torch = torch
        self._model = model
        self._source_config = source_config
        self._trainable_names = tuple(name for name, _parameter in trainable_named)
        self._runtime_identity = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "device": source_config.device,
            "precision": "float32",
        }
        self._parameters = parameters
        self._source_adapter = {
            "tree_sha256": before_source,
            "adapter_model_sha256": before_model,
        }
        self._optimizer_identity = {
            "optimizer": "torch.optim.AdamW",
            "scheduler": "CosineAnnealingLR_per_batch",
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "restart_learning_rate": continuation_config.restart_learning_rate,
        }

    def _loss(
        self,
        contexts: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ):
        torch = self._torch
        context = torch.tensor(
            contexts, dtype=torch.float32, device=self._source_config.device
        )
        target = torch.tensor(targets, dtype=torch.float32, device=self._source_config.device)
        outputs = self._model(
            past_values=context,
            future_values=target,
            forecast_context_len=self._source_config.context_length,
            truncate_negative=False,
            force_flip_invariance=True,
        )
        loss = outputs.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("TimesFM LoRA continuation produced a non-finite loss")
        return loss

    def _train_batch(
        self,
        contexts: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float:
        torch = self._torch
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)
        loss = self._loss(contexts, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(),
            max_norm=self._source_config.gradient_clip_norm,
        )
        self._optimizer.step()
        self._scheduler.step()
        return float(loss.detach().to(device="cpu", dtype=torch.float32).item())

    def _validation_batch(
        self,
        contexts: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]],
    ) -> float:
        torch = self._torch
        self._model.eval()
        with torch.no_grad():
            loss = self._loss(contexts, targets)
        return float(loss.detach().to(device="cpu", dtype=torch.float32).item())

    def _snapshot_adapter(self) -> object:
        return {
            name: parameter.detach().to(device="cpu").clone()
            for name, parameter in self._model.named_parameters()
            if name in self._trainable_names
        }

    def _restore_adapter(self, state: object) -> None:
        if type(state) is not dict or set(state) != set(self._trainable_names):
            raise RuntimeError("TimesFM LoRA best-adapter state is malformed")
        with self._torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if name in state:
                    expected = state[name]
                    if tuple(expected.shape) != tuple(parameter.shape):
                        raise RuntimeError("TimesFM LoRA best-adapter tensor shape changed")
                    parameter.copy_(expected.to(device=parameter.device, dtype=parameter.dtype))

    def _save_adapter(self, path: pathlib.Path) -> None:
        self._model.save_pretrained(path, safe_serialization=True)

    def _identity(self) -> Mapping[str, object]:
        return {
            "runtime": dict(self._runtime_identity),
            "parameters": dict(self._parameters),
            "source_adapter": dict(self._source_adapter),
            "optimizer": dict(self._optimizer_identity),
        }


def _training_batches(
    table: pa.Table,
    config: LoRATrainingConfig,
    n_series: int,
):
    epochs = table["epoch"].to_pylist()
    series_ids = table["series_id"].to_pylist()
    for epoch in range(config.epochs):
        indices = [index for index, value in enumerate(epochs) if value == epoch]
        if len(indices) != n_series:
            raise ValueError(f"TimesFM LoRA epoch {epoch} does not contain every eligible series")
        if len({series_ids[index] for index in indices}) != n_series:
            raise ValueError(f"TimesFM LoRA epoch {epoch} repeats or omits a series")
        indices.sort(key=lambda index: series_ids[index].encode("utf-8"))
        for start in range(0, len(indices), config.batch_size):
            selected = pa.array(indices[start : start + config.batch_size], type=pa.int64())
            batch = table.take(selected)
            contexts = batch["context"].to_pylist()
            targets = batch["target"].to_pylist()
            if not np.isfinite(np.asarray(contexts, dtype=np.float32)).all():
                raise ValueError("TimesFM LoRA context batch contains a non-finite value")
            if not np.isfinite(np.asarray(targets, dtype=np.float32)).all():
                raise ValueError("TimesFM LoRA target batch contains a non-finite value")
            yield contexts, targets


def train_lora(
    training_rows_dir: str | pathlib.Path,
    training_rows_sha256: str,
    training_rows_manifest_sha256: str,
    *,
    checkpoint_dir: str | pathlib.Path,
    model_revision: str,
    out_dir: str | pathlib.Path,
    config: LoRATrainingConfig,
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    backend: _TrainingBackend | None = None,
) -> LoRAAdapterArtifact:
    """Train all registered epochs and atomically publish exactly one final PEFT adapter.

    There is no validation callback, early stopping, best-epoch selection, or intermediate save
    path in this function.  Training loss may be observed by a runtime, but it cannot select the
    artifact.  ``backend`` is the same public-boundary injection seam used by functional tests;
    production callers omit it and use Transformers plus PEFT.
    """
    _validate_training_config(config)
    runtime_lock = validate_runtime_lock(
        runtime_lock_path,
        runtime_lock_file_sha256,
        runtime_lock_record_path=runtime_lock_record_path,
        runtime_lock_sha256=runtime_lock_sha256,
    )
    rows_dir = pathlib.Path(training_rows_dir)
    manifest, _rows_manifest_path, table = _validate_training_rows(
        rows_dir,
        training_rows_sha256,
        training_rows_manifest_sha256,
        config,
    )
    base_dir = pathlib.Path(checkpoint_dir)
    _validate_base_checkpoint(base_dir, model_revision)
    rows_record = manifest["rows"]
    assert isinstance(rows_record, dict)
    panel_record = manifest["panel"]
    assert isinstance(panel_record, dict)
    panel_content_sha256 = panel_record["content_sha256"]
    assert isinstance(panel_content_sha256, str)
    n_series = rows_record["n_series"]
    assert isinstance(n_series, int)
    total_steps = config.epochs * math.ceil(n_series / config.batch_size)
    runtime = backend or _PeftTrainingRuntime(base_dir, model_revision, config, total_steps)

    target = pathlib.Path(out_dir)
    _validate_empty_target(target, "TimesFM LoRA adapter")
    temporary = pathlib.Path(
        tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.timesfm-lora.")
    )
    steps = 0
    try:
        for contexts, targets in _training_batches(table, config, n_series):
            loss = runtime._train_batch(contexts, targets)
            if not math.isfinite(float(loss)):
                raise RuntimeError("TimesFM LoRA backend returned a non-finite training loss")
            steps += 1
        if steps != total_steps:
            raise AssertionError(
                f"TimesFM LoRA ran {steps} batches, expected the registered {total_steps}"
            )
        final_runtime_lock = validate_runtime_lock(
            runtime_lock_path,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
        )
        if final_runtime_lock != runtime_lock:
            raise RuntimeError("TimesFM worker runtime lock changed during LoRA training")
        runtime._save_adapter(temporary)
        if (temporary / LORA_MANIFEST).exists():
            raise RuntimeError("PEFT backend attempted to replace the governed LoRA manifest")
        identity = runtime._identity()
        if type(identity) is not dict or set(identity) != {"runtime", "parameters"}:
            raise ValueError("TimesFM LoRA backend identity must contain runtime and parameters")
        runtime_identity = identity["runtime"]
        if type(runtime_identity) is not dict or "runtime_lock" in runtime_identity:
            raise ValueError("TimesFM LoRA backend runtime identity is malformed")
        artifact_manifest = {
            "schema": LORA_ARTIFACT_SCHEMA,
            "base_model": {
                "model_id": TRANSFORMERS_MODEL_ID,
                "revision": TRANSFORMERS_REVISION,
                "config_sha256": TRANSFORMERS_CONFIG_SHA256,
                "weights_sha256": TRANSFORMERS_WEIGHTS_SHA256,
            },
            "training_rows": {
                "sha256": training_rows_sha256,
                "manifest_sha256": training_rows_manifest_sha256,
                "n_rows": table.num_rows,
                "cutoff": manifest["cutoff"],
                "panel_content_sha256": panel_content_sha256,
            },
            "training": _config_record(config),
            "runtime": {
                **runtime_identity,
                "runtime_lock": _runtime_lock_record(runtime_lock),
            },
            "parameters": identity["parameters"],
            "publication": {
                "saved_after_all_epochs": True,
                "intermediate_checkpoints": 0,
                "evaluation_runs": 0,
            },
        }
        adapter_manifest_path = temporary / LORA_MANIFEST
        adapter_manifest_path.write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        adapter_digest = _tree_sha256(temporary)
        _publish_directory(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return LoRAAdapterArtifact(
        path=str(target),
        sha256=adapter_digest,
        manifest_path=str(target / LORA_MANIFEST),
        n_training_rows=table.num_rows,
        training_rows_sha256=training_rows_sha256,
        panel_content_sha256=panel_content_sha256,
    )


def _authenticate_lora_continuation_sources(
    training_rows_dir: str | pathlib.Path,
    training_rows_sha256: str,
    training_rows_manifest_sha256: str,
    *,
    source_adapter_dir: str | pathlib.Path,
    source_adapter_sha256: str,
    source_adapter_model_sha256: str,
    checkpoint_dir: str | pathlib.Path,
    model_revision: str,
    source_config: LoRATrainingConfig,
    expected_panel_content_sha256: str,
    expected_training_cutoff: str,
    panel_validation_path: str | pathlib.Path | None,
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    recorded_source_adapter_dir: pathlib.Path | None = None,
    recorded_runtime_lock: RuntimeLockIdentity | None = None,
) -> tuple[
    dict[str, object],
    pa.Table,
    RuntimeLockIdentity,
    dict[str, object],
    dict[str, object],
]:
    """Authenticate every immutable warm-start input and return its inert identities."""
    _validate_training_config(source_config)
    if source_config.epochs != 3:
        raise ValueError("LoRA continuation requires the authenticated three-row source pool")
    expected_source_tree = _require_sha256(
        source_adapter_sha256, "source_adapter_sha256"
    )
    expected_source_model = _require_sha256(
        source_adapter_model_sha256, "source_adapter_model_sha256"
    )
    expected_panel = _require_sha256(
        expected_panel_content_sha256, "expected_panel_content_sha256"
    )
    if source_config == PROSPECTIVE_M5_LORA_CONFIG:
        theta1913 = M5_LORA_CUTOFF_IDENTITIES[-1]
        if (
            expected_training_cutoff != theta1913.cutoff
            or training_rows_sha256 != theta1913.training_rows_sha256
            or expected_panel != M5_PANEL_CONTENT_SHA256
        ):
            raise ValueError(
                "production LoRA continuation requires the sealed theta1913 row pool"
            )
    base = pathlib.Path(checkpoint_dir)
    _validate_base_checkpoint(base, model_revision)
    if recorded_runtime_lock is None:
        runtime = validate_runtime_lock(
            runtime_lock_path,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
        )
    else:
        runtime = _validate_snapshotted_runtime_lock(
            runtime_lock_path,
            runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_identity=recorded_runtime_lock,
        )
    rows_artifact = validate_training_rows_artifact(
        training_rows_dir,
        training_rows_sha256,
        training_rows_manifest_sha256,
        config=source_config,
        expected_cutoff=expected_training_cutoff,
        expected_panel_content_sha256=expected_panel,
        panel_validation_path=panel_validation_path,
    )
    rows_manifest, _rows_manifest_path, table = _validate_training_rows(
        pathlib.Path(training_rows_dir),
        training_rows_sha256,
        training_rows_manifest_sha256,
        source_config,
    )
    source_path = pathlib.Path(source_adapter_dir)
    validate_lora_adapter(
        source_path,
        expected_source_tree,
        config=source_config,
        expected_training_rows_sha256=training_rows_sha256,
        expected_training_rows_manifest_sha256=training_rows_manifest_sha256,
        expected_panel_content_sha256=expected_panel,
        expected_training_cutoff=expected_training_cutoff,
        forecast_origin=expected_training_cutoff,
        runtime_lock_path=runtime_lock_path,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
        runtime_lock_record_path=runtime_lock_record_path,
        runtime_lock_sha256=runtime_lock_sha256,
        _recorded_runtime_lock=recorded_runtime_lock,
    )
    observed_source_model = _adapter_model_file_sha256(source_path)
    if observed_source_model != expected_source_model:
        raise ValueError(
            "TimesFM LoRA source adapter_model.safetensors hash differs: "
            f"expected {expected_source_model}, got {observed_source_model}"
        )
    source_manifest_path = source_path / LORA_MANIFEST
    source_manifest = _validate_lora_manifest(source_path)
    if source_manifest.get("schema") != LORA_ARTIFACT_SCHEMA:
        raise ValueError("LoRA continuation source must be an exact v1 final adapter")
    source_parameters = source_manifest.get("parameters")
    if (
        type(source_parameters) is not dict
        or set(source_parameters) != {"trainable", "total"}
        or any(type(value) is not int or value < 1 for value in source_parameters.values())
    ):
        raise ValueError("LoRA continuation source has invalid parameter counts")
    recorded_source_path = recorded_source_adapter_dir or source_path
    if not recorded_source_path.is_absolute():
        raise ValueError("recorded TimesFM source-adapter path must be absolute")
    source_identity = {
        "path": str(recorded_source_path),
        "tree_sha256": expected_source_tree,
        "adapter_model_file": "adapter_model.safetensors",
        "adapter_model_sha256": expected_source_model,
        "manifest_path": str(recorded_source_path / LORA_MANIFEST),
        "manifest_sha256": _file_sha256(source_manifest_path),
        "manifest_schema": LORA_ARTIFACT_SCHEMA,
    }
    rows_identity = {
        "sha256": rows_artifact.rows_sha256,
        "manifest_sha256": rows_artifact.manifest_sha256,
        "n_rows": rows_artifact.n_rows,
        "cutoff": rows_artifact.cutoff,
        "panel_content_sha256": rows_artifact.panel_content_sha256,
    }
    if rows_manifest.get("cutoff") != expected_training_cutoff:
        raise ValueError("LoRA continuation row manifest changed its cutoff")
    if (
        _tree_sha256(source_path) != expected_source_tree
        or _adapter_model_file_sha256(source_path) != expected_source_model
        or _file_sha256(source_manifest_path) != source_identity["manifest_sha256"]
    ):
        raise RuntimeError("TimesFM LoRA source changed while it was authenticated")
    return source_identity, table, runtime, rows_identity, dict(source_parameters)


def _validate_continuation_backend_identity(
    identity: object,
    *,
    source_identity: Mapping[str, object],
    expected_parameters: Mapping[str, object],
    runtime_lock: RuntimeLockIdentity,
    source_config: LoRATrainingConfig,
    continuation_config: LoRAContinuationConfig,
) -> dict[str, object]:
    if type(identity) is not dict or set(identity) != {
        "runtime",
        "parameters",
        "source_adapter",
        "optimizer",
    }:
        raise ValueError("TimesFM LoRA continuation backend identity is malformed")
    runtime = identity["runtime"]
    packages = dict(runtime_lock.package_versions)
    expected_runtime = {
        "torch": packages["torch"],
        "transformers": packages["transformers"],
        "peft": packages["peft"],
        "device": source_config.device,
        "precision": "float32",
    }
    if not json_values_equal_exact(runtime, expected_runtime):
        raise ValueError("TimesFM LoRA continuation runtime identity is malformed")
    if not json_values_equal_exact(
        identity["parameters"], dict(expected_parameters)
    ):
        raise ValueError("TimesFM LoRA continuation parameter counts changed")
    if not json_values_equal_exact(identity["source_adapter"], {
        "tree_sha256": source_identity["tree_sha256"],
        "adapter_model_sha256": source_identity["adapter_model_sha256"],
    }):
        raise ValueError("TimesFM LoRA continuation backend loaded another source adapter")
    expected_optimizer = {
        "optimizer": "torch.optim.AdamW",
        "scheduler": "CosineAnnealingLR_per_batch",
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "restart_learning_rate": continuation_config.restart_learning_rate,
    }
    if not json_values_equal_exact(identity["optimizer"], expected_optimizer):
        raise ValueError("TimesFM LoRA continuation did not use a fresh optimizer/scheduler")
    return identity


def _mean_backend_loss(
    batches,
    callback,
    *,
    label: str,
) -> float:
    weighted_loss = 0.0
    n_rows = 0
    for contexts, targets in batches:
        loss = float(callback(contexts, targets))
        if not math.isfinite(loss):
            raise RuntimeError(f"TimesFM LoRA backend returned a non-finite {label} loss")
        weighted_loss += loss * len(contexts)
        n_rows += len(contexts)
    if n_rows < 1:
        raise ValueError(f"TimesFM LoRA continuation has no {label} rows")
    result = weighted_loss / n_rows
    if not math.isfinite(result):
        raise RuntimeError(f"TimesFM LoRA continuation {label} loss is non-finite")
    return result


@dataclass
class _ContinuationInputSnapshots:
    """Owned private copies used by exactly one continuation call."""

    temporary: tempfile.TemporaryDirectory[str]
    root: pathlib.Path
    training_rows_dir: pathlib.Path
    panel_validation_path: pathlib.Path
    source_adapter_dir: pathlib.Path
    checkpoint_dir: pathlib.Path
    runtime_lock_path: pathlib.Path
    recorded_source_adapter_dir: pathlib.Path
    recorded_runtime_lock: RuntimeLockIdentity

    def cleanup(self) -> None:
        self.temporary.cleanup()


def _materialize_continuation_input_snapshots(
    training_rows_dir: str | pathlib.Path,
    training_rows_sha256: str,
    training_rows_manifest_sha256: str,
    *,
    source_adapter_dir: str | pathlib.Path,
    source_adapter_sha256: str,
    source_adapter_model_sha256: str,
    checkpoint_dir: str | pathlib.Path,
    model_revision: str,
    source_config: LoRATrainingConfig,
    expected_panel_content_sha256: str,
    expected_training_cutoff: str,
    panel_validation_path: str | pathlib.Path | None,
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    recorded_source_adapter_dir: str | pathlib.Path | None,
    recorded_runtime_lock: RuntimeLockIdentity | Mapping[str, object] | None,
) -> _ContinuationInputSnapshots:
    """Authenticate/copy/re-authenticate all warm-start inputs before model loading."""
    rows_source = pathlib.Path(training_rows_dir)
    adapter_source = pathlib.Path(source_adapter_dir)
    base_source = pathlib.Path(checkpoint_dir)
    lock_source = pathlib.Path(runtime_lock_path)
    recorded_adapter = (
        pathlib.Path(recorded_source_adapter_dir)
        if recorded_source_adapter_dir is not None
        else adapter_source
    )
    recorded_runtime = _coerce_runtime_lock_identity(recorded_runtime_lock)
    before = _authenticate_lora_continuation_sources(
        rows_source,
        training_rows_sha256,
        training_rows_manifest_sha256,
        source_adapter_dir=adapter_source,
        source_adapter_sha256=source_adapter_sha256,
        source_adapter_model_sha256=source_adapter_model_sha256,
        checkpoint_dir=base_source,
        model_revision=model_revision,
        source_config=source_config,
        expected_panel_content_sha256=expected_panel_content_sha256,
        expected_training_cutoff=expected_training_cutoff,
        panel_validation_path=panel_validation_path,
        runtime_lock_path=lock_source,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
        runtime_lock_record_path=runtime_lock_record_path,
        runtime_lock_sha256=runtime_lock_sha256,
        recorded_source_adapter_dir=recorded_adapter,
        recorded_runtime_lock=recorded_runtime,
    )
    if panel_validation_path is None:
        rows_manifest = _strict_record(
            rows_source / TRAINING_ROWS_MANIFEST,
            "TimesFM training-rows manifest",
        )
        panel_record = rows_manifest.get("panel")
        if type(panel_record) is not dict or type(panel_record.get("path")) is not str:
            raise ValueError("TimesFM training rows do not record a source panel path")
        panel_source = pathlib.Path(panel_record["path"])
    else:
        panel_source = pathlib.Path(panel_validation_path)
    temporary: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="autoforecast-timesfm-continuation."
    )
    root = pathlib.Path(temporary.name).resolve()
    private_rows = root / "training-rows"
    private_adapter = root / "source-adapter"
    private_base = root / "base"
    private_lock = root / "runtime-lock.json"
    try:
        _copy_private_tree(rows_source, private_rows, label="TimesFM training rows")
        private_panel = _copy_private_file_or_tree(
            panel_source,
            root,
            label="TimesFM source panel",
        )
        _copy_private_tree(adapter_source, private_adapter, label="TimesFM source adapter")
        _copy_private_tree(base_source, private_base, label="TimesFM base checkpoint")
        _copy_private_file(lock_source, private_lock, label="TimesFM runtime lock")
        after = _authenticate_lora_continuation_sources(
            rows_source,
            training_rows_sha256,
            training_rows_manifest_sha256,
            source_adapter_dir=adapter_source,
            source_adapter_sha256=source_adapter_sha256,
            source_adapter_model_sha256=source_adapter_model_sha256,
            checkpoint_dir=base_source,
            model_revision=model_revision,
            source_config=source_config,
            expected_panel_content_sha256=expected_panel_content_sha256,
            expected_training_cutoff=expected_training_cutoff,
            panel_validation_path=panel_source,
            runtime_lock_path=lock_source,
            runtime_lock_file_sha256=runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_source_adapter_dir=recorded_adapter,
            recorded_runtime_lock=before[2],
        )
        if (
            not json_values_equal_exact(after[0], before[0])
            or after[2] != before[2]
            or not json_values_equal_exact(after[3], before[3])
            or not json_values_equal_exact(after[4], before[4])
        ):
            raise RuntimeError("TimesFM continuation inputs changed while snapshotted")
        return _ContinuationInputSnapshots(
            temporary=temporary,
            root=root,
            training_rows_dir=private_rows,
            panel_validation_path=private_panel,
            source_adapter_dir=private_adapter,
            checkpoint_dir=private_base,
            runtime_lock_path=private_lock,
            recorded_source_adapter_dir=recorded_adapter,
            recorded_runtime_lock=before[2],
        )
    except BaseException:
        temporary.cleanup()
        raise


def continue_lora(
    training_rows_dir: str | pathlib.Path,
    training_rows_sha256: str,
    training_rows_manifest_sha256: str,
    *,
    source_adapter_dir: str | pathlib.Path,
    source_adapter_sha256: str,
    source_adapter_model_sha256: str,
    checkpoint_dir: str | pathlib.Path,
    model_revision: str,
    out_dir: str | pathlib.Path,
    source_config: LoRATrainingConfig,
    continuation_config: LoRAContinuationConfig,
    expected_panel_content_sha256: str,
    expected_training_cutoff: str,
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    panel_validation_path: str | pathlib.Path | None = None,
    recorded_source_adapter_dir: str | pathlib.Path | None = None,
    recorded_runtime_lock: RuntimeLockIdentity | Mapping[str, object] | None = None,
    backend: _ContinuationBackend | None = None,
) -> LoRAAdapterArtifact:
    """Snapshot every warm-start input, then run one governed continuation."""
    _validate_training_config(source_config)
    _validate_continuation_config(continuation_config)
    target = pathlib.Path(out_dir)
    _validate_empty_target(target, "TimesFM LoRA continuation adapter")
    snapshots = _materialize_continuation_input_snapshots(
        training_rows_dir,
        training_rows_sha256,
        training_rows_manifest_sha256,
        source_adapter_dir=source_adapter_dir,
        source_adapter_sha256=source_adapter_sha256,
        source_adapter_model_sha256=source_adapter_model_sha256,
        checkpoint_dir=checkpoint_dir,
        model_revision=model_revision,
        source_config=source_config,
        expected_panel_content_sha256=expected_panel_content_sha256,
        expected_training_cutoff=expected_training_cutoff,
        panel_validation_path=panel_validation_path,
        runtime_lock_path=runtime_lock_path,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
        runtime_lock_record_path=runtime_lock_record_path,
        runtime_lock_sha256=runtime_lock_sha256,
        recorded_source_adapter_dir=recorded_source_adapter_dir,
        recorded_runtime_lock=recorded_runtime_lock,
    )
    try:
        return _continue_lora_from_snapshots(
            snapshots.training_rows_dir,
            training_rows_sha256,
            training_rows_manifest_sha256,
            source_adapter_dir=snapshots.source_adapter_dir,
            source_adapter_sha256=source_adapter_sha256,
            source_adapter_model_sha256=source_adapter_model_sha256,
            checkpoint_dir=snapshots.checkpoint_dir,
            model_revision=model_revision,
            out_dir=out_dir,
            source_config=source_config,
            continuation_config=continuation_config,
            expected_panel_content_sha256=expected_panel_content_sha256,
            expected_training_cutoff=expected_training_cutoff,
            panel_validation_path=snapshots.panel_validation_path,
            runtime_lock_path=snapshots.runtime_lock_path,
            runtime_lock_file_sha256=runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            backend=backend,
            snapshot_root=snapshots.root,
            recorded_source_adapter_dir=snapshots.recorded_source_adapter_dir,
            recorded_runtime_lock=snapshots.recorded_runtime_lock,
        )
    finally:
        snapshots.cleanup()


def _continue_lora_from_snapshots(
    training_rows_dir: str | pathlib.Path,
    training_rows_sha256: str,
    training_rows_manifest_sha256: str,
    *,
    source_adapter_dir: str | pathlib.Path,
    source_adapter_sha256: str,
    source_adapter_model_sha256: str,
    checkpoint_dir: str | pathlib.Path,
    model_revision: str,
    out_dir: str | pathlib.Path,
    source_config: LoRATrainingConfig,
    continuation_config: LoRAContinuationConfig,
    expected_panel_content_sha256: str,
    expected_training_cutoff: str,
    panel_validation_path: pathlib.Path,
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    backend: _ContinuationBackend | None,
    snapshot_root: pathlib.Path,
    recorded_source_adapter_dir: pathlib.Path,
    recorded_runtime_lock: RuntimeLockIdentity,
) -> LoRAAdapterArtifact:
    """Warm-start one exact v1 adapter with validation-based early stopping.

    The three authenticated source-row epochs remain a fixed pool. Each additional epoch uses
    one row per training series and cycles that pool; validation always uses every source row for
    a disjoint deterministic 10% series split. Epoch zero is a valid best candidate. No optimizer
    or scheduler state is loaded, no intermediate adapter is published, and accuracy evaluation
    is outside this boundary.
    """
    _validate_training_config(source_config)
    _validate_continuation_config(continuation_config)
    target = pathlib.Path(out_dir)
    _validate_empty_target(target, "TimesFM LoRA continuation adapter")
    private_input_identity = _source_tree_identity(
        snapshot_root,
        label="private TimesFM continuation inputs",
    )
    source_identity, table, runtime_lock, rows_identity, source_parameters = (
        _authenticate_lora_continuation_sources(
            training_rows_dir,
            training_rows_sha256,
            training_rows_manifest_sha256,
            source_adapter_dir=source_adapter_dir,
            source_adapter_sha256=source_adapter_sha256,
            source_adapter_model_sha256=source_adapter_model_sha256,
            checkpoint_dir=checkpoint_dir,
            model_revision=model_revision,
            source_config=source_config,
            expected_panel_content_sha256=expected_panel_content_sha256,
            expected_training_cutoff=expected_training_cutoff,
            panel_validation_path=panel_validation_path,
            runtime_lock_path=runtime_lock_path,
            runtime_lock_file_sha256=runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_source_adapter_dir=recorded_source_adapter_dir,
            recorded_runtime_lock=recorded_runtime_lock,
        )
    )
    if _source_tree_identity(
        snapshot_root,
        label="private TimesFM continuation inputs",
    ) != private_input_identity:
        raise RuntimeError("private TimesFM continuation inputs changed while rows were read")
    train_series, validation_series, split_record = _continuation_series_split(
        table, source_config
    )
    steps_per_epoch = math.ceil(len(train_series) / source_config.batch_size)
    total_steps = continuation_config.max_additional_epochs * steps_per_epoch
    training_runtime = backend or _PeftContinuationRuntime(
        pathlib.Path(checkpoint_dir),
        model_revision,
        pathlib.Path(source_adapter_dir),
        source_adapter_sha256,
        source_adapter_model_sha256,
        source_config,
        continuation_config,
        total_steps,
        source_parameters,
    )
    if _source_tree_identity(
        snapshot_root,
        label="private TimesFM continuation inputs",
    ) != private_input_identity:
        raise RuntimeError("private TimesFM continuation inputs changed during provider load")
    initial_identity = _validate_continuation_backend_identity(
        training_runtime._identity(),
        source_identity=source_identity,
        expected_parameters=source_parameters,
        runtime_lock=runtime_lock,
        source_config=source_config,
        continuation_config=continuation_config,
    )

    def validation_loss() -> float:
        return _mean_backend_loss(
            _selected_training_batches(
                table,
                selected_series=validation_series,
                batch_size=source_config.batch_size,
                source_epoch=None,
            ),
            training_runtime._validation_batch,
            label="validation",
        )

    baseline_loss = validation_loss()
    best_loss = baseline_loss
    last_validation_loss = baseline_loss
    best_epoch = 0
    best_state = training_runtime._snapshot_adapter()
    waited = 0
    actual_epochs = 0
    last_training_loss = float("nan")
    stop_reason = "maximum_additional_epochs"
    validation_trace: list[dict[str, object]] = [
        {
            "additional_epoch": 0,
            "source_row_epoch": None,
            "training_loss": None,
            "validation_loss": baseline_loss,
        }
    ]
    for additional_epoch in range(1, continuation_config.max_additional_epochs + 1):
        source_epoch = (additional_epoch - 1) % 3
        last_training_loss = _mean_backend_loss(
            _selected_training_batches(
                table,
                selected_series=train_series,
                batch_size=source_config.batch_size,
                source_epoch=source_epoch,
            ),
            training_runtime._train_batch,
            label="training",
        )
        last_validation_loss = validation_loss()
        actual_epochs = additional_epoch
        validation_trace.append(
            {
                "additional_epoch": additional_epoch,
                "source_row_epoch": source_epoch,
                "training_loss": last_training_loss,
                "validation_loss": last_validation_loss,
            }
        )
        threshold = max(
            continuation_config.absolute_min_delta,
            continuation_config.relative_min_delta * abs(best_loss),
        )
        if last_validation_loss < best_loss - threshold:
            best_loss = last_validation_loss
            best_epoch = additional_epoch
            best_state = training_runtime._snapshot_adapter()
            waited = 0
        else:
            waited += 1
            if waited >= continuation_config.patience:
                stop_reason = "early_stopping_patience"
                break
    training_runtime._restore_adapter(best_state)
    if actual_epochs < 1 or not math.isfinite(last_training_loss):
        raise RuntimeError("TimesFM LoRA continuation completed no additional epoch")
    final_identity = _validate_continuation_backend_identity(
        training_runtime._identity(),
        source_identity=source_identity,
        expected_parameters=source_parameters,
        runtime_lock=runtime_lock,
        source_config=source_config,
        continuation_config=continuation_config,
    )
    if final_identity != initial_identity:
        raise RuntimeError("TimesFM LoRA continuation backend identity changed during training")
    if _source_tree_identity(
        snapshot_root,
        label="private TimesFM continuation inputs",
    ) != private_input_identity:
        raise RuntimeError("private TimesFM continuation inputs changed during training")

    temporary = pathlib.Path(
        tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.timesfm-lora-continuation.")
    )
    try:
        reauthenticated = _authenticate_lora_continuation_sources(
            training_rows_dir,
            training_rows_sha256,
            training_rows_manifest_sha256,
            source_adapter_dir=source_adapter_dir,
            source_adapter_sha256=source_adapter_sha256,
            source_adapter_model_sha256=source_adapter_model_sha256,
            checkpoint_dir=checkpoint_dir,
            model_revision=model_revision,
            source_config=source_config,
            expected_panel_content_sha256=expected_panel_content_sha256,
            expected_training_cutoff=expected_training_cutoff,
            panel_validation_path=panel_validation_path,
            runtime_lock_path=runtime_lock_path,
            runtime_lock_file_sha256=runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_source_adapter_dir=recorded_source_adapter_dir,
            recorded_runtime_lock=recorded_runtime_lock,
        )
        if (
            not json_values_equal_exact(reauthenticated[0], source_identity)
            or reauthenticated[2] != runtime_lock
            or not json_values_equal_exact(reauthenticated[3], rows_identity)
            or not json_values_equal_exact(reauthenticated[4], source_parameters)
        ):
            raise RuntimeError("TimesFM LoRA continuation inputs changed during training")
        if _source_tree_identity(
            snapshot_root,
            label="private TimesFM continuation inputs",
        ) != private_input_identity:
            raise RuntimeError("private TimesFM continuation inputs changed during recheck")
        training_runtime._save_adapter(temporary)
        if (temporary / LORA_MANIFEST).exists():
            raise RuntimeError("PEFT backend attempted to replace the continuation manifest")
        _adapter_model_file_sha256(temporary)
        post_save = _authenticate_lora_continuation_sources(
            training_rows_dir,
            training_rows_sha256,
            training_rows_manifest_sha256,
            source_adapter_dir=source_adapter_dir,
            source_adapter_sha256=source_adapter_sha256,
            source_adapter_model_sha256=source_adapter_model_sha256,
            checkpoint_dir=checkpoint_dir,
            model_revision=model_revision,
            source_config=source_config,
            expected_panel_content_sha256=expected_panel_content_sha256,
            expected_training_cutoff=expected_training_cutoff,
            panel_validation_path=panel_validation_path,
            runtime_lock_path=runtime_lock_path,
            runtime_lock_file_sha256=runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_source_adapter_dir=recorded_source_adapter_dir,
            recorded_runtime_lock=recorded_runtime_lock,
        )
        if (
            not json_values_equal_exact(post_save[0], source_identity)
            or post_save[2] != runtime_lock
            or not json_values_equal_exact(post_save[3], rows_identity)
            or not json_values_equal_exact(post_save[4], source_parameters)
        ):
            raise RuntimeError("TimesFM LoRA continuation inputs changed during final save")
        if _source_tree_identity(
            snapshot_root,
            label="private TimesFM continuation inputs",
        ) != private_input_identity:
            raise RuntimeError("private TimesFM continuation inputs changed during final save")
        manifest = {
            "schema": LORA_CONTINUATION_ARTIFACT_SCHEMA,
            "base_model": _base_model_record(),
            "source_adapter": source_identity,
            "training_rows": rows_identity,
            "split": split_record,
            "training": _continuation_training_record(
                source_config, continuation_config
            ),
            "stopping": {
                "metric": "mean_native_loss_on_fixed_validation_rows",
                "baseline_epoch": 0,
                "baseline_validation_loss": baseline_loss,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "last_training_loss": last_training_loss,
                "last_validation_loss": last_validation_loss,
                "actual_additional_epochs": actual_epochs,
                "patience": continuation_config.patience,
                "waited_at_stop": waited,
                "stop_reason": stop_reason,
                "improvement_rule": (
                    "validation_loss < best_loss - "
                    "max(absolute_min_delta,relative_min_delta*abs(best_loss))"
                ),
                "absolute_min_delta": continuation_config.absolute_min_delta,
                "relative_min_delta": continuation_config.relative_min_delta,
                "best_adapter_restored": True,
                "validation_trace": validation_trace,
            },
            "runtime": {
                **final_identity["runtime"],
                "runtime_lock": _runtime_lock_record(runtime_lock),
            },
            "parameters": final_identity["parameters"],
            "publication": {
                "saved_after_best_adapter_restoration": True,
                "intermediate_checkpoints": 0,
                "accuracy_evaluation_runs": 0,
                "validation_runs": actual_epochs + 1,
            },
        }
        manifest_path = temporary / LORA_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        _validate_lora_manifest(temporary)
        adapter_digest = _tree_sha256(temporary)
        _publish_directory(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return LoRAAdapterArtifact(
        path=str(target),
        sha256=adapter_digest,
        manifest_path=str(target / LORA_MANIFEST),
        n_training_rows=rows_identity["n_rows"],
        training_rows_sha256=rows_identity["sha256"],
        panel_content_sha256=rows_identity["panel_content_sha256"],
    )


def validate_lora_continuation_adapter(
    adapter_dir: str | pathlib.Path,
    adapter_sha256: str,
    *,
    training_rows_dir: str | pathlib.Path,
    training_rows_sha256: str,
    training_rows_manifest_sha256: str,
    source_adapter_dir: str | pathlib.Path,
    source_adapter_sha256: str,
    source_adapter_model_sha256: str,
    checkpoint_dir: str | pathlib.Path,
    model_revision: str,
    source_config: LoRATrainingConfig,
    continuation_config: LoRAContinuationConfig,
    expected_panel_content_sha256: str,
    expected_training_cutoff: str,
    forecast_origin: str,
    runtime_lock_path: str | pathlib.Path,
    runtime_lock_file_sha256: str,
    runtime_lock_record_path: Sequence[str],
    runtime_lock_sha256: str,
    panel_validation_path: str | pathlib.Path | None = None,
    recorded_source_adapter_dir: str | pathlib.Path | None = None,
    recorded_runtime_lock: RuntimeLockIdentity | Mapping[str, object] | None = None,
) -> LoRAAdapterArtifact:
    """Authenticate one final continuation adapter and its complete immediate lineage."""
    _validate_continuation_config(continuation_config)
    path = pathlib.Path(adapter_dir)
    expected_digest = _require_sha256(adapter_sha256, "adapter_sha256")
    before = _tree_sha256(path)
    if before != expected_digest:
        raise ValueError(
            f"TimesFM LoRA continuation hash mismatch: expected {expected_digest}, got {before}"
        )
    source_identity, table, runtime, rows_identity, source_parameters = (
        _authenticate_lora_continuation_sources(
            training_rows_dir,
            training_rows_sha256,
            training_rows_manifest_sha256,
            source_adapter_dir=source_adapter_dir,
            source_adapter_sha256=source_adapter_sha256,
            source_adapter_model_sha256=source_adapter_model_sha256,
            checkpoint_dir=checkpoint_dir,
            model_revision=model_revision,
            source_config=source_config,
            expected_panel_content_sha256=expected_panel_content_sha256,
            expected_training_cutoff=expected_training_cutoff,
            panel_validation_path=panel_validation_path,
            runtime_lock_path=runtime_lock_path,
            runtime_lock_file_sha256=runtime_lock_file_sha256,
            runtime_lock_record_path=runtime_lock_record_path,
            runtime_lock_sha256=runtime_lock_sha256,
            recorded_source_adapter_dir=(
                pathlib.Path(recorded_source_adapter_dir)
                if recorded_source_adapter_dir is not None
                else None
            ),
            recorded_runtime_lock=_coerce_runtime_lock_identity(recorded_runtime_lock),
        )
    )
    manifest = _validate_lora_manifest(path)
    if manifest.get("schema") != LORA_CONTINUATION_ARTIFACT_SCHEMA:
        raise ValueError("continuation validation requires a continuation manifest")
    _train_series, _validation_series, split = _continuation_series_split(
        table, source_config
    )
    if not json_values_equal_exact(manifest.get("source_adapter"), source_identity):
        raise ValueError("TimesFM LoRA continuation binds another source adapter")
    if not json_values_equal_exact(manifest.get("training_rows"), rows_identity):
        raise ValueError("TimesFM LoRA continuation binds different training rows")
    if not json_values_equal_exact(manifest.get("split"), split):
        raise ValueError("TimesFM LoRA continuation series split changed")
    if not json_values_equal_exact(
        manifest.get("training"),
        _continuation_training_record(source_config, continuation_config),
    ):
        raise ValueError("TimesFM LoRA continuation controls changed")
    if not json_values_equal_exact(manifest.get("parameters"), source_parameters):
        raise ValueError("TimesFM LoRA continuation parameter counts changed")
    manifest_runtime = manifest.get("runtime")
    runtime_packages = dict(runtime.package_versions)
    expected_runtime = {
        "torch": runtime_packages["torch"],
        "transformers": runtime_packages["transformers"],
        "peft": runtime_packages["peft"],
        "device": source_config.device,
        "precision": "float32",
        "runtime_lock": _runtime_lock_record(runtime),
    }
    if not json_values_equal_exact(manifest_runtime, expected_runtime):
        raise ValueError("TimesFM LoRA continuation runtime lock changed")
    stopping = manifest["stopping"]
    assert isinstance(stopping, dict)
    if (
        not json_values_equal_exact(
            stopping.get("patience"), continuation_config.patience
        )
        or not json_values_equal_exact(
            stopping.get("absolute_min_delta"), continuation_config.absolute_min_delta
        )
        or not json_values_equal_exact(
            stopping.get("relative_min_delta"), continuation_config.relative_min_delta
        )
    ):
        raise ValueError("TimesFM LoRA continuation stopping controls changed")
    try:
        cutoff_day = dt.date.fromisoformat(expected_training_cutoff)
        forecast_day = dt.date.fromisoformat(forecast_origin)
    except (TypeError, ValueError) as exc:
        raise ValueError("continuation cutoff and forecast_origin must be canonical dates") from exc
    if (
        cutoff_day.isoformat() != expected_training_cutoff
        or forecast_day.isoformat() != forecast_origin
        or cutoff_day > forecast_day
    ):
        raise ValueError("LoRA continuation may not backcast before its training cutoff")
    after = _tree_sha256(path)
    if after != before:
        raise RuntimeError("TimesFM LoRA continuation changed during validation")
    if (
        _tree_sha256(pathlib.Path(source_adapter_dir)) != source_adapter_sha256
        or _adapter_model_file_sha256(pathlib.Path(source_adapter_dir))
        != source_adapter_model_sha256
    ):
        raise RuntimeError("TimesFM LoRA source changed during continuation validation")
    return LoRAAdapterArtifact(
        path=str(path),
        sha256=before,
        manifest_path=str(path / LORA_MANIFEST),
        n_training_rows=rows_identity["n_rows"],
        training_rows_sha256=rows_identity["sha256"],
        panel_content_sha256=rows_identity["panel_content_sha256"],
    )


__all__ = [
    "GOVERNED_HORIZON",
    "LORA_ARTIFACT_SCHEMA",
    "LORA_CONTINUATION_ARTIFACT_SCHEMA",
    "LORA_MANIFEST",
    "LoRAAdapterArtifact",
    "LoRAContinuationConfig",
    "LoRATrainingConfig",
    "M5LoRACutoffIdentity",
    "M5_LORA_CUTOFF_IDENTITIES",
    "M5_PANEL_CONTENT_SHA256",
    "M5_SALES_TRAIN_EVALUATION_SHA256",
    "lora_adapter_tree_sha256",
    "PROSPECTIVE_M5_LORA_CONFIG",
    "TRAINING_ROWS_MANIFEST",
    "TRAINING_ROWS_INDEX",
    "TRAINING_ROWS_SCHEMA",
    "RUNTIME_LOCK_CANONICALIZATION",
    "RuntimeLockIdentity",
    "TRANSFORMERS_CONFIG_SHA256",
    "TRANSFORMERS_MODEL_ID",
    "TRANSFORMERS_REVISION",
    "TRANSFORMERS_WEIGHTS_SHA256",
    "TimesFMTransformersBackend",
    "TrainingRowsArtifact",
    "continue_lora",
    "prepare_lora_training_rows",
    "train_lora",
    "validate_lora_adapter",
    "validate_lora_continuation_adapter",
    "validate_runtime_lock",
    "validate_training_rows_artifact",
    "runtime_lock_projection_sha256",
]
