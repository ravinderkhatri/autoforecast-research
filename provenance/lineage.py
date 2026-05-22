"""MLMD's four types, as plain data, so the local records ARE the graph.

WHY THIS VOCABULARY AND NOT A RUN LOG. The obvious alternative -- MLflow's experiment/run/params/
metrics -- was evaluated and rejected, and not because it is an extra hop. A run log records WHAT
happened; ML Metadata records what CONSUMED and PRODUCED what. Those are different shapes, and the
translation runs one way only: lineage a run log never captured cannot be reconstructed from it.
Adopting a run log on the way to Vertex ML Metadata means arriving with runs and no provenance.

This project is already lineage-shaped. Raw CSVs become a panel, the panel and an origin become a
feature window, a window and a model become a forecast, a forecast and actuals become a score, and
scores become a gate result that cites a manifest hash. `autoforecast/protocol.py` is a hand-rolled
version of exactly this. Writing it in MLMD's vocabulary means a Vertex backend is a WRITER rather
than a translator.

THE FOUR TYPES, and they are MLMD's, not an invention here:

  * `Artifact`  -- a thing that exists: a dataset, a model, a metrics record. Has a URI and a
    content hash.
  * `Execution` -- a thing that ran: a training job, a scoring job. Has a state and parameters.
  * `Context`   -- what groups them: an experiment, a pipeline run.
  * `Event`     -- an Execution consumed (INPUT) or produced (OUTPUT) an Artifact. **This is the
    edge**, and it is the part a run log has no place to put.

CONTENT HASHING IS THE DATA VERSIONING. A run currently records which config it used and not which
data it saw, and this repository has already been bitten once by a silent data change -- a builder
that appended instead of replacing, duplicating every row, caught only because a later schema
change made two generations disagree. For a static benchmark, a content hash recorded with the run
IS data versioning; DVC and lakeFS earn their keep when data changes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import shutil
import stat
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Artifact",
    "Context",
    "Event",
    "Execution",
    "Graph",
    "LineageError",
    "directory_content_digest",
    "directory_digest",
    "parquet_files",
    "snapshot_parquet_directory",
]

EventType = Literal["INPUT", "OUTPUT"]
State = Literal["RUNNING", "COMPLETE", "FAILED", "CACHED"]

DIGEST_CHUNK = 1 << 20
"""Bytes per read when hashing. A panel is gigabytes; reading it whole to hash it would use more
memory than the job it is describing."""

_DIRECTORY_DIGEST_DOMAIN = b"forecast-directory-digest-v2\0"
_DIRECTORY_CONTENT_DIGEST_DOMAIN = b"forecast-directory-content-digest-v1\0"


class LineageError(ValueError):
    """A lineage graph was described in a way that cannot be true."""


def _require_nonblank_string(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise LineageError(f"{label} must be a non-blank string")


def _values_equal_exact(observed: object, expected: object) -> bool:
    """Compare nested records without bool/int/float or list/tuple aliases."""
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict) and isinstance(observed, dict):
        return observed.keys() == expected.keys() and all(
            _values_equal_exact(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list | tuple) and isinstance(observed, list | tuple):
        return len(observed) == len(expected) and all(
            _values_equal_exact(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _digest_files(
    path: str | pathlib.Path, suffixes: tuple[str, ...]
) -> list[tuple[str, pathlib.Path]]:
    root = pathlib.Path(path)
    lexical = root if root.is_absolute() else pathlib.Path.cwd() / root
    for ancestor in (lexical, *lexical.parents):
        if ancestor.is_symlink():
            # Darwin exposes these stable OS-owned aliases for all temporary paths.
            if ancestor.as_posix() in {"/var", "/tmp"} and ancestor.resolve().as_posix() in {
                "/private/var",
                "/private/tmp",
            }:
                continue
            raise LineageError(
                f"{root} traverses symlink {ancestor}; linked bytes are not a stable "
                "artifact identity"
            )
    if root.is_symlink():
        raise LineageError(f"{root} is a symlink; linked bytes are not a stable artifact identity")
    if not root.exists():
        raise LineageError(
            f"{root} does not exist; a digest of nothing compares equal to every other nothing"
        )
    if not root.is_file() and not root.is_dir():
        raise LineageError(f"{root} is not a regular file or directory")
    entries = [] if root.is_file() else list(root.rglob("*"))
    linked = next((entry for entry in entries if entry.is_symlink()), None)
    if linked is not None:
        raise LineageError(
            f"{root} contains symlink {linked.relative_to(root)}; linked bytes are not stable"
        )
    for entry in entries:
        mode = entry.stat().st_mode
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise LineageError(f"{root} contains special file {entry.relative_to(root)}")
    parquet_only = suffixes == (".parquet",)
    if parquet_only:
        for entry in (root, *entries):
            if entry.is_file() and entry.stat().st_nlink != 1:
                raise LineageError(f"{entry} is hard-linked; Parquet artifacts require owned files")
        for entry in entries:
            if entry.name == "_spark_metadata":
                raise LineageError(f"{root} contains unsupported streaming metadata {entry}")
    if root.is_file():
        files = [] if suffixes and root.suffix not in suffixes else [(root.name, root)]
    else:
        files = sorted(
            (
                (p.relative_to(root).as_posix(), p)
                for p in entries
                if p.is_file()
                and (
                    _parquet_member(p.relative_to(root))
                    if parquet_only
                    else not suffixes or p.suffix in suffixes
                )
            ),
            key=lambda item: item[0],
        )
    if not files:
        raise LineageError(f"{root} contains no files matching {suffixes or 'any suffix'}")
    return files


def _parquet_member(relative: pathlib.Path) -> bool:
    """Require every file visible to either directory reader to be a Parquet part.

    Both readers ignore dot names and ordinary underscore markers. Spark also sees
    underscore partition names containing '=' and Parquet summary metadata prefixes.
    Those exceptions must not hide unauthenticated bytes behind an ignored suffix.
    """
    for name in relative.parts:
        if name.startswith(".") or (
            name.startswith("_")
            and "=" not in name
            and not name.startswith(("_metadata", "_common_metadata"))
        ):
            return False
    if relative.suffix != ".parquet":
        raise LineageError(
            f"reader-visible file {relative} is not a .parquet part; "
            "keep sidecars and summary metadata outside the dataset directory"
        )
    return True


def parquet_files(path: str | pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return the validated Parquet inventory used by both directory digest functions."""
    return tuple(member for _relative, member in _digest_files(path, (".parquet",)))


def directory_digest(path: str | pathlib.Path, *, suffixes: tuple[str, ...] = ()) -> str:
    """Hash relative filenames and contents for a stable directory artifact identity.

    Modification times and ownership are ignored. A size change during the read is rejected, but
    callers can digest again after consuming it to detect persistent same-size rewrites.
    Use an authenticated private snapshot when consumption must use one fixed byte version.
    A ``('.parquet',)`` filter also validates the complete directory inventory:
    reader-visible files must have that exact suffix, hidden markers are ignored, and
    links, special files, and streaming metadata are refused. Other filters select suffixes.
    """
    files = _digest_files(path, suffixes)
    digest = hashlib.sha256()
    digest.update(_DIRECTORY_DIGEST_DOMAIN)
    digest.update(len(files).to_bytes(8, "big"))
    for rel, f in files:
        relative_name = rel.encode("utf-8")
        digest.update(b"file\0")
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        with open(f, "rb") as handle:
            expected_size = os.fstat(handle.fileno()).st_size
            digest.update(expected_size.to_bytes(8, "big"))
            observed_size = 0
            while chunk := handle.read(DIGEST_CHUNK):
                digest.update(chunk)
                observed_size += len(chunk)
        if observed_size != expected_size:
            raise LineageError(f"{f} changed size while its digest was being computed")
    return digest.hexdigest()


def directory_content_digest(path: str | pathlib.Path, *, suffixes: tuple[str, ...] = ()) -> str:
    """Hash the multiset of file bytes without names, detecting byte-identical renamed copies."""
    files = _digest_files(path, suffixes)
    payloads: list[tuple[int, bytes]] = []
    for _relative, file_path in files:
        file_digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            expected_size = os.fstat(handle.fileno()).st_size
            observed_size = 0
            while chunk := handle.read(DIGEST_CHUNK):
                file_digest.update(chunk)
                observed_size += len(chunk)
        if observed_size != expected_size:
            raise LineageError(
                f"{file_path} changed size while its content digest was being computed"
            )
        payloads.append((expected_size, file_digest.digest()))
    digest = hashlib.sha256()
    digest.update(_DIRECTORY_CONTENT_DIGEST_DOMAIN)
    digest.update(len(payloads).to_bytes(8, "big"))
    for size, payload_digest in sorted(payloads):
        digest.update(size.to_bytes(8, "big"))
        digest.update(payload_digest)
    return digest.hexdigest()


@dataclass(frozen=True)
class Artifact:
    """Something that exists and can be pointed at.

    Attributes:
        id: Stable identifier within the graph.
        type_name: MLMD artifact type -- `Dataset`, `Model`, `Metrics`, `Forecast`.
        uri: Where it lives. A local path or an object-store prefix.
        content_hash: Digest of the bytes, or None when hashing was not possible or not wanted.
            **None is not a failure**, but it does mean this artefact cannot be shown to be the
            same one a later run read.
        properties: Anything else worth recording. Kept flat because MLMD's properties are.
    """

    id: str
    type_name: str
    uri: str
    content_hash: str | None = None
    properties: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Raises: LineageError: no id or no type."""
        _require_nonblank_string(self.id, "artifact id")
        _require_nonblank_string(self.type_name, "artifact type_name")
        _require_nonblank_string(self.uri, "artifact uri")
        if self.content_hash is not None and (
            type(self.content_hash) is not str or not self.content_hash.strip()
        ):
            raise LineageError("artifact content_hash must be None or a non-empty string")
        if type(self.properties) is not dict:
            raise LineageError("artifact properties must be a mapping")


@dataclass(frozen=True)
class Execution:
    """Something that ran.

    Attributes:
        id: Stable identifier within the graph.
        type_name: MLMD execution type -- `Train`, `Score`, `Build`.
        state: Where it got to. `FAILED` executions are recorded, not omitted -- a graph that
            holds only successes cannot answer why a result is missing.
        parameters: The configuration it ran under.
    """

    id: str
    type_name: str
    state: State = "COMPLETE"
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Raises: LineageError: no id, no type, or an unknown state."""
        _require_nonblank_string(self.id, "execution id")
        _require_nonblank_string(self.type_name, "execution type_name")
        if self.state not in ("RUNNING", "COMPLETE", "FAILED", "CACHED"):
            raise LineageError(f"unknown execution state {self.state!r}")
        if type(self.parameters) is not dict:
            raise LineageError("execution parameters must be a mapping")


@dataclass(frozen=True)
class Context:
    """What groups executions and artifacts -- an experiment, a pipeline run."""

    id: str
    type_name: str
    properties: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Raises: LineageError: no id or no type."""
        _require_nonblank_string(self.id, "context id")
        _require_nonblank_string(self.type_name, "context type_name")
        if type(self.properties) is not dict:
            raise LineageError("context properties must be a mapping")


@dataclass(frozen=True)
class Event:
    """An execution consumed or produced an artifact. **The edge.**

    Attributes:
        execution_id: Which execution.
        artifact_id: Which artifact.
        type: `INPUT` if consumed, `OUTPUT` if produced.
    """

    execution_id: str
    artifact_id: str
    type: EventType

    def __post_init__(self) -> None:
        """Raises: LineageError: an unknown event type."""
        _require_nonblank_string(self.execution_id, "event execution_id")
        _require_nonblank_string(self.artifact_id, "event artifact_id")
        if self.type not in ("INPUT", "OUTPUT"):
            raise LineageError(f"event type must be INPUT or OUTPUT, got {self.type!r}")


@dataclass
class Graph:
    """One run's lineage: its artifacts, executions, contexts and the edges between them.

    Backend-independent by construction. `json_backend` writes it beside the run and
    `vertex_backend` writes the same object to Vertex ML Metadata, so moving to the cloud does not
    change what is recorded -- only where.
    """

    context: Context
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    executions: dict[str, Execution] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.context, Context):
            raise LineageError("a graph context must be a Context record")

    def add_artifact(self, artifact: Artifact) -> Artifact:
        """Register an artifact.

        Raises:
            LineageError: the id is already taken by a DIFFERENT artifact. Re-registering the
                same one is fine and happens whenever a step reads what a previous step wrote;
                silently overwriting with a different uri or hash would rewrite history.
        """
        if not isinstance(artifact, Artifact):
            raise LineageError("graph artifacts must be Artifact records")
        existing = self.artifacts.get(artifact.id)
        if existing is not None and not _values_equal_exact(
            dataclasses.asdict(existing), dataclasses.asdict(artifact)
        ):
            raise LineageError(
                f"artifact {artifact.id!r} is already registered with a different uri or hash "
                f"({existing.uri} vs {artifact.uri}); overwriting it would rewrite history"
            )
        self.artifacts[artifact.id] = artifact
        return artifact

    def add_execution(self, execution: Execution) -> Execution:
        """Register or update an execution from ``RUNNING`` to one immutable terminal state."""
        if not isinstance(execution, Execution):
            raise LineageError("graph executions must be Execution records")
        existing = self.executions.get(execution.id)
        if existing is not None:
            if existing.type_name != execution.type_name:
                raise LineageError(
                    f"execution {execution.id!r} is already registered with a different type; "
                    "a state update cannot rewrite what ran"
                )
            if not execution.parameters:
                execution = dataclasses.replace(execution, parameters=existing.parameters)
            elif not _values_equal_exact(existing.parameters, execution.parameters):
                raise LineageError(
                    f"execution {execution.id!r} is already registered with different "
                    "parameters; a state update cannot rewrite what ran"
                )
            if existing.state != execution.state and existing.state != "RUNNING":
                raise LineageError(
                    f"execution {execution.id!r} is already terminal in state "
                    f"{existing.state}; a later update cannot rewrite it as {execution.state}"
                )
        self.executions[execution.id] = execution
        return execution

    def consumed(self, execution_id: str, artifact_id: str) -> Event:
        """Record that an execution read an artifact."""
        return self._edge(execution_id, artifact_id, "INPUT")

    def produced(self, execution_id: str, artifact_id: str) -> Event:
        """Record that an execution wrote an artifact."""
        return self._edge(execution_id, artifact_id, "OUTPUT")

    def _edge(self, execution_id: str, artifact_id: str, kind: EventType) -> Event:
        """Raises: LineageError: either end is unregistered -- a dangling edge describes a
        relationship between things the graph cannot name."""
        if execution_id not in self.executions:
            raise LineageError(f"no execution {execution_id!r}; register it before its edges")
        if artifact_id not in self.artifacts:
            raise LineageError(f"no artifact {artifact_id!r}; register it before its edges")
        event = Event(execution_id=execution_id, artifact_id=artifact_id, type=kind)
        if event not in self.events:
            self.events.append(event)
        return event

    def inputs_of(self, execution_id: str) -> list[Artifact]:
        """What an execution read. The question a run log cannot answer."""
        return [
            self.artifacts[e.artifact_id]
            for e in self.events
            if e.execution_id == execution_id and e.type == "INPUT"
        ]

    def outputs_of(self, execution_id: str) -> list[Artifact]:
        """What an execution wrote."""
        return [
            self.artifacts[e.artifact_id]
            for e in self.events
            if e.execution_id == execution_id and e.type == "OUTPUT"
        ]

    def producers_of(self, artifact_id: str) -> list[Execution]:
        """Which executions claim to have written this. **More than one is a finding**, not an
        error to raise here -- it means two runs wrote the same uri, and the graph should be able
        to say so rather than refuse to record it."""
        return [
            self.executions[e.execution_id]
            for e in self.events
            if e.artifact_id == artifact_id and e.type == "OUTPUT"
        ]

    def as_record(self) -> dict[str, object]:
        """A JSON-safe record of the whole graph."""
        return {
            "context": dataclasses.asdict(self.context),
            "artifacts": [dataclasses.asdict(a) for a in self.artifacts.values()],
            "executions": [dataclasses.asdict(e) for e in self.executions.values()],
            "events": [dataclasses.asdict(e) for e in self.events],
        }


def snapshot_parquet_directory(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    expected_sha256: str,
    label: str,
) -> pathlib.Path:
    """Copy authenticated Parquet bytes into an operation-owned input boundary."""
    if destination.exists():
        raise LineageError(f"{label} snapshot destination already exists")
    if directory_digest(source, suffixes=(".parquet",)) != expected_sha256:
        raise LineageError(f"{label} changed before its snapshot")
    parts = parquet_files(source)
    destination.mkdir()
    try:
        for part in parts:
            relative = pathlib.Path(part.name) if source.is_file() else part.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(part, target, follow_symlinks=False)
        if (directory_digest(destination, suffixes=(".parquet",)) != expected_sha256
                or directory_digest(source, suffixes=(".parquet",)) != expected_sha256):
            raise LineageError(f"{label} changed while its snapshot was copied")
        return destination
    except BaseException:
        shutil.rmtree(destination)
        raise
