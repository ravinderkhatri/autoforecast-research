"""Write a lineage graph beside the run, and read it back. **The source of truth.**

WHY LOCAL JSON OWNS AND A SERVICE MIRRORS. Gate records in this project are claim-bearing: a
result cites a manifest hash, and the chain from raw data to that claim has to be reconstructible
from the repository alone. A tracking service is a dependency that can be unavailable, reset,
re-projected, or quietly diverge from what actually ran -- and none of those failures announce
themselves. So Vertex ML Metadata receives a copy; this file holds the original.

The practical consequence is a rule, not a preference: **a run must be able to complete with no
network.** `vertex_backend` failing is a warning; this failing is the run failing.

FORMAT. One JSON document per graph, not a directory of files. The graph is small -- tens of nodes
-- and a single document cannot half-exist. A directory can, and a half-written lineage graph is
worse than none because it looks complete.

WHAT ROUND-TRIPPING IS FOR, and it is not tidiness. `read` is what a later run uses to ask what
produced the artefact it is about to consume. A graph that can be written and not read is a log.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import tempfile

from provenance.lineage import Artifact, Context, Event, Execution, Graph
from provenance.storage import Location

__all__ = ["FILENAME", "JsonBackendError", "read", "write"]

FILENAME = "lineage.json"
"""Fixed name, so a reader never has to be told where the graph is -- only which run directory."""

SCHEMA = 1
"""Version of the document layout.

Recorded so a reader can refuse a format it does not understand rather than silently missing
fields it never looks for. An unrecognised version is the one failure mode a lineage store must
not paper over: the whole point is that the record means what it says.
"""


class JsonBackendError(ValueError):
    """A lineage document could not be written or trusted."""


_DOCUMENT_FIELDS = {"schema", "context", "artifacts", "executions", "events"}
_CONTEXT_FIELDS = {"id", "type_name", "properties"}
_ARTIFACT_FIELDS = {"id", "type_name", "uri", "content_hash", "properties"}
_EXECUTION_FIELDS = {"id", "type_name", "state", "parameters"}
_EVENT_FIELDS = {"execution_id", "artifact_id", "type"}


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object, refusing repeated keys before the parser can collapse them."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise JsonBackendError(f"lineage JSON object repeats key {key!r}")
        result[key] = value
    return result


def _validate_canonical_json(value: object, path: str = "$") -> None:
    """Require the finite, JSON-native value domain used by the local source of truth."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise JsonBackendError(f"lineage value {path} is non-finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_canonical_json(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise JsonBackendError(
                    f"lineage object key {path}[{key!r}] is not a string")
            _validate_canonical_json(item, f"{path}.{key}")
        return
    raise JsonBackendError(
        f"lineage value {path} is not canonical JSON: {value!r}")


def _require_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise JsonBackendError(f"lineage {label} has the wrong exact field set")
    return value


def _require_nonblank_string(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise JsonBackendError(f"lineage {label} must be a non-blank string")


def _validate_document(document: object) -> dict[str, object]:
    """Validate the exact v1 shape before either serialization or graph construction."""
    document = _require_object(document, _DOCUMENT_FIELDS, "document")
    if type(document.get("schema")) is not int or document["schema"] != SCHEMA:
        raise JsonBackendError(
            f"lineage has schema {document.get('schema')!r}, this reader understands {SCHEMA}. "
            "Refusing rather than parsing hopefully: a partly understood lineage record "
            "silently drops whatever it did not recognise, and still looks complete")

    context = _require_object(document.get("context"), _CONTEXT_FIELDS, "context")
    _require_nonblank_string(context.get("id"), "context.id")
    _require_nonblank_string(context.get("type_name"), "context.type_name")
    if type(context.get("properties")) is not dict:
        raise JsonBackendError("lineage context.properties must be an object")

    collections = {
        "artifacts": (_ARTIFACT_FIELDS, ("id", "type_name", "uri")),
        "executions": (_EXECUTION_FIELDS, ("id", "type_name", "state")),
        "events": (_EVENT_FIELDS, ("execution_id", "artifact_id", "type")),
    }
    for label, (fields, string_fields) in collections.items():
        records = document.get(label)
        if type(records) is not list:
            raise JsonBackendError(f"lineage {label} must be a list")
        for index, raw in enumerate(records):
            record = _require_object(raw, fields, f"{label}[{index}]")
            for field in string_fields:
                _require_nonblank_string(record.get(field), f"{label}[{index}].{field}")
            if label == "artifacts":
                content_hash = record.get("content_hash")
                if content_hash is not None \
                        and (type(content_hash) is not str or not content_hash.strip()):
                    raise JsonBackendError(
                        f"lineage artifacts[{index}].content_hash is malformed")
                if type(record.get("properties")) is not dict:
                    raise JsonBackendError(
                        f"lineage artifacts[{index}].properties must be an object")
            elif label == "executions" and type(record.get("parameters")) is not dict:
                raise JsonBackendError(
                    f"lineage executions[{index}].parameters must be an object")

    for label in ("artifacts", "executions"):
        values = [record["id"] for record in document[label]]
        if len(values) != len(set(values)):
            raise JsonBackendError(f"lineage {label} contain duplicate identities")
    event_keys = [
        (record["execution_id"], record["artifact_id"], record["type"])
        for record in document["events"]
    ]
    if len(event_keys) != len(set(event_keys)):
        raise JsonBackendError("lineage events contain duplicate edges")

    _validate_canonical_json(document)
    return document


def write(graph: Graph, into: Location) -> Location:
    """Write `graph` into `into/lineage.json`. Returns the file's location.

    Args:
        graph: What to record.
        into: The run directory. Created if local.

    Returns:
        Where it was written.

    Raises:
        JsonBackendError: the location is remote. This backend deliberately has no cloud client --
            see the module docstring on staying dependency-free. A remote run writes locally and
            uploads, or uses `vertex_backend`.
    """
    if into.is_remote:
        raise JsonBackendError(
            f"{into} is remote and this backend has no {into.scheme} client. Write locally and "
            "upload, or use vertex_backend -- adding a cloud dependency here would make the "
            "source of truth depend on the network")
    target = into.ensure_dir().child(FILENAME)
    target_path = target.as_path()
    document = {"schema": SCHEMA, **graph.as_record()}
    _validate_document(document)
    try:
        payload = json.dumps(document, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise JsonBackendError(f"lineage graph is not canonical finite JSON: {exc}") from exc
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, target_path)
        directory_fd = os.open(target_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def read(source: Location) -> Graph:
    """Read a graph back from a run directory or a `lineage.json` path.

    This is what a later run uses to ask what produced the artefact it is about to consume, which
    is the question the whole vocabulary exists for.

    Args:
        source: A run directory containing `lineage.json`, or the file itself.

    Returns:
        The `Graph`.

    Raises:
        JsonBackendError: nothing is there, the document is not JSON, or its schema is one this
            reader does not understand -- refused rather than parsed hopefully, because a partly
            understood lineage record is one that quietly drops the fields it did not recognise.
        LineageError: the document describes a graph that cannot be true -- a dangling edge, an
            unknown state. Raised by the graph itself, deliberately not caught here: a file that
            round-trips into an invalid graph is corrupt, and saying so is the point.
    """
    if source.is_remote:
        raise JsonBackendError(f"{source} is remote; this backend reads local files only")
    path = source.as_path()
    if path.is_dir():
        path = path / FILENAME
    if not path.exists():
        raise JsonBackendError(f"no lineage document at {path}")
    try:
        raw = path.read_text()
    except (OSError, UnicodeError) as exc:
        raise JsonBackendError(f"cannot read lineage document at {path}: {exc}") from exc
    try:
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise JsonBackendError(f"{path} is not valid JSON: {exc}") from exc
    except JsonBackendError as exc:
        raise JsonBackendError(f"{path}: {exc}") from exc

    try:
        document = _validate_document(document)
    except JsonBackendError as exc:
        raise JsonBackendError(f"{path}: {exc}") from exc

    graph = Graph(context=Context(**document["context"]))
    for record in document.get("artifacts", []):
        graph.add_artifact(Artifact(**record))
    for record in document.get("executions", []):
        graph.add_execution(Execution(**record))
    for record in document.get("events", []):
        event = Event(**record)
        # Routed through the graph's own edge check, so a document naming an artifact it never
        # declared is rejected on read rather than becoming a reference to nothing.
        graph._edge(event.execution_id, event.artifact_id, event.type)
    return graph


def describe(source: Location) -> str:
    """A one-line summary of a stored graph, for a log line.

    Raises:
        JsonBackendError: as `read`.
    """
    graph = read(source)
    inputs = sum(1 for e in graph.events if e.type == "INPUT")
    outputs = len(graph.events) - inputs
    return (f"{graph.context.type_name} {graph.context.id}: {len(graph.executions)} execution(s), "
            f"{len(graph.artifacts)} artifact(s), {inputs} consumed, {outputs} produced")
