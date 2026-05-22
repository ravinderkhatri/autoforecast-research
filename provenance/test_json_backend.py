"""Tests for the local lineage store, which is the SOURCE OF TRUTH.

Shaped around one property: a graph that can be written and not read back exactly is a log, not a
lineage record. So the round trip is tested on edges and identity, not just on field presence.

Run: python3 provenance/test_json_backend.py
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from provenance import json_backend
from provenance.json_backend import JsonBackendError, describe, read, write
from provenance.lineage import Artifact, Context, Execution, Graph, LineageError
from provenance.storage import Location


def graph() -> Graph:
    g = Graph(context=Context(id="ladder", type_name="Experiment",
                              properties={"dataset": "m5"}))
    g.add_artifact(Artifact(id="panel", type_name="Dataset", uri="/data/panel",
                            content_hash="abc123", properties={"rows": 59181090}))
    g.add_artifact(Artifact(id="fc", type_name="Forecast", uri="/out/fc"))
    g.add_artifact(Artifact(id="metrics", type_name="Metrics", uri="/out/result.json"))
    g.add_execution(Execution(id="rung3", type_name="Train",
                              parameters={"encoder_days": 364, "train_stride": 364}))
    g.consumed("rung3", "panel")
    g.produced("rung3", "fc")
    g.produced("rung3", "metrics")
    return g


def here() -> Location:
    return Location(tempfile.mkdtemp())


def replacement_graph() -> Graph:
    replacement = graph()
    replacement.context = Context(id="ladder-next", type_name="Experiment",
                                  properties={"dataset": "m5"})
    return replacement


def temporary_lineage_files(at: Location) -> list[pathlib.Path]:
    return list(at.as_path().glob(f".{json_backend.FILENAME}.*.tmp"))


def test_a_graph_survives_the_round_trip_including_its_edges() -> None:
    """Field presence is not enough. A store that keeps the nodes and loses the edges has kept a
    run log and thrown away the lineage, which is the whole reason for the vocabulary."""
    original, at = graph(), here()
    write(original, at)
    back = read(at)
    assert back.context == original.context
    assert back.artifacts == original.artifacts
    assert back.executions == original.executions
    assert sorted(back.events, key=str) == sorted(original.events, key=str)
    assert [a.id for a in back.inputs_of("rung3")] == ["panel"]
    assert sorted(a.id for a in back.outputs_of("rung3")) == ["fc", "metrics"]


def test_the_content_hash_survives_because_it_is_the_data_version() -> None:
    """A run records which config it used; the hash is the only thing recording which DATA it
    saw. Losing it in the round trip loses the data versioning entirely."""
    at = here()
    write(graph(), at)
    assert read(at).artifacts["panel"].content_hash == "abc123"
    assert read(at).artifacts["panel"].properties["rows"] == 59181090


def test_write_flushes_and_syncs_a_sibling_temp_before_replace_then_syncs_the_directory() -> None:
    """The ordering is the durability contract: complete bytes reach the sibling temporary file
    before it can replace the last valid lineage, and the directory entry is synced afterwards."""
    at = here()
    write(graph(), at)
    target = at.as_path() / json_backend.FILENAME
    original = target.read_bytes()
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            events.append("file_fsync")
            temporary = temporary_lineage_files(at)
            assert len(temporary) == 1
            assert json.loads(temporary[0].read_text())["context"]["id"] == "ladder-next"
        elif stat.S_ISDIR(mode):
            events.append("directory_fsync")
            assert read(at).context.id == "ladder-next"
            assert temporary_lineage_files(at) == []
        else:
            raise AssertionError(f"fsync called on unexpected mode {mode:o}")
        real_fsync(fd)

    def recording_replace(source: str | pathlib.Path, destination: str | pathlib.Path) -> None:
        source_path = pathlib.Path(source)
        destination_path = pathlib.Path(destination)
        events.append("replace")
        assert source_path.parent == destination_path.parent == at.as_path()
        assert target.read_bytes() == original
        assert json.loads(source_path.read_text())["context"]["id"] == "ladder-next"
        real_replace(source, destination)

    with (
        mock.patch.object(json_backend.os, "fsync", recording_fsync),
        mock.patch.object(json_backend.os, "replace", recording_replace),
    ):
        write(replacement_graph(), at)

    assert events == ["file_fsync", "replace", "directory_fsync"]


def test_file_sync_failure_preserves_existing_lineage_and_cleans_the_temporary_file() -> None:
    at = here()
    write(graph(), at)
    target = at.as_path() / json_backend.FILENAME
    original = target.read_bytes()

    with mock.patch.object(json_backend.os, "fsync", side_effect=OSError("sync failed")):
        try:
            write(replacement_graph(), at)
        except OSError as error:
            assert "sync failed" in str(error)
        else:
            raise AssertionError("reported success after the temporary file could not be synced")

    assert target.read_bytes() == original
    assert read(at).context.id == "ladder"
    assert temporary_lineage_files(at) == []


def test_replace_failure_preserves_existing_lineage_and_cleans_the_temporary_file() -> None:
    at = here()
    write(graph(), at)
    target = at.as_path() / json_backend.FILENAME
    original = target.read_bytes()

    with mock.patch.object(json_backend.os, "replace", side_effect=OSError("replace failed")):
        try:
            write(replacement_graph(), at)
        except OSError as error:
            assert "replace failed" in str(error)
        else:
            raise AssertionError("reported success after the atomic replacement failed")

    assert target.read_bytes() == original
    assert read(at).context.id == "ladder"
    assert temporary_lineage_files(at) == []


def test_reading_accepts_the_directory_or_the_file() -> None:
    at = here()
    written = write(graph(), at)
    assert read(at).context.id == read(written).context.id == "ladder"


def test_a_missing_document_is_refused() -> None:
    try:
        read(here())
    except JsonBackendError as e:
        assert "no lineage document" in str(e)
        return
    raise AssertionError("read a graph from an empty directory")


def test_an_unreadable_document_is_refused() -> None:
    at = here()
    (at.as_path() / json_backend.FILENAME).write_text("{not json")
    try:
        read(at)
    except JsonBackendError as e:
        assert "not valid JSON" in str(e)
        return
    raise AssertionError("parsed a corrupt document")


def test_invalid_utf8_is_reported_as_an_unreadable_lineage_document() -> None:
    at = here()
    (at.as_path() / json_backend.FILENAME).write_bytes(b"\xff")
    try:
        read(at)
    except JsonBackendError as error:
        assert "cannot read lineage document" in str(error)
        return
    raise AssertionError("invalid UTF-8 escaped the lineage error contract")


def test_an_unknown_schema_is_refused_rather_than_parsed_hopefully() -> None:
    """A partly understood lineage record silently drops whatever it did not recognise, and still
    looks complete. That is the one failure a provenance store must not paper over."""
    at = here()
    write(graph(), at)
    path = at.as_path() / json_backend.FILENAME
    doc = json.loads(path.read_text())
    doc["schema"] = 99
    path.write_text(json.dumps(doc))
    try:
        read(at)
    except JsonBackendError as e:
        assert "parsing hopefully" in str(e)
        return
    raise AssertionError("read a document written by a future version")


def test_schema_version_document_shape_and_record_identities_are_type_exact() -> None:
    for mutation in (
        lambda doc: doc.__setitem__("schema", True),
        lambda doc: doc.__setitem__("schema", 1.0),
        lambda doc: doc.__setitem__("future_field", "silently ignored"),
        lambda doc: doc["context"].__setitem__("future_field", "silently ignored"),
        lambda doc: doc["artifacts"].append(dict(doc["artifacts"][0])),
        lambda doc: doc["executions"].append(dict(doc["executions"][0])),
        lambda doc: doc["events"].append(dict(doc["events"][0])),
    ):
        at = here()
        write(graph(), at)
        path = at.as_path() / json_backend.FILENAME
        doc = json.loads(path.read_text())
        mutation(doc)
        path.write_text(json.dumps(doc))
        try:
            read(at)
        except JsonBackendError:
            continue
        raise AssertionError(f"lineage reader accepted a non-v1 document: {doc!r}")


def test_reader_refuses_duplicate_object_keys_before_json_can_collapse_them() -> None:
    mutations = (
        ('"schema": 1', '"schema": 99,\n  "schema": 1'),
        ('"id": "ladder"', '"id": "wrong",\n    "id": "ladder"'),
    )
    for needle, replacement in mutations:
        at = here()
        write(graph(), at)
        path = at.as_path() / json_backend.FILENAME
        original = path.read_text()
        assert original.count(needle) == 1
        path.write_text(original.replace(needle, replacement, 1))
        try:
            read(at)
        except JsonBackendError as exc:
            assert "repeats key" in str(exc)
            continue
        raise AssertionError(f"lineage reader collapsed duplicate key {needle!r}")


def test_write_refuses_non_json_values_instead_of_stringifying_them() -> None:
    for value in (object(), (1, 2), float("nan"), float("inf")):
        malformed = graph()
        malformed.context = Context(
            id="ladder", type_name="Experiment", properties={"opaque": value})
        try:
            write(malformed, here())
        except JsonBackendError:
            continue
        raise AssertionError(f"lineage write changed non-canonical value {value!r}")


def test_json_list_properties_round_trip_and_reregister_exactly() -> None:
    original = graph()
    original.artifacts["panel"] = Artifact(
        id="panel", type_name="Dataset", uri="/data/panel", content_hash="abc123",
        properties={"shape": [30_490, 28]})
    at = here()
    write(original, at)
    back = read(at)
    same = Artifact(
        id="panel", type_name="Dataset", uri="/data/panel", content_hash="abc123",
        properties={"shape": [30_490, 28]})
    back.add_artifact(same)
    assert back.artifacts["panel"] is same


def test_v1_document_remains_readable_without_the_current_writer() -> None:
    # A literal v1 fixture catches format drift independently of write().
    path = here().as_path() / json_backend.FILENAME
    path.write_text(json.dumps({
        "schema": 1,
        "context": {"id": "example-v1", "type_name": "Experiment", "properties": {}},
        "artifacts": [
            {"id": name, "type_name": kind, "uri": f"/example/{name}",
             "content_hash": None, "properties": {}}
            for name, kind in (("panel", "Dataset"), ("forecast", "Forecast"), ("score", "Metrics"))
        ],
        "executions": [{"id": "fit", "type_name": "Train", "state": "COMPLETE",
                        "parameters": {"horizon": 7}}],
        "events": [
            {"execution_id": "fit", "artifact_id": name, "type": direction}
            for name, direction in (("panel", "INPUT"), ("forecast", "OUTPUT"), ("score", "OUTPUT"))
        ],
    }))
    back = read(Location(str(path)))
    assert back.context.id == "example-v1"
    assert len(back.artifacts) == 3
    assert len(back.executions) == 1
    assert len(back.events) == 3
    assert [artifact.id for artifact in back.inputs_of("fit")] == ["panel"]
    assert sorted(artifact.id for artifact in back.outputs_of("fit")) == ["forecast", "score"]
    assert back.executions["fit"].parameters == {"horizon": 7}


def test_a_document_naming_an_undeclared_artifact_is_rejected_on_read() -> None:
    """Edges are routed back through the graph's own check, so a reference to nothing is caught
    where it is introduced rather than becoming a silent dangling pointer."""
    at = here()
    write(graph(), at)
    path = at.as_path() / json_backend.FILENAME
    doc = json.loads(path.read_text())
    doc["events"].append({"execution_id": "rung3", "artifact_id": "ghost", "type": "INPUT"})
    path.write_text(json.dumps(doc))
    try:
        read(at)
    except LineageError as e:
        assert "no artifact" in str(e)
        return
    raise AssertionError("read an edge pointing at an undeclared artifact")


def test_writing_remotely_is_refused_so_the_truth_never_needs_a_network() -> None:
    """A run must be able to complete with no network. vertex_backend failing is a warning; this
    failing is the run failing."""
    try:
        write(graph(), Location("gs://bucket/run"))
    except JsonBackendError as e:
        assert "depend on the network" in str(e)
        return
    raise AssertionError("wrote the source of truth to a bucket")


def test_reading_remotely_is_refused() -> None:
    try:
        read(Location("gs://bucket/run"))
    except JsonBackendError as e:
        assert "local files only" in str(e)
        return
    raise AssertionError("claimed to read a graph from a bucket")


def test_describe_summarises_without_the_caller_parsing_anything() -> None:
    at = here()
    write(graph(), at)
    line = describe(at)
    assert "Experiment ladder" in line and "1 execution" in line
    assert "1 consumed" in line and "2 produced" in line


def test_a_failed_execution_round_trips_as_failed() -> None:
    """A graph of only successes cannot say why a result is missing, so the state has to survive."""
    completed, at = graph(), here()
    g = Graph(
        context=completed.context,
        artifacts=dict(completed.artifacts),
        events=list(completed.events),
    )
    g.add_execution(Execution(
        id="rung3", type_name="Train", state="FAILED",
        parameters={"encoder_days": 364, "train_stride": 364}))
    write(g, at)
    assert read(at).executions["rung3"].state == "FAILED"
    assert read(at).executions["rung3"].parameters == {
        "encoder_days": 364, "train_stride": 364}


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
