"""Tests for the MLMD-shaped lineage graph.

The graph exists to answer one question a run log cannot: what did this execution CONSUME, and
what produced the thing it consumed. So the tests are shaped around edges and identity, not around
whether a metric was stored.

Run: python3 provenance/test_lineage.py
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from provenance.lineage import (
    Artifact,
    Context,
    Event,
    Execution,
    Graph,
    LineageError,
    directory_content_digest,
    directory_digest,
)


def tree(files: dict[str, bytes]) -> pathlib.Path:
    root = pathlib.Path(tempfile.mkdtemp())
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def graph() -> Graph:
    g = Graph(context=Context(id="exp", type_name="Experiment"))
    g.add_artifact(Artifact(id="panel", type_name="Dataset", uri="/data/panel"))
    g.add_artifact(Artifact(id="fc", type_name="Forecast", uri="/out/fc"))
    g.add_execution(Execution(id="train", type_name="Train"))
    g.consumed("train", "panel")
    g.produced("train", "fc")
    return g


# ---------------------------------------------------------------- content hashing


def test_the_same_bytes_hash_the_same_under_a_different_path() -> None:
    """The only property that makes a digest useful: copying a panel must not change its identity,
    or every cloud run would report reading different data from the local one."""
    a = tree({"part-0.parquet": b"abc", "sub/part-1.parquet": b"def"})
    b = tree({"part-0.parquet": b"abc", "sub/part-1.parquet": b"def"})
    assert directory_digest(a) == directory_digest(b)
    assert a != b, "the two trees are the same directory, so this proves nothing"


def test_changed_bytes_change_the_digest() -> None:
    """The failure this exists to catch: a builder that appended instead of replacing, duplicating
    every row, which went unnoticed until a schema change made two generations disagree."""
    a = tree({"part-0.parquet": b"abc"})
    b = tree({"part-0.parquet": b"abc-and-a-duplicate"})
    assert directory_digest(a) != directory_digest(b)


def test_filename_and_content_are_framed_without_ambiguity() -> None:
    """The old ``name + content`` stream represented both trees as the bytes ``abc``."""
    assert b"a" + b"bc" == b"ab" + b"c", "fixture no longer reproduces the old collision"
    assert directory_digest(tree({"a": b"bc"})) != directory_digest(tree({"ab": b"c"}))


def test_each_file_record_is_framed_without_boundary_ambiguity() -> None:
    """The old stream could not distinguish a second file from more content in the first."""
    assert (
        b"a" + b"x" + b"b" + b"y" == b"a" + b"xby"
    ), "fixture no longer reproduces the old collision"
    assert directory_digest(tree({"a": b"x", "b": b"y"})) != directory_digest(tree({"a": b"xby"}))


def test_directory_digest_uses_canonical_posix_relative_names() -> None:
    """This fixed vector fails on Windows if a platform-native backslash enters the stream."""
    relative_name = b"sub/part.parquet"
    content = b"payload"
    framed = (
        b"forecast-directory-digest-v2\0"
        + (1).to_bytes(8, "big")
        + b"file\0"
        + len(relative_name).to_bytes(8, "big")
        + relative_name
        + len(content).to_bytes(8, "big")
        + content
    )
    assert (
        directory_digest(tree({"sub/part.parquet": content})) == hashlib.sha256(framed).hexdigest()
    )


def test_a_renamed_file_changes_the_digest() -> None:
    """Names are hashed with contents. Two parquet parts with swapped names are not the same
    dataset to any reader that lists them in order."""
    a = tree({"part-0.parquet": b"x", "part-1.parquet": b"y"})
    b = tree({"part-0.parquet": b"y", "part-1.parquet": b"x"})
    assert directory_digest(a) != directory_digest(b)


def test_content_digest_detects_a_byte_identical_copy_with_renamed_files() -> None:
    original = tree({"part-0.parquet": b"x", "part-1.parquet": b"y"})
    renamed = tree({"a.parquet": b"y", "b.parquet": b"x"})
    assert directory_digest(original) != directory_digest(renamed)
    assert directory_content_digest(original) == directory_content_digest(renamed)


def test_suffix_filtering_ignores_markers_that_vary_between_identical_writes() -> None:
    """Spark writes `_SUCCESS` and `.crc` sidecars that differ between writes of identical data.
    Hashing them makes the digest report a difference that is not one."""
    a = tree({"part-0.parquet": b"same", "_SUCCESS": b"", ".part-0.crc": b"aaaa"})
    b = tree({"part-0.parquet": b"same", "_SUCCESS": b"", ".part-0.crc": b"bbbb"})
    assert directory_digest(a) != directory_digest(b), "unfiltered, the sidecars should differ"
    assert directory_digest(a, suffixes=(".parquet",)) == directory_digest(
        b, suffixes=(".parquet",)
    )


def test_parquet_inventory_rejects_reader_visible_files_with_another_suffix() -> None:
    """A reader discovers Parquet bytes by content, even when a suffix filter would miss them."""
    import pytest

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for name in (
        "extra.bin", "extra", "extra.PARQUET", "extra.json", "p=1/extra",
        "part.parquet._COPYING_",
    ):
        root = tree({})
        pq.write_table(pa.table({"value": [1]}), root / "part.parquet")
        assert pq.read_table(root).num_rows == 1
        before = directory_digest(root, suffixes=(".parquet",))
        extra = root / name
        extra.parent.mkdir(exist_ok=True)
        extra.write_bytes((root / "part.parquet").read_bytes())
        assert pq.read_table(root).num_rows == 2, "fixture must demonstrate extra reader input"
        for digest in (directory_digest, directory_content_digest):
            with pytest.raises(LineageError, match="reader-visible file.*not a .parquet part"):
                digest(root, suffixes=(".parquet",))
        extra.unlink()
        assert directory_digest(root, suffixes=(".parquet",)) == before


def test_parquet_inventory_rejects_invalid_sidecars_and_reader_metadata() -> None:
    import pytest

    for name in (
        "manifest.json", "partition=a/notes.txt", "_metadata", "_common_metadata",
        "_metadata_backup", "_common_metadata.old", "_partition=a/extra.bin", "_extra=value.bin",
    ):
        root = tree({"part.parquet": b"data", name: b'{"unrelated": true}'})
        for digest in (directory_digest, directory_content_digest):
            with pytest.raises(LineageError, match="reader-visible file.*not a .parquet part"):
                digest(root, suffixes=(".parquet",))
    root = tree({"part.parquet": b"data", "_spark_metadata/0": b"external-file-list"})
    for digest in (directory_digest, directory_content_digest):
        with pytest.raises(LineageError, match="unsupported streaming metadata"):
            digest(root, suffixes=(".parquet",))


def test_parquet_inventory_ignores_hidden_marker_files_and_trees() -> None:
    import pytest

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tree({})
    pq.write_table(pa.table({"value": [1]}), root / "part.parquet")
    before = [digest(root, suffixes=(".parquet",))
              for digest in (directory_digest, directory_content_digest)]
    for name in (
        "_SUCCESS", ".part.parquet.crc", "_temporary/part.parquet", ".staging/part.parquet",
        "partition=a/_temporary/nested.bin", "partition=a/.hidden.parquet",
    ):
        marker = root / name
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes((root / "part.parquet").read_bytes())
    assert pq.read_table(root).num_rows == 1
    assert before == [digest(root, suffixes=(".parquet",))
                      for digest in (directory_digest, directory_content_digest)]


def test_parquet_inventory_authenticates_partition_names_and_contents() -> None:
    import pytest

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tree({})
    for site, value in (("a", 1), ("b", 2)):
        partition = root / f"site={site}" / "day=2026-01-01"
        partition.mkdir(parents=True)
        pq.write_table(pa.table({"value": [value]}), partition / "part.parquet")
    assert sorted(pq.read_table(root)["value"].to_pylist()) == [1, 2]
    before = directory_digest(root, suffixes=(".parquet",))
    content_before = directory_content_digest(root, suffixes=(".parquet",))
    (root / "site=a").rename(root / "site=c")
    assert directory_digest(root, suffixes=(".parquet",)) != before
    assert directory_content_digest(root, suffixes=(".parquet",)) == content_before
    pq.write_table(pa.table({"value": [3]}), root / "site=c/day=2026-01-01/part.parquet")
    assert directory_content_digest(root, suffixes=(".parquet",)) != content_before


def test_parquet_inventory_authenticates_spark_visible_underscore_names() -> None:
    for name in (
        "_partition=a/part.parquet", "_metadata_extra.parquet", "_common_metadata.parquet",
    ):
        root = tree({"part.parquet": b"data", name: b"old"})
        before = [digest(root, suffixes=(".parquet",))
                  for digest in (directory_digest, directory_content_digest)]
        (root / name).write_bytes(b"new")
        after = [digest(root, suffixes=(".parquet",))
                 for digest in (directory_digest, directory_content_digest)]
        assert all(a != b for a, b in zip(before, after, strict=True))


def test_parquet_inventory_rejects_hidden_dangling_and_directory_links() -> None:
    import pytest

    for name, target in (
        (".hidden", "missing"), ("extra.bin", "part.parquet"), ("_temporary", "."),
    ):
        root = tree({"part.parquet": b"data"})
        (root / name).symlink_to(root / target)
        for digest in (directory_digest, directory_content_digest):
            with pytest.raises(LineageError, match="contains symlink"):
                digest(root, suffixes=(".parquet",))


def test_parquet_inventory_rejects_hard_links() -> None:
    import pytest

    for name in ("extra.parquet", "_temporary/extra"):
        root = tree({"part.parquet": b"data"})
        extra = root / name
        extra.parent.mkdir(exist_ok=True)
        extra.hardlink_to(root / "part.parquet")
        for digest in (directory_digest, directory_content_digest):
            for path in (root, root / "part.parquet"):
                with pytest.raises(LineageError, match="hard-linked"):
                    digest(path, suffixes=(".parquet",))


def test_digest_inventory_rejects_special_files_without_opening_them() -> None:
    import pytest

    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable on this platform")
    for name in ("extra.parquet", ".hidden", "_temporary/pipe"):
        root = tree({"part.parquet": b"data"})
        pipe = root / name
        pipe.parent.mkdir(exist_ok=True)
        os.mkfifo(pipe)
        for digest in (directory_digest, directory_content_digest):
            with pytest.raises(LineageError, match="contains special file"):
                digest(root, suffixes=(".parquet",))
            with pytest.raises(LineageError, match="not a regular file or directory"):
                digest(pipe)


def test_non_parquet_suffix_filters_keep_selective_source_hashing() -> None:
    a = tree({"module.py": b"same", ".cache.py": b"same", "README.md": b"before"})
    b = tree({"module.py": b"same", ".cache.py": b"same", "README.md": b"after"})
    assert directory_digest(a, suffixes=(".py",)) == directory_digest(b, suffixes=(".py",))
    (b / ".cache.py").write_bytes(b"changed")
    assert directory_digest(a, suffixes=(".py",)) != directory_digest(b, suffixes=(".py",))


def test_hashing_something_absent_is_refused() -> None:
    """A digest of nothing compares equal to every other nothing, so two runs that both failed to
    find their data would look like they read the same thing."""
    try:
        directory_digest("/no/such/place")
    except LineageError as e:
        assert "compares equal to every other nothing" in str(e)
        return
    raise AssertionError("hashed a path that does not exist")


def test_a_directory_with_no_matching_files_is_refused() -> None:
    try:
        directory_digest(tree({"_SUCCESS": b""}), suffixes=(".parquet",))
    except LineageError as e:
        assert "no files matching" in str(e)
        return
    raise AssertionError("digested a directory with nothing in it")


def test_suffix_filtering_rejects_a_single_file_with_the_wrong_suffix() -> None:
    root = tree({"panel.txt": b"not parquet"})
    try:
        directory_digest(root / "panel.txt", suffixes=(".parquet",))
    except LineageError as error:
        assert "no files matching" in str(error)
        return
    raise AssertionError("digested a non-Parquet file through a Parquet-only filter")


def test_directory_digest_refuses_linked_roots_and_members() -> None:
    root = tree({"part.parquet": b"data"})
    linked_root = root.parent / f"{root.name}-linked"
    linked_root.symlink_to(root, target_is_directory=True)
    try:
        directory_digest(linked_root, suffixes=(".parquet",))
    except LineageError as error:
        assert "symlink" in str(error)
    else:
        raise AssertionError("digested a symlinked dataset root")

    member_root = tree({"part.parquet": b"data"})
    (member_root / "linked.parquet").symlink_to(member_root / "part.parquet")
    try:
        directory_digest(member_root, suffixes=(".parquet",))
    except LineageError as error:
        assert "contains symlink" in str(error)
    else:
        raise AssertionError("digested a dataset containing a symlink")


def test_directory_digest_refuses_a_symlinked_ancestor() -> None:
    parent = pathlib.Path(tempfile.mkdtemp())
    real = parent / "real"
    dataset = real / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "part.parquet").write_bytes(b"data")
    alias = parent / "alias"
    alias.symlink_to(real, target_is_directory=True)
    try:
        directory_digest(alias / "dataset", suffixes=(".parquet",))
    except LineageError as error:
        assert "traverses symlink" in str(error)
        return
    raise AssertionError("digested a dataset through a symlinked ancestor")


# ---------------------------------------------------------------- the graph


def test_the_graph_answers_what_an_execution_consumed_and_produced() -> None:
    """The question a run log has no place to put."""
    g = graph()
    assert [a.id for a in g.inputs_of("train")] == ["panel"]
    assert [a.id for a in g.outputs_of("train")] == ["fc"]
    assert [e.id for e in g.producers_of("fc")] == ["train"]
    assert g.producers_of("panel") == [], "the panel was read, not produced, by this graph"


def test_re_registering_the_same_artifact_is_fine() -> None:
    """A step reading what a previous step wrote registers it again, and that is normal."""
    g = graph()
    g.add_artifact(Artifact(id="fc", type_name="Forecast", uri="/out/fc"))
    assert len(g.artifacts) == 2


def test_re_registering_a_DIFFERENT_artifact_under_the_same_id_is_refused() -> None:
    """Silently overwriting the uri or hash would rewrite history, and the graph is the record."""
    g = graph()
    try:
        g.add_artifact(Artifact(id="fc", type_name="Forecast", uri="/somewhere/else"))
    except LineageError as e:
        assert "rewrite history" in str(e)
        return
    raise AssertionError("overwrote an artifact with a different one")


def test_artifact_reregistration_is_recursively_type_exact() -> None:
    g = Graph(context=Context(id="exp", type_name="Experiment"))
    original = Artifact(id="panel", type_name="Dataset", uri="/data/panel", properties={"rows": 1})
    g.add_artifact(original)
    try:
        g.add_artifact(
            Artifact(id="panel", type_name="Dataset", uri="/data/panel", properties={"rows": True})
        )
    except LineageError:
        assert g.artifacts["panel"] is original
        return
    raise AssertionError("bool/int property alias silently rewrote an artifact identity")


def test_a_dangling_edge_is_refused() -> None:
    """An edge to something unregistered describes a relationship between things the graph cannot
    name, and it would serialise as a valid-looking reference to nothing."""
    g = graph()
    for call, missing in (
        (lambda: g.consumed("train", "ghost"), "artifact"),
        (lambda: g.produced("ghost", "fc"), "execution"),
    ):
        try:
            call()
        except LineageError as e:
            assert missing in str(e)
        else:
            raise AssertionError(f"recorded an edge to a missing {missing}")


def test_two_executions_writing_one_uri_is_reported_not_refused() -> None:
    """More than one producer is a FINDING -- two runs wrote the same place -- and the graph
    should be able to say so rather than refuse to record it."""
    g = graph()
    g.add_execution(Execution(id="rerun", type_name="Train"))
    g.produced("rerun", "fc")
    assert sorted(e.id for e in g.producers_of("fc")) == ["rerun", "train"]


def test_a_failed_execution_is_recorded_rather_than_omitted() -> None:
    """A graph holding only successes cannot answer why a result is missing."""
    g = Graph(context=Context(id="exp", type_name="Experiment"))
    g.add_execution(Execution(id="train", type_name="Train", state="RUNNING"))
    g.add_execution(Execution(id="train", type_name="Train", state="FAILED"))
    assert g.executions["train"].state == "FAILED"
    assert len(g.executions) == 1, "the update should replace, not duplicate"


def test_execution_state_update_preserves_parameters_and_cannot_rewrite_identity() -> None:
    g = Graph(context=Context(id="exp", type_name="Experiment"))
    original = Execution(id="train", type_name="Train", state="RUNNING", parameters={"batch": 1})
    g.add_execution(original)
    updated = g.add_execution(Execution(id="train", type_name="Train", state="FAILED"))
    assert updated.state == "FAILED"
    assert updated.parameters == {"batch": 1}

    for replacement in (
        Execution(id="train", type_name="Score", parameters={"batch": 1}),
        Execution(id="train", type_name="Train", parameters={"batch": True}),
    ):
        before = g.executions["train"]
        try:
            g.add_execution(replacement)
        except LineageError:
            assert g.executions["train"] is before
            continue
        raise AssertionError("execution state update rewrote its type or parameters")


def test_execution_terminal_states_cannot_be_rewritten() -> None:
    for terminal in ("COMPLETE", "FAILED", "CACHED"):
        g = Graph(context=Context(id="exp", type_name="Experiment"))
        g.add_execution(
            Execution(id="train", type_name="Train", state="RUNNING", parameters={"seed": 0})
        )
        finished = g.add_execution(Execution(id="train", type_name="Train", state=terminal))
        assert finished.state == terminal and finished.parameters == {"seed": 0}
        assert g.add_execution(finished) is finished
        for replacement in {"COMPLETE", "FAILED", "CACHED", "RUNNING"} - {terminal}:
            try:
                g.add_execution(
                    Execution(
                        id="train", type_name="Train", state=replacement, parameters={"seed": 0}
                    )
                )
            except LineageError as error:
                assert "already terminal" in str(error)
                assert g.executions["train"] is finished
                continue
            raise AssertionError(f"execution state {terminal} was rewritten as {replacement}")


def test_an_unknown_state_or_event_type_is_refused() -> None:
    try:
        Execution(id="x", type_name="Train", state="NEARLY")
    except LineageError as e:
        assert "unknown execution state" in str(e)
    else:
        raise AssertionError("accepted an unknown state")
    try:
        graph()._edge("train", "panel", "SIDEWAYS")
    except LineageError as e:
        assert "INPUT or OUTPUT" in str(e)
        return
    raise AssertionError("accepted an unknown event type")


def test_an_artifact_or_execution_without_a_type_is_refused() -> None:
    for build in (
        lambda: Artifact(id="a", type_name="", uri="/x"),
        lambda: Execution(id="", type_name="Train"),
        lambda: Context(id="c", type_name=""),
    ):
        try:
            build()
        except LineageError:
            continue
        raise AssertionError("accepted an untyped or unnamed node")


def test_lineage_node_and_edge_identifiers_are_exact_strings() -> None:
    malformed = (
        lambda: Artifact(id=1, type_name="Dataset", uri="/x"),
        lambda: Artifact(id="a", type_name="Dataset", uri=1),
        lambda: Execution(id=1, type_name="Train"),
        lambda: Context(id="c", type_name=1),
        lambda: Event(execution_id=1, artifact_id="a", type="INPUT"),
        lambda: Graph(context="not a Context"),
    )
    for build in malformed:
        try:
            build()
        except LineageError:
            continue
        raise AssertionError("lineage accepted a non-string identifier or non-Context graph")


def test_the_record_round_trips_the_whole_graph() -> None:
    """What the backends write. Losing an edge here loses the lineage silently."""
    import json

    rec = graph().as_record()
    assert json.loads(json.dumps(rec))
    assert {a["id"] for a in rec["artifacts"]} == {"panel", "fc"}
    assert len(rec["events"]) == 2
    assert {e["type"] for e in rec["events"]} == {"INPUT", "OUTPUT"}


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
