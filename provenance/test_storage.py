"""Tests for the storage abstraction, including the remote paths this machine cannot execute.

A `gs://` location is exercised for everything that does NOT need a cloud client -- composition,
scheme detection, refusal to become a Path -- because those are the behaviours that decide whether
a runner can express a cloud path at all, and they are testable without GCP.

Run: python3 provenance/test_storage.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from provenance.storage import Location, StorageError, Workspace, workspace_from_env


def test_a_local_path_is_made_absolute() -> None:
    """A relative path means something different depending on where a job started, and a cloud
    runner starts somewhere else."""
    here = Location("data/m5_parquet")
    assert here.uri.startswith("/"), here.uri
    assert here.uri.endswith("data/m5_parquet")
    assert not here.is_remote and here.scheme == "file"


def test_a_remote_uri_is_recognised_and_left_alone() -> None:
    """Resolving it as a path would turn `gs://bucket/x` into `gs:/bucket/x` and the failure would
    surface far away, as a FileNotFoundError on a nonsense path."""
    for uri, scheme in (("gs://bucket/panel", "gs"), ("s3://bucket/panel", "s3"),
                        ("abfss://c@a/panel", "abfss")):
        loc = Location(uri)
        assert loc.is_remote and loc.scheme == scheme
        assert loc.uri == uri, "a remote uri was rewritten"


def test_children_compose_the_same_way_local_and_remote() -> None:
    assert Location("gs://b/root").child("a", "b").uri == "gs://b/root/a/b"
    local = Location(tempfile.mkdtemp()).child("a", "b")
    assert local.uri.endswith("/a/b") and not local.is_remote


def test_an_absolute_or_climbing_segment_is_refused() -> None:
    """`..` is a literal key segment on an object store, not a parent, so the result would point
    at something that does not exist rather than at a directory above."""
    root = Location("gs://bucket/root")
    for bad in ("/etc", "gs://other/x"):
        try:
            root.child(bad)
        except StorageError as e:
            assert "absolute" in str(e)
        else:
            raise AssertionError(f"composed an absolute segment {bad!r}")
    try:
        root.child("../escape")
    except StorageError as e:
        assert "'..'" in str(e)
        return
    raise AssertionError("composed a climbing segment")


def test_a_remote_location_refuses_to_become_a_path() -> None:
    """The caller is about to hand it to an API that only understands filesystems."""
    try:
        Location("gs://bucket/panel").as_path()
    except StorageError as e:
        assert "not a filesystem path" in str(e)
        return
    raise AssertionError("converted a gs:// uri to a Path")


def test_a_remote_location_refuses_an_existence_check() -> None:
    """Answering needs a cloud client, and this module stays dependency-free on purpose."""
    try:
        Location("gs://bucket/panel").exists()
    except StorageError as e:
        assert "without a gs client" in str(e)
        return
    raise AssertionError("claimed to know whether a bucket key exists")


def test_ensure_dir_is_a_no_op_remotely_rather_than_an_error() -> None:
    """Object stores have no directories. Raising would make every caller check which kind of
    location it holds, which is the coupling this module removes."""
    remote = Location("gs://bucket/out")
    assert remote.ensure_dir() is remote or remote.ensure_dir().uri == remote.uri
    local = Location(pathlib.Path(tempfile.mkdtemp()) / "made" / "here")
    local.ensure_dir()
    assert local.as_path().is_dir()


def test_an_empty_location_is_refused() -> None:
    """An empty uri silently becomes the working directory, which differs between a laptop and a
    container."""
    for bad in ("", "   "):
        try:
            Location(bad)
        except StorageError as e:
            assert "empty one silently becomes" in str(e)
        else:
            raise AssertionError(f"accepted {bad!r} as a location")


def test_scratch_may_not_be_remote() -> None:
    """Scratch holds re-derivable intermediates; every write would become a network round trip."""
    try:
        Workspace(panel=Location("gs://b/p"), outputs=Location("gs://b/o"),
                  scratch=Location("gs://b/s"))
    except StorageError as e:
        assert "network round trip" in str(e)
        return
    raise AssertionError("accepted a remote scratch root")


def test_a_workspace_mixes_a_remote_panel_with_local_scratch() -> None:
    """The shape a cloud run actually has: data and outputs in a bucket, spills on the node."""
    ws = Workspace(panel=Location("gs://b/panel"), outputs=Location("gs://b/out"),
                   scratch=Location(tempfile.mkdtemp()))
    assert ws.run_dir("experiments", "rung1").uri == "gs://b/out/experiments/rung1"
    d = ws.describe()
    assert d["panel_scheme"] == "gs" and d["outputs_scheme"] == "gs"


def test_the_environment_defaults_reproduce_todays_local_layout() -> None:
    """Setting nothing must change nothing, or this becomes a migration instead of an option."""
    for key in ("FORECAST_PANEL", "FORECAST_OUTPUTS", "FORECAST_SCRATCH"):
        os.environ.pop(key, None)
    ws = workspace_from_env("/tmp/repo")
    assert ws.panel.uri.endswith("/data/m5_parquet")
    assert ws.outputs.uri.endswith("/gates")
    assert not ws.panel.is_remote


def test_the_environment_can_point_at_a_bucket() -> None:
    """Two variables and the same script runs against GCS. That is the whole migration."""
    os.environ["FORECAST_PANEL"] = "gs://bucket/m5"
    os.environ["FORECAST_OUTPUTS"] = "gs://bucket/gates"
    try:
        ws = workspace_from_env("/tmp/repo")
        assert ws.panel.uri == "gs://bucket/m5" and ws.panel.is_remote
        assert ws.outputs.is_remote and not ws.scratch.is_remote
        assert ws.describe()["panel_scheme"] == "gs"
    finally:
        os.environ.pop("FORECAST_PANEL")
        os.environ.pop("FORECAST_OUTPUTS")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
