"""Volume publication uses the native no-overwrite contract, never FUSE append."""

import hashlib
import io
import json
import logging
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from autoforecast import _execution as events
from autoforecast import read_execution_events
from autoforecast.benchmark import DISTRIBUTED_SCHEMA
from databricks.sdk.errors import AlreadyExists, NotFound
from pyspark.sql import SparkSession

from mlops import _safe_files, _volume_files


@pytest.fixture
def volume(tmp_path, monkeypatch):
    class Files:
        def __init__(self):
            self.values = {}
            self.requests = []
            self.lock = threading.Lock()
            self.files = self
            self.api_client = self
            self._cfg = SimpleNamespace(workspace_id=None)
            self.corrupt = False

        def do(self, method, path, *, query, headers, data):
            from urllib.parse import unquote

            assert method == "PUT"
            assert headers["Content-Type"] == "application/octet-stream"
            if self._cfg.workspace_id:
                assert headers["X-Databricks-Workspace-Id"] == self._cfg.workspace_id
            key = unquote(path.removeprefix("/api/2.0/fs/files"))
            value = data.read() if hasattr(data, "read") else data
            with self.lock:
                self.requests.append((key, query["overwrite"]))
                if key in self.values and not query["overwrite"]:
                    raise AlreadyExists("file exists")
                self.values[key] = value + (b"changed" if self.corrupt else b"")

        def download(self, path):
            if path not in self.values:
                raise NotFound("file missing")
            return SimpleNamespace(contents=io.BytesIO(self.values[path]))

        def get_metadata(self, path):
            if path not in self.values:
                raise NotFound("file missing")
            return SimpleNamespace(content_length=len(self.values[path]), last_modified="fixed")

        def get_directory_metadata(self, path):
            if not Path(path).is_dir() and not any(p.startswith(path + "/") for p in self.values):
                raise NotFound("directory missing")

        def list_directory_contents(self, path):
            children = {}
            for key, value in self.values.items():
                if not key.startswith(path + "/"):
                    continue
                relative = key[len(path) + 1:]
                name, separator, _rest = relative.partition("/")
                children[name] = SimpleNamespace(
                    path=path + "/" + name, is_directory=bool(separator),
                    file_size=None if separator else len(value), last_modified=1,
                )
            return list(children.values())

        def delete(self, path):
            del self.values[path]

        def delete_directory(self, path):
            assert not any(key.startswith(path + "/") for key in self.values)

    client = Files()
    monkeypatch.setattr(_volume_files, "_client", lambda: client)
    monkeypatch.setattr(_volume_files, "is_volume", lambda path: path.is_relative_to(tmp_path))
    return tmp_path, client


@pytest.mark.parametrize("operation", ["publication", "lock", "audit"])
def test_volume_writes_keep_sdk_workspace_routing(volume, operation):
    root, client = volume
    client._cfg.workspace_id = "1234567890"
    path = root / "events.jsonl"
    if operation == "publication":
        _volume_files.publish_blocks(path, [b"complete"])
        assert client.values[str(path)] == b"complete"
    elif operation == "lock":
        with _volume_files.exclusive_lock(path):
            assert str(path) in client.values
        assert str(path) not in client.values
    else:
        client.values[str(path)] = b"first\n"
        _volume_files.append_bytes(path, b"second\n")
        assert client.values[str(path)] == b"first\nsecond\n"


def test_volume_publication_and_retry_read_native_bytes_despite_stale_mount(volume):
    root, client = volume
    _safe_files.publish_bytes_set(root, {"record.json": b"original"})
    assert not (root / "record.json").exists()  # The fake mount has no fresh API contents.
    assert _safe_files.read_regular_bytes(root, "record.json") == b"original"
    _safe_files.publish_bytes_set(root, {"record.json": b"original"})
    with pytest.raises(_safe_files.SafeFileError, match="differs"):
        _safe_files.publish_bytes_set(root, {"record.json": b"replacement"})
    assert client.values[str(root / "record.json")] == b"original"
    assert all(not overwrite for _, overwrite in client.requests)


def test_volume_file_has_one_winner_under_concurrent_publication(volume):
    root, client = volume

    def write(value):
        try:
            _safe_files.publish_bytes(root, "winner", value)
            return value
        except _safe_files.SafeFileError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        winners = [value for value in pool.map(write, [b"a", b"b", b"c", b"d"]) if value]
    assert len(winners) == 1
    assert client.values[str(root / "winner")] == winners[0]


def test_volume_readback_rejects_changed_bytes(volume):
    root, client = volume
    client.corrupt = True
    with pytest.raises(_safe_files.SafeFileError):
        _safe_files.publish_bytes(root, "record", b"expected")
    assert client.values[str(root / "record")] == b"expectedchanged"


def test_volume_audit_keeps_each_event_and_releases_lock(volume):
    root, client = volume
    path = root / "events.jsonl"
    _safe_files.append_audit_bytes(path, b'{"stage":"START"}\n')
    _safe_files.append_audit_bytes(path, b'{"stage":"PASS"}\n')
    assert _volume_files.read_bytes(path) == b'{"stage":"START"}\n{"stage":"PASS"}\n'
    assert str(root / "events.jsonl.lock") not in client.values


def test_volume_lock_never_steals_an_owner_or_hides_primary_failure(volume):
    root, client = volume
    path = root / "lock"
    with _volume_files.exclusive_lock(path):
        owner = client.values[str(path)]
        with pytest.raises(OSError, match="could not claim"):
            with _volume_files.exclusive_lock(path):
                pytest.fail("entered another writer's lock")
        assert client.values[str(path)] == owner
    assert str(path) not in client.values
    with pytest.raises(ValueError, match="primary"):
        with _volume_files.exclusive_lock(path):
            raise ValueError("primary")
    assert str(path) not in client.values


def test_volume_lock_preserves_changed_owner_and_original_error(volume):
    root, client = volume
    path = root / "lock"
    with pytest.raises(ValueError, match="primary") as caught:
        with _volume_files.exclusive_lock(path):
            client.values[str(path)] = b"different owner"
            raise ValueError("primary")
    assert client.values[str(path)] == b"different owner"
    assert "cleanup also failed" in caught.value.__notes__[0]


def test_volume_registration_lookup_uses_native_existence(volume):
    from autoforecast.study_registration import _read_saved, _record_path

    root, _ = volume
    _safe_files.publish_bytes(root, "study.json", b'{"value":1}')
    assert _record_path(root, "study") == root / "study.json"
    assert _read_saved(root / "study.json") == {"value": 1}
    _safe_files.publish_bytes(root, "study.yaml", b"value: 1")
    with pytest.raises(ValueError, match="conflicting JSON and YAML"):
        _record_path(root, "study")


def test_volume_delta_provisioning_holds_native_lock(volume):
    from autoforecast.integrations.delta import DeltaStudyStore

    root, client = volume
    path = root / "provision.lock"
    commands = []

    def sql(command):
        assert str(path) in client.values
        commands.append(command)

    # New tables have no comments yet, so preparing also describes every table and column.
    catalog = SimpleNamespace(getTable=lambda _name: SimpleNamespace(description=None),
                              listColumns=lambda _name: [])
    store = DeltaStudyStore(
        cast(SparkSession, SimpleNamespace(sql=sql, catalog=catalog)), namespace="neutral.studies",
    )
    store.validate_destinations = lambda: None
    store.prepare(lock_path=path)
    # Every statement, including the comments, ran while the lock was held.
    assert sum(command.startswith("CREATE TABLE") for command in commands) == len(store.tables)
    assert sum(command.startswith("COMMENT ON TABLE") for command in commands) == len(store.tables)
    assert any(command.startswith("ALTER TABLE") for command in commands)
    assert str(path) not in client.values


def test_volume_snapshot_reads_through_api_and_refuses_existing_destination(volume):
    root, client = volume
    source = root / "source"
    source.mkdir()
    _safe_files.publish_bytes(source, "model.bin", b"model bytes")
    destination = root / "retained"
    _safe_files.snapshot_regular_files(source, destination, ["model.bin"])
    assert client.values[str(destination / "model.bin")] == b"model bytes"
    with pytest.raises(_safe_files.SafeFileError):
        _safe_files.snapshot_regular_files(source, destination, ["model.bin"])
    _safe_files.copy_regular_file(root, "copied.bin", source / "model.bin")
    assert _safe_files.read_regular_bytes(root, "copied.bin") == b"model bytes"


def test_volume_publication_still_rejects_parent_traversal_and_link_roots(volume):
    root, client = volume
    with pytest.raises(_safe_files.SafeFileError):
        _safe_files.publish_bytes(root, "../outside", b"no")
    (root / "alias").symlink_to(root, target_is_directory=True)
    with pytest.raises(_safe_files.SafeFileError):
        _safe_files.publish_bytes(root / "alias", "record", b"no")
    assert client.values == {}


def test_volume_hashes_detect_remote_changes_even_with_stale_mounted_bytes(volume):
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from autoforecast.benchmark import parquet_content_hash

    from provenance.lineage import directory_content_digest, directory_digest

    root, client = volume
    folder = root / "observations"
    folder.mkdir()

    def parquet(value):
        output = io.BytesIO()
        pq.write_table(pa.table({"item": [1], "value": [value]}), output)
        return output.getvalue()

    before = parquet(2.0)
    after = parquet(3.0)
    (folder / "part.parquet").write_bytes(before)  # Simulate stale FUSE contents.
    _safe_files.publish_bytes(folder, "part.parquet", before)
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary)
        (local / "part.parquet").write_bytes(before)
        expected = (directory_digest(local, suffixes=(".parquet",)),
                    directory_content_digest(local, suffixes=(".parquet",)),
                    parquet_content_hash(local))
    assert (directory_digest(folder, suffixes=(".parquet",)),
            directory_content_digest(folder, suffixes=(".parquet",)),
            parquet_content_hash(folder)) == expected
    client.values[str(folder / "part.parquet")] = after
    assert (folder / "part.parquet").read_bytes() == before
    assert directory_digest(folder, suffixes=(".parquet",)) != expected[0]
    assert directory_content_digest(folder, suffixes=(".parquet",)) != expected[1]
    assert parquet_content_hash(folder) != expected[2]


def test_volume_hashing_keeps_reader_visible_file_checks(volume):
    from provenance.lineage import LineageError, directory_digest

    root, _ = volume
    _safe_files.publish_bytes(root, "unexpected.csv", b"unsafe inventory")
    with pytest.raises(LineageError, match="not a .parquet part"):
        directory_digest(root, suffixes=(".parquet",))


def test_parquet_snapshot_downloads_parts_but_not_reader_ignored_markers(volume, monkeypatch):
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from autoforecast import _logical_hash

    from provenance.lineage import directory_digest

    root, client = volume
    folder = root / "frozen"
    folder.mkdir()
    parts = {}
    for index in range(12):
        output = io.BytesIO()
        pq.write_table(pa.table({"x": [index]}), output)
        parts[f"part-{index:05d}.parquet"] = output.getvalue()
    markers = {"_SUCCESS": b"", "_committed_7": b"{}", "_started_7": b"",
               ".part-00000.parquet.crc": b"checksum"}
    for name, value in {**parts, **markers}.items():
        client.values[str(folder / name)] = value
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary)
        for name, value in {**parts, **markers}.items():
            (local / name).write_bytes(value)
        expected = (directory_digest(local, suffixes=(".parquet",)),
                    _logical_hash.fingerprint(local), directory_digest(local))
    downloads = []
    download = client.download

    def counted(path):
        downloads.append(Path(path).name)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    assert directory_digest(folder, suffixes=(".parquet",)) == expected[0]
    assert _logical_hash.fingerprint(folder) == expected[1]
    assert set(downloads) == set(parts)
    downloads.clear()
    # A digest of every file still needs the marker bytes.
    assert directory_digest(folder) == expected[2]
    assert set(downloads) == {*parts, *markers}


def test_volume_snapshot_identity_downloads_source_once_and_copies_its_bytes(volume, monkeypatch):
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from provenance.lineage import directory_digest, snapshot_parquet_identity

    root, client = volume
    source = root / "admitted"
    source.mkdir()
    parts = {}
    for index in range(3):
        output = io.BytesIO()
        pq.write_table(pa.table({"x": [index]}), output)
        parts[f"part-{index:05d}.parquet"] = output.getvalue()
    for name, value in {**parts, "_SUCCESS": b""}.items():
        client.values[str(source / name)] = value
    downloads = []
    download = client.download

    def counted(path):
        downloads.append(Path(path).name)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    with tempfile.TemporaryDirectory() as temporary:
        copy, sha256 = snapshot_parquet_identity(source, Path(temporary) / "copy", label="admitted")
        # The identity describes the copied bytes; the caller rechecks the source later.
        assert sorted(downloads) == sorted(parts)
        assert directory_digest(copy, suffixes=(".parquet",)) == sha256
        assert sorted(p.name for p in copy.iterdir()) == sorted(parts)
    downloads.clear()
    assert directory_digest(source, suffixes=(".parquet",)) == sha256


def test_volume_logical_reuse_downloads_once_and_ignores_stale_mount(volume, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from autoforecast import _logical_hash

    from provenance.lineage import directory_digest

    root, client = volume
    folder = root / "frozen"
    folder.mkdir()
    output = io.BytesIO()
    pq.write_table(pa.table({"x": [1, 2]}), output)
    _safe_files.publish_bytes(folder, "part.parquet", output.getvalue())
    expected = _logical_hash.fingerprint(folder)
    physical = directory_digest(folder, suffixes=(".parquet",))
    (folder / "part.parquet").write_bytes(b"stale mounted content")
    downloads = []
    download = client.download

    def counted(path):
        downloads.append(path)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    assert _logical_hash.verify_files(folder, physical_sha256=physical, logical=expected,
                                      trusted=True) == expected["sha256"]
    assert downloads == [str(folder / "part.parquet")]
    client.values[str(folder / "part.parquet")] = b"changed remote content"
    with pytest.raises(ValueError, match="bytes or inventory changed"):
        _logical_hash.verify_files(folder, physical_sha256=physical, logical=expected, trusted=True)


def test_volume_identity_reads_current_remote_bytes(volume):
    import hashlib

    root, client = volume
    path = root / "model.bin"
    path.write_bytes(b"old")
    client.values[str(path)] = b"current"
    assert _safe_files._regular_file_identity(path) == (hashlib.sha256(b"current").hexdigest(), 7)


def test_volume_snapshot_rejects_changes_during_download(volume, monkeypatch):
    root, client = volume
    client.values[str(root / "part.parquet")] = b"original"
    download = client.download

    def changing_download(path):
        response = download(path)
        client.values[str(root / "new.parquet")] = b"new part"
        return response

    monkeypatch.setattr(client, "download", changing_download)
    with pytest.raises(OSError, match="directory changed"):
        with _volume_files.local_snapshot(root):
            pytest.fail("changed inventory must not reach the reader")


def test_volume_download_failure_does_not_return_partial_identity(volume, monkeypatch):
    root, client = volume

    def failed_download(path):
        raise OSError("connection ended")

    monkeypatch.setattr(client, "download", failed_download)
    with pytest.raises(_safe_files.SafeFileError, match="could not read immutable"):
        _safe_files._regular_file_identity(root / "model.bin")


@pytest.mark.parametrize("kind", ["bundle", "attempt"])
def test_volume_retained_audit_is_sealed_and_reloadable_before_mount_refresh(volume, kind):
    import json

    from autoforecast._execution import _stage_event
    from autoforecast.experiments import load_completed_attempt, open_local_attempt
    from autoforecast.test_experiments import _metadata

    from mlops.run_bundle import load_completed_run_bundle, open_run_bundle

    root, client = volume
    if kind == "bundle":
        output = open_run_bundle(root, run_id="check", task="verification")
        load = load_completed_run_bundle
    else:
        output = open_local_attempt(root, study_id="study", trial_id="trial", attempt_id="attempt",
                                    metadata=_metadata(learning_rate=0.01))
        load = load_completed_attempt
    # The API can see a nested artifact before its parent appears in the mount.
    output.write_bytes("artifacts/nested.bin", b"model bytes")
    (output.path / "artifacts").rmdir()
    audit = output.path / "events.jsonl"
    _stage_event(audit, "check", "START")
    _stage_event(audit, "check", "PASS")
    output.finalize(status="completed", summary={})
    assert not audit.exists()
    record = json.loads(_volume_files.read_bytes(output.path / "artifact_manifest.json"))
    assert "events.jsonl" in {entry["path"] for entry in record["artifacts"]}
    load(output.path)
    client.values[str(audit)] = b"changed"
    with pytest.raises(ValueError):
        load(output.path)


@pytest.mark.parametrize("terminal", [None, "artifact_manifest.json", "completion.json"])
def test_volume_guide_recovery_checks_remote_terminal_markers(volume, terminal):
    from mlops.run_report import _discard_unsealed_guide

    root, client = volume
    client.values[str(root / "RUN.md")] = b"guide"
    if terminal:
        client.values[str(root / terminal)] = b"terminal"
    _discard_unsealed_guide(root, b"guide", ValueError("original"))
    assert (str(root / "RUN.md") in client.values) == bool(terminal)
    if not terminal:
        client.values[str(root / "RUN.md")] = b"another guide"
        _discard_unsealed_guide(root, b"guide", ValueError("original"))
        assert client.values[str(root / "RUN.md")] == b"another guide"


def test_volume_study_load_and_panel_use_native_bytes(volume, monkeypatch):
    import hashlib
    import tempfile

    import autoforecast.study as module

    root, client = volume
    folder = root / ("a" * 64)
    folder.mkdir()
    manifest = folder / "benchmark.json"
    manifest.write_bytes(b"stale manifest")
    client.values[str(manifest)] = b"current manifest"
    contract = b'''schema_version: 1
dataset:
  id: synthetic
  panel:
    series_col: item
    time_col: day
    target_col: value
    frequency: D
    season_length: 1
    target_estimand: observed_nonnegative
'''
    client.values[str(folder / "dataset.yaml")] = contract
    record = {"schema": DISTRIBUTED_SCHEMA, "recipe": {"validation_origins": ["2024-01-05"]},
              "study_id": "a" * 64,
              "dataset_sha256": hashlib.sha256(contract).hexdigest(),
              "role_paths": {"selection": "selection"}, "artifacts": {"selection": "0" * 64}}

    def verified(path, sha, **kwargs):
        assert path == manifest and sha == hashlib.sha256(b"current manifest").hexdigest()
        return path, record

    monkeypatch.setattr(module, "verify_benchmark", verified)
    monkeypatch.setattr(module, "read_benchmark_manifest", verified)
    with tempfile.TemporaryDirectory() as scratch:
        study = module.load_study("a" * 64, registry_root=root,
                                  spark=cast(SparkSession, object()), scratch_root=scratch)
        try:
            assert study.panel.target_col == "value"
        finally:
            study.close()


def test_unknown_study_keeps_local_error_contract_without_spark(tmp_path):
    from autoforecast.benchmark import BenchmarkError
    from autoforecast.study import load_study

    with pytest.raises(FileNotFoundError, match="Unknown study_id"):
        load_study("0" * 64, registry_root=tmp_path)
    (tmp_path / ("0" * 64)).mkdir()
    with pytest.raises(BenchmarkError, match="cannot read"):
        load_study("0" * 64, registry_root=tmp_path)
    with pytest.raises(BenchmarkError):
        load_study("0" * 64, registry_root=tmp_path / "absent")


def test_volume_parquet_snapshot_uses_remote_parts_despite_stale_mount(volume):
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from provenance.lineage import directory_digest, snapshot_parquet_directory

    root, client = volume
    source = root / "source"
    source.mkdir()
    buffer = io.BytesIO()
    pq.write_table(pa.table({"value": [2.0]}), buffer)
    client.values[str(source / "part.parquet")] = buffer.getvalue()
    (source / "part.parquet").write_bytes(b"stale bytes")
    expected = directory_digest(source, suffixes=(".parquet",))
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "snapshot"
        snapshot_parquet_directory(source, destination, expected_sha256=expected, label="data")
        assert pq.read_table(destination).to_pylist() == [{"value": 2.0}]
    retained = root / "retained"
    snapshot_parquet_directory(source, retained, expected_sha256=expected, label="data")
    assert directory_digest(retained, suffixes=(".parquet",)) == expected


def test_volume_inventory_accepts_volume_root_and_normalizes_relative_paths(monkeypatch):
    root = Path("/Volumes/catalog/schema/volume")
    calls = []
    files = SimpleNamespace(
        get_directory_metadata=lambda path: calls.append(path),
        list_directory_contents=lambda path: calls.append(path) or [],
    )
    monkeypatch.setattr(_volume_files, "_client", lambda: SimpleNamespace(files=files))
    monkeypatch.setattr(_volume_files.os, "getcwd", lambda: "/")
    assert _volume_files._inventory(root) == (True, {})
    assert _volume_files._inventory(Path("Volumes/catalog/schema/volume")) == (True, {})
    assert calls == [str(root)] * 4
    with pytest.raises(ValueError, match="volume files require"):
        _volume_files._path(root)


def test_volume_publication_rejects_same_length_tail_corruption(volume, monkeypatch):
    root, client = volume
    original = client.download
    reads = []

    class Stream(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    def changed_download(path):
        value = original(path).contents.read()
        return SimpleNamespace(contents=Stream(value[:-1] + b"x"))

    monkeypatch.setattr(client, "download", changed_download)
    with pytest.raises(_safe_files.SafeFileError):
        _safe_files.publish_bytes(root, "model", b"a" * (2 * 1024 * 1024 + 7))
    assert len(reads) >= 4


@pytest.mark.parametrize("defect", ["no_contents", "short", "read_failure"])
def test_volume_snapshot_rejects_incomplete_download(volume, monkeypatch, defect):
    root, client = volume
    client.values[str(root / "part.parquet")] = b"full payload"

    class Broken(io.BytesIO):
        def read(self, size=-1):
            raise OSError("stream failed")

    stream = {"no_contents": None, "short": io.BytesIO(b"short"), "read_failure": Broken()}
    monkeypatch.setattr(client, "download", lambda _: SimpleNamespace(contents=stream[defect]))
    with pytest.raises(OSError):
        with _volume_files.local_snapshot(root):
            pytest.fail("incomplete bytes must not reach the reader")


@pytest.mark.parametrize("fail_after_write", [False, True])
def test_volume_audit_write_failure_is_visible_and_releases_lock(volume, monkeypatch,
                                                               fail_after_write):
    root, client = volume
    path = root / "events.jsonl"
    client.values[str(path)] = b"first\n"
    put = _volume_files._put

    def failure(target, contents, *, overwrite):
        if target == path:
            if fail_after_write:
                put(target, b"unexpected", overwrite=overwrite)
                return
            raise OSError("upload failed")
        put(target, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", failure)
    with pytest.raises(OSError):
        _safe_files.append_audit_bytes(path, b"second\n")
    assert str(path) + ".lock" not in client.values
    if not fail_after_write:
        assert client.values[str(path)] == b"first\n"


@pytest.mark.parametrize("primary", [False, True])
def test_volume_lock_cleanup_failure_preserves_primary(volume, monkeypatch, primary):
    root, client = volume

    def denied(_):
        raise OSError("delete denied")

    monkeypatch.setattr(client, "delete", denied)
    with pytest.raises(ValueError if primary else OSError) as caught:
        with _volume_files.exclusive_lock(root / "lock"):
            if primary:
                raise ValueError("primary error")
    assert str(root / "lock") in client.values
    if primary:
        assert str(caught.value) == "primary error"
        assert "delete denied" in caught.value.__notes__[0]


def test_volume_copy_crosses_local_boundary(volume):
    import tempfile

    root, _ = volume
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary)
        (local / "source").write_bytes(b"model bytes")
        _safe_files.copy_regular_file(root, "model", local / "source")
        _safe_files.copy_regular_file(local, "returned", root / "model")
        assert (local / "returned").read_bytes() == b"model bytes"


def test_volume_snapshot_failure_cleans_only_its_remote_parts(volume, monkeypatch):
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from provenance.lineage import LineageError, directory_digest, snapshot_parquet_directory

    root, client = volume
    destination = root / "snapshot"
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary)
        for part in ("one.parquet", "two.parquet"):
            pq.write_table(pa.table({"value": [1]}), source / part)
        expected = directory_digest(source, suffixes=(".parquet",))
        # A remote destination is occupied even when the mount has no entry for it.
        client.values[str(destination / "original.parquet")] = b"keep"
        with pytest.raises(LineageError, match="already exists"):
            snapshot_parquet_directory(source, destination, expected_sha256=expected, label="data")
        assert client.values.pop(str(destination / "original.parquet")) == b"keep"
        original = _volume_files.publish_blocks
        count = 0
        lock = threading.Lock()

        def interrupted(*args):
            nonlocal count
            with lock:
                count += 1
                failing = count == 2
            if failing:
                raise OSError("copy interrupted")
            return original(*args)

        # Parts are published concurrently; the second publication fails.
        monkeypatch.setattr(_volume_files, "publish_blocks", interrupted)
        with pytest.raises(OSError, match="copy interrupted"):
            snapshot_parquet_directory(source, destination, expected_sha256=expected, label="data")
        assert count == 2
        assert not any(key.startswith(str(destination) + "/") for key in client.values)


def test_volume_publish_error_reports_upload_failure(volume, monkeypatch):
    root, _ = volume

    def disconnected(*_):
        raise OSError("upload disconnected")

    monkeypatch.setattr(_volume_files, "publish_blocks", disconnected)
    with pytest.raises(_safe_files.SafeFileError, match="could not publish immutable") as caught:
        _safe_files.publish_bytes(root, "record.json", b"data")
    assert str(caught.value.__cause__) == "upload disconnected"
    assert "directory hierarchy" not in str(caught.value)


def test_volume_guide_cleanup_without_a_guide_does_not_add_failure_note(volume):
    from mlops.run_report import _discard_unsealed_guide

    root, _ = volume
    error = ValueError("original")
    _discard_unsealed_guide(root, b"guide", error)
    assert not getattr(error, "__notes__", [])


def test_volume_tuning_profile_reads_current_native_bytes(volume):
    import tempfile

    from autoforecast.study_tuning import _config
    from autoforecast.test_study_tuning import config

    root, client = volume
    with tempfile.TemporaryDirectory() as temporary:
        payload = config(Path(temporary)).read_bytes()
    path = root / "profile.yaml"
    client.values[str(path)] = payload
    assert not path.exists()
    assert _config(path)[0] == payload


def test_volume_tuning_reload_passes_native_contract_to_parser(volume, monkeypatch):
    import hashlib

    import autoforecast.study_tuning as module

    root, client = volume
    attempt = root / "attempt"
    attempt.mkdir()
    client.values[str(attempt / "artifact_manifest.json")] = b"{}"
    dataset = root / "dataset.yaml"
    dataset.write_bytes(b"stale contract")
    client.values[str(dataset)] = b"current contract"
    profile = b"profile"
    context = {"implementation": {}, "profile_sha256": hashlib.sha256(profile).hexdigest(),
               "max_trials": 1, "model_id": "custom", "study": {
                   "path": str(root / "benchmark.json"), "sha256": "study-sha"}}
    selection = {"selected": [{"attempt_path": "attempt", "artifact_manifest_sha256": "sha"}],
                 "candidates": [{}],
                 "comparability_fingerprint": hashlib.sha256(
                     module._json_bytes(context)).hexdigest()}
    monkeypatch.setattr(module, "load_study_selection", lambda _: selection)
    monkeypatch.setattr(module, "load_completed_attempt", lambda _: (
        {}, {"artifact_manifest_sha256": "sha"}))
    monkeypatch.setattr(module, "_retained_manifest", lambda *_: {})
    monkeypatch.setattr(module, "_retained_json", lambda _a, name, _m: (
        {"model": {"id": "custom"}} if name == "metadata/attempt.json" else
        {"schema": "autoforecast.external-trial.v2", "context": context}))
    monkeypatch.setattr(module, "_retained_bytes", lambda *_: profile)
    monkeypatch.setattr(module, "verify_retained", lambda *_: None)
    monkeypatch.setattr(module, "verify_benchmark", lambda *a, **kw: (root / "benchmark.json", {}))

    class ReachedParser(Exception):
        pass

    def checked(payload, path):
        assert payload == b"current contract" and path == dataset
        raise ReachedParser

    monkeypatch.setattr(module, "_load_dataset_contract_bytes", checked)
    with pytest.raises(ReachedParser):
        module.load_tuning_result(root)


def test_volume_incomplete_admission_requires_recovery(volume, monkeypatch):
    import autoforecast.study_registration as module

    root, client = volume
    staging = root / "staging"
    staging.mkdir()
    original = root / "admissions" / "study" / "partial.parquet"
    client.values[str(original)] = b"keep"

    class Frame:
        schema = None
        write = SimpleNamespace(parquet=lambda _: None,
                                option=lambda *_: SimpleNamespace(parquet=lambda _: None))

        def select(self, *_):
            return self

        def repartition(self, *_):
            return self

        def join(self, *_):
            return self

        def unionByName(self, *_):
            return self

        def withColumn(self, *_):
            return self

        def limit(self, *_):
            return self

        def count(self):
            return 0

    registry = SimpleNamespace(root=root, spark=SimpleNamespace(
        read=SimpleNamespace(parquet=lambda _: Frame())))
    monkeypatch.setattr(module, "_same_keys", lambda *_: None)
    monkeypatch.setattr(module, "F", SimpleNamespace(lit=lambda value: value))
    frozen = {"schema": DISTRIBUTED_SCHEMA, "study_id": "study",
              "recipe": {"source_keys": ["item"]},
              "artifacts": {"source-keys": "sha"}, "role_paths": {"source-keys": "source-keys"}}
    identity = {"native_policy": {"excluded_keys": [], "excluded_new_keys": []}}
    with pytest.raises(ValueError, match="incomplete admission destination"):
        module._save_admission(registry, staging, identity, frozen, format="json")
    assert client.values[str(original)] == b"keep"


@pytest.mark.parametrize("stale_mount", [False, True])
def test_volume_admission_seals_remote_directories_absent_from_mounted_listing(
    volume, monkeypatch, stale_mount
):
    import json

    import autoforecast.study_registration as registration
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, client = volume
    staging = root / "admission-staging"
    staging.mkdir()
    output = io.BytesIO()
    pq.write_table(pa.table({"item": ["sensor"]}), output)
    part = output.getvalue()
    expected_names = {"starting_keys", "retained_keys", "exclusions", "final_keys",
                      "native_exclusions"}
    for name in ("starting_keys", "retained_keys", "exclusions"):
        client.values[str(staging / name / "part.parquet")] = part
        if not stale_mount:
            (staging / name).mkdir()

    class Frame:
        schema = None

        def __init__(self):
            self.write = SimpleNamespace(
                parquet=self.materialize,
                option=lambda *_: SimpleNamespace(parquet=self.materialize))

        def materialize(self, path):
            client.values[str(Path(path) / "part.parquet")] = part
            if not stale_mount:
                Path(path).mkdir()

        def select(self, *_):
            return self

        def repartition(self, *_):
            return self

        def join(self, *_):
            return self

        def unionByName(self, *_):
            return self

        def withColumn(self, *_):
            return self

        def limit(self, *_):
            return self

        def count(self):
            return 1

    frame = Frame()
    registry = SimpleNamespace(root=root, scratch_root=root, spark=SimpleNamespace(
        read=SimpleNamespace(parquet=lambda _: frame)))
    # This fixture models native storage only; use the same offline hash format
    # while the real Spark encoder is exercised by the logical hash tests.
    capture = registration._logical_hash._capture
    monkeypatch.setattr(registration._logical_hash, "_capture",
                        lambda path, *, spark=None: capture(path))
    monkeypatch.setattr(registration, "_same_keys", lambda *_: None)
    monkeypatch.setattr(registration, "F", SimpleNamespace(lit=lambda value: value))
    rename = registration.os.rename

    def mounted_rename(source, destination):
        # Reflect a mount-side directory rename in the synthetic native service.
        record = (source / "admission.json").read_bytes()
        rename(source, destination)
        for key in list(client.values):
            if key.startswith(str(source) + "/"):
                client.values[str(destination) + key[len(str(source)):]] = client.values.pop(key)
        client.values[str(destination / "admission.json")] = record

    monkeypatch.setattr(registration.os, "rename", mounted_rename)
    frozen = {"schema": DISTRIBUTED_SCHEMA, "study_id": "study",
              "recipe": {"source_keys": ["item"]},
              "artifacts": {"source-keys": "sha"}, "role_paths": {"source-keys": "source-keys"}}
    identity = {"native_policy": {"excluded_keys": [], "excluded_new_keys": []}}
    binding = registration._save_admission(registry, staging, identity, frozen, format="json")
    record = json.loads(client.values[str(root / binding["path"])])
    assert set(record["artifacts"]) == expected_names, (
        "admission record omitted remote evidence directories", record["artifacts"]
    )

    from hashlib import sha256

    record_path = root / binding["path"]
    original = client.values[str(record_path)]
    unexpected = str(record_path.parent / "unexpected" / "part.parquet")
    client.values[unexpected] = part
    with pytest.raises(ValueError, match="directory set"):
        registration._verify_admission(root, binding)
    del client.values[unexpected]
    missing = str(record_path.parent / "exclusions" / "part.parquet")
    excluded_bytes = client.values.pop(missing)
    with pytest.raises(ValueError, match="directory set"):
        registration._verify_admission(root, binding)
    client.values[missing] = excluded_bytes
    record["artifacts"] = {}
    payload = json.dumps(record).encode()
    client.values[str(record_path)] = payload
    incomplete = dict(binding, sha256=registration.digest(record),
                      physical_sha256=sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="required cohort evidence"):
        registration._verify_admission(root, incomplete)
    client.values[str(record_path)] = original
    registration._verify_admission(root, binding)


def test_volume_staging_inputs_and_cleanup_use_remote_listing(volume):
    from autoforecast.study_registration import _remove_staging_directory, _staging_directories

    root, client = volume
    staging = root / "staging"
    staging.mkdir()
    client.values[str(staging / "prepared" / "part.parquet")] = b"data"
    client.values[str(staging / "reference_input" / "part.parquet")] = b"references"
    client.values[str(staging / "note.json")] = b"note"
    assert _staging_directories(staging) == [staging / "prepared", staging / "reference_input"]
    _remove_staging_directory(staging / "prepared")
    assert _staging_directories(staging) == [staging / "reference_input"]
    assert client.values[str(staging / "note.json")] == b"note"
    _remove_staging_directory(staging / "absent")


@pytest.mark.parametrize("operation", ["read", "identity", "copy", "snapshot"])
def test_volume_reads_do_not_require_mounted_parent_directories(volume, operation):
    import hashlib

    root, client = volume
    source = root / "unmounted/nested/model.bin"
    client.values[str(source)] = b"complete model"
    assert not source.parent.exists()
    if operation == "read":
        assert _safe_files.read_regular_bytes(source.parent, source.name) == b"complete model"
    elif operation == "identity":
        assert _safe_files._regular_file_identity(source) == (
            hashlib.sha256(b"complete model").hexdigest(), len(b"complete model"),
        )
    elif operation == "copy":
        _safe_files.copy_regular_file(root, "copied.bin", source)
        assert client.values[str(root / "copied.bin")] == b"complete model"
    else:
        _safe_files.snapshot_regular_files(source.parent, root / "snapshot", [source.name])
        assert client.values[str(root / "snapshot" / source.name)] == b"complete model"


def test_volume_selection_ranks_verified_bytes_and_reloads_without_mounted_attempts(volume):
    import json
    import shutil

    from autoforecast.experiments import (
        load_study_selection,
        open_local_attempt,
        seal_study_selection,
    )
    from autoforecast.test_experiments import _metadata

    root, _ = volume
    for number, value in enumerate((1.0, 2.0), start=1):
        attempt = open_local_attempt(
            root, study_id="smoke", trial_id=f"trial{number}", attempt_id="attempt1",
            metadata={**_metadata(learning_rate=0.001 * number),
                      "study": {"direction": "minimize", "top_k": 1}},
        )
        validation = {"objective": "validation/objective", "value": value,
                      "test_metrics": "not_evaluated"}
        attempt.write_json("evaluation/validation.json", validation)
        attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
        # Cached scores would choose the other trial if ranking used the mount.
        (attempt.path / "evaluation/validation.json").write_text(
            json.dumps({**validation, "value": 10.0 - value}),
        )
    study = root / "studies/smoke"
    seal_study_selection(study, top_k=1, direction="minimize", required_trial_count=2)
    assert load_study_selection(study)["selected"][0]["trial_id"] == "trial1"
    shutil.rmtree(study / "trials")  # Remote files remain, but their mount entries vanish.
    assert load_study_selection(study)["selected"][0]["trial_id"] == "trial1"


def test_volume_relative_guide_cleanup_uses_an_absolute_api_path(volume, monkeypatch):
    from mlops.run_report import _discard_unsealed_guide

    root, client = volume
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        _volume_files, "is_volume", lambda path: path.absolute().is_relative_to(root),
    )
    client.values[str(root / "bundle/RUN.md")] = b"guide"
    error = ValueError("original failure")
    _discard_unsealed_guide(Path("bundle"), b"guide", error)
    assert str(root / "bundle/RUN.md") not in client.values
    assert not getattr(error, "__notes__", [])


def test_volume_relative_bundle_can_retry_after_manifest_failure(volume, monkeypatch):
    from mlops.run_bundle import load_completed_run_bundle, open_run_bundle

    root, _ = volume
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        _volume_files, "is_volume", lambda path: path.absolute().is_relative_to(root),
    )
    bundle = open_run_bundle(Path("stores"), run_id="relative", task="verification")
    write_json = type(bundle).write_json

    def fail_manifest(self, relative, payload):
        if relative == "artifact_manifest.json":
            raise OSError("injected manifest failure")
        return write_json(self, relative, payload)

    monkeypatch.setattr(type(bundle), "write_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure") as caught:
        bundle.finalize(status="completed", summary={})
    assert not getattr(caught.value, "__notes__", [])
    monkeypatch.setattr(type(bundle), "write_json", write_json)
    bundle.finalize(status="completed", summary={})
    load_completed_run_bundle(bundle.path)


def test_volume_runs_are_isolated_and_only_completed_results_load(volume):
    from mlops.run_bundle import load_completed_run_bundle, open_run_bundle

    root, client = volume
    barrier = threading.Barrier(2)

    def run(run_id):
        bundle = open_run_bundle(root, run_id=run_id, task="verification")
        bundle.write_bytes("model.bin", run_id.encode())
        with pytest.raises(ValueError):
            load_completed_run_bundle(bundle.path)
        # Both writers are active, but their artifacts have separate owners.
        barrier.wait(timeout=10)
        bundle.finalize(status="completed", summary={"model": run_id})
        return bundle

    with ThreadPoolExecutor(max_workers=2) as pool:
        bundles = list(pool.map(run, ["run-a", "run-b"]))
    for bundle in bundles:
        assert load_completed_run_bundle(bundle.path)["run_id"] == bundle.run_id
        assert client.values[str(bundle.path / "model.bin")] == bundle.run_id.encode()
        writes = [path for path, _ in client.requests if path.startswith(str(bundle.path) + "/")]
        assert writes[-1] == str(bundle.path / "completion.json")
        with pytest.raises(ValueError, match="already exists"):
            open_run_bundle(root, run_id=bundle.run_id, task="verification")
        with pytest.raises(ValueError, match="completed"):
            bundle.write_bytes("late.bin", b"no")
    # Changing one published run invalidates it without affecting the other.
    client.values[str(bundles[0].path / "model.bin")] = b"changed"
    with pytest.raises(ValueError):
        load_completed_run_bundle(bundles[0].path)
    assert load_completed_run_bundle(bundles[1].path)["run_id"] == "run-b"


def test_volume_interrupted_publication_requires_a_fresh_attempt(volume, monkeypatch):
    from mlops.run_bundle import load_completed_run_bundle, open_run_bundle

    root, client = volume
    failed = open_run_bundle(root, run_id="interrupted", task="verification")
    failed.write_bytes("model.bin", b"first model")
    publish = _volume_files.publish_blocks

    def stop_before_completion(path, blocks):
        if path == failed.path / "completion.json":
            raise OSError("interrupted before completion")
        return publish(path, blocks)

    monkeypatch.setattr(_volume_files, "publish_blocks", stop_before_completion)
    with pytest.raises(ValueError):
        failed.finalize(status="completed", summary={})
    assert str(failed.path / "artifact_manifest.json") in client.values
    assert str(failed.path / "completion.json") not in client.values
    with pytest.raises(ValueError):
        load_completed_run_bundle(failed.path)
    retained = dict(client.values)
    fresh = open_run_bundle(root, run_id="new-attempt", task="verification")
    fresh.write_bytes("model.bin", b"second model")
    fresh.finalize(status="completed", summary={})
    assert load_completed_run_bundle(fresh.path)["status"] == "completed"
    assert all(client.values[path] == value for path, value in retained.items())


def test_volume_snapshot_checks_previously_frozen_hash(volume):
    from provenance.lineage import LineageError, directory_digest, snapshot_parquet_directory

    root, client = volume
    source = root / "source"
    source.mkdir()
    client.values[str(source / "a.parquet")] = b"first version a"
    client.values[str(source / "b.parquet")] = b"first version b"
    expected = directory_digest(source, suffixes=(".parquet",))
    # A stable inventory with changed bytes must not establish a new trusted hash.
    client.values[str(source / "b.parquet")] = b"other version b"
    with pytest.raises(LineageError, match="changed before its snapshot"):
        snapshot_parquet_directory(
            source, root / "snapshot", expected_sha256=expected, label="training",
        )
    assert not any(path.startswith(str(root / "snapshot") + "/") for path in client.values)


def test_native_selection_is_shown_as_sealed_before_mount_refresh(volume):
    from autoforecast import experiment_report
    from autoforecast.experiments import (
        load_study_selection,
        open_local_attempt,
        seal_study_selection,
    )
    from autoforecast.test_experiments import _metadata

    root, client = volume
    attempt = open_local_attempt(
        root, study_id="smoke", trial_id="trial1", attempt_id="attempt1",
        metadata={**_metadata(learning_rate=0.001),
                  "study": {"direction": "minimize", "top_k": 1}},
    )
    attempt.write_json("evaluation/validation.json", {
        "objective": "validation/objective", "value": 1.0, "test_metrics": "not_evaluated",
    })
    attempt.finalize(status="completed", summary={})
    # Previous artifact cache entries are current; only the just-published selection lags.
    for name, raw in client.values.items():
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    study = root / "studies/smoke"
    selection = seal_study_selection(study, top_k=1, direction="minimize", required_trial_count=1)
    assert not selection.exists()
    assert load_study_selection(study)["selected"][0]["trial_id"] == "trial1"
    report = experiment_report.render_study_report(study)
    assert "Selection is sealed: top_k=1" in report.read_text()
    assert "Selected (rank 1)" in report.read_text()


@pytest.mark.parametrize("cached", ["missing", "stale"])
def test_native_selection_parameter_resolution_uses_authenticated_bytes(
    volume, monkeypatch, cached,
):
    import hashlib

    from autoforecast import parameter_sources
    from autoforecast.experiments import load_study_selection, seal_study_selection
    from autoforecast.test_parameter_sources import _project

    root, client = volume
    # The existing fixture is valid and exercises all ordinary provenance checks.
    with monkeypatch.context() as local:
        local.setattr(_volume_files, "is_volume", lambda _: False)
        project, candidate, study, selection, attempt = _project(root / "fixture")
    for path in (root / "fixture" / "hpo").rglob("*"):
        if path.is_file():
            client.values[str(path)] = path.read_bytes()
    del client.values[str(selection)]
    selection.unlink()
    seal_study_selection(selection.parent, top_k=1, direction="minimize", required_trial_count=1)
    trusted = client.values[str(selection)]
    if cached == "stale":
        # A cache can retain bytes from an older result at this path.
        selection.write_bytes(b'{"obsolete_selection":true}\n')
    assert load_study_selection(selection.parent)["selected"]
    record = parameter_sources.parameter_resolution_from_selection(
        candidate, study, selection, kind="reuse_selection", project_root=project,
    )
    parameters, source = record["effective_parameters"], record["source"]
    assert isinstance(parameters, dict) and isinstance(source, dict)
    assert parameters["model.hidden_size"] == 32
    assert source["selection_sha256"] == hashlib.sha256(trusted).hexdigest()


def test_volume_progress_reuses_inventory_bytes_and_does_not_recursively_audit(volume):
    import json

    from autoforecast._execution import Execution, _stage, operation

    root, client = volume
    source = root / 'source' / 'part.parquet'
    payload = b'x' * (2 * 1024 * 1024 + 13)
    client.values[str(source)] = payload
    audit = root / 'progress' / 'events.jsonl'
    with Execution(audit, ['copy']) as execution:
        with _stage(audit, 'copy'):
            with _volume_files.local_snapshot(source.parent) as copied:
                assert (copied / source.name).read_bytes() == payload
            execution.flush_audit(audit)
            before = len(client.requests)
            with operation('Count loop') as update:
                start = len(client.requests)
                for count in range(100):
                    update(completed=count, total=100, unit='rows')
                assert len(client.requests) <= start + 1  # Only the queued START upload.
            execution.flush_audit(audit)
            assert len(client.requests) - before == 2  # One immutable write per lifecycle event.
    rows = [json.loads(line) for line in _volume_files.read_audit_bytes(audit).splitlines()]
    assert len(rows) == 6
    assert all(row['status'] in {'START', 'PASS'} for row in rows)


def test_download_reports_unequal_file_bytes_and_final_file_count(volume, monkeypatch):
    import json

    from autoforecast import _execution as events

    root, client = volume
    client.values[str(root / 'data' / 'small.parquet')] = b'a'
    client.values[str(root / 'data' / 'large.parquet')] = b'b' * (1024 * 1024 + 4)
    snapshots = []
    original = events.progress
    def record(**counts):
        original(**counts)
        snapshots.append(counts)
    monkeypatch.setattr(events, 'progress', record)
    audit = root / 'audit' / 'events.jsonl'
    with events.Execution(audit, ['download']):
        with events._stage(audit, 'download'):
            with _volume_files.local_snapshot(root / 'data') as local:
                assert (local / 'small.parquet').read_bytes() == b'a'
    assert snapshots[-1]['completed'] == snapshots[-1]['total'] == 1024 * 1024 + 5
    assert snapshots[-1]['files'] == 2
    rows = [json.loads(line) for line in _volume_files.read_audit_bytes(audit).splitlines()]
    finished = next(r for r in rows if r['stage'] == 'Download snapshot for integrity checks'
                    and r['status'] == 'PASS')
    assert finished['completed'] == 1024 * 1024 + 5 and finished['files'] == 2
    assert len(rows) == 4


@pytest.mark.parametrize("mutate", [False, True])
def test_volume_capture_returns_binding_and_rechecks_remote_bytes(volume, monkeypatch, mutate):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from autoforecast import _logical_hash
    from autoforecast.study_registration import _artifact_record

    root, client = volume
    folder = root / "capture"
    folder.mkdir()
    output = io.BytesIO()
    pq.write_table(pa.table({"x": [1, 2]}), output)
    _safe_files.publish_bytes(folder, "part.parquet", output.getvalue())
    download = client.download
    downloads = []

    def counted(path):
        downloads.append(path)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    offline = _logical_hash._offline

    def calculate(*args):
        result = offline(*args)
        if mutate:
            # The final read must see a remote change, even with an unchanged mount.
            client.values[str(folder / "part.parquet")] = b"changed captured bytes"
        return result

    monkeypatch.setattr(_logical_hash, "_offline", calculate)
    if mutate:
        with pytest.raises(ValueError, match="changed during logical hashing"):
            _artifact_record(folder)
    else:
        record = _artifact_record(folder)
        assert record["logical_hash"]["rows"] == 2
        assert len(record["physical_sha256"]) == 64
    assert downloads == [str(folder / "part.parquet")] * 2


def test_standalone_volume_load_pins_native_manifest_before_full_verification(volume, monkeypatch):
    import hashlib

    from autoforecast import study as api
    from autoforecast.benchmark import BenchmarkError

    root, client = volume
    study_id = "a" * 64
    path = root / study_id / "benchmark.json"
    client.values[str(path)] = b"original native manifest"
    assert not path.exists()
    original_sha = hashlib.sha256(client.values[str(path)]).hexdigest()

    checked = []

    def verified(manifest, expected_sha256, **kwargs):
        assert manifest == path and expected_sha256 == original_sha
        assert not kwargs.get("_reuse_verified", False)
        # A remote replacement during the check cannot change the original binding.
        client.values[str(path)] = b"replacement during full verification"
        checked.append(expected_sha256)
        assert hashlib.sha256(client.values[str(path)]).hexdigest() != expected_sha256
        raise BenchmarkError("benchmark manifest changed")

    def manifest_record(manifest, expected_sha256):
        assert manifest == path and expected_sha256 == original_sha
        assert hashlib.sha256(client.values[str(path)]).hexdigest() == expected_sha256
        return manifest, {"schema": DISTRIBUTED_SCHEMA}

    monkeypatch.setattr(api, "read_benchmark_manifest", manifest_record)
    monkeypatch.setattr(api, "verify_benchmark", verified)
    with pytest.raises(BenchmarkError, match="manifest changed"):
        api.load_study(study_id, registry_root=root, spark=cast(SparkSession, object()))
    assert checked == [original_sha]


@pytest.mark.parametrize("changed", [False, True])
def test_untrusted_volume_hash_uses_native_bytes_even_with_spark(volume, monkeypatch, changed):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from autoforecast import _logical_hash

    from provenance.lineage import directory_digest

    root, client = volume
    folder = root / "existing"
    folder.mkdir()
    output = io.BytesIO()
    pq.write_table(pa.table({"x": [1, 2]}), output)
    _safe_files.publish_bytes(folder, "part.parquet", output.getvalue())
    expected = _logical_hash.fingerprint(folder)
    if changed:
        replacement = io.BytesIO()
        pq.write_table(pa.table({"x": [7, 8]}), replacement)
        client.values[str(folder / "part.parquet")] = replacement.getvalue()
    # The editable manifest can claim today's physical SHA, while the mount
    # still exposes the original rows. A fresh digest is not trusted authority.
    physical = directory_digest(folder, suffixes=(".parquet",))
    fingerprint = _logical_hash.fingerprint
    seen = []

    def private_download(path, *, spark=None):
        assert spark is None
        assert not _volume_files.is_volume(path)
        seen.append(path)
        return fingerprint(path)

    monkeypatch.setattr(_logical_hash, "fingerprint", private_download)
    options = dict(physical_sha256=physical, logical=expected, trusted=False, spark=object())
    if changed:
        with pytest.raises(ValueError, match="logical identity changed"):
            _logical_hash.verify_files(folder, **options)
    else:
        assert _logical_hash.verify_files(folder, **options) == expected["sha256"]
    assert len(seen) == 1


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("native_storage", [False, True])
def test_delta_capture_cleans_native_files_and_preserves_primary_failure(
    volume, monkeypatch, outcome, cleanup_fails, native_storage,
):
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq
    from autoforecast.integrations.delta import DeltaStudyStore

    from provenance.lineage import directory_digest

    root, client = volume
    if not native_storage:
        monkeypatch.setattr(_volume_files, "is_volume", lambda path: False)
    source = root / "source"
    source.mkdir()
    output = io.BytesIO()
    pq.write_table(pa.table({"item": ["A", "B"], "value": [1, 2]}), output)
    _safe_files.publish_bytes(source, "part.parquet", output.getvalue())
    expected = directory_digest(source, suffixes=(".parquet",))
    original = dict(client.values)
    store = DeltaStudyStore(cast(SparkSession, object()), namespace="example")
    primary = RuntimeError("freeze failed") if outcome == "error" else KeyboardInterrupt()
    captured = []

    def freeze(path, *, keys, scratch_root):
        captured.append(path)
        assert keys == ["item"] and path.parent == scratch_root
        assert directory_digest(path, suffixes=(".parquet",)) == expected
        # This emulator deliberately keeps native files absent from the mount.
        assert (path / "part.parquet").exists() is not native_storage
        if outcome != "success":
            raise primary
        return {"status": "captured"}

    monkeypatch.setattr(store, "_freeze_captured_observations", freeze)
    remove = _volume_files.remove_tree
    cleaned = []

    def cleanup(path):
        cleaned.append(path)
        if cleanup_fails:
            raise OSError("snapshot cleanup failed")
        remove(path)

    monkeypatch.setattr(_volume_files, "remove_tree", cleanup)
    filesystem_remove = shutil.rmtree

    def filesystem_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith("af-delta-source-"):
            if native_storage:
                raise OSError("mounted cleanup must not run after native cleanup")
            cleaned.append(Path(path))
            if cleanup_fails:
                raise OSError("snapshot cleanup failed")
        return filesystem_remove(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", filesystem_cleanup)
    if outcome != "success":
        with pytest.raises(type(primary)) as caught:
            store.freeze_observations(source, keys=["item"], scratch_root=root)
        assert caught.value is primary
        if cleanup_fails:
            assert "snapshot cleanup failed" in " ".join(primary.__notes__)
    elif cleanup_fails:
        with pytest.raises(OSError, match="snapshot cleanup failed"):
            store.freeze_observations(source, keys=["item"], scratch_root=root)
    else:
        assert store.freeze_observations(source, keys=["item"], scratch_root=root) == {
            "status": "captured",
        }
    assert len(captured) == 1 and cleaned == [captured[0].parent]
    # The native API is authoritative even while its mounted listing is stale.
    if not native_storage and not cleanup_fails:
        assert not captured[0].parent.exists()
    assert _safe_files.read_regular_bytes(source, "part.parquet") == output.getvalue()
    assert {key: client.values[key] for key in original} == original
    if not cleanup_fails:
        assert client.values == original


def test_segmented_audit_never_rewrites_prior_events_and_reopens_locally(volume, monkeypatch):
    import json

    from autoforecast import read_execution_events
    from autoforecast._execution import Execution, _stage_event

    root, client = volume
    audit = root / 'execution.jsonl'
    monkeypatch.setattr(_volume_files, 'append_bytes', lambda *a: pytest.fail('rewrote audit'))
    with Execution(audit, []):
        for index in range(20):
            _stage_event(audit, 'work', 'INFO', completed=index)
    assert len(client.requests) == 23  # Initial lock, descriptor, events and closure.
    assert all(not overwrite for _, overwrite in client.requests)
    rows = read_execution_events(audit)
    assert [r['completed'] for r in rows] == list(range(20))
    assert [r['audit_segment']['sequence'] for r in rows] == list(range(20))
    assert max(map(len, client.values.values())) < 1024
    # Copying a completed bundle keeps its event reader usable offline.
    for name, value in client.values.items():
        local = Path(name)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(value)
    monkeypatch.setattr(_volume_files, 'is_volume', lambda p: False)
    assert read_execution_events(audit) == rows
    ordinary = root / 'local.jsonl'
    ordinary.write_text(json.dumps({'status': 'PASS'}) + '\n')
    assert read_execution_events(ordinary) == [{'status': 'PASS'}]


def test_segmented_audit_lost_acknowledgement_reuses_exact_bytes(volume, monkeypatch):
    from autoforecast import read_execution_events
    from autoforecast._execution import Execution, _stage_event

    root, client = volume
    download = client.download
    failed = []

    def lost_ack(path):
        if '.events/' in path and not failed:
            failed.append(path)
            raise OSError('lost read acknowledgement')
        return download(path)

    monkeypatch.setattr(client, 'download', lost_ack)
    audit = root / 'execution.jsonl'
    with Execution(audit, []):
        _stage_event(audit, 'work', 'PASS')
    assert len(read_execution_events(audit)) == 1
    assert len(client.requests) == 4  # Initial lock, descriptor, event and closure.
    assert failed


def test_segmented_audit_retry_finishes_pending_event_before_next_event(volume, monkeypatch):
    from autoforecast import read_execution_events
    from autoforecast._execution import Execution

    root, client = volume
    audit = root / 'execution.jsonl'
    write = _volume_files._put
    blocked = True

    def unavailable(path, contents, *, overwrite):
        if '.events/' in str(path) and blocked:
            raise OSError('connection unavailable')
        return write(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, '_put', unavailable)
    execution = Execution(audit, [])
    with pytest.raises(OSError):
        execution.write_event(audit, {'status': 'START', 'timestamp': 'original'})
    pending = execution.audit_segments[audit]['pending']
    blocked = False
    execution.write_event(audit, {'status': 'FAIL', 'timestamp': 'later'})
    rows = read_execution_events(audit)
    assert [r['timestamp'] for r in rows] == ['original', 'later']
    assert pending in client.values.values()
    with pytest.raises(OSError, match='conflicts'):
        Execution(audit, []).write_event(audit, {'status': 'START'})
    assert len(read_execution_events(audit)) == 2


@pytest.mark.parametrize('corruption', ['bytes', 'missing', 'duplicate', 'owner'])
def test_segmented_audit_rejects_corrupted_sequence(volume, corruption):
    import json

    from autoforecast import read_execution_events
    from autoforecast._execution import Execution

    root, client = volume
    audit = root / 'execution.jsonl'
    writer = Execution(audit, [])
    for status in ['START', 'INFO', 'PASS']:
        writer.write_event(audit, {'status': status})
    segments = sorted(p for p in client.values if '.events/' in p)
    if corruption == 'bytes':
        client.values[segments[1]] += b'changed'
    elif corruption == 'missing':
        del client.values[segments[1]]
    elif corruption == 'duplicate':
        duplicate = segments[1].replace('000000000001-', '000000000000-')
        client.values[duplicate] = client.values[segments[1]]
    else:
        descriptor = json.loads(client.values[str(audit)])
        descriptor['owner'] = 'different'
        client.values[str(audit)] = json.dumps(descriptor).encode()
    with pytest.raises(ValueError, match='audit'):
        read_execution_events(audit)


def test_closed_audit_detects_lost_tail_and_refuses_more_events(volume):
    from autoforecast import read_execution_events
    from autoforecast._execution import Execution

    root, client = volume
    audit = root / 'events.jsonl'
    with Execution(audit, []) as writer:
        writer.write_event(audit, {'status': 'START'})
        writer.write_event(audit, {'status': 'PASS'})
    with pytest.raises(ValueError, match='closed'):
        writer.write_event(audit, {'status': 'INFO'})
    last = sorted(p for p in client.values if '.events/' in p and not p.endswith('closed.json'))[-1]
    del client.values[last]
    with pytest.raises(ValueError, match='closure'):
        read_execution_events(audit)


def test_segmented_audit_serializes_threads_and_rejects_unowned_append(volume):
    from autoforecast import read_execution_events
    from autoforecast._execution import Execution

    root, client = volume
    audit = root / 'events.jsonl'
    with Execution(audit, []) as writer:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda n: writer.write_event(audit, {'item': n}), range(20)))
        original = client.values[str(audit)]
        with pytest.raises(OSError, match='owning execution'):
            _volume_files.append_bytes(audit, b'not an owned event\n')
        assert client.values[str(audit)] == original
    rows = read_execution_events(audit)
    assert sorted(row['item'] for row in rows) == list(range(20))
    assert [row['audit_segment']['sequence'] for row in rows] == list(range(20))


def audit_bundle(root, client, count=1):
    audit = root / "events.jsonl"
    with events.Execution(audit, []) as writer:
        for n in range(count):
            writer.write_event(audit, {"item": n})
    return audit


@pytest.mark.parametrize("defect", ["extra", "owner_type"])
def test_descriptor_shape_is_checked(volume, defect):
    root, client = volume
    audit = audit_bundle(root, client, 0)
    # A zero-event execution creates no descriptor; use a deliberately empty valid bundle.
    client.values[str(audit)] = json.dumps({"schema": "autoforecast.audit-segments.v1",
                                          "owner": "expected"}).encode()
    audit.with_name(audit.name + ".events").mkdir()
    descriptor = json.loads(client.values[str(audit)])
    if defect == "extra":
        descriptor["unexpected"] = True
    else:
        descriptor["owner"] = 1
    client.values[str(audit)] = json.dumps(descriptor).encode()
    with pytest.raises(ValueError, match="descriptor"):
        read_execution_events(audit)


@pytest.mark.parametrize("field,value", [("previous", "unrelated"), ("sequence", 17)])
def test_event_links_are_independently_checked(volume, field, value):
    root, client = volume
    audit = root / "events.jsonl"
    events.Execution(audit, []).write_event(audit, {"item": 0})
    part = next(p for p in client.values if ".events/" in p)
    event = json.loads(client.values.pop(part))
    event["audit_segment"][field] = value
    payload = (json.dumps(event) + "\n").encode()
    target = str(pathlib.Path(part).with_name(
        "000000000000-" + hashlib.sha256(payload).hexdigest() + ".json",
    ))
    client.values[target] = payload
    with pytest.raises(ValueError, match="sequence or writer"):
        read_execution_events(audit)


@pytest.mark.parametrize("field,value", [
    ("owner", "other"), ("count", 9), ("count", True),
    ("last", "other"), ("status", "SUCCESS"), ("extra", True),
])
def test_every_closure_field_is_independently_checked(volume, field, value):
    root, client = volume
    audit = audit_bundle(root, client)
    closed = str(audit.with_name(audit.name + ".events") / "closed.json")
    terminal = json.loads(client.values[closed])
    terminal[field] = value
    client.values[closed] = json.dumps(terminal).encode()
    with pytest.raises(ValueError, match="closure"):
        read_execution_events(audit)


def test_descriptor_is_rechecked_after_read(volume, monkeypatch):
    root, client = volume
    audit = audit_bundle(root, client)
    read = _volume_files.read_bytes
    count = 0
    def changing(path):
        nonlocal count
        payload = read(path)
        if path == audit:
            count += 1
            if count == 1:
                client.values[str(path)] = payload + b" "
        return payload
    monkeypatch.setattr(_volume_files, "read_bytes", changing)
    with pytest.raises(ValueError, match="descriptor changed"):
        read_execution_events(audit)


@pytest.mark.parametrize("outcome", ["FAIL", "CANCELLED", "RESOLVED", "INCOMPLETE", "PASS"])
def test_closure_status_matches_execution_outcome(volume, outcome):
    root, client = volume
    audit = root / "events.jsonl"
    stages = ["work"]
    failure = RuntimeError("primary") if outcome == "FAIL" else KeyboardInterrupt()
    if outcome in {"FAIL", "CANCELLED"}:
        with pytest.raises(type(failure)) as caught:
            with events.Execution(audit, stages):
                with events._stage(audit, "work"):
                    raise failure
        assert caught.value is failure
    else:
        with events.Execution(audit, stages):
            status = {"PASS": "PASS", "RESOLVED": "SKIP", "INCOMPLETE": "INFO"}[outcome]
            events._stage_event(audit, "work", status)
    closed = str(audit.with_name(audit.name + ".events") / "closed.json")
    assert json.loads(client.values[closed])["status"] == outcome


def test_running_execution_uploads_events_without_waiting(volume, monkeypatch):
    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put
    release = threading.Event()

    def slow_event(path, contents, *, overwrite):
        name = str(path).split("/")[-1]
        if ".events/" in str(path) and name != "closed.json" and not name.startswith("0" * 12):
            assert release.wait(10)
        return put(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", slow_event)
    with events.Execution(audit, []) as writer:
        writer.write_event(audit, {"item": 0})  # Claims the audit before returning.
        for n in range(1, 20):
            writer.write_event(audit, {"item": n})  # Must not wait for the blocked uploads.
        assert not any(p.split("/")[-1].startswith("000000000019-") for p in client.values)
        release.set()
    rows = read_execution_events(audit)
    assert [row["item"] for row in rows] == list(range(20))
    closed = str(audit.with_name(audit.name + ".events") / "closed.json")
    assert json.loads(client.values[closed])["count"] == 20


@pytest.mark.parametrize("persistent", [False, True])
def test_failed_background_upload_retries_identical_bytes(volume, monkeypatch, persistent):
    from contextlib import suppress

    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put
    failures = []

    def flaky(path, contents, *, overwrite):
        if str(path).split("/")[-1].startswith("000000000002-") and (persistent or not failures):
            failures.append(path)
            raise OSError("connection unavailable")
        return put(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", flaky)
    if persistent:
        with pytest.raises(OSError, match="unacknowledged"):
            with events.Execution(audit, []) as writer:
                for n in range(4):
                    with suppress(OSError):  # A later event may see the retry fail first.
                        writer.write_event(audit, {"item": n})
        assert not any(p.endswith("closed.json") for p in client.values)
        return
    with events.Execution(audit, []) as writer:
        for n in range(4):
            writer.write_event(audit, {"item": n})
    assert failures
    assert [row["item"] for row in read_execution_events(audit)] == list(range(4))


def test_other_readers_see_a_valid_prefix_while_uploads_are_running(volume, monkeypatch):
    import time

    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put
    release = threading.Event()

    def slow_second_event(path, contents, *, overwrite):
        if str(path).split("/")[-1].startswith("000000000002-"):
            assert release.wait(10)
        return put(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", slow_second_event)
    with events.Execution(audit, []) as writer:
        for n in range(6):
            writer.write_event(audit, {"item": n})
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not any(
            p.split("/")[-1].startswith("000000000005-") for p in client.values
        ):
            time.sleep(0.01)
        # A reader in another process does not flush this execution's uploads.
        rows = [json.loads(line) for line in
                _volume_files.read_audit_bytes(audit).splitlines()]
        assert [row["item"] for row in rows] == list(range(len(rows)))
        release.set()
    assert [row["item"] for row in read_execution_events(audit)] == list(range(6))


def test_interruption_after_submission_keeps_a_consistent_closed_audit(volume, monkeypatch):
    root, client = volume
    audit = root / "events.jsonl"
    with pytest.raises(KeyboardInterrupt):
        with events.Execution(audit, []) as writer:
            assert writer.uploader is not None
            submit = writer.uploader.submit
            calls = []

            def interrupted(*args, **kwargs):
                future = submit(*args, **kwargs)
                calls.append(future)
                if len(calls) == 2:
                    raise KeyboardInterrupt("after the upload was queued")
                return future

            monkeypatch.setattr(writer.uploader, "submit", interrupted)
            for n in range(5):
                writer.write_event(audit, {"item": n})
    rows = read_execution_events(audit)
    assert [row["item"] for row in rows] == [0, 1, 2]
    closed = str(audit.with_name(audit.name + ".events") / "closed.json")
    assert json.loads(client.values[closed])["count"] == 3


def test_reader_inside_execution_sees_its_own_queued_events(volume):
    root, _ = volume
    audit = root / "events.jsonl"
    with events.Execution(audit, []) as writer:
        for n in range(6):
            writer.write_event(audit, {"item": n})
        assert [row["item"] for row in read_execution_events(audit)] == list(range(6))


def test_interruption_at_any_line_of_an_event_write_leaves_a_valid_closed_audit(volume):
    import sys
    from contextlib import suppress

    root, client = volume
    code = events.Execution.write_event.__code__
    lines = sorted({line for _start, _end, line in code.co_lines() if line is not None})
    for line in lines:
        audit = root / f"interrupted-{line}.jsonl"

        def trace(frame, action, _arg, line=line):
            if frame.f_code is code and action == "line" and frame.f_lineno == line:
                sys.settrace(None)
                raise KeyboardInterrupt(f"line {line}")
            return trace

        with suppress(KeyboardInterrupt):  # A line this path does not reach writes normally.
            with events.Execution(audit, []) as writer:
                writer.write_event(audit, {"item": 0})
                writer.write_event(audit, {"item": 1})
                sys.settrace(trace)
                try:
                    writer.write_event(audit, {"item": 2})
                finally:
                    sys.settrace(None)
        rows = read_execution_events(audit)
        closed = json.loads(client.values[str(audit.with_name(audit.name + ".events")
                                              / "closed.json")])
        assert [row["item"] for row in rows] == list(range(len(rows))), line
        assert closed["count"] == len(rows) >= 2, line


def test_failed_upload_keeps_later_events_unsaved_until_it_is_retried(volume, monkeypatch):
    from contextlib import suppress

    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put
    started, release = threading.Event(), threading.Event()

    def failing(path, contents, *, overwrite):
        name = str(path).split("/")[-1]
        if name.startswith("000000000001-"):
            started.set()
            assert release.wait(10)
        if name.startswith("000000000002-"):
            raise OSError("persistent upload error")
        return put(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", failing)
    with pytest.raises(OSError, match="unacknowledged"):
        with events.Execution(audit, []) as writer:
            for n in range(6):
                writer.write_event(audit, {"item": n})
            assert started.wait(10)
            release.set()
            for _payload, future in list(writer.audit_segments[audit]["uploads"].values()):
                with suppress(OSError):
                    future.result(timeout=10)
            # Event 2 failed, so events 3-5 must not be saved after the gap.
            saved = {p.split("/")[-1][:12] for p in client.values if ".events/" in p}
            assert saved == {"000000000000", "000000000001"}
            rows = [json.loads(line) for line in
                    _volume_files.read_audit_bytes(audit).splitlines()]
            assert [row["item"] for row in rows] == [0, 1]
    assert not any(p.endswith("closed.json") for p in client.values)


def test_audit_writer_that_cannot_be_created_installs_no_context(volume, monkeypatch):
    root, _ = volume
    before = events._CURRENT.get()
    failure = RuntimeError("cannot create the audit writer")

    def cannot_create(*args, **kwargs):
        raise failure

    monkeypatch.setattr(events, "ThreadPoolExecutor", cannot_create)
    with pytest.raises(RuntimeError) as caught:
        with events.Execution(root / "events.jsonl", []):
            pytest.fail("entry must fail")
    assert caught.value is failure
    assert events._CURRENT.get() is before


def test_failed_heartbeat_start_keeps_its_error_when_cleanup_is_interrupted(volume, monkeypatch):
    root, _ = volume
    before = events._CURRENT.get()
    failure = RuntimeError("heartbeat start failed")
    create = events.ThreadPoolExecutor

    def interrupted_cleanup(*args, **kwargs):
        pool = create(*args, **kwargs)
        shutdown = pool.shutdown

        def interrupted(*args, **kwargs):
            shutdown(*args, **kwargs)
            raise KeyboardInterrupt("cleanup interrupted")

        pool.shutdown = interrupted
        return pool

    def failed_start(self):
        raise failure

    monkeypatch.setattr(events, "ThreadPoolExecutor", interrupted_cleanup)
    monkeypatch.setattr(events, "_console", lambda: SimpleNamespace(interval=10))
    monkeypatch.setattr(threading.Thread, "start", failed_start)
    with pytest.raises(RuntimeError) as caught:
        with events.Execution(root / "events.jsonl", []):
            pytest.fail("entry must fail")
    assert caught.value is failure
    assert events._CURRENT.get() is before


@pytest.mark.parametrize("primary", [None, ValueError("body"), KeyboardInterrupt("body")],
                         ids=["none", "error", "interrupt"])
def test_interrupted_writer_shutdown_restores_context_and_keeps_the_primary(
    volume, monkeypatch, primary,
):
    root, _ = volume
    before = events._CURRENT.get()
    secondary = KeyboardInterrupt("shutdown interrupted")
    with pytest.raises(BaseException) as caught:
        with events.Execution(root / "events.jsonl", []) as writer:
            writer.write_event(writer.audit, {"item": 0})
            assert writer.uploader is not None
            shutdown = writer.uploader.shutdown

            def interrupted(*args, **kwargs):
                shutdown(*args, **kwargs)
                raise secondary

            monkeypatch.setattr(writer.uploader, "shutdown", interrupted)
            if primary is not None:
                raise primary
    assert caught.value is (primary or secondary)
    assert events._CURRENT.get() is before


def test_interrupting_the_wait_for_an_upload_propagates(volume, monkeypatch):
    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put
    release = threading.Event()

    def slow(path, contents, *, overwrite):
        if str(path).split("/")[-1].startswith("000000000001-"):
            assert release.wait(10)
        return put(path, contents, overwrite=overwrite)

    def interrupted_wait(*args, **kwargs):
        release.set()
        raise KeyboardInterrupt("interrupted while waiting")

    monkeypatch.setattr(_volume_files, "_put", slow)
    monkeypatch.setattr(events, "futures_wait", interrupted_wait)
    with pytest.raises(KeyboardInterrupt, match="interrupted while waiting"):
        with events.Execution(audit, []) as writer:
            writer.write_event(audit, {"item": 0})
            writer.write_event(audit, {"item": 1})
    # Not closed: what was saved stays a valid prefix.
    assert not any(p.endswith("closed.json") for p in client.values)
    rows = [json.loads(line) for line in _volume_files.read_audit_bytes(audit).splitlines()]
    assert [row["item"] for row in rows] == [0, 1]


def test_interrupted_retry_at_close_keeps_its_type(volume, monkeypatch):
    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put
    interruption, calls = KeyboardInterrupt("retry interrupted"), []

    def failing(path, contents, *, overwrite):
        if str(path).split("/")[-1].startswith("000000000001-"):
            calls.append(path)
            raise OSError("worker upload failed") if len(calls) == 1 else interruption
        return put(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", failing)
    with pytest.raises(KeyboardInterrupt) as caught:
        with events.Execution(audit, []) as writer:
            writer.write_event(audit, {"item": 0})
            writer.write_event(audit, {"item": 1})
    assert caught.value is interruption and len(calls) == 2
    assert not any(p.endswith("closed.json") for p in client.values)


def test_upload_failure_found_at_close_marks_the_run_unverified(volume, monkeypatch):
    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put

    def failing(path, contents, *, overwrite):
        if str(path).split("/")[-1].startswith("000000000001-"):
            raise OSError("remote unavailable")
        return put(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", failing)
    with pytest.raises(OSError, match="unacknowledged"):
        with events.Execution(audit, ["work"]) as writer:
            events._stage_event(audit, "work", "START")
            events._stage_event(audit, "work", "PASS")  # Saved in the background.
    assert writer.result == "UNVERIFIED" and writer.audit_error
    assert f"Audit write failed: {writer.audit_error}" in writer.render()
    assert not any(p.endswith("closed.json") for p in client.values)


def _fail_upload(monkeypatch, stage, status, error=OSError):
    """Fail every upload of one event, which is saved in the background; others save."""
    publish = _volume_files.publish_audit_segment

    def unavailable(path, payload, **kwargs):
        event = json.loads(payload)
        if (event["stage"], event["status"]) == (stage, status):
            raise error(f"{stage} event unavailable")
        return publish(path, payload, **kwargs)

    monkeypatch.setattr(_volume_files, "publish_audit_segment", unavailable)


@pytest.mark.parametrize("level", [logging.INFO, logging.ERROR])
@pytest.mark.parametrize("options", [
    {"display": "logs"}, {"display": "progress"}, {"display": "both"}, {"progress": "off"},
])
def test_upload_failure_found_at_close_is_shown_for_every_display(volume, monkeypatch,
                                                                  options, level):
    root, _ = volume
    logger = logging.getLogger("autoforecast.stages")
    monkeypatch.setattr(logger, "handlers", [])  # configure_logging adds to this list
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)
    monkeypatch.setattr(events, "_DISPLAY", None)
    stream = io.StringIO()
    events.configure_logging(stream=stream, **options)
    _fail_upload(monkeypatch, "work", "PASS")
    with pytest.raises(OSError, match="unacknowledged"):
        with events.Execution(root / "events.jsonl", ["work"], level=level, study_id="input"):
            events._stage_event(root / "events.jsonl", "work", "START")
            events.record_result(study_id="saved", workflow_path="saved-workflow")
            events._stage_event(root / "events.jsonl", "work", "PASS")
    # The saved result stays visible, as the failed event line showed it before.
    assert "UNVERIFIED" in stream.getvalue()
    assert "saved-workflow" in stream.getvalue()
    assert "input_study_id" in stream.getvalue()  # The run's input study keeps its own name.


# An interruption while waiting for the last upload keeps the details, as a failure does.
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_completed_run_keeps_its_recovery_fields_when_the_last_upload_fails(volume,
                                                                            monkeypatch, error):
    import contextlib
    import dataclasses

    from autoforecast import study_workflow

    root, client = volume
    workflow = root / "completed-workflow"
    workflow.mkdir()
    (workflow / "artifact_manifest.json").write_text("{}")
    (root / "model.json").write_text("{}")

    @dataclasses.dataclass
    class Result:
        path: Path
        study_id: str = "registered"
        run_id: str = "completed"
        status: str = "completed"

    @contextlib.contextmanager
    def load_study(_):
        yield SimpleNamespace(registration={"sha256": "a" * 64})

    registry = SimpleNamespace(root=root, runtime={}, store=None, load_study=load_study,
                               _registration_path=lambda _: None)
    monkeypatch.setattr(study_workflow, "run_forecast_workflow", lambda *a, **k: Result(workflow))
    _fail_upload(monkeypatch, "run", "PASS", error)
    with pytest.raises(error, match="run event unavailable") as caught:
        study_workflow.run_study("registered", cast(Any, lambda _: None), registry=registry,
                                 model_config=root / "model.json", implementation={})
    recovery = vars(caught.value)
    assert (recovery["workflow_path"], recovery["study_id"], recovery["registered_study_id"]) == (
        workflow, "registered", "registered")
    assert recovery["audit_path"] is not None
    assert not any(p.endswith("closed.json") for p in client.values)


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_completed_registration_keeps_its_evidence_note_when_the_last_upload_fails(
        volume, monkeypatch, error):
    from autoforecast import study_registration as registration
    from autoforecast.test_execution_boundaries import _Registry
    from autoforecast.test_study_registration import configuration

    root, _ = volume
    registry = _Registry(root=root / "registry", scratch_root=root / "scratch", runtime={},
                         store=None)
    monkeypatch.setattr(registration, "_register_study", lambda *a, **k: "a" * 64)
    _fail_upload(monkeypatch, "Study registered", "INFO", error)
    with pytest.raises(error, match="Study registered event unavailable") as caught:
        registration.register_study(configuration(root / "input"), registry=cast(Any, registry))
    assert any(note.startswith("Registration attempt evidence: ")
               for note in getattr(caught.value, "__notes__", []))


def test_download_progress_advances_while_a_large_file_is_still_arriving(volume, monkeypatch):
    import contextlib
    import time

    root, client = volume
    folder = root / "large"
    folder.mkdir()
    client.values[str(folder / "part.parquet")] = b"x" * (2 * 1024 * 1024)
    stalled, release, updates = threading.Event(), threading.Event(), []

    class Stalling(io.BytesIO):
        def read(self, size=-1):
            if self.tell() == 1024 * 1024:
                stalled.set()
                assert release.wait(10)
            return super().read(size)

    @contextlib.contextmanager
    def recorded(*args, **kwargs):
        yield lambda **state: updates.append(state)

    def copy():
        with _volume_files.local_snapshot(folder):
            pass

    monkeypatch.setattr(client, "download", lambda path: SimpleNamespace(
        contents=Stalling(client.values[path])))
    monkeypatch.setattr(_volume_files, "operation", recorded)
    with ThreadPoolExecutor(max_workers=1) as pool:
        copied = pool.submit(copy)
        try:
            assert stalled.wait(10)
            deadline = time.monotonic() + 5
            while not updates and time.monotonic() < deadline:
                time.sleep(0.01)
            during = list(updates)
        finally:
            release.set()
        copied.result(timeout=10)
    assert during and during[-1]["completed"] == 1024 * 1024 and during[-1]["files"] == 0
    assert updates[-1] == {"completed": 2 * 1024 * 1024, "total": 2 * 1024 * 1024,
                           "unit": "bytes", "files": 1}


def test_volume_snapshot_downloads_files_concurrently(volume, monkeypatch):
    root, client = volume
    folder = root / "parts"
    folder.mkdir()
    for n in range(2):
        client.values[str(folder / f"part-{n}.parquet")] = b"part %d" % n
    together = threading.Barrier(2, timeout=5)
    download = client.download

    def overlapping(path):
        together.wait()  # Breaks unless both downloads are in flight at once.
        return download(path)

    monkeypatch.setattr(client, "download", overlapping)
    with _volume_files.local_snapshot(folder) as copy:
        assert sorted(p.read_bytes() for p in copy.iterdir()) == [b"part 0", b"part 1"]


def test_pending_event_prevents_closure(volume, monkeypatch):
    root, client = volume
    audit = root / "events.jsonl"
    put = _volume_files._put
    def fail_event(path, contents, *, overwrite):
        if str(path).endswith(".json") and not str(path).endswith("closed.json"):
            raise OSError("unavailable")
        return put(path, contents, overwrite=overwrite)
    monkeypatch.setattr(_volume_files, "_put", fail_event)
    with pytest.raises(OSError, match="unacknowledged"):
        with events.Execution(audit, []) as writer:
            with pytest.raises(OSError):
                writer.write_event(audit, {"item": 1})
    assert not any(p.endswith("closed.json") for p in client.values)


@pytest.mark.parametrize("after_commit", [False, True])
def test_closure_failure_is_visible_and_context_restored(volume, monkeypatch, after_commit):
    root, _ = volume
    audit = root / "events.jsonl"
    finish = _volume_files.finish_audit
    error = OSError("closure unavailable")
    def unavailable(*args, **kwargs):
        if after_commit:
            finish(*args, **kwargs)
        raise error
    monkeypatch.setattr(_volume_files, "finish_audit", unavailable)
    with pytest.raises(OSError) as caught:
        with events.Execution(audit, []) as writer:
            writer.write_event(audit, {"item": 1})
    assert caught.value is error
    assert events._CURRENT.get() is None
    assert writer.thread is None or not writer.thread.is_alive()


@pytest.mark.parametrize("after_commit", [False, True])
def test_closure_failure_preserves_primary(volume, monkeypatch, after_commit):
    root, _ = volume
    audit = root / "events.jsonl"
    finish = _volume_files.finish_audit
    error = RuntimeError("primary")
    def unavailable(*args, **kwargs):
        if after_commit:
            finish(*args, **kwargs)
        raise OSError("closure unavailable")
    monkeypatch.setattr(_volume_files, "finish_audit", unavailable)
    with pytest.raises(RuntimeError) as caught:
        with events.Execution(audit, []) as writer:
            writer.write_event(audit, {"item": 1})
            raise error
    assert caught.value is error
    assert events._CURRENT.get() is None
    assert writer.thread is None or not writer.thread.is_alive()


def test_child_segments_remain_sealed_by_normal_bundle_completion(volume):
    from mlops.run_bundle import load_completed_run_bundle, open_run_bundle
    root, client = volume
    bundle = open_run_bundle(root / "bundles", run_id="check", task="verification")
    child = bundle.path / "events.jsonl"
    with events.Execution(root / "attempt.jsonl", []):
        events._stage_event(child, "work", "START")
        events._stage_event(child, "work", "PASS")
        bundle.finalize(status="completed", summary={})
        assert load_completed_run_bundle(bundle.path)["status"] == "completed"
    assert load_completed_run_bundle(bundle.path)["status"] == "completed"
    assert len(read_execution_events(child)) == 2
    last = sorted(p for p in client.values if str(child) + ".events/" in p)[-1]
    del client.values[last]
    with pytest.raises(ValueError):
        load_completed_run_bundle(bundle.path)


@pytest.mark.parametrize("member", ["event", "closure"])
@pytest.mark.parametrize("value", [None, [], 1, "invalid"])
def test_audit_records_require_json_objects(volume, member, value):
    root, client = volume
    audit = audit_bundle(root, client)
    directory = str(audit) + ".events/"
    if member == "closure":
        client.values[directory + "closed.json"] = json.dumps(value).encode()
    else:
        part = next(p for p in client.values if p.startswith(directory)
                    and not p.endswith("closed.json"))
        del client.values[part]
        payload = json.dumps(value).encode()
        name = "000000000000-" + hashlib.sha256(payload).hexdigest() + ".json"
        client.values[directory + name] = payload
    with pytest.raises(ValueError, match="sequence or writer|closure"):
        read_execution_events(audit)


@pytest.mark.parametrize("schema", ["missing", "autoforecast.audit-segments.v2", "typo",
                                    None, True, 0, 1.5, [], {},
                                    ("invalid-json", b"{broken"),
                                    ("scalar", b"false"), ("array", b"[]")])
@pytest.mark.parametrize("operation", ["read", "read_offline", "append"])
def test_segment_sidecar_refuses_invalid_descriptor(volume, schema, operation, monkeypatch):
    root, client = volume
    audit = root / "audit.jsonl"
    with events.Execution(audit, []) as writer:
        writer.write_event(audit, {"stage": "work", "status": "PASS"})
    descriptor = json.loads(client.values[str(audit)])
    if isinstance(schema, tuple):
        client.values[str(audit)] = schema[1]
    else:
        if schema == "missing":
            descriptor.pop("schema")
        else:
            descriptor["schema"] = schema
        client.values[str(audit)] = json.dumps(descriptor).encode()
    if operation == "read_offline":
        for name, payload in client.values.items():
            target = type(audit)(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        monkeypatch.setattr(_volume_files, "is_volume", lambda path: False)
    if operation.startswith("read"):
        with pytest.raises(ValueError):
            read_execution_events(audit)
    else:
        before = dict(client.values)
        with pytest.raises(OSError, match="owning execution"):
            _volume_files.append_bytes(audit, b'\n{"stage":"foreign","status":"PASS"}\n')
        assert client.values == before


@pytest.mark.parametrize("status", [None, True, 1, 1.5, "", "SUCCESS", [], {},
                                    ["PASS"], {"status": "PASS"}])
def test_audit_closure_rejects_invalid_status_types(volume, status):
    root, client = volume
    audit = root / "audit.jsonl"
    with events.Execution(audit, []) as writer:
        writer.write_event(audit, {"stage": "work", "status": "PASS"})
    closed = str(audit) + ".events/closed.json"
    terminal = json.loads(client.values[closed])
    terminal["status"] = status
    client.values[closed] = json.dumps(terminal).encode()
    with pytest.raises(ValueError):
        read_execution_events(audit)


def test_audit_formats_share_initial_writer_ownership(volume, monkeypatch):
    root, client = volume
    audit = root / "race.jsonl"
    put = _volume_files._put
    attempts = []
    failures = []

    def race_at_overwrite(path, contents, *, overwrite):
        if path == audit and overwrite and not attempts:
            attempts.append(True)
            # A second thread can enter here after the legacy writer checked the
            # path, while that writer still owns its remote .lock. Execute that
            # precise interleaving without depending on thread scheduling.
            try:
                with events.Execution(audit, []):
                    events._stage_event(audit, "segment-writer", "PASS")
            except OSError as error:
                failures.append(str(error))
        return put(path, contents, overwrite=overwrite)

    monkeypatch.setattr(_volume_files, "_put", race_at_overwrite)
    events._stage_event(audit, "legacy-writer", "PASS")
    assert attempts
    assert len(failures) == 1, (
        "both writers acknowledged success; the legacy overwrite erased the "
        f"immutable descriptor: {client.values[str(audit)]!r}"
    )
    assert [row["stage"] for row in read_execution_events(audit)] == ["legacy-writer"]


@pytest.mark.parametrize("operation", ["raw_append", "execution"])
@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("pretty", [False, True])
@pytest.mark.parametrize("encoding", [
    "utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be",
    "utf-32", "utf-32-le", "utf-32-be",
])
@pytest.mark.parametrize("prefix", ["", " \t\r\n"])
def test_offline_segmented_audit_cannot_be_appended(
    volume, monkeypatch, operation, sidecar, pretty, encoding, prefix,
):
    from mlops import _safe_files

    root, client = volume
    audit = root / "offline.jsonl"
    with events.Execution(audit, []):
        events._stage_event(audit, "original", "PASS")
    for name, payload in client.values.items():
        if not sidecar and name != str(audit):
            continue
        target = type(audit)(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    if not sidecar:
        # The fake mount created this empty directory during the remote write.
        # A descriptor-only copy also represents interruption before its mkdir.
        audit.with_name(audit.name + ".events").rmdir()
    monkeypatch.setattr(_volume_files, "is_volume", lambda path: False)
    descriptor = json.dumps(json.loads(audit.read_bytes()), indent=2 if pretty else None)
    audit.write_bytes((prefix + descriptor).encode(encoding))
    before = audit.read_bytes()
    write_error = None
    try:
        if operation == "raw_append":
            _safe_files.append_audit_bytes(audit, b'{"stage":"foreign","status":"PASS"}\n')
        else:
            with events.Execution(audit, []):
                events._stage_event(audit, "foreign", "PASS")
    except OSError as error:
        write_error = str(error)
    try:
        read_execution_events(audit)
        read_error = None
    except ValueError as error:
        read_error = str(error)
    assert write_error is not None, (
        f"append acknowledged success, descriptor_changed={audit.read_bytes() != before}, "
        f"reader={read_error!r}"
    )
    assert audit.read_bytes() == before
    assert [row["stage"] for row in read_execution_events(audit)] == (
        ["original"] if sidecar else []
    )


def test_local_append_inspects_one_event_in_a_growing_jsonl_log(tmp_path, monkeypatch):
    import os

    path = tmp_path / "audit.jsonl"
    first = b'{"stage":"first","status":"START"}\n'
    history = first + b'{"stage":"later","status":"INFO"}\n' * 100000
    path.write_bytes(history)
    open_file, inspected = os.fdopen, []

    class Counted:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def readline(self):
            value = self.stream.readline()
            inspected.append(len(value))
            return value

    monkeypatch.setattr(os, "fdopen", lambda *a, **k: Counted(open_file(*a, **k)))
    last = b'{"stage":"last","status":"PASS"}\n'
    _safe_files.append_audit_bytes(path, last)
    assert inspected == [len(first)]
    assert path.read_bytes() == history + last


def _volume_attempt(root, files):
    from autoforecast.experiments import open_local_attempt
    from autoforecast.test_experiments import _metadata

    attempt = open_local_attempt(
        root, study_id="smoke", trial_id="trial1", attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    for name, value in files.items():
        attempt.write_bytes(name, value)
    return attempt


def test_volume_attempt_is_checked_from_one_owned_copy(volume, monkeypatch):
    from autoforecast.experiments import ExperimentError, load_completed_attempt

    root, client = volume
    attempt = _volume_attempt(root, {f"model/part{n}.json": b"x" * (n + 1) for n in range(12)})
    attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
    downloads = []
    download = client.download

    def counted(path):
        downloads.append(path)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    load_completed_attempt(attempt.path)
    files = [key for key in client.values if key.startswith(str(attempt.path) + "/")]
    # Every file is read exactly once; the second sweep and per-file probes use the copy.
    assert sorted(downloads) == sorted(files)
    client.values[str(attempt.path / "model/part3.json")] = b"yyyy"  # Same size.
    with pytest.raises(ExperimentError, match="does not match bytes"):
        load_completed_attempt(attempt.path)


def test_volume_manifest_hashes_files_concurrently(volume, monkeypatch):
    from autoforecast.experiments import load_completed_attempt

    root, client = volume
    attempt = _volume_attempt(root, {f"model/part{n}.json": b"x" * (n + 1) for n in range(4)})
    barrier = threading.Barrier(2, timeout=10)
    download = client.download

    def paired(path):
        # One file at a time would leave each model read waiting here until the timeout.
        if "/model/" in path:
            barrier.wait()
        return download(path)

    monkeypatch.setattr(client, "download", paired)
    attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
    monkeypatch.setattr(client, "download", download)
    assert load_completed_attempt(attempt.path)[1]["status"] == "completed"


def test_volume_artifact_retention_reads_the_source_once(volume, monkeypatch):
    from autoforecast.hpo import HPOError
    from autoforecast.hpo_bridge import _retain_run_artifact_directory

    from provenance.lineage import directory_digest

    root, client = volume
    source, destination = root / "scratch/model", root / "attempt/windows/0000/model"
    for number in range(6):
        client.values[str(source / f"series{number}.json")] = b"state" * (number + 1)
    expected = directory_digest(source)
    downloads = []
    download = client.download

    def counted(path):
        downloads.append(path)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    assert _retain_run_artifact_directory(source, destination, changed="changed") == expected
    sources = [path for path in downloads if path.startswith(str(source) + "/")]
    assert sorted(sources) == sorted(k for k in client.values if k.startswith(str(source) + "/"))
    for number in range(6):
        name = f"series{number}.json"
        assert client.values[str(destination / name)] == client.values[str(source / name)]
    # A copy that leaves out a file must not return the source digest.
    publish = _volume_files.publish_files
    monkeypatch.setattr(_volume_files, "publish_files", lambda pairs: publish(list(pairs)[1:]))
    with pytest.raises(HPOError, match="incomplete copy"):
        _retain_run_artifact_directory(source, root / "other", changed="incomplete copy")


def test_volume_parquet_snapshot_reads_source_once_and_skips_destination_rehash(
    volume, monkeypatch,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from provenance.lineage import directory_digest, snapshot_parquet_directory

    root, client = volume
    source, destination = root / "actuals", root / "scratch/actuals"
    destination.parent.mkdir()
    for index in range(3):
        output = io.BytesIO()
        pq.write_table(pa.table({"x": [index]}), output)
        client.values[str(source / f"part-{index:05d}.parquet")] = output.getvalue()
    expected = directory_digest(source, suffixes=(".parquet",))
    downloads = []
    download = client.download

    def counted(path):
        downloads.append(path)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    snapshot_parquet_directory(source, destination, expected_sha256=expected, label="actuals")
    # The source once, and each published part once, when it is read back.
    assert sorted(downloads) == sorted(
        [str(source / f"part-{i:05d}.parquet") for i in range(3)]
        + [str(destination / f"part-{i:05d}.parquet") for i in range(3)]
    )
    assert directory_digest(destination, suffixes=(".parquet",)) == expected


def test_volume_bundle_export_checks_one_owned_copy(volume, monkeypatch):
    from autoforecast import run_mlflow

    from mlops.run_bundle import open_run_bundle

    root, client = volume
    bundle = open_run_bundle(root, run_id="exported", task="verification")
    for number in range(5):
        bundle.write_bytes(f"models/part{number}.json", b"m" * (number + 1))
    bundle.finalize(status="completed", summary={})
    downloads = []
    download = client.download

    def counted(path):
        downloads.append(path)
        return download(path)

    monkeypatch.setattr(client, "download", counted)
    with run_mlflow._completed_run_snapshot(bundle.path) as (snapshot, completion):  # noqa: SLF001
        assert completion["run_id"] == "exported"
        assert (snapshot / "models/part2.json").read_bytes() == b"mmm"
    files = [key for key in client.values if key.startswith(str(bundle.path) + "/")]
    assert sorted(downloads) == sorted(files)
    client.values[str(bundle.path / "models/part2.json")] = b"xxx"
    with pytest.raises(run_mlflow.RunMLflowError, match="not verified"):
        with run_mlflow._completed_run_snapshot(bundle.path):  # noqa: SLF001
            pass
