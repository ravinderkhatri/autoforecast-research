"""Volume publication uses the native no-overwrite contract, never FUSE append."""

import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
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

    store = DeltaStudyStore(
        cast(SparkSession, SimpleNamespace(sql=sql)), namespace="neutral.studies",
    )
    store.validate_destinations = lambda: None
    store.prepare(lock_path=path)
    assert len(commands) == 1 + len(store.tables)
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
    folder = root / "study"
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
    record = {"recipe": {"validation_origins": ["2024-01-05"]}, "study_id": "study",
              "dataset_sha256": hashlib.sha256(contract).hexdigest(),
              "role_paths": {"selection": "selection"}, "artifacts": {"selection": "0" * 64}}
    monkeypatch.setattr(module, "resolve_study", lambda *_: (manifest, record))

    def verified(path, sha):
        assert path == manifest and sha == hashlib.sha256(b"current manifest").hexdigest()
        return path, record

    monkeypatch.setattr(module, "verify_benchmark", verified)
    monkeypatch.setattr(module, "read_benchmark_manifest", verified)
    with tempfile.TemporaryDirectory() as scratch:
        study = module.load_study("study", registry_root=root,
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
        original = _safe_files.copy_regular_file
        count = 0

        def interrupted(*args):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("copy interrupted")
            return original(*args)

        monkeypatch.setattr(_safe_files, "copy_regular_file", interrupted)
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
    monkeypatch.setattr(module, "verify_benchmark", lambda *_: (root / "benchmark.json", {}))

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
        write = SimpleNamespace(parquet=lambda _: None)

        def select(self, *_):
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
    frozen = {"study_id": "study", "recipe": {"source_keys": ["item"]},
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
            self.write = SimpleNamespace(parquet=self.materialize)

        def materialize(self, path):
            client.values[str(Path(path) / "part.parquet")] = part
            if not stale_mount:
                Path(path).mkdir()

        def select(self, *_):
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
    registry = SimpleNamespace(root=root, spark=SimpleNamespace(
        read=SimpleNamespace(parquet=lambda _: frame)))
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
    frozen = {"study_id": "study", "recipe": {"source_keys": ["item"]},
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
    with Execution(audit, ['copy']):
        with _stage(audit, 'copy'):
            with _volume_files.local_snapshot(source.parent) as copied:
                assert (copied / source.name).read_bytes() == payload
            before = len(client.requests)
            with operation('Count loop') as update:
                start = len(client.requests)
                for count in range(100):
                    update(completed=count, total=100, unit='rows')
                assert len(client.requests) == start
            assert len(client.requests) - before == 4  # Two lifecycle appends, each with a lock.
    rows = [json.loads(line) for line in client.values[str(audit)].splitlines()]
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
    rows = [json.loads(line) for line in client.values[str(audit)].splitlines()]
    finished = next(r for r in rows if r['stage'] == 'Download snapshot for integrity checks'
                    and r['status'] == 'PASS')
    assert finished['completed'] == 1024 * 1024 + 5 and finished['files'] == 2
    assert len(rows) == 4
