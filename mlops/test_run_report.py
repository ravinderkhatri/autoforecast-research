"""Execution naming and retained guides preserve existing integrity boundaries."""

import concurrent.futures
import datetime as dt
import json
import re
import uuid

import pytest

from mlops.provenance import ProvenanceError
from mlops.run_bundle import load_completed_run_bundle, open_run_bundle
from mlops.run_report import new_run_id, run_guide


def test_same_second_runs_have_distinct_ids_and_collision_never_overwrites(tmp_path, monkeypatch):
    import mlops.run_report as reports

    class FixedClock:
        @staticmethod
        def now(tz):
            assert tz is dt.UTC
            return dt.datetime(2026, 10, 28, 8, 15, 30, tzinfo=tz)

    monkeypatch.setattr(reports, "datetime", FixedClock)

    def create(_):
        return open_run_bundle(tmp_path, run_id=new_run_id("workflow"), task="forecast")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        bundles = list(pool.map(create, range(32)))
    assert len({bundle.path for bundle in bundles}) == 32
    assert all(re.fullmatch(r"workflow-20261028t081530z-[0-9a-f]{32}", b.run_id) for b in bundles)
    assert all(uuid.UUID(hex=b.run_id.rsplit("-", 1)[1]).version == 4 for b in bundles)
    target = bundles[0]
    before = (target.path / "metadata/run.json").read_bytes()
    fixed_uuid = uuid.UUID(hex=target.run_id.rsplit("-", 1)[1])
    monkeypatch.setattr(reports.uuid, "uuid4", lambda: fixed_uuid)
    with pytest.raises(ProvenanceError, match="already exists"):
        create(0)
    assert (target.path / "metadata/run.json").read_bytes() == before


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("owner", ["bundle", "attempt"])
def test_bundle_and_attempt_guides_are_accurate_and_sealed(tmp_path, status, owner):
    if owner == "bundle":
        bundle = open_run_bundle(tmp_path, run_id="example", task="forecast")
        load = load_completed_run_bundle
        error = ProvenanceError
    else:
        from autoforecast.experiments import (
            ExperimentError,
            load_completed_attempt,
            open_local_attempt,
        )
        from autoforecast.test_experiments import _metadata

        bundle = open_local_attempt(
            tmp_path, study_id="search", trial_id="trial1", attempt_id="attempt1",
            metadata=_metadata(learning_rate=0.1),
        )
        def load(path):
            return load_completed_attempt(path)[1]

        error = ExperimentError
    bundle.write_bytes("model/weights.bin", b"native model")
    if status == "failed":
        bundle.write_json("failure.json", {"error": "model stopped"})
    bundle.finalize(status=status, summary={})
    assert load(bundle.path)["status"] == status
    guide = (bundle.path / "RUN.md").read_text()
    headings = [line for line in guide.splitlines() if line.startswith("Execution status:")]
    assert headings == [f"Execution status: **{status}**."]
    assert "model/weights.bin" in guide
    assert "forecast.parquet" not in guide
    manifest = json.loads((bundle.path / "artifact_manifest.json").read_text())
    assert [entry["path"] for entry in manifest["artifacts"]].count("RUN.md") == 1
    (bundle.path / "RUN.md").write_text(guide + "changed")
    with pytest.raises(error, match="artifact"):
        load(bundle.path)


def test_guide_escapes_file_names_and_does_not_claim_to_seal_exports():
    guide = run_guide(identity={"run_id": "test"}, status="exported", sealed=False,
                      files=['odd|<script>".json', "with space.parquet"]).decode()
    assert '<script>' not in guide
    assert "odd&#124;&lt;script&gt;" in guide
    assert 'href="with%20space.parquet"' in guide
    assert "does not seal them" in guide
    callback = run_guide(identity={}, status="completed", files=["artifacts/RUN.md"]).decode()
    assert "explicitly retained by the model callback" in callback
    assert "File guide for this retained execution" not in callback


@pytest.mark.parametrize("prefix", ["", "../escape", "Uppercase", "space here"])
def test_run_prefix_cannot_escape_the_destination(prefix):
    with pytest.raises(ValueError, match="prefix"):
        new_run_id(prefix)


def test_guide_filename_is_reserved(tmp_path):
    bundle = open_run_bundle(tmp_path, run_id="example", task="forecast")
    bundle.write_bytes("RUN.md", b"caller guide")
    with pytest.raises(ProvenanceError, match="reserved"):
        bundle.finalize(status="completed", summary={})
    assert not (bundle.path / "completion.json").exists()


@pytest.fixture(params=["bundle", "attempt"])
def recovery_store(tmp_path, request):
    if request.param == "bundle":
        store = open_run_bundle(tmp_path, run_id="example", task="forecast")
        load = load_completed_run_bundle
    else:
        from autoforecast.experiments import load_completed_attempt, open_local_attempt
        from autoforecast.test_experiments import _metadata

        store = open_local_attempt(tmp_path, study_id="search", trial_id="trial1",
                                   attempt_id="attempt1", metadata=_metadata(learning_rate=0.1))

        def load(path):
            return load_completed_attempt(path)[1]
    store.write_bytes("model/weights.bin", b"native model")
    return store, load


def test_guide_recovery_after_manifest_publication_failure(recovery_store, monkeypatch):
    store, load = recovery_store
    original = type(store).write_json
    failure = OSError("one-shot manifest publication failure")
    injected = False

    def publish(self, relative, value):
        nonlocal injected
        if self is store and relative == "artifact_manifest.json" and not injected:
            injected = True
            raise failure
        return original(self, relative, value)

    monkeypatch.setattr(type(store), "write_json", publish)
    with pytest.raises(OSError) as caught:
        store.finalize(status="completed", summary={})
    assert caught.value is failure
    assert not (store.path / "RUN.md").exists()
    assert not (store.path / "artifact_manifest.json").exists()
    assert (store.path / "model/weights.bin").read_bytes() == b"native model"
    store.write_json("failure.json", {"error": str(failure)})
    store.finalize(status="failed", summary={})
    assert load(store.path)["status"] == "failed"
    guide = (store.path / "RUN.md").read_text()
    assert "Execution status: **failed**" in guide and "failure.json" in guide


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_invalid_terminal_summary_does_not_publish_guide(recovery_store, value):
    store, _ = recovery_store
    with pytest.raises((ValueError, TypeError)):
        store.finalize(status="completed", summary={"invalid": value})
    assert not (store.path / "RUN.md").exists()
    assert not (store.path / "artifact_manifest.json").exists()


def test_caller_guide_is_never_owned_by_finalizer(recovery_store):
    store, _ = recovery_store
    store.write_bytes("RUN.md", b"caller guide")
    with pytest.raises(ValueError, match="reserved"):
        store.finalize(status="completed", summary={})
    assert (store.path / "RUN.md").read_bytes() == b"caller guide"


@pytest.mark.parametrize("change", ["contents", "symlink", "directory", "root_symlink"])
def test_failed_guide_cleanup_preserves_unowned_paths(
    recovery_store, monkeypatch, tmp_path, change,
):
    store, _ = recovery_store
    original = type(store).write_json
    target = tmp_path / "other"
    target.mkdir()
    (target / "RUN.md").write_bytes(b"other guide")
    failure = OSError("manifest publication failed")

    def publish(self, relative, value):
        if self is store and relative == "artifact_manifest.json":
            guide = store.path / "RUN.md"
            if change == "root_symlink":
                store.path.rename(store.path.with_name("moved"))
                store.path.symlink_to(target, target_is_directory=True)
            else:
                guide.unlink()
                if change == "contents":
                    guide.write_bytes(b"changed guide")
                elif change == "symlink":
                    guide.symlink_to(target / "RUN.md")
                else:
                    guide.mkdir()
            raise failure
        return original(self, relative, value)

    monkeypatch.setattr(type(store), "write_json", publish)
    with pytest.raises(OSError) as caught:
        store.finalize(status="completed", summary={})
    assert caught.value is failure
    assert (target / "RUN.md").read_bytes() == b"other guide"
    guide = store.path / "RUN.md"
    if change == "contents":
        assert guide.read_bytes() == b"changed guide"
    elif change == "symlink":
        assert guide.is_symlink()
    elif change == "directory":
        assert guide.is_dir()
    else:
        assert store.path.is_symlink()


def test_post_manifest_failure_keeps_terminal_files(recovery_store, monkeypatch):
    store, _ = recovery_store
    original = type(store).write_json
    failure = OSError("completion publication failed")
    saved = {}

    def publish(self, relative, value):
        if self is store and relative == "completion.json":
            saved.update({name: (store.path / name).read_bytes()
                          for name in ("RUN.md", "artifact_manifest.json")})
            raise failure
        return original(self, relative, value)

    monkeypatch.setattr(type(store), "write_json", publish)
    with pytest.raises(OSError) as caught:
        store.finalize(status="completed", summary={})
    assert caught.value is failure
    assert saved and all((store.path / name).read_bytes() == value for name, value in saved.items())
    assert not (store.path / "completion.json").exists()


def test_guide_cleanup_error_preserves_primary_exception(recovery_store, monkeypatch):
    import os

    store, _ = recovery_store
    original_write, original_unlink = type(store).write_json, os.unlink
    failure = OSError("manifest publication failed")
    injected = False

    def publish(self, relative, value):
        if self is store and relative == "artifact_manifest.json":
            raise failure
        return original_write(self, relative, value)

    def unlink(path, *args, **kwargs):
        nonlocal injected
        if str(path) == "RUN.md":
            injected = True
            raise OSError("cleanup unavailable")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(store), "write_json", publish)
    monkeypatch.setattr(os, "unlink", unlink)
    with pytest.raises(OSError) as caught:
        store.finalize(status="completed", summary={})
    assert caught.value is failure
    assert injected and any("cleanup unavailable" in note for note in failure.__notes__)
    assert (store.path / "RUN.md").is_file()
