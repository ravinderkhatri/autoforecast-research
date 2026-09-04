"""Functional tests for complete, immutable executable-source snapshots."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast import executable_source as subject  # noqa: E402


def _refuses(call, text: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted invalid executable source: {text}")


def _git(repository: pathlib.Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _repository() -> tuple[pathlib.Path, pathlib.Path]:
    workspace = pathlib.Path(tempfile.mkdtemp()).resolve()
    repository = workspace / "repository"
    repository.mkdir()
    for source_root in subject.SOURCE_ROOTS:
        (repository / source_root).mkdir(parents=True)
    # Keep the subprocess import inside this synthetic snapshot even when the test environment
    # has the real checkout installed editable. Namespace-only fixture roots can otherwise lose
    # to the checkout's regular packages later on sys.path.
    for package in (
        "autoforecast",
        "autoforecast/adapters",
        "forecast",
        "forecast/models",
        "forecast/models/mft",
    ):
        (repository / package / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
        (repository / package / "__init__.py").write_text("")
    for index, relative in enumerate(subject.CRITICAL_EXECUTABLE_SOURCES):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {index}\n")
    for relative in subject.SUPPORTING_EXECUTABLE_SOURCES:
        (repository / relative).write_text(f"fixture for {relative}\n")
    (repository / "forecast" / "models" / "mft" / "transitive.py").write_text(
        "VALUE = 'transitive'\n"
    )
    (repository / "autoforecast" / "adapters" / "mft.py").write_text(
        "from forecast.models.mft import transitive\nVALUE = transitive.VALUE\n"
    )
    cache = (
        repository / "forecast" / "models" / "mft" / "__pycache__" / "ignored.pyc"
    )
    cache.parent.mkdir()
    cache.write_text("raise AssertionError\n")
    generated = repository / "gates" / "generated.json"
    generated.parent.mkdir()
    generated.write_text("{}")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "source-test@example.invalid")
    _git(repository, "config", "user.name", "Source Test")
    _git(repository, "add", *subject.SOURCE_ROOTS, *subject.SUPPORTING_EXECUTABLE_SOURCES)
    _git(repository, "commit", "--quiet", "-m", "source fixture")
    return workspace, repository


def test_live_manifest_hashes_complete_tracked_python_tree_and_rejects_mutation() -> None:
    _workspace, repository = _repository()
    manifest = subject.build_live_executable_source_manifest(repository)
    assert manifest["source_roots"] == list(subject.SOURCE_ROOTS)
    assert manifest["critical_sources"] == list(subject.CRITICAL_EXECUTABLE_SOURCES)
    assert "forecast/models/mft/transitive.py" in manifest["files"]
    assert set(subject.SUPPORTING_EXECUTABLE_SOURCES) <= set(manifest["files"])
    assert "forecast/models/mft/__pycache__/ignored.pyc" not in manifest["files"]
    (repository / "forecast" / "models" / "mft" / "transitive.py").write_text(
        "VALUE = 'changed'\n"
    )
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "differs from its committed blob",
    )


def test_live_manifest_rejects_assume_unchanged_hidden_mutation_and_deletion() -> None:
    _workspace, repository = _repository()
    relative = "forecast/models/mft/transitive.py"
    source = repository / relative
    original = source.read_text()
    _git(repository, "update-index", "--assume-unchanged", relative)
    source.write_text("VALUE = 'hidden mutation'\n")
    assert _git(repository, "status", "--porcelain=v1", "--", relative) == ""
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "differs from its committed blob",
    )
    source.write_text(original)
    source.unlink()
    assert _git(repository, "status", "--porcelain=v1", "--", relative) == ""
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "source path is absent",
    )


def test_live_manifest_rejects_skip_worktree_hidden_mutation() -> None:
    _workspace, repository = _repository()
    relative = "forecast/models/mft/transitive.py"
    source = repository / relative
    _git(repository, "update-index", "--skip-worktree", relative)
    source.write_text("VALUE = 'hidden skip-worktree mutation'\n")
    assert _git(repository, "status", "--porcelain=v1", "--", relative) == ""
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "differs from its committed blob",
    )


def test_live_manifest_rejects_dirty_index_when_filesystem_matches_head() -> None:
    _workspace, repository = _repository()
    relative = "forecast/models/mft/transitive.py"
    source = repository / relative
    original = source.read_text()
    source.write_text("VALUE = 'staged only'\n")
    _git(repository, "add", relative)
    source.write_text(original)
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "source index differs from HEAD",
    )


def test_live_manifest_rejects_symlink_and_directory_type_replacements() -> None:
    workspace, repository = _repository()
    relative = "forecast/models/mft/transitive.py"
    source = repository / relative
    external = workspace / "same-bytes.py"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "source path is not a regular file",
    )

    _other_workspace, other_repository = _repository()
    other_source = other_repository / relative
    other_source.unlink()
    other_source.mkdir()
    _refuses(
        lambda: subject.build_live_executable_source_manifest(other_repository),
        "source path is not a regular file",
    )


def test_live_manifest_rejects_hardlinked_source_even_when_bytes_match() -> None:
    workspace, repository = _repository()
    source = repository / "forecast" / "models" / "mft" / "transitive.py"
    os.link(source, workspace / "external-source-alias.py")
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "must not be hard-linked",
    )


def test_live_manifest_rejects_python_symlink_committed_in_head() -> None:
    _workspace, repository = _repository()
    relative = "forecast/models/mft/committed_link.py"
    (repository / relative).symlink_to("transitive.py")
    _git(repository, "add", relative)
    _git(repository, "commit", "--quiet", "-m", "commit invalid source symlink")
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "committed Python source is not a regular blob",
    )


def test_live_manifest_rejects_new_python_until_it_is_committed() -> None:
    _workspace, repository = _repository()
    relative = "autoforecast/new_task_component.py"
    source = repository / relative
    source.write_text("VALUE = 'new task code'\n")
    _refuses(
        lambda: subject.build_live_executable_source_manifest(repository),
        "source path set differs from HEAD",
    )
    _git(repository, "add", relative)
    _git(repository, "commit", "--quiet", "-m", "add task source")
    manifest = subject.build_live_executable_source_manifest(repository)
    assert relative in manifest["files"]


def test_materialized_snapshot_is_content_addressed_complete_and_refuses_deletion() -> None:
    workspace, repository = _repository()
    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    validated = subject.validate_executable_source_snapshot(authorization)
    snapshot = pathlib.Path(validated["root"])
    assert snapshot.name == validated["manifest_sha256"]
    assert not (snapshot / "gates").exists()
    assert all((snapshot / path).is_file() for path in subject.SUPPORTING_EXECUTABLE_SOURCES)
    assert not (snapshot / "forecast" / "models" / "mft" / "__pycache__").exists()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(snapshot)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "from autoforecast.adapters import mft as ours; print(ours.VALUE); print(ours.__file__)",
        ],
        cwd=snapshot,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    value, module_path = imported.stdout.splitlines()
    assert value == "transitive"
    assert (
        pathlib.Path(module_path).resolve()
        == (snapshot / "autoforecast" / "adapters" / "mft.py").resolve()
    )
    subject.validate_executable_source_snapshot(authorization)
    transitive = snapshot / "forecast" / "models" / "mft" / "transitive.py"
    transitive.parent.chmod(0o755)
    transitive.unlink()
    transitive.parent.chmod(0o555)
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "entry tree differs",
    )


def test_snapshot_validation_requires_each_literal_support_file() -> None:
    for relative in ("pyproject.toml", "uv.lock"):
        workspace, repository = _repository()
        authorization = subject.materialize_executable_source_snapshot(
            repository, workspace / "snapshots"
        )
        root = pathlib.Path(authorization["root"])
        root.chmod(0o755)
        omitted = root / relative
        omitted.chmod(0o644)
        omitted.unlink()
        manifest = copy.deepcopy(authorization["manifest"])
        del manifest["files"][relative]
        digest = subject._canonical_sha256(manifest)  # noqa: SLF001
        target = root.parent / digest
        revised = {
            "schema": authorization["schema"],
            "root": str(target),
            "manifest": manifest,
            "manifest_sha256": digest,
        }
        stored = root / subject.SOURCE_MANIFEST_FILENAME
        stored.chmod(0o644)
        stored.write_text(json.dumps(revised, indent=2, sort_keys=True) + "\n")
        stored.chmod(0o444)
        root.rename(target)
        target.chmod(0o555)
        _refuses(
            lambda revised=revised: subject.validate_executable_source_snapshot(revised),
            "supporting executable sources are absent",
        )


def test_snapshot_rejects_unexpected_bytecode_native_extension_and_directory() -> None:
    workspace, repository = _repository()
    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    snapshot = pathlib.Path(authorization["root"])
    model_directory = snapshot / "forecast" / "models" / "mft"
    for name in ("rogue.pyc", "rogue.so"):
        model_directory.chmod(0o755)
        unexpected = model_directory / name
        unexpected.write_bytes(b"unregistered executable bytes")
        unexpected.chmod(0o444)
        model_directory.chmod(0o555)
        _refuses(
            lambda: subject.validate_executable_source_snapshot(authorization),
            "entry tree differs",
        )
        model_directory.chmod(0o755)
        unexpected.unlink()
        model_directory.chmod(0o555)

    model_directory.chmod(0o755)
    unexpected_directory = model_directory / "unregistered_package"
    unexpected_directory.mkdir(mode=0o555)
    model_directory.chmod(0o555)
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "entry tree differs",
    )


def test_snapshot_rejects_symlink_even_when_target_bytes_match() -> None:
    workspace, repository = _repository()
    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    snapshot = pathlib.Path(authorization["root"])
    source = snapshot / "forecast" / "models" / "mft" / "transitive.py"
    external = workspace / "same-snapshot-bytes.py"
    external.write_bytes(source.read_bytes())
    source.parent.chmod(0o755)
    source.unlink()
    source.symlink_to(external)
    source.parent.chmod(0o555)
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "contains a symbolic entry",
    )


def test_snapshot_rejects_symlinked_parent_and_external_hardlink() -> None:
    workspace, repository = _repository()
    external_parent = workspace / "external-snapshots"
    external_parent.mkdir()
    linked_parent = workspace / "linked-snapshots"
    linked_parent.symlink_to(external_parent, target_is_directory=True)
    _refuses(
        lambda: subject.materialize_executable_source_snapshot(repository, linked_parent),
        "must be canonical and symlink-free",
    )

    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    snapshot = pathlib.Path(authorization["root"])
    os.link(
        snapshot / "forecast" / "models" / "mft" / "transitive.py",
        workspace / "external-snapshot-alias.py",
    )
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "contains a hard-linked file",
    )


def test_materialization_reuses_complete_snapshot_and_ignores_preserved_partial_attempt() -> None:
    workspace, repository = _repository()
    parent = workspace / "snapshots"
    parent.mkdir()
    (parent / ".source.interrupted").mkdir()
    first = subject.materialize_executable_source_snapshot(repository, parent)
    second = subject.materialize_executable_source_snapshot(repository, parent)
    assert second == first
    subject.validate_executable_source_snapshot(second)


def test_materialization_preserves_prior_valid_snapshot_after_source_commit() -> None:
    workspace, repository = _repository()
    parent = workspace / "snapshots"
    first = subject.materialize_executable_source_snapshot(repository, parent)

    changed = repository / "forecast" / "models" / "mft" / "transitive.py"
    changed.write_text("VALUE = 'next committed source'\n")
    _git(repository, "add", "forecast/models/mft/transitive.py")
    _git(repository, "commit", "--quiet", "-m", "next source fixture")

    second = subject.materialize_executable_source_snapshot(repository, parent)
    assert second != first
    assert pathlib.Path(first["root"]).is_dir()
    assert pathlib.Path(second["root"]).is_dir()
    subject.validate_executable_source_snapshot(first)
    subject.validate_live_executable_source_identity(second)


def test_snapshot_rejects_writable_root_directory_and_file() -> None:
    workspace, repository = _repository()
    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    snapshot = pathlib.Path(authorization["root"])
    source_directory = snapshot / "forecast" / "models" / "mft"
    source = source_directory / "transitive.py"

    snapshot.chmod(0o755)
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "snapshot root is writable",
    )
    snapshot.chmod(0o555)
    source_directory.chmod(0o755)
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "snapshot directory is writable",
    )
    source_directory.chmod(0o555)
    source.chmod(0o644)
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "snapshot file is writable",
    )


def test_snapshot_rejects_semantically_equal_but_reserialized_authorization() -> None:
    workspace, repository = _repository()
    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    stored_manifest = pathlib.Path(authorization["root"]) / subject.SOURCE_MANIFEST_FILENAME
    stored_manifest.chmod(0o644)
    stored_manifest.write_text(json.dumps(authorization, sort_keys=True) + "\n")
    stored_manifest.chmod(0o444)
    _refuses(
        lambda: subject.validate_executable_source_snapshot(authorization),
        "stored executable source authorization differs",
    )


def test_live_identity_recheck_keeps_snapshot_valid_but_refuses_checkout_change() -> None:
    workspace, repository = _repository()
    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    subject.validate_live_executable_source_identity(authorization)
    live_transitive = repository / "forecast" / "models" / "mft" / "transitive.py"
    live_transitive.write_text("VALUE = 'changed after snapshot'\n")
    _refuses(
        lambda: subject.validate_live_executable_source_identity(authorization),
        "differs from its committed blob",
    )
    subject.validate_executable_source_snapshot(authorization)


def test_snapshot_runs_storage_hashing_without_importing_installed_mlops() -> None:
    workspace, repository = _repository()
    source = pathlib.Path(__file__).resolve().parent.parent
    # Use the real hashing and storage code, including their package imports.
    for package in ("mlops", "provenance"):
        for original in (source / package).rglob("*.py"):
            target = repository / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    # Storage progress uses this stdlib-only module inside the frozen snapshot.
    shutil.copyfile(source / "autoforecast/_execution.py",
                    repository / "autoforecast/_execution.py")
    _git(repository, "add", "mlops", "provenance", "autoforecast/_execution.py")
    _git(repository, "commit", "--quiet", "-m", "retain storage dependencies")
    authorization = subject.materialize_executable_source_snapshot(
        repository, workspace / "snapshots"
    )
    snapshot = pathlib.Path(authorization["root"])
    data = workspace / "data"
    data.mkdir()
    (data / "part.bin").write_bytes(b"frozen input")
    imported = subprocess.run(
        [sys.executable, "-B", "-c", """
import pathlib
import sys
from mlops import _volume_files
from provenance.lineage import directory_digest
assert pathlib.Path(_volume_files.__file__).resolve().is_relative_to(pathlib.Path.cwd())
print(directory_digest(sys.argv[1]))
""", str(data)],
        cwd=snapshot,
        env={**os.environ, "PYTHONPATH": str(snapshot)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    from provenance.lineage import directory_digest

    assert imported.stdout.strip() == directory_digest(data)
    subject.validate_executable_source_snapshot(authorization)
    # Storage code is part of the identity, so edits must invalidate the snapshot.
    storage = snapshot / "mlops/_volume_files.py"
    storage.chmod(0o644)
    storage.write_text("raise RuntimeError('changed storage')\n")
    storage.chmod(0o444)
    _refuses(lambda: subject.validate_executable_source_snapshot(authorization), "differs")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("PASS " + name)
