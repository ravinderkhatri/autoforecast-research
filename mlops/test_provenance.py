"""Functional checks for task-agnostic reproducibility records."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import urllib.parse

import pytest
from autoforecast.experiments import load_completed_attempt, open_local_attempt

import mlops._safe_files as safe_files
from mlops._safe_files import (
    SafeFileError,
    copy_regular_file,
    publish_bytes,
    publish_bytes_set,
    read_regular_bytes,
    snapshot_regular_files,
)
from mlops.provenance import (
    CheckpointCapability,
    DataSnapshot,
    ProvenanceError,
    capture_runtime_identity,
    capture_source_identity,
    local_data_reference,
    sanitize_repository_origin,
    sanitize_web_url,
    validate_data_reference,
    verify_retained_ref,
    write_run_provenance,
)
from mlops.run_bundle import load_completed_run_bundle, open_run_bundle


def _run(root: pathlib.Path, *args: str) -> None:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)


def _metadata() -> dict[str, object]:
    return {
        "comparability_fingerprint": "a" * 64,
        "hyperparameters": {"training.learning_rate": 0.001},
        "data": {
            "name": "functional-data",
            "fingerprint": "b" * 64,
            "observed_date_range": ["2025-01-01", "2025-01-10"],
            "training_date_range": ["2025-01-01", "2025-01-07"],
            "validation_date_range": ["2025-01-01", "2025-01-07"],
            "test_date_range": ["2025-01-08", "2025-01-10"],
            "validation_target_date_range": ["2025-01-06", "2025-01-07"],
            "test_target_date_range": ["2025-01-08", "2025-01-10"],
            "training_series": 2,
            "validation_series": 1,
            "test": "reserved_not_evaluated",
        },
    }


def test_safe_file_publish_bytes_is_exclusive_and_descriptor_relative(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()

    published = publish_bytes(root, "nested/value.bin", b"first")

    assert published.read_bytes() == b"first"
    with pytest.raises(SafeFileError, match="artifact path|could not publish immutable artifact"):
        publish_bytes(root, "nested/value.bin", b"replacement")
    assert published.read_bytes() == b"first"


def test_safe_file_copy_regular_file_refuses_a_linked_source(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    linked = tmp_path / "linked.bin"
    linked.symlink_to(source)

    copied = copy_regular_file(root, "copied/source.bin", source)

    assert copied.read_bytes() == b"source"
    with pytest.raises(SafeFileError, match="regular non-link file"):
        copy_regular_file(root, "copied/linked.bin", linked)
    assert not (root / "copied/linked.bin").exists()


def test_snapshot_regular_files_copies_only_the_declared_file_set(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested/included.bin").write_bytes(b"included")
    (source / "excluded.bin").write_bytes(b"excluded")

    destination = snapshot_regular_files(
        source,
        tmp_path / "snapshot",
        ["nested/included.bin"],
    )

    assert (destination / "nested/included.bin").read_bytes() == b"included"
    assert not (destination / "excluded.bin").exists()


def test_snapshot_tree_types_records_regular_directory_and_link_types(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    directory = root / "directory"
    directory.mkdir()
    regular = root / "regular.bin"
    regular.write_bytes(b"value")
    linked = root / "linked.bin"
    linked.symlink_to(regular)

    observed = safe_files.snapshot_tree_types(root)

    assert observed[directory] == "directory"
    assert observed[regular] == "file"
    assert observed[linked] == "symlink"


def test_regular_file_identity_refuses_a_symlinked_leaf(tmp_path: pathlib.Path) -> None:
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"value")
    linked = tmp_path / "linked.bin"
    linked.symlink_to(regular)

    with pytest.raises(SafeFileError, match="immutable artifact"):
        safe_files._regular_file_identity(linked)  # noqa: SLF001


def test_snapshot_pins_the_source_root_for_the_entire_file_set(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.bin").write_bytes(b"original-a")
    (source / "b.bin").write_bytes(b"original-b")
    original_open = safe_files._open_regular_at  # noqa: SLF001
    calls = 0

    def swap_source_after_first_open(root_fd: int, relative_path: str) -> int:
        nonlocal calls
        descriptor = original_open(root_fd, relative_path)
        calls += 1
        if calls == 1:
            source.rename(tmp_path / "original-source")
            source.mkdir()
            (source / "b.bin").write_bytes(b"replacement-b")
        return descriptor

    monkeypatch.setattr(safe_files, "_open_regular_at", swap_source_after_first_open)
    destination = snapshot_regular_files(source, tmp_path / "snapshot", ["a.bin", "b.bin"])

    assert (destination / "a.bin").read_bytes() == b"original-a"
    assert (destination / "b.bin").read_bytes() == b"original-b"


def test_snapshot_refuses_a_destination_root_replaced_during_publication(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.bin").write_bytes(b"a")
    (source / "b.bin").write_bytes(b"b")
    destination = tmp_path / "snapshot"
    displaced = tmp_path / "displaced-snapshot"
    original_publish = safe_files._publish_blocks_at  # noqa: SLF001
    calls = 0

    def replace_destination_after_first_publish(
        root_fd: int, relative_path: str, blocks: object
    ) -> None:
        nonlocal calls
        original_publish(root_fd, relative_path, blocks)  # type: ignore[arg-type]
        calls += 1
        if calls == 1:
            destination.rename(displaced)
            destination.mkdir()

    monkeypatch.setattr(safe_files, "_publish_blocks_at", replace_destination_after_first_publish)
    with pytest.raises(SafeFileError, match="destination changed"):
        snapshot_regular_files(source, destination, ["a.bin", "b.bin"])

    assert list(destination.iterdir()) == []
    assert (displaced / "a.bin").read_bytes() == b"a"
    assert (displaced / "b.bin").read_bytes() == b"b"


def test_snapshot_refuses_a_linked_destination_ancestor(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.bin").write_bytes(b"value")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SafeFileError, match="create or publish"):
        snapshot_regular_files(source, linked / "snapshot", ["value.bin"])

    assert not (outside / "snapshot").exists()


def test_safe_file_publication_refuses_linked_root_and_source_ancestors(
    tmp_path: pathlib.Path,
) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(SafeFileError, match="link-free directory hierarchy"):
        publish_bytes(linked_root, "escaped.bin", b"must not publish")
    assert not (real_root / "escaped.bin").exists()

    source_root = tmp_path / "source-root"
    source_root.mkdir()
    source = source_root / "value.bin"
    source.write_bytes(b"source")
    linked_source_root = tmp_path / "linked-source-root"
    linked_source_root.symlink_to(source_root, target_is_directory=True)
    with pytest.raises(SafeFileError, match="regular non-link file"):
        copy_regular_file(real_root, "copied.bin", linked_source_root / source.name)
    assert not (real_root / "copied.bin").exists()


def test_read_regular_bytes_uses_a_link_free_root(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    publish_bytes(root, "value.bin", b"sealed")
    assert read_regular_bytes(root, "value.bin") == b"sealed"

    linked = tmp_path / "linked-artifacts"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(SafeFileError, match="immutable artifact"):
        read_regular_bytes(linked, "value.bin")


def test_read_regular_bytes_refuses_a_file_changed_during_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    value = root / "value.bin"
    value.write_bytes(b"sealed")
    real_fstat = safe_files.os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        observed = real_fstat(descriptor)
        calls += 1
        if calls == 3:
            value.write_bytes(b"changed")
            observed = real_fstat(descriptor)
        return observed

    monkeypatch.setattr(safe_files.os, "fstat", changed_fstat)
    with pytest.raises(SafeFileError, match="changed while it was read"):
        read_regular_bytes(root, "value.bin")


def test_publish_bytes_set_refuses_a_root_replaced_between_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    displaced = tmp_path / "displaced"
    real_publish = safe_files._publish_blocks_at  # noqa: SLF001
    calls = 0

    def publish_then_swap(root_fd: int, relative_path: str, blocks: object) -> None:
        nonlocal calls
        real_publish(root_fd, relative_path, blocks)  # type: ignore[arg-type]
        calls += 1
        if calls == 1:
            root.rename(displaced)
            root.mkdir()

    monkeypatch.setattr(safe_files, "_publish_blocks_at", publish_then_swap)
    with pytest.raises(SafeFileError, match="root changed"):
        publish_bytes_set(root, {"one.bin": b"one", "two.bin": b"two"})

    assert (displaced / "one.bin").read_bytes() == b"one"
    assert (displaced / "two.bin").read_bytes() == b"two"
    assert list(root.iterdir()) == []


def test_run_bundle_writer_refuses_a_symlinked_destination_parent(
    tmp_path: pathlib.Path,
) -> None:
    bundle = open_run_bundle(tmp_path / "bundles", run_id="safe-write", task="forecast")
    outside = tmp_path / "outside"
    outside.mkdir()
    (bundle.path / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProvenanceError, match="could not write bundle artifact"):
        bundle.write_json("linked/escaped.json", {"secret": False})

    assert not (outside / "escaped.json").exists()


@pytest.mark.parametrize(
    "reference",
    (
        "https://user:secret@example.test/data",
        "https://example.test/data?signature=secret",
        "https://example.test/data#access-token",
        "local://dataset\\credential",
        "token@example.test:team/repo",
    ),
)
def test_data_references_refuse_embedded_credentials(reference: str) -> None:
    with pytest.raises(ProvenanceError, match="credentials or signed query"):
        validate_data_reference(reference)


def test_data_reference_and_dashboard_url_normalize_malformed_authorities() -> None:
    with pytest.raises(ProvenanceError, match="invalid authority"):
        validate_data_reference("https://[unclosed/data")
    for malformed in (
        "https:////user:secret@example.test/data",
        "https:///user:secret@example.test/data",
        "https:///@user:secret@example.test/data",
        "s3:////access:secret@bucket/key",
    ):
        with pytest.raises(ProvenanceError, match="hierarchical"):
            validate_data_reference(malformed)
    assert sanitize_web_url("https://[unclosed/mlflow") is None
    assert validate_data_reference("dataset@1") == "dataset@1"


def _repository(root: pathlib.Path) -> None:
    (root / "autoforecast").mkdir()
    (root / "configs").mkdir()
    (root / "autoforecast" / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs" / "model.yaml").write_text("model: demo\n", encoding="utf-8")
    (root / "configs" / "run.yaml").write_text("run: demo\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _run(root, "init")
    _run(root, "config", "user.name", "Functional Test")
    _run(root, "config", "user.email", "test@example.test")
    _run(root, "add", ".")
    _run(root, "commit", "-m", "initial")


def test_source_identity_strips_credentials_and_queries_from_git_origin(
    tmp_path: pathlib.Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _repository(repository)
    _run(
        repository,
        "remote",
        "add",
        "origin",
        "https://token:secret@example.test/team/repo.git?signature=private",
    )

    source = capture_source_identity(repository)

    assert source.origin_url == "https://example.test/team/repo.git"


def test_source_identity_sanitizes_scp_and_ipv6_git_origins(tmp_path: pathlib.Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _repository(repository)
    _run(repository, "remote", "add", "origin", "token@example.test:team/repo.git")
    scp_origin = capture_source_identity(repository).origin_url
    assert scp_origin == "example.test:team/repo.git"
    assert sanitize_repository_origin(scp_origin) == scp_origin
    _run(repository, "remote", "set-url", "origin", "https://token@[2001:db8::1]:8443/x.git?q=s")
    assert capture_source_identity(repository).origin_url == "https://[2001:db8::1]:8443/x.git"


@pytest.mark.parametrize(
    "host",
    ("file", "ftp", "ftps", "git", "http", "https", "ssh", "GIT", "HtTpS"),
)
def test_sanitize_repository_origin_keeps_scheme_named_scp_hosts_idempotent(
    host: str,
) -> None:
    canonical = f"{host}:team/repo.git"

    assert sanitize_repository_origin(f"alice@{canonical}") == canonical
    assert sanitize_repository_origin(canonical) == canonical


def test_sanitize_repository_origin_rejects_malformed_secret_bearing_urls() -> None:
    assert sanitize_repository_origin("https:///user:secret@example.test/repo.git") is None
    assert sanitize_repository_origin("https:////user:secret@example.test/repo.git") is None
    assert sanitize_repository_origin("https:user:secret@example.test/repo.git") is None
    assert sanitize_repository_origin("https://exa mple.test/repo.git") is None
    assert sanitize_repository_origin("https://example.test/%ZZ") is None
    assert sanitize_repository_origin("https://exa{mple}.test/repo.git") is None
    assert sanitize_repository_origin("https://example.test/{repo}.git") is None
    assert sanitize_repository_origin("https://example.test/|repo.git") is None
    assert sanitize_repository_origin("") is None
    assert sanitize_repository_origin("   ") is None
    assert sanitize_repository_origin("://example.test/repo.git") is None


def test_dashboard_url_retains_only_a_credential_free_web_base() -> None:
    assert (
        sanitize_web_url("https://user:secret@[2001:db8::1]:8443/mlflow?q=s#token")
        == "https://[2001:db8::1]:8443/mlflow"
    )
    assert sanitize_web_url("sqlite:///tracking.db") is None


def test_clean_git_identity_and_provenance_are_captured_without_copying_source(
    tmp_path: pathlib.Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _repository(repository)
    (repository / "data").mkdir()
    (repository / "data" / "raw.parquet").write_bytes(b"untracked data")
    source = capture_source_identity(repository)
    _run(repository, "tag", "-a", "autoforecast/run-functional", "-m", "retained")
    verify_retained_ref(source, "refs/tags/autoforecast/run-functional")
    runtime = capture_runtime_identity(repository)
    attempt = open_local_attempt(
        tmp_path / "runs",
        study_id="functional",
        trial_id="demo",
        attempt_id="one",
        metadata=_metadata(),
    )
    resolved_payload = b"resolved: true\n"
    write_run_provenance(
        attempt,
        source=source,
        runtime=runtime,
        data=DataSnapshot("m5_r1", "c" * 64, "m5_daily@release_1"),
        configuration_files={
            "model": repository / "configs/model.yaml",
            "run": repository / "configs/run.yaml",
        },
        configuration_payloads={"resolved": (".yaml", resolved_payload)},
        lockfile_source=repository / "uv.lock",
        checkpoint=CheckpointCapability(
            "inference_only",
            "model/model.pt",
            "d" * 64,
            missing_resume_state=("optimizer_state", "rng_state"),
        ),
    )
    attempt.finalize(status="completed", summary={})
    metadata, _completion = load_completed_attempt(attempt.path)
    assert metadata["study_id"] == "functional"
    assert not (attempt.path / "provenance" / "source.tar.zst").exists()
    assert (attempt.path / "provenance/config/model.yaml").read_text() == "model: demo\n"
    assert (attempt.path / "provenance/config/resolved.yaml").read_text() == "resolved: true\n"
    manifest = json.loads((attempt.path / "provenance/manifest.json").read_text())
    assert manifest["configuration"]["resolved"] == {
        "path": "provenance/config/resolved.yaml",
        "sha256": hashlib.sha256(resolved_payload).hexdigest(),
    }
    assert (attempt.path / "provenance/runtime/uv.lock").read_text() == "version = 1\n"
    assert "optimizer_state" in (attempt.path / "provenance/checkpoint.json").read_text()


def test_relevant_dirty_code_is_rejected_while_untracked_data_is_not(
    tmp_path: pathlib.Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _repository(repository)
    (repository / "data").mkdir()
    (repository / "data" / "scratch.parquet").write_bytes(b"not source")
    assert capture_source_identity(repository).commit
    (repository / "configs" / "model.yaml").write_text("model: changed\n", encoding="utf-8")
    try:
        capture_source_identity(repository)
    except ProvenanceError as error:
        assert "configs/model.yaml" in str(error)
    else:
        raise AssertionError("official identity accepted a dirty model configuration")


def test_resume_capability_is_not_claimed_for_weights_only_checkpoints() -> None:
    try:
        CheckpointCapability("inference_only", "model.pt", "e" * 64)
    except ProvenanceError as error:
        assert "state why" in str(error)
    else:
        raise AssertionError("weights-only checkpoint silently became resume-capable")
    assert CheckpointCapability("resume_capable", "state.pt", "f" * 64).kind == "resume_capable"


def test_common_bundle_can_capture_an_llm_or_supervised_run_without_forecast_fields(
    tmp_path: pathlib.Path,
) -> None:
    bundle = open_run_bundle(tmp_path, run_id="llm-finetune-1", task="llm_finetune")
    bundle.write_json("metrics/result.json", {"loss": 0.42})
    bundle.finalize(status="completed", summary={"metric": "loss"})
    completion = bundle.path / "completion.json"
    assert completion.is_file()
    assert "llm_finetune" in (bundle.path / "metadata/run.json").read_text()
    assert load_completed_run_bundle(bundle.path)["status"] == "completed"


def test_completed_bundle_terminal_payload_is_sealed(tmp_path: pathlib.Path) -> None:
    bundle = open_run_bundle(tmp_path, run_id="sealed-terminal", task="forecast")
    bundle.finalize(status="completed", summary={"score": 1})
    completion_path = bundle.path / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = "failed"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ProvenanceError, match="terminal payload is not sealed"):
        load_completed_run_bundle(bundle.path)


def test_completed_bundle_rejects_non_string_status(tmp_path: pathlib.Path) -> None:
    bundle = open_run_bundle(tmp_path, run_id="invalid-status", task="forecast")
    bundle.finalize(status="completed", summary={})
    completion_path = bundle.path / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = []
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ProvenanceError, match="terminal records are invalid"):
        load_completed_run_bundle(bundle.path)


def test_completed_bundle_identity_must_match_run_definition(tmp_path: pathlib.Path) -> None:
    bundle = open_run_bundle(tmp_path, run_id="original-run", task="forecast")
    bundle.finalize(status="completed", summary={})
    completion_path = bundle.path / "completion.json"
    manifest_path = bundle.path / "artifact_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion["run_id"] = "substitute-run"
    terminal = {
        key: value for key, value in completion.items() if key != "artifact_manifest_sha256"
    }
    terminal_bytes = (json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode()
    manifest["terminal_payload_sha256"] = hashlib.sha256(terminal_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion["artifact_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ProvenanceError, match="identity differs"):
        load_completed_run_bundle(bundle.path)


def test_completed_bundle_task_must_match_run_definition(tmp_path: pathlib.Path) -> None:
    bundle = open_run_bundle(tmp_path, run_id="task-bound", task="forecast")
    bundle.finalize(status="completed", summary={})
    definition_path = bundle.path / "metadata/run.json"
    manifest_path = bundle.path / "artifact_manifest.json"
    completion_path = bundle.path / "completion.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    definition["task"] = "language"
    definition_path.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n")
    entry = next(item for item in manifest["artifacts"] if item["path"] == "metadata/run.json")
    entry["sha256"] = hashlib.sha256(definition_path.read_bytes()).hexdigest()
    entry["bytes"] = definition_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion["artifact_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ProvenanceError, match="identity differs"):
        load_completed_run_bundle(bundle.path)


def test_completed_bundle_rejects_legacy_unsealed_manifest(tmp_path: pathlib.Path) -> None:
    bundle = open_run_bundle(tmp_path, run_id="legacy-bundle", task="forecast")
    bundle.finalize(status="completed", summary={})
    manifest_path = bundle.path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "mlops.artifact-manifest.v1"
    manifest.pop("terminal_payload_sha256")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ProvenanceError, match="legacy run bundle.*must be rerun"):
        load_completed_run_bundle(bundle.path)


@pytest.mark.parametrize("leaf", ("panel#one", "panel?one", "panel%one", "panel one"))
def test_local_data_reference_percent_encodes_filesystem_path(
    tmp_path: pathlib.Path, leaf: str
) -> None:
    panel = tmp_path / leaf
    reference = local_data_reference(panel, authority="parquet")
    assert validate_data_reference(reference) == reference
    parsed = urllib.parse.urlsplit(reference)
    assert (parsed.scheme, parsed.netloc) == ("local", "parquet")
    assert urllib.parse.unquote(parsed.path) == panel.resolve().as_posix()


def test_completed_bundle_rejects_manifest_artifact_replaced_by_symlink(
    tmp_path: pathlib.Path,
) -> None:
    bundle = open_run_bundle(tmp_path / "bundles", run_id="linked-run", task="forecast")
    artifact = bundle.write_json("provenance/source.json", {"commit": "a" * 40})
    bundle.finalize(status="completed", summary={})
    external = tmp_path / "external.json"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)
    with pytest.raises(ProvenanceError, match="symlinks"):
        load_completed_run_bundle(bundle.path)


def test_run_bundle_creation_refuses_a_dangling_link_at_the_claimed_leaf(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "bundles"
    root.mkdir()
    claimed = root / "claimed"
    claimed.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(ProvenanceError, match="already exists"):
        open_run_bundle(root, run_id="claimed", task="forecast")


def test_run_bundle_creation_refuses_a_linked_hierarchy_ancestor(
    tmp_path: pathlib.Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-bundles"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProvenanceError, match="cannot create run bundle"):
        open_run_bundle(linked, run_id="claimed", task="forecast")

    assert list(outside.iterdir()) == []


def test_completed_bundle_rejects_link_swapped_during_final_enumeration(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = open_run_bundle(tmp_path / "bundles", run_id="swap", task="forecast")
    artifact = bundle.write_json("metrics/result.json", {"value": 1})
    bundle.finalize(status="completed", summary={})
    external = tmp_path / "external.json"
    external.write_bytes(artifact.read_bytes())
    real_rglob = pathlib.Path.rglob
    calls = 0

    def swapping_rglob(path: pathlib.Path, pattern: str):
        nonlocal calls
        values = tuple(real_rglob(path, pattern))
        if path == bundle.path:
            calls += 1
            if calls == 2:
                artifact.unlink()
                artifact.symlink_to(external)
        return iter(values)

    monkeypatch.setattr(pathlib.Path, "rglob", swapping_rglob)
    with pytest.raises(ProvenanceError, match="symlinks"):
        load_completed_run_bundle(bundle.path)


def test_completed_bundle_rejects_directory_replaced_by_an_undeclared_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = open_run_bundle(tmp_path, run_id="type-swap", task="forecast")
    bundle.finalize(status="completed", summary={})
    rogue = bundle.path / "rogue"
    rogue.mkdir()
    real_rglob = pathlib.Path.rglob
    calls = 0

    def swapping_rglob(path: pathlib.Path, pattern: str):
        nonlocal calls
        values = tuple(real_rglob(path, pattern))
        if path == bundle.path:
            calls += 1
            if calls == 2:
                rogue.rmdir()
                rogue.write_text("undeclared", encoding="utf-8")
        return iter(values)

    monkeypatch.setattr(pathlib.Path, "rglob", swapping_rglob)
    with pytest.raises(ProvenanceError, match="path set or type changed"):
        load_completed_run_bundle(bundle.path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no FIFO support")
def test_completed_bundle_rejects_undeclared_special_files(tmp_path: pathlib.Path) -> None:
    bundle = open_run_bundle(tmp_path, run_id="special-file", task="forecast")
    bundle.finalize(status="completed", summary={})
    os.mkfifo(bundle.path / "undeclared.fifo")
    with pytest.raises(ProvenanceError, match="unsupported path type"):
        load_completed_run_bundle(bundle.path)


def test_bundle_rejects_unmanifested_symlinked_directories(tmp_path: pathlib.Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("not a bundle artifact", encoding="utf-8")

    unsealed = open_run_bundle(tmp_path / "unsealed", run_id="linked-dir", task="forecast")
    (unsealed.path / "model").symlink_to(external, target_is_directory=True)
    with pytest.raises(ProvenanceError, match="symlinks"):
        unsealed.finalize(status="completed", summary={})

    sealed = open_run_bundle(tmp_path / "sealed", run_id="linked-dir", task="forecast")
    sealed.finalize(status="completed", summary={})
    (sealed.path / "model").symlink_to(external, target_is_directory=True)
    with pytest.raises(ProvenanceError, match="symlinks"):
        load_completed_run_bundle(sealed.path)


def test_completed_bundle_rejects_duplicate_terminal_json_keys(tmp_path: pathlib.Path) -> None:
    bundle = open_run_bundle(tmp_path, run_id="duplicate-json", task="forecast")
    bundle.finalize(status="completed", summary={})
    (bundle.path / "completion.json").write_text(
        '{"schema":"mlops.run-bundle.v2","schema":"mlops.run-bundle.v2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="duplicate JSON key"):
        load_completed_run_bundle(bundle.path)


@pytest.mark.parametrize("mutation", ("duplicate_path", "boolean_size", "extra_completion"))
def test_completed_bundle_rejects_ambiguous_manifest_shapes(
    tmp_path: pathlib.Path, mutation: str
) -> None:
    bundle = open_run_bundle(tmp_path, run_id="ambiguous-manifest", task="forecast")
    bundle.write_json("metrics/result.json", {"value": 1})
    bundle.finalize(status="completed", summary={})
    manifest_path = bundle.path / "artifact_manifest.json"
    completion_path = bundle.path / "completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if mutation == "duplicate_path":
        manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    elif mutation == "boolean_size":
        manifest["artifacts"][0]["bytes"] = True
    else:
        completion["unexpected"] = "field"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion["artifact_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ProvenanceError, match="invalid"):
        load_completed_run_bundle(bundle.path)
