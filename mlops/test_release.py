"""Functional retained-source verification for a generic completed bundle."""

from __future__ import annotations

import pathlib
import subprocess

from mlops import (
    DataSnapshot,
    capture_runtime_identity,
    capture_source_identity,
    open_run_bundle,
    verify_retained_run,
    write_run_provenance,
)


def _git(root: pathlib.Path, *args: str) -> None:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_completed_bundle_requires_its_named_source_tag_to_still_resolve(
    tmp_path: pathlib.Path,
) -> None:
    repository = tmp_path / "repository"
    (repository / "autoforecast").mkdir(parents=True)
    (repository / "configs").mkdir()
    (repository / "autoforecast" / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "configs" / "run.yaml").write_text("run: demo\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.name", "Functional")
    _git(repository, "config", "user.email", "test@example.test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "initial")
    source = capture_source_identity(repository)
    _git(repository, "tag", "-a", "autoforecast/run-functional", "-m", "retained")
    bundle = open_run_bundle(tmp_path / "runs", run_id="functional", task="supervised")
    write_run_provenance(
        bundle, source=source, runtime=capture_runtime_identity(repository),
        data=DataSnapshot("functional-data", "a" * 64, "dataset@version=1"),
        configuration_files={"run": repository / "configs" / "run.yaml"},
        lockfile_source=repository / "uv.lock",
    )
    bundle.finalize(status="completed", summary={})
    retained = verify_retained_run(
        bundle.path, retained_ref="refs/tags/autoforecast/run-functional",
    )
    assert retained.run_id == "functional" and retained.source_commit == source.commit
