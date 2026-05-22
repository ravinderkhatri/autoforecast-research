"""Functional evidence for the local promotion-to-batch-deployment hand-off."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

import pytest

from mlops import (
    DataSnapshot,
    capture_runtime_identity,
    capture_source_identity,
    open_run_bundle,
    write_run_provenance,
)
from mlops.deployment import DeploymentError, create_local_batch_deployment
from mlops.promotion import promote_completed_run

_PARSER_LIMIT_INTEGER = "9" * 5000


def _git(root: pathlib.Path, *args: str) -> None:
    outcome = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert outcome.returncode == 0, outcome.stderr


@pytest.fixture
def approved_promotion(tmp_path: pathlib.Path):
    repo = tmp_path / "repo"
    (repo / "autoforecast").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "autoforecast/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "configs/run.yaml").write_text("name: functional\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Functional")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "tag", "-a", "autoforecast/run-functional", "-m", "retained")
    bundle = open_run_bundle(tmp_path / "runs", run_id="functional", task="forecast")
    write_run_provenance(
        bundle,
        source=capture_source_identity(repo),
        runtime=capture_runtime_identity(repo),
        data=DataSnapshot("functional", "a" * 64, "dataset@1"),
        configuration_files={"run_request": repo / "configs/run.yaml"},
        lockfile_source=repo / "uv.lock",
    )
    bundle.write_json(
        "evaluation/heldout.json",
        {
            "schema": "autoforecast.heldout-evaluation.v1",
            "objective": "heldout/mae",
            "value": 0.5,
        },
    )
    bundle.finalize(status="completed", summary={})
    promotion = promote_completed_run(
        bundle.path,
        retained_ref="refs/tags/autoforecast/run-functional",
        model_id="demo_model",
        alias="champion",
        record_root=repo / "promotions",
        repository=repo,
    )
    return repo, promotion


def test_approved_promotion_becomes_one_frozen_local_batch_hand_off(approved_promotion) -> None:
    repo, promotion = approved_promotion
    deployment = create_local_batch_deployment(
        promotion.path,
        deployment_root=repo / "deployments",
        repository=repo,
    )
    descriptor = json.loads((deployment.path / "deployment.json").read_text(encoding="utf-8"))
    assert descriptor["mode"] == "local_batch_forecast"
    assert descriptor["source"]["retained_ref"] == "refs/tags/autoforecast/run-functional"
    evidence_path = pathlib.Path(json.loads(promotion.path.read_text())["heldout_evidence_path"])
    assert descriptor["heldout_evidence"] == {
        "path": str(evidence_path),
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }
    assert (deployment.path / "config/run_request.yaml").read_text(encoding="utf-8") == (
        "name: functional\n"
    )
    with pytest.raises(DeploymentError, match="already exists"):
        create_local_batch_deployment(
            promotion.path,
            deployment_root=repo / "deployments",
            repository=repo,
        )


@pytest.mark.parametrize("fault", [
    "missing", "schema", "objective", "value", "source", "path_type", "relative_path", "object",
    "malformed_json", "invalid_utf8", "parser_integer_limit", "empty_metric", "nonfinite",
    "summary",
    "alternate", "symlink", "symlink_parent", "huge_integer",
])
def test_create_local_batch_deployment_rejects_changed_heldout_evidence(
    approved_promotion, fault, monkeypatch,
):
    repo, promotion = approved_promotion
    record = json.loads(promotion.path.read_text())
    evidence_path = pathlib.Path(record["heldout_evidence_path"])
    if fault in {"path_type", "relative_path", "summary"}:
        if fault == "summary":
            record["heldout"]["value"] = 9.0
        else:
            record["heldout_evidence_path"] = (
                None if fault == "path_type" else "evaluation/heldout.json"
            )
            if fault == "relative_path":
                monkeypatch.chdir(evidence_path.parent.parent)
        promotion.path.write_text(json.dumps(record))
    elif fault == "missing":
        evidence_path.unlink()
    elif fault == "symlink_parent":
        alias = repo / "aliased_bundle"
        alias.symlink_to(evidence_path.parent.parent, target_is_directory=True)
        record["heldout_evidence_path"] = str(alias / "evaluation" / "heldout.json")
        promotion.path.write_text(json.dumps(record))
    elif fault in {"source", "alternate", "symlink"}:
        replacement = (repo / "unrelated" / "evaluation" / "heldout.json"
                       if fault == "source" else evidence_path.with_name("alternate.json"))
        replacement.parent.mkdir(parents=True, exist_ok=True)
        replacement.write_bytes(evidence_path.read_bytes())
        if fault == "symlink":
            evidence_path.unlink()
            evidence_path.symlink_to(replacement)
            replacement = evidence_path
        record["heldout_evidence_path"] = str(replacement)
        promotion.path.write_text(json.dumps(record))
    elif fault == "invalid_utf8":
        evidence_path.write_bytes(b"\xff")
    elif fault == "parser_integer_limit":
        evidence_path.write_text(
            '{"schema":"autoforecast.heldout-evaluation.v1",'
            '"objective":"heldout/mae","value":' + _PARSER_LIMIT_INTEGER + "}"
        )
    elif fault in {"object", "malformed_json"}:
        evidence_path.write_text("[]" if fault == "object" else "{")
    else:
        evidence = json.loads(evidence_path.read_text())
        key, value = {
            "schema": ("schema", "invalid"), "objective": ("objective", "other/metric"),
            "value": ("value", True), "empty_metric": ("objective", ""),
            "nonfinite": ("value", float("inf")),
            "huge_integer": ("value", 10**400),
        }[fault]
        evidence[key] = value
        evidence_path.write_text(json.dumps(evidence))
        if fault in {"value", "empty_metric", "nonfinite", "huge_integer"}:
            # Matching malformed summaries must not mask the type/finite/name guards.
            summary_key = "metric" if key == "objective" else "value"
            record["heldout"][summary_key] = value
            promotion.path.write_text(json.dumps(record))
    with pytest.raises(DeploymentError, match="held-out"):
        create_local_batch_deployment(
            promotion.path, deployment_root=repo / "deployments", repository=repo,
        )
    assert not (repo / "deployments" / "demo_model" / "champion").exists()


def test_create_local_batch_deployment_wraps_promotion_json_integer_limit(
    approved_promotion,
):
    repo, promotion = approved_promotion
    promotion.path.write_text('{"oversized":' + _PARSER_LIMIT_INTEGER + "}")
    with pytest.raises(DeploymentError, match="cannot read promotion record"):
        create_local_batch_deployment(
            promotion.path, deployment_root=repo / "deployments", repository=repo,
        )


@pytest.mark.parametrize("competing_state", ["temporary_owned", "published"])
def test_create_local_batch_deployment_does_not_replace_concurrent_handoff(
    approved_promotion, monkeypatch, competing_state,
):
    from mlops import deployment as module

    repo, promotion = approved_promotion
    root = repo / "deployments"
    destination = root / "demo_model" / "champion"
    original = module._configurations

    def competing_handoff(bundle, target):
        if competing_state == "temporary_owned":
            # Another creator cannot own the current writer's staging directory.
            with pytest.raises(FileExistsError):
                create_local_batch_deployment(
                    promotion.path, deployment_root=root, repository=repo,
                )
        else:
            # A competing writer finishes after the outer exists check. A completed hand-off
            # is nonempty; POSIX directory rename must preserve it and refuse the new writer.
            destination.mkdir()
            (destination / "deployment.json").write_text("competing hand-off\n")
        return original(bundle, target)

    monkeypatch.setattr(module, "_configurations", competing_handoff)
    if competing_state == "temporary_owned":
        result = create_local_batch_deployment(
            promotion.path, deployment_root=root, repository=repo,
        )
        assert json.loads((result.path / "deployment.json").read_text())["mode"] == (
            "local_batch_forecast"
        )
    else:
        with pytest.raises(OSError):
            create_local_batch_deployment(
                promotion.path, deployment_root=root, repository=repo,
            )
        assert (destination / "deployment.json").read_text() == "competing hand-off\n"
    assert not (destination.parent / ".champion.tmp").exists()
