"""Functional test for a retained, held-out local promotion record."""

from __future__ import annotations

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
from mlops.promotion import PromotionError, promote_completed_run

_PARSER_LIMIT_INTEGER = "9" * 5000


def _git(root: pathlib.Path, *args: str) -> None:
    outcome = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert outcome.returncode == 0, outcome.stderr


@pytest.fixture
def completed_bundle(
    tmp_path: pathlib.Path,
):
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
    _git(
        repo, "-c", "user.name=Functional", "-c", "user.email=test@example.test",
        "commit", "-m", "initial",
    )
    _git(repo, "tag", "-a", "autoforecast/run-functional", "-m", "retained")
    source = capture_source_identity(repo)
    bundle = open_run_bundle(tmp_path / "runs", run_id="functional", task="forecast")
    write_run_provenance(
        bundle, source=source, runtime=capture_runtime_identity(repo),
        data=DataSnapshot("functional", "a" * 64, "dataset@1"),
        configuration_files={"run": repo / "configs/run.yaml"}, lockfile_source=repo / "uv.lock",
    )
    bundle.write_json("evaluation/heldout.json", {
        "schema": "autoforecast.heldout-evaluation.v1", "objective": "heldout/mae", "value": 0.5,
    })
    bundle.finalize(status="completed", summary={})
    return repo, bundle


def test_heldout_bundle_can_be_recorded_as_a_non_deploying_local_champion(completed_bundle):
    repo, bundle = completed_bundle
    record = promote_completed_run(
        bundle.path, retained_ref="refs/tags/autoforecast/run-functional", model_id="demo_model",
        alias="champion", record_root=repo / "promotions", repository=repo,
    )
    text = record.path.read_text(encoding="utf-8")
    assert "approved_local_alias_not_deployed" in text and "heldout/mae" in text


def test_promote_completed_run_never_replaces_a_concurrent_approval(completed_bundle, monkeypatch):
    from mlops import promotion as module

    repo, bundle = completed_bundle
    destination = repo / "promotions" / "demo_model" / "champion.json"
    original = module._approver

    def competing_approval(repository):
        # Deterministic interleaving: competitor finishes after the initial exists check.
        destination.write_text("existing approval\n")
        return original(repository)

    monkeypatch.setattr(module, "_approver", competing_approval)
    with pytest.raises(PromotionError, match="already exists"):
        promote_completed_run(
            bundle.path, retained_ref="refs/tags/autoforecast/run-functional",
            model_id="demo_model", alias="champion", record_root=repo / "promotions",
            repository=repo,
        )
    assert destination.read_text() == "existing approval\n"
    assert list(destination.parent.iterdir()) == [destination], "owned temporary file leaked"


def test_promote_completed_run_cleans_only_its_temporary_on_write_failure(
    completed_bundle, monkeypatch,
):
    from mlops import promotion as module

    repo, bundle = completed_bundle
    root = repo / "promotions" / "demo_model"
    root.mkdir(parents=True)
    unrelated = root / ".unrelated.tmp"
    unrelated.write_text("another writer")

    def fail_sync(_fd):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(module.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="simulated disk failure"):
        promote_completed_run(
            bundle.path, retained_ref="refs/tags/autoforecast/run-functional",
            model_id="demo_model", alias="champion", record_root=repo / "promotions",
            repository=repo,
        )
    assert list(root.iterdir()) == [unrelated]
    assert unrelated.read_text() == "another writer"


def _hpo_case(
    tmp_path: pathlib.Path,
    *,
    heldout_id: str = "heldout-0000",
    study_id: str = "hpo-functional",
):
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
    source = capture_source_identity(repo)
    study_root = tmp_path / "studies" / study_id
    source_attempt = study_root / "trials" / "trial-0000" / "attempts" / "attempt-0000"
    source_attempt.mkdir(parents=True)
    bundle = open_run_bundle(source_attempt / "source-run", run_id="functional", task="forecast")
    write_run_provenance(
        bundle, source=source, runtime=capture_runtime_identity(repo),
        data=DataSnapshot("functional", "a" * 64, "dataset@1"),
        configuration_files={"run": repo / "configs/run.yaml"}, lockfile_source=repo / "uv.lock",
    )
    bundle.finalize(status="completed", summary={})
    (source_attempt / "evaluation").mkdir()
    governed_path = source_attempt / "evaluation" / "governed-run.json"
    governed_path.write_text(
        json.dumps({
            "schema": "autoforecast.hpo-governed-run.v1",
            "source_bundle": bundle.path.relative_to(source_attempt).as_posix(),
        }) + "\n",
        encoding="utf-8",
    )
    heldout_attempt = study_root / "heldout" / "trials" / "trial-0000" / "attempts" / heldout_id
    (heldout_attempt / "evaluation").mkdir(parents=True)
    heldout_path = heldout_attempt / "evaluation" / "heldout.json"
    heldout_path.write_text(
        json.dumps({
            "schema": "autoforecast.heldout-evaluation.v1",
            "objective": "heldout/mae",
            "value": 0.5,
            "selected_source": {"trial_id": "trial-0000", "attempt_id": "attempt-0000"},
        }) + "\n",
        encoding="utf-8",
    )
    return {
        "repo": repo,
        "study_root": study_root,
        "source_attempt": source_attempt,
        "governed_path": governed_path,
        "bundle": bundle,
        "heldout_attempt": heldout_attempt,
        "heldout_path": heldout_path,
    }


def _promote_hpo(case, *, record_name: str = "promotions"):
    return promote_completed_run(
        case["heldout_attempt"],
        retained_ref="refs/tags/autoforecast/run-functional",
        model_id="demo_model",
        alias="champion",
        record_root=case["repo"] / record_name,
        repository=case["repo"],
    )


def _apply_hpo_fault(case, fault: str) -> pathlib.Path:
    heldout_path = case["heldout_path"]
    if fault == "misplaced":
        attempt = (
            case["study_root"] / "not-heldout" / "trials" / "trial-0000"
            / "attempts" / case["heldout_attempt"].name
        )
        (attempt / "evaluation").mkdir(parents=True)
        replacement = attempt / "evaluation" / "heldout.json"
        replacement.write_bytes(heldout_path.read_bytes())
        return replacement
    if fault == "invalid_heldout_id":
        attempt = case["heldout_attempt"].with_name("INVALID")
        (attempt / "evaluation").mkdir(parents=True)
        replacement = attempt / "evaluation" / "heldout.json"
        replacement.write_bytes(heldout_path.read_bytes())
        return replacement
    if fault == "source_bundle_parent_symlink":
        alias = case["source_attempt"] / "source-alias"
        source_root = case["source_attempt"] / "source-run"
        alias.symlink_to(source_root, target_is_directory=True)
        governed = json.loads(case["governed_path"].read_text())
        governed["source_bundle"] = (
            pathlib.PurePosixPath(alias.name)
            / case["bundle"].path.relative_to(source_root).as_posix()
        ).as_posix()
        case["governed_path"].write_text(json.dumps(governed))
        return heldout_path
    evidence = json.loads(heldout_path.read_text())
    if fault == "trial_id_traversal":
        evidence["selected_source"]["trial_id"] = "trial-0000/../trial-0000"
    elif fault == "attempt_id_traversal":
        evidence["selected_source"]["attempt_id"] = "../attempts/attempt-0000"
    elif fault == "malformed_selected_source":
        evidence["selected_source"] = []
    else:
        raise AssertionError(f"unknown HPO fault: {fault}")
    heldout_path.write_text(json.dumps(evidence))
    return heldout_path


@pytest.mark.parametrize(
    ("state", "message"),
    [("missing", "missing or unreadable"), ("directory", "regular-file locator")],
)
def test_canonical_regular_file_distinguishes_missing_and_wrong_kind(tmp_path, state, message):
    from mlops import promotion as module

    path = tmp_path.resolve() / "candidate"
    if state == "directory":
        path.mkdir()
    with pytest.raises(PromotionError, match=message):
        module._canonical_regular_file(path, "candidate evidence")


@pytest.mark.parametrize(
    ("state", "message"),
    [("missing", "missing or unreadable"), ("file", "directory locator")],
)
def test_canonical_directory_distinguishes_missing_and_wrong_kind(tmp_path, state, message):
    from mlops import promotion as module

    path = tmp_path.resolve() / "candidate"
    if state == "file":
        path.write_text("not a directory")
    with pytest.raises(PromotionError, match=message):
        module._canonical_directory(path, "candidate bundle")


def test_read_heldout_evidence_returns_the_exact_bytes_it_parses(tmp_path):
    from mlops import promotion as module

    path = tmp_path.resolve() / "evaluation" / "heldout.json"
    path.parent.mkdir()
    raw = (
        b'{"schema":"autoforecast.heldout-evaluation.v1",'
        b'"objective":"heldout/mae","value":0.5}\n'
    )
    path.write_bytes(raw)
    parsed_raw, _, metric, value = module._read_heldout_evidence(path)
    assert (parsed_raw, metric, value) == (raw, "heldout/mae", 0.5)


def test_governed_bundle_for_heldout_evidence_rejects_shallow_hpo_hierarchy():
    from mlops import promotion as module

    selected = {"selected_source": {"trial_id": "trial-0000", "attempt_id": "attempt-0000"}}
    with pytest.raises(PromotionError, match="canonical study hierarchy"):
        module._governed_bundle_for_heldout_evidence(pathlib.Path("/attempt"), selected)


@pytest.mark.parametrize("selected", [
    [],
    {},
    {"trial_id": "INVALID", "attempt_id": "attempt-0000"},
    {"trial_id": "trial-0000", "attempt_id": "../attempts/attempt-0000"},
])
def test_governed_bundle_for_heldout_evidence_rejects_malformed_or_invalid_selected_ids(
    selected,
):
    from mlops import promotion as module

    root = pathlib.Path(
        "/studies/study-0000/heldout/trials/trial-0000/attempts/heldout-0000"
    )
    with pytest.raises(PromotionError, match="malformed|canonical identifiers"):
        module._governed_bundle_for_heldout_evidence(root, {"selected_source": selected})


@pytest.mark.parametrize(
    "fault",
    ["huge_integer", "parser_integer_limit", "malformed_json", "invalid_utf8", "object"],
)
def test_promote_completed_run_rejects_malformed_heldout_metric(completed_bundle, fault):
    repo, bundle = completed_bundle
    evidence_path = bundle.path / "evaluation" / "heldout.json"
    if fault == "huge_integer":
        evidence = json.loads(evidence_path.read_text())
        evidence["value"] = 10**400
        evidence_path.write_text(json.dumps(evidence))
    elif fault == "parser_integer_limit":
        evidence_path.write_text(
            '{"schema":"autoforecast.heldout-evaluation.v1",'
            '"objective":"heldout/mae","value":' + _PARSER_LIMIT_INTEGER + "}"
        )
    elif fault == "invalid_utf8":
        evidence_path.write_bytes(b"\xff")
    else:
        evidence_path.write_text("{" if fault == "malformed_json" else "[]")
    with pytest.raises(PromotionError, match="held-out"):
        promote_completed_run(
            bundle.path, retained_ref="refs/tags/autoforecast/run-functional",
            model_id="demo_model", alias="champion", record_root=repo / "promotions",
            repository=repo,
        )
    assert not (repo / "promotions" / "demo_model" / "champion.json").exists()


@pytest.mark.parametrize("heldout_id", ["heldout-0000", "evaluation-0000"])
def test_hpo_heldout_evidence_promotes_its_selected_governed_bundle(
    tmp_path: pathlib.Path, monkeypatch, heldout_id,
) -> None:
    case = _hpo_case(tmp_path, heldout_id=heldout_id)
    monkeypatch.chdir(tmp_path)
    record = promote_completed_run(
        case["heldout_attempt"].relative_to(tmp_path),
        retained_ref="refs/tags/autoforecast/run-functional",
        model_id="demo_model",
        alias="champion", record_root=case["repo"] / "promotions", repository=case["repo"],
    )
    assert record.retained.run_id == "functional"
    payload = record.path.read_text(encoding="utf-8")
    assert str(case["heldout_path"]) in payload
    from mlops.deployment import create_local_batch_deployment

    deployment = create_local_batch_deployment(
        record.path, deployment_root=case["repo"] / "deployments", repository=case["repo"],
    )
    assert str(case["heldout_path"]) in (
        deployment.path / "deployment.json"
    ).read_text()


@pytest.mark.parametrize("fault", [
    "misplaced", "invalid_heldout_id", "trial_id_traversal", "attempt_id_traversal",
    "source_bundle_parent_symlink", "malformed_selected_source",
])
def test_promote_completed_run_rejects_noncanonical_hpo_evidence(tmp_path, fault):
    case = _hpo_case(tmp_path)
    replacement = _apply_hpo_fault(case, fault)
    case["heldout_attempt"] = replacement.parent.parent
    with pytest.raises(PromotionError, match="canonical|malformed"):
        _promote_hpo(case)
    assert not (case["repo"] / "promotions" / "demo_model" / "champion.json").exists()


def test_promote_completed_run_rejects_invalid_hpo_study_id(tmp_path):
    case = _hpo_case(tmp_path, study_id="INVALID")
    with pytest.raises(PromotionError, match="canonical study hierarchy"):
        _promote_hpo(case)


def _write_governed_json_integer_limit(case) -> None:
    relative_bundle = case["bundle"].path.relative_to(case["source_attempt"]).as_posix()
    case["governed_path"].write_text(
        '{"schema":"autoforecast.hpo-governed-run.v1","source_bundle":'
        + json.dumps(relative_bundle)
        + ',"oversized":'
        + _PARSER_LIMIT_INTEGER
        + "}"
    )


def test_promote_completed_run_wraps_governed_json_integer_limit(tmp_path):
    case = _hpo_case(tmp_path)
    _write_governed_json_integer_limit(case)
    with pytest.raises(PromotionError, match="cannot read"):
        _promote_hpo(case)


@pytest.mark.parametrize("fault", [
    "misplaced", "invalid_heldout_id", "trial_id_traversal", "attempt_id_traversal",
    "source_bundle_parent_symlink", "malformed_selected_source",
])
def test_create_local_batch_deployment_rejects_noncanonical_hpo_evidence(tmp_path, fault):
    from mlops.deployment import DeploymentError, create_local_batch_deployment

    case = _hpo_case(tmp_path)
    record = _promote_hpo(case)
    replacement = _apply_hpo_fault(case, fault)
    record_data = json.loads(record.path.read_text())
    record_data["heldout_evidence_path"] = str(replacement)
    record.path.write_text(json.dumps(record_data))
    with pytest.raises(DeploymentError, match="held-out"):
        create_local_batch_deployment(
            record.path, deployment_root=case["repo"] / "deployments", repository=case["repo"],
        )
    assert not (case["repo"] / "deployments" / "demo_model" / "champion").exists()


def test_create_local_batch_deployment_wraps_governed_json_integer_limit(tmp_path):
    from mlops.deployment import DeploymentError, create_local_batch_deployment

    case = _hpo_case(tmp_path)
    record = _promote_hpo(case)
    _write_governed_json_integer_limit(case)
    with pytest.raises(DeploymentError, match="cannot read"):
        create_local_batch_deployment(
            record.path, deployment_root=case["repo"] / "deployments", repository=case["repo"],
        )
