"""Functional tests for local study/trial/attempt records and their comparison page."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile

import pytest

from autoforecast.experiment_report import render_study_report
from autoforecast.experiments import (
    ExperimentError,
    load_completed_attempt,
    load_study_selection,
    open_local_attempt,
    open_local_heldout_attempt,
    require_heldout_evaluation_approval,
    seal_study_selection,
)
from autoforecast.observability import open_training_log


def _metadata(*, learning_rate: float) -> dict[str, object]:
    return {
        "comparability_fingerprint": "a" * 64,
        "hyperparameters": {"training.learning_rate": learning_rate},
        "data": {"observed_date_range": ["2024-01-01", "2024-02-14"],
                 "training_date_range": ["2024-01-01", "2024-02-12"],
                 "validation_date_range": ["2024-01-01", "2024-02-12"],
                 "test_date_range": ["2024-02-13", "2024-02-14"],
                 "validation_target_date_range": ["2024-02-11", "2024-02-12"],
                 "test_target_date_range": ["2024-02-13", "2024-02-14"],
                 "test": "reserved_not_evaluated",
                 "fingerprint": "b" * 64, "name": "test-data",
                 "training_series": 5, "validation_series": 1},
    }


def test_open_local_heldout_attempt_uses_the_separate_heldout_hierarchy(
    tmp_path: pathlib.Path,
) -> None:
    attempt = open_local_heldout_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )

    assert attempt.path == (
        tmp_path / "studies/smoke/heldout/trials/trial1/attempts/attempt1"
    )
    record = json.loads((attempt.path / "metadata/attempt.json").read_text(encoding="utf-8"))
    assert (record["study_id"], record["trial_id"], record["attempt_id"]) == (
        "smoke",
        "trial1",
        "attempt1",
    )
    assert not (tmp_path / "studies/smoke/trials/trial1/attempts/attempt1").exists()


def test_local_attempt_tensorboard_dir_is_precreated_inside_the_attempt(
    tmp_path: pathlib.Path,
) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )

    assert attempt.tensorboard_dir == attempt.path / "tensorboard"
    assert attempt.tensorboard_dir.is_dir()


def test_local_attempt_preserves_same_day_hourly_range_ordering(tmp_path: pathlib.Path) -> None:
    metadata = _metadata(learning_rate=0.001)
    data = metadata["data"]
    assert isinstance(data, dict)
    data.update(
        observed_date_range=["2024-01-01T00:00:00", "2024-01-01T11:00:00"],
        training_date_range=["2024-01-01T00:00:00", "2024-01-01T07:00:00"],
        validation_date_range=["2024-01-01T00:00:00", "2024-01-01T09:00:00"],
        test_date_range=["2024-01-01T10:00:00", "2024-01-01T11:00:00"],
        validation_target_date_range=["2024-01-01T08:00:00", "2024-01-01T09:00:00"],
        test_target_date_range=["2024-01-01T10:00:00", "2024-01-01T11:00:00"],
    )

    attempt = open_local_attempt(
        tmp_path,
        study_id="hourly",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=metadata,
    )

    stored = json.loads((attempt.path / "metadata/attempt.json").read_text())
    assert stored["data"]["validation_target_date_range"] == [
        "2024-01-01T08:00:00", "2024-01-01T09:00:00",
    ]


def test_attempt_copy_refuses_a_symlinked_destination_parent(tmp_path: pathlib.Path) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (attempt.path / "model" / "linked").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "source.bin"
    source.write_bytes(b"safe")

    with pytest.raises(ExperimentError, match="could not copy attempt source"):
        attempt.copy_file("model/linked/escaped.bin", source)

    assert not (outside / "escaped.bin").exists()


def test_attempt_write_bytes_refuses_unsafe_or_linked_paths(tmp_path: pathlib.Path) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (attempt.path / "model" / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ExperimentError, match="must be relative"):
        attempt.write_bytes("../escaped.bin", b"unsafe")
    absolute_escape = tmp_path / "absolute-escaped.bin"
    with pytest.raises(ExperimentError, match="must be relative"):
        attempt.write_bytes(str(absolute_escape), b"unsafe")
    with pytest.raises(ExperimentError, match="could not write attempt artifact"):
        attempt.write_bytes("model/linked/escaped.bin", b"unsafe")

    assert not (attempt.path.parent / "escaped.bin").exists()
    assert not absolute_escape.exists()
    assert not (outside / "escaped.bin").exists()


def test_load_study_selection_accepts_only_a_selection_bound_to_its_path(
    tmp_path: pathlib.Path,
) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata={**_metadata(learning_rate=0.001), "study": {"direction": "minimize"}},
    )
    attempt.write_json(
        "evaluation/validation.json",
        {
            "schema": "autoforecast.hpo-validation.v1",
            "objective": "validation/objective",
            "value": 0.5,
            "metrics": {},
            "test_metrics": "not_evaluated",
        },
    )
    attempt.finalize(status="completed", summary={})
    study_root = tmp_path / "studies/smoke"
    selection_path = seal_study_selection(
        study_root, top_k=1, direction="minimize", required_trial_count=1
    )
    selection = load_study_selection(study_root)
    assert selection["selected"][0]["rank"] == 1

    selection["study_id"] = "another-study"
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    with pytest.raises(ExperimentError, match="identity does not match"):
        load_study_selection(study_root)


def test_load_study_selection_recomputes_ranked_candidates_before_reuse(
    tmp_path: pathlib.Path,
) -> None:
    for number, value in enumerate((0.4, 0.8), start=1):
        attempt = open_local_attempt(
            tmp_path,
            study_id="smoke",
            trial_id=f"trial{number}",
            attempt_id="attempt1",
            metadata={**_metadata(learning_rate=number / 1000), "study": {"direction": "minimize"}},
        )
        attempt.write_json(
            "evaluation/validation.json",
            {
                "schema": "autoforecast.hpo-validation.v1",
                "objective": "validation/objective",
                "value": value,
                "metrics": {},
                "test_metrics": "not_evaluated",
            },
        )
        attempt.finalize(status="completed", summary={})
    study_root = tmp_path / "studies/smoke"
    selection_path = seal_study_selection(
        study_root, top_k=1, direction="minimize", required_trial_count=2
    )
    crafted = json.loads(selection_path.read_text(encoding="utf-8"))
    crafted["selected"] = [crafted["candidates"][1]]
    selection_path.write_text(json.dumps(crafted, sort_keys=True), encoding="utf-8")
    with pytest.raises(ExperimentError, match="ranked candidates"):
        load_study_selection(study_root)


def test_equal_objectives_break_ties_by_parameters_before_random_attempt_ids(
    tmp_path: pathlib.Path,
) -> None:
    parameters = (("trial-z", 0.001), ("trial-a", 0.003))
    identities: dict[str, str] = {}
    for trial_id, learning_rate in parameters:
        metadata = {**_metadata(learning_rate=learning_rate), "study": {"direction": "minimize"}}
        identities[trial_id] = hashlib.sha256(
            json.dumps(metadata["hyperparameters"], separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        attempt = open_local_attempt(
            tmp_path,
            study_id="smoke",
            trial_id=trial_id,
            attempt_id=f"attempt-{trial_id}",
            metadata=metadata,
        )
        attempt.write_json(
            "evaluation/validation.json",
            {
                "schema": "autoforecast.hpo-validation.v1",
                "objective": "validation/objective",
                "value": 0.5,
                "metrics": {},
                "test_metrics": "not_evaluated",
            },
        )
        attempt.finalize(status="completed", summary={})
    selection = json.loads(
        seal_study_selection(
            tmp_path / "studies/smoke",
            top_k=1,
            direction="minimize",
            required_trial_count=2,
        ).read_text(encoding="utf-8")
    )
    expected = min(identities, key=identities.get)
    assert selection["selected"][0]["trial_id"] == expected


def test_completed_attempt_has_durable_epoch_records_and_a_comparable_report() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for number, learning_rate in enumerate((0.001, 0.003), start=1):
            attempt = open_local_attempt(
                root, study_id="smoke", trial_id=f"trial{number}", attempt_id="attempt1",
                metadata=_metadata(learning_rate=learning_rate),
            )
            log = open_training_log(None, metrics_path=attempt.metrics_path)
            for epoch in range(1, 3):
                log.log_epoch(
                    epoch, train_objective=2.0 / epoch, validation_objective=3.0 / epoch,
                    learning_rate=learning_rate, gradient_norm=0.5, optimizer_steps=epoch,
                    training_examples_seen=epoch * 5, seconds=0.1, process_peak_rss_bytes=10,
                    cuda_peak_memory_bytes=None,
                )
            log.close()
            attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
        report = render_study_report(root / "studies" / "smoke")
        assert "Validation objectives are overlaid" in report.read_text(encoding="utf-8")
        attempt_path = root / "studies/smoke/trials/trial1/attempts/attempt1"
        completion = json.loads((attempt_path / "completion.json").read_text())
        manifest = json.loads((attempt_path / "artifact_manifest.json").read_text())
        assert completion["status"] == "completed"
        assert "metrics/epochs.jsonl" in {entry["path"] for entry in manifest["artifacts"]}


def test_completed_attempt_terminal_payload_is_sealed(tmp_path: pathlib.Path) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    attempt.finalize(status="completed", summary={})
    completion_path = attempt.path / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = "failed"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ExperimentError, match="terminal payload is not sealed"):
        load_completed_attempt(attempt.path)


def test_completed_attempt_rejects_non_string_status(tmp_path: pathlib.Path) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    attempt.finalize(status="completed", summary={})
    completion_path = attempt.path / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = []
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ExperimentError, match="unsupported attempt schema"):
        load_completed_attempt(attempt.path)


def test_completed_attempt_identity_must_match_metadata(tmp_path: pathlib.Path) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    attempt.finalize(status="completed", summary={})
    metadata_path = attempt.path / "metadata/attempt.json"
    manifest_path = attempt.path / "artifact_manifest.json"
    completion_path = attempt.path / "completion.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metadata["trial_id"] = "trial2"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    entry = next(
        item for item in manifest["artifacts"] if item["path"] == "metadata/attempt.json"
    )
    entry["sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    entry["bytes"] = metadata_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion["artifact_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ExperimentError, match="trial_id differs"):
        load_completed_attempt(attempt.path)


def test_completed_attempt_rejects_malformed_data_test(tmp_path: pathlib.Path) -> None:
    metadata = _metadata(learning_rate=0.001)
    metadata["data"]["test"] = []  # type: ignore[index]
    with pytest.raises(ExperimentError, match="data.test"):
        open_local_attempt(
            tmp_path,
            study_id="smoke",
            trial_id="trial1",
            attempt_id="attempt1",
            metadata=metadata,
        )


def test_completed_attempt_rejects_legacy_unsealed_manifest(tmp_path: pathlib.Path) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    attempt.finalize(status="completed", summary={})
    manifest_path = attempt.path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "autoforecast.artifact-manifest.v1"
    manifest.pop("terminal_payload_sha256")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ExperimentError, match="legacy attempt.*must be rerun"):
        load_completed_attempt(attempt.path)


def test_study_report_links_to_the_exact_live_mlflow_run_and_marks_checkpoint_status() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        attempt = open_local_attempt(
            root, study_id="smoke", trial_id="trial1", attempt_id="attempt1",
            metadata=_metadata(learning_rate=0.001),
        )
        attempt.metrics_path.write_text(
            '{"epoch": 1, "optimization/validation_objective": 1.0}\n', encoding="utf-8",
        )
        attempt.write_json("tracking/mlflow.json", {
            "schema": "autoforecast.mlflow-tracking-link.v2",
            "status": "started",
            "tracking_uri_sha256": hashlib.sha256(
                b"http://127.0.0.1:5001"
            ).hexdigest(),
            "tracking_web_url": "https://user:secret@example.test/mlflow?signature=private",
            "experiment_id": "7",
            "run_id": "abc123",
        })
        (attempt.path / "model" / "model.pt").write_bytes(b"checkpoint")
        attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
        page = render_study_report(
            root / "studies" / "smoke", mlflow_url="http://127.0.0.1:5001"
        ).read_text(encoding="utf-8")
        assert "/#/experiments/7/runs/abc123" in page
        assert "Checkpoint captured; not registered until held-out approval" in page
        sanitized = render_study_report(
            root / "studies" / "smoke",
            mlflow_url="https://user:secret@example.test/mlflow?signature=private",
        ).read_text(encoding="utf-8")
        assert "https://example.test/mlflow/#/experiments/7/runs/abc123" in sanitized
        assert "secret" not in sanitized and "signature" not in sanitized
        without_fallback = render_study_report(
            root / "studies" / "smoke"
        ).read_text(encoding="utf-8")
        assert "https://example.test/mlflow/#/experiments/7/runs/abc123" in without_fallback
        assert "secret" not in without_fallback and "signature" not in without_fallback


def test_attempt_refuses_reuse_and_post_completion_mutation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        attempt = open_local_attempt(
            directory, study_id="smoke", trial_id="trial1", attempt_id="attempt1",
            metadata=_metadata(learning_rate=0.001),
        )
        attempt.finalize(status="failed", summary={})
        try:
            attempt.write_json("evaluation/result.json", {})
        except ExperimentError as exc:
            assert "finalized" in str(exc)
        else:
            raise AssertionError("finalized attempt accepted a new artifact")
        with pytest.raises(ExperimentError, match="finalized"):
            attempt.write_bytes("evaluation/result.bin", b"late")
        assert not (attempt.path / "evaluation/result.json").exists()
        assert not (attempt.path / "evaluation/result.bin").exists()
        load_completed_attempt(attempt.path)
        try:
            open_local_attempt(
                directory, study_id="smoke", trial_id="trial1", attempt_id="attempt1",
                metadata=_metadata(learning_rate=0.001),
            )
        except ExperimentError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("retry silently reused a previous attempt directory")


def test_attempt_rejects_unmanifested_symlinked_directories(tmp_path: pathlib.Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("not attempt evidence", encoding="utf-8")

    unsealed = open_local_attempt(
        tmp_path / "unsealed",
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    (unsealed.path / "model").rmdir()
    (unsealed.path / "model").symlink_to(external, target_is_directory=True)
    with pytest.raises(ExperimentError, match="symlinks"):
        unsealed.finalize(status="completed", summary={})

    sealed = open_local_attempt(
        tmp_path / "sealed",
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    sealed.finalize(status="completed", summary={})
    (sealed.path / "model").rmdir()
    (sealed.path / "model").symlink_to(external, target_is_directory=True)
    with pytest.raises(ExperimentError, match="symlinks"):
        load_completed_attempt(sealed.path)


def test_attempt_creation_refuses_a_dangling_link_at_the_claimed_leaf(
    tmp_path: pathlib.Path,
) -> None:
    parent = tmp_path / "studies/smoke/trials/trial1/attempts"
    parent.mkdir(parents=True)
    (parent / "attempt1").symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ExperimentError, match="already exists"):
        open_local_attempt(
            tmp_path,
            study_id="smoke",
            trial_id="trial1",
            attempt_id="attempt1",
            metadata=_metadata(learning_rate=0.001),
        )


def test_attempt_creation_refuses_a_linked_hierarchy_ancestor(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "studies").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ExperimentError, match="cannot create attempt path"):
        open_local_attempt(
            root,
            study_id="smoke",
            trial_id="trial1",
            attempt_id="attempt1",
            metadata=_metadata(learning_rate=0.001),
        )

    assert list(outside.iterdir()) == []


def test_completed_attempt_rejects_link_swapped_during_final_enumeration(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    artifact = attempt.write_json("evaluation/value.json", {"value": 1})
    attempt.finalize(status="completed", summary={})
    external = tmp_path / "external.json"
    external.write_bytes(artifact.read_bytes())
    real_rglob = pathlib.Path.rglob
    calls = 0

    def swapping_rglob(path: pathlib.Path, pattern: str):
        nonlocal calls
        values = tuple(real_rglob(path, pattern))
        if path == attempt.path:
            calls += 1
            if calls == 2:
                artifact.unlink()
                artifact.symlink_to(external)
        return iter(values)

    monkeypatch.setattr(pathlib.Path, "rglob", swapping_rglob)
    with pytest.raises(ExperimentError, match="symlinks"):
        load_completed_attempt(attempt.path)


def test_completed_attempt_rejects_directory_replaced_by_an_undeclared_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="type-swap",
        metadata=_metadata(learning_rate=0.001),
    )
    attempt.finalize(status="completed", summary={})
    rogue = attempt.path / "rogue"
    rogue.mkdir()
    real_rglob = pathlib.Path.rglob
    calls = 0

    def swapping_rglob(path: pathlib.Path, pattern: str):
        nonlocal calls
        values = tuple(real_rglob(path, pattern))
        if path == attempt.path:
            calls += 1
            if calls == 2:
                rogue.rmdir()
                rogue.write_text("undeclared", encoding="utf-8")
        return iter(values)

    monkeypatch.setattr(pathlib.Path, "rglob", swapping_rglob)
    with pytest.raises(ExperimentError, match="path set or type changed"):
        load_completed_attempt(attempt.path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no FIFO support")
def test_completed_attempt_rejects_undeclared_special_files(tmp_path: pathlib.Path) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="special-file",
        metadata=_metadata(learning_rate=0.001),
    )
    attempt.finalize(status="completed", summary={})
    os.mkfifo(attempt.path / "undeclared.fifo")
    with pytest.raises(ExperimentError, match="unsupported path type"):
        load_completed_attempt(attempt.path)


@pytest.mark.parametrize("mutation", ("duplicate_path", "extra_entry"))
def test_completed_attempt_rejects_ambiguous_manifest_entries(
    tmp_path: pathlib.Path, mutation: str
) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="smoke",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata=_metadata(learning_rate=0.001),
    )
    attempt.finalize(status="completed", summary={})
    manifest_path = attempt.path / "artifact_manifest.json"
    completion_path = attempt.path / "completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if mutation == "duplicate_path":
        manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    else:
        manifest["artifacts"][0]["unexpected"] = "field"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completion["artifact_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ExperimentError, match="manifest"):
        load_completed_attempt(attempt.path)


def test_sealed_validation_selection_authorizes_only_top_k_and_renders_a_portable_trace_link() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for number, value in enumerate((0.9, 0.7), start=1):
            attempt = open_local_attempt(
                root, study_id="smoke", trial_id=f"trial{number}", attempt_id="attempt1",
                metadata={
                    **_metadata(learning_rate=0.001 * number),
                    "study": {"direction": "minimize"},
                },
            )
            attempt.write_json("evaluation/validation.json", {
                "schema": "autoforecast.hpo-validation.v1",
                "objective": "validation/objective",
                "value": value,
                "metrics": {"validation/mae": value},
                "test_metrics": "not_evaluated",
            })
            attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
        study_root = root / "studies" / "smoke"
        selection = seal_study_selection(
            study_root, top_k=1, direction="minimize", required_trial_count=2,
        )
        assert selection.exists()
        approved = require_heldout_evaluation_approval(
            study_root, trial_id="trial2", attempt_id="attempt1",
        )
        assert approved["validation_objective"] == 0.7
        try:
            require_heldout_evaluation_approval(
                study_root, trial_id="trial1", attempt_id="attempt1",
            )
        except ExperimentError as exc:
            assert "not approved" in str(exc)
        else:
            raise AssertionError("unselected validation candidate was approved for held-out scoring")
        try:
            seal_study_selection(study_root, top_k=1, direction="minimize")
        except ExperimentError as exc:
            assert "sealed" in str(exc)
        else:
            raise AssertionError("a sealed selection was replaced")
        report = render_study_report(
            study_root, mlflow_url="https://traces.example.test/project",
        ).read_text(encoding="utf-8")
        assert "Selection is sealed: top_k=1" in report
        assert "Selected (rank 1)" in report
        assert "https://traces.example.test/project" in report
        assert "127.0.0.1" not in report


def test_attempt_refuses_missing_test_range_before_creating_a_partial_record() -> None:
    with tempfile.TemporaryDirectory() as directory:
        metadata = _metadata(learning_rate=0.001)
        data = metadata["data"]
        assert type(data) is dict
        data.pop("test_date_range")
        try:
            open_local_attempt(
                directory, study_id="smoke", trial_id="trial1", attempt_id="attempt1",
                metadata=metadata,
            )
        except ExperimentError as exc:
            assert "test_date_range" in str(exc)
        else:
            raise AssertionError("attempt accepted missing test split provenance")
        assert not list(pathlib.Path(directory).iterdir())


def test_report_refuses_an_attempt_whose_hashed_artifact_changed_after_completion() -> None:
    with tempfile.TemporaryDirectory() as directory:
        attempt = open_local_attempt(
            directory, study_id="smoke", trial_id="trial1", attempt_id="attempt1",
            metadata=_metadata(learning_rate=0.001),
        )
        attempt.write_json("evaluation/result.json", {"state": "original"})
        attempt.finalize(status="completed", summary={})
        (attempt.path / "evaluation/result.json").write_text('{"state":"changed"}\n')
        try:
            render_study_report(pathlib.Path(directory) / "studies" / "smoke")
        except ExperimentError as exc:
            assert "does not match bytes" in str(exc)
        else:
            raise AssertionError("report rendered a post-completion mutation")
