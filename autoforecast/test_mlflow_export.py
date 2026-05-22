"""Functional export of a verified Run Bundle into a real local MLflow store."""

from __future__ import annotations

import json
import pathlib
import tempfile
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

import autoforecast.mlflow_export as mlflow_export
from autoforecast.experiments import open_local_attempt
from autoforecast.mlflow_export import (
    MLflowExportError,
    MLflowModel,
    export_completed_bundle,
    publish_model_from_completed_bundle,
    start_training_tracking,
)


@pytest.mark.parametrize(
    "source_bundle_uri",
    (
        "https://user:password@example.test/bundle",
        "https://example.test/bundle?X-Amz-Signature=secret",
        "s3:////access:secret@bucket/key",
    ),
)
def test_mlflow_evidence_refuses_secret_bearing_bundle_references(
    source_bundle_uri: str,
) -> None:
    metadata = {
        "study_id": "study",
        "trial_id": "trial",
        "attempt_id": "attempt",
        "comparability_fingerprint": "a" * 64,
        "data": {"fingerprint": "b" * 64},
    }
    completion = {"status": "completed", "artifact_manifest_sha256": "c" * 64}

    with pytest.raises(MLflowExportError, match="safe retained reference"):
        mlflow_export._tags(  # noqa: SLF001
            pathlib.Path("missing"),
            metadata,
            completion,
            source_bundle_uri=source_bundle_uri,
        )
    with pytest.raises(MLflowExportError, match="safe retained reference"):
        mlflow_export._evidence(  # noqa: SLF001
            metadata,
            completion,
            metrics={},
            source_bundle_uri=source_bundle_uri,
        )


def test_logged_model_packages_every_local_import_namespace() -> None:
    assert {pathlib.Path(path).name for path in mlflow_export._model_code_paths()} == {  # noqa: SLF001
        "autoforecast",
        "forecast",
        "provenance",
        "spec",
    }


def test_publish_model_from_completed_bundle_reuses_the_verified_linked_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="backfill-study",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata={
            "comparability_fingerprint": "a" * 64,
            "model": {"id": "candidate", "adapter": "mft"},
            "study": {"direction": "minimize"},
            "hyperparameters": {"training.learning_rate": 0.001},
            "data": {
                "name": "bundle-data",
                "fingerprint": "b" * 64,
                "observed_date_range": ["2024-01-01", "2024-02-14"],
                "training_date_range": ["2024-01-01", "2024-02-12"],
                "validation_date_range": ["2024-01-01", "2024-02-12"],
                "test_date_range": ["2024-02-13", "2024-02-14"],
                "validation_target_date_range": ["2024-02-11", "2024-02-12"],
                "test_target_date_range": ["2024-02-13", "2024-02-14"],
                "training_series": 5,
                "validation_series": 1,
                "test": "reserved_not_evaluated",
            },
        },
    )
    attempt.write_json("tracking/mlflow.json", {"run_id": "run-42"})
    attempt.write_json(
        "evaluation/validation.json",
        {
            "schema": "autoforecast.hpo-validation.v1",
            "objective": "validation/objective",
            "value": 0.7,
            "metrics": {},
            "test_metrics": "not_evaluated",
        },
    )
    (attempt.path / "model/model.pt").write_bytes(b"sealed checkpoint fixture")
    attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})

    fake_mlflow = MagicMock()
    client = fake_mlflow.tracking.MlflowClient.return_value
    run = client.get_run.return_value
    run.data.tags = {
        "autoforecast.study_id": "backfill-study",
        "autoforecast.trial_id": "trial1",
        "autoforecast.attempt_id": "attempt1",
    }
    run.info.experiment_id = "experiment-7"
    run.info.run_id = "run-42"
    client.search_logged_models.return_value = []
    fake_mlflow.start_run.return_value = nullcontext()
    publication = MLflowModel("logged-model-1", "autoforecast__candidate", "1")
    snapshot_paths: list[pathlib.Path] = []

    def log_snapshot_model(_sink, snapshot_path, _metadata):
        snapshot_paths.append(snapshot_path)
        assert snapshot_path != attempt.path
        assert (snapshot_path / "model/model.pt").read_bytes() == b"sealed checkpoint fixture"
        return publication

    log_candidate_model = MagicMock(side_effect=log_snapshot_model)

    monkeypatch.setattr(mlflow_export, "_mlflow", lambda: fake_mlflow)
    monkeypatch.setattr(mlflow_export, "_log_candidate_model", log_candidate_model)

    result = publish_model_from_completed_bundle(
        attempt.path,
        tracking_uri="sqlite:///tracking.db",
    )

    assert result is publication
    fake_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///tracking.db")
    fake_mlflow.tracking.MlflowClient.assert_called_once_with("sqlite:///tracking.db")
    client.get_run.assert_called_once_with("run-42")
    client.search_logged_models.assert_called_once_with(
        experiment_ids=["experiment-7"],
        filter_string="source_run_id = 'run-42'",
    )
    fake_mlflow.start_run.assert_called_once_with(run_id="run-42")
    logged_sink, logged_path, logged_metadata = log_candidate_model.call_args.args
    assert logged_sink is fake_mlflow
    assert logged_path == snapshot_paths[0]
    assert logged_path != attempt.path
    assert logged_metadata["study_id"] == "backfill-study"
    fake_mlflow.set_tag.assert_called_once_with(
        "autoforecast.model_status", "logged_candidate_registered"
    )


def test_completed_attempt_snapshot_is_detached_before_remote_upload(
    tmp_path: pathlib.Path,
) -> None:
    attempt = open_local_attempt(
        tmp_path,
        study_id="snapshot-study",
        trial_id="trial1",
        attempt_id="attempt1",
        metadata={
            "comparability_fingerprint": "a" * 64,
            "model": {"id": "candidate"},
            "study": {"direction": "minimize"},
            "hyperparameters": {},
            "data": {
                "name": "data",
                "fingerprint": "b" * 64,
                "observed_date_range": ["2024-01-01", "2024-01-02"],
                "training_date_range": ["2024-01-01", "2024-01-01"],
                "validation_date_range": ["2024-01-01", "2024-01-01"],
                "test_date_range": ["2024-01-02", "2024-01-02"],
                "validation_target_date_range": ["2024-01-01", "2024-01-01"],
                "test_target_date_range": ["2024-01-02", "2024-01-02"],
                "training_series": 1,
                "validation_series": 1,
                "test": "reserved_not_evaluated",
            },
        },
    )
    attempt.write_json(
        "evaluation/validation.json",
        {
            "schema": "autoforecast.hpo-validation.v1",
            "value": 1.0,
            "metrics": {},
        },
    )
    attempt.finalize(status="completed", summary={})

    with mlflow_export._completed_bundle_snapshot(  # noqa: SLF001
        attempt.path, source_bundle_uri=None
    ) as (snapshot, _metadata, _completion, _metrics):
        original = attempt.path / "evaluation/validation.json"
        expected = (snapshot / "evaluation/validation.json").read_bytes()
        original.write_bytes(b"changed after snapshot\n")
        assert (snapshot / "evaluation/validation.json").read_bytes() == expected


def test_completed_bundle_exports_one_way_to_real_mlflow_with_safe_evidence() -> None:
    mlflow = pytest.importorskip("mlflow")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        attempt = open_local_attempt(
            root,
            study_id="export-study",
            trial_id="trial1",
            attempt_id="attempt1",
            metadata={
                "comparability_fingerprint": "a" * 64,
                "model": {"id": "example-model"},
                "study": {"direction": "minimize"},
                "hyperparameters": {"training.learning_rate": 0.001},
                "data": {
                    "name": "client-data-not-exported",
                    "fingerprint": "b" * 64,
                    "observed_date_range": ["2024-01-01", "2024-02-14"],
                    "training_date_range": ["2024-01-01", "2024-02-12"],
                    "validation_date_range": ["2024-01-01", "2024-02-12"],
                    "test_date_range": ["2024-02-13", "2024-02-14"],
                    "validation_target_date_range": ["2024-02-11", "2024-02-12"],
                    "test_target_date_range": ["2024-02-13", "2024-02-14"],
                    "training_series": 5,
                    "validation_series": 1,
                    "test": "reserved_not_evaluated",
                },
            },
        )
        attempt.write_json("evaluation/validation.json", {
            "schema": "autoforecast.hpo-validation.v1",
            "objective": "validation/objective",
            "value": 0.7,
            "metrics": {"evaluation/mae": 0.6},
            "test_metrics": "not_evaluated",
        })
        attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
        tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
        exported = export_completed_bundle(
            attempt.path,
            tracking_uri=tracking_uri,
            experiment_name="autoforecast-functional-test",
        )
        run = mlflow.tracking.MlflowClient(tracking_uri).get_run(exported.run_id)
        assert run.data.tags["autoforecast.study_id"] == "export-study"
        assert run.data.params["training.learning_rate"] == "0.001"
        assert run.data.metrics["evaluation/objective"] == 0.7
        artifact = mlflow.artifacts.download_artifacts(
            run_id=exported.run_id,
            artifact_path="autoforecast/evidence.json",
            tracking_uri=tracking_uri,
        )
        evidence = pathlib.Path(artifact).read_text(encoding="utf-8")
        assert "client-data-not-exported" not in evidence
        assert '"fingerprint": "bbbb' in evidence


def test_live_tracking_uploads_epoch_history_and_safe_completed_artifacts() -> None:
    """A real MLflow store receives the same durable records a training run writes."""
    mlflow = pytest.importorskip("mlflow")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        attempt = open_local_attempt(
            root,
            study_id="live-study",
            trial_id="trial1",
            attempt_id="attempt1",
            metadata={
                "comparability_fingerprint": "c" * 64,
                "model": {"id": "example-model"},
                "study": {"direction": "minimize"},
                "hyperparameters": {"training.learning_rate": 0.001},
                "data": {
                    "name": "client-data-not-exported",
                    "fingerprint": "d" * 64,
                    "observed_date_range": ["2024-01-01", "2024-02-14"],
                    "training_date_range": ["2024-01-01", "2024-02-12"],
                    "validation_date_range": ["2024-01-01", "2024-02-12"],
                    "test_date_range": ["2024-02-13", "2024-02-14"],
                    "validation_target_date_range": ["2024-02-11", "2024-02-12"],
                    "test_target_date_range": ["2024-02-13", "2024-02-14"],
                    "training_series": 5,
                    "validation_series": 1,
                    "test": "reserved_not_evaluated",
                },
            },
        )
        metadata = json.loads((attempt.path / "metadata" / "attempt.json").read_text())
        tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
        live = start_training_tracking(
            metadata=metadata, tracking_uri=tracking_uri,
            experiment_name="autoforecast-live-functional-test",
        )
        attempt.metrics_path.write_text(
            '{"epoch": 1, "optimization/train_objective": 1.2, '
            '"optimization/validation_objective": 0.8}\n',
            encoding="utf-8",
        )
        live.log_epoch(1, {
            "optimization/train_objective": 1.2,
            "optimization/validation_objective": 0.8,
        })
        (attempt.path / "model" / "model.pt").write_bytes(b"checkpoint")
        attempt.write_json("evaluation/validation.json", {
            "schema": "autoforecast.hpo-validation.v1",
            "objective": "validation/objective",
            "value": 0.8,
            "metrics": {"evaluation/mae": 0.7},
            "test_metrics": "not_evaluated",
        })
        attempt.finalize(status="completed", summary={"test_metrics": "not_evaluated"})
        live.complete(attempt.path)

        client = mlflow.tracking.MlflowClient(tracking_uri)
        run = client.get_run(live.run_id)
        assert run.info.status == "FINISHED"
        history = client.get_metric_history(live.run_id, "optimization/train_objective")
        assert len(history) == 1 and history[0].step == 1
        artifacts = {item.path for item in client.list_artifacts(live.run_id, "model")}
        assert artifacts == {"model/model.pt"}
