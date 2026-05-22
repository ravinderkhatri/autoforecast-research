"""Functional MLflow export check for one sealed generic bundle."""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest
from mlops import open_run_bundle

import autoforecast.run_mlflow as run_mlflow_module
from autoforecast.run_mlflow import (
    LiveRunMLflow,
    RunMLflowLink,
    export_run_bundle,
    start_live_run_bundle_tracking,
)


def test_parameter_source_tag_and_hash_come_from_one_file_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    path = root / "metadata/parameter-resolution.json"
    path.parent.mkdir(parents=True)
    original = b'{"source":{"kind":"manual"}}\n'
    replacement = b'{"source":{"kind":"defaults"}}\n'
    path.write_bytes(original)
    real_read = pathlib.Path.read_bytes
    reads = 0

    def adversarial_read(source: pathlib.Path) -> bytes:
        nonlocal reads
        if source == path:
            reads += 1
            return original if reads == 1 else replacement
        return real_read(source)

    monkeypatch.setattr(pathlib.Path, "read_bytes", adversarial_read)
    tags = run_mlflow_module._parameter_source_tags(root)  # noqa: SLF001
    assert tags == {
        "autoforecast.parameter_source": "manual",
        "autoforecast.parameter_resolution_sha256": hashlib.sha256(original).hexdigest(),
    }
    assert reads == 1


def test_live_run_mlflow_worker_context_contains_only_attachment_identifiers() -> None:
    live = LiveRunMLflow(
        RunMLflowLink(
            tracking_uri="sqlite:///tracking.db",
            experiment_id="experiment-7",
            run_id="run-42",
        ),
        object(),
    )

    context = live.worker_context
    assert context == {
        "tracking_uri": "sqlite:///tracking.db",
        "experiment_id": "experiment-7",
        "run_id": "run-42",
    }
    context["run_id"] = "caller-mutated"
    assert live.worker_context["run_id"] == "run-42"
    retained = live.link.as_record()
    assert retained["schema"] == "autoforecast.generic-mlflow-link.v2"
    assert retained["tracking_uri_sha256"] == hashlib.sha256(
        b"sqlite:///tracking.db"
    ).hexdigest()
    assert "tracking_uri" not in retained
    public = RunMLflowLink(
        "https://user:secret@example.test/mlflow?signature=private",
        "experiment-7",
        "run-42",
    ).as_record()
    assert public["tracking_web_url"] == "https://example.test/mlflow"
    assert "secret" not in json.dumps(public)


def test_completed_run_snapshot_is_detached_before_remote_upload(
    tmp_path: pathlib.Path,
) -> None:
    bundle = open_run_bundle(tmp_path, run_id="snapshot-run", task="forecast")
    artifact = bundle.write_json("metadata/resolved-run.json", {"model_id": "demo"})
    bundle.finalize(status="completed", summary={})

    with run_mlflow_module._completed_run_snapshot(bundle.path) as (snapshot, _completion):  # noqa: SLF001
        expected = (snapshot / "metadata/resolved-run.json").read_bytes()
        artifact.write_bytes(b"changed after snapshot\n")
        assert (snapshot / "metadata/resolved-run.json").read_bytes() == expected


def test_sealed_generic_bundle_exports_metrics_params_and_model_artifact(
    tmp_path: pathlib.Path,
) -> None:
    mlflow = pytest.importorskip("mlflow")
    bundle = open_run_bundle(tmp_path, run_id="functional-mlflow", task="forecast")
    bundle.write_json(
        "metadata/resolved-run.json",
        {"model_id": "demo", "adapter": "mft"},
    )
    bundle.write_json(
        "metadata/execution-plan.json",
        {
            "effective_parameters": {"training.learning_rate": 0.01},
        },
    )
    bundle.write_json(
        "metadata/parameter-resolution.json",
        {"source": {"kind": "manual"}},
    )
    bundle.write_json("provenance/source.json", {"commit": "a" * 40})
    bundle.write_json("provenance/data.json", {"fingerprint": "b" * 64})
    bundle.write_json("provenance/runtime.json", {"lockfile_sha256": "c" * 64})
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"functional checkpoint")
    bundle.copy_file("model/model.pt", checkpoint)
    bundle.finalize(status="completed", summary={})
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "model_id": "demo",
                        "score": {"rmse": 1.25, "n_scored": 4},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    link = export_run_bundle(
        bundle.path,
        tracking_uri=uri,
        experiment_name="Functional generic bundle",
        comparison_path=comparison,
    )
    run = mlflow.tracking.MlflowClient(uri).get_run(link.run_id)
    assert run.data.params["training.learning_rate"] == "0.01"
    assert run.data.metrics["evaluation.rmse"] == 1.25
    assert run.data.tags["autoforecast.model_registry"] == "not_registered_pending_promotion"
    assert run.data.tags["autoforecast.parameter_source"] == "manual"
    assert len(run.data.tags["autoforecast.parameter_resolution_sha256"]) == 64


def test_live_generic_tracking_reuses_one_mlflow_run_after_bundle_verification(
    tmp_path: pathlib.Path,
) -> None:
    mlflow = pytest.importorskip("mlflow")
    bundle = open_run_bundle(tmp_path, run_id="functional-live-mlflow", task="forecast")
    bundle.write_json("metadata/resolved-run.json", {"model_id": "demo", "adapter": "tft"})
    bundle.write_json(
        "metadata/execution-plan.json",
        {
            "effective_parameters": {"training.learning_rate": 0.01},
        },
    )
    bundle.write_json("provenance/source.json", {"commit": "a" * 40})
    bundle.write_json("provenance/data.json", {"fingerprint": "b" * 64})
    bundle.write_json("provenance/runtime.json", {"lockfile_sha256": "c" * 64})
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    live = start_live_run_bundle_tracking(
        bundle.path,
        tracking_uri=uri,
        experiment_name="Functional live generic bundle",
    )
    bundle.write_json("metrics/epochs.jsonl", {"epoch": 1, "train_loss": 0.5})
    # The durable epoch file is JSONL in real runs; make this fixture equally shaped.
    metrics = bundle.path / "metrics/epochs.jsonl"
    metrics.write_text('{"epoch": 1, "train_loss": 0.5}\n', encoding="utf-8")
    bundle.finalize(status="completed", summary={})
    live.complete(bundle.path)
    client = mlflow.tracking.MlflowClient(uri)
    experiment = client.get_experiment_by_name("Functional live generic bundle")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and runs[0].info.run_id == live.link.run_id
    assert runs[0].data.metrics["train_loss"] == 0.5


def test_two_sequential_models_close_live_runs_before_the_next_model_starts(
    tmp_path: pathlib.Path,
) -> None:
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    lives: list[LiveRunMLflow] = []
    for model_id in ("first", "second"):
        bundle = open_run_bundle(tmp_path / "bundles", run_id=model_id, task="forecast")
        bundle.write_json(
            "metadata/resolved-run.json", {"model_id": model_id, "adapter": "tft"}
        )
        bundle.write_json(
            "metadata/execution-plan.json", {"effective_parameters": {"training.epochs": 1}}
        )
        bundle.write_json("provenance/source.json", {"commit": "a" * 40})
        bundle.write_json("provenance/data.json", {"fingerprint": "b" * 64})
        bundle.write_json("provenance/runtime.json", {"lockfile_sha256": "c" * 64})
        live = start_live_run_bundle_tracking(
            bundle.path,
            tracking_uri=uri,
            experiment_name="Sequential live generic bundles",
        )
        bundle.finalize(status="completed", summary={})
        live.complete(bundle.path)
        assert mlflow.active_run() is None
        lives.append(live)

    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "candidates": [
                    {"model_id": "first", "score": {"rmse": 1.0}},
                    {"model_id": "second", "score": {"rmse": 0.5}},
                ]
            }
        ),
        encoding="utf-8",
    )
    for model_id, live in zip(("first", "second"), lives, strict=True):
        live.project_comparison(comparison, model_id=model_id)

    client = mlflow.tracking.MlflowClient(uri)
    assert client.get_run(lives[0].link.run_id).data.metrics["evaluation.rmse"] == 1.0
    assert client.get_run(lives[1].link.run_id).data.metrics["evaluation.rmse"] == 0.5
