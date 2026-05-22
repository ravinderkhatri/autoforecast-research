"""Functional proof that one generic request produces an approval-gated local bundle."""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
from pyspark.sql import SparkSession
from scripts.run_autoforecast import ROOT


@pytest.mark.skipif(not os.environ.get("JAVA_HOME"), reason="Spark requires JAVA_HOME")
def test_synthetic_request_runs_from_yaml_through_report_and_sealed_bundle(tmp_path: pathlib.Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    spark = (SparkSession.builder.master("local[1]").appName("generic-run-functional")
             .config("spark.ui.enabled", "false").getOrCreate())
    try:
        records = [
            (series, dt.date(2024, 1, 1) + dt.timedelta(days=offset),
             float(offset + (0 if series == "a" else 1)))
            for series in ("a", "b") for offset in range(16)
        ]
        data_path = tmp_path / "panel"
        spark.createDataFrame(records, ["series_id", "date", "target"]).write.parquet(str(data_path))
    finally:
        spark.stop()
    # A real run refuses dirty source. Copy the current implementation into an ephemeral Git
    # repository so this test exercises that production guard instead of bypassing it.
    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT, repository,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", ".pytest_cache", ".ruff_cache", ".autoforecast",
            "data", "gates", "mlruns",
            "mlartifacts", "lightning_logs", "__pycache__",
        ),
    )
    assert not (repository / ".autoforecast").exists()
    # Commit the tracking-enabled request with the source snapshot.  The runner must refuse a
    # dirty source tree, so modifying it after the commit would invalidate this proof.
    request_path = repository / "configs/autoforecast/examples/synthetic_seasonal_naive.yaml"
    request_path.write_text(
        request_path.read_text(encoding="utf-8").replace(
            "mlflow: {enabled: false}", "mlflow: {enabled: true}",
        ),
        encoding="utf-8",
    )
    for command in (
        ("git", "init"), ("git", "config", "user.email", "test@example.test"),
        ("git", "config", "user.name", "Functional"), ("git", "add", "."),
        ("git", "commit", "-m", "functional source"),
    ):
        outcome = subprocess.run(command, cwd=repository, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stderr
    output_root = tmp_path / "runs"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    environment = os.environ.copy()
    environment["MLFLOW_TRACKING_URI"] = tracking_uri
    environment["AUTOFORECAST_MLFLOW_EXPERIMENT"] = "generic-run-functional"
    outcome = subprocess.run(
        [
            sys.executable, "scripts/run_autoforecast.py",
            "configs/autoforecast/examples/synthetic_seasonal_naive.yaml",
            "--data-path", str(data_path), "--data-snapshot-id", "synthetic-2024-01-16",
            "--data-reference", "local://synthetic-2024-01-16", "--output-root", str(output_root),
            "--epochs", "1", "--timeout-seconds", "120",
        ],
        cwd=repository, capture_output=True, text=True, env=environment, timeout=180,
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    scratch = output_root / ".autoforecast-worker-scratch"
    assert scratch.is_dir() and not any(scratch.iterdir())
    runs = [path for path in output_root.iterdir() if path.is_dir() and path != scratch]
    assert len(runs) == 1
    run = runs[0]
    completion = next((run / "attempts/seasonal_naive").glob("*/completion.json"))
    assert completion.is_file()
    assert (run / "evaluation/candidate-comparison.json").is_file()
    assert (run / "observability.html").is_file()
    assert "Training completed" in (run / "report.html").read_text(encoding="utf-8")
    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("generic-run-functional")
    assert experiment is not None
    tracked = client.search_runs([experiment.experiment_id])
    assert len(tracked) == 1
    assert tracked[0].info.status == "FINISHED"
    assert tracked[0].data.tags["autoforecast.tracking_phase"] == "verified_completed_bundle"
