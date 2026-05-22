"""Real SFT/HMFNet-and-variant campaign proof, including supervisor worker adoption."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BREW_JAVA = pathlib.Path("/opt/homebrew/opt/openjdk@17")


def _java_home() -> pathlib.Path | None:
    configured = os.environ.get("JAVA_HOME")
    if configured and (pathlib.Path(configured) / "bin/java").is_file():
        return pathlib.Path(configured)
    if (BREW_JAVA / "bin/java").is_file():
        return BREW_JAVA
    return None


@pytest.mark.skipif(_java_home() is None, reason="functional campaign requires Java 17 for Spark")
@pytest.mark.parametrize(
    ("variant", "request_name", "model_id"),
    (
        (
            "v2",
            "synthetic_sft_single_model_2epochs.yaml",
            "sft",
        ),
        (
            "v3",
            "synthetic_hmfnet_single_model_2epochs.yaml",
            "hmfnet",
        ),
    ),
)
def test_single_revision_two_epoch_benchmark_finishes_every_stage_with_live_progress(
    tmp_path: pathlib.Path,
    variant: str,
    request_name: str,
    model_id: str,
) -> None:
    records = [
        {
            "series_id": series,
            "date": dt.date(2024, 1, 1) + dt.timedelta(days=offset),
            "target": float((offset % 7) + series_index),
        }
        for series_index, series in enumerate(("a", "b"))
        for offset in range(128)
    ]
    inputs = {}
    for index, name in enumerate(("selection", "final-training", "final-score")):
        path = tmp_path / name
        path.mkdir()
        pq.write_table(
            pa.Table.from_pylist(
                [{**record, "target": record["target"] + float(index * 100)} for record in records]
            ),
            path / "part.parquet",
        )
        inputs[name] = path
    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT,
        repository,
        ignore=shutil.ignore_patterns(
            ".git",
            ".autoforecast",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "data",
            "gates",
            "mlruns",
            "mlartifacts",
            "lightning_logs",
            "__pycache__",
        ),
    )
    for command in (
        ("git", "init"),
        ("git", "config", "user.email", "test@example.test"),
        ("git", "config", "user.name", "Functional"),
        ("git", "add", "."),
        ("git", "commit", "-m", "functional source"),
    ):
        outcome = subprocess.run(command, cwd=repository, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stderr
    request = repository / "configs/autoforecast/examples" / request_name
    output_root = repository / f".autoforecast/single-{variant}-functional"
    governed_python = ROOT / ".venv/bin/python"
    executable = str(governed_python if governed_python.is_file() else pathlib.Path(sys.executable))
    command = [
        executable,
        "scripts/run_forecast_campaign.py",
        "run",
        str(request),
        "--selection-data-path",
        str(inputs["selection"]),
        "--final-training-data-path",
        str(inputs["final-training"]),
        "--final-score-data-path",
        str(inputs["final-score"]),
        "--output-root",
        str(output_root),
        "--poll-seconds",
        "0.05",
    ]
    java = _java_home()
    assert java is not None
    environment = {
        **os.environ,
        "AUTOFORECAST_LIVE_PROGRESS": "1",
        "JAVA_HOME": str(java),
        "PATH": f"{java / 'bin'}:{os.environ['PATH']}",
        "SPARK_LOCAL_IP": "127.0.0.1",
    }
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for expected in (
        "###### Stage 1/3: Train on training data and evaluate on validation data",
        "Stage 1/3 | Phase 1/6 — Prepare governed training and validation slices",
        "Stage 1/3 | Phase 2/6 — Run Spark data and model preflight",
        "Stage 1/3 | Phase 3/6 — Build model tensors",
        "Stage 1/3 | Phase 4/6 — Train, validate, and select the update budget",
        "Epoch 1/2 — Fixed holdout after",
        "Stage 1/3 | Phase 6/6 — Freeze the validation result and model configuration",
        "###### Stage 2/3: Fresh fit on training + validation data",
        "###### Stage 3/3: Evaluate once on test data",
        "Stage 3/3 | Phase 2/3 — Calculate test WRMSSE",
        "Stage 3/3 | Phase 3/3 — Verify and seal the test report",
        "Stage 1/3 | Phase 2/6 — Run Spark data and model preflight | "
        "Step 1/3 — Start the bounded local Spark runtime",
        "Stage 1/3 | Phase 3/6 — Build model tensors | "
        "Step 1/2 — Load and validate the governed panel arrays",
        "Epoch 1/2",
        "Epoch 2/2",
    ):
        assert expected in completed.stderr
        assert "Substage" not in completed.stderr
        assert "weighted_event: 0isolated adapter worker" not in completed.stderr
        assert re.search(
            r"#+ FINAL RMSE SUMMARY \| Validation RMSE: \d+\.\d{6} \| "
            r"Test RMSE: \d+\.\d{6} #+",
            completed.stderr,
        )
    status = json.loads(completed.stdout)
    assert status["status"] == "completed"
    campaign_root = pathlib.Path(status["campaign_root"])
    state = json.loads((campaign_root / "state.json").read_text())
    assert [item["id"] for item in state["stages"]] == [
        "v0fs0o0",
        "aggregate_selection",
        "final_fit",
        "final_score",
        "replay_report",
    ]
    assert all(item["status"] == "completed" for item in state["stages"])
    report_path = next(campaign_root.glob("stages/replay_report/attempt-*/campaign-report.json"))
    report = json.loads(report_path.read_text())
    assert report["replay"]["status"] == "verified"
    assert report["final_score"]["status"] == "scored"
    assert [item["model_id"] for item in report["validation"]["candidates"]] == [model_id]


@pytest.mark.skipif(_java_home() is None, reason="functional campaign requires Java 17 for Spark")
def test_real_sft_hmfnet_variant_campaign_recovers_supervisor_and_finishes_every_stage(
    tmp_path: pathlib.Path,
) -> None:
    records = [
        {
            "series_id": series,
            "date": dt.date(2024, 1, 1) + dt.timedelta(days=offset),
            "target": float((offset % 7) + series_index),
        }
        for series_index, series in enumerate(("a", "b"))
        for offset in range(128)
    ]
    inputs = {}
    for index, name in enumerate(("selection", "final-training", "final-score")):
        path = tmp_path / name
        path.mkdir()
        distinct_records = [
            {**record, "target": record["target"] + float(index * 100)} for record in records
        ]
        pq.write_table(pa.Table.from_pylist(distinct_records), path / "part.parquet")
        inputs[name] = path
    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT,
        repository,
        ignore=shutil.ignore_patterns(
            ".git",
            ".autoforecast",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "data",
            "gates",
            "mlruns",
            "mlartifacts",
            "lightning_logs",
            "__pycache__",
        ),
    )
    request_path = (
        repository / "configs/autoforecast/examples/synthetic_sft_hmfnet_variants_2epochs.yaml"
    )
    for command in (
        ("git", "init"),
        ("git", "config", "user.email", "test@example.test"),
        ("git", "config", "user.name", "Functional"),
        ("git", "add", "."),
        ("git", "commit", "-m", "functional source"),
    ):
        outcome = subprocess.run(command, cwd=repository, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stderr
    output_root = repository / ".autoforecast/functional-campaigns"
    governed_python = ROOT / ".venv/bin/python"
    executable = str(governed_python if governed_python.is_file() else pathlib.Path(sys.executable))
    command = [
        executable,
        "scripts/run_forecast_campaign.py",
        "run",
        str(request_path),
        "--selection-data-path",
        str(inputs["selection"]),
        "--final-training-data-path",
        str(inputs["final-training"]),
        "--final-score-data-path",
        str(inputs["final-score"]),
        "--output-root",
        str(output_root),
        "--poll-seconds",
        "0.05",
    ]
    environment = os.environ.copy()
    java = _java_home()
    assert java is not None
    environment["JAVA_HOME"] = str(java)
    environment["PATH"] = f"{java / 'bin'}:{environment['PATH']}"
    environment["SPARK_LOCAL_IP"] = "127.0.0.1"
    original = subprocess.Popen(
        command,
        cwd=repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        identity: pathlib.Path | None = None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            identities = list(output_root.glob("*/stages/v*/attempt-0001/worker-identity.json"))
            if identities:
                identity = identities[0]
                break
            if original.poll() is not None:
                stdout, stderr = original.communicate()
                raise AssertionError(f"campaign ended before restart probe: {stdout}{stderr}")
            time.sleep(0.1)
        assert identity is not None and identity.is_file()
        original.terminate()
        original.wait(timeout=10)

        recovered = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        assert "[preflight] data.selection:" in recovered.stderr
        assert "parameter_mode=manual (HPO off" in recovered.stderr
        status = json.loads(recovered.stdout)
        assert status["status"] == "completed"
        campaign_root = pathlib.Path(status["campaign_root"])
        state = json.loads((campaign_root / "state.json").read_text())
        assert all(stage["status"] == "completed" for stage in state["stages"])
        first_stage = next(stage for stage in state["stages"] if stage["id"].startswith("v"))
        assert first_stage["attempts"] == 1
        report = next(campaign_root.glob("stages/replay_report/attempt-*/campaign-report.json"))
        evidence = json.loads(report.read_text())
        assert evidence["replay"]["status"] == "verified"
        assert evidence["final_score"]["status"] == "scored"
        candidates = evidence["validation"]["candidates"]
        assert sorted(item["model_id"] for item in candidates) == [
            "hmfnet",
            "sft",
            "sft",
        ]
        assert all(item["variant_id"].startswith("manual-") for item in candidates)
        assert len({item["variant_id"] for item in candidates}) == 3
        resolved = json.loads(
            (pathlib.Path(status["compiled_request"]) / "resolved.json").read_text()
        )
        assert all(
            item["candidate_id"]
            == f"@{len(item['model_id'])}:{item['model_id']}:{item['variant_id']}"
            for item in resolved["models"]
        )
        validation_children = [
            item for item in resolved["children"] if item["kind"] == "validation"
        ]
        assert [item["candidate_id"] for item in validation_children] == [
            item["candidate_id"] for item in resolved["models"]
        ]
        for child in validation_children:
            stage = next(item for item in state["stages"] if item["id"] == child["id"])
            attempt = campaign_root / "stages" / child["id"] / f"attempt-{stage['attempts']:04d}"
            native_evidence = list(
                (attempt / "run").glob("*/attempts/*/*/evaluation/forecast-revision-evidence")
            )
            assert len(native_evidence) == 1
            assert (native_evidence[0] / "COMPLETE.json").is_file()
            training = [
                json.loads(line)
                for line in (native_evidence[0] / "training.jsonl").read_text().splitlines()
            ]
            assert {item["epoch"] for item in training} == {1, 2}
    finally:
        if original.poll() is None:
            original.terminate()
            original.wait(timeout=10)
