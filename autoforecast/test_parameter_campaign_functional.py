"""Real two-model parameter-source campaign with two neural training epochs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast.parameter_sources import validate_parameter_source_capsule
from provenance.lineage import directory_digest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BREW_JAVA = pathlib.Path("/opt/homebrew/opt/openjdk@17")


def _java_home() -> pathlib.Path | None:
    configured = os.environ.get("JAVA_HOME")
    if configured and (pathlib.Path(configured) / "bin/java").is_file():
        return pathlib.Path(configured)
    if (BREW_JAVA / "bin/java").is_file():
        return BREW_JAVA
    return None


def _campaign_diagnostics(root: pathlib.Path) -> str:
    """Keep a failed worker's reason visible in the CI test report."""
    sections = []
    for name in ("stage-result.json", "stderr.log", "worker-wrapper.log"):
        for path in sorted(root.rglob(name)):
            sections.append(f"\n{path.relative_to(root)}\n" +
                            path.read_text(errors="replace")[-16000:])
    return "".join(sections)


def test_campaign_failure_report_includes_worker_reason(tmp_path):
    attempt = tmp_path / "stages/t00/attempt-0001"
    attempt.mkdir(parents=True)
    (attempt / "stage-result.json").write_text('{"status":"failed","exit_code":2}')
    (attempt / "stderr.log").write_text("worker setup failed")
    (attempt / "worker-wrapper.log").write_text("wrapper could not start the process")
    (attempt / "unrelated.txt").write_text("unrelated content")
    report = _campaign_diagnostics(tmp_path)
    assert '"exit_code":2' in report and "worker setup failed" in report
    assert "wrapper could not start the process" in report
    assert "stages/t00/attempt-0001" in report and "unrelated content" not in report


@pytest.mark.skipif(_java_home() is None, reason="functional campaign requires Java 17 for Spark")
def test_tune_then_manual_campaign_runs_two_epochs_and_finishes_replay(
    tmp_path: pathlib.Path,
) -> None:
    start = dt.date(2024, 1, 1)
    records = [
        {
            "series_id": series,
            "date": start + dt.timedelta(days=offset),
            "target": float(2 + series_index + offset % 7),
        }
        for series_index, series in enumerate(("a", "b"))
        for offset in range(70)
    ]
    inputs: dict[str, pathlib.Path] = {}
    for index, name in enumerate(("selection", "final-training", "final-score")):
        path = tmp_path / name
        path.mkdir()
        shifted = [{**record, "target": record["target"] + 20.0 * index} for record in records]
        pq.write_table(pa.Table.from_pylist(shifted), path / "part.parquet")
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
    # Keep the functional HPO trial small while exercising the exact same schema and bridge.
    shallow_path = repository / "configs/models/global/shallow_global/model.yaml"
    shallow = yaml.safe_load(shallow_path.read_text())
    compact_values = {
        "model.num_leaves": 7,
        "model.min_data_in_leaf": 1,
        "model.num_boost_round": 5,
        "training.learning_rate": 0.05,
    }
    for name, value in compact_values.items():
        shallow["hyperparameters"][name]["default"] = value
        shallow["hyperparameters"][name]["search"] = {
            "kind": "categorical",
            "values": [value],
        }
    shallow_path.write_text(yaml.safe_dump(shallow, sort_keys=False), encoding="utf-8")

    fingerprint = directory_digest(inputs["selection"], suffixes=(".parquet",))
    study_id = "parameter-campaign-smoke"
    study_path = repository / "configs/hpo/parameter_campaign_smoke.yaml"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    study_path.write_text(
        f"""schema_version: 1
study:
  id: {study_id}
  model_config: configs/models/global/shallow_global/model.yaml
  backend: optuna
  direction: minimize
  objective: validation/objective
  max_trials: 1
  top_k: 1
  seed: 7
  comparability_fingerprint: {hashlib.sha256(study_id.encode()).hexdigest()}
  evaluator: builtin
  data:
    id: selection-v1
    fingerprint: {fingerprint}
    observed_date_range: ['2024-01-01', '2024-03-10']
    training_date_range: ['2024-01-01', '2024-01-31']
    validation_date_range: ['2024-01-01', '2024-01-31']
    test_date_range: ['2024-02-03', '2024-02-04']
    validation_target_date_range: ['2024-02-01', '2024-02-02']
    test_target_date_range: ['2024-02-03', '2024-02-04']
    training_series: 2
    validation_series: 2
    test: reserved_not_evaluated
""",
        encoding="utf-8",
    )
    budget = {
        "maximum_epochs": 2,
        "patience": 2,
        "timeout_seconds": 240,
        "max_attempts": 1,
        "early_stopping_absolute_min_delta": 1e-6,
        "early_stopping_relative_min_delta": 0.0,
    }
    request = {
        "schema_version": 6,
        "forecast_campaign": {
            "purpose": "forecast_campaign",
            "name": "parameter-source-functional",
            "dataset": {
                "contract": "configs/datasets/synthetic_daily.yaml",
                "selection": {
                    "source_uri": "local://selection-v1",
                },
                "final_training": {
                    "source_uri": "local://training-v1",
                },
                "final_score": {
                    "source_uri": "local://score-v1",
                },
            },
            "models": {
                "catalog": "configs/models",
                "candidates": [
                    {
                        "model_id": "shallow_global",
                        "parameters": {
                            "mode": "tune",
                            "study_config": "configs/hpo/parameter_campaign_smoke.yaml",
                        },
                    },
                    {
                        "model_id": "mft",
                        "parameters": {
                            "mode": "manual",
                            "values": {
                                "model.d_model": 8,
                                "model.heads": 1,
                                "model.factor_rank": 1,
                                "model.trunk_stages": 1,
                                "model.decoder_layers": 1,
                                "model.history_attention_layers": 1,
                                "model.commercial_attention_layers": 1,
                                "model.event_kernel_layers": 1,
                            },
                        },
                    },
                ],
            },
            "temporal": {
                "training_start": "2024-01-01",
                "initialization_origin": None,
                "validation_origins": ["2024-02-10"],
                "validation_horizon": 2,
                "final_train_cutoff": "2024-02-25",
                "final_test_horizon": 2,
            },
            "seeds": [0],
            "initialization_arms": ["fresh"],
            "selection": {
                "metric": "mae",
                "missing_actual": "no_data",
                "direction": "minimize",
                "tie_break": [
                    "mean_score",
                    "score_stddev",
                    "model_id",
                    "variant_id",
                    "initialization",
                ],
            },
            "final_evidence_role": "blind_test",
            "budgets": {
                "initialization": dict(budget),
                "validation": dict(budget),
                "final_fit": dict(budget),
            },
            "compute": {
                "precision": "32-true",
                "device": "cpu",
                "batch_size": 32,
                "gradient_microbatch_size": 16,
                # macOS accounts the Spark JVM plus Python process tree above 3 GiB even for
                # this two-series fixture; 4 GiB retains a strict, CI-portable ceiling.
                "memory_gib": 4.0,
                "min_free_disk_gib": 0.1,
            },
            "tracking": {"mlflow": False},
        },
    }
    request_path = repository / "configs/autoforecast/examples/parameter_campaign.yaml"
    request_path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    for command in (
        ("git", "init"),
        ("git", "config", "user.email", "test@example.test"),
        ("git", "config", "user.name", "Functional"),
        ("git", "add", "."),
        ("git", "commit", "-m", "functional source"),
    ):
        outcome = subprocess.run(command, cwd=repository, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stderr

    governed_python = ROOT / ".venv/bin/python"
    executable = str(governed_python if governed_python.is_file() else pathlib.Path(sys.executable))
    environment = os.environ.copy()
    java = _java_home()
    assert java is not None
    environment.update(
        {
            "JAVA_HOME": str(java),
            "PATH": f"{java / 'bin'}:{environment['PATH']}",
            "SPARK_LOCAL_IP": "127.0.0.1",
        }
    )
    outcome = subprocess.run(
        [
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
            str(repository / ".autoforecast/parameter-campaigns"),
            "--poll-seconds",
            "0.05",
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert outcome.returncode == 0, (outcome.stdout + outcome.stderr + _campaign_diagnostics(
        repository / ".autoforecast/parameter-campaigns",
    ))
    assert "[preflight] data.selection:" in outcome.stderr
    assert "parameter_mode=tune (HPO on" in outcome.stderr
    assert "parameter_mode=manual (HPO off" in outcome.stderr
    status = json.loads(outcome.stdout)
    campaign_root = pathlib.Path(status["campaign_root"])
    assert status["status"] == "completed"
    report_path = next(campaign_root.glob("stages/replay_report/attempt-*/campaign-report.json"))
    report = json.loads(report_path.read_text())
    assert report["replay"]["status"] == "verified"
    assert report["final_score"]["status"] == "scored"
    score_attempt = next(campaign_root.glob("stages/final_score/attempt-*"))
    assert (
        directory_digest(score_attempt / "scoring-inputs/forecast", suffixes=(".parquet",))
        == report["final_score"]["forecast_input_artifact_sha256"]
    )
    assert (
        directory_digest(score_attempt / "scoring-inputs/actuals", suffixes=(".parquet",))
        == report["final_score"]["score_input_artifact_sha256"]
    )
    assert {item["parameter_mode"] for item in report["validation"]["candidates"]} == {
        "tune",
        "manual",
    }
    custom_result = next(
        campaign_root.glob("stages/v1fs0o0/attempt-*/run/*/attempts/mft/*")
    )
    epochs = [
        json.loads(line)
        for line in (custom_result / "metrics/epochs.jsonl").read_text().splitlines()
    ]
    assert [item["epoch"] for item in epochs] == [1, 2]
    tuned = next(campaign_root.glob("stages/t00/attempt-*/parameter-resolution.json"))
    tuned_record = json.loads(tuned.read_text())
    assert tuned_record["source"]["kind"] == "tune"
    validation_panel = next(campaign_root.glob("stages/t00/attempt-*/hpo-validation-panel"))
    validation_dates = (
        pads.dataset(validation_panel, format="parquet")
        .to_table(columns=["date"])["date"]
        .to_pylist()
    )
    # Evaluation includes pre-origin history for scale/weight metrics such as WRMSSE, while
    # remaining physically bounded before every reserved final-test target.
    assert min(validation_dates) == dt.date(2024, 1, 1)
    assert max(validation_dates) == dt.date(2024, 2, 2)
    outer_evaluation_panels = sorted(
        campaign_root.glob("stages/v*/attempt-*/evaluation-panel")
    )
    assert len(outer_evaluation_panels) == 2
    for panel in outer_evaluation_panels:
        outer_dates = (
            pads.dataset(panel, format="parquet").to_table(columns=["date"])["date"].to_pylist()
        )
        assert min(outer_dates) == dt.date(2024, 1, 1)
        assert max(outer_dates) == dt.date(2024, 2, 12)
    tuned_bundle = next(
        campaign_root.glob("stages/v0fs0o0/attempt-*/run/*/attempts/shallow_global/*")
    )
    relocated_bundle = tmp_path / "relocated-tuned-bundle"
    shutil.copytree(tuned_bundle, relocated_bundle)
    validate_parameter_source_capsule(
        relocated_bundle,
        expected_parameter_resolution_sha256=hashlib.sha256(tuned.read_bytes()).hexdigest(),
    )
