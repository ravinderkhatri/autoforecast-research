"""Campaign recovery and availability parity on synthetic series."""

import json
import pathlib
from dataclasses import replace

import pytest
import yaml

from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_campaign import models
from autoforecast.test_study_tuning import freeze, open_study
from autoforecast.test_study_tuning import spark as _spark


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


@pytest.mark.parametrize("execution", ["sequential", "local"])
def test_campaign_retries_failed_model_and_keeps_completed_candidate(spark, tmp_path, monkeypatch, execution):
    from pathlib import Path

    manifest = freeze(tmp_path / "study", spark)
    declared = models(tmp_path)
    declared[1] = replace(declared[1], entrypoint="autoforecast.test_study_campaign:fail_once")
    marker = tmp_path / "failure-marker"
    monkeypatch.setenv("AUTOFORECAST_TEST_FAILURE_MARKER", str(marker))
    if execution == "local":
        monkeypatch.setenv(
            "AUTOFORECAST_TEST_COMPLETED_PEER",
            str(tmp_path / "campaign"),
        )
    with open_study(manifest, spark) as study:
        result = study.run_campaign(
            models=declared,
            output_dir=tmp_path / "campaign",
            final_evaluation="all_models",
            max_attempts=2,
            timeout_seconds=600,
            execution=execution,
            max_parallel_models=2 if execution == "local" else 1,
        )
        assert result["state"]["status"] == "completed", result
        stage = next(row for row in result["state"]["stages"] if row["id"] == "validation")
        assert stage["attempts"] == 2
        root = Path(result["campaign_path"])
        failed = json.loads((root / "stages/validation/attempt-0001/summary.json").read_text())
        passed = json.loads((root / "stages/validation/attempt-0002/summary.json").read_text())
        assert set(failed["failures"]) == {"1"}
        if execution == "sequential":
            assert failed["failures"]["1"] == "injected model failure"
        assert failed["results"]["0"] == passed["results"]["0"]
        record = result["results"]["results"]["0"]
        workflow = load_forecast_workflow(record["workflow_path"])
        model_file = next(p for p in workflow.testing["model_artifacts"].rglob("*") if p.is_file())
        model_file.write_text("changed model bytes")
        with pytest.raises(ValueError):
            study.run_campaign(
                models=declared,
                output_dir=tmp_path / "campaign",
                final_evaluation="all_models",
                max_attempts=2,
                timeout_seconds=600,
                execution=execution,
                max_parallel_models=2 if execution == "local" else 1,
            )
    if execution == "local":
        import time

        import psutil

        identities = json.loads(marker.read_text())
        survivors = []
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            survivors = []
            for identity in identities:
                try:
                    child = psutil.Process(identity["pid"])
                    if child.create_time() == identity["created"] and child.status() != psutil.STATUS_ZOMBIE:
                        survivors.append(child)
                except psutil.NoSuchProcess:
                    pass
            if not survivors:
                break
            time.sleep(0.2)
        for child in survivors:
            child.kill()
        assert not survivors, "crashed model left live child processes"


@pytest.mark.parametrize("policy", ["exclude_unavailable", "use_available"])
def test_fixed_and_tuned_models_share_frozen_availability(spark, tmp_path, policy):
    from pathlib import Path

    from autoforecast.test_study_workflow import register

    # Two registered series; B starts on day 3 and cannot supply three days at
    # the first tuning origin. Exclusion removes B globally; available mode keeps
    # the original denominator and permits B in later windows when eligible.
    manifest = register(tmp_path / "study", spark, start_b=3)
    declared = models(tmp_path)
    third = tmp_path / "last-value.yaml"
    third.write_text(Path(declared[0].config).read_text())
    value = yaml.safe_load(third.read_text())
    value["model"]["id"] = "last-value"
    third.write_text(yaml.safe_dump(value))
    declared.append(replace(
        declared[0], config=third, entrypoint="autoforecast.test_study_campaign:last_value_predict"
    ))
    for index, declaration in enumerate(declared):
        path = Path(declaration.config)
        value = yaml.safe_load(path.read_text())
        value.update(series_policy=policy, minimum_history=1 if index == 0 else 3)
        value["selection"] = {"origins": "all", "metric": "rmse"}
        path.write_text(yaml.safe_dump(value))
    declared[1] = replace(
        declared[1], fixed_parameters=None,
        entrypoint="autoforecast.test_study_campaign:unit_drift_predict",
    )
    # A single fixed parameter choice makes executor parity independent of adaptive search order.
    tuned = pathlib.Path(declared[1].config)
    profile = yaml.safe_load(tuned.read_text())
    for parameter in profile["hyperparameters"].values():
        parameter["search"] = {"kind": "categorical", "values": [parameter["default"]]}
    tuned.write_text(yaml.safe_dump(profile))
    results = []
    with open_study(manifest, spark) as study:
        for mode, parallel in (("sequential", 1), ("local", 2)):
            results.append(study.run_campaign(
                models=declared, output_dir=tmp_path / mode, timeout_seconds=600,
                execution=mode, max_parallel_models=parallel,
            ))
    snapshots = []
    for result in results:
        assert result["state"]["status"] == "completed", result
        candidate_runs = [
            load_forecast_workflow(row["workflow_path"])
            for _, row in sorted(result["results"]["results"].items())
        ]
        snapshot = []
        for run in candidate_runs:
            stages = []
            for stage in (run.validation, run.testing):
                stages.append({
                    "contract": stage["evaluation_contract"],
                    "score": {key: stage["score"][key] for key in (
                        "mae", "rmse", "coverage", "population", "evaluation_status", "metric_status"
                    )},
                })
            snapshot.append({"parameters": run.parameters, "stages": stages})
        snapshots.append(snapshot)
    assert snapshots[0] == snapshots[1]
    assert results[0]["results"]["selected_model"] == results[1]["results"]["selected_model"]
    runs = candidate_runs
    assert len(runs) == 3
    assert all(run.provenance["options"]["metric"] == "rmse" for run in runs)
    assert all(run.provenance["study"] == runs[0].provenance["study"] for run in runs)
    assert all(
        run.validation["evaluation_contract"] == runs[0].validation["evaluation_contract"]
        for run in runs
    )
    assert runs[2].validation["score"]["mae"] == pytest.approx(1.5)
    effective = json.loads(Path(runs[0].provenance["study"]["path"]).read_text())
    assert effective["execution_policy"]["minimum_history"] == 3
    assert effective["cohort_series_count"] == (1 if policy == "exclude_unavailable" else 2)
