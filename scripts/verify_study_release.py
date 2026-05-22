"""Installed acceptance for study coverage, unequal starts, and retained-forecast scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile
from typing import cast
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from autoforecast.benchmark import verify_benchmark
from autoforecast.results import export_results, read_results

from scripts.verify_timeseries_release import run_cli


def verify_scalar_identity(campaign: pathlib.Path, project: pathlib.Path) -> None:
    """Transient scalar reads must not change the chosen model or exported scores."""
    from autoforecast.campaign import CampaignStore
    from autoforecast.forecast_campaign import (
        ForecastCampaignError,
        aggregate_selection,
        replay_report,
    )
    from autoforecast.project import experiment_project

    command = CampaignStore.from_root(campaign).plan.stages[0].command
    resolved = pathlib.Path(command[command.index("--resolved") + 1])
    resolved_sha = command[command.index("--resolved-sha256") + 1]
    read_bytes, read_text = pathlib.Path.read_bytes, pathlib.Path.read_text
    for operation, filename in (
        ("reader", "child-result.json"), ("reader", "final-score.json"),
        ("reader", "selection-lock.json"), ("reader", "candidate-comparison.json"),
        ("selection", "child-result.json"), ("selection", "candidate-comparison.json"),
        ("selection", "paired-scores"), ("replay", "child-result.json"),
        ("replay", "final-score.json"), ("replay", "final-comparison"),
        ("replay", "selection-lock.json"), ("replay", "validation-summary.json"),
    ):
        injected = []

        def changed(path, raw, *, expected_filename=filename, observed=injected):
            matches = path.name == expected_filename
            if expected_filename == "paired-scores":
                matches = path.name in {"child-result.json", "candidate-comparison.json"}
            elif expected_filename == "final-comparison":
                matches = path.name == "candidate-comparison.json" and "final_score" in path.parts
            if not matches or not path.is_relative_to(campaign):
                return raw
            value = json.loads(raw)
            if expected_filename == "paired-scores":
                target = value["candidates"][0] if "candidates" in value else value
                score = 100.0 if target["model_id"] == "seasonal_naive" else 0.0
                if "candidates" in value:
                    target["score"][value["metric"]] = score
                else:
                    target["score"] = score
            elif path.name == "candidate-comparison.json":
                value["candidates"][0]["score"][value["metric"]] = 987654.0
            elif expected_filename == "selection-lock.json":
                value["chosen"]["model_id"] = "unexpected-model"
            else:
                value["score"] = 987654.0
            observed.append(path)
            return json.dumps(value).encode()

        def bytes_read(path):
            return changed(path, read_bytes(path))

        def text_read(path, *args, **kwargs):
            return changed(path, read_text(path, *args, **kwargs).encode()).decode()

        with patch.object(pathlib.Path, "read_bytes", bytes_read), \
                patch.object(pathlib.Path, "read_text", text_read):
            try:
                if operation == "reader":
                    read_results(campaign)
                else:
                    with (
                        tempfile.TemporaryDirectory(prefix="study-scalar-probe-") as temporary,
                        experiment_project(project),
                    ):
                        consume = aggregate_selection if operation == "selection" else replay_report
                        consume(resolved, resolved_sha, campaign_root=campaign,
                                attempt_dir=pathlib.Path(temporary))
            except ForecastCampaignError as error:
                assert "identity differs" in str(error), error
            except ValueError as error:
                assert "study comparison changed" in str(error), error
            else:
                raise AssertionError(f"{operation} accepted transient {filename}")
        assert injected, f"probe did not reach {operation}/{filename}"
    print("PASS: journal-bound scalar reads reject transient changes in selection and results")

    from autoforecast import forecast_campaign as campaign_ops

    with experiment_project(project):
        record = campaign_ops._load_resolved(resolved, resolved_sha)
    child_paths = [
        campaign_ops._completed_output(campaign, cast(str, child["id"]), "child-result.json")
        for child in cast(list[dict[str, object]], record["children"])
        if child["kind"] == "validation"
    ]
    expected_hashes = sorted(campaign_ops._load_completed_json(path)[1] for path in child_paths)
    original_lock, _ = campaign_ops._load_completed_json(
        campaign_ops._completed_output(campaign, "aggregate_selection", "selection-lock.json")
    )
    real_hash = campaign_ops._sha_file
    for filename in ("child-result.json", "validation-summary.json"):
        rereads = []

        def changed_hash(path, *, target=filename, observed=rereads):
            if path.name == target:
                observed.append(path)
                return "0" * 64
            return real_hash(path)

        with (
            tempfile.TemporaryDirectory(prefix="study-producer-probe-") as temporary,
            experiment_project(project),
            patch.object(campaign_ops, "_sha_file", changed_hash),
        ):
            attempt = pathlib.Path(temporary)
            aggregate_selection(resolved, resolved_sha, campaign_root=campaign,
                                attempt_dir=attempt)
            summary_bytes = (attempt / "validation-summary.json").read_bytes()
            summary = json.loads(summary_bytes)
            lock = json.loads((attempt / "selection-lock.json").read_bytes())
            assert sorted(sha for candidate in summary["candidates"]
                          for sha in candidate["child_result_sha256"]) == expected_hashes
            assert lock["validation_summary_sha256"] == hashlib.sha256(summary_bytes).hexdigest()
            assert lock == original_lock
            assert not rereads, f"selection reopened scalar identity: {filename}"
    print("PASS: selection hashes use authenticated child and serialized summary bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=pathlib.Path)
    args = parser.parse_args()
    project = args.project.resolve()
    print(
        "Study population: two synthetic series, with declared starts on days 1 and 15; "
        "14 required forecast cells at each of two validation origins and one test origin. "
        "No cohort exclusions; original missing actuals are masked and counted.", flush=True,
    )
    run_cli("init-example", project, cwd=project.parent)
    recipe_path = project / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe.update(metric="wrmsse", actual_missing="mask", series_starts={
        "rule": "Synthetic starts declared before the first validation origin", "values": [
            {"keys": {"shop": "a|b", "sku": 1}, "start": "2024-01-01"},
            {"keys": {"shop": "a", "sku": 2}, "start": "2024-01-15"},
        ],
    })
    recipe_path.write_text(json.dumps(recipe))
    request_path = project / "campaign.yaml"
    request = yaml.safe_load(request_path.read_text())
    request["forecast_campaign"]["selection"]["metric"] = "wrmsse"
    request_path.write_text(yaml.safe_dump(request))
    references = project / "scoring-references.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"keys": {"shop": shop, "sku": sku}, "stage": stage, "origin": origin,
         "weight": weight, "scale": scale}
        for stage, origin in [
            *(("validation", value) for value in recipe["validation_origins"]),
            ("test", recipe["final_train_cutoff"]),
        ]
        for shop, sku, weight, scale in (("a|b", 1, 14.0, 2.5), ("a", 2, 9.0, 4.0))
    ]), references)
    source = project / "data/observations.parquet"
    table = pq.read_table(source)
    rows = []
    for row in table.to_pylist():
        day = (row["day"] - table["day"][0].as_py()).days + 1
        if row["sku"] == 2 and (day < 15 or day == 23 or day > 35):
            continue
        if row["sku"] == 1 and day in (22, 38):
            row["units"] = None
        if row["sku"] == 1 and day == 37:
            row["units"] = 0.0
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), source)
    frozen_path = run_cli("freeze", "--project-root", project, "--source", "data",
                          "--scoring-references", references, cwd=project)
    frozen = verify_benchmark(frozen_path)[1]
    study_id = frozen["study_id"]
    campaign = run_cli("run", "--project-root", project, "--study-id", study_id, cwd=project)
    results = read_results(campaign)
    assert results.metadata["study_id"] == study_id
    assert results.metadata["final_score"]["metric"] == "wrmsse"
    assert results.metadata["final_score"]["status"] == "unavailable"
    assert results.metadata["final_score"]["score"] is None
    assert len(results.runs) == 5 and results.rows.num_rows == 70
    assert results.series_scores is not None and results.series_scores.num_rows == 10
    verify_scalar_identity(campaign, project)
    validation_coverage = {}
    for row in results.series_scores.to_pylist():
        if row["stage"] != "validation":
            continue
        key = (row["origin"], row["series_id"])
        counts = tuple(row[name] for name in (
            "expected", "originally_missing", "zero_filled", "excluded", "scored",
        ))
        assert validation_coverage.setdefault(key, counts) == counts
    test = [row for row in results.series_scores.to_pylist() if row["stage"] == "test"]
    assert sorted(row["scored"] for row in test) == [0, 6]
    assert sum(row["expected"] for row in test) == 14
    assert sum(row["originally_missing"] for row in test) == 8
    assert next(row for row in test if row["scored"] == 0)["rmse"] is None
    export_results(results, project / "export")
    assert pq.read_table(project / "export/series-scores.parquet").equals(results.series_scores)
    resumed = run_cli("run", "--project-root", project, "--study-id", study_id, cwd=project)
    assert resumed == campaign
    original_files = {
        str(path.relative_to(campaign)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in campaign.rglob("*") if path.is_file()
    }
    # Change only the scoring snapshot: observed replacement labels produce a new identity.
    replacement = [dict(row) for row in table.to_pylist()]
    for row in replacement:
        row["units"] += 1.0
    labels = project / "revised-actuals.parquet"
    pq.write_table(pa.Table.from_pylist(replacement, schema=table.schema), labels)
    revised_path = run_cli(
        "freeze", "--project-root", project, "--source", "data",
        "--validation-score", labels, "--final-score", labels,
        "--scoring-references", references, cwd=project,
    )
    revised = verify_benchmark(revised_path)[1]
    assert revised["study_id"] != study_id
    rescored = run_cli(
        "rescore", "--project-root", project, "--campaign", campaign,
        "--study-id", revised["study_id"], "--output", "rescored", cwd=project,
    )
    record = json.loads(rescored.read_text())
    assert record["original_study_id"] == study_id
    assert record["scoring_study_id"] == revised["study_id"]
    for item in record["scores"]:
        comparison = json.loads(pathlib.Path(item["comparison"]).read_text())
        assert comparison["metric"] == "wrmsse"
        candidate = comparison["candidates"][0]
        assert candidate["status"] == "scored" and candidate["score"]["wrmsse"] is not None
    assert {entry["run_id"] for entry in record["scores"]} == {
        run["run_id"] for run in results.runs
    }
    assert all(hashlib.sha256((campaign / path).read_bytes()).hexdigest() == value
               for path, value in original_files.items())
    incompatible = {**recipe, "final_test_horizon": 6}
    recipe_path.write_text(json.dumps(incompatible))
    mismatch = run_cli("freeze", "--project-root", project, "--source", "data",
                       "--scoring-references", references, cwd=project)
    try:
        run_cli("rescore", "--project-root", project, "--campaign", campaign,
                "--study-id", verify_benchmark(mismatch)[1]["study_id"],
                "--output", "incompatible-rescore", cwd=project)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("rescoring accepted an incompatible forecast boundary")
    assert not (project / "incompatible-rescore").exists()
    # Moving one declared start to the first origin leaves one period: seasonal-naive needs seven.
    short_project = project.parent / (project.name + "-short")
    run_cli("init-example", short_project, cwd=project.parent)
    short_recipe = json.loads((short_project / "recipe.json").read_text())
    short_recipe["series_starts"] = {
        "rule": "Synthetic short history", "values": [
            {"keys": {"shop": "a|b", "sku": 1}, "start": "2024-01-01"},
            {"keys": {"shop": "a", "sku": 2}, "start": short_recipe["validation_origins"][0]},
        ],
    }
    (short_project / "recipe.json").write_text(json.dumps(short_recipe))
    short_path = run_cli("freeze", "--project-root", short_project, "--source", "data", cwd=project)
    try:
        run_cli("run", "--project-root", short_project, "--benchmark", short_path, cwd=project)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("short frozen series was silently dropped")
    summaries = list((short_project / ".autoforecast/runs").rglob("run-summary.json"))
    records = [json.loads(path.read_text()) for path in summaries]
    assert any("Study requires all 2 frozen series" in json.dumps(record) for record in records)
    assert not list((short_project / ".autoforecast/runs").rglob("forecast.json"))
    empty_project = project.parent / (project.name + "-empty-test")
    run_cli("init-example", empty_project, cwd=project.parent)
    empty_recipe = json.loads((empty_project / "recipe.json").read_text())
    empty_recipe.update(actual_missing="mask", validation_origins=["2024-01-21"])
    (empty_project / "recipe.json").write_text(json.dumps(empty_recipe))
    empty_request = yaml.safe_load((empty_project / "campaign.yaml").read_text())
    empty_request["forecast_campaign"]["temporal"]["validation_origins"] = ["2024-01-21"]
    empty_request["forecast_campaign"]["models"]["candidates"] = [
        {"model_id": "last_value", "parameters": {"mode": "defaults"}},
    ]
    (empty_project / "campaign.yaml").write_text(yaml.safe_dump(empty_request))
    empty_source = empty_project / "data/observations.parquet"
    empty_table = pq.read_table(empty_source)
    empty_rows = empty_table.to_pylist()
    for row in empty_rows:
        if row["day"].isoformat() > empty_recipe["final_train_cutoff"]:
            row["units"] = None
    pq.write_table(pa.Table.from_pylist(empty_rows, schema=empty_table.schema), empty_source)
    empty_path = run_cli("freeze", "--project-root", empty_project, "--source", "data", cwd=project)
    empty_campaign = run_cli("run", "--project-root", empty_project,
                             "--benchmark", empty_path, cwd=project)
    empty_results = read_results(empty_campaign)
    assert empty_results.metadata["final_score"]["status"] == "unavailable"
    assert empty_results.metadata["final_score"]["score"] is None
    assert empty_results.rows.num_rows == 28  # Full validation and test forecast grids.
    assert empty_results.series_scores is not None
    heldout = [row for row in empty_results.series_scores.to_pylist() if row["stage"] == "test"]
    assert len(heldout) == 2 and all(row["scored"] == 0 for row in heldout)
    print("PASS: flat WRMSSE, coverage, export, resume, rescoring and complete-cohort preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
