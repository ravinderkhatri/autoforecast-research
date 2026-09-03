"""Read retained custom-model evidence into the existing analysis tables."""

from __future__ import annotations

import hashlib
import math
import pathlib
import tempfile
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from mlops._safe_files import read_regular_bytes

from autoforecast.benchmark import artifact_path, digest, verify_benchmark
from autoforecast.experiments import load_completed_attempt
from autoforecast.frequency import parse_frequency
from autoforecast.hpo import HPOError
from autoforecast.results import ROW_SCHEMA, CampaignResults
from autoforecast.study_tuning import (
    _json_bytes,
    _retained_bytes,
    _retained_json,
    _retained_manifest,
    _window_result,
    load_tuning_result,
)
from autoforecast.study_workflow import load_forecast_workflow
from provenance.lineage import snapshot_parquet_directory


def _search_windows(path: pathlib.Path) -> tuple[list, list, dict, str]:
    """Authenticate selected searches or sealed failed attempts without fabricating a winner."""
    path = path.absolute()
    selection_path = path / "selection.json"
    selection_bytes = selection_path.read_bytes() if selection_path.exists() else None
    selected = load_tuning_result(path) if selection_bytes is not None else None
    windows, runs, inventory, unsealed = [], [], [], []
    context = None
    for attempt in sorted(path.glob("trials/*/attempts/*")):
        if not (attempt / "completion.json").exists():
            unsealed.append(str(attempt))
            continue
        manifest_bytes = read_regular_bytes(attempt, "artifact_manifest.json")
        _, completion = load_completed_attempt(attempt)
        manifest = _retained_manifest(completion, manifest_bytes)
        metadata = _retained_json(attempt, "metadata/attempt.json", manifest)
        evidence = _retained_json(attempt, "evaluation/external-run.json", manifest)
        if (
            type(evidence) is not dict
            or evidence.get("schema") not in {
                "autoforecast.external-trial.v1", "autoforecast.external-trial.v2",
            }
            or type(evidence.get("context")) is not dict
            or type(evidence.get("windows")) is not list
        ):
            raise HPOError("analysis requires current external trial context and windows")
        current = evidence["context"]
        profile = _retained_bytes(attempt, "metadata/external-profile.yaml", manifest)
        if (
            current.get("schema") != "autoforecast.external-tuning.v2"
            or hashlib.sha256(_json_bytes(current)).hexdigest()
            != metadata["comparability_fingerprint"]
            or current["study"]["sha256"] != metadata["data"]["fingerprint"]
            or current["study_id"] != metadata["data"]["id"]
            or hashlib.sha256(profile).hexdigest() != current["profile_sha256"]
            or metadata["model"]["config_sha256"] != current["profile_sha256"]
            or metadata["model"]["id"] != current["model_id"]
            or metadata["study_id"] != path.name
            or (context is not None and context != current)
        ):
            raise HPOError("retained attempt context, profile or search identity differs")
        context = current
        records = tuple(_window_result(attempt, item) for item in evidence["windows"])
        expected = current["windows"]
        if (
            len(records) > len(expected)
            or (completion["status"] == "completed" and len(records) != len(expected))
            or any(
                window["evaluation_contract"] != expected[index]["evaluation_contract"]
                or window["parameter_sha256"] != digest(metadata["hyperparameters"])
                or window["forecasts"].adapter != current["model_id"]
                for index, window in enumerate(records)
            )
        ):
            raise HPOError("retained window differs from its frozen trial contract")
        objectives = [window["score"].get(current["metric"])
                      if type(window["score"]) is dict else None for window in records]
        if any(type(value) not in (int, float) or not math.isfinite(cast(int | float, value))
               for value in objectives):
            raise HPOError("analysis requires a finite completed window objective")
        if completion["status"] == "completed" and (
            not objectives
            or type(completion["summary"].get("validation_objective")) not in (int, float)
            or math.fsum(cast(list[int | float], objectives)) / len(objectives)
            != completion["summary"]["validation_objective"]
        ):
            raise HPOError("completed attempt objective differs from retained window scores")
        run = {
            "run_id": f"{path.name}/{metadata['trial_id']}/{metadata['attempt_id']}",
            "trial_id": metadata["trial_id"], "attempt_id": metadata["attempt_id"],
            "status": completion["status"], "path": str(attempt),
            "parameters": metadata["hyperparameters"], "summary": completion["summary"],
            "artifacts_paths": [str(item) for item in sorted(attempt.glob("**/artifacts"))
                                if item.is_dir()],
        }
        if completion["status"] == "failed":
            run["failure"] = _retained_json(attempt, "evaluation/failure.json", manifest)
        runs.append(run)
        for index, window in enumerate(records):
            windows.append((window, {**run, "stage": "selection", "window": index,
                                     "variant_id": metadata["trial_id"]}))
        load_completed_attempt(attempt)
        if read_regular_bytes(attempt, "artifact_manifest.json") != manifest_bytes:
            raise HPOError("retained attempt changed during analysis")
        inventory.append((str(attempt), hashlib.sha256(manifest_bytes).hexdigest()))
    if context is None or (selected is None and not any(run["status"] == "failed" for run in runs)):
        raise HPOError("analysis requires a selected search or a sealed failed search attempt")
    if selected is not None and selected.provenance != context:
        raise HPOError("selection differs from retained attempt context")
    if (selection_path.read_bytes() if selection_path.exists() else None) != selection_bytes:
        raise HPOError("search selection changed during analysis")
    selection_sha = hashlib.sha256(selection_bytes).hexdigest() if selection_bytes else None
    info = {"provenance": context, "status": "completed" if selected else "failed",
            "unsealed_attempts": unsealed}
    return windows, runs, info, digest({"attempts": inventory, "unsealed": unsealed,
                                      "selection": selection_sha})


def _tables(windows: list, runs: list, metadata: dict) -> CampaignResults:
    """Snapshot verified Parquet bytes before exposing tables to analysis callers."""
    predictions, scores = [], []
    with tempfile.TemporaryDirectory(prefix="autoforecast-analysis-") as temporary:
        snapshots = pathlib.Path(temporary)
        counter = 0

        def read(path, sha):
            nonlocal counter
            counter += 1
            retained = snapshot_parquet_directory(
                pathlib.Path(path), snapshots / str(counter), expected_sha256=sha,
                label="analysis evidence",
            )
            return pq.read_table(retained)

        study = metadata["study"]
        study_path, frozen = verify_benchmark(pathlib.Path(study["path"]), study["sha256"])
        source_keys = read(artifact_path(study_path, frozen, "source-keys"),
                           frozen["artifacts"]["source-keys"])
        frequency = frozen["recipe"]["frequency"]
        schema = ROW_SCHEMA
        if parse_frequency(frequency).key_kind == "timestamp":
            schema = schema.set(schema.get_field_index("target_date"),
                                pa.field("target_date", pa.timestamp("us")))
        for window, run in windows:
            contract = window["evaluation_contract"]
            if contract["study"] != study:
                raise ValueError("analysis windows belong to different frozen studies")
            forecast = window["forecasts"]
            table = read(forecast.path, window["forecast_sha256"])
            labels = {
                "benchmark_id": frozen["benchmark_id"], "model_id": forecast.adapter,
                "variant_id": run.get("variant_id"), "run_id": run["run_id"],
                "stage": run["stage"], "origin": forecast.origin, "status": "emitted",
            }
            columns = {
                "target_date": table["date"], "series_id": table["series_id"],
                "prediction": table["yhat"],
                **{key: pa.array([value] * table.num_rows, type=pa.string())
                   for key, value in labels.items()},
            }
            predictions.append(pa.Table.from_arrays([columns[f.name].cast(f.type) for f in schema],
                                                    schema=schema))
            score = window["score"]
            evidence = score.get("series_scores")
            if evidence is not None:
                score_table = read(evidence["path"], evidence["sha256"])
                for key, value in labels.items():
                    if key == "status":
                        continue
                    score_table = score_table.append_column(
                        key, pa.array([value] * score_table.num_rows, type=pa.string()),
                    )
                scores.append(score_table)
            runs.append({
                **run, "origin": forecast.origin, "model_id": forecast.adapter,
                "variant_id": run.get("variant_id"),
                "horizon": forecast.horizon, "frequency": forecast.frequency,
                "forecast_path": forecast.path, "model_path": str(window["model_artifacts"]),
                "score": score, "evaluation_contract": contract,
                "parameter_sha256": window["parameter_sha256"],
                "artifacts_path": str(window["artifacts"]) if "artifacts" in window else None,
            })
        verify_benchmark(study_path, study["sha256"])
    metadata = {**metadata, "frequency": frequency, "comparison_scope": "fixed_study",
                "target_unit": frozen["recipe"]["target_unit"],
                "input_retention": "frozen study snapshots referenced by evaluation contracts",
                "cohort_series_count": frozen["cohort_series_count"]}
    return CampaignResults(
        pa.concat_tables(predictions) if predictions else pa.Table.from_pylist([], schema=schema),
        tuple(runs), metadata, source_keys,
        pa.concat_tables(scores, promote_options="default") if scores else None,
    )


def read_workflow_results(
    path: str | pathlib.Path, *, include_tuning: bool = True,
) -> CampaignResults:
    """Read completed workflow stages and, by default, its retained search attempts."""
    run = load_forecast_workflow(path)
    windows, runs = [], []
    search_identity = None
    if include_tuning and run.tuning_path is not None:
        windows, runs, _, search_identity = _search_windows(run.tuning_path)
    for stage, window in (("validation", run.validation), ("test", run.testing)):
        if window is not None:
            windows.append((window, {"run_id": run.path.name, "stage": stage,
                                     "status": "completed", "path": str(run.path)}))
    result = _tables(windows, runs, {
        "schema": "autoforecast.results.v1", "workflow_path": str(run.path),
        "study": run.provenance["study"], "provenance": run.provenance,
        "tuning_path": str(run.tuning_path) if run.tuning_path else None,
        "includes_tuning": include_tuning and run.tuning_path is not None,
    })
    load_forecast_workflow(path)
    if include_tuning and run.tuning_path is not None:
        if _search_windows(run.tuning_path)[3] != search_identity:
            raise HPOError("search attempts changed during analysis")
    return result


def read_tuning_results(path: str | pathlib.Path) -> CampaignResults:
    """Read a selected or failed search's authenticated completed windows and attempt metadata."""
    root = pathlib.Path(path).absolute()
    windows, runs, info, identity = _search_windows(root)
    result = _tables(windows, runs, {
        "schema": "autoforecast.results.v1", "tuning_path": str(root),
        "study": info["provenance"]["study"], "provenance": info["provenance"],
        "search_status": info["status"], "unsealed_attempts": info["unsealed_attempts"],
    })
    if _search_windows(root)[3] != identity:
        raise HPOError("search attempts changed during analysis")
    return result
