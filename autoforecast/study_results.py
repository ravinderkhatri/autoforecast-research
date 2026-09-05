"""Read retained custom-model evidence into the existing analysis tables."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import cast

import pyarrow as pa
import pyarrow.dataset as ds
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

PUBLICATION_SCHEMAS = {
    "study_results": (
        "study_id string, study_sha string, run_id string, stage string, model_id string, "
        "workflow_path string, workflow_sha string, evidence_json string, model_uri string, "
        "model_sha string, mlflow_uri string, additional_info string, "
        "search_id string, trial_id string, attempt_id string, window_index int, "
        "origin string, horizon int"
    ),
    "forecasts": (
        "study_id string, study_sha string, run_id string, stage string, "
        "series_id string, date string, yhat double, "
        "search_id string, trial_id string, attempt_id string, window_index int, "
        "origin string, horizon int"
    ),
}


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
            "unsealed_attempts": unsealed, "selected": selected}
    return windows, runs, info, digest({"attempts": inventory, "unsealed": unsealed,
                                      "selection": selection_sha})


def _workflow_windows(workflow, *, include_tuning=True):
    """Collect one workflow's authenticated windows without loading forecast rows."""
    collected, runs = [], []
    search_sha = None
    if include_tuning and workflow.tuning_path is not None:
        windows, runs, search, search_sha = _search_windows(workflow.tuning_path)
        selected = search["selected"]
        saved_config = json.loads((workflow.path / "metadata/winner-config.json").read_bytes())
        context = search["provenance"]
        chosen = saved_config["source"]
        chosen_path = (workflow.tuning_path / "trials" / chosen["trial_id"]
                       / "attempts" / chosen["attempt_id"] / "metadata/winner-config.json")
        # Use values already checked by the loader; do not reopen mutable winner files.
        if (selected.parameters != workflow.parameters
                or selected.provenance != context
                or selected.trial_id != chosen["trial_id"]
                or selected.winner_config != chosen_path
                or context["profile_sha256"] != saved_config["profile_sha256"]
                or context["seed"] != saved_config["seed"]
                or context["study"] != workflow.provenance["study"]
                or context["implementation"] != workflow.provenance["implementation"]
                or context["model_id"] != workflow.validation["forecasts"].adapter):
            raise ValueError("tuning results belong to a different workflow configuration")
        identities = set()
        for window, trial in windows:
            # One fit must have one identity, even if a stored folder was copied.
            key = (trial["trial_id"], trial["attempt_id"], trial["window"])
            attempt_path = (workflow.tuning_path / "trials" / trial["trial_id"]
                            / "attempts" / trial["attempt_id"])
            if key in identities or pathlib.Path(trial["path"]) != attempt_path:
                raise ValueError("duplicate tuning window or mismatched attempt path")
            identities.add(key)
            details = {
                "search_id": workflow.tuning_path.name,
                "trial_id": trial["trial_id"], "attempt_id": trial["attempt_id"],
                "window_index": trial["window"],
                "model_uri": str(window["model_artifacts"]),
                "parameters": trial["parameters"], "trial_status": trial["status"],
                "trial_summary": trial["summary"], "search_sha": search_sha,
                "selected": (trial["trial_id"], trial["attempt_id"]) ==
                            (chosen["trial_id"], chosen["attempt_id"]),
            }
            collected.append((window, {**trial, **details}))
    for stage, evidence in (("validation", workflow.validation), ("test", workflow.testing)):
        if evidence is not None:
            collected.append((evidence, {
                "run_id": workflow.path.name, "stage": stage, "status": "completed",
                "path": str(workflow.path), "parameters": workflow.parameters,
            }))
    return collected, runs, search_sha


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
    windows, runs, search_identity = _workflow_windows(run, include_tuning=include_tuning)
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

@contextmanager
def _publication(definition, study_sha, workflow_path, *, scratch_root, mlflow_links=None):
    """Keep verified forecast snapshots alive until the destination has finished writing."""
    from autoforecast.run_mlflow import (
        RunMLflowLink,
        _completed_run_snapshot,
        _mlflow,
        _verify_workflow_export,
    )

    durable = pathlib.Path(workflow_path).absolute()
    with (
        _completed_run_snapshot(durable, scratch_root=scratch_root) as (root, completion),
        tempfile.TemporaryDirectory(dir=scratch_root, prefix="study-publication-") as temporary,
    ):
        workflow = load_forecast_workflow(root)
        if workflow.testing is None or completion["task"] != "forecast-workflow":
            raise ValueError("publication requires both completed workflow stages")
        if (workflow.study_id != definition["study_id"]
                or workflow.provenance["study"]["sha256"] != definition["library_sha256"]):
            raise ValueError("workflow belongs to a different registered study")
        binding = workflow.provenance.get("registration")
        if binding is not None:
            allowed = {(definition["study_id"], study_sha)}
            if parent := definition.get("parent"):
                allowed.add((parent["study_id"], parent["study_sha256"]))
            if (binding["study_id"], binding["sha256"]) not in allowed:
                raise ValueError("workflow registration binding differs from its baseline")
        manifest_sha = hashlib.sha256(
            (root / "artifact_manifest.json").read_bytes(),
        ).hexdigest()
        if mlflow_links is not None and set(mlflow_links) != {"validation", "test"}:
            raise ValueError("tracking links must cover both workflow stages")
        forecasts, results = [], []
        windows, _, search_sha = _workflow_windows(workflow)
        for index, (evidence, details) in enumerate(windows):
            stage = "tuning" if details["stage"] == "selection" else details["stage"]
            details = {**{key: details[key] for key in (
                "parameters", "search_id", "trial_id", "attempt_id", "window_index",
                "trial_status", "trial_summary", "search_sha", "selected",
            ) if key in details}, "model_uri": (
                details["model_uri"] if stage == "tuning" else str(durable / stage / "model")
            )}
            checked = snapshot_parquet_directory(
                pathlib.Path(evidence["forecasts"].path), pathlib.Path(temporary) / str(index),
                expected_sha256=evidence["forecast_sha256"], label="publication forecast",
            )
            evidence = {**evidence, "forecasts": replace(evidence["forecasts"], path=str(checked))}
            link = (mlflow_links or {}).get(stage)
            if link is not None:
                if not isinstance(link, RunMLflowLink):
                    raise ValueError("tracking links must be verified MLflow run links")
                client = _mlflow().tracking.MlflowClient(tracking_uri=link.tracking_uri)
                remote = client.get_run(link.run_id)
                if any(remote.data.tags.get(key) != value for key, value in {
                    "autoforecast.manifest_sha256": manifest_sha,
                    "autoforecast.workflow_id": durable.name,
                    "autoforecast.stage": stage,
                    "autoforecast.study_sha256": definition["library_sha256"],
                }.items()):
                    raise ValueError("tracking link belongs to different workflow evidence")
                _verify_workflow_export(client, link.run_id, evidence)
            forecast = evidence["forecasts"]
            labels = {
                "study_id": definition["study_id"], "study_sha": study_sha,
                "run_id": durable.name, "stage": stage,
                **{key: details.get(key) for key in (
                    "search_id", "trial_id", "attempt_id", "window_index",
                )}, "origin": forecast.origin, "horizon": forecast.horizon,
            }
            # Check every retained value before any destination starts writing.
            for _ in _forecast_batches(forecast.path):
                pass
            forecasts.append((forecast.path, labels))
            # Keep the exact retained score and its durable paths. The temporary
            # snapshot is only an authenticated transport, never a returned link.
            additional = {
                "model_format": evidence["model_format"],
                "evaluation_contract": evidence["evaluation_contract"],
                **details, "provenance": workflow.provenance,
                "tracking": link.as_record() if link is not None else None,
            }
            results.append((
                definition["study_id"], study_sha, durable.name, stage, forecast.adapter,
                str(durable), manifest_sha, json.dumps(evidence["score"], sort_keys=True),
                details["model_uri"], evidence["model_sha256"],
                f"runs:/{link.run_id}/models" if link is not None else None,
                json.dumps(additional, sort_keys=True),
                *[details.get(key) for key in (
                    "search_id", "trial_id", "attempt_id", "window_index",
                )],
                forecast.origin, forecast.horizon,
            ))
        # Reject a changed search before any table write. Scores stay as recorded.
        if (workflow.tuning_path is not None
                and _search_windows(workflow.tuning_path)[3] != search_sha):
            raise ValueError("tuning results changed during publication")
        scope = {"study_id": definition["study_id"], "study_sha": study_sha,
                 "run_id": durable.name}
        yield {
            "forecasts": forecasts, "study_results": results, "scope": scope,
            "status": {"status": "published", "study_id": definition["study_id"],
                       "study_sha256": study_sha, "run_id": durable.name,
                       "workflow_sha256": manifest_sha},
        }


def _forecast_batches(path):
    """Read the validated Hive layout and refuse lossy SQL floating-point values."""
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    required = ["series_id", "date", "yhat"]
    if missing := set(required).difference(dataset.schema.names):
        raise ValueError(f"publication forecast is missing required columns: {sorted(missing)}")
    error = "publication requires yhat values representable as float64 without precision loss"
    for batch in dataset.scanner(columns=required, batch_size=65536).to_batches():
        values = batch.column("yhat")
        if pa.types.is_floating(values.type):
            # Forecast validation already checked finiteness; widening floats is exact.
            yield batch.set_column(2, "yhat", values.cast(pa.float64()))
            continue
        numbers = []
        for value in values.to_pylist():
            try:
                original = Decimal(value) if isinstance(value, str) else value
                number = float(original)
            except (TypeError, ValueError, OverflowError, InvalidOperation) as exc:
                raise ValueError(error) from exc
            if isinstance(original, bool) or not math.isfinite(number) or original != number:
                raise ValueError(error)
            numbers.append(number)
        yield batch.set_column(2, "yhat", pa.array(numbers, type=pa.float64()))


def _publication_batches(publication, kind):
    """Yield portable Arrow batches, keeping model values and full window identities."""
    types = {"string": pa.string(), "int": pa.int32(), "double": pa.float64()}
    fields = [part.strip().split() for part in PUBLICATION_SCHEMAS[kind].split(",")]
    schema = pa.schema([(name, types[dtype]) for name, dtype in fields])
    if kind == "study_results":
        yield pa.RecordBatch.from_pylist(
            [dict(zip(schema.names, row, strict=True)) for row in publication[kind]], schema=schema,
        )
        return
    for path, labels in publication["forecasts"]:
        for batch in _forecast_batches(path):
            columns = {}
            for field in schema:
                if field.name in {"series_id", "date", "yhat"}:
                    column = batch.column(field.name)
                    if field.name == "date" and pa.types.is_timestamp(column.type):
                        # Hourly keys are exact hours. Safe casting refuses lost precision.
                        column = column.cast(pa.timestamp("s"))
                    columns[field.name] = column.cast(field.type)
                else:
                    columns[field.name] = pa.array(
                        [labels[field.name]] * batch.num_rows, type=field.type,
                    )
            yield pa.RecordBatch.from_arrays(list(columns.values()), schema=schema)
