"""Verified, client-neutral rows from existing campaign artifacts; failures stay run-level."""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
from dataclasses import dataclass
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from autoforecast._exact_json import loads_exact_json
from autoforecast.benchmark import (
    _evaluation_contract_from_record,
    artifact_path,
    study_source_roles,
    verify_benchmark,
)
from autoforecast.campaign import CampaignStore
from autoforecast.forecast_campaign import (
    _completed_output,
    _load_completed_json,
    _load_json_with_identity,
    _verified_forecast,
)
from autoforecast.frequency import parse_frequency, period_points
from provenance.lineage import directory_digest, snapshot_parquet_directory

ROW_SCHEMA = pa.schema(
    [
        ("benchmark_id", pa.string()),
        ("model_id", pa.string()),
        ("variant_id", pa.string()),
        ("run_id", pa.string()),
        ("stage", pa.string()),
        ("origin", pa.string()),
        ("target_date", pa.date32()),
        ("series_id", pa.string()),
        ("prediction", pa.float64()),
        ("status", pa.string()),
    ]
)


@dataclass(frozen=True)
class CampaignResults:
    rows: pa.Table
    runs: tuple[dict, ...]
    metadata: dict
    source_keys: pa.Table | None
    series_scores: pa.Table | None = None


def _verified_dataset_frequency(
    resolved: dict, working_directory: pathlib.Path | None
) -> str | None:
    """Check the pinned dataset cadence when its source contract is available."""
    source = resolved.get("source")
    if type(source) is not dict:
        return None
    relative = source.get("dataset_contract")
    expected = source.get("dataset_contract_sha256")
    if type(relative) is not str or type(expected) is not str or working_directory is None:
        return None
    candidate = pathlib.Path(relative)
    path = candidate if candidate.is_absolute() else working_directory / candidate
    root = working_directory.resolve()
    try:
        resolved_path = path.resolve(strict=True)
    except OSError:
        return None
    if not resolved_path.is_file() or not resolved_path.is_relative_to(root):
        raise ValueError("campaign dataset contract path is invalid")
    raw = resolved_path.read_bytes()
    if len(expected) != 64 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("campaign dataset contract identity differs")
    from autoforecast.data_admission import _load_dataset_contract_bytes

    contract = _load_dataset_contract_bytes(raw, resolved_path)
    return parse_frequency(str(contract.panel_roles.get("frequency", "D"))).code


def _resolved_frequency(resolved: dict, contract_frequency: str | None) -> str:
    """Recover one cadence from authenticated temporal and dataset evidence."""
    temporal = resolved.get("temporal")
    if type(temporal) is not dict:
        raise ValueError("campaign temporal requires frequency without step_days")
    declared = temporal.get("frequency")
    if type(declared) is not str or "step_days" in temporal:
        raise ValueError("campaign temporal requires frequency without step_days")
    frequency = parse_frequency(declared).code
    if contract_frequency is not None and frequency != contract_frequency:
        raise ValueError("campaign frequency differs from the dataset contract")
    return frequency


def read_results(campaign_root: str | pathlib.Path) -> CampaignResults:
    """Verify journal/output hashes and identities before exposing forecasts or run metadata."""
    with tempfile.TemporaryDirectory(prefix="autoforecast-results-") as temporary:
        return _read_results(campaign_root, pathlib.Path(temporary))


def read_workflow_results(
    path: str | pathlib.Path, *, include_tuning: bool = True,
) -> CampaignResults:
    """Read verified custom-model workflow forecasts, scores and optional tuning windows."""
    from autoforecast.study_results import read_workflow_results as read

    return read(path, include_tuning=include_tuning)


def read_tuning_results(path: str | pathlib.Path) -> CampaignResults:
    """Read verified custom-model tuning attempts, including every retained window."""
    from autoforecast.study_results import read_tuning_results as read

    return read(path)


def _read_results(campaign_root: str | pathlib.Path, snapshots: pathlib.Path) -> CampaignResults:
    def read_verified_parquet(path: str | pathlib.Path, sha256: str, **kwargs) -> pa.Table:
        snapshot = snapshot_parquet_directory(
            pathlib.Path(path), snapshots / str(len(list(snapshots.iterdir()))),
            expected_sha256=sha256, label="result artifact",
        )
        return pq.read_table(snapshot, **kwargs)

    store = CampaignStore.from_root(campaign_root)
    store.verify_completed_outputs()
    command = store.plan.stages[0].command
    resolved_path = pathlib.Path(command[command.index("--resolved") + 1])
    expected = command[command.index("--resolved-sha256") + 1]
    resolved, resolved_sha256 = _load_json_with_identity(resolved_path, "campaign configuration")
    resolved = cast(dict, resolved)
    if resolved_sha256 != expected:
        raise ValueError("campaign configuration changed")
    if resolved.get("schema") not in {
        "autoforecast.forecast-campaign-resolved.v4",
        "autoforecast.forecast-campaign-resolved.v5",
    }:
        raise ValueError("unsupported campaign result schema")
    benchmark = resolved.get("benchmark")
    metadata = {
        "schema": "autoforecast.results.v1",
        "campaign_root": str(store.root),
        "resolved_sha256": expected,
        "source": resolved["source"],
        "metric_contract": resolved["selection"],
        "comparison_scope": "fixed_benchmark" if benchmark else "exploratory",
    }
    stages = cast(list[dict], store.state()["stages"])
    completed = {item["id"] for item in stages if item["status"] == "completed"}
    if "final_score" in completed:
        metadata["final_score"], _ = _load_completed_json(
            _completed_output(store.root, "final_score", "final-score.json")
        )
    source_keys = None
    contract_frequency = None
    if benchmark:
        path, frozen = verify_benchmark(pathlib.Path(benchmark["path"]), benchmark["sha256"])
        if frozen["benchmark_id"] != benchmark["id"]:
            raise ValueError("campaign benchmark identity conflict")
        from autoforecast.data_admission import _load_dataset_contract_bytes

        dataset_path = path.parent / "dataset.yaml"
        dataset_bytes = dataset_path.read_bytes()
        if hashlib.sha256(dataset_bytes).hexdigest() != frozen["dataset_sha256"]:
            raise ValueError("study dataset contract changed")
        target = _load_dataset_contract_bytes(dataset_bytes, dataset_path).panel(
            artifact_path(path, frozen, "selection")
        )
        contract_frequency = parse_frequency(target.frequency).code
        source_keys = read_verified_parquet(
            artifact_path(path, frozen, "source-keys"), frozen["artifacts"]["source-keys"]
        )
        metadata.update(
            {
                "benchmark_id": benchmark["id"],
                "target_unit": frozen["recipe"]["target_unit"],
                "target_estimand": target.target_estimand,
                "source_key_types": frozen["key_types"],
                "cohort_series_count": frozen["cohort_series_count"],
                "evaluations": benchmark["evaluations"],
                "actual_snapshots": {
                    role: {"path": str(artifact_path(path, frozen, role)),
                           "sha256": frozen["artifacts"][role]}
                    for role in study_source_roles(frozen["recipe"])
                    if role in frozen["artifacts"]
                },
            }
        )
        if "study_id" in frozen:
            metadata.update(
                study_id=frozen["study_id"], metric_contract=frozen["metric_definition"],
                missing_policies=frozen["policies"],
                history_starts=frozen["recipe"]["series_starts"],
                coverage=frozen["coverage_identities"], history_counts=frozen["counts"],
            )
    else:
        working_directory = getattr(store, "working_directory", None)
        contract_frequency = _verified_dataset_frequency(
            resolved,
            pathlib.Path(working_directory) if working_directory is not None else None,
        )
    frequency = _resolved_frequency(resolved, contract_frequency)
    rows, runs, series_scores = [], [], []
    children = {child["id"]: child for child in resolved["children"]}
    lock = None
    lock_sha256 = None
    if "aggregate_selection" in completed:
        lock_path = _completed_output(store.root, "aggregate_selection", "selection-lock.json")
        lock, lock_sha256 = _load_completed_json(lock_path)
        lock = cast(dict, lock)
        if lock["resolved_sha256"] != expected:
            raise ValueError("selection lock differs from campaign configuration")
    for stage in stages:
        if stage["id"] not in children and stage["id"] != "final_fit":
            continue
        child = children.get(stage["id"], {})
        stage_name = "test" if stage["id"] == "final_fit" else child.get("kind", "validation")
        base = {
            "campaign_id": store.root.name,
            "stage_id": stage["id"],
            "attempt": stage["attempts"],
            "status": stage["status"],
            "model_id": child.get("model_id"),
            "variant_id": child.get("variant_id"),
            "seed": child.get("seed"),
            "initialization": child.get("initialization"),
            "stage": stage_name,
        }
        if stage["id"] == "final_fit" and lock is not None:
            chosen = lock["chosen"]
            base.update(
                {
                    key: chosen[key]
                    for key in (
                        "model_id",
                        "variant_id",
                        "initialization",
                        "parameter_resolution",
                        "parameter_resolution_sha256",
                    )
                }
            )
            base.update(seed=chosen["final_seed"], selection_lock_sha256=lock_sha256)
        if stage["status"] != "completed":
            runs.append(base)
            continue
        result_path = _completed_output(store.root, stage["id"], "child-result.json")
        result, _ = _load_completed_json(result_path)
        result = cast(dict, result)
        if child and any(
            result[key] != child[key]
            for key in ("model_id", "variant_id", "seed", "origin", "horizon", "initialization")
        ):
            raise ValueError("child result identity differs from campaign")
        if stage["id"] == "final_fit":
            if (
                lock is None
                or lock_sha256 != result["selection_lock_sha256"]
                or lock["chosen"]["final_seed"] != result["seed"]
                or any(
                    lock["chosen"][key] != result[key]
                    for key in (
                        "model_id",
                        "variant_id",
                        "parameter_resolution_sha256",
                        "initialization",
                    )
                )
            ):
                raise ValueError("final result differs from validation selection lock")
        forecast, _ = _verified_forecast(
            pathlib.Path(result["forecast_record_path"]),
            expected_sha256=result["forecast_record_sha256"],
            expected_adapter=result["adapter"],
            expected_origin=result["origin"],
            expected_horizon=result["horizon"],
        )
        forecast_frequency = parse_frequency(forecast.frequency).code
        if frequency is not None and forecast_frequency != frequency:
            raise ValueError("forecast frequency differs from campaign results")
        frequency = forecast_frequency
        if (
            directory_digest(forecast.path, suffixes=(".parquet",))
            != result["forecast_artifact_sha256"]
        ):
            raise ValueError("forecast artifact digest changed")
        run_id = pathlib.Path(result["run_path"]).name
        if "study_id" in metadata:
            comparison_path = None
            if stage_name == "test" and "final_score" in completed:
                comparison_path = (
                    _completed_output(store.root, "final_score", "final-score.json").parent
                    / "evaluation/candidate-comparison.json"
                )
                comparison_sha = metadata["final_score"]["comparison_sha256"]
            elif stage_name == "validation":
                comparison_path = (
                    pathlib.Path(result["run_path"]) / "evaluation/candidate-comparison.json"
                )
                comparison_sha = result["comparison_sha256"]
            if comparison_path is not None:
                comparison_bytes = comparison_path.read_bytes()
                if hashlib.sha256(comparison_bytes).hexdigest() != comparison_sha:
                    raise ValueError("study comparison changed")
                comparison = loads_exact_json(comparison_bytes)
                if not isinstance(comparison, dict):
                    raise ValueError("study comparison must be an object")
                expected_contract = _evaluation_contract_from_record(
                    path, frozen, stage=stage_name, origin=result["origin"],
                    horizon=result["horizon"], metric=resolved["selection"]["metric"],
                )
                if comparison.get("evaluation_contract") != expected_contract:
                    raise ValueError("study comparison evaluation identity differs")
                candidates = comparison.get("candidates")
                if (not isinstance(candidates, list) or len(candidates) != 1
                        or candidates[0].get("model_id") != result["model_id"]):
                    raise ValueError("study per-series model identity differs")
                for candidate in candidates:
                    evidence = candidate.get("score", {}).get("series_scores")
                    if evidence is None:
                        raise ValueError("study comparison lacks per-series scores and coverage")
                    if (directory_digest(evidence["path"], suffixes=(".parquet",))
                            != evidence["sha256"]):
                        raise ValueError("study per-series scores changed")
                    table = read_verified_parquet(evidence["path"], evidence["sha256"])
                    for name, value in {
                        "study_id": metadata["study_id"], "model_id": result["model_id"],
                        "variant_id": result["variant_id"], "run_id": run_id,
                        "stage": stage_name, "origin": result["origin"],
                    }.items():
                        table = table.append_column(name, pa.array([value] * table.num_rows))
                    series_scores.append(table)

        runs.append(
            {
                **base,
                **result,
                "run_id": run_id,
                "frequency": frequency,
                "stage": stage_name,
                "declined_cell_identity": "unavailable; aggregate disposition only",
            }
        )
        forecast_rows = read_verified_parquet(
            forecast.path, result["forecast_artifact_sha256"], columns=["series_id", "date", "yhat"]
        )
        for row in forecast_rows.to_pylist():
            rows.append(
                {
                    "benchmark_id": benchmark["id"] if benchmark else None,
                    "model_id": result["model_id"],
                    "variant_id": result["variant_id"],
                    "run_id": run_id,
                    "stage": stage_name,
                    "origin": result["origin"],
                    "target_date": row["date"],
                    "series_id": str(row["series_id"]),
                    "prediction": row["yhat"],
                    "status": "emitted",
                }
            )
    store.verify_completed_outputs()
    metadata["frequency"] = frequency or "D"
    schema = ROW_SCHEMA
    if parse_frequency(metadata["frequency"]).key_kind == "timestamp":
        schema = schema.set(
            schema.get_field_index("target_date"), pa.field("target_date", pa.timestamp("us"))
        )
    table = pa.Table.from_pylist(rows, schema=schema)
    identities = [(r["run_id"], r["series_id"], r["target_date"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate result cell identity")
    return CampaignResults(
        table, tuple(runs), metadata, source_keys,
        pa.concat_tables(series_scores) if series_scores else None,
    )


def export_results(results: CampaignResults, output: pathlib.Path) -> pathlib.Path:
    """Publish an idempotent local export; conflicting retries are refused."""
    from mlops._safe_files import publish_bytes_set

    stream = pa.BufferOutputStream()
    pq.write_table(results.rows, stream)
    payloads = {
        "predictions.parquet": stream.getvalue().to_pybytes(),
        "metadata.json": (
            json.dumps({**results.metadata, "runs": results.runs}, sort_keys=True, indent=2) + "\n"
        ).encode(),
    }
    if results.source_keys is not None:
        stream = pa.BufferOutputStream()
        pq.write_table(results.source_keys, stream)
        payloads["source-keys.parquet"] = stream.getvalue().to_pybytes()
    if results.series_scores is not None:
        stream = pa.BufferOutputStream()
        pq.write_table(results.series_scores, stream)
        payloads["series-scores.parquet"] = stream.getvalue().to_pybytes()
    from mlops.run_report import run_guide

    payloads["RUN.md"] = run_guide(
        identity={"kind": "derived analysis export"}, status="exported",
        files=[*payloads, "RUN.md"], sealed=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    publish_bytes_set(output, payloads)
    return output


def comparison_plot(
    results: CampaignResults, *, series_id: str, output: pathlib.Path
) -> pathlib.Path:
    """Plot exactly the reader's exported predictions; variants and stages have distinct labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in results.rows.to_pylist() if row["series_id"] == series_id]
    if not selected:
        raise ValueError("series has no emitted predictions")
    groups = {}
    for row in selected:
        key = (row["model_id"], row["variant_id"], row["run_id"], row["stage"], row["origin"])
        groups.setdefault(key, []).append(row)
    traces = []
    identity = ("model_id", "variant_id", "run_id", "stage", "origin")
    for key, values in sorted(groups.items(), key=lambda item: tuple(
        "" if value is None else value for value in item[0]
    )):
        by_date = {row["target_date"]: row["prediction"] for row in values}
        matches = [item for item in results.runs
                   if tuple(item.get(name) for name in identity) == key]
        if len(matches) != 1:
            raise ValueError("plot requires exactly one matching forecast window record")
        run = matches[0]
        dates = period_points(key[4], run["frequency"], run["horizon"])
        traces.append((dates, [by_date.get(day, float("nan")) for day in dates],
                       " / ".join("(no variant)" if part is None else part for part in key)))
    figure, axis = plt.subplots(figsize=(12, 5))
    try:
        for dates, predictions, label in traces:
            axis.plot(
                dates,  # pyright: ignore[reportArgumentType]  # Matplotlib accepts date units.
                predictions, marker=".", label=label,
            )
        axis.set(
            title=f"{series_id} — {results.metadata['comparison_scope']}",
            ylabel=results.metadata.get("target_unit", "prediction"),
            xlabel="Target time",
        )
        axis.legend(fontsize="x-small")
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output)
    finally:
        plt.close(figure)
    return output
