"""Verified, client-neutral rows from existing campaign artifacts; failures stay run-level."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from autoforecast._exact_json import load_exact_json as _load_exact_json
from autoforecast.benchmark import verify_benchmark
from autoforecast.campaign import CampaignStore
from autoforecast.forecast_campaign import (
    _completed_output,
    _load_json_with_identity,
    _verified_forecast,
)
from provenance.lineage import directory_digest

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


def load_exact_json(path: pathlib.Path) -> dict:
    value = _load_exact_json(path)
    if type(value) is not dict:
        raise ValueError("result evidence must be a JSON object")
    return value


@dataclass(frozen=True)
class CampaignResults:
    rows: pa.Table
    runs: tuple[dict, ...]
    metadata: dict
    source_keys: pa.Table | None


def read_results(campaign_root: str | pathlib.Path) -> CampaignResults:
    """Verify journal/output hashes and identities before exposing forecasts or run metadata."""
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
        "autoforecast.forecast-campaign-resolved.v3",
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
        "comparison_scope": "fixed_benchmark" if benchmark else "legacy_exploratory",
    }
    stages = cast(list[dict], store.state()["stages"])
    completed = {item["id"] for item in stages if item["status"] == "completed"}
    if "final_score" in completed:
        metadata["final_score"] = load_exact_json(
            _completed_output(store.root, "final_score", "final-score.json")
        )
    source_keys = None
    if benchmark:
        path, frozen = verify_benchmark(pathlib.Path(benchmark["path"]), benchmark["sha256"])
        if frozen["benchmark_id"] != benchmark["id"]:
            raise ValueError("campaign benchmark identity conflict")
        from autoforecast.data_admission import load_dataset_contract

        target = load_dataset_contract(path.parent / "dataset.yaml").panel(
            path.parent / "selection"
        )
        source_keys = pq.read_table(path.parent / "source-keys")
        metadata.update(
            {
                "benchmark_id": benchmark["id"],
                "target_unit": frozen["recipe"]["target_unit"],
                "target_estimand": target.target_estimand,
                "source_key_types": frozen["key_types"],
                "cohort_series_count": frozen["cohort_series_count"],
                "evaluations": benchmark["evaluations"],
                "actual_snapshots": {
                    role: {"path": str(path.parent / role), "sha256": frozen["artifacts"][role]}
                    for role in ("selection", "final_training", "final_score")
                },
            }
        )
    rows, runs = [], []
    children = {child["id"]: child for child in resolved["children"]}
    lock = None
    lock_sha256 = None
    if "aggregate_selection" in completed:
        lock_path = _completed_output(store.root, "aggregate_selection", "selection-lock.json")
        lock, lock_sha256 = _load_json_with_identity(lock_path, "selection lock")
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
        result = load_exact_json(result_path)
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
        if (
            directory_digest(forecast.path, suffixes=(".parquet",))
            != result["forecast_artifact_sha256"]
        ):
            raise ValueError("forecast artifact digest changed")
        run_id = pathlib.Path(result["run_path"]).name
        runs.append(
            {
                **base,
                **result,
                "run_id": run_id,
                "stage": stage_name,
                "declined_cell_identity": "unavailable; aggregate disposition only",
            }
        )
        for row in pq.read_table(forecast.path, columns=["series_id", "date", "yhat"]).to_pylist():
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
    table = pa.Table.from_pylist(rows, schema=ROW_SCHEMA)
    identities = [(r["run_id"], r["series_id"], r["target_date"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate result cell identity")
    return CampaignResults(table, tuple(runs), metadata, source_keys)


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
    figure, axis = plt.subplots(figsize=(12, 5))
    import datetime as dt

    for key, values in sorted(groups.items()):
        by_date = {row["target_date"]: row["prediction"] for row in values}
        run = next(item for item in results.runs if item.get("run_id") == key[2])
        origin = dt.date.fromisoformat(key[4])
        dates = [origin + dt.timedelta(days=i) for i in range(1, run["horizon"] + 1)]
        axis.plot(
            dates,  # pyright: ignore[reportArgumentType]  # Matplotlib accepts date units.
            [by_date.get(day, float("nan")) for day in dates],
            marker=".",
            label=" / ".join(key),
        )
    axis.set(
        title=f"{series_id} — {results.metadata['comparison_scope']}",
        ylabel=results.metadata.get("target_unit", "prediction"),
        xlabel="Target date",
    )
    axis.legend(fontsize="x-small")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output
