"""Post-hoc scoring of verified campaign forecasts against another immutable study."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import shutil
import tempfile
from typing import cast

from pyspark.sql import SparkSession

from autoforecast.benchmark import STUDY_SCHEMA, evaluation_contract, verify_benchmark
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.forecast_campaign import _load_json_with_identity, _verified_forecast
from autoforecast.results import read_results
from autoforecast.run_evaluation import compare_run_forecasts
from provenance.lineage import directory_digest, snapshot_parquet_directory


def rescore_campaign(
    campaign_root: pathlib.Path, study_path: pathlib.Path, output_dir: pathlib.Path,
    spark: SparkSession,
) -> pathlib.Path:
    """Create separate scores; preserve the original selection, model IDs, runs and forecasts."""
    campaign_root, study_path = campaign_root.resolve(), study_path.resolve()
    original = read_results(campaign_root)
    study_sha = hashlib.sha256(study_path.read_bytes()).hexdigest()
    study_path, study = verify_benchmark(study_path, study_sha)
    if study["schema"] != STUDY_SCHEMA:
        raise ValueError("rescoring requires an explicitly frozen version-2 study")
    dataset_path = study_path.parent / "dataset.yaml"
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != study["dataset_sha256"]:
        raise ValueError("dataset contract changed before rescoring")
    dataset = _load_dataset_contract_bytes(dataset_bytes, dataset_path)
    output_dir = output_dir.resolve()
    if (output_dir.is_relative_to(campaign_root.resolve())
            or output_dir.is_relative_to(study_path.parent.resolve())):
        raise ValueError("post-hoc scores must be outside the original campaign and frozen study")
    runs = [run for run in original.runs
            if run["status"] == "completed" and run["stage"] in {"validation", "test"}]
    if not runs:
        raise ValueError("campaign has no verified validation or test forecasts")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        records = []
        for index, run in enumerate(runs):
            forecast, _ = _verified_forecast(
                pathlib.Path(run["forecast_record_path"]),
                expected_sha256=run["forecast_record_sha256"], expected_adapter=run["adapter"],
                expected_origin=run["origin"], expected_horizon=run["horizon"],
            )
            fixed = evaluation_contract(
                study_path, stage=run["stage"], origin=run["origin"], horizon=run["horizon"],
                metric=study["recipe"]["metric"],
            )
            if fixed["study"]["sha256"] != study_sha:
                raise ValueError("scoring study changed between retained forecasts")
            panel = dataset.panel(fixed["actuals"]["path"])
            with tempfile.TemporaryDirectory(prefix="autoforecast-rescore-") as temporary:
                forecast_snapshot = snapshot_parquet_directory(
                    pathlib.Path(forecast.path), pathlib.Path(temporary) / "forecast",
                    expected_sha256=run["forecast_artifact_sha256"], label="retained forecast",
                )
                comparison = compare_run_forecasts(
                    panel, {run["model_id"]: dataclasses.replace(
                        forecast, path=str(forecast_snapshot),
                    )}, spark, missing_actual="no_data", metric=study["recipe"]["metric"],
                    output_dir=output_dir / f"score-{index:04d}", fixed_contract=fixed,
                )
            comparison_record, comparison_sha = _load_json_with_identity(
                comparison, "new post-hoc comparison",
            )
            candidate = cast(list[dict], comparison_record["candidates"])[0]
            if candidate["status"] not in {"scored", "unavailable"}:
                raise ValueError(f"retained forecast could not be scored: {candidate['detail']}")
            if (directory_digest(forecast.path, suffixes=(".parquet",))
                    != run["forecast_artifact_sha256"]):
                raise ValueError("original forecast changed during rescoring")
            records.append({
                "model_id": run["model_id"], "variant_id": run["variant_id"],
                "run_id": run["run_id"], "origin": run["origin"], "stage": run["stage"],
                "forecast_record_sha256": run["forecast_record_sha256"],
                "forecast_artifact_sha256": run["forecast_artifact_sha256"],
                "comparison": str(comparison),
                "comparison_sha256": comparison_sha,
            })
        refreshed = read_results(campaign_root)
        if refreshed.metadata != original.metadata or refreshed.runs != original.runs:
            raise ValueError("original campaign changed during rescoring")
        verify_benchmark(study_path, study_sha)
        record = {
            "schema": "autoforecast.posthoc-rescore.v1", "evidence_role": "posthoc_rescore",
            "original_campaign": str(campaign_root.resolve()),
            "original_resolved_sha256": original.metadata["resolved_sha256"],
            "original_benchmark_id": original.metadata.get("benchmark_id"),
            "original_study_id": original.metadata.get("study_id"),
            "scoring_study_id": study["study_id"],
            "scoring_study": str(study_path),
            "scoring_study_sha256": study_sha,
            "scores": records,
        }
        result = output_dir / "rescore.json"
        result.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return result
    except BaseException:
        shutil.rmtree(output_dir)
        raise
