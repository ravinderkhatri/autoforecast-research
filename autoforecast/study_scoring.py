"""Scoring for verified studies, using the existing forecast validation and cell errors."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import pathlib
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.benchmark import (
    MISSING_COLUMN,
    _evaluation_contract_from_record,
    artifact_path,
    verify_benchmark,
    window_series_count,
)
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.frequency import parse_origin
from autoforecast.panel import Forecast, ForecastError, Panel
from autoforecast.ranking import SelectionError, _exact_grid_restrictions
from autoforecast.score_spark import MissingActualError, ScoreError, score_forecast
from autoforecast.study_horizon_metrics import (
    ForecastAccuracyBreakdown,
    score_horizon_metrics,
    validate_forecast_accuracy_breakdown,
)
from provenance.lineage import LineageError, directory_digest, snapshot_parquet_directory

SELECTION_METRICS = ("mae", "rmse", "wape", "wrmsse")
REPORTING_METRICS = (*SELECTION_METRICS, "bias")


def validate_study_metrics(metrics: object) -> tuple[str, ...]:
    """Validate the capabilities exposed by the study scorer, without guessing aliases."""
    if (not isinstance(metrics, list | tuple) or not metrics
            or any(type(name) is not str or name not in REPORTING_METRICS for name in metrics)
            or len(set(metrics)) != len(metrics)):
        raise ValueError("metrics must be a nonempty unique list of mae, rmse, wape, wrmsse, bias")
    return tuple(metrics)


def verify_study_evaluation(contract: dict) -> tuple[pathlib.Path, dict]:
    """Resolve an existing study and rederive the complete scoring contract from its bytes."""
    study = contract["study"]
    path, record = verify_benchmark(pathlib.Path(study["path"]), study["sha256"])
    stage, origin, horizon, _frequency = contract["boundary"].split("/")
    expected = _evaluation_contract_from_record(
        path, record, stage=stage, origin=origin, horizon=int(horizon),
        metric=contract["metric_policy"]["metric"],
    )
    if expected != contract:
        raise ScoreError("scoring contract differs from its verified study")
    return path, record


def score_study(
    panel: Panel, forecast: Forecast, spark: SparkSession, *, contract: dict,
    output_dir: pathlib.Path,
    forecast_accuracy_breakdown: ForecastAccuracyBreakdown = "none",
    metrics: tuple[str, ...] | None = None,
) -> dict:
    """Score the full frozen grid and retain per-series coverage and unavailable scores."""
    validate_forecast_accuracy_breakdown(forecast_accuracy_breakdown)
    if metrics is not None:
        metrics = validate_study_metrics(metrics)
    contract = copy.deepcopy(contract)
    forecast = Forecast.from_record(forecast.as_record())
    try:
        forecast.validate()
    except ForecastError as error:
        raise ScoreError(str(error)) from error
    path, study = verify_study_evaluation(contract)
    actuals = contract["actuals"]
    dataset_path = path.parent / "dataset.yaml"
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != study["dataset_sha256"]:
        raise ScoreError("dataset contract changed before scoring")
    dataset = _load_dataset_contract_bytes(dataset_bytes, dataset_path)
    if dataclasses.asdict(panel) != dataclasses.asdict(dataset.panel(panel.path)):
        raise ScoreError("scoring panel roles differ from the frozen study")
    if (forecast.n_series != window_series_count(study, contract["boundary"]) or forecast.n_declined
            or forecast.n_emitted != contract["grid"]["n_cells"]):
        raise ScoreError("forecast population or dispositions differ from the frozen cohort")
    forecast_sha = directory_digest(forecast.path, suffixes=(".parquet",))
    inputs = {
        "actuals": (pathlib.Path(panel.path), actuals["sha256"]),
        "forecast": (pathlib.Path(forecast.path), forecast_sha),
        "grid": (pathlib.Path(contract["grid"]["path"]), contract["grid"]["sha256"]),
        "coverage": (pathlib.Path(contract["coverage"]["path"]), contract["coverage"]["sha256"]),
    }
    if (metrics is not None and "wrmsse" in metrics
            or contract["metric_policy"]["metric"] == "wrmsse"):
        if "scoring_references" not in study["artifacts"]:
            raise ScoreError("requested WRMSSE requires registered scoring references")
        inputs["references"] = (artifact_path(path, study, "scoring_references"),
                                study["artifacts"]["scoring_references"])
    with tempfile.TemporaryDirectory(prefix="autoforecast-score-") as temporary:
        root = pathlib.Path(temporary)
        try:
            snapshots = {
                name: snapshot_parquet_directory(source, root / name,
                                                  expected_sha256=sha, label=name)
                for name, (source, sha) in inputs.items()
            }
        except LineageError as error:
            raise ScoreError(str(error)) from error
        private_panel = dataclasses.replace(panel, path=str(snapshots["actuals"]))
        private_forecast = dataclasses.replace(forecast, path=str(snapshots["forecast"]))
        score = _score_study_snapshot(
            private_panel, private_forecast, spark, contract=contract,
            snapshots=snapshots, output_dir=output_dir,
            forecast_accuracy_breakdown=forecast_accuracy_breakdown,
            metrics=metrics,
        )
    if (directory_digest(panel.path, suffixes=(".parquet",)) != actuals["sha256"]
            or directory_digest(forecast.path, suffixes=(".parquet",)) != forecast_sha):
        raise ScoreError("scoring inputs changed during evaluation")
    verify_study_evaluation(contract)
    return score


def _score_study_snapshot(
    panel: Panel, forecast: Forecast, spark: SparkSession, *, contract: dict,
    snapshots: dict[str, pathlib.Path], output_dir: pathlib.Path,
    forecast_accuracy_breakdown: ForecastAccuracyBreakdown = "none",
    metrics: tuple[str, ...] | None = None,
) -> dict:
    """Consume only the verified copies owned by this scoring operation."""
    try:
        _, count = _exact_grid_restrictions(
            {forecast.adapter: {forecast.origin: forecast}}, (forecast.origin,),
            {forecast.origin: str(snapshots["grid"])}, spark,
        )
    except SelectionError as error:
        raise ScoreError(str(error)) from error
    if count != contract["grid"]["n_cells"]:
        raise ScoreError("forecast grid differs from the frozen denominator")
    policy = contract["metric_policy"]
    available = contract.get("availability", {}).get("series_policy") == "use_available"
    need_horizon = forecast_accuracy_breakdown != "none" or available and "references" in snapshots

    raw_path = output_dir / "cell-sums"
    horizon_path = output_dir / "horizon-cell-sums"
    result = score_forecast(
        panel, forecast, spark, missing_actual="no_data",
        eligible_grid=str(snapshots["grid"]), original_missing_column=MISSING_COLUMN,
        series_scores_path=str(raw_path),
        horizon_sums_path=str(horizon_path) if need_horizon else None,
    )
    sums = spark.read.parquet(str(raw_path))
    coverage = sums.select(
        "series_id", "expected", "originally_missing",
        F.col("n_zero_filled").alias("zero_filled"),
        F.col("n_no_data").alias("excluded"), F.col("n_scored").alias("scored"),
    )
    frozen = spark.read.parquet(str(snapshots["coverage"])).select(*coverage.columns)
    if (coverage.exceptAll(frozen).limit(1).count()
            or frozen.exceptAll(coverage).limit(1).count()):
        raise ScoreError("scoring coverage differs from the frozen denominator")
    if policy["actual_missing"] == "error" and not available and result.n_dropped_no_data:
        raise MissingActualError(
            f"{contract['boundary']}: scoring requires complete actuals; "
            f"expected={count}, missing={result.n_dropped_no_data}"
        )
    n = F.col("n_scored")
    per_series = sums.select(
        "series_id", "expected", "originally_missing",
        F.col("n_zero_filled").alias("zero_filled"),
        F.col("n_no_data").alias("excluded"), n.alias("scored"),
        F.when(n > 0, F.col("sum_abs_error") / n).alias("mae"),
        F.when(n > 0, F.sqrt(F.col("sum_squared_error") / n)).alias("rmse"),
        F.when(F.col("sum_actual") > 0,
               F.col("sum_abs_error") / F.col("sum_actual")).alias("wape"),
    )
    score = {
        "n_scored": result.n_scored, "n_dropped_ineligible": result.n_dropped_ineligible,
        "n_dropped_no_data": result.n_dropped_no_data, "n_zero_filled": result.n_zero_filled,
        "mae": result.mae if result.n_scored else None,
        "rmse": result.rmse if result.n_scored else None,
        "wape": result.absolute_error_pct_of_volume if result.n_scored else None,
        "bias": result.bias if result.n_scored else None,
        "coverage": contract["coverage"]["counts"],
        "metric_definition": policy,
    }
    references = None
    wrmsse_reason = None
    if "references" in snapshots:
        if metrics is None and per_series.filter(F.col("scored") == 0).limit(1).count():
            raise MissingActualError(
                "WRMSSE requires observations for every retained series"
            )
        references = spark.read.parquet(str(snapshots["references"]))
        references = references.filter(F.col("boundary") == contract["boundary"])
        per_series = per_series.join(references.select("series_id", "weight", "scale"),
                                     "series_id", "left")
        if available:
            cells = spark.read.parquet(str(horizon_path))
            common = contract["availability"]["common_dates"]
            weighted = cells.filter(F.col("date").isin(
                [parse_origin(point, forecast.frequency) for point in common],
            )).groupBy(
                "series_id",
            ).agg(F.sum("n_scored").alias("wrmsse_scored"),
                  F.sum("sum_squared_error").alias("wrmsse_squared_error"))
            per_series = per_series.join(weighted, "series_id", "left")
            rmsse = F.sqrt(F.col("wrmsse_squared_error") / F.col("wrmsse_scored")
                           / F.col("scale"))
            if forecast.n_series != contract["availability"]["retained_series"]:
                wrmsse_reason = "unfittable_weighted_series"
            elif not common:
                wrmsse_reason = "no_common_complete_dates"
        else:
            rmsse = F.col("rmse") / F.sqrt(F.col("scale"))
        per_series = per_series.withColumn("rmsse", F.when(
            (F.col("scored") > 0) & (F.col("scale") > 0), rmsse,
        ))
        totals = per_series.agg(
            F.count(F.when(F.col("rmsse").isNull(), 1)).alias("undefined"),
            F.sum("weight").alias("weight"),
            F.sum(F.col("weight") * F.col("rmsse")).alias("weighted"),
        ).first()
        assert totals is not None
        if totals["undefined"] and wrmsse_reason is None:
            wrmsse_reason = "undefined_scale_or_missing_actuals"
        value = None if wrmsse_reason else float(totals["weighted"] / totals["weight"])
        if value is not None and not math.isfinite(value):
            raise ScoreError("weighted score overflowed")
        score["wrmsse"] = value
        weight_row = references.agg(F.sum("weight").alias("total")).first()
        assert weight_row is not None
        total_weight = weight_row["total"]
        per_series = per_series.withColumn("normalized_weight", F.col("weight") / total_weight)
        per_series = per_series.withColumn(
            "wrmsse_contribution", F.col("normalized_weight") * F.col("rmsse"),
        )
        if available:
            score["wrmsse_coverage"] = {
                "common_dates": contract["availability"]["common_dates"],
                "scored_dates": len(contract["availability"]["common_dates"]),
                "expected_dates": forecast.horizon,
                "weighted_series": contract["availability"]["retained_series"],
            }
    requested = metrics or tuple(name for name in REPORTING_METRICS if name in score)
    score["reported_metrics"] = list(requested)
    score["metric_status"] = {name: {
        "status": "scored" if score.get(name) is not None else "unavailable",
        "reason": (None if score.get(name) is not None else wrmsse_reason if name == "wrmsse"
                   else "zero_actual_volume" if name == "wape" and result.n_scored
                   else "no_observed_actuals"),
    } for name in requested}
    score["metric_definitions"] = {name: {
        "direction": "diagnostic" if name == "bias" else "minimize",
        "aggregation": ("common_complete_dates_fixed_weighted_cohort" if available
                        else "all_frozen_series") if name == "wrmsse" else "available_cells",
        "version": "single-level-wrmsse.v2" if name == "wrmsse" else "cell-errors.v2",
    } for name in requested}
    if "availability" in contract:
        score["availability"] = contract["availability"]
        score["coverage"] = dict(score["coverage"])
        score["coverage"]["original_expected"] = (
            contract["availability"]["original_series"] * forecast.horizon
        )
    original_count = contract.get("availability", {}).get("original_series", forecast.n_series)
    score["population"] = {
        "original_series": original_count, "trainable_series": forecast.n_series,
        "forecasted_series": forecast.n_series,
        "scored_series": per_series.filter(F.col("scored") > 0).count(),
        "expected_cells": original_count * forecast.horizon,
        "observed_cells": contract.get("availability", {}).get("observed_cells", result.n_scored),
        "filled_cells": result.n_zero_filled,
        "scored_cells": result.n_scored,
    }
    score["evaluation_status"] = (
        "pending_actuals" if not result.n_scored else
        "partial" if result.n_scored < original_count * forecast.horizon else "completed"
    )
    series_path = output_dir / "series-scores"
    per_series.write.parquet(str(series_path))
    score["series_scores"] = {
        "path": str(series_path), "sha256": directory_digest(series_path, suffixes=(".parquet",)),
    }
    if forecast_accuracy_breakdown != "none":
        score["forecast_accuracy_breakdown"] = score_horizon_metrics(
            horizon_path, forecast, spark, mode=forecast_accuracy_breakdown,
            policy=policy, references=references,
            availability=contract.get("availability"),
        )
    return score
