"""Scoring for verified studies, using the existing forecast validation and cell errors."""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import cast

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.benchmark import (
    MISSING_COLUMN,
    _evaluation_contract_from_record,
    artifact_path,
    verify_benchmark,
)
from autoforecast.data_admission import load_dataset_contract
from autoforecast.panel import Forecast, Panel
from autoforecast.score_spark import MissingActual, ScoreError, score_forecast
from provenance.lineage import directory_digest


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
) -> dict:
    """Score the full frozen grid and retain per-series coverage and unavailable scores."""
    path, study = verify_study_evaluation(contract)
    actuals = contract["actuals"]
    expected_panel = load_dataset_contract(path.parent / "dataset.yaml").panel(panel.path)
    if dataclasses.asdict(panel) != dataclasses.asdict(expected_panel):
        raise ScoreError("scoring panel roles differ from the frozen study")
    if directory_digest(panel.path, suffixes=(".parquet",)) != actuals["sha256"]:
        raise ScoreError("scoring actuals differ from the frozen study role")
    policy = contract["metric_policy"]
    missing = cast(
        MissingActual, "sparse_zero" if policy["actual_missing"] == "zero" else "no_data",
    )
    raw_path = output_dir / "cell-sums"
    result = score_forecast(
        panel, forecast, spark, missing_actual=missing,
        eligible_grid=contract["grid"]["path"], original_missing_column=MISSING_COLUMN,
        series_scores_path=str(raw_path),
    )
    sums = spark.read.parquet(str(raw_path))
    coverage = sums.select(
        "series_id", "expected", "originally_missing",
        F.col("n_zero_filled").alias("zero_filled"),
        F.col("n_no_data").alias("excluded"), F.col("n_scored").alias("scored"),
    )
    frozen = spark.read.parquet(contract["coverage"]["path"]).select(*coverage.columns)
    if (coverage.exceptAll(frozen).limit(1).count()
            or frozen.exceptAll(coverage).limit(1).count()):
        raise ScoreError("scoring coverage differs from the frozen denominator")
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
    if policy["metric"] == "wrmsse":
        references = spark.read.parquet(str(artifact_path(path, study, "scoring_references")))
        references = references.filter(F.col("boundary") == contract["boundary"])
        per_series = per_series.join(references.select("series_id", "weight", "scale"),
                                     "series_id", "left")
        per_series = per_series.withColumn(
            "rmsse", F.when((F.col("scored") > 0) & (F.col("scale") > 0),
                            F.col("rmse") / F.sqrt(F.col("scale"))),
        )
        totals = per_series.agg(
            F.count(F.when(F.col("rmsse").isNull(), 1)).alias("undefined"),
            F.sum("weight").alias("weight"),
            F.sum(F.col("weight") * F.col("rmsse")).alias("weighted"),
        ).first()
        assert totals is not None
        value = None if totals["undefined"] else float(totals["weighted"] / totals["weight"])
        if value is not None and not math.isfinite(value):
            raise ScoreError("weighted score overflowed")
        score["wrmsse"] = value
        per_series = per_series.withColumn("normalized_weight", F.col("weight") / totals["weight"])
        per_series = per_series.withColumn(
            "wrmsse_contribution", F.col("normalized_weight") * F.col("rmsse"),
        )
    series_path = output_dir / "series-scores"
    per_series.write.parquet(str(series_path))
    score["series_scores"] = {
        "path": str(series_path), "sha256": directory_digest(series_path, suffixes=(".parquet",)),
    }
    if directory_digest(panel.path, suffixes=(".parquet",)) != actuals["sha256"]:
        raise ScoreError("scoring actuals changed during evaluation")
    verify_study_evaluation(contract)
    return score
