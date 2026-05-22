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
)
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.panel import Forecast, ForecastError, Panel
from autoforecast.ranking import SelectionError, _exact_grid_restrictions
from autoforecast.score_spark import MissingActualError, ScoreError, score_forecast
from provenance.lineage import LineageError, directory_digest, snapshot_parquet_directory


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
    if (forecast.n_series != study["cohort_series_count"] or forecast.n_declined
            or forecast.n_emitted != contract["grid"]["n_cells"]):
        raise ScoreError("forecast population or dispositions differ from the frozen cohort")
    forecast_sha = directory_digest(forecast.path, suffixes=(".parquet",))
    inputs = {
        "actuals": (pathlib.Path(panel.path), actuals["sha256"]),
        "forecast": (pathlib.Path(forecast.path), forecast_sha),
        "grid": (pathlib.Path(contract["grid"]["path"]), contract["grid"]["sha256"]),
        "coverage": (pathlib.Path(contract["coverage"]["path"]), contract["coverage"]["sha256"]),
    }
    if contract["metric_policy"]["metric"] == "wrmsse":
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
        )
    if (directory_digest(panel.path, suffixes=(".parquet",)) != actuals["sha256"]
            or directory_digest(forecast.path, suffixes=(".parquet",)) != forecast_sha):
        raise ScoreError("scoring inputs changed during evaluation")
    verify_study_evaluation(contract)
    return score


def _score_study_snapshot(
    panel: Panel, forecast: Forecast, spark: SparkSession, *, contract: dict,
    snapshots: dict[str, pathlib.Path], output_dir: pathlib.Path,
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

    raw_path = output_dir / "cell-sums"
    result = score_forecast(
        panel, forecast, spark, missing_actual="no_data",
        eligible_grid=str(snapshots["grid"]), original_missing_column=MISSING_COLUMN,
        series_scores_path=str(raw_path),
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
    if policy["actual_missing"] == "error" and result.n_dropped_no_data:
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
    if policy["metric"] == "wrmsse":
        if per_series.filter(F.col("scored") == 0).limit(1).count():
            raise MissingActualError(
                "WRMSSE requires observations for every retained series"
            )
        references = spark.read.parquet(str(snapshots["references"]))
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
    return score
