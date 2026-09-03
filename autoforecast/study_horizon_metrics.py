"""Accuracy diagnostics by forecast lead, using the study's frozen scoring references."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from autoforecast.frequency import parse_frequency, parse_origin, period_points
from autoforecast.panel import Forecast
from autoforecast.score_spark import ScoreError
from provenance.lineage import directory_digest

ForecastAccuracyBreakdown = Literal["per_step", "cumulative", "both", "none"]


def validate_forecast_accuracy_breakdown(value: object) -> None:
    """Reject unknown modes before allocating an evaluation run or reading inputs."""
    if type(value) is not str or value not in ("per_step", "cumulative", "both", "none"):
        raise ValueError(
            "forecast_accuracy_breakdown must be per_step, cumulative, both, or none"
        )


def score_horizon_metrics(
    path: Path, forecast: Forecast, spark: SparkSession, *, mode: ForecastAccuracyBreakdown,
    policy: dict, references: DataFrame | None,
    availability: dict | None = None,
) -> dict:
    """Reduce validated cell statistics; collect only one aggregate per requested lead.

    Cumulative scores use prefix sums within each series before the nonlinear RMSSE
    reduction. An unscoreable retained series makes that diagnostic WRMSSE unavailable.
    The caller has already enforced the full-window missing-actual policy.
    """
    digest = directory_digest(path, suffixes=(".parquet",))
    cadence = parse_frequency(forecast.frequency)
    points = period_points(forecast.origin, cadence, forecast.horizon)
    key_type = "timestamp_ntz" if cadence.key_kind == "timestamp" else "date"
    leads = spark.createDataFrame(
        [(point, lead) for lead, point in enumerate(points, 1)],
        f"date {key_type}, lead int",
    )
    cells = spark.read.parquet(str(path)).join(F.broadcast(leads), ["date"])
    common_dates = availability is not None and availability["series_policy"] == "use_available"
    if common_dates:
        assert availability is not None
        complete = F.col("date").isin([
            parse_origin(point, cadence) for point in availability["common_dates"]
        ])
        cells = cells.withColumn("wrmsse_n", F.when(complete, F.col("n_scored")).otherwise(0))
        cells = cells.withColumn("wrmsse_ss", F.when(
            complete, F.col("sum_squared_error"),
        ).otherwise(0.0))
    columns = (
        "expected", "originally_missing", "n_scored", "n_no_data", "n_zero_filled",
        "sum_abs_error", "sum_squared_error", "sum_error", "sum_actual",
    )
    if common_dates:
        columns += ("wrmsse_n", "wrmsse_ss")
    result = {
        "schema": "autoforecast.forecast-accuracy-breakdown.v1", "mode": mode,
        "frequency": cadence.code, "origin": forecast.origin, "horizon": forecast.horizon,
        "metric_definition": policy,
        "series_sums": {"path": str(path), "sha256": digest},
    }
    modes = ("per_step", "cumulative") if mode == "both" else (mode,)
    for kind in modes:
        values = cells
        if kind == "cumulative":
            window = Window.partitionBy("series_id").orderBy("lead").rowsBetween(
                Window.unboundedPreceding, Window.currentRow,
            )
            values = cells.select(
                "series_id", "lead", *[F.sum(name).over(window).alias(name) for name in columns],
            )
        expressions = [F.sum(name).alias(name) for name in columns]
        if references is not None:
            values = values.join(references.select("series_id", "weight", "scale"),
                                 "series_id", "left")
            n_name, ss_name = ("wrmsse_n", "wrmsse_ss") if common_dates else (
                "n_scored", "sum_squared_error",
            )
            rmsse = F.when(
                (F.col(n_name) > 0) & (F.col("scale") > 0),
                F.sqrt(F.col(ss_name) / F.col(n_name))
                / F.sqrt(F.col("scale")),
            )
            expressions.extend([
                F.count(F.when(F.col(n_name) == 0, 1)).alias("missing_series"),
                F.count(F.when(F.col("scale").isNull() | (F.col("scale") <= 0), 1))
                .alias("undefined_scales"),
                F.sum("weight").alias("weight"),
                F.sum(F.col("weight") * rmsse).alias("weighted"),
            ])
        rows = values.groupBy("lead").agg(*expressions).orderBy("lead").collect()
        metrics = []
        for row in rows:
            lead, n = row["lead"], row["n_scored"]
            volume = row["sum_actual"]
            score = {
                "start_lead": 1 if kind == "cumulative" else lead, "end_lead": lead,
                "start_time": points[0 if kind == "cumulative" else lead - 1].isoformat(),
                "end_time": points[lead - 1].isoformat(),
                "mae": row["sum_abs_error"] / n if n else None,
                "rmse": math.sqrt(row["sum_squared_error"] / n) if n else None,
                "wape": row["sum_abs_error"] / volume if volume else None,
                "bias": row["sum_error"] / n if n else None,
                "coverage": {
                    "expected": row["expected"], "originally_missing": row["originally_missing"],
                    "zero_filled": row["n_zero_filled"], "excluded": row["n_no_data"], "scored": n,
                },
            }
            reason = "no_observed_actuals" if not n else None
            if policy["metric"] == "wape" and n and not volume:
                reason = "zero_actual_volume"
            if references is not None:
                reason = ("missing_series_actuals" if row["missing_series"] else
                          "undefined_scale" if row["undefined_scales"] else None)
                score["wrmsse"] = None if reason else row["weighted"] / row["weight"]
                if common_dates:
                    assert availability is not None
                    if len(availability["eligible_ids"]) != availability["retained_series"]:
                        reason = "unfittable_weighted_series"
                        score["wrmsse"] = None
                    score["wrmsse_coverage"] = {
                        "scored_cells": row["wrmsse_n"],
                        "weighted_series": availability["retained_series"],
                    }
            if any(isinstance(value, float) and not math.isfinite(value)
                   for value in score.values()):
                raise ScoreError("horizon metric overflowed")
            score.update(status="unavailable" if score[policy["metric"]] is None else "scored",
                         unavailable_reason=reason)
            metrics.append(score)
        result[kind] = metrics
    if directory_digest(path, suffixes=(".parquet",)) != digest:
        raise ScoreError("horizon metric evidence changed during scoring")
    return result
