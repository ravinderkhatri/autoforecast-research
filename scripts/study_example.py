"""Freeze a tiny study, load its ID and check forecasts without model training."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from functools import partial
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from autoforecast import resources
from autoforecast.benchmark import freeze_benchmark, verify_benchmark
from autoforecast.frequency import canonical_origin, period_points
from autoforecast.panel import Forecast
from autoforecast.study import Study
from autoforecast.study import load_study as _load_study
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def example_forecast(
    study: Study, spark: SparkSession, output: pathlib.Path, stage: str,
) -> Forecast:
    """Stand in for a model's predictions using a constant, without reading actuals."""
    if stage not in {"validation", "test"}:
        raise ValueError("stage must be validation or test")
    recipe, panel = study.recipe, study.panel
    origin = (study.training_date_range if stage == "validation" else study.refit_date_range)[1]
    origin_text = canonical_origin(origin, panel.frequency)
    horizon = recipe["validation_horizon" if stage == "validation" else "final_test_horizon"]
    dates = spark.createDataFrame(
        [(point,) for point in period_points(origin_text, panel.frequency, horizon)],
        T.StructType([T.StructField(
            "date", T.TimestampNTZType() if panel.frequency == "H" else T.DateType(), False,
        )]),
    )
    keys = study.series_mapping.select(F.col(panel.series_col).alias("series_id"))
    predictions = keys.crossJoin(dates).withColumn("yhat", F.lit(2.0))
    predictions.write.parquet(str(output))
    return Forecast(
        adapter="study-example", path=str(output), origin=origin_text, horizon=horizon,
        n_series=study.n_series, disposition={"emitted": study.n_series * horizon},
        target_estimand=panel.target_estimand, frequency=panel.frequency,
    )


def run_example(project: pathlib.Path, spark: SparkSession) -> dict:
    """Use a caller-owned session; all inputs and results live in the example project."""
    project.mkdir(parents=True, exist_ok=False)
    print(
        "Population: two synthetic series, no exclusions; four required forecast cells "
        "per stage. Series B starts on day 3; its final actuals are absent and masked.",
        flush=True,
    )
    contract = project / "dataset.yaml"
    contract.write_text(yaml.safe_dump({
        "schema_version": 1, "dataset": {"id": "study-example", "panel": {
            "series_col": "shop", "time_col": "day", "target_col": "units",
            "frequency": "D", "season_length": 2, "target_estimand": "observed_nonnegative",
        }},
    }))
    source = project / "observations.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"shop": shop, "day": dt.date(2024, 1, day), "units": float(day % 4)}
        for shop, start, end in (("A", 1, 9), ("B", 3, 7))
        for day in range(start, end + 1)
    ]), source)
    recipe = {
        "schema_version": 2, "source_keys": ["shop"], "training_start": "2024-01-01",
        "validation_origins": ["2024-01-05"], "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07", "final_test_horizon": 2,
        "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
        "feature_availability": {}, "metric": "mae", "actual_missing": "mask",
        "series_starts": {"rule": "Declared before validation", "values": [
            {"keys": {"shop": "A"}, "start": "2024-01-01"},
            {"keys": {"shop": "B"}, "start": "2024-01-03"},
        ]},
    }
    registry = project / ".autoforecast/benchmarks"
    frozen = freeze_benchmark(
        contract_path=contract, recipe=recipe, output_root=registry, spark=spark,
        sources=dict.fromkeys(
            ("selection", "validation_score", "final_training", "final_score"), source,
        ),
    )
    study_id = verify_benchmark(frozen)[1]["study_id"]
    # Bind the environment once; model code then needs only the study ID.
    load_study = partial(_load_study, registry_root=registry, spark=spark)
    with load_study(study_id) as study:
        for stage in ("training", "validation", "refit", "testing"):
            frame = getattr(study, f"{stage}_data")
            print(stage, getattr(study, f"{stage}_date_range"), frame.count())
        print("forecasting", study.forecasting_date_range)
        study.series_starts.show(truncate=False)
        validation = example_forecast(
            study, spark, project / "validation-predictions", "validation",
        )
        testing = example_forecast(study, spark, project / "test-predictions", "test")
        result = study.evaluate_forecasts(
            validation_forecasts=validation, testing_forecasts=testing, model_id="constant-example",
        )
    print(json.dumps({stage: result[stage]["score"] for stage in ("validation", "test")}, indent=2))
    print(result["result_path"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=pathlib.Path, help="new directory for this example")
    args = parser.parse_args()
    spark, owned = resources._acquire_embedded_spark_runtime(
        SparkSession, SparkContext, app_name="study-example",
        profile_factory=lambda: resources.SparkProfile(
            cores=2, driver_memory="1g", shuffle_partitions=2, worker_reuse=True,
            series_per_partition=1, python_worker_budget_gb=1.0,
        ),
    )
    spark = cast(SparkSession, spark)
    try:
        run_example(args.project.resolve(), spark)
    finally:
        if owned:
            spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
