"""Profile and validate one runtime dataset through Spark before model admission."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoforecast import resources
from autoforecast.data_admission import load_dataset_contract
from autoforecast.data_observability import (
    CheckReport,
    inspect_panel,
    run_incident_from_error,
    write_data_observability_report,
)
from pyspark import SparkContext
from pyspark.sql import SparkSession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_contract", type=pathlib.Path)
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    spark = resources.acquire_local_runtime(
        SparkSession, SparkContext, profile=resources.spark_profile(python_udfs=False),
        app_name="autoforecast_data_preflight",
    )
    try:
        panel = load_dataset_contract(args.dataset_contract).panel(args.data_path)
        try:
            report = inspect_panel(panel, spark, as_of=args.as_of)
        except Exception as exc:
            report = run_incident_from_error(exc)
        output = write_data_observability_report(report, args.output)
        if isinstance(report, CheckReport):
            print(f"Data check report: {output / 'report.html'}")
            return 0 if report.accepted else 2
        print(f"Run incident report: {output / 'report.html'}")
        return 2
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
