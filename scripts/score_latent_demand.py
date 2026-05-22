"""Score an unconstrained-demand forecast against GOVERNED latent actuals -- the only route.

The scorers refuse to compare a latent-demand forecast with observed sales, because sales are
not the quantity. This script is the sanctioned other half: it loads a latent-actuals artifact
through `autoforecast.latent_actuals` -- which verifies the governance manifest (method,
source, registration) and the content hash before yielding the only panel latent scoring
accepts -- and scores the forecast against it. No manifest, an edited artifact, or a wrong
estimand refuses loudly.

Run:
  JAVA_HOME=... PYSPARK_PYTHON=... python3 -u scripts/score_latent_demand.py \
      --actuals <artifact root with latent_manifest.json + actuals/> \
      --forecast <forecast parquet> --origin 2016-04-24 --horizon 28 --n-series N
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoforecast import latent_actuals, resources
from autoforecast.panel import Forecast
from autoforecast.score_spark import score_forecast
from pyspark import SparkContext
from pyspark.sql import SparkSession


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--actuals", required=True)
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--adapter", default="mft")
    parser.add_argument("--origin", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--n-series", type=int, required=True)
    args = parser.parse_args(argv)

    panel = latent_actuals.load(args.actuals)
    print(f"governed latent actuals verified: {args.actuals}", flush=True)

    spark, owned = resources._acquire_embedded_spark_runtime(
        SparkSession, SparkContext, app_name="latent_score",
        profile_factory=lambda: resources.spark_profile(python_udfs=True))
    try:
        if owned:
            spark.sparkContext.setLogLevel("ERROR")
        return _run_with_spark(args, panel, spark)
    finally:
        if owned:
            spark.stop()


def _run_with_spark(args, panel, spark: SparkSession) -> int:
    """Score while respecting the caller's Spark ownership."""
    forecast = Forecast(adapter=args.adapter, path=args.forecast, origin=args.origin,
                        horizon=args.horizon, n_series=args.n_series,
                        disposition={"emitted": args.n_series * args.horizon},
                        target_estimand="unconstrained_demand").validate()
    score = score_forecast(panel, forecast, spark, missing_actual="no_data")
    print(score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
