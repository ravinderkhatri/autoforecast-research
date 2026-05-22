"""Run two short, comparable local custom-model attempts for observability inspection.

This is development smoke tooling on a generated six-series panel.  It proves artifact shape and
telemetry wiring only; it is explicitly not M5 evidence, not an HPO study, and does not score a
test set.  The two trials differ only in learning rate and use the same data/split/model contract.

Run with the normal Spark environment, for example:

    JAVA_HOME=... .venv/bin/python scripts/run_observability_smoke.py
"""

from __future__ import annotations

# ruff: noqa: E402
import datetime as dt
import hashlib
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# A local Spark worker must use the driver's minor Python version.  This smoke command is
# self-contained, so bind it explicitly instead of inheriting a system-wide PySpark default.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from autoforecast.adapters import mft as ours
from autoforecast.experiment_report import render_study_report
from autoforecast.experiments import open_local_attempt
from autoforecast.panel import Panel
from pyspark import SparkContext
from pyspark.sql import SparkSession

OUT = ROOT / "gates" / "autoforest" / "hpo" / "observability-smoke-v2"
STUDY_ID = "observability-smoke-v2"
ATTEMPT_ID = "attempt1"
ORIGIN = dt.date(2024, 2, 12)
HORIZON = 2
EPOCHS = 4
SEED = 17


def _rows() -> list[tuple[str, dt.date, float, str]]:
    day0 = dt.date(2024, 1, 1)
    return [
        (f"series{series}", day0 + dt.timedelta(days=offset),
         float((series * 2 + offset) % 5), "exact")
        for series in range(6) for offset in range(47)
    ]


def _data_fingerprint(rows: list[tuple[str, dt.date, float, str]]) -> str:
    canonical = "\n".join(
        f"{series}|{date.isoformat()}|{quantity:.1f}|{state}"
        for series, date, quantity, state in rows
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _metadata(learning_rate: float, data_fingerprint: str) -> dict[str, object]:
    return {
        "purpose": "development_observability_smoke_not_claim_bearing",
        "comparability_fingerprint": hashlib.sha256(
            f"synthetic-v1|{data_fingerprint}|{ORIGIN}|{HORIZON}|{EPOCHS}|{SEED}".encode()
        ).hexdigest(),
        "model": {"id": "mft", "adapter": "mft"},
        "hyperparameters": {
            "model.d_model": 8,
            "training.epochs_max": EPOCHS,
            "training.learning_rate": learning_rate,
            "training.seed": SEED,
            "training.batch_size": 16,
        },
        "data": {
            "name": "generated-six-series-smoke-v1",
            "fingerprint": data_fingerprint,
            "observed_date_range": ["2024-01-01", "2024-02-16"],
            "training_date_range": ["2024-01-01", ORIGIN.isoformat()],
            "validation_date_range": ["2024-01-01", ORIGIN.isoformat()],
            "test_date_range": ["2024-02-13", "2024-02-14"],
            "validation_target_date_range": ["2024-02-11", "2024-02-12"],
            "test_target_date_range": ["2024-02-13", "2024-02-14"],
            "training_series": 5,
            "validation_series": 1,
            "validation_split": "series_holdout(seed=17, fraction=1/6)",
            "test": "reserved_not_evaluated",
        },
    }


def main() -> int:
    if OUT.exists():
        raise RuntimeError(
            f"{OUT} already exists; this smoke command never overwrites an experiment record")
    rows = _rows()
    fingerprint = _data_fingerprint(rows)
    spark = (SparkSession.builder.master("local[2]").appName("observability_smoke")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        data_path = OUT / "_input"
        spark.createDataFrame(
            rows, "id string, date date, qty double, observation_state string",
        ).write.parquet(str(data_path))
        panel = Panel(
            path=str(data_path), series_col="id", time_col="date", target_col="qty",
            exposure_col="observation_state", season_length=7,
        ).validate()
        for number, learning_rate in enumerate((0.001, 0.003), start=1):
            attempt = open_local_attempt(
                OUT, study_id=STUDY_ID, trial_id=f"lr-{number:04d}", attempt_id=ATTEMPT_ID,
                metadata=_metadata(learning_rate, fingerprint),
            )
            forecast = ours.fit_predict(
                panel, ORIGIN.isoformat(), HORIZON, spark=spark,
                out_dir=attempt.path / "evaluation" / "forecast",
                checkpoint_out=attempt.path / "model" / "model.pt",
                training_log_dir=attempt.tensorboard_dir,
                training_metrics_path=attempt.metrics_path,
                epochs=EPOCHS, patience=EPOCHS, d_model=8, n_draws=2, n_paths=2,
                seed=SEED, batch_size=16, gradient_microbatch_size=16, lr=learning_rate,
                price_ladder="skip", quiet_epoch_logging=True,
                _feature_scratch=attempt.path / "_features",
            )
            attempt.write_json("evaluation/result.json", {
                "schema": "autoforecast.development-smoke-evaluation.v1",
                "forecast": {"adapter": forecast.adapter, "n_series": forecast.n_series,
                             "horizon": HORIZON},
                "test_metrics": "not_evaluated",
            })
            attempt.finalize(status="completed", summary={
                "test_metrics": "not_evaluated",
                "selection_metric": "validation/objective",
            })
        report = render_study_report(OUT / "studies" / STUDY_ID)
        print(f"Study report: {report}")
        print(f"TensorBoard: tensorboard --logdir {OUT / 'studies' / STUDY_ID}")
        return 0
    finally:
        spark.stop()
        SparkContext._active_spark_context = None


if __name__ == "__main__":
    raise SystemExit(main())
