"""Functional model-readiness coverage over one real, data-agnostic daily panel."""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from autoforecast.adapters import known_adapter_names
from autoforecast.data_observability import CheckReport, inspect_panel
from autoforecast.frequency import canonical_origin, period_points
from autoforecast.model_readiness import (
    approve_run,
    create_resolved_run,
    model_registration,
    prepare_model_check_input,
    run_model_checks,
)
from autoforecast.panel import Panel


def _spark():
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Spark functional test needs JAVA_HOME")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("test_model_readiness")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def test_every_registered_model_receives_prepared_model_check_input() -> None:
    """A complete client-like panel produces an approval for every implemented adapter."""
    with tempfile.TemporaryDirectory() as temporary:
        path = pathlib.Path(temporary) / "client_daily.parquet"
        start = dt.date(2023, 1, 1)
        days = [start + dt.timedelta(days=index) for index in range(1_024)]
        pq.write_table(
            pa.table(
                {
                    "item": [item for item in ("a", "b") for _ in days],
                    "date": pa.array(days * 2, type=pa.date32()),
                    "sales": [float(index % 11) for _ in ("a", "b") for index in range(len(days))],
                }
            ),
            path,
        )
        panel = Panel(str(path), "item", "date", "sales")
        spark = _spark()
        data_checks = inspect_panel(panel, spark, as_of=days[-1].isoformat())
        assert isinstance(data_checks, CheckReport) and data_checks.accepted
        for adapter in known_adapter_names():
            horizon = 28 if adapter in {"timesfm_standalone", "tft_q", "tft_hnb"} else 4
            run = create_resolved_run(
                model_id=adapter,
                adapter=adapter,
                origin=days[-1].isoformat(),
                horizon=horizon,
            )
            prepared = prepare_model_check_input(panel, spark, data_checks.facts, run)
            report = run_model_checks(prepared)
            approval = approve_run(data_checks, report, prepared)
            assert model_registration(adapter).adapter == adapter
            assert report.model_id == run.model_id and approval.approved
            assert approval.config_hash == run.config_hash
            assert approval.data_fingerprint == data_checks.facts.data_fingerprint


def test_tft_readiness_uses_complete_105_day_window_and_28_day_horizon() -> None:
    registration = model_registration("tft_q")
    run = create_resolved_run(
        model_id="tft_q", adapter="tft_q", origin="2016-04-24", horizon=28
    )
    panel = Panel("/unused", "item", "date", "sales")
    assert registration.required_history(panel, run) == 105
    assert registration.requires_complete_window
    assert registration.governed_horizon == 28


def test_create_resolved_run_refuses_noncanonical_origin_identity() -> None:
    adapter = known_adapter_names()[0]
    with pytest.raises(ValueError, match="canonical"):
        create_resolved_run(
            model_id=adapter,
            adapter=adapter,
            origin="20240101",
            horizon=4,
        )


@pytest.mark.parametrize(
    "frequency,anchor",
    [
        ("H", "2024-03-01T20:00:00"),
        ("D", "2024-03-01"),
        ("W", "2024-02-23"),
        ("M", "2023-12-31"),
        ("MS", "2024-01-01"),
        ("Y", "2021-12-31"),
        ("YS", "2022-01-01"),
    ],
)
def test_last_value_readiness_uses_complete_period_windows_for_every_cadence(
    tmp_path, frequency: str, anchor: str
) -> None:
    spark = _spark()
    points = period_points(anchor, frequency, 3)
    source = tmp_path / f"readiness-{frequency}"
    key_type = "timestamp_ntz" if frequency == "H" else "date"
    spark.createDataFrame(
        [("a", point, float(index)) for index, point in enumerate(points, 1)],
        f"item string, date {key_type}, sales double",
    ).write.parquet(str(source))
    panel = Panel(source, "item", "date", "sales", frequency=frequency)
    origin = canonical_origin(points[-1], frequency)
    data_checks = inspect_panel(panel, spark, as_of=origin)
    assert isinstance(data_checks, CheckReport) and data_checks.accepted
    run = create_resolved_run(
        model_id=f"last-{frequency}",
        adapter="last_value",
        origin=origin,
        horizon=1,
        frequency=frequency,
    )

    prepared = prepare_model_check_input(panel, spark, data_checks.facts, run)
    report = run_model_checks(prepared)

    assert prepared.series_with_complete_window == 1
    frequency_check = next(item for item in report.checks if item.id == "model_frequency")
    assert frequency_check.status == "passed"


@pytest.mark.parametrize(
    "frequency,anchor",
    [
        ("H", "2024-03-01T20:00:00"),
        ("D", "2024-03-01"),
        ("W", "2024-02-23"),
        ("M", "2023-12-31"),
        ("MS", "2024-01-01"),
        ("Y", "2021-12-31"),
        ("YS", "2022-01-01"),
    ],
)
def test_seasonal_readiness_rejects_a_missing_required_period_for_every_cadence(
    tmp_path, frequency: str, anchor: str
) -> None:
    spark = _spark()
    points = period_points(anchor, frequency, 3)
    source = tmp_path / f"readiness-gap-{frequency}"
    key_type = "timestamp_ntz" if frequency == "H" else "date"
    spark.createDataFrame(
        [("a", points[0], 1.0), ("a", points[2], 3.0)],
        f"item string, date {key_type}, sales double",
    ).write.parquet(str(source))
    panel = Panel(
        source, "item", "date", "sales", season_length=3, frequency=frequency
    )
    origin = canonical_origin(points[-1], frequency)
    data_checks = inspect_panel(panel, spark, as_of=origin)
    assert isinstance(data_checks, CheckReport) and data_checks.accepted
    assert next(item for item in data_checks.checks if item.id == "time_grain").status == "warning"
    run = create_resolved_run(
        model_id=f"seasonal-{frequency}",
        adapter="seasonal_naive",
        origin=origin,
        horizon=1,
        frequency=frequency,
    )

    prepared = prepare_model_check_input(panel, spark, data_checks.facts, run)
    report = run_model_checks(prepared)

    assert prepared.series_with_complete_window == 0
    assert next(item for item in report.checks if item.id == "model_recent_window").status == "failed"
    assert not approve_run(data_checks, report, prepared).approved


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
