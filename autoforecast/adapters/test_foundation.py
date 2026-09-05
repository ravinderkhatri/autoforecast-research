"""Public-boundary functional tests for the optional TimesFM 2.5 adapter.

The real model is intentionally not downloaded by this file.  A deterministic provider is
injected through ``fit_predict`` while the complete production boundary still runs: real Parquet,
Spark origin slicing, exposure handling, bounded provider calls, Parquet publication, Forecast
validation, and registry capability enforcement.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession

from autoforecast import registry, resources
from autoforecast.adapters import foundation as subject
from autoforecast.adapters import register_defaults, register_isolated
from autoforecast.adapters.foundation import MODEL_REVISION, NAME, fit_predict
from autoforecast.benchmarks.m5_contract import M5_HORIZON_DAYS
from autoforecast.panel import Panel

ORIGIN = "2020-01-04"
HORIZON = 2
_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[2]").appName("test_timesfm")
                  .config("spark.sql.shuffle.partitions", "2")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def _panel(rows: list[tuple], root: pathlib.Path | None = None) -> Panel:
    root = root or pathlib.Path(tempfile.mkdtemp())
    path = root / "panel.parquet"
    schema = pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("qty", pa.float64()),
        pa.field("state", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("category", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist([
        {"id": sid, "date": dt.date.fromisoformat(day), "qty": qty, "state": state,
         "price": price, "category": category}
        for sid, day, qty, state, price, category in rows
    ], schema=schema), path)
    return Panel(
        path=path,
        series_col="id",
        time_col="date",
        target_col="qty",
        exposure_col="state",
        known_future=("price",),
        static=("category",),
        season_length=7,
    ).validate()


def _rows(*, future_a: float = 900.0, future_price: float = 90.0) -> list[tuple]:
    rows = [
        ("a", "2020-01-01", 99.0, "unobserved", 1.0, "x"),
        ("a", "2020-01-02", 88.0, "unobserved", 1.0, "x"),
        ("a", "2020-01-03", 0.0, "exact", 1.0, "x"),
        ("a", "2020-01-04", 2.0, "exact", 1.0, "x"),
        ("b", "2020-01-01", 77.0, "unobserved", 1.0, "y"),
        ("b", "2020-01-02", 66.0, "unobserved", 1.0, "y"),
        ("b", "2020-01-03", 55.0, "unobserved", 1.0, "y"),
        ("b", "2020-01-04", 44.0, "unobserved", 1.0, "y"),
        ("c", "2020-01-01", 1.0, "exact", 1.0, "z"),
        ("c", "2020-01-02", 0.0, "exact", 1.0, "z"),
        ("c", "2020-01-03", 3.0, "lower_bound", 1.0, "z"),
        ("c", "2020-01-04", 0.0, "exact", 1.0, "z"),
        ("d", "2020-01-01", 0.0, "exact", 1.0, "q"),
        ("d", "2020-01-02", 0.0, "exact", 1.0, "q"),
        ("d", "2020-01-03", 0.0, "exact", 1.0, "q"),
        ("d", "2020-01-04", 0.0, "exact", 1.0, "q"),
    ]
    for sid in ("a", "b", "c", "d"):
        rows.extend([
            (sid, "2020-01-05", future_a if sid == "a" else 100.0,
             "exact", future_price, {"a": "future-x", "b": "future-y",
                                      "c": "future-z", "d": "future-q"}[sid]),
            (sid, "2020-01-06", 200.0, "exact", future_price, "future"),
        ])
    return rows


class RecordingBackend:
    """Distinct mean and median values make the 2.5 route executable in tests."""

    def __init__(self, *, negative_mean: bool = False) -> None:
        self.calls: list[list[list[float]]] = []
        self.negative_mean = negative_mean

    def forecast(self, inputs, horizon: int):
        materialized = [list(values) for values in inputs]
        self.calls.append(materialized)
        medians, quantiles = [], []
        for values in materialized:
            base = sum(values)
            median_row, quantile_row = [], []
            for step in range(horizon):
                median_row.append(100.0 + base + step)
                mean = base + step
                if self.negative_mean and step == 0:
                    mean = -1.0
                quantile_row.append([mean, *[10.0 + q + step for q in range(9)]])
            medians.append(median_row)
            quantiles.append(quantile_row)
        return medians, quantiles


def _run(panel: Panel, backend, *, statistic: str = "mean", batch_size: int = 2):
    out = pathlib.Path(tempfile.mkdtemp()) / "forecast"
    out.mkdir()
    forecast = fit_predict(
        panel,
        ORIGIN,
        HORIZON,
        spark=spark(),
        out_dir=out,
        context_days=4,
        batch_size=batch_size,
        forecast_statistic=statistic,
        backend=backend,
    )
    rows = sorted(pq.read_table(out).to_pylist(), key=lambda row: (row["series_id"], row["date"]))
    return forecast, rows


def test_public_adapter_trims_only_leading_unknowns_preserves_zeros_and_bounds_batches() -> None:
    backend = RecordingBackend()
    forecast, rows = _run(_panel(_rows()), backend)

    submitted = [values for call in backend.calls for values in call]
    assert submitted == [[0.0, 2.0], [1.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    assert max(map(len, backend.calls)) == 2
    assert forecast.n_series == 4
    assert forecast.disposition == {"emitted": 6, "declined_all_unobserved": 2}
    assert [(row["series_id"], row["date"]) for row in rows] == [
        (sid, dt.date(2020, 1, day)) for sid in ("a", "c", "d") for day in (5, 6)
    ]
    assert {row["timesfm_statistic"] for row in rows} == {"mean"}


def test_public_adapter_stages_range_sorted_context_without_task_result_transport() -> None:
    session = spark()
    coalesce_key = "spark.sql.adaptive.coalescePartitions.enabled"
    shuffle_key = "spark.sql.shuffle.partitions"
    previous_coalesce = session.conf.get(coalesce_key)
    previous_shuffle = session.conf.get(shuffle_key)
    session.conf.set(coalesce_key, "true")
    session.conf.set(shuffle_key, "4")
    root = pathlib.Path(tempfile.mkdtemp())
    panel = _panel(
        [
            ("only", f"2020-01-{day:02d}", float(day), "exact", 1.0, "x")
            for day in range(1, 13)
        ],
        root,
    )
    target = root / "forecast"
    target.mkdir()
    observed: list[tuple[int, str, int]] = []
    dataframe_type = type(session.range(1))
    writer_type = type(session.range(1).write)
    original_parquet = writer_type.parquet

    def observed_parquet(writer, *args, **kwargs):
        result = original_parquet(writer, *args, **kwargs)
        path = pathlib.Path(args[0])
        nonempty_parts = sum(
            pq.ParquetFile(part).metadata.num_rows > 0
            for part in path.glob("part-*.parquet")
        )
        observed.append(
            (
                writer._df.rdd.getNumPartitions(),
                writer._spark.conf.get(coalesce_key),
                nonempty_parts,
            )
        )
        return result

    backend = RecordingBackend()
    try:
        with (
            mock.patch.object(
                dataframe_type,
                "toLocalIterator",
                side_effect=AssertionError("forecast used Spark task-result transport"),
            ),
            mock.patch.object(writer_type, "parquet", observed_parquet),
        ):
            forecast = subject.fit_predict(
                panel,
                "2020-01-12",
                HORIZON,
                spark=session,
                out_dir=target,
                context_days=12,
                batch_size=2,
                backend=backend,
            )
        assert forecast.disposition == {"emitted": 2}
        assert backend.calls == [[[float(day) for day in range(1, 13)]]]
        assert len(observed) == 1 and observed[0][:2] == (4, "true"), observed
        assert observed[0][2] > 1, observed
        assert session.conf.get(coalesce_key) == "true"
        assert session.conf.get(shuffle_key) == "4"
        assert not (target / ".canonical-source").exists()
    finally:
        session.conf.set(coalesce_key, previous_coalesce)
        session.conf.set(shuffle_key, previous_shuffle)


def test_public_adapter_cleans_private_context_stage_after_stream_failure() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    panel = _panel(_rows(), root)
    target = root / "forecast"
    target.mkdir()
    backend = RecordingBackend()
    original_parquet_file = subject.pq.ParquetFile

    class FailingParquetFile:
        def __init__(self, path) -> None:
            self._delegate = original_parquet_file(path)

        def iter_batches(self, **_kwargs):
            raise RuntimeError("injected context staging read failure")

    try:
        with mock.patch.object(subject.pq, "ParquetFile", FailingParquetFile):
            subject.fit_predict(
                panel,
                ORIGIN,
                HORIZON,
                spark=spark(),
                out_dir=target,
                context_days=4,
                batch_size=2,
                backend=backend,
            )
    except RuntimeError as exc:
        assert str(exc) == "injected context staging read failure"
    else:
        raise AssertionError("injected context staging read failure did not escape")
    assert backend.calls == []
    assert list(target.iterdir()) == []
    assert list(root.glob(".forecast.timesfm.*")) == []


def test_public_adapter_cannot_see_future_targets_or_declared_covariates() -> None:
    first_backend, second_backend = RecordingBackend(), RecordingBackend()
    _, first = _run(_panel(_rows(future_a=900.0, future_price=90.0)), first_backend)
    _, second = _run(_panel(_rows(future_a=9_000_000.0, future_price=9_000.0)), second_backend)
    assert first_backend.calls == second_backend.calls
    assert first == second


def test_public_adapter_routes_timesfm_25_mean_and_median_without_conflating_them() -> None:
    mean_backend, median_backend = RecordingBackend(), RecordingBackend()
    _, means = _run(_panel(_rows()), mean_backend, statistic="mean")
    _, medians = _run(_panel(_rows()), median_backend, statistic="median")
    assert [row["yhat"] for row in means[:2]] == [2.0, 3.0]
    assert [row["yhat"] for row in medians[:2]] == [102.0, 103.0]
    assert {row["timesfm_statistic"] for row in medians} == {"median"}


def test_public_adapter_refuses_unknown_or_missing_time_inside_observed_history() -> None:
    cases = (
        [
            ("a", "2020-01-01", 1.0, "exact", 1.0, "x"),
            ("a", "2020-01-02", 0.0, "unobserved", 1.0, "x"),
            ("a", "2020-01-03", 2.0, "exact", 1.0, "x"),
            ("a", "2020-01-04", 3.0, "exact", 1.0, "x"),
        ],
        [
            ("a", "2020-01-01", 1.0, "exact", 1.0, "x"),
            ("a", "2020-01-03", 2.0, "exact", 1.0, "x"),
            ("a", "2020-01-04", 3.0, "exact", 1.0, "x"),
        ],
    )
    for rows in cases:
        backend = RecordingBackend()
        try:
            _run(_panel(rows), backend)
        except ValueError as exc:
            assert "compressing" in str(exc) or "not consecutive" in str(exc), str(exc)
            assert backend.calls == []
            continue
        raise AssertionError(f"accepted an interior unknown interval: {rows!r}")


def test_public_adapter_validates_provider_shape_values_and_publishes_atomically() -> None:
    class BadBackend:
        def __init__(self, result) -> None:
            self.result = result

        def forecast(self, _inputs, _horizon):
            return self.result

    cases = (
        ([[1.0]], [[[[1.0] * 10]]]),
        ([[1.0, 2.0]] * 3, [[[1.0] * 9 for _ in range(2)] for _ in range(3)]),
        (
            [[1.0, 2.0]] * 3,
            [[[float("nan"), *([1.0] * 9)] for _ in range(2)] for _ in range(3)],
        ),
        (
            [[1.0, 2.0]] * 3,
            [[[1.0] * 9 + [float("nan")] for _ in range(2)] for _ in range(3)],
        ),
    )
    panel = _panel([row for row in _rows() if row[0] != "b"])
    for result in cases:
        target = pathlib.Path(tempfile.mkdtemp()) / "forecast"
        target.mkdir()
        try:
            fit_predict(
                panel, ORIGIN, HORIZON, spark=spark(), out_dir=target,
                context_days=4, batch_size=3, backend=BadBackend(result))
        except ValueError:
            assert list(target.iterdir()) == [], "a failed provider published partial evidence"
            continue
        raise AssertionError(f"accepted malformed TimesFM output: {result!r}")


def test_public_adapter_preserves_negative_model_output_and_counts_every_cell() -> None:
    forecast, rows = _run(_panel(_rows()), RecordingBackend(negative_mean=True))
    assert forecast.disposition == {
        "emitted": 6,
        "declined_all_unobserved": 2,
    }
    assert sum(row["yhat"] == -1.0 for row in rows) == 3
    assert all(row["yhat"] != 0.0 for row in rows)


def test_public_provider_boundary_is_offline_pinned_and_defaults_to_1024_context() -> None:
    observed: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, **kwargs) -> None:
            observed["config"] = kwargs

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            observed["model_id"] = model_id
            observed["load"] = kwargs
            return cls()

        def compile(self, _config) -> None:
            observed["compiled"] = True

        def forecast(self, *, horizon, inputs):
            medians = [[1.0] * horizon for _ in inputs]
            quantiles = [[[2.0] * 10 for _ in range(horizon)] for _ in inputs]
            return medians, quantiles

    provider_module = SimpleNamespace(
        TimesFM_2p5_200M_torch=FakeModel,
        ForecastConfig=FakeConfig,
    )
    target = pathlib.Path(tempfile.mkdtemp()) / "forecast"
    target.mkdir()
    with mock.patch(
            "autoforecast.adapters.foundation.importlib.import_module",
            return_value=provider_module):
        fit_predict(
            _panel(_rows()), ORIGIN, M5_HORIZON_DAYS, spark=spark(), out_dir=target)

    assert observed["load"] == {
        "revision": MODEL_REVISION,
        "force_download": False,
        "local_files_only": True,
    }
    assert observed["config"]["max_context"] == 1024
    assert observed["config"]["per_core_batch_size"] == 32
    assert observed["compiled"] is True

    refused_target = pathlib.Path(tempfile.mkdtemp()) / "forecast"
    refused_target.mkdir()
    try:
        fit_predict(
            _panel(_rows()), ORIGIN, M5_HORIZON_DAYS, spark=spark(), out_dir=refused_target,
            model_revision="a" * 40)
    except ValueError as exc:
        assert "exact registered model ID" in str(exc)
        assert list(refused_target.iterdir()) == []
    else:
        raise AssertionError("a nonregistered TimesFM checkpoint reached inference")


def test_public_adapter_declines_all_unobserved_before_provider_construction() -> None:
    panel = _panel([
        ("a", "2020-01-01", 11.0, "unobserved", 1.0, "x"),
        ("a", "2020-01-02", 12.0, "unobserved", 1.0, "x"),
        ("a", "2020-01-03", 13.0, "unobserved", 1.0, "x"),
        ("a", "2020-01-04", 14.0, "unobserved", 1.0, "x"),
    ])
    target = pathlib.Path(tempfile.mkdtemp()) / "forecast"
    target.mkdir()
    with mock.patch(
            "autoforecast.adapters.foundation.importlib.import_module",
            side_effect=AssertionError("all-unobserved series constructed the provider")):
        forecast = fit_predict(
            panel, ORIGIN, M5_HORIZON_DAYS, spark=spark(), out_dir=target)
    assert forecast.disposition == {"declined_all_unobserved": M5_HORIZON_DAYS}
    assert pq.read_table(target).num_rows == 0


def test_public_isolated_registration_is_sales_only_and_does_not_import_provider() -> None:
    registry._reset_for_test()
    profile = resources.SparkProfile(
        cores=1, driver_memory="1g", shuffle_partitions=1, worker_reuse=False,
        series_per_partition=1, python_worker_budget_gb=1.0)
    before = set(sys.modules)
    register_isolated((NAME,), worker_profile=profile)
    entry = registry.get(NAME)
    assert entry.consumes_exposure is True
    assert entry.handles_right_censoring is False
    assert entry.supported_estimands == frozenset({"sales_given_availability"})
    assert entry.handicap and "right-censored" in entry.handicap
    assert "timesfm" not in (set(sys.modules) - before)

    latent = Panel(
        path="unused", series_col="id", time_col="date", target_col="qty",
        exposure_col="state", target_estimand="unconstrained_demand")
    try:
        registry.run(NAME, latent, ORIGIN, HORIZON)
    except registry.RegistryError as exc:
        assert "does not support target_estimand" in str(exc)
    else:
        raise AssertionError("isolated TimesFM ran for unconstrained demand")
    finally:
        registry._reset_for_test()


def test_public_default_registration_treats_timesfm_as_optional_and_rejects_legacy_api() -> None:
    registry._reset_for_test()
    legacy = SimpleNamespace(TimesFm=object())
    with mock.patch(
            "autoforecast.adapters.foundation.importlib.import_module", return_value=legacy):
        names = register_defaults(only=("foundation",), quiet=True)
    assert NAME not in names
    registry._reset_for_test()


def test_daily_adapter_refuses_weekly_panel_before_provider_or_spark_io() -> None:
    class ReadSentinel:
        @property
        def read(self):
            raise AssertionError("weekly panel reached TimesFM/Spark")

    weekly = Panel("/not-read", "id", "date", "qty", frequency="W")
    try:
        fit_predict(
            weekly,
            ORIGIN,
            HORIZON,
            spark=ReadSentinel(),
            out_dir="/not-written",
            backend=RecordingBackend(),
        )
    except ValueError as exc:
        assert "daily" in str(exc)
        return
    raise AssertionError("TimesFM accepted a weekly panel")


if __name__ == "__main__":
    failed = []
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_")
    ]
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001 - runner reports failed functions only
            failed.append((name, exc))
    if _SPARK is not None:
        _SPARK.stop()
    if failed:
        for name, exc in failed:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(f"PASS {len(tests)} TimesFM functional tests")
