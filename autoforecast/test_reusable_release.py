"""Behavior-facing acceptance tests for the reusable release's new authority boundaries."""

import datetime as dt
import json
import os
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark, resources
from autoforecast.cli import create_example
from autoforecast.data_admission import load_dataset_contract
from autoforecast.panel import Forecast
from autoforecast.run_evaluation import EvaluationError, compare_run_forecasts


@pytest.fixture(scope="module")
def spark():
    from pyspark import SparkContext
    from pyspark.sql import SparkSession

    profile = resources.SparkProfile(
        cores=2,
        driver_memory="1g",
        shuffle_partitions=2,
        worker_reuse=True,
        series_per_partition=10,
        python_worker_budget_gb=1,
    )
    session = resources.acquire_local_runtime(
        SparkSession, SparkContext, profile=profile, app_name="reusable-release-tests"
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    from scripts.run_autoforecast import _shutdown_owned_preflight_runtime

    _shutdown_owned_preflight_runtime(session)


def _inputs(tmp_path):
    project = tmp_path / "project"
    create_example(project)
    recipe = json.loads((project / "recipe.json").read_text())
    return (
        project,
        recipe,
        {role: project / "data" for role in ("selection", "final_training", "validation_score", "final_score")},
    )


def _freeze(project, recipe, sources, spark):
    return benchmark.freeze_benchmark(
        contract_path=project / "dataset.yaml",
        sources=sources,
        recipe=recipe,
        output_root=project / "frozen",
        spark=spark,
    )


@pytest.fixture(scope="module")
def frozen(tmp_path_factory, spark):
    project, recipe, sources = _inputs(tmp_path_factory.mktemp("frozen"))
    path = _freeze(project, recipe, sources, spark)
    return path, project, recipe, sources


@pytest.mark.parametrize("frequency,start,end,expected", [
    ("H", "2024-02-29T23:00:00", "2024-03-01T01:00:00",
     ["2024-02-29T23:00:00", "2024-03-01T00:00:00", "2024-03-01T01:00:00"]),
    ("D", "2024-02-28", "2024-03-01", ["2024-02-28", "2024-02-29", "2024-03-01"]),
    ("W", "2024-02-23", "2024-03-08", ["2024-02-23", "2024-03-01", "2024-03-08"]),
    ("3D", "2024-02-26", "2024-03-03", ["2024-02-26", "2024-02-29", "2024-03-03"]),
    ("M", "2024-02-29", "2024-04-30", ["2024-02-29", "2024-03-31", "2024-04-30"]),
    ("MS", "2023-12-01", "2024-02-01", ["2023-12-01", "2024-01-01", "2024-02-01"]),
    ("Y", "2099-12-31", "2101-12-31", ["2099-12-31", "2100-12-31", "2101-12-31"]),
    ("YS", "2099-01-01", "2101-01-01", ["2099-01-01", "2100-01-01", "2101-01-01"]),
])
def test_historical_grid_is_distributed_and_preserves_exact_keys(
    spark, monkeypatch, frequency, start, end, expected,
):
    monkeypatch.setattr(
        spark, "createDataFrame", lambda *a, **k: pytest.fail("historical keys reached the driver")
    )
    original_zone = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "Asia/Kolkata")
    try:
        cadence = benchmark.parse_frequency(frequency)
        frame = benchmark._period_frame(
            spark, benchmark.parse_origin(start, cadence), benchmark.parse_origin(end, cadence),
            cadence, "time",
        )
        assert frame.schema["time"].dataType.typeName() == (
            "timestamp_ntz" if frequency == "H" else "date"
        )
        assert [row["time"].isoformat() for row in frame.orderBy("time").collect()] == expected
    finally:
        spark.conf.set("spark.sql.session.timeZone", original_zone)


def test_large_hourly_history_can_be_planned_without_driver_enumeration(spark, monkeypatch):
    monkeypatch.setattr(
        spark, "createDataFrame", lambda *a, **k: pytest.fail("historical keys reached the driver")
    )
    cadence = benchmark.parse_frequency("H")
    start, end = dt.datetime(1, 1, 1), dt.datetime(9998, 1, 1)
    frame = benchmark._period_frame(spark, start, end, cadence, "time")
    assert [row["time"] for row in frame.limit(2).collect()] == [
        start, start + dt.timedelta(hours=1),
    ]


def test_freeze_composite_series_and_idempotent_identity(frozen, spark):
    path, project, recipe, sources = frozen
    record = benchmark.verify_benchmark(path)[1]
    assert record["source_series_count"] == record["cohort_series_count"] == 2
    assert record["excluded_new_series"] == 0
    keys = pq.read_table(path.parent / "source-keys").to_pylist()
    assert {tuple((r["shop"], r["sku"])) for r in keys} == {("a|b", 1), ("a", 2)}
    assert all(
        json.loads(r["__af_series_id"]) == {"shop": r["shop"], "sku": r["sku"]} for r in keys
    )
    assert _freeze(project, recipe, sources, spark) == path
    assert len(record["grids"]) == 3
    assert len({g["logical_sha256"] for g in record["grids"].values()}) == 3


def test_freeze_preserves_internal_and_trailing_gaps_with_original_missing_flags(tmp_path, spark):
    project, recipe, sources = _inputs(tmp_path)
    recipe.update(
        training_start="2024-01-01",
        validation_origins=["2024-01-08"],
        validation_horizon=2,
        final_train_cutoff="2024-01-10",
        final_test_horizon=2,
    )
    rows = [
        {"shop": "x", "sku": 1, "day": dt.date(2024, 1, day), "units": 0.0}
        for day in (1, 2, 3, 6, 9, 10, 11, 12)
    ]
    pq.write_table(pa.Table.from_pylist(rows), project / "data/observations.parquet")
    path = _freeze(project, recipe, sources, spark)
    record = benchmark.verify_benchmark(path)[1]
    prepared = pq.read_table(benchmark.artifact_path(path, record, "selection")).to_pylist()
    assert len(prepared) == 8
    assert all(row["units"] is None if row[benchmark.MISSING_COLUMN] else row["units"] == 0
               for row in prepared)
    assert {row["day"].day for row in prepared if row[benchmark.MISSING_COLUMN]} == {4, 5, 7, 8}
    assert record["counts"]["selection"] == {
        "required_observations": 8, "written_rows": 8, "originally_missing": 4, "zero_filled": 0,
    }


def test_freeze_rejects_duplicate_source_observations(tmp_path, spark):
    project, recipe, sources = _inputs(tmp_path)
    table = pq.read_table(project / "data")
    duplicate = table.slice(0, 1)
    pq.write_table(pa.concat_tables([table, duplicate]), project / "data/observations.parquet")

    with pytest.raises(benchmark.BenchmarkError, match="duplicate source observations"):
        _freeze(project, recipe, sources, spark)


def test_freeze_rejects_a_weekly_source_row_off_the_declared_phase(tmp_path, spark):
    project = tmp_path / "project"
    create_example(project, frequency="W")
    recipe = json.loads((project / "recipe.json").read_text())
    sources = {role: project / "data" for role in ("selection", "final_training", "validation_score", "final_score")}
    table = pq.read_table(project / "data")
    row = table.slice(0, 1).to_pylist()[0]
    row["day"] += dt.timedelta(days=1)
    pq.write_table(
        pa.concat_tables([table, pa.Table.from_pylist([row], schema=table.schema)]),
        project / "data/observations.parquet",
    )

    with pytest.raises(benchmark.BenchmarkError, match="off the declared W grid"):
        _freeze(project, recipe, sources, spark)


def test_freeze_conflicting_overlap_and_explicit_revision(tmp_path, spark):
    project, recipe, sources = _inputs(tmp_path)
    rows = pq.read_table(project / "data").to_pylist()
    revised = project / "revised"
    revised.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{**r, "units": r["units"] + 1} for r in rows]),
        revised / "part.parquet",
    )
    sources["final_training"] = revised
    with pytest.raises(benchmark.BenchmarkError, match="conflicting overlapping history"):
        _freeze(project, recipe, sources, spark)
    recipe["history_policy"] = "revised_snapshot"
    path = _freeze(project, recipe, sources, spark)
    assert benchmark.verify_benchmark(path)[1]["recipe"]["history_policy"] == "revised_snapshot"


def test_freeze_unknown_future_features(tmp_path, spark):
    project, recipe, sources = _inputs(tmp_path)
    contract = yaml.safe_load((project / "dataset.yaml").read_text())
    contract["dataset"]["panel"]["known_future"] = ["promotion"]
    (project / "dataset.yaml").write_text(yaml.safe_dump(contract))
    with pytest.raises(benchmark.BenchmarkError, match="availability-evidence"):
        _freeze(project, recipe, sources, spark)


def test_freeze_source_mutation(tmp_path, spark, monkeypatch):
    project, recipe, sources = _inputs(tmp_path)
    original = benchmark.directory_digest
    calls = 0

    def changing(path, **kwargs):
        nonlocal calls
        if path == project / "data":
            calls += 1
            if calls > 3:
                return "0" * 64
        return original(path, **kwargs)

    monkeypatch.setattr(benchmark, "directory_digest", changing)
    with pytest.raises(benchmark.BenchmarkError, match="changed before its snapshot"):
        _freeze(project, recipe, sources, spark)


def _forecast(path, grid, *, omit=False, replace_key=False):
    rows = pq.read_table(grid).to_pylist()
    if omit:
        rows = rows[1:]
    if replace_key:
        rows[0]["series_id"] = "wrong-series"
    rows = [{**row, "yhat": 2.0} for row in rows]
    path.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), path / "part.parquet")
    return Forecast(
        "seasonal_naive",
        str(path),
        "2024-01-21",
        7,
        2,
        {"emitted": len(rows), "declined_no_source": 14 - len(rows)},
    )


@pytest.mark.parametrize("defect", ["omit", "replace", "missing_actual", "wrong_origin"])
def test_fixed_grid_rejects_population_changes(frozen, tmp_path, spark, defect):
    path, _, _, _ = frozen
    fixed = benchmark.evaluation_contract(
        path, stage="validation", origin="2024-01-21", horizon=7, metric="rmse"
    )
    forecast = _forecast(
        tmp_path / "forecast",
        fixed["grid"]["path"],
        omit=defect == "omit",
        replace_key=defect == "replace",
    )
    panel = load_dataset_contract(path.parent / "dataset.yaml").panel(
        path.parent / "final_training"
    )
    if defect == "missing_actual":
        target = tmp_path / "actuals"
        target.mkdir()
        table = pq.read_table(panel.path)
        rows = [r for r in table.to_pylist() if r["day"] != dt.date(2024, 1, 22)]
        pq.write_table(pa.Table.from_pylist(rows), target / "part.parquet")
        panel = replace(panel, path=str(target))
        result = compare_run_forecasts(
            panel,
            {"naive": forecast},
            spark,
            missing_actual="no_data",
            metric="rmse",
            output_dir=tmp_path / "score",
            fixed_contract=fixed,
        )
        record = json.loads(result.read_text())
        assert record["ranking"] is None
        assert record["candidates"][0]["status"] == "not_scored"
        assert "actuals changed" in record["candidates"][0]["detail"]
    else:
        if defect == "wrong_origin":
            fixed = benchmark.evaluation_contract(
                path, stage="validation", origin="2024-01-28", horizon=7, metric="rmse"
            )
        with pytest.raises(
            EvaluationError, match="canonical eligible grid|wrong evaluation boundary"
        ):
            compare_run_forecasts(
                panel,
                {"naive": forecast},
                spark,
                missing_actual="no_data",
                metric="rmse",
                output_dir=tmp_path / "score",
                fixed_contract=fixed,
            )


def test_fixed_grid_accepts_the_explicit_month_end_alias(tmp_path, spark):
    from autoforecast.adapters.naive import seasonal_naive

    project = tmp_path / "project"
    create_example(project, frequency="M")
    contract = yaml.safe_load((project / "dataset.yaml").read_text())
    contract["dataset"]["panel"]["frequency"] = "ME"
    (project / "dataset.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
    recipe = json.loads((project / "recipe.json").read_text())
    sources = {role: project / "data" for role in ("selection", "final_training", "validation_score", "final_score")}
    benchmark_path = _freeze(project, recipe, sources, spark)
    origin = recipe["validation_origins"][0]
    fixed = benchmark.evaluation_contract(
        benchmark_path, stage="validation", origin=origin, horizon=7, metric="rmse"
    )
    panel = load_dataset_contract(benchmark_path.parent / "dataset.yaml").panel(
        benchmark_path.parent / "selection"
    )
    forecast = seasonal_naive(
        panel, origin, 7, spark=spark, out_dir=tmp_path / "month-end-forecast"
    )

    scoring_panel = load_dataset_contract(benchmark_path.parent / "dataset.yaml").panel(
        fixed["actuals"]["path"]
    )
    result = compare_run_forecasts(
        scoring_panel,
        {"seasonal_naive": forecast},
        spark,
        missing_actual="no_data",
        metric="rmse",
        output_dir=tmp_path / "score",
        fixed_contract=fixed,
    )

    record = json.loads(result.read_text())
    assert record["candidates"][0]["status"] == "scored"


@pytest.mark.parametrize("existing", ["active", "default", "context"])
def test_entrypoint_preserves_existing_spark(tmp_path, monkeypatch, existing):
    from scripts import run_autoforecast as runner

    class Session:
        _activeSession = object() if existing == "active" else None
        _instantiatedSession = object() if existing == "default" else None

        @classmethod
        def getActiveSession(cls):
            return cls._activeSession

    class Context:
        _active_spark_context = object() if existing == "context" else None

    monkeypatch.setattr(runner, "SparkSession", Session)
    monkeypatch.setattr(runner, "SparkContext", Context)
    environment = dict(os.environ)
    identities = Session._activeSession, Session._instantiatedSession, Context._active_spark_context
    with pytest.raises(RuntimeError, match="fresh Spark"):
        runner.run_request(
            tmp_path / "missing.yaml",
            data_path=tmp_path,
            data_snapshot_id="test",
            data_reference="local://test",
            output_root=tmp_path,
            epochs=1,
            timeout_seconds=10,
        )
    assert dict(os.environ) == environment
    assert identities == (
        Session._activeSession,
        Session._instantiatedSession,
        Context._active_spark_context,
    )


def test_m5_policy_needs_no_fixture(tmp_path):
    from autoforecast.benchmarks.policy import is_m5_daily

    path = tmp_path / "generic.yaml"
    path.write_text("schema_version: 1\ndataset: {}\n")
    assert not is_m5_daily(path)


def test_new_adapter_metadata_is_complete_without_native_imports(tmp_path):
    from autoforecast.adapter_contracts import descriptor
    from autoforecast.data_admission import get_model_input_contract
    from autoforecast.model_catalog import CatalogError, discover_candidates

    project, _, _ = _inputs(tmp_path)
    assert {c.id for c in discover_candidates(project / "models")} == {
        "seasonal_naive",
        "last_value",
    }
    assert get_model_input_contract("last_value").minimum_history(season_length=7, horizon=7) == 1
    assert descriptor("last_value").operations == ("fit_predict",)
    path = project / "models/last_value/model.yaml"
    value = yaml.safe_load(path.read_text())
    value["hyperparameters"] = {"ignored": 1}
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(CatalogError, match="no supported parameters"):
        discover_candidates(project / "models")


@pytest.mark.parametrize("changed", ["final_score", "source-keys", "grid", "contract", "inventory"])
def test_benchmark_identity_rejects_replacement_with_updated_checksums(frozen, tmp_path, changed):
    import hashlib
    import shutil

    path = tmp_path / "copy" / "benchmark.json"
    shutil.copytree(frozen[0].parent, path.parent)
    record = json.loads(path.read_text())
    if changed in {"final_score", "source-keys", "grid"}:
        item = next(iter(record["grids"].values())) if changed == "grid" else None
        directory = path.parent / (item["path"] if item else changed)
        table = pq.read_table(directory)
        rows = table.to_pylist()
        if changed == "final_score":
            rows[0]["units"] += 1000
        elif changed == "source-keys":
            rows[0]["shop"] = "replacement"
        else:
            rows[0]["series_id"] = "replacement"
        shutil.rmtree(directory)
        directory.mkdir()
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), directory / "part.parquet")
        physical = benchmark.directory_digest(directory, suffixes=(".parquet",))
        if item:
            item["sha256"] = physical
        else:
            record["artifacts"][changed] = physical
    elif changed == "contract":
        contract = path.parent / "dataset.yaml"
        contract.write_text(contract.read_text().replace("units", "replacement"))
        record["dataset_sha256"] = hashlib.sha256(contract.read_bytes()).hexdigest()
    else:
        record["grids"] = {}
    path.write_text(json.dumps(record))
    with pytest.raises(benchmark.BenchmarkError, match="changed"):
        benchmark.verify_benchmark(path)


def test_benchmark_parses_the_bytes_it_authenticated(frozen, tmp_path, monkeypatch):
    import hashlib
    import shutil

    path = tmp_path / "copy" / "benchmark.json"
    shutil.copytree(frozen[0].parent, path.parent)
    payload = path.read_bytes()
    original = benchmark.loads_exact_json

    def replace_after_authentication(authenticated):
        replacement = json.loads(payload)
        replacement["recipe"]["target_unit"] = "changed after hash"
        replacement["benchmark_id"] = benchmark.digest(benchmark._identity(replacement))
        path.write_text(json.dumps(replacement))
        return original(authenticated)

    monkeypatch.setattr(benchmark, "loads_exact_json", replace_after_authentication)
    _, result = benchmark.verify_benchmark(path, hashlib.sha256(payload).hexdigest())
    assert result["benchmark_id"] == json.loads(payload)["benchmark_id"]


def test_benchmark_identity_survives_parquet_layout_changes(frozen, tmp_path):
    import shutil

    path = tmp_path / "copy" / "benchmark.json"
    shutil.copytree(frozen[0].parent, path.parent)
    record = json.loads(path.read_text())
    directory = path.parent / "final_score"
    table = pq.read_table(directory)
    shutil.rmtree(directory)
    directory.mkdir()
    pq.write_table(table.slice(0, 4), directory / "first.parquet")
    pq.write_table(table.slice(4), directory / "second.parquet")
    record["artifacts"]["final_score"] = benchmark.directory_digest(
        directory, suffixes=(".parquet",)
    )
    path.write_text(json.dumps(record))
    assert benchmark.verify_benchmark(path)[1]["benchmark_id"] == record["benchmark_id"]


@pytest.mark.parametrize("blank", ["", " ", "\t\n\r", "\u00a0", "\u2003", "\u2028", "\x1c", "\u3000"])
def test_freeze_rejects_blank_string_source_keys_from_typed_parquet(tmp_path, spark, blank):
    project, recipe, sources = _inputs(tmp_path)
    rows = pq.read_table(project / "data").to_pylist()
    rows = [{**row, "shop": blank if row["sku"] == 1 else row["shop"]} for row in rows]
    pq.write_table(pa.Table.from_pylist(rows), project / "data/observations.parquet")
    with pytest.raises(ValueError, match="blank string key"):
        _freeze(project, recipe, sources, spark)


@pytest.mark.parametrize("name", ["__af_series_id", "__af_unused", "__AF_series_id", "__Af_count"])
def test_freeze_rejects_reserved_raw_columns_even_when_unmapped(tmp_path, spark, name):
    project, recipe, sources = _inputs(tmp_path)
    table = pq.read_table(project / "data")
    table = table.append_column(name, pa.array(["reserved"] * table.num_rows))
    pq.write_table(table, project / "data/observations.parquet")
    with pytest.raises(benchmark.BenchmarkError, match="__af_ are reserved"):
        _freeze(project, recipe, sources, spark)


@pytest.mark.parametrize("time_column", ["count", "last"])
def test_count_and_last_columns_freeze_and_run_a_baseline_with_integral_zero_keys(
    tmp_path, spark, time_column,
):
    from autoforecast.adapters.naive import seasonal_naive

    project, recipe, sources = _inputs(tmp_path)
    contract = yaml.safe_load((project / "dataset.yaml").read_text())
    contract["dataset"]["panel"]["time_col"] = time_column
    (project / "dataset.yaml").write_text(yaml.safe_dump(contract))
    rows = pq.read_table(project / "data").to_pylist()
    rows = [{**{key: value for key, value in row.items() if key != "day"}, time_column: row["day"],
             "sku": 0 if row["sku"] == 1 else row["sku"]} for row in rows]
    pq.write_table(pa.Table.from_pylist(rows), project / "data/observations.parquet")
    path = _freeze(project, recipe, sources, spark)
    record = benchmark.verify_benchmark(path)[1]
    assert record["cohort_series_count"] == 2 and record["excluded_new_series"] == 0
    assert 0 in pq.read_table(path.parent / "source-keys")["sku"].to_pylist()
    panel = load_dataset_contract(path.parent / "dataset.yaml").panel(path.parent / "selection")
    forecast = seasonal_naive(
        panel, recipe["validation_origins"][-1], 7, spark=spark, out_dir=tmp_path / "forecast",
    )
    predictions = pq.read_table(forecast.path)
    assert forecast.n_series == 2 and predictions.num_rows == 14
    assert sorted(predictions["yhat"].to_pylist()) == sorted([*range(3, 10), *range(4, 11)])


def test_last_time_column_preserves_missing_observation_flags_without_alias_collision(tmp_path, spark):
    project, recipe, sources = _inputs(tmp_path)
    contract = yaml.safe_load((project / "dataset.yaml").read_text())
    contract["dataset"]["panel"]["time_col"] = "last"
    (project / "dataset.yaml").write_text(yaml.safe_dump(contract))
    rows = pq.read_table(project / "data").to_pylist()
    rows = [{**{key: value for key, value in row.items() if key != "day"}, "last": row["day"]}
            for row in rows if not (row["sku"] == 1 and row["day"] == dt.date(2024, 1, 7))]
    pq.write_table(pa.Table.from_pylist(rows), project / "data/observations.parquet")
    path = _freeze(project, recipe, sources, spark)
    record = json.loads(path.read_text())
    rows = pq.read_table(benchmark.artifact_path(path, record, "selection")).to_pylist()
    missing = [row for row in rows if row[benchmark.MISSING_COLUMN]]
    assert len(missing) == 1
    assert missing[0]["sku"] == 1 and missing[0]["last"] == dt.date(2024, 1, 7)
    assert missing[0]["units"] is None
    assert record["counts"]["selection"]["originally_missing"] == 1
