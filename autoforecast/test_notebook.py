"""Notebook boundaries reuse frozen populations, stage slices and the shared scorer."""

import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pyspark.sql import SparkSession
from scripts.notebook_study_example import example_forecast

from autoforecast import benchmark, notebook
from autoforecast.frequency import canonical_origin
from autoforecast.notebook import load_study
from autoforecast.panel import Forecast
from autoforecast.project import experiment_project
from autoforecast.study_scoring import score_study
from autoforecast.test_study import inputs
from provenance.lineage import directory_digest


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]").appName("test_notebook_study")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false")
               .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module", params=[("daily", "zero"), ("daily", "mask"),
                                      ("hourly", "zero"), ("hourly", "mask")])
def frozen(request, tmp_path_factory, spark):
    # Population: A/B retained, one late series excluded, four cells per comparison.
    # The hourly client uses different source keys, column names and calendar dates.
    client, policy = request.param
    root = tmp_path_factory.mktemp(f"notebook-{client}-{policy}")
    contract, sources, recipe = inputs(root, actual_missing=policy)
    contract_record = yaml.safe_load(contract.read_text())
    rows = pq.read_table(sources["selection"]).to_pylist()
    panel = contract_record["dataset"]["panel"]
    panel.update(known_future=["promo"], exposure_col="exposure")
    recipe["feature_availability"] = {"promo": "known_at"}
    for row in rows:
        row.update(promo=1.0, exposure="exact", known_at=dt.datetime(2023, 1, 1))
    if client == "hourly":
        def point(value):
            return dt.datetime(2025, 3, 1) + dt.timedelta(hours=int(value[-2:]) - 1)
        rows = [{"account": row.pop("id"), "at": point(row.pop("date").isoformat()),
                 "amount": row.pop("y"), **row} for row in rows]
        panel.update(series_col="account", time_col="at", target_col="amount", frequency="H")
        recipe.update(source_keys=["account"], frequency="H")
        for key in ("training_start", "final_train_cutoff"):
            recipe[key] = point(recipe[key]).isoformat()
        recipe["validation_origins"] = [point(recipe["validation_origins"][0]).isoformat()]
        for entry in recipe["series_starts"]["values"]:
            entry["keys"] = {"account": entry["keys"]["id"]}
            entry["start"] = point(entry["start"]).isoformat()
    contract.write_text(yaml.safe_dump(contract_record))
    pq.write_table(pa.Table.from_pylist(rows), sources["selection"])
    manifest = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=root / ".autoforecast/benchmarks", spark=spark,
    )
    return root, manifest, recipe


def test_notebook_stage_slices_ranges_keys_and_borrowed_runtime(frozen, spark):
    root, manifest, recipe = frozen
    before = directory_digest(manifest.parent)
    with experiment_project(root), load_study(manifest.parent.name) as study:
        assert study.study_id == manifest.parent.name
        panel = study.panel
        for stage, count in (("training", 8), ("validation", 4), ("refit", 12), ("testing", 4)):
            rows = getattr(study, stage + "_data").collect()
            assert len(rows) == count
            low, high = getattr(study, stage + "_date_range")
            assert min(row[panel.time_col] for row in rows) == low
            assert max(row[panel.time_col] for row in rows) == high
        assert study.forecasting_date_range == study.testing_date_range
        assert canonical_origin(study.training_date_range[1], panel.frequency) == recipe["validation_origins"][0]
        assert canonical_origin(study.refit_date_range[1], panel.frequency) == recipe["final_train_cutoff"]
        mapping = study.series_mapping.collect()
        assert len(mapping) == 2
        key = recipe["source_keys"][0]
        assert {row[key] for row in mapping} == {"A", "B"}
        assert {row[panel.series_col] for row in mapping} == {
            json.dumps({key: label}, separators=(",", ":")) for label in ("A", "B")
        }
        starts = {row[key]: row[benchmark.START_COLUMN] for row in study.series_starts.collect()}
        for row in study.training_data.collect():
            assert row[panel.time_col] >= starts[row[key]]
        testing = study.testing_data.collect()
        zeros = [row for row in testing if row[panel.target_col] == 0]
        assert any(not row[benchmark.MISSING_COLUMN] for row in zeros)
        assert any(row[benchmark.MISSING_COLUMN] for row in zeros)
        assert sum(row[benchmark.MISSING_COLUMN] for row in testing) == 2
        assert study.policies["actual_missing"] == recipe["actual_missing"]
        assert study.scoring_references is None
        altered = study.recipe
        altered["metric"] = "mae"
        assert study.metric_definition["metric"] == "rmse"
        with pytest.raises(dataclasses.FrozenInstanceError):
            study._sha256 = "0" * 64
    with pytest.raises(ValueError, match="closed"):
        _ = study.training_data
    study.close()
    assert spark.range(1).count() == 1
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as again:
        assert again.recipe == json.loads(manifest.read_text())["recipe"]
    assert directory_digest(manifest.parent) == before


def test_notebook_scores_match_direct_scorer_and_retain_run_identities(frozen, spark, tmp_path):
    _, manifest, recipe = frozen
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        validation = example_forecast(study, spark, tmp_path / "val", "validation")
        testing = example_forecast(study, spark, tmp_path / "test", "test")
        results = [study.run_validation(validation_forecasts=validation, testing_forecasts=testing,
                                        model_id="external-model") for _ in range(2)]
        assert results[0]["run_id"] != results[1]["run_id"]
        for stage, forecast in (("validation", validation), ("test", testing)):
            result = results[0][stage]
            fixed = benchmark.evaluation_contract(
                manifest, stage=stage, origin=forecast.origin, horizon=forecast.horizon, metric="rmse",
            )
            direct = score_study(dataclasses.replace(study.panel, path=fixed["actuals"]["path"]),
                                 forecast, spark, contract=fixed, output_dir=tmp_path / f"{stage}-direct")
            expected = {k: v for k, v in direct.items() if k != "series_scores"}
            assert {k: v for k, v in result["score"].items() if k != "series_scores"} == expected
            assert result["score"]["coverage"] == {
                "expected": 4, "originally_missing": 2,
                "zero_filled": 2 if recipe["actual_missing"] == "zero" else 0,
                "excluded": 2 if recipe["actual_missing"] == "mask" else 0,
                "scored": 2 if recipe["actual_missing"] == "mask" else 4,
            }
            for name in ("comparison", "forecast"):
                assert hashlib.sha256(pathlib.Path(result[name]).read_bytes()).hexdigest() == result[name + "_sha256"]
            retained = Forecast.from_record(json.loads(pathlib.Path(result["forecast"]).read_text()))
            assert directory_digest(retained.path, suffixes=(".parquet",)) == result["forecast_artifact_sha256"]
        assert results[0]["study_id"] == study.study_id
        assert results[0]["training_provenance"] == "unverified_notebook"
        assert not results[0]["managed_campaign"]
        assert json.loads(pathlib.Path(results[0]["result_path"]).read_text()) == results[0]


def test_notebook_unknown_access_and_changed_artifacts_are_distinct(frozen, spark, tmp_path, monkeypatch):
    _, manifest, _ = frozen
    with pytest.raises(FileNotFoundError, match="Unknown study_id"):
        load_study("0" * 64, registry_root=tmp_path, spark=spark)
    with pytest.raises(benchmark.BenchmarkError, match="SHA-256"):
        load_study("../escape", registry_root=tmp_path, spark=spark)
    (tmp_path / ("0" * 64)).mkdir()
    with pytest.raises(benchmark.BenchmarkError, match="cannot read benchmark manifest"):
        load_study("0" * 64, registry_root=tmp_path, spark=spark)
    original = pathlib.Path.read_bytes
    def denied(path):
        if path == manifest:
            raise PermissionError("test access denied")
        return original(path)
    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "read_bytes", denied)
        with pytest.raises(benchmark.BenchmarkError, match="access denied"):
            load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        frame = study.training_data
        record = json.loads(manifest.read_text())
        part = next(benchmark.artifact_path(manifest, record, "selection").rglob("*.parquet"))
        payload = part.read_bytes()
        try:
            part.write_bytes(b"changed")
            # Lazy actions consume the authenticated private copy, not the replaced source.
            assert frame.count() == 8
            with pytest.raises(benchmark.BenchmarkError, match="artifact changed"):
                _ = study.training_data
        finally:
            part.write_bytes(payload)
        raw = manifest.read_bytes()
        try:
            manifest.write_bytes(raw + b"\n")
            with pytest.raises(benchmark.BenchmarkError, match="manifest changed"):
                _ = study.recipe
        finally:
            manifest.write_bytes(raw)


@pytest.mark.parametrize("change", ["origin", "horizon", "frequency", "missing", "extra", "duplicate", "series", "estimand"])
def test_notebook_rejects_incorrect_forecast_contract(frozen, spark, tmp_path, change):
    _, manifest, _ = frozen
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        forecast = example_forecast(study, spark, tmp_path / "predictions", "validation")
        testing = example_forecast(study, spark, tmp_path / "testing", "test")
        if change in {"origin", "horizon", "frequency", "estimand"}:
            name, value = {"origin": ("origin", study.recipe["final_train_cutoff"]),
                           "horizon": ("horizon", 1), "frequency": ("frequency", "W"),
                           "estimand": ("target_estimand", "unconstrained_demand")}[change]
            forecast = dataclasses.replace(forecast, **{name: value})
        else:
            rows = pq.read_table(forecast.path).to_pylist()
            if change == "missing":
                rows.pop()
            elif change == "extra":
                rows.append({**rows[0], "series_id": "unexpected"})
            elif change == "duplicate":
                rows[1] = rows[0]
            else:
                rows[0]["series_id"] = "unexpected"
            shutil.rmtree(forecast.path)
            pathlib.Path(forecast.path).mkdir()
            pq.write_table(pa.Table.from_pylist(rows), pathlib.Path(forecast.path) / "part.parquet")
        runs = manifest.parent.parent.parent / "notebook-runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        with pytest.raises(ValueError):
            study.run_validation(validation_forecasts=forecast, testing_forecasts=testing, model_id="bad")
        assert set(runs.iterdir()) == before


def test_notebook_rejects_multiple_validation_origins(tmp_path, spark):
    contract, sources, recipe = inputs(tmp_path)
    recipe["validation_origins"] = ["2024-01-03", "2024-01-05"]
    manifest = benchmark.freeze_benchmark(contract_path=contract, sources=sources, recipe=recipe,
                                          output_root=tmp_path / "registry", spark=spark)
    with pytest.raises(benchmark.BenchmarkError, match="exactly one validation origin"):
        load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)


@pytest.mark.parametrize("policy,scale", [("zero", None), ("mask", 4.0)])
def test_notebook_keeps_undefined_wrmsse_and_frozen_weights(tmp_path, spark, policy, scale):
    from autoforecast.test_study import scoring_references
    contract, sources, recipe = inputs(tmp_path, actual_missing=policy)
    recipe["metric"] = "wrmsse"
    manifest = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe, output_root=tmp_path / "registry",
        spark=spark, scoring_references=scoring_references(tmp_path, scale_b=scale),
    )
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        assert study.scoring_references.count() == 4
        result = study.run_validation(
            validation_forecasts=example_forecast(study, spark, tmp_path / "validation", "validation"),
            testing_forecasts=example_forecast(study, spark, tmp_path / "testing", "test"), model_id="weights",
        )
        assert result["test"]["score"]["wrmsse"] is None
        assert result["test"]["status"] == "unavailable"
        per_series = pq.read_table(result["test"]["score"]["series_scores"]["path"]).to_pylist()
        assert sorted(row["normalized_weight"] for row in per_series) == pytest.approx([9/23, 14/23])


def test_notebook_requires_supplied_or_active_session(frozen, monkeypatch):
    _, manifest, _ = frozen
    monkeypatch.setattr(SparkSession, "getActiveSession", lambda: None)
    with pytest.raises(ValueError, match="Spark session"):
        load_study(manifest.parent.name, registry_root=manifest.parent.parent)


def test_notebook_reauthenticates_between_scores(frozen, spark, tmp_path, monkeypatch):
    _, manifest, _ = frozen
    original = notebook.compare_run_forecasts
    raw = manifest.read_bytes()
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        manifest.write_bytes(raw + b"\n")
        return result
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        validation = example_forecast(study, spark, tmp_path / "validation", "validation")
        testing = example_forecast(study, spark, tmp_path / "testing", "test")
        monkeypatch.setattr(notebook, "compare_run_forecasts", changed)
        runs = manifest.parent.parent.parent / "notebook-runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        try:
            with pytest.raises(benchmark.BenchmarkError, match="changed"):
                study.run_validation(validation_forecasts=validation, testing_forecasts=testing, model_id="change")
            assert set(runs.iterdir()) == before
        finally:
            manifest.write_bytes(raw)


def test_notebook_authenticates_dataset_and_private_snapshot(frozen, spark):
    _, manifest, _ = frozen
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        dataset = manifest.parent / "dataset.yaml"
        payload = dataset.read_bytes()
        try:
            dataset.write_bytes(payload + b"\n")
            with pytest.raises(benchmark.BenchmarkError, match="dataset contract changed"):
                _ = study.panel
        finally:
            dataset.write_bytes(payload)
        _ = study.training_data
        part = next((pathlib.Path(study._snapshots.name) / "selection").rglob("*.parquet"))
        part.write_bytes(b"changed")
        with pytest.raises(benchmark.BenchmarkError, match="snapshot changed"):
            _ = study.training_data


def test_notebook_snapshot_rejects_changes_during_copy(frozen, spark, monkeypatch):
    _, manifest, _ = frozen
    record = json.loads(manifest.read_text())
    part = next(benchmark.artifact_path(manifest, record, "selection").rglob("*.parquet"))
    payload, copyfile = part.read_bytes(), shutil.copyfile
    observed = []
    def changing_copy(source, destination, **kwargs):
        if source == part:
            observed.append(source)
            part.write_bytes(b"changed")
        return copyfile(source, destination, **kwargs)
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        try:
            monkeypatch.setattr(shutil, "copyfile", changing_copy)
            with pytest.raises(ValueError, match="changed"):
                _ = study.training_data
            assert observed
        finally:
            part.write_bytes(payload)


def test_notebook_rejects_retained_forecast_changes_after_scoring(frozen, spark, tmp_path, monkeypatch):
    _, manifest, _ = frozen
    compare = notebook.compare_run_forecasts
    def changed(*args, **kwargs):
        result = compare(*args, **kwargs)
        output = pathlib.Path(kwargs["output_dir"])
        if output.parent.name == "test":
            part = next((output.parent.parent / "validation/forecasts").rglob("*.parquet"))
            part.write_bytes(b"changed")
        return result
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        validation = example_forecast(study, spark, tmp_path / "validation", "validation")
        testing = example_forecast(study, spark, tmp_path / "testing", "test")
        monkeypatch.setattr(notebook, "compare_run_forecasts", changed)
        with pytest.raises(benchmark.BenchmarkError, match="retained notebook artifacts changed"):
            study.run_validation(validation_forecasts=validation, testing_forecasts=testing, model_id="change")
