"""Study interface boundaries reuse frozen populations, stage slices and the shared scorer."""

import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.readwriter import DataFrameReader
from scripts.study_example import example_forecast

from autoforecast import benchmark
from autoforecast import study as study_api
from autoforecast.frequency import canonical_origin, parse_origin
from autoforecast.panel import Forecast
from autoforecast.project import experiment_project
from autoforecast.study import load_study
from autoforecast.study_scoring import score_study
from autoforecast.test_study import inputs
from provenance.lineage import directory_digest


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]").appName("test_study_interface")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false")
               .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module", params=[("daily", "mask"), ("hourly", "mask")])
def frozen(request, tmp_path_factory, spark):
    # Population: A/B retained, one late series excluded, four cells per comparison.
    # The hourly client uses different source keys, column names and calendar dates.
    client, policy = request.param
    root = tmp_path_factory.mktemp(f"study-{client}-{policy}")
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
    if client == "hourly":
        # Distinct scoring and later refit values expose accidental role substitution.
        validation = root / "validation-actuals.parquet"
        pq.write_table(pa.Table.from_pylist([
            {**row, "amount": None if row["amount"] is None else row["amount"] + 5}
            for row in rows
        ]), validation)
        refit = root / "refit-history.parquet"
        cutoff = parse_origin(recipe["validation_origins"][0], "H")
        pq.write_table(pa.Table.from_pylist([
            {**row, "amount": (row["amount"] + 7 if row["amount"] is not None
                               and row["at"] > cutoff else row["amount"])}
            for row in rows
        ]), refit)
        sources.update(validation_score=validation, final_training=refit)
    manifest = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=root / ".autoforecast/benchmarks", spark=spark,
    )
    return root, manifest, recipe


def test_study_interface_stage_slices_ranges_keys_and_borrowed_runtime(frozen, spark):
    root, manifest, recipe = frozen
    before = directory_digest(manifest.parent)
    with experiment_project(root), load_study(manifest.parent.name) as study:
        assert study.study_id == manifest.parent.name
        panel = study.panel
        record = json.loads(manifest.read_text())
        roles = dict(training="selection", validation="validation_score",
                     refit="final_training", testing="final_score")
        for stage, count in (("training", 8), ("validation", 4), ("refit", 12), ("testing", 4)):
            rows = getattr(study, stage + "_data").collect()
            assert len(rows) == count
            low, high = getattr(study, stage + "_date_range")
            assert min(row[panel.time_col] for row in rows) == low
            assert max(row[panel.time_col] for row in rows) == high
            expected = pq.read_table(benchmark.artifact_path(manifest, record, roles[stage])).to_pylist()
            expected = [row for row in expected if low <= row[panel.time_col] <= high]
            def row_key(row):
                return row[panel.series_col], row[panel.time_col]
            assert sorted([row.asDict() for row in rows], key=row_key) == sorted(expected, key=row_key)
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
        assert starts == {entry["keys"][key]: parse_origin(entry["start"], panel.frequency)
                          for entry in recipe["series_starts"]["values"]}
        for row in study.training_data.collect():
            assert row[panel.time_col] >= starts[row[key]]
        testing = study.testing_data.collect()
        zeros = [row for row in testing if row[panel.target_col] == 0]
        assert any(not row[benchmark.MISSING_COLUMN] for row in zeros)
        assert all(not row[benchmark.MISSING_COLUMN] for row in zeros)
        assert any(row[panel.target_col] is None and row[benchmark.MISSING_COLUMN]
                   for row in testing)
        assert sum(row[benchmark.MISSING_COLUMN] for row in testing) == 2
        assert study.policies["actual_missing"] == recipe["actual_missing"]
        assert study.scoring_references is None
        altered = study.recipe
        altered["metric"] = "mae"
        altered["series_starts"]["values"][0]["start"] = "unexpected"
        assert study.metric_definition["metric"] == "rmse"
        assert study.recipe["series_starts"] == recipe["series_starts"]
        study.policies["actual_missing"] = "unexpected"
        study.metric_definition["metric"] = "unexpected"
        assert study.policies == record["policies"]
        assert study.metric_definition == record["metric_definition"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            study._sha256 = "0" * 64
    with pytest.raises(ValueError, match="closed"):
        _ = study.training_data
    study.close()
    assert spark.range(1).count() == 1
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as again:
        assert again.recipe == json.loads(manifest.read_text())["recipe"]
    assert directory_digest(manifest.parent) == before


def test_study_interface_scores_match_direct_scorer_and_retain_run_identities(frozen, spark, tmp_path):
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
                "zero_filled": 0,
                "excluded": 2 if recipe["actual_missing"] == "mask" else 0,
                "scored": 2 if recipe["actual_missing"] == "mask" else 4,
            }
            for name in ("scoring", "forecast"):
                assert hashlib.sha256(pathlib.Path(result[name]).read_bytes()).hexdigest() == result[name + "_sha256"]
            retained = Forecast.from_record(json.loads(pathlib.Path(result["forecast"]).read_text()))
            assert directory_digest(retained.path, suffixes=(".parquet",)) == result["forecast_artifact_sha256"]
        assert results[0]["schema"] == "autoforecast.study-check.v1"
        assert pathlib.Path(results[0]["result_path"]).parent.parent == (
            manifest.parent.parent.parent / "study-runs"
        )
        assert results[0]["study_id"] == study.study_id
        assert results[0]["model_id"] == "external-model"
        assert results[0]["verified_input_artifacts"] == json.loads(manifest.read_text())["artifacts"]
        assert results[0]["study"] == {"path": str(manifest),
                                       "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        assert results[0]["training_provenance"] == "unverified_external"
        assert not results[0]["managed_campaign"]
        assert json.loads(pathlib.Path(results[0]["result_path"]).read_text()) == results[0]


def test_study_interface_unknown_access_and_changed_artifacts_are_distinct(frozen, spark, tmp_path, monkeypatch):
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
def test_study_interface_rejects_incorrect_forecast_contract(frozen, spark, tmp_path, change):
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
        runs = manifest.parent.parent.parent / "study-runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        with pytest.raises(ValueError):
            study.run_validation(validation_forecasts=forecast, testing_forecasts=testing, model_id="bad")
        assert set(runs.iterdir()) == before


def test_study_interface_rejects_multiple_validation_origins(tmp_path, spark):
    contract, sources, recipe = inputs(tmp_path)
    recipe["validation_origins"] = ["2024-01-03", "2024-01-05"]
    manifest = benchmark.freeze_benchmark(contract_path=contract, sources=sources, recipe=recipe,
                                          output_root=tmp_path / "registry", spark=spark)
    with pytest.raises(benchmark.BenchmarkError, match="exactly one validation origin"):
        load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)


@pytest.mark.parametrize("policy,scale", [("mask", None), ("mask", 4.0)])
def test_study_interface_keeps_undefined_wrmsse_and_frozen_weights(tmp_path, spark, policy, scale):
    from autoforecast.test_study import scoring_references
    contract, sources, recipe = inputs(tmp_path, actual_missing=policy)
    recipe["metric"] = "wrmsse"
    manifest = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe, output_root=tmp_path / "registry",
        spark=spark, scoring_references=scoring_references(tmp_path, scale_b=scale),
    )
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        assert study.scoring_references.count() == 4
        from autoforecast.score_spark import ScoreError
        # Missing all B test actuals is now an error, regardless of the scale.
        with pytest.raises(ScoreError, match="every retained series"):
            study.run_validation(
                validation_forecasts=example_forecast(study, spark, tmp_path / "validation", "validation"),
                testing_forecasts=example_forecast(study, spark, tmp_path / "testing", "test"),
                model_id="weights",
            )


def test_study_interface_requires_supplied_or_active_session(frozen, monkeypatch):
    _, manifest, _ = frozen
    monkeypatch.setattr(SparkSession, "getActiveSession", lambda: None)
    with pytest.raises(ValueError, match="Spark session"):
        load_study(manifest.parent.name, registry_root=manifest.parent.parent)


def test_study_interface_supplied_session_takes_precedence(frozen, spark, monkeypatch):
    _, manifest, _ = frozen
    def unexpected_lookup():
        raise AssertionError("supplied session must take precedence")
    with monkeypatch.context() as patch:
        patch.setattr(SparkSession, "getActiveSession", unexpected_lookup)
        study = load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)
        assert study._spark is spark
    with study:
        assert study.training_data.count() == 8


def test_study_interface_rejects_identity_changed_while_loading(frozen, spark, monkeypatch):
    _, manifest, _ = frozen
    verified_path, record = benchmark.verify_benchmark(manifest)
    def changed(path, expected_sha256):
        assert path == manifest
        assert expected_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
        return verified_path, {**record, "study_id": "b" * 64}
    monkeypatch.setattr(study_api, "verify_benchmark", changed)
    with pytest.raises(benchmark.BenchmarkError, match="identity changed while loading"):
        load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)


def test_study_interface_rejects_invalid_model_names_and_swapped_valid_stages(frozen, spark, tmp_path):
    _, manifest, _ = frozen
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        validation = example_forecast(study, spark, tmp_path / "validation", "validation").validate()
        testing = example_forecast(study, spark, tmp_path / "testing", "test").validate()
        for model_id in (None, "", "  "):
            with pytest.raises(ValueError, match="^model_id must be a non-blank string"):
                study.run_validation(validation_forecasts=None, testing_forecasts=None, model_id=model_id)
        with pytest.raises(ValueError, match="wrong evaluation boundary"):
            study.run_validation(validation_forecasts=testing, testing_forecasts=validation, model_id="swap")


@pytest.mark.parametrize("changed_stage", ["validation", "test"])
def test_study_interface_reauthenticates_between_scores(frozen, spark, tmp_path, monkeypatch, changed_stage):
    _, manifest, _ = frozen
    original = study_api.score_study
    raw = manifest.read_bytes()
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["contract"]["boundary"].split("/")[0] == changed_stage:
            manifest.write_bytes(raw + b"\n")
        return result
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        validation = example_forecast(study, spark, tmp_path / "validation", "validation")
        testing = example_forecast(study, spark, tmp_path / "testing", "test")
        monkeypatch.setattr(study_api, "score_study", changed)
        runs = manifest.parent.parent.parent / "study-runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        try:
            with pytest.raises(benchmark.BenchmarkError, match="changed"):
                study.run_validation(validation_forecasts=validation, testing_forecasts=testing, model_id="change")
            assert set(runs.iterdir()) == before
        finally:
            manifest.write_bytes(raw)


def test_study_interface_authenticates_dataset_and_private_snapshot(frozen, spark):
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


def test_study_interface_snapshot_rejects_changes_during_copy(frozen, spark, monkeypatch):
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


@pytest.mark.parametrize("artifact", ["forecasts", "forecast.json", "scoring", "series-scores"])
def test_study_interface_rejects_retained_forecast_changes_after_scoring(frozen, spark, tmp_path, monkeypatch, artifact):
    _, manifest, _ = frozen
    compare = study_api.score_study
    def changed(*args, **kwargs):
        result = compare(*args, **kwargs)
        output = pathlib.Path(kwargs["output_dir"])
        if output.parent.name == "test":
            validation = output.parent.parent / "validation"
            if artifact == "forecasts":
                target = next((validation / "forecasts").rglob("*.parquet"))
            elif artifact == "forecast.json":
                target = validation / "forecast.json"
            elif artifact == "scoring":
                target = validation / "score.json"
            else:
                target = next((validation / "scores/series-scores").rglob("*.parquet"))
            target.write_bytes(target.read_bytes() + b"\n")
        return result
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        validation = example_forecast(study, spark, tmp_path / "validation", "validation")
        testing = example_forecast(study, spark, tmp_path / "testing", "test")
        monkeypatch.setattr(study_api, "score_study", changed)
        runs = manifest.parent.parent.parent / "study-runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        with pytest.raises(benchmark.BenchmarkError, match="retained study .* changed"):
            study.run_validation(validation_forecasts=validation, testing_forecasts=testing, model_id="change")
        assert set(runs.iterdir()) == before


def test_study_interface_preserves_registry_path_integrity(frozen, spark, tmp_path):
    _, manifest, _ = frozen
    link = tmp_path / "registry-link"
    link.symlink_to(manifest.parent.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_study(manifest.parent.name, registry_root=link, spark=spark)
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / manifest.parent.name).symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(benchmark.BenchmarkError, match="cannot read benchmark manifest"):
        load_study(manifest.parent.name, registry_root=registry, spark=spark)
    with pytest.raises(benchmark.BenchmarkError, match="cannot read benchmark manifest"):
        load_study(manifest.parent.name, registry_root=tmp_path / "missing-registry", spark=spark)


def test_study_interface_failed_context_entry_releases_snapshots(frozen, spark):
    _, manifest, _ = frozen
    study = load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)
    _ = study.training_data
    private = pathlib.Path(study._snapshots.name)
    original = manifest.read_bytes()
    try:
        manifest.write_bytes(original + b"\n")
        with pytest.raises(benchmark.BenchmarkError, match="manifest changed"), study:
            pytest.fail("changed study entered the context")
        assert not private.exists()
        assert spark.range(1).count() == 1
    finally:
        manifest.write_bytes(original)
        study.close()


def test_example_refuses_unknown_stage(frozen, spark, tmp_path):
    _, manifest, _ = frozen
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        with pytest.raises(ValueError, match="stage"):
            example_forecast(study, spark, tmp_path / "typo", "validaton")
        assert not (tmp_path / "typo").exists()


def test_closed_study_cannot_reopen_or_remove_a_reused_directory(frozen, spark):
    _, manifest, _ = frozen
    study = load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)
    private = pathlib.Path(study._snapshots.name)
    study.close()
    private.mkdir()
    marker = private / "later-owner.txt"
    marker.write_text("owned by a later operation")
    try:
        with pytest.raises(ValueError, match="closed"):
            _ = study.recipe
        study.close()
        assert marker.read_text() == "owned by a later operation"
    finally:
        marker.unlink(missing_ok=True)
        if private.exists():
            private.rmdir()


@pytest.mark.parametrize("role,property_name", [
    ("selection", "training_data"), ("source-keys", "series_starts"),
])
def test_study_interface_authenticates_private_bytes_after_spark_read(
    frozen, spark, monkeypatch, role, property_name,
):
    _, manifest, _ = frozen
    original = DataFrameReader.parquet
    observed = []
    def changed(reader, *paths, **kwargs):
        frame = original(reader, *paths, **kwargs)
        path = pathlib.Path(paths[0])
        if path.name == role:
            observed.append(path)
            next(path.rglob("*.parquet")).write_bytes(b"changed during reader callback")
        return frame
    monkeypatch.setattr(DataFrameReader, "parquet", changed)
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        with pytest.raises(benchmark.BenchmarkError, match="snapshot changed"):
            getattr(study, property_name)
    assert observed


def test_study_interface_captures_both_forecasts_before_either_score(frozen, spark, tmp_path, monkeypatch):
    _, manifest, recipe = frozen
    score = study_api.score_study
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        validation = example_forecast(study, spark, tmp_path / "validation", "validation")
        testing = example_forecast(study, spark, tmp_path / "testing", "test")
        expected = directory_digest(testing.path, suffixes=(".parquet",))
        observed = []
        def changed(*args, **kwargs):
            result = score(*args, **kwargs)
            if kwargs["contract"]["boundary"].startswith("validation/"):
                for part in pathlib.Path(testing.path).glob("*.parquet"):
                    table = pq.read_table(part)
                    column = table.schema.get_field_index("yhat")
                    pq.write_table(table.set_column(column, "yhat", pa.array([100.0] * len(table))), part)
                observed.append(testing.path)
            return result
        monkeypatch.setattr(study_api, "score_study", changed)
        result = study.run_validation(validation_forecasts=validation, testing_forecasts=testing, model_id="capture")
        assert observed
        assert directory_digest(testing.path, suffixes=(".parquet",)) != expected
        assert result["test"]["forecast_artifact_sha256"] == expected
        expected_rmse = (53 / 2) ** 0.5
        assert result["test"]["score"]["rmse"] == pytest.approx(expected_rmse)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("default_only", [False, True])
def test_example_main_preserves_a_borrowed_session(spark, tmp_path, monkeypatch, fail, default_only):
    from scripts import study_example as example
    seen = []
    def exercise(project, session):
        seen.append((project, session))
        if fail:
            raise RuntimeError("example failure")
        return {}
    if default_only:
        monkeypatch.setattr(SparkSession, "getActiveSession", lambda: None)
    monkeypatch.setattr(example, "run_example", exercise)
    monkeypatch.setattr(sys, "argv", ["study_example", str(tmp_path / "project")])
    if fail:
        with pytest.raises(RuntimeError, match="example failure"):
            example.main()
    else:
        assert example.main() == 0
    assert seen == [((tmp_path / "project").resolve(), spark)]
    assert spark.range(1).count() == 1


@pytest.mark.parametrize("fail", [False, True])
def test_example_main_preserves_a_bare_spark_context(tmp_path, fail):
    # A separate process supplies a real context with no pre-existing SQL session.
    process = subprocess.run([sys.executable, "-c", """
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from scripts import study_example as example

context = SparkContext(conf=SparkConf().setMaster('local[1]').setAppName('caller-context')
                       .set('spark.ui.enabled', 'false'))
assert SparkSession.getActiveSession() is None
assert SparkSession._instantiatedSession is None
fail = sys.argv.pop() == 'True'
seen = []
def run(project, session):
    assert session.sparkContext is context
    seen.append(project)
    if fail:
        raise RuntimeError('example failure')
    return {}
example.run_example = run
try:
    try:
        example.main()
    except RuntimeError as error:
        assert fail and str(error) == 'example failure'
    else:
        assert not fail
    assert len(seen) == 1
    assert context.parallelize([1]).count() == 1
finally:
    context.stop()
""", str(tmp_path / "project"), str(fail)], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("fail", [False, True])
def test_example_main_stops_its_owned_session(tmp_path, fail):
    # A separate process starts with no caller runtime and checks the real owned-session boundary.
    process = subprocess.run([sys.executable, "-c", """
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from scripts import study_example as example

assert SparkContext._active_spark_context is None
assert SparkSession.getActiveSession() is None
assert SparkSession._instantiatedSession is None
fail = sys.argv.pop() == 'True'
seen = []
def run(project, session):
    assert session.sparkContext is SparkContext._active_spark_context
    seen.append(session)
    if fail:
        raise RuntimeError('example failure')
    return {}
example.run_example = run
try:
    try:
        result = example.main()
    except RuntimeError as error:
        assert fail and str(error) == 'example failure'
    else:
        assert not fail and result == 0
    assert len(seen) == 1
    assert seen[0].sparkContext._jsc is None
    assert SparkContext._active_spark_context is None
finally:
    context = SparkContext._active_spark_context
    if context is not None:
        context.stop()
""", str(tmp_path / "project"), str(fail)], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
