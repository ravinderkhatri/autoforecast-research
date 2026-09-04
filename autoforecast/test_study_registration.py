"""Registration tests use two declared synthetic series and no external source data."""

import datetime as dt
import json
import os
import pathlib
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pyspark.sql import functions as F

from autoforecast.study_registration import StudyRegistry, register_study
from autoforecast.test_study_tuning import config as model_profile
from autoforecast.test_study_tuning import implementation, predict
from autoforecast.test_study_tuning import spark as _spark


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def configuration(root, *, actual_missing="mask"):
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([
        {"item": sid, "day": dt.date(2024, 1, day),
         "value": None if day in (4, 6, 8) and sid == "B" else float(day)}
        for sid in ("A", "B") for day in range(1 if sid == "A" else 2, 10)
    ]), root / "source.parquet")
    value = {
        "schema_version": 1,
        "dataset": {"schema_version": 1, "dataset": {"id": "synthetic", "panel": {
            "series_col": "item", "time_col": "day", "target_col": "value",
            "frequency": "D", "season_length": 1, "target_estimand": "observed_nonnegative",
        }}},
        "recipe": {
            "schema_version": 2, "source_keys": ["item"], "training_start": "2024-01-01",
            "validation_origins": ["2024-01-05"], "validation_horizon": 2,
            "final_train_cutoff": "2024-01-07", "final_test_horizon": 2,
            "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
            "feature_availability": {}, "metric": "mae", "history_missing": "zero",
            "actual_missing": actual_missing,
            "tuning_windows": [{"origin": "2024-01-03", "horizon": 2}],
        },
        "source": {"kind": "parquet", "path": "source.parquet"},
        "storage": {"root": "registry"},
    }
    path = root / "study.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.mark.parametrize("policy", ["mask", "zero"])
def test_registration_retries_and_raw_actuals_remain_distinct_from_fit(spark, tmp_path, policy):
    path = configuration(tmp_path, actual_missing=policy)
    if policy == "zero":
        source = tmp_path / "source.parquet"
        table = pq.read_table(source)
        late = pa.Table.from_pylist([
            {"item": "late", "day": dt.date(2024, 1, 9), "value": 1.0},
        ], schema=table.schema)
        pq.write_table(pa.concat_tables([table, late]), source)
        settings = yaml.safe_load(path.read_text())
        settings["recipe"]["new_series_policy"] = "exclude_after_first_origin"
        path.write_text(yaml.safe_dump(settings))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    assert register_study(path, registry=registry) == study_id
    definition, sha = registry.definition(study_id)
    assert len(sha) == 64 and definition["admission"]["retained_series"] == 2
    assert definition["source"]["starting_series"] == (3 if policy == "zero" else 2)
    assert definition["source"]["eligible_series"] == 2
    assert definition["admission"]["excluded_new_keys"] == (
        [{"item": "late"}] if policy == "zero" else []
    )
    (tmp_path / "source.parquet").unlink()
    with registry.load_study(study_id) as study:
        assert study.training_data.filter("value is null").count() == 0
        assert study.validation_data.filter("value is null").count() == 1
        assert study.refit_data.filter("value is null").count() == 0
        assert study.training_data.count() == 9  # B starts on day 2, never day 1.
        assert study.policies["history_missing"] == "zero"
    baseline = registry.root / "registrations" / (study_id + ".json")
    record = json.loads(baseline.read_text())
    record["definition"]["admission"]["retained_series"] = 17
    baseline.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="baseline"):
        registry.load_study(study_id)
    from autoforecast import run_study

    with pytest.raises(ValueError, match="baseline") as caught:
        run_study(study_id, lambda _: pytest.fail("hash failure must precede fit"),
                  registry=registry, model_config=model_profile(tmp_path),
                  implementation=implementation())
    audit = [json.loads(line) for line in caught.value.audit_path.read_text().splitlines()]
    assert any(e["stage"] == "load_and_verify" and e["status"] == "FAIL" for e in audit)


def test_bad_config_and_unknown_id_do_not_provision(spark, tmp_path):
    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    with pytest.raises(FileNotFoundError, match="Unknown"):
        registry.load_study("missing")
    assert not registry.root.exists()
    with pytest.raises(ValueError, match="POSIX"):
        StudyRegistry("s3://unsupported/root", spark=spark)
    path.write_text(path.read_text() + "schema_version: 1\n")
    with pytest.raises(ValueError, match="duplicate"):
        register_study(path, registry=registry)
    assert not registry.root.exists()


@pytest.mark.parametrize("policy", ["mask", "zero"])
def test_package_facade_runs_without_hpo_and_retains_scores_after_close(spark, tmp_path, policy):
    from autoforecast import register_study, run_study
    from autoforecast.study_workflow import load_forecast_workflow

    path = configuration(tmp_path, actual_missing=policy)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    config = yaml.safe_load(profile.read_text())
    config.pop("hpo")
    profile.write_text(yaml.safe_dump(config))
    calls = []

    def model(request):
        calls.append(request.purpose)
        assert request.training_data.filter("value is null").count() == 0
        return predict(request)

    result = run_study(study_id, model, registry=registry, model_config=profile,
                       implementation=implementation(), log_level="ERROR")
    assert calls == ["validation", "refit"]
    assert result.tuning_path is None
    assert result.study_id == study_id and result.parent_study_id is None
    assert {name: result.summary()[name] for name in (
        "original_series", "retained_series", "excluded_series",
    )} == {"original_series": 2, "retained_series": 2, "excluded_series": 0}
    assert result.coverage["validation"]["originally_missing"] == 1
    assert result.coverage["validation"]["zero_filled"] == (1 if policy == "zero" else 0)
    assert result.validation_metrics["mae"] == pytest.approx(5.0 if policy == "zero" else 20 / 3)
    assert result.testing_metrics["mae"] == pytest.approx(6.5 if policy == "zero" else 26 / 3)
    assert load_forecast_workflow(result.path).coverage == result.coverage
    assert list(registry.scratch_root.iterdir()) == []
    audit = result.publication["audit_path"]
    events = [json.loads(line) for line in pathlib.Path(audit).read_text().splitlines()]
    assert [(e["stage"], e["status"]) for e in events if e["status"] == "SKIP"] == [
        ("tracking_setup", "SKIP"), ("tracking_publication", "SKIP"), ("table_publication", "SKIP"),
    ]
    assert events[-1]["status"] == "PASS"
    if policy == "mask":
        def second_model(request):
            assert request.model_parameters["level"] == 10.0
            return predict(request)

        other = run_study(study_id, second_model, registry=registry, model_config=profile,
                          implementation={**implementation(), "version": "2"},
                          fixed_parameters={"model.level": 10.0})
        assert other.study_id == result.study_id and other.run_id != result.run_id
        assert other.coverage == result.coverage
        assert other.testing_metrics["mae"] == pytest.approx(4 / 3)
        assert load_forecast_workflow(result.path).testing_metrics == result.testing_metrics


def test_fresh_process_uses_storage_only_and_builtin_retains_fitted_history(
    spark, tmp_path, monkeypatch,
):
    path = configuration(tmp_path)
    monkeypatch.setenv("AUTOFORECAST_REGISTRY_ROOT", str(tmp_path / "environment-registry"))
    monkeypatch.setenv("AUTOFORECAST_SCRATCH_ROOT", str(tmp_path / "environment-scratch"))
    registry = StudyRegistry.from_environment(spark=spark)
    study_id = register_study(path, spark=spark)
    assert not (tmp_path / "registry").exists()
    assert registry.scratch_root == tmp_path / "environment-scratch"
    path.unlink()
    (tmp_path / "source.parquet").unlink()
    script = '''
import json, sys
from pyspark.sql import SparkSession
from autoforecast import run_study
spark = (SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "2").getOrCreate())
try:
    result = run_study(sys.argv[1], "seasonal_naive")
    print("RESULT=" + json.dumps(result.summary()))
finally:
    spark.stop()
'''
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("AUTOFORECAST_")}
    environment["AUTOFORECAST_REGISTRY_ROOT"] = str(registry.root)
    completed = subprocess.run([sys.executable, "-c", script, study_id], env=environment,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(next(line.removeprefix("RESULT=") for line in completed.stdout.splitlines()
                             if line.startswith("RESULT=")))
    assert result["study_id"] == study_id
    from autoforecast.study_workflow import load_forecast_workflow

    workflow = load_forecast_workflow(registry.root / "runs" / result["run_id"])
    model = workflow.validation["model_artifacts"]
    assert pq.read_table(model / "history").num_rows == 9
    assert json.loads((model / "panel.json").read_text())["origin"] == "2024-01-05"
    from autoforecast.adapters.naive import seasonal_naive
    from autoforecast.panel import Panel

    state = json.loads((model / "panel.json").read_text())
    origin = state.pop("origin")
    state["path"] = str(model / "history")
    restored = seasonal_naive(Panel.from_record(state), origin, 2, spark=spark,
                              out_dir=tmp_path / "reloaded-predictions")
    expected = pq.read_table(workflow.validation["forecasts"].path).to_pylist()
    actual = pq.read_table(restored.path).to_pylist()
    assert sorted(actual, key=lambda r: (r["series_id"], r["date"])) == sorted(
        expected, key=lambda r: (r["series_id"], r["date"]),
    )


def test_tuning_zero_policy_and_failure_audit_keep_original_exception(spark, tmp_path, monkeypatch):
    from autoforecast import run_mlflow, run_study
    from autoforecast.study import Study

    path = configuration(tmp_path, actual_missing="zero")
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path, budget=1)
    calls = []

    def model(request):
        calls.append(request.purpose)
        return predict(request)

    result = run_study(study_id, model, registry=registry, model_config=profile,
                       implementation=implementation())
    assert calls == ["selection", "validation", "refit"]
    events = [json.loads(line) for line in (result.path / "validation/events.jsonl")
              .read_text().splitlines()]
    fit = next(e for e in events if e["stage"] == "fit_predict" and e["status"] == "PASS")
    assert fit["history_zero_filled"] == 1 and fit["warnings"] == ["history_zero_filled"]
    from autoforecast.study_tuning import load_tuning_result

    selected = load_tuning_result(result.tuning_path)
    assert selected.windows[0]["score"]["coverage"]["zero_filled"] == 1
    monkeypatch.setattr(run_mlflow, "_mlflow", lambda: pytest.fail("tracking disabled"))
    real_close = Study.close

    def bad_close(study):
        real_close(study)
        raise OSError("cleanup failure")

    monkeypatch.setattr(Study, "close", bad_close)

    def broken(request):
        artifact = request.output_dir / "artifacts"
        artifact.mkdir()
        (artifact / "bad").symlink_to(tmp_path / "absent")
        raise RuntimeError("primary fitting failure")

    with pytest.raises(RuntimeError, match="primary fitting failure") as caught:
        run_study(study_id, broken, registry=registry, model_config=profile,
                  implementation=implementation(), fixed_parameters={}, log_level="CRITICAL")
    events = [json.loads(line) for line in caught.value.audit_path.read_text().splitlines()]
    assert events[-1]["status"] == "FAIL"
    assert not any(e["stage"] == "table_publication" for e in events)
    assert not list(registry.scratch_root.iterdir())


def test_reference_exclusion_is_one_recorded_cohort_and_empty_total_blocks(spark, tmp_path):
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["recipe"]["metric"] = "wrmsse"
    value["references"] = {"path": "references.parquet"}
    value["invalid_references"] = "exclude"
    value["dataset"]["schema_version"] = 2
    value["dataset"]["dataset"]["required_series_count"] = 2
    path.write_text(yaml.safe_dump(value))
    rows = [{"keys": {"item": sid}, "stage": stage, "origin": origin,
             "weight": 1.0 if sid == "A" else -2.0, "scale": 1.0 if sid == "A" else 0.0}
            for sid in ("A", "B") for stage, origin in
            (("tuning", "2024-01-03"), ("validation", "2024-01-05"), ("test", "2024-01-07"))]
    reference_path = tmp_path / "references.parquet"
    pq.write_table(pa.Table.from_pylist(rows), reference_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    assert register_study(path, registry=registry) == study_id
    definition, _ = registry.definition(study_id)
    assert definition["source"]["starting_series"] == 2
    assert definition["admission"]["retained_series"] == 1
    assert definition["admission"]["excluded_keys"] == [{"item": "B"}]
    assert {row["issue_code"] for row in definition["admission"]["findings"]} == {
        "INVALID_WEIGHT", "INVALID_SCALE",
    }
    assert len(definition["references"]["excluded_references"]) == 3
    with registry.load_study(study_id) as study:
        assert study.n_series == 1
        assert study.training_data.count() == 5
        assert study.validation_data.count() == study.testing_data.count() == 2
    for row in rows:
        if row["keys"]["item"] == "A":
            row["weight"] = 0.0
    pq.write_table(pa.Table.from_pylist(rows), reference_path)
    with pytest.raises(ValueError, match="positive total"):
        register_study(path, registry=registry)
    assert len(list((registry.root / "registrations").glob("*.json"))) == 1


def test_derived_registration_and_explicit_controls_override_environment(spark, tmp_path, monkeypatch):
    from autoforecast import run_mlflow, run_study

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    parent_id = register_study(path, registry=registry)
    parent, parent_sha = registry.definition(parent_id)
    profile = model_profile(tmp_path)
    value = yaml.safe_load(profile.read_text())
    value.pop("hpo")
    value["series_policy"] = "exclude_unavailable"
    value["minimum_history"] = 3  # A has 3 observed rows at the first inner origin; B has 2.
    profile.write_text(yaml.safe_dump(value))
    monkeypatch.setenv("AUTOFORECAST_TRACKING", "mlflow")
    monkeypatch.setenv("AUTOFORECAST_PUBLISH", "true")
    monkeypatch.setattr(run_mlflow, "_mlflow", lambda: pytest.fail("explicit tracking disabled"))
    result = run_study(parent_id, predict, registry=registry, model_config=profile,
                       implementation=implementation(), tracking="none", publish=False,
                       mlflow_experiment=None, tracking_uri=None)
    assert result.study_id != parent_id and result.parent_study_id == parent_id
    child, _ = registry.definition(result.study_id)
    assert child["snapshot"] == parent["snapshot"]
    assert child["parent"] == {"study_id": parent_id, "study_sha256": parent_sha}
    assert child["effective_series"] == 1
    assert result.summary()["retained_series"] == 1
    assert result.summary()["excluded_series"] == 1
    assert registry.definition(parent_id) == (parent, parent_sha)
    with registry.load_study(result.study_id) as study:
        assert study.n_series == 1


def test_calculated_references_reuse_existing_daily_calculator(spark, tmp_path):
    from autoforecast.benchmark import artifact_path

    path = configuration(tmp_path)
    source = tmp_path / "source.parquet"
    table = pq.read_table(source)
    # This calculator smoke uses complete observed history; explicit null history
    # intentionally produces unavailable scales and is covered by its own tests.
    table = pa.Table.from_pylist([
        {**row, "value": float(row["day"].day)} for row in table.to_pylist()
    ])
    pq.write_table(table.append_column("importance", pa.array([1.0] * len(table))), source)
    settings = yaml.safe_load(path.read_text())
    settings["recipe"]["metric"] = "wrmsse"
    settings["references"] = {"weight_col": "importance", "missing_history": "na",
                               "duplicate_policy": "error", "scale_start": "history_start"}
    path.write_text(yaml.safe_dump(settings))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    definition, _ = registry.definition(study_id)
    assert definition["references"]["preparation"] == settings["references"]
    with registry.load_study(study_id) as study:
        record = study._record()
        references = pq.read_table(artifact_path(study._path, record, "scoring_references"))
        inner = [row for row in references.to_pylist() if row["boundary"].startswith("tuning/")]
        assert {json.loads(row["series_id"])["item"]: row["weight"] for row in inner} == {
            "A": 3.0, "B": 2.0,
        }
        assert all(row["scale"] == 1.0 for row in inner)


@pytest.mark.parametrize("declared", [False, True])
def test_daily_timestamp_source_registers_inferred_and_declared_starts(spark, tmp_path, declared):
    path = configuration(tmp_path)
    frame = spark.read.parquet(str(tmp_path / "source.parquet"))
    replacement = tmp_path / "timestamp-source"
    frame.withColumn("day", F.col("day").cast("timestamp_ntz")).write.parquet(str(replacement))
    value = yaml.safe_load(path.read_text())
    value["source"]["path"] = str(replacement)
    if declared:
        value["recipe"]["series_starts"] = {"rule": "declared", "values": [
            {"keys": {"item": "A"}, "start": "2024-01-01"},
            {"keys": {"item": "B"}, "start": "2024-01-02"},
        ]}
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    with registry.load_study(study_id) as study:
        assert study.training_data.count() == 9


@pytest.mark.parametrize("case", [
    "season_length", "estimand", "negative_target_policy", "frequency_mismatch",
    "availability_mapping", "future_without_exposure", "status_probabilities", "plan",
    "weight_window_days", "scale_lag", "scale_start", "missing_history", "duplicate_policy",
])
def test_invalid_scientific_config_rejects_before_source(spark, tmp_path, monkeypatch, case):
    from types import SimpleNamespace

    import autoforecast.study_registration as registration

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    panel = value["dataset"]["dataset"]["panel"]
    if case == "season_length":
        panel["season_length"] = 0
    elif case == "estimand":
        panel["target_estimand"] = "observed_count"
    elif case == "negative_target_policy":
        panel["negative_target_policy"] = "floor"
    elif case == "frequency_mismatch":
        panel["frequency"] = "2D"
    elif case in {"availability_mapping", "future_without_exposure"}:
        panel["known_future"] = ["weather"]
        if case == "future_without_exposure":
            value["recipe"]["feature_availability"] = {"weather": "known_at"}
    elif case == "status_probabilities":
        panel["status_prob_cols"] = {"exact": "p_exact", "lower_bound": "p_lower",
                                     "unobserved": "p_none"}
    elif case == "plan":
        panel["plan"] = {"path": "unused", "time_col": "day"}
    else:
        value["recipe"]["metric"] = "wrmsse"
        value["references"] = {"weight_col": "importance", "missing_history": "na",
                                "duplicate_policy": "error"}
        value["references"][case] = {
            "weight_window_days": 0, "scale_lag": 0, "scale_start": "unknown",
            "missing_history": "unknown", "duplicate_policy": "unknown",
        }[case]
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry.from_config(path, spark=spark)
    touched = []
    class GuardedReader:
        def parquet(self, path):
            touched.append(path)
            raise AssertionError("SOURCE ACCESS BEFORE SCIENTIFIC CONFIG REJECTION")
    class GuardedSpark:
        read = GuardedReader()
    registry.spark = GuardedSpark()
    registry.store = SimpleNamespace(prepare=lambda **_: pytest.fail("provision reached"))
    try:
        with pytest.raises(ValueError):
            registration.register_study(path, registry=registry)
    finally:
        print(json.dumps({"case": case, "source_touched": bool(touched),
                          "root_created": registry.root.exists()}))
    assert not touched
    assert not registry.root.exists()


def test_explicit_publish_overrides_invalid_environment_before_validation(tmp_path, monkeypatch):
    from autoforecast import run_study
    class ReachedRegistrationLookup(RuntimeError):
        pass
    class Registry:
        runtime = {}
        root = tmp_path
        def _registration_path(self, study_id):
            raise ReachedRegistrationLookup("explicit publish=False survived ignored environment")
    monkeypatch.setenv("AUTOFORECAST_PUBLISH", "irrelevant-invalid-value")
    with pytest.raises(ReachedRegistrationLookup):
        run_study("known-id", "seasonal_naive", registry=Registry(), publish=False)


def test_zero_checkpoint_reader_uses_filled_target_and_factual_mask(spark, tmp_path):
    import dataclasses
    import hashlib

    import numpy as np

    from autoforecast.benchmark import evaluation_contract
    from autoforecast.study_validation import StudyValidation

    path = configuration(tmp_path, actual_missing="zero")
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    with registry.load_study(study_id) as study:
        fixed = evaluation_contract(study._path, stage="validation", origin="2024-01-05",
                                    horizon=2, metric="mae")
        contract = tmp_path / "checkpoint-contract.json"
        contract.write_text(json.dumps(fixed))
        window = StudyValidation.read(
            contract, hashlib.sha256(contract.read_bytes()).hexdigest(),
            panel=dataclasses.replace(study.panel, path=fixed["actuals"]["path"]),
            origin="2024-01-05", horizon=2,
            series_ids=tuple(row[study.panel.series_col] for row in study.series_mapping.collect()),
        )
    assert np.isfinite(window.target).all()
    assert not window.target.flags.writeable and not window.observed.flags.writeable
    assert (~window.observed).sum() == 1
    assert (window.target[~window.observed] == 0).all()
    assert window.score(np.zeros((2, 2)))["mae"] == pytest.approx(5.0)


def test_registered_load_preserves_primary_hash_error_when_close_fails(tmp_path, monkeypatch):
    import autoforecast.study_registration as registration
    registry = StudyRegistry.__new__(StudyRegistry)
    registry.root = tmp_path
    registry.spark = object()
    registry.scratch_root = tmp_path
    monkeypatch.setattr(registry, "definition", lambda _: ({"library_sha256": "a" * 64}, "sha"))
    class ChangedStudy:
        _sha256 = "b" * 64
        def close(self):
            raise OSError("secondary cleanup failure")
    monkeypatch.setattr(registration, "load_study", lambda *args, **kwargs: ChangedStudy())
    with pytest.raises(ValueError, match="manifest changed"):
        registry.load_study("id")


def test_context_entry_preserves_primary_hash_error_when_close_fails(tmp_path, monkeypatch):
    from autoforecast.study import Study
    class Snapshots:
        name = str(tmp_path)
        def cleanup(self):
            raise OSError("secondary cleanup failure")
    study = Study(tmp_path / "benchmark.json", "sha", object(), Snapshots())
    def fail_record(self, **kwargs):
        raise ValueError("primary manifest changed")
    monkeypatch.setattr(Study, "_record", fail_record)
    with pytest.raises(ValueError, match="primary manifest changed"):
        study.__enter__()


@pytest.mark.parametrize("reference_source", ["calculated", "supplied"])
def test_new_series_exclusion_precedes_reference_admission(spark, tmp_path, reference_source):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = configuration(tmp_path)
    source = tmp_path / "source.parquet"
    rows = [{**row, "store": 1, "value": float(row["day"].day), "importance": 1.0}
            for row in pq.read_table(source).to_pylist()]
    rows.append({"item": "late", "store": 1, "day": dt.date(2024, 1, 9), "value": 1.0,
                 "importance": 1.0})
    pq.write_table(pa.Table.from_pylist(rows), source)
    value = yaml.safe_load(path.read_text())
    value["recipe"].update(metric="wrmsse", new_series_policy="exclude_after_first_origin",
                            source_keys=["item", "store"])
    value["references"] = {"weight_col": "importance", "missing_history": "na",
                            "duplicate_policy": "error", "scale_start": "history_start"}
    if reference_source == "supplied":
        value["references"] = {"path": "references.parquet"}
        references = [
            {"keys": {"store": 1, "item": item}, "stage": stage, "origin": origin,
             "weight": None if item == "late" else 1.0,
             "scale": None if item == "late" else 1.0}
            for item in ("A", "B", "late") for stage, origin in (
                ("tuning", "2024-01-03"), ("validation", "2024-01-05"), ("test", "2024-01-07"),
            )
        ]
        pq.write_table(pa.Table.from_pylist(references), tmp_path / "references.parquet")
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    definition, _ = registry.definition(study_id)
    assert definition["admission"]["excluded_new_keys"] == [{"item": "late", "store": 1}]
    assert definition["effective_series"] == 2
    assert definition["source"]["eligible_series"] == 2
    assert definition["admission"]["excluded_keys"] == []
    value["recipe"]["new_series_policy"] = "error"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="new series absent"):
        register_study(path, registry=registry)
    assert len(list((registry.root / "registrations").glob("*.json"))) == 1


def test_zero_actual_available_wrmsse_and_horizons_match_filled_cells(spark, tmp_path):
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast import run_study
    path = configuration(tmp_path, actual_missing="zero")
    value = yaml.safe_load(path.read_text())
    value["recipe"]["metric"] = "wrmsse"
    value["references"] = {"path": "references.parquet"}
    path.write_text(yaml.safe_dump(value))
    pq.write_table(pa.Table.from_pylist([
        {"keys": {"item": item}, "stage": stage, "origin": origin,
         "weight": 1.0, "scale": 1.0}
        for item in ("A", "B") for stage, origin in (
            ("tuning", "2024-01-03"), ("validation", "2024-01-05"), ("test", "2024-01-07"))
    ]), tmp_path / "references.parquet")
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    model = yaml.safe_load(profile.read_text())
    model.pop("hpo")
    model.update(series_policy="use_available", minimum_history=1,
                 reporting={"metrics": ["wrmsse"], "forecast_accuracy_breakdown": "both"})
    profile.write_text(yaml.safe_dump(model))
    result = run_study(study_id, predict, registry=registry, model_config=profile,
                       implementation=implementation())
    expected = (math.sqrt((6**2 + 7**2) / 2) + math.sqrt((0**2 + 7**2) / 2)) / 2
    assert result.validation_metrics["wrmsse"] == pytest.approx(expected)
    score = result.validation["score"]
    assert score["availability"]["common_dates"] == ["2024-01-06", "2024-01-07"]
    assert score["population"]["observed_cells"] == 3
    assert score["population"]["filled_cells"] == 1
    assert score["population"]["scored_cells"] == 4
    breakdown = score["forecast_accuracy_breakdown"]
    assert [row["wrmsse"] for row in breakdown["per_step"]] == pytest.approx([3.0, 7.0])
    assert [row["wrmsse"] for row in breakdown["cumulative"]] == pytest.approx([3.0, expected])
    assert score["coverage"]["original_expected"] == 4
    assert breakdown["cumulative"][-1]["coverage"] == {
        "expected": 4, "originally_missing": 1, "zero_filled": 1, "excluded": 0, "scored": 4,
    }


@pytest.mark.parametrize("updates", [
    {"unknown_key": True},
    {"source": {"kind": "csv", "path": "source.parquet"}},
    {"source": {"kind": "delta", "sql": "source.sql", "tables": {}}},
    {"invalid_references": "ignore"},
    {"runtime": {"tracking": "other"}},
    {"runtime": {"publish": "false"}},
    {"runtime": {"log_level": "LOUD"}},
    {"runtime": {"tracking_uri": " "}},
    {"runtime": {"mlflow_experiment": " "}},
])
def test_invalid_configuration_never_opens_source_or_provisions(tmp_path, updates):
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value.update(updates)
    path.write_text(yaml.safe_dump(value))
    (tmp_path / "source.sql").write_text("SELECT * FROM observations")
    (tmp_path / "source.parquet").unlink()
    with pytest.raises(ValueError):
        register_study(path)
    assert not (tmp_path / "registry").exists()
    value = yaml.safe_load(configuration(tmp_path).read_text())
    value["recipe"]["metric"] = "wrmsse"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="WRMSSE requires"):
        register_study(path)
    assert not (tmp_path / "registry").exists()


def test_structural_findings_and_source_count_block_before_freezing(spark, tmp_path, monkeypatch):
    import autoforecast.study_registration as registration

    path = configuration(tmp_path)
    source = tmp_path / "source.parquet"
    table = pq.read_table(source)
    registry = StudyRegistry.from_config(path, spark=spark)
    monkeypatch.setattr(registration, "freeze_benchmark", lambda **_: pytest.fail("freeze reached"))
    pq.write_table(pa.concat_tables([table, table]), source)
    with pytest.raises(ValueError, match="structural source issues"):
        register_study(path, registry=registry)
    findings = [pq.read_table(folder).to_pylist()
                for folder in (registry.root / "checks").glob("*/source")]
    assert any(row["issue_code"] == "DUPLICATE_DATE" for rows in findings for row in rows)
    pq.write_table(table, source)
    settings = yaml.safe_load(path.read_text())
    settings["dataset"]["schema_version"] = 2
    settings["dataset"]["dataset"]["required_series_count"] = 3
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="required_series_count before exclusions"):
        register_study(path, registry=registry)
    assert not (registry.root / "registrations").exists()


def test_environment_config_overrides_and_registration_views_are_isolated(spark, tmp_path,
                                                                         monkeypatch):
    first = configuration(tmp_path / "first")
    second = configuration(tmp_path / "second")
    monkeypatch.setenv("AUTOFORECAST_CONFIG", str(first))
    monkeypatch.delenv("AUTOFORECAST_REGISTRY_ROOT", raising=False)
    configured = StudyRegistry.from_environment(spark=spark)
    assert configured.root == first.parent / "registry"
    monkeypatch.setenv("AUTOFORECAST_REGISTRY_ROOT", str(tmp_path / "override"))
    overridden = StudyRegistry.from_environment(spark=spark)
    assert overridden.root == tmp_path / "override"
    for path, registry in ((first, configured), (second, StudyRegistry.from_config(
        second, spark=spark,
    ))):
        study_id = register_study(path, registry=registry)
        with registry.load_study(study_id) as study:
            binding = study.registration
            expected = study.registration
            binding["sha256"] = "changed"
            binding["definition"]["admission"]["retained_series"] = 999
            assert study.registration == expected
            assert study.training_data.count() == 9
    assert configured.root != registry.root
    assert not overridden.root.exists()


@pytest.mark.parametrize("explicit,environment,expected", [
    (False, "invalid", "lookup"), (True, "invalid", "lookup"),
    (None, "invalid", "environment"), (None, "false", "lookup"),
    ("false", "true", "boolean"),
])
def test_effective_publish_control_is_resolved_before_validation(tmp_path, monkeypatch,
                                                                explicit, environment, expected):
    from autoforecast import run_study

    class LookupReached(RuntimeError):
        pass

    class Registry:
        runtime = {}
        root = tmp_path

        def _registration_path(self, study_id):
            raise LookupReached("lookup")

    monkeypatch.setenv("AUTOFORECAST_PUBLISH", environment)
    options = {} if explicit is None else {"publish": explicit}
    if expected == "lookup":
        with pytest.raises(LookupReached):
            run_study("known", "seasonal_naive", registry=Registry(), **options)
    else:
        with pytest.raises(ValueError, match="AUTOFORECAST_PUBLISH" if expected == "environment"
                           else "publish must be boolean"):
            run_study("known", "seasonal_naive", registry=Registry(), **options)


def test_model_entrypoint_and_runtime_failures_precede_fitting(spark, tmp_path, monkeypatch):
    import autoforecast.test_study_tuning as model_module
    from autoforecast import run_study
    from autoforecast.study_campaign import StudyModel

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    calls = []

    def model(request):
        calls.append(request.purpose)
        return predict(request)

    monkeypatch.setattr(model_module, "predict", model)
    selected = StudyModel("autoforecast.test_study_tuning:predict", implementation(), profile, {})
    result = run_study(study_id, selected, registry=registry)
    assert result.tuning_path is None and calls == ["validation", "refit"]
    for options, message in (
        ({"implementation": implementation()}, "StudyModel already declares"),
        ({"model_config": profile}, "StudyModel already declares"),
        ({"tracking": "other"}, "tracking must"),
        ({"publish": 1}, "publish must be boolean"),
        ({"log_level": "LOUD"}, "log_level must"),
        ({"tracking": "none", "mlflow_experiment": "example"}, "destinations conflict"),
        ({"tracking": "none", "tracking_uri": "local"}, "destinations conflict"),
    ):
        with pytest.raises(ValueError, match=message):
            run_study(study_id, selected, registry=registry, **options)
    with pytest.raises(ValueError, match="callbacks require"):
        run_study(study_id, model, registry=registry)
    with pytest.raises(ValueError, match="model must be"):
        run_study(study_id, "unknown", registry=registry)
    assert calls == ["validation", "refit"]


def test_tracking_setup_failures_do_not_fit_or_create_experiments(spark, tmp_path, monkeypatch):
    import mlflow

    from autoforecast import run_mlflow, run_study

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    uri = "sqlite:///" + str(tmp_path / "tracking.db")
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    deleted = client.create_experiment("deleted", artifact_location=str(tmp_path / "artifacts"))
    client.delete_experiment(deleted)
    monkeypatch.setattr(client, "create_experiment", lambda *a, **k: pytest.fail("implicit create"))
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda **_: client)
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(run_mlflow, "_mlflow", lambda: mlflow)
    profile = model_profile(tmp_path)

    def forbidden(request):
        pytest.fail("tracking setup failure must precede fitting")

    for destination, experiment, message in (
        ("", "missing", "tracking requires"), (uri, "missing", "existing active"),
        (uri, deleted, "existing active"),
    ):
        with pytest.raises(run_mlflow.RunMLflowError, match=message) as caught:
            run_study(study_id, forbidden, registry=registry, model_config=profile,
                      implementation=implementation(), tracking="mlflow",
                      tracking_uri=destination, mlflow_experiment=experiment)
        events = [json.loads(line) for line in caught.value.audit_path.read_text().splitlines()]
        assert any(e["stage"] == "tracking_setup" and e["status"] == "FAIL" for e in events)
        assert not any(e["stage"] in {"workflow", "table_publication"} for e in events)
    monkeypatch.setattr(mlflow, "active_run", lambda: object())
    with pytest.raises(run_mlflow.RunMLflowError, match="end the active"):
        run_study(study_id, forbidden, registry=registry, model_config=profile,
                  implementation=implementation(), tracking="mlflow",
                  tracking_uri=uri, mlflow_experiment="missing")


def test_unavailable_testing_is_retained_in_facade_events_and_reload(spark, tmp_path):
    from autoforecast import run_study
    from autoforecast.study_workflow import load_forecast_workflow

    path = configuration(tmp_path)
    source = tmp_path / "source.parquet"
    rows = [{**row, "value": None if row["day"].day >= 8 else row["value"]}
            for row in pq.read_table(source).to_pylist()]
    pq.write_table(pa.Table.from_pylist(rows), source)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    value = yaml.safe_load(profile.read_text())
    value.pop("hpo")
    value.update(series_policy="use_available", minimum_history=1)
    profile.write_text(yaml.safe_dump(value))
    result = run_study(study_id, predict, registry=registry, model_config=profile,
                       implementation=implementation(), log_level="ERROR")
    assert result.status == "pending_actuals"
    assert result.testing_metrics == dict.fromkeys(("mae", "rmse", "wape", "bias"))
    assert result.testing["score"]["metric_status"]["mae"]["reason"]
    assert load_forecast_workflow(result.path).summary()["testing"] == result.testing_metrics
    events = [json.loads(line) for line in (result.path / "test/events.jsonl").read_text().splitlines()]
    assert any(e["stage"] == "scoring" and e["status"] == "UNAVAILABLE" for e in events)
    audit = [json.loads(line) for line in pathlib.Path(
        result.publication["audit_path"],
    ).read_text().splitlines()]
    assert audit[-1]["stage"] == "run" and audit[-1]["status"] == "UNAVAILABLE"
    assert not list(registry.scratch_root.iterdir())


def test_scoring_failure_preserves_primary_error_and_real_stage_events(spark, tmp_path, monkeypatch):
    from mlops.run_bundle import RunBundle

    from autoforecast import run_study, study_tuning

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    original_write = RunBundle.write_json
    attempts = []

    def fail_record(bundle, name, record):
        if name == "failure.json":
            attempts.append(name)
            raise OSError("secondary failure-record error")
        return original_write(bundle, name, record)

    def fail_score(*args, **kwargs):
        raise RuntimeError("primary scoring failure")

    monkeypatch.setattr(RunBundle, "write_json", fail_record)
    monkeypatch.setattr(study_tuning, "score_study", fail_score)
    with pytest.raises(RuntimeError, match="primary scoring failure") as caught:
        run_study(study_id, predict, registry=registry, model_config=profile,
                  implementation=implementation(), fixed_parameters={})
    assert attempts == ["failure.json"]
    attempt_events = [json.loads(line)
                      for line in caught.value.audit_path.read_text().splitlines()]
    assert [(e["stage"], e["status"]) for e in attempt_events if e["status"] == "FAIL"] == [
        ("workflow", "FAIL"), ("run", "FAIL"),
    ]
    assert not any(e["stage"] in {"table_publication", "tracking_publication"}
                   for e in attempt_events)
    events = [json.loads(line) for file in (registry.root / "runs").glob("*/validation/events.jsonl")
              for line in file.read_text().splitlines()]
    assert [(e["stage"], e["status"]) for e in events if e["status"] != "START"] == [
        ("fit_predict", "PASS"), ("artifact_retention", "PASS"), ("scoring", "FAIL"),
    ]
    assert not list((registry.root / "runs").glob("*/test/events.jsonl"))
    assert not list(registry.scratch_root.iterdir())


def test_secondary_audit_and_trial_retention_errors_preserve_primary(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from autoforecast import study_tuning
    from autoforecast.study import Study

    def bad_event(path, stage, status, **context):
        if status == "FAIL":
            raise OSError("secondary audit error")

    monkeypatch.setattr(study_tuning, "_stage_event", bad_event)
    with pytest.raises(RuntimeError, match="primary operation"):
        with study_tuning._stage(tmp_path / "audit", "operation"):
            raise RuntimeError("primary operation")

    def bad_fit(*args, **kwargs):
        raise RuntimeError("primary fit")

    def bad_retention(*args, **kwargs):
        raise OSError("secondary retention")

    monkeypatch.setattr(study_tuning, "_fit_window", bad_fit)
    bridge = study_tuning._ExternalTrialBridge(
        object(), lambda _: None, (object(),),
        json.dumps({"model_id": "synthetic", "seed": 1,
                    "options": {"metrics": [], "breakdown": "none"}}).encode(),
        b"{}", b"{}",
    )
    request = SimpleNamespace(
        attempt=SimpleNamespace(path=tmp_path, write_bytes=lambda *a: None, write_json=bad_retention),
        model_parameters={}, training_parameters={}, seed=1,
        study=SimpleNamespace(id="synthetic"), trial_id="trial",
    )
    with pytest.raises(RuntimeError, match="primary fit"):
        bridge(request)
    study = Study(tmp_path / "benchmark.json", "sha", object(),
                  SimpleNamespace(name=str(tmp_path), cleanup=bad_retention))
    with pytest.raises(OSError, match="secondary retention"):
        study.__exit__(None, None, None)


@pytest.mark.parametrize("model,training", [({}, {}), ({"other": 1}, {}),
                                             ({"season_length": 1}, {"epochs": 1})])
def test_builtin_settings_reject_unsupported_parameters_before_data_access(model, training):
    from types import SimpleNamespace

    from autoforecast.adapters.naive import fit_study_baseline

    request = SimpleNamespace(model_parameters=model, training_parameters=training)
    with pytest.raises(ValueError, match="require only model.season_length"):
        fit_study_baseline(request)


@pytest.mark.parametrize("defect", ["extra", "duplicate", "missing"])
def test_reference_exclusion_cannot_hide_structural_cohort_errors(spark, tmp_path, defect):
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["recipe"]["metric"] = "wrmsse"
    value["references"] = {"path": "references.parquet"}
    value["invalid_references"] = "exclude"
    path.write_text(yaml.safe_dump(value))
    rows = [
        {"keys": {"item": item}, "stage": stage, "origin": origin,
         "weight": 1.0 if item == "A" else -1.0, "scale": 1.0}
        for item in ("A", "B") for stage, origin in (
            ("tuning", "2024-01-03"), ("validation", "2024-01-05"), ("test", "2024-01-07"),
        )
    ]
    if defect == "extra":
        rows.append({**rows[-1], "keys": {"item": "unrelated"}})
    elif defect == "duplicate":
        rows.append(dict(rows[-1]))
    else:
        rows.pop()
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "references.parquet")
    registry = StudyRegistry.from_config(path, spark=spark)
    with pytest.raises(ValueError, match="cover every retained series and boundary once"):
        register_study(path, registry=registry)
    assert not (registry.root / "registrations").exists()


@pytest.mark.parametrize("selector", ["REGISTRY_ROOT", "SCRATCH_ROOT", "DELTA_NAMESPACE"])
def test_registration_rejects_invalid_selected_environment_before_io(
    tmp_path, monkeypatch, selector,
):
    path = configuration(tmp_path)
    monkeypatch.setenv("AUTOFORECAST_" + selector, "")
    with pytest.raises(ValueError, match="storage environment"):
        register_study(path, spark=object())
    assert not (tmp_path / "registry").exists()


def test_explicit_registration_registry_overrides_storage_environment(spark, tmp_path, monkeypatch):
    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    for selector in ("REGISTRY_ROOT", "SCRATCH_ROOT", "DELTA_NAMESPACE"):
        monkeypatch.setenv("AUTOFORECAST_" + selector, "")
    study_id = register_study(path, registry=registry)
    assert registry.definition(study_id)[0]["study_id"] == study_id


def test_summary_keeps_cohort_retention_distinct_from_stage_forecastability(spark, tmp_path):
    from autoforecast import run_study

    path = configuration(tmp_path)
    source = tmp_path / "source.parquet"
    rows = [r for r in pq.read_table(source).to_pylist() if r["item"] == "A" or r["day"].day >= 5]
    rows = [
        {**r, "value": None if r["item"] == "B" and r["day"].day == 5 else r["value"]} for r in rows
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    registry = StudyRegistry.from_config(path, spark=spark)
    parent = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    value = yaml.safe_load(profile.read_text())
    value.pop("hpo")
    value.update(series_policy="use_available", minimum_history=1)
    profile.write_text(yaml.safe_dump(value))
    result = run_study(
        parent, predict, registry=registry, model_config=profile, implementation=implementation()
    )
    report = json.loads((result.path / "metadata/preflight.json").read_text())
    summary = result.summary()
    print(
        "COHORT_VS_FORECAST="
        + json.dumps(
            {
                "summary": summary,
                "preflight_retained": report["retained_count"],
                "preflight_excluded": report["excluded_ids"],
                "validation_population": result.validation["score"]["population"],
                "validation_availability": result.validation["score"]["availability"],
                "testing_population": result.testing["score"]["population"],
            }
        )
    )
    assert report["retained_count"] == 2 and report["excluded_ids"] == []
    assert result.validation["score"]["population"]["forecasted_series"] == 1
    assert result.testing["score"]["population"]["forecasted_series"] == 2
    assert summary["retained_series"] == 2 and summary["excluded_series"] == 0
    assert summary["forecasted_series"] == {"validation": 1, "test": 2}
    from autoforecast.study_workflow import load_forecast_workflow

    reloaded = load_forecast_workflow(result.path).summary()
    assert {key: reloaded[key] for key in (
        "retained_series", "excluded_series", "forecasted_series",
    )} == {"retained_series": 2, "excluded_series": 0,
           "forecasted_series": {"validation": 1, "test": 2}}


def test_preflight_event_links_existing_report_when_metadata_copy_fails(
    spark, tmp_path, monkeypatch
):
    from mlops.run_bundle import RunBundle

    from autoforecast import run_study

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    value = yaml.safe_load(profile.read_text())
    value.pop("hpo")
    profile.write_text(yaml.safe_dump(value))
    native = RunBundle.write_json

    def fail_report_copy(self, name, value):
        if name == "metadata/preflight.json":
            raise OSError("injected preflight metadata copy failure")
        return native(self, name, value)

    monkeypatch.setattr(RunBundle, "write_json", fail_report_copy)
    with pytest.raises(OSError, match="injected preflight metadata copy failure"):
        run_study(
            study_id,
            predict,
            registry=registry,
            model_config=profile,
            implementation=implementation(),
        )
    events = [
        json.loads(line)
        for p in (registry.root / "runs").glob("*/events.jsonl")
        for line in p.read_text().splitlines()
    ]
    passed = [e for e in events if e["stage"] == "preflight" and e["status"] == "PASS"]
    factual = list((registry.root / "runs").glob("*/preflight/*/report.json"))
    print(
        "PREFLIGHT_REPORT_LINK="
        + json.dumps(
            {
                "events": passed,
                "factual_reports": [str(p) for p in factual],
                "linked_exists": [pathlib.Path(e["report_path"]).exists() for e in passed],
            }
        )
    )
    assert factual and all(p.exists() for p in factual)
    assert passed and all(pathlib.Path(e["report_path"]).exists() for e in passed)


@pytest.mark.parametrize("mode", ["fixed", "tuned"])
def test_post_tuning_copy_failure_is_not_a_failed_or_executed_search(
    spark, tmp_path, monkeypatch, mode
):
    from types import SimpleNamespace

    from mlops.run_bundle import RunBundle

    from autoforecast import run_study, study_workflow

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path, budget=1)
    if mode == "fixed":
        value = yaml.safe_load(profile.read_text())
        value.pop("hpo")
        profile.write_text(yaml.safe_dump(value))
        failed_path = "metadata/winner-config.json"
    else:
        # This seam replaces only the completed search; the exact post-search retention path runs.
        monkeypatch.setattr(
            study_workflow,
            "tune_study",
            lambda *a, **k: SimpleNamespace(path=tmp_path / "completed-search"),
        )
        failed_path = "metadata/tuning-reference.json"
    native = RunBundle.write_json

    def fail_post_tuning_copy(self, name, value):
        if name == failed_path:
            raise OSError("injected post-tuning retention failure")
        return native(self, name, value)

    monkeypatch.setattr(RunBundle, "write_json", fail_post_tuning_copy)
    with pytest.raises(OSError, match="injected post-tuning retention failure"):
        run_study(
            study_id,
            predict,
            registry=registry,
            model_config=profile,
            implementation=implementation(),
        )
    events = [
        json.loads(line)
        for p in (registry.root / "runs").glob("*/events.jsonl")
        for line in p.read_text().splitlines()
    ]
    print(
        "POST_TUNING_PHASE="
        + json.dumps(
            {
                "mode": mode,
                "failed_path": failed_path,
                "events": [(e["stage"], e["status"]) for e in events],
            }
        )
    )
    assert any(
        e["stage"] == "tuning" and e["status"] == ("SKIP" if mode == "fixed" else "PASS")
        for e in events
    )
    assert not any(e["stage"] == "tuning" and e["status"] == "FAIL" for e in events)
    assert any(e["stage"] == "artifact_retention" and e["status"] == "FAIL" for e in events)
