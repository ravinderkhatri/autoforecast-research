"""Public supplied-stage workflows use small synthetic data with known dates and scores."""

import copy
import datetime as dt
import hashlib
import json
import pathlib
import shutil

import pytest
from pyspark.sql import functions as F

from autoforecast import register_study
from autoforecast.study_registration import StudyRegistry
from autoforecast.test_study_tuning import config as model_profile
from autoforecast.test_study_tuning import implementation, predict
from autoforecast.test_study_tuning import spark as _spark


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def stages(spark, *, validation=True):
    frame = spark.createDataFrame([
        (key, dt.date(2024, 1, day), float(day))
        for key in ("A", "B") for day in range(1 if key == "A" else 2, 10)
    ], "item string, day date, value double")
    return dict(
        training_data=frame.filter(F.col("day") <= dt.date(2024, 1, 5 if validation else 7)),
        testing_data=frame.filter(F.col("day") >= dt.date(2024, 1, 8)),
        series_id=["item"], date_col="day", target_col="value", target_unit="units", frequency="D",
        training_date_range=("2024-01-01", "2024-01-05" if validation else "2024-01-07"),
        testing_date_range=("2024-01-08", "2024-01-09"), metric="mae",
        **(dict(validation_data=frame.filter(F.col("day").between(dt.date(2024, 1, 6),
                                                                dt.date(2024, 1, 7))),
                validation_date_range=("2024-01-06", "2024-01-07")) if validation else {}),
    )


@pytest.mark.parametrize("validation", [False, True])
def test_direct_registration_workflow_and_replay(spark, tmp_path, validation):
    from autoforecast.study_workflow import load_forecast_workflow

    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    study_id = register_study(registry=registry, **stages(spark, validation=validation))
    definition, _ = registry.definition(study_id)
    assert (definition["source"]["stages"]["validation"] is not None) == validation
    calls = []

    def model(request):
        calls.append(request.purpose)
        assert request.training_data.filter(F.col("day") >= dt.date(2024, 1, 8)).count() == 0
        return predict(request)

    with registry.load_study(study_id) as study:
        assert (study.validation_data is not None) == validation
        assert (study.validation_date_range is not None) == validation
        assert study.testing_date_range == (dt.date(2024, 1, 8), dt.date(2024, 1, 9))
        if not validation:
            assert study.training_date_range == study.refit_date_range
            assert study.training_data.exceptAll(study.refit_data).count() == 0
            record = study._record()
            assert "validation_score" not in record["artifacts"]
            assert record["role_paths"]["selection"] == record["role_paths"]["final_training"]
            with pytest.raises(ValueError, match="outer validation is not configured"):
                study.run_forecast_workflow(fit_predict=lambda _: pytest.fail("unexpected fit"),
                                           config=model_profile(tmp_path), fixed_parameters={},
                                           implementation=implementation(), validation_only=True)
            with pytest.raises(ValueError, match="tuning_windows"):
                study.tune(fit_predict=lambda _: pytest.fail("unexpected tuning"),
                           config=model_profile(tmp_path), implementation=implementation())
        if validation:
            result = study.run_forecast_workflow(fit_predict=model, config=model_profile(tmp_path),
                                               fixed_parameters={}, implementation=implementation())
        else:
            from autoforecast import run_study

            result = run_study(study_id, model, registry=registry, model_config=model_profile(tmp_path),
                               fixed_parameters={}, implementation=implementation())
    assert calls == (["validation", "refit"] if validation else ["refit"])
    assert (result.validation_metrics is not None) == validation
    assert result.testing_metrics["mae"] == pytest.approx(8.5)
    if validation:
        assert result.validation_metrics["mae"] == pytest.approx(6.5)
    assert result.summary()["original_series"] == 2
    assert (result.path / "validation").exists() == validation
    reopened = load_forecast_workflow(result.path)
    assert {k: v for k, v in reopened.summary().items() if k != "publication"} == {
        k: v for k, v in result.summary().items() if k != "publication"
    }
    assert reopened.publication["status"] == "not_requested"
    # Recomputing file hashes must not disguise a changed validation contract.
    from autoforecast.study_tuning import HPOError, load_selected_configuration
    from autoforecast.test_study_workflow import _reseal_attempt

    for defect in ("validation", "parameters"):
        changed = tmp_path / defect / result.path.name
        shutil.copytree(result.path, changed)
        payload_path = changed / "result.json"
        payload = json.loads(payload_path.read_text())
        if defect == "validation":
            payload["schema"] = "autoforecast.forecast-workflow.v2"
            payload["validation"] = None if validation else payload["testing"]
            message = "validation absence"
        else:
            payload["parameters"]["model.level"] = 9.0
            message = "configuration"
        payload_path.write_text(json.dumps(payload))
        _reseal_attempt(changed)
        for loader, path in ((load_forecast_workflow, changed),
                             (load_selected_configuration, changed / "metadata/winner-config.json")):
            with pytest.raises(HPOError, match=message):
                loader(path)
    work = json.loads((result.path / "metadata/work-estimate.json").read_text())
    assert work["total_callback_fits"] == (2 if validation else 1)
    from autoforecast.study_results import _publication

    definition, sha = registry.definition(study_id)
    with _publication(definition, sha, result.path, scratch_root=registry.scratch_root) as tables:
        assert {row[3] for row in tables["study_results"]} == (
            {"validation", "test"} if validation else {"test"})
    if not validation:
        from autoforecast.study_results import read_workflow_results

        rows = read_workflow_results(result.path).rows
        assert set(rows["stage"].to_pylist()) == {"test"}
        prediction = result.testing["forecasts"]
        with registry.load_study(study_id) as study:
            external = study.evaluate_forecasts(testing_forecasts=prediction, model_id="external")
            assert "validation" not in external
            assert external["test"]["score"]["mae"] == result.testing_metrics["mae"]
            with pytest.raises(ValueError, match="validation_forecasts"):
                study.evaluate_forecasts(testing_forecasts=prediction,
                                         validation_forecasts=prediction, model_id="external")
        import mlflow

        from autoforecast.run_mlflow import export_forecast_workflow

        uri = "sqlite:///" + str(tmp_path / "tracking.db")
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        experiment = client.create_experiment("test-only", artifact_location=str(tmp_path / "mlruns"))
        links = export_forecast_workflow(result.path, tracking_uri=uri, experiment_id=experiment)
        assert set(links) == {"test"}
        assert export_forecast_workflow(result.path, tracking_uri=uri,
                                        experiment_id=experiment) == links
        assert len(client.search_runs([experiment])) == 1
        _assert_resealed_contracts_rejected(result, definition, sha, registry, tmp_path)
        _assert_forged_admission_rejected(study_id, result, registry, tmp_path)
        with _publication(definition, sha, result.path, scratch_root=registry.scratch_root,
                          mlflow_links=links) as tables:
            assert {row[3] for row in tables["study_results"]} == {"test"}
        for wrong in ({}, {**links, "validation": links["test"]}):
            with pytest.raises(ValueError, match="exactly the configured test stage"):
                with _publication(definition, sha, result.path, scratch_root=registry.scratch_root,
                                  mlflow_links=wrong):
                    pytest.fail("wrong stage links must not reach a destination")
        baseline = registry.root / "registrations" / (study_id + ".json")
        original = baseline.read_bytes()
        envelope = json.loads(original)
        envelope["definition"]["source"]["stages"]["training"]["planned_range"][0] = "2024-01-02"
        from autoforecast.benchmark import digest

        envelope["sha256"] = digest(envelope["definition"])
        baseline.write_text(json.dumps(envelope))
        try:
            with pytest.raises(ValueError, match="stage provenance"):
                registry.load_study(study_id)
        finally:
            baseline.write_bytes(original)
        # Each public registry operation must reject changed frozen data before doing work.
        from types import SimpleNamespace

        manifest = registry.root / "benchmarks" / study_id / "benchmark.json"
        frozen = json.loads(manifest.read_text())
        source = next((manifest.parent / frozen["role_paths"]["final_score"]).rglob("*.parquet"))
        original_data = source.read_bytes()
        source.write_bytes(original_data + b"changed")
        try:
            for operation in (
                lambda: registry.load_study(study_id),
                lambda: run_study(study_id, lambda _: pytest.fail("tampered study must not fit"),
                                  registry=registry, model_config=model_profile(tmp_path),
                                  fixed_parameters={}, implementation=implementation()),
                lambda: registry.publish_results(
                    study_id, result.path,
                    destination=SimpleNamespace(publish_results=lambda *a, **k:
                                                pytest.fail("tampered study must not publish"))),
            ):
                with pytest.raises(ValueError, match="benchmark artifact changed: final_score"):
                    operation()
        finally:
            source.write_bytes(original_data)


def test_no_validation_tuning_uses_only_declared_inner_history(spark, tmp_path):
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=False)
    args["tuning_schedule"] = {"count": 2, "horizon": 1, "stride": 1}
    study_id = register_study(registry=registry, **args)
    calls = []

    def model(request):
        calls.append((request.purpose, request.origin))
        latest = request.training_data.agg(F.max("day")).first()[0]
        assert latest <= dt.date.fromisoformat(request.origin)
        if request.purpose == "selection":
            assert max(request.forecast_dates) <= dt.date(2024, 1, 7)
        return predict(request)

    profile = model_profile(tmp_path, budget=1)
    import yaml

    value = yaml.safe_load(profile.read_text())
    value["selection"] = {"origins": "all"}
    profile.write_text(yaml.safe_dump(value))
    with registry.load_study(study_id) as study:
        assert study.recipe["tuning_windows"] == [
            {"origin": "2024-01-05", "horizon": 1}, {"origin": "2024-01-06", "horizon": 1},
        ]
        run = study.run_forecast_workflow(fit_predict=model, config=profile,
                                         implementation=implementation())
        assert run.validation is None and study.validation_data is None
    assert calls == [("selection", "2024-01-05"), ("selection", "2024-01-06"),
                     ("refit", "2024-01-07")]
    from autoforecast.study_results import _publication

    definition, sha = registry.definition(study_id)
    with _publication(definition, sha, run.path, scratch_root=registry.scratch_root) as tables:
        assert {row[3] for row in tables["study_results"]} == {"tuning", "test"}


def test_direct_invalid_modes_dates_and_hooks_do_not_read_data(spark, tmp_path, monkeypatch):
    from autoforecast.study_registration import _capture_stages

    assert callable(_capture_stages)
    monkeypatch.setattr("autoforecast.study_registration._capture_stages",
                        lambda *a: pytest.fail("invalid settings reached data"))
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=False)
    for changes, message in [
        ({"training_data": None}, "training_data"),
        ({"series_id": []}, "series_id"),
        ({"validation_date_range": ("2024-01-06", "2024-01-07")}, "supply both"),
        ({"validation_data": args["testing_data"]}, "supply both"),
        ({"testing_date_range": ("2024-01-07", "2024-01-09")}, "one period after"),
        ({"testing_date_range": ("2024-01-09", "2024-01-10")}, "one period after"),
        ({"testing_date_range": ("2024-01-09", "2024-01-08")}, "start must not follow"),
        ({"frequency": "unsupported"}, "frequency"),
        ({"metric": "wrmsse"}, "references"),
        ({"actual_missing": ""}, "actual_missing"),
        ({"feature_roles": {"infer": True}}, "feature_roles"),
        ({"feature_roles": []}, "feature_roles"),
        ({"feature_roles": False}, "feature_roles"),
        ({"feature_availability": []}, "feature_availability"),
        ({"feature_availability": False}, "feature_availability"),
    ]:
        with pytest.raises(ValueError, match=message):
            register_study(registry=registry, **{**args, **changes})
    with pytest.raises(ValueError, match="cannot combine"):
        register_study("missing.yaml", registry=registry, **args)
    with pytest.raises(ValueError, match="already be prepared"):
        register_study(registry=registry, prepare=object(), **args)
    assert not registry.root.exists()


@pytest.mark.parametrize("defect", ["empty", "outside", "duplicate", "missing_key", "extra_key"])
def test_direct_stage_checks_reject_wrong_rows(spark, tmp_path, defect):
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=True)
    frame = args["validation_data"]
    if defect == "empty":
        frame = frame.limit(0)
    elif defect == "outside":
        frame = frame.unionByName(args["training_data"].limit(1))
    elif defect == "duplicate":
        frame = frame.unionByName(frame.limit(1))
    elif defect == "missing_key":
        frame = frame.filter("item = 'A'")
    else:
        frame = frame.unionByName(frame.limit(1).withColumn("item", F.lit("extra")))
    args["validation_data"] = frame
    with pytest.raises(ValueError, match="validation_data"):
        register_study(registry=registry, **args)
    assert not list((registry.root / "registrations").glob("*.json"))


@pytest.mark.parametrize("stage,validation", [
    ("training", False), ("training", True), ("testing", False), ("testing", True),
    ("validation", True),
])
def test_empty_supplied_stage_explains_how_to_fix_it(spark, tmp_path, stage, validation):
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=validation)
    args[f"{stage}_data"] = args[f"{stage}_data"].limit(0)
    with pytest.raises(ValueError, match=f"{stage}_data: supplied stage is empty") as caught:
        register_study(registry=registry, **args)
    message = str(caught.value)
    assert f"provide nonempty {stage} data" in message
    if stage == "validation":
        assert "omit both validation_data and validation_date_range" in message
    else:
        assert "omit" not in message
    assert not list((registry.root / "registrations").glob("*.json"))


def test_direct_no_validation_masked_tail_and_references(spark, tmp_path):
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=False)
    for key in ("training_data", "testing_data"):
        args[key] = args[key].withColumn("revenue", F.col("value") * 2)
    args["testing_data"] = args["testing_data"].filter(F.col("day") == dt.date(2024, 1, 8))
    args.update(metric="wrmsse", actual_missing="mask", references={
        "weight_col": "revenue", "missing_history": "na", "duplicate_policy": "error",
        "scale_start": "history_start", "weight_window_days": 3,
    })
    study_id = register_study(registry=registry, **args)
    with registry.load_study(study_id) as study:
        assert study.recipe["final_test_horizon"] == 2
        refs = study.scoring_references.collect()
        assert {row.boundary.split("/")[0] for row in refs} == {"test"}
        run = study.run_forecast_workflow(fit_predict=predict, config=model_profile(tmp_path),
                                         fixed_parameters={}, implementation=implementation())
        assert run.validation is None
        assert run.coverage["test"]["originally_missing"] == 2
        assert run.coverage["test"]["scored"] == 2


def test_direct_no_validation_campaign_reports_without_choosing_winner(spark, tmp_path):
    from autoforecast.study_campaign import StudyModel

    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=False)
    args["tuning_schedule"] = {"count": 1, "horizon": 1, "stride": 1}
    study_id = register_study(registry=registry, **args)
    import yaml

    models = []
    for name, parameters in (("fixed", {}), ("tuned", None)):
        folder = tmp_path / name
        folder.mkdir()
        profile = model_profile(folder, budget=1)
        payload = yaml.safe_load(profile.read_text())
        payload["model"]["id"] = name
        profile.write_text(yaml.safe_dump(payload))
        models.append(StudyModel(entrypoint="autoforecast.test_study_tuning:predict",
                                 config=profile, fixed_parameters=parameters,
                                 implementation=implementation()))
    with registry.load_study(study_id) as study:
        with pytest.raises(ValueError, match="requires outer validation"):
            study.run_campaign(models=models, final_evaluation="validation_winner")
        options = dict(models=models, final_evaluation="all_models", execution="sequential",
                       output_dir=tmp_path / "campaign")
        result = study.run_campaign(**options)
        replayed = study.run_campaign(**options)
    assert set(result["phases"]) == {"final"}
    assert result["results"]["selected_model"] is None
    assert len(result["results"]["results"]) == 2
    assert replayed["results"] == result["results"]
    from autoforecast.study_workflow import load_forecast_workflow

    for record in result["results"]["results"].values():
        workflow = load_forecast_workflow(record["workflow_path"])
        assert workflow.validation is None
        assert (workflow.tuning_path is not None) == (record["model_id"] == "tuned")
        estimate = json.loads((workflow.path / "metadata/work-estimate.json").read_text())
        assert estimate["total_callback_fits"] == (2 if record["model_id"] == "tuned" else 1)
        assert not (workflow.path / "validation").exists()
        assert pathlib.Path(record["workflow_path"]).is_dir()


def test_supplied_lazy_frames_are_evaluated_once(spark, tmp_path):
    calls = spark.sparkContext.accumulator(0)

    @F.udf("double")
    def counted(value):
        calls.add(1)
        return value

    args = stages(spark, validation=False)
    for name in ("training_data", "testing_data"):
        args[name] = args[name].withColumn("value", counted.asNondeterministic()("value"))
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    study_id = register_study(registry=registry, **args)
    assert calls.value == 17
    with registry.load_study(study_id) as study:
        assert study.training_data.count() == 13
        assert study.testing_data.count() == 4
    assert calls.value == 17


def test_configured_no_validation_has_the_same_scientific_definition(spark, tmp_path):
    import yaml

    from autoforecast.test_study_registration import configuration

    args = stages(spark, validation=False)
    registry = StudyRegistry(tmp_path / "direct", spark=spark)
    direct_id = register_study(registry=registry, **args)
    path = configuration(tmp_path / "configured", actual_missing="error")
    value = yaml.safe_load(path.read_text())
    value["dataset"]["dataset"]["id"] = "supplied-stages"
    value["recipe"].update(schema_version=3, validation_origins=[], validation_horizon=None,
                            history_missing="preserve")
    value["recipe"].pop("tuning_windows")
    path.write_text(yaml.safe_dump(value))
    args["training_data"].unionByName(args["testing_data"]).write.parquet(str(tmp_path / "whole"))
    value["source"]["path"] = str(tmp_path / "whole")
    path.write_text(yaml.safe_dump(value))
    configured = StudyRegistry.from_config(path, spark=spark)
    configured_id = register_study(path, registry=configured)
    with registry.load_study(direct_id) as direct, configured.load_study(configured_id) as via_config:
        assert direct.recipe == via_config.recipe
        assert direct.training_data.exceptAll(via_config.training_data).count() == 0
        assert direct.testing_data.exceptAll(via_config.testing_data).count() == 0
        assert direct.validation_data is None and via_config.validation_data is None
        assert direct_id != configured_id  # Source provenance is deliberately different.


@pytest.mark.parametrize("policy", ["error", "zero", "mask"])
def test_planned_test_range_keeps_missing_tail(spark, tmp_path, policy):
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=False)
    args["testing_data"] = spark.createDataFrame([
        (key, dt.date(2024, 1, day), float(day))
        for key in ("A", "B") for day in range(8, 18)
    ], "item string, day date, value double")
    args.update(testing_date_range=("2024-01-08", "2024-01-21"), actual_missing=policy)
    if policy == "error":
        with pytest.raises(ValueError, match="missing"):
            register_study(registry=registry, **args)
        assert not list((registry.root / "registrations").glob("*.json"))
        return
    study_id = register_study(registry=registry, **args)
    with registry.load_study(study_id) as study:
        assert study.recipe["final_test_horizon"] == 14
        calls = []

        def model(request):
            calls.append(request.forecast_dates)
            assert len(request.forecast_dates) == 14
            return predict(request)

        result = study.run_forecast_workflow(fit_predict=model, config=model_profile(tmp_path),
                                            fixed_parameters={}, implementation=implementation())
    assert len(calls) == 1 and calls[0][-1] == dt.date(2024, 1, 21)
    assert result.coverage["test"]["originally_missing"] == 8
    assert result.coverage["test"]["scored"] == (28 if policy == "zero" else 20)


def test_masked_validation_stays_missing_when_refit_fills_history(spark, tmp_path):
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark)
    args["validation_data"] = args["validation_data"].filter(
        ~((F.col("item") == "A") & (F.col("day") == dt.date(2024, 1, 6))))
    args.update(actual_missing="mask", history_missing="zero")
    study_id = register_study(registry=registry, **args)
    with registry.load_study(study_id) as study:
        a_id = study.series_mapping.filter(F.col("item") == "A").first()[study.panel.series_col]
        filled = study.refit_data.filter(
            (F.col(study.panel.series_col) == a_id) & (F.col("day") == dt.date(2024, 1, 6)))
        assert filled.first()["value"] == 0.0
        result = study.run_forecast_workflow(fit_predict=predict, config=model_profile(tmp_path),
                                            fixed_parameters={}, implementation=implementation())
    assert result.coverage["validation"]["originally_missing"] == 1
    assert result.coverage["validation"]["scored"] == 3
    assert result.coverage["test"]["originally_missing"] == 0


@pytest.mark.parametrize("stage", ["validation", "testing"])
@pytest.mark.parametrize("policy", ["error", "zero", "mask"])
def test_whole_series_missingness_uses_the_same_policy_in_both_routes(spark, tmp_path, stage, policy):
    import yaml

    from autoforecast.test_study_registration import configuration

    args = stages(spark)
    args[stage + "_data"] = args[stage + "_data"].filter(F.col("item") == "A")
    args["actual_missing"] = policy
    registry = StudyRegistry(tmp_path / "direct", spark=spark)
    path = configuration(tmp_path / "configured", actual_missing=policy)
    value = yaml.safe_load(path.read_text())
    value["recipe"].pop("tuning_windows")
    value["recipe"]["history_missing"] = "preserve"
    complete = args["training_data"].unionByName(args["validation_data"])
    complete = complete.unionByName(args["testing_data"])
    complete.write.parquet(str(tmp_path / "whole"))
    value["source"]["path"] = str(tmp_path / "whole")
    path.write_text(yaml.safe_dump(value))
    configured = StudyRegistry.from_config(path, spark=spark)
    if policy == "error":
        with pytest.raises(ValueError, match="missing|population"):
            register_study(registry=registry, **args)
        with pytest.raises(ValueError, match="missing|population"):
            register_study(path, registry=configured)
        return
    direct_id = register_study(registry=registry, **args)
    configured_id = register_study(path, registry=configured)
    with registry.load_study(direct_id) as direct, configured.load_study(configured_id) as via_config:
        assert direct.n_series == via_config.n_series == 2
        direct_actuals = getattr(direct, stage + "_data")
        configured_actuals = getattr(via_config, stage + "_data")
        assert direct_actuals.exceptAll(configured_actuals).count() == 0
        assert direct_actuals.filter(F.col("value").isNull()).count() == 2
        run = direct.run_forecast_workflow(fit_predict=predict, config=model_profile(tmp_path),
                                           fixed_parameters={}, implementation=implementation())
    score_stage = "validation" if stage == "validation" else "test"
    assert run.coverage[score_stage]["originally_missing"] == 2
    assert run.coverage[score_stage]["scored"] == (4 if policy == "zero" else 2)
    metric = run.validation_metrics if stage == "validation" else run.testing_metrics
    expected = 6.5 if stage == "validation" else 8.5
    assert metric["mae"] == pytest.approx(expected / 2 if policy == "zero" else expected)


def test_direct_plan_is_logged_before_capture_and_failures_keep_evidence(spark, tmp_path, monkeypatch):
    from autoforecast import study_registration

    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark)
    args["testing_data"] = args["testing_data"].unionByName(args["training_data"].limit(1))
    capture = study_registration._capture_stages

    def checked_capture(*values):
        audit = next((registry.root / "checks").glob("*/audit.jsonl"))
        events = [json.loads(line) for line in audit.read_text().splitlines()]
        summary = next(event for event in events if event["stage"] == "Supplied stage plan")
        assert set(summary["stages"]) == {"training", "validation", "testing"}
        assert summary["stages"]["testing"] == {
            "kind": "dataframe", "planned_range": ["2024-01-08", "2024-01-09"],
        }
        assert summary["date_column"] == "day" and summary["target_column"] == "value"
        assert not any(event["stage"] == "Capture supplied stage" for event in events)
        return capture(*values)

    monkeypatch.setattr(study_registration, "_capture_stages", checked_capture)
    with pytest.raises(ValueError, match="testing_data.*outside") as caught:
        register_study(registry=registry, log_level="CRITICAL", **args)
    notes = caught.value.__notes__
    assert len(notes) == 1 and "Registration attempt evidence:" in notes[0]
    from pathlib import Path

    evidence = Path(notes[0].split(": ", 1)[1])
    assert json.loads((evidence / "attempt.json").read_text())["status"] == "failed"
    events = [json.loads(line) for line in (evidence / "audit.jsonl").read_text().splitlines()]
    rejected = next(event for event in events if event["stage"] == "Stage input rejected")
    assert rejected["window"] == "testing" and rejected["rows_outside"] == 1


def test_direct_identity_binds_roles_ranges_cohort_policy_and_owned_source(spark, tmp_path):
    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    args = stages(spark, validation=False)
    source = tmp_path / "input"
    args["training_data"].write.parquet(str(source))
    args["training_data"] = spark.read.parquet(str(source))
    baseline = register_study(registry=registry, **args)
    identities = {baseline}
    # Identical rows with a different stage assignment describe a different study.
    identities.add(register_study(registry=registry, **stages(spark, validation=True)))
    identities.add(register_study(registry=registry, **{**args, "actual_missing": "zero"}))
    changed_range = {**args, "training_date_range": ("2024-01-02", "2024-01-07"),
                     "training_data": args["training_data"].filter(F.col("day") >= "2024-01-02")}
    identities.add(register_study(registry=registry, **changed_range))
    cohort = {**args, **{name: args[name].filter(F.col("item") == "A")
                        for name in ("training_data", "testing_data")}}
    identities.add(register_study(registry=registry, **cohort))
    assert len(identities) == 5
    # Replacing the caller's source cannot change an already captured study.
    stages(spark, validation=False)["training_data"].withColumn("value", F.lit(999.0)).write.mode(
        "overwrite").parquet(str(source))
    with registry.load_study(baseline) as study:
        assert study.training_data.agg(F.max("value")).first()[0] == 7.0
        assert study.n_series == 2


def _assert_resealed_contracts_rejected(result, definition, sha, registry, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.study_results import _publication, read_workflow_results
    from autoforecast.study_tuning import load_selected_configuration
    from autoforecast.study_workflow import load_forecast_workflow
    from autoforecast.test_study_workflow import _reseal_attempt
    from provenance.lineage import directory_digest

    manifest = pathlib.Path(result.provenance["study"]["path"])
    frozen = json.loads(manifest.read_text())
    for defect in ("workflow-v1", "validation-v1", "actuals", "grid", "coverage", "policy", "origin"):
        changed = tmp_path / ("forged-" + defect) / result.path.name
        shutil.copytree(result.path, changed)
        path = changed / "result.json"
        record = json.loads(path.read_text())
        window = record["testing"]
        contract = window["evaluation_contract"]
        if defect.endswith("v1"):
            shutil.copytree(changed / "test", changed / "validation")
            fake = copy.deepcopy(window)
            fake["boundary"] = fake["boundary"].replace("test/", "validation/", 1)
            fake["evaluation_contract"]["boundary"] = fake["boundary"]
            fake["forecast"]["path"] = "validation/forecasts"
            fake["model_path"] = "validation/model"
            record["validation"] = fake
            record["schema"] = "autoforecast.forecast-workflow.v1"
            if defect == "validation-v1":
                record["schema"] = "autoforecast.forecast-validation.v1"
                record.pop("testing")
                for name in ("completion.json", "metadata/run.json"):
                    metadata_path = changed / name
                    metadata = json.loads(metadata_path.read_text())
                    metadata["task"] = "forecast-validation"
                    metadata_path.write_text(json.dumps(metadata))
            (changed / "validation/result.json").write_text(json.dumps(fake))
        elif defect == "actuals":
            contract["actuals"] = {"role": "selection", "sha256": frozen["artifacts"]["selection"],
                                   "path": str(manifest.parent / frozen["role_paths"]["selection"])}
        elif defect == "grid":
            contract["grid"]["path"] = str(manifest.parent / frozen["role_paths"]["selection"])
        elif defect == "coverage":
            contract["coverage"]["counts"]["expected"] = 400
        elif defect == "policy":
            contract["metric_policy"]["actual_missing"] = "zero"
        else:
            window["boundary"] = contract["boundary"] = "test/2024-01-06/2/D"
            window["forecast"]["origin"] = "2024-01-06"
            forecasts = changed / "test/forecasts/part.parquet"
            table = pq.read_table(forecasts)
            index = table.schema.get_field_index("date")
            shifted = [point.as_py() - dt.timedelta(days=1) for point in table.column("date")]
            table = table.set_column(index, "date", pa.array(shifted, type=table.schema[index].type))
            pq.write_table(table, forecasts)
            window["forecast_sha256"] = directory_digest(forecasts.parent, suffixes=(".parquet",))
        (changed / "test/result.json").write_text(json.dumps(window))
        path.write_text(json.dumps(record))
        _reseal_attempt(changed)
        for loader, target in ((load_forecast_workflow, changed),
                               (load_selected_configuration, changed / "metadata/winner-config.json"),
                               (read_workflow_results, changed)):
            with pytest.raises(ValueError, match="validation absence|stage configuration"):
                loader(target)
        with pytest.raises(ValueError, match="validation absence|stage configuration"):
            with _publication(definition, sha, changed, scratch_root=registry.scratch_root):
                pytest.fail("forged evidence must not reach a publication destination")


def _assert_forged_admission_rejected(study_id, result, registry, tmp_path):
    from types import SimpleNamespace

    from autoforecast import run_study
    from autoforecast.benchmark import digest
    from autoforecast.study_registration import _artifact_record, _bytes

    path = registry.root / "registrations" / (study_id + ".json")
    original = path.read_bytes()
    envelope = json.loads(original)
    frozen_root = registry.root / "benchmarks" / study_id
    frozen = json.loads((frozen_root / "benchmark.json").read_text())
    evidence_root = registry.root / "admissions" / study_id
    evidence_root.mkdir(parents=True)
    names = ("starting_keys", "retained_keys", "exclusions", "final_keys", "native_exclusions")
    for name in names:
        shutil.copytree(frozen_root / frozen["role_paths"]["selection"], evidence_root / name)
    evidence = {"schema": "autoforecast.admission.v1", "study_id": study_id,
                "identity": envelope["definition"]["source"],
                "artifacts": {name: _artifact_record(evidence_root / name) for name in names}}
    raw = _bytes(evidence)
    evidence_path = evidence_root / "admission.json"
    evidence_path.write_bytes(raw)
    altered = copy.deepcopy(envelope)
    altered["definition"]["source"]["stages"]["training"]["planned_range"][0] = "2024-01-02"
    altered["definition"]["admission_evidence"] = {
        "path": str(evidence_path.relative_to(registry.root)), "sha256": digest(evidence),
        "physical_sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        for schema, kind in (("autoforecast.registration.v2", "supplied_stages"),
                             ("autoforecast.registration.v1", "supplied_stages"),
                             ("autoforecast.registration.v1", "parquet")):
            altered["definition"]["schema"] = schema
            altered["definition"]["source"]["kind"] = kind
            altered["sha256"] = digest(altered["definition"])
            path.write_bytes(_bytes(altered))
            for operation in (
                lambda: registry.definition(study_id),
                lambda: registry.load_study(study_id),
                lambda: run_study(study_id, lambda _: pytest.fail("forged source must not fit"),
                                  registry=registry, model_config=model_profile(tmp_path),
                                  fixed_parameters={}, implementation=implementation()),
                lambda: registry.publish_results(
                    study_id, result.path,
                    destination=SimpleNamespace(publish_results=lambda *a, **k:
                                                pytest.fail("forged source must not publish"))),
            ):
                with pytest.raises(ValueError, match="supplied stage provenance"):
                    operation()
    finally:
        path.write_bytes(original)


def test_direct_derived_cohort_keeps_its_source_binding(spark, tmp_path):
    import yaml

    from autoforecast import run_study
    from autoforecast.study_results import _publication

    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    parent = register_study(registry=registry, **stages(spark, validation=False))
    profile = model_profile(tmp_path)
    settings = yaml.safe_load(profile.read_text())
    settings.update(series_policy="exclude_unavailable", minimum_history=7)
    profile.write_text(yaml.safe_dump(settings))
    result = run_study(parent, predict, registry=registry, model_config=profile,
                       fixed_parameters={}, implementation=implementation())
    assert result.study_id != parent
    assert result.parent_study_id == parent
    definition, sha = registry.definition(result.study_id)
    assert definition["source"] == registry.definition(parent)[0]["source"]
    with registry.load_study(result.study_id) as study:
        assert study.n_series == 1
    with _publication(definition, sha, result.path, scratch_root=registry.scratch_root) as tables:
        assert {row[3] for row in tables["study_results"]} == {"test"}
    replay = run_study(result.study_id, predict, registry=registry, model_config=profile,
                       fixed_parameters={}, implementation=implementation())
    assert replay.study_id == result.study_id
    assert replay.testing_metrics["mae"] == pytest.approx(8.5)


def test_final_only_study_rescores_retained_forecasts_and_rejects_native_selection(
    spark, tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast import benchmark, forecast_campaign, rescore
    from autoforecast.panel import Forecast
    from provenance.lineage import directory_digest

    registry = StudyRegistry(tmp_path / "registry", spark=spark)
    study_id = register_study(registry=registry, **stages(spark, validation=False))
    with registry.load_study(study_id) as study:
        study_path, target_estimand = study._path, study.panel.target_estimand
    fixed = benchmark.evaluation_contract(
        study_path, stage="test", origin="2024-01-07", horizon=2, metric="mae",
    )
    rows = pq.read_table(fixed["grid"]["path"]).to_pylist()
    predictions = tmp_path / "predictions.parquet"
    # Keep the retained predictions unchanged while evaluating against the new study schema.
    pq.write_table(pa.Table.from_pylist([{**row, "yhat": 0.0} for row in rows]), predictions)
    forecast = Forecast("test", str(predictions), "2024-01-07", 2, 2,
                        {"emitted": len(rows)}, target_estimand=target_estimand)
    forecast_record = tmp_path / "forecast.json"
    forecast_record.write_text(json.dumps(forecast.as_record()))
    digest = directory_digest(predictions, suffixes=(".parquet",))
    original = SimpleNamespace(
        metadata={"resolved_sha256": "a" * 64, "study_id": study_id},
        runs=({"status": "completed", "stage": "test", "adapter": "test", "model_id": "test",
               "variant_id": "default", "run_id": "retained-run", "origin": forecast.origin,
               "horizon": forecast.horizon, "forecast_record_path": str(forecast_record),
               "forecast_record_sha256": hashlib.sha256(forecast_record.read_bytes()).hexdigest(),
               "forecast_artifact_sha256": digest},),
    )
    # Campaign journal loading has separate coverage; real forecast checks and scoring run here.
    monkeypatch.setattr(rescore, "read_results", lambda _: original)
    result = rescore.rescore_campaign(tmp_path / "campaign", study_path, tmp_path / "rescore", spark)
    record = json.loads(result.read_text())
    assert record["scoring_study_id"] == study_id and len(record["scores"]) == 1
    score = record["scores"][0]
    comparison = json.loads(pathlib.Path(score["comparison"]).read_text())
    assert comparison["candidates"][0]["score"]["mae"] == pytest.approx(8.5)
    assert score["stage"] == "test" and score["forecast_artifact_sha256"] == digest
    assert directory_digest(predictions, suffixes=(".parquet",)) == digest

    with pytest.raises(benchmark.BenchmarkError, match="outer validation is not configured"):
        benchmark.evaluation_contract(
            study_path, stage="validation", origin="2024-01-05", horizon=2, metric="mae",
        )
    # A wrong date in a stage that exists remains a different input error.
    with pytest.raises(benchmark.BenchmarkError, match="evaluation boundary absent"):
        benchmark.evaluation_contract(
            study_path, stage="test", origin="2024-01-06", horizon=2, metric="mae",
        )

    test_run = original.runs[0]
    validation_run = {**test_run, "stage": "validation", "origin": "2024-01-05"}
    for index, runs in enumerate(((validation_run,), (validation_run, test_run),
                                  (test_run, validation_run))):
        original.runs = runs
        unavailable = tmp_path / f"mixed-{index}" / "rescore"
        with monkeypatch.context() as patch:
            # Reject the unavailable stage before reading or scoring any retained forecast.
            patch.setattr(rescore, "_verified_forecast", lambda *a, **k:
                          pytest.fail("unavailable validation must fail before forecast work"))
            with pytest.raises(ValueError, match="rescoring validation forecasts requires outer validation"):
                rescore.rescore_campaign(tmp_path / "campaign", study_path, unavailable, spark)
        assert not unavailable.parent.exists()
    assert directory_digest(predictions, suffixes=(".parquet",)) == digest

    # The native winner-selection compiler must reject this study before creating any stages.
    monkeypatch.setattr(forecast_campaign, "_root", lambda: tmp_path)
    monkeypatch.setattr(forecast_campaign, "load_forecast_campaign_request", lambda _:
                        SimpleNamespace(contract=study_path.parent / "dataset.yaml"))
    with pytest.raises(forecast_campaign.ForecastCampaignError, match="require outer validation"):
        forecast_campaign.compile_forecast_campaign(
            tmp_path / "request.yaml", benchmark_path=study_path,
            selection_data_path=tmp_path / "selection", final_training_data_path=tmp_path / "training",
            final_score_data_path=tmp_path / "testing", compiled_root=tmp_path / "compiled",
        )
    assert not (tmp_path / "compiled").exists()
