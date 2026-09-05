"""Public workflows keep negative model output and bind one saved-source baseline."""

import hashlib
import importlib.util
import json
import os
import sys

import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import register_study, run_study
from autoforecast.study_registration import StudyRegistry
from autoforecast.study_tuning import load_tuning_result
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_registration import configuration
from autoforecast.test_study_tuning import config as model_profile
from autoforecast.test_study_tuning import spark as _spark
from provenance.lineage import directory_digest


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def application(root, monkeypatch):
    source = root / "model_code"
    source.mkdir()
    (source / "helper.py").write_text("VALUE = -2.0\n")
    module_path = source / "synthetic_saved_model.py"
    module_path.write_text('''import json
import pyarrow as pa
import pyarrow.parquet as pq
from autoforecast.study_tuning import FitPredictResult
from autoforecast.panel import Forecast
calls = []
mutate = None

def fit_predict(request):
    calls.append(request.purpose)
    ids = [row[request.panel.series_col] for row in request.series_mapping.collect()]
    forecasts = request.output_dir / "forecasts"
    forecasts.mkdir()
    pq.write_table(pa.Table.from_pylist([
        {"series_id": sid, "date": day, "yhat": -2.0}
        for sid in ids for day in request.forecast_dates
    ]), forecasts / "part.parquet")
    model = request.output_dir / "model"
    model.mkdir()
    (model / "parameters.json").write_text(json.dumps(request.model_parameters))
    if mutate is not None:
        mutate.write_text("VALUE = -3.0")
    return FitPredictResult(Forecast(
        adapter=request.model_id, path=str(forecasts), origin=request.origin,
        horizon=len(request.forecast_dates), n_series=len(ids),
        disposition={"emitted": len(ids) * len(request.forecast_dates)},
        target_estimand=request.panel.target_estimand, frequency=request.panel.frequency,
    ), "synthetic-json-v1")

def alternate(request):
    return fit_predict(request)
''')
    spec = importlib.util.spec_from_file_location("synthetic_saved_model", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, source


@pytest.mark.parametrize("search", [False, True])
def test_registered_negative_workflow_captures_source_once_and_reloads(spark, tmp_path,
                                                                    monkeypatch, search):
    module, source = application(tmp_path, monkeypatch)
    path = configuration(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["runtime"] = {"implementation": {"source_roots": {"model": "model_code"}}}
    settings["recipe"]["tuning_windows"].append({"origin": "2024-01-04", "horizon": 1})
    path.write_text(yaml.safe_dump(settings))
    json_path = tmp_path / "study.json"
    json_path.write_text(json.dumps(settings))
    registry = StudyRegistry.from_config(path, spark=spark)
    assert StudyRegistry.from_config(json_path, spark=spark).runtime == registry.runtime
    study_id = register_study(path, registry=registry)
    before = registry.definition(study_id)[1]
    profile = model_profile(tmp_path, budget=1)
    model_settings = yaml.safe_load(profile.read_text())
    model_settings["selection"] = {"origins": "all"}
    profile.write_text(yaml.safe_dump(model_settings))
    from autoforecast.study_campaign import StudyModel

    model = StudyModel("synthetic_saved_model:fit_predict", config=profile) if search else module.fit_predict
    result = run_study(study_id, model, registry=registry,
                       model_config=None if search else profile,
                       fixed_parameters=None if search else {}, tracking="none", publish=False)
    assert module.calls == (["selection", "selection"] if search else []) + ["validation", "refit"]
    expected = result.provenance["implementation"]["source_sha256"]
    assert result.provenance["implementation_evidence"]["manifest_sha256"] == expected
    assert result.validation_metrics["mae"] == pytest.approx(26 / 3)
    assert result.testing_metrics["mae"] == pytest.approx(32 / 3)
    for stage in (result.validation, result.testing):
        forecast = stage["forecasts"]
        assert set(pq.read_table(forecast.path)["yhat"].to_pylist()) == {-2.0}
        assert directory_digest(forecast.path, suffixes=(".parquet",)) == stage["forecast_sha256"]
    if search:
        tuning = load_tuning_result(result.tuning_path)
        assert tuning.provenance["implementation"]["source_sha256"] == expected
        assert tuning.provenance["implementation_evidence"]["manifest_sha256"] == expected
    with registry.load_study(study_id) as study:
        with pytest.raises(ValueError, match="different model implementation"):
            study.run_forecast_workflow(fit_predict=module.alternate,
                                       selected_config=result.selected_config)
        if search:
            with pytest.raises(ValueError, match="implementation"):
                study.tune(fit_predict=module.alternate, config=profile,
                           reuse_tuning=result.tuning_path)
    (source / "helper.py").write_text("VALUE = -3.0\n")
    assert registry.definition(study_id)[1] == before
    assert load_forecast_workflow(result.path).validation_metrics == result.validation_metrics
    with registry.load_study(study_id) as study:
        if not search:
            evaluated = study.evaluate_forecasts(
                validation_forecasts=result.validation["forecasts"],
                testing_forecasts=result.testing["forecasts"],
                model_id=result.validation["forecasts"].adapter,
                output_dir=tmp_path / "supplied-evaluation",
            )
            assert evaluated["validation"]["score"]["mae"] == result.validation_metrics["mae"]
        with pytest.raises(ValueError, match="different model implementation"):
            study.run_forecast_workflow(fit_predict=module.fit_predict,
                                       selected_config=result.selected_config)
    retained = result.path / "metadata/implementation/source/model/helper.py"
    retained.write_text("tampered")
    with pytest.raises((ValueError, RuntimeError)):
        load_forecast_workflow(result.path)


def test_source_change_inside_callback_refuses_completed_stage(spark, tmp_path, monkeypatch):
    module, source = application(tmp_path, monkeypatch)
    path = configuration(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["runtime"] = {"implementation": {"source_roots": {"model": "model_code"}}}
    path.write_text(yaml.safe_dump(settings))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    module.mutate = source / "helper.py"
    with pytest.raises(ValueError, match="source changed"):
        run_study(study_id, module.fit_predict, registry=registry,
                  model_config=model_profile(tmp_path), fixed_parameters={})
    assert module.calls == ["validation"]
    assert not list((registry.root / "runs").glob("*/result.json"))


def test_campaign_workers_retain_automatic_source_and_reject_changed_helpers(
    spark, tmp_path, monkeypatch,
):
    from autoforecast.study_campaign import StudyModel, _model_job

    module, source = application(tmp_path, monkeypatch)
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setenv("PYTHONPATH", str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    path = configuration(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["runtime"] = {"implementation": {"source_roots": {"model": "model_code"}}}
    path.write_text(yaml.safe_dump(settings))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = model_profile(tmp_path, budget=1)
    import autoforecast.implementation as implementation

    original_dependencies = implementation.installed_dependencies
    original_revision = implementation.library_revision
    inventories = iter([(("synthetic-package", "1"),), (("synthetic-package", "1"),),
                        (("synthetic-package", "2"),)])
    revisions = iter([{"revision": "1"}, {"revision": "1"}, {"revision": "2"}])
    monkeypatch.setattr(implementation, "installed_dependencies", lambda: next(inventories))
    monkeypatch.setattr(implementation, "library_revision", lambda: next(revisions))
    second_profile = tmp_path / "second.yaml"
    second_settings = yaml.safe_load(profile.read_text())
    second_settings["model"]["id"] = "second-candidate"
    second_profile.write_text(yaml.safe_dump(second_settings))
    second = StudyModel("synthetic_saved_model:fit_predict", config=second_profile,
                        fixed_parameters={}, memory_gib=2.0)
    declaration = StudyModel("synthetic_saved_model:fit_predict", config=profile,
                             fixed_parameters={}, memory_gib=2.0)
    original_publish = implementation.publish_bytes_set

    def interrupted_publish(root, payloads):
        first_path = next(iter(payloads))
        original_publish(root, {first_path: payloads[first_path]})
        raise OSError("interrupted source publication")

    with registry.load_study(study_id) as study:
        monkeypatch.setattr(implementation, "publish_bytes_set", interrupted_publish)
        with pytest.raises(OSError, match="interrupted source publication"):
            study.run_campaign(models=[declaration, second], execution="sequential",
                               output_dir=tmp_path / "campaign", timeout_seconds=300)
        monkeypatch.setattr(implementation, "publish_bytes_set", original_publish)
        result = study.run_campaign(models=[declaration, second], execution="sequential",
                                    output_dir=tmp_path / "campaign", timeout_seconds=300)
        assert result["state"]["status"] == "completed", result
        workflow = load_forecast_workflow(result["results"]["results"]["0"]["workflow_path"])
        assert workflow.provenance["implementation_evidence"]["mode"] == "saved_source"
        assert workflow.validation_metrics["mae"] == pytest.approx(26 / 3)
        assert workflow.testing_metrics["mae"] == pytest.approx(32 / 3)
    second_workflow = load_forecast_workflow(result["results"]["results"]["1"]["workflow_path"])
    evidence = json.loads((second_workflow.path / "metadata/implementation/evidence.json").read_bytes())
    assert evidence["environment"]["installed_packages"] == [["synthetic-package", "2"]]
    assert evidence["library"] == {"revision": "2"}
    assert "worker_environment" in evidence and "worker_library" in evidence
    request_path = next((tmp_path / "campaign").glob("request-*/request.json"))
    request = json.loads(request_path.read_bytes())
    assert (request["models"][0]["implementation_source"]["path"]
            != request["models"][1]["implementation_source"]["path"])
    expected = request["models"][0]["implementation"]["source_sha256"]
    assert workflow.provenance["implementation"]["source_sha256"] == expected
    monkeypatch.setattr(implementation, "installed_dependencies", original_dependencies)
    monkeypatch.setattr(implementation, "library_revision", original_revision)
    (source / "helper.py").write_text("VALUE = -9.0\n")
    with pytest.raises(ValueError, match="source changed"):
        _model_job(str(request_path), hashlib.sha256(request_path.read_bytes()).hexdigest(),
                   0, "validation", result["campaign_path"], None)
