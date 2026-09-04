"""The outputs map shows where a study's inputs and a workflow's outputs are, from saved records."""

import json
import logging
import pathlib
from types import SimpleNamespace

import pytest

from autoforecast import outputs as outputs_module
from autoforecast.outputs import OutputItem, OutputsMap, study_outputs, workflow_outputs
from autoforecast.panel import Forecast
from autoforecast.study_workflow import ForecastWorkflowResult

STUDY_ID = "a" * 64


def _registry(root: pathlib.Path, *, store_tables=None):
    from autoforecast.study_registration import StudyRegistry

    definition = {
        "source": {
            "sql": "select ...",
            "tables": {"sales": {"table": "cat.gold.sales", "version": 1036},
                       "shops": {"table": "cat.silver.shops", "version": 300}},
            "rows": 1079823, "starting_series": 1792, "eligible_series": 1180,
        },
        "effective_series": 1153,
    }
    (root / "registrations").mkdir(parents=True)
    (root / "registrations" / f"{STUDY_ID}.json").write_text(
        json.dumps({"definition": definition, "sha256": "b" * 64}))
    registry = SimpleNamespace(root=root, spark=None, store=None)
    registry._registration_path = lambda study_id: StudyRegistry._registration_path(
        registry, study_id)
    if store_tables:
        registry.store = SimpleNamespace(tables=store_tables)
    return registry


def _evidence(root: pathlib.Path, stage: str) -> dict:
    return {
        "forecasts": Forecast("custom", str(root / stage / "forecasts"), "2026-10-05", 14, 1153,
                              {"emitted": 14 * 1153}, target_estimand="observed_nonnegative"),
        "model_artifacts": root / stage / "model", "model_sha256": "c" * 64,
        "model_format": "series-bundle-v3",
        "score": {"reported_metrics": ["mae", "wape"], "mae": 1.25, "wape": 0.3125},
    }


def _result(root: pathlib.Path, **changes) -> ForecastWorkflowResult:
    values = {
        "path": root / "runs" / "workflow-1", "status": "completed",
        "parameters": {"model.changepoint_prior_scale": 0.05},
        "validation": _evidence(root, "validation"), "testing": _evidence(root, "test"),
        "provenance": {"study": {"path": str(root / "benchmarks" / STUDY_ID / "benchmark.json"),
                                 "sha256": "d" * 64}},
        "tuning_path": root / "runs" / "hpo" / "studies" / "tune-1",
        "selected_config": root / "runs" / "workflow-1" / "metadata" / "winner-config.json",
        "search_config": root / "runs" / "workflow-1" / "metadata" / "search-config.yaml",
        "tracking": {"validation": {"experiment_id": "7", "run_id": "r1", "url": None}},
    }
    return ForecastWorkflowResult(**{**values, **changes})


def test_map_sections_render_as_text_html_frame_and_dict():
    item = OutputItem("data", "Read", "sales", "cat.gold.sales <v1>", "Daily sales", "https://x/y")
    other = OutputItem("training", "Test", "Forecasts", "/Volumes/c/s/v/f", "14 days")
    outputs = OutputsMap((("data", "Study ID", "abc"),), (item, other))
    assert outputs.select("data").items == (item,)
    assert outputs.select("training").facts == ()
    with pytest.raises(ValueError, match="section"):
        outputs.select("models")
    text = str(outputs)
    assert "== Study data ==" in text and "Study ID: abc" in text and "https://x/y" in text
    page = outputs._repr_html_()
    # Locations and descriptions are escaped; only real links become anchors.
    assert "&lt;v1&gt;" in page and '<a href="https://x/y"' in page and page.count("<a ") == 1
    # Notebooks, including Databricks display(), render the HTML from the mime bundle.
    assert outputs._repr_mimebundle_() == {"text/plain": text, "text/html": page}
    assert list(outputs.to_frame()["name"]) == ["sales", "Forecasts"]
    assert outputs.to_dict()["facts"] == [{"section": "data", "label": "Study ID", "value": "abc"}]
    assert str(OutputsMap()) == "No study data or training results yet."


def test_study_map_comes_from_the_saved_registration(tmp_path):
    registry = _registry(tmp_path, store_tables={
        "studies": "cat.me.af_studies", "study_data_issues": "cat.me.af_study_data_issues"})
    outputs = study_outputs(registry, STUDY_ID)
    facts = dict((label, value) for _, label, value in outputs.facts)
    assert facts["Study ID"] == STUDY_ID
    assert facts["Series"] == "1,792 found · 1,180 after business rules · 1,153 forecast"
    by_name = {item.name: item for item in outputs.items}
    assert by_name["sales"].location == "cat.gold.sales (version 1036)"
    assert by_name["shops"].location == "cat.silver.shops (version 300)"
    assert by_name["primary query"].description == "1,079,823 rows returned"
    assert by_name["studies"].location == "cat.me.af_studies"
    assert STUDY_ID in by_name["studies"].description
    assert by_name["Frozen study data"].location == str(tmp_path / "benchmarks" / STUDY_ID)
    # Off Databricks nothing links anywhere.
    assert all(item.link is None for item in outputs.items)
    with pytest.raises(FileNotFoundError):
        study_outputs(registry, "e" * 64)
    registry.store.tables = {"studies": "cat.me.af_studies"}
    assert "studies" in {item.name for item in study_outputs(registry, STUDY_ID).items}


def test_databricks_links_point_to_catalog_explorer_and_mlflow(tmp_path, monkeypatch):
    from mlops import _volume_files

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
    host = "https://adb-1.2.azuredatabricks.net"
    monkeypatch.setattr(_volume_files, "_client",
                        lambda: SimpleNamespace(config=SimpleNamespace(host=host + "/")))
    root = pathlib.Path("/Volumes/cat/me/storage/registered-v1")
    registry = _registry(tmp_path, store_tables={
        "studies": "cat.me.af_studies", "study_data_issues": "cat.me.af_study_data_issues"})
    registry.root = root
    registry._registration_path = lambda _study_id: tmp_path / "registrations" / f"{STUDY_ID}.json"
    by_name = {item.name: item for item in study_outputs(registry, STUDY_ID).items}
    assert by_name["sales"].link == f"{host}/explore/data/cat/gold/sales"
    assert by_name["studies"].link == f"{host}/explore/data/cat/me/af_studies"
    assert by_name["Frozen study data"].link == f"{host}/explore/data/volumes/cat/me/storage"
    assert outputs_module.mlflow_run_link(host, "7", "r1") == f"{host}/ml/experiments/7/runs/r1"
    links = outputs_module.tracking_records({"test": SimpleNamespace(experiment_id="7",
                                                                     run_id="r2")})
    assert links == {"test": {"experiment_id": "7", "run_id": "r2",
                              "url": f"{host}/ml/experiments/7/runs/r2"}}


def test_training_map_lists_search_models_forecasts_scores_and_mlflow(tmp_path):
    result = _result(tmp_path)
    outputs = workflow_outputs(result)
    facts = dict((label, value) for _, label, value in outputs.facts)
    assert facts["Selected settings"] == "changepoint_prior_scale=0.05"
    assert facts["Validation scores"] == "mae 1.25 · wape 0.3125"
    names = [(item.group, item.name) for item in outputs.items]
    assert ("Parameter search", "Cross-validation results") in names
    assert ("Validation", "Trained model") in names and ("Test", "Forecasts") in names
    assert ("Validation", "MLflow run") in names and ("Test", "MLflow run") not in names
    forecasts = next(i for i in outputs.items if i.group == "Test" and i.name == "Forecasts")
    assert forecasts.description == "1,153 series × 14 periods from 2026-10-05"
    # The result combines the run's data map with its training map.
    data = OutputsMap((("data", "Study ID", STUDY_ID),), ())
    combined = _result(tmp_path, data_outputs=data).outputs()
    assert combined.facts[0] == ("data", "Study ID", STUDY_ID)
    assert _result(tmp_path, data_outputs=data).outputs("training").select("data").facts == ()
    assert _result(tmp_path).outputs("data").items == ()
    fixed = workflow_outputs(_result(tmp_path, tuning_path=None))
    selected = next(item for item in fixed.items if item.name == "Selected settings")
    assert selected.group == "Model configuration"
    assert "search profile" not in selected.description


def test_show_logs_plain_text_outside_a_notebook(caplog):
    outputs = OutputsMap((("data", "Study ID", "abc"),), ())
    with caplog.at_level(logging.INFO, logger="autoforecast.outputs.test"):
        outputs_module.show(outputs, logging.getLogger("autoforecast.outputs.test"))
    assert "Study ID: abc" in caplog.text


def test_mlflow_runs_record_source_tables_study_and_outputs(tmp_path):
    from autoforecast import run_mlflow

    _registry(tmp_path)
    workflow = SimpleNamespace(
        provenance={"study": {"path": str(tmp_path / "benchmarks" / STUDY_ID / "benchmark.json")}},
        study_id=STUDY_ID, parent_study_id=None)
    calls = []
    client = SimpleNamespace(
        log_inputs=lambda run_id, datasets: calls.append(("inputs", run_id, datasets)),
        set_tag=lambda run_id, key, value: calls.append(("tag", run_id, key, value)),
        log_dict=lambda run_id, value, name: calls.append(("dict", run_id, name, value)))
    outputs = OutputsMap((("data", "Study ID", STUDY_ID),), ())
    run_mlflow._log_study_inputs(client, "r1", workflow, outputs)
    datasets = calls[0][2]
    sources = [json.loads(item.dataset.source) for item in datasets]
    assert [(s["delta_table_name"], s["delta_table_version"]) for s in sources] == [
        ("cat.gold.sales", 1036), ("cat.silver.shops", 300)]
    assert {item.dataset.source_type for item in datasets} == {"delta_table"}
    assert calls[1] == ("tag", "r1", "autoforecast.study_id", STUDY_ID)
    assert calls[2][:3] == ("dict", "r1", "autoforecast/outputs.json")
    # A tracking server that refuses the inputs never fails the export.
    def refused(*_args, **_kwargs):
        raise RuntimeError("not supported")

    client.log_inputs = refused
    calls.clear()
    run_mlflow._log_study_inputs(client, "r2", workflow, outputs)
    assert calls == [
        ("tag", "r2", "autoforecast.study_id", STUDY_ID),
        ("dict", "r2", "autoforecast/outputs.json", outputs.to_dict()),
    ]


def test_mlflow_sources_include_auxiliary_queries(tmp_path, monkeypatch):
    from autoforecast import run_mlflow, study_registration

    _registry(tmp_path)
    registration = tmp_path / "registrations" / f"{STUDY_ID}.json"
    saved = json.loads(registration.read_text())
    saved["definition"]["admission_evidence"] = {"path": "checks/admission.json"}
    registration.write_text(json.dumps(saved))
    monkeypatch.setattr(study_registration, "_verify_admission", lambda *_args, **_kwargs: {
        "identity": {"sources": {"population": {"tables": {
            "products": {"table": "cat.core.products", "version": 7},
        }}}},
    })
    workflow = SimpleNamespace(
        provenance={"study": {"path": str(tmp_path / "benchmarks" / STUDY_ID / "benchmark.json")}},
        study_id=STUDY_ID, parent_study_id=None)
    assert run_mlflow._study_source_tables(workflow) == [
        ("cat.core.products", 7), ("cat.gold.sales", 1036), ("cat.silver.shops", 300),
    ]


def test_supplied_delta_stage_tables_appear_in_study_and_mlflow_maps(tmp_path):
    from autoforecast import run_mlflow

    registry = _registry(tmp_path)
    registration = tmp_path / "registrations" / f"{STUDY_ID}.json"
    saved = json.loads(registration.read_text())
    saved["definition"]["source"] = {"kind": "supplied_stages", "stages": {
        "training": {"source": {"tables": {
            "train": {"table": "cat.gold.sales", "version": 4}}}},
        "testing": {"source": {"tables": {
            "test": {"table": "cat.gold.sales", "version": 5}}}},
    }}
    registration.write_text(json.dumps(saved))
    workflow = SimpleNamespace(
        provenance={"study": {"path": str(tmp_path / "benchmarks" / STUDY_ID / "benchmark.json")}},
        study_id=STUDY_ID, parent_study_id=None)
    assert run_mlflow._study_source_tables(workflow) == [
        ("cat.gold.sales", 4), ("cat.gold.sales", 5),
    ]
    locations = {item.location for item in study_outputs(registry, STUDY_ID).items}
    assert {"cat.gold.sales (version 4)", "cat.gold.sales (version 5)"} <= locations
