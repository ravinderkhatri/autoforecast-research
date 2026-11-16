"""Public model lists preserve registered study identity across campaign workers."""

import json
from types import SimpleNamespace

import pytest

from autoforecast import run_study
from autoforecast.study_builtins import builtin_profile
from autoforecast.study_campaign import StudyModel
from autoforecast.study_model_list import (
    StudyCampaignResult,
    load_study_campaign,
    prepare_candidates,
)
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_builtins import daily_study
from autoforecast.test_study_tuning import spark as _spark


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


@pytest.mark.parametrize("options,match", [
    ({"model": "seasonal_naive", "models": ["seasonal_naive"]}, "exactly one"),
    ({"models": []}, "nonempty ordered"),
    ({"models": "auto_ets"}, "nonempty ordered"),
    ({"models": {"auto_ets"}}, "nonempty ordered"),
    ({"models": ["tft"]}, "unsupported"),
    ({"models": ["seasonal_naive"] * 2}, "distinct"),
    ({"models": [lambda request: None]}, "importable"),
    ({"model": "seasonal_naive", "execution": "local"}, "require models"),
    ({"models": ["seasonal_naive"], "model_config": "absent"}, "per-candidate"),
    ({"models": ["seasonal_naive"], "fixed_parameters": {}}, "per-candidate"),
    ({"models": ["seasonal_naive"], "model_options": {"other": {}}}, "absent"),
    ({"models": ["seasonal_naive"], "model_options": {"seasonal_naive": {"typo": 1}}}, "invalid"),
    ({"models": ["seasonal_naive"], "execution": "cluster"}, "ray_address"),
    ({"models": ["seasonal_naive"], "execution": "cluster", "ray_address": ""}, "nonempty"),
    ({"models": ["seasonal_naive"], "max_parallel_models": True}, "positive integer"),
    ({"models": ["seasonal_naive"], "max_parallel_models": 2}, "sequential"),
])
def test_run_study_model_list_contract(options, match):
    # All errors precede registry construction, source reads and model fitting.
    with pytest.raises(ValueError, match=match):
        run_study(study_id="absent", **options)


def test_profiles_are_frozen_before_registration(tmp_path):
    path = tmp_path / "profile.json"
    profile = builtin_profile("window_average", SimpleNamespace(season_length=7))
    path.write_text(json.dumps(profile))
    candidates = prepare_candidates(
        ["window_average"], {"window_average": {"model_config": path}}, {},
    )
    path.write_text("changed")
    assert json.loads(candidates[0]["profile"]) == profile
    assert candidates[0]["declaration"].fixed_parameters == {}
    with pytest.raises(ValueError, match="hpo"):
        prepare_candidates(["window_average"], {"window_average": {"fixed_parameters": None}}, {})


@pytest.mark.parametrize("invalid", ["fixed_parameter", "selection_metric"])
def test_profile_choices_fail_before_registration(tmp_path, invalid):
    profile = builtin_profile("window_average", SimpleNamespace(season_length=7))
    options = {}
    if invalid == "fixed_parameter":
        options["fixed_parameters"] = {"model.window": -1}
        match = "model.window"
    else:
        profile["selection"] = {"metric": "unknown"}
        match = "selection.metric"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match=match):
        run_study(study_id="absent", models=["window_average"],
                  model_options={"window_average": {"model_config": path, **options}})


def test_candidate_snapshot_preserves_captured_source_evidence(tmp_path, monkeypatch):
    from autoforecast.implementation import resolve_implementation
    from autoforecast.test_implementation import two_callbacks
    from autoforecast.test_study_tuning import config

    module, scope = two_callbacks(tmp_path, monkeypatch)
    implementation = resolve_implementation(None, scope, module.first, "custom-constant")
    declared = StudyModel("synthetic_two_callbacks:first", implementation, config(tmp_path),
                          fixed_parameters={})
    candidate = prepare_candidates([declared], {}, {})[0]["declaration"]
    assert candidate.implementation is implementation
    assert candidate.implementation.binding == implementation.binding
    (tmp_path / "source/model.py").write_text("def first(request): return 999\n")
    with pytest.raises(ValueError, match="source changed"):
        resolve_implementation(candidate.implementation, None, module.first, "custom-constant")


@pytest.mark.parametrize("corruption", ["definition", "envelope_hash", "registered_id"])
def test_worker_refuses_changed_registration_before_opening_inputs(tmp_path, corruption):
    import hashlib

    from autoforecast.benchmark import digest
    from autoforecast.study_campaign import _verified_model_inputs

    definition = {"study_id": "original", "library_sha256": "a" * 64}
    request = {"models": [{}], "registration": {
        "study_id": "original", "sha256": digest(definition),
    }, "registration_record": {"definition": dict(definition), "sha256": digest(definition)}}
    if corruption == "definition":
        request["registration_record"]["definition"]["library_sha256"] = "b" * 64
    elif corruption == "envelope_hash":
        request["registration_record"]["sha256"] = "c" * 64
    else:
        request["registration"]["study_id"] = "foreign"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="worker registration identity"):
        _verified_model_inputs(path, hashlib.sha256(path.read_bytes()).hexdigest(), 0)


@pytest.mark.parametrize("mode", ["once", "every_stage"])
def test_worker_applies_retained_file_check_policy(tmp_path, mode):
    import hashlib

    from mlops import _volume_files

    from autoforecast.study_campaign import _campaign_file_checks

    path = tmp_path / "request.json"
    path.write_text(json.dumps({"public_settings": {"file_checks": mode}}))

    @_campaign_file_checks
    def observe(request_path, request_sha):
        return _volume_files.every_stage()

    with _volume_files.file_checks("once"):
        assert observe(path, hashlib.sha256(path.read_bytes()).hexdigest()) == (mode == "every_stage")
        assert not _volume_files.every_stage()


@pytest.mark.parametrize("execution,parallel", [("sequential", 1), ("local", 2)])
def test_public_campaign_create_reuse_and_reload(spark, tmp_path, execution, parallel):
    from autoforecast.study_registration import StudyRegistry

    config = daily_study(tmp_path / "input")
    registry = StudyRegistry.from_config(config, spark=spark)
    arguments = dict(
        models=["seasonal_naive", "window_average"], registry=registry,
        model_options={name: {"memory_gib": 2.0} for name in
                       ("seasonal_naive", "window_average")},
        execution=execution, max_parallel_models=parallel,
        timeout_seconds=600, output_dir=tmp_path / "campaign",
    )
    result = run_study(
        study_info={"study_id": None, "create_study": True, "publish_results": False},
        config=config, **arguments,
    )
    assert isinstance(result, StudyCampaignResult)
    assert result.status == "completed", result.summary()
    assert result.study_id == result.requested_study_id
    assert [row["model_id"] for row in result.candidates] == ["seasonal_naive", "window_average"]
    for row in result.candidates:
        workflow = load_forecast_workflow(row["workflow_path"])
        assert workflow.study_id == result.study_id
        assert workflow.provenance["registration"]["study_id"] == result.requested_study_id
        assert workflow.validation_metrics and workflow.testing_metrics
    reloaded = load_study_campaign(result.path)
    assert reloaded.summary() == result.summary()
    expected_workflows = {row["workflow_path"] for row in result.candidates}
    for saved in (result, reloaded):
        training = saved.outputs("training")
        assert {item.location for item in training.items if item.name == "All saved evidence"} == (
            expected_workflows)
        assert {item.section for item in training.items} == {"training"}
    assert result.outputs("data").items
    with pytest.raises(ValueError, match="section must be"):
        reloaded.outputs("unknown")
    repeated = run_study(study_id=result.requested_study_id, **arguments)
    assert repeated.path == result.path
    assert repeated.candidates == result.candidates
    # A bound registration field cannot be changed without invalidating the workflow.
    request_path = next((tmp_path / "campaign").glob("request-*/request.json"))
    request = json.loads(request_path.read_bytes())
    assert request["registration"]["sha256"] == registry.definition(result.study_id)[1]
    if execution == "local":
        records = [json.loads((result.path / f"models/candidate-{i}/validation/result.json")
                             .read_text()) for i in range(2)]
        assert records[0]["pid"] != records[1]["pid"]
        assert max(row["started"] for row in records) < min(row["finished"] for row in records)
    from autoforecast.study_campaign import _load_model_result

    record_path = result.path / "models/candidate-0/validation/result.json"
    changed_request = {**request, "registration": {**request["registration"], "sha256": "0" * 64}}
    record = json.loads(record_path.read_text())
    from autoforecast.benchmark import digest

    original = record_path.read_bytes()
    try:
        # Even a matching request digest cannot excuse a foreign registration binding.
        record_path.write_text(json.dumps({**record, "request_sha256": digest(changed_request)}))
        with pytest.raises(ValueError, match="registration binding"):
            _load_model_result(record_path, changed_request, request["models"][0], "validation")
        record_path.write_text(json.dumps({**record, "pid": record["pid"] + 1}))
        with pytest.raises(ValueError, match="completed campaign phase"):
            load_study_campaign(result.path)
    finally:
        record_path.write_bytes(original)
    original_request = request_path.read_bytes()
    try:
        request_path.write_text(json.dumps(changed_request))
        with pytest.raises(ValueError, match="input identity changed"):
            load_study_campaign(result.path)
    finally:
        request_path.write_bytes(original_request)
    registry.close()


def test_study_model_requires_profile_before_registration():
    with pytest.raises(ValueError, match="requires config"):
        prepare_candidates([StudyModel("autoforecast.test_study_tuning:predict")], {}, {})


def test_malformed_implementation_fails_before_registration(tmp_path):
    from autoforecast.test_study_tuning import config

    model = StudyModel("autoforecast.test_study_tuning:predict", {}, config(tmp_path),
                       fixed_parameters={})
    with pytest.raises(ValueError, match="implementation requires"):
        run_study(study_id="absent", models=[model])


@pytest.mark.parametrize("missing", ["implementation", "helper"])
def test_creation_rejects_unusable_model_before_registration(spark, tmp_path, monkeypatch, missing):
    from functools import wraps

    from autoforecast import study_registration
    from autoforecast.test_study_tuning import config, implementation

    @wraps(study_registration.register_study)
    def unexpected_registration(*args, **kwargs):
        pytest.fail("invalid model reached registration")

    monkeypatch.setattr(study_registration, "register_study", unexpected_registration)
    model = StudyModel(
        "autoforecast.test_study_tuning:predict",
        None if missing == "implementation" else implementation(), config(tmp_path),
        fixed_parameters={}, source_modules=("missing_model_list_helper",) if missing == "helper" else (),
    )
    with study_registration.StudyRegistry(tmp_path / "registry", spark=spark) as registry:
        with pytest.raises((ValueError, ModuleNotFoundError), match=(
            "supply implementation" if missing == "implementation" else "missing_model_list_helper"
        )):
            run_study(
                study_info={"study_id": None, "create_study": True, "publish_results": False},
                config=daily_study(tmp_path / "input"), registry=registry, models=[model],
            )
        assert not list((registry.root / "registrations").glob("*.json"))


def test_create_list_uses_configured_capture_with_explicit_registry(spark, tmp_path):
    from pathlib import Path

    import yaml

    from autoforecast import test_study_tuning
    from autoforecast.study_registration import StudyRegistry

    source = daily_study(tmp_path / "input")
    configured = yaml.safe_load(source.read_text())
    configured["runtime"] = {"implementation": {
        "source_roots": {"model": str(Path(test_study_tuning.__file__).parent)},
        "include": ["test_study_tuning.py"],
    }}
    source.write_text(yaml.safe_dump(configured))
    model = StudyModel("autoforecast.test_study_tuning:predict", None,
                       test_study_tuning.config(tmp_path), fixed_parameters={})
    with StudyRegistry(tmp_path / "registry", spark=spark) as registry:
        assert not registry.runtime.get("implementation")
        result = run_study(
            study_info={"study_id": None, "create_study": True, "publish_results": False},
            config=source, registry=registry, models=[model],
            output_dir=tmp_path / "campaign", timeout_seconds=600,
        )
        assert result.status == "completed", result.summary()
        workflow = load_forecast_workflow(result.candidates[0]["workflow_path"])
        assert workflow.provenance["implementation"]["version"] == "saved-source-v1"
        assert workflow.provenance["registration"]["study_id"] == result.requested_study_id
        assert load_study_campaign(result.path).summary() == result.summary()
        assert not registry.runtime.get("implementation")


def test_study_ready_models_are_connected_or_explicitly_deferred():
    from autoforecast.adapter_contracts import DESCRIPTORS
    from autoforecast.study_builtins import BUILTIN_STUDY_MODELS

    deferred = {"autogluon_zoo", "hmfnet", "last_value", "mft", "pact", "sft", "tft",
                "tft_hnb", "tft_q", "timesfm_hnb", "timesfm_standalone", "timesfm_v8"}
    ready = {name for name, descriptor in DESCRIPTORS.items() if descriptor.supports_shared_studies}
    connected = set(BUILTIN_STUDY_MODELS)
    assert not deferred & connected
    assert ready == (connected - {"window_average"}) | deferred


@pytest.mark.parametrize("resource,value,match", [
    ("cpus", 0, "CPU"), ("cpus", True, "CPU"), ("gpus", -1, "GPU"),
    ("memory_gib", float("nan"), "finite"), ("memory_gib", -2, "finite"),
    ("gpus", 1, "GPU reservations"), ("node_labels", {"pool": "x"}, "explicitly supplied"),
])
def test_resources_fail_before_registration(resource, value, match):
    with pytest.raises(ValueError, match=match):
        run_study(study_id="missing", models=["seasonal_naive"],
                  model_options={"seasonal_naive": {resource: value}})


def test_public_local_campaign_requires_concurrent_workers(spark, tmp_path, monkeypatch):
    from dataclasses import replace

    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_campaign import models

    config = daily_study(tmp_path / "input")
    with StudyRegistry.from_config(config, spark=spark) as registry:
        study_id = register_study(config, registry=registry)
        declarations = [replace(model, entrypoint="autoforecast.test_study_campaign:rendezvous_predict")
                        for model in models(tmp_path)]
        rendezvous = tmp_path / "rendezvous"
        rendezvous.mkdir()
        monkeypatch.setenv("AUTOFORECAST_TEST_VALIDATION_RENDEZVOUS", str(rendezvous))
        result = run_study(study_id=study_id, models=declarations, registry=registry,
                           execution="local", max_parallel_models=2, timeout_seconds=600,
                           output_dir=tmp_path / "campaign")
        assert result.status == "completed", result.summary()
        records = [json.loads((result.path / f"models/candidate-{i}/validation/result.json")
                             .read_text()) for i in range(2)]
        assert len({record["pid"] for record in records}) == 2
        assert max(r["started"] for r in records) < min(r["finished"] for r in records)
        assert len(list(rendezvous.iterdir())) == 2


def test_campaign_partial_publication_retry_does_not_refit(spark, tmp_path):
    import mlflow

    from autoforecast.study_registration import StudyRegistry, register_study

    config = daily_study(tmp_path / "input")
    real = StudyRegistry.from_config(config, spark=spark)
    study_id = register_study(config, registry=real)

    class DestinationRegistry:
        # An injected destination failure; registration and all model workflows remain real.
        store = SimpleNamespace(validate_destinations=lambda: None)
        failed = False
        calls = []
        rows = {}

        def __getattr__(self, name):
            return getattr(real, name)

        def publish_results(self, saved_id, path, *, mlflow_links=None):
            run = load_forecast_workflow(path)
            assert run.study_id == saved_id
            assert set(mlflow_links) == {"validation", "test"}
            self.calls.append(run.run_id)
            if len(self.calls) == 2 and not self.failed:
                self.failed = True
                raise OSError("injected second publication failure")
            self.rows[run.run_id] = run.testing_metrics
            return {"status": "published", "run_id": run.run_id}

    registry = DestinationRegistry()
    arguments = dict(study_id=study_id, registry=registry, publish=True,
                     models=["seasonal_naive", "window_average"], timeout_seconds=600,
                     tracking="mlflow", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                     mlflow_experiment="Public model-list publication",
                     output_dir=tmp_path / "campaign")
    client = mlflow.tracking.MlflowClient(arguments["tracking_uri"])
    client.create_experiment(arguments["mlflow_experiment"],
                             artifact_location=(tmp_path / "mlflow-artifacts").as_uri())
    with pytest.raises(OSError, match="second publication") as caught:
        run_study(**arguments)
    saved = load_study_campaign(caught.value.campaign_path)
    assert saved.status == "publication_pending"
    assert len(saved.publication) == 1
    run_ids = [candidate["run_id"] for candidate in saved.candidates]
    repeated = run_study(**arguments)
    assert repeated.status == "completed"
    assert [row["run_id"] for row in repeated.candidates] == run_ids
    assert set(registry.rows) == set(run_ids)
    assert all(row["status"] == "published" for row in repeated.publication.values())
    experiment = client.get_experiment_by_name(arguments["mlflow_experiment"])
    assert experiment is not None
    assert len(client.search_runs([experiment.experiment_id])) == 4
    assert all(set(links) == {"validation", "test"} for links in repeated.tracking.values())
    assert load_study_campaign(repeated.path).summary() == repeated.summary()
    real.close()


@pytest.mark.parametrize("execution,parallel", [("sequential", 1), ("local", 2)])
def test_all_builtin_names_produce_verified_workflows(spark, tmp_path, execution, parallel):
    from autoforecast.study_builtins import BUILTIN_STUDY_MODELS
    from autoforecast.study_registration import StudyRegistry, register_study

    config = daily_study(tmp_path / "input")
    names = list(BUILTIN_STUDY_MODELS)
    with StudyRegistry.from_config(config, spark=spark) as registry:
        saved_id = register_study(config, registry=registry)
        result = run_study(
            saved_id, models=names, registry=registry, execution=execution,
            max_parallel_models=parallel, timeout_seconds=900,
            model_options={name: {"memory_gib": 2.0} for name in names},
            output_dir=tmp_path / "campaign",
        )
        assert result.status == "completed", result.summary()
        assert [row["model_id"] for row in result.candidates] == names
        contracts = set()
        for row in result.candidates:
            workflow = load_forecast_workflow(row["workflow_path"])
            assert workflow.testing_metrics and workflow.validation_metrics
            assert workflow.provenance["registration"]["study_id"] == saved_id
            contracts.add(json.dumps(workflow.validation["evaluation_contract"], sort_keys=True))
            assert workflow.model_links["test"]["path"]
        assert len(contracts) == 1
        assert load_study_campaign(result.path).summary() == result.summary()


def test_public_cluster_probes_workers_and_keeps_external_cluster(spark, tmp_path):
    import ray

    from autoforecast.study_registration import StudyRegistry, register_study

    config = daily_study(tmp_path / "input")
    context = ray.init(address="local", num_cpus=2, include_dashboard=False,
                       labels={"pool": "forecast"})
    try:
        with StudyRegistry.from_config(config, spark=spark) as registry:
            saved_id = register_study(config, registry=registry)
            result = run_study(
                saved_id, models=["seasonal_naive", "window_average"], registry=registry,
                execution="cluster", ray_address=context.address_info["gcs_address"],
                max_parallel_models=2, timeout_seconds=600,
                model_options={name: {"memory_gib": 2.0, "node_labels": {"pool": "forecast"}}
                               for name in ("seasonal_naive", "window_average")},
                output_dir=tmp_path / "campaign",
            )
            assert result.status == "completed", result.summary()
            assert ray.is_initialized()
            admission = json.loads((result.path / "stages/validation/attempt-0001"
                                    / "worker-admission.json").read_text())
            assert admission["nodes"][0]["candidates"] == [0, 1]
            assert admission["nodes"][0]["node_id"] in {
                node["NodeID"] for node in ray.nodes() if node["Alive"]}
            assert load_study_campaign(result.path).summary() == result.summary()
            import hashlib
            import subprocess
            import sys

            request = next((tmp_path / "campaign").glob("request-*/request.json"))
            sha = hashlib.sha256(request.read_bytes()).hexdigest()
            probe = tmp_path / "unlocked-probe"
            probe.mkdir()
            (probe / "coordinator").write_text(sha)
            command = (
                "import sys; from autoforecast.study_campaign import _probe_worker; "
                "_probe_worker(sys.argv[1], sys.argv[2], [0], sys.argv[3])"
            )
            refused = subprocess.run([sys.executable, "-c", command, str(request), sha, str(probe)],
                                     capture_output=True, text=True, timeout=90)
            assert refused.returncode != 0
            assert "output storage does not share locks with coordinator" in refused.stderr
            assert not (probe / "worker").exists()
    finally:
        ray.shutdown()


def test_public_campaign_reports_derived_id_without_validation(spark, tmp_path):
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_explicit_study import stages

    with StudyRegistry(tmp_path / "registry", spark=spark) as registry:
        saved_id = register_study(registry=registry, feature_roles={"season_length": 1},
                                  **stages(spark, validation=False))
        options = {}
        for name in ("seasonal_naive", "window_average"):
            profile = builtin_profile(name, SimpleNamespace(season_length=1))
            profile.update(series_policy="exclude_unavailable", minimum_history=7)
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(profile))
            options[name] = {"model_config": path, "memory_gib": 2.0}
        arguments = dict(study_id=saved_id, registry=registry, models=list(options),
                         model_options=options, timeout_seconds=600,
                         output_dir=tmp_path / "campaign")
        with pytest.raises(ValueError, match="requires outer validation"):
            run_study(**arguments, final_evaluation="validation_winner")
        result = run_study(**arguments)
        assert result.status == "completed", result.summary()
        assert result.study_id != saved_id
        assert result.requested_study_id == saved_id
        assert result.selected_model is None
        assert all(row["validation_metrics"] is None for row in result.candidates)
        definition, _ = registry.definition(result.study_id)
        assert definition["effective_series"] == 1
        assert definition["parent"]["study_id"] == saved_id
        from autoforecast.study_results import _publication

        for row in result.candidates:
            with _publication(definition, registry.definition(result.study_id)[1],
                              row["workflow_path"], scratch_root=registry.scratch_root) as tables:
                assert {record[3] for record in tables["study_results"]} == {"test"}
        assert load_study_campaign(result.path).summary() == result.summary()


def test_public_campaign_rejects_volume_output_before_fitting():
    from autoforecast.study_model_list import run_registered_campaign

    with pytest.raises(ValueError, match="/Volumes campaign output is not supported"):
        run_registered_campaign(
            None, registry=None, candidates=[], options={}, settings={}, experiment_id=None,
            audit=None, output_dir="/Volumes/catalog/schema/volume/campaign", data_outputs=None,
        )


def test_public_mixed_fixed_and_tuned_candidates_match_single_model(spark, tmp_path):
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_campaign import models

    config = daily_study(tmp_path / "input")
    profile = builtin_profile("window_average", SimpleNamespace(season_length=7))
    profile["hpo"] = {"backend": "optuna", "max_trials": 2, "seed": 42}
    path = tmp_path / "window.json"
    path.write_text(json.dumps(profile))
    custom = models(tmp_path)[0]
    with StudyRegistry.from_config(config, spark=spark) as registry:
        saved_id = register_study(config, registry=registry)
        single = run_study(saved_id, "window_average", registry=registry, model_config=path,
                           output_dir=tmp_path / "single")
        result = run_study(
            saved_id, models=["window_average", custom], registry=registry,
            model_options={"window_average": {"model_config": path}},
            output_dir=tmp_path / "campaign", timeout_seconds=600,
            final_evaluation="validation_winner",
        )
        assert result.status == "completed", result.summary()
        assert result.selected_model == "window_average"
        assert result.candidates[1]["status"] == "not_final_selected"
        tuned = load_forecast_workflow(result.candidates[0]["workflow_path"])
        checkpoint = json.loads((result.path / "models/candidate-0/validation/result.json").read_text())
        validation = load_forecast_workflow(checkpoint["workflow_path"])
        assert validation.tuning_path is not None
        assert validation.parameters == tuned.parameters
        assert tuned.parameters == single.parameters
        # Separate Spark reductions can differ in their final floating-point bits.
        assert tuned.validation_metrics == pytest.approx(single.validation_metrics, abs=1e-12, rel=1e-12)
        assert tuned.testing_metrics == pytest.approx(single.testing_metrics, abs=1e-12, rel=1e-12)
        assert load_study_campaign(result.path).summary() == result.summary()


def test_failed_public_campaign_preserves_completed_peer(spark, tmp_path, monkeypatch):
    from dataclasses import replace

    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_campaign import models

    config = daily_study(tmp_path / "input")
    declared = models(tmp_path)
    declared[1] = replace(declared[1], entrypoint="autoforecast.test_study_campaign:fail_once")
    monkeypatch.setenv("AUTOFORECAST_TEST_FAILURE_MARKER", str(tmp_path / "failed-once"))
    with StudyRegistry.from_config(config, spark=spark) as registry:
        saved_id = register_study(config, registry=registry)
        arguments = dict(study_id=saved_id, models=declared, registry=registry,
                         output_dir=tmp_path / "campaign", timeout_seconds=600)
        result = run_study(**arguments)
        assert result.status == "failed", result.summary()
        assert result.candidates[0]["workflow_path"]
        assert "workflow_path" not in result.candidates[1]
        assert result.candidates[1]["error"]
        assert load_study_campaign(result.path).summary() == result.summary()
        # The existing retry budget is exhausted. A repeat must not silently start a new fit.
        repeated = run_study(**arguments)
        assert repeated.status == "failed"
        assert repeated.candidates == result.candidates


def fail_final_predict(request):
    from autoforecast.test_study_tuning import predict

    if request.purpose == "refit":
        raise RuntimeError("injected final fit failure")
    return predict(request)


def test_public_campaign_retains_validation_winner_when_final_fails(spark, tmp_path):
    from dataclasses import replace

    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_campaign import models

    config = daily_study(tmp_path / "input")
    candidate = replace(models(tmp_path)[0],
                        entrypoint="autoforecast.test_study_model_list:fail_final_predict")
    with StudyRegistry.from_config(config, spark=spark) as registry:
        saved_id = register_study(config, registry=registry)
        result = run_study(saved_id, models=[candidate], registry=registry,
                           output_dir=tmp_path / "campaign", timeout_seconds=600)
        assert result.status == "failed", result.summary()
        assert result.selected_model == "constant-0"
        assert result.candidates[0]["status"] == "incomplete"
        assert result.candidates[0]["validation_metrics"] is not None
        assert result.candidates[0]["testing_metrics"] is None
        assert result.candidates[0]["error"]
        assert load_study_campaign(result.path).summary() == result.summary()


def test_public_campaign_keeps_missing_test_actuals_unavailable(spark, tmp_path):
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.study_registration import StudyRegistry, register_study

    config = daily_study(tmp_path / "input")
    source = config.parent / "source.parquet"
    table = pq.read_table(source)
    pq.write_table(pa.Table.from_pylist([
        row for row in table.to_pylist() if row["day"] <= dt.date(2024, 4, 23)
    ], schema=table.schema), source)
    options = {}
    for name in ("seasonal_naive", "window_average"):
        profile = builtin_profile(name, SimpleNamespace(season_length=7))
        profile["series_policy"] = "use_available"
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(profile))
        options[name] = {"model_config": path, "memory_gib": 2.0}
    with StudyRegistry.from_config(config, spark=spark) as registry:
        saved_id = register_study(config, registry=registry)
        result = run_study(saved_id, models=list(options), model_options=options,
                           registry=registry, output_dir=tmp_path / "campaign",
                           timeout_seconds=600)
        assert result.status == "pending_actuals", result.summary()
        assert all(row["status"] == "pending_actuals" for row in result.candidates)
        assert all(row["testing_metrics"]["mae"] is None for row in result.candidates)
        assert load_study_campaign(result.path).status == "pending_actuals"


@pytest.mark.parametrize("probe_fails", [False, True])
def test_registered_cluster_schedules_only_admitted_nodes(tmp_path, monkeypatch, probe_fails):
    from pathlib import Path

    from ray.util.scheduling_strategies import NodeLabelSchedulingStrategy

    from autoforecast.test_study_campaign import _RayRecorder, ray_phase

    request, run, _ = ray_phase.__wrapped__(tmp_path, monkeypatch)
    admitted_id = "a" * 56
    request["registration"] = {"study_id": "registered", "sha256": "retained"}
    for model in request["models"]:
        model["node_labels"] = {"pool": "forecast"}
    (tmp_path / "campaign").mkdir()

    class AdmissionRay(_RayRecorder):
        def remote(self, **options):
            def wrap(function):
                if function.__name__ != "_probe_worker":
                    return self

                class Probe:
                    def options(inner, **placement):
                        assert placement["scheduling_strategy"].node_id == admitted_id
                        return inner

                    def remote(inner, path, sha, indices, folder):
                        (Path(folder) / "worker").write_text(sha)
                        return ("probe", {"host": "worker", "candidates": indices})

                return Probe()
            return wrap

        def get(self, reference, **options):
            if isinstance(reference, tuple):
                if probe_fails:
                    raise ValueError("worker admission refused")
                return reference[1]
            return super().get(reference)

        def cancel(self, reference, force):
            if not isinstance(reference, tuple):
                super().cancel(reference, force)

    ray = AdmissionRay([])
    monkeypatch.setattr(ray, "nodes", lambda: [
        {"Alive": True, "NodeID": node, "Resources": {"CPU": 8, "memory": 8 * 1024**3},
         "Labels": {"pool": pool, "ray.io/node-id": node}}
        for node, pool in ((admitted_id, "forecast"), ("b" * 56, "other"))
    ])
    if probe_fails:
        with pytest.raises(ValueError, match="worker admission refused"):
            run(ray)
        assert not ray.submissions
    else:
        run(ray)
        assert len(ray.submissions) == len(request["models"])
        for _, options in ray.submissions:
            assert isinstance(options["scheduling_strategy"], NodeLabelSchedulingStrategy)
            selectors = options["scheduling_strategy"].hard
            assert {item.key: item.operator.values for item in selectors} == {
                "pool": ["forecast"], "ray.io/node-id": [admitted_id],
            }
    assert ray.closed
