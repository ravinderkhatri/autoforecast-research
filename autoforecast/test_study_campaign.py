"""Public custom-model campaign: synthetic A/B series, no customer data."""

import contextlib
import copy
import hashlib
import importlib
import json
import os
import pathlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from autoforecast import study_campaign
from autoforecast.study_campaign import StudyModel, _entrypoint, _read
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_tuning import config, freeze, implementation, open_study
from autoforecast.test_study_tuning import spark as _spark


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def models(tmp_path):
    declarations = []
    for index, level in enumerate((0.0, 10.0)):
        root = tmp_path / str(index)
        root.mkdir()
        path = config(root, budget=1)
        value = yaml.safe_load(path.read_text())
        value["model"]["id"] = f"constant-{index}"
        value["selection"] = {"stage": "validation", "metric": "mae"}
        path.write_text(yaml.safe_dump(value))
        declarations.append(
            StudyModel(
                "autoforecast.test_study_tuning:predict",
                implementation(),
                path,
                fixed_parameters={"model.level": level},
                cpus=1,
                memory_gib=2.0,
            )
        )
    return declarations


def last_value_predict(request):
    """A distinct model: repeat each series' last observation plus a fixed offset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.panel import Forecast
    from autoforecast.study_tuning import FitPredictResult

    latest = {}
    for row in request.training_data.orderBy(request.panel.time_col).collect():
        latest[row[request.panel.series_col]] = row[request.panel.target_col]
    folder = request.output_dir / "predictions"
    folder.mkdir()
    pq.write_table(
        pa.Table.from_pylist([
            {"series_id": key, "date": day, "yhat": value + request.model_parameters["level"]}
            for key, value in latest.items()
            for day in request.forecast_dates
        ]),
        folder / "part.parquet",
    )
    model = request.output_dir / "model"
    model.mkdir()
    (model / "last-values.json").write_text(json.dumps(latest))
    return FitPredictResult(
        Forecast(
            adapter=request.model_id, path=str(folder), origin=request.origin,
            horizon=len(request.forecast_dates), n_series=len(latest),
            disposition={"emitted": len(latest) * len(request.forecast_dates)},
            target_estimand=request.panel.target_estimand, frequency=request.panel.frequency,
        ),
        "last-values-json-v1",
    )


def unit_drift_predict(request):
    """A second synthetic custom model: last value plus one unit per forecast step."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.study_tuning import FitPredictResult

    result = last_value_predict(request)
    steps = {day: index + 1 for index, day in enumerate(request.forecast_dates)}
    part = request.output_dir / "predictions/part.parquet"
    rows = pq.read_table(part).to_pylist()
    for row in rows:
        row["yhat"] += steps[row["date"]]
    pq.write_table(pa.Table.from_pylist(rows), part)
    (request.output_dir / "model/drift.json").write_text('{"units_per_step": 1}')
    return FitPredictResult(result.forecasts, "unit-drift-json-v1")


def _validation_rendezvous(directory, model_id, *, timeout_seconds=90):
    """Make local validation overlap even when the second Ray worker starts slowly."""
    import time

    peer = {"constant-0": "constant-1", "constant-1": "constant-0"}[model_id]
    with (directory / model_id).open("x") as marker:
        marker.write(str(os.getpid()))
    deadline = time.monotonic() + timeout_seconds
    while not (directory / peer).is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"validation peer {peer} did not arrive; concurrent workers required")
        time.sleep(0.05)


def rendezvous_predict(request):
    from autoforecast.test_study_tuning import predict

    if request.purpose == "validation":
        _validation_rendezvous(
            pathlib.Path(os.environ["AUTOFORECAST_TEST_VALIDATION_RENDEZVOUS"]), request.model_id,
        )
    return predict(request)


def test_validation_rendezvous_waits_for_both_models(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        arrivals = [pool.submit(_validation_rendezvous, tmp_path, f"constant-{index}",
                                timeout_seconds=5) for index in (0, 1)]
        for arrival in arrivals:
            arrival.result(timeout=6)
    assert {path.name for path in tmp_path.iterdir()} == {"constant-0", "constant-1"}


def test_validation_rendezvous_refuses_a_single_worker(tmp_path):
    with pytest.raises(TimeoutError, match="concurrent workers required"):
        _validation_rendezvous(tmp_path, "constant-0", timeout_seconds=0)


def test_rendezvous_predict_skips_final_refit(monkeypatch):
    from autoforecast import test_study_tuning

    result = object()
    monkeypatch.setattr(test_study_tuning, "predict", lambda request: result)
    monkeypatch.delenv("AUTOFORECAST_TEST_VALIDATION_RENDEZVOUS", raising=False)
    assert rendezvous_predict(SimpleNamespace(purpose="refit")) is result


@pytest.mark.parametrize(
    "execution,parallel,policy",
    [("sequential", 1, "validation_winner"), ("local", 2, "all_models")],
)
def test_campaign_selects_validation_winner_and_replays_without_refitting(
    spark, tmp_path, monkeypatch, execution, parallel, policy
):
    manifest = freeze(tmp_path / "study", spark)
    declared = models(tmp_path)
    if execution == "local":
        rendezvous = tmp_path / "validation-rendezvous"
        rendezvous.mkdir()
        monkeypatch.setenv("AUTOFORECAST_TEST_VALIDATION_RENDEZVOUS", str(rendezvous))
        declared = [replace(model, entrypoint="autoforecast.test_study_campaign:rendezvous_predict")
                    for model in declared]
    with open_study(manifest, spark) as study:
        result = study.run_campaign(
            models=declared,
            output_dir=tmp_path / "campaign",
            execution=execution,
            max_parallel_models=parallel,
            final_evaluation=policy,
            timeout_seconds=600,
        )
        assert result["state"]["status"] == "completed", result
        summary = result["results"]
        assert summary["selected_model"] == "constant-1"
        assert set(summary["results"]) == ({"1"} if policy == "validation_winner" else {"0", "1"})
        assert len(summary["candidates"]) == 2
        workflow = load_forecast_workflow(summary["results"]["1"]["workflow_path"])
        assert workflow.testing is not None
        assert workflow.provenance["work_estimate"]["total_callback_fits"] == 1
        repeated = study.run_campaign(
            models=declared,
            output_dir=tmp_path / "campaign",
            execution=execution,
            max_parallel_models=parallel,
            final_evaluation=policy,
            timeout_seconds=600,
        )
        assert repeated["results"] == summary
        guide = pathlib.Path(result["campaign_path"]) / "RUN.md"
        assert "Execution status: **completed**" in guide.read_text()
        assert "state.json" in guide.read_text()
        if policy == "validation_winner":
            from pathlib import Path

            replay_options = dict(
                models=declared, output_dir=tmp_path / "campaign", execution=execution,
                max_parallel_models=parallel, final_evaluation=policy, timeout_seconds=600,
            )

            def rejects_changed_evidence():
                try:
                    study.run_campaign(**replay_options)
                except ValueError:
                    return True
                return False

            record_path = Path(result["campaign_path"]) / "models/candidate-0/validation/result.json"
            original = record_path.read_bytes()
            record = json.loads(original)
            try:
                record_path.write_text(json.dumps({**record, "pid": -1}))
                rejects_record = rejects_changed_evidence()
            finally:
                record_path.write_bytes(original)
            losing = load_forecast_workflow(record["workflow_path"])
            model_file = next(path for path in losing.validation["model_artifacts"].rglob("*") if path.is_file())
            model_file.write_text("changed losing-candidate model")
            rejects_model = rejects_changed_evidence()
            assert (rejects_record, rejects_model) == (True, True)
    if parallel == 2:
        root = __import__("pathlib").Path(result["campaign_path"])
        jobs = [
            json.loads(
                (root / "models" / f"candidate-{index}" / "validation/result.json").read_text()
            )
            for index in (0, 1)
        ]
        assert jobs[0]["pid"] != jobs[1]["pid"]
        assert max(job["started"] for job in jobs) < min(job["finished"] for job in jobs)
        assert {int(path.read_text()) for path in rendezvous.iterdir()} == {
            job["pid"] for job in jobs
        }


def test_worker_rejects_changed_request_and_import_code(tmp_path):
    path = tmp_path / "request.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        _read(path, "0" * 64)
    with pytest.raises(ValueError, match="differs on worker"):
        _entrypoint("autoforecast.test_study_tuning:predict", "0" * 64)


def test_worker_rejects_runtime_and_helper_changes_before_loading_model(tmp_path, monkeypatch):
    import hashlib

    from autoforecast import study_campaign

    runtime = study_campaign._runtime_identity()
    monkeypatch.setattr(study_campaign, "_entrypoint", lambda *_: pytest.fail("model must not load"))
    cases = [
        ({**runtime, "source_sha256": "0" * 64}, {}, "worker library"),
        (runtime, {"autoforecast.test_study_tuning": "0" * 64}, "model helper code"),
    ]
    for identity, helpers, message in cases:
        path = tmp_path / "request.json"
        path.write_text(json.dumps({"runtime": identity, "models": [{"source_modules": helpers}]}))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match=message):
            study_campaign._model_job(str(path), sha, 0, "validation", str(tmp_path / "outputs"), None)
    assert not (tmp_path / "outputs").exists()


def test_existing_local_ray_cluster_is_reused_and_remains_available(spark, tmp_path):
    import ray

    manifest = freeze(tmp_path / "study", spark)
    declared = models(tmp_path)
    declared = [replace(model, node_labels={"pool": "forecast"}) for model in declared]
    context = ray.init(address="local", num_cpus=2, include_dashboard=False, labels={"pool": "forecast"})
    try:
        address = context.address_info["gcs_address"]
        with open_study(manifest, spark) as study:
            result = study.run_campaign(
                models=declared, output_dir=tmp_path / "cluster-campaign",
                execution="cluster", ray_address=address, max_parallel_models=2,
                final_evaluation="validation_winner", timeout_seconds=600,
            )
        assert result["state"]["status"] == "completed", result
        assert result["results"]["selected_model"] == "constant-1"
        assert ray.is_initialized()
        assert any(node["Alive"] and node["Labels"].get("pool") == "forecast" for node in ray.nodes())
        request = json.loads(next((tmp_path / "cluster-campaign").glob("request-*/request.json")).read_text())
        assert all(model["node_labels"] == {"pool": "forecast"} for model in request["models"])
    finally:
        ray.shutdown()


def fail_once(request):
    """Inject one worker failure into the synthetic recovery acceptance test."""
    import os
    from pathlib import Path

    from autoforecast.test_study_tuning import predict

    marker = Path(os.environ["AUTOFORECAST_TEST_FAILURE_MARKER"])
    if not marker.exists():
        peer = os.environ.get("AUTOFORECAST_TEST_COMPLETED_PEER")
        if peer:
            import signal
            import time

            import psutil

            deadline = time.monotonic() + 90
            while not list(Path(peer).glob("custom-models-*/models/candidate-0/validation/result.json")) and time.monotonic() < deadline:
                time.sleep(0.1)
            if not list(Path(peer).glob("custom-models-*/models/candidate-0/validation/result.json")):
                raise RuntimeError("completed peer was not retained before the crash")
            process = psutil.Process()
            marker.write_text(json.dumps([
                {"pid": child.pid, "created": child.create_time()}
                for child in [process, *process.children(recursive=True)]
            ]))
            os.kill(os.getpid(), signal.SIGKILL)
        marker.write_text("injected failure")
        raise RuntimeError("injected model failure")
    return predict(request)






def test_campaign_rejects_impossible_sequential_resources_before_fitting(tmp_path):
    from autoforecast.study_campaign import run_study_campaign

    declared = models(tmp_path)
    with pytest.raises(ValueError, match="GPU reservations require"):
        run_study_campaign(
            None, models=[replace(declared[0], gpus=1)], output_dir=tmp_path / "rejected"
        )
    with pytest.raises(ValueError, match="CPU request exceeds"):
        run_study_campaign(
            None, models=[replace(declared[0], cpus=10**9)], output_dir=tmp_path / "rejected"
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("execution,expected_memory", [("local", 12.0), ("cluster", 2.0)])
def test_campaign_reserves_only_the_largest_possible_concurrent_jobs(
    spark, tmp_path, monkeypatch, execution, expected_memory
):
    from autoforecast.campaign import CampaignStore, CampaignSupervisor

    manifest = freeze(tmp_path / "study", spark)
    declared = models(tmp_path)
    declared = [replace(model, memory_gib=memory) for model, memory in zip(declared, (4.0, 6.0), strict=True)]
    # Exercise real preflight and plan persistence; inspect the admission contract
    # without launching model jobs or connecting to a cluster.
    monkeypatch.setattr(CampaignSupervisor, "run", lambda self: self.store.state())
    with open_study(manifest, spark) as study:
        result = study.run_campaign(
            models=declared, output_dir=tmp_path / "campaign", execution=execution,
            max_parallel_models=5,
            ray_address="unused-plan-only-address" if execution == "cluster" else None,
        )
    store = CampaignStore.from_root(result["campaign_path"])
    assert [stage.resources.memory_gib for stage in store.plan.stages] == [expected_memory] * 2


@pytest.mark.parametrize("measurement_fails", [False, True])
def test_memory_watch_stops_over_budget_or_unmeasurable_job(tmp_path, monkeypatch, measurement_fails):
    import os
    import signal
    import threading

    from autoforecast import campaign, study_campaign

    def measure(_pid):
        if measurement_fails:
            raise RuntimeError("test measurement failure")
        return 2 * 1024**3

    signals = []
    monkeypatch.setattr(campaign, "_process_tree_rss", measure)
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    study_campaign._watch_memory(threading.Event(), tmp_path, 1.0)
    assert signals == [(os.getpid(), signal.SIGTERM)]
    records = list(tmp_path.glob("memory-limit-*.json"))
    assert len(records) == 1
    reason = json.loads(records[0].read_text())["reason"]
    assert ("measurement failed" if measurement_fails else "exceeded") in reason


def blocking_fit(request):
    """Record real worker identities, then wait beyond the phase deadline."""
    import os
    import time
    from pathlib import Path

    import psutil

    worker = psutil.Process()
    Path(os.environ["AUTOFORECAST_TEST_BLOCK_MARKER"]).write_text(
        json.dumps([
            {"pid": process.pid, "created": process.create_time()}
            for process in [worker, *worker.children(recursive=True)]
        ])
    )
    time.sleep(180)
    raise AssertionError("the deadline did not cancel this model")


def test_local_timeout_stops_model_and_its_spark_children(spark, tmp_path, monkeypatch):
    import time

    import psutil

    manifest = freeze(tmp_path / "study", spark)
    declared = models(tmp_path)
    marker = tmp_path / "worker-identities.json"
    monkeypatch.setenv("AUTOFORECAST_TEST_BLOCK_MARKER", str(marker))
    with open_study(manifest, spark) as study:
        result = study.run_campaign(
            models=[replace(declared[0], entrypoint="autoforecast.test_study_campaign:blocking_fit")],
            output_dir=tmp_path / "campaign",
            execution="local",
            timeout_seconds=60,
        )
    assert result["state"]["status"] == "failed", result
    guide = pathlib.Path(result["campaign_path"]) / "RUN.md"
    assert "Execution status: **failed**" in guide.read_text()
    assert marker.exists(), "the test must reach the model before its timeout"
    identities = json.loads(marker.read_text())

    def surviving_processes():
        live = []
        for record in identities:
            try:
                process = psutil.Process(record["pid"])
                if process.create_time() == record["created"] and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process)
            except psutil.NoSuchProcess:
                pass
        return live

    deadline = time.monotonic() + 20
    while surviving_processes() and time.monotonic() < deadline:
        time.sleep(0.2)
    survivors = surviving_processes()
    # Clean up only authenticated test-owned processes if the acceptance check fails.
    for process in survivors:
        process.kill()
    assert not survivors, f"cancelled model left live processes: {[p.pid for p in survivors]}"


class _RayRecorder:
    def __init__(self, nodes, failed=(), timeout=False):
        self.resources = nodes
        self.failed = failed
        self.timeout = timeout
        self.inflight = set()
        self.peak = 0
        self.cancelled = []
        self.submissions = []
        self.worker_arguments = []
        self.closed = False

    def is_initialized(self):
        return False

    def init(self, **kwargs):
        self.init_options = kwargs

    def nodes(self):
        return [{"Alive": True, "Resources": r} for r in self.resources]

    def remote(self, **kwargs):
        self.task_options = kwargs
        return lambda fn: self

    def options(self, **kwargs):
        self.latest_options = kwargs

        class Submission:
            def remote(inner, *args, **kwargs):
                assert isinstance(kwargs["parent_attempt"], str)
                self.worker_arguments.append((args, kwargs.copy()))
                index = args[2]
                self.inflight.add(index)
                self.peak = max(self.peak, len(self.inflight))
                self.submissions.append((index, self.latest_options.copy()))
                return index

        return Submission()

    def wait(self, refs, **kwargs):
        if self.timeout:
            raise TimeoutError("injected timeout")
        return [refs[0]], refs[1:]

    def get(self, index):
        self.inflight.remove(index)
        if index in self.failed:
            raise RuntimeError("injected worker failure")
        return {"index": index}

    def cancel(self, ref, force):
        self.cancelled.append((ref, force))
        self.inflight.remove(ref)

    def shutdown(self):
        self.closed = True


@pytest.fixture
def ray_phase(tmp_path, monkeypatch):
    request = {
        "models": [{"id": f"m{i}", "cpus": 1, "gpus": 0, "memory_gib": 1} for i in range(3)],
        "execution": "cluster",
        "ray_address": "test-only",
        "max_parallel_models": 2,
        "timeout_seconds": 60,
        "final_evaluation": "all_models",
    }
    monkeypatch.setattr(study_campaign, "_read", lambda *args: request)
    monkeypatch.setattr(
        study_campaign,
        "_load_model_result",
        lambda path, *args, **kwargs: {"index": int(path.parts[-3].split("-")[-1])},
    )

    def run(ray):
        monkeypatch.setitem(sys.modules, "ray", ray)
        study_campaign._run_phase(
            "request", "sha", "validation", str(tmp_path / "campaign"), str(tmp_path)
        )

    return request, run, tmp_path


def test_three_jobs_respect_two_job_cap(ray_phase, monkeypatch):
    monkeypatch.setenv("AUTOFORECAST_LOG_LEVEL", "WARNING")
    request, run, _ = ray_phase
    ray = _RayRecorder([{"CPU": 8, "GPU": 0, "memory": 8 * 1024**3}])
    run(ray)
    assert ray.peak == 2 and len(ray.submissions) == 3 and ray.closed
    assert len(ray.worker_arguments) == 3
    assert all(call[1]["parent_attempt"] for call in ray.worker_arguments)
    for _, options in ray.submissions:
        assert "parent_attempt" not in options
        assert (
            options["num_cpus"] == 1 and options["num_gpus"] == 0 and options["memory"] == 1024**3
        )
        assert options["runtime_env"]["env_vars"] == {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "AUTOFORECAST_LOG_LEVEL": "WARNING",
        }


@pytest.mark.parametrize("resource,value", [("cpus", 3), ("gpus", 2), ("memory_gib", 5)])
def test_each_model_must_fit_one_node(ray_phase, resource, value):
    request, run, _ = ray_phase
    request["models"] = [{**request["models"][0], resource: value}]
    ray = _RayRecorder([{"CPU": 2, "GPU": 1, "memory": 4 * 1024**3}] * 2)
    with pytest.raises(ValueError, match="no worker node"):
        run(ray)
    assert ray.submissions == [] and ray.closed


def test_ray_failure_is_retained_and_prevents_success(ray_phase):
    _, run, root = ray_phase
    ray = _RayRecorder([{"CPU": 8, "GPU": 0, "memory": 8 * 1024**3}], failed={1})
    with pytest.raises(RuntimeError, match="one or more"):
        run(ray)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["failures"] == {"1": "injected worker failure"}
    assert set(summary["results"]) == {"0", "2"} and ray.closed


def test_timeout_cancels_all_pending_ray_jobs(ray_phase):
    _, run, _ = ray_phase
    ray = _RayRecorder([{"CPU": 8, "GPU": 0, "memory": 8 * 1024**3}], timeout=True)
    with pytest.raises(TimeoutError, match="injected timeout"):
        run(ray)
    assert set(ray.cancelled) == {(0, True), (1, True)} and ray.closed


@pytest.mark.parametrize("kind", ["entrypoint", "helper"])
def test_changed_module_bytes_invalidate_captured_identity(tmp_path, monkeypatch, kind):
    name = "review_" + kind
    path = tmp_path / (name + ".py")
    path.write_text("def fit(_worker_request): return None\n")
    # Track the cache entry so fixture cleanup also removes the module imported below.
    monkeypatch.setitem(sys.modules, name, None)
    del sys.modules[name]
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    entry = name + ":fit"
    expected = (
        study_campaign._entrypoint(entry)[1]
        if kind == "entrypoint"
        else study_campaign._module_hash(name)
    )
    path.write_text("def fit(_worker_request): return 1\n")
    if kind == "entrypoint":
        with pytest.raises(ValueError, match="differs on worker"):
            study_campaign._entrypoint(entry, expected)
    else:
        assert study_campaign._module_hash(name) != expected


def test_changed_runtime_source_bytes_change_identity(tmp_path, monkeypatch):
    folder = tmp_path / "autoforecast"
    folder.mkdir()
    fake = folder / "study_campaign.py"
    fake.write_text("value=1\n")
    monkeypatch.setattr(study_campaign, "__file__", str(fake))
    before = study_campaign._runtime_identity()
    fake.write_text("value=2\n")
    after = study_campaign._runtime_identity()
    assert before["source_sha256"] != after["source_sha256"]
    assert before["packages"] == after["packages"] and before["python"] == after["python"]


def _worker_request(tmp_path, record):
    path = tmp_path / "_worker_request.json"
    path.write_text(json.dumps(record))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_changed_request_bytes_rejected_after_capture(tmp_path):
    path, sha = _worker_request(tmp_path, {"models": []})
    assert study_campaign._read(path, sha) == {"models": []}
    pathlib.Path(path).write_text('{"models":[1]}')
    with pytest.raises(ValueError, match="identity changed"):
        study_campaign._read(path, sha)


def test_worker_dependency_mismatch_stops_before_study_or_job(tmp_path, monkeypatch):
    monkeypatch.setattr(study_campaign, "_runtime_identity", lambda: {})
    monkeypatch.setattr(
        study_campaign, "_entrypoint", lambda *a: (lambda r: pytest.fail("callback"), "a" * 64)
    )
    monkeypatch.setattr(
        study_campaign, "verify_benchmark", lambda *a: pytest.fail("study before dependency check")
    )
    model = {
        "source_modules": {},
        "entrypoint": "unused:fit",
        "code_sha256": "a" * 64,
        "implementation": {"dependencies": {"pyarrow": "0.invalid"}},
    }
    path, sha = _worker_request(tmp_path, {"runtime": {}, "models": [model]})
    with pytest.raises(ValueError, match="worker dependency differs"):
        study_campaign._model_job(path, sha, 0, "validation", str(tmp_path / "outputs"), None)
    assert not (tmp_path / "outputs").exists()


def test_worker_study_digest_mismatch_stops_before_job(spark, tmp_path, monkeypatch):
    manifest = freeze(tmp_path / "study", spark)
    monkeypatch.setattr(study_campaign, "_runtime_identity", lambda: {})
    monkeypatch.setattr(
        study_campaign, "_entrypoint", lambda *a: (lambda r: pytest.fail("callback"), "a" * 64)
    )
    model = {
        "source_modules": {},
        "entrypoint": "unused:fit",
        "code_sha256": "a" * 64,
        "implementation": {"dependencies": {}},
    }
    path, sha = _worker_request(
        tmp_path,
        {"runtime": {}, "models": [model], "study": {"path": str(manifest), "sha256": "0" * 64}},
    )
    with pytest.raises(ValueError, match="changed|identity|SHA|digest"):
        study_campaign._model_job(path, sha, 0, "validation", str(tmp_path / "outputs"), None)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("defect", ["metric", "policy", "windows", "model_id"])
def test_incompatible_comparison_rejected_before_preflight(tmp_path, monkeypatch, defect):
    from autoforecast import study_preflight

    declared = [replace(m, fixed_parameters=None) for m in models(tmp_path)]
    path = pathlib.Path(declared[1].config)
    value = yaml.safe_load(path.read_text())
    if defect == "metric":
        value["selection"]["metric"] = "rmse"
    elif defect == "policy":
        value["series_policy"] = "use_available"
    elif defect == "model_id":
        value["model"]["id"] = "constant-0"
    else:
        value["selection"] = {"stage": "tuning", "origins": ["2024-01-03"], "metric": "mae"}
    path.write_text(yaml.safe_dump(value))
    study = SimpleNamespace(
        recipe={
            "metric": "mae",
            "frequency": "D",
            "validation_origins": ["2024-01-07"],
            "validation_horizon": 2,
            "tuning_windows": [{"origin": "2024-01-03", "horizon": 2}],
        }
    )
    monkeypatch.setattr(
        study_preflight,
        "preflight_study",
        lambda *a: pytest.fail("incompatible model declarations reached preflight"),
    )
    with pytest.raises(
        ValueError,
        match={
            "metric": "same comparison metric",
            "policy": "same comparison metric",
            "windows": "same selection windows",
            "model_id": "distinct profile model ID",
        }[defect],
    ):
        study_campaign.run_study_campaign(
            study, models=declared, output_dir=tmp_path / "study_campaign"
        )
    assert not (tmp_path / "study_campaign").exists()


def test_model_job_starts_monitor_before_fit_and_applies_resources(tmp_path, monkeypatch):
    from pyspark.sql import SparkSession

    from autoforecast import study

    for key in ("PYSPARK_PYTHON", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    state = {"started": False, "stopped": False, "joined": False, "config": {}}

    class Monitor:
        def __init__(self, target, args, daemon):
            assert target is study_campaign._watch_memory
            assert args[2] == 4 and daemon
            self.stop = args[0]

        def start(self):
            state["started"] = True

        def join(self, timeout):
            state["joined"] = True
            assert self.stop.is_set()

    class Spark:
        def stop(self):
            state["stopped"] = True

    class Builder:
        def master(self, value):
            state["master"] = value
            return self

        def appName(self, value):
            return self

        def config(self, name, value):
            state["config"][name] = value
            return self

        def getOrCreate(self):
            assert state["started"], "Spark started without a memory monitor"
            return Spark()

    @contextlib.contextmanager
    def fake_study(*args, **kwargs):
        def run(**options):
            assert state["started"], "fit ran without a memory monitor"
            assert state["master"] == "local[3]"
            assert 0 < int(state["config"]["spark.driver.memory"][:-1]) <= 4 * 1024
            assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "1"
            root = tmp_path / "workflow"
            root.mkdir()
            (root / "artifact_manifest.json").write_text("{}")
            return SimpleNamespace(path=root, study_id="study", run_id="workflow")

        yield SimpleNamespace(run_forecast_workflow=run)

    model = {
        "id": "test",
        "source_modules": {},
        "entrypoint": "test:fit",
        "code_sha256": "a" * 64,
        "implementation": {"dependencies": {}},
        "memory_gib": 4,
        "cpus": 3,
        "profile": {},
        "fixed_parameters": {},
    }
    request = {
        "runtime": {},
        "models": [model],
        "study": {"path": str(tmp_path / "registry/study/benchmark.json"), "sha256": "a" * 64},
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(study_campaign, "_runtime_identity", lambda: {})
    monkeypatch.setattr(study_campaign, "_entrypoint", lambda *a: (lambda r: None, "a" * 64))
    monkeypatch.setattr(study_campaign, "verify_benchmark", lambda *a: None)
    monkeypatch.setattr(study_campaign, "threading",
                        SimpleNamespace(Thread=Monitor, Event=__import__("threading").Event))
    monkeypatch.setattr(SparkSession, "builder", Builder())
    monkeypatch.setattr(study, "load_study", fake_study)
    monkeypatch.setattr(
        study_campaign, "_load_model_result", lambda p, *a, **kw: json.loads(p.read_text())
    )
    study_campaign._model_job(
        str(path), sha, 0, "validation", str(tmp_path / "study_campaign"), None
    )
    assert state["stopped"] and state["joined"]


@pytest.fixture(scope="module")
def retained_handoff(spark, tmp_path_factory):
    from autoforecast.test_study_workflow import register, workflow_evidence

    # Exactly A/B from the shared synthetic fixture; no series are excluded.
    manifest = register(tmp_path_factory.mktemp("handoff-study"), spark, count=1)
    return workflow_evidence.__wrapped__(manifest, spark, tmp_path_factory)


@pytest.fixture
def valid_final_result(tmp_path, retained_handoff):
    validation, run, _ = retained_handoff
    profile = yaml.safe_load(run.search_config.read_bytes())
    model = {
        "id": profile["model"]["id"],
        "implementation": run.provenance["implementation"],
        "profile": profile,
        "fixed_parameters": {},
    }
    request = {"study": run.provenance["study"], "models": [model]}
    record = {
        "request_sha256": study_campaign.digest(request),
        "model_id": model["id"],
        "phase": "final",
        "workflow_path": str(run.path),
        "workflow_sha256": hashlib.sha256(
            (run.path / "artifact_manifest.json").read_bytes()
        ).hexdigest(),
    }
    checkpoint = {
        "workflow_path": run.provenance["validation_run"]["path"],
        "workflow_sha256": run.provenance["validation_run"]["manifest_sha256"],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(record))
    assert (
        study_campaign._load_model_result(
            path, request, model, "final", validation_record=checkpoint
        )
        == record
    )
    return path, request, model, record, checkpoint, validation.path


@pytest.mark.parametrize(
    "case",
    [
        "request",
        "model",
        "phase",
        "study",
        "implementation",
        "checkpoint_path",
        "checkpoint_sha",
        "checkpoint_absent",
        "profile",
        "parameters",
        "manifest",
        "wrong_workflow_phase",
    ],
)
def test_final_result_is_bound_to_campaign_and_exact_checkpoint(valid_final_result, case):
    path, request, model, record, checkpoint, validation_path = copy.deepcopy(valid_final_result)
    if case == "request":
        record["request_sha256"] = "0" * 64
    elif case == "model":
        record["model_id"] = "another-model"
    elif case == "phase":
        record["phase"] = "validation"
    elif case == "study":
        request["study"]["sha256"] = "0" * 64
    elif case == "implementation":
        model["implementation"]["version"] = "2"
    elif case == "checkpoint_path":
        checkpoint["workflow_path"] = str(path.parent / "another-checkpoint")
    elif case == "checkpoint_sha":
        checkpoint["workflow_sha256"] = "0" * 64
    elif case == "checkpoint_absent":
        checkpoint = None
    elif case == "profile":
        model["profile"]["hpo"]["seed"] = 123
    elif case == "parameters":
        model["fixed_parameters"] = {"model.level": 9.0}
    elif case == "manifest":
        record["workflow_sha256"] = "0" * 64
    else:
        record["workflow_path"] = str(validation_path)
    if case in {"study", "implementation", "profile", "parameters"}:
        record["request_sha256"] = study_campaign.digest(request)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        study_campaign._load_model_result(
            path, request, model, "final", validation_record=checkpoint
        )


@pytest.mark.parametrize(
    "defect", ["contract", None, float("nan"), float("inf"), -float("inf"), True]
)
def test_final_refuses_incomparable_or_unavailable_validation(tmp_path, monkeypatch, defect):
    request = {
        "models": [{"id": "a"}, {"id": "b"}],
        "metric": "mae",
        "final_evaluation": "validation_winner",
        "execution": "sequential",
    }
    records = {str(i): {"workflow_path": str(i)} for i in range(2)}
    workflows = {
        str(i): SimpleNamespace(
            validation={"evaluation_contract": {"cohort": "same"}, "score": {"mae": float(i + 1)}}
        )
        for i in range(2)
    }
    if defect == "contract":
        workflows["1"].validation["evaluation_contract"] = {"cohort": "other"}
    else:
        workflows["1"].validation["score"]["mae"] = defect
    monkeypatch.setattr(study_campaign, "_read", lambda *a: request)
    monkeypatch.setattr(study_campaign, "_validation_results", lambda *a: records)
    monkeypatch.setattr(study_campaign, "load_forecast_workflow", lambda p: workflows[p])
    monkeypatch.setattr(
        study_campaign,
        "_model_job",
        lambda *a: pytest.fail("unusable comparison reached final fit"),
    )
    with pytest.raises(ValueError, match="contracts differ|objective is unavailable"):
        study_campaign._run_phase(
            "request", "sha", "final", str(tmp_path / "campaign"), str(tmp_path)
        )
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("completed,tracking", [(True, True), (True, False), (False, True)])
def test_campaign_exports_only_verified_completed_results(
    tmp_path, monkeypatch, completed, tracking
):
    from autoforecast import run_mlflow, study_preflight

    root = tmp_path / "campaign"
    root.mkdir()
    final_record = {"workflow_path": str(tmp_path / "retained-workflow")}
    summary = root / "stages/final/attempt-0001/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"results": {"0": final_record}}))
    state = {
        "status": "completed" if completed else "failed",
        "stages": [
            {"id": "validation", "status": "completed"},
            {"id": "final", "status": "completed" if completed else "failed"},
        ]
    }
    store = SimpleNamespace(root=root)
    monkeypatch.setattr(study_campaign.CampaignStore, "open_or_create", lambda *a, **k: store)
    monkeypatch.setattr(
        study_campaign, "CampaignSupervisor", lambda s: SimpleNamespace(run=lambda: state)
    )
    monkeypatch.setattr(study_campaign, "_output_root", lambda *a: root)
    monkeypatch.setattr(study_campaign, "_runtime_identity", lambda: {})
    monkeypatch.setattr(
        study_preflight,
        "preflight_study",
        lambda *a, output_dir: SimpleNamespace(study_path=tmp_path / "benchmark.json", study_sha256="a" * 64),
    )
    verified = []
    exported = []
    monkeypatch.setattr(study_campaign, "_validation_results", lambda *a: {"0": {}})

    def verify(*a, **k):
        verified.append("checked")
        return final_record

    monkeypatch.setattr(study_campaign, "_load_model_result", verify)

    def export(path, **options):
        assert verified == ["checked"]
        exported.append((path, options))
        return {
            "validation": run_mlflow.RunMLflowLink("sqlite:///synthetic", "1", "v"),
            "test": run_mlflow.RunMLflowLink("sqlite:///synthetic", "1", "t"),
        }

    monkeypatch.setattr(run_mlflow, "export_forecast_workflow", export)
    study = SimpleNamespace(
        recipe={
            "metric": "mae",
            "frequency": "D",
            "validation_origins": ["2024-01-07"],
            "validation_horizon": 2,
        }
    )
    options = {"tracking_uri": "sqlite:///synthetic", "experiment_id": "1"} if tracking else {}
    result = study_campaign.run_study_campaign(
        study, models=models(tmp_path), output_dir=root, **options
    )
    if completed and tracking:
        assert exported == [(final_record["workflow_path"], options)]
        assert result["mlflow_links"]["0"]["test"]["run_id"] == "t"
    else:
        assert exported == [] and "mlflow_links" not in result


@pytest.mark.parametrize("labels", [[], {"": "pool"}, {"pool": ""}, {"pool": 1}, {1: "pool"}])
def test_node_labels_reject_invalid_declarations_before_preflight(tmp_path, labels):
    declared = models(tmp_path)
    with pytest.raises(ValueError, match="node_labels must"):
        study_campaign.run_study_campaign(
            None, models=[replace(declared[0], node_labels=labels)],
            output_dir=tmp_path / "output", execution="cluster", ray_address="test-only",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("execution", ["sequential", "local"])
def test_node_labels_require_existing_cluster(tmp_path, execution):
    declared = models(tmp_path)
    with pytest.raises(ValueError, match="node_labels require"):
        study_campaign.run_study_campaign(
            None, models=[replace(declared[0], node_labels={"pool": "forecast"})],
            output_dir=tmp_path / "output", execution=execution,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("compatible", [False, True])
def test_node_labels_and_resources_must_match_the_same_worker(ray_phase, monkeypatch, compatible):
    from ray.util.scheduling_strategies import In, NodeLabelSchedulingStrategy

    request, run, _ = ray_phase
    request["models"] = [{**request["models"][0], "node_labels": {"pool": "forecast"}}]
    ray = _RayRecorder([])
    monkeypatch.setattr(ray, "nodes", lambda: [
        {"Alive": True, "Resources": {"CPU": 8, "memory": 8 * 1024**3}, "Labels": {"pool": "other"}},
        {"Alive": True, "Resources": {"CPU": 8 if compatible else 0, "memory": 8 * 1024**3},
         "Labels": {"pool": "forecast"}},
    ])
    if compatible:
        run(ray)
        placement = ray.submissions[0][1]["scheduling_strategy"]
        assert isinstance(placement, NodeLabelSchedulingStrategy)
        assert len(placement.hard) == 1
        assert placement.hard[0].key == "pool"
        assert isinstance(placement.hard[0].operator, In)
        assert placement.hard[0].operator.values == ["forecast"]
    else:
        with pytest.raises(ValueError, match="resources and exact node labels"):
            run(ray)
        assert not ray.submissions
    assert ray.closed


@pytest.mark.parametrize("metric,expected", [("mae", "0"), ("rmse", "1")])
def test_final_selection_uses_requested_metric_when_rankings_disagree(tmp_path, monkeypatch, metric, expected):
    # Absolute errors [0, 4] beat [2.5, 2.5] by MAE, but lose by RMSE.
    request = {
        "models": [{"id": "spiky"}, {"id": "steady"}], "metric": metric,
        "final_evaluation": "validation_winner", "execution": "sequential",
    }
    records = {str(index): {"workflow_path": str(index)} for index in range(2)}
    workflows = {
        "0": SimpleNamespace(validation={"evaluation_contract": {"cohort": "same"},
                                       "score": {"mae": 2.0, "rmse": 8**0.5}}),
        "1": SimpleNamespace(validation={"evaluation_contract": {"cohort": "same"},
                                       "score": {"mae": 2.5, "rmse": 2.5}}),
    }
    for workflow in workflows.values():
        workflow.testing = {"score": {"mae": 100.0, "rmse": 100.0}}
    monkeypatch.setattr(study_campaign, "_read", lambda *args: request)
    monkeypatch.setattr(study_campaign, "_validation_results", lambda *args: records)
    monkeypatch.setattr(study_campaign, "load_forecast_workflow", lambda path: workflows[path])
    calls = []

    def fit(*args, parent_attempt):
        assert parent_attempt == str(tmp_path)
        calls.append(str(args[2]))
        return {"workflow_path": str(args[2])}

    monkeypatch.setattr(study_campaign, "_model_job", fit)
    monkeypatch.setattr(study_campaign, "_load_model_result", lambda *args, **kw: {"workflow_path": expected})
    study_campaign._run_phase("request", "sha", "final", str(tmp_path / "campaign"), str(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert calls == [expected]
    assert summary["selected_model"] == request["models"][int(expected)]["id"]
