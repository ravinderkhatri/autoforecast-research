"""Retained readers consume sealed bytes; failed executions own only their output."""

import contextlib
import hashlib
import json

import pytest
from mlops.provenance import ProvenanceError

from autoforecast import study_tuning as tuning
from autoforecast.experiments import load_completed_attempt
from autoforecast.hpo import HPOError, _study_lease
from autoforecast.results import read_tuning_results, read_workflow_results
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_tuning import implementation, open_study, predict
from autoforecast.test_study_tuning import spark as _spark
from autoforecast.test_study_workflow import profile, register


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


@pytest.fixture(scope="module")
def saved(tmp_path_factory, spark):
    # Population: two registered series, no exclusions, two horizon cells each.
    root = tmp_path_factory.mktemp("retained-boundaries")
    manifest = register(root / "input", spark)
    cfg = profile(root, selection={"stage": "validation"})
    failed_source = root / "failed-source"
    failed_source.mkdir()
    failed_cfg = profile(failed_source, selection={"stage": "tuning", "origins": "all"})
    calls = 0

    def fail_second(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("declared callback failure")
        return predict(request)

    with open_study(manifest, spark) as study:
        run = study.run_forecast_workflow(
            fit_predict=predict, implementation=implementation(), config=cfg,
            output_dir=root / "workflow", validation_only=True,
        )
        with pytest.raises(RuntimeError, match="declared callback failure") as caught:
            study.tune(fit_predict=fail_second, implementation=implementation(), config=failed_cfg,
                       output_dir=root / "failed")
    return manifest, cfg, run, caught.value.tuning_path


@pytest.mark.parametrize("consumer", [
    "workflow", "workflow_analysis", "tuning", "tuning_analysis",
    "workflow_configuration", "tuning_configuration", "failed_score", "failure_message",
    "both_workflow_scores", "workflow_stage", "workflow_provenance", "attempt_metadata",
])
def test_public_readers_reject_transient_unsealed_bytes(saved, monkeypatch, consumer):
    _, _, run, failed = saved
    selected = json.loads((run.tuning_path / "selection.json").read_text())["selected"][0]
    selected_attempt = run.tuning_path / selected["attempt_path"]
    root, relative = run.path, "result.json"
    def reader():
        return load_forecast_workflow(run.path)
    if consumer == "workflow_analysis":
        def reader():
            return read_workflow_results(run.path, include_tuning=False)
    elif consumer in {"tuning", "tuning_analysis", "tuning_configuration", "attempt_metadata"}:
        root, relative = selected_attempt, "evaluation/external-run.json"
        def reader():
            return tuning.load_tuning_result(run.tuning_path)
        if consumer == "tuning_analysis":
            def reader():
                return read_tuning_results(run.tuning_path)
        elif consumer == "tuning_configuration":
            relative = "metadata/winner-config.json"
            def reader():
                return tuning.load_selected_configuration(root / relative)
        elif consumer == "attempt_metadata":
            relative = "metadata/attempt.json"
    elif consumer == "workflow_configuration":
        relative = "metadata/winner-config.json"
        def reader():
            return tuning.load_selected_configuration(root / relative)
    elif consumer == "workflow_stage":
        relative = "validation/result.json"
    elif consumer == "workflow_provenance":
        relative = "metadata/provenance.json"
    elif consumer in {"failed_score", "failure_message"}:
        attempts = list(failed.glob("trials/*/attempts/*"))
        status = "failed"
        root = next(p for p in attempts if load_completed_attempt(p)[1]["status"] == status)
        relative = ("evaluation/failure.json" if consumer == "failure_message"
                    else "evaluation/external-run.json")
        def reader():
            return read_tuning_results(failed)
    before = (root / relative).read_bytes()
    original = tuning.read_regular_bytes
    hits = []

    def transient_bytes(directory, name):
        payload = original(directory, name)
        if directory == root and (name == relative or (
            consumer == "both_workflow_scores" and name == "validation/result.json"
        )):
            hits.append(name)
            value = json.loads(payload)
            if name == "result.json":
                value["validation"]["score"]["rmse"] = 999999.
            elif name == "validation/result.json":
                value["score"]["rmse"] = 999999.
            elif name == "evaluation/external-run.json":
                value["windows"][0]["score"]["rmse"] = 999999.
            elif name == "evaluation/failure.json":
                value["message"] = "unsealed message"
            else:
                value["unsealed_field"] = "unsealed settings"
            return json.dumps(value).encode()
        return payload

    monkeypatch.setattr(tuning, "read_regular_bytes", transient_bytes)
    with pytest.raises(HPOError, match="sealed bytes"):
        reader()
    assert hits
    assert (root / relative).read_bytes() == before


def test_manifest_bytes_must_match_verified_completion(saved, monkeypatch):
    from autoforecast import study_workflow

    root = saved[2].path
    original = study_workflow.read_regular_bytes

    def changed_manifest(directory, relative):
        payload = original(directory, relative)
        if relative == "artifact_manifest.json":
            value = json.loads(payload)
            value["artifacts"][0]["sha256"] = "0" * 64
            return json.dumps(value).encode()
        return payload

    monkeypatch.setattr(study_workflow, "read_regular_bytes", changed_manifest)
    with pytest.raises(HPOError, match="manifest differs"):
        load_forecast_workflow(root)


def test_completion_payload_must_match_captured_manifest(saved, monkeypatch):
    from autoforecast import study_workflow

    original = study_workflow.load_completed_run_bundle

    def changed_completion(root):
        completion = original(root)
        completion["summary"]["unsealed"] = True
        return completion

    monkeypatch.setattr(study_workflow, "load_completed_run_bundle", changed_completion)
    with pytest.raises(HPOError, match="sealed terminal payload"):
        load_forecast_workflow(saved[2].path)


@pytest.mark.parametrize("locked", [False, True])
def test_tuning_collision_cannot_change_another_search(saved, spark, monkeypatch, locked):
    manifest, cfg, run, _ = saved
    search = run.tuning_path
    before = {p.relative_to(search): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in search.rglob("*") if p.is_file()}
    monkeypatch.setattr(tuning, "new_run_id", lambda _: search.name)

    def forbidden(_):
        pytest.fail("collision must not fit")

    lease = _study_lease(search.parent.parent, search.name) if locked else contextlib.nullcontext()
    with open_study(manifest, spark) as study, lease:
        with pytest.raises(HPOError, match="already exists|already running") as caught:
            study.tune(fit_predict=forbidden, implementation=implementation(), config=cfg,
                       output_dir=search.parent.parent)
    assert not hasattr(caught.value, "tuning_path")
    assert {p.relative_to(search): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in search.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("execution", ["workflow", "tuning"])
@pytest.mark.parametrize("shape", ["empty", "nested_empty"])
@pytest.mark.parametrize("failure", [False, True])
def test_empty_optional_artifacts_preserve_callback_outcome(
    saved, spark, tmp_path, execution, shape, failure,
):
    manifest, cfg, _, _ = saved

    def callback(request):
        artifact = request.output_dir / "artifacts"
        (artifact / "nested" if shape == "nested_empty" else artifact).mkdir(parents=True)
        if failure:
            raise RuntimeError("declared callback failure")
        return predict(request)

    with open_study(manifest, spark) as study:
        kwargs = dict(fit_predict=callback, implementation=implementation(), config=cfg,
                      output_dir=tmp_path / "outputs")
        call = study.tune if execution == "tuning" else study.run_forecast_workflow
        if execution == "workflow":
            kwargs.update(fixed_parameters={}, validation_only=True)
        if failure:
            with pytest.raises(RuntimeError, match="declared callback failure"):
                call(**kwargs)
        else:
            result = call(**kwargs)
            windows = result.windows if execution == "tuning" else [result.validation]
            assert windows and all("artifacts" not in window for window in windows)


@pytest.mark.parametrize("shape", ["symlink", "fifo"])
def test_optional_artifacts_reject_unsafe_empty_tree(saved, spark, tmp_path, shape, monkeypatch):
    import os

    manifest, cfg, _, _ = saved
    retained_errors = []
    retain = tuning._retain_callback_artifacts

    def observe_rejection(work, output):
        try:
            return retain(work, output)
        except HPOError as exc:
            retained_errors.append(exc)
            assert not (output / "artifacts").exists()
            raise

    monkeypatch.setattr(tuning, "_retain_callback_artifacts", observe_rejection)

    def callback(request):
        artifact = request.output_dir / "artifacts"
        artifact.mkdir()
        if shape == "symlink":
            (artifact / "link").symlink_to(tmp_path, target_is_directory=True)
        else:
            os.mkfifo(artifact / "pipe")
        raise RuntimeError("callback failed")

    with open_study(manifest, spark) as study, pytest.raises(RuntimeError, match="callback failed"):
        study.run_forecast_workflow(
            fit_predict=callback, implementation=implementation(), config=cfg,
            fixed_parameters={}, output_dir=tmp_path / "outputs",
        )
    # Unsafe artifacts are still rejected; recovery must not mask the model failure.
    assert len(retained_errors) == 1
    assert "symlinks or special" in str(retained_errors[0])


@pytest.mark.parametrize("present", [False, True])
def test_unsealed_workflow_is_a_validation_error(tmp_path, present):
    root = tmp_path / "workflow"
    if present:
        root.mkdir()
    with pytest.raises(ProvenanceError):
        load_forecast_workflow(root)
