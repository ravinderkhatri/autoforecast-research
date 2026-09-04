"""Progress respects protected paths, context boundaries and measured work."""

import dataclasses
import io
import json
import logging
import pathlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mlops import _safe_files, _volume_files

from autoforecast import _execution as events
from autoforecast import benchmark, run_mlflow, study_campaign, study_tuning, study_workflow
from autoforecast import study_registration as registration
from autoforecast.integrations import delta
from autoforecast.study_registration import StudyRegistry
from autoforecast.tuning import loads_external_tuning_spec, materialize_runtime_parameters
from provenance import lineage


@pytest.fixture(autouse=True)
def reset_logger():
    logger = logging.getLogger("autoforecast.stages")
    saved = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    yield
    logger.handlers, logger.level, logger.propagate = saved
    events._OPERATION.set(None)


@pytest.mark.parametrize("kind", ["frozen", "sealed", "symlink"])
def test_workflow_never_writes_unvalidated_output_root(tmp_path, monkeypatch, kind):
    protected = tmp_path / "study"
    protected.mkdir()
    manifest = protected / "benchmark.json"
    manifest.write_text("{}")
    snapshots = protected / "snapshots"
    snapshots.mkdir()
    (snapshots / "part.parquet").write_bytes(b"protected bytes")
    if kind == "frozen":
        output = snapshots
    elif kind == "sealed":
        output = tmp_path / "completed"
        output.mkdir()
        (output / "artifact_manifest.json").write_text("{}")
    else:
        target = tmp_path / "target"
        target.mkdir()
        output = tmp_path / "link"
        output.symlink_to(target, target_is_directory=True)
    study = SimpleNamespace(
        _path=manifest,
        study_id="study",
        _record=lambda **_: {"sources": {"selection": {"path": str(snapshots)}}},
    )
    monkeypatch.setattr(
        study_workflow,
        "_config",
        lambda *a, **k: (b"profile", SimpleNamespace(model_id="model"), 1, 0, {}),
    )
    monkeypatch.setattr(study_workflow, "_implementation_record", lambda *_: {})
    monkeypatch.setattr(study_workflow, "resolve_implementation", lambda *a: {})
    monkeypatch.setattr(study_workflow, "_resolve_options", lambda *a: {})
    before = sorted(str(p.relative_to(output)) for p in output.rglob("*"))
    with pytest.raises(Exception, match="overlaps|symlink"):
        study_workflow.run_forecast_workflow(
            study,
            fit_predict=lambda _: None,
            implementation={},
            config="profile",
            output_dir=output,
        )
    after = sorted(str(p.relative_to(output)) for p in output.rglob("*"))
    assert after == before


def test_nested_callbacks_retain_trial_window_and_resolvable_parent(tmp_path):
    audit = tmp_path / "attempt.jsonl"
    stage_file = tmp_path / "trial-events.jsonl"
    with events.Execution(audit, ["workflow"], attempt_id="attempt", workflow_id="workflow"):
        with events._stage(audit, "workflow"):
            with events._stage(stage_file, "fit_predict", trial_id="trial-7", window="window-3"):
                events.note("Callback inputs", callback="example")
                with events.operation("Calculate requested scores"):
                    pass
    rows = [json.loads(x) for x in audit.read_text().splitlines()]
    callback = next(r for r in rows if r["stage"] == "Callback inputs")
    assert (callback.get("trial_id"), callback.get("window")) == ("trial-7", "window-3")
    score = next(
        r for r in rows if r["stage"] == "Calculate requested scores" and r["status"] == "START"
    )
    assert (score.get("trial_id"), score.get("window")) == ("trial-7", "window-3")
    parent_rows = [
        json.loads(line)
        for line in __import__("pathlib").Path(score["parent_audit"]).read_text().splitlines()
    ]
    assert any(r.get("operation_id") == score["parent_operation"] for r in parent_rows)


def test_overrun_is_not_rendered_as_known_percentage(tmp_path):
    with events.Execution(tmp_path / "events.jsonl", ["hash"]) as execution:
        with events._stage(execution.audit, "hash"):
            with events.operation("Digest growing file"):
                events.progress(completed=150, total=100, unit="bytes", phase="file sample")
                display = execution.render()
                assert "150.0%" not in display and "remaining=-50" not in display
                assert "total=unknown" in display


class TTY(io.StringIO):
    def isatty(self):
        return True


def test_reconfigure_from_tty_auto_to_plain_disables_ansi(tmp_path):
    stream = TTY()
    events.configure_logging(stream=stream, progress="auto")
    events.configure_logging(stream=stream, progress="plain")
    with events.Execution(tmp_path / "events.jsonl", ["work"]):
        with events._stage(tmp_path / "events.jsonl", "work"):
            pass
    assert "\x1b" not in stream.getvalue()


def test_reconfigure_from_plain_to_tty_auto_enables_inplace_output(tmp_path):
    stream = TTY()
    events.configure_logging(stream=stream, progress="plain")
    events.configure_logging(stream=stream, progress="auto")
    with events.Execution(tmp_path / "events.jsonl", ["work"]):
        with events._stage(tmp_path / "events.jsonl", "work"):
            pass
    assert "\x1b[2K" in stream.getvalue()


def test_reconfigure_notebook_auto_to_plain_stops_display_updates(tmp_path, monkeypatch):
    displays = []
    ipython = ModuleType("IPython")
    display_module = ModuleType("IPython.display")
    ipython.get_ipython = lambda: SimpleNamespace(kernel=True)
    display_module.display = lambda *a, **kw: displays.append("display")
    display_module.update_display = lambda *a, **kw: displays.append("update")
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    events.configure_logging(progress="auto")
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain")
    with events.Execution(tmp_path / "events.jsonl", ["work"]):
        with events._stage(tmp_path / "events.jsonl", "work"):
            pass
    assert not displays
    assert "Completed:" in stream.getvalue()


def test_failed_operation_start_restores_context(tmp_path, monkeypatch):
    import mlops._safe_files as files

    def fail(*_):
        raise OSError("unavailable audit")

    with events.Execution(tmp_path / "events.jsonl", ["work"]) as execution:
        with monkeypatch.context() as patch:
            patch.setattr(files, "append_audit_bytes", fail)
            with pytest.raises(OSError, match="unavailable audit"):
                with events.operation("nested"):
                    pass
        assert events._OPERATION.get() is None
        # A failed append has an unknown storage outcome; retain it for the panel.
        assert next(iter(execution.operations.values()))["status"] == "UNVERIFIED"
    assert events._OPERATION.get() is None


@pytest.mark.parametrize(
    "name", ["model.auth", "model.account_key", "model.private_key", "model.connection_string"]
)
def test_opaque_parameter_values_are_not_printed_or_replicated(tmp_path, name):
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="off")
    audit = tmp_path / "events.jsonl"
    secret = "example-credential-value"
    with events.Execution(audit, ["fit"]):
        with events._stage(audit, "fit"):
            events.note("Selected parameters", parameters={name: secret})
    assert secret not in stream.getvalue()
    assert secret not in audit.read_text()


def test_nested_operation_parent_is_a_durable_event(tmp_path):
    audit = tmp_path / "attempt.jsonl"
    with events.Execution(audit, ["workflow"]):
        with events._stage(audit, "workflow", trial_id="trial-7", window="window-3"):
            with events.operation("Calculate requested scores"):
                pass
    rows = [json.loads(x) for x in audit.read_text().splitlines()]
    score = next(
        r for r in rows if r["stage"] == "Calculate requested scores" and r["status"] == "START"
    )
    assert any(r.get("operation_id") == score["parent_operation"] for r in rows)


def test_growing_file_digest_never_displays_above_100_percent(tmp_path, monkeypatch):
    import provenance.lineage as lineage

    file = tmp_path / "growing.parquet"
    file.write_bytes(b"123")
    original = events.progress
    snapshots = []

    def record_progress(**kwargs):
        original(**kwargs)
        snapshots.append(events._CURRENT.get().render())

    monkeypatch.setattr(lineage, "progress", record_progress)
    monkeypatch.setattr(lineage.os, "fstat", lambda fd: SimpleNamespace(st_size=1))
    with events.Execution(tmp_path / "events.jsonl", ["hash"]):
        with events._stage(tmp_path / "events.jsonl", "hash"):
            with pytest.raises(lineage.LineageError, match="changed size"):
                lineage.directory_digest(file)
    assert snapshots
    assert all(
        "300.0%" not in snapshot and "remaining=-2" not in snapshot for snapshot in snapshots
    )


def test_campaign_coordinator_uses_actual_campaign_id(tmp_path, monkeypatch):
    from autoforecast import study_campaign

    campaign = tmp_path / "campaign-name-1234"
    attempt = campaign / "stages" / "validation" / "attempt-1"
    attempt.mkdir(parents=True)
    request = {
        "models": [{"id": "m1"}],
        "execution": "sequential",
        "final_evaluation": "all_models",
    }
    monkeypatch.setattr(study_campaign, "_read", lambda *a: request)
    monkeypatch.setattr(study_campaign, "_model_job", lambda *a, **kw: {"example": 1})
    monkeypatch.setattr(study_campaign, "_load_model_result", lambda *a, **kw: {"example": 1})
    study_campaign._run_phase("request", "request-sha", "validation", str(campaign), str(attempt))
    rows = [json.loads(x) for x in (attempt / "progress.jsonl").read_text().splitlines()]
    assert rows and {r["campaign_id"] for r in rows} == {campaign.name}


def test_valid_string_parameters_can_carry_sensitive_values():
    from autoforecast.tuning import loads_external_tuning_spec

    profile = {
        "schema_version": 2,
        "model": {"id": "example", "adapter": "external"},
        "hyperparameters": {
            "model.auth": {
                "default": "example-credential-value",
                "valid": {"kind": "categorical", "values": ["example-credential-value"]},
                "search": {"kind": "categorical", "values": ["example-credential-value"]},
            }
        },
    }
    assert (
        loads_external_tuning_spec(json.dumps(profile).encode()).parameters[0].name == "model.auth"
    )


def test_registration_config_quiet_level_applies_to_preparation_message(tmp_path, monkeypatch):
    from autoforecast import study_registration as registration

    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="off")
    registry = SimpleNamespace(
        root=tmp_path / "registry", scratch_root=tmp_path / "scratch", runtime={}, store=None
    )
    config = {
        "dataset": {},
        "recipe": {
            "source_keys": ["sku"],
            "training_start": "2024-01-01",
            "validation_origins": ["2024-02-01"],
            "history_missing": "error",
        },
        "source": {"kind": "parquet"},
        "runtime": {"log_level": "ERROR"},
    }
    monkeypatch.setattr(registration, "_configuration", lambda _: (config, tmp_path))
    monkeypatch.setattr(registration, "_register_study", lambda *a: "study-id")
    assert registration.register_study("settings.json", registry=registry) == "study-id"
    assert stream.getvalue() == ""


def test_direct_tune_reports_planned_work(tmp_path, monkeypatch):
    from autoforecast import study_tuning

    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="off")
    study = SimpleNamespace(
        study_id="study",
        recipe={"metric": "mae"},
        n_series=5,
        _snapshots=SimpleNamespace(name=str(tmp_path)),
        _record=lambda **k: {"study_id": "study"},
    )
    monkeypatch.setattr(
        study_tuning,
        "_config",
        lambda *a, **k: (b"profile", SimpleNamespace(model_id="model"), 3, 0, {}),
    )
    monkeypatch.setattr(study_tuning, "_implementation_record", lambda _: {})
    monkeypatch.setattr(study_tuning, "resolve_implementation", lambda *a: {})
    monkeypatch.setattr(
        study_tuning,
        "_resolve_options",
        lambda *a: {"windows": [{"origin": "x", "horizon": 1}], "stage": "tuning"},
    )
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)
    monkeypatch.setattr(
        study_tuning, "preflight_study", lambda *a, **k: SimpleNamespace(study_id="study")
    )
    monkeypatch.setattr(
        study_tuning,
        "_prepare_window",
        lambda *a, **k: (_ for _ in ()).throw(OSError("window stop")),
    )
    with pytest.raises(OSError, match="window stop"):
        study_tuning.tune_study(
            study, fit_predict=lambda _: None, config="profile", implementation={}
        )
    assert "Tuning work" in stream.getvalue()


def test_recovered_operation_is_not_shown_as_current_failure(tmp_path):
    audit = tmp_path / "events.jsonl"
    with events.Execution(audit, ["work"]) as execution:
        with events._stage(audit, "work"):
            with pytest.raises(OSError):
                with events.operation("download"):
                    raise OSError("retry")
            with events.operation("download"):
                pass
    assert execution.result == "PASS"
    assert "Operation: download" not in execution.render()


def test_two_live_terminal_attempts_remain_visible(tmp_path):
    stream = TTY()
    events.configure_logging(stream=stream, progress="auto")
    audit_a, audit_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with events.Execution(audit_a, ["stage-a"], attempt_id="attempt-a"):
        events._stage_event(audit_a, "stage-a", "START")
        with events.Execution(audit_b, ["stage-b"], attempt_id="attempt-b"):
            events._stage_event(audit_b, "stage-b", "START")
            last_live_line = stream.getvalue().rsplit("\r\x1b[2K", 1)[-1]
            assert "attempt-a" in last_live_line and "attempt-b" in last_live_line


def test_registration_summary_names_real_source_target_and_boundaries(tmp_path, monkeypatch):
    from autoforecast import study_registration as registration
    from autoforecast.test_study_registration import configuration

    config_path = configuration(tmp_path, actual_missing="mask")
    registry = SimpleNamespace(
        root=tmp_path / "registry", scratch_root=tmp_path / "scratch", runtime={}, store=None
    )
    monkeypatch.setattr(registration, "_register_study", lambda *a: "study-id")
    registration.register_study(config_path, registry=registry)
    audit = next((registry.root / "checks").glob("*/audit.jsonl"))
    rows = [json.loads(x) for x in audit.read_text().splitlines()]
    summary = next(row for row in rows if row["stage"] == "Registration inputs")
    assert str(tmp_path / "source.parquet") in json.dumps(summary)
    assert "value" in json.dumps(summary)
    assert "2024-01-07" in json.dumps(summary)
    assert "mask" in json.dumps(summary)


def test_logical_hash_outcome_includes_dataset_and_measured_rows(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.benchmark import parquet_content_hash

    source = tmp_path / "audit-original.parquet"
    pq.write_table(pa.table({"key": ["a", "b"]}), source)
    audit = tmp_path / "events.jsonl"
    with events.Execution(audit, ["sources"]):
        with events._stage(audit, "sources"):
            parquet_content_hash(source)
    rows = [json.loads(x) for x in audit.read_text().splitlines()]
    outcomes = [
        r
        for r in rows
        if r["stage"] == "Calculate logical row SHA (driver)" and r["status"] == "PASS"
    ]
    assert len(outcomes) == 1
    assert outcomes[0].get("dataset") == str(source)
    assert outcomes[0].get("rows") == 2


@pytest.mark.parametrize("after_lock", [False, True])
def test_reused_campaign_model_is_labelled_verified(tmp_path, monkeypatch, after_lock):
    from autoforecast import study_campaign

    model = {
        "id": "variant",
        "entrypoint": "sample:fit",
        "code_sha256": "a" * 64,
        "implementation": {"dependencies": {}},
        "source_modules": {},
        "fixed_parameters": {},
    }
    request = {
        "models": [model],
        "runtime": {},
        "study": {"path": str(tmp_path / "benchmark.json"), "sha256": "b" * 64},
    }
    monkeypatch.setattr(study_campaign, "_read", lambda *a: request)
    monkeypatch.setattr(study_campaign, "_runtime_identity", lambda: {})
    monkeypatch.setattr(study_campaign, "_entrypoint", lambda *a: (lambda _: None, "a" * 64))
    monkeypatch.setattr(study_campaign, "verify_benchmark", lambda *a: None)
    monkeypatch.setattr(
        study_campaign,
        "_load_model_result",
        lambda *a, **k: {"checked": True, "workflow_path": str(tmp_path / "saved-workflow")},
    )
    folder = tmp_path / "campaign" / "models" / "candidate-0" / "validation"
    folder.mkdir(parents=True)
    if after_lock:
        from autoforecast import campaign

        original = campaign._open_lock

        def open_after_other_writer(path):
            (folder / "result.json").write_text("{}")
            return original(path)

        monkeypatch.setattr(campaign, "_open_lock", open_after_other_writer)
    else:
        (folder / "result.json").write_text("{}")
    events.configure_logging(stream=io.StringIO(), progress="off")
    assert study_campaign._model_job(
        "request", "sha", 0, "validation", str(tmp_path / "campaign"), None
    ) == {"checked": True, "workflow_path": str(tmp_path / "saved-workflow")}
    audit = next((tmp_path / "campaign" / "progress" / "execution-attempts").glob("*.jsonl"))
    assert "reused" in audit.read_text().lower() and "verified" in audit.read_text().lower()


def test_publication_checks_study_before_creating_attempt(tmp_path):
    from autoforecast.study_registration import StudyRegistry

    error = FileNotFoundError("unknown study")

    def missing(_):
        raise error

    registry = SimpleNamespace(root=tmp_path / "protected", store=object(), definition=missing)
    with pytest.raises(FileNotFoundError) as caught:
        StudyRegistry.publish_results(registry, "unknown", tmp_path / "workflow")
    assert caught.value is error
    assert not registry.root.exists()


def test_pass_audit_failure_keeps_exception_and_closes_scope(tmp_path, monkeypatch):
    import mlops._safe_files as files

    original = files.append_audit_bytes
    error = OSError("completion audit unavailable")

    def append(path, payload):
        if json.loads(payload)["status"] == "PASS":
            raise error
        return original(path, payload)

    monkeypatch.setattr(files, "append_audit_bytes", append)
    audit = tmp_path / "events.jsonl"
    with pytest.raises(OSError) as caught:
        with events.Execution(audit, ["save"]) as run:
            with events._stage(audit, "save"):
                pass
    assert caught.value is error
    assert events._OPERATION.get() is None and events._CURRENT.get() is None
    # The body succeeded, but its completion event could not be verified.
    assert run.states["save"] == "UNVERIFIED"
    assert [json.loads(line)["status"] for line in audit.read_text().splitlines()] == ["START"]


def test_trial_window_context_restores_after_success_and_failure(tmp_path):
    audit = tmp_path / "events.jsonl"
    with events.Execution(audit, ["workflow"], workflow_id="workflow"):
        with events._stage(audit, "workflow"):
            for trial, window, failed in [("one", "first", False), ("two", "second", True)]:
                try:
                    with events._stage(
                        tmp_path / "fits.jsonl", "fit", trial_id=trial, window=window
                    ):
                        events.note("inside")
                        if failed:
                            raise ValueError("callback failed")
                except ValueError:
                    pass
                events.note("outside")
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    inside = [r for r in rows if r["stage"] == "inside"]
    assert [(r["trial_id"], r["window"]) for r in inside] == [("one", "first"), ("two", "second")]
    assert all("trial_id" not in r and "window" not in r for r in rows if r["stage"] == "outside")


def test_terminal_panel_tracks_wrapped_rows_and_redraws_after_message(tmp_path, monkeypatch):
    monkeypatch.setattr(events.shutil, "get_terminal_size", lambda: SimpleNamespace(columns=30))
    stream = TTY()
    events.configure_logging(stream=stream, progress="auto")
    console = events._console()
    execution = events.Execution(tmp_path / "events.jsonl", ["work"])
    text = "RUN example " + "long name " * 10 + "\u754c"
    console.panel(execution, text)
    panel = stream.getvalue().split("\x1b[2K")[-1]
    assert all(len(line) < 30 for line in panel.splitlines())
    assert "\\u754c" in panel.replace("\n", "")
    rows = len(panel.splitlines())
    stream.seek(0)
    stream.truncate()
    console.emit(
        logging.LogRecord("autoforecast.stages", logging.INFO, "", 0, "Saved result", (), None)
    )
    output = stream.getvalue()
    assert output.count("\x1b[1A") == rows - 1
    assert "Saved result\n" in output
    assert output.endswith(panel)
    assert console.live_rows == rows


@dataclasses.dataclass
class FakeResult:
    path: object
    status: str = "completed"
    run_id: str = "workflow-1"
    study_id: str = "study-1"
    publication: object = None


@pytest.mark.parametrize("publish", [False, True])
def test_run_study_accepts_successful_mlflow_links(tmp_path, monkeypatch, publish):
    path = tmp_path / "workflow"
    path.mkdir()
    (path / "artifact_manifest.json").write_text("{}")
    profile = tmp_path / "profile.json"
    profile.write_text('{"model": {"id": "example"}}')
    tracking = {
        "validation": run_mlflow.RunMLflowLink("https://tracking.example", "1", "validation"),
        "test": run_mlflow.RunMLflowLink("https://tracking.example", "1", "test"),
    }
    published = []

    @contextmanager
    def load_study(_study_id):
        yield SimpleNamespace(registration={"sha256": "verified"})

    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={"tracking": "mlflow", "tracking_uri": "https://tracking.example"},
        store=SimpleNamespace(validate_destinations=lambda: None, tables={}),
        _registration_path=lambda *_: None,
        load_study=load_study,
        publish_results=lambda *a, **k: published.append(k) or {"status": "published"},
    )
    monkeypatch.setattr(study_workflow, "run_forecast_workflow", lambda *a, **k: FakeResult(path))
    monkeypatch.setattr(run_mlflow, "resolve_study_experiment", lambda **_: "1")
    monkeypatch.setattr(run_mlflow, "export_forecast_workflow", lambda *a, **k: tracking)
    result = study_workflow.run_study(
        "study-1", lambda _: None, registry=registry, model_config=profile, publish=publish
    )
    assert result.status == "completed"
    assert bool(published) == publish
    if publish:
        assert published[0]["mlflow_links"] is tracking
    audit = result.publication["audit_path"]
    records = [json.loads(line) for line in pathlib.Path(audit).read_text().splitlines()]
    event = next(
        e for e in records if e["stage"] == "tracking_publication" and e["status"] == "PASS"
    )
    assert event["links"] == {stage: link.as_record() for stage, link in tracking.items()}


@pytest.mark.parametrize(
    "field", ["parameters", "selected_parameters", "model_parameters", "training_parameters"]
)
@pytest.mark.parametrize(
    "name", ["api_key", "password", "auth", "account_key", "connection_string", "custom_setting"]
)
def test_numeric_sensitive_parameter_is_not_logged(tmp_path, field, name):
    secret = 987654321
    domain = {"kind": "int", "low": secret, "high": secret, "step": 1}
    model_field = f"model.{name}"
    profile = {
        "schema_version": 2,
        "model": {"id": "example", "adapter": "external"},
        "hyperparameters": {model_field: {"default": secret, "valid": domain, "search": domain}},
    }
    spec = loads_external_tuning_spec(json.dumps(profile).encode())
    values = materialize_runtime_parameters(spec, {model_field: secret})
    stream = io.StringIO()
    logger = logging.getLogger("autoforecast.stages")
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    try:
        events.configure_logging(stream=stream, progress="off")
        audit = tmp_path / "events.jsonl"
        with events.Execution(audit, ["fit"]):
            with events._stage(audit, "fit"):
                events.note("Selected parameters", **{field: values["model"]})
        assert str(secret) not in audit.read_text()
        assert str(secret) not in stream.getvalue()
    finally:
        logger.handlers, logger.level, logger.propagate = previous


def test_merge_progress_does_not_write_one_audit_pair_per_chunk(tmp_path, monkeypatch):
    source = tmp_path / "data.parquet"
    pq.write_table(pa.table({"value": list(range(256))}), source)
    baseline = benchmark.parquet_content_hash(source)
    monkeypatch.setattr(benchmark, "_ROW_HASH_CHUNK_SIZE", 1)
    monkeypatch.setattr(benchmark, "_ROW_HASH_MERGE_FAN_IN", 4)
    audit = tmp_path / "events.jsonl"
    with events.Execution(audit, ["hash"]):
        with events._stage(audit, "hash"):
            digest = benchmark.parquet_content_hash(source)
    assert digest == baseline
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    merges = [r for r in rows if r["stage"] == "Merge sorted logical-row hash runs"]
    print("AUDIT_ROWS:", len(rows), "MERGE_AUDIT_ROWS:", len(merges))
    assert len(merges) <= 2


@pytest.mark.parametrize("kind", ["registration", "model"])
def test_final_panel_contains_result_and_evidence(tmp_path, kind):
    audit = tmp_path / "evidence" / "audit.jsonl"
    output = tmp_path / "results" / "workflow-42"
    kwargs = {"attempt_id": "attempt-1"}
    if kind == "model":
        kwargs["study_id"] = "study-42"
    with events.Execution(audit, ["work"], **kwargs) as execution:
        with events._stage(audit, "work"):
            if kind == "model":
                events.bind_identity(workflow_id="workflow-42", model_id="model-42")
                events.note("Workflow saved", workflow_path=str(output), evidence=str(audit))
            else:
                events.note("Study registered", study_id="study-42", evidence=str(audit))
            events.record_result(
                study_id="study-42",
                evidence=str(audit),
                workflow_path=str(output) if kind == "model" else None,
            )
    panel = execution.render()
    print("FINAL PANEL:\n" + panel)
    assert "study-42" in panel
    assert str(audit) in panel
    if kind == "model":
        assert str(output) in panel


@pytest.mark.parametrize("error", [ValueError("primary"), KeyboardInterrupt("cancel")])
def test_stage_reports_failure_even_when_failure_audit_is_unwritable(tmp_path, monkeypatch, error):
    stream = io.StringIO()
    logger = logging.getLogger("autoforecast.stages")
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    original = _safe_files.append_audit_bytes

    def append(path, payload):
        if json.loads(payload)["status"] in {"FAIL", "CANCELLED"}:
            raise OSError("disk unavailable")
        return original(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    try:
        events.configure_logging(stream=stream, progress="plain", interval=60)
        audit = tmp_path / "audit.jsonl"
        with pytest.raises(type(error)) as caught:
            with events.Execution(audit, ["copy"], attempt_id="attempt") as execution:
                with events._stage(audit, "copy"):
                    raise error
        assert caught.value is error
        panel = execution.render()
        print("FINAL PANEL:\n" + panel)
        print("CONSOLE:\n" + stream.getvalue())
        status = "fail" if isinstance(error, Exception) else "cancelled"
        assert f"copy ({status})" in panel
        assert "running=0" in panel
        assert f"{status.upper()} copy" in stream.getvalue()
        assert "Audit write failed: OSError" in panel
        assert all(json.loads(line)["status"] == "START" for line in audit.read_text().splitlines())
    finally:
        logger.handlers, logger.level, logger.propagate = previous


@pytest.mark.parametrize("mode", ["off", "plain"])
def test_reconfigure_during_attempt_does_not_resurrect_old_panel(tmp_path, mode):
    stream = TTY()
    logger = logging.getLogger("autoforecast.stages")
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    try:
        events.configure_logging(stream=stream, progress="auto")
        audit = tmp_path / "a.jsonl"
        with events.Execution(audit, ["fit"], attempt_id="finished") as run:
            with events._stage(audit, "fit"):
                events.configure_logging(stream=stream, progress=mode)
        events.configure_logging(stream=stream, progress="auto")
        stream.seek(0)
        stream.truncate()
        events.note("next message")
        print("AFTER RECONFIGURE:", stream.getvalue())
        assert run.display_id not in events._console().panels
        assert "finished" not in stream.getvalue()
    finally:
        logger.handlers, logger.level, logger.propagate = previous


def test_repeated_terminal_setup_preserves_both_active_attempts(tmp_path):
    stream = TTY()
    events.configure_logging(stream=stream)
    with events.Execution(tmp_path / "a.jsonl", ["a"]) as first:
        with events._stage(first.audit, "a"):
            with events.Execution(tmp_path / "b.jsonl", ["b"]) as second:
                with events._stage(second.audit, "b"):
                    events.configure_logging(stream=stream)
                    assert set(events._console().panels) == {first.display_id, second.display_id}
            assert set(events._console().panels) == {first.display_id}
    assert events._console().panels == {}


def test_switching_terminal_stream_clears_old_display(tmp_path):
    old, new = TTY(), TTY()
    events.configure_logging(stream=old)
    with events.Execution(tmp_path / "a.jsonl", ["work"]) as execution:
        with events._stage(execution.audit, "work"):
            before = old.getvalue()
            events.configure_logging(stream=new)
            assert old.getvalue()[len(before) :].endswith("\r\x1b[2K")
            assert events._console().panels == {}
    assert "PASS" in new.getvalue()
    assert events._console().panels == {}


@dataclasses.dataclass
class DerivedResult:
    path: pathlib.Path
    study_id: str
    provenance: dict
    status: str = "completed"
    run_id: str = "workflow-1"
    publication: object = None


@pytest.mark.parametrize(
    "kind", ["direct", "public-no-publish", "public-publish", "public-publication-fails"]
)
def test_derived_result_receipt_uses_effective_id(tmp_path, monkeypatch, kind):
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain", interval=60)
    output = tmp_path / "workflow"
    output.mkdir()
    (output / "artifact_manifest.json").write_text("{}")
    result = DerivedResult(
        output, "derived-study", {"study": {"path": str(tmp_path / "derived.json")}}
    )
    if kind == "direct":
        monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)

        @events.report_workflow
        def run(study, *, output_dir=None):
            events.bind_identity(workflow_id="workflow-1")
            return result

        actual = run(SimpleNamespace(study_id="parent-study"))
        assert actual is result
    else:
        profile = tmp_path / "profile.json"
        profile.write_text('{"model":{"id":"example"}}')

        @contextmanager
        def load_study(_):
            yield SimpleNamespace(registration={"sha256": "verified"})

        def publish(*args, **kwargs):
            assert args[0] == "derived-study"
            if kind == "public-publication-fails":
                raise OSError("publication failed")
            return {"status": "published"}

        registry = SimpleNamespace(
            root=tmp_path / "registry",
            scratch_root=tmp_path,
            runtime={},
            _registration_path=lambda *_: None,
            load_study=load_study,
            retain_derived=lambda *args: None,
            publish_results=publish,
            store=SimpleNamespace(validate_destinations=lambda: None, tables={}),
        )
        monkeypatch.setattr(study_workflow, "run_forecast_workflow", lambda *a, **k: result)
        if kind == "public-publication-fails":
            with pytest.raises(OSError):
                study_workflow.run_study(
                    "parent-study",
                    lambda _: None,
                    registry=registry,
                    model_config=profile,
                    publish=True,
                )
        else:
            actual = study_workflow.run_study(
                "parent-study",
                lambda _: None,
                registry=registry,
                model_config=profile,
                publish=kind == "public-publish",
            )
            assert actual.study_id == "derived-study"
    final = stream.getvalue().rsplit("RUN ", 1)[-1]
    print("FINAL PANEL:\n" + final)
    assert "study_id: derived-study" in final


@pytest.mark.parametrize("kind", ["plain", "tty", "notebook"])
def test_publication_identity_is_redacted_in_all_panel_modes(tmp_path, monkeypatch, kind):
    displayed = []
    stream = TTY() if kind == "tty" else io.StringIO()
    if kind == "notebook":
        ipython = ModuleType("IPython")
        display = ModuleType("IPython.display")
        ipython.get_ipython = lambda: SimpleNamespace(kernel=True)
        display.display = lambda data, **kw: displayed.append(data["text/plain"])
        display.update_display = lambda data, **kw: displayed.append(data["text/plain"])
        monkeypatch.setitem(sys.modules, "IPython", ipython)
        monkeypatch.setitem(sys.modules, "IPython.display", display)
        events.configure_logging()
    else:
        events.configure_logging(stream=stream, progress="auto" if kind == "tty" else "plain")
    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        store=None,
        definition=lambda _: ({}, "verified"),
    )

    def fail(*args, **kwargs):
        raise ValueError("invalid workflow")

    with pytest.raises(ValueError):
        StudyRegistry.publish_results(
            registry,
            "study-id",
            "https://user:sensitive-password@example.test/workflow?token=sensitive-token",
            destination=SimpleNamespace(publish_results=fail),
        )
    observed = "\n".join(displayed) if kind == "notebook" else stream.getvalue()
    audit = next((registry.root / "execution-attempts").glob("*.jsonl")).read_text()
    assert "sensitive-password" not in audit and "sensitive-token" not in audit
    print("PANEL:", observed)
    assert "sensitive-password" not in observed and "sensitive-token" not in observed
    raw = "https://user:sensitive-password@example.test/workflow?token=sensitive-token"
    with events.Execution(tmp_path / "display.jsonl", ["display"], workflow_path=raw) as execution:
        with events._stage(execution.audit, "display"):
            pass
    panel = execution.render()
    assert "sensitive-password" not in panel and "sensitive-token" not in panel
    assert "[redacted location]" in panel


def test_failed_hpo_trial_does_not_become_outer_run_id(tmp_path):
    audit = tmp_path / "driver.jsonl"
    with pytest.raises(ValueError):
        with events.Execution(
            audit, ["workflow"], study_id="study-1", workflow_id="workflow-1"
        ) as execution:
            with events._stage(audit, "workflow", study_id="study-1"):
                with events._stage(
                    tmp_path / "trial.jsonl",
                    "fit_predict",
                    study_id="study-1",
                    run_id="hpo-study-9",
                    trial_id="trial-3",
                ):
                    raise ValueError("fit failed")
    panel = execution.render()
    print("FINAL PANEL:\n" + panel)
    assert "run_id: hpo-study-9" not in panel


@pytest.mark.parametrize(
    "alias",
    [
        "auth",
        "token",
        "secret",
        "password",
        "credential",
        "api_key",
        "account_key",
        "private_key",
        "access_key",
        "connection_string",
        "source_auth",
        "privateKey",
        "observations",
    ],
)
def test_delta_alias_named_like_secret_keeps_table_and_version(tmp_path, monkeypatch, alias):
    class Result:
        def first(self):
            return {"id": "table-id", "location": "safe-location"}

    class Reader:
        def option(self, *args):
            return self

        def table(self, *args):
            return object()

    class Spark:
        read = Reader()
        catalog = SimpleNamespace(
            getTable=lambda _: SimpleNamespace(
                catalog="catalog", namespace=["schema"], name="source"
            )
        )

        def sql(self, query, *args, **kwargs):
            return Result()

    monkeypatch.setattr(delta, "latest_delta_version", lambda *_: 7)
    audit = tmp_path / "events.jsonl"
    with events.Execution(audit, ["sources"]):
        with events._stage(audit, "sources"):
            resolved = delta.read_sources(
                Spark(),
                queries={"primary": f"SELECT * FROM `{alias}`"},
                tables={alias: "catalog.schema.source"},
            )
    rows = [json.loads(x) for x in audit.read_text().splitlines()]
    event = next(row for row in rows if row["stage"] == "Resolved sources")
    print("ACTUAL ALIAS RECORD:", event["aliases"])
    assert resolved["primary"][1]["tables"][alias] == {
        "table": "catalog.schema.source",
        "version": 7,
    }
    assert {
        row["alias"]: {"table": row["table"], "version": row["version"]} for row in event["aliases"]
    }[alias] == {"table": "catalog.schema.source", "version": 7}


@pytest.mark.parametrize(
    "alias",
    [
        "auth",
        "token",
        "secret",
        "password",
        "credential",
        "api_key",
        "account_key",
        "private_key",
        "access_key",
        "connection_string",
        "source_auth",
        "privateKey",
        "observations",
    ],
)
def test_registration_query_hash_names_are_not_redacted(tmp_path, monkeypatch, alias):
    data = {
        "dataset": {},
        "recipe": {
            "source_keys": ["sku"],
            "training_start": "2024-01-01",
            "validation_origins": ["2024-02-01"],
            "history_missing": "error",
        },
        "source": {
            "kind": "delta",
            "sql": "SELECT 1",
            "tables": {"a": "catalog.schema.source"},
            "audit_queries": {alias: "SELECT 1"},
        },
        "runtime": {},
    }
    monkeypatch.setattr(registration, "_configuration", lambda _: (data, tmp_path))
    monkeypatch.setattr(registration, "_register_study", lambda *a: "study-id")
    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path / "scratch",
        runtime={},
        store=None,
        spark=None,
    )
    prepare = SimpleNamespace(validate=lambda _: None)
    assert (
        registration.register_study("config.json", registry=registry, prepare=prepare) == "study-id"
    )
    audit = next((registry.root / "checks").glob("*/audit.jsonl"))
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    summary = next(row for row in rows if row["stage"] == "Registration inputs")
    print("QUERY HASHES:", summary["query_sha256"])
    assert {row["query"]: row["sha256"] for row in summary["query_sha256"]}[alias] != "[redacted]"
    assert len({row["query"]: row["sha256"] for row in summary["query_sha256"]}[alias]) == 64


@dataclasses.dataclass
class NestedResult:
    path: pathlib.Path
    study_id: str = "child-study"
    status: str = "completed"
    run_id: str = "child-workflow"
    publication: object = None


def test_independent_nested_run_keeps_its_own_identity(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text('{"model":{"id":"example"}}')
    output = tmp_path / "child-workflow"
    output.mkdir()
    (output / "artifact_manifest.json").write_text("{}")

    @contextmanager
    def load_study(_):
        yield SimpleNamespace(registration={"sha256": "verified"})

    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=None,
        _registration_path=lambda *_: None,
        load_study=load_study,
    )
    monkeypatch.setattr(
        study_workflow, "run_forecast_workflow", lambda *a, **k: NestedResult(output)
    )
    parent = tmp_path / "parent.jsonl"
    with events.Execution(parent, ["fit"], study_id="parent-study", attempt_id="parent"):
        with events._stage(parent, "fit", study_id="parent-study", trial_id="parent-trial"):
            result = study_workflow.run_study(
                "child-study", lambda _: None, registry=registry, model_config=profile
            )
    rows = [
        json.loads(x)
        for x in pathlib.Path(result.publication["audit_path"]).read_text().splitlines()
    ]
    print(
        "CHILD IDENTITIES:",
        [(row["stage"], row.get("study_id"), row.get("trial_id")) for row in rows],
    )
    assert all(row.get("study_id") == "child-study" for row in rows)
    assert all(row.get("trial_id") != "parent-trial" for row in rows)


def test_nested_successful_publication_does_not_become_failed_parent_result(tmp_path):
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain")
    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        store=None,
        definition=lambda _: ({}, "verified"),
    )
    destination = SimpleNamespace(publish_results=lambda *a, **k: {"status": "published"})
    audit = tmp_path / "parent.jsonl"
    with pytest.raises(ValueError):
        with events.Execution(audit, ["workflow"], study_id="parent-study") as execution:
            with events._stage(audit, "workflow", study_id="parent-study"):
                StudyRegistry.publish_results(
                    registry, "unrelated-study", "/previous/workflow", destination=destination
                )
                raise ValueError("parent fit failed")
    panel = execution.render()
    print("PARENT FINAL PANEL:\n" + panel)
    assert "study_id: unrelated-study" not in panel
    assert "workflow_path: /previous/workflow" not in panel


def test_independent_nested_worker_keeps_its_own_identity(tmp_path):
    audit = tmp_path / "parent.jsonl"
    with events.Execution(audit, ["fit"], study_id="parent-study"):
        with events._stage(audit, "fit", study_id="parent-study", trial_id="parent-trial"):
            with events.standalone_attempt(
                tmp_path / "child", "Model job", separate=True, study_id="child-study"
            ):
                events.note("child operation")
    child = next((tmp_path / "child" / "execution-attempts").glob("*.jsonl"))
    rows = [json.loads(x) for x in child.read_text().splitlines()]
    print(
        "CHILD WORKER IDENTITIES:",
        [(row["stage"], row.get("study_id"), row.get("trial_id")) for row in rows],
    )
    assert all(row.get("study_id") == "child-study" for row in rows)
    assert all(row.get("trial_id") != "parent-trial" for row in rows)


def test_independent_nested_registration_has_no_parent_trial_identity(tmp_path, monkeypatch):
    from autoforecast import study_registration as registration

    data = {
        "dataset": {},
        "recipe": {
            "source_keys": ["sku"],
            "training_start": "2024-01-01",
            "validation_origins": ["2024-02-01"],
            "history_missing": "error",
        },
        "source": {"kind": "parquet"},
        "runtime": {},
    }
    monkeypatch.setattr(registration, "_configuration", lambda _: (data, tmp_path))
    monkeypatch.setattr(registration, "_register_study", lambda *a: "new-study")
    registry = SimpleNamespace(
        root=tmp_path / "registry", scratch_root=tmp_path / "scratch", runtime={}, store=None
    )
    audit = tmp_path / "parent.jsonl"
    with events.Execution(audit, ["fit"], study_id="parent-study"):
        with events._stage(audit, "fit", study_id="parent-study", trial_id="parent-trial"):
            assert registration.register_study("config.json", registry=registry) == "new-study"
    child = next((registry.root / "checks").glob("*/audit.jsonl"))
    rows = [json.loads(x) for x in child.read_text().splitlines()]
    print(
        "CHILD REGISTRATION IDENTITIES:",
        [(row["stage"], row.get("study_id"), row.get("trial_id")) for row in rows],
    )
    assert all(row.get("study_id") != "parent-study" for row in rows)
    assert all(row.get("trial_id") != "parent-trial" for row in rows)


def test_nested_successful_workflow_does_not_become_failed_parent_result(tmp_path, monkeypatch):
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path / "child")

    @events.report_workflow
    def child(study):
        return NestedResult(tmp_path / "child-workflow")

    audit = tmp_path / "parent.jsonl"
    with pytest.raises(ValueError):
        with events.Execution(audit, ["workflow"], study_id="parent-study") as execution:
            with events._stage(audit, "workflow", study_id="parent-study"):
                child(SimpleNamespace(study_id="child-study"))
                raise ValueError("parent fit failed")
    panel = execution.render()
    print("PARENT FINAL PANEL:\n" + panel)
    assert "study_id: child-study" not in panel
    assert "workflow_path: " + str(tmp_path / "child-workflow") not in panel


def test_concurrent_to_thread_run_keeps_its_own_identity(tmp_path, monkeypatch):
    import asyncio

    profile = tmp_path / "profile.json"
    profile.write_text('{"model":{"id":"example"}}')
    output = tmp_path / "child-workflow"
    output.mkdir()
    (output / "artifact_manifest.json").write_text("{}")

    @contextmanager
    def load_study(_):
        yield SimpleNamespace(registration={"sha256": "verified"})

    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=None,
        _registration_path=lambda *_: None,
        load_study=load_study,
    )
    monkeypatch.setattr(
        study_workflow, "run_forecast_workflow", lambda *a, **k: NestedResult(output)
    )

    async def start():
        return await asyncio.to_thread(
            study_workflow.run_study,
            "child-study",
            lambda _: None,
            registry=registry,
            model_config=profile,
        )

    parent = tmp_path / "parent.jsonl"
    with events.Execution(parent, ["fit"], study_id="parent-study", attempt_id="parent"):
        with events._stage(parent, "fit", study_id="parent-study", trial_id="parent-trial"):
            result = asyncio.run(start())
    rows = [
        json.loads(x)
        for x in pathlib.Path(result.publication["audit_path"]).read_text().splitlines()
    ]
    print(
        "CONCURRENT CHILD IDENTITIES:",
        [(row["stage"], row.get("study_id"), row.get("trial_id")) for row in rows],
    )
    assert all(row.get("study_id") == "child-study" for row in rows)
    assert all(row.get("trial_id") != "parent-trial" for row in rows)


@pytest.mark.parametrize("kind", ["logical_hash", "directory_hash", "content_hash", "volume_copy"])
@pytest.mark.parametrize("shape", ["single", "adjacent", "ambiguous"])
def test_url_derived_path_does_not_log_credentials(tmp_path, monkeypatch, kind, shape):
    sensitive = pathlib.Path(
        {
            "single": "https://user:sensitive-password@example.test/workflow?token=sensitive-token",
            "adjacent": "https://user:sensitive-password@example.test/a,https://user:sensitive-token@example.org/b",
            "ambiguous": "workspace:/forecast@archive/saved",
        }[shape]
    )
    source = tmp_path / sensitive
    source.parent.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1, 2]}), source)
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="off")
    audit = tmp_path / "audit.jsonl"
    if kind == "volume_copy":
        payload = b"data"
        monkeypatch.setattr(
            _volume_files, "_inventory", lambda _: (False, {"": (False, len(payload), None)})
        )
        monkeypatch.setattr(
            _volume_files,
            "_client",
            lambda: SimpleNamespace(
                files=SimpleNamespace(
                    download=lambda _: SimpleNamespace(contents=io.BytesIO(payload))
                )
            ),
        )
        monkeypatch.setattr(_volume_files.tempfile, "tempdir", str(tmp_path))
    with events.Execution(audit, ["work"]):
        with events._stage(audit, "work"):
            if kind == "logical_hash":
                benchmark.parquet_content_hash(source)
            elif kind == "directory_hash":
                lineage.directory_digest(source)
            elif kind == "content_hash":
                lineage.directory_content_digest(source)
            else:
                volume_path = pathlib.Path("/Volumes/catalog/schema/volume") / sensitive
                with _volume_files.local_snapshot(volume_path) as downloaded:
                    assert downloaded.read_bytes() == payload
    records = audit.read_text()
    output = stream.getvalue()
    print("NORMALIZED SOURCE:", str(source))
    assert "sensitive-password" not in records and "sensitive-token" not in records
    assert "sensitive-password" not in output and "sensitive-token" not in output
    assert "redacted" in records.lower() and "redacted" in output.lower()


@pytest.mark.parametrize(
    "kind",
    [
        "registration_source",
        "model_config",
        "model_output",
        "publication_receipt",
        "audit_location",
    ],
)
@pytest.mark.parametrize("shape", ["single", "adjacent", "ambiguous"])
def test_url_derived_path_is_not_copied_by_public_receipts(tmp_path, monkeypatch, kind, shape):
    import dataclasses
    from contextlib import contextmanager

    import yaml

    from autoforecast import study_registration, study_workflow
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    sensitive = tmp_path / pathlib.Path(
        {
            "single": "https://user:sensitive-password@example.test/workflow?token=sensitive-token",
            "adjacent": "https://user:sensitive-password@example.test/a,https://user:sensitive-token@example.org/b",
            "ambiguous": "workspace:/forecast@archive/saved",
        }[shape]
    )
    sensitive.mkdir(parents=True)
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain", interval=60)
    if kind == "registration_source":
        path = configuration(tmp_path / "input")
        data = yaml.safe_load(path.read_text())
        source = sensitive / "source.parquet"
        pq.write_table(pa.table({"x": [1]}), source)
        data["source"]["path"] = str(source)
        path.write_text(yaml.safe_dump(data))
        registry = SimpleNamespace(
            root=tmp_path / "registry", scratch_root=tmp_path / "scratch", runtime={}, store=None
        )
        monkeypatch.setattr(study_registration, "_register_study", lambda *a: "study-id")
        study_registration.register_study(path, registry=registry)
        audit = next((registry.root / "checks").glob("*/audit.jsonl"))
    elif kind in {"model_config", "model_output"}:

        @dataclasses.dataclass
        class Result:
            path: pathlib.Path
            study_id: str = "study-id"
            status: str = "completed"
            run_id: str = "workflow-id"
            publication: object = None

        config = (sensitive if kind == "model_config" else tmp_path) / "model.json"
        config.write_text('{"model":{"id":"example"}}')
        output = (sensitive if kind == "model_output" else tmp_path) / "result"
        output.mkdir()
        (output / "artifact_manifest.json").write_text("{}")

        @contextmanager
        def load(_):
            yield SimpleNamespace(registration={"sha256": "verified"})

        registry = SimpleNamespace(
            root=tmp_path / "registry",
            scratch_root=tmp_path,
            runtime={},
            store=None,
            _registration_path=lambda *_: None,
            load_study=load,
        )
        monkeypatch.setattr(study_workflow, "run_forecast_workflow", lambda *a, **k: Result(output))
        result = study_workflow.run_study(
            "study-id", lambda _: None, registry=registry, model_config=config
        )
        audit = pathlib.Path(result.publication["audit_path"])
    elif kind == "publication_receipt":
        registry = SimpleNamespace(
            root=tmp_path / "registry",
            scratch_root=tmp_path,
            store=None,
            definition=lambda _: ({}, "sha"),
        )
        StudyRegistry.publish_results(
            registry,
            "study-id",
            sensitive,
            destination=SimpleNamespace(publish_results=lambda *a, **k: {"status": "published"}),
        )
        audit = next((registry.root / "execution-attempts").glob("*.jsonl"))
    else:
        audit = sensitive / "audit.jsonl"
        with events.Execution(audit, ["work"]):
            with events._stage(audit, "work"):
                pass
    records = audit.read_text()
    output = stream.getvalue()
    print("PUBLIC OUTPUT:", output)
    assert "sensitive-password" not in records and "sensitive-token" not in records
    assert "sensitive-password" not in output and "sensitive-token" not in output
    assert "redacted" in output.lower()
    # These two paths appear only in the final panel, not in lifecycle event fields.
    if kind not in {"publication_receipt", "audit_location"}:
        assert "redacted" in records.lower()


@pytest.mark.parametrize("scheme", ["https", "s3", "abfss", "dbfs"])
def test_invalid_uri_registration_never_logs_credentials(tmp_path, monkeypatch, scheme):
    import yaml

    from autoforecast import study_registration
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path / "input")
    value = yaml.safe_load(path.read_text())
    value["source"]["path"] = (
        f"{scheme}://user:sensitive-password@example.test/data?token=sensitive-token"
    )
    path.write_text(yaml.safe_dump(value))
    registry = SimpleNamespace(
        root=tmp_path / "registry", scratch_root=tmp_path / "scratch", runtime={}, store=None
    )

    def fail(*args):
        raise FileNotFoundError("unsupported local path")

    monkeypatch.setattr(study_registration, "_register_study", fail)
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="off")
    with pytest.raises(FileNotFoundError):
        study_registration.register_study(path, registry=registry)
    audit = next((registry.root / "checks").glob("*/audit.jsonl")).read_text()
    print("AUDIT:", audit)
    assert "sensitive-password" not in audit and "sensitive-token" not in audit
    assert (
        "sensitive-password" not in stream.getvalue() and "sensitive-token" not in stream.getvalue()
    )


def test_standalone_publication_records_its_own_result(tmp_path):
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain")
    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        store=None,
        definition=lambda _: ({}, "verified"),
    )
    destination = SimpleNamespace(publish_results=lambda *a, **k: {"status": "published"})
    StudyRegistry.publish_results(
        registry, "saved-study", tmp_path / "saved-workflow", destination=destination
    )
    final_panel = stream.getvalue().rsplit("RUN ", 1)[-1]
    assert "study_id: saved-study" in final_panel
    assert str(tmp_path / "saved-workflow") in final_panel
    assert "PASS" in final_panel


def test_execution_restores_parent_context_when_heartbeat_cannot_start(tmp_path, monkeypatch):
    events.configure_logging(stream=io.StringIO())
    with events.Execution(tmp_path / "parent.jsonl", ["fit"]) as parent:
        with events._stage(parent.audit, "fit", trial_id="parent-trial"):
            before = (
                events._CURRENT.get(),
                events._IDENTITY.get(),
                events._OPERATION.get(),
                events._PARENT_AUDIT.get(),
            )
            with monkeypatch.context() as patch:

                def fail_start(self):
                    raise RuntimeError("thread unavailable")

                patch.setattr(events.threading.Thread, "start", fail_start)
                with pytest.raises(RuntimeError, match="thread unavailable"):
                    with events.Execution(tmp_path / "child.jsonl", ["child"]):
                        pytest.fail("body must not execute")
            after = (
                events._CURRENT.get(),
                events._IDENTITY.get(),
                events._OPERATION.get(),
                events._PARENT_AUDIT.get(),
            )
            assert before == after
    assert events._CURRENT.get() is None


def test_workflow_child_inherits_quiet_level_and_restores_parent_identity(tmp_path, monkeypatch):
    stream = io.StringIO()
    events.configure_logging(stream=stream)
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path / "child")

    @events.report_workflow
    def child(study):
        events.bind_identity(workflow_id="child-workflow")
        events.note("Child result checked")
        return NestedResult(tmp_path / "child-result")

    with events.Execution(
        tmp_path / "parent.jsonl", ["fit"], level=logging.ERROR, workflow_id="parent-workflow"
    ) as parent:
        with events._stage(parent.audit, "fit", trial_id="parent-trial"):
            child(SimpleNamespace(study_id="child-study"))
            assert parent.identity["workflow_id"] == "parent-workflow"
            assert events._IDENTITY.get()["trial_id"] == "parent-trial"
    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "message",
    [
        "authentication failed: password=synthetic-secret",
        "remote setup failed: api_key=synthetic-secret",
        "request rejected: Bearer synthetic-secret",
        "authorization: Basic synthetic-secret",
        "connection_string=synthetic-secret",
    ],
)
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_failure_audit_does_not_copy_exception_secrets(tmp_path, monkeypatch, message, error_type):
    error = error_type(message)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        study_workflow,
        "_config",
        lambda *a, **k: (b"profile", SimpleNamespace(model_id="example"), 1, 0, {}),
    )
    monkeypatch.setattr(study_workflow, "_implementation_record", lambda *_: {})
    monkeypatch.setattr(study_workflow, "resolve_implementation", lambda *a: {})
    monkeypatch.setattr(study_workflow, "_resolve_options", lambda *a: {})
    monkeypatch.setattr(study_workflow, "_output_root", lambda *a: tmp_path)
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)
    monkeypatch.setattr(study_workflow, "preflight_study", fail)
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="off")
    with pytest.raises(error_type) as caught:
        study_workflow.run_forecast_workflow(
            SimpleNamespace(study_id="study-id"),
            fit_predict=lambda _: None,
            implementation={},
            config="profile",
        )
    assert caught.value is error
    audit = "\n".join(path.read_text() for path in tmp_path.rglob("*.jsonl"))
    print("WORKFLOW AUDIT:", audit)
    assert "synthetic-secret" not in audit
    assert "synthetic-secret" not in stream.getvalue()
    expected = "FAIL" if error_type is ValueError else "CANCELLED"
    assert any(json.loads(line)["status"] == expected for line in audit.splitlines())


@pytest.mark.parametrize("separator", [",", ";", ")", "]", "|", " ", "\n"])
def test_multiple_uri_field_is_omitted_as_one_value(tmp_path, separator):
    original = (
        "https://first:secret-one@example.test/a"
        + separator
        + "https://second:secret-two@example.org/b"
    )
    audit = tmp_path / "audit.jsonl"
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain")
    with events.Execution(audit, ["work"]):
        with events._stage(audit, "work"):
            events.note("Requested locations", location=original)
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    row = next(row for row in rows if row["stage"] == "Requested locations")
    assert row["location"] == "[redacted location]"
    assert "secret-one" not in stream.getvalue() and "secret-two" not in stream.getvalue()


@pytest.mark.parametrize(
    "location",
    ["/data/forecast@archive/saved", "https://example.test/public/run", "workspace:/plain/saved"],
)
def test_location_without_sensitive_uri_details_is_preserved(tmp_path, location):
    audit = tmp_path / "audit.jsonl"
    with events.Execution(audit, ["work"]):
        with events._stage(audit, "work"):
            events.note("Saved location", location=location)
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert next(row["location"] for row in rows if row["stage"] == "Saved location") == location


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_registration_failure_after_configuration_belongs_to_sources(
    tmp_path, monkeypatch, error_type
):
    from autoforecast.test_study_registration import configuration

    config = configuration(tmp_path)
    registry = SimpleNamespace(
        root=tmp_path / "registry", scratch_root=tmp_path / "scratch", runtime={}, store=None
    )
    error = error_type("source initialization stopped")

    def fail_before_source_work(*args):
        raise error

    monkeypatch.setattr(registration, "_register_study", fail_before_source_work)
    with pytest.raises(error_type) as caught:
        registration.register_study(config, registry=registry)
    assert caught.value is error
    audit = next((registry.root / "checks").glob("*/audit.jsonl"))
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["status"] for row in rows if row["stage"] == "configuration"] == ["START", "PASS"]
    assert [row["status"] for row in rows if row["stage"] == "sources"] == [
        "START",
        "FAIL" if error_type is OSError else "CANCELLED",
    ]


@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("write_status", ["START", "PASS"])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_audit_uncertainty_does_not_invent_a_second_outcome(
    tmp_path, monkeypatch, planned, write_status, committed, error_type
):
    audit = tmp_path / "audit.jsonl"
    original = _safe_files.append_audit_bytes
    error = error_type("append outcome unknown")
    work = []

    def append(path, payload):
        row = json.loads(payload)
        if row["stage"] == "fit" and row["status"] == write_status:
            if committed:
                original(path, payload)
            raise error
        return original(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    with pytest.raises(error_type) as caught:
        with events.Execution(audit, ["fit"] if planned else []) as execution:
            with events._stage(audit, "fit") if planned else events.operation("fit"):
                work.append("ran")
    assert caught.value is error
    assert work == (["ran"] if write_status == "PASS" else [])
    rows = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
    expected = (["START"] if write_status == "PASS" else []) + ([write_status] if committed else [])
    assert [row["status"] for row in rows] == expected
    assert execution.result == ("FAIL" if error_type is OSError else "CANCELLED")
    assert execution.audit_error == error_type.__name__
    assert next(iter(execution.operations.values()))["status"] == "UNVERIFIED"
    if planned:
        assert execution.states["fit"] == "UNVERIFIED"
    assert "event storage could not be verified" in execution.render()
    assert "not saved" not in execution.render()
    assert events._CURRENT.get() is None and events._OPERATION.get() is None


@pytest.mark.parametrize("record_format", ["json", "yaml"])
@pytest.mark.parametrize("moment", ["sources", "admission", "freeze", "publication", "note", "append"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_later_registration_failure_preserves_completed_stages(
    tmp_path, monkeypatch, moment, error_type, record_format
):
    from autoforecast.test_study_registration import configuration

    registry = SimpleNamespace(
        root=tmp_path / "registry", scratch_root=tmp_path / "scratch", runtime={}, store=None
    )
    error = error_type("later boundary stopped")
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain")
    original = _safe_files.append_audit_bytes

    def append(path, payload):
        row = json.loads(payload)
        if moment == "note" and row["stage"] == "Study registered":
            raise error
        original(path, payload)
        if moment == "append" and row["stage"] == "publication" and row["status"] == "PASS":
            raise error

    def body(value, registry, prepare, settings, check_id, issues_root, event):
        for stage in ("sources", "admission", "freeze", "publication"):
            if stage != "sources":
                event(stage, "START")
            if stage == "publication":
                (issues_root / "published.txt").write_text("study-1")
                events.record_result(study_id="study-1", evidence=str(issues_root))
            event(stage, "PASS")
            if moment == stage:
                raise error
        return "study-1"

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    monkeypatch.setattr(registration, "_register_study", body)
    config_path = configuration(tmp_path)
    config = registration.read_record(config_path)
    config["records"] = {"format": record_format}
    config_path.write_bytes(registration.record_bytes(config, format="yaml"))
    with pytest.raises(error_type) as caught:
        registration.register_study(config_path, registry=registry)
    assert caught.value is error
    attempt = next((registry.root / "checks").iterdir())
    rows = [json.loads(line) for line in (attempt / "audit.jsonl").read_text().splitlines()]
    stage = moment if moment in {"sources", "admission", "freeze"} else "publication"
    assert [row["status"] for row in rows if row["stage"] == stage] == ["START", "PASS"]
    evidence = registration.read_record(attempt / ("attempt." + record_format))
    assert evidence["stage"] == "registration"
    assert evidence["last_stage"] == stage
    assert rows[-1]["stage"] == "registration"
    assert rows[-1]["status"] == ("FAIL" if error_type is OSError else "CANCELLED")
    final = stream.getvalue().rsplit("RUN ", 1)[-1]
    if stage == "publication":
        assert (attempt / "published.txt").read_text() == "study-1"
        assert "study_id: study-1" in final
    assert f"{stage} ({'unverified' if moment == 'append' else 'pass'})" in final


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("stage", ["preflight", "tuning"])
def test_workflow_completion_audit_uncertainty_has_no_second_terminal(
    tmp_path, monkeypatch, committed, stage
):
    error = OSError("preflight audit outcome unknown")
    original = _safe_files.append_audit_bytes

    def append(path, payload):
        row = json.loads(payload)
        if row["stage"] == stage and row["status"] == "PASS":
            if committed:
                original(path, payload)
            raise error
        return original(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    monkeypatch.setattr(
        study_workflow,
        "_config",
        lambda *a, **k: (b"profile", SimpleNamespace(model_id="example"), 1, 0, {}),
    )
    monkeypatch.setattr(study_workflow, "_implementation_record", lambda *_: {})
    monkeypatch.setattr(study_workflow, "resolve_implementation", lambda *a: {})
    monkeypatch.setattr(
        study_workflow, "_resolve_options", lambda *a: {"windows": [{}], "stage": "tuning"}
    )
    monkeypatch.setattr(study_workflow, "_output_root", lambda *a: tmp_path)
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)
    result = SimpleNamespace(
        study_id="study-1",
        report={"issues": [], "excluded_ids": [], "windows": {}},
        report_path=tmp_path / "preflight.json",
    )
    monkeypatch.setattr(study_workflow, "preflight_study", lambda *a, **k: result)
    monkeypatch.setattr(
        study_workflow, "tune_study", lambda *a, **k: SimpleNamespace(path=tmp_path / "search")
    )
    with pytest.raises(OSError) as caught:
        study_workflow.run_forecast_workflow(
            SimpleNamespace(study_id="study-1"),
            fit_predict=lambda _: None,
            implementation={},
            config="profile",
        )
    assert caught.value is error
    audit = next(tmp_path.glob("workflow-*/events.jsonl"))
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    relevant = [r for r in rows if r["stage"] == stage]
    print("OBSERVED", stage, committed, [(r["status"], r.get("operation_id")) for r in relevant])
    assert [r["status"] for r in relevant] == ["START"] + (["PASS"] if committed else [])
    failure = next(row for row in rows if row["stage"] == "workflow_attempt")
    assert failure["status"] == "FAIL" and failure["last_stage"] == stage


@pytest.mark.parametrize("mode", ["sequential", "local"])
@pytest.mark.parametrize("committed", [None, False, True])
@pytest.mark.parametrize("audit_error", [OSError, KeyboardInterrupt, SystemExit])
def test_model_failure_is_retained_if_failure_audit_write_is_unverified(
    tmp_path, monkeypatch, mode, committed, audit_error
):
    root = tmp_path / "campaign"
    attempt = root / "stages" / "validation" / "attempt-1"
    attempt.mkdir(parents=True)
    request = {
        "models": [{"id": "m1", "cpus": 1, "gpus": 0, "memory_gib": 1}],
        "execution": mode,
        "max_parallel_models": 1,
        "timeout_seconds": 10,
        "final_evaluation": "all_models",
    }
    monkeypatch.setattr(study_campaign, "_read", lambda *args: request)
    primary = ValueError("synthetic model failure")
    secondary = audit_error("failure audit unavailable")

    def fail_model(*args, **kwargs):
        raise primary

    monkeypatch.setattr(study_campaign, "_model_job", fail_model)
    if mode == "local":

        class Worker:
            def options(self, **kw):
                return self

            def remote(self, *args, **kwargs):
                return "ref"

        fake = SimpleNamespace(
            is_initialized=lambda: False,
            init=lambda **kwargs: None,
            nodes=lambda: [
                {
                    "Alive": True,
                    "Resources": {"CPU": 2, "GPU": 0, "memory": 2 * 1024**3},
                    "Labels": {},
                }
            ],
            remote=lambda **kwargs: lambda _: Worker(),
            wait=lambda refs, **kwargs: ([refs[0]], []),
            get=lambda ref: fail_model(),
            cancel=lambda *a, **k: None,
            shutdown=lambda: None,
        )
        monkeypatch.setitem(sys.modules, "ray", fake)
    append = _safe_files.append_audit_bytes

    def write(path, payload):
        row = json.loads(payload)
        if row["stage"] == "Model 1: m1" and row["status"] == "FAIL" and committed is not None:
            if committed:
                append(path, payload)
            raise secondary
        return append(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", write)
    with pytest.raises(RuntimeError, match="campaign models failed") as caught:
        study_campaign._run_phase("request", "sha", "validation", str(root), str(attempt))
    summary = attempt / "summary.json"
    print(
        "OBSERVED", mode, committed, type(caught.value).__name__, "summary_exists", summary.exists()
    )
    assert summary.exists()
    assert json.loads(summary.read_text())["failures"]["0"] == str(primary)


@dataclasses.dataclass
class RunSummaryResult:
    path: pathlib.Path
    status: str
    run_id: str = "workflow-1"
    study_id: str = "study-1"
    publication: object = None


@pytest.mark.parametrize("status", ["completed", "pending_actuals"])
@pytest.mark.parametrize("committed", [False, True])
def test_run_summary_audit_error_does_not_duplicate_run_outcome(
    tmp_path, monkeypatch, status, committed
):
    output = tmp_path / "workflow"
    output.mkdir()
    (output / "artifact_manifest.json").write_text("{}")
    profile = tmp_path / "profile.json"
    profile.write_text('{"model":{"id":"example"}}')

    @contextmanager
    def load(_):
        yield SimpleNamespace(registration={"sha256": "verified"})

    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=None,
        _registration_path=lambda *_: None,
        load_study=load,
    )
    monkeypatch.setattr(
        study_workflow, "run_forecast_workflow", lambda *a, **k: RunSummaryResult(output, status)
    )
    original = _safe_files.append_audit_bytes
    error = OSError("run summary audit interrupted")
    terminal = "PASS" if status == "completed" else "UNAVAILABLE"

    def append(path, payload):
        row = json.loads(payload)
        if row["stage"] == "run" and row["status"] == terminal:
            if committed:
                original(path, payload)
            raise error
        return original(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    with pytest.raises(OSError) as caught:
        study_workflow.run_study("study-1", lambda _: None, registry=registry, model_config=profile)
    assert caught.value is error
    audit = next((registry.root / "run-attempts").glob("*.jsonl"))
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    outcomes = [r["status"] for r in rows if r["stage"] == "run"]
    print("OBSERVED", status, committed, outcomes)
    assert outcomes == ([terminal] if committed else [])
    assert rows[-1]["stage"] == "run_attempt" and rows[-1]["status"] == "FAIL"


@pytest.mark.parametrize("mode", ["sequential", "local"])
@pytest.mark.parametrize("target", ["START", "PASS"])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt])
def test_manual_campaign_stage_uncertainty_is_visible_without_second_outcome(
    tmp_path, monkeypatch, mode, target, committed, kind
):
    root = tmp_path / "campaign"
    attempt = root / "stages" / "validation" / "attempt-1"
    attempt.mkdir(parents=True)
    request = {
        "models": [{"id": "m1", "cpus": 1, "gpus": 0, "memory_gib": 1}],
        "execution": mode,
        "max_parallel_models": 1,
        "timeout_seconds": 10,
        "final_evaluation": "all_models",
    }
    monkeypatch.setattr(study_campaign, "_read", lambda *args: request)
    record = {"workflow_path": "saved"}
    called = []
    stopped = []

    def model(*args, **kwargs):
        called.append(True)
        return record

    monkeypatch.setattr(study_campaign, "_model_job", model)
    monkeypatch.setattr(study_campaign, "_load_model_result", lambda *args, **kwargs: record)
    if mode == "local":

        class Worker:
            def options(self, **kw):
                return self

            def remote(self, *args, **kwargs):
                called.append(True)
                return "ref"

        fake = SimpleNamespace(
            is_initialized=lambda: False,
            init=lambda **kwargs: None,
            nodes=lambda: [
                {
                    "Alive": True,
                    "Resources": {"CPU": 2, "GPU": 0, "memory": 2 * 1024**3},
                    "Labels": {},
                }
            ],
            remote=lambda **kwargs: lambda _: Worker(),
            wait=lambda refs, **kwargs: ([refs[0]], []),
            get=lambda ref: record,
            cancel=lambda *a, **k: None,
            shutdown=lambda: stopped.append(True),
        )
        monkeypatch.setitem(sys.modules, "ray", fake)
    output = io.StringIO()
    events.configure_logging(stream=output, progress="plain", interval=60)
    append = _safe_files.append_audit_bytes
    error = kind("audit outcome unknown")

    def write(path, payload):
        row = json.loads(payload)
        if row["stage"] == "Model 1: m1" and row["status"] == target:
            if committed:
                append(path, payload)
            raise error
        return append(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", write)
    with pytest.raises(kind) as caught:
        study_campaign._run_phase("request", "sha", "validation", str(root), str(attempt))
    assert caught.value is error
    rows = [json.loads(line) for line in (attempt / "progress.jsonl").read_text().splitlines()]
    statuses = [r["status"] for r in rows if r["stage"] == "Model 1: m1"]
    assert statuses == (["START"] if target == "PASS" else []) + ([target] if committed else [])
    assert "UNVERIFIED Model 1: m1" in output.getvalue()
    assert "event storage could not be verified" in output.getvalue()
    if mode == "local":
        assert stopped == [True]


@pytest.mark.parametrize("case", ["success", "publication_failure"])
@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_receipt_precedes_publication_and_cancellation_closes_execution(tmp_path, monkeypatch, case, kind):
    out = tmp_path / "workflow"
    out.mkdir()
    (out / "artifact_manifest.json").write_text("{}")
    profile = tmp_path / "profile.json"
    profile.write_text('{"model":{"id":"example"}}')

    @contextmanager
    def load(_):
        yield SimpleNamespace(registration={"sha256": "verified"})

    primary = ValueError("publication primary")
    secondary = kind("receipt interrupted")

    def publish(*args, **kwargs):
        raise primary

    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=SimpleNamespace(validate_destinations=lambda: None, tables={}),
        _registration_path=lambda *_: None,
        load_study=load,
        publish_results=publish,
    )
    monkeypatch.setattr(
        study_workflow, "run_forecast_workflow", lambda *a, **k: RunSummaryResult(out, "completed")
    )
    seen = []

    class Tracked(events.Execution):
        def __enter__(self):
            seen.append(self)
            return super().__enter__()

    monkeypatch.setattr(study_workflow, "Execution", Tracked)

    # Receipt reporting now happens before publication. The former late-cleanup
    # interruption cannot occur; verify one early receipt and normal error cleanup.
    recorded = []

    def interrupt(**kwargs):
        recorded.append(kwargs)
        if case == "success":
            raise secondary
        events.record_result(**kwargs)

    monkeypatch.setattr(study_workflow, "record_result", interrupt)
    events.configure_logging(stream=io.StringIO(), progress="off", interval=60)
    caught = None
    try:
        study_workflow.run_study(
            "study-1",
            lambda _: None,
            registry=registry,
            model_config=profile,
            publish=case == "publication_failure",
        )
    except BaseException as error:
        caught = error
    execution = seen[0]
    current = events._CURRENT.get()
    alive = execution.thread.is_alive()
    print(
        "OBSERVED",
        case,
        kind.__name__,
        type(caught).__name__,
        "context_leaked",
        current is execution,
        "heartbeat_alive",
        alive,
    )
    if current is execution:
        execution.__exit__(type(caught), caught, None)
    assert len(recorded) == 1
    if case == "publication_failure":
        assert execution.receipt["workflow_id"] == "workflow-1"
    assert events._CURRENT.get() is None
    assert current is None and not alive
    if case == "publication_failure":
        assert caught is primary
    else:
        assert caught is secondary


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("committed", [False, True])
def test_run_failure_audit_secondary_cancellation_does_not_replace_primary(
    tmp_path, monkeypatch, kind, committed
):
    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=None,
        _registration_path=lambda *_: None,
    )
    reached = []
    original = _safe_files.append_audit_bytes
    secondary = kind("secondary interruption")

    def append(path, payload):
        row = json.loads(payload)
        if row["stage"] == "run_attempt" and row["status"] == "FAIL":
            if committed:
                original(path, payload)
            reached.append(True)
            raise secondary
        return original(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    caught = None
    try:
        study_workflow.run_study("study-1", "invalid-model", registry=registry)
    except BaseException as error:
        caught = error
    print("OBSERVED", kind.__name__, committed, type(caught).__name__, str(caught))
    assert type(caught) is ValueError and "model must be" in str(caught)
    assert reached == [True]


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("committed", [False, True])
def test_workflow_failure_audit_secondary_cancellation_does_not_replace_primary(
    tmp_path, monkeypatch, kind, committed
):
    primary = ValueError("preflight primary")
    secondary = kind("secondary interruption")

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(
        study_workflow,
        "_config",
        lambda *a, **k: (b"profile", SimpleNamespace(model_id="example"), 1, 0, {}),
    )
    monkeypatch.setattr(study_workflow, "_implementation_record", lambda *_: {})
    monkeypatch.setattr(study_workflow, "resolve_implementation", lambda *a: {})
    monkeypatch.setattr(study_workflow, "_resolve_options", lambda *a: {})
    monkeypatch.setattr(study_workflow, "_output_root", lambda *a: tmp_path)
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)
    monkeypatch.setattr(study_workflow, "preflight_study", fail)
    reached = []
    original = _safe_files.append_audit_bytes

    def append(path, payload):
        row = json.loads(payload)
        if row["stage"] == "workflow_attempt" and row["status"] == "FAIL":
            if committed:
                original(path, payload)
            reached.append(True)
            raise secondary
        return original(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    caught = None
    try:
        study_workflow.run_forecast_workflow(
            SimpleNamespace(study_id="study-1"),
            fit_predict=lambda _: None,
            implementation={},
            config="profile",
        )
    except BaseException as error:
        caught = error
    print("OBSERVED", kind.__name__, committed, type(caught).__name__, str(caught))
    assert caught is primary
    assert reached == [True]


class InterruptingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.armed = True
        self.interruption = KeyboardInterrupt("display interrupted")

    def write(self, text):
        if self.armed and text.startswith("PASS fit |"):
            self.armed = False
            raise self.interruption
        return super().write(text)


@pytest.mark.parametrize("kind", ["planned", "nested"])
def test_durable_pass_survives_console_interruption_without_relabel(tmp_path, kind):
    logger = logging.getLogger("autoforecast.stages")
    old = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    stream = InterruptingStream()
    events.configure_logging(stream=stream, progress="off")
    path = tmp_path / "events.jsonl"
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            with events.Execution(path, ["fit"] if kind == "planned" else ["work"]) as execution:
                if kind == "planned":
                    with events._stage(path, "fit"):
                        pass
                else:
                    with events._stage(path, "work"):
                        with events.operation("fit"):
                            pass
        assert caught.value is stream.interruption
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        statuses = [row["status"] for row in rows if row["stage"] == "fit"]
        print("OBSERVED", kind, statuses, "states", execution.states, "panel", execution.render())
        assert statuses == ["START", "PASS"]
        if kind == "planned":
            assert execution.states["fit"] == "PASS"
        else:
            assert not any(
                op["name"] == "fit" and not op.get("status") for op in execution.operations.values()
            )
    finally:
        logger.handlers, logger.level, logger.propagate = old


@pytest.mark.parametrize("write_status", ["START", "PASS"])
@pytest.mark.parametrize("committed", [False, True])
def test_audit_error_is_not_replaced_by_secondary_console_interruption(
    tmp_path, monkeypatch, write_status, committed
):
    from mlops import _safe_files

    class SecondaryStream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.armed = True
            self.error = KeyboardInterrupt("secondary display interruption")

        def write(self, text):
            if self.armed and text.startswith("UNVERIFIED fit |"):
                self.armed = False
                raise self.error
            return super().write(text)

    logger = logging.getLogger("autoforecast.stages")
    old = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    stream = SecondaryStream()
    events.configure_logging(stream=stream, progress="off")
    original = _safe_files.append_audit_bytes
    primary = OSError("audit unavailable")

    def append(path, payload):
        row = json.loads(payload)
        if row["stage"] == "fit" and row["status"] == write_status:
            if committed:
                original(path, payload)
            raise primary
        return original(path, payload)

    monkeypatch.setattr(_safe_files, "append_audit_bytes", append)
    caught = None
    execution = None
    try:
        try:
            with events.Execution(tmp_path / "events.jsonl", ["fit"]) as execution:
                with events._stage(tmp_path / "events.jsonl", "fit"):
                    pass
        except BaseException as error:
            caught = error
        print(
            "SECONDARY",
            write_status,
            committed,
            type(caught).__name__,
            "audit_error",
            execution.audit_error,
            "state",
            execution.states,
        )
        assert caught is primary
        assert execution.audit_error == "OSError"
        assert execution.states["fit"] == "UNVERIFIED"
    finally:
        logger.handlers, logger.level, logger.propagate = old


def test_public_run_does_not_relabel_committed_configuration_pass(tmp_path):
    from types import SimpleNamespace

    from autoforecast import configure_logging, run_study

    class PublicStream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.armed = True
            self.error = KeyboardInterrupt("display interrupted")

        def write(self, text):
            if self.armed and text.startswith("PASS configuration |"):
                self.armed = False
                raise self.error
            return super().write(text)

    logger = logging.getLogger("autoforecast.stages")
    old = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    stream = PublicStream()
    configure_logging(stream=stream, progress="off")
    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=None,
        _registration_path=lambda *_: None,
    )
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            run_study(
                "study-1", lambda _: None, registry=registry, model_config=tmp_path / "profile"
            )
        assert caught.value is stream.error
        audit = next((registry.root / "run-attempts").glob("*.jsonl"))
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        statuses = [row["status"] for row in rows if row["stage"] == "configuration"]
        print("PUBLIC", statuses)
        assert statuses == ["START", "PASS"]
    finally:
        logger.handlers, logger.level, logger.propagate = old


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_receipt_interrupt_is_not_swallowed_by_caller_exception_context(
    tmp_path, monkeypatch, outer, kind
):
    output = tmp_path / "workflow"
    output.mkdir()
    (output / "artifact_manifest.json").write_text("{}")
    profile = tmp_path / "profile.json"
    profile.write_text('{"model":{"id":"example"}}')

    @contextmanager
    def load(_):
        yield SimpleNamespace(registration={"sha256": "verified"})

    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=None,
        _registration_path=lambda *_: None,
        load_study=load,
    )
    monkeypatch.setattr(
        study_workflow,
        "run_forecast_workflow",
        lambda *a, **k: RunSummaryResult(output, "completed"),
    )
    interruption = kind("receipt interrupted")

    def interrupt(**kwargs):
        raise interruption

    monkeypatch.setattr(study_workflow, "record_result", interrupt)
    events.configure_logging(stream=io.StringIO(), progress="off", interval=60)
    captured = []

    def invoke():
        try:
            result = study_workflow.run_study(
                "study-1", lambda _: None, registry=registry, model_config=profile
            )
        except BaseException as error:
            captured.append(error)
        else:
            captured.append(result)

    if outer:
        try:
            raise ValueError("already handled outside run")
        except ValueError:
            invoke()
    else:
        invoke()
    print("OBSERVED", outer, kind.__name__, type(captured[0]).__name__)
    assert captured[0] is interruption
    assert events._CURRENT.get() is None


@pytest.mark.parametrize("label", [None, "Seasonal naive", "Moving average"])
def test_public_model_label_is_operational_metadata_only(tmp_path, monkeypatch, label):
    output = tmp_path / "workflow"
    output.mkdir()
    (output / "artifact_manifest.json").write_text("{}")
    profile = tmp_path / "profile.json"
    original_profile = '{"model":{"id":"comparison_models"}}'
    profile.write_text(original_profile)
    calls = []

    @contextmanager
    def load(_):
        yield SimpleNamespace(registration={"sha256": "verified"})

    registry = SimpleNamespace(
        root=tmp_path / "registry", scratch_root=tmp_path, runtime={}, store=None,
        _registration_path=lambda *_: None, load_study=load,
    )

    def workflow(*args, **kwargs):
        calls.append(kwargs)
        return RunSummaryResult(output, "completed")

    monkeypatch.setattr(study_workflow, "run_forecast_workflow", workflow)
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="off")
    result = study_workflow.run_study(
        "study-1", lambda _: None, registry=registry, model_config=profile, model_label=label,
    )
    assert result.study_id == "study-1" and result.path == output
    assert calls[0]["model_label"] == label
    assert "model_label" not in calls[0]["registration"]
    assert profile.read_text() == original_profile
    rows = [json.loads(line) for line in pathlib.Path(result.publication["audit_path"])
            .read_text().splitlines()]
    assert all(row.get("model_label") == label for row in rows)
    assert next(row["model_id"] for row in rows if row["stage"] == "Model configuration") == (
        "comparison_models"
    )
    if label is not None:
        assert f"model_label={label}" in stream.getvalue()
    else:
        assert "model_label=" not in stream.getvalue()


@pytest.mark.parametrize("label", [None, "Seasonal naive", "Moving average"])
def test_direct_workflow_label_does_not_change_returned_result(tmp_path, monkeypatch, label):
    result = RunSummaryResult(tmp_path / "result", "completed")
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)
    received = []

    @events.report_workflow
    def workflow(study, *, model_label=None):
        received.append(model_label)
        events.bind_identity(model_id="comparison_models")
        with events.operation("Fit model"):
            events.note("Callback inputs", model_parameters={"variant": "private-setting"})
        return result

    assert workflow(SimpleNamespace(study_id="study-1"), model_label=label) is result
    assert received == [label]
    audit = next((tmp_path / "execution-attempts").glob("*.jsonl"))
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert all(row.get("model_label") == label for row in rows)
    assert "private-setting" not in audit.read_text()
    assert events._CURRENT.get() is None


@pytest.mark.parametrize("label", ["", "   ", 12, True, "x" * 101, "two\nlines", "control\x1b"])
@pytest.mark.parametrize("route", ["public", "direct", "method"])
def test_invalid_model_label_fails_before_creating_an_attempt(tmp_path, monkeypatch, label, route):
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)
    with pytest.raises(ValueError, match="model_label"):
        if route == "public":
            study_workflow.run_study("study-1", "seasonal_naive", model_label=label)
        elif route == "direct":
            study_workflow.run_forecast_workflow(
                SimpleNamespace(study_id="study-1"), fit_predict=lambda _: None, model_label=label,
            )
        else:
            from autoforecast.study import Study

            Study.__new__(Study).run_forecast_workflow(fit_predict=lambda _: None, model_label=label)
    assert list(tmp_path.iterdir()) == []


class StartInterruptingStream(io.StringIO):
    def __init__(self, error_type):
        super().__init__()
        self.armed = True
        self.error = error_type("console interrupt")

    def write(self, text):
        if self.armed and text.startswith("START fit |"):
            self.armed = False
            raise self.error
        return super().write(text)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("kind", ["planned", "nested"])
def test_interrupted_start_output_does_not_leave_nested_operation_running(
    tmp_path, kind, error_type
):
    logger = logging.getLogger("autoforecast.stages")
    old = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    stream = StartInterruptingStream(error_type)
    events.configure_logging(stream=stream, progress="off")
    path = tmp_path / "events.jsonl"
    try:
        with pytest.raises(error_type) as caught:
            with events.Execution(path, ["fit"] if kind == "planned" else ["work"]) as execution:
                if kind == "planned":
                    with events._stage(path, "fit"):
                        pass
                else:
                    with events._stage(path, "work"):
                        with events.operation("fit"):
                            pass
        assert caught.value is stream.error
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        statuses = [r["status"] for r in rows if r["stage"] == "fit"]
        operations = [o for o in execution.operations.values() if o["name"] == "fit"]
        print("OBSERVED", kind, statuses, "operations", operations, "panel", execution.render())
        assert all(o.get("status") != "START" for o in operations)
        assert not operations or all(
            o.get("status") in {"CANCELLED", "UNVERIFIED"} for o in operations
        )
    finally:
        logger.handlers, logger.level, logger.propagate = old


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_public_run_interrupted_nested_start_has_terminal_outcome(tmp_path, error_type):
    from types import SimpleNamespace

    from autoforecast import configure_logging, run_study

    stream = StartInterruptingStream(error_type)
    logger = logging.getLogger("autoforecast.stages")
    old = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    configure_logging(stream=stream, progress="plain")

    def validate():
        with events.operation("fit"):
            pass

    registry = SimpleNamespace(
        root=tmp_path / "registry",
        scratch_root=tmp_path,
        runtime={},
        store=SimpleNamespace(validate_destinations=validate, tables={}),
        _registration_path=lambda *_: None,
    )
    try:
        with pytest.raises(error_type) as caught:
            run_study(
                "study-1", lambda _: None, registry=registry, model_config="profile", publish=True
            )
        assert caught.value is stream.error
        audit = next((registry.root / "run-attempts").glob("*.jsonl"))
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        statuses = [r["status"] for r in rows if r["stage"] == "fit"]
        final = stream.getvalue().rsplit("RUN ", 1)[-1]
        print("PUBLIC", statuses, "FINAL", final)
        assert statuses == ["START", "CANCELLED"]
        assert "Operation: fit" not in final or "| CANCELLED" in final
    finally:
        logger.handlers, logger.level, logger.propagate = old


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_standalone_audited_stage_interrupted_at_start_is_not_left_open(tmp_path, error_type):
    logger = logging.getLogger("autoforecast.stages")
    old = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    stream = StartInterruptingStream(error_type)
    events.configure_logging(stream=stream, progress="off")
    path = tmp_path / "events.jsonl"
    try:
        with pytest.raises(error_type) as caught:
            with events._stage(path, "fit"):
                pass
        assert caught.value is stream.error
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        statuses = [r["status"] for r in rows if r["stage"] == "fit"]
        print("STANDALONE", statuses)
        assert statuses == ["START", "CANCELLED"]
    finally:
        logger.handlers, logger.level, logger.propagate = old


@pytest.mark.parametrize("label", [None, "Moving average"])
def test_study_method_keeps_public_label_and_original_result(tmp_path, monkeypatch, label):
    from autoforecast.study import Study

    study = Study.__new__(Study)
    monkeypatch.setattr(Study, "_record", lambda *a, **k: {"study_id": "study-1"})
    monkeypatch.setattr(study_tuning, "_output_root", lambda *a: tmp_path)
    result = RunSummaryResult(tmp_path / "saved", "completed")
    received = []

    @events.report_workflow
    def workflow(actual_study, **kwargs):
        assert actual_study is study
        received.append(kwargs)
        events.bind_identity(model_id="comparison_models")
        events.note("Selected model")
        return result

    monkeypatch.setattr(study_workflow, "run_forecast_workflow", workflow)
    stream = io.StringIO()
    events.configure_logging(stream=stream, progress="plain")
    parameters = {"model.window": 7}
    kwargs = {} if label is None else {"model_label": label}
    returned = study.run_forecast_workflow(
        fit_predict=lambda _: None, config="profile", fixed_parameters=parameters, **kwargs,
    )
    assert returned is result
    assert received[0]["model_label"] == label
    assert received[0]["fixed_parameters"] is parameters
    assert parameters == {"model.window": 7}
    audit = next((tmp_path / "execution-attempts").glob("*.jsonl"))
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert all(row.get("model_label") == label for row in rows)
    if label is not None:
        assert f"model_label={label}" in stream.getvalue()
    assert events._CURRENT.get() is None
