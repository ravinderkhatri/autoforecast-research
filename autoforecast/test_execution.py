"""Operational progress has its own counters and never changes model evidence."""

import concurrent.futures
import contextlib
import io
import json
import logging
import subprocess
import sys
import threading

import pytest

from autoforecast import _execution as events
from autoforecast import configure_logging
from autoforecast import study_registration as registration


@pytest.fixture(autouse=True)
def isolated_console():
    logger = logging.getLogger("autoforecast.stages")
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    yield
    logger.handlers, logger.level, logger.propagate = previous


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_automatic_progress_follows_cells_without_configuring_logging():
    script = '''
import contextlib, io, logging, tempfile
from pathlib import Path
from autoforecast._execution import Execution, _stage
root = logging.getLogger()
application_stream = io.StringIO()
handler = logging.StreamHandler(application_stream)
root.addHandler(handler)
root.setLevel(logging.WARNING)
library = logging.getLogger("autoforecast.stages")
with tempfile.TemporaryDirectory() as folder:
    streams = []
    for index in range(2):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            with Execution(Path(folder)/f"{index}.jsonl", ["prepare"], attempt_id=str(index)):
                with _stage(Path(folder)/f"{index}.jsonl", "prepare"):
                    pass
        text = stream.getvalue()
        assert "prepare" in text and "100.0%" in text
        streams.append(text)
        stream.close()
    assert "'attempt_id': '1'" not in streams[0]
    assert "'attempt_id': '0'" not in streams[1]
assert root.handlers == [handler] and root.level == logging.WARNING
assert library.handlers == [] and library.level == logging.NOTSET
assert application_stream.getvalue() == ""
print("PASS")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "PASS\n"


@pytest.mark.parametrize("quiet", ["runtime", "library", "disabled"])
def test_automatic_progress_respects_quiet_settings(tmp_path, monkeypatch, quiet):
    logger = logging.getLogger("autoforecast.stages")
    monkeypatch.setattr(logger, "level", logging.WARNING if quiet == "library" else logging.NOTSET)
    monkeypatch.setattr(logger, "disabled", quiet == "disabled")
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    with events.Execution(tmp_path / "audit.jsonl", ["prepare"],
                          level=logging.WARNING if quiet == "runtime" else logging.INFO):
        with events._stage(tmp_path / "audit.jsonl", "prepare"):
            pass
    assert stream.getvalue() == ""
    assert records(tmp_path / "audit.jsonl")


def test_default_events_use_existing_application_handler_once(tmp_path, monkeypatch):
    stream = io.StringIO()
    logger = logging.getLogger("autoforecast.stages")
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    logger.setLevel(logging.INFO)
    events._stage_event(tmp_path / "audit.jsonl", "example", "START")
    assert stream.getvalue().count("START example") == 1
    assert logger.handlers == [handler]


def test_fresh_process_explicit_setup_and_repeat_preserve_root():
    script = '''
import io, logging, tempfile
from pathlib import Path
root = logging.getLogger()
handler = logging.StreamHandler(io.StringIO())
root.addHandler(handler)
root.setLevel(logging.ERROR)
from autoforecast import configure_logging
from autoforecast import study_registration as registration
assert logging.getLogger('autoforecast.stages').handlers == []
from autoforecast._execution import _stage_event
from autoforecast import _execution as events
assert events._FALLBACK_CONSOLE is None
stream = io.StringIO()
configure_logging(stream=stream, progress='off')
configure_logging(stream=stream, progress='off')
with tempfile.TemporaryDirectory() as d:
    _stage_event(Path(d)/'audit.jsonl', 'example', 'START')
assert stream.getvalue().count('START example') == 1
assert root.handlers == [handler] and root.level == logging.ERROR
print('PASS')
'''
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout == "PASS\n"


def test_level_precedence_and_quiet_audit(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOFORECAST_LOG_LEVEL", raising=False)
    assert events.resolve_level() == logging.INFO
    assert events.resolve_level(runtime={"log_level": "ERROR"}) == logging.ERROR
    monkeypatch.setenv("AUTOFORECAST_LOG_LEVEL", "WARNING")
    assert events.resolve_level(runtime={"log_level": "ERROR"}) == logging.WARNING
    assert events.resolve_level("DEBUG", {"log_level": "ERROR"}) == logging.DEBUG
    with pytest.raises(ValueError):
        events.resolve_level("LOUD")
    stream = io.StringIO()
    configure_logging(stream=stream)
    path = tmp_path / "audit.jsonl"
    with events.Execution(path, ["save"], level=logging.ERROR):
        with events._stage(path, "save"):
            events.note("decision", selected="checked output")
    assert stream.getvalue() == ""
    assert [(r["stage"], r["status"]) for r in records(path)] == [
        ("save", "START"), ("decision", "INFO"), ("save", "PASS"),
    ]


def test_stage_position_completed_remaining_and_runtime_skip(tmp_path):
    path = tmp_path / "audit.jsonl"
    with events.Execution(path, [f"s{i}" for i in range(1, 11)]) as run:
        events._stage_event(path, "s1", "START")
        assert "Stage 1/10" in run.render()
        assert "0/10 (0.0% of stages completed)" in run.render()
        events._stage_event(path, "s1", "PASS")
        assert "1/10 (10.0% of stages completed)" in run.render()
        assert "Completed: Stage 1: s1 (pass)" in run.render()
        assert "Remaining (not completed): Stage 2: s2 (waiting)" in run.render()
        events._stage_event(path, "s2", "SKIP")
        assert "Resolved: 2/10; skipped=1" in run.render()
        assert "1/10 (10.0%" in run.render()
    assert run.result == "INCOMPLETE"


@pytest.mark.parametrize("explicit_setup", [False, True])
def test_notebook_same_display_throttle_and_no_progress_audit(tmp_path, monkeypatch, explicit_setup):
    from types import ModuleType, SimpleNamespace
    calls = []
    ipython, display = ModuleType("IPython"), ModuleType("IPython.display")
    ipython.get_ipython = lambda: SimpleNamespace(kernel=True)
    display.display = lambda data, **kw: calls.append(("create", data, kw))
    display.update_display = lambda data, **kw: calls.append(("update", data, kw))
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(events, "_FALLBACK_CONSOLE", None)
    logging.getLogger("autoforecast.stages").setLevel(logging.NOTSET)
    if explicit_setup:
        configure_logging(interval=10)
    now = [100.0]
    monkeypatch.setattr(events.time, "monotonic", lambda: now[0])
    path = tmp_path / "audit.jsonl"
    with events.Execution(path, ["verify"]) as run:
        with events._stage(path, "verify"):
            with events.operation("Download", dataset="example") as progress:
                initial = len(records(path))
                progress(completed=600, total=1000, unit="bytes")
                before = len(calls)
                now[0] += 10
                progress(completed=600, total=1000, unit="bytes")
                assert len(calls) == before + 1
                assert "60.0%; remaining=400 bytes" in calls[-1][1]["text/plain"]
                configure_logging(interval=10)
                now[0] += 10
                progress(completed=700, total=1000, unit="bytes")
                assert "70.0%; remaining=300 bytes" in calls[-1][1]["text/plain"]
                assert len(records(path)) == initial
        assert "1/1 (100.0% of stages completed)" in calls[-1][1]["text/plain"]
    assert [c[0] for c in calls].count("create") == 1
    assert len({c[2]["display_id"] for c in calls}) == 1
    assert calls[-1][2]["display_id"] == run.display_id
    assert run.thread is not None and not run.thread.is_alive()
    assert run.result == "PASS"
    with events.Execution(tmp_path / "next.jsonl", ["verify"]) as next_run:
        with events._stage(tmp_path / "next.jsonl", "verify"):
            pass
    assert [c[0] for c in calls].count("create") == 2
    assert next_run.display_id != run.display_id


@pytest.mark.parametrize("error", [ValueError("original"), KeyboardInterrupt(), SystemExit(1)])
def test_failure_keeps_original_exception_counters_and_no_later_success(tmp_path, error):
    path = tmp_path / "audit.jsonl"
    with pytest.raises(type(error)) as caught:
        with events.Execution(path, ["copy", "verify"]) as run:
            with events._stage(path, "copy"):
                with events.operation("Read") as progress:
                    progress(completed=400, unit="rows")
                    raise error
    assert caught.value is error
    assert "400 rows; total=unknown" in run.render()
    assert "0/2 (0.0% of stages completed)" in run.render()
    assert run.states["verify"] == "WAITING"
    assert records(path)[-1]["status"] == ("FAIL" if isinstance(error, Exception) else "CANCELLED")
    assert all(r["status"] != "PASS" for r in records(path))
    assert events._CURRENT.get() is None and events._OPERATION.get() is None


def test_audit_failure_does_not_replace_callback_failure(tmp_path, monkeypatch):
    import mlops._safe_files as files
    path = tmp_path / "audit.jsonl"
    error = ValueError("callback failed")
    with pytest.raises(ValueError) as caught:
        with events.Execution(path, ["fit"]):
            with events._stage(path, "fit"):
                monkeypatch.setattr(files, "append_audit_bytes",
                                    lambda *_: (_ for _ in ()).throw(OSError("audit failed")))
                raise error
    assert caught.value is error


def test_nested_retry_reuse_and_parallel_completion_count_once(tmp_path):
    path = tmp_path / "audit.jsonl"
    with events.Execution(path, ["a", "b"]) as run:
        events._stage_event(path, "a", "START", attempt=1)
        events._stage_event(path, "b", "START")
        with events.operation("child"):
            assert "0/2" in run.render()
        events._stage_event(path, "b", "PASS")
        events._stage_event(path, "a", "FAIL", attempt=1)
        events._stage_event(path, "a", "START", attempt=2)
        events._stage_event(path, "a", "PASS", reused="verified", attempt=2)
        events._stage_event(path, "a", "PASS", reused="verified", attempt=2)
        assert "2/2 (100.0%" in run.render()
    assert run.result == "PASS"


def test_concurrent_attempts_do_not_share_panels_or_evidence(tmp_path):
    configure_logging(stream=io.StringIO())
    panels = []
    events._console().display = (
        lambda data, **kw: panels.append(kw["display_id"]),
        lambda data, **kw: panels.append(kw["display_id"]),
    )
    barrier = threading.Barrier(2)
    def work(name):
        path = tmp_path / (name + ".jsonl")
        with events.Execution(path, ["fit"], attempt_id=name):
            with events._stage(path, "fit"):
                barrier.wait(timeout=5)
                events.note("input", model_id=name)
        return records(path)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(work, ["first", "second"]))
    assert {r["attempt_id"] for r in results[0]} == {"first"}
    assert {r["attempt_id"] for r in results[1]} == {"second"}
    assert len(set(panels)) == 2


def test_display_error_is_harmless_and_plain_output_has_no_ansi(tmp_path):
    stream = io.StringIO()
    configure_logging(stream=stream, progress="plain")
    path = tmp_path / "audit.jsonl"
    with events.Execution(path, ["read"]) as run:
        with events._stage(path, "read"):
            with events.operation("Unknown rows") as progress:
                progress(completed=12, unit="rows")
                assert "12 rows; total=unknown" in run.render()
                assert "whole-run ETA: unavailable" in run.render()
        events._console().display = (lambda *a, **k: 1 / 0, lambda *a, **k: 1 / 0)
    assert run.result == "PASS"
    assert "\x1b" not in stream.getvalue()
    assert "Completed: Stage 1: read (pass)" in stream.getvalue()


@pytest.mark.parametrize("interval", [0, -1, float('nan'), float('inf'), True, '10'])
def test_bad_interval_refused(interval):
    with pytest.raises(ValueError):
        configure_logging(interval=interval)


def test_hash_progress_keeps_logical_digest_and_uses_unknown_row_total(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.benchmark import parquet_content_hash
    source = tmp_path / 'data.parquet'
    pq.write_table(pa.table({'key': ['a', 'b'], 'value': [-2.0, 3.0]}), source)
    before = parquet_content_hash(source)
    path = tmp_path / 'audit.jsonl'
    with events.Execution(path, ['hash']):
        with events._stage(path, 'hash'):
            after = parquet_content_hash(source)
    assert before == after
    assert source.read_bytes()  # The source is still present.
    assert any(r['stage'] == 'Calculate logical row SHA (driver)' for r in records(path))


def test_sensitive_fields_and_url_credentials_are_not_retained(tmp_path):
    path = tmp_path / 'audit.jsonl'
    with events.Execution(path, ['configuration']):
        with events._stage(path, 'configuration'):
            events.note('inputs', parameters={'api_key': 'secret-value', 'level': -5},
                        destination='https://user:password@example.test/path?token=private')
    text = path.read_text()
    assert 'secret-value' not in text and 'password' not in text and '?token=private' not in text
    assert '[value retained in configuration evidence]' in text and '[redacted location]' in text
    assert records(path)[1]['parameters']['level'] == '[value retained in configuration evidence]'


def test_heartbeat_updates_display_but_not_durable_audit(tmp_path):
    configure_logging(stream=io.StringIO(), interval=0.01)
    refreshed = threading.Event()
    handler = events._console()
    handler.display = (lambda *_a, **_k: None, lambda *_a, **_k: refreshed.set())
    path = tmp_path / 'audit.jsonl'
    with events.Execution(path, ['waiting']) as run:
        with events._stage(path, 'waiting'):
            assert refreshed.wait(timeout=2)
            assert len(records(path)) == 1
    assert not run.thread.is_alive()


def test_worker_standalone_scope_restores_parent_identity(tmp_path):
    with events.Execution(tmp_path / 'driver.jsonl', ['models'], attempt_id='driver') as driver:
        with events.standalone_attempt(tmp_path, 'worker', separate=True,
                                       campaign_id='campaign', model_variant='variant'):
            assert events._CURRENT.get() is not driver
            events.bind_identity(workflow_id='workflow')
            events.note('callback', trial_id='trial', window='1')
        assert events._CURRENT.get() is driver
        assert 'workflow_id' not in driver.identity
    rows = records(next((tmp_path / 'execution-attempts').glob('*.jsonl')))
    assert all(r['campaign_id'] == 'campaign' and r['model_variant'] == 'variant' for r in rows)
    assert rows[-1]['workflow_id'] == 'workflow'


@pytest.fixture
def default_progress(monkeypatch):
    loggers = [logging.getLogger(name) for name in ("", "autoforecast", "autoforecast.stages")]
    saved = [
        (
            logger,
            logger.handlers[:],
            logger.filters[:],
            logger.level,
            logger.propagate,
            logger.disabled,
        )
        for logger in loggers
    ]
    disabled = logging.getLogger().manager.disable
    monkeypatch.setattr(events, "_FALLBACK_CONSOLE", None)
    monkeypatch.delenv("AUTOFORECAST_LOG_LEVEL", raising=False)
    for logger in loggers:
        logger.handlers, logger.filters, logger.propagate, logger.disabled = [], [], True, False
        logger.setLevel(logging.WARNING if logger is loggers[0] else logging.NOTSET)
    logging.disable(logging.NOTSET)
    try:
        yield
    finally:
        for logger, handlers, filters, level, propagate, quiet in saved:
            logger.handlers, logger.filters = handlers, filters
            logger.propagate, logger.disabled = propagate, quiet
            logger.setLevel(level)
        logging.disable(disabled)


@pytest.mark.usefixtures("default_progress")
@pytest.mark.parametrize("level", [logging.ERROR, logging.CRITICAL])
def test_root_explicit_quiet_hides_automatic_panels(tmp_path, level):
    logging.getLogger().setLevel(level)
    logging.getLogger().addHandler(logging.StreamHandler(io.StringIO()))
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        with events.Execution(tmp_path / "audit.jsonl", ["fit"]):
            with events._stage(tmp_path / "audit.jsonl", "fit"):
                pass
    assert text.getvalue() == ""


@pytest.mark.usefixtures("default_progress")
@pytest.mark.parametrize("level", [logging.WARNING, logging.ERROR, logging.CRITICAL])
def test_application_handler_quiet_is_not_bypassed_by_panel(tmp_path, level):
    logger = logging.getLogger("autoforecast.stages")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(level)
    logger.addHandler(handler)
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        with events.Execution(tmp_path / "audit.jsonl", ["fit"]):
            with events._stage(tmp_path / "audit.jsonl", "fit"):
                pass
    assert text.getvalue() == ""


@pytest.mark.usefixtures("default_progress")
def test_planning_event_visible_with_default_root_application_handler(tmp_path, monkeypatch):
    logging.getLogger().addHandler(logging.StreamHandler(io.StringIO()))
    data = {"dataset": {}, "recipe": {}, "source": {}, "runtime": {}}
    monkeypatch.setattr(registration, "_configuration", lambda _, *, connections=None: (data, tmp_path))
    stop = RuntimeError("stop at preparation validation")

    class Hook:
        def validate(self, _):
            raise stop

        def prepare(self, _):
            pass

    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        with pytest.raises(RuntimeError) as caught:
            registration.register_study("config.json", prepare=Hook())
    assert caught.value is stop
    assert "START Preparing execution plan" in text.getvalue()


@pytest.mark.usefixtures("default_progress")
def test_console_discovery_error_does_not_abort_operation(tmp_path, monkeypatch):
    class Output(io.StringIO):
        def isatty(self):
            raise OSError("output capability unavailable")

    import sys

    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setattr(sys, "stdout", Output())
    marker = []
    with events.Execution(tmp_path / "audit.jsonl", ["fit"]):
        marker.append("entered")
        with events._stage(tmp_path / "audit.jsonl", "fit"):
            pass
    assert marker == ["entered"]


@pytest.mark.usefixtures("default_progress")
@pytest.mark.parametrize("mode", ["auto", "plain", "off"])
def test_fallback_terminal_handoff_clears_prior_panel(tmp_path, monkeypatch, mode):
    class TTY(io.StringIO):
        def isatty(self):
            return True

    import sys

    monkeypatch.setitem(sys.modules, "IPython", None)
    stream = TTY()
    with contextlib.redirect_stdout(stream):
        with events.Execution(tmp_path / "audit.jsonl", ["fit"]) as execution:
            with events._stage(tmp_path / "audit.jsonl", "fit"):
                fallback = events._FALLBACK_CONSOLE
                assert fallback.live and execution.display_id in fallback.panels
                offset = len(stream.getvalue())
                events.configure_logging(progress=mode)
                assert "\x1b[2K" in stream.getvalue()[offset:]
        assert execution.display_id not in fallback.panels


@pytest.mark.usefixtures("default_progress")
@pytest.mark.parametrize("policy", ["logger_filter", "handler_filter", "null_handler"])
def test_application_filter_quiet_is_not_bypassed(tmp_path, policy):
    logger = logging.getLogger("autoforecast.stages")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = (
        logging.NullHandler() if policy == "null_handler" else logging.StreamHandler(io.StringIO())
    )
    logger.addHandler(handler)
    if policy == "logger_filter":
        logger.addFilter(lambda _: False)
    if policy == "handler_filter":
        handler.addFilter(lambda _: False)
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        with events.Execution(tmp_path / "audit.jsonl", ["fit"]):
            with events._stage(tmp_path / "audit.jsonl", "fit"):
                pass
    assert text.getvalue() == ""


@pytest.mark.usefixtures("default_progress")
@pytest.mark.parametrize("explicit_stream", [False, True])
def test_repeat_customization_after_previous_cell_stream_closed(tmp_path, explicit_stream):
    previous, current = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(previous):
        configure_logging(stream=previous if explicit_stream else None)
    previous.close()
    with contextlib.redirect_stdout(current):
        configure_logging(stream=current if explicit_stream else None)
        with events.Execution(tmp_path / "audit.jsonl", ["fit"]):
            with events._stage(tmp_path / "audit.jsonl", "fit"):
                pass
    assert "1/1 (100.0%" in current.getvalue()
    assert len(logging.getLogger("autoforecast.stages").handlers) == 1


@pytest.mark.usefixtures("default_progress")
@pytest.mark.parametrize("level", [logging.INFO, logging.ERROR])
def test_notebook_discovery_failure_preserves_operation(tmp_path, monkeypatch, level):
    from types import ModuleType

    calls = []
    ipython, display = ModuleType("IPython"), ModuleType("IPython.display")
    def unavailable():
        calls.append("discovery")
        raise RuntimeError("frontend unavailable")
    ipython.get_ipython = unavailable
    display.display = display.update_display = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    original = ValueError("model failure")
    with pytest.raises(ValueError) as caught:
        with events.Execution(tmp_path / "audit.jsonl", ["fit"], level=level):
            with events._stage(tmp_path / "audit.jsonl", "fit"):
                raise original
    assert caught.value is original
    assert bool(calls) == (level == logging.INFO)
    assert events._FALLBACK_CONSOLE is None


@pytest.mark.usefixtures("default_progress")
def test_closed_stdout_reconfiguration_with_live_fallback(tmp_path, monkeypatch):
    class TTY(io.StringIO):
        def isatty(self):
            return True

    import sys

    monkeypatch.setitem(sys.modules, "IPython", None)
    first, second = TTY(), TTY()
    with contextlib.redirect_stdout(first):
        with events.Execution(tmp_path / "one.jsonl", ["work"]) as execution:
            with events._stage(execution.audit, "work"):
                first.close()
                sys.stdout = second
                events.configure_logging(progress="plain")
    assert execution.result == "PASS"
    assert (
        events._CURRENT.get() is None
        and execution.thread is not None
        and not execution.thread.is_alive()
    )
    assert "work" in second.getvalue()


@pytest.mark.usefixtures("default_progress")
@pytest.mark.parametrize("mode", ["plain", "off"])
@pytest.mark.parametrize("broken", ["isatty", "IPython"])
def test_explicit_plain_or_off_avoids_old_auto_discovery(tmp_path, monkeypatch, mode, broken):
    import sys
    from types import ModuleType

    class TTY(io.StringIO):
        def isatty(self):
            return True

    class NoCapability(io.StringIO):
        def isatty(self):
            raise OSError("unsupported terminal capability")

    monkeypatch.setitem(sys.modules, "IPython", None)
    old = TTY()
    with contextlib.redirect_stdout(old):
        with events.Execution(tmp_path / "audit.jsonl", ["fit"]):
            with events._stage(tmp_path / "audit.jsonl", "fit"):
                pass
    assert events._FALLBACK_CONSOLE is not None
    current = NoCapability() if broken == "isatty" else io.StringIO()
    if broken == "IPython":
        module, display = ModuleType("IPython"), ModuleType("IPython.display")

        def fail():
            raise RuntimeError("frontend discovery unavailable")

        module.get_ipython = fail
        display.display = display.update_display = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "IPython", module)
        monkeypatch.setitem(sys.modules, "IPython.display", display)
    with contextlib.redirect_stdout(current):
        events.configure_logging(progress=mode)
        assert events._console().progress == mode


@pytest.mark.parametrize("value", [None, [], [1], True, 1, "invalid"])
def test_local_audit_reader_rejects_non_object_events(tmp_path, value):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"stage": "first"}) + "\n" + json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="JSON objects"):
        events.read_execution_events(path)


def test_local_audit_reader_keeps_objects_and_ignores_blank_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    expected = [{"stage": "first", "status": "START"}, {"stage": "first", "status": "PASS"}]
    path.write_text("\n" + "\n\n".join(json.dumps(item) for item in expected) + "\n")
    assert events.read_execution_events(path) == expected


@pytest.mark.parametrize("display,panel,info", [
    ("progress", True, False), ("logs", False, True), ("both", True, True),
])
def test_display_chooses_console_content_and_keeps_every_audit_event(
    tmp_path, monkeypatch, display, panel, info,
):
    monkeypatch.setattr(events, "_DISPLAY", None)
    stream = io.StringIO()
    configure_logging(stream=stream, display=display,
                      **({} if display == "logs" else {"progress": "plain"}))
    path = tmp_path / "audit.jsonl"
    with events.Execution(path, ["read"]):
        with events._stage(path, "read") as outcome:
            events.note("Checked inputs", rows=3)
            outcome["status"] = "UNAVAILABLE"
    output = stream.getvalue()
    assert ("RUN {" in output) is panel
    assert ("INFO Checked inputs" in output) is info
    assert "UNAVAILABLE read" in output  # Warnings stay visible in every display.
    assert [(row["stage"], row["status"]) for row in records(path)] == [
        ("read", "START"), ("Checked inputs", "INFO"), ("read", "UNAVAILABLE"),
    ]


@pytest.mark.parametrize("display,progress", [
    ("progress", "off"), ("both", "off"), ("logs", "auto"), ("logs", "plain"), ("quiet", None),
])
def test_contradictory_display_options_are_refused(monkeypatch, display, progress):
    monkeypatch.setattr(events, "_DISPLAY", None)
    with pytest.raises(ValueError, match="display"):
        configure_logging(stream=io.StringIO(), display=display, progress=progress)
    assert events._DISPLAY is None


def test_display_is_fixed_when_a_run_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_DISPLAY", None)
    stream = io.StringIO()
    configure_logging(stream=stream, progress="plain", display="progress")
    with events.Execution(tmp_path / "first.jsonl", ["read"]):
        configure_logging(stream=stream, progress="plain", display="both")
        events.note("Inside the first run")
    with events.Execution(tmp_path / "second.jsonl", ["read"]):
        events.note("Inside the second run")
    assert "Inside the first run" not in stream.getvalue()
    assert "Inside the second run" in stream.getvalue()


def test_display_precedence_and_runtime_validation(monkeypatch):
    monkeypatch.setattr(events, "_DISPLAY", None)
    assert events.resolve_display({}) == "both"
    assert events.resolve_display({"display": "logs"}) == "logs"
    configure_logging(stream=io.StringIO(), display="progress")
    assert events.resolve_display({"display": "logs"}) == "progress"
    configure_logging(stream=io.StringIO(), progress="off")  # Omitting display restores the default.
    assert events.resolve_display({}) == "both"
    assert registration._runtime({"display": "progress"})["display"] == "progress"
    with pytest.raises(ValueError, match="display"):
        registration._runtime({"display": "quiet"})
