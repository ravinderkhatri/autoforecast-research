"""Operational progress has its own counters and never changes model evidence."""

import concurrent.futures
import io
import json
import logging
import subprocess
import sys
import threading

import pytest

from autoforecast import _execution as events
from autoforecast import configure_logging


@pytest.fixture(autouse=True)
def isolated_console():
    logger = logging.getLogger("autoforecast.stages")
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    yield
    logger.handlers, logger.level, logger.propagate = previous


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fresh_process_explicit_setup_and_repeat_preserve_root():
    script = '''
import io, logging, tempfile
from pathlib import Path
root = logging.getLogger()
handler = logging.StreamHandler(io.StringIO())
root.addHandler(handler)
root.setLevel(logging.ERROR)
from autoforecast import configure_logging
assert logging.getLogger('autoforecast.stages').handlers == []
from autoforecast._execution import _stage_event
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


def test_notebook_same_display_throttle_and_no_progress_audit(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    calls = []
    ipython, display = ModuleType("IPython"), ModuleType("IPython.display")
    ipython.get_ipython = lambda: SimpleNamespace(kernel=True)
    display.display = lambda data, **kw: calls.append(("create", data, kw))
    display.update_display = lambda data, **kw: calls.append(("update", data, kw))
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
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
