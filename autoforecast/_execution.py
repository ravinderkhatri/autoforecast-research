"""Shared operational events, separate from model and dataset identity."""

from __future__ import annotations

import contextvars
import datetime as dt
import functools
import hashlib
import inspect
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from contextlib import contextmanager, nullcontext, suppress
from typing import Any

_LOG_LEVEL = contextvars.ContextVar("autoforecast_log_level", default=logging.INFO)
_CURRENT: contextvars.ContextVar[Execution | None] = contextvars.ContextVar(
    "autoforecast_execution", default=None,
)
_OPERATION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "autoforecast_operation", default=None,
)
_AUDITING = contextvars.ContextVar("autoforecast_writing_audit", default=False)
_SETUP_LOCK = threading.RLock()
_FALLBACK_CONSOLE = None
# Each remote audit event costs an upload and a readback (about 0.5 s on a volume).
# A running execution uploads them off the main path, one at a time and in order,
# so every saved audit is a complete prefix that readers in other processes accept.
_DISPLAYS = ("progress", "logs", "both")
# An explicit configure_logging(display=...) choice; otherwise runtime.display applies.
_DISPLAY = None


def resolve_level(explicit=None, runtime=None):
    """Use the same explicit, environment, configuration, default order everywhere."""
    name = explicit if explicit is not None else os.environ.get(
        "AUTOFORECAST_LOG_LEVEL", (runtime or {}).get("log_level", "INFO"),
    )
    if not isinstance(name, str) or name not in logging.getLevelNamesMapping():
        raise ValueError("log_level must be a standard logging level name")
    return logging.getLevelNamesMapping()[name]


def resolve_display(runtime=None):
    """Use configure_logging's display, then runtime.display, then both."""
    display = _DISPLAY if _DISPLAY is not None else (runtime or {}).get("display", "both")
    if display not in _DISPLAYS:
        raise ValueError("display must be progress, logs or both")
    return display


def validate_model_label(value):
    """Accept a short public name for logs, independent of model configuration."""
    if value is not None and (
        not isinstance(value, str) or not value.strip()
        or len(value) > 100 or not value.isprintable()
    ):
        raise ValueError(
            "model_label must be a nonblank, printable string of at most 100 characters",
        )
    return value


def _text(value):
    # One fact cannot inject extra log lines or terminal control sequences.
    return str(value).replace("\n", " ").replace("\r", " ").replace("\x1b", "")[:1000]


def _parameter_values(value):
    # Any custom value, including a number, may be a credential.
    # Keep parameter names here and values in the existing configuration evidence.
    if isinstance(value, dict):
        return {str(k): _parameter_values(v) for k, v in value.items()}
    return "[value retained in configuration evidence]"


def _safe(value: Any) -> Any:
    """Exclude common secret fields and URL credentials from operational evidence."""
    if isinstance(value, dict):
        return {str(k): _parameter_values(v) if k in {
            "parameters", "model_parameters", "training_parameters", "selected_parameters",
        } else "[redacted]" if re.search(
            r"password|secret|token|credential|auth|(?:api|account|private|access)[_-]?key"
            r"|connection[_-]?string", str(k), re.I,
        ) else _safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_safe(v) for v in value]
    if isinstance(value, str) and any(marker in value for marker in ("@", "?", "#")):
        # A normalized URI and a local filename can look identical. Omit the whole
        # ambiguous value instead of leaking a second URI or inventing a usable path.
        if re.search(r"(?<![a-zA-Z0-9+.-])[a-zA-Z][a-zA-Z0-9+.-]*:/", value):
            return "[redacted location]"
    return value


def _message(event):
    identity = " ".join(f"{k}={_text(event[k])}" for k in (
        "attempt_id", "workflow_id", "model_id", "model_label", "trial_id", "window",
    ) if event.get(k) is not None)
    facts = " | ".join(f"{k}={_text(v)}" for k, v in event.items() if k not in {
        "timestamp", "stage", "status", "attempt_id", "workflow_id", "model_id",
        "model_label", "trial_id", "window", "operation_id", "parent_operation", "parent_audit",
        "plan_stage",
    } and v is not None)
    return f"{event['status']} {event['stage']} | {identity}" + (f" | {facts}" if facts else "")


class _Console(logging.StreamHandler):
    """Shared progress renderer; notebook refreshes never clear cell output."""

    def __init__(self, stream, progress, interval):
        self.dynamic_stream = stream is None
        super().__init__(sys.stdout if stream is None else stream)
        self.interval = interval
        self.live = False
        self.live_rows = 0
        self.panels = {}
        self.set_mode(stream, progress)
        self.setFormatter(logging.Formatter("%(message)s"))

    def set_mode(self, stream, progress):
        assert self.lock is not None
        with self.lock:
            self.dynamic_stream = stream is None
            if self.dynamic_stream and self.stream is not sys.stdout:
                self.stream = sys.stdout
                self.live = False
                self.live_rows = 0
            same_terminal = getattr(self, "terminal", False)
            old_stream = self.stream
            with suppress(Exception):
                self._clear_panel()
            self.live = False
            self.live_rows = 0
            self._set_mode(stream, progress)
            if not (same_terminal and self.terminal and old_stream is self.stream):
                self.panels.clear()
            if self.terminal:
                self._draw_panels()

    def _sync_stream(self):
        # A later notebook cell can own a different stdout, or close the old one.
        if self.dynamic_stream and self.stream is not sys.stdout:
            self.stream = sys.stdout
            self.live = False
            self.live_rows = 0
            self._set_mode(None, self.progress)

    def _set_mode(self, stream, progress):
        self.progress = progress
        self.display = None
        self.terminal = False
        if stream is not None and stream is not self.stream:
            with suppress(Exception):
                self.flush()
            self.stream = stream
        if progress == "auto" and stream is None:
            with suppress(ImportError, AttributeError):
                from IPython import get_ipython
                from IPython.display import display, update_display

                if getattr(get_ipython(), "kernel", None) is not None:
                    self.display = (display, update_display)
        if progress == "auto" and self.display is None:
            self.terminal = bool(getattr(self.stream, "isatty", lambda: False)())

    def emit(self, record):
        # A run copies its display when it starts; later configuration affects later runs.
        execution = _CURRENT.get()
        display = execution.display if execution is not None else resolve_display()
        if display == "progress" and record.levelno < logging.WARNING:
            return
        assert self.lock is not None
        with self.lock:
            self._sync_stream()
            self._clear_panel()
            super().emit(record)
            if self.terminal:
                self._draw_panels()

    def _clear_panel(self):
        if self.live:
            self.stream.write("\r\x1b[2K")
            for _ in range(self.live_rows - 1):
                self.stream.write("\x1b[1A\r\x1b[2K")
        self.live = False
        self.live_rows = 0

    def _draw_panels(self):
        # Explicit short lines avoid terminal auto-wrap, including at the right edge.
        # Escape wide characters so the row count stays correct without extra packages.
        width = max(1, shutil.get_terminal_size().columns - 1)
        lines = []
        for text in self.panels.values():
            text = text.replace("\n", " | ").encode("ascii", "backslashreplace").decode()
            lines.extend(text[start:start + width] for start in range(0, len(text), width))
        if lines:
            self.stream.write("\r\x1b[2K" + "\n".join(lines))
        self.live_rows = len(lines)
        self.live = bool(lines)
        self.flush()

    def panel(self, execution, text, *, final=False, transition=False):
        assert self.lock is not None
        with self.lock:
            self._sync_stream()
            if final:
                self.panels.pop(execution.display_id, None)
            if self.progress == "off":
                return
            if self.display:
                first, update = self.display
                try:
                    (update if execution.displayed else first)(
                        {"text/plain": text}, raw=True, display_id=execution.display_id,
                    )
                except Exception:
                    self.display = None  # A frontend without updates still gets readable text.
                else:
                    execution.displayed = True
                    return
            if self.terminal:
                self._clear_panel()
                self.panels[execution.display_id] = text
                if final:
                    self.stream.write("\r\x1b[2K" + text + "\n")
                    self.panels.pop(execution.display_id)
                self._draw_panels()
            else:
                if not transition:
                    text = "\n".join(line for line in text.splitlines()
                                     if line.startswith(("RUN ", "Stage ", "Operation:")))
                self.stream.write(text + "\n")
                self.flush()


def configure_logging(*, stream=None, progress=None, interval=10.0, display=None):
    """Show library events and progress without changing the root logger.

    Basic operation progress appears automatically. Call this for customization.
    Repeating this call updates the same handler.
    ``display`` chooses the console content for runs started afterwards: ``'progress'``
    (the panel, plus warnings and errors), ``'logs'`` (event lines, no panel) or
    ``'both'``. Omit it to use runtime.display, or both. Saved audit events never change.
    ``progress`` chooses the renderer: ``'auto'``, ``'plain'`` for job logs, or ``'off'``.
    Runtime log_level controls verbosity. Applications may instead configure Python logging.
    """
    global _FALLBACK_CONSOLE, _DISPLAY
    if display is not None and display not in _DISPLAYS:
        raise ValueError("display must be progress, logs or both")
    if progress is None:
        progress = "off" if display == "logs" else "auto"
    if progress not in {"auto", "plain", "off"}:
        raise ValueError("progress must be auto, plain or off")
    if display == "logs" and progress != "off":
        raise ValueError("display='logs' shows no panel; omit progress or use progress='off'")
    if display in {"progress", "both"} and progress == "off":
        raise ValueError(f"display={display!r} needs progress='auto' or 'plain'")
    if isinstance(interval, bool) or not isinstance(interval, int | float) or not (
        0 < interval < float("inf")
    ):
        raise ValueError("interval must be a finite positive number")
    with _SETUP_LOCK:
        logger = logging.getLogger("autoforecast.stages")
        owned = next((h for h in logger.handlers if isinstance(h, _Console)), None)
        if owned is None:
            owned = _FALLBACK_CONSOLE
        if owned is None:
            owned = _Console(stream, progress, interval)
        else:
            # Preserve the handler and active display identities on repeated setup.
            owned.interval = interval
            owned.set_mode(stream, progress)
        if owned not in logger.handlers:
            logger.addHandler(owned)
        if owned is _FALLBACK_CONSOLE:
            _FALLBACK_CONSOLE = None
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        _DISPLAY = display


def _console():
    global _FALLBACK_CONSOLE
    logger = logging.getLogger("autoforecast.stages")
    if logger.disabled or logger.manager.disable >= logging.INFO:
        return None
    parent = logger
    # Root WARNING is a Python/platform default, not an operation quiet setting.
    while parent is not None and parent is not logging.getLogger():
        if parent.level != logging.NOTSET:
            if parent.level > logging.INFO:
                return None
            break
        parent = parent.parent
    with _SETUP_LOCK:
        owned = next((h for h in logger.handlers if isinstance(h, _Console)), None)
        if owned is not None:
            return owned
        parent = logger
        while parent is not None and parent is not logging.getLogger():
            # An application that configures library handlers owns this output.
            if parent.handlers or parent.filters or not parent.propagate:
                return None
            parent = parent.parent
        if logging.getLogger().level > logging.WARNING:
            return None
        if _FALLBACK_CONSOLE is None:
            # A broken optional display must not stop forecasting or hide its error.
            with suppress(Exception):
                _FALLBACK_CONSOLE = _Console(None, "auto", 10.0)
        return _FALLBACK_CONSOLE


def _emit(event, level, *, announce=False):
    severity = (logging.ERROR if event["status"] in {"FAIL", "CANCELLED", "UNVERIFIED"}
                else logging.WARNING if event["status"] == "UNAVAILABLE"
                or event.get("warnings") else logging.INFO)
    if severity >= level:
        with suppress(Exception):
            logger = logging.getLogger("autoforecast.stages")
            logger.log(severity, "%s", _message(event))
            # Registration can validate source plans before it has an audit directory.
            # Show that initial step when the platform's default logger hides INFO.
            # display='logs' has no panel, so its event lines are shown the same way.
            execution = _CURRENT.get()
            lines = (execution is not None and execution.display == "logs"
                     and severity < logging.WARNING)
            if ((announce or lines)
                    and (not logger.hasHandlers() or not logger.isEnabledFor(severity))):
                handler = _console()
                if handler is not None and handler is _FALLBACK_CONSOLE:
                    handler.handle(logger.makeRecord(
                        logger.name, severity, __file__, 0, _message(event), (), None,
                    ))


class Execution:
    """A finite parent-stage plan. Nested work never changes its denominator."""

    def __init__(self, audit, stages, *, level=logging.INFO, display=None, **identity):
        self.audit = pathlib.Path(audit)
        if display is not None and display not in _DISPLAYS:
            raise ValueError("display must be progress, logs or both")
        self.display = display
        self.stages = tuple(stages)
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("execution stages must have distinct names")
        self.identity = {"attempt_id": uuid.uuid4().hex, **identity}
        self.level = level
        self.states = dict.fromkeys(self.stages, "WAITING")
        self.stage_ids = {name: uuid.uuid4().hex for name in self.stages}
        self.operation_attempts = {}
        self.started = time.monotonic()
        self.starts = {}
        self.operations = {}
        self.lock = threading.RLock()
        self.audit_segments = {}
        self.audit_closed = False
        self.stop = threading.Event()
        self.thread = None
        self.display_id = "autoforecast-" + uuid.uuid4().hex
        self.displayed = False
        self.last_refresh = self.started
        self.result = "RUNNING"
        self.receipt = {}
        self.audit_error = None
        self.uploader = None

    def write_event(self, path, event):
        """Keep one writer and ordered, retry-stable event bytes for each remote audit.

        Sequence, hash link and bytes are fixed and recorded here before any upload. The
        first event claims the audit for this writer before returning. While the execution
        is running, later events of its own audit upload in the background, in sequence
        order. After a failed upload nothing later is uploaded until the failed event is
        retried with identical bytes, before the next event or at close. A hard process
        failure can lose the unsaved tail; what was saved remains a valid prefix. Other
        audit paths, such as a run bundle's events, stay synchronous because their owners
        may seal them before this execution ends.
        """
        from mlops import _volume_files

        with self.lock:
            if self.audit_closed:
                raise ValueError("execution audit is closed")
            state = self.audit_segments.setdefault(path, {
                "owner": uuid.uuid4().hex, "sequence": 0, "previous": None, "pending": None,
                "uploads": {}, "blocked": False,
            })

            def acknowledge():
                sha = _volume_files.publish_audit_segment(
                    path, state["pending"], owner=state["owner"], sequence=state["sequence"],
                )
                state.update(sequence=state["sequence"] + 1, previous=sha, pending=None)

            if state["pending"] is not None:
                acknowledge()
            self._settle(path, state, wait=False)
            event["audit_segment"] = {key: state[key] for key in ("owner", "sequence", "previous")}
            payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
            if state["sequence"] == 0 or self.uploader is None or path != self.audit:
                state["pending"] = payload
                acknowledge()
                return
            # Record the event before submitting it, so an interruption between the two
            # leaves an unsubmitted entry that _settle uploads, never a reused sequence.
            # One update installs the entry and advances the sequence together: an
            # interruption leaves neither or both, so the closure count stays right.
            sequence, entry = state["sequence"], [payload, None]
            previous = hashlib.sha256(payload).hexdigest()
            state.update(uploads={**state["uploads"], sequence: entry},
                         sequence=sequence + 1, previous=previous)
            entry[1] = self.uploader.submit(self._upload, path, state, sequence, payload)

    @staticmethod
    def _upload(path, state, sequence, payload):
        from mlops import _volume_files

        if state["blocked"]:
            raise OSError("an earlier audit event is not saved yet")
        try:
            _volume_files.publish_audit_segment(path, payload, owner=state["owner"],
                                                sequence=sequence)
        except BaseException:
            state["blocked"] = True
            raise

    def _settle(self, path, state, *, wait):
        """Save finished or failed uploads in order; with ``wait``, save every event.

        A failed or never-submitted event is retried here with identical bytes before any
        later one, so the saved audit stays a prefix.
        """
        from mlops import _volume_files

        for sequence, (payload, future) in sorted(state["uploads"].items()):
            if future is not None and not future.done():
                if not wait:
                    break
                # Wait without re-raising the worker's error, so the caller's own
                # interruption still propagates.
                futures_wait([future])
            if future is None or future.exception() is not None:
                _volume_files.publish_audit_segment(path, payload, owner=state["owner"],
                                                    sequence=sequence)
            del state["uploads"][sequence]
        if not state["uploads"]:
            state["blocked"] = False

    def flush_audit(self, path):
        """Finish background uploads for one audit, for a reader inside this execution."""
        with self.lock:
            state = self.audit_segments.get(pathlib.Path(path))
            if state is not None:
                self._settle(pathlib.Path(path), state, wait=True)

    def observe(self, path, event):
        with self.lock:
            name, status = event["stage"], event["status"]
            if event.get("audit_error"):
                self.audit_error = event["audit_error"]
            operation = self.operations.get(event.get("operation_id"))
            if operation is not None:
                operation["status"] = status
                if status != "START":
                    elapsed = event.get("elapsed_seconds")
                    operation.setdefault("_frozen_elapsed", elapsed if elapsed is not None
                                         else time.monotonic() - operation["start"])
            planned = (pathlib.Path(path) == self.audit and name in self.states
                       and event.get("plan_stage", not event.get("operation_id")))
            if planned:
                self.states[name] = status
                if status == "START":
                    self.starts[name] = time.monotonic()
            self.refresh(force=planned)

    def _run_identity(self):
        """Name the input study separately when the saved result names another one."""
        identity = _safe(self.identity)
        if self.receipt.get("study_id") and identity.get("study_id") != self.receipt["study_id"]:
            if "study_id" in identity:
                identity["input_study_id"] = identity.pop("study_id")
        return identity

    def render(self):
        now = time.monotonic()
        completed = [s for s in self.stages if self.states[s] == "PASS"]
        skipped = [s for s in self.stages if self.states[s] == "SKIP"]
        running = [s for s in self.stages if self.states[s] == "START"]
        remaining = [s for s in self.stages if self.states[s] not in {"PASS", "SKIP"}]
        total = len(self.stages)
        def names(items):
            return " | ".join(f"Stage {self.stages.index(s) + 1}: {s}"
                              f" ({self.states[s].lower()})" for s in items) or "None"
        lines = [f"RUN {_text(self._run_identity())} | {self.result}"]
        for name in running:
            lines.append(f"Stage {self.stages.index(name) + 1}/{total}: {name} | "
                         f"elapsed={now - self.starts[name]:.1f}s")
        for op in self.operations.values():
            done, amount, unit = op.get("completed"), op.get("total"), op.get("unit")
            elapsed = op.get("_frozen_elapsed", now - op["start"])
            line = f"Operation: {op['name']} | elapsed={elapsed:.1f}s"
            if done is not None:
                line += (f" | {done} {unit}; total="
                         + (str(amount) if amount is not None else "unknown"))
                if amount is not None and amount > 0:
                    line += f" | {done / amount * 100:.1f}%; remaining={amount - done} {unit}"
            if done is None and op.get("status") in {None, "START"}:
                line += " | waiting; measurable progress unavailable"
            if "files" in op:
                line += f" | files={op['files']}/{op.get('files_total', 'unknown')}"
            if op.get("phase"):
                line += f" | {_text(op['phase'])}"
            if op.get("status"):
                line += f" | {op['status']}"
            lines.append(line)
        lines += ["Stage percentage: unavailable (operation counters are separate)",
                  f"Overall stages completed: {len(completed)}/{total} "
                  f"({100 * len(completed) / total if total else 0:.1f}% of stages completed)",
                  f"Resolved: {len(completed) + len(skipped)}/{total}; skipped={len(skipped)}; "
                  f"unfinished={len(remaining)}; running={len(running)}; "
                  f"waiting={sum(v == 'WAITING' for v in self.states.values())}",
                  f"Elapsed run: {now - self.started:.1f}s; whole-run ETA: unavailable",
                  "Completed: " + names(completed), "Skipped: " + names(skipped),
                  "Remaining (not completed): " + names(remaining)]
        if self.result != "RUNNING":
            lines.append(f"Audit location: {_text(_safe(str(self.audit)))}")
            lines.extend(f"{key}: {_text(value)}" for key, value in self.receipt.items())
            if self.audit_error:
                lines.append(f"Audit write failed: {self.audit_error}; "
                             "event storage could not be verified")
        return "\n".join(lines)

    def refresh(self, *, force=False, final=False):
        if self.level > logging.INFO or self.display == "logs":
            return
        handler = _console()
        now = time.monotonic()
        if handler is None:
            return
        if force or now - self.last_refresh >= handler.interval:
            self.last_refresh = now
            with suppress(Exception):
                handler.panel(self, self.render(), final=final, transition=force)

    def _heartbeat(self):
        while True:
            handler = _console()
            if self.stop.wait(handler.interval if handler else 10):
                break
            with self.lock:
                self.refresh()

    def __enter__(self):
        parent = _CURRENT.get()
        if parent:
            self.identity.setdefault("parent_attempt", parent.identity["attempt_id"])
        if self.display is None:
            self.display = parent.display if parent else resolve_display()
        # Allocate before installing any context, so a failure here leaves nothing to undo.
        self.uploader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autoforecast-audit")
        self.token = _CURRENT.set(self)
        self.level_token = _LOG_LEVEL.set(self.level)
        self.identity_token = _IDENTITY.set({})
        self.operation_token = _OPERATION.set(None)
        self.parent_token = _PARENT_AUDIT.set(None)
        try:
            if self.level <= logging.INFO and _console():
                self.thread = threading.Thread(target=self._heartbeat, daemon=True)
                self.thread.start()
        except BaseException:
            self.stop.set()
            # Nothing was submitted yet; keep the startup failure if shutdown is interrupted.
            with suppress(BaseException):
                self.uploader.shutdown(wait=True)
            self._restore_context()
            raise
        return self

    def __exit__(self, kind, error, traceback):
        # A failure already on its way out (the body's, else _finish's) wins over a
        # cleanup failure, and the context is restored whatever happens.
        failing = kind is not None
        try:
            with suppress(BaseException) if failing else nullcontext():
                self._finish(kind)
        except BaseException:
            failing = True
            raise
        finally:
            uploader, self.uploader = self.uploader, None
            try:
                with suppress(BaseException) if failing else nullcontext():
                    if uploader is not None:
                        uploader.shutdown(wait=True)
            finally:
                self._restore_context()

    def _restore_context(self):
        _PARENT_AUDIT.reset(self.parent_token)
        _OPERATION.reset(self.operation_token)
        _IDENTITY.reset(self.identity_token)
        _LOG_LEVEL.reset(self.level_token)
        _CURRENT.reset(self.token)

    def _finish(self, kind):
        self.stop.set()
        if self.thread:
            with suppress(BaseException) if kind is not None else nullcontext():
                self.thread.join()
        with self.lock:
            if kind:
                for name, status in list(self.states.items()):
                    if status == "START":
                        with suppress(BaseException):
                            _stage_event(self.audit, name,
                                         "CANCELLED" if not issubclass(kind, Exception) else "FAIL",
                                         error_type=kind.__name__)
            self.result = ("CANCELLED" if kind and not issubclass(kind, Exception) else "FAIL"
                           if kind else "PASS" if all(v == "PASS" for v in self.states.values())
                           else "RESOLVED" if all(v in {"PASS", "SKIP"}
                                                  for v in self.states.values()) else "INCOMPLETE")
            for path, segments in self.audit_segments.items():
                try:
                    self._settle(path, segments, wait=True)
                except Exception as failure:
                    # Show that saved evidence is incomplete before reporting the failure;
                    # an interruption propagates unchanged and leaves a valid prefix.
                    if self.result not in {"FAIL", "CANCELLED"}:
                        self.result = "UNVERIFIED"
                    self.audit_error = type(failure).__name__
                    # The panel may be hidden, so report it, and the saved result, as an
                    # event line as well.
                    with suppress(BaseException):
                        _emit(_safe({**self._run_identity(), **self.receipt,
                                     "stage": "Execution audit", "status": "UNVERIFIED",
                                     "audit_error": self.audit_error}), self.level)
                    self.refresh(force=True, final=True)
                    raise OSError("execution audit has an unacknowledged event") from failure
            state = self.audit_segments.get(self.audit)
            if state is not None:
                if state["pending"] is not None:
                    raise OSError("execution audit has an unacknowledged event")
                from mlops import _volume_files

                _volume_files.finish_audit(
                    self.audit, owner=state["owner"], count=state["sequence"],
                    last=state["previous"], status=self.result,
                )
            self.audit_closed = True
            self.refresh(force=True, final=True)


_IDENTITY = contextvars.ContextVar("autoforecast_operation_identity", default={})
_PARENT_AUDIT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "autoforecast_parent_audit", default=None,
)
_CORRELATION = {"study_id", "workflow_id", "run_id", "search_id", "trial_id", "window",
                "window_index", "window_total", "model_id", "model_label", "model_variant",
                "purpose", "origin"}


def _identity():
    execution = _CURRENT.get()
    return {**(execution.identity if execution else {}), **_IDENTITY.get()}


def _stage_event(path, stage, status, *, elapsed=None, _recorded=None, **context):
    """Retain lifecycle evidence; frequent display updates never call this writer."""
    execution = _CURRENT.get()
    path = pathlib.Path(path)
    if execution and path == execution.audit and stage in execution.states:
        context.setdefault("operation_id", execution.stage_ids[stage])
        context.setdefault("plan_stage", True)
    event = _safe({"timestamp": dt.datetime.now(dt.UTC).isoformat(),
                   **_identity(), **context, "stage": stage, "status": status,
                   "elapsed_seconds": elapsed})
    token = _AUDITING.set(True)
    try:
        from mlops import _volume_files
        from mlops._safe_files import append_audit_bytes

        if execution and _volume_files.is_volume(path):
            execution.write_event(path, event)
        else:
            payload = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
            append_audit_bytes(path, payload)
    except BaseException as error:
        # Append may have committed before readback or close failed. Do not invent
        # a second terminal outcome, or claim that the first event was not saved.
        event["audit_error"] = type(error).__name__
        event["attempted_status"] = status
        if status not in {"FAIL", "CANCELLED"}:
            event["status"] = "UNVERIFIED"
        # Preserve the audit error if displaying it is also interrupted.
        with suppress(BaseException):
            if execution:
                execution.observe(path, event)
            _emit(event, _LOG_LEVEL.get() if execution else resolve_level())
        raise
    finally:
        _AUDITING.reset(token)
    # Let the owning scope know START committed before interruptible display work.
    if _recorded is not None:
        _recorded()
    # Record the verified outcome before calling fallible display code.
    if execution:
        execution.observe(path, event)
    _emit(event, _LOG_LEVEL.get() if execution else resolve_level())


def read_execution_events(path):
    """Read local JSONL or remote event segments; this does not certify run completion."""
    from mlops import _volume_files
    path = pathlib.Path(path)
    execution = _CURRENT.get()
    if execution is not None:
        execution.flush_audit(path)
    payload = _volume_files.read_audit_bytes(path)
    events = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("audit events must be JSON objects")
    return events


def _parent():
    execution = _CURRENT.get()
    if _OPERATION.get() is not None:
        return _OPERATION.get(), _PARENT_AUDIT.get()
    if execution:
        running = [name for name, state in execution.states.items() if state == "START"]
        if len(running) == 1:
            return execution.stage_ids[running[0]], str(execution.audit)
    return None, None


@contextmanager
def _scope(path, name, *, planned=False, **facts):
    """One lifecycle owns its context, audit identity, counters and cleanup."""
    execution = _CURRENT.get()
    parent, parent_audit = _parent()
    inherited = {**_IDENTITY.get(), **{k: v for k, v in facts.items() if k in _CORRELATION}}
    key = (execution.stage_ids[name] if planned and execution and path == execution.audit
           and name in execution.stage_ids else uuid.uuid4().hex)
    logical = (name, str(facts.get("dataset", "")), tuple(sorted(inherited.items())))
    start = time.monotonic()
    identity_token = _IDENTITY.set(inherited)
    operation_token = _OPERATION.set(key)
    path_token = _PARENT_AUDIT.set(str(path) if path is not None else None)
    started = False
    terminal_attempted = False
    failed = False
    outcome = {}
    if execution:
        with execution.lock:
            previous = [k for k, v in execution.operations.items()
                        if v["logical"] == logical
                        and v.get("status") in {"FAIL", "CANCELLED", "UNVERIFIED"}]
            for old in previous:
                execution.operations.pop(old)
            attempt = execution.operation_attempts.get(logical, 0) + 1
            execution.operation_attempts[logical] = attempt
            execution.operations[key] = {"name": name, "start": start, "logical": logical,
                                         **_safe(facts)}
    else:
        attempt = 1

    def start_recorded():
        nonlocal started
        started = True

    def event(status, **details):
        metadata = {**facts, "operation_id": key, "parent_operation": parent,
                    "parent_audit": parent_audit, "operation_attempt": attempt,
                    "plan_stage": planned, **details}
        if path is not None:
            _stage_event(path, name, status,
                         _recorded=start_recorded if status == "START" else None, **metadata)
        else:
            if status == "START":
                start_recorded()
            _emit(_safe({**_identity(), "stage": name, "status": status, **metadata}),
                  _LOG_LEVEL.get() if execution else resolve_level())

    try:
        event("START")
        yield outcome
        status = outcome.pop("status", "PASS")
        counts = {}
        if execution:
            with execution.lock:
                op = execution.operations[key]
                counts = {k: op[k] for k in ("completed", "total", "unit", "phase", "files")
                          if k in op}
                if op.get("unit") == "rows":
                    counts["rows"] = op["completed"]
        terminal_attempted = True
        event(status, elapsed=time.monotonic() - start, **counts, **outcome)
        if execution:
            with execution.lock:
                execution.operations.pop(key)
    except BaseException as error:
        failed = True
        if started and not terminal_attempted:
            status = "FAIL" if isinstance(error, Exception) else "CANCELLED"
            elapsed = time.monotonic() - start
            if execution:
                with execution.lock:
                    execution.operations[key].update(status=status, _frozen_elapsed=elapsed)
            with suppress(BaseException):
                event(status, elapsed=elapsed, error_type=type(error).__name__)
        raise
    finally:
        _PARENT_AUDIT.reset(path_token)
        _OPERATION.reset(operation_token)
        _IDENTITY.reset(identity_token)
        if execution:
            with suppress(BaseException) if failed else nullcontext():
                with execution.lock:
                    execution.refresh()


@contextmanager
def _stage(path, name, **context):
    with _scope(pathlib.Path(path), name, planned=True, **context) as outcome:
        yield outcome


def note(name, **facts):
    """Report an operational decision, also for direct lower-level callers."""
    if _AUDITING.get():
        return
    execution = _CURRENT.get()
    parent, parent_audit = _parent()
    if execution:
        _stage_event(execution.audit, name, "INFO", parent_operation=parent,
                     parent_audit=parent_audit, **facts)
    else:
        _emit(_safe({**_identity(), "stage": name, "status": "INFO", **facts}), resolve_level())


@contextmanager
def operation(name, **facts):
    """Describe nested work, including loops with no known denominator."""
    execution = _CURRENT.get()
    if _AUDITING.get():
        yield lambda **counts: None
        return
    with _scope(execution.audit if execution else None, name, **facts):
        yield progress


def progress(*, completed, total=None, unit="items", **facts):
    """Update existing counters without adding reads, audit writes or an invented ETA."""
    execution, key = _CURRENT.get(), _OPERATION.get()
    if execution is not None and key is not None and not _AUDITING.get():
        # The existing integrity check diagnoses changed data. Do not invent a percentage
        # or replace that error when observed bytes exceed the original inventory.
        if total is not None and completed > total:
            total = None
        with execution.lock:
            execution.operations[key].update(completed=completed, total=total,
                                             unit=unit, **_safe(facts))
            execution.refresh()


def reported(name, *, dataset=False):
    """Instrument an existing operation without changing its arguments or result."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            facts = {"dataset": str(args[0] if args else kwargs["path"])} if dataset else {}
            with operation(name, **facts):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def callable_name(callback):
    return (f"{getattr(callback, '__module__', type(callback).__module__)}."
            f"{getattr(callback, '__qualname__', type(callback).__qualname__)}")


def bind_identity(**identity):
    execution = _CURRENT.get()
    if execution:
        with execution.lock:
            execution.identity.update(identity)


def record_result(*, study_id, workflow_id=None, workflow_path=None, evidence=None):
    """Keep the authoritative saved result separate from nested trial event IDs."""
    execution = _CURRENT.get()
    if execution:
        with execution.lock:
            execution.receipt = _safe({key: value for key, value in {
                "study_id": study_id, "workflow_id": workflow_id,
                "workflow_path": workflow_path, "evidence": evidence,
            }.items() if value is not None})


def report_workflow(function=None, *, validate=None):
    """Give direct calls and process workers an attempt after argument validation."""
    if function is None:
        return lambda callback: report_workflow(callback, validate=validate)

    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(study, *args, **kwargs):
        from autoforecast.study_tuning import _output_root

        label = validate_model_label(kwargs.get("model_label"))
        signature.bind(study, *args, **kwargs)
        if validate is not None:
            validate(*args, **kwargs)
        parent = _CURRENT.get()
        root = _output_root(study, kwargs.get("output_dir"), "study-workflows")
        audit = root / "execution-attempts" / (uuid.uuid4().hex + ".jsonl")
        with Execution(audit, ("workflow",), level=parent.level if parent else resolve_level(),
                       study_id=study.study_id,
                       **({"model_label": label} if label is not None else {})):
            with _stage(audit, "workflow") as outcome:
                result = function(study, *args, **kwargs)
                record_result(study_id=result.study_id, workflow_id=result.run_id,
                              workflow_path=str(result.path))
                outcome["workflow_path"] = str(result.path)
                outcome["result_study_id"] = result.study_id
                if result.status == "pending_actuals":
                    outcome["status"] = "UNAVAILABLE"
                return result
    return wrapped


@contextmanager
def standalone_attempt(root, name, *, separate=False, **identity):
    """Start a process-local attempt from inert IDs, or join the caller's attempt."""
    if _CURRENT.get() is not None and not separate:
        with operation(name, **identity):
            yield False
    else:
        audit = pathlib.Path(root) / "execution-attempts" / (uuid.uuid4().hex + ".jsonl")
        with Execution(audit, [name], level=resolve_level(), **identity):
            with _stage(audit, name):
                yield True
