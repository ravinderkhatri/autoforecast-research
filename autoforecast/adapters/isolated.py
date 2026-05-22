"""Process-isolated adapter execution for native runtimes that cannot safely coexist.

The parent owns ranking and scoring.  A worker owns exactly one adapter import and one Spark
session, writes the ordinary forecast Parquet, and returns only the inert ``Forecast`` record.
The JSON request is retained beside the run, making the exact adapter arguments inspectable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from autoforecast import resources
from autoforecast._exact_json import ExactJSONError, loads_exact_json
from autoforecast.acceptance_retirement import canonical_json_sha256
from autoforecast.adapters import _validate_daily_span
from autoforecast.executable_source import validate_executable_source_snapshot
from autoforecast.panel import Forecast, Panel

REUSE_EXEMPTION = (
    "process-boundary adapter; it delegates forecasting to the registered maintained adapter "
    "inside a fresh worker and contains no forecasting algorithm"
)
_WORKER_TAIL_BYTES = 64 * 1024
_ERROR_TAIL_BYTES = 4096
_PROCESS_GROUP_PERMISSION_GRACE_SECONDS = 1.0
_REQUEST_SCHEMA = "autoforecast.isolated-request.v1"
_RESPONSE_SCHEMA = "autoforecast.isolated-response.v1"


def _node_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _private_request_state(
    root: pathlib.Path,
    request_path: pathlib.Path,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Bind the private request directory, inode, metadata, and exact bytes."""
    directory = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        directory_before = os.fstat(directory)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise RuntimeError("isolated private request root is not a directory")
        if os.listdir(directory) != [request_path.name]:
            raise RuntimeError("isolated private request root has an unexpected entry")
        request = os.open(
            request_path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory,
        )
        try:
            request_before = os.fstat(request)
            if (
                not stat.S_ISREG(request_before.st_mode)
                or request_before.st_nlink != 1
            ):
                raise RuntimeError(
                    "isolated private request is not one single-link regular file"
                )
            digest = hashlib.sha256()
            while block := os.read(request, 1024 * 1024):
                digest.update(block)
            request_after = os.fstat(request)
        finally:
            os.close(request)
        directory_after = os.fstat(directory)
    finally:
        os.close(directory)
    try:
        root_path_state = _node_state(root.lstat())
        request_path_state = _node_state(request_path.lstat())
    except OSError as exc:
        raise RuntimeError("isolated private request disappeared") from exc
    if (
        _node_state(directory_before) != _node_state(directory_after)
        or _node_state(directory_after) != root_path_state
        or _node_state(request_before) != _node_state(request_after)
        or _node_state(request_after) != request_path_state
    ):
        raise RuntimeError("isolated private request changed while it was inspected")
    return root_path_state, request_path_state, digest.hexdigest()


def _write_new_file(path: pathlib.Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
    finally:
        os.close(descriptor)


def _read_private_response(path: pathlib.Path) -> tuple[bytes, object]:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(
                "isolated adapter worker response is not one single-link regular file"
            )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_state = _node_state(path.lstat())
    except OSError as exc:
        raise RuntimeError("isolated adapter worker response disappeared") from exc
    if _node_state(before) != _node_state(after) or _node_state(after) != path_state:
        raise RuntimeError("isolated adapter worker response changed while it was read")
    payload = b"".join(chunks)
    try:
        return payload, loads_exact_json(payload)
    except ExactJSONError as exc:
        raise RuntimeError(
            f"isolated adapter worker returned unreadable or ambiguous JSON: {exc}"
        ) from exc


def _append_tail(tail: bytearray, block: bytes) -> None:
    tail.extend(block)
    if len(tail) > _WORKER_TAIL_BYTES:
        del tail[:-_WORKER_TAIL_BYTES]


def _signal_worker_group(process: subprocess.Popen[bytes], sig: int) -> bool:
    """Signal a worker group, tolerating only the leader-exit EPERM race."""
    process_group = process.pid
    try:
        os.killpg(process_group, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as permission_error:
        if process.poll() is None:
            try:
                process.wait(timeout=_PROCESS_GROUP_PERMISSION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                raise permission_error from None
        retry_deadline = time.monotonic() + _PROCESS_GROUP_PERMISSION_GRACE_SECONDS
        while True:
            try:
                os.killpg(process_group, sig)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                remaining = retry_deadline - time.monotonic()
                if remaining <= 0:
                    raise permission_error from None
                time.sleep(min(0.05, remaining))


def _terminate_worker(process: subprocess.Popen[bytes]) -> None:
    if not _signal_worker_group(process, signal.SIGTERM):
        process.wait()
        return
    deadline = time.monotonic() + 10
    while True:
        if not _signal_worker_group(process, 0):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _signal_worker_group(process, signal.SIGKILL)
            break
        time.sleep(min(0.05, remaining))
    process.wait()


def _run_monitored_worker(
    argv: list[str],
    *,
    cwd: pathlib.Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    heartbeat_seconds: int,
    worker_log_path: pathlib.Path,
    heartbeat_sink: Callable[[str], None] | None,
) -> subprocess.CompletedProcess[str]:
    """Continuously drain a worker, retain its full log, and enforce registered controls."""
    if (
        type(timeout_seconds) is not int
        or type(heartbeat_seconds) is not int
        or timeout_seconds < 2
        or heartbeat_seconds < 1
        or heartbeat_seconds >= timeout_seconds
    ):
        raise ValueError(
            "isolated worker timeout_seconds/heartbeat_seconds must be integers with "
            "timeout >= 2 and 1 <= heartbeat < timeout"
        )
    if not worker_log_path.is_absolute():
        raise ValueError("isolated worker_log_path must be absolute")
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    emit = heartbeat_sink or (lambda message: print(message, file=sys.stderr, flush=True))
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_tail = bytearray()
    stderr_tail = bytearray()
    log_lock = threading.Lock()
    relay_lock = threading.Lock()
    relay_live = env.get("AUTOFORECAST_LIVE_PROGRESS") == "1"
    with worker_log_path.open("wb") as worker_log:

        def drain(stream, tail: bytearray, label: bytes) -> None:
            for block in iter(lambda: stream.read1(64 * 1024), b""):
                _append_tail(tail, block)
                with log_lock:
                    worker_log.write(label)
                    worker_log.write(block)
                    worker_log.flush()
                if relay_live:
                    with relay_lock:
                        output = getattr(sys.stderr, "buffer", None)
                        if output is not None:
                            output.write(block)
                            output.flush()
                        else:
                            sys.stderr.write(block.decode("utf-8", errors="replace"))
                            sys.stderr.flush()

        stdout_thread = threading.Thread(
            target=drain,
            args=(process.stdout, stdout_tail, b"[stdout] "),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=drain,
            args=(process.stderr, stderr_tail, b"[stderr] "),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        started = time.monotonic()
        deadline = started + timeout_seconds
        next_heartbeat = started + heartbeat_seconds
        timed_out = False
        try:
            while process.poll() is None:
                now = time.monotonic()
                wait_seconds = max(0.01, min(deadline, next_heartbeat) - now)
                try:
                    process.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    if now >= deadline:
                        timed_out = True
                        _terminate_worker(process)
                        break
                    if now >= next_heartbeat:
                        # A relayed tqdm bar owns the interactive terminal line.  The outer
                        # campaign already writes a durable heartbeat, so duplicating this
                        # message on stderr only splices text into the live bar.
                        if not relay_live or heartbeat_sink is not None:
                            emit(
                                "isolated adapter worker still running; "
                                f"pid={process.pid}, elapsed_seconds={int(now - started)}"
                            )
                        next_heartbeat += heartbeat_seconds
            _terminate_worker(process)
        except BaseException:
            _terminate_worker(process)
            raise
        finally:
            stdout_thread.join(timeout=30)
            stderr_thread.join(timeout=30)
            if stdout_thread.is_alive() or stderr_thread.is_alive():
                _terminate_worker(process)
                raise RuntimeError("isolated worker output drain did not terminate")

    stdout = bytes(stdout_tail).decode("utf-8", errors="replace")
    stderr = bytes(stderr_tail).decode("utf-8", errors="replace")
    if timed_out:
        raise RuntimeError(
            f"isolated adapter worker exceeded timeout_seconds={timeout_seconds}; "
            f"stderr_tail={stderr[-_ERROR_TAIL_BYTES:]}"
        )
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _jsonable(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError(
                "isolated adapter mappings require exact string keys; coercing keys could "
                "collapse distinct configuration fields")
        return {key: _jsonable(v) for key, v in value.items()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        f"isolated adapter argument {value!r} ({type(value).__name__}) is not JSON-serialisable. "
        "Only inert configuration may cross the process boundary")


def run(adapter: str, panel: Panel, origin: str, horizon: int, *,
        source_snapshot: Mapping[str, object], worker_profile: resources.SparkProfile,
        timeout_seconds: int, heartbeat_seconds: int,
        worker_log_path: str | pathlib.Path,
        tracking_context: Mapping[str, str] | None = None,
        worker_python: str | pathlib.Path | None = None,
        heartbeat_sink: Callable[[str], None] | None = None,
        **kwargs: object) -> Forecast:
    """Run ``adapter`` in a fresh Python process and return its validated record."""
    if "worker_runtime_identity" in kwargs:
        raise ValueError(
            "worker_runtime_identity is unsupported; select the interpreter with worker_python"
        )
    _validate_daily_span(panel, origin, horizon, adapter=adapter)
    source = validate_executable_source_snapshot(source_snapshot)
    if not isinstance(worker_profile, resources.SparkProfile):
        raise TypeError(
            "isolated adapter execution requires an explicit worker SparkProfile. The scoring "
            "parent is alive concurrently, so deriving both from the whole machine overcommits it")
    # The worker creates its own session.  A live PySpark object cannot be serialised and, more
    # importantly, would couple the supposedly isolated native process back to the parent's JVM.
    kwargs.pop("spark", None)
    out_dir = kwargs.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        raise ValueError("an isolated adapter requires a non-empty string out_dir")
    target = pathlib.Path(out_dir)
    runtime = pathlib.Path(sys.executable if worker_python is None else worker_python)
    if worker_python is not None and not runtime.is_absolute():
        raise ValueError(
            "isolated worker_python must be an absolute path to a locked environment; PATH "
            f"discovery is not reproducible, got {worker_python!r}")
    runtime = runtime.absolute()
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise ValueError(
            f"isolated worker_python is not an executable file: {runtime}")
    runtime_prefix = runtime.parent.parent.resolve()
    runtime_record: dict[str, object] = {
        "python": str(runtime),
        "prefix": str(runtime_prefix),
    }
    request_path = target.with_suffix(".request.json")
    response_path = target.with_suffix(".response.json")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "adapter": adapter,
        "panel": _jsonable(dataclasses.asdict(panel)),
        "origin": origin,
        "horizon": horizon,
        "kwargs": _jsonable(kwargs),
        "worker_profile": dataclasses.asdict(worker_profile),
        "worker_runtime": runtime_record,
        "source_snapshot": _jsonable(source),
    }
    tracking_uri: str | None = None
    if tracking_context is not None:
        required = {"tracking_uri", "run_id", "experiment_id"}
        if set(tracking_context) != required or not all(
            isinstance(value, str) and value for value in tracking_context.values()
        ):
            raise ValueError("tracking_context must contain non-blank MLflow identity strings")
        tracking_uri = tracking_context["tracking_uri"]
        request["tracking_context"] = {
            "tracking_uri_sha256": hashlib.sha256(tracking_uri.encode()).hexdigest(),
            "run_id": tracking_context["run_id"],
            "experiment_id": tracking_context["experiment_id"],
        }
    try:
        request_sha256 = canonical_json_sha256(request)
        request_envelope = {
            "schema": _REQUEST_SCHEMA,
            "request_sha256": request_sha256,
            "request": request,
        }
        encoded_request = json.dumps(
            request_envelope, indent=2, sort_keys=True, allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"isolated adapter {adapter!r} request is not strict JSON: {exc}") from exc
    _write_new_file(request_path, encoded_request, mode=0o600)
    source_root = source["root"]
    if not isinstance(source_root, str):  # validated by executable-source contract
        raise ValueError("executable source snapshot root is invalid")
    root = pathlib.Path(source_root)
    log_path = pathlib.Path(worker_log_path)
    worker_env = dict(os.environ)
    worker_env["PYSPARK_PYTHON"] = str(runtime)
    worker_env["PYTHONPATH"] = str(root)
    worker_env.pop("AUTOFORECAST_WORKER_MLFLOW_TRACKING_URI", None)
    if tracking_uri is not None:
        # The retained request binds the endpoint hash; the actual credential-bearing URI is a
        # transient child-process secret and must never enter the immutable run bundle.
        worker_env["AUTOFORECAST_WORKER_MLFLOW_TRACKING_URI"] = tracking_uri
    with tempfile.TemporaryDirectory(prefix="autoforecast-isolated-request-") as request_root, \
            tempfile.TemporaryDirectory(prefix="autoforecast-isolated-response-") as response_root:
        private_request_root = pathlib.Path(request_root)
        private_request_path = private_request_root / "request.json"
        private_response_path = pathlib.Path(response_root) / "response.json"
        _write_new_file(private_request_path, encoded_request, mode=0o400)
        pinned_request_state = _private_request_state(
            private_request_root, private_request_path
        )
        try:
            argv = [
                str(runtime),
                "-m",
                "autoforecast.adapters.worker",
                str(private_request_path),
                str(private_response_path),
            ]
            completed = _run_monitored_worker(
                argv,
                cwd=root,
                env=worker_env,
                timeout_seconds=timeout_seconds,
                heartbeat_seconds=heartbeat_seconds,
                worker_log_path=log_path,
                heartbeat_sink=heartbeat_sink,
            )
        finally:
            try:
                validate_executable_source_snapshot(source)
            finally:
                if _private_request_state(
                    private_request_root, private_request_path
                ) != pinned_request_state:
                    raise RuntimeError(
                        "isolated private request changed while the worker consumed it"
                    )
        if (
            adapter in {"tft_q", "tft_hnb", }
            and not private_response_path.exists()
        ):
            checkpoint_value = kwargs.get("checkpoint_out")
            version = {"tft_q": "v6", "tft_hnb": "v6", }[
                adapter
            ]
            checkpoint = (
                target.with_name(f"{target.name}-{version}-artifacts") / "model.pt"
                if checkpoint_value is None else pathlib.Path(checkpoint_value)
                if isinstance(checkpoint_value, str | pathlib.Path) else None
            )
            if (
                checkpoint is not None
                and target.is_absolute() and checkpoint.is_absolute()
                and not target.is_symlink() and target.is_dir()
                and not checkpoint.is_symlink() and checkpoint.is_file()
                and not (target / "part-00000.parquet").is_symlink()
                and (target / "part-00000.parquet").is_file()
            ):
                completed = _run_monitored_worker(
                    argv, cwd=root, env=worker_env,
                    timeout_seconds=timeout_seconds, heartbeat_seconds=heartbeat_seconds,
                    worker_log_path=log_path.with_name(log_path.name + ".recovery"),
                    heartbeat_sink=heartbeat_sink,
                )
                validate_executable_source_snapshot(source)
                if (
                    _private_request_state(private_request_root, private_request_path)
                    != pinned_request_state
                ):
                    raise RuntimeError("isolated private request changed during artifact recovery")
        if completed.returncode != 0:
            raise RuntimeError(
                f"isolated adapter {adapter!r} failed with exit code {completed.returncode}; "
                f"stderr_tail={completed.stderr[-_ERROR_TAIL_BYTES:]}; "
                f"request retained at {request_path}; log retained at {log_path}")
        if not private_response_path.exists():
            raise RuntimeError(
                f"isolated adapter {adapter!r} returned no forecast response"
            )
        response_bytes, response = _read_private_response(private_response_path)
        if (
            type(response) is not dict
            or set(response) != {"schema", "request_sha256", "forecast"}
            or response.get("schema") != _RESPONSE_SCHEMA
            or response.get("request_sha256") != request_sha256
            or type(response.get("forecast")) is not dict
        ):
            raise RuntimeError(
                f"isolated adapter {adapter!r} returned no exact request-bound forecast"
            )
        forecast = Forecast.from_record(response["forecast"]).validate()
        _write_new_file(response_path, response_bytes, mode=0o600)
        return forecast


__all__ = ["run"]
