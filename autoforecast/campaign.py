"""Durable local orchestration for dependency-ordered research stages.

The campaign layer owns process order and recovery only. Forecasting semantics stay in the
existing governed runners. Commands are executed as argument vectors without a shell; their exact
plan bytes, working directory, attempts, outputs and terminal records are hash-bound.

This is deliberately executor-neutral. A later Airflow adapter can invoke the same stage worker
requests and consume the same journal without moving protocol decisions into a DAG definition.
"""

from __future__ import annotations

import codecs
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Literal, cast

from autoforecast._exact_json import ExactJSONError, load_exact_json
from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast._numeric import is_finite_real
from autoforecast.notifications import campaign_notification_from_environment
from autoforecast.protocol import (
    _atomic_write_text,
    _ensure_directory_durable,
    _fsync_directory,
)
from autoforecast.training_lease import HOST_TRAINING_LEASE_FD, open_host_training_lease

__all__ = [
    "CampaignError",
    "CampaignPlan",
    "CampaignStore",
    "CampaignSupervisor",
    "StageResources",
    "StageSpec",
    "load_campaign_plan",
    "run_stage_worker",
]

PLAN_SCHEMA = "autoforecast.campaign-plan.v1"
EVENT_SCHEMA = "autoforecast.campaign-event.v1"
STATE_SCHEMA = "autoforecast.campaign-state.v1"
WORKER_REQUEST_SCHEMA = "autoforecast.campaign-worker-request.v2"
WORKER_IDENTITY_SCHEMA = "autoforecast.campaign-worker-identity.v2"
CHILD_IDENTITY_SCHEMA = "autoforecast.campaign-child-identity.v2"
WORKER_HEARTBEAT_SCHEMA = "autoforecast.campaign-worker-heartbeat.v1"
WORKER_RESULT_SCHEMA = "autoforecast.campaign-worker-result.v2"
NOTIFICATION_INTENT_SCHEMA = "autoforecast.campaign-terminal-notification-intent.v1"

_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALIDATION_STAGE_ID = re.compile(r"^v(\d+)([fws])s(\d+)o(\d+)$")
_PARENT_STAGE_ID = re.compile(r"^p(\d+)w(\d+)$")
_TUNING_STAGE_ID = re.compile(r"^t(\d+)$")
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PLACEHOLDERS = {
    "{campaign_root}",
    "{attempt_dir}",
    "{stage_id}",
    "{attempt}",
    "{previous_attempt_dir}",
}
_CONTROL_PATHS = {
    ".child.lock",
    ".worker.lock",
    "child-identity.json",
    "heartbeat.json",
    "stage-result.json",
    "stderr.log",
    "stdout.log",
    "worker-identity.json",
    "worker-request.json",
    "worker-token",
    "worker-wrapper.log",
}

StageState = Literal["planned", "running", "completed", "failed", "blocked"]


def _stage_title(stage_id: str) -> str:
    """Render frozen stage IDs as concise user-facing names without changing plan bytes."""
    fixed = {
        "aggregate_selection": "Select the best validated candidate",
        "final_fit": "Fresh fit on training + validation data",
        "final_score": "Evaluate the frozen model on test data",
        "replay_report": "Verify artifacts and publish the final report",
    }
    if stage_id in fixed:
        return fixed[stage_id]
    validation = _VALIDATION_STAGE_ID.fullmatch(stage_id)
    if validation:
        model, arm, seed, origin = validation.groups()
        initialization = {
            "f": "fresh initialization",
            "w": "weights-only fine-tuning",
            "s": "full-state continuation",
        }[arm]
        return (
            f"Candidate {int(model) + 1} validation "
            f"({initialization}, seed {int(seed) + 1}, origin {int(origin) + 1})"
        )
    parent = _PARENT_STAGE_ID.fullmatch(stage_id)
    if parent:
        model, seed = parent.groups()
        return f"Build warm-start parent (candidate {int(model) + 1}, seed {int(seed) + 1})"
    tuning = _TUNING_STAGE_ID.fullmatch(stage_id)
    if tuning:
        return f"Tune candidate {int(tuning.group(1)) + 1} parameters"
    return stage_id.replace("_", " ").capitalize()


def _stage_banner(message: str) -> str:
    """Center one main workflow event between visible hash borders."""
    body = f" {message.strip()} "
    columns = shutil.get_terminal_size(fallback=(120, 24)).columns
    width = max(len(body) + 12, min(max(columns, 80), 160))
    return body.center(width, "#")


def _workflow_stage(
    stages: tuple[StageSpec, ...], stage: StageSpec
) -> tuple[int, int, str, bool] | None:
    """Group a standard one-model benchmark's durable checkpoints into three user stages."""
    ids = tuple(item.id for item in stages)
    if (
        len(ids) != 5
        or _VALIDATION_STAGE_ID.fullmatch(ids[0]) is None
        or ids[1:] != (
            "aggregate_selection",
            "final_fit",
            "final_score",
            "replay_report",
        )
    ):
        return None
    presentations = {
        ids[0]: (1, 3, "Train on training data and evaluate on validation data", False),
        "aggregate_selection": (
            1,
            3,
            "Freeze the validation result and model configuration",
            True,
        ),
        "final_fit": (2, 3, "Fresh fit on training + validation data", False),
        "final_score": (3, 3, "Evaluate once on test data", False),
        "replay_report": (3, 3, "Verify and seal the test report", True),
    }
    return presentations[stage.id]
CampaignState = Literal["planned", "running", "completed", "failed"]


class CampaignError(RuntimeError):
    """A campaign plan, durable record, lease, or worker result is unsafe."""


def _stderr_isatty() -> bool:
    """Return whether stderr is interactive, including daemonized processes safely."""
    return sys.stderr is not None and sys.stderr.isatty()


def _read_tail(path: pathlib.Path, size: int) -> str:
    """Read only the last ``size`` bytes of a potentially large text log."""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - size))
        return handle.read(size).decode("utf-8", errors="replace")


def _json(value: object) -> str:
    try:
        return json.dumps(value, allow_nan=False, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"campaign record is not finite JSON: {exc}") from exc


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_lock(path: pathlib.Path):
    """Open one stable regular lock inode without following a caller-supplied link."""
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CampaignError(f"campaign lock path is unsafe: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise CampaignError(f"campaign lock path is not one stable regular file: {path}")
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        return os.fdopen(descriptor, "a+b")
    except BaseException:
        os.close(descriptor)
        raise


def _lock_is_held(path: pathlib.Path) -> bool:
    """Report whether another process owns a safe flock without trusting a PID."""
    with _open_lock(path) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def _lease_holder_pids(lock_path: pathlib.Path) -> set[int]:
    """Return kernel-observed processes that currently have one held lease open."""
    if not _lock_is_held(lock_path):
        return set()
    lsof = shutil.which("lsof")
    if lsof is None and pathlib.Path("/usr/sbin/lsof").is_file():
        lsof = "/usr/sbin/lsof"
    if lsof is None:
        raise CampaignError(
            "cannot authenticate a campaign lease holder because lsof is unavailable"
        )
    observed = subprocess.run(
        [lsof, "-t", "--", str(lock_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if observed.returncode not in {0, 1}:
        raise CampaignError("cannot identify the process holding a campaign lease")
    holders: set[int] = set()
    for raw in observed.stdout.splitlines():
        with contextlib.suppress(ValueError):
            pid = int(raw.strip())
            if pid > 1:
                holders.add(pid)
    if not holders and _lock_is_held(lock_path):
        raise CampaignError("campaign lease is held without an identifiable owner")
    return holders


def _require_lease_holder(lock_path: pathlib.Path, pid: int, label: str) -> None:
    if pid not in _lease_holder_pids(lock_path):
        raise CampaignError(f"{label} identity does not own its campaign lease")


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{label} must be a non-empty relative path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or ".." in path.parts:
        raise CampaignError(f"{label} must be a relative path without '..'")
    return path.as_posix()


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise CampaignError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str, *, positive: bool) -> float:
    if not is_finite_real(value):
        raise CampaignError(f"{label} must be a finite number")
    number = float(value)
    if (positive and number <= 0) or (not positive and number < 0):
        comparison = "positive" if positive else "non-negative"
        raise CampaignError(f"{label} must be {comparison}")
    return number


@dataclass(frozen=True)
class StageResources:
    """Admission declaration for one sequential stage."""

    device: Literal["cpu", "cuda", "mps"]
    memory_gib: float
    min_free_disk_gib: float

    def as_record(self) -> dict[str, object]:
        return {
            "device": self.device,
            "memory_gib": self.memory_gib,
            "min_free_disk_gib": self.min_free_disk_gib,
        }


@dataclass(frozen=True)
class StageSpec:
    """One immutable process and the stages that must complete before it."""

    id: str
    needs: tuple[str, ...]
    command: tuple[str, ...]
    timeout_seconds: int
    max_attempts: int
    retry_mode: Literal["restart", "command_managed_resume"]
    required_outputs: tuple[str, ...]
    resources: StageResources

    def as_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "needs": list(self.needs),
            "command": list(self.command),
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "retry_mode": self.retry_mode,
            "required_outputs": list(self.required_outputs),
            "resources": self.resources.as_record(),
        }


@dataclass(frozen=True)
class CampaignPlan:
    """Strict source plan plus the SHA-256 of its exact bytes."""

    name: str
    stages: tuple[StageSpec, ...]
    source_sha256: str
    source_bytes: bytes

    def as_record(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "name": self.name,
            "source_sha256": self.source_sha256,
            "stages": [stage.as_record() for stage in self.stages],
        }


def _load_yaml(payload: bytes, source: pathlib.Path) -> object:
    try:
        return loads_exact_yaml(payload)
    except ExactYAMLError as exc:
        raise CampaignError(f"invalid campaign YAML in {source}: {exc}") from exc


def _parse_resources(value: object, stage_id: str) -> StageResources:
    expected = {"device", "memory_gib", "min_free_disk_gib"}
    if type(value) is not dict or set(value) != expected:
        raise CampaignError(f"stage {stage_id!r} resources require exactly {sorted(expected)}")
    device = value["device"]
    if type(device) is not str or device not in {"cpu", "cuda", "mps"}:
        raise CampaignError(f"stage {stage_id!r} resource device is unsupported")
    return StageResources(
        cast(Literal["cpu", "cuda", "mps"], device),
        _finite_number(value["memory_gib"], f"stage {stage_id!r} memory_gib", positive=True),
        _finite_number(
            value["min_free_disk_gib"],
            f"stage {stage_id!r} min_free_disk_gib",
            positive=False,
        ),
    )


def _parse_stage(value: object) -> StageSpec:
    expected = {
        "id",
        "needs",
        "command",
        "timeout_seconds",
        "max_attempts",
        "retry_mode",
        "required_outputs",
        "resources",
    }
    if type(value) is not dict or set(value) != expected:
        raise CampaignError(f"each campaign stage requires exactly {sorted(expected)}")
    stage_id = value["id"]
    if not isinstance(stage_id, str) or _ID.fullmatch(stage_id) is None:
        raise CampaignError(f"stage id must match {_ID.pattern}")
    needs = value["needs"]
    if type(needs) is not list or not all(isinstance(item, str) for item in needs):
        raise CampaignError(f"stage {stage_id!r} needs must be a list of stage IDs")
    if len(set(needs)) != len(needs) or stage_id in needs:
        raise CampaignError(f"stage {stage_id!r} needs must be unique and cannot include itself")
    command = value["command"]
    if (
        type(command) is not list
        or not command
        or not all(isinstance(item, str) and item and "\x00" not in item for item in command)
    ):
        raise CampaignError(f"stage {stage_id!r} command must be non-empty string arguments")
    outputs = value["required_outputs"]
    if type(outputs) is not list:
        raise CampaignError(f"stage {stage_id!r} required_outputs must be a list")
    required_outputs = tuple(
        _safe_relative(item, f"stage {stage_id!r} required output") for item in outputs
    )
    folded_outputs = tuple(output.casefold() for output in required_outputs)
    if len(set(folded_outputs)) != len(folded_outputs):
        raise CampaignError(
            f"stage {stage_id!r} required_outputs must be unique, including by case"
        )
    reserved = sorted(
        output
        for output in required_outputs
        if pathlib.PurePosixPath(output).parts[0].casefold() in _CONTROL_PATHS
    )
    if reserved:
        raise CampaignError(
            f"stage {stage_id!r} required_outputs use wrapper-owned control paths: {reserved}"
        )
    retry_mode = value["retry_mode"]
    if type(retry_mode) is not str or retry_mode not in {
        "restart",
        "command_managed_resume",
    }:
        raise CampaignError(
            f"stage {stage_id!r} retry_mode must be restart or command_managed_resume"
        )
    if retry_mode == "command_managed_resume" and not any(
        "{previous_attempt_dir}" in argument for argument in command
    ):
        raise CampaignError(
            f"stage {stage_id!r} command_managed_resume must receive "
            "{previous_attempt_dir} explicitly"
        )
    if retry_mode == "restart" and any(
        "{previous_attempt_dir}" in argument for argument in command
    ):
        raise CampaignError(f"stage {stage_id!r} restart may not receive {{previous_attempt_dir}}")
    return StageSpec(
        stage_id,
        tuple(needs),
        tuple(command),
        _positive_int(value["timeout_seconds"], f"stage {stage_id!r} timeout", minimum=2),
        _positive_int(value["max_attempts"], f"stage {stage_id!r} max_attempts"),
        cast(Literal["restart", "command_managed_resume"], retry_mode),
        required_outputs,
        _parse_resources(value["resources"], stage_id),
    )


def _validate_graph(stages: tuple[StageSpec, ...]) -> None:
    by_id = {stage.id: stage for stage in stages}
    if len(by_id) != len(stages):
        raise CampaignError("campaign stage IDs must be unique")
    for stage in stages:
        missing = sorted(set(stage.needs) - set(by_id))
        if missing:
            raise CampaignError(f"stage {stage.id!r} has unknown dependencies: {missing}")
    completed: set[str] = set()
    while len(completed) < len(stages):
        ready = [
            stage.id
            for stage in stages
            if stage.id not in completed and set(stage.needs) <= completed
        ]
        if not ready:
            unresolved = [stage.id for stage in stages if stage.id not in completed]
            raise CampaignError(f"campaign dependency graph contains a cycle: {unresolved}")
        completed.update(ready)


def load_campaign_plan(path: str | pathlib.Path) -> CampaignPlan:
    """Load a strict, finite, acyclic campaign plan from exact source bytes."""
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise CampaignError(f"campaign plan must be a regular non-symlink file: {source}")
    try:
        payload = source.read_bytes()
        payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CampaignError(f"campaign plan is not readable UTF-8: {source}") from exc
    value = _load_yaml(payload, source)
    if type(value) is not dict or set(value) != {"schema_version", "campaign"}:
        raise CampaignError("campaign plan requires exactly schema_version and campaign")
    campaign = value["campaign"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(campaign) is not dict
        or set(campaign)
        != {
            "name",
            "stages",
        }
    ):
        raise CampaignError("campaign schema_version 1 requires exactly name and stages")
    name = campaign["name"]
    if not isinstance(name, str) or _ID.fullmatch(name) is None:
        raise CampaignError(f"campaign name must match {_ID.pattern}")
    raw_stages = campaign["stages"]
    if type(raw_stages) is not list or not raw_stages:
        raise CampaignError("campaign stages must be a non-empty list")
    stages = tuple(_parse_stage(item) for item in raw_stages)
    _validate_graph(stages)
    return CampaignPlan(name, stages, _digest_bytes(payload), payload)


def _event_path(root: pathlib.Path, sequence: int) -> pathlib.Path:
    return root / "events" / f"{sequence:08d}.json"


def _strict_json(path: pathlib.Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = load_exact_json(path)
    except ExactJSONError as exc:
        raise CampaignError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise CampaignError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _write_new_json(path: pathlib.Path, value: object) -> str:
    if path.exists() or path.is_symlink():
        raise CampaignError(f"immutable campaign record already exists: {path}")
    payload = (_json(value) + "\n").encode("utf-8")
    _atomic_write_text(path, payload.decode("utf-8"))
    return _digest_bytes(payload)


def _write_new_worker_token(path: pathlib.Path, token: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            payload = (token + "\n").encode("ascii")
            while payload:
                written = os.write(descriptor, payload)
                payload = payload[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
    except OSError as error:
        raise CampaignError(f"could not publish worker launch token: {path}") from error


def _load_worker_token(path: pathlib.Path, *, consume: bool) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            named = os.stat(path, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_mode & 0o077
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise CampaignError("worker launch token is not one private regular file")
            payload = os.read(descriptor, 128)
            if os.read(descriptor, 1):
                raise CampaignError("worker launch token is invalid")
            token = payload.decode("ascii").rstrip("\n")
            if len(token) != 64 or set(token) - set("0123456789abcdef"):
                raise CampaignError("worker launch token is invalid")
            if consume:
                os.unlink(path)
                _fsync_directory(path.parent)
            return token
        finally:
            os.close(descriptor)
    except CampaignError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise CampaignError(f"cannot read worker launch token: {path}") from error


def _publish_worker_result(path: pathlib.Path, value: object) -> str:
    """Replace only a child-created terminal claim, then publish the wrapper's result."""
    if path.exists() or path.is_symlink():
        try:
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                raise CampaignError("child replaced the worker result path with a directory")
            path.unlink()
            _fsync_directory(path.parent)
        except CampaignError:
            raise
        except OSError as error:
            raise CampaignError("could not remove a child-created worker result") from error
    return _write_new_json(path, value)


def _initialize_root(root: pathlib.Path, plan: CampaignPlan) -> None:
    """Publish a complete initialized directory with one same-filesystem rename."""
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{root.name}.initializing-", dir=root.parent))
    try:
        _ensure_directory_durable(staging / "events")
        _ensure_directory_durable(staging / "stages")
        _write_new_json(staging / "plan.json", plan.as_record())
        _atomic_write_text(staging / "plan-source.yaml", plan.source_bytes.decode("utf-8"))
        _fsync_directory(staging)
        try:
            os.rename(staging, root)
        except OSError:
            if not root.is_dir():
                raise
        else:
            _fsync_directory(root.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _event_extra_fields(kind: str) -> set[str]:
    fields = {
        "campaign_created": {"stage_ids", "working_directory"},
        "stage_launching": {"stage_id", "attempt", "worker_request_sha256"},
        "stage_completed": {"stage_id", "attempt", "worker_result_sha256"},
        "stage_failed": {"stage_id", "attempt", "worker_result_sha256", "reason"},
        "campaign_completed": set(),
        "campaign_failed": {"failed_stage_ids"},
    }
    try:
        return fields[kind]
    except KeyError:
        raise CampaignError(f"unsupported campaign event kind: {kind!r}") from None


def _validate_event(
    event: dict[str, object],
    *,
    plan: CampaignPlan,
    sequence: int,
    previous_sha256: str | None,
) -> None:
    kind = event.get("kind")
    if not isinstance(kind, str):
        raise CampaignError("campaign event kind must be a string")
    expected = {
        "schema",
        "sequence",
        "kind",
        "plan_sha256",
        "previous_sha256",
        "time_ns",
    } | _event_extra_fields(kind)
    if set(event) != expected:
        raise CampaignError(f"campaign event {sequence} fields differ for kind {kind!r}")
    if (
        event["schema"] != EVENT_SCHEMA
        or type(event["sequence"]) is not int
        or event["sequence"] != sequence
        or event["plan_sha256"] != plan.source_sha256
        or event["previous_sha256"] != previous_sha256
        or type(event["time_ns"]) is not int
        or event["time_ns"] < 1
    ):
        raise CampaignError(f"campaign event {sequence} has invalid identity or chain")
    if kind == "campaign_created":
        if (
            event["stage_ids"] != [stage.id for stage in plan.stages]
            or not isinstance(event["working_directory"], str)
            or not event["working_directory"]
        ):
            raise CampaignError("campaign_created has invalid stage IDs or working directory")
    elif kind == "stage_launching":
        request_sha = event["worker_request_sha256"]
        if (
            type(event["stage_id"]) is not str
            or event["stage_id"] not in {stage.id for stage in plan.stages}
            or type(event["attempt"]) is not int
            or event["attempt"] < 1
            or not isinstance(request_sha, str)
            or _SHA256.fullmatch(request_sha) is None
        ):
            raise CampaignError("stage_launching has invalid stage, attempt, or request digest")
    elif kind in {"stage_completed", "stage_failed"}:
        result_sha = event["worker_result_sha256"]
        if (
            type(event["stage_id"]) is not str
            or event["stage_id"] not in {stage.id for stage in plan.stages}
            or type(event["attempt"]) is not int
            or event["attempt"] < 1
            or result_sha is not None
            and (not isinstance(result_sha, str) or _SHA256.fullmatch(result_sha) is None)
        ):
            raise CampaignError(f"{kind} has invalid stage, attempt, or result digest")
        if kind == "stage_failed" and (not isinstance(event["reason"], str) or not event["reason"]):
            raise CampaignError("stage_failed requires a non-empty reason")
    elif kind == "campaign_failed" and (
        type(event["failed_stage_ids"]) is not list
        or not all(isinstance(item, str) for item in event["failed_stage_ids"])
    ):
        raise CampaignError("campaign_failed stage IDs must be a list of strings")


def _project(plan: CampaignPlan, events: tuple[dict[str, object], ...]) -> dict[str, object]:
    if not events or events[0].get("kind") != "campaign_created":
        raise CampaignError("campaign journal must begin with campaign_created")
    stages: dict[str, dict[str, object]] = {
        stage.id: {
            "id": stage.id,
            "status": "planned",
            "attempts": 0,
            "active_attempt": None,
            "last_result_sha256": None,
        }
        for stage in plan.stages
    }
    first = events[0]
    if first.get("stage_ids") != [stage.id for stage in plan.stages]:
        raise CampaignError("campaign_created stage order differs from the frozen plan")
    terminal: CampaignState | None = None
    for event in events[1:]:
        kind = cast(str, event["kind"])
        if terminal is not None:
            raise CampaignError("campaign journal contains events after a terminal event")
        if kind == "stage_launching":
            stage_id = event["stage_id"]
            if stage_id not in stages:
                raise CampaignError(f"journal references unknown stage {stage_id!r}")
            state = stages[cast(str, stage_id)]
            attempt = event["attempt"]
            if (
                state["status"] != "planned"
                or type(attempt) is not int
                or attempt != cast(int, state["attempts"]) + 1
            ):
                raise CampaignError(f"stage {stage_id!r} launch is not the next valid attempt")
            spec = next(item for item in plan.stages if item.id == stage_id)
            if any(stages[dependency]["status"] != "completed" for dependency in spec.needs):
                raise CampaignError(f"stage {stage_id!r} launched before its dependencies")
            state.update(status="running", attempts=attempt, active_attempt=attempt)
        elif kind in {"stage_completed", "stage_failed"}:
            stage_id = event["stage_id"]
            if stage_id not in stages:
                raise CampaignError(f"journal references unknown stage {stage_id!r}")
            state = stages[cast(str, stage_id)]
            if state["status"] != "running" or event["attempt"] != state["active_attempt"]:
                raise CampaignError(f"stage {stage_id!r} terminal event has no matching launch")
            result_sha = event["worker_result_sha256"]
            if result_sha is not None and (
                not isinstance(result_sha, str) or _SHA256.fullmatch(result_sha) is None
            ):
                raise CampaignError(f"stage {stage_id!r} result digest is invalid")
            spec = next(item for item in plan.stages if item.id == stage_id)
            status: StageState = "completed"
            if kind == "stage_failed":
                non_retryable = str(event.get("reason", "")).startswith("non-retryable:")
                status = (
                    "planned"
                    if cast(int, state["attempts"]) < spec.max_attempts and not non_retryable
                    else "failed"
                )
            state.update(
                status=status,
                active_attempt=None,
                last_result_sha256=result_sha,
            )
        elif kind == "campaign_completed":
            if any(item["status"] != "completed" for item in stages.values()):
                raise CampaignError("campaign completed before every stage completed")
            terminal = "completed"
        elif kind == "campaign_failed":
            actual = [stage.id for stage in plan.stages if stages[stage.id]["status"] == "failed"]
            if event["failed_stage_ids"] != actual or not actual:
                raise CampaignError("campaign_failed does not name the exhausted failed stages")
            terminal = "failed"
        else:
            raise CampaignError(f"campaign event {kind!r} is invalid after creation")
    if terminal is None:
        if any(item["status"] == "failed" for item in stages.values()):
            campaign_status: CampaignState = "failed"
        elif any(item["status"] == "running" for item in stages.values()):
            campaign_status = "running"
        else:
            campaign_status = "planned"
    else:
        campaign_status = terminal
    if campaign_status == "failed":
        failed = {stage_id for stage_id, value in stages.items() if value["status"] == "failed"}
        changed = True
        while changed:
            changed = False
            for spec in plan.stages:
                if stages[spec.id]["status"] == "planned" and set(spec.needs) & failed:
                    stages[spec.id]["status"] = "blocked"
                    failed.add(spec.id)
                    changed = True
    return {
        "schema": STATE_SCHEMA,
        "plan_sha256": plan.source_sha256,
        "sequence": len(events),
        "status": campaign_status,
        "stages": [stages[stage.id] for stage in plan.stages],
    }


class CampaignStore:
    """Immutable plan, hash-chained events, and an atomic current-state projection."""

    def __init__(self, root: pathlib.Path, plan: CampaignPlan) -> None:
        self.root = root
        self.plan = plan

    @classmethod
    def open_or_create(
        cls,
        plan_path: str | pathlib.Path,
        *,
        output_root: str | pathlib.Path,
        working_directory: str | pathlib.Path,
    ) -> CampaignStore:
        plan = load_campaign_plan(plan_path)
        output_input = pathlib.Path(output_root).expanduser()
        working_input = pathlib.Path(working_directory).expanduser()
        if output_input.is_symlink() or working_input.is_symlink():
            raise CampaignError("campaign output and working directories may not be symlinks")
        output = output_input.resolve()
        working = working_input.resolve()
        if not working.is_dir():
            raise CampaignError(f"campaign working directory is missing: {working}")
        _ensure_directory_durable(output)
        root = output / f"{plan.name}-{plan.source_sha256[:12]}"
        if root.is_symlink():
            raise CampaignError(f"campaign root is not a regular directory: {root}")
        created = not root.exists()
        if created:
            _initialize_root(root, plan)
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise CampaignError(f"campaign root is not a regular directory: {root}")
        store = cls(root, plan)
        store._verify_plan_files()
        with _CampaignLease(root):
            events = store.events()
            if not events:
                store.append(
                    "campaign_created",
                    stage_ids=[stage.id for stage in plan.stages],
                    working_directory=str(working),
                )
            elif events[0].get("working_directory") != str(working):
                raise CampaignError("campaign working directory differs from its first launch")
            else:
                store.reconcile_state()
                store.verify_completed_outputs()
        return store

    @classmethod
    def from_root(cls, root: str | pathlib.Path) -> CampaignStore:
        supplied = pathlib.Path(root).expanduser()
        if supplied.is_symlink():
            raise CampaignError(f"campaign root is not a regular directory: {supplied}")
        campaign_root = supplied.resolve()
        if not campaign_root.is_dir():
            raise CampaignError(f"campaign root is not a regular directory: {campaign_root}")
        plan = load_campaign_plan(campaign_root / "plan-source.yaml")
        store = cls(campaign_root, plan)
        store._verify_plan_files()
        store.state()
        store.verify_completed_outputs()
        return store

    @property
    def working_directory(self) -> pathlib.Path:
        events = self.events()
        value = events[0].get("working_directory") if events else None
        if not isinstance(value, str):
            raise CampaignError("campaign journal has no working directory")
        return pathlib.Path(value)

    def _verify_plan_files(self) -> None:
        source = self.root / "plan-source.yaml"
        if source.is_symlink() or not source.is_file():
            raise CampaignError("persisted campaign plan source is missing or linked")
        if _digest_path(source) != self.plan.source_sha256:
            raise CampaignError("persisted campaign plan bytes differ from their SHA-256")
        record = _strict_json(self.root / "plan.json", "campaign canonical plan")
        if record != self.plan.as_record():
            raise CampaignError("persisted canonical campaign plan differs from source plan")

    def events(self) -> tuple[dict[str, object], ...]:
        directory = self.root / "events"
        if directory.is_symlink() or not directory.is_dir():
            raise CampaignError("campaign events directory is missing or linked")
        paths = sorted(directory.iterdir())
        expected_names = [f"{index:08d}.json" for index in range(1, len(paths) + 1)]
        if [path.name for path in paths] != expected_names:
            raise CampaignError("campaign event filenames are not a contiguous journal")
        events: list[dict[str, object]] = []
        previous: str | None = None
        for sequence, path in enumerate(paths, 1):
            event = _strict_json(path, f"campaign event {sequence}")
            _validate_event(event, plan=self.plan, sequence=sequence, previous_sha256=previous)
            previous = _digest_path(path)
            events.append(event)
        if events:
            _project(self.plan, tuple(events))
        return tuple(events)

    def append(self, kind: str, **fields: object) -> dict[str, object]:
        self._verify_plan_files()
        events = self.events()
        if events:
            self.reconcile_state()
        sequence = len(events) + 1
        previous = _digest_path(_event_path(self.root, sequence - 1)) if events else None
        event = {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "kind": kind,
            "plan_sha256": self.plan.source_sha256,
            "previous_sha256": previous,
            "time_ns": time.time_ns(),
            **fields,
        }
        _validate_event(event, plan=self.plan, sequence=sequence, previous_sha256=previous)
        candidate = (*events, event)
        projection = _project(self.plan, candidate)
        event_sha = _write_new_json(_event_path(self.root, sequence), event)
        projection["journal_tip_sha256"] = event_sha
        _atomic_write_text(self.root / "state.json", _json(projection) + "\n")
        return projection

    def _validated_state(self, *, repair_stale: bool) -> dict[str, object]:
        events = self.events()
        projection = _project(self.plan, events)
        projection["journal_tip_sha256"] = _digest_path(_event_path(self.root, len(events)))
        path = self.root / "state.json"
        if not path.exists():
            if repair_stale:
                _atomic_write_text(path, _json(projection) + "\n")
            return projection
        state = _strict_json(path, "campaign current state")
        if state == projection:
            return projection
        sequence = state.get("sequence")
        if type(sequence) is int and 1 <= sequence < len(events):
            prefix = _project(self.plan, events[:sequence])
            prefix["journal_tip_sha256"] = _digest_path(_event_path(self.root, sequence))
            if state == prefix:
                if repair_stale:
                    _atomic_write_text(path, _json(projection) + "\n")
                return projection
        raise CampaignError("campaign current state is not a valid journal projection")

    def reconcile_state(self) -> dict[str, object]:
        return self._validated_state(repair_stale=True)

    def state(self) -> dict[str, object]:
        return self._validated_state(repair_stale=False)

    def verify_completed_outputs(self) -> None:
        """Re-hash every completed stage before its evidence can unlock later work."""
        state = self.state()
        for item in cast(list[dict[str, object]], state["stages"]):
            if item["status"] != "completed":
                continue
            stage = next(stage for stage in self.plan.stages if stage.id == item["id"])
            attempt = cast(int, item["attempts"])
            attempt_dir = self.root / "stages" / stage.id / f"attempt-{attempt:04d}"
            result, result_sha = _validate_worker_result(
                attempt_dir / "stage-result.json",
                attempt_dir / "worker-request.json",
                stage,
                attempt,
            )
            if (
                result["plan_sha256"] != self.plan.source_sha256
                or result_sha != item["last_result_sha256"]
            ):
                raise CampaignError(f"completed stage {stage.id!r} differs from its journal")

    def supervisor_observation(self) -> dict[str, object] | None:
        """Read the non-governing liveness observation, if one has been published."""
        path = self.root / "supervisor-heartbeat.json"
        if not path.exists():
            return None
        value = _strict_json(path, "campaign supervisor heartbeat")
        expected = {
            "schema",
            "plan_sha256",
            "session",
            "pid",
            "host",
            "time_ns",
            "observed_state",
            "running_stage",
            "detail",
        }
        state = value.get("observed_state")
        running_stage = value.get("running_stage")
        detail = value.get("detail")
        if (
            set(value) != expected
            or value.get("schema") != "autoforecast.campaign-supervisor-heartbeat.v1"
            or value.get("plan_sha256") != self.plan.source_sha256
            or not isinstance(value.get("session"), str)
            or not cast(str, value.get("session"))
            or type(value.get("pid")) is not int
            or cast(int, value.get("pid")) < 1
            or not isinstance(value.get("host"), str)
            or not cast(str, value.get("host"))
            or type(value.get("time_ns")) is not int
            or cast(int, value.get("time_ns")) < 1
            or type(state) is not str
            or state not in {"planned", "running", "completed", "failed", "error"}
            or (
                running_stage is not None
                and (
                    type(running_stage) is not str
                    or running_stage not in {stage.id for stage in self.plan.stages}
                )
            )
            or (detail is not None and (not isinstance(detail, str) or not detail))
            or (state == "running" and running_stage is None)
            or (state in {"planned", "completed", "failed"} and running_stage is not None)
            or (state == "error" and detail is None)
            or (state != "error" and detail is not None)
        ):
            raise CampaignError("campaign supervisor heartbeat is invalid")
        return value


class _CampaignLease:
    def __init__(self, root: pathlib.Path) -> None:
        self.path = root / ".supervisor.lock"
        self.handle = None

    def __enter__(self) -> _CampaignLease:
        self.handle = _open_lock(self.path)
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise CampaignError("campaign already has an active supervisor") from exc
        return self

    def __exit__(self, typ, value, traceback) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _expand_command(
    stage: StageSpec,
    *,
    campaign_root: pathlib.Path,
    attempt_dir: pathlib.Path,
    attempt: int,
) -> tuple[str, ...]:
    replacements = {
        "{campaign_root}": str(campaign_root),
        "{attempt_dir}": str(attempt_dir),
        "{stage_id}": stage.id,
        "{attempt}": str(attempt),
        "{previous_attempt_dir}": (
            str(campaign_root / "stages" / stage.id / "attempt-0000")
            if attempt == 1
            else str(campaign_root / "stages" / stage.id / f"attempt-{attempt - 1:04d}")
        ),
    }
    pattern = re.compile(
        "|".join(re.escape(token) for token in sorted(replacements, key=len, reverse=True))
    )
    expanded = []
    for argument in stage.command:
        value = pattern.sub(lambda match: replacements[match.group(0)], argument)
        expanded.append(value)
    return tuple(expanded)


def _worker_request(
    store: CampaignStore,
    stage: StageSpec,
    attempt: int,
    attempt_dir: pathlib.Path,
    worker_token_sha256: str,
) -> dict[str, object]:
    lease_root = store.working_directory / ".autoforecast" / "device-leases"
    _ensure_directory_durable(lease_root)
    return {
        "schema": WORKER_REQUEST_SCHEMA,
        "plan_sha256": store.plan.source_sha256,
        "campaign_root": str(store.root),
        "stage_id": stage.id,
        "attempt": attempt,
        "attempt_dir": str(attempt_dir),
        "worker_token_sha256": worker_token_sha256,
        "working_directory": str(store.working_directory),
        "command": list(
            _expand_command(
                stage,
                campaign_root=store.root,
                attempt_dir=attempt_dir,
                attempt=attempt,
            )
        ),
        "timeout_seconds": stage.timeout_seconds,
        "heartbeat_seconds": max(1, min(60, stage.timeout_seconds // 10)),
        "required_outputs": list(stage.required_outputs),
        "retry_mode": stage.retry_mode,
        "previous_attempt_dir": (
            None
            if attempt == 1
            else str(store.root / "stages" / stage.id / f"attempt-{attempt - 1:04d}")
        ),
        "device_lease_path": str(lease_root / f"{stage.resources.device}.lock"),
        "resources": stage.resources.as_record(),
    }


def _validate_worker_request(value: dict[str, object], path: pathlib.Path) -> None:
    expected = {
        "schema",
        "plan_sha256",
        "campaign_root",
        "stage_id",
        "attempt",
        "attempt_dir",
        "worker_token_sha256",
        "working_directory",
        "command",
        "timeout_seconds",
        "heartbeat_seconds",
        "required_outputs",
        "retry_mode",
        "previous_attempt_dir",
        "device_lease_path",
        "resources",
    }
    if set(value) != expected or value.get("schema") != WORKER_REQUEST_SCHEMA:
        raise CampaignError("worker request fields or schema differ")
    plan_sha = value["plan_sha256"]
    if not isinstance(plan_sha, str) or _SHA256.fullmatch(plan_sha) is None:
        raise CampaignError("worker request plan SHA-256 is invalid")
    token_sha = value["worker_token_sha256"]
    if not isinstance(token_sha, str) or _SHA256.fullmatch(token_sha) is None:
        raise CampaignError("worker request token SHA-256 is invalid")
    stage_id = value["stage_id"]
    if not isinstance(stage_id, str) or _ID.fullmatch(stage_id) is None:
        raise CampaignError("worker request stage ID is invalid")
    attempt = _positive_int(value["attempt"], "worker request attempt")
    for field in ("attempt_dir", "campaign_root", "working_directory"):
        if not isinstance(value[field], str) or not value[field]:
            raise CampaignError(f"worker request {field} must be a non-empty path string")
    attempt_dir = pathlib.Path(cast(str, value["attempt_dir"])).resolve()
    if path.resolve() != attempt_dir / "worker-request.json":
        raise CampaignError("worker request path does not match its attempt directory")
    campaign_root = pathlib.Path(cast(str, value["campaign_root"])).resolve()
    expected_attempt = campaign_root / "stages" / stage_id / f"attempt-{attempt:04d}"
    if attempt_dir != expected_attempt:
        raise CampaignError("worker request attempt directory escapes campaign stage layout")
    working = pathlib.Path(cast(str, value["working_directory"])).resolve()
    if not working.is_dir():
        raise CampaignError("worker request working directory is missing")
    retry_mode = value["retry_mode"]
    if type(retry_mode) is not str or retry_mode not in {
        "restart",
        "command_managed_resume",
    }:
        raise CampaignError("worker request retry_mode is invalid")
    previous = value["previous_attempt_dir"]
    expected_previous = (
        None
        if attempt == 1
        else str(campaign_root / "stages" / stage_id / f"attempt-{attempt - 1:04d}")
    )
    if previous != expected_previous:
        raise CampaignError("worker request previous attempt identity is invalid")
    command = value["command"]
    if (
        type(command) is not list
        or not command
        or not all(isinstance(item, str) and item and "\x00" not in item for item in command)
    ):
        raise CampaignError("worker request command is invalid")
    timeout = _positive_int(value["timeout_seconds"], "worker timeout", minimum=2)
    heartbeat = _positive_int(value["heartbeat_seconds"], "worker heartbeat")
    if heartbeat >= timeout:
        raise CampaignError("worker heartbeat must be shorter than timeout")
    outputs = value["required_outputs"]
    if type(outputs) is not list:
        raise CampaignError("worker required_outputs must be a list")
    folded_outputs: set[str] = set()
    for output in outputs:
        relative = _safe_relative(output, "worker required output")
        folded = relative.casefold()
        if folded in folded_outputs:
            raise CampaignError("worker required_outputs must be unique, including by case")
        folded_outputs.add(folded)
        if pathlib.PurePosixPath(relative).parts[0].casefold() in _CONTROL_PATHS:
            raise CampaignError("worker required output uses a wrapper-owned control path")
    resources = _parse_resources(value["resources"], stage_id)
    lease_path = pathlib.Path(cast(str, value["device_lease_path"])).resolve()
    expected_lease = working / ".autoforecast" / "device-leases" / f"{resources.device}.lock"
    if lease_path != expected_lease:
        raise CampaignError("worker request device lease escapes the shared lease directory")


def _output_manifest(
    attempt_dir: pathlib.Path, outputs: tuple[str, ...]
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for relative in outputs:
        root = attempt_dir.joinpath(*pathlib.PurePosixPath(relative).parts)
        if root.resolve() != root or root.is_symlink() or not root.exists():
            raise CampaignError(f"required stage output is missing or linked: {relative}")
        if root.is_file():
            manifest.append({"path": relative, "kind": "file", "sha256": _digest_path(root)})
            continue
        if not root.is_dir():
            raise CampaignError(
                f"required stage output is not a regular file/directory: {relative}"
            )
        files: list[dict[str, str]] = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise CampaignError(f"required stage output contains a symlink: {path}")
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(attempt_dir).as_posix(),
                        "sha256": _digest_path(path),
                    }
                )
            elif not path.is_dir():
                raise CampaignError(f"required stage output contains a special path: {path}")
        manifest.append({"path": relative, "kind": "directory", "files": files})
    return manifest


def _atomic_heartbeat(path: pathlib.Path, value: dict[str, object]) -> None:
    _atomic_write_text(path, _json(value) + "\n")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        # A sandbox or platform policy can deny group signalling even though the direct child is
        # still ours. Do not mistake that denial for process exit and leak a live model worker.
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.05)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _observed_process_create_time_ns(pid: int) -> int | None:
    """Read the kernel-backed process birth identity; return None only after process exit."""
    try:
        import psutil
    except ImportError as error:
        raise CampaignError(
            "campaign recovery requires psutil to authenticate process identities"
        ) from error
    try:
        return int(round(psutil.Process(pid).create_time() * 1_000_000_000))
    except psutil.NoSuchProcess:
        return None
    except psutil.Error as error:
        raise CampaignError(
            f"cannot authenticate process identity for pid {pid}: {error}"
        ) from error


def _current_process_create_time_ns() -> int:
    observed = _observed_process_create_time_ns(os.getpid())
    if observed is None:  # pragma: no cover - the executing process necessarily exists
        raise CampaignError("cannot authenticate the current campaign process identity")
    return observed


def _terminate_pid_group(pid: int, process_create_time_ns: int) -> None:
    """Terminate one detached group only while its PID still names the recorded process."""
    observed = _observed_process_create_time_ns(pid)
    if observed is None:
        return
    if observed != process_create_time_ns:
        raise CampaignError(
            f"process identity changed for pid {pid}; refusing to signal a reused PID"
        )
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.05)
    if _pid_alive(pid):
        observed = _observed_process_create_time_ns(pid)
        if observed is None:
            return
        if observed != process_create_time_ns:
            raise CampaignError(
                f"process identity changed for pid {pid}; refusing to signal a reused PID"
            )
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)


def _terminate_escaped_lease_holders(lock_path: pathlib.Path) -> None:
    """Terminate same-user descendants that escaped the command process group.

    A stage receives a unique child-lock descriptor.  A process still holding that descriptor
    after the command group is gone can only be a forked command descendant in the supported
    local, single-user execution model.  ``lsof`` is used only on this exceptional path; ordinary
    model workers and their subprocesses remain governed by the process group.
    """
    if not _lock_is_held(lock_path):
        return
    holders = _lease_holder_pids(lock_path) - {os.getpid()}
    if not holders:
        if not _lock_is_held(lock_path):
            return
        raise CampaignError("child lease remains held without an identifiable command descendant")
    for pid in holders:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while _lock_is_held(lock_path) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _lock_is_held(lock_path):
        for pid in holders:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while _lock_is_held(lock_path) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _lock_is_held(lock_path):
        raise CampaignError("escaped command descendant did not release its child lease")


def _process_tree_rss(pid: int) -> int:
    """Return current resident bytes for one command tree without requiring psutil."""
    try:
        import psutil
    except ImportError:
        try:
            outcome = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,rss="],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise CampaignError(
                "cannot measure worker memory: install the campaign runtime with psutil or "
                "permit the ps process-list command"
            ) from error
        if outcome.returncode != 0:
            raise CampaignError("cannot measure worker memory with ps") from None
        rows: dict[int, tuple[int, int]] = {}
        for line in outcome.stdout.splitlines():
            try:
                process_id, parent_id, rss_kib = (int(value) for value in line.split())
            except (TypeError, ValueError) as exc:
                raise CampaignError("cannot measure worker memory: malformed ps row") from exc
            if process_id < 1 or parent_id < 0 or rss_kib < 0:
                raise CampaignError("cannot measure worker memory: invalid ps row") from None
            rows[process_id] = (parent_id, rss_kib)
        descendants = {pid}
        changed = True
        while changed:
            changed = False
            for process_id, (parent_id, _rss) in rows.items():
                if parent_id in descendants and process_id not in descendants:
                    descendants.add(process_id)
                    changed = True
        if pid not in rows:
            raise CampaignError("cannot measure worker memory: process is absent from ps") from None
        return sum(rows[item][1] * 1024 for item in descendants)
    else:
        try:
            process = psutil.Process(pid)
            rss_bytes = process.memory_info().rss
            for descendant in process.children(recursive=True):
                try:
                    rss_bytes += descendant.memory_info().rss
                except psutil.NoSuchProcess:
                    # A descendant may exit between the tree walk and its RSS read.
                    continue
            return rss_bytes
        except psutil.NoSuchProcess as exc:
            raise CampaignError("cannot measure worker memory: process exited") from exc
        except psutil.Error as exc:
            raise CampaignError(f"cannot measure worker memory: {exc}") from exc


def run_stage_worker(request_path: str | pathlib.Path) -> int:
    """Execute one immutable request and publish a heartbeat and atomic terminal result."""
    supplied = pathlib.Path(request_path).expanduser()
    if supplied.is_symlink():
        raise CampaignError("campaign worker request may not be a symlink")
    path = supplied.resolve()
    request = _strict_json(path, "campaign worker request")
    _validate_worker_request(request, path)
    attempt_dir = pathlib.Path(cast(str, request["attempt_dir"]))
    lock_path = attempt_dir / ".worker.lock"
    with _open_lock(lock_path) as lock:
        # A supervisor liveness probe briefly takes this flock before reporting it free.
        # Tolerate that bounded hand-off without allowing two long-lived workers to overlap.
        lock_deadline = time.monotonic() + 1.0
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                if time.monotonic() >= lock_deadline:
                    raise CampaignError("campaign stage already has an active worker") from exc
                time.sleep(0.01)
            else:
                break
        result_path = attempt_dir / "stage-result.json"
        if result_path.exists():
            raise CampaignError("campaign stage attempt already has a terminal result")
        request_sha = _digest_path(path)
        started = time.time_ns()
        identity = {
            "schema": WORKER_IDENTITY_SCHEMA,
            "worker_request_sha256": request_sha,
            "wrapper_pid": os.getpid(),
            "process_create_time_ns": _current_process_create_time_ns(),
            "host": socket.gethostname(),
            "started_at_ns": started,
        }
        _write_new_json(attempt_dir / "worker-identity.json", identity)
        worker_token = _load_worker_token(attempt_dir / "worker-token", consume=True)
        if _digest_bytes(worker_token.encode("ascii")) != request["worker_token_sha256"]:
            raise CampaignError("worker launch token differs from its frozen request")
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        child: subprocess.Popen[bytes] | None = None
        exit_code: int | None = None
        reason: str | None = None
        status = "failed"
        device_lock = None
        host_lock = None
        child_lock = None
        peak_rss_bytes = 0
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                deadline = time.monotonic() + cast(int, request["timeout_seconds"])
                heartbeat_seconds = cast(int, request["heartbeat_seconds"])
                next_heartbeat = time.monotonic()
                device_lock = _open_lock(pathlib.Path(cast(str, request["device_lease_path"])))
                host_lock = open_host_training_lease()
                while exit_code is None:
                    try:
                        fcntl.flock(host_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        acquired = False
                    else:
                        try:
                            fcntl.flock(device_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            fcntl.flock(host_lock.fileno(), fcntl.LOCK_UN)
                            acquired = False
                        else:
                            acquired = True
                    if not acquired:
                        now = time.monotonic()
                        if now >= next_heartbeat:
                            _atomic_heartbeat(
                                attempt_dir / "heartbeat.json",
                                {
                                    "schema": WORKER_HEARTBEAT_SCHEMA,
                                    "worker_request_sha256": request_sha,
                                    "wrapper_pid": os.getpid(),
                                    "child_pid": None,
                                    "phase": "waiting_for_device",
                                    "rss_bytes": 0,
                                    "time_ns": time.time_ns(),
                                },
                            )
                            next_heartbeat = now + heartbeat_seconds
                        if now >= deadline:
                            exit_code = 75
                            reason = (
                                "stage timed out waiting for exclusive "
                                f"{cast(dict[str, object], request['resources'])['device']} device"
                            )
                            break
                        time.sleep(min(1.0, max(0.01, deadline - now)))
                    else:
                        break
                if exit_code is None:
                    try:
                        child_lock = _open_lock(attempt_dir / ".child.lock")
                        fcntl.flock(child_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        child_environment = dict(os.environ)
                        child_environment["AUTOFORECAST_CAMPAIGN_CHILD_LOCK_FD"] = str(
                            child_lock.fileno()
                        )
                        child_environment["AUTOFORECAST_CAMPAIGN_DEVICE_LOCK_FD"] = str(
                            device_lock.fileno()
                        )
                        child_environment[HOST_TRAINING_LEASE_FD] = str(host_lock.fileno())
                        child = subprocess.Popen(
                            [sys.executable, "-P", "-m", "autoforecast.campaign_child", str(path)],
                            cwd=cast(str, request["working_directory"]),
                            stdout=stdout,
                            stderr=stderr,
                            start_new_session=True,
                            pass_fds=(
                                child_lock.fileno(),
                                device_lock.fileno(),
                                host_lock.fileno(),
                            ),
                            env=child_environment,
                        )
                        child_lock.close()
                        child_lock = None
                        device_lock.close()
                        device_lock = None
                        host_lock.close()
                        host_lock = None
                    except OSError as exc:
                        exit_code = 127 if exc.errno == errno.ENOENT else 126
                        reason = f"command could not start: {type(exc).__name__}: {exc}"
                if child is not None:
                    resources = cast(dict[str, object], request["resources"])
                    memory_limit = cast(float, resources["memory_gib"]) * 1024**3
                    while exit_code is None:
                        now = time.monotonic()
                        try:
                            rss_bytes = _process_tree_rss(child.pid)
                        except CampaignError as exc:
                            try:
                                child.wait(timeout=0.05)
                            except subprocess.TimeoutExpired:
                                pass
                            if child.poll() is not None:
                                exit_code = child.returncode
                                break
                            _terminate_process_group(child)
                            exit_code = child.returncode
                            reason = f"stage memory measurement failed: {exc}"
                            break
                        peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
                        if rss_bytes > memory_limit:
                            _terminate_process_group(child)
                            exit_code = child.returncode
                            reason = (
                                f"stage exceeded memory_gib={resources['memory_gib']}"
                            )
                            break
                        if now >= next_heartbeat:
                            _atomic_heartbeat(
                                attempt_dir / "heartbeat.json",
                                {
                                    "schema": WORKER_HEARTBEAT_SCHEMA,
                                    "worker_request_sha256": request_sha,
                                    "wrapper_pid": os.getpid(),
                                    "child_pid": child.pid,
                                    "phase": "running",
                                    "rss_bytes": rss_bytes,
                                    "time_ns": time.time_ns(),
                                },
                            )
                            next_heartbeat = now + heartbeat_seconds
                        if now >= deadline:
                            _terminate_process_group(child)
                            exit_code = child.returncode
                            reason = f"stage exceeded timeout_seconds={request['timeout_seconds']}"
                            break
                        try:
                            exit_code = child.wait(
                                timeout=max(
                                    0.01,
                                    min(next_heartbeat, deadline, now + 1.0) - now,
                                )
                            )
                        except subprocess.TimeoutExpired:
                            continue
                    _terminate_process_group(child)
                    _terminate_escaped_lease_holders(attempt_dir / ".child.lock")
            artifacts: list[dict[str, object]] = []
            if exit_code == 0:
                try:
                    artifacts = _output_manifest(
                        attempt_dir,
                        tuple(cast(list[str], request["required_outputs"])),
                    )
                except CampaignError as exc:
                    reason = str(exc)
                else:
                    status = "completed"
            elif reason is None:
                marker = "[autoforecast:non-retryable]"
                diagnostic_paths = [stderr_path]
                diagnostic_paths.extend(sorted(attempt_dir.rglob("worker.log")))
                diagnostic_text = "\n".join(
                    _read_tail(path, 4000)
                    for path in diagnostic_paths
                    if path.is_file() and not path.is_symlink()
                )
                if marker in diagnostic_text:
                    suffix = diagnostic_text.split(marker, 1)[1].lstrip().splitlines()
                    detail = suffix[0].strip() if suffix else ""
                    reason = f"non-retryable: {detail or 'command rejected its data/configuration'}"
                else:
                    reason = f"command exited with status {exit_code}"
            result = {
                "schema": WORKER_RESULT_SCHEMA,
                "worker_token": worker_token,
                "worker_request_sha256": request_sha,
                "plan_sha256": request["plan_sha256"],
                "stage_id": request["stage_id"],
                "attempt": request["attempt"],
                "status": status,
                "exit_code": exit_code,
                "reason": reason,
                "wrapper_pid": os.getpid(),
                "child_pid": None if child is None else child.pid,
                "started_at_ns": started,
                "finished_at_ns": time.time_ns(),
                "peak_rss_bytes": peak_rss_bytes,
                "artifacts": artifacts,
            }
            _publish_worker_result(result_path, result)
            return 0 if status == "completed" else 2
        except BaseException as error:
            if child is not None:
                _terminate_process_group(child)
            if (attempt_dir / ".child.lock").exists():
                _terminate_escaped_lease_holders(attempt_dir / ".child.lock")
            internal_exit_code = exit_code if type(exit_code) is int else 70
            _publish_worker_result(
                result_path,
                {
                    "schema": WORKER_RESULT_SCHEMA,
                    "worker_token": worker_token,
                    "worker_request_sha256": request_sha,
                    "plan_sha256": request["plan_sha256"],
                    "stage_id": request["stage_id"],
                    "attempt": request["attempt"],
                    "status": "failed",
                    "exit_code": internal_exit_code,
                    "reason": (
                        f"campaign worker internal failure: {type(error).__name__}: {error}"
                    ),
                    "wrapper_pid": os.getpid(),
                    "child_pid": None if child is None else child.pid,
                    "started_at_ns": started,
                    "finished_at_ns": time.time_ns(),
                    "peak_rss_bytes": peak_rss_bytes,
                    "artifacts": [],
                },
            )
            return 2
        finally:
            if child_lock is not None:
                fcntl.flock(child_lock.fileno(), fcntl.LOCK_UN)
                child_lock.close()
            if device_lock is not None:
                fcntl.flock(device_lock.fileno(), fcntl.LOCK_UN)
                device_lock.close()
            if host_lock is not None:
                fcntl.flock(host_lock.fileno(), fcntl.LOCK_UN)
                host_lock.close()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.Error:
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _admit(stage: StageSpec, campaign_root: pathlib.Path) -> None:
    gib = 1024**3
    free_disk = shutil.disk_usage(campaign_root).free / gib
    if free_disk < stage.resources.min_free_disk_gib:
        raise CampaignError(
            f"stage {stage.id!r} needs {stage.resources.min_free_disk_gib:.2f} GiB free disk; "
            f"only {free_disk:.2f} GiB is currently free"
        )


def _validate_worker_result(
    path: pathlib.Path,
    request_path: pathlib.Path,
    stage: StageSpec,
    attempt: int,
) -> tuple[dict[str, object], str]:
    value = _strict_json(path, "campaign worker result")
    expected = {
        "schema",
        "worker_token",
        "worker_request_sha256",
        "plan_sha256",
        "stage_id",
        "attempt",
        "status",
        "exit_code",
        "reason",
        "wrapper_pid",
        "child_pid",
        "started_at_ns",
        "finished_at_ns",
        "peak_rss_bytes",
        "artifacts",
    }
    if set(value) != expected or value.get("schema") != WORKER_RESULT_SCHEMA:
        raise CampaignError("campaign worker result fields or schema differ")
    request = _strict_json(request_path, "campaign worker request")
    _validate_worker_request(request, request_path)
    integer_fields = ("attempt", "exit_code", "wrapper_pid", "started_at_ns", "finished_at_ns")
    if (
        type(value["worker_token"]) is not str
        or len(cast(str, value["worker_token"])) != 64
        or bool(set(cast(str, value["worker_token"])) - set("0123456789abcdef"))
        or _digest_bytes(cast(str, value["worker_token"]).encode("ascii"))
        != request["worker_token_sha256"]
        or value["worker_request_sha256"] != _digest_path(request_path)
        or value["plan_sha256"] != request["plan_sha256"]
        or value["stage_id"] != stage.id
        or type(value["attempt"]) is not int
        or value["attempt"] != attempt
        or type(value["status"]) is not str
        or value["status"] not in {"completed", "failed"}
        or any(type(value[field]) is not int for field in integer_fields)
        or cast(int, value["wrapper_pid"]) < 1
        or cast(int, value["started_at_ns"]) < 1
        or cast(int, value["finished_at_ns"]) < cast(int, value["started_at_ns"])
        or (
            value["child_pid"] is not None
            and (type(value["child_pid"]) is not int or cast(int, value["child_pid"]) < 1)
        )
        or type(value["peak_rss_bytes"]) is not int
        or cast(int, value["peak_rss_bytes"]) < 0
        or type(value["artifacts"]) is not list
    ):
        raise CampaignError("campaign worker result identity differs from its launch")
    if value["status"] == "completed":
        actual = _output_manifest(path.parent, stage.required_outputs)
        if value["exit_code"] != 0 or value["reason"] is not None or value["artifacts"] != actual:
            raise CampaignError("completed worker result does not match current output bytes")
    else:
        if not isinstance(value["reason"], str) or not value["reason"] or value["artifacts"] != []:
            raise CampaignError("failed worker result must contain a reason and no artifacts")
    return value, _digest_path(path)


class CampaignSupervisor:
    """Sequentially execute a frozen campaign while one durable lease is held."""

    def __init__(self, store: CampaignStore, *, poll_seconds: float = 1.0) -> None:
        if not isinstance(poll_seconds, int | float) or isinstance(poll_seconds, bool):
            raise CampaignError("poll_seconds must be a positive finite number")
        self.poll_seconds = float(poll_seconds)
        if not (0 < self.poll_seconds < float("inf")):
            raise CampaignError("poll_seconds must be a positive finite number")
        self.store = store
        self.session = uuid.uuid4().hex
        self._last_heartbeat = 0.0
        self._live_offsets: dict[str, int] = {}
        self._live_decoders: dict[str, codecs.IncrementalDecoder] = {}

    def _tail_live_output(self, attempt_dir: pathlib.Path) -> None:
        """Relay newly written stage logs to an interactive supervisor terminal."""
        if not (
            _stderr_isatty()
            or (sys.stderr is not None and os.environ.get("AUTOFORECAST_LIVE_PROGRESS") == "1")
        ):
            return
        paths = [attempt_dir / "stdout.log", attempt_dir / "stderr.log"]
        paths.extend(sorted(attempt_dir.glob("*/worker.log")))
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            key = str(path)
            try:
                size = path.stat().st_size
                offset = self._live_offsets.get(key, 0)
                if size < offset:
                    offset = 0
                    self._live_decoders.pop(key, None)
                with path.open("rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                self._live_offsets[key] = offset + len(chunk)
            except OSError:
                continue
            if chunk:
                stream = getattr(sys.stderr, "buffer", None)
                if stream is not None:
                    stream.write(chunk)
                    stream.flush()
                else:
                    decoder = self._live_decoders.setdefault(
                        key, codecs.getincrementaldecoder("utf-8")(errors="replace")
                    )
                    sys.stderr.write(decoder.decode(chunk))
                    sys.stderr.flush()

    def _campaign_stage_position(self, stage: StageSpec) -> tuple[int, int]:
        """Return a one-based stable position from the immutable compiled plan."""
        return self.store.plan.stages.index(stage) + 1, len(self.store.plan.stages)

    def _display_stage(self, stage: StageSpec) -> tuple[int, int, str, bool]:
        """Return user workflow position without changing durable plan/checkpoint identity."""
        grouped = _workflow_stage(self.store.plan.stages, stage)
        if grouped is not None:
            return grouped
        index, total = self._campaign_stage_position(stage)
        return index, total, _stage_title(stage.id), False

    @staticmethod
    def _live_progress_enabled() -> bool:
        return _stderr_isatty() or (
            sys.stderr is not None and os.environ.get("AUTOFORECAST_LIVE_PROGRESS") == "1"
        )

    def _announce_stage(self, stage: StageSpec, attempt: int) -> None:
        """Make the outer campaign position visible before the child emits its own bar."""
        if not self._live_progress_enabled():
            return
        index, total, title, substage = self._display_stage(stage)
        if substage:
            phase_index, phase_total = (
                (6, 6) if stage.id == "aggregate_selection" else (3, 3)
            )
            message = (
                f"Stage {index}/{total} | Phase {phase_index}/{phase_total} — {title} "
                f"[{stage.id}] - attempt {attempt}/{stage.max_attempts}"
            )
        else:
            message = _stage_banner(
                f"Stage {index}/{total}: {title} "
                f"[{stage.id}] - attempt {attempt}/{stage.max_attempts}"
            )
        print(message, file=sys.stderr, flush=True)

    def _heartbeat(
        self,
        state: dict[str, object],
        *,
        detail: str | None = None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat < 30:
            return
        running = next(
            (
                item["id"]
                for item in cast(list[dict[str, object]], state["stages"])
                if item["status"] == "running"
            ),
            None,
        )
        _atomic_heartbeat(
            self.store.root / "supervisor-heartbeat.json",
            {
                "schema": "autoforecast.campaign-supervisor-heartbeat.v1",
                "plan_sha256": self.store.plan.source_sha256,
                "session": self.session,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "time_ns": time.time_ns(),
                "observed_state": "error" if detail is not None else state["status"],
                "running_stage": running,
                "detail": detail,
            },
        )
        self._last_heartbeat = now

    def _stage_state(self, stage_id: str) -> dict[str, object]:
        state = self.store.state()
        return next(
            cast(dict[str, object], item)
            for item in cast(list[dict[str, object]], state["stages"])
            if item["id"] == stage_id
        )

    def _publish_terminal_notification(self, state: dict[str, object]) -> None:
        """Dispatch at most once by sealing intent before the non-transactional side effect."""
        path = self.store.root / "terminal-notification.json"
        intent_path = self.store.root / "terminal-notification-intent.json"
        if path.exists():
            value = _strict_json(path, "campaign terminal notification")
            if (
                value.get("schema") != "autoforecast.campaign-terminal-notification.v1"
                or value.get("campaign_name") != self.store.plan.name
                or value.get("campaign_status") != state["status"]
                or value.get("artifact_path") != str(self.store.root)
            ):
                raise CampaignError("campaign terminal notification identity is invalid")
            return
        if intent_path.exists():
            intent = _strict_json(intent_path, "campaign terminal notification intent")
            expected_intent = {
                "schema",
                "campaign_name",
                "campaign_status",
                "artifact_path",
            }
            if (
                set(intent) != expected_intent
                or intent.get("schema") != NOTIFICATION_INTENT_SCHEMA
                or intent.get("campaign_name") != self.store.plan.name
                or intent.get("campaign_status") != state["status"]
                or intent.get("artifact_path") != str(self.store.root)
            ):
                raise CampaignError("campaign terminal notification intent is invalid")
            _write_new_json(
                path,
                {
                    "schema": "autoforecast.campaign-terminal-notification.v1",
                    "status": "indeterminate_after_interruption",
                    "campaign_name": self.store.plan.name,
                    "campaign_status": state["status"],
                    "configured_sinks": [],
                    "delivered_sinks": [],
                    "failures": [
                        "a prior process sealed dispatch intent but did not seal its result; "
                        "delivery is not retried because the sink is not transactional"
                    ],
                    "artifact_path": str(self.store.root),
                },
            )
            return
        _write_new_json(
            intent_path,
            {
                "schema": NOTIFICATION_INTENT_SCHEMA,
                "campaign_name": self.store.plan.name,
                "campaign_status": state["status"],
                "artifact_path": str(self.store.root),
            },
        )
        try:
            result = campaign_notification_from_environment(
                campaign_name=self.store.plan.name,
                status=cast(str, state["status"]),
                path=str(self.store.root),
            )
        except Exception as error:  # notification is a non-governing side effect
            result = {
                "schema": "autoforecast.campaign-terminal-notification.v1",
                "status": "failed",
                "campaign_name": self.store.plan.name,
                "campaign_status": state["status"],
                "configured_sinks": [],
                "delivered_sinks": [],
                "failures": [f"notification setup failed ({type(error).__name__})"],
                "artifact_path": str(self.store.root),
            }
        _write_new_json(path, result)

    def _attempt_dir(self, stage: StageSpec, attempt: int) -> pathlib.Path:
        return self.store.root / "stages" / stage.id / f"attempt-{attempt:04d}"

    def _prepare_attempt(self, stage: StageSpec, attempt: int) -> pathlib.Path:
        stages_root = self.store.root / "stages"
        if (
            stages_root.is_symlink()
            or not stages_root.is_dir()
            or stages_root.resolve() != stages_root
        ):
            raise CampaignError("campaign stages root is missing or linked")
        parent = self.store.root / "stages" / stage.id
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
                raise CampaignError(f"campaign stage directory is unsafe: {parent}")
        else:
            _ensure_directory_durable(parent)
        attempt_dir = self._attempt_dir(stage, attempt)
        if attempt_dir.exists():
            if (
                attempt_dir.is_symlink()
                or not attempt_dir.is_dir()
                or attempt_dir.resolve() != attempt_dir
            ):
                raise CampaignError(f"stage attempt path is unsafe: {attempt_dir}")
        else:
            attempt_dir.mkdir()
            _fsync_directory(parent)
        token_path = attempt_dir / "worker-token"
        if token_path.exists() or token_path.is_symlink():
            worker_token = _load_worker_token(token_path, consume=False)
        else:
            worker_token = secrets.token_hex(32)
            _write_new_worker_token(token_path, worker_token)
        request = _worker_request(
            self.store,
            stage,
            attempt,
            attempt_dir,
            _digest_bytes(worker_token.encode("ascii")),
        )
        payload = (_json(request) + "\n").encode("utf-8")
        request_sha = _digest_bytes(payload)
        self.store.append(
            "stage_launching",
            stage_id=stage.id,
            attempt=attempt,
            worker_request_sha256=request_sha,
        )
        request_path = attempt_dir / "worker-request.json"
        if request_path.exists():
            if _digest_path(request_path) != request_sha:
                raise CampaignError("existing worker request differs from the launch journal")
        else:
            _write_new_json(request_path, request)
        return request_path

    def _launch_worker(self, request_path: pathlib.Path) -> subprocess.Popen[bytes]:
        diagnostics_path = request_path.parent / "worker-wrapper.log"
        try:
            descriptor = os.open(
                diagnostics_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                0o600,
            )
        except OSError as exc:
            raise CampaignError(f"campaign worker log path is unsafe: {diagnostics_path}") from exc
        diagnostics = os.fdopen(descriptor, "ab")
        environment = os.environ.copy()
        request = _strict_json(request_path, "campaign worker request")
        stage_id = cast(str, request["stage_id"])
        stage = next(stage for stage in self.store.plan.stages if stage.id == stage_id)
        stage_index, stage_total, _title, _substage = self._display_stage(stage)
        environment["AUTOFORECAST_CAMPAIGN_STAGE_INDEX"] = str(stage_index)
        environment["AUTOFORECAST_CAMPAIGN_STAGE_TOTAL"] = str(stage_total)
        if _workflow_stage(self.store.plan.stages, stage) is not None:
            role = {
                self.store.plan.stages[0].id: "validation",
                "aggregate_selection": "selection_lock",
                "final_fit": "final_fit",
                "final_score": "final_score",
                "replay_report": "report",
            }[stage.id]
            environment["AUTOFORECAST_WORKFLOW_LAYOUT"] = "single_model_benchmark_v1"
            environment["AUTOFORECAST_WORKFLOW_STAGE_ROLE"] = role
        if self._live_progress_enabled():
            environment["AUTOFORECAST_LIVE_PROGRESS"] = "1"
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(_PROJECT_ROOT) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        try:
            return subprocess.Popen(
                [
                    sys.executable,
                    "-P",
                    "-m",
                    "autoforecast.campaign_worker",
                    str(request_path),
                ],
                cwd=self.store.working_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=diagnostics,
                stderr=diagnostics,
                start_new_session=True,
            )
        finally:
            diagnostics.close()

    @staticmethod
    def _worker_identity(path: pathlib.Path, request_sha: str) -> dict[str, object]:
        identity = _strict_json(path, "campaign worker identity")
        expected = {
            "schema",
            "worker_request_sha256",
            "wrapper_pid",
            "process_create_time_ns",
            "host",
            "started_at_ns",
        }
        if (
            set(identity) != expected
            or identity.get("schema") != WORKER_IDENTITY_SCHEMA
            or identity.get("worker_request_sha256") != request_sha
            or type(identity.get("host")) is not str
            or not cast(str, identity.get("host"))
            or type(identity.get("wrapper_pid")) is not int
            or cast(int, identity.get("wrapper_pid")) < 1
            or type(identity.get("process_create_time_ns")) is not int
            or cast(int, identity.get("process_create_time_ns")) < 1
            or type(identity.get("started_at_ns")) is not int
            or cast(int, identity.get("started_at_ns")) < 1
        ):
            raise CampaignError("campaign worker identity is invalid")
        return identity

    @staticmethod
    def _child_identity(path: pathlib.Path, request_sha: str) -> dict[str, object]:
        identity = _strict_json(path, "campaign child identity")
        expected = {
            "schema",
            "worker_request_sha256",
            "child_pid",
            "process_create_time_ns",
            "host",
            "started_at_ns",
        }
        if (
            set(identity) != expected
            or identity.get("schema") != CHILD_IDENTITY_SCHEMA
            or identity.get("worker_request_sha256") != request_sha
            or type(identity.get("host")) is not str
            or not cast(str, identity.get("host"))
            or type(identity.get("child_pid")) is not int
            or cast(int, identity.get("child_pid")) < 1
            or type(identity.get("process_create_time_ns")) is not int
            or cast(int, identity.get("process_create_time_ns")) < 1
            or type(identity.get("started_at_ns")) is not int
            or cast(int, identity.get("started_at_ns")) < 1
        ):
            raise CampaignError("campaign child identity is invalid")
        return identity

    def _terminate_stranded_child(self, attempt_dir: pathlib.Path, request_sha: str) -> None:
        """Stop a child whose wrapper died while its inherited lease proves it is original."""
        child_lock = attempt_dir / ".child.lock"
        if not child_lock.exists() or not _lock_is_held(child_lock):
            return
        identity_path = attempt_dir / "child-identity.json"
        deadline = time.monotonic() + 2
        while not identity_path.exists() and time.monotonic() < deadline:
            if not _lock_is_held(child_lock):
                return
            time.sleep(0.02)
        if not identity_path.exists():
            raise CampaignError(
                "campaign child lease is held without an authenticated identity; "
                "refusing to launch a duplicate"
            )
        identity = self._child_identity(identity_path, request_sha)
        child_pid = cast(int, identity["child_pid"])
        if _observed_process_create_time_ns(child_pid) is not None:
            _require_lease_holder(child_lock, child_pid, "campaign child")
            _terminate_pid_group(
                child_pid,
                cast(int, identity["process_create_time_ns"]),
            )
        _terminate_escaped_lease_holders(child_lock)
        deadline = time.monotonic() + 10
        while _lock_is_held(child_lock) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _lock_is_held(child_lock):
            raise CampaignError(
                "campaign child did not release its lease after termination; "
                "refusing to launch a duplicate"
            )

    def _observe_running(self, stage: StageSpec, attempt: int) -> None:
        attempt_dir = self._attempt_dir(stage, attempt)
        self._announce_stage(stage, attempt)
        request_path = attempt_dir / "worker-request.json"
        launching = next(
            event
            for event in reversed(self.store.events())
            if event["kind"] == "stage_launching"
            and event["stage_id"] == stage.id
            and event["attempt"] == attempt
        )
        if not request_path.is_file():
            worker_token = _load_worker_token(attempt_dir / "worker-token", consume=False)
            request = _worker_request(
                self.store,
                stage,
                attempt,
                attempt_dir,
                _digest_bytes(worker_token.encode("ascii")),
            )
            _write_new_json(request_path, request)
        if _digest_path(request_path) != launching["worker_request_sha256"]:
            raise CampaignError("worker request differs from its launch event")
        request_sha = _digest_path(request_path)
        request = _strict_json(request_path, "campaign worker request")
        _validate_worker_request(request, request_path)
        stale_after = max(30, 3 * cast(int, request["heartbeat_seconds"]))
        result_path = attempt_dir / "stage-result.json"
        identity_path = attempt_dir / "worker-identity.json"
        worker_lock_path = attempt_dir / ".worker.lock"
        launched: subprocess.Popen[bytes] | None = None
        if not identity_path.exists() and not result_path.exists():
            wrapper_active = worker_lock_path.exists() and _lock_is_held(worker_lock_path)
            if not wrapper_active:
                if not (attempt_dir / "worker-token").is_file():
                    self.store.append(
                        "stage_failed",
                        stage_id=stage.id,
                        attempt=attempt,
                        worker_result_sha256=None,
                        reason="worker wrapper exited before publishing identity",
                    )
                    return
                launched = self._launch_worker(request_path)
        while True:
            self._heartbeat(self.store.state())
            self._tail_live_output(attempt_dir)
            result_ready = result_path.exists() and not (
                worker_lock_path.exists() and _lock_is_held(worker_lock_path)
            )
            if result_ready:
                result, result_sha = _validate_worker_result(
                    result_path, request_path, stage, attempt
                )
                # The worker can publish its result immediately after its final log
                # line. Relay once more after the wrapper lock is gone so terminal
                # users do not lose completion metrics such as the WRMSSE summary.
                self._tail_live_output(attempt_dir)
                if result["plan_sha256"] != self.store.plan.source_sha256:
                    raise CampaignError("worker result plan SHA-256 differs from campaign")
                if result["status"] == "completed":
                    self.store.append(
                        "stage_completed",
                        stage_id=stage.id,
                        attempt=attempt,
                        worker_result_sha256=result_sha,
                    )
                else:
                    self.store.append(
                        "stage_failed",
                        stage_id=stage.id,
                        attempt=attempt,
                        worker_result_sha256=result_sha,
                        reason=result["reason"],
                    )
                if self._live_progress_enabled():
                    index, total, title, substage = self._display_stage(stage)
                    if result["status"] == "completed":
                        grouped = _workflow_stage(self.store.plan.stages, stage)
                        if grouped is not None and not substage and stage.id != "final_fit":
                            phase_index, phase_total = (
                                (6, 7)
                                if stage.id == self.store.plan.stages[0].id
                                else (2, 3)
                            )
                            message = (
                                f"Stage {index}/{total} | Phase {phase_index}/{phase_total} — "
                                f"COMPLETED: {title}"
                            )
                        else:
                            message = _stage_banner(
                                f"Stage {index}/{total}: {title} — COMPLETED"
                            )
                        print(message, file=sys.stderr, flush=True)
                    else:
                        print(
                            _stage_banner(
                                f"Stage {index}/{total}: {title} — FAILED: "
                                f"{result['reason']} | details: {result_path}"
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                return
            if identity_path.exists():
                identity = self._worker_identity(identity_path, request_sha)
                wrapper_active = worker_lock_path.exists() and _lock_is_held(worker_lock_path)
                if not wrapper_active:
                    time.sleep(min(0.1, self.poll_seconds))
                    if result_path.exists():
                        continue
                    self._terminate_stranded_child(attempt_dir, request_sha)
                    self.store.append(
                        "stage_failed",
                        stage_id=stage.id,
                        attempt=attempt,
                        worker_result_sha256=None,
                        reason="worker wrapper exited without a terminal result",
                    )
                    return
                heartbeat_path = attempt_dir / "heartbeat.json"
                heartbeat_time = cast(int, identity["started_at_ns"])
                if heartbeat_path.exists():
                    heartbeat = _strict_json(heartbeat_path, "campaign worker heartbeat")
                    expected_heartbeat = {
                        "schema",
                        "worker_request_sha256",
                        "wrapper_pid",
                        "child_pid",
                        "phase",
                        "rss_bytes",
                        "time_ns",
                    }
                    if (
                        set(heartbeat) != expected_heartbeat
                        or heartbeat.get("schema") != WORKER_HEARTBEAT_SCHEMA
                        or heartbeat.get("worker_request_sha256") != request_sha
                        or heartbeat.get("wrapper_pid") != identity["wrapper_pid"]
                        or type(heartbeat.get("phase")) is not str
                        or heartbeat.get("phase") not in {"waiting_for_device", "running"}
                        or type(heartbeat.get("rss_bytes")) is not int
                        or cast(int, heartbeat.get("rss_bytes")) < 0
                        or (
                            heartbeat.get("phase") == "waiting_for_device"
                            and heartbeat.get("child_pid") is not None
                        )
                        or (
                            heartbeat.get("phase") == "running"
                            and (
                                type(heartbeat.get("child_pid")) is not int
                                or cast(int, heartbeat.get("child_pid")) < 1
                            )
                        )
                        or type(heartbeat.get("time_ns")) is not int
                        or cast(int, heartbeat.get("time_ns")) < 1
                    ):
                        raise CampaignError("campaign worker heartbeat is invalid")
                    heartbeat_time = cast(int, heartbeat["time_ns"])
                heartbeat_age = (time.time_ns() - heartbeat_time) / 1e9
                if heartbeat_age > stale_after:
                    self._terminate_stranded_child(attempt_dir, request_sha)
                    _require_lease_holder(
                        worker_lock_path,
                        cast(int, identity["wrapper_pid"]),
                        "campaign worker",
                    )
                    _terminate_pid_group(
                        cast(int, identity["wrapper_pid"]),
                        cast(int, identity["process_create_time_ns"]),
                    )
                    deadline = time.monotonic() + 10
                    while _lock_is_held(worker_lock_path) and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if _lock_is_held(worker_lock_path):
                        raise CampaignError(
                            f"stage {stage.id!r} stale wrapper did not release its lease"
                        )
                    if result_path.exists():
                        continue
                    self.store.append(
                        "stage_failed",
                        stage_id=stage.id,
                        attempt=attempt,
                        worker_result_sha256=None,
                        reason="worker heartbeat became stale before a terminal result",
                    )
                    return
            else:
                wrapper_active = worker_lock_path.exists() and _lock_is_held(worker_lock_path)
                if not wrapper_active and (launched is None or launched.poll() is not None):
                    time.sleep(min(0.1, self.poll_seconds))
                    if identity_path.exists() or result_path.exists():
                        continue
                    if worker_lock_path.exists() and _lock_is_held(worker_lock_path):
                        continue
                    status = "detached" if launched is None else str(launched.returncode)
                    self.store.append(
                        "stage_failed",
                        stage_id=stage.id,
                        attempt=attempt,
                        worker_result_sha256=None,
                        reason=(
                            "worker wrapper exited before publishing identity " f"(status={status})"
                        ),
                    )
                    return
            time.sleep(self.poll_seconds)

    def run(self) -> dict[str, object]:
        """Run/recover until all stages complete or one exhausts its retry policy."""
        with _CampaignLease(self.store.root):
            while True:
                self.store._verify_plan_files()  # noqa: SLF001 - supervisor owns this store
                state = self.store.state()
                self._heartbeat(state)
                self.store.verify_completed_outputs()
                status = state["status"]
                if status == "completed":
                    self._heartbeat(state, force=True)
                    self._publish_terminal_notification(state)
                    return state
                stage_states = {
                    item["id"]: item for item in cast(list[dict[str, object]], state["stages"])
                }
                running = next(
                    (
                        stage
                        for stage in self.store.plan.stages
                        if stage_states[stage.id]["status"] == "running"
                    ),
                    None,
                )
                if running is not None:
                    self._observe_running(
                        running,
                        cast(int, stage_states[running.id]["active_attempt"]),
                    )
                    continue
                exhausted = [
                    stage.id
                    for stage in self.store.plan.stages
                    if stage_states[stage.id]["status"] == "failed"
                ]
                if exhausted:
                    events = self.store.events()
                    if events[-1]["kind"] == "campaign_failed":
                        self._heartbeat(state, force=True)
                        self._publish_terminal_notification(state)
                        return state
                    state = self.store.append("campaign_failed", failed_stage_ids=exhausted)
                    self._heartbeat(state, force=True)
                    self._publish_terminal_notification(state)
                    return state
                if all(
                    stage_states[stage.id]["status"] == "completed"
                    for stage in self.store.plan.stages
                ):
                    state = self.store.append("campaign_completed")
                    self._heartbeat(state, force=True)
                    self._publish_terminal_notification(state)
                    return state
                ready = next(
                    (
                        stage
                        for stage in self.store.plan.stages
                        if stage_states[stage.id]["status"] == "planned"
                        and all(
                            stage_states[dependency]["status"] == "completed"
                            for dependency in stage.needs
                        )
                    ),
                    None,
                )
                if ready is None:
                    raise CampaignError(
                        "campaign has no running, terminal, or dependency-ready stage"
                    )
                _admit(ready, self.store.root)
                attempt = cast(int, stage_states[ready.id]["attempts"]) + 1
                self._prepare_attempt(ready, attempt)
                self._observe_running(ready, attempt)
