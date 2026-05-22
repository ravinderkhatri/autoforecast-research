"""Durable development-run transactions around native PyTorch state dictionaries.

No model imports or training mathematics. See test_training_checkpoint.py for the
lease, publication, corruption, recovery, retention, runtime and RNG boundaries.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import pathlib
import platform
import random
import socket
import sys
import tempfile
import time
import uuid

import numpy as np
import torch

from autoforecast._exact_json import load_exact_json, loads_exact_json
from autoforecast.protocol import _atomic_write_text, _ensure_directory_durable, _fsync_directory


def configure_training_runtime(device: str, threads: int) -> dict:
    """Bind determinism controls, not a claim of cross-hardware reproducibility."""
    if device not in {"cpu", "cuda"} or type(threads) is not int or threads < 1:
        raise ValueError("invalid training runtime device/threads")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA explicitly requested but unavailable")
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError("CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return training_runtime(device)


def training_runtime(device: str) -> dict:
    """Capture controls whose changes invalidate exact continuation/replay."""
    return dict(
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        device=device,
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        default_dtype=str(torch.get_default_dtype()),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        mkldnn=torch.backends.mkldnn.enabled,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cuda_version=torch.version.cuda,
        cuda_current_device=torch.cuda.current_device() if device == "cuda" else None,
        cuda_devices=[
            dict(
                name=torch.cuda.get_device_name(i),
                capability=list(torch.cuda.get_device_capability(i)),
                memory=torch.cuda.get_device_properties(i).total_memory,
                uuid=str(torch.cuda.get_device_properties(i).uuid),
            )
            for i in range(torch.cuda.device_count())
        ]
        if device == "cuda"
        else [],
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None,
    )


def capture_training_rng(generator: np.random.Generator) -> dict:
    """Use only weights-only-loadable primitives/tensors, including NumPy's legacy RNG."""
    legacy = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy_legacy=(legacy[0], torch.tensor(legacy[1].astype(np.int64)), *legacy[2:]),
        generator=generator.bit_generator.state,
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    )


def restore_training_rng(state: dict, generator: np.random.Generator, *, device="cpu") -> None:
    """Restore LAST, after rebuilding model/optimizer and deterministic feature caches."""
    if not isinstance(state, dict) or set(state) != {
        "python",
        "numpy_legacy",
        "generator",
        "torch",
        "cuda",
    }:
        raise ValueError("invalid RNG checkpoint schema")
    cuda_states = state["cuda"]
    device_count = torch.cuda.device_count() if device == "cuda" or cuda_states else 0
    if (
        not isinstance(cuda_states, list)
        or len(cuda_states) != device_count
        or device == "cuda"
        and device_count < 1
    ):
        raise ValueError("CUDA RNG device count changed or required states are missing")

    previous = capture_training_rng(generator)

    def apply(candidate) -> None:
        legacy = candidate["numpy_legacy"]
        random.setstate(candidate["python"])
        np.random.set_state((legacy[0], legacy[1].numpy().astype(np.uint32), *legacy[2:]))
        generator.bit_generator.state = candidate["generator"]
        torch.set_rng_state(candidate["torch"])
        if candidate["cuda"]:
            torch.cuda.set_rng_state_all(candidate["cuda"])

    try:
        legacy = state["numpy_legacy"]
        python_probe = random.Random()
        python_probe.setstate(state["python"])
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state((legacy[0], legacy[1].numpy().astype(np.uint32), *legacy[2:]))
        generator_probe = np.random.Generator(type(generator.bit_generator)())
        generator_probe.bit_generator.state = state["generator"]
        torch.Generator().set_state(state["torch"])
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.device.type != "cpu"
            or value.ndim != 1
            for value in cuda_states
        ):
            raise TypeError("invalid CUDA RNG tensor")
        apply(state)
    except (
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        IndexError,
        AttributeError,
        OverflowError,
    ) as exc:
        try:
            apply(previous)
        except BaseException as rollback_error:
            exc.add_note(
                "failed to restore the live RNG state after rejecting a checkpoint: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )
        raise ValueError("invalid RNG checkpoint state") from exc


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _journal_record(previous, record, *, initial_phase="selection"):
    """The same contiguous phase/update contract for appending and accepting a saved prefix."""
    if (
        initial_phase not in {"selection", "refit"}
        or not isinstance(record, dict)
        or record.get("phase") not in {"selection", "refit"}
        or type(record.get("update")) is not int
        or record["update"] < 1
    ):
        raise ValueError("training log requires phase and positive integer update")
    current = (record["phase"], record["update"])
    if previous:
        expected = (previous[0], previous[1] + 1)
        transition = previous[0] == "selection" and current == ("refit", 1)
        if current != expected and not transition:
            raise ValueError("training log update order is not contiguous")
    elif current != (initial_phase, 1):
        raise ValueError(f"training log must start at {initial_phase} update one")
    return current


class TrainingRun:
    """One exclusive writer, immutable generations, atomic latest pointer and log frontier.

    Caller saves ONLY at safe logical-update boundaries. A fresh or resumed run holds
    its lease through final export. ``model.pt`` is intentionally not accepted here.
    """

    def __init__(
        self,
        root,
        binding,
        *,
        device="cpu",
        resume=None,
        source_check=None,
        every_updates=100,
        every_seconds=300.0,
        keep_last=2,
    ):
        # The caller selects this namespace; normal filesystem/root aliases are supported.
        # Refuse aliases/special members *inside* it, and bind the canonical output identity.
        self.root = pathlib.Path(root).resolve()
        self.binding = _json(binding)
        contract = binding.get("training_contract") if isinstance(binding, dict) else None
        self.initial_phase = (
            "refit"
            if isinstance(contract, dict) and contract.get("mode") == "fixed_refit"
            else "selection"
        )
        self.device = device
        self.runtime = training_runtime(device)
        self.resume = pathlib.Path(resume).resolve() if resume else None
        if self.resume is not None and self.resume != self.root / "training_resume.json":
            raise ValueError("resume requires this output's latest training_resume.json pointer")
        if (
            type(every_updates) is not int
            or every_updates < 1
            or type(keep_last) is not int
            or keep_last < 2
            or type(every_seconds) not in (int, float)
            or not np.isfinite(every_seconds)
            or every_seconds <= 0
        ):
            raise ValueError("invalid checkpoint cadence/retention")
        self.source_check = source_check or (lambda: None)
        self.every_updates, self.every_seconds, self.keep_last = (
            every_updates,
            every_seconds,
            keep_last,
        )
        self.session = uuid.uuid4().hex
        self.lease = self.log_file = None
        self.generation = 0
        self.last_saved_update = 0
        self.last_saved_time = time.monotonic()
        self.last_record = None
        self.records = 0
        self.log_hash = hashlib.sha256()
        self.loaded = None
        self._pending_recovery = None

    def _path(self, relative, *, directory=False):
        """Refuse linked/special managed paths before any recovery mutation or open."""
        path = self.root / relative
        if (
            not path.is_relative_to(self.root)
            or path.resolve() != path
            or path.exists()
            and (
                not (path.is_dir() if directory else path.is_file())
                or not directory
                and path.stat().st_nlink != 1
            )
        ):
            raise ValueError(f"unsafe linked or special run path: {relative}")
        return path

    def _files(self, directory: str, pattern: str):
        """Validate each managed member, not just its directory, before any reader/open."""
        return [
            self._path(path.relative_to(self.root))
            for path in sorted(self._path(directory, directory=True).glob(pattern))
        ]

    def __enter__(self):
        if not self.resume:
            _ensure_directory_durable(self.root.parent)
            self.root.mkdir(exist_ok=False)
            _fsync_directory(self.root.parent)
        elif not self.root.is_dir():
            raise ValueError("resume output directory is missing")
        for name in ("checkpoints", "lifecycle", "recovery"):
            self._path(name, directory=True)
        for name in (
            "training.jsonl",
            "training_resume.json",
            "COMPLETE.json",
            "model.pt",
            "forecast.parquet",
        ):
            self._path(name)
        for name in ("checkpoints", "lifecycle"):
            self._files(name, "*")
        self.lease = self._path(".training.lock").open("a+b")
        try:
            try:
                fcntl.flock(self.lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("training output already has an active writer") from exc
            if (self.root / "COMPLETE.json").exists():
                raise ValueError("completed runs cannot be resumed or overwritten")
            _ensure_directory_durable(self.root / "checkpoints")
            _ensure_directory_durable(self.root / "lifecycle")
            self.source_check()
            if self.resume:
                self.loaded = self.load()
            else:
                self.log_file = self._path("training.jsonl").open("xb")
            previous = self._files("lifecycle", "*.json")
            open_sessions = set()
            for path in previous:
                event = load_exact_json(path)
                if event["kind"] == "session_start":
                    open_sessions.add(event["session"])
                if event["kind"] == "session_end":
                    open_sessions.discard(event["session"])
            self.event(
                "session_start",
                resumed=bool(self.resume),
                previous_sessions_without_end=sorted(open_sessions),
            )
            return self
        except BaseException as failure:
            self._close_handles(failure)
            raise

    def _close_handles(self, failure: BaseException | None) -> BaseException | None:
        """Attempt every owned close, retaining the primary error and secondary diagnostics."""
        for label, handle in (("log_file", self.log_file), ("lease", self.lease)):
            if handle is None:
                continue
            try:
                handle.close()
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                else:
                    failure.add_note(f"failed to close training {label}: {close_error}")
        return failure

    def __exit__(self, typ, value, traceback):
        failure = value
        try:
            try:
                self.event(
                    "session_end",
                    status="exception" if typ else "returned",
                    exception_type=typ.__name__ if typ else None,
                )
            except BaseException as event_error:
                if failure is None:
                    failure = event_error
                else:
                    failure.add_note(
                        "failed to write the final training lifecycle event: "
                        f"{type(event_error).__name__}: {event_error}"
                    )
        finally:
            failure = self._close_handles(failure)
        if value is None and failure is not None:
            raise failure

    def event(self, kind: str, **fields) -> None:
        """Atomic immutable lifecycle records; no claim about an unobserved kill cause."""
        previous = self._files("lifecycle", "*.json")
        sequence = int(previous[-1].stem) + 1 if previous else 1
        record = dict(
            kind=kind,
            sequence=sequence,
            session=self.session,
            pid=os.getpid(),
            host=socket.gethostname(),
            time_ns=time.time_ns(),
            generation=self.generation,
            previous_sha256=_digest(previous[-1].read_bytes()) if previous else None,
            **fields,
        )
        _atomic_write_text(self._path(f"lifecycle/{sequence:08d}.json"), _json(record) + "\n")

    def log(self, record: dict) -> None:
        """Append one completed logical update; checkpoint later commits its frontier."""
        current = _journal_record(self.last_record, record, initial_phase=self.initial_phase)
        payload = (_json(record) + "\n").encode()
        self.log_file.write(payload)
        self.log_file.flush()
        self.log_hash.update(payload)
        self.records += 1
        self.last_record = current

    def due(self, update: int) -> bool:
        """Wall cadence is observed at safe boundaries, never mid-backward."""
        return (
            update - self.last_saved_update >= self.every_updates
            or time.monotonic() - self.last_saved_time >= self.every_seconds
        )

    def save(self, state: dict, *, reason: str) -> None:
        """Durably publish payload -> manifest -> pointer; only then prune old payloads."""
        started = time.perf_counter()
        self.source_check()
        if training_runtime(self.device) != self.runtime:
            raise ValueError("training runtime changed before checkpoint")
        self.log_file.flush()
        os.fsync(self.log_file.fileno())
        existing = sorted(
            path
            for path in self._files("checkpoints", "*")
            if path.suffix in {".json", ".pt"} and path.stem.isdigit()
        )
        generation = max(self.generation, int(existing[-1].stem) if existing else 0) + 1
        stem = f"{generation:08d}"
        destination = self._path(f"checkpoints/{stem}.pt")
        # Exclusive temp creation + os.replace retain the previous generation on any failure.
        fd, temporary = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(
                    dict(
                        schema="revision_training_state_v1",
                        state=state,
                        binding=self.binding,
                        runtime=self.runtime,
                        output=str(self.root),
                    ),
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            payload_hash = _digest(pathlib.Path(temporary).read_bytes())
            self.source_check()
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            pathlib.Path(temporary).unlink(missing_ok=True)
        manifest = dict(
            schema="revision_training_manifest_v1",
            generation=generation,
            previous_generation=self.generation,
            payload=destination.name,
            payload_sha256=payload_hash,
            output=str(self.root),
            reason=reason,
            phase=state["phase"],
            update=state["update"],
            log_bytes=self.log_file.tell(),
            log_sha256=self.log_hash.hexdigest(),
            records=self.records,
            last_record=self.last_record,
        )
        manifest_path = destination.with_suffix(".json")
        _atomic_write_text(manifest_path, _json(manifest) + "\n")
        pointer = dict(
            schema="revision_training_pointer_v1",
            generation=generation,
            manifest=manifest_path.name,
            manifest_sha256=_digest(manifest_path.read_bytes()),
        )
        _atomic_write_text(self._path("training_resume.json"), _json(pointer) + "\n")
        self.generation = generation
        self.last_saved_update = state["update"]
        self.last_saved_time = time.monotonic()
        self.event(
            "checkpoint",
            reason=reason,
            phase=state["phase"],
            update=state["update"],
            seconds=time.perf_counter() - started,
            timing_scope="serialization_and_publication_before_retention",
            payload_sha256=payload_hash,
        )
        # Only files named by this run's durable manifests are candidates for retirement.
        manifests = self._files("checkpoints", "*.json")
        retained, cursor = set(), generation
        for _ in range(self.keep_last):
            if not cursor:
                break
            retained.add(cursor)
            cursor = load_exact_json(self._path(f"checkpoints/{cursor:08d}.json"))[
                "previous_generation"
            ]
        # Uncommitted manifests cannot displace the previous committed checkpoint.
        for old in manifests:
            if int(old.stem) in retained:
                continue
            old_payload = self._path(old.with_suffix(".pt").relative_to(self.root))
            if old_payload.exists():
                old_payload.unlink()
                self.event("payload_retired", manifest=old.name)
        _fsync_directory(destination.parent)

    def load(self) -> dict:
        """Read and verify checkpoint bytes; defer destructive recovery until trainer acceptance."""
        pointer = load_exact_json(self.resume)
        if (
            not isinstance(pointer, dict)
            or set(pointer) != {"schema", "generation", "manifest", "manifest_sha256"}
            or pointer["schema"] != "revision_training_pointer_v1"
            or type(pointer["generation"]) is not int
            or pointer["generation"] < 1
            or pointer["manifest"] != f"{pointer['generation']:08d}.json"
        ):
            raise ValueError("invalid latest checkpoint pointer schema/path")
        manifest_path = self._path(f"checkpoints/{pointer['manifest']}")
        raw = manifest_path.read_bytes()
        if _digest(raw) != pointer["manifest_sha256"]:
            raise ValueError("checkpoint manifest digest mismatch")
        manifest = loads_exact_json(raw)
        if (
            manifest.get("schema") != "revision_training_manifest_v1"
            or manifest.get("generation") != pointer["generation"]
            or type(manifest.get("previous_generation")) is not int
            or not 0 <= manifest["previous_generation"] < pointer["generation"]
            or manifest.get("payload") != f"{pointer['generation']:08d}.pt"
            or manifest.get("output") != str(self.root)
            or type(manifest.get("log_bytes")) is not int
            or manifest["log_bytes"] < 0
            or type(manifest.get("records")) is not int
            or manifest["records"] < 0
            or manifest.get("phase") not in {"selection", "refit"}
            or type(manifest.get("update")) is not int
            or manifest["update"] < 0
        ):
            raise ValueError("invalid checkpoint manifest schema/path/frontier")
        payload_path = self._path(f"checkpoints/{manifest['payload']}")
        committed = [load_exact_json(p) for p in self._files("lifecycle", "*.json")]
        if any(
            event["kind"] == "checkpoint" and event["generation"] > pointer["generation"]
            for event in committed
        ):
            raise ValueError("stale checkpoint pointer would roll back a committed generation")
        payload = payload_path.read_bytes()
        if _digest(payload) != manifest.get("payload_sha256"):
            raise ValueError("checkpoint payload digest mismatch")
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if (
            not isinstance(saved, dict)
            or set(saved) != {"schema", "state", "binding", "runtime", "output"}
            or saved["schema"] != "revision_training_state_v1"
            or saved["binding"] != self.binding
            or saved["runtime"] != self.runtime
            or saved["output"] != str(self.root)
        ):
            raise ValueError("checkpoint source/config/input/runtime/output binding mismatch")
        if saved["state"].get("phase") != manifest.get("phase") or saved["state"].get(
            "update"
        ) != manifest.get("update"):
            raise ValueError("checkpoint state and manifest counters differ")
        path = self._path("training.jsonl")
        log = path.read_bytes()
        prefix, suffix = log[: manifest["log_bytes"]], log[manifest["log_bytes"] :]
        lines = prefix.splitlines()
        last = json.loads(lines[-1]) if lines else None
        if (
            len(prefix) != manifest["log_bytes"]
            or _digest(prefix) != manifest.get("log_sha256")
            or len(lines) != manifest["records"]
            or ([last["phase"], last["update"]] if last else None) != manifest["last_record"]
        ):
            raise ValueError("checkpoint training log prefix/frontier mismatch")
        frontier = None
        for line in lines:
            frontier = _journal_record(
                frontier, loads_exact_json(line), initial_phase=self.initial_phase
            )
        phase, update = manifest["phase"], manifest["update"]
        completed = update > 0 and frontier == (phase, update)
        initial_empty = update == 0 and phase == self.initial_phase and frontier is None
        research_transition = (
            update == 0
            and self.initial_phase == "selection"
            and phase == "refit"
            and frontier is not None
            and frontier[0] == "selection"
        )
        if not (completed or initial_empty or research_transition):
            raise ValueError("checkpoint trainer counters differ from committed log frontier")
        self.generation = pointer["generation"]
        self._pending_recovery = (manifest, prefix, suffix)
        self.log_hash.update(prefix)
        self.records = manifest["records"]
        self.last_record = tuple(manifest["last_record"]) if last else None
        self.last_saved_update = manifest["update"]
        return saved["state"]

    def _commit_recovery(self) -> None:
        """Called only after model, optimizer, feature admission and RNG state are accepted.

        Failed preflight may append lifecycle diagnostics, but never rewinds the existing log
        or moves partial exports. The lease remains held across both stages.
        """
        if self._pending_recovery is None:
            return
        manifest, prefix, suffix = self._pending_recovery
        path = self._path("training.jsonl")
        if path.read_bytes() != prefix + suffix:
            raise ValueError("training log changed during checkpoint preflight")
        artifacts = [self._path(name) for name in ("model.pt", "forecast.parquet")]
        self.source_check()
        if suffix:
            self._archive_bytes("training.uncommitted.jsonl", suffix)
            with path.open("r+b") as stream:
                stream.truncate(manifest["log_bytes"])
                stream.flush()
                os.fsync(stream.fileno())
        self.log_file = path.open("ab")
        self.event(
            "resume_loaded",
            payload_sha256=manifest["payload_sha256"],
            archived_log_bytes=len(suffix),
        )
        for artifact in artifacts:
            if artifact.exists():
                self._archive_bytes(artifact.name, artifact.read_bytes())
                artifact.unlink()
                _fsync_directory(self.root)
        self._pending_recovery = None

    def _archive_bytes(self, name: str, payload: bytes) -> None:
        directory = self._path(f"recovery/{self.session}", directory=True)
        _ensure_directory_durable(directory)
        path = self._path(f"recovery/{self.session}/{name}")
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
        _fsync_directory(self.root)
        self.event(
            "artifact_archived",
            path=str(path.relative_to(self.root)),
            bytes=len(payload),
            sha256=_digest(payload),
        )
