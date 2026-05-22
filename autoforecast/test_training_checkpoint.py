"""Structural fault tests, not forecasting/accuracy evidence; real training tests are separate."""

import hashlib
import io
import json
import os
import random
import select
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autoforecast import training_checkpoint as checkpoint
from autoforecast.training_checkpoint import (
    TrainingRun,
    capture_training_rng,
    configure_training_runtime,
    restore_training_rng,
    training_runtime,
)
from autoforecast.training_progress import PipelineProgress, TrainingProgress


def state(update=0):
    return dict(
        phase="selection",
        update=update,
        weights=torch.tensor([2.5]),
        rng=capture_training_rng(np.random.default_rng(4)),
    )


def test_revision_runtime(monkeypatch):
    first = configure_training_runtime("cpu", 1)
    assert first == training_runtime("cpu")
    assert first["deterministic"] and not first["deterministic_warn_only"]
    for device, threads in (("mps", 1), ("cpu", True), ("cpu", 0)):
        with pytest.raises(ValueError, match="runtime"):
            configure_training_runtime(device, threads)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        configure_training_runtime("cuda", 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS"):
        configure_training_runtime("cuda", 1)
    torch.set_default_dtype(torch.float64)
    try:
        assert training_runtime("cpu") != first
    finally:
        torch.set_default_dtype(torch.float32)


def test_training_rng(monkeypatch):
    generator = np.random.default_rng(21)
    saved = capture_training_rng(generator)
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    expected = (random.random(), np.random.random(), generator.random(), torch.rand(4))
    restore_training_rng(saved, generator)
    actual = (random.random(), np.random.random(), generator.random(), torch.rand(4))
    assert expected[:3] == actual[:3]
    assert torch.equal(expected[3], actual[3])
    with pytest.raises(ValueError, match="RNG"):
        restore_training_rng({}, generator)
    broken = dict(saved, torch=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="RNG"):
        restore_training_rng(broken, generator)
    with pytest.raises(ValueError, match="RNG checkpoint schema"):
        restore_training_rng(dict(saved, extra=True), generator)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: None)
    with pytest.raises(ValueError, match="CUDA RNG device count"):
        restore_training_rng(dict(saved, cuda=[saved["torch"], saved["torch"]]), generator)
    with pytest.raises(ValueError, match="required states are missing"):
        restore_training_rng(dict(saved, cuda=[]), generator, device="cuda")


def test_failed_rng_restore_is_transactional() -> None:
    """A rejected checkpoint must not partially replace the process's live RNG streams."""
    generator = np.random.default_rng(22)
    foreign = capture_training_rng(np.random.default_rng(91))
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    generator_before = generator.bit_generator.state
    torch_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="RNG"):
        restore_training_rng(dict(foreign, torch=torch.tensor([1.0])), generator)

    numpy_after = np.random.get_state()
    assert random.getstate() == python_before
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert generator.bit_generator.state == generator_before
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_training_cuda_device_binding(tmp_path, monkeypatch):
    """Structural inventory only: a different selected ordinal/physical UUID cannot resume."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "same-model")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(total_memory=100, uuid=f"gpu-{i}"),
    )
    first = training_runtime("cuda")
    root = tmp_path / "run"
    with TrainingRun(root, {}, device="cuda") as run:
        run.save(state(), reason="initial")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    second = training_runtime("cuda")
    assert first["cuda_devices"] == second["cuda_devices"]
    assert first["cuda_current_device"] == 0 and second["cuda_current_device"] == 1
    with pytest.raises(ValueError, match="binding mismatch"):
        with TrainingRun(root, {}, device="cuda", resume=root / "training_resume.json"):
            pass
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(total_memory=100, uuid=f"other-{i}"),
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        with TrainingRun(root, {}, device="cuda", resume=root / "training_resume.json"):
            pass


def test_training_run_lease(tmp_path):
    root = tmp_path / "run"
    with TrainingRun(root, {}) as run:
        run.save(state(), reason="initial")
        with pytest.raises(ValueError, match="active writer"):
            with TrainingRun(root, {}, resume=root / "training_resume.json"):
                pass
        with pytest.raises(FileExistsError):
            with TrainingRun(root, {}):
                pass
    with pytest.raises(RuntimeError, match="interrupted"):
        with TrainingRun(root, {}, resume=root / "training_resume.json"):
            raise RuntimeError("interrupted")
    events = [json.loads(p.read_text()) for p in sorted((root / "lifecycle").glob("*.json"))]
    assert events[-1]["exception_type"] == "RuntimeError"
    assert events[-1]["status"] == "exception"
    (root / "COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="completed"):
        with TrainingRun(root, {}, resume=root / "training_resume.json"):
            pass
    with pytest.raises(ValueError, match="missing"):
        with TrainingRun(
            tmp_path / "missing", {}, resume=tmp_path / "missing/training_resume.json"
        ):
            pass


def test_training_run_exit_preserves_body_exception_when_lifecycle_write_fails(
    tmp_path, monkeypatch
):
    """Cleanup diagnostics must not replace the exception that interrupted training."""
    with pytest.raises(RuntimeError, match="training failed") as caught:
        with TrainingRun(tmp_path / "failed-run", {}) as run:
            monkeypatch.setattr(
                run,
                "event",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("lifecycle full")),
            )
            raise RuntimeError("training failed")
    assert any("lifecycle full" in note for note in getattr(caught.value, "__notes__", ()))
    assert run.log_file.closed and run.lease.closed

    with pytest.raises(OSError, match="lifecycle full"):
        with TrainingRun(tmp_path / "returned-run", {}) as run:
            monkeypatch.setattr(
                run,
                "event",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("lifecycle full")),
            )
    assert run.log_file.closed and run.lease.closed


@pytest.mark.parametrize("stage", ["enter", "body", "returned"])
@pytest.mark.parametrize("broken", [("log_file",), ("lease",), ("log_file", "lease")])
def test_training_run_closes_every_handle_without_masking_failure(tmp_path, stage, broken):
    """Every owned handle is attempted, including when an earlier close raises."""

    class FailedClose:
        def __init__(self, real, label):
            self.real, self.label = real, label

        @property
        def closed(self):
            return self.real.closed

        def close(self):
            self.real.close()
            raise OSError(f"{self.label} close failed")

    run = TrainingRun(tmp_path / "run", {})

    def break_handles():
        for label in broken:
            setattr(run, label, FailedClose(getattr(run, label), label))

    if stage == "enter":

        def fail_entry(*args, **kwargs):
            break_handles()
            raise RuntimeError("entry failed")

        run.event = fail_entry

    expected = OSError if stage == "returned" else RuntimeError
    message = (
        "close failed"
        if stage == "returned"
        else f"{'entry' if stage == 'enter' else stage} failed"
    )
    try:
        with pytest.raises(expected, match=message) as caught:
            with run:
                break_handles()
                if stage == "body":
                    raise RuntimeError("body failed")
        assert run.log_file.closed and run.lease.closed
        notes = " ".join(getattr(caught.value, "__notes__", ()))
        for label in broken:
            assert f"{label} close failed" in str(caught.value) + notes
    finally:
        for handle in (run.log_file, run.lease):
            if isinstance(handle, FailedClose):
                handle.real.close()
            elif handle is not None:
                handle.close()


@pytest.mark.parametrize(
    "target",
    [
        "training.jsonl",
        "checkpoints",
        "lifecycle",
        "recovery",
        ".training.lock",
        "hardlink",
        "fifo",
    ],
)
def test_training_run_paths(tmp_path, target):
    root = tmp_path / "run"
    with TrainingRun(root, {}) as run:
        run.log(dict(phase="selection", update=1, loss=1.0))
        run.save(state(1), reason="update")
    name = "training.jsonl" if target in {"hardlink", "fifo"} else target
    path = root / name
    outside = tmp_path / "external"
    if name in {"checkpoints", "lifecycle", "recovery"}:
        if path.exists():
            path.rename(outside)
        else:
            outside.mkdir()
    else:
        original = path.read_bytes()
        path.rename(outside)
        outside.write_bytes(original + b"uncommitted external suffix\n")
    if target == "hardlink":
        os.link(outside, path)
    elif target == "fifo":
        os.mkfifo(path)
    else:
        path.symlink_to(outside)
    expected = (
        {p.name: p.read_bytes() for p in outside.iterdir() if p.is_file()}
        if outside.is_dir()
        else outside.read_bytes()
    )
    with pytest.raises(ValueError, match="unsafe linked or special run path"):
        with TrainingRun(root, {}, resume=root / "training_resume.json"):
            pass
    actual = (
        {p.name: p.read_bytes() for p in outside.iterdir() if p.is_file()}
        if outside.is_dir()
        else outside.read_bytes()
    )
    assert actual == expected


def test_training_run_root_aliases(tmp_path):
    """Explicit canonical destinations are allowed; aliases cannot bypass output identity."""
    target = tmp_path / "selected-destination"
    target.mkdir()
    canary = target / "unrelated.txt"
    canary.write_bytes(b"unchanged")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(target, target_is_directory=True)
    with TrainingRun(parent_alias / "run", {}) as run:
        assert run.root == (target / "run").resolve()
        run.save(state(), reason="initial")
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(target / "run", target_is_directory=True)
    with TrainingRun(root_alias, {}, resume=root_alias / "training_resume.json") as run:
        assert run.loaded["update"] == 0
    other = tmp_path / "different-run"
    shutil.copytree(target / "run", other)
    wrong_alias = tmp_path / "wrong-alias"
    wrong_alias.symlink_to(other, target_is_directory=True)
    before = (other / "training.jsonl").read_bytes()
    with pytest.raises(ValueError, match="manifest schema/path/frontier"):
        with TrainingRun(wrong_alias, {}, resume=wrong_alias / "training_resume.json"):
            pass
    assert canary.read_bytes() == b"unchanged"
    assert (other / "training.jsonl").read_bytes() == before


def test_training_process_kill(tmp_path):
    """Actual SIGKILL/OS lease recovery, using a native structural checkpoint, not a new fit."""
    configure_training_runtime("cpu", 1)
    root = tmp_path / "run"
    script = """
import sys, time
from autoforecast.training_checkpoint import TrainingRun, configure_training_runtime
from autoforecast.test_training_checkpoint import state
configure_training_runtime("cpu", 1)
with TrainingRun(sys.argv[1], {}) as run:
    run.log(dict(phase="selection", update=1, loss=1.0))
    run.save(state(1), reason="update")
    run.log(dict(phase="selection", update=2, loss=2.0))
    print("READY", flush=True)
    time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([child.stdout], [], [], 20)[0], "Checkpoint worker never became ready"
        assert child.stdout.readline().strip() == "READY"
        with pytest.raises(ValueError, match="active writer"):
            with TrainingRun(root, {}, resume=root / "training_resume.json"):
                pass
        child.kill()
        assert child.wait(timeout=10) == -9
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        assert run.loaded["update"] == 1
        assert run.records == 1
        run._commit_recovery()
    events = [json.loads(p.read_text()) for p in sorted((root / "lifecycle").glob("*.json"))]
    starts = [e for e in events if e["kind"] == "session_start"]
    assert starts[-1]["previous_sessions_without_end"] == [starts[0]["session"]]
    assert any(
        p.read_bytes().find(b'"update":2') >= 0 for p in (root / "recovery").rglob("*.jsonl")
    )


def test_training_run_log(tmp_path):
    with TrainingRun(tmp_path / "run", {}) as run:
        for record in (
            {},
            {"phase": "selection", "update": True},
            {"phase": "refit", "update": 1},
            {"phase": "selection", "update": 2},
            {"phase": "selection", "update": 1, "loss": float("nan")},
        ):
            with pytest.raises(ValueError):
                run.log(record)
        run.log(dict(phase="selection", update=1, loss=2.0))
        for phase, update in (("selection", 1), ("selection", 3), ("refit", 2)):
            with pytest.raises(ValueError, match="contiguous"):
                run.log(dict(phase=phase, update=update))
        run.log(dict(phase="refit", update=1))
        with pytest.raises(ValueError, match="contiguous"):
            run.log(dict(phase="selection", update=2))
        assert run.records == 2


def test_fixed_refit_training_log_starts_at_refit_and_resumes(tmp_path):
    root = tmp_path / "fixed-refit"
    binding = {
        "training_contract": {
            "mode": "fixed_refit",
            "validation_origin": None,
            "fixed_update_budget": 2,
        }
    }
    state = {"phase": "refit", "update": 1, "weights": torch.tensor([1.0])}
    with TrainingRun(root, binding, every_updates=1) as run:
        with pytest.raises(ValueError, match="refit update one"):
            run.log({"phase": "selection", "update": 1})
        run.log({"phase": "refit", "update": 1, "loss": 1.0})
        run.save(state, reason="update")
    with TrainingRun(root, binding, resume=root / "training_resume.json") as run:
        assert run.loaded["phase"] == "refit"
        assert run.last_record == ("refit", 1)
        run._commit_recovery()


def test_training_checkpoint_save(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with TrainingRun(root, {"seed": 42}, every_updates=2, every_seconds=10) as run:
        assert not run.due(1) and run.due(2)
        run.last_saved_time -= 11
        assert run.due(0)
        run.save(state(), reason="initial")
        old = (root / "training_resume.json").read_bytes()
        original = checkpoint._atomic_write_text

        def fail_pointer(path, payload):
            if path.name == "training_resume.json":
                raise OSError("injected atomic publication failure")
            original(path, payload)

        monkeypatch.setattr(checkpoint, "_atomic_write_text", fail_pointer)
        with pytest.raises(OSError, match="publication"):
            run.save(state(), reason="failure")
        assert (root / "training_resume.json").read_bytes() == old
        assert (root / "checkpoints/00000001.pt").is_file()
        monkeypatch.setattr(checkpoint, "_atomic_write_text", original)
    with TrainingRun(root, {"seed": 42}, resume=root / "training_resume.json") as run:
        assert torch.equal(run.loaded["weights"], state()["weights"])
        run._commit_recovery()
        run.save(state(), reason="recovered")
        assert run.generation == 3  # Never overwrite an orphaned immutable generation.
        assert (root / "checkpoints/00000001.pt").exists()
        assert not (root / "checkpoints/00000002.pt").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "binding",
        "runtime",
        "pointer",
        "manifest",
        "payload",
        "prefix",
        "stale",
        "symlink",
        "manifest_symlink",
        "manifest_schema",
        "state_counter",
        "payload_counter",
        "short_frontier",
    ],
)
def test_training_checkpoint_load(tmp_path, fault):
    root = tmp_path / "run"
    with TrainingRun(root, {"seed": 42}) as run:
        run.save(state(), reason="initial")
        old = (root / "training_resume.json").read_bytes()
        run.log(dict(phase="selection", update=1, loss=1.0))
        run.save(state(1), reason="update")
    binding = {"seed": 9 if fault == "binding" else 42}
    pointer = root / "training_resume.json"
    if fault == "runtime":
        torch.set_num_threads(2 if torch.get_num_threads() == 1 else 1)
    if fault == "pointer":
        pointer.write_text('{"schema":"wrong"}')
    if fault == "manifest":
        path = root / "checkpoints/00000002.json"
        altered = json.loads(path.read_text())
        altered["reason"] = "changed without updating digest"
        path.write_text(json.dumps(altered))
    if fault == "payload":
        (root / "checkpoints/00000002.pt").write_bytes(b"truncated")
    if fault == "prefix":
        path = root / "training.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"loss":1.0', b'"loss":9.0'))
    if fault == "stale":
        pointer.write_bytes(old)
    if fault == "symlink":
        path = root / "checkpoints/00000002.pt"
        outside = tmp_path / "external.pt"
        path.rename(outside)
        path.symlink_to(outside)
    if fault == "manifest_symlink":
        path = root / "checkpoints/00000002.json"
        outside = tmp_path / "external.json"
        path.rename(outside)
        path.symlink_to(outside)
    if fault in {"manifest_schema", "state_counter", "payload_counter", "short_frontier"}:
        path = root / "checkpoints/00000002.json"
        altered = json.loads(path.read_text())
        if fault in {"manifest_schema", "state_counter"}:
            altered["schema" if fault == "manifest_schema" else "update"] = (
                "wrong" if fault == "manifest_schema" else 22
            )
        elif fault == "payload_counter":
            payload_path = root / "checkpoints/00000002.pt"
            payload = torch.load(payload_path, weights_only=True)
            payload["state"]["update"] = 2
            torch.save(payload, payload_path)
            altered["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        else:
            altered.update(
                log_bytes=0, records=0, last_record=None, log_sha256=hashlib.sha256(b"").hexdigest()
            )
        path.write_text(json.dumps(altered))
        changed_pointer = json.loads(pointer.read_text())
        changed_pointer["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        pointer.write_text(json.dumps(changed_pointer))
    messages = dict(
        binding="binding mismatch",
        runtime="binding mismatch",
        pointer="pointer schema",
        manifest="manifest digest",
        payload="payload digest",
        prefix="log prefix",
        stale="stale checkpoint",
        symlink="unsafe linked or special",
        manifest_symlink="unsafe linked or special",
        manifest_schema="manifest schema/path",
        state_counter="counters differ",
        payload_counter="counters differ",
        short_frontier="committed log frontier",
    )
    with pytest.raises(ValueError, match=messages[fault]):
        with TrainingRun(root, binding, resume=pointer):
            pass
    configure_training_runtime("cpu", 1)
    with pytest.raises(ValueError, match="latest"):
        TrainingRun(root, {}, resume=root / "model.pt")


@pytest.mark.parametrize(
    "member", ["checkpoints/00000002.json", "checkpoints/00000002.pt", "lifecycle/00000001.json"]
)
@pytest.mark.parametrize("kind", ["fifo", "hardlink"])
def test_checkpoint_member_paths(tmp_path, member, kind):
    """Bounded process makes a FIFO regression fail, not hang pytest or touch a live run."""
    configure_training_runtime("cpu", 1)
    root = tmp_path / "run"
    with TrainingRun(root, {}) as run:
        run.save(state(), reason="initial")
        run.log(dict(phase="selection", update=1, loss=1.0))
        run.save(state(1), reason="update")
        run.log(dict(phase="selection", update=2, loss=2.0))
    target, external = root / member, tmp_path / "external"
    target.rename(external)
    if kind == "fifo":
        os.mkfifo(target)
    else:
        os.link(external, target)
    before, log = external.read_bytes(), (root / "training.jsonl").read_bytes()
    script = """
import sys
from pathlib import Path
from autoforecast.training_checkpoint import TrainingRun, configure_training_runtime
configure_training_runtime("cpu", 1)
root = Path(sys.argv[1])
try:
    with TrainingRun(root, {}, resume=root / "training_resume.json"):
        pass
except ValueError as exc:
    assert "unsafe linked or special run path" in str(exc)
else:
    raise AssertionError("unsafe member was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert external.read_bytes() == before and (root / "training.jsonl").read_bytes() == log


def test_checkpoint_parent_durability(tmp_path, monkeypatch):
    """Directory-sync ordering, not a simulated claim of real power-loss survival."""
    from autoforecast import protocol

    calls = []
    native = protocol._fsync_directory

    def synced(path):
        calls.append(path.resolve())
        native(path)

    monkeypatch.setattr(protocol, "_fsync_directory", synced)
    monkeypatch.setattr(checkpoint, "_fsync_directory", synced)
    root = tmp_path / "first/second/run"
    with TrainingRun(root, {}) as run:
        run.save(state(), reason="initial")
    assert calls[:3] == [tmp_path.resolve(), (tmp_path / "first").resolve(), root.parent.resolve()]
    assert calls[3:5] == [root.resolve(), root.resolve()]
    with pytest.raises(FileExistsError):
        with TrainingRun(root, {}):
            pass


def test_training_checkpoint_recovery(tmp_path):
    root = tmp_path / "run"
    with TrainingRun(root, {}) as run:
        run.log(dict(phase="selection", update=1, loss=1.0))
        run.save(state(1), reason="update")
        prefix = (root / "training.jsonl").read_bytes()
        run.log(dict(phase="selection", update=2, loss=2.0))
        suffix = (root / "training.jsonl").read_bytes()[len(prefix) :]
    (root / "model.pt").write_bytes(b"partial export")
    (root / "forecast.parquet").write_bytes(b"partial parquet")
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        assert (root / "training.jsonl").read_bytes() == prefix + suffix
        assert (root / "model.pt").read_bytes() == b"partial export"
        run._commit_recovery()
        run._commit_recovery()  # Idempotent once accepted.
        assert (root / "training.jsonl").read_bytes() == prefix
        assert next((root / "recovery").rglob("training.uncommitted.jsonl")).read_bytes() == suffix
        assert next((root / "recovery").rglob("model.pt")).read_bytes() == b"partial export"
        assert not (root / "model.pt").exists() and not (root / "forecast.parquet").exists()
        run.log(dict(phase="selection", update=2, loss=3.0))
        run.save(state(2), reason="recovered")
    records = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
    assert [row["update"] for row in records] == [1, 2] and records[-1]["loss"] == 3


def test_training_checkpoint_preflight_log_change(tmp_path):
    root = tmp_path / "run"
    with TrainingRun(root, {}) as run:
        run.save(state(), reason="initial")
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        (root / "training.jsonl").write_bytes(b"changed during preflight")
        with pytest.raises(ValueError, match="changed during checkpoint preflight"):
            run._commit_recovery()
        assert (root / "training.jsonl").read_bytes() == b"changed during preflight"
        assert not (root / "recovery").exists()


def test_training_checkpoint_retention(tmp_path, monkeypatch):
    root = tmp_path / "run"
    with TrainingRun(root, {}) as run:
        for update in range(1, 5):
            run.log(dict(phase="selection", update=update))
            run.save(state(update), reason="update")
        assert [p.name for p in sorted((root / "checkpoints").glob("*.pt"))] == [
            "00000003.pt",
            "00000004.pt",
        ]
        assert len(list((root / "checkpoints").glob("*.json"))) == 4
        run.source_check = lambda: (_ for _ in ()).throw(ValueError("source changed"))
        with pytest.raises(ValueError, match="source changed"):
            run.save(state(4), reason="fail")
        assert len(list((root / "checkpoints").glob("*.pt"))) == 2
        run.source_check = lambda: None
        torch.set_num_threads(2 if torch.get_num_threads() == 1 else 1)
        with pytest.raises(ValueError, match="runtime changed"):
            run.save(state(4), reason="fail")
    configure_training_runtime("cpu", 1)
    for changes in ({"every_updates": True}, {"every_seconds": float("nan")}, {"keep_last": 1}):
        with pytest.raises(ValueError, match="cadence"):
            TrainingRun(tmp_path / "other", {}, **changes)


@pytest.mark.parametrize("mode", ["auto", "plain", "off", "tty"])
def test_training_progress(mode):
    class Stream(io.StringIO):
        def isatty(self):
            return mode == "tty"

    stream = Stream()
    progress = TrainingProgress("auto" if mode == "tty" else mode, stream=stream)
    progress.begin("selection", 4, initial=2)
    record = dict(
        phase="selection",
        update=3,
        epoch=2,
        epochs=2,
        steps_per_epoch=2,
        loss=1.2,
        learning_rate=0.001,
    )
    progress.update(record, running_mean=2.0, best=None, stale=0, latest_validation=None)
    progress.stage("validation batch 1/2", force=True)
    record.update(update=4, validation={"origin": {"wrmsse": 1.0}})
    progress.update(
        record,
        running_mean=1.8,
        best=0.9,
        stale=1,
        latest_validation=dict(update=4, epoch=2, metric=1.0, metric_name="wrmsse", nll=2.5),
    )
    progress.validation_summary(
        dict(update=4, epoch=2, metric=1.0, metric_name="wrmsse", nll=2.5),
        train_loss=1.8,
        best=0.9,
        stale=1,
        patience=3,
    )
    progress.close()
    value = stream.getvalue()
    if mode == "off":
        assert value == ""
    else:
        assert "Epoch 2/2 - selection (train + validation/early stopping)" in value
        assert "2/2" in value and "validation" in value
        assert "loss: 1.8" in value and "val_loss: 2.5" in value
        assert "val_wrmsse: 1" in value and "learning_rate: 0.001" in value
        assert "last_val_" not in value
    with pytest.raises(ValueError, match="progress"):
        TrainingProgress("unknown")
    with pytest.raises(ValueError, match="initial"):
        progress.begin("selection", 4, initial=5)
    with pytest.raises(ValueError, match="metric/update"):
        progress.update(
            dict(record, loss=float("nan")),
            running_mean=1,
            best=None,
            stale=0,
            latest_validation=None,
        )


def test_training_progress_falls_back_when_optional_tqdm_is_absent(monkeypatch) -> None:
    """The custom-Torch runtime does not declare tqdm, so auto mode must remain runnable."""
    from autoforecast import training_progress

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(training_progress, "tqdm", None, raising=False)
    monkeypatch.setattr(training_progress, "_tqdm", None, raising=False)
    stream = Terminal()
    progress = TrainingProgress("auto", stream=stream)
    progress.begin("selection", 1)
    progress.close()
    assert progress.mode == "plain"
    assert "selection: start" in stream.getvalue()


def test_training_progress_live_campaign_uses_tty_renderer(monkeypatch) -> None:
    """The campaign relays a worker log, so its explicit live flag stands in for isatty()."""
    monkeypatch.setenv("AUTOFORECAST_LIVE_PROGRESS", "1")
    progress = TrainingProgress("auto", stream=io.StringIO())
    assert progress.mode == "tty"


def test_training_progress_includes_campaign_stage_position(monkeypatch) -> None:
    """The worker receives campaign position as display-only supervisor metadata."""
    monkeypatch.setenv("AUTOFORECAST_CAMPAIGN_STAGE_INDEX", "2")
    monkeypatch.setenv("AUTOFORECAST_CAMPAIGN_STAGE_TOTAL", "7")
    stream = io.StringIO()
    progress = TrainingProgress("plain", stream=stream)
    progress.begin("selection", 1)
    progress.update(
        dict(phase="selection", update=1, loss=1.2, learning_rate=0.001),
        running_mean=1.2,
        best=None,
        stale=0,
        latest_validation=None,
    )
    progress.close()
    assert (
        "Stage 2/7 | Phase — selection (train + validation/early stopping) | "
        "selection: start" in stream.getvalue()
    )
    assert (
        "Stage 2/7 | Phase — selection (train + validation/early stopping) | "
        "selection 1/1" in stream.getvalue()
    )


def test_pipeline_progress_shows_substage_spark_tasks_time_and_ram(monkeypatch) -> None:
    from autoforecast import training_progress

    class Tracker:
        def getActiveStageIds(self):
            return [4]

        def getStageInfo(self, _stage_id):
            return SimpleNamespace(numCompletedTasks=3, numTasks=12)

    spark = SimpleNamespace(sparkContext=SimpleNamespace(statusTracker=lambda: Tracker()))
    monkeypatch.setenv("AUTOFORECAST_LIVE_PROGRESS", "1")
    monkeypatch.setenv("AUTOFORECAST_CAMPAIGN_STAGE_INDEX", "2")
    monkeypatch.setenv("AUTOFORECAST_CAMPAIGN_STAGE_TOTAL", "6")
    monkeypatch.setattr(training_progress, "_process_tree_memory", lambda: (2.5, 8.25))
    stream = io.StringIO()
    progress = PipelineProgress("Spark preflight", 2, stream=stream, interval_seconds=0)
    with progress.step("Inspect the panel", spark=spark):
        pass
    value = stream.getvalue()
    assert "Stage 2/6 | Phase — Spark preflight | Step 1/2 — Inspect the panel\n" in value
    assert "Spark preflight\n  Step" not in value
    assert "Spark engine stage ID: 4 (diagnostic only)" in value
    assert "Spark tasks: 3/12" in value
    assert "elapsed:" in value and "ETA:" in value
    assert "process_tree_RAM: 2.50 GiB" in value
    assert "available_RAM: 8.25 GiB" in value


def test_training_progress_separates_training_and_validation_batches(monkeypatch) -> None:
    monkeypatch.setenv("AUTOFORECAST_CAMPAIGN_STAGE_INDEX", "1")
    monkeypatch.setenv("AUTOFORECAST_CAMPAIGN_STAGE_TOTAL", "3")
    monkeypatch.setenv("AUTOFORECAST_WORKFLOW_LAYOUT", "single_model_benchmark_v1")
    monkeypatch.setenv("AUTOFORECAST_WORKFLOW_STAGE_ROLE", "validation")
    stream = io.StringIO()
    progress = TrainingProgress("plain", stream=stream)
    progress.begin("selection", 1)
    progress.update(
        {
            "phase": "selection",
            "update": 1,
            "epoch": 1,
            "epochs": 2,
            "steps_per_epoch": 1,
            "loss": 1.2,
            "learning_rate": 0.001,
        },
        running_mean=1.2,
        best=None,
        stale=0,
        latest_validation=None,
    )
    progress.begin_validation("2016-04-24", index=1, total=1, batches=2, kind="fixed holdout")
    progress.validation_status("running inference")
    progress.validation_batch(1, 2)
    progress.validation_batch(2, 2)
    progress.end_validation()
    progress.validation_summary(
        {"metric": 0.91, "metric_name": "wrmsse", "nll": 1.1},
        train_loss=1.2,
        best=0.91,
        stale=0,
        patience=10,
    )
    value = stream.getvalue()
    assert "Stage 1/3 | Phase 4/6 — Train, validate, and select the update budget" in value
    assert "Epoch 1/2 — Fixed holdout after 2016-04-24" in value
    assert "validation batch 2/2" in value
    assert "Epoch 1/2 complete - loss: 1.2 - val_loss: 1.1 - val_wrmsse: 0.91" in value
    assert "wait: 0/10" in value
