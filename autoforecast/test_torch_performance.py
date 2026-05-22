"""Bounded CPU/CUDA profiler contract tests independent of model architecture."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from autoforecast.torch_performance import TorchPerformanceProfiler
from autoforecast.torch_profile_contract import (
    TorchProfileConfig,
    profile_request_from_environment,
)


def test_environment_switch_is_exact_and_disabled_by_default() -> None:
    assert profile_request_from_environment({}) is None
    assert profile_request_from_environment({"AUTOFORECAST_TORCH_PROFILE": "0"}) is None
    assert profile_request_from_environment({"AUTOFORECAST_TORCH_PROFILE": "1"}) == (
        TorchProfileConfig().as_record()
    )
    for value in ("true", "yes", "2", ""):
        with pytest.raises(ValueError, match="exactly 0 or 1"):
            profile_request_from_environment({"AUTOFORECAST_TORCH_PROFILE": value})


def test_profile_record_is_strict_and_bounded() -> None:
    record = TorchProfileConfig().as_record()
    assert TorchProfileConfig.from_record(record).as_record() == record
    with pytest.raises(ValueError, match="unknown or missing"):
        TorchProfileConfig.from_record({**record, "unexpected": True})
    for name, value in (
        ("skip_steps", 1),
        ("warmup_steps", 1),
        ("capture_steps", 4),
        ("record_shapes", False),
        ("profile_memory", False),
        ("with_stack", False),
        ("with_flops", False),
    ):
        with pytest.raises(ValueError, match="invalid bounded"):
            TorchProfileConfig.from_record({**record, name: value})
    with pytest.raises(ValueError, match="invalid bounded"):
        TorchProfileConfig.from_record({**record, "skip_steps": True})


def test_cpu_profile_publishes_training_and_validation_artifacts(tmp_path) -> None:
    profiler = TorchPerformanceProfiler(
        TorchProfileConfig(),
        tmp_path,
        device="cpu",
    )
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    for _ in range(10):
        profiler.begin_step("training")
        optimizer.zero_grad(set_to_none=True)
        values = torch.randn(8, 4)
        with profiler.record("training", "forward_and_loss"):
            loss = model(values).square().mean()
        with profiler.record("training", "backward"):
            loss.backward()
        with profiler.record("training", "optimizer_step"):
            optimizer.step()
        profiler.end_step("training")
    with torch.no_grad():
        for _ in range(10):
            profiler.begin_step("validation")
            with profiler.record("validation", "forward"):
                model(torch.randn(8, 4))
            profiler.end_step("validation")
    profiler.close()
    profiler.close()

    manifest = json.loads((profiler.session_dir / "manifest.json").read_text())
    assert manifest["device"] == "cpu"
    assert manifest["activities"] == ["cpu"]
    for phase in ("training", "validation"):
        assert manifest["phases"][phase]["status"] == "captured"
        assert manifest["phases"][phase]["observed_steps"] == 9
        assert manifest["phases"][phase]["captured_steps"] == 5
        artifacts = manifest["phases"][phase]["artifacts"]
        assert {"chrome_trace", "operator_table"} <= set(artifacts)
        for relative in artifacts.values():
            assert (profiler.session_dir / relative).stat().st_size > 0
        if "folded_stacks" not in artifacts:
            assert manifest["phases"][phase]["unavailable_artifacts"] == [
                "folded_stacks_empty_on_runtime"
            ]
        trace = (profiler.session_dir / artifacts["chrome_trace"]).read_text()
        assert f"autoforecast.{phase}." in trace
    operators = (profiler.session_dir / "training.operators.txt").read_text()
    assert "forward_and_loss" in operators
    assert "optimizer_step" in operators


def test_short_phase_switches_do_not_overlap_or_publish_empty_traces(tmp_path) -> None:
    profiler = TorchPerformanceProfiler(TorchProfileConfig(), tmp_path, device="cpu")
    profiler.begin_step("training")
    with profiler.record("training", "forward"):
        torch.ones(1).square()
    profiler.end_step("training")

    # Switching before the nine-step schedule completes used to start a second, overlapping
    # Kineto profiler.  It must now seal training before validation starts.
    profiler.begin_step("validation")
    with profiler.record("validation", "forward"):
        torch.ones(1).square()
    profiler.end_step("validation")
    profiler.close()

    manifest = json.loads((profiler.session_dir / "manifest.json").read_text())
    assert manifest["phases"]["training"] == {
        "artifacts": {},
        "captured_steps": 0,
        "observed_steps": 1,
        "publication_error": None,
        "status": "not_captured",
        "unavailable_artifacts": [],
    }
    assert manifest["phases"]["validation"]["status"] == "not_captured"


def test_partial_active_window_is_reported_as_partial(tmp_path) -> None:
    profiler = TorchPerformanceProfiler(TorchProfileConfig(), tmp_path, device="cpu")
    for _ in range(5):
        profiler.begin_step("training")
        with profiler.record("training", "forward"):
            torch.ones(1).square()
        profiler.end_step("training")
    profiler.close()

    manifest = json.loads((profiler.session_dir / "manifest.json").read_text())
    training = manifest["phases"]["training"]
    assert training["observed_steps"] == 5
    assert training["captured_steps"] == 1
    assert training["status"] == "partial"
    assert {"chrome_trace", "operator_table"} <= set(training["artifacts"])
    assert all(
        (profiler.session_dir / relative).stat().st_size > 0
        for relative in training["artifacts"].values()
    )


def test_cross_phase_guard_is_independent_of_trainer_call_order(monkeypatch, tmp_path) -> None:
    class ExclusiveProfiler:
        active = None

        def start(self):
            if ExclusiveProfiler.active is not None:
                raise RuntimeError("overlapping profiler")
            ExclusiveProfiler.active = self

        def step(self):
            pass

        def stop(self):
            if ExclusiveProfiler.active is self:
                ExclusiveProfiler.active = None

    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: ExclusiveProfiler())
    profiler = TorchPerformanceProfiler(TorchProfileConfig(), tmp_path, device="cpu")
    profiler.begin_step("training")
    profiler.end_step("training")
    profiler.begin_step("validation")
    profiler.end_step("validation")
    profiler.close()
    assert ExclusiveProfiler.active is None


def test_publication_failure_is_atomic_and_close_is_retryable(monkeypatch, tmp_path) -> None:
    profiler = TorchPerformanceProfiler(TorchProfileConfig(), tmp_path, device="cpu")
    for _ in range(5):
        profiler.begin_step("training")
        with profiler.record("training", "forward"):
            torch.ones(1).square()
        profiler.end_step("training")
    backend = profiler._states["training"].profiler
    monkeypatch.setattr(
        backend,
        "export_stacks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stack export failed")),
    )
    with pytest.raises(RuntimeError, match="stack export failed"):
        profiler.close()
    assert not list(profiler.session_dir.glob("training.*"))

    profiler.close()
    manifest = json.loads((profiler.session_dir / "manifest.json").read_text())
    assert manifest["phases"]["training"]["status"] == "failed"
    assert manifest["phases"]["training"]["publication_error"] == {
        "message": "stack export failed",
        "type": "RuntimeError",
    }


def test_cuda_activity_selection_requires_cupti_support(monkeypatch, tmp_path) -> None:
    cpu = torch.profiler.ProfilerActivity.CPU
    cuda = torch.profiler.ProfilerActivity.CUDA
    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: {cpu})
    with pytest.raises(RuntimeError, match="PyTorch/CUPTI"):
        TorchPerformanceProfiler(TorchProfileConfig(), tmp_path, device="cuda")
    monkeypatch.setattr(torch.profiler, "supported_activities", lambda: {cpu, cuda})
    assert TorchPerformanceProfiler._resolve_activities("cuda") == (cpu, cuda)


def test_sessions_are_append_only(tmp_path) -> None:
    first = TorchPerformanceProfiler(TorchProfileConfig(), tmp_path, device="cpu")
    first.close()
    second = TorchPerformanceProfiler(TorchProfileConfig(), tmp_path, device="cpu")
    second.close()
    assert first.session_dir.name == "session-0001"
    assert second.session_dir.name == "session-0002"


def test_parent_serializes_profile_before_isolated_worker_boundary(monkeypatch, tmp_path) -> None:
    from scripts import run_autoforecast

    context = SimpleNamespace(
        attempt_dir=tmp_path,
        input=SimpleNamespace(
            run=SimpleNamespace(adapter="hmfnet"),
        ),
    )
    plan = SimpleNamespace(as_record=lambda: {"adapter_kwargs": {}})
    monkeypatch.setenv("AUTOFORECAST_TORCH_PROFILE", "1")
    kwargs = run_autoforecast._worker_kwargs(
        context,
        plan,
        parent_checkpoint=None,
        training_controls={},
        calendar_events=(),
        event_max_support=14,
        training_mode="fixed_validation",
        validation_data_path=tmp_path / "validation",
        fixed_update_budget=None,
        validation_metric="wrmsse",
    )
    assert kwargs["performance_profile"] == TorchProfileConfig().as_record()
    assert kwargs["training_log_dir"] == str(tmp_path / "tensorboard")
    assert kwargs["validation_metric"] == "wrmsse"


def test_parent_keeps_revision_validation_controls_out_of_legacy_adapter(tmp_path) -> None:
    from scripts import run_autoforecast

    context = SimpleNamespace(
        attempt_dir=tmp_path,
        input=SimpleNamespace(run=SimpleNamespace(adapter="mft")),
    )
    plan = SimpleNamespace(as_record=lambda: {"adapter_kwargs": {}})

    kwargs = run_autoforecast._worker_kwargs(
        context,
        plan,
        parent_checkpoint=None,
        training_controls={},
        calendar_events=(),
        event_max_support=14,
        training_mode="fixed_validation",
        validation_data_path=tmp_path / "validation",
        fixed_update_budget=None,
        validation_metric="wrmsse",
    )

    assert "validation_metric" not in kwargs
    assert "validation_data_path" not in kwargs
    assert "fixed_update_budget" not in kwargs


def test_parent_passes_tft_fixed_refit_budget_and_governed_checkpoint(tmp_path) -> None:
    from scripts import run_autoforecast

    context = SimpleNamespace(
        attempt_dir=tmp_path,
        input=SimpleNamespace(run=SimpleNamespace(adapter="tft_q")),
    )
    plan = SimpleNamespace(as_record=lambda: {"adapter_kwargs": {"max_epochs": 50000}})
    kwargs = run_autoforecast._worker_kwargs(
        context,
        plan,
        parent_checkpoint=None,
        training_controls={
            "batch_size": 32,
            "device": "mps",
            "patience": 10,
            "early_stopping_absolute_min_delta": 1e-6,
        },
        calendar_events=(),
        event_max_support=14,
        training_mode="fixed_refit",
        validation_data_path=None,
        fixed_update_budget=120,
        validation_metric=None,
    )
    assert kwargs["checkpoint_out"] == str(tmp_path / "model" / "model.pt")
    assert kwargs["training_mode"] == "fixed_refit"
    assert kwargs["fixed_update_budget"] == 120
    assert kwargs["max_epochs"] == 50000
    assert "precision" not in kwargs


def test_parent_routes_mixed_precision_to_tft_and_auto_device(tmp_path) -> None:
    from scripts import run_autoforecast

    context = SimpleNamespace(
        attempt_dir=tmp_path,
        input=SimpleNamespace(run=SimpleNamespace(adapter="tft_q")),
    )
    plan = SimpleNamespace(as_record=lambda: {"adapter_kwargs": {}})
    kwargs = run_autoforecast._worker_kwargs(
        context,
        plan,
        parent_checkpoint=None,
        training_controls={
            "batch_size": 32,
            "device": None,
            "patience": 10,
            "early_stopping_absolute_min_delta": 1e-6,
            "precision": "16-mixed",
        },
        calendar_events=(),
        event_max_support=14,
        training_mode="fixed_validation",
        validation_data_path=None,
        fixed_update_budget=None,
        validation_metric=None,
    )
    assert kwargs["precision"] == "16-mixed"
    assert kwargs["device"] == "auto"


def test_parent_profile_contract_does_not_import_torch_runtime() -> None:
    code = """
import pathlib
import sys
from types import SimpleNamespace
from scripts import run_autoforecast
context = SimpleNamespace(
    attempt_dir=pathlib.Path('/tmp/autoforecast-profile-contract'),
    input=SimpleNamespace(run=SimpleNamespace(adapter='hmfnet')),
)
plan = SimpleNamespace(as_record=lambda: {'adapter_kwargs': {}})
kwargs = run_autoforecast._worker_kwargs(
    context,
    plan,
    parent_checkpoint=None,
    training_controls={},
    calendar_events=(),
    event_max_support=14,
    training_mode='fixed_validation',
    validation_data_path=pathlib.Path('/tmp/validation'),
    fixed_update_budget=None,
    validation_metric='wrmsse',
)
assert kwargs['performance_profile']['schema'].endswith('.v1')
assert 'torch' not in sys.modules
"""
    environment = dict(os.environ, AUTOFORECAST_TORCH_PROFILE="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
