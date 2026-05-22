"""Bounded, opt-in PyTorch performance traces for shared training loops.

This module is deliberately independent of any model architecture.  Callers mark semantic
regions and steps; PyTorch/Kineto records CPU operators and, on NVIDIA hosts, CUDA/CUPTI
operators.  Profiling is absent unless the parent process passes an authenticated config.
"""

from __future__ import annotations

import json
import pathlib
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from autoforecast.torch_profile_contract import TORCH_PROFILE_SCHEMA, TorchProfileConfig

_PROFILE_PHASES = ("training", "validation")


@dataclass
class _PhaseState:
    observed_steps: int = 0
    captured_steps: int = 0
    active: bool = False
    complete: bool = False
    profiler: Any = None
    artifacts: dict[str, str] = field(default_factory=dict)
    unavailable_artifacts: list[str] = field(default_factory=list)
    publication_error: BaseException | None = None


class TorchPerformanceProfiler:
    """Capture short training and validation samples without owning model semantics."""

    def __init__(
        self,
        config: TorchProfileConfig,
        output_root: str | pathlib.Path,
        *,
        device: str,
    ) -> None:
        if not isinstance(config, TorchProfileConfig):
            raise TypeError("Torch profiler requires a validated TorchProfileConfig")
        if device not in {"cpu", "cuda"}:
            raise ValueError("Torch performance profiling supports cpu or cuda")
        self.config = config
        self.device = device
        self.activities = self._resolve_activities(device)
        self.output_root = pathlib.Path(output_root).resolve()
        self.session_dir = self._create_session_dir(self.output_root)
        self._states = {phase: _PhaseState() for phase in _PROFILE_PHASES}
        self._closed = False

    @staticmethod
    def _resolve_activities(device: str) -> tuple[torch.profiler.ProfilerActivity, ...]:
        supported = torch.profiler.supported_activities()
        cpu = torch.profiler.ProfilerActivity.CPU
        if cpu not in supported:
            raise RuntimeError("this PyTorch build cannot record CPU profiler activity")
        activities = [cpu]
        if device == "cuda":
            cuda = torch.profiler.ProfilerActivity.CUDA
            if cuda not in supported:
                raise RuntimeError(
                    "the requested NVIDIA profile is unavailable: PyTorch/CUPTI has no CUDA "
                    "profiler activity"
                )
            activities.append(cuda)
        return tuple(activities)

    @staticmethod
    def _create_session_dir(root: pathlib.Path) -> pathlib.Path:
        root.mkdir(parents=True, exist_ok=True)
        for index in range(1, 10_000):
            candidate = root / f"session-{index:04d}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            return candidate
        raise RuntimeError("Torch profiler session namespace is exhausted")

    def begin_step(self, phase: str) -> bool:
        """Start one step in PyTorch's bounded wait/warm-up/active schedule."""
        state = self._state(phase)
        if self._closed or state.complete:
            return False
        if state.active:
            raise RuntimeError(f"Torch profiler phase {phase!r} already has an active step")
        state.observed_steps += 1
        if state.profiler is None:
            # Kineto profilers are thread-global.  This guard makes the reusable profiler
            # safe even if a caller changes phase before the short schedule is complete.
            for other_phase, other_state in self._states.items():
                if other_phase != phase and other_state.profiler is not None:
                    self.finish_phase(other_phase)
            state.profiler = torch.profiler.profile(
                activities=list(self.activities),
                schedule=torch.profiler.schedule(
                    wait=self.config.skip_steps,
                    warmup=self.config.warmup_steps,
                    active=self.config.capture_steps,
                    repeat=1,
                ),
                on_trace_ready=lambda profiler, phase=phase: self._publish_phase(phase, profiler),
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
                with_flops=self.config.with_flops,
                acc_events=True,
            )
            state.profiler.start()
        state.active = True
        return True

    @contextmanager
    def record(self, phase: str, name: str) -> Iterator[None]:
        """Annotate one semantic region for PyTorch and NVIDIA Nsight timelines."""
        state = self._state(phase)
        if not state.active:
            yield
            return
        label = f"autoforecast.{phase}.{name}"
        with torch.profiler.record_function(label):
            if self.device == "cuda":
                with torch.cuda.nvtx.range(label):
                    yield
            else:
                yield

    def end_step(self, phase: str) -> None:
        state = self._state(phase)
        if not state.active:
            return
        captured = state.observed_steps > self.config.skip_steps + self.config.warmup_steps
        if captured:
            # ``on_trace_ready`` runs inside ``step()``.  Update the count first so the
            # callback and the final manifest observe the completed active step.
            state.captured_steps += 1
        try:
            state.profiler.step()
        except BaseException:
            if captured:
                state.captured_steps -= 1
            state.active = False
            raise
        state.active = False
        if state.captured_steps == self.config.capture_steps:
            self._stop_phase_profiler(phase)

    def finish_phase(self, phase: str) -> None:
        """Publish a completed or partial phase trace exactly once."""
        state = self._state(phase)
        if state.complete:
            return
        if state.active:
            self.end_step(phase)
            if state.complete:
                return
        if state.profiler is not None:
            self._stop_phase_profiler(phase)
        else:
            state.complete = True

    def _stop_phase_profiler(self, phase: str) -> None:
        """Release Kineto state even when its publication callback fails."""
        state = self._state(phase)
        profiler = state.profiler
        if profiler is None:
            state.complete = True
            return
        stop_error: BaseException | None = None
        try:
            profiler.stop()
        except BaseException as error:
            stop_error = error
        finally:
            # ``stop`` may fail after Kineto has already disabled the native profiler.  Keeping
            # that dead object makes every later close fail with "profiler is not running".
            state.profiler = None
            state.complete = True
        if state.publication_error is not None:
            if stop_error is not None and stop_error is not state.publication_error:
                state.publication_error.add_note(f"Torch profiler stop also failed: {stop_error!r}")
            raise state.publication_error
        if stop_error is not None:
            raise stop_error

    def _publish_phase(self, phase: str, profiler: Any) -> None:
        """Publish exactly the active window selected by PyTorch's scheduler."""
        state = self._state(phase)
        if state.captured_steps == 0:
            return
        if state.artifacts or state.publication_error is not None:
            raise RuntimeError(f"Torch profiler phase {phase!r} attempted duplicate publication")
        stem = self.session_dir / phase
        trace = stem.with_suffix(".trace.json")
        operators = stem.with_suffix(".operators.txt")
        stacks = stem.with_suffix(".stacks.txt")
        temporary = {
            trace: trace.with_name(trace.name + ".tmp"),
            operators: operators.with_name(operators.name + ".tmp"),
            stacks: stacks.with_name(stacks.name + ".tmp"),
        }
        try:
            profiler.export_chrome_trace(str(temporary[trace]))
            sort_key = "self_cuda_time_total" if self.device == "cuda" else "self_cpu_time_total"
            table = profiler.key_averages(
                group_by_input_shape=True,
                group_by_stack_n=5,
            ).table(sort_by=sort_key, row_limit=200)
            temporary[operators].write_text(table + "\n", encoding="utf-8")
            profiler.export_stacks(str(temporary[stacks]), metric=sort_key)
            if temporary[trace].stat().st_size == 0 or temporary[operators].stat().st_size == 0:
                raise RuntimeError("Torch profiler produced an empty core trace artifact")
            temporary[trace].replace(trace)
            temporary[operators].replace(operators)
            artifacts = {
                "chrome_trace": trace.name,
                "operator_table": operators.name,
            }
            if temporary[stacks].stat().st_size:
                temporary[stacks].replace(stacks)
                artifacts["folded_stacks"] = stacks.name
            else:
                temporary[stacks].unlink()
                state.unavailable_artifacts.append("folded_stacks_empty_on_runtime")
            state.artifacts = artifacts
        except BaseException as error:
            # Publication is an all-or-nothing evidence operation.  The callback cannot raise
            # through Kineto without corrupting its lifecycle, so remember the original error;
            # the owning ``end_step``/``finish_phase`` re-raises it after native cleanup.
            for path in (*temporary.values(), *temporary.keys()):
                path.unlink(missing_ok=True)
            state.artifacts.clear()
            state.unavailable_artifacts.clear()
            state.publication_error = error

    def close(self) -> None:
        """Close every phase and write a small machine-readable artifact index."""
        if self._closed:
            return
        for phase in _PROFILE_PHASES:
            self.finish_phase(phase)
        manifest = {
            "schema": TORCH_PROFILE_SCHEMA,
            "device": self.device,
            "activities": [activity.name.lower() for activity in self.activities],
            "config": self.config.as_record(),
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "phases": {
                phase: {
                    "observed_steps": state.observed_steps,
                    "captured_steps": state.captured_steps,
                    "status": (
                        "failed"
                        if state.publication_error is not None
                        else "captured"
                        if state.artifacts and state.captured_steps == self.config.capture_steps
                        else "partial"
                        if state.artifacts
                        else "not_captured"
                    ),
                    "artifacts": state.artifacts,
                    "unavailable_artifacts": state.unavailable_artifacts,
                    "publication_error": (
                        None
                        if state.publication_error is None
                        else {
                            "type": type(state.publication_error).__name__,
                            "message": str(state.publication_error),
                        }
                    ),
                }
                for phase, state in self._states.items()
            },
        }
        (self.session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._closed = True

    def _state(self, phase: str) -> _PhaseState:
        try:
            return self._states[phase]
        except KeyError as exc:
            raise ValueError(f"unknown Torch profiler phase {phase!r}") from exc
