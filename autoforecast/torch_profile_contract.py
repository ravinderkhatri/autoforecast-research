"""Dependency-free contract for requesting an isolated-worker PyTorch profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

TORCH_PROFILE_SCHEMA = "autoforecast.torch-performance-profile.v1"


@dataclass(frozen=True)
class TorchProfileConfig:
    """Strict, bounded profiler request passed as inert data to an isolated worker."""

    schema: str = TORCH_PROFILE_SCHEMA
    skip_steps: int = 2
    warmup_steps: int = 2
    capture_steps: int = 5
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True

    def __post_init__(self) -> None:
        # This is a versioned protocol, not a user-tunable training surface.  Keeping the
        # exact schedule and evidence set frozen makes the opt-in switch predictably bounded
        # and comparable across CPU and NVIDIA runs.  A future protocol change requires a
        # schema bump rather than an undocumented numerical knob.
        expected = (
            TORCH_PROFILE_SCHEMA,
            2,
            2,
            5,
            True,
            True,
            True,
            True,
        )
        actual = (
            self.schema,
            self.skip_steps,
            self.warmup_steps,
            self.capture_steps,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
        )
        wrong_types = any(
            type(value) is not type(reference)
            for value, reference in zip(actual, expected, strict=True)
        )
        if actual != expected or wrong_types:
            raise ValueError("invalid bounded Torch profiler configuration")

    def as_record(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TorchProfileConfig:
        if type(record) is not dict or set(record) != set(cls().as_record()):
            raise ValueError("Torch profiler request has an unknown or missing field")
        return cls(**record)


def profile_request_from_environment(
    environment: Mapping[str, str],
) -> dict[str, object] | None:
    """Parse the sole user-facing switch before crossing the worker boundary."""
    value = environment.get("AUTOFORECAST_TORCH_PROFILE")
    if value in {None, "0"}:
        return None
    if value != "1":
        raise ValueError("AUTOFORECAST_TORCH_PROFILE must be exactly 0 or 1")
    return TorchProfileConfig().as_record()
