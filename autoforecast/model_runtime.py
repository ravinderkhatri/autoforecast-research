"""Declared optional runtimes for isolated AutoForecast adapters.

Profiles select a runtime by its stable identifier.  This module owns the corresponding Python
module checks and installation hint so a YAML file cannot silently claim a dependency exists or
choose an arbitrary package to import.  The check uses module discovery only: native libraries
are loaded exclusively in the isolated model worker.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from dataclasses import dataclass

__all__ = ["ModelRuntime", "RuntimeCheck", "check_runtime", "runtime", "runtime_ids"]


@dataclass(frozen=True)
class ModelRuntime:
    """One maintained runtime family selected by a model profile."""

    id: str
    required_modules: tuple[str, ...]
    extra: str | None


@dataclass(frozen=True)
class RuntimeCheck:
    """Dependency-free availability evidence for one requested model runtime."""

    runtime: ModelRuntime
    missing_modules: tuple[str, ...]

    @property
    def available(self) -> bool:
        return not self.missing_modules

    @property
    def summary(self) -> str:
        if self.available:
            return f"Runtime '{self.runtime.id}' is available."
        missing = ", ".join(self.missing_modules)
        hint = (
            f" Install with: uv sync --extra {self.runtime.extra}."
            if self.runtime.extra is not None
            else ""
        )
        return (
            f"Runtime '{self.runtime.id}' is unavailable; missing Python module(s): {missing}."
            f"{hint}"
        )


_RUNTIMES = {
    "builtin": ModelRuntime("builtin", (), None),
    "classical": ModelRuntime("classical", ("statsforecast",), "classical"),
    "custom_torch": ModelRuntime("custom_torch", ("torch", "tensorboard"), "custom-torch"),
    "shallow": ModelRuntime("shallow", ("lightgbm", "pandas"), "shallow"),
    "tft": ModelRuntime("tft", ("torch", "lightning", "pytorch_forecasting"), "tft"),
    "autogluon_zoo": ModelRuntime(
        "autogluon_zoo", ("autogluon.timeseries",), "autogluon-zoo"
    ),
    "timesfm": ModelRuntime("timesfm", ("timesfm", "torch"), "timesfm"),
    "timesfm3": ModelRuntime("timesfm3", ("timesfm3", "torch", "lightning", "peft"), "timesfm3"),
}


def runtime_ids() -> tuple[str, ...]:
    """Known profile runtime identifiers, sorted for deterministic validation messages."""
    return tuple(sorted(_RUNTIMES))


def runtime(runtime_id: str) -> ModelRuntime:
    """Return one known runtime definition without importing its native packages."""
    try:
        return _RUNTIMES[runtime_id]
    except KeyError:
        raise ValueError(
            f"unknown model runtime {runtime_id!r}; known={list(runtime_ids())}"
        ) from None


def _module_available(module: str) -> bool:
    """Discover dotted packages without executing their parent initializers."""
    parts = module.split(".")
    spec = importlib.util.find_spec(parts[0])
    for index in range(1, len(parts)):
        if spec is None or spec.submodule_search_locations is None:
            return False
        spec = importlib.machinery.PathFinder.find_spec(
            ".".join(parts[:index + 1]), spec.submodule_search_locations
        )
    return spec is not None


def check_runtime(runtime_id: str) -> RuntimeCheck:
    """Check declared modules without importing model code or a native runtime."""
    definition = runtime(runtime_id)
    missing = tuple(
        module for module in definition.required_modules if not _module_available(module)
    )
    return RuntimeCheck(definition, missing)
