"""Code-owned MLflow behavior for adapters; model YAML never configures tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autoforecast.adapter_contracts import DESCRIPTORS, descriptor

TrackingMode = Literal["autolog", "reporter", "none"]
TrackingFlavor = Literal[
    "lightning", "sklearn", "lightgbm", "xgboost", "tensorflow", "keras", "none"
]


@dataclass(frozen=True)
class TrackingPlan:
    """One safe tracking behavior selected by an adapter registration."""

    mode: TrackingMode
    flavor: TrackingFlavor
    log_models: bool = False

    def validate(self) -> TrackingPlan:
        if self.mode == "autolog" and self.flavor == "none":
            raise ValueError("autolog tracking requires a framework flavor")
        if self.mode != "autolog" and self.flavor != "none":
            raise ValueError("reporter/none tracking must not select an autolog flavor")
        if self.mode != "autolog" and self.log_models:
            raise ValueError("only autolog tracking may log framework models")
        return self


_PLANS = {name: TrackingPlan(*descriptor(name).tracking) for name in DESCRIPTORS}


def tracking_plan(adapter: str) -> TrackingPlan:
    """Return the centrally registered plan; unknown adapters fail closed."""
    try:
        return _PLANS[adapter].validate()
    except KeyError as error:
        raise ValueError(f"adapter {adapter!r} has no tracking plan") from error


def enable_autologging(plan: TrackingPlan) -> None:
    """Enable exactly one supported MLflow integration inside an already-owned run."""
    plan.validate()
    if plan.mode != "autolog":
        return
    try:
        import mlflow
    except ImportError as error:  # pragma: no cover - optional boundary
        raise RuntimeError(
            "MLflow tracking requires the standard AutoForecast installation with dependencies"
        ) from error
    flavor = getattr(mlflow, "pytorch" if plan.flavor == "lightning" else plan.flavor)
    flavor.autolog(log_models=plan.log_models)
