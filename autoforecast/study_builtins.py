"""Library models a saved study can train by name, for example ``run_study(study_id, "auto_ets")``.

Each name maps to a fit function in an adapter module. With no ``model_config``, every setting
is fixed at its default (so no parameter search runs); a supplied profile can search them.
Adapter modules are imported only when their model is chosen, because each brings its own
dependency (statsforecast, LightGBM).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import inspect
import pathlib
from collections.abc import Callable

from autoforecast.panel import Panel

__all__ = ["BUILTIN_STUDY_MODELS", "BuiltinStudyModel", "builtin_profile", "builtin_study_model"]


@dataclasses.dataclass(frozen=True)
class BuiltinStudyModel:
    """One library fit function for saved studies.

    Attributes:
        module: Adapter module in ``autoforecast.adapters``; its source SHA-256 identifies the
            implementation.
        fit: Name of the module's function that takes a ``FitPredictRequest``.
        defaults: The study panel -> the default value of each setting.
        packages: Installed packages whose versions are part of the implementation identity.
    """

    module: str
    fit: str
    defaults: Callable[[Panel], dict[str, object]]
    packages: tuple[str, ...] = ()


def _season(panel: Panel) -> dict[str, object]:
    return {"season_length": panel.season_length}


def _lightgbm(panel: Panel) -> dict[str, object]:
    # The adapter's own defaults, so the two can never differ.
    from autoforecast.adapters.shallow import STUDY_SETTINGS, fit_predict

    signature = inspect.signature(fit_predict).parameters
    return {name: signature[name].default for name in STUDY_SETTINGS}


BUILTIN_STUDY_MODELS = {
    "seasonal_naive": BuiltinStudyModel("naive", "fit_study_baseline", _season),
    "window_average": BuiltinStudyModel(
        "naive", "fit_study_window_average", lambda panel: {"window": panel.season_length},
    ),
    "croston": BuiltinStudyModel("classical", "fit_study_croston", _season, ("statsforecast",)),
    "auto_ets": BuiltinStudyModel("classical", "fit_study_auto_ets", _season, ("statsforecast",)),
    "auto_theta": BuiltinStudyModel(
        "classical", "fit_study_auto_theta", _season, ("statsforecast",),
    ),
    "shallow_global": BuiltinStudyModel(
        "shallow", "fit_study_shallow_global", _lightgbm, ("lightgbm",),
    ),
}


def builtin_study_model(name: str) -> tuple[Callable, dict[str, object]]:
    """The fit function and implementation identity of the built-in model ``name``."""
    model = BUILTIN_STUDY_MODELS[name]
    try:
        module = importlib.import_module(f"autoforecast.adapters.{model.module}")
        dependencies = {package: importlib.metadata.version(package)
                        for package in model.packages}
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise ValueError(
            f"{name} needs {', '.join(model.packages)} installed in this environment"
        ) from error
    assert module.__file__ is not None
    return getattr(module, model.fit), {
        "name": name,
        "version": "1",
        "source_sha256": hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest(),
        "dependencies": dependencies,
    }


def builtin_profile(name: str, panel: Panel) -> dict[str, object]:
    """The default profile: each setting fixed at its default value, so no search runs."""
    parameters: dict[str, object] = {}
    for setting, value in BUILTIN_STUDY_MODELS[name].defaults(panel).items():
        domain = ({"kind": "int", "low": value, "high": value, "step": 1} if type(value) is int
                  else {"kind": "categorical", "values": [value]})
        parameters[f"model.{setting}"] = {"default": value, "valid": domain, "search": domain}
    return {"schema_version": 2, "model": {"id": name, "adapter": "external"},
            "hyperparameters": parameters}
