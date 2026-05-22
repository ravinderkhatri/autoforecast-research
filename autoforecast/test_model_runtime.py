"""Functional checks for profile-selected optional model runtimes."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from autoforecast.model_runtime import check_runtime, runtime_ids


def test_profile_runtime_check_reports_missing_modules_without_importing_them() -> None:
    with mock.patch(
        "autoforecast.model_runtime.importlib.util.find_spec",
        side_effect=lambda name: None if name == "statsforecast" else object(),
    ):
        result = check_runtime("classical")
    assert not result.available
    assert result.missing_modules == ("statsforecast",)
    assert "uv sync --extra classical" in result.summary


def test_builtin_runtime_is_available_without_optional_packages() -> None:
    assert check_runtime("builtin").available
    assert {"builtin", "classical", "custom_torch"} <= set(runtime_ids())


@pytest.mark.parametrize(
    ("layout", "available"),
    [("absent", False), ("module", False), ("missing_child", False),
     ("package", True), ("namespace", True)],
)
def test_dotted_runtime_discovery_never_imports_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, available: bool,
) -> None:
    for name in tuple(sys.modules):
        if name == "autogluon" or name.startswith("autogluon."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    initializer = "raise RuntimeError('availability check imported model code')\n"
    parent = tmp_path / "autogluon"
    if layout == "module":
        parent.with_suffix(".py").write_text(initializer)
    elif layout != "absent":
        parent.mkdir()
        if layout != "namespace":
            (parent / "__init__.py").write_text(initializer)
        if layout in {"package", "namespace"}:
            child = parent / "timeseries"
            child.mkdir()
            (child / "__init__.py").write_text(initializer)

    result = check_runtime("autogluon_zoo")

    assert result.available is available
    assert result.missing_modules == (() if available else ("autogluon.timeseries",))
    if not available:
        assert "uv sync --extra autogluon-zoo" in result.summary
    assert not any(name == "autogluon" or name.startswith("autogluon.") for name in sys.modules)
