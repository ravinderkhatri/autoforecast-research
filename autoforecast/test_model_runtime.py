"""Functional checks for profile-selected optional model runtimes."""

from __future__ import annotations

from unittest import mock

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
