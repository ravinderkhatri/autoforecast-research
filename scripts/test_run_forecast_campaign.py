"""CLI contract tests for governed forecast-campaign compilation roots."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

MODULE_PATH = pathlib.Path(__file__).with_name("run_forecast_campaign.py")
SPEC = importlib.util.spec_from_file_location("run_forecast_campaign", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class _Store:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root

    def state(self) -> dict[str, object]:
        return {"status": "planned"}

    def supervisor_observation(self) -> dict[str, object]:
        return {}


def test_prepare_forwards_a_dedicated_compiled_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    request = root / "request.yaml"
    compiled_root = root / ".autoforecast/compiled"
    output_root = root / ".autoforecast/runs"
    observed: dict[str, object] = {}

    class _Compiled:
        directory = compiled_root / "request-artifacts"
        plan_path = directory / "campaign-plan.yaml"
        preflight_lines: tuple[str, ...] = ()

    def compile_request(*_args: object, **kwargs: object) -> _Compiled:
        observed.update(kwargs)
        return _Compiled()

    monkeypatch.setattr(subject, "ROOT", root)
    monkeypatch.setattr(subject, "compile_forecast_campaign", compile_request)
    monkeypatch.setattr(
        subject.CampaignStore,
        "open_or_create",
        lambda *_args, **kwargs: _Store(pathlib.Path(kwargs["output_root"])),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_forecast_campaign.py",
            "prepare",
            str(request),
            "--selection-data-path",
            str(root / "selection"),
            "--final-training-data-path",
            str(root / "training"),
            "--final-score-data-path",
            str(root / "score"),
            "--output-root",
            str(output_root),
            "--compiled-root",
            str(compiled_root),
        ],
    )

    assert subject.main() == 0
    assert observed["compiled_root"] == compiled_root.resolve()


def test_prepare_rejects_an_external_output_before_compilation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    compiled = False

    def compile_request(*_args: object, **_kwargs: object) -> object:
        nonlocal compiled
        compiled = True
        return object()

    monkeypatch.setattr(subject, "ROOT", root)
    monkeypatch.setattr(subject, "compile_forecast_campaign", compile_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_forecast_campaign.py",
            "prepare",
            str(root / "request.yaml"),
            "--selection-data-path",
            str(root / "selection"),
            "--final-training-data-path",
            str(root / "training"),
            "--final-score-data-path",
            str(root / "score"),
            "--output-root",
            str(tmp_path / "outside"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        subject.main()
    assert not compiled


def test_prepare_rejects_an_external_compiled_root_before_compilation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    compiled = False

    def compile_request(*_args: object, **_kwargs: object) -> object:
        nonlocal compiled
        compiled = True
        return object()

    monkeypatch.setattr(subject, "ROOT", root)
    monkeypatch.setattr(subject, "compile_forecast_campaign", compile_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_forecast_campaign.py",
            "prepare",
            str(root / "request.yaml"),
            "--selection-data-path",
            str(root / "selection"),
            "--final-training-data-path",
            str(root / "training"),
            "--final-score-data-path",
            str(root / "score"),
            "--output-root",
            str(root / ".autoforecast/runs"),
            "--compiled-root",
            str(tmp_path / "outside-compiled"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        subject.main()
    assert not compiled
