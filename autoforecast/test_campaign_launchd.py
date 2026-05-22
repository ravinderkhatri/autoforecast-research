"""Checks for the opt-in launchd campaign watchdog artifact."""

from __future__ import annotations

import os
import pathlib
import plistlib
import sys

import pytest
from scripts import run_campaign

from autoforecast.campaign import CampaignError, CampaignStore
from autoforecast.campaign_launchd import prepare_launchd_service
from autoforecast.test_campaign import ROOT, _plan, _stage


def test_launchd_service_checks_status_only_without_starting_training(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
    monkeypatch.setenv("PATH", "/opt/homebrew/opt/openjdk@17/bin:/usr/bin:/bin")
    store = CampaignStore.open_or_create(
        _plan(tmp_path, [_stage("train", "pass")]),
        output_root=tmp_path / "runs",
        working_directory=ROOT,
    )
    runner = ROOT / "scripts/run_campaign.py"
    service = prepare_launchd_service(store, runner_path=runner)
    record = plistlib.loads(service.path.read_bytes())

    assert record["StartInterval"] == 3600
    assert record["RunAtLoad"] is True
    assert "KeepAlive" not in record
    assert record["ProcessType"] == "Background"
    assert record["ProgramArguments"] == [
        sys.executable, str(runner), "status", str(store.root)
    ]
    assert record["EnvironmentVariables"] == {
        "JAVA_HOME": "/opt/homebrew/opt/openjdk@17",
        "PATH": "/opt/homebrew/opt/openjdk@17/bin:/usr/bin:/bin",
        "PYSPARK_DRIVER_PYTHON": sys.executable,
        "PYSPARK_PYTHON": sys.executable,
    }
    assert service.bootstrap_command == (
        "launchctl",
        "bootstrap",
        f"gui/{os.getuid()}",
        str(service.path),
    )
    assert service.bootout_command == (
        "launchctl",
        "bootout",
        f"gui/{os.getuid()}/{service.label}",
    )
    assert service.path.is_file()

    same = prepare_launchd_service(store, runner_path=runner)
    assert same == service
    with pytest.raises(CampaignError, match="differs from the frozen campaign"):
        prepare_launchd_service(store, runner_path=runner, interval_seconds=7200)


def test_launchd_service_refuses_an_existing_background_training_plist(
    tmp_path: pathlib.Path,
) -> None:
    store = CampaignStore.open_or_create(
        _plan(tmp_path, [_stage("train", "pass")]),
        output_root=tmp_path / "runs",
        working_directory=ROOT,
    )
    service = prepare_launchd_service(store, runner_path=ROOT / "scripts/run_campaign.py")
    legacy = plistlib.loads(service.path.read_bytes())
    legacy["ProgramArguments"] = [
        sys.executable, str(ROOT / "scripts/run_campaign.py"), "run",
        str(store.root / "plan-source.yaml"),
    ]
    service.path.write_bytes(plistlib.dumps(legacy, fmt=plistlib.FMT_XML, sort_keys=True))

    with pytest.raises(CampaignError, match="differs from the frozen campaign"):
        prepare_launchd_service(store, runner_path=ROOT / "scripts/run_campaign.py")
    assert plistlib.loads(service.path.read_bytes())["ProgramArguments"][2] == "run"


@pytest.mark.parametrize("interval", [0, 59, 60.0, True])
def test_launchd_interval_is_a_built_in_integer_of_at_least_one_minute(
    tmp_path: pathlib.Path, interval: object
) -> None:
    store = CampaignStore.open_or_create(
        _plan(tmp_path, [_stage("train", "pass")]),
        output_root=tmp_path / "runs",
        working_directory=ROOT,
    )
    with pytest.raises(CampaignError, match="integer >= 60"):
        prepare_launchd_service(
            store,
            runner_path=ROOT / "scripts/run_campaign.py",
            interval_seconds=interval,  # type: ignore[arg-type]
        )


def test_launchd_refuses_a_symlinked_runner(tmp_path: pathlib.Path) -> None:
    store = CampaignStore.open_or_create(
        _plan(tmp_path, [_stage("train", "pass")]),
        output_root=tmp_path / "runs",
        working_directory=ROOT,
    )
    runner = tmp_path / "runner-link.py"
    runner.symlink_to(ROOT / "scripts/run_campaign.py")
    with pytest.raises(CampaignError, match="non-symlink"):
        prepare_launchd_service(store, runner_path=runner)


def test_launchd_refuses_a_dangling_symlink_at_the_plist_path(tmp_path: pathlib.Path) -> None:
    store = CampaignStore.open_or_create(
        _plan(tmp_path, [_stage("train", "pass")]),
        output_root=tmp_path / "runs",
        working_directory=ROOT,
    )
    path = store.root / "supervisor.launchd.plist"
    path.symlink_to(tmp_path / "missing.plist")

    with pytest.raises(CampaignError, match="differs from the frozen campaign"):
        prepare_launchd_service(store, runner_path=ROOT / "scripts/run_campaign.py")


def test_watchdog_run_stops_relaunching_a_terminal_failed_campaign(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, [_stage("train", "pass")])
    arguments = [
        "run_campaign.py",
        "run",
        str(plan),
        "--output-root",
        str(tmp_path / "runs"),
        "--working-directory",
        str(ROOT),
    ]
    monkeypatch.setattr(
        run_campaign.CampaignSupervisor,
        "run",
        lambda _self: {"status": "failed", "stages": []},
    )
    monkeypatch.setattr(sys, "argv", [*arguments, "--watchdog"])
    assert run_campaign.main() == 0
    monkeypatch.setattr(sys, "argv", arguments)
    assert run_campaign.main() == 2
