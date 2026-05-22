"""Generate an opt-in macOS launchd status check for one frozen local campaign."""

from __future__ import annotations

import os
import pathlib
import plistlib
import sys
from dataclasses import dataclass

from autoforecast.campaign import CampaignError, CampaignStore
from autoforecast.protocol import _atomic_write_text

__all__ = ["LaunchdService", "prepare_launchd_service"]


@dataclass(frozen=True)
class LaunchdService:
    """The generated plist and exact commands for explicit activation/removal."""

    label: str
    path: pathlib.Path

    @property
    def bootstrap_command(self) -> tuple[str, ...]:
        return ("launchctl", "bootstrap", f"gui/{os.getuid()}", str(self.path))

    @property
    def bootout_command(self) -> tuple[str, ...]:
        return ("launchctl", "bootout", f"gui/{os.getuid()}/{self.label}")


def prepare_launchd_service(
    store: CampaignStore,
    *,
    runner_path: str | pathlib.Path,
    interval_seconds: int = 3600,
) -> LaunchdService:
    """Write a status-only plist; background launchd must never start training."""
    if type(interval_seconds) is not int or interval_seconds < 60:
        raise CampaignError("launchd watchdog interval_seconds must be an integer >= 60")
    supplied_runner = pathlib.Path(runner_path).expanduser()
    if supplied_runner.is_symlink():
        raise CampaignError(
            f"campaign runner must be a regular non-symlink file: {supplied_runner}"
        )
    runner = supplied_runner.resolve()
    if not runner.is_file():
        raise CampaignError(f"campaign runner must be a regular non-symlink file: {runner}")
    label = f"com.autoforecast.campaign.{store.plan.name}.{store.plan.source_sha256[:12]}"
    path = store.root / "supervisor.launchd.plist"
    runtime_environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "PYSPARK_DRIVER_PYTHON": sys.executable,
        "PYSPARK_PYTHON": sys.executable,
    }
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        runtime_environment["JAVA_HOME"] = java_home

    def payload_for() -> bytes:
        arguments = [
            sys.executable,
            str(runner),
            "status",
            str(store.root),
        ]
        payload: dict[str, object] = {
            "Label": label,
            "ProgramArguments": arguments,
            "RunAtLoad": True,
            "StartInterval": interval_seconds,
            "ThrottleInterval": 30,
            "ProcessType": "Background",
            "StandardOutPath": str(store.root / "supervisor.stdout.log"),
            "StandardErrorPath": str(store.root / "supervisor.stderr.log"),
            "WorkingDirectory": str(store.working_directory),
        }
        payload["EnvironmentVariables"] = runtime_environment
        return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)

    payload = payload_for()
    if path.is_symlink():
        raise CampaignError("existing launchd service differs from the frozen campaign")
    if path.exists():
        if not path.is_file():
            raise CampaignError("existing launchd service differs from the frozen campaign")
        if path.read_bytes() != payload:
            raise CampaignError("existing launchd service differs from the frozen campaign")
    else:
        _atomic_write_text(path, payload.decode("utf-8"))
    return LaunchdService(label, path)
