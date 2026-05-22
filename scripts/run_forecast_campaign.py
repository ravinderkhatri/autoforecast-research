#!/usr/bin/env python3
"""Compile, run, recover, inspect, or prepare a watchdog for a forecasting campaign."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.campaign import CampaignError, CampaignStore, CampaignSupervisor  # noqa: E402
from autoforecast.campaign_launchd import prepare_launchd_service  # noqa: E402
from autoforecast.forecast_campaign import (  # noqa: E402
    ForecastCampaignError,
    compile_forecast_campaign,
)
from autoforecast.project import project_entrypoint, project_root

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _status(store: CampaignStore) -> dict[str, object]:
    return {
        "campaign_root": str(store.root),
        **store.state(),
        "supervisor": store.supervisor_observation(),
        "report": str(store.root / "stages/replay_report/attempt-0001/report.html"),
    }


@project_entrypoint
def main() -> int:
    root = project_root(ROOT)
    parser = argparse.ArgumentParser(description="Run an end-to-end forecasting campaign.")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "prepare-launchd"):
        command = commands.add_parser(action)
        command.add_argument("request", type=pathlib.Path)
        command.add_argument("--selection-data-path", type=pathlib.Path, required=True)
        command.add_argument("--final-training-data-path", type=pathlib.Path, required=True)
        command.add_argument("--final-score-data-path", type=pathlib.Path, required=True)
        command.add_argument(
            "--output-root",
            type=pathlib.Path,
            default=root / ".autoforecast/forecast-campaigns",
        )
        command.add_argument(
            "--compiled-root",
            type=pathlib.Path,
            default=root / ".autoforecast/forecast-campaign-specs",
            help="dedicated immutable compiled-artifact root below the source checkout",
        )
        if action == "run":
            command.add_argument("--poll-seconds", type=float, default=1.0)
        if action == "prepare-launchd":
            command.add_argument("--interval-seconds", type=int, default=3600)
    status = commands.add_parser("status")
    status.add_argument("campaign_root", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.action == "status":
            store = CampaignStore.from_root(args.campaign_root)
            result = _status(store)
        else:
            output_root = args.output_root.resolve()
            compiled_root = args.compiled_root.resolve()
            if not output_root.is_relative_to(root.resolve()):
                raise ForecastCampaignError(
                    "forecast campaign output must stay below the repository so generated "
                    "immutable child requests remain inside the governed source boundary"
                )
            if not compiled_root.is_relative_to(root.resolve()):
                raise ForecastCampaignError(
                    "forecast campaign compiled artifacts must stay below the repository"
                )
            compiled = compile_forecast_campaign(
                args.request,
                selection_data_path=args.selection_data_path,
                final_training_data_path=args.final_training_data_path,
                final_score_data_path=args.final_score_data_path,
                compiled_root=compiled_root,
            )
            for line in compiled.preflight_lines:
                print(f"[preflight] {line}", file=sys.stderr)
            store = CampaignStore.open_or_create(
                compiled.plan_path,
                output_root=output_root,
                working_directory=root,
            )
            result = {
                **_status(store),
                "compiled_request": str(compiled.directory),
                "plan": str(compiled.plan_path),
            }
            if args.action == "run":
                CampaignSupervisor(store, poll_seconds=args.poll_seconds).run()
                result = {**result, **_status(store)}
            elif args.action == "prepare-launchd":
                service = prepare_launchd_service(
                    store,
                    runner_path=root / "scripts/run_campaign.py",
                    interval_seconds=args.interval_seconds,
                )
                result["launchd"] = {
                    "label": service.label,
                    "plist": str(service.path),
                    "install": list(service.bootstrap_command),
                    "uninstall": list(service.bootout_command),
                }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {"planned", "running", "completed"} else 2
    except (CampaignError, ForecastCampaignError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
