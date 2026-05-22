#!/usr/bin/env python3
"""Create, run, resume, or inspect a local AutoForecast campaign."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.campaign import (  # noqa: E402
    CampaignError,
    CampaignStore,
    CampaignSupervisor,
)
from autoforecast.campaign_launchd import prepare_launchd_service  # noqa: E402


def _render(state: dict[str, object], store: CampaignStore) -> None:
    print(
        json.dumps(
            {
                "campaign_root": str(store.root),
                **state,
                "supervisor": store.supervisor_observation(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a durable local AutoForecast campaign.")
    commands = parser.add_subparsers(dest="action", required=True)

    run = commands.add_parser("run", help="create or resume a campaign until terminal")
    run.add_argument("plan", type=pathlib.Path)
    run.add_argument("--output-root", type=pathlib.Path, required=True)
    run.add_argument("--working-directory", type=pathlib.Path, default=pathlib.Path.cwd())
    run.add_argument("--poll-seconds", type=float, default=1.0)
    run.add_argument(
        "--watchdog",
        action="store_true",
        help="exit successfully for terminal failed campaigns so launchd stops relaunching them",
    )

    status = commands.add_parser("status", help="read a campaign without executing work")
    status.add_argument("campaign_root", type=pathlib.Path)

    launchd = commands.add_parser(
        "prepare-launchd",
        help="generate, but do not install, a macOS watchdog for one campaign",
    )
    launchd.add_argument("plan", type=pathlib.Path)
    launchd.add_argument("--output-root", type=pathlib.Path, required=True)
    launchd.add_argument("--working-directory", type=pathlib.Path, default=pathlib.Path.cwd())
    launchd.add_argument("--interval-seconds", type=int, default=3600)

    args = parser.parse_args()
    try:
        if args.action == "status":
            store = CampaignStore.from_root(args.campaign_root)
            state = store.state()
        elif args.action == "prepare-launchd":
            store = CampaignStore.open_or_create(
                args.plan,
                output_root=args.output_root,
                working_directory=args.working_directory,
            )
            service = prepare_launchd_service(
                store,
                runner_path=pathlib.Path(__file__),
                interval_seconds=args.interval_seconds,
            )
            print(
                json.dumps(
                    {
                        "campaign_root": str(store.root),
                        "label": service.label,
                        "plist": str(service.path),
                        "install": list(service.bootstrap_command),
                        "uninstall": list(service.bootout_command),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        else:
            store = CampaignStore.open_or_create(
                args.plan,
                output_root=args.output_root,
                working_directory=args.working_directory,
            )
            state = CampaignSupervisor(store, poll_seconds=args.poll_seconds).run()
    except CampaignError as exc:
        parser.error(str(exc))
    _render(state, store)
    terminal_for_watchdog = getattr(args, "watchdog", False) and state["status"] == "failed"
    return 0 if state["status"] == "completed" or terminal_for_watchdog else 2


if __name__ == "__main__":
    raise SystemExit(main())
