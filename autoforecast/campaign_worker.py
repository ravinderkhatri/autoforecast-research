"""Process entry point for one durable campaign stage."""

from __future__ import annotations

import argparse

from autoforecast.campaign import CampaignError, run_stage_worker


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one immutable AutoForecast campaign stage.")
    parser.add_argument("request")
    args = parser.parse_args()
    try:
        return run_stage_worker(args.request)
    except CampaignError as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
