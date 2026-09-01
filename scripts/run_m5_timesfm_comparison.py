#!/usr/bin/env python3
"""Run complete frozen TimesFM-HNB then LoRA V8 benchmarks, stopping if either campaign fails."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_NAMES = (
    "m5_public_training_d1913_v2",
    "m5_private_training_d1941_v2",
    "m5_organizer_private_replay_d1941_d1969_v1",
)


def run_campaign(action: str, request: Path, data_root: Path) -> dict:
    """Use exactly the compiler, supervisor and verified report reader used by the CLI."""
    sys.path.insert(0, str(ROOT))
    from autoforecast.campaign import CampaignStore, CampaignSupervisor
    from autoforecast.forecast_campaign import _completed_output, compile_forecast_campaign

    compiled = compile_forecast_campaign(
        request,
        selection_data_path=data_root / (DATA_NAMES[0] + "_timesfm"),
        final_training_data_path=data_root / (DATA_NAMES[1] + "_timesfm"),
        final_score_data_path=data_root / (DATA_NAMES[2] + "_timesfm"),
        compiled_root=ROOT / ".autoforecast/forecast-campaign-specs",
    )
    for line in compiled.preflight_lines:
        print(f"[preflight] {line}", flush=True)
    store = CampaignStore.open_or_create(
        compiled.plan_path,
        output_root=ROOT / ".autoforecast/forecast-campaigns",
        working_directory=ROOT,
    )
    if action == "prepare":
        return {"campaign_root": str(store.root), "status": store.state()["status"]}
    CampaignSupervisor(store, poll_seconds=1.0).run()
    if store.state()["status"] != "completed":
        raise RuntimeError(f"Campaign did not complete: {store.root}")
    report_path = _completed_output(store.root, "replay_report", "campaign-report.json")
    report = json.loads(report_path.read_text())
    score = report["final_score"]
    return {
        "model": score["model_id"],
        "metric": score["metric"],
        "score": score["score"],
        "report": str(report_path),
    }


def run_pair(action: str, data_root: Path, *, device: str, epochs: int = 2) -> None:
    """V8 starts only after TimesFM-HNB has completed all requested stages successfully."""
    results = []
    for version, label in (("hnb", "TimesFM-HNB"), ("v8", "V8")):
        print(f"=== {label} {action}: full M5 ===", flush=True)
        budget = "2epoch" if epochs == 2 else "50k"
        request = ROOT / f"configs/autoforecast/m5_timesfm_{version}_{budget}_{device}.yaml"
        results.append(run_campaign(action, request, data_root))
    print(json.dumps({"comparison": results}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    parser.add_argument("--epochs", type=int, choices=(2, 50000), default=2)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.action == "run":
        processes = subprocess.run(
            [
                "pgrep",
                "-f",
                "run_forecast_campaign.py|autoforecast.adapters.worker",
            ],
            capture_output=True,
            text=True,
        )
        if processes.returncode == 0:
            parser.error("A forecasting worker/campaign is active; inspect its process tree first.")
    environment = {"AUTOFORECAST_LIVE_PROGRESS": "1", "PYSPARK_PYTHON": sys.executable}
    os.environ.update(environment)
    run_pair(args.action, args.data_root.resolve(), device=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
