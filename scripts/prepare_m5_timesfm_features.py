#!/usr/bin/env python3
"""Prepare full released M5 inputs for the frozen/LoRA comparison; never trains."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from autoforecast.m5_timesfm_features import materialize

ROOT = pathlib.Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    for name in (
        "m5_public_training_d1913_v2",
        "m5_private_training_d1941_v2",
        "m5_organizer_private_replay_d1941_d1969_v1",
    ):
        result = materialize(
            ROOT / "data" / name, ROOT / "data" / (name + "_timesfm"), ROOT / "data/m5/calendar.csv"
        )
        print(json.dumps({"dataset": name, **result}), flush=True)
