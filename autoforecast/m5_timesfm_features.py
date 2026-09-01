"""Materialize released M5 calendar/SNAP inputs without reading future sales as features."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from autoforecast.timesfm_hnb_source import file_sha256
from provenance.lineage import directory_digest


def calendar_features(path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    calendar = pd.read_csv(path)
    dates = pd.to_datetime(calendar["date"])
    if dates.duplicated().any():
        raise ValueError("M5 calendar has duplicate dates")
    result = pd.DataFrame({"date": dates.dt.date})
    for label, phase, period in (
        ("weekday", dates.dt.dayofweek, 7),
        ("monthday", dates.dt.day - 1, 31),
        ("yearday", dates.dt.dayofyear - 1, 365.25),
    ):
        result[f"cal_{label}_sin"] = np.sin(2 * np.pi * phase / period).astype("float32")
        result[f"cal_{label}_cos"] = np.cos(2 * np.pi * phase / period).astype("float32")
    result["cal_time"] = ((dates - pd.Timestamp("2011-01-29")).dt.days / 365.25).astype("float32")
    for field in ("event_name", "event_type"):
        vocabulary = sorted(
            set(calendar[f"{field}_1"].dropna()) | set(calendar[f"{field}_2"].dropna())
        )
        for value in vocabulary:
            name = "cal_" + field + "_" + re.sub("[^a-z0-9]+", "_", value.lower()).strip("_")
            if name in result:
                raise ValueError("M5 event vocabulary column collision")
            result[name] = (
                (calendar[f"{field}_1"] == value) | (calendar[f"{field}_2"] == value)
            ).astype("float32")
    shared = tuple(name for name in result if name != "date")
    for state in ("CA", "TX", "WI"):
        result[f"snap_{state}"] = calendar[f"snap_{state}"].astype("float32")
    return result, shared


def enrich_batch(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Many-to-one join preserves every input row, target, mask, and series/date identity."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    result = frame.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
        sort=False,
        indicator="_calendar_match",
    )
    if not result["_calendar_match"].eq("both").all():
        raise ValueError("M5 panel date is absent from the released calendar")
    if not result["state_id"].isin(["CA", "TX", "WI"]).all():
        raise ValueError("M5 panel has unknown SNAP state")
    result["snap"] = np.select(
        [result["state_id"].eq(state) for state in ("CA", "TX", "WI")],
        [result[f"snap_{state}"] for state in ("CA", "TX", "WI")],
    ).astype("float32")
    return result.drop(columns=["_calendar_match", "snap_CA", "snap_TX", "snap_WI"])


def materialize(source: Path, destination: Path, calendar_path: Path) -> dict:
    """Publish once with a provenance manifest; existing outputs must match every digest."""
    source_sha = directory_digest(str(source), suffixes=(".parquet",))
    identity = {
        "schema": "m5-timesfm-all-features.v1",
        "input_sha256": source_sha,
        "calendar_sha256": file_sha256(calendar_path),
    }
    manifest_path = destination / "_features.json"
    if destination.exists() or manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            any(manifest.get(k) != v for k, v in identity.items())
            or directory_digest(str(destination), suffixes=(".parquet",)) != manifest["sha256"]
        ):
            raise ValueError("Existing enriched M5 artifact differs from its provenance")
        return manifest
    calendar, shared = calendar_features(calendar_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.TemporaryDirectory(
        prefix=destination.name + "-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "panel"
        stage.mkdir()
        dataset = ds.dataset(source, format="parquet", partitioning="hive")
        for index, batch in enumerate(dataset.to_batches(batch_size=131072)):
            enriched = enrich_batch(batch.to_pandas(), calendar)
            table = pa.Table.from_pandas(enriched, preserve_index=False)
            pq.write_table(table, stage / f"part-{index:06d}.parquet", compression="zstd")
            total += len(enriched)
            if index % 100 == 0:
                print(f"M5 feature join: {source.name} rows={total:,}", flush=True)
        if (
            directory_digest(str(source), suffixes=(".parquet",)) != source_sha
            or file_sha256(calendar_path) != identity["calendar_sha256"]
        ):
            raise RuntimeError("M5 input changed during feature materialization")
        manifest = {
            **identity,
            "sha256": directory_digest(str(stage), suffixes=(".parquet",)),
            "rows": total,
            "shared_known_future": list(shared),
        }
        (stage / "_features.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(stage, destination)
    return manifest
