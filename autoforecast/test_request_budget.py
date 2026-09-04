"""Bound how often a search reads each trial file from a Unity Catalog volume.

The fake volume serves the Files API from the real filesystem, so Spark, mounted paths
and native API calls see the same bytes, and every API download is counted per file.
"""

import collections
import io
import json
import os
import pathlib
import threading
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from databricks.sdk.errors import AlreadyExists, NotFound
from mlops import _volume_files

from autoforecast import _logical_hash, experiments
from autoforecast.panel import Forecast
from autoforecast.study import load_study
from autoforecast.study_tuning import FitPredictResult, tune_study
from autoforecast.test_study_tuning import config, freeze, implementation, spark  # noqa: F401

MODEL_FILES = 30


class LocalFiles:
    """Files API over the local filesystem, counting downloads per path."""

    def __init__(self):
        self.downloads = collections.Counter()
        self.lock = threading.Lock()
        self.files = self
        self.api_client = self
        self._cfg = SimpleNamespace(workspace_id=None)

    def do(self, method, path, *, query, headers, data):
        from urllib.parse import unquote

        assert method == "PUT"
        target = pathlib.Path(unquote(path.removeprefix("/api/2.0/fs/files")))
        value = data.read() if hasattr(data, "read") else data
        with self.lock:
            if target.exists() and not query["overwrite"]:
                raise AlreadyExists("file exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)

    def download(self, path):
        with self.lock:
            self.downloads[path] += 1
        try:
            return SimpleNamespace(contents=io.BytesIO(pathlib.Path(path).read_bytes()))
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as error:
            raise NotFound("file missing") from error

    def get_metadata(self, path):
        target = pathlib.Path(path)
        if not target.is_file():
            raise NotFound("file missing")
        status = target.stat()
        return SimpleNamespace(content_length=status.st_size, last_modified=status.st_mtime_ns)

    def get_directory_metadata(self, path):
        if not pathlib.Path(path).is_dir():
            raise NotFound("directory missing")

    def list_directory_contents(self, path):
        entries = []
        for entry in os.scandir(path):
            directory = entry.is_dir(follow_symlinks=False)
            status = entry.stat(follow_symlinks=False)
            entries.append(SimpleNamespace(
                path=entry.path, is_directory=directory,
                file_size=None if directory else status.st_size,
                last_modified=status.st_mtime_ns,
            ))
        return entries

    def delete(self, path):
        pathlib.Path(path).unlink()

    def delete_directory(self, path):
        pathlib.Path(path).rmdir()


def many_files(request):
    """A cheap model that saves one small state file per series, like the Prophet notebook."""
    rows = request.series_mapping.collect()
    forecasts = request.output_dir / "predictions"
    forecasts.mkdir()
    pq.write_table(pa.Table.from_pylist([
        {"series_id": row[request.panel.series_col], "date": day,
         "yhat": request.model_parameters["level"] + request.training_parameters["offset"]}
        for row in rows for day in request.forecast_dates
    ]), forecasts / "part.parquet")
    model = request.output_dir / "model"
    model.mkdir()
    for number in range(MODEL_FILES):
        (model / f"series-{number:03d}.json").write_text(json.dumps(request.model_parameters))
    return FitPredictResult(Forecast(
        adapter=request.model_id, path=str(forecasts), origin=request.origin,
        horizon=len(request.forecast_dates), n_series=len(rows),
        disposition={"emitted": len(rows) * len(request.forecast_dates)},
        target_estimand=request.panel.target_estimand, frequency=request.panel.frequency,
    ), "many-json-v1")


def test_search_reads_each_trial_file_a_bounded_number_of_times(spark, tmp_path, monkeypatch):  # noqa: F811
    root = tmp_path.resolve()
    manifest = freeze(root / "study", spark, windows="multiple")
    files = LocalFiles()
    monkeypatch.setattr(_volume_files, "_client", lambda: files)
    monkeypatch.setattr(
        _volume_files, "is_volume",
        lambda path: pathlib.Path(os.path.abspath(path)).is_relative_to(root),
    )
    full_checks = []
    check = experiments._load_completed_attempt

    def counted(path):
        full_checks.append(path)
        return check(path)

    monkeypatch.setattr(experiments, "_load_completed_attempt", counted)
    recalculated = []
    fingerprint = _logical_hash.fingerprint

    def recounted(*args, **kwargs):
        recalculated.append(args[0])
        return fingerprint(*args, **kwargs)

    monkeypatch.setattr(_logical_hash, "fingerprint", recounted)
    scratch = root / "scratch"
    scratch.mkdir()
    profile = config(root, budget=2)
    # Score every tuning window, as the Prophet notebook's profile does.
    value = yaml.safe_load(profile.read_text())
    value["selection"] = {"stage": "tuning", "origins": "all", "aggregation": "mean"}
    profile.write_text(yaml.safe_dump(value))
    study = load_study(manifest.parent.name, registry_root=manifest.parent.parent,
                       spark=spark, scratch_root=scratch)
    # Count only the search's work, not loading the study, which checks every file in full.
    files.downloads.clear()
    recalculated.clear()
    with study:
        result = tune_study(study, fit_predict=many_files, config=profile,
                            implementation=implementation(), output_dir=root / "runs")
    trials = result.path / "trials"
    assert trials.is_relative_to(root)
    # Selection checks each trial in full once; the report and every reload reuse it.
    assert len(full_checks) == 2
    # Reloads check every frozen file's bytes and reuse the recorded row identities.
    assert recalculated == []
    reads = {pathlib.Path(path): count for path, count in files.downloads.items()
             if pathlib.Path(path).is_relative_to(trials)}
    models = {path: count for path, count in reads.items() if path.parent.name == "model"}
    assert len(models) == 2 * 2 * MODEL_FILES
    # Each model file: read back when published, checked at the trial end, hashed for the
    # manifest and checked once by selection; the winner's are hashed again on reload.
    assert max(models.values()) <= 5
    # Terminal records are re-read whenever a reused check confirms them. Outside a run,
    # stage events use the locked JSONL append; a run writes event segments instead.
    terminal = {"completion.json", "artifact_manifest.json"}
    others = sorted(((count, str(path.relative_to(trials))) for path, count in reads.items()
                     if path.name not in terminal and not path.name.startswith("events.jsonl")),
                    reverse=True)
    # Small JSON records are read by every selection and reload step. A winning window's
    # forecasts: read back when published, hashed and copied for scoring, checked at the
    # trial end, hashed for the manifest, checked by selection and hashed again on reload.
    records = [item for item in others if item[1].endswith(".json")]
    data = [item for item in others if not item[1].endswith(".json")]
    assert records[0][0] <= 8, records[:6]
    assert data[0][0] <= 7, data[:6]
    # Frozen study data is checked in full by the preflight, at the search's start and on
    # reload, and otherwise only where a fit or its scoring reads it; never around windows.
    frozen = {str(pathlib.Path(path).relative_to(manifest.parent)): count
              for path, count in files.downloads.items()
              if pathlib.Path(path).is_relative_to(manifest.parent) and path.endswith(".parquet")}
    unused = {name: count for name, count in frozen.items()
              if name.startswith(("final_", "grids/validation", "grids/test",
                                  "coverage/validation", "coverage/test"))}
    assert unused and max(unused.values()) <= 5, unused
    assert max(frozen.values()) <= 11, frozen
