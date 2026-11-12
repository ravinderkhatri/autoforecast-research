"""Bound how often a run reads each saved file from a Unity Catalog volume.

Inside a run each saved file is downloaded once per stage, apart from the read-back that
confirms a write; each stage ends with one fresh check (``_volume_files.fresh_reads``).

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
import pytest
import yaml
from databricks.sdk.errors import AlreadyExists, NotFound
from mlops import _safe_files, _volume_files

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
        self.listings = collections.Counter()
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
        with self.lock:
            self.listings[path] += 1
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
    files.listings.clear()
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
    # Each model file: read back when published, read once for the trial end and its
    # manifest, then once by selection's final check; the winner's once more by the
    # search's final check, which starts after selection's.
    assert max(models.values()) <= 4
    # Terminal records are re-read whenever a reused check confirms them. Outside a run,
    # stage events use the locked JSONL append; a run writes event segments instead.
    terminal = {"completion.json", "artifact_manifest.json"}
    others = sorted(((count, str(path.relative_to(trials))) for path, count in reads.items()
                     if path.name not in terminal and not path.name.startswith("events.jsonl")),
                    reverse=True)
    # Within the search each file is read once; selection's and the search's final checks
    # read it again. A winning window's forecasts: read back when published, read once
    # for scoring, the trial end and the manifest, then by the two final checks.
    records = [item for item in others if item[1].endswith(".json")]
    data = [item for item in others if not item[1].endswith(".json")]
    assert records[0][0] <= 4, records[:6]
    assert data[0][0] <= 4, data[:6]
    # Frozen study data is read once in the search (the coverage check, the fits and their
    # scoring share it), again after the coverage check's Spark reads, and by the
    # search's final check: three times, never once per window or fit.
    frozen = {str(pathlib.Path(path).relative_to(manifest.parent)): count
              for path, count in files.downloads.items()
              if pathlib.Path(path).is_relative_to(manifest.parent) and path.endswith(".parquet")}
    unused = {name: count for name, count in frozen.items()
              if name.startswith(("final_", "grids/validation", "grids/test",
                                  "coverage/validation", "coverage/test"))}
    assert unused and max(unused.values()) <= 3, unused
    assert max(frozen.values()) <= 3, frozen
    listed = {str(pathlib.Path(path).relative_to(trials)): count
              for path, count in files.listings.items() if pathlib.Path(path).is_relative_to(trials)}
    # Folders are listed when their files are copied, hashed or checked; finding the
    # attempts lists only the trial and attempt levels, never the whole search tree.
    assert sum(listed.values()) <= 175, sorted(listed.items(), key=lambda item: -item[1])[:8]


def _volume(root, monkeypatch):
    files = LocalFiles()
    monkeypatch.setattr(_volume_files, "_client", lambda: files)
    monkeypatch.setattr(
        _volume_files, "is_volume",
        lambda path: pathlib.Path(os.path.abspath(path)).is_relative_to(root),
    )
    return files


@pytest.mark.parametrize("file_checks", ["once", "every_stage"])
def test_run_study_downloads_and_hashes_each_saved_file_once(spark, tmp_path, monkeypatch,  # noqa: F811
                                                             file_checks):
    """By default each saved file is downloaded and hashed once per run; every_stage
    checks do that once per stage."""
    """A complete run on a volume reads each saved file once per stage.

    A stage ends with one fresh check (``fresh_reads``): after the coverage check's Spark
    reads, at the end of the search, at the end of the workflow. Inside a stage no saved
    file is downloaded twice, apart from the read-back that confirms each write.
    """
    import contextlib
    from urllib.parse import unquote

    import mlflow

    from autoforecast import run_mlflow, run_study, study_tuning, study_workflow
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import predict

    root = tmp_path.resolve()
    files = _volume(root, monkeypatch)
    stage, downloads, uploads = [0], collections.Counter(), collections.Counter()
    download, put = files.download, files.do

    def counted_download(path):
        downloads[(path, stage[0])] += 1
        return download(path)

    def counted_put(method, path, **kwargs):
        uploads[(unquote(path.removeprefix("/api/2.0/fs/files")), stage[0])] += 1
        return put(method, path, **kwargs)

    files.download, files.do = counted_download, counted_put
    fresh = _volume_files.fresh_reads

    @contextlib.contextmanager
    def staged():
        reads = _volume_files._FROZEN.get()
        if reads is not None and not reads.depth:
            stage[0] += 1
        with fresh():
            yield

    monkeypatch.setattr(_volume_files, "fresh_reads", staged)
    from provenance import lineage

    hashed = collections.Counter()
    for name in ("_directory_digest_bytes", "_directory_content_digest_bytes"):
        digest = getattr(lineage, name)
        monkeypatch.setattr(lineage, name, lambda path, suffixes, _digest=digest: (
            hashed.update([(os.path.abspath(path), stage[0])]) or _digest(path, suffixes)))
    identity = _safe_files._regular_file_identity

    def counted_identity(path):
        # Inside a run a volume path only finds its local copy; local bytes are hashed.
        if not _volume_files.is_volume(pathlib.Path(path)):
            hashed[(os.path.abspath(path), stage[0])] += 1
        return identity(path)

    monkeypatch.setattr(_safe_files, "_regular_file_identity", counted_identity)
    coverage_checks = []
    for owner in (study_workflow, study_tuning):
        check = owner.preflight_study
        monkeypatch.setattr(owner, "preflight_study",
                            lambda *a, _check=check, **k: coverage_checks.append(1) or _check(*a, **k))
    path = configuration(root / "config")
    value = yaml.safe_load(path.read_text())
    # Two tuning windows: the search fits every one, so files are reused across windows.
    value["recipe"]["tuning_windows"] = [{"origin": "2024-01-02", "horizon": 2},
                                         {"origin": "2024-01-03", "horizon": 2}]
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry)
    profile = config(root, budget=2)
    settings = yaml.safe_load(profile.read_text())
    settings["selection"] = {"stage": "tuning", "origins": "all", "aggregation": "mean"}
    profile.write_text(yaml.safe_dump(settings))
    uri = "sqlite:///" + str(root / "tracking.db")
    mlflow.tracking.MlflowClient(tracking_uri=uri).create_experiment(
        "budget", artifact_location=str(root / "mlflow-artifacts"))
    monkeypatch.setattr(run_mlflow, "_mlflow", lambda: mlflow)
    downloads.clear()
    uploads.clear()
    hashed.clear()
    choice = {} if file_checks == "once" else {"file_checks": file_checks}
    result = run_study(
        study_info={"study_id": study_id, "create_study": False, "publish_results": False,
                    **choice},
        registry=registry, model=predict, model_config=profile, implementation=implementation(),
        tracking="mlflow", tracking_uri=uri, mlflow_experiment="budget",
    )
    assert result.testing_metrics is not None
    # The run records which checks it made.
    from autoforecast import read_execution_events

    audit = next((registry.root / "run-attempts").glob("attempt-*.jsonl"))
    recorded = json.dumps([event for event in read_execution_events(audit)
                           if event.get("stage") == "configuration"])
    assert f'"file_checks": "{file_checks}"' in recorded, recorded
    # The workflow hands its coverage check to the search instead of repeating it.
    assert coverage_checks == [1]
    # Audit events are written and acknowledged one file at a time, so they are left out.
    saved = {key: count for key, count in downloads.items() if ".jsonl" not in key[0]}
    study = str(registry.root / "benchmarks" / study_id) + os.sep
    reads = collections.Counter()
    for (name, _stage), count in saved.items():
        if name.startswith(study):
            reads[name] += count
    if file_checks == "once":
        # One download per file for the whole run: a write's read-back is that download.
        per_file = collections.Counter()
        for (name, _stage), count in saved.items():
            per_file[name] += count
        assert max(per_file.values()) == 1, per_file.most_common(5)
        assert {name for (name, _), count in hashed.items() if count > 1} == set()
        per_copy = collections.Counter()
        for (name, _stage), count in hashed.items():
            per_copy[name] += count
        assert max(per_copy.values()) == 1, per_copy.most_common(5)
        assert reads and set(reads.values()) == {1}, reads
    else:
        # Once per stage, plus one read-back per write.
        extra = {key: count for key, count in saved.items() if count > 1 + uploads[key]}
        assert extra == {}, sorted(extra.items())[:10]
        assert max(hashed.values()) == 1, hashed.most_common(5)
        # Each frozen study file is read when the run loads it, after the coverage check's
        # Spark reads, at the end of the search and at the end of the workflow.
        assert reads and set(reads.values()) == {4}, reads


@pytest.mark.parametrize("target", ["window input", "earlier forecast", "study grid",
                                    "scoring copy", "sealed trial"])
def test_search_final_check_catches_damage_between_fits(spark, tmp_path, monkeypatch,  # noqa: F811
                                                        target):
    """With every-stage checks nothing is re-checked between fits, and the search's final
    check finds any change."""
    from autoforecast.test_study_tuning import predict

    root = tmp_path.resolve()
    manifest = freeze(root / "study", spark, windows="multiple")
    _volume(root, monkeypatch)
    scratch = root / "scratch"
    scratch.mkdir()
    profile = config(root, budget=2)
    value = yaml.safe_load(profile.read_text())
    value["selection"] = {"stage": "tuning", "origins": "all", "aggregation": "mean"}
    profile.write_text(yaml.safe_dump(value))
    calls = []

    def rewrite(part):
        # The same rows with another compression: later fits still read them, but the
        # file's bytes differ, which every byte-level check must notice.
        pq.write_table(pq.read_table(part), part, compression="gzip")
        # A local Spark output keeps a checksum beside each part; volumes have none.
        (part.parent / f".{part.name}.crc").unlink(missing_ok=True)

    def damaging(request):
        result = predict(request)
        calls.append(request)
        if target == "sealed trial" and len(calls) == 3:
            # The first trial is sealed. Read it once in this stage, then change it:
            # selection's final check must still read fresh bytes.
            attempt = next((root / "runs").rglob(
                "trials/trial-0000/attempts/*/completion.json")).parent
            experiments.load_completed_attempt(attempt)
            guide = attempt / "RUN.md"
            guide.write_text(guide.read_text() + "\nchanged\n")
        elif target != "sealed trial" and len(calls) == 2:
            parts = {
                "window input": request.training_data.inputFiles(),
                "earlier forecast": [str(p) for p in (root / "runs").rglob("windows/0000/forecasts/*.parquet")],
                "study grid": [str(p) for p in manifest.parent.glob("grids/tuning/*/*/*/*.parquet")],
                "scoring copy": [str(p) for p in scratch.rglob("scoring-inputs/*/*.parquet")],
            }[target]
            assert parts, target
            for part in parts:
                rewrite(pathlib.Path(part.removeprefix("file:")))
        return result

    study = load_study(manifest.parent.name, registry_root=manifest.parent.parent,
                       spark=spark, scratch_root=scratch)
    with study, pytest.raises(Exception, match="changed|does not match") as caught:
        tune_study(study, fit_predict=damaging, config=profile,
                   implementation=implementation(), output_dir=root / "runs",
                   file_checks="every_stage")
    # The search ran every fit: the change was found by its final check, before any
    # result was used, not by a check between fits.
    assert len(calls) == 4, (len(calls), caught.value)


def test_checks_once_trust_the_bytes_they_checked(spark, tmp_path, monkeypatch):  # noqa: F811
    """By default a run checks the study once and keeps using those checked bytes.

    A change to the saved study during the run is not looked for again in that run; the
    next load of the study finds it.
    """
    from autoforecast.test_study_tuning import predict

    root = tmp_path.resolve()
    manifest = freeze(root / "study", spark, windows="multiple")
    _volume(root, monkeypatch)
    scratch = root / "scratch"
    scratch.mkdir()
    profile = config(root, budget=2)
    value = yaml.safe_load(profile.read_text())
    value["selection"] = {"stage": "tuning", "origins": "all", "aggregation": "mean"}
    profile.write_text(yaml.safe_dump(value))
    calls = []

    def damaging(request):
        result = predict(request)
        calls.append(request)
        if len(calls) == 2:
            for part in manifest.parent.glob("grids/tuning/*/*/*/*.parquet"):
                pq.write_table(pq.read_table(part), part, compression="gzip")
                (part.parent / f".{part.name}.crc").unlink(missing_ok=True)
        return result

    study = load_study(manifest.parent.name, registry_root=manifest.parent.parent,
                       spark=spark, scratch_root=scratch)
    with study:
        result = tune_study(study, fit_predict=damaging, config=profile,
                            implementation=implementation(), output_dir=root / "runs")
    assert len(calls) == 4 and result.parameters is not None
    with pytest.raises(Exception, match="changed|does not match"):
        load_study(manifest.parent.name, registry_root=manifest.parent.parent,
                   spark=spark, scratch_root=scratch)


def test_registration_repeats_only_the_checks_around_its_hook(spark, tmp_path, monkeypatch):  # noqa: F811
    """Registration reads each file once, apart from the checks that bracket user code.

    The source capture and the staging folders are checked again around the ``prepare``
    hook and around Spark's reads of them (their own tests cover each change those
    checks catch). Everything else registration writes is read once, plus one read-back
    per write; the published study and its admission record are checked once.
    """
    import dataclasses
    import datetime as dt
    from urllib.parse import unquote

    from pyspark.sql import functions as F

    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    class WithReferences(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            windows = context.observations.sparkSession.createDataFrame(
                [(stage, origin) for stage, origin, _ in context.settings.windows],
                "stage string, origin string")
            refs = result.retained_keys.select(F.struct("item").alias("keys")).crossJoin(windows)
            refs = refs.withColumn("weight", F.lit(1.0)).withColumn("scale", F.lit(1.0))
            return dataclasses.replace(result, scoring_references=refs)

    root = tmp_path.resolve()
    files = _volume(root, monkeypatch)
    downloads, uploads = collections.Counter(), collections.Counter()
    download, put = files.download, files.do

    def counted_download(path):
        downloads[path] += 1
        return download(path)

    def counted_put(method, path, **kwargs):
        uploads[unquote(path.removeprefix("/api/2.0/fs/files"))] += 1
        return put(method, path, **kwargs)

    files.download, files.do = counted_download, counted_put
    path = configuration(root / "config")
    pq.write_table(pa.Table.from_pylist([
        {"item": f"sensor-{sid:03d}", "day": dt.date(2024, 1, day), "value": float(day)}
        for sid in range(100) for day in range(1, 10)]), root / "config" / "source.parquet")
    value = yaml.safe_load(path.read_text())
    value["preparation"] = {"exclude_count": 5}
    value["references"] = {"prepared": True}
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry, prepare=WithReferences())
    # Audit events and lock files are not data checks.
    reads = {name: count for name, count in downloads.items()
             if ".jsonl" not in name and not name.endswith(".lock")}
    extra = {name: count - 1 - uploads[name] for name, count in reads.items()
             if count > 1 + uploads[name]}
    # The checks around the hook and around Spark's reads of files the run does not own:
    # the source and audit captures and the staging folders. The data contract may be
    # rewritten during admission, so it is never reused and is read by each check.
    def bracketed_name(name):
        parts = pathlib.PurePosixPath(name).parts
        capture = next((parts[i + 1] for i, part in enumerate(parts[:-1])
                        if part.startswith("af-register-")), None)
        return ("admission-staging" in parts or capture in {"source", "dataset.yaml"}
                or (capture or "").startswith("audit-"))

    bracketed = {name: count for name, count in extra.items() if bracketed_name(name)}
    assert set(extra) == set(bracketed), sorted(
        (name.split("/registry/")[-1], extra[name]) for name in set(extra) - set(bracketed))
    # The published study and its admission record: one check each.
    published = [str(registry.root / "benchmarks" / study_id),
                 str(registry.root / "admissions" / study_id)]
    once = {name: count for name, count in reads.items()
            if name.startswith(tuple(published)) and count > 1 + uploads[name]}
    assert once == {}, once


def test_registration_rechecks_what_its_hook_could_change(spark, tmp_path, monkeypatch):  # noqa: F811
    """On a volume, nothing read before or during the prepare hook is reused after it."""
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    root = tmp_path.resolve()
    _volume(root, monkeypatch)
    path = configuration(root / "config")
    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry, prepare=SensorPreparation())

    class InspectAndChange(SensorPreparation):
        def prepare(self, context):
            # Read the existing study inside registration, then change one of its files.
            with load_study(study_id, registry_root=registry.root / "benchmarks", spark=spark):
                pass
            registry.definition(study_id)
            part = next((registry.root / "benchmarks" / study_id / "selection").glob("*.parquet"))
            pq.write_table(pq.read_table(part), part, compression="gzip")
            return super().prepare(context)

    with pytest.raises(ValueError, match="changed"):
        register_study(path, registry=registry, prepare=InspectAndChange())
