"""Saved workflow results, integrity checks, publication and recovery."""

import copy
import datetime as dt
import decimal
import hashlib
import json
import pathlib
import shutil
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark
from autoforecast import test_study_workflow as _workflow
from autoforecast.hpo import HPOError
from autoforecast.panel import Forecast, Panel
from autoforecast.study_preflight import StudyPreflightError
from autoforecast.study_tuning import (
    FitPredictResult,
    load_selected_configuration,
    load_tuning_result,
)
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_tuning import (
    config,
    implementation,
    open_study,
    predict,
)
from autoforecast.test_study_workflow import profile, register
from provenance.lineage import directory_digest

# Reuse the fixtures; each test module gets its own Spark session in the runner.
spark = _workflow.spark
complete = _workflow.complete
incomplete = _workflow.incomplete


def _publish_changed_record(copied: pathlib.Path, record: dict) -> pathlib.Path:
    identity = benchmark.digest(benchmark._identity(record))
    record["study_id"] = record["benchmark_id"] = identity
    (copied / "benchmark.json").write_text(json.dumps(record))
    final = copied.parent / identity
    copied.rename(final)
    return final / "benchmark.json"


def _copy_and_corrupt_coverage(
    manifest: pathlib.Path, root: pathlib.Path, stage: str, defect: str
) -> pathlib.Path:
    copied = root / "registry/original"
    shutil.copytree(manifest.parent, copied)
    record = json.loads((copied / "benchmark.json").read_text())
    key = next(key for key in record["coverage"] if key.startswith(stage + "/"))
    path = copied / record["coverage"][key]["path"]
    table = pq.read_table(path)
    rows = table.to_pylist()
    if defect == "duplicate_id":
        rows[0]["series_id"] = rows[1]["series_id"]
    elif defect == "wrong_type":
        table = table.set_column(
            table.schema.get_field_index("scored"),
            "scored",
            pa.array(table["scored"].to_pylist(), type=pa.float64()),
        ).replace_schema_metadata(None)
        rows = table.to_pylist()
    else:
        rows[0]["scored"] += 1
        rows[1]["scored"] -= 1
    shutil.rmtree(path)
    path.mkdir(parents=True)
    pq.write_table(type(table).from_pylist(rows, schema=table.schema), path / "part.parquet")
    record["coverage"][key]["sha256"] = directory_digest(path, suffixes=(".parquet",))
    record["logical_hashes"]["coverage/" + key] = benchmark._logical_hash.fingerprint(path)
    record["coverage_identities"][key]["content_sha256"] = (
        record["logical_hashes"]["coverage/" + key]["sha256"]
    )
    return _publish_changed_record(copied, record)


@pytest.mark.parametrize("stage", ["tuning", "validation", "test"])
@pytest.mark.parametrize("defect", ["duplicate_id", "wrong_type", "redistribution"])
def test_preflight_rejects_coverage_mask_mismatch_before_callback(
    complete, spark, tmp_path, stage, defect
):
    manifest = _copy_and_corrupt_coverage(complete, tmp_path / "coverage-mask", stage, defect)
    cfg_root = tmp_path / "coverage-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"origins": ["2024-01-03"]})
    calls = []

    def fit(request):
        calls.append(request.origin)
        return predict(request)

    with open_study(manifest, spark) as study:
        with pytest.raises(StudyPreflightError, match="coverage|mask"):
            study.tune(
                fit_predict=fit,
                config=cfg,
                implementation=implementation(),
            )
    assert calls == []


def test_saved_configuration_keeps_saved_seed_with_current_policy_config(complete, spark, tmp_path):
    source_root = tmp_path / "source-config"
    source_root.mkdir()
    source = profile(source_root, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        selected = study.tune(
            fit_predict=predict,
            config=source,
            implementation=implementation(),
            output_dir=tmp_path / "source-search",
        )
    saved = load_selected_configuration(selected.winner_config)

    current_root = tmp_path / "current-config"
    current_root.mkdir()
    current = profile(current_root, selection={"stage": "validation"})
    value = yaml.safe_load(current.read_text())
    value["hpo"]["seed"] = 123
    value["hyperparameters"]["model.level"]["search"] = {
        "kind": "categorical",
        "values": [0.0],
    }
    current.write_text(yaml.safe_dump(value))
    seeds = []

    def fit(request):
        seeds.append(request.seed)
        return predict(request)

    with open_study(complete, spark) as study:
        study.run_forecast_workflow(
            fit_predict=fit,
            config=current,
            selected_config=selected.winner_config,
            implementation=implementation(),
            output_dir=tmp_path / "future-run",
        )
    assert seeds == [saved["seed"], saved["seed"]]


def test_winner_retains_unique_source_and_parameter_digest(complete, spark, tmp_path):
    cfg_root = tmp_path / "winner-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        selected = study.tune(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "winner-search",
        )
    saved = load_selected_configuration(selected.winner_config)
    chosen = json.loads((selected.path / "selection.json").read_text())["selected"][0]
    assert saved["parameter_sha256"] == benchmark.digest(saved["parameters"])
    assert saved["source"]["search_id"] == selected.path.name
    assert saved["source"]["attempt_id"] == chosen["attempt_id"]
    assert saved["source"]["study"] == selected.provenance["study"]
    saved["source"]["study"]["sha256"] = "0" * 64
    selected.winner_config.write_text(json.dumps(saved))
    attempt = selected.winner_config.parent.parent
    manifest_sha = _reseal_attempt(attempt)
    path = selected.path / "selection.json"
    selection = json.loads(path.read_text())
    for group in (selection["candidates"], selection["selected"]):
        for item in group:
            if item["attempt_path"] == chosen["attempt_path"]:
                item["artifact_manifest_sha256"] = manifest_sha
    path.write_text(json.dumps(selection))
    with pytest.raises(HPOError, match="source"):
        load_selected_configuration(selected.winner_config)


def _copy_manifest(manifest: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    target = root / manifest.parent.name
    shutil.copytree(manifest.parent, target)
    return target / "benchmark.json"


def test_custom_output_cannot_overlap_frozen_study_artifacts(complete, spark, tmp_path):
    manifest = _copy_manifest(complete, tmp_path / "copied-registry")
    record = benchmark.read_benchmark_manifest(manifest)[1]
    output = benchmark.artifact_path(manifest, record, "selection")
    cfg_root = tmp_path / "overlap-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"origins": ["2024-01-03"]})
    value = yaml.safe_load(cfg.read_text())
    value["hpo"]["max_trials"] = 1
    cfg.write_text(yaml.safe_dump(value))
    calls = []

    def fit(request):
        calls.append(request.origin)
        return predict(request)

    try:
        with open_study(manifest, spark) as study:
            study.tune(
                fit_predict=fit,
                config=cfg,
                implementation=implementation(),
                output_dir=output,
            )
    except Exception:
        pass
    assert calls == []
    benchmark.verify_benchmark(manifest)


def test_workflow_output_cannot_invalidate_selected_configuration_source(complete, spark, tmp_path):
    cfg_root = tmp_path / "overlap-source-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        selected = study.tune(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "overlap-source-search",
        )
    source_attempt = selected.winner_config.parent.parent
    calls = []

    def fit(request):
        calls.append(request.origin)
        return predict(request)

    try:
        with open_study(complete, spark) as study:
            study.run_forecast_workflow(
                fit_predict=fit,
                selected_config=selected.winner_config,
                implementation=implementation(),
                output_dir=source_attempt,
            )
    except Exception:
        pass
    assert calls == []
    assert load_selected_configuration(selected.winner_config)["parameters"] == selected.parameters


def test_preflight_report_names_every_integrity_failure(complete, spark, tmp_path):
    copied = tmp_path / "report-registry/original"
    shutil.copytree(complete.parent, copied)
    record = json.loads((copied / "benchmark.json").read_text())
    defects = []
    for role, defect in (("final_training", "unexpected"), ("final_score", "duplicate")):
        relative = record["role_paths"][role]
        path = copied / relative
        if relative in defects:
            continue
        table = pq.read_table(path)
        rows = table.to_pylist()
        if defect == "unexpected":
            old = rows[0]["__af_series_id"]
            for row in rows:
                if row["__af_series_id"] == old:
                    row["__af_series_id"] = '{"sku":"unexpected"}'
        else:
            rows.append(rows[0])
        shutil.rmtree(path)
        path.mkdir()
        pq.write_table(type(table).from_pylist(rows, schema=table.schema), path / "part.parquet")
        for name, target_path in record["role_paths"].items():
            if target_path == relative:
                record["artifacts"][name] = directory_digest(path, suffixes=(".parquet",))
                record["logical_hashes"][name] = benchmark._logical_hash.fingerprint(path)
                record["logical_snapshots"][name] = record["logical_hashes"][name]["sha256"]
        defects.append(relative)
    manifest = _publish_changed_record(copied, record)
    cfg_root = tmp_path / "report-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"origins": "all"}, series_policy="use_available")
    with open_study(manifest, spark) as study, pytest.raises(StudyPreflightError) as caught:
        study.preflight(config=cfg)
    report = json.loads(caught.value.report_path.read_text())
    observed = {(issue["code"], issue["stage"]) for issue in report["issues"]}
    assert ("unexpected_id", "final_training") in observed
    assert ("duplicate_series_date", "final_score") in observed
    assert report["roles"]["final_training"]["missing_ids"]
    assert report["roles"]["final_score"]["actual_count"] == report["original_count"]


def test_validation_selection_succeeds_without_inner_windows(spark, tmp_path):
    from autoforecast.test_study_tuning import freeze

    manifest = freeze(tmp_path / "no-inner-study", spark, windows=False)
    cfg = config(tmp_path, budget=1)
    value = yaml.safe_load(cfg.read_text())
    value["selection"] = {"stage": "validation"}
    cfg.write_text(yaml.safe_dump(value))
    calls = []

    def fit(request):
        calls.append(request.origin)
        return predict(request)

    with open_study(manifest, spark) as study:
        result = study.tune(
            fit_predict=fit,
            config=cfg,
            implementation=implementation(),
        )
    assert calls == ["2024-01-05"]
    assert result.windows[0]["boundary"].startswith("validation/")


def test_exact_reuse_rejects_changed_config_and_study(complete, spark, tmp_path):
    cfg_root = tmp_path / "exact-reuse-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        selected = study.tune(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "exact-reuse-source",
        )
    changed_root = tmp_path / "changed-config"
    changed_root.mkdir()
    changed = profile(changed_root, selection={"stage": "validation"})
    value = yaml.safe_load(changed.read_text())
    value["hpo"]["seed"] = 10
    changed.write_text(yaml.safe_dump(value))
    with open_study(complete, spark) as study, pytest.raises(Exception, match="differs"):
        study.tune(
            fit_predict=lambda _: pytest.fail("changed config must not fit"),
            config=changed,
            implementation=implementation(),
            reuse_tuning=selected.path,
        )
    # Keep dates/options identical: only the frozen reference data identity changes.
    other = register(tmp_path / "exact-reuse-other", spark, count=3, scale_b=2.0)
    with open_study(other, spark) as study, pytest.raises(Exception, match="differs"):
        study.tune(
            fit_predict=lambda _: pytest.fail("changed study must not fit"),
            config=cfg,
            implementation=implementation(),
            reuse_tuning=selected.path,
        )


def _reseal_attempt(attempt: pathlib.Path) -> str:
    from autoforecast.experiments import _file_identity, _json_bytes

    manifest_path = attempt / "artifact_manifest.json"
    completion_path = attempt / "completion.json"
    completion = json.loads(completion_path.read_text())
    terminal = {
        key: value for key, value in completion.items() if key != "artifact_manifest_sha256"
    }
    artifacts = []
    for path in sorted(
        path
        for path in attempt.rglob("*")
        if path.is_file() and path not in {manifest_path, completion_path}
    ):
        sha, size = _file_identity(path)
        artifacts.append(
            {
                "path": path.relative_to(attempt).as_posix(),
                "sha256": sha,
                "bytes": size,
            }
        )
    manifest = {
        "schema": json.loads(manifest_path.read_text())["schema"],
        "artifacts": artifacts,
        "terminal_payload_sha256": hashlib.sha256(_json_bytes(terminal)).hexdigest(),
    }
    manifest_path.write_bytes(_json_bytes(manifest))
    completion["artifact_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completion_path.write_bytes(_json_bytes(completion))
    return completion["artifact_manifest_sha256"]


@pytest.mark.parametrize("missing", ["windows", "winner-config.json", "preflight.json"])
def test_tuning_reload_requires_current_evidence(complete, spark, tmp_path, missing):
    cfg_root = tmp_path / "obsolete-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        selected = study.tune(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "obsolete-search",
        )
    selection_path = selected.path / "selection.json"
    selection = json.loads(selection_path.read_text())
    chosen = selection["selected"][0]
    attempt = selected.path / chosen["attempt_path"]
    evidence_path = attempt / "evaluation/external-run.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["schema"] == "autoforecast.external-trial.v1"
    if missing == "windows":
        evidence.pop("windows")
        evidence.pop("aggregation")
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    else:
        (attempt / "metadata" / missing).unlink()
    manifest_sha = _reseal_attempt(attempt)
    for group in (selection["candidates"], selection["selected"]):
        for item in group:
            if item["attempt_path"] == chosen["attempt_path"]:
                item["artifact_manifest_sha256"] = manifest_sha
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    with pytest.raises(HPOError, match="current.*evidence"):
        load_tuning_result(selected.path)


def test_workflow_reload_binds_selected_settings_to_both_fits(complete, spark, tmp_path):
    cfg = profile(tmp_path, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        result = study.run_forecast_workflow(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "original",
        )
    assert load_forecast_workflow(result.path).parameters == result.parameters
    for field in (
        "parameters",
        "validation",
        "testing",
        "seed",
        "implementation",
        "source",
        "options",
        "tuning_path",
        "status",
        "swapped_stages",
        "stage_result",
        "search_profile",
        "winner",
        "task",
    ):
        root = tmp_path / field
        shutil.copytree(result.path, root)
        path = root / "result.json"
        record = json.loads(path.read_text())
        if field == "task":
            for relative in ("completion.json", "metadata/run.json"):
                task_path = root / relative
                task_record = json.loads(task_path.read_text())
                task_record["task"] = "forecast-validation"
                task_path.write_text(json.dumps(task_record))
        elif field == "options":
            record["provenance"]["options"]["minimum_history"] += 1
        elif field == "tuning_path":
            record[field] = "/wrong/search"
        elif field == "status":
            record[field] = "pending_actuals"
        elif field == "swapped_stages":
            record["validation"], record["testing"] = record["testing"], record["validation"]
            (root / "validation/result.json").write_text(json.dumps(record["validation"]))
            (root / "test/result.json").write_text(json.dumps(record["testing"]))
        elif field == "stage_result":
            duplicate = json.loads((root / "validation/result.json").read_text())
            duplicate["parameter_sha256"] = "0" * 64
            (root / "validation/result.json").write_text(json.dumps(duplicate))
        elif field == "search_profile":
            search = root / "metadata/search-config.yaml"
            search.write_bytes(search.read_bytes() + b"\n")
        elif field == "winner":
            winner = root / "metadata/winner-config.json"
            saved = json.loads(winner.read_text())
            saved["parameters"]["model.level"] = 999.0
            winner.write_text(json.dumps(saved))
        elif field == "parameters":
            record[field]["model.level"] = 999.0
        elif field in ("validation", "testing"):
            record[field]["parameter_sha256"] = "0" * 64
            folder = "test" if field == "testing" else "validation"
            (root / folder / "result.json").write_text(json.dumps(record[field]))
        elif field == "source":
            record["provenance"]["configuration_source"]["trial_id"] = "wrong-trial"
        elif field == "implementation":
            record["provenance"][field]["version"] = "changed"
        else:
            record["provenance"][field] += 1
        path.write_text(json.dumps(record))
        _reseal_attempt(root)
        with pytest.raises(HPOError, match="configuration|schema|unsupported forecast workflow result"):
            load_forecast_workflow(root)
        with pytest.raises(HPOError, match="configuration|schema"):
            load_selected_configuration(root / "metadata/winner-config.json")


@pytest.mark.parametrize(
    "role,field",
    [
        ("source-keys", "sku"),
        ("final_training", "sku"),
        ("source-keys", benchmark.START_COLUMN),
    ],
)
def test_preflight_checks_source_key_and_start_mapping(complete, spark, tmp_path, role, field):
    copied = tmp_path / "registry/original"
    shutil.copytree(complete.parent, copied)
    record = json.loads((copied / "benchmark.json").read_text())
    relative = record["role_paths"][role]
    path = copied / relative
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0][field] = "wrong-key" if field == "sku" else dt.date(2024, 1, 2)
    shutil.rmtree(path)
    path.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path / "part.parquet")
    for name, target in record["role_paths"].items():
        if target == relative:
            record["artifacts"][name] = directory_digest(path, suffixes=(".parquet",))
            record["logical_hashes"][name] = benchmark._logical_hash.fingerprint(path)
            record["logical_snapshots"][name] = record["logical_hashes"][name]["sha256"]
    manifest = _publish_changed_record(copied, record)
    cfg = profile(tmp_path, selection={"stage": "validation"})
    with open_study(manifest, spark) as study:
        with pytest.raises(StudyPreflightError, match="mapping|source.key|canonical"):
            study.tune(
                fit_predict=lambda _: pytest.fail("invalid mapping must not reach a callback"),
                config=cfg,
                implementation=implementation(),
            )


def test_reporting_and_model_identity_keep_available_masks_and_selection_fixed(
    incomplete, spark, tmp_path
):
    results, calls = [], []

    def fit(request):
        ids = sorted(row[request.panel.series_col] for row in request.series_mapping.collect())
        calls.append((request.origin, ids, dict(request.model_parameters), request.seed))
        return predict(request)

    for name, metrics in (("first", ["mae"]), ("second", ["mae", "rmse", "wrmsse", "bias"])):
        folder = tmp_path / name
        folder.mkdir()
        cfg = profile(
            folder,
            selection={"origins": "all"},
            reporting={"metrics": metrics},
            series_policy="use_available",
        )
        with open_study(incomplete, spark) as study:
            results.append(
                study.tune(
                    fit_predict=fit,
                    config=cfg,
                    implementation={**implementation(), "name": name},
                )
            )
    first, second = results
    assert len(calls) == 8 and calls[:4] == calls[4:]
    assert first.parameters == second.parameters and first.tuning_score == second.tuning_score
    assert first.provenance["study_id"] == second.provenance["study_id"]
    for left, right in zip(first.windows, second.windows, strict=True):
        assert left["evaluation_contract"] == right["evaluation_contract"]
        availability = left["evaluation_contract"]["availability"]
        assert availability["mask_sha256"]
        assert availability["eligible_ids"] == calls[0][1]


def _one_trial(path: pathlib.Path) -> pathlib.Path:
    value = yaml.safe_load(path.read_text())
    value["hpo"]["max_trials"] = 1
    path.write_text(yaml.safe_dump(value))
    return path


@pytest.mark.parametrize("metric", ["mae", "wape"])
@pytest.mark.parametrize("zero", [False, True])
def test_decimal_targets_supported_by_preflight(spark, tmp_path, metric, zero):
    original = register(tmp_path / "original", spark, count=1)
    record = benchmark.read_benchmark_manifest(original)[1]
    source = pathlib.Path(record["sources"]["selection"]["path"])
    rows = pq.read_table(source).to_pylist()
    target = pa.array(
        [decimal.Decimal("0" if zero else str(row["units"])).quantize(decimal.Decimal("0.01"))
         for row in rows],
        type=pa.decimal128(12, 2),
    )
    table = pq.read_table(source)
    table = table.set_column(table.schema.get_field_index("units"), "units", target)
    decimal_source = tmp_path / "decimal-source.parquet"
    pq.write_table(table, decimal_source)
    decimal_study = benchmark.freeze_benchmark(
        contract_path=source.parent / "dataset.yaml",
        sources=dict.fromkeys(
            ("selection", "final_training", "validation_score", "final_score"),
            decimal_source,
        ),
        recipe=record["recipe"],
        scoring_references=pathlib.Path(record["sources"]["scoring_references"]["path"]),
        output_root=tmp_path / "decimal-registry",
        spark=spark,
    )
    cfg = profile(tmp_path, selection={"stage": "validation", "metric": metric},
                  reporting={"metrics": [metric]})
    with open_study(decimal_study, spark) as study:
        if zero and metric == "wape":
            with pytest.raises(StudyPreflightError, match="zero_actual_volume"):
                study.preflight(config=cfg)
        else:
            assert study.preflight(config=cfg).report["status"] == "passed"


def test_available_population_separates_observed_from_scored_cells(spark, tmp_path):
    manifest = register(tmp_path / "late-study", spark, start_b=4)
    cfg = profile(
        tmp_path,
        selection={"origins": ["2024-01-03"]},
        reporting={"metrics": ["mae", "wrmsse"]},
        series_policy="use_available",
    )
    with open_study(manifest, spark) as study:
        preflight = study.preflight(config=cfg)
        key = next(key for key in preflight.report["windows"] if key.startswith("tuning/2024-01-03"))
        window = preflight.report["windows"][key]
        # B has both target observations but cannot yet be fitted at the origin.
        assert window["expected_cells"] == window["observed_cells"] == 4
        assert window["forecast_cells"] == window["scored_cells"] == 2
        record = benchmark.read_benchmark_manifest(preflight.study_path)[1]
        for invalid in (-1, True, 5):
            changed = copy.deepcopy(record)
            changed["execution_policy"]["windows"][key]["observed_cells"] = invalid
            with pytest.raises(benchmark.BenchmarkError, match="observed"):
                benchmark._validate_manifest(changed)


def test_available_score_separates_observed_from_scored_cells(spark, tmp_path):
    manifest = register(tmp_path / "late-score-study", spark, start_b=4)
    cfg = _one_trial(profile(
        tmp_path,
        selection={"origins": ["2024-01-03"]},
        reporting={"metrics": ["mae"]},
        series_policy="use_available",
    ))
    with open_study(manifest, spark) as study:
        result = study.tune(fit_predict=predict, config=cfg, implementation=implementation())
    population = result.windows[0]["score"]["population"]
    assert population["expected_cells"] == population["observed_cells"] == 4
    assert population["forecasted_series"] == 1 and population["scored_cells"] == 2
    report = json.loads(result.validation_report.read_text())
    window = next(value for key, value in report["windows"].items()
                  if key.startswith("tuning/2024-01-03"))
    availability = result.windows[0]["evaluation_contract"]["availability"]
    assert availability["mask_sha256"] == window["mask_sha256"]
    assert availability["observed_cells"] == window["observed_cells"]


def test_available_horizon_status_does_not_attach_wrmsse_reason_to_scored_mae(spark, tmp_path):
    manifest = register(tmp_path / "horizon-study", spark, start_b=4)
    cfg = _one_trial(profile(
        tmp_path,
        selection={"origins": ["2024-01-03"]},
        reporting={"metrics": ["mae", "wrmsse"], "forecast_accuracy_breakdown": "per_step"},
        series_policy="use_available",
    ))
    with open_study(manifest, spark) as study:
        result = study.tune(fit_predict=predict, config=cfg, implementation=implementation())
    rows = result.windows[0]["score"]["forecast_accuracy_breakdown"]["per_step"]
    assert all(row["status"] == "scored" for row in rows)
    assert all(row["unavailable_reason"] is None for row in rows)
    assert all(row["metric_status"]["wrmsse"] == {
        "status": "unavailable", "reason": "unfittable_weighted_series",
    } for row in rows)
    assert result.windows[0]["score"]["metric_status"]["wrmsse"]["reason"] == (
        "unfittable_weighted_series"
    )


def test_output_root_rejects_completed_search_namespace(spark, tmp_path):
    manifest = register(tmp_path / "output-study", spark, count=1)
    cfg = _one_trial(profile(tmp_path, selection={"stage": "validation"}))
    with open_study(manifest, spark) as study:
        selected = study.tune(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "searches",
        )
        assert (selected.path / "selection.json").is_file()
        for relative in (".", "child", "attempts", "attempts/new/nested"):
            for run in (study.tune, study.run_forecast_workflow):
                with pytest.raises(HPOError, match="overlaps"):
                    run(fit_predict=predict, config=cfg, implementation=implementation(),
                        output_dir=selected.path / relative)


# Every admission name the registry reader looks up (study_registration._record_path).
@pytest.mark.parametrize("marker", ["admission.json", "admission.yaml", "admission.yml",
                                    "capture.json"])
def test_output_root_rejects_sealed_admission_and_capture_evidence(spark, tmp_path, marker):
    manifest = register(tmp_path / "output-study", spark, count=1)
    cfg = _one_trial(profile(tmp_path, selection={"stage": "validation"}))
    sealed = tmp_path / "registry-copy" / "sealed-evidence"
    sealed.mkdir(parents=True)
    (sealed / marker).write_text("{}")
    with open_study(manifest, spark) as study:
        for relative in (".", "prepared", "runs/new"):
            with pytest.raises(HPOError, match="overlaps"):
                study.run_forecast_workflow(fit_predict=predict, config=cfg,
                                            implementation=implementation(),
                                            output_dir=sealed / relative)


def test_schedule_rejects_impossible_count_before_enumerating(monkeypatch):
    calls = 0
    real_advance = benchmark.advance

    def bounded_advance(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 4:
            raise AssertionError("resolver enumerated an already-impossible schedule")
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(benchmark, "advance", bounded_advance)
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.resolve_tuning_schedule(
            {
                "frequency": "D",
                "training_start": "9998-01-01",
                "validation_origins": ["9999-01-01"],
            },
            {"horizon": 1, "stride": 1, "count": 1_000_000},
        )
    assert calls <= 4


def test_freeze_rejects_conflicting_reference_metric_declaration(spark, tmp_path):
    original = register(tmp_path / "reference-original", spark, count=1)
    record = benchmark.read_benchmark_manifest(original)[1]
    recipe = copy.deepcopy(record["recipe"])
    recipe["reference_metrics"] = ["mae"]
    source = pathlib.Path(record["sources"]["selection"]["path"])
    sources = {
        name: pathlib.Path(record["sources"][name]["path"])
        for name in ("selection", "final_training", "validation_score", "final_score")
    }
    with pytest.raises(benchmark.BenchmarkError, match="reference_metrics"):
        benchmark.freeze_benchmark(
            contract_path=source.parent / "dataset.yaml",
            sources=sources,
            recipe=recipe,
            scoring_references=pathlib.Path(
                record["sources"]["scoring_references"]["path"]
            ),
            output_root=tmp_path / "reference-registry",
            spark=spark,
        )


def test_collected_integrity_issues_mark_affected_role_blocked(spark, tmp_path):
    manifest = register(tmp_path / "report-study", spark, count=1)
    copied = tmp_path / "report-registry/original"
    shutil.copytree(manifest.parent, copied)
    record = json.loads((copied / "benchmark.json").read_text())
    role = "final_score"
    relative = record["role_paths"][role]
    path = copied / relative
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.append(rows[0])
    shutil.rmtree(path)
    path.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path / "part.parquet")
    for name, target in record["role_paths"].items():
        if target == relative:
            record["artifacts"][name] = directory_digest(path, suffixes=(".parquet",))
            record["logical_hashes"][name] = benchmark._logical_hash.fingerprint(path)
            record["logical_snapshots"][name] = record["logical_hashes"][name]["sha256"]
    changed = _publish_changed_record(copied, record)
    cfg = profile(tmp_path, selection={"stage": "validation"})
    with open_study(changed, spark) as study, pytest.raises(StudyPreflightError) as caught:
        study.preflight(config=cfg)
    report = json.loads(caught.value.report_path.read_text())
    assert any(issue["stage"] == role for issue in report["issues"])
    assert report["roles"][role]["status"] == "blocked"
    assert report["roles"][report["current_check"]["stage"]]["status"] == "blocked"


def test_fixed_workflow_has_no_optimizer_and_reuses_saved_settings(complete, spark, tmp_path, monkeypatch):
    import autoforecast.study_workflow as workflow

    cfg = profile(tmp_path, selection={"metric": "rmse"}, reporting={"metrics": ["mae"]})
    value = yaml.safe_load(cfg.read_text())
    value.pop("hpo")
    cfg.write_text(yaml.safe_dump(value))
    calls = []

    def fit(request):
        calls.append((request.purpose, request.origin, request.seed, request.model_parameters))
        return predict(request)

    def unexpected_search(*args, **kwargs):
        pytest.fail("fixed configuration must not invoke an optimizer")

    monkeypatch.setattr(workflow, "tune_study", unexpected_search)
    with open_study(complete, spark) as study:
        result = study.run_forecast_workflow(
            fit_predict=fit, config=cfg, implementation=implementation(),
            fixed_parameters={"model.level": 4.0}, output_dir=tmp_path / "fixed",
        )
        assert calls == [
            ("validation", "2024-01-07", 42, {"level": 4.0}),
            ("refit", "2024-01-09", 42, {"level": 4.0}),
        ]
        assert result.tuning_path is None
        assert result.provenance["mode"] == "fixed_configuration"
        assert result.provenance["work_estimate"]["trial_callback_fits"] == 0
        assert result.provenance["options"]["metric"] == "rmse"
        assert result.provenance["options"]["metrics"] == ["mae", "rmse"]
        assert load_selected_configuration(result.selected_config)["source"]["kind"] == "fixed"
        reused = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), selected_config=result.selected_config,
            config=cfg,
            output_dir=tmp_path / "reused-fixed",
        )
        assert reused.parameters == result.parameters
        assert len(calls) == 4

        assert reused.provenance["mode"] == "saved_configuration"
        assert reused.provenance["options"]["metric"] == "rmse"
        assert reused.provenance["options"]["metrics"] == ["mae", "rmse"]
        with pytest.raises(ValueError, match="fixed_parameters requires"):
            study.run_forecast_workflow(
                fit_predict=fit, implementation=implementation(), config=cfg,
                fixed_parameters={}, selected_config=result.selected_config,
            )
        with pytest.raises(ValueError, match="undeclared"):
            study.run_forecast_workflow(
                fit_predict=fit, implementation=implementation(), config=cfg,
                fixed_parameters={"model.nonexistent": 1},
            )
        assert len(calls) == 4

    # Keep all duplicate metadata consistent so only the fixed-source binding can reject this.
    root = tmp_path / "tampered" / result.path.name
    shutil.copytree(result.path, root)
    winner = json.loads((root / "metadata/winner-config.json").read_text())
    winner["source"]["workflow_id"] = "different-workflow"
    record = json.loads((root / "result.json").read_text())
    record["provenance"]["configuration_source"] = winner["source"]
    (root / "result.json").write_text(json.dumps(record))
    (root / "metadata/winner-config.json").write_text(json.dumps(winner))
    (root / "metadata/provenance.json").write_text(json.dumps(record["provenance"]))
    _reseal_attempt(root)
    with pytest.raises(HPOError, match="fixed configuration"):
        load_forecast_workflow(root)


@pytest.mark.parametrize("policy", ["use_available", "exclude_unavailable"])
@pytest.mark.parametrize("fixed", [False, True])
def test_checkpoint_resume_rechecks_same_effective_study(spark, tmp_path, policy, fixed):
    # Exactly A/B; B begins on day 7 and lacks three days at validation.
    manifest = register(tmp_path / "study", spark, start_b=7, count=1)
    cfg = _one_trial(profile(
        tmp_path, series_policy=policy, minimum_history=3,
        selection={"origins": "all", "metric": "rmse"},
    ))
    calls = []

    def fit(request):
        calls.append(request.purpose)
        return predict(request)

    with open_study(manifest, spark) as study:
        checkpoint = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), config=cfg,
            fixed_parameters={} if fixed else None, validation_only=True,
            output_dir=tmp_path / "validation",
        )
        expected = ["validation"] if fixed else ["selection", "validation"]
        assert calls == expected and checkpoint.testing is None
        assert checkpoint.provenance["study"]["sha256"] != study._sha256
        completed = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), validation_run=checkpoint.path,
            output_dir=tmp_path / "completed",
        )
        assert calls == [*expected, "refit"]
        assert completed.testing is not None
        assert completed.provenance["study"] == checkpoint.provenance["study"]
        assert completed.provenance["options"] == checkpoint.provenance["options"]
        assert completed.validation["score"]["rmse"] == checkpoint.validation["score"]["rmse"]
    other = register(tmp_path / "other-study", spark, start_b=7, count=1, scale_b=2.0)
    with open_study(other, spark) as study, pytest.raises(HPOError, match="exact study"):
        study.run_forecast_workflow(
            fit_predict=lambda _: pytest.fail("different study must not fit"),
            implementation=implementation(), validation_run=checkpoint.path,
            output_dir=tmp_path / "rejected",
        )


def test_validation_checkpoint_refits_only_after_selection(complete, spark, tmp_path, monkeypatch):
    from autoforecast import study_tuning, study_workflow

    cfg = profile(tmp_path)
    calls = []

    def fit(request):
        calls.append(request.purpose)
        return predict(request)

    with open_study(complete, spark) as study:
        validation = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), config=cfg,
            fixed_parameters={}, validation_only=True, output_dir=tmp_path / "validation",
        )
        assert calls == ["validation"]
        assert validation.testing is None
        original_score = study_workflow.score_study

        def only_score_test(*args, **kwargs):
            assert kwargs["contract"]["boundary"].startswith("test/"), "validation was rescored"
            return original_score(*args, **kwargs)

        monkeypatch.setattr(study_workflow, "score_study", only_score_test)
        monkeypatch.setattr(study_tuning, "score_study", only_score_test)
        completed = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), validation_run=validation.path,
            output_dir=tmp_path / "completed",
        )
        assert calls == ["validation", "refit"]
        assert completed.validation["score"]["mae"] == validation.validation["score"]["mae"]
        assert completed.validation["score"]["series_scores"]["sha256"] == validation.validation["score"]["series_scores"]["sha256"]
        retained_references = (
            completed.validation["score"]["series_scores"],
            completed.validation["score"]["forecast_accuracy_breakdown"]["series_sums"],
        )
        assert all(pathlib.Path(ref["path"]).is_relative_to(completed.path) for ref in retained_references)
        validation.path.rename(validation.path.with_name("removed-checkpoint"))
        assert load_forecast_workflow(completed.path).testing is not None
        for reference in retained_references:
            assert pathlib.Path(reference["path"]).is_dir()
            assert directory_digest(reference["path"], suffixes=(".parquet",)) == reference["sha256"]
        assert completed.provenance["work_estimate"]["total_callback_fits"] == 1
        assert completed.testing is not None

    import mlflow

    from autoforecast.run_mlflow import export_forecast_workflow
    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    experiment = client.create_experiment("checkpoint-export", artifact_location=str(tmp_path / "artifacts"))
    original_upload = mlflow.tracking.MlflowClient.log_artifacts
    fail_upload = True

    def interrupted_upload(self, run_id, local_dir, artifact_path=None):
        nonlocal fail_upload
        if fail_upload and self.get_run(run_id).data.tags["autoforecast.stage"] == "test":
            fail_upload = False
            raise RuntimeError("injected upload interruption")
        return original_upload(self, run_id, local_dir, artifact_path)

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_artifacts", interrupted_upload)
    with pytest.raises(RuntimeError, match="injected upload"):
        export_forecast_workflow(completed.path, tracking_uri=client.tracking_uri, experiment_id=experiment)
    original_run_ids = {remote.info.run_id for remote in client.search_runs([experiment])}
    assert len(original_run_ids) == 2
    links = export_forecast_workflow(completed.path, tracking_uri=client.tracking_uri, experiment_id=experiment)
    assert {link.run_id for link in links.values()} == original_run_ids
    replay = export_forecast_workflow(completed.path, tracking_uri=client.tracking_uri, experiment_id=experiment)
    assert replay == links
    assert len(client.search_runs([experiment])) == 2
    for stage, link in links.items():
        remote = client.get_run(link.run_id)
        assert remote.info.status == "FINISHED"
        evidence = completed.validation if stage == "validation" else completed.testing
        assert remote.data.metrics["mae"] == evidence["score"]["mae"]
    client.log_dict(
        links["test"].run_id, {**completed.testing["score"], "mae": 999.0}, "checked_score.json"
    )
    with pytest.raises(ValueError, match="remote scores differ"):
        export_forecast_workflow(completed.path, tracking_uri=client.tracking_uri, experiment_id=experiment)
    client.log_dict(links["test"].run_id, completed.testing["score"], "checked_score.json")
    client.log_text(links["test"].run_id, "changed remotely", "models/constant.json")
    with pytest.raises(ValueError, match="remote model artifacts differ"):
        export_forecast_workflow(completed.path, tracking_uri=client.tracking_uri, experiment_id=experiment)


@pytest.fixture(scope="module")
def workflow_evidence(complete, spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("workflow-export-evidence")
    cfg = profile(root, selection={"stage": "validation", "metric": "rmse"})
    value = yaml.safe_load(cfg.read_text())
    value["hpo"]["seed"] = 17
    value["hyperparameters"]["model.level"]["default"] = 4.0
    value["hyperparameters"]["model.level"]["search"]["values"].append(4.0)
    offset = value["hyperparameters"]["training.offset"]
    offset["default"] = 3
    offset["valid"]["high"] = 5
    offset["search"]["high"] = 5
    cfg.write_text(yaml.safe_dump(value))
    calls = []

    def fit(request):
        calls.append(
            (
                request.purpose,
                request.seed,
                dict(request.model_parameters),
                dict(request.training_parameters),
            )
        )
        return predict(request)

    with open_study(complete, spark) as study:
        checkpoint = study.run_forecast_workflow(
            fit_predict=fit,
            implementation=implementation(),
            config=cfg,
            fixed_parameters={},
            validation_only=True,
            output_dir=root / "validation",
        )
        completed = study.run_forecast_workflow(
            fit_predict=fit,
            implementation=implementation(),
            validation_run=checkpoint.path,
            output_dir=root / "completed",
        )
    return checkpoint, completed, calls


def test_empty_fixed_parameters_use_profile_defaults_and_seed(workflow_evidence):
    checkpoint, completed, calls = workflow_evidence
    assert calls == [
        ("validation", 17, {"level": 4.0}, {"offset": 3}),
        ("refit", 17, {"level": 4.0}, {"offset": 3}),
    ]
    assert (
        checkpoint.parameters == completed.parameters == {"model.level": 4.0, "training.offset": 3}
    )


@pytest.mark.parametrize("completed", [False, True])
def test_workflow_reload_rejects_changed_frozen_inputs(workflow_evidence, completed):
    result = workflow_evidence[int(completed)]
    source = result.provenance["study"]
    path, record = benchmark.read_benchmark_manifest(pathlib.Path(source["path"]), source["sha256"])
    folders = {benchmark.artifact_path(path, record, role) for role in record["artifacts"]}
    folders.update(path.parent / item["path"] for group in ("grids", "coverage")
                   for item in record[group].values())
    files = [next(folder.glob("*.parquet")) for folder in sorted(folders)]
    files.append(path.parent / "dataset.yaml")
    for part in files:
        original = part.read_bytes()
        try:
            part.write_bytes(original + b"changed")
            with pytest.raises(benchmark.BenchmarkError):
                load_forecast_workflow(result.path)
            part.unlink()
            with pytest.raises((benchmark.BenchmarkError, OSError)):
                load_forecast_workflow(result.path)
        finally:
            part.write_bytes(original)
    assert load_forecast_workflow(result.path).parameters == result.parameters


@pytest.mark.parametrize("evidence", ["series_scores", "series_sums"])
def test_checkpoint_resume_rejects_corruption_during_copy(
    workflow_evidence, complete, spark, tmp_path, monkeypatch, evidence
):
    from mlops import _safe_files

    checkpoint = workflow_evidence[0]
    score = checkpoint.validation["score"]
    reference = (
        score["series_scores"] if evidence == "series_scores"
        else score["forecast_accuracy_breakdown"]["series_sums"]
    )
    source = checkpoint.path / "validation"
    relative = pathlib.Path(reference["path"]).relative_to(source)
    original_copy = _safe_files.snapshot_regular_files
    corrupted = []

    def corrupt_copy(src, dest, paths):
        result = original_copy(src, dest, paths)
        if pathlib.Path(src) == source:
            copied_file = next((pathlib.Path(dest) / relative).rglob("*.parquet"))
            copied_file.write_bytes(copied_file.read_bytes() + b"corrupted-copy")
            corrupted.append(copied_file)
        return result

    monkeypatch.setattr(_safe_files, "snapshot_regular_files", corrupt_copy)
    with open_study(complete, spark) as study:
        with pytest.raises(HPOError, match="validation score evidence changed during retention"):
            study.run_forecast_workflow(
                fit_predict=lambda _: pytest.fail("corrupt checkpoint reached final fitting"),
                implementation=implementation(), validation_run=checkpoint.path,
                output_dir=tmp_path / "resumed",
            )
    assert len(corrupted) == 1
    assert load_forecast_workflow(checkpoint.path).testing is None


@pytest.fixture
def workflow_tracking(tmp_path):
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/tracking.db")
    experiment = client.create_experiment(
        "workflow-test", artifact_location=str(tmp_path / "artifacts")
    )
    return client, experiment


@pytest.mark.parametrize("defect", ["duplicate", "deleted", "identity"])
def test_workflow_export_rejects_conflicting_remote_state(
    workflow_evidence, workflow_tracking, defect
):
    from autoforecast.run_mlflow import RunMLflowError, export_forecast_workflow

    completed = workflow_evidence[1]
    client, experiment = workflow_tracking
    options = dict(tracking_uri=client.tracking_uri, experiment_id=experiment)
    links = export_forecast_workflow(completed.path, **options)
    run_id = links["validation"].run_id
    if defect == "duplicate":
        client.create_run(experiment, tags=client.get_run(run_id).data.tags)
    elif defect == "deleted":
        client.delete_run(run_id)
    else:
        client.set_tag(run_id, "autoforecast.study_sha256", "0" * 64)
    with pytest.raises(RunMLflowError):
        export_forecast_workflow(completed.path, **options)


def test_workflow_export_repairs_interrupted_termination(workflow_evidence, workflow_tracking):
    from autoforecast.run_mlflow import export_forecast_workflow

    completed = workflow_evidence[1]
    client, experiment = workflow_tracking
    options = dict(tracking_uri=client.tracking_uri, experiment_id=experiment)
    links = export_forecast_workflow(completed.path, **options)
    for link in links.values():
        client.set_terminated(link.run_id, status="FAILED")
    assert export_forecast_workflow(completed.path, **options) == links
    assert all(client.get_run(link.run_id).info.status == "FINISHED" for link in links.values())
    assert len(client.search_runs([experiment])) == 2


@pytest.mark.parametrize("checkpoint", [False, True])
def test_workflow_export_rejects_invalid_local_evidence_before_client(
    workflow_evidence, tmp_path, monkeypatch, checkpoint
):
    from autoforecast import run_mlflow

    if checkpoint:
        source = workflow_evidence[0].path
    else:
        source = tmp_path / workflow_evidence[1].path.name
        shutil.copytree(workflow_evidence[1].path, source)
        (source / "result.json").write_text("{}")
    monkeypatch.setattr(
        run_mlflow, "_mlflow", lambda: pytest.fail("remote client before local verification")
    )
    with pytest.raises(run_mlflow.RunMLflowError):
        run_mlflow.export_forecast_workflow(source, tracking_uri="unused", experiment_id="1")


@pytest.mark.parametrize("kind", ["missing", "deleted"])
def test_workflow_export_requires_existing_active_experiment(
    workflow_evidence, workflow_tracking, kind
):
    import mlflow

    from autoforecast.run_mlflow import RunMLflowError, export_forecast_workflow

    client, experiment = workflow_tracking
    if kind == "deleted":
        client.delete_experiment(experiment)
    else:
        experiment = "99999999"
    with pytest.raises((RunMLflowError, mlflow.exceptions.MlflowException)):
        export_forecast_workflow(
            workflow_evidence[1].path, tracking_uri=client.tracking_uri, experiment_id=experiment
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"validation_only": True},
        {"config": "new.yaml"},
        {"selected_config": "saved.json"},
        {"reuse_tuning": "search"},
        {"fixed_parameters": {}},
        {"tuning_origin": "2024-01-03"},
    ],
)
def test_resume_rejects_configuration_conflicts_before_loading(extra):
    from autoforecast.study_workflow import run_forecast_workflow

    with pytest.raises(ValueError, match="validation_run cannot combine"):
        run_forecast_workflow(
            object(),
            fit_predict=lambda _: pytest.fail("conflicting request must not fit"),
            implementation=implementation(),
            validation_run="absent",
            **extra,
        )


def test_resume_rejects_completed_workflow_and_changed_implementation(
    workflow_evidence, complete, spark
):
    checkpoint, completed, _ = workflow_evidence
    with open_study(complete, spark) as study:
        for path, identity, message in (
            (completed.path, implementation(), "validation checkpoint"),
            (checkpoint.path, {**implementation(), "version": "changed"}, "implementation"),
        ):
            with pytest.raises(HPOError, match=message):
                study.run_forecast_workflow(
                    fit_predict=lambda _: pytest.fail("invalid resume must not fit"),
                    implementation=identity,
                    validation_run=path,
                )


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("field", ["seed", "implementation", "configuration_source"])
def test_resealed_provenance_must_match_saved_settings(workflow_evidence, tmp_path, completed, field):
    source = workflow_evidence[int(completed)]
    root = tmp_path / source.path.name
    shutil.copytree(source.path, root)
    assert load_forecast_workflow(root).parameters == source.parameters
    path = root / "result.json"
    record = json.loads(path.read_text())
    if field == "seed":
        record["provenance"][field] += 1
    elif field == "implementation":
        record["provenance"][field]["version"] = "different-version"
    else:
        record["provenance"][field]["workflow_id"] = "different-origin"
    path.write_text(json.dumps(record))
    (root / "metadata/provenance.json").write_text(json.dumps(record["provenance"]))
    _reseal_attempt(root)
    for load, target in (
        (load_forecast_workflow, root),
        (load_selected_configuration, root / "metadata/winner-config.json"),
    ):
        with pytest.raises(HPOError, match="configuration differs"):
            load(target)


@pytest.mark.parametrize("field", ["kind", "workflow_id", "study", "tuning_path"])
def test_resealed_fixed_checkpoint_preserves_its_origin(workflow_evidence, tmp_path, field):
    checkpoint = workflow_evidence[0]
    root = tmp_path / checkpoint.path.name
    shutil.copytree(checkpoint.path, root)
    assert load_forecast_workflow(root).testing is None
    path, winner_path = root / "result.json", root / "metadata/winner-config.json"
    record, saved = json.loads(path.read_text()), json.loads(winner_path.read_text())
    if field == "tuning_path":
        record[field] = "/different/search"
    elif field == "study":
        saved["source"][field]["sha256"] = "0" * 64
    else:
        saved["source"][field] = "different-origin"
    record["provenance"]["configuration_source"] = saved["source"]
    path.write_text(json.dumps(record))
    winner_path.write_text(json.dumps(saved))
    (root / "metadata/provenance.json").write_text(json.dumps(record["provenance"]))
    _reseal_attempt(root)
    for load, target in (
        (load_forecast_workflow, root),
        (load_selected_configuration, winner_path),
    ):
        with pytest.raises(HPOError, match="fixed configuration|source"):
            load(target)


@pytest.mark.parametrize("payload,valid", [
    ('{"mae":1.0,"missing":null,"count":1}', True),
    ('{ "count": 1, "missing": null, "mae": 1.0 }', True),
    ('{"mae":true,"missing":null,"count":1}', False),
    ('{"mae":1.0,"missing":null,"count":true}', False),
    ('{"mae":1.0,"missing":0,"count":1}', False),
    ('{"mae":99,"mae":1.0,"missing":null,"count":1}', False),
])
def test_export_verification_preserves_json_types_and_rejects_duplicate_keys(tmp_path, payload, valid):
    from autoforecast.run_mlflow import RunMLflowError, _verify_workflow_export

    models = tmp_path / "models"
    models.mkdir()
    (models / "model.bin").write_bytes(b"synthetic model")
    score = tmp_path / "checked_score.json"
    score.write_text(payload)

    class Client:
        def download_artifacts(self, run_id, name, folder):
            return str(models if name == "models" else score)

    evidence = {
        "model_sha256": directory_digest(models),
        "score": {"mae": 1.0, "missing": None, "count": 1},
    }
    if valid:
        _verify_workflow_export(Client(), "synthetic", evidence)
    else:
        with pytest.raises(RunMLflowError, match="remote score"):
            _verify_workflow_export(Client(), "synthetic", evidence)


def test_workflow_owns_spark_visible_scratch_without_global_temp_state(complete, spark, tmp_path):
    import tempfile

    from autoforecast.study import load_study

    scratch = tmp_path / "shared-scratch"
    scratch.mkdir()
    global_before = tempfile.tempdir
    captured = []

    def callback(request):
        for path in (pathlib.Path(request.panel.path), request.output_dir):
            assert path.is_relative_to(scratch)
            captured.append(path)
        return predict(request)

    with load_study(complete.parent.name, registry_root=complete.parent.parent,
                    spark=spark, scratch_root=scratch) as study:
        result = study.run_forecast_workflow(
            fit_predict=callback, implementation=implementation(), config=config(tmp_path),
            fixed_parameters={}, output_dir=tmp_path / "runs",
        )
        assert all(pathlib.Path(evidence["model_artifacts"]).exists()
                   for evidence in (result.validation, result.testing))
    assert tempfile.tempdir == global_before
    assert not list(scratch.iterdir())
    assert all(not path.exists() for path in captured)
    assert load_forecast_workflow(result.path).testing["score"] == result.testing["score"]


def test_export_preserves_upload_error_when_failure_termination_also_fails(
    workflow_evidence, workflow_tracking, monkeypatch,
):
    import mlflow

    from autoforecast.run_mlflow import export_forecast_workflow

    client, experiment = workflow_tracking
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda **_: client)

    def fail_upload(*args, **kwargs):
        raise RuntimeError("primary upload failure")

    def fail_termination(*args, **kwargs):
        raise OSError("secondary termination failure")

    monkeypatch.setattr(client, "log_dict", fail_upload)
    monkeypatch.setattr(client, "set_terminated", fail_termination)
    with pytest.raises(RuntimeError, match="primary upload failure"):
        export_forecast_workflow(workflow_evidence[1].path,
                                 tracking_uri=client.tracking_uri, experiment_id=experiment)


def test_callback_artifact_copy_failure_has_retention_fail_event(tmp_path, monkeypatch):
    from autoforecast import study_tuning

    origin = "2024-01-05"
    panel = Panel(
        str(tmp_path / "training"),
        "series_id",
        "day",
        "value",
        target_estimand="observed_nonnegative",
    )
    context = {"origin": origin, "horizon": 2, "evaluation_contract": {"study_id": "known"}}
    inputs = SimpleNamespace(panel=panel, mapping_path=tmp_path / "mapping", verify=lambda: context)
    spark = SimpleNamespace(read=SimpleNamespace(parquet=lambda _: object()))
    output = tmp_path / "validation"

    def callback(request):
        artifact = request.output_dir / "artifacts"
        artifact.mkdir()
        (artifact / "trace.json").write_text("{}")
        return FitPredictResult(
            Forecast(
                "custom",
                str(request.output_dir / "forecasts"),
                origin,
                2,
                1,
                {"emitted": 2},
                target_estimand="observed_nonnegative",
            ),
            "test-format",
        )

    def fail_copy(*args, **kwargs):
        raise OSError("injected artifact copy failure")

    monkeypatch.setattr(study_tuning, "_retain_run_artifact_directory", fail_copy)
    with pytest.raises(OSError, match="injected artifact copy failure"):
        study_tuning._fit_window(
            spark,
            callback,
            inputs,
            output,
            model_id="custom",
            model_parameters={},
            training_parameters={},
            seed=1,
            metrics=["mae"],
            breakdown="none",
        )
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    print("ARTIFACT_EVENTS=" + json.dumps([(e["stage"], e["status"]) for e in events]))
    assert any(e["stage"] == "artifact_retention" and e["status"] == "FAIL" for e in events)


def test_workflow_seal_failure_never_announces_workflow_pass(tmp_path, monkeypatch):
    from mlops.run_bundle import RunBundle

    from autoforecast import study_workflow
    from autoforecast.test_study_tuning import config, implementation

    manifest = tmp_path / "benchmarks" / "known" / "benchmark.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recipe = {
        "metric": "mae",
        "frequency": "D",
        "validation_origins": ["2024-01-05"],
        "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07",
        "final_test_horizon": 2,
    }
    study = SimpleNamespace(
        study_id="known",
        _path=manifest,
        _sha256="a" * 64,
        _snapshots=SimpleNamespace(name=str(scratch)),
        _spark=object(),
        recipe=recipe,
        _record=lambda **_: {"sources": {}},
    )
    preflight = SimpleNamespace(
        study_id="known", study_path=manifest, study_sha256="a" * 64, report={}
    )
    preflight.report_path = tmp_path / "preflight-report.json"
    preflight.report_path.write_text(json.dumps(preflight.report))
    monkeypatch.setattr(study_workflow, "preflight_study", lambda *args, **kwargs: preflight)
    monkeypatch.setattr(
        study_workflow,
        "_prepare_window",
        lambda *args, **kwargs: SimpleNamespace(verify=lambda: {}),
    )
    monkeypatch.setattr(
        study_workflow,
        "_fit_window",
        lambda *args, **kwargs: {
            "forecast": {"path": "forecasts"},
            "model_path": "model",
            "score": {"evaluation_status": "completed"},
        },
    )
    original = RunBundle.finalize

    def fail_completed_seal(self, *, status, summary):
        if status == "completed":
            raise OSError("injected completion write failure")
        return original(self, status=status, summary=summary)

    monkeypatch.setattr(RunBundle, "finalize", fail_completed_seal)
    output = tmp_path / "runs"
    with pytest.raises(OSError, match="injected completion write failure"):
        study_workflow.run_forecast_workflow(
            study,
            fit_predict=lambda _: None,
            implementation=implementation(),
            config=config(tmp_path),
            fixed_parameters={},
            output_dir=output,
        )
    events = [
        json.loads(line)
        for f in output.glob("*/events.jsonl")
        for line in f.read_text().splitlines()
    ]
    print("SEAL_EVENTS=" + json.dumps([(e["stage"], e["status"]) for e in events]))
    assert not any(e["stage"] == "workflow" and e["status"] == "PASS" for e in events)


def test_workflow_loads_the_winner_with_the_search_checks(tmp_path, monkeypatch):
    """The winner's settings reuse the trial checks made by the search, in one scope."""
    from autoforecast import experiments, study_workflow
    from autoforecast.test_study_tuning import config, implementation

    manifest = tmp_path / "benchmarks" / "known" / "benchmark.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recipe = {
        "metric": "mae",
        "frequency": "D",
        "validation_origins": ["2024-01-05"],
        "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07",
        "final_test_horizon": 2,
    }
    study = SimpleNamespace(
        study_id="known",
        _path=manifest,
        _sha256="a" * 64,
        _snapshots=SimpleNamespace(name=str(scratch)),
        _spark=object(),
        recipe=recipe,
        _record=lambda **_: {"sources": {}},
    )
    preflight = SimpleNamespace(
        study_id="known", study_path=manifest, study_sha256="a" * 64, report={}
    )
    preflight.report_path = tmp_path / "preflight-report.json"
    preflight.report_path.write_text(json.dumps(preflight.report))
    monkeypatch.setattr(study_workflow, "preflight_study", lambda *args, **kwargs: preflight)
    scopes = []
    selected = SimpleNamespace(path=tmp_path / "search", winner_config=tmp_path / "winner.json")

    def tune(*args, **kwargs):
        scopes.append(experiments._CHECKED_ATTEMPTS.get())
        return selected

    def load(path):
        scopes.append(experiments._CHECKED_ATTEMPTS.get())
        raise OSError("stop after loading the winner")

    monkeypatch.setattr(study_workflow, "tune_study", tune)
    monkeypatch.setattr(study_workflow, "load_selected_configuration", load)
    profile = config(tmp_path, budget=1)
    profile.write_text(profile.read_text() + "selection:\n  stage: validation\n")
    with pytest.raises(OSError, match="stop after loading the winner"):
        study_workflow.run_forecast_workflow(
            study, fit_predict=lambda _: None, implementation=implementation(),
            config=profile, output_dir=tmp_path / "runs",
        )
    assert len(scopes) == 2 and scopes[0] is not None and scopes[0] is scopes[1]


@pytest.mark.parametrize("operation", ["artifact_retention", "scoring"])
def test_reused_validation_operation_has_current_failure_stage(tmp_path, monkeypatch, operation):
    import yaml

    from autoforecast import study_workflow
    from autoforecast.test_study_tuning import config, implementation

    manifest = tmp_path / "benchmarks" / "known" / "benchmark.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recipe = {
        "metric": "mae",
        "frequency": "D",
        "validation_origins": ["2024-01-05"],
        "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07",
        "final_test_horizon": 2,
    }
    study = SimpleNamespace(
        study_id="known",
        _path=manifest,
        _sha256="a" * 64,
        _snapshots=SimpleNamespace(name=str(scratch)),
        _spark=object(),
        recipe=recipe,
        _record=lambda **_: {"sources": {}},
    )
    preflight = SimpleNamespace(
        study_id="known", study_path=manifest, study_sha256="a" * 64, report={}
    )
    preflight.report_path = tmp_path / "preflight-report.json"
    preflight.report_path.write_text(json.dumps(preflight.report))
    panel = Panel(
        str(tmp_path / "training"),
        "series_id",
        "day",
        "value",
        target_estimand="observed_nonnegative",
    )
    monkeypatch.setattr(study_workflow, "preflight_study", lambda *args, **kwargs: preflight)
    monkeypatch.setattr(
        study_workflow,
        "_prepare_window",
        lambda *args, **kwargs: SimpleNamespace(
            verify=lambda: {"evaluation_contract": {"actuals": {"path": "actuals"}}}, panel=panel
        ),
    )
    profile = config(tmp_path, budget=1)
    value = yaml.safe_load(profile.read_text())
    value["selection"] = {"stage": "validation"}
    profile.write_text(yaml.safe_dump(value))
    forecast = Forecast(
        "custom-constant",
        str(tmp_path / "selected-forecast"),
        "2024-01-05",
        2,
        1,
        {"emitted": 2},
        target_estimand="observed_nonnegative",
    )
    window = {
        "model_artifacts": tmp_path / "model",
        "model_sha256": "a" * 64,
        "forecasts": forecast,
        "forecast_sha256": "b" * 64,
        "evaluation_contract": {"actuals": {"path": "actuals"}},
    }
    selected = SimpleNamespace(
        path=tmp_path / "search", winner_config=tmp_path / "winner.json", windows=(window,)
    )
    saved = {
        "parameters": {"model.level": 0.0, "training.offset": 0},
        "source": {},
        "profile_text": profile.read_text(),
    }
    monkeypatch.setattr(study_workflow, "tune_study", lambda *args, **kwargs: selected)
    monkeypatch.setattr(study_workflow, "load_tuning_result", lambda *args, **kwargs: selected)
    monkeypatch.setattr(
        study_workflow, "load_selected_configuration", lambda *args, **kwargs: saved
    )

    def copy(*args, **kwargs):
        if operation == "artifact_retention":
            raise OSError("injected selected artifact retention failure")
        return window["model_sha256"]

    monkeypatch.setattr(study_workflow, "_retain_run_artifact_directory", copy)
    monkeypatch.setattr(
        study_workflow, "snapshot_parquet_directory", lambda source, dest, **kwargs: dest
    )

    def fail_score(*args, **kwargs):
        raise OSError("injected selected scoring failure")

    monkeypatch.setattr(study_workflow, "score_study", fail_score)
    output = tmp_path / "runs"
    with pytest.raises(OSError, match="injected selected"):
        study_workflow.run_forecast_workflow(
            study,
            fit_predict=lambda _: None,
            implementation=implementation(),
            config=profile,
            output_dir=output,
        )
    events = [
        json.loads(line)
        for f in output.glob("**/events.jsonl")
        for line in f.read_text().splitlines()
    ]
    print(
        "REUSE_" + operation + "_EVENTS=" + json.dumps([(e["stage"], e["status"]) for e in events])
    )
    assert any(e["stage"] == operation and e["status"] == "FAIL" for e in events)


def test_completed_workflow_links_survive_secondary_audit_failure(tmp_path, monkeypatch):
    from autoforecast import study_workflow

    path = tmp_path / "completed-workflow"
    path.mkdir()
    (path / "artifact_manifest.json").write_text("{}")
    profile = tmp_path / "model.yaml"
    profile.write_text("{}")
    result = SimpleNamespace(path=path, run_id=path.name, study_id="known", status="completed")

    class Study:
        registration = {"sha256": "a" * 64}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class Registry:
        root = tmp_path
        runtime = {}
        store = SimpleNamespace(validate_destinations=lambda: None)

        def _registration_path(self, _):
            pass

        def load_study(self, _):
            return Study()

        def publish_results(self, *args, **kwargs):
            raise OSError("primary publication error")

    native = study_workflow._stage_event

    injected = []

    def broken_terminal_event(path, stage, status, **context):
        if stage == "run_attempt" and status == "FAIL":
            injected.append(True)
            raise OSError("secondary audit write error")
        return native(path, stage, status, **context)

    monkeypatch.setattr(study_workflow, "_stage_event", broken_terminal_event)
    monkeypatch.setattr(study_workflow, "run_forecast_workflow", lambda *args, **kwargs: result)
    with pytest.raises(OSError, match="primary publication error") as caught:
        study_workflow.run_study(
            "known",
            lambda _: None,
            registry=Registry(),
            model_config=profile,
            implementation={},
            publish=True,
        )
    assert injected == [True]
    print(
        "RECOVERY_LINKS="
        + json.dumps(
            {
                "workflow_path": str(getattr(caught.value, "workflow_path", None)),
                "audit_path": str(getattr(caught.value, "audit_path", None)),
            }
        )
    )
    assert caught.value.workflow_path == path
    assert caught.value.audit_path.exists()


@pytest.mark.parametrize("failed_write", ["sidecar", "event", "seal"])
def test_failure_sidecar_error_does_not_skip_terminal_stage_or_failed_seal(
    tmp_path, monkeypatch, failed_write,
):
    from mlops.run_bundle import RunBundle

    from autoforecast import study_workflow
    from autoforecast.test_study_tuning import config, implementation

    manifest = tmp_path / "benchmarks" / "known" / "benchmark.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recipe = {
        "metric": "mae",
        "frequency": "D",
        "validation_origins": ["2024-01-05"],
        "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07",
        "final_test_horizon": 2,
    }
    study = SimpleNamespace(
        study_id="known",
        _path=manifest,
        _sha256="a" * 64,
        _snapshots=SimpleNamespace(name=str(scratch)),
        _spark=object(),
        recipe=recipe,
        _record=lambda **_: {"sources": {}},
    )
    preflight = SimpleNamespace(
        study_id="known", study_path=manifest, study_sha256="a" * 64, report={}
    )
    preflight.report_path = tmp_path / "preflight-report.json"
    preflight.report_path.write_text(json.dumps(preflight.report))
    monkeypatch.setattr(study_workflow, "preflight_study", lambda *args, **kwargs: preflight)
    monkeypatch.setattr(
        study_workflow,
        "_prepare_window",
        lambda *args, **kwargs: SimpleNamespace(verify=lambda: {}),
    )

    def fail_fit(*args, **kwargs):
        raise RuntimeError("primary operation failure")

    monkeypatch.setattr(study_workflow, "_fit_window", fail_fit)
    native = RunBundle.write_json

    def fail_sidecar(self, name, record):
        if name == "failure.json" and failed_write == "sidecar":
            raise OSError("secondary sidecar write failure")
        return native(self, name, record)

    monkeypatch.setattr(RunBundle, "write_json", fail_sidecar)
    native_event, native_seal = study_workflow._stage_event, RunBundle.finalize

    def fail_event(path, stage, status, **context):
        if status == "FAIL" and failed_write == "event":
            raise OSError("secondary event failure")
        return native_event(path, stage, status, **context)

    def fail_seal(self, **kwargs):
        if failed_write == "seal":
            raise OSError("secondary seal failure")
        return native_seal(self, **kwargs)

    monkeypatch.setattr(study_workflow, "_stage_event", fail_event)
    monkeypatch.setattr(RunBundle, "finalize", fail_seal)
    output = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="primary operation failure"):
        study_workflow.run_forecast_workflow(
            study,
            fit_predict=lambda _: None,
            implementation=implementation(),
            config=config(tmp_path),
            fixed_parameters={},
            output_dir=output,
        )
    events = [
        json.loads(line)
        for f in output.glob("*/events.jsonl")
        for line in f.read_text().splitlines()
    ]
    completions = list(output.glob("*/completion.json"))
    print(
        "SECONDARY_SIDECAR="
        + json.dumps(
            {
                "events": [(e["stage"], e["status"]) for e in events],
                "completion_count": len(completions),
            }
        )
    )
    assert any(e["status"] == "FAIL" for e in events) == (failed_write != "event")
    assert len(completions) == (0 if failed_write == "seal" else 1)
    if completions:
        assert json.loads(completions[0].read_text())["status"] == "failed"
    assert bool(list(output.glob("*/failure.json"))) == (failed_write != "sidecar")


def test_validation_checkpoint_copy_failure_has_retention_fail_event(tmp_path, monkeypatch):
    import yaml
    from mlops import _safe_files

    from autoforecast import study_workflow
    from autoforecast.study_tuning import _resolve_options
    from autoforecast.test_study_tuning import config, implementation

    manifest = tmp_path / "benchmarks" / "known" / "benchmark.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recipe = {
        "metric": "mae",
        "frequency": "D",
        "validation_origins": ["2024-01-05"],
        "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07",
        "final_test_horizon": 2,
    }
    study = SimpleNamespace(
        study_id="known",
        _path=manifest,
        _sha256="a" * 64,
        _snapshots=SimpleNamespace(name=str(scratch)),
        _spark=object(),
        recipe=recipe,
        _record=lambda **_: {"sources": {}},
    )
    preflight = SimpleNamespace(
        study_id="known", study_path=manifest, study_sha256="a" * 64, report={}
    )
    preflight.report_path = tmp_path / "preflight-report.json"
    preflight.report_path.write_text(json.dumps(preflight.report))
    fixed = {"actuals": {"path": "actuals"}}
    monkeypatch.setattr(study_workflow, "preflight_study", lambda *args, **kwargs: preflight)
    monkeypatch.setattr(
        study_workflow,
        "_prepare_window",
        lambda *args, **kwargs: SimpleNamespace(verify=lambda: {"evaluation_contract": fixed}),
    )
    profile = config(tmp_path)
    value = yaml.safe_load(profile.read_text())
    value.pop("hpo")
    options = _resolve_options(study, {"selection": {"stage": "validation"}}, None)
    options["selection_boundaries"] = []
    old = tmp_path / "previous"
    (old / "validation").mkdir(parents=True)
    (old / "validation" / "trace.json").write_text("{}")
    (old / "artifact_manifest.json").write_text("{}")
    previous = SimpleNamespace(
        path=old,
        testing=None,
        selected_config=tmp_path / "winner.json",
        provenance={
            "study": {"path": str(manifest.resolve()), "sha256": "a" * 64},
            "options": options,
        },
        validation={"evaluation_contract": fixed},
    )
    saved = {
        "parameters": {"model.level": 0.0, "training.offset": 0},
        "source": {},
        "profile_text": yaml.safe_dump(value),
        "profile": value,
        "implementation": implementation(),
        "model_id": "custom-constant",
        "seed": 9,
    }
    monkeypatch.setattr(study_workflow, "load_forecast_workflow", lambda *args, **kwargs: previous)
    monkeypatch.setattr(
        study_workflow, "load_selected_configuration", lambda *args, **kwargs: saved
    )

    def fail_snapshot(*args, **kwargs):
        raise OSError("injected checkpoint retention failure")

    monkeypatch.setattr(_safe_files, "snapshot_regular_files", fail_snapshot)
    output = tmp_path / "runs"
    with pytest.raises(OSError, match="injected checkpoint retention failure"):
        study_workflow.run_forecast_workflow(
            study,
            fit_predict=lambda _: None,
            implementation=implementation(),
            validation_run=old,
            output_dir=output,
        )
    events = [
        json.loads(line)
        for f in output.glob("**/events.jsonl")
        for line in f.read_text().splitlines()
    ]
    print("CHECKPOINT_EVENTS=" + json.dumps([(e["stage"], e["status"]) for e in events]))
    assert any(e["stage"] == "artifact_retention" and e["status"] == "FAIL" for e in events)


def test_startup_records_sanitized_tracking_and_table_destinations(tmp_path, monkeypatch):
    from autoforecast import run_mlflow, study_workflow

    class Registry:
        root = tmp_path
        runtime = {
            "tracking": "mlflow",
            "publish": True,
            "tracking_uri": "https://user:password@tracking.example?token=secret#private",
            "mlflow_experiment": "42",
        }
        store = SimpleNamespace(
            validate_destinations=lambda: None,
            tables={"study_results": "generic.results", "forecasts": "generic.forecasts"},
        )

        def _registration_path(self, _):
            pass

        def load_study(self, _):
            raise RuntimeError("stop after tracking setup")

    monkeypatch.setattr(run_mlflow, "resolve_study_experiment", lambda **_: "42")
    with pytest.raises(RuntimeError, match="stop after tracking setup") as caught:
        study_workflow.run_study(
            "known",
            lambda _: None,
            registry=Registry(),
            model_config=tmp_path / "model.yaml",
            implementation={},
        )
    events = [json.loads(line) for line in caught.value.audit_path.read_text().splitlines()]
    startup = [e for e in events if e["stage"] in {"configuration", "tracking_setup"}]
    print("STARTUP_CONTEXT=" + json.dumps(startup))
    assert "https://tracking.example" in json.dumps(startup)
    assert all(secret not in json.dumps(startup)
               for secret in ("password", "token=secret", "private", "user:"))
    assert "42" in json.dumps(startup)
    assert "generic.results" in json.dumps(startup)


def test_summary_includes_model_and_retained_excluded_counts(tmp_path):
    from autoforecast.panel import Forecast
    from autoforecast.study_workflow import ForecastWorkflowResult

    score = {
        "reported_metrics": ["mae"],
        "mae": 1.0,
        "coverage": {"expected": 2, "original_expected": 4},
        "population": {"original_series": 2, "forecasted_series": 1},
        "availability": {"retained_series": 1},
    }
    result = ForecastWorkflowResult(
        tmp_path / "run",
        "completed",
        {},
        {
            "score": score,
            "forecasts": Forecast(
                "my_model", str(tmp_path / "forecasts"), "2024-01-01", 2, 1, {"emitted": 2}
            ),
        },
        None,
        {
            "study": {"path": str(tmp_path / "study" / "benchmark.json")},
            "registration": {"study_id": "parent"},
        },
        None,
        tmp_path / "selected",
        tmp_path / "search",
    )
    summary = result.summary()
    print("RESULT_SUMMARY=" + json.dumps(summary))
    assert "my_model" in json.dumps(summary)
    assert summary["retained_series"] == 1 and summary["excluded_series"] == 1


def test_preflight_emits_required_warning_for_permitted_series_exclusion(tmp_path, monkeypatch):
    from autoforecast import study_workflow
    from autoforecast.test_study_tuning import config, implementation

    manifest = tmp_path / "benchmarks" / "known" / "benchmark.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    recipe = {
        "metric": "mae",
        "frequency": "D",
        "validation_origins": ["2024-01-05"],
        "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07",
        "final_test_horizon": 2,
    }
    study = SimpleNamespace(
        study_id="known",
        _path=manifest,
        _sha256="a" * 64,
        _snapshots=SimpleNamespace(name=str(scratch)),
        _spark=object(),
        recipe=recipe,
        _record=lambda **_: {"sources": {}},
    )
    report = {
        "original_count": 2,
        "retained_count": 1,
        "excluded_ids": ["B"],
        "issues": [{"code": "insufficient_history", "severity": "availability"}],
    }
    preflight = SimpleNamespace(
        study_id="known", study_path=manifest, study_sha256="a" * 64, report=report
    )
    preflight.report_path = tmp_path / "preflight-report.json"
    preflight.report_path.write_text(json.dumps(preflight.report))
    monkeypatch.setattr(study_workflow, "preflight_study", lambda *args, **kwargs: preflight)

    def stop_after_preflight(*args, **kwargs):
        raise RuntimeError("stop after permitted exclusion")

    monkeypatch.setattr(study_workflow, "_prepare_window", stop_after_preflight)
    output = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="stop after permitted exclusion"):
        study_workflow.run_forecast_workflow(
            study,
            fit_predict=lambda _: None,
            implementation=implementation(),
            config=config(tmp_path),
            fixed_parameters={},
            output_dir=output,
        )
    events = [
        json.loads(line)
        for f in output.glob("*/events.jsonl")
        for line in f.read_text().splitlines()
    ]
    passed = [e for e in events if e["stage"] == "preflight" and e["status"] == "PASS"]
    print("PREFLIGHT_CONTEXT=" + json.dumps(passed))
    assert any(e.get("warnings") for e in passed)
    assert any(e.get("retained_series", e.get("retained_count")) == 1 for e in passed)


@pytest.mark.parametrize("options,exception,message", [
    ({"fit_predict": None}, TypeError, "fit_predict must be"),
    ({"validation_only": 1}, TypeError, "validation_only must be boolean"),
    ({"fixed_parameters": {}}, ValueError, "fixed_parameters requires config"),
    ({"fixed_parameters": [], "config": "unused.yaml"}, TypeError,
     "fixed_parameters must be a mapping"),
    ({"selected_config": "unused.json", "reuse_tuning": "unused"}, ValueError,
     "selected_config conflicts"),
    ({"selected_config": "unused.json", "tuning_origin": "2024-01-01"}, ValueError,
     "selected_config conflicts"),
    ({}, ValueError, "provide config to tune"),
])
def test_invalid_workflow_arguments_do_not_access_study_or_write_audit(
    tmp_path, options, exception, message,
):
    from autoforecast.study_workflow import run_forecast_workflow

    request = {"fit_predict": lambda _: pytest.fail("invalid request must not fit"),
               "output_dir": tmp_path / "runs", **options}
    with pytest.raises(exception, match=message):
        run_forecast_workflow(object(), **request)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("unknown", ["model_lable", "validation_rnu", "selected_confgi"])
def test_unknown_workflow_argument_is_rejected_before_study_access(tmp_path, unknown):
    from autoforecast.study_workflow import run_forecast_workflow

    with pytest.raises(TypeError, match="unexpected keyword"):
        run_forecast_workflow(
            object(), fit_predict=lambda _: pytest.fail("must not fit"), config="unused.yaml",
            fixed_parameters={}, output_dir=tmp_path / "runs", **{unknown: "unused"},
        )
    assert not (tmp_path / "runs").exists()
