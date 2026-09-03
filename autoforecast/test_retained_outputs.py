"""Retained outputs are useful after callback scratch space has disappeared."""

import json
import pathlib
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mlops.provenance import ProvenanceError
from mlops.run_bundle import load_completed_run_bundle

from autoforecast.experiments import ExperimentError, load_completed_attempt, open_local_attempt
from autoforecast.results import (
    comparison_plot,
    export_results,
    read_tuning_results,
    read_workflow_results,
)
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_experiments import _metadata
from autoforecast.test_study_tuning import config, freeze, implementation, open_study, predict
from autoforecast.test_study_tuning import spark as _spark
from autoforecast.test_study_workflow import profile, register


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def _assert_analysis_matches_retained_files(tables, manifest, *, workflow=None, search=None):
    """Use original evidence as the oracle, independently of reader/export output."""
    frozen = json.loads(manifest.read_text())
    assert tables.source_keys.equals(pq.read_table(manifest.parent / "source-keys"))
    windows = []
    if search is not None:
        for attempt in sorted(search.glob("trials/*/attempts/*")):
            if not (attempt / "completion.json").exists():
                continue
            metadata = json.loads((attempt / "metadata/attempt.json").read_text())
            evidence = json.loads((attempt / "evaluation/external-run.json").read_text())
            run_id = f"{search.name}/{metadata['trial_id']}/{metadata['attempt_id']}"
            windows.extend((attempt, window, run_id, "selection", metadata["trial_id"])
                           for window in evidence["windows"])
    if workflow is not None:
        record = json.loads((workflow / "result.json").read_text())
        for name, stage in (("validation", "validation"), ("testing", "test")):
            if name in record:
                windows.append((workflow, record[name], workflow.name, stage, None))
    expected_predictions, expected_scores = [], []
    for root, window, run_id, stage, variant_id in windows:
        labels = {
            "benchmark_id": frozen["benchmark_id"], "model_id": window["forecast"]["adapter"],
            "variant_id": variant_id, "run_id": run_id, "stage": stage,
            "origin": window["forecast"]["origin"],
        }
        for row in pq.read_table(root / window["forecast"]["path"]).to_pylist():
            expected_predictions.append({
                **labels, "target_date": row["date"], "series_id": row["series_id"],
                "prediction": row["yhat"], "status": "emitted",
            })
        evidence = window["score"].get("series_scores")
        if evidence is not None:
            expected_scores.extend({**row, **labels}
                                   for row in pq.read_table(evidence["path"]).to_pylist())
        returned = [row for row in tables.runs
                    if all(row.get(key) == labels[key]
                           for key in ("run_id", "stage", "origin", "model_id"))]
        assert len(returned) == 1
        assert returned[0]["score"] == window["score"]
        assert returned[0]["horizon"] == window["forecast"]["horizon"]
        assert returned[0]["frequency"] == window["forecast"]["frequency"]
    assert tables.rows.to_pylist() == expected_predictions
    assert (tables.series_scores.to_pylist() if tables.series_scores is not None else []) == (
        expected_scores
    )
    assert tables.metadata["comparison_scope"] == "fixed_study"
    assert tables.metadata["target_unit"] == frozen["recipe"]["target_unit"]


def test_attempt_guide_covers_failure_and_is_authenticated(tmp_path):
    attempt = open_local_attempt(tmp_path, study_id="search", trial_id="trial1",
                                 attempt_id="attempt1", metadata=_metadata(learning_rate=0.1))
    attempt.write_json("failure.json", {"error": "test failure"})
    attempt.finalize(status="failed", summary={})
    assert load_completed_attempt(attempt.path)[1]["status"] == "failed"
    guide = attempt.path / "RUN.md"
    assert "failure.json" in guide.read_text()
    guide.write_text("tampered")
    with pytest.raises(ExperimentError, match="artifact"):
        load_completed_attempt(attempt.path)


def test_workflow_outputs_retain_diagnostics_reuse_and_analysis(tmp_path, spark, monkeypatch):
    # Population: the fixture's two registered series, no exclusions, two horizon cells each.
    manifest = register(tmp_path / "input", spark)
    cfg = profile(tmp_path, selection={"stage": "validation"})
    scratch = []

    def fit(request):
        scratch.append(request.output_dir)
        result = predict(request)
        artifacts = request.output_dir / "artifacts"
        artifacts.mkdir()
        (artifacts / "training-log.jsonl").write_text(json.dumps({"purpose": request.purpose}))
        (request.output_dir / "scratch-only.txt").write_text("not retained")
        return result

    with open_study(manifest, spark) as study:
        preflight = study.preflight(config=cfg, output_dir=tmp_path / "explicit-preflight")
        assert preflight.report_path.parent.parent == tmp_path / "explicit-preflight"
        assert preflight.report_path.is_file()
        checkpoint = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), config=cfg, validation_only=True,
            output_dir=tmp_path / "experiment",
        )
        calls = len(scratch)
        assert calls == 2  # Two trials; outer validation reuses the selected model.
        run = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), validation_run=checkpoint.path,
            output_dir=tmp_path / "experiment",
        )
        assert len(scratch) == calls + 1
        from mlops import _safe_files

        original_copy = _safe_files.snapshot_regular_files

        def corrupt_copy(source, destination, *args, **kwargs):
            result = original_copy(source, destination, *args, **kwargs)
            target = destination / "artifacts/training-log.jsonl"
            if target.exists():
                target.write_text("corrupted during copy")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(_safe_files, "snapshot_regular_files", corrupt_copy)
            with pytest.raises(ValueError, match="callback artifacts changed during retention"):
                study.run_forecast_workflow(
                    fit_predict=fit, implementation=implementation(), validation_run=checkpoint.path,
                    output_dir=tmp_path / "bad-resume",
                )
        assert len(scratch) == calls + 1  # Corruption must stop the final fit.
        repeated = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), config=cfg,
            fixed_parameters={"model.level": 3.0}, validation_only=True,
        )
        assert repeated.path.parent == manifest.parent.parent.parent / "study-workflows" / study.study_id
        identical = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), config=cfg,
            fixed_parameters={"model.level": 3.0}, validation_only=True,
        )
        changed = study.run_forecast_workflow(
            fit_predict=fit, implementation=implementation(), config=cfg,
            fixed_parameters={"model.level": 4.0}, validation_only=True,
        )
        assert len({item.path for item in (repeated, identical, changed, run, checkpoint)}) == 5
        assert repeated.validation["parameter_sha256"] == identical.validation["parameter_sha256"]
        assert changed.validation["parameter_sha256"] != repeated.validation["parameter_sha256"]
        assert re.fullmatch(r"workflow-\d{8}t\d{6}z-[0-9a-f]{32}", run.path.name)

    assert all(not path.exists() for path in scratch)
    for retained, stages in ((checkpoint, ("validation",)), (run, ("validation", "test"))):
        for stage in stages:
            expected = {"purpose": "selection" if stage == "validation" else "refit"}
            assert (retained.path / stage / "artifacts/training-log.jsonl").read_text() == (
                json.dumps(expected)
            )
        assert not list(retained.path.rglob("scratch-only.txt"))
        assert "training-log.jsonl" in (retained.path / "RUN.md").read_text()
        assert list((retained.path / "preflight").glob("*/report.json"))
        load_forecast_workflow(retained.path)
    for retained in (repeated, identical, changed):
        assert (retained.path / "validation/artifacts/training-log.jsonl").read_text() == (
            json.dumps({"purpose": "validation"})
        )
    trial_logs = list(checkpoint.tuning_path.glob("trials/*/attempts/*/artifacts/training-log.jsonl"))
    assert len(trial_logs) == 2
    assert all(path.read_text() == json.dumps({"purpose": "selection"}) for path in trial_logs)

    # Original checkpoint includes its HPO attempts; continuation records the reused source.
    tables = read_workflow_results(checkpoint.path)
    _assert_analysis_matches_retained_files(
        tables, manifest, workflow=checkpoint.path, search=checkpoint.tuning_path,
    )
    assert tables.rows.num_rows == 12  # Two trial windows plus one validation window, four cells each.
    assert set(tables.rows["stage"].to_pylist()) == {"selection", "validation"}
    validation_only = read_workflow_results(checkpoint.path, include_tuning=False)
    _assert_analysis_matches_retained_files(validation_only, manifest, workflow=checkpoint.path)
    assert validation_only.rows.num_rows == 4
    assert set(validation_only.rows["stage"].to_pylist()) == {"validation"}
    assert validation_only.metadata["includes_tuning"] is False
    tuned = read_tuning_results(checkpoint.tuning_path)
    _assert_analysis_matches_retained_files(tuned, manifest, search=checkpoint.tuning_path)
    assert tuned.rows.num_rows == 8
    assert (checkpoint.tuning_path / "RUN.md").is_file()
    assert "report.html" in (checkpoint.tuning_path / "RUN.md").read_text()
    resumed = read_workflow_results(run.path)
    _assert_analysis_matches_retained_files(resumed, manifest, workflow=run.path)
    assert resumed.rows.num_rows == 8
    assert resumed.series_scores is not None
    assert resumed.source_keys.num_rows == 2
    destination = tmp_path / "analysis"
    export_results(resumed, destination)
    export_results(resumed, destination)
    assert pq.read_table(destination / "predictions.parquet").equals(resumed.rows)
    assert json.loads((destination / "metadata.json").read_text())["workflow_path"] == str(run.path)
    assert "predictions.parquet" in (destination / "RUN.md").read_text()
    for name, result in (("checkpoint", tables), ("workflow", resumed), ("tuning", tuned)):
        plot = comparison_plot(result, series_id=result.rows["series_id"][0].as_py(),
                               output=tmp_path / f"{name}.png")
        assert plot.stat().st_size > 0
    (run.path / "validation/artifacts/training-log.jsonl").write_text("changed")
    with pytest.raises(ProvenanceError, match="artifact"):
        read_workflow_results(run.path)


def test_callback_failure_keeps_logs_and_symlinks_are_rejected(tmp_path, spark):
    manifest = register(tmp_path / "input", spark)
    cfg = profile(tmp_path)
    with open_study(manifest, spark) as study:
        def failed(request):
            artifacts = request.output_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "training-log.jsonl").write_text('{"step": 1}\n')
            raise RuntimeError("training stopped")

        with pytest.raises(RuntimeError, match="training stopped"):
            study.run_forecast_workflow(
                fit_predict=failed, implementation=implementation(), config=cfg,
                fixed_parameters={}, output_dir=tmp_path / "failure",
            )
        path = next((tmp_path / "failure").glob("workflow-*"))
        assert load_completed_run_bundle(path)["status"] == "failed"
        assert (path / "validation/artifacts/training-log.jsonl").read_text() == '{"step": 1}\n'
        assert "training stopped" in (path / "failure.json").read_text()

        def linked(request):
            result = predict(request)
            artifacts = request.output_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "linked.json").symlink_to(cfg)
            return result

        with pytest.raises(ValueError, match="symlink"):
            study.run_forecast_workflow(
                fit_predict=linked, implementation=implementation(), config=cfg,
                fixed_parameters={}, output_dir=tmp_path / "linked",
            )
        path = next((tmp_path / "linked").glob("workflow-*"))
        assert load_completed_run_bundle(path)["status"] == "failed"
        assert not (path / "validation/artifacts/linked.json").exists()


def test_hourly_tuning_export_preserves_times_and_null_scores(tmp_path, spark):
    manifest = freeze(tmp_path / "input", spark, hourly=True, missing="all")
    with open_study(manifest, spark) as study:
        tuned = study.tune(fit_predict=predict, config=config(tmp_path, budget=1),
                           implementation=implementation(), output_dir=tmp_path / "search")
    result = read_tuning_results(tuned.path)
    _assert_analysis_matches_retained_files(result, manifest, search=tuned.path)
    assert result.rows.schema.field("target_date").type == pa.timestamp("us")
    assert {value.hour for value in result.rows["target_date"].to_pylist()} == {3, 4}
    assert result.rows.num_rows == 4
    assert result.series_scores is not None
    assert any(value is None for value in result.series_scores["rmse"].to_pylist())
    assert result.source_keys.schema == pq.read_table(manifest.parent / "source-keys").schema
    destination = tmp_path / "export"
    export_results(result, destination)
    assert pq.read_table(destination / "series-scores.parquet").equals(result.series_scores)
    (destination / "metadata.json").write_text("conflict")
    with pytest.raises(OSError, match="differ|conflict|exist"):
        export_results(result, destination)


def test_standalone_evaluation_destination_and_guide(tmp_path, spark):
    manifest = register(tmp_path / "input", spark)
    cfg = profile(tmp_path)
    with open_study(manifest, spark) as study:
        run = study.run_forecast_workflow(fit_predict=predict, implementation=implementation(),
                                        config=cfg, fixed_parameters={}, output_dir=tmp_path / "fit")
        result = study.evaluate_forecasts(
            validation_forecasts=run.validation["forecasts"], testing_forecasts=run.testing["forecasts"],
            model_id="custom-constant", output_dir=tmp_path / "scoring",
        )
    path = pathlib.Path(result["result_path"]).parent
    assert path.parent == tmp_path / "scoring"
    assert load_completed_run_bundle(path)["status"] == "completed"
    assert "validation/score.json" in (path / "RUN.md").read_text()


@pytest.mark.parametrize("failed_call,expected_rows", [(1, 0), (2, 4), (3, 8), (5, 16)])
def test_failed_search_and_later_workflow_failure_keep_analysis_navigation(
    tmp_path, spark, monkeypatch, failed_call, expected_rows,
):
    # Population: two registered series, two cells per window, no exclusions.
    manifest = register(tmp_path / "input", spark)
    cfg = profile(tmp_path, selection={"origins": "all"})
    calls = 0

    def fail_after_completed_windows(request):
        nonlocal calls
        calls += 1
        artifacts = request.output_dir / "artifacts"
        artifacts.mkdir()
        (artifacts / "progress.json").write_text(json.dumps({"call": calls}))
        if calls == failed_call:
            raise RuntimeError("declared training failure")
        return predict(request)

    with open_study(manifest, spark) as study:
        with pytest.raises(RuntimeError, match="declared training failure"):
            study.run_forecast_workflow(
                fit_predict=fail_after_completed_windows, implementation=implementation(),
                config=cfg, output_dir=tmp_path / "experiment",
            )
    workflow = next((tmp_path / "experiment").glob("workflow-*"))
    assert load_completed_run_bundle(workflow)["status"] == "failed"
    failure = json.loads((workflow / "failure.json").read_text())
    search = pathlib.Path(failure["tuning_path"])
    assert search.is_dir()
    assert json.loads((workflow / "metadata/tuning-reference.json").read_text())["path"] == str(search)
    assert "metadata/tuning-reference.json" in (workflow / "RUN.md").read_text()
    assert (search / "RUN.md").is_file()
    tables = read_tuning_results(search)
    _assert_analysis_matches_retained_files(tables, manifest, search=search)
    assert tables.rows.num_rows == expected_rows
    assert tables.source_keys.num_rows == 2
    assert tables.metadata["search_status"] == ("completed" if failed_call == 5 else "failed")
    assert (search / "selection.json").exists() is (failed_call == 5)
    export_results(tables, tmp_path / "analysis")
    assert pq.read_table(tmp_path / "analysis/predictions.parquet").num_rows == expected_rows
    if failed_call == 5:
        assert failure["phase"] == "validation"
        selection = search / "selection.json"
        selected_bytes = selection.read_bytes()
        selection.unlink()
        with pytest.raises(ValueError, match="selected search or a sealed failed search attempt"):
            read_tuning_results(search)
        selection.write_bytes(selected_bytes)
        return
    failed = next(row for row in tables.runs if row["status"] == "failed")
    assert failed["failure"]["message"] == "declared training failure"
    attempt = pathlib.Path(failed["path"])
    assert (attempt / "metadata/external-profile.yaml").read_bytes() == cfg.read_bytes()
    assert failed["artifacts_paths"]
    retained_context = attempt / "evaluation/external-run.json"
    original = retained_context.read_bytes()
    retained_context.write_text("{}")
    with pytest.raises(ExperimentError, match="artifact"):
        read_tuning_results(search)
    retained_context.write_bytes(original)

    from autoforecast.test_study_workflow import _reseal_attempt

    for target, replacement in (
        (retained_context, json.dumps({**json.loads(original), "context": {
            **json.loads(original)["context"], "seed": 900,
        }}).encode()),
        (attempt / "metadata/external-profile.yaml", cfg.read_bytes() + b"\n# changed\n"),
    ):
        before = target.read_bytes()
        target.write_bytes(replacement)
        _reseal_attempt(attempt)
        with pytest.raises(ValueError, match="context, profile or search identity"):
            read_tuning_results(search)
        target.write_bytes(before)
        _reseal_attempt(attempt)
    assert read_tuning_results(search).rows.num_rows == expected_rows
    if json.loads(original)["windows"]:
        changed = json.loads(original)
        changed["windows"][0]["score"][changed["context"]["metric"]] = None
        retained_context.write_text(json.dumps(changed))
        _reseal_attempt(attempt)
        with pytest.raises(ValueError, match="finite completed window objective"):
            read_tuning_results(search)
        retained_context.write_bytes(original)
        _reseal_attempt(attempt)
    if failed_call == 3:
        completed = next(row for row in tables.runs if row["status"] == "completed")
        completed_path = pathlib.Path(completed["path"])
        terminal_path = completed_path / "completion.json"
        terminal_bytes = terminal_path.read_bytes()
        terminal = json.loads(terminal_bytes)
        terminal["summary"]["validation_objective"] += 1.0
        terminal_path.write_text(json.dumps(terminal))
        _reseal_attempt(completed_path)
        with pytest.raises(ValueError, match="objective differs from retained window scores"):
            read_tuning_results(search)
        terminal_path.write_bytes(terminal_bytes)
        _reseal_attempt(completed_path)

    import hashlib

    from autoforecast.study_tuning import _json_bytes

    metadata_path = attempt / "metadata/attempt.json"
    metadata_bytes = metadata_path.read_bytes()
    changed = json.loads(original)
    changed["context"]["study"]["sha256"] = "0" * 64
    metadata = json.loads(metadata_bytes)
    metadata["comparability_fingerprint"] = hashlib.sha256(
        _json_bytes(changed["context"])).hexdigest()
    metadata_path.write_bytes(_json_bytes(metadata))
    retained_context.write_bytes(_json_bytes(changed))
    _reseal_attempt(attempt)
    with pytest.raises(ValueError, match="context, profile or search identity"):
        read_tuning_results(search)
    metadata_path.write_bytes(metadata_bytes)
    retained_context.write_bytes(original)
    _reseal_attempt(attempt)

    from autoforecast import study_results

    original_read = study_results._retained_json
    guide = attempt / "RUN.md"
    guide_bytes = guide.read_bytes()
    switched = False

    def switch_sealed_attempt(root, relative, manifest):
        nonlocal switched
        value = original_read(root, relative, manifest)
        if root / relative == retained_context and not switched:
            switched = True
            guide.write_bytes(guide_bytes + b"\nChanged during evidence read.\n")
            _reseal_attempt(attempt)
        return value

    with monkeypatch.context() as patch:
        patch.setattr(study_results, "_retained_json", switch_sealed_attempt)
        with pytest.raises(ValueError, match="attempt changed during analysis"):
            read_tuning_results(search)
    guide.write_bytes(guide_bytes)
    _reseal_attempt(attempt)
    original_tables = study_results._tables

    def change_attempt_inventory(*args, **kwargs):
        result = original_tables(*args, **kwargs)
        (search / "trials/injected/attempts/unsealed").mkdir(parents=True)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(study_results, "_tables", change_attempt_inventory)
        with pytest.raises(ValueError, match="attempts changed during analysis"):
            read_tuning_results(search)


def test_campaign_keeps_original_yaml_separate_from_normalized_profile(tmp_path, spark):
    from autoforecast.test_study_campaign import models

    manifest = register(tmp_path / "input", spark)
    configs = tmp_path / "configs"
    configs.mkdir()
    declarations = models(configs)[:1]
    source = pathlib.Path(declarations[0].config)
    original = b"# Exact user source, including this comment.\n" + source.read_bytes()
    source.write_bytes(original)
    with open_study(manifest, spark) as study:
        default_root = (manifest.parent.parent.parent / "study-campaigns" / study.study_id).resolve()
        result = study.run_campaign(
            models=declarations, name="retained-source", execution="sequential",
        )
        campaign = pathlib.Path(result["campaign_path"])
        assert campaign.parent == default_root
        retained = campaign / "profiles/candidate-0.yaml"
        assert retained.read_bytes() == original
        assert "profiles/candidate-0.yaml" in (campaign / "RUN.md").read_text()
        request = json.loads(next(default_root.glob("request-*/request.json")).read_text())
        assert request["models"][0]["original_profile_text"].encode() == original
        assert list((campaign / "models/candidate-0/validation").glob("profile.json"))
        repeated = study.run_campaign(
            models=declarations, name="retained-source", execution="sequential",
        )
        assert repeated["campaign_path"] == str(campaign)
        retained.write_bytes(original + b"# replaced\n")
        with pytest.raises(ValueError, match="retained original profile differs"):
            study.run_campaign(
                models=declarations, name="retained-source", execution="sequential",
                )
