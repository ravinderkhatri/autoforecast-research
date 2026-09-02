"""External models use the existing HPO engine and the frozen inner scoring window."""

import dataclasses
import datetime as dt
import hashlib
import json
import math
import pathlib
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pyspark.sql import SparkSession

from autoforecast import benchmark
from autoforecast.experiments import ExperimentError, require_heldout_evaluation_approval
from autoforecast.hpo import HPOError
from autoforecast.panel import Forecast, ForecastError
from autoforecast.score_spark import ScoreError
from autoforecast.study import load_study
from autoforecast.study_tuning import FitPredictResult, load_tuning_result
from autoforecast.tuning import TuningError, loads_external_tuning_spec, loads_tuning_spec
from provenance.lineage import LineageError


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]").appName("external-tuning-tests")
               .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false")
               .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def implementation():
    return {"name": "test-constant", "version": "1", "source_sha256": "a" * 64,
            "dependencies": {"pyarrow": pa.__version__}}


def config(root, *, budget=2, backend="optuna", seed=9):
    path = root / "external.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 2, "model": {"id": "custom-constant", "adapter": "external"},
        "hyperparameters": {
            "model.level": {"default": 0.0, "valid": {"kind": "float", "low": 0.0, "high": 10.0, "log": False},
                            "search": {"kind": "categorical", "values": [0.0, 10.0]}},
            "training.offset": {"default": 0, "valid": {"kind": "int", "low": 0, "high": 0, "step": 1},
                                "search": {"kind": "int", "low": 0, "high": 0, "step": 1}},
        }, "hpo": {"backend": backend, "max_trials": budget, "seed": seed},
    }))
    return path


def freeze(root, spark, *, hourly=False, monthly=False, windows=True, missing=False, metric="rmse", zero_targets=False):
    # Population: two series A/B, zero exclusions, four cells in the inner window.
    root.mkdir(parents=True, exist_ok=True)
    def point(day):
        if monthly:
            return dt.date(2024, day, 1)
        return dt.datetime(2024, 1, 1) + dt.timedelta(hours=day - 1) if hourly else dt.date(2024, 1, day)
    source = root / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"sku": sid, "at": point(day), "units": None if missing and sid == "B" and (day in (4, 5) if missing == "all" else day == 5) else 0.0 if zero_targets and day in (4, 5) else float(day)}
        for sid in ("A", "B") for day in range(1, 10)
    ]), source)
    dataset = root / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"schema_version": 1, "dataset": {"id": "external-test", "panel": {
        "series_col": "sku", "time_col": "at", "target_col": "units", "frequency": "H" if hourly else "MS" if monthly else "D",
        "season_length": 1, "target_estimand": "observed_nonnegative",
    }}}))
    recipe = {"schema_version": 2, "source_keys": ["sku"], "training_start": point(1).isoformat(),
              "frequency": "H" if hourly else "MS" if monthly else "D", "validation_origins": [point(5).isoformat()],
              "validation_horizon": 2, "final_train_cutoff": point(7).isoformat(), "final_test_horizon": 2,
              "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
              "feature_availability": {}, "metric": metric, "actual_missing": "mask" if missing else "error"}
    if windows:
        recipe["tuning_windows"] = [{"origin": point(3).isoformat(), "horizon": 2}]
        if windows == "multiple":
            recipe["tuning_windows"].append({"origin": point(4).isoformat(), "horizon": 1})
    refs = None
    if metric == "wrmsse":
        refs = root / "refs.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"keys": {"sku": sid}, "stage": stage, "origin": point(day).isoformat(), "weight": weight, "scale": 1.0}
            for sid, weight in (("A", 0.8), ("B", 0.2))
            for stage, day in (("tuning", 3), ("validation", 5), ("test", 7))
        ]), refs)
    return benchmark.freeze_benchmark(contract_path=dataset, sources=dict.fromkeys(
        ("selection", "final_training", "validation_score", "final_score"), source,
    ), recipe=recipe, scoring_references=refs, output_root=root / ".autoforecast/benchmarks", spark=spark)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory, spark):
    return freeze(tmp_path_factory.mktemp("external-study"), spark)


def predict(request):
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
    (model / "constant.json").write_text(json.dumps(request.model_parameters))
    return FitPredictResult(Forecast(
        adapter=request.model_id, path=str(forecasts), origin=request.origin,
        horizon=len(request.forecast_dates), n_series=len(rows),
        disposition={"emitted": len(rows) * len(request.forecast_dates)},
        target_estimand=request.panel.target_estimand, frequency=request.panel.frequency,
    ), "constant-json-v1")


def open_study(manifest, spark):
    return load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark)


def test_custom_model_tunes_scores_reloads_and_keeps_outer_labels_out(frozen, spark, tmp_path):
    seen = []
    input_paths = []
    profile = config(tmp_path, budget=4)
    identity = implementation()
    def fit(request):
        input_paths.append(pathlib.Path(request.panel.path))
        history = request.training_data.collect()
        assert len(history) == 6
        assert {row["units"] for row in history} == {1.0, 2.0, 3.0}
        # Verify physical bytes, not just a lazy Spark filter over future labels.
        assert max(row["at"] for row in pq.read_table(request.panel.path).to_pylist()) == dt.date(2024, 1, 3)
        assert request.forecast_dates == (dt.date(2024, 1, 4), dt.date(2024, 1, 5))
        assert request.seed == 9 and not hasattr(request, "study")
        seen.append(request.model_parameters["level"])
        identity["version"] = "changed-during-callback"
        result = predict(request)
        request.model_parameters["level"] = -999  # mutation must not rewrite HPO evidence
        return result
    with open_study(frozen, spark) as study:
        result = study.tune(fit_predict=fit, config=profile, implementation=identity)
    assert input_paths and all(not path.exists() for path in input_paths)
    assert len(seen) == 4 and set(seen) == {0.0, 10.0}
    assert result.parameters == {"model.level": 0.0, "training.offset": 0}
    assert result.tuning_score == pytest.approx(math.sqrt((4**2 + 5**2) / 2))
    assert result.provenance["implementation"] == implementation()
    assert result.provenance["evaluation_contract"]["boundary"].startswith("tuning/")
    assert len(result.trial_history) == 4 and result.model_id == "custom-constant"
    reloaded = load_tuning_result(result.path)
    assert reloaded.parameters == result.parameters
    assert reloaded.tuning_score == result.tuning_score
    assert result.model_format == reloaded.model_format == "constant-json-v1"
    assert reloaded.provenance == result.provenance
    selected = json.loads((result.path / "selection.json").read_text())["selected"][0]
    assert result.trial_id == reloaded.trial_id == selected["trial_id"]
    assert json.loads((reloaded.model_artifacts / "constant.json").read_text()) == {"level": 0.0}
    assert {row["yhat"] for row in pq.read_table(reloaded.forecasts.path).to_pylist()} == {0.0}
    assert spark.range(1).count() == 1
    # Every trial's stored forecast and objective prove sampled values reached model execution.
    for entry in result.trial_history:
        attempt = result.path / entry["attempt_path"]
        metadata = json.loads((attempt / "metadata/attempt.json").read_text())
        data = metadata["data"]
        assert data["fingerprint"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
        assert data["observed_date_range"] == ["2024-01-01", "2024-01-09"]
        assert data["training_date_range"] == ["2024-01-01", "2024-01-03"]
        assert data["validation_target_date_range"] == ["2024-01-04", "2024-01-05"]
        assert data["test_target_date_range"] == ["2024-01-08", "2024-01-09"]
        _, bound = benchmark.verify_benchmark(frozen, data["fingerprint"])
        for role, start, end in (("selection", "2024-01-01", "2024-01-05"),
                                 ("final_score", "2024-01-08", "2024-01-09")):
            rows = pq.read_table(benchmark.artifact_path(frozen, bound, role)).to_pylist()
            assert min(r["at"] for r in rows).isoformat() == start
            assert max(r["at"] for r in rows).isoformat() == end
        level = metadata["hyperparameters"]["model.level"]
        assert entry["validation_objective"] == pytest.approx(math.sqrt(((level-4)**2 + (level-5)**2)/2))
        assert {r["yhat"] for r in pq.read_table(attempt / "forecasts").to_pylist()} == {level}
        assert not (attempt / "evaluation/governed-run.json").exists()
    selection_path = result.path / "selection.json"
    original_selection = selection_path.read_bytes()
    try:
        changed = json.loads(original_selection)
        changed.update(top_k=2, selected=changed["candidates"][:2])
        selection_path.write_text(json.dumps(changed))
        with pytest.raises(ExperimentError, match="top_k"):
            load_tuning_result(result.path)
        extra = changed["selected"][1]
        with pytest.raises(ExperimentError, match="top_k"):
            require_heldout_evaluation_approval(
                result.path, trial_id=extra["trial_id"], attempt_id=extra["attempt_id"],
            )
    finally:
        selection_path.write_bytes(original_selection)
    other = next(entry for entry in result.trial_history if entry["trial_id"] != result.trial_id)
    for artifact in (
        result.model_artifacts / "constant.json", result.path / "selection.json",
        next(pathlib.Path(result.forecasts.path).glob("*.parquet")),
        result.model_artifacts.parent / "metadata/external-profile.yaml",
        result.model_artifacts.parent / "evaluation/external-run.json",
        result.path / other["attempt_path"] / "model/constant.json",
    ):
        original = artifact.read_bytes()
        try:
            artifact.write_bytes(original + b" ") if artifact.name != "selection.json" else artifact.write_text(
                json.dumps({**json.loads(original), "selected": []}))
            with pytest.raises(ExperimentError):
                load_tuning_result(result.path)
        finally:
            artifact.write_bytes(original)


@pytest.mark.parametrize("defect", ["origin", "missing", "duplicate", "series", "model", "empty-model", "linked-model", "input", "study", "return", "adapter", "horizon", "frequency", "mapping", "escaped", "format", "forecast-return"])
def test_bad_callback_cannot_publish_selection(frozen, spark, tmp_path, defect):
    local = tmp_path / "registry"
    shutil.copytree(frozen.parent, local / frozen.parent.name)
    manifest = local / frozen.parent.name / "benchmark.json"
    scratch = []
    input_paths = []
    def bad(request):
        scratch.append(request.output_dir)
        input_paths.append(pathlib.Path(request.panel.path))
        result = predict(request)
        if defect in {"adapter", "horizon", "frequency"}:
            value = {"adapter": "wrong-model", "horizon": 1, "frequency": "W"}[defect]
            return dataclasses.replace(result, forecasts=dataclasses.replace(result.forecasts, **{defect: value}))
        if defect == "mapping":
            # Mutate the physical mapping directory whose digest is owned by the bridge.
            mapping = pathlib.Path(request.series_mapping.inputFiles()[0].removeprefix("file:"))
            mapping.write_bytes(b"changed")
        if defect == "escaped":
            external = tmp_path / "escaped"
            shutil.copytree(request.output_dir / "predictions", external)
            return dataclasses.replace(result, forecasts=dataclasses.replace(result.forecasts, path=str(external)))
        if defect == "forecast-return":
            return dataclasses.replace(result, forecasts=None)
        if defect == "format":
            return dataclasses.replace(result, model_format=" ")
        if defect == "origin":
            return dataclasses.replace(result, forecasts=dataclasses.replace(result.forecasts, origin="2024-01-05"))
        if defect in {"missing", "duplicate", "series"}:
            file = request.output_dir / "predictions/part.parquet"
            rows = pq.read_table(file).to_pylist()
            if defect == "missing":
                rows.pop()
            elif defect == "duplicate":
                rows[-1] = rows[0]
            else:
                rows[0]["series_id"] = "unknown"
            pq.write_table(pa.Table.from_pylist(rows), file)
        if defect == "model":
            shutil.rmtree(request.output_dir / "model")
        if defect == "empty-model":
            (request.output_dir / "model/constant.json").unlink()
        if defect == "linked-model":
            (request.output_dir / "model/link").symlink_to(request.output_dir / "predictions/part.parquet")
        if defect == "input":
            file = next(pathlib.Path(request.panel.path).glob("*.parquet"))
            file.write_bytes(b"changed")
        if defect == "study":
            manifest.write_bytes(manifest.read_bytes() + b" ")
        if defect == "return":
            return 1.0
        return result
    with open_study(manifest, spark) as study, pytest.raises((HPOError, ScoreError, ForecastError, LineageError, benchmark.BenchmarkError)):
        study.tune(fit_predict=bad, config=config(tmp_path, budget=1), implementation=implementation())
    roots = list((tmp_path / "study-tuning/studies").glob("*"))
    assert len(roots) == 1 and not (roots[0] / "selection.json").exists()
    completions = list(roots[0].glob("trials/*/attempts/*/completion.json"))
    assert len(completions) == 1
    assert json.loads(completions[0].read_text())["status"] == "failed"
    assert input_paths and all(not path.exists() for path in input_paths)
    assert scratch and all(not path.exists() for path in scratch)
    assert spark.range(1).count() == 1


@pytest.mark.parametrize("hourly,monthly,metric,missing", [(True, False, "rmse", False), (False, False, "wrmsse", False), (False, False, "wrmsse", True), (False, True, "mae", False)])
def test_frequency_and_frozen_metric_policy(spark, tmp_path, hourly, monthly, metric, missing):
    manifest = freeze(tmp_path / "study", spark, hourly=hourly, monthly=monthly, metric=metric, missing=missing)
    with open_study(manifest, spark) as study:
        result = study.tune(fit_predict=predict, config=config(tmp_path, budget=1), implementation=implementation())
    level = result.parameters["model.level"]
    full = math.sqrt(((level-4)**2 + (level-5)**2)/2)
    expected = 0.8 * full + 0.2 * abs(level-4) if missing else full
    if metric == "mae":
        expected = (abs(level-4) + abs(level-5))/2
    assert result.tuning_score == pytest.approx(expected)
    assert result.forecasts.frequency == ("H" if hourly else "MS" if monthly else "D")
    score = json.loads((result.model_artifacts.parent / "evaluation/score.json").read_text())
    assert score["coverage"]["zero_filled"] == 0
    assert score["coverage"]["scored"] == (3 if missing else 4)


@pytest.mark.parametrize("windows,origin,error", [(False, None, "declared before"), ("multiple", None, "explicitly"), (True, "2024-01-05", "absent")])
def test_missing_or_ambiguous_window_fails_before_callback(spark, tmp_path, windows, origin, error):
    manifest = freeze(tmp_path / "study", spark, windows=windows)
    with open_study(manifest, spark) as study, pytest.raises(HPOError, match=error):
        study.tune(fit_predict=lambda _: pytest.fail("callback called"), config=config(tmp_path),
                   implementation=implementation(), tuning_origin=origin)


@pytest.mark.parametrize("option,value,error", [("backend", "ray", "sequential optuna"), ("budget", True, "positive integer"), ("budget", 0, "positive integer"), ("seed", -1, "seed"), ("seed", 2**32, "seed"), ("seed", True, "seed")])
def test_invalid_search_settings_fail_before_study_access(tmp_path, option, value, error):
    from autoforecast.study import Study
    with pytest.raises(TuningError, match=error):
        Study.tune(object(), fit_predict=predict, config=config(tmp_path, **{option: value}), implementation=implementation())


@pytest.mark.parametrize("field,value", [("name", ""), ("version", 1), ("source_sha256", "bad"), ("dependencies", {"pkg": 1}), ("dependencies", {"": "1"}), ("dependencies", {"pkg": " "})])
def test_invalid_implementation_fails_before_data_access(tmp_path, field, value):
    from autoforecast.study import Study
    identity = {**implementation(), field: value}
    with pytest.raises(ValueError, match="implementation"):
        Study.tune(object(), fit_predict=predict, config=tmp_path / "absent", implementation=identity)


def test_external_profile_does_not_relax_builtin_registry_and_validates_domains(tmp_path):
    record = yaml.safe_load(config(tmp_path).read_text()); record.pop("hpo")
    raw = yaml.safe_dump(record).encode()
    assert loads_external_tuning_spec(raw).adapter == "external"
    with pytest.raises(TuningError, match="not HPO-registered"):
        loads_tuning_spec(raw)
    record["model"]["adapter"] = "misspelled-model"
    with pytest.raises(TuningError, match="adapter: external"):
        loads_external_tuning_spec(yaml.safe_dump(record).encode())
    record["model"]["adapter"] = "external"
    record["hyperparameters"]["model.level"]["search"]["values"] = [100.0]
    with pytest.raises(TuningError):
        loads_external_tuning_spec(yaml.safe_dump(record).encode())


def test_explicit_window_and_fresh_run_identity(spark, tmp_path):
    manifest = freeze(tmp_path / "study", spark, windows="multiple")
    profile = config(tmp_path, budget=1)
    with open_study(manifest, spark) as study:
        first = study.tune(fit_predict=predict, config=profile, implementation=implementation(), tuning_origin="2024-01-04")
        second = study.tune(fit_predict=predict, config=profile, implementation=implementation(), tuning_origin="2024-01-04")
    assert first.path != second.path
    assert first.forecasts.origin == "2024-01-04" and first.forecasts.horizon == 1
    assert first.tuning_score == abs(first.parameters["model.level"] - 5)
    assert first.parameters == second.parameters


def test_external_constraints_and_parameter_names_are_enforced(tmp_path):
    record = yaml.safe_load(config(tmp_path).read_text()); record.pop("hpo")
    record["hyperparameters"]["model.level"] = {"default": 0,
        "valid": {"kind": "int", "low": 0, "high": 10, "step": 1},
        "search": {"kind": "categorical", "values": [0, 10]}}
    record["constraints"] = [{"kind": "less_equal", "dividend": "model.level", "divisor": "training.offset"}]
    with pytest.raises(TuningError, match="constraint|less than"):
        loads_external_tuning_spec(yaml.safe_dump(record).encode())
    record.pop("constraints")
    record["hyperparameters"]["unqualified"] = record["hyperparameters"].pop("model.level")
    with pytest.raises(TuningError, match="parameter name"):
        loads_external_tuning_spec(yaml.safe_dump(record).encode())


@pytest.mark.parametrize("value", [False, 1, [], "", " ", "\t", "\r\n", "\u00a0"])
def test_invalid_origin_rejected_before_study_access(tmp_path, value):
    from autoforecast.study import Study
    with pytest.raises(ValueError, match="tuning_origin"):
        Study.tune(object(), fit_predict=predict, config=tmp_path / "absent", implementation=implementation(), tuning_origin=value)


def test_noncallable_and_closed_study_reject_before_fitting(frozen, spark, tmp_path):
    from autoforecast.study import Study
    with pytest.raises(TypeError, match="callable"):
        Study.tune(object(), fit_predict=None, config=tmp_path / "absent", implementation=implementation())
    study = open_study(frozen, spark)
    study.close()
    with pytest.raises(ValueError, match="closed"):
        study.tune(fit_predict=predict, config=config(tmp_path), implementation=implementation())


@pytest.mark.parametrize("metric,missing,zero,error", [("wape", False, True, "finite tuning objective"), ("wrmsse", "all", False, "every retained series")])
def test_unavailable_objectives_never_select_a_trial(spark, tmp_path, metric, missing, zero, error):
    manifest = freeze(tmp_path / "study", spark, metric=metric, missing=missing, zero_targets=zero)
    with open_study(manifest, spark) as study, pytest.raises((HPOError, ScoreError), match=error):
        study.tune(fit_predict=predict, config=config(tmp_path, budget=1), implementation=implementation())
    assert not list((tmp_path / "study/.autoforecast/study-tuning").rglob("selection.json"))


@pytest.mark.parametrize("defect", ["no-hpo", "hpo-list", "hpo-extra", "root-extra", "implementation", "symlink"])
def test_external_configuration_is_explicit_and_inert(tmp_path, defect):
    from autoforecast.study import Study
    profile = config(tmp_path)
    record = yaml.safe_load(profile.read_text())
    if defect == "no-hpo":
        record.pop("hpo")
    elif defect == "hpo-list":
        record["hpo"] = []
    elif defect == "hpo-extra":
        record["hpo"]["evaluator"] = "some_module:execute"
    elif defect == "root-extra":
        record["dates"] = ["2024-01-01"]
    elif defect == "implementation":
        record["implementation"] = {"module": "some_module:execute"}
    profile.write_text(yaml.safe_dump(record))
    if defect == "symlink":
        link = tmp_path / "linked.yaml"
        link.symlink_to(profile)
        profile = link
    with pytest.raises(TuningError):
        Study.tune(object(), fit_predict=predict, config=profile, implementation=implementation())


@pytest.mark.parametrize("defect", ["missing", "extra"])
def test_implementation_key_set_is_exact(tmp_path, defect):
    from autoforecast.study import Study
    identity = implementation()
    if defect == "missing":
        identity.pop("dependencies")
    else:
        identity["module"] = "some_module:run"
    with pytest.raises(ValueError, match="requires"):
        Study.tune(object(), fit_predict=predict, config=tmp_path / "absent", implementation=identity)


def test_profile_bytes_are_frozen_before_callback(frozen, spark, tmp_path):
    profile = config(tmp_path, budget=1)
    original = profile.read_bytes()
    def fit(request):
        profile.write_bytes(b"changed after capture")
        return predict(request)
    with open_study(frozen, spark) as study:
        result = study.tune(fit_predict=fit, config=profile, implementation=implementation())
    assert (result.model_artifacts.parent / "metadata/external-profile.yaml").read_bytes() == original
    assert result.provenance["profile_sha256"] == hashlib.sha256(original).hexdigest()


def test_tuning_input_snapshots_are_cleaned_when_preparation_fails(frozen, spark, tmp_path, monkeypatch):
    from pyspark.sql.readwriter import DataFrameWriter

    created = []
    def fail_write(self, path, *args, **kwargs):
        directory = pathlib.Path(path)
        directory.mkdir(parents=True)
        (directory / "partial-write").write_bytes(b"partial")
        created.append(directory)
        raise RuntimeError("training preparation failed")
    monkeypatch.setattr(DataFrameWriter, "parquet", fail_write)
    with open_study(frozen, spark) as study, pytest.raises(RuntimeError, match="preparation failed"):
        study.tune(fit_predict=predict, config=config(tmp_path, budget=1), implementation=implementation())
    assert created and all(not path.parent.exists() for path in created)
    assert spark.range(1).count() == 1


def test_tuning_root_symlink_rejected_before_writing_inputs(spark, tmp_path, monkeypatch):
    from pyspark.sql.readwriter import DataFrameWriter
    manifest = freeze(tmp_path / "study", spark)
    target = tmp_path / "redirected"
    target.mkdir()
    (tmp_path / "study/.autoforecast/study-tuning").symlink_to(target, target_is_directory=True)
    writes = []
    original = DataFrameWriter.parquet
    def observe_write(self, path, *args, **kwargs):
        writes.append(path)
        return original(self, path, *args, **kwargs)
    monkeypatch.setattr(DataFrameWriter, "parquet", observe_write)
    with open_study(manifest, spark) as study:
        with pytest.raises((HPOError, LineageError), match="symlink"):
            study.tune(fit_predict=predict, config=config(tmp_path), implementation=implementation())
    assert writes == []
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("defect", ["duplicate", "nonfinite", "date", "key"])
def test_malformed_external_profile_raises_tuning_error_before_study_access(tmp_path, defect):
    from autoforecast.study import Study
    profile = config(tmp_path)
    if defect == "duplicate":
        profile.write_text(profile.read_text() + "hpo: {}\n")
    else:
        record = yaml.safe_load(profile.read_text())
        if defect == "nonfinite":
            record["hyperparameters"]["model.level"]["default"] = float("nan")
        elif defect == "date":
            record["model"]["id"] = dt.date(2024, 1, 1)
        else:
            record[7] = "unexpected key"
        profile.write_text(yaml.safe_dump(record))
    with pytest.raises(TuningError):
        Study.tune(object(), fit_predict=predict, config=profile, implementation=implementation())
