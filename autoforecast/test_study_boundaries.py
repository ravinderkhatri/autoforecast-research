"""Failure boundaries for frozen study policies, references, and score evidence."""

import copy
import datetime as dt
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark
from autoforecast.data_admission import DatasetContractError, load_dataset_contract
from autoforecast.forecast_campaign import ForecastCampaignError, _test_score_value
from autoforecast.panel import Forecast, Panel
from autoforecast.score_spark import ScoreError, score_forecast
from autoforecast.test_study import freeze, inputs
from autoforecast.test_study import spark as _spark_fixture

spark = _spark_fixture


@pytest.fixture(scope="module")
def study(tmp_path_factory, spark):
    return freeze(tmp_path_factory.mktemp("study-boundaries"), spark)


@pytest.mark.parametrize("field", [
    "study_id", "role_paths", "metric_definition", "coverage_identity", "coverage_count",
])
def test_study_manifest_rejects_changed_claims(study, field):
    original = study.read_bytes()
    record = json.loads(original)
    if field == "study_id":
        record[field] = "0" * 64
    elif field == "role_paths":
        record[field]["selection"] = "final_score"
    elif field == "metric_definition":
        record[field]["actual_missing"] = "error"
    else:
        evidence = next(iter(record["coverage_identities"].values()))
        if field == "coverage_identity":
            evidence["content_sha256"] = "0" * 64
        else:
            evidence["counts"]["scored"] += 1
    try:
        study.write_text(json.dumps(record))
        with pytest.raises(benchmark.BenchmarkError):
            benchmark.verify_benchmark(study)
    finally:
        study.write_bytes(original)


@pytest.mark.parametrize("count", [None, True, 0, -1, 2.0, "2"])
def test_v2_dataset_requires_an_exact_positive_cohort_count(tmp_path, count):
    contract, _, _ = inputs(tmp_path)
    value = yaml.safe_load(contract.read_text())
    value["schema_version"] = 2
    if count is not None:
        value["dataset"]["required_series_count"] = count
    contract.write_text(yaml.safe_dump(value))
    with pytest.raises(DatasetContractError):
        load_dataset_contract(contract)


@pytest.mark.parametrize("case", [
    "columns", "key_type", "weight_type", "null_key", "null_weight", "nan_weight",
    "infinite_scale", "negative_scale", "negative_weight", "duplicate", "missing",
    "extra", "zero_total",
])
def test_scoring_references_refuse_invalid_values_and_populations(tmp_path, spark, case):
    _, _, recipe = inputs(tmp_path)
    rows = [
        {"keys": {"id": series}, "stage": stage, "origin": origin, "weight": 1., "scale": 2.}
        for series in ("A", "B")
        for stage, origin in (("validation", "2024-01-05"), ("test", "2024-01-07"))
    ]
    schema = pa.schema([
        ("keys", pa.struct([("id", pa.string())])), ("stage", pa.string()),
        ("origin", pa.string()), ("weight", pa.float64()), ("scale", pa.float64()),
    ])
    if case == "null_key":
        rows[0]["keys"]["id"] = None
    elif case == "null_weight":
        rows[0]["weight"] = None
    elif case == "nan_weight":
        rows[0]["weight"] = float("nan")
    elif case == "infinite_scale":
        rows[0]["scale"] = float("inf")
    elif case in {"negative_scale", "negative_weight"}:
        # Keep the boundary total positive so the negative-weight check must catch it.
        rows[0][case.removeprefix("negative_")] = -0.5
    elif case == "duplicate":
        rows.append(rows[0].copy())
    elif case == "missing":
        rows.pop()
    elif case == "extra":
        rows.append({**rows[0], "keys": {"id": "late"}})
    elif case == "zero_total":
        for row in rows:
            row["weight"] = 0.
    table = pa.Table.from_pylist(rows, schema=schema)
    if case == "columns":
        table = table.drop(["scale"])
    elif case == "key_type":
        table = table.set_column(0, "keys", pa.array([{"id": 1}] * len(rows)))
    elif case == "weight_type":
        table = table.set_column(3, "weight", pa.array(["1"] * len(rows)))
    path = tmp_path / "references.parquet"
    pq.write_table(table, path)
    key_map = spark.createDataFrame([("A", '{"id":"A"}'), ("B", '{"id":"B"}')],
                                    "id string, __af_series_id string")
    with pytest.raises(benchmark.BenchmarkError):
        benchmark._prepare_scoring_references(path, key_map, ["id"], recipe, spark)


@pytest.mark.parametrize("case", ["absent", "type", "null", "nonzero", "no_grid", "no_output"])
def test_original_missing_flags_are_validated_before_scoring(tmp_path, spark, case):
    day = dt.date(2024, 1, 2)
    flag = "missing"
    fields = [("id", pa.string()), ("date", pa.date32()), ("y", pa.float64())]
    row = {"id": "A", "date": day, "y": 0.}
    if case != "absent":
        fields.append((flag, pa.int64() if case == "type" else pa.bool_()))
        row[flag] = 1 if case == "type" else None if case == "null" else True
    if case == "nonzero":
        row["y"] = 7.
    source = tmp_path / "actuals.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=pa.schema(fields)), source)
    prediction = tmp_path / "predictions.parquet"
    pq.write_table(pa.table({"series_id": ["A"], "date": [day], "yhat": [2.]}), prediction)
    grid = tmp_path / "grid.parquet"
    pq.write_table(pa.table({"series_id": ["A"], "date": [day]}), grid)
    panel = Panel(str(source), "id", "date", "y", target_estimand="observed_nonnegative")
    forecast = Forecast("test", str(prediction), "2024-01-01", 1, 1, {"emitted": 1},
                        target_estimand="observed_nonnegative")
    with pytest.raises(ScoreError):
        score_forecast(
            panel, forecast, spark, missing_actual="no_data", original_missing_column=flag,
            eligible_grid=None if case == "no_grid" else str(grid),
            series_scores_path=None if case == "no_output" else str(tmp_path / "scores"),
        )


@pytest.mark.parametrize("case", ["model", "score", "missing_metric", "metric", "ranking", "contract", "multiple"])
def test_unavailable_test_score_requires_matching_evidence(case):
    fixed = {"schema": "autoforecast.fixed-evaluation.v2", "metric_policy": {"metric": "rmse"}}
    comparison = {
        "evaluation_contract": copy.deepcopy(fixed), "ranking": None,
        "candidates": [{"model_id": "model", "status": "unavailable",
                        "score": {"rmse": None, "metric_definition": fixed["metric_policy"]}}],
    }
    assert _test_score_value(comparison, "rmse", "model", fixed) is None
    candidate = comparison["candidates"][0]
    if case == "model":
        candidate["model_id"] = "other"
    elif case == "score":
        candidate["score"]["rmse"] = 0.
    elif case == "missing_metric":
        candidate["score"].pop("rmse")
    elif case == "metric":
        candidate["score"]["metric_definition"] = {"metric": "mae"}
    elif case == "ranking":
        comparison["ranking"] = {"order": ["model"]}
    elif case == "contract":
        comparison["evaluation_contract"]["schema"] = "wrong"
    else:
        comparison["candidates"].append(copy.deepcopy(candidate))
    with pytest.raises(ForecastCampaignError):
        _test_score_value(comparison, "rmse", "model", fixed)


@pytest.fixture(scope="module")
def result_evidence(study, spark, tmp_path_factory):
    from autoforecast.run_evaluation import compare_run_forecasts
    from autoforecast.test_study_integrity import _validation_forecast

    root = tmp_path_factory.mktemp("study-result-evidence")
    fixed, forecast = _validation_forecast(study, root / "forecast.parquet")
    panel = load_dataset_contract(study.parent / "dataset.yaml").panel(fixed["actuals"]["path"])
    comparison = compare_run_forecasts(
        panel, {"model": forecast}, spark, missing_actual="no_data", metric="rmse",
        output_dir=root / "evaluation", fixed_contract=fixed,
    )
    return forecast, json.loads(comparison.read_text())


def _study_reader(tmp_path, monkeypatch, study, result_evidence):
    import hashlib

    from autoforecast.test_results import _reader_boundary
    from provenance.lineage import directory_digest

    forecast, comparison = result_evidence
    child = {"id": "validation", "kind": "validation", "model_id": "model",
             "variant_id": "default", "seed": 0, "initialization": "fresh",
             "origin": forecast.origin, "horizon": forecast.horizon}
    record = json.loads(study.read_bytes())
    _reader_boundary(tmp_path, monkeypatch, [child],
                     [{"id": "validation", "status": "completed", "attempts": 1}],
                     {"path": str(study), "sha256": hashlib.sha256(study.read_bytes()).hexdigest(),
                      "id": record["study_id"], "evaluations": {}})
    stage = tmp_path / "validation"
    evaluation = stage / "run/evaluation"
    evaluation.mkdir(parents=True)
    comparison_path = evaluation / "candidate-comparison.json"
    comparison_path.write_text(json.dumps(comparison))
    forecast_record = stage / "forecast.json"
    forecast_record.write_text(json.dumps(forecast.as_record()))
    result = {
        **child, "adapter": forecast.adapter, "run_path": str(stage / "run"),
        "forecast_record_path": str(forecast_record),
        "forecast_record_sha256": hashlib.sha256(forecast_record.read_bytes()).hexdigest(),
        "forecast_artifact_sha256": directory_digest(forecast.path, suffixes=(".parquet",)),
        "comparison_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
    }
    result_path = stage / "child-result.json"
    result_path.write_text(json.dumps(result))
    return comparison_path, result_path


@pytest.mark.parametrize("case", ["absent", "tampered", "model", "boundary"])
def test_result_reader_refuses_missing_or_mislabeled_series_evidence(
    tmp_path, monkeypatch, study, result_evidence, case,
):
    import hashlib

    from autoforecast.results import read_results

    comparison_path, result_path = _study_reader(tmp_path, monkeypatch, study, result_evidence)
    assert read_results(tmp_path).series_scores.num_rows == 2
    comparison = json.loads(comparison_path.read_bytes())
    candidate = comparison["candidates"][0]
    if case == "absent":
        candidate["score"].pop("series_scores")
    elif case == "tampered":
        candidate["score"]["series_scores"]["sha256"] = "0" * 64
    elif case == "model":
        candidate["model_id"] = "other"
    else:
        comparison["evaluation_contract"]["boundary"] = "test/2024-01-07/2/D"
    comparison_path.write_text(json.dumps(comparison))
    result = json.loads(result_path.read_bytes())
    result["comparison_sha256"] = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="study"):
        read_results(tmp_path)


@pytest.mark.parametrize("role", ["source-keys", "forecast", "series_scores"])
def test_result_reader_consumes_verified_copies(tmp_path, monkeypatch, study, result_evidence, role):
    import pathlib

    import autoforecast.results as reader

    _study_reader(tmp_path, monkeypatch, study, result_evidence)
    expected = reader.read_results(tmp_path)
    forecast, comparison = result_evidence
    record = json.loads(study.read_bytes())
    source = {
        "source-keys": benchmark.artifact_path(study, record, "source-keys"),
        "forecast": pathlib.Path(forecast.path),
        "series_scores": pathlib.Path(comparison["candidates"][0]["score"]["series_scores"]["path"]),
    }[role]
    part = source if source.is_file() else next(source.glob("*.parquet"))
    raw = part.read_bytes()
    table = pq.read_table(part)
    column = {"source-keys": "id", "forecast": "yhat", "series_scores": "rmse"}[role]
    index = table.schema.get_field_index(column)
    replacement = "wrong" if role == "source-keys" else 999.
    changed = table.set_column(index, table.schema.field(index),
                               pa.array([replacement] * table.num_rows, type=table.schema.field(index).type))
    read = reader.pq.read_table
    snapshot = reader.snapshot_parquet_directory
    copies = []
    def remember_copy(path, *args, **kwargs):
        result = snapshot(path, *args, **kwargs)
        if pathlib.Path(path).resolve() == source.resolve():
            copies.append(result.resolve())
        return result
    injected = 0
    def changing_read(path, *args, **kwargs):
        nonlocal injected
        if copies and pathlib.Path(path).resolve() in [*copies, source.resolve()]:
            injected += 1
            try:
                pq.write_table(changed, part)
                return read(path, *args, **kwargs)
            finally:
                part.write_bytes(raw)
        return read(path, *args, **kwargs)
    monkeypatch.setattr(reader, "snapshot_parquet_directory", remember_copy)
    monkeypatch.setattr(reader.pq, "read_table", changing_read)
    observed = reader.read_results(tmp_path)
    assert injected == 1
    assert observed.rows.equals(expected.rows)
    assert observed.source_keys.equals(expected.source_keys)
    assert observed.series_scores.equals(expected.series_scores)


def test_direct_study_scorer_validates_unchecked_forecast_metadata(
    tmp_path, study, result_evidence, spark,
):
    import dataclasses

    from autoforecast.study_scoring import score_study

    forecast, comparison = result_evidence
    # A zero-valued boolean evades sum-only population checks but is not a valid count.
    invalid = dataclasses.replace(forecast, disposition={"emitted": 4, "declined": False})
    fixed = comparison["evaluation_contract"]
    panel = load_dataset_contract(study.parent / "dataset.yaml").panel(fixed["actuals"]["path"])
    with pytest.raises(ScoreError):
        score_study(panel, invalid, spark, contract=fixed, output_dir=tmp_path / "scores")


def test_result_reader_parses_the_comparison_bytes_it_authenticates(
    tmp_path, monkeypatch, study, result_evidence,
):
    import pathlib

    from autoforecast.results import read_results

    comparison_path, _ = _study_reader(tmp_path, monkeypatch, study, result_evidence)
    original = comparison_path.read_bytes()
    changed = json.loads(original)
    changed["candidates"][0]["model_id"] = "other"
    changed_bytes = json.dumps(changed).encode()
    read_bytes = pathlib.Path.read_bytes
    reads = 0
    def changing_read(path):
        nonlocal reads
        if path.resolve() == comparison_path.resolve():
            reads += 1
            return original if reads == 1 else changed_bytes
        return read_bytes(path)
    monkeypatch.setattr(pathlib.Path, "read_bytes", changing_read)
    results = read_results(tmp_path)
    assert reads == 1
    assert results.series_scores["model_id"].to_pylist() == ["model", "model"]


@pytest.mark.parametrize("fault", [
    None, "output", "stage_result", "unlisted", "directory", "attempt", "status",
])
def test_completed_json_is_bound_to_the_journal_bytes(tmp_path, monkeypatch, fault):
    """A completion check cannot authorize different scalar bytes in a later read."""
    import hashlib
    from types import SimpleNamespace

    from autoforecast import forecast_campaign as campaign

    attempt = tmp_path / "stages/validation/attempt-0001"
    attempt.mkdir(parents=True)
    output = attempt / "child-result.json"
    output.write_text(json.dumps({"score": 1.0}))
    output_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    artifact = {"path": output.name, "kind": "file", "sha256": output_sha}
    if fault == "unlisted":
        artifact["path"] = "other.json"
    if fault == "directory":
        artifact["kind"] = "directory"
    result = attempt / "stage-result.json"
    result.write_text(json.dumps({"artifacts": [artifact]}))
    state = {"id": "validation", "status": "completed", "attempts": 1,
             "last_result_sha256": hashlib.sha256(result.read_bytes()).hexdigest()}
    if fault == "attempt":
        state["attempts"] = 2
    if fault == "status":
        state["status"] = "running"
    store = SimpleNamespace(state=lambda: {"stages": [state]})
    monkeypatch.setattr(campaign.CampaignStore, "from_root", lambda _: store)
    real_read = type(output).read_bytes

    def transient_read(path):
        raw = real_read(path)
        if (fault == "output" and path == output) or (fault == "stage_result" and path == result):
            return raw + b"\n"
        return raw

    monkeypatch.setattr(type(output), "read_bytes", transient_read)
    if fault is None:
        value, observed = campaign._load_completed_json(output)
        assert value == {"score": 1.0} and observed == output_sha
    else:
        with pytest.raises(campaign.ForecastCampaignError):
            campaign._load_completed_json(output)


@pytest.mark.parametrize("filename", ["child-result.json", "validation-summary.json"])
def test_selection_records_captured_identities(tmp_path, monkeypatch, filename):
    """Selection carries journal identities and written bytes through to its lock."""
    import hashlib

    from autoforecast import forecast_campaign as subject
    from autoforecast.campaign import CampaignSupervisor
    from autoforecast.test_campaign import _stage, _store

    child = {
        "kind": "validation", "id": "validation", "child_id": "validation",
        "candidate_id": "baseline", "model_id": "last_value", "variant_id": "default",
        "adapter": "last_value", "initialization": "fresh", "seed": 0, "origin": "2024-01-05",
        "parameter_mode": "defaults", "parameter_resolution": {},
        "parameter_resolution_sha256": "a" * 64, "score": 1.0,
        "runtime_seconds": 2.0, "peak_rss_bytes": 100,
        "replay": {"status": "verified_artifact_bytes"},
    }
    raw = json.dumps(child)
    code = ("from pathlib import Path; "
            f"Path(r'{{attempt_dir}}/child-result.json').write_text({raw!r})")
    store = _store(tmp_path, [_stage("validation", code, outputs=["child-result.json"])])
    assert CampaignSupervisor(store, poll_seconds=0.01).run()["status"] == "completed"
    output = subject._completed_output(store.root, "validation", "child-result.json")
    _, expected_sha = subject._load_completed_json(output)
    resolved = {
        "children": [child], "seeds": [0], "selection": {"metric": "rmse"},
        "temporal": {"validation_origins": ["2024-01-05"], "validation_horizon": 2},
        "initialization_capabilities": {}, "source": {}, "data": {"selection": {}},
        "budgets": {"final_fit": 1},
    }
    monkeypatch.setattr(subject, "_load_resolved", lambda *_: resolved)
    # Isolate model/checkpoint validation; exercise the real completed-stage journal boundary.
    monkeypatch.setattr(subject, "_validated_child_result",
                        lambda _resolved, _child, path, **_: subject._load_completed_json(path))
    real_hash = subject._sha_file
    rereads = []

    def changed_hash(path):
        if path.name == filename:
            rereads.append(path)
            return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(subject, "_sha_file", changed_hash)
    attempt = tmp_path / "selection"
    attempt.mkdir()
    subject.aggregate_selection(tmp_path / "resolved.json", "b" * 64,
                                campaign_root=store.root, attempt_dir=attempt)
    summary_bytes = (attempt / "validation-summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    lock = json.loads((attempt / "selection-lock.json").read_bytes())
    assert summary["candidates"][0]["child_result_sha256"] == [expected_sha]
    assert lock["validation_summary_sha256"] == hashlib.sha256(summary_bytes).hexdigest()
    assert lock["chosen"]["mean_score"] == 1.0
    assert not rereads
