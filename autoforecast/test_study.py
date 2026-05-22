"""Shared studies retain their population, missingness, and immutable identities."""

import copy
import datetime as dt
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pyspark.sql import SparkSession

from autoforecast import benchmark
from autoforecast.data_admission import load_dataset_contract


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]").appName("test_shared_study")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false").getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def inputs(root, *, actual_missing="zero"):
    """Two retained series; one late arrival; four cells in each scoring window."""
    root.mkdir(parents=True, exist_ok=True)
    contract = root / "dataset.yaml"
    contract.write_text(yaml.safe_dump({
        "schema_version": 1,
        "dataset": {"id": "synthetic-study", "panel": {
            "series_col": "id", "time_col": "date", "target_col": "y",
            "frequency": "D", "season_length": 2, "target_estimand": "observed_nonnegative",
        }},
    }))
    rows = []
    for series, first in (("A", 1), ("B", 3)):
        for day in range(first, 10):
            if series == "B" and day in (5, 7, 9):
                continue
            value = float(day)
            if (series, day) in (("A", 4), ("A", 6), ("B", 8)):
                value = None
            if (series, day) == ("A", 8):
                value = 0.0
            rows.append({"id": series, "date": dt.date(2024, 1, day), "y": value})
    rows.append({"id": "late", "date": dt.date(2024, 1, 8), "y": 1.0})
    source = root / "observations.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([
        ("id", pa.string()), ("date", pa.date32()), ("y", pa.float64()),
    ])), source)
    recipe = {
        "schema_version": 2, "source_keys": ["id"], "training_start": "2024-01-01",
        "validation_origins": ["2024-01-05"], "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07", "final_test_horizon": 2,
        "history_policy": "fixed", "new_series_policy": "exclude_after_first_origin",
        "target_unit": "units", "feature_availability": {}, "metric": "rmse",
        "actual_missing": actual_missing,
        "series_starts": {"rule": "declared usable history at the first origin", "values": [
            {"keys": {"id": "A"}, "start": "2024-01-01"},
            {"keys": {"id": "B"}, "start": "2024-01-03"},
        ]},
    }
    sources = dict.fromkeys(("selection", "final_training", "validation_score", "final_score"),
                            source)
    return contract, sources, recipe


def freeze(root, spark, *, actual_missing="zero", mutate=None, study_id=None):
    contract, sources, recipe = inputs(root, actual_missing=actual_missing)
    if mutate:
        mutate(recipe)
    return benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=root / "frozen", spark=spark, study_id=study_id,
    )


@pytest.mark.parametrize("policy,scored", [("zero", 4), ("mask", 2)])
def test_study_freezes_unequal_starts_original_missingness_and_complete_grids(
    tmp_path, spark, policy, scored,
):
    path = freeze(tmp_path, spark, actual_missing=policy)
    _, record = benchmark.verify_benchmark(path)
    assert record["schema"] == benchmark.STUDY_SCHEMA
    assert record["study_id"] == record["benchmark_id"]
    assert record["source_series_count"] == 3
    assert record["excluded_new_series"] == 1
    assert record["cohort_series_count"] == 2
    assert load_dataset_contract(path.parent / "dataset.yaml").required_series_count == 2
    assert record["recipe"]["history_missing"] == "zero"
    assert record["counts"]["selection"]["required_observations"] == 8
    assert record["counts"]["selection"]["originally_missing"] == 2
    selection = pq.read_table(benchmark.artifact_path(path, record, "selection")).to_pylist()
    assert len(selection) == 8
    assert min(row["date"] for row in selection if row["id"] == "B") == dt.date(2024, 1, 3)
    assert all(row["y"] == 0 for row in selection if row[benchmark.MISSING_COLUMN])
    assert benchmark.artifact_path(path, record, "validation_score") == (
        benchmark.artifact_path(path, record, "final_training")
    )
    for boundary, grid in record["grids"].items():
        assert grid["n_cells"] == 4
        assert record["coverage_identities"][boundary]["counts"] == {
            "expected": 4, "originally_missing": 2, "scored": scored,
            "zero_filled": 2 if policy == "zero" else 0,
            "excluded": 2 if policy == "mask" else 0,
        }
    actuals = pq.read_table(benchmark.artifact_path(path, record, "final_score")).to_pylist()
    zero = next(row for row in actuals if row["id"] == "A" and row["date"].day == 8)
    assert zero["y"] == 0 and zero[benchmark.MISSING_COLUMN] is False
    assert benchmark.resolve_study(path.parent.parent, record["study_id"])[0] == path


def test_study_uniform_default_does_not_infer_start_from_first_record(tmp_path, spark):
    path = freeze(tmp_path, spark, mutate=lambda recipe: recipe.pop("series_starts"))
    _, record = benchmark.verify_benchmark(path)
    rows = pq.read_table(benchmark.artifact_path(path, record, "selection")).to_pylist()
    leading = [row for row in rows if row["id"] == "B" and row["date"].day < 3]
    assert len(leading) == 2
    assert all(row["y"] == 0 and row[benchmark.MISSING_COLUMN] for row in leading)
    assert record["recipe"]["series_starts"]["rule"] == "study_lower_bound"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_study_invalid_values_are_not_missing(tmp_path, spark, value):
    contract, sources, recipe = inputs(tmp_path)
    source = sources["selection"]
    rows = pq.read_table(source).to_pylist()
    rows[0]["y"] = value
    pq.write_table(pa.Table.from_pylist(rows), source)
    with pytest.raises(benchmark.BenchmarkError, match="non-finite observed target"):
        benchmark.freeze_benchmark(
            contract_path=contract, sources=sources, recipe=recipe,
            output_root=tmp_path / "frozen", spark=spark,
        )


def test_study_registration_reuses_bytes_and_rejects_changed_rules(tmp_path, spark):
    path = freeze(tmp_path, spark)
    original = path.read_bytes()
    record = json.loads(original)
    contract, sources, recipe = inputs(tmp_path)
    assert benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=path.parent.parent, spark=spark, study_id=record["study_id"],
    ) == path
    assert path.read_bytes() == original
    changed = copy.deepcopy(recipe)
    changed["actual_missing"] = "mask"
    with pytest.raises(benchmark.BenchmarkError, match="different settings or data"):
        benchmark.freeze_benchmark(
            contract_path=contract, sources=sources, recipe=changed,
            output_root=path.parent.parent, spark=spark, study_id=record["study_id"],
        )
    assert path.read_bytes() == original
    new_path = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=changed,
        output_root=path.parent.parent, spark=spark,
    )
    assert new_path != path


@pytest.mark.parametrize("policy,scored", [("zero", 4), ("mask", 2)])
def test_study_freezes_inner_tuning_population_and_missingness(tmp_path, spark, policy, scored):
    # Population: the same A/B cohort as outer validation; four inner target cells.
    # A on day 4 is null and B on day 5 is absent; the late series remains excluded.
    path = freeze(
        tmp_path, spark, actual_missing=policy,
        mutate=lambda recipe: recipe.update(tuning_windows=[{"origin": "2024-01-03", "horizon": 2}]),
    )
    _, record = benchmark.verify_benchmark(path)
    fixed = benchmark.evaluation_contract(
        path, stage="tuning", origin="2024-01-03", horizon=2, metric="rmse",
    )
    assert fixed["actuals"]["role"] == "selection"
    assert fixed["actuals"]["sha256"] == record["artifacts"]["selection"]
    assert fixed["grid"]["n_cells"] == 4
    assert fixed["coverage"]["counts"] == {
        "expected": 4, "originally_missing": 2, "scored": scored,
        "zero_filled": 2 if policy == "zero" else 0,
        "excluded": 2 if policy == "mask" else 0,
    }
    with pytest.raises(benchmark.BenchmarkError, match="boundary absent"):
        benchmark.evaluation_contract(
            path, stage="tuning", origin="2024-01-02", horizon=2, metric="rmse",
        )
    changed = copy.deepcopy(record)
    changed["recipe"]["tuning_windows"][0]["horizon"] = 1
    with pytest.raises(benchmark.BenchmarkError, match="grid inventory"):
        benchmark._validate_manifest(changed)


@pytest.mark.parametrize("windows", [
    None, [], {}, [True], [{"origin": "2024-01-03", "horizon": True}],
    [{"origin": "2024-01-03", "horizon": 0}],
    [{"origin": "2024-01-03", "horizon": 3}],
    [{"origin": "2023-12-31", "horizon": 2}],
    [{"origin": "2024-01-02", "horizon": 2}],  # B has not reached its declared start.
    [{"origin": "2024-01-03", "horizon": 1}, {"origin": "2024-01-03", "horizon": 2}],
    [{"origin": "2024-01-04", "horizon": 1}, {"origin": "2024-01-03", "horizon": 2}],
    [{"origin": "2024-01-03", "horizon": 2, "metric": "mae"}],
])
def test_study_rejects_invalid_inner_windows_before_runtime(tmp_path, windows):
    contract, sources, recipe = inputs(tmp_path)
    recipe["tuning_windows"] = windows
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.freeze_benchmark(
            contract_path=contract, sources=sources, recipe=recipe,
            output_root=tmp_path / "frozen", spark=object(),
        )


def test_obsolete_recipe_is_rejected_before_runtime(tmp_path):
    contract, sources, recipe = inputs(tmp_path)
    recipe["schema_version"] = 1
    for key in ("metric", "actual_missing", "series_starts"):
        recipe.pop(key)
    sources.pop("validation_score")
    with pytest.raises(benchmark.BenchmarkError, match="schema_version must be 2"):
        benchmark.freeze_benchmark(
            contract_path=contract, sources=sources, recipe=recipe,
            output_root=tmp_path / "frozen", spark=object(),
        )


def score_fixture(path, spark, output, *, stage="validation", remove=False):
    from autoforecast.panel import Forecast
    from autoforecast.run_evaluation import compare_run_forecasts

    _, record = benchmark.verify_benchmark(path)
    origin = "2024-01-05" if stage == "validation" else "2024-01-07"
    contract = benchmark.evaluation_contract(
        path, stage=stage, origin=origin, horizon=2, metric=record["recipe"]["metric"],
    )
    grid = pq.read_table(contract["grid"]["path"]).to_pylist()
    rows = [{**row, "yhat": 2.0} for row in grid]
    if remove:
        rows.pop()
    forecast_path = output / "forecast"
    forecast_path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), forecast_path / "part.parquet")
    forecast = Forecast("test", str(forecast_path), origin, 2, 2,
                        {"emitted": len(rows), "declined": 4 - len(rows)},
                        target_estimand="observed_nonnegative")
    panel = load_dataset_contract(path.parent / "dataset.yaml").panel(contract["actuals"]["path"])
    comparison = compare_run_forecasts(
        panel, {"test": forecast}, spark, missing_actual="no_data",
        metric=record["recipe"]["metric"], output_dir=output / "comparison", fixed_contract=contract,
    )
    return json.loads(comparison.read_text())


@pytest.mark.parametrize("policy", ["zero", "mask"])
def test_study_scoring_uses_original_missingness_and_reports_every_series(tmp_path, spark, policy):
    import math

    path = freeze(tmp_path, spark, actual_missing=policy)
    # Validation actuals: A=(missing,7), B=(6,missing); predictions all 2.
    # Test actuals: A=(recorded 0,9), B=(missing,missing).
    for stage, errors in (("validation", [4, 25, 16, 4]), ("test", [4, 49, 4, 4])):
        result = score_fixture(path, spark, tmp_path / stage, stage=stage)
        score = result["candidates"][0]["score"]
        assert result["missing_actual"] == ("sparse_zero" if policy == "zero" else "no_data")
        expected_errors = errors if policy == "zero" else (
            [25, 16] if stage == "validation" else [4, 49]
        )
        assert score["rmse"] == pytest.approx(math.sqrt(sum(expected_errors) / len(expected_errors)))
        assert result["ranking"]["n_common_cells"] == 4
        assert score["coverage"]["scored"] == (4 if policy == "zero" else 2)
        series = pq.read_table(score["series_scores"]["path"]).to_pylist()
        assert len(series) == 2
        if stage == "test" and policy == "mask":
            empty = next(row for row in series if row["scored"] == 0)
            assert empty["rmse"] is None and empty["mae"] is None
            assert empty["expected"] == empty["excluded"] == 2
    from autoforecast.run_evaluation import EvaluationError
    with pytest.raises(EvaluationError, match="grid"):
        score_fixture(path, spark, tmp_path / "incomplete", remove=True)


def test_validation_actuals_are_independent_of_refit_values(tmp_path, spark):
    contract, sources, recipe = inputs(tmp_path)
    validation = tmp_path / "validation.parquet"
    rows = pq.read_table(sources["selection"]).to_pylist()
    for row in rows:
        if row["id"] == "A" and row["date"].day == 6:
            row["y"] = 100.0
    pq.write_table(pa.Table.from_pylist(rows), validation)
    sources["validation_score"] = validation
    path = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=tmp_path / "frozen", spark=spark,
    )
    _, record = benchmark.verify_benchmark(path)
    assert record["role_paths"]["validation_score"] != record["role_paths"]["final_training"]
    score = score_fixture(path, spark, tmp_path / "scores")["candidates"][0]["score"]
    assert score["rmse"] == pytest.approx(((98**2 + 25 + 16 + 4) / 4) ** 0.5)
    assert score["coverage"]["originally_missing"] == 1


def scoring_references(root, *, scale_b=4.0):
    path = root / "references.parquet"
    rows = [
        {"keys": {"id": key}, "stage": stage, "origin": origin, "weight": weight, "scale": scale}
        for stage, origin in (("validation", "2024-01-05"), ("test", "2024-01-07"))
        for key, weight, scale in (("A", 14.0, 2.5), ("B", 9.0, scale_b))
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.mark.parametrize("policy,scale_b", [("zero", 4.0), ("mask", 4.0), ("mask", 0.0)])
def test_flat_wrmsse_fixed_weights_and_undefined_cases(tmp_path, spark, policy, scale_b):
    import math

    contract, sources, recipe = inputs(tmp_path, actual_missing=policy)
    recipe["metric"] = "wrmsse"
    path = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=tmp_path / "frozen", spark=spark,
        scoring_references=scoring_references(tmp_path, scale_b=scale_b),
    )
    validation = score_fixture(path, spark, tmp_path / "validation")
    score = validation["candidates"][0]["score"]
    if scale_b:
        mse_a, mse_b = (14.5, 10) if policy == "zero" else (25, 16)
        expected = (14 * math.sqrt(mse_a / 2.5) + 9 * math.sqrt(mse_b / 4)) / 23
        assert score["wrmsse"] == pytest.approx(expected)
        assert validation["candidates"][0]["status"] == "scored"
    else:
        assert score["wrmsse"] is None
        assert validation["ranking"] is None
    test = score_fixture(path, spark, tmp_path / "test", stage="test")
    if policy == "mask":
        assert test["candidates"][0]["score"]["wrmsse"] is None
        assert test["candidates"][0]["status"] == "unavailable"
        assert test["ranking"] is None
    _, frozen = benchmark.verify_benchmark(path)
    table_path = benchmark.artifact_path(path, frozen, "scoring_references")
    part = next(table_path.glob("*.parquet"))
    part.write_bytes(part.read_bytes() + b"changed scale")
    with pytest.raises(benchmark.BenchmarkError, match="artifact changed"):
        benchmark.verify_benchmark(path)


def test_all_missing_heldout_score_is_unavailable(tmp_path, spark):
    from autoforecast.forecast_campaign import _test_score_value

    contract, sources, recipe = inputs(tmp_path, actual_missing="mask")
    source = sources["selection"]
    table = pq.read_table(source)
    rows = table.to_pylist()
    for row in rows:
        if row["date"].day > 7:
            row["y"] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), source)
    path = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=tmp_path / "frozen", spark=spark,
    )
    result = score_fixture(path, spark, tmp_path / "test", stage="test")
    score = result["candidates"][0]["score"]
    assert score["n_scored"] == 0 and score["rmse"] is None
    assert result["ranking"] is None and result["candidates"][0]["status"] == "unavailable"
    series = pq.read_table(score["series_scores"]["path"]).to_pylist()
    assert len(series) == 2 and all(row["scored"] == 0 and row["rmse"] is None for row in series)
    assert _test_score_value(result, "rmse", "test", result["evaluation_contract"]) is None


def test_complete_flat_wrmsse_matches_hand_calculation(tmp_path, spark):
    import math

    from autoforecast.panel import Forecast
    from autoforecast.run_evaluation import compare_run_forecasts

    contract, sources, recipe = inputs(tmp_path)
    source = sources["selection"]
    table = pq.read_table(source)
    rows = [row for row in table.to_pylist() if row["date"].day not in (6, 7)]
    rows.extend({"id": key, "date": dt.date(2024, 1, day), "y": value}
                for key, value in (("A", 5.0), ("B", 6.0)) for day in (6, 7))
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), source)
    recipe["metric"] = "wrmsse"
    path = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=tmp_path / "frozen", spark=spark,
        scoring_references=scoring_references(tmp_path),
    )
    fixed = benchmark.evaluation_contract(
        path, stage="validation", origin="2024-01-05", horizon=2, metric="wrmsse",
    )
    rows = pq.read_table(fixed["grid"]["path"]).to_pylist()
    for row in rows:
        row["yhat"] = 4.0 if row["date"].day == 6 else 6.0
    forecast_path = tmp_path / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), forecast_path)
    forecast = Forecast("test", str(forecast_path), "2024-01-05", 2, 2,
                        {"emitted": 4}, target_estimand="observed_nonnegative")
    panel = load_dataset_contract(path.parent / "dataset.yaml").panel(fixed["actuals"]["path"])
    output = compare_run_forecasts(
        panel, {"test": forecast}, spark, missing_actual="no_data", metric="wrmsse",
        output_dir=tmp_path / "comparison", fixed_contract=fixed,
    )
    score = json.loads(output.read_text())["candidates"][0]["score"]
    rendered = output.with_suffix(".html").read_text()
    assert "Scored cells" in rendered and "Series by level" not in rendered
    assert score["wrmsse"] == pytest.approx((14 * math.sqrt(1 / 2.5) + 9 * math.sqrt(2 / 4)) / 23)


def test_study_compilation_preserves_identity_roles_and_safe_training_slices(tmp_path, spark, monkeypatch):
    import hashlib

    from mlops.provenance import validate_data_reference

    from autoforecast import forecast_campaign as campaign
    from autoforecast.cli import create_example
    from autoforecast.project import experiment_project
    from provenance.lineage import directory_digest

    project = tmp_path / "project"
    create_example(project)
    recipe = json.loads((project / "recipe.json").read_text())
    path = benchmark.freeze_benchmark(
        contract_path=project / "dataset.yaml",
        sources=dict.fromkeys(("selection", "final_training", "validation_score", "final_score"), project / "data"),
        recipe=recipe, output_root=project / "frozen", spark=spark,
    )
    _, record = benchmark.verify_benchmark(path)
    request_path = project / "campaign.yaml"
    request = yaml.safe_load(request_path.read_text())
    request["forecast_campaign"]["dataset"]["contract"] = str((path.parent / "dataset.yaml").relative_to(project))
    request_path.write_text(yaml.safe_dump(request))
    monkeypatch.setattr(campaign, "_git_source_identity", lambda *_args: {})
    with experiment_project(project):
        compiled = campaign.compile_forecast_campaign(
            request_path, benchmark_path=path, compiled_root=project / "compiled",
            **{f"{role}_data_path": benchmark.artifact_path(path, record, role)
               for role in ("selection", "final_training", "final_score")},
        )
    resolved = json.loads(compiled.resolved_path.read_text())
    assert resolved["study_id"] == record["study_id"]
    actuals = resolved["data"]["validation_score"]
    assert validate_data_reference(actuals["reference"]) == actuals["reference"]
    assert actuals["sha256"] == record["artifacts"]["validation_score"]
    for boundary, fixed in resolved["benchmark"]["evaluations"].items():
        stage, origin, horizon, _frequency = boundary.split("/")
        with experiment_project(project):
            assert campaign._fixed_evaluation(resolved, stage, origin, int(horizon)) == fixed
    source = benchmark.artifact_path(path, record, "selection")
    destination = tmp_path / "training"
    campaign._materialize_training_panel(
        source, destination, contract_path=path.parent / "dataset.yaml",
        origin=recipe["validation_origins"][0], horizon=recipe["validation_horizon"],
        training_start=recipe["training_start"],
        expected_source_sha256=directory_digest(source, suffixes=(".parquet",)),
    )
    assert benchmark.MISSING_COLUMN not in pq.read_table(destination).column_names
    assert benchmark.MISSING_COLUMN not in spark.read.parquet(str(destination)).columns
    assert max(pq.read_table(destination)["day"].to_pylist()).isoformat() == recipe["validation_origins"][0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == resolved["benchmark"]["sha256"]
    # Fixed manual parameters work; HPO-derived sources cannot claim these outer windows.
    profile = project / "models/shallow_global/model.yaml"
    profile.parent.mkdir()
    profile.write_bytes((campaign.ROOT / "configs/models/global/shallow_global/model.yaml").read_bytes())
    request["forecast_campaign"]["models"]["candidates"] = [
        {"model_id": "shallow_global", "parameters": {"mode": "defaults"}},
    ]
    for mode in ("manual", "tune", "reuse_tuned"):
        parameters = {"mode": mode}
        if mode == "manual":
            parameters["values"] = {"model.num_leaves": 31}
        else:
            parameters["study_config"] = "not-opened.yaml"
            if mode == "reuse_tuned":
                parameters["selection"] = "not-opened-selection.json"
        request["forecast_campaign"]["models"]["candidates"][0]["parameters"] = parameters
        request_path.write_text(yaml.safe_dump(request))
        with experiment_project(project):
            arguments = dict(
                benchmark_path=path, compiled_root=project / mode,
                **{f"{role}_data_path": benchmark.artifact_path(path, record, role)
                   for role in ("selection", "final_training", "final_score")},
            )
            if mode == "manual":
                manual = campaign.compile_forecast_campaign(request_path, **arguments)
                value = json.loads(manual.resolved_path.read_bytes())
                assert value["study_id"] == record["study_id"]
                assert value["parameter_profiles"][0]["mode"] == "manual"
                assert value["parameter_profiles"][0]["resolution"]["effective_parameters"]["model.num_leaves"] == 31
            else:
                with pytest.raises(campaign.ForecastCampaignError, match="inner HPO windows"):
                    campaign.compile_forecast_campaign(request_path, **arguments)
                assert not (project / mode).exists()



@pytest.mark.parametrize("change", ["start_before_bound", "start_after_origin", "missing_key", "duplicate", "unknown_policy", "empty_rule"])
def test_invalid_study_declarations_are_refused(tmp_path, change):
    _, _, recipe = inputs(tmp_path)
    values = recipe["series_starts"]["values"]
    if change == "start_before_bound":
        values[0]["start"] = "2023-12-31"
    elif change == "start_after_origin":
        values[0]["start"] = "2024-01-06"
    elif change == "missing_key":
        values[0]["keys"] = {}
    elif change == "duplicate":
        values.append(copy.deepcopy(values[0]))
    elif change == "unknown_policy":
        recipe["actual_missing"] = "guess"
    else:
        recipe["series_starts"]["rule"] = ""
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.validate_recipe(recipe)


def test_recorded_zero_and_imputed_zero_have_different_study_identities(tmp_path, spark):
    contract, sources, recipe = inputs(tmp_path)
    path = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=tmp_path / "frozen", spark=spark,
    )
    payload = path.read_bytes()
    original = json.loads(payload)
    source = sources["selection"]
    table = pq.read_table(source)
    rows = table.to_pylist()
    for row in rows:
        if row["id"] == "A" and row["date"].day == 4:
            row["y"] = 0.0
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), source)
    changed = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=tmp_path / "frozen", spark=spark,
    )
    _, record = benchmark.verify_benchmark(changed)
    assert record["study_id"] != original["study_id"]
    assert record["logical_snapshots"]["selection"] != original["logical_snapshots"]["selection"]
    assert path.read_bytes() == payload
    # ID lookup authenticates the stored data, independent of the changed mutable source.
    assert benchmark.resolve_study(tmp_path / "frozen", original["study_id"])[0] == path


def test_rescoring_pins_study_bytes_for_the_entire_operation(tmp_path, spark, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from autoforecast import rescore
    from autoforecast.panel import Forecast
    from provenance.lineage import directory_digest

    path = freeze(tmp_path / "study", spark)
    fixed = benchmark.evaluation_contract(
        path, stage="validation", origin="2024-01-05", horizon=2, metric="rmse",
    )
    predictions = tmp_path / "predictions.parquet"
    rows = pq.read_table(fixed["grid"]["path"]).to_pylist()
    pq.write_table(pa.Table.from_pylist([{**row, "yhat": 2.0} for row in rows]), predictions)
    forecast = Forecast("test", str(predictions), "2024-01-05", 2, 2,
                        {"emitted": 4}, target_estimand="observed_nonnegative")
    forecast_record = tmp_path / "forecast.json"
    forecast_record.write_text(json.dumps(forecast.as_record()))
    original = SimpleNamespace(
        metadata={"resolved_sha256": "a" * 64, "study_id": fixed["study_id"]},
        runs=({"status": "completed", "stage": "validation", "adapter": "test",
               "model_id": "test", "variant_id": "default", "run_id": "retained-run",
               "origin": forecast.origin, "horizon": forecast.horizon,
               "forecast_record_path": str(forecast_record),
               "forecast_record_sha256": hashlib.sha256(forecast_record.read_bytes()).hexdigest(),
               "forecast_artifact_sha256": directory_digest(predictions, suffixes=(".parquet",))},),
    )
    # Journal/forecast authentication has installed functional coverage. Inject a manifest
    # change after real scoring to exercise the operation-wide immutable-byte boundary.
    monkeypatch.setattr(rescore, "read_results", lambda _: original)
    compare = rescore.compare_run_forecasts

    def change_after_score(*args, **kwargs):
        result = compare(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(rescore, "compare_run_forecasts", change_after_score)
    with pytest.raises(benchmark.BenchmarkError, match="manifest changed"):
        rescore.rescore_campaign(tmp_path / "campaign", path, tmp_path / "rescore", spark)
    assert not (tmp_path / "rescore").exists()
    assert hashlib.sha256(forecast_record.read_bytes()).hexdigest() == original.runs[0]["forecast_record_sha256"]
