"""Public workflow contracts: windows, selection, availability and future reuse."""

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

from autoforecast import benchmark
from autoforecast.hpo import HPOError
from autoforecast.study_preflight import StudyPreflightError
from autoforecast.study_tuning import load_selected_configuration, load_tuning_result
from autoforecast.study_workflow import load_forecast_workflow
from autoforecast.test_study_tuning import (
    config,
    implementation,
    open_study,
    predict,
)
from autoforecast.test_study_tuning import (
    spark as _spark,
)
from provenance.lineage import directory_digest


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


def profile(root, **options):
    path = config(root, budget=2)
    value = yaml.safe_load(path.read_text())
    value.update(options)
    path.write_text(yaml.safe_dump(value))
    return path


def register(
    root,
    spark,
    *,
    missing=(),
    start_b=1,
    metric="mae",
    shift=0,
    count=2,
    scale_b=1.0,
    ids=("A", "B"),
    starts=None,
    tuning_horizon=2,
):
    # Population: exactly the supplied IDs (A/B by default); nulls stay null.
    root.mkdir(parents=True, exist_ok=True)
    starts = starts or {"B": start_b}

    def point(day):
        return dt.date(2024, 1, 1) + dt.timedelta(days=day - 1 + shift)

    source = root / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sku": sid,
                    "at": point(day),
                    "units": None if (sid, day) in missing else float(day),
                }
                for sid in ids
                for day in range(starts.get(sid, 1), 12)
            ]
        ),
        source,
    )
    dataset = root / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": {
                    "id": "workflow-test",
                    "panel": {
                        "series_col": "sku",
                        "time_col": "at",
                        "target_col": "units",
                        "frequency": "D",
                        "season_length": 1,
                        "target_estimand": "observed_nonnegative",
                    },
                },
            }
        )
    )
    recipe = {
        "schema_version": 2,
        "source_keys": ["sku"],
        "training_start": point(1).isoformat(),
        "frequency": "D",
        "validation_origins": [point(7).isoformat()],
        "validation_horizon": 2,
        "final_train_cutoff": point(9).isoformat(),
        "final_test_horizon": 2,
        "history_policy": "fixed",
        "new_series_policy": "error",
        "target_unit": "units",
        "feature_availability": {},
        "metric": metric,
        "actual_missing": "mask" if missing or any(v > 1 for v in starts.values()) else "error",
        "tuning_windows": [
            {"origin": point(day).isoformat(), "horizon": tuning_horizon}
            for day in range(3, 3 + count)
        ],
    }
    if any(v > 1 for v in starts.values()):
        recipe["series_starts"] = {
            "rule": "declared",
            "values": [
                {"keys": {"sku": sid}, "start": point(starts.get(sid, 1)).isoformat()}
                for sid in ids
            ],
        }
    refs = root / "references.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "keys": {"sku": sid},
                    "stage": stage,
                    "origin": point(day).isoformat(),
                    "weight": weight,
                    "scale": 1.0 if sid == "A" else scale_b,
                }
                for sid, weight in ((sid, 0.8 if sid == "A" else 0.2) for sid in ids)
                for stage, day in [
                    *(("tuning", day) for day in range(3, 3 + count)),
                    ("validation", 7),
                    ("test", 9),
                ]
            ]
        ),
        refs,
    )
    return benchmark.freeze_benchmark(
        contract_path=dataset,
        sources=dict.fromkeys(
            ("selection", "final_training", "validation_score", "final_score"), source
        ),
        recipe=recipe,
        scoring_references=refs,
        output_root=root / ".autoforecast/benchmarks",
        spark=spark,
    )


@pytest.fixture(scope="module")
def complete(tmp_path_factory, spark):
    return register(tmp_path_factory.mktemp("workflow-complete"), spark, count=3)


@pytest.fixture(scope="module")
def incomplete(tmp_path_factory, spark):
    return register(
        tmp_path_factory.mktemp("workflow-incomplete"), spark, missing=(("B", 5),), tuning_horizon=3
    )


@pytest.fixture(scope="module")
def three_day(tmp_path_factory, spark):
    return register(tmp_path_factory.mktemp("workflow-three-day"), spark, tuning_horizon=3)


def test_multiwindow_hand_calculation_and_completed_search_reuse(complete, spark, tmp_path):
    cfg = profile(tmp_path, selection={"origins": "all"}, reporting={"metrics": ["mae", "wrmsse"]})
    calls = []

    def fit(request):
        calls.append((request.origin, dict(request.model_parameters), request.seed))
        history = pq.read_table(request.panel.path).to_pylist()
        assert max(row["at"] for row in history) == dt.date.fromisoformat(request.origin)
        return predict(request)

    with open_study(complete, spark) as study:
        result = study.tune(
            fit_predict=fit,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "saved",
        )
        assert len(calls) == 6
        assert [x[0] for x in calls[:3]] == ["2024-01-03", "2024-01-04", "2024-01-05"]
        assert all(c[1:] == calls[0][1:] for c in calls[:3])
        # Winner 10 gives per-window MAEs 5.5,4.5,3.5; 0 gives 4.5,5.5,6.5.
        assert result.parameters["model.level"] == 10
        assert result.tuning_score == pytest.approx(4.5)
        assert result.forecasts is None and result.model_artifacts is None
        assert len(result.windows) == 3
        expected = [
            math.sqrt((6**2 + 5**2) / 2),
            math.sqrt((5**2 + 4**2) / 2),
            math.sqrt((4**2 + 3**2) / 2),
        ]
        assert [w["score"]["wrmsse"] for w in result.windows] == pytest.approx(expected)
        assert result.winner_config.is_relative_to(tmp_path / "saved")
        assert result.search_config.read_bytes() == cfg.read_bytes()
        again = study.tune(
            fit_predict=lambda _: pytest.fail("reuse must not fit"),
            config=cfg,
            implementation=implementation(),
            reuse_tuning=result.path,
        )
        assert again.path == result.path
        with pytest.raises(HPOError, match="differs"):
            study.tune(
                fit_predict=lambda _: pytest.fail("must not fit"),
                config=cfg,
                implementation={**implementation(), "version": "2"},
                reuse_tuning=result.path,
            )
    assert load_tuning_result(result.path).parameters == result.parameters
    assert load_selected_configuration(result.winner_config)["parameters"] == result.parameters


@pytest.mark.parametrize("selection_stage,expected_calls", [("tuning", 8), ("validation", 3)])
def test_workflow_counts_and_saved_configuration_future_reuse(
    complete, spark, tmp_path, selection_stage, expected_calls
):
    selection = {"stage": selection_stage}
    if selection_stage == "tuning":
        selection["origins"] = "all"
    cfg = profile(tmp_path, selection=selection, reporting={"metrics": ["mae", "rmse", "wrmsse"]})
    seen = []

    def fit(request):
        history = pq.read_table(request.panel.path).to_pylist()
        assert max(row["at"] for row in history) == dt.date.fromisoformat(request.origin)
        seen.append(
            (request.origin, dict(request.model_parameters), dict(request.training_parameters))
        )
        return predict(request)

    with open_study(complete, spark) as study:
        result = study.run_forecast_workflow(
            fit_predict=fit,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "runs",
        )
    assert len(seen) == expected_calls
    assert result.provenance["work_estimate"]["total_callback_fits"] == expected_calls
    assert result.status == "completed"
    assert seen[-1][0] == "2024-01-09"
    expected_parameters = (
        {"level": result.parameters["model.level"]},
        {"offset": result.parameters["training.offset"]},
    )
    assert seen[-1][1:] == expected_parameters
    if selection_stage == "tuning":
        assert seen[-2] == ("2024-01-07", *expected_parameters)
    assert result.testing["model_artifacts"] != result.validation["model_artifacts"]
    assert load_forecast_workflow(result.path).parameters == result.parameters
    assert result.search_config.read_bytes() == cfg.read_bytes()
    if selection_stage == "validation":
        with open_study(complete, spark) as study:
            reopened = study.run_forecast_workflow(
                fit_predict=fit,
                config=result.search_config,
                reuse_tuning=result.tuning_path,
                implementation=implementation(),
                output_dir=tmp_path / "reopened",
            )
        assert len(seen) == expected_calls + 1  # only a new final refit
        assert reopened.validation["forecast_sha256"] == result.validation["forecast_sha256"]
        assert reopened.provenance["work_estimate"]["total_callback_fits"] == 1
    future = register(tmp_path / "future", spark, shift=31, count=1)
    seen.clear()
    with open_study(future, spark) as study:
        reused = study.run_forecast_workflow(
            fit_predict=fit,
            selected_config=result.selected_config,
            implementation=implementation(),
            output_dir=tmp_path / "future-runs",
        )
    assert len(seen) == 2  # two fresh fits, zero HPO trials
    assert reused.tuning_path is None
    assert yaml.safe_load(reused.search_config.read_text()) == yaml.safe_load(cfg.read_text())
    assert reused.parameters == result.parameters
    assert all(c[1]["level"] == result.parameters["model.level"] for c in seen)
    assert seen[0][0] == "2024-02-07"
    assert reused.provenance["configuration_source"] == result.provenance["configuration_source"]
    if selection_stage == "validation":
        with open_study(future, spark) as study:
            retuned = study.tune(
                fit_predict=fit, config=reused.search_config, implementation=implementation()
            )
        assert len(seen) == 4  # two earlier fixed fits plus two newly requested HPO trials
        assert retuned.provenance["study"]["sha256"] != result.provenance["study"]["sha256"]


def test_three_availability_policies_and_common_date_wrmsse(incomplete, spark, tmp_path):
    with open_study(incomplete, spark) as study:
        cfg = profile(
            tmp_path,
            selection={"origins": ["2024-01-03"]},
            reporting={"metrics": ["mae", "wrmsse"]},
        )
        strict = study.preflight(config=cfg)
        assert strict.report["retained_count"] == 2  # frozen mask is explicitly allowed
        cfg = profile(
            tmp_path,
            selection={"origins": ["2024-01-03"]},
            reporting={"metrics": ["mae", "wrmsse"]},
            series_policy="exclude_unavailable",
        )
        reduced = study.preflight(config=cfg)
        assert reduced.report["retained_count"] == 1
        assert study.preflight(config=cfg).study_id == reduced.study_id
        cfg = profile(
            tmp_path,
            selection={"origins": ["2024-01-03"]},
            reporting={"metrics": ["mae", "wrmsse"]},
            series_policy="use_available",
        )
        available = study.tune(fit_predict=predict, config=cfg, implementation=implementation())
    score = available.windows[0]["score"]
    assert score["n_scored"] == 5
    assert score["wrmsse_coverage"]["common_dates"] == ["2024-01-04", "2024-01-06"]
    level = available.parameters["model.level"]
    assert score["wrmsse"] == pytest.approx(math.sqrt(((level - 4) ** 2 + (level - 6) ** 2) / 2))
    assert score["mae"] == pytest.approx(
        (2 * abs(level - 4) + abs(level - 5) + 2 * abs(level - 6)) / 5
    )
    assert score["coverage"]["original_expected"] == 6


def test_preflight_failure_retains_report_without_callback(complete, spark, tmp_path):
    cfg = profile(tmp_path, selection={"origins": "all"}, minimum_history=100)
    with open_study(complete, spark) as study:
        with pytest.raises(StudyPreflightError) as caught:
            study.run_forecast_workflow(
                fit_predict=lambda _: pytest.fail("no fitting before preflight"),
                config=cfg,
                implementation=implementation(),
                output_dir=tmp_path / "failed",
            )
    report = json.loads(caught.value.report_path.read_text())
    assert report["status"] == "blocked" and report["callback_count"] == 0
    failed = list((tmp_path / "failed").glob("workflow-*/failure.json"))
    assert len(failed) == 1 and json.loads(failed[0].read_text())["phase"] == "preflight"


@pytest.mark.parametrize("stage", ["tuning", "validation"])
@pytest.mark.parametrize("metric", ["mae", "rmse", "wape", "wrmsse"])
def test_configured_objective_changes_winner(complete, three_day, spark, tmp_path, stage, metric):
    selection = {"stage": stage, "metric": metric}
    if stage == "tuning":
        selection["origins"] = ["2024-01-03"]
    cfg = profile(tmp_path, selection=selection, reporting={"metrics": ["mae", "rmse", "wrmsse"]})
    seen = set()

    def fit(request):
        level = request.model_parameters["level"]
        seen.add(level)
        result = predict(request)
        rows = pq.read_table(result.forecasts.path).to_pylist()
        last = request.forecast_dates[-1]
        for row in rows:
            # Inner: configurations have errors [0,0,6] versus [2.5,2.5,2.5].
            error = (
                (2.0 * len(request.forecast_dates) if row["date"] == last else 0.0)
                if level == 0
                else 2.5
            )
            row["yhat"] = row["date"].day + error
        pq.write_table(
            pa.Table.from_pylist(rows), pathlib.Path(result.forecasts.path) / "part.parquet"
        )
        return result

    with open_study(three_day if stage == "tuning" else complete, spark) as study:
        before = study.recipe
        result = study.tune(fit_predict=fit, config=cfg, implementation=implementation())
        assert study.recipe == before and study.recipe["metric"] == "mae"
    assert seen == {0, 10}
    assert result.parameters["model.level"] == (10 if metric in ("rmse", "wrmsse") else 0)
    assert result.provenance["metric"] == metric


@pytest.mark.parametrize(
    "cadence,start,outer",
    [
        ("H", "2024-01-01T00:00:00", "2024-01-10T00:00:00"),
        ("D", "2024-01-01", "2024-06-01"),
        ("W", "2024-01-01", "2024-06-03"),
        ("MS", "2020-01-01", "2024-01-01"),
        ("YS", "2000-01-01", "2024-01-01"),
    ],
)
@pytest.mark.parametrize("count", [1, 2, 5, None])
def test_schedule_resolves_counts_and_cadences(cadence, start, outer, count):
    from autoforecast.frequency import advance, parse_origin, period_points

    recipe = {"frequency": cadence, "training_start": start, "validation_origins": [outer]}
    schedule = {"horizon": 2, "stride": 2}
    if count is not None:
        schedule["count"] = count
    windows = benchmark.resolve_tuning_schedule(recipe, schedule)
    assert len(windows) == (3 if count is None else count)
    assert list(period_points(windows[-1]["origin"], cadence, 2))[-1] == parse_origin(
        outer, cadence
    )
    for left, right in zip(windows, windows[1:], strict=False):
        assert advance(parse_origin(left["origin"], cadence), cadence, 2) == parse_origin(
            right["origin"], cadence
        )


def test_schedule_rejects_conflicts_and_impossible_history(complete):
    recipe = benchmark.read_benchmark_manifest(complete)[1]["recipe"]
    with pytest.raises(benchmark.BenchmarkError, match="conflict|combine|tuning_windows"):
        benchmark.validate_recipe({**recipe, "tuning_schedule": {"horizon": 2, "stride": 2}})
    for schedule in (
        {"horizon": 2, "stride": 2, "count": 100},
        {"horizon": 0, "stride": 2},
        {"horizon": 2, "stride": True},
        {"horizon": 2, "stride": 2, "count": -1},
        {"horizon": 2, "stride": 2, "count": 0},
        {"horizon": 2, "stride": 0, "count": 1},
    ):
        with pytest.raises(benchmark.BenchmarkError):
            benchmark.resolve_tuning_schedule(recipe, schedule)


def test_final_failure_is_retained_and_cannot_reload_as_success(complete, spark, tmp_path):
    cfg = profile(tmp_path, selection={"stage": "validation"})
    calls = []

    def fit(request):
        calls.append(request.origin)
        if request.origin == "2024-01-09":
            raise RuntimeError("deliberate final refit failure")
        return predict(request)

    with open_study(complete, spark) as study:
        with pytest.raises(RuntimeError, match="deliberate"):
            study.run_forecast_workflow(
                fit_predict=fit,
                config=cfg,
                implementation=implementation(),
                output_dir=tmp_path / "runs",
            )
    assert len(calls) == 3
    paths = list((tmp_path / "runs").glob("workflow-*"))
    assert len(paths) == 1
    assert json.loads((paths[0] / "failure.json").read_text())["phase"] == "final_refit"
    assert not (paths[0] / "result.json").exists()
    with pytest.raises(HPOError, match="successfully"):
        load_forecast_workflow(paths[0])


def test_late_history_masks_are_frozen_and_wrmsse_unavailable(spark, tmp_path):
    manifest = register(tmp_path / "study", spark, start_b=4)
    cfg = profile(tmp_path, selection={"origins": "all"}, reporting={"metrics": ["mae", "wrmsse"]})
    calls = []

    def fit(request):
        calls.append((request.origin, request.series_mapping.count()))
        return predict(request)

    with open_study(manifest, spark) as study:
        with pytest.raises(StudyPreflightError, match="availability"):
            study.tune(fit_predict=fit, config=cfg, implementation=implementation())
        assert calls == []
        cfg = profile(
            tmp_path,
            selection={"origins": "all"},
            reporting={"metrics": ["mae", "wrmsse"]},
            series_policy="use_available",
        )
        result = study.tune(fit_predict=fit, config=cfg, implementation=implementation())
        assert calls == [("2024-01-03", 1), ("2024-01-04", 2)] * 2
        first = result.windows[0]["score"]
        assert first["wrmsse"] is None
        assert first["metric_status"]["wrmsse"]["reason"] == "unfittable_weighted_series"
        assert first["population"]["original_series"] == 2
        assert first["population"]["forecasted_series"] == 1
        assert first["evaluation_status"] == "partial"
        cfg = profile(
            tmp_path,
            selection={"origins": "all", "metric": "wrmsse"},
            series_policy="use_available",
        )
        calls.clear()
        with pytest.raises(StudyPreflightError, match="unfittable_weighted_series"):
            study.tune(fit_predict=fit, config=cfg, implementation=implementation())
        assert calls == []


def test_optional_undefined_scale_and_pending_testing(spark, tmp_path):
    manifest = register(
        tmp_path / "study",
        spark,
        count=1,
        scale_b=0.0,
        missing=tuple((sid, day) for sid in ("A", "B") for day in (10, 11)),
    )
    cfg = profile(
        tmp_path,
        selection={"stage": "validation"},
        reporting={"metrics": ["mae", "wrmsse"]},
        series_policy="use_available",
    )
    with open_study(manifest, spark) as study:
        result = study.run_forecast_workflow(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "runs",
        )
    assert result.validation["score"]["mae"] is not None
    assert result.validation["score"]["wrmsse"] is None
    assert result.status == "pending_actuals"
    assert result.testing["score"]["mae"] is None
    assert result.testing["score"]["population"]["observed_cells"] == 0
    assert result.testing["forecasts"].n_series == 2


def test_saved_config_incompatibility_and_tampering_block_before_fitting(complete, spark, tmp_path):
    cfg = profile(tmp_path, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        selected = study.tune(fit_predict=predict, config=cfg, implementation=implementation())
        with pytest.raises(HPOError, match="implementation"):
            study.run_forecast_workflow(
                fit_predict=lambda _: pytest.fail("must not fit"),
                selected_config=selected.winner_config,
                implementation={**implementation(), "version": "2"},
            )
        changed = yaml.safe_load(cfg.read_text())
        changed["hyperparameters"]["model.level"]["valid"]["high"] = 1.0
        changed["hyperparameters"]["model.level"]["search"] = {
            "kind": "categorical",
            "values": [0.0],
        }
        cfg.write_text(yaml.safe_dump(changed))
        from autoforecast.tuning import TuningError

        with pytest.raises(TuningError, match="valid domain"):
            study.run_forecast_workflow(
                fit_predict=lambda _: pytest.fail("must not fit"),
                config=cfg,
                selected_config=selected.winner_config,
                implementation=implementation(),
            )
        selected.winner_config.write_text(selected.winner_config.read_text() + " ")
        from autoforecast.experiments import ExperimentError

        with pytest.raises(ExperimentError):
            study.run_forecast_workflow(
                fit_predict=lambda _: pytest.fail("must not fit"),
                selected_config=selected.winner_config,
                implementation=implementation(),
            )


def test_earlier_window_model_and_nonselected_trial_tampering_rejected(complete, spark, tmp_path):
    import shutil

    from autoforecast.experiments import ExperimentError

    cfg = profile(tmp_path, selection={"origins": "all"})
    with open_study(complete, spark) as study:
        selected = study.tune(fit_predict=predict, config=cfg, implementation=implementation())
    for candidate in selected.trial_history:
        copy = tmp_path / candidate["trial_id"]
        shutil.copytree(selected.path, copy)
        model = copy / candidate["attempt_path"] / "windows/0000/model/constant.json"
        model.write_text("changed first-window model")
        with pytest.raises(ExperimentError):
            load_tuning_result(copy)


def test_exclusion_union_is_shared_across_models_at_thousand_series(spark, tmp_path):
    ids = ("A", "B", "C", *(f"S{i:04d}" for i in range(3, 1000)))
    manifest = register(
        tmp_path / "study", spark, ids=ids, starts={"A": 3}, missing=(("B", 5), ("C", 10))
    )
    cfg = profile(
        tmp_path,
        selection={"stage": "validation"},
        minimum_history=2,
        reporting={"metrics": ["mae", "wrmsse"]},
    )
    seen = []

    def fit(request):
        seen.append(set(row[request.panel.series_col] for row in request.series_mapping.collect()))
        return predict(request)

    with open_study(manifest, spark) as study:
        with pytest.raises(StudyPreflightError):
            study.tune(fit_predict=fit, config=cfg, implementation=implementation())
        assert not seen
        cfg = profile(
            tmp_path,
            selection={"stage": "validation"},
            minimum_history=2,
            reporting={"metrics": ["mae", "wrmsse"]},
            series_policy="exclude_unavailable",
        )
        report = study.preflight(config=cfg)
        assert report.report["original_count"] == 1000 and report.report["retained_count"] == 997
        results = [
            study.tune(
                fit_predict=fit, config=cfg, implementation={**implementation(), "name": name}
            )
            for name in ("model-one", "model-two")
        ]
    assert len(seen) == 4 and all(len(ids) == 997 and ids == seen[0] for ids in seen)
    assert all(result.provenance["study_id"] == report.study_id for result in results)
    with open_study(report.study_path, spark) as effective:
        refs = effective.scoring_references.collect()
        assert all(row["weight"] == pytest.approx(1 / 997) for row in refs)
        assert effective.n_series == 997


def test_available_policy_never_masks_missing_forecasts(incomplete, spark, tmp_path):
    cfg = profile(tmp_path, selection={"origins": ["2024-01-03"]}, series_policy="use_available")

    def drop_masked(request):
        result = predict(request)
        file = pathlib.Path(result.forecasts.path) / "part.parquet"
        rows = pq.read_table(file).to_pylist()
        rows = [
            r for r in rows if not (json.loads(r["series_id"])["sku"] == "B" and r["date"].day == 5)
        ]
        assert len(rows) == 5
        pq.write_table(pa.Table.from_pylist(rows), file)
        return result

    from autoforecast.panel import ForecastError
    from autoforecast.score_spark import ScoreError

    with open_study(incomplete, spark) as study, pytest.raises((ScoreError, ForecastError)):
        study.tune(fit_predict=drop_masked, config=cfg, implementation=implementation())


@pytest.mark.parametrize(
    "role,defect",
    [
        ("final_training", "wrong_id"),
        ("final_score", "wrong_id"),
        ("validation_score", "duplicate"),
        ("final_training", "duplicate"),
        ("final_training", "wrong_date"),
        *[
            (role, defect)
            for role in ("selection", "final_training", "validation_score", "final_score")
            for defect in ("missing_row", "missing_id")
        ],
    ],
)
def test_preflight_checks_exact_ids_and_duplicates_before_any_fit(
    complete, spark, tmp_path, role, defect
):
    # Deliberately construct a self-consistent byte manifest with invalid row semantics.
    # This distinguishes semantic preflight checks from merely detecting changed hashes.
    import shutil

    from provenance.lineage import directory_digest

    copied = tmp_path / "registry/original"
    shutil.copytree(complete.parent, copied)
    record = json.loads((copied / "benchmark.json").read_text())
    relative = record["role_paths"][role]
    path = copied / relative
    table = pq.read_table(path)
    rows = table.to_pylist()
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "wrong_date":
        rows[0]["at"] = dt.date(2023, 12, 31)
    elif defect == "missing_row":
        rows = rows[1:]
    elif defect == "missing_id":
        rows = [row for row in rows if row["__af_series_id"] != rows[0]["__af_series_id"]]
    else:
        old = rows[0]["__af_series_id"]
        for row in rows:
            if row["__af_series_id"] == old:
                row["__af_series_id"] = '{"sku":"unexpected"}'
    shutil.rmtree(path)
    path.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path / "part.parquet")
    for name, target in record["role_paths"].items():
        if target == relative:
            record["artifacts"][name] = directory_digest(path, suffixes=(".parquet",))
            record["logical_snapshots"][name] = benchmark._parquet_hash(path)
    identity = benchmark.digest(benchmark._identity(record))
    record["study_id"] = record["benchmark_id"] = identity
    (copied / "benchmark.json").write_text(json.dumps(record))
    final = copied.parent / identity
    copied.rename(final)
    cfg = profile(tmp_path, selection={"origins": "all"}, series_policy="use_available")
    with open_study(final / "benchmark.json", spark) as study:
        with pytest.raises(
            StudyPreflightError, match="unexpected|duplicate|date|missing"
        ) as caught:
            study.tune(
                fit_predict=lambda _: pytest.fail(
                    "corrupt downstream input must block before fitting"
                ),
                config=cfg,
                implementation=implementation(),
            )
    report = json.loads(caught.value.report_path.read_text())
    assert report["status"] == "blocked" and report["callback_count"] == 0
    if defect == "wrong_date":
        assert report["issues"][0]["stage"] == role
        assert report["roles"][role]["status"] == "blocked"
        assert report["roles"]["final_score"]["status"] == "not_checked"


def test_testing_target_values_cannot_change_selection(complete, spark, tmp_path):
    record = benchmark.read_benchmark_manifest(complete)[1]
    source = pathlib.Path(record["sources"]["final_score"]["path"])
    table = pq.read_table(source)
    rows = table.to_pylist()
    for row in rows:
        if row["at"].day >= 10:
            row["units"] *= 10000
    changed = tmp_path / "changed-testing.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), changed)
    sources = {
        key: pathlib.Path(record["sources"][key]["path"])
        for key in ("selection", "final_training", "validation_score", "final_score")
    }
    sources["final_score"] = changed
    other = benchmark.freeze_benchmark(
        contract_path=source.parent / "dataset.yaml",
        sources=sources,
        recipe=record["recipe"],
        scoring_references=pathlib.Path(record["sources"]["scoring_references"]["path"]),
        output_root=tmp_path / "registry",
        spark=spark,
    )
    cfg = profile(tmp_path, selection={"stage": "validation"})
    results = []
    for manifest in (complete, other):
        with open_study(manifest, spark) as study:
            results.append(
                study.tune(fit_predict=predict, config=cfg, implementation=implementation())
            )
    assert results[0].parameters == results[1].parameters
    assert results[0].tuning_score == results[1].tuning_score
    assert results[0].provenance["study"]["sha256"] != results[1].provenance["study"]["sha256"]


def test_wrmsse_reporting_requires_registered_references(spark, tmp_path):
    from autoforecast.test_study_tuning import freeze

    manifest = freeze(tmp_path / "study", spark, metric="mae")
    cfg = profile(tmp_path, reporting={"metrics": ["mae", "wrmsse"]})
    with (
        open_study(manifest, spark) as study,
        pytest.raises(StudyPreflightError, match="references"),
    ):
        study.tune(
            fit_predict=lambda _: pytest.fail("no references must fail before fitting"),
            config=cfg,
            implementation=implementation(),
        )


def test_schedule_is_resolved_once_during_registration(complete, spark, tmp_path):
    record = benchmark.read_benchmark_manifest(complete)[1]
    recipe = dict(record["recipe"])
    recipe.pop("tuning_windows")
    recipe["tuning_schedule"] = {"horizon": 2, "stride": 1}  # default count=3 -> origins 3,4,5
    sources = {
        key: pathlib.Path(record["sources"][key]["path"])
        for key in ("selection", "final_training", "validation_score", "final_score")
    }
    resolved = benchmark.freeze_benchmark(
        contract_path=sources["selection"].parent / "dataset.yaml",
        sources=sources,
        recipe=recipe,
        scoring_references=pathlib.Path(record["sources"]["scoring_references"]["path"]),
        output_root=tmp_path / "registry",
        spark=spark,
    )
    with open_study(resolved, spark) as study:
        assert study.recipe["resolved_tuning_schedule"] == {"horizon": 2, "stride": 1, "count": 3}
        assert study.recipe["tuning_windows"] == record["recipe"]["tuning_windows"]
        assert "tuning_schedule" not in study.recipe
    assert benchmark.read_benchmark_manifest(complete)[1]["recipe"] == record["recipe"]


@pytest.mark.parametrize(
    "options",
    [
        {"selection": {"stage": "test"}},
        {"selection": {"metric": "bias"}},
        {"selection": {"direction": "maximize"}},
        {"selection": {"aggregation": "pooled"}},
        {"selection": {"unexpected": 1}},
        {"selection": {"origins": []}},
        {"selection": {"origins": ["2024-01-03", "2024-01-03"]}},
        {"selection": {"stage": "validation", "origins": "all"}},
        {"reporting": {"metrics": ["mae", "unknown"]}},
        {"reporting": {"metrics": ["mae", "mae"]}},
        {"reporting": {"forecast_accuracy_breakdown": "daily"}},
        {"series_policy": "fill_zero"},
        {"minimum_history": 0},
        {"minimum_history": True},
    ],
)
def test_policy_rejects_invalid_configuration_before_callback(complete, spark, tmp_path, options):
    cfg = profile(tmp_path, selection={"origins": "all"})
    values = yaml.safe_load(cfg.read_text())
    values.update(options)
    cfg.write_text(yaml.safe_dump(values))
    with open_study(complete, spark) as study, pytest.raises(ValueError):
        study.tune(
            fit_predict=lambda _: pytest.fail("invalid policy must not fit"),
            config=cfg,
            implementation=implementation(),
        )


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
        )
        rows = table.to_pylist()
    else:
        rows[0]["scored"] += 1
        rows[1]["scored"] -= 1
    shutil.rmtree(path)
    path.mkdir(parents=True)
    pq.write_table(type(table).from_pylist(rows, schema=table.schema), path / "part.parquet")
    record["coverage"][key]["sha256"] = directory_digest(path, suffixes=(".parquet",))
    record["coverage_identities"][key]["content_sha256"] = benchmark._parquet_hash(path)
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
                record["logical_snapshots"][name] = benchmark._parquet_hash(path)
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


@pytest.mark.parametrize("wrong_score", [False, True])
def test_legacy_v1_reload_rejects_score_objective_mismatch(complete, spark, tmp_path, wrong_score):
    cfg_root = tmp_path / "legacy-config"
    cfg_root.mkdir()
    cfg = profile(cfg_root, selection={"stage": "validation"})
    with open_study(complete, spark) as study:
        selected = study.tune(
            fit_predict=predict,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "legacy-search",
        )
    selection_path = selected.path / "selection.json"
    selection = json.loads(selection_path.read_text())
    chosen = selection["selected"][0]
    attempt = selected.path / chosen["attempt_path"]
    evidence_path = attempt / "evaluation/external-run.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["schema"] == "autoforecast.external-trial.v1"
    evidence.pop("windows")
    evidence.pop("aggregation")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    score_path = attempt / "evaluation/score.json"
    score = json.loads(score_path.read_text())
    if wrong_score:
        score[evidence["context"]["metric"]] = 999.0
    score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")
    manifest_sha = _reseal_attempt(attempt)
    for group in (selection["candidates"], selection["selected"]):
        for item in group:
            if item["attempt_path"] == chosen["attempt_path"]:
                item["artifact_manifest_sha256"] = manifest_sha
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    if wrong_score:
        with pytest.raises(HPOError, match="score|objective|selection"):
            load_tuning_result(selected.path)
    else:
        assert load_tuning_result(selected.path).tuning_score == selected.tuning_score


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
    ):
        root = tmp_path / field
        shutil.copytree(result.path, root)
        path = root / "result.json"
        record = json.loads(path.read_text())
        if field == "options":
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
        elif field == "source":
            record["provenance"]["configuration_source"]["trial_id"] = "wrong-trial"
        elif field == "implementation":
            record["provenance"][field]["version"] = "changed"
        else:
            record["provenance"][field] += 1
        path.write_text(json.dumps(record))
        _reseal_attempt(root)
        with pytest.raises(HPOError, match="configuration"):
            load_forecast_workflow(root)
        with pytest.raises(HPOError, match="configuration"):
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
            record["logical_snapshots"][name] = benchmark._parquet_hash(path)
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
