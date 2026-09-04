"""Public workflow contracts: windows, selection, availability and future reuse."""

import copy
import datetime as dt
import decimal
import hashlib
import json
import math
import pathlib
import shutil
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark
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
    purposes = []

    def fit(request):
        purposes.append(request.purpose)
        history = pq.read_table(request.panel.path).to_pylist()
        assert max(row["at"] for row in history) == dt.date.fromisoformat(request.origin)
        seen.append(
            (request.origin, dict(request.model_parameters), dict(request.training_parameters))
        )
        return predict(request)

    # macOS exposes /tmp as an alias of /private/tmp; caller spelling is not identity.
    supplied = (
        pathlib.Path(str(complete).removeprefix("/private"))
        if str(complete).startswith("/private/tmp/")
        else complete
    )
    with open_study(supplied, spark) as study:
        result = study.run_forecast_workflow(
            fit_predict=fit,
            config=cfg,
            implementation=implementation(),
            output_dir=tmp_path / "runs",
        )
    assert len(seen) == expected_calls
    assert purposes == (
        ["selection"] * 6 + ["validation", "refit"]
        if selection_stage == "tuning"
        else ["selection", "selection", "refit"]
    )
    assert result.provenance["work_estimate"]["total_callback_fits"] == expected_calls
    assert result.status == "completed"
    assert result.provenance["study"]["path"] == str(complete.resolve())
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
            record["logical_snapshots"][name] = benchmark.parquet_content_hash(path)
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
    assert report["roles"][role]["status"] == "blocked"
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
    record["coverage_identities"][key]["content_sha256"] = benchmark.parquet_content_hash(path)
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
                record["logical_snapshots"][name] = benchmark.parquet_content_hash(path)
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
            record["logical_snapshots"][name] = benchmark.parquet_content_hash(path)
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
            record["logical_snapshots"][name] = benchmark.parquet_content_hash(path)
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

    monkeypatch.setattr(study_tuning, "_copy_run_artifact_directory", fail_copy)
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

    monkeypatch.setattr(study_workflow, "_copy_run_artifact_directory", copy)
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

    def broken_terminal_event(path, stage, status, **context):
        if stage == "run" and status == "FAIL":
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
