"""Forecast accuracy breakdowns preserve frequency, denominators and frozen references."""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark
from autoforecast.data_admission import load_dataset_contract
from autoforecast.frequency import advance, canonical_origin, parse_frequency, parse_origin
from autoforecast.panel import Forecast
from autoforecast.score_spark import ScoreError
from autoforecast.study import Study, load_study
from autoforecast.study_scoring import score_study
from provenance.lineage import directory_digest

pytest_plugins = ["autoforecast.test_study"]


def build_study(root, spark, *, frequency="D", metric="wrmsse", missing=None, scale_b=9.0,
                zero_actuals=False, validation_horizon=3, tiny_first_actual=False, optional_references=False):
    """Two retained series, three validation steps and two test steps; no exclusions."""
    root.mkdir(parents=True, exist_ok=True)
    cadence = parse_frequency(frequency)
    start = {"H": "2026-10-03T21:00:00", "D": "2026-10-03", "W": "2026-10-04",
             "M": "2023-10-31", "MS": "2023-10-01", "Y": "2020-12-31",
             "YS": "2020-01-01"}[cadence.code if cadence.anchor else cadence.unit]
    points = [advance(parse_origin(start, cadence), cadence, i) for i in range(8)]
    rows = []
    for series in ("A", "B"):
        for index, point in enumerate(points):
            value = (10.0 - (2 - index) * (2 if series == "A" else math.sqrt(9.0 if scale_b is None else scale_b))
                     if index < 3 else 0.0 if zero_actuals else 10.0)
            if index == 4 and (missing == "both" or (missing == "B" and series == "B")):
                value = None
            if index == 3 and tiny_first_actual:
                value = 1e-309
            rows.append({"item": series, "time": point, "units": value})
    source = root / "actuals.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    dataset = root / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"schema_version": 1, "dataset": {"id": "horizons", "panel": {
        "series_col": "item", "time_col": "time", "target_col": "units",
        "frequency": frequency, "season_length": 1, "target_estimand": "observed_nonnegative",
    }}}))
    origins = [canonical_origin(points[i], cadence) for i in (2, 5)]
    recipe = {
        "schema_version": 2, "source_keys": ["item"], "training_start": start,
        "validation_origins": origins[:1], "validation_horizon": validation_horizon,
        "final_train_cutoff": origins[1], "final_test_horizon": 2,
        "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
        "feature_availability": {}, "metric": metric, "frequency": cadence.code,
        "actual_missing": "mask" if missing else "error",
    }
    references = None
    if metric == "wrmsse" or optional_references:
        references = root / "references.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"keys": {"item": series}, "stage": stage, "origin": origin,
             "weight": weight, "scale": scale}
            for series, weight, scale in (("A", 80.0, 4.0), ("B", 20.0, scale_b))
            for stage, origin in zip(("validation", "test"), origins, strict=True)
        ]), references)
    manifest = benchmark.freeze_benchmark(
        contract_path=dataset, recipe=recipe, scoring_references=references,
        sources=dict.fromkeys(("selection", "validation_score", "final_training", "final_score"),
                             source), output_root=root / "registry", spark=spark,
    )
    forecasts = []
    for stage, origin, length, times in zip(
        ("validation", "test"), origins, (validation_horizon, 2),
        (points[3:3 + validation_horizon], points[6:]), strict=True,
    ):
        contract = benchmark.evaluation_contract(
            manifest, stage=stage, origin=origin, horizon=length, metric=metric,
        )
        cells = pq.read_table(contract["grid"]["path"]).to_pylist()
        path = root / f"{stage}-forecasts.parquet"
        pq.write_table(pa.Table.from_pylist([
            {**cell, "yhat": 2.0 if zero_actuals else 10.0 + (
                [0.0, -4.0, 8.0] if json.loads(cell["series_id"])["item"] == "A"
                else [3.0, 0.0, 12.0]
            )[times.index(cell["date"])]} for cell in cells
        ]), path)
        forecasts.append(Forecast("fixture", str(path), origin, length, 2, {"emitted": 2 * length},
                                  target_estimand="observed_nonnegative", frequency=cadence.code))
    return manifest, forecasts, points


def evaluate(manifest, forecast, spark, output, mode="both", metrics=None):
    recipe = benchmark.verify_benchmark(manifest)[1]["recipe"]
    contract = benchmark.evaluation_contract(
        manifest, stage="validation", origin=forecast.origin, horizon=forecast.horizon,
        metric=recipe["metric"],
    )
    panel = load_dataset_contract(manifest.parent / "dataset.yaml").panel(contract["actuals"]["path"])
    return score_study(panel, forecast, spark, contract=contract, output_dir=output,
                       forecast_accuracy_breakdown=mode, metrics=metrics)


def assert_breakdown_schema(report, modes, *, wrmsse=True):
    assert set(report) == {
        "schema", "mode", "frequency", "origin", "horizon", "metric_definition", "series_sums",
    } | modes
    keys = {
        "start_lead", "end_lead", "start_time", "end_time", "mae", "rmse", "wape",
        "bias", "coverage", "status", "unavailable_reason", "metric_status",
    } | ({"wrmsse"} if wrmsse else set())
    for mode in modes:
        assert all(set(row) == keys for row in report[mode])


@pytest.mark.parametrize("frequency", ["H", "D", "W", "M", "MS", "Y", "YS", "2D", "ME", "YE"])
def test_horizon_metrics_use_forecast_periods_and_square_errors_before_aggregation(
    tmp_path, spark, frequency,
):
    manifest, forecasts, points = build_study(tmp_path, spark, frequency=frequency)
    score = evaluate(manifest, forecasts[0], spark, tmp_path / "score")
    breakdown = score["forecast_accuracy_breakdown"]
    assert breakdown["schema"] == "autoforecast.forecast-accuracy-breakdown.v2"
    assert breakdown["frequency"] == parse_frequency(frequency).code
    assert breakdown["horizon"] == 3 and breakdown["origin"] == forecasts[0].origin
    assert_breakdown_schema(breakdown, {"per_step", "cumulative"})
    each, cumulative = breakdown["per_step"], breakdown["cumulative"]
    if frequency in {"M", "ME"}:
        assert each[1]["end_time"] == "2024-02-29"
    if frequency == "H":
        assert each[0]["end_time"] == "2026-10-04T00:00:00"
    if frequency == "2D":
        assert each[0]["end_time"] == "2026-10-09"
    assert [row["wrmsse"] for row in each] == pytest.approx([0.2, 1.6, 4.0])
    expected = 0.8 * math.sqrt(20 / 3) + 0.2 * math.sqrt(17 / 3)
    assert score["wrmsse"] == pytest.approx(expected)
    assert cumulative[1]["wrmsse"] == pytest.approx(0.8 * math.sqrt(2) + 0.2 * math.sqrt(0.5))
    assert cumulative[1]["mae"] == 1.75 and cumulative[1]["rmse"] == 2.5
    assert cumulative[1]["bias"] == -0.25 and cumulative[1]["wape"] == 0.175
    assert each[0] == cumulative[0]
    assert score["wrmsse"] != pytest.approx(sum(row["wrmsse"] for row in each) / 3)
    for key in ("mae", "rmse", "wape", "bias", "wrmsse"):
        assert cumulative[-1][key] == pytest.approx(score[key])
    assert cumulative[-1]["coverage"] == score["coverage"]
    for index, row in enumerate(each, 1):
        assert row["start_lead"] == row["end_lead"] == index
        assert row["start_time"] == row["end_time"] == points[index + 2].isoformat()
        assert row["coverage"] == dict(expected=2, originally_missing=0, zero_filled=0,
                                       excluded=0, scored=2)
        assert row["status"] == "scored" and row["unavailable_reason"] is None
        assert cumulative[index - 1]["start_lead"] == 1
        assert cumulative[index - 1]["end_lead"] == index
        assert cumulative[index - 1]["start_time"] == points[3].isoformat()
    evidence = breakdown["series_sums"]
    assert directory_digest(evidence["path"], suffixes=(".parquet",)) == evidence["sha256"]


def test_public_default_modes_saved_results_and_different_stage_lengths(tmp_path, spark):
    manifest, forecasts, _ = build_study(tmp_path, spark)
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        default = study.evaluate_forecasts(validation_forecasts=forecasts[0], testing_forecasts=forecasts[1],
                                           model_id="fixture")
        assert json.loads(Path(default["result_path"]).read_text()) == default
        for stage, length in (("validation", 3), ("test", 2)):
            report = default[stage]["score"]["forecast_accuracy_breakdown"]
            assert report["mode"] == "per_step" and "cumulative" not in report
            assert len(report["per_step"]) == length
            assert_breakdown_schema(report, {"per_step"})
        for mode in ("none", "cumulative", "both"):
            result = study.evaluate_forecasts(
                validation_forecasts=forecasts[0], testing_forecasts=forecasts[1],
                model_id="fixture", forecast_accuracy_breakdown=mode,
            )
            for stage in ("validation", "test"):
                score = result[stage]["score"]
                assert score["wrmsse"] == default[stage]["score"]["wrmsse"]
                if mode == "none":
                    assert "forecast_accuracy_breakdown" not in score
                    assert not (Path(result[stage]["scoring"]).parent / "scores/horizon-cell-sums").exists()
                else:
                    report = score["forecast_accuracy_breakdown"]
                    assert report["mode"] == mode and "cumulative" in report
                    assert ("per_step" in report) == (mode == "both")
                    assert_breakdown_schema(report, {"per_step", "cumulative"}
                                            if mode == "both" else {"cumulative"})


class EqualBothList(list):
    def __eq__(self, other):
        return other == "both"


class EqualBothStr(str):
    def __eq__(self, other):
        return other == "both"


@pytest.mark.parametrize("mode", [
    "individual", "wrong", None, True, {}, [], np.array("both"),
    EqualBothList(), EqualBothStr("wrong"),
])
def test_invalid_breakdown_mode_fails_before_input_access_or_output(tmp_path, mode):
    with pytest.raises(ValueError, match="forecast_accuracy_breakdown"):
        object.__new__(Study).evaluate_forecasts(
            validation_forecasts=None, testing_forecasts=None, model_id="fixture",
            forecast_accuracy_breakdown=mode,
        )
    with pytest.raises(ValueError, match="forecast_accuracy_breakdown"):
        score_study(None, None, None, contract=None, output_dir=tmp_path,
                    forecast_accuracy_breakdown=mode)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("missing", ["B", "both"])
def test_masked_lead_wrmsse_is_unavailable_without_reweighting(tmp_path, spark, missing):
    manifest, forecasts, _ = build_study(tmp_path, spark, missing=missing)
    score = evaluate(manifest, forecasts[0], spark, tmp_path / "score")
    report = score["forecast_accuracy_breakdown"]
    second = report["per_step"][1]
    assert second["wrmsse"] is None and second["status"] == "unavailable"
    assert second["unavailable_reason"] == "missing_series_actuals"
    assert second["coverage"] == dict(expected=2, originally_missing=1 if missing == "B" else 2,
                                      zero_filled=0, excluded=1 if missing == "B" else 2,
                                      scored=1 if missing == "B" else 0)
    assert second["mae"] == (4.0 if missing == "B" else None)
    assert report["metric_definition"]["label"] == "masked WRMSSE"
    assert report["cumulative"][1]["wrmsse"] == pytest.approx(
        0.8 * math.sqrt(2) + 0.2 if missing == "B" else 0.2,
    )
    assert report["cumulative"][-1]["wrmsse"] == pytest.approx(score["wrmsse"])


@pytest.mark.parametrize("scale", [None, 0.0])
def test_undefined_scale_is_unavailable_at_each_lead(tmp_path, spark, scale):
    manifest, forecasts, _ = build_study(tmp_path, spark, scale_b=scale)
    score = evaluate(manifest, forecasts[0], spark, tmp_path / "score")
    assert score["wrmsse"] is None
    for mode in ("per_step", "cumulative"):
        for row in score["forecast_accuracy_breakdown"][mode]:
            assert row["wrmsse"] is None and row["status"] == "unavailable"
            assert row["unavailable_reason"] == "undefined_scale"


@pytest.mark.parametrize("metric", ["mae", "rmse", "wape"])
def test_breakdown_supports_existing_cell_metrics_without_wrmsse_references(tmp_path, spark, metric):
    manifest, forecasts, _ = build_study(tmp_path, spark, metric=metric)
    score = evaluate(manifest, forecasts[0], spark, tmp_path / "score")
    assert_breakdown_schema(score["forecast_accuracy_breakdown"], {"per_step", "cumulative"},
                            wrmsse=False)
    for mode in ("per_step", "cumulative"):
        for row in score["forecast_accuracy_breakdown"][mode]:
            assert "wrmsse" not in row and row["status"] == "scored"
    assert score["forecast_accuracy_breakdown"]["cumulative"][-1][metric] == pytest.approx(score[metric])


@pytest.mark.parametrize("metric", ["wape", "mae"])
@pytest.mark.parametrize("references", ["none", "valid", "undefined"])
def test_zero_volume_wape_has_an_explicit_unavailable_reason(tmp_path, spark, metric, references):
    manifest, forecasts, _ = build_study(
        tmp_path, spark, metric=metric, zero_actuals=True,
        optional_references=references != "none", scale_b=None if references == "undefined" else 9.0,
    )
    score = evaluate(
        manifest, forecasts[0], spark, tmp_path / "score",
        metrics=("mae", "wape", "wrmsse") if references != "none" else None,
    )
    for mode in ("per_step", "cumulative"):
        for row in score["forecast_accuracy_breakdown"][mode]:
            assert row["wape"] is None and row["mae"] == row["rmse"] == 2.0
            assert row["status"] == ("unavailable" if metric == "wape" else "scored")
            assert row["unavailable_reason"] == ("zero_actual_volume" if metric == "wape" else None)
            assert row["metric_status"]["wape"] == {"status": "unavailable", "reason": "zero_actual_volume"}
            if references != "none":
                assert row["metric_status"]["wrmsse"] == {
                    "status": "unavailable" if references == "undefined" else "scored",
                    "reason": "undefined_scale" if references == "undefined" else None,
                }


def test_one_step_per_step_cumulative_and_overall_agree(tmp_path, spark):
    manifest, forecasts, _ = build_study(tmp_path, spark, validation_horizon=1)
    score = evaluate(manifest, forecasts[0], spark, tmp_path / "score")
    report = score["forecast_accuracy_breakdown"]
    assert report["horizon"] == 1
    assert report["per_step"] == report["cumulative"]
    assert report["per_step"][0]["wrmsse"] == pytest.approx(score["wrmsse"])


def test_horizon_overflow_raises_and_retains_failed_evidence(tmp_path, spark):
    manifest, forecasts, _ = build_study(tmp_path, spark, metric="wape", tiny_first_actual=True)
    baseline = evaluate(manifest, forecasts[0], spark, tmp_path / "baseline", mode="none")
    assert math.isfinite(baseline["wape"])
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        with pytest.raises(ScoreError, match="horizon metric overflowed"):
            study.evaluate_forecasts(validation_forecasts=forecasts[0], testing_forecasts=forecasts[1],
                                     model_id="fixture")
    from mlops.run_bundle import load_completed_run_bundle

    completed = list((tmp_path / "study-runs").rglob("completion.json"))
    assert len(completed) == 1
    assert load_completed_run_bundle(completed[0].parent)["status"] == "failed"


def test_diagnostic_evidence_changed_during_reduction_is_rejected(tmp_path, spark, monkeypatch):
    from autoforecast import study_horizon_metrics

    manifest, forecasts, _ = build_study(tmp_path, spark)
    digest = study_horizon_metrics.directory_digest
    calls = []
    def changed(path, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            part = next(Path(path).glob("*.parquet"))
            part.write_bytes(part.read_bytes() + b"\n")
        return digest(path, **kwargs)
    monkeypatch.setattr(study_horizon_metrics, "directory_digest", changed)
    with load_study(manifest.parent.name, registry_root=manifest.parent.parent, spark=spark) as study:
        with pytest.raises(ScoreError, match="horizon metric evidence changed"):
            study.evaluate_forecasts(validation_forecasts=forecasts[0], testing_forecasts=forecasts[1],
                                     model_id="fixture")
    assert len(calls) == 2
    from mlops.run_bundle import load_completed_run_bundle

    completed = list((tmp_path / "study-runs").rglob("completion.json"))
    assert len(completed) == 1
    assert load_completed_run_bundle(completed[0].parent)["status"] == "failed"


def test_masked_empty_step_cell_metrics_are_unavailable(tmp_path, spark):
    manifest, forecasts, _ = build_study(tmp_path, spark, metric="rmse", missing="both")
    score = evaluate(manifest, forecasts[0], spark, tmp_path / "score")
    second = score["forecast_accuracy_breakdown"]["per_step"][1]
    assert all(second[name] is None for name in ("mae", "rmse", "wape", "bias"))
    assert second["status"] == "unavailable" and second["unavailable_reason"] == "no_observed_actuals"
    assert score["rmse"] is not None
