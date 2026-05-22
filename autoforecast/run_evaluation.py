"""Run-local, explicitly exploratory forecast comparison.

This is deliberately not a promotion leaderboard.  It scores only forecasts produced by one
frozen request, on their common emitted cells, and labels the result accordingly.  A governed
promotion still uses the multi-origin ``ranking`` protocol and a held-out acceptance period.
"""

from __future__ import annotations

import html
import json
import pathlib
from collections.abc import Mapping
from typing import Literal, cast

from pyspark.sql import SparkSession

from autoforecast.benchmarks.m5 import M5_LEVELS
from autoforecast.frequency import parse_frequency
from autoforecast.panel import Forecast, Panel
from autoforecast.ranking import SelectionError, _exact_grid_restrictions, common_cells
from autoforecast.score import Score
from autoforecast.score_spark import MissingActual, ScoreError, score_forecast
from autoforecast.wrmsse import Wrmsse, WrmsseError
from autoforecast.wrmsse import score as score_wrmsse
from provenance.lineage import directory_digest

__all__ = ["EvaluationError", "compare_run_forecasts"]

Metric = Literal["rmse", "mae", "wape", "wrmsse"]


class EvaluationError(ValueError):
    """The declared run-local comparison cannot be performed honestly."""


def _score_record(score: Score | Wrmsse | dict) -> dict[str, object]:
    if isinstance(score, dict):
        return score
    if isinstance(score, Wrmsse):
        return score.as_record()
    return {
        "n_scored": score.n_scored,
        "n_dropped_ineligible": score.n_dropped_ineligible,
        "n_dropped_no_data": score.n_dropped_no_data,
        "n_zero_filled": score.n_zero_filled,
        "mae": score.mae,
        "rmse": score.rmse,
        "wape": score.absolute_error_pct_of_volume,
        "bias": score.bias,
    }


def _value(score: Score | Wrmsse | dict, metric: Metric) -> float | None:
    """Extract only the declared metric family; never reinterpret hierarchy and cell scores."""
    if isinstance(score, dict):
        return score[metric]
    if isinstance(score, Wrmsse):
        if metric != "wrmsse":
            raise EvaluationError("a hierarchy score cannot supply a cell metric")
        return score.score
    if metric == "wrmsse":
        raise EvaluationError("a cell score cannot supply WRMSSE")
    return score.absolute_error_pct_of_volume if metric == "wape" else float(getattr(score, metric))


def _write_html(path: pathlib.Path, record: Mapping[str, object]) -> None:
    is_wrmsse = record.get("metric") == "wrmsse"
    fixed = record.get("evaluation_contract")
    flat = isinstance(fixed, dict) and fixed.get("schema") == "autoforecast.fixed-evaluation.v2"
    rows = []
    for item in record["candidates"]:  # type: ignore[index]
        assert isinstance(item, dict)
        score = item.get("score")
        if is_wrmsse and isinstance(score, dict):
            cells = (
                f"<td>{html.escape(str(score.get('wrmsse')))}</td>"
                f"<td>{html.escape(str(score.get('n_scored' if flat else 'n_series_scored')))}</td>"
            )
        elif isinstance(score, dict):
            values = (score.get("rmse"), score.get("mae"), score.get("wape"), score.get("n_scored"))
            cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        else:
            cells = "<td colspan='2'>—</td>" if is_wrmsse else "<td colspan='4'>—</td>"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['model_id']))}</td>"
            f"<td>{html.escape(str(item['status']))}</td>{cells}"
            f"<td>{html.escape(str(item.get('detail', '')))}</td></tr>"
        )
    ranking = record.get("ranking")
    if record.get("comparison_kind") == "fixed_benchmark":
        note = (
            "Fixed benchmark: complete frozen forecast population; the declared actual policy "
            "controls scored coverage. See per-series coverage for exclusions."
        )
    elif is_wrmsse:
        note = (
            "M5 full-grid WRMSSE: every candidate must score the complete published "
            "series-by-horizon grid; omitted cells are refused rather than compared."
        )
    else:
        note = (
            f"Exploratory rank by {html.escape(str(ranking['metric']))} on "
            f"{html.escape(str(ranking['n_common_cells']))} common emitted cells."
            if isinstance(ranking, dict)
            else "No rank was published; every candidate must have a valid common-cell score."
        )
    columns = (
        ("<th>WRMSSE</th><th>Scored cells</th>" if flat
         else "<th>WRMSSE</th><th>Series by level</th>")
        if is_wrmsse
        else "<th>RMSE</th><th>MAE</th><th>WAPE</th><th>Scored cells</th>"
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Run candidate comparison</title>"
        "<style>body{font:14px system-ui;margin:32px;color:#172033}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d7dce5;padding:8px;text-align:left}"
        "th{background:#f4f7fb}</style>"
        "<h1>Candidate comparison</h1>"
        "<p>Comparison scope: "
        + note
        + "</p><table><tr><th>Model</th><th>Status</th>"
        + columns
        + "<th>Detail</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )


def compare_run_forecasts(
    panel: Panel,
    forecasts: Mapping[str, Forecast],
    spark: SparkSession,
    *,
    missing_actual: MissingActual,
    metric: Metric,
    output_dir: str | pathlib.Path,
    fixed_contract: Mapping[str, object] | None = None,
) -> pathlib.Path:
    """Score one request's forecasts on the exact common emitted cells and publish evidence."""
    if metric not in ("rmse", "mae", "wape", "wrmsse"):
        raise EvaluationError(f"unsupported run comparison metric: {metric!r}")
    if not forecasts:
        raise EvaluationError("run comparison requires at least one completed forecast")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = sorted(forecasts.items())
    restrict_to: str | None = None
    common_count: int | None = None
    comparison_error: str | None = None
    grid_path = None
    study = (fixed_contract is not None
             and fixed_contract.get("schema") == "autoforecast.fixed-evaluation.v2")
    if study:
        assert fixed_contract is not None
        from autoforecast.study_scoring import verify_study_evaluation
        verify_study_evaluation(dict(fixed_contract))
        policy = cast(dict, fixed_contract["metric_policy"])
        missing_actual = "sparse_zero" if policy["actual_missing"] == "zero" else "no_data"
        if policy["metric"] != metric:
            raise EvaluationError("metric differs from the frozen study")
    if fixed_contract is not None:
        if not study and (
            fixed_contract.get("schema") != "autoforecast.fixed-evaluation.v1"
            or fixed_contract.get("metric_policy")
            != {"metric": metric, "missing_actual": "error", "fallback": "none"}
            or missing_actual != "no_data"
            or metric == "wrmsse"
        ):
            raise EvaluationError("fixed scoring requires the frozen generic metric policy")
        grid_record = cast(dict, fixed_contract["grid"])
        grid_path = grid_record["path"]
        if directory_digest(grid_path, suffixes=(".parquet",)) != grid_record["sha256"]:
            raise EvaluationError("frozen scoring grid changed")
        from autoforecast.grid import content_hash

        frequency = parse_frequency(items[0][1].frequency).code
        if content_hash(grid_path, spark, frequency=frequency) != grid_record["logical_sha256"]:
            raise EvaluationError("frozen scoring grid logical identity changed")
        origin = items[0][1].origin
        forecast = items[0][1]
        expected_suffix = f"/{origin}/{forecast.horizon}/{frequency}"
        if not str(fixed_contract["boundary"]).endswith(expected_suffix):
            raise EvaluationError("forecast uses the wrong evaluation boundary")
        try:
            _, common_count = _exact_grid_restrictions(
                {name: {origin: f} for name, f in items}, (origin,), {origin: grid_path}, spark
            )
        except SelectionError as error:
            raise EvaluationError(str(error)) from error
        if common_count != grid_record["n_cells"]:
            raise EvaluationError("fixed grid denominator changed")
    if fixed_contract is None and len(items) > 1 and metric != "wrmsse":
        try:
            restrict_to, common_count = common_cells(
                [forecast for _, forecast in items], spark, out / "common-cells"
            )
        except SelectionError as error:
            comparison_error = str(error)
    candidates: list[dict[str, object]] = []
    scored: list[tuple[str, Score | Wrmsse | dict]] = []
    for model_id, forecast in items:
        try:
            score: Score | Wrmsse | dict
            if study:
                assert fixed_contract is not None
                from autoforecast.study_scoring import score_study
                score = score_study(
                    panel, forecast, spark, contract=dict(fixed_contract),
                    output_dir=out / f"candidate-{len(candidates):04d}",
                )
            elif metric == "wrmsse":
                # This is explicitly the M5 convention: the metric owns M5's 12 hierarchy
                # levels and dollar-sales weight column, and refuses a panel that lacks either.
                # Unlike a common-cell point comparison, every candidate must cover the entire
                # published series-by-horizon grid or it is not comparable.
                score = score_wrmsse(panel, forecast, spark, levels=M5_LEVELS, weight_col="revenue")
            else:
                score = score_forecast(
                    panel,
                    forecast,
                    spark,
                    missing_actual=missing_actual,
                    restrict_to=restrict_to,
                    eligible_grid=grid_path,
                )
                if fixed_contract is not None and (
                    score.n_scored != common_count or score.n_zero_filled
                ):
                    raise ScoreError("missing or ineligible actual blocks fixed score")
        except (ScoreError, WrmsseError) as error:
            candidates.append({"model_id": model_id, "status": "not_scored", "detail": str(error)})
        else:
            scored.append((model_id, score))
            candidates.append(
                {
                    "model_id": model_id,
                    "status": ("unavailable"
                               if study and _value(score, metric) is None else "scored"),
                    "detail": "" if not study or _value(score, metric) is not None else
                        "The frozen policy has no defined aggregate for this coverage or scale.",
                    "score": _score_record(score),
                }
            )
    ranking: dict[str, object] | None = None
    if (comparison_error is None and len(scored) == len(items)
            and (not study or all(_value(score, metric) is not None for _, score in scored))):
        ordered = sorted(
            scored,
            key=lambda pair: (
                float("inf") if _value(pair[1], metric) is None else _value(pair[1], metric),
                pair[0],
            ),
        )
        if study:
            n_common_cells = common_count
        elif metric == "wrmsse":
            n_common_cells = items[0][1].n_series * items[0][1].horizon
        else:
            first_score = scored[0][1]
            if not isinstance(first_score, Score):
                raise EvaluationError("cell comparison requires a cell score")
            n_common_cells = common_count if common_count is not None else first_score.n_scored
        ranking = {
            "metric": metric,
            "n_common_cells": n_common_cells,
            "order": [model_id for model_id, _score in ordered],
        }
    record: dict[str, object] = {
        "schema": "autoforecast.run-candidate-comparison.v1",
        "comparison_kind": (
            "fixed_benchmark"
            if fixed_contract is not None
            else "m5_full_grid_wrmsse"
            if metric == "wrmsse"
            else "exploratory_common_cells"
        ),
        "metric": metric,
        "missing_actual": missing_actual,
        "candidates": candidates,
        "ranking": ranking,
        "comparison_error": comparison_error,
        **({"evaluation_contract": dict(fixed_contract)} if fixed_contract is not None else {}),
    }
    if (
        fixed_contract is not None
        and directory_digest(cast(str, grid_path), suffixes=(".parquet",))
        != cast(dict, fixed_contract["grid"])["sha256"]
    ):
        raise EvaluationError("scoring grid changed during evaluation")
    json_path = out / "candidate-comparison.json"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_html(out / "candidate-comparison.html", record)
    return json_path
