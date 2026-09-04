"""Render a compact static comparison page from completed local attempts."""

from __future__ import annotations

import html
import json
import pathlib
from collections.abc import Iterable
from typing import Any

from mlops import sanitize_web_url
from mlops._safe_files import is_regular_file
from mlops.run_report import run_guide

from autoforecast._exact_json import load_exact_json
from autoforecast.experiments import ExperimentError, load_completed_attempt, load_study_selection

__all__ = ["render_benchmark_report", "render_study_report"]


def _epochs(path: pathlib.Path) -> list[dict[str, Any]]:
    metrics = path / "metrics" / "epochs.jsonl"
    if not metrics.exists():
        return []
    return [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line]


def _line(points: Iterable[float], *, low: float, high: float, width: int, height: int) -> str:
    values = list(points)
    if not values:
        return ""
    denominator = max(high - low, 1e-12)
    return " ".join(
        f"{index * width / max(len(values) - 1, 1):.1f},"
        f"{height - (value - low) / denominator * height:.1f}"
        for index, value in enumerate(values)
    )


def _optimization_validation(record: dict[str, Any]) -> float | None:
    """Read the optimization metric recorded by the current training loop."""
    value = record.get("optimization/validation_objective")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _mlflow_run_link(attempt: pathlib.Path, fallback_url: str | None) -> str | None:
    """Return the precise MLflow run page when this attempt was live-tracked."""
    safe_url = sanitize_web_url(fallback_url)
    tracking = attempt / "tracking" / "mlflow.json"
    if tracking.is_file():
        try:
            record = load_exact_json(tracking)
        except ValueError:
            record = None
        if isinstance(record, dict):
            safe_url = (
                sanitize_web_url(record.get("tracking_web_url"))
                or sanitize_web_url(record.get("tracking_uri"))
                or safe_url
            )
            experiment_id, run_id = record.get("experiment_id"), record.get("run_id")
            if (
                safe_url is not None
                and isinstance(experiment_id, str)
                and isinstance(run_id, str)
            ):
                return f"{safe_url}/#/experiments/{experiment_id}/runs/{run_id}"
    return safe_url


def render_study_report(
    study_root: str | pathlib.Path, *, mlflow_url: str | None = None,
) -> pathlib.Path:
    """Render one validation study; held-out results belong to a separate benchmark report."""
    root = pathlib.Path(study_root)
    attempts = sorted(root.glob("trials/*/attempts/*"))
    rows: list[tuple[pathlib.Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for attempt in attempts:
        if not (attempt / "completion.json").exists():
            continue
        metadata, completion = load_completed_attempt(attempt)
        if completion.get("status") == "completed":
            rows.append((attempt, metadata, completion, _epochs(attempt)))
    if not rows:
        raise ExperimentError(f"no completed attempts in {root}")
    selection = load_study_selection(root) if is_regular_file(root / "selection.json") else None
    fingerprints = {
        str(metadata.get("comparability_fingerprint", ""))
        for _, metadata, _, _ in rows
    }
    comparable = len(fingerprints) == 1 and "" not in fingerprints
    curves = [
        [value for record in epochs if (value := _optimization_validation(record)) is not None]
        for _, _, _, epochs in rows
    ]
    values = [value for curve in curves for value in curve]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    colours = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706")
    svg = "".join(
        f'<polyline fill="none" stroke="{colours[index % len(colours)]}" stroke-width="2" '
        f'points="{_line(curve, low=low, high=high, width=720, height=220)}" />'
        for index, curve in enumerate(curves)
    ) if comparable and values else ""
    selected = {
        (str(candidate["trial_id"]), str(candidate["attempt_id"])): int(candidate["rank"])
        for candidate in selection["selected"]
    } if selection is not None else {}
    table_rows = []
    for index, (attempt, metadata, _completion, epochs) in enumerate(rows):
        final = epochs[-1] if epochs else {}
        final_validation = _optimization_validation(final)
        params = metadata.get("hyperparameters", {})
        label = f"{metadata['trial_id']} / {metadata['attempt_id']}"
        swatch = colours[index % len(colours)]
        trace = html.escape(attempt.relative_to(root).as_posix())
        selection_status = (
            f"Selected (rank {selected[(str(metadata['trial_id']), str(metadata['attempt_id']))]})"
            if (str(metadata["trial_id"]), str(metadata["attempt_id"])) in selected
            else "Not selected" if selection is not None else "Selection not sealed"
        )
        run_url = _mlflow_run_link(attempt, mlflow_url)
        trace_view = (
            f"<a href='{html.escape(run_url, quote=True)}' target='_blank' "
            "rel='noopener noreferrer'>Open MLflow run</a>"
            if run_url else "MLflow not configured"
        )
        has_checkpoint = any(
            path.is_file() for directory in (attempt / "model", attempt / "checkpoints")
            if directory.is_dir() for path in directory.rglob("*")
        )
        model_status = (
            "Checkpoint captured; not registered until held-out approval"
            if has_checkpoint else "No checkpoint captured"
        )
        epoch_seconds = sum(float(epoch.get("execution/epoch_seconds", 0))
                            for epoch in epochs)
        peak_rss = max(
            (int(epoch.get("execution/process_peak_rss_bytes", 0))
             for epoch in epochs),
            default=0,
        )
        table_rows.append(
            "<tr>"
            f"<td><span class='legend' style='background:{swatch}'></span>{html.escape(label)}</td>"
            f"<td><code>{html.escape(json.dumps(params, sort_keys=True))}</code></td>"
            f"<td>{len(epochs)}</td>"
            f"<td>{final_validation if final_validation is not None else '—'}</td>"
            f"<td>{epoch_seconds:.3f}</td>"
            f"<td>{peak_rss}</td>"
            f"<td><code>{html.escape(str(metadata.get('data', {})))}</code></td>"
            f"<td>{html.escape(selection_status)}</td>"
            f"<td>{html.escape(model_status)}</td>"
            f"<td>{trace_view}<br><code>{trace}</code></td>"
            "</tr>"
        )
    note = ("Validation objectives are overlaid because every completed attempt carries the same "
            "comparability fingerprint." if comparable else
            "No overlay or ranking: completed attempts have different or missing comparability "
            "fingerprints.")
    selection_note = (
        f"Selection is sealed: top_k={selection['top_k']}; held-out evaluation is permitted only "
        "for selected attempts." if selection is not None else
        "Selection is not sealed: this page is an unranked validation report."
    )
    document = f"""<!doctype html>
<html lang='en'><meta charset='utf-8'><title>Study comparison</title>
<style>
body{{font:14px system-ui;margin:32px;color:#172033}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d7dce5;padding:8px;text-align:left;vertical-align:top}}
th{{background:#f4f7fb}}
code{{font-size:11px;white-space:pre-wrap}}.legend{{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:50%}}
svg{{border:1px solid #d7dce5;background:#fff;max-width:100%}}
</style>
<h1>Study comparison: {html.escape(root.name)}</h1>
<p>{html.escape(note)} Test metrics are intentionally absent: test data is never used for trial
selection.</p>
<p>{html.escape(selection_note)}</p>
<p>Validation objective (lower is better; objective units only—this is not WRMSSE):</p>
<svg viewBox='0 0 720 220' role='img' aria-label='validation objective by epoch'>{svg}</svg>
<table><thead><tr><th>trial / attempt</th><th>hyperparameters</th><th>epochs</th>
<th>final internal validation objective</th><th>epoch time (s)</th><th>peak RSS (bytes)</th>
<th>data contract</th><th>selection</th><th>model artifact</th><th>observability</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
</html>"""
    report = root / "report.html"
    report.write_text(document, encoding="utf-8")
    from autoforecast.protocol import _atomic_write_text

    _atomic_write_text(root / "RUN.md", run_guide(
        identity={"search_id": root.name}, status="selected" if selection else "attempts recorded",
        files=[p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()],
        sealed=False,
    ).decode("utf-8"))
    return report


def _heldout_rows(study_root: pathlib.Path) -> tuple[str, str, list[dict[str, Any]]]:
    """Read only completed selected held-out bundles from one explicitly named study."""
    selection = load_study_selection(study_root)
    selected = {
        (str(candidate["trial_id"]), str(candidate["attempt_id"])): candidate
        for candidate in selection["selected"]
        if type(candidate) is dict
    }
    rows: list[dict[str, Any]] = []
    for attempt in sorted(study_root.glob("heldout/trials/*/attempts/*")):
        if not (attempt / "completion.json").is_file():
            continue
        metadata, completion = load_completed_attempt(attempt)
        if completion.get("status") != "completed":
            continue
        result = load_exact_json(attempt / "evaluation" / "heldout.json")
        if (not isinstance(result, dict)
                or result.get("schema") != "autoforecast.heldout-evaluation.v1"):
            raise ExperimentError(f"invalid held-out result: {attempt}")
        value = result.get("value")
        source = result.get("selected_source")
        if (not isinstance(value, int | float) or isinstance(value, bool)
                or not isinstance(source, dict)):
            raise ExperimentError(f"invalid held-out value or source identity: {attempt}")
        source_trial_id, source_attempt_id = source.get("trial_id"), source.get("attempt_id")
        if not isinstance(source_trial_id, str) or not isinstance(source_attempt_id, str):
            raise ExperimentError(f"held-out result has an invalid selected identity: {attempt}")
        key = (source_trial_id, source_attempt_id)
        if key not in selected:
            raise ExperimentError(
                f"held-out result did not originate from a selected candidate: {attempt}")
        candidate = selected[key]
        if (metadata.get("comparability_fingerprint") != selection["comparability_fingerprint"]
                or source.get("artifact_manifest_sha256")
                != candidate.get("artifact_manifest_sha256")):
            raise ExperimentError(f"held-out result does not match its sealed selection: {attempt}")
        data = metadata.get("data")
        if not isinstance(data, dict) or data.get("test") != "evaluated":
            raise ExperimentError(f"held-out result has no evaluated test contract: {attempt}")
        model = metadata.get("model")
        model_id = model.get("id") if isinstance(model, dict) else None
        rows.append({
            "study_id": study_root.name,
            "model_id": model_id if isinstance(model_id, str) else "unknown",
            "trial_id": source_trial_id,
            "attempt_id": source_attempt_id,
            "heldout_objective": float(value),
            "metrics": result.get("metrics", {}),
            "path": attempt.as_posix(),
        })
    return str(selection["comparability_fingerprint"]), str(selection["direction"]), rows


def render_benchmark_report(
    study_roots: Iterable[str | pathlib.Path],
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """Render one held-out benchmark for explicitly supplied compatible studies.

    The caller names the studies to compare. This prevents a report generator from discovering
    unrelated work and turning it into a misleading global leaderboard.
    """
    studies = tuple(pathlib.Path(root) for root in study_roots)
    if not studies:
        raise ExperimentError("benchmark needs at least one explicit study root")
    contracts = [_heldout_rows(study) for study in studies]
    fingerprints = {contract[0] for contract in contracts}
    directions = {contract[1] for contract in contracts}
    if len(fingerprints) != 1 or len(directions) != 1:
        raise ExperimentError("benchmark studies have incompatible comparison contracts")
    rows = [row for _fingerprint, _direction, values in contracts for row in values]
    if not rows:
        raise ExperimentError("benchmark has no completed selected held-out evaluations")
    direction = next(iter(directions))
    rows.sort(key=lambda row: (
        float(row["heldout_objective"]) * (1 if direction == "minimize" else -1),
        str(row["study_id"]),
        str(row["trial_id"]),
        str(row["attempt_id"]),
    ))
    last_value: float | None = None
    rank = 0
    for index, row in enumerate(rows, start=1):
        value = float(row["heldout_objective"])
        if last_value is None or value != last_value:
            rank = index
            last_value = value
        row["rank"] = rank
    table = "".join(
        "<tr>"
        f"<td>{row['rank']}</td><td>{html.escape(str(row['model_id']))}</td>"
        f"<td>{html.escape(str(row['study_id']))}</td>"
        f"<td>{html.escape(str(row['trial_id']))} / {html.escape(str(row['attempt_id']))}</td>"
        f"<td>{row['heldout_objective']}</td>"
        f"<td><code>{html.escape(json.dumps(row['metrics'], sort_keys=True))}</code></td>"
        "</tr>"
        for row in rows
    )
    document = f"""<!doctype html>
<html lang='en'><meta charset='utf-8'><title>AutoForecast benchmark</title>
<style>
body{{font:14px system-ui;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d7dce5;padding:8px;text-align:left;vertical-align:top}}
th{{background:#f4f7fb}}code{{font-size:11px;white-space:pre-wrap}}
</style>
<h1>Held-out benchmark</h1>
<p>Comparison contract: <code>{html.escape(next(iter(fingerprints)))}</code></p>
<p>Only sealed selected candidates with completed held-out evidence are ranked.
Direction: {html.escape(direction)}. Validation metrics and smoke evidence are absent.</p>
<table><thead><tr><th>rank</th><th>model</th><th>study</th><th>selected candidate</th>
<th>held-out objective</th><th>held-out metrics</th></tr></thead>
<tbody>{table}</tbody></table></html>"""
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
