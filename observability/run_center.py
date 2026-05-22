"""Publish one static Run Center from explicit verified-report paths.

This is presentation integration, not forecasting logic: AutoForecast writes immutable evidence;
the caller names the reports to link. The publisher never scans the repository or recomputes a
result.
"""

from __future__ import annotations

import argparse
import html
import os
import pathlib
from collections.abc import Iterable

__all__ = ["RunCenterError", "build_run_center", "main"]


class RunCenterError(ValueError):
    """A requested static Run Center link is missing or unsafe."""


def _links(
    paths: Iterable[str | pathlib.Path], *, output_parent: pathlib.Path, label: str,
) -> str:
    rows: list[str] = []
    for raw in paths:
        path = pathlib.Path(raw)
        if not path.is_file():
            raise RunCenterError(f"{label} report must be an existing file: {path}")
        relative = os.path.relpath(path.resolve(), start=output_parent.resolve())
        href = pathlib.PurePosixPath(relative).as_posix()
        rows.append(
            f"<li><a href='{html.escape(href, quote=True)}'>"
            f"{html.escape(path.parent.name)}</a></li>",
        )
    return "".join(rows) or "<li class='muted'>No published report.</li>"


def build_run_center(
    output_path: str | pathlib.Path,
    *,
    data_reports: Iterable[str | pathlib.Path] = (),
    study_reports: Iterable[str | pathlib.Path] = (),
    benchmark_reports: Iterable[str | pathlib.Path] = (),
    model_catalogues: Iterable[str | pathlib.Path] = (),
    tensorboard_url: str | None = None,
    mlflow_url: str | None = None,
) -> pathlib.Path:
    """Write a small static landing page with explicit report links."""
    if tensorboard_url and not tensorboard_url.startswith(("http://", "https://")):
        raise RunCenterError("tensorboard_url must be an HTTP(S) URL when supplied")
    if mlflow_url and not mlflow_url.startswith(("http://", "https://")):
        raise RunCenterError("mlflow_url must be an HTTP(S) URL when supplied")
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = _links(data_reports, output_parent=output.parent, label="data")
    studies = _links(study_reports, output_parent=output.parent, label="study")
    benchmarks = _links(benchmark_reports, output_parent=output.parent, label="benchmark")
    catalogues = _links(model_catalogues, output_parent=output.parent, label="model catalogue")
    trace = (
        f"<a class='button' href='{html.escape(tensorboard_url, quote=True)}' "
        "target='_blank' rel='noopener noreferrer'>Open TensorBoard</a>"
        if tensorboard_url else
        "<span class='muted'>TensorBoard endpoint not configured.</span>"
    )
    tracking = (
        f"<a class='button' href='{html.escape(mlflow_url, quote=True)}' "
        "target='_blank' rel='noopener noreferrer'>Open MLflow</a>"
        if mlflow_url else
        "<span class='muted'>MLflow endpoint not configured.</span>"
    )
    document = f"""<!doctype html>
<html lang='en'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>AutoForecast Run Center</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f6f8fc;color:#172033;
font:15px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1100px;margin:auto;padding:42px 24px}}
h1{{font-size:34px;letter-spacing:-.04em;margin:0}}.intro{{color:#58657a;max-width:720px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:28px}}
section{{background:#fff;border:1px solid #e1e7f0;border-radius:14px;padding:20px;
box-shadow:0 4px 18px #243b6b0a}}
h2{{font-size:18px;margin:0 0 6px}}p{{margin:0 0 12px;color:#58657a}}
ul{{margin:0;padding-left:20px}}a{{color:#2452c5}}.button{{display:inline-block;background:#2452c5;
color:#fff;border-radius:7px;padding:8px 12px;text-decoration:none;font-weight:650;font-size:13px}}
.muted{{color:#718096}}footer{{color:#7b8798;font-size:12px;margin-top:24px}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style>
<main><p class='muted'>AutoForecast · Run Center</p><h1>Verified forecasting evidence</h1>
<p class='intro'>Open data checks, validation-only studies, held-out benchmarks, and model
requirements. This page links completed evidence; it does not rank models or start services.</p>
<div class='grid'>
<section><h2>Data checks</h2><p>Generic data checks and model-readiness reports.</p>
<ul>{data}</ul></section>
<section><h2>Studies</h2><p>Validation-only trial comparison and sealed top-k selection.</p>
<ul>{studies}</ul></section>
<section><h2>Benchmarks</h2><p>Held-out results under one frozen comparison contract.</p>
<ul>{benchmarks}</ul></section>
<section><h2>Model catalogue</h2><p>Model capabilities and data requirements; not a ranking.</p>
<ul>{catalogues}</ul></section>
</div><section style='margin-top:16px'><h2>Training traces</h2>
<p>Per-attempt TensorBoard traces supplement a study; they do not determine eligibility.</p>
{trace}</section>
<section style='margin-top:16px'><h2>Experiment tracking</h2>
<p>Completed-run parameters and summary metrics are available in MLflow when configured.</p>
{tracking}</section>
<footer>Generated from explicit report paths. Technical incidents remain separate from data-quality
and model-readiness outcomes.</footer></main></html>"""
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    """Publish a Run Center without a web framework or repository scan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-report", action="append", default=[])
    parser.add_argument("--study-report", action="append", default=[])
    parser.add_argument("--benchmark-report", action="append", default=[])
    parser.add_argument("--model-catalogue", action="append", default=[])
    parser.add_argument("--tensorboard-url")
    parser.add_argument("--mlflow-url")
    args = parser.parse_args()
    output = build_run_center(
        args.output,
        data_reports=args.data_report,
        study_reports=args.study_report,
        benchmark_reports=args.benchmark_report,
        model_catalogues=args.model_catalogue,
        tensorboard_url=args.tensorboard_url,
        mlflow_url=args.mlflow_url,
    )
    print(output)


if __name__ == "__main__":
    main()
