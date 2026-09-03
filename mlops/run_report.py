"""Small file guides for existing artifact stores; no model or storage dependency."""

from __future__ import annotations

import html
import os
import pathlib
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from urllib.parse import quote

from mlops._safe_files import _open_directory_fd, _read_regular_bytes_at


def _discard_unsealed_guide(root: pathlib.Path, payload: bytes, error: Exception) -> None:
    """Discard only this invocation's guide, before terminal evidence is published."""
    try:
        descriptor = _open_directory_fd(root)
        try:
            for name in ("artifact_manifest.json", "completion.json"):
                try:
                    os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                return
            if _read_regular_bytes_at(descriptor, "RUN.md") == payload:
                os.unlink("RUN.md", dir_fd=descriptor)
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception as cleanup_error:
        error.add_note(f"Generated guide cleanup failed: {cleanup_error}")


def new_run_id(prefix: str) -> str:
    """Return a readable UTC execution ID; exclusive creation enforces uniqueness."""
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in prefix):
        raise ValueError("run prefix must use lowercase letters, digits, _ or -")
    return f"{prefix}-{datetime.now(UTC):%Y%m%dt%H%M%Sz}-{uuid.uuid4().hex}"


def _purpose(name: str) -> str:
    path = pathlib.PurePosixPath(name)
    parts = path.parts
    owners = {
        "model": "Native saved model; load with the producing model's framework.",
        "forecasts": "Forecast dataset; read the enclosing directory as Parquet.",
        "scores": "Score or coverage dataset; read Parquet parts together.",
        "artifacts": "Additional file explicitly retained by the model callback.",
    }
    for part in parts[:-1]:
        if part in owners:
            return owners[part]
    if path.name == "RUN.md":
        return "File guide for this retained execution."
    if path.name == "artifact_manifest.json":
        return "Exact file inventory, byte sizes and SHA-256 checksums."
    if path.name == "completion.json":
        return "Terminal execution status and manifest identity."
    if path.name == "predictions.parquet":
        return "Analysis forecast rows with model, run, stage, origin and target-date identity."
    if path.name == "series-scores.parquet":
        return "Per-series scores and coverage; unavailable metrics remain null."
    if path.name == "source-keys.parquet":
        return "Typed source keys mapped to the canonical series ID."
    if path.name in {"failure.json", "error.json"}:
        return "Failure details; missing outputs were not fabricated."
    if path.name == "winner-config.json":
        return "Effective selected or fixed parameters and their provenance."
    if path.name == "selection.json":
        return "Trial ranking and selected attempts; no held-out selection."
    if path.name == "result.json" or "evaluation" in parts:
        return "Evaluation results, boundaries, metrics and artifact references."
    if "provenance" in path.name or "preflight" in path.name:
        return "Input identities, eligibility checks or configuration provenance."
    if path.suffix in {".yaml", ".yml"}:
        return "Retained configuration; inspect the file for its scope."
    if "metrics" in parts or "tensorboard" in parts or path.suffix == ".jsonl":
        return "Recorded training measurements or event log."
    if path.name == "report.html":
        return "Human-readable comparison report."
    if path.suffix == ".parquet":
        return "Typed table for analysis."
    if path.suffix == ".json":
        return "Machine-readable execution metadata."
    return "Retained file; its format and meaning are defined by the producer."


def run_guide(
    *, identity: Mapping[str, str], status: str, files: Iterable[str], sealed: bool = True,
) -> bytes:
    """Describe actual files without reading models or interpreting arbitrary JSON."""
    lines = ["# Saved run outputs", "", f"Execution status: **{html.escape(status)}**.", ""]
    for key, value in identity.items():
        lines.append(f"- {html.escape(key)}: {html.escape(value)}")
    lines += [
        "",
        "Execution success does not imply that every evaluation has actuals or a valid score. "
        "Read the evaluation status, coverage and missing-value fields in the result records.",
        "",
        "This folder retains produced output files. Input datasets and earlier searches may "
        "be referenced in provenance or result records. Keep those referenced directories "
        "for later verification. Temporary training slices and undeclared scratch files "
        "are not retained here. Models keep their native format; this guide does not load them.",
        "",
        ("This guide is included in the sealed artifact manifest. Use the library's verified "
         "loader before analysis. Do not append files to a completed bundle; write derived "
         "exports to a separate directory." if sealed else
         "This guide is a snapshot of a coordinator or derived export. The source journals "
         "and sealed child bundles remain authoritative; this guide does not seal them."),
        "", "| File | Purpose |", "|---|---|",
    ]
    for name in sorted(set(files)):
        label = html.escape(name).replace("|", "&#124;").replace("\n", "&#10;")
        link = f'<a href="{quote(name, safe="/")}">{label}</a>'
        lines.append(f"| {link} | {_purpose(name)} |")
    lines += ["", "Parquet tables are readable with Spark or PyArrow. JSON records contain "
              "metadata and metrics. YAML records source or effective configurations; "
              "inspect their scope. Filenames "
              "ending in `.crc` and `_SUCCESS` are storage-engine bookkeeping.", ""]
    return "\n".join(lines).encode("utf-8")
