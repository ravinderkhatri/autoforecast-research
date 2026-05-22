"""Build one static navigation page for explicitly configured completed report artifacts.

The hub is intentionally downstream-only: forecasting code emits its own report artifacts, while
this module only indexes those artifacts for a chosen publication environment.  It never scans the
whole repository or starts TensorBoard/TFDV services.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import pathlib
from dataclasses import dataclass
from urllib.parse import quote, urlparse

__all__ = ["HubError", "build_hub", "main"]


class HubError(ValueError):
    """The explicit hub configuration or publication destination is unsafe."""


@dataclass(frozen=True)
class _Source:
    id: str
    evidence_class: str
    report_glob: str
    tensorboard: bool


@dataclass(frozen=True)
class _Entry:
    source: _Source
    report: pathlib.Path
    relative_report: pathlib.Path
    status: str
    fingerprint: str
    as_of: str
    data_checks: str


def _load_sources(path: pathlib.Path) -> tuple[_Source, ...]:
    if path.is_symlink() or not path.is_file():
        raise HubError(f"sources configuration must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HubError(f"invalid JSON in sources configuration: {exc}") from exc
    if type(value) is not dict or set(value) != {"schema_version", "sources"}:
        raise HubError("sources configuration requires exactly schema_version and sources")
    if value["schema_version"] != 1 or type(value["sources"]) is not list:
        raise HubError("sources configuration requires schema_version: 1 and a sources list")
    sources: list[_Source] = []
    identifiers: set[str] = set()
    for item in value["sources"]:
        if type(item) is not dict or set(item) != {
            "id",
            "evidence_class",
            "report_glob",
            "tensorboard",
        }:
            raise HubError(
                "each source requires exactly id, evidence_class, report_glob, tensorboard",
            )
        source = _Source(**item)
        if any(
            type(field) is not str or not field.strip()
            for field in (source.id, source.evidence_class, source.report_glob)
        ):
            raise HubError("source id, evidence_class, and report_glob must be non-blank strings")
        if type(source.tensorboard) is not bool:
            raise HubError("source tensorboard must be a boolean")
        parts = pathlib.PurePosixPath(source.report_glob).parts
        if pathlib.PurePosixPath(source.report_glob).is_absolute() or ".." in parts:
            raise HubError(
                "source report_glob must remain below repository root: " f"{source.report_glob!r}",
            )
        if source.id in identifiers:
            raise HubError(f"duplicate source id: {source.id!r}")
        identifiers.add(source.id)
        sources.append(source)
    return tuple(sources)


def _metadata(report: pathlib.Path) -> tuple[str, str, str, str]:
    """Read the small standardized data-preflight sidecar when it exists.

    Other report types retain ``report available`` rather than inferring a stronger completion or
    acceptance claim from HTML presentation markup.
    """
    sidecar = report.with_name("data_validation_report.json")
    if not sidecar.is_file():
        return "report available", "—", "—", "—"
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "report available", "—", "—", "—"
    if type(value) is not dict:
        return "report available", "—", "—", "—"
    # Schema v1 represented a failed Spark read as a failed data check.  That is
    # not useful to a person deciding whether data passed quality checks: no
    # measurements were made.  Keep the detailed incident in its own report,
    # but say plainly here that checks did not run.
    if value.get("schema_version") == 1 and value.get("profile") is None:
        legacy_as_of = value.get("as_of")
        return (
            "data not checked",
            "—",
            legacy_as_of if type(legacy_as_of) is str else "—",
            "Data could not be inspected. Data checks were not run.",
        )
    if value.get("schema_version") == 2 and value.get("kind") == "run_incident":
        incident = value.get("incident")
        summary = incident.get("summary") if type(incident) is dict else None
        return (
            "data not checked",
            "—",
            "—",
            summary if type(summary) is str and summary else "Data checks were not run.",
        )
    if value.get("schema_version") == 2 and value.get("kind") == "data_checks":
        facts = value.get("facts")
        checks = value.get("checks")
        accepted = value.get("accepted")
        fingerprint = facts.get("data_fingerprint") if type(facts) is dict else None
        as_of = facts.get("as_of") if type(facts) is dict else None
        results = checks if type(checks) is list else []
        failed = sum(type(check) is dict and check.get("status") == "failed" for check in results)
        warning = sum(type(check) is dict and check.get("status") == "warning" for check in results)
        finding = next(
            (
                check.get("summary")
                for check in results
                if type(check) is dict
                and check.get("status") in {"failed", "warning"}
                and type(check.get("summary")) is str
            ),
            "",
        )
        data_checks = f"{failed} failed · {warning} warning(s)"
        if finding:
            data_checks = f"{data_checks} — {finding}"
        return (
            "data checks passed" if accepted is True else "data checks failed",
            fingerprint if type(fingerprint) is str and fingerprint else "—",
            as_of if type(as_of) is str and as_of else "—",
            data_checks,
        )
    accepted = value.get("accepted")
    status = (
        "data accepted"
        if accepted is True
        else "data rejected"
        if accepted is False
        else "report available"
    )
    fingerprint = value.get("data_fingerprint")
    as_of = value.get("as_of")
    blockers = value.get("blockers")
    warnings = value.get("warnings")
    blocker_count = len(blockers) if type(blockers) is list else 0
    warning_count = len(warnings) if type(warnings) is list else 0
    finding = next(
        (
            item
            for collection in (blockers, warnings)
            if type(collection) is list
            for item in collection
            if type(item) is str and item
        ),
        "",
    )
    data_checks = f"{blocker_count} blocker(s) · {warning_count} warning(s)"
    if finding:
        data_checks = f"{data_checks} — {finding}"
    return (
        status,
        fingerprint if type(fingerprint) is str and fingerprint else "—",
        as_of if type(as_of) is str and as_of else "—",
        data_checks,
    )


def _entries(sources: tuple[_Source, ...], repository_root: pathlib.Path) -> tuple[_Entry, ...]:
    entries: list[_Entry] = []
    for source in sources:
        for report in sorted(repository_root.glob(source.report_glob)):
            if not report.is_file():
                continue
            resolved = report.resolve()
            if not resolved.is_relative_to(repository_root):
                raise HubError(f"configured report escaped repository root: {report}")
            status, fingerprint, as_of, data_checks = _metadata(resolved)
            entries.append(
                _Entry(
                    source=source,
                    report=resolved,
                    relative_report=resolved.relative_to(repository_root),
                    status=status,
                    fingerprint=fingerprint,
                    as_of=as_of,
                    data_checks=data_checks,
                )
            )
    return tuple(entries)


def _latest_per_data_snapshot(entries: tuple[_Entry, ...]) -> tuple[_Entry, ...]:
    """Keep one current result per measured snapshot for landing-page summaries.

    Immutable reports may be re-rendered as the presentation improves.  A second HTML version
    must not make the same fingerprint look like a second successful data check.
    """
    latest: dict[tuple[str, str, str], _Entry] = {}
    for entry in entries:
        key = (entry.source.id, entry.fingerprint, entry.as_of)
        previous = latest.get(key)
        is_newer = previous is None or (
            entry.report.stat().st_mtime_ns >= previous.report.stat().st_mtime_ns
        )
        if is_newer:
            latest[key] = entry
    return tuple(latest.values())


def _public_url(relative: pathlib.Path, public_base_url: str) -> str:
    """Map a repository-relative report path to an explicitly configured published URL."""
    encoded = quote(relative.as_posix(), safe="/")
    if not public_base_url:
        return encoded
    parsed = urlparse(public_base_url)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raise HubError("public_base_url must be empty, root-relative, or an HTTP(S) URL")
    if parsed.scheme and not parsed.netloc:
        raise HubError("public_base_url HTTP(S) URL needs a host")
    if not parsed.scheme and not public_base_url.startswith("/"):
        raise HubError("public_base_url must be empty, root-relative, or an HTTP(S) URL")
    return f"{public_base_url.rstrip('/')}/{encoded}" if public_base_url != "/" else f"/{encoded}"


def _render(
    entries: tuple[_Entry, ...],
    *,
    public_base_url: str,
    tensorboard_url: str | None,
) -> str:
    rows = []
    for entry in entries:
        report_url = _public_url(entry.relative_report, public_base_url)
        label = f"{entry.source.id} / {entry.report.parent.name}"
        search = " ".join(
            (entry.source.evidence_class, label, entry.status, entry.fingerprint, entry.as_of),
        )
        tensorboard = (
            f"<a href='{html.escape(tensorboard_url, quote=True)}' target='_blank' "
            "rel='noopener noreferrer'>TensorBoard</a>"
            if entry.source.tensorboard and tensorboard_url
            else "not configured"
        )
        rows.append(
            f"<tr data-search='{html.escape(search.lower(), quote=True)}'>"
            f"<td>{html.escape(entry.source.evidence_class)}</td>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(entry.status)}</td>"
            f"<td><code>{html.escape(entry.as_of)}</code></td>"
            f"<td>{html.escape(entry.data_checks)}</td>"
            f"<td><code>{html.escape(entry.fingerprint)}</code></td>"
            f"<td><a href='{html.escape(report_url, quote=True)}'>report</a></td>"
            f"<td>{tensorboard}</td></tr>"
        )
    body = "".join(rows) or "<tr><td colspan='8'>No configured reports found.</td></tr>"
    data_preflights = tuple(entry for entry in entries if entry.source.id == "data-preflight")
    training_entries = tuple(entry for entry in entries if entry.source.tensorboard)
    smoke_entries = tuple(entry for entry in entries if entry.source.id == "model-execution-smoke")
    current_data_preflights = _latest_per_data_snapshot(data_preflights)
    completed_statuses = {
        "data accepted",
        "data rejected",
        "data checks passed",
        "data checks failed",
    }
    completed_data_checks = tuple(
        entry for entry in current_data_preflights if entry.status in completed_statuses
    )
    latest_completed_data_check = max(
        completed_data_checks,
        key=lambda entry: (entry.as_of, entry.relative_report.as_posix()),
        default=None,
    )
    if not data_preflights:
        quality_summary = (
            "No data-preflight artifact has been published. Data quality is unknown; it is not "
            "an implied pass."
        )
    else:
        accepted = sum(
            entry.status in {"data accepted", "data checks passed"}
            for entry in current_data_preflights
        )
        rejected = sum(
            entry.status in {"data rejected", "data checks failed"}
            for entry in current_data_preflights
        )
        not_checked = sum(entry.status == "data not checked" for entry in current_data_preflights)
        if latest_completed_data_check:
            quality_summary = (
                f"Latest completed check: {latest_completed_data_check.status} "
                f"({latest_completed_data_check.report.parent.name}). "
                f"History: {accepted} passed, {rejected} failed, {not_checked} not checked."
            )
        else:
            quality_summary = (
                f"No completed data check yet. {not_checked} published attempt(s) did not run."
            )
    latest_data_report = (
        _public_url(latest_completed_data_check.relative_report, public_base_url)
        if latest_completed_data_check
        else None
    )
    latest_smoke_report = (
        _public_url(smoke_entries[-1].relative_report, public_base_url) if smoke_entries else None
    )
    data_action = (
        f"<a class='button secondary' href='{html.escape(latest_data_report, quote=True)}'>"
        "Open data-check report</a>"
        if latest_data_report
        else "<span class='muted'>No data-check report has been published.</span>"
    )
    training_action = (
        f"<a class='button' href='{html.escape(tensorboard_url, quote=True)}' target='_blank' "
        "rel='noopener noreferrer'>Open TensorBoard</a>"
        if tensorboard_url and training_entries
        else "<span class='muted'>No TensorBoard endpoint is configured.</span>"
    )
    smoke_action = (
        f"<a class='button secondary' href='{html.escape(latest_smoke_report, quote=True)}'>"
        "Open smoke-run report</a>"
        if latest_smoke_report
        else "<span class='muted'>No model execution report has been published.</span>"
    )
    generated = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang='en'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Forecast evidence</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;margin:0;background:#f6f8fc;color:#172033}}
main{{max-width:1240px;margin:auto;padding:42px 28px 56px}}
.eyebrow{{color:#3b5ccc;font-weight:700;font-size:12px;letter-spacing:.08em;
text-transform:uppercase;margin:0 0 8px}}
h1{{font-size:34px;letter-spacing:-.04em;margin:0}}
h2{{font-size:18px;margin:0 0 8px}}
.intro{{max-width:760px;color:#536174;margin:10px 0 30px}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0 34px}}
.card{{background:#fff;border:1px solid #e1e7f0;border-radius:14px;padding:20px;
box-shadow:0 4px 18px #243b6b0a}}
.card p{{color:#536174;margin:0 0 18px;min-height:68px}}
.card .count{{color:#172033;font-weight:750;font-size:24px;margin-bottom:4px}}
.button{{display:inline-block;background:#2452c5;color:#fff;border-radius:7px;padding:8px 12px;
text-decoration:none;font-weight:650;font-size:13px}}
.button.secondary{{background:#edf2ff;color:#2546a5}}
.notice{{background:#eef4ff;border:1px solid #d5e1ff;border-radius:10px;padding:13px 16px;
margin:0 0 24px;color:#35466b}}
.muted{{color:#6b778b}}
.section-heading{{display:flex;align-items:baseline;justify-content:space-between;gap:16px;
margin:34px 0 12px}}
.section-heading p{{color:#687589;margin:0;font-size:13px}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid #e1e7f0;border-radius:12px}}
table{{border-collapse:collapse;width:100%;min-width:900px}}
th,td{{border-bottom:1px solid #e9edf4;padding:11px 12px;text-align:left;vertical-align:top}}
th{{background:#f7f9fd;color:#54627a;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
code{{font-size:11px;word-break:break-all}}
tr:last-child td{{border-bottom:0}}
a{{color:#2452c5}}
input{{border:1px solid #cbd5e1;border-radius:7px;padding:8px 10px;width:24rem;
max-width:100%;font:inherit}}
footer{{color:#7b8798;font-size:12px;margin-top:22px}}
@media (max-width:800px){{
main{{padding:28px 16px}}
h1{{font-size:29px}}
.cards{{grid-template-columns:1fr}}
.card p{{min-height:0}}
.section-heading{{display:block}}
.section-heading p{{margin-top:5px}}
}}
</style>
<main><p class='eyebrow'>AutoForecast · evidence centre</p>
<h1>Data checks, model runs, and training traces</h1>
<p class='intro'>One place to open completed evidence. A data-read problem is shown as
“not checked”; it is never presented as a failed quality check.</p>
<section class='cards' aria-label='Evidence shortcuts'>
<article class='card'><div class='count'>{len(data_preflights)}</div><h2>Data quality</h2>
<p>{html.escape(quality_summary)}</p>{data_action}</article>
<article class='card'><div class='count'>{len(training_entries)}</div><h2>Training graphs</h2>
<p>Per-epoch loss, validation objective, learning rate, and comparable development attempts
live in TensorBoard.</p>{training_action}</article>
<article class='card'><div class='count'>{len(smoke_entries)}</div><h2>Model execution</h2>
<p>Runtime smoke evidence confirms wiring. It is not a benchmark and does not imply model
quality.</p>{smoke_action}</article>
</section>
<aside class='notice'><strong>Model groups:</strong> custom architectures live in
<code>forecast/models/</code>.
AutoGluon zoo is an external-model adapter, not a new in-house model; it stays under
<code>autoforecast/adapters/</code> with its catalog profile under
<code>configs/models/zoo/</code>.</aside>
<section class='section-heading'><div><h2>Published evidence</h2>
<p>Use the report for details; use TensorBoard only for training traces.</p></div>
<label>Filter <input id='filter' type='search'
placeholder='study, fingerprint, evidence class'></label></section>
<div class='table-wrap'><table><thead><tr><th>evidence class</th><th>artifact</th><th>status</th>
<th>as of</th>
<th>data checks</th><th>data fingerprint</th><th>report</th><th>training traces</th></tr></thead>
<tbody>{body}</tbody></table></div>
<footer>Generated {html.escape(generated)}. This page indexes explicitly configured artifacts;
it does not rank models or start services.</footer></main>
<script>
const filter = document.getElementById('filter');
filter.addEventListener('input', () => {{
  const query = filter.value.trim().toLowerCase();
  document.querySelectorAll('tbody tr[data-search]').forEach((row) => {{
    row.hidden = query && !row.dataset.search.includes(query);
  }});
}});
</script></html>
"""


def build_hub(
    sources_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    repository_root: str | pathlib.Path,
    public_base_url: str = "/",
    tensorboard_url: str | None = None,
) -> pathlib.Path:
    """Build a mutable navigation page from an explicit source allowlist.

    The reports it links remain untouched. ``public_base_url`` is supplied by the publication
    environment because a local server, GCS, and reverse proxy do not share a valid URL root.
    """
    root = pathlib.Path(repository_root).resolve()
    if not root.is_dir():
        raise HubError(f"repository_root must be a directory: {root}")
    tensorboard = tensorboard_url.rstrip("/") if tensorboard_url else None
    if tensorboard and urlparse(tensorboard).scheme not in {"http", "https"}:
        raise HubError("tensorboard_url must be an HTTP(S) URL")
    entries = _entries(_load_sources(pathlib.Path(sources_path)), root)
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            _render(entries, public_base_url=public_base_url, tensorboard_url=tensorboard),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> int:
    """Generate an observability index for one explicit local or published artifact root."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=pathlib.Path,
        default=pathlib.Path("observability/sources.json"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(".autoforecast/observability/index.html"),
    )
    parser.add_argument("--repository-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--public-base-url", default="/")
    parser.add_argument("--tensorboard-url")
    args = parser.parse_args()
    output = build_hub(
        args.sources,
        args.output,
        repository_root=args.repository_root,
        public_base_url=args.public_base_url,
        tensorboard_url=args.tensorboard_url,
    )
    print(f"Observability hub: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
