"""Spark-first data checks, profiling facts, and a compact static report.

``Panel.validate`` verifies the declared Parquet schema and ``data_admission`` decides which
models can consume a panel.  This module fills the separate operational boundary: inspect the
actual Spark rows at an explicit as-of date, classify universal data-check failures, and preserve
the resulting facts and snapshot hash for a study. A source-read failure is a ``RunIncident``, not
a data check. It intentionally has no pandas path.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import pathlib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from spec.forecast_spec import STATE_CODES

from autoforecast.model_api import CheckReport, CheckResult, CheckStatus, DataFacts, RunIncident
from autoforecast.panel import Panel, PanelError
from autoforecast.profiling import ProfileError, profile
from provenance.lineage import LineageError, directory_digest

__all__ = [
    "CheckReport",
    "CheckResult",
    "DataCheckError",
    "DataFacts",
    "RunIncident",
    "inspect_panel",
    "run_incident_from_error",
    "write_data_observability_report",
]


class DataCheckError(ValueError):
    """A report destination or requested inspection boundary is unsafe."""


def _as_of(value: str) -> str:
    if not isinstance(value, str):
        raise DataCheckError("as_of must be an ISO date string")
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise DataCheckError(f"as_of must be an ISO date, got {value!r}") from exc


def run_incident_from_error(error: Exception) -> RunIncident:
    """Convert a failed read/measurement boundary into a user-safe operational incident."""
    lines = str(error).splitlines()
    detail = next((line.strip() for line in lines if line.strip()), type(error).__name__)
    return RunIncident(
        id="data_read_failed",
        name="Could not read the data",
        summary="Could not read the requested data. Data checks were not run.",
        technical_detail=f"{type(error).__name__}: {detail}",
    )


def _atomic_json(path: pathlib.Path, value: dict[str, object]) -> None:
    raw = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _empty_facts(*, as_of: str, data_fingerprint: str = "") -> DataFacts:
    """Return honest unknown facts when definition checks stop measurement early."""
    return DataFacts(
        as_of=as_of,
        data_fingerprint=data_fingerprint,
        data_range=("", ""),
        n_rows=0,
        n_series=0,
        n_dates=0,
        duplicate_series_timestamps=0,
        missing_daily_timestamps=None,
        null_counts={},
        inconsistent_static_series={},
        profile=None,
        spark_schema={},
    )


def _definition_failure(error: PanelError) -> CheckResult:
    """Turn a contract validation error into one plain, structured data-check failure."""
    detail = str(error)
    if "columns claimed by the panel but absent" in detail:
        check_id, name = "required_columns", "Required columns"
        summary = "One or more columns named in the data definition are missing."
    elif "incompatible Arrow type" in detail:
        check_id, name = "column_types", "Column types"
        summary = "One or more required columns has an unsuitable data type."
    else:
        check_id, name = "data_definition", "Data definition"
        summary = "The data definition does not match the selected dataset."
    return CheckResult(check_id, name, "failed", summary, technical_detail=detail)


def _definition_passes() -> list[CheckResult]:
    """Record successful static checks explicitly so users see what was checked."""
    return [
        CheckResult(
            "data_definition", "Data definition", "passed", "The data definition is valid."
        ),
        CheckResult(
            "required_columns", "Required columns", "passed", "All required columns exist."
        ),
        CheckResult(
            "column_types", "Column types", "passed", "Required columns have suitable types."
        ),
        CheckResult(
            "forecast_meaning",
            "Forecast meaning",
            "passed",
            "The requested forecast meaning matches the declared data contract.",
        ),
    ]


def inspect_panel(panel: Panel, spark: SparkSession, *, as_of: str) -> CheckReport | RunIncident:
    """Run universal data checks through an explicit date, returning facts or a run incident."""
    cutoff = _as_of(as_of)
    if not pathlib.Path(panel.path).exists():
        return RunIncident(
            id="data_not_found",
            name="Could not read the data",
            summary="The requested data location does not exist. Data checks were not run.",
        )
    try:
        fingerprint = directory_digest(panel.path, suffixes=(".parquet",))
    except LineageError as exc:
        return run_incident_from_error(exc)
    try:
        panel.validate()
    except PanelError as exc:
        facts = _empty_facts(as_of=cutoff, data_fingerprint=fingerprint)
        return CheckReport("data", facts, (_definition_failure(exc),))

    raw = spark.read.parquet(panel.path)
    frame = raw.filter(F.to_date(F.col(panel.time_col)) <= F.lit(cutoff).cast("date"))
    schema = {field.name: field.dataType.simpleString() for field in raw.schema.fields}
    aggregate = frame.agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.countDistinct(panel.series_col).alias("n_series"),
        F.countDistinct(panel.time_col).alias("n_dates"),
        F.min(F.to_date(F.col(panel.time_col))).alias("first_date"),
        F.max(F.to_date(F.col(panel.time_col))).alias("last_date"),
        F.sum(F.col(panel.series_col).isNull().cast("long")).alias("null_series"),
        F.sum(F.col(panel.target_col).isNull().cast("long")).alias("null_target"),
    ).collect()[0]
    null_time = int(raw.filter(F.col(panel.time_col).isNull()).count())
    null_counts = {
        panel.series_col: int(aggregate["null_series"] or 0),
        panel.time_col: null_time,
        panel.target_col: int(aggregate["null_target"] or 0),
    }
    n_rows = int(aggregate["n_rows"] or 0)
    checks = _definition_passes()
    if not n_rows:
        checks.append(
            CheckResult(
                "data_available",
                "Data available for this date",
                "failed",
                f"No data is available on or before {cutoff}.",
                numerator=0,
            )
        )
        facts = _empty_facts(as_of=cutoff, data_fingerprint=fingerprint)
        return CheckReport("data", facts, tuple(checks))

    data_range = (aggregate["first_date"].isoformat(), aggregate["last_date"].isoformat())
    checks.append(
        CheckResult(
            "data_available",
            "Data available for this date",
            "passed",
            f"{n_rows:,} rows are available on or before {cutoff}.",
            numerator=n_rows,
        )
    )
    missing_required = sum(null_counts.values())
    checks.append(
        CheckResult(
            "required_values",
            "Required values are present",
            "failed" if missing_required else "passed",
            (
                f"{missing_required:,} required values are missing."
                if missing_required
                else "Series, date, and forecast values are present."
            ),
            numerator=missing_required,
            denominator=n_rows,
        )
    )
    duplicate_row = (
        frame.groupBy(panel.series_col, panel.time_col)
        .count()
        .filter(F.col("count") > 1)
        .agg(F.sum(F.col("count") - 1).alias("duplicates"))
        .collect()[0]
    )
    duplicate_count = int(duplicate_row["duplicates"] or 0)
    checks.append(
        CheckResult(
            "one_value_per_date",
            "One value per series and date",
            "failed" if duplicate_count else "passed",
            (
                f"{duplicate_count:,} duplicate series/date rows need correction or aggregation."
                if duplicate_count
                else "Each series/date has one forecast value."
            ),
            numerator=duplicate_count,
            denominator=n_rows,
        )
    )
    target_as_double = F.col(panel.target_col).cast("double")
    invalid_target = frame.filter(
        F.isnan(target_as_double) | (F.abs(target_as_double) == F.lit(float("inf")))
    ).count()
    checks.append(
        CheckResult(
            "valid_target_values",
            "Forecast values are usable",
            "failed" if invalid_target else "passed",
            (
                f"{invalid_target:,} forecast values are NaN or infinite."
                if invalid_target
                else "Forecast values are finite numbers."
            ),
            numerator=invalid_target,
            denominator=n_rows,
        )
    )
    negative = frame.filter(F.col(panel.target_col) < 0).count()
    if panel.negative_target_policy == "refuse":
        target_status: CheckStatus = "failed" if negative else "passed"
        target_summary = (
            f"{negative:,} negative forecast values are not allowed by this data definition."
            if negative
            else "Forecast values follow the declared negative-value rule."
        )
    else:
        target_status = "warning" if negative else "passed"
        target_summary = (
            f"{negative:,} negative forecast values require the declared preparation rule."
            if negative
            else "Forecast values follow the declared negative-value rule."
        )
    checks.append(
        CheckResult(
            "target_policy",
            "Forecast value rule",
            target_status,
            target_summary,
            numerator=negative,
            denominator=n_rows,
        )
    )

    static_violations: dict[str, int] = {}
    for column in panel.static + panel.static_reals:
        value_with_missingness = F.struct(
            F.col(column).isNull().alias("is_missing"), F.col(column).alias("value")
        )
        varying = (
            frame.groupBy(panel.series_col)
            .agg(F.countDistinct(value_with_missingness).alias("levels"))
            .filter(F.col("levels") > 1)
            .count()
        )
        if varying:
            static_violations[column] = varying
    if panel.static or panel.static_reals:
        static_count = sum(static_violations.values())
        checks.append(
            CheckResult(
                "static_information",
                "Series information stays consistent",
                "failed" if static_count else "passed",
                (
                    f"Static information changes within {static_count:,} series."
                    if static_count
                    else "Declared series information stays consistent."
                ),
                numerator=static_count,
                denominator=int(aggregate["n_series"] or 0),
            )
        )
    else:
        checks.append(
            CheckResult(
                "static_information",
                "Series information stays consistent",
                "not_applicable",
                "No static series fields were declared.",
            )
        )

    availability_invalid = 0
    if panel.exposure_col is not None:
        availability_invalid = frame.filter(
            F.col(panel.exposure_col).isNotNull() & ~F.col(panel.exposure_col).isin(*STATE_CODES)
        ).count()
    elif panel.status_prob_cols is not None:
        probabilities = [F.col(column).cast("double") for column in panel.status_prob_cols.values()]
        probability_sum = sum(probabilities[1:], probabilities[0])
        invalid_probability = F.lit(False)
        for probability in probabilities:
            invalid_probability = (
                invalid_probability | probability.isNull() | (probability < 0) | (probability > 1)
            )
        availability_invalid = frame.filter(
            invalid_probability | (F.abs(probability_sum - F.lit(1.0)) > F.lit(1e-6))
        ).count()
    if panel.exposure_col is not None or panel.status_prob_cols is not None:
        checks.append(
            CheckResult(
                "availability_status",
                "Availability information is valid",
                "failed" if availability_invalid else "passed",
                (
                    f"{availability_invalid:,} availability records have invalid values."
                    if availability_invalid
                    else "Availability information uses valid values."
                ),
                numerator=availability_invalid,
                denominator=n_rows,
            )
        )
    else:
        checks.append(
            CheckResult(
                "availability_status",
                "Availability information is valid",
                "not_applicable",
                "No availability information was declared.",
            )
        )

    missing_daily: int | None = None
    if panel.frequency == "D":
        per_series = frame.groupBy(panel.series_col).agg(
            F.min(F.to_date(F.col(panel.time_col))).alias("first"),
            F.max(F.to_date(F.col(panel.time_col))).alias("last"),
            F.countDistinct(panel.time_col).alias("observed"),
        )
        missing_row = per_series.agg(
            F.sum(F.datediff(F.col("last"), F.col("first")) + 1 - F.col("observed")).alias(
                "missing"
            )
        ).collect()[0]
        missing_daily = int(missing_row["missing"] or 0)
        checks.append(
            CheckResult(
                "time_grain",
                "Dates match the dataset frequency",
                "warning" if missing_daily else "passed",
                (
                    f"{missing_daily:,} daily dates are missing within observed series ranges."
                    if missing_daily
                    else "Dates match the declared daily frequency."
                ),
                numerator=missing_daily,
            )
        )
    else:
        checks.append(
            CheckResult(
                "time_grain",
                "Dates match the dataset frequency",
                "not_checked",
                f"Date-gap checking is not yet available for {panel.frequency!r} data.",
            )
        )
    coverage = tuple(dict.fromkeys((*panel.known_future, *panel.static, *panel.static_reals)))
    try:
        panel_profile = profile(
            panel.path,
            spark,
            series_col=panel.series_col,
            time_col=panel.time_col,
            target_col=panel.target_col,
            as_of=cutoff,
            exposure_col=panel.exposure_col,
            coverage_cols=coverage,
        )
    except ProfileError as exc:
        return run_incident_from_error(exc)
    facts = DataFacts(
        as_of=cutoff,
        data_fingerprint=fingerprint,
        data_range=data_range,
        n_rows=n_rows,
        n_series=int(aggregate["n_series"] or 0),
        n_dates=int(aggregate["n_dates"] or 0),
        duplicate_series_timestamps=duplicate_count,
        missing_daily_timestamps=missing_daily,
        null_counts=null_counts,
        inconsistent_static_series=static_violations,
        profile=panel_profile,
        spark_schema=schema,
    )
    return CheckReport("data", facts, tuple(checks))


def write_data_observability_report(
    outcome: CheckReport | RunIncident,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Write fresh, portable data-check or run-incident artifacts plus a compact HTML view."""
    root = pathlib.Path(output_dir)
    if root.exists():
        raise DataCheckError(f"data observability output already exists: {root}")
    root.mkdir(parents=True)
    record = outcome.as_record()
    _atomic_json(root / "data_validation_report.json", record)
    if isinstance(outcome, RunIncident):
        document = f"""<!doctype html>
<html lang='en'><meta charset='utf-8'><title>Run status</title>
<style>body{{font:14px system-ui;margin:32px;color:#172033}}</style>
<h1>Run status: data checks not run</h1>
<p>{html.escape(outcome.summary)}</p>
<p>Use the technical log for support details.</p>
</html>"""
        (root / "report.html").write_text(document, encoding="utf-8")
        return root

    facts = outcome.facts
    _atomic_json(
        root / "data_profile.json",
        {
            "schema_version": 2,
            "facts": facts.as_record(),
        },
    )
    status = "passed" if outcome.accepted else "failed"
    status_labels = {
        "passed": "Passed",
        "failed": "Failed",
        "warning": "Warning",
        "not_checked": "Not checked",
        "not_applicable": "Not applicable",
    }
    status_counts = {
        check_status: sum(check.status == check_status for check in outcome.checks)
        for check_status in status_labels
    }
    checks = "".join(
        "<tr>"
        f"<td><strong>{html.escape(check.name)}</strong>"
        f"<div class='check-id'>{html.escape(check.id)}</div></td>"
        f"<td><span class='status {html.escape(check.status)}'>"
        f"{status_labels[check.status]}</span></td>"
        f"<td>{html.escape(check.summary)}</td>"
        f"<td>{_check_measurement(check)}</td></tr>"
        for check in outcome.checks
    )
    profile = facts.profile.as_record() if facts.profile else {}
    profile_cards = "".join(
        _information_card(label, value)
        for label, value in (
            ("Eligible data", _percentage(profile.get("eligible_share"))),
            ("Zero values", _percentage(profile.get("zero_rate_eligible"))),
            ("History per series", _history_range(profile)),
            ("Scale range", _scale_range(profile.get("scale_range"))),
        )
    )
    notes = profile.get("notes")
    notes_html = (
        "".join(f"<li>{html.escape(note)}</li>" for note in notes)
        if type(notes) is list and all(type(note) is str for note in notes)
        else "<li>No additional data information was produced.</li>"
    )
    schema_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(data_type)}</code></td></tr>"
        for name, data_type in sorted(facts.spark_schema.items())
    ) or "<tr><td colspan='2'>Schema was not measured.</td></tr>"
    document = f"""<!doctype html>
<html lang='en'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Data check report</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;margin:0;background:#f6f8fc;color:#172033}}
main{{max-width:1120px;margin:auto;padding:42px 28px 56px}}
.eyebrow{{color:#3b5ccc;font-weight:700;font-size:12px;letter-spacing:.08em;text-transform:uppercase}}
h1{{font-size:32px;letter-spacing:-.04em;margin:6px 0 4px}}
h2{{font-size:18px;margin:0 0 6px}} p{{color:#59677b;margin:0}}
.summary{{display:grid;grid-template-columns:1.3fr repeat(3,1fr);gap:12px;margin:24px 0}}
.card{{background:#fff;border:1px solid #e1e7f0;border-radius:12px;padding:16px}}
.card .value{{font-size:23px;font-weight:750;color:#172033}}
.card .label{{font-size:12px;color:#687589}}
.outcome{{border-left:5px solid #16804a}} .outcome.failed{{border-left-color:#be2c2c}}
.outcome .value{{text-transform:capitalize}}
.metadata{{font-size:13px;color:#657388;margin-top:10px}}
.section{{margin-top:32px}}
.section p{{font-size:13px;margin-bottom:12px}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid #e1e7f0;border-radius:12px}}
table{{border-collapse:collapse;width:100%;min-width:720px}}
th,td{{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #e9edf4}}
th{{background:#f7f9fd;color:#59677b;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
tr:last-child td{{border-bottom:0}}
.check-id{{font:11px ui-monospace,SFMono-Regular,monospace;color:#778398;margin-top:2px}}
.status{{display:inline-block;border-radius:999px;padding:2px 8px;font-size:12px;font-weight:700;
white-space:nowrap}}
.status.passed{{background:#e7f7ed;color:#167243}}
.status.failed{{background:#ffeded;color:#ab2222}}
.status.warning{{background:#fff4d6;color:#8b5b00}}
.status.not_checked,.status.not_applicable{{background:#edf0f5;color:#58667b}}
.information{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.note{{background:#fff;border:1px solid #e1e7f0;border-radius:10px;padding:14px}}
.note .label{{color:#687589;font-size:12px}}
.note .value{{font-size:18px;font-weight:700;margin-top:2px}}
.notes{{background:#eef4ff;border:1px solid #d8e4ff;border-radius:10px;padding:14px 18px;
margin-top:14px;color:#3a4c71}}
.notes ul{{margin:8px 0 0;padding-left:18px}}
details{{margin-top:14px}}
code{{font-size:12px;word-break:break-all}}
@media (max-width:760px){{
main{{padding:28px 16px}}
h1{{font-size:28px}}
.summary,.information{{grid-template-columns:repeat(2,1fr)}}
.summary .outcome{{grid-column:span 2}}
}}
</style>
<main><div class='eyebrow'>Data quality · {html.escape(facts.as_of)}</div>
<h1>Data check {status}</h1>
<p>Snapshot from {html.escape(facts.data_range[0])} to {html.escape(facts.data_range[1])}.</p>
<section class='summary' aria-label='Snapshot summary'>
<article class='card outcome {status}'><div class='value'>{status}</div>
<div class='label'>{status_counts['passed']} passed · {status_counts['warning']} warning ·
{status_counts['failed']} failed</div>
<div class='metadata'>Fingerprint <code>{html.escape(facts.data_fingerprint)}</code></div></article>
<article class='card'><div class='value'>{facts.n_rows:,}</div>
<div class='label'>Rows checked</div></article>
<article class='card'><div class='value'>{facts.n_series:,}</div>
<div class='label'>Series checked</div></article>
<article class='card'><div class='value'>{facts.n_dates:,}</div>
<div class='label'>Dates checked</div></article>
</section>
<section class='section'><h2>Checks</h2><p>Every check is shown with its result and measurement.</p>
<div class='table-wrap'><table><thead><tr><th>Check</th><th>Status</th>
<th>What we found</th><th>Measurement</th></tr></thead>
<tbody>{checks}</tbody></table></div></section>
<section class='section'><h2>Data information</h2>
<p>Useful panel characteristics for choosing and training a model.</p>
<div class='information'>{profile_cards}</div>
<aside class='notes'><strong>What this data suggests</strong>
<ul>{notes_html}</ul></aside>
<details><summary>Schema and snapshot details</summary><div class='table-wrap'><table>
<thead><tr><th>Column</th><th>Type</th></tr></thead>
<tbody>{schema_rows}</tbody></table></div></details></section></main>
</html>"""
    (root / "report.html").write_text(document, encoding="utf-8")
    return root


def _check_measurement(check: CheckResult) -> str:
    """Turn a check count into a labelled human-readable measurement."""
    if check.numerator is None:
        return "—"
    labels = {
        "data_available": "available rows",
        "required_values": "missing required values",
        "one_value_per_date": "duplicate rows",
        "valid_target_values": "invalid forecast values",
        "target_policy": "values outside the declared rule",
        "availability_status": "invalid availability values",
        "time_grain": "missing daily dates",
    }
    label = labels.get(check.id, "affected values")
    if check.denominator is None:
        return f"{check.numerator:,} {label}"
    return f"{check.numerator:,} {label} of {check.denominator:,} rows"


def _information_card(label: str, value: str) -> str:
    return (
        "<article class='note'><div class='label'>"
        f"{html.escape(label)}</div><div class='value'>{html.escape(value)}</div></article>"
    )


def _percentage(value: object) -> str:
    return f"{value:.1%}" if type(value) in {float, int} else "—"


def _history_range(profile: dict[str, object]) -> str:
    minimum = profile.get("min_eligible_history")
    maximum = profile.get("max_eligible_history")
    if type(minimum) is int and type(maximum) is int:
        return f"{minimum:,}–{maximum:,} days"
    return "—"


def _scale_range(value: object) -> str:
    return f"{value:.2f}×" if type(value) in {float, int} else "—"
