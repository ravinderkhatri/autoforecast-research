"""Spark-first data checks, profiling facts, and a compact static report.

``Panel.validate`` verifies the declared Parquet schema and ``data_admission`` decides which
models can consume a panel.  This module fills the separate operational boundary: inspect the
actual Spark rows at an explicit as-of date, classify universal data-check failures, and preserve
the resulting facts and snapshot hash for a study. A source-read failure is a ``RunIncident``, not
a data check. It intentionally has no pandas path.
"""

from __future__ import annotations

import html
import json
import os
import pathlib

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from spec.forecast_spec import STATE_CODES

from autoforecast.column_names import fresh_internal_names
from autoforecast.frequency import canonical_origin, parse_frequency, parse_origin
from autoforecast.grid import canonical_key, read_time_parquet, time_literal
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
    "report_source_issues",
    "summarize_source_issues",
    "run_incident_from_error",
    "write_data_observability_report",
]


class DataCheckError(ValueError):
    """A report destination or requested inspection boundary is unsafe."""


def _as_of(value: str, frequency: str) -> str:
    try:
        cadence = parse_frequency(frequency)
        return canonical_origin(parse_origin(value, cadence), cadence)
    except (TypeError, ValueError) as exc:
        raise DataCheckError(
            f"as_of must be canonical for frequency {frequency!r}, got {value!r}"
        ) from exc


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


def _nonfinite(column):
    value = column.cast("double")
    return F.isnan(value) | (F.abs(value) == F.lit(float("inf")))


def _source_value_issues(frame, *, keys, time_col, numeric, forbid_blank):
    """Shared factual predicates; policy and calendar interpretation belong to callers."""
    from pyspark.sql.types import StringType

    from autoforecast.benchmark import _PYTHON_WHITESPACE

    def finding(code, column, predicate, message):
        return F.when(predicate, F.struct(
            F.lit(column).alias("column"), F.lit(code).alias("issue_code"),
            F.col(column).cast("string").alias("observed_value"),
            F.lit(message).alias("message"),
        ))

    predicates = [finding("NULL_ID", key, F.col(key).isNull(), "Series key is null.")
                  for key in keys]
    if forbid_blank:
        predicates.extend(
            finding("BLANK_ID", key, F.translate(F.col(key), _PYTHON_WHITESPACE, "") == "",
                    "The study identifier contract forbids blank keys.")
            for key in keys if isinstance(frame.schema[key].dataType, StringType)
        )
    predicates.append(finding("NULL_DATE", time_col, F.col(time_col).isNull(),
                              "Observation time is null."))
    predicates.extend(finding("NONFINITE_VALUE", name, _nonfinite(F.col(name)),
                              "Declared numeric value is NaN or infinite.") for name in numeric)
    values = frame.select(
        F.struct(*[F.col(key) for key in keys]).alias("keys"),
        F.col(time_col).cast("string").alias("time"),
        F.explode(F.array(*predicates)).alias("issue"),
    ).filter(F.col("issue").isNotNull())
    return values.select(
        "keys", F.lit(None).cast("string").alias("stage"),
        F.lit(None).cast("string").alias("origin"), "time", "issue.*",
        F.lit("failed").alias("status"), F.lit(1).cast("long").alias("affected_rows"),
    )


def _duplicate_keys(frame, keys, count_name):
    return frame.groupBy(*keys).agg(F.count(F.lit(1)).alias(count_name)).filter(
        F.col(count_name) > 1,
    )


def report_source_issues(frame: DataFrame, *, contract, recipe: dict) -> DataFrame:
    """Report factual source issues against an existing dataset/study contract.

    The recipe declares unique source-key/time observations and nonblank string
    keys. Its range and every tuning/validation/test window are explicit. Declared
    series_starts also describe expected series wholly absent from the source;
    otherwise starts come from each series' first in-range observation. Findings
    preserve typed source keys, and time/origin use string representations.

    Missing history/actual findings are warnings, not admission decisions. Finite
    negatives are not nonfinite values. This function does not fill, aggregate,
    exclude, persist, or change the input. Duplicate findings count excess rows
    per key/time; other findings count one affected cell. Multiple windows can
    intentionally report the same cell under different origins.
    """
    from pyspark.sql.types import BooleanType, NumericType, StructField, StructType

    from autoforecast.benchmark import (
        START_COLUMN,
        _normalise_recipe,
        _period_frame,
        study_windows,
    )
    from autoforecast.column_names import validate_column_names
    from autoforecast.frequency import advance

    recipe = _normalise_recipe(recipe)
    panel = contract.panel(".")
    panel.validate_column_names()
    keys, time_col = recipe["source_keys"], panel.time_col
    validate_column_names(frame.columns)
    numeric = list(dict.fromkeys(
        [panel.target_col, *panel.static_reals, *panel.past_only, *panel.known_future],
    ))
    if set([*keys, time_col, *numeric]) - set(frame.columns):
        raise DataCheckError("source lacks declared key, time or numeric columns")
    if any(not isinstance(frame.schema[name].dataType, NumericType) and not (
        name in (*panel.past_only, *panel.known_future)
        and isinstance(frame.schema[name].dataType, BooleanType)
    ) for name in numeric):
        raise DataCheckError("declared numeric source columns require numeric storage types")
    cadence = parse_frequency(panel.frequency)
    if cadence.code != recipe["frequency"]:
        raise DataCheckError("source contract and study frequency differ")
    from autoforecast.grid import canonical_key

    time_type = frame.schema[time_col].dataType.typeName()
    allowed = {"timestamp_ntz"} if cadence.key_kind == "timestamp" else {"date", "timestamp_ntz"}
    if time_type not in allowed:
        raise DataCheckError("source time requires date or timezone-naive timestamp storage")
    canonical_time, valid_time = canonical_key(time_col, cadence)
    if frame.filter(F.col(time_col).isNotNull() & ~valid_time).limit(1).count():
        raise DataCheckError("source contains off-grid or lossy time values")
    frame = frame.withColumn(time_col, canonical_time)
    start = parse_origin(recipe["training_start"], cadence)
    end = advance(parse_origin(recipe["final_train_cutoff"], cadence), cadence,
                  recipe["final_test_horizon"])
    issues = _source_value_issues(frame, keys=keys, time_col=time_col,
                                  numeric=numeric, forbid_blank=True)

    def rows(source, code, column, message, *, stage=None, origin=None,
             value=None, status="failed", affected=None):
        return source.select(
            F.struct(*[F.col(key) for key in keys]).alias("keys"),
            F.lit(stage).cast("string").alias("stage"),
            F.lit(origin).cast("string").alias("origin"),
            F.col(time_col).cast("string").alias("time"),
            F.lit(column).alias("column"), F.lit(code).alias("issue_code"),
            (F.lit(None) if value is None else value).cast("string").alias("observed_value"),
            F.lit(message).alias("message"), F.lit(status).alias("status"),
            (F.lit(1) if affected is None else affected).cast("long").alias("affected_rows"),
        )

    in_range = F.col(time_col).between(time_literal(start, cadence), time_literal(end, cadence))
    issues = issues.unionByName(rows(
        frame.filter(~in_range), "OUTSIDE_DATE_RANGE", time_col,
        "Observation lies outside the declared study input range.", value=F.col(time_col),
    ))
    count_name, = fresh_internal_names(frame.columns, ("_duplicate_count",))
    duplicates = _duplicate_keys(frame, [*keys, time_col], count_name)
    issues = issues.unionByName(rows(
        duplicates, "DUPLICATE_DATE", time_col,
        "The study contract requires one observation per source key and time.",
        value=F.col(count_name), affected=F.col(count_name) - 1,
    ))
    valid = frame.filter(in_range & F.col(time_col).isNotNull())
    for key in keys:
        valid = valid.filter(F.col(key).isNotNull())
    declarations = recipe["series_starts"]["values"]
    if declarations:
        schema = StructType([*(frame.schema[key] for key in keys),
                             StructField(START_COLUMN, frame.schema[time_col].dataType)])
        cohort = frame.sparkSession.createDataFrame([
            (*[entry["keys"][key] for key in keys], parse_origin(entry["start"], cadence))
            for entry in declarations
        ], schema)
    else:
        first_origin = parse_origin(recipe["validation_origins"][0], cadence)
        cohort = valid.filter(F.col(time_col) <= time_literal(first_origin, cadence)).groupBy(
            *keys,
        ).agg(F.min(time_col).alias(START_COLUMN))
    present = valid.filter(F.col(panel.target_col).isNotNull()).select(*keys, time_col).distinct()
    for stage, origin, horizon in study_windows(recipe):
        cutoff = parse_origin(origin, cadence)
        for actual, low, high in (
            (False, start, cutoff),
            (True, advance(cutoff, cadence, 1), advance(cutoff, cadence, horizon)),
        ):
            grid = cohort.crossJoin(_period_frame(frame.sparkSession, low, high, cadence, time_col))
            if not actual:
                grid = grid.filter(F.col(time_col) >= F.col(START_COLUMN))
            missing = grid.join(present, [*keys, time_col], "left_anti")
            issues = issues.unionByName(rows(
                missing, "MISSING_ACTUAL" if actual else "MISSING_HISTORY", panel.target_col,
                "Expected calendar cell has no non-null target.",
                stage=stage, origin=origin, status="warning",
            ))
    return issues


def summarize_source_issues(issues: DataFrame) -> DataFrame:
    """Summarize the detailed report without re-running source predicates."""
    return issues.groupBy("stage", "origin", "column", "issue_code", "status").agg(
        F.count(F.lit(1)).alias("findings"), F.sum("affected_rows").alias("affected_rows"),
    )


def inspect_panel(panel: Panel, spark: SparkSession, *, as_of: str) -> CheckReport | RunIncident:
    """Run universal data checks through an explicit date, returning facts or a run incident."""
    cutoff = _as_of(as_of, panel.frequency)
    cadence = parse_frequency(panel.frequency)
    cutoff_point = parse_origin(cutoff, cadence)
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

    raw = read_time_parquet(spark, panel.path, panel.time_col)
    (
        time_name,
        valid_time_name,
        duplicate_count_name,
        duplicate_sum_name,
        levels_name,
        previous_name,
        first_name,
        last_name,
        observed_name,
        missing_name,
    ) = fresh_internal_names(
        raw.columns,
        (
            "_time_key",
            "_valid_time_key",
            "count",
            "duplicates",
            "levels",
            "_previous",
            "first",
            "last",
            "observed",
            "missing",
        ),
    )
    time_key, valid_time = canonical_key(panel.time_col, cadence)
    if cadence.key_kind == "timestamp" and (
        raw.schema[panel.time_col].dataType.typeName() != "timestamp_ntz"
    ):
        valid_time = valid_time & F.lit(False)
    prepared = raw.withColumn(time_name, time_key).withColumn(valid_time_name, valid_time)
    frame = prepared.filter(F.col(time_name) <= time_literal(cutoff_point, cadence))
    schema = {field.name: field.dataType.simpleString() for field in raw.schema.fields}
    aggregate = frame.agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.countDistinct(panel.series_col).alias("n_series"),
        F.countDistinct(time_name).alias("n_dates"),
        F.min(time_name).alias("first_date"),
        F.max(time_name).alias("last_date"),
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
        _duplicate_keys(frame, [panel.series_col, time_name], duplicate_count_name)
        .agg(F.sum(F.col(duplicate_count_name) - 1).alias(duplicate_sum_name))
        .collect()[0]
    )
    duplicate_count = int(duplicate_row[duplicate_sum_name] or 0)
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
    invalid_target = frame.filter(_nonfinite(F.col(panel.target_col))).count()
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
            .agg(F.countDistinct(value_with_missingness).alias(levels_name))
            .filter(F.col(levels_name) > 1)
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

    # A malformed key has no reliable position relative to ``as_of``. Check it on the source
    # before the bounded comparison, just as the required-value check already does for null time
    # keys; otherwise Spark's null filtering would make malformed keys disappear silently.
    invalid_time_keys = prepared.filter(
        ~F.coalesce(F.col(valid_time_name), F.lit(False))
    ).count()
    ordered = (
        frame.filter(F.col(valid_time_name))
        .select(panel.series_col, time_name)
        .distinct()
        .withColumn(
            previous_name,
            F.lag(time_name).over(Window.partitionBy(panel.series_col).orderBy(time_name)),
        )
    )
    if cadence.unit == "H":
        phase_errors = ordered.filter(
            F.col(previous_name).isNotNull()
            & (
                F.pmod(
                    F.unix_timestamp(time_name) - F.unix_timestamp(previous_name),
                    F.lit(3600 * cadence.multiplier),
                )
                != 0
            )
        ).count()
    elif cadence.unit in {"D", "W"}:
        stride = cadence.multiplier if cadence.unit == "D" else 7
        # W and nD have no intrinsic calendar anchor. The declared inspection boundary is the
        # run's phase authority; pairwise deltas alone accept a complete lattice shifted away
        # from that origin. Count every valid historical row that cannot belong to the run grid.
        phase_errors = frame.filter(
            F.col(valid_time_name)
            & (
                F.pmod(
                    F.datediff(time_name, time_literal(cutoff_point, cadence)),
                    F.lit(stride),
                )
                != 0
            )
        ).count()
    elif cadence.unit == "M":
        phase_errors = 0
    else:
        phase_errors = 0
    off_grid = invalid_time_keys + phase_errors

    per_series = ordered.groupBy(panel.series_col).agg(
        F.min(time_name).alias(first_name),
        F.max(time_name).alias(last_name),
        F.count(F.lit(1)).alias(observed_name),
    )
    if cadence.unit == "H":
        expected = (
            (F.unix_timestamp(last_name) - F.unix_timestamp(first_name))
            / F.lit(3600 * cadence.multiplier)
            + 1
        )
    elif cadence.unit in {"D", "W"}:
        stride = cadence.multiplier if cadence.unit == "D" else 7
        expected = F.datediff(last_name, first_name) / F.lit(stride) + 1
    elif cadence.unit == "M":
        expected = F.round(F.months_between(last_name, first_name)) + 1
    else:
        expected = F.year(last_name) - F.year(first_name) + 1
    missing_row = per_series.agg(
        F.sum(expected.cast("long") - F.col(observed_name)).alias(missing_name)
    ).collect()[0]
    missing_timestamps = max(0, int(missing_row[missing_name] or 0))
    missing_daily = missing_timestamps if panel.frequency == "D" else None
    checks.append(
        CheckResult(
            "time_grain",
            "Times match the dataset frequency",
            "failed" if off_grid else ("warning" if missing_timestamps else "passed"),
            (
                f"{off_grid:,} time keys are off the declared {panel.frequency!r} cadence."
                if off_grid
                else (
                    f"{missing_timestamps:,} periods are missing within observed series ranges."
                    if missing_timestamps
                    else f"Time keys match the declared {panel.frequency!r} frequency."
                )
            ),
            numerator=off_grid or missing_timestamps,
        )
    )
    coverage = tuple(dict.fromkeys((*panel.known_future, *panel.static, *panel.static_reals)))
    panel_profile = None
    if not off_grid:
        try:
            panel_profile = profile(
                panel.path,
                spark,
                series_col=panel.series_col,
                time_col=panel.time_col,
                target_col=panel.target_col,
                as_of=cutoff,
                frequency=panel.frequency,
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
        missing_timestamps=(None if panel.frequency == "D" else missing_timestamps),
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
<div class='label'>Time keys checked</div></article>
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
        "time_grain": "missing or off-grid periods",
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
        return f"{minimum:,}–{maximum:,} observations"
    return "—"


def _scale_range(value: object) -> str:
    return f"{value:.2f}×" if type(value) in {float, int} else "—"
