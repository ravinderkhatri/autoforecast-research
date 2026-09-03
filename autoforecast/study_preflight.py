"""Study-wide availability checks and immutable derived scoring populations."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow.compute import field
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from autoforecast import grid
from autoforecast.benchmark import (
    MISSING_COLUMN,
    START_COLUMN,
    BenchmarkError,
    _actuals_role,
    _identity,
    _inclusive_period_count,
    _parquet_hash,
    _publish_frozen_benchmark,
    artifact_path,
    boundary,
    digest,
    study_windows,
    verify_benchmark,
)
from autoforecast.frequency import canonical_origin, parse_frequency, parse_origin, period_points
from provenance.lineage import directory_digest

if TYPE_CHECKING:
    from autoforecast.study import Study


class StudyPreflightError(BenchmarkError):
    """A blocked run; ``report_path`` retains checks completed before the blocking error."""

    def __init__(self, message: str, report_path: pathlib.Path):
        super().__init__(f"{message}; report: {report_path}")
        self.report_path = report_path


@dataclasses.dataclass(frozen=True)
class PreflightResult:
    """Resolved study and retained availability report; no model has been fitted."""

    study_path: pathlib.Path
    study_id: str
    study_sha256: str
    report_path: pathlib.Path
    report: dict


def _write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _rows(path: pathlib.Path, columns: list[str] | None = None):
    source = ds.dataset(path, format="parquet")
    for batch in source.to_batches(columns=columns, batch_size=8192):
        yield from batch.to_pylist()


def _source_key_error(frame: DataFrame, record: dict, series_col: str) -> str | None:
    """Use registration's canonical encoding to verify every stored key representation."""
    keys = record["recipe"]["source_keys"]
    if any(
        key not in frame.columns
        or frame.schema[key].dataType.simpleString() != record["key_types"][key]
        for key in keys
    ):
        return "source key storage types differ from registration"
    invalid = F.col(series_col).isNull() | (
        F.col(series_col) != F.to_json(F.struct(*[F.col(key) for key in keys]))
    )
    for key in keys:
        invalid = invalid | F.col(key).isNull()
    if frame.filter(invalid).limit(1).count():
        return "source key mapping differs from canonical series IDs"
    return None


def preflight_study(
    study: Study, options: dict, *, output_dir: str | pathlib.Path | None = None,
) -> PreflightResult:
    """Validate every registered boundary before tuning or downstream fitting.

    Population: the original registered canonical IDs. Strict mode retains them;
    exclusion removes one availability-based union; available mode reports that
    original denominator alongside frozen eligibility and masks for each window.
    Test values are never scored here: only presence and structural validity matter.
    """
    root = (pathlib.Path(output_dir).absolute() if output_dir is not None else
            study._path.parent.parent.parent / "study-preflight" / study.study_id)
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise BenchmarkError("preflight output root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    output = pathlib.Path(tempfile.mkdtemp(prefix="check-", dir=root)) / "report.json"
    report = {
        "schema": "autoforecast.study-preflight.v1",
        "status": "checking",
        "study": {"path": str(study._path), "sha256": study._sha256},
        "series_policy": options["series_policy"],
        "minimum_history": options["minimum_history"],
        "issues": [],
        "windows": {},
        "callback_count": 0,
    }
    try:
        result = _preflight(study, options, report, output)
    except Exception as error:
        report["status"] = "blocked"
        context = report.get("current_check") or {}
        if context.get("stage") in report.get("roles", {}):
            report["roles"][context["stage"]]["status"] = "blocked"
        if not report["issues"]:
            report["issues"].append(
                {
                    "code": "integrity_or_contract",
                    "severity": "error",
                    "explanation": str(error),
                    "series_id": None,
                    "stage": context.get("stage"),
                    "origin": context.get("origin"),
                    "dates": [],
                }
            )
        _write_json(output, report)
        raise StudyPreflightError(str(error), output) from error
    report["current_check"] = None
    _write_json(output, report)
    return result


def _preflight(study: Study, options: dict, report: dict, output: pathlib.Path) -> PreflightResult:
    record = study._record(data=True)
    if "execution_policy" in record:
        frozen = record["execution_policy"]
        if (
            options["series_policy"] != frozen["series_policy"]
            or options["minimum_history"] != frozen["minimum_history"]
        ):
            raise BenchmarkError("availability settings differ from this derived study")
        # Re-resolve from the authenticated parent, then require the same definition.
        from autoforecast.study import load_study

        parent_path = pathlib.Path(record["parent"]["path"])
        with load_study(
            record["parent"]["study_id"],
            registry_root=parent_path.parent.parent,
            spark=study._spark,
        ) as parent:
            result = _preflight(parent, options, report, output)
        if result.study_id != record["study_id"]:
            raise BenchmarkError("derived availability contract changed")
        return result
    report.update(
        original_count=record["cohort_series_count"],
        current_check={"stage": "source-keys", "origin": None},
    )
    panel, recipe = study.panel, record["recipe"]
    frequency = panel.frequency
    cadence = parse_frequency(frequency)
    history_start = parse_origin(recipe["training_start"], frequency)
    mappings = list(_rows(artifact_path(study._path, record, "source-keys")))
    ids = sorted(row[panel.series_col] for row in mappings)
    if (
        len(ids) != record["cohort_series_count"]
        or len(set(ids)) != len(ids)
        or any(type(sid) is not str or not sid for sid in ids)
    ):
        raise BenchmarkError("source mapping canonical IDs differ from the frozen cohort")
    mapping_frame = study._spark.read.parquet(
        str(artifact_path(study._path, record, "source-keys"))
    )
    key_error = _source_key_error(mapping_frame, record, panel.series_col)
    if key_error:
        raise BenchmarkError(key_error)
    keys = recipe["source_keys"]
    declared_starts = {
        tuple(item["keys"][key] for key in keys): parse_origin(item["start"], frequency)
        for item in recipe["series_starts"]["values"]
    }
    if any(
        row[START_COLUMN] != declared_starts.get(tuple(row[key] for key in keys), history_start)
        for row in mappings
    ):
        raise BenchmarkError("source mapping history starts differ from registration")
    id_set = set(ids)
    starts = {row[panel.series_col]: row[START_COLUMN] for row in mappings}
    windows = study_windows(recipe)
    by_key = {
        boundary(stage, origin, horizon, frequency): (stage, origin, horizon)
        for stage, origin, horizon in windows
    }
    points = {
        key: tuple(period_points(origin, frequency, horizon))
        for key, (_, origin, horizon) in by_key.items()
    }
    histories = {key: {sid: 0 for sid in ids} for key in by_key}
    actuals: dict[str, dict] = {key: {} for key in by_key}
    role_counts = {}
    report.update(
        study_id=record["study_id"],
        original_count=len(ids),
        original_cohort_sha256=digest(ids),
        preparation_rule=record["policies"],
    )
    report["roles"] = {
        role: {
            "expected_count": len(ids),
            "actual_count": None,
            "missing_ids": None,
            "extra_ids": None,
            "status": "not_checked",
        }
        for role in ("selection", "final_training", "validation_score", "final_score")
    }
    for role in report["roles"]:
        issue_start = len(report["issues"])
        report["current_check"] = {"stage": role, "origin": None}
        source = ds.dataset(artifact_path(study._path, record, role), format="parquet")
        schema = source.schema
        if (
            not pa.types.is_string(schema.field(panel.series_col).type)
            or not pa.types.is_boolean(schema.field(MISSING_COLUMN).type)
            or not (
                pa.types.is_integer(schema.field(panel.target_col).type)
                or pa.types.is_floating(schema.field(panel.target_col).type)
                or pa.types.is_decimal(schema.field(panel.target_col).type)
            )
        ):
            raise BenchmarkError(f"{role}: invalid ID, missing marker or target storage type")
        if (
            frequency == "H"
            and (
                not pa.types.is_timestamp(schema.field(panel.time_col).type)
                or schema.field(panel.time_col).type.tz is not None
            )
        ) or (frequency != "H" and not pa.types.is_date(schema.field(panel.time_col).type)):
            raise BenchmarkError(f"{role}: invalid time storage type")
        role_path = artifact_path(study._path, record, role)
        frame = study._spark.read.parquet(str(role_path))
        key_error = _source_key_error(frame, record, panel.series_col)
        if key_error:
            report["issues"].append(
                {
                    "code": "source_key_mapping",
                    "severity": "error",
                    "stage": role,
                    "series_id": None,
                    "origin": None,
                    "dates": [],
                    "explanation": key_error,
                }
            )
        duplicate = (
            frame.groupBy(panel.series_col, panel.time_col)
            .count()
            .filter(F.col("count") != 1)
            .collect()
        )
        for row in duplicate:
            report["issues"].append(
                {
                    "code": "duplicate_series_date",
                    "severity": "error",
                    "stage": role,
                    "series_id": row[panel.series_col],
                    "origin": None,
                    "dates": [str(row[panel.time_col])],
                    "explanation": "duplicate series/date in frozen prepared data",
                }
            )
        test_points = points[next(key for key in by_key if key.startswith("test/"))]
        lower = test_points[0] if role == "final_score" else history_start
        upper = (
            test_points[-1]
            if role == "final_score"
            else parse_origin(
                recipe["validation_origins"][-1]
                if role == "selection"
                else recipe["final_train_cutoff"],
                frequency,
            )
        )
        written_upper = (
            test_points[-1]
            if panel.known_future
            and role
            in {
                "selection",
                "final_training",
            }
            else upper
        )
        present = set()
        counts = {
            sid: {
                "required_observations": 0,
                "written_rows": 0,
                "originally_missing": 0,
                "zero_filled": 0,
            }
            for sid in ids
        }
        for row in _rows(
            artifact_path(study._path, record, role),
            [panel.series_col, panel.time_col, panel.target_col, MISSING_COLUMN],
        ):
            sid, point, value = row[panel.series_col], row[panel.time_col], row[panel.target_col]
            present.add(sid)
            if sid not in id_set:
                report["issues"].append(
                    {
                        "code": "unexpected_id",
                        "severity": "error",
                        "series_id": sid,
                        "stage": role,
                        "origin": None,
                        "dates": [str(point)],
                        "explanation": "ID is absent from the frozen canonical cohort",
                    }
                )
                continue
            text = canonical_origin(point, frequency)
            if parse_origin(text, frequency) != point:
                raise BenchmarkError(f"{role}: invalid or duplicate series/date {sid!r} {text}")
            if not max(lower, starts[sid]) <= point <= written_upper:
                raise BenchmarkError(f"{role}: date outside its frozen range: {sid!r} {text}")
            _inclusive_period_count(history_start, point, cadence, label=f"{role} date {text}")
            missing = row[MISSING_COLUMN]
            if (
                type(missing) is not bool
                or missing != (value is None)
                or value is not None
                and (not math.isfinite(value) or value < 0)
            ):
                raise BenchmarkError(f"{role}: invalid target or original missing marker")
            counts[sid]["written_rows"] += 1
            if point <= upper:
                counts[sid]["required_observations"] += 1
                counts[sid]["originally_missing"] += int(missing)
            for key, (stage, origin, _horizon) in by_key.items():
                training_role = "final_training" if stage == "test" else "selection"
                if (
                    role == training_role
                    and starts[sid] <= point <= parse_origin(origin, frequency)
                    and value is not None
                ):
                    histories[key][sid] += 1
                if role == _actuals_role(stage) and point in points[key]:
                    actuals[key][(sid, point)] = (value, missing)
        role_counts[role] = counts
        report.setdefault("roles", {})[role] = {
            "expected_count": len(ids),
            "actual_count": len(present),
            "missing_ids": sorted(id_set - present),
            "extra_ids": sorted(present - id_set, key=str),
            "actual_ids_sha256": digest(sorted(present, key=str)),
            "status": "checked",
        }
        for sid in ids:
            first = max(lower, starts[sid])
            expected_count = (
                _inclusive_period_count(first, upper, cadence, label=role) if first <= upper else 0
            )
            if counts[sid]["required_observations"] != expected_count:
                report["issues"].append(
                    {
                        "code": "missing_or_extra_prepared_rows",
                        "severity": "error",
                        "stage": role,
                        "series_id": sid,
                        "origin": None,
                        "dates": [],
                        "explanation": (
                            f"frozen prepared rows: expected={expected_count}, "
                            f"actual={counts[sid]['required_observations']}; "
                            "missing demand must retain its row and missing marker"
                        ),
                    }
                )
        totals = {
            name: sum(row[name] for row in counts.values()) for name in record["counts"][role]
        }
        if totals != record["counts"][role]:
            report["issues"].append(
                {
                    "code": "prepared_counts_mismatch",
                    "severity": "error",
                    "stage": role,
                    "series_id": None,
                    "origin": None,
                    "dates": [],
                    "explanation": "prepared row and missingness counts differ from frozen counts",
                }
            )
        if len(report["issues"]) > issue_start:
            report["roles"][role]["status"] = "blocked"
    if report["issues"]:
        report["current_check"] = {
            "stage": report["issues"][0]["stage"], "origin": None,
        }
        raise BenchmarkError(
            "frozen prepared data integrity failed: "
            + ", ".join(sorted({issue["code"] for issue in report["issues"]}))
        )
    for key, (_stage, _origin, _horizon) in by_key.items():
        report["current_check"] = {"stage": _stage, "origin": _origin}
        grid_path = study._path.parent / record["grids"][key]["path"]
        expected = {(sid, point) for sid in ids for point in points[key]}
        seen_grid = set()
        for row in _rows(grid_path, ["series_id", "date"]):
            cell = (row["series_id"], row["date"])
            if cell in seen_grid or cell not in expected:
                raise BenchmarkError(f"{key}: duplicate or unexpected forecast-grid cell {cell}")
            seen_grid.add(cell)
        if seen_grid != expected:
            raise BenchmarkError(f"{key}: forecast grid is missing required series/date cells")
        coverage_path = study._path.parent / record["coverage"][key]["path"]
        coverage = list(_rows(coverage_path))
        expected_coverage = []
        for sid in ids:
            missing = sum(actuals[key].get((sid, point), (None, True))[1] for point in points[key])
            excluded = missing if recipe["actual_missing"] == "mask" else 0
            expected_coverage.append(
                {
                    "series_id": sid,
                    "expected": len(points[key]),
                    "originally_missing": missing,
                    "zero_filled": 0,
                    "excluded": excluded,
                    "scored": len(points[key]) - excluded,
                }
            )
        if (
            any(
                type(row.get(name)) is not int
                for row in coverage
                for name in ("expected", "originally_missing", "zero_filled", "excluded", "scored")
            )
            or sorted(coverage, key=lambda row: str(row["series_id"])) != expected_coverage
        ):
            raise BenchmarkError(f"{key}: frozen coverage differs from actual rows and masks")
    references = {}
    report["current_check"] = {"stage": "scoring_references", "origin": None}
    if "scoring_references" in record["artifacts"]:
        for row in _rows(artifact_path(study._path, record, "scoring_references")):
            key, sid = row["boundary"], row["series_id"]
            if key not in by_key or sid not in id_set or (key, sid) in references:
                raise BenchmarkError(
                    "scoring references have duplicate or unexpected IDs/boundaries"
                )
            weight, scale = row["weight"], row["scale"]
            if (
                type(weight) not in (float, int)
                or not math.isfinite(weight)
                or weight < 0
                or scale is not None
                and (type(scale) not in (float, int) or not math.isfinite(scale) or scale < 0)
            ):
                raise BenchmarkError("invalid frozen scoring weights or scales")
            references[key, sid] = row
        if set(references) != {(key, sid) for key in by_key for sid in ids}:
            raise BenchmarkError(
                "scoring references do not cover every series/boundary exactly once"
            )
        for key in by_key:
            total = math.fsum(references[key, sid]["weight"] for sid in ids)
            if not 0 < total < math.inf:
                raise BenchmarkError("scoring boundary needs positive finite total weight")
    if "wrmsse" in options["metrics"] and not references:
        raise BenchmarkError("requested WRMSSE requires registered scoring references")
    policy, minimum = options["series_policy"], options["minimum_history"]
    excluded = set()
    for key, (stage, origin, _horizon) in by_key.items():
        for sid in ids:
            if histories[key][sid] < minimum:
                excluded.add(sid)
                report["issues"].append(
                    {
                        "code": "insufficient_history",
                        "series_id": sid,
                        "stage": stage,
                        "origin": origin,
                        "dates": [],
                        "severity": "error" if policy == "error" else "availability",
                        "explanation": (
                            f"usable observations={histories[key][sid]}; " f"required={minimum}"
                        ),
                    }
                )
            missing = [
                canonical_origin(p, frequency)
                for p in points[key]
                if actuals[key].get((sid, p), (None, True))[0] is None
            ]
            if missing:
                excluded.add(sid)
                report["issues"].append(
                    {
                        "code": "unavailable_actuals",
                        "series_id": sid,
                        "stage": stage,
                        "origin": origin,
                        "dates": missing,
                        "severity": "error"
                        if policy == "error" and recipe["actual_missing"] == "error"
                        else "availability",
                        "explanation": "actuals absent after the frozen preparation policy",
                    }
                )
    if any(issue["severity"] == "error" for issue in report["issues"]):
        raise BenchmarkError("study preflight blocked by required data availability")
    retained = sorted(id_set - excluded) if policy == "exclude_unavailable" else ids
    if not retained:
        raise BenchmarkError("availability policy excludes every series")
    resolved, coverage_rows = {}, {}
    for key, (stage, origin, horizon) in by_key.items():
        eligible = [sid for sid in retained if histories[key][sid] >= minimum]
        if not eligible:
            raise BenchmarkError(f"{key}: no series has the required training history")
        masks, coverage = [], []
        has_positive_volume = False
        for sid in eligible:
            observed = 0
            for point in points[key]:
                value, missing = actuals[key].get((sid, point), (None, True))
                masks.append([sid, canonical_origin(point, frequency), not missing])
                observed += int(not missing)
                # Test target magnitudes do not enter any objective capability decision.
                if stage != "test" and value is not None:
                    has_positive_volume |= value > 0
            coverage.append(
                {
                    "series_id": sid,
                    "expected": horizon,
                    "originally_missing": horizon - observed,
                    "zero_filled": 0,
                    "excluded": horizon - observed,
                    "scored": observed,
                }
            )
        common = [
            canonical_origin(point, frequency)
            for point in points[key]
            if len(eligible) == len(retained)
            and all(actuals[key].get((sid, point), (None, True))[0] is not None for sid in retained)
        ]
        observed_cells = sum(
            not actuals[key].get((sid, point), (None, True))[1]
            for sid in ids for point in points[key]
        )
        resolved[key] = {
            "observed_cells": observed_cells,
            "eligible_ids": eligible,
            "mask_sha256": digest(masks),
            "common_dates": common,
        }
        coverage_rows[key] = coverage
        scored = sum(row["scored"] for row in coverage)
        wrmsse_reason = None
        if references:
            if not 0 < math.fsum(references[key, sid]["weight"] for sid in retained) < math.inf:
                raise BenchmarkError(f"{key}: retained weights have no finite positive total")
            if any(not references[key, sid]["scale"] for sid in retained):
                wrmsse_reason = "undefined_scale"
            elif len(eligible) != len(retained):
                wrmsse_reason = "unfittable_weighted_series"
            elif policy == "use_available" and not common:
                wrmsse_reason = "no_common_complete_dates"
            elif any(row["scored"] == 0 for row in coverage):
                wrmsse_reason = "missing_series_actuals"
        report["windows"][key] = {
            "stage": stage,
            "origin": origin,
            "horizon": horizon,
            "original_series": len(ids),
            "retained_series": len(retained),
            "trainable_series": len(eligible),
            "expected_cells": len(ids) * horizon,
            "forecast_cells": len(eligible) * horizon,
            "observed_cells": observed_cells,
            "filled_cells": 0,
            "scored_cells": scored,
            "scored_series": sum(r["scored"] > 0 for r in coverage),
            "common_wrmsse_dates": len(common),
            "wrmsse_unavailable_reason": wrmsse_reason,
            "status": "pending_actuals"
            if not scored
            else "partial"
            if scored < len(ids) * horizon
            else "ready",
            **resolved[key],
        }
        if key in options["selection_boundaries"]:
            objective = options["metric"]
            reason = (
                "empty_scoring_mask"
                if not scored
                else "zero_actual_volume"
                if objective == "wape" and not has_positive_volume
                else wrmsse_reason
                if objective == "wrmsse"
                else None
            )
            if reason:
                report["issues"].append(
                    {
                        "code": reason,
                        "severity": "error",
                        "series_id": None,
                        "stage": stage,
                        "origin": origin,
                        "dates": [],
                        "explanation": "required objective unavailable",
                    }
                )
                raise BenchmarkError(f"{key}: required {objective} objective unavailable: {reason}")
    report.update(
        retained_count=len(retained),
        excluded_ids=sorted(id_set - set(retained)),
        effective_cohort_sha256=digest(retained),
        interpretation="retrospective_availability" if policy != "error" else "frozen_cohort",
    )
    path = study._path
    if policy == "use_available" or policy == "exclude_unavailable" and retained != ids:
        path = _derive(
            study, record, retained, resolved, coverage_rows, role_counts, minimum, policy
        )
    _, effective = verify_benchmark(path)
    report.update(
        status="passed",
        effective_study_id=effective["study_id"],
        effective_study_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return PreflightResult(
        path, effective["study_id"], report["effective_study_sha256"], output, report
    )


def _derive(study, original, retained, windows, coverage_rows, role_counts, minimum, policy):
    """Publish one content-addressed definition; unchanged role Parquet stays in the parent."""
    record = copy.deepcopy(original)
    original_ids = sorted(
        r[study.panel.series_col]
        for r in _rows(artifact_path(study._path, original, "source-keys"))
    )
    record["parent"] = {
        "path": str(study._path.resolve()),
        "sha256": study._sha256,
        "study_id": original["study_id"],
    }
    record["execution_policy"] = {
        "series_policy": policy,
        "minimum_history": minimum,
        "original_ids": original_ids,
        "retained_ids": retained,
        "excluded_ids": sorted(set(original_ids) - set(retained)),
        "cohort_sha256": digest(retained),
        "windows": windows,
        "interpretation": "retrospective_availability",
    }
    record["cohort_series_count"] = len(retained)
    record["excluded_new_series"] = record["source_series_count"] - len(retained)
    record["inherited_roles"] = sorted(record["artifacts"])
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=".derive-", dir=study._path.parent.parent))
    try:
        import yaml

        dataset = yaml.safe_load((study._path.parent / "dataset.yaml").read_text())
        dataset["dataset"]["required_series_count"] = len(retained)
        (scratch / "dataset.yaml").write_text(yaml.safe_dump(dataset))
        record["dataset_sha256"] = hashlib.sha256(
            (scratch / "dataset.yaml").read_bytes()
        ).hexdigest()
        if retained != original_ids:
            for role in ("source-keys", "scoring_references"):
                if role not in record["artifacts"]:
                    continue
                table = ds.dataset(
                    artifact_path(study._path, original, role), format="parquet"
                ).to_table()
                key = study.panel.series_col if role == "source-keys" else "series_id"
                table = table.filter(field(key).isin(retained))
                if role == "scoring_references":
                    rows = table.to_pylist()
                    totals = {}
                    for row in rows:
                        totals[row["boundary"]] = totals.get(row["boundary"], 0.0) + row["weight"]
                    for row in rows:
                        row["weight"] /= totals[row["boundary"]]
                    table = pa.Table.from_pylist(rows, schema=table.schema)
                (scratch / role).mkdir()
                pq.write_table(table, scratch / role / "part.parquet")
                record["role_paths"][role] = role
                record["artifacts"][role] = directory_digest(scratch / role, suffixes=(".parquet",))
                record["logical_snapshots"][role] = _parquet_hash(scratch / role)
                record["inherited_roles"].remove(role)
            for role, counts in role_counts.items():
                record["counts"][role] = {
                    name: sum(counts[sid][name] for sid in retained)
                    for name in next(iter(counts.values()))
                }
            starts = record["recipe"]["series_starts"]["values"]
            if starts:
                source_keys = record["recipe"]["source_keys"]
                kept = {tuple(r[k] for k in source_keys) for r in _rows(scratch / "source-keys")}
                record["recipe"]["series_starts"]["values"] = [
                    r for r in starts if tuple(r["keys"][k] for k in source_keys) in kept
                ]
        for key, window in windows.items():
            path = scratch / "grids" / key
            path.mkdir(parents=True)
            table = ds.dataset(
                study._path.parent / original["grids"][key]["path"], format="parquet"
            ).to_table()
            table = table.filter(field("series_id").isin(window["eligible_ids"]))
            pq.write_table(table, path / "part.parquet")
            logical = grid.content_hash(path, study._spark, frequency=study.panel.frequency)
            record["grids"][key].update(
                sha256=directory_digest(path, suffixes=(".parquet",)),
                logical_sha256=logical,
                n_cells=len(table),
            )
            record["grid_identities"][key] = {
                "content_sha256": _parquet_hash(path),
                "logical_sha256": logical,
                "n_cells": len(table),
            }
            path = scratch / "coverage" / key
            path.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(coverage_rows[key]), path / "part.parquet")
            record["coverage"][key]["sha256"] = directory_digest(path, suffixes=(".parquet",))
            record["coverage_identities"][key] = {
                "content_sha256": _parquet_hash(path),
                "counts": {
                    name: sum(row[name] for row in coverage_rows[key])
                    for name in (
                        "expected",
                        "originally_missing",
                        "zero_filled",
                        "excluded",
                        "scored",
                    )
                },
            }
        identity = digest(_identity(record))
        record.update(study_id=identity, benchmark_id=identity)
        _write_json(scratch / "benchmark.json", record)
        destination = study._path.parent.parent / identity
        _publish_frozen_benchmark(scratch, destination, identity)
        return destination / "benchmark.json"
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
