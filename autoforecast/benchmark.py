"""Freeze a time-series benchmark before any candidate can choose its scoring population."""

from __future__ import annotations

import datetime as dt
import errno
import hashlib
import heapq
import json
import pathlib
import shutil
import tempfile
from contextlib import ExitStack
from functools import partial

from mlops._safe_files import is_regular_file, read_path_bytes

from autoforecast._exact_json import loads_exact_json
from autoforecast._execution import operation, progress, reported
from autoforecast.column_names import validate_column_names
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.frequency import (
    Cadence,
    FrequencyError,
    TimePoint,
    advance,
    canonical_origin,
    parse_frequency,
    parse_origin,
    period_points,
)
from provenance.lineage import LineageError, directory_digest, snapshot_parquet_directory

STUDY_SCHEMA = "autoforecast.frozen-benchmark.v4"
OPTIONAL_STAGE_SCHEMA = "autoforecast.frozen-benchmark.v5"
STUDY_SCHEMAS = (STUDY_SCHEMA, OPTIONAL_STAGE_SCHEMA)
MISSING_COLUMN = "__af_original_missing"
START_COLUMN = "__af_history_start"


class BenchmarkError(ValueError):
    """The requested population cannot be frozen without changing its meaning."""


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def boundary(stage: str, origin: str, horizon: int, frequency: str = "D") -> str:
    cadence = parse_frequency(frequency)
    return f"{stage}/{canonical_origin(origin, cadence)}/{horizon}/{cadence.code}"


def _recipe_cadence(recipe: dict) -> Cadence:
    try:
        return parse_frequency(recipe.get("frequency", "D"))
    except FrequencyError as error:
        raise BenchmarkError(f"preparation recipe frequency is invalid: {error}") from error


def _recipe_point(value: object, cadence: Cadence) -> TimePoint:
    if type(value) is not str:
        message = (
            "recipe dates must be ISO YYYY-MM-DD strings"
            if cadence.code == "D"
            else f"recipe time points must be canonical for frequency {cadence.code}"
        )
        raise BenchmarkError(message)
    try:
        return parse_origin(value, cadence)
    except FrequencyError as error:
        message = (
            "recipe dates must be ISO YYYY-MM-DD strings"
            if cadence.code == "D"
            else f"recipe time points must be canonical for frequency {cadence.code}"
        )
        raise BenchmarkError(message) from error


def _inclusive_period_count(
    start: TimePoint, end: TimePoint, cadence: Cadence, *, label: str
) -> int:
    """Count an aligned inclusive grid without enumerating its historical observations."""
    if start > end:
        raise BenchmarkError(f"{label} start must not follow its end")
    if cadence.unit == "H":
        assert isinstance(start, dt.datetime) and isinstance(end, dt.datetime)
        periods = (end - start) // dt.timedelta(hours=1)
    elif cadence.unit in {"D", "W"}:
        stride = cadence.multiplier if cadence.unit == "D" else 7
        periods = (end.toordinal() - start.toordinal()) // stride
    elif cadence.unit == "M":
        periods = (end.year - start.year) * 12 + end.month - start.month
    else:
        periods = end.year - start.year
    try:
        if advance(start, cadence, periods) == end:
            return periods + 1
    except (FrequencyError, OverflowError) as error:
        raise BenchmarkError(f"{label} is outside the representable cadence grid") from error
    raise BenchmarkError(f"{label} endpoints are not aligned to frequency {cadence.code}")


def _period_frame(spark, start: TimePoint, end: TimePoint, cadence: Cadence, column: str):
    """Generate historical keys in Spark, keeping driver memory independent of the span."""
    from pyspark.sql import functions as F

    from autoforecast.grid import time_literal

    count = _inclusive_period_count(start, end, cadence, label="observation window")
    index = F.col("id")
    first = time_literal(start, cadence)
    if cadence.unit == "H":
        point = F.timestamp_add("HOUR", index, first)
    elif cadence.unit in {"D", "W"}:
        stride = cadence.multiplier if cadence.unit == "D" else 7
        point = F.date_add(first, (index * stride).cast("int"))
    else:
        months = index * (12 if cadence.unit == "Y" else 1)
        point = F.add_months(first, months.cast("int"))
        if cadence.anchor == "end":
            point = F.last_day(point)
    # Python's representable dates bound even an hourly span below 88 million periods, within
    # Spark range's signed-long index and the integer day/month arithmetic above.
    return spark.range(count).select(point.alias(column))


_PYTHON_WHITESPACE = "".join(chr(value) for value in range(0x110000) if chr(value).isspace())

_ROW_HASH_CHUNK_SIZE = 65536
_ROW_HASH_BATCH_SIZE = 8192
_ROW_HASH_MERGE_FAN_IN = 64


def _write_hash_run(path: pathlib.Path, rows: list[bytes]) -> None:
    rows.sort()
    with path.open("wb") as output:
        output.writelines(rows)


def _merge_hash_runs(paths: list[pathlib.Path], destination: pathlib.Path) -> None:
    with ExitStack() as stack, destination.open("wb") as output:
        inputs = [stack.enter_context(path.open("rb")) for path in paths]
        output.writelines(heapq.merge(*[iter(partial(source.read, 32), b"") for source in inputs]))


def _canonical_row_value(value, dtype):
    """Normalize typed maps recursively; arrays retain order and nulls stay null."""
    import pyarrow as pa

    if value is None:
        return None
    if pa.types.is_map(dtype):
        entries = value.items() if isinstance(value, dict) else value
        normalized = [(_canonical_row_value(key, dtype.key_type),
                       _canonical_row_value(item, dtype.item_type)) for key, item in entries]
        encoded_keys = [json.dumps(key, sort_keys=True, default=str) for key, _ in normalized]
        if any(key is None for key, _ in normalized) or len(set(encoded_keys)) != len(encoded_keys):
            raise BenchmarkError("logical row hashing requires non-null, unique map keys")
        return sorted(normalized, key=lambda pair: json.dumps(
            pair, sort_keys=True, separators=(",", ":"), default=str,
        ))
    if pa.types.is_struct(dtype):
        return {field.name: _canonical_row_value(value[field.name], field.type) for field in dtype}
    if (pa.types.is_list(dtype) or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)):
        return [_canonical_row_value(item, dtype.value_type) for item in value]
    return value


@reported("Calculate logical row SHA (driver)", dataset=True)
def parquet_content_hash(path: pathlib.Path) -> str:
    """Hash typed rows with bounded sort buffers and order-independent map entries.

    Each run holds at most 65,536 32-byte digests; merge passes open at most 64 runs.
    Arrow batches hold at most 8,192 rows. Temporary runs are removed on success or failure.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    from mlops import _volume_files

    if _volume_files.is_volume(path):
        with _volume_files.local_snapshot(path) as local:
            return parquet_content_hash.__wrapped__(local)

    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=False)
    columns = sorted(dataset.schema.names)
    schema = [(name, str(dataset.schema.field(name).type)) for name in columns]
    nested = {name: dataset.schema.field(name).type for name in columns
              if pa.types.is_nested(dataset.schema.field(name).type)}
    hashed = hashlib.sha256(json.dumps(schema).encode())
    with tempfile.TemporaryDirectory(prefix="autoforecast-row-hashes-") as temporary:
        root = pathlib.Path(temporary)
        rows: list[bytes] = []
        run_count = 0
        processed = 0
        for batch in dataset.scanner(
            columns=columns, batch_size=_ROW_HASH_BATCH_SIZE,
            batch_readahead=1, fragment_readahead=1, use_threads=False,
        ).to_batches():
            for row in batch.to_pylist():
                for name, dtype in nested.items():
                    row[name] = _canonical_row_value(row[name], dtype)
                payload = json.dumps(
                    row, sort_keys=True, separators=(",", ":"), default=str,
                ).encode()
                rows.append(hashlib.sha256(payload).digest())
                if len(rows) == _ROW_HASH_CHUNK_SIZE:
                    _write_hash_run(root / f"0-{run_count}", rows)
                    run_count += 1
                    rows.clear()
            processed += batch.num_rows
            progress(completed=processed, unit="rows", phase="hash rows")
        if rows:
            _write_hash_run(root / f"0-{run_count}", rows)
            run_count += 1
            rows.clear()
        progress(completed=processed, unit="rows", phase="merge sorted hash runs")
        level = 0
        if run_count > 1:
            with operation("Merge sorted logical-row hash runs", engine="driver"):
                while run_count > 1:
                    merged_count = 0
                    total = (run_count + _ROW_HASH_MERGE_FAN_IN - 1) // _ROW_HASH_MERGE_FAN_IN
                    for first in range(0, run_count, _ROW_HASH_MERGE_FAN_IN):
                        paths = [root / f"{level}-{index}" for index in range(
                            first, min(first + _ROW_HASH_MERGE_FAN_IN, run_count),
                        )]
                        _merge_hash_runs(paths, root / f"{level + 1}-{merged_count}")
                        merged_count += 1
                        progress(completed=merged_count,
                                 total=total,
                                 unit="runs", phase=f"merge pass {level + 1}")
                        for consumed in paths:
                            consumed.unlink()
                    run_count = merged_count
                    level += 1
        if run_count:
            with (root / f"{level}-0").open("rb") as stream:
                for chunk in iter(partial(stream.read, 1 << 20), b""):
                    hashed.update(chunk)
    return hashed.hexdigest()


def _identity(record: dict) -> dict:
    identity = {
        key: record[key]
        for key in (
            "schema",
            "recipe",
            "key_types",
            "contract_sha256",
            "logical_snapshots",
            "source_series_count",
            "excluded_new_series",
            "cohort_series_count",
            "dataset_sha256",
            "grid_identities",
        )
    }
    identity.update({key: record[key] for key in (
        "role_paths", "metric_definition", "coverage_identities", "counts",
    )})
    if "execution_policy" in record:
        identity.update({key: record[key] for key in (
            "execution_policy", "parent", "inherited_roles",
        )})
    if "admission_identity" in record:
        identity["admission_identity"] = record["admission_identity"]
    return identity


def training_cutoff(recipe: dict, *, last: bool = False) -> str:
    """The permitted selection history ends before validation, or at final training."""
    origins = recipe["validation_origins"]
    return origins[-1 if last else 0] if origins else recipe["final_train_cutoff"]


def study_source_roles(recipe: dict) -> tuple[str, ...]:
    """Configured source roles in their established preflight order."""
    return ("selection", "final_training", *(
        ("validation_score",) if recipe["validation_origins"] else ()
    ), "final_score")


def _normalise_recipe(recipe: dict) -> dict:
    """Freeze the current study defaults before computing its identity."""
    validate_recipe(recipe)
    recipe = dict(recipe)
    if "tuning_schedule" in recipe:
        schedule = {"count": 3, **recipe.pop("tuning_schedule")}
        recipe["tuning_windows"] = resolve_tuning_schedule(recipe, schedule)
        recipe["resolved_tuning_schedule"] = schedule
    return {
        **recipe,
        "frequency": _recipe_cadence(recipe).code,
        "history_missing": recipe.get("history_missing", "preserve"),
        "actual_missing": recipe.get("actual_missing", "error"),
        "series_starts": recipe.get(
            "series_starts", {"rule": "study_lower_bound", "values": []},
        ),
    }


def resolve_tuning_schedule(recipe: dict, schedule: dict) -> list[dict]:
    """Resolve inner windows backwards from the first outer origin in cadence periods.

    Horizon and stride must be explicit. Only the count defaults to three; this
    operation never expands an already registered list of windows.
    """
    if (type(schedule) is not dict or not {"horizon", "stride"} <= set(schedule)
            or set(schedule) - {"horizon", "stride", "count"}):
        raise BenchmarkError("tuning_schedule requires horizon, stride and optional count")
    count, horizon, stride = schedule.get("count", 3), schedule["horizon"], schedule["stride"]
    if any(type(value) is not int or value < 1 for value in (count, horizon, stride)):
        raise BenchmarkError("tuning schedule count, horizon and stride must be positive integers")
    cadence = _recipe_cadence(recipe)
    end = _recipe_point(training_cutoff(recipe), cadence)
    start = _recipe_point(recipe["training_start"], cadence)
    try:
        first = advance(end, cadence, -horizon - stride * (count - 1))
    except (FrequencyError, OverflowError, ValueError) as error:
        raise BenchmarkError("tuning schedule exceeds supported dates") from error
    if first < start:
        raise BenchmarkError("tuning schedule starts before available training history")
    origins = [advance(end, cadence, -horizon - stride * i) for i in reversed(range(count))]
    return [{"origin": canonical_origin(origin, cadence), "horizon": horizon} for origin in origins]


def _metric_definition(recipe: dict) -> dict:
    weighted = recipe["metric"] == "wrmsse"
    return {
        "metric": recipe["metric"],
        "version": "single-level-wrmsse.v2" if weighted else "cell-errors.v2",
        "label": ("masked WRMSSE" if recipe["actual_missing"] == "mask" else "WRMSSE")
        if weighted else recipe["metric"],
        "actual_missing": recipe["actual_missing"],
        "aggregation": "all_frozen_series" if weighted else "available_cells",
        "empty_series": "error" if weighted else "unavailable",
        "empty_aggregate": "unavailable",
        "undefined_scale": "unavailable",
        "weight_redistribution": "forbidden",
    }


def artifact_path(path: pathlib.Path, record: dict, role: str) -> pathlib.Path:
    """Resolve a logical role inside an already authenticated benchmark."""
    if role not in record["artifacts"]:
        raise BenchmarkError(f"unknown benchmark role: {role}")
    if role in record.get("inherited_roles", []):
        parent_path = pathlib.Path(record["parent"]["path"])
        _, parent = read_benchmark_manifest(parent_path, record["parent"]["sha256"])
        if "execution_policy" in parent:
            raise BenchmarkError("derived studies must reference an original frozen definition")
        return artifact_path(parent_path, parent, role)
    return path.parent / record["role_paths"][role]


def window_series_count(record: dict, key: str) -> int:
    """Return the frozen forecast population, distinct from the reporting denominator."""
    policy = record.get("execution_policy")
    return (len(policy["windows"][key]["eligible_ids"]) if policy
            else record["cohort_series_count"])


def resolve_study(output_root: pathlib.Path, study_id: str) -> tuple[pathlib.Path, dict]:
    """Resolve an existing content-addressed study; an ID never supplies business rules."""
    if (type(study_id) is not str or len(study_id) != 64
            or any(character not in "0123456789abcdef" for character in study_id)):
        raise BenchmarkError("study_id must be a lowercase SHA-256 identity")
    path, record = verify_benchmark(output_root / study_id / "benchmark.json")
    if (record["schema"] not in STUDY_SCHEMAS or record.get("study_id") != study_id
            or record["benchmark_id"] != study_id):
        raise BenchmarkError("study_id must resolve to the matching current-format study")
    return path, record


def validate_recipe(recipe: dict) -> None:
    """Validate exact preparation fields before acquiring a runtime."""
    from autoforecast._study_config import validate_shape

    try:
        validate_shape(recipe, "recipe")
    except ValueError as error:
        raise BenchmarkError(str(error)) from error
    if "tuning_schedule" in recipe:
        if "tuning_windows" in recipe or "resolved_tuning_schedule" in recipe:
            raise BenchmarkError("tuning_schedule conflicts with explicit or resolved windows")
        resolved = dict(recipe)
        schedule = resolved.pop("tuning_schedule")
        validate_recipe(resolved)
        resolved["tuning_windows"] = resolve_tuning_schedule(recipe, schedule)
        validate_recipe(resolved)
        return
    if "resolved_tuning_schedule" in recipe:
        schedule = recipe["resolved_tuning_schedule"]
        if (type(schedule) is not dict or "count" not in schedule
                or recipe.get("tuning_windows") != resolve_tuning_schedule(recipe, schedule)):
            raise BenchmarkError("resolved tuning schedule differs from frozen windows")
    if "reference_metrics" in recipe and recipe["reference_metrics"] != ["wrmsse"]:
        raise BenchmarkError("reference_metrics currently supports only [wrmsse]")
    if recipe.get("history_missing", "preserve") not in ("preserve", "zero"):
        raise BenchmarkError("history_missing must be preserve or zero")
    if recipe.get("actual_missing", "error") not in ("error", "mask", "zero"):
        raise BenchmarkError(
            "actual_missing must be error, mask or explicit zero preparation"
        )
    if recipe["metric"] not in ("rmse", "mae", "wape", "wrmsse"):
        raise BenchmarkError("study metric must be rmse, mae, wape, or single-level wrmsse")
    cadence = _recipe_cadence(recipe)
    if (type(recipe["history_policy"]) is not str
            or recipe["history_policy"] not in {"fixed", "revised_snapshot"}):
        raise BenchmarkError("history_policy must be fixed or revised_snapshot")
    if (type(recipe["new_series_policy"]) is not str
            or recipe["new_series_policy"] not in {"error", "exclude_after_first_origin"}):
        raise BenchmarkError("new_series_policy must be error or exclude_after_first_origin")
    if not isinstance(recipe["target_unit"], str) or not recipe["target_unit"].strip():
        raise BenchmarkError("declare the target unit")
    keys = recipe["source_keys"]
    if (type(keys) is not list or not keys
            or any(type(key) is not str or not key.strip() for key in keys)
            or len(set(keys)) != len(keys)):
        raise BenchmarkError("source_keys must be a nonempty unique list")
    try:
        validate_column_names(keys)
    except ValueError as error:
        raise BenchmarkError(str(error)) from error
    if type(recipe["validation_origins"]) is not list:
        raise BenchmarkError("validation_origins must be a list of canonical time points")
    if type(recipe["feature_availability"]) is not dict or any(
        type(key) is not str or not key.strip() or type(value) is not str or not value.strip()
        for key, value in recipe["feature_availability"].items()
    ):
        raise BenchmarkError("feature_availability must map features to evidence column names")
    try:
        validate_column_names([
            *recipe["feature_availability"], *recipe["feature_availability"].values(),
        ])
    except ValueError as error:
        raise BenchmarkError(str(error)) from error
    start = _recipe_point(recipe["training_start"], cadence)
    origins = [_recipe_point(day, cadence) for day in recipe["validation_origins"]]
    cutoff = _recipe_point(recipe["final_train_cutoff"], cadence)
    horizon, test_horizon = recipe["validation_horizon"], recipe["final_test_horizon"]
    try:
        invalid = (
            type(test_horizon) is not int or test_horizon < 1
            or start > cutoff
            or (not origins and (recipe["schema_version"] != 3 or horizon is not None))
            or (bool(origins) and (
                origins != sorted(set(origins)) or start > origins[0]
                or type(horizon) is not int or horizon < 1
                or advance(origins[-1], cadence, horizon) > cutoff
            ))
        )
        advance(cutoff, cadence, test_horizon)
    except (FrequencyError, OverflowError):
        invalid = True
    if invalid:
        raise BenchmarkError("invalid training/validation/testing boundaries")
    if "tuning_windows" in recipe:
        windows = recipe["tuning_windows"]
        if type(windows) is not list or not windows:
            raise BenchmarkError("tuning_windows must be a nonempty list of origin/horizon records")
        tuning_origins = []
        for window in windows:
            if (type(window) is not dict or set(window) != {"origin", "horizon"}
                    or type(window["horizon"]) is not int or window["horizon"] < 1):
                raise BenchmarkError("each tuning window requires an origin and positive horizon")
            origin = _recipe_point(window["origin"], cadence)
            try:
                if (
                    not start
                    <= origin
                    < advance(origin, cadence, window["horizon"])
                    <= _recipe_point(training_cutoff(recipe), cadence)
                ):
                    raise BenchmarkError("tuning targets must precede all outer validation targets")
            except (FrequencyError, OverflowError) as error:
                raise BenchmarkError("invalid tuning window boundary") from error
            _inclusive_period_count(start, origin, cadence, label="tuning window")
            tuning_origins.append(origin)
        if tuning_origins != sorted(set(tuning_origins)):
            raise BenchmarkError("tuning windows must have unique, sorted origins")
    for label, point in [
        *(('validation window', origin) for origin in origins),
        ("final training window", cutoff),
    ]:
        _inclusive_period_count(start, point, cadence, label=label)
    if "series_starts" in recipe:
        declaration = recipe["series_starts"]
        if (type(declaration) is not dict or set(declaration) != {"rule", "values"}
                or type(declaration["rule"]) is not str or not declaration["rule"].strip()
                or type(declaration["values"]) is not list):
            raise BenchmarkError("series_starts requires a nonblank rule and a values list")
        seen = set()
        for entry in declaration["values"]:
            if (type(entry) is not dict or set(entry) != {"keys", "start"}
                    or type(entry["keys"]) is not dict or set(entry["keys"]) != set(keys)):
                raise BenchmarkError("each series start requires the exact source keys and start")
            values = tuple(entry["keys"][key] for key in keys)
            if any(type(value) not in (str, int) or (type(value) is str and not value.strip())
                   for value in values):
                raise BenchmarkError("series start keys must be nonblank strings or integers")
            if values in seen:
                raise BenchmarkError("duplicate declared series start")
            seen.add(values)
            first = _recipe_point(entry["start"], cadence)
            if not start <= first <= _recipe_point(training_cutoff(recipe), cadence):
                raise BenchmarkError("series starts must lie within history and the first origin")
            _inclusive_period_count(start, first, cadence, label="series history start")


def study_windows(recipe: dict) -> list[tuple[str, str, int]]:
    """Enumerate all frozen selection and test boundaries from a validated recipe."""
    return [
        ("tuning", window["origin"], window["horizon"])
        for window in recipe.get("tuning_windows", [])
    ] + [
        ("validation", origin, recipe["validation_horizon"])
        for origin in recipe["validation_origins"]
    ] + [("test", recipe["final_train_cutoff"], recipe["final_test_horizon"])]


def _actuals_role(stage: str) -> str:
    """Keep inner selection labels bound to the pre-outer selection snapshot."""
    try:
        return {
            "tuning": "selection", "validation": "validation_score", "test": "final_score",
        }[stage]
    except KeyError as error:
        raise BenchmarkError("unknown study evaluation stage") from error


def validate_study_panel(panel, recipe: dict) -> None:
    """Check panel and recipe metadata before source access or storage writes."""
    panel.validate_metadata()
    if (
        parse_frequency(panel.frequency).code != _recipe_cadence(recipe).code
        or panel.target_estimand not in ("sales_given_availability", "observed_nonnegative")
        or panel.negative_target_policy != "refuse"
        or panel.status_prob_cols
        or panel.plan
    ):
        raise BenchmarkError(
            "recipe frequency must match a supported nonnegative observed-target panel"
        )
    availability = recipe["feature_availability"]
    try:
        validate_column_names(list(dict.fromkeys([
            *recipe["source_keys"], panel.time_col, panel.target_col, *panel.known_future,
            *panel.static, *panel.static_reals, *panel.past_only,
            *availability.values(), *([panel.exposure_col] if panel.exposure_col else []),
        ])))
    except ValueError as error:
        raise BenchmarkError(str(error)) from error
    if not isinstance(availability, dict) or set(availability) != set(panel.known_future):
        raise BenchmarkError("every known_future feature needs an availability-evidence column")
    if panel.known_future and not panel.exposure_col:
        raise BenchmarkError("future features require exposure_col for existing target masking")


def freeze_benchmark(
    *,
    contract_path: pathlib.Path,
    sources: dict[str, pathlib.Path],
    recipe: dict,
    output_root: pathlib.Path,
    spark,
    study_id: str | None = None,
    scoring_references: pathlib.Path | None = None,
    scratch_root: str | pathlib.Path | None = None,
    admission_identity: dict | None = None,
) -> pathlib.Path:
    """Freeze one study from authenticated, operation-owned input snapshots."""
    with ExitStack() as inputs:
        try:
            return _freeze_benchmark(
                contract_path=contract_path, sources=sources, recipe=recipe,
                output_root=output_root, spark=spark, study_id=study_id,
                scoring_references=scoring_references, inputs=inputs, scratch_root=scratch_root,
                admission_identity=admission_identity,
            )
        except LineageError as error:
            raise BenchmarkError(str(error)) from error


def _freeze_benchmark(
    *,
    contract_path: pathlib.Path,
    sources: dict[str, pathlib.Path],
    recipe: dict,
    output_root: pathlib.Path,
    spark,
    inputs: ExitStack,
    study_id: str | None = None,
    scoring_references: pathlib.Path | None = None,
    scratch_root: str | pathlib.Path | None = None,
    admission_identity: dict | None = None,
) -> pathlib.Path:
    """Validate source scope, normalize keys, and publish immutable role snapshots and grids.

    Population: all distinct source keys observed by the earliest validation origin.
    New series can only be excluded by the explicit pre-test rule in the recipe.
    Studies materialize declared missing-target policies from each series'
    frozen history start.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DateType,
        IntegralType,
        NumericType,
        StringType,
        StructField,
        StructType,
    )

    from autoforecast import grid

    recipe = json.loads(json.dumps(recipe, allow_nan=False))
    if admission_identity is not None:
        admission_identity = json.loads(json.dumps(admission_identity, allow_nan=False))
        _validate_admission_identity(admission_identity)
    if scoring_references is not None and recipe.get("metric") != "wrmsse":
        recipe.setdefault("reference_metrics", ["wrmsse"])
    recipe = _normalise_recipe(recipe)
    cadence = _recipe_cadence(recipe)
    source_roles = set(study_source_roles(recipe))
    if set(sources) != source_roles:
        raise BenchmarkError(f"provide exactly these source roles: {sorted(source_roles)}")
    if ((recipe["metric"] == "wrmsse" or recipe.get("reference_metrics"))
            and scoring_references is None):
        raise BenchmarkError("wrmsse requires frozen scoring_references")
    if study_id is not None:
        resolve_study(output_root, study_id)
    keys = recipe["source_keys"]
    start = _recipe_point(recipe["training_start"], cadence)
    cutoff = _recipe_point(recipe["final_train_cutoff"], cadence)
    test_horizon = recipe["final_test_horizon"]
    end = advance(cutoff, cadence, test_horizon)
    if contract_path.is_symlink() or not is_regular_file(contract_path):
        raise BenchmarkError("dataset contract must be a regular file")
    contract_bytes = read_path_bytes(contract_path)
    contract = _load_dataset_contract_bytes(contract_bytes, contract_path)
    panel = contract.panel(sources["selection"])
    validate_study_panel(panel, recipe)
    availability = recipe["feature_availability"]
    for path in sources.values():
        contract.panel(path).validate()
    all_sources = dict(sources)
    if scoring_references is not None:
        all_sources["scoring_references"] = scoring_references
    before = {name: directory_digest(path, suffixes=(".parquet",))
              for name, path in all_sources.items()}
    snapshot_root = pathlib.Path(inputs.enter_context(tempfile.TemporaryDirectory(
        prefix="autoforecast-freeze-", dir=scratch_root,
    )))
    read_paths = {}
    physical_inputs = {}
    for name, path in all_sources.items():
        identity = (path.resolve(), before[name])
        if identity not in physical_inputs:
            physical_inputs[identity] = snapshot_parquet_directory(
                path, snapshot_root / name, expected_sha256=before[name], label=name,
            )
        read_paths[name] = physical_inputs[identity]
    frames = {}
    key_types = None
    internal = "__af_series_id"
    for name in sources:
        frame = grid.read_time_parquet(spark, read_paths[name], panel.time_col)
        try:
            validate_column_names(frame.columns)
        except ValueError as error:
            raise BenchmarkError(str(error)) from error
        if any(column.casefold().startswith("__af_") for column in frame.columns):
            raise BenchmarkError("source columns beginning __af_ are reserved for preparation")
        if any(k not in frame.columns for k in keys):
            raise BenchmarkError("source keys absent")
        if not isinstance(frame.schema[panel.target_col].dataType, NumericType):
            raise BenchmarkError("study targets must have a numeric, non-boolean storage type")
        types = {key: frame.schema[key].dataType.simpleString() for key in keys}
        if key_types is not None and key_types != types:
            raise BenchmarkError("source key types differ between roles")
        key_types = types
        if any(not isinstance(frame.schema[k].dataType, IntegralType | StringType) for k in keys):
            raise BenchmarkError("source keys must be integral or string, without lossy casts")
        time_type = frame.schema[panel.time_col].dataType
        time_type_name = time_type.typeName()
        if cadence.key_kind == "timestamp" and time_type_name != "timestamp_ntz":
            raise BenchmarkError("hourly source keys must be timezone-naive Parquet timestamps")
        if cadence.key_kind != "timestamp" and not (
            isinstance(time_type, DateType)
            or time_type_name in {"timestamp", "timestamp_ntz"}
        ):
            raise BenchmarkError("non-hourly source keys must be Parquet date values")
        canonical_time, valid_time = grid.canonical_key(panel.time_col, cadence)
        invalid = ~valid_time
        for key in keys:
            invalid = invalid | F.col(key).isNull()
            if isinstance(frame.schema[key].dataType, StringType):
                invalid = invalid | (F.translate(F.col(key), _PYTHON_WHITESPACE, "") == "")
        if frame.filter(invalid).limit(1).count():
            raise BenchmarkError("source contains null time/key or blank string key")
        frame = frame.withColumn(panel.time_col, canonical_time).withColumn(
            internal, F.to_json(F.struct(*[F.col(k) for k in keys]))
        )
        if (frame.groupBy(internal, panel.time_col)
                .agg(F.count(F.lit(1)).alias("__af_count"))
                .filter(F.col("__af_count") != 1).limit(1).count()):
            raise BenchmarkError("duplicate source observations")
        frames[name] = frame
    selection = frames["selection"]
    cohort = (
        selection.filter(
            F.col(panel.time_col)
            <= grid.time_literal(_recipe_point(training_cutoff(recipe), cadence), cadence)
        )
        .select(internal)
        .distinct()
    )
    all_keys = frames["selection"].select(internal)
    for name, frame in frames.items():
        if name == "selection":
            continue
        all_keys = all_keys.unionByName(frame.select(internal))
    original_count, final_count = all_keys.distinct().count(), cohort.count()
    if final_count == 0:
        raise BenchmarkError("pre-test cohort is empty")
    excluded = original_count - final_count
    if excluded and recipe["new_series_policy"] == "error":
        raise BenchmarkError(f"{excluded} new series absent at the first validation origin")
    key_map = selection.join(cohort, internal).select(internal, *keys).distinct()
    declarations = recipe["series_starts"]["values"]
    if declarations:
        schema = StructType([
            *(selection.schema[key] for key in keys),
            StructField(START_COLUMN, selection.schema[panel.time_col].dataType, False),
        ])
        try:
            starts = spark.createDataFrame([
                (*[entry["keys"][key] for key in keys],
                 _recipe_point(entry["start"], cadence))
                for entry in declarations
            ], schema).withColumn(internal, F.to_json(F.struct(*[F.col(k) for k in keys])))
        except (TypeError, ValueError) as error:
            raise BenchmarkError("declared start keys differ from source key types") from error
        declared_ids = starts.select(internal)
        if (cohort.join(declared_ids, internal, "left_anti").limit(1).count()
                or declared_ids.join(cohort, internal, "left_anti").limit(1).count()):
            raise BenchmarkError("declare exactly one history start for every retained series")
        key_map = key_map.join(starts.select(internal, START_COLUMN), internal)
    else:
        key_map = key_map.withColumn(START_COLUMN, grid.time_literal(start, cadence))
    cohort = key_map.select(internal, START_COLUMN)
    counts = {}
    for name, frame in frames.items():
        retained = frame.join(cohort, internal)
        low, high = (
            (advance(cutoff, cadence, 1), end)
            if name == "final_score"
            else (
                start,
                _recipe_point(training_cutoff(recipe, last=True), cadence)
                if name == "selection"
                else cutoff,
            )
        )
        upper = end if panel.known_future and name in {"selection", "final_training"} else high
        window = retained.filter(
            F.col(panel.time_col).between(
                grid.time_literal(low, cadence), grid.time_literal(upper, cadence)
            )
        )
        periods = _period_frame(spark, low, upper, cadence, panel.time_col)
        if window.join(periods, panel.time_col, "left_anti").limit(1).count():
            raise BenchmarkError(
                f"{name}: source contains time keys off the declared {cadence.code} grid"
            )
        required = window.filter(
            F.col(panel.time_col) <= grid.time_literal(high, cadence)
        )
        expected_cells = cohort.crossJoin(
            periods.filter(F.col(panel.time_col) <= grid.time_literal(high, cadence))
        )
        expected_cells = expected_cells.filter(F.col(panel.time_col) >= F.col(START_COLUMN))
        required = required.filter(F.col(panel.time_col) >= F.col(START_COLUMN))
        window = window.filter(F.col(panel.time_col) >= F.col(START_COLUMN))
        target = F.col(panel.target_col).cast("double")
        if (
            required.filter(
                F.col(panel.target_col).isNotNull()
                & (target.isNull() | F.isnan(target) | (target < 0)
                   | (F.abs(target) == float("inf")))
            )
            .limit(1)
            .count()
        ):
            raise BenchmarkError(f"{name}: null, negative, or non-finite observed target")
        if (
            panel.exposure_col
            and required.filter(
                F.col(panel.target_col).isNotNull()
                & (F.col(panel.exposure_col).isNull() | (F.col(panel.exposure_col) != "exact"))
            )
            .limit(1)
            .count()
        ):
            raise BenchmarkError(
                "fixed observed benchmark requires exact observations; "
                "late/unobserved feeds refused"
            )
        for feature, known_at in availability.items():
            known_time = F.expr(f"try_cast(`{known_at}` as timestamp_ntz)")
            first_origin = grid.time_literal(
                _recipe_point(study_windows(recipe)[0][1], cadence), cadence,
            ).cast("timestamp_ntz")
            if (
                known_at not in retained.columns
                or retained.filter(
                    known_time.isNull() | (known_time > first_origin)
                )
                .limit(1)
                .count()
            ):
                raise BenchmarkError(f"unknown future feature at first origin: {feature}")
        required = (
            expected_cells.select(internal, panel.time_col)
            .join(required.drop(*keys, START_COLUMN), [internal, panel.time_col], "left")
            .join(key_map.select(internal, *keys), internal)
            .withColumn(MISSING_COLUMN, F.col(panel.target_col).isNull())
        )
        future = window.filter(
            F.col(panel.time_col) > grid.time_literal(high, cadence)
        ).drop(START_COLUMN).withColumn(MISSING_COLUMN, F.col(panel.target_col).isNull())
        window = required.unionByName(future)
        # Keep verified future-known rows for the existing physical target-mask materializer.
        frames[name] = window
        counts[name] = {
            "required_observations": required.count(),
            "written_rows": frames[name].count(),
        }
        missing_count = required.filter(F.col(MISSING_COLUMN)).count()
        counts[name].update(originally_missing=missing_count, zero_filled=0)
    if recipe["history_policy"] == "fixed":
        last_origin = grid.time_literal(
            _recipe_point(training_cutoff(recipe, last=True), cadence), cadence
        )
        left = frames["selection"].filter(F.col(panel.time_col) <= last_origin)
        right = frames["final_training"].filter(F.col(panel.time_col) <= last_origin)
        if (
            set(left.columns) != set(right.columns)
            or left.select(sorted(left.columns))
            .exceptAll(right.select(sorted(right.columns)))
            .limit(1)
            .count()
        ):
            raise BenchmarkError(
                "conflicting overlapping history between selection and final training"
            )
    identity = {
        "schema": OPTIONAL_STAGE_SCHEMA if recipe["schema_version"] == 3 else STUDY_SCHEMA,
        "recipe": recipe,
        "key_types": key_types,
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_series_count": original_count,
        "excluded_new_series": excluded,
        "cohort_series_count": final_count,
    }
    if admission_identity is not None:
        identity["admission_identity"] = admission_identity

    def authenticate_sources():
        if read_path_bytes(contract_path) != contract_bytes or any(
            directory_digest(path, suffixes=(".parquet",)) != before[name]
            for name, path in all_sources.items()
        ):
            raise BenchmarkError("source changed during preparation")

    authenticate_sources()
    output_root.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=".freeze-", dir=output_root))
    try:
        import yaml

        roles = {**contract.panel_roles, "series_col": internal}
        prepared_contract = {
            "schema_version": 2,
            "dataset": {"id": contract.id, "panel": roles, "required_series_count": final_count},
        }
        (scratch / "dataset.yaml").write_text(yaml.safe_dump(prepared_contract))
        key_map.write.parquet(str(scratch / "source-keys"))
        for name, frame in frames.items():
            frame.write.parquet(str(scratch / name))
        artifact_roles = [*sorted(sources), "source-keys"]
        if scoring_references is not None:
            references = _prepare_scoring_references(
                read_paths["scoring_references"], key_map, keys, recipe, spark,
            )
            references.write.parquet(str(scratch / "scoring_references"))
            artifact_roles.append("scoring_references")
        grids = {}
        coverage = {}
        coverage_identities = {}
        for stage, origin_text, span in study_windows(recipe):
            origin = _recipe_point(origin_text, cadence)
            origin_text = canonical_origin(origin, cadence)
            key = boundary(stage, origin_text, span, cadence.code)
            relative = "grids/" + key
            path = grid.build(
                scratch / "source-keys",
                origin_text,
                span,
                spark,
                scratch / relative,
                series_col=internal,
                time_col=panel.time_col,
                frequency=cadence.code,
            )
            grids[key] = {
                "path": relative,
                "logical_sha256": grid.content_hash(path, spark, frequency=cadence.code),
                "sha256": directory_digest(path, suffixes=(".parquet",)),
                "n_cells": final_count * span,
            }
            actuals = frames[_actuals_role(stage)]
            cells = actuals.filter(
                F.col(panel.time_col).between(
                    grid.time_literal(advance(origin, cadence, 1), cadence),
                    grid.time_literal(advance(origin, cadence, span), cadence),
                )
            )
            # A late-start series has no invented training rows. Its forecast grid
            # still records unavailable actuals before that start explicitly.
            cells = spark.read.parquet(str(path)).select(
                F.col("series_id").alias(internal), F.col("date").alias(panel.time_col),
            ).join(cells, [internal, panel.time_col], "left").withColumn(
                MISSING_COLUMN, F.col(panel.target_col).isNull(),
            )
            per_series = cells.groupBy(internal).agg(
                F.count(F.lit(1)).alias("expected"),
                F.sum(F.col(MISSING_COLUMN).cast("long")).alias("originally_missing"),
            ).withColumnRenamed(internal, "series_id")
            per_series = per_series.withColumn(
                "zero_filled", F.col("originally_missing")
                if recipe["actual_missing"] == "zero" else F.lit(0).cast("long"),
            ).withColumn(
                "excluded", F.col("originally_missing")
                if recipe["actual_missing"] == "mask" else F.lit(0).cast("long"),
            ).withColumn("scored", F.col("expected") - F.col("excluded"))
            if (per_series.count() != final_count
                    or per_series.filter(F.col("expected") != span).limit(1).count()):
                raise BenchmarkError("prepared actuals do not cover the frozen forecast grid")
            missing_cells = cells.filter(F.col(MISSING_COLUMN))
            if recipe["actual_missing"] == "error" and missing_cells.limit(1).count():
                raise BenchmarkError(
                    f"{key}: scoring requires complete actuals; expected={final_count * span}, "
                    f"missing={missing_cells.count()}. "
                    "Correct the source data or explicitly select actual_missing=mask."
                )
            names = ("expected", "originally_missing", "zero_filled", "excluded", "scored")
            totals = per_series.agg(*[F.sum(name).alias(name) for name in names]).first()
            assert totals is not None
            coverage_path = "coverage/" + key
            per_series.write.parquet(str(scratch / coverage_path))
            coverage_identities[key] = {
                "content_sha256": parquet_content_hash(scratch / coverage_path),
                "counts": {name: int(totals[name]) for name in names},
            }
            coverage[key] = {
                "path": coverage_path,
                "sha256": directory_digest(scratch / coverage_path, suffixes=(".parquet",)),
            }
        authenticate_sources()
        logical_snapshots = {name: parquet_content_hash(scratch / name) for name in artifact_roles}
        role_paths = {name: name for name in artifact_roles}
        # Logical roles remain distinct even when their exact typed rows share storage.
        written = {}
        for name in artifact_roles:
            logical = logical_snapshots[name]
            if logical in written:
                role_paths[name] = written[logical]
                shutil.rmtree(scratch / name)
            else:
                written[logical] = name
        identity.update(
            logical_snapshots=logical_snapshots,
            dataset_sha256=hashlib.sha256(read_path_bytes(scratch / "dataset.yaml")).hexdigest(),
            grid_identities={
                key: {
                    "content_sha256": parquet_content_hash(scratch / item["path"]),
                    "logical_sha256": item["logical_sha256"],
                    "n_cells": item["n_cells"],
                }
                for key, item in grids.items()
            },
        )
        identity.update(
            role_paths=role_paths, metric_definition=_metric_definition(recipe),
            coverage_identities=coverage_identities, counts=counts,
        )
        benchmark_id = digest(identity)
        if study_id is not None and study_id != benchmark_id:
            raise BenchmarkError(
                "supplied study_id has different settings or data; create a new study"
            )
        destination = output_root / benchmark_id
        record = {
            **identity,
            "benchmark_id": benchmark_id,
            "study_id": benchmark_id,
            "grids": grids,
            "coverage": coverage,
            "counts": counts,
            "policies": {
                "history_missing": recipe["history_missing"],
                "actual_missing": recipe["actual_missing"],
                "missing_definition": "absent_row_or_null_target",
                "confirmed_zero": "observed", "missing_indicator": MISSING_COLUMN,
            },
            "sources": {
                name: {"path": str(path.resolve()), "sha256": before[name]}
                for name, path in all_sources.items()
            },
            "artifacts": {
                name: directory_digest(scratch / role_paths[name], suffixes=(".parquet",))
                for name in artifact_roles
            },
            "dataset_sha256": hashlib.sha256(read_path_bytes(scratch / "dataset.yaml")).hexdigest(),
        }
        (scratch / "benchmark.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        _publish_frozen_benchmark(scratch, destination, benchmark_id)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    return destination / "benchmark.json"


def _normalise_reference_frame(raw, key_map, keys):
    """Validate source-key types and build the existing canonical reference identities."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType, StringType, StructType

    if set(raw.columns) != {"keys", "stage", "origin", "weight", "scale"}:
        raise BenchmarkError("scoring references require keys, stage, origin, weight, and scale")
    key_type = raw.schema["keys"].dataType
    if (not isinstance(key_type, StructType) or set(key_type.names) != set(keys)
            or any(key_type[key].dataType != key_map.schema[key].dataType for key in keys)
            or any(not isinstance(raw.schema[name].dataType, StringType)
                   for name in ("stage", "origin"))
            or any(not isinstance(raw.schema[name].dataType, NumericType)
                   for name in ("weight", "scale"))):
        raise BenchmarkError("scoring reference key, origin, weight, or scale types are invalid")
    invalid = F.col("keys").isNull()
    for key in keys:
        invalid = invalid | F.col("keys")[key].isNull()
    if raw.filter(invalid).limit(1).count():
        raise BenchmarkError("scoring reference keys cannot be null")
    frame = raw.select(
        F.to_json(F.struct(*[F.col("keys")[key].alias(key) for key in keys])).alias("series_id"),
        "stage", "origin", F.col("weight").cast("double").alias("weight"),
        F.col("scale").cast("double").alias("scale"),
    )
    return frame


def _reference_boundaries(frame, key_map, recipe, spark):
    """Require the exact cohort at every boundary before any numeric exclusion."""
    from pyspark.sql import functions as F

    cadence = _recipe_cadence(recipe)
    boundaries = study_windows(recipe)
    declared = spark.createDataFrame([
        (stage, origin, boundary(stage, origin, horizon, cadence.code))
        for stage, origin, horizon in boundaries
    ], "stage string, origin string, boundary string")
    expected = key_map.select(F.col("__af_series_id").alias("series_id")).crossJoin(declared)
    columns = ["series_id", "stage", "origin"]
    if (frame.groupBy(*columns).count().filter(F.col("count") != 1).limit(1).count()
            or frame.join(expected, columns, "left_anti").limit(1).count()
            or expected.join(frame, columns, "left_anti").limit(1).count()):
        raise BenchmarkError(
            "scoring references must cover every retained series and boundary once"
        )
    return declared


def _prepare_scoring_references(path, key_map, keys, recipe, spark):
    """Bind validated numeric references to complete frozen boundaries."""
    from pyspark.sql import functions as F

    frame = _normalise_reference_frame(spark.read.parquet(str(path)), key_map, keys)
    invalid_weight = (F.col("weight").isNull() | F.isnan("weight")
                      | (F.abs(F.col("weight")) == float("inf")) | (F.col("weight") < 0))
    invalid_scale = (F.col("scale").isNotNull()
                     & (F.isnan("scale") | (F.abs(F.col("scale")) == float("inf"))
                        | (F.col("scale") < 0)))
    if frame.filter(invalid_weight | invalid_scale).limit(1).count():
        raise BenchmarkError("scoring references contain invalid weights or scales")
    declared = _reference_boundaries(frame, key_map, recipe, spark)
    totals = frame.groupBy("stage", "origin").agg(F.sum("weight").alias("total"))
    if totals.filter(
        F.col("total").isNull() | (F.col("total") <= 0)
        | F.isnan("total") | (F.col("total") == float("inf"))
    ).limit(1).count():
        raise BenchmarkError("each scoring boundary needs a finite positive total weight")
    return frame.join(declared, ["stage", "origin"]).select(
        "series_id", "boundary", "weight", "scale",
    )


def _publish_frozen_benchmark(
    scratch: pathlib.Path, destination: pathlib.Path, benchmark_id: str,
) -> None:
    """Publish atomically, or authenticate an identical concurrent winner."""
    try:
        try:
            scratch.rename(destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _, winner = verify_benchmark(destination / "benchmark.json")
            if winner["benchmark_id"] != benchmark_id:
                raise BenchmarkError(
                    "published destination has a different benchmark identity"
                ) from error
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _validate_admission_identity(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "sha256"}
            or value["schema"] not in {"autoforecast.admission-identity.v1",
                                       "autoforecast.supplied-stage-identity.v1"}
            or not isinstance(value["sha256"], str) or len(value["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in value["sha256"])):
        raise BenchmarkError("invalid admission identity binding")


def _validate_manifest(record: object) -> dict:
    """Reject malformed persisted records before any identity or filesystem lookup."""
    def mapping(value, fields, label):
        if type(value) is not dict or set(value) != set(fields):
            raise BenchmarkError(f"benchmark {label} fields changed or missing")
        return value

    def sha256(value, label):
        if (type(value) is not str or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise BenchmarkError(f"benchmark {label} must be a lowercase SHA-256 digest")

    def count(value, label, minimum=0):
        if type(value) is not int or value < minimum:
            raise BenchmarkError(f"benchmark {label} must be an integer >= {minimum}")

    if type(record) is not dict:
        raise BenchmarkError("benchmark manifest must be an object")
    if record.get("schema") not in STUDY_SCHEMAS:
        raise BenchmarkError("unsupported benchmark schema; prepare a current study")
    fields = {
        "schema", "recipe", "key_types", "contract_sha256", "logical_snapshots",
        "source_series_count", "excluded_new_series", "cohort_series_count",
        "dataset_sha256", "grid_identities", "benchmark_id", "grids", "counts",
        "policies", "sources", "artifacts",
        "study_id", "role_paths", "metric_definition", "coverage", "coverage_identities",
    }
    if "execution_policy" in record:
        fields |= {"execution_policy", "parent", "inherited_roles"}
    if "admission_identity" in record:
        fields.add("admission_identity")
        _validate_admission_identity(record["admission_identity"])
    mapping(record, fields, "manifest")
    recipe = record["recipe"]
    validate_recipe(recipe)
    expected_schema = OPTIONAL_STAGE_SCHEMA if recipe["schema_version"] == 3 else STUDY_SCHEMA
    if record["schema"] != expected_schema:
        raise BenchmarkError("recipe version differs from the frozen study schema")
    if (recipe != _normalise_recipe(recipe)
            or record["study_id"] != record["benchmark_id"]
            or record["metric_definition"] != _metric_definition(recipe)):
        raise BenchmarkError("study identity, defaults, or metric definition changed")
    for name in ("benchmark_id", "contract_sha256", "dataset_sha256"):
        sha256(record[name], name)
    types = mapping(record["key_types"], recipe["source_keys"], "key_types")
    if any(type(value) is not str or value not in {"tinyint", "smallint", "int", "bigint", "string"}
           for value in types.values()):
        raise BenchmarkError("benchmark source key types must be integral or string")
    for name in ("source_series_count", "excluded_new_series", "cohort_series_count"):
        count(record[name], name, 1 if name == "cohort_series_count" else 0)
    cohort = record["cohort_series_count"]
    if "execution_policy" in record:
        parent = mapping(record["parent"], {"path", "sha256", "study_id"}, "parent")
        sha256(parent["sha256"], "parent.sha256")
        sha256(parent["study_id"], "parent.study_id")
        if type(parent["path"]) is not str or not pathlib.Path(parent["path"]).is_absolute():
            raise BenchmarkError("parent path must be absolute")
        execution = mapping(record["execution_policy"], {
            "series_policy", "minimum_history", "original_ids", "retained_ids",
            "excluded_ids", "cohort_sha256", "windows", "interpretation",
        }, "execution_policy")
        if execution["series_policy"] not in ("exclude_unavailable", "use_available"):
            raise BenchmarkError("unknown frozen availability policy")
        count(execution["minimum_history"], "minimum_history", 1)
        for name in ("original_ids", "retained_ids", "excluded_ids"):
            ids = execution[name]
            if (type(ids) is not list or any(type(s) is not str or not s for s in ids)
                    or ids != sorted(set(ids))):
                raise BenchmarkError("availability IDs must be unique sorted strings")
        if (not execution["retained_ids"] or len(execution["retained_ids"]) != cohort
                or set(execution["retained_ids"]) & set(execution["excluded_ids"])
                or set(execution["retained_ids"]) | set(execution["excluded_ids"])
                != set(execution["original_ids"])
                or execution["cohort_sha256"] != digest(execution["retained_ids"])
                or execution["interpretation"] != "retrospective_availability"):
            raise BenchmarkError("availability cohort identity does not reconcile")
        if execution["series_policy"] == "use_available" and execution["excluded_ids"]:
            raise BenchmarkError("available-data policy must retain the original denominator")
        windows = {boundary(s, o, h, recipe.get("frequency", "D")): h
                   for s, o, h in study_windows(recipe)}
        mapping(execution["windows"], windows, "availability windows")
        for key, item in execution["windows"].items():
            mapping(item, {"eligible_ids", "mask_sha256", "common_dates", "observed_cells"},
                    "window eligibility")
            count(item["observed_cells"], "observed_cells")
            if item["observed_cells"] > len(execution["original_ids"]) * windows[key]:
                raise BenchmarkError("observed cells exceed the original window population")
            ids = item["eligible_ids"]
            if (type(ids) is not list or not ids or any(type(s) is not str for s in ids)
                    or ids != sorted(set(ids)) or not set(ids) <= set(execution["retained_ids"])):
                raise BenchmarkError("invalid window eligible IDs")
            if (execution["series_policy"] == "exclude_unavailable"
                    and ids != execution["retained_ids"]):
                raise BenchmarkError("exclusion policy requires one fixed cohort")
            sha256(item["mask_sha256"], "window mask")
            dates = item["common_dates"]
            stage, origin, horizon, frequency = key.split("/")
            allowed = {canonical_origin(point, frequency)
                       for point in period_points(origin, frequency, int(horizon))}
            if (type(dates) is not list or any(type(d) is not str for d in dates)
                    or dates != sorted(set(dates)) or not set(dates) <= allowed):
                raise BenchmarkError("common scoring dates must be unique sorted strings")
        inherited = record["inherited_roles"]
        if (type(inherited) is not list or any(type(s) is not str for s in inherited)
                or inherited != sorted(set(inherited))
                or not set(inherited) <= set(record["artifacts"])):
            raise BenchmarkError("invalid inherited role inventory")
    if record["source_series_count"] - cohort != record["excluded_new_series"]:
        raise BenchmarkError("benchmark population counts do not reconcile")
    source_roles = set(study_source_roles(recipe))
    if recipe["metric"] == "wrmsse" or recipe.get("reference_metrics"):
        source_roles.add("scoring_references")
    roles = source_roles | {"source-keys"}
    for name in ("artifacts", "logical_snapshots"):
        for role, value in mapping(record[name], roles, name).items():
            sha256(value, f"{name}.{role}")
    paths = mapping(record["role_paths"], roles, "role_paths")
    if any(type(value) is not str or value not in roles for value in paths.values()):
        raise BenchmarkError("study role paths must name frozen artifact directories")
    for role, value in mapping(record["sources"], source_roles, "sources").items():
        mapping(value, {"path", "sha256"}, f"sources.{role}")
        if type(value["path"]) is not str or not value["path"].strip():
            raise BenchmarkError("benchmark source path must be a nonblank string")
        sha256(value["sha256"], f"sources.{role}.sha256")
    policies = {
        "history_missing": recipe["history_missing"], "actual_missing": recipe["actual_missing"],
        "missing_definition": "absent_row_or_null_target",
        "confirmed_zero": "observed", "missing_indicator": MISSING_COLUMN,
    }
    if record["policies"] != policies:
        raise BenchmarkError("benchmark observation policies changed")
    cadence = _recipe_cadence(recipe)
    start = _recipe_point(recipe["training_start"], cadence)
    cutoff = _recipe_point(recipe["final_train_cutoff"], cadence)
    selection_end = _recipe_point(training_cutoff(recipe, last=True), cadence)
    role_periods = {
        "selection": _inclusive_period_count(
            start, selection_end, cadence, label="selection role"
        ),
        "final_training": _inclusive_period_count(
            start, cutoff, cadence, label="final training role"
        ),
        "final_score": recipe["final_test_horizon"],
    }
    if recipe["validation_origins"]:
        role_periods["validation_score"] = role_periods["final_training"]
    for role, values in mapping(record["counts"], role_periods, "counts").items():
        fields = {"required_observations", "written_rows", "originally_missing", "zero_filled"}
        mapping(values, fields, f"counts.{role}")
        for key, value in values.items():
            count(value, f"counts.{role}.{key}")
        required = values["required_observations"]
        variable_history = role != "final_score"
        if ((not variable_history and required != cohort * role_periods[role])
                or (variable_history and not cohort <= required <= cohort * role_periods[role])
                or values["written_rows"] < values["required_observations"]):
            raise BenchmarkError("benchmark observation counts do not reconcile")
        if not (0 <= values["originally_missing"] <= required
                and values["zero_filled"] == 0):
            raise BenchmarkError("study missingness counts do not reconcile")
    boundaries = {boundary(stage, origin, span, cadence.code): span
                  for stage, origin, span in study_windows(recipe)}
    grids = mapping(record["grids"], boundaries, "grid inventory")
    identities = mapping(record["grid_identities"], boundaries, "grid identity inventory")
    for key, horizon in boundaries.items():
        item = mapping(grids[key], {"path", "sha256", "logical_sha256", "n_cells"}, "grid")
        identity = mapping(
            identities[key], {"content_sha256", "logical_sha256", "n_cells"}, "grid identity",
        )
        if item["path"] != "grids/" + key:
            raise BenchmarkError("benchmark grid path changed")
        for name in ("sha256", "logical_sha256"):
            sha256(item[name], f"grid.{name}")
        for name in ("content_sha256", "logical_sha256"):
            sha256(identity[name], f"grid identity.{name}")
        for value in (item["n_cells"], identity["n_cells"]):
            count(value, "grid n_cells", 1)
            if value != window_series_count(record, key) * horizon:
                raise BenchmarkError("benchmark grid population count changed")
        if item["logical_sha256"] != identity["logical_sha256"]:
            raise BenchmarkError("benchmark grid logical identity changed")
    coverage = mapping(record["coverage"], boundaries, "coverage")
    identities = mapping(record["coverage_identities"], boundaries, "coverage_identities")
    names = {"expected", "originally_missing", "zero_filled", "excluded", "scored"}
    for key, horizon in boundaries.items():
        item = mapping(coverage[key], {"path", "sha256"}, "coverage artifact")
        identity = mapping(identities[key], {"content_sha256", "counts"}, "coverage identity")
        if item["path"] != "coverage/" + key:
            raise BenchmarkError("study coverage path changed")
        sha256(item["sha256"], "coverage sha256")
        sha256(identity["content_sha256"], "coverage content_sha256")
        totals = mapping(identity["counts"], names, "coverage counts")
        for name, value in totals.items():
            count(value, f"coverage.{name}")
        missing = totals["originally_missing"]
        if (totals["expected"] != window_series_count(record, key) * horizon
                or missing > totals["expected"]
                or totals["zero_filled"] != (missing if recipe["actual_missing"] == "zero" else 0)
                or (recipe["actual_missing"] == "error" and missing != 0)
                or totals["excluded"] != (missing if recipe["actual_missing"] == "mask" else 0)
                or totals["scored"] != totals["expected"] - totals["excluded"]):
            raise BenchmarkError("study scoring coverage counts do not reconcile")
    return record


def read_benchmark_manifest(
    path: pathlib.Path, expected_sha256: str | None = None,
) -> tuple[pathlib.Path, dict]:
    """Validate manifest bytes and structure without scanning frozen data artifacts."""
    if path.is_symlink():
        raise BenchmarkError("benchmark manifest must not be a symlink")
    try:
        payload = read_path_bytes(path)
        if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise BenchmarkError("benchmark manifest changed")
        record = loads_exact_json(payload)
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read benchmark manifest: {error}") from error
    record = _validate_manifest(record)
    if digest(_identity(record)) != record["benchmark_id"]:
        raise BenchmarkError("benchmark logical identity changed")
    return path, record


def verify_benchmark(
    path: pathlib.Path, expected_sha256: str | None = None
) -> tuple[pathlib.Path, dict]:
    """Authenticate frozen outputs; original mutable sources are recorded but never re-read."""
    path, record = read_benchmark_manifest(path, expected_sha256)
    if "parent" in record:
        parent_path, parent = read_benchmark_manifest(
            pathlib.Path(record["parent"]["path"]), record["parent"]["sha256"],
        )
        if ("parent" in parent or parent["study_id"] != record["parent"]["study_id"]
                or parent["study_id"] == record["study_id"]):
            raise BenchmarkError("invalid derived study parent")
        verify_benchmark(parent_path, record["parent"]["sha256"])
        for role in record["inherited_roles"]:
            if (record["artifacts"][role] != parent["artifacts"][role]
                    or record["logical_snapshots"][role] != parent["logical_snapshots"][role]):
                raise BenchmarkError("inherited artifact identity changed")
    checked = {}
    for name, expected in record["artifacts"].items():
        target = artifact_path(path, record, name)
        if (name not in record.get("inherited_roles", [])
                and not target.resolve().is_relative_to(path.parent.resolve())):
            raise BenchmarkError(f"benchmark artifact escaped its directory: {name}")
        if target not in checked:
            physical = directory_digest(target, suffixes=(".parquet",))
            if physical != expected:
                raise BenchmarkError(f"benchmark artifact changed: {name}")
            checked[target] = physical, parquet_content_hash(target)
        if checked[target] != (expected, record["logical_snapshots"][name]):
            raise BenchmarkError(f"benchmark artifact changed: {name}")
    for key, item in record["grids"].items():
        target = path.parent / item["path"]
        if (
            not target.resolve().is_relative_to(path.parent.resolve())
            or directory_digest(target, suffixes=(".parquet",)) != item["sha256"]
            or parquet_content_hash(target) != record["grid_identities"][key]["content_sha256"]
            or item["logical_sha256"] != record["grid_identities"][key]["logical_sha256"]
            or item["n_cells"] != record["grid_identities"][key]["n_cells"]
        ):
            raise BenchmarkError("benchmark scoring grid changed")
    for key, item in record.get("coverage", {}).items():
        target = path.parent / item["path"]
        if (not target.resolve().is_relative_to(path.parent.resolve())
                or directory_digest(target, suffixes=(".parquet",)) != item["sha256"]
                or parquet_content_hash(target)
                != record["coverage_identities"][key]["content_sha256"]):
            raise BenchmarkError("study scoring coverage changed")
    if (
        hashlib.sha256(read_path_bytes(path.parent / "dataset.yaml")).hexdigest()
        != record["dataset_sha256"]
    ):
        raise BenchmarkError("prepared dataset contract changed")
    return path, record


def evaluation_contract(
    path: pathlib.Path, *, stage: str, origin: str, horizon: int, metric: str
) -> dict:
    """Bind a single evaluation boundary to the independently frozen population."""
    _, record = verify_benchmark(path)
    return _evaluation_contract_from_record(
        path, record, stage=stage, origin=origin, horizon=horizon, metric=metric,
    )


def _evaluation_contract_from_record(
    path: pathlib.Path, record: dict, *, stage: str, origin: str, horizon: int, metric: str,
) -> dict:
    """Derive a boundary from the record authenticated by the current compilation."""
    if stage == "validation" and not record["recipe"]["validation_origins"]:
        raise BenchmarkError("outer validation is not configured; use a study with validation data")
    cadence = _recipe_cadence(record["recipe"])
    key = boundary(stage, origin, horizon, cadence.code)
    if key not in record["grids"]:
        raise BenchmarkError("evaluation boundary absent from frozen benchmark")
    path = path.resolve()
    if metric != record["recipe"]["metric"]:
        raise BenchmarkError("metric differs from the frozen study")
    role = _actuals_role(stage)
    contract = {
        "schema": "autoforecast.fixed-evaluation.v2",
        "benchmark_id": record["benchmark_id"], "study_id": record["study_id"],
        "study": {"path": str(path.resolve()),
                  "sha256": hashlib.sha256(read_path_bytes(path)).hexdigest()},
        "boundary": key,
        "grid": {**record["grids"][key],
                 "path": str(path.parent / record["grids"][key]["path"])},
        "metric_policy": record["metric_definition"],
        "actuals": {"role": role, "path": str(artifact_path(path, record, role)),
                    "sha256": record["artifacts"][role]},
        "coverage": {**record["coverage"][key],
                     **record["coverage_identities"][key],
                     "path": str(path.parent / record["coverage"][key]["path"])},
    }
    if "execution_policy" in record:
        policy = record["execution_policy"]
        contract["availability"] = {
            "series_policy": policy["series_policy"],
            "original_series": len(policy["original_ids"]),
            "retained_series": len(policy["retained_ids"]),
            "cohort_sha256": policy["cohort_sha256"],
            **policy["windows"][key], "interpretation": policy["interpretation"],
        }
    return contract
