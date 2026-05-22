"""Freeze a time-series benchmark before any candidate can choose its scoring population."""

from __future__ import annotations

import errno
import hashlib
import heapq
import json
import pathlib
import shutil
import tempfile
from contextlib import ExitStack
from functools import partial

from autoforecast._exact_json import loads_exact_json
from autoforecast.column_names import validate_column_names
from autoforecast.data_admission import load_dataset_contract
from autoforecast.frequency import (
    Cadence,
    FrequencyError,
    TimePoint,
    advance,
    canonical_origin,
    parse_frequency,
    parse_origin,
)
from provenance.lineage import directory_digest

SCHEMA = "autoforecast.frozen-benchmark.v1"


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


def _inclusive_period_points(
    start: TimePoint, end: TimePoint, cadence: Cadence, *, label: str
) -> tuple[TimePoint, ...]:
    """Return a bounded inclusive cadence grid and reject endpoints on another phase."""
    if start > end:
        raise BenchmarkError(f"{label} start must not follow its end")
    points: list[TimePoint] = []
    index = 0
    try:
        while True:
            point = advance(start, cadence, index)
            if point > end:
                break
            points.append(point)
            if point == end:
                return tuple(points)
            index += 1
    except (FrequencyError, OverflowError) as error:
        raise BenchmarkError(f"{label} is outside the representable cadence grid") from error
    raise BenchmarkError(f"{label} endpoints are not aligned to frequency {cadence.code}")


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


def _parquet_hash(path: pathlib.Path) -> str:
    """Hash typed rows with bounded sort buffers, preserving the original v1 identity.

    Each run holds at most 65,536 32-byte digests; merge passes open at most 64 runs.
    Arrow batches hold at most 8,192 rows. Temporary runs are removed on success or failure.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
    columns = sorted(dataset.schema.names)
    schema = [(name, str(dataset.schema.field(name).type)) for name in columns]
    hashed = hashlib.sha256(json.dumps(schema).encode())
    with tempfile.TemporaryDirectory(prefix="autoforecast-row-hashes-") as temporary:
        root = pathlib.Path(temporary)
        rows: list[bytes] = []
        run_count = 0
        for batch in dataset.scanner(
            columns=columns, batch_size=_ROW_HASH_BATCH_SIZE,
            batch_readahead=1, fragment_readahead=1, use_threads=False,
        ).to_batches():
            for row in batch.to_pylist():
                payload = json.dumps(
                    row, sort_keys=True, separators=(",", ":"), default=str,
                ).encode()
                rows.append(hashlib.sha256(payload).digest())
                if len(rows) == _ROW_HASH_CHUNK_SIZE:
                    _write_hash_run(root / f"0-{run_count}", rows)
                    run_count += 1
                    rows.clear()
        if rows:
            _write_hash_run(root / f"0-{run_count}", rows)
            run_count += 1
            rows.clear()
        level = 0
        while run_count > 1:
            merged_count = 0
            for first in range(0, run_count, _ROW_HASH_MERGE_FAN_IN):
                paths = [root / f"{level}-{index}" for index in range(
                    first, min(first + _ROW_HASH_MERGE_FAN_IN, run_count),
                )]
                _merge_hash_runs(paths, root / f"{level + 1}-{merged_count}")
                merged_count += 1
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
    return {
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


def validate_recipe(recipe: dict) -> None:
    """Validate exact preparation fields before acquiring a runtime."""
    expected = {
        "schema_version",
        "source_keys",
        "training_start",
        "validation_origins",
        "validation_horizon",
        "final_train_cutoff",
        "final_test_horizon",
        "history_policy",
        "new_series_policy",
        "target_unit",
        "feature_availability",
    }
    accepted_fields = (expected, expected | {"frequency"})
    if (type(recipe) is not dict or set(recipe) not in accepted_fields
            or type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1):
        raise BenchmarkError("preparation recipe requires exactly the documented v1 fields")
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
            not origins
            or origins != sorted(set(origins))
            or start > origins[0]
            or any(type(h) is not int or h < 1 for h in (horizon, test_horizon))
            or advance(origins[-1], cadence, horizon) > cutoff
        )
        advance(cutoff, cadence, test_horizon)
    except (FrequencyError, OverflowError):
        invalid = True
    if invalid:
        raise BenchmarkError("invalid training/validation/testing boundaries")
    for label, point in [
        *(('validation window', origin) for origin in origins),
        ("final training window", cutoff),
    ]:
        _inclusive_period_points(start, point, cadence, label=label)


def freeze_benchmark(
    *,
    contract_path: pathlib.Path,
    sources: dict[str, pathlib.Path],
    recipe: dict,
    output_root: pathlib.Path,
    spark,
) -> pathlib.Path:
    """Validate source scope, normalize keys, and publish immutable role snapshots and grids.

    Population: all distinct source keys observed by the earliest validation origin.
    New series can only be excluded by the explicit pre-test rule in the recipe.
    Every retained series must have every cadence observation in each required role.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DateType,
        IntegralType,
        StringType,
    )

    from autoforecast import grid

    validate_recipe(recipe)
    cadence = _recipe_cadence(recipe)
    if set(sources) != {"selection", "final_training", "final_score"}:
        raise BenchmarkError("provide selection, final_training, and final_score sources")
    keys = recipe["source_keys"]
    start = _recipe_point(recipe["training_start"], cadence)
    origins = [_recipe_point(day, cadence) for day in recipe["validation_origins"]]
    cutoff = _recipe_point(recipe["final_train_cutoff"], cadence)
    horizon, test_horizon = recipe["validation_horizon"], recipe["final_test_horizon"]
    end = advance(cutoff, cadence, test_horizon)
    contract_bytes = contract_path.read_bytes()
    contract = load_dataset_contract(contract_path)
    panel = contract.panel(sources["selection"])
    if (
        parse_frequency(panel.frequency).code != cadence.code
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
            *keys, panel.time_col, panel.target_col, *panel.known_future,
            *panel.static, *panel.static_reals, *panel.past_only,
            *availability.values(), *([panel.exposure_col] if panel.exposure_col else []),
        ])))
    except ValueError as error:
        raise BenchmarkError(str(error)) from error
    if not isinstance(availability, dict) or set(availability) != set(panel.known_future):
        raise BenchmarkError("every known_future feature needs an availability-evidence column")
    if panel.known_future and not panel.exposure_col:
        raise BenchmarkError("future features require exposure_col for existing target masking")
    for path in sources.values():
        contract.panel(path).validate()
    before = {
        name: directory_digest(path, suffixes=(".parquet",)) for name, path in sources.items()
    }
    frames = {}
    key_types = None
    internal = "__af_series_id"
    for name, path in sources.items():
        frame = spark.read.parquet(str(path))
        try:
            validate_column_names(frame.columns)
        except ValueError as error:
            raise BenchmarkError(str(error)) from error
        if any(column.casefold().startswith("__af_") for column in frame.columns):
            raise BenchmarkError("source columns beginning __af_ are reserved for preparation")
        if any(k not in frame.columns for k in keys):
            raise BenchmarkError("source keys absent")
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
    cohort = selection.filter(
        F.col(panel.time_col) <= grid.time_literal(origins[0], cadence)
    ).select(internal).distinct()
    all_keys = frames["selection"].select(internal)
    for frame in (frames["final_training"], frames["final_score"]):
        all_keys = all_keys.unionByName(frame.select(internal))
    original_count, final_count = all_keys.distinct().count(), cohort.count()
    if final_count == 0:
        raise BenchmarkError("pre-test cohort is empty")
    excluded = original_count - final_count
    if excluded and recipe["new_series_policy"] == "error":
        raise BenchmarkError(f"{excluded} new series absent at the first validation origin")
    key_map = selection.join(cohort, internal).select(internal, *keys).distinct()
    counts = {}
    for name, frame in frames.items():
        retained = frame.join(cohort, internal)
        low, high = (
            (advance(cutoff, cadence, 1), end)
            if name == "final_score"
            else (start, origins[-1] if name == "selection" else cutoff)
        )
        upper = end if panel.known_future and name != "final_score" else high
        window = retained.filter(
            F.col(panel.time_col).between(
                grid.time_literal(low, cadence), grid.time_literal(upper, cadence)
            )
        )
        accepted_points = _inclusive_period_points(low, upper, cadence, label=f"{name} role")
        periods = spark.createDataFrame(
            [(canonical_origin(point, cadence),) for point in accepted_points],
            "__af_time_text string",
        ).select(
            F.col("__af_time_text")
            .cast("timestamp_ntz" if cadence.key_kind == "timestamp" else "date")
            .alias(panel.time_col)
        )
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
        absent = expected_cells.join(required, [internal, panel.time_col], "left_anti")
        if absent.limit(1).count():
            bounds = required.groupBy(internal).agg(F.max(panel.time_col).alias("__af_last"))
            gaps = absent.join(bounds, internal, "left")
            trailing = gaps.filter(
                F.col("__af_last").isNull() | (F.col(panel.time_col) > F.col("__af_last"))
            ).count()
            total = absent.count()
            raise BenchmarkError(
                f"{name}: missing observations: internal/leading={total - trailing}, "
                f"trailing={trailing}; unknown zeros refused"
            )
        target = F.col(panel.target_col).cast("double")
        if (
            required.filter(
                target.isNull() | F.isnan(target) | (target < 0) | (F.abs(target) == float("inf"))
            )
            .limit(1)
            .count()
        ):
            raise BenchmarkError(f"{name}: null, negative, or non-finite observed target")
        if (
            panel.exposure_col
            and required.filter(
                F.col(panel.exposure_col).isNull() | (F.col(panel.exposure_col) != "exact")
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
            first_origin = grid.time_literal(origins[0], cadence).cast("timestamp_ntz")
            if (
                known_at not in retained.columns
                or retained.filter(
                    known_time.isNull() | (known_time > first_origin)
                )
                .limit(1)
                .count()
            ):
                raise BenchmarkError(f"unknown future feature at first origin: {feature}")
        # Keep verified future-known rows for the existing physical target-mask materializer.
        frames[name] = window
        counts[name] = {
            "required_observations": required.count(),
            "written_rows": frames[name].count(),
        }
    if recipe["history_policy"] == "fixed":
        last_origin = grid.time_literal(origins[-1], cadence)
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
        "schema": SCHEMA,
        "recipe": recipe,
        "key_types": key_types,
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "source_series_count": original_count,
        "excluded_new_series": excluded,
        "cohort_series_count": final_count,
    }

    def authenticate_sources():
        if contract_path.read_bytes() != contract_bytes or any(
            directory_digest(path, suffixes=(".parquet",)) != before[name]
            for name, path in sources.items()
        ):
            raise BenchmarkError("source changed during preparation")

    authenticate_sources()
    output_root.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix=".freeze-", dir=output_root))
    try:
        import yaml

        roles = {**contract.panel_roles, "series_col": internal}
        (scratch / "dataset.yaml").write_text(
            yaml.safe_dump({"schema_version": 1, "dataset": {"id": contract.id, "panel": roles}})
        )
        key_map.write.parquet(str(scratch / "source-keys"))
        for name, frame in frames.items():
            frame.write.parquet(str(scratch / name))
        grids = {}
        for stage, origin, span in [("validation", o, horizon) for o in origins] + [
            ("test", cutoff, test_horizon)
        ]:
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
        authenticate_sources()
        identity.update(
            logical_snapshots={
                name: _parquet_hash(scratch / name) for name in (*sources, "source-keys")
            },
            dataset_sha256=hashlib.sha256((scratch / "dataset.yaml").read_bytes()).hexdigest(),
            grid_identities={
                key: {
                    "content_sha256": _parquet_hash(scratch / item["path"]),
                    "logical_sha256": item["logical_sha256"],
                    "n_cells": item["n_cells"],
                }
                for key, item in grids.items()
            },
        )
        benchmark_id = digest(identity)
        destination = output_root / benchmark_id
        record = {
            **identity,
            "benchmark_id": benchmark_id,
            "grids": grids,
            "counts": counts,
            "policies": {
                "absent_rows": "error",
                "null_targets": "error",
                "confirmed_zero": "observed",
                "late_feeds": "error",
                "retirement": "error",
            },
            "sources": {
                name: {"path": str(path.resolve()), "sha256": before[name]}
                for name, path in sources.items()
            },
            "artifacts": {
                name: directory_digest(scratch / name, suffixes=(".parquet",))
                for name in (*sources, "source-keys")
            },
            "dataset_sha256": hashlib.sha256((scratch / "dataset.yaml").read_bytes()).hexdigest(),
        }
        (scratch / "benchmark.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        _publish_frozen_benchmark(scratch, destination, benchmark_id)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    return destination / "benchmark.json"


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
    mapping(record, {
        "schema", "recipe", "key_types", "contract_sha256", "logical_snapshots",
        "source_series_count", "excluded_new_series", "cohort_series_count",
        "dataset_sha256", "grid_identities", "benchmark_id", "grids", "counts",
        "policies", "sources", "artifacts",
    }, "manifest")
    if record["schema"] != SCHEMA:
        raise BenchmarkError("unsupported benchmark schema")
    recipe = record["recipe"]
    validate_recipe(recipe)
    for name in ("benchmark_id", "contract_sha256", "dataset_sha256"):
        sha256(record[name], name)
    types = mapping(record["key_types"], recipe["source_keys"], "key_types")
    if any(type(value) is not str or value not in {"tinyint", "smallint", "int", "bigint", "string"}
           for value in types.values()):
        raise BenchmarkError("benchmark source key types must be integral or string")
    for name in ("source_series_count", "excluded_new_series", "cohort_series_count"):
        count(record[name], name, 1 if name == "cohort_series_count" else 0)
    cohort = record["cohort_series_count"]
    if record["source_series_count"] - cohort != record["excluded_new_series"]:
        raise BenchmarkError("benchmark population counts do not reconcile")
    source_roles = {"selection", "final_training", "final_score"}
    roles = source_roles | {"source-keys"}
    for name in ("artifacts", "logical_snapshots"):
        for role, value in mapping(record[name], roles, name).items():
            sha256(value, f"{name}.{role}")
    for role, value in mapping(record["sources"], source_roles, "sources").items():
        mapping(value, {"path", "sha256"}, f"sources.{role}")
        if type(value["path"]) is not str or not value["path"].strip():
            raise BenchmarkError("benchmark source path must be a nonblank string")
        sha256(value["sha256"], f"sources.{role}.sha256")
    policies = {
        "absent_rows": "error", "null_targets": "error", "confirmed_zero": "observed",
        "late_feeds": "error", "retirement": "error",
    }
    if record["policies"] != policies:
        raise BenchmarkError("benchmark observation policies changed")
    cadence = _recipe_cadence(recipe)
    start = _recipe_point(recipe["training_start"], cadence)
    cutoff = _recipe_point(recipe["final_train_cutoff"], cadence)
    selection_end = _recipe_point(recipe["validation_origins"][-1], cadence)
    role_periods = {
        "selection": len(_inclusive_period_points(
            start, selection_end, cadence, label="selection role"
        )),
        "final_training": len(_inclusive_period_points(
            start, cutoff, cadence, label="final training role"
        )),
        "final_score": recipe["final_test_horizon"],
    }
    for role, values in mapping(record["counts"], source_roles, "counts").items():
        mapping(values, {"required_observations", "written_rows"}, f"counts.{role}")
        for key, value in values.items():
            count(value, f"counts.{role}.{key}")
        if (values["required_observations"] != cohort * role_periods[role]
                or values["written_rows"] < values["required_observations"]):
            raise BenchmarkError("benchmark observation counts do not reconcile")
    boundaries = {
        boundary(
            "validation", origin, recipe["validation_horizon"], cadence.code
        ): recipe["validation_horizon"]
        for origin in recipe["validation_origins"]
    }
    boundaries[boundary(
        "test", recipe["final_train_cutoff"], recipe["final_test_horizon"], cadence.code
    )] = (
        recipe["final_test_horizon"]
    )
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
            if value != cohort * horizon:
                raise BenchmarkError("benchmark grid population count changed")
        if item["logical_sha256"] != identity["logical_sha256"]:
            raise BenchmarkError("benchmark grid logical identity changed")
    return record


def read_benchmark_manifest(
    path: pathlib.Path, expected_sha256: str | None = None,
) -> tuple[pathlib.Path, dict]:
    """Validate manifest bytes and structure without scanning frozen data artifacts."""
    if path.is_symlink():
        raise BenchmarkError("benchmark manifest must not be a symlink")
    try:
        payload = path.read_bytes()
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
    for name, expected in record["artifacts"].items():
        target = path.parent / name
        if (
            not target.resolve().is_relative_to(path.parent.resolve())
            or directory_digest(target, suffixes=(".parquet",)) != expected
            or _parquet_hash(target) != record["logical_snapshots"][name]
        ):
            raise BenchmarkError(f"benchmark artifact changed: {name}")
    for key, item in record["grids"].items():
        target = path.parent / item["path"]
        if (
            not target.resolve().is_relative_to(path.parent.resolve())
            or directory_digest(target, suffixes=(".parquet",)) != item["sha256"]
            or _parquet_hash(target) != record["grid_identities"][key]["content_sha256"]
            or item["logical_sha256"] != record["grid_identities"][key]["logical_sha256"]
            or item["n_cells"] != record["grid_identities"][key]["n_cells"]
        ):
            raise BenchmarkError("benchmark scoring grid changed")
    if (
        hashlib.sha256((path.parent / "dataset.yaml").read_bytes()).hexdigest()
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
    cadence = _recipe_cadence(record["recipe"])
    key = boundary(stage, origin, horizon, cadence.code)
    if key not in record["grids"]:
        raise BenchmarkError("evaluation boundary absent from frozen benchmark")
    if metric not in {"rmse", "mae", "wape"}:
        raise BenchmarkError("generic fixed benchmark supports rmse, mae, or wape; M5 owns WRMSSE")
    return {
        "schema": "autoforecast.fixed-evaluation.v1",
        "benchmark_id": record["benchmark_id"],
        "boundary": key,
        "grid": {**record["grids"][key], "path": str(path.parent / record["grids"][key]["path"])},
        "metric_policy": {"metric": metric, "missing_actual": "error", "fallback": "none"},
    }
