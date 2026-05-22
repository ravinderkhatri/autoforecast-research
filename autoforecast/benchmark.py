"""Freeze a daily benchmark before any candidate can choose its scoring population."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import shutil
import tempfile

from autoforecast._exact_json import loads_exact_json
from autoforecast.data_admission import load_dataset_contract
from provenance.lineage import directory_digest

SCHEMA = "autoforecast.frozen-benchmark.v1"


class BenchmarkError(ValueError):
    """The requested population cannot be frozen without changing its meaning."""


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def boundary(stage: str, origin: str, horizon: int, frequency: str = "D") -> str:
    return f"{stage}/{origin}/{horizon}/{frequency}"


def _parquet_hash(path: pathlib.Path) -> str:
    """Hash typed row values, independent of Parquet names, order, and partitioning."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
    columns = sorted(dataset.schema.names)
    schema = [(name, str(dataset.schema.field(name).type)) for name in columns]
    hashed = hashlib.sha256(json.dumps(schema).encode())
    rows = []
    for batch in dataset.scanner(columns=columns).to_batches():
        for row in batch.to_pylist():
            payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode()
            rows.append(hashlib.sha256(payload).digest())
    for value in sorted(rows):
        hashed.update(value)
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
    """Validate exact daily preparation fields before acquiring a runtime."""
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
    if (type(recipe) is not dict or set(recipe) != expected
            or type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1):
        raise BenchmarkError("preparation recipe requires exactly the documented v1 fields")
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
    def date(value):
        if type(value) is not str:
            raise BenchmarkError("recipe dates must be ISO YYYY-MM-DD strings")
        try:
            parsed = dt.date.fromisoformat(value)
        except ValueError as error:
            raise BenchmarkError("recipe dates must be ISO YYYY-MM-DD strings") from error
        if parsed.isoformat() != value:
            raise BenchmarkError("recipe dates must be ISO YYYY-MM-DD strings")
        return parsed

    if type(recipe["validation_origins"]) is not list:
        raise BenchmarkError("validation_origins must be a list of ISO dates")
    if type(recipe["feature_availability"]) is not dict or any(
        type(key) is not str or not key.strip() or type(value) is not str or not value.strip()
        for key, value in recipe["feature_availability"].items()
    ):
        raise BenchmarkError("feature_availability must map features to evidence column names")
    start = date(recipe["training_start"])
    origins = [date(day) for day in recipe["validation_origins"]]
    cutoff = date(recipe["final_train_cutoff"])
    horizon, test_horizon = recipe["validation_horizon"], recipe["final_test_horizon"]
    if (
        not origins
        or origins != sorted(set(origins))
        or start > origins[0]
        or any(type(h) is not int or h < 1 for h in (horizon, test_horizon))
        or horizon > (cutoff - origins[-1]).days
        or test_horizon > (dt.date.max - cutoff).days
    ):
        raise BenchmarkError("invalid daily training/validation/testing boundaries")


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
    Every retained series must have every daily observation in each required role.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DateType, IntegralType, StringType

    from autoforecast import grid

    validate_recipe(recipe)
    if set(sources) != {"selection", "final_training", "final_score"}:
        raise BenchmarkError("provide selection, final_training, and final_score sources")
    keys = recipe["source_keys"]
    start = dt.date.fromisoformat(recipe["training_start"])
    origins = [dt.date.fromisoformat(day) for day in recipe["validation_origins"]]
    cutoff = dt.date.fromisoformat(recipe["final_train_cutoff"])
    horizon, test_horizon = recipe["validation_horizon"], recipe["final_test_horizon"]
    end = cutoff + dt.timedelta(days=test_horizon)
    contract_bytes = contract_path.read_bytes()
    contract = load_dataset_contract(contract_path)
    panel = contract.panel(sources["selection"])
    if (
        panel.frequency != "D"
        or panel.target_estimand not in ("sales_given_availability", "observed_nonnegative")
        or panel.negative_target_policy != "refuse"
        or panel.status_prob_cols
        or panel.plan
    ):
        raise BenchmarkError(
            "release supports daily nonnegative observed targets and prepared snapshots"
        )
    availability = recipe["feature_availability"]
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
        if internal in frame.columns or any(k not in frame.columns for k in keys):
            raise BenchmarkError("source keys absent or reserved __af_series_id already present")
        types = {key: frame.schema[key].dataType.simpleString() for key in keys}
        if key_types is not None and key_types != types:
            raise BenchmarkError("source key types differ between roles")
        key_types = types
        if any(not isinstance(frame.schema[k].dataType, IntegralType | StringType) for k in keys):
            raise BenchmarkError("source keys must be integral or string, without lossy casts")
        if not isinstance(frame.schema[panel.time_col].dataType, DateType):
            raise BenchmarkError("source dates must be Parquet date values")
        invalid = F.col(panel.time_col).isNull()
        for key in keys:
            invalid = invalid | F.col(key).isNull()
        if frame.filter(invalid).limit(1).count():
            raise BenchmarkError("source contains null key/date")
        frame = frame.withColumn(internal, F.to_json(F.struct(*[F.col(k) for k in keys])))
        if frame.groupBy(internal, panel.time_col).count().filter("count != 1").limit(1).count():
            raise BenchmarkError("duplicate source observations")
        frames[name] = frame
    selection = frames["selection"]
    cohort = (
        selection.filter(F.col(panel.time_col) <= F.lit(origins[0])).select(internal).distinct()
    )
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
            (cutoff + dt.timedelta(days=1), end)
            if name == "final_score"
            else (start, origins[-1] if name == "selection" else cutoff)
        )
        required = retained.filter(F.col(panel.time_col).between(low, high))
        days = spark.range((high - low).days + 1).select(
            F.date_add(F.lit(low), F.col("id").cast("int")).alias(panel.time_col)
        )
        expected_cells = cohort.crossJoin(days)
        absent = expected_cells.join(required, [internal, panel.time_col], "left_anti")
        if absent.limit(1).count():
            bounds = required.groupBy(internal).agg(F.max(panel.time_col).alias("last"))
            gaps = absent.join(bounds, internal, "left")
            trailing = gaps.filter(
                F.col("last").isNull() | (F.col(panel.time_col) > F.col("last"))
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
            if (
                known_at not in retained.columns
                or retained.filter(
                    F.to_timestamp(F.col(known_at)).isNull()
                    | (
                        F.to_timestamp(F.col(known_at))
                        > F.lit(dt.datetime.combine(origins[0], dt.time.min))
                    )
                )
                .limit(1)
                .count()
            ):
                raise BenchmarkError(f"unknown future feature at first origin: {feature}")
        # Keep verified future-known rows for the existing physical target-mask materializer.
        frames[name] = retained.filter(
            F.col(panel.time_col).between(
                low, end if panel.known_future and name != "final_score" else high
            )
        )
        counts[name] = {
            "required_observations": required.count(),
            "written_rows": frames[name].count(),
        }
    if recipe["history_policy"] == "fixed":
        left = frames["selection"].filter(F.col(panel.time_col) <= F.lit(origins[-1]))
        right = frames["final_training"].filter(F.col(panel.time_col) <= F.lit(origins[-1]))
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
            key = boundary(stage, origin.isoformat(), span)
            relative = "grids/" + key
            path = grid.build(
                scratch / "source-keys",
                origin.isoformat(),
                span,
                spark,
                scratch / relative,
                series_col=internal,
                time_col=panel.time_col,
            )
            grids[key] = {
                "path": relative,
                "logical_sha256": grid.content_hash(path, spark),
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
        if destination.exists():
            return verify_benchmark(destination / "benchmark.json")[0]
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
        scratch.rename(destination)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    return destination / "benchmark.json"


def verify_benchmark(
    path: pathlib.Path, expected_sha256: str | None = None
) -> tuple[pathlib.Path, dict]:
    """Authenticate frozen outputs; original mutable sources are recorded but never re-read."""
    payload = path.read_bytes()
    if path.is_symlink() or (
        expected_sha256 and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise BenchmarkError("benchmark manifest changed")
    record = loads_exact_json(payload)
    if type(record) is not dict:
        raise BenchmarkError("benchmark manifest must be an object")
    if record.get("schema") != SCHEMA:
        raise BenchmarkError("unsupported benchmark schema")
    if digest(_identity(record)) != record["benchmark_id"]:
        raise BenchmarkError("benchmark logical identity changed")
    roles = {"selection", "final_training", "final_score", "source-keys"}
    recipe = record["recipe"]
    boundaries = {
        boundary("validation", origin, recipe["validation_horizon"])
        for origin in recipe["validation_origins"]
    } | {boundary("test", recipe["final_train_cutoff"], recipe["final_test_horizon"])}
    if (
        set(record["artifacts"]) != roles
        or set(record["logical_snapshots"]) != roles
        or set(record["grids"]) != boundaries
        or set(record["grid_identities"]) != boundaries
    ):
        raise BenchmarkError("benchmark role or boundary inventory changed")
    for name, expected in record["artifacts"].items():
        target = (path.parent / name).resolve()
        if (
            not target.is_relative_to(path.parent.resolve())
            or directory_digest(target, suffixes=(".parquet",)) != expected
            or _parquet_hash(target) != record["logical_snapshots"][name]
        ):
            raise BenchmarkError(f"benchmark artifact changed: {name}")
    for key, item in record["grids"].items():
        target = (path.parent / item["path"]).resolve()
        if (
            not target.is_relative_to(path.parent.resolve())
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
    key = boundary(stage, origin, horizon)
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
