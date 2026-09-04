"""Register once, then load verified studies without reopening their source SQL.

The existing benchmark freezer owns calendars, cohorts and study identities.
This coordinator binds that artifact to source provenance and an optional Delta
snapshot. The local registration baseline must be protected by storage ACLs.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import pathlib
import shutil
import tempfile
import uuid
from contextlib import contextmanager, suppress
from typing import Any, Protocol

from mlops._safe_files import publish_bytes, read_regular_bytes
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from autoforecast._exact_yaml import loads_record, read_record, record_bytes
from autoforecast.benchmark import (
    START_COLUMN,
    _normalise_recipe,
    _normalise_reference_frame,
    _reference_boundaries,
    artifact_path,
    digest,
    freeze_benchmark,
    parquet_content_hash,
    study_windows,
    validate_study_panel,
    verify_benchmark,
)
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.data_observability import report_source_issues
from autoforecast.frequency import canonical_origin, parse_frequency, parse_origin
from autoforecast.grid import time_literal
from autoforecast.study import load_study
from autoforecast.study_references import (
    check_wrmsse_references,
    prepare_wrmsse_references,
    validate_reference_settings,
)
from provenance.lineage import directory_digest

EXCLUSION_REASON_COLUMN = "__af_exclusion_reason"


@dataclasses.dataclass(frozen=True)
class PreparationSettings:
    """Captured settings. Each mapping property returns a detached value."""

    _payload: bytes

    @property
    def dataset(self) -> dict:
        return json.loads(self._payload)["dataset"]

    @property
    def recipe(self) -> dict:
        return json.loads(self._payload)["recipe"]

    @property
    def parameters(self) -> dict:
        return json.loads(self._payload)["parameters"]

    @property
    def windows(self) -> tuple:
        return tuple(study_windows(self.recipe))


@dataclasses.dataclass(frozen=True)
class PreparationContext:
    """Managed snapshots and factual findings supplied to trusted application code."""

    settings: PreparationSettings
    observations: DataFrame
    audits: dict[str, DataFrame]
    starting_keys: DataFrame
    findings: DataFrame
    scratch_directory: pathlib.Path


@dataclasses.dataclass(frozen=True)
class PreparationResult:
    """Application decisions, with typed key frames and explicit exclusion reasons.

    Key frames have exactly the configured source-key columns. Exclusions add a
    nonblank string ``EXCLUSION_REASON_COLUMN``. Optional starts add a canonical
    string ``START_COLUMN``. Both use the existing reserved source-column prefix.
    Findings use the supplied factual findings schema. Policies contain named
    scalar decisions; large populations remain Parquet artifacts.
    """

    observations: DataFrame
    starting_keys: DataFrame
    retained_keys: DataFrame
    exclusions: DataFrame
    identifier: str
    version: str
    policy: dict
    actions: tuple[str, ...] = ()
    planned_actions: tuple[str, ...] = ()
    findings: DataFrame | None = None
    starts: DataFrame | None = None
    reference_input: DataFrame | None = None
    scoring_references: DataFrame | None = None


class StudyPreparation(Protocol):
    """Validate domain settings without I/O, then prepare managed input snapshots."""

    def validate(self, settings: PreparationSettings) -> None: ...

    def prepare(self, context: PreparationContext) -> PreparationResult: ...


def _bytes(value) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode()


def _fields(value, required, optional=(), *, label) -> dict:
    if not isinstance(value, dict) or not set(required) <= value.keys() or (
        value.keys() - set(required) - set(optional)
    ):
        raise ValueError(f"{label} requires {sorted(required)}; optional: {sorted(optional)}")
    return value


def _configuration(path):
    path = pathlib.Path(path).absolute()
    value = read_record(path)
    value = _fields(value, {"schema_version", "dataset", "recipe", "source", "storage"},
                    {"references", "invalid_references", "runtime", "preparation", "records"},
                    label="study config")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("study config requires schema_version: 1")
    # All scientific validation happens before acquiring source data or provisioning tables.
    value["recipe"] = _normalise_recipe(value["recipe"])
    contract = _load_dataset_contract_bytes(_bytes(value["dataset"]), path)
    panel = contract.panel(".")
    validate_study_panel(panel, value["recipe"])
    source = value["source"]
    if isinstance(source, dict) and source.get("kind") == "parquet":
        _fields(source, {"kind", "path"}, label="Parquet source")
        source["path"] = str((path.parent / source["path"]).absolute())
    elif isinstance(source, dict) and source.get("kind") == "delta":
        _fields(source, {"kind", "sql", "tables"},
                {"audit_queries", "population", "require_matching_population"},
                label="Delta source")
        source["sql"] = (path.parent / source["sql"]).read_text()
        if not isinstance(source["tables"], dict) or not source["tables"]:
            raise ValueError("Delta source requires alias-to-table mappings")
        audits = source.get("audit_queries", {})
        if (not isinstance(audits, dict) or any(not name.isidentifier() or name == "primary"
                                               for name in audits)):
            raise ValueError("audit query names must be identifiers other than primary")
        source["audit_queries"] = {name: (path.parent / sql).read_text()
                                   for name, sql in audits.items()}
        if source.get("population", "primary") not in {"primary", *audits}:
            raise ValueError("population must name a managed query")
        if type(source.get("require_matching_population", True)) is not bool:
            raise ValueError("require_matching_population must be boolean")
    else:
        raise ValueError("source.kind must be parquet or delta")
    if value.get("invalid_references", "error") not in {"error", "exclude"}:
        raise ValueError("invalid_references must be error or exclude")
    if (value["recipe"]["metric"] == "wrmsse" or value["recipe"].get("reference_metrics")) and (
        "references" not in value
    ):
        raise ValueError("WRMSSE requires configured scoring references")
    if "references" in value:
        if value["recipe"]["frequency"] != "D":
            raise ValueError("registration reference reporting currently requires daily studies")
        refs = value["references"]
        if "prepared" in refs:
            _fields(refs, {"prepared"}, label="references")
            if refs["prepared"] is not True:
                raise ValueError("references.prepared must be true")
        elif "path" in refs:
            _fields(refs, {"path"}, label="references")
            refs["path"] = str((path.parent / refs["path"]).absolute())
        else:
            _fields(refs, {"weight_col", "missing_history", "duplicate_policy"},
                    {"weight_window_days", "scale_lag", "scale_start"}, label="references")
            validate_reference_settings(
                series_id=value["recipe"]["source_keys"], date_col=panel.time_col,
                target_col=panel.target_col, **refs,
            )
    _runtime(value.get("runtime", {}))
    _fields(value.get("records", {}), (), {"format"}, label="records")
    if value.get("records", {}).get("format", "json") not in {"json", "yaml"}:
        raise ValueError("records.format must be json or yaml")
    _scalar_settings(value.get("preparation", {}), "preparation")
    _fields(value["storage"], {"root"}, {"scratch_root", "namespace", "tables"}, label="storage")
    return value, path.parent


def _scalar_settings(value, label):
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key.strip()
        or (item is not None and type(item) not in (str, bool, int, float))
        for key, item in value.items()
    ):
        raise ValueError(f"{label} requires named scalar settings")
    _bytes(value)
    return value


def _runtime(value):
    _fields(value, (), {"tracking", "publish", "log_level", "tracking_uri", "mlflow_experiment"},
            label="runtime")
    if value.get("tracking", "none") not in {"none", "mlflow"}:
        raise ValueError("tracking must be none or mlflow")
    if type(value.get("publish", False)) is not bool:
        raise ValueError("publish must be boolean")
    level = value.get("log_level", "INFO")
    if not isinstance(level, str) or level not in logging.getLevelNamesMapping():
        raise ValueError("log_level must be a standard logging level name")
    for name in ("tracking_uri", "mlflow_experiment"):
        if value.get(name) is not None and (
            not isinstance(value[name], str) or not value[name].strip()
        ):
            raise ValueError(f"{name} must be a nonblank string or null")
    return dict(value)


def _record_path(root, name):
    paths = [root / (name + suffix) for suffix in (".json", ".yaml", ".yml")]
    present = [p for p in paths if p.exists() or p.is_symlink()]
    if len(present) > 1:
        raise ValueError("conflicting JSON and YAML record copies")
    return present[0] if present else None


def _read_saved(path):
    value = loads_record(read_regular_bytes(path.parent, path.name),
                         format="json" if path.suffix == ".json" else "yaml")
    if not isinstance(value, dict):
        raise ValueError("saved records require a mapping")
    return value


@contextmanager
def _publication_lock(root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".registration.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _artifact_record(path):
    return {"logical_sha256": parquet_content_hash(path),
            "physical_sha256": directory_digest(path, suffixes=(".parquet",))}


def _verify_admission(root, binding):
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256", "physical_sha256"}
            or pathlib.Path(binding["path"]).is_absolute()
            or ".." in pathlib.Path(binding["path"]).parts):
        raise ValueError("invalid admission record binding")
    path = root / binding["path"]
    if _record_path(path.parent, path.stem) != path:
        raise ValueError("admission record is missing or ambiguous")
    payload = read_regular_bytes(path.parent, path.name)
    if hashlib.sha256(payload).hexdigest() != binding["physical_sha256"]:
        raise ValueError("admission record bytes differ from their baseline")
    record = _read_saved(path)
    if digest(record) != binding["sha256"]:
        raise ValueError("admission record differs from its baseline")
    for name, expected in record["artifacts"].items():
        if not name.isidentifier() or _artifact_record(path.parent / name) != expected:
            raise ValueError("admission artifact differs from its baseline")
    return record


def _save_admission(registry, staging, identity, frozen, *, format):
    """Publish immutable evidence once; retries retain the first bytes and encoding."""
    study_id = frozen["study_id"]
    if _record_path(registry.root / "registrations", study_id) is not None:
        registry.definition(study_id)
    final_keys = registry.spark.read.parquet(str(artifact_path(
        registry.root / "benchmarks" / study_id / "benchmark.json", frozen, "source-keys",
    ))).select(*frozen["recipe"]["source_keys"])
    final_keys.write.parquet(str(staging / "final_keys"))
    starting = registry.spark.read.parquet(str(staging / "starting_keys"))
    retained = registry.spark.read.parquet(str(staging / "retained_keys"))
    keys = frozen["recipe"]["source_keys"]
    _same_keys(final_keys, retained.join(final_keys, keys, "inner"), keys, "final cohort")
    native_exclusions = retained.join(final_keys, keys, "left_anti")
    reasons = []
    for field, reason in (("excluded_keys", "INVALID_REFERENCE"),
                          ("excluded_new_keys", "NEW_SERIES_POLICY")):
        values = identity["native_policy"][field]
        frame = (registry.spark.createDataFrame(values, retained.schema) if values
                 else retained.limit(0))
        reasons.append(frame.withColumn(EXCLUSION_REASON_COLUMN, F.lit(reason)))
    native_reasons = reasons[0].unionByName(reasons[1])
    _same_keys(native_exclusions, native_reasons, keys, "native exclusion reasons")
    native_reasons.write.parquet(str(staging / "native_exclusions"))
    excluded = registry.spark.read.parquet(str(staging / "exclusions")).select(*keys)
    _same_keys(starting, final_keys.unionByName(excluded).unionByName(native_exclusions),
               keys, "final population")
    record = {"schema": "autoforecast.admission.v1", "identity": identity,
              "study_id": study_id,
              "outcome": {"starting": starting.count(), "application_retained": retained.count(),
                          "final_retained": final_keys.count(),
                          "native_admission": frozen["recipe"]},
              "artifacts": {p.name: _artifact_record(p) for p in sorted(staging.iterdir())
                            if p.is_dir()}}
    destination = registry.root / "admissions" / study_id
    with _publication_lock(registry.root):
        existing = _record_path(destination, "admission")
        if existing is None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("incomplete admission destination requires explicit recovery")
            (staging / ("admission." + format)).write_bytes(record_bytes(record, format=format))
            os.rename(staging, destination)
            existing = destination / ("admission." + format)
        else:
            old = _read_saved(existing)
            def comparable(item):
                return {**item, "artifacts": {
                    name: value["logical_sha256"] for name, value in item["artifacts"].items()
                }}
            if comparable(old) != comparable(record):
                raise ValueError("conflicting immutable admission evidence")
            record = old
        binding = {"path": str(existing.relative_to(registry.root)), "sha256": digest(record),
                   "physical_sha256": hashlib.sha256(
                       read_regular_bytes(existing.parent, existing.name)).hexdigest()}
        _verify_admission(registry.root, binding)
    return binding


def _same_keys(left, right, keys, label):
    if ([(k, left.schema[k].dataType) for k in keys]
            != [(k, right.schema[k].dataType) for k in keys]
            or left.select(*keys).exceptAll(right.select(*keys)).limit(1).count()
            or right.select(*keys).exceptAll(left.select(*keys)).limit(1).count()):
        raise ValueError(f"{label} has different exact membership")


def _check_key_frame(frame, template, keys, *, extra=(), label):
    if not isinstance(frame, DataFrame) or set(frame.columns) != {*keys, *extra}:
        raise ValueError(f"{label} requires exact key columns and {list(extra)}")
    if any(frame.schema[k].dataType != template.schema[k].dataType for k in keys):
        raise ValueError(f"{label} key types differ from the managed population")
    invalid = F.lit(False)
    for key in keys:
        invalid = invalid | F.col(key).isNull()
    counts = frame.agg(F.count("*").alias("rows"),
                       F.count_distinct(F.struct(*keys)).alias("keys"),
                       F.max(invalid.cast("int")).alias("nulls")).first()
    assert counts is not None  # An ungrouped aggregate always returns one row.
    if counts["rows"] != counts["keys"] or counts["nulls"]:
        raise ValueError(f"{label} requires unique nonnull keys")
    for name in extra:
        if (frame.schema[name].dataType.typeName() != "string"
                or frame.filter(F.col(name).isNull()
                                | (F.trim(F.col(name)) == "")).limit(1).count()):
            raise ValueError(f"{label}.{name} requires nonblank strings")


def _save_findings(registry, frame, issues_root, name, check_id):
    frame.write.parquet(str(issues_root / name))
    saved = registry.spark.read.parquet(str(issues_root / name))
    if registry.store is not None:
        registry.store.append_once("study_data_issues", saved.select(
            F.lit(check_id).alias("check_id"), F.lit(None).cast("string").alias("study_id"),
            F.to_json(F.struct("*")).alias("issue_json"),
        ), scope={"check_id": check_id})
    return saved


def _prepare_observations(hook, settings, frame, audits, population, findings, *,
                          value, root, staging, registry, issues_root, check_id):
    """Capture callback outputs, then reconcile against private managed baselines."""
    keys = value["recipe"]["source_keys"]
    _check_key_frame(population, population, keys, label="managed population")
    population.write.parquet(str(staging / "starting_keys"))
    frame.select(*keys).distinct().write.parquet(str(staging / "input_keys"))
    population = registry.spark.read.parquet(str(staging / "starting_keys"))
    primary_keys = registry.spark.read.parquet(str(staging / "input_keys"))
    if any(population.schema[k].dataType != primary_keys.schema[k].dataType for k in keys):
        raise ValueError("managed population and primary keys have different types")
    matching = value["source"].get("require_matching_population", True)
    population_findings = findings.limit(0)
    for name, left, right, code in (
        ("input_added_keys", primary_keys, population, "INPUT_POPULATION_ADDITION"),
        ("input_missing_keys", population, primary_keys, "INPUT_POPULATION_LOSS"),
    ):
        difference = left.join(right, keys, "left_anti")
        difference.write.parquet(str(staging / name))
        columns = {"keys": F.struct(*keys), "issue_code": F.lit(code),
                   "message": F.lit("Primary membership differs from the declared population."),
                   "status": F.lit("failed" if matching else "warning"),
                   "affected_rows": F.lit(1).cast("long")}
        population_findings = population_findings.unionByName(difference.select(*[
            columns.get(field.name, F.lit(None).cast(field.dataType)).alias(field.name)
            for field in findings.schema
        ]))
    population_findings = _save_findings(registry, population_findings, issues_root,
                                          "population", check_id + "-population")
    population_findings.write.parquet(str(staging / "population_findings"))
    if matching:
        _same_keys(population, primary_keys, keys, "primary and declared population")
    findings = findings.unionByName(population_findings)
    findings.write.parquet(str(staging / "source_findings"))
    paths = [root / "source", *(root / ("audit-" + name) for name in audits),
             *(staging / name for name in ("starting_keys", "input_keys", "source_findings",
                                           "population_findings", "input_added_keys",
                                           "input_missing_keys"))]
    baselines = {p: directory_digest(p, suffixes=(".parquet",)) for p in paths}
    scratch = root / "application"
    scratch.mkdir()
    parameters = settings.parameters
    settings_baseline = settings._payload
    result = hook.prepare(PreparationContext(
        settings, frame, dict(audits), population,
        registry.spark.read.parquet(str(staging / "source_findings")), scratch,
    ))
    if not isinstance(result, PreparationResult):
        raise ValueError("preparation must return PreparationResult")
    if result.findings is not None:
        if [(f.name, f.dataType) for f in result.findings.schema] != [
            (f.name, f.dataType) for f in findings.schema
        ]:
            raise ValueError("application findings must use the native findings schema")
        app_findings = _save_findings(registry, result.findings, issues_root,
                                      "application", check_id + "-application")
        app_findings.write.parquet(str(staging / "application_findings"))
        if app_findings.filter(
            F.col("status").isNull() | ~F.col("status").isin("passed", "warning", "failed")
            | F.col("affected_rows").isNull() | (F.col("affected_rows") < 0)
        ).limit(1).count():
            raise ValueError("invalid application finding status or count")
        if app_findings.filter(F.col("status") == "failed").limit(1).count():
            raise ValueError("application findings block registration")
    for name in ("identifier", "version"):
        if not isinstance(getattr(result, name), str) or not getattr(result, name).strip():
            raise ValueError(f"preparation {name} must be nonblank")
    policy = json.loads(_bytes(_scalar_settings(result.policy, "preparation policy")))
    for name in ("actions", "planned_actions"):
        actions = getattr(result, name)
        if not isinstance(actions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in actions
        ):
            raise ValueError("preparation actions require tuples of nonblank strings")
    metadata = {"identifier": result.identifier, "version": result.version,
                "parameters": parameters, "policy": policy, "actions": list(result.actions),
                "planned_actions": list(result.planned_actions)}
    frames = {"prepared": result.observations, "declared_starting": result.starting_keys,
              "retained_keys": result.retained_keys, "exclusions": result.exclusions}
    for name in ("starts", "reference_input", "scoring_references"):
        if getattr(result, name) is not None:
            frames[name] = getattr(result, name)
    for name, output in frames.items():
        if not isinstance(output, DataFrame):
            raise ValueError(f"preparation {name} must be a Spark DataFrame")
        output.write.parquet(str(staging / name))
    captured = {name: registry.spark.read.parquet(str(staging / name)) for name in frames}
    output_baselines = {staging / name: _artifact_record(staging / name) for name in frames}
    if settings._payload != settings_baseline or any(
        directory_digest(p, suffixes=(".parquet",)) != sha for p, sha in baselines.items()
    ):
        raise ValueError("managed source or population changed during preparation")
    for name in ("declared_starting", "retained_keys", "exclusions"):
        _check_key_frame(captured[name], population, keys,
                         extra=(EXCLUSION_REASON_COLUMN,) if name == "exclusions" else (),
                         label=name)
    _same_keys(population, captured["declared_starting"], keys, "declared starting population")
    _same_keys(population, captured["retained_keys"].unionByName(
        captured["exclusions"].select(*keys)), keys, "application admission")
    _same_keys(captured["prepared"].select(*keys).distinct(), captured["retained_keys"],
               keys, "prepared observations")
    cadence = parse_frequency(value["recipe"]["frequency"])
    panel = value["dataset"]["dataset"]["panel"]
    lower = time_literal(parse_origin(value["recipe"]["training_start"], cadence), cadence)
    upper = time_literal(parse_origin(value["recipe"]["validation_origins"][0], cadence), cadence)
    history_keys = captured["prepared"].filter(
        F.col(panel["time_col"]).between(lower, upper),
    ).select(*keys).distinct()
    if "starts" in captured:
        _check_key_frame(captured["starts"], population, keys,
                         extra=(START_COLUMN,), label="starts")
        _same_keys(captured["starts"], captured["retained_keys"], keys, "declared starts")
        declarations = []
        marked = history_keys.withColumn("__af_initial_cohort", F.lit(True))
        for row in captured["starts"].join(marked, keys, "left").collect():
            entry = {"keys": {k: row[k] for k in keys},
                     "start": canonical_origin(parse_origin(row[START_COLUMN], cadence), cadence)}
            if row["__af_initial_cohort"]:
                declarations.append(entry)
        value["recipe"]["series_starts"] = {"rule": "declared", "values": sorted(
            declarations, key=lambda item: _bytes(item["keys"]))}
    else:
        starts = value["recipe"]["series_starts"]
        if starts["rule"] == "declared" and starts["values"]:
            schema = StructType([*(population.schema[k] for k in keys),
                                 StructField(START_COLUMN, StringType(), False)])
            declared = registry.spark.createDataFrame([
                (*[entry["keys"][k] for k in keys], entry["start"])
                for entry in starts["values"]
            ], schema).join(captured["exclusions"].select(*keys), keys, "left_anti")
            _same_keys(declared, history_keys, keys, "declared starts")
            starts["values"] = sorted([
                {"keys": {k: row[k] for k in keys}, "start": row[START_COLUMN]}
                for row in declared.collect()
            ], key=lambda item: _bytes(item["keys"]))
    if "reference_input" in captured and "scoring_references" in captured:
        raise ValueError("provide reference input or precomputed references, not both")
    if any(_artifact_record(p) != baseline for p, baseline in output_baselines.items()):
        raise ValueError("prepared materialization changed during admission checks")
    return captured, metadata


class StudyRegistry:
    """One configured artifact root, optionally bound to five shared Delta tables.

    Construction and reads do not create directories or tables. Artifact/scratch
    paths require filesystem visibility to Spark workers and POSIX atomic file
    publication. Object-store URLs are not filesystem paths.
    """

    def __init__(self, root, *, spark: SparkSession | None = None, scratch_root=None,
                 namespace=None, tables=None,
                 runtime=None):
        for location in (root, scratch_root):
            if location is not None and (not str(location).strip() or "://" in str(location)):
                raise ValueError("registry/scratch roots require nonempty POSIX filesystem paths")
        self.root = pathlib.Path(root).expanduser().absolute()
        session = spark if spark is not None else SparkSession.getActiveSession()
        if session is None:
            raise ValueError("provide or activate a Spark session")
        self.spark: SparkSession = session
        self.scratch_root = pathlib.Path(scratch_root or self.root / "scratch").absolute()
        self.runtime = _runtime(runtime or {})
        self.store = None
        if namespace is not None:
            from autoforecast.integrations.delta import DeltaStudyStore

            self.store = DeltaStudyStore(self.spark, namespace=namespace, tables=tables)
        elif tables is not None:
            raise ValueError("table overrides require a Delta namespace")

    @classmethod
    def from_config(cls, config, *, spark=None):
        """Read only storage/runtime settings; never read the source SQL file."""
        path = pathlib.Path(config).absolute()
        value = _fields(read_record(path), {"storage"},
                        {"schema_version", "dataset", "recipe", "source", "references",
                         "invalid_references", "runtime", "preparation", "records"},
                        label="registry config")
        return cls._from_storage(value["storage"], value.get("runtime", {}), path.parent, spark)

    @classmethod
    def _from_storage(cls, storage, runtime, base, spark, *, use_environment=False):
        _fields(storage, set(), {"root", "scratch_root", "namespace", "tables"}, label="storage")
        storage = dict(storage)
        if use_environment:
            for name, key in (("REGISTRY_ROOT", "root"), ("SCRATCH_ROOT", "scratch_root"),
                              ("DELTA_NAMESPACE", "namespace")):
                if "AUTOFORECAST_" + name in os.environ:
                    value = os.environ["AUTOFORECAST_" + name]
                    if not value.strip() or (key != "namespace" and "://" in value):
                        raise ValueError(
                            "storage environment requires nonempty POSIX paths/namespace"
                        )
                    storage[key] = (str(pathlib.Path(value).expanduser().absolute())
                                    if key != "namespace" else value)
        _fields(storage, {"root"}, {"scratch_root", "namespace", "tables"}, label="storage")
        settings = dict(storage)
        for key in ("root", "scratch_root"):
            if key in settings:
                if not str(settings[key]).strip() or "://" in str(settings[key]):
                    raise ValueError("storage roots require nonempty POSIX filesystem paths")
                settings[key] = base / settings[key]
        return cls(**settings, runtime=runtime, spark=spark)

    @classmethod
    def from_environment(cls, *, spark=None):
        """Resolve explicit storage environment variables or AUTOFORECAST_CONFIG."""
        storage, runtime, base = {}, {}, pathlib.Path.cwd()
        if os.environ.get("AUTOFORECAST_CONFIG"):
            path = pathlib.Path(os.environ["AUTOFORECAST_CONFIG"]).absolute()
            value = _fields(read_record(path), {"storage"},
                            {"schema_version", "dataset", "recipe", "source", "references",
                             "invalid_references", "runtime", "preparation", "records"},
                            label="registry config")
            storage, runtime, base = dict(value["storage"]), value.get("runtime", {}), path.parent
        if "root" not in storage and "AUTOFORECAST_REGISTRY_ROOT" not in os.environ:
            raise ValueError(
                "configure AUTOFORECAST_REGISTRY_ROOT or AUTOFORECAST_CONFIG, or registry=",
            )
        return cls._from_storage(storage, runtime, base, spark, use_environment=True)

    def _registration_path(self, study_id):
        if not isinstance(study_id, str) or not study_id or pathlib.Path(study_id).name != study_id:
            raise ValueError("invalid study ID")
        path = _record_path(self.root / "registrations", study_id)
        if path is None:
            raise FileNotFoundError(f"Unknown study_id: {study_id}")
        return path

    def definition(self, study_id):
        """Authenticate the saved binding and pinned snapshot without source access."""
        path = self._registration_path(study_id)
        envelope = _read_saved(path)
        definition = envelope["definition"]
        if envelope["sha256"] != digest(definition) or definition["study_id"] != study_id:
            raise ValueError("registered definition identity differs from its baseline")
        if definition["schema"] != "autoforecast.registration.v1":
            raise ValueError("unsupported registration format")
        if bool(definition["snapshot"]) != (self.store is not None):
            raise ValueError("registry storage does not match the registered definition")
        if self.store is not None:
            rows, _ = self.store.read("studies", {"study_id": study_id})
            rows = rows.collect()
            expected = (study_id, envelope["sha256"], _bytes(definition).decode())
            if len(rows) != 1 or tuple(rows[0]) != expected:
                raise ValueError("registered table definition differs from its protected baseline")
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            self.store.verify_observations(definition["snapshot"], scratch_root=self.scratch_root)
        manifest = self.root / "benchmarks" / study_id / "benchmark.json"
        _, record = verify_benchmark(manifest, definition["library_sha256"])
        if record["study_id"] != study_id:
            raise ValueError("registered library study identity differs")
        if "admission_evidence" in definition:
            admission = _verify_admission(self.root, definition["admission_evidence"])
            if (record.get("admission_identity") != {
                "schema": "autoforecast.admission-identity.v1",
                "sha256": digest(admission["identity"]),
            } or admission["study_id"] != record.get("parent", {}).get("study_id", study_id)):
                raise ValueError("admission evidence does not belong to the frozen study")
        elif "admission_identity" in record:
            raise ValueError("registered study is missing required admission evidence")
        return definition, envelope["sha256"]

    def load_study(self, study_id):
        """Return the existing Study context manager after all registry checks pass."""
        definition, sha = self.definition(study_id)
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        study = load_study(study_id, registry_root=self.root / "benchmarks", spark=self.spark,
                           scratch_root=self.scratch_root)
        if study._sha256 != definition["library_sha256"]:
            with suppress(Exception):
                study.close()
            raise ValueError("library manifest changed during registered loading")
        return dataclasses.replace(study, _registration=_bytes({"definition": definition,
                                                               "sha256": sha}))

    def _publish_definition(self, definition, *, format="json"):
        study_id = definition["study_id"]
        sha = digest(definition)
        baseline = self.root / "registrations"
        baseline.mkdir(parents=True, exist_ok=True)
        envelope = {"definition": definition, "sha256": sha}
        with _publication_lock(self.root):
            existing = _record_path(baseline, study_id)
            if existing is not None:
                if _read_saved(existing) != envelope:
                    raise ValueError("conflicting immutable study registration")
            if self.store is not None:
                from autoforecast.integrations.delta import SCHEMAS

                self.store.append_once("studies", self.spark.createDataFrame(
                    [(study_id, sha, _bytes(definition).decode())], SCHEMAS["studies"],
                ), scope={"study_id": study_id})
            if existing is None:
                publish_bytes(baseline, study_id + "." + format,
                              record_bytes(envelope, format=format))
        return study_id

    def retain_derived(self, parent_id, manifest):
        """Register an existing preflight-derived identity, reusing its parent snapshot."""
        parent, parent_sha = self.definition(parent_id)
        raw = pathlib.Path(manifest).read_bytes()
        _, record = verify_benchmark(pathlib.Path(manifest), hashlib.sha256(raw).hexdigest())
        expected = self.root / "benchmarks" / record["study_id"] / "benchmark.json"
        if pathlib.Path(manifest).resolve() != expected.resolve():
            raise ValueError("derived study must be created within the configured registry")
        if record.get("parent", {}).get("sha256") != parent["library_sha256"]:
            raise ValueError("derived study does not belong to this registered parent")
        definition = {**parent, "study_id": record["study_id"],
                      "library_sha256": hashlib.sha256(raw).hexdigest(),
                      "parent": {"study_id": parent_id, "study_sha256": parent_sha},
                      "effective_series": record["cohort_series_count"],
                      "execution_policy": record["execution_policy"]}
        format = "json" if self._registration_path(parent_id).suffix == ".json" else "yaml"
        return self._publish_definition(definition, format=format)

    def publish_results(self, study_id, workflow_path, *, mlflow_links=None):
        """Publish already checked evidence; retries never fit or rescore."""
        if self.store is None:
            raise ValueError("result publication requires a configured Delta registry")
        definition, sha = self.definition(study_id)
        return self.store.publish_results(
            definition, sha, workflow_path, scratch_root=self.scratch_root,
            mlflow_links=mlflow_links,
        )


def register_study(config, *, registry=None, spark=None,
                   prepare: StudyPreparation | None = None) -> str:
    """Validate, report and freeze one study; never fit a model or start tracking."""
    logger = logging.getLogger(__name__)
    check_id = uuid.uuid4().hex
    logger.info("configuration START attempt=%s", check_id)
    try:
        value, base = _configuration(config)
        settings = PreparationSettings(_bytes({
            "dataset": value["dataset"], "recipe": value["recipe"],
            "parameters": value.get("preparation", {}),
        }))
        if prepare is not None:
            settings_baseline = settings._payload
            prepare.validate(settings)
            if settings._payload != settings_baseline:
                raise ValueError("preparation validation changed captured settings")
        elif value.get("preparation") or value.get("references", {}).get("prepared"):
            raise ValueError("preparation settings require an explicit preparation hook")
        elif value["source"].get("audit_queries"):
            raise ValueError("managed audit queries require an explicit preparation hook")
        registry = registry or StudyRegistry._from_storage(
            value["storage"], value.get("runtime", {}), base, spark, use_environment=True,
        )
        if value["source"]["kind"] == "delta":
            from autoforecast.integrations.delta import validate_source_queries

            validate_source_queries(registry.spark, queries={
                "primary": value["source"]["sql"], **value["source"]["audit_queries"],
            }, tables=value["source"]["tables"])
    except Exception:
        logger.error("configuration FAIL attempt=%s", check_id)
        raise
    registry.root.mkdir(parents=True, exist_ok=True)
    registry.scratch_root.mkdir(parents=True, exist_ok=True)
    issues_root = registry.root / "checks" / check_id
    issues_root.mkdir(parents=True)
    phase = "configuration"

    def event(stage, status, **facts):
        nonlocal phase
        phase = stage
        entry = {"attempt_id": check_id, "stage": stage, "status": status,
                 "evidence": str(issues_root), **facts}
        logger.info("%s %s attempt=%s %s", stage, status, check_id,
                    " ".join(f"{name}={item}" for name, item in facts.items()))
        with (issues_root / "audit.jsonl").open("ab") as stream:
            stream.write(_bytes(entry))

    event("configuration", "START")
    event("configuration", "PASS")
    format = value.get("records", {}).get("format", "json")
    try:
        return _register_study(value, registry, prepare, settings, check_id, issues_root, event)
    except Exception as error:
        logger.error("%s FAIL attempt=%s exception=%s", phase, check_id, type(error).__name__)
        try:
            event(phase, "FAIL", exception=type(error).__name__)
            (issues_root / ("attempt." + format)).write_bytes(record_bytes(
                {"status": "failed", "stage": phase, "exception": type(error).__name__},
                format=format,
            ))
        except Exception:
            logger.error("failure evidence could not be saved attempt=%s", check_id)
            error.add_note("Registration failure evidence could not be saved.")
        raise


def _register_study(value, registry, prepare, preparation_settings, check_id, issues_root, event):
    configuration_baseline = preparation_settings._payload
    session = registry.spark
    event("sources", "START")
    if registry.store is not None:
        registry.store.prepare(lock_path=registry.root / ".delta-provision.lock")
    with tempfile.TemporaryDirectory(prefix="af-register-", dir=registry.scratch_root) as temporary:
        root = pathlib.Path(temporary)
        contract_path = root / "dataset.yaml"
        contract_path.write_bytes(_bytes(value["dataset"]))
        contract = _load_dataset_contract_bytes(contract_path.read_bytes(), contract_path)
        recipe = value["recipe"]
        source = value["source"]
        source_record: dict[str, Any]
        audits = {}
        query_records = {}
        if source["kind"] == "delta":
            from autoforecast.integrations.delta import read_sources

            managed = read_sources(session, queries={"primary": source["sql"],
                                   **source["audit_queries"]}, tables=source["tables"])
            frame, source_record = managed.pop("primary")
            if prepare is None:
                source_record.pop("resolved_tables")
            for name, (audit, provenance) in managed.items():
                path = root / ("audit-" + name)
                audit.write.parquet(str(path))
                audits[name] = session.read.parquet(str(path))
                query_records[name] = {**provenance, "logical_sha256": parquet_content_hash(path),
                                       "schema": audits[name].schema.jsonValue(),
                                       "rows": audits[name].count()}
        else:
            frame = session.read.parquet(source["path"])
            source_record = {"kind": "parquet"}
        frame.write.parquet(str(root / "source"))
        frame = session.read.parquet(str(root / "source"))
        source_record.update({"logical_sha256": parquet_content_hash(root / "source"),
                              "schema": frame.schema.jsonValue(), "rows": frame.count()})
        keys = recipe["source_keys"]
        panel = contract.panel(root / "source")
        source_record["starting_series"] = frame.select(*keys).distinct().count()
        event("sources", "PASS", rows=source_record["rows"],
              starting_series=source_record["starting_series"])
        event("admission", "START")
        cadence = parse_frequency(recipe["frequency"])
        start = time_literal(parse_origin(recipe["training_start"], cadence), cadence)
        origin = time_literal(parse_origin(recipe["validation_origins"][0], cadence), cadence)
        findings = report_source_issues(frame, contract=contract, recipe=recipe)
        findings = _save_findings(registry, findings, issues_root, "source", check_id)
        if findings.filter(F.col("status") == "failed").limit(1).count():
            raise ValueError(f"structural source issues block registration: {issues_root}")
        admission_identity = None
        captured = {}
        staging = issues_root / "admission-staging"
        preparation_metadata = None
        prepared_baseline = None
        if prepare is not None:
            staging.mkdir()
            population_name = source.get("population", "primary")
            population_frame = frame if population_name == "primary" else audits[population_name]
            population = population_frame.select(*keys).distinct()
            captured, preparation_metadata = _prepare_observations(
                prepare, preparation_settings, frame, audits, population, findings,
                value=value, root=root,
                staging=staging, registry=registry, issues_root=issues_root, check_id=check_id,
            )
            frame = captured["prepared"]
            prepared_baseline = _artifact_record(staging / "prepared")
            # Apply declared application eligibility before native cohort-count checks.
            if contract.required_series_count is not None:
                original_candidates = session.read.parquet(str(root / "source")).filter(
                    F.col(panel.time_col).between(start, origin),
                ).select(*keys).distinct()
                if original_candidates.count() != contract.required_series_count:
                    raise ValueError(
                        "source cohort differs from required_series_count before preparation")
                value["dataset"]["dataset"]["required_series_count"] = (
                    frame.filter(F.col(panel.time_col).between(start, origin))
                    .select(*keys).distinct().count())
                contract_path.write_bytes(_bytes(value["dataset"]))
                contract = _load_dataset_contract_bytes(contract_path.read_bytes(), contract_path)
            prepared_report = _save_findings(
                registry, report_source_issues(frame, contract=contract, recipe=recipe),
                issues_root, "prepared", check_id + "-prepared",
            )
            prepared_report.write.parquet(str(staging / "prepared_findings"))
            if prepared_report.filter(F.col("status") == "failed").limit(1).count():
                raise ValueError("structural prepared-data issues block registration")
            if captured.get("scoring_references") is not None and not value.get(
                "references", {}).get("prepared"):
                raise ValueError("precomputed references require references.prepared")
            if captured.get("reference_input") is not None and (
                "references" not in value or "path" in value["references"]
                or "prepared" in value["references"]
            ):
                raise ValueError("prepared reference input requires reference calculation settings")
        if recipe["series_starts"]["rule"] == "study_lower_bound":
            starts = frame.filter(F.col(panel.time_col).between(start, origin)).groupBy(*keys).agg(
                F.min(panel.time_col).alias("_start"),
            )
            recipe["series_starts"] = {"rule": "declared", "values": sorted([
                {"keys": {key: row[key] for key in keys},
                 "start": canonical_origin(row["_start"], cadence)}
                for row in starts.collect()
            ], key=lambda item: _bytes(item["keys"]))}
        candidate_keys = frame.filter(F.col(panel.time_col).between(start, origin)).select(
            *keys,
        ).distinct()
        source_record["eligible_series"] = candidate_keys.count()
        if (contract.required_series_count is not None
                and contract.required_series_count != source_record["eligible_series"]):
            raise ValueError("source cohort differs from required_series_count before exclusions")
        new_keys = [row.asDict() for row in frame.select(*keys).distinct().join(
            candidate_keys, keys, "left_anti",
        ).collect()]
        if new_keys and recipe["new_series_policy"] == "error":
            raise ValueError("new series absent at the first validation origin")
        reference_source = frame.join(candidate_keys, keys, "inner")
        references = None
        exclusions = []
        reference_definition = None
        exclusion_findings = []
        if "references" in value:
            settings = value["references"]
            origins = [(stage, dt.date.fromisoformat(origin))
                       for stage, origin, _ in study_windows(recipe)]
            if settings.get("prepared") and "scoring_references" not in captured:
                raise ValueError("preparation did not supply configured scoring references")
            if "reference_input" in captured:
                reference_source = captured["reference_input"].join(candidate_keys, keys, "inner")
            refs = (captured["scoring_references"] if settings.get("prepared") else
                    session.read.parquet(settings["path"]) if "path" in settings else
                    prepare_wrmsse_references(
                        reference_source, series_id=keys, date_col=panel.time_col,
                        target_col=panel.target_col,
                        origins=origins, validate=False, **settings,
                    ))
            refs.write.parquet(str(root / "reference-input"))
            refs = session.read.parquet(str(root / "reference-input"))
            reference_definition = {
                "preparation": settings if "path" not in settings else {"kind": "parquet"},
                "logical_sha256": parquet_content_hash(root / "reference-input"),
            }
            key_map = candidate_keys.withColumn("__af_series_id", F.to_json(F.struct(*keys)))
            normalized = _normalise_reference_frame(refs, key_map, keys)
            if new_keys:
                late_keys = frame.select(*keys).distinct().join(candidate_keys, keys, "left_anti")
                late_ids = late_keys.select(F.to_json(F.struct(*keys)).alias("series_id"))
                normalized = normalized.join(late_ids, "series_id", "left_anti")
                refs = refs.withColumn("__af_reference_id", F.to_json(F.struct(*[
                    F.col("keys")[key].alias(key) for key in keys
                ]))).join(late_ids.withColumnRenamed("series_id", "__af_reference_id"),
                          "__af_reference_id", "left_anti").drop("__af_reference_id")
            _reference_boundaries(normalized, key_map, recipe, session)
            report = check_wrmsse_references(refs, origins=origins)
            report.write.parquet(str(issues_root / "references"))
            if prepare is not None:
                report.write.parquet(str(staging / "reference_findings"))
            if registry.store is not None:
                registry.store.append_once("study_data_issues", report.select(
                    F.lit(check_id + "-references").alias("check_id"),
                    F.lit(None).cast("string").alias("study_id"),
                    F.to_json(F.struct("*")).alias("issue_json"),
                ), scope={"check_id": check_id + "-references"})
            if report.limit(1).count():
                if value.get("invalid_references", "error") == "error":
                    raise ValueError(f"invalid scoring references; findings: {issues_root}")
                excluded = report.filter(F.col("keys").isNotNull()).select("keys").distinct()
                exclusions = [r.asDict(recursive=True)["keys"] for r in excluded.collect()]
                exclusion_findings = [json.loads(row["json"]) for row in report.select(
                    "keys", "stage", "origin", "field", "issue_code",
                    F.col("observed_value").cast("string").alias("observed_value"),
                ).select(F.to_json(F.struct("*")).alias("json")).collect()]
                reference_definition["excluded_references"] = sorted([
                    json.loads(row["json"]) for row in refs.join(
                    excluded, "keys", "inner",
                ).select("keys", "stage", "origin", F.col("weight").cast("string"),
                         F.col("scale").cast("string")).select(
                             F.to_json(F.struct("*")).alias("json"),
                         ).collect()], key=_bytes)
                refs = refs.join(excluded, "keys", "left_anti")
                if check_wrmsse_references(refs, origins=origins).limit(1).count():
                    raise ValueError("retained scoring references have no valid positive total")
                excluded = excluded.select("keys.*")
                frame = frame.join(excluded, keys, "left_anti")
                if recipe["series_starts"]["rule"] == "declared":
                    recipe["series_starts"]["values"] = [
                        item for item in recipe["series_starts"]["values"]
                        if item["keys"] not in exclusions
                    ]
            references = root / "references"
            refs.write.parquet(str(references))
        retained = frame.filter(F.col(panel.time_col).between(start, origin)).select(
            *keys,
        ).distinct().count()
        if retained == 0:
            raise ValueError("registration has no retained series")
        if exclusions and contract.required_series_count is not None:
            value["dataset"]["dataset"]["required_series_count"] = retained
            contract_path.write_bytes(_bytes(value["dataset"]))
        frame.write.parquet(str(root / "admitted"))
        if (prepared_baseline is not None
                and _artifact_record(staging / "prepared") != prepared_baseline):
            raise ValueError("prepared observations changed before freezing")
        if prepare is not None:
            admission_identity = {
                "schema": "autoforecast.preparation-evidence.v1",
                "configuration": json.loads(configuration_baseline),
                "sources": {"primary": source_record, **query_records},
                "population_query": source.get("population", "primary"),
                "require_matching_population": source.get("require_matching_population", True),
                "preparation": preparation_metadata, "references": reference_definition,
                "native_policy": {"invalid_references": value.get("invalid_references", "error"),
                                  "excluded_keys": sorted(exclusions, key=_bytes),
                                  "findings": sorted(exclusion_findings, key=_bytes),
                                  "excluded_new_keys": sorted(new_keys, key=_bytes)},
                "captured_inputs": {
                    p.name: {"logical_sha256": parquet_content_hash(p),
                             "schema": session.read.parquet(str(p)).schema.jsonValue()}
                    for p in sorted(staging.iterdir()) if p.is_dir()
                },
            }
        event("admission", "PASS", retained_series=retained, excluded_references=len(exclusions))
        event("freeze", "START")
        manifest = freeze_benchmark(
            contract_path=contract_path, sources=dict.fromkeys(
                ("selection", "final_training", "validation_score", "final_score"),
                root / "admitted",
            ), recipe=recipe, output_root=registry.root / "benchmarks", spark=session,
            scoring_references=references, scratch_root=registry.scratch_root,
            admission_identity=({"schema": "autoforecast.admission-identity.v1",
                                 "sha256": digest(admission_identity)}
                                if admission_identity is not None else None),
        )
        snapshot = (registry.store.freeze_observations(
            root / ("admitted" if prepare is not None else "source"),
            keys=keys, scratch_root=registry.scratch_root,
        ) if registry.store is not None else None)
        library_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        _, frozen = verify_benchmark(manifest, library_sha)
        event("freeze", "PASS", study_id=frozen["study_id"])
        definition = {
            "schema": "autoforecast.registration.v1", "study_id": manifest.parent.name,
            "library_sha256": library_sha,
            "source": source_record, "snapshot": snapshot, "recipe": frozen["recipe"],
            "references": reference_definition,
            "effective_series": retained,
            "admission": {"invalid_references": value.get("invalid_references", "error"),
                          "excluded_keys": sorted(exclusions, key=lambda item: _bytes(item)),
                          "findings": sorted(exclusion_findings, key=_bytes),
                          "excluded_new_keys": sorted(new_keys, key=_bytes),
                          "retained_series": retained},
        }
        event("publication", "START", study_id=frozen["study_id"])
        format = value.get("records", {}).get("format", "json")
        if admission_identity is not None:
            # Source/preparation hashes attest to captured inputs. Frozen model rows
            # live in the existing benchmark; do not retain duplicate raw inputs.
            for name in ("prepared", "declared_starting", "reference_input", "scoring_references"):
                if (staging / name).exists():
                    shutil.rmtree(staging / name)
            definition["admission_evidence"] = _save_admission(
                registry, staging, admission_identity, frozen, format=format,
            )
        if registry.store is not None:
            for report_id in (check_id, check_id + "-references", check_id + "-application",
                              check_id + "-prepared", check_id + "-population"):
                registry.store.link_issues(report_id, definition["study_id"])
        (issues_root / ("registration." + format)).write_bytes(
            record_bytes({"study_id": definition["study_id"]}, format=format))
        study_id = registry._publish_definition(definition, format=format)
        registry.definition(study_id)
        if staging.exists():
            shutil.rmtree(staging)
        event("publication", "PASS", study_id=study_id, retained_series=retained)
        return study_id
