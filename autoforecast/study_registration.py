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
import tempfile
import uuid
from contextlib import suppress

from mlops._safe_files import SafeFileError, publish_bytes, read_regular_bytes
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast._exact_yaml import loads_exact_yaml
from autoforecast.benchmark import (
    _normalise_recipe,
    _normalise_reference_frame,
    _reference_boundaries,
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
    value = loads_exact_yaml(path.read_bytes())
    value = _fields(value, {"schema_version", "dataset", "recipe", "source", "storage"},
                    {"references", "invalid_references", "runtime"}, label="study config")
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
        _fields(source, {"kind", "sql", "tables"}, label="Delta source")
        source["sql"] = (path.parent / source["sql"]).read_text()
        if not isinstance(source["tables"], dict) or not source["tables"]:
            raise ValueError("Delta source requires alias-to-table mappings")
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
        if "path" in refs:
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
    _fields(value["storage"], {"root"}, {"scratch_root", "namespace", "tables"}, label="storage")
    return value, path.parent


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


class StudyRegistry:
    """One configured artifact root, optionally bound to five shared Delta tables.

    Construction and reads do not create directories or tables. Artifact/scratch
    paths require filesystem visibility to Spark workers and POSIX atomic file
    publication. Object-store URLs are not filesystem paths.
    """

    def __init__(self, root, *, spark=None, scratch_root=None, namespace=None, tables=None,
                 runtime=None):
        for location in (root, scratch_root):
            if location is not None and (not str(location).strip() or "://" in str(location)):
                raise ValueError("registry/scratch roots require nonempty POSIX filesystem paths")
        self.root = pathlib.Path(root).expanduser().absolute()
        self.spark = spark if spark is not None else SparkSession.getActiveSession()
        if self.spark is None:
            raise ValueError("provide or activate a Spark session")
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
        value = _fields(loads_exact_yaml(path.read_bytes()), {"storage"},
                        {"schema_version", "dataset", "recipe", "source", "references",
                         "invalid_references", "runtime"}, label="registry config")
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
            value = _fields(loads_exact_yaml(path.read_bytes()), {"storage"},
                            {"schema_version", "dataset", "recipe", "source", "references",
                             "invalid_references", "runtime"}, label="registry config")
            storage, runtime, base = dict(value["storage"]), value.get("runtime", {}), path.parent
        if "root" not in storage and "AUTOFORECAST_REGISTRY_ROOT" not in os.environ:
            raise ValueError(
                "configure AUTOFORECAST_REGISTRY_ROOT or AUTOFORECAST_CONFIG, or registry=",
            )
        return cls._from_storage(storage, runtime, base, spark, use_environment=True)

    def _registration_path(self, study_id):
        if not isinstance(study_id, str) or not study_id or pathlib.Path(study_id).name != study_id:
            raise ValueError("invalid study ID")
        path = self.root / "registrations" / (study_id + ".json")
        if not path.exists():
            raise FileNotFoundError(f"Unknown study_id: {study_id}")
        return path

    def definition(self, study_id):
        """Authenticate the saved binding and pinned snapshot without source access."""
        path = self._registration_path(study_id)
        envelope = json.loads(read_regular_bytes(path.parent, path.name))
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

    def _publish_definition(self, definition):
        study_id = definition["study_id"]
        sha = digest(definition)
        baseline = self.root / "registrations"
        baseline.mkdir(parents=True, exist_ok=True)
        payload = _bytes({"definition": definition, "sha256": sha})
        try:
            publish_bytes(baseline, study_id + ".json", payload)
        except SafeFileError:
            if not (baseline / (study_id + ".json")).exists():
                raise
            if read_regular_bytes(baseline, study_id + ".json") != payload:
                raise ValueError("conflicting immutable study registration") from None
        if self.store is not None:
            from autoforecast.integrations.delta import SCHEMAS

            self.store.append_once("studies", self.spark.createDataFrame(
                [(study_id, sha, _bytes(definition).decode())], SCHEMAS["studies"],
            ), scope={"study_id": study_id})
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
        return self._publish_definition(definition)

    def publish_results(self, study_id, workflow_path, *, mlflow_links=None):
        """Publish already checked evidence; retries never fit or rescore."""
        if self.store is None:
            raise ValueError("result publication requires a configured Delta registry")
        definition, sha = self.definition(study_id)
        return self.store.publish_results(
            definition, sha, workflow_path, scratch_root=self.scratch_root,
            mlflow_links=mlflow_links,
        )


def register_study(config, *, registry=None, spark=None) -> str:
    """Validate, report and freeze one study; never fit a model or start tracking."""
    value, base = _configuration(config)
    registry = registry or StudyRegistry._from_storage(
        value["storage"], value.get("runtime", {}), base, spark, use_environment=True,
    )
    registry.root.mkdir(parents=True, exist_ok=True)
    registry.scratch_root.mkdir(parents=True, exist_ok=True)
    if registry.store is not None:
        registry.store.prepare(lock_path=registry.root / ".delta-provision.lock")
    session = registry.spark
    check_id = uuid.uuid4().hex
    issues_root = registry.root / "checks" / check_id
    issues_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="af-register-", dir=registry.scratch_root) as temporary:
        root = pathlib.Path(temporary)
        contract_path = root / "dataset.yaml"
        contract_path.write_bytes(_bytes(value["dataset"]))
        contract = _load_dataset_contract_bytes(contract_path.read_bytes(), contract_path)
        recipe = value["recipe"]
        source = value["source"]
        if source["kind"] == "delta":
            from autoforecast.integrations.delta import read_source

            frame, source_record = read_source(session, sql=source["sql"], tables=source["tables"])
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
        cadence = parse_frequency(recipe["frequency"])
        start = time_literal(parse_origin(recipe["training_start"], cadence), cadence)
        origin = time_literal(parse_origin(recipe["validation_origins"][0], cadence), cadence)
        findings = report_source_issues(frame, contract=contract, recipe=recipe)
        findings.write.parquet(str(issues_root / "source"))
        if registry.store is not None:
            encoded = findings.select(F.to_json(F.struct("*")).alias("issue_json"))
            registry.store.append_once("study_data_issues", encoded.select(
                F.lit(check_id).alias("check_id"), F.lit(None).cast("string").alias("study_id"),
                "issue_json",
            ), scope={"check_id": check_id})
        if findings.filter(F.col("status") == "failed").limit(1).count():
            raise ValueError(f"structural source issues block registration: {issues_root}")
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
            refs = (session.read.parquet(settings["path"]) if "path" in settings else
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
                exclusion_findings = [json.loads(row) for row in report.select(
                    "keys", "stage", "origin", "field", "issue_code",
                    F.col("observed_value").cast("string").alias("observed_value"),
                ).toJSON().collect()]
                reference_definition["excluded_references"] = sorted([
                    json.loads(row) for row in refs.join(
                    excluded, "keys", "inner",
                ).select("keys", "stage", "origin", F.col("weight").cast("string"),
                         F.col("scale").cast("string")).toJSON().collect()], key=_bytes)
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
        manifest = freeze_benchmark(
            contract_path=contract_path, sources=dict.fromkeys(
                ("selection", "final_training", "validation_score", "final_score"),
                root / "admitted",
            ), recipe=recipe, output_root=registry.root / "benchmarks", spark=session,
            scoring_references=references, scratch_root=registry.scratch_root,
        )
        snapshot = (registry.store.freeze_observations(
            root / "source", keys=keys, scratch_root=registry.scratch_root,
        ) if registry.store is not None else None)
        library_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        _, frozen = verify_benchmark(manifest, library_sha)
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
        if registry.store is not None:
            for report_id in (check_id, check_id + "-references"):
                registry.store.link_issues(report_id, definition["study_id"])
        study_id = registry._publish_definition(definition)
        (issues_root / "registration.json").write_bytes(_bytes({"study_id": study_id}))
        registry.definition(study_id)
        return study_id
