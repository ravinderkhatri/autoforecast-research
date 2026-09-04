"""Concrete Delta persistence; source access and credentials belong to the caller.

Shared tables use fixed envelopes so independent studies may have different source
column names/types. Typed data is verified after decoding, using the same bounded
Parquet content hash as the existing freezer. Delta directories are never scanned
as raw Parquet. Files still require the library's supported filesystem semantics.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, cast

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from autoforecast.benchmark import digest, parquet_content_hash

SCHEMAS = {
    "studies": "study_id string, study_sha string, definition_json string",
    "frozen_observations": "snapshot_id string, row_json string",
    "study_data_issues": "check_id string, study_id string, issue_json string",
    "study_results": (
        "study_id string, study_sha string, run_id string, stage string, model_id string, "
        "workflow_path string, workflow_sha string, evidence_json string, model_uri string, "
        "model_sha string, mlflow_uri string, additional_info string"
    ),
    "forecasts": (
        "study_id string, study_sha string, run_id string, stage string, "
        "series_id string, date string, yhat double"
    ),
}
JSON_OPTIONS = {
    "ignoreNullFields": "false", "timeZone": "UTC",
    "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
    "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    "dateFormat": "yyyy-MM-dd", "mode": "FAILFAST",
}


def quote_table(name: str) -> str:
    """Quote a multipart identifier; never accept an expression in its place."""
    if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
        raise ValueError("table names require nonempty identifier parts")
    return ".".join("`" + part.replace("`", "``") + "`" for part in name.split("."))


def latest_delta_version(spark: SparkSession, table: str) -> int:
    row = spark.sql(f"DESCRIBE HISTORY {quote_table(table)} LIMIT 1").first()
    if row is None or type(row["version"]) is not int or row["version"] < 0:
        raise ValueError("source has no valid Delta version")
    return row["version"]


def validate_source_queries(spark: SparkSession, *, queries: Mapping[str, str],
                            tables: Mapping[str, str]):
    """Validate every managed query before any source lookup or execution.

    SQL is a caller-owned SELECT over the declared aliases. Commands, external
    relations and unrecognised logical operators are rejected before execution.
    This uses Spark's parser rather than attempting to sanitize SQL with regexes.
    """
    if not tables or any(not key.isidentifier() for key in tables):
        raise ValueError("source tables require simple nonempty SQL aliases")
    if len({key.casefold() for key in tables}) != len(tables):
        raise ValueError("source aliases must be unambiguous")
    for table in tables.values():
        quote_table(table)
    if (not queries or any(not isinstance(name, str) or not name.isidentifier()
                           or not isinstance(sql, str) or not sql.strip()
                           for name, sql in queries.items())):
        raise ValueError("managed queries require names and nonempty SQL")
    session = spark.newSession()
    timezone = spark.conf.get("spark.sql.session.timeZone")
    if timezone is not None:
        session.conf.set("spark.sql.session.timeZone", timezone)
    # Catalyst is a dynamic JVM API, outside PySpark's typed Python surface.
    operators = {
        "Project", "Filter", "Join", "SubqueryAlias", "Aggregate", "Sort", "GlobalLimit",
        "LocalLimit", "Distinct", "Union", "Intersect", "Except", "UnresolvedRelation",
        "UnresolvedHaving", "Window", "RepartitionByExpression", "Repartition", "OneRowRelation",
    }
    aliases = {key.casefold() for key in tables}

    def items(sequence):
        iterator = sequence.iterator()
        while iterator.hasNext():
            yield iterator.next()

    def inspect_expression(expression):
        cls = expression.getClass()
        while cls is not None:
            if cls.getSimpleName() == "SubqueryExpression":
                inspect_plan(expression.plan())
                break
            cls = cls.getSuperclass()
        for child in items(expression.children()):
            inspect_expression(child)

    def inspect_plan(node):
        name = node.getClass().getSimpleName()
        if name not in operators:
            raise ValueError("source SQL contains an unsupported or writable operator")
        if name == "UnresolvedRelation":
            parts = list(items(node.multipartIdentifier()))
            if len(parts) != 1 or parts[0].casefold() not in aliases:
                raise ValueError("source SQL may read only declared pinned aliases")
        for expression in items(node.expressions()):
            inspect_expression(expression)
        for child in items(node.children()):
            inspect_plan(child)

    for sql in queries.values():
        plan = cast(Any, session._jsparkSession).sessionState().sqlParser().parsePlan(sql)
        inspect_plan(plan)
    return session


def read_sources(spark: SparkSession, *, queries: Mapping[str, str],
                 tables: Mapping[str, str]):
    """Execute declared queries against one version per physical Delta table."""
    session = validate_source_queries(spark, queries=queries, tables=tables)
    physical = {}
    resolved = {}
    locations = {}
    for table in dict.fromkeys(tables.values()):
        detail = spark.sql(f"DESCRIBE DETAIL {quote_table(table)}").first()
        if detail is None:
            raise ValueError("source has no Delta table identity")
        info = spark.catalog.getTable(quote_table(table))
        target = ".".join([info.catalog or spark.catalog.currentCatalog(),
                           *(info.namespace or [spark.catalog.currentDatabase()]), info.name])
        identity = (detail["id"], detail["location"])
        if identity not in physical:
            physical[identity] = latest_delta_version(spark, target)
        resolved[table] = physical[identity]
        locations[table] = {"id": detail["id"], "location": detail["location"],
                            "read_target": target}
    versions = {alias: {"table": table, "version": resolved[table]}
                for alias, table in tables.items()}
    for alias, source in versions.items():
        session.read.option("versionAsOf", source["version"]).table(
            quote_table(locations[source["table"]]["read_target"]),
        ).createOrReplaceTempView(alias)
    return {name: (session.sql(sql), {"sql": sql, "tables": versions,
                                    "resolved_tables": locations})
            for name, sql in queries.items()}


def read_source(spark: SparkSession, *, sql: str, tables: Mapping[str, str]):
    """Read one managed query through the shared pinned source boundary."""
    frame, provenance = read_sources(spark, queries={"primary": sql}, tables=tables)["primary"]
    return frame, {"sql": provenance["sql"], "tables": provenance["tables"]}


def _transport_type(dtype):
    if isinstance(dtype, T.StructType):
        if not dtype.fields:
            raise ValueError("empty structs are not supported in study storage")
        for field in dtype:
            _transport_type(field.dataType)
    elif isinstance(dtype, T.ArrayType):
        _transport_type(dtype.elementType)
    elif isinstance(dtype, T.MapType):
        if not isinstance(dtype.keyType, T.StringType):
            raise ValueError("study JSON storage requires string map keys")
        _transport_type(dtype.valueType)
    elif not isinstance(dtype,
        T.StringType | T.BooleanType | T.ByteType | T.ShortType | T.IntegerType | T.LongType
        | T.FloatType | T.DoubleType | T.DecimalType | T.DateType | T.TimestampType
        | T.TimestampNTZType | T.BinaryType,
    ):
        raise ValueError(f"unsupported study storage type: {dtype.simpleString()}")


def _canonical_value(column, dtype):
    if isinstance(dtype, T.MapType):
        entries = F.array_sort(F.map_entries(column), lambda a, b:
                               F.when(a["key"] < b["key"], -1)
                               .when(a["key"] > b["key"], 1).otherwise(0))
        return F.map_from_entries(F.transform(entries, lambda entry: F.struct(
            entry["key"].alias("key"),
            _canonical_value(entry["value"], dtype.valueType).alias("value"),
        )))
    if isinstance(dtype, T.ArrayType):
        return F.transform(column, lambda item: _canonical_value(item, dtype.elementType))
    if isinstance(dtype, T.StructType):
        return F.when(column.isNotNull(), F.struct(*[
            _canonical_value(column.getField(field.name), field.dataType).alias(field.name)
            for field in dtype
        ]))
    return column


def _same_rows(left: DataFrame, right: DataFrame) -> bool:
    def signature(frame):
        return [(f.name, f.dataType.json()) for f in frame.schema]
    return (signature(left) == signature(right)
            and not left.exceptAll(right).limit(1).count()
            and not right.exceptAll(left).limit(1).count())


class DeltaStudyStore:
    """Five shared tables, using native idempotent appends and exact scoped readback.

    Transaction IDs identify logical keys, not payload hashes. Existing matching
    rows are verified before retry, including after Delta transaction-ID expiry.
    Definitions/results are written last by the coordinator; multiple tables and
    files are not one transaction. Readers discover only those final rows.
    """

    def __init__(self, spark: SparkSession, *, namespace: str,
                 tables: Mapping[str, str] | None = None):
        quote_table(namespace)
        if tables is not None and set(tables) - set(SCHEMAS):
            raise ValueError("unknown study table role")
        self.spark = spark
        self.namespace = namespace
        self.tables = {kind: (tables or {}).get(kind, namespace + "." + kind) for kind in SCHEMAS}
        for name in self.tables.values():
            quote_table(name)
        if len(set(self.tables.values())) != len(self.tables):
            raise ValueError("study table roles must have distinct names")

    def prepare(self, *, lock_path: pathlib.Path) -> None:
        """Serialize initial DDL with the registry's shared POSIX file lock.

        Delta row appends are concurrent; first-time metastore/table creation is
        not. Every provisioner of these destinations must use the same lock file.
        The existing file-lock primitive never steals a live writer's lock, and
        the OS releases it on process exit. This is not an object-store lock.
        """
        import fcntl

        from autoforecast.campaign import _open_lock

        with _open_lock(lock_path) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {quote_table(self.namespace)}")
            for kind, schema in SCHEMAS.items():
                command = (
                    f"CREATE TABLE IF NOT EXISTS {quote_table(self.tables[kind])} "
                    f"({schema}) USING DELTA"
                )
                self.spark.sql(command)
            self.validate_destinations()

    def validate_destinations(self) -> None:
        """Read-only precheck; a model run never silently provisions tables."""
        for kind, name in self.tables.items():
            if not self.spark.catalog.tableExists(name):
                raise ValueError(f"configured study table does not exist: {kind}")
            actual = self.spark.table(name)
            expected = self.spark.createDataFrame([], SCHEMAS[kind])
            if [(f.name, f.dataType) for f in actual.schema] != [
                (f.name, f.dataType) for f in expected.schema
            ]:
                raise ValueError(f"configured study table has a different schema: {kind}")
            latest_delta_version(self.spark, name)

    def read(self, kind: str, scope: Mapping[str, str]) -> tuple[DataFrame, int]:
        name = self.tables[kind]
        version = latest_delta_version(self.spark, name)
        frame = self.spark.read.option("versionAsOf", version).table(name)
        for key, value in scope.items():
            frame = frame.filter(F.col(key) == F.lit(value))
        return frame, version

    def link_issues(self, check_id: str, study_id: str) -> None:
        """Attach one report to its study once, preserving failed-attempt reports."""
        if any(not isinstance(value, str) or not value for value in (check_id, study_id)):
            raise ValueError("issue linkage requires check and study IDs")
        before, _ = self.read("study_data_issues", {"check_id": check_id})
        if before.filter(F.col("study_id").isNotNull()
                         & (F.col("study_id") != study_id)).limit(1).count():
            raise ValueError("issue report already belongs to a different study")
        expected = before.withColumn("study_id", F.lit(study_id))
        table = quote_table(self.tables["study_data_issues"]).replace("{", "{{").replace("}", "}}")
        self.spark.sql(
            f"UPDATE {table} "
            "SET study_id = {study} WHERE check_id = {check} AND study_id IS NULL",
            study=study_id, check=check_id,
        )
        after, _ = self.read("study_data_issues", {"check_id": check_id})
        if not _same_rows(expected, after):
            raise ValueError("issue linkage differs from the checked report")

    def append_once(self, kind: str, frame: DataFrame, *, scope: Mapping[str, str]) -> None:
        """Reuse identical logical writes; reject conflicting or incomplete scopes."""
        if not scope or any(not isinstance(v, str) or not v for v in scope.values()):
            raise ValueError("immutable writes require a nonempty logical scope")
        for key, value in scope.items():
            if key not in frame.columns or frame.filter(
                F.col(key).isNull() | (F.col(key) != F.lit(value))
            ).limit(1).count():
                raise ValueError("submitted rows escape their immutable write scope")
        existing, _ = self.read(kind, scope)
        if existing.limit(1).count():
            if not _same_rows(frame, existing):
                raise ValueError(f"conflicting immutable {kind} records")
            return
        transaction = "autoforecast-" + digest({"table": self.tables[kind], "scope": dict(scope)})
        try:
            (frame.write.format("delta").mode("append").option("txnAppId", transaction)
             .option("txnVersion", 0).saveAsTable(self.tables[kind]))
        except Exception:
            # A simultaneous identical transaction may have won. Never swallow
            # a failed write unless its complete scoped content is already present.
            with suppress(Exception):
                checked, _ = self.read(kind, scope)
                if _same_rows(frame, checked):
                    return
            raise
        checked, _ = self.read(kind, scope)
        if not _same_rows(frame, checked):
            raise ValueError(f"Delta readback differs from immutable {kind} submission")

    def freeze_observations(self, path: pathlib.Path, *, keys: list[str],
                            scratch_root: pathlib.Path) -> dict:
        """Store a materialized source once; verify transport and typed content."""
        frame = self.spark.read.parquet(str(path))
        _transport_type(frame.schema)
        if self.spark.conf.get("spark.sql.session.timeZone") != "UTC":
            raise ValueError("study storage requires spark.sql.session.timeZone=UTC")
        record = {
            "algorithm": "typed-parquet-rows-v2", "schema": frame.schema.jsonValue(),
            "columns": frame.columns, "source_keys": list(keys),
            "logical_sha256": parquet_content_hash(path),
            "rows": frame.count(), "series": frame.select(*keys).distinct().count(),
            "codec": dict(JSON_OPTIONS),
        }
        record["snapshot_id"] = digest(record)
        encoded = frame.select(F.to_json(F.struct(*[
            _canonical_value(F.col("`" + f.name.replace("`", "``") + "`"), f.dataType)
            .alias(f.name) for f in frame.schema
        ]), JSON_OPTIONS).alias("row_json"))
        with tempfile.TemporaryDirectory(prefix="af-delta-encode-", dir=scratch_root) as temporary:
            root = pathlib.Path(temporary)
            encoded.write.parquet(str(root / "encoded"))
            encoded = self.spark.read.parquet(str(root / "encoded"))
            record["transport_sha256"] = parquet_content_hash(root / "encoded")
            self._decode_verified(encoded, record, root / "decoded")
            submitted = encoded.withColumn("snapshot_id", F.lit(record["snapshot_id"])).select(
                "snapshot_id", "row_json",
            )
            self.append_once("frozen_observations", submitted,
                             scope={"snapshot_id": record["snapshot_id"]})
            self.verify_observations(record, scratch_root=root)
        return record

    def _decode_verified(self, encoded: DataFrame, record: dict, destination: pathlib.Path):
        schema = T.StructType.fromJson(record["schema"])
        decoded = encoded.select(F.from_json("row_json", schema, record["codec"]).alias("row"))
        decoded.select("row.*").write.parquet(str(destination))
        checked = self.spark.read.parquet(str(destination))
        if (checked.columns != record["columns"]
                or checked.schema.jsonValue() != record["schema"]
                or checked.count() != record["rows"]
                or checked.select(*record["source_keys"]).distinct().count() != record["series"]
                or parquet_content_hash(destination) != record["logical_sha256"]):
            raise ValueError("stored study rows differ from their protected typed snapshot")

    def verify_observations(self, record: dict, *, scratch_root: pathlib.Path) -> int:
        """Verify a pinned stored snapshot; never replace expected hashes on load."""
        if record["algorithm"] != "typed-parquet-rows-v2" or record["codec"] != JSON_OPTIONS:
            raise ValueError("unsupported study snapshot format")
        frame, version = self.read("frozen_observations", {"snapshot_id": record["snapshot_id"]})
        with tempfile.TemporaryDirectory(prefix="af-delta-read-", dir=scratch_root) as temporary:
            root = pathlib.Path(temporary)
            frame.select("row_json").write.parquet(str(root / "encoded"))
            if parquet_content_hash(root / "encoded") != record["transport_sha256"]:
                raise ValueError("stored study transport differs from its protected snapshot")
            self._decode_verified(self.spark.read.parquet(str(root / "encoded")), record,
                                  root / "decoded")
        return version

    def publish_results(self, definition, study_sha, workflow_path, *, scratch_root,
                        mlflow_links=None):
        """Publish one sealed workflow, forecasts first and discovery rows last."""
        from autoforecast.run_mlflow import (
            RunMLflowLink,
            _completed_run_snapshot,
            _mlflow,
            _verify_workflow_export,
        )
        from autoforecast.study_workflow import load_forecast_workflow

        self.validate_destinations()
        durable = pathlib.Path(workflow_path).absolute()
        with _completed_run_snapshot(durable, scratch_root=scratch_root) as (root, completion):
            workflow = load_forecast_workflow(root)
            if workflow.testing is None or completion["task"] != "forecast-workflow":
                raise ValueError("publication requires both completed workflow stages")
            if (workflow.study_id != definition["study_id"]
                    or workflow.provenance["study"]["sha256"] != definition["library_sha256"]):
                raise ValueError("workflow belongs to a different registered study")
            binding = workflow.provenance.get("registration")
            if binding is not None:
                allowed = {(definition["study_id"], study_sha)}
                if parent := definition.get("parent"):
                    allowed.add((parent["study_id"], parent["study_sha256"]))
                if (binding["study_id"], binding["sha256"]) not in allowed:
                    raise ValueError("workflow registration binding differs from its baseline")
            manifest_sha = hashlib.sha256(
                (root / "artifact_manifest.json").read_bytes(),
            ).hexdigest()
            if mlflow_links is not None and set(mlflow_links) != {"validation", "test"}:
                raise ValueError("tracking links must cover both workflow stages")
            forecasts = None
            results = []
            stages = (("validation", workflow.validation), ("test", workflow.testing))
            for stage, evidence in stages:
                link = (mlflow_links or {}).get(stage)
                if link is not None:
                    if not isinstance(link, RunMLflowLink):
                        raise ValueError("tracking links must be verified MLflow run links")
                    client = _mlflow().tracking.MlflowClient(tracking_uri=link.tracking_uri)
                    remote = client.get_run(link.run_id)
                    if any(remote.data.tags.get(key) != value for key, value in {
                        "autoforecast.manifest_sha256": manifest_sha,
                        "autoforecast.workflow_id": durable.name,
                        "autoforecast.stage": stage,
                        "autoforecast.study_sha256": definition["library_sha256"],
                    }.items()):
                        raise ValueError("tracking link belongs to different workflow evidence")
                    _verify_workflow_export(client, link.run_id, evidence)
                forecast = evidence["forecasts"]
                frame = self.spark.read.parquet(forecast.path).select(
                    F.lit(definition["study_id"]).alias("study_id"),
                    F.lit(study_sha).alias("study_sha"), F.lit(durable.name).alias("run_id"),
                    F.lit(stage).alias("stage"), F.col("series_id").cast("string"),
                    F.col("date").cast("string"), F.col("yhat").cast("double"),
                )
                forecasts = frame if forecasts is None else forecasts.unionByName(frame)
                # Keep the exact retained score and its durable paths. The temporary
                # snapshot is only an authenticated transport, never a returned link.
                saved = json.loads((root / stage / "result.json").read_bytes())
                additional = {
                    "model_format": saved["model_format"],
                    "evaluation_contract": saved["evaluation_contract"],
                    "parameters": workflow.parameters, "provenance": workflow.provenance,
                    "tracking": link.as_record() if link is not None else None,
                }
                results.append((
                    definition["study_id"], study_sha, durable.name, stage, forecast.adapter,
                    str(durable), manifest_sha, json.dumps(saved["score"], sort_keys=True),
                    str(durable / stage / "model"), saved["model_sha256"],
                    f"runs:/{link.run_id}/models" if link is not None else None,
                    json.dumps(additional, sort_keys=True),
                ))
            scope = {"study_id": definition["study_id"], "study_sha": study_sha,
                     "run_id": durable.name}
            assert forecasts is not None  # Both validated stages were collected above.
            self.append_once("forecasts", forecasts, scope=scope)
            # Atomic two-stage discovery: a retry may reuse an orphan forecast append.
            self.append_once("study_results", self.spark.createDataFrame(
                results, SCHEMAS["study_results"],
            ), scope=scope)
        return {"status": "published", "study_id": definition["study_id"],
                "study_sha256": study_sha, "run_id": durable.name, "workflow_sha256": manifest_sha}

    def dashboard_sql(self) -> str:
        """Return a read-only join; view creation remains an explicit caller action."""
        return (
            f"SELECT r.*, s.definition_json FROM {quote_table(self.tables['study_results'])} r "
            f"JOIN {quote_table(self.tables['studies'])} s "
            "ON r.study_id = s.study_id AND r.study_sha = s.study_sha"
        )
