"""Concrete Delta persistence; source access and credentials belong to the caller.

Shared tables use fixed envelopes so independent studies may have different source
column names/types. Typed data is verified after decoding, using the same bounded
Parquet content hash as the existing freezer. Delta directories are never scanned
as raw Parquet. Files still require the library's supported filesystem semantics.
"""

from __future__ import annotations

import pathlib
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack, suppress

from mlops import _volume_files
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from sqlglot import exp, parse
from sqlglot.errors import ParseError
from sqlglot.expressions.dml import DML

from autoforecast._execution import note, reported
from autoforecast.benchmark import digest, parquet_content_hash
from autoforecast.study_results import PUBLICATION_SCHEMAS, _publication

SCHEMAS = {
    "studies": "study_id string, study_sha string, definition_json string",
    "frozen_observations": "snapshot_id string, row_json string",
    "study_data_issues": "check_id string, study_id string, issue_json string",
    **PUBLICATION_SCHEMAS,
}
JSON_OPTIONS = {
    "ignoreNullFields": "false", "timeZone": "UTC",
    "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
    "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    "dateFormat": "yyyy-MM-dd", "mode": "FAILFAST",
}


def quote_table(name: str) -> str:
    """Quote a multipart identifier; never accept an expression in its place."""
    if (not isinstance(name, str) or not name or "${" in name
            or any(not part for part in name.split("."))):
        raise ValueError("table names require nonempty parts without variable substitution")
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
    SQLGlot checks the Spark SQL syntax without accessing a JVM or Spark session.
    Returns query templates that bind only the declared, version-pinned frames.
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
    # Parse locally: Spark Connect does not expose the JVM SQL parser. Validate
    # every query before looking up source tables or writing study records.
    bindings = {alias.casefold(): f"af_source_{index}" for index, alias in enumerate(tables)}
    return {name: _bound_source_sql(sql, bindings) for name, sql in queries.items()}


def _bound_source_sql(sql: str, bindings: Mapping[str, str]) -> str:
    """Check a SELECT and replace only its parsed source names with DataFrame slots.

    Args:
        sql: Original Spark SQL. Its literals, comments and expressions stay intact.
        bindings: Case-folded source aliases mapped to Spark SQL keyword arguments.

    Returns:
        SQL for spark.sql(..., **frames). Rejects writes, undeclared relations and
        source-version overrides before Spark is called. Braces in caller SQL are
        escaped so they cannot become Python format fields.
    """
    # Spark expands these variables after parsing, even inside quoted text.
    # Reject them so the executed query cannot gain unverified table references.
    if "${" in sql:
        raise ValueError("source SQL does not support variable substitution")
    try:
        statements = [statement for statement in parse(sql, read="spark")
                      if statement is not None and not isinstance(statement, exp.Semicolon)]
    except ParseError as error:
        raise ValueError("source SQL is not a supported SELECT") from error
    queries = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)
    if len(statements) != 1 or not isinstance(statements[0], queries):
        raise ValueError("source SQL requires one read-only SELECT")
    tree = statements[0]
    # Keep the supported query surface explicit. In particular, SELECT INTO and
    # WITH commands must not reach Spark as apparent read-only queries.
    allowed = {
        exp.Select: {"expressions", "from_", "joins", "where", "group", "having",
                     "order", "limit", "offset", "distinct", "windows",
                     "sort", "distribute", "cluster"},
        exp.Union: {"this", "expression", "distinct", "by_name", "order", "limit", "offset"},
        exp.Intersect: {"this", "expression", "distinct", "order", "limit", "offset"},
        exp.Except: {"this", "expression", "distinct", "order", "limit", "offset"},
        exp.Subquery: {"this", "alias", "order", "limit", "offset"},
    }
    replacements = []
    for node in tree.walk():
        if isinstance(node, exp.DDL | DML | exp.Command | exp.Into | exp.With
                      | exp.Lateral | exp.QueryTransform):
            raise ValueError("source SQL contains an unsupported or writable operator")
        if isinstance(node, exp.Query):
            if type(node) not in allowed or any(
                value and key not in allowed[type(node)] | {"sort", "distribute", "cluster"}
                for key, value in node.args.items()
            ):
                raise ValueError("source SQL contains an unsupported or writable operator")
        if isinstance(node, exp.From | exp.Join) and not isinstance(
            node.this, exp.Table | exp.Subquery
        ):
            raise ValueError("source SQL may read only declared pinned aliases")
        # A table alias must be a name. Parameter aliases can hide @v0 time travel.
        if isinstance(node, exp.TableAlias) and not isinstance(node.this, exp.Identifier):
            raise ValueError("source SQL requires literal table aliases")
        if not isinstance(node, exp.Table):
            continue
        if (not isinstance(node.this, exp.Identifier) or node.name.casefold() not in bindings
                or any(value and key not in {"this", "alias", "joins"}
                       for key, value in node.args.items())):
            raise ValueError("source SQL may read only declared pinned aliases")
        # Token spans let us retain the original Spark expression syntax rather
        # than regenerate it in a second SQL dialect.
        start, end = node.this.meta.get("start"), node.this.meta.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(sql):
            raise ValueError("source SQL has an unsupported relation token")
        replacement = "{" + bindings[node.name.casefold()] + "}"
        if not node.alias:
            replacement += " AS " + quote_table(node.name)
        replacements.append((start, end + 1, replacement))
    parts = []
    cursor = 0
    for start, end, replacement in sorted(replacements):
        if start < cursor:
            raise ValueError("source SQL has overlapping relation tokens")
        parts.extend((sql[cursor:start].replace("{", "{{").replace("}", "}}"), replacement))
        cursor = end
    parts.append(sql[cursor:].replace("{", "{{").replace("}", "}}"))
    return "".join(parts)


@reported("Resolve source aliases and pinned Delta versions")
def read_sources(spark: SparkSession, *, queries: Mapping[str, str],
                 tables: Mapping[str, str]):
    """Execute declared queries against one version per physical Delta table."""
    bound_queries = validate_source_queries(spark, queries=queries, tables=tables)
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
    note("Resolved sources", aliases=[{"alias": alias, **version}
                                     for alias, version in versions.items()], queries=list(queries),
         version_note="Delta table versions; row-date conditions remain in the SQL evidence")
    # Pass pinned DataFrames through Spark's public SQL API. Spark Connect carries
    # their plans with the query; no caller-owned temporary view is overwritten.
    frames = {
        f"af_source_{index}": spark.read.option("versionAsOf", source["version"]).table(
            quote_table(locations[source["table"]]["read_target"]),
        ) for index, source in enumerate(versions.values())
    }
    return {name: (spark.sql(bound_queries[name], args=None, **frames),
                   {"sql": sql, "tables": versions, "resolved_tables": locations})
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

    @reported("Create or validate registry tables")
    def prepare(self, *, lock_path: pathlib.Path) -> None:
        """Serialize initial DDL with the registry's shared lock.

        Delta row appends are concurrent; first-time table creation is not.
        Every provisioner must use the same lock path. Local locks are released
        by the OS on exit. Volume locks use native create-if-absent writes and
        require explicit recovery if a writer stops before releasing its lock.
        """
        import fcntl

        from autoforecast.campaign import _open_lock

        with ExitStack() as stack:
            if _volume_files.is_volume(lock_path):
                stack.enter_context(_volume_files.exclusive_lock(lock_path))
            else:
                lock = stack.enter_context(_open_lock(lock_path))
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

    @reported("Write or verify existing Delta records")
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
            note("Delta rows reused and verified", table=self.tables[kind], scope=dict(scope))
            return
        transaction = "autoforecast-" + digest({"table": self.tables[kind], "scope": dict(scope)})
        try:
            (frame.write.format("delta").mode("append").option("txnAppId", transaction)
             .option("txnVersion", 0).saveAsTable(self.tables[kind]))
        except Exception:
            # A simultaneous identical transaction may have won. Never swallow
            # a failed write unless its complete scoped content is already present.
            reused = False
            with suppress(Exception):
                checked, _ = self.read(kind, scope)
                reused = _same_rows(frame, checked)
            if reused:
                note("Delta rows reused and verified", table=self.tables[kind], scope=dict(scope))
                return
            raise
        checked, _ = self.read(kind, scope)
        if not _same_rows(frame, checked):
            raise ValueError(f"Delta readback differs from immutable {kind} submission")
        note("Delta rows written and verified", table=self.tables[kind], scope=dict(scope))

    @reported("Freeze and verify Delta observations")
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

    @reported("Verify frozen Delta snapshot")
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

    @reported("Publish retained result tables")
    def publish_results(self, definition, study_sha, workflow_path, *, scratch_root,
                        mlflow_links=None):
        """Publish tuning windows and final scores under the same study and run IDs.

        Args:
            definition: Verified registered study definition.
            study_sha: Hash of that definition, supplied by the registry.
            workflow_path: Folder containing a completed model workflow.
            scratch_root: Folder for temporary, verified copies of run files.
            mlflow_links: Optional verified links for validation and testing.

        Returns:
            Publication status and the study, workflow and hash identifiers.

        Existing identical rows are reused on retry. Forecast rows are written
        first; result rows make the whole run visible to the dashboard afterward.
        No model is fitted and no score is recalculated here.
        """
        self.validate_destinations()
        with _publication(definition, study_sha, workflow_path, scratch_root=scratch_root,
                          mlflow_links=mlflow_links) as publication:
            forecasts = None
            for path, labels in publication["forecasts"]:
                frame = self.spark.read.parquet(path).select(*[
                    (F.col(name) if name in {"series_id", "date", "yhat"}
                     else F.lit(labels[name])).cast(dtype).alias(name)
                    for name, dtype in (part.strip().split()
                                        for part in SCHEMAS["forecasts"].split(","))
                ])
                forecasts = frame if forecasts is None else forecasts.unionByName(frame)
            assert forecasts is not None  # A full workflow always has both final stages.
            self.append_once("forecasts", forecasts, scope=publication["scope"])
            self.append_once("study_results", self.spark.createDataFrame(
                publication["study_results"], SCHEMAS["study_results"],
            ), scope=publication["scope"])
            return publication["status"]

    def dashboard_sql(self) -> str:
        """Return a read-only join; view creation remains an explicit caller action."""
        return (
            f"SELECT r.*, s.definition_json FROM {quote_table(self.tables['study_results'])} r "
            f"JOIN {quote_table(self.tables['studies'])} s "
            "ON r.study_id = s.study_id AND r.study_sha = s.study_sha"
        )
