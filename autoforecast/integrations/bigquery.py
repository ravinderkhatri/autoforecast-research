"""Publish retained results with atomic BigQuery jobs and verified retry identities."""

from __future__ import annotations

import datetime as dt
import pathlib
import re
import tempfile
import uuid

import pyarrow.parquet as pq

from autoforecast.benchmark import digest
from autoforecast.study_results import PUBLICATION_SCHEMAS, _publication, _publication_batches


class BigQueryResults:
    """Result destination using a caller-owned BigQuery client and existing dataset.

    Install ``autoforecast[bigquery]``. Supply ``project.dataset`` and its location.
    ``create_tables()`` creates missing tables. Forecasts publish before result
    rows, which mark a complete run. Retries reuse identical rows. Temporary
    staging tables are deleted after use and expire after one day if interrupted.
    """

    def __init__(
        self, client, *, dataset, location, results="study_results", forecasts="forecasts"
    ):
        if not re.fullmatch(r"[a-z][a-z0-9-]*\.[A-Za-z_][A-Za-z0-9_]*", dataset):
            raise ValueError("dataset must be project.dataset")
        if (
            any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in (results, forecasts))
            or results == forecasts
        ):
            raise ValueError("destination tables must have distinct simple names")
        if not isinstance(location, str) or not location.strip():
            raise ValueError("BigQuery dataset location is required")
        self.client, self.dataset, self.location = client, dataset, location
        self.job_project = dataset.split(".")[0]
        self.tables = {
            "study_results": f"{dataset}.{results}",
            "forecasts": f"{dataset}.{forecasts}",
        }

    @staticmethod
    def _schema(kind):
        from google.cloud import bigquery

        types = {"string": "STRING", "int": "INTEGER", "double": "FLOAT"}
        return [
            bigquery.SchemaField(name, types[dtype])
            for name, dtype in (
                part.strip().split() for part in PUBLICATION_SCHEMAS[kind].split(",")
            )
        ]

    def create_tables(self):
        """Create missing destination tables without replacing existing data or schemas."""
        from google.cloud import bigquery

        for kind, name in self.tables.items():
            self.client.create_table(
                bigquery.Table(name, schema=self._schema(kind)), exists_ok=True
            )

    def _query(self, query, parameters):
        from google.cloud import bigquery

        return self.client.query(
            query,
            location=self.location,
            job_config=bigquery.QueryJobConfig(query_parameters=parameters),
        )

    def _append(self, kind, publication, directory):
        from google.api_core.exceptions import Conflict, NotFound
        from google.cloud import bigquery

        table = self.client.get_table(self.tables[kind])
        if table.table_type != "TABLE" or table.schema != self._schema(kind):
            raise ValueError(f"destination schema differs for {kind}")
        scope = publication["scope"]
        parameters = [
            bigquery.ScalarQueryParameter(key, "STRING", value) for key, value in scope.items()
        ]
        where = " AND ".join(f"{key} = @{key}" for key in scope)
        staging = f"{self.tables[kind]}_af_{uuid.uuid4().hex}"
        stage = bigquery.Table(staging, schema=self._schema(kind))
        stage.expires = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
        stage = self.client.create_table(stage)
        try:
            file = pathlib.Path(directory) / f"{kind}.parquet"
            with pq.ParquetWriter(
                file, next(_publication_batches(publication, kind)).schema
            ) as writer:
                for batch in _publication_batches(publication, kind):
                    writer.write_batch(batch)
            with file.open("rb") as stream:
                self.client.load_table_from_file(
                    stream,
                    staging,
                    location=self.location,
                    job_config=bigquery.LoadJobConfig(
                        source_format="PARQUET", write_disposition="WRITE_EMPTY",
                        create_disposition="CREATE_NEVER",
                    ),
                ).result()
            existing = (f"SELECT *, COUNT(*) AS _n FROM `{self.tables[kind]}` "
                        f"WHERE {where} GROUP BY ALL")
            incoming = f"SELECT *, COUNT(*) AS _n FROM `{staging}` GROUP BY ALL"
            comparison = f"""SELECT
                EXISTS(SELECT 1 FROM `{self.tables[kind]}` WHERE {where}) AS present,
                NOT EXISTS((({existing}) EXCEPT DISTINCT ({incoming})) UNION ALL
                           (({incoming}) EXCEPT DISTINCT ({existing}))) AS identical"""
            before = next(iter(self._query(comparison, parameters).result()))
            if before.present:
                if not before.identical:
                    raise ValueError(f"publication conflicts with existing {kind}")
                return
            # All clients use the destination project for the same job namespace.
            # Scope, not payload, controls the identity: competing payloads cannot both append.
            prefix = "af_" + digest(
                {"table": self.tables[kind], "created": table.created.isoformat(), "scope": scope}
            )
            for attempt in range(100):
                job_id = f"{prefix}_{attempt}"
                try:
                    job = self.client.get_job(
                        job_id, project=self.job_project, location=self.location
                    )
                except NotFound:
                    try:
                        job = self.client.query(
                            f"INSERT INTO `{self.tables[kind]}` SELECT * FROM `{staging}`",
                            job_id=job_id,
                            project=self.job_project,
                            location=self.location,
                            job_retry=None,
                        )
                    except Conflict:
                        job = self.client.get_job(
                            job_id, project=self.job_project, location=self.location
                        )
                if job.done() and job.error_result is not None:
                    continue  # A failed atomic job wrote no rows. Reuse the next stable ID.
                job.result()
                break
            else:
                raise RuntimeError("publication exceeded 100 failed BigQuery job attempts")
            if not next(iter(self._query(comparison, parameters).result())).identical:
                raise ValueError(f"publication conflicts with existing {kind}")
        finally:
            self.client.delete_table(staging, not_found_ok=True)

    def publish_results(
        self, definition, study_sha, workflow_path, *, scratch_root, mlflow_links=None
    ):
        """Publish without fitting, rescoring, clipping or rounding model output."""
        with (
            _publication(
                definition,
                study_sha,
                workflow_path,
                scratch_root=scratch_root,
                mlflow_links=mlflow_links,
            ) as publication,
            tempfile.TemporaryDirectory(
                dir=scratch_root, prefix="bigquery-publication-"
            ) as directory,
        ):
            for kind in ("forecasts", "study_results"):
                self._append(kind, publication, directory)
            return publication["status"]


class BigQuerySnapshots:
    """Provider snapshot transport, pending protected study-input qualification.

    Read-only rows alone do not protect deletion, expiry or access policy. Public
    study registration and replay refuse this backend until the restricted-role
    service gate qualifies its complete lifecycle. These primitives remain usable
    by the service tests; they do not establish a table-mode study guarantee.

    Python uploads and downloads stream bounded batches through the driver. This
    transport does not install a Spark connector or claim distributed throughput.
    The caller owns authentication; no credentials enter a snapshot reference.
    """

    backend = "bigquery"

    def __init__(self, client, *, dataset, location):
        # Reuse the destination's existing identifier and location validation.
        settings = BigQueryResults(client, dataset=dataset, location=location)
        self.client, self.dataset, self.location = client, settings.dataset, settings.location

    @property
    def namespace(self) -> str:
        return self.dataset

    def preflight(self):
        dataset = self.client.get_dataset(self.dataset)
        if dataset.location.casefold() != self.location.casefold():
            raise ValueError("snapshot destination location differs from the configured location")

    @staticmethod
    def _schema(schema):
        import pyarrow as pa
        from google.cloud import bigquery

        result = []
        for field in schema:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field.name):
                raise ValueError("BigQuery input columns require simple names")
            dtype = field.type
            if pa.types.is_boolean(dtype):
                native = "BOOLEAN"
            elif pa.types.is_signed_integer(dtype):
                native = "INTEGER"
            elif pa.types.is_floating(dtype):
                native = "FLOAT"
            elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
                native = "STRING"
            elif pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
                native = "BYTES"
            elif pa.types.is_date32(dtype):
                native = "DATE"
            elif pa.types.is_timestamp(dtype) and dtype.unit in {"s", "ms", "us"}:
                native = "DATETIME" if dtype.tz is None else "TIMESTAMP"
            elif pa.types.is_decimal(dtype) and 0 <= dtype.scale <= 38:
                native = ("NUMERIC" if dtype.scale <= 9 and dtype.precision - dtype.scale <= 29
                          else "BIGNUMERIC")
            else:
                raise ValueError(f"BigQuery snapshot transport cannot preserve type {dtype}")
            result.append(bigquery.SchemaField(field.name, native,
                                               mode="NULLABLE" if field.nullable else "REQUIRED"))
        return result

    @staticmethod
    def _input(path):
        import math

        import pyarrow as pa

        from autoforecast.snapshots import ColumnLayout

        files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
        if not files:
            raise ValueError("snapshot capture requires typed Parquet input")
        schema, rows = None, 0
        for file in files:
            with pq.ParquetFile(file) as reader:
                current = reader.schema_arrow.remove_metadata()
                if schema is not None and not current.equals(schema, check_metadata=False):
                    raise ValueError("snapshot capture contains different file schemas")
                schema = current
                rows += reader.metadata.num_rows
                layout = ColumnLayout(current)
                floats = [index for index, field in enumerate(layout.schema)
                          if pa.types.is_floating(field.type)]
                if floats:
                    for batch in reader.iter_batches(batch_size=8192):
                        for row in batch.to_pylist():
                            values = layout.encode(row)
                            if any(values[index] is not None and values[index] == 0
                                   and math.copysign(1, values[index]) < 0 for index in floats):
                                raise ValueError("BigQuery cannot retain negative zero exactly")
        assert schema is not None
        return files, schema, rows

    def _identity(self, table, *, capture_id):
        definition = table.snapshot_definition
        if table.table_type != "SNAPSHOT" or definition is None:
            raise ValueError("native input is not a read-only snapshot")
        if table.expires is not None:
            raise ValueError("retained read-only snapshot still has an expiration")
        if table.labels.get("autoforecast_capture") != capture_id:
            raise ValueError("retained read-only snapshot capture label differs")
        return {"created": table.created.isoformat(),
                "owner": table.labels.get("autoforecast_owner"),
                "base": str(definition.base_table_reference),
                "snapshot_time": definition.snapshot_time.isoformat(),
                "schema": [field.to_api_repr() for field in table.schema]}

    def capture(self, path, *, capture_id, checkpoint=None):
        from google.cloud import bigquery

        from autoforecast.snapshots import ColumnLayout, arrow_schema

        if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
            raise ValueError("snapshot capture requires an owned random ID")
        self.preflight()
        files, schema, rows = self._input(path)
        layout = ColumnLayout(schema)
        fields = self._schema(layout.schema)
        target = f"{self.dataset}.af_input_{capture_id}"
        staging = target + "_staging"
        stage = bigquery.Table(staging, schema=fields)
        owner = uuid.uuid4().hex
        stage.labels = {"autoforecast_capture": capture_id, "autoforecast_owner": owner}
        stage.expires = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
        complete = False
        from contextlib import ExitStack

        from google.api_core.exceptions import NotFound

        from autoforecast.snapshots import _cleanup_on_exit

        def remove_owned(name):
            try:
                table = self.client.get_table(name)
            except NotFound:
                return
            if (table.labels.get("autoforecast_capture") == capture_id
                    and table.labels.get("autoforecast_owner") == owner):
                self.client.delete_table(name, not_found_ok=True)

        def cleanup():
            with ExitStack() as removals:
                for name in (staging,) if complete else (staging, target):
                    removals.callback(remove_owned, name)

        with _cleanup_on_exit(cleanup, label="BigQuery capture cleanup needs recovery"):
            self.client.create_table(stage)
            with layout.transport(files) as transferred:
                for index, file in enumerate(transferred):
                    with file.open("rb") as stream:
                        self.client.load_table_from_file(
                            stream, staging, location=self.location,
                            job_id=f"af_input_{capture_id}_load_{index}",
                            job_config=bigquery.LoadJobConfig(
                                source_format="PARQUET", schema=fields,
                                write_disposition="WRITE_EMPTY" if index == 0 else "WRITE_APPEND",
                                create_disposition="CREATE_NEVER",
                            ),
                        ).result()
            self.client.query(
                f"CREATE SNAPSHOT TABLE `{target}` CLONE `{staging}` "
                f"OPTIONS(labels=[('autoforecast_capture','{capture_id}'), "
                f"('autoforecast_owner','{owner}')])",
                location=self.location, job_id=f"af_input_{capture_id}_seal", job_retry=None,
            ).result()
            table = self.client.get_table(target)
            # A dataset default must not silently shorten the promised retention.
            if table.expires is not None:
                table.expires = None
                self.client.update_table(table, ["expires"])
                table = self.client.get_table(target)
            incarnation = self._identity(table, capture_id=capture_id)
            if table.num_rows != rows or table.schema != fields:
                raise ValueError("BigQuery snapshot transport changed rows or schema")
            record = {"object": target, "capture_id": capture_id, "incarnation": incarnation,
                      "arrow_schema": arrow_schema(schema), "rows": rows}
            if checkpoint is not None:
                checkpoint(record)
            complete = True
            return record

    def inspect(self, record, *, retained=False):
        target = record["object"]
        if target != f"{self.dataset}.af_input_{record['capture_id']}":
            raise ValueError("snapshot belongs to a different destination")
        table = self.client.get_table(target)
        if (self._identity(table, capture_id=record["capture_id"]) != record["incarnation"]
                or table.num_rows != record["rows"]):
            raise ValueError("retained snapshot incarnation or rows changed")

    def retain(self, record):
        """Native snapshots prohibit row writes from creation, without a sealing transition."""
        self.inspect(record)

    def drop_unretained(self, record):
        """Remove an attempt-owned snapshot after checking its exact native identity."""
        self.release(record)

    def read(self, record, destination):
        import pyarrow as pa

        from autoforecast.snapshots import ColumnLayout, restore_schema

        self.inspect(record)
        schema = restore_schema(record["arrow_schema"])
        layout = ColumnLayout(schema)
        destination.mkdir(parents=True)
        rows = self.client.list_rows(record["object"], page_size=8192)
        with pq.ParquetWriter(destination / "part.parquet", schema) as writer:
            for page in rows.pages:
                # The REST iterator bounds each page and decodes native scalar types.
                writer.write_table(pa.Table.from_pylist([
                    layout.decode([row[name] for name in layout.schema.names]) for row in page
                ], schema=schema))
        self.inspect(record)

    def release(self, record):
        from google.api_core.exceptions import NotFound

        try:
            self.inspect(record)
        except NotFound:
            return
        self.client.delete_table(record["object"])
        try:
            self.client.get_table(record["object"])
        except NotFound:
            return
        raise OSError("snapshot still exists after release")


def read_sources(client, *, queries, tables, location, output_root):
    """Capture declared BigQuery SELECTs against one explicit server timestamp.

    Only native base tables without active streaming buffers are supported. The
    retained files contain committed data visible at that table-version boundary.
    Each query result is consumed once with bounded REST pages.
    """
    import pyarrow as pa
    from google.cloud import bigquery

    from autoforecast.snapshots import bind_source_queries

    for name in tables.values():
        if not re.fullmatch(r"[a-z][a-z0-9-]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*",
                            name):
            raise ValueError("BigQuery source tables require project.dataset.table")
    if any("@af_capture_time" in query.casefold() for query in queries.values()):
        raise ValueError("af_capture_time is reserved for the source capture")
    relations = {alias: f"(SELECT * FROM `{name}` FOR SYSTEM_TIME AS OF @af_capture_time)"
                 for alias, name in tables.items()}
    bound = bind_source_queries(queries, tables, dialect="bigquery", sources=relations)
    before = {}
    for alias, name in tables.items():
        table = client.get_table(name)
        if (table.table_type != "TABLE" or table.external_data_configuration is not None
                or table.streaming_buffer is not None):
            raise ValueError("BigQuery capture requires base tables without streaming buffers")
        if table.location.casefold() != location.casefold():
            raise ValueError("BigQuery source location differs from the capture location")
        before[alias] = {"table": name, "created": table.created.isoformat()}
    timestamp = next(iter(client.query("SELECT CURRENT_TIMESTAMP() AS captured",
                                       location=location).result())).captured
    parameters = [bigquery.ScalarQueryParameter("af_capture_time", "TIMESTAMP", timestamp)]
    result = {}
    for index, (name, query) in enumerate(bound.items()):
        job = client.query(query, location=location,
                           job_config=bigquery.QueryJobConfig(query_parameters=parameters))
        rows = job.result(page_size=8192)
        # BigQuery's maintained schema conversion supplies empty-query types too.
        from google.cloud.bigquery._pandas_helpers import bq_to_arrow_schema

        schema = bq_to_arrow_schema(rows.schema)
        if schema is None:
            raise ValueError("BigQuery source contains unsupported Arrow types")
        path = pathlib.Path(output_root) / ("query-" + str(index))
        path.mkdir(parents=True)
        count = 0
        with pq.ParquetWriter(path / "part.parquet", schema) as writer:
            for page in rows.pages:
                batch = pa.Table.from_pylist([dict(row) for row in page], schema=schema)
                count += batch.num_rows
                writer.write_table(batch)
        result[name] = (path, {"kind": "bigquery", "tables": before, "sql": queries[name],
                              "capture_time": timestamp.isoformat(), "rows": count})
    for original in before.values():
        table = client.get_table(original["table"])
        if table.created.isoformat() != original["created"] or table.streaming_buffer is not None:
            raise ValueError("BigQuery source changed incarnation or gained a streaming buffer")
    return result
