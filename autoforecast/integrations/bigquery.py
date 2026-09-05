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
