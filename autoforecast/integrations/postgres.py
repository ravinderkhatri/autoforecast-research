"""Publish retained results to caller-owned PostgreSQL tables using one transaction."""

from __future__ import annotations

import uuid

from autoforecast.study_results import PUBLICATION_SCHEMAS, _publication, _publication_batches


class PostgresResults:
    """Result destination using an idle psycopg connection and explicit table names.

    Install ``autoforecast[postgres]``. The caller owns credentials and the
    connection. ``create_tables()`` creates missing tables in an existing schema.
    Publication locks both tables, checks existing rows, and commits both together.
    """

    def __init__(
        self, connection, *, schema="public", results="study_results", forecasts="forecasts"
    ):
        from psycopg import sql

        if any(not isinstance(name, str) or not name for name in (schema, results, forecasts)):
            raise ValueError("schema and table names must be nonempty strings")
        if results == forecasts:
            raise ValueError("result and forecast tables must differ")
        self.connection = connection
        self.tables = {
            "study_results": sql.Identifier(schema, results),
            "forecasts": sql.Identifier(schema, forecasts),
        }

    def _idle(self):
        from psycopg.pq import TransactionStatus

        if self.connection.info.transaction_status != TransactionStatus.IDLE:
            raise ValueError("publication requires an idle connection with no pending transaction")

    def create_tables(self):
        """Create missing destination tables; never alter or replace an existing table."""
        from psycopg import sql

        self._idle()
        types = {
            "string": sql.SQL("text"),
            "double": sql.SQL("double precision"),
            "int": sql.SQL("integer"),
        }
        with self.connection.transaction(), self.connection.cursor() as cursor:
            for kind, table in self.tables.items():
                fields = [part.strip().split() for part in PUBLICATION_SCHEMAS[kind].split(",")]
                columns = sql.SQL(", ").join(
                    sql.SQL("{} {}").format(sql.Identifier(name), types[dtype])
                    for name, dtype in fields
                )
                cursor.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(table, columns))

    def publish_results(
        self, definition, study_sha, workflow_path, *, scratch_root, mlflow_links=None
    ):
        """Copy verified rows atomically, accepting only identical repeats of a run."""
        from psycopg import sql

        self._idle()
        with _publication(
            definition,
            study_sha,
            workflow_path,
            scratch_root=scratch_root,
            mlflow_links=mlflow_links,
        ) as publication:
            with self.connection.transaction(), self.connection.cursor() as cursor:
                # The comparison and insert must share one lock and READ COMMITTED snapshot.
                cursor.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                cursor.execute(
                    sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(
                        sql.SQL(", ").join(sorted(self.tables.values(), key=str)),
                    )
                )
                scope = publication["scope"]
                where = sql.SQL(" AND ").join(
                    sql.SQL("{} = %s").format(sql.Identifier(key)) for key in scope
                )
                comparisons = []
                for kind in ("forecasts", "study_results"):
                    table = self.tables[kind]
                    fields = [part.strip().split() for part in PUBLICATION_SCHEMAS[kind].split(",")]
                    cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(table))
                    expected = [
                        (name, {"string": 25, "double": 701, "int": 23}[dtype])
                        for name, dtype in fields
                    ]
                    if [
                        (column.name, column.type_code) for column in cursor.description
                    ] != expected:
                        raise ValueError(f"destination schema differs for {kind}")
                    temporary = sql.Identifier("af_" + uuid.uuid4().hex)
                    cursor.execute(
                        sql.SQL(
                            "CREATE TEMP TABLE {} (LIKE {}) ON COMMIT DROP",
                        ).format(temporary, table)
                    )
                    with cursor.copy(sql.SQL("COPY {} FROM STDIN").format(temporary)) as copy:
                        for batch in _publication_batches(publication, kind):
                            for row in batch.to_pylist():
                                copy.write_row(tuple(row.values()))
                    cursor.execute(
                        sql.SQL("SELECT EXISTS(SELECT 1 FROM {} WHERE {})").format(
                            table,
                            where,
                        ),
                        tuple(scope.values()),
                    )
                    present = cursor.fetchone()[0]
                    existing = sql.SQL("SELECT * FROM {} WHERE {}").format(table, where)
                    incoming = sql.SQL("SELECT * FROM {}").format(temporary)
                    comparison = sql.SQL(
                        "SELECT EXISTS(({} EXCEPT ALL {}) UNION ALL ({} EXCEPT ALL {}))",
                    ).format(existing, incoming, incoming, existing)
                    comparisons.append((kind, comparison))
                    if not present:
                        cursor.execute(
                            sql.SQL("INSERT INTO {} SELECT * FROM {}").format(
                                table,
                                temporary,
                            )
                        )
                # Flush deferred triggers, then check both tables before committing.
                # A trigger on result rows may also have changed earlier forecast rows.
                cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
                for kind, comparison in comparisons:
                    cursor.execute(comparison, tuple(scope.values()) * 2)
                    if cursor.fetchone()[0]:
                        raise ValueError(f"publication conflicts with existing {kind}")
            return publication["status"]
