"""Bounded SQL Server source capture using one database snapshot transaction."""

from __future__ import annotations

import pathlib
import re

import pyarrow as pa
import pyarrow.parquet as pq


def connect(dsn):
    """Use a standard connection string supplied through the application's credentials."""
    from mssql_python import connect as open_connection

    return open_connection(dsn, autocommit=True)


def _require_arrow_streaming(connection):
    # The DB-API text fetch path strips trailing NUL from MAX columns. Native
    # Arrow batches use the returned lengths and leave the SELECT unchanged.
    with connection.cursor() as cursor:
        if not callable(getattr(cursor, "arrow_reader", None)):
            raise ValueError(
                "SQL Server value transfer requires mssql-python 1.17 Arrow streaming; "
                "install autoforecast[sqlserver] and use its SQL Server connection"
            )


def _source_schema(connection, query):
    """Read declared result types before fetching, even for empty or all-null results."""
    scalar = {
        "bigint": pa.int64(),
        "int": pa.int32(),
        "smallint": pa.int16(),
        "tinyint": pa.uint8(),
        "bit": pa.bool_(),
        "real": pa.float32(),
        "float": pa.float64(),
        "date": pa.date32(),
        "money": pa.decimal128(19, 4),
        "smallmoney": pa.decimal128(10, 4),
    }
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT name,system_type_name,is_nullable,precision,scale,error_number "
            "FROM sys.dm_exec_describe_first_result_set(?,NULL,0) "
            "ORDER BY column_ordinal",
            (query,),
        )
        fields = []
        for name, declared, nullable, precision, scale, error in cursor.fetchall():
            if error or not name or not declared:
                raise ValueError("SQL Server SELECT requires named, statically typed columns")
            base = declared.split("(", 1)[0].lower()
            if base in scalar:
                dtype = scalar[base]
            elif base in {"numeric", "decimal"} and 0 <= scale <= precision <= 38:
                dtype = pa.decimal128(precision, scale)
            elif base in {"varchar", "nvarchar", "char", "nchar"}:
                dtype = pa.string()
            elif base in {"binary", "varbinary"}:
                dtype = pa.binary()
            elif base == "datetime2" and scale <= 6:
                dtype = pa.timestamp("us")
            else:
                raise ValueError(
                    f"SQL Server source cannot preserve {declared}; cast explicitly in the SELECT"
                )
            fields.append(pa.field(name, dtype, nullable=bool(nullable)))
    if not fields or len({field.name for field in fields}) != len(fields):
        raise ValueError("SQL Server SELECT must give every output column a distinct name")
    return pa.schema(fields)


def read_sources(connection, *, queries, tables, output_root):
    """Capture related SELECTs in one SNAPSHOT transaction without changing database settings.

    Table names use schema.table in the connection's current database. A saved
    observation time describes this read; it is not a promise of arbitrary future
    historical reads. Captured files retain the rows after the transaction ends.
    """
    from autoforecast.snapshots import bind_source_queries

    names = {}
    for alias, name in tables.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*",
            name,
        ):
            raise ValueError("SQL Server source tables require schema.table in one database")
        names[alias] = ".".join("[" + part + "]" for part in name.split("."))
    bound = bind_source_queries(queries, tables, dialect="tsql", sources=names)
    _require_arrow_streaming(connection)
    if not connection.autocommit:
        raise ValueError("source capture needs an idle autocommit connection")
    with connection.cursor() as cursor:
        cursor.execute("SELECT @@TRANCOUNT")
        if cursor.fetchone()[0]:
            raise ValueError("source capture needs an idle autocommit connection")
        cursor.execute(
            "SELECT snapshot_isolation_state FROM sys.databases " "WHERE database_id=DB_ID()"
        )
        if cursor.fetchone()[0] != 1:
            raise ValueError(
                "SQL Server source capture requires ALLOW_SNAPSHOT_ISOLATION ON; "
                "ask the database administrator to configure it"
            )
        cursor.execute(
            "SELECT transaction_isolation_level FROM sys.dm_exec_sessions "
            "WHERE session_id=@@SPID"
        )
        original_isolation = cursor.fetchone()[0]
    isolation_names = {
        1: "READ UNCOMMITTED",
        2: "READ COMMITTED",
        3: "REPEATABLE READ",
        4: "SERIALIZABLE",
        5: "SNAPSHOT",
    }
    if original_isolation not in isolation_names:
        raise ValueError("SQL Server session has an unsupported isolation setting")
    result = {}
    primary_error = None
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL SNAPSHOT")
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("SELECT DB_NAME(),DB_ID(),SYSUTCDATETIME()")
            database, database_id, observed_at = cursor.fetchone()
            provenance = {}
            for alias, name in tables.items():
                cursor.execute(
                    "SELECT t.object_id,t.create_date,t.modify_date,t.is_memory_optimized,"
                    "t.is_filetable,t.is_external FROM sys.tables t "
                    "WHERE t.object_id=OBJECT_ID(?,N'U')",
                    (name,),
                )
                table = cursor.fetchone()
                if table is None or any(table[3:]):
                    raise ValueError("SQL Server capture requires ordinary disk-based base tables")
                # Start the source read view before evaluating related queries. Concurrent
                # DDL cannot be substituted by a retry against a newer live source.
                cursor.execute("SELECT TOP (1) 1 FROM " + names[alias])
                cursor.fetchall()
                provenance[alias] = {
                    "table": name,
                    "object_id": table[0],
                    "created_at": table[1].isoformat(),
                    "modified_at": table[2].isoformat(),
                }
        for index, (name, query) in enumerate(bound.items()):
            schema = _source_schema(connection, query)
            destination = pathlib.Path(output_root) / ("query-" + str(index))
            destination.mkdir(parents=True)
            count = 0
            with (
                connection.cursor() as cursor,
                pq.ParquetWriter(
                    destination / "part.parquet",
                    schema,
                ) as writer,
            ):
                cursor.execute(query)
                with cursor.arrow_reader(batch_size=8192) as batches:
                    for batch in batches:
                        writer.write_table(pa.Table.from_arrays(batch.columns, schema=schema))
                        count += batch.num_rows
            result[name] = (
                destination,
                {
                    "kind": "sqlserver",
                    "tables": provenance,
                    "sql": queries[name],
                    "rows": count,
                    "capture_boundary": {
                        "isolation": "SNAPSHOT",
                        "database": database,
                        "database_id": database_id,
                        "observed_at": observed_at.isoformat(),
                    },
                },
            )
        with connection.cursor() as cursor:
            for saved in provenance.values():
                cursor.execute(
                    "SELECT object_id,create_date,modify_date FROM sys.tables "
                    "WHERE object_id=OBJECT_ID(?,N'U')",
                    (saved["table"],),
                )
                current = cursor.fetchone()
                if (
                    current is None
                    or current[0] != saved["object_id"]
                    or current[1].isoformat() != saved["created_at"]
                    or current[2].isoformat() != saved["modified_at"]
                ):
                    raise ValueError("SQL Server source table changed during capture")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # Explicit SQL also rolls back when the driver itself remains in autocommit mode.
        try:
            with connection.cursor() as cursor:
                cursor.execute("IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION")
                cursor.execute(
                    "SET TRANSACTION ISOLATION LEVEL " + isolation_names[original_isolation]
                )
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "SQL Server session cleanup also failed: " + type(cleanup_error).__name__
            )
    return result


def _archive_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value):
        raise ValueError("archive names require simple SQL Server identifiers")
    return value


def _execute(connection, statement, values=()):
    with connection.cursor() as cursor:
        cursor.execute(statement, values)
        return [tuple(row) for row in cursor.fetchall()] if cursor.description else []


def _snapshot_procedure(schema, registration, release):
    for name in (schema, registration, release):
        _archive_name(name)
    return f"""CREATE PROCEDURE [{schema}].af_snapshot
    @operation VARCHAR(32), @capture_id VARCHAR(32), @details NVARCHAR(MAX)
    WITH EXECUTE AS OWNER AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    IF @@TRANCOUNT<>1 OR XACT_STATE()<>1
        THROW 51000,'snapshot writes require one explicit transaction',1;
    IF @capture_id IS NULL OR LEN(@capture_id)<>32 OR
       @capture_id COLLATE Latin1_General_100_BIN2 LIKE '%[^0-9a-f]%' OR ISJSON(@details)<>1
        THROW 51000,'invalid capture identity',1;
    DECLARE @actor SYSNAME=ORIGINAL_LOGIN();
    DECLARE @name SYSNAME='af_input_'+@capture_id;
    DECLARE @qualified NVARCHAR(300)=QUOTENAME('{schema}')+'.'+QUOTENAME(@name);
    DECLARE @sql NVARCHAR(MAX), @seal VARCHAR(32), @oid INT, @saved_oid INT;
    DECLARE @state VARCHAR(16), @transaction BIGINT;
    IF @operation='begin' AND @actor='{registration}'
    BEGIN
        IF EXISTS(SELECT 1 FROM [{schema}].af_captures WHERE capture_id=@capture_id)
            THROW 51000,'capture identity already exists',1;
        DECLARE @sources NVARCHAR(MAX)=JSON_QUERY(@details,'$.sources');
        IF @sources IS NULL OR ISJSON(@sources,ARRAY)<>1 OR DATALENGTH(@sources)>7500
            THROW 51000,'invalid source identity binding',1;
        DECLARE @source_binding NVARCHAR(3750)=@sources;
        DECLARE @columns TABLE(position INT,name SYSNAME,kind VARCHAR(32),nullable VARCHAR(5),
                               precision_value INT,scale_value INT);
        INSERT INTO @columns
            SELECT CONVERT(INT,j.[key]),c.name,c.kind,c.nullable,c.precision_value,c.scale_value
            FROM OPENJSON(@details,'$.columns') j CROSS APPLY OPENJSON(j.value)
            WITH (name NVARCHAR(128) '$.name',kind VARCHAR(32) '$.kind',
                  nullable VARCHAR(5) '$.nullable',
                  precision_value INT '$.precision',scale_value INT '$.scale') c;
        IF NOT EXISTS(SELECT 1 FROM @columns) OR EXISTS(SELECT 1 FROM @columns WHERE
            name IS NULL OR name='' OR LEN(name)>64 OR
            name COLLATE Latin1_General_100_BIN2 LIKE '%[^A-Za-z0-9_]%' OR
            LEFT(name,1) COLLATE Latin1_General_100_BIN2 LIKE '[0-9]' OR
            kind IS NULL OR kind NOT IN ('bigint','bit','real','float','text','binary',
                                          'date','timestamp','decimal') OR
            nullable IS NULL OR nullable NOT IN ('true','false') OR
            (kind='decimal' AND (precision_value IS NULL OR scale_value IS NULL OR
                precision_value NOT BETWEEN 1 AND 38 OR
                scale_value NOT BETWEEN 0 AND precision_value)))
            THROW 51000,'unsupported capture columns',1;
        SELECT @sql=STRING_AGG(CONVERT(NVARCHAR(MAX),QUOTENAME(name)+' '+
            CASE kind WHEN 'bigint' THEN 'BIGINT' WHEN 'bit' THEN 'BIT'
                WHEN 'real' THEN 'REAL' WHEN 'float' THEN 'FLOAT(53)'
                WHEN 'text' THEN 'NVARCHAR(MAX)' WHEN 'binary' THEN 'VARBINARY(MAX)'
                WHEN 'date' THEN 'DATE' WHEN 'timestamp' THEN 'DATETIME2(6)'
                WHEN 'decimal' THEN 'DECIMAL('+CONVERT(VARCHAR(2),precision_value)+','+
                    CONVERT(VARCHAR(2),scale_value)+')' END+
            CASE nullable WHEN 'true' THEN ' NULL' ELSE ' NOT NULL' END),',')
            WITHIN GROUP (ORDER BY position) FROM @columns;
        SET @sql='CREATE TABLE '+@qualified+' ('+@sql+')';
        EXEC sys.sp_executesql @sql;
        SET @oid=OBJECT_ID(@qualified,'U');
        SET @seal=REPLACE(CONVERT(VARCHAR(36),NEWID()),'-','');
        EXEC sys.sp_addextendedproperty @name=N'autoforecast_seal',@value=@seal,
            @level0type=N'SCHEMA',@level0name=N'{schema}',@level1type=N'TABLE',@level1name=@name;
        EXEC sys.sp_addextendedproperty @name=N'autoforecast_sources',@value=@source_binding,
            @level0type=N'SCHEMA',@level0name=N'{schema}',@level1type=N'TABLE',@level1name=@name;
        INSERT INTO [{schema}].af_captures VALUES
            (@capture_id,@oid,@seal,CURRENT_TRANSACTION_ID(),'building');
        SET @sql='GRANT INSERT ON OBJECT::'+@qualified+' TO [{registration}]';
        EXEC sys.sp_executesql @sql;
        RETURN;
    END;
    SELECT @saved_oid=object_id,@seal=seal,@transaction=transaction_id,@state=state
        FROM [{schema}].af_captures WITH (UPDLOCK,HOLDLOCK) WHERE capture_id=@capture_id;
    SET @oid=OBJECT_ID(@qualified);
    IF @saved_oid IS NULL AND @oid IS NULL AND
       ((@operation='release' AND @actor='{release}') OR
        (@operation='drop_unretained' AND @actor='{registration}')) RETURN;
    IF @saved_oid IS NULL THROW 51000,'capture identity is missing',1;
    IF @operation='seal' AND @actor='{registration}'
    BEGIN
        IF @state<>'building' OR @transaction<>CURRENT_TRANSACTION_ID() OR @oid<>@saved_oid
            THROW 51000,'capture must seal in its creating transaction',1;
        SET @sql='REVOKE INSERT ON OBJECT::'+@qualified+' FROM [{registration}]';
        EXEC sys.sp_executesql @sql;
        UPDATE [{schema}].af_captures SET state='sealed' WHERE capture_id=@capture_id;
        RETURN;
    END;
    IF JSON_VALUE(@details,'$.seal') IS NULL OR JSON_VALUE(@details,'$.seal')<>@seal OR
       TRY_CONVERT(INT,JSON_VALUE(@details,'$.oid')) IS NULL OR
       TRY_CONVERT(INT,JSON_VALUE(@details,'$.oid'))<>@saved_oid
        THROW 51000,'original capture identity differs',1;
    IF @state='released' AND @oid IS NOT NULL
        THROW 51000,'released capture name was reused',1;
    IF @oid IS NOT NULL AND (@oid<>@saved_oid OR NOT EXISTS(
        SELECT 1 FROM sys.extended_properties WHERE class=1 AND major_id=@oid AND minor_id=0
        AND name='autoforecast_seal' AND CONVERT(VARCHAR(32),value)=@seal))
        THROW 51000,'capture object identity changed',1;
    IF @operation='retain' AND @actor='{registration}' AND @oid IS NOT NULL
        AND @state IN ('sealed','retained')
        UPDATE [{schema}].af_captures SET state='retained' WHERE capture_id=@capture_id;
    ELSE IF (@operation='release' AND @actor='{release}') OR
        (@operation='drop_unretained' AND @actor='{registration}'
         AND @state IN ('sealed','released'))
    BEGIN
        IF @oid IS NOT NULL
        BEGIN
            SET @sql='DROP TABLE '+@qualified;
            EXEC sys.sp_executesql @sql;
        END;
        UPDATE [{schema}].af_captures SET state='released' WHERE capture_id=@capture_id;
    END
    ELSE THROW 51000,'capture operation is not permitted',1;
END"""


def provision_snapshot_schema(
    connection, *, schema, owner, registration_role, reader_role, release_role
):
    """Provision a new SQL Server archive using a separate administrator session.

    Supply a user WITHOUT LOGIN and three ordinary SQL login/user pairs with the
    same respective names. Read-only catalog visibility lets replay verify grants.
    Neither TRUSTWORTHY nor database-wide isolation settings are changed.
    """
    names = (schema, owner, registration_role, reader_role, release_role)
    for name in names:
        _archive_name(name)
    if len(set(names[1:])) != 4:
        raise ValueError("archive owner and application accounts must differ")
    if not connection.autocommit or _execute(connection, "SELECT @@TRANCOUNT")[0][0]:
        raise ValueError("archive provisioning needs an idle autocommit connection")
    database = _execute(connection, "SELECT DB_NAME()")[0][0]
    if database in {"master", "model", "msdb", "tempdb"}:
        raise ValueError("provision an archive in a user database")
    roles = _execute(
        connection,
        """SELECT name,type_desc,authentication_type_desc,SUSER_SNAME(sid)
        FROM sys.database_principals WHERE name IN (?,?,?,?)""",
        names[1:],
    )
    expected = {
        (owner, "SQL_USER", "NONE", None),
        *((role, "SQL_USER", "INSTANCE", role) for role in names[2:]),
    }
    if set(roles) != expected:
        raise ValueError("archive needs a user WITHOUT LOGIN and separate ordinary SQL logins")
    from contextlib import ExitStack

    from autoforecast.snapshots import _cleanup_on_exit

    committed = False
    cleanup = ExitStack()
    cleanup.callback(_execute, connection, "USE [" + database.replace("]", "]]") + "]")
    cleanup.callback(lambda: None if committed else _execute(
        connection, "IF @@TRANCOUNT>0 ROLLBACK",
    ))
    with _cleanup_on_exit(cleanup.close, label="SQL Server provision cleanup failed"):
        _execute(connection, "BEGIN TRANSACTION")
        _execute(connection, f"CREATE SCHEMA [{schema}] AUTHORIZATION [{owner}]")
        _execute(connection, f"GRANT CREATE TABLE TO [{owner}]")
        _execute(
            connection,
            f"""CREATE TABLE [{schema}].af_policy (
            singleton BIT PRIMARY KEY CHECK(singleton=1), owner SYSNAME NOT NULL,
            registration SYSNAME NOT NULL, reader SYSNAME NOT NULL, releaser SYSNAME NOT NULL
        )""",
        )
        _execute(connection, f"INSERT INTO [{schema}].af_policy VALUES(1,?,?,?,?)", names[1:])
        _execute(
            connection,
            f"""CREATE TABLE [{schema}].af_captures (
            capture_id VARCHAR(32) PRIMARY KEY, object_id INT NOT NULL,
            seal VARCHAR(32) NOT NULL, transaction_id BIGINT NOT NULL,
            state VARCHAR(16) NOT NULL CHECK(state IN ('building','sealed','retained','released'))
        )""",
        )
        _execute(connection, _snapshot_procedure(schema, registration_role, release_role))
        for role in names[2:]:
            _execute(connection, f"GRANT SELECT ON SCHEMA::[{schema}] TO [{role}]")
        for role in (registration_role, release_role):
            _execute(connection, f"GRANT EXECUTE ON OBJECT::[{schema}].af_snapshot TO [{role}]")
        _execute(connection, f"GRANT VIEW SECURITY DEFINITION TO [{reader_role}]")
        _execute(connection, f"GRANT VIEW DEFINITION ON SCHEMA::[{schema}] TO [{reader_role}]")
        _execute(connection, "USE master")
        _execute(connection, f"GRANT VIEW ANY SECURITY DEFINITION TO [{reader_role}]")
        _execute(connection, "USE [" + database.replace("]", "]]") + "]")
        _execute(connection, "COMMIT")
        committed = True


class SQLServerSnapshots:
    """Typed retained inputs under a protected SQL Server schema owner."""

    backend = "sqlserver"

    def __init__(self, writer, reader, *, schema, releaser=None):
        self.schema = _archive_name(schema)
        self.writer, self.reader, self.releaser = writer, reader, releaser
        self._source_users: tuple[tuple[str, str, str], ...] = ()

    @property
    def namespace(self):
        return self.schema

    @staticmethod
    def _idle(connection):
        if (
            connection is None
            or not connection.autocommit
            or _execute(
                connection,
                "SELECT @@TRANCOUNT",
            )[0][0]
        ):
            raise ValueError("snapshot connections require idle autocommit sessions")

    @staticmethod
    def _database(connection):
        name, guid = _execute(
            connection,
            "SELECT name,CONVERT(VARCHAR(36),service_broker_guid) "
            "FROM sys.databases WHERE database_id=DB_ID()",
        )[0]
        return {"name": name, "guid": guid}

    def _user_mappings(self):
        return _execute(
            self.reader,
            """SELECT p.name,CONVERT(VARCHAR(170),p.sid,2),s.name
            FROM sys.database_principals p JOIN sys.server_principals s ON s.sid=p.sid
            WHERE p.principal_id<>1 AND p.type='S' AND p.authentication_type_desc='INSTANCE'
            AND s.type='S'""",
        )

    def _check_sources(self, sources):
        if type(sources) is not list:
            raise ValueError("SQL Server source identity binding is missing")
        mappings = self._user_mappings() if sources else ()
        if any(type(source) is not list or tuple(source) not in mappings for source in sources):
            raise ValueError("SQL Server source archive user mapping changed")

    def _policy(self):
        reader = self.reader
        visibility = _execute(
            reader,
            """SELECT
            HAS_PERMS_BY_NAME(NULL,NULL,'VIEW ANY SECURITY DEFINITION'),
            HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','VIEW SECURITY DEFINITION'),
            HAS_PERMS_BY_NAME(?,'SCHEMA','VIEW DEFINITION')""",
            (self.schema,),
        )[0]
        if visibility != (1, 1, 1):
            raise ValueError(
                "archive reader needs complete security and routine metadata visibility"
            )
        rows = _execute(
            reader, f"SELECT owner,registration,reader,releaser FROM [{self.schema}].af_policy"
        )
        if len(rows) != 1 or len(set(rows[0])) != 4:
            raise ValueError("archive role policy changed")
        owner, registration, replay, release = rows[0]
        for name in rows[0]:
            _archive_name(name)
        if _execute(reader, "SELECT ORIGINAL_LOGIN(),USER_NAME()")[0] != (replay, replay):
            raise ValueError("archive requires its configured restricted reader")
        principals = _execute(
            reader,
            """SELECT p.name,p.principal_id,p.authentication_type_desc,
            s.name,s.principal_id FROM sys.database_principals p
            LEFT JOIN sys.server_principals s ON s.sid=p.sid
            WHERE p.name IN (?,?,?,?)""",
            rows[0],
        )
        mapped = {
            name: (identifier, authentication, login, login_id)
            for name, identifier, authentication, login, login_id in principals
        }
        if (
            set(mapped) != set(rows[0])
            or mapped[owner][1:] != ("NONE", None, None)
            or any(mapped[name][1:3] != ("INSTANCE", name) for name in rows[0][1:])
        ):
            raise ValueError("archive needs a protected user WITHOUT LOGIN and distinct SQL logins")
        owner_id = mapped[owner][0]
        if _execute(
            reader, "SELECT principal_id FROM sys.schemas WHERE name=?", (self.schema,)
        ) != [
            (owner_id,),
        ]:
            raise ValueError("archive schema ownership changed")
        if _execute(
            reader,
            """SELECT 1 FROM sys.databases WHERE database_id=DB_ID()
            AND (is_trustworthy_on=1 OR is_db_chaining_on=1)""",
        ):
            raise ValueError(
                "archive requires TRUSTWORTHY and cross-database chaining to remain off"
            )
        for name in rows[0]:
            identifier = mapped[name][0]
            if _execute(
                reader,
                """SELECT 1 FROM sys.database_role_members
                WHERE member_principal_id=? OR role_principal_id=?""",
                (identifier, owner_id),
            ):
                raise ValueError("archive application users must not acquire database roles")
            if name == owner:
                continue
            if _execute(
                reader,
                """SELECT 1 FROM sys.databases WHERE database_id=DB_ID()
                AND owner_sid=(SELECT sid FROM sys.database_principals WHERE principal_id=?)""",
                (identifier,),
            ):
                raise ValueError("archive application users must not own the database")
        # Apply the same restricted-account rule to source users and PUBLIC on replay.
        if _execute(
            reader,
            """WITH application_logins AS (
                SELECT p.principal_id AS user_id,s.principal_id FROM sys.database_principals p
                JOIN sys.server_principals s ON s.sid=p.sid
                WHERE p.principal_id<>1 AND p.type IN ('S','U','G','E','X'))
            SELECT 1 FROM sys.database_role_members
            WHERE member_principal_id IN (SELECT user_id FROM application_logins)
            UNION ALL
            SELECT 1 FROM sys.server_role_members
            WHERE member_principal_id IN (SELECT principal_id FROM application_logins)
            UNION ALL
            SELECT 1 FROM sys.server_permissions
            WHERE (grantee_principal_id=2 OR grantee_principal_id IN
                (SELECT principal_id FROM application_logins))
            AND state IN ('G','W') AND (state='W' OR NOT (
                (class=100 AND permission_name IN
                    ('CONNECT SQL','VIEW ANY DATABASE','VIEW ANY SECURITY DEFINITION'))
                OR (class=105 AND permission_name='CONNECT')))""",
        ):
            raise ValueError(
                "archive application accounts must not have database roles or "
                "server administration rights"
            )
        if _execute(
            reader,
            """SELECT 1 FROM sys.database_role_members m
            JOIN sys.database_principals r ON r.principal_id=m.role_principal_id
            WHERE r.name IN ('db_datawriter','db_ddladmin','db_securityadmin')""",
        ):
            raise ValueError("archive cannot admit ordinary database-wide mutation roles")
        # Reject principal authority for every recipient, including intermediary users,
        # roles and PUBLIC. The controlled procedure needs no impersonation grants.
        if _execute(
            reader,
            """SELECT 1 FROM sys.database_permissions p
            WHERE p.state IN ('G','W') AND p.grantee_principal_id NOT IN (1,?) AND
            ((p.class=0 AND (p.state='W' OR (p.permission_name NOT IN ('CONNECT','SELECT')
                                           AND p.permission_name NOT LIKE 'VIEW %')))
             OR (p.class=4 AND (p.state='W' OR p.permission_name<>'VIEW DEFINITION'))
             OR ((p.class=3 AND p.major_id=SCHEMA_ID(?)) OR
                 (p.class=1 AND p.major_id IN (SELECT object_id FROM sys.objects
                                             WHERE schema_id=SCHEMA_ID(?)))) AND
                (p.state='W' OR p.permission_name NOT IN ('SELECT','VIEW DEFINITION','EXECUTE') OR
                 (p.permission_name='EXECUTE' AND
                    (p.class<>1 OR p.major_id<>OBJECT_ID(?) OR
                     p.grantee_principal_id NOT IN (?,?)))))""",
            (
                owner_id,
                self.schema,
                self.schema,
                self.schema + ".af_snapshot",
                mapped[registration][0],
                mapped[release][0],
            ),
        ):
            raise ValueError("archive mutation or impersonation grants changed")
        # CREATE TABLE is the owner's only permitted database-wide mutation permission.
        if _execute(
            reader,
            """SELECT 1 FROM sys.database_permissions WHERE
            grantee_principal_id=? AND class=0 AND state IN ('G','W')
            AND (state='W' OR (permission_name NOT IN ('CREATE TABLE','CONNECT','SELECT')
                              AND permission_name NOT LIKE 'VIEW %'))""",
            (owner_id,),
        ):
            raise ValueError("archive owner has unexpected database privileges")
        for table in ("af_policy", "af_captures"):
            self._protected_table(reader, self.schema + "." + table, owner_id)
        procedure = _execute(
            reader,
            """SELECT m.definition,m.execute_as_principal_id,
            coalesce(o.principal_id,s.principal_id) FROM sys.sql_modules m
            JOIN sys.objects o ON o.object_id=m.object_id
            JOIN sys.schemas s ON s.schema_id=o.schema_id WHERE m.object_id=OBJECT_ID(?)""",
            (self.schema + ".af_snapshot",),
        )
        if procedure != [(_snapshot_procedure(self.schema, registration, release), -2, owner_id)]:
            raise ValueError("archive sealing operation changed")
        return owner_id, registration, replay, release

    @staticmethod
    def _protected_table(connection, name, owner):
        rows = _execute(
            connection,
            """SELECT coalesce(t.principal_id,s.principal_id),
            t.is_memory_optimized,t.temporal_type,t.is_filetable,
            CASE WHEN EXISTS(SELECT 1 FROM sys.triggers WHERE parent_id=t.object_id)
                 THEN 1 ELSE 0 END,
            CASE WHEN EXISTS(SELECT 1 FROM sys.security_predicates
                             WHERE target_object_id=t.object_id) THEN 1 ELSE 0 END
            FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
            WHERE t.object_id=OBJECT_ID(?)""",
            (name,),
        )
        if not rows:
            raise FileNotFoundError("retained input or protected archive table is missing")
        if rows != [(owner, False, 0, False, 0, 0)]:
            raise ValueError("archive table ownership, storage or row policies changed")

    def preflight(self):
        self._idle(self.writer)
        self._idle(self.reader)
        _, registration, _, _ = self._policy()
        if _execute(self.writer, "SELECT ORIGINAL_LOGIN(),USER_NAME()")[0] != (
            registration,
            registration,
        ):
            raise ValueError("archive requires its configured restricted registration account")
        if self._database(self.writer) != self._database(self.reader):
            raise ValueError("archive connections reach different databases")
        self._check_sources([list(user) for user in self._source_users])

    def check_source(self, connection):
        import uuid

        from autoforecast.snapshots import _cleanup_on_exit

        self._idle(connection)
        self._idle(self.reader)
        _, _, _, release = self._policy()
        *values, login, sid = _execute(
            connection,
            """SELECT IS_SRVROLEMEMBER('sysadmin'),
            HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','CONTROL'),
            HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','ALTER ANY USER'),
            HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','ALTER ANY ROLE'),
            CASE WHEN ORIGINAL_LOGIN()=SUSER_SNAME() THEN 0 ELSE 1 END,
            CASE WHEN ORIGINAL_LOGIN()=? THEN 1 ELSE 0 END,
            ORIGINAL_LOGIN(),CONVERT(VARCHAR(170),SUSER_SID(),2)""",
            (release,),
        )[0]
        if any(value != 0 for value in values):
            raise ValueError("SQL Server table inputs require an ordinary source account")
        mappings = [user for user in self._user_mappings() if user[1:] == (sid, login)]
        if len(mappings) != 1:
            raise ValueError(
                "SQL Server table inputs require an ordinary source user mapping "
                "in the archive database on the archive server"
            )
        resource = "af_source_" + uuid.uuid4().hex

        def release_lock():
            if _execute(
                self.reader,
                """DECLARE @result INT;
                EXEC @result=sys.sp_releaseapplock @Resource=?,@LockOwner='Session',
                    @DbPrincipal='public'; SELECT @result""",
                (resource,),
            ) != [(0,)]:
                raise ValueError("SQL Server source identity lock cleanup failed")

        with _cleanup_on_exit(release_lock, label="SQL Server source identity cleanup failed"):
            acquired = _execute(
                self.reader,
                """DECLARE @result INT;
                EXEC @result=sys.sp_getapplock @Resource=?,@LockMode='Exclusive',
                    @LockOwner='Session',@LockTimeout=0,@DbPrincipal='public'; SELECT @result""",
                (resource,),
            )
            if acquired not in ([(0,)], [(1,)]):
                raise ValueError("SQL Server source server identity could not be verified")
            database = self._database(self.reader)["name"].replace("]", "]]")
            if _execute(
                connection,
                f"""EXEC [{database}].sys.sp_executesql
                N'SELECT APPLOCK_TEST(N''public'',@resource,N''Exclusive'',N''Session'')',
                N'@resource NVARCHAR(255)',?""",
                (resource,),
            ) != [(0,)]:
                raise ValueError(
                    "SQL Server table inputs require independent sessions on the same "
                    "archive server; source server identity could not be verified"
                )
        self._source_users = tuple(sorted({*self._source_users, mappings[0]}))

    def _name(self, record):
        key = record["capture_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", key) or record["object"] != (
            self.schema + ".af_input_" + key
        ):
            raise ValueError("snapshot belongs to a different destination")
        return f"[{self.schema}].[af_input_{key}]"

    @staticmethod
    def _columns(schema):
        result = []
        for field in schema:
            _archive_name(field.name)
            dtype = field.type
            column = {"name": field.name, "nullable": field.nullable}
            if pa.types.is_signed_integer(dtype):
                kind = "bigint"
            elif pa.types.is_unsigned_integer(dtype):
                kind = "decimal"
                column.update(precision=20, scale=0)
            elif pa.types.is_boolean(dtype):
                kind = "bit"
            elif pa.types.is_float32(dtype):
                kind = "real"
            elif pa.types.is_float64(dtype):
                kind = "float"
            elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
                kind = "text"
            elif pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
                kind = "binary"
            elif pa.types.is_date32(dtype):
                kind = "date"
            elif pa.types.is_timestamp(dtype) and dtype.unit in {"s", "ms", "us"}:
                kind = "timestamp"
            elif pa.types.is_decimal(dtype) and 0 <= dtype.scale <= dtype.precision <= 38:
                kind = "decimal"
                column.update(precision=dtype.precision, scale=dtype.scale)
            else:
                raise ValueError(f"SQL Server snapshot transport cannot preserve {dtype}")
            column["kind"] = kind
            result.append(column)
        return result

    def _operation(self, connection, action, record, *, details=None):
        import json

        self._name(record)
        _execute(
            connection,
            f"EXEC [{self.schema}].af_snapshot ?,?,?",
            (
                action,
                record["capture_id"],
                json.dumps(details if details is not None else record.get("incarnation", {})),
            ),
        )

    def _incarnation(self, connection, record):
        import json

        name = record["object"]
        row = _execute(
            connection,
            """SELECT t.object_id,t.create_date,
            coalesce(t.principal_id,s.principal_id),CONVERT(VARCHAR(32),e.value),
            CONVERT(NVARCHAR(3750),sources.value)
            FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id
            LEFT JOIN sys.extended_properties e ON e.class=1 AND e.major_id=t.object_id
                AND e.minor_id=0 AND e.name='autoforecast_seal'
            LEFT JOIN sys.extended_properties sources ON sources.class=1
                AND sources.major_id=t.object_id AND sources.minor_id=0
                AND sources.name='autoforecast_sources'
            WHERE t.object_id=OBJECT_ID(?)""",
            (name,),
        )
        if len(row) != 1 or not row[0][3]:
            raise FileNotFoundError("retained input is missing or has no original seal")
        oid, created, owner, seal, sources = row[0]
        if sources is None:
            raise ValueError("SQL Server source identity binding is missing")
        sources = json.loads(sources)
        if type(sources) is not list:
            raise ValueError("SQL Server source identity binding is invalid")
        columns = _execute(
            connection,
            """SELECT name,system_type_id,max_length,precision,scale,
            is_nullable,collation_name,is_computed,is_identity FROM sys.columns
            WHERE object_id=? ORDER BY column_id""",
            (oid,),
        )
        return {
            "database": self._database(connection),
            "oid": oid,
            "created": created.isoformat(),
            "owner": owner,
            "seal": seal,
            "columns": [list(column) for column in columns],
            "sources": sources,
        }

    def _write(self, connection, action, record):
        self._idle(connection)
        if self._database(connection) != record["incarnation"]["database"]:
            raise ValueError("retained input database changed")
        try:
            _execute(connection, "BEGIN TRANSACTION")
            self._operation(connection, action, record)
            _execute(connection, "COMMIT")
        except BaseException as error:
            try:
                _execute(connection, "IF @@TRANCOUNT>0 ROLLBACK")
            except Exception as rollback:
                error.add_note(f"Capture transaction cleanup failed: {type(rollback).__name__}")
            raise

    def capture(self, path, *, capture_id, checkpoint=None):
        import datetime as dt
        from decimal import Decimal

        from autoforecast.snapshots import ColumnLayout, arrow_schema

        self.preflight()
        _require_arrow_streaming(self.reader)
        source_users = self._source_users
        record = {
            "object": self.schema + ".af_input_" + capture_id,
            "capture_id": capture_id,
            "incarnation": {"sources": [list(user) for user in source_users]},
        }
        target = self._name(record)
        files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
        schema, count = None, 0
        for file in files:
            with pq.ParquetFile(file) as source:
                current = source.schema_arrow.remove_metadata()
                if schema is not None and not schema.equals(current, check_metadata=False):
                    raise ValueError("native input file schemas differ")
                schema, count = current, count + source.metadata.num_rows
        if schema is None:
            raise ValueError("native inputs require typed Parquet files")
        layout = ColumnLayout(schema)
        columns = self._columns(layout.schema)
        try:
            _execute(self.writer, "BEGIN TRANSACTION")
            self._operation(
                self.writer, "begin", record,
                details={"columns": columns, "sources": [list(user) for user in source_users]},
            )
            statement = "INSERT INTO " + target + " VALUES (" + ",".join(
                "CONVERT(NVARCHAR(MAX),?)" if column["kind"] == "text" else "?"
                for column in columns
            ) + ")"
            inserted = 0
            with self.writer.cursor() as cursor:
                for file in files:
                    with pq.ParquetFile(file) as source:
                        for batch in source.iter_batches(batch_size=8192):
                            values = []
                            for row in batch.to_pylist():
                                converted = []
                                for field, value in zip(
                                    layout.schema, layout.encode(row), strict=True
                                ):
                                    if value is not None and (
                                        pa.types.is_string(field.type)
                                        or pa.types.is_large_string(field.type)
                                    ):
                                        # Binary parameters carry embedded NUL with an
                                        # explicit length; SQL Server decodes UTF-16LE.
                                        value = value.encode("utf-16le")
                                    elif value is not None and pa.types.is_unsigned_integer(
                                        field.type
                                    ):
                                        value = Decimal(value)
                                    elif (
                                        value is not None
                                        and pa.types.is_timestamp(field.type)
                                        and field.type.tz is not None
                                    ):
                                        value = value.astimezone(dt.UTC).replace(tzinfo=None)
                                    converted.append(value)
                                values.append(tuple(converted))
                            cursor.executemany(statement, values)
                            inserted += len(values)
            if (
                inserted != count
                or _execute(self.writer, "SELECT COUNT_BIG(*) FROM " + target)[0][0] != count
            ):
                raise ValueError("native input row count changed during transport")
            self._operation(self.writer, "seal", record)
            record.update(
                incarnation=self._incarnation(self.writer, record),
                arrow_schema=arrow_schema(schema),
                rows=count,
            )
            if checkpoint is not None:
                checkpoint(record)
            if (
                type(record.get("incarnation")) is not dict
                or record["incarnation"].get("sources") != [list(user) for user in source_users]
            ):
                raise ValueError("SQL Server source identity binding changed at checkpoint")
            _execute(self.writer, "COMMIT")
        except BaseException as error:
            try:
                _execute(self.writer, "IF @@TRANCOUNT>0 ROLLBACK")
            except Exception as rollback:
                error.add_note(f"Capture transaction cleanup failed: {type(rollback).__name__}")
            raise
        self.inspect(record)
        return record

    def inspect(self, record, *, retained=False):
        self._name(record)
        owner, _, _, _ = self._policy()
        self._check_sources(record["incarnation"].get("sources"))
        if self._incarnation(self.reader, record) != record["incarnation"]:
            raise ValueError("native input incarnation or schema changed")
        self._protected_table(self.reader, record["object"], owner)
        saved = _execute(
            self.reader,
            f"SELECT object_id,seal,state FROM [{self.schema}].af_captures " "WHERE capture_id=?",
            (record["capture_id"],),
        )
        if (
            len(saved) != 1
            or saved[0][:2] != (record["incarnation"]["oid"], record["incarnation"]["seal"])
            or saved[0][2] not in ({"retained"} if retained else {"sealed", "retained"})
        ):
            raise ValueError("capture identity or retained state differs")
        self._source_users = tuple(sorted({
            *self._source_users,
            *(tuple(source) for source in record["incarnation"]["sources"]),
        }))

    def retain(self, record):
        self.inspect(record)
        self._write(self.writer, "retain", record)
        self.inspect(record, retained=True)

    def _check_release_object(self, record):
        self._name(record)
        self._check_sources(record["incarnation"].get("sources"))
        if _execute(self.reader, "SELECT OBJECT_ID(?)", (record["object"],))[0][0] is not None:
            if self._incarnation(self.reader, record) != record["incarnation"]:
                raise ValueError("native input incarnation or schema changed")

    def drop_unretained(self, record):
        self._policy()
        self._check_release_object(record)
        self._write(self.writer, "drop_unretained", record)

    def read(self, record, destination):
        import datetime as dt

        from autoforecast.snapshots import ColumnLayout, restore_schema

        self.inspect(record)
        _require_arrow_streaming(self.reader)
        schema = restore_schema(record["arrow_schema"])
        layout = ColumnLayout(schema)
        destination.mkdir(parents=True)
        with (
            self.reader.cursor() as cursor,
            pq.ParquetWriter(destination / "part.parquet", schema) as writer,
        ):
            cursor.execute("SELECT * FROM " + self._name(record))
            with cursor.arrow_reader(batch_size=8192) as batches:
                for batch in batches:
                    values = []
                    rows = zip(*(column.to_pylist() for column in batch.columns), strict=True)
                    for row in rows:
                        decoded = []
                        for field, value in zip(layout.schema, row, strict=True):
                            if value is not None and pa.types.is_unsigned_integer(field.type):
                                value = int(value)
                            elif (
                                value is not None
                                and pa.types.is_timestamp(field.type)
                                and field.type.tz is not None
                            ):
                                value = value.replace(tzinfo=dt.UTC)
                            decoded.append(value)
                        values.append(layout.decode(decoded))
                    writer.write_table(pa.Table.from_pylist(values, schema=schema))
        self.inspect(record)

    def release_preflight(self):
        self._idle(self.releaser)
        _, _, _, release = self._policy()
        if _execute(self.releaser, "SELECT ORIGINAL_LOGIN(),USER_NAME()")[0] != (release, release):
            raise ValueError("archive cleanup requires its separate release account")
        if self._database(self.releaser) != self._database(self.reader):
            raise ValueError("archive connections reach different databases")

    def release(self, record):
        self.release_preflight()
        self._check_release_object(record)
        self._write(self.releaser, "release", record)
