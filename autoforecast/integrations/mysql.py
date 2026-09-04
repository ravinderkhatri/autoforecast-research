"""Bounded MySQL source capture using one InnoDB read view."""

from __future__ import annotations

import codecs
import datetime as dt
import pathlib
import re
import uuid

import pyarrow as pa
import pyarrow.parquet as pq


def connect(dsn):
    """Open a named MySQL URI; advanced deployments can supply a configured client."""
    from urllib.parse import unquote, urlsplit

    import pymysql

    address = urlsplit(dsn)
    if (address.scheme != "mysql" or not address.hostname or not address.username
            or not address.path.strip("/") or "/" in address.path[1:]
            or address.query or address.fragment):
        raise ValueError("MySQL DSN must be mysql://user:password@host:port/database; "
                         "supply a configured client for additional driver settings")
    return pymysql.connect(host=address.hostname, port=address.port or 3306,
                           user=unquote(address.username), password=unquote(address.password or ""),
                           database=unquote(address.path[1:]), charset="utf8mb4", autocommit=True)


def _source_schema(fields):
    """Map driver-declared types before fetching rows, including empty SELECT results."""
    from pymysql.constants import FIELD_TYPE, FLAG

    integers = {FIELD_TYPE.TINY: (pa.int8(), pa.uint8()),
                FIELD_TYPE.SHORT: (pa.int16(), pa.uint16()),
                FIELD_TYPE.LONG: (pa.int32(), pa.uint32()),
                FIELD_TYPE.INT24: (pa.int32(), pa.uint32()),
                FIELD_TYPE.LONGLONG: (pa.int64(), pa.uint64())}
    text = {FIELD_TYPE.STRING, FIELD_TYPE.VAR_STRING, FIELD_TYPE.VARCHAR,
            FIELD_TYPE.TINY_BLOB, FIELD_TYPE.MEDIUM_BLOB, FIELD_TYPE.LONG_BLOB, FIELD_TYPE.BLOB}
    result = []
    for field in fields:
        kind = field.type_code
        unsigned = bool(field.flags & FLAG.UNSIGNED)
        if kind in integers:
            dtype = integers[kind][unsigned]
        elif kind in {FIELD_TYPE.FLOAT, FIELD_TYPE.DOUBLE}:
            dtype = pa.float32() if kind == FIELD_TYPE.FLOAT else pa.float64()
        elif kind in {FIELD_TYPE.DECIMAL, FIELD_TYPE.NEWDECIMAL}:
            # The protocol length includes a sign (unless unsigned) and decimal point.
            precision = field.length - int(not unsigned) - int(field.scale > 0)
            if not 0 <= field.scale <= precision <= 38:
                raise ValueError("cast MySQL decimals to an explicit precision no larger than 38")
            dtype = pa.decimal128(precision, field.scale)
        elif kind in text:
            dtype = pa.binary() if field.charsetnr == 63 else pa.string()
        elif kind in {FIELD_TYPE.DATE, FIELD_TYPE.NEWDATE}:
            dtype = pa.date32()
        elif kind in {FIELD_TYPE.DATETIME, FIELD_TYPE.TIMESTAMP}:
            dtype = pa.timestamp("us", tz="UTC" if kind == FIELD_TYPE.TIMESTAMP else None)
        else:
            raise ValueError(f"source transport cannot preserve MySQL field type {kind}")
        result.append(pa.field(field.name, dtype, nullable=not bool(field.flags & FLAG.NOT_NULL)))
    if len({field.name for field in result}) != len(result):
        raise ValueError("source SELECT must give every output column a distinct name")
    return pa.schema(result)


def _transport_charset(connection):
    """Check driver and server encodings; session SQL can leave the driver label stale."""
    try:
        encoding = codecs.lookup(connection.encoding).name
    except (AttributeError, LookupError, TypeError) as error:
        raise ValueError(
            "MySQL transport requires a utf8mb4 driver using UTF-8 encoding",
        ) from error
    if connection.charset != "utf8mb4" or encoding != "utf-8":
        raise ValueError("MySQL transport requires a utf8mb4 driver using UTF-8 encoding")
    with connection.cursor() as cursor:
        cursor.execute("SELECT @@session.character_set_client, "
                       "@@session.character_set_connection, @@session.character_set_results")
        if cursor.fetchone() != ("utf8mb4", "utf8mb4", "utf8mb4"):
            raise ValueError("MySQL transport requires utf8mb4 session character sets")


def read_sources(connection, *, queries, tables, output_root):
    """Read declared persistent InnoDB tables in one repeatable-read transaction.

    Writes and DDL never run inside this source transaction. Its read view ends
    when capture finishes; the resulting files, not a timestamp, retain the rows.
    All result pages are bounded and consumed once. Caller connection settings are
    restored, and a failure never silently retries against a newer read view.
    """
    import pymysql
    from pymysql.constants import SERVER_STATUS
    from sqlglot import Dialect, TokenType

    from autoforecast.snapshots import bind_source_queries

    if (not connection.get_autocommit()
            or connection.server_status & SERVER_STATUS.SERVER_STATUS_IN_TRANS):
        raise ValueError("source capture needs an idle autocommit connection")
    if connection.charset != "utf8mb4":
        raise ValueError("MySQL source capture requires a utf8mb4 connection")
    _transport_charset(connection)
    with connection.cursor() as cursor:
        cursor.execute("SELECT @@session.sql_mode")
        modes = set(cursor.fetchone()[0].upper().split(","))
    if modes.intersection({"ANSI_QUOTES", "NO_BACKSLASH_ESCAPES"}):
        raise ValueError("MySQL source SQL requires ANSI_QUOTES and NO_BACKSLASH_ESCAPES off")
    names = {}
    for alias, name in tables.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", name,
        ):
            raise ValueError("MySQL source tables require database.table")
        names[alias] = "`.`".join(name.split("."))
    bound = bind_source_queries(queries, tables, dialect="mysql", sources={
        alias: "`" + name + "`" for alias, name in names.items()
    })
    with connection.cursor() as cursor:
        cursor.execute("SELECT @@session.time_zone")
        timezone = cursor.fetchone()[0]
    result = {}
    from contextlib import ExitStack

    from autoforecast.snapshots import _cleanup_on_exit

    def restore_timezone():
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION time_zone=%s", (timezone,))

    cleanup = ExitStack()
    cleanup.callback(restore_timezone)
    cleanup.callback(connection.rollback)
    with _cleanup_on_exit(cleanup.close, label="MySQL source cleanup failed"):
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION time_zone = '+00:00'")
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
            provenance = {}
            for alias, name in names.items():
                # Validate before opening the table, then recheck after taking its
                # transaction-long metadata lock so a concurrent replacement fails.
                for locked in (False, True):
                    if locked:
                        cursor.execute("SELECT * FROM `" + name + "` LIMIT 0")
                    cursor.execute("SELECT TABLE_TYPE, ENGINE FROM information_schema.TABLES "
                                   "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
                                   tables[alias].split("."))
                    if cursor.fetchone() != ("BASE TABLE", "InnoDB"):
                        raise ValueError("MySQL capture requires persistent InnoDB base tables")
                cursor.execute("SHOW CREATE TABLE `" + name + "`")
                definition = cursor.fetchone()[1]
                if not re.match(r"CREATE TABLE\s", definition, re.IGNORECASE):
                    raise ValueError("MySQL capture requires persistent InnoDB base tables")
                provenance[alias] = {"table": tables[alias], "definition": definition}
            cursor.execute("SELECT @@server_uuid, UTC_TIMESTAMP(6)")
            server, observed_at = cursor.fetchone()
        for index, (name, query) in enumerate(bound.items()):
            # Embed one validated statement without changing its quoted text.
            for token in reversed(Dialect.get_or_raise("mysql").tokenize(query)):
                if token.token_type is not TokenType.SEMICOLON:
                    break
                query = query[:token.start] + query[token.end + 1:]
            # End line comments before adding the derived-table closing bracket.
            query = "\n" + query + "\n"
            # MySQL's text protocol rounds FLOAT output to six significant digits.
            # Describe without fetching rows, then widen only those fields for transport.
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM (" + query + ") AS af_source LIMIT 0")
                schema = _source_schema(cursor._result.fields)
            columns = []
            for field in schema:
                quoted = "`" + field.name.replace("`", "``") + "`"
                columns.append("CAST(" + quoted + " AS DOUBLE) AS " + quoted
                               if pa.types.is_float32(field.type) else quoted)
            transported = "SELECT " + ",".join(columns) + " FROM (" + query + ") AS af_source"
            with connection.cursor(pymysql.cursors.SSCursor) as cursor:
                cursor.execute(transported)
                destination = pathlib.Path(output_root) / ("query-" + str(index))
                destination.mkdir(parents=True)
                count = 0
                with pq.ParquetWriter(destination / "part.parquet", schema) as writer:
                    while rows := cursor.fetchmany(8192):
                        values = []
                        for row in rows:
                            values.append({field.name: (
                                value.replace(tzinfo=dt.UTC) if value is not None
                                and pa.types.is_timestamp(field.type) and field.type.tz == "UTC"
                                else value
                            ) for field, value in zip(schema, row, strict=True)})
                        batch = pa.Table.from_pylist(values, schema=schema)
                        writer.write_table(batch)
                        count += batch.num_rows
                result[name] = (destination, {
                    "kind": "mysql", "tables": provenance, "sql": queries[name], "rows": count,
                    "transport_sql": transported,
                    "capture_boundary": {"isolation": "REPEATABLE READ", "server_uuid": server,
                                         "observed_at": observed_at.isoformat()},
                })
    return result


def _archive_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,47}", value):
        raise ValueError("archive names require simple MySQL identifiers of at most 48 characters")
    return value


def _execute(connection, statement, values=None):
    with connection.cursor() as cursor:
        cursor.execute(statement, values)
        rows = cursor.fetchall()
        while cursor.nextset():
            pass
        return rows


def _snapshot_body(database, owner, registration, reader, release):
    # All substituted names are validated identifiers, never SQL supplied by the caller.
    for value in (database, owner, registration, reader, release):
        _archive_name(value)
    return f"""BEGIN
    DECLARE actor VARCHAR(128) DEFAULT SUBSTRING_INDEX(USER(),'@',1);
    DECLARE lock_name VARCHAR(64);
    DECLARE held BOOLEAN DEFAULT FALSE;
    DECLARE saved_seal VARCHAR(32);
    DECLARE saved_state VARCHAR(16);
    DECLARE saved_id BIGINT UNSIGNED;
    DECLARE actual_id BIGINT UNSIGNED;
    DECLARE actual_seal TEXT;
    DECLARE target VARCHAR(64);
    DECLARE declaration LONGTEXT DEFAULT '';
    DECLARE field_name VARCHAR(64);
    DECLARE field_type VARCHAR(64);
    DECLARE field_nullable BOOLEAN;
    DECLARE idx INT DEFAULT 0;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        IF held THEN DO RELEASE_LOCK(lock_name); END IF;
        RESIGNAL;
    END;
    IF actor NOT IN ('{registration}','{reader}','{release}') THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='archive caller is not an application account';
    END IF;
    IF (SELECT COUNT(*) FROM mysql.user WHERE User='{owner}' AND Host='%'
          AND account_locked='Y')<>1
       OR EXISTS(SELECT 1 FROM mysql.user WHERE User='{owner}' AND
          CONCAT(Select_priv,Insert_priv,Update_priv,Delete_priv,Create_priv,Drop_priv,
            Reload_priv,Shutdown_priv,File_priv,Grant_priv,References_priv,Index_priv,Alter_priv,
            Super_priv,Create_tmp_table_priv,Lock_tables_priv,Execute_priv,Repl_slave_priv,
            Create_view_priv,Create_routine_priv,Alter_routine_priv,Create_user_priv,Event_priv,
            Trigger_priv,Create_tablespace_priv,Create_role_priv,Drop_role_priv) REGEXP 'Y')
       OR EXISTS(SELECT 1 FROM mysql.global_grants WHERE USER='{owner}')
       OR EXISTS(SELECT 1 FROM mysql.user WHERE User IN
          ('{owner}','{registration}','{reader}','{release}') AND Host<>'%')
       OR EXISTS(SELECT 1 FROM mysql.user WHERE User='')
       OR @@mandatory_roles<>''
       OR EXISTS(SELECT 1 FROM mysql.role_edges WHERE TO_USER IN
          ('{owner}','{registration}','{reader}','{release}')
          OR FROM_USER IN ('{owner}','{release}'))
       OR EXISTS(SELECT 1 FROM mysql.proxies_priv WHERE Proxied_user IN
          ('{owner}','{registration}','{reader}','{release}')) THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='archive account isolation changed';
    END IF;
    IF EXISTS(SELECT 1 FROM mysql.user WHERE User IN ('{registration}','{reader}','{release}')
        AND CONCAT(Select_priv,Insert_priv,Update_priv,Delete_priv,Create_priv,Drop_priv,
            Reload_priv,Shutdown_priv,Process_priv,File_priv,Grant_priv,References_priv,
            Index_priv,Alter_priv,Super_priv,Create_tmp_table_priv,Lock_tables_priv,
            Execute_priv,Repl_slave_priv,Create_view_priv,Create_routine_priv,Alter_routine_priv,
            Create_user_priv,Event_priv,Trigger_priv,Create_tablespace_priv,
            Create_role_priv,Drop_role_priv) REGEXP 'Y')
       OR EXISTS(SELECT 1 FROM mysql.global_grants WHERE USER IN
          ('{registration}','{reader}','{release}') AND PRIV<>'SHOW_ROUTINE') THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT='archive application account has global privileges';
    END IF;
    IF EXISTS(SELECT 1 FROM mysql.user u WHERE User<>'{owner}'
        AND (Super_priv='N'
             OR EXISTS(SELECT 1 FROM mysql.role_edges e
                       WHERE e.FROM_USER=u.User AND e.FROM_HOST=u.Host)
             OR EXISTS(SELECT 1 FROM mysql.proxies_priv p
                       WHERE p.Proxied_user=u.User AND p.Proxied_host=u.Host))
        AND CONCAT(Insert_priv,Update_priv,Delete_priv,Create_priv,Drop_priv,Grant_priv,
            File_priv,Alter_priv,Super_priv,Create_user_priv,Create_role_priv,Drop_role_priv,
            Create_routine_priv,Alter_routine_priv,Create_view_priv,Trigger_priv) REGEXP 'Y')
       OR EXISTS(SELECT 1 FROM mysql.global_grants g JOIN mysql.user u
                 ON g.USER=u.User AND g.HOST=u.Host
                 WHERE u.User<>'{owner}' AND g.PRIV<>'SHOW_ROUTINE'
                 AND (u.Super_priv='N'
                      OR EXISTS(SELECT 1 FROM mysql.role_edges e
                                WHERE e.FROM_USER=u.User AND e.FROM_HOST=u.Host)
                      OR EXISTS(SELECT 1 FROM mysql.proxies_priv p
                                WHERE p.Proxied_user=u.User AND p.Proxied_host=u.Host))
                 AND (u.account_locked='N'
                      OR EXISTS(SELECT 1 FROM mysql.role_edges e
                                WHERE e.FROM_USER=u.User AND e.FROM_HOST=u.Host)
                      OR EXISTS(SELECT 1 FROM mysql.proxies_priv p
                                WHERE p.Proxied_user=u.User AND p.Proxied_host=u.Host)))
       OR EXISTS(SELECT 1 FROM mysql.db WHERE ('{database}' LIKE Db OR 'mysql' LIKE Db)
          AND User<>'{owner}'
          AND CONCAT(Insert_priv,Update_priv,Delete_priv,Create_priv,Drop_priv,Grant_priv,
              References_priv,Index_priv,Alter_priv,Create_tmp_table_priv,Lock_tables_priv,
              Create_view_priv,Create_routine_priv,Alter_routine_priv,Execute_priv,
              Event_priv,Trigger_priv) REGEXP 'Y')
       OR EXISTS(SELECT 1 FROM mysql.tables_priv WHERE Db IN ('{database}','mysql')
          AND User<>'{owner}'
          AND (REPLACE(REPLACE(Table_priv,'Select',''),'Insert','') NOT IN ('',',')
               OR (FIND_IN_SET('Insert',Table_priv)>0 AND
                   (Db<>'{database}' OR User<>'{registration}'
                    OR Table_name NOT REGEXP '^af_build_[0-9a-f]{{32}}$'))))
       OR EXISTS(SELECT 1 FROM mysql.columns_priv WHERE Db IN ('{database}','mysql')
          AND User<>'{owner}'
          AND Column_priv<>'Select') THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='archive mutation privileges changed';
    END IF;
    IF operation='policy' THEN
        SELECT @@server_uuid;
    ELSEIF operation='check_source' AND actor='{registration}' THEN
        IF NOT EXISTS(SELECT 1 FROM mysql.user WHERE
            CONCAT(User,'@',Host)=JSON_UNQUOTE(JSON_EXTRACT(details,'$.source_account')))
           OR EXISTS(
            WITH RECURSIVE authority_edges AS (
                SELECT FROM_USER,FROM_HOST,TO_USER,TO_HOST FROM mysql.role_edges
                UNION DISTINCT
                SELECT Proxied_user,Proxied_host,User,Host FROM mysql.proxies_priv
            ), source_roles AS (
                SELECT User,Host FROM mysql.user WHERE
                    CONCAT(User,'@',Host)=JSON_UNQUOTE(JSON_EXTRACT(details,'$.source_account'))
                UNION DISTINCT
                SELECT e.FROM_USER,e.FROM_HOST FROM authority_edges e JOIN source_roles r
                    ON e.TO_USER=r.User AND e.TO_HOST=r.Host
            )
            SELECT 1 FROM source_roles r JOIN mysql.user u
                ON r.User=u.User AND r.Host=u.Host
            WHERE u.User IN ('{owner}','{registration}','{release}')
               OR CONCAT(Insert_priv,Update_priv,Delete_priv,Create_priv,Drop_priv,
                    Reload_priv,Shutdown_priv,File_priv,Grant_priv,References_priv,
                    Index_priv,Alter_priv,Super_priv,Create_tmp_table_priv,Lock_tables_priv,
                    Execute_priv,Repl_slave_priv,Create_view_priv,Create_routine_priv,
                    Alter_routine_priv,Create_user_priv,Event_priv,Trigger_priv,
                    Create_tablespace_priv,Create_role_priv,Drop_role_priv) REGEXP 'Y'
               OR EXISTS(SELECT 1 FROM mysql.global_grants g WHERE g.USER=u.User
                    AND g.HOST=u.Host AND g.PRIV<>'SHOW_ROUTINE')
               OR EXISTS(SELECT 1 FROM mysql.db d WHERE d.User=u.User AND d.Host=u.Host
                    AND ('{database}' LIKE d.Db OR 'mysql' LIKE d.Db)
                    AND CONCAT(d.Insert_priv,d.Update_priv,d.Delete_priv,d.Create_priv,
                        d.Drop_priv,d.Grant_priv,d.References_priv,d.Index_priv,d.Alter_priv,
                        d.Create_tmp_table_priv,d.Lock_tables_priv,d.Create_view_priv,
                        d.Create_routine_priv,d.Alter_routine_priv,d.Execute_priv,
                        d.Event_priv,d.Trigger_priv) REGEXP 'Y')
               OR EXISTS(SELECT 1 FROM mysql.tables_priv t WHERE t.User=u.User AND t.Host=u.Host
                    AND t.Db IN ('{database}','mysql') AND t.Table_priv NOT IN ('','Select'))
               OR EXISTS(SELECT 1 FROM mysql.columns_priv c WHERE c.User=u.User AND c.Host=u.Host
                    AND c.Db IN ('{database}','mysql') AND c.Column_priv<>'Select')
               OR EXISTS(SELECT 1 FROM mysql.procs_priv p WHERE p.User=u.User AND p.Host=u.Host
                    AND p.Db IN ('{database}','mysql') AND p.Proc_priv<>''
                    AND NOT (p.Db='{database}' AND p.Routine_name='af_snapshot'
                             AND p.Routine_type='PROCEDURE' AND p.Proc_priv='Execute'))
           ) THEN
            SIGNAL SQLSTATE '45000'
                SET MESSAGE_TEXT='source account has archive administration rights';
        END IF;
    ELSE
        IF capture_key NOT REGEXP '^[0-9a-f]{{32}}$' OR
           JSON_UNQUOTE(JSON_EXTRACT(details,'$.seal')) NOT REGEXP '^[0-9a-f]{{32}}$' THEN
            SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='invalid capture identity';
        END IF;
        SET lock_name=SHA2(CONCAT('{database}',':',capture_key),256);
        SET held=GET_LOCK(lock_name,0);
        IF NOT held THEN
            SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='capture operation is already active';
        END IF;
        IF operation='begin' AND actor='{registration}' THEN
            IF JSON_TYPE(JSON_EXTRACT(details,'$.columns'))<>'ARRAY'
               OR JSON_LENGTH(details,'$.columns')<1 THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='capture columns are required';
            END IF;
            WHILE idx<JSON_LENGTH(details,'$.columns') DO
                SET field_name=JSON_UNQUOTE(
                    JSON_EXTRACT(details,CONCAT('$.columns[',idx,'].name')));
                SET field_type=JSON_UNQUOTE(
                    JSON_EXTRACT(details,CONCAT('$.columns[',idx,'].type')));
                SET field_nullable=JSON_EXTRACT(details,CONCAT('$.columns[',idx,'].nullable'))=true;
                IF field_name NOT REGEXP '^[A-Za-z_][A-Za-z0-9_]{{0,63}}$' OR
                   NOT (field_type IN ('BIGINT','BIGINT UNSIGNED','TINYINT','FLOAT','DOUBLE',
                       'LONGTEXT','LONGBLOB','DATE','DATETIME(6)') OR
                       field_type REGEXP '^DECIMAL[(][0-9]{{1,2}},[0-9]{{1,2}}[)]$') THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='unsupported capture column';
                END IF;
                SET declaration=CONCAT(declaration,IF(idx=0,'',','),'`',field_name,'` ',
                                       field_type,IF(field_nullable,' NULL',' NOT NULL'));
                SET idx=idx+1;
            END WHILE;
            SET saved_seal=JSON_UNQUOTE(JSON_EXTRACT(details,'$.seal'));
            INSERT INTO `{database}`.af_captures VALUES(capture_key,saved_seal,NULL,'building');
            SET @af_statement=CONCAT('CREATE TABLE `{database}`.`af_build_',capture_key,
                '` (',declaration,') ENGINE=InnoDB COMMENT=',QUOTE(saved_seal));
            PREPARE af_command FROM @af_statement;
            EXECUTE af_command; DEALLOCATE PREPARE af_command;
            SELECT TABLE_ID INTO actual_id FROM information_schema.INNODB_TABLES
                WHERE NAME=CONCAT('{database}/af_build_',capture_key);
            UPDATE `{database}`.af_captures SET object_id=actual_id WHERE capture_id=capture_key;
            SET @af_statement=CONCAT('GRANT INSERT ON `{database}`.`af_build_',capture_key,
                                    '` TO ''{registration}''@''%''');
            PREPARE af_command FROM @af_statement;
            EXECUTE af_command; DEALLOCATE PREPARE af_command;
        ELSE
            SELECT seal,object_id,state INTO saved_seal,saved_id,saved_state
                FROM `{database}`.af_captures WHERE capture_id=capture_key;
            IF saved_seal IS NULL OR saved_seal<>JSON_UNQUOTE(JSON_EXTRACT(details,'$.seal')) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='capture identity is missing or different';
            END IF;
            SET target=CONCAT(IF(saved_state='building','af_build_','af_input_'),capture_key);
            IF saved_state='released' AND EXISTS(SELECT 1 FROM information_schema.TABLES
                WHERE TABLE_SCHEMA='{database}' AND TABLE_NAME IN
                    (CONCAT('af_build_',capture_key),CONCAT('af_input_',capture_key))) THEN
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='released capture name was reused';
            END IF;
            IF saved_state<>'released' THEN
                SELECT TABLE_ID INTO actual_id FROM information_schema.INNODB_TABLES
                    WHERE NAME=CONCAT('{database}/',target);
                SELECT TABLE_COMMENT INTO actual_seal FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA='{database}' AND TABLE_NAME=target AND ENGINE='InnoDB';
                IF saved_state='building' AND actual_id IS NULL THEN
                    SET target=CONCAT('af_input_',capture_key);
                    SELECT TABLE_ID INTO actual_id FROM information_schema.INNODB_TABLES
                        WHERE NAME=CONCAT('{database}/',target);
                    SELECT TABLE_COMMENT INTO actual_seal FROM information_schema.TABLES
                        WHERE TABLE_SCHEMA='{database}' AND TABLE_NAME=target AND ENGINE='InnoDB';
                END IF;
                IF saved_state='building' AND saved_id IS NULL AND actual_id IS NOT NULL
                   AND actual_seal=saved_seal THEN
                    SET saved_id=actual_id;
                    UPDATE `{database}`.af_captures SET object_id=saved_id
                        WHERE capture_id=capture_key;
                END IF;
                IF actual_id IS NULL AND
                   ((operation='release' AND actor='{release}') OR
                    (operation='drop_unretained' AND actor='{registration}'
                     AND saved_state IN ('building','sealed')))
                   AND NOT EXISTS(SELECT 1 FROM information_schema.TABLES
                       WHERE TABLE_SCHEMA='{database}' AND TABLE_NAME IN
                           (CONCAT('af_build_',capture_key),CONCAT('af_input_',capture_key))) THEN
                    UPDATE `{database}`.af_captures SET state='released'
                        WHERE capture_id=capture_key;
                    SET saved_state='released';
                ELSEIF actual_id IS NULL OR saved_id IS NULL
                    OR actual_id<>saved_id OR actual_seal<>saved_seal THEN
                    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='capture object identity changed';
                END IF;
            END IF;
            IF operation='seal' AND actor='{registration}' AND saved_state='building' THEN
                SET @af_statement=CONCAT('REVOKE INSERT ON `{database}`.`',target,
                                        '` FROM ''{registration}''@''%''');
                PREPARE af_command FROM @af_statement;
                EXECUTE af_command; DEALLOCATE PREPARE af_command;
                SET @af_statement=CONCAT('RENAME TABLE `{database}`.`',target,
                                        '` TO `{database}`.`af_input_',capture_key,'`');
                PREPARE af_command FROM @af_statement;
                EXECUTE af_command; DEALLOCATE PREPARE af_command;
                UPDATE `{database}`.af_captures SET state='sealed' WHERE capture_id=capture_key;
            ELSEIF operation='retain' AND actor='{registration}'
                   AND saved_state IN ('sealed','retained') THEN
                UPDATE `{database}`.af_captures SET state='retained' WHERE capture_id=capture_key;
            ELSEIF (operation='release' AND actor='{release}') OR
                   (operation='drop_unretained' AND actor='{registration}'
                    AND saved_state IN ('building','sealed','released')) THEN
                IF saved_state<>'released' THEN
                    IF EXISTS(SELECT 1 FROM mysql.tables_priv WHERE Db='{database}'
                        AND Table_name=CONCAT('af_build_',capture_key) AND User='{registration}'
                        AND FIND_IN_SET('Insert',Table_priv)>0) THEN
                        SET @af_statement=CONCAT('REVOKE INSERT ON `{database}`.`af_build_',
                            capture_key,'` FROM ''{registration}''@''%''');
                        PREPARE af_command FROM @af_statement;
                        EXECUTE af_command; DEALLOCATE PREPARE af_command;
                    END IF;
                    SET @af_statement=CONCAT('DROP TABLE `{database}`.`',target,'`');
                    PREPARE af_command FROM @af_statement;
                    EXECUTE af_command; DEALLOCATE PREPARE af_command;
                    UPDATE `{database}`.af_captures SET state='released'
                        WHERE capture_id=capture_key;
                END IF;
            ELSEIF operation='inspect' AND saved_state IN ('sealed','retained') THEN
                SELECT saved_state,actual_id;
            ELSE
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='capture operation is not permitted';
            END IF;
        END IF;
        DO RELEASE_LOCK(lock_name);
    END IF;
END"""


def provision_snapshot_database(connection, *, database, owner, registration_role,
                                reader_role, release_role):
    """Provision a dedicated archive with existing distinct user@% accounts.

    Run once as an administrator. The owner must be locked and must not be a role
    available to an application account. The archive must not already exist.
    """
    names = (database, owner, registration_role, reader_role, release_role)
    for name in names:
        _archive_name(name)
    if len(set(names[1:])) != 4:
        raise ValueError("archive accounts must be distinct")
    if not connection.get_autocommit():
        raise ValueError("archive provisioning requires an idle autocommit connection")
    for name in names[1:]:
        if _execute(connection, "SELECT Host,account_locked FROM mysql.user WHERE User=%s",
                    (name,)) != (("%", "Y" if name == owner else "N"),):
            raise ValueError("provide distinct user@% accounts and a locked owner")
    _execute(connection, f"CREATE DATABASE `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_bin")
    _execute(connection, f"GRANT ALL ON `{database}`.* TO '{owner}'@'%' WITH GRANT OPTION")
    _execute(connection, f"GRANT SELECT ON mysql.* TO '{owner}'@'%'")
    _execute(connection, f"GRANT PROCESS ON *.* TO '{owner}'@'%'")
    _execute(connection, f'''CREATE TABLE `{database}`.af_captures (
        capture_id VARCHAR(32) PRIMARY KEY, seal VARCHAR(32) NOT NULL,
        object_id BIGINT UNSIGNED, state VARCHAR(16) NOT NULL
    ) ENGINE=InnoDB''')
    _execute(connection, f'''CREATE TABLE `{database}`.af_policy (
        owner VARCHAR(48), registration VARCHAR(48), reader VARCHAR(48), releaser VARCHAR(48)
    ) ENGINE=InnoDB''')
    _execute(connection, f"INSERT INTO `{database}`.af_policy VALUES(%s,%s,%s,%s)", names[1:])
    body = _snapshot_body(database, *names[1:])
    _execute(connection, f'''CREATE DEFINER='{owner}'@'%' PROCEDURE `{database}`.af_snapshot(
        IN operation VARCHAR(32), IN capture_key VARCHAR(32), IN details JSON
    ) SQL SECURITY DEFINER {body}''')
    for role in names[2:]:
        _execute(connection, f"GRANT SELECT ON `{database}`.* TO '{role}'@'%'")
        _execute(connection, f"GRANT EXECUTE ON PROCEDURE `{database}`.af_snapshot TO '{role}'@'%'")
        _execute(connection, f"GRANT SHOW_ROUTINE ON *.* TO '{role}'@'%'")


class MySQLSnapshots:
    """Retained typed InnoDB inputs in a separately provisioned protected archive."""

    backend = "mysql"

    def __init__(self, writer, reader, *, database, releaser=None):
        self.database = _archive_name(database)
        self.writer, self.reader, self.releaser = writer, reader, releaser

    @property
    def namespace(self):
        return self.database

    def _policy(self):
        _transport_charset(self.reader)
        rows = _execute(self.reader, f"SELECT * FROM `{self.database}`.af_policy")
        if len(rows) != 1 or len(set(rows[0])) != 4:
            raise ValueError("archive role policy changed")
        owner, registration, reader, release = rows[0]
        if _execute(self.reader, "SELECT CURRENT_USER()")[0][0] != reader + "@%":
            raise ValueError("archive requires its configured restricted reader")
        definition = _execute(self.reader, f"SHOW CREATE PROCEDURE `{self.database}`.af_snapshot")
        statement = definition[0][2]
        body = _snapshot_body(self.database, owner, registration, reader, release)
        if not statement.endswith(body) or f"DEFINER=`{owner}`@`%`" not in statement:
            raise ValueError("archive sealing operation changed")
        self._call("policy", connection=self.reader)
        return rows[0]

    def _call(self, operation, record=None, *, connection=None, details=None):
        import json

        record = record or {}
        client = connection or self.writer
        if record and record["incarnation"]["server"] != _execute(
            client, "SELECT @@server_uuid",
        )[0][0]:
            raise ValueError("retained input server changed")
        return _execute(client,
                        f"CALL `{self.database}`.af_snapshot(%s,%s,%s)",
                        (operation, record.get("capture_id"), json.dumps(
                            details if details is not None else record.get("incarnation", {}))))

    @staticmethod
    def _idle(connection):
        from pymysql.constants import SERVER_STATUS

        if (connection is None or not connection.get_autocommit()
                or connection.server_status & SERVER_STATUS.SERVER_STATUS_IN_TRANS
                or connection.charset != "utf8mb4"):
            raise ValueError("archive connections require idle utf8mb4 autocommit sessions")
        _transport_charset(connection)

    def preflight(self):
        self._idle(self.writer)
        self._idle(self.reader)
        policy = self._policy()
        if _execute(self.writer, "SELECT CURRENT_USER()")[0][0] != policy[1] + "@%":
            raise ValueError("archive requires its configured restricted registration account")
        if _execute(self.writer, "SELECT @@server_uuid") != _execute(self.reader,
                                                                    "SELECT @@server_uuid"):
            raise ValueError("archive connections reach different MySQL servers")
        modes = _execute(self.writer, "SELECT @@session.sql_mode")[0][0].split(",")
        if not {"STRICT_ALL_TABLES", "STRICT_TRANS_TABLES"}.intersection(modes):
            raise ValueError("MySQL capture requires strict SQL mode to refuse lossy writes")

    def check_source(self, connection):
        """Reject source credentials that also administer the same archive server."""
        import pymysql

        self._idle(connection)
        if _execute(connection, "SELECT @@server_uuid") == _execute(self.reader,
                                                                  "SELECT @@server_uuid"):
            account = _execute(connection, "SELECT CURRENT_USER()")[0][0]
            try:
                self._call("check_source", details={"source_account": account})
            except pymysql.Error as error:
                if len(error.args) < 2 or error.args[0] != 1644:
                    raise
                raise ValueError(error.args[1]) from error

    @staticmethod
    def _columns(schema):
        result = []
        for field in schema:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", field.name):
                raise ValueError("MySQL capture columns require simple identifiers")
            dtype = field.type
            if pa.types.is_signed_integer(dtype):
                native = "BIGINT"
            elif pa.types.is_unsigned_integer(dtype):
                native = "BIGINT UNSIGNED"
            elif pa.types.is_boolean(dtype):
                native = "TINYINT"
            elif pa.types.is_float32(dtype):
                # A DOUBLE retains every float32 value and avoids text-protocol rounding.
                native = "DOUBLE"
            elif pa.types.is_float64(dtype):
                native = "DOUBLE"
            elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
                native = "LONGTEXT"
            elif pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
                native = "LONGBLOB"
            elif pa.types.is_date32(dtype):
                native = "DATE"
            elif pa.types.is_timestamp(dtype) and dtype.unit in {"s", "ms", "us"}:
                native = "DATETIME(6)"
            elif pa.types.is_decimal(dtype) and 0 <= dtype.scale <= 30 and dtype.precision <= 38:
                native = f"DECIMAL({dtype.precision},{dtype.scale})"
            else:
                raise ValueError(f"MySQL snapshot transport cannot preserve {dtype}")
            result.append({"name": field.name, "type": native, "nullable": field.nullable})
        return result

    def _name(self, record):
        key = record["capture_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", key) or record["object"] != (
            self.database + ".af_input_" + key
        ):
            raise ValueError("capture belongs to a different destination")
        return f"`{self.database}`.`af_input_{key}`"

    def capture(self, path, *, capture_id, checkpoint=None):
        from autoforecast.snapshots import ColumnLayout, arrow_schema

        self.preflight()
        record = {"object": self.database + ".af_input_" + capture_id, "capture_id": capture_id}
        self._name(record)
        files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
        schema, count = None, 0
        for file in files:
            with pq.ParquetFile(file) as source:
                current = source.schema_arrow.remove_metadata()
                if schema is not None and not schema.equals(current, check_metadata=False):
                    raise ValueError("native input file schemas differ")
                schema = current
                count += source.metadata.num_rows
        if schema is None:
            raise ValueError("native inputs require typed Parquet files")
        layout = ColumnLayout(schema)
        columns = self._columns(layout.schema)
        incarnation = {"server": _execute(self.reader, "SELECT @@server_uuid")[0][0],
                       "seal": uuid.uuid4().hex, "columns": columns}
        record.update(incarnation=incarnation, arrow_schema=arrow_schema(schema), rows=count)
        # MySQL DDL commits implicitly. Save the recovery identity before any DDL.
        if checkpoint is not None:
            checkpoint(record)
        try:
            _transport_charset(self.writer)
            self._call("begin", record)
            statement = (f"INSERT INTO `{self.database}`.`af_build_{capture_id}` VALUES (" +
                         ",".join(["%s"] * len(columns)) + ")")
            with self.writer.cursor() as cursor:
                for file in files:
                    with pq.ParquetFile(file) as source:
                        for batch in source.iter_batches(batch_size=8192):
                            values = []
                            for row in batch.to_pylist():
                                values.append([
                                    value.astimezone(dt.UTC).replace(tzinfo=None)
                                    if value is not None and pa.types.is_timestamp(field.type)
                                    and field.type.tz is not None else value
                                    for field, value in zip(layout.schema, layout.encode(row),
                                                           strict=True)
                                ])
                            cursor.executemany(statement, values)
            self._call("seal", record)
            self.inspect(record)
            if _execute(self.reader, "SELECT count(*) FROM " + self._name(record))[0][0] != count:
                raise ValueError("native input row count changed during transport")
            return record
        except BaseException as error:
            try:
                self.drop_unretained(record)
            except Exception as cleanup:
                error.add_note(f"Capture cleanup needs recovery: {type(cleanup).__name__}")
            raise

    def inspect(self, record, *, retained=False):
        self._name(record)
        self._policy()
        if record["incarnation"]["server"] != _execute(self.reader, "SELECT @@server_uuid")[0][0]:
            raise ValueError("retained input server changed")
        result = self._call("inspect", record, connection=self.reader)
        if retained and result[0][0] != "retained":
            raise ValueError("input capture is not retained")
        self._check_columns(record, "af_input_" + record["capture_id"])

    def _check_columns(self, record, table_name):
        from autoforecast.snapshots import ColumnLayout, restore_schema

        schema = restore_schema(record["arrow_schema"])
        expected = self._columns(ColumnLayout(schema).schema)
        if record["incarnation"]["columns"] != expected:
            raise ValueError("retained input schema binding changed")
        actual = _execute(self.reader, '''SELECT COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,
            CHARACTER_SET_NAME,COLLATION_NAME FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION''',
                          (self.database, table_name))
        expected_columns = tuple((column["name"], column["type"].lower(),
                                  "YES" if column["nullable"] else "NO",
                                  "utf8mb4" if column["type"] == "LONGTEXT" else None,
                                  "utf8mb4_bin" if column["type"] == "LONGTEXT" else None)
                                 for column in expected)
        if actual != expected_columns:
            raise ValueError("retained input columns changed")

    def _check_release_object(self, record):
        tables = _execute(
            self.reader,
            "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s "
            "AND TABLE_NAME IN (%s,%s)",
            (self.database, "af_input_" + record["capture_id"], "af_build_" + record["capture_id"]),
        )
        if len(tables) > 1:
            raise ValueError("capture object identity changed")
        for (name,) in tables:
            self._check_columns(record, name)

    def retain(self, record):
        self.inspect(record)
        self._call("retain", record)
        self.inspect(record, retained=True)

    def drop_unretained(self, record):
        self._name(record)
        if not self._absent(record):
            self._check_release_object(record)
            self._call("drop_unretained", record)

    def _absent(self, record):
        """A checkpoint may precede the first DDL; absence never authorizes deletion."""
        self._policy()
        if record["incarnation"]["server"] != _execute(self.reader, "SELECT @@server_uuid")[0][0]:
            raise ValueError("retained input server changed")
        return not _execute(self.reader, f"SELECT 1 FROM `{self.database}`.af_captures "
                            "WHERE capture_id=%s", (record["capture_id"],)) and not _execute(
            self.reader, "SELECT 1 FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s "
            "AND TABLE_NAME IN (%s,%s)", (self.database, "af_input_" + record["capture_id"],
                                         "af_build_" + record["capture_id"]),
        )

    def read(self, record, destination):
        import pymysql

        from autoforecast.snapshots import ColumnLayout, restore_schema

        self.inspect(record)
        schema = restore_schema(record["arrow_schema"])
        layout = ColumnLayout(schema)
        destination.mkdir(parents=True)
        with self.reader.cursor(pymysql.cursors.SSCursor) as cursor, pq.ParquetWriter(
            destination / "part.parquet", schema,
        ) as writer:
            cursor.execute("SELECT * FROM " + self._name(record))
            while rows := cursor.fetchmany(8192):
                values = []
                for row in rows:
                    decoded = []
                    for field, value in zip(layout.schema, row, strict=True):
                        if value is not None and pa.types.is_boolean(field.type):
                            value = bool(value)
                        elif (value is not None and pa.types.is_timestamp(field.type)
                              and field.type.tz is not None):
                            value = value.replace(tzinfo=dt.UTC)
                        decoded.append(value)
                    values.append(layout.decode(decoded))
                writer.write_table(pa.Table.from_pylist(values, schema=schema))
        self.inspect(record)

    def release_preflight(self):
        self._idle(self.releaser)
        policy = self._policy()
        if _execute(self.releaser, "SELECT CURRENT_USER()")[0][0] != policy[3] + "@%":
            raise ValueError("archive cleanup requires its separate release account")
        if _execute(self.releaser, "SELECT @@server_uuid") != _execute(self.reader,
                                                                      "SELECT @@server_uuid"):
            raise ValueError("archive connections reach different MySQL servers")

    def release(self, record):
        self._name(record)
        self.release_preflight()
        if not self._absent(record):
            self._check_release_object(record)
            self._call("release", record, connection=self.releaser)
