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


def _snapshot_operation(schema):
    """The installed operation accepts typed metadata, never caller-supplied SQL."""
    from psycopg import sql

    # All object names are quoted by format(%I); type text has a finite allowlist.
    # The protected owner creates the table, so no ownership handoff is needed.
    body = r"""
DECLARE
    namespace constant text := SCHEMA_LITERAL;
    policy record;
    saved record;
    actor oid := (SELECT oid FROM pg_catalog.pg_roles WHERE rolname=session_user);
    target text;
    field jsonb;
    columns text := '';
    kind text;
    relation oid;
    token text;
BEGIN
    EXECUTE format('SELECT * FROM %I.af_policy', namespace) INTO STRICT policy;
    IF actor NOT IN (policy.registration_role, policy.release_role) THEN
        RAISE EXCEPTION 'snapshot operation requires its configured account'
            USING ERRCODE='42501';
    END IF;
    IF capture_id IS NULL OR capture_id !~ '^[a-f0-9]{32}$' THEN
        RAISE EXCEPTION 'invalid capture ID';
    END IF;
    target := 'af_input_' || capture_id;
    IF operation = 'begin' THEN
        IF actor <> policy.registration_role THEN
            RAISE EXCEPTION 'only registration may begin a capture' USING ERRCODE='42501';
        END IF;
        IF jsonb_typeof(details) IS DISTINCT FROM 'array' OR jsonb_array_length(details)=0 THEN
            RAISE EXCEPTION 'capture requires typed columns';
        END IF;
        FOR field IN SELECT * FROM jsonb_array_elements(details) LOOP
            IF jsonb_typeof(field) IS DISTINCT FROM 'object'
               OR (field->>'name') IS NULL OR (field->>'name') !~ '^[A-Za-z_][A-Za-z0-9_]*$'
               OR octet_length(field->>'name') > 63
               OR jsonb_typeof(field->'nullable') IS DISTINCT FROM 'boolean' THEN
                RAISE EXCEPTION 'invalid capture column';
            END IF;
            kind := field->>'type';
            IF kind IS NULL OR (kind NOT IN
                ('BIGINT','BOOLEAN','REAL','DOUBLE PRECISION','TEXT','BYTEA','DATE',
                 'TIMESTAMP','TIMESTAMPTZ') AND kind !~ '^NUMERIC\([0-9]{1,2},[0-9]{1,2}\)$') THEN
                RAISE EXCEPTION 'unsupported capture column type';
            END IF;
            columns := columns || CASE WHEN columns='' THEN '' ELSE ', ' END ||
                format('%I %s%s', field->>'name', kind,
                       CASE WHEN (field->>'nullable')::boolean THEN '' ELSE ' NOT NULL' END);
        END LOOP;
        EXECUTE format('CREATE TABLE %I.%I (%s)', namespace, target, columns);
        relation := to_regclass(format('%I.%I', namespace, target));
        token := gen_random_uuid()::text;
        EXECUTE format('REVOKE ALL ON %I.%I FROM PUBLIC, %I, %I', namespace, target,
                       pg_get_userbyid(policy.registration_role),
                       pg_get_userbyid(policy.reader_role));
        EXECUTE format('GRANT SELECT ON %I.%I TO %I, %I', namespace, target,
                       pg_get_userbyid(policy.registration_role),
                       pg_get_userbyid(policy.reader_role));
        EXECUTE format('GRANT INSERT ON %I.%I TO %I', namespace, target,
                       pg_get_userbyid(policy.registration_role));
        EXECUTE format('INSERT INTO %I.af_captures VALUES ($1,$2,$3,$4,$5,$6)', namespace)
            USING capture_id, relation, token, actor, pg_current_xact_id(), 'building';
        RETURN;
    END IF;
    EXECUTE format('SELECT * FROM %I.af_captures WHERE capture_id=$1 FOR UPDATE', namespace)
        INTO saved USING capture_id;
    IF saved.capture_id IS NULL THEN
        IF operation IN ('drop_unretained','release') AND
           to_regclass(format('%I.%I', namespace, target)) IS NULL THEN
            RETURN;
        END IF;
        RAISE EXCEPTION 'capture state is missing';
    END IF;
    relation := to_regclass(format('%I.%I', namespace, target));
    IF relation IS DISTINCT FROM saved.object_oid THEN
        RAISE EXCEPTION 'refusing a different capture incarnation';
    END IF;
    EXECUTE format('LOCK TABLE %I.%I IN ACCESS EXCLUSIVE MODE', namespace, target);
    IF operation='seal' THEN
        IF actor <> saved.creator OR saved.state <> 'building'
           OR saved.transaction_id <> pg_current_xact_id() THEN
            RAISE EXCEPTION 'capture must be sealed in its creating transaction';
        END IF;
        EXECUTE format('REVOKE INSERT ON %I.%I FROM %I', namespace, target,
                       pg_get_userbyid(policy.registration_role));
        EXECUTE format('COMMENT ON TABLE %I.%I IS %L', namespace, target, saved.seal);
        EXECUTE format('UPDATE %I.af_captures SET state=$1 WHERE capture_id=$2', namespace)
            USING 'sealed', capture_id;
        RETURN;
    END IF;
    IF details->>'owner' IS DISTINCT FROM
       (SELECT relowner::text FROM pg_class WHERE oid=relation) OR
       details->'columns' IS DISTINCT FROM (SELECT jsonb_agg(
           a.attname || ':' || format_type(a.atttypid,a.atttypmod) || ':' || a.attnotnull::text
           ORDER BY a.attnum) FROM pg_attribute a WHERE a.attrelid=relation
           AND a.attnum>0 AND NOT a.attisdropped) THEN
        RAISE EXCEPTION 'capture ownership or schema differs';
    END IF;
    IF saved.state NOT IN ('sealed','retained') OR
       details->>'seal' IS DISTINCT FROM saved.seal OR
       details->>'oid' IS DISTINCT FROM saved.object_oid::text OR
       obj_description(relation,'pg_class') IS DISTINCT FROM saved.seal THEN
        RAISE EXCEPTION 'capture identity or sealed state differs';
    END IF;
    IF operation='retain' AND actor=policy.registration_role THEN
        EXECUTE format('UPDATE %I.af_captures SET state=$1 WHERE capture_id=$2', namespace)
            USING 'retained', capture_id;
    ELSIF operation='drop_unretained' AND actor=policy.registration_role THEN
        IF saved.state='retained' THEN
            RAISE EXCEPTION 'registration cannot delete a retained capture' USING ERRCODE='42501';
        END IF;
        EXECUTE format('DROP TABLE %I.%I', namespace, target);
        EXECUTE format('DELETE FROM %I.af_captures WHERE capture_id=$1', namespace)
            USING capture_id;
    ELSIF operation='release' AND actor=policy.release_role THEN
        EXECUTE format('DROP TABLE %I.%I', namespace, target);
        EXECUTE format('DELETE FROM %I.af_captures WHERE capture_id=$1', namespace)
            USING capture_id;
    ELSE
        RAISE EXCEPTION 'snapshot operation is not allowed' USING ERRCODE='42501';
    END IF;
END
"""
    return body.replace('SCHEMA_LITERAL', sql.Literal(schema).as_string())


def provision_snapshot_schema(connection, *, schema, owner, registration_role,
                              reader_role, release_role):
    """Provision a new archive using administrator credentials, once per database.

    Supply existing distinct roles. The owner must be NOLOGIN with no members.
    Registration and training never receive ownership or release credentials.
    This operation refuses an existing schema instead of modifying its policy.
    """
    import re

    from psycopg import sql
    from psycopg.pq import TransactionStatus

    names = (schema, owner, registration_role, reader_role, release_role)
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name)
           or len(name.encode()) > 63 for name in names):
        raise ValueError('archive names require simple PostgreSQL identifiers')
    if len(set(names[1:])) != 4:
        raise ValueError('archive owner, registration, reader and release roles must differ')
    if not connection.autocommit or connection.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError('provisioning requires an idle autocommit connection')
    with connection.transaction():
        roles = dict(connection.execute(
            'SELECT rolname, oid FROM pg_roles WHERE rolname=ANY(%s)', (list(names[1:]),),
        ).fetchall())
        if set(roles) != set(names[1:]):
            raise ValueError('create the separate database roles before provisioning the archive')
        owner_flags = connection.execute(
            'SELECT rolcanlogin,rolsuper,rolcreaterole,rolcreatedb,rolreplication,rolbypassrls '
            'FROM pg_roles WHERE oid=%s', (roles[owner],),
        ).fetchone()
        if any(owner_flags) or connection.execute(
            'SELECT 1 FROM pg_auth_members WHERE roleid=%s OR member=%s',
            (roles[owner], roles[owner]),
        ).fetchone():
            raise ValueError('archive owner must be an isolated unprivileged NOLOGIN role')
        connection.execute(sql.SQL('CREATE SCHEMA {} AUTHORIZATION {}').format(
            sql.Identifier(schema), sql.Identifier(owner),
        ))
        connection.execute(sql.SQL('REVOKE ALL ON SCHEMA {} FROM PUBLIC').format(
            sql.Identifier(schema),
        ))
        for role in (registration_role, reader_role, release_role):
            connection.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(
                sql.Identifier(schema), sql.Identifier(role),
            ))
        connection.execute(sql.SQL('SET LOCAL ROLE {}').format(sql.Identifier(owner)))
        connection.execute(sql.SQL('''CREATE TABLE {} (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            registration_role oid NOT NULL, reader_role oid NOT NULL, release_role oid NOT NULL
        )''').format(sql.Identifier(schema, 'af_policy')))
        connection.execute(sql.SQL('INSERT INTO {} VALUES (true,%s,%s,%s)').format(
            sql.Identifier(schema, 'af_policy'),
        ), (roles[registration_role], roles[reader_role], roles[release_role]))
        connection.execute(sql.SQL('''CREATE TABLE {} (
            capture_id text PRIMARY KEY, object_oid oid NOT NULL UNIQUE, seal text NOT NULL,
            creator oid NOT NULL, transaction_id xid8 NOT NULL,
            state text NOT NULL CHECK (state IN ('building','sealed','retained'))
        )''').format(sql.Identifier(schema, 'af_captures')))
        for table in ('af_policy', 'af_captures'):
            connection.execute(sql.SQL('REVOKE ALL ON {} FROM PUBLIC').format(
                sql.Identifier(schema, table),
            ))
            for role in (registration_role, reader_role, release_role):
                connection.execute(sql.SQL('GRANT SELECT ON {} TO {}').format(
                    sql.Identifier(schema, table), sql.Identifier(role),
                ))
        routine = sql.Identifier(schema, 'af_snapshot')
        connection.execute(sql.SQL('''CREATE FUNCTION {}(operation text, capture_id text,
            details jsonb DEFAULT '{{}}'::jsonb) RETURNS void LANGUAGE plpgsql SECURITY DEFINER
            SET search_path=pg_catalog,pg_temp AS {}''').format(
                routine, sql.Literal(_snapshot_operation(schema)),
            ))
        connection.execute(sql.SQL('REVOKE ALL ON FUNCTION {}(text,text,jsonb) FROM PUBLIC').format(
            routine,
        ))
        for role in (registration_role, release_role):
            connection.execute(sql.SQL(
                'GRANT EXECUTE ON FUNCTION {}(text,text,jsonb) TO {}',
            ).format(
                routine, sql.Identifier(role),
            ))
        # SET LOCAL ROLE is restored by the transaction, including on failure.


def _unsafe_snapshot_authority(connection, release_role, *, role_oid=None):
    """Check every membership, including roles reachable by explicit SET ROLE."""
    return connection.execute('''SELECT coalesce(bool_or(
        r.rolsuper OR r.rolcreaterole OR r.rolcreatedb OR r.rolreplication
        OR r.rolbypassrls OR r.oid=d.datdba OR r.rolname=%s
        OR r.rolname IN ('pg_write_all_data','pg_write_server_files',
                        'pg_execute_server_program')), true)
        FROM pg_roles actor
        JOIN pg_roles r ON pg_has_role(actor.oid,r.oid,'MEMBER')
        CROSS JOIN pg_database d
        WHERE actor.oid=coalesce(%s::oid,session_user::regrole::oid)
          AND d.datname=current_database()''', (release_role, role_oid)).fetchone()[0]


class PostgresSnapshots:
    """Typed inputs under a protected owner in a separately provisioned archive.

    Registration streams each capture once, then loses write access before commit.
    Training reads sealed inputs; new windows use the restricted registration account.
    Supply release credentials only to a cleanup process,
    never to registration callbacks or training. Administrators remain trusted.
    """

    backend = "postgresql"

    def __init__(self, writer, reader, *, schema="public", releaser=None):
        import re

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema) or len(schema) > 63:
            raise ValueError("snapshot schema requires a simple PostgreSQL identifier")
        self.writer, self.reader, self.schema = writer, reader, schema
        self.releaser = releaser

    @property
    def namespace(self) -> str:
        return self.schema

    def preflight(self):
        """Refuse unprovisioned archives and privileged application credentials."""
        from psycopg.pq import TransactionStatus

        if self.writer is None:
            raise ValueError('capture requires the configured registration connection')
        for connection in (self.writer, self.reader):
            if (not connection.autocommit
                    or connection.info.transaction_status != TransactionStatus.IDLE):
                raise ValueError('snapshot connections must be idle and use autocommit')
        policy = self._archive_policy()
        database, actor = self.writer.execute(
            'SELECT current_database(),session_user::regrole::oid',
        ).fetchone()
        reader_database = self.reader.execute('SELECT current_database()').fetchone()[0]
        if database != reader_database or actor != policy['registration']:
            raise ValueError('native inputs require the configured restricted registration account')
        self.reader_role = self.reader.execute('SELECT session_user').fetchone()[0]

    def check_source(self, connection):
        """Keep PostgreSQL source credentials outside the archive administration boundary.

        Use an ordinary source account even on a separate server. Hostnames and
        connection URLs cannot prove that two connections reach different clusters.
        Membership is checked even when a role requires an explicit SET ROLE.
        """
        from psycopg.pq import TransactionStatus

        if (not connection.autocommit
                or connection.info.transaction_status != TransactionStatus.IDLE):
            raise ValueError('source capture needs an idle autocommit connection')
        policy = self._archive_policy()
        release = self.reader.execute('SELECT rolname FROM pg_roles WHERE oid=%s',
                                      (policy['release'],)).fetchone()[0]
        if _unsafe_snapshot_authority(connection, release):
            raise ValueError('PostgreSQL table inputs require an ordinary source account '
                             'without administrator, database-owner or archive release rights')

    def _archive_policy(self):
        """Check actual ownership and effective grants at each external read boundary."""
        from psycopg import sql

        reader = self.reader
        row = reader.execute('''SELECT n.nspowner, r.rolcanlogin, r.rolsuper,
            r.rolcreaterole,r.rolcreatedb,r.rolreplication,r.rolbypassrls
            FROM pg_namespace n JOIN pg_roles r ON r.oid=n.nspowner WHERE n.nspname=%s''',
                             (self.schema,)).fetchone()
        if row is None or any(row[1:]):
            raise ValueError('snapshot schema requires a protected NOLOGIN owner')
        owner = row[0]
        if reader.execute('SELECT 1 FROM pg_auth_members WHERE roleid=%s OR member=%s',
                          (owner, owner)).fetchone():
            raise ValueError('snapshot owner must not have role memberships or members')
        if reader.execute("""SELECT 1 FROM pg_roles WHERE NOT rolsuper
            AND rolname !~ '^pg_' AND pg_has_role(oid,'pg_write_all_data','MEMBER')""").fetchone():
            raise ValueError('table-only archives cannot admit ordinary write-all-data roles')
        for table in ('af_policy', 'af_captures'):
            self._protected_table(self.schema + '.' + table, owner)
        rows = reader.execute(sql.SQL('SELECT registration_role,reader_role,release_role FROM {}')
                              .format(sql.Identifier(self.schema, 'af_policy'))).fetchall()
        if len(rows) != 1 or len(set(rows[0]) | {owner}) != 4:
            raise ValueError('snapshot role policy is invalid')
        registration, replay, release = rows[0]
        release_name = reader.execute('SELECT rolname FROM pg_roles WHERE oid=%s',
                                      (release,)).fetchone()
        if release_name is None:
            raise ValueError('snapshot role policy is invalid')
        actor = reader.execute('SELECT session_user::regrole::oid').fetchone()[0]
        if actor != replay:
            raise ValueError('native inputs require the configured restricted reader')
        for role in (registration, replay):
            if _unsafe_snapshot_authority(reader, release_name[0], role_oid=role):
                raise ValueError('application roles must not own the database or hold '
                                 'administrator or release rights')
        if reader.execute("SELECT pg_has_role(%s,%s,'MEMBER')",
                          (replay, registration)).fetchone()[0]:
            raise ValueError('training must not assume the registration role')
        schema_grants = reader.execute('''SELECT 1 FROM pg_namespace n,
            LATERAL aclexplode(coalesce(n.nspacl,acldefault('n',n.nspowner))) a
            WHERE n.nspname=%s AND a.grantee<>%s AND a.privilege_type='CREATE' ''',
                                      (self.schema, owner)).fetchone()
        if schema_grants:
            raise ValueError('only the protected owner may create archive objects')
        routine = reader.execute('''SELECT p.proowner,p.prosecdef,p.proconfig,p.prosrc,
            l.lanname,p.prorettype,p.proretset
            FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            JOIN pg_language l ON l.oid=p.prolang
            WHERE n.nspname=%s AND p.proname='af_snapshot'
            AND p.proargtypes='25 25 3802'::oidvector''',
                                 (self.schema,)).fetchone()
        expected = (owner, True, ['search_path=pg_catalog, pg_temp'],
                    _snapshot_operation(self.schema), 'plpgsql', 2278, False)
        if routine != expected:
            raise ValueError('snapshot sealing operation differs from the provisioned contract')
        grants = reader.execute('''SELECT a.grantee,a.privilege_type,a.is_grantable
            FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace,
            LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
            WHERE n.nspname=%s AND p.proname='af_snapshot' ''', (self.schema,)).fetchall()
        if any(grantee not in (owner, registration, release)
               or (grantee != owner and grantable) for grantee, _, grantable in grants):
            raise ValueError('snapshot sealing operation has unexpected execute grants')
        return {'owner': owner, 'registration': registration, 'reader': replay, 'release': release}

    def _protected_table(self, name, owner):
        row = self.reader.execute('''SELECT c.relowner,c.relpersistence,c.relkind,
            c.relrowsecurity,c.relforcerowsecurity,c.relhasrules,
            EXISTS(SELECT 1 FROM pg_trigger t WHERE t.tgrelid=c.oid),
            EXISTS(SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
                   WHERE a.grantee<>c.relowner AND
                   (a.privilege_type<>'SELECT' OR a.is_grantable)),
            EXISTS(SELECT 1 FROM pg_attribute a,LATERAL aclexplode(a.attacl) x
                   WHERE a.attrelid=c.oid AND x.grantee<>c.relowner AND
                   (x.privilege_type<>'SELECT' OR x.is_grantable))
            FROM pg_class c WHERE c.oid=to_regclass(%s)''', (name,)).fetchone()
        if row is None:
            raise FileNotFoundError('retained input table or archive policy is missing')
        if row[:3] != (owner, 'p', 'r') or any(row[3:]):
            raise ValueError('retained tables require protected ownership without mutation grants')

    def _operation(self, operation, record, *, connection=None, details=None):
        from psycopg import sql
        from psycopg.types.json import Jsonb

        connection = connection or self.writer
        if connection is None:
            raise ValueError('this operation requires a separately configured connection')
        self._name(record)
        connection.execute(sql.SQL('SELECT {}(%s,%s,%s)').format(
            sql.Identifier(self.schema, 'af_snapshot'),
        ), (operation, record['capture_id'], Jsonb(
            details if details is not None else record.get('incarnation', {}),
        )))

    def _name(self, record):
        import re

        from psycopg import sql

        capture_id = record["capture_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
            raise ValueError("invalid native capture ID")
        name = "af_input_" + capture_id
        if record["object"] != self.schema + "." + name:
            raise ValueError("snapshot belongs to a different destination")
        return sql.Identifier(self.schema, name)

    @staticmethod
    def _columns(schema):
        import re

        import pyarrow as pa

        columns = []
        for field in schema:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field.name) or len(field.name) > 63:
                raise ValueError("snapshot columns require simple PostgreSQL identifiers")
            dtype = field.type
            if pa.types.is_signed_integer(dtype):
                native = "BIGINT"
            elif pa.types.is_boolean(dtype):
                native = "BOOLEAN"
            elif pa.types.is_float32(dtype):
                native = "REAL"
            elif pa.types.is_float64(dtype):
                native = "DOUBLE PRECISION"
            elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
                native = "TEXT"
            elif pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
                native = "BYTEA"
            elif pa.types.is_date32(dtype):
                native = "DATE"
            elif pa.types.is_timestamp(dtype) and dtype.unit in {"s", "ms", "us"}:
                native = "TIMESTAMP" if dtype.tz is None else "TIMESTAMPTZ"
            elif pa.types.is_decimal(dtype):
                native = f"NUMERIC({dtype.precision},{dtype.scale})"
            else:
                raise ValueError(f"PostgreSQL snapshot transport cannot preserve {dtype}")
            columns.append({"name": field.name, "type": native, "nullable": field.nullable})
        return columns

    def _incarnation(self, connection, record):
        with connection.cursor() as cursor:
            cursor.execute("""SELECT c.oid, c.relowner, c.relpersistence, c.relkind,
                obj_description(c.oid, 'pg_class'),
                ARRAY(SELECT a.attname || ':' || format_type(a.atttypid,a.atttypmod)
                             || ':' || a.attnotnull::text
                      FROM pg_attribute a WHERE a.attrelid=c.oid AND a.attnum>0
                      AND NOT a.attisdropped ORDER BY a.attnum)
                FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname=%s AND c.relname=%s""",
                           (self.schema, "af_input_" + record["capture_id"]))
            row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError("retained input table is missing")
        oid, owner, persistence, kind, seal, columns = row
        if persistence != "p" or kind != "r" or not seal:
            raise ValueError("input is not the sealed persistent snapshot")
        return {"oid": oid, "owner": owner, "seal": seal, "columns": columns}

    def capture(self, path, *, capture_id, checkpoint=None):
        import pyarrow.parquet as pq
        from psycopg import sql

        from autoforecast.snapshots import ColumnLayout, arrow_schema

        self.preflight()
        record = {"object": self.schema + ".af_input_" + capture_id, "capture_id": capture_id}
        name = self._name(record)
        files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
        if not files:
            raise ValueError("native input requires typed Parquet files")
        schema, rows = None, 0
        for file in files:
            with pq.ParquetFile(file) as source:
                if schema is not None and not source.schema_arrow.equals(schema,
                                                                        check_metadata=False):
                    raise ValueError("native input file schemas differ")
                schema = source.schema_arrow.remove_metadata()
                rows += source.metadata.num_rows
        assert schema is not None
        layout = ColumnLayout(schema)
        columns = self._columns(layout.schema)
        with self.writer.transaction(), self.writer.cursor() as cursor:
            self._operation("begin", record, details=columns)
            with cursor.copy(sql.SQL("COPY {} FROM STDIN").format(name)) as copy:
                for file in files:
                    with pq.ParquetFile(file) as source:
                        for batch in source.iter_batches(batch_size=8192):
                            for row in batch.to_pylist():
                                copy.write_row(layout.encode(row))
            if cursor.rowcount != rows:
                raise ValueError("native input row count changed during COPY")
            self._operation("seal", record)
            record["incarnation"] = self._incarnation(self.writer, record)
            record.update(arrow_schema=arrow_schema(schema), rows=rows)
            if checkpoint is not None:
                checkpoint(record)
        try:
            self.inspect(record)
        except BaseException:
            from autoforecast.snapshots import _cleanup_on_exit

            with _cleanup_on_exit(lambda: self.drop_unretained(record),
                                  label="Capture cleanup needs recovery"):
                raise
        return record

    def inspect(self, record, *, retained=False):
        self._name(record)
        if self._incarnation(self.reader, record) != record["incarnation"]:
            raise ValueError("native input incarnation or schema changed")
        policy = self._archive_policy()
        self._protected_table(record["object"], policy["owner"])
        from psycopg import sql

        saved = self.reader.execute(sql.SQL(
            "SELECT object_oid,seal,state FROM {} WHERE capture_id=%s",
        ).format(sql.Identifier(self.schema, "af_captures")), (record["capture_id"],)).fetchone()
        if (saved is None or saved[:2] != (record["incarnation"]["oid"],
                                           record["incarnation"]["seal"])
                or saved[2] not in ({"retained"} if retained else {"sealed", "retained"})):
            raise ValueError("capture identity or retained state differs")
        if not self.reader.execute("SELECT has_table_privilege(current_user,%s,'SELECT')",
                                   (record["object"],)).fetchone()[0]:
            raise ValueError("input reader needs SELECT on the retained capture")

    def retain(self, record):
        """Permanently remove the registration account's ability to discard a capture."""
        self.inspect(record)
        self._operation("retain", record)
        self.inspect(record, retained=True)

    def drop_unretained(self, record):
        """Clean up unpublished data without authority to delete retained experiments."""
        self._operation("drop_unretained", record)

    def read(self, record, destination):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from psycopg import sql

        from autoforecast.snapshots import ColumnLayout, restore_schema

        self.inspect(record)
        schema = restore_schema(record["arrow_schema"])
        layout = ColumnLayout(schema)
        destination.mkdir(parents=True)
        with self.reader.transaction():
            self.reader.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            with (self.reader.cursor(name="af_read_" + record["capture_id"]) as cursor,
                  pq.ParquetWriter(destination / "part.parquet", schema) as writer):
                cursor.execute(sql.SQL("SELECT {} FROM {}").format(
                    sql.SQL(", ").join(map(sql.Identifier, layout.schema.names)),
                    self._name(record),
                ))
                while rows := cursor.fetchmany(8192):
                    writer.write_table(pa.Table.from_pylist([
                        layout.decode(row) for row in rows
                    ], schema=schema))
        self.inspect(record)

    def release_preflight(self):
        """Validate cleanup credentials without loading them in a registration process."""
        from psycopg.pq import TransactionStatus

        if self.releaser is None:
            raise ValueError("published capture release requires separate release credentials")
        if (not self.releaser.autocommit
                or self.releaser.info.transaction_status != TransactionStatus.IDLE):
            raise ValueError("release connection must be idle and use autocommit")
        policy = self._archive_policy()
        database, role = self.releaser.execute(
            "SELECT current_database(),session_user::regrole::oid",
        ).fetchone()
        if (database != self.reader.execute("SELECT current_database()").fetchone()[0]
                or role != policy["release"]):
            raise ValueError("release requires the configured separate release account")

    def release(self, record):
        """Delete only this incarnation using a separately configured release account."""
        self.release_preflight()
        self._operation("release", record, connection=self.releaser)


def _source_string_grammar(connection):
    """Refuse session settings that change PostgreSQL string token boundaries."""
    if connection.execute("SHOW standard_conforming_strings").fetchone()[0] != "on":
        raise ValueError("PostgreSQL source capture requires standard_conforming_strings=on")


def read_sources(connection, *, queries, tables, output_root):
    """Read related SELECTs in one read-only repeatable-read transaction.

    Relation locks cover all declared tables before the first data snapshot. This
    also prevents TRUNCATE or name replacement between the related SELECTs.
    Source sessions must keep standard_conforming_strings=on throughout capture.
    """
    import pathlib
    import re
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq
    from psycopg import sql
    from psycopg.pq import TransactionStatus

    from autoforecast.snapshots import bind_source_queries

    if not connection.autocommit or connection.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError("source capture needs an idle autocommit connection")
    _source_string_grammar(connection)
    for name in tables.values():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("PostgreSQL source tables require schema.table")
    identifiers = {alias: sql.Identifier(*name.split(".")) for alias, name in tables.items()}
    bound = bind_source_queries(queries, tables, dialect="postgres", sources={
        alias: name.as_string(connection) for alias, name in identifiers.items()
    })
    result = {}
    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        connection.execute("SET LOCAL TIME ZONE 'UTC'")
        connection.execute(sql.SQL("LOCK TABLE {} IN ACCESS SHARE MODE").format(
            sql.SQL(", ").join(identifiers.values()),
        ))
        snapshot = connection.execute("SELECT pg_current_snapshot()::text").fetchone()[0]
        provenance = {alias: {"table": name, "oid": connection.execute(
            "SELECT %s::regclass::oid", (name,),
        ).fetchone()[0]} for alias, name in tables.items()}
        for index, (name, query) in enumerate(bound.items()):
            _source_string_grammar(connection)
            with connection.cursor(name="af_source_" + uuid.uuid4().hex) as cursor:
                cursor.execute(query)
                schema = _source_arrow_schema(connection, cursor.description)
                destination = pathlib.Path(output_root) / ("query-" + str(index))
                destination.mkdir(parents=True)
                count = 0
                with pq.ParquetWriter(destination / "part.parquet", schema) as writer:
                    while rows := cursor.fetchmany(8192):
                        table = pa.Table.from_pylist([
                            dict(zip(schema.names, row, strict=True)) for row in rows
                        ], schema=schema)
                        writer.write_table(table)
                        count += table.num_rows
                result[name] = (destination, {"kind": "postgresql", "tables": provenance,
                                             "snapshot": snapshot, "sql": queries[name],
                                             "rows": count})
        # A SELECT can call set_config(); refusing before commit restores caller settings.
        _source_string_grammar(connection)
    return result


def _source_arrow_schema(connection, columns):
    """Keep declared SQL types, including an empty result; never infer from sample rows."""
    import pyarrow as pa

    scalar = {"bool": pa.bool_(), "int2": pa.int16(), "int4": pa.int32(), "int8": pa.int64(),
              "float4": pa.float32(), "float8": pa.float64(), "text": pa.string(),
              "varchar": pa.string(), "bpchar": pa.string(), "bytea": pa.binary(),
              "date": pa.date32(), "timestamp": pa.timestamp("us"),
              "timestamptz": pa.timestamp("us", tz="UTC")}
    fields = []
    for column in columns:
        typename = connection.execute("SELECT typname FROM pg_type WHERE oid=%s",
                                      (column.type_code,)).fetchone()[0]
        dtype = scalar.get(typename)
        if typename == "numeric":
            precision, scale = column.precision, column.scale
            if precision is None or scale is None or not 0 <= scale <= precision <= 38:
                raise ValueError("cast numeric source expressions to an explicit precision <= 38")
            dtype = pa.decimal128(precision, scale)
        if dtype is None:
            raise ValueError(f"source transport cannot preserve PostgreSQL type {typename}")
        fields.append(pa.field(column.name, dtype))
    if len({field.name for field in fields}) != len(fields):
        raise ValueError("source SELECT must give every output column a distinct name")
    return pa.schema(fields)
