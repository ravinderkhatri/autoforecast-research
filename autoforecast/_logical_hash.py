"""Versioned typed-row identity, shared by Spark and offline Parquet readers.

v1 encodes null as ``n`` and each value as ``v<ASCII length>:<payload>``.
Strings use UTF-8 hex, binary uses hex, integers/decimals use base ten, dates
use epoch days and timestamps use epoch microseconds (UTC or naive separately).
Floats use hexadecimal double notation, preserving signed zero; all NaNs share
one representation. Arrays retain order; structs sort fields by name and maps
sort encoded entries. Schema types participate, nullability and metadata do not.
Each row is SHA-256 hashed. Its first byte selects one of 256 logical buckets.
Each bucket hashes its sorted 32-byte digests, retaining duplicates. The final
domain-separated JSON binds the schema, format, count and all 256 summaries.
Duplicate-heavy data can concentrate sorting work in one logical bucket; the
format never changes its bucket rule to conceal that skew.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import pathlib
import tempfile
from contextlib import nullcontext

from autoforecast._execution import note, operation, progress

ALGORITHM = "autoforecast.bucket-sha256.v1"
ENCODING = "typed-ascii.v1"
BUCKETS = 256
_BUCKET_DOMAIN = b"autoforecast.bucket-sha256.v1/bucket\x00"
# Evidence tables are often small. Decode them on the driver instead of scheduling
# Spark jobs, which cost seconds each regardless of size. The row ceiling bounds time:
# the driver decodes about 100,000 six-column rows per second (8 November local
# measurement). _LOCAL_MAX_BYTES bounds the file bytes and the decoded size of each batch.
_LOCAL_MAX_ROWS = 250_000
_LOCAL_MAX_BYTES = 8 * 1024 * 1024
_OFFLINE_BATCH_ROWS = 8192


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _type(dtype):
    import pyarrow as pa

    if pa.types.is_struct(dtype):
        return {"struct": _schema(dtype)}
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return {"array": _type(dtype.value_type)}
    if pa.types.is_map(dtype):
        return {"map": [_type(dtype.key_type), _type(dtype.item_type)]}
    if pa.types.is_decimal128(dtype) and 0 <= dtype.scale <= dtype.precision <= 38:
        return {"decimal": [dtype.precision, dtype.scale]}
    if pa.types.is_timestamp(dtype) and dtype.unit == "us":
        return "timestamp_utc" if dtype.tz else "timestamp_ntz"
    names = {
        pa.bool_(): "bool",
        pa.int8(): "i8",
        pa.int16(): "i16",
        pa.int32(): "i32",
        pa.int64(): "i64",
        pa.float32(): "f32",
        pa.float64(): "f64",
        pa.string(): "string",
        pa.binary(): "binary",
        pa.date32(): "date",
    }
    if dtype in names:
        return names[dtype]
    raise ValueError(f"unsupported {ALGORITHM} type: {dtype}")


def _schema(schema):
    names = [field.name for field in schema]
    if len(names) != len(set(names)):
        raise ValueError("logical hashing requires unique column names")
    return [[field.name, _type(field.type)] for field in sorted(schema, key=lambda f: f.name)]


def _parquet_schema(footer):
    """Validate physical types before accepting any embedded Spark declaration."""
    schema = footer.schema_arrow
    metadata = schema.metadata or {}
    legacy_flags = {b"org.apache.spark.legacyDateTime", b"org.apache.spark.legacyINT96"}
    version = metadata.get(b"org.apache.spark.version")
    version = tuple(int(part) for part in version.split(b".")[:2]) if version else None
    old_temporals = version is not None and any(
        (version < (3, 0) and footer.schema.column(i).logical_type.type in {"DATE", "TIMESTAMP"})
        or (version < (3, 1) and footer.schema.column(i).physical_type == "INT96")
        for i in range(len(footer.schema))
    )
    if legacy_flags.intersection(metadata) or old_temporals:
        raise ValueError(
            "logical hashing does not support legacy-rebased Parquet; "
            "create a corrected-calendar Parquet capture before hashing"
        )
    # Walk native Parquet leaves alongside Arrow containers. The embedded Spark
    # schema may disambiguate representation, but may not hide physical types.
    import pyarrow as pa

    leaves = iter(footer.schema.column(i) for i in range(len(footer.schema)))

    def physical_type(dtype):
        if pa.types.is_struct(dtype):
            return pa.struct([field.with_type(physical_type(field.type)) for field in dtype])
        if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
            constructor = pa.list_ if pa.types.is_list(dtype) else pa.large_list
            return constructor(dtype.value_field.with_type(physical_type(dtype.value_type)))
        if pa.types.is_map(dtype):
            return pa.map_(
                dtype.key_field.with_type(physical_type(dtype.key_type)),
                dtype.item_field.with_type(physical_type(dtype.item_type)),
                keys_sorted=dtype.keys_sorted,
            )
        leaf = next(leaves)
        if leaf.physical_type == "INT96":
            return pa.timestamp("us", tz="UTC")
        if pa.types.is_timestamp(dtype):
            if dtype.unit not in {"ms", "us"}:
                raise ValueError(f"unsupported {ALGORITHM} physical timestamp: {dtype}")
            return pa.timestamp("us", tz=dtype.tz)
        return dtype

    physical = _schema(pa.schema([
        field.with_type(physical_type(field.type)) for field in footer.schema.to_arrow_schema()
    ]))
    if next(leaves, None) is not None:
        raise ValueError("Parquet logical and physical schemas disagree")
    declared = metadata.get(b"org.apache.spark.sql.parquet.row.metadata")
    if declared is not None:
        # Compare inert schema data. Spark's JSON parser can import and construct
        # user-defined Python types, so file metadata must never reach it.
        def identity(value):
            if isinstance(value, list):
                return [identity(item) for item in value]
            if isinstance(value, dict):
                incidental = {"metadata", "nullable", "containsNull", "valueContainsNull"}
                result = {key: identity(item) for key, item in value.items()
                          if key not in incidental}
                if "fields" in result:
                    result["fields"] = sorted(result["fields"], key=lambda field: field["name"])
                return result
            return value

        try:
            actual = identity(json.loads(declared))
            expected = identity(_spark_type({"struct": physical}).jsonValue())
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError("malformed Spark-declared Parquet schema") from error
        if actual != expected:
            raise ValueError("Spark-declared schema differs from physical Parquet schema")
    return physical


def _spark_type(spec):
    from pyspark.sql import types as T

    if isinstance(spec, dict):
        if "struct" in spec:
            return T.StructType([T.StructField(n, _spark_type(t)) for n, t in spec["struct"]])
        if "array" in spec:
            return T.ArrayType(_spark_type(spec["array"]))
        if "map" in spec:
            return T.MapType(*[_spark_type(t) for t in spec["map"]])
        if "decimal" in spec:
            parameters = spec["decimal"]
            if (type(parameters) is not list or len(parameters) != 2
                    or any(type(value) is not int for value in parameters)
                    or not 1 <= parameters[0] <= 38
                    or not 0 <= parameters[1] <= parameters[0]):
                raise ValueError("decimal schema requires integer precision and scale")
            return T.DecimalType(*parameters)
    types = {
        "bool": T.BooleanType,
        "i8": T.ByteType,
        "i16": T.ShortType,
        "i32": T.IntegerType,
        "i64": T.LongType,
        "f32": T.FloatType,
        "f64": T.DoubleType,
        "string": T.StringType,
        "binary": T.BinaryType,
        "date": T.DateType,
        "timestamp_utc": T.TimestampType,
        "timestamp_ntz": T.TimestampNTZType,
    }
    if not isinstance(spec, str) or spec not in types:
        raise ValueError("unsupported logical schema type")
    return types[spec]()


def _wrap(payload):
    return f"v{len(payload)}:{payload}"


def _value(value, spec):
    if value is None:
        return "n"
    if isinstance(spec, dict):
        if "struct" in spec:
            payload = "".join(_value(value[n], t) for n, t in spec["struct"])
        elif "array" in spec:
            payload = "".join(_value(v, spec["array"]) for v in value)
        elif "map" in spec:
            entries = list(value.items()) if isinstance(value, dict) else value
            keys = [_value(k, spec["map"][0]) for k, _ in entries]
            if "n" in keys or len(keys) != len(set(keys)):
                raise ValueError("logical hashing requires non-null unique map keys")
            payload = "".join(
                sorted(
                    k + _value(v, spec["map"][1]) for k, (_, v) in zip(keys, entries, strict=False)
                )
            )
        else:
            payload = format(value, f".{spec['decimal'][1]}f")
            if value == 0:
                payload = payload.lstrip("-")
    elif spec == "string":
        payload = value.encode("utf-8").hex()
    elif spec == "binary":
        payload = bytes(value).hex()
    elif spec == "bool":
        payload = "1" if value else "0"
    elif spec == "date":
        payload = (str(value) if isinstance(value, int)
                   else str((value - dt.date(1970, 1, 1)).days))
    elif spec.startswith("timestamp"):
        if isinstance(value, int):
            return _wrap(str(value))
        epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC if spec == "timestamp_utc" else None)
        if getattr(value, "nanosecond", 0):
            raise ValueError("logical timestamp encoding requires microsecond precision")
        if spec == "timestamp_utc" and value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        delta = value - epoch
        payload = str((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds)
    elif spec in {"f32", "f64"}:
        number = float(value)
        if math.isnan(number):
            payload = "NaN"
        elif math.isinf(number):
            payload = "Infinity" if number > 0 else "-Infinity"
        else:
            mantissa, exponent = number.hex().split("p")
            payload = mantissa.rstrip("0").rstrip(".")
            if "." not in payload:
                payload += ".0"
            payload += "p" + str(int(exponent))
    else:
        payload = str(value)
    return _wrap(payload)


def _expression(column, spec):
    from pyspark.sql import functions as F

    if isinstance(spec, dict):
        if "struct" in spec:
            parts = [_expression(column.getField(n), t) for n, t in spec["struct"]]
            payload = F.concat(*parts) if parts else F.lit("")
        elif "array" in spec:
            payload = F.concat_ws("", F.transform(column, lambda v: _expression(v, spec["array"])))
        elif "map" in spec:
            entries = F.map_entries(column)
            keys = F.transform(entries, lambda e: _expression(e["key"], spec["map"][0]))
            encoded = F.transform(
                entries,
                lambda e: F.concat(
                    _expression(e["key"], spec["map"][0]), _expression(e["value"], spec["map"][1])
                ),
            )
            payload = F.when(
                F.size(F.array_distinct(keys)) != F.size(keys),
                F.raise_error("logical hashing requires unique map keys"),
            ).otherwise(F.concat_ws("", F.sort_array(encoded)))
        else:
            # Fixed decimal places avoid ANSI-dependent scientific notation.
            payload = F.regexp_replace(F.format_number(column, spec["decimal"][1]), ",", "")
    elif spec == "string":
        payload = F.lower(F.hex(F.encode(column, "UTF-8")))
    elif spec == "binary":
        payload = F.lower(F.hex(column))
    elif spec == "bool":
        payload = F.when(column, "1").otherwise("0")
    elif spec == "date":
        payload = F.datediff(column, F.lit("1970-01-01")).cast("string")
    elif spec == "timestamp_utc":
        payload = F.unix_micros(column).cast("string")
    elif spec == "timestamp_ntz":
        epoch = F.lit("1970-01-01 00:00:00").cast("timestamp_ntz")
        payload = F.timestamp_diff("MICROSECOND", epoch, column).cast("string")
    elif spec in {"f32", "f64"}:
        payload = F.format_string("%a", column.cast("double"))
    else:
        payload = column.cast("string")
    return F.when(column.isNull(), F.lit("n")).otherwise(
        F.concat(F.lit("v"), F.length(payload).cast("string"), F.lit(":"), payload)
    )


def _bucket_summaries(batches):
    """Stream sorted Arrow batches; memory is bounded by the batch and 256 summaries."""
    import pyarrow as pa

    current, count, hashed = None, 0, hashlib.sha256(_BUCKET_DOMAIN)
    output = []
    for batch in batches:
        for key, value in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=False
        ):
            if key != current:
                if current is not None:
                    output.append((current, count, hashed.hexdigest()))
                current, count, hashed = key, 0, hashlib.sha256(_BUCKET_DOMAIN)
            if len(value) != 32 or value[0] != key:
                raise ValueError("invalid logical row digest")
            hashed.update(value)
            count += 1
    if current is not None:
        output.append((current, count, hashed.hexdigest()))
    if output:
        yield pa.RecordBatch.from_arrays(
            [
                pa.array([r[0] for r in output], type=pa.int32()),
                pa.array([r[1] for r in output], type=pa.int64()),
                pa.array([r[2] for r in output], type=pa.string()),
            ],
            names=["bucket", "rows", "sha256"],
        )


def _finish(schema, summaries, rows):
    seen = {}
    for item in summaries:
        key, count, sha = item["bucket"], item["rows"], item["sha256"]
        if (
            type(key) is not int
            or not 0 <= key < BUCKETS
            or key in seen
            or type(count) is not int
            or count < 1
            or type(sha) is not str
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("invalid or repeated logical bucket summary")
        seen[key] = [key, count, sha]
    if sum(item[1] for item in seen.values()) != rows:
        raise ValueError("logical bucket counts differ from Parquet row count")
    empty = hashlib.sha256(_BUCKET_DOMAIN).hexdigest()
    buckets = [seen.get(key, [key, 0, empty]) for key in range(BUCKETS)]
    metadata = {
        "algorithm": ALGORITHM,
        "encoding": ENCODING,
        "bucket_count": BUCKETS,
        "schema": schema,
        "rows": rows,
    }
    sha = hashlib.sha256(
        _json({"domain": "dataset", **metadata, "buckets": buckets}).encode()
    ).hexdigest()
    return {**metadata, "sha256": sha}


def validate_metadata(record):
    """Reject unknown algorithms and malformed persisted encodings before reuse."""
    if (
        type(record) is not dict
        or set(record) != {"algorithm", "encoding", "bucket_count", "schema", "rows", "sha256"}
        or record["algorithm"] != ALGORITHM
        or record["encoding"] != ENCODING
        or type(record["bucket_count"]) is not int
        or record["bucket_count"] != BUCKETS
        or type(record["rows"]) is not int
        or record["rows"] < 0
        or type(record["sha256"]) is not str
        or len(record["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in record["sha256"])
    ):
        raise ValueError("unsupported or malformed logical hash metadata")
    try:
        from pyspark.sql.pandas.types import to_arrow_schema
        from pyspark.sql.types import StructType

        declared = _spark_type({"struct": record["schema"]})
        assert isinstance(declared, StructType)
        normalized = _schema(to_arrow_schema(declared, timezone="UTC"))
        if normalized != record["schema"]:
            raise ValueError("noncanonical schema")
    except (KeyError, TypeError, ValueError, OverflowError,
            AssertionError, AttributeError, IndexError) as error:
        raise ValueError("invalid logical hash schema") from error



def _temporal_type(dtype, *, epochs):
    """Keep Arrow temporal values out of Python's narrower calendar range."""
    import pyarrow as pa

    if pa.types.is_date32(dtype):
        return pa.int32() if epochs else dtype
    if pa.types.is_timestamp(dtype):
        return pa.int64() if epochs else pa.timestamp("us", tz=dtype.tz)
    if pa.types.is_struct(dtype):
        return pa.struct([
            field.with_type(_temporal_type(field.type, epochs=epochs)) for field in dtype
        ])
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        constructor = pa.list_ if pa.types.is_list(dtype) else pa.large_list
        return constructor(dtype.value_field.with_type(
            _temporal_type(dtype.value_type, epochs=epochs),
        ))
    if pa.types.is_map(dtype):
        return pa.map_(
            dtype.key_field.with_type(_temporal_type(dtype.key_type, epochs=epochs)),
            dtype.item_field.with_type(_temporal_type(dtype.item_type, epochs=epochs)),
            keys_sorted=dtype.keys_sorted,
        )
    return dtype

def _offline(path, schema, rows, batch_rows=_OFFLINE_BATCH_ROWS):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.benchmark import _merge_hash_runs, _write_hash_run
    from provenance.lineage import parquet_files

    def batches():
        for file in parquet_files(path):
            with pq.ParquetFile(file, coerce_int96_timestamp_unit="us") as reader:
                yield from reader.iter_batches(
                    columns=[n for n, _ in schema],
                    batch_size=batch_rows,
                    use_threads=False,
                )

    with tempfile.TemporaryDirectory(prefix="af-logical-hash-") as temp:
        root = pathlib.Path(temp)
        chunk, runs, processed = [], [], 0
        for batch in batches():
            # Normalize timestamp units before replacing temporal leaves with
            # integers. Arrow casts recurse through lists, structs and maps.
            micros = pa.schema([
                field.with_type(_temporal_type(field.type, epochs=False)) for field in batch.schema
            ])
            epochs = pa.schema([
                field.with_type(_temporal_type(field.type, epochs=True)) for field in batch.schema
            ])
            for row in batch.cast(micros).cast(epochs).to_pylist():
                encoded = "".join(_value(row[n], t) for n, t in schema)
                chunk.append(hashlib.sha256(encoded.encode("ascii")).digest())
                if len(chunk) == 65536:
                    target = root / str(len(runs))
                    _write_hash_run(target, chunk)
                    runs.append(target)
                    chunk.clear()
            processed += batch.num_rows
            progress(completed=processed, total=rows, unit="rows", phase="hash rows")
        if chunk:
            target = root / str(len(runs))
            _write_hash_run(target, chunk)
            runs.append(target)
        level = 0
        while len(runs) > 1:
            merged = []
            for start in range(0, len(runs), 64):
                target = root / f"merge-{level}-{start}"
                _merge_hash_runs(runs[start : start + 64], target)
                merged.append(target)
                for consumed in runs[start : start + 64]:
                    consumed.unlink()
            runs = merged
            level += 1
        summaries = []
        if runs:
            with runs[0].open("rb") as stream:
                key, count, hashed = None, 0, hashlib.sha256(_BUCKET_DOMAIN)
                for value in iter(lambda: stream.read(32), b""):
                    if value[0] != key:
                        if key is not None:
                            summaries.append(
                                {"bucket": key, "rows": count, "sha256": hashed.hexdigest()}
                            )
                        key, count, hashed = value[0], 0, hashlib.sha256(_BUCKET_DOMAIN)
                    hashed.update(value)
                    count += 1
                summaries.append({"bucket": key, "rows": count, "sha256": hashed.hexdigest()})
        return _finish(schema, summaries, rows)


def fingerprint(path, *, spark=None):
    """Return the typed identity of a fresh, operation-owned Parquet capture."""
    return _capture(path, spark=spark)[0]


def _capture(path, *, spark=None):
    """Hash an operation-owned immutable capture, never a moving source query.

    Spark must be able to read the captured path on its workers. A native volume
    download supplies metadata and file authentication, not a driver-local Spark
    input. Callers retain exclusive ownership of the captured path throughout.
    Registration, freeze, preflight and Delta each create fresh captures first.
    """
    import pyarrow.parquet as pq
    from mlops import _volume_files

    from provenance.lineage import directory_digest, parquet_files

    path = pathlib.Path(path)
    with (
        _volume_files.local_snapshot(path, parquet_only=True)
        if _volume_files.is_volume(path) else nullcontext(path)
    ) as local:
        files = parquet_files(local)
        if not files:
            raise ValueError("logical hashing requires a Parquet schema")
        schema, rows, foreign_int96 = None, 0, False
        # Strict upper bound on one row's decoded bytes. A decoded value is never larger
        # than its column chunk, dictionary included; Arrow adds at most an 8-byte offset
        # and a validity bit for each nesting level. One row of a repeated column can hold
        # every value of the chunk beyond one per other row.
        row_bytes = 0
        for file in files:
            if (
                any("=" in part for part in file.relative_to(local).parts[:-1])
                if local.is_dir()
                else False
            ):
                raise ValueError("logical hashing requires a materialized unpartitioned snapshot")
            with pq.ParquetFile(file) as footer:
                current = _parquet_schema(footer)
                if schema is not None and current != schema:
                    raise ValueError("Parquet files have conflicting logical schemas")
                schema = current
                rows += footer.metadata.num_rows
                for i in range(footer.metadata.num_row_groups):
                    group = footer.metadata.row_group(i)
                    row_bytes = max(row_bytes, sum(
                        (group.column(j).total_uncompressed_size
                         + 8 * (footer.schema.column(j).max_definition_level + 1))
                        * max(1, group.column(j).num_values - group.num_rows + 1)
                        for j in range(group.num_columns)
                    ))
                foreign_int96 |= (
                    not (footer.metadata.created_by or "").startswith("parquet-mr")
                    and any(footer.schema.column(i).physical_type == "INT96"
                            for i in range(len(footer.schema)))
                )
        assert schema is not None
        fallback = foreign_int96 and spark is not None and (
            spark.conf.get("spark.sql.parquet.int96TimestampConversion", "false").lower() == "true"
        )
        # Dictionary and repetition encoding can expand far beyond file size, so size
        # each decoded batch by the strict per-row bound instead of the file size.
        batch_rows = min(_OFFLINE_BATCH_ROWS, _LOCAL_MAX_BYTES // max(1, row_bytes))
        small = (rows <= _LOCAL_MAX_ROWS and batch_rows >= 1
                 and sum(file.stat().st_size for file in files) <= _LOCAL_MAX_BYTES)
        if fallback or small:
            # Spark applies this session-only conversion to foreign INT96 writers.
            # Decode the authenticated bytes instead of changing the caller's session.
            spark = None
        before = directory_digest(local, suffixes=(".parquet",))
        with operation(
            "Calculate distributed logical SHA" if spark else "Calculate logical SHA offline",
            engine="spark" if spark else "driver",
            algorithm=ALGORITHM,
            dataset=str(path),
            files=len(files),
            bytes=sum(file.stat().st_size for file in files),
            rows=rows,
            **({"reason": "foreign INT96 requires decoding without session timezone conversion"}
               if fallback else {"reason": "small capture fits local hashing limits"}
               if small else {}),
        ):
            if spark is None:
                result = _offline(local, schema, rows, max(1, batch_rows))
            else:

                frame = (
                    spark.read.schema(_spark_type({"struct": schema}))
                    .option("datetimeRebaseMode", "CORRECTED")
                    .option("int96RebaseMode", "CORRECTED")
                    .parquet(str(path))
                )
                result = _fingerprint_frame(frame, schema, rows=rows)
            if directory_digest(path, suffixes=(".parquet",)) != before:
                raise ValueError("captured Parquet files changed during logical hashing")
            progress(completed=rows, total=rows, unit="rows", phase="logical SHA complete")
        return result, before


def fingerprint_frame(frame):
    """Hash rows from a pinned, retained native capture without a Parquet export.

    The adapter must keep the same captured read alive through hashing and use.
    This function does not turn a mutable query or an evictable cache into a
    snapshot. Its canonical format is identical to the Parquet implementations.
    """
    from pyspark.sql.pandas.types import to_arrow_schema

    schema = _schema(to_arrow_schema(frame.schema, timezone="UTC"))
    return _fingerprint_frame(frame, schema)


def _fingerprint_frame(frame, schema, *, rows=None):
    from pyspark.sql import functions as F

    expressions = [_expression(F.col("`" + n.replace("`", "``") + "`"), t)
                   for n, t in schema]
    payload = F.concat(*expressions) if expressions else F.lit("")
    hashes = frame.select(F.sha2(payload, 256).alias("hex"))
    hashes = (hashes.select(
        F.conv(F.substring("hex", 1, 2), 16, 10).cast("int").alias("bucket"),
        F.unhex("hex").alias("digest"),
    ).repartition("bucket").sortWithinPartitions("bucket", "digest"))
    with operation("Sort and combine bucket digests", engine="spark", algorithm=ALGORITHM):
        summaries = hashes.mapInArrow(_bucket_summaries, "bucket int, rows long, sha256 string")
        collected = [row.asDict() for row in summaries.limit(BUCKETS + 1).collect()]
        return _finish(schema, collected,
                       rows if rows is not None else sum(item["rows"] for item in collected))


def verify_files(
    path, *, physical_sha256, logical, trusted, spark=None, _scratch_root=None,
):
    """Reuse trusted bindings; otherwise decode authenticated bytes again.

    Spark verification requires an explicit worker-visible scratch root for a
    fresh owned copy. Without one, decode locally and leave Spark caches alone.
    """
    from mlops import _volume_files

    from provenance.lineage import directory_digest

    path = pathlib.Path(path)
    if logical is not None:
        validate_metadata(logical)
    with operation("Verify frozen file bytes", engine="driver", dataset=str(path)):
        with (
            _volume_files.local_snapshot(path, parquet_only=True)
            if _volume_files.is_volume(path)
            else nullcontext(path)
        ) as local:
            from provenance.lineage import parquet_files

            files = parquet_files(local)
            inventory = {"dataset": str(path), "files": len(files),
                         "bytes": sum(file.stat().st_size for file in files)}
            if directory_digest(local, suffixes=(".parquet",)) != physical_sha256:
                raise ValueError("frozen file bytes or inventory changed")
            if trusted and logical is not None:
                note(
                    "Reuse verified logical identity",
                    algorithm=logical["algorithm"],
                    rows=logical["rows"],
                    **inventory,
                    reason="authenticated manifest and matching file bytes",
                )
                return logical["sha256"]
            note("Full logical verification", reason="no authenticated binding or legacy format")
            if logical is None:
                from autoforecast.benchmark import parquet_content_hash

                observed_sha = parquet_content_hash(local)
            elif spark is not None and _scratch_root is not None:
                import shutil

                from provenance.lineage import snapshot_parquet_directory

                # Spark may retain cached rows for an existing path. Read a new,
                # authenticated capture without changing the caller's cache.
                root = pathlib.Path(tempfile.mkdtemp(prefix="af-verify-hash-", dir=_scratch_root))
                failure = None
                try:
                    owned = snapshot_parquet_directory(
                        local, root / "data", expected_sha256=physical_sha256,
                        label="logical verification",
                    )
                    observed = fingerprint(owned, spark=spark)
                    if directory_digest(owned, suffixes=(".parquet",)) != physical_sha256:
                        raise ValueError("verification snapshot bytes changed")
                    if observed != logical:
                        raise ValueError("frozen logical identity changed")
                    observed_sha = observed["sha256"]
                except BaseException as error:
                    failure = error
                    raise
                finally:
                    try:
                        if _volume_files.is_volume(root):
                            _volume_files.remove_tree(root)
                        else:
                            shutil.rmtree(root)
                    except Exception as error:
                        if failure is None:
                            raise
                        failure.add_note(f"Verification snapshot cleanup also failed: {error}")
            else:
                # Without a declared worker-visible scratch area, decode the
                # authenticated bytes locally. Never reopen a cached Spark path.
                observed = fingerprint(local)
                if observed != logical:
                    raise ValueError("frozen logical identity changed")
                observed_sha = observed["sha256"]
            if directory_digest(local, suffixes=(".parquet",)) != physical_sha256:
                raise ValueError("frozen file bytes changed during logical verification")
            if _volume_files.is_volume(path) and (
                directory_digest(path, suffixes=(".parquet",)) != physical_sha256
            ):
                raise ValueError("frozen file bytes changed during logical verification")
            return observed_sha
