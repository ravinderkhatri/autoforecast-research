"""Convert explicitly mapped CSV observations to the daily Parquet input contract."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import pathlib
import re


def import_csv(
    source: pathlib.Path,
    output: pathlib.Path,
    *,
    source_keys: list[str],
    time_col: str,
    target_col: str,
) -> pathlib.Path:
    """Preserve string keys and observed zeros; reject ambiguous or invalid observations.

    Only the explicitly mapped columns are imported. Extra source columns are recorded
    in Parquet metadata. Use typed Parquet directly for feature-rich datasets.
    The source is read once so the retained digest identifies exactly the parsed bytes.
    This local convenience importer materializes the selected observations in memory.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mlops._safe_files import publish_bytes_set

    columns = [*source_keys, time_col, target_col]
    if (
        not source_keys
        or any(not isinstance(name, str) or not name.strip() for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("declare distinct, nonempty source key, date, and target columns")
    if output.suffix != ".parquet":
        raise ValueError("output must end in .parquet")
    payload = source.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    try:
        header = reader.fieldnames
        if not header or len(set(header)) != len(header) or not set(columns) <= set(header):
            raise ValueError("CSV needs unique headers and every mapped column")
        rows = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV row {line}: field count differs from header")
            if any(not row[key].strip() for key in source_keys):
                raise ValueError(f"CSV row {line}: empty series key")
            date = row[time_col]
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date):
                raise ValueError(f"CSV row {line}: date must be YYYY-MM-DD")
            try:
                day = dt.date.fromisoformat(date)
                target = float(row[target_col])
            except ValueError as exc:
                raise ValueError(f"CSV row {line}: invalid date or numeric target") from exc
            if not math.isfinite(target) or target < 0:
                raise ValueError(f"CSV row {line}: target must be finite and nonnegative")
            key = (*(row[name] for name in source_keys), day)
            if key in seen:
                raise ValueError(f"CSV row {line}: duplicate series/date observation")
            seen.add(key)
            rows.append(
                {**{name: row[name] for name in source_keys}, time_col: day, target_col: target}
            )
    except csv.Error as error:
        raise ValueError(f"CSV row {reader.reader.line_num}: malformed CSV: {error}") from error
    if not rows:
        raise ValueError("CSV contains no observations")
    schema = pa.schema(
        [(name, pa.string()) for name in source_keys]
        + [(time_col, pa.date32()), (target_col, pa.float64())]
    )
    record = {
        "schema": "autoforecast.csv-import.v1",
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_keys": source_keys,
        "time_col": time_col,
        "target_col": target_col,
        "excluded_columns": [name for name in header if name not in columns],
        "series_count": len({key[:-1] for key in seen}),
        "row_count": len(rows),
        "first_date": min(row[time_col] for row in rows).isoformat(),
        "last_date": max(row[time_col] for row in rows).isoformat(),
    }
    schema = schema.with_metadata({b"autoforecast.csv_import": json.dumps(record).encode()})
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
    parquet = sink.getvalue().to_pybytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes_set(output.parent, {output.name: parquet})
    return output
