"""Lossless, timezone-naive Parquet time-column storage evidence."""

from __future__ import annotations

import base64
import binascii
import pathlib
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from provenance.lineage import LineageError, parquet_files


class ParquetTimeError(ValueError):
    """A Parquet time column lacks lossless timezone-naive storage evidence."""


@dataclass(frozen=True)
class ParquetTimeStorage:
    """Footer evidence needed by runtimes that distinguish INT96 from native timestamps."""

    all_int96: bool


def validate_parquet_time_storage(
    path: str | pathlib.Path,
    time_col: str,
    *,
    allow_hive_partition: bool = False,
) -> ParquetTimeStorage:
    """Validate every visible fragment without relying on reconstructed INT96 schemas."""
    root = pathlib.Path(path)
    try:
        fragments = parquet_files(root)
    except LineageError as error:
        raise ParquetTimeError(str(error)) from error
    if not fragments:
        raise ParquetTimeError(f"Parquet artifact {path} contains no visible fragments")

    int96: list[tuple[pathlib.Path, str]] = []
    for fragment in fragments:
        try:
            with pq.ParquetFile(fragment) as part:
                physical = [column for column in part.schema if column.path == time_col]
                if not physical and allow_hive_partition and _hive_partition_has(
                    root, fragment, time_col
                ):
                    continue
                if len(physical) != 1:
                    raise ParquetTimeError(
                        f"time column {time_col!r} in {fragment} does not have exactly one "
                        "physical Parquet field"
                    )
                if physical[0].physical_type == "INT96":
                    field_type = _original_int96_type(part, fragment, time_col)
                    int96.append((fragment, field_type.unit))
                else:
                    index = part.schema_arrow.get_field_index(time_col)
                    if index < 0:
                        raise ParquetTimeError(
                            f"time column {time_col!r} is absent from {fragment}"
                        )
                    field_type = part.schema_arrow.field(index).type
                if pa.types.is_timestamp(field_type) and field_type.tz is not None:
                    raise ParquetTimeError(
                        f"time column {time_col!r} in {fragment} is timezone-bearing "
                        f"({field_type}); normalize it before import"
                    )
        except ParquetTimeError:
            raise
        except (OSError, pa.ArrowException) as error:
            raise ParquetTimeError(
                f"could not inspect Parquet time column {time_col!r} in {fragment}: {error}"
            ) from error

    _require_microsecond_int96(int96, time_col)
    return ParquetTimeStorage(all_int96=len(int96) == len(fragments))


def _hive_partition_has(root: pathlib.Path, fragment: pathlib.Path, column: str) -> bool:
    """Whether this fragment receives a nonblank exact-name column from its relative path."""
    if root.is_file():
        return False
    try:
        parts = fragment.relative_to(root).parts[:-1]
    except ValueError:
        return False
    prefix = f"{column}="
    return any(part.startswith(prefix) and len(part) > len(prefix) for part in parts)


def _original_int96_type(
    part: pq.ParquetFile, fragment: pathlib.Path, time_col: str,
) -> pa.TimestampType:
    """Return the original INT96 timestamp type; reconstructed schemas erase its timezone."""
    encoded = (part.metadata.metadata or {}).get(b"ARROW:schema")
    if encoded is None:
        raise ParquetTimeError(
            f"INT96 time column {time_col!r} in {fragment} lacks original embedded "
            "ARROW:schema proof of timezone-naive storage"
        )
    try:
        original = pa.ipc.read_schema(
            pa.BufferReader(base64.b64decode(encoded, validate=True)),
        )
    except (pa.ArrowException, binascii.Error, ValueError) as error:
        raise ParquetTimeError(
            f"INT96 time column {time_col!r} in {fragment} has invalid original embedded "
            "ARROW:schema"
        ) from error
    index = original.get_field_index(time_col)
    if index < 0 or not pa.types.is_timestamp(original.field(index).type):
        raise ParquetTimeError(
            f"INT96 time column {time_col!r} in {fragment} is not a timestamp in its original "
            "embedded ARROW:schema"
        )
    return original.field(index).type


def _require_microsecond_int96(
    columns: list[tuple[pathlib.Path, str]], time_col: str,
) -> None:
    """Refuse nanoseconds Spark would truncate; retain wider coarser timestamp ranges."""
    for fragment, unit in columns:
        if unit != "ns":
            continue
        try:
            with pq.ParquetFile(fragment, coerce_int96_timestamp_unit="ns") as part:
                for batch in part.iter_batches(columns=[time_col], batch_size=65_536):
                    pc.cast(batch.column(0), pa.timestamp("us"), safe=True)
        except pa.ArrowInvalid as error:
            raise ParquetTimeError(
                f"INT96 time column {time_col!r} in {fragment} would lose sub-microsecond "
                "precision"
            ) from error


__all__ = [
    "ParquetTimeError",
    "ParquetTimeStorage",
    "validate_parquet_time_storage",
]
