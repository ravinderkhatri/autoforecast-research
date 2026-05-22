"""Real UCI Bike Sharing count panels for revision portability tests.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset (CC BY 4.0).
Fanaee-T and Gama (2013), doi:10.1007/s13748-013-0040-3. No modeled weather is treated
as a forecast: temperature is a historical-only input in the optional feature test.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import pathlib
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

ARCHIVE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"


def build_parquet(
    archive: pathlib.Path, out_path: pathlib.Path, *, weekly: bool
) -> pathlib.Path:
    """Materialize the pinned real archive; tested by ``test_real_count_portability``.

    Weekly observations sum complete consecutive seven-day blocks only; the final three
    days outside the 104 complete blocks are not called a week. Outputs never overwrite.
    """
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Bike Sharing source archive differs from the recorded primary source")
    if type(weekly) is not bool:
        raise ValueError("weekly must be boolean")
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        raw = list(csv.DictReader(io.StringIO(source.read("day.csv").decode("utf-8"))))
    dates = [dt.date.fromisoformat(row["dteday"]) for row in raw]
    if len(raw) != 731 or any(
        b - a != dt.timedelta(days=1) for a, b in zip(dates, dates[1:], strict=False)
    ):
        raise ValueError("source is not the complete ordered daily count dataset")
    width = 7 if weekly else 1
    rows = []
    for start in range(0, len(raw) - width + 1, width):
        window = raw[start : start + width]
        for segment in ("casual", "registered"):
            rows.append(
                dict(
                    series=segment,
                    timestamp=dates[start + width - 1],
                    rentals=sum(int(row[segment]) for row in window),
                    workdays=sum(int(row["workingday"]) for row in window),
                    temperature=sum(float(row["temp"]) for row in window) / width,
                    weather=str(window[-1]["weathersit"]),
                    membership=segment,
                    registered_member=float(segment == "registered"),
                )
            )
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {
            b"source_sha256": ARCHIVE_SHA256.encode(),
            b"license": b"CC BY 4.0",
            b"source": b"https://doi.org/10.24432/C5W894",
            b"aggregation": b"complete_seven_day_sums" if weekly else b"original_daily_counts",
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("xb") as output:
        pq.write_table(table, output)
    return out_path
