"""CSV mapping must preserve identity and reject missing or ambiguous observations."""

import datetime as dt
import hashlib
import json
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

from autoforecast.tabular import import_csv


def convert(tmp_path, text, **kwargs):
    source = tmp_path / "input.csv"
    source.write_text(text)
    return import_csv(
        source, tmp_path / "data/observations.parquet",
        source_keys=kwargs.get("source_keys", ["shop", "sku"]),
        time_col=kwargs.get("time_col", "day"), target_col=kwargs.get("target_col", "units"),
    )


def test_csv_preserves_keys_zeros_dates_and_source_identity(tmp_path):
    path = convert(tmp_path, 'shop,sku,day,units,note\n"a,b",001,2024-01-01,0,private\na,NA,2024-01-02,2.5,extra\n')
    rows = pq.read_table(path).to_pylist()
    assert rows == [
        {"shop": "a,b", "sku": "001", "day": dt.date(2024, 1, 1), "units": 0.0},
        {"shop": "a", "sku": "NA", "day": dt.date(2024, 1, 2), "units": 2.5},
    ]
    record = json.loads(pq.read_metadata(path).metadata[b"autoforecast.csv_import"])
    assert record["source_sha256"] == hashlib.sha256((tmp_path / "input.csv").read_bytes()).hexdigest()
    assert list(path.parent.iterdir()) == [path]
    assert record["row_count"] == record["series_count"] == 2
    assert record["excluded_columns"] == ["note"]


@pytest.mark.parametrize("row,match", [
    ("a,1,2024-01-01,", "invalid date or numeric"),
    ("a,1,2024-01-01,NaN", "finite and nonnegative"),
    ("a,1,2024-01-01,inf", "finite and nonnegative"),
    ("a,1,2024-01-01,-1", "finite and nonnegative"),
    ('a,1,2024-01-01,"unclosed', "malformed CSV"),
    ("a,1,2024-01-01,1e999", "finite and nonnegative"),
    ("a,1,2024-02-30,1", "invalid date or numeric"),
    ("a,1,2024-01-01T00:00:00,1", "YYYY-MM-DD"),
    ("a,1,20240101,1", "YYYY-MM-DD"),
    ("a,1,2024-1-1,1", "YYYY-MM-DD"),
    ("a,,2024-01-01,1", "empty series key"),
    (" ,1,2024-01-01,1", "empty series key"),
    ("a,1,2024-01-01", "field count"),
    ("a,1,2024-01-01,1,extra", "field count"),
    ("a,1,2024-01-01,1\na,1,2024-01-01,2", "duplicate series/date"),
])
def test_csv_refuses_invalid_observations_without_output(tmp_path, row, match):
    with pytest.raises(ValueError, match=match):
        convert(tmp_path, "shop,sku,day,units\n" + row + "\n")
    assert not (tmp_path / "data/observations.parquet").exists()


@pytest.mark.parametrize("text", ["", "shop,sku,day,units\n", "shop,sku,day\na,1,2024-01-01\n", "shop,sku,day,units,units\na,1,2024-01-01,1,2\n"])
def test_csv_refuses_empty_and_invalid_headers(tmp_path, text):
    with pytest.raises(ValueError):
        convert(tmp_path, text)


@pytest.mark.parametrize("columns", [{"source_keys": []}, {"source_keys": ["shop", "shop"]}, {"time_col": "sku"}, {"target_col": ""}])
def test_csv_refuses_ambiguous_mapping(tmp_path, columns):
    with pytest.raises(ValueError, match="distinct, nonempty"):
        convert(tmp_path, "shop,sku,day,units\na,1,2024-01-01,1\n", **columns)


def test_csv_retries_identical_bytes_and_refuses_conflicting_output(tmp_path):
    text = "shop,sku,day,units\na,1,2024-01-01,1\n"
    path = convert(tmp_path, text)
    original = path.read_bytes()
    assert convert(tmp_path, text).read_bytes() == original
    with pytest.raises(OSError):
        convert(tmp_path, text.replace(",1\n", ",2\n"))
    assert path.read_bytes() == original


def test_csv_cli_maps_non_retail_column_names(tmp_path):
    source = tmp_path / "meter.csv"
    source.write_text("meter,date,kwh\n0007,2024-01-01,12.5\n")
    output = tmp_path / "meter.parquet"
    result = subprocess.run([
        sys.executable, "-m", "autoforecast.cli", "import-csv", str(source), str(output),
        "--id-columns", "meter", "--date-column", "date", "--target-column", "kwh",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert pq.read_table(output).to_pylist() == [
        {"meter": "0007", "date": dt.date(2024, 1, 1), "kwh": 12.5},
    ]


def test_csv_refuses_case_ambiguous_mapped_names(tmp_path):
    with pytest.raises(ValueError, match="distinct ignoring case"):
        convert(tmp_path, "shop,SHOP,day,units\na,b,2024-01-01,1\n",
                source_keys=["shop", "SHOP"])
    assert not (tmp_path / "data/observations.parquet").exists()
