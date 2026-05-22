"""Direct structural tests for the M5 private-replay loader.

The quantities below are invented zeros used only to exercise file-shape contracts. They are not
M5 observations, are never scored, and must not be reported as empirical findings. A minimal
calendar fixture tests the supplied calendar boundary; no downloaded or private data is opened.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from autoforecast.benchmarks.m5 import load_private_evaluation_long

PRIVATE_DAYS = tuple(f"d_{day}" for day in range(1942, 1970))
STRUCTURAL_IDENTITY = {
    "item_id": "STRUCTURAL_ITEM",
    "dept_id": "STRUCTURAL_DEPT",
    "cat_id": "STRUCTURAL_CAT",
    "store_id": "STRUCTURAL_STORE",
    "state_id": "STRUCTURAL_STATE",
}


def _write_inert_private_shape(
    labels_dir: pathlib.Path,
    *,
    days: tuple[str, ...] = PRIVATE_DAYS,
    omit_identity: tuple[str, ...] = (),
) -> None:
    labels_dir.mkdir()
    row = {
        name: value for name, value in STRUCTURAL_IDENTITY.items() if name not in omit_identity
    }
    row.update({day: 0 for day in days})
    pd.DataFrame([row]).to_csv(labels_dir / "sales_test_evaluation.csv", index=False)


def test_load_private_evaluation_long_maps_an_inert_shape_through_the_supplied_calendar(
    tmp_path: pathlib.Path,
) -> None:
    labels_dir = tmp_path / "structural-private-shape"
    _write_inert_private_shape(labels_dir)
    calendar_dir = tmp_path / "structural-calendar"
    calendar_dir.mkdir()
    pd.DataFrame({
        "d": PRIVATE_DAYS,
        "date": pd.date_range("2016-05-23", "2016-06-19", freq="D"),
        "wm_yr_wk": [1] * len(PRIVATE_DAYS),
    }).to_csv(calendar_dir / "calendar.csv", index=False)

    long = load_private_evaluation_long(labels_dir, calendar_dir=calendar_dir)

    assert long["d"].tolist() == list(PRIVATE_DAYS)
    assert long["qty"].eq(0).all()
    assert long["id"].unique().tolist() == [
        "STRUCTURAL_ITEM_STRUCTURAL_STORE_evaluation"
    ]
    assert long["date"].tolist() == list(
        pd.date_range("2016-05-23", "2016-06-19", freq="D")
    )


def test_load_private_evaluation_long_refuses_an_absent_label_file(
    tmp_path: pathlib.Path,
) -> None:
    labels_dir = tmp_path / "absent-private-labels"
    labels_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="sales_test_evaluation.csv"):
        load_private_evaluation_long(labels_dir, calendar_dir=tmp_path / "not-opened")


def test_load_private_evaluation_long_refuses_a_structural_fixture_missing_identity(
    tmp_path: pathlib.Path,
) -> None:
    labels_dir = tmp_path / "missing-identity-shape"
    _write_inert_private_shape(labels_dir, omit_identity=("state_id",))

    with pytest.raises(ValueError, match=r"missing identity columns: \['state_id'\]"):
        load_private_evaluation_long(labels_dir, calendar_dir=tmp_path / "not-opened")


def test_load_private_evaluation_long_refuses_a_structural_fixture_missing_a_day(
    tmp_path: pathlib.Path,
) -> None:
    labels_dir = tmp_path / "missing-day-shape"
    _write_inert_private_shape(labels_dir, days=PRIVATE_DAYS[:-1])

    with pytest.raises(ValueError, match=r"exactly d_1942\.\.d_1969 in order"):
        load_private_evaluation_long(labels_dir, calendar_dir=tmp_path / "not-opened")


def test_load_private_evaluation_long_refuses_a_structural_fixture_with_reordered_days(
    tmp_path: pathlib.Path,
) -> None:
    labels_dir = tmp_path / "reordered-day-shape"
    reordered = (PRIVATE_DAYS[1], PRIVATE_DAYS[0], *PRIVATE_DAYS[2:])
    _write_inert_private_shape(labels_dir, days=reordered)

    with pytest.raises(ValueError, match=r"exactly d_1942\.\.d_1969 in order"):
        load_private_evaluation_long(labels_dir, calendar_dir=tmp_path / "not-opened")
