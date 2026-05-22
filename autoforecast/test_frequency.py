"""Discriminating tests for the shared five-cadence time contract."""

from __future__ import annotations

import datetime as dt
import pathlib
import subprocess
import sys

import pytest

from autoforecast.frequency import (
    Cadence,
    FrequencyError,
    advance,
    canonical_origin,
    frequency_is_supported,
    parse_frequency,
    parse_origin,
    period_points,
)


def test_frequency_import_does_not_import_pandas() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    code = (
        "import sys; import autoforecast.frequency; "
        "assert 'pandas' not in sys.modules, sorted(k for k in sys.modules if k.startswith('pandas'))"
    )
    subprocess.run(
        [sys.executable, "-c", code], cwd=root, check=True, capture_output=True, text=True,
    )


@pytest.mark.parametrize(
    "declaration,code,unit,multiplier,key_kind,anchor",
    [
        ("H", "H", "H", 1, "timestamp", None),
        ("D", "D", "D", 1, "date", None),
        ("1D", "D", "D", 1, "date", None),
        ("2D", "2D", "D", 2, "date", None),
        ("W", "W", "W", 1, "date", None),
        ("M", "M", "M", 1, "date", "end"),
        ("ME", "M", "M", 1, "date", "end"),
        ("MS", "MS", "M", 1, "date", "start"),
        ("Y", "Y", "Y", 1, "date", "end"),
        ("YE", "Y", "Y", 1, "date", "end"),
        ("YS", "YS", "Y", 1, "date", "start"),
    ],
)
def test_parse_frequency_exposes_family_anchor_and_physical_key(
    declaration, code, unit, multiplier, key_kind, anchor,
) -> None:
    cadence = parse_frequency(declaration)
    assert (cadence.code, cadence.unit, cadence.multiplier, cadence.key_kind, cadence.anchor) == (
        code, unit, multiplier, key_kind, anchor,
    )


@pytest.mark.parametrize("value", ["", "h", "0D", "01D", "-1D", "YE-FEB", "QS", None])
def test_parse_frequency_rejects_unknown_or_noncanonical_declarations(value) -> None:
    with pytest.raises(FrequencyError, match="supported|non-blank"):
        parse_frequency(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fields",
    [
        ("ME", "M", 1, "date", "end"),
        ("H", "D", 1, "date", None),
        ("D", "D", 2, "date", None),
        ("D", "D", True, "date", None),
        ("D", "D", 1.0, "date", None),
        ("H", "H", 1, "date", None),
        ("M", "M", 1, "date", "start"),
        ("2D", "D", 1, "date", None),
    ],
)
def test_cadence_constructor_rejects_every_inconsistent_field(fields) -> None:
    with pytest.raises(FrequencyError, match="inconsistent"):
        Cadence(*fields)  # type: ignore[arg-type]


def test_parse_frequency_revalidates_an_existing_cadence_object() -> None:
    forged = object.__new__(Cadence)
    for name, value in zip(
        ("code", "unit", "multiplier", "key_kind", "anchor"),
        ("H", "D", 1, "date", None),
        strict=True,
    ):
        object.__setattr__(forged, name, value)

    with pytest.raises(FrequencyError, match="inconsistent"):
        parse_frequency(forged)
    with pytest.raises(FrequencyError, match="inconsistent"):
        frequency_is_supported(forged, ("D",))


def test_nonunit_daily_stride_requires_an_explicit_capability() -> None:
    families = ("D",)
    assert frequency_is_supported("D", families)
    assert frequency_is_supported("1D", families)
    assert not frequency_is_supported("7D", families)
    assert frequency_is_supported("7D", families, supports_strided_daily=True)


def test_hourly_points_preserve_exact_hours_across_midnight() -> None:
    points = period_points("2024-03-01T22:00:00", "H", 3)
    assert points == (
        dt.datetime(2024, 3, 1, 23),
        dt.datetime(2024, 3, 2, 0),
        dt.datetime(2024, 3, 2, 1),
    )
    assert canonical_origin(points[0], "H") == "2024-03-01T23:00:00"


@pytest.mark.parametrize(
    "origin",
    [
        "2024-03-01",
        "2024-03-01T22:00",
        "2024-03-01T22:30:00",
        "2024-03-01T22:00:01",
        "2024-03-01T22:00:00+00:00",
    ],
)
def test_hourly_origin_rejects_lossy_or_noncanonical_values(origin: str) -> None:
    with pytest.raises(FrequencyError, match="hour"):
        parse_origin(origin, "H")


def test_fixed_date_cadences_count_periods() -> None:
    assert period_points("2024-01-01", "D", 2) == (
        dt.date(2024, 1, 2), dt.date(2024, 1, 3),
    )
    assert period_points("2024-01-01", "2D", 2) == (
        dt.date(2024, 1, 3), dt.date(2024, 1, 5),
    )
    assert period_points("2024-01-03", "W", 2) == (
        dt.date(2024, 1, 10), dt.date(2024, 1, 17),
    )


def test_month_end_points_handle_leap_february_and_are_rolling_origin_invariant() -> None:
    from_january = period_points("2024-01-31", "M", 3)
    assert from_january == (
        dt.date(2024, 2, 29), dt.date(2024, 3, 31), dt.date(2024, 4, 30),
    )
    assert period_points("2024-02-29", "ME", 1) == (from_january[1],)


def test_month_and_year_start_aliases_remain_on_their_standard_anchors() -> None:
    assert period_points("2024-01-01", "MS", 2) == (
        dt.date(2024, 2, 1), dt.date(2024, 3, 1),
    )
    assert period_points("2024-01-01", "YS", 2) == (
        dt.date(2025, 1, 1), dt.date(2026, 1, 1),
    )


def test_year_end_points_remain_december_31_across_a_leap_year() -> None:
    points = period_points("2023-12-31", "Y", 2)
    assert points == (dt.date(2024, 12, 31), dt.date(2025, 12, 31))
    assert period_points("2024-12-31", "YE", 1) == (points[1],)


@pytest.mark.parametrize(
    "origin,frequency",
    [
        ("2024-01-30", "M"),
        ("2024-01-02", "MS"),
        ("2024-02-29", "Y"),
        ("2024-12-31", "YS"),
    ],
)
def test_calendar_cadences_reject_off_anchor_origins(origin: str, frequency: str) -> None:
    with pytest.raises(FrequencyError, match="aligned"):
        parse_origin(origin, frequency)


def test_date_keys_refuse_lossy_nonmidnight_or_timezone_aware_timestamps() -> None:
    for point in (
        dt.datetime(2024, 1, 1, 1),
        dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
    ):
        with pytest.raises(FrequencyError, match="midnight|timezone"):
            canonical_origin(point, "D")
    assert canonical_origin(dt.datetime(2024, 1, 1), "D") == "2024-01-01"


def test_horizon_and_period_count_are_strict_integers() -> None:
    for horizon in (0, -1, True, 1.0):
        with pytest.raises(FrequencyError, match="horizon"):
            period_points("2024-01-01", "D", horizon)  # type: ignore[arg-type]
    with pytest.raises(FrequencyError, match="periods"):
        advance(dt.date(2024, 1, 1), "D", True)
