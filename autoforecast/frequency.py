"""Shared time-cadence parsing and calendar arithmetic.

The five public cadence families are hourly, daily, weekly, monthly, and yearly.  Monthly and
yearly declarations use explicit standard anchors so rolling an origin never changes the grid:
``M``/``ME`` are month end, ``MS`` is month start, ``Y``/``YE`` are December 31, and ``YS`` is
January 1.  Calendar arithmetic imports pandas lazily; importing adapter metadata therefore does
not import a numerical runtime.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

TimePoint = dt.date | dt.datetime


class FrequencyError(ValueError):
    """A cadence or time point is outside the supported canonical contract."""


@dataclass(frozen=True)
class Cadence:
    """One parsed cadence declaration.

    ``unit`` is the model-capability family. ``code`` keeps anchor semantics for identities:
    compatibility aliases ``ME`` and ``YE`` canonicalise to ``M`` and ``Y`` respectively.
    """

    code: str
    unit: Literal["H", "D", "W", "M", "Y"]
    multiplier: int
    key_kind: Literal["timestamp", "date"]
    anchor: Literal["start", "end"] | None = None

    def __post_init__(self) -> None:
        _validate_cadence(self)


FrequencyLike = str | Cadence

_CadenceFields = tuple[
    str,
    Literal["H", "D", "W", "M", "Y"],
    int,
    Literal["timestamp", "date"],
    Literal["start", "end"] | None,
]

_FIXED_FIELDS: dict[str, _CadenceFields] = {
    "H": ("H", "H", 1, "timestamp", None),
    "D": ("D", "D", 1, "date", None),
    "W": ("W", "W", 1, "date", None),
    "M": ("M", "M", 1, "date", "end"),
    "ME": ("M", "M", 1, "date", "end"),
    "MS": ("MS", "M", 1, "date", "start"),
    "Y": ("Y", "Y", 1, "date", "end"),
    "YE": ("Y", "Y", 1, "date", "end"),
    "YS": ("YS", "Y", 1, "date", "start"),
}


def _canonical_fields(code: object) -> _CadenceFields:
    if not isinstance(code, str) or not code:
        raise FrequencyError(f"frequency must be a non-blank string, got {code!r}")
    known = _FIXED_FIELDS.get(code)
    if known is not None:
        return known
    match = re.fullmatch(r"([1-9][0-9]*)D", code)
    if match is not None:
        multiplier = int(match.group(1))
        return ("D" if multiplier == 1 else code, "D", multiplier, "date", None)
    raise FrequencyError(
        "supported frequencies are H, D, W, M/ME, MS, Y/YE, YS, or a positive integral nD"
    )


def _validate_cadence(cadence: Cadence) -> None:
    expected = _canonical_fields(cadence.code)
    actual = (
        cadence.code,
        cadence.unit,
        cadence.multiplier,
        cadence.key_kind,
        cadence.anchor,
    )
    if (
        type(cadence.code) is not str
        or type(cadence.unit) is not str
        or type(cadence.multiplier) is not int
        or type(cadence.key_kind) is not str
        or (cadence.anchor is not None and type(cadence.anchor) is not str)
        or actual != expected
    ):
        raise FrequencyError(
            f"Cadence fields {actual!r} are inconsistent; canonical fields are {expected!r}"
        )


_PUBLIC = {declaration: Cadence(*fields) for declaration, fields in _FIXED_FIELDS.items()}


def parse_frequency(code: FrequencyLike) -> Cadence:
    """Parse a supported cadence without importing pandas or Spark.

    Positive integral ``nD`` declarations remain accepted for existing revision workflows.
    They are members of the daily family but retain their multiplier and canonical code.
    """
    if isinstance(code, Cadence):
        _validate_cadence(code)
        return code
    known = _PUBLIC.get(code) if isinstance(code, str) else None
    return known if known is not None else Cadence(*_canonical_fields(code))


def frequency_is_supported(
    frequency: FrequencyLike,
    supported_families: Collection[str],
    *,
    supports_strided_daily: bool = False,
) -> bool:
    """Apply both family and non-unit daily-stride capability declarations.

    ``nD`` represents a stride of multiple days, but membership in the ``D`` family does
    not prove that a model emits ``n``-day-spaced forecasts. Adapters must opt in separately.
    """
    cadence = parse_frequency(frequency)
    return cadence.unit in supported_families and (
        cadence.multiplier == 1 or supports_strided_daily
    )


def parse_origin(origin: str, frequency: FrequencyLike) -> TimePoint:
    """Parse a canonical origin for ``frequency``.

    Hourly origins are timezone-naive exact hours. All other origins are ISO dates, with monthly
    and yearly origins additionally required to lie on their declared standard anchor.
    """
    cadence = parse_frequency(frequency)
    if not isinstance(origin, str):
        raise FrequencyError(f"origin must be a canonical string, got {origin!r}")
    if cadence.key_kind == "timestamp":
        try:
            point = dt.datetime.fromisoformat(origin)
        except ValueError as exc:
            raise FrequencyError(
                f"hourly origin must be canonical YYYY-MM-DDTHH:00:00, got {origin!r}"
            ) from exc
        _validate_hour(point, "origin")
        if point.isoformat(timespec="seconds") != origin:
            raise FrequencyError(
                f"hourly origin must be canonical YYYY-MM-DDTHH:00:00, got {origin!r}"
            )
        return point
    try:
        point = dt.date.fromisoformat(origin)
    except ValueError as exc:
        raise FrequencyError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}") from exc
    if point.isoformat() != origin:
        raise FrequencyError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}")
    _validate_anchor(point, cadence, "origin")
    return point


def canonical_origin(point: str | TimePoint, frequency: FrequencyLike) -> str:
    """Return the canonical serialized form of one cadence-aligned point."""
    cadence = parse_frequency(frequency)
    normalized = _coerce_point(point, cadence)
    if isinstance(normalized, dt.datetime):
        return normalized.isoformat(timespec="seconds")
    return normalized.isoformat()


def advance(
    origin_point: str | TimePoint,
    frequency: FrequencyLike,
    periods: int,
) -> TimePoint:
    """Advance from the supplied anchor by an integral number of cadence periods."""
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise FrequencyError(f"periods must be an integer, got {periods!r}")
    cadence = parse_frequency(frequency)
    point = _coerce_point(origin_point, cadence)
    try:
        if cadence.unit == "H":
            assert isinstance(point, dt.datetime)
            return point + dt.timedelta(hours=periods)
        assert isinstance(point, dt.date) and not isinstance(point, dt.datetime)
        if cadence.unit == "D":
            return point + dt.timedelta(days=cadence.multiplier * periods)
        if cadence.unit == "W":
            return point + dt.timedelta(days=7 * periods)

        # Pandas is pinned by the runtime that executes calendar-cadence pipelines. Keep this
        # import off metadata/discovery paths and use anchored offsets rather than relative month
        # deltas, whose clipping breaks rolling-origin equivalence after February.
        from pandas import Timestamp
        from pandas.tseries.offsets import MonthBegin, MonthEnd, YearBegin, YearEnd

        offset_type = {
            ("M", "start"): MonthBegin,
            ("M", "end"): MonthEnd,
            ("Y", "start"): YearBegin,
            ("Y", "end"): YearEnd,
        }[(cadence.unit, cadence.anchor)]
        result = Timestamp(point) + offset_type(periods)
        return result.date()
    except (OverflowError, ValueError) as exc:
        raise FrequencyError(
            f"{cadence.code} cannot represent {periods} periods from "
            f"{canonical_origin(point, cadence)}"
        ) from exc


def period_points(
    origin_or_point: str | TimePoint,
    frequency: FrequencyLike,
    horizon: int,
) -> tuple[TimePoint, ...]:
    """Return the ``horizon`` cadence points strictly after the original anchor."""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise FrequencyError(f"horizon must be a positive integer, got {horizon!r}")
    cadence = parse_frequency(frequency)
    origin = _coerce_point(origin_or_point, cadence)
    return tuple(advance(origin, cadence, step) for step in range(1, horizon + 1))


def _coerce_point(point: str | TimePoint, cadence: Cadence) -> TimePoint:
    if isinstance(point, str):
        return parse_origin(point, cadence)
    # Arrow timestamp[ns] values may be pandas Timestamp subclasses. Their nanosecond
    # remainder is not represented by datetime.microsecond, time(), or date().
    if isinstance(point, dt.datetime) and getattr(point, "nanosecond", 0):
        raise FrequencyError(f"time point has sub-microsecond precision: {point!r}")
    if cadence.key_kind == "timestamp":
        if not isinstance(point, dt.datetime):
            raise FrequencyError(f"{cadence.code} requires a datetime point, got {point!r}")
        _validate_hour(point, "time point")
        return point
    if isinstance(point, dt.datetime):
        if point.tzinfo is not None or point.time() != dt.time():
            raise FrequencyError(
                f"{cadence.code} requires a date or timezone-naive midnight timestamp, got "
                f"{point!r}"
            )
        normalized = point.date()
    elif isinstance(point, dt.date):
        normalized = point
    else:
        raise FrequencyError(f"{cadence.code} requires a date point, got {point!r}")
    _validate_anchor(normalized, cadence, "time point")
    return normalized


def _validate_hour(point: dt.datetime, label: str) -> None:
    if point.tzinfo is not None or any((point.minute, point.second, point.microsecond)):
        raise FrequencyError(
            f"hourly {label} must be timezone-naive and aligned to an exact hour, got {point!r}"
        )


def _validate_anchor(point: dt.date, cadence: Cadence, label: str) -> None:
    if cadence.unit == "M":
        anchored = point.day == (1 if cadence.anchor == "start" else calendar.monthrange(
            point.year, point.month
        )[1])
    elif cadence.unit == "Y":
        anchored = (point.month, point.day) == ((1, 1) if cadence.anchor == "start" else (12, 31))
    else:
        return
    if not anchored:
        anchor_name = "month start" if cadence.unit == "M" and cadence.anchor == "start" else \
            "month end" if cadence.unit == "M" else \
            "January 1" if cadence.anchor == "start" else "December 31"
        raise FrequencyError(
            f"{cadence.code} {label} must be aligned to {anchor_name}, got {point.isoformat()}"
        )


__all__ = [
    "Cadence",
    "FrequencyError",
    "TimePoint",
    "advance",
    "canonical_origin",
    "frequency_is_supported",
    "parse_frequency",
    "parse_origin",
    "period_points",
]
