"""Signed distance from each horizon day to each calendar event.

SIGNED, NOT ABSOLUTE. The three days before Christmas and the three days after are opposite
states of the world -- one is a build-up, the other is a dead zone -- and an absolute distance
makes them identical. This is the single most common way an event feature is built wrongly.

`event_known_at` IS ENFORCED HERE. An event may only contribute to a forecast made at an origin
if it was ANNOUNCED by that origin. Without it a model trained on historical promotions learns to
foresee unannounced ones: it scores beautifully in backtest, and live it cannot do it. Public
holidays are known years ahead and pass trivially; a retailer's own promotion often is not.
"""

from __future__ import annotations

import datetime as dt
import pathlib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

__all__ = ["CalendarError", "build"]


class CalendarError(ValueError):
    """An event table was given that cannot be used at this origin."""


def build(
    events: list[tuple[str, str, str | None]],
    origin: str,
    horizon: int,
    series: list[str],
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    max_support: int = 30,
    event_windows: dict[str, tuple[int, int]] | None = None,
) -> pathlib.Path:
    """Write `series_id · date · <event>_distance · <event>_present`. Returns the path.

    Args:
        events: `(name, date, known_at)` triples. **The same name may -- and for a recurring
            event MUST -- appear with several dates.** "Easter" with its 2013-2016 dates is ONE
            event whose kernel and deviations pool across four occurrences; naming each year
            separately splits the N the pooling design exists to grow ("a product has seen about
            TWO Easters"), and it starved multi-year training windows entirely: a window whose
            target spanned Easter 2015 saw an "Easter" dated 2016, 350 days away and out of
            support, so a real seasonal spike trained with no event feature at all.
            `known_at` is when the event became public; `None` means always known, which is
            right for a fixed public holiday and WRONG for a promotion. There is no default for
            a reason -- guessing "always known" for a retailer's own campaign is exactly the
            leak this argument exists to prevent.
        origin: Forecast origin. An occurrence whose `known_at` is after this is DROPPED.
        horizon: Days to emit.
        series: Series to emit for.
        spark: Active session.
        out_dir: Parquet destination.
        max_support: Days either side beyond which the event is marked absent. Matches the
            kernel's fixed window in `first_build`.
        event_windows: PER-EVENT `(lower, upper)` signed-day ceilings overriding the shared
            symmetric one -- `ForecastDataSpec.event_windows`. A pay-day effect has no
            build-up and a holiday has no payback tail; one shared window forces both to carry
            the other's dead days. Each bound must bracket zero, and names not listed keep the
            shared ceiling.

    Returns:
        The written path. Distance is measured to the NEAREST announced occurrence of the name;
        with occurrences a year apart and a support of +/-30 days, at most one is ever in range,
        so "nearest" is not blending two.

    Raises:
        CalendarError: a malformed origin, event date or announcement date, or the same
            `(name, date)` pair listed twice -- which would be one occurrence counted as two.
    """
    if type(horizon) is not int or horizon < 1:
        raise CalendarError(f"horizon must be a positive integer, got {horizon!r}")
    try:
        origin_day = dt.date.fromisoformat(origin)
    except (TypeError, ValueError) as exc:
        raise CalendarError(f"forecast origin has a malformed ISO date {origin!r}") from exc
    if origin_day.isoformat() != origin:
        raise CalendarError(
            f"forecast origin must be canonical ISO YYYY-MM-DD, got {origin!r}")

    occurrences: dict[str, list[dt.date]] = {}
    seen: set[tuple[str, dt.date]] = set()
    for name, when, known_at in events:
        try:
            event_day = dt.date.fromisoformat(when)
        except (TypeError, ValueError) as exc:
            raise CalendarError(f"event {name!r} has a malformed date {when!r}") from exc
        if (name, event_day) in seen:
            raise CalendarError(
                f"event {name!r} lists {when} twice; one occurrence would be counted as two")
        seen.add((name, event_day))
        if known_at is not None:
            try:
                known_day = dt.date.fromisoformat(known_at)
            except (TypeError, ValueError) as exc:
                raise CalendarError(
                    f"event {name!r} has a malformed known_at ISO date {known_at!r}") from exc
            if known_day > origin_day:
                continue                 # not announced at forecast time; it does not exist yet
        occurrences.setdefault(name, []).append(event_day)
    names = list(dict.fromkeys(e[0] for e in events))
    windows = {}
    for name in names:
        low, high = (event_windows or {}).get(name, (-max_support, max_support))
        if not (low <= 0 <= high):
            raise CalendarError(
                f"event {name!r} window ({low}, {high}) must bracket zero: a window that "
                "excludes the event's own day is a different feature wearing this one's name")
        windows[name] = (low, high)

    rows = [(sid, origin_day + dt.timedelta(days=step)) for sid in series
            for step in range(1, horizon + 1)]
    df = spark.createDataFrame(rows, "series_id string, date date")
    for name in names:
        days = occurrences.get(name, [])
        if not days:
            # Announced never (or not yet): emitted as absent rather than omitted, because the
            # model's event columns are positional and a client-specific column set would change
            # the tensor's width between origins.
            df = (df.withColumn(f"{name}_distance", F.lit(0))
                  .withColumn(f"{name}_present", F.lit(False)))
            continue
        nearest = F.datediff(F.col("date"), F.lit(days[0].isoformat()))
        for day in days[1:]:
            candidate = F.datediff(F.col("date"), F.lit(day.isoformat()))
            nearest = F.when(F.abs(candidate) < F.abs(nearest), candidate).otherwise(nearest)
        low, high = windows[name]
        in_window = (nearest >= low) & (nearest <= high)
        df = (df.withColumn(f"{name}_distance", F.when(in_window, nearest).otherwise(0))
              .withColumn(f"{name}_present", in_window))
    df.write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir))
