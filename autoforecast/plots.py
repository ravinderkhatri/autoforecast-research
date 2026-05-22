"""Plot history, held-out actuals and point forecasts together. Series are selected across
zero-rate deciles in Spark; only the selected histories are collected to the driver. Prediction
intervals are not drawn by this module.
"""

from __future__ import annotations

import pathlib

import matplotlib
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from autoforecast.frequency import TimePoint, advance, parse_frequency, parse_origin
from autoforecast.grid import canonical_key, time_literal
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel

matplotlib.use("Agg")  # headless: these are written to disk, never shown
import matplotlib.pyplot as plt  # noqa: E402  must follow the backend selection

INELIGIBLE_SHADE = "#E0E0E0"


def series_across_zero_rate_deciles(
    panel: Panel, spark: SparkSession, origin: str, *, n: int = 8, min_history: int = 90
) -> list[tuple[str, float]]:
    """Pick `n` series spanning the zero-rate deciles, measured on eligible cells before `origin`.

    Deterministic: the lowest series id in each decile is taken, so a plot is reproducible from
    the manifest without carrying a random seed. Random sampling here would make two runs of the
    same gate produce different evidence.

    Args:
        panel: Source of history.
        spark: Active session.
        origin: Only cells at or before this date are used, so the selection cannot see the
            held-out period it is about to be plotted against.
        n: How many series, one per decile.
        min_history: Eligible observations a series needs to be worth plotting. A series with
            twelve observed days produces a plot that shows nothing.

    Returns:
        Up to `n` `(series_id, zero_rate)` pairs, ordered by zero rate. The rate is returned WITH
        the series and not recomputed by the caller: the plot labels each panel with it, and an
        earlier version labelled panels with the zero rate of the drawn 148-day window instead --
        a different quantity, non-monotonic down the figure, and indistinguishable to a reader
        from the one the figure claims to be spanning.

        Fewer than `n` when the panel has fewer than `n` series with enough history -- returned
        rather than raised, because a small benchmark is a legitimate input and a caller can see
        the length.
    """
    panel.validate()
    cadence = parse_frequency(panel.frequency)
    origin_point = parse_origin(origin, cadence)
    time_key, valid_time_key = canonical_key(panel.time_col, cadence)
    source = spark.read.parquet(panel.path)
    if cadence.key_kind == "timestamp" and (
        source.schema[panel.time_col].dataType.typeName() != "timestamp_ntz"
    ):
        raise ValueError(
            f"hourly panel time column {panel.time_col!r} must have timestamp_ntz physical type"
        )
    prepared = source.withColumn(
        "__af_plot_time", time_key
    ).withColumn("__af_plot_valid_time", valid_time_key)
    if prepared.filter(~F.col("__af_plot_valid_time")).limit(1).count():
        raise ValueError(
            f"panel contains a malformed, lossy, or off-anchor {cadence.code} time key"
        )
    df = prepared.filter(F.col("__af_plot_time") <= time_literal(origin_point, cadence))
    if panel.has_exposure:
        df = df.filter(F.col(panel.exposure_col).isin(*sorted(ELIGIBLE_STATES)))
    rates = (df.groupBy(F.col(panel.series_col).alias("sid"))
             .agg(F.avg(F.when(F.col(panel.target_col) == 0, 1.0).otherwise(0.0)).alias("zr"),
                  F.count(F.lit(1)).alias("n_obs"))
             .filter(F.col("n_obs") >= min_history))
    ranked = rates.withColumn("decile", F.ntile(n).over(Window.orderBy("zr", "sid")))
    # r22: `groupBy(decile).agg(min(sid), min(zr))` computed the two minima INDEPENDENTLY, so the
    # zero rate reported beside a series belonged to a different series in the same decile -- the
    # panels came out labelled 55%, 50%, 54%, which is not the ordering the figure claims. Taking
    # one whole row per decile keeps the series and its rate together.
    # One WHOLE row per decile. `groupBy(decile).agg(min(sid), min(zr))` would take the two
    # minima independently, pairing a series with another series' rate -- harmless for ordering,
    # since deciles are contiguous, but wrong the moment the rate is displayed.
    pick = (ranked.withColumn("rn", F.row_number().over(
                Window.partitionBy("decile").orderBy("zr", "sid")))
            .filter(F.col("rn") == 1).orderBy("zr", "sid").collect())
    return [(r["sid"], float(r["zr"])) for r in pick]


def diagnostic_plot(
    panel: Panel,
    forecast: Forecast,
    spark: SparkSession,
    out_path: str | pathlib.Path,
    *,
    n_series: int = 8,
    encoder_days: int = 120,
) -> pathlib.Path:
    """Draw history, held-out actuals and forecast for a spread of series. Returns the path.

    Ineligible cells are shaded rather than plotted, because a forecast on a never-ranged cell is
    not an error and must not read as one -- which is the same distinction the metric makes, drawn.

    Args:
        panel: Actuals and history.
        forecast: What to draw. Only its emitted cells exist; a declined cell simply has no line,
            which is the honest picture.
        spark: Active session.
        out_path: PNG destination. Parent directories are created.
        n_series: Series to draw, one per zero-rate decile.
        encoder_days: Historical periods before the origin to show. The name is retained for
            daily-call compatibility.

    Returns:
        The written path.

    Raises:
        ValueError: no series had enough history to plot.
    """
    if panel.frequency != forecast.frequency:
        raise ValueError(
            f"panel cadence {panel.frequency!r} differs from forecast cadence "
            f"{forecast.frequency!r}; the plot would use inconsistent dates"
        )
    if type(forecast.horizon) is not int or forecast.horizon < 1:
        raise ValueError(
            f"forecast horizon must be a positive integer, got {forecast.horizon!r}"
        )
    if type(encoder_days) is not int or encoder_days < 1:
        raise ValueError(f"encoder_days must be a positive period count, got {encoder_days!r}")
    forecast.validate()
    out_path = pathlib.Path(out_path)
    try:
        cadence = parse_frequency(forecast.frequency)
        origin = parse_origin(forecast.origin, cadence)
        advance(origin, cadence, forecast.horizon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"forecast origin must be canonical for {forecast.frequency!r}, got "
            f"{forecast.origin!r}"
        ) from exc
    picked = series_across_zero_rate_deciles(panel, spark, forecast.origin, n=n_series)
    ids = [sid for sid, _ in picked]
    if not ids:
        raise ValueError(
            f"no series in {panel.path} has enough eligible history to plot -- nothing to draw")

    hist = _collect_history(
        panel,
        spark,
        ids,
        origin,
        encoder_days,
        forecast.horizon,
    )
    pred = _collect_forecast(forecast, spark, ids)

    fig, axes = plt.subplots(len(ids), 1, figsize=(11, 2.1 * len(ids)), sharex=True)
    axes = axes if len(ids) > 1 else [axes]
    for ax, (sid, zero_rate) in zip(axes, picked, strict=True):
        _draw_series(
            ax,
            sid,
            hist.get(sid, []),
            pred.get(sid, []),
            origin,
            zero_rate,
            forecast.frequency,
        )
    axes[0].legend(loc="upper left", fontsize=7, ncol=4)
    axes[-1].set_xlabel("date")
    fig.suptitle(f"{forecast.adapter} · origin {forecast.origin} · h={forecast.horizon} · "
                 f"series span the zero-rate deciles", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _collect_history(
    panel: Panel, spark: SparkSession, ids: list[str], origin: TimePoint,
    encoder_days: int, horizon: int,
) -> dict[str, list[tuple[TimePoint, float, bool]]]:
    """Pull (date, qty, eligible) for the chosen series only, over the drawn window.

    Filtered in Spark before collecting. Reading the panel and then subsetting on the driver is
    the step that works on M5 and dies on an estate.
    """
    cadence = parse_frequency(panel.frequency)
    lo, hi = advance(origin, cadence, -encoder_days), advance(origin, cadence, horizon)
    source = spark.read.parquet(panel.path)
    if cadence.key_kind == "timestamp" and (
        source.schema[panel.time_col].dataType.typeName() != "timestamp_ntz"
    ):
        raise ValueError(
            f"hourly panel time column {panel.time_col!r} must have timestamp_ntz physical type"
        )
    time_key, valid_time_key = canonical_key(panel.time_col, cadence)
    cols = [F.col(panel.series_col).alias("sid"), time_key.alias("d"),
            valid_time_key.alias("_valid_time"),
            F.col(panel.target_col).cast("double").alias("y")]
    cols.append(F.col(panel.exposure_col).alias("st") if panel.has_exposure
                else F.lit("exact").alias("st"))
    rows = (source.select(*cols)
            .filter(F.col("sid").isin(*ids))
            .filter(F.col("_valid_time"))
            .filter(
                (F.col("d") >= time_literal(lo, cadence))
                & (F.col("d") <= time_literal(hi, cadence))
            )
            .orderBy("sid", "d").collect())
    out: dict[str, list[tuple[TimePoint, float, bool]]] = {}
    for r in rows:
        out.setdefault(r["sid"], []).append((r["d"], r["y"], r["st"] in ELIGIBLE_STATES))
    return out


def _collect_forecast(
    forecast: Forecast, spark: SparkSession, ids: list[str]
) -> dict[str, list[tuple[TimePoint, float]]]:
    """Pull (date, yhat) for the chosen series only."""
    rows = (spark.read.parquet(forecast.path)
            .filter(F.col("series_id").isin(*ids))
            .select("series_id", "date", "yhat").orderBy("series_id", "date").collect())
    out: dict[str, list[tuple[TimePoint, float]]] = {}
    for r in rows:
        out.setdefault(r["series_id"], []).append((r["date"], float(r["yhat"])))
    return out


def _draw_series(
    ax, sid: str, hist, pred, origin: TimePoint, zero_rate: float, frequency: str
) -> None:
    """One panel: history, held-out actuals, forecast, and shaded ineligible spans.

    `zero_rate` is the SELECTION's rate -- eligible cells over the whole history to the origin --
    passed in rather than recomputed here. Recomputing it over the drawn window gave a different
    number, which made the figure's own ordering look wrong.
    """
    past = [(d, y) for d, y, el in hist if d <= origin]
    future = [(d, y) for d, y, el in hist if d > origin]
    ax.plot([d for d, _ in past], [y for _, y in past], lw=0.8, color="#546E7A", label="history")
    if future:
        ax.plot([d for d, _ in future], [y for _, y in future], lw=1.2, color="#2E7D32",
                label="held-out actual")
    if pred:
        ax.plot([d for d, _ in pred], [y for _, y in pred], lw=1.2, color="#C62828",
                ls="--", label="forecast")
    for d, _, el in hist:
        if not el:
            # One span per ineligible day. Drawn rather than omitted so a gap in the line is
            # visibly "not ranged" instead of ambiguously "no data".
            ax.axvspan(d, advance(d, frequency, 1), color=INELIGIBLE_SHADE, lw=0,
                       label="ineligible" if d == hist[0][0] else None)
    ax.axvline(origin, color="#000000", lw=0.8, ls=":")
    ax.set_ylabel(f"{sid[:18]}\nzero {zero_rate:.0%}", fontsize=6)
    ax.tick_params(labelsize=7)


__all__ = ["diagnostic_plot", "series_across_zero_rate_deciles"]
