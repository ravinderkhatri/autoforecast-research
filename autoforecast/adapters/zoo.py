"""AutoGluon-TimeSeries: a whole model zoo behind one call. OPTIONAL EXTRA.

This is the adapter the two-package split was designed around. AutoGluon already wraps
StatsForecast's locals, GluonTS's neural models, tabular LightGBM and Chronos, and ensembles
them -- so ranking many models is a solved problem and writing one would be rung-0
reimplementation.

**It cannot express this architecture's contract, and it competes anyway -- with that on the
record.** Three gaps, none incidental:

  · no cell-level loss exclusion, so a never-ranged cell cannot be kept out of the loss;
  · no censored observations, so a stockout day trains the model toward the suppressed number;
  · missing values are FILLED, which is precisely the failure range eligibility exists to
    prevent -- a never-ranged cell filled with zero teaches the occurrence head an assortment
    decision rather than a demand event.

So it registers with `consumes_exposure=False` and `registry.register` refuses it unless it
declares `HANDICAP`. Every comparison it appears in carries that string. An adapter that cannot
honour the contract is allowed to compete; it is not allowed to compete silently.

**Its selection rule is also the leak `ranking.py` exists to prevent** -- "the model that achieves
the best validation score will be used", by default a single window of the last
`prediction_length` steps. That is why this adapter is given ONE origin at a time and scored by
our harness: AutoGluon's own model choice happens inside a single call, on data our development
origins already cover, and its reported validation number is never used here.

OPTIONAL, and deliberately not a core dependency: it pulls torch, gluonts and ray. No gate before
Phase 3 depends on it, so the import is inside the function and its absence is a clear message
rather than an ImportError at collection time.
"""

from __future__ import annotations

import datetime as dt
import pathlib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters import _AUTOGLUON_ZOO_HANDICAP, _validate_daily_span
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel

NAME = "autogluon_zoo"

EMITTED = "emitted"
DECLINED_NO_HISTORY = "declined_no_eligible_history"

HANDICAP = _AUTOGLUON_ZOO_HANDICAP

PRESETS = "medium_quality"
"""AutoGluon preset. Part of the tuning budget Phase 3 requires to be equal across compared
models, so it is a named constant that gets recorded rather than an inline default."""


def fit_predict(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    presets: str = PRESETS,
    time_limit: int = 600,
) -> Forecast:
    """Fit AutoGluon's zoo on eligible history up to `origin` and forecast `horizon` days.

    Args:
        panel: Data. Eligible cells only are passed -- which is the most this adapter can do
            about the contract, since AutoGluon has no way to mark a cell as unobserved.
        origin: Last date the zoo may see, ISO.
        horizon: Days to forecast.
        spark: Active session, used to read the panel and write the result.
        out_dir: Where the forecast Parquet is written.
        presets: AutoGluon quality preset.
        time_limit: Seconds AutoGluon may spend. Recorded explicitly for equal-budget comparisons.

    Returns:
        A validated `Forecast` whose mean forecast is used -- AutoGluon emits quantiles, and the
        harness scores a point forecast at this phase.

    Raises:
        ValueError: non-positive horizon, or no series with eligible history.
        ImportError: `autogluon.timeseries` is not installed, with what to install.
    """
    _validate_daily_span(panel, origin, horizon, adapter=NAME)
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:  # pragma: no cover -- exercised by test_zoo when absent
        raise ImportError(
            "autogluon.timeseries is not installed. It is an OPTIONAL extra for this adapter "
            "only -- `pip install autogluon.timeseries` -- and it pulls torch, gluonts and ray. "
            "No gate before Phase 3 depends on it."
        ) from exc

    frame, n_series, n_declined = _eligible_frame(panel, spark, origin, horizon)
    if frame.empty:
        raise ValueError(f"no series has eligible history at or before {origin}")

    data = TimeSeriesDataFrame.from_data_frame(frame, id_column="item_id",
                                              timestamp_column="timestamp")
    predictor = TimeSeriesPredictor(prediction_length=horizon, target="target",
                                    eval_metric="MASE", verbosity=0)
    predictor.fit(data, presets=presets, time_limit=time_limit)
    out = _to_forecast_frame(predictor.predict(data), origin, horizon)

    spark.createDataFrame(out).write.mode("overwrite").parquet(str(out_dir))
    disposition = {EMITTED: int(len(out))}
    if n_declined:
        disposition[DECLINED_NO_HISTORY] = n_declined
    return Forecast(adapter=NAME, path=str(out_dir), origin=origin, horizon=horizon,
                    n_series=n_series, disposition=disposition,
                    target_estimand=panel.target_estimand,
                    frequency=panel.frequency).validate()


def _eligible_frame(panel: Panel, spark: SparkSession, origin: str, horizon: int):
    """Eligible history to `origin`, in AutoGluon's column contract, plus the series counts.

    Filtering to eligible cells is the most that can be done here. It is NOT equivalent to the
    exposure contract: removing an interior ineligible stretch leaves the surrounding days
    adjacent, and AutoGluon will then FILL the gap it was given -- so a never-ranged period comes
    back as imputed demand. That is the first line of `HANDICAP` and it cannot be fixed from
    outside the library.
    """
    df = spark.read.parquet(panel.path).filter(
        F.col(panel.time_col) <= F.to_date(F.lit(origin)))
    total = (spark.read.parquet(panel.path)
             .select(panel.series_col).distinct().count())
    if panel.has_exposure:
        df = df.filter(F.col(panel.exposure_col).isin(*sorted(ELIGIBLE_STATES)))
    df = df.select(F.col(panel.series_col).alias("item_id"),
                   F.col(panel.time_col).cast("timestamp").alias("timestamp"),
                   F.col(panel.target_col).cast("double").alias("target"))
    kept = df.select("item_id").distinct().count()
    return df.orderBy("item_id", "timestamp").toPandas(), total, (total - kept) * horizon


def _to_forecast_frame(predictions, origin: str, horizon: int):
    """AutoGluon's quantile frame -> `series_id · date · yhat`, using the mean column.

    AutoGluon returns a `mean` column alongside quantiles. The harness scores a point forecast at
    this phase, so the mean is what is written -- and the quantiles are dropped rather than
    silently reduced to one, which is a different claim.
    """
    import pandas as pd
    d0 = dt.date.fromisoformat(origin)
    flat = predictions.reset_index()
    rows = []
    for item, group in flat.groupby("item_id", sort=True):
        for step, (_, r) in enumerate(group.iterrows()):
            if step >= horizon:
                break
            rows.append((str(item), d0 + dt.timedelta(days=step + 1),
                         float(max(r["mean"], 0.0))))
    return pd.DataFrame(rows, columns=["series_id", "date", "yhat"])


__all__ = ["HANDICAP", "NAME", "PRESETS", "fit_predict"]
