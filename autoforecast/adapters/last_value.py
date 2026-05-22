"""A second small baseline, reusing the governed seasonal-naive observation semantics."""

from dataclasses import replace

from autoforecast.adapters.naive import seasonal_naive


def fit_predict(panel, origin, horizon, *, spark, out_dir):
    """Repeat the origin observation; absent or unobserved cells remain declined."""
    forecast = seasonal_naive(
        replace(panel, season_length=1), origin, horizon, spark=spark, out_dir=out_dir
    )
    return replace(forecast, adapter="last_value")
