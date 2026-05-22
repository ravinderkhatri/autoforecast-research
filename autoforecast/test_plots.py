"""Tests for the diagnostic plot, on REAL M5 data through Spark.

A plot is hard to assert on, so these check the things that would make one MISLEADING rather than
its appearance: that the series are selected without seeing the held-out period, that the
selection is reproducible, that intermittency is actually spanned, and that a file lands.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters.naive import seasonal_naive
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel
from autoforecast.plots import diagnostic_plot, series_across_zero_rate_deciles

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
# The decile-ordering defect did NOT appear on one store -- it needed enough series for a decile
# to hold more than one, which is the whole panel. A test scoped to CA_1 passed while the figure
# was mislabelled, so the selection test uses all ten stores deliberately.
ALL_STORES = ROOT / "data" / "m5_parquet"
ORIGIN = "2016-04-24"
HORIZON = 28

_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    """One local session for the file."""
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME is not set; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[*]").appName("test_plots")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false")
                  .getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def panel(path: pathlib.Path | None = None) -> Panel:
    source = STORE if path is None else path
    if source in {STORE, ALL_STORES} and not source.is_dir():
        pytest.skip("public M5 Parquet is not installed")
    return Panel(path=source, series_col="id", time_col="date", target_col="qty",
                 exposure_col="observation_state", season_length=7).validate()


def test_the_sample_spans_real_intermittency_rather_than_clustering() -> None:
    # A random sample of a 59%-zero catalogue is a sample of slow movers. Deciles are the point.
    p = panel(ALL_STORES)
    picked = series_across_zero_rate_deciles(p, spark(), ORIGIN, n=8)
    ids = [sid for sid, _ in picked]
    assert len(ids) == 8
    d = spark().read.parquet(str(ALL_STORES)).filter(
        (F.col("date") <= F.lit(ORIGIN)) & F.col("observation_state").isin(*sorted(ELIGIBLE_STATES)))
    rates = {r["id"]: r["zr"] for r in
             d.filter(F.col("id").isin(*ids))
              .groupBy("id").agg(F.avg(F.when(F.col("qty") == 0, 1.0).otherwise(0.0)).alias("zr"))
              .collect()}
    got = [rates[i] for i in ids]
    assert got == sorted(got), f"returned out of zero-rate order: {[f'{z:.0%}' for z in got]}"
    # The RETURNED rate must be the selected series' own rate, recomputed here independently.
    # Pairing a series with another series' rate is invisible in the ordering -- deciles are
    # contiguous, so the order stays monotonic either way -- and shows up only on the figure.
    for sid, claimed in picked:
        assert abs(claimed - rates[sid]) < 1e-9, (
            f"{sid}: selection reported zero rate {claimed:.4f}, its actual rate is "
            f"{rates[sid]:.4f} -- the pair is mismatched")
    assert max(got) - min(got) > 0.30, (
        f"the eight series span only {max(got)-min(got):.1%} of zero rate; that is a cluster, "
        "not a spread across deciles")


def test_the_selection_is_deterministic_so_a_gate_is_reproducible() -> None:
    a = series_across_zero_rate_deciles(panel(), spark(), ORIGIN, n=8)
    b = series_across_zero_rate_deciles(panel(), spark(), ORIGIN, n=8)
    assert a == b, "two runs of the same gate would show different evidence"


def test_the_selection_cannot_see_the_held_out_period() -> None:
    # If the zero rate were computed over all history, an origin change would not move it. It must.
    early = series_across_zero_rate_deciles(panel(), spark(), "2013-01-01", n=8)
    late = series_across_zero_rate_deciles(panel(), spark(), ORIGIN, n=8)
    assert early != late, (
        "the same series were chosen from a 2013 origin and a 2016 one, so the selection is not "
        "restricted to data available at the origin")


def test_a_series_with_too_little_history_is_not_offered() -> None:
    # min_history above anything M5 can supply must yield nothing rather than a useless plot.
    assert series_across_zero_rate_deciles(panel(), spark(), ORIGIN, n=8, min_history=10**6) == []


def test_diagnostic_plot_writes_a_png_for_a_real_forecast() -> None:
    p = panel()
    d = tempfile.mkdtemp()
    fc = seasonal_naive(p, ORIGIN, HORIZON, spark=spark(), out_dir=d)
    out = diagnostic_plot(p, fc, spark(), pathlib.Path(d) / "plots" / "naive.png", n_series=4)
    assert out.exists() and out.stat().st_size > 10_000, (
        f"wrote {out.stat().st_size if out.exists() else 0} bytes; that is not a rendered figure")


def test_diagnostic_plot_refuses_when_nothing_has_enough_history() -> None:
    p = panel()
    d = tempfile.mkdtemp()
    fc = seasonal_naive(p, ORIGIN, 7, spark=spark(), out_dir=d)
    import autoforecast.plots as plots
    real = plots.series_across_zero_rate_deciles
    plots.series_across_zero_rate_deciles = lambda *a, **k: []
    try:
        diagnostic_plot(p, fc, spark(), pathlib.Path(d) / "x.png")
    except ValueError as e:
        assert "nothing to draw" in str(e); return
    finally:
        plots.series_across_zero_rate_deciles = real
    raise AssertionError("drew a figure with no series")


def test_diagnostic_plot_requests_the_full_declared_forecast_period_count(monkeypatch, tmp_path) -> None:
    import autoforecast.plots as plots

    seen: list[int] = []
    monkeypatch.setattr(
        plots, "series_across_zero_rate_deciles", lambda *args, **kwargs: [("series", 0.5)]
    )
    monkeypatch.setattr(
        plots,
        "_collect_history",
        lambda _panel, _spark, _ids, _origin, _encoder, span: seen.append(span) or {},
    )
    monkeypatch.setattr(plots, "_collect_forecast", lambda *args, **kwargs: {})
    forecast = Forecast(
        "weekly", "/not-read", "2024-01-01", 2, 1, {"emitted": 2}, frequency="W"
    )
    plots.diagnostic_plot(
        Panel("/not-read", "id", "date", "qty", frequency="W"),
        forecast,
        object(),
        tmp_path / "weekly.png",
        n_series=1,
    )
    assert seen == [2]


def test_diagnostic_plot_refuses_panel_forecast_cadence_mismatch(monkeypatch, tmp_path) -> None:
    import autoforecast.plots as plots

    monkeypatch.setattr(
        plots, "series_across_zero_rate_deciles", lambda *args, **kwargs: [("series", 0.5)]
    )
    monkeypatch.setattr(plots, "_collect_history", lambda *args, **kwargs: {})
    monkeypatch.setattr(plots, "_collect_forecast", lambda *args, **kwargs: {})
    forecast = Forecast(
        "weekly", "/not-read", "2024-01-01", 2, 1, {"emitted": 2}, frequency="W"
    )
    with pytest.raises(ValueError, match="cadence"):
        plots.diagnostic_plot(
            Panel("/not-read", "id", "date", "qty", frequency="D"),
            forecast,
            object(),
            tmp_path / "mismatched.png",
            n_series=1,
        )


@pytest.mark.parametrize("origin,horizon", [("20240101", 2), ("2024-01-01", True)])
def test_diagnostic_plot_refuses_noncanonical_span_before_io(
    monkeypatch, tmp_path, origin, horizon
) -> None:
    import autoforecast.plots as plots

    monkeypatch.setattr(
        plots, "series_across_zero_rate_deciles", lambda *args, **kwargs: [("series", 0.5)]
    )
    monkeypatch.setattr(plots, "_collect_history", lambda *args, **kwargs: {})
    monkeypatch.setattr(plots, "_collect_forecast", lambda *args, **kwargs: {})
    forecast = Forecast(
        "candidate", "/not-read", origin, horizon, 1, {"emitted": int(horizon)}
    )
    with pytest.raises(ValueError, match="canonical|horizon"):
        plots.diagnostic_plot(
            Panel("/not-read", "id", "date", "qty"),
            forecast,
            object(),
            tmp_path / "invalid-span.png",
            n_series=1,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
