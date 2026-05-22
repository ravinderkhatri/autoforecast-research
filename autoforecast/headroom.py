"""Estimate residual headroom above a baseline with a leave-one-out oracle. The oracle uses the
other observations with the same series and weekday within the held-out horizon. It is a
diagnostic proxy, never a deployable adapter or a fair model comparator. Its remaining error and
reducible_share depend on the chosen horizon and available covariates.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from dataclasses import dataclass

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from autoforecast._numeric import is_finite_real
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel, PanelError, period_days
from autoforecast.ranking import MissingForecastPolicy, finalize_forecast_on_grid
from autoforecast.score import Score
from autoforecast.score_spark import MissingActual, score_forecast

ORACLE = "_oracle_leave_one_out"
"""Name written into the oracle's `Forecast`. Leading underscore because it is not a competitor,
and `registry.get` will never return it -- nothing registers it."""


class HeadroomError(ValueError):
    """A headroom estimate was asked for in a way that would not mean what it appears to."""


@dataclass(frozen=True)
class Headroom:
    """How much of the best model's error an oracle can remove.

    Attributes:
        best_adapter: Which model this is headroom ABOVE.
        best_mae: Its MAE on the common cells.
        oracle_mae: The leave-one-out oracle's MAE on the same cells.
        n_scored: Cells both were scored on.
    """

    best_adapter: str
    best_mae: float
    oracle_mae: float
    n_scored: int

    def __post_init__(self) -> None:
        """Refuse an estimate that cannot be read as a share.

        Raises:
            HeadroomError: nothing was scored, or the best MAE is zero, which would make every
                share a division by zero rather than a perfect model worth reporting.
        """
        if not isinstance(self.best_adapter, str) or not self.best_adapter.strip():
            raise HeadroomError("best_adapter must be a non-blank name")
        if (not isinstance(self.n_scored, int) or isinstance(self.n_scored, bool)
                or self.n_scored < 1):
            raise HeadroomError("no cells were scored, so there is no headroom to estimate")
        if any(not is_finite_real(value) or value < 0
               for value in (self.best_mae, self.oracle_mae)):
            raise HeadroomError(
                f"headroom MAEs must be finite and non-negative, got best={self.best_mae!r}, "
                f"oracle={self.oracle_mae!r}")
        if self.best_mae <= 0:
            raise HeadroomError(
                f"{self.best_adapter} has MAE {self.best_mae}, so headroom is undefined. A zero "
                "MAE on real data is a leak, not a result")

    @property
    def reducible_share(self) -> float:
        """Fraction of the best model's error the oracle removes. Clamped at 0.

        Negative would mean the oracle is WORSE than the fitted model, which happens when a model
        genuinely uses covariates the oracle cannot see. Reported as zero headroom rather than as
        a negative, because "the oracle lost" is not evidence that error is reducible.
        """
        return max(0.0, (self.best_mae - self.oracle_mae) / self.best_mae)

    @property
    def irreducible_share(self) -> float:
        """Fraction of the best model's error that survives even an oracle."""
        return 1.0 - self.reducible_share

    @property
    def oracle_lost(self) -> bool:
        """Whether the oracle was beaten -- worth reporting, never silently clamped away."""
        return self.oracle_mae > self.best_mae

    def as_record(self) -> dict[str, object]:
        """A JSON-safe record for the gate result."""
        return {"best_adapter": self.best_adapter, "best_mae": self.best_mae,
                "oracle_mae": self.oracle_mae, "n_scored": self.n_scored,
                "reducible_share": self.reducible_share,
                "irreducible_share": self.irreducible_share,
                "oracle_lost": self.oracle_lost}


def oracle_forecast(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
) -> Forecast:
    """A leave-one-out day-of-week oracle over the HELD-OUT window. Not a forecaster.

    For each eligible cell in the declared `horizon`-period grid, predicts the mean of the
    other cells with the same series and day of week inside that window: `(sum - y) / (n - 1)`.

    It sees the answer. That is the point -- it bounds what any predictor could achieve if it knew
    the horizon's day-of-week structure exactly -- and it is why this function is not registered.

    Args:
        panel: Data. Must carry actuals for the whole horizon.
        origin: Forecast origin, ISO.
        horizon: Periods after `origin` to cover.
        spark: Active session.
        out_dir: Where the oracle's Parquet is written.

    Returns:
        A validated `Forecast` labelled `ORACLE`.

    Raises:
        HeadroomError: no eligible cell has a same-day-of-week partner in the window, so
            leave-one-out is undefined everywhere.
    """
    if type(horizon) is not int or horizon < 1:
        raise HeadroomError(f"horizon must be a positive integer, got {horizon!r}")
    try:
        d0 = dt.date.fromisoformat(origin)
        stride = period_days(panel.frequency)
        dates = tuple(
            d0 + dt.timedelta(days=stride * step) for step in range(1, horizon + 1)
        )
    except (TypeError, ValueError, OverflowError, PanelError) as exc:
        raise HeadroomError("origin/frequency does not define a supported period grid") from exc
    if d0.isoformat() != origin:
        raise HeadroomError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}")
    df = spark.read.parquet(panel.path).filter(
        F.to_date(F.col(panel.time_col)).isin(*dates))
    if panel.has_exposure:
        df = df.filter(F.col(panel.exposure_col).isin(*sorted(ELIGIBLE_STATES)))
    df = df.select(F.col(panel.series_col).alias("series_id"),
                   F.col(panel.time_col).alias("date"),
                   F.col(panel.target_col).cast("double").alias("y"))

    w = Window.partitionBy("series_id", F.dayofweek("date"))
    scored = (df.withColumn("_sum", F.sum("y").over(w))
              .withColumn("_n", F.count(F.lit(1)).over(w))
              .filter(F.col("_n") > 1)
              .withColumn("yhat", (F.col("_sum") - F.col("y")) / (F.col("_n") - 1))
              .select("series_id", "date", "yhat"))
    n_emitted = scored.count()
    if n_emitted == 0:
        raise HeadroomError(
            f"no eligible cell in the {horizon}-period grid after {origin} has another cell with "
            "the same day of week, so a leave-one-out oracle is undefined")
    n_series = df.select("series_id").distinct().count()
    scored.write.mode("overwrite").parquet(str(out_dir))

    requested = n_series * horizon
    disposition = {"emitted": n_emitted}
    if requested > n_emitted:
        disposition["declined_ineligible_or_no_partner"] = requested - n_emitted
    return Forecast(adapter=ORACLE, path=str(out_dir), origin=origin, horizon=horizon,
                    n_series=n_series, disposition=disposition,
                    target_estimand=panel.target_estimand,
                    frequency=panel.frequency).validate()


def estimate(
    panel: Panel,
    best_adapter: str,
    best_score: Score,
    origins: tuple[str, ...],
    horizon: int,
    spark: SparkSession,
    out_root: pathlib.Path,
    *,
    missing_actual: MissingActual,
    restrict: dict[str, str] | None = None,
    eligible_grids: dict[str, str] | None = None,
    eligible_grid_hashes: dict[str, str] | None = None,
    missing_forecast_policy: MissingForecastPolicy = "error",
) -> Headroom:
    """Estimate headroom above `best_score` by running the oracle over the same origins and cells.

    Args:
        panel: Data.
        best_adapter: Name of the model being measured against.
        best_score: Its pooled score, over exactly these origins and cells.
        origins: The origins `best_score` was computed on.
        horizon: Periods per origin.
        spark: Active session.
        out_root: Where the oracle's forecasts go.
        missing_actual: Passed to `score_forecast`; no default there, and none here.
        restrict: origin -> common-cells Parquet, so the oracle is scored on the SAME cells the
            model was. Without it the oracle would be scored on its own coverage and the two
            numbers would not be a comparison.
        eligible_grids: origin -> canonical eligible-grid Parquet, required wherever an absent
            panel row is to be classified as a sparse zero.
        eligible_grid_hashes: Frozen semantic hashes for those grids. Required with ``zero``
            fallback so the oracle's finalized artifact has the same cache/provenance identity as
            the model artifacts it bounds.
        missing_forecast_policy: The manifest policy applied to the oracle too. A headroom oracle
            that silently receives a different omission policy is not a ceiling on the model's
            scored population.

    Returns:
        A `Headroom`.

    Raises:
        HeadroomError: the oracle and the model were not scored on the same number of cells,
            which would make the difference between their MAEs partly a difference of population.
    """
    if missing_forecast_policy not in ("error", "zero"):
        raise HeadroomError(
            "missing_forecast_policy must be 'error' or 'zero', got "
            f"{missing_forecast_policy!r}")
    if missing_forecast_policy == "zero" and (
            eligible_grids is None or eligible_grid_hashes is None):
        raise HeadroomError(
            "zero missing-forecast policy requires eligible grids and their frozen hashes for "
            "every oracle origin")
    parts = []
    for origin in origins:
        raw = oracle_forecast(
            panel, origin, horizon, spark=spark, out_dir=str(out_root / ORACLE / origin))
        if missing_forecast_policy == "zero":
            assert eligible_grids is not None and eligible_grid_hashes is not None
            try:
                raw, _ = finalize_forecast_on_grid(
                    raw, eligible_grids[origin], eligible_grid_hashes[origin], spark,
                    out_root / ORACLE / f"{origin}.canonical",
                    missing_forecast_policy=missing_forecast_policy)
            except KeyError as exc:
                raise HeadroomError(
                    f"zero missing-forecast policy has no grid/hash for oracle origin "
                    f"{origin}") from exc
        parts.append(score_forecast(
            panel, raw, spark, missing_actual=missing_actual,
            restrict_to=(restrict or {}).get(origin),
            eligible_grid=(eligible_grids or {}).get(origin)))
    oracle = Score.combine(parts)
    if oracle.n_scored != best_score.n_scored:
        raise HeadroomError(
            f"the oracle scored {oracle.n_scored} cells and {best_adapter} scored "
            f"{best_score.n_scored}. Comparing their MAEs would compare two populations as well "
            "as two predictors -- pass `restrict` so both are scored on the common cells")
    return Headroom(best_adapter=best_adapter, best_mae=best_score.mae,
                    oracle_mae=oracle.mae, n_scored=oracle.n_scored)


__all__ = ["ORACLE", "Headroom", "HeadroomError", "estimate", "oracle_forecast"]
