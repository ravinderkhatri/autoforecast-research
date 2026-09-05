"""Tests for lead-time calibration and forecast stability.

Both exist because accuracy alone misses them. A model can be more accurate and still be rejected
by operations for churning orders, and it can score well on point error while its intervals are
the wrong width -- which nobody notices until the service level is not met, or until someone asks
why so much stock is being held.

Calibration is checked against a KNOWN distribution, because that is the only way to know what the
right answer is: paths drawn from a Poisson must cover their own quantiles.

Run: python3 autoforecast/test_calibration.py
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import torch
from pyspark.sql import SparkSession

from autoforecast import grid
from autoforecast.calibration import (
    Calibration,
    CalibrationError,
    coverage,
    reliability_table,
)
from autoforecast.panel import Forecast
from autoforecast.stability import Stability, StabilityError, compare, measure
from provenance.lineage import directory_digest

N_PATHS, N_CELLS, HORIZON = 800, 500, 28

_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    """One small local session for the adversarial evidence-artifact tests."""
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME is required for Spark stability tests")
        _SPARK = (SparkSession.builder.master("local[2]").appName("test_calibration_stability")
                  .config("spark.sql.shuffle.partitions", "2")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def poisson_paths(rate: float = 3.0, seed: int = 0):
    """Paths and actuals drawn from the SAME distribution. A correctly calibrated forecast is
    exactly this, so the coverage must land on its nominal levels."""
    g = torch.Generator().manual_seed(seed)
    paths = torch.poisson(torch.full((N_PATHS, N_CELLS, HORIZON), rate), generator=g)
    actual = torch.poisson(torch.full((N_CELLS, HORIZON), rate), generator=g)
    return paths, actual


def test_paths_from_the_right_distribution_cover_their_nominal_levels() -> None:
    """The only way to know the right answer is to draw both from the same distribution."""
    paths, actual = poisson_paths()
    cal = coverage(paths, actual, lead_time=5, review_period=2)
    assert cal.is_calibrated(tolerance=0.05), str(cal)
    assert cal.worst_gap < 0.05


def test_coverage_randomises_the_boundary_because_demand_is_integer() -> None:
    """Found by a test that expected 50% and got 56%. The forecast quantile is an ATOM with real
    probability mass, so `P(Y <= q)` overshoots by up to that mass -- discreteness, not
    miscalibration. Gating on the plain figure would reject a perfectly calibrated forecast for
    being made of whole units."""
    paths, actual = poisson_paths()
    cal = coverage(paths, actual, lead_time=5, review_period=2)
    plain_gap = max(abs(n - e) for n, e in zip(cal.nominal, cal.empirical_plain, strict=True))
    assert plain_gap > cal.worst_gap, (
        f"plain coverage gap {plain_gap:.3f} did not exceed randomised {cal.worst_gap:.3f}; if they "
        "agree, the distribution is not discrete and this test proves nothing")
    # The plain figure is still reported, because it is what an ordering policy experiences.
    assert all(p >= m for p, m in zip(cal.empirical_plain, cal.empirical, strict=True))


def test_paths_that_are_too_narrow_are_caught_and_named() -> None:
    """Under-covering means stockouts at a service level the business believes it is buying."""
    paths, actual = poisson_paths(rate=3.0)
    narrow = paths.mean(dim=0, keepdim=True).round().expand_as(paths).clone()  # zero spread
    cal = coverage(narrow, actual, lead_time=5, review_period=2)
    assert not cal.is_calibrated()
    assert any("TOO NARROW" in n for n in cal.notes), cal.notes


def test_paths_that_are_too_wide_are_caught_and_named_as_the_quieter_failure() -> None:
    """Over-covering never generates a complaint, which is why it survives -- it is capital and
    shelf space spent on a level nobody asked for."""
    paths, actual = poisson_paths(rate=3.0)
    wide = paths * 3.0
    cal = coverage(wide, actual, lead_time=5, review_period=2)
    assert not cal.is_calibrated()
    assert any("TOO WIDE" in n for n in cal.notes), cal.notes


def test_summing_per_day_quantiles_over_orders_and_the_gap_is_measured() -> None:
    """On this positively dependent fixture, adding daily 95th percentiles over-orders. This is
    one failure direction; a different dependence structure can reverse it."""
    paths, actual = poisson_paths()
    cal = coverage(paths, actual, lead_time=6, review_period=1)
    assert cal.naive_sum_ratio > 1.0, \
        "this fixture no longer demonstrates marginal-quantile over-ordering"
    assert any("over-ordering" in n for n in cal.notes) or cal.naive_sum_ratio <= 1.05


def test_summed_marginal_quantiles_can_under_order_too() -> None:
    """Exactly one of 100 days carries one unit, rotating by path. Every daily 95th percentile
    is zero while total demand is always one -- a counterexample to the old universal claim."""
    paths = torch.zeros(100, 1, 100)
    for path in range(100):
        paths[path, 0, path] = 1
    actual = torch.zeros(1, 100)
    actual[0, 0] = 1
    cal = coverage(paths, actual, lead_time=100, review_period=0, nominal=(0.95,))
    assert cal.naive_sum_ratio == 0.0
    assert any("under-ordering" in note for note in cal.notes)


def test_a_cover_period_past_the_horizon_is_refused() -> None:
    """Extending it would invent dependence past the last day the factor was drawn for."""
    paths, actual = poisson_paths()
    try:
        coverage(paths, actual, lead_time=20, review_period=20)
    except CalibrationError as e:
        assert "invent dependence" in str(e); return
    raise AssertionError("computed coverage past the horizon that was drawn")


def test_mismatched_actuals_are_refused() -> None:
    paths, _ = poisson_paths()
    try:
        coverage(paths, torch.zeros(3, HORIZON), lead_time=2, review_period=1)
    except CalibrationError as e:
        assert "against paths" in str(e); return
    raise AssertionError("measured coverage against actuals of the wrong shape")


def test_a_nominal_level_outside_the_unit_interval_is_refused() -> None:
    paths, actual = poisson_paths()
    for levels in ((0.5, 1.0), ()):
        try:
            coverage(paths, actual, lead_time=2, review_period=1, nominal=levels)
        except CalibrationError as e:
            assert "(0,1)" in str(e)
            continue
        raise AssertionError(f"accepted invalid nominal levels {levels}")


def test_discrete_quantile_is_an_observed_order_statistic_not_an_interpolation() -> None:
    """For paths {0,1}, q=.25 lies halfway through the atom at observed boundary zero."""
    paths = torch.tensor([[[0.0]], [[1.0]]])
    actual = torch.tensor([[0.0]])
    cal = coverage(paths, actual, lead_time=1, review_period=0, nominal=(0.25,))
    assert cal.empirical == (0.5,)
    assert cal.empirical_plain == (1.0,)


def test_a_perfect_point_mass_is_calibrated_at_every_requested_level() -> None:
    """A fixed half-boundary falsely calls this perfect distribution too narrow above 50%."""
    paths = torch.zeros(1000, 40, 7)
    actual = torch.zeros(40, 7)
    levels = (0.5, 0.8, 0.95, 0.99)
    cal = coverage(paths, actual, lead_time=5, review_period=2, nominal=levels)
    assert torch.allclose(torch.tensor(cal.empirical), torch.tensor(levels))
    assert cal.worst_gap < 1e-12 and cal.is_calibrated(), cal
    assert cal.naive_sum_ratio == 1.0


def test_calibration_refuses_invalid_path_counts() -> None:
    actual = torch.zeros(1, 1)
    invalid = [
        (torch.tensor([[[0.5]]]), "integer"),
        (torch.tensor([[[-1.0]]]), "non-negative"),
        (torch.tensor([[[float("nan")]]]), "finite"),
        (torch.tensor([[[float("inf")]]]), "finite"),
    ]
    for paths, fragment in invalid:
        try:
            coverage(paths, actual, lead_time=1, review_period=0, nominal=(0.5,))
        except CalibrationError as e:
            assert fragment in str(e), (paths, e)
            continue
        raise AssertionError(f"accepted invalid paths {paths}")


def test_calibration_refuses_invalid_actual_counts() -> None:
    paths = torch.tensor([[[0.0]], [[1.0]]])
    for value, fragment in ((-1.0, "non-negative"), (0.5, "integer"),
                            (float("nan"), "finite"), (float("inf"), "finite")):
        try:
            coverage(paths, torch.tensor([[value]]), lead_time=1, review_period=0,
                     nominal=(0.5,))
        except CalibrationError as e:
            assert fragment in str(e), (value, e)
            continue
        raise AssertionError(f"accepted invalid actual {value!r}")


def test_the_reliability_table_checks_the_shape_not_one_level() -> None:
    """A model well calibrated at 95% and badly at 50% has a distribution of the wrong SHAPE, and
    only the ordering policy at 95% would ever notice -- until the policy changes."""
    paths, actual = poisson_paths()
    table = reliability_table(paths, actual, lead_time=5, review_period=2, bins=9)
    assert len(table) == 9
    assert all(abs(n - e) < 0.06 for n, e, _ in table), table
    assert [n for n, _, _ in table] == sorted(n for n, _, _ in table)


def test_a_reliability_table_needs_enough_bins_to_describe_a_curve() -> None:
    paths, actual = poisson_paths()
    try:
        reliability_table(paths, actual, lead_time=2, review_period=1, bins=1)
    except CalibrationError as e:
        assert "at least 2 bins" in str(e); return
    raise AssertionError("described a curve with one point")


def test_a_calibration_over_no_cells_is_refused() -> None:
    try:
        Calibration(cover_days=7, nominal=(0.5,), empirical=(0.5,), empirical_plain=(0.5,),
                    n_cells=0, naive_sum_ratio=1.0)
    except CalibrationError as e:
        assert "not one" in str(e); return
    raise AssertionError("reported coverage over nothing")


def test_is_calibrated_refuses_invalid_tolerances() -> None:
    cal = Calibration(
        cover_days=1,
        nominal=(0.5,),
        empirical=(0.5,),
        empirical_plain=(0.5,),
        n_cells=1,
        naive_sum_ratio=1.0,
    )

    class NumericAlias(float):
        pass

    for invalid in (True, -0.01, float("nan"), float("inf"), 10**400,
                    NumericAlias(0.05), "0.05"):
        try:
            cal.is_calibrated(tolerance=invalid)
        except CalibrationError as exc:
            assert "tolerance" in str(exc)
            continue
        raise AssertionError(f"accepted invalid calibration tolerance {invalid!r}")

    assert cal.is_calibrated(tolerance=0)


def test_calibration_record_refuses_non_finite_or_out_of_domain_evidence() -> None:
    bad = (
        dict(nominal=(float("nan"),), empirical=(0.5,), empirical_plain=(0.5,),
             naive_sum_ratio=1.0),
        dict(nominal=(10**400,), empirical=(0.5,), empirical_plain=(0.5,),
             naive_sum_ratio=1.0),
        dict(nominal=(0.5,), empirical=(float("inf"),), empirical_plain=(0.5,),
             naive_sum_ratio=1.0),
        dict(nominal=(0.5,), empirical=(0.5,), empirical_plain=(1.1,),
             naive_sum_ratio=1.0),
        dict(nominal=(0.5,), empirical=(0.5,), empirical_plain=(0.5,),
             naive_sum_ratio=float("nan")),
    )
    for fields in bad:
        try:
            Calibration(cover_days=7, n_cells=1, **fields)
        except CalibrationError:
            continue
        raise AssertionError(f"accepted malformed calibration evidence: {fields!r}")


def test_cumulative_count_overflow_is_refused() -> None:
    paths = torch.tensor([[[1e308, 1e308]]], dtype=torch.float64)
    actual = torch.zeros((1, 2), dtype=torch.float64)
    try:
        coverage(paths, actual, lead_time=1, review_period=1, nominal=(0.5,))
    except CalibrationError as exc:
        assert "overflowed" in str(exc)
        return
    raise AssertionError("published calibration after cumulative demand overflowed")


# ---------------------------------------------------------------- stability

def _forecast_artifact(
    root: pathlib.Path,
    name: str,
    origin: str,
    horizon: int,
    rows: list[tuple[str, dt.date, float | None]],
    frequency: str = "D",
) -> Forecast:
    path = root / name
    spark().createDataFrame(rows, "series_id string, date date, yhat double") \
        .write.mode("overwrite").parquet(str(path))
    declined = horizon - len(rows)
    disposition = {"emitted": len(rows)}
    if declined:
        disposition["declined_fixture"] = declined
    return Forecast(adapter="candidate", path=str(path), origin=origin, horizon=horizon,
                    n_series=1, disposition=disposition, frequency=frequency)


def _overlap_artifact(root: pathlib.Path, days: list[dt.date]) -> tuple[pathlib.Path, str]:
    path = root / "overlap"
    spark().createDataFrame([("s1", day) for day in days], "series_id string, date date") \
        .write.mode("overwrite").parquet(str(path))
    return path, grid.content_hash(path, spark())


def test_stability_measures_the_exact_registered_overlap() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    earlier = _forecast_artifact(root, "earlier", "2024-01-01", 3, [
        ("s1", dt.date(2024, 1, 2), 1.0),
        ("s1", dt.date(2024, 1, 3), 2.0),
        ("s1", dt.date(2024, 1, 4), 4.0),
    ])
    later = _forecast_artifact(root, "later", "2024-01-02", 3, [
        ("s1", dt.date(2024, 1, 3), 3.0),
        ("s1", dt.date(2024, 1, 4), 2.0),
        ("s1", dt.date(2024, 1, 5), 9.0),
    ])
    overlap, digest = _overlap_artifact(
        root, [dt.date(2024, 1, 3), dt.date(2024, 1, 4)])
    out = measure(earlier, later, spark(), expected_overlap=overlap,
                  expected_overlap_hash=digest)
    assert out.n_shared_cells == 2
    assert abs(out.mean_absolute_revision - 1.5) < 1e-12


def test_stability_keeps_signed_forecasts_and_scales_by_baseline_magnitude() -> None:
    tmp_path = pathlib.Path(tempfile.mkdtemp())
    earlier = _forecast_artifact(tmp_path, "earlier", "2024-01-01", 5, [
        ("s1", dt.date(2024, 1, 2), -1.0),
        ("s1", dt.date(2024, 1, 3), -2.0),
        ("s1", dt.date(2024, 1, 4), -6.7e-15),
        ("s1", dt.date(2024, 1, 5), 3.0),
        ("s1", dt.date(2024, 1, 6), 0.0),
    ])
    later = _forecast_artifact(tmp_path, "later", "2024-01-02", 5, [
        ("s1", dt.date(2024, 1, 3), -3.0),
        ("s1", dt.date(2024, 1, 4), 0.0),
        ("s1", dt.date(2024, 1, 5), -3.0),
        ("s1", dt.date(2024, 1, 6), -1.0),
        ("s1", dt.date(2024, 1, 7), 9.0),
    ])
    overlap, digest = _overlap_artifact(
        tmp_path, [dt.date(2024, 1, day) for day in range(3, 7)])
    before = [directory_digest(f.path) for f in (earlier, later)]

    result = measure(earlier, later, spark(), expected_overlap=overlap,
                     expected_overlap_hash=digest)

    # Shared-cell absolute changes: 1, 6.7e-15, 6, 1.
    # Relative changes use baseline magnitudes: 1/2, 6.7e-15/1, 6/3, 1/1.
    assert result.n_shared_cells == 4
    assert result.mean_absolute_revision == pytest.approx((8 + 6.7e-15) / 4)
    assert result.mean_relative_revision == pytest.approx((3.5 + 6.7e-15) / 4)
    assert result.p95_relative_revision == pytest.approx(2.0)
    assert result.share_moving_over_20pct == pytest.approx(0.75)
    assert [directory_digest(f.path) for f in (earlier, later)] == before


def test_stability_uses_the_declared_period_grid_and_refuses_frequency_mismatch() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    earlier = _forecast_artifact(root, "weekly-earlier", "2024-01-01", 3, [
        ("s1", dt.date(2024, 1, 8), 1.0),
        ("s1", dt.date(2024, 1, 15), 2.0),
        ("s1", dt.date(2024, 1, 22), 4.0),
    ], frequency="W")
    later = _forecast_artifact(root, "weekly-later", "2024-01-08", 3, [
        ("s1", dt.date(2024, 1, 15), 3.0),
        ("s1", dt.date(2024, 1, 22), 2.0),
        ("s1", dt.date(2024, 1, 29), 9.0),
    ], frequency="W")
    overlap, digest = _overlap_artifact(
        root, [dt.date(2024, 1, 15), dt.date(2024, 1, 22)])
    result = measure(
        earlier, later, spark(), expected_overlap=overlap, expected_overlap_hash=digest)
    assert result.n_shared_cells == 2

    with pytest.raises(StabilityError, match="frequency"):
        measure(
            earlier,
            dataclasses.replace(later, frequency="7D"),
            spark(),
            expected_overlap=overlap,
            expected_overlap_hash=digest,
        )


def test_stability_preserves_hourly_overlap_across_midnight() -> None:
    root = pathlib.Path(tempfile.mkdtemp())

    def hourly_forecast(name: str, origin: str, rows: list[tuple[str, str, float]]) -> Forecast:
        path = root / name
        (
            spark().createDataFrame(
                rows, "series_id string, raw_date string, yhat double"
            )
            .selectExpr("series_id", "cast(raw_date as timestamp_ntz) as date", "yhat")
            .write.parquet(str(path))
        )
        return Forecast(
            "candidate", path, origin, 3, 1, {"emitted": 3}, frequency="H"
        )

    earlier = hourly_forecast(
        "hourly-earlier",
        "2024-03-01T20:00:00",
        [
            ("s1", "2024-03-01T21:00:00", 1.0),
            ("s1", "2024-03-01T22:00:00", 2.0),
            ("s1", "2024-03-01T23:00:00", 4.0),
        ],
    )
    later = hourly_forecast(
        "hourly-later",
        "2024-03-01T21:00:00",
        [
            ("s1", "2024-03-01T22:00:00", 3.0),
            ("s1", "2024-03-01T23:00:00", 2.0),
            ("s1", "2024-03-02T00:00:00", 9.0),
        ],
    )
    overlap = root / "hourly-overlap"
    (
        spark().createDataFrame(
            [
                ("s1", "2024-03-01T22:00:00"),
                ("s1", "2024-03-01T23:00:00"),
            ],
            "series_id string, raw_date string",
        )
        .selectExpr("series_id", "cast(raw_date as timestamp_ntz) as date")
        .write.parquet(str(overlap))
    )
    digest = grid.content_hash(overlap, spark(), frequency="H")

    result = measure(
        earlier, later, spark(), expected_overlap=overlap, expected_overlap_hash=digest
    )

    assert result.n_shared_cells == 2
    assert result.mean_absolute_revision == pytest.approx(1.5)


def test_stability_refuses_a_selectively_missing_or_extra_overlap_cell() -> None:
    """The former inner join silently dropped either discrepancy and improved the metric."""
    root = pathlib.Path(tempfile.mkdtemp())
    earlier = _forecast_artifact(root, "earlier", "2024-01-01", 3, [
        ("s1", dt.date(2024, 1, 2), 1.0),
        ("s1", dt.date(2024, 1, 3), 2.0),
        ("s1", dt.date(2024, 1, 4), 4.0),
    ])
    missing_later = _forecast_artifact(root, "later_missing", "2024-01-02", 3, [
        ("s1", dt.date(2024, 1, 3), 3.0),
        ("s1", dt.date(2024, 1, 5), 9.0),
    ])
    full_later = _forecast_artifact(root, "later_full", "2024-01-02", 3, [
        ("s1", dt.date(2024, 1, 3), 3.0),
        ("s1", dt.date(2024, 1, 4), 2.0),
        ("s1", dt.date(2024, 1, 5), 9.0),
    ])
    full_overlap, full_hash = _overlap_artifact(
        root, [dt.date(2024, 1, 3), dt.date(2024, 1, 4)])
    try:
        measure(earlier, missing_later, spark(), expected_overlap=full_overlap,
                expected_overlap_hash=full_hash)
    except StabilityError as exc:
        assert "1 expected key(s) missing" in str(exc)
    else:
        raise AssertionError("stability silently dropped a selectively missing high-revision cell")

    one_cell = root / "one_cell_overlap"
    spark().createDataFrame(
        [("s1", dt.date(2024, 1, 3))], "series_id string, date date") \
        .write.mode("overwrite").parquet(str(one_cell))
    try:
        measure(earlier, full_later, spark(), expected_overlap=one_cell,
                expected_overlap_hash=grid.content_hash(one_cell, spark()))
    except StabilityError as exc:
        assert "1 unregistered key(s) present" in str(exc)
    else:
        raise AssertionError("stability accepted an overlap larger than the registered cohort")


def test_stability_revalidates_duplicate_date_value_and_schema_artifacts() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    good_earlier = _forecast_artifact(root, "good_earlier", "2024-01-01", 3, [
        ("s1", dt.date(2024, 1, 2), 1.0),
        ("s1", dt.date(2024, 1, 3), 2.0),
        ("s1", dt.date(2024, 1, 4), 4.0),
    ])
    good_later = _forecast_artifact(root, "good_later", "2024-01-02", 3, [
        ("s1", dt.date(2024, 1, 3), 3.0),
        ("s1", dt.date(2024, 1, 4), 2.0),
        ("s1", dt.date(2024, 1, 5), 9.0),
    ])
    overlap, digest = _overlap_artifact(
        root, [dt.date(2024, 1, 3), dt.date(2024, 1, 4)])
    malformed = [
        ("duplicate", [
            ("s1", dt.date(2024, 1, 3), 3.0),
            ("s1", dt.date(2024, 1, 3), 4.0),
            ("s1", dt.date(2024, 1, 5), 9.0),
        ], "duplicate"),
        ("bad_date", [
            ("s1", dt.date(2024, 1, 3), 3.0),
            ("s1", dt.date(2024, 1, 4), 2.0),
            ("s1", dt.date(2024, 1, 6), 9.0),
        ], "outside"),
        ("negative_infinity", [
            ("s1", dt.date(2024, 1, 3), 3.0),
            ("s1", dt.date(2024, 1, 4), float("-inf")),
            ("s1", dt.date(2024, 1, 5), 9.0),
        ], "finite"),
        ("nan", [
            ("s1", dt.date(2024, 1, 3), 3.0),
            ("s1", dt.date(2024, 1, 4), float("nan")),
            ("s1", dt.date(2024, 1, 5), 9.0),
        ], "finite"),
    ]
    for name, rows, fragment in malformed:
        bad = _forecast_artifact(root, name, "2024-01-02", 3, rows)
        try:
            measure(good_earlier, bad, spark(), expected_overlap=overlap,
                    expected_overlap_hash=digest)
        except StabilityError as exc:
            assert fragment in str(exc), (name, exc)
            continue
        raise AssertionError(f"stability accepted malformed forecast {name}")

    missing_schema_path = root / "missing_schema"
    spark().createDataFrame(
        [("s1", dt.date(2024, 1, 3), 1.0)], "series_id string, date date, wrong double") \
        .write.mode("overwrite").parquet(str(missing_schema_path))
    missing_schema = Forecast(
        adapter="candidate", path=str(missing_schema_path), origin="2024-01-02", horizon=3,
        n_series=1, disposition={"emitted": 1, "declined_fixture": 2})
    try:
        measure(good_earlier, missing_schema, spark(), expected_overlap=overlap,
                expected_overlap_hash=digest)
    except StabilityError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("stability accepted a forecast without yhat")

    try:
        measure(good_earlier, good_later, spark(), expected_overlap=overlap,
                expected_overlap_hash="0" * 64)
    except StabilityError as exc:
        assert "changed after registration" in str(exc)
    else:
        raise AssertionError("stability trusted a mutable overlap path without its registered hash")


def stab(name: str, rel: float) -> Stability:
    return Stability(adapter=name, n_shared_cells=1000, mean_absolute_revision=rel * 2,
                     mean_relative_revision=rel, p95_relative_revision=rel * 3,
                     share_moving_over_20pct=rel)


def test_compare_ranks_by_revision_and_names_who_clears_the_bound() -> None:
    """A model 5% more accurate that swings 40% week to week generates order churn, and
    operations will reject it whatever its MASE."""
    out = compare({"churny": stab("churny", 0.40), "steady": stab("steady", 0.05),
                   "middling": stab("middling", 0.12)}, bound=0.15)
    assert out["most_stable"] == "steady"
    assert out["order"] == ["steady", "middling", "churny"]
    assert out["within_bound"] == ["steady", "middling"]
    assert out["outside_bound"] == ["churny"]
    assert "order churn" in out["note"]


def test_a_stability_bound_must_be_positive_and_representable() -> None:
    for bound in (0.0, 10**400):
        try:
            compare({"a": stab("a", 0.1)}, bound=bound)
        except StabilityError as e:
            assert "positive" in str(e)
            continue
        raise AssertionError(f"accepted invalid stability bound {bound!r}")


def test_a_stability_over_no_shared_cells_is_refused() -> None:
    """Comparing forecasts that do not overlap produces a number about nothing."""
    try:
        Stability(adapter="a", n_shared_cells=0, mean_absolute_revision=0.0,
                  mean_relative_revision=0.0, p95_relative_revision=0.0,
                  share_moving_over_20pct=0.0)
    except StabilityError as e:
        assert "number about nothing" in str(e); return
    raise AssertionError("reported stability between forecasts that share no cells")


def test_stability_evidence_refuses_blank_identity_boolean_counts_and_values() -> None:
    base = {
        "adapter": "candidate",
        "n_shared_cells": 1,
        "mean_absolute_revision": 0.1,
        "mean_relative_revision": 0.1,
        "p95_relative_revision": 0.2,
        "share_moving_over_20pct": 0.1,
    }
    for change in (
        {"adapter": ""},
        {"n_shared_cells": True},
        {"mean_absolute_revision": True},
        {"mean_relative_revision": "0.1"},
        {"p95_relative_revision": math.nan},
        {"p95_relative_revision": 10**400},
    ):
        try:
            Stability(**(base | change))
        except StabilityError:
            continue
        raise AssertionError(f"accepted malformed stability evidence {change!r}")


def test_comparing_nothing_is_refused() -> None:
    try:
        compare({}, bound=0.1)
    except StabilityError as e:
        assert "nothing to compare" in str(e); return
    raise AssertionError("ranked an empty set")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
