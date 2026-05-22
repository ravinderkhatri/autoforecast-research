"""M5 integration tests require separately supplied CSV or Parquet data.

Synthetic output-safety tests always run. Missing external data is reported as a skip.
Run with pytest from the repository root.
"""
import pathlib
import sys
from contextlib import contextmanager
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

M5_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "m5"
HAVE_M5 = all((M5_DIR / name).is_file() for name in (
    "calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv",
))
requires_m5 = pytest.mark.skipif(not HAVE_M5, reason="M5 CSV dataset is not installed")

_CACHE: dict = {}


def _long():
    if "long" not in _CACHE:
        from autoforecast.benchmarks.m5 import exposure_from_first_sale, load_sales_long
        _CACHE["long"] = exposure_from_first_sale(load_sales_long())
    return _CACHE["long"]


@requires_m5
def test_calendar_covers_every_day_column():
    """A day column with no calendar row would be dropped by the join, silently shortening
    history for whichever series happened to use it."""
    from autoforecast.benchmarks.m5 import load_calendar
    cal = load_calendar()
    assert len(cal) == 1969
    assert cal["date"].is_monotonic_increasing
    assert cal["d"].nunique() == len(cal), "a duplicated d_n would multiply rows on the join"


@requires_m5
def test_the_long_shape_is_one_row_per_store_product_day():
    """3,049 products x 10 stores x 1,941 days. If this is not exact, the melt lost or
    duplicated something."""
    long = _long()
    assert len(long) == 3049 * 10 * 1941 == 59_181_090


@requires_m5
def test_leading_zeros_are_a_fifth_of_the_grid():
    """Check the measured share of M5 cells preceding their first positive observation."""
    long = _long()
    ineligible = 1 - long["eligible"].mean()
    assert 0.20 < ineligible < 0.22, f"expected ~20.9% ineligible, measured {ineligible:.1%}"


@requires_m5
def test_applying_eligibility_moves_the_zero_rate_by_a_measurable_amount():
    """68.0% of all cells are zero; 59.5% of ELIGIBLE cells are. The 8.5-point gap is exactly the
    quantity the exposure contract recovers, and it is why a naive zero rate overstates
    intermittency."""
    long = _long()
    naive = (long["qty"] == 0).mean()
    real = (long.loc[long["eligible"], "qty"] == 0).mean()
    assert 0.67 < naive < 0.69, f"naive zero rate {naive:.1%}"
    assert 0.58 < real < 0.61, f"eligible zero rate {real:.1%}"
    assert naive - real > 0.07, "the contract must change the number materially or it is decoration"


@requires_m5
def test_the_measured_zero_rate_justifies_a_hurdle_head():
    """`resolve()` steps down to a plain count head below a material zero share. M5 is well above
    it, so this dataset exercises the hurdle rather than the fallback."""
    assert (_long().loc[_long()["eligible"], "qty"] == 0).mean() > 0.30


@requires_m5
def test_a_products_first_sale_is_never_after_its_last_eligible_day():
    """Eligibility is derived FROM the first sale, so an eligible cell before it means the
    derivation inverted somewhere."""
    long = _long()
    bad = long[long["eligible"] & (long["date"] < long["first_sale"])]
    assert len(bad) == 0, f"{len(bad):,} eligible cells precede the first sale"


@requires_m5
def test_no_negative_quantities():
    """M5 has no negative quantities and does not exercise the returns path."""
    assert (_long()["qty"] < 0).sum() == 0


@requires_m5
def test_first_sale_date_is_the_earliest_day_with_a_positive_quantity():
    """On REAL M5. `first_sale_date` is the only thing standing in for a range-eligibility feed
    here, so every eligibility figure this repository quotes about M5 rests on it. It was reached
    only through `exposure_from_first_sale` until the direct-test audit was widened to cover
    autoforecast/ -- exactly the composition-not-layer gap A1 exists to close."""
    from autoforecast.benchmarks.m5 import first_sale_date
    long = _long()
    fs = first_sale_date(long)
    assert list(fs.columns) == ["store_id", "item_id", "first_sale"]
    # One row per (store, product) that ever sold -- never more, or the merge would fan out.
    assert len(fs) == len(fs.drop_duplicates(["store_id", "item_id"]))
    # Check real pairs end to end against the raw rows, rather than against a reimplementation.
    # Pick a pair that HAS leading zeros: for a product that sold on day one the "nothing before
    # the first sale" check is vacuous, and an earlier version of this test asserted
    # `max() == 0` on that empty selection -- which is NaN, so it failed while proving nothing.
    with_leading = fs[fs["first_sale"] > long["date"].min()]
    assert not with_leading.empty, "no product has leading zeros, so M5 is not what we think"
    for _, row in with_leading.head(3).iterrows():
        pair = long[(long["store_id"] == row["store_id"]) & (long["item_id"] == row["item_id"])]
        before = pair[pair["date"] < row["first_sale"]]
        assert not before.empty, "picked a pair with no history before its first sale"
        assert (before["qty"] > 0).sum() == 0, \
            "a positive quantity exists before the reported first sale"
        on_day = pair[pair["date"] == row["first_sale"]]["qty"]
        assert len(on_day) == 1 and on_day.iloc[0] > 0, \
            "the reported first-sale day did not sell anything"


@requires_m5
def test_load_prices_covers_the_weeks_products_were_actually_sold():
    """On REAL M5. WRMSSE weights every series by its share of DOLLAR sales, so this file is what
    the metric's weights rest on -- and a silently missing price becomes a silently zero weight,
    which drops a series from the score without dropping it from the data."""
    from autoforecast.benchmarks.m5 import load_prices
    prices = load_prices()
    assert list(prices.columns) == ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    assert len(prices) == len(prices.drop_duplicates(["store_id", "item_id", "wm_yr_wk"])), \
        "duplicate (store, item, week) would fan the sales rows out on the merge"
    assert prices["sell_price"].min() > 0, "a zero or negative price would zero a series' weight"
    # Every week a product actually sold in a store must have a price, or revenue is null there.
    long = _long()
    sold = long[long["qty"] > 0][["store_id", "item_id", "wm_yr_wk"]].drop_duplicates()
    merged = sold.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    assert merged["sell_price"].isna().sum() == 0, \
        f"{merged['sell_price'].isna().sum()} (store, item, week) combinations sold with no price"


def test_build_parquet_returns_a_path_and_partitions_by_store():
    """The loader publishes a Parquet path partitioned by store for downstream processing."""
    out = M5_DIR.parent / "m5_parquet"
    if not out.exists():
        pytest.skip("M5 Parquet dataset is not installed")
    assert isinstance(out, pathlib.Path)
    partitions = sorted(out.glob("store_id=*"))
    assert len(partitions) == 10, "partitioned by store, as the estate is"


def test_one_parquet_fragment_per_partition_so_rows_are_not_duplicated():
    """`to_parquet(partition_cols=...)` APPENDS. Until build_parquet cleared its output, a second
    build left the first build's fragments beside the new ones -- every row present twice, and
    two date types in one directory. Spark refused to read it, which is the only reason it was
    caught; a rebuild with an unchanged schema would have doubled the data silently."""
    out = M5_DIR.parent / "m5_parquet"
    if not out.exists():
        pytest.skip("M5 Parquet dataset is not installed")
    multi = {p.name: len(list(p.glob("*.parquet"))) for p in sorted(out.glob("store_id=*"))}
    extra = {k: v for k, v in multi.items() if v != 1}
    assert not extra, f"partitions holding more than one fragment: {extra}"


def test_dates_are_written_as_dates_not_nanosecond_timestamps():
    """Spark 4 refuses `TIMESTAMP(NANOS)` outright, and M5's grain is a day -- a nanosecond
    timestamp invites a spurious intraday component into every window downstream. This is the
    assertion that keeps the M5 path readable by the SAME engine the estate path uses."""
    out = M5_DIR.parent / "m5_parquet"
    if not out.exists():
        pytest.skip("M5 Parquet dataset is not installed")
    import pyarrow.parquet as pq
    frag = next(iter(sorted(out.glob("store_id=*/*.parquet"))))
    schema = pq.read_schema(frag)
    for col in ("date", "first_sale"):
        assert str(schema.field(col).type) == "date32[day]", \
            f"{col} is {schema.field(col).type}, which Spark cannot join on as a date"


def test_clear_previous_build_refuses_a_directory_it_did_not_write():
    """SYNTHETIC (structure): there is no real instance of a misconfigured output path, and the
    point is what the function REFUSES. A build step that deletes whatever it is pointed at is one
    misconfiguration away from taking a real dataset with it."""
    import tempfile

    from autoforecast.benchmarks.m5 import _clear_previous_build
    with tempfile.TemporaryDirectory() as d:
        target = pathlib.Path(d) / "out"
        target.mkdir()
        (target / "notes.md").write_text("someone else's file")
        try:
            _clear_previous_build(target)
        except ValueError as e:
            assert "did not write" in str(e)
            assert (target / "notes.md").exists(), "raised but deleted anyway"
            return
    raise AssertionError("deleted a directory holding a file it did not write")


def test_build_takes_the_panel_generation_lock_before_touching_output() -> None:
    """A long-lived reader must be able to exclude the destructive rebuild from its first byte."""
    from autoforecast.benchmarks import m5

    target = pathlib.Path("/unused/test-m5-panel")

    @contextmanager
    def refuse(path, *, exclusive, blocking=False):
        assert path == target and exclusive is True and blocking is False
        raise RuntimeError("held by reader")
        yield  # pragma: no cover - makes this a contextmanager without touching the target

    with mock.patch.object(m5, "panel_generation_lock", side_effect=refuse), \
            mock.patch.object(m5, "_clear_previous_build") as clear:
        try:
            m5.build_parquet(target)
        except RuntimeError as exc:
            assert str(exc) == "held by reader"
        else:
            raise AssertionError("build touched its output without the generation lock")
    clear.assert_not_called()


def test_panel_generation_lock_path_canonicalizes_symlink_aliases() -> None:
    """One panel inode must have one lock even when a consumer uses a different path spelling."""
    import tempfile

    from autoforecast.benchmarks import m5

    root = pathlib.Path(tempfile.mkdtemp())
    target = root / "m5_parquet"
    target.mkdir()
    alias = root / "panel_alias"
    alias.symlink_to(target, target_is_directory=True)

    assert m5.panel_generation_lock_path(target) == m5.panel_generation_lock_path(alias)
    with m5.panel_generation_lock(target, exclusive=True):
        try:
            with m5.panel_generation_lock(alias, exclusive=False):
                raise AssertionError("symlink alias acquired a second panel-generation lock")
        except RuntimeError as exc:
            assert "being rebuilt" in str(exc)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-rs"]))
