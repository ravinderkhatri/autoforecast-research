"""Adapt M5 CSVs to a daily panel and publish partitioned Parquet. Calendar rows map d_n columns to
dates and weekly prices to daily observations. The source-specific pandas conversion returns a
Parquet path for downstream processing. Exposure from first sale marks leading assortment zeros
separately from observed demand zeros.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import pathlib
import shutil
from contextlib import contextmanager

import pandas as pd

M5_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "m5"


def panel_generation_lock_path(out_dir: pathlib.Path) -> pathlib.Path:
    """Stable inode shared by the M5 panel producer and encoder-ladder consumer."""
    path = pathlib.Path(out_dir).resolve(strict=False)
    return path.parent / f".{path.name}.panel.lock"


@contextmanager
def panel_generation_lock(
    out_dir: pathlib.Path, *, exclusive: bool, blocking: bool = False,
):
    """Hold one M5 panel generation stable across a build or a complete consumer command.

    The builder takes an exclusive lock; ladder scoring/training/verification take a shared lock.
    The lock file is deliberately adjacent to, rather than inside, ``out_dir`` because a
    successful build replaces that directory. It is never unlinked, so a waiting process cannot
    lock a new inode while another process still owns the old one.
    """
    lock_path = panel_generation_lock_path(out_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError:
            role = "being rebuilt" if not exclusive else "in use by a forecast command"
            raise RuntimeError(
                f"M5 panel {out_dir} is {role}; refusing a mixed input generation") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

M5_LEVELS: tuple[tuple[str, ...], ...] = (
    (),                              # 1  total
    ("state_id",),                   # 2
    ("store_id",),                   # 3
    ("cat_id",),                     # 4
    ("dept_id",),                    # 5
    ("state_id", "cat_id"),          # 6
    ("state_id", "dept_id"),         # 7
    ("store_id", "cat_id"),          # 8
    ("store_id", "dept_id"),         # 9
    ("item_id",),                    # 10
    ("item_id", "state_id"),         # 11
    ("item_id", "store_id"),         # 12
)
"""The twelve aggregation levels WRMSSE averages over, in the competition's published order.

They live HERE and not in `wrmsse.py` because they are a fact about M5's hierarchy, not part of
the metric's definition. Other datasets have their own hierarchy with the same formula.
`wrmsse.score` takes the levels as an argument.

Level 12 is store x item, which is this repository's forecasting grain; level 1 is the grand
total. A forecast can score well at level 12 and badly at level 1, which is exactly the
cross-store cancellation the honest metric was built to expose -- so the two are worth reading
together rather than only through their mean.
"""


def load_calendar(m5_dir: pathlib.Path = M5_DIR) -> pd.DataFrame:
    """`d_n` -> real date, plus the event and SNAP columns.

    SNAP is a US food-assistance disbursement schedule and is per state, so it is a known-future
    covariate rather than a holiday: it moves demand and its dates are published in advance.
    """
    cal = pd.read_csv(m5_dir / "calendar.csv", parse_dates=["date"])
    return cal


def build_revision_calendar(out_path: pathlib.Path, m5_dir: pathlib.Path = M5_DIR) -> pathlib.Path:
    """Materialize the released benchmark calendar only; never open a sales-label file.

    This is a development-only known-future declaration, not a live as-of plan feed.
    ``test_revision_calendar`` checks event/type/SNAP preservation and non-overwrite.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = pathlib.Path(out_path)
    calendar = load_calendar(m5_dir)
    columns = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    pieces = []
    for state in ("CA", "TX", "WI"):
        part = calendar[["date", *columns]].copy()
        dates = pd.Series(part["date"])
        part["date"] = dates.dt.date
        part["group"] = state
        part["snap"] = calendar[f"snap_{state}"].astype(float)
        pieces.append(part)
    table = pa.Table.from_pandas(pd.concat(pieces, ignore_index=True), preserve_index=False)
    contract = dict(time_col="date", group_col="group", group_column="state_id",
                    known_at_col=None, episode_col=None, preannounced=True, complete_grid=True,
                    event_pairs=[["event_name_1", "event_type_1"],
                                 ["event_name_2", "event_type_2"]],
                    known_future=["snap"], binary_columns=["snap"],
                    variants={"v1_corrected": {
                        "known_future": [], "events": [f"name:{name}" for name in (
                            "Christmas", "Easter", "LaborDay", "MemorialDay", "SuperBowl",
                            "Thanksgiving")]}})
    table = table.replace_schema_metadata({b"availability": b"released_benchmark_calendar_v1",
        b"revision_calendar_v1": json.dumps(contract, sort_keys=True).encode()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("xb") as handle:
        pq.write_table(table, handle)
    return out_path


def build_parquet(out_dir: pathlib.Path, m5_dir: pathlib.Path = M5_DIR,
                  evaluation: bool = True) -> pathlib.Path:
    """Write the long, eligibility-marked frame to Parquet and return the PATH.

    This is the plane boundary. Everything downstream reads Parquet exactly as it will read the
    estate's Parquet, so the M5 run exercises the real path rather than a pandas twin of it --
    which was the substance of the objection to using pandas here at all.

    Partitioned by store, because that is how the estate is partitioned and a partitioning that
    only appears in production is one nobody has tested.

    NOT IDEMPOTENT UNTIL NOW, and the failure was silent in the worst way: `to_parquet` with
    `partition_cols` APPENDS fragments, so a second build left the first build's files in place.
    Every row was then present twice, and after a schema change the two generations disagreed --
    which is how it was caught, by Spark refusing to read the directory. A duplicate-row dataset
    that still reads is the version of this bug that does not announce itself.
    """
    with panel_generation_lock(out_dir, exclusive=True):
        _clear_previous_build(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_panel(out_dir, exposure_from_first_sale(load_sales_long(m5_dir, evaluation)), m5_dir)
        return out_dir


def build_private_evaluation_parquet(
    out_dir: pathlib.Path,
    *,
    m5_dir: pathlib.Path = M5_DIR,
    private_labels_dir: pathlib.Path,
) -> pathlib.Path:
    """Build a clean M5 private-evaluation replay panel without leaking its labels into features.

    The regular Kaggle training archive ends at ``d_1941``. The organizer's post-competition
    archive separately provides ``sales_test_evaluation.csv`` (``d_1942..d_1969``) so a historical
    replay can be scored. Those rows are appended solely as held-out actuals. First-sale exposure
    is deliberately derived from the pre-origin training frame, so a private-label sale cannot
    make a future item look known at the forecast origin.
    """
    with panel_generation_lock(out_dir, exclusive=True):
        _clear_previous_build(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        history = load_sales_long(m5_dir, evaluation=True)
        private_actuals = load_private_evaluation_long(private_labels_dir, calendar_dir=m5_dir)
        frame = pd.concat((history, private_actuals), ignore_index=True)
        _write_panel(out_dir, exposure_from_first_sale(frame, first_sale_source=history), m5_dir)
        return out_dir


def build_feature_complete_forecast_parquet(
    out_dir: pathlib.Path,
    *,
    m5_dir: pathlib.Path = M5_DIR,
) -> pathlib.Path:
    """Build an M5 forecast panel with published future calendar and price features only.

    Standard M5 inputs contain sales through ``d_1941`` but publish calendar and sell-price
    values through ``d_1969``.  This output appends those 28 feature rows per series with a null
    target and ``unobserved`` status.  It is valid for an archived forecast run, but deliberately
    cannot yield a WRMSSE claim: use :func:`build_private_evaluation_parquet` when organizer
    labels are available.
    """
    with panel_generation_lock(out_dir, exclusive=True):
        _clear_previous_build(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        history = load_sales_long(m5_dir, evaluation=True)
        history_end = history["date"].max()
        future_days = load_calendar(m5_dir).loc[
            lambda frame: frame["date"] > history_end,
            ["d", "date", "wm_yr_wk"],
        ]
        if future_days.empty:
            raise ValueError("M5 calendar has no published days after the sales-history cutoff")
        identity = history[
            ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        ].drop_duplicates()
        future = identity.merge(future_days, how="cross")
        future["qty"] = pd.Series(pd.NA, index=future.index, dtype="Int64")
        frame = pd.concat((history, future), ignore_index=True)
        panel = exposure_from_first_sale(frame, first_sale_source=history)
        panel.loc[panel["date"] > history_end, "observation_state"] = "unobserved"
        _write_panel(out_dir, panel, m5_dir)
        return out_dir


def build_origin_masked_forecast_parquet(
    out_dir: pathlib.Path,
    *,
    origin: str,
    horizon: int,
    m5_dir: pathlib.Path = M5_DIR,
) -> pathlib.Path:
    """Build a training-only M5 panel whose targets end exactly at ``origin``.

    This is the data-plane boundary for a replay: calendar and price rows for the requested
    horizon remain available, while every sales target after ``origin`` is absent from the panel
    handed to the training worker.  The corresponding observed panel belongs to the score-only
    evaluator and must be supplied through a separate runtime input after forecasting finishes.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError("M5 masked forecast horizon must be a positive integer")
    try:
        origin_day = dt.date.fromisoformat(origin)
    except (TypeError, ValueError) as exc:
        raise ValueError("M5 masked forecast origin must be an ISO date") from exc
    if origin_day.isoformat() != origin:
        raise ValueError(
            f"M5 masked forecast origin must be canonical ISO YYYY-MM-DD, got {origin!r}"
        )
    with panel_generation_lock(out_dir, exclusive=True):
        _clear_previous_build(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        history = load_sales_long(m5_dir, evaluation=True)
        observed = history.loc[history["date"].dt.date <= origin_day].copy()
        if observed.empty or observed["date"].dt.date.max() != origin_day:
            raise ValueError(
                f"M5 training history does not end at requested origin {origin_day.isoformat()}"
            )
        final_day = origin_day + dt.timedelta(days=horizon)
        future_days = load_calendar(m5_dir).loc[
            lambda frame: (frame["date"].dt.date > origin_day)
            & (frame["date"].dt.date <= final_day),
            ["d", "date", "wm_yr_wk"],
        ]
        if len(future_days) != horizon:
            raise ValueError(
                f"M5 calendar must provide exactly {horizon} days after {origin_day.isoformat()}"
            )
        identity = history[
            ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        ].drop_duplicates()
        future = identity.merge(future_days, how="cross")
        future["qty"] = pd.Series(pd.NA, index=future.index, dtype="Int64")
        frame = pd.concat((observed, future), ignore_index=True)
        panel = exposure_from_first_sale(frame, first_sale_source=observed)
        panel.loc[panel["date"].dt.date > origin_day, "observation_state"] = "unobserved"
        _write_panel(out_dir, panel, m5_dir)
        return out_dir


def _write_panel(out_dir: pathlib.Path, frame: pd.DataFrame, m5_dir: pathlib.Path) -> None:
    """Attach published M5 dollar weights and write one owned, date-typed Parquet panel."""
    # WRMSSE weights each series by its share of DOLLAR sales, so the metric cannot be computed
    # without price. Joined here rather than in the metric: the metric is generic and must not
    # know where a particular benchmark keeps its prices.
    frame = frame.merge(load_prices(m5_dir), on=["store_id", "item_id", "wm_yr_wk"], how="left")
    frame["revenue"] = frame["qty"] * frame["sell_price"]
    # Write the daily columns as DATE, not as a timestamp. pandas defaults to `timestamp[ns]`,
    # which Spark 4 refuses outright (`PARQUET_TYPE_ILLEGAL: TIMESTAMP(NANOS)`), and which is
    # wrong anyway: M5's grain is a day, and a nanosecond timestamp invites a spurious intraday
    # component into every join downstream.
    for col in ("date", "first_sale"):
        frame[col] = frame[col].dt.date
    frame.to_parquet(out_dir, partition_cols=["store_id"], index=False)


def _clear_previous_build(out_dir: pathlib.Path) -> None:
    """Remove a previous build so fragments cannot accumulate. Refuses anything it did not write.

    Deletion is narrow on purpose: the directory must contain nothing but `store_id=` partition
    directories and `.parquet` files, or this raises rather than removing it. A build step that
    will delete whatever it is pointed at is one misconfiguration away from taking a real dataset
    with it.

    Raises:
        ValueError: `out_dir` exists and holds anything other than this function's own output.
    """
    if not out_dir.exists():
        return
    unexpected = [p.name for p in out_dir.iterdir()
                  if not (p.is_dir() and p.name.startswith("store_id=")
                          or p.suffix == ".parquet")]
    if unexpected:
        raise ValueError(
            f"{out_dir} holds files this build did not write: {sorted(unexpected)[:5]}. "
            "Refusing to delete it -- point build_parquet at a directory it owns"
        )
    shutil.rmtree(out_dir)


def load_sales_long(m5_dir: pathlib.Path = M5_DIR,
                    evaluation: bool = True) -> pd.DataFrame:
    """Wide `d_1..d_N` columns -> one row per (store, product, date).

    `evaluation` selects the file with the full history. The `validation` file stops 28 days
    earlier, and scoring against it while training on the other is a leak that looks like a
    strong result.
    """
    name = "sales_train_evaluation.csv" if evaluation else "sales_train_validation.csv"
    wide = pd.read_csv(m5_dir / name)
    return _sales_wide_to_long(wide, calendar_dir=m5_dir)


def load_private_evaluation_long(
    private_labels_dir: pathlib.Path,
    *,
    calendar_dir: pathlib.Path,
) -> pd.DataFrame:
    """Load the organizer's released private labels as M5 replay actuals, never as training data."""
    path = private_labels_dir / "sales_test_evaluation.csv"
    if not path.is_file():
        raise FileNotFoundError(f"M5 private replay needs organizer labels at {path}")
    wide = pd.read_csv(path)
    identity = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing = sorted(set(identity) - set(wide.columns))
    if missing:
        raise ValueError(f"M5 private-label file is missing identity columns: {missing}")
    expected_days = [f"d_{day}" for day in range(1942, 1970)]
    actual_days = [column for column in wide.columns if column.startswith("d_")]
    if actual_days != expected_days:
        raise ValueError(
            "M5 private-label file must contain exactly d_1942..d_1969 in order; got "
            f"{actual_days[:2]}..{actual_days[-2:]}"
        )
    # The organizer's label file omits Kaggle's submission id. Reconstruct the same bottom id
    # from the immutable item-store identity; it is used only to align forecast rows.
    wide.insert(0, "id", wide["item_id"].astype(str) + "_" + wide["store_id"].astype(str)
                + "_evaluation")
    return _sales_wide_to_long(wide, calendar_dir=calendar_dir)


def _sales_wide_to_long(wide: pd.DataFrame, *, calendar_dir: pathlib.Path) -> pd.DataFrame:
    """Convert one M5 sales table to long rows and bind its day keys to the supplied calendar."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing = sorted(set(id_cols) - set(wide.columns))
    if missing:
        raise ValueError(f"M5 sales file is missing columns: {missing}")
    day_cols = [column for column in wide.columns if column.startswith("d_")]
    long = wide.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="qty")
    cal = load_calendar(calendar_dir)[["d", "date", "wm_yr_wk"]]
    long = long.merge(cal, on="d", how="left")
    if long["date"].isna().any():
        raise ValueError("a day column has no calendar entry; the join would silently drop rows")
    return long


def load_prices(m5_dir: pathlib.Path = M5_DIR) -> pd.DataFrame:
    """Weekly sell prices, one row per (store, item, `wm_yr_wk`).

    Prices are WEEKLY in M5 while sales are daily, so a daily price needs the calendar's
    `wm_yr_wk` to join through -- which `load_sales_long` already carries.

    A price is absent before a product was first sold in a store and after it was delisted, and
    that absence is informative: it is M5's only other signal about ranging, and it agrees with
    `first_sale_date` without being derived from it. It is left as NULL rather than filled.
    """
    return pd.read_csv(m5_dir / "sell_prices.csv")


def first_sale_date(long: pd.DataFrame) -> pd.DataFrame:
    """The date each (store, product) first sold anything.

    This is M5's stand-in for a range-eligibility feed, and it is a WEAKER one -- it cannot tell a
    product that was stocked and did not sell from one that was not stocked at all. It is the
    honest upper bound on what M5 can support, and `exposure_from_first_sale` says so.
    """
    sold = long[long["qty"] > 0]
    return (sold.groupby(["store_id", "item_id"], as_index=False)["date"]
                .min().rename(columns={"date": "first_sale"}))


def exposure_from_first_sale(
    long: pd.DataFrame,
    *,
    first_sale_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mark cells before a product's first sale in a store as NOT eligible.

    M5's own documentation notes that leading zeros are absence of listing rather than absence of
    demand, and every serious M5 solution trims them. Not trimming them trains the occurrence head
    on "was this listed yet", which is the assortment-versus-demand confusion the exposure contract
    exists to prevent.

    LIMIT, stated because it bounds every M5 result: this cannot see a product that was ranged and
    sold nothing, nor one that was delisted and relisted. M5 has no stock feed, so `eligible` here
    is strictly coarser than `exposure_daily.is_ranged` on real data.
    """
    source = long if first_sale_source is None else first_sale_source
    out = long.merge(first_sale_date(source), on=["store_id", "item_id"], how="left")
    out["eligible"] = out["first_sale"].notna() & (out["date"] >= out["first_sale"])
    # Speak the contract's vocabulary, not a local one. `ObservationState` is defined once, in
    # spec/forecast_spec.py, and every adapter reads this column rather than `eligible` -- a
    # per-dataset boolean would make each benchmark loader invent its own eligibility dialect.
    # M5 emits no `lower_bound`: it has no stock feed, so no cell is known to be censored. That is
    # a property of the dataset and is why M5 bounds what can be claimed about censoring.
    out["observation_state"] = out["eligible"].map({True: "exact", False: "unobserved"})
    return out
