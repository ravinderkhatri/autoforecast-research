"""The exposure contract: was this cell EXPOSED to demand at all?

Four questions, and a cell is exposed only if all four say yes:

  · **ranged** -- was the product listed in this store on this day?
  · **open** -- was the store trading?
  · **lifecycle** -- was the day inside the product's life, between launch and delisting?
  · **plan** -- is the cell one the business intends to forecast at all?

Never-ranged cells describe assortment rather than observed demand. Treating them as observed
zeros trains an occurrence head to predict product listings instead of demand.

**"EVER STOCKED IN THE WINDOW" IS NOT A VALID LABEL**, and the reason is the one this module is
built around. It misclassifies a launch (not ranged for the first half of the window, ranged
after), a delisting (the reverse), and a feed outage (never ranged, as far as the label can tell).
Worse, it LEAKS THE FUTURE INTO THE PAST: a product ranged only in December is marked ranged in
January too, so the January cells carry information that did not exist in January. What is needed
is an EFFECTIVE-DATED `is_ranged(store, product, date)`, and building one from a status history is
what `from_status_history` does.
"""

from __future__ import annotations

import pathlib

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

__all__ = ["ExposureError", "coverage", "eligible_expression", "from_first_sale",
           "from_status_history"]


class ExposureError(ValueError):
    """An exposure label was requested that the available columns cannot support."""


def eligible_expression(
    *, ranged_col: str | None = None, open_col: str | None = None,
    lifecycle_col: str | None = None, planned_col: str | None = None,
) -> Column:
    """The conjunction of whichever exposure facts a client actually has.

    An ABSENT column means "we cannot rule the cell out on that ground", so it contributes True.
    That is deliberately optimistic and it is the right default: asserting a cell was not ranged
    because the client has no ranging feed would discard real demand evidence on the strength of
    a column that does not exist. What it costs is stated in `coverage` -- a client with no
    ranging feed has a weaker eligibility signal, and the profiler reports that rather than
    pretending otherwise.
    """
    terms = [F.col(c) for c in (ranged_col, open_col, lifecycle_col, planned_col) if c]
    if not terms:
        return F.lit(True)
    out = terms[0]
    for term in terms[1:]:
        out = out & term
    return out


def from_status_history(
    status: DataFrame, *, series_col: str, time_col: str, status_col: str,
    ranged_values: tuple[str, ...],
) -> DataFrame:
    """Turn a sparse status history into an EFFECTIVE-DATED daily ranging flag.

    A status history records CHANGES -- "listed on 3 March", "delisted on 12 November" -- and the
    days between carry the last status that was set. Forward-filling that is the whole job, and
    doing it with a window that looks BACKWARD ONLY is what keeps it effective-dated: a
    `last_value` over an unbounded window in either direction would let November's delisting
    decide March's flag.

    Args:
        status: The status history, one row per change.
        series_col: Series identifier.
        time_col: Date the status took effect.
        status_col: The status value.
        ranged_values: Which status values count as ranged.

    Returns:
        The same frame with an `is_ranged` column, forward-filled from the last status at or
        before each row's date.

    Raises:
        ExposureError: `ranged_values` is empty, which would mark everything unranged.
    """
    if not ranged_values:
        raise ExposureError(
            "no status value counts as ranged, so every cell would be excluded. If the intent is "
            "'this client has no ranging feed', pass ranged_col=None instead of an empty set")
    backward = (Window.partitionBy(series_col).orderBy(time_col)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    return status.withColumn(
        "is_ranged",
        F.last(F.col(status_col).isin(*ranged_values), ignorenulls=True).over(backward))


def from_first_sale(
    df: DataFrame, *, series_col: str, time_col: str, target_col: str,
) -> DataFrame:
    """The WEAKEST usable ranging label: everything from a series' first positive sale onward.

    Used when a client has no ranging feed at all, and public benchmarks generally do not. It is
    honest about being coarse:

      · it CANNOT see a product that was ranged and simply never sold -- those cells are marked
        unranged and their genuine zero-demand evidence is lost;
      · it CANNOT see a delisting, so a product withdrawn in June stays "ranged" to the end;
      · it CANNOT see a relisting gap.

    So it is strictly coarser than an effective-dated feed, and every result computed on it
    inherits that. What it DOES catch is the leading run before a product existed in a store,
    which on M5 is 20.9% of the grid -- the largest single piece of the problem.

    Raises:
        ExposureError: nothing in the frame ever sold, so there is no first sale to key on.
    """
    first = (df.filter(F.col(target_col) > 0).groupBy(series_col)
             .agg(F.min(time_col).alias("_first_sale")))
    if first.limit(1).count() == 0:
        raise ExposureError(
            "no positive sale anywhere in the frame, so a first-sale label cannot be derived. "
            "Either the target column is wrong or this client has no sales at all")
    return (df.join(first, series_col, "left")
            .withColumn("is_ranged", F.col("_first_sale").isNotNull()
                        & (F.col(time_col) >= F.col("_first_sale")))
            .drop("_first_sale"))


def coverage(df: DataFrame, *, eligible_col: str = "is_ranged") -> DataFrame:
    """Count grid cells and report the share admitted by the exposure label."""
    return df.agg(
        F.count(F.lit(1)).alias("n_cells"),
        F.avg(F.col(eligible_col).cast("double")).alias("eligible_share"),
        F.countDistinct(F.col(df.columns[0])).alias("n_first_col_values"),
    )


def apply(
    panel_path: str | pathlib.Path,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    *,
    series_col: str,
    time_col: str,
    target_col: str,
    ranged_col: str | None = None,
    open_col: str | None = None,
    lifecycle_col: str | None = None,
    planned_col: str | None = None,
    fall_back_to_first_sale: bool = False,
) -> pathlib.Path:
    """Write the panel with an `is_eligible` column. Returns the path.

    Args:
        panel_path: Parquet source.
        spark: Active session.
        out_dir: Parquet destination.
        series_col: Series identifier.
        time_col: Date column.
        target_col: Quantity, used only by the first-sale fallback.
        ranged_col: Effective-dated ranging flag, if the client has one.
        open_col: Store-trading flag.
        lifecycle_col: Inside-product-life flag.
        planned_col: Business intends to forecast this cell.
        fall_back_to_first_sale: When no exposure column at all is available, derive the coarse
            first-sale label rather than marking everything eligible. **Opt-in**, because the
            fallback is strictly weaker and a caller should choose it knowingly -- it cannot see
            a ranged product that never sold, a delisting, or a relisting gap.

    Raises:
        ExposureError: no exposure column was given and the fallback was not requested. Marking
            everything eligible would treat assortment decisions as demand evidence.
    """
    df = spark.read.parquet(str(panel_path))
    supplied = [c for c in (ranged_col, open_col, lifecycle_col, planned_col) if c]
    if not supplied:
        if not fall_back_to_first_sale:
            raise ExposureError(
                "no exposure column was supplied. Marking every cell eligible would feed the "
                "never-ranged cells into the loss as demand evidence. "
                "Pass fall_back_to_first_sale=True to accept the "
                "coarse label instead, knowing it cannot see a delisting or a product that was "
                "ranged and never sold")
        df = from_first_sale(df, series_col=series_col, time_col=time_col,
                             target_col=target_col)
        ranged_col = "is_ranged"
    out = df.withColumn("is_eligible", eligible_expression(
        ranged_col=ranged_col, open_col=open_col, lifecycle_col=lifecycle_col,
        planned_col=planned_col))
    out.write.mode("overwrite").parquet(str(out_dir))
    return pathlib.Path(str(out_dir))
