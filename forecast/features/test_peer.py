"""Tests for the peer-product graph, on REAL M5 through Spark.

This is the branch most likely to help and most likely to leak, which is why it is staged. The
tests are almost entirely about the leak and about the one structural property a co-purchase graph
cannot give you: substitutes.

Run: JAVA_HOME=... python3 forecast/features/test_peer.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pytest
from forecast.features import peer_products
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"

_SPARK: SparkSession | None = None


def _require_real_m5_store() -> None:
    if not STORE.exists():
        pytest.skip(f"real M5 Parquet fixture is not present: {STORE}")


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_peer")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def catalogue():
    """Real M5 products with their real taxonomy."""
    _require_real_m5_store()
    return (spark().read.parquet(str(STORE))
            .select("item_id", "dept_id", "cat_id").distinct().limit(120))


def baskets():
    """Days a product sold, treated as co-occurrence. Real M5, standing in for baskets -- M5 has
    no basket data, and this is declared rather than pretended: what is under test is the graph
    arithmetic, which does not care where the co-occurrence came from."""
    _require_real_m5_store()
    return (spark().read.parquet(str(STORE))
            .filter((F.col("qty") > 0) & (F.col("date") >= F.lit("2016-01-01")))
            .select(F.col("date").alias("basket"), "item_id").limit(40_000))


def test_co_purchase_is_degree_corrected_not_a_popularity_ranking() -> None:
    """Raw co-occurrence gives every product the same neighbours -- bananas co-occur with
    everything -- and the branch then learns nothing product-specific."""
    edges = peer_products.complement_edges(
        baskets(), basket_col="basket", product_col="item_id", top_k=5, min_support=5)
    assert edges.count() > 0
    # The most popular product must not be everybody's top neighbour, which is what an
    # uncorrected count produces.
    tops = edges.filter(F.col("_r" if "_r" in edges.columns else "weight").isNotNull()) \
                .groupBy("peer").agg(F.count(F.lit(1)).alias("n")).orderBy(F.desc("n")).collect()
    total_products = edges.select("product").distinct().count()
    assert tops[0]["n"] < total_products, \
        "one product is the neighbour of every other, which is a popularity ranking"


def test_a_pair_seen_twice_is_dropped_as_noise() -> None:
    """A pair seen twice has a spectacular lift and means nothing."""
    strict = peer_products.complement_edges(
        baskets(), basket_col="basket", product_col="item_id", top_k=5, min_support=200)
    loose = peer_products.complement_edges(
        baskets(), basket_col="basket", product_col="item_id", top_k=5, min_support=2)
    assert strict.count() < loose.count()


def test_substitutes_come_from_taxonomy_because_co_purchase_cannot_find_them() -> None:
    """Two brands of the same thing usually NEVER co-occur, so a co-purchase graph scores them at
    approximately zero -- a graph built only from co-occurrence has no substitutes in it at all,
    and the branch is then structurally unable to see cannibalisation while appearing to work."""
    edges = peer_products.substitute_edges(
        catalogue(), product_col="item_id", taxonomy_cols=("cat_id", "dept_id"), top_k=5)
    assert edges.count() > 0
    joined = (edges.join(catalogue().select(F.col("item_id").alias("product"),
                                            F.col("dept_id").alias("pd")), "product")
              .join(catalogue().select(F.col("item_id").alias("peer"),
                                       F.col("dept_id").alias("sd")), "peer"))
    same_dept = joined.filter(F.col("pd") == F.col("sd")).count()
    assert same_dept / joined.count() > 0.8, \
        "substitutes are not concentrated in the same department, so similarity is not working"


def test_the_finest_taxonomy_level_outweighs_the_coarsest() -> None:
    """Two products in the same sub-category are far more substitutable than two in the same
    department."""
    cat = spark().createDataFrame(
        [("a", "FOODS", "FOODS_1"), ("b", "FOODS", "FOODS_1"), ("c", "FOODS", "FOODS_2")],
        "item_id string, cat_id string, dept_id string")
    edges = peer_products.substitute_edges(
        cat, product_col="item_id", taxonomy_cols=("cat_id", "dept_id"), top_k=5)
    w = {(r["product"], r["peer"]): r["weight"] for r in edges.collect()}
    assert w[("a", "b")] > w[("a", "c")], "same department did not outweigh same category alone"


def test_no_taxonomy_is_refused_rather_than_falling_back_to_co_purchase() -> None:
    structural_catalogue = spark().createDataFrame([("a",)], "item_id string")
    try:
        peer_products.substitute_edges(
            structural_catalogue, product_col="item_id", taxonomy_cols=()
        )
    except peer_products.PeerError as e:
        assert "no substitutes in it" in str(e); return
    raise AssertionError("silently produced a graph with no substitutes")


def test_a_neighbours_sales_may_not_be_carried() -> None:
    """THE leak this module is built to refuse. A neighbour's sales are unknown at forecast time,
    so using them means needing a forecast of the neighbour first -- and in BACKTEST they are
    available, so the leak scores beautifully and cannot be reproduced live."""
    d = tempfile.mkdtemp()
    spark().createDataFrame([("a", "2016-05-01", 3.0, 0.2)],
                            "series_id string, date string, qty double, discount_depth double") \
           .withColumn("date", F.to_date("date")).write.mode("overwrite").parquet(d)
    edges = spark().createDataFrame([("a", "b", 1.0)], "product string, peer string, weight double")
    for leaky in ("qty", "sales", "demand", "revenue"):
        try:
            peer_products.build_messages(d, edges, edges, spark(), "/tmp/never",
                                         signal_cols=(leaky,))
        except peer_products.PeerError as e:
            assert "TARGET rather than a plan" in str(e) or "absent" in str(e); continue
        raise AssertionError(f"carried {leaky!r} from a neighbour")


def test_complements_and_substitutes_stay_in_separate_channels() -> None:
    """Summing them here would do exactly what softmax pooling does, and for the same reason: a
    large complement effect would hide a substitution before the model saw either."""
    d = tempfile.mkdtemp()
    spark().createDataFrame(
        [("x", "2016-05-01", 0.5), ("y", "2016-05-01", -0.5)],
        "series_id string, date string, discount_depth double") \
        .withColumn("date", F.to_date("date")).write.mode("overwrite").parquet(d)
    comp = spark().createDataFrame([("a", "x", 1.0)], "product string, peer string, weight double")
    subs = spark().createDataFrame([("a", "y", 1.0)], "product string, peer string, weight double")
    out = spark().read.parquet(str(peer_products.build_messages(
        d, comp, subs, spark(), tempfile.mkdtemp(), signal_cols=("discount_depth",))))
    row = out.collect()[0]
    assert abs(row["discount_depth_complement"] - 0.5) < 1e-9
    assert abs(row["discount_depth_substitute"] + 0.5) < 1e-9
    assert "discount_depth" not in out.columns, "the two channels were merged into one"


def test_a_product_with_no_neighbours_gets_zero_and_a_flag() -> None:
    """'No substitutes found' is a real state, distinct from 'substitutes found and they are
    doing nothing'. A null would erase the difference."""
    d = tempfile.mkdtemp()
    spark().createDataFrame([("x", "2016-05-01", 0.5)],
                            "series_id string, date string, discount_depth double") \
           .withColumn("date", F.to_date("date")).write.mode("overwrite").parquet(d)
    comp = spark().createDataFrame([("a", "x", 1.0)], "product string, peer string, weight double")
    empty = spark().createDataFrame([], "product string, peer string, weight double")
    out = spark().read.parquet(str(peer_products.build_messages(
        d, comp, empty, spark(), tempfile.mkdtemp(), signal_cols=("discount_depth",))))
    row = out.collect()[0]
    assert row["discount_depth_substitute"] == 0.0
    assert row["discount_depth_substitute_present"] is False
    assert row["discount_depth_complement_present"] is True


def test_a_non_positive_top_k_is_refused() -> None:
    structural_baskets = spark().createDataFrame([("b", "a")], "basket string, item_id string")
    try:
        peer_products.complement_edges(
            structural_baskets, basket_col="basket", product_col="item_id", top_k=0
        )
    except peer_products.PeerError as e:
        assert "top_k" in str(e); return
    raise AssertionError("built a graph keeping zero neighbours")


def test_complement_edges_does_not_create_an_unowned_cache(monkeypatch) -> None:
    """A lazy result cannot safely return a private cache with no lifecycle owner."""
    from pyspark.sql import DataFrame

    tiny = spark().createDataFrame(
        [("b1", "a"), ("b1", "b"), ("b2", "a"), ("b2", "b")],
        "basket string, item_id string",
    )

    def reject_cache(_frame: DataFrame) -> DataFrame:
        raise AssertionError("complement_edges created a cache it cannot release")

    monkeypatch.setattr(type(tiny), "cache", reject_cache)
    edges = peer_products.complement_edges(
        tiny, basket_col="basket", product_col="item_id", top_k=2, min_support=1
    )
    assert edges.collect(), "cache-free complement plan produced no edge"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
