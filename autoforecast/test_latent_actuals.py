"""Governed latent actuals: the only door to a claim-bearing latent-demand score.

Run: JAVA_HOME=... PYSPARK_PYTHON=... python3 autoforecast/test_latent_actuals.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pyspark.sql import SparkSession

from autoforecast import latent_actuals
from autoforecast.panel import Forecast
from autoforecast.score_spark import score_forecast
from provenance.lineage import directory_digest

_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[2]").appName("test_latent")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def artifact(manifest_overrides: dict | None = None, skip_manifest: bool = False) -> str:
    root = pathlib.Path(tempfile.mkdtemp())
    day0 = dt.date(2016, 4, 25)
    rows = [(f"s{i}", day0 + dt.timedelta(days=j), float(2 + (i + j) % 3))
            for i in range(4) for j in range(7)]
    spark().createDataFrame(rows, "series_id string, date date, latent_demand double") \
        .write.mode("overwrite").parquet(str(root / "actuals"))
    manifest = {"method": "control-store experiment, 12 store pairs",
                "source": "workstream latent-2026Q2",
                "estimand": "unconstrained_demand",
                "content_sha256": directory_digest(root / "actuals", suffixes=(".parquet",)),
                "registered_on": "2026-08-25"}
    manifest.update(manifest_overrides or {})
    if not skip_manifest:
        (root / latent_actuals.MANIFEST_NAME).write_text(json.dumps(manifest))
    return str(root)


def test_a_governed_artifact_loads_and_actually_scores_latent_demand() -> None:
    """The full door: verified manifest -> the only unconstrained-demand panel there is ->
    the existing scorer's estimand-equality check passes LEGITIMATELY, with no scorer bypass."""
    panel = latent_actuals.load(artifact())
    assert panel.target_estimand == "unconstrained_demand"
    out = tempfile.mkdtemp()
    day0 = dt.date(2016, 4, 25)
    rows = [(f"s{i}", day0 + dt.timedelta(days=j), 2.5) for i in range(4) for j in range(7)]
    spark().createDataFrame(rows, "series_id string, date date, yhat double") \
        .write.mode("overwrite").parquet(out)
    forecast = Forecast(adapter="mft", path=out, origin="2016-04-24", horizon=7, n_series=4,
                        disposition={"emitted": 28},
                        target_estimand="unconstrained_demand").validate()
    score = score_forecast(panel, forecast, spark(), missing_actual="no_data")
    assert score.n_scored == 28, "a governed latent panel must score every covered cell"


def test_everything_ungoverned_is_refused_with_the_reason() -> None:
    cases = [
        (dict(skip_manifest=True), "no latent_manifest"),
        (dict(manifest_overrides={"method": "  "}), "claim left unmade"),
        (dict(manifest_overrides={"estimand": "sales_given_availability"}),
         "nothing else"),
    ]
    for kwargs, needle in cases:
        try:
            latent_actuals.load(artifact(**kwargs))
        except latent_actuals.LatentActualsError as e:
            assert needle in str(e), (needle, str(e))
            continue
        raise AssertionError(f"an ungoverned artifact loaded: {kwargs}")


def test_ambiguous_or_nonstandard_latent_manifest_is_refused_before_data_access() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    manifest = root / latent_actuals.MANIFEST_NAME
    cases = (
        ('{"estimand":"unconstrained_demand","estimand":"sales_given_availability"}',
         "duplicate JSON object key"),
        ('{"method":"study","source":"sealed","estimand":"unconstrained_demand",'
         '"content_sha256":"abc","registered_on":"2026-08-25","extra":NaN}',
         "non-finite JSON number"),
        ('["not", "an", "object"]', "must be one JSON object"),
    )
    for payload, needle in cases:
        manifest.write_text(payload)
        try:
            latent_actuals.load(root)
        except latent_actuals.LatentActualsError as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError(f"ambiguous latent manifest was accepted: {payload}")


def test_a_mutated_artifact_stops_loading() -> None:
    """The content hash is the point: an edited latent file must stop loading rather than
    quietly scoring -- a latent 'actual' someone touched is a number with no provenance."""
    path = pathlib.Path(artifact())
    fragment = sorted((path / "actuals").glob("*.parquet"))[0]
    fragment.write_bytes(fragment.read_bytes() + b"\x00")
    try:
        latent_actuals.load(path)
    except latent_actuals.LatentActualsError as e:
        assert "do not match the registered content" in str(e)
        return
    raise AssertionError("edited artifact bytes scored as registered evidence")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
    if _SPARK is not None:
        _SPARK.stop()
