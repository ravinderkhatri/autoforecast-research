"""Tests for ranking. The Spark cases run on REAL M5; the guard cases are structural.

The guards are what this module is for, so each is tested on its own -- a guard nothing would
notice failing is not a guard.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import shutil
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from pyspark.sql import SparkSession

from autoforecast import grid, ranking, registry
from autoforecast.adapters import register_defaults
from autoforecast.panel import Forecast, Panel
from autoforecast.protocol import EvaluationManifest
from autoforecast.score import Score

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
HORIZON = 28

# Real M5 origins, chosen to SATISFY the manifest rather than to be tidy. Two constraints bind:
#
#   · the manifest needs at least THREE rolling origins, so a two-origin split is refused;
#   · a development origin's 28-day horizon may not reach the acceptance period, which rules out
#     2016-03-27 -- its window ends exactly on 2016-04-24, the acceptance origin.
#
# So the development origins step back in 28-day strides from 2016-02-28, and the last usable
# origin with real actuals behind it (2016-04-24) is reserved for acceptance.
DEV_ORIGINS = ("2016-01-03", "2016-01-31", "2016-02-28")
ACC_ORIGIN = "2016-04-24"
ALL_ORIGINS = (*DEV_ORIGINS, ACC_ORIGIN)

_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    """One local session for the file."""
    global _SPARK
    if _SPARK is None:
        for var in ("JAVA_HOME", "PYSPARK_PYTHON"):
            if not os.environ.get(var):
                raise RuntimeError(
                    f"{var} is not set. PYSPARK_PYTHON matters here because the statsforecast "
                    "adapters run a Python UDF per series, so the EXECUTOR needs fugue installed. "
                    "Set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[4]").appName("test_select")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.driver.memory", "4g")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def panel() -> Panel:
    if not STORE.is_dir():
        pytest.skip("public M5 Parquet is not installed")
    return Panel(path=STORE, series_col="id", time_col="date", target_col="qty",
                 exposure_col="observation_state", season_length=7).validate()


def manifest(
    acceptance: tuple[str, ...] = (ACC_ORIGIN,),
    grid_hashes: dict[str, str] | None = None,
    missing_forecast_policy: str = "error",
    target_estimand: str = "sales_given_availability",
) -> EvaluationManifest:
    return EvaluationManifest(
        name="m5_phase2", target_estimand=target_estimand,
        origins=ALL_ORIGINS, horizon_days=HORIZON,
        scoring_grain=("store", "product", "day"),
        eligible_rule="m5.exposure_from_first_sale (first sale in store)",
        zero_fill_rule="M5 is dense; an absent row is a source failure, not a zero",
        source_cutoffs={"m5": "2016-05-22T00:00:00Z"}, clients=("m5",), excluded_clients={},
        encoder_days=365, acceptance_origins=acceptance,
        eligible_grid_hashes=dict(grid_hashes or {}),
        missing_forecast_policy=missing_forecast_policy,
        latent_demand_source=("latent-demand.parquet"
                              if target_estimand == "unconstrained_demand" else None))


def fake(mae: float) -> Score:
    """A Score with just the fields a ranking reads. SYNTHETIC (structure): the guards under test
    are about which dict decided the winner, not about any forecast."""
    return Score(n_scored=10, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
                 mae=mae, rmse=mae, bias=0.0, bias_pct_of_volume=0.0, mase=None,
                 sum_abs_error=mae * 10, sum_squared_error=mae * mae * 10,
                 sum_error=0.0, sum_actual=10.0, sum_naive_abs_error=0.0)


# ---------------------------------------------------------------- guards (structural)

def test_a_ranking_cannot_be_chosen_on_the_acceptance_scores() -> None:
    """THE guard. Development says b wins; acceptance says a. Naming a is the leak this module
    exists to prevent, and it must fail at construction rather than be caught by review."""
    try:
        ranking.Ranking(manifest_hash="h", metric="mae",
                       development={"a": fake(2.0), "b": fake(1.0)},
                       acceptance={"a": fake(0.5), "b": fake(3.0)}, chosen="a")
    except ranking.SelectionError as e:
        assert "the development winner on mae is 'b'" in str(e); return
    raise AssertionError("built a ranking whose winner was picked using the acceptance scores")


def test_a_ranking_accepts_the_development_winner() -> None:
    r = ranking.Ranking(manifest_hash="h", metric="mae",
                       development={"a": fake(2.0), "b": fake(1.0)},
                       acceptance={"a": fake(0.5), "b": fake(3.0)}, chosen="b")
    assert r.chosen == "b" and r.order == ("b", "a")


def test_ranking_result_refuses_malformed_identity_population_and_adapter_sets() -> None:
    base = {
        "manifest_hash": "manifest",
        "metric": "mae",
        "development": {"a": fake(1.0)},
        "acceptance": {"a": fake(1.0)},
        "chosen": "a",
        "n_common_cells": 10,
    }
    for change in (
        {"manifest_hash": ""},
        {"development": {"": fake(1.0)}, "acceptance": {}, "chosen": ""},
        {"acceptance": {"other": fake(1.0)}},
        {"chosen": "other"},
        {"n_common_cells": True},
        {"handicaps": {"other": "not ranked"}},
    ):
        try:
            ranking.Ranking(**(base | change))
        except ranking.SelectionError:
            continue
        raise AssertionError(f"accepted malformed ranking evidence {change!r}")


def test_ranking_by_bias_is_refused() -> None:
    """A forecast can have zero bias and be wrong in both directions at every cell, so ranking by
    it selects for cancellation."""
    try:
        ranking.Ranking(manifest_hash="h", metric="bias", development={"a": fake(1.0)},
                       acceptance={}, chosen="a")
    except ranking.SelectionError as e:
        assert "selects for errors that cancel" in str(e); return
    raise AssertionError("ranked by bias")


def test_rank_refuses_a_panel_manifest_estimand_mismatch_before_running() -> None:
    source = Panel(path=STORE, series_col="id", time_col="date", target_col="qty")
    try:
        ranking.rank(source, manifest(target_estimand="unconstrained_demand"),
                     ("a", "b"), HORIZON, None, pathlib.Path("/tmp/not-used"),
                     missing_actual="no_data")
    except ranking.SelectionError as exc:
        assert "panel target_estimand" in str(exc) and "different quantities" in str(exc)
        return
    raise AssertionError("ranked a sales panel under a latent-demand manifest")


def test_rank_refuses_non_daily_panel_against_a_days_based_manifest() -> None:
    source = Panel(
        path=STORE, series_col="id", time_col="date", target_col="qty", frequency="W"
    )
    with pytest.raises(ranking.SelectionError, match="daily"):
        ranking.rank(
            source,
            manifest(),
            ("a", "b"),
            HORIZON,
            None,
            pathlib.Path("/tmp/not-used"),
            missing_actual="no_data",
        )


def test_rank_refuses_unwired_latent_demand_actuals_before_running() -> None:
    latent_panel = Panel(path=STORE, series_col="id", time_col="date", target_col="qty",
                         exposure_col="observation_state",
                         target_estimand="unconstrained_demand")
    try:
        ranking.rank(latent_panel, manifest(target_estimand="unconstrained_demand"),
                     ("a", "b"), HORIZON, None, pathlib.Path("/tmp/not-used"),
                     missing_actual="no_data")
    except ranking.SelectionError as exc:
        assert "content-bound actuals artifact" in str(exc)
        return
    raise AssertionError("scored latent demand against an unwired source description")


def test_an_undefined_metric_never_wins() -> None:
    """A None ratio maps to +inf rather than sorting first, which is what `None < float` would
    do in some orderings and what a naive `min` on mixed types would raise on."""
    undefined = Score(n_scored=1, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
                      mae=1.0, rmse=1.0, bias=0.0, bias_pct_of_volume=None, mase=None,
                      sum_abs_error=1.0, sum_squared_error=1.0, sum_error=0.0,
                      sum_actual=0.0, sum_naive_abs_error=0.0)
    r = ranking.Ranking(manifest_hash="h", metric="absolute_error_pct_of_volume",
                       development={"defined": fake(9.0), "undefined": undefined},
                       acceptance={}, chosen="defined")
    assert r.order == ("defined", "undefined")


def test_as_record_is_json_safe() -> None:
    import json
    r = ranking.Ranking(manifest_hash="h", metric="mae",
                       development={"a": fake(2.0), "b": fake(1.0)}, acceptance={}, chosen="b")
    assert json.loads(json.dumps(r.as_record()))["chosen"] == "b"


def test_run_options_reach_only_the_adapter_run_for_that_origin() -> None:
    """The registered build must cross the generic registry boundary intact.

    This is deliberately a structural unit test: the bug was that the call contained only Spark
    and an output directory, so every tensor shape still looked valid while the candidate ran its
    smoke defaults instead of the manifest's configuration.
    """
    root = pathlib.Path(tempfile.mkdtemp())
    seen: list[dict[str, object]] = []

    def fake_run(name, _panel, origin, horizon, **kw):
        seen.append(dict(kw))
        return Forecast(adapter=name, path=str(root / "forecast"), origin=origin,
                        horizon=horizon, n_series=1,
                        disposition={"declined_for_test": horizon})

    def options(adapter, origin, target):
        assert adapter == "candidate" and origin == "2020-01-01"
        return {"encoder_days": 364, "checkpoint_out": str(target.with_suffix(".pt"))}

    with mock.patch.object(ranking.registry, "run", side_effect=fake_run):
        ranking.run_over_origins(
            "candidate", mock.Mock(frequency="D"), ("2020-01-01",), 7, object(), root,
            run_options=options)
    assert seen[0]["encoder_days"] == 364
    assert seen[0]["checkpoint_out"].endswith("2020-01-01.pt")
    assert "spark" in seen[0] and "out_dir" in seen[0]


def test_run_options_cannot_replace_harness_owned_arguments() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        ranking.run_over_origins(
            "candidate", object(), ("2020-01-01",), 7, object(), root,
            run_options=lambda *_: {"out_dir": "/tmp/not-the-evaluated-output"})
    except ranking.SelectionError as exc:
        assert "harness-owned" in str(exc); return
    raise AssertionError("adapter options replaced the evaluated output path")


def test_run_over_origins_refuses_a_cached_forecast_with_wrong_frequency(tmp_path) -> None:
    """A structurally valid cached forecast must still match the requested cadence."""
    sidecar = tmp_path / "candidate" / "2020-01-01.json"
    sidecar.parent.mkdir()
    sidecar.write_text("{}")
    cached = Forecast("candidate", "/not-read", "2020-01-01", 7, 1,
                      {"declined_for_test": 7}, frequency="W")
    with mock.patch.object(ranking, "_reuse_record", return_value=cached), \
            mock.patch.object(ranking.registry, "run") as run:
        with pytest.raises(ranking.SelectionError, match="cached frequency"):
            ranking.run_over_origins(
                "candidate", mock.Mock(frequency="D"), ("2020-01-01",), 7, object(),
                tmp_path, reuse=True)
        run.assert_not_called()


def test_run_cannot_cache_claimed_emissions_without_an_artifact() -> None:
    """A resumable sidecar cannot truthfully hash output bytes that were never written."""
    root = pathlib.Path(tempfile.mkdtemp())

    def missing_output(name, _panel, origin, horizon, **_kw):
        return Forecast(adapter=name, path=str(root / "absent"), origin=origin,
                        horizon=horizon, n_series=1, disposition={"emitted": horizon})

    with mock.patch.object(ranking.registry, "run", side_effect=missing_output):
        try:
            ranking.run_over_origins(
                "candidate", mock.Mock(frequency="D"), ("2020-01-01",), 7, object(), root)
        except ranking.SelectionError as exc:
            assert "did not materialise" in str(exc)
            assert not (root / "candidate" / "2020-01-01.json").exists()
            return
    raise AssertionError("cached emitted cells with no forecast artifact")


def test_reuse_rejects_a_crafted_stale_cache_for_every_request_dimension() -> None:
    """A sidecar is not a cache hit merely because it occupies the requested pathname.

    This reproduces the old false hit without Spark: an options-free resume used to accept a
    sidecar produced with non-default options, and the same record could be copied under another
    origin or adapter. Panel bytes, Panel roles, horizon and manifest were not recorded at all.
    Every variation below describes a different forecast request and must fail before the adapter
    or the cached Forecast artifact is touched.
    """
    root = pathlib.Path(tempfile.mkdtemp())
    panel_file = root / "panel.parquet"
    panel_file.write_bytes(b"panel-version-one")
    source = Panel(path=panel_file, series_col="id", time_col="date", target_col="qty")
    origin = "2020-01-01"

    def fake_run(name, _panel, run_origin, run_horizon, **_kw):
        return Forecast(adapter=name, path=str(root / "not-materialised"), origin=run_origin,
                        horizon=run_horizon, n_series=1,
                        disposition={"declined_for_test": run_horizon})

    def original_options(*_):
        return {"epochs": 3, "features": ("price", "promo")}
    with mock.patch.object(ranking.registry, "run", side_effect=fake_run):
        ranking.run_over_origins(
            "candidate", source, (origin,), 7, object(), root,
            run_options=original_options, manifest_hash="manifest-one")
    sidecar = root / "candidate" / f"{origin}.json"

    # First establish that an exact request really is reusable and does not invoke the adapter.
    with mock.patch.object(ranking.registry, "run",
                           side_effect=AssertionError("exact reuse invoked the adapter")):
        exact = ranking.run_over_origins(
            "candidate", source, (origin,), 7, object(), root, reuse=True,
            run_options=original_options, manifest_hash="manifest-one")
    assert exact[origin].horizon == 7

    def rejects(label: str, invoke) -> None:
        with mock.patch.object(ranking.registry, "run",
                               side_effect=AssertionError("stale cache invoked the adapter")):
            try:
                invoke()
            except ranking.SelectionError as exc:
                assert "cannot reuse" in str(exc), (label, exc)
                return
        raise AssertionError(f"reused cache after {label} changed")

    # The original bug: because the new call supplied no callback, the options hash was not
    # checked at all. Empty/default options cannot inherit an explicitly configured fit.
    rejects("adapter options", lambda: ranking.run_over_origins(
        "candidate", source, (origin,), 7, object(), root, reuse=True,
        manifest_hash="manifest-one"))
    rejects("manifest", lambda: ranking.run_over_origins(
        "candidate", source, (origin,), 7, object(), root, reuse=True,
        run_options=original_options, manifest_hash="manifest-two"))
    rejects("horizon", lambda: ranking.run_over_origins(
        "candidate", source, (origin,), 8, object(), root, reuse=True,
        run_options=original_options, manifest_hash="manifest-one"))

    changed_roles = Panel(path=panel_file, series_col="id", time_col="date", target_col="qty",
                          season_length=14)
    rejects("Panel declaration", lambda: ranking.run_over_origins(
        "candidate", changed_roles, (origin,), 7, object(), root, reuse=True,
        run_options=original_options, manifest_hash="manifest-one"))

    other_origin = "2020-01-02"
    other_origin_sidecar = root / "candidate" / f"{other_origin}.json"
    shutil.copyfile(sidecar, other_origin_sidecar)
    rejects("origin", lambda: ranking.run_over_origins(
        "candidate", source, (other_origin,), 7, object(), root, reuse=True,
        run_options=original_options, manifest_hash="manifest-one"))

    other_adapter_sidecar = root / "other" / f"{origin}.json"
    other_adapter_sidecar.parent.mkdir(parents=True)
    shutil.copyfile(sidecar, other_adapter_sidecar)
    rejects("adapter", lambda: ranking.run_over_origins(
        "other", source, (origin,), 7, object(), root, reuse=True,
        run_options=original_options, manifest_hash="manifest-one"))

    panel_file.write_bytes(b"panel-version-two")
    rejects("Panel content", lambda: ranking.run_over_origins(
        "candidate", source, (origin,), 7, object(), root, reuse=True,
        run_options=original_options, manifest_hash="manifest-one"))


@pytest.mark.parametrize("name", ["extra.bin", "extra", "extra.PARQUET", "manifest.json"])
def test_reuse_revalidates_reader_inventory_after_panel_digest_was_cached(
    tmp_path: pathlib.Path, name: str,
) -> None:
    from provenance.lineage import LineageError

    data = tmp_path / "panel"
    data.mkdir()
    (data / "part.parquet").write_bytes(b"panel-version-one")
    source = Panel(path=data, series_col="id", time_col="date", target_col="qty")
    origin = "2020-01-01"

    def fake_run(adapter, _panel, run_origin, run_horizon, **_kw):
        return Forecast(
            adapter, str(tmp_path / "not-materialised"), run_origin, run_horizon, 1,
            {"declined_for_test": run_horizon},
        )

    with mock.patch.object(ranking.registry, "run", side_effect=fake_run):
        ranking.run_over_origins("candidate", source, (origin,), 7, object(), tmp_path)
    # The second call must actually reuse the memoized digest, not just exercise a cold cache.
    before = ranking._panel_content_digest.cache_info().hits
    with mock.patch.object(ranking.registry, "run") as run:
        ranking.run_over_origins(
            "candidate", source, (origin,), 7, object(), tmp_path, reuse=True,
        )
        run.assert_not_called()
    assert ranking._panel_content_digest.cache_info().hits > before
    (data / name).write_bytes(b"untracked-reader-input")
    with mock.patch.object(ranking.registry, "run") as run:
        with pytest.raises(LineageError, match="reader-visible file"):
            ranking.run_over_origins(
                "candidate", source, (origin,), 7, object(), tmp_path, reuse=True,
            )
        run.assert_not_called()


def test_reuse_rejects_changed_code_identity_and_legacy_sidecars() -> None:
    """Same commit is not same code in a dirty tree, and old unbound records fail closed."""
    root = pathlib.Path(tempfile.mkdtemp())
    panel_file = root / "panel.parquet"
    panel_file.write_bytes(b"panel")
    source = Panel(path=panel_file, series_col="id", time_col="date", target_col="qty")
    origin = "2020-01-01"

    def fake_run(name, _panel, run_origin, run_horizon, **_kw):
        return Forecast(adapter=name, path=str(root / "not-materialised"), origin=run_origin,
                        horizon=run_horizon, n_series=1,
                        disposition={"declined_for_test": run_horizon})

    with mock.patch.object(ranking, "_code_identity", return_value={"source": "before"}), \
            mock.patch.object(ranking.registry, "run", side_effect=fake_run):
        ranking.run_over_origins("candidate", source, (origin,), 7, object(), root,
                                 manifest_hash="manifest")
    with mock.patch.object(ranking, "_code_identity", return_value={"source": "after"}):
        try:
            ranking.run_over_origins("candidate", source, (origin,), 7, object(), root,
                                     reuse=True, manifest_hash="manifest")
        except ranking.SelectionError as exc:
            assert "['code']" in str(exc)
        else:
            raise AssertionError("reused a forecast produced by different code")

    sidecar = root / "candidate" / f"{origin}.json"
    import json
    legacy = json.loads(sidecar.read_text())
    legacy.pop("cache_key")
    legacy.pop("cache_identity")
    sidecar.write_text(json.dumps(legacy))
    try:
        ranking.run_over_origins("candidate", source, (origin,), 7, object(), root,
                                 reuse=True, manifest_hash="manifest")
    except ranking.SelectionError as exc:
        assert "predates cache identity" in str(exc)
    else:
        raise AssertionError("reused a legacy sidecar with no request identity")


def test_reuse_refuses_ambiguous_or_nonstandard_sidecar_json_before_artifact_access() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    sidecar = root / "cached.json"
    cases = (
        '{"cache_identity":{},"cache_identity":{},"cache_key":"x"}',
        '{"cache_identity":{"version":2},"cache_key":"x","metric":NaN}',
        '["not", "a", "sidecar"]',
    )
    for payload in cases:
        sidecar.write_text(payload)
        try:
            ranking._reuse_record(sidecar, {}, "candidate", "2020-01-01", 7)
        except ranking.SelectionError as exc:
            assert "cannot reuse" in str(exc), str(exc)
        else:
            raise AssertionError(f"ambiguous cache sidecar was accepted: {payload}")


def test_rank_binds_every_cached_run_to_the_manifest_and_its_horizon() -> None:
    """The manifest hash must reach both development and acceptance cache identities."""
    frozen_hash = "a" * 64
    m = manifest(grid_hashes={origin: frozen_hash for origin in ALL_ORIGINS},
                 missing_forecast_policy="zero")
    root = pathlib.Path(tempfile.mkdtemp())
    calls: list[dict[str, object]] = []

    def fake_runs(adapter, _panel, origins, horizon, _spark, _root, **kwargs):
        calls.append({"adapter": adapter, "origins": origins, "horizon": horizon, **kwargs})
        return {origin: Forecast(adapter=adapter, path=str(root / adapter / origin),
                                 origin=origin, horizon=horizon, n_series=1,
                                 disposition={"declined_for_test": horizon})
                for origin in origins}

    grids = {origin: f"/grid/{origin}" for origin in ALL_ORIGINS}
    with mock.patch.object(ranking.registry, "get"), \
            mock.patch.object(ranking.registry, "handicaps", return_value={}), \
            mock.patch.object(ranking, "eligible_grid_content_hash",
                              return_value=frozen_hash), \
            mock.patch.object(ranking, "run_over_origins", side_effect=fake_runs), \
            mock.patch.object(ranking, "_exact_grid_restrictions",
                              return_value=(grids, 1)) as exact, \
            mock.patch.object(ranking, "pool_scores", return_value=fake(1.0)) as pooled:
        ranking.rank(mock.Mock(target_estimand=m.target_estimand, frequency="D"), m,
                     ("a", "b"), HORIZON, object(), root,
                     missing_actual="no_data", eligible_grids=grids)
    assert len(calls) == 4  # two adapters, development and acceptance
    assert all(call["manifest_hash"] == m.hash for call in calls)
    assert all(call["missing_forecast_policy"] == "zero" for call in calls)
    assert all(call["eligible_grid_hashes"] == m.eligible_grid_hashes for call in calls)
    assert all(call["eligible_grids"] == grids for call in calls)
    assert len(pooled.call_args_list) == 4
    assert all(call.kwargs["eligible_grids"] == grids for call in pooled.call_args_list)
    assert exact.call_count == 2

    try:
        ranking.rank(mock.Mock(target_estimand=m.target_estimand, frequency="D"), m,
                     ("a", "b"), HORIZON - 1, object(), root,
                     missing_actual="no_data")
    except ranking.SelectionError as exc:
        assert "does not match manifest horizon" in str(exc)
    else:
        raise AssertionError("ran a forecast window different from the manifest")


def test_governed_rank_requires_a_grid_for_exactly_every_manifest_origin() -> None:
    """A partial grid map cannot make some origins governed and leave the rest exploratory."""
    m = manifest(grid_hashes={origin: "a" * 64 for origin in ALL_ORIGINS})
    partial = {origin: f"/grid/{origin}" for origin in m.origins[:-1]}
    with mock.patch.object(
            ranking.registry, "get",
            side_effect=AssertionError("adapter lookup happened before grid contract validation")):
        try:
            ranking.rank(mock.Mock(target_estimand=m.target_estimand, frequency="D"), m,
                         ("a", "b"), HORIZON, object(), pathlib.Path("/tmp/out"),
                         missing_actual="no_data", eligible_grids=partial)
        except ranking.SelectionError as exc:
            assert "exactly the manifest origins" in str(exc)
            assert ACC_ORIGIN in str(exc)
        else:
            raise AssertionError("rank accepted a partial eligible-grid mapping")


def test_governed_rank_refuses_a_comparator_that_omits_one_grid_cell() -> None:
    """Reproduce the selection leak: one comparator's decline must not shrink everyone's grid."""
    import datetime as dt

    root = pathlib.Path(tempfile.mkdtemp())
    runs: dict[str, dict[str, Forecast]] = {"complete": {}, "declines": {}}
    grids: dict[str, str] = {}
    for index, origin in enumerate(ALL_ORIGINS):
        day = dt.date.fromisoformat(origin) + dt.timedelta(days=1)
        grid_path = root / "grids" / origin
        spark().createDataFrame([("hard", day), ("easy", day)],
                                "series_id string, date date").write.parquet(str(grid_path))
        grids[origin] = str(grid_path)
        for adapter in runs:
            path = root / adapter / origin
            rows = [("hard", day, 1.0), ("easy", day, 1.0)]
            if adapter == "declines" and index == 0:
                rows = [("easy", day, 1.0)]
            spark().createDataFrame(
                rows, "series_id string, date date, yhat double").write.parquet(str(path))
            runs[adapter][origin] = Forecast(
                adapter=adapter, path=str(path), origin=origin, horizon=HORIZON,
                n_series=2, disposition={"emitted_test": len(rows),
                                         "declined_test": 2 * HORIZON - len(rows)})

    frozen = {origin: grid.content_hash(path, spark()) for origin, path in grids.items()}
    m = manifest(acceptance=(), grid_hashes=frozen)

    def fake_runs(adapter, _panel, origins, *_args, **_kwargs):
        return {origin: runs[adapter][origin] for origin in origins}

    with mock.patch.object(ranking.registry, "get"), \
            mock.patch.object(ranking, "run_over_origins", side_effect=fake_runs), \
            mock.patch.object(
                ranking, "pool_scores",
                side_effect=AssertionError("a score was computed before exact-grid validation")):
        try:
            ranking.rank(mock.Mock(target_estimand=m.target_estimand, frequency="D"), m,
                         ("complete", "declines"), HORIZON, spark(), root / "out",
                         missing_actual="no_data", eligible_grids=grids)
        except ranking.SelectionError as exc:
            assert "declines/" in str(exc)
            assert "missing=1, extra=0" in str(exc)
            assert "decline set" in str(exc)
        else:
            raise AssertionError("a comparator omitted a hard grid cell and ranking continued")


def test_zero_policy_fills_only_omitted_canonical_keys_and_accounts_for_them() -> None:
    """A cold start remains in the exact grid and is visibly a baseline, not model output."""
    import datetime as dt
    import json

    root = pathlib.Path(tempfile.mkdtemp())
    source_panel = Panel(path=root / "panel", series_col="id", time_col="date",
                         target_col="qty")
    origin = "2020-01-01"
    day1, day2 = dt.date(2020, 1, 2), dt.date(2020, 1, 3)
    grid_path = root / "grid"
    spark().createDataFrame(
        [("seen", day1), ("seen", day2), ("cold", day1)],
        "series_id string, date date").write.parquet(str(grid_path))
    digest = grid.content_hash(grid_path, spark())

    def emits_one(name, _panel, run_origin, run_horizon, **kw):
        spark().createDataFrame(
            [("seen", day1, 4.0)],
            "series_id string, date date, yhat double").write.parquet(kw["out_dir"])
        return Forecast(name, kw["out_dir"], run_origin, run_horizon, 1,
                        {"emitted": 1, "declined_model_failure": 1})

    with mock.patch.object(ranking.registry, "run", side_effect=emits_one):
        finalized = ranking.run_over_origins(
            "candidate", source_panel, (origin,), 2, spark(), root / "runs",
            manifest_hash="manifest", missing_forecast_policy="zero",
            eligible_grids={origin: str(grid_path)},
            eligible_grid_hashes={origin: digest})[origin]

    rows = {(r["series_id"], r["date"].isoformat()): (r["yhat"], r["forecast_source"])
            for r in spark().read.parquet(finalized.path).collect()}
    assert rows == {
        ("seen", day1.isoformat()): (4.0, "adapter"),
        ("seen", day2.isoformat()): (0.0, "fallback_zero"),
        ("cold", day1.isoformat()): (0.0, "fallback_zero"),
    }
    assert finalized.disposition == {
        "emitted_adapter": 1,
        "emitted_fallback_zero": 2,
        "declined_outside_eligible_grid": 1,
    }
    sidecar = json.loads((root / "runs" / "candidate" / f"{origin}.json").read_text())
    assert sidecar["grid_reconciliation"]["source_forecast"]["disposition"] == {
        "emitted": 1, "declined_model_failure": 1}
    assert sidecar["forecast_content_hash"] == ranking._forecast_content_hash(finalized.path)
    ranking._exact_grid_restrictions(
        {"candidate": {origin: finalized}}, (origin,), {origin: str(grid_path)}, spark())


def test_zero_reconciliation_releases_grid_when_source_validation_fails(monkeypatch) -> None:
    """Every exception after cache ownership begins must release the canonical grid."""
    class CachedGrid:
        released = False

        def cache(self):
            return self

        def unpersist(self) -> None:
            self.released = True

    cached = CachedGrid()
    source = Forecast(
        "candidate", "/not-materialised", "2020-01-01", 1, 1, {"emitted": 1}
    )
    monkeypatch.setattr(ranking, "_canonical_grid", lambda *_args, **_kwargs: cached)
    monkeypatch.setattr(
        ranking,
        "_validated_source_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ranking.SelectionError("injected source validation failure")
        ),
    )
    with pytest.raises(ranking.SelectionError, match="injected source validation failure"):
        ranking._reconcile_zero_fallback(
            source, "/grid", "a" * 64, object(), pathlib.Path("/final")
        )
    assert cached.released, "failed reconciliation retained its canonical-grid cache"


def test_zero_policy_preserves_and_validates_the_declared_period_grid() -> None:
    import datetime as dt

    root = pathlib.Path(tempfile.mkdtemp())
    origin = "2020-01-01"
    days = [dt.date(2020, 1, 8), dt.date(2020, 1, 15)]
    grid_path = root / "weekly-grid"
    spark().createDataFrame(
        [("series", day) for day in days], "series_id string, date date"
    ).write.parquet(str(grid_path))
    source_path = root / "weekly-source"
    spark().createDataFrame(
        [("series", days[0], 4.0)], "series_id string, date date, yhat double"
    ).write.parquet(str(source_path))
    source = Forecast(
        "candidate", str(source_path), origin, 2, 1,
        {"emitted": 1, "declined_fixture": 1}, frequency="W",
    )
    finalized, _ = ranking.finalize_forecast_on_grid(
        source,
        grid_path,
        grid.content_hash(grid_path, spark()),
        spark(),
        root / "weekly-final",
        missing_forecast_policy="zero",
    )
    assert finalized.frequency == "W"
    assert {row["date"] for row in spark().read.parquet(finalized.path).collect()} == set(days)


def test_zero_policy_keeps_two_hourly_keys_on_one_day_distinct() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    origin = "2024-03-01T22:00:00"
    hours = ("2024-03-01T23:00:00", "2024-03-02T00:00:00")
    grid_path = root / "hourly-grid"
    (
        spark().createDataFrame(
            [("series", hour) for hour in hours], "series_id string, raw_date string"
        )
        .selectExpr("series_id", "cast(raw_date as timestamp_ntz) as date")
        .write.parquet(str(grid_path))
    )
    source_path = root / "hourly-source"
    (
        spark().createDataFrame(
            [("series", hours[0], 4.0)],
            "series_id string, raw_date string, yhat double",
        )
        .selectExpr("series_id", "cast(raw_date as timestamp_ntz) as date", "yhat")
        .write.parquet(str(source_path))
    )
    source = Forecast(
        "candidate",
        source_path,
        origin,
        2,
        1,
        {"emitted": 1, "declined_fixture": 1},
        frequency="H",
    )

    finalized, _ = ranking.finalize_forecast_on_grid(
        source,
        grid_path,
        grid.content_hash(grid_path, spark(), frequency="H"),
        spark(),
        root / "hourly-final",
        missing_forecast_policy="zero",
    )
    rows = spark().read.parquet(finalized.path).orderBy("date").collect()

    assert [row["date"].isoformat() for row in rows] == list(hours)
    assert [row["yhat"] for row in rows] == [4.0, 0.0]


def test_common_cells_refuses_equal_horizons_with_different_frequencies() -> None:
    forecasts = [
        Forecast("a", "/not-read-a", "2020-01-01", 2, 1, {"emitted": 2}),
        Forecast("b", "/not-read-b", "2020-01-01", 2, 1, {"emitted": 2}, frequency="W"),
    ]
    with pytest.raises(ranking.SelectionError, match="frequency"):
        ranking.common_cells(forecasts, object(), "/not-written")


@pytest.mark.parametrize(
    "day,error",
    [
        (dt.date(2024, 2, 15), "aligned|outside"),
        (dt.date(2024, 3, 31), "outside"),
    ],
)
def test_common_cells_refuses_uncanonical_or_out_of_window_keys(
    tmp_path, day, error
) -> None:
    artifact = tmp_path / "forecast"
    spark().createDataFrame(
        [("a", day, 1.0)], "series_id string, date date, yhat double"
    ).write.parquet(str(artifact))
    forecast = Forecast(
        "candidate", artifact, "2024-01-31", 1, 1, {"emitted": 1}, frequency="M"
    )
    output = tmp_path / "common"

    with pytest.raises(ranking.SelectionError, match=error):
        ranking.common_cells([forecast], spark(), output)

    assert not output.exists()


def test_common_cells_writes_canonical_daily_keys_from_midnight_timestamps(tmp_path) -> None:
    timestamp_artifact = tmp_path / "timestamp"
    spark().createDataFrame(
        [("a", dt.datetime(2024, 1, 2), 1.0)],
        "series_id string, date timestamp_ntz, yhat double",
    ).write.parquet(str(timestamp_artifact))
    date_artifact = tmp_path / "date"
    spark().createDataFrame(
        [("a", dt.date(2024, 1, 2), 2.0)],
        "series_id string, date date, yhat double",
    ).write.parquet(str(date_artifact))
    forecasts = [
        Forecast("timestamp", timestamp_artifact, "2024-01-01", 1, 1, {"emitted": 1}),
        Forecast("date", date_artifact, "2024-01-01", 1, 1, {"emitted": 1}),
    ]

    output, count = ranking.common_cells(forecasts, spark(), tmp_path / "common")
    frame = spark().read.parquet(output)

    assert count == 1
    assert frame.schema["date"].dataType.typeName() == "date"
    assert frame.first()["date"] == dt.date(2024, 1, 2)


@pytest.mark.parametrize("origin,horizon", [("20200101", 2), ("2020-01-01", True)])
def test_common_cells_refuses_noncanonical_span_before_spark_io(origin, horizon) -> None:
    forecasts = [
        Forecast("a", "/not-read", origin, horizon, 1, {"emitted": int(horizon)})
    ]
    with pytest.raises(ranking.SelectionError, match="canonical|horizon"):
        ranking.common_cells(forecasts, object(), "/not-written")


def test_finalize_forecast_on_grid_error_policy_validates_without_rewriting() -> None:
    forecast = Forecast(
        "candidate", "/not/materialised", "2020-01-01", 1, 1,
        {"declined_model_failure": 1})
    finalized, reconciliation = ranking.finalize_forecast_on_grid(
        forecast, "/unused/grid", "a" * 64, object(), "/unused/output",
        missing_forecast_policy="error")
    assert finalized is forecast and reconciliation is None


def test_zero_policy_rejects_extra_duplicate_and_malformed_raw_rows() -> None:
    """Reconciliation is not a filter that can erase bad adapter output."""
    import datetime as dt

    origin = "2020-01-01"
    day = dt.date(2020, 1, 2)
    cases = {
        "extra": ([('ok', day, '1.0'), ('extra', day, '2.0')], "outside"),
        "duplicate": ([('ok', day, '1.0'), ('ok', day, '2.0')], "duplicate"),
        # The malformed row is also outside the grid. Value validation must happen before the
        # anti-join, otherwise grid filtering would hide it.
        "malformed": ([('ok', day, '1.0'), ('extra', day, 'not-a-number')], "yhat"),
    }
    for label, (raw_rows, wanted) in cases.items():
        root = pathlib.Path(tempfile.mkdtemp())
        grid_path = root / "grid"
        spark().createDataFrame(
            [("ok", day)], "series_id string, date date").write.parquet(str(grid_path))
        digest = grid.content_hash(grid_path, spark())

        def bad_run(name, _panel, run_origin, run_horizon, raw_rows=raw_rows, **kw):
            spark().createDataFrame(
                raw_rows, "series_id string, date date, yhat string").write.parquet(kw["out_dir"])
            return Forecast(name, kw["out_dir"], run_origin, run_horizon, 2, {"emitted": 2})

        with mock.patch.object(ranking.registry, "run", side_effect=bad_run):
            try:
                ranking.run_over_origins(
                    "candidate", Panel(root / "panel", "id", "date", "qty"), (origin,), 1,
                    spark(), root / "runs", manifest_hash="manifest",
                    missing_forecast_policy="zero",
                    eligible_grids={origin: str(grid_path)},
                    eligible_grid_hashes={origin: digest})
            except (ranking.SelectionError, ValueError) as exc:
                assert wanted in str(exc).lower(), (label, exc)
                continue
        raise AssertionError(f"zero fallback hid a {label} raw forecast")


def test_zero_policy_and_grid_hash_changes_invalidate_reuse() -> None:
    import datetime as dt

    root = pathlib.Path(tempfile.mkdtemp())
    origin = "2020-01-01"
    day = dt.date(2020, 1, 2)
    first_grid = root / "grid-one"
    spark().createDataFrame(
        [("a", day)], "series_id string, date date").write.parquet(str(first_grid))
    first_hash = grid.content_hash(first_grid, spark())

    def decline_all(name, _panel, run_origin, run_horizon, **kw):
        return Forecast(name, kw["out_dir"], run_origin, run_horizon, 1,
                        {"declined_model_failure": 1})

    source_panel = Panel(root / "panel", "id", "date", "qty")
    with mock.patch.object(ranking.registry, "run", side_effect=decline_all):
        ranking.run_over_origins(
            "candidate", source_panel, (origin,), 1, spark(), root / "runs",
            manifest_hash="manifest", missing_forecast_policy="zero",
            eligible_grids={origin: str(first_grid)},
            eligible_grid_hashes={origin: first_hash})

    with mock.patch.object(
            ranking.registry, "run",
            side_effect=AssertionError("exact zero-policy reuse invoked the adapter")):
        reused = ranking.run_over_origins(
            "candidate", source_panel, (origin,), 1, spark(), root / "runs", reuse=True,
            manifest_hash="manifest", missing_forecast_policy="zero",
            eligible_grids={origin: str(first_grid)},
            eligible_grid_hashes={origin: first_hash})[origin]
    assert reused.disposition == {"emitted_adapter": 0, "emitted_fallback_zero": 1}

    def rejects(invoke) -> None:
        with mock.patch.object(
                ranking.registry, "run",
                side_effect=AssertionError("a changed fallback request invoked the adapter")):
            try:
                invoke()
            except ranking.SelectionError as exc:
                assert "cached request differs" in str(exc)
                assert "missing_forecast" in str(exc)
                return
        raise AssertionError("reused a forecast under a changed fallback contract")

    rejects(lambda: ranking.run_over_origins(
        "candidate", source_panel, (origin,), 1, spark(), root / "runs", reuse=True,
        manifest_hash="manifest", missing_forecast_policy="error"))

    changed_grid = root / "grid-two"
    spark().createDataFrame(
        [("b", day)], "series_id string, date date").write.parquet(str(changed_grid))
    changed_hash = grid.content_hash(changed_grid, spark())
    rejects(lambda: ranking.run_over_origins(
        "candidate", source_panel, (origin,), 1, spark(), root / "runs", reuse=True,
        manifest_hash="manifest", missing_forecast_policy="zero",
        eligible_grids={origin: str(changed_grid)},
        eligible_grid_hashes={origin: changed_hash}))


def test_governed_rank_refuses_grid_bytes_different_from_the_manifest() -> None:
    frozen_hash = "a" * 64
    m = manifest(grid_hashes={origin: frozen_hash for origin in ALL_ORIGINS})
    paths = {origin: f"/grid/{origin}" for origin in ALL_ORIGINS}
    with mock.patch.object(ranking, "eligible_grid_content_hash", return_value="b" * 64):
        try:
            ranking._require_complete_grid_mapping(m, paths, object())
        except ranking.SelectionError as exc:
            assert "materialised cell set changed" in str(exc)
        else:
            raise AssertionError("ranking accepted grid content different from the frozen hash")


def test_grid_content_hash_is_layout_independent_and_key_sensitive() -> None:
    """Grid identity follows sorted key content, not Spark's generated part-file names."""
    import datetime as dt

    root = pathlib.Path(tempfile.mkdtemp())
    rows = [("a", dt.date(2020, 1, 2)), ("b", dt.date(2020, 1, 3))]
    hashes = []
    for name, values in (("forward", rows), ("reverse", list(reversed(rows))),
                         ("changed", [rows[0], ("c", rows[1][1])])):
        path = root / name
        spark().createDataFrame(values, "series_id string, date date").write.parquet(str(path))
        hashes.append(grid.content_hash(path, spark()))
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_reuse_refuses_a_structurally_valid_forecast_whose_values_changed() -> None:
    """Schema validation cannot prove that cached forecast bytes are the produced bytes."""
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = pathlib.Path(tempfile.mkdtemp())
    panel_file = root / "panel.parquet"
    panel_file.write_bytes(b"panel")
    source = Panel(path=panel_file, series_col="id", time_col="date", target_col="qty")
    artifact = root / "forecast.parquet"
    origin = "2020-01-01"

    def write_forecast(value: float) -> None:
        pq.write_table(pa.table({"series_id": ["s1"], "date": [dt.date(2020, 1, 2)],
                                 "yhat": [value]}), artifact)

    def fake_run(name, _panel, run_origin, run_horizon, **_kw):
        write_forecast(1.0)
        return Forecast(adapter=name, path=str(artifact), origin=run_origin,
                        horizon=run_horizon, n_series=1, disposition={"emitted": 1})

    with mock.patch.object(ranking.registry, "run", side_effect=fake_run):
        ranking.run_over_origins("candidate", source, (origin,), 1, object(), root,
                                 manifest_hash="manifest")

    # This remains perfectly valid Forecast Parquet: same schema, key, date, row count and value
    # domain. Only content identity can show it is not the artifact the adapter produced.
    write_forecast(999.0)
    try:
        ranking.run_over_origins("candidate", source, (origin,), 1, object(), root,
                                 reuse=True, manifest_hash="manifest")
    except ranking.SelectionError as exc:
        assert "forecast artifact changed" in str(exc)
    else:
        raise AssertionError("reused forecast values overwritten after the run")


# ---------------------------------------------------------------- real M5 through Spark

def test_rank_needs_at_least_two_adapters() -> None:
    register_defaults() if "seasonal_naive" not in registry.names() else None
    source = Panel(path=STORE, series_col="id", time_col="date", target_col="qty")
    try:
        ranking.rank(source, manifest(), ("seasonal_naive",), HORIZON, object(),
                    pathlib.Path(tempfile.mkdtemp()), missing_actual="no_data")
    except ranking.SelectionError as e:
        assert "at least two adapters" in str(e); return
    raise AssertionError("ranked one adapter against nothing")


def test_a_manifest_that_reserved_every_origin_never_reaches_ranking() -> None:
    """The guard lives in EvaluationManifest, not in rank(): a manifest reserving every origin
    cannot be CONSTRUCTED, so a duplicate check inside rank() would be unreachable and would
    silently drift. This test asserts where the rule actually lives."""
    try:
        manifest(acceptance=ALL_ORIGINS)
    except ValueError as e:
        assert "nothing to develop against" in str(e); return
    raise AssertionError("built a manifest with no development origins")


def test_the_manifest_refuses_a_development_window_that_reaches_acceptance() -> None:
    """Not a ranking.py guard, but the reason DEV_ORIGIN is 2016-02-28 and not 2016-03-27 -- and
    if it ever stops holding, every ranking in this file silently becomes a leak."""
    try:
        EvaluationManifest(
            name="m5_bad", target_estimand="sales_given_availability",
            origins=("2016-01-31", "2016-02-28", "2016-03-27", ACC_ORIGIN),
            horizon_days=HORIZON,
            scoring_grain=("store", "product", "day"), eligible_rule="x", zero_fill_rule="y",
            source_cutoffs={"m5": "2016-05-22T00:00:00Z"}, clients=("m5",), excluded_clients={},
            encoder_days=365, acceptance_origins=(ACC_ORIGIN,))
    except ValueError as e:
        assert "acceptance period" in str(e); return
    raise AssertionError("accepted a development horizon ending on the acceptance origin")


def test_rank_on_real_m5_chooses_on_development_and_reports_on_acceptance() -> None:
    register_defaults() if "seasonal_naive" not in registry.names() else None
    out = pathlib.Path(tempfile.mkdtemp())
    r = ranking.rank(panel(), manifest(), ("seasonal_naive", "croston"), HORIZON, spark(), out,
                    missing_actual="no_data")
    assert set(r.development) == {"seasonal_naive", "croston"}
    assert set(r.acceptance) == {"seasonal_naive", "croston"}
    assert r.chosen == r.order[0]
    # Both adapters must have been scored on the SAME cells, or the comparison is not one.
    assert r.n_common_cells > 0
    ns = {s.n_scored for s in r.development.values()}
    assert len(ns) == 1, f"adapters scored on different numbers of cells: {ns}"
    assert r.manifest_hash == manifest().hash


def test_common_cells_refuses_forecasts_from_different_windows() -> None:
    register_defaults() if "seasonal_naive" not in registry.names() else None
    out = pathlib.Path(tempfile.mkdtemp())
    a = registry.run("seasonal_naive", panel(), DEV_ORIGINS[0], 7, spark=spark(),
                     out_dir=str(out / "a"))
    b = registry.run("seasonal_naive", panel(), ACC_ORIGIN, 7, spark=spark(),
                     out_dir=str(out / "b"))
    try:
        ranking.common_cells([a, b], spark(), out / "x")
    except ranking.SelectionError as e:
        assert "different (origin, horizon)" in str(e); return
    raise AssertionError("intersected cells across two different forecast windows")


def test_reuse_skips_an_origin_that_already_ran_and_is_off_by_default() -> None:
    """A gate run over four origins and four adapters takes hours; losing three completed origins
    to a crash in the fourth is how a measurement stops getting made. Reuse is opt-in and now
    requires an exact, content-hashed request identity."""
    register_defaults() if "seasonal_naive" not in registry.names() else None
    out = pathlib.Path(tempfile.mkdtemp())
    first = ranking.run_over_origins("seasonal_naive", panel(), (DEV_ORIGINS[0],), 7, spark(), out)
    sidecar = out / "seasonal_naive" / f"{DEV_ORIGINS[0]}.json"
    assert sidecar.exists(), "no record was written, so nothing could be resumed"

    with mock.patch.object(ranking.registry, "run",
                           side_effect=AssertionError("exact reuse invoked the adapter")):
        resumed = ranking.run_over_origins(
            "seasonal_naive", panel(), (DEV_ORIGINS[0],), 7, spark(), out, reuse=True)
    assert resumed[DEV_ORIGINS[0]].path == first[DEV_ORIGINS[0]].path

    original_run = ranking.registry.run
    with mock.patch.object(ranking.registry, "run", wraps=original_run) as invoked:
        fresh = ranking.run_over_origins(
            "seasonal_naive", panel(), (DEV_ORIGINS[0],), 7, spark(), out)
        assert invoked.call_count == 1, "reuse is on by default, so the adapter was not invoked"
    assert fresh[DEV_ORIGINS[0]].path == first[DEV_ORIGINS[0]].path, \
        "a fresh run wrote somewhere other than the harness-owned target"


def test_pool_scores_reduces_by_raw_sums_not_by_averaging() -> None:
    """Two origins score different numbers of cells once an adapter declines, so averaging their
    MAEs is wrong. The pooled MAE must equal total abs error over total cells."""
    register_defaults() if "seasonal_naive" not in registry.names() else None
    out = pathlib.Path(tempfile.mkdtemp())
    runs = ranking.run_over_origins("seasonal_naive", panel(), (DEV_ORIGINS[0], ACC_ORIGIN), 7,
                                    spark(), out)
    pooled = ranking.pool_scores(panel(), runs, spark(), missing_actual="no_data")
    assert abs(pooled.mae - pooled.sum_abs_error / pooled.n_scored) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
