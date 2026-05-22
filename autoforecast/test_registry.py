"""Tests for the registry. Every case here is about the CONTROL, not about data.

SYNTHETIC BY NECESSITY: there is no real instance of a
registration refusal. The adapters under test are stubs because what is being checked is what the
registry refuses to accept, not what any forecaster computes.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast import registry
from autoforecast.panel import Forecast, Panel
from autoforecast.resources import SparkProfile

PANEL = Panel("/tmp/unused", "id", "date", "qty")
FULL = {"consumes_exposure": True, "handles_right_censoring": True,
        "supported_estimands": frozenset({"sales_given_availability",
                                           "unconstrained_demand"})}
LIMITED = {"consumes_exposure": False, "handles_right_censoring": False,
           "supported_estimands": frozenset({"sales_given_availability"})}


WORKER_PROFILE = SparkProfile(
    cores=2,
    driver_memory="4g",
    shuffle_partitions=32,
    worker_reuse=True,
    series_per_partition=100,
    python_worker_budget_gb=4.0,
)


def stub(name: str):
    """An adapter that returns a well-formed Forecast labelled `name`."""
    def fn(panel: Panel, origin: str, horizon: int, **kw: object) -> Forecast:
        return Forecast(name, "/tmp/f", origin, horizon, 1, {"emitted": horizon},
                        target_estimand=panel.target_estimand,
                        frequency=panel.frequency).validate()
    return fn


def fresh() -> None:
    registry._reset_for_test()


def test_generic_observed_target_is_supported_only_by_declared_adapters() -> None:
    from dataclasses import replace

    from autoforecast.adapters import _registration

    fresh()
    panel = replace(PANEL, target_estimand="observed_nonnegative")
    for name in ("last_value", "seasonal_naive"):
        registry.register(name, stub(name), **_registration(name))
        result = registry.run(name, panel, "2024-01-01", 7)
        assert result.target_estimand == "observed_nonnegative"
    registry.register("tft_hnb", stub("tft_hnb"), **_registration("tft_hnb"))
    try:
        registry.run("tft_hnb", panel, "2024-01-01", 7)
    except registry.RegistryError as error:
        assert "does not support target_estimand" in str(error)
    else:
        raise AssertionError("retail-only adapter accepted a generic measurement")
    fresh()


def test_register_then_get_returns_the_entry() -> None:
    fresh()
    registry.register("a", stub("a"), **FULL)
    assert registry.get("a").name == "a"
    assert registry.get("a").consumes_exposure


def test_register_refuses_a_duplicate_name() -> None:
    fresh()
    registry.register("a", stub("a"), **FULL)
    try:
        registry.register("a", stub("a"), **FULL)
    except registry.RegistryError as e:
        assert "already registered" in str(e); return
    raise AssertionError("registered the same name twice")


def test_register_refuses_a_non_conforming_adapter_with_no_handicap() -> None:
    # This is the control that makes "handicaps recorded" real rather than aspirational.
    fresh()
    try:
        registry.register("zoo", stub("zoo"), **LIMITED)
    except registry.RegistryError as e:
        assert "may not compete silently" in str(e); return
    raise AssertionError("let an adapter that ignores exposure compete without saying so")


def test_register_refuses_a_handicap_on_a_conforming_adapter() -> None:
    fresh()
    try:
        registry.register("a", stub("a"), **FULL, handicap="none really")
    except registry.RegistryError as e:
        assert "readers learn to skip" in str(e); return
    raise AssertionError("accepted a handicap that says nothing -- the field stops being read")


def test_register_refuses_unconstrained_demand_without_censoring() -> None:
    fresh()
    try:
        registry.register(
            "pretender", stub("pretender"), consumes_exposure=True,
            handles_right_censoring=False,
            supported_estimands=frozenset({"unconstrained_demand"}),
            handicap="treats stockout sales as exact",
        )
    except registry.RegistryError as exc:
        assert "without right-censor handling" in str(exc)
        return
    raise AssertionError("registered latent demand support without a censored likelihood")


def test_register_refuses_malformed_capability_metadata() -> None:
    cases = (
        dict(name="", forecast=stub("x"), consumes_exposure=True,
             handles_right_censoring=True, supported_estimands=FULL["supported_estimands"]),
        dict(name="not_callable", forecast=None, consumes_exposure=True,
             handles_right_censoring=True, supported_estimands=FULL["supported_estimands"]),
        dict(name="integer_bool", forecast=stub("integer_bool"), consumes_exposure=1,
             handles_right_censoring=True, supported_estimands=FULL["supported_estimands"]),
        dict(name="blank_handicap", forecast=stub("blank_handicap"), consumes_exposure=False,
             handles_right_censoring=False, supported_estimands=LIMITED["supported_estimands"],
             handicap="  "),
        dict(name="integer_stride", forecast=stub("integer_stride"), **FULL,
             supports_strided_daily=1),
    )
    for case in cases:
        fresh()
        try:
            registry.register(**case)
        except registry.RegistryError:
            continue
        raise AssertionError(f"registered malformed capability metadata: {case!r}")


def test_names_is_sorted_so_a_manifest_is_reproducible() -> None:
    fresh()
    for n in ("zoo", "a", "m"):
        registry.register(n, stub(n), **FULL)
    assert registry.names() == ("a", "m", "zoo")


def test_handicaps_returns_only_the_declared_ones() -> None:
    fresh()
    registry.register("clean", stub("clean"), **FULL)
    registry.register("zoo", stub("zoo"), **LIMITED,
                      handicap="fills missing values; no cell-level loss exclusion")
    assert registry.handicaps(("clean", "zoo")) == {
        "zoo": "fills missing values; no cell-level loss exclusion"}
    assert registry.handicaps(("clean",)) == {}


def test_get_names_what_is_registered_when_the_lookup_fails() -> None:
    fresh()
    registry.register("a", stub("a"), **FULL)
    try:
        registry.get("b")
    except registry.RegistryError as e:
        assert "unknown adapter 'b'" in str(e) and "('a',)" in str(e); return
    raise AssertionError("a bare lookup failure sends the reader to the wrong file")


def test_run_refuses_a_forecast_labelled_with_another_adapter() -> None:
    # Nothing downstream could detect this: the score would be attributed to the wrong model.
    fresh()
    registry.register("a", stub("impostor"), **FULL)
    try:
        registry.run("a", PANEL, "2016-04-24", 7)
    except registry.RegistryError as e:
        assert "labelled 'impostor'" in str(e); return
    raise AssertionError("accepted a Forecast labelled with a different adapter")


def test_run_passes_through_a_correctly_labelled_forecast() -> None:
    fresh()
    registry.register("a", stub("a"), **FULL)
    out = registry.run("a", PANEL, "2016-04-24", 7)
    assert out.adapter == "a" and out.horizon == 7 and out.n_emitted == 7


@pytest.mark.parametrize(
    "frequency,canonical,origin",
    [
        ("H", "H", "2024-03-01T22:00:00"),
        ("D", "D", "2024-03-01"),
        ("1D", "D", "2024-03-01"),
        ("W", "W", "2024-02-28"),
        ("M", "M", "2024-02-29"),
        ("ME", "M", "2024-02-29"),
        ("MS", "MS", "2024-03-01"),
        ("Y", "Y", "2023-12-31"),
        ("YE", "Y", "2023-12-31"),
        ("YS", "YS", "2024-01-01"),
    ],
)
def test_run_allows_explicitly_supported_frequency_families(
    frequency, canonical, origin,
) -> None:
    fresh()
    registry.register(
        "a",
        stub("a"),
        **FULL,
        supported_frequency_families=frozenset({"H", "D", "W", "M", "Y"}),
    )
    out = registry.run(
        "a", Panel("/tmp/unused", "id", "date", "qty", frequency=frequency), origin, 2
    )
    assert out.frequency == canonical and out.origin == origin
    fresh()


def test_run_refuses_noncanonical_span_before_adapter_execution() -> None:
    fresh()
    called = False

    def must_not_run(*args: object, **kwargs: object) -> Forecast:
        nonlocal called
        called = True
        raise AssertionError("invalid request reached adapter execution")

    registry.register("a", must_not_run, **FULL)
    for origin, horizon in (("20160424", 7), ("2016-04-24", True)):
        try:
            registry.run("a", PANEL, origin, horizon)
        except registry.RegistryError as exc:
            assert "canonical" in str(exc) or "horizon" in str(exc)
        else:
            raise AssertionError(f"accepted origin={origin!r}, horizon={horizon!r}")
    assert not called


def test_run_refuses_an_adapter_that_changes_the_panel_cadence() -> None:
    fresh()
    registry.register("a", stub("a"), **FULL)
    weekly = Panel("/tmp/unused", "id", "date", "qty", frequency="W")
    try:
        registry.run("a", weekly, "2016-04-24", 2)
    except registry.RegistryError as exc:
        assert "frequency" in str(exc) and "'W'" in str(exc) and "'D'" in str(exc)
        return
    raise AssertionError("accepted daily output for a weekly panel")


def test_run_requires_explicit_capability_for_a_nonunit_daily_stride() -> None:
    panel = Panel("/tmp/unused", "id", "date", "qty", frequency="7D")
    fresh()
    registry.register("exact_daily", stub("exact_daily"), **FULL)
    with pytest.raises(registry.RegistryError, match="supports_strided_daily=False"):
        registry.run("exact_daily", panel, "2024-01-01", 2)
    fresh()
    registry.register(
        "strided_daily", stub("strided_daily"), **FULL, supports_strided_daily=True
    )
    assert registry.run("strided_daily", panel, "2024-01-01", 2).frequency == "7D"
    fresh()


def test_run_refuses_an_adapter_that_changes_the_requested_forecast_span() -> None:
    fresh()

    def shifted(panel: Panel, origin: str, horizon: int, **kw: object) -> Forecast:
        return Forecast(
            "shifted", "/tmp/f", "2016-04-25", horizon + 1, 1,
            {"emitted": horizon + 1}, target_estimand=panel.target_estimand,
        )

    registry.register("shifted", shifted, **FULL)
    try:
        registry.run("shifted", PANEL, "2016-04-24", 7)
    except registry.RegistryError as error:
        assert "returned span" in str(error)
        return
    raise AssertionError("accepted a forecast for a different origin/horizon")


def test_run_refuses_a_non_forecast_return_value() -> None:
    fresh()
    registry.register("bad", lambda *args, **kwargs: {"adapter": "bad"}, **FULL)
    try:
        registry.run("bad", PANEL, "2016-04-24", 1)
    except registry.RegistryError as exc:
        assert "expected a Forecast" in str(exc)
        return
    raise AssertionError("accepted an adapter return without the Forecast contract")


def test_run_refuses_an_estimand_the_adapter_does_not_support() -> None:
    fresh()
    registry.register("sales_only", stub("sales_only"), **LIMITED,
                      handicap="observed sales only")
    latent_panel = Panel("/tmp/unused", "id", "date", "qty", exposure_col="state",
                         target_estimand="unconstrained_demand")
    try:
        registry.run("sales_only", latent_panel, "2016-04-24", 7)
    except registry.RegistryError as exc:
        assert "does not support target_estimand 'unconstrained_demand'" in str(exc)
        return
    raise AssertionError("ran an observed-sales adapter for latent demand")


def test_register_defaults_registers_the_naive_adapter() -> None:
    fresh()
    from autoforecast.adapters import register_defaults
    assert "seasonal_naive" in register_defaults()
    entry = registry.get("seasonal_naive")
    assert entry.consumes_exposure and not entry.handles_right_censoring
    assert entry.supported_estimands == frozenset({"sales_given_availability", "observed_nonnegative"})
    assert entry.handicap


def test_isolated_registration_does_not_import_native_adapter_modules() -> None:
    """LightGBM and torch must not enter the gate parent merely because names were registered."""
    import sys
    fresh()
    before = set(sys.modules)
    from autoforecast.adapters import register_isolated
    register_isolated(("shallow_global", "mft"), worker_profile=WORKER_PROFILE)
    added = set(sys.modules) - before
    assert "autoforecast.adapters.shallow" not in added
    assert "autoforecast.adapters.mft" not in added
    assert registry.names() == ("mft", "shallow_global")


def test_isolated_registration_refuses_an_unmapped_name() -> None:
    fresh()
    from autoforecast.adapters import register_isolated
    try:
        register_isolated(("a-model-that-has-no-worker",), worker_profile=WORKER_PROFILE)
    except ValueError as exc:
        assert "no isolated adapter mapping" in str(exc); return
    raise AssertionError("registered an adapter the isolated worker cannot load")


def test_isolated_zoo_retains_its_exposure_handicap() -> None:
    """A process boundary must not relabel a nonconforming comparator as conforming."""
    fresh()
    from autoforecast.adapters import register_isolated
    register_isolated(("autogluon_zoo",), worker_profile=WORKER_PROFILE)
    entry = registry.get("autogluon_zoo")
    assert not entry.consumes_exposure
    assert entry.handicap and "no cell-level loss exclusion" in entry.handicap


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
