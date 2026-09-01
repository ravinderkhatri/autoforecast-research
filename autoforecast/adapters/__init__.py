"""Adapters, and the one place they are registered.

**NOTHING IS IMPORTED AT PACKAGE LEVEL**, and that is not tidiness. Every adapter module pulls a
different heavy dependency -- lightgbm, statsforecast, torch, pytorch-forecasting, autogluon -- and
importing them all to reach one of them has two costs, one of which is a crash:

  · **On this platform, LightGBM and torch segfault together.** Measured: `shallow.fit_predict`
    completes normally when torch is absent from `sys.modules` and dies with SIGSEGV when it is
    present. Both ship their own OpenMP runtime, and loading two of them into one process is a
    known hazard. Importing `mft` (torch) from this file made every LightGBM
    run inherit it, and `test_shallow.py` went from 8 passing to a segfault with no assertion and
    no traceback.
  · **The optional extras stop being optional.** `zoo` needs autogluon and `tft` needs
    pytorch-forecasting, neither of which is a core dependency -- a client using only the naive
    adapter should not need either installed to import this package.

So registration imports each module INSIDE the function, one at a time, and an adapter whose
dependency is missing is skipped and REPORTED rather than taking the package down with it.
"""

from __future__ import annotations

import functools
import importlib

from autoforecast import registry
from autoforecast.adapter_contracts import DESCRIPTORS, descriptor
from autoforecast.frequency import advance, frequency_is_supported, parse_frequency, parse_origin
from autoforecast.panel import Panel
from autoforecast.resources import SparkProfile
from autoforecast.tracking_plan import tracking_plan

__all__ = [
    "adapter_capabilities",
    "known_adapter_names",
    "register_defaults",
    "register_isolated",
    "validate_execution_controls",
]


_ISOLATED_MODULE = {name: descriptor(name).module for name in DESCRIPTORS}


def _validate_span(panel: Panel, origin: str, horizon: int, *, adapter: str):
    """Fail before runtime work when an adapter cannot represent the requested period span."""
    cadence = parse_frequency(panel.frequency)
    definition = descriptor(adapter)
    supported = definition.supported_frequency_families
    if not frequency_is_supported(
        cadence,
        supported,
        supports_strided_daily=definition.supports_strided_daily,
    ):
        raise ValueError(
            f"{adapter} supports frequency families {supported} with "
            f"supports_strided_daily={definition.supports_strided_daily}, "
            f"got {panel.frequency!r}"
        )
    if type(horizon) is not int or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    try:
        origin_point = parse_origin(origin, cadence)
        advance(origin_point, cadence, horizon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "origin and horizon must produce a canonical representable "
            f"{panel.frequency!r} span, got "
            f"origin={origin!r}, horizon={horizon!r}"
        ) from exc
    return origin_point


def _validate_daily_span(panel: Panel, origin: str, horizon: int, *, adapter: str):
    """Compatibility validator for implementations that still require exact daily input."""
    cadence = parse_frequency(panel.frequency)
    if cadence.unit != "D" or cadence.multiplier != 1:
        raise ValueError(
            f"{adapter} supports exact daily panels only, got frequency {panel.frequency!r}"
        )
    return _validate_span(panel, origin, horizon, adapter=adapter)


def known_adapter_names() -> tuple[str, ...]:
    """Names the catalog may reference without importing any optional adapter runtime.

    This is intentionally not ``registry.names()``: the latter is process state after loading
    dependencies, whereas validating a declarative catalog must be possible before any native
    runtime (notably torch or LightGBM) is imported.
    """
    return tuple(sorted(_ISOLATED_MODULE))


def adapter_capabilities(name: str) -> dict[str, object]:
    """Return dependency-free registration metadata for pre-fit selection tooling."""
    if name not in _ISOLATED_MODULE:
        raise ValueError(f"unknown adapter {name!r}; known={list(known_adapter_names())}")
    return {**_registration(name), "tracking": tracking_plan(name), "descriptor": descriptor(name)}


def validate_execution_controls(
    name: str,
    *,
    device: str | None,
    batch_size: int,
    gradient_microbatch_size: int | None,
    has_generic_events: bool,
    precision: str = "32-true",
) -> None:
    """Reject common controls that an adapter cannot execute before allocating resources."""
    if name not in _ISOLATED_MODULE:
        raise ValueError(f"unknown adapter {name!r}; known={list(known_adapter_names())}")
    if name in {"timesfm_hnb", "timesfm_v8"}:
        if precision != "32-true" or device not in (None, "auto", "cpu", "cuda", "mps"):
            raise ValueError("TimesFM-HNB requires 32-true precision and a supported source device")
        if gradient_microbatch_size not in {None, batch_size}:
            raise ValueError(
                "TimesFM-HNB has no gradient accumulation; microbatch must equal batch_size"
            )
        if has_generic_events:
            raise ValueError(
                "TimesFM-HNB consumes numeric event fields through declared known_future roles"
            )
        return
    if name in {"tft", "tft_q", "tft_hnb"} and precision == "16-mixed" and device == "cpu":
        raise ValueError(f"{name} float16 mixed precision needs CUDA or MPS hardware")
    if name not in {
        "sft",
        "hmfnet",
        "pact",
    }:
        return
    if device not in (None, "cpu", "cuda"):
        raise ValueError(f"{name} currently supports automatic, cpu, or cuda device")
    if gradient_microbatch_size not in {None, batch_size}:
        raise ValueError(
            f"{name} has no gradient accumulation; microbatch must be null or equal batch_size"
        )
    if has_generic_events:
        raise ValueError(
            f"{name} requires a versioned calendar input and cannot consume generic events"
        )

# Contract metadata belongs in this dependency-free registration module. Importing ``zoo`` just
# to retrieve it also imports PySpark and defeats the point of parent-side isolated registration.
_AUTOGLUON_ZOO_HANDICAP = (
    "no cell-level loss exclusion; no censored observations; FILLS missing values, so a "
    "never-ranged cell enters training as an observed zero. Its own model selection also scores "
    "on a single trailing validation window, which is why only our harness's origins decide "
    "anything here."
)

_NO_CENSOR_HANDICAP = (
    "filters ineligible cells but treats positive stockout sales as exact observations; no "
    "right-censored likelihood, so it supports observed sales only"
)
_NAIVE_CENSOR_HANDICAP = (
    "copies a lower-bound source value and labels it biased-low; no right-censored likelihood, "
    "so it forecasts observed quantities and cannot recover latent demand"
)
_PACT_SALES_HANDICAP = (
    "records eligible lower-bound sales as realised sales for the "
    "sales_given_availability estimand; it does not identify unconstrained censored demand"
)


def _registration(name: str) -> dict[str, object]:
    """Dependency-free capability metadata shared by local and isolated registration."""
    definition = descriptor(name)
    result = {
        "consumes_exposure": definition.consumes_exposure,
        "handles_right_censoring": definition.handles_right_censoring,
        "supported_estimands": frozenset(definition.target_estimands),
        "supported_frequency_families": frozenset(
            definition.supported_frequency_families
        ),
        "supports_strided_daily": definition.supports_strided_daily,
    }
    if definition.handles_right_censoring:
        return result
    if name == "pact":
        handicap = _PACT_SALES_HANDICAP
    elif name == "autogluon_zoo":
        handicap = _AUTOGLUON_ZOO_HANDICAP
    elif name in {"seasonal_naive", "last_value"}:
        handicap = _NAIVE_CENSOR_HANDICAP
    else:
        handicap = _NO_CENSOR_HANDICAP
    return {**result, "handicap": handicap}


def register_isolated(names: tuple[str, ...], *, worker_profile: SparkProfile) -> tuple[str, ...]:
    """Register adapters as subprocess-backed entries without importing their native runtimes.

    This is the safe registration path for a comparison containing both LightGBM and torch.  The
    two OpenMP runtimes abort when loaded into one Python process on the measured platform; lazy
    imports alone do not help because the first native library remains loaded when the second
    adapter runs.  Each invocation therefore gets a fresh worker process and its own Spark session,
    while the parent process retains the common scoring/ranking code.
    """
    unknown = sorted(set(names) - set(_ISOLATED_MODULE))
    if unknown:
        raise ValueError(
            f"no isolated adapter mapping for {unknown}; known: {sorted(_ISOLATED_MODULE)}")
    if not isinstance(worker_profile, SparkProfile):
        raise TypeError(
            "register_isolated requires an explicit worker_profile; the live scoring parent and "
            "the child cannot both be sized from the whole machine")
    from autoforecast.adapters import isolated
    for name in names:
        registry.register(
            name,
            functools.partial(isolated.run, name, worker_profile=worker_profile),
            **_registration(name),
        )
    return registry.names()


def register_defaults(*, quiet: bool = False,
                      only: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Register every adapter whose dependencies are present. Returns the registered names.

    Args:
        quiet: Suppress the report of adapters that could not be registered. **Off by default** --
            an adapter silently missing from a comparison is a comparison with a hole in it, and
            the hole is invisible in the results table.
        only: Register just these, or None for all. **This exists because importing an adapter
            has side effects on the whole process, not because it is tidier.** The module
            docstring above describes LightGBM and torch sharing a process; lazy imports keep
            that out of `import autoforecast.adapters`, but registering BOTH still loads both,
            and the failure is:

                OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
                initialized.

            after which LightGBM aborts mid-train. A phase that does not need
            `mft` must not
            register it, and Phase 2 does not: its comparators are the classical models and the
            shallow global one. The documented `KMP_DUPLICATE_LIB_OK=TRUE` escape is described by
            OpenMP itself as able to "silently produce incorrect results", which is disqualifying
            for a gate -- a wrong number that arrives quietly is worse than a crash.

    Returns:
        The registered names, sorted.

    Raises:
        ValueError: `only` names an adapter that does not exist, which would otherwise leave a
            comparison quietly short of a comparator.
    """
    missing: dict[str, str] = {}

    def _try(name: str, load):
        try:
            load()
        except ImportError as exc:                       # an optional extra is not installed
            missing[name] = str(exc).split("\n")[0]

    def _naive():
        from autoforecast.adapters import naive
        registry.register(naive.NAME, naive.seasonal_naive, **_registration(naive.NAME))

    def _classical():
        from autoforecast.adapters import classical
        for name in sorted(classical.MODELS):
            registry.register(name, functools.partial(classical.forecast_with, name),
                              **_registration(name))

    def _shallow():
        from autoforecast.adapters import shallow
        registry.register(shallow.NAME, shallow.fit_predict, **_registration(shallow.NAME))

    def _tft():
        # TFT filters to eligible cells like the classical adapters, so it HONOURS the exposure
        # eligibility half of the exposure contract, but not right-censoring. That distinction is
        # now registered independently; Phase 3 additionally splits TFT's broader limitations
        # into structural and configuration handicaps.
        from autoforecast.adapters import tft
        registry.register(tft.NAME, tft.fit_predict, **_registration(tft.NAME))

    def _tft_q():
        from autoforecast.adapters import tft_q
        registry.register("tft_q", tft_q.fit_predict_quantile, **_registration("tft_q"))
        registry.register("tft_hnb", tft_q.fit_predict_hnb, **_registration("tft_hnb"))

    def _timesfm_hnb():
        from autoforecast.adapters import timesfm_hnb

        registry.register(
            timesfm_hnb.NAME, timesfm_hnb.fit_predict, **_registration(timesfm_hnb.NAME)
        )

    def _timesfm_v8():
        from autoforecast.adapters import timesfm_v8
        registry.register(timesfm_v8.NAME, timesfm_v8.fit_predict, **_registration(timesfm_v8.NAME))

    def _zoo():
        # AutoGluon genuinely cannot honour the contract: no cell-level loss exclusion, no
        # censored observations, and it FILLS missing values. It may compete; registration
        # refuses it without the handicap, so it cannot compete silently.
        from autoforecast.adapters import zoo
        registry.register(zoo.NAME, zoo.fit_predict, **_registration(zoo.NAME))

    def _mft():
        from autoforecast.adapters import mft
        registry.register(
            mft.NAME,
            mft.fit_predict,
            **_registration(mft.NAME),
        )
        registry.register(
            mft.SFT_NAME,
            functools.partial(
                mft.fit_predict_revision_adapter,
                expected_variant="v2",
            ),
            **_registration(mft.SFT_NAME),
        )
        registry.register(
            mft.HMFNET_NAME,
            functools.partial(
                mft.fit_predict_revision_adapter,
                expected_variant="v3",
            ),
            **_registration(mft.HMFNET_NAME),
        )
        registry.register(
            mft.PACT_NAME,
            functools.partial(
                mft.fit_predict_revision_adapter,
                expected_variant="v5",
            ),
            **_registration(mft.PACT_NAME),
        )

    def _foundation():
        from autoforecast.adapters import foundation
        # Registration is the dependency preflight for an in-process run.  Isolated parent-side
        # registration remains dependency-free; its worker performs this check after crossing the
        # process boundary.
        foundation._provider_module()
        registry.register(
            foundation.NAME,
            foundation.fit_predict,
            **_registration(foundation.NAME),
        )

    def _descriptor_loader(name):
        definition = descriptor(name)
        module = importlib.import_module("autoforecast.adapters." + definition.module)
        registry.register(name, getattr(module, definition.callable), **_registration(name))

    loaders = (
        ("naive", _naive), ("classical", _classical), ("shallow", _shallow),
        ("tft", _tft), ("tft_q", _tft_q), ("timesfm_hnb", _timesfm_hnb),
        ("timesfm_v8", _timesfm_v8), ("zoo", _zoo), ("foundation", _foundation),
        ("mft", _mft),
    )
    legacy_modules = {name for name, _ in loaders}
    loaders += tuple(
        (definition.module, functools.partial(_descriptor_loader, name))
        for name, definition in DESCRIPTORS.items() if definition.module not in legacy_modules
    )
    if only is not None:
        known = {n for n, _ in loaders}
        unknown = sorted(set(only) - known)
        if unknown:
            raise ValueError(
                f"unknown adapter module(s) {unknown}; known: {sorted(known)}. A comparison "
                "silently short of a comparator is one with a hole in it")
        loaders = tuple((n, f) for n, f in loaders if n in only)
    for name, load in loaders:
        _try(name, load)

    if missing and not quiet:
        for name, why in sorted(missing.items()):
            print(f"  adapter NOT registered: {name} -- {why}")
    return registry.names()
