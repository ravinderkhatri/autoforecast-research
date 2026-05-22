"""name -> adapter, and the handicaps each one must declare before it may compete.

The registry exists so that adding a comparator is a declaration rather than a code change in the
scoring path. Its enforcement is capability-based: eligibility filtering, right-censor handling,
and the target estimands an adapter can produce are separate declarations. An adapter with a
limitation may still compete on a compatible estimand, but it may not compete silently.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

from autoforecast.panel import Forecast, Panel, PanelError, period_days

ForecastFn = Callable[..., Forecast]


class RegistryError(ValueError):
    """An adapter was registered twice, is unknown, or competes without declaring a handicap."""


@dataclass(frozen=True)
class Entry:
    """One registered implementation and what it admits about itself.

    Attributes:
        name: Registry key. Also the value written to `Forecast.adapter`.
        forecast: Callable taking `(panel, origin, horizon, **kwargs)` and returning a `Forecast`.
        consumes_exposure: Whether it reads `Panel.exposure_col` and honours `ELIGIBLE_STATES`.
            False is legitimate -- `autogluon` has no cell-level loss exclusion and fills missing
            values -- but it changes what a comparison means, so it forces `handicap`.
        handles_right_censoring: Whether lower-bound observations reach a censored likelihood
            rather than being treated as exact sales or discarded.
        supported_estimands: Quantities this adapter can honestly emit. Supporting
            ``unconstrained_demand`` requires right-censor handling.
        handicap: What this adapter cannot honour, in words that belong in a results table.
            Required for either exposure or censoring limitations; refused for an adapter
            declaring both capabilities.
    """

    name: str
    forecast: ForecastFn
    consumes_exposure: bool
    handles_right_censoring: bool
    supported_estimands: frozenset[str]
    handicap: str | None = None


_ENTRIES: dict[str, Entry] = {}


def register(
    name: str,
    forecast: ForecastFn,
    *,
    consumes_exposure: bool,
    handles_right_censoring: bool,
    supported_estimands: frozenset[str],
    handicap: str | None = None,
) -> Entry:
    """Add an implementation to the registry.

    Args:
        name: Registry key, unique.
        forecast: `(panel, origin, horizon, **kwargs) -> Forecast`.
        consumes_exposure: Whether it honours the exposure contract. Keyword-only, because a
            positional bool at a call site is exactly the argument nobody reads.
        handles_right_censoring: Whether it models ``lower_bound`` as censored evidence.
        supported_estimands: Non-empty subset of the declared target meanings.
        handicap: Required when either exposure or right-censoring capability is absent.

    Returns:
        The stored `Entry`.

    Raises:
        RegistryError: the name is already taken or the capability declaration is inconsistent.
    """
    if not isinstance(name, str) or not name.strip():
        raise RegistryError("adapter name must be a non-blank string")
    if not callable(forecast):
        raise RegistryError(f"{name!r}: forecast must be callable")
    if not isinstance(consumes_exposure, bool) or not isinstance(handles_right_censoring, bool):
        raise RegistryError(
            f"{name!r}: consumes_exposure and handles_right_censoring must be booleans")
    if handicap is not None and (not isinstance(handicap, str) or not handicap.strip()):
        raise RegistryError(f"{name!r}: handicap must be a non-blank string or None")
    if name in _ENTRIES:
        raise RegistryError(f"adapter already registered: {name!r}")
    valid_estimands = {"sales_given_availability", "unconstrained_demand", "observed_nonnegative"}
    if (not isinstance(supported_estimands, frozenset) or not supported_estimands
            or not supported_estimands <= valid_estimands):
        raise RegistryError(
            f"{name!r}: supported_estimands must be a non-empty frozenset within "
            f"{sorted(valid_estimands)}, got {supported_estimands!r}")
    if "unconstrained_demand" in supported_estimands and not handles_right_censoring:
        raise RegistryError(
            f"{name!r} claims unconstrained_demand without right-censor handling. Filtering "
            "eligible cells does not recover demand suppressed by stockouts")
    limited = not consumes_exposure or not handles_right_censoring
    if limited and not handicap:
        raise RegistryError(
            f"{name!r} has an exposure/censoring limitation and must declare a handicap. An "
            "adapter may compete on a compatible estimand; it may not compete silently"
        )
    if not limited and handicap:
        raise RegistryError(
            f"{name!r} declares full exposure and censoring capability, so a handicap would be "
            "noise -- and a field that is sometimes noise is a field readers learn to skip"
        )
    entry = Entry(name=name, forecast=forecast, consumes_exposure=consumes_exposure,
                  handles_right_censoring=handles_right_censoring,
                  supported_estimands=supported_estimands, handicap=handicap)
    _ENTRIES[name] = entry
    return entry


def get(name: str) -> Entry:
    """Look up a registered adapter.

    Raises:
        RegistryError: no adapter of that name, listing what is registered -- a bare KeyError
            here sends the reader to the wrong file.
    """
    try:
        return _ENTRIES[name]
    except KeyError:
        raise RegistryError(f"unknown adapter {name!r}; registered: {names()}") from None


def names() -> tuple[str, ...]:
    """Every registered adapter name, sorted, so a run manifest is reproducible."""
    return tuple(sorted(_ENTRIES))


def handicaps(among: tuple[str, ...]) -> dict[str, str]:
    """Handicaps that must be printed alongside any comparison of `among`.

    Args:
        among: Adapter names being compared.

    Returns:
        name -> handicap, for those that declared one. Empty means every adapter in the
        comparison honours the exposure contract.

    Raises:
        RegistryError: a name is not registered.
    """
    return {n: get(n).handicap for n in among if get(n).handicap}


def run(name: str, panel: Panel, origin: str, horizon: int, **kwargs: object) -> Forecast:
    """Run one registered adapter and check it returned what it claims.

    Verifies the returned forecast matches the requested adapter and temporal span. A mislabeled
    or shifted output would otherwise attribute a score to the wrong experiment.

    Raises:
        RegistryError: the adapter is unknown, or returned a `Forecast` labelled with a
            different adapter name.
    """
    entry = get(name)
    if panel.target_estimand not in entry.supported_estimands:
        raise RegistryError(
            f"adapter {name!r} does not support target_estimand {panel.target_estimand!r}; "
            f"supported={sorted(entry.supported_estimands)}")
    if type(horizon) is not int or horizon < 1:
        raise RegistryError(f"horizon must be a positive integer, got {horizon!r}")
    try:
        origin_day = dt.date.fromisoformat(origin)
        stride = period_days(panel.frequency)
        origin_day + dt.timedelta(days=horizon * stride)
    except (TypeError, ValueError, OverflowError, PanelError) as exc:
        raise RegistryError("origin/frequency does not define a supported forecast span") from exc
    if origin_day.isoformat() != origin:
        raise RegistryError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}")
    out = entry.forecast(panel, origin, horizon, **kwargs)
    if not isinstance(out, Forecast):
        raise RegistryError(
            f"adapter {name!r} returned {type(out).__name__}, expected a Forecast contract")
    out = out.validate()
    if out.adapter != name:
        raise RegistryError(
            f"adapter {name!r} returned a Forecast labelled {out.adapter!r}; a mislabelled "
            "result attributes the score to the wrong implementation"
        )
    if out.target_estimand != panel.target_estimand:
        raise RegistryError(
            f"adapter {name!r} returned target_estimand {out.target_estimand!r}, but its input "
            f"panel requested {panel.target_estimand!r}")
    if out.frequency != panel.frequency:
        raise RegistryError(
            f"adapter {name!r} returned frequency {out.frequency!r}, but its input panel "
            f"requested {panel.frequency!r}")
    if out.origin != origin or out.horizon != horizon:
        raise RegistryError(
            f"adapter {name!r} returned span origin={out.origin!r}, horizon={out.horizon!r}; "
            f"requested origin={origin!r}, horizon={horizon!r}"
        )
    return out


def _reset_for_test() -> None:
    """Clear the registry. Test-only: module-level state otherwise leaks between test cases."""
    _ENTRIES.clear()


__all__ = ["Entry", "RegistryError", "get", "handicaps", "names", "register", "run"]
