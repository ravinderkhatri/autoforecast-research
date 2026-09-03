"""Data-agnostic admission from source-column mapping to eligible model candidates."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, cast

from autoforecast._exact_yaml import ExactYAMLError, loads_exact_yaml
from autoforecast.adapter_contracts import DESCRIPTORS, descriptor
from autoforecast.adapters import adapter_capabilities, known_adapter_names
from autoforecast.frequency import FrequencyError, frequency_is_supported
from autoforecast.model_catalog import Candidate, CatalogError, SelectionPolicy, resolve_candidates
from autoforecast.panel import Panel, PanelError, PlanSource
from autoforecast.profiling import Profile

if TYPE_CHECKING:
    from autoforecast.data_observability import CheckReport

__all__ = [
    "AutoForecastPreflight",
    "CompatibilityReport",
    "DatasetContract",
    "DatasetContractError",
    "ModelInputContract",
    "get_model_input_contract",
    "load_dataset_contract",
    "preflight_candidates",
    "validate_model_compatibility",
]


class DatasetContractError(ValueError):
    """A source-column mapping is malformed or tries to supply a runtime data path."""


@dataclass(frozen=True)
class DatasetContract:
    """A versioned mapping from one source's names to generic Panel roles."""

    id: str
    panel_roles: dict[str, Any]
    source_path: pathlib.Path
    required_series_count: int | None = None

    def panel(self, path: str | pathlib.Path) -> Panel:
        """Build a Panel for an explicit runtime path; never commit the data location."""
        roles = dict(self.panel_roles)
        if type(roles.get("plan")) is dict:
            roles["plan"] = PlanSource(**roles["plan"])
        return Panel(path=str(path), **roles)


def load_dataset_contract(path: str | pathlib.Path) -> DatasetContract:
    """Load one strict, versioned source-column mapping."""
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise DatasetContractError(f"dataset contract must be a regular file, not {source}")
    return _load_dataset_contract_bytes(source.read_bytes(), source)


def _load_dataset_contract_bytes(raw: bytes, source: pathlib.Path) -> DatasetContract:
    """Parse dataset-contract bytes whose identity was authenticated by the caller."""
    try:
        value = loads_exact_yaml(raw)
    except ExactYAMLError as exc:
        raise DatasetContractError(f"invalid YAML in {source}: {exc}") from exc
    if type(value) is not dict or set(value) != {"schema_version", "dataset"}:
        raise DatasetContractError(
            "dataset contract must contain exactly schema_version and dataset"
        )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] not in (1, 2)
        or type(value["dataset"]) is not dict
    ):
        raise DatasetContractError(
            "dataset contract requires schema_version: 1 or 2 and a dataset mapping"
        )
    dataset = value["dataset"]
    expected = {"id", "panel"}
    if value["schema_version"] == 2:
        expected.add("required_series_count")
    if set(dataset) != expected or type(dataset["id"]) is not str or not dataset["id"]:
        raise DatasetContractError("dataset requires non-blank id and a panel mapping")
    required_series_count = dataset.get("required_series_count")
    if value["schema_version"] == 2 and (
        type(required_series_count) is not int or required_series_count < 1
    ):
        raise DatasetContractError("required_series_count must be a positive integer")
    roles = dataset["panel"]
    if type(roles) is not dict:
        raise DatasetContractError("dataset.panel must be a mapping")
    if any(type(key) is not str for key in roles):
        raise DatasetContractError("dataset.panel field names must be strings")
    if not {"series_col", "time_col", "target_col"} <= roles.keys():
        raise DatasetContractError("dataset.panel requires series_col, time_col and target_col")
    unknown = sorted(set(roles) - ({field.name for field in fields(Panel)} - {"path"}))
    if unknown or "path" in roles:
        raise DatasetContractError(f"dataset.panel has unknown or forbidden fields: {unknown}")
    normalised: dict[str, Any] = dict(roles)
    for name in ("known_future", "static", "static_reals", "past_only", "shared_known_future"):
        if name in normalised:
            if type(normalised[name]) is not list:
                raise DatasetContractError(f"dataset.panel.{name} must be a YAML list")
            normalised[name] = tuple(normalised[name])
    if "covariate_families" in normalised:
        families = normalised["covariate_families"]
        if type(families) is not list or any(type(family) is not list for family in families):
            raise DatasetContractError("dataset.panel.covariate_families must be a list of lists")
        normalised["covariate_families"] = tuple(tuple(family) for family in families)
    if normalised.get("plan") is not None and type(normalised["plan"]) is not dict:
        raise DatasetContractError("dataset.panel.plan must be a YAML mapping or null")
    contract = DatasetContract(dataset["id"], normalised, source, required_series_count)
    try:
        contract.panel(".").validate_column_names()
    except (PanelError, TypeError) as exc:
        raise DatasetContractError(str(exc)) from exc
    return contract


@dataclass(frozen=True)
class ModelInputContract:
    """Smallest declared input boundary for one registered adapter."""

    frequency: str
    history_rule: str
    supported_frequency_families: tuple[str, ...] = ("D",)
    supports_strided_daily: bool = False

    def minimum_history(self, *, season_length: int, horizon: int) -> int:
        """Return the adapter's implemented per-series observation floor."""
        if self.history_rule == "one":
            return 1
        if self.history_rule == "two":
            return 2
        if self.history_rule == "season":
            return season_length
        if self.history_rule == "auto_ets":
            return max(2 * season_length, 12)
        if self.history_rule == "tft":
            return 180 + horizon
        if self.history_rule == "tft_q":
            # Seven encoder warm-up days, 42-day target lag and two 28-day targets.
            return 7 + 42 + 2 * horizon
        if self.history_rule == "timesfm":
            return 1024
        if self.history_rule == "mft":
            return 28 + horizon
        raise ValueError(f"unknown history rule {self.history_rule!r}")

    def supports(self, frequency: str) -> bool:
        """Whether the adapter declares the cadence family and any non-unit daily stride."""
        return frequency_is_supported(
            frequency,
            self.supported_frequency_families,
            supports_strided_daily=self.supports_strided_daily,
        )


_CONTRACTS = {
    name: ModelInputContract(
        descriptor(name).frequency,
        descriptor(name).history,
        descriptor(name).supported_frequency_families,
        descriptor(name).supports_strided_daily,
    )
    for name in DESCRIPTORS
}


def get_model_input_contract(adapter: str) -> ModelInputContract:
    """Return the declared contract or reject an adapter without one."""
    try:
        return _CONTRACTS[adapter]
    except KeyError:
        raise ValueError(f"no input contract declared for adapter {adapter!r}") from None


@dataclass(frozen=True)
class CompatibilityReport:
    """Pre-fit result; adapters still account for individual declined series."""

    adapter: str
    status: str
    minimum_history: int
    reasons: tuple[str, ...]

    @property
    def may_run(self) -> bool:
        """Whether at least one series can reach the adapter without known contract failure."""
        return self.status in {"eligible", "eligible_with_declines"}


def validate_model_compatibility(
    adapter: str,
    panel: Panel,
    profile: Profile,
    *,
    horizon: int,
    data_checks: CheckReport | None = None,
) -> CompatibilityReport:
    """Check a profiled panel before importing an adapter to fit.

    When supplied, failed data checks stop model checks without copying generic data failures into
    a model-specific reason. The optional argument preserves the lightweight compatibility API for
    callers that have already performed the equivalent authoritative data gate.
    """
    if adapter not in known_adapter_names():
        raise ValueError(f"unknown adapter {adapter!r}; known={list(known_adapter_names())}")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    contract = get_model_input_contract(adapter)
    minimum_history = contract.minimum_history(season_length=panel.season_length, horizon=horizon)
    reasons: list[str] = []
    if data_checks is not None and not data_checks.accepted:
        reasons.append("Data checks failed. Model checks were not run.")
    try:
        panel.validate()
    except PanelError as exc:
        reasons.append(str(exc))
    try:
        frequency_supported = contract.supports(panel.frequency)
    except FrequencyError:
        frequency_supported = False
    if not frequency_supported:
        reasons.append(
            f"requires frequency family in {contract.supported_frequency_families!r}, "
            f"got {panel.frequency!r}"
        )
    supported_estimands = cast(frozenset[str], adapter_capabilities(adapter)["supported_estimands"])
    if panel.target_estimand not in supported_estimands:
        reasons.append(f"does not support target_estimand={panel.target_estimand!r}")
    if reasons:
        return CompatibilityReport(adapter, "ineligible", minimum_history, tuple(reasons))
    if profile.max_eligible_history < minimum_history:
        return CompatibilityReport(
            adapter,
            "ineligible",
            minimum_history,
            (
                f"no series has {minimum_history} eligible observations; "
                f"maximum is {profile.max_eligible_history}",
            ),
        )
    if profile.min_eligible_history < minimum_history:
        return CompatibilityReport(
            adapter,
            "eligible_with_declines",
            minimum_history,
            (
                f"some series have fewer than {minimum_history} eligible observations; "
                "adapter disposition accounting remains required",
            ),
        )
    return CompatibilityReport(adapter, "eligible", minimum_history, ())


@dataclass(frozen=True)
class AutoForecastPreflight:
    """Pre-ranking compatibility decision, with every exclusion explained."""

    policy_id: str
    candidates: tuple[Candidate, ...]
    reports: tuple[CompatibilityReport, ...]

    @property
    def eligible(self) -> tuple[Candidate, ...]:
        """Candidates that may run; adapters retain per-series disposition accounting."""
        pairs = zip(self.candidates, self.reports, strict=True)
        return tuple(candidate for candidate, report in pairs if report.may_run)


def preflight_candidates(
    candidates: tuple[Candidate, ...],
    policy: SelectionPolicy,
    panel: Panel,
    profile: Profile,
    *,
    horizon: int,
    data_checks: CheckReport | None = None,
) -> AutoForecastPreflight:
    """Select data-compatible candidates before fitting or scoring anything."""
    selected = resolve_candidates(candidates, policy)
    reports = tuple(
        validate_model_compatibility(
            candidate.adapter,
            panel,
            profile,
            horizon=horizon,
            data_checks=data_checks,
        )
        for candidate in selected
    )
    result = AutoForecastPreflight(policy.id, selected, reports)
    if len(result.eligible) < policy.top_k:
        raise CatalogError(
            f"selection {policy.id!r} has {len(result.eligible)} data-compatible candidates, "
            f"fewer than top_k={policy.top_k}"
        )
    return result
