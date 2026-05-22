"""Small, dependency-free records shared by data checks and model readiness.

Model authors only need :class:`ModelCheckInput` and :class:`CheckResult`.  Spark reads,
storage and framework-specific objects stay in the admission service.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from autoforecast.profiling import Profile

__all__ = [
    "CheckReport",
    "CheckResult",
    "CheckStatus",
    "DataFacts",
    "ModelCheckInput",
    "ResolvedRun",
    "RunApproval",
    "RunIncident",
]

CheckStatus = Literal["passed", "failed", "warning", "not_checked", "not_applicable"]


@dataclass(frozen=True)
class CheckResult:
    """One user-readable data or model readiness result."""

    id: str
    name: str
    status: CheckStatus
    summary: str
    scope: str = "dataset"
    numerator: int | None = None
    denominator: int | None = None
    technical_detail: str | None = None

    @property
    def blocks_run(self) -> bool:
        return self.status == "failed"

    def as_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "scope": self.scope,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "technical_detail": self.technical_detail,
        }


@dataclass(frozen=True)
class DataFacts:
    """Source-bound measurements made once for a dataset and as-of date."""

    as_of: str
    data_fingerprint: str
    data_range: tuple[str, str]
    n_rows: int
    n_series: int
    n_dates: int
    duplicate_series_timestamps: int
    missing_daily_timestamps: int | None
    null_counts: dict[str, int]
    inconsistent_static_series: dict[str, int]
    profile: Profile | None
    spark_schema: dict[str, str]
    missing_timestamps: int | None = None

    def as_record(self) -> dict[str, object]:
        record = {
            "as_of": self.as_of,
            "data_fingerprint": self.data_fingerprint,
            "data_range": list(self.data_range),
            "n_rows": self.n_rows,
            "n_series": self.n_series,
            "n_dates": self.n_dates,
            "duplicate_series_timestamps": self.duplicate_series_timestamps,
            "missing_daily_timestamps": self.missing_daily_timestamps,
            "null_counts": self.null_counts,
            "inconsistent_static_series": self.inconsistent_static_series,
            "profile": None if self.profile is None else self.profile.as_record(),
            "spark_schema": self.spark_schema,
        }
        if self.missing_timestamps is not None:
            record["missing_timestamps"] = self.missing_timestamps
        return record


@dataclass(frozen=True)
class CheckReport:
    """Results for one frozen data-check or model-check request."""

    kind: Literal["data", "model"]
    facts: DataFacts
    checks: tuple[CheckResult, ...]
    model_id: str | None = None
    schema_version: int = 2

    @property
    def accepted(self) -> bool:
        return not any(check.blocks_run for check in self.checks)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(check.summary for check in self.checks if check.blocks_run)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(check.summary for check in self.checks if check.status == "warning")

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": f"{self.kind}_checks",
            "accepted": self.accepted,
            "model_id": self.model_id,
            "facts": self.facts.as_record(),
            "checks": [check.as_record() for check in self.checks],
        }


@dataclass(frozen=True)
class RunIncident:
    """A technical failure; deliberately not a data/model quality result."""

    id: str
    name: str
    summary: str
    technical_detail: str | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "kind": "run_incident",
            "status": "not_checked",
            "incident": {
                "id": self.id,
                "name": self.name,
                "summary": self.summary,
                "technical_detail": self.technical_detail,
            },
        }


@dataclass(frozen=True)
class ResolvedRun:
    """The immutable, allowlisted settings for one model attempt."""

    model_id: str
    adapter: str
    origin: str
    horizon: int
    settings: tuple[tuple[str, object], ...] = ()
    frequency: str = "D"

    def setting(self, name: str, default: object) -> object:
        return dict(self.settings).get(name, default)

    @property
    def config_hash(self) -> str:
        payload = {
            "model_id": self.model_id,
            "adapter": self.adapter,
            "origin": self.origin,
            "horizon": self.horizon,
            "settings": list(self.settings),
        }
        if self.frequency != "D":
            payload["frequency"] = self.frequency
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class ModelCheckInput:
    """Prepared run details and measured facts given to one model's readiness checker.

    It intentionally contains aggregate evidence only.  A model checker cannot read a new
    snapshot, start Spark, or quietly choose a different split while deciding readiness.
    """

    run: ResolvedRun
    data: DataFacts
    frequency: str
    target_estimand: str
    has_probabilistic_availability: bool
    total_series: int
    eligible_series: int
    series_meeting_history: int
    series_with_complete_window: int | None
    training_examples: int | None
    positive_training_examples: int | None
    irregular_series: int
    required_history: int


@dataclass(frozen=True)
class RunApproval:
    """Authorization to execute exactly the checked model request."""

    approved: bool
    model_id: str
    adapter: str
    config_hash: str
    data_fingerprint: str
    reason: str
