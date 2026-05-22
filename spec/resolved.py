"""The RESOLVED BUILD: the one object a training run may construct a model from.

`ForecastModelSpec.resolve()` degrades a configuration to what a client's columns support, and
`validate()` says whether the result is buildable -- but until this existed, nothing CONSUMED
either: `fit_predict` took two dozen loose keyword arguments, no run recorded which resolved
configuration it believed it was building, and the resolver's one head step-down (hurdle to
plain NB) had no executable path to a model. A spec nothing constructs from is documentation.

This object is the missing edge, and it is deliberately narrow:

  * it is produced ONLY by `ResolvedBuild.from_specs`, which runs the resolver and then REFUSES
    any combination `validate()` marks with an error -- rejection happens before training, not
    as a crash inside it;
  * it carries exactly the architecture decisions the runtime consumes, as plain values -- an
    adapter reads fields, never re-derives policy;
  * `build_hash` is the sha256 of its canonical form, persisted with every run and checkpoint,
    so "which architecture trained" is a recorded fact rather than a reconstruction. Advisory
    NOTE findings ride along for the record but stay OUTSIDE the hash: rewording a note must
    not read as a different architecture.

Pure stdlib, like everything in `spec/`: the data plane imports this without torch, and the
model plane without Spark.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

from spec.forecast_spec import ForecastDataSpec, ForecastModelSpec

__all__ = ["RESOLVED_BUILD_SCHEMA", "ResolveError", "ResolvedBuild"]


RESOLVED_BUILD_SCHEMA = "autoforecast.resolved-build.v2"
"""Canonical and only supported schema for resolved architectures."""


class ResolveError(ValueError):
    """The resolved data/model combination is not buildable, and training must not start."""


@dataclasses.dataclass(frozen=True)
class ResolvedBuild:
    """The architecture decisions a training run constructs from. See the module docstring.

    Attributes mirror the resolved `ForecastModelSpec` fields the runtime actually consumes,
    plus `zero_share` (the profiler measurement the head decision was made from, or None for
    unmeasured) and the advisory `notes` from validation.
    """

    schema: str
    schema_version: int
    head: str
    d_model: int
    attention_heads: int
    patch_size: int
    topology: str
    factor_rank: int
    factor_dynamics: str | None
    event_kernel: str | None
    event_max_support: tuple[int, int]
    input_normalisation: str | None
    demand_branch_dropout: float
    covariate_dropout: float
    price_response_mode: str
    price_occurrence_slope_max: float | None
    price_size_slope_max: float | None
    static_variable_selection: bool
    embedding_residual_dim: int
    n_sample_paths: int
    emit: str
    zero_share: float | None
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Refuse values that no validated resolver is allowed to publish or execute."""
        schema_versions = {
            RESOLVED_BUILD_SCHEMA: 2,
            "autoforecast.resolved-revision.v1": 1,
            "autoforecast.resolved-revision.v3": 3,
            "autoforecast.resolved-revision.v5": 5,
        }
        if type(self.schema) is not str or self.schema not in schema_versions:
            raise ResolveError(f"unsupported resolved build schema {self.schema!r}")
        if (
            type(self.schema_version) is not int
            or self.schema_version != schema_versions[self.schema]
        ):
            raise ResolveError(
                f"resolved build schema/version mismatch: {self.schema!r} "
                f"version {self.schema_version!r}"
            )

        heads = {"hurdle_negbin", "negbin", "quantile", "point"}
        if self.schema != RESOLVED_BUILD_SCHEMA:
            heads = {
                "hurdle_negbin",
                "shifted_hurdle_negbin",
                "zero_truncated_hurdle_negbin",
                "negbin",
            }
        if type(self.head) is not str or self.head not in heads:
            raise ResolveError(f"unsupported resolved head {self.head!r}")

        for name in ("d_model", "attention_heads", "patch_size", "n_sample_paths"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ResolveError(f"{name} must be a positive built-in integer, got {value!r}")
        if self.d_model % self.attention_heads:
            raise ResolveError("d_model must be divisible by attention_heads")
        if type(self.embedding_residual_dim) is not int or self.embedding_residual_dim < 0:
            raise ResolveError(
                "embedding_residual_dim must be a nonnegative built-in integer, got "
                f"{self.embedding_residual_dim!r}"
            )

        if self.topology not in {"bottom_up_factor", "bottom_up"} or type(self.topology) is not str:
            raise ResolveError(f"unsupported resolved topology {self.topology!r}")
        if type(self.factor_rank) is not int or self.factor_rank < 0:
            raise ResolveError(
                f"factor_rank must be a nonnegative built-in integer, got {self.factor_rank!r}"
            )
        if self.factor_dynamics is not None and (
            type(self.factor_dynamics) is not str or self.factor_dynamics not in {"ar1", "iid"}
        ):
            raise ResolveError(f"unsupported factor_dynamics {self.factor_dynamics!r}")
        if self.topology == "bottom_up":
            if self.factor_rank != 0 or self.factor_dynamics is not None:
                raise ResolveError("bottom_up requires factor_rank=0 and factor_dynamics=None")
        elif self.factor_rank < 1:
            raise ResolveError("bottom_up_factor requires factor_rank >= 1")

        if self.event_kernel is not None and (
            type(self.event_kernel) is not str
            or self.event_kernel not in {"distributed_lag", "fixed_window"}
        ):
            raise ResolveError(f"unsupported event_kernel {self.event_kernel!r}")
        if (
            type(self.event_max_support) is not tuple
            or len(self.event_max_support) != 2
            or any(type(value) is not int for value in self.event_max_support)
            or self.event_max_support[0] > 0
            or self.event_max_support[1] < 0
        ):
            raise ResolveError(
                "event_max_support must be a pair of built-in integers spanning day zero, got "
                f"{self.event_max_support!r}"
            )
        if type(self.input_normalisation) is not str or self.input_normalisation not in {
            "revin", "per_series_z"
        }:
            raise ResolveError(
                f"unsupported resolved input_normalisation {self.input_normalisation!r}"
            )
        for name in ("demand_branch_dropout", "covariate_dropout"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ResolveError(
                    f"{name} must be a finite built-in float in [0, 1), got {value!r}"
                )

        price_modes = {"none", "soft_penalty_v1", "monotone_raw_v1"}
        if type(self.price_response_mode) is not str or self.price_response_mode not in price_modes:
            raise ResolveError(f"unsupported price_response_mode {self.price_response_mode!r}")
        caps = (self.price_occurrence_slope_max, self.price_size_slope_max)
        if self.price_response_mode == "monotone_raw_v1":
            if any(type(value) is not float or not math.isfinite(value) or value <= 0.0
                   for value in caps):
                raise ResolveError(
                    "monotone_raw_v1 requires finite positive built-in float price caps"
                )
        elif caps != (None, None):
            raise ResolveError("price caps must be null outside monotone_raw_v1")

        if type(self.static_variable_selection) is not bool:
            raise ResolveError("static_variable_selection must be a built-in boolean")
        if self.emit not in {"sample_paths", "parameters"} or type(self.emit) is not str:
            raise ResolveError(f"unsupported resolved emit mode {self.emit!r}")
        if self.emit == "sample_paths" and self.n_sample_paths < 500:
            raise ResolveError("n_sample_paths must be at least 500 when emit='sample_paths'")
        if self.zero_share is not None and (
            type(self.zero_share) is not float
            or not math.isfinite(self.zero_share)
            or not 0.0 <= self.zero_share <= 1.0
        ):
            raise ResolveError(
                "zero_share must be null or a finite built-in float in [0, 1], got "
                f"{self.zero_share!r}"
            )
        if type(self.notes) is not tuple or any(
            type(note) is not str or not note.strip() for note in self.notes
        ):
            raise ResolveError("notes must be a tuple of non-blank strings")

    @classmethod
    def from_specs(
        cls, data: ForecastDataSpec, model: ForecastModelSpec, *, zero_share: float | None = None
    ) -> ResolvedBuild:
        """Resolve, validate, and either return a buildable configuration or refuse.

        Raises:
            ResolveError: `validate()` returned at least one error-severity finding. The message
                carries every error, because a builder that reports one of three blockers gets
                run three times.
        """
        if zero_share is not None:
            if (
                type(zero_share) not in (int, float)
                or (type(zero_share) is float and not math.isfinite(zero_share))
                or not 0.0 <= zero_share <= 1.0
            ):
                raise ResolveError(
                    f"zero_share must be a finite built-in number in [0, 1], got " f"{zero_share!r}"
                )
            zero_share = float(zero_share)

        data_findings = data.validate()
        resolved = model.resolve(data, zero_share=zero_share)
        findings = resolved.validate(data)
        errors = [message for message in data_findings if not message.startswith("NOTE:")]
        errors.extend(str(f) for f in findings if f.severity == "error")
        if errors:
            raise ResolveError(
                "the resolved data/model combination is not buildable; training must not "
                "start:\n  - " + "\n  - ".join(errors)
            )
        return cls(
            schema=RESOLVED_BUILD_SCHEMA,
            schema_version=2,
            head=resolved.head,
            d_model=resolved.d_model,
            attention_heads=resolved.heads,
            patch_size=resolved.patch_size,
            topology=resolved.topology,
            factor_rank=resolved.factor_rank if resolved.topology == "bottom_up_factor" else 0,
            factor_dynamics=(
                resolved.factor_dynamics if resolved.topology == "bottom_up_factor" else None
            ),
            event_kernel=resolved.event_kernel if data.events else None,
            event_max_support=tuple(resolved.event_max_support),
            input_normalisation=resolved.input_normalisation,
            demand_branch_dropout=float(resolved.demand_branch_dropout),
            covariate_dropout=float(resolved.covariate_dropout),
            price_response_mode=(resolved.price_response_mode if data.price_col else "none"),
            price_occurrence_slope_max=(
                None
                if resolved.price_response_mode != "monotone_raw_v1"
                else float(resolved.price_occurrence_slope_max)
            ),
            price_size_slope_max=(
                None
                if resolved.price_response_mode != "monotone_raw_v1"
                else float(resolved.price_size_slope_max)
            ),
            static_variable_selection=resolved.static_variable_selection,
            embedding_residual_dim=data.embedding_residual_dim,
            n_sample_paths=resolved.n_sample_paths,
            emit=resolved.emit,
            zero_share=zero_share,
            notes=(tuple(message for message in data_findings if message.startswith("NOTE:"))
                   + tuple(str(f) for f in findings if f.severity == "note")),
        )

    def canonical(self) -> dict[str, object]:
        """The architecture fields in a stable, JSON-safe form. Notes are excluded: they are
        advisory prose, and rewording one must not read as a different architecture."""
        record = dataclasses.asdict(self)
        record.pop("notes")
        record["event_max_support"] = list(self.event_max_support)
        return dict(sorted(record.items()))

    @classmethod
    def for_revision(
        cls,
        data: ForecastDataSpec,
        model: ForecastModelSpec,
        *,
        variant: str,
        zero_share: float | None = None,
        head_law: str | None = None,
    ) -> ResolvedBuild:
        """Resolve executable count revisions; tested by ``test_resolved_revision_build``.

        Reuse the existing degradation/validation, then bind the exact positive law. This
        separate schema cannot be mistaken for a production/scenario resolved build.
        """
        if (
            variant not in {"v1_corrected", "v2", "v3", "v5"}
            or data.target_family != "count"
            or data.target_allows_negative
            or model.head not in {"hurdle_negbin", "negbin"}
            or data.target_estimand not in {"sales_given_availability", "observed_count"}
        ):
            raise ResolveError("revisions implement observed nonnegative counts and count heads")
        if variant in {"v2", "v3"} and model.factor_rank:
            raise ResolveError("SFT/HMFNet do not implement a shared probabilistic factor")
        if variant == "v5" and model.factor_rank:
            raise ResolveError("PACT does not implement a shared probabilistic factor")
        if variant == "v5":
            if data.target_estimand != "sales_given_availability":
                raise ResolveError("PACT implements recorded sales only")
            if head_law not in {"negbin", "zero_truncated_hurdle_negbin"}:
                raise ResolveError("PACT requires an explicit registered segment-total law")
        elif head_law is not None:
            raise ResolveError("only PACT accepts a PACT-specific segment-total law")
        result = cls.from_specs(data, model, zero_share=zero_share)
        return dataclasses.replace(
            result,
            schema=(
                "autoforecast.resolved-revision.v5"
                if variant == "v5"
                else
                "autoforecast.resolved-revision.v3"
                if variant == "v3"
                else "autoforecast.resolved-revision.v1"
            ),
            schema_version=5 if variant == "v5" else 3 if variant == "v3" else 1,
            head=(
                head_law
                if variant == "v5"
                else
                "shifted_hurdle_negbin"
                if variant in {"v2", "v3"} and result.head == "hurdle_negbin"
                else result.head
            ),
        )

    def as_record(self) -> dict[str, object]:
        """Return the complete JSON-safe record required by an isolated worker."""
        return {**self.canonical(), "notes": list(self.notes)}

    @classmethod
    def from_record(cls, value: object) -> ResolvedBuild:
        """Rebuild a typed, exact resolved architecture from an authenticated JSON request."""
        if type(value) is not dict:
            raise ResolveError("resolved build record must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ResolveError(f"resolved build record fields differ; expected={sorted(expected)}")
        raw = dict(value)
        if type(raw["event_max_support"]) is not list or type(raw["notes"]) is not list:
            raise ResolveError("resolved build record has non-JSON tuple fields")
        raw["event_max_support"] = tuple(raw["event_max_support"])
        raw["notes"] = tuple(raw["notes"])
        try:
            result = cls(**raw)
        except TypeError as error:
            raise ResolveError("resolved build record cannot construct a build") from error
        if result.as_record() != value:
            raise ResolveError("resolved build record is not canonical")
        return result

    @property
    def build_hash(self) -> str:
        """sha256 of the canonical form -- the identity a run and its checkpoint record."""
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
