"""Registered gates and content-addressed evaluation manifests.

Thresholds are registered before measurement. Results bind their manifest and gate
identities, so edits and inconsistent verdicts are detectable. Caller-supplied
scores are unverifiable here: measurement and label separation cannot be proven;
those guarantees require an independent evaluator and external access control.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from autoforecast._exact_json import ExactJSONError, load_exact_json, loads_exact_json
from autoforecast._numeric import is_finite_real

REGISTRY = pathlib.Path(__file__).resolve().parent.parent / "gates" / "registered"
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "gates" / "results"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_ATTEMPT_FIELDS = frozenset({"phase", "manifest_hash", "passed"})
_ATTEMPT_FIELDS = frozenset({*_LEGACY_ATTEMPT_FIELDS, "attempt_number"})


def _fsync_directory(path: pathlib.Path) -> None:
    """Make a newly named filesystem entry durable before returning."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory_durable(path: pathlib.Path) -> None:
    """Create missing ancestors and sync each directory entry that names one."""
    missing: list[pathlib.Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise RuntimeError(f"cannot find an existing parent for {path}")
        cursor = parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _fsync_directory(directory.parent)


def _atomic_write_text(path: pathlib.Path, payload: str) -> None:
    """Durably replace one JSON artifact without exposing a partial canonical file."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _attempt_ledger_lock():
    """Serialize claim-registry and attempt state on one persistent protocol inode.

    Registration, attempt recording, result publication, and gate consumption share one order.
    Using the existing ignored lock file avoids a second lock-order graph and prevents a runtime
    registration from leaving an untracked lock inside ``gates/registered``.  The inode is never
    unlinked: replacing a held lock path would let another process lock a different file.
    """
    _ensure_directory_durable(RESULTS)
    lock_path = RESULTS / ".attempts.jsonl.lock"
    created = not lock_path.exists()
    with lock_path.open("a+", encoding="utf-8") as handle:
        if created:
            handle.flush()
            os.fsync(handle.fileno())
            _fsync_directory(lock_path.parent)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_name(value: str, field_name: str) -> None:
    """Keep claim-bearing artifact names inside their registry directories.

    Names become file names. Restricting them here prevents an accidental slash or `..` from
    writing a manifest/result somewhere other than the governed registry.
    """
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty filesystem-safe identifier")


def _validate_iso_date(value: str, field_name: str) -> None:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD), got {value!r}")


def _validate_iso_timestamp(value: str, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"{field_name} must be an ISO timestamp, got {value!r}") from None
    if parsed.utcoffset() is None:
        raise ValueError(
            f"{field_name} must include a UTC offset so it identifies one instant, got {value!r}")


def content_hash(obj: Any) -> str:
    """SHA-256 over canonical JSON.

    `sort_keys` and a fixed separator make the hash independent of dict ordering and of how the
    file happened to be written, so two runs that used the same configuration agree even if the
    JSON was regenerated. Without canonicalisation a reformat would look like a protocol change.
    """
    _validate_canonical_json(obj)
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_canonical_json(value: object, path: str = "$") -> None:
    """Validate finite JSON values, accepting tuples as JSON-array input.

    Tuples intentionally canonicalise to arrays. Existing protocol objects use tuples in
    memory and lists after JSON reload, and JSON gives both the same stable representation.
    Scalar types and object-key identity are preserved exactly; opaque values are never
    stringified into apparently reproducible evidence.
    """
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"protocol JSON {path} is non-finite")
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_canonical_json(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"protocol JSON object key {path}[{key!r}] is not a string")
            _validate_canonical_json(item, f"{path}.{key}")
        return
    raise TypeError(f"protocol JSON {path} has unsupported value {value!r}")


def _json_values_equal_exact(observed: object, expected: object) -> bool:
    """Compare JSON recursively, with list/tuple arrays equivalent and scalars type-exact."""
    if type(observed) in (list, tuple) and type(expected) in (list, tuple):
        return (len(observed) == len(expected)
                and all(_json_values_equal_exact(left, right)
                        for left, right in zip(observed, expected, strict=True)))
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return (observed.keys() == expected.keys()
                and all(_json_values_equal_exact(observed[key], value)
                        for key, value in expected.items()))
    return observed == expected


@dataclass(frozen=True)
class EvaluationManifest:
    """What a comparison was run against. Frozen, hashed, and referenced by every result.

    Every field here changes the number a comparison produces. Two results are comparable only if
    they carry the same manifest hash -- which is the whole reason this is a hashed artifact
    rather than prose in a README.
    """
    name: str
    target_estimand: Literal["sales_given_availability", "unconstrained_demand", "observed_nonnegative"]
    origins: tuple[str, ...]                 # ISO dates; ALL forecast origins, frozen
    horizon_days: int
    scoring_grain: tuple[str, ...]           # e.g. ("store", "product", "day") -- the PUBLISHED grain
    eligible_rule: str                       # how the eligible grid was derived, e.g. "exposure_daily.is_ranged"
    zero_fill_rule: str                      # what an absent row means inside the eligible window
    source_cutoffs: dict[str, str]           # per-source as-of timestamps; late arrivals excluded
    clients: tuple[str, ...]
    excluded_clients: dict[str, str]         # dataset -> explicit eligibility exclusion reason

    encoder_days: int = 0
    """How far back a forecast at an origin looks. Required whenever acceptance origins are
    reserved, because without it the temporal ordering cannot be checked."""

    max_gate_attempts: int = 3
    """How many times this manifest may be gated before its acceptance set is spent.

    Repeated submission can leak holdout information: submit, fail, tweak, resubmit. Each
    submission returns information about the acceptance set, so a frozen holdout becomes a
    slow-feedback tuning set -- and nothing about the manifest changes, so nothing detects it.

    Counting is not prevention. It makes the count VISIBLE, so a pass on attempt seven is read
    as what it is. Attempts are stored per manifest hash, so re-registering a gate does not reset
    them; only a genuinely different manifest gets a fresh budget, and one whose separation checks
    it must still pass.

    A reporting policy may return pass/fail without diagnostic metrics, so
    the developer cannot see the gradient to tune along. That is a reporting decision, not a
    protocol one, and it belongs to whoever runs the gate."""

    campaign_of_date: dict[str, str] = field(default_factory=dict)
    """Optional date -> campaign id. When supplied, acceptance and development horizons must not
    touch the same campaign. Date-level separation is not enough: a promotion spans days, and a
    split that puts one campaign on both sides leaks the outcome of that campaign."""

    latent_demand_source: str | None = None
    """Where ground truth for `unconstrained_demand` comes from. Required by that estimand,
    because observed sales cannot score it -- see the check in `__post_init__`."""

    acceptance_origins: tuple[str, ...] = ()
    """Origins RESERVED for the gate. Never used to choose a model, only to approve one.

    O4, decided 2026-08-22: **do not use the same forecast periods both to choose the model and to
    approve it.** Tuning against an origin and then accepting on it means the gate is scored on
    data the model was fitted toward, and the number it produces is optimistic by an amount nobody
    can bound.

    The remainder (`development_origins`) is what tuning, feature selection, early stopping and
    every failed experiment may touch. `evaluate_gate` refuses a result scored on anything else,
    and the split is inside the manifest hash -- so moving an origin from acceptance to
    development after seeing a result changes the hash and invalidates every result that cites it.

    Empty means the whole manifest is development-only, which is legitimate for exploration and
    is refused by the gate."""

    eligible_grid_hashes: dict[str, str] = field(default_factory=dict)
    """Optional origin -> SHA-256 of the materialised eligible key set.

    ``eligible_rule`` describes an intended derivation; it does not identify the cells that were
    actually materialised. A governed comparison populates this for every origin so the manifest
    hash changes if even one ``series_id · date`` key changes. Empty remains valid for legacy and
    explicitly exploratory manifests and is omitted from their serialisation, preserving the
    hashes of already-registered artifacts.
    """

    missing_forecast_policy: Literal["error", "zero"] = "error"
    """How a governed run handles canonical eligible-grid keys an adapter did not emit.

    ``"error"`` preserves the strict contract: one omitted key aborts the comparison. ``"zero"``
    asks the harness (never the adapter) to emit a deterministic zero for every omitted canonical
    key, recording those cells separately from model output.  The latter is the explicit
    cold-start/model-failure baseline used when every frozen cell must remain runnable.  It is
    meaningful only with content-bound ``eligible_grid_hashes`` and is part of the manifest hash.
    A model failure must be a declared omission in a valid ``Forecast``; an adapter exception or
    malformed artifact still aborts, because treating broken evidence as an all-zero forecast
    would make operational failure look like measured model performance.
    """

    def __post_init__(self) -> None:
        _validate_name(self.name, "manifest name")
        if self.target_estimand not in ("sales_given_availability", "unconstrained_demand",
                                        "observed_nonnegative"):
            raise ValueError(f"unsupported target_estimand: {self.target_estimand!r}")
        if (self.latent_demand_source is not None
                and (type(self.latent_demand_source) is not str
                     or not self.latent_demand_source.strip())):
            raise ValueError("latent_demand_source must be null or a non-blank string")
        if self.target_estimand == "unconstrained_demand" and self.latent_demand_source is None:
            raise ValueError(
                "O9: `unconstrained_demand` cannot be scored against observed sales. Forecast "
                "latent demand of 10 on a censored day with 5 sold, and an actual-based metric "
                "records five units of OVER-forecasting -- so it selects models that reproduce "
                "SUPPRESSED sales, which is the opposite of the estimand. Either set "
                "`target_estimand='sales_given_availability'`, or name a `latent_demand_source` "
                "that supplies ground truth (a censored-demand benchmark, a held-out "
                "uncensored period, or an explicit censoring model whose assumptions are stated).")
        if type(self.origins) not in (list, tuple):
            raise ValueError("origins must be a sequence of ISO date strings")
        if len(self.origins) < 3:
            raise ValueError("the frozen evaluation manifest needs at least three rolling origins")
        for origin in self.origins:
            _validate_iso_date(origin, "origin")
        if len(set(self.origins)) != len(self.origins):
            raise ValueError(f"duplicate origins: {self.origins}")
        if not isinstance(self.eligible_grid_hashes, dict):
            raise ValueError("eligible_grid_hashes must map origins to SHA-256 digests")
        if self.eligible_grid_hashes:
            if any(not isinstance(origin, str) for origin in self.eligible_grid_hashes):
                raise ValueError("eligible_grid_hashes keys must be manifest origin strings")
            missing_grids = set(self.origins) - set(self.eligible_grid_hashes)
            unexpected_grids = set(self.eligible_grid_hashes) - set(self.origins)
            if missing_grids or unexpected_grids:
                raise ValueError(
                    "eligible_grid_hashes must contain exactly the manifest origins; "
                    f"missing={sorted(missing_grids)}, "
                    f"unexpected={sorted(unexpected_grids)}")
            for origin, digest in self.eligible_grid_hashes.items():
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError(
                        f"eligible_grid_hashes[{origin!r}] must be a lowercase SHA-256 hex digest")
        if self.missing_forecast_policy not in ("error", "zero"):
            raise ValueError(
                "missing_forecast_policy must be 'error' or 'zero', got "
                f"{self.missing_forecast_policy!r}")
        if self.missing_forecast_policy == "zero" and not self.eligible_grid_hashes:
            raise ValueError(
                "missing_forecast_policy='zero' requires content-bound eligible_grid_hashes. "
                "Without the exact canonical keys, the harness cannot define which omissions "
                "receive the fallback")
        if type(self.horizon_days) is not int or self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if type(self.encoder_days) is not int or self.encoder_days < 0:
            raise ValueError("encoder_days must be a non-negative integer")
        if type(self.max_gate_attempts) is not int or self.max_gate_attempts < 1:
            raise ValueError("max_gate_attempts must be a positive integer")
        if (type(self.scoring_grain) not in (list, tuple)
                or not self.scoring_grain
                or any(not isinstance(v, str) or not v.strip() for v in self.scoring_grain)
                or len(set(self.scoring_grain)) != len(self.scoring_grain)):
            raise ValueError("scoring_grain is what makes the metric honest; it cannot be empty")
        if not isinstance(self.eligible_rule, str) or not self.eligible_rule.strip():
            raise ValueError("eligible_rule cannot be blank")
        if not isinstance(self.zero_fill_rule, str) or not self.zero_fill_rule.strip():
            raise ValueError("zero_fill_rule cannot be blank")
        if type(self.source_cutoffs) is not dict or not self.source_cutoffs:
            raise ValueError("source_cutoffs cannot be empty; point-in-time evaluation needs them")
        for source, cutoff in self.source_cutoffs.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source_cutoff names cannot be blank")
            _validate_iso_timestamp(cutoff, f"source cutoff for {source!r}")
        if (type(self.clients) not in (list, tuple)
                or not self.clients
                or any(not isinstance(v, str) or not v.strip() for v in self.clients)
                or len(set(self.clients)) != len(self.clients)):
            raise ValueError("clients must contain at least one unique, non-blank included client")
        if type(self.acceptance_origins) not in (list, tuple):
            raise ValueError("acceptance_origins must be a sequence of ISO date strings")
        if type(self.excluded_clients) is not dict:
            raise ValueError("excluded_clients must map client names to reasons")
        if (any(not isinstance(name, str) or not name.strip()
                for name in self.excluded_clients)
                or any(not isinstance(reason, str) or not reason.strip()
                       for reason in self.excluded_clients.values())):
            raise ValueError("every excluded client needs a non-blank string name and reason")
        if type(self.campaign_of_date) is not dict:
            raise ValueError("campaign_of_date must map ISO dates to campaign identifiers")
        for campaign_date, campaign in self.campaign_of_date.items():
            _validate_iso_date(campaign_date, "campaign date")
            if not isinstance(campaign, str) or not campaign.strip():
                raise ValueError("campaign identifiers must be non-blank strings")
        if self.acceptance_origins:
            for origin in self.acceptance_origins:
                _validate_iso_date(origin, "acceptance origin")
            if len(set(self.acceptance_origins)) != len(self.acceptance_origins):
                raise ValueError(f"duplicate acceptance origins: {self.acceptance_origins}")
            missing = set(self.acceptance_origins) - set(self.origins)
            if missing:
                raise ValueError(f"acceptance origins not in origins: {sorted(missing)}")
            if not self.development_origins:
                raise ValueError(
                    "every origin is reserved for acceptance, leaving nothing to develop against "
                    "-- tuning would then have to touch the acceptance set")
            self._check_temporal_separation()
            self._check_campaign_separation()
        overlap = set(self.clients) & set(self.excluded_clients)
        if overlap:
            raise ValueError(f"clients both included and excluded: {sorted(overlap)}")

    def _span(self, origin: str) -> tuple[_dt.date, _dt.date]:
        """The dates a forecast at this origin TOUCHES: its encoder window and its horizon."""
        o = _dt.date.fromisoformat(origin)
        return (o - _dt.timedelta(days=self.encoder_days),
                o + _dt.timedelta(days=self.horizon_days))

    def _check_temporal_separation(self) -> None:
        """Every acceptance horizon must fall strictly after everything development touches.

        Check both directions of temporal overlap:

          · forward -- an acceptance origin's horizon can sit INSIDE a development horizon, so the
            model was tuned on the very days it is judged on;
          · backward -- a LATER development origin's encoder window looks back over the acceptance
            period, so tuning has already seen the outcomes it will be judged on.

        Reserving a date STRING prevents neither. What matters is the dates each origin touches.
        """
        if self.encoder_days <= 0:
            raise ValueError(
                "acceptance origins are reserved but encoder_days is unset. Temporal separation "
                "cannot be checked without knowing how far back an origin looks, and an unchecked "
                "reservation is bookkeeping rather than a control.")
        acc_start = min(_dt.date.fromisoformat(o) for o in self.acceptance_origins)
        for origin in self.development_origins:
            dev_lo, dev_hi = self._span(origin)
            if dev_hi >= acc_start:
                raise ValueError(
                    f"development origin {origin} touches {dev_lo}..{dev_hi}, which reaches the "
                    f"acceptance period beginning {acc_start}. Every development origin -- its "
                    "encoder window AND its horizon -- must end strictly before the earliest "
                    "acceptance origin, or the model is tuned on what it is judged by.")

    def _check_campaign_separation(self) -> None:
        """No campaign may appear on both sides. A promotion spans days; splitting by date puts
        the same campaign's outcome in development and in acceptance."""
        if not self.campaign_of_date:
            return
        def campaigns(origins):
            out = set()
            for o in origins:
                lo, hi = self._span(o)
                d = lo
                while d <= hi:
                    c = self.campaign_of_date.get(d.isoformat())
                    if c:
                        out.add(c)
                    d += _dt.timedelta(days=1)
            return out
        shared = campaigns(self.development_origins) & campaigns(self.acceptance_origins)
        if shared:
            raise ValueError(
                f"campaigns on both sides of the split: {sorted(shared)}. Promotions span days, "
                "so a date-level split leaks a campaign's outcome into development.")

    @property
    def development_origins(self) -> tuple[str, ...]:
        """Origins tuning may touch: everything not reserved for acceptance."""
        reserved = set(self.acceptance_origins)
        return tuple(o for o in self.origins if o not in reserved)

    @property
    def hash(self) -> str:
        return content_hash(self._record())

    def _record(self) -> dict[str, Any]:
        """Canonical manifest record, retaining hashes of pre-grid legacy artifacts."""
        record = asdict(self)
        if not self.eligible_grid_hashes:
            # This field did not exist in registered manifests before grid byte binding. Including
            # an empty default would rewrite their hash and make their acceptance set unreadable.
            record.pop("eligible_grid_hashes")
        if self.missing_forecast_policy == "error":
            # Preserve the hash of manifests registered before a missing-forecast policy existed.
            # Governed zero fallback is always serialised and therefore always changes the hash.
            record.pop("missing_forecast_policy")
        return record

    def save(self, path: pathlib.Path | None = None) -> pathlib.Path:
        path = path or (REGISTRY / f"manifest_{self.name}.json")
        # ``asdict`` returns a detached tree. Hash and serialize this ONE snapshot so mutation of a
        # caller-owned nested mapping cannot make the envelope disagree with itself between two
        # independent ``self._record()`` / ``self.hash`` traversals.
        record = self._record()
        digest = content_hash(record)
        payload = json.dumps(
            {"manifest": record, "hash": digest}, indent=1, allow_nan=False)
        with _attempt_ledger_lock():
            _ensure_directory_durable(path.parent)
            if path.exists():
                try:
                    existing = load_exact_json(path)
                except ExactJSONError as exc:
                    raise ValueError(
                        f"existing manifest {path.name} is unreadable: {exc}") from exc
                if content_hash(existing["manifest"]) != digest:
                    raise ValueError(
                        f"{path.name} already exists with a DIFFERENT hash. A manifest is frozen "
                        "once written -- results already reference it. Write a new named manifest "
                        "instead of editing this one.")
            _atomic_write_text(path, payload)
        return path

    def require_registered(self) -> None:
        """Refuse evaluation against an in-memory manifest that was never frozen.

        A hash of an object created after scores were seen is not a registered protocol. The
        default registry artifact must already exist and must still contain this exact manifest.
        """
        path = REGISTRY / f"manifest_{self.name}.json"
        if not path.exists():
            raise RuntimeError(
                f"manifest {self.name!r} has not been registered. Save it before any evaluation")
        try:
            blob = load_exact_json(path)
        except ExactJSONError as exc:
            raise RuntimeError(
                f"registered manifest {self.name!r} is unreadable: {exc}") from exc
        if blob.get("hash") != self.hash or content_hash(blob.get("manifest")) != self.hash:
            raise RuntimeError(f"registered manifest {self.name!r} differs from the one evaluated")


@dataclass(frozen=True)
class ExtraCriterion:
    """An additional registered absolute bound, such as stability or coverage, which must pass
    alongside the primary metric.
    """
    metric: str
    bound: float
    direction: Literal["lower_is_better", "higher_is_better"]

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("extra criterion metric cannot be blank")
        if not is_finite_real(self.bound):
            raise ValueError(f"extra criterion {self.metric!r}: bound must be finite")
        if self.direction not in ("lower_is_better", "higher_is_better"):
            raise ValueError(f"extra criterion {self.metric!r}: bad direction")

    def evaluate(self, value: float) -> tuple[bool, str]:
        if not is_finite_real(value):
            raise ValueError(f"extra criterion {self.metric!r}: value must be finite, got {value!r}")
        ok = value <= self.bound if self.direction == "lower_is_better" else value >= self.bound
        return ok, f"{'PASS' if ok else 'FAIL'} {self.metric}={value} vs bound {self.bound}"


@dataclass(frozen=True)
class RegisteredGate:
    """A threshold committed BEFORE the phase runs.

    `direction` is explicit rather than inferred from the metric name, because 'lower is better'
    is true of MAE and false of coverage, and guessing it wrong silently inverts the gate.
    """
    phase: str
    metric: str
    comparators: tuple[str, ...]             # ALL must be beaten. Gates 2 and 4 name several,
                                             # and a single-comparator API let the weakest one
                                             # be passed in and recorded as a PASS.
    min_improvement: float                   # registered margin, in the metric's own units
    direction: Literal["lower_is_better", "higher_is_better"]
    rationale: str                           # why THIS number is the minimum that would matter
    registered_on: str                       # ISO date, supplied by the caller -- never now()
    extra_criteria: tuple[ExtraCriterion, ...] = ()
    """Conditions beyond the primary metric. ALL must hold -- a gate is a conjunction. See
    `ExtraCriterion`; empty means the gate behaves exactly as a single-metric gate."""

    measurement_contract: dict[str, dict[str, Any]] = field(default_factory=dict)
    """How every extra score is measured, frozen alongside its threshold.

    A threshold alone does not define a gate. Coverage changes with its nominal levels, cover
    period, path count, and cohort; stability changes with either origin. Before this field those
    choices lived only in a mutable runner/result file, so they could change without changing the
    registered gate hash. Keys must exactly match ``extra_criteria``. Empty is retained only to
    reconstruct legacy registrations; a new extra-criterion gate cannot be saved without it.
    """

    per_client_policy: Literal["pooled_only", "no_client_regresses", "majority"] = \
        "no_client_regresses"
    """Policy for dataset-level regressions. A pooled improvement can hide regressions in smaller
    datasets; pooled_only must therefore be chosen explicitly and justified.
    """

    relation: Literal["beats", "gap_from"] = "beats"
    """Comparison direction. beats requires improvement over every comparator; gap_from measures
    remaining headroom below a stronger comparator such as an oracle.
    """

    def __post_init__(self) -> None:
        _validate_name(self.phase, "phase")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("metric cannot be blank")
        if (type(self.comparators) not in (list, tuple)
                or not self.comparators
                or any(not isinstance(v, str) or not v.strip() for v in self.comparators)
                or len(set(self.comparators)) != len(self.comparators)):
            raise ValueError("comparators must contain at least one unique, non-blank name")
        if not is_finite_real(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement must be a finite, non-negative number")
        if self.direction not in ("lower_is_better", "higher_is_better"):
            raise ValueError(f"unsupported gate direction: {self.direction!r}")
        if self.relation not in ("beats", "gap_from"):
            raise ValueError(f"unsupported gate relation: {self.relation!r}")
        if (type(self.extra_criteria) not in (list, tuple)
                or any(not isinstance(criterion, ExtraCriterion)
                       for criterion in self.extra_criteria)):
            raise ValueError("extra_criteria must contain only ExtraCriterion records")
        names = [self.metric] + [c.metric for c in self.extra_criteria]
        if len(set(names)) != len(names):
            raise ValueError(f"a metric appears twice in gate {self.phase!r}: {names}")
        if not isinstance(self.measurement_contract, dict):
            raise ValueError("measurement_contract must map extra metric names to definitions")
        if self.measurement_contract:
            expected = {c.metric for c in self.extra_criteria}
            actual = set(self.measurement_contract)
            if actual != expected:
                raise ValueError(
                    "measurement_contract must contain exactly the extra criteria; "
                    f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
            for metric, definition in self.measurement_contract.items():
                if not isinstance(metric, str) or not metric.strip():
                    raise ValueError("measurement_contract keys must be non-blank metric names")
                if not isinstance(definition, dict) or not definition:
                    raise ValueError(
                        f"measurement_contract[{metric!r}] must be a non-empty definition")
            try:
                # Deliberately no ``default=str``: opaque objects and non-finite numbers do not
                # form a reproducible cross-process protocol value.
                _validate_canonical_json(self.measurement_contract)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "measurement_contract must contain only finite JSON values") from exc
        if self.per_client_policy not in ("pooled_only", "no_client_regresses", "majority"):
            raise ValueError(f"unsupported per_client_policy: {self.per_client_policy!r}")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("a registered threshold needs a non-blank rationale")
        _validate_iso_date(self.registered_on, "registered_on")

    @property
    def hash(self) -> str:
        return content_hash(self._record())

    def _record(self) -> dict[str, Any]:
        """Canonical gate record, retaining hashes of pre-contract registrations."""
        record = asdict(self)
        if not self.measurement_contract:
            record.pop("measurement_contract")
        return record

    def save(self) -> pathlib.Path:
        path = REGISTRY / f"{self.phase}.json"
        record = self._record()
        digest = content_hash(record)
        payload = json.dumps({"gate": record, "hash": digest}, indent=1, allow_nan=False)
        with _attempt_ledger_lock():
            _ensure_directory_durable(path.parent)
            try:
                existing = load_exact_json(path) if path.exists() else None
            except ExactJSONError as exc:
                raise ValueError(f"registered gate {self.phase!r} is unreadable: {exc}") from exc
            if self.extra_criteria and not self.measurement_contract:
                # Existing artifacts predate measurement binding and must remain readable. They may
                # be re-saved idempotently, but no NEW gate can repeat that omission.
                is_matching_legacy = bool(
                    existing and "measurement_contract" not in existing.get("gate", {})
                    and existing.get("hash") == digest)
                if not is_matching_legacy:
                    raise ValueError(
                        "a new gate with extra criteria must register a measurement_contract. A "
                        "bound without the cohort, origins, levels, and algorithm that produce its "
                        "score can be changed after registration without changing the gate")
            if existing is not None and existing.get("hash") != digest:
                raise ValueError(
                    f"gate {self.phase} is already registered with a different threshold. "
                    "Changing a registered gate after the fact is the exact failure this file "
                    "exists to prevent. Amend by recording a NEW phase id and the reason.")
            _atomic_write_text(path, payload)
        return path

    @classmethod
    def load(cls, phase: str) -> RegisteredGate:
        _validate_name(phase, "phase")
        path = REGISTRY / f"{phase}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no registered gate for {phase!r}. The threshold must be committed BEFORE the "
                "phase runs; a gate registered afterwards proves nothing.")
        try:
            blob = load_exact_json(path)
        except ExactJSONError as exc:
            raise ValueError(f"registered gate {phase!r} is unreadable: {exc}") from exc
        raw = dict(blob["gate"])
        raw["comparators"] = tuple(raw.get("comparators", ()))
        # extras round-trip as dicts through JSON; rebuild them or `.evaluate()` will not exist
        raw["extra_criteria"] = tuple(ExtraCriterion(**c)
                                      for c in raw.get("extra_criteria", ()))
        gate = cls(**raw)
        if gate.hash != blob["hash"]:
            raise ValueError(f"{path.name} has been edited since registration: hash mismatch")
        return gate

    def passes(self, comparator_scores: dict[str, float], candidate_score: float) -> bool:
        """PASS requires beating EVERY named comparator by the registered margin.

        Takes a mapping rather than a float so a caller cannot supply only the weakest baseline.
        A missing comparator is a failure, not an omission.
        """
        if not is_finite_real(candidate_score):
            raise ValueError("candidate score must be finite -- the evaluation is broken otherwise")
        missing = set(self.comparators) - set(comparator_scores)
        if missing:
            raise ValueError(f"gate {self.phase} requires beating {sorted(missing)}; not supplied")
        unexpected = set(comparator_scores) - set(self.comparators)
        if unexpected:
            raise ValueError(f"gate {self.phase} received unregistered comparators: {sorted(unexpected)}")
        for name in self.comparators:
            ref = comparator_scores[name]
            if not is_finite_real(ref):
                raise ValueError(f"comparator {name!r} score must be finite")
            gain = (ref - candidate_score) if self.direction == "lower_is_better" \
                else (candidate_score - ref)
            if self.relation == "gap_from":
                # The comparator is expected to win. `gain` is candidate improvement, so a valid
                # headroom gap is NEGATIVE by at least the margin. `abs(gain)` was wrong: it also
                # passed when the alleged skill ceiling was worse than the shallow candidate.
                if gain > -self.min_improvement:
                    return False
            elif gain < self.min_improvement:
                return False
        return True

    def check_extras(self, extra_scores: dict[str, float]) -> tuple[bool, list[str]]:
        """Every extra criterion must hold. A missing score is a FAILURE, not an omission --
        otherwise Gate 4 passes by simply not measuring stability."""
        if type(extra_scores) is not dict:
            raise ValueError("extra scores must be a metric-to-value mapping")
        expected = {criterion.metric for criterion in self.extra_criteria}
        unexpected = set(extra_scores) - expected
        if unexpected:
            raise ValueError(
                f"gate {self.phase} received unregistered extra scores: {sorted(unexpected)}")
        reasons, ok = [], True
        for c in self.extra_criteria:
            if c.metric not in extra_scores:
                reasons.append(f"FAIL {c.metric}: required by the gate and not supplied")
                ok = False
                continue
            passed, why = c.evaluate(extra_scores[c.metric])
            reasons.append(why)
            ok = ok and passed
        return ok, reasons

    def check_per_client(
        self,
        per_client: dict[str, float],
        comparator_per_client: dict[str, dict[str, float]],
        n_clients: int | None = None,
        *,
        clients: tuple[str, ...] | None = None,
    ) -> tuple[bool, str]:
        """Refuse a pooled win that is a loss for the clients hidden inside it.

        A client must be compared with each comparator *on that same client*.  Comparing a
        client's score with a pooled comparator score mixes different series populations and can
        reverse the conclusion (the original implementation did exactly that).  The no-regression
        reference is therefore the strongest registered comparator for each client: the minimum
        for a lower-is-better metric and the maximum for a higher-is-better metric.

        ``n_clients`` remains only for compatibility with direct single-client callers.  A
        multi-client check needs ``clients`` so an omitted client cannot make a partial mapping
        look complete.
        """
        if type(per_client) is not dict:
            raise ValueError("per_client must be a client-to-score mapping")
        if type(comparator_per_client) is not dict:
            raise ValueError("comparator_per_client must map comparators to client score mappings")
        if self.per_client_policy == "pooled_only":
            if per_client or comparator_per_client:
                raise ValueError(
                    "pooled_only gates cannot carry unregistered per-client evidence")
            return True, "NOTE per-client behaviour explicitly NOT checked (pooled_only)"
        if clients is None:
            if n_clients is None or n_clients <= 1:
                # Nothing can hide: the pooled result IS that client. Requiring a per-client
                # breakdown of one client would fail every single-client phase, M5 included.
                if per_client or comparator_per_client:
                    raise ValueError(
                        "single-client evidence needs the exact manifest client name or must be "
                        "omitted")
                return True, "PASS per-client: single-client manifest, pooled result is the client"
            raise ValueError(
                "multi-client checks require the exact manifest client names; a client count "
                "cannot detect a partial per-client score mapping")

        expected_clients = tuple(clients)
        if (not expected_clients or len(set(expected_clients)) != len(expected_clients)
                or any(not isinstance(v, str) or not v.strip() for v in expected_clients)):
            raise ValueError("manifest clients must be unique, non-blank strings")
        if n_clients is not None and n_clients != len(expected_clients):
            raise ValueError(
                f"client count {n_clients} disagrees with {len(expected_clients)} client names")
        if len(expected_clients) <= 1:
            # Nothing can hide: the pooled result IS that client. Requiring a per-client breakdown
            # of one client would fail every single-client phase, M5 included, for no benefit.
            if not per_client and not comparator_per_client:
                return True, "PASS per-client: single-client manifest, pooled result is the client"

        def require_exact_scores(label: str, scores: object,
                                 expected: tuple[str, ...]) -> dict[str, float]:
            if not isinstance(scores, dict):
                raise ValueError(f"{label} must be a client-to-score mapping")
            if any(not isinstance(k, str) or not k.strip() for k in scores):
                raise ValueError(f"{label} keys must be non-blank client names")
            missing = set(expected) - set(scores)
            unexpected = set(scores) - set(expected)
            if missing or unexpected:
                raise ValueError(
                    f"{label} must contain exactly the manifest clients; "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
            for client, value in scores.items():
                if not is_finite_real(value):
                    raise ValueError(f"{label}[{client!r}] must be a finite numeric score")
            return scores

        candidate = require_exact_scores("candidate per-client scores", per_client,
                                         expected_clients)
        if any(not isinstance(k, str) or not k.strip() for k in comparator_per_client):
            raise ValueError("comparator_per_client keys must be non-blank comparator names")
        missing_comparators = set(self.comparators) - set(comparator_per_client)
        unexpected_comparators = set(comparator_per_client) - set(self.comparators)
        if missing_comparators or unexpected_comparators:
            raise ValueError(
                "comparator_per_client must contain exactly the registered comparators; "
                f"missing={sorted(missing_comparators)}, "
                f"unexpected={sorted(unexpected_comparators)}")
        comparator_clients = {
            name: require_exact_scores(
                f"per-client scores for comparator {name!r}",
                comparator_per_client[name], expected_clients)
            for name in self.comparators
        }

        worse: list[str] = []
        for client in expected_clients:
            same_client = [comparator_clients[name][client] for name in self.comparators]
            strongest = (min(same_client) if self.direction == "lower_is_better"
                         else max(same_client))
            if (candidate[client] > strongest if self.direction == "lower_is_better"
                    else candidate[client] < strongest):
                worse.append(client)
        worse.sort()
        if self.per_client_policy == "no_client_regresses" and worse:
            return False, (
                f"FAIL per-client: {worse} worse than their strongest same-client comparator -- "
                "a pooled pass would have hidden this")
        if self.per_client_policy == "majority" and len(worse) * 2 >= len(expected_clients):
            return False, f"FAIL per-client: {len(worse)}/{len(expected_clients)} regressed"
        return (True, f"PASS per-client ({self.per_client_policy}): "
                      f"{len(worse)}/{len(expected_clients)} regressed")


@dataclass(frozen=True)
class GateResult:
    """The PASS/FAIL artifact a later phase requires before it may start.

    A result cannot be constructed as PASS without both hashes, which is what stops
    `baseline_gate = True` from happening again in a different shape.
    """
    phase: str
    manifest_hash: str
    gate_hash: str
    metric: str
    comparator_scores: dict[str, float]
    candidate_score: float
    passed: bool                             # a CACHED opinion; require_passed recomputes it
    expected_cells: int = 0
    scored_cells: int = 0
    """O7: a result carried a score and no proof the grid it scored was COMPLETE. A run that
    silently covered a tenth of the eligible cells produced a number indistinguishable from a
    full one -- and a partial run is systematically biased, because the cells that drop out are
    the sparse ones. `evaluate_gate` requires these to agree."""

    scored_origins: tuple[str, ...] = ()
    """WHICH origins produced these scores. O4: a gate scored on development origins is scored on
    data the model was tuned toward, so `evaluate_gate` requires this to be exactly the
    manifest's acceptance set."""
    per_client: dict[str, float] = field(default_factory=dict)
    extra_scores: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    notes: str = ""
    comparator_per_client: dict[str, dict[str, float]] = field(default_factory=dict)
    measurement_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Exact registered measurement definition attested by the producing evaluator.

    This intentionally equals, rather than paraphrases, ``RegisteredGate.measurement_contract``.
    The gate hash says what was registered; this field wires the result to that definition so a
    runner cannot calculate the right-named score with different origins, cohort, or path count.
    Empty preserves compatibility for gates with no registered measurement contract.
    """
    attempt_number: int | None = None
    """One-based manifest attempt identity assigned by the durable ledger.

    ``None`` is retained only for artifacts written before ordered attempts existed and for the
    documented direct-authoring limitation. Once a phase has a numbered attempt, an unnumbered or
    older result may be retained in immutable history but cannot replace its canonical result.
    """

    def save(self) -> pathlib.Path:
        _validate_name(self.phase, "phase")
        body = asdict(self)
        if self.attempt_number is None:
            # Preserve the byte/hash contract of pre-numbering artifacts. The authority check below
            # permits this shape only while no numbered attempt exists for the phase.
            body.pop("attempt_number")
        digest = content_hash(body)
        # The digest is key-order independent, so its immutable filename must have one equally
        # canonical byte representation. Otherwise two equivalent comparator mappings could share
        # a digest but disagree byte-for-byte and make an idempotent re-save look like corruption.
        blob = json.dumps(
            body | {"hash": digest}, indent=1, sort_keys=True, allow_nan=False)

        # The canonical path is the latest attempt used by require_passed. Every distinct attempt
        # is also retained under its content hash, so a fail followed by a pass cannot erase the
        # failure history. Re-saving identical content is idempotent and does not invent an attempt.
        with _attempt_ledger_lock():
            is_latest = _attempt_publication_is_latest_unlocked(self)
            history = RESULTS / self.phase
            _ensure_directory_durable(history)
            immutable = history / f"{digest}.json"
            if immutable.exists():
                if immutable.read_text(encoding="utf-8") != blob:
                    raise RuntimeError(
                        f"immutable gate-result history {immutable.name} disagrees with its hash")
            else:
                _atomic_write_text(immutable, blob)
            if not is_latest:
                raise RuntimeError(
                    f"gate result attempt {self.attempt_number} for phase {self.phase!r} is no "
                    "longer the latest recorded attempt. It was retained in immutable history but "
                    "cannot replace the canonical result")
            path = RESULTS / f"{self.phase}.json"
            _atomic_write_text(path, blob)
        return path


def evaluate_gate(phase: str, manifest: EvaluationManifest,
                  metric: str, comparator_scores: dict[str, float], candidate_score: float,
                  per_client: dict[str, float] | None = None,
                  extra_scores: dict[str, float] | None = None,
                  scored_origins: tuple[str, ...] | None = None,
                  expected_cells: int = 0, scored_cells: int = 0,
                  notes: str = "",
                  comparator_per_client: dict[str, dict[str, float]] | None = None,
                  measurement_evidence: dict[str, dict[str, Any]] | None = None) -> GateResult:
    """Score a phase against its REGISTERED threshold and produce the artifact.

    Loading the gate rather than accepting a threshold argument is deliberate: a caller cannot
    pass the bar it wants.
    """
    manifest.require_registered()
    gate = RegisteredGate.load(phase)

    # O4, decided 2026-08-22: do not use the same forecast periods to choose the model and to
    # approve it. A gate scored on development origins is scored on data the model was tuned
    # toward, and is optimistic by an amount nobody can bound.
    if not manifest.acceptance_origins:
        raise ValueError(
            f"manifest {manifest.name!r} reserves no acceptance origins, so every origin is one "
            "tuning may touch. A gate cannot be run against it -- reserve a held-out set.")
    raw_scored = () if scored_origins is None else scored_origins
    if (type(raw_scored) not in (list, tuple)
            or any(type(origin) is not str for origin in raw_scored)
            or len(raw_scored) != len(manifest.acceptance_origins)
            or set(raw_scored) != set(manifest.acceptance_origins)):
        raise ValueError(
            f"gate {phase!r} must be scored on EXACTLY the acceptance origins "
            f"{sorted(manifest.acceptance_origins)}; got {raw_scored!r}. Scoring on development "
            "origins approves a model on the periods it was chosen with.")
    # Acceptance origins are a set semantically, but result hashes are ordered JSON. Store one
    # canonical order so a caller's iteration order cannot create two claim identities for the
    # same evidence.
    scored = tuple(sorted(raw_scored))
    if metric != gate.metric:
        raise ValueError(
            f"gate {phase!r} is registered for metric {gate.metric!r}, not {metric!r}")
    evidence = dict(measurement_evidence or {})
    if not _json_values_equal_exact(evidence, gate.measurement_contract):
        raise ValueError(
            f"gate {phase!r} measurement_evidence must exactly equal its registered "
            "measurement_contract. A same-named score measured on different origins, levels, "
            "paths, or cohort is not evidence for this gate")
    if per_client is not None and not isinstance(per_client, dict):
        raise ValueError("per_client must be a client-to-score mapping")
    client_scores = dict(per_client or {})
    for client, value in client_scores.items():
        if not isinstance(client, str) or not client.strip() or not is_finite_real(value):
            raise ValueError("per-client scores need non-blank clients and finite numeric values")
    if comparator_per_client is not None and not isinstance(comparator_per_client, dict):
        raise ValueError("comparator_per_client must map comparator names to client score mappings")
    comparator_client_scores: dict[str, dict[str, float]] = {}
    for comparator, scores in (comparator_per_client or {}).items():
        if not isinstance(comparator, str) or not comparator.strip() or not isinstance(scores, dict):
            raise ValueError(
                "comparator_per_client needs non-blank comparator names and score mappings")
        comparator_client_scores[comparator] = dict(scores)
    if (type(expected_cells) is not int or type(scored_cells) is not int
            or expected_cells <= 0 or scored_cells <= 0):
        raise ValueError(
            "a gate result must state expected_cells and scored_cells. Without them a run that "
            "covered a tenth of the eligible grid is indistinguishable from a complete one, and "
            "the cells that drop out are the sparse ones -- so the bias is systematic.")
    if scored_cells != expected_cells:
        raise ValueError(
            f"grid coverage is incomplete: scored {scored_cells:,} of {expected_cells:,} eligible "
            f"cells ({scored_cells / expected_cells:.1%}). Score the full eligible grid, or amend "
            "the manifest's eligibility rule so the expectation matches what is scorable.")
    if extra_scores is not None and type(extra_scores) is not dict:
        raise ValueError("extra_scores must map registered metric names to scores")
    extras = dict(extra_scores or {})
    extras_ok, extra_reasons = gate.check_extras(extras)
    pc_ok, pc_reason = gate.check_per_client(
        client_scores, comparator_client_scores, clients=manifest.clients)
    primary_ok = gate.passes(comparator_scores, candidate_score)
    reasons = tuple([f"{'PASS' if primary_ok else 'FAIL'} {metric} (primary)"]
                    + extra_reasons + [pc_reason])
    passed = primary_ok and extras_ok and pc_ok
    attempt_number = _record_attempt(
        phase,
        manifest.hash,
        passed,
        manifest_name=manifest.name,
        max_gate_attempts=manifest.max_gate_attempts,
    )
    return GateResult(
        phase=phase,
        manifest_hash=manifest.hash,
        gate_hash=gate.hash,
        metric=metric,
        scored_origins=scored,
        expected_cells=expected_cells,
        scored_cells=scored_cells,
        extra_scores=extras,
        reasons=reasons,
        comparator_scores=dict(comparator_scores),
        candidate_score=candidate_score,
        passed=passed,
        attempt_number=attempt_number,
        per_client=client_scores,
        comparator_per_client=comparator_client_scores,
        measurement_evidence=evidence,
        notes=notes,
    )


def attempts_against(manifest_hash: str) -> list[dict]:
    """Every gate attempt recorded against this manifest, oldest first.

    Attempts live beside the results and are append-only, so the file-drawer leak -- submit, fail,
    tweak, resubmit -- is visible in the record rather than invisible in a developer's memory.
    """
    with _attempt_ledger_lock():
        return [
            record for record in _read_attempt_ledger_unlocked()
            if record["manifest_hash"] == manifest_hash
        ]


def _validate_attempt_record(record: object, line_number: int) -> dict[str, object]:
    """Validate one append-only attempt without accepting JSON shape or scalar aliases."""
    if (type(record) is not dict
            or record.keys() not in (_LEGACY_ATTEMPT_FIELDS, _ATTEMPT_FIELDS)):
        raise RuntimeError(
            f"gate-attempt ledger line {line_number} must be an object with exactly either "
            f"{sorted(_LEGACY_ATTEMPT_FIELDS)} or {sorted(_ATTEMPT_FIELDS)}")
    phase = record["phase"]
    manifest_hash = record["manifest_hash"]
    passed = record["passed"]
    if type(phase) is not str or not _SAFE_NAME.fullmatch(phase):
        raise RuntimeError(
            f"gate-attempt ledger line {line_number} has an invalid phase identifier")
    if type(manifest_hash) is not str or not _SHA256.fullmatch(manifest_hash):
        raise RuntimeError(
            f"gate-attempt ledger line {line_number} has an invalid manifest hash")
    if type(passed) is not bool:
        raise RuntimeError(
            f"gate-attempt ledger line {line_number} has a non-boolean passed field")
    if "attempt_number" in record and (
            type(record["attempt_number"]) is not int or record["attempt_number"] < 1):
        raise RuntimeError(
            f"gate-attempt ledger line {line_number} has an invalid attempt number")
    return record


def _read_attempt_ledger_unlocked() -> list[dict[str, object]]:
    """Read and validate the complete ledger while the stable ledger lock is held."""
    log = RESULTS / "attempts.jsonl"
    if not log.exists():
        return []
    try:
        payload = log.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"gate-attempt ledger is unreadable: {exc}") from exc
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise RuntimeError(
            "gate-attempt ledger has a truncated final line (missing newline terminator)")
    records: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    numbered_records_seen = False
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise RuntimeError(
                f"gate-attempt ledger line {line_number} is invalid JSON: empty record")
        try:
            decoded = loads_exact_json(line)
        except ExactJSONError as exc:
            raise RuntimeError(
                f"gate-attempt ledger line {line_number} is invalid JSON: {exc}") from exc
        record = _validate_attempt_record(decoded, line_number)
        manifest_hash = record["manifest_hash"]
        counts[manifest_hash] = counts.get(manifest_hash, 0) + 1
        if "attempt_number" in record:
            numbered_records_seen = True
            if record["attempt_number"] != counts[manifest_hash]:
                raise RuntimeError(
                    f"gate-attempt ledger line {line_number} has attempt number "
                    f"{record['attempt_number']}, expected {counts[manifest_hash]} for its "
                    "manifest")
        elif numbered_records_seen:
            raise RuntimeError(
                f"gate-attempt ledger line {line_number} uses the legacy schema after numbered "
                "attempts began; its publication order is ambiguous")
        records.append(record)
    return records


def _attempt_publication_is_latest_unlocked(result: GateResult) -> bool:
    """Validate one result's ledger identity and report whether it owns the phase canonical."""
    records = _read_attempt_ledger_unlocked()
    phase_records = [record for record in records if record["phase"] == result.phase]
    numbered_phase_records = [
        record for record in phase_records if "attempt_number" in record]
    if result.attempt_number is None:
        if numbered_phase_records:
            raise RuntimeError(
                f"unnumbered gate result for phase {result.phase!r} cannot publish after numbered "
                "attempts exist")
        if phase_records:
            latest_legacy = phase_records[-1]
            if (latest_legacy["manifest_hash"] != result.manifest_hash
                    or latest_legacy["passed"] is not result.passed):
                raise RuntimeError(
                    f"unnumbered gate result for phase {result.phase!r} disagrees with the latest "
                    "legacy ledger verdict")
        return True
    if type(result.attempt_number) is not int or result.attempt_number < 1:
        raise RuntimeError("gate result attempt_number must be a positive integer or null")
    matching = [
        record for record in numbered_phase_records
        if record["manifest_hash"] == result.manifest_hash
        and record["attempt_number"] == result.attempt_number
    ]
    if len(matching) != 1 or matching[0]["passed"] is not result.passed:
        raise RuntimeError(
            f"gate result attempt {result.attempt_number} for phase {result.phase!r} has no exact "
            "durable ledger record")
    latest = phase_records[-1]
    return (
        latest.get("attempt_number") == result.attempt_number
        and latest["manifest_hash"] == result.manifest_hash
    )


def _append_attempt_unlocked(record: dict[str, object]) -> None:
    """Append one complete line and durably publish a newly created ledger name."""
    log = RESULTS / "attempts.jsonl"
    created = not log.exists()
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        _fsync_directory(log.parent)


def _record_attempt(
        phase: str,
        manifest_hash: str,
        passed: bool,
        *,
        manifest_name: str,
        max_gate_attempts: int) -> int:
    """Atomically validate the ledger, enforce the budget, and append this attempt."""
    with _attempt_ledger_lock():
        records = _read_attempt_ledger_unlocked()
        prior = [item for item in records if item["manifest_hash"] == manifest_hash]
        if len(prior) >= max_gate_attempts:
            raise RuntimeError(
                f"manifest {manifest_name!r} has already been gated {len(prior)} times "
                f"(limit {max_gate_attempts}). Its acceptance set is spent: each attempt "
                "returned information about it, so further submissions tune against it. Reserve "
                "a fresh set from data none of these attempts touched.")
        attempt_number = len(prior) + 1
        record = {
            "phase": phase,
            "manifest_hash": manifest_hash,
            "passed": passed,
            "attempt_number": attempt_number,
        }
        _validate_attempt_record(record, 0)
        _append_attempt_unlocked(record)
        return attempt_number


def require_passed(phase: str, manifest_hash: str) -> GateResult:
    """Revalidate the canonical result at one serialization point in attempt order."""
    with _attempt_ledger_lock():
        return _require_passed_unlocked(phase, manifest_hash)


def _require_passed_unlocked(phase: str, manifest_hash: str) -> GateResult:
    """Refuse progression unless the registered phase has a verified passing artifact on disk."""
    _validate_name(phase, "phase")
    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise ValueError("the current manifest hash is mandatory when requiring a gate")
    path = RESULTS / f"{phase}.json"
    if not path.exists():
        raise RuntimeError(
            f"phase {phase!r} has no gate result. {phase} must run and pass before this phase "
            "may start.")
    try:
        blob = load_exact_json(path)
    except ExactJSONError as exc:
        raise RuntimeError(f"gate result for {phase!r} is unreadable: {exc}") from exc
    stored = blob.pop("hash", None)
    if content_hash(blob) != stored:
        raise RuntimeError(f"gate result for {phase!r} has been edited since it was written")
    raw_scored_origins = blob.get("scored_origins", [])
    if type(raw_scored_origins) is not list:
        raise RuntimeError(f"result for {phase!r} has non-array scored origins")
    if type(blob.get("extra_scores", {})) is not dict:
        raise RuntimeError(f"result for {phase!r} has non-object extra scores")
    blob["scored_origins"] = tuple(raw_scored_origins)
    blob["reasons"] = tuple(blob.get("reasons", ()))
    result = GateResult(**blob)

    if type(result.passed) is not bool:
        raise RuntimeError(f"result for {phase!r} has a non-boolean passed verdict")
    if type(result.expected_cells) is not int or type(result.scored_cells) is not int:
        raise RuntimeError(f"result for {phase!r} has non-integer grid counts")
    if (result.attempt_number is not None
            and (type(result.attempt_number) is not int or result.attempt_number < 1)):
        raise RuntimeError(f"result for {phase!r} has an invalid attempt number")
    if not _attempt_publication_is_latest_unlocked(result):
        raise RuntimeError(
            f"result for {phase!r} is not the latest recorded gate attempt and cannot unlock the "
            "next phase")

    # Recompute rather than trusting a hand-authored passed=True and a
    # self-consistent hash -- candidate 99.0 against comparator 1.0 was accepted. Hashing proves
    # the file has not been EDITED; it does not prove it was PRODUCED by evaluate_gate. That is
    # `baseline_gate = True` in a different shape, so the stored flag is now advisory and the
    # registered gate is the authority.
    gate = RegisteredGate.load(phase)
    if result.phase != phase:
        raise RuntimeError(
            f"result stored for {phase!r} claims to belong to phase {result.phase!r}")
    if result.gate_hash != gate.hash:
        raise RuntimeError(
            f"result for {phase!r} was scored against a DIFFERENT gate than the one registered "
            "now -- rerun the evaluation against the current gate")
    if result.metric != gate.metric:
        raise RuntimeError(
            f"result for {phase!r} reports metric {result.metric!r}, but the gate requires "
            f"{gate.metric!r}")
    if not _json_values_equal_exact(
            result.measurement_evidence, gate.measurement_contract):
        raise RuntimeError(
            f"result for {phase!r} does not attest the registered measurement_contract. "
            "Its extra scores may have been measured with different origins, levels, path "
            "counts, or cohort")
    if result.manifest_hash != manifest_hash:
        raise RuntimeError(
            f"result for {phase!r} used manifest {result.manifest_hash[:12]}, but this phase is "
            f"running under {manifest_hash[:12]} -- the comparison is not on the same grid")
    manifest = _manifest_with_hash(manifest_hash)
    if result.expected_cells <= 0 or result.scored_cells <= 0 \
            or result.expected_cells != result.scored_cells:
        raise RuntimeError(
            f"result for {phase!r} does not prove a complete positive grid: expected_cells="
            f"{result.expected_cells}, scored_cells={result.scored_cells}. A zero-cell or partial "
            "artifact cannot unlock a later phase")
    if (any(type(origin) is not str for origin in result.scored_origins)
            or len(result.scored_origins) != len(manifest.acceptance_origins)
            or set(result.scored_origins) != set(manifest.acceptance_origins)
            or result.scored_origins != tuple(sorted(result.scored_origins))):
        raise RuntimeError(
            f"result for {phase!r} reports origins {result.scored_origins!r}, but registered "
            f"manifest {manifest.name!r} reserved {sorted(manifest.acceptance_origins)}. A result "
            "scored on development or on no origins cannot unlock a later phase")
    try:
        extras_ok, extra_reasons = gate.check_extras(result.extra_scores)
        pc_ok, pc_reason = gate.check_per_client(
            result.per_client,
            result.comparator_per_client,
            clients=manifest.clients)
        primary_ok = gate.passes(result.comparator_scores, result.candidate_score)
    except ValueError as exc:
        raise RuntimeError(f"result for {phase!r} contains invalid gate evidence: {exc}") from exc
    recomputed = primary_ok and extras_ok and pc_ok
    why = [f"{'PASS' if primary_ok else 'FAIL'} {gate.metric} (primary)"] + extra_reasons + [pc_reason]
    if recomputed != result.passed:
        raise RuntimeError(
            f"stored verdict for {phase!r} says passed={result.passed}, but the registered gate "
            f"recomputes {recomputed}. The artifact was not produced by evaluate_gate.\n  "
            + "\n  ".join(why))
    if not recomputed:
        raise RuntimeError(
            f"phase {phase!r} FAILED its registered gate. A failed gate is a valid outcome -- "
            "stop, or amend the protocol. Do not proceed.\n  " + "\n  ".join(why))
    return result


def _manifest_with_hash(wanted: str) -> EvaluationManifest:
    """Load and revalidate the registered manifest identified by ``wanted``.

    ``require_passed`` previously trusted only the hash copied into the result.  That let a
    self-consistently hashed artifact claim zero cells and no acceptance origins: the numeric gate
    recomputed to PASS because the structural acceptance rules were never replayed.  The manifest
    is the authority for those rules, so requiring a result necessarily requires finding it.
    """
    for path in sorted(REGISTRY.glob("manifest_*.json")):
        try:
            record = load_exact_json(path)
        except ExactJSONError:
            continue
        if record.get("hash") != wanted:
            continue
        raw = dict(record.get("manifest", {}))
        if content_hash(raw) != wanted:
            raise RuntimeError(f"registered manifest {path.name} has been edited: hash mismatch")
        for field_name in ("origins", "scoring_grain", "clients", "acceptance_origins"):
            raw[field_name] = tuple(raw.get(field_name, ()))
        manifest = EvaluationManifest(**raw)
        if manifest.hash != wanted:
            raise RuntimeError(f"registered manifest {path.name} does not reconstruct to its hash")
        return manifest
    raise RuntimeError(
        f"no registered manifest has hash {wanted[:12]}. A result cannot unlock a phase when the "
        "acceptance set it claims to use is unavailable")
