"""The panel contract: what every adapter is handed, and what it may assume.

A `Panel` is a POINTER to Parquet plus the roles its columns play. It is deliberately not a
DataFrame. Adapters are third-party code by design -- `statsforecast`, `autogluon`,
`pytorch-forecasting` -- and an object crossing that boundary is a promise its methods still mean
what they meant when the harness checked them. A path and a set of column names cannot be
mutated out from under a check.

The observation vocabulary is NOT redefined here. `ObservationState` lives in
`spec/forecast_spec.py`, the product surface above both packages, and is imported by both.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import pathlib
import sys
from dataclasses import dataclass, field

import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pyarrow.types as patypes

from autoforecast.column_names import validate_column_names
from autoforecast.frequency import (
    FrequencyError,
    TimePoint,
    advance,
    canonical_origin,
    parse_frequency,
    parse_origin,
    period_points,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
# noqa: E402 -- sys.path must be extended before the spec can be imported.
from spec.forecast_spec import ELIGIBLE_STATES, ObservationState  # noqa: E402


class PanelError(ValueError):
    """A panel does not describe the Parquet it points at, or describes it inconsistently."""


def period_days(frequency: str) -> int:
    """Compatibility helper for callers that still require an integral-day stride.

    Hourly, monthly, and yearly cadences must use :mod:`autoforecast.frequency`; they are never
    approximated as a number of days.
    """
    try:
        cadence = parse_frequency(frequency)
    except FrequencyError as exc:
        raise PanelError(str(exc)) from exc
    if cadence.unit == "W":
        return 7
    if cadence.unit == "D":
        return cadence.multiplier
    raise PanelError(
        f"frequency {frequency!r} is not an integral-day cadence; use period_points instead"
    )


class _DefaultSeasonLength(int):
    """Distinguish an omitted default from an explicitly supplied integer without changing API."""


_DEFAULT_SEASON_LENGTH = _DefaultSeasonLength(7)
_SEASON_DEFAULTS = {"H": 24, "D": 7, "W": 52, "M": 12, "Y": 1}


@dataclass(frozen=True)
class PlanSource:
    """An effective-dated plan feed: PROOF of what was known at the origin, not a declaration.

    `known_future` alone says a column is plannable; it does not prove the value used at a
    given origin was known then -- a price corrected after the fact silently rewrites history
    in a backtest. A plan source carries `valid_date` (the day the value applies to),
    `known_at` (when the record became known -- the as-of filter), and `snapshot` (which
    extract, ranking two records for one day by recency of knowledge). The adapter selects
    as-of each origin and binds the selected rows' content hash into the run record.

    Attributes:
        path: Parquet of effective-dated records.
        valid_date_col: The day each record applies to.
        known_at_col: When the record became known.
        snapshot_col: Extract identifier for tie-breaks, or None to rank by known_at alone.
        price_col: Plan column holding the planned price, if any.
        promo_col: Plan column holding the planned promotion state, if any.
    """

    path: str
    valid_date_col: str
    known_at_col: str
    snapshot_col: str | None = None
    price_col: str | None = None
    promo_col: str | None = None

    def __post_init__(self) -> None:
        if self.price_col is None and self.promo_col is None:
            raise PanelError("a plan source must carry a price or a promotion column")
        try:
            validate_column_names(c for c in (
                self.valid_date_col, self.known_at_col, self.snapshot_col,
                self.price_col, self.promo_col,
            ) if c is not None)
        except ValueError as exc:
            raise PanelError(str(exc)) from exc
        object.__setattr__(self, "path", str(self.path))


@dataclass(frozen=True)
class Panel:
    """A dataset an adapter can forecast, described by path and column roles.

    Attributes:
        path: Parquet file or partitioned directory. Normalised to `str` on construction.
        series_col: Identifies one time series. Forecasts are produced per distinct value.
        time_col: The cadence key: a timezone-naive exact-hour timestamp for H, and a date or
            exact-midnight timestamp for date cadences. One row per (series, time) -- `validate`
            does not check uniqueness, which needs a full scan; scoring checks it in-window.
        target_col: The quantity to forecast.
        status_prob_cols: SOFT observation states -- `{state: probability column}` over
            exactly the three contract states, for feeds where the state is inferred rather
            than measured. Mutually exclusive with `exposure_col`, which is the observation
            contract's own rule. The likelihood marginalises over these; the encoder conditions
            on the modal reading.
        exposure_col: Holds `ObservationState` per cell. `None` means the dataset carries no
            exposure information at all, which is true of most public benchmarks -- it is a
            property of the data, not a switch, and adapters see it through `has_exposure`.
        oos_col: Optional boolean out-of-stock input feature. This is separate from
            `exposure_col`: a zero-sale stockout is likelihood-unobserved (its lower bound is
            vacuous) but must remain visible to the encoder as shelf state.
        known_future: Columns known for the whole horizon at forecast time (price, planned promo).
        past_only: Optional historical numeric covariates. Values after a forecast origin are
            unavailable even when present in the source Parquet.
        price_col: Which column is the PRICE, if this client has one. Declared rather than
            detected: an adapter that sniffs for a known name silently drops the commercial branch
            on a dataset that calls it something else -- for example `price` or `sell_price` --
            and the model then looks as though it does not
            support price at all rather than as though it was never given it.
        promo_col: Which column is the PROMOTION flag, if this client has one. Same reasoning.
        static: Columns constant per series, CATEGORICAL (category, department). One-hot
            encoded by the adapter, `MAX_ONE_HOT` guarded.
        static_reals: Columns constant per series, NUMERIC (pack weight, shelf capacity).
            Standardised on the training panel and appended to the identity attributes; each
            real is its own variable-selection group. Declared separately from `static` because
            a numeric column one-hot encoded explodes its vocabulary and a categorical fed
            through standardisation invents an order.
        factor_group: Column a shared latent factor is indexed by, if the dataset supports one.
            **Declared, never inferred.** `ForecastDataSpec.factor_group_col` records why: the
            hierarchy gives the levels but not which one carries the factor, and chain, region and
            banner are all defensible and give DIFFERENT models. The custom adapter therefore
            follows `ForecastModelSpec.resolve`: without this declaration it removes the shared
            factor and uses plain bottom-up independent predictive marginals. It must never invent
            a single global group, which would make every series move together for no declared
            business reason.
        product_col: Which column identifies the PRODUCT within a series. The likelihood
            contract indexes the shared factor by `(factor_group, product, step)` -- one drawn
            path per product per group, with STORES loading on it -- so an adapter that cannot
            tell products apart cannot build that index and degrades to one path per group,
            which couples every product in the group to one shock. Like `factor_group`, this may
            name a column that is also a static attribute; it is an index role, not a feature.
        store_col: Which column selects a series' factor LOADING row. Distinct from
            `factor_group` in principle: the group says which shock a series follows, the
            loading says how hard it is hit.
        plan: Effective-dated plan feed (`PlanSource`) proving what was known at each origin.
            When declared, the adapter reads HORIZON prices and promotions from the plan
            selected as-of the origin -- never from the panel's own future rows, which are
            recorded outcomes wearing a plan's name -- and binds the selection's content hash
            into the checkpoint. Without it, horizon commercial values fall back to the
            panel's rows, which is a declared backtest convenience, not provenance.
        category_col: Which level anchors the COLD-START priors: the frequency baseline is
            Beta-shrunk toward this level's selling rate, the size baseline toward its mean
            positive size. Without it both fall back to panel-wide figures, which on a mixed
            panel pulls every launch toward the fast-moving category's numbers.
        negative_target_policy: What happens to a negative recorded quantity -- the decision
            ``ForecastDataSpec.target_allows_negative`` says to make HERE rather than discover
            in a training crash. 'refuse' (default): a negative reaching the likelihood raises,
            loudly. 'floor': returns and corrections are clamped to zero at ingestion -- a
            declared cleansing rule, chosen by the caller who knows what their negatives mean.
            Routing returns to their own series is client ETL work, not a panel switch.
        target_estimand: Meaning of ``target_col``. ``observed_nonnegative`` is a measured
            nonnegative quantity, including fractional measurements. ``sales_given_availability`` is the observed
            sold quantity. ``unconstrained_demand`` is latent demand and may be requested only
            from an adapter that explicitly handles right-censored observations. This role is
            carried through every Forecast so a metric cannot compare different quantities.
        season_length: Periods in one seasonal cycle. Default 7 represents a weekly cycle
            in daily observations. Other cycles must be declared; nothing infers them.
    """

    path: str
    series_col: str
    time_col: str
    target_col: str
    exposure_col: str | None = None
    status_prob_cols: dict[str, str] | None = None
    oos_col: str | None = None
    known_future: tuple[str, ...] = ()
    price_col: str | None = None
    promo_col: str | None = None
    static: tuple[str, ...] = ()
    static_reals: tuple[str, ...] = ()
    factor_group: str | None = None
    product_col: str | None = None
    store_col: str | None = None
    category_col: str | None = None
    plan: PlanSource | None = None
    season_length: int = _DEFAULT_SEASON_LENGTH
    frequency: str = "D"
    target_estimand: str = "sales_given_availability"
    negative_target_policy: str = "refuse"
    past_only: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalise `path` so a `pathlib.Path` caller and a `str` caller hash identically."""
        if not isinstance(self.path, str | pathlib.Path):
            raise PanelError(f"panel path must be a string or Path, got {type(self.path).__name__}")
        if not str(self.path).strip():
            raise PanelError("panel path cannot be blank")
        object.__setattr__(self, "path", str(self.path))
        if isinstance(self.season_length, _DefaultSeasonLength):
            try:
                default = _SEASON_DEFAULTS[parse_frequency(self.frequency).unit]
            except FrequencyError as exc:
                raise PanelError(str(exc)) from exc
            object.__setattr__(self, "season_length", default)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Panel:
        """Restore the typed Panel contract from a JSON/checkpoint mapping.

        JSON turns tuples into arrays and nested dataclasses into objects.  Reconstruct both
        explicitly so process-isolated and replay entry points receive the same types as an
        in-process caller.  Missing optional fields retain dataclass defaults for old records;
        unknown fields fail closed instead of being silently discarded.
        """
        if type(record) is not dict:
            raise PanelError("panel record must be one JSON object")
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(record) - fields)
        if unknown:
            raise PanelError(f"panel record has unknown field(s) {unknown}")
        restored = dict(record)
        for name in ("known_future", "static", "static_reals", "past_only"):
            value = restored.get(name, ())
            if type(value) not in (list, tuple):
                raise PanelError(f"panel record {name} must be a JSON array")
            restored[name] = tuple(value)
        plan = restored.get("plan")
        if type(plan) is dict:
            try:
                restored["plan"] = PlanSource(**plan)
            except (PanelError, TypeError) as exc:
                raise PanelError(f"panel record has malformed PlanSource: {exc}") from exc
        elif plan is not None and not isinstance(plan, PlanSource):
            raise PanelError("panel record plan must be a PlanSource object or null")
        try:
            return cls(**restored)
        except TypeError as exc:
            raise PanelError(f"panel record has malformed fields: {exc}") from exc

    @property
    def has_exposure(self) -> bool:
        """Whether this panel can distinguish an observed zero from an ineligible cell."""
        return self.exposure_col is not None

    @property
    def role_columns(self) -> tuple[str, ...]:
        """Every distinct column the panel claims a role for, in a fixed order.

        Semantic/index roles commonly alias feature roles: a price is normally also a
        ``known_future`` covariate, and a product id is normally also ``static``.  Return each
        physical column once while preserving the declaration order.
        """
        cols = [self.series_col, self.time_col, self.target_col]
        if self.exposure_col is not None:
            cols.append(self.exposure_col)
        if self.status_prob_cols is not None:
            cols.extend(self.status_prob_cols[k] for k in sorted(self.status_prob_cols))
        if self.oos_col is not None:
            cols.append(self.oos_col)
        cols.extend(self.known_future)
        cols.extend(self.past_only)
        cols.extend(self.static)
        cols.extend(self.static_reals)
        cols.extend(c for c in (
            self.price_col,
            self.promo_col,
            self.factor_group,
            self.product_col,
            self.store_col,
            self.category_col,
        ) if c is not None)
        return tuple(dict.fromkeys(cols))

    def validate_column_names(self) -> None:
        """Validate mapped identifiers without opening data or starting a runtime."""
        try:
            names = list(self.role_columns)
            validate_column_names(names)
        except (TypeError, KeyError, AttributeError, ValueError) as exc:
            raise PanelError(f"invalid panel column mapping: {exc}") from exc

    def validate(self) -> Panel:
        """Refuse a panel that does not describe its own Parquet. Returns self so it can chain.

        Checks the Parquet SCHEMA, not just the strings the caller passed. A column name that is
        merely present in the dataclass proves nothing about the file, and every
        adapter downstream would fail later and less legibly.

        Raises:
            PanelError: the path does not exist; a role column is missing from the Parquet
                schema; one column is given two roles; the target is also declared known-future;
                or `season_length` is not positive.
        """
        p = pathlib.Path(self.path)
        if not p.exists():
            raise PanelError(f"panel path does not exist: {self.path}")
        if self.plan is not None and not isinstance(self.plan, PlanSource):
            raise PanelError("plan must be a PlanSource or None")
        if self.status_prob_cols is not None:
            if not isinstance(self.status_prob_cols, dict):
                raise PanelError("status_prob_cols must be a dict of state to column name")
            if self.exposure_col is not None:
                raise PanelError(
                    "status_col OR status_prob_cols, not both -- the observation contract's "
                    "own rule (ObservationSpec)")
            expected = {"exact", "lower_bound", "unobserved"}
            if set(self.status_prob_cols) != expected:
                raise PanelError(
                    f"status_prob_cols must cover exactly {sorted(expected)}, got "
                    f"{sorted(self.status_prob_cols)}")
            invalid_prob_columns = {
                state: column
                for state, column in self.status_prob_cols.items()
                if not isinstance(column, str) or not column.strip()
            }
            if invalid_prob_columns:
                raise PanelError(
                    "status_prob_cols must map every state to a non-empty column name, got "
                    f"{invalid_prob_columns!r}"
                )
        if (not isinstance(self.season_length, int) or isinstance(self.season_length, bool)
                or self.season_length < 1):
            raise PanelError(f"season_length must be >= 1, got {self.season_length}")
        try:
            cadence = parse_frequency(self.frequency)
        except FrequencyError as exc:
            raise PanelError(str(exc)) from exc
        if self.target_estimand not in ("sales_given_availability", "unconstrained_demand",
                                        "observed_count", "observed_nonnegative"):
            raise PanelError(f"unsupported target_estimand {self.target_estimand!r}")
        if self.negative_target_policy not in ("refuse", "floor"):
            raise PanelError(
                f"unsupported negative_target_policy {self.negative_target_policy!r}; the "
                "contract defines refuse and floor, and routing returns to their own series "
                "is client ETL work rather than a panel switch")
        if self.target_estimand == "unconstrained_demand" and self.exposure_col is None \
                and self.status_prob_cols is None:
            raise PanelError(
                "unconstrained_demand requires an observation contract -- an exposure/state "
                "column or status_prob_cols; without right-censoring evidence it is "
                "indistinguishable from observed sales")

        scalar_roles = [
            ("series_col", self.series_col, False),
            ("time_col", self.time_col, False),
            ("target_col", self.target_col, False),
            ("exposure_col", self.exposure_col, True),
            ("oos_col", self.oos_col, True),
            ("price_col", self.price_col, True),
            ("promo_col", self.promo_col, True),
            ("factor_group", self.factor_group, True),
            ("product_col", self.product_col, True),
            ("store_col", self.store_col, True),
            ("category_col", self.category_col, True),
        ]
        for role, column, optional in scalar_roles:
            if optional and column is None:
                continue
            if not isinstance(column, str) or not column.strip():
                raise PanelError(
                    f"{role} must name a non-empty column"
                    + (" or be None" if optional else "")
                    + f", got {column!r}"
                )
        for role, columns in (
            ("known_future", self.known_future),
            ("past_only", self.past_only),
            ("static", self.static),
            ("static_reals", self.static_reals),
        ):
            if not isinstance(columns, tuple):
                raise PanelError(f"{role} must be a tuple of column names, got {columns!r}")
            for index, column in enumerate(columns):
                if not isinstance(column, str) or not column.strip():
                    raise PanelError(
                        f"{role}[{index}] must name a non-empty column, got {column!r}"
                    )

        if self.target_col in self.known_future:
            raise PanelError(
                f"target_col {self.target_col!r} is also declared known_future -- that is the "
                "target leaking into the features, and it is silent unless refused here"
            )

        # Feature roles describe how a value is consumed and must be unambiguous.  The semantic
        # roles below are different: they say what a column MEANS, and are deliberately allowed
        # to point at a static/known-future feature column.
        feature_roles = [self.series_col, self.time_col, self.target_col]
        if self.exposure_col is not None:
            feature_roles.append(self.exposure_col)
        if self.status_prob_cols is not None:
            feature_roles.extend(self.status_prob_cols.values())
        if self.oos_col is not None:
            feature_roles.append(self.oos_col)
        feature_roles.extend(self.known_future)
        feature_roles.extend(self.past_only)
        feature_roles.extend(self.static)
        feature_roles.extend(self.static_reals)
        dupes = sorted({c for c in feature_roles if feature_roles.count(c) > 1})
        if dupes:
            raise PanelError(f"columns given more than one role: {dupes}")

        semantic = [(name, column) for name, column, _ in scalar_roles[5:]
                    if column is not None]
        semantic_dupes = sorted({column for _, column in semantic
                                 if sum(c == column for _, c in semantic) > 1})
        if semantic_dupes:
            owners = {column: [name for name, c in semantic if c == column]
                      for column in semantic_dupes}
            raise PanelError(f"columns given ambiguous semantic/index roles: {owners}")
        protected = {
            self.series_col: "series_col",
            self.time_col: "time_col",
            self.target_col: "target_col",
        }
        if self.exposure_col is not None:
            protected[self.exposure_col] = "exposure_col"
        if self.oos_col is not None:
            protected[self.oos_col] = "oos_col"
        collisions = [(role, column, protected[column]) for role, column in semantic
                      if column in protected]
        if collisions:
            raise PanelError(
                "semantic/index roles may alias static or known_future columns, but not core "
                f"roles: {collisions}"
            )

        self.validate_column_names()
        roles = self.role_columns
        schema = pq.read_schema(_any_fragment(p))
        if len(schema.names) != len(set(schema.names)):
            raise PanelError("Parquet column names must be unique; rename duplicate headers")
        present = set(schema.names) | _partition_columns(p)
        try:
            validate_column_names(present)
        except ValueError as exc:
            raise PanelError(str(exc)) from exc
        missing = [c for c in roles if c not in present]
        if missing:
            raise PanelError(
                f"columns claimed by the panel but absent from the Parquet schema: {missing}. "
                f"present: {sorted(present)}"
            )
        fields = {field.name: field.type for field in schema}
        time_type = fields.get(self.time_col)
        if cadence.key_kind == "timestamp":
            if not patypes.is_timestamp(time_type):
                raise PanelError(
                    f"time_col={self.time_col!r} for hourly cadence requires a timezone-naive "
                    f"Arrow timestamp, got {time_type}")
        else:
            _require_type(fields, self.time_col, "time_col", patypes.is_date,
                          patypes.is_timestamp)
        if patypes.is_timestamp(time_type) and getattr(time_type, "tz", None) is not None:
            raise PanelError(
                f"time_col={self.time_col!r} must be timezone-naive, got {time_type}")
        _require_type(fields, self.target_col, "target_col", patypes.is_integer,
                      patypes.is_floating, patypes.is_decimal)
        for column in self.static_reals:
            _require_type(fields, column, f"static_reals[{column!r}]", patypes.is_integer,
                          patypes.is_floating, patypes.is_decimal)
        for column in self.past_only:
            _require_type(fields, column, f"past_only[{column!r}]", patypes.is_integer,
                          patypes.is_floating, patypes.is_decimal, patypes.is_boolean)
        if self.price_col is not None:
            _require_type(fields, self.price_col, "price_col", patypes.is_integer,
                          patypes.is_floating, patypes.is_decimal)
        if self.promo_col is not None:
            _require_type(fields, self.promo_col, "promo_col", patypes.is_boolean,
                          patypes.is_integer, patypes.is_floating, patypes.is_decimal)
        if self.exposure_col is not None:
            _require_type(fields, self.exposure_col, "exposure_col", patypes.is_string,
                          patypes.is_large_string)
        if self.oos_col is not None:
            _require_type(fields, self.oos_col, "oos_col", patypes.is_boolean)
        if self.status_prob_cols is not None:
            for state, column in self.status_prob_cols.items():
                _require_type(fields, column, f"status_prob_cols[{state!r}]", patypes.is_integer,
                              patypes.is_floating, patypes.is_decimal)
        return self


def _require_type(fields: dict[str, object], column: str, role: str, *predicates) -> None:
    """Refuse a role whose physical Arrow type contradicts its declared semantics."""
    column_type = fields.get(column)
    if column_type is None:
        raise PanelError(f"{role}={column!r} is partition-only and has no physical Arrow type")
    if not any(predicate(column_type) for predicate in predicates):
        raise PanelError(f"{role}={column!r} has incompatible Arrow type {column_type}")


def _partition_columns(root: pathlib.Path) -> set[str]:
    """Hive-style partition keys under `root`, read from the directory names.

    A partitioned Parquet dataset stores its partition column in the DIRECTORY NAME
    (`store_id=CA_1/`) and not in any fragment's schema, so a schema read alone cannot see it --
    and `validate` would then reject a column Spark reads perfectly well. M5 is partitioned by
    store and every WRMSSE level above the item uses that column, so this is not an edge case.
    """
    if root.is_file():
        return set()
    keys = set()
    for child in root.rglob("*"):
        if child.is_dir() and "=" in child.name:
            keys.add(child.name.split("=", 1)[0])
    return keys


def _any_fragment(p: pathlib.Path) -> pathlib.Path:
    """Return a single Parquet file from `p`, whether it is a file or a partitioned directory.

    Reading one fragment's schema is O(1); opening the whole dataset walks every partition, which
    on a client estate with 100k files is the difference between milliseconds and minutes.

    LIMIT, stated because it was hit: this does NOT detect a directory whose fragments disagree
    with each other. `m5.build_parquet` appended rather than replacing, leaving two generations
    with different date types in one directory -- `validate` passed on the first fragment and
    Spark failed on the second. The fix belongs with the PRODUCER, which now clears its output;
    checking every footer here would push a per-file cost onto every panel to catch a defect only
    a broken writer can create.

    Raises:
        PanelError: `p` is a directory containing no `.parquet` file.
    """
    if p.is_file():
        return p
    for frag in sorted(p.rglob("*.parquet")):
        return frag
    raise PanelError(f"no .parquet fragment under {p}")


def _as_time_key(value: object, frequency: str, adapter: str) -> TimePoint:
    """Normalise one Arrow forecast key without truncating any timestamp component."""
    cadence = parse_frequency(frequency)
    try:
        if isinstance(value, str):
            return parse_origin(value, cadence)
        if cadence.key_kind == "timestamp":
            if not isinstance(value, dt.datetime):
                raise FrequencyError(f"{cadence.code} requires a timestamp key, got {value!r}")
            return parse_origin(canonical_origin(value, cadence), cadence)
        if isinstance(value, dt.datetime):
            if value.tzinfo is not None or value.time() != dt.time():
                raise FrequencyError(
                    f"{cadence.code} date key cannot discard a non-midnight or timezone-aware "
                    f"timestamp: {value!r}"
                )
            value = value.date()
        if not isinstance(value, dt.date):
            raise FrequencyError(f"{cadence.code} requires a date key, got {value!r}")
        # Asking for a zero-step calendar point validates M/Y anchor semantics.
        return advance(value, cadence, 0)
    except FrequencyError as exc:
        raise ForecastError(f"{adapter!r}: invalid forecast time key: {exc}") from exc


@dataclass(frozen=True)
class Forecast:
    """What an adapter returns: a pointer to Parquet, plus an account of every cell requested.

    Attributes:
        adapter: Registry name of the implementation that produced this.
        path: Parquet of `series · date · yhat`, where `date` is the cadence-dependent time key.
            A path, for the same reason `Panel` carries one.
        origin: Last key the adapter was allowed to see: canonical exact-hour ISO for H and an
            ISO date for all date cadences.
        horizon: Periods forecast beyond `origin`.
        n_series: Distinct series the adapter was asked to forecast.
        disposition: Bucket -> count, covering EVERY requested cell exactly once. Buckets whose
            name starts with `emitted` were written to `path`; the rest were declined, with the
            reason as the bucket name. `validate` requires the total to equal
            `n_series * horizon`, which is what makes a silent shortfall impossible: an adapter
            that quietly drops the hard cells would otherwise score well on what is left.
        target_estimand: Quantity forecast by ``yhat``. It must match both the input Panel and the
            evaluation manifest; the scorer rejects cross-estimand comparisons.
        frequency: Declared cadence. The horizon counts periods.
    """

    adapter: str
    path: str
    origin: str
    horizon: int
    n_series: int
    disposition: dict[str, int] = field(default_factory=dict)
    target_estimand: str = "sales_given_availability"
    frequency: str = "D"

    def __post_init__(self) -> None:
        """Normalise `path`, matching `Panel`."""
        if not isinstance(self.path, str | pathlib.Path):
            raise ForecastError(
                f"{self.adapter!r}: forecast path must be a string or Path, got "
                f"{type(self.path).__name__}")
        if not str(self.path).strip():
            raise ForecastError(f"{self.adapter!r}: forecast path cannot be blank")
        object.__setattr__(self, "path", str(self.path))

    @property
    def n_emitted(self) -> int:
        """Cells actually written to `path` -- the buckets whose names begin with `emitted`."""
        return sum(v for k, v in self.disposition.items() if k.startswith("emitted"))

    @property
    def n_declined(self) -> int:
        """Cells the adapter refused, for any reason."""
        return sum(v for k, v in self.disposition.items() if not k.startswith("emitted"))

    def validate(self) -> Forecast:
        """Refuse a forecast whose manifest or existing Parquet artifact is inconsistent. Chains.

        Raises:
            ForecastError: the disposition does not total `n_series * horizon`, a bucket count is
                negative, `horizon`/`n_series` is not positive, or an existing Parquet has the
                wrong schema, row count, keys, dates, or forecast values.
        """
        if not isinstance(self.adapter, str) or not self.adapter.strip():
            raise ForecastError("forecast adapter must be a non-blank string")
        try:
            parse_origin(self.origin, self.frequency)
        except FrequencyError as exc:
            raise ForecastError(
                f"{self.adapter!r}: invalid origin/frequency contract: {exc}") from exc
        if (not isinstance(self.horizon, int) or isinstance(self.horizon, bool)
                or not isinstance(self.n_series, int) or isinstance(self.n_series, bool)
                or self.horizon < 1 or self.n_series < 1):
            raise ForecastError(
                f"{self.adapter!r}: horizon and n_series must be >= 1, got "
                f"horizon={self.horizon}, n_series={self.n_series}"
            )
        try:
            period_points(self.origin, self.frequency, self.horizon)
        except (FrequencyError, OverflowError) as exc:
            raise ForecastError(
                f"{self.adapter!r}: frequency {self.frequency!r} does not produce a supported "
                "representable forecast calendar"
            ) from exc
        if self.target_estimand not in ("sales_given_availability", "unconstrained_demand",
                                        "observed_count", "observed_nonnegative"):
            raise ForecastError(
                f"{self.adapter!r}: unsupported target_estimand {self.target_estimand!r}")
        if not isinstance(self.disposition, dict) or any(
                not isinstance(key, str) or not key.strip() for key in self.disposition):
            raise ForecastError(
                f"{self.adapter!r}: disposition must be a dict with non-blank string keys")
        invalid_counts = sorted(str(k) for k, value in self.disposition.items()
                                if not isinstance(value, int) or isinstance(value, bool))
        if invalid_counts:
            raise ForecastError(
                f"{self.adapter!r}: disposition counts must be integers: {invalid_counts}")
        negative = sorted(k for k, v in self.disposition.items() if v < 0)
        if negative:
            raise ForecastError(f"{self.adapter!r}: negative disposition counts: {negative}")
        requested = self.n_series * self.horizon
        total = sum(self.disposition.values())
        if total != requested:
            raise ForecastError(
                f"{self.adapter!r}: disposition accounts for {total} cells but {requested} were "
                f"requested ({self.n_series} series x {self.horizon} periods). Every requested "
                f"cell must be emitted or declined with a reason. buckets={self.disposition}"
            )
        artifact = pathlib.Path(self.path)
        if artifact.exists():
            self._validate_parquet(artifact)
        return self

    def _validate_parquet(self, artifact: pathlib.Path) -> None:
        """Validate an artifact with Arrow, including Spark-style partition directories.

        A non-existent path is left to the caller because manifests are also constructed before
        materialisation by registry tests and remote runners.  Once the path exists, however, it
        is evidence and must agree with the manifest rather than merely looking Parquet-shaped.
        """
        try:
            dataset = pads.dataset(artifact, format="parquet", partitioning="hive")
        except Exception as exc:
            raise ForecastError(
                f"{self.adapter!r}: forecast path exists but is not a readable Parquet artifact: "
                f"{self.path}: {exc}"
            ) from exc
        required = ("series_id", "date", "yhat")
        missing = [column for column in required if column not in dataset.schema.names]
        if missing:
            raise ForecastError(
                f"{self.adapter!r}: forecast Parquet is missing required columns {missing}; "
                f"present: {dataset.schema.names}"
            )
        cadence = parse_frequency(self.frequency)
        date_type = dataset.schema.field("date").type
        if cadence.key_kind == "timestamp" and (
            not patypes.is_timestamp(date_type) or getattr(date_type, "tz", None) is not None
        ):
            raise ForecastError(
                f"{self.adapter!r}: hourly forecast date must be a timezone-naive Arrow "
                f"timestamp, got {date_type}"
            )
        if patypes.is_timestamp(date_type) and getattr(date_type, "tz", None) is not None:
            raise ForecastError(
                f"{self.adapter!r}: forecast date must be timezone-naive, got {date_type}")
        allowed_dates = set(period_points(self.origin, cadence, self.horizon))

        n_rows = 0
        seen: set[tuple[object, TimePoint]] = set()
        emitted_series: set[object] = set()
        try:
            batches = dataset.scanner(columns=list(required)).to_batches()
            for batch in batches:
                for row in batch.to_pylist():
                    n_rows += 1
                    sid = row["series_id"]
                    if sid is None or (isinstance(sid, str) and not sid.strip()):
                        raise ForecastError(
                            f"{self.adapter!r}: forecast Parquet contains a null/blank series_id"
                        )
                    day = _as_time_key(row["date"], self.frequency, self.adapter)
                    if day not in allowed_dates:
                        raise ForecastError(
                            f"{self.adapter!r}: forecast date {day.isoformat()} is outside the "
                            f"requested window {min(allowed_dates)} through {max(allowed_dates)}"
                        )
                    key = (sid, day)
                    if key in seen:
                        raise ForecastError(
                            f"{self.adapter!r}: duplicate forecast key (series_id, date)={key!r}"
                        )
                    seen.add(key)
                    emitted_series.add(sid)
                    value = row["yhat"]
                    if isinstance(value, bool):
                        raise ForecastError(
                            f"{self.adapter!r}: yhat must be numeric rather than boolean, got "
                            f"{value!r} at {key!r}")
                    try:
                        number = float(value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ForecastError(
                            f"{self.adapter!r}: yhat must be numeric, got {value!r} at {key!r}"
                        ) from exc
                    if not math.isfinite(number) or number < 0.0:
                        raise ForecastError(
                            f"{self.adapter!r}: yhat must be finite and non-negative, got "
                            f"{value!r} at {key!r}"
                        )
        except ForecastError:
            raise
        except Exception as exc:
            raise ForecastError(
                f"{self.adapter!r}: could not inspect forecast Parquet {self.path}: {exc}"
            ) from exc
        if n_rows != self.n_emitted:
            raise ForecastError(
                f"{self.adapter!r}: manifest claims {self.n_emitted} emitted cells but the "
                f"Parquet holds {n_rows}"
            )
        if len(emitted_series) > self.n_series:
            raise ForecastError(
                f"{self.adapter!r}: Parquet emits {len(emitted_series)} distinct series but the "
                f"manifest says only {self.n_series} were requested"
            )

    def as_record(self) -> dict[str, object]:
        """A JSON-safe dict for the run manifest, with the derived totals made explicit."""
        rec = dataclasses.asdict(self)
        rec["n_emitted"] = self.n_emitted
        rec["n_declined"] = self.n_declined
        return rec

    @classmethod
    def from_record(cls, rec: dict[str, object]) -> Forecast:
        """Rebuild from `as_record`, dropping the derived totals it added.

        Used to resume a long gate run without re-running an adapter. `n_emitted` and
        `n_declined` are recomputed from `disposition` rather than trusted from the file, so a
        record that has been edited disagrees with itself and `validate` catches it.
        """
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in rec.items() if k in fields})


class ForecastError(ValueError):
    """A forecast cannot account for the cells it was asked to produce."""


__all__ = [
    "ELIGIBLE_STATES",
    "Forecast",
    "ForecastError",
    "ObservationState",
    "Panel",
    "PanelError",
    "PlanSource",
]
