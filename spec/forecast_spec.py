"""Declarative data and model specifications for daily demand forecasting.

Observation states distinguish exact values, lower bounds, and unobserved cells.
The supported topology forecasts series bottom-up with a shared latent factor.
Use the profiler to measure dataset properties before resolving model defaults."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Finding:
    """A validation result with machine-readable severity and a readable string representation."""
    severity: Literal["error", "note"]
    message: str

    def __str__(self) -> str:
        return f"NOTE: {self.message}" if self.severity == "note" else self.message

    def __eq__(self, other) -> bool:          # so `x in errs` still works against plain strings
        return str(self) == other if isinstance(other, str) else NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))

    def startswith(self, p: str) -> bool:     # keeps `e.startswith("NOTE:")` callers working
        return str(self).startswith(p)
from collections.abc import Sequence
from typing import Literal

# ======================================================================================
# 1. THE OBSERVATION CONTRACT
# ======================================================================================
# The model is never told about stock. It is told, per cell, how the recorded number relates to
# the quantity we want to predict. "Out of stock" is a REASON; the model needs the LIKELIHOOD
# CONTRACT. Separating them is what lets one model serve clients whose stock feeds range from
# excellent to unusable.
ObservationState = Literal["exact", "lower_bound", "unobserved"]

STATE_CODES: dict[str, int] = {"exact": 0, "lower_bound": 1, "unobserved": 2}
"""Integer codes for the tensor path, defined ONCE, here, beside the vocabulary they encode.

The Spark feature builders emit these codes and the torch likelihood consumes them, and the two
run in different planes that must not import each other's engines -- so the mapping lives in the
product surface both already import. Before this existed the trainer rebuilt the mapping from a
BOOLEAN (`eligible -> exact, else unobserved`), which is how `lower_bound` -- the state the
entire censoring contract exists for -- became unreachable from training: a stockout that sold 5
first was scored as exactly 5, and a vacuous all-day stockout as an observed zero.
"""

ELIGIBLE_STATES: frozenset[str] = frozenset({"exact", "lower_bound"})
"""States a cell may hold and still carry information about demand.

`unobserved` is excluded: it covers both "never ranged" and "no data", and a model trained on a
never-ranged cell learns an assortment decision rather than a demand event. `lower_bound` is
INCLUDED -- a stockout day that sold before it ran out still observed an occurrence, and
discarding it would throw away the sale as well as the censoring.

It lives HERE, in the product surface, and not in `autoforecast/panel.py` where it started.
Both packages need it, and the harness owning it meant `forecast/` had to import the harness to
know what an eligible cell is. Keeping the contract here lets the model run independently.
"""


@dataclass
class ObservationSpec:
    status_col: str | None = None
    """Column holding `exact` / `lower_bound` / `unobserved`. None disables censoring."""

    status_prob_cols: dict[ObservationState, str] | None = None
    """Soft alternative: P(state) per cell, for when the state is inferred rather than measured.
    The likelihood marginalises over the states."""

    quality_gate: float | None = 0.10
    """Maximum tolerated share of positive-sales rows under a censoring condition.

    Calibrate this provisional threshold to your feed. Positive sales with zero
    end-of-day stock may represent a sellout; this is a feed-plausibility check.
    A failure disables censoring and range inference from the same untrusted feed."""

    on_gate_failure: Literal["raise", "disable", "infer"] = "disable"
    """What to do when `quality_gate` trips.

    'disable' is the default because a broken stock feed is a DATA problem, not a modelling one,
    and the model should still forecast that client -- just without censoring correction. 'raise'
    stops the run, which is right in a batch where silent degradation would go unnoticed. 'infer'
    attempts to recover the state probabilistically and is the most dangerous option, because it
    produces a plausible number from a feed already known to disagree with itself.

    A tripped gate must also disable RANGE inference wherever that depends on the same stock
    feed -- otherwise the broken signal re-enters through the exposure table."""


# ======================================================================================
# 2. EXPOSURE -- when a cell could have sold at all
# ======================================================================================
@dataclass
class ExposureSpec:
    """A zero means nothing until you know the cell was capable of selling.

    Three different reasons a cell shows zero, which must not be conflated:
      not ranged   -- the product is not listed in that store        -> excluded
      not open     -- the store was shut that day                    -> excluded
      out of stock -- listed and open, but no stock                  -> censored
    """

    ranged_col: str | None = None
    """Effective-dated is_ranged(store, product, date). NOT 'ever stocked in the window' --
    that misclassifies launches and delistings, and leaks the future into the past."""

    store_open_col: str | None = None
    lifecycle_col: str | None = None
    """launch / active / discontinued / superseded. A delisting is not a demand collapse."""

    future_range_plan_col: str | None = None
    """Known-future ranging. A range contract that only looks backwards cannot express
    'this item will be removed from 400 stores next month'."""


# ======================================================================================
# 3. COVARIATE BRANCHES
# ======================================================================================
@dataclass
class Branch:
    reals: Sequence[str] = ()
    categoricals: Sequence[str] = ()
    attend_over: Literal["time", "channels", "events", None] = None
    """MUST differ from the trunk's axis, or the branch is redundant with it.
    `None` is correct for STATICS -- not because they lack an axis, but because attention
    produces a data-dependent weighting and statics do not vary, so the query is constant and
    it collapses to a fixed per-series combination an MLP already spans."""
    maskable: bool = True
    is_demand_history: bool = False
    """Demand history may attend over daily and weekly patches, at a different resolution from the
    trunk.
    """
    provenance_col: str | None = None
    """Column distinguishing measured from imputed values so imputation is not mistaken for price
    stability.
    """


# ======================================================================================
# 4. THE DATA CONTRACT
# ======================================================================================
@dataclass
class ForecastDataSpec:
    time_idx: str
    target: str
    group_ids: Sequence[str]
    """Series identity. Embeddings are FACTORISED -- one table per id, composed."""

    target_estimand: Literal["sales_given_availability", "unconstrained_demand", "observed_count"] = "sales_given_availability"  # noqa: E501
    """WHAT the target column means, which was previously left as an undefined string.

    · 'sales_given_availability' -- what was actually sold. Directly observed. This is what the
      likelihood fits when censoring is off, and it is the honest default.
    · 'unconstrained_demand'     -- what would have sold had stock been infinite. NOT observed;
      recovered only through the censoring contract, and only as well as that contract is
      specified.

    These are DIFFERENT QUANTITIES and a comparison across them is meaningless. Replenishment
    wants the second; finance reporting usually wants the first. Naming it stops two
    implementations optimising different things while both claiming conformance."""

    target_allows_negative: bool = False
    """Returns and corrections produce negative units. A negative value entering a Negative
    Binomial log-likelihood evaluates log-gamma of a negative argument and produces NaN, which
    then propagates through every shared parameter in the batch. Either floor the target at zero
    or route returns to their own series -- but decide it here rather than discovering it in a
    training crash."""
    freq: str = "D"
    max_encoder_length: int = 364
    max_prediction_length: int = 28

    target_family: str = field(default="count", kw_only=True)
    """Executable revision family; unsupported continuous/signed heads refuse before fitting.
    ``test_resolved_revision_build`` checks this boundary. Existing retail routes are unchanged.
    """

    observation: ObservationSpec = field(default_factory=ObservationSpec)
    exposure: ExposureSpec = field(default_factory=ExposureSpec)

    lead_time_plus_review: int | None = None
    """The client's maximum LEAD TIME + REVIEW PERIOD, in days. The forecast exists so that
    replenishment can integrate demand over this span; if the horizon is shorter than it, the
    sample paths cannot answer the only question they were built for. `max_prediction_length`
    defaults to 28 and nothing previously checked it against operational reality."""

    price_col: str | None = None
    """WHICH column is unit price. Required by a non-``none`` price-response mode: a response
    cannot be applied to a column the model cannot identify. Naming it in
    `time_varying_known_reals` is not enough -- the contract applies to one specific channel,
    not to the group."""

    promo_col: str | None = None
    """WHICH column is promotion state, for the same reason and for intervention bundling."""

    oos_col: str | None = None
    """Out-of-stock input feature. The likelihood mask does not tell the encoder that a shelf was
    empty.
    """

    factor_group_col: str | None = None
    """WHICH grouping the shared latent factor is indexed by. `hierarchy` gives the levels but
    not which one carries the factor, and an engineer cannot infer it: chain, region and banner
    are all defensible and give different models."""

    static_categoricals: Sequence[str] = ()
    static_reals: Sequence[str] = ()
    time_varying_known_reals: Sequence[str] = ()
    time_varying_known_categoricals: Sequence[str] = ()
    time_varying_unknown_reals: Sequence[str] = ()
    time_varying_unknown_categoricals: Sequence[str] = ()

    text_features: dict[str, str] = field(default_factory=dict)
    embedding_residual_dim: int = 16
    """Width of the per-product residual embedding, the only identity component subject to weight
    decay. Frozen text embeddings and structured attributes supply information for unseen
    products; the residual captures information learned from their own history.
    """

    allow_joint_embedding: bool = False
    """Allow a joint embedding over the cross-product of series keys.

    Disabled by default: joint tables grow with the key cross-product and cannot
    represent unseen combinations. Factorized embeddings combine known identities."""

    # --- hierarchy: now load-bearing, not just for reporting -------------------------
    hierarchy: Sequence[str] = ()
    """Coarsening chain, for example ["store_id", "chain_id"]. It identifies stores sharing a
    latent factor. Censored store observations cannot supply an exact top-down total.
    """

    aggregate_level: str | None = None
    """Reporting grain. Aggregate forecasts sum bottom-level sample paths to preserve coherence."""

    # --- anchored events --------------------------------------------------------------
    events: str | None = None
    event_windows: dict[str, tuple[int, int]] = field(default_factory=dict)
    """{event: (lower, upper)} in signed days -- now a CEILING on support, not the effect shape.
    The shape within it is learned; see ForecastModelSpec.event_kernel."""

    event_known_at_col: str | None = None
    """When each event became KNOWN. A public holiday is known years ahead; an unannounced
    promotion cannot lift demand before shoppers know it exists. Without this the model will
    happily learn 'anticipation' of a promotion nobody had heard of -- which is a leak, and at
    forecast time there is nothing to anticipate with."""

    events_preannounced: bool = field(default=False, kw_only=True)
    """Explicit fixed-calendar availability assumption, not timestamp provenance.
    Used by the development revision calendar contract; ``test_revision_origin_integrity``
    checks timestamped calendars separately. Benchmark assumptions are recorded as such.
    """

    event_episode_col: str | None = None
    """Campaign / occurrence id. Validation MUST split by whole episode, not by random days:
    the effective sample size is the number of EPISODES, not the number of rows. Two Easters are
    two replications no matter how many stores observed them."""

    peer_graph: str | None = None
    """Table of directed product-to-product edges for `cross_product`. Required before the peer
    branch can be enabled -- `validate()` refuses the combination otherwise, because a graph-less
    peer branch returns a STRUCTURAL zero that reads exactly like a measured zero.

    Edges must come from degree-corrected co-purchase (complements) UNION taxonomy and attribute
    similarity (substitutes). Raw top-k co-occurrence is a popularity list: bread co-occurs with
    everything, and true substitutes almost never share a basket at all."""

    branches: dict[str, Branch] | None = None

    def validate(self) -> list[str]:
        errs: list[str] = []
        obs, exp = self.observation, self.exposure
        if obs.status_col and obs.status_prob_cols:
            errs.append("observation: status_col OR status_prob_cols, not both")
        if not exp.ranged_col:
            errs.append("NOTE: exposure.ranged_col is unset. A cell that was never stocked is not "
                        "a demand zero, and without this flag the occurrence head learns whether a "
                        "product is LISTED here rather than whether it SELLS. Measure the affected "
                        "share on your own data. "
                        "This is usually the single largest capability loss available, but it is a data "
                        "limitation rather than a misconfiguration, so the model still "
                        "builds and capability_report() names what was lost.")
        if not exp.store_open_col:
            errs.append("NOTE: no store_open_col -- a closed store will be read as zero demand")
        if len(self.group_ids) > 1 and self.allow_joint_embedding:
            errs.append("allow_joint_embedding with several group_ids implies one row per "
                        "cross-product; unseen combinations get no embedding")
        if self.branches:
            for name, b in self.branches.items():
                if b.attend_over == "time" and not b.is_demand_history:
                    errs.append(f"branch {name!r} attends over 'time', the trunk's own axis, and "
                                "is not the demand-history branch -- two layers would be "
                                "learning the same thing on the same axis")
                if b.maskable and not b.provenance_col:
                    errs.append(f"NOTE: branch {name!r} is maskable but declares no provenance_col; "
                                "imputed values will be indistinguishable from measured ones")
        if self.event_windows and not self.events:
            errs.append("event_windows given without an events calendar")
        if type(self.events_preannounced) is not bool:
            errs.append("events_preannounced must be boolean")
        return errs


# ======================================================================================
# 5. THE MODEL CONTRACT
# ======================================================================================
HURDLE_ZERO_SHARE_MIN = 0.10
"""Minimum measured eligible zero share for selecting a hurdle head.

Below this provisional threshold, the resolver selects a plain negative-binomial
head. Evaluate the threshold on your own training/validation data."""

PriceResponseMode = Literal["none", "soft_penalty_v1", "monotone_raw_v1"]
"""Versioned executable own-price route; kept named so its field docstring stays auditable."""


@dataclass
class ForecastModelSpec:
    d_model: int = 64
    heads: int = 4
    branch_layers: int = 1
    trunk_layers: int = 2
    decoder_layers: int = 2
    patch_size: int = 7

    # --- input scaling -----------------------------------------------------------------
    input_normalisation: Literal["revin", "per_series_z", None] = "revin"
    """Per-series reversible normalization on the demand branch.

    Without scaling, large series can dominate gradient contributions regardless
    of their relative fit quality. Output offsets do not normalize model inputs."""

    # --- topology: total + allocation ---------------------------------------------------
    topology: Literal["bottom_up_factor", "top_down", "bottom_up"] = "bottom_up_factor"
    """Forecast at STORE level, but with a SHARED LATENT FACTOR that couples stores.

    Why not plain bottom-up: independent per-store marginals cannot represent cross-store
    dependence. A promotion or a heatwave moves many stores together, so summing independent
    marginals gets the aggregate UNCERTAINTY wrong even when the mean is right -- and uncertainty
    is what sets safety stock.

    Why not top-down either: the chain total is NOT OBSERVED whenever any constituent store is
    censored or un-ranged. Summing recorded store sales yields a LOWER BOUND, not a target. So
    top-down would train the trunk on a systematically depressed aggregate -- making censored
    data less statistically valid, not more. Top-down also creates a baseline grain mismatch
    (chain-level selling frequency is ~1, erasing the slow-mover benefit), erases opposing
    store-level effects during aggregation, and a Dirichlet-multinomial imposes COMPETITIVE
    covariance -- stores trading share against each other -- so it cannot express groups of
    stores moving together.

    The factor model keeps store-level exposure, censoring, covariates and baselines intact, and
    obtains dependence from a low-rank shared factor instead. Aggregates remain coherent by
    summation. Precedent: low-rank Gaussian copula processes / DeepVAR (Ch18 of the in-house
    curriculum)."""

    factor_rank: int = 8
    """Rank of the shared latent factor per (chain, product, day). Each store loads on it, so
    stores move together to the extent their loadings agree. Rank 0 reduces to plain bottom-up
    with independent marginals."""

    factor_dynamics: Literal["ar1", "iid", None] = "ar1"
    """Give the factor autoregressive dynamics across the horizon. This is what supplies TEMPORAL
    dependence: without it, drawing independently per day and calling the result a 'sample path'
    does not create the dependence that a lead-time convolution requires."""

    # --- output head ---------------------------------------------------------------------
    head: Literal["hurdle_negbin", "negbin", "quantile", "point"] = "hurdle_negbin"
    positive_count_truncated: bool = True
    """The hurdle's positive-count component must be ZERO-TRUNCATED.

    An untruncated Negative Binomial places mass at zero. Paired with an occurrence head that has
    already decided whether the day is zero, both components then model P(0), the likelihood
    double-counts it, and the occurrence head is biased to compensate.

    Zero truncation is therefore part of the likelihood contract."""
    dispersion: Literal["pooled", "per_series", "global"] = "pooled"

    baseline_offset: bool = True
    baseline_smoothing: Literal["beta_prior", "clamp", None] = "beta_prior"
    """How the lagged FREQUENCY baseline is stabilised before it enters the occurrence head.

    'clamp' was the earlier default and is wrong: for a slow mover the lagged frequency is often
    exactly 0, and logit of a clamped 1e-4 is -9.2 -- the occurrence head would need +6 of its own
    signal just to say "5% chance", and it never recovers.

    'beta_prior' shrinks toward a category rate instead, so a zero-frequency window produces a
    small positive prior rather than a cliff."""

    size_smoothing: Literal["category_prior", None] = "category_prior"
    """The FREQUENCY baseline is Beta-shrunk. The SIZE baseline -- the lagged mean of non-zero
    days -- had no equivalent, and for a product that has never sold it is 0/0. It then enters
    a log link and takes the gradient with it.

    Shrink toward the category's mean positive size, which for a new product is the only
    evidence there is. This is the cold-start path, so it is not an edge case: every launch
    goes through it."""
    """How the lagged baseline is stabilised.

    `clamp` is WRONG and was the previous default. For a slow mover the lagged selling frequency
    is often exactly 0; clamping to eps gives logit(1e-4) = -9.2, so the network must produce
    about +6 of activation merely to predict a 5% chance of a sale. It will not, and the
    occurrence head collapses to the naive historical rate.

    `beta_prior` shrinks toward a category-level prior instead:
        b_freq = (selling_days + alpha) / (n + alpha + beta)
    Never zero, never pathological, and it borrows strength exactly where data is thinnest."""

    # --- events ----------------------------------------------------------------------------
    event_pooling: Literal["partial", "none"] = "partial"
    """Per-event embedding SHARED across SKUs, plus a weight-decayed per-(SKU, event) deviation.

    Required because the usable history contains roughly TWO Easters and TWO Christmases. Asking
    attention to learn pre/post dynamics from N=2 per SKU either overfits those two years or
    smooths the peaks away. Partial pooling is a hierarchical prior that solves N=2 by borrowing
    strength across the catalogue. Attention over events is a refinement ON TOP, not a
    replacement."""

    # --- event response shape ------------------------------------------------------------------
    event_kernel: Literal["distributed_lag", "fixed_window", None] = "fixed_window"
    """How an event's influence varies with signed distance.

    NOT raw attention over a +/-30 day window. Softmax attention is relative, so a distant
    irrelevant event still receives weight; attention weights are not effect sizes and rarely
    reach exactly zero; and 61 free daily coefficients per event is untenable with two annual
    occurrences.

    Instead a hierarchically regularised distributed-lag kernel:
        k(tau) = g(tau) * [ B(tau)' theta_event  +  B(tau)' u_(sku,event) ]
    with B a small pre/post B-spline basis, theta a shared event-type shape, u a strongly shrunk
    per-SKU deviation, and g ORDERED GATES so more distant lags are progressively likelier to be
    exactly zero. Separate gates before and after give asymmetric learned N- and N+."""

    event_max_support: tuple[int, int] = (-30, 30)
    """User-set ceiling on lag support. The model chooses an EFFECTIVE window inside this; it
    cannot discover an effect at day 45 when the ceiling is 30."""

    event_interactions: Literal["low_rank_heredity", None] = None
    """Two overlapping events do not simply add. Additive main kernels plus sparse low-rank pair
    interactions, with HEREDITY (an interaction is shrunk hard unless both main effects exist)
    and ANOVA-CENTRING (the interaction is zero when either event is absent, so it cannot quietly
    absorb a main effect).

    Identifiability, bluntly: if two events almost always co-occur, only their BUNDLE response is
    identified. Collapse them into a compound event token, or report the bundle with wide
    uncertainty. Any per-event split -- Shapley included -- is then a reporting convention, not
    an empirical result."""

    # --- promotion pull-forward -----------------------------------------------------------------
    promo_decomposition: Literal["shift_plus_net", None] = None
    """A promotion partly MOVES demand and partly CREATES it. On the expected-UNIT scale:

        delta_mu = A_shift * [ p_plus - rho * p_minus ]  +  A_net * q

    p_plus / p_minus are normalised spike and payback shapes, rho in [0,1] is the fraction paid
    back inside the window, A_net is genuinely incremental demand.

    Do NOT impose hard zero-sum conservation: promotions do create category demand, customers
    switch stores and channels, and the dip may fall outside the horizon. And apply any penalty
    on the expected COUNT -- zero-sum in log or logit space does not imply zero-sum in units.

    Honest limit: from aggregate sales alone A_shift, rho and A_net are only WEAKLY identified.
    Separating purchase acceleration from consumption expansion needs household panels,
    inventory, customer ids or control stores."""

    # --- cross-product transmission ---------------------------------------------------------------
    cross_product: Literal["signed_graph", None] = None
    """Optional signed graph of known-future commercial inputs. Separate positive and negative
    channels make complement and substitution effects explicit. Candidate edges combine residual
    co-purchase and attribute similarity. Neighbor sales at the forecast date are unavailable
    and must never enter the graph.
    """

    cross_product_factor: bool = False
    """A SECOND latent factor with PRODUCT loadings.

    Correction worth recording: the rank-8 factor is indexed by (chain, product, day) with STORES
    loading on it, so it gives cross-STORE dependence for one product -- NOT cross-product
    dependence. Residual common causes across products (weather, footfall, chain shocks) need
    their own factor, drawn once per sample path so cross-product forecasts stay coherent.

    It captures covariance, not causation, and must not be read as spillover."""

    # --- static conditioning ------------------------------------------------------------------
    static_conditioning: Sequence[Literal["gates", "trunk_film", "decoder", "dispersion"]] = (
        "gates", "trunk_film", "dispersion")
    static_variable_selection: bool = True

    # --- residual / normalisation structure ----------------------------------------------------
    zero_init_output: bool = True
    norm_before_gate: bool = True
    trunk_norm: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    demand_branch_dropout: float = 0.15
    gated_fusion: bool = True
    cross_series: Literal[None, "siblings", "graph", "both"] = None

    # --- scenario contrast (formerly "attribution") ----------------------------------------------
    attribution: Literal["prospective_scenario", None] = "prospective_scenario"
    """Prospective conditional scenario contrast, computed by paired reruns on shared draws. It
    measures a modeled difference, not a realized causal effect. Retrospective causal claims
    require an identified experimental or quasi-experimental design.
    """

    attribution_origin: Literal["pre_treatment", "any"] = "pre_treatment"
    """The forecast origin the contrast is evaluated FROM, and the most subtle error in the
    previous design.

    If the model is re-run AFTER a campaign, its encoder window and lagged baselines contain
    sales the campaign itself caused. Switching the promo flag off while retaining those sales
    conditions on DESCENDANTS OF THE TREATMENT -- the contrast is contaminated at the input,
    not merely confounded, and it will understate the effect by an amount nobody can bound.

    'pre_treatment' forces the origin before the campaign starts, so nothing downstream of the
    treatment is in the conditioning set. 'any' is provided only for diagnostics and is refused
    for client-facing output."""

    intervention_bundles: Sequence[Sequence[str]] = (("price", "promo", "display", "campaign"),)
    """Inputs that MUST move together when a scenario is constructed.

    Turning promo off while leaving the discounted UnitPrice, the display flag and the campaign
    media in place describes a world that has never existed and never could. The model will
    still return a number, because it is a smooth function evaluated off-manifold -- and that
    number is extrapolation through a physically impossible input combination.

    Bundling makes "no promotion" mean list price, no display, no media."""

    attribution_drivers: Sequence[str] = ("price", "promo", "event")
    """Which driver groups get their own contrast line. Each must be an input the caller can SET
    at inference, which is why they are all known-future. Anything the model infers from history
    -- the latent factor, the lagged baseline -- cannot be intervened on and belongs to the
    residual. This is why `baseline` is deliberately absent from the default; labelling it as a
    driver while no intervention can set it advertised a contrast the runtime could not express.
    'peer' and 'weather' are also absent: peer requires a graph that is off by default, and
    weather is not in any client contract."""

    attribution_allocation: Literal["shapley", "sequential", None] = None
    """Allocation is disabled by default. Grouping drivers changes allocated values without
    changing the forecast, and exact Shapley allocation requires exponentially many evaluations.
    Standalone contrasts do not sum to the total.
    """

    attribution_scope: Literal["query", "batch"] = "query"
    """Contrasts are computed for the SERIES AND WINDOW ASKED ABOUT, not precomputed for the
    catalogue. Point estimates use the closed form
    `E[Y] = p * mu / (1 - P_NB(0))` and need no count sampling at all: `mu` is the mean of the
    UNTRUNCATED negative binomial behind a zero-truncated positive-size component. Only spreads
    and cross-day or cross-store functionals consume paths, and they REUSE the draws of the
    factual run. An implementation that re-samples per scenario injects Monte Carlo noise into a
    difference the pairing exists to remove."""

    attribution_window: Literal["event_support", "horizon", "custom"] = "event_support"
    """Over WHICH DAYS the contrast is integrated -- and this choice can invert the answer.

    A promotion produces a spike then a payback dip. Integrating over the promotion week counts
    the spike and none of the payback. 'event_support' integrates over the fitted kernel's whole
    support.

    BUT the support must FIT INSIDE the horizon. If the campaign sits late in a 28-day horizon
    and the kernel reaches +20 days, the payback falls outside the forecast entirely and the
    contrast overstates the promotion -- the exact failure this setting exists to prevent.
    `validate()` now checks this."""

    attribution_uncertainty: Literal["campaign_bootstrap", "paths_only", None] = "campaign_bootstrap"
    """WHICH uncertainty is reported, and the previous version reported the wrong one.

    'paths_only' is the spread across shared sample paths. Because the draws are deliberately
    paired, that spread is narrow BY CONSTRUCTION -- it reflects a chosen coupling of potential
    outcomes, not evidence. It excludes parameter uncertainty and excludes between-campaign
    variation, which are the dominant terms. Quoting it as "90% confidence" -- which the earlier
    documentation did -- states precision the estimate does not have.

    'campaign_bootstrap' resamples whole CAMPAIGNS. It is far wider, it will often span zero,
    and spanning zero is the single most useful thing it can tell a client."""

    attribution_require_overlap: bool = True
    """Refuse the query when there is no comparable untreated evidence.

    Counting episodes is not enough: eleven campaigns all run at Christmas on already-rising
    products give a high count and no untreated comparison at all. Require untreated
    observations at comparable demand states, and suppress the answer rather than returning a
    number nothing supports.

    This replaces `attribution_report_support` as the primary guard -- support was reassuring
    and measured the wrong thing."""

    attribution_report_support: bool = True
    """Still reported, now as secondary context rather than as evidence of validity."""

    price_response_mode: PriceResponseMode = "soft_penalty_v1"
    """How own price is allowed to reach the fitted distribution.

    ``soft_penalty_v1`` is the historical architecture: raw price and derived discount depth
    travel through unrestricted commercial/decoder routes, while a finite counterfactual hinge
    encourages (but cannot guarantee) non-increasing expected demand.

    ``monotone_raw_v1`` is the structural route. Raw horizon price and discount depth are removed
    from every unrestricted route. The only live numerical coordinate is
    ``discount_depth = 1 - price/reference`` with a strictly positive reference measured from
    history at or before the origin and then frozen. Price-free context produces non-negative,
    bounded occurrence and size slopes; each day receives only its own masked depth adjustment.
    Holding represented context, availability and the reference fixed, raising own price therefore
    cannot raise that day's fitted expected demand. This is a model-shape contract, not a causal
    elasticity estimate.

    ``none`` leaves the ordinary predictive price routes unconstrained. The mode is explicit so a
    historical soft checkpoint can never be reinterpreted as structural merely because a boolean
    retained the same value."""

    price_occurrence_slope_max: float | None = None
    """Finite positive cap on the structural occurrence-logit slope.

    Required only by ``monotone_raw_v1``. It has no default because the cap is a claim-bearing
    numerical design choice that must be registered before paid training."""

    price_size_slope_max: float | None = None
    """Finite positive cap on the structural log-size slope, governed separately from log-odds.

    Required only by ``monotone_raw_v1`` and likewise deliberately has no hidden default."""

    # --- staging ---------------------------------------------------------------------------
    first_build: bool = True
    """Restrict the initial model to established components. Adaptive event gates, interactions,
    peer graphs, cross-product factors and promotion decomposition require separate ablation
    evidence before enabling them.
    """

    # Experiment evidence is verified by the evaluation protocol, outside the model spec.

    # --- output contract -----------------------------------------------------------------------
    emit: Literal["sample_paths", "parameters", "point"] = "sample_paths"
    """Coherent integer SAMPLE PATHS, with shared draws preserving chain totals, store
    allocation, temporal dependence and cross-store dependence.

    Marginal parameters are NOT sufficient. Replenishment needs the distribution of CUMULATIVE
    demand over lead time plus review period, and quantiles do not add across days or stores
    while independent daily parameters carry no temporal dependence. You cannot convolve
    marginals into a lead-time distribution without a dependence assumption; sampling supplies
    one."""

    path_materialisation: Literal["on_demand", "reduced", "full"] = "on_demand"
    """1,000 paths x stores x products x horizon is not storable at client scale -- for a large
    catalogue it runs to trillions of values, and the previous spec said "sample paths are the
    output" without ever sizing it.

      · 'on_demand' -- persist model state and a seed; regenerate paths per query. Default.
      · 'reduced'   -- persist only the functionals replenishment consumes (cumulative L+R
                       quantiles, coverage), computed at generation time.
      · 'full'      -- materialise everything. Permitted, but `validate()` will size it first.
    """

    n_sample_paths: int = 1000
    """200 is too few for the tail: a P99 is then determined by roughly the second-largest draw.
    Service levels are commonly set at P95-P99, so the sample count must resolve them."""

    # --- training ------------------------------------------------------------------------------
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 10
    origins_per_series: int = 4
    """Training forecast origins sampled per series per epoch. More origins expose additional
    history and increase training cost. Select the value for the dataset and record it in the
    model profile.
    """

    covariate_dropout: float = 0.2
    """Probability of masking a whole covariate channel during training. This reduces dependence on
    channels that may be absent in another dataset or at inference.
    """

    def resolve(self, data: ForecastDataSpec, *,
                zero_share: float | None = None) -> ForecastModelSpec:
        """Resolve the model to the columns supported by the data contract. Missing roles disable
        their dependent features, and capability_report records the resulting limitations. A
        measured zero_share below HURDLE_ZERO_SHARE_MIN selects a plain negative-binomial head;
        None retains the configured head. Call before validate and build the resolved
        specification.
        """
        import dataclasses as _dc
        r = _dc.replace(self)

        if zero_share is not None:
            if not 0.0 <= zero_share <= 1.0:
                raise ValueError(f"zero_share must lie in [0, 1], got {zero_share!r}")
            if zero_share < HURDLE_ZERO_SHARE_MIN and r.head == "hurdle_negbin":
                r = _dc.replace(r, head="negbin")

        if not data.hierarchy or not data.factor_group_col:
            # no grouping to share a factor over -> independent marginals, and say what that costs
            r = _dc.replace(r, topology="bottom_up", factor_rank=0, factor_dynamics=None,
                            emit="parameters" if r.emit == "sample_paths" else r.emit)
        if not data.price_col:
            r = _dc.replace(r, price_response_mode="none",
                            price_occurrence_slope_max=None,
                            price_size_slope_max=None,
                            attribution_drivers=tuple(d for d in r.attribution_drivers
                                                      if d != "price"))
        if not data.promo_col:
            r = _dc.replace(r, attribution_drivers=tuple(d for d in r.attribution_drivers
                                                         if d != "promo"))
        if not data.events:
            r = _dc.replace(r, event_kernel=None, event_pooling="none",
                            attribution_drivers=tuple(d for d in r.attribution_drivers
                                                      if d != "event"))
        elif not data.event_known_at_col and data.events_preannounced is not True:
            # The event branch itself remains a separately reported/validated data limitation,
            # but an event cannot be offered as a prospective intervention when the contract
            # cannot establish that it was knowable at the origin.
            r = _dc.replace(r, attribution_drivers=tuple(
                d for d in r.attribution_drivers if d != "event"))
        if not data.peer_graph:
            r = _dc.replace(r, cross_product=None, cross_product_factor=False)
        if not (data.observation.status_col or data.observation.status_prob_cols):
            r = _dc.replace(r, head="negbin" if r.head == "hurdle_negbin" else r.head) \
                if False else r          # censoring absent does not change the head
        # Announcement dates guard EVENT anticipation; they have no bearing on whether a planned
        # price or promotion can be changed prospectively.  The former condition disabled every
        # commercial contrast on a client with no event table, even when both commercial columns
        # were present. The currently runnable scenario is a commercial bundle whose reference
        # requires each product's list price, so price remains the load-bearing input.
        if not r.attribution_drivers or not data.price_col:
            r = _dc.replace(r, attribution=None)
        return r

    def capabilities(self, data: ForecastDataSpec) -> dict[str, list[str]]:
        """What this model CAN and CANNOT do given the columns this client actually has.

        Only three inputs are mandatory, matching pytorch-forecasting's TimeSeriesDataSet:
        `time_idx`, `target`, `group_ids`. A date, a quantity, and something to say which series
        a row belongs to. EVERYTHING ELSE IS OPTIONAL and degrades a named capability rather
        than blocking the build.

        Datasets have different feature coverage: some record promotion flags with no discount
        depth, some have no calendar table,
        some have no stock feed. A contract that refuses to build without them would be
        unusable, and one that silently skips them would ship a model nobody can interpret.

        So: build what the data supports, and SAY what was skipped and what it costs."""
        have, lost = [], []

        have.append("baseline demand forecasting from history, calendar position and seasonality")

        if data.exposure and data.exposure.ranged_col:
            have.append("range eligibility -- zeros from unstocked cells excluded from the loss")
        else:
            lost.append("NO RANGE ELIGIBILITY: without it the occurrence head learns assortment "
                        "rather than demand. Usually the single most damaging omission -- how "
                        "damaging depends on your never-stocked share, which the profiler "
                        "measures on the supplied dataset.")

        if data.observation.status_col or data.observation.status_prob_cols:
            have.append("censoring -- stockout days do not teach the model that demand was zero")
        else:
            lost.append("NO CENSORING: stockouts read as zero demand, biasing forecasts DOWN. The "
                        "size depends on your measured stockout rate and the affected products")

        if data.oos_col:
            have.append("stockout as an input feature -- yesterday's empty shelf informs today")
        else:
            lost.append("no stockout feature: the encoder cannot see that the shelf was empty")

        if data.price_col:
            have.append("price response")
            if not data.promo_col:
                lost.append("price present but no promotion flag: a discount and a base-price "
                            "change are indistinguishable")
        else:
            lost.append("NO PRICE: no predictive price response or discount scenarios; "
                        "resolve() sets price_response_mode='none'")

        if data.events:
            have.append(f"calendar and event effects ({len(data.events)} declared)")
            if not data.event_known_at_col and data.events_preannounced is not True:
                lost.append("events without announcement dates: pre-event anticipation cannot be "
                            "learned honestly, so only the on-day and after effects are fitted")
            if not data.event_episode_col:
                lost.append("events without episode ids: validation cannot split by campaign, so "
                            "reported event accuracy will be optimistic")
        else:
            lost.append("NO CALENDAR OR EVENTS: holidays and promotions become unexplained "
                        "spikes absorbed into the baseline, which will lag them by construction")

        if data.hierarchy and data.factor_group_col:
            have.append("cross-store dependence via the shared latent factor")
        else:
            missing = "hierarchy" if not data.hierarchy else "factor_group_col"
            lost.append(f"no {missing}: the shared factor cannot be indexed, so store forecasts "
                        "are independent, aggregate uncertainty is understated and lead-time "
                        "sums are not calibrated. resolve() steps topology down to 'bottom_up' "
                        "and emit to 'parameters'")

        if self.cross_product and data.peer_graph:
            have.append("cross-product transmission")
        else:
            lost.append("no peer graph: cannibalisation and halo effects are structurally "
                        "invisible -- reported as zero, which is not the same as measured zero")

        settable_scenario_drivers = tuple(
            name for name, present in (
                ("price", bool(data.price_col)),
                ("promo", bool(data.promo_col)),
            ) if present and name in self.attribution_drivers)
        if self.attribution and data.price_col and settable_scenario_drivers:
            have.append("prospective commercial scenario contrasts over "
                        + ", ".join(settable_scenario_drivers))
        else:
            lost.append("no runnable commercial scenario contrast: the bundled reference "
                        "requires a declared planned/list-price column")

        return {"available": have, "skipped": lost}

    def capability_report(self, data: ForecastDataSpec) -> str:
        c = self.capabilities(data)
        L = ["MODEL CAPABILITY REPORT", "=" * 60,
             "", f"BUILDS: yes -- {len(c['available'])} capabilities available", ""]
        L += [f"  + {x}" for x in c["available"]]
        L += ["", f"SKIPPED -- {len(c['skipped'])} capabilities unavailable, and what it costs:", ""]
        L += [f"  - {x}" for x in c["skipped"]]
        return "\n".join(L)

    def validate(self, data: ForecastDataSpec) -> list[Finding]:
        """Returns typed Findings. `str(f)` renders as before, and `f.startswith("NOTE:")` still
        works, so existing callers are unaffected; new callers filter on `.severity`."""
        raw: list[str] = []
        errs = raw
        if self.d_model % self.heads:
            errs.append("d_model must be divisible by heads")
        if self.topology == "top_down":
            errs.append("topology='top_down': the chain total is unobserved whenever any store is "
                        "censored, so summing recorded sales gives a lower bound, not a target")
        if self.topology == "bottom_up":
            errs.append("NOTE: topology='bottom_up' with factor_rank=0 -- independent marginals "
                        "carry no cross-store dependence, so AGGREGATE uncertainty is understated "
                        "and lead-time sums are not calibrated. This is the correct degradation "
                        "when a client supplies no hierarchy, and emit steps down to 'parameters' "
                        "because coherent sample paths are no longer meaningful. It is a NOTE "
                        "rather than an error precisely because it is a data limitation, not a "
                        "misconfiguration.")
        if self.topology == "bottom_up_factor" and self.factor_rank < 1:
            errs.append("bottom_up_factor requires factor_rank >= 1")
        if self.topology == "bottom_up_factor" and not data.hierarchy:
            errs.append("bottom_up_factor requires a hierarchy to define which stores share a factor")
        if self.emit == "sample_paths" and self.factor_dynamics is None:
            errs.append("emit='sample_paths' with factor_dynamics=None: independent daily draws "
                        "carry no temporal dependence, so lead-time sums will be mis-calibrated")
        if self.emit == "sample_paths" and self.n_sample_paths < 500:
            errs.append(f"n_sample_paths={self.n_sample_paths} is too few to resolve a P99 tail")
        if self.baseline_offset and self.baseline_smoothing == "clamp":
            errs.append("baseline_smoothing='clamp' produces logit(eps) ~ -9.2 for a slow mover "
                        "and the occurrence head will not recover; use 'beta_prior'")
        if self.input_normalisation is None:
            errs.append("input_normalisation=None: where series differ in scale by orders of "
                        "magnitude, per-series losses do too, so gradient contributions are dominated "
                        "by fast movers regardless of fit quality.")
        if self.event_pooling == "none" and data.events:
            errs.append("event_pooling='none' with only ~2 observations per SKU per annual event")
        if data.events and not data.event_known_at_col and data.events_preannounced is not True:
            errs.append("events declared without event_known_at_col: the model can learn to "
                        "anticipate a promotion nobody had been told about")
        if data.events and not data.event_episode_col:
            errs.append("NOTE: no event_episode_col -- validation must split by whole campaign, "
                        "not by random days; effective sample size is EPISODES, not rows")
        if self.cross_product == "signed_graph" and not data.peer_graph:
            errs.append("cross_product='signed_graph' requires peer_graph")
        if self.cross_product_factor and self.factor_rank < 1:
            errs.append("cross_product_factor requires a factor mechanism")
        # --- scenario contrast ---------------------------------------------------------
        if self.attribution and self.emit == "point":
            errs.append("attribution with emit='point': a contrast quoted with no spread, and a "
                        "spread that includes zero is the most useful thing a client can hear")
        if self.attribution and self.attribution_origin != "pre_treatment":
            errs.append("attribution_origin != 'pre_treatment': re-running after the campaign "
                        "puts sales the campaign CAUSED into the encoder and the lagged "
                        "baseline, so switching the driver off conditions on descendants of the "
                        "treatment and contaminates the contrast at the input")
        if self.attribution and self.attribution_uncertainty == "paths_only":
            errs.append("attribution_uncertainty='paths_only': the draws are deliberately "
                        "paired, so that spread is narrow BY CONSTRUCTION and excludes "
                        "parameter and between-campaign variation -- it is not a confidence "
                        "interval for the uplift and must not be quoted as one")
        if self.attribution and not self.intervention_bundles:
            errs.append("attribution with no intervention_bundles: switching promo off while "
                        "retaining the discounted price, display and media describes a state "
                        "that has never existed, and the model will extrapolate through it")
        if self.attribution and "baseline" in self.attribution_drivers:
            errs.append("'baseline' is not an attribution driver: it is inferred from demand "
                        "history and no prospective scenario input can set it; keep it in the "
                        "residual rather than advertising an intervention the runtime cannot run")
        if self.attribution and not self.attribution_require_overlap:
            errs.append("attribution_require_overlap=False: episode COUNT is not evidence of "
                        "identification -- eleven campaigns all run at Christmas on rising "
                        "products give a high count and no untreated comparison at all")
        if self.attribution and self.attribution_allocation in ("shapley", "sequential"):
            errs.append(f"attribution_allocation={self.attribution_allocation!r}: an allocation "
                        "convention chosen after the value function does not create "
                        "identification; regrouping drivers changes every number in the "
                        "waterfall without changing the forecast")
        if self.attribution and self.attribution_window == "horizon" and self.event_kernel:
            errs.append("attribution_window='horizon': the contrast is truncated at the forecast "
                        "edge by choice, so a payback beyond it is never debited and every "
                        "promotion is overstated")
        if self.attribution and self.attribution_window == "event_support" and \
                self.event_kernel and data.max_prediction_length < self.event_max_support[1]:
            errs.append(f"NOTE: horizon {data.max_prediction_length}d is shorter than the event "
                        f"post-support ceiling {self.event_max_support[1]}d, so part of any "
                        "payback lies beyond the forecast. This is often unavoidable -- but the "
                        "contrast is then GROSS, not net, and must be labelled that way rather "
                        "than presented as the promotion's net effect")
        if self.attribution and "price" in self.attribution_drivers and not data.price_col:
            errs.append("price contrasts require data.price_col: a sign constraint and a "
                        "one-driver contrast cannot be applied to a column the model cannot "
                        "identify")
        if self.attribution and "peer" in self.attribution_drivers and self.cross_product is None:
            errs.append("'peer' listed as an attribution driver but cross_product=None: there "
                        "is no channel through which a neighbour's discount can move this "
                        "product, so the answer is structurally zero, not measured")
        price_modes = {"none", "soft_penalty_v1", "monotone_raw_v1"}
        if self.price_response_mode not in price_modes:
            errs.append(
                f"unsupported price_response_mode={self.price_response_mode!r}; expected one "
                f"of {sorted(price_modes)}")
        if self.price_response_mode != "none" and not data.price_col:
            errs.append(
                f"price_response_mode={self.price_response_mode!r} requires data.price_col")
        caps = (self.price_occurrence_slope_max, self.price_size_slope_max)
        if self.price_response_mode == "monotone_raw_v1":
            import math
            for name, value in zip(
                    ("price_occurrence_slope_max", "price_size_slope_max"), caps,
                    strict=True):
                if (type(value) not in (int, float) or isinstance(value, bool)
                        or not math.isfinite(float(value)) or value <= 0.0):
                    errs.append(
                        f"{name} must be a finite positive number for "
                        f"price_response_mode='monotone_raw_v1', got {value!r}")
        elif caps != (None, None):
            errs.append(
                "price slope caps are executable only for "
                "price_response_mode='monotone_raw_v1'; set both to None")
        if (self.attribution and "price" in self.attribution_drivers
                and self.price_response_mode != "none"):
            errs.append(
                "NOTE: a constrained price response makes the price contrast a ONE-SIDED "
                "model estimator -- a clearance markdown on a dying line can be reported as "
                "adding units; label price contrasts accordingly")

        # --- staging and the baseline gate ------------------------------------------------
        speculative = [n for n, v in (
            ("event_kernel='distributed_lag'", self.event_kernel == "distributed_lag"),
            ("event_interactions", self.event_interactions is not None),
            ("cross_product", self.cross_product is not None),
            ("cross_product_factor", self.cross_product_factor),
            ("promo_decomposition", self.promo_decomposition is not None)) if v]
        if self.first_build and speculative:
            errs.append(f"first_build=True with {', '.join(speculative)} enabled: these are "
                        "weakly identified on this data and unmeasured against any baseline. "
                        "Use fixed_window events and no graph for the first build, then set "
                        "first_build=False once the baseline gate is passed")
        if self.first_build:
            errs.append("NOTE: first_build=True -- speculative components are refused. Whether an "
                        "improvement may be CLAIMED is not checkable here: see the evaluation protocol, "
                        "which is where the baseline gate is enforced as a process step")

        # --- operational horizon and output volume ----------------------------------------
        if data.lead_time_plus_review is None:
            errs.append("NOTE: lead_time_plus_review unset -- nothing checks that the horizon "
                        "covers the span replenishment must integrate over")
        elif data.max_prediction_length < data.lead_time_plus_review:
            errs.append(f"horizon {data.max_prediction_length}d is shorter than lead time plus "
                        f"review period ({data.lead_time_plus_review}d): the paths cannot answer "
                        "the only question they exist for")
        if self.emit == "sample_paths" and self.path_materialisation == "full":
            errs.append("path_materialisation='full': 1,000 paths x every cell x horizon is not "
                        "storable at catalogue scale; use 'on_demand' or 'reduced'")

        # --- contract columns named mechanisms depend on ----------------------------------
        if self.topology == "bottom_up_factor" and not data.factor_group_col:
            errs.append("bottom_up_factor requires data.factor_group_col: hierarchy gives the "
                        "levels but not which one carries the factor, and chain, region and "
                        "banner are all defensible and give different models")
        if data.observation.status_col and not data.oos_col:
            errs.append("NOTE: stockouts are masked in the likelihood but absent from the "
                        "inputs -- masking removes the day from the loss, it does not tell the "
                        "encoder the shelf was empty, and an empty shelf yesterday changes "
                        "demand today")
        if self.baseline_offset and self.size_smoothing is None:
            errs.append("size_smoothing=None: the SIZE baseline is the lagged mean of non-zero "
                        "days, which for a never-sold product is 0/0 and then enters a log link "
                        "-- every product launch goes through this path")

        if self.emit == "point":
            errs.append("emit='point': an expected demand of 0.2 rounds to 0, the store orders "
                        "nothing, and then stocks out")
        if self.head in ("hurdle_negbin", "negbin") and data.observation.status_col is None:
            errs.append("NOTE: a zero-aware head with no observation contract fits P(zero) to "
                        "availability rather than to demand wherever stockouts occur")
        if not self.static_conditioning:
            errs.append("NOTE: statics would enter only as a fusion token -- less than the TFT "
                        "this replaces, which injects them at four points")
        return [Finding("note", m[6:].lstrip()) if m.startswith("NOTE:") else Finding("error", m)
                for m in raw]
