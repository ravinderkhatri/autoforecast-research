"""Reference likelihood and sampling contracts in standard-library Python. Defines the truncated
hurdle-negative-binomial parameterization, observation states, factor link, AR(1) constraint,
inference objective and ancestral sampler. Model implementations are checked against these
reference calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import betainc

__all__ = [
    "AR1",
    "CENSORED_TREATMENT",
    "Hurdle",
    "LikelihoodError",
    "ObservationState",
    "ancestral_path",
    "ancestral_path_plain",
    "factor_link",
    "iwae_bound",
    "log_likelihood",
    "nb_log_pmf",
    "nb_log_sf",
    "plain_log_likelihood",
    "plain_nb_log_sf",
    "soft_informative_weight",
    "soft_log_likelihood",
    "ztnb_log_pmf",
]

ObservationState = str  # "exact" | "lower_bound" | "unobserved" -- see spec.forecast_spec

CENSORED_TREATMENT = "right_censored"
"""A positive lower_bound contributes log(p) plus the conditional survival probability of demand at
or above the observed quantity. A zero lower_bound provides no information about occurrence or
magnitude and contributes zero.
"""


def _is_finite_real(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


class LikelihoodError(ValueError):
    """A parameter or observation was passed that the contract does not define a value for."""


# ---------------------------------------------------------------- 1. the parameterisation

def nb_log_pmf(y: int, mu: float, k: float) -> float:
    """log P(Y = y) for a Negative Binomial with MEAN `mu` and DISPERSION `k`.

        P(y) = Gamma(y+k) / (Gamma(k) y!) * (k/(k+mu))^k * (mu/(k+mu))^y

    `k` is the shape: LARGER k means LESS overdispersion, and k -> inf gives the Poisson. This is
    the mean-dispersion parameterisation, not the (r, p) one -- they are interconvertible, and
    mixing them is the commonest way two NB implementations disagree.

    Args:
        y: Observed count, >= 0.
        mu: Mean, > 0.
        k: Dispersion, > 0.

    Raises:
        LikelihoodError: y negative or non-integral, or mu/k not strictly positive.
    """
    _check_count(y)
    _check_positive(mu, "mu")
    _check_positive(k, "k")
    log_p0 = -_k_log1p_mu_over_k(mu, k)
    if y == 0:
        # Besides being cheaper, this exact branch avoids both failure modes of the generic
        # formula at the edge of binary64: `lgamma(k)-lgamma(k)` can be inf-inf for huge k and
        # `0*log1p(k/mu)` is NaN when k/mu overflows.
        return log_p0

    # For very large k relative to an integer count, subtracting two lgammas loses the entire
    # rising-factorial increment (and eventually becomes inf-inf). Expand
    #
    #   log Gamma(k+y)-log Gamma(k) = y log(k) + sum_i log1p(i/k)
    #
    # and combine `y log(k)` with the NB probability ratio before either large term is formed.
    # Four terms of log1p's alternating series are beyond binary64 accuracy on this branch.
    ratio = y / k
    if k >= 1_000_000.0 and ratio <= 1e-4:
        correction = _large_k_rising_correction(float(y), k)
        return (correction - math.lgamma(y + 1) + y * math.log(mu) + log_p0
                - y * _log1p_ratio(mu, k))

    # In the ordinary regime lgamma is accurate and faster than a count-sized recurrence. The
    # log-ratio helper cannot overflow even when k/mu cannot be represented.
    try:
        combination = math.lgamma(y + k) - math.lgamma(k) - math.lgamma(y + 1)
    except (OverflowError, ValueError):
        combination = math.nan
    if not math.isfinite(combination):
        raise LikelihoodError(
            f"cannot evaluate the finite NB gamma ratio for y={y}, mu={mu}, k={k}; the count "
            "is outside the numerically supported exact-count domain")
    return combination + log_p0 - y * _log1p_ratio(k, mu)


def _log1p_ratio(numerator: float, denominator: float) -> float:
    """`log1p(numerator / denominator)` without overflowing the ratio."""
    if numerator <= denominator:
        return math.log1p(numerator / denominator)
    delta = math.log(numerator) - math.log(denominator)
    return delta + math.log1p(math.exp(-delta))


def _k_log1p_mu_over_k(mu: float, k: float) -> float:
    """`k * log1p(mu/k)` while retaining a ratio smaller than binary64.

    When `mu/k` underflows, the product does not: its limit is `mu`. Computing the ratio first
    therefore changes a finite log probability into zero. Scaling by `mu` on that branch keeps
    the lost exponent in range.
    """
    if mu <= k:
        ratio = mu / k
        return mu if ratio == 0.0 else mu * (math.log1p(ratio) / ratio)
    delta = math.log(mu) - math.log(k)
    return k * (delta + math.log1p(math.exp(-delta)))


def _large_k_rising_correction(y: float, k: float) -> float:
    """First four terms of `sum_{i=0}^{y-1} log1p(i/k)` for `y/k <= 1e-4`."""
    inverse_k = 1.0 / k
    ratio = y / k
    previous = (y - 1.0) / k
    twice_previous = (2.0 * y - 1.0) / k
    first = y * previous / 2.0
    second = y * previous * twice_previous / 12.0
    third = y * ratio * previous * previous / 12.0
    fourth_power_sum = 3.0 * ratio * ratio - 3.0 * ratio * inverse_k - inverse_k * inverse_k
    fourth = y * previous * twice_previous * fourth_power_sum / 120.0
    return first - second + third - fourth


def ztnb_log_pmf(y: int, mu: float, k: float) -> float:
    """log P(Y = y | Y >= 1) -- the ZERO-TRUNCATED Negative Binomial. THE TRUNCATION NORMALISATION.

        P(y | y>=1) = P_NB(y) / (1 - P_NB(0)),   y >= 1
        P_NB(0)     = (k/(k+mu))^k

    Dividing by `1 - P_NB(0)` is not a detail. Without it the occurrence head has already decided
    whether the day is zero and the size component ALSO places mass at zero, so the likelihood
    double-counts P(0) and the occurrence head is biased to compensate -- a model that trains,
    converges, and is quietly wrong. `ForecastModelSpec.positive_count_truncated` enforces this.

    Raises:
        LikelihoodError: `y < 1`, or invalid NB parameters.
    """
    _check_count(y)
    if y < 1:
        raise LikelihoodError(
            f"the zero-truncated component is defined only for y >= 1, got {y}. A zero is the "
            "OCCURRENCE component's event and must not reach here")
    log_p0 = nb_log_pmf(0, mu, k)
    if log_p0 >= 0.0:
        raise LikelihoodError(
            f"P_NB(0) is not below 1 for mu={mu}, k={k}, so the truncation constant is not "
            "positive")
    normalizer = -math.expm1(log_p0)
    a = -log_p0
    if a <= 0.01:
        # Directly subtracting log(normalizer) from log P(y) cancels two values of order
        # `-log(mu)` when mu is tiny. The value may round tolerably, but its derivative becomes
        # inf-inf in the tensor implementation. This algebra cancels that scale analytically.
        centered = _centered_combinatorial(y, k)
        bridge = _log_r_over_log1p_r(mu, k) + _log_a_over_one_minus_exp_minus_a(a)
        return (centered + (y - 1) * math.log(mu) - y * _log1p_ratio(mu, k)
                - a + bridge)
    return nb_log_pmf(y, mu, k) - math.log(normalizer)


def _centered_combinatorial(y: int, k: float) -> float:
    """`log((k)_y/y!) - y log(k)`, stable in the large-dispersion limit."""
    if y == 1:
        return 0.0
    ratio = y / k
    if k >= 1_000_000.0 and ratio <= 1e-4:
        return _large_k_rising_correction(float(y), k) - math.lgamma(y + 1)
    return (math.lgamma(y + k) - math.lgamma(k) - math.lgamma(y + 1)
            - y * math.log(k))


def _log_r_over_log1p_r(mu: float, k: float) -> float:
    """`log((mu/k) / log1p(mu/k))`, including its removable limit at zero."""
    delta = math.log(mu) - math.log(k)
    if delta <= math.log(0.01):
        r = math.exp(delta)
        return (r / 2.0 - 5.0 * r**2 / 24.0 + r**3 / 8.0
                - 251.0 * r**4 / 2880.0 + 19.0 * r**5 / 288.0
                - 19087.0 * r**6 / 362880.0)
    log1p_r = delta + math.log1p(math.exp(-delta)) if delta > 0.0 else math.log1p(math.exp(delta))
    return delta - math.log(log1p_r)


def _log_a_over_one_minus_exp_minus_a(a: float) -> float:
    """`log(a / (1-exp(-a)))`, with its cancellation-free expansion near zero."""
    if a <= 0.01:
        return a / 2.0 - a * a / 24.0 + a**4 / 2880.0 - a**6 / 181440.0
    return math.log(a) - math.log(-math.expm1(-a))


def nb_log_sf(c: int, mu: float, k: float) -> float:
    """log P(Y >= c | Y >= 1) for the zero-truncated NB -- the CENSORED contribution.

    Uses the CLOSED FORM, not a sum:

        P(Y >= c) = I_x(c, k),    x = mu / (mu + k)          (regularised incomplete beta)
        P(Y >= c | Y >= 1) = P(Y >= c) / (1 - P_NB(0))

    An earlier version computed `1 - sum_{y=1}^{c-1} P_ZTNB(y)`, and the parity test against the
    torch head caught what is wrong with that: the survivor is the difference of two numbers that
    agree to fifteen digits, so it loses precision exactly where the survivor is SMALL -- which is
    the tail, which is large stockouts, which is where the censoring contract matters most. At
    c=24, mu=2, k=3 the two forms already disagree in the ninth significant figure, and the gap
    widens from there until the subtraction underflows to zero entirely.

    `scipy.special.betainc` is the standard implementation and is a dependency rather than a
    reimplementation: the regularised incomplete beta is a scientific algorithm, and rewriting one
    to avoid an import is exactly what the reuse ladder forbids.

    Args:
        c: The observed floor, >= 1. `P(Y >= 1 | Y >= 1)` is 1, so `c = 1` returns 0.0.
        mu: Mean of the untruncated NB.
        k: Dispersion.

    Raises:
        LikelihoodError: `c < 1`, or the tail cannot be certified inside the explicit finite
            support used by the fallback calculation.
    """
    _check_count(c)
    if c < 1:
        raise LikelihoodError(f"a censored floor must be >= 1, got {c}")
    _check_positive(mu, "mu")
    _check_positive(k, "k")
    if c == 1:
        return 0.0
    survivor = float(betainc(float(c), k, _mu_over_sum(mu, k)))
    log_p0 = nb_log_pmf(0, mu, k)
    positive_mass = -math.expm1(log_p0)
    # A nonzero subnormal is not an accuracy certificate. Below the smallest NORMAL binary64,
    # adjacent representable probabilities are separated by an increasingly large relative gap.
    # `betainc` can therefore return a plausible nonzero tail whose log is wrong by 1e-7 even
    # though the certified log-space sum below remains accurate to about 1e-12. Use the closed
    # form only while its probability retains normal relative precision.
    if float.fromhex("0x1.0p-1022") <= survivor <= positive_mass:
        return math.log(survivor) - math.log(positive_mass)
    # At an extreme ratio, binary64 can round x=mu/(mu+k) to exactly one. `betainc` then returns
    # an untruncated survivor of one even when the total positive mass is 1e-305, yielding an
    # impossible POSITIVE conditional log probability. Treat that like underflow and require the
    # proved log-tail calculation below; it will either compute the value or explicitly refuse an
    # unrepresentably broad support.
    # betainc returns a probability, so it underflows below ~1e-308 even though the LOG of that
    # probability is perfectly representable. Falling back to a log-space tail sum rather than
    # raising, because the torch head computes exactly this and the two must agree: a contract
    # that refuses what the implementation can compute is not a contract, it is a narrower one.
    return _log_sf_tail(c, mu, k) - math.log(positive_mass)


def _log_sf_tail(c: int, mu: float, k: float, eps: float = 1e-15,
                 max_terms: int = 10_000) -> float:
    """`log P(Y >= c)` with a proved geometric bound on the omitted remainder.

    For `q = mu/(mu+k)`, `P(y+1)/P(y) = q(y+k)/(y+1)`. The ratio decreases after the
    mode for `k > 1`; for `k <= 1` it is bounded by `q`. Once the tail is decreasing, its
    remainder after the included terms is therefore no more than `P(b)/(1-rho)`, where `b` is
    the first omitted count. Terms are added until that bound is at most `eps` times the included
    mass. This mode-aware bound matters near Poisson: `q` alone can be tiny even while adjacent
    probabilities around the mode are approximately one.
    """
    if not _is_finite_real(eps) or not 0.0 < eps < 1.0:
        raise LikelihoodError(f"tail relative error must be finite and in (0,1), got {eps!r}")
    q = _mu_over_sum(mu, k)
    if q >= 1.0:
        raise LikelihoodError(
            f"the upper tail for c={c}, mu={mu}, k={k} is broader than binary64 can resolve "
            f"within the {max_terms}-term support contract")
    log_sum = -math.inf
    for terms in range(1, max_terms + 1):
        y = c + terms - 1
        log_term = nb_log_pmf(y, mu, k)
        if log_sum == -math.inf:
            log_sum = log_term
        else:
            top = max(log_sum, log_term)
            log_sum = top + math.log(math.exp(log_sum - top) + math.exp(log_term - top))

        first_omitted = y + 1
        rho = q * (first_omitted + k) / (first_omitted + 1) if k > 1.0 else q
        if rho < 1.0:
            log_bound = nb_log_pmf(first_omitted, mu, k) - math.log1p(-rho)
            if log_bound <= log_sum + math.log(eps):
                return log_sum
    raise LikelihoodError(
        f"the upper tail cannot reach relative error {eps:g} within {max_terms} terms for "
        f"c={c}, mu={mu}, k={k}; truncating it would understate the censored survivor")


def _mu_over_sum(mu: float, k: float) -> float:
    """`mu / (mu+k)` without overflowing a finite sum."""
    if mu <= k:
        ratio = mu / k
        return ratio / (1.0 + ratio)
    ratio = k / mu
    return 1.0 / (1.0 + ratio)


@dataclass(frozen=True)
class Hurdle:
    """One cell's predicted distribution: occurrence probability, size mean, dispersion.

    Attributes:
        p: P(Y > 0). The occurrence head's output, after the Beta-shrunk baseline offset.
        mu: Mean of the UNTRUNCATED NB behind the size component. Not E[Y], and not E[Y|Y>0].
        k: Dispersion, pooled from static attributes with a shrunk residual by default.
    """

    p: float
    mu: float
    k: float

    def __post_init__(self) -> None:
        """Raises: LikelihoodError: p outside (0,1) exclusive, or mu/k not positive."""
        if not _is_finite_real(self.p) or not 0.0 < self.p < 1.0:
            raise LikelihoodError(
                f"occurrence probability must be in (0,1) exclusive, got {self.p}. A hard 0 or 1 "
                "makes the log-likelihood infinite for the other outcome and kills the gradient")
        _check_positive(self.mu, "mu")
        _check_positive(self.k, "k")

    @property
    def mean(self) -> float:
        """E[Y] = p * mu / (1 - P_NB(0)).

        **Not `p * mu`.** The size component is zero-truncated, so its mean is `mu / (1-P_NB(0))`,
        which is strictly greater than `mu`. Using `p * mu` under-forecasts every cell, by more
        the more intermittent the series -- exactly where this project's data lives.
        """
        log_p0 = nb_log_pmf(0, self.mu, self.k)
        a = -log_p0
        log_conditional_mean = (_log_r_over_log1p_r(self.mu, self.k)
                                + _log_a_over_one_minus_exp_minus_a(a))
        log_mean = math.log(self.p) + log_conditional_mean
        if log_mean > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            raise LikelihoodError(
                f"the hurdle mean for p={self.p}, mu={self.mu}, k={self.k} exceeds the largest "
                "finite binary64 value")
        return math.exp(log_mean)


# ---------------------------------------------------------------- 2. per-state contributions

def log_likelihood(state: ObservationState, y: int, cell: Hurdle) -> float:
    """The cell's contribution to the log-likelihood, by observation state.

    | state | observed | contribution |
    |---|---|---|
    | `exact` | 0 | `log(1 - p)` |
    | `exact` | y >= 1 | `log p + log P_ZTNB(y)` |
    | `lower_bound` | c >= 1 | `log p + log P(Y >= c | Y >= 1)` — right-censored |
    | `lower_bound` | 0 | `0.0` — vacuous; out of stock all day, sold nothing |
    | `unobserved` | any | `0.0` — never ranged, or no data |

    Args:
        state: One of `exact`, `lower_bound`, `unobserved`.
        y: The recorded quantity, >= 0.
        cell: The predicted distribution.

    Returns:
        The log-likelihood contribution. Zero means the cell is UNINFORMATIVE, never that it was
        perfectly predicted.

    Raises:
        LikelihoodError: unknown state, or a negative/non-integral quantity.
    """
    _check_count(y)
    if state == "unobserved":
        return 0.0
    if state == "lower_bound":
        # Vacuous: "demand >= 0" says nothing, and the occurrence is unknown too.
        return 0.0 if y == 0 else math.log(cell.p) + nb_log_sf(y, cell.mu, cell.k)
    if state == "exact":
        if y == 0:
            return math.log1p(-cell.p)
        return math.log(cell.p) + ztnb_log_pmf(y, cell.mu, cell.k)
    raise LikelihoodError(
        f"unknown observation state {state!r}; the contract defines exact, lower_bound and "
        "unobserved, and a state it does not define must not be given a default")


# ---------------------------------------------------------------- 3. the factor-to-head link

def factor_link(
    eta_occurrence: float,
    eta_size: float,
    factor: tuple[float, ...],
    loading_occurrence: tuple[float, ...],
    loading_size: tuple[float, ...],
    log_k: float,
) -> Hurdle:
    """Combine each head's own linear predictor with the shared latent factor.

        logit(p) = eta_occ  + <loading_occ,  f>
        log(mu)  = eta_size + <loading_size, f>
        k        = exp(log_k)                     -- the factor does NOT enter the dispersion

    **Indices, stated because they are the part an implementation gets silently wrong.** The
    factor `f` is indexed by `(factor_group, product, horizon_step)` -- `factor_group` is whatever
    `ForecastModelSpec.factor_group_col` names, which is a CHOICE (chain, region and banner all
    induce different dependence) and is why the spec refuses to infer it. The LOADINGS are indexed
    by **store**. So every store in a group shares one drawn factor path and moves with the others
    to the extent their loadings agree -- which is where cross-store dependence comes from,
    without a chain total we cannot observe.

    Two loading vectors, one factor: a store may be more sensitive in whether it sells at all than
    in how much, and forcing one loading to serve both would make the two heads move together by
    construction rather than by evidence.

    The dispersion is deliberately outside the link. `k` is pooled from static attributes; letting
    a latent daily factor move it would let the model explain a bad day by widening the
    distribution rather than by shifting its mean, which is the failure mode that makes intervals
    look calibrated on average and useless during promotions.

    Raises:
        LikelihoodError: the factor and either loading vector have different lengths.
    """
    if not (len(factor) == len(loading_occurrence) == len(loading_size)):
        raise LikelihoodError(
            f"factor rank {len(factor)} but loadings of length {len(loading_occurrence)} and "
            f"{len(loading_size)}; a mismatched loading silently drops or repeats a dimension")
    _check_finite(eta_occurrence, "occurrence linear predictor")
    _check_finite(eta_size, "size linear predictor")
    _check_finite(log_k, "log dispersion")
    for label, values in (
        ("factor", factor),
        ("occurrence loading", loading_occurrence),
        ("size loading", loading_size),
    ):
        for index, value in enumerate(values):
            _check_finite(value, f"{label}[{index}]")
    dot_occ = sum(a * b for a, b in zip(factor, loading_occurrence, strict=True))
    dot_size = sum(a * b for a, b in zip(factor, loading_size, strict=True))
    occurrence_logit = eta_occurrence + dot_occ
    size_log_mean = eta_size + dot_size
    _check_finite(occurrence_logit, "factor-linked occurrence predictor")
    _check_finite(size_log_mean, "factor-linked size predictor")
    # Stable on both signs. The direct `exp(-x)` branch raises OverflowError for a perfectly
    # finite, very negative predictor instead of returning/refusing the representable boundary.
    if occurrence_logit >= 0.0:
        exp_neg = math.exp(-occurrence_logit)
        p = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(occurrence_logit)
        p = exp_pos / (1.0 + exp_pos)
    return Hurdle(
        p=p,
        mu=_positive_exp(size_log_mean, "factor-linked size mean"),
        k=_positive_exp(log_k, "factor-linked dispersion"),
    )


# ---------------------------------------------------------------- 4. the AR(1) constraint

@dataclass(frozen=True)
class AR1:
    """The shared factor's dynamics across the horizon, with its stationarity constraint.

        f_0 ~ N(0, I)
        f_t = phi * f_{t-1} + sqrt(1 - phi^2) * eps_t,     eps_t ~ N(0, I)

    **The innovation scale is `sqrt(1 - phi^2)`, not 1.** That is what keeps the marginal variance
    at exactly 1 for every t. With unit innovations the factor's variance grows as
    `1 + phi^2 + phi^4 + ...`, so later horizon days get a wider latent term than earlier ones for
    no modelled reason -- the forecast fans out with lead time by construction rather than because
    uncertainty grows, and the loadings absorb a scale that is drifting under them.

    A stationary marginal also makes the scale identifiable: without it, `phi` and the loading
    magnitudes trade off and neither is determined.

    Attributes:
        phi: Persistence, per factor dimension, strictly inside (-1, 1).
    """

    phi: tuple[float, ...]

    def __post_init__(self) -> None:
        """Raises: LikelihoodError: any `|phi| >= 1`, which is non-stationary."""
        bad = [
            p for p in self.phi
            if not _is_finite_real(p) or abs(p) >= 1.0
        ]
        if bad:
            raise LikelihoodError(
                f"AR(1) persistence must be finite numeric values satisfying |phi| < 1; got "
                f"{bad}. At |phi| = 1 the factor is a random walk with no stationary marginal, "
                "and the horizon's later days acquire unbounded variance. Parameterise as "
                "phi = tanh(theta) to make this unreachable")

    @property
    def innovation_scale(self) -> tuple[float, ...]:
        """`sqrt(1 - phi^2)` per dimension -- the scale that holds the marginal variance at 1."""
        return tuple(math.sqrt(1.0 - p * p) for p in self.phi)


# ---------------------------------------------------------------- 5. the inference objective

def _validate_soft_state_probs(probs: dict[str, float]) -> None:
    """Require the exact, finite probability distribution shared by both soft-state APIs."""
    expected = {"exact", "lower_bound", "unobserved"}
    if set(probs) != expected:
        raise LikelihoodError(
            f"soft states must cover exactly {sorted(expected)}, got {sorted(probs)}; a missing "
            "state is a silent zero and an extra one is undefined")
    total = 0.0
    for state, probability in probs.items():
        if not _is_finite_real(probability) or probability < 0.0:
            raise LikelihoodError(
                f"P({state}) must be finite and non-negative, got {probability!r}")
        total += float(probability)
    if abs(total - 1.0) > 1e-9:
        raise LikelihoodError(
            f"state probabilities sum to {total!r}, not 1; renormalising here would hide a "
            "broken feed inside a likelihood")


def soft_log_likelihood(probs: dict[str, float], y: int, cell: Hurdle) -> float:
    """One cell's contribution when the observation STATE itself is uncertain.

    `ObservationSpec.status_prob_cols` declares per-state probabilities for feeds where the
    state is inferred rather than measured, and has always said "the likelihood marginalises
    over the states" -- this is that marginalisation, fixed here so the tensor implementation
    has something to be held to:

        L(y) = sum_s P(s) * L_s(y),   L_unobserved = 1  (no constraint, not a zero likelihood)

    computed in log space. The hard-state contract is the degenerate case: probability one on
    one state reproduces `log_likelihood` exactly, and the parity test pins that.

    Args:
        probs: `{state: P(state)}` over exactly the three contract states. Must be finite,
            non-negative, and sum to one within 1e-9 -- a distribution that does not sum to one
            is a feed bug, and renormalising it here would hide the bug inside a likelihood.
        y: The recorded quantity, >= 0.
        cell: The predicted distribution.

    Raises:
        LikelihoodError: unknown or missing states, an invalid distribution, or a bad count.
    """
    _check_count(y)
    _validate_soft_state_probs(probs)
    # log sum_s P(s) exp(ll_s), skipping zero-probability states so an impossible state's
    # log-likelihood (which may be -inf territory) is never evaluated.
    parts = []
    for state, probability in probs.items():
        if probability > 0.0:
            parts.append(math.log(probability) + log_likelihood(state, y, cell))
    top = max(parts)
    return top + math.log(sum(math.exp(v - top) for v in parts))


def soft_informative_weight(probs: dict[str, float], y: int) -> float:
    """How much of a soft cell is INFORMATIVE -- the normalisation weight for a mean loss.

    The hard mask generalises to an expectation: `exact` counts fully, `lower_bound` counts
    only when the floor is positive (a zero floor is vacuous), `unobserved` never. Dividing a
    batch loss by the SUM of these weights is the soft version of dividing by the informative
    count, and reduces to it exactly on hard states. The probabilities are validated by the same
    complete-distribution guard as `soft_log_likelihood`; normalisation cannot accept a feed the
    likelihood refuses.
    """
    _check_count(y)
    _validate_soft_state_probs(probs)
    return float(probs["exact"]) + (float(probs["lower_bound"]) if y >= 1 else 0.0)


def plain_nb_log_sf(c: int, mu: float, k: float) -> float:
    """`log P(Y >= c)` for the UNTRUNCATED NB -- the PLAIN head's censored contribution.

    The plain Negative Binomial head exists because `ForecastModelSpec.resolve()` steps
    `hurdle_negbin` down to `negbin` where the measured zero rate does not justify a hurdle --
    and a head the resolver can select must have its likelihood fixed by this contract, or two
    implementations of the step-down would disagree exactly the way the hurdle once did.

    Same closed form and the same certified fallback as `nb_log_sf`, WITHOUT the division by
    the positive mass: there is no occurrence component to have already claimed the zeros, so
    the survivor is unconditional.

    Args:
        c: The observed floor, >= 1. (`c = 0` is vacuous -- "demand >= 0" -- and is routed to a
            zero contribution by `plain_log_likelihood`, never here.)
        mu: NB mean.
        k: Dispersion.

    Raises:
        LikelihoodError: `c < 1`, invalid parameters, or an uncertifiable tail.
    """
    _check_count(c)
    if c < 1:
        raise LikelihoodError(f"a censored floor must be >= 1, got {c}")
    _check_positive(mu, "mu")
    _check_positive(k, "k")
    survivor = float(betainc(float(c), k, _mu_over_sum(mu, k)))
    # Same accuracy discipline as the conditional form: trust the closed form only inside the
    # normal binary64 range, and require the certified log-space sum outside it.
    if float.fromhex("0x1.0p-1022") <= survivor <= 1.0:
        return math.log(survivor)
    return _log_sf_tail(c, mu, k)


def plain_log_likelihood(state: ObservationState, y: int, mu: float, k: float) -> float:
    """The PLAIN Negative Binomial head's contribution, by observation state.

    | state | observed | contribution |
    |---|---|---|
    | `exact` | y >= 0 | `log P_NB(y)` — a zero is an ordinary count here, not a hurdle event |
    | `lower_bound` | c >= 1 | `log P(Y >= c)` — right-censored, unconditional |
    | `lower_bound` | 0 | `0.0` — vacuous; demand >= 0 says nothing |
    | `unobserved` | any | `0.0` — excluded |

    The mean is `mu`, not `p * mu / (1 - P_NB(0))` -- there is no occurrence component and no
    truncation, which is precisely what makes this the correct degradation for a panel whose
    zero rate does not need one.

    Raises:
        LikelihoodError: unknown state, or a negative/non-integral quantity.
    """
    _check_count(y)
    if state == "unobserved":
        return 0.0
    if state == "lower_bound":
        return 0.0 if y == 0 else plain_nb_log_sf(y, mu, k)
    if state == "exact":
        return nb_log_pmf(y, mu, k)
    raise LikelihoodError(
        f"unknown observation state {state!r}; the contract defines exact, lower_bound and "
        "unobserved, and a state it does not define must not be given a default")


def ancestral_path_plain(
    mus: list[float], ks: list[float], uniforms: list[float],
) -> list[int]:
    """One integer sample path for the PLAIN NB head: one uniform per step, CDF inverted from 0.

    One stage, not two -- there is no occurrence draw to condition on, so the untruncated CDF is
    the right one and zero is an ordinary outcome of it. Randomness is passed in for the same
    reasons as `ancestral_path`: reproducibility from a recorded seed, and the caller owns the
    correlation structure (the factor supplies dependence; these uniforms must be independent
    given it).

    Raises:
        LikelihoodError: mismatched lengths, or a uniform that is not a built-in real in [0, 1).
    """
    if not (len(mus) == len(ks) == len(uniforms)):
        raise LikelihoodError(
            f"{len(mus)} means, {len(ks)} dispersions and {len(uniforms)} uniforms; a path "
            "draws exactly once per step")
    out = []
    for mu, k, u in zip(mus, ks, uniforms, strict=True):
        _check_uniform(u)
        out.append(_invert_nb(u, mu, k))
    return out


def _invert_nb(u: float, mu: float, k: float, max_count: int = 10_000) -> int:
    """Smallest y >= 0 whose UNTRUNCATED CDF reaches `u`. Same cap discipline as `_invert_ztnb`:
    hitting the cap raises rather than returning it, because a silently capped draw puts a spike
    at the cap in every quantile computed from these paths.

    Raises:
        LikelihoodError: the CDF did not reach `u` within `max_count`.
    """
    total = 0.0
    for y in range(0, max_count + 1):
        total += math.exp(nb_log_pmf(y, mu, k))
        if u < total:
            return y
    raise LikelihoodError(
        f"the NB CDF did not reach u={u} within {max_count} counts for mu={mu}, k={k}")


def iwae_bound(block_log_liks: list[float]) -> float:
    """The training objective: an importance-weighted bound over factor draws.

        L = log( (1/S) * sum_s exp( sum_cells log p(y_cell | f^(s)) ) )

    The factor is latent, the likelihood is not conjugate to it, and marginalising exactly is
    intractable -- so the objective is a bound, and WHICH bound is part of the contract.

    Args:
        block_log_liks: One TOTAL log-likelihood per factor draw, each summed over every cell that
            SHARES that draw. The blocking is the part that matters: the factor path is shared
            across the stores of one `(factor_group, product)`, so the sum inside the exponential
            must run over exactly that block. Summing over a smaller unit treats stores as
            independent and throws away the dependence the factor exists to create; summing over
            a larger one couples groups that share no factor.

    Returns:
        The bound. Monotonically non-decreasing in the number of draws, and exact as S -> inf.

    Raises:
        LikelihoodError: no draws were supplied.
    """
    if not block_log_liks:
        raise LikelihoodError("the bound needs at least one factor draw")
    for index, value in enumerate(block_log_liks):
        _check_finite(value, f"block log-likelihood[{index}]")
    m = max(block_log_liks)
    total = sum(math.exp(v - m) for v in block_log_liks)
    return m + math.log(total / len(block_log_liks))


# ---------------------------------------------------------------- 6. ancestral sampling

def ancestral_path(
    cells: list[Hurdle],
    uniforms_occurrence: list[float],
    uniforms_size: list[float],
) -> list[int]:
    """One integer sample path for a series, given per-cell parameters and uniform draws.

    Two-stage, matching the hurdle exactly:

      1. `z_t = 1` if `u_occ[t] < p_t`, else the day is a zero;
      2. when `z_t = 1`, invert the ZERO-TRUNCATED CDF at `u_size[t]` -- never the untruncated
         one, which can return 0 and would contradict the occurrence draw that just happened.

    Randomness is passed IN rather than drawn here so a path is reproducible from a recorded seed
    and so the caller controls the correlation across cells: the FACTOR supplies the dependence
    between stores and days, and these uniforms must be independent given it. Drawing them inside
    would hide that distinction.

    Args:
        cells: One `Hurdle` per horizon step, already conditioned on a drawn factor path.
        uniforms_occurrence: One U(0,1) per step.
        uniforms_size: One U(0,1) per step.

    Returns:
        Integer demand per step.

    Raises:
        LikelihoodError: the three lists have different lengths, or a uniform is not a built-in
            real in [0,1).
    """
    if not (len(cells) == len(uniforms_occurrence) == len(uniforms_size)):
        raise LikelihoodError(
            f"{len(cells)} cells but {len(uniforms_occurrence)} occurrence and "
            f"{len(uniforms_size)} size uniforms; a path must draw once per step for each stage")
    out = []
    for cell, u_occ, u_size in zip(cells, uniforms_occurrence, uniforms_size, strict=True):
        _check_uniform(u_occ)
        _check_uniform(u_size)
        if u_occ >= cell.p:
            out.append(0)
            continue
        out.append(_invert_ztnb(u_size, cell))
    return out


def _invert_ztnb(u: float, cell: Hurdle, max_count: int = 10_000) -> int:
    """Smallest y >= 1 whose zero-truncated CDF reaches `u`.

    Walks up from 1 because retail counts are small and the walk terminates in a few steps for
    almost every cell. `max_count` is a guard against a pathological parameterisation, and hitting
    it raises rather than returning the cap -- a silently capped draw would put a spike at the cap
    in every quantile computed from these paths.

    Raises:
        LikelihoodError: the CDF did not reach `u` within `max_count`.
    """
    total = 0.0
    for y in range(1, max_count + 1):
        total += math.exp(ztnb_log_pmf(y, cell.mu, cell.k))
        if u < total:
            return y
    raise LikelihoodError(
        f"the zero-truncated CDF did not reach u={u} within {max_count} counts for mu={cell.mu}, "
        f"k={cell.k}. Returning the cap would put a spike there in every quantile drawn from "
        "these paths")


def _check_count(y: int) -> None:
    """Raises: LikelihoodError: `y` is not a non-negative integer."""
    if isinstance(y, bool) or not isinstance(y, int) or y < 0:
        raise LikelihoodError(f"observed quantity must be a non-negative integer, got {y!r}")


def _check_uniform(value: float) -> None:
    """Raises: LikelihoodError: `value` is not an exact built-in real in [0, 1)."""
    if not _is_finite_real(value) or not 0.0 <= value < 1.0:
        raise LikelihoodError(f"uniforms must be finite built-in reals in [0,1), got {value!r}")


def _check_positive(value: float, name: str) -> None:
    """Raises: LikelihoodError: `value` is not finite and strictly positive."""
    if not _is_finite_real(value) or value <= 0:
        raise LikelihoodError(f"{name} must be finite and strictly positive, got {value!r}")


def _check_finite(value: float, name: str) -> None:
    """Raises: LikelihoodError: `value` is not a finite real number."""
    if not _is_finite_real(value):
        raise LikelihoodError(f"{name} must be a finite real number, got {value!r}")


def _positive_exp(value: float, name: str) -> float:
    """Exponentiate a finite link value without leaking overflow or silent underflow."""
    try:
        out = math.exp(value)
    except OverflowError as exc:
        raise LikelihoodError(f"{name} is outside the finite binary64 domain: exp({value!r})") \
            from exc
    if not math.isfinite(out) or out <= 0.0:
        raise LikelihoodError(f"{name} is outside the positive finite binary64 domain: "
                              f"exp({value!r})={out!r}")
    return out
