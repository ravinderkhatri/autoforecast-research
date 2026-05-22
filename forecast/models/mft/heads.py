"""The output head: hurdle occurrence + zero-truncated Negative Binomial size, in torch.

**This module implements [`spec/likelihood.py`](../../spec/likelihood.py) and is required to agree
with it.** `test_heads.py` carries the parity test -- the same device that keeps `score.py` and
`score_spark.py` honest, and for the same reason: two implementations of one definition is a real
risk, and the answer is to make disagreement fail the build rather than to pick one and hope.

The contract is the authority. Where this file has a choice the contract does not fix -- how to
vectorise, which stable form of a log -- it makes it here and says why. Where the contract fixes
something, this file follows it exactly, including the parts that look like details:

  · the size component is ZERO-TRUNCATED, so its log-pmf subtracts `log(1 - P_NB(0))`;
  · `lower_bound` contributes the RIGHT-CENSORED term, not a masked one;
  · `E[Y] = p * mu / (1 - P_NB(0))`, which is not `p * mu`.

NUMERICS ARE PART OF THE CONTRACT HERE. `log(1 - exp(x))` for `x < 0` is the expression this whole
head turns on, and the naive form loses all precision at one end or the other. `log1mexp` below
switches between the two stable branches, and the parity test runs it at parameters where the
naive form is visibly wrong.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "ETA_OCC_LIMIT",
    "LOG_K_MAX",
    "LOG_K_MIN",
    "MAX_CENSOR_COUNT",
    "TAIL_TERMS_CAP",
    "HeadError",
    "HurdleNegBinHead",
    "NegBinHead",
    "hurdle_log_likelihood",
    "hurdle_mean",
    "log1mexp",
    "nb_log_pmf",
    "hurdle_log_likelihood_soft",
    "nb_log_survivor",
    "negbin_log_likelihood",
    "negbin_log_likelihood_soft",
    "soft_informative_weight",
    "ztnb_log_pmf",
    "ztnb_log_sf",
]

MAX_CENSOR_COUNT = 512
"""Largest censored floor accepted by the survivor term.

The differentiable torch implementation sums the upper tail from `c`. Vectorised over a batch,
one extreme floor can therefore decide the memory of the whole step.

Exceeding it RAISES rather than clipping. A clipped floor would score a 900-unit stockout as if it
were a 512-unit one, which understates demand in exactly the cells the censoring contract exists
to protect -- and silently, because the loss would still fall.
"""


class HeadError(ValueError):
    """A head was given parameters or observations the likelihood contract does not define."""


# ---------------------------------------------------------------- numerics


def log1mexp(x: Tensor) -> Tensor:
    """Numerically stable `log(1 - exp(x))` for `x < 0`.

    Two branches, because no single expression is accurate across the range:

      · `x > -log 2` (so `exp(x)` is near 1): `log(-expm1(x))`. Here `1 - exp(x)` is a small
        difference of two near-equal numbers, and `expm1` computes it without cancellation.
      · otherwise: `log1p(-exp(x))`. Here `exp(x)` is small and `log1p` keeps the precision that
        `log(1 + tiny)` would lose.

    This is the standard `log1mexp` split. It matters because `1 - P_NB(0)` IS the truncation
    constant: when the size head predicts a very small mean, `P_NB(0)` approaches 1 and the naive
    form returns `log(0) = -inf`, taking the gradient with it.

    Args:
        x: Tensor of strictly negative values.

    Raises:
        HeadError: any element is >= 0, which would make `1 - exp(x)` non-positive.
    """
    if not x.is_floating_point():
        raise HeadError(f"log1mexp requires a floating tensor, got {x.dtype}")
    if bool((torch.isnan(x) | (x >= 0)).any()):
        raise HeadError(
            "log1mexp requires x < 0; a non-negative element means P_NB(0) >= 1, i.e. the size "
            "head has collapsed onto no demand. Clamping past that hides a real failure"
        )
    cut = -math.log(2.0)
    return torch.where(x > cut, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


# ---------------------------------------------------------------- the distribution


def _validate_nb_inputs(y: Tensor, mu: Tensor, k: Tensor, *, minimum: int = 0) -> None:
    """Validate the NB domain before `log`, `lgamma`, or implicit broadcasting sees it."""
    if not all(isinstance(v, Tensor) for v in (y, mu, k)):
        raise HeadError("count, mu and k must be torch tensors")
    if not mu.is_floating_point() or not k.is_floating_point():
        raise HeadError(f"mu and k must be floating tensors, got {mu.dtype} and {k.dtype}")
    if y.dtype == torch.bool or y.is_complex():
        raise HeadError(f"counts must use a real integer or floating dtype, got {y.dtype}")
    if not (y.device == mu.device == k.device):
        raise HeadError("count, mu and k must be on the same device")
    try:
        torch.broadcast_shapes(y.shape, mu.shape, k.shape)
    except RuntimeError as exc:
        raise HeadError(
            f"count {tuple(y.shape)}, mu {tuple(mu.shape)} and k {tuple(k.shape)} are not "
            "broadcast-compatible"
        ) from exc
    if bool((~torch.isfinite(mu) | (mu <= 0)).any()):
        raise HeadError("mu must contain only finite, strictly positive values")
    if bool((~torch.isfinite(k) | (k <= 0)).any()):
        raise HeadError("k must contain only finite, strictly positive values")
    if bool((~torch.isfinite(y)).any()):
        raise HeadError("counts must be finite")
    if y.is_floating_point() and bool((y != torch.floor(y)).any()):
        raise HeadError("counts must be integers, even when stored in a floating tensor")
    if bool((y < minimum).any()):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise HeadError(f"counts must be {qualifier} integers")


def _nb_log_pmf_unchecked(y: Tensor, mu: Tensor, k: Tensor) -> Tensor:
    """NB log-pmf after validation; separated so a tail does not rescan every expanded term."""
    y, mu, k = torch.broadcast_tensors(y, mu, k)
    log_p0 = -_k_log1p_mu_over_k(mu, k)
    # Start with the exact y=0 result. This avoids both inf-inf in the gamma ratio and the
    # otherwise undefined `0 * log1p(k/mu)` at opposite ends of the finite floating domain.
    result = log_p0.clone()
    positive = y > 0
    positive_y, positive_mu, positive_k = y[positive], mu[positive], k[positive]

    gamma_sum = positive_y + positive_k
    combination = (
        torch.lgamma(gamma_sum) - torch.lgamma(positive_k) - torch.lgamma(positive_y + 1.0)
    )
    ratio = positive_y / positive_k
    unstable = (
        ~torch.isfinite(combination)
        | (gamma_sum == positive_k)
        | ((positive_k >= 1_000_000.0) & (ratio <= 1e-4))
    )

    ordinary = combination + log_p0[positive] - positive_y * _log1p_ratio(positive_k, positive_mu)
    correction = _large_k_rising_correction(positive_y, positive_k)
    large_k = (
        correction
        - torch.lgamma(positive_y + 1.0)
        + positive_y * torch.log(positive_mu)
        + log_p0[positive]
        - positive_y * _log1p_ratio(positive_mu, positive_k)
    )

    # Index before scattering so NaNs from an unused inf-inf gamma expression do not enter the
    # autograd graph multiplied by zero (which would still be NaN).
    positive_result = large_k.clone()
    stable = ~unstable
    positive_result = positive_result.masked_scatter(stable, ordinary[stable])
    return result.masked_scatter(positive, positive_result)


def _k_log1p_mu_over_k(mu: Tensor, k: Tensor) -> Tensor:
    """`k * log1p(mu/k)`, retaining the finite scaled limit when the ratio underflows."""
    mu, k = torch.broadcast_tensors(mu, k)
    small = mu <= k
    small_mu, small_k = mu[small], k[small]
    ratio = small_mu / small_k
    use_series = ratio <= 1e-4
    series_ratio = ratio[use_series]
    multiplier = (
        1.0
        - series_ratio / 2.0
        + series_ratio.square() / 3.0
        - series_ratio.pow(3) / 4.0
        + series_ratio.pow(4) / 5.0
    )
    small_result = torch.empty_like(ratio).masked_scatter(
        use_series, small_mu[use_series] * multiplier
    )
    direct = ~use_series
    direct_ratio = ratio[direct]
    small_result = small_result.masked_scatter(
        direct, small_mu[direct] * (torch.log1p(direct_ratio) / direct_ratio)
    )

    result = torch.empty_like(mu).masked_scatter(small, small_result)
    large = ~small
    large_mu, large_k = mu[large], k[large]
    delta = torch.log(large_mu) - torch.log(large_k)
    large_result = large_k * (delta + torch.log1p(torch.exp(-delta)))
    return result.masked_scatter(large, large_result)


def _log1p_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    """`log1p(numerator/denominator)` without an overflowing ratio."""
    numerator, denominator = torch.broadcast_tensors(numerator, denominator)
    small = numerator <= denominator
    result = torch.empty_like(numerator).masked_scatter(
        small, torch.log1p(numerator[small] / denominator[small])
    )
    large = ~small
    delta = torch.log(numerator[large]) - torch.log(denominator[large])
    direct = delta + torch.log1p(torch.exp(-delta))
    return result.masked_scatter(large, direct)


def _softplus(x: Tensor) -> Tensor:
    """Exact floating-point softplus, without `nn.functional.softplus`'s linear shortcut.

    PyTorch's neural-network softplus deliberately returns `x` exactly above a threshold. That is
    appropriate for activations, but at x=27.6 it discards a 1e-12 term that is subsequently
    subtracted from the zero-truncation normalizer. `logaddexp(x, 0)` retains that term.
    """
    return torch.logaddexp(x, torch.zeros_like(x))


def _large_k_rising_correction(y: Tensor, k: Tensor) -> Tensor:
    """Four-term expansion of `sum_i log1p(i/k)` on the selected large-k branch."""
    inverse_k = 1.0 / k
    ratio = y / k
    previous = (y - 1.0) / k
    twice_previous = (2.0 * y - 1.0) / k
    first = y * previous / 2.0
    second = y * previous * twice_previous / 12.0
    third = y * ratio * previous * previous / 12.0
    fourth_power_sum = 3.0 * ratio.square() - 3.0 * ratio * inverse_k - inverse_k.square()
    fourth = y * previous * twice_previous * fourth_power_sum / 120.0
    return first - second + third - fourth


def nb_log_pmf(y: Tensor, mu: Tensor, k: Tensor) -> Tensor:
    """`log P(Y = y)` for the mean-dispersion Negative Binomial. Matches `likelihood.nb_log_pmf`.

    `k` is the shape: larger means less overdispersion, and `k -> inf` gives the Poisson. The two
    log ratios use `log1p` forms so neither a direct ratio nor `k + mu` loses their range first.
    """
    _validate_nb_inputs(y, mu, k)
    return _nb_log_pmf_unchecked(y, mu, k)


def _log_p_zero(mu: Tensor, k: Tensor) -> Tensor:
    """`log P_NB(0) = k * (log k - log(k+mu))`. The truncation constant's input."""
    return -_k_log1p_mu_over_k(mu, k)


def ztnb_log_pmf(y: Tensor, mu: Tensor, k: Tensor) -> Tensor:
    """`log P(Y = y | Y >= 1)` -- the zero-truncated component, WITH its normalisation.

    Subtracting `log(1 - P_NB(0))` is the whole point. Without it the occurrence head has already
    decided whether the day is zero and this component also places mass there, so the likelihood
    double-counts `P(0)` and the occurrence head is biased to compensate: a model that trains,
    converges and is quietly wrong.

    Counts and parameters are checked before gamma/log arithmetic. The internal likelihood path
    uses the unchecked primitive only after doing the same validation once for the whole cell.
    """
    _validate_nb_inputs(y, mu, k, minimum=1)
    # The ordinary cells below still go through the public NB primitive. The small-normalizer
    # cells use an algebraically equivalent form because building the ordinary graph first would
    # already create the inf-inf gradient this branch exists to prevent.
    y, mu, k = torch.broadcast_tensors(y, mu, k)
    log_p0 = _log_p_zero(mu, k)
    # When the positive mass is tiny, direct subtraction cancels two values of order
    # `-log(mu)`. More importantly, autograd differentiates those terms before cancelling them,
    # producing inf-inf for subnormal means. Cancel the scale algebraically on those cells.
    small = -log_p0 <= 0.01
    small_y, small_mu, small_k = y[small], mu[small], k[small]
    a = -log_p0[small]
    centered = _centered_combinatorial(small_y, small_k)
    bridge = _log_r_over_log1p_r(small_mu, small_k) + _log_a_over_one_minus_exp_minus_a(a)
    stable = (
        centered
        + (small_y - 1.0) * torch.log(small_mu)
        - small_y * _log1p_ratio(small_mu, small_k)
        - a
        + bridge
    )
    result = torch.empty_like(mu).masked_scatter(small, stable)

    # Keep the public NB primitive on the ordinary live path, while never constructing the
    # cancellation-prone graph for the small-normalizer cells it cannot differentiate safely.
    ordinary_cells = ~small
    ordinary = nb_log_pmf(y[ordinary_cells], mu[ordinary_cells], k[ordinary_cells]) - log1mexp(
        log_p0[ordinary_cells]
    )
    return result.masked_scatter(ordinary_cells, ordinary)


def _centered_combinatorial(y: Tensor, k: Tensor) -> Tensor:
    """`log((k)_y/y!) - y log(k)`, retaining the large-k correction."""
    gamma_sum = y + k
    combination = torch.lgamma(gamma_sum) - torch.lgamma(k) - torch.lgamma(y + 1.0)
    ratio = y / k
    unstable = (
        ~torch.isfinite(combination) | (gamma_sum == k) | ((k >= 1_000_000.0) & (ratio <= 1e-4))
    )
    ordinary = combination - y * torch.log(k)
    expanded = _large_k_rising_correction(y, k) - torch.lgamma(y + 1.0)
    result = expanded.clone()
    stable = ~unstable
    result = result.masked_scatter(stable, ordinary[stable])
    # The rising product for y=1 is exactly k, so the centered term is exactly zero. Avoid an
    # avoidable subtraction of log(k) from itself even in the ordinary branch.
    at_one = y == 1
    return result.masked_scatter(at_one, torch.zeros_like(result[at_one]))


def _log_r_over_log1p_r(mu: Tensor, k: Tensor) -> Tensor:
    """`log((mu/k) / log1p(mu/k))`, including the removable limit at zero."""
    delta = torch.log(mu) - torch.log(k)
    use_series = delta <= math.log(0.01)
    r = mu[use_series] / k[use_series]
    series = (
        r / 2.0
        - 5.0 * r.square() / 24.0
        + r.pow(3) / 8.0
        - 251.0 * r.pow(4) / 2880.0
        + 19.0 * r.pow(5) / 288.0
        - 19087.0 * r.pow(6) / 362880.0
    )
    result = torch.empty_like(delta).masked_scatter(use_series, series)
    direct_cells = ~use_series
    direct_delta = delta[direct_cells]
    direct = direct_delta - torch.log(_softplus(direct_delta))
    return result.masked_scatter(direct_cells, direct)


def _log_a_over_one_minus_exp_minus_a(a: Tensor) -> Tensor:
    """`log(a / (1-exp(-a)))`, with a cancellation-free expansion near zero."""
    use_series = a <= 0.01
    small = a[use_series]
    series = small / 2.0 - small.square() / 24.0 + small.pow(4) / 2880.0 - small.pow(6) / 181440.0
    result = torch.empty_like(a).masked_scatter(use_series, series)
    direct_cells = ~use_series
    direct_a = a[direct_cells]
    direct = torch.log(direct_a) - log1mexp(-direct_a)
    return result.masked_scatter(direct_cells, direct)


def ztnb_log_sf(c: Tensor, mu: Tensor, k: Tensor) -> Tensor:
    """`log P(Y >= c | Y >= 1)` -- the RIGHT-CENSORED contribution for a stockout.

    Computed as a `logsumexp` over the UPPER tail:

        log P(Y >= c) = logsumexp_{y=c}^{c+N-1} log P_NB(y)

    **Not** `log(1 - sum_{y<c} P(y))`, which is what this function did first and which the parity
    test against `spec/likelihood.py` demolished. That form asks for the difference of two numbers
    that agree to fifteen digits, so it loses everything exactly where the survivor is SMALL --
    the tail, which is large stockouts, which is where censoring matters most. Measured against
    the exact incomplete-beta reference at `c=200, mu=0.05, k=1.5`: the true value is **-681.0**
    and the subtraction returned **-34.03**, off by 647 nats -- pure floating-point residue, with
    no sign that anything was wrong. It also REFUSED valid cells (c=50, mu=2, k=3) whose survivor
    the exact form computes without difficulty.

    The tail form has neither failure. It never subtracts, so there is no cancellation, and it
    stays in log space, so a survivor of 1e-296 is as representable as one of 0.5.

    **TWO REGIMES, split by the survivor itself, because neither form covers the other's.**
    The certified upper-tail sum converges at a rate set by `q = mu/(mu+k)`: at `q ~ 0.996`
    (a fast mover with heavy overdispersion, `mu=55, k=0.2` -- an entirely ordinary censored
    cell) it needs ~8,000 terms and the cap refused a value the pure-Python contract computes
    in closed form. But within the `MAX_CENSOR_COUNT` contract, slow decay and a small survivor
    cannot coexist: a survivor below `COMPLEMENT_MIN_SURVIVOR` after at most 512 counts forces
    an average per-step decay fast enough that the certified tail converges with better than 2x
    margin. So:

      * survivor >= `COMPLEMENT_MIN_SURVIVOR` -- computed as `log1p(-exp(log lower mass))` over
        the at-most-`c-1 <= 511` lower terms. The subtraction the docstring above demolished is
        safe HERE because it is only used where the survivor is large: the absolute error of the
        lower mass is ~1e-15, so the relative error of the survivor is at most ~1e-11 at the
        threshold -- inside the parity tolerance -- and the old failure cases (survivor 1e-296)
        never reach this branch.
      * survivor below it -- the certified tail, whose mode-aware bound now provably converges.

    `N` for the tail is chosen only after reaching the NB mode. For `q = mu/(mu+k)`, adjacent
    probabilities obey

        P(y+1) / P(y) = q * (y+k)/(y+1).

    The old implementation used `q` itself as an upper bound. That is false for `k > 1` and can
    omit almost the whole distribution in the near-Poisson regime. Beyond the mode the displayed
    ratio decreases when `k > 1`; when `k <= 1` it is bounded above by `q`. Thus, if `b` is the
    first omitted count and `rho` is that valid ratio bound, the omitted remainder is at most
    `P(b)/(1-rho)`. `_tail_terms` grows the differentiable finite sum until this proved bound is at
    most `TAIL_RELATIVE_ERROR` times the included mass.

    Raises:
        HeadError: a floor below 1 or above `MAX_CENSOR_COUNT`, or a small-survivor tail needing
            more than `TAIL_TERMS_CAP` terms to converge (unreachable for parameters the floor
            contract admits; kept as the honest failure rather than a truncation).
    """
    _validate_nb_inputs(c, mu, k, minimum=1)
    c, mu, k = torch.broadcast_tensors(c, mu, k)
    top = int(c.max().item())
    if top > MAX_CENSOR_COUNT:
        raise HeadError(
            f"censored floor {top} exceeds MAX_CENSOR_COUNT={MAX_CENSOR_COUNT}. Clipping it would "
            "score a large stockout as a smaller one -- understating demand in exactly the cells "
            "the censoring contract protects, and silently, because the loss would still fall"
        )

    # Conditional on being positive, Y >= 1 has probability one by definition. Besides being
    # exact, this avoids approximating a normalising constant by the same finite tail it divides.
    at_one = c == 1
    if bool(at_one.all()):
        return mu * 0.0 + k * 0.0  # exact, with a zero-gradient autograd connection

    # Size the tail from cells that actually need one. A c=1 cell is exact zero work even when
    # its NB mode is enormous; including it in the batch-wide maximum made an unrelated ordinary
    # censored cell raise TAIL_TERMS_CAP merely because it shared a batch with that vacuous floor.
    active = ~at_one
    active_c, active_mu, active_k = c[active], mu[active], k[active]
    output_dtype = (mu + k).dtype
    # lgamma(k+y)-lgamma(k) loses millinats in float32 near the k=8,000 clamp. Censored cells are
    # already selected before this call, so use float64 for this numerically sensitive path and
    # cast the final log probability back; the cast remains differentiable.
    work_mu, work_k = active_mu.to(torch.float64), active_k.to(torch.float64)
    c64 = active_c.to(torch.float64)
    log_positive = log1mexp(_log_p_zero(work_mu, work_k))
    active_result = (_log_survivor64(c64, work_mu, work_k) - log_positive).to(output_dtype)
    # masked_scatter keeps the active values connected to autograd while c=1 cells retain a
    # zero-gradient connection to their own parameters.
    return (mu * 0.0 + k * 0.0).masked_scatter(active, active_result)


def _log_survivor64(c64: Tensor, work_mu: Tensor, work_k: Tensor) -> Tensor:
    """UNCONDITIONAL `log P(Y >= c)` for validated float64 inputs, `c >= 1`, regime-split.

    The shared core of the conditional (`ztnb_log_sf`) and plain (`nb_log_survivor`) censored
    contributions -- one implementation, so the two heads cannot drift apart numerically. See
    `ztnb_log_sf` for the regime argument; the split and both thresholds live here.
    """
    # LOWER MASS, bounded by the floor contract at 511 terms: log sum_{y=0}^{c-1} P(y), masked
    # per cell because floors differ within a batch.
    lower_top = int(c64.max().item())
    lower_y = torch.arange(lower_top, device=c64.device, dtype=torch.float64)
    lower_terms = _nb_log_pmf_unchecked(lower_y, work_mu.unsqueeze(-1), work_k.unsqueeze(-1))
    below = lower_y.unsqueeze(0) < c64.unsqueeze(-1)
    log_lower = torch.logsumexp(
        torch.where(below, lower_terms, torch.full_like(lower_terms, -math.inf)), dim=-1
    )

    # Regime split. `log_lower <= log(1 - S_MIN)` means the survivor is at least S_MIN and the
    # complement is accurate; rounding can push a numerically-complete lower mass to >= 0, and
    # those cells belong to the tail branch, where the certified sum handles them.
    complement_ok = log_lower <= math.log1p(-COMPLEMENT_MIN_SURVIVOR)
    result64 = torch.zeros_like(work_mu)
    if bool(complement_ok.any()):
        result64 = result64.masked_scatter(complement_ok, log1mexp(log_lower[complement_ok]))
    small = ~complement_ok
    if bool(small.any()):
        tail_c, tail_mu, tail_k = c64[small], work_mu[small], work_k[small]
        n_terms = _tail_terms(tail_c, tail_mu, tail_k)
        offsets = torch.arange(n_terms, device=c64.device, dtype=torch.float64)
        y = tail_c.unsqueeze(-1) + offsets  # [small, n_terms]
        log_terms = _nb_log_pmf_unchecked(y, tail_mu.unsqueeze(-1), tail_k.unsqueeze(-1))
        result64 = result64.masked_scatter(small, torch.logsumexp(log_terms, dim=-1))
    return result64


def nb_log_survivor(c: Tensor, mu: Tensor, k: Tensor) -> Tensor:
    """UNCONDITIONAL `log P(Y >= c)` -- the PLAIN head's censored contribution.

    Same regime-split core as `ztnb_log_sf`, WITHOUT the division by the positive mass: the
    plain Negative Binomial has no occurrence component that already claimed the zeros. `c = 1`
    needs no special case -- the complement over the single lower term P(0) IS `log(1 - P(0))`,
    exactly. Matches `likelihood.plain_nb_log_sf`.

    Raises:
        HeadError: a floor below 1 or above `MAX_CENSOR_COUNT`, or an uncertifiable tail.
    """
    _validate_nb_inputs(c, mu, k, minimum=1)
    c, mu, k = torch.broadcast_tensors(c, mu, k)
    top = int(c.max().item())
    if top > MAX_CENSOR_COUNT:
        raise HeadError(
            f"censored floor {top} exceeds MAX_CENSOR_COUNT={MAX_CENSOR_COUNT}. Clipping it "
            "would score a large stockout as a smaller one, silently"
        )
    output_dtype = (mu + k).dtype
    result64 = _log_survivor64(c.to(torch.float64), mu.to(torch.float64), k.to(torch.float64))
    return result64.to(output_dtype)


TAIL_TERMS_CAP = 8192
"""Most terms the upper-tail sum will take before refusing.

The cap bounds the memory of one step. Exceeding it raises rather than truncating, because a
truncated tail understates the survivor and so makes a large stockout look more surprising than
the model actually thinks it is.
"""

TAIL_RELATIVE_ERROR = 1e-14
"""Maximum proved omitted-tail mass relative to the included upper-tail mass."""

COMPLEMENT_MIN_SURVIVOR = 1e-2
"""Smallest survivor the complement form may compute.

The threshold is set by MEASURED error, not the naive estimate: the lower-mass logsumexp over
up to 511 near-Poisson terms accumulates ~1e-13 in the log, and the subtraction amplifies by
lower/survivor, so at a 1e-4 threshold the log error reached 4.4e-9 -- outside the 1e-9 parity
tolerance. At 1e-2 the amplification is bounded by 1e2 and the measured error sits near 1e-11.
Below the threshold the certified tail takes over, and provably CAN: a survivor under 1e-2
after at most `MAX_CENSOR_COUNT` counts forces an average per-step decay rho <= 0.9911, so the
certified bound converges within ~4,100 terms against a `TAIL_TERMS_CAP` of 8,192 -- 2x margin
on both sides of the boundary. The two regimes tile the floor contract's whole domain; before
the split, q ~ 0.996 cells (mu=55, k=0.2 -- an ordinary fast mover under heavy overdispersion)
were REFUSED while the contract computed them."""


def _tail_terms(
    c: Tensor,
    mu: Tensor,
    k: Tensor,
    eps: float = TAIL_RELATIVE_ERROR,
) -> int:
    """Choose a shared term count using a valid, mode-aware geometric remainder bound.

    Selection is intentionally performed on detached float64 values: the integer number of terms
    is not differentiable, but every term in the resulting sum remains attached to `mu` and `k`.
    The returned value guarantees, for every broadcast cell,

        omitted mass <= eps * included mass.

    Raises:
        HeadError: `eps` is invalid or convergence needs more than `TAIL_TERMS_CAP` terms.
    """
    try:
        finite_eps = type(eps) in (int, float) and math.isfinite(float(eps))
    except OverflowError:
        finite_eps = False
    if not finite_eps or not 0.0 < eps < 1.0:
        raise HeadError(f"tail relative error must be finite and in (0,1), got {eps!r}")

    with torch.no_grad():
        c64, mu64, k64 = torch.broadcast_tensors(
            c.detach().to(torch.float64),
            mu.detach().to(torch.float64),
            k.detach().to(torch.float64),
        )
        mode = torch.where(k64 > 1.0, torch.floor(mu64 * (1.0 - 1.0 / k64)), torch.zeros_like(mu64))
        # Include at least through every cell's mode, so the remainder is monotonically decaying.
        n_terms = max(1, int((mode - c64 + 1.0).clamp_min(1.0).max().item()))
        if n_terms > TAIL_TERMS_CAP:
            raise HeadError(
                f"the upper tail needs at least {n_terms} terms to pass its mode, above "
                f"TAIL_TERMS_CAP={TAIL_TERMS_CAP}. Truncating it would understate the survivor"
            )

        log_eps = math.log(eps)
        while True:
            offsets = torch.arange(n_terms, device=mu64.device, dtype=torch.float64)
            terms_y = c64.unsqueeze(-1) + offsets
            log_included = torch.logsumexp(
                _nb_log_pmf_unchecked(terms_y, mu64.unsqueeze(-1), k64.unsqueeze(-1)), dim=-1
            )

            first_omitted = c64 + n_terms
            q = torch.sigmoid(torch.log(mu64) - torch.log(k64))
            ratio = torch.where(
                k64 > 1.0,
                q * (first_omitted + k64) / (first_omitted + 1.0),
                q,
            )
            decreasing = ratio < 1.0
            safe_ratio = torch.where(decreasing, ratio, torch.zeros_like(ratio))
            log_first_omitted = _nb_log_pmf_unchecked(first_omitted, mu64, k64)
            log_remainder_bound = log_first_omitted - torch.log1p(-safe_ratio)
            converged = decreasing & (log_remainder_bound <= log_included + log_eps)
            if bool(converged.all()):
                return n_terms
            if n_terms == TAIL_TERMS_CAP:
                worst = float((log_remainder_bound - log_included).max().item())
                raise HeadError(
                    f"the upper tail cannot reach relative error {eps:g} within "
                    f"TAIL_TERMS_CAP={TAIL_TERMS_CAP} (worst log remainder ratio {worst:.3f}). "
                    "Truncating it would understate the survivor"
                )
            n_terms = min(TAIL_TERMS_CAP, max(n_terms + 1, n_terms * 2))


def hurdle_mean(p: Tensor, mu: Tensor, k: Tensor) -> Tensor:
    """`E[Y] = p * mu / (1 - P_NB(0))`.

    **Not `p * mu`.** The size component is zero-truncated, so its mean is `mu / (1 - P_NB(0))`,
    strictly greater than `mu`. Using `p * mu` under-forecasts every cell and by more the more
    intermittent the series -- which is where this project's data lives.
    """
    if not isinstance(p, Tensor) or not p.is_floating_point() or p.device != mu.device:
        raise HeadError("p must be a floating tensor on the same device as mu and k")
    _validate_nb_inputs(torch.zeros_like(mu), mu, k)
    if bool((~torch.isfinite(p) | (p < 0) | (p > 1)).any()):
        raise HeadError("p must contain only finite probabilities in [0,1]")
    try:
        p, mu, k = torch.broadcast_tensors(p, mu, k)
    except RuntimeError as exc:
        raise HeadError("p, mu and k must be broadcast-compatible") from exc
    a = -_log_p_zero(mu, k)
    log_conditional_mean = _log_r_over_log1p_r(mu, k) + _log_a_over_one_minus_exp_minus_a(a)
    positive = p > 0
    log_mean = torch.log(p[positive]) + log_conditional_mean[positive]
    limit = math.log(torch.finfo(mu.dtype).max)
    if bool((log_mean > limit).any()):
        raise HeadError(
            f"the hurdle mean exceeds the largest finite value of {mu.dtype}; returning inf "
            "would corrupt path summaries"
        )
    result = torch.zeros_like(log_conditional_mean)
    return result.masked_scatter(positive, torch.exp(log_mean))


# ---------------------------------------------------------------- the likelihood


def hurdle_log_likelihood(
    y: Tensor,
    p: Tensor,
    mu: Tensor,
    k: Tensor,
    state: Tensor,
) -> Tensor:
    """Per-cell log-likelihood, routed by observation state. Matches `likelihood.log_likelihood`.

    | state | observed | contribution |
    |---|---|---|
    | `exact` (0) | 0 | `log(1 - p)` |
    | `exact` (0) | y >= 1 | `log p + log P_ZTNB(y)` |
    | `lower_bound` (1) | c >= 1 | `log p + log P(Y >= c | Y >= 1)` |
    | `lower_bound` (1) | 0 | `0` — vacuous |
    | `unobserved` (2) | any | `0` — excluded |

    Args:
        y: Observed counts, `>= 0`.
        p: Occurrence probability in `(0,1)`.
        mu: Untruncated NB mean, `> 0`.
        k: Dispersion, `> 0`.
        state: Integer codes — 0 `exact`, 1 `lower_bound`, 2 `unobserved`. Integers rather than
            strings because this runs per cell per step; the codes are fixed by `STATE_CODES`.

    Returns:
        Per-cell contribution. **Zero means UNINFORMATIVE, never well predicted** -- so a caller
        must divide by the count of INFORMATIVE cells, not by the tensor's size, or a batch of
        never-ranged cells reads as a batch with a very good loss.

    Raises:
        HeadError: an unknown state code, or a censored floor beyond `MAX_CENSOR_COUNT`.
    """
    if not (y.shape == p.shape == mu.shape == k.shape == state.shape):
        raise HeadError(
            f"y {tuple(y.shape)}, p {tuple(p.shape)}, mu {tuple(mu.shape)}, k {tuple(k.shape)} "
            f"and state {tuple(state.shape)} must describe the same cells"
        )
    _validate_nb_inputs(y, mu, k)
    if bool((~torch.isfinite(p) | (p <= 0) | (p >= 1)).any()):
        raise HeadError("occurrence probabilities must be finite and strictly inside (0,1)")
    if state.dtype == torch.bool or state.is_complex():
        raise HeadError(f"observation state codes must use a real integer dtype, got {state.dtype}")
    if bool((~torch.isfinite(state)).any()):
        raise HeadError("observation state codes must be finite integers")
    if state.is_floating_point() and bool((state != torch.floor(state)).any()):
        raise HeadError("observation state codes must be integers")
    if bool(((state < 0) | (state > 2)).any()):
        raise HeadError(
            "unknown observation state code; the contract defines 0=exact, 1=lower_bound, "
            "2=unobserved, and a code it does not define must not be given a default"
        )

    log_p = torch.log(p)
    log_1mp = torch.log1p(-p)
    positive = y >= 1

    # `exact`, y >= 1. Evaluate only cells which use this term so an irrelevant extreme parameter
    # cannot enter gamma arithmetic, and retain ztnb_log_pmf's small-normalizer cancellation.
    exact_term = log_1mp.clone()
    exact_positive = (state == 0) & positive
    if bool(exact_positive.any()):
        selected = log_p[exact_positive] + ztnb_log_pmf(
            y[exact_positive], mu[exact_positive], k[exact_positive]
        )
        exact_term = exact_term.masked_scatter(exact_positive, selected)

    # `lower_bound`, c >= 1. Same substitution, for the same reason.
    censored_cells = (state == 1) & positive
    if bool(censored_cells.any()):
        # Evaluate only actual censored cells. Computing a full positive tail for every exact and
        # unobserved cell is both wasted memory and a way for an irrelevant parameter to trip the
        # tail cap.
        censored_term = torch.zeros_like(exact_term)
        selected = log_p[censored_cells] + ztnb_log_sf(
            y[censored_cells], mu[censored_cells], k[censored_cells]
        )
        censored_term = censored_term.masked_scatter(censored_cells, selected)
    else:
        censored_term = torch.zeros_like(exact_term)

    out = torch.where(state == 0, exact_term, torch.zeros_like(exact_term))
    return torch.where(censored_cells, censored_term, out)


def negbin_log_likelihood(y: Tensor, mu: Tensor, k: Tensor, state: Tensor) -> Tensor:
    """Per-cell log-likelihood for the PLAIN NB head. Matches `likelihood.plain_log_likelihood`.

    | state | observed | contribution |
    |---|---|---|
    | `exact` (0) | y >= 0 | `log P_NB(y)` — a zero is an ordinary count, not a hurdle event |
    | `lower_bound` (1) | c >= 1 | `log P(Y >= c)` — right-censored, unconditional |
    | `lower_bound` (1) | 0 | `0` — vacuous |
    | `unobserved` (2) | any | `0` — excluded |

    This head exists because `resolve(zero_share=...)` steps `hurdle_negbin` down to `negbin`
    where the measured zero rate does not justify a hurdle. `informative_mask` applies
    unchanged: which cells carry information is a property of the observation, not the head.

    Raises:
        HeadError: an unknown state code, mismatched shapes, or a floor beyond the cap.
    """
    if not (y.shape == mu.shape == k.shape == state.shape):
        raise HeadError(
            f"y {tuple(y.shape)}, mu {tuple(mu.shape)}, k {tuple(k.shape)} and state "
            f"{tuple(state.shape)} must describe the same cells"
        )
    _validate_nb_inputs(y, mu, k)
    if state.dtype == torch.bool or state.is_complex():
        raise HeadError(f"observation state codes must use a real integer dtype, got {state.dtype}")
    if bool((~torch.isfinite(state)).any()):
        raise HeadError("observation state codes must be finite integers")
    if state.is_floating_point() and bool((state != torch.floor(state)).any()):
        raise HeadError("observation state codes must be integers")
    if bool(((state < 0) | (state > 2)).any()):
        raise HeadError(
            "unknown observation state code; the contract defines 0=exact, 1=lower_bound, "
            "2=unobserved, and a code it does not define must not be given a default"
        )

    out = torch.zeros_like(mu)
    exact = state == 0
    if bool(exact.any()):
        out = out.masked_scatter(
            exact, _nb_log_pmf_unchecked(y[exact].to(mu.dtype), mu[exact], k[exact])
        )
    censored = (state == 1) & (y >= 1)
    if bool(censored.any()):
        out = out.masked_scatter(censored, nb_log_survivor(y[censored], mu[censored], k[censored]))
    return out


from spec.forecast_spec import STATE_CODES  # noqa: E402  -- one mapping, defined at the surface

assert STATE_CODES == {"exact": 0, "lower_bound": 1, "unobserved": 2}, (
    "the state codes moved under this module; every docstring and routing branch here is "
    "written against 0=exact, 1=lower_bound, 2=unobserved"
)

ETA_OCC_LIMIT = 14.0
"""Largest |logit| the occurrence head may emit.

`sigmoid` saturates to EXACTLY 1.0 in float32 once the logit passes ~16.6, and then
`log1p(-p)` is `-inf`. The `-inf` itself would be discarded by the `torch.where` that routes
zero and positive observations -- but `where`'s backward multiplies the unselected branch's
local gradient by zero, and `0 * inf` is NaN, which poisons every shared parameter in the
batch. The likelihood contract requires `p` strictly inside (0,1) (`Hurdle.__post_init__`
raises outside it); this clamp is the tensor path's way of holding that requirement instead of
crashing mid-epoch. 14 keeps `1 - p` at ~8.3e-7, comfortably representable, while clamping a
logit no real cell needs -- a per-day sale probability of 0.9999992 is not a forecast, it is a
parameterisation on its way out of the support."""

LOG_K_MIN, LOG_K_MAX = -7.0, 9.0
"""Bounds on the pooled log-dispersion, `k` in [~1e-3, ~8100].

The v3 spec table says the dispersion head is "linear, CLAMPED, partially pooled" and the
clamp was never implemented. Unclamped, `k = exp(linear)` can collapse toward zero -- where
`P_NB(0) -> 1` for any mean and `ztnb_log_pmf` starts refusing cells -- or explode toward
Poisson while contributing nothing but gradient noise. Both ends of this range are far outside
what retail counts support, so the clamp binds only when the head is already wrong."""


def informative_mask(y: Tensor, state: Tensor) -> Tensor:
    """Which cells actually contribute -- everything except `unobserved` and a vacuous stockout.

    Exists because `hurdle_log_likelihood` returns 0 for both a perfectly-predicted impossibility
    and an uninformative cell, and a mean over the whole tensor cannot tell them apart. Dividing
    by this count is what makes the reported loss comparable between batches with different
    amounts of missing exposure.
    """
    return (state == 0) | ((state == 1) & (y >= 1))


def _validated_state_probabilities(
    state_probs: Tensor,
    cell_shape: torch.Size,
    cell_device: torch.device,
) -> Tensor:
    """Return a complete per-cell state distribution, or fail before it can broadcast.

    The scalar reference receives Python floats and can enforce its source-level ``1e-9`` sum
    tolerance directly.  The live tensor feed is stored as float32, where independently rounded
    probabilities that sum to exactly one at source precision can miss one by a few ulps.  Two
    dtype epsilons, capped at float32 precision, are the conservative storage envelope; a
    lower-precision tensor does not buy a wider scientific tolerance. The stored values are
    summed in float64 so reduction rounding cannot widen it. No values are renormalised, and
    float64 retains the reference's stricter ``1e-9`` boundary.
    """
    if not isinstance(state_probs, Tensor):
        raise HeadError("state probabilities must be a floating torch tensor")
    if state_probs.dim() < 1 or state_probs.shape[-1] != len(STATE_CODES):
        raise HeadError(
            f"state probabilities must be a floating [..., {len(STATE_CODES)}] tensor in "
            f"STATE_CODES order, got {state_probs.dtype} {tuple(state_probs.shape)}"
        )
    if not state_probs.is_floating_point():
        raise HeadError(
            f"state probabilities must be a floating [..., {len(STATE_CODES)}] tensor in "
            f"STATE_CODES order, got {state_probs.dtype} {tuple(state_probs.shape)}"
        )
    if state_probs.shape[:-1] != cell_shape:
        raise HeadError(
            f"state probabilities {tuple(state_probs.shape)} do not describe the cells "
            f"{tuple(cell_shape)}"
        )
    if state_probs.device != cell_device:
        raise HeadError(
            f"state probabilities must be on the same device as their cells, got "
            f"{state_probs.device} and {cell_device}"
        )
    if bool((~torch.isfinite(state_probs) | (state_probs < 0)).any()):
        raise HeadError("state probabilities must be finite and non-negative")
    # Audit the STORED values in float64. A low-precision three-term reduction can round a sum
    # just outside the allowed envelope back onto its boundary and make ``> tolerance`` accept it.
    total = state_probs.to(torch.float64).sum(dim=-1)
    storage_eps = min(torch.finfo(state_probs.dtype).eps, torch.finfo(torch.float32).eps)
    sum_tolerance = max(1e-9, 2.0 * storage_eps)
    if bool(((total - 1.0).abs() > sum_tolerance).any()):
        raise HeadError(
            "state probabilities must sum to one per cell within storage precision; "
            "renormalising here would hide a broken feed inside a likelihood"
        )
    return state_probs


def _soft_mix(ll_exact: Tensor, ll_lower: Tensor, state_probs: Tensor) -> Tensor:
    """`log( P(exact) e^{ll_exact} + P(lower) e^{ll_lower} + P(unobserved) )` per cell.

    The shared marginalisation core: `log(0) = -inf` makes a zero-probability state vanish from
    the logsumexp exactly, so an impossible state's likelihood is mixed with weight zero rather
    than evaluated into a NaN product.
    """
    if ll_lower.shape != ll_exact.shape or ll_lower.device != ll_exact.device:
        raise HeadError(
            "exact and lower-bound likelihood terms must describe the same cells and device"
        )
    state_probs = _validated_state_probabilities(state_probs, ll_exact.shape, ll_exact.device)
    log_probs = torch.log(state_probs)
    terms = torch.stack(
        [log_probs[..., 0] + ll_exact, log_probs[..., 1] + ll_lower, log_probs[..., 2]], dim=-1
    )
    return torch.logsumexp(terms, dim=-1)


def hurdle_log_likelihood_soft(
    y: Tensor, p: Tensor, mu: Tensor, k: Tensor, state_probs: Tensor
) -> Tensor:
    """The hurdle contribution when the STATE is uncertain: `likelihood.soft_log_likelihood`.

    `ObservationSpec.status_prob_cols` has always said "the likelihood marginalises over the
    states"; this is the tensor half of that sentence. Probability one on one state reproduces
    `hurdle_log_likelihood` exactly, and the parity test pins it.
    """
    zeros = torch.zeros(y.shape, dtype=torch.long, device=y.device)
    ll_exact = hurdle_log_likelihood(y, p, mu, k, zeros)
    ll_lower = hurdle_log_likelihood(y, p, mu, k, torch.ones_like(zeros))
    return _soft_mix(ll_exact, ll_lower, state_probs)


def negbin_log_likelihood_soft(y: Tensor, mu: Tensor, k: Tensor, state_probs: Tensor) -> Tensor:
    """The plain head's soft contribution -- same mixture, plain hard terms."""
    zeros = torch.zeros(y.shape, dtype=torch.long, device=y.device)
    ll_exact = negbin_log_likelihood(y, mu, k, zeros)
    ll_lower = negbin_log_likelihood(y, mu, k, torch.ones_like(zeros))
    return _soft_mix(ll_exact, ll_lower, state_probs)


def soft_informative_weight(y: Tensor, state_probs: Tensor) -> Tensor:
    """Expected informative mass per cell: `P(exact) + P(lower_bound) * [y >= 1]`.

    The soft generalisation of `informative_mask` -- the loss divides by the SUM of these, and
    on degenerate (hard) probabilities it reduces to the informative count exactly.
    """
    if not isinstance(y, Tensor):
        raise HeadError("counts must be a torch tensor")
    if y.dtype == torch.bool or y.is_complex():
        raise HeadError(f"counts must use a real integer or floating dtype, got {y.dtype}")
    if bool((~torch.isfinite(y)).any()):
        raise HeadError("counts must be finite")
    if y.is_floating_point() and bool((y != torch.floor(y)).any()):
        raise HeadError("counts must be integers, even when stored in a floating tensor")
    if bool((y < 0).any()):
        raise HeadError("counts must be non-negative integers")
    state_probs = _validated_state_probabilities(state_probs, y.shape, y.device)
    return state_probs[..., 0] + state_probs[..., 1] * (y >= 1).to(state_probs.dtype)


# ---------------------------------------------------------------- the module

_INDEX_DTYPES = (torch.int32, torch.int64)


def _validated_head_cell_shape(
    hidden: Tensor,
    static: Tensor,
    *baselines: tuple[str, Tensor],
    d_model: int,
    n_static: int,
    parameter: Tensor,
) -> torch.Size:
    """Validate head inputs before a singleton baseline can broadcast into a predictor."""
    values = (hidden, static, *(value for _, value in baselines))
    if not all(isinstance(value, Tensor) for value in values):
        raise HeadError("hidden, static and baseline inputs must be torch tensors")
    if not all(value.is_floating_point() for value in values):
        raise HeadError("hidden, static and baseline inputs must be floating tensors")
    if hidden.dim() < 1 or hidden.shape[-1] != d_model:
        raise HeadError(f"hidden must end in d_model={d_model}, got shape {tuple(hidden.shape)}")
    cell_shape = hidden.shape[:-1]
    if static.dim() < 1 or static.shape[:-1] != cell_shape or static.shape[-1] != n_static:
        raise HeadError(
            f"static must have one exact cell shape {tuple(cell_shape)} followed by "
            f"n_static={n_static}, got {tuple(static.shape)}"
        )
    for name, baseline in baselines:
        if baseline.shape != cell_shape:
            raise HeadError(
                f"{name} must have one exact cell shape {tuple(cell_shape)}, got "
                f"{tuple(baseline.shape)}; singleton broadcasting is not allowed"
            )
    if any(value.device != parameter.device for value in values):
        raise HeadError("hidden, static and baseline inputs must be on the head parameter device")
    return cell_shape


def _validated_series_index(
    series_index: Tensor,
    residual: nn.Embedding,
    *predictors: Tensor,
) -> Tensor:
    """Validate a per-cell residual index before the embedding lookup can broadcast or fail."""
    if not isinstance(series_index, Tensor):
        raise HeadError("series_index must be an integral torch tensor")
    if series_index.dtype not in _INDEX_DTYPES:
        raise HeadError(
            f"series_index must use an integral int32 or int64 dtype, got {series_index.dtype}"
        )

    cell_shape = predictors[0].shape
    if any(predictor.shape != cell_shape for predictor in predictors[1:]):
        raise HeadError(
            "head predictors must have one exact cell shape before applying a per-series "
            "residual; broadcasting a singleton predictor would assign the wrong residual"
        )
    if series_index.shape != cell_shape:
        raise HeadError(
            f"series_index {tuple(series_index.shape)} must exactly match predictor cells "
            f"{tuple(cell_shape)}; scalar or singleton broadcasting is not allowed"
        )

    predictor_device = predictors[0].device
    if series_index.device != predictor_device or series_index.device != residual.weight.device:
        raise HeadError(
            f"series_index must be on the predictor and residual device, got "
            f"{series_index.device}, {predictor_device}, and {residual.weight.device}"
        )
    if bool(((series_index < 0) | (series_index >= residual.num_embeddings)).any()):
        raise HeadError(
            f"series_index values must lie in [0, {residual.num_embeddings}), before the "
            "per-series residual lookup"
        )
    return series_index


class HurdleNegBinHead(nn.Module):
    """Maps a hidden representation to `(p, mu, k)`, anchored to shrunk lagged baselines.

    Both heads are OFFSET by a baseline rather than predicting from scratch:

      · occurrence, in logit space, by a **Beta-shrunk** lagged frequency. Not clamped: for a slow
        mover the lagged frequency is often exactly 0, and `logit(1e-4)` is -9.2, so the network
        would need +6 of its own signal just to say "5% chance" and never recovers.
      · size, in log space, by a **category-shrunk** lagged mean positive size. For a product that
        has never sold that mean is 0/0, and it would take the gradient with it through the log.

    Both shrinkages happen upstream, in the feature modules -- this head consumes them and would
    be wrong to recompute them. What it owns is that they enter as OFFSETS, so an untrained
    network starts at the baseline rather than at zero.

    Dispersion is **pooled from static attributes** with a per-series residual, and the latent
    factor deliberately does NOT enter it: letting a daily factor widen `k` would let the model
    explain a bad day by widening the distribution rather than shifting its mean, which is how
    intervals end up calibrated on average and useless during promotions.

    Attributes:
        d_model: Width of the incoming representation.
        n_static: Width of the static-attribute vector that pools the dispersion.
    """

    def __init__(
        self, d_model: int, n_static: int, *, log_k_init: float = 0.0, n_series: int = 0
    ) -> None:
        """Args:
        d_model: Width of the trunk's output.
        n_static: Width of the static-attribute vector.
        log_k_init: Starting value for the pooled `log k`. 0.0 means `k = 1`, a geometric
            size distribution -- deliberately overdispersed at initialisation, because
            starting under-dispersed makes the first epochs punish every large day.
        n_series: Rows in the per-series dispersion residual, 0 to disable. The v3 spec
            table pools dispersion from static attributes WITH a weight-decayed per-series
            residual: attributes say what a product of this kind is like, the residual says
            what this one turned out to be like, and the decay keeps the second small until
            the series has earned it. Zero-initialised, so an untrained model is exactly
            the pooled one.
        """
        super().__init__()
        self.occurrence = nn.Linear(d_model, 1)
        self.size = nn.Linear(d_model, 1)
        self.dispersion = nn.Linear(n_static, 1)
        self.dispersion_residual = nn.Embedding(n_series, 1) if n_series else None
        if self.dispersion_residual is not None:
            nn.init.zeros_(self.dispersion_residual.weight)
        self.log_k_base = nn.Parameter(torch.tensor(float(log_k_init)))
        # Start both heads AT the baseline: zero BIAS means the offset is the prediction on step
        # one, which is the whole point of anchoring them.
        #
        # The weights are SMALL, not zero, and the difference is not cosmetic. With weights of
        # exactly zero the chain rule sends zero gradient to `hidden`, so the entire encoder --
        # branches, fusion, trunk, decoder -- receives NOTHING on the first step and only starts
        # learning once the head's own weights have moved. A gradient test that walked the
        # components found it: "demand received no gradient". Small weights keep the prediction
        # within a hair of the baseline while letting the stack train from step one.
        for layer in (self.occurrence, self.size, self.dispersion):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden: Tensor,
        baseline_logit: Tensor,
        baseline_log_size: Tensor,
        static: Tensor,
        factor_occurrence: Tensor | None = None,
        factor_size: Tensor | None = None,
        series_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return `(p, mu, k)`.

            logit(p) = clamp( baseline_logit    + w_occ . hidden  + <loading_occ,  f> )
            log(mu)  =        baseline_log_size + w_size . hidden + <loading_size, f>
            log(k)   = clamp( log_k_base        + w_disp . static + residual[series] )

        The factor terms arrive already contracted with their per-store loadings -- `factor.py`
        owns that, because the loadings are indexed by store while the factor is indexed by
        `(factor_group, product, step)`, and the head has no business knowing either index.

        The occurrence logit is clamped to `±ETA_OCC_LIMIT` and the log-dispersion to
        `[LOG_K_MIN, LOG_K_MAX]`; both docstrings above say why. `mu` is deliberately NOT
        clamped: a runaway mean is refused downstream (`support_size`, `ztnb_log_pmf`) as a
        real failure, and clamping past it would hide the collapse the refusal exists to show.

        Args:
            hidden: `[..., d_model]` from the decoder.
            baseline_logit: `[...]` Beta-shrunk lagged frequency, in logit space.
            baseline_log_size: `[...]` category-shrunk lagged positive size, in log space.
            static: `[..., n_static]` static attributes, for the pooled dispersion.
            factor_occurrence: `[...]` or None. Already `<loading_occ, f>`.
            factor_size: `[...]` or None. Already `<loading_size, f>`.
            series_index: `[...]` long tensor selecting each cell's dispersion-residual row,
                same shape as `baseline_logit` (broadcast by the caller). Required when the
                head was built with `n_series`, refused when it was not -- a silently ignored
                index and a silently skipped residual are the same bug from two directions.

        Returns:
            `p` in (0,1), `mu > 0`, `k > 0`, each broadcast to the cell shape.

        Raises:
            HeadError: `series_index` given without a residual table, or a table without one.
        """
        eta_occ, eta_size, log_k = self.linear_predictors(
            hidden, baseline_logit, baseline_log_size, static, series_index
        )
        return self.from_predictors(eta_occ, eta_size, log_k, factor_occurrence, factor_size)

    def linear_predictors(
        self,
        hidden: Tensor,
        baseline_logit: Tensor,
        baseline_log_size: Tensor,
        static: Tensor,
        series_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The factor-independent half: `(eta_occ, eta_size, log_k)`, pre-clamp.

        THE SPLIT, AND WHY IT EXISTS. `linear_predictors` computes the parts that do NOT depend
        on a factor draw -- the projections of `hidden`, the pooled dispersion and its per-series
        residual -- and `from_predictors` adds the drawn factor, then applies the clamps and
        transforms. Under the old single method, integrating over 1,000 factor realisations re-ran
        `self.occurrence(hidden)` and `self.dispersion(static)` over tensors expanded to
        `[draws, cells, horizon, d]`, i.e. a thousand times, for values that are identical in every
        one of them. `forward` still composes the two, so there is one code path and callers that do
        not care are unaffected.

            Returns unclamped predictors deliberately: the clamps belong with the transforms in
            `from_predictors`, so adding the factor and then clamping happens in that order -- the
            order the single method used.
        """
        _validated_head_cell_shape(
            hidden,
            static,
            ("baseline_logit", baseline_logit),
            ("baseline_log_size", baseline_log_size),
            d_model=self.occurrence.in_features,
            n_static=self.dispersion.in_features,
            parameter=self.occurrence.weight,
        )
        eta_occ = baseline_logit + self.occurrence(hidden).squeeze(-1)
        eta_size = baseline_log_size + self.size(hidden).squeeze(-1)
        log_k = self.log_k_base + self.dispersion(static).squeeze(-1)
        if self.dispersion_residual is not None:
            if series_index is None:
                raise HeadError(
                    "this head carries a per-series dispersion residual but no series_index was "
                    "passed; skipping it silently would make the residual a table nothing reads"
                )
            checked_index = _validated_series_index(
                series_index, self.dispersion_residual, eta_occ, eta_size, log_k
            )
            log_k = log_k + self.dispersion_residual(checked_index).squeeze(-1)
        elif series_index is not None:
            raise HeadError(
                "series_index was passed but this head has no dispersion residual; the caller "
                "believes a mechanism exists that was never built"
            )
        return eta_occ, eta_size, log_k

    def from_predictors(
        self,
        eta_occ: Tensor,
        eta_size: Tensor,
        log_k: Tensor,
        factor_occurrence: Tensor | None = None,
        factor_size: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Add the drawn factor to cached predictors and return `(p, mu, k)`.

        The per-draw half. Everything here is elementwise, so integrating over many factor
        realisations costs arithmetic on `[draws, cells, horizon]` rather than a repeated
        projection of `[draws, cells, horizon, d_model]`.
        """
        if factor_occurrence is not None:
            eta_occ = eta_occ + factor_occurrence
        if factor_size is not None:
            eta_size = eta_size + factor_size
        return (
            torch.sigmoid(eta_occ.clamp(-ETA_OCC_LIMIT, ETA_OCC_LIMIT)),
            torch.exp(eta_size),
            torch.exp(log_k.clamp(LOG_K_MIN, LOG_K_MAX)),
        )

    def loss(
        self,
        y: Tensor,
        state: Tensor,
        hidden: Tensor,
        baseline_logit: Tensor,
        baseline_log_size: Tensor,
        static: Tensor,
        factor_occurrence: Tensor | None = None,
        factor_size: Tensor | None = None,
        series_index: Tensor | None = None,
    ) -> Tensor:
        """Mean negative log-likelihood over INFORMATIVE cells only.

        Divided by the informative count rather than by the tensor size. A batch that is mostly
        never-ranged would otherwise report a very small loss simply because most of its cells
        contribute zero -- and the loss would fall as coverage got worse.

        Raises:
            HeadError: the batch contains no informative cell, so there is nothing to average.
        """
        p, mu, k = self.forward(
            hidden,
            baseline_logit,
            baseline_log_size,
            static,
            factor_occurrence,
            factor_size,
            series_index,
        )
        ll = hurdle_log_likelihood(y, p, mu, k, state)
        mask = informative_mask(y, state)
        n = mask.sum()
        if int(n) == 0:
            raise HeadError(
                "no informative cell in this batch: every cell is unobserved or a vacuous "
                "stockout. Averaging over the tensor instead would report a loss of zero"
            )
        return -(ll * mask).sum() / n


class NegBinHead(nn.Module):
    """Maps a hidden representation to `(mu, k)` -- the PLAIN Negative Binomial head.

    The head `resolve(zero_share=...)` steps down to when the measured zero rate does not
    justify a hurdle: no occurrence component, no truncation, `E[Y] = mu`. Everything else
    mirrors `HurdleNegBinHead` deliberately -- the size prediction is offset by the same
    category-shrunk lagged baseline, the dispersion is pooled from statics with the same
    clamped range and the same weight-decayed per-series residual, and the factor's size
    loading enters the same linear predictor -- so the choice between the two heads changes
    exactly one thing: whether zeros get their own component.
    """

    def __init__(
        self, d_model: int, n_static: int, *, log_k_init: float = 0.0, n_series: int = 0
    ) -> None:
        super().__init__()
        self.size = nn.Linear(d_model, 1)
        self.dispersion = nn.Linear(n_static, 1)
        self.dispersion_residual = nn.Embedding(n_series, 1) if n_series else None
        if self.dispersion_residual is not None:
            nn.init.zeros_(self.dispersion_residual.weight)
        self.log_k_base = nn.Parameter(torch.tensor(float(log_k_init)))
        # Same small-not-zero initialisation as the hurdle head, for the same measured reason:
        # zero weights send zero gradient upstream and starve the encoder.
        for layer in (self.size, self.dispersion):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden: Tensor,
        baseline_log_size: Tensor,
        static: Tensor,
        factor_size: Tensor | None = None,
        series_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return `(mu, k)`.

            log(mu) =        baseline_log_size + w_size . hidden + <loading_size, f>
            log(k)  = clamp( log_k_base        + w_disp . static + residual[series] )

        Raises:
            HeadError: `series_index` and the residual table do not travel together.
        """
        eta_size, log_k = self.linear_predictors(hidden, baseline_log_size, static, series_index)
        return self.from_predictors(eta_size, log_k, factor_size)

    def linear_predictors(
        self,
        hidden: Tensor,
        baseline_log_size: Tensor,
        static: Tensor,
        series_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The factor-independent half: `(eta_size, log_k)`, pre-clamp. See the hurdle head's
        `linear_predictors` for why this is separable."""
        _validated_head_cell_shape(
            hidden,
            static,
            ("baseline_log_size", baseline_log_size),
            d_model=self.size.in_features,
            n_static=self.dispersion.in_features,
            parameter=self.size.weight,
        )
        eta_size = baseline_log_size + self.size(hidden).squeeze(-1)
        log_k = self.log_k_base + self.dispersion(static).squeeze(-1)
        if self.dispersion_residual is not None:
            if series_index is None:
                raise HeadError(
                    "this head carries a per-series dispersion residual but no series_index was "
                    "passed; skipping it silently would make the residual a table nothing reads"
                )
            checked_index = _validated_series_index(
                series_index, self.dispersion_residual, eta_size, log_k
            )
            log_k = log_k + self.dispersion_residual(checked_index).squeeze(-1)
        elif series_index is not None:
            raise HeadError(
                "series_index was passed but this head has no dispersion residual; the caller "
                "believes a mechanism exists that was never built"
            )
        return eta_size, log_k

    def from_predictors(
        self,
        eta_size: Tensor,
        log_k: Tensor,
        factor_size: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Add the drawn factor to cached predictors and return `(mu, k)`."""
        if factor_size is not None:
            eta_size = eta_size + factor_size
        return torch.exp(eta_size), torch.exp(log_k.clamp(LOG_K_MIN, LOG_K_MAX))
