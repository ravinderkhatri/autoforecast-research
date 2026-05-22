"""Ancestral sampling: coherent integer sample paths, the model's actual output.

Implements part 6 of [`spec/likelihood.py`](../../spec/likelihood.py), and `test_sample.py` holds
it to the contract exactly as `heads.py` and `factor.py` are held.

**Paths, not quantiles, are the output.** Quantiles are a summary of these. The distinction is the
whole reason the factor exists: ordering decisions are made on the CUMULATIVE demand over a lead
time plus a review period, and a per-day marginal quantile cannot be summed into one. A path can.

WHERE COHERENCE COMES FROM, precisely. Two cells in the same `(factor_group, product)` are
dependent because they were computed from the SAME drawn factor path -- that happens upstream, in
`factor.py`, before this module sees anything. Given the factor, the two uniform draws per cell are
INDEPENDENT, and they have to be: if this module added its own correlation the dependence would be
counted twice and the lead-time sums would be too wide.

So the sampler's job is narrow and it is stated here so nobody widens it: draw the hurdle, invert
the zero-truncated CDF, return integers.
"""

from __future__ import annotations

import math

import torch
from forecast.models.mft.heads import nb_log_pmf, ztnb_log_pmf
from torch import Tensor

__all__ = ["SampleError", "sample_paths", "sample_paths_negbin", "support_size"]


class SampleError(ValueError):
    """Sampling was asked for something the distribution cannot supply."""


MAX_SUPPORT = 4096
"""Largest count the inverted CDF will consider.

The support has to be truncated somewhere to vectorise the inversion, and where matters: a draw
capped at the ceiling puts a SPIKE there, and every quantile computed from these paths inherits
it. So exceeding the cap raises instead, and `support_size` derives the required extent from the
distribution rather than assuming one.
"""


def _validate_tail_mass(tail_mass: float) -> float:
    """Return a valid omitted-mass target, shared by every support routine."""
    if type(tail_mass) not in (int, float):
        raise SampleError(f"tail_mass must be finite and in (0,1), got {tail_mass!r}")
    try:
        converted = float(tail_mass)
    except (OverflowError, TypeError, ValueError):
        converted = float("nan")
    if not math.isfinite(converted) or not 0.0 < converted < 1.0:
        raise SampleError(f"tail_mass must be finite and in (0,1), got {tail_mass!r}")
    return converted


def _offending_cells(
    mask: Tensor,
    mu: Tensor,
    k: Tensor,
    *,
    source_flat_indices: Tensor | None = None,
    limit: int = 5,
) -> str:
    """Human-checkable flattened indices and unrounded parameters for a support refusal."""
    flat_mask = mask.reshape(-1)
    indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1).tolist()
    flat_mu, flat_k = mu.reshape(-1), k.reshape(-1)
    source_indices = (
        torch.arange(flat_mu.numel(), device=mu.device)
        if source_flat_indices is None
        else source_flat_indices
    )
    source_indices = source_indices.reshape(-1)
    shown = ", ".join(
        f"flat {int(source_indices[index])} "
        f"(mu={float(flat_mu[index]):.17g}, k={float(flat_k[index]):.17g})"
        for index in indices[:limit]
    )
    if len(indices) > limit:
        shown += f", ... {len(indices) - limit} more"
    return shown


def _support_sizes(
    mu: Tensor,
    k: Tensor,
    tail_mass: float,
    *,
    zero_truncated: bool,
    source_flat_indices: Tensor | None = None,
) -> Tensor:
    """Per-cell geometric-tail proof for either ZTNB or ordinary NB support."""
    _validate_size_parameters(mu, k)
    target = _validate_tail_mass(tail_mass)
    mu64, k64 = mu.detach().to(torch.float64), k.detach().to(torch.float64)
    mode = torch.zeros_like(mu64)
    has_positive_mode = k64 > 1.0
    mode = mode.masked_scatter(
        has_positive_mode,
        torch.floor(mu64[has_positive_mode] * (1.0 - 1.0 / k64[has_positive_mode])),
    )
    if bool((~torch.isfinite(mode)).any()):
        raise SampleError(
            "the distribution's mode is not representable for finite parameters; sampling "
            "cannot choose a proved finite support"
        )

    log_target = math.log(target)

    def bounded_at(n: Tensor) -> Tensor:
        """Prove the complete tail after elementwise final included counts ``n``."""
        first = n.to(torch.float64) + 1.0
        q = _mu_over_sum(mu64, k64)
        rho = torch.where(k64 > 1.0, q * (first + k64) / (first + 1.0), q)
        log_first = (
            ztnb_log_pmf(first, mu64, k64) if zero_truncated else nb_log_pmf(first, mu64, k64)
        )
        log_bound = log_first - torch.log1p(-rho)
        return (rho < 1.0) & (log_bound <= log_target)

    # ZTNB starts at one. Plain NB includes zero, and for a sufficiently concentrated cell zero
    # may already be a proved complete support. The sampler may round either result UP to a tier.
    minimum = 1 if zero_truncated else 0
    distribution = "zero-truncated NB" if zero_truncated else "plain NB"
    # Compare in float64 BEFORE converting to int64. A valid finite mode can be larger than
    # int64 and PyTorch then saturates/wraps the cast, hiding the very overflow we must report.
    above = mode > float(MAX_SUPPORT)
    if bool(above.any()):
        raise SampleError(
            f"{int(above.sum())} cell(s) have a {distribution} mode already above "
            f"MAX_SUPPORT={MAX_SUPPORT}; capping would put a spike at the ceiling. "
            f"Offending cells: "
            f"{_offending_cells(above, mu64, k64, source_flat_indices=source_flat_indices)}"
        )

    low = mode.to(torch.int64).clamp_min(minimum)
    ceiling = torch.full_like(low, MAX_SUPPORT)
    unbounded = ~bounded_at(ceiling)
    if bool(unbounded.any()):
        qualifier = "zero-truncated " if zero_truncated else ""
        raise SampleError(
            f"{int(unbounded.sum())} cell(s) need a support above MAX_SUPPORT={MAX_SUPPORT} "
            f"to guarantee omitted {qualifier}mass <= {target}. Capping would put a spike at "
            f"the ceiling. Offending cells: "
            f"{_offending_cells(unbounded, mu64, k64, source_flat_indices=source_flat_indices)}"
        )

    high = ceiling.clone()
    # MAX_SUPPORT is 4096. Fourteen iterations are deliberately more than the twelve bisections
    # needed for an interval of that width, and ``done`` freezes cells that have converged early.
    for _ in range(int(math.log2(MAX_SUPPORT)) + 2):
        middle = (low + high) // 2
        done = low + 1 >= high
        ok = bounded_at(middle)
        high = torch.where(done, high, torch.where(ok, middle, high))
        low = torch.where(done, low, torch.where(ok, low, middle))
    return torch.where(bounded_at(low), low, high)


def support_size(mu: Tensor, k: Tensor, tail_mass: float = 1e-12) -> int:
    """How far the support must run so the omitted upper tail is below `tail_mass`.

    This is a guarantee, not a normal approximation. If support ends at `n`, the first omitted
    probability is `P_ZTNB(n+1)`. Past the mode, adjacent NB probabilities have ratio

        q * (y+k)/(y+1),  q = mu/(mu+k).

    For `k > 1` that ratio decreases with `y`; for `k <= 1` it is bounded above by `q`. Therefore
    the complete omitted zero-truncated tail is bounded by `P_ZTNB(n+1)/(1-rho)`. We binary-search
    for a common `n` whose proved bound is at most `tail_mass` for every cell. The former heuristic
    used `q` as though it were the ratio itself; around a near-Poisson mode it can be close to one
    while `q` is tiny, causing a material tail to be silently renormalised away.

    Raises:
        SampleError: parameters are invalid or the required support exceeds `MAX_SUPPORT`.
    """
    return int(_support_sizes(mu, k, tail_mass, zero_truncated=True).max())


PMF_TERM_SLACK = 32768.0
"""Floor on the CDF tolerance, in units of float64 eps, covering PMF EVALUATION error.

Not accumulation -- that scales with the term count and is handled separately. This covers the
error in the terms themselves, which is what actually dominates at the edges of the head's
parameter domain:

  * a distribution concentrated on one or two counts puts a term of size ~1 in the sum, and that
    single `exp(ztnb_log_pmf(...))` carries its own error whole (mu=0.03, k=316: ~660 ulps);
  * large `k` is worse. `ztnb_log_pmf` differences two lgammas that are both enormous there --
    at k~8000 they are each around 63,000 -- so catastrophic cancellation amplifies the relative
    error of a small result.

MEASURED rather than chosen. Sweeping the head's domain and comparing torch's truncated sum
against a scipy `nbinom` reference: the TRUE omitted tail never exceeded 9.25e-13, comfortably
inside the 1e-12 proof, while TORCH's evaluation of that same sum was off by up to 3.28e-12
(~14,768 ulps) at mu=0.016, k=5152. 32768 eps ~ 7.3e-12 covers that with roughly 2x margin.

WHAT THIS COSTS, said plainly: the lower check can no longer detect a dropped tail smaller than
about 8.3e-12 through tier 2,048, and the tier-scaled term raises that to about 1.6e-11 at the
4,096 ceiling (including the 1e-12 proved-tail allowance). Torch's own PMF arithmetic is not
accurate enough to certify below that level. It still catches a MATERIAL tail loss, which is
orders of magnitude larger and is what the guard is for. A tighter certificate needs a more
accurate PMF, not a smaller threshold."""

UPPER_SUM_GUARD = 1e-9
"""How far above 1 the truncated CDF may sum before it is called broken.

Deliberately loose. A probability sum exceeding 1 cannot mean a dropped tail -- only arithmetic
error -- so this side of the check exists to catch gross failure, not to certify anything. The
tight side is the lower bound, which is the actual contract."""


def support_sizes(mu: Tensor, k: Tensor, tail_mass: float = 1e-12) -> Tensor:
    """PER-CELL proved support. `[*mu.shape]` int64, each the smallest `n` bounding THAT cell.

    `support_size` returns the maximum bound for the whole batch. That is correct but expensive
    if it is then used as one common CDF width, in a way that is easy to miss:
    the batch's largest REQUIRED SUPPORT sizes the CDF table that EVERY cell pays for. Measured on
    the benchmark's fixed-k=1.5 fixture at 2,000 cells x 28 days, one cell moving from mu=2 to
    mu=100 takes the shared support
    from 54 to 1,978 and the coexisting float64 PMF+CDF tables from 0.05 GB to 1.77 GB -- and at
    mu=300, k=1.5 it stops being a
    slowdown and REFUSES the whole batch, because one cell needs more than `MAX_SUPPORT` and
    every other cell is refused with it.

    Per-cell bounds remove the width amplification. Each cell gets the support its own parameters
    demand, so a high-volume cell is expensive for itself alone. A cell that genuinely exceeds
    `MAX_SUPPORT` still refuses the call, but is now identified by index and exact parameters.

    The bound proved is identical to `support_size`'s -- the same mode-aware geometric argument,
    the same `tail_mass` -- computed by a vectorised binary search instead of a scalar one.
    """
    return _support_sizes(mu, k, tail_mass, zero_truncated=True)


def _support_sizes_negbin(mu: Tensor, k: Tensor, tail_mass: float = 1e-12) -> Tensor:
    """Per-cell PLAIN-NB support, proved against absolute rather than conditional tail mass."""
    return _support_sizes(mu, k, tail_mass, zero_truncated=False)


def _mu_over_sum(mu: Tensor, k: Tensor) -> Tensor:
    """`mu/(mu+k)` without overflowing the sum when both operands are finite."""
    return torch.sigmoid(torch.log(mu) - torch.log(k))


def _validate_size_parameters(mu: Tensor, k: Tensor) -> None:
    """Validate sampling parameters before any logarithm or gamma arithmetic."""
    if not isinstance(mu, Tensor) or not isinstance(k, Tensor):
        raise SampleError("mu and k must be torch tensors")
    if mu.shape != k.shape:
        raise SampleError(f"mu {tuple(mu.shape)} and k {tuple(k.shape)} must describe same cells")
    if mu.numel() == 0:
        raise SampleError("mu and k must describe at least one cell")
    if mu.device != k.device:
        raise SampleError("mu and k must be on the same device")
    if not mu.is_floating_point() or not k.is_floating_point():
        raise SampleError(f"mu and k must be floating tensors, got {mu.dtype} and {k.dtype}")
    if bool((~torch.isfinite(mu) | (mu <= 0)).any()):
        raise SampleError("mu must contain only finite, strictly positive values")
    if bool((~torch.isfinite(k) | (k <= 0)).any()):
        raise SampleError("k must contain only finite, strictly positive values")


def sample_paths(
    p: Tensor,
    mu: Tensor,
    k: Tensor,
    n_draws: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw `n_draws` integer paths for cells whose parameters are already factor-conditioned.

    Two stages, matching the hurdle exactly:

      1. `z ~ Bernoulli(p)`; a failed draw is a zero and stops there.
      2. otherwise invert the ZERO-TRUNCATED CDF -- never the untruncated one, which can return 0
         and would contradict the occurrence draw that just happened.

    Args:
        p: `[...]` occurrence probabilities, already conditioned on a drawn factor path.
        mu: `[...]` untruncated NB means, same shape.
        k: `[...]` dispersions, same shape.
        n_draws: Paths to draw. 1,000 is the architecture's figure.
        generator: For reproducibility. A gate result that cannot be reproduced is not one.

    Returns:
        Integer tensor `[n_draws, *p.shape]`.

    Raises:
        SampleError: shapes disagree, `n_draws` is below 1, or the support would exceed the cap.
    """
    if not all(isinstance(v, Tensor) for v in (p, mu, k)):
        raise SampleError("p, mu and k must be torch tensors")
    if not (p.shape == mu.shape == k.shape):
        raise SampleError(
            f"p {tuple(p.shape)}, mu {tuple(mu.shape)}, k {tuple(k.shape)} must describe the "
            "same cells; a broadcast here would silently pair a cell with another cell's size"
        )
    _validate_size_parameters(mu, k)
    if p.device != mu.device or not p.is_floating_point():
        raise SampleError("p must be a floating tensor on the same device as mu and k")
    if bool((~torch.isfinite(p) | (p < 0) | (p > 1)).any()):
        raise SampleError("p must contain only finite probabilities in [0,1]")
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
        raise SampleError(f"n_draws must be >= 1, got {n_draws}")

    # PER-CELL SUPPORT, NOT THE BATCH MAXIMUM. Using `support_size` as one CDF width makes the
    # largest required support in the batch the width that every cell pays for.
    # Measured here at fixed k=1.5: one cell moving mu=2 -> mu=100 took the shared support from 54
    # to 1,978 and
    # the coexisting float64 PMF+CDF tables from 0.05 GB to 1.77 GB. A cell past MAX_SUPPORT still
    # refuses the call, with its
    # index and parameters named by `support_sizes` rather than a misleading batch-wide maximum.
    # Each cell now gets the support its own parameters demand, rounded up to a power of two so
    # the number of distinct table shapes stays small. Rounding UP can only ADD support, so the
    # proved <= tail_mass omission still holds.
    # Exact p=0 cells can never enter the positive-size component. Validate their numerical
    # parameters above, but do not require an otherwise unreachable ZTNB support: the answer is
    # deterministically zero even when that conditional distribution would exceed the table cap.
    positive = p.reshape(-1) > 0
    tiers = torch.zeros(p.numel(), dtype=torch.long, device=p.device)
    if bool(positive.any()):
        flat_mu, flat_k = mu.reshape(-1), k.reshape(-1)
        sizes = _support_sizes(
            flat_mu[positive],
            flat_k[positive],
            1e-12,
            zero_truncated=True,
            source_flat_indices=torch.nonzero(positive, as_tuple=False).squeeze(-1),
        )
        positive_tiers = torch.pow(2, torch.ceil(torch.log2(sizes.double().clamp_min(1.0)))).long()
        tiers[positive] = positive_tiers.clamp(1, MAX_SUPPORT)

    tail = (*p.shape, n_draws)
    # Uniforms for EVERY cell, in the same order and shapes as before occurrence-first
    # subsetting, so that optimisation does not itself move the RNG stream.  This is deliberately
    # only a stream claim: the corrected per-cell support/normalisation may map a rare tail
    # uniform to a different count than the former shared-support implementation.
    u_occ = torch.rand(tail, device=p.device, dtype=p.dtype, generator=generator)
    u_size = torch.rand(tail, device=p.device, dtype=torch.float64, generator=generator)

    occurred = u_occ < p.unsqueeze(-1)  # [*cells, n_draws]
    flat_occurred = occurred.reshape(-1, n_draws)
    flat_u = u_size.reshape(-1, n_draws)
    work_mu = mu.to(torch.float64).reshape(-1)
    work_k = k.to(torch.float64).reshape(-1)
    size = torch.zeros_like(flat_u, dtype=torch.long)

    # Certify every positive-probability cell independently of sampled occurrence draws.
    # Apply occurrence only after constructing the conditional size distribution.
    needed = positive
    for tier in torch.unique(tiers[needed]).tolist():
        # TWO SOURCES OF ERROR, AND THE SLACK NEEDS BOTH. Accumulation over `tier` terms costs
        # about `C * tier * eps`. But each term is `exp(ztnb_log_pmf(...))`, whose lgamma and exp
        # carry several ulps RELATIVE TO THE TERM -- and when the distribution is concentrated on
        # a single count, one term is of size ~1 and its own error dominates entirely, no matter
        # how few terms there are. Measured: mu=0.03, k=316 (both ordinary -- k is well under the
        # head's exp(LOG_K_MAX)=8103) needs support 6, tiers to 8, and lands at 1 + 1.47e-13,
        # roughly 660 ulps, where a purely tier-scaled slack of 2.8e-14 called it a contract
        # violation and refused a distribution the pre-change sampler happily sampled.
        eps = torch.finfo(torch.float64).eps
        numerical_slack = max(PMF_TERM_SLACK * eps, 16.0 * tier * eps)
        rows = torch.nonzero(needed & (tiers == tier), as_tuple=False).squeeze(-1)
        counts = torch.arange(1, tier + 1, device=mu.device, dtype=torch.float64)
        # [rows, tier] -- built ONCE per tier and shared by every draw of those cells. An earlier
        # version expanded this to [n_draws * n_cells, n] before searchsorted, which at 1,000
        # paths over 85,372 M5 cells is 4.3 BILLION floats: the process was killed outright.
        pmf = torch.exp(
            ztnb_log_pmf(
                counts.expand(rows.numel(), tier),
                work_mu[rows].unsqueeze(-1),
                work_k[rows].unsqueeze(-1),
            )
        )
        cdf = pmf.cumsum(dim=-1)
        cdf_last = cdf[..., -1]
        if bool((~torch.isfinite(cdf_last) | (cdf_last <= 0)).any()):
            raise SampleError("the zero-truncated CDF is non-finite or empty for valid parameters")
        # ASYMMETRIC ON PURPOSE. The LOWER bound is the contract: a CDF materially short of 1
        # means a real tail was dropped, and that must refuse. An excess ABOVE 1 cannot indicate
        # a missing tail -- a probability sum exceeding 1 is arithmetically impossible -- so the
        # upper side is only a gross-failure guard and has no business being tight.
        if bool((cdf_last < 1.0 - (1e-12 + numerical_slack)).any()):
            raise SampleError(
                "the computed support omitted more mass than its proved 1e-12 bound; refusing to "
                "renormalise a material tail"
            )
        if bool((cdf_last > 1.0 + UPPER_SUM_GUARD).any()):
            raise SampleError(
                f"the truncated CDF summed to more than 1 + {UPPER_SUM_GUARD}, which is not "
                "round-off but broken arithmetic in the PMF evaluation"
            )
        # Condition on the retained support by scaling the query, not by blindly forcing the last
        # CDF value to one. The total-variation change is at most the proved 1e-12 omitted mass.
        query = flat_u[rows] * cdf_last.unsqueeze(-1)
        idx = torch.searchsorted(cdf.contiguous(), query.contiguous())
        if bool((idx >= tier).any()):
            raise SampleError("inverse CDF stepped past its proved support due to numerical error")
        size[rows] = idx + 1  # +1: index 0 is the count 1

    drawn = torch.where(flat_occurred, size, torch.zeros_like(size)).reshape(tail)
    # Draws are generated on the LAST axis for searchsorted's sake and returned on the FIRST,
    # because every caller iterates paths.
    return drawn.movedim(-1, 0)


def sample_paths_negbin(
    mu: Tensor,
    k: Tensor,
    n_draws: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw `n_draws` integer paths from the PLAIN Negative Binomial -- the resolved head for a
    panel whose zero rate does not justify a hurdle.

    ONE stage, matching `likelihood.ancestral_path_plain`: no occurrence draw, so the
    UNTRUNCATED CDF is the right one and zero is an ordinary outcome of it. Same float64 CDF,
    proved-support and scaled-query discipline as the hurdle sampler, because the tail-spike
    failure it guards against is a property of inverse-CDF sampling, not of the hurdle.

    Args:
        mu: `[...]` NB means, already factor-conditioned.
        k: `[...]` dispersions, same shape.
        n_draws: Paths to draw.
        generator: For reproducibility.

    Returns:
        Integer tensor `[n_draws, *mu.shape]`.

    Raises:
        SampleError: shapes disagree, `n_draws` is below 1, or the support exceeds the cap.
    """
    if not all(isinstance(v, Tensor) for v in (mu, k)):
        raise SampleError("mu and k must be torch tensors")
    if mu.shape != k.shape:
        raise SampleError(
            f"mu {tuple(mu.shape)} and k {tuple(k.shape)} must describe the same cells"
        )
    _validate_size_parameters(mu, k)
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
        raise SampleError(f"n_draws must be >= 1, got {n_draws}")

    work_mu, work_k = mu.to(torch.float64).reshape(-1), k.to(torch.float64).reshape(-1)
    # Support certification is deterministic and may refuse.  Do it before consuming the
    # caller's generator: a failed request must not shift every later seeded sample merely
    # because its parameters were known in advance to exceed the certified support cap.
    sizes = _support_sizes_negbin(mu, k)
    tiers = torch.pow(2, torch.ceil(torch.log2(sizes.double().clamp_min(1.0)))).long()
    tiers = tiers.clamp(1, MAX_SUPPORT).reshape(-1)
    tail = (*mu.shape, n_draws)
    u = torch.rand(tail, device=mu.device, dtype=torch.float64, generator=generator)
    flat_u = u.reshape(-1, n_draws)
    drawn = torch.empty_like(flat_u, dtype=torch.long)

    # The plain head needs the same per-cell tiers and measured arithmetic tolerance as the
    # hurdle head. The previous path used one batch-maximum support and a symmetric 64-eps test:
    # one busy cell enlarged every neighbour, and valid near-Poisson parameters were refused when
    # lgamma cancellation made their finite CDF exceed one by a few e-12.
    for tier in torch.unique(tiers).tolist():
        rows = torch.nonzero(tiers == tier, as_tuple=False).squeeze(-1)
        counts = torch.arange(0, tier + 1, device=mu.device, dtype=torch.float64)
        pmf = torch.exp(
            nb_log_pmf(
                counts.expand(rows.numel(), tier + 1),
                work_mu[rows].unsqueeze(-1),
                work_k[rows].unsqueeze(-1),
            )
        )
        cdf = pmf.cumsum(dim=-1)
        cdf_last = cdf[..., -1]
        eps = torch.finfo(torch.float64).eps
        numerical_slack = max(PMF_TERM_SLACK * eps, 16.0 * (tier + 1) * eps)
        if bool((~torch.isfinite(cdf_last) | (cdf_last <= 0)).any()):
            raise SampleError("the NB CDF is non-finite or empty for valid parameters")
        if bool((cdf_last < 1.0 - (1e-12 + numerical_slack)).any()):
            raise SampleError(
                "the computed plain-NB support omitted more mass than its proved 1e-12 bound; "
                "refusing to renormalise a material tail"
            )
        if bool((cdf_last > 1.0 + UPPER_SUM_GUARD).any()):
            raise SampleError(
                f"the truncated plain-NB CDF summed to more than 1 + {UPPER_SUM_GUARD}, which "
                "is not round-off but broken arithmetic in the PMF evaluation"
            )
        query = flat_u[rows] * cdf_last.unsqueeze(-1)
        idx = torch.searchsorted(cdf.contiguous(), query.contiguous())
        if bool((idx > tier).any()):
            raise SampleError("inverse CDF stepped past its proved support due to numerical error")
        drawn[rows] = idx  # index IS the count: support starts at zero

    return drawn.reshape(tail).movedim(-1, 0)


def lead_time_totals(paths: Tensor, lead_time: int, review_period: int) -> Tensor:
    """Cumulative demand over `lead_time + review_period` days, per path.

    The reason paths exist. An ordering decision is made against the distribution of demand over
    the cover period, and that distribution cannot be recovered from per-day marginal quantiles --
    summing the 95th percentile of each day gives a number far above the 95th percentile of the
    sum, by an amount that depends on the dependence between days. Here the dependence is already
    in the paths, so the sum is simply taken.

    Args:
        paths: `[n_draws, ..., horizon]` from `sample_paths`.
        lead_time: Days until an order arrives.
        review_period: Days until the next ordering opportunity.

    Returns:
        `[n_draws, ...]` totals over the first `lead_time + review_period` days.

    Raises:
        SampleError: the paths or controls are invalid, the cover exceeds the drawn horizon, or
            finite daily counts cannot be represented as cumulative totals.
    """
    if not isinstance(paths, Tensor):
        raise SampleError(f"paths must be a torch tensor, got {type(paths).__name__}")
    if paths.ndim < 2:
        raise SampleError("paths must have at least a path and final horizon dimension")
    if (
        type(lead_time) is not int
        or lead_time < 0
        or type(review_period) is not int
        or review_period < 0
    ):
        raise SampleError(
            f"lead_time and review_period must be built-in non-negative integers, got "
            f"{lead_time!r} and {review_period!r}"
        )
    if paths.dtype == torch.bool or paths.is_complex():
        raise SampleError(f"paths must contain integer-valued demand counts, got {paths.dtype}")
    try:
        finite = torch.isfinite(paths)
    except (RuntimeError, TypeError) as exc:
        raise SampleError("paths must contain numeric demand counts") from exc
    if not bool(finite.all()):
        raise SampleError("paths must contain only finite demand counts")
    unsigned_dtypes = (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    if paths.dtype not in unsigned_dtypes and bool((paths < 0).any()):
        raise SampleError("paths must contain only non-negative demand counts")
    if paths.is_floating_point() and bool((paths != torch.floor(paths)).any()):
        raise SampleError("paths must contain integer-valued demand counts")

    cover = lead_time + review_period
    if cover < 1:
        raise SampleError(f"cover period must be >= 1, got {lead_time} + {review_period}")
    if cover > paths.shape[-1]:
        raise SampleError(
            f"cover period {cover} exceeds the {paths.shape[-1]}-day horizon drawn. Extrapolating "
            "would invent dependence past the last day the factor was drawn for"
        )
    covered_paths = paths[..., :cover]
    if not paths.is_floating_point():
        audit_totals = covered_paths.to(torch.float64).sum(dim=-1)
        if bool((audit_totals > torch.iinfo(torch.int64).max).any()):
            raise SampleError("non-negative daily paths overflowed while forming lead-time totals")
    totals = covered_paths.sum(dim=-1)
    if totals.is_floating_point() and not bool(torch.isfinite(totals).all()):
        raise SampleError("finite daily paths overflowed while forming lead-time totals")
    if totals.dtype not in unsigned_dtypes and bool((totals < 0).any()):
        raise SampleError("non-negative daily paths overflowed while forming lead-time totals")
    return totals
