"""The shared latent factor: what makes stores move together without a chain total.

Implements parts 3 and 4 of [`spec/likelihood.py`](../../spec/likelihood.py) -- the factor-to-head
link and the configured AR(1) or IID dynamics -- and `test_factor.py` holds it to them.

WHY THIS EXISTS AT ALL. Top-down (forecast the chain total, allocate down by learned shares) was
considered and rejected: the chain total is NOT OBSERVABLE when any store is censored, so summing
recorded sales gives a lower bound rather than a target, and you would be fixing a censoring
problem with a censored target. This is the replacement. Every store keeps its own exposure,
censoring, covariates and baselines, and the dependence between them comes from a rank-8 factor
they all load on.

TWO INDEX SETS, AND CONFLATING THEM IS THE BUG THIS MODULE EXISTS TO AVOID:

  · the FACTOR is drawn per `(factor_group, product, horizon_step)`. `factor_group` is whatever
    `ForecastModelSpec.factor_group_col` names -- chain, region and banner induce different
    dependence, and the spec refuses to infer it because an engineer cannot either.
  · the LOADINGS are per STORE, and there are two of them, because a store may be more sensitive
    in whether it sells at all than in how much.

So every store in a group shares one drawn path and moves with the others exactly to the extent
their loadings agree. That is the whole mechanism.

THE FACTOR NEVER TOUCHES THE DISPERSION -- see `heads.py`. Letting a latent daily term widen `k`
would let the model explain a bad day by widening the distribution instead of shifting its mean,
which is how intervals end up calibrated on average and useless during promotions.

WHAT THIS IS, IN THE LITERATURE'S TERMS. This module cited nothing until 2026-08-25, which reads
as ignorance of a large and relevant field rather than as the independent derivation it was. The
honest placement:

    A custom stochastic shared dynamic-factor model with nonlinear hurdle-NB emissions, related
    to global/local forecasting models and to low-rank multivariate forecasting, but NOT an
    implementation of Deep Factors or GPVar.

The distinction is not pedantry, because it changes what borrowing from either would cost:

  * **Deep Factors** (Wang, Smola, Maddix, Gasthaus, Foster, Januschowski, ICML 2019,
    https://proceedings.mlr.press/v97/wang19k.html) makes its global factors DETERMINISTIC --
    explicitly, to avoid costly global inference -- and puts the stochasticity in LOCAL
    per-series random effects. This design does the opposite: the shared factor is stochastic
    and its realisation is common across stores, which is precisely the mechanism that produces
    systemic AGGREGATE uncertainty. Adopting the paper's structure would weaken the thing this
    module exists to supply.
  * **GPVar / low-rank Gaussian copula** (Salinas, Bohlke-Schneider, Callot, Medico, Gasthaus,
    NeurIPS 2019) is closer in spirit -- low-rank shared Gaussian dependence plus idiosyncratic
    variation -- but it models dependence on a transformed Gaussian SCALE. Its likelihood uses
    an invertible `Phi^-1 o F_hat` marginal transform and differentiates a linearly interpolated
    empirical CDF. That recipe is not a drop-in exact likelihood for these discrete hurdle-NB
    observations, their atom at zero, or right-censored stockout days; using it would require a
    different observation and censoring model. Its advantage is also in TRAINING; prediction
    there still scales linearly in Monte Carlo samples.

THE EXACT COHERENCE GRAPH, because "coherent sample paths" is easy to over-read. The factor is
drawn per `(factor_group, product, step)`, so what it shares is:

  * ACROSS STORES, within one `(factor_group, product)` block -- yes, by construction;
  * ACROSS DAYS -- through the AR(1) when configured; no in the explicit IID ablation;
  * ACROSS PRODUCTS -- **no**. Separate products draw independent factor paths.

So a chain total taken across many products can still understate a common residual shock --
unmodelled weather, footfall, a macro event -- because nothing in this module correlates the
residuals of one product with another's. Known covariates may move several products together
through the conditional mean, but that is not the same thing. A cross-product factor is a
separate modelling addition; until residual evidence justifies one, aggregate claims should be
scoped to the dependence this module actually supplies.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["MAX_PHI", "FactorError", "SharedLatentFactor"]

MAX_PHI = 1.0 - 1e-6
"""Largest persistence the parameterisation can reach.

`tanh` saturates to exactly 1.0 in float64 above |theta| ~ 19, so `phi = tanh(theta)` does NOT
guarantee stationarity -- it only postpones the failure until an optimiser pushes persistence far
enough. At phi = 1 the innovation scale `sqrt(1 - phi^2)` is exactly zero: the factor freezes at
its first draw and every horizon day becomes identical.

1e-6 of headroom leaves an innovation scale of ~0.0014, which is small but not degenerate.
"""


class FactorError(ValueError):
    """The factor was asked for something its parameterisation does not define."""


class SharedLatentFactor(nn.Module):
    """A rank-`r` dynamic factor per group, with per-store loadings into each head.

    Attributes:
        rank: Factor dimension. 8 by default (`ForecastModelSpec.factor_rank`); rank 0 would
            reduce this to plain bottom-up with independent marginals, which is the thing it
            exists to improve on, so it is refused rather than silently supported.
        n_stores: How many distinct loading rows to carry.
    """

    def __init__(
        self,
        rank: int,
        n_stores: int,
        *,
        init_phi: float = 0.7,
        factor_dynamics: Literal["ar1", "iid"] = "ar1",
    ) -> None:
        """Args:
            rank: Factor dimension, >= 1.
            n_stores: Number of stores (loading rows), >= 1.
            init_phi: Starting persistence. 0.7 rather than 0, because a factor initialised with
                no persistence has to learn that consecutive days are related before it can learn
                anything else, and the lead-time sums that motivate the factor need exactly that.
                Ignored when `factor_dynamics="iid"`.
            factor_dynamics: `"ar1"` for a stationary autoregressive path, or `"iid"` for an
                independent standard-normal draw at every horizon step. Only AR(1) owns a learned
                persistence parameter; their checkpoint state keys are intentionally distinct.

        Raises:
            FactorError: rank or n_stores is not a positive integer, `init_phi` is outside
                (-MAX_PHI, MAX_PHI), or the factor-dynamics value is unknown.
        """
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise FactorError(
                f"factor rank must be a positive integer, got {rank!r}. Rank 0 is plain "
                "bottom-up with "
                "independent marginals -- which is the topology this factor replaces, so asking "
                "for it here is almost certainly a mistake rather than a configuration"
            )
        if not isinstance(n_stores, int) or isinstance(n_stores, bool) or n_stores < 1:
            raise FactorError(f"n_stores must be a positive integer, got {n_stores!r}")
        if type(init_phi) not in (int, float) or not -MAX_PHI < init_phi < MAX_PHI:
            raise FactorError(
                f"init_phi must lie strictly inside (-{MAX_PHI}, {MAX_PHI}), got {init_phi!r}"
            )
        if factor_dynamics not in ("ar1", "iid"):
            raise FactorError(f"factor_dynamics must be 'ar1' or 'iid', got {factor_dynamics!r}")
        self.rank = rank
        self.n_stores = n_stores
        self.factor_dynamics = factor_dynamics
        # phi = MAX_PHI * tanh(theta), and the scale factor is not decoration.
        #
        # `tanh` alone does NOT keep |phi| below 1: in float64 it SATURATES to exactly 1.0 above
        # |theta| ~ 19, and an optimiser that pushes persistence up will get there. The test that
        # sets theta to 50 found this immediately -- phi came back as exactly 1.0, which is a
        # random walk with no stationary marginal, unbounded variance over the horizon, and an
        # innovation scale of exactly zero. Multiplying by (1 - 1e-6) makes the bound hold at
        # saturation rather than up to it.
        if factor_dynamics == "ar1":
            # Preserve the historical native-Torch bits whenever its inverse is finite. A value
            # just inside MAX_PHI can round to +/-1 in float32, so only that saturated result uses
            # the double-precision fallback. The default and boundary paths are held separately
            # by test_default_phi_initializer_preserves_the_historical_parameter_bits and
            # test_init_phi_is_finite_and_strictly_inside_the_reachable_range.
            native_theta = torch.atanh(torch.tensor(init_phi) / MAX_PHI)
            initial_theta = (
                float(native_theta)
                if bool(torch.isfinite(native_theta))
                else math.atanh(float(init_phi) / MAX_PHI)
            )
            self.theta = nn.Parameter(torch.full((rank,), initial_theta))
        self.loading_occurrence = nn.Embedding(n_stores, rank)
        self.loading_size = nn.Embedding(n_stores, rank)
        # Small, not zero: identical loadings would make every store move identically and the
        # gradient could not separate them. Small keeps the factor's initial contribution near
        # nothing, so the baselines still dominate at step one.
        nn.init.normal_(self.loading_occurrence.weight, std=0.01)
        nn.init.normal_(self.loading_size.weight, std=0.01)

    @property
    def phi(self) -> Tensor:
        """Persistence strictly inside (-1, 1), or exact zero for configured IID dynamics."""
        if self.factor_dynamics == "iid":
            return self.loading_occurrence.weight.new_zeros(self.rank)
        return MAX_PHI * torch.tanh(self.theta)

    @property
    def innovation_scale(self) -> Tensor:
        """`sqrt(1 - phi^2)` -- the scale that holds the marginal variance at exactly 1.

        **Not 1.** With unit innovations the variance grows as `1 + phi^2 + phi^4 + ...`, so later
        horizon days get a wider latent term than earlier ones for no modelled reason: the
        forecast fans out with lead time BY CONSTRUCTION rather than because uncertainty grows,
        and the loadings absorb a scale drifting under them. A stationary marginal also makes the
        scale identifiable -- otherwise `phi` and the loading magnitudes trade off and neither is
        determined. IID is the exact phi=0 boundary, where this scale is exactly 1.
        """
        if self.factor_dynamics == "iid":
            return self.loading_occurrence.weight.new_ones(self.rank)
        return torch.sqrt(1.0 - self.phi * self.phi)

    def draw(
        self,
        n_groups: int,
        horizon: int,
        n_draws: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw factor paths. Returns `[n_draws, n_groups, horizon, rank]`.

        AR(1): `f_0 ~ N(0, I)`, then
        `f_t = phi * f_{t-1} + sqrt(1 - phi^2) * eps_t`. IID returns the independent standard
        normals directly, with exact zero persistence.

        Reparameterised: the randomness is a plain normal and everything else is differentiable,
        so gradients reach `theta` through the path. Sampling `phi` inside a loop that detached
        would train the loadings against a factor whose persistence never learned anything.

        Args:
            n_groups: Distinct `(factor_group, product)` combinations in the batch.
            horizon: Steps to draw.
            n_draws: Independent paths. The training objective is an importance-weighted bound
                over these (`likelihood.iwae_bound`), so more than one is the normal case.
            generator: For reproducible draws. A gate result that cannot be reproduced is not one.

        Raises:
            FactorError: any of the sizes is below 1.
        """
        sizes = (n_groups, horizon, n_draws)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in sizes
        ):
            raise FactorError(
                f"n_groups={n_groups!r}, horizon={horizon!r}, n_draws={n_draws!r}: "
                "all must be positive integers"
            )
        device = self.loading_occurrence.weight.device
        eps = torch.randn(
            n_draws,
            n_groups,
            horizon,
            self.rank,
            device=device,
            dtype=self.loading_occurrence.weight.dtype,
            generator=generator,
        )
        if self.factor_dynamics == "iid":
            # Returning the generator's normals directly is stronger than running the AR loop
            # with phi=0: it makes IID mode exact, without even a multiply/add that could obscure
            # what distribution is being sampled.
            return eps
        phi, scale = self.phi, self.innovation_scale
        # Step 0 is the stationary marginal itself, so the path starts at variance 1 rather than
        # climbing to it over the first few steps.
        out = [eps[:, :, 0, :]]
        for step in range(1, horizon):
            out.append(phi * out[-1] + scale * eps[:, :, step, :])
        return torch.stack(out, dim=2)

    def contract(
        self,
        path: Tensor,
        group_index: Tensor,
        store_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Contract a drawn path with each store's loadings. Returns `(occurrence, size)` terms.

        This is where the two index sets meet, and it is the only place they should. `path` is
        indexed by group; `group_index` says which group each cell belongs to; `store_index` says
        whose loadings to use. Getting either wrong gives a model that trains perfectly well and
        has the wrong dependence structure.

        Args:
            path: `[n_draws, n_groups, horizon, rank]` from `draw`.
            group_index: `[n_cells]` group of each cell.
            store_index: `[n_cells]` store of each cell.

        Returns:
            Two tensors of `[n_draws, n_cells, horizon]`, ready to add to the heads' linear
            predictors -- which is exactly what `heads.HurdleNegBinHead.forward` expects.

        Raises:
            FactorError: the two index tensors differ in length, or an index is out of range.
        """
        if (
            not isinstance(path, Tensor)
            or path.ndim != 4
            or not path.is_floating_point()
            or any(size < 1 for size in path.shape[:3])
        ):
            raise FactorError(
                "path must be a nonempty floating draw path with shape "
                "[n_draws, n_groups, horizon, factor rank]"
            )
        if path.shape[-1] != self.rank:
            raise FactorError(
                f"draw path factor rank {path.shape[-1]} disagrees with model rank {self.rank}"
            )
        if not isinstance(group_index, Tensor) or not isinstance(store_index, Tensor):
            raise FactorError("group_index and store_index must be integer tensors")
        if group_index.ndim != 1 or store_index.ndim != 1:
            raise FactorError("group_index and store_index must be rank-1 tensors")
        if group_index.numel() == 0 or store_index.numel() == 0:
            raise FactorError("group_index and store_index must describe at least one cell")
        if group_index.dtype not in (torch.int32, torch.int64) or store_index.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise FactorError("group_index and store_index must be integer tensors")
        if group_index.shape != store_index.shape:
            raise FactorError(
                f"group_index {tuple(group_index.shape)} and store_index "
                f"{tuple(store_index.shape)} must describe the same cells"
            )
        if int(group_index.min()) < 0 or int(store_index.min()) < 0:
            raise FactorError("group and store indices must be non-negative")
        if int(group_index.max()) >= path.shape[1]:
            raise FactorError(
                f"group index {int(group_index.max())} but only {path.shape[1]} groups were drawn"
            )
        if int(store_index.max()) >= self.n_stores:
            raise FactorError(
                f"store index {int(store_index.max())} but only {self.n_stores} loading rows"
            )
        selected = path[:, group_index, :, :]  # [draws, cells, horizon, rank]
        occ = self.loading_occurrence(store_index)  # [cells, rank]
        size = self.loading_size(store_index)
        return (
            (selected * occ.unsqueeze(0).unsqueeze(2)).sum(-1),
            (selected * size.unsqueeze(0).unsqueeze(2)).sum(-1),
        )
