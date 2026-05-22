"""Alternating attention over branches and time, with RMSNorm and static FiLM conditioning. Static
attributes shift and scale trunk features without occupying a sequence position.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["FiLM", "RMSNorm", "Trunk", "TrunkError"]


class TrunkError(ValueError):
    """The trunk was given a shape it cannot process."""


class RMSNorm(nn.Module):
    """Root-mean-square normalisation: scale only, no mean subtraction."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class FiLM(nn.Module):
    """Feature-wise linear modulation from static attributes: `gamma * x + beta`.

    `gamma` is produced as `1 + delta` with `delta` initialised at zero, so an untrained FiLM is
    the identity. Initialising `gamma` at zero instead would silently switch the trunk off for the
    first epochs, and the loss would still fall -- because the baselines in the head carry the
    prediction on their own.
    """

    def __init__(self, d_static: int, d_model: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(d_static, d_model)
        self.to_beta = nn.Linear(d_static, d_model)
        for layer in (self.to_gamma, self.to_beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor, static: Tensor) -> Tensor:
        gamma = 1.0 + self.to_gamma(static).unsqueeze(1)
        beta = self.to_beta(static).unsqueeze(1)
        return gamma * x + beta


class _Stage(nn.Module):
    """One attention stage plus a feed-forward block, pre-normed."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Linear(ff_mult * d_model, d_model)
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        attended, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ff(self.norm2(x))


class Trunk(nn.Module):
    """Two alternating [group, time] stages with FiLM conditioning between them.

    Input is `[batch, n_branches, time, d_model]`: the branch tokens carried across the encoder
    positions. A GROUP stage attends over `n_branches` at each time step; a TIME stage attends
    over positions within each branch. Reshaping between them is the whole implementation, and
    getting the transpose wrong produces a model that trains and relates the wrong axes -- which
    is why the shapes are asserted rather than assumed.
    """

    def __init__(self, d_model: int, d_static: int, *, n_stages: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.n_stages = n_stages
        self.group_stages = nn.ModuleList(_Stage(d_model, n_heads) for _ in range(n_stages))
        self.time_stages = nn.ModuleList(_Stage(d_model, n_heads) for _ in range(n_stages))
        self.film = nn.ModuleList(FiLM(d_static, d_model) for _ in range(n_stages))
        self.out = RMSNorm(d_model)

    def forward(
        self,
        x: Tensor,
        static: Tensor,
        branch_mask: Tensor | None = None,
        time_mask: Tensor | None = None,
    ) -> Tensor:
        """Args:
            x: `[batch, n_branches, time, d_model]`.
            static: `[batch, d_static]` attributes for FiLM.
            branch_mask: Optional `[batch, n_branches]`.
            time_mask: Optional `[batch, time]`, True only for real calendar positions.

        Returns:
            `[batch, n_branches, time, d_model]`.

        Raises:
            TrunkError: `x` is not four-dimensional.
        """
        if x.dim() != 4:
            raise TrunkError(
                f"expected [batch, n_branches, time, d_model], got {tuple(x.shape)}. The group "
                "and time axes are transposed between stages, so a wrong rank here silently "
                "relates the wrong things"
            )
        b, n, t, d = x.shape
        if time_mask is not None:
            if (
                not isinstance(time_mask, Tensor)
                or time_mask.shape != (b, t)
                or time_mask.device != x.device
            ):
                raise TrunkError("time_mask must match [batch, time] on the input device")
            if time_mask.dtype != torch.bool:
                if time_mask.is_complex() or not bool(torch.isfinite(time_mask).all()) or bool(
                    ((time_mask != 0) & (time_mask != 1)).any()
                ):
                    raise TrunkError("time_mask must contain exact binary values")
                time_mask = time_mask.bool()
            if bool((~time_mask).all(dim=1).any()):
                raise TrunkError("each row needs at least one available time position")
            x = torch.where(time_mask[:, None, :, None], x, torch.zeros_like(x))
        for i in range(self.n_stages):
            # GROUP: attend across branches, independently at each time step.
            g = x.permute(0, 2, 1, 3).reshape(b * t, n, d)
            gm = None
            if branch_mask is not None:
                gm = (~branch_mask).unsqueeze(1).expand(b, t, n).reshape(b * t, n)
            g = self.group_stages[i](g, key_padding_mask=gm)
            x = g.reshape(b, t, n, d).permute(0, 2, 1, 3)
            # TIME: attend across positions, independently within each branch.
            s = x.reshape(b * n, t, d)
            tm = (
                None
                if time_mask is None
                else (~time_mask).unsqueeze(1).expand(b, n, t).reshape(b * n, t)
            )
            s = self.time_stages[i](s, key_padding_mask=tm)
            x = s.reshape(b, n, t, d)
            if time_mask is not None:
                x = torch.where(time_mask[:, None, :, None], x, torch.zeros_like(x))
            # FiLM after both, so static attributes modulate a representation that has already
            # seen its own family and its own history.
            x = self.film[i](x.reshape(b, n * t, d), static).reshape(b, n, t, d)
            if time_mask is not None:
                x = torch.where(time_mask[:, None, :, None], x, torch.zeros_like(x))
        return self.out(x)
