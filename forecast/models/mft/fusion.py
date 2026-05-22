"""Gated fusion of the five branch tokens, then attention over them as a group.

TWO STAGES, AND THE ORDER MATTERS.

**The AoA gate first.** Attention-on-Attention: each branch's token is passed through a
sigmoid gate computed from the token itself and the others. Plain attention over five tokens
cannot switch a branch OFF -- softmax weights sum to one, so a branch with nothing to say still
receives a share of the total. A multiplicative gate can go to zero. That is the difference
between "price is uninformative for this client" and "price gets a fifth of the weight anyway".

**Group attention second.** Once the gates have decided how much each family should contribute,
attention decides how the families relate. Doing it the other way round would ask attention to
relate families whose relevance had not been established.

A BRANCH THE CLIENT DOES NOT HAVE IS MASKED, NOT ZEROED. A zero token is a point in embedding
space and the model will learn what it means; an absent branch has to be absent from the softmax,
or four branches of evidence get diluted by a fifth that carries none.
"""

from __future__ import annotations

import torch
from forecast.models.mft.branches import _binary_mask
from torch import Tensor, nn

__all__ = ["FusionError", "GatedFusion"]


class FusionError(ValueError):
    """Fusion was given a branch set it cannot combine."""


class GatedFusion(nn.Module):
    """AoA gate over branch tokens, then multi-head attention across them.

    Attributes:
        n_branches: How many branch tokens arrive. Five in the full design; a client missing a
            family passes a mask rather than a different number, so one trained model serves both.
    """

    def __init__(
        self, n_branches: int, d_model: int, *, n_heads: int = 4, d_static: int = 0
    ) -> None:
        super().__init__()
        if n_branches < 1:
            raise FusionError(f"n_branches must be >= 1, got {n_branches}")
        self.n_branches = n_branches
        self.branch_id = nn.Parameter(torch.randn(1, n_branches, d_model) * 0.02)
        # The gate sees a branch's own token AND the pooled context, so "price is uninformative"
        # can be decided relative to what the other branches are carrying rather than in isolation.
        # STATIC ATTRIBUTES REACH THE GATE, which is the third leg of
        # `ForecastModelSpec.static_conditioning` alongside the trunk's FiLM and the head's
        # dispersion. Without it the gate sees only the branch tokens and their pooled context,
        # so it can notice that a price token looks uninformative for this CELL but never that
        # price is uninformative for this KIND of product -- which is the durable version of the
        # same judgement and the one identity is there to supply.
        self.d_static = d_static
        self.gate = nn.Sequential(nn.Linear(2 * d_model + d_static, d_model), nn.Sigmoid())
        self.value = nn.Linear(2 * d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # `ForecastModelSpec.norm_before_gate`. The branches arrive at whatever magnitude their
        # own encoders produce -- a demand token and a one-hot-driven identity token can differ
        # by an order of magnitude -- and a sigmoid gate fed unnormalised inputs spends its range
        # on the loudest branch. Normalising BEFORE the gate is what makes "how much should this
        # branch contribute" a question about content rather than about scale. The audit
        # previously pointed at `self.norm` below as satisfying this field; that norm sits after
        # the attention and never touches what the gate sees.
        self.pre = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: Tensor, branch_mask: Tensor | None = None, static: Tensor | None = None
    ) -> Tensor:
        """Args:
            tokens: `[batch, n_branches, d_model]`, one per branch.
            static: Optional `[batch, d_static]` attributes, required when `d_static` was set.
                The gate is conditioned on them, so "price is uninformative for this kind of
                product" is expressible rather than only "for this cell".
            branch_mask: Optional `[batch, n_branches]`, True where the branch exists for this
                client. A masked branch is excluded from the pooled context AND from the
                attention keys, so it contributes nothing rather than contributing a zero.

        Returns:
            `[batch, n_branches, d_model]` -- gated, related, and still per-branch, because the
            trunk wants the sequence rather than a single pooled vector.

        Raises:
            FusionError: the branch count is wrong, or a row masks out every branch.
        """
        if tokens.dim() != 3 or tokens.shape[1] != self.n_branches:
            raise FusionError(
                f"expected [batch, {self.n_branches}, d_model], got {tuple(tokens.shape)}"
            )
        x = self.pre(tokens + self.branch_id)
        if branch_mask is None:
            context = x.mean(dim=1, keepdim=True)
        else:
            branch_mask = _binary_mask(
                branch_mask, tokens[..., 0], "fusion branch mask", error_type=FusionError
            )
            if bool((~branch_mask).all(dim=-1).any()):
                raise FusionError(
                    "a row masks out every branch; there is nothing to fuse and attention over an "
                    "empty key set returns NaN. A client with no usable family needs to be "
                    "refused at the data contract, not here"
                )
            m = branch_mask.unsqueeze(-1).to(x.dtype)
            context = (x * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        paired = torch.cat([x, context.expand_as(x)], dim=-1)
        # The GATE sees the static attributes; `value` does not, and the asymmetry is the design.
        # The gate decides how much a branch contributes and identity is evidence about that; the
        # value is the branch's own content and mixing identity into it would let the trunk read
        # identity twice, once here and once through its own FiLM conditioning.
        gate_input = paired
        if self.d_static:
            if static is None:
                raise FusionError(
                    "this gate was built to condition on static attributes but none were passed; "
                    "defaulting to zeros would silently make it the unconditioned gate again"
                )
            gate_input = torch.cat(
                [paired, static.unsqueeze(1).expand(-1, paired.shape[1], -1)], dim=-1
            )
        gated = self.gate(gate_input) * self.value(paired)
        attended, _ = self.attn(
            gated,
            gated,
            gated,
            key_padding_mask=None if branch_mask is None else ~branch_mask,
            need_weights=False,
        )
        out = self.norm(gated + attended)
        if branch_mask is not None:
            out = out * branch_mask.unsqueeze(-1).to(out.dtype)
        return out
