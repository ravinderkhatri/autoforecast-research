"""Static variable selection: which attributes are worth listening to, per series.

WHAT IT IS FOR. The identity branch receives every static attribute with equal standing --
department, category, state, store, all concatenated and projected. An attribute that carries
nothing for a particular series still occupies width in that projection and still contributes
gradient, and there is no way for the model to say "store does not matter for this one".

`ForecastModelSpec.static_variable_selection` asks for the mechanism that lets it. This is the
TFT construction, simplified to what a one-hot attribute block actually needs: a small network
reads all the attributes together and emits a SOFTMAX weight per attribute group, which then
scales that group before projection.

**SOFTMAX AND NOT A SIGMOID, and the difference is the point.** A sigmoid gate per attribute can
turn everything on at once, which is what the model already does implicitly. A softmax forces the
weights to compete: raising one lowers another, so the network has to decide what matters rather
than accumulate. That is what makes the weights READABLE -- `selection_weights` is the only place
in this architecture that answers "which attribute is this forecast leaning on", and a
non-competing gate would answer "all of them, somewhat".

**GROUPS, NOT COLUMNS.** `dept_id` arrives as seven one-hot columns; weighting each column
separately would let the model prefer some departments to others, which is not variable selection
-- it is a bias term in disguise, and the projection can already express it. The weight applies to
the whole group, so what is being selected is the ATTRIBUTE.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["SelectionError", "StaticVariableSelection"]


class SelectionError(ValueError):
    """Variable selection was configured in a way that cannot describe the attributes given."""


class StaticVariableSelection(nn.Module):
    """Per-group softmax weights over static attributes.

    Attributes:
        widths: Columns per attribute group, in the order they are concatenated.
        n_groups: How many attributes are being selected among.
    """

    def __init__(self, widths: tuple[int, ...], d_hidden: int = 32) -> None:
        """Args:
            widths: Columns belonging to each attribute group. For M5's
                `(dept_id, cat_id, state_id, store_id)` one-hot blocks that is `(7, 3, 3, 10)`.
            d_hidden: Width of the network that reads all attributes before deciding.

        Raises:
            SelectionError: no groups, or a non-positive width. A group of zero columns would
                receive a weight that scales nothing, and the softmax would still spend mass on it
                -- taking that mass from the groups that do carry something.
        """
        super().__init__()
        if not widths:
            raise SelectionError("no attribute groups to select among")
        if any(w < 1 for w in widths):
            raise SelectionError(
                f"every group needs at least one column, got {widths}. A zero-width group still "
                "takes softmax mass from the groups that carry something"
            )
        self.widths = tuple(widths)
        self.n_groups = len(widths)
        total = sum(widths)
        # The selector sees ALL attributes before deciding about any one of them, which is what
        # makes it selection rather than an independent gate per group.
        self.score = nn.Sequential(
            nn.Linear(total, d_hidden), nn.GELU(), nn.Linear(d_hidden, self.n_groups)
        )
        # NEARLY uniform at the start, so an untrained model behaves as the unselected one did
        # and any departure is something that was learned.
        #
        # Small weights, NOT zero, and the difference is one this project has already paid for
        # twice. A zero final weight sends zero gradient to everything upstream of it: the hidden
        # layer would be frozen until the output layer happened to move, which is the same trap
        # the hurdle head hit (its comment records the measurement) and the same one the event
        # kernel hits by design. std=1e-3 over a 32-wide hidden layer gives logits around 0.006,
        # so the softmax starts within a fraction of a percent of uniform while every parameter
        # can still learn from the first step.
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)

    def selection_weights(self, attributes: Tensor) -> Tensor:
        """`[batch, n_groups]` softmax weights, one per attribute group.

        The interpretable output. Weights near uniform mean the model did not find one attribute
        more useful than another, which is a finding about the panel rather than a failure.
        """
        return torch.softmax(self.score(attributes), dim=-1)

    def forward(self, attributes: Tensor) -> tuple[Tensor, Tensor]:
        """Scale each attribute group by its selected weight.

        Args:
            attributes: `[batch, sum(widths)]`, groups concatenated in `widths` order.

        Returns:
            `(weighted, weights)` -- the scaled attributes, and `[batch, n_groups]` for reading.

        Raises:
            SelectionError: the width does not match what was configured, which would silently
                apply a group's weight to the wrong columns.
        """
        total = sum(self.widths)
        if attributes.shape[-1] != total:
            raise SelectionError(
                f"expected {total} attribute columns ({self.widths}), got "
                f"{attributes.shape[-1]}; a mismatch would scale the wrong columns"
            )
        weights = self.selection_weights(attributes)
        # Repeat each group's weight across its own columns, so the whole attribute moves together.
        per_column = torch.repeat_interleave(
            weights, torch.tensor(self.widths, device=weights.device), dim=-1
        )
        # Multiplied by n_groups so a uniform selection is the IDENTITY. Without it, softmax over
        # four groups would scale every attribute by 0.25 and the change would look like a
        # learning-rate change on the identity branch rather than like selection.
        return attributes * per_column * self.n_groups, weights
