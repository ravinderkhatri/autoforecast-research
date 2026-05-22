"""Per-series annual amplitude: how much of the shared annual shape THIS series actually shows.

WHAT THIS IS FOR. The annual Fourier basis in the known-future tensor gives every series the same
coordinate -- where in the year each horizon day falls. Pooled harmonic weights then give every
series the same annual RESPONSE, which is wrong in an obvious way: a barbecue line and a school
supply line both have strong annual seasonality, in opposite months, and one shared curve fits
neither.

WHY IT IS SHRUNK AND NOT FREE, which is the whole design. A series in a five-year panel has seen
about FIVE Januaries. Free per-series harmonic weights would fit six coefficients to five annual
observations, and the fit would be exact and meaningless. So the weight is a shared shape plus a
per-series DEVIATION scaled by a small constant, and the deviation starts at zero:

    weight[series] = shared + shrink * deviation[series]

This is the same construction as `branches.CalendarEventBranch.kernel_weights`, deliberately.
That module makes an event seen twice estimable by borrowing the shape of every other event; this
one makes a series' annual response estimable by borrowing the shape of every other series. The
argument is identical and so is the machinery -- a second, differently-shaped solution to the same
problem would be one more thing to keep consistent.

WHERE IT ENTERS. Additively on the occurrence and size linear predictors, exactly where
`factor.SharedLatentFactor.contract` puts its terms. Seasonality is a modulation of the RATE, not
another view of the cell, so it does not belong in the branch stack -- a branch token would make
the trunk attend to it against price and identity, which is a different claim about what it is.

TWO SETS OF WEIGHTS, not one. Occurrence and size are separate questions: a product can sell on
more DAYS in December without selling more per transaction, and a single seasonal coefficient
forces those to move together. The hurdle exists precisely to keep them apart.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["SeasonalAmplitudeError", "SeasonalAmplitude"]


class SeasonalAmplitudeError(ValueError):
    """Seasonal amplitude was configured in a way that cannot be estimated."""


class SeasonalAmplitude(nn.Module):
    """A shared annual harmonic weight plus a shrunk per-series deviation.

    Attributes:
        n_terms: Width of the annual basis, `2 * harmonics`.
        shrink: Scale on the per-series deviation. Small on purpose -- see the module docstring.
    """

    def __init__(self, n_terms: int, n_series: int, *, shrink: float = 0.1) -> None:
        """Args:
            n_terms: Columns in the annual Fourier basis. Must be even -- the basis is sine and
                cosine PAIRS, and an odd width means a harmonic lost half of itself somewhere.
            n_series: Rows in the per-series deviation table.
            shrink: Scale on the deviation. At 0 the model is pooled and every series shares one
                annual response; at 1 the deviation is unconstrained and fits five observations
                with six coefficients.

        Raises:
            SeasonalAmplitudeError: a non-even or non-positive basis, no series, or a shrink
                outside [0, 1].
        """
        super().__init__()
        if n_terms < 2 or n_terms % 2:
            raise SeasonalAmplitudeError(
                f"n_terms must be a positive even number (sine/cosine pairs), got {n_terms}"
            )
        if n_series < 1:
            raise SeasonalAmplitudeError(f"n_series must be >= 1, got {n_series}")
        if not 0.0 <= shrink <= 1.0:
            raise SeasonalAmplitudeError(
                f"shrink must lie in [0,1], got {shrink}. Above 1 the per-series deviation is "
                "amplified rather than shrunk, which is the opposite of what makes five annual "
                "observations estimable"
            )
        self.n_terms = n_terms
        self.shrink = shrink
        # Zero-initialised, so the model starts POOLED and every series shares one annual shape.
        # A run that ends with the deviations still near zero is reporting that pooled seasonality
        # was sufficient on this panel, which is a finding rather than a failure to train.
        self.shared_occurrence = nn.Parameter(torch.zeros(n_terms))
        self.shared_size = nn.Parameter(torch.zeros(n_terms))
        self.deviation_occurrence = nn.Embedding(n_series, n_terms)
        self.deviation_size = nn.Embedding(n_series, n_terms)
        nn.init.zeros_(self.deviation_occurrence.weight)
        nn.init.zeros_(self.deviation_size.weight)

    def weights(self, series_index: Tensor) -> tuple[Tensor, Tensor]:
        """Per-series `(occurrence, size)` harmonic weights, `[cells, n_terms]` each."""
        occ = self.shared_occurrence.unsqueeze(0) + self.shrink * self.deviation_occurrence(
            series_index
        )
        size = self.shared_size.unsqueeze(0) + self.shrink * self.deviation_size(series_index)
        return occ, size

    def deviation_scale(self) -> tuple[float, float]:
        """Mean absolute per-series deviation, `(occurrence, size)`.

        The number that answers whether this module earned its place. Near zero after training
        means the shrinkage held and pooled seasonality was enough; well away from zero means
        series genuinely differ in annual amplitude and the pooled version was averaging real
        structure away.
        """
        return (
            float(self.deviation_occurrence.weight.detach().abs().mean()),
            float(self.deviation_size.weight.detach().abs().mean()),
        )

    def forward(self, annual: Tensor, series_index: Tensor) -> tuple[Tensor, Tensor]:
        """Seasonal contribution to the occurrence and size linear predictors.

        Args:
            annual: `[cells, horizon, n_terms]` annual Fourier basis, the same columns
                `adapters.mft.annual_columns` names in the known-future tensor.
            series_index: `[cells]` which series each cell belongs to.

        Returns:
            Two `[cells, horizon]` tensors, ready to add to the head's linear predictors -- the
            same shape and the same insertion point as `factor.SharedLatentFactor.contract`.

        Raises:
            SeasonalAmplitudeError: the basis width disagrees with what was configured, which
                would otherwise broadcast a harmonic onto the wrong weight.
        """
        if annual.shape[-1] != self.n_terms:
            raise SeasonalAmplitudeError(
                f"annual basis is {annual.shape[-1]} wide but {self.n_terms} was configured; a "
                "broadcast here would pair each harmonic with another harmonic's weight"
            )
        if annual.shape[0] != series_index.shape[0]:
            raise SeasonalAmplitudeError(
                f"{annual.shape[0]} cells of basis against {series_index.shape[0]} series "
                "indices; these must describe the same cells"
            )
        w_occ, w_size = self.weights(series_index)
        return ((annual * w_occ.unsqueeze(1)).sum(-1), (annual * w_size.unsqueeze(1)).sum(-1))
