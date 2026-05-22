"""Tests for per-series annual amplitude.

The module exists because a pooled annual response gives a barbecue line and a school-supply line
the same curve, in the same months. The tests are shaped around the two ways that can go wrong:
the deviation never moves (and the module is decoration), or it moves freely (and it fits five
annual observations with six coefficients).

Run: python3 forecast/models/mft/test_seasonality.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import torch
from forecast.models.mft.seasonality import (
    SeasonalAmplitude,
    SeasonalAmplitudeError,
)

N_TERMS, N_SERIES, HORIZON = 6, 40, 28


def basis(n_cells: int = N_SERIES, horizon: int = HORIZON, harmonics: int = 3) -> torch.Tensor:
    """The annual Fourier basis over a horizon, as `adapters.mft` emits it."""
    out = torch.zeros(n_cells, horizon, 2 * harmonics)
    for step in range(horizon):
        phase = 2.0 * math.pi * (step * 12) / 365.25  # spread across the year
        for k in range(1, harmonics + 1):
            out[:, step, 2 * (k - 1)] = math.sin(k * phase)
            out[:, step, 2 * (k - 1) + 1] = math.cos(k * phase)
    return out


def test_it_starts_pooled_so_every_series_gets_the_same_response() -> None:
    """Zero-initialised deviations mean the model begins with one shared annual shape. A run that
    ENDS there is reporting that pooled seasonality was sufficient, which is a finding."""
    m = SeasonalAmplitude(N_TERMS, N_SERIES)
    occ, size = m(basis(), torch.arange(N_SERIES))
    assert torch.allclose(occ[0], occ[1]) and torch.allclose(
        size[0], size[1]
    ), "two series got different annual responses before anything was trained"
    assert m.deviation_scale() == (0.0, 0.0)


def test_the_deviation_is_shrunk_rather_than_free() -> None:
    """A series in a five-year panel has seen about five Januaries. Free per-series harmonic
    weights would fit six coefficients to five observations exactly and meaninglessly."""
    m = SeasonalAmplitude(N_TERMS, N_SERIES, shrink=0.1)
    with torch.no_grad():
        m.deviation_occurrence.weight[3] = 1.0
    occ, _ = m.weights(torch.tensor([3]))
    # The deviation is 1.0 but enters at the shrink scale, not at full strength.
    assert torch.allclose(occ[0], torch.full((N_TERMS,), 0.1)), occ[0]


def test_occurrence_and_size_carry_separate_weights() -> None:
    """A product can sell on more DAYS in December without selling more per transaction. One
    shared coefficient forces those to move together, and the hurdle exists to keep them apart."""
    m = SeasonalAmplitude(N_TERMS, N_SERIES)
    with torch.no_grad():
        m.shared_occurrence += 0.5  # occurrence only
    occ, size = m(basis(), torch.arange(N_SERIES))
    assert float(occ.abs().sum()) > 0.0
    assert (
        float(size.abs().sum()) == 0.0
    ), "moving the occurrence weight moved the size response too"


def test_a_series_can_learn_an_amplitude_the_pool_does_not_have() -> None:
    """The property the module is for: fit two series with OPPOSITE annual seasonality and check
    the deviations separate. A pooled response cannot represent both and would land near zero."""
    torch.manual_seed(0)
    m = SeasonalAmplitude(N_TERMS, 2, shrink=1.0)  # shrink off, to test capacity not prior
    x = basis(n_cells=2)
    target = torch.stack([x[0, :, 0], -x[1, :, 0]])  # series 0 and 1 in antiphase
    idx = torch.tensor([0, 1])
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        occ, _ = m(x, idx)
        ((occ - target) ** 2).mean().backward()
        opt.step()
    occ, _ = m(x, idx)
    assert float(((occ - target) ** 2).mean()) < 0.05, "could not fit two opposite seasonalities"
    a, b = m.weights(idx)[0]
    assert (
        float((a - b).abs().sum()) > 0.5
    ), "the two series ended with the same weights, so nothing per-series was learned"


def test_shrinkage_at_zero_makes_the_model_purely_pooled() -> None:
    """The ablation's other end: `shrink=0` is the pooled model, so the two configurations differ
    by exactly one number and the comparison is clean."""
    m = SeasonalAmplitude(N_TERMS, N_SERIES, shrink=0.0)
    with torch.no_grad():
        m.deviation_occurrence.weight.normal_(0, 5.0)  # large, and it must not matter
    occ, _ = m(basis(), torch.arange(N_SERIES))
    assert torch.allclose(occ[0], occ[7]), "shrink=0 still let a per-series deviation through"


def test_a_mismatched_basis_width_is_refused() -> None:
    """Broadcasting here would pair each harmonic with another harmonic's weight, and the shapes
    are compatible enough that nothing else would notice."""
    m = SeasonalAmplitude(N_TERMS, N_SERIES)
    try:
        m(torch.zeros(N_SERIES, HORIZON, 4), torch.arange(N_SERIES))
    except SeasonalAmplitudeError as e:
        assert "another harmonic's weight" in str(e)
        return
    raise AssertionError("accepted a basis of the wrong width")


def test_a_cell_count_that_disagrees_with_the_index_is_refused() -> None:
    m = SeasonalAmplitude(N_TERMS, N_SERIES)
    try:
        m(basis(n_cells=5), torch.arange(3))
    except SeasonalAmplitudeError as e:
        assert "same cells" in str(e)
        return
    raise AssertionError("paired five cells of basis with three series indices")


def test_an_odd_basis_width_is_refused() -> None:
    """The basis is sine/cosine PAIRS; an odd width means a harmonic lost half of itself."""
    try:
        SeasonalAmplitude(5, N_SERIES)
    except SeasonalAmplitudeError as e:
        assert "sine/cosine pairs" in str(e)
        return
    raise AssertionError("accepted an odd basis width")


def test_a_shrink_above_one_is_refused() -> None:
    """Above 1 the deviation is amplified rather than shrunk, which is the opposite of what makes
    five annual observations estimable."""
    try:
        SeasonalAmplitude(N_TERMS, N_SERIES, shrink=1.5)
    except SeasonalAmplitudeError as e:
        assert "amplified rather than shrunk" in str(e)
        return
    raise AssertionError("accepted a shrink above 1")


def test_an_empty_series_table_is_refused() -> None:
    try:
        SeasonalAmplitude(N_TERMS, 0)
    except SeasonalAmplitudeError as e:
        assert "n_series must be" in str(e)
        return
    raise AssertionError("built a per-series table with no series")


def test_the_deviation_scale_reports_whether_the_module_earned_its_place() -> None:
    """The number to read after a run. Near zero means shrinkage held and pooled seasonality was
    enough; away from zero means series genuinely differ and pooling was averaging real structure
    away."""
    m = SeasonalAmplitude(N_TERMS, N_SERIES)
    assert m.deviation_scale() == (0.0, 0.0)
    with torch.no_grad():
        m.deviation_occurrence.weight.fill_(0.25)
        m.deviation_size.weight.fill_(0.5)
    occ, size = m.deviation_scale()
    assert abs(occ - 0.25) < 1e-6 and abs(size - 0.5) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
