"""Tests for static variable selection.

Run: python3 forecast/models/mft/test_selection.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import torch
from forecast.models.mft.selection import SelectionError, StaticVariableSelection

WIDTHS = (7, 3, 3, 10)  # M5: dept_id, cat_id, state_id, store_id


def attributes(n: int = 32) -> torch.Tensor:
    x = torch.zeros(n, sum(WIDTHS))
    at = 0
    for w in WIDTHS:
        x[torch.arange(n), at + torch.randint(0, w, (n,))] = 1.0
        at += w
    return x


def test_an_untrained_selector_is_all_but_the_identity() -> None:
    """NEARLY uniform, not exactly. Small final weights rather than zero, because a zero output
    weight sends zero gradient to the hidden layer and freezes it -- the trap the hurdle head and
    the event kernel both hit. The cost is a softmax that starts a fraction of a percent off
    uniform instead of exactly on it."""
    m = StaticVariableSelection(WIDTHS)
    x = attributes()
    out, weights = m(x)
    assert torch.allclose(out, x, rtol=0.02), "selection materially changed the attributes"
    assert torch.allclose(weights, torch.full_like(weights, 0.25), atol=0.01)


def test_every_layer_of_the_scorer_learns_from_the_first_step() -> None:
    """What the zero init would have broken. The hidden layer must not wait for the output layer
    to move before it receives any signal at all."""
    m = StaticVariableSelection(WIDTHS)
    out, _ = m(attributes())
    out.sum().backward()
    assert float(m.score[0].weight.grad.abs().sum()) > 0, "the hidden layer is starved"
    assert float(m.score[-1].weight.grad.abs().sum()) > 0


def test_the_weights_compete_because_a_softmax_makes_them() -> None:
    """A sigmoid gate can turn everything on at once, which is what the model already does
    implicitly. Competition is what makes the weights readable."""
    m = StaticVariableSelection(WIDTHS)
    with torch.no_grad():
        m.score[-1].bias[0] = 4.0
    weights = m.selection_weights(attributes())
    assert float(weights[0, 0]) > 0.8
    assert abs(float(weights[0].sum()) - 1.0) < 1e-5, "weights do not sum to one"


def test_a_whole_group_moves_together_not_its_levels() -> None:
    """Weighting each one-hot column separately would let the model prefer some departments to
    others, which is a bias term in disguise -- the projection can already express that."""
    m = StaticVariableSelection(WIDTHS)
    with torch.no_grad():
        m.score[-1].bias[1] = 5.0
    x = torch.ones(1, sum(WIDTHS))
    out, _ = m(x)
    dept, cat = out[0, :7], out[0, 7:10]
    assert float(dept.std()) < 1e-6 and float(cat.std()) < 1e-6, "levels within a group diverged"
    assert float(cat.mean()) > float(dept.mean()), "the up-weighted group did not rise"


def test_the_weights_are_readable_per_series() -> None:
    """`selection_weights` is the only place this architecture answers 'which attribute is this
    forecast leaning on'."""
    m = StaticVariableSelection(WIDTHS)
    weights = m.selection_weights(attributes(16))
    assert weights.shape == (16, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(16), atol=1e-5)


def test_a_mismatched_width_is_refused() -> None:
    """A mismatch would apply a group's weight to the wrong columns."""
    m = StaticVariableSelection(WIDTHS)
    try:
        m(torch.zeros(4, 5))
    except SelectionError as e:
        assert "scale the wrong columns" in str(e)
        return
    raise AssertionError("selected over attributes of the wrong width")


def test_an_empty_or_zero_width_group_is_refused() -> None:
    """A zero-width group still takes softmax mass from the groups that carry something."""
    for bad in ((), (7, 0, 3)):
        try:
            StaticVariableSelection(bad)
        except SelectionError:
            continue
        raise AssertionError(f"accepted widths {bad}")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
