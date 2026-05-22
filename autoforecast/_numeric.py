"""Internal predicates for numeric evidence that must fit the binary64 metric domain."""

from __future__ import annotations

import math
from typing import TypeGuard


def is_finite_real(value: object) -> TypeGuard[int | float]:
    """Return whether a strict JSON number is representable as a finite Python float."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False
