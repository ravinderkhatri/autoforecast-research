"""Direct tests for the shared binary64 evidence predicate."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast._numeric import is_finite_real


def test_is_finite_real_accepts_only_representable_builtin_numbers() -> None:
    class HostileInt(int):
        def __float__(self):
            raise RuntimeError("untrusted coercion ran")

    for value in (0, -7, 0.0, -1.5, 1e308):
        assert is_finite_real(value) is True, value
    for value in (
        True, False, "1", None, float("nan"), float("inf"), 10**400, HostileInt(1),
    ):
        assert is_finite_real(value) is False, value


if __name__ == "__main__":
    test_is_finite_real_accepts_only_representable_builtin_numbers()
    print("PASS test_is_finite_real_accepts_only_representable_builtin_numbers")
