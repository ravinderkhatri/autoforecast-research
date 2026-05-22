"""Direct tests for Wrmsse count/label guards that survived the 2026-08-27 mutation pass."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.wrmsse import Wrmsse, WrmsseError


def _refused(fragment: str, **kw) -> None:
    try:
        Wrmsse(**kw)
    except WrmsseError as exc:
        assert fragment in str(exc), f"unexpected refusal: {exc}"
        return
    raise AssertionError(f"Wrmsse accepted {kw}")


def test_series_counts_must_name_exactly_the_scored_levels() -> None:
    """W1: a count for a level that was not scored, or a scored level without a count."""
    _refused("must describe exactly the scored levels", score=1.0, per_level={"total": 1.0},
             n_series_scored={"grp": 1})
    _refused("must describe exactly the scored levels", score=1.0,
             per_level={"total": 1.0, "grp": 1.0}, n_series_dropped={"total": 0})


def test_series_counts_must_be_non_negative_integers() -> None:
    """W2: True, 3049.0 or -1 are not series counts."""
    _refused("non-negative integers", score=1.0, per_level={"total": 1.0},
             n_series_scored={"total": True})
    _refused("non-negative integers", score=1.0, per_level={"total": 1.0},
             n_series_scored={"total": 3049.0})
    _refused("non-negative integers", score=1.0, per_level={"total": 1.0},
             n_series_scored={"total": -1})



if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
