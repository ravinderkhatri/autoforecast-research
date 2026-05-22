"""Direct tests for Score validator guards that survived the 2026-08-27 mutation pass."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast.score import Score

VALID = dict(n_scored=4, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
             mae=1.0, rmse=1.0, bias=1.0, bias_pct_of_volume=0.5, mase=None,
             sum_abs_error=4.0, sum_squared_error=4.0, sum_error=4.0, sum_actual=8.0,
             sum_naive_abs_error=0.0)


def _refused(message_fragment: str, **over) -> None:
    try:
        Score(**{**VALID, **over})
    except ValueError as exc:
        assert message_fragment in str(exc), f"{over}: unexpected refusal {exc}"
        return
    raise AssertionError(f"Score accepted {over}")


def test_bias_must_agree_with_its_raw_sum() -> None:
    """S3: the displayed signed bias is evidence only if it equals sum_error / n_scored."""
    _refused("bias 0.9 disagrees", bias=0.9)


def test_bias_pct_of_volume_must_agree_with_its_raw_sums() -> None:
    """S4: the ladder's eligibility bar reads this field; it must equal sum_error/sum_actual."""
    _refused("bias_pct_of_volume disagrees", bias_pct_of_volume=0.49)
    _refused("bias_pct_of_volume disagrees", sum_actual=0.0, bias_pct_of_volume=0.0)


def test_mase_must_agree_with_its_raw_sums_or_be_absent() -> None:
    """S1: MASE without a naive denominator, or disagreeing with it, is not evidence."""
    _refused("mase disagrees", mase=1.0)
    _refused("mase disagrees", mase=1.5, sum_naive_abs_error=4.0)
    Score(**{**VALID, "mase": 1.0, "sum_naive_abs_error": 4.0})       # consistent: accepted


def test_counts_must_be_true_integers() -> None:
    """S5: True and 853720.0 are not cell counts, even where their arithmetic would agree."""
    _refused("non-negative integers", n_scored=True, mae=4.0, rmse=4.0, bias=4.0)
    n = 853720
    _refused("non-negative integers", n_scored=float(n), sum_abs_error=float(n),
             sum_squared_error=float(n), sum_error=float(n), sum_actual=2.0 * n)
    _refused("non-negative integers", n_dropped_ineligible=1.0)


def test_negative_sums_are_refused_before_any_ratio_check() -> None:
    """S6: a negative absolute-error, squared-error, actual or naive sum is refused as such."""
    _refused("cannot be negative", sum_naive_abs_error=-1.0)
    _refused("cannot be negative", sum_squared_error=-4.0, rmse=1.0)


def test_a_zero_cell_score_cannot_carry_raw_sums() -> None:
    """S7: with nothing scored there is no evidence to sum."""
    _refused("cannot carry non-zero raw sums", n_scored=0, mae=0.0, rmse=0.0, bias=0.0,
             bias_pct_of_volume=None, sum_abs_error=1.0, sum_squared_error=0.0,
             sum_error=0.0, sum_actual=0.0)


def test_combining_only_empty_partitions_is_refused_not_divided_by_zero() -> None:
    """S8: an all-empty reduction must refuse with ValueError, not ZeroDivisionError."""
    empty = Score(**{**VALID, "n_scored": 0, "n_dropped_ineligible": 3, "mae": 0.0, "rmse": 0.0,
                     "bias": 0.0, "bias_pct_of_volume": None, "sum_abs_error": 0.0,
                     "sum_squared_error": 0.0, "sum_error": 0.0, "sum_actual": 0.0})
    try:
        Score.combine([empty, empty])
    except ValueError as exc:
        assert "nothing to combine" in str(exc)
        return
    raise AssertionError("combined two empty partitions into a score")

def test_the_human_readable_score_format_is_pinned() -> None:
    """`str(Score)` is persisted in every rung record and compared byte-for-byte on reuse; a
    cosmetic format change would silently invalidate every completed rung. Change it on purpose."""
    assert str(Score(**VALID)) == '[cell] n=4 (zero-filled 0, ineligible dropped 0, no-data dropped 0)  RMSE=1.000  MAE=1.000  bias=+1.000 (+50.0% of volume)'


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
