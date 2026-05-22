"""Tests for the TensorBoard adapter.

Checks whether a run can be traced to its manifest and whether bias is logged as its own scalar.
Synthetic scores exercise the logging contract without depending on a dataset.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from autoforecast import observability as obs
from autoforecast.score import Score

HASH = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"


def score(mae: float = 1.2, bias: float = -0.3) -> Score:
    return Score(n_scored=100, n_dropped_ineligible=0, n_dropped_no_data=0, n_zero_filled=0,
                 mae=mae, rmse=mae, bias=bias, bias_pct_of_volume=bias / 2, mase=None,
                 sum_abs_error=mae * 100, sum_squared_error=mae * mae * 100,
                 sum_error=bias * 100, sum_actual=200.0, sum_naive_abs_error=0.0)


def events(path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path.glob("events.out.tfevents.*"))


def test_open_run_puts_the_manifest_hash_in_the_path() -> None:
    """A run folder that cannot be traced to its evaluation contract is not claim-bearing."""
    with tempfile.TemporaryDirectory() as d:
        run = obs.open_run(d, HASH, "croston")
        assert run.path.parent.name == HASH
        assert run.path.name == "croston"
        assert run.path.is_dir()


def test_two_adapters_under_one_manifest_share_a_parent() -> None:
    """Which is what makes them comparable in the UI rather than two unrelated runs."""
    with tempfile.TemporaryDirectory() as d:
        a = obs.open_run(d, HASH, "croston")
        b = obs.open_run(d, HASH, "shallow_global")
        assert a.path.parent == b.path.parent and a.path != b.path


def test_open_run_refuses_a_malformed_or_truncated_manifest_hash() -> None:
    with tempfile.TemporaryDirectory() as d:
        for bad in ("", "   ", "abc123def456", "A" * 64, "g" * 64, True):
            try:
                obs.open_run(d, bad, "x")
            except obs.ObservabilityError as e:
                assert "claim-bearing" in str(e); continue
            raise AssertionError(f"opened an untraceable run for hash {bad!r}")


def test_open_run_refuses_a_blank_label() -> None:
    with tempfile.TemporaryDirectory() as d:
        for label in (" ", "../escape", "a/b", ".", "..", True):
            try:
                obs.open_run(d, HASH, label)
            except obs.ObservabilityError as e:
                assert "path" in str(e); continue
            raise AssertionError(f"unsafe TensorBoard label was accepted: {label!r}")


def test_full_hashes_with_the_same_twelve_character_prefix_do_not_collide() -> None:
    with tempfile.TemporaryDirectory() as d:
        first = "a" * 64
        second = "a" * 12 + "b" * 52
        one = obs.open_run(d, first, "m5")
        two = obs.open_run(d, second, "m5")
        assert one.path != two.path
        assert one.path.parent.name == first
        assert two.path.parent.name == second


def test_log_score_writes_an_event_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = obs.open_run(d, HASH, "croston")
        obs.log_score(run, score())
        assert events(run.path), "no TensorBoard event file was written"


def test_bias_is_logged_as_its_own_signed_scalar() -> None:
    """The defect this project exists to fix is a metric blind to over-forecasting. If bias were
    folded into the error scalar, or logged as a magnitude, the training view would repeat it."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with tempfile.TemporaryDirectory() as d:
        run = obs.open_run(d, HASH, "croston")
        obs.log_score(run, score(mae=1.2, bias=-0.3))
        acc = EventAccumulator(str(run.path)); acc.Reload()
        tags = acc.Tags()["scalars"]
        assert "gate/bias" in tags and "gate/mae" in tags
        assert acc.Scalars("gate/bias")[0].value < 0, "bias lost its sign"
        assert abs(acc.Scalars("gate/mae")[0].value - 1.2) < 1e-6


def test_log_slices_keeps_every_decile_separate() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with tempfile.TemporaryDirectory() as d:
        run = obs.open_run(d, HASH, "shallow_global")
        obs.log_slices(run, {f"decile_{i:02d}": score(mae=1.0 + i / 10) for i in range(10)})
        acc = EventAccumulator(str(run.path)); acc.Reload()
        tags = acc.Tags()["scalars"]
        assert len([t for t in tags if t.startswith("slice_mae/")]) == 10


def test_log_slices_refuses_a_blank_slice_name() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = obs.open_run(d, HASH, "x")
        try:
            obs.log_slices(run, {"": score()})
        except obs.ObservabilityError as e:
            assert "share one tag" in str(e); return
        raise AssertionError("two slices would collapse into one tag")


def test_log_training_curve_writes_one_point_per_iteration() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with tempfile.TemporaryDirectory() as d:
        run = obs.open_run(d, HASH, "shallow_global")
        obs.log_training_curve(run, "train/binary_logloss", [0.6, 0.5, 0.45])
        acc = EventAccumulator(str(run.path)); acc.Reload()
        assert len(acc.Scalars("train/binary_logloss")) == 3


def test_log_training_curve_refuses_an_empty_curve() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = obs.open_run(d, HASH, "x")
        try:
            obs.log_training_curve(run, "train/loss", [])
        except obs.ObservabilityError as e:
            assert "missing run" in str(e); return
        raise AssertionError("wrote an empty curve, which reads as a run that never logged")


def test_training_log_writes_one_complete_tensorboard_epoch_and_refuses_mixed_runs() -> None:
    """A real event reader sees the model, optimiser, and resource series at the same step."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "training"
        metrics_path = pathlib.Path(d) / "metrics" / "epochs.jsonl"
        run = obs.open_training_log(path, metrics_path=metrics_path)
        run.log_epoch(
            1,
            train_objective=1.25,
            validation_objective=1.5,
            learning_rate=0.003,
            gradient_norm=0.75,
            optimizer_steps=4,
            training_examples_seen=12,
            seconds=2.5,
            process_peak_rss_bytes=123456,
            cuda_peak_memory_bytes=None,
        )
        run.close()

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(path))
        events.Reload()
        tags = set(events.Tags()["scalars"])
        assert tags == {
            "optimization/train_objective", "optimization/validation_objective",
            "optimizer/learning_rate",
            "optimizer/gradient_norm", "progress/optimizer_steps",
            "progress/training_examples_seen", "execution/epoch_seconds",
            "execution/process_peak_rss_bytes",
        }
        assert events.Scalars("optimization/train_objective")[0].value == 1.25
        assert events.Scalars("optimization/validation_objective")[0].value == 1.5
        assert events.Scalars("progress/optimizer_steps")[0].value == 4.0
        assert json.loads(metrics_path.read_text()) == {
            "epoch": 1,
            "optimizer/gradient_norm": 0.75,
            "optimizer/learning_rate": 0.003,
            "progress/optimizer_steps": 4,
            "progress/training_examples_seen": 12,
            "execution/epoch_seconds": 2.5,
            "execution/process_peak_rss_bytes": 123456.0,
            "optimization/train_objective": 1.25,
            "optimization/validation_objective": 1.5,
        }
        try:
            obs.open_training_log(path)
        except obs.ObservabilityError as exc:
            assert "not empty" in str(exc)
        else:
            raise AssertionError("mixed two model generations into one TensorBoard directory")


def test_production_training_telemetry_writes_jsonl_without_tensorboard() -> None:
    """Normal model training has one durable metric stream and no duplicate event file."""
    with tempfile.TemporaryDirectory() as d:
        metrics_path = pathlib.Path(d) / "metrics" / "epochs.jsonl"
        run = obs.open_training_log(None, metrics_path=metrics_path)
        run.log_epoch(
            1,
            train_objective=1.25,
            validation_objective=1.5,
            learning_rate=0.003,
            gradient_norm=0.75,
            optimizer_steps=4,
            training_examples_seen=12,
            seconds=2.5,
            process_peak_rss_bytes=123456,
            cuda_peak_memory_bytes=None,
        )
        run.close()
        record = json.loads(metrics_path.read_text())
        assert record["optimization/train_objective"] == 1.25
        assert record["optimization/validation_objective"] == 1.5
        assert not list(pathlib.Path(d).rglob("events.out.tfevents.*"))



def test_the_leaderboard_reaches_tensorboard_as_text_and_scalars() -> None:
    """The ranking currently lives in a markdown file, visible where the repository is and nowhere
    else. tfevents is the format Vertex AI TensorBoard ingests, so writing it here is also what
    carries the table to the cloud when compute moves."""
    import tempfile
    run = obs.open_run(tempfile.mkdtemp(), HASH, "m5")
    rows = [
        {"label": "shallow_global", "wrmsse": 0.6953, "trained_here": False, "mae": 1.053},
        {"label": "ours 3", "wrmsse": 0.8272, "trained_here": True, "mae": 1.101,
         "gap": 0.095, "minutes": 43.0, "peak_gb": 6.1},
        {"label": "ours 0", "wrmsse": None, "trained_here": True},   # pending, must be skipped
    ]
    obs.log_leaderboard(run, rows, published={"M5 winner": 0.520})
    written = list(pathlib.Path(run.path).glob("*tfevents*"))
    assert written, "no events file was written"
    assert written[0].stat().st_size > 0


def test_a_row_without_a_label_is_refused() -> None:
    """An unattributable scalar is worse than a missing one, because it still draws a line."""
    import tempfile
    run = obs.open_run(tempfile.mkdtemp(), HASH, "m5")
    try:
        obs.log_leaderboard(run, [{"label": "", "wrmsse": 0.5}], published={})
    except obs.ObservabilityError as e:
        assert "unattributable" in str(e); return
    raise AssertionError("logged a scalar nobody can attribute")


def test_published_figures_are_kept_apart_from_measured_ones() -> None:
    """A chart that mixes them invites reading a quoted competition number as something this
    repository produced."""
    import inspect
    src = inspect.getsource(obs.log_leaderboard)
    assert "m5/published/" in src and "m5/measured/" in src
    assert src.index("m5/published/") != src.index("m5/measured/")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
