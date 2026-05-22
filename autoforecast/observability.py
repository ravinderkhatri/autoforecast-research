"""Optional TensorBoard logging tied to the evaluation manifest. Records training telemetry and
separate accuracy and bias metrics with their dataset slices.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from autoforecast.score import Score

# Import SummaryWriter lazily to avoid loading conflicting OpenMP runtimes before Spark forks.


class ObservabilityError(ValueError):
    """A run was opened in a way that would make its logs untraceable."""


@dataclass(frozen=True)
class RunLog:
    """One TensorBoard run, tied to the manifest it was produced under.

    Attributes:
        path: The run directory, `<root>/<full_manifest_hash>/<label>`.
        manifest_hash: The evaluation contract this run is claim-bearing under.
        label: What distinguishes this run from others under the same manifest -- normally the
            adapter name, so several adapters under one contract sit side by side in the UI.
    """

    path: pathlib.Path
    manifest_hash: str
    label: str


@dataclass
class TrainingLog:
    """Durable, per-epoch telemetry for one trainable-model invocation.

    This is deliberately separate from :class:`RunLog`: a training trace is operational
    evidence for one model-generation directory, whereas ``RunLog`` is keyed by an evaluation
    manifest and stores a scored comparison.  Mixing their directories would make a retry look
    like a continuation of an earlier optimiser run.
    """

    path: pathlib.Path | None
    _writer: Any | None
    _metrics_path: pathlib.Path | None = None
    _epoch_observer: Callable[[int, Mapping[str, float]], None] | None = None
    _next_epoch: int = 1
    _last_optimizer_steps: int = 0
    _last_training_examples_seen: int = 0

    def log_epoch(
        self,
        epoch: int,
        *,
        train_objective: float,
        validation_objective: float,
        learning_rate: float,
        gradient_norm: float,
        optimizer_steps: int,
        training_examples_seen: int,
        seconds: float,
        process_peak_rss_bytes: int,
        cuda_peak_memory_bytes: int | None,
    ) -> None:
        """Write one complete epoch and flush it before the next optimiser epoch starts."""
        if type(epoch) is not int or epoch != self._next_epoch:
            raise ObservabilityError(
                f"training epochs must be contiguous from 1; expected {self._next_epoch}, "
                f"got {epoch!r}")
        numeric = {
            "train_objective": train_objective,
            "validation_objective": validation_objective,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "seconds": seconds,
        }
        for name, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or value != value:
                raise ObservabilityError(f"{name} must be a finite numeric scalar, got {value!r}")
        if any(abs(float(value)) == float("inf") for value in numeric.values()):
            raise ObservabilityError("training telemetry cannot contain infinity")
        if (type(optimizer_steps) is not int or optimizer_steps < 1
                or optimizer_steps < self._last_optimizer_steps
                or type(training_examples_seen) is not int or training_examples_seen < 1
                or training_examples_seen < self._last_training_examples_seen
                or type(process_peak_rss_bytes) is not int or process_peak_rss_bytes < 0
                or (cuda_peak_memory_bytes is not None
                    and (type(cuda_peak_memory_bytes) is not int or cuda_peak_memory_bytes < 0))):
            raise ObservabilityError("training telemetry has invalid discrete resource fields")
        scalars: dict[str, float] = {
            "optimization/train_objective": train_objective,
            "optimization/validation_objective": validation_objective,
            "optimizer/learning_rate": learning_rate,
            "optimizer/gradient_norm": gradient_norm,
            "progress/optimizer_steps": float(optimizer_steps),
            "progress/training_examples_seen": float(training_examples_seen),
            "execution/epoch_seconds": seconds,
            "execution/process_peak_rss_bytes": float(process_peak_rss_bytes),
        }
        if cuda_peak_memory_bytes is not None:
            scalars["execution/cuda_peak_memory_bytes"] = float(cuda_peak_memory_bytes)
        if self._metrics_path is not None:
            record = {"epoch": epoch, **scalars}
            with self._metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
                handle.flush()
        if self._writer is not None:
            for tag, value in scalars.items():
                self._writer.add_scalar(tag, value, epoch)
            self._writer.flush()
        if self._epoch_observer is not None:
            self._epoch_observer(epoch, scalars)
        self._last_optimizer_steps = optimizer_steps
        self._last_training_examples_seen = training_examples_seen
        self._next_epoch += 1

    def close(self) -> None:
        """Release the event writer after all recorded epochs have been flushed."""
        if self._writer is not None:
            self._writer.close()


def open_run(root: str | pathlib.Path, manifest_hash: str, label: str) -> RunLog:
    """Create the run directory for `label` under `manifest_hash`.

    Args:
        root: Where runs live.
        manifest_hash: The complete lowercase SHA-256 from `EvaluationManifest.hash`, which names
            the parent directory and groups every run under one exact contract in the UI.
        label: Adapter name, or whatever distinguishes this run.

    Returns:
        A `RunLog`.

    Raises:
        ObservabilityError: the hash is not a complete lowercase SHA-256 or the label is unsafe.
            A truncated hash can collide, and a path-like label can escape or merge directories.
    """
    if not isinstance(manifest_hash, str) or re.fullmatch(r"[0-9a-f]{64}", manifest_hash) is None:
        raise ObservabilityError(
            "a run needs its complete lowercase SHA-256 manifest hash: a truncated or malformed "
            "TensorBoard identity can collide and is not claim-bearing")
    if (not isinstance(label, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label) is None
            or label in (".", "..")):
        raise ObservabilityError(
            "a run needs one safe non-path label, or runs can merge or escape their directory")
    path = pathlib.Path(root) / manifest_hash / label
    path.mkdir(parents=True, exist_ok=True)
    return RunLog(path=path, manifest_hash=manifest_hash, label=label)


def open_training_log(
    path: str | pathlib.Path | None,
    *,
    metrics_path: str | pathlib.Path | None = None,
    epoch_observer: Callable[[int, Mapping[str, float]], None] | None = None,
) -> TrainingLog:
    """Open durable per-epoch telemetry, with TensorBoard only when explicitly requested.

    A nonempty directory is refused rather than appended to: epoch one of a retry must not draw
    a false continuous line from a prior model state.
    """
    if path is None and metrics_path is None and epoch_observer is None:
        raise ObservabilityError(
            "training telemetry needs a metrics path, observer, or TensorBoard path")
    root = None if path is None else pathlib.Path(path)
    writer = None
    if root is not None:
        from torch.utils.tensorboard import SummaryWriter
        if root.exists() and any(root.iterdir()):
            raise ObservabilityError(
                f"training TensorBoard directory is not empty: {root}; use a fresh generation path")
        root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(root))
    metrics = None if metrics_path is None else pathlib.Path(metrics_path)
    if metrics is not None:
        if metrics.exists():
            raise ObservabilityError(
                f"training metrics file already exists: {metrics}; use a fresh attempt path")
        metrics.parent.mkdir(parents=True, exist_ok=True)
    return TrainingLog(
        path=root, _writer=writer, _metrics_path=metrics, _epoch_observer=epoch_observer,
    )


def log_score(run: RunLog, score: Score, step: int = 0, *, prefix: str = "gate") -> None:
    """Log a `Score` with bias kept separate from error.

    Bias is not logged as a magnitude and never folded into the error scalar. A chronic
    over-forecast and a chronic under-forecast are different business failures, and an absolute
    summary hides which one you have -- which is defect 1 of the deployed metric.

    Args:
        run: From `open_run`.
        score: What to log.
        step: TensorBoard step. For a fitted model this is the boosting iteration or epoch.
        prefix: Scalar namespace.
    """
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(run.path)) as w:
        w.add_scalar(f"{prefix}/mae", score.mae, step)
        w.add_scalar(f"{prefix}/bias", score.bias, step)
        w.add_scalar(f"{prefix}/n_scored", score.n_scored, step)
        if score.bias_pct_of_volume is not None:
            w.add_scalar(f"{prefix}/bias_pct_of_volume", score.bias_pct_of_volume, step)
        if score.absolute_error_pct_of_volume is not None:
            w.add_scalar(f"{prefix}/wape", score.absolute_error_pct_of_volume, step)
        if score.mase is not None:
            w.add_scalar(f"{prefix}/mase", score.mase, step)


def log_slices(run: RunLog, slices: dict[str, Score], step: int = 0) -> None:
    """Log one `Score` per slice -- normally per zero-rate decile.

    A model can improve overall while getting worse on slow movers, and slow movers are most of
    the catalogue: M5 is 59.5% zeros on eligible cells. An aggregate curve cannot show that, and
    by the time a gate result shows it the run is over.

    Args:
        run: From `open_run`.
        slices: slice name -> `Score`. Names appear in the TensorBoard tag, so they should sort
            in a meaningful order (`decile_00`, `decile_01`, ...).
        step: TensorBoard step.

    Raises:
        ObservabilityError: a slice name is blank, which would collapse two slices into one tag.
    """
    blank = [k for k in slices if not k or not k.strip()]
    if blank:
        raise ObservabilityError("every slice needs a name, or two slices share one tag")
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(run.path)) as w:
        for name, s in sorted(slices.items()):
            w.add_scalar(f"slice_mae/{name}", s.mae, step)
            w.add_scalar(f"slice_bias/{name}", s.bias, step)


def log_training_curve(run: RunLog, metric: str, values: list[float]) -> None:
    """Log a per-iteration curve, so a run's trajectory is visible while it trains.

    Args:
        run: From `open_run`.
        metric: Scalar name, e.g. `train/binary_logloss`.
        values: One value per iteration, in order.

    Raises:
        ObservabilityError: `values` is empty -- an empty curve reads in the UI as a run that
            was never logged rather than one that logged nothing.
    """
    if not values:
        raise ObservabilityError(f"no values to log for {metric!r}; an empty curve reads as a "
                                 "missing run rather than an empty one")
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(run.path)) as w:
        for i, v in enumerate(values):
            w.add_scalar(metric, v, i)



def log_leaderboard(run: RunLog, rows: list[dict], *, published: dict[str, float]) -> None:
    """Write the standing ranking as a TensorBoard text summary plus one scalar per model.

    **So the ranking travels with the runs.** It currently lives in a markdown file, which means
    it is visible where the repository is and nowhere else. `tfevents` is the format Vertex AI
    TensorBoard ingests, so writing it here is also what carries the table to the cloud when
    compute moves -- no second export, no second format.

    The published competition figures are logged in their own `m5/published` group, kept apart
    from `m5/measured`, because a chart that mixes them invites reading a quoted leaderboard
    number as something this repository produced.

    Args:
        run: An open run, from `open_run`.
        rows: `{"label", "wrmsse", "trained_here", ...}` per model, already ranked.
        published: Competition figures, for scale.

    Raises:
        ObservabilityError: a row has no label, which would log a scalar nobody can attribute.
    """
    from torch.utils.tensorboard import SummaryWriter

    for row in rows:
        if not row.get("label"):
            raise ObservabilityError(
                "a leaderboard row has no label; an unattributable scalar is worse than a "
                "missing one because it still draws a line")

    header = "| rank | model | built by | WRMSSE | vs winner |\n|---|---|---|---|---|\n"
    best = min(published.values()) if published else None
    lines = []
    for i, row in enumerate([r for r in rows if r.get("wrmsse") is not None], 1):
        who = "us" if row.get("trained_here") else "library"
        delta = (f"{(row['wrmsse'] / best - 1) * 100:+.1f}%" if best else "-")
        lines.append(f"| {i} | {row['label']} | {who} | {row['wrmsse']:.4f} | {delta} |")

    with SummaryWriter(log_dir=str(run.path)) as w:
        w.add_text("m5/leaderboard", header + "\n".join(lines))
        for name, value in published.items():
            w.add_scalar(f"m5/published/{name.replace(' ', '_')}", value, 0)
        for row in rows:
            if row.get("wrmsse") is None:
                continue
            group = "ours" if row.get("trained_here") else "comparator"
            w.add_scalar(f"m5/measured/{group}/{row['label'].replace(' ', '_')}",
                         row["wrmsse"], 0)
            # Secondary metrics on the same run, so a model is one place rather than four charts.
            for key, tag in (("mae", "mae"), ("gap", "coverage_gap"),
                             ("minutes", "minutes"), ("peak_gb", "peak_rss_gb")):
                if row.get(key) is not None:
                    w.add_scalar(f"m5/{tag}/{row['label'].replace(' ', '_')}", row[key], 0)

__all__ = ["ObservabilityError", "RunLog", "TrainingLog", "log_leaderboard", "log_score",
           "log_slices", "log_training_curve", "open_training_log",
           "open_run"]
