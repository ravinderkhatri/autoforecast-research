"""Reusable Keras-style progress for PyTorch training and Spark-backed pipelines."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # custom-torch does not require the optional terminal renderer
    _tqdm = None

__all__ = ["PipelineProgress", "TrainingProgress"]


_WORKFLOW_PHASES = {
    "validation": {
        "input": (1, 6, "Prepare governed training and validation slices"),
        "preflight": (2, 6, "Run Spark data and model preflight"),
        "data": (3, 6, "Build model tensors"),
        "selection": (4, 6, "Train, validate, and select the update budget"),
        "forecast": (5, 6, "Generate and score the validation forecast"),
        "score": (5, 6, "Generate and score the validation forecast"),
    },
    "selection_lock": {"seal": (6, 6, "Freeze the validated configuration")},
    "final_fit": {
        "input": (1, 5, "Prepare combined training and validation data"),
        "preflight": (2, 5, "Run Spark data and model preflight"),
        "data": (3, 5, "Build model tensors"),
        "selection": (4, 5, "Fit the selected architecture on combined data"),
        "refit": (4, 5, "Fit the selected architecture on combined data"),
        "forecast": (5, 5, "Generate the sealed test forecast"),
    },
    "final_score": {
        "input": (1, 3, "Verify sealed test inputs"),
        "score": (2, 3, "Calculate test WRMSSE"),
    },
    "report": {"seal": (3, 3, "Verify artifacts and publish the report")},
}


def _campaign_stage_label() -> str | None:
    """Read optional supervisor metadata without letting display-only env break training."""
    index, total = (
        os.environ.get("AUTOFORECAST_CAMPAIGN_STAGE_INDEX"),
        os.environ.get("AUTOFORECAST_CAMPAIGN_STAGE_TOTAL"),
    )
    try:
        parsed_index, parsed_total = int(index or ""), int(total or "")
    except ValueError:
        return None
    if not 1 <= parsed_index <= parsed_total:
        return None
    return f"Stage {parsed_index}/{parsed_total}"


def _workflow_phase_label(key: str, fallback: str) -> str:
    """Resolve display-only workflow context; malformed metadata never controls work."""
    if os.environ.get("AUTOFORECAST_WORKFLOW_LAYOUT") != "single_model_benchmark_v1":
        return f"Phase — {fallback}"
    role = os.environ.get("AUTOFORECAST_WORKFLOW_STAGE_ROLE", "")
    phase = _WORKFLOW_PHASES.get(role, {}).get(key)
    if phase is None:
        return f"Phase — {fallback}"
    index, total, title = phase
    return f"Phase {index}/{total} — {title}"


def _process_tree_memory() -> tuple[float | None, float | None]:
    """Return process-tree RSS and system-available GiB without making psutil mandatory."""
    try:
        import psutil

        process = psutil.Process()
        members = (process, *process.children(recursive=True))
        rss = 0
        for member in members:
            try:
                rss += member.memory_info().rss
            except psutil.Error:
                continue
        return rss / 1024**3, psutil.virtual_memory().available / 1024**3
    except (ImportError, OSError):
        return None, None


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class PipelineProgress:
    """Logical phase/step progress with read-only Spark and RAM telemetry."""

    def __init__(
        self,
        name: str,
        total: int,
        *,
        phase_key: str | None = None,
        stream=None,
        interval_seconds: float = 15.0,
    ):
        if not isinstance(name, str) or not name.strip() or type(total) is not int or total < 1:
            raise ValueError("pipeline progress requires a name and positive total")
        if not isinstance(interval_seconds, int | float) or interval_seconds < 0:
            raise ValueError("pipeline progress interval must be non-negative")
        self.name = name.strip()
        self.total = total
        self.stream = stream if stream is not None else sys.stderr
        is_tty = bool(self.stream is not None and getattr(self.stream, "isatty", lambda: False)())
        self.enabled = is_tty or os.environ.get("AUTOFORECAST_LIVE_PROGRESS") == "1"
        self.interval_seconds = float(interval_seconds)
        self.campaign_stage = _campaign_stage_label()
        self.phase_label = _workflow_phase_label(phase_key or name.lower(), self.name)
        self.completed = 0
        self.started = time.monotonic()
        self._step_started = self.started
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._spark = None
        self._label = ""
        self._operation = ""
        self._write_lock = threading.Lock()

    def _spark_tasks(self) -> tuple[tuple[int, ...], int, int, tuple[str, ...]]:
        if self._spark is None:
            return (), 0, 0, ()
        try:
            tracker = self._spark.sparkContext.statusTracker()
            ids = tuple(sorted(tracker.getActiveStageIds()))
            info = [tracker.getStageInfo(stage_id) for stage_id in ids]
            live = [item for item in info if item is not None]
            return (
                ids,
                sum(item.numCompletedTasks for item in live),
                sum(item.numTasks for item in live),
                tuple(
                    str(getattr(item, "name", "")).strip()
                    for item in live
                    if str(getattr(item, "name", "")).strip()
                ),
            )
        except Exception:
            return (), 0, 0, ()

    def _line(self, status: str) -> str:
        stage_ids, tasks_done, tasks_total, stage_names = self._spark_tasks()
        fraction = min(1.0, self.completed / self.total)
        width = 20
        filled = min(width, int(fraction * width))
        bar = "━" * filled + "╺" * int(filled < width) + " " * max(0, width - filled - 1)
        elapsed = time.monotonic() - self.started
        eta = "unavailable"
        if self.completed:
            eta = _duration(elapsed * (self.total - self.completed) / self.completed)
        rss, available = _process_tree_memory()
        metrics = [f"elapsed: {_duration(elapsed)}", f"ETA: {eta}"]
        if self._operation:
            metrics.append(f"operation: {self._operation}")
        if stage_ids:
            metrics.extend(
                (
                    f"Spark engine stage ID: {','.join(map(str, stage_ids))} (diagnostic only)",
                    f"Spark tasks: {tasks_done}/{tasks_total}",
                )
            )
        if stage_names:
            metrics.append(f"engine operation: {'; '.join(stage_names)}")
        if rss is not None:
            metrics.append(f"process_tree_RAM: {rss:.2f} GiB")
        if available is not None:
            metrics.append(f"available_RAM: {available:.2f} GiB")
        index = min(self.completed + int(status == "RUNNING"), self.total)
        return (
            f"Step {index}/{self.total} {bar} {fraction * 100:5.1f}% - "
            + " - ".join(metrics)
            + (f" - {status}" if status else "")
        )

    def _write(self, status: str, *, final: bool) -> None:
        if self.enabled:
            with self._write_lock:
                print(self._line(status), file=self.stream, flush=True)

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write("RUNNING", final=False)

    def operation(self, label: str) -> None:
        """Name the current business operation without treating Spark job IDs as workflow."""
        if not isinstance(label, str) or not label.strip():
            raise ValueError("pipeline operation must be non-blank")
        self._operation = label.strip()
        if self._spark is not None:
            try:
                self._spark.sparkContext.setJobDescription(self._operation)
            except Exception:
                pass

    @contextmanager
    def step(self, label: str, *, spark=None):
        """Run one logical substage; telemetry never controls or mutates the operation."""
        if not isinstance(label, str) or not label.strip() or self._monitor is not None:
            raise ValueError("pipeline progress requires one non-blank active substage")
        if self.completed >= self.total:
            raise ValueError("pipeline progress has no remaining substages")
        self._label = label.strip()
        self._spark = spark
        self._step_started = time.monotonic()
        prefix = f"{self.campaign_stage} | " if self.campaign_stage else ""
        if self.enabled:
            print(
                f"{prefix}{self.phase_label} | "
                f"Step {self.completed + 1}/{self.total} — {self._label}",
                file=self.stream,
                flush=True,
            )
        self._stop.clear()
        if self.enabled and self.interval_seconds:
            self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor.start()
        try:
            yield
        except BaseException:
            self._finish_monitor()
            self._write("FAILED", final=True)
            raise
        else:
            self._finish_monitor()
            self.completed += 1
            self._write("DONE", final=True)
        finally:
            self._spark = None
            self._label = ""
            self._operation = ""

    def _finish_monitor(self) -> None:
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=max(1.0, self.interval_seconds + 1.0))
            self._monitor = None


class TrainingProgress:
    """TTY bar or bounded plain lines; no model/RNG access. test_training_progress covers modes."""

    def __init__(self, mode="auto", *, stream=None):
        if mode not in {"auto", "plain", "off"}:
            raise ValueError("progress must be auto, plain or off")
        self.stream = stream if stream is not None else sys.stderr
        live = os.environ.get("AUTOFORECAST_LIVE_PROGRESS") == "1"
        self.mode = (
            "tty"
            if mode == "auto" and self.stream is not None and (self.stream.isatty() or live)
            else mode
        )
        self.bar = None
        self.last_line = 0.0
        self.stage_name = None
        self.phase = "setup"
        self.total = self.initial = self.current = 0
        self.started = time.monotonic()
        self.display_started = self.started
        self.display_initial = 0
        self.campaign_stage = _campaign_stage_label()
        self.phase_label = _workflow_phase_label("selection", "Training setup")
        self.display_epoch = None
        self.bar_kind = None
        self.validation_position = None

    @staticmethod
    def _phase_title(phase):
        """Explain the two optimization phases without changing their durable names."""
        if os.environ.get("AUTOFORECAST_WORKFLOW_LAYOUT") == "single_model_benchmark_v1":
            role = os.environ.get("AUTOFORECAST_WORKFLOW_STAGE_ROLE")
            if phase == "selection" and role == "validation":
                return "training"
            if phase == "refit" and role == "final_fit":
                return "training on train + validation"
        return {
            "selection": "selection (train + validation/early stopping)",
            "refit": "refit (fresh training with the frozen update budget)",
        }.get(phase, phase)

    def _label(self, value):
        """Prefix child progress with immutable stage and display-only phase context."""
        parts = [part for part in (self.campaign_stage, self.phase_label, value) if part]
        return " | ".join(parts)

    def begin(self, phase, total, *, initial=0):
        if (
            type(total) is not int
            or total < 1
            or type(initial) is not int
            or not 0 <= initial <= total
            or not isinstance(phase, str)
            or not phase
        ):
            raise ValueError("invalid progress phase/budget/initial counter")
        self.close()
        self.phase, self.total, self.initial, self.current = phase, total, initial, initial
        self.phase_label = _workflow_phase_label(phase, self._phase_title(phase))
        self.started = time.monotonic()
        self.display_started = self.started
        self.display_initial = initial
        self.stage_name = None
        self.display_epoch = None
        if self.mode == "tty" and _tqdm is None:
            self.mode = "plain"
        self.stage(f"{phase}: start at update {initial}/{total}", force=True)

    def _position(self, record):
        """Return the Keras step/epoch position, or the update-budget position."""
        keys = ("epoch", "epochs", "steps_per_epoch")
        present = tuple(key in record for key in keys)
        if not any(present):
            return record["update"], self.total, None
        if not all(present):
            raise ValueError("invalid progress epoch metadata")
        epoch, epochs, steps = (record[key] for key in keys)
        if (
            type(epoch) is not int
            or type(epochs) is not int
            or type(steps) is not int
            or not 1 <= epoch <= epochs
            or steps < 1
            or epoch != (record["update"] - 1) // steps + 1
        ):
            raise ValueError("invalid progress epoch metadata")
        return (record["update"] - 1) % steps + 1, steps, (epoch, epochs)

    def _prepare_bar(self, record):
        """Create one retained bar per epoch, as Keras ``fit`` does."""
        step, steps, epoch = self._position(record)
        if epoch is not None and epoch != self.display_epoch:
            self.close()
            self.display_epoch = epoch
            self.display_started = time.monotonic()
            self.display_initial = step - 1
            if self.mode != "off":
                print(
                    self._label(f"Epoch {epoch[0]}/{epoch[1]} - {self._phase_title(self.phase)}"),
                    file=self.stream,
                    flush=True,
                )
        if self.mode == "tty" and self.bar is None:
            assert _tqdm is not None
            self.bar = _tqdm(
                total=steps,
                initial=step - 1,
                unit="step",
                file=self.stream,
                mininterval=1.0,
                dynamic_ncols=True,
                leave=True,
                bar_format="{n_fmt}/{total_fmt} {bar:20} {elapsed}<{remaining} {rate_fmt}{postfix}",
            )
            self.bar_kind = "training"
        return step, steps

    def stage(self, label, *, force=False):
        now = time.monotonic()
        if self.mode == "off":
            return
        changed = self.stage_name is None or label.split(" ")[0] != self.stage_name.split(" ")[0]
        self.stage_name = label
        if self.bar:
            self.bar.set_description(self._label(f"{self.phase} | {label}"), refresh=force)
        elif force or changed or now - self.last_line >= 15:
            print(self._label(label), file=self.stream, flush=True)
            self.last_line = now

    def update(self, record, *, running_mean, best, stale, latest_validation):
        values = [record["loss"], record["learning_rate"], running_mean]
        if best is not None:
            values.append(best)
        if (
            any(not math.isfinite(v) for v in values)
            or type(stale) is not int
            or stale < 0
            or not self.current < record["update"] <= self.total
        ):
            raise ValueError("invalid progress metric/update/stale counter")
        self.current = record["update"]
        step, steps = self._prepare_bar(record)
        metrics = [f"loss: {running_mean:.6g}"]
        metrics.append(f"learning_rate: {record['learning_rate']:.3g}")
        suffix = " - " + " - ".join(metrics)
        if self.bar:
            self.bar.set_description("", refresh=False)
            self.bar.set_postfix_str(suffix, refresh=False)
            self.bar.update(step - self.bar.n)
        elif self.mode != "off":
            elapsed = time.monotonic() - self.display_started
            rate = (step - self.display_initial) / max(elapsed, 1e-9)
            eta = (steps - step) / rate if rate else 0
            if (
                self.current == self.initial + 1
                or self.current == self.total
                or "validation" in record
                or "epoch_coverage" in record
                or time.monotonic() - self.last_line >= 15
            ):
                print(
                    f"{self._label(self.phase)} {step}/{steps}{suffix} "
                    f"elapsed={elapsed:.1f}s rate={rate:.3g}/s ETA={eta:.1f}s",
                    file=self.stream,
                    flush=True,
                )
                self.last_line = time.monotonic()

    def begin_validation(self, origin, *, index, total, batches, kind="validation"):
        """Close the completed training bar and start an explicit validation-batch bar."""
        if (
            not isinstance(origin, str)
            or not origin
            or type(index) is not int
            or type(total) is not int
            or not 1 <= index <= total
            or type(batches) is not int
            or batches < 1
            or not isinstance(kind, str)
            or not kind.strip()
        ):
            raise ValueError("invalid validation progress position")
        self.close()
        self.validation_position = (index, total, batches)
        epoch = (
            f"Epoch {self.display_epoch[0]}/{self.display_epoch[1]}"
            if self.display_epoch is not None
            else self._phase_title(self.phase)
        )
        if self.mode != "off":
            title = kind.strip().capitalize()
            if total > 1:
                title += f" {index}/{total}"
            print(
                self._label(f"{epoch} — {title} after {origin}"),
                file=self.stream,
                flush=True,
            )
        self.display_started = time.monotonic()
        self.last_line = 0.0
        if self.mode == "tty":
            assert _tqdm is not None
            self.bar = _tqdm(
                total=batches,
                initial=0,
                unit="batch",
                file=self.stream,
                mininterval=1.0,
                dynamic_ncols=True,
                leave=True,
                bar_format="{n_fmt}/{total_fmt} {bar:20} {elapsed}<{remaining} {rate_fmt}{postfix}",
            )
            self.bar_kind = "validation"

    def validation_batch(self, completed, total):
        """Advance the explicit validation bar after one inference batch completes."""
        if (
            self.validation_position is None
            or type(completed) is not int
            or type(total) is not int
            or not 1 <= completed <= total
            or total != self.validation_position[2]
        ):
            raise ValueError("invalid validation batch progress")
        if self.bar is not None:
            self.bar.update(completed - self.bar.n)
        elif self.mode != "off" and (
            completed == 1 or completed == total or time.monotonic() - self.last_line >= 15
        ):
            elapsed = time.monotonic() - self.display_started
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / rate if rate else 0.0
            print(
                self._label(
                    f"validation batch {completed}/{total} - elapsed={elapsed:.1f}s "
                    f"rate={rate:.3g}/s ETA={eta:.1f}s"
                ),
                file=self.stream,
                flush=True,
            )
            self.last_line = time.monotonic()

    def validation_status(self, label):
        if self.mode == "off":
            return
        if self.bar is not None:
            self.bar.set_postfix_str(f" - {label}", refresh=True)
        else:
            print(self._label(label), file=self.stream, flush=True)

    def end_validation(self):
        self.close()
        self.validation_position = None

    def validation_summary(self, latest_validation, *, train_loss, best, stale, patience):
        if self.mode == "off":
            return
        if (
            not all(
                isinstance(value, int | float) and math.isfinite(value)
                for value in (train_loss, best)
            )
            or type(stale) is not int
            or type(patience) is not int
            or not 0 <= stale <= patience
        ):
            raise ValueError("invalid validation summary")
        metric_name = latest_validation.get("metric_name", "metric")
        fields = [f"loss: {train_loss:.6g}"]
        if latest_validation.get("nll") is not None:
            fields.append(f"val_loss: {latest_validation['nll']:.6g}")
        fields.extend(
            [
                f"val_{metric_name}: {latest_validation['metric']:.6g}",
                f"best_val_{metric_name}: {best:.6g}",
                f"wait: {stale}/{patience}",
            ]
        )
        epoch = (
            f"Epoch {self.display_epoch[0]}/{self.display_epoch[1]} complete"
            if self.display_epoch is not None
            else "Validation complete"
        )
        print(self._label(epoch + " - " + " - ".join(fields)), file=self.stream)

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        self.bar_kind = None
