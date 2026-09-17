"""Callbacks that observe or steer the training loop.

A callback receives hooks at well defined points and may set
``trainer.stop_training = True`` to end training early.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from everyo._logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from everyo.training.trainer import Trainer

__all__ = [
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "CSVLogger",
    "LearningRateScheduler",
]

_LOGGER = get_logger(__name__)


class Callback:
    """Base class; override only the hooks you need."""

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after the last epoch."""

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Called at the start of each epoch (``epoch`` is 1-based)."""

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Called after each epoch with that epoch's metrics."""

    def on_batch_begin(self, trainer: Trainer, batch: int) -> None:
        """Called before each training batch."""

    def on_batch_end(self, trainer: Trainer, batch: int, logs: dict[str, float]) -> None:
        """Called after each training batch."""


class CallbackList(Callback):
    """Dispatch hooks to a list of callbacks, in order."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])
        for index, callback in enumerate(self.callbacks):
            if not isinstance(callback, Callback):
                raise TypeError(
                    f"Callback {index} is a {type(callback).__name__}; it must "
                    "subclass everyo.training.callbacks.Callback."
                )

    def on_train_begin(self, trainer: Trainer) -> None:
        """Forward the hook to every callback."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Trainer) -> None:
        """Forward the hook to every callback."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Forward the hook to every callback."""
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Forward the hook to every callback."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer: Trainer, batch: int) -> None:
        """Forward the hook to every callback."""
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch)

    def on_batch_end(self, trainer: Trainer, batch: int, logs: dict[str, float]) -> None:
        """Forward the hook to every callback."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch, logs)


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to watch, e.g. ``"val_loss"``.
        patience: Number of epochs without improvement to tolerate.
        min_delta: Minimum change that counts as an improvement.
        mode: ``"min"`` (lower is better) or ``"max"``.
        restore_best_weights: Restore the best parameters when stopping.
        verbose: Print a line when training stops and when weights are
            restored. Without it the restore is invisible, and the metrics
            reported after ``fit`` look inconsistent with the last epoch —
            because they describe the restored best model, not the last one.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        if int(patience) < 0:
            raise ValueError(f"patience must be non-negative, got {patience}.")
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = bool(verbose)
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self._best_state: dict[str, Any] | None = None

    def on_train_begin(self, trainer: Trainer) -> None:
        """Reset the internal counters."""
        self.best = math.inf if self.mode == "min" else -math.inf
        self.wait = 0
        self.stopped_epoch = 0
        self._best_state = None

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Track the monitored metric and stop when patience runs out."""
        current = logs.get(self.monitor)
        if current is None:
            _LOGGER.warning(
                "EarlyStopping is monitoring '%s', which was not recorded this "
                "epoch. Available metrics: %s",
                self.monitor,
                ", ".join(sorted(logs)) or "none",
            )
            return

        if _is_improvement(current, self.best, self.mode, self.min_delta):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self._best_state = trainer.model.state_dict()
            return

        self.wait += 1
        if self.wait > self.patience:
            self.stopped_epoch = epoch
            trainer.stop_training = True
            if self.restore_best_weights and self._best_state is not None:
                trainer.model.load_state_dict(self._best_state)
                message = (
                    f"Early stopping at epoch {epoch}: no improvement in "
                    f"{self.monitor} for {self.patience + 1} epochs. Restored "
                    f"the best weights from epoch {self.best_epoch} "
                    f"({self.monitor}={self.best:.6f}) — metrics measured after "
                    f"fit() describe that model, not the last epoch."
                )
            else:
                message = (
                    f"Early stopping at epoch {epoch}: no improvement in "
                    f"{self.monitor} for {self.patience + 1} epochs."
                )
            _LOGGER.info("%s", message)
            if self.verbose:
                print(message, flush=True)


class ModelCheckpoint(Callback):
    """Save the model during training.

    Args:
        path: Destination file (``.evo`` archive).
        monitor: Metric used when ``save_best_only`` is set.
        mode: ``"min"`` or ``"max"``.
        save_best_only: Only overwrite the file when the metric improves.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = bool(save_best_only)
        self.best = math.inf if mode == "min" else -math.inf
        self.saved_epochs: list[int] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Write a checkpoint when appropriate."""
        from everyo.serialization.save import save

        if not self.save_best_only:
            save(trainer.model, self.path, metadata={"epoch": epoch, **logs})
            self.saved_epochs.append(epoch)
            return

        current = logs.get(self.monitor)
        if current is None:
            _LOGGER.warning(
                "ModelCheckpoint is monitoring '%s', which was not recorded.",
                self.monitor,
            )
            return
        if _is_improvement(current, self.best, self.mode, 0.0):
            self.best = current
            save(trainer.model, self.path, metadata={"epoch": epoch, **logs})
            self.saved_epochs.append(epoch)
            _LOGGER.info("Saved checkpoint to %s (%s=%.6f).", self.path, self.monitor, current)


class ProgressLogger(Callback):
    """Print a one-line summary per epoch.

    Args:
        every: Print every ``n`` epochs (the first and last are always printed).
        stream: Set to ``False`` to disable printing entirely.
    """

    def __init__(self, every: int = 1, *, stream: bool = True) -> None:
        self.every = max(int(every), 1)
        self.stream = bool(stream)
        self._start = 0.0

    def on_train_begin(self, trainer: Trainer) -> None:
        """Record the wall-clock start time."""
        self._start = time.perf_counter()

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Print the epoch summary."""
        if not self.stream:
            return
        if epoch != 1 and epoch != trainer.total_epochs and epoch % self.every:
            return
        parts = [f"Epoch {epoch}/{trainer.total_epochs}"]
        for key, value in logs.items():
            if key in ("epoch", "time"):
                continue
            parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
        parts.append(f"time={logs.get('time', 0.0):.2f}s")
        print(" | ".join(parts), flush=True)

    def on_train_end(self, trainer: Trainer) -> None:
        """Print the total training duration."""
        if self.stream:
            print(
                f"Training finished in {time.perf_counter() - self._start:.2f}s.",
                flush=True,
            )


class CSVLogger(Callback):
    """Append epoch metrics to a CSV file as training proceeds."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, float]) -> None:
        """Rewrite the CSV with the history recorded so far."""
        trainer.history.to_csv(self.path)


class LearningRateScheduler(Callback):
    """Set the optimizer learning rate from a function of the epoch.

    Args:
        schedule: Callable ``(epoch, current_lr) -> new_lr``.
    """

    def __init__(self, schedule) -> None:
        if not callable(schedule):
            raise TypeError("schedule must be callable, e.g. lambda epoch, lr: lr * 0.9")
        self.schedule = schedule

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Update the optimizer learning rate for this epoch."""
        new_lr = float(self.schedule(epoch, trainer.optimizer.lr))
        if new_lr <= 0:
            raise ValueError(f"The schedule produced a non-positive learning rate: {new_lr}.")
        trainer.optimizer.lr = new_lr
