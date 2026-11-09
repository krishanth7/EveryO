"""CPU-first advanced training controls: schedules, search and safe checkpoints."""

from __future__ import annotations

import itertools
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from everyo.nn.module import Module
from everyo.optim.optimizer import Optimizer

__all__ = [
    "WarmupCosineScheduler",
    "ReduceOnPlateauScheduler",
    "grid_search",
    "random_search",
    "MedianPruner",
    "TrainingAnomalyDetector",
    "clip_gradients",
    "save_training_checkpoint",
    "load_training_checkpoint",
]


class WarmupCosineScheduler:
    """Step-based linear warmup followed by cosine annealing to ``min_lr``."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        if total_steps < 1 or not 0 <= warmup_steps < total_steps:
            raise ValueError("require total_steps > warmup_steps >= 0")
        if not np.isfinite(min_lr) or not 0 <= min_lr <= optimizer.lr:
            raise ValueError("min_lr must be finite and between 0 and initial lr")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lr = optimizer.lr
        self.steps = 0

    def step(self) -> float:
        self.steps += 1
        step = min(self.steps, self.total_steps)
        if step <= self.warmup_steps:
            lr = self.base_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        self.optimizer.lr = float(lr)
        return self.optimizer.lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, data: Mapping[str, Any]) -> None:
        if any(
            data[key] != getattr(self, key)
            for key in ("base_lr", "total_steps", "warmup_steps", "min_lr")
        ):
            raise ValueError("scheduler configuration mismatch")
        steps = int(data["steps"])
        if not 0 <= steps <= self.total_steps:
            raise ValueError("invalid scheduler step count")
        self.steps = steps
        if steps:
            self.steps -= 1
            self.step()


class ReduceOnPlateauScheduler:
    """Reduce LR after ``patience`` consecutive non-improving observations."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        factor: float = 0.5,
        patience: int = 2,
        min_lr: float = 0.0,
        mode: str = "min",
    ) -> None:
        if not 0 < factor < 1 or patience < 0 or min_lr < 0 or mode not in {"min", "max"}:
            raise ValueError("invalid plateau scheduler parameters")
        self.optimizer, self.factor, self.patience, self.min_lr, self.mode = (
            optimizer,
            factor,
            patience,
            min_lr,
            mode,
        )
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad_epochs = 0

    def step(self, metric: float) -> float:
        if not np.isfinite(metric):
            raise ValueError("scheduler metric must be finite")
        improved = metric < self.best if self.mode == "min" else metric > self.best
        if improved:
            self.best, self.bad_epochs = float(metric), 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                self.optimizer.lr = max(self.min_lr, self.optimizer.lr * self.factor)
                self.bad_epochs = 0
        return self.optimizer.lr


def grid_search(space: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Yield every Cartesian combination in deterministic key order."""
    keys = sorted(space)
    values = [tuple(space[key]) for key in keys]
    if any(not value for value in values):
        raise ValueError("grid dimensions cannot be empty")
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def random_search(
    space: Mapping[str, Sequence[Any]], *, trials: int, seed: int = 0
) -> Iterator[dict[str, Any]]:
    """Reproducible sampling with replacement from discrete candidates."""
    if trials < 0:
        raise ValueError("trials must be nonnegative")
    keys = sorted(space)
    values = [tuple(space[key]) for key in keys]
    if any(not value for value in values):
        raise ValueError("search dimensions cannot be empty")
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        yield {key: choices[int(rng.integers(len(choices)))] for key, choices in zip(keys, values)}


class MedianPruner:
    """Prune poor trials compared with completed peer trials at the same step."""

    def __init__(self, *, mode: str = "min", min_trials: int = 3, warmup_steps: int = 1) -> None:
        if mode not in {"min", "max"} or min_trials < 1 or warmup_steps < 0:
            raise ValueError("invalid pruning configuration")
        self.mode, self.min_trials, self.warmup_steps = mode, min_trials, warmup_steps

    def should_prune(
        self, *, step: int, value: float, completed_peers: Sequence[Mapping[int, float]]
    ) -> bool:
        if not np.isfinite(value):
            raise ValueError("trial metric must be finite")
        if step < self.warmup_steps:
            return False
        peers = [peer[step] for peer in completed_peers if step in peer and np.isfinite(peer[step])]
        if len(peers) < self.min_trials:
            return False
        median = float(np.median(peers))
        return bool(value > median if self.mode == "min" else value < median)


class TrainingAnomalyDetector:
    """Fail fast on NaN/Inf metrics, gradients, and optionally excessive norms."""

    def __init__(self, *, max_grad_norm: float | None = None) -> None:
        if max_grad_norm is not None and (not np.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive")
        self.max_grad_norm = max_grad_norm

    def check(self, optimizer: Optimizer, *, loss: float | None = None) -> float:
        if loss is not None and not np.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        norm = gradient_norm(optimizer)
        if not np.isfinite(norm):
            raise FloatingPointError("non-finite gradient norm")
        if self.max_grad_norm is not None and norm > self.max_grad_norm:
            raise FloatingPointError("gradient norm exceeds configured limit")
        return norm


def gradient_norm(optimizer: Optimizer) -> float:
    return math.sqrt(
        sum(
            float(np.sum(np.asarray(grad, dtype=np.float64) ** 2))
            for _, grad in optimizer._pending()
        )
    )


def clip_gradients(optimizer: Optimizer, max_norm: float) -> float:
    """Clip in-place to global L2 norm; return pre-clipping norm."""
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    norm = gradient_norm(optimizer)
    if not np.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    if norm > max_norm:
        scale = max_norm / norm
        for parameter, grad in optimizer._pending():
            parameter.grad = grad * scale
    return norm


def save_training_checkpoint(
    model: Module,
    optimizer: Optimizer,
    path: str | Path,
    *,
    epoch: int,
    extras: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save parameters, buffers and optimizer moments without pickle."""
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    names = list(model.state_dict())
    for index, value in enumerate(model.state_dict().values()):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays are not supported")
        arrays[f"model_{index}"] = array
    opt_states: list[list[str]] = []
    for index, parameter in enumerate(optimizer.parameters):
        state = optimizer.state.get(id(parameter), {})
        keys = sorted(state)
        opt_states.append(keys)
        for key in keys:
            array = np.asarray(state[key])
            if array.dtype.hasobject:
                raise TypeError("optimizer object state is not supported")
            arrays[f"optim_{index}_{key}"] = array
    metadata = {
        "version": 1,
        "epoch": int(epoch),
        "names": names,
        "optimizer_class": type(optimizer).__name__,
        "optimizer_config": optimizer.get_config(),
        "step_count": optimizer.step_count,
        "opt_states": opt_states,
        "extras": dict(extras or {}),
    }
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    fd, temporary = tempfile.mkstemp(prefix=".everyo-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_training_checkpoint(
    model: Module, optimizer: Optimizer, path: str | Path
) -> dict[str, Any]:
    """Restore a compatible model/optimizer and return epoch plus user extras.

    All shapes, keys and optimizer type are checked before mutating state.
    Checkpoints must come from trusted storage; ``allow_pickle=False`` is used.
    """
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata["version"] != 1 or metadata["optimizer_class"] != type(optimizer).__name__:
            raise ValueError("checkpoint format or optimizer type mismatch")
        saved_config = dict(metadata["optimizer_config"])
        current_config = dict(optimizer.get_config())
        # Learning rate is expected to change under scheduling; all other
        # hyperparameters must still describe the same optimizer.
        saved_config.pop("lr", None)
        current_config.pop("lr", None)
        if json.dumps(saved_config, sort_keys=True) != json.dumps(current_config, sort_keys=True):
            raise ValueError("checkpoint optimizer hyperparameter mismatch")
        current = model.state_dict()
        if metadata["names"] != list(current) or len(metadata["opt_states"]) != len(
            optimizer.parameters
        ):
            raise ValueError("checkpoint model/optimizer layout mismatch")
        restored = {}
        for index, (name, existing) in enumerate(current.items()):
            array = archive[f"model_{index}"]
            if array.shape != existing.shape or array.dtype != existing.dtype:
                raise ValueError(f"checkpoint tensor mismatch: {name}")
            restored[name] = array.copy()
        states = []
        for index, (parameter, keys) in enumerate(
            zip(optimizer.parameters, metadata["opt_states"])
        ):
            state = {}
            for key in keys:
                value = archive[f"optim_{index}_{key}"]
                if value.ndim != 0 and value.shape != parameter.shape:
                    raise ValueError("checkpoint optimizer tensor shape mismatch")
                state[key] = value.copy()
            states.append(state)
    model.load_state_dict(restored)
    optimizer.state = {
        id(param): state for param, state in zip(optimizer.parameters, states) if state
    }
    optimizer.step_count = int(metadata["step_count"])
    optimizer.lr = float(metadata["optimizer_config"]["lr"])
    return {"epoch": int(metadata["epoch"]), "extras": metadata["extras"]}
