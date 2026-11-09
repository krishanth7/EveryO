"""Checkpoint model, optimizer and NumPy RNG without executing pickle code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from everyo.serialization import load, save


def _optimizer_snapshot(optimizer: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    names: dict[str, np.ndarray] = {}
    parameter_states = []
    for index, parameter in enumerate(optimizer.parameters):
        state = optimizer.state.get(id(parameter), {})
        keys = []
        for key, value in state.items():
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiub":
                raise ValueError("optimizer state must contain numeric arrays")
            storage_key = f"{index}_{key}"
            names[storage_key] = value
            keys.append(key)
        parameter_states.append(keys)
    return {
        "type": type(optimizer).__name__,
        "step_count": optimizer.step_count,
        "config": optimizer.get_config(),
        "state_keys": parameter_states,
    }, names


def save_training_checkpoint(
    model: Any,
    optimizer: Any,
    path: str | Path,
    *,
    rng: np.random.Generator | None = None,
    tokenizer: dict[str, Any] | None = None,
) -> Path:
    """Write a .evo model and .train.npz companion; avoid untrusted pickle."""
    target = Path(path)
    if target.suffix != ".evo":
        raise ValueError("training checkpoints must use a .evo extension")
    meta, arrays = _optimizer_snapshot(optimizer)
    meta["rng_state"] = rng.bit_generator.state if rng is not None else None
    meta["tokenizer"] = tokenizer
    metadata = json.dumps(meta, allow_nan=False)
    temp = target.with_suffix(".train.npz")
    temp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temp, __meta__=np.array(metadata), **arrays)
    try:
        return save(model, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def load_training_checkpoint(
    path: str | Path,
    optimizer: Any,
    *,
    model: Any | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    target = Path(path)
    companion = target.with_suffix(".train.npz")
    if not companion.is_file():
        raise ValueError("missing optimizer checkpoint companion")
    try:
        with np.load(companion, allow_pickle=False) as data:
            metadata = json.loads(str(data["__meta__"].item()))
            if metadata["type"] != type(optimizer).__name__ or len(metadata["state_keys"]) != len(
                optimizer.parameters
            ):
                raise ValueError("optimizer type or parameter count mismatch")
            if metadata["config"] != json.loads(json.dumps(optimizer.get_config())):
                raise ValueError("optimizer configuration mismatch")
            staged = []
            for index, keys in enumerate(metadata["state_keys"]):
                state = {}
                for key in keys:
                    array = np.asarray(data[f"{index}_{key}"])
                    if (
                        array.shape != optimizer.parameters[index].shape
                        or not np.isfinite(array).all()
                    ):
                        raise ValueError("invalid optimizer state shape or values")
                    state[key] = array.copy()
                staged.append(state)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("corrupt training checkpoint") from exc
    restored = load(target, model=model)
    optimizer.state.clear()
    for param, state in zip(optimizer.parameters, staged):
        if state:
            optimizer.state[id(param)] = state
    optimizer.step_count = int(metadata["step_count"])
    if rng is not None and metadata["rng_state"] is not None:
        rng.bit_generator.state = metadata["rng_state"]
    return restored, metadata["tokenizer"]
