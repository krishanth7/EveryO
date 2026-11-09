"""Phase 15: bounded CPU batch planning and output-parity verification."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from everyo.robustness import bounded_prediction
from everyo.validation import validate_array


def plan_microbatches(samples: int, *, batch_size: int, max_batch: int = 64) -> list[slice]:
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be positive")
    if (
        not isinstance(batch_size, int)
        or not isinstance(max_batch, int)
        or not 0 < batch_size <= max_batch
    ):
        raise ValueError("batch_size must be within configured max_batch")
    return [
        slice(start, min(start + batch_size, samples)) for start in range(0, samples, batch_size)
    ]


def batched_cpu_predict(
    fn: Callable[[np.ndarray], Any],
    inputs: Any,
    *,
    batch_size: int = 32,
    max_batch: int = 64,
    max_elements: int = 1_000_000,
) -> np.ndarray:
    data = validate_array(inputs, max_elements=max_elements)
    if data.ndim < 2:
        raise ValueError("input must include batch and feature dimensions")
    slices = plan_microbatches(len(data), batch_size=batch_size, max_batch=max_batch)
    outputs = [bounded_prediction(fn, data[part], max_elements=max_elements) for part in slices]
    if any(chunk.shape[1:] != outputs[0].shape[1:] for chunk in outputs):
        raise ValueError("model returned inconsistent output shapes")
    result = np.concatenate(outputs, axis=0)
    if result.size > max_elements:
        raise ValueError("combined output exceeds element limit")
    return result


def cpu_parity_report(
    fn: Callable[[np.ndarray], Any], inputs: Any, *, batch_size: int = 32, tolerance: float = 1e-8
) -> dict[str, float | bool]:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    data = validate_array(inputs)
    full = bounded_prediction(fn, data)
    split = batched_cpu_predict(fn, data, batch_size=batch_size)
    if full.shape != split.shape:
        return {"passed": False, "max_absolute_error": float("inf")}
    error = float(np.max(np.abs(full - split)))
    return {"passed": error <= tolerance, "max_absolute_error": error}
