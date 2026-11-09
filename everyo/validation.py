"""Explicit CPU inference guards; never coerce non-finite values silently."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_array(
    value: Any, *, shape: tuple[int | None, ...] | None = None, max_elements: int = 1_000_000
) -> np.ndarray:
    if not isinstance(max_elements, int) or max_elements <= 0:
        raise ValueError("max_elements must be positive")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiub":
        raise ValueError("numeric input is required")
    if raw.size == 0 or raw.size > max_elements:
        raise ValueError("input size is zero or exceeds the configured limit")
    if shape is not None and (
        raw.ndim != len(shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(raw.shape, shape)
        )
    ):
        raise ValueError(f"expected input shape {shape}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("input contains NaN or infinity")
    return raw


def validate_predictions(
    value: Any, *, batch_size: int, max_elements: int = 1_000_000
) -> np.ndarray:
    output = validate_array(value, max_elements=max_elements)
    if output.ndim < 1 or output.shape[0] != batch_size:
        raise ValueError("output batch dimension does not match input")
    return output
