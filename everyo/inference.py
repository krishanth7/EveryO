"""Bounded, synchronous CPU inference for ordinary numeric EveryO models."""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.validation import validate_array, validate_predictions


class CPUInferenceRuntime:
    def __init__(self, model: Any, *, max_batch: int = 64, max_elements: int = 1_000_000):
        if max_batch <= 0 or max_elements <= 0:
            raise ValueError("resource limits must be positive")
        self.model = model
        self.max_batch = int(max_batch)
        self.max_elements = int(max_elements)
        if hasattr(model, "eval"):
            model.eval()

    def predict(self, inputs: Any) -> np.ndarray:
        array = validate_array(inputs, max_elements=self.max_elements)
        if array.ndim < 2 or array.shape[0] > self.max_batch:
            raise ValueError("input must be a nonempty batch within max_batch")
        result = self.model(array)
        values = np.asarray(result.data if hasattr(result, "data") else result)
        return validate_predictions(
            values, batch_size=len(array), max_elements=self.max_elements
        ).copy()

    def predict_batched(self, inputs: Any, *, batch_size: int | None = None) -> np.ndarray:
        array = validate_array(inputs, max_elements=self.max_elements)
        if batch_size is None:
            batch_size = min(32, self.max_batch)
        if array.ndim < 2 or not 0 < batch_size <= self.max_batch:
            raise ValueError("batch_size must be within runtime limits")
        results = [
            self.predict(array[i : i + batch_size]) for i in range(0, len(array), batch_size)
        ]
        return np.concatenate(results, axis=0)
