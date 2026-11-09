"""Deterministic CPU robustness and failure-mode qualification utilities.

Noise and metamorphic checks are diagnostic stress tests, not certified
adversarial defenses or security guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from everyo.validation import validate_array


def gaussian_perturbation(
    inputs: Any, *, sigma: float, seed: int = 0, clip: tuple[float, float] | None = None
) -> np.ndarray:
    data = validate_array(inputs).astype(np.float64, copy=True)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if clip is not None and (len(clip) != 2 or not np.isfinite(clip).all() or clip[0] > clip[1]):
        raise ValueError("clip must specify finite lower/upper bounds")
    noisy = data + np.random.default_rng(seed).normal(0, sigma, data.shape)
    if clip is not None:
        noisy = np.clip(noisy, *clip)
    return noisy


def feature_dropout(
    inputs: Any, *, probability: float, seed: int = 0, replacement: float = 0.0
) -> np.ndarray:
    data = validate_array(inputs).astype(np.float64, copy=True)
    if not np.isfinite(probability) or not 0 <= probability <= 1 or not np.isfinite(replacement):
        raise ValueError("invalid dropout probability or replacement")
    mask = np.random.default_rng(seed).random(data.shape) < probability
    data[mask] = replacement
    return data


def bounded_prediction(
    fn: Callable[[np.ndarray], Any], inputs: Any, *, max_elements: int = 1_000_000
) -> np.ndarray:
    """Enforce finite batch-aligned outputs from a callable."""
    data = validate_array(inputs, max_elements=max_elements)
    if data.ndim < 2:
        raise ValueError("inputs must contain a batch dimension")
    raw = fn(data)
    output = validate_array(raw.data if hasattr(raw, "data") else raw, max_elements=max_elements)
    if output.ndim < 1 or len(output) != len(data):
        raise ValueError("model returned mismatched output batch")
    return output.copy()


def repeatability_check(
    fn: Callable[[np.ndarray], Any], inputs: Any, *, tolerance: float = 1e-10
) -> dict[str, float | bool]:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    first = bounded_prediction(fn, inputs)
    second = bounded_prediction(fn, inputs)
    if first.shape != second.shape:
        return {"passed": False, "max_absolute_error": float("inf")}
    difference = float(np.max(np.abs(first - second)))
    return {"passed": difference <= tolerance, "max_absolute_error": difference}


def batch_consistency_check(
    fn: Callable[[np.ndarray], Any], inputs: Any, *, tolerance: float = 1e-8
) -> dict[str, float | bool]:
    """Compare one-batch inference with processing each row independently."""
    data = validate_array(inputs)
    if data.ndim < 2 or len(data) < 2:
        raise ValueError("at least two input rows are required")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    together = bounded_prediction(fn, data)
    separately = np.concatenate([bounded_prediction(fn, row[None, ...]) for row in data], axis=0)
    if together.shape != separately.shape:
        return {"passed": False, "max_absolute_error": float("inf")}
    error = float(np.max(np.abs(together - separately)))
    return {"passed": error <= tolerance, "max_absolute_error": error}


@dataclass(frozen=True)
class RobustnessReport:
    baseline_accuracy: float
    noisy_accuracy: float
    dropout_accuracy: float
    relative_degradation: float
    passed: bool


def robustness_report(
    fn: Callable[[np.ndarray], Any],
    inputs: Any,
    labels: Any,
    *,
    sigma: float = 0.05,
    dropout: float = 0.05,
    seed: int = 0,
    max_degradation: float = 0.20,
) -> RobustnessReport:
    data = validate_array(inputs)
    expected = np.asarray(labels)
    if data.ndim < 2 or expected.shape != (len(data),) or expected.dtype.kind not in "iu":
        raise ValueError("labels must be integer indices aligned with inputs")
    if not np.isfinite(max_degradation) or not 0 <= max_degradation <= 1:
        raise ValueError("max_degradation must be in [0,1]")

    def accuracy(x: np.ndarray) -> float:
        scores = bounded_prediction(fn, x)
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError("classifier must return (batch, classes) logits")
        if np.any(expected < 0) or np.any(expected >= scores.shape[1]):
            raise ValueError("label index out of range")
        return float(np.mean(scores.argmax(axis=1) == expected))

    baseline = accuracy(data)
    noisy = accuracy(gaussian_perturbation(data, sigma=sigma, seed=seed))
    dropped = accuracy(feature_dropout(data, probability=dropout, seed=seed))
    reduction = baseline - min(noisy, dropped)
    return RobustnessReport(
        baseline, noisy, dropped, float(reduction), bool(reduction <= max_degradation)
    )


def stress_inputs(inputs: Any, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Reproducible, finite variants for integration-test fault injection."""
    data = validate_array(inputs)
    return {
        "unchanged": data.copy(),
        "zero": np.zeros_like(data),
        "small_noise": gaussian_perturbation(data, sigma=0.01, seed=seed),
        "dropout": feature_dropout(data, probability=0.1, seed=seed),
    }
