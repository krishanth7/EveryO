"""Privacy-conscious, CPU-only model-data and prediction monitoring.

Only aggregate statistics are retained. Drift thresholds are heuristics, not
statistical proof of changing real-world model quality.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


def _matrix(values: Any, *, columns: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0 or raw.dtype.kind not in "fiub":
        raise ValueError("features must be a nonempty numeric (samples,features) matrix")
    if columns is not None and raw.shape[1] != columns:
        raise ValueError("feature dimension mismatch")
    if not np.isfinite(raw).all():
        raise ValueError("feature values must be finite")
    return raw.astype(np.float64, copy=True)


@dataclass(frozen=True)
class ReferenceProfile:
    count: int
    means: tuple[float, ...]
    stddevs: tuple[float, ...]
    histogram: tuple[tuple[float, ...], ...]
    edges: tuple[tuple[float, ...], ...]

    @property
    def features(self) -> int:
        return len(self.means)


def build_reference_profile(values: Any, *, bins: int = 8) -> ReferenceProfile:
    samples = _matrix(values)
    if bins < 2 or bins > 128:
        raise ValueError("bins must be between 2 and 128")
    edges, histograms = [], []
    for col in samples.T:
        minimum, maximum = float(col.min()), float(col.max())
        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5
        edge = np.linspace(minimum, maximum, bins + 1)
        counts = np.histogram(np.clip(col, edge[0], edge[-1]), bins=edge)[0]
        edges.append(tuple(float(v) for v in edge))
        histograms.append(tuple(float(v) for v in counts / len(col)))
    return ReferenceProfile(
        len(samples),
        tuple(float(v) for v in samples.mean(axis=0)),
        tuple(float(v) for v in samples.std(axis=0)),
        tuple(histograms),
        tuple(edges),
    )


def population_stability_index(
    reference: ReferenceProfile, values: Any, *, epsilon: float = 1e-6
) -> np.ndarray:
    """Per-feature PSI; reference histogram edges are fixed for comparability."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    batch = _matrix(values, columns=reference.features)
    scores = []
    for i, col in enumerate(batch.T):
        edges = np.asarray(reference.edges[i])
        actual = np.histogram(np.clip(col, edges[0], edges[-1]), bins=edges)[0] / len(col)
        baseline = np.asarray(reference.histogram[i])
        p, q = np.maximum(baseline, epsilon), np.maximum(actual, epsilon)
        scores.append(float(np.sum((q - p) * np.log(q / p))))
    return np.asarray(scores)


def mean_shift_scores(
    reference: ReferenceProfile, values: Any, *, floor: float = 1e-6
) -> np.ndarray:
    if floor <= 0:
        raise ValueError("floor must be positive")
    batch = _matrix(values, columns=reference.features)
    return np.abs(batch.mean(axis=0) - reference.means) / np.maximum(reference.stddevs, floor)


def prediction_entropy(probabilities: Any) -> float:
    """Mean categorical uncertainty; rows must already be valid probabilities."""
    scores = _matrix(probabilities)
    if np.any(scores < 0) or not np.allclose(scores.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must be nonnegative and sum to one")
    return float(-(scores * np.log(np.maximum(scores, 1e-12))).sum(axis=1).mean())


class RollingAccuracy:
    """Bounded label-feedback accuracy; never retains the raw input features."""

    def __init__(self, window: int = 1000):
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")
        self.correct: deque[bool] = deque(maxlen=window)

    def update(self, predictions: Any, targets: Any) -> float:
        pred, expected = np.asarray(predictions), np.asarray(targets)
        if (
            pred.ndim != 1
            or expected.shape != pred.shape
            or pred.dtype.kind not in "iu"
            or expected.dtype.kind not in "iu"
        ):
            raise ValueError("predictions and labels must be aligned integer vectors")
        for result in pred == expected:
            self.correct.append(bool(result))
        return self.accuracy

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.correct)) if self.correct else float("nan")


def drift_report(
    reference: ReferenceProfile,
    values: Any,
    *,
    psi_threshold: float = 0.2,
    shift_threshold: float = 3.0,
) -> dict[str, Any]:
    if psi_threshold <= 0 or shift_threshold <= 0:
        raise ValueError("thresholds must be positive")
    psi = population_stability_index(reference, values)
    shifts = mean_shift_scores(reference, values)
    alerts = [int(i) for i in np.where((psi > psi_threshold) | (shifts > shift_threshold))[0]]
    return {
        "drift_detected": bool(alerts),
        "alert_features": alerts,
        "psi": psi.tolist(),
        "mean_shift": shifts.tolist(),
        "samples": len(_matrix(values)),
    }


def monitoring_summary(
    reference: ReferenceProfile, values: Any, probabilities: Any | None = None
) -> dict[str, Any]:
    report = drift_report(reference, values)
    if probabilities is not None:
        report["prediction_entropy"] = prediction_entropy(probabilities)
    return report
