"""Evaluation metrics operating on NumPy arrays or tensors."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from everyo.core.tensor import Tensor
from everyo.exceptions import EveryOShapeError

__all__ = [
    "accuracy",
    "binary_accuracy",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "confusion_matrix",
    "get_metric",
    "METRICS",
]


def _array(values: Any) -> np.ndarray:
    return np.asarray(values.data if isinstance(values, Tensor) else values)


def _labels(values: np.ndarray) -> np.ndarray:
    """Reduce probabilities/one-hot rows to integer class labels."""
    if values.ndim > 1 and values.shape[-1] > 1:
        return values.argmax(axis=-1)
    return values.reshape(-1)


def accuracy(prediction: Any, target: Any) -> float:
    """Fraction of correctly classified samples.

    Accepts logits, probabilities, one-hot rows or integer labels on either
    side and reduces both to class labels before comparing.
    """
    pred = _labels(_array(prediction))
    true = _labels(_array(target))
    if pred.shape != true.shape:
        raise EveryOShapeError(
            f"accuracy() compared {pred.shape} predictions against {true.shape} "
            "targets; they must describe the same number of samples."
        )
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == true))


def binary_accuracy(prediction: Any, target: Any, threshold: float = 0.5) -> float:
    """Accuracy for a single-output binary classifier."""
    pred = (_array(prediction).reshape(-1) >= threshold).astype(np.int64)
    true = (_array(target).reshape(-1) >= threshold).astype(np.int64)
    if pred.shape != true.shape:
        raise EveryOShapeError(
            f"binary_accuracy() compared {pred.shape} predictions against {true.shape} targets."
        )
    return float(np.mean(pred == true)) if pred.size else 0.0


def mean_absolute_error(prediction: Any, target: Any) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(_array(prediction) - _array(target))))


def mean_squared_error(prediction: Any, target: Any) -> float:
    """Mean squared error."""
    difference = _array(prediction) - _array(target)
    return float(np.mean(difference * difference))


def r2_score(prediction: Any, target: Any) -> float:
    """Coefficient of determination; ``1.0`` is a perfect fit."""
    pred = _array(prediction).reshape(-1)
    true = _array(target).reshape(-1)
    total = float(np.sum((true - true.mean()) ** 2))
    if total == 0.0:
        return 0.0
    residual = float(np.sum((true - pred) ** 2))
    return 1.0 - residual / total


def confusion_matrix(prediction: Any, target: Any, num_classes: int | None = None) -> np.ndarray:
    """Return the ``(num_classes, num_classes)`` confusion matrix.

    Rows are true classes, columns are predicted classes.
    """
    pred = _labels(_array(prediction)).astype(np.int64)
    true = _labels(_array(target)).astype(np.int64)
    classes = int(num_classes) if num_classes else int(max(pred.max(), true.max()) + 1)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


#: Metrics addressable by name from :class:`~everyo.training.trainer.Trainer`.
METRICS: dict[str, Callable[[Any, Any], float]] = {
    "accuracy": accuracy,
    "binary_accuracy": binary_accuracy,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r2": r2_score,
}


def get_metric(name: str) -> Callable[[Any, Any], float]:
    """Look up a metric function by name."""
    key = str(name).lower()
    if key not in METRICS:
        raise ValueError(
            f"Unknown metric {name!r}. Available metrics: {', '.join(sorted(METRICS))}."
        )
    return METRICS[key]
