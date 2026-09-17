"""Feature preprocessing helpers.

All helpers work on NumPy arrays and return NumPy arrays, so they can be used
before data ever becomes a tensor.  The scalers follow a ``fit``/``transform``
pattern: statistics are computed on the training split only, which is what
keeps the evaluation honest.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.core.tensor import Tensor
from everyo.exceptions import EveryOShapeError

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "normalize",
    "standardize",
    "min_max_scale",
    "one_hot_encode",
    "shuffle_arrays",
]

_EPS = 1e-8


def _as_array(values: Any) -> np.ndarray:
    array = values.data if isinstance(values, Tensor) else np.asarray(values)
    return array.astype(np.float32, copy=False) if array.dtype != np.float64 else array


class StandardScaler:
    """Scale features to zero mean and unit variance.

    Example:
        >>> import numpy as np
        >>> from everyo.data.preprocessing import StandardScaler
        >>> scaler = StandardScaler().fit(np.array([[0.0], [2.0]]))
        >>> float(scaler.transform(np.array([[1.0]]))[0, 0])
        0.0
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, features: Any) -> StandardScaler:
        """Compute per-column mean and standard deviation."""
        array = _as_array(features)
        self.mean_ = array.mean(axis=0)
        self.std_ = array.std(axis=0)
        return self

    def transform(self, features: Any) -> np.ndarray:
        """Standardise ``features`` using the fitted statistics."""
        if self.mean_ is None or self.std_ is None:
            raise EveryOShapeError(
                "StandardScaler.transform() was called before fit(). Call "
                "fit(train_features) first."
            )
        array = _as_array(features)
        if array.shape[1:] != self.mean_.shape:
            raise EveryOShapeError(
                f"StandardScaler was fitted on features with shape "
                f"{(None, *self.mean_.shape)} but received {array.shape}."
            )
        return (array - self.mean_) / np.maximum(self.std_, _EPS)

    def fit_transform(self, features: Any) -> np.ndarray:
        """Fit on ``features`` and return the transformed array."""
        return self.fit(features).transform(features)

    def inverse_transform(self, features: Any) -> np.ndarray:
        """Undo :meth:`transform`."""
        if self.mean_ is None or self.std_ is None:
            raise EveryOShapeError("StandardScaler has not been fitted yet.")
        return _as_array(features) * np.maximum(self.std_, _EPS) + self.mean_


class MinMaxScaler:
    """Scale features into ``[feature_range[0], feature_range[1]]``."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        low, high = feature_range
        if low >= high:
            raise ValueError(f"feature_range must be increasing, got ({low}, {high}).")
        self.feature_range = (float(low), float(high))
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, features: Any) -> MinMaxScaler:
        """Record per-column minima and maxima."""
        array = _as_array(features)
        self.min_ = array.min(axis=0)
        self.max_ = array.max(axis=0)
        return self

    def transform(self, features: Any) -> np.ndarray:
        """Rescale ``features`` into the configured range."""
        if self.min_ is None or self.max_ is None:
            raise EveryOShapeError("MinMaxScaler.transform() was called before fit().")
        array = _as_array(features)
        span = np.maximum(self.max_ - self.min_, _EPS)
        low, high = self.feature_range
        return (array - self.min_) / span * (high - low) + low

    def fit_transform(self, features: Any) -> np.ndarray:
        """Fit on ``features`` and return the transformed array."""
        return self.fit(features).transform(features)


def standardize(features: Any, axis: int = 0) -> np.ndarray:
    """Return ``features`` with zero mean and unit variance along ``axis``."""
    array = _as_array(features)
    mean = array.mean(axis=axis, keepdims=True)
    std = array.std(axis=axis, keepdims=True)
    return (array - mean) / np.maximum(std, _EPS)


def min_max_scale(features: Any, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Rescale ``features`` column-wise into ``[low, high]``."""
    return MinMaxScaler((low, high)).fit_transform(features)


def normalize(features: Any, axis: int = 1, order: int = 2) -> np.ndarray:
    """Scale each row (or column) to unit norm."""
    array = _as_array(features)
    norm = np.linalg.norm(array, ord=order, axis=axis, keepdims=True)
    return array / np.maximum(norm, _EPS)


def one_hot_encode(labels: Any, num_classes: int | None = None) -> np.ndarray:
    """Convert integer labels into one-hot rows.

    Args:
        labels: Integer class indices.
        num_classes: Number of columns; inferred from the data when ``None``.
    """
    array = (
        np.asarray(labels.data if isinstance(labels, Tensor) else labels)
        .astype(np.int64)
        .reshape(-1)
    )
    if array.size == 0:
        return np.zeros((0, num_classes or 0), dtype=np.float32)
    if array.min() < 0:
        raise EveryOShapeError(f"Class labels must be non-negative, found {array.min()}.")
    classes = int(num_classes) if num_classes is not None else int(array.max()) + 1
    if array.max() >= classes:
        raise EveryOShapeError(f"Label {array.max()} is out of range for num_classes={classes}.")
    encoded = np.zeros((array.size, classes), dtype=np.float32)
    encoded[np.arange(array.size), array] = 1.0
    return encoded


def shuffle_arrays(*arrays: Any, seed: int | None = None) -> tuple[np.ndarray, ...]:
    """Shuffle several arrays with one shared permutation."""
    converted = [_as_array(array) for array in arrays]
    lengths = {len(array) for array in converted}
    if len(lengths) > 1:
        raise EveryOShapeError(
            f"All arrays must have the same length, got lengths {sorted(lengths)}."
        )
    order = np.random.default_rng(seed).permutation(lengths.pop() if lengths else 0)
    return tuple(array[order] for array in converted)
