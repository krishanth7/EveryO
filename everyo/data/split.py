"""Dataset splitting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.data.dataset import Dataset, Subset
from everyo.exceptions import EveryOShapeError

__all__ = ["train_test_split", "random_split", "stratified_split"]


def train_test_split(
    *arrays: Any,
    test_size: float = 0.2,
    seed: int | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, ...]:
    """Split arrays into train and test parts.

    Args:
        *arrays: Arrays sharing the same first dimension.
        test_size: Fraction of samples held out, in ``(0, 1)``.
        seed: Seed for the shuffling generator.
        shuffle: Shuffle before splitting.

    Returns:
        ``(a_train, a_test, b_train, b_test, ...)`` following the input order.

    Example:
        >>> import numpy as np
        >>> from everyo.data.split import train_test_split
        >>> x_train, x_test = train_test_split(np.arange(10).reshape(10, 1), test_size=0.3, seed=0)
        >>> len(x_train), len(x_test)
        (7, 3)
    """
    if not arrays:
        raise ValueError("train_test_split() requires at least one array.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError(f"test_size must be a fraction strictly between 0 and 1, got {test_size}.")

    converted = [np.asarray(array.data if hasattr(array, "data") else array) for array in arrays]
    lengths = {len(array) for array in converted}
    if len(lengths) > 1:
        raise EveryOShapeError(
            f"All arrays must share the same number of samples, got {sorted(lengths)}."
        )
    total = lengths.pop()
    if total < 2:
        raise EveryOShapeError(f"Need at least 2 samples to split, got {total}.")

    order = np.random.default_rng(seed).permutation(total) if shuffle else np.arange(total)
    n_test = max(1, int(round(total * float(test_size))))
    n_test = min(n_test, total - 1)
    test_idx, train_idx = order[:n_test], order[n_test:]

    result: list[np.ndarray] = []
    for array in converted:
        result.extend((array[train_idx], array[test_idx]))
    return tuple(result)


def random_split(
    dataset: Dataset, fractions: tuple[float, ...], *, seed: int | None = None
) -> list[Subset]:
    """Split a dataset into subsets with the given fractions.

    Args:
        dataset: Dataset to split.
        fractions: Fractions that must sum to approximately 1.
        seed: Seed for the permutation.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {sum(fractions)}.")
    total = len(dataset)
    order = np.random.default_rng(seed).permutation(total)

    subsets: list[Subset] = []
    start = 0
    for index, fraction in enumerate(fractions):
        size = total - start if index == len(fractions) - 1 else int(round(total * fraction))
        subsets.append(Subset(dataset, order[start : start + size]))
        start += size
    return subsets


def stratified_split(
    features: Any,
    labels: Any,
    *,
    test_size: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split while preserving the class balance of ``labels``.

    Returns:
        ``(x_train, x_test, y_train, y_test)``.
    """
    x = np.asarray(features.data if hasattr(features, "data") else features)
    y = np.asarray(labels.data if hasattr(labels, "data") else labels)
    if len(x) != len(y):
        raise EveryOShapeError(
            f"features and labels must have equal length, got {len(x)} and {len(y)}."
        )
    flat = y.reshape(len(y), -1)
    keys = flat.argmax(axis=1) if flat.shape[1] > 1 else flat.reshape(-1)

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for value in np.unique(keys):
        members = np.flatnonzero(keys == value)
        rng.shuffle(members)
        n_test = max(1, int(round(len(members) * float(test_size))))
        n_test = min(n_test, max(len(members) - 1, 0))
        test_idx.extend(members[:n_test].tolist())
        train_idx.extend(members[n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
