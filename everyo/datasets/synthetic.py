"""Synthetic datasets generated locally, with no downloads.

These generators cover the classic teaching problems: a linear regression, two
interleaving moons, Gaussian blobs and spirals.  Each takes a seed so results
are reproducible.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_regression",
    "make_blobs",
    "make_moons",
    "make_spirals",
    "make_xor",
]


def make_regression(
    *,
    n_samples: int = 200,
    n_features: int = 1,
    noise: float = 0.1,
    bias: float = 0.0,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linear regression problem ``y = X w + bias + noise``.

    Returns:
        ``(features, targets)`` with shapes ``(n_samples, n_features)`` and
        ``(n_samples, 1)``.
    """
    if n_samples <= 0 or n_features <= 0:
        raise ValueError("n_samples and n_features must be positive.")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    weights = rng.uniform(-3.0, 3.0, size=(n_features, 1)).astype(np.float32)
    targets = features @ weights + bias
    targets = targets + rng.normal(0.0, noise, size=targets.shape)
    return features, targets.astype(np.float32)


def make_blobs(
    *,
    n_samples: int = 300,
    n_features: int = 2,
    centers: int = 3,
    spread: float = 0.8,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate isotropic Gaussian clusters for multi-class classification."""
    if centers <= 0:
        raise ValueError("centers must be positive.")
    rng = np.random.default_rng(seed)
    means = rng.uniform(-6.0, 6.0, size=(centers, n_features))
    per_center = n_samples // centers
    features, labels = [], []
    for index in range(centers):
        count = per_center if index < centers - 1 else n_samples - per_center * (centers - 1)
        features.append(rng.normal(means[index], spread, size=(count, n_features)))
        labels.append(np.full(count, index))
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


def make_moons(
    *, n_samples: int = 300, noise: float = 0.15, seed: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half circles — a classic non-linear problem."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    outer = np.linspace(0, np.pi, half)
    inner = np.linspace(0, np.pi, n_samples - half)
    x = np.vstack(
        [
            np.column_stack([np.cos(outer), np.sin(outer)]),
            np.column_stack([1 - np.cos(inner), 0.5 - np.sin(inner)]),
        ]
    )
    y = np.concatenate([np.zeros(half), np.ones(n_samples - half)]).astype(np.int64)
    x = (x + rng.normal(0.0, noise, size=x.shape)).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order]


def make_spirals(
    *,
    n_samples: int = 300,
    classes: int = 3,
    noise: float = 0.15,
    turns: float = 1.5,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate interleaved spiral arms, one per class."""
    rng = np.random.default_rng(seed)
    per_class = n_samples // classes
    features, labels = [], []
    for index in range(classes):
        radius = np.linspace(0.1, 1.0, per_class)
        theta = (
            np.linspace(0.0, turns * 2 * np.pi, per_class)
            + index * 2 * np.pi / classes
            + rng.normal(0.0, noise, size=per_class)
        )
        features.append(np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]))
        labels.append(np.full(per_class, index))
    x = np.concatenate(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


def make_xor(
    *, n_samples: int = 400, noise: float = 0.25, seed: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the XOR problem, which no linear model can solve."""
    rng = np.random.default_rng(seed)
    corners = rng.integers(0, 2, size=(n_samples, 2)).astype(np.float32)
    labels = np.logical_xor(corners[:, 0] > 0.5, corners[:, 1] > 0.5).astype(np.int64)
    features = (corners * 2.0 - 1.0) + rng.normal(0.0, noise, size=corners.shape)
    return features.astype(np.float32), labels
