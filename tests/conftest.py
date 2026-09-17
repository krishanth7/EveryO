"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator, so every test is reproducible."""
    return np.random.default_rng(1234)


@pytest.fixture
def small_model() -> eo.Sequential:
    """A tiny deterministic classifier used across several test modules."""
    return eo.Sequential(
        eo.Linear(4, 8, seed=0),
        eo.ReLU(),
        eo.Linear(8, 3, seed=1),
    )


@pytest.fixture
def classification_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A small, linearly separable 3-class problem."""
    features = rng.normal(size=(120, 4)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] > 0).astype(np.int64) + (features[:, 2] > 0.5).astype(
        np.int64
    )
    return features, np.clip(labels, 0, 2)


def numeric_gradient(fn, values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of a scalar function, for gradient checks."""
    gradient = np.zeros_like(values)
    iterator = np.nditer(values, flags=["multi_index"])
    while not iterator.finished:
        index = iterator.multi_index
        original = values[index]
        values[index] = original + epsilon
        plus = fn(values)
        values[index] = original - epsilon
        minus = fn(values)
        values[index] = original
        gradient[index] = (plus - minus) / (2.0 * epsilon)
        iterator.iternext()
    return gradient
