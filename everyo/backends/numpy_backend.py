"""CPU backend built on NumPy.

This module is the reference implementation of every numerical kernel EveryO
needs.  It is deliberately thin — NumPy already does the heavy lifting — but
keeping the kernels behind named functions gives the CUDA backend a precise
contract to match, and gives tests a CPU baseline to compare GPU results
against.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NAME",
    "add",
    "subtract",
    "multiply",
    "divide",
    "matmul",
    "relu",
    "relu_backward",
    "sum_all",
    "is_available",
    "describe",
]

NAME = "numpy"


def is_available() -> bool:
    """The NumPy backend is always available."""
    return True


def describe() -> dict[str, str]:
    """Return a short description of the backend for ``everyo info``."""
    return {"name": NAME, "version": np.__version__}


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition with NumPy broadcasting."""
    return np.add(a, b)


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise subtraction with NumPy broadcasting."""
    return np.subtract(a, b)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication with NumPy broadcasting."""
    return np.multiply(a, b)


def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division with NumPy broadcasting."""
    return np.divide(a, b)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product."""
    return np.matmul(a, b)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit, ``max(x, 0)``."""
    return np.maximum(x, 0)


def relu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gradient of :func:`relu` with respect to its input."""
    return grad * (x > 0)


def sum_all(x: np.ndarray) -> np.ndarray:
    """Sum every element of ``x`` into a 0-d array."""
    return np.asarray(x.sum())
