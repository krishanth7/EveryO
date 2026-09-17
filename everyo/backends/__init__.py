"""Computation backends.

* :mod:`everyo.backends.numpy_backend` is the always-available CPU reference.
* :mod:`everyo.backends.tensorflow_backend` is an optional integration used for
  cross-checking results and for hardware-accelerated reference training.

Importing this package never imports TensorFlow.
"""

from __future__ import annotations

from typing import Any

from everyo.backends import numpy_backend

__all__ = ["numpy_backend", "available_backends", "describe_backends"]


def available_backends() -> list[str]:
    """Return the names of backends usable in this process."""
    names = ["numpy"]
    from everyo.backends import tensorflow_backend

    if tensorflow_backend.is_available():
        names.append("tensorflow")
    from everyo.cuda.availability import is_available as cuda_available

    if cuda_available():
        names.append("cuda")
    return names


def describe_backends() -> dict[str, Any]:
    """Return a JSON-serialisable description of every backend."""
    from everyo.backends import tensorflow_backend
    from everyo.cuda.availability import runtime_info

    return {
        "numpy": numpy_backend.describe(),
        "tensorflow": tensorflow_backend.describe(),
        "cuda": runtime_info(),
    }
