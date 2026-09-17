"""Python side of the optional CUDA backend.

Every function here mirrors a kernel in ``cuda/src`` and has the same signature
as its NumPy counterpart in :mod:`everyo.backends.numpy_backend`.  When the
native extension is missing the functions either fall back to NumPy (default)
or raise :class:`~everyo.exceptions.EveryOCudaError` when ``strict=True``.

Buffers are plain NumPy arrays: the binding layer copies host memory to the
device, launches the kernel, and copies the result back.  That is not the
fastest possible design, but it keeps the data model simple and makes the
CPU/GPU comparison in ``benchmarks/`` honest.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from everyo._logging import get_logger
from everyo.backends import numpy_backend
from everyo.cuda.availability import get_extension, is_available, unavailable_reason
from everyo.exceptions import EveryOCudaError

__all__ = [
    "KERNEL_NAMES",
    "get_kernel",
    "add",
    "multiply",
    "matmul",
    "relu",
    "sum_all",
]

_LOGGER = get_logger(__name__)

#: Kernels implemented in ``cuda/src`` and exported by the bindings.
KERNEL_NAMES = ("add", "multiply", "matmul", "relu", "sum_all")


def _require(strict: bool) -> object | None:
    """Return the extension module, honouring ``strict``."""
    module = get_extension()
    if module is None and strict:
        raise EveryOCudaError(
            f"A CUDA-only operation was requested but CUDA is unavailable: {unavailable_reason()}"
        )
    return module


def _prepare(*arrays: np.ndarray) -> list[np.ndarray]:
    """Make arrays contiguous float32, the dtype the kernels are compiled for."""
    return [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]


def get_kernel(name: str) -> Callable[..., np.ndarray] | None:
    """Return the CUDA implementation of ``name``, or ``None`` when unavailable.

    Used by :mod:`everyo.core.operations` to dispatch tensors whose device is
    ``cuda`` without paying an import cost on CPU-only machines.
    """
    if name not in KERNEL_NAMES or not is_available():
        return None
    return globals()[name]


def add(a: np.ndarray, b: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Element-wise addition on the GPU (shapes must match exactly)."""
    module = _require(strict)
    if module is None or a.shape != b.shape:
        return numpy_backend.add(a, b)
    lhs, rhs = _prepare(a, b)
    return module.vector_add(lhs.ravel(), rhs.ravel()).reshape(a.shape)


def multiply(a: np.ndarray, b: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Element-wise multiplication on the GPU (shapes must match exactly)."""
    module = _require(strict)
    if module is None or a.shape != b.shape:
        return numpy_backend.multiply(a, b)
    lhs, rhs = _prepare(a, b)
    return module.vector_multiply(lhs.ravel(), rhs.ravel()).reshape(a.shape)


def matmul(a: np.ndarray, b: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Matrix product on the GPU for 2-D float inputs."""
    module = _require(strict)
    if module is None or a.ndim != 2 or b.ndim != 2:
        return numpy_backend.matmul(a, b)
    lhs, rhs = _prepare(a, b)
    return module.matmul(lhs, rhs)


def relu(x: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """ReLU activation on the GPU."""
    module = _require(strict)
    if module is None:
        return numpy_backend.relu(x)
    (values,) = _prepare(x)
    return module.relu(values.ravel()).reshape(x.shape)


def sum_all(x: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Full reduction on the GPU."""
    module = _require(strict)
    if module is None:
        return numpy_backend.sum_all(x)
    (values,) = _prepare(x)
    return np.asarray(module.sum(values.ravel()), dtype=np.float32)
