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
    """Make arrays contiguous. They are already float32; see :func:`_is_float32`."""
    return [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]


def _is_float32(*arrays: np.ndarray) -> bool:
    """Return ``True`` when every array is already float32.

    The kernels are compiled for float32 only. Casting other dtypes to float32
    would silently lose precision for float64 tensors and corrupt integers
    outside float32's exactly representable range, so anything else is served
    by the NumPy backend instead.
    """
    return all(np.asarray(a).dtype == np.float32 for a in arrays)


def _reject_dtype(strict: bool, *arrays: np.ndarray) -> None:
    """Raise when a CUDA-only call was made with a dtype the kernels lack."""
    if strict:
        dtypes = ", ".join(sorted({str(np.asarray(a).dtype) for a in arrays}))
        raise EveryOCudaError(
            f"The EveryO CUDA kernels are compiled for float32 only, but this "
            f"call passed {dtypes}. Cast the data with .astype('float32') to "
            f"use the GPU, or drop strict=True to compute on the CPU without "
            f"losing precision."
        )


def get_kernel(name: str) -> Callable[..., np.ndarray] | None:
    """Return the CUDA implementation of ``name``, or ``None`` when unavailable.

    Used by :mod:`everyo.core.operations` to dispatch tensors whose device is
    ``cuda`` without paying an import cost on CPU-only machines. The returned
    callable serves non-float32 inputs from the NumPy backend rather than
    casting them, so dtypes are never silently downgraded.
    """
    if name not in KERNEL_NAMES or not is_available():
        return None
    return globals()[name]


def add(a: np.ndarray, b: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Element-wise addition on the GPU (shapes must match exactly)."""
    module = _require(strict)
    if not _is_float32(a, b):
        _reject_dtype(strict, a, b)
        return numpy_backend.add(a, b)
    if module is None or a.shape != b.shape:
        return numpy_backend.add(a, b)
    lhs, rhs = _prepare(a, b)
    return module.vector_add(lhs.ravel(), rhs.ravel()).reshape(a.shape)


def multiply(a: np.ndarray, b: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Element-wise multiplication on the GPU (shapes must match exactly)."""
    module = _require(strict)
    if not _is_float32(a, b):
        _reject_dtype(strict, a, b)
        return numpy_backend.multiply(a, b)
    if module is None or a.shape != b.shape:
        return numpy_backend.multiply(a, b)
    lhs, rhs = _prepare(a, b)
    return module.vector_multiply(lhs.ravel(), rhs.ravel()).reshape(a.shape)


def matmul(a: np.ndarray, b: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Matrix product on the GPU for 2-D float inputs."""
    module = _require(strict)
    if not _is_float32(a, b):
        _reject_dtype(strict, a, b)
        return numpy_backend.matmul(a, b)
    if module is None or a.ndim != 2 or b.ndim != 2:
        return numpy_backend.matmul(a, b)
    lhs, rhs = _prepare(a, b)
    return module.matmul(lhs, rhs)


def relu(x: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """ReLU activation on the GPU."""
    module = _require(strict)
    if not _is_float32(x):
        _reject_dtype(strict, x)
        return numpy_backend.relu(x)
    if module is None:
        return numpy_backend.relu(x)
    (values,) = _prepare(x)
    return module.relu(values.ravel()).reshape(x.shape)


def sum_all(x: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Full reduction on the GPU."""
    module = _require(strict)
    if not _is_float32(x):
        _reject_dtype(strict, x)
        return numpy_backend.sum_all(x)
    if module is None:
        return numpy_backend.sum_all(x)
    (values,) = _prepare(x)
    return np.asarray(module.sum(values.ravel()), dtype=np.float32)
