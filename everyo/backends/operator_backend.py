"""Unified operator dispatch for EveryO execution backends.

The core tensor package talks only to :class:`OperatorBackend`; it does not
import NumPy or CUDA kernels directly.  This keeps fallback policy in one
place and gives future backends a small, testable contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from everyo.backends import numpy_backend
from everyo.core.device import Device

__all__ = [
    "DispatchResult",
    "OperatorBackend",
    "NumpyOperatorBackend",
    "CudaOperatorBackend",
    "backend_for",
    "dispatch",
]


@dataclass(frozen=True)
class DispatchResult:
    """Result of one backend decision, including observable fallback data."""

    value: np.ndarray
    requested_backend: str
    executed_backend: str
    fallback_reason: str | None = None

    @property
    def fell_back(self) -> bool:
        return self.requested_backend != self.executed_backend


@runtime_checkable
class OperatorBackend(Protocol):
    """Contract implemented by numerical execution backends."""

    name: str

    def is_available(self) -> bool: ...

    def supports(self, operation: str, *arrays: np.ndarray) -> bool: ...

    def execute(self, operation: str, *arrays: np.ndarray) -> np.ndarray: ...

    def unsupported_reason(self, operation: str, *arrays: np.ndarray) -> str: ...

    def to_host(self, value: Any) -> np.ndarray: ...

    def from_host(self, value: Any) -> np.ndarray: ...


class NumpyOperatorBackend:
    """Always-available reference backend."""

    name = "numpy"

    def is_available(self) -> bool:
        return True

    def supports(self, operation: str, *arrays: np.ndarray) -> bool:
        return callable(getattr(numpy_backend, operation, None))

    def execute(self, operation: str, *arrays: np.ndarray) -> np.ndarray:
        return np.asarray(getattr(numpy_backend, operation)(*arrays))

    def unsupported_reason(self, operation: str, *arrays: np.ndarray) -> str:
        return f"NumPy backend has no operator named {operation!r}."

    def to_host(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def from_host(self, value: Any) -> np.ndarray:
        return np.asarray(value)


class CudaOperatorBackend:
    """Optional CUDA backend with explicit, documented fallback decisions."""

    name = "cuda"

    def is_available(self) -> bool:
        from everyo.cuda.availability import is_available

        return is_available()

    def supports(self, operation: str, *arrays: np.ndarray) -> bool:
        from everyo.cuda import interface

        return interface.supports(operation, *arrays)

    def execute(self, operation: str, *arrays: np.ndarray) -> np.ndarray:
        from everyo.cuda import interface

        kernel = interface.get_kernel(operation)
        if kernel is None:  # pragma: no cover - guarded by supports
            raise RuntimeError(f"CUDA operator {operation!r} is unavailable.")
        return np.asarray(kernel(*arrays, strict=True))

    def unsupported_reason(self, operation: str, *arrays: np.ndarray) -> str:
        from everyo.cuda import interface

        return interface.unsupported_reason(operation, *arrays)

    def to_host(self, value: Any) -> np.ndarray:
        # Native kernels currently return host arrays.  This boundary remains
        # explicit so a device-owned buffer can replace it without changing
        # core operations.
        return np.asarray(value)

    def from_host(self, value: Any) -> np.ndarray:
        return np.asarray(value)


_NUMPY = NumpyOperatorBackend()
_CUDA = CudaOperatorBackend()


def backend_for(device: Device | str) -> OperatorBackend:
    """Resolve the backend for a device without importing the CUDA extension."""
    name = device.type if isinstance(device, Device) else str(device).split(":", 1)[0]
    return _CUDA if name == "cuda" else _NUMPY


def dispatch(operation: str, device: Device, *arrays: np.ndarray) -> DispatchResult:
    """Execute an operator and report whether/why CPU fallback occurred."""
    requested = backend_for(device)
    if requested.supports(operation, *arrays):
        return DispatchResult(
            requested.execute(operation, *arrays), requested.name, requested.name
        )

    reason = requested.unsupported_reason(operation, *arrays)
    if not _NUMPY.supports(operation, *arrays):
        raise NotImplementedError(reason)
    return DispatchResult(
        _NUMPY.execute(operation, *arrays), requested.name, _NUMPY.name, reason
    )
