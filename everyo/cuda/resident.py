"""GPU-resident inference tensors backed by the native CUDA extension."""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.cuda.availability import get_extension
from everyo.exceptions import EveryOCudaError, EveryOShapeError

__all__ = ["ResidentTensor", "to_device"]


class ResidentTensor:
    """A float32 tensor whose storage remains on the CUDA device.

    Upload occurs once in :func:`to_device`; supported chained operations keep
    their results on-device. Calling :meth:`numpy` is the explicit download.
    This inference API intentionally does not participate in autograd yet.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._handle.shape)

    @property
    def size(self) -> int:
        return int(self._handle.size)

    @property
    def device(self) -> str:
        return "cuda"

    def numpy(self) -> np.ndarray:
        return np.asarray(self._handle.numpy())

    def _binary(self, other: ResidentTensor, operation: str) -> ResidentTensor:
        if not isinstance(other, ResidentTensor):
            raise TypeError("GPU-resident operations require two ResidentTensor operands.")
        module = get_extension()
        return ResidentTensor(getattr(module, operation)(self._handle, other._handle))

    def __add__(self, other: ResidentTensor) -> ResidentTensor:
        return self._binary(other, "device_add")

    def __mul__(self, other: ResidentTensor) -> ResidentTensor:
        return self._binary(other, "device_multiply")

    def __matmul__(self, other: ResidentTensor) -> ResidentTensor:
        return self._binary(other, "device_matmul")

    def relu(self) -> ResidentTensor:
        return ResidentTensor(get_extension().device_relu(self._handle))

    def sum(self) -> float:
        return float(get_extension().device_sum(self._handle))

    def __repr__(self) -> str:
        return f"ResidentTensor(shape={self.shape}, dtype='float32', device='cuda')"


def to_device(value: Any) -> ResidentTensor:
    """Upload an array once and return GPU-owned storage."""
    module = get_extension()
    if module is None:
        raise EveryOCudaError("GPU-resident tensors require the built EveryO CUDA extension.")
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise EveryOCudaError("GPU-resident tensors currently require float32 input.")
    if array.size == 0:
        raise EveryOShapeError("GPU-resident tensors cannot be empty.")
    return ResidentTensor(module.DeviceTensor(np.ascontiguousarray(array)))
