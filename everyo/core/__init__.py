"""Core tensor machinery: dtypes, devices, the tensor type and its operations."""

from __future__ import annotations

from everyo.core.autograd import enable_grad, no_grad, set_grad_enabled
from everyo.core.device import Device, device, resolve_device
from everyo.core.dtype import DTYPES, dtype_name, resolve_dtype
from everyo.core.tensor import Tensor, as_tensor, is_tensor, tensor

__all__ = [
    "DTYPES",
    "Device",
    "Tensor",
    "as_tensor",
    "device",
    "dtype_name",
    "enable_grad",
    "is_tensor",
    "no_grad",
    "resolve_device",
    "resolve_dtype",
    "set_grad_enabled",
    "tensor",
]
