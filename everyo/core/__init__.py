"""Core tensor machinery: dtypes, devices, the tensor type and its operations."""

from __future__ import annotations

from everyo.core.autograd import enable_grad, no_grad, set_grad_enabled
from everyo.core.convolution import avg_pool2d, conv2d, max_pool2d
from everyo.core.device import Device, device, resolve_device
from everyo.core.dtype import DTYPES, dtype_name, resolve_dtype
from everyo.core.tensor import Tensor, as_tensor, is_tensor, tensor

__all__ = [
    "DTYPES",
    "avg_pool2d",
    "conv2d",
    "max_pool2d",
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
