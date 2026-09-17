"""Optional CUDA acceleration for EveryO.

Nothing in this package is required to use EveryO.  Import it freely: it never
raises when CUDA is missing.

Example:
    >>> import everyo as eo
    >>> eo.cuda.is_available()  # False on a CPU-only machine
    False
"""

from __future__ import annotations

from everyo.cuda.availability import (
    EXTENSION_NAME,
    device_count,
    is_available,
    reset_cache,
    runtime_info,
    unavailable_reason,
)
from everyo.cuda.interface import KERNEL_NAMES, add, matmul, multiply, relu, sum_all

__all__ = [
    "EXTENSION_NAME",
    "KERNEL_NAMES",
    "add",
    "device_count",
    "is_available",
    "matmul",
    "multiply",
    "relu",
    "reset_cache",
    "runtime_info",
    "sum_all",
    "unavailable_reason",
]
