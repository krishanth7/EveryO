"""Exception hierarchy used across EveryO.

Every error raised by EveryO derives from :class:`EveryOError`, which makes it
possible to catch framework problems without also catching unrelated Python
errors.  Messages are written to tell the caller what was attempted, what went
wrong and (where useful) how to fix it.
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "EveryOError",
    "EveryOShapeError",
    "EveryODeviceError",
    "EveryODTypeError",
    "EveryOGradientError",
    "EveryOSerializationError",
    "EveryOConfigurationError",
    "EveryOBackendError",
    "EveryOCudaError",
]


class EveryOError(Exception):
    """Base class for all EveryO errors."""


class EveryOShapeError(EveryOError):
    """Raised when tensor shapes are incompatible with a requested operation."""

    @classmethod
    def for_matmul(cls, left: Sequence[int], right: Sequence[int]) -> EveryOShapeError:
        """Build the standard message for an invalid matrix multiplication."""
        return cls(
            f"Cannot multiply matrices with shapes {tuple(left)} and "
            f"{tuple(right)}. Expected the inner dimensions to match: "
            f"{tuple(left)[-1]} != {tuple(right)[0] if len(right) == 1 else tuple(right)[-2]}."
        )

    @classmethod
    def for_broadcast(
        cls, left: Sequence[int], right: Sequence[int], operation: str
    ) -> EveryOShapeError:
        """Build the standard message for an invalid broadcast."""
        return cls(
            f"Cannot broadcast shapes {tuple(left)} and {tuple(right)} in "
            f"'{operation}'. Dimensions must either be equal or one of them "
            f"must be 1, compared from the trailing axis."
        )


class EveryODeviceError(EveryOError):
    """Raised for unknown devices or impossible device requests."""


class EveryODTypeError(EveryOError):
    """Raised when a dtype is unknown or unsupported for an operation."""


class EveryOGradientError(EveryOError):
    """Raised when automatic differentiation is used incorrectly."""


class EveryOSerializationError(EveryOError):
    """Raised when a model archive cannot be written or read."""


class EveryOConfigurationError(EveryOError):
    """Raised when a configuration file or dictionary is invalid."""


class EveryOBackendError(EveryOError):
    """Raised when an optional backend is unavailable or misbehaving."""


class EveryOCudaError(EveryOBackendError):
    """Raised when a CUDA-only operation is requested but cannot be served."""
