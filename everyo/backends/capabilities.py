"""Operation capability descriptors for backend dispatch.

Capabilities tell the dispatch system whether a backend can handle a given
operation for the supplied tensor dtypes, shapes and device.  This prevents
silent fallback to CPU for operations the user expects to run on an
accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OperationCapability",
    "STANDARD_OPERATIONS",
]


@dataclass(frozen=True)
class OperationCapability:
    """Describes a single operation a backend can execute."""

    name: str
    supported_dtypes: frozenset[str] = frozenset({"float32", "float64"})
    min_ndim: int = 0
    max_ndim: int = 64
    requires_contiguous: bool = False
    supports_broadcasting: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# Operations every backend should *try* to support.  A backend that cannot
# implement one simply omits it from its capabilities.
STANDARD_OPERATIONS: tuple[str, ...] = (
    # Elementwise arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "negative",
    "power",
    "abs",
    # Elementwise maths
    "exp",
    "log",
    "sqrt",
    "clip",
    # Activations
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    # Linear algebra
    "matmul",
    "dot",
    # Reductions
    "sum",
    "mean",
    "max",
    "min",
    "variance",
    # Shape
    "reshape",
    "transpose",
    "flatten",
    "concatenate",
    "stack",
    # Creation
    "zeros",
    "ones",
    "full",
    "eye",
    # Convolution
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
    # Normalization
    "batch_norm",
    "layer_norm",
)
