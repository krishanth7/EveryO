"""Activation functions, available both as functions and as modules.

The numerics live in :mod:`everyo.core.operations` so that autograd and the
CUDA dispatch path see them; the classes here are thin wrappers that make
activations usable inside :class:`~everyo.nn.sequential.Sequential`.
"""

from __future__ import annotations

from typing import Any

from everyo.core import operations as ops
from everyo.core.tensor import Tensor
from everyo.nn.module import Module, register_module

__all__ = [
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LogSoftmax",
]

relu = ops.relu
sigmoid = ops.sigmoid
tanh = ops.tanh
softmax = ops.softmax
log_softmax = ops.log_softmax


@register_module
class ReLU(Module):
    """Rectified linear unit, ``max(x, 0)``."""

    def forward(self, x: Any) -> Tensor:
        """Apply ReLU element-wise."""
        return ops.relu(x)


@register_module
class Sigmoid(Module):
    """Logistic sigmoid, mapping values into ``(0, 1)``."""

    def forward(self, x: Any) -> Tensor:
        """Apply the sigmoid element-wise."""
        return ops.sigmoid(x)


@register_module
class Tanh(Module):
    """Hyperbolic tangent, mapping values into ``(-1, 1)``."""

    def forward(self, x: Any) -> Tensor:
        """Apply tanh element-wise."""
        return ops.tanh(x)


@register_module
class Softmax(Module):
    """Softmax over ``axis``, producing a probability distribution."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = int(axis)

    def forward(self, x: Any) -> Tensor:
        """Apply softmax along :attr:`axis`."""
        return ops.softmax(x, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {"axis": self.axis}


@register_module
class LogSoftmax(Module):
    """Logarithm of the softmax, numerically stabler than ``log(softmax(x))``."""

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = int(axis)

    def forward(self, x: Any) -> Tensor:
        """Apply log-softmax along :attr:`axis`."""
        return ops.log_softmax(x, axis=self.axis)

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {"axis": self.axis}
