"""Post-training dynamic int8 quantization.

Weights are stored as signed int8 values with one symmetric scale per output
channel. Activations remain floating point, making this a safe inference-only
optimization that requires no calibration dataset.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOError, EveryOShapeError
from everyo.nn.layers import Linear
from everyo.nn.module import Module

__all__ = ["QuantizationError", "QuantizedArray", "QuantizedLinear", "quantize", "quantize_dynamic"]


class QuantizationError(EveryOError):
    """Raised when a model cannot be quantized safely."""


@dataclass(frozen=True)
class QuantizedArray:
    """An int8 array and its symmetric scale factors."""

    values: np.ndarray
    scale: np.ndarray
    axis: int | None

    def dequantize(self) -> np.ndarray:
        """Reconstruct float32 values."""
        if self.axis is None:
            return self.values.astype(np.float32) * self.scale
        shape = [1] * self.values.ndim
        shape[self.axis] = self.values.shape[self.axis]
        return self.values.astype(np.float32) * self.scale.reshape(shape)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes + self.scale.nbytes)


def quantize(array: Any, *, axis: int | None = None) -> QuantizedArray:
    """Symmetrically quantize an array to int8, optionally per channel."""
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        raise QuantizationError("Cannot quantize an empty array.")
    if not np.all(np.isfinite(values)):
        raise QuantizationError("Quantization requires finite values.")
    if axis is not None:
        axis = int(axis) % values.ndim
        reduce_axes = tuple(index for index in range(values.ndim) if index != axis)
        maximum = np.max(np.abs(values), axis=reduce_axes)
    else:
        maximum = np.asarray(np.max(np.abs(values)))
    scale = np.maximum(maximum / 127.0, np.finfo(np.float32).tiny).astype(np.float32)
    broadcast = scale
    if axis is not None:
        shape = [1] * values.ndim
        shape[axis] = values.shape[axis]
        broadcast = scale.reshape(shape)
    encoded = np.clip(np.rint(values / broadcast), -127, 127).astype(np.int8)
    return QuantizedArray(encoded, scale, axis)


class QuantizedLinear(Module):
    """Inference-only Linear layer with per-output-channel int8 weights."""

    def __init__(self, layer: Linear) -> None:
        super().__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.use_bias = layer.use_bias
        packed = quantize(layer.weight.data, axis=1)
        self.register_buffer("qweight", packed.values)
        self.register_buffer("weight_scale", packed.scale)
        if self.use_bias:
            self.register_buffer("bias", layer.bias.data.astype(np.float32, copy=True))

    @property
    def weight(self) -> np.ndarray:
        """Dequantized weight, primarily for inspection and export."""
        return QuantizedArray(self.qweight, self.weight_scale, 1).dequantize()

    def forward(self, x: Any) -> Tensor:
        value = as_tensor(x)
        if value.shape[-1] != self.in_features:
            raise EveryOShapeError(
                f"QuantizedLinear(in_features={self.in_features}) received shape {value.shape}."
            )
        output = ops.matmul(value, Tensor(self.weight, device=value.device))
        if self.use_bias:
            output = ops.add(output, Tensor(self.bias, device=value.device))
        return output

    def get_config(self) -> dict[str, Any]:
        raise QuantizationError("QuantizedLinear must be exported through its state dictionary.")


def quantize_dynamic(model: Module, *, inplace: bool = False) -> Module:
    """Replace every Linear child with a QuantizedLinear for CPU inference."""
    result = model if inplace else copy.deepcopy(model)

    def replace(parent: Module) -> None:
        for name, child in list(parent._modules.items()):  # noqa: SLF001
            if isinstance(child, Linear):
                setattr(parent, name, QuantizedLinear(child))
            else:
                replace(child)

    if isinstance(result, Linear):
        return QuantizedLinear(result).eval()
    replace(result)
    return result.eval()
