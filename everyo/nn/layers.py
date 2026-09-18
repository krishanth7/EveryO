"""Layers available in EveryO v0.1.0."""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.autocast import autocast_like
from everyo.core.convolution import _pair, avg_pool2d, conv2d, max_pool2d
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError
from everyo.nn.initialization import get_initializer, zeros_init
from everyo.nn.module import Module, Parameter, register_module

__all__ = ["Linear", "Flatten", "Dropout", "Conv2D", "MaxPool2D", "AvgPool2D"]


@register_module
class Linear(Module):
    """Fully connected layer computing ``x @ W + b``.

    Args:
        in_features: Size of the last input dimension.
        out_features: Size of the last output dimension.
        bias: Whether to learn an additive bias.
        initializer: Name of a scheme from :mod:`everyo.nn.initialization`.
        seed: Optional seed making the initial weights reproducible.

    Example:
        >>> import everyo as eo
        >>> layer = eo.Linear(4, 2, seed=0)
        >>> layer(eo.zeros(3, 4)).shape
        (3, 2)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        initializer: str = "he_uniform",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise EveryOShapeError(
                f"Linear requires positive feature counts, got "
                f"in_features={in_features}, out_features={out_features}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(bias)
        self.initializer = str(initializer)
        self.seed = seed

        init_fn = get_initializer(initializer)
        weight = init_fn((self.in_features, self.out_features), seed=seed)
        self.weight = Parameter(np.asarray(weight, dtype=np.float32), name="weight")
        if self.use_bias:
            self.bias = Parameter(
                np.asarray(zeros_init((self.out_features,)), dtype=np.float32),
                name="bias",
            )

    def forward(self, x: Any) -> Tensor:
        """Apply the affine transformation to ``x``."""
        value = as_tensor(x)
        if value.ndim == 0:
            raise EveryOShapeError(
                "Linear expects an input with at least one dimension, got a scalar."
            )
        if value.shape[-1] != self.in_features:
            raise EveryOShapeError(
                f"Linear(in_features={self.in_features}) received an input with "
                f"shape {value.shape}; the last dimension must be "
                f"{self.in_features}, not {value.shape[-1]}."
            )
        out = ops.matmul(value, self.weight)
        if self.use_bias:
            # Under autocast the matmul returns float16; casting the bias to
            # match keeps the activation in reduced precision instead of letting
            # NumPy promote the sum back to float32. Outside autocast this is a
            # no-op and adds no graph node.
            out = ops.add(out, autocast_like(self.bias))
        return out

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.use_bias,
            "initializer": self.initializer,
        }


@register_module
class Flatten(Module):
    """Flatten every dimension from ``start_dim`` onwards.

    Typically used to turn ``(batch, height, width)`` images into
    ``(batch, height * width)`` feature vectors.
    """

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)

    def forward(self, x: Any) -> Tensor:
        """Flatten ``x`` from :attr:`start_dim`."""
        return ops.flatten(as_tensor(x), start_dim=self.start_dim)

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {"start_dim": self.start_dim}


@register_module
class Dropout(Module):
    """Randomly zero a fraction of the inputs during training.

    Uses inverted dropout: surviving activations are scaled by ``1 / (1 - p)``
    so that the expected value is unchanged and no rescaling is needed at
    inference time.  In evaluation mode the layer is the identity.

    Args:
        p: Probability of dropping an element, in ``[0, 1)``.
        seed: Optional seed for reproducible masks.
    """

    def __init__(self, p: float = 0.5, *, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= float(p) < 1.0:
            raise ValueError(
                f"Dropout probability must be in [0, 1), got {p}. A probability "
                "of 1.0 would zero every activation."
            )
        self.p = float(p)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def forward(self, x: Any) -> Tensor:
        """Apply dropout when training; return ``x`` unchanged when evaluating."""
        value = as_tensor(x)
        if not self.training or self.p == 0.0:
            return value
        keep = 1.0 - self.p
        mask = (self._rng.random(value.shape) < keep).astype(value.data.dtype) / keep
        return ops.multiply(value, Tensor(mask, device=value.device))

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {"p": self.p}


@register_module
class Conv2D(Module):
    """2-D convolution over a batch of images.

    Tensors are ``NHWC``: ``(batch, height, width, channels)``. The learned
    kernel has shape ``(kernel_h, kernel_w, in_channels, out_channels)``, which
    is TensorFlow's layout — so results can be, and are, compared against
    ``tf.nn.conv2d`` directly.

    Args:
        in_channels: Channels the input carries.
        out_channels: Number of filters to learn.
        kernel_size: Int or ``(height, width)``.
        stride: Int or ``(stride_h, stride_w)``.
        padding: ``"valid"`` (default), ``"same"``, or an int applied to both
            sides of both axes.
        bias: Learn one bias per output channel.
        initializer: Scheme from :mod:`everyo.nn.initialization`. The default
            suits the ReLU that usually follows.
        seed: Makes the initial kernel reproducible.

    Example:
        >>> import everyo as eo
        >>> layer = eo.Conv2D(1, 8, 3, padding="same", seed=0)
        >>> layer(eo.zeros(4, 8, 8, 1)).shape
        (4, 8, 8, 8)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any = 3,
        *,
        stride: Any = 1,
        padding: Any = "valid",
        bias: bool = True,
        initializer: str = "he_uniform",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise EveryOShapeError(
                f"Conv2D requires positive channel counts, got "
                f"in_channels={in_channels}, out_channels={out_channels}."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = padding
        self.use_bias = bool(bias)
        self.initializer = str(initializer)
        self.seed = seed

        shape = (*self.kernel_size, self.in_channels, self.out_channels)
        init_fn = get_initializer(initializer)
        self.weight = Parameter(
            np.asarray(init_fn(shape, seed=seed), dtype=np.float32), name="weight"
        )
        if self.use_bias:
            self.bias = Parameter(
                np.asarray(zeros_init((self.out_channels,)), dtype=np.float32), name="bias"
            )

    def forward(self, x: Any) -> Tensor:
        """Convolve ``x`` with the learned kernel."""
        value = as_tensor(x)
        if value.ndim == 4 and value.shape[-1] != self.in_channels:
            raise EveryOShapeError(
                f"Conv2D(in_channels={self.in_channels}) received an input with "
                f"shape {value.shape}; the last axis must be {self.in_channels} "
                f"channel(s), not {value.shape[-1]}."
            )
        return conv2d(
            value,
            self.weight,
            self.bias if self.use_bias else None,
            stride=self.stride,
            padding=self.padding,
        )

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": self.padding,
            "bias": self.use_bias,
            "initializer": self.initializer,
        }


class _Pool2D(Module):
    """Shared plumbing for the pooling layers."""

    def __init__(self, pool_size: Any = 2, *, stride: Any = None, padding: Any = "valid") -> None:
        super().__init__()
        self.pool_size = _pair(pool_size, "pool_size")
        self.stride = None if stride is None else _pair(stride, "stride")
        self.padding = padding

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "pool_size": list(self.pool_size),
            "stride": None if self.stride is None else list(self.stride),
            "padding": self.padding,
        }


@register_module
class MaxPool2D(_Pool2D):
    """Max pooling: keep the strongest activation in each window.

    ``stride`` defaults to ``pool_size``, giving non-overlapping windows that
    halve the resolution for the usual ``pool_size=2``. The gradient reaches
    only the winning element of each window.

    Example:
        >>> import everyo as eo
        >>> eo.MaxPool2D(2)(eo.zeros(2, 8, 8, 3)).shape
        (2, 4, 4, 3)
    """

    def forward(self, x: Any) -> Tensor:
        """Downsample ``x`` by taking the maximum of each window."""
        return max_pool2d(x, self.pool_size, stride=self.stride, padding=self.padding)


@register_module
class AvgPool2D(_Pool2D):
    """Average pooling: keep the mean activation of each window.

    With ``padding="same"`` the mean is taken over real cells only, never over
    the padding, so edge windows are not darkened.

    Example:
        >>> import everyo as eo
        >>> pooled = eo.AvgPool2D(2)(eo.ones(1, 4, 4, 1))
        >>> pooled.shape
        (1, 2, 2, 1)
        >>> float(pooled.data[0, 0, 0, 0])
        1.0
    """

    def forward(self, x: Any) -> Tensor:
        """Downsample ``x`` by averaging each window."""
        return avg_pool2d(x, self.pool_size, stride=self.stride, padding=self.padding)
