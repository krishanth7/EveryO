"""Layers available in EveryO v0.1.0."""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError
from everyo.nn.initialization import get_initializer, zeros_init
from everyo.nn.module import Module, Parameter, register_module

__all__ = ["Linear", "Flatten", "Dropout"]


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
            out = ops.add(out, self.bias)
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
