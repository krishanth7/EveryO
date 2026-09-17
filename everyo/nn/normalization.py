"""Normalization layers.

Both layers here are *composed* from the differentiable primitives in
:mod:`everyo.core.operations` rather than carrying hand-written backward
passes. That is a deliberate choice: the primitives are already checked against
finite differences, so composing them makes the gradient of a normalization
layer correct by construction instead of correct by review.

The two layers answer the same question — "normalise these activations" — with
different notions of *across what*:

* :class:`BatchNorm1D` and :class:`BatchNorm2D` normalise each feature across
  the batch. Powerful, but the statistics depend on the other samples, so they
  keep a running estimate for inference and behave differently in train and
  eval mode.
* :class:`LayerNorm` normalises each sample across its own features. It needs
  no running statistics and behaves identically in both modes, which is why
  transformers use it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from everyo.core import operations as ops
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError
from everyo.nn.module import Module, Parameter, register_module

__all__ = ["LayerNorm", "BatchNorm1D", "BatchNorm2D"]


class _BatchNorm(Module):
    """Shared implementation for the batch normalization layers."""

    #: Axes the statistics are taken over; set by each subclass.
    _reduce_axes: tuple[int, ...] = (0,)
    #: Rank the layer accepts, for a clear error message.
    _expected_ndim: int = 2

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if int(num_features) <= 0:
            raise EveryOShapeError(
                f"{type(self).__name__} requires a positive feature count, got {num_features}."
            )
        if float(eps) <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if not 0.0 <= float(momentum) <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}.")

        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)

        if self.affine:
            self.weight = Parameter(np.ones(self.num_features, dtype=np.float32), name="weight")
            self.bias = Parameter(np.zeros(self.num_features, dtype=np.float32), name="bias")
        if self.track_running_stats:
            self.register_buffer("running_mean", np.zeros(self.num_features, dtype=np.float32))
            self.register_buffer("running_var", np.ones(self.num_features, dtype=np.float32))
            self.register_buffer("num_batches_tracked", np.zeros((), dtype=np.int64))

    def _broadcast_shape(self, ndim: int) -> tuple[int, ...]:
        """Shape that lines the per-feature vectors up with the input."""
        return tuple(self.num_features if axis == ndim - 1 else 1 for axis in range(ndim))

    def forward(self, x: Any) -> Tensor:
        """Normalise ``x`` per feature, using batch or running statistics."""
        value = as_tensor(x)
        if value.ndim != self._expected_ndim:
            raise EveryOShapeError(
                f"{type(self).__name__} expects a {self._expected_ndim}-D input "
                f"{self._shape_hint()}, got shape {value.shape}."
            )
        if value.shape[-1] != self.num_features:
            raise EveryOShapeError(
                f"{type(self).__name__}(num_features={self.num_features}) received "
                f"an input whose last axis is {value.shape[-1]}."
            )

        if self.training or not self.track_running_stats:
            mean = ops.mean(value, axis=self._reduce_axes, keepdims=True)
            centred = ops.subtract(value, mean)
            variance = ops.mean(
                ops.multiply(centred, centred), axis=self._reduce_axes, keepdims=True
            )
            if self.training and self.track_running_stats:
                self._update_running_stats(value, mean, variance)
        else:
            shape = self._broadcast_shape(value.ndim)
            mean = Tensor(self.running_mean.reshape(shape))
            variance = Tensor(self.running_var.reshape(shape))
            centred = ops.subtract(value, mean)

        normalised = ops.divide(centred, ops.sqrt(ops.add(variance, self.eps)))
        if not self.affine:
            return normalised

        shape = self._broadcast_shape(value.ndim)
        return ops.add(
            ops.multiply(normalised, ops.reshape(self.weight, shape)),
            ops.reshape(self.bias, shape),
        )

    def _update_running_stats(self, value: Tensor, mean: Tensor, variance: Tensor) -> None:
        """Blend this batch's statistics into the running estimates.

        The variance is corrected to the unbiased estimator before being
        stored, which is what inference should use; the biased estimate is
        still what normalises the current batch.
        """
        count = int(np.prod([value.shape[axis] for axis in self._reduce_axes]))
        correction = count / max(count - 1, 1)
        batch_mean = np.asarray(mean.data).reshape(-1)
        batch_var = np.asarray(variance.data).reshape(-1) * correction

        self.running_mean = (
            (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean
        ).astype(np.float32)
        self.running_var = (
            (1.0 - self.momentum) * self.running_var + self.momentum * batch_var
        ).astype(np.float32)
        self.num_batches_tracked = self.num_batches_tracked + 1

    def _shape_hint(self) -> str:
        return "(batch, features)"

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "track_running_stats": self.track_running_stats,
        }


@register_module
class BatchNorm1D(_BatchNorm):
    """Batch normalization for ``(batch, features)`` activations.

    Args:
        num_features: Size of the last axis.
        eps: Added to the variance before the square root.
        momentum: Weight of the current batch in the running estimates.
        affine: Learn a per-feature scale and shift.
        track_running_stats: Keep running statistics for evaluation mode. With
            this off, evaluation uses the batch's own statistics, which makes
            predictions depend on how inputs were batched.

    Example:
        >>> import everyo as eo
        >>> layer = eo.BatchNorm1D(4)
        >>> layer(eo.randn(8, 4)).shape
        (8, 4)
    """

    _reduce_axes = (0,)
    _expected_ndim = 2


@register_module
class BatchNorm2D(_BatchNorm):
    """Batch normalization for ``(batch, height, width, channels)`` images.

    Statistics are taken per channel across the batch *and* both spatial axes,
    which is what makes it the right partner for :class:`~everyo.nn.layers.Conv2D`.

    Example:
        >>> import everyo as eo
        >>> eo.BatchNorm2D(3)(eo.randn(2, 8, 8, 3)).shape
        (2, 8, 8, 3)
    """

    _reduce_axes = (0, 1, 2)
    _expected_ndim = 4

    def _shape_hint(self) -> str:
        return "(batch, height, width, channels)"


@register_module
class LayerNorm(Module):
    """Layer normalization over the last dimension(s) of each sample.

    Unlike batch normalization, the statistics come from a single sample, so
    the layer behaves identically in training and evaluation, needs no running
    state, and does not care how the data was batched. That independence is why
    transformers normalise this way.

    Args:
        normalized_shape: Size of the trailing axis, or a tuple of trailing
            axis sizes, to normalise over.
        eps: Added to the variance before the square root.
        affine: Learn an elementwise scale and shift.

    Example:
        >>> import everyo as eo
        >>> eo.LayerNorm(16)(eo.randn(4, 10, 16)).shape
        (4, 10, 16)
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        shape = (
            (int(normalized_shape),)
            if isinstance(normalized_shape, int)
            else tuple(int(dim) for dim in normalized_shape)
        )
        if not shape or any(dim <= 0 for dim in shape):
            raise EveryOShapeError(
                f"LayerNorm needs a positive normalized_shape, got {normalized_shape!r}."
            )
        if float(eps) <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.normalized_shape = shape
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = Parameter(np.ones(shape, dtype=np.float32), name="weight")
            self.bias = Parameter(np.zeros(shape, dtype=np.float32), name="bias")

    def forward(self, x: Any) -> Tensor:
        """Normalise each sample over its trailing axes."""
        value = as_tensor(x)
        rank = len(self.normalized_shape)
        if value.ndim < rank or tuple(value.shape[-rank:]) != self.normalized_shape:
            raise EveryOShapeError(
                f"LayerNorm(normalized_shape={self.normalized_shape}) received an "
                f"input with shape {value.shape}; its trailing axes must be "
                f"{self.normalized_shape}."
            )

        axes = tuple(range(value.ndim - rank, value.ndim))
        mean = ops.mean(value, axis=axes, keepdims=True)
        centred = ops.subtract(value, mean)
        variance = ops.mean(ops.multiply(centred, centred), axis=axes, keepdims=True)
        normalised = ops.divide(centred, ops.sqrt(ops.add(variance, self.eps)))

        if not self.affine:
            return normalised
        return ops.add(ops.multiply(normalised, self.weight), self.bias)

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "normalized_shape": list(self.normalized_shape),
            "eps": self.eps,
            "affine": self.affine,
        }
