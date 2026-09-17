"""Loss functions.

Every loss is a :class:`~everyo.nn.module.Module` whose ``forward`` returns a
differentiable tensor, so ``loss.backward()`` populates parameter gradients.
Each loss also supports the usual ``reduction`` choices: ``"mean"``, ``"sum"``
or ``"none"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError
from everyo.nn.module import Module, register_module

__all__ = [
    "Loss",
    "MSELoss",
    "MAELoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "mse_loss",
    "mae_loss",
    "binary_cross_entropy",
    "cross_entropy",
]

_VALID_REDUCTIONS = ("mean", "sum", "none")
#: Probabilities are clipped away from 0 and 1 before taking a logarithm.
_CLIP_EPS = 1e-7


class Loss(Module):
    """Base class handling the ``reduction`` argument shared by all losses."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(
                f"Unknown reduction {reduction!r}. Valid options are: "
                f"{', '.join(_VALID_REDUCTIONS)}."
            )
        self.reduction = reduction

    def _reduce(self, per_element: Tensor) -> Tensor:
        """Apply the configured reduction to an element-wise loss tensor."""
        if self.reduction == "mean":
            return ops.mean(per_element)
        if self.reduction == "sum":
            return ops.sum(per_element)
        return per_element

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this loss."""
        return {"reduction": self.reduction}


def _check_same_shape(prediction: Tensor, target: Tensor, name: str) -> None:
    if prediction.shape != target.shape:
        raise EveryOShapeError(
            f"{name} expects predictions and targets with the same shape, but "
            f"got {prediction.shape} and {target.shape}. If your targets are "
            "class indices, use CrossEntropyLoss instead."
        )


@register_module
class MSELoss(Loss):
    """Mean squared error, the default choice for regression.

    Example:
        >>> import everyo as eo
        >>> loss = eo.MSELoss()
        >>> float(loss(eo.tensor([2.0]), eo.tensor([1.0])).item())
        1.0
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        """Return the squared error between ``prediction`` and ``target``."""
        pred = as_tensor(prediction)
        true = as_tensor(target)
        _check_same_shape(pred, true, "MSELoss")
        difference = ops.subtract(pred, true)
        return self._reduce(ops.multiply(difference, difference))


@register_module
class MAELoss(Loss):
    """Mean absolute error; less sensitive to outliers than :class:`MSELoss`."""

    def forward(self, prediction: Any, target: Any) -> Tensor:
        """Return the absolute error between ``prediction`` and ``target``."""
        pred = as_tensor(prediction)
        true = as_tensor(target)
        _check_same_shape(pred, true, "MAELoss")
        return self._reduce(ops.abs(ops.subtract(pred, true)))


@register_module
class BCELoss(Loss):
    """Binary cross entropy over probabilities in ``[0, 1]``.

    Predictions are clipped to ``[1e-7, 1 - 1e-7]`` before the logarithm so that
    a confident-but-wrong prediction produces a large finite loss instead of
    ``inf``.  When your model outputs raw scores, prefer
    :class:`BCEWithLogitsLoss`.
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        """Return the binary cross entropy of ``prediction`` against ``target``."""
        pred = as_tensor(prediction)
        true = as_tensor(target)
        _check_same_shape(pred, true, "BCELoss")
        data = pred.data
        if data.size and (data.min() < 0.0 or data.max() > 1.0):
            raise EveryOShapeError(
                "BCELoss expects probabilities in [0, 1] but received values in "
                f"[{data.min():.4g}, {data.max():.4g}]. Apply a Sigmoid first, "
                "or use BCEWithLogitsLoss."
            )
        safe = ops.clip(pred, _CLIP_EPS, 1.0 - _CLIP_EPS)
        positive = ops.multiply(true, ops.log(safe))
        negative = ops.multiply(ops.subtract(1.0, true), ops.log(ops.subtract(1.0, safe)))
        return self._reduce(ops.negative(ops.add(positive, negative)))


@register_module
class BCEWithLogitsLoss(Loss):
    """Binary cross entropy computed directly from logits.

    Uses the stable form ``max(x, 0) - x * y + log(1 + exp(-|x|))`` so that
    large-magnitude logits do not overflow.
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        """Return the binary cross entropy of logits against ``target``."""
        logits = as_tensor(prediction)
        true = as_tensor(target)
        _check_same_shape(logits, true, "BCEWithLogitsLoss")
        return self._reduce(ops.binary_cross_entropy_with_logits(logits, true))


@register_module
class CrossEntropyLoss(Loss):
    """Multi-class cross entropy over raw logits.

    The module applies a log-softmax internally, so the model's final layer
    should be a plain :class:`~everyo.nn.layers.Linear` without an activation.

    Targets may be either integer class indices of shape ``(batch,)`` or
    one-hot rows of shape ``(batch, num_classes)``.

    Example:
        >>> import everyo as eo
        >>> logits = eo.tensor([[2.0, 0.5, 0.1]])
        >>> float(eo.CrossEntropyLoss()(logits, [0]).item()) < 0.6
        True
    """

    def forward(self, prediction: Any, target: Any) -> Tensor:
        """Return the cross entropy of ``prediction`` logits against ``target``."""
        logits = as_tensor(prediction)
        if logits.ndim != 2:
            raise EveryOShapeError(
                f"CrossEntropyLoss expects logits with shape (batch, classes), got {logits.shape}."
            )
        num_classes = logits.shape[1]
        target_array = np.asarray(target.data if isinstance(target, Tensor) else target)

        if target_array.ndim == 1 or (target_array.ndim == 2 and target_array.shape[1] == 1):
            indices = target_array.reshape(-1)
            if indices.shape[0] != logits.shape[0]:
                raise EveryOShapeError(
                    f"CrossEntropyLoss received {indices.shape[0]} targets for "
                    f"{logits.shape[0]} predictions; they must match."
                )
            if not np.all(np.equal(np.mod(indices, 1), 0)):
                raise EveryOShapeError(
                    "Class-index targets must be whole numbers. Pass one-hot "
                    "targets if you need soft labels."
                )
            true = ops.one_hot(indices, num_classes)
        elif target_array.ndim == 2:
            if target_array.shape != logits.shape:
                raise EveryOShapeError(
                    f"One-hot targets must have shape {logits.shape}, got {target_array.shape}."
                )
            true = as_tensor(target)
        else:
            raise EveryOShapeError(
                "CrossEntropyLoss targets must be 1-D class indices or 2-D "
                f"one-hot rows, got an array with {target_array.ndim} dimensions."
            )

        log_probabilities = ops.log_softmax(logits, axis=1)
        per_sample = ops.negative(ops.sum(ops.multiply(true, log_probabilities), axis=1))
        return self._reduce(per_sample)


def mse_loss(prediction: Any, target: Any, reduction: str = "mean") -> Tensor:
    """Functional form of :class:`MSELoss`."""
    return MSELoss(reduction=reduction)(prediction, target)


def mae_loss(prediction: Any, target: Any, reduction: str = "mean") -> Tensor:
    """Functional form of :class:`MAELoss`."""
    return MAELoss(reduction=reduction)(prediction, target)


def binary_cross_entropy(prediction: Any, target: Any, reduction: str = "mean") -> Tensor:
    """Functional form of :class:`BCELoss`."""
    return BCELoss(reduction=reduction)(prediction, target)


def cross_entropy(prediction: Any, target: Any, reduction: str = "mean") -> Tensor:
    """Functional form of :class:`CrossEntropyLoss`."""
    return CrossEntropyLoss(reduction=reduction)(prediction, target)
