"""Stochastic gradient descent, with optional momentum and weight decay."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from everyo.core.tensor import Tensor
from everyo.optim.optimizer import Optimizer

__all__ = ["SGD"]


class SGD(Optimizer):
    """Stochastic gradient descent.

    Args:
        parameters: Parameters to update.
        lr: Learning rate.
        momentum: Momentum coefficient in ``[0, 1)``.  ``0`` disables momentum.
        weight_decay: L2 penalty added to the gradient.
        nesterov: Use Nesterov accelerated gradient (requires ``momentum > 0``).

    Example:
        >>> import everyo as eo
        >>> model = eo.Sequential(eo.Linear(2, 1, seed=0))
        >>> optimizer = eo.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.01,
        *,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(parameters, lr)
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError(
                f"momentum must be in [0, 1), got {momentum}. Values of 1 or "
                "more make the update diverge."
            )
        if float(weight_decay) < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")
        if nesterov and float(momentum) <= 0.0:
            raise ValueError("Nesterov momentum requires momentum > 0.")

        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)

    def step(self) -> None:
        """Update every parameter that has a gradient."""
        self.step_count += 1
        for parameter, gradient in self._pending():
            grad = np.asarray(gradient, dtype=parameter.data.dtype)
            if self.weight_decay:
                grad = grad + self.weight_decay * parameter.data

            if self.momentum:
                state = self.state_for(parameter)
                buffer = state.get("velocity")
                buffer = grad.copy() if buffer is None else self.momentum * buffer + grad
                state["velocity"] = buffer
                grad = grad + self.momentum * buffer if self.nesterov else buffer

            parameter.data = parameter.data - self.lr * grad

    def get_config(self) -> dict[str, Any]:
        """Return the hyper-parameters of this optimizer."""
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
        }
