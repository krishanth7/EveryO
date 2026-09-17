"""Adam: adaptive moment estimation (Kingma & Ba, 2015)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from everyo.core.tensor import Tensor
from everyo.optim.optimizer import Optimizer

__all__ = ["Adam"]


class Adam(Optimizer):
    """Adam optimizer with bias-corrected first and second moments.

    Args:
        parameters: Parameters to update.
        lr: Learning rate.
        betas: Decay rates ``(beta1, beta2)`` for the moment estimates.
        eps: Term added to the denominator for numerical stability.
        weight_decay: L2 penalty added to the gradient.
        amsgrad: Keep the running maximum of the second moment.

    Example:
        >>> import everyo as eo
        >>> model = eo.Sequential(eo.Linear(2, 1, seed=0))
        >>> optimizer = eo.Adam(model.parameters(), lr=0.001)
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.001,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(parameters, lr)
        beta1, beta2 = betas
        for name, value in (("beta1", beta1), ("beta2", beta2)):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if float(weight_decay) < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.amsgrad = bool(amsgrad)

    def step(self) -> None:
        """Update every parameter that has a gradient."""
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count

        for parameter, gradient in self._pending():
            grad = np.asarray(gradient, dtype=np.float64)
            if self.weight_decay:
                grad = grad + self.weight_decay * parameter.data

            state = self.state_for(parameter)
            if not state:
                state["m"] = np.zeros_like(grad)
                state["v"] = np.zeros_like(grad)
                if self.amsgrad:
                    state["v_max"] = np.zeros_like(grad)

            state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * grad * grad

            m_hat = state["m"] / bias_correction1
            if self.amsgrad:
                state["v_max"] = np.maximum(state["v_max"], state["v"])
                v_hat = state["v_max"] / bias_correction2
            else:
                v_hat = state["v"] / bias_correction2

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            parameter.data = parameter.data - update.astype(parameter.data.dtype)

    def get_config(self) -> dict[str, Any]:
        """Return the hyper-parameters of this optimizer."""
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }
