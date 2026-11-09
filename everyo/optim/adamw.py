"""Decoupled-weight-decay AdamW for EveryO tensors."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from everyo.core.tensor import Tensor
from everyo.optim.optimizer import Optimizer

__all__ = ["AdamW"]


class AdamW(Optimizer):
    """Adam with decoupled weight decay; parameters without gradients are untouched."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.001,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(parameters, lr)
        if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
            raise ValueError("betas must contain values in [0, 1)")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas
        for parameter, gradient in self._pending():
            grad = np.asarray(gradient, dtype=np.float64)
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError("non-finite gradient in AdamW")
            state = self.state_for(parameter)
            if not state:
                state["m"] = np.zeros_like(grad)
                state["v"] = np.zeros_like(grad)
                state["steps"] = np.asarray(0, dtype=np.int64)
            state["steps"] = state["steps"] + 1
            steps = int(state["steps"])
            state["m"] = beta1 * state["m"] + (1 - beta1) * grad
            state["v"] = beta2 * state["v"] + (1 - beta2) * grad**2
            correction1 = 1 - beta1**steps
            correction2 = 1 - beta2**steps
            update = (state["m"] / correction1) / (np.sqrt(state["v"] / correction2) + self.eps)
            parameter.data = (
                parameter.data * (1 - self.lr * self.weight_decay) - self.lr * update
            ).astype(parameter.data.dtype)

    def get_config(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }
