"""RMSprop optimizer with optional centered variance and momentum."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from everyo.core.tensor import Tensor
from everyo.optim.optimizer import Optimizer

__all__ = ["RMSprop"]


class RMSprop(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.01,
        *,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        centered: bool = False,
    ) -> None:
        super().__init__(parameters, lr)
        if not np.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        self.alpha, self.eps, self.momentum = float(alpha), float(eps), float(momentum)
        self.weight_decay, self.centered = float(weight_decay), bool(centered)

    def step(self) -> None:
        self.step_count += 1
        for parameter, gradient in self._pending():
            grad = np.asarray(gradient, dtype=np.float64)
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError("non-finite gradient in RMSprop")
            if self.weight_decay:
                grad = grad + self.weight_decay * parameter.data
            state = self.state_for(parameter)
            if not state:
                state["square_avg"] = np.zeros_like(grad)
                if self.centered:
                    state["grad_avg"] = np.zeros_like(grad)
                if self.momentum:
                    state["momentum_buffer"] = np.zeros_like(grad)
            state["square_avg"] = self.alpha * state["square_avg"] + (1 - self.alpha) * grad**2
            variance = state["square_avg"]
            if self.centered:
                state["grad_avg"] = self.alpha * state["grad_avg"] + (1 - self.alpha) * grad
                variance = np.maximum(variance - state["grad_avg"] ** 2, 0)
            update = grad / (np.sqrt(variance) + self.eps)
            if self.momentum:
                state["momentum_buffer"] = self.momentum * state["momentum_buffer"] + update
                update = state["momentum_buffer"]
            parameter.data = (parameter.data - self.lr * update).astype(parameter.data.dtype)

    def get_config(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "alpha": self.alpha,
            "eps": self.eps,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "centered": self.centered,
        }
