"""Optimizer base class."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from everyo.core.tensor import Tensor
from everyo.exceptions import EveryOError

__all__ = ["Optimizer"]


class Optimizer:
    """Base class for gradient based optimizers.

    Args:
        parameters: The tensors to update.  Typically ``model.parameters()``.
        lr: Learning rate; must be positive.

    Subclasses implement :meth:`step`, which is called after ``loss.backward()``.
    """

    def __init__(self, parameters: Iterable[Tensor], lr: float) -> None:
        params = list(parameters)
        if not params:
            raise EveryOError(
                "The optimizer received no parameters. Pass model.parameters() "
                "from a model that owns at least one Linear layer."
            )
        for index, parameter in enumerate(params):
            if not isinstance(parameter, Tensor):
                raise EveryOError(
                    f"Optimizer parameter {index} is a {type(parameter).__name__}, "
                    "but every parameter must be an EveryO tensor."
                )
        if float(lr) <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}.")

        self.parameters: list[Tensor] = params
        self.lr = float(lr)
        self.step_count = 0
        #: Per-parameter optimizer state, keyed by ``id(parameter)``.
        self.state: dict[int, dict[str, np.ndarray]] = {}

    def zero_grad(self) -> None:
        """Clear the gradients of every parameter.

        Gradients accumulate, so this must be called before each backward pass
        unless accumulation is intended.
        """
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        """Apply one optimization step.  Implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} does not implement step().")

    def _pending(self) -> list[tuple[Tensor, np.ndarray]]:
        """Return ``(parameter, gradient)`` pairs that have a gradient."""
        return [
            (parameter, parameter.grad)
            for parameter in self.parameters
            if parameter.requires_grad and parameter.grad is not None
        ]

    def state_for(self, parameter: Tensor) -> dict[str, np.ndarray]:
        """Return (creating if needed) the state dictionary of ``parameter``."""
        return self.state.setdefault(id(parameter), {})

    def get_config(self) -> dict[str, Any]:
        """Return the hyper-parameters of this optimizer."""
        return {"lr": self.lr}

    def __repr__(self) -> str:
        settings = ", ".join(f"{key}={value!r}" for key, value in self.get_config().items())
        return f"{type(self).__name__}({settings}, parameters={len(self.parameters)})"
