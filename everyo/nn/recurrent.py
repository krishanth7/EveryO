"""Recurrent layers: RNN, LSTM and GRU.

Each layer walks the time axis, carrying a hidden state forward. The step
arithmetic is written with EveryO's differentiable primitives, so
backpropagation through time is just the autograd engine walking the graph the
loop built — there is no separate BPTT implementation to keep in sync.

One optimisation is worth knowing about: the input projection is computed for
*every* timestep in a single matrix multiplication before the loop starts. Only
the recurrent term, which genuinely depends on the previous step, is computed
inside it. That turns ``T`` small matmuls into one large one.

Inputs are ``(batch, time, features)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError
from everyo.nn.initialization import get_initializer
from everyo.nn.module import Module, Parameter, register_module

__all__ = ["RNN", "LSTM", "GRU"]


class _RecurrentBase(Module):
    """Shared plumbing: parameters, validation and the time loop."""

    #: Number of gate blocks packed into the weight matrices.
    _gates: int = 1

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        return_sequences: bool = True,
        initializer: str = "xavier_uniform",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if int(input_size) <= 0 or int(hidden_size) <= 0:
            raise EveryOShapeError(
                f"{type(self).__name__} requires positive sizes, got "
                f"input_size={input_size}, hidden_size={hidden_size}."
            )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.use_bias = bool(bias)
        self.return_sequences = bool(return_sequences)
        self.initializer = str(initializer)
        self.seed = seed

        width = self._gates * self.hidden_size
        init_fn = get_initializer(initializer)
        self.weight_ih = Parameter(
            np.asarray(init_fn((self.input_size, width), seed=seed), dtype=np.float32),
            name="weight_ih",
        )
        self.weight_hh = Parameter(
            np.asarray(
                init_fn((self.hidden_size, width), seed=None if seed is None else seed + 1),
                dtype=np.float32,
            ),
            name="weight_hh",
        )
        if self.use_bias:
            self.bias = Parameter(np.zeros(width, dtype=np.float32), name="bias")

    # ------------------------------------------------------------------
    def _check_input(self, value: Tensor) -> None:
        if value.ndim != 3:
            raise EveryOShapeError(
                f"{type(self).__name__} expects a 3-D input shaped "
                f"(batch, time, features), got shape {value.shape}."
            )
        if value.shape[-1] != self.input_size:
            raise EveryOShapeError(
                f"{type(self).__name__}(input_size={self.input_size}) received an "
                f"input whose last axis is {value.shape[-1]}."
            )
        if value.shape[1] == 0:
            raise EveryOShapeError(
                f"{type(self).__name__} received a sequence of length 0; there is "
                "nothing to unroll."
            )

    def _projected_inputs(self, value: Tensor) -> Tensor:
        """Project every timestep at once: ``(batch, time, gates * hidden)``."""
        batch, steps, _ = value.shape
        flat = ops.reshape(value, (batch * steps, self.input_size))
        projected = ops.matmul(flat, self.weight_ih)
        if self.use_bias:
            projected = ops.add(projected, self.bias)
        return ops.reshape(projected, (batch, steps, self._gates * self.hidden_size))

    @staticmethod
    def _gate(combined: Tensor, index: int, size: int) -> Tensor:
        """Slice gate ``index`` out of the packed activation."""
        return combined[:, index * size : (index + 1) * size]

    def forward(self, x: Any) -> Tensor:
        """Run the sequence through the layer.

        Returns:
            ``(batch, time, hidden_size)`` when ``return_sequences`` is set,
            otherwise the final step only, ``(batch, hidden_size)``.
        """
        value = as_tensor(x)
        self._check_input(value)
        batch, steps, _ = value.shape

        projected = self._projected_inputs(value)
        state = self._initial_state(batch)
        outputs: list[Tensor] = []

        for step in range(steps):
            state = self._step(projected[:, step, :], state)
            outputs.append(self._output(state))

        if not self.return_sequences:
            return outputs[-1]
        return ops.stack(outputs, axis=1)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _initial_state(self, batch: int) -> Any:
        return ops.zeros(batch, self.hidden_size)

    def _output(self, state: Any) -> Tensor:
        return state

    def _step(self, projected_step: Tensor, state: Any) -> Any:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "bias": self.use_bias,
            "return_sequences": self.return_sequences,
            "initializer": self.initializer,
        }


@register_module
class RNN(_RecurrentBase):
    """Elman recurrent layer: ``h = tanh(x W_ih + h W_hh + b)``.

    The simplest thing that can carry state across time, and the one whose
    gradient vanishes fastest — useful for short sequences and for seeing why
    :class:`LSTM` and :class:`GRU` exist.

    Args:
        input_size: Feature width of each timestep.
        hidden_size: Width of the hidden state.
        bias: Learn a bias term.
        return_sequences: Return every timestep, or only the last.
        initializer: Scheme from :mod:`everyo.nn.initialization`.
        seed: Makes initialisation reproducible.

    Example:
        >>> import everyo as eo
        >>> eo.RNN(8, 16)(eo.zeros(4, 12, 8)).shape
        (4, 12, 16)
    """

    _gates = 1

    def _step(self, projected_step: Tensor, state: Tensor) -> Tensor:
        return ops.tanh(ops.add(projected_step, ops.matmul(state, self.weight_hh)))


@register_module
class LSTM(_RecurrentBase):
    """Long short-term memory.

    Four gates are packed into one weight matrix in the order ``input, forget,
    cell, output``. The forget gate's bias starts at 1 by default
    (``unit_forget_bias``), which keeps the cell state from decaying away
    before the layer has learned anything. The cell state ``c`` is the layer's long-term memory: the
    forget gate decides what it keeps, the input gate what it adds, and the
    output gate how much of it reaches the hidden state. Because ``c`` is
    updated additively, its gradient does not get multiplied down at every
    step — which is the entire point.

    Args:
        input_size: Feature width of each timestep.
        hidden_size: Width of the hidden and cell states.
        bias: Learn bias terms.
        return_sequences: Return every timestep, or only the last.
        initializer: Scheme from :mod:`everyo.nn.initialization`.
        unit_forget_bias: Initialise the forget gate's bias to 1.
        seed: Makes initialisation reproducible.

    Example:
        >>> import everyo as eo
        >>> eo.LSTM(8, 16, return_sequences=False)(eo.zeros(4, 12, 8)).shape
        (4, 16)
    """

    _gates = 4

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        return_sequences: bool = True,
        initializer: str = "xavier_uniform",
        unit_forget_bias: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            bias=bias,
            return_sequences=return_sequences,
            initializer=initializer,
            seed=seed,
        )
        self.unit_forget_bias = bool(unit_forget_bias)
        if self.use_bias and self.unit_forget_bias:
            # Start the forget gate open. With a zero bias the gate sits at
            # sigmoid(0) = 0.5, so the cell state is halved at every step and
            # has decayed by ~1e-12 after 40 steps — destroying the long-range
            # memory the cell path exists to provide. Biasing it to 1 starts
            # the layer in "remember by default" and lets it learn to forget.
            values = self.bias.numpy()
            values[hidden_size : 2 * hidden_size] = 1.0
            self.bias.data = values

    def _initial_state(self, batch: int) -> tuple[Tensor, Tensor]:
        return ops.zeros(batch, self.hidden_size), ops.zeros(batch, self.hidden_size)

    def _output(self, state: tuple[Tensor, Tensor]) -> Tensor:
        return state[0]

    def _step(self, projected_step: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        hidden, cell = state
        combined = ops.add(projected_step, ops.matmul(hidden, self.weight_hh))
        size = self.hidden_size

        input_gate = ops.sigmoid(self._gate(combined, 0, size))
        forget_gate = ops.sigmoid(self._gate(combined, 1, size))
        candidate = ops.tanh(self._gate(combined, 2, size))
        output_gate = ops.sigmoid(self._gate(combined, 3, size))

        new_cell = ops.add(ops.multiply(forget_gate, cell), ops.multiply(input_gate, candidate))
        new_hidden = ops.multiply(output_gate, ops.tanh(new_cell))
        return new_hidden, new_cell

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        config = super().get_config()
        config["unit_forget_bias"] = self.unit_forget_bias
        return config


@register_module
class GRU(_RecurrentBase):
    """Gated recurrent unit.

    Three gates in the order ``reset, update, candidate``. A GRU reaches
    roughly LSTM-quality memory with one gate and one state fewer, so it trains
    faster and is a reasonable default for shorter sequences.

    Note:
        The reset gate is applied to the *recurrent* contribution only, which
        is the formulation in the original paper and the one cuDNN implements.

    Example:
        >>> import everyo as eo
        >>> eo.GRU(8, 16)(eo.zeros(4, 12, 8)).shape
        (4, 12, 16)
    """

    _gates = 3

    def _step(self, projected_step: Tensor, state: Tensor) -> Tensor:
        size = self.hidden_size
        recurrent = ops.matmul(state, self.weight_hh)

        reset = ops.sigmoid(
            ops.add(self._gate(projected_step, 0, size), self._gate(recurrent, 0, size))
        )
        update = ops.sigmoid(
            ops.add(self._gate(projected_step, 1, size), self._gate(recurrent, 1, size))
        )
        candidate = ops.tanh(
            ops.add(
                self._gate(projected_step, 2, size),
                ops.multiply(reset, self._gate(recurrent, 2, size)),
            )
        )
        return ops.add(
            ops.multiply(ops.subtract(1.0, update), candidate),
            ops.multiply(update, state),
        )
