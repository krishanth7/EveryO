"""Mixed-precision training.

Mixed precision runs the expensive parts of a network in 16-bit while keeping
the parts that need range in 32-bit. Two mechanisms make that safe:

**Autocast** decides per operation which precision to use. Matrix
multiplications and convolutions — long chains of multiply-accumulate, where
16 bits of mantissa is plenty — run in float16. Reductions, exponentials,
normalization and losses stay in float32, because summing thousands of values
or exponentiating one is exactly where 16 bits runs out of range. Parameters
and their gradients are never stored in float16: the gradients handed back by
an autocast operation are cast up to the parameter's own dtype, which is the
"fp32 master weights" arrangement.

**Gradient scaling** fixes the other half of the problem. float16's smallest
normal value is about 6.1e-5, and real gradients are routinely smaller than
that, so they flush to zero and the weights never move. Multiplying the loss by
a large constant before ``backward()`` shifts the whole gradient distribution
into representable range; :class:`GradScaler` divides it back out before
stepping. The scale is chosen dynamically: raise it while steps succeed, halve
it the moment a gradient overflows to infinity, and skip that step.

.. warning::
   On a CPU this saves **memory, not time**. NumPy has no native float16
   arithmetic — it upcasts to float32 to compute and casts back — so float16
   work is typically *slower* here. The speedup mixed precision is famous for
   comes from GPU tensor cores. EveryO implements the numerics honestly so the
   behaviour is right; it does not pretend to make your CPU faster.

Example:
    >>> import everyo as eo
    >>> model = eo.Sequential(eo.Linear(8, 1, seed=0))
    >>> optimizer = eo.SGD(model.parameters(), lr=0.1)
    >>> scaler = eo.GradScaler()
    >>> inputs, targets = eo.ones(4, 8), eo.zeros(4, 1)
    >>> with eo.autocast():
    ...     residual = model(inputs) - targets
    >>> loss = eo.mean(residual * residual)
    >>> optimizer.zero_grad()
    >>> scaler.backward(loss)
    >>> applied = scaler.step(optimizer)
    >>> scaler.update()
"""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo._logging import get_logger
from everyo.core.autocast import (
    AUTOCAST_OPS,
    autocast,
    autocast_dtype,
    autocast_like,
    is_autocast_enabled,
)
from everyo.core.tensor import Tensor

__all__ = [
    "AUTOCAST_OPS",
    "GradScaler",
    "autocast",
    "autocast_dtype",
    "autocast_like",
    "is_autocast_enabled",
]

_LOGGER = get_logger(__name__)


class GradScaler:
    """Dynamic loss scaling for float16 training.

    Args:
        init_scale: Starting scale factor.
        growth_factor: Multiplier applied after ``growth_interval`` good steps.
        backoff_factor: Multiplier applied when a gradient overflows.
        growth_interval: Consecutive good steps before the scale grows.
        enabled: Set ``False`` to make every method a pass-through.

    Example:
        >>> import everyo as eo
        >>> model = eo.Sequential(eo.Linear(4, 2, seed=0))
        >>> optimizer = eo.Adam(model.parameters(), lr=0.01)
        >>> scaler = eo.GradScaler()
        >>> with eo.autocast():
        ...     loss = eo.mean(model(eo.ones(2, 4)))
        >>> scaler.backward(loss)
        >>> applied = scaler.step(optimizer)
        >>> scaler.update()
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        *,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        if float(init_scale) <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}.")
        if float(growth_factor) <= 1.0:
            raise ValueError(f"growth_factor must exceed 1, got {growth_factor}.")
        if not 0.0 < float(backoff_factor) < 1.0:
            raise ValueError(f"backoff_factor must be in (0, 1), got {backoff_factor}.")
        if int(growth_interval) <= 0:
            raise ValueError(f"growth_interval must be positive, got {growth_interval}.")

        self.enabled = bool(enabled)
        self._scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._good_steps = 0
        #: Number of steps skipped because a gradient was not finite.
        self.skipped_steps = 0
        #: Number of optimizer steps actually applied.
        self.applied_steps = 0
        self._found_inf = False

    def get_scale(self) -> float:
        """Return the current scale factor."""
        return self._scale if self.enabled else 1.0

    def scale(self, loss: Tensor) -> Tensor:
        """Multiply ``loss`` by the current scale, ready for ``backward()``."""
        if not self.enabled:
            return loss
        from everyo.core import operations as ops

        return ops.multiply(loss, self._scale)

    def backward(self, loss: Tensor) -> None:
        """Scale ``loss`` and run its backward pass.

        Equivalent to ``scaler.scale(loss).backward()``, except that NumPy's
        overflow warnings are suppressed for the duration. An overflow in a
        scaled float16 backward pass is not a bug to be warned about -- it is
        the signal :meth:`step` is watching for, and it responds by skipping the
        step and halving the scale. Leaving the warning on would print a stream
        of ``RuntimeWarning: overflow encountered in matmul`` at exactly the
        moments the scaler is working correctly.

        Use ``scaler.scale(loss).backward()`` directly if you would rather see
        them.
        """
        scaled = self.scale(loss)
        if not self.enabled:
            scaled.backward()
            return
        with np.errstate(over="ignore", invalid="ignore"):
            scaled.backward()

    def unscale_(self, optimizer: Any) -> bool:
        """Divide the gradients back down and report whether they are finite.

        Returns:
            ``True`` when every gradient is finite, ``False`` when the step
            should be skipped.
        """
        if not self.enabled:
            return True

        finite = True
        for parameter in optimizer.parameters:
            if parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad, dtype=np.float32) / self._scale
            if not np.all(np.isfinite(gradient)):
                finite = False
            parameter.grad = gradient.astype(parameter.data.dtype)

        self._found_inf = not finite
        return finite

    def step(self, optimizer: Any) -> bool:
        """Unscale, then step the optimizer unless a gradient overflowed.

        Returns:
            ``True`` if the step was applied, ``False`` if it was skipped.
        """
        if not self.enabled:
            optimizer.step()
            self.applied_steps += 1
            return True

        if self.unscale_(optimizer):
            optimizer.step()
            self.applied_steps += 1
            return True

        self.skipped_steps += 1
        _LOGGER.debug("Skipping optimizer step: a gradient overflowed at scale %g.", self._scale)
        return False

    def update(self, new_scale: float | None = None) -> None:
        """Grow or shrink the scale based on the last step."""
        if not self.enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
            self._good_steps = 0
            return

        if self._found_inf:
            self._scale = max(self._scale * self.backoff_factor, 1.0)
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self._scale *= self.growth_factor
                self._good_steps = 0
        self._found_inf = False

    def state_dict(self) -> dict[str, Any]:
        """Return the scaler state, for checkpointing."""
        return {
            "scale": self._scale,
            "good_steps": self._good_steps,
            "skipped_steps": self.skipped_steps,
            "applied_steps": self.applied_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        self._scale = float(state["scale"])
        self._good_steps = int(state.get("good_steps", 0))
        self.skipped_steps = int(state.get("skipped_steps", 0))
        self.applied_steps = int(state.get("applied_steps", 0))

    def __repr__(self) -> str:
        return (
            f"GradScaler(scale={self._scale:g}, applied={self.applied_steps}, "
            f"skipped={self.skipped_steps}, enabled={self.enabled})"
        )
