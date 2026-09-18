"""Autocast: per-operation precision control.

This module holds the thread-local autocast state and the ``maybe_cast`` hook
that the operations on the allowlist call. It lives under ``everyo.core`` so
that :mod:`everyo.core.operations` can import it without a cycle; the public
names are re-exported from :mod:`everyo.precision` and the top-level package.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

import numpy as np

from everyo.core.dtype import resolve_dtype
from everyo.exceptions import EveryOError

__all__ = [
    "AUTOCAST_OPS",
    "autocast",
    "autocast_dtype",
    "autocast_like",
    "is_autocast_enabled",
    "match_reduced_dtype",
]

#: Operations that run in reduced precision under autocast. Everything else
#: stays in float32 — the list is short on purpose. Reductions, exponentials,
#: normalization statistics and losses are exactly where 16 bits runs out of
#: range, so they are deliberately absent.
AUTOCAST_OPS = frozenset({"matmul", "conv2d"})


class _AutocastState(threading.local):
    enabled: bool = False
    dtype: Any = np.float16


_STATE = _AutocastState()


def is_autocast_enabled() -> bool:
    """Return ``True`` when autocast is active on this thread."""
    return _STATE.enabled


def autocast_dtype() -> Any:
    """Return the dtype autocast is currently casting to."""
    return _STATE.dtype


class autocast:
    """Run the heavy operations of a block in reduced precision.

    Only the operations in :data:`AUTOCAST_OPS` are affected: ``matmul`` (and
    therefore every ``Linear`` and attention projection) and ``conv2d``. Their
    floating-point inputs are cast down for the forward pass, so activations
    become ``float16``; their gradients are cast back up on the way out, so
    parameters and their gradients stay ``float32``. That is the master-weight
    arrangement standard mixed precision relies on.

    Args:
        enabled: Set ``False`` to make the block a no-op, which is handy for
            switching mixed precision off from a config flag.
        dtype: Reduced dtype to use; ``"float16"`` by default.

    Raises:
        EveryOError: If ``dtype`` is not a floating dtype.

    Example:
        >>> import everyo as eo
        >>> layer = eo.Linear(4, 3, seed=0)
        >>> with eo.autocast():
        ...     output = layer(eo.zeros(2, 4))
        >>> output.dtype
        'float16'
        >>> layer.weight.dtype
        'float32'
    """

    def __init__(self, enabled: bool = True, *, dtype: Any = "float16") -> None:
        self.enabled = bool(enabled)
        self.dtype = resolve_dtype(dtype)
        if not np.issubdtype(self.dtype, np.floating):
            raise EveryOError(
                f"autocast needs a floating dtype, got '{np.dtype(self.dtype).name}'."
            )
        self._previous: list[tuple[bool, Any]] = []

    def __enter__(self) -> autocast:
        self._previous.append((_STATE.enabled, _STATE.dtype))
        _STATE.enabled = self.enabled
        _STATE.dtype = self.dtype
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._previous:
            _STATE.enabled, _STATE.dtype = self._previous.pop()

    def __call__(self, func: Any) -> Any:
        """Use the same instance as a decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with autocast(self.enabled, dtype=self.dtype):
                return func(*args, **kwargs)

        return wrapper


def autocast_like(tensor: Any) -> Any:
    """Differentiably cast a tensor to the autocast dtype.

    This is the single mechanism autocast is built on. An operation on the
    allowlist routes each floating operand through here before computing, which
    inserts an explicit cast node in the graph: the forward pass then runs in
    reduced precision, and the cast node's backward pass converts the incoming
    gradient back to the operand's own dtype. A float32 parameter therefore
    keeps a float32 gradient even though the operation around it ran in
    float16 — the "fp32 master weights" arrangement — while the activations and
    their gradients genuinely stay in float16.

    Layers also call it for the operands they add to an autocast result (a bias,
    a residual branch), so the addition does not promote the activation back to
    float32.

    When autocast is off, or the tensor is already the target dtype, or it is
    not a floating tensor, this returns the input unchanged and adds no node to
    the graph.
    """
    if not _STATE.enabled:
        return tensor
    from everyo.core import operations as ops
    from everyo.core.tensor import as_tensor

    value = as_tensor(tensor)
    if not np.issubdtype(value.data.dtype, np.floating) or value.data.dtype == _STATE.dtype:
        return value
    return ops.to(value, dtype=_STATE.dtype)


def match_reduced_dtype(gradient: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Give ``gradient`` the reduced dtype of the tensor it belongs to.

    The gradient of a float16 activation is itself a float16 quantity, and it
    has to be represented as one for mixed precision to behave honestly: this
    is what makes small gradients actually underflow, which is in turn what
    :class:`~everyo.precision.GradScaler` exists to prevent. Without this the
    backward pass would quietly run in float32 and loss scaling would be
    decorative.

    Only reduced dtypes (narrower than float32) are applied; everything else is
    returned untouched, so the default float32 path is unaffected.
    """
    if data.dtype.itemsize < 4 and np.issubdtype(data.dtype, np.floating):
        return gradient.astype(data.dtype, copy=False)
    return gradient
