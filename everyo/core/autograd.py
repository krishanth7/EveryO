"""Reverse-mode automatic differentiation engine.

EveryO builds a dynamic computation graph: every differentiable operation
records the tensors it consumed and a closure that converts an incoming
gradient into gradients for those inputs.  :func:`backward` then walks the graph
in reverse topological order and accumulates ``Tensor.grad``.

The engine is intentionally compact so that it can be read end to end, but it
is a real implementation: the gradients it produces are checked against
finite-difference approximations in ``tests/test_autograd.py``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np

from everyo.exceptions import EveryOGradientError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from everyo.core.tensor import Tensor

__all__ = [
    "Node",
    "backward",
    "is_grad_enabled",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "unbroadcast",
    "topological_order",
]


class _GradState(threading.local):
    """Thread-local flag controlling whether new graph nodes are recorded."""

    enabled: bool = True


_STATE = _GradState()


def is_grad_enabled() -> bool:
    """Return ``True`` when operations should record gradient information."""
    return _STATE.enabled


class set_grad_enabled:
    """Context manager and decorator that sets the gradient-recording flag."""

    def __init__(self, mode: bool) -> None:
        self.mode = bool(mode)
        self._previous = is_grad_enabled()

    def __enter__(self) -> set_grad_enabled:
        self._previous = is_grad_enabled()
        _STATE.enabled = self.mode
        return self

    def __exit__(self, *exc_info: object) -> None:
        _STATE.enabled = self._previous

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with set_grad_enabled(self.mode):
                return func(*args, **kwargs)

        return wrapper


class no_grad(set_grad_enabled):
    """Disable gradient tracking inside the block.

    Example:
        >>> import everyo as eo
        >>> x = eo.tensor([1.0], requires_grad=True)
        >>> with eo.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
    """

    def __init__(self) -> None:
        super().__init__(False)


class enable_grad(set_grad_enabled):
    """Re-enable gradient tracking inside the block."""

    def __init__(self) -> None:
        super().__init__(True)


class Node:
    """A node in the computation graph.

    Attributes:
        operation: Human readable name of the producing operation.
        parents: Input tensors that participated in the operation.
        backward_fn: Callable mapping the output gradient to a tuple of
            gradients, one per parent (``None`` for parents that do not need one).
    """

    __slots__ = ("operation", "parents", "backward_fn")

    def __init__(
        self,
        operation: str,
        parents: Sequence[Tensor],
        backward_fn: Callable[[np.ndarray], Sequence[np.ndarray | None]],
    ) -> None:
        self.operation = operation
        self.parents = tuple(parents)
        self.backward_fn = backward_fn

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Node(operation={self.operation!r}, parents={len(self.parents)})"


def unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``gradient`` back to ``shape`` after NumPy broadcasting.

    When a forward operation broadcast an input, the gradient flowing back has
    the broadcast (larger) shape.  Summing over the expanded axes restores the
    original shape, which is exactly the derivative of broadcasting.

    Args:
        gradient: Gradient with the broadcast shape.
        shape: Shape of the original operand.

    Returns:
        A gradient array with exactly ``shape``.
    """
    gradient = np.asarray(gradient)
    if gradient.shape == shape:
        return gradient

    # Sum away leading axes that broadcasting added.
    extra_dims = gradient.ndim - len(shape)
    for _ in range(extra_dims):
        gradient = gradient.sum(axis=0)

    # Sum (keeping dims) over axes that were size 1 in the original operand.
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


def topological_order(root: Tensor) -> list[Tensor]:
    """Return graph tensors ordered so that consumers precede their producers.

    The traversal is iterative to keep deep graphs from exhausting the Python
    recursion limit.
    """
    order: list[Tensor] = []
    visited: set[int] = set()
    # (tensor, children_expanded)
    stack: list[tuple[Tensor, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if expanded:
            order.append(node)
            continue
        if key in visited:
            continue
        visited.add(key)
        stack.append((node, True))
        if node.grad_node is not None:
            for parent in node.grad_node.parents:
                if id(parent) not in visited:
                    stack.append((parent, False))

    order.reverse()
    return order


def backward(root: Tensor, gradient: np.ndarray | None = None) -> None:
    """Accumulate gradients for every tensor that contributed to ``root``.

    Args:
        root: The tensor to differentiate, typically a scalar loss.
        gradient: Seed gradient.  Defaults to ones for a scalar ``root``.

    Raises:
        EveryOGradientError: If ``root`` does not require gradients, if a seed
            gradient is needed but missing, or if the seed shape is wrong.
    """
    if not root.requires_grad:
        raise EveryOGradientError(
            "backward() was called on a tensor that does not require gradients. "
            "Create it with requires_grad=True, or make sure it was produced "
            "from tensors that do (and not inside an eo.no_grad() block)."
        )

    if gradient is None:
        if root.size != 1:
            raise EveryOGradientError(
                f"backward() on a non-scalar tensor with shape {root.shape} "
                "needs an explicit gradient argument of the same shape. "
                "Reduce the tensor first (for example with .mean()) or pass "
                "gradient=... explicitly."
            )
        seed = np.ones_like(root.data)
    else:
        seed = np.asarray(gradient, dtype=root.data.dtype)
        if seed.shape != root.shape:
            raise EveryOGradientError(
                f"The gradient passed to backward() has shape {seed.shape} but "
                f"the tensor has shape {root.shape}; they must match."
            )

    gradients: dict[int, np.ndarray] = {id(root): seed}

    for node in topological_order(root):
        incoming = gradients.get(id(node))
        if incoming is None:
            continue
        if node.requires_grad and (node.is_leaf or node.retains_grad):
            node.accumulate_grad(incoming)
        if node.grad_node is None:
            continue

        parent_grads = node.grad_node.backward_fn(incoming)
        parents = node.grad_node.parents
        if len(parent_grads) != len(parents):  # pragma: no cover - internal bug guard
            raise EveryOGradientError(
                f"Operation '{node.grad_node.operation}' produced "
                f"{len(parent_grads)} gradients for {len(parents)} inputs."
            )
        for parent, parent_grad in zip(parents, parent_grads):
            if parent_grad is None or not parent.requires_grad:
                continue
            parent_grad = np.asarray(parent_grad)
            existing = gradients.get(id(parent))
            gradients[id(parent)] = parent_grad if existing is None else existing + parent_grad


def collect_parents(tensors: Iterable[Any]) -> list[Tensor]:
    """Return the subset of ``tensors`` that are tensors requiring gradients."""
    from everyo.core.tensor import Tensor

    return [t for t in tensors if isinstance(t, Tensor) and t.requires_grad]
