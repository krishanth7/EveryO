"""CPU-only numerical higher-order derivatives and graph inspection.

These APIs intentionally do not claim native higher-order automatic
differentiation. EveryO's reverse-mode closures return NumPy arrays, so
the second derivative is approximated by finite differences of freshly
computed first derivatives. No caller-owned tensor is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from everyo.core.autograd import enable_grad, no_grad, topological_order
from everyo.core.tensor import Tensor, tensor


@dataclass(frozen=True)
class GraphReport:
    """Observability snapshot for the graph reachable from a tensor."""

    tensors: int
    operation_nodes: int
    edges: int
    saved_input_bytes: int
    retained_gradient_bytes: int


def graph_report(output: Tensor) -> GraphReport:
    """Count graph nodes, links and diagnostic snapshot memory."""
    if not isinstance(output, Tensor):
        raise TypeError("output must be a Tensor")
    nodes = topological_order(output)
    operations = [value.grad_node for value in nodes if value.grad_node is not None]
    return GraphReport(
        tensors=len(nodes),
        operation_nodes=len(operations),
        edges=sum(len(node.parents) for node in operations),
        saved_input_bytes=sum(
            sum(saved.nbytes for saved in node._input_snapshots)
            for node in operations
            if node._input_snapshots is not None
        ),
        retained_gradient_bytes=sum(value.grad.nbytes for value in nodes if value.grad is not None),
    )


def clear_intermediate_grads(output: Tensor) -> int:
    """Release retained intermediate gradients, preserving leaf gradients.

    This does not alter graph connectivity or invalidates subsequent backward.
    Returns the number of non-leaf gradient buffers cleared.
    """
    if not isinstance(output, Tensor):
        raise TypeError("output must be a Tensor")
    removed = 0
    for value in topological_order(output):
        if not value.is_leaf and value.grad is not None:
            value.grad = None
            removed += 1
    return removed


def _source(value: Tensor) -> np.ndarray:
    if not isinstance(value, Tensor) or value.dtype != "float64" or str(value.device) != "cpu":
        raise TypeError("input must be a float64 CPU Tensor")
    if not np.all(np.isfinite(value.data)):
        raise ValueError("input contains non-finite values")
    return value.data.copy()


def _step(eps: float) -> float:
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    return float(eps)


def _evaluate(function: Callable[[Tensor], Tensor], values: np.ndarray) -> np.ndarray:
    with no_grad():
        result = function(tensor(values.copy(), dtype="float64"))
    if not isinstance(result, Tensor):
        raise TypeError("function must return a Tensor")
    if not np.all(np.isfinite(result.data)):
        raise ValueError("function produced non-finite values")
    return result.data.astype("float64", copy=True)


def _gradient(function: Callable[[Tensor], Tensor], values: np.ndarray) -> np.ndarray:
    argument = tensor(values.copy(), dtype="float64", requires_grad=True)
    with enable_grad():
        output = function(argument)
    if not isinstance(output, Tensor) or output.size != 1:
        raise ValueError("function must return a scalar Tensor")
    if not np.all(np.isfinite(output.data)):
        raise ValueError("function produced a non-finite scalar")
    output.backward()
    if argument.grad is None or not np.all(np.isfinite(argument.grad)):
        raise ValueError("function must produce a finite first derivative")
    return argument.grad.astype("float64", copy=True)


def jacobian(
    function: Callable[[Tensor], Tensor], value: Tensor, *, eps: float = 1e-5
) -> np.ndarray:
    """Central-difference Jacobian with shape ``output.shape + input.shape``.

    Raises if output shapes change across probes. Complexity is two forward
    evaluations per input element. Input and output must be finite float64 CPU.
    """
    base, step = _source(value), _step(eps)
    reference = _evaluate(function, base)
    result = np.empty(reference.shape + base.shape, dtype="float64")
    for index in np.ndindex(base.shape):
        plus, minus = base.copy(), base.copy()
        plus[index] += step
        minus[index] -= step
        higher, lower = _evaluate(function, plus), _evaluate(function, minus)
        if higher.shape != reference.shape or lower.shape != reference.shape:
            raise ValueError("function output shape changed during Jacobian probes")
        result[(Ellipsis,) + index] = (higher - lower) / (2 * step)
    return result


def jvp(
    function: Callable[[Tensor], Tensor],
    value: Tensor,
    direction: np.ndarray,
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference Jacobian-vector product without building a Jacobian."""
    base, step = _source(value), _step(eps)
    vector = np.asarray(direction, dtype="float64")
    if vector.shape != base.shape or not np.all(np.isfinite(vector)):
        raise ValueError("direction must be finite and match input shape")
    plus = _evaluate(function, base + step * vector)
    minus = _evaluate(function, base - step * vector)
    if plus.shape != minus.shape:
        raise ValueError("function output shape changed during directional probes")
    return (plus - minus) / (2 * step)


def hessian(
    function: Callable[[Tensor], Tensor], value: Tensor, *, eps: float = 1e-4
) -> np.ndarray:
    """Finite-difference-of-autograd Hessian for a scalar function.

    The output has shape ``input.shape + input.shape``. This is approximate,
    and its columns correspond to input perturbations. No graph is retained.
    """
    base, step = _source(value), _step(eps)
    result = np.empty(base.shape + base.shape, dtype="float64")
    for index in np.ndindex(base.shape):
        plus, minus = base.copy(), base.copy()
        plus[index] += step
        minus[index] -= step
        result[(Ellipsis,) + index] = (_gradient(function, plus) - _gradient(function, minus)) / (
            2 * step
        )
    return result


def hvp(
    function: Callable[[Tensor], Tensor],
    value: Tensor,
    direction: np.ndarray,
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    """Hessian-vector product via two first-order reverse-mode passes."""
    base, step = _source(value), _step(eps)
    vector = np.asarray(direction, dtype="float64")
    if vector.shape != base.shape or not np.all(np.isfinite(vector)):
        raise ValueError("direction must be finite and match input shape")
    return (
        _gradient(function, base + step * vector) - _gradient(function, base - step * vector)
    ) / (2 * step)
