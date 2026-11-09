"""CPU-only numerical verification utilities for custom EveryO operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from everyo.core.autograd import enable_grad, no_grad
from everyo.core.tensor import Tensor, tensor


@dataclass(frozen=True)
class GradcheckResult:
    """Summary of an analytic-versus-central-difference gradient comparison."""

    passed: bool
    checked_elements: int
    max_absolute_error: float
    max_relative_error: float
    failing_elements: int


def gradcheck(
    function: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> GradcheckResult:
    """Check gradients with central differences, without mutating caller tensors.

    The function must be deterministic and return a scalar EveryO tensor.
    Inputs must be nonempty, finite, float64 CPU tensors. Each probe builds
    fresh tensors so no prior gradients or computation graphs are modified.
    """
    if not inputs:
        raise ValueError("gradcheck requires at least one input")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")
    arrays = []
    for index, value in enumerate(inputs):
        if not isinstance(value, Tensor) or value.dtype != "float64" or str(value.device) != "cpu":
            raise TypeError(f"input {index} must be a float64 CPU Tensor")
        if not np.all(np.isfinite(value.data)):
            raise ValueError(f"input {index} contains non-finite values")
        arrays.append(value.numpy())

    def evaluate(values: list[np.ndarray], *, backward: bool) -> tuple[float, list[np.ndarray]]:
        args = [tensor(a.copy(), dtype="float64", requires_grad=backward) for a in values]
        with enable_grad() if backward else no_grad():
            output = function(*args)
        if not isinstance(output, Tensor) or output.size != 1:
            raise ValueError("gradcheck function must return a scalar EveryO Tensor")
        result = float(output.item())
        if not np.isfinite(result):
            raise ValueError("gradcheck function returned a non-finite value")
        if backward:
            if not output.requires_grad:
                raise ValueError("gradcheck function output does not track gradients")
            output.backward()
            grads = []
            for arg in args:
                if arg.grad is None:
                    raise ValueError(
                        "gradcheck function did not produce a gradient for every input"
                    )
                grads.append(arg.grad.copy())
            return result, grads
        return result, []

    _, analytical = evaluate(arrays, backward=True)
    max_abs = 0.0
    max_rel = 0.0
    failures = 0
    checked = 0
    for array_index, base in enumerate(arrays):
        for element_index in np.ndindex(base.shape):
            plus = [a.copy() for a in arrays]
            minus = [a.copy() for a in arrays]
            plus[array_index][element_index] += eps
            minus[array_index][element_index] -= eps
            forward, _ = evaluate(plus, backward=False)
            backward_value, _ = evaluate(minus, backward=False)
            numerical = (forward - backward_value) / (2 * eps)
            actual = float(analytical[array_index][element_index])
            error = abs(actual - numerical)
            relative = error / max(abs(actual), abs(numerical), 1e-12)
            max_abs = max(max_abs, error)
            max_rel = max(max_rel, relative)
            failures += int(not np.isfinite(actual) or error > atol + rtol * abs(numerical))
            checked += 1
    return GradcheckResult(failures == 0, checked, max_abs, max_rel, failures)
