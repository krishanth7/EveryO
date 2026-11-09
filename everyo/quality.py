"""CPU tensor correctness, diagnostics and reproducible benchmark helpers.

These utilities intentionally require no optional AI backends or GPU hardware.
"""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOGradientError, EveryOShapeError


@dataclass(frozen=True)
class TensorDiagnostic:
    """Inspectable tensor metadata and finite-value counts."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    size: int
    nan_count: int
    positive_inf_count: int
    negative_inf_count: int
    gradient_shape_valid: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic dictionary."""
        record = asdict(self)
        record["shape"] = list(self.shape)
        return record


def inspect_tensor(value: Tensor) -> TensorDiagnostic:
    """Report tensor invariants without mutating the tensor or its gradients."""
    if not isinstance(value, Tensor):
        raise TypeError("inspect_tensor expects an EveryO Tensor")
    array = value.data
    return TensorDiagnostic(
        shape=value.shape,
        dtype=value.dtype,
        device=str(value.device),
        requires_grad=value.requires_grad,
        size=value.size,
        nan_count=int(np.isnan(array).sum()),
        positive_inf_count=int(np.isposinf(array).sum()),
        negative_inf_count=int(np.isneginf(array).sum()),
        gradient_shape_valid=value.grad is None or np.shape(value.grad) == value.shape,
    )


def assert_tensor_valid(value: Tensor, *, finite: bool = False) -> None:
    """Raise actionable errors for malformed gradient state or non-finite data."""
    report = inspect_tensor(value)
    if not report.gradient_shape_valid:
        raise EveryOGradientError(
            f"Gradient shape {np.shape(value.grad)} differs from tensor shape {report.shape}."
        )
    if finite and (report.nan_count or report.positive_inf_count or report.negative_inf_count):
        raise ValueError(
            "Tensor contains non-finite values: "
            f"nan={report.nan_count}, +inf={report.positive_inf_count}, "
            f"-inf={report.negative_inf_count}."
        )


def safe_assign_(value: Tensor, data: Any) -> Tensor:
    """Replace untracked tensor storage after shape/dtype validation.

    Tracked tensors cannot be modified through this API. Direct writes to
    ``tensor.data``/NumPy views remain an unsafe escape hatch and are not
    intercepted by EveryO; callers must avoid them during autograd.
    """
    if not isinstance(value, Tensor):
        raise TypeError("safe_assign_ expects an EveryO Tensor")
    if value.requires_grad or value.grad_node is not None:
        raise EveryOGradientError(
            "Cannot mutate a tensor involved in autograd; detach and clone it"
        )
    replacement = np.asarray(data)
    if replacement.shape != value.shape:
        raise EveryOShapeError(
            f"Replacement shape {replacement.shape} does not match {value.shape}."
        )
    if replacement.dtype != value.data.dtype:
        raise TypeError(f"Replacement dtype {replacement.dtype} does not match {value.data.dtype}.")
    value.data = replacement.copy()
    return value


def index_select(value: Tensor, axis: int, indices: Sequence[int] | np.ndarray) -> Tensor:
    """Differentiable integer selection, including repeated indices."""
    from everyo.core.operations import slice_

    value = as_tensor(value)
    if value.ndim == 0 or not -value.ndim <= axis < value.ndim:
        raise EveryOShapeError(f"Axis {axis} is invalid for shape {value.shape}.")
    index_array = np.asarray(indices)
    if index_array.ndim != 1 or not np.issubdtype(index_array.dtype, np.integer):
        raise TypeError("indices must be a one-dimensional integer sequence")
    selectors: list[Any] = [slice(None)] * value.ndim
    selectors[axis] = index_array.astype(np.int64, copy=False)
    return slice_(value, tuple(selectors))


def logsumexp(
    value: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Numerically stable differentiable log(sum(exp(x))).

    Supports finite inputs and nonempty reduction dimensions. The backward
    pass is a softmax-weighted broadcast of the incoming gradient.
    """
    from everyo.core.operations import _make, _normalise_axis

    value = as_tensor(value)
    if not np.issubdtype(value.data.dtype, np.floating):
        raise TypeError("logsumexp requires a floating-point tensor")
    if not np.all(np.isfinite(value.data)):
        raise ValueError("logsumexp requires finite input values")
    axes = _normalise_axis(axis, value.ndim)
    if axes is not None and len(set(axes)) != len(axes):
        raise EveryOShapeError("Reduction axes must be unique")
    if axes is None and not value.size:
        raise EveryOShapeError("Cannot reduce an empty tensor")
    if axes is not None and any(value.shape[a] == 0 for a in axes):
        raise EveryOShapeError("Cannot reduce an empty axis")
    normalized = np.max(value.data, axis=axes, keepdims=True)
    weights = np.exp(value.data - normalized)
    denominator = np.sum(weights, axis=axes, keepdims=True)
    kept = normalized + np.log(denominator)
    result = kept if keepdims else np.squeeze(kept, axis=axes)

    def backward_fn(gradient: np.ndarray) -> tuple[np.ndarray]:
        expanded = np.asarray(gradient)
        if not keepdims:
            if axes is None:
                expanded = expanded.reshape((1,) * value.ndim)
            else:
                for dimension in sorted(axes):
                    expanded = np.expand_dims(expanded, dimension)
        return (np.broadcast_to(expanded, value.shape) * (weights / denominator),)

    return _make(
        result, (value,), "logsumexp", backward_fn, attributes={"axis": axes, "keepdims": keepdims}
    )


@dataclass(frozen=True)
class BenchmarkResult:
    """A reproducible run record, with timing kept separate from correctness."""

    name: str
    runs: int
    warmups: int
    seed: int
    median_ms: float
    min_ms: float
    max_ms: float
    checksum: float
    python: str
    numpy: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        return asdict(self)


def compare_benchmarks(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    *,
    maximum_slowdown: float = 1.25,
) -> float:
    """Return candidate/baseline median ratio or raise on a regression.

    A baseline from a different runtime or platform is rejected because such
    comparisons are generally misleading. Results still require repeated runs
    on controlled hardware before being used as release gates.
    """
    if not np.isfinite(maximum_slowdown) or maximum_slowdown <= 0:
        raise ValueError("maximum_slowdown must be finite and positive")
    fields = ("name", "seed", "python", "numpy", "platform")
    if any(getattr(baseline, field) != getattr(candidate, field) for field in fields):
        raise ValueError("Benchmark identity, seed or environment does not match")
    if baseline.median_ms <= 0:
        raise ValueError("Baseline median must be positive")
    if not np.isclose(baseline.checksum, candidate.checksum, rtol=1e-12, atol=1e-12):
        raise ValueError("Benchmark output checksum changed")
    ratio = candidate.median_ms / baseline.median_ms
    if ratio > maximum_slowdown:
        raise RuntimeError(f"CPU performance regression: {ratio:.2f}x baseline")
    return ratio


def benchmark_cpu(
    name: str,
    operation: Callable[[np.random.Generator], np.ndarray | Tensor],
    *,
    runs: int = 10,
    warmups: int = 3,
    seed: int = 0,
) -> BenchmarkResult:
    """Benchmark an operation with a fresh seeded RNG for every invocation.

    The benchmark detects nondeterministic results and reports hardware/runtime
    identity. Timing is not a portable performance guarantee.
    """
    if not name or not isinstance(name, str):
        raise ValueError("benchmark name must be a nonempty string")
    if not isinstance(runs, int) or runs < 1 or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("runs must be positive and warmups non-negative integers")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    durations = []
    reference = None
    for index in range(warmups + runs):
        rng = np.random.default_rng(seed)
        before = time.perf_counter_ns()
        output = operation(rng)
        elapsed_ms = (time.perf_counter_ns() - before) / 1e6
        array = output.data if isinstance(output, Tensor) else np.asarray(output)
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError("benchmark output must be a finite numeric array")
        if reference is None:
            reference = array.copy()
        elif not np.array_equal(reference, array):
            raise ValueError("Benchmark output changed between seeded invocations")
        if index >= warmups:
            durations.append(elapsed_ms)
    assert reference is not None
    return BenchmarkResult(
        name=name,
        runs=runs,
        warmups=warmups,
        seed=seed,
        median_ms=statistics.median(durations),
        min_ms=min(durations),
        max_ms=max(durations),
        checksum=float(np.sum(reference.astype(np.float64))),
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
    )
