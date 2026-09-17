"""Differentiable tensor operations.

Each public function here does three things:

1. validates its inputs and produces a helpful error when they are wrong,
2. computes the forward result (dispatching to the CUDA backend when the
   tensors live on a CUDA device and a kernel exists),
3. records a :class:`~everyo.core.autograd.Node` so the result can be
   differentiated.

Gradients are only recorded when at least one input requires them and gradient
mode is enabled, which keeps inference cheap.
"""

from __future__ import annotations

import builtins
from typing import Any, Sequence

import numpy as np

from everyo.core import autograd
from everyo.core.autograd import Node, unbroadcast
from everyo.core.device import Device, DeviceLike, resolve_device
from everyo.core.dtype import DEFAULT_FLOAT_DTYPE, is_floating, resolve_dtype
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError

__all__ = [
    # creation
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "arange",
    "linspace",
    "eye",
    "uniform",
    "normal",
    "rand",
    "randn",
    "one_hot",
    # arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "negative",
    "power",
    "exp",
    "log",
    "sqrt",
    "abs",
    "clip",
    # linear algebra
    "matmul",
    "dot",
    # reductions
    "sum",
    "mean",
    "max",
    "min",
    "variance",
    # shape
    "reshape",
    "transpose",
    "flatten",
    "concatenate",
    "stack",
    "slice_",
    # activations (numerics; the nn package wraps these as modules)
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
]

_EPSILON = 1e-12


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _result_device(tensors: Sequence[Tensor]) -> Device:
    """Pick the device for an operation result.

    Tensor buffers always live in host memory, so mixing devices is harmless.
    When any operand is on CUDA the result stays on CUDA, which keeps chained
    GPU operations on the GPU path.
    """
    for item in tensors:
        if item.device.is_cuda:
            return item.device
    return tensors[0].device if tensors else resolve_device("cpu")


def _dispatch(name: str, device: Device, *arrays: np.ndarray) -> np.ndarray:
    """Run kernel ``name`` on the CUDA backend when possible, else on NumPy."""
    if device.is_cuda:
        from everyo.cuda import interface as cuda_interface

        kernel = cuda_interface.get_kernel(name)
        if kernel is not None:
            return np.asarray(kernel(*arrays))
    from everyo.backends import numpy_backend

    return np.asarray(getattr(numpy_backend, name)(*arrays))


def _make(
    data: np.ndarray,
    inputs: Sequence[Tensor],
    operation: str,
    backward_fn: Any,
    device: Device | None = None,
) -> Tensor:
    """Wrap ``data`` in a tensor, attaching a graph node when required."""
    data = np.asarray(data)
    device = _result_device(inputs) if device is None else device
    track = (
        autograd.is_grad_enabled()
        and backward_fn is not None
        and any(item.requires_grad for item in inputs)
    )
    node = Node(operation, list(inputs), backward_fn) if track else None
    return Tensor._wrap(data, device=device, node=node)


def _binary_inputs(a: Any, b: Any, operation: str) -> tuple[Tensor, Tensor]:
    """Convert both operands to tensors and validate broadcast compatibility."""
    left = as_tensor(a)
    right = as_tensor(b)
    try:
        np.broadcast_shapes(left.shape, right.shape)
    except ValueError as exc:
        raise EveryOShapeError.for_broadcast(left.shape, right.shape, operation) from exc
    return left, right


def _normalise_axis(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...] | None:
    """Turn ``axis`` into a tuple of non-negative axes, or ``None`` for 'all'."""
    if axis is None:
        return None
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalised = []
    for item in axes:
        value = item + ndim if item < 0 else item
        if not 0 <= value < builtins.max(ndim, 1):
            raise EveryOShapeError(
                f"Axis {item} is out of range for a tensor with {ndim} dimension(s)."
            )
        normalised.append(value)
    return tuple(normalised)


def _expand_for_reduction(
    gradient: np.ndarray,
    shape: tuple[int, ...],
    axes: tuple[int, ...] | None,
    keepdims: bool,
) -> np.ndarray:
    """Broadcast a reduced gradient back to the pre-reduction ``shape``."""
    if axes is not None and not keepdims:
        for axis in sorted(axes):
            gradient = np.expand_dims(gradient, axis)
    return np.broadcast_to(gradient, shape)


# ----------------------------------------------------------------------
# Creation
# ----------------------------------------------------------------------
def _creation(
    data: np.ndarray,
    dtype: Any | None,
    device: DeviceLike,
    requires_grad: bool,
) -> Tensor:
    return Tensor(
        data,
        dtype=dtype or DEFAULT_FLOAT_DTYPE,
        device=device,
        requires_grad=requires_grad,
    )


def zeros(
    *shape: int | Sequence[int],
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with zeros."""
    dims = _shape_args(shape)
    return _creation(np.zeros(dims), dtype, device, requires_grad)


def ones(
    *shape: int | Sequence[int],
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with ones."""
    dims = _shape_args(shape)
    return _creation(np.ones(dims), dtype, device, requires_grad)


def full(
    shape: Sequence[int],
    value: float,
    *,
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with ``value``."""
    return _creation(np.full(tuple(shape), value), dtype, device, requires_grad)


def zeros_like(reference: Tensor, *, requires_grad: bool = False) -> Tensor:
    """Return zeros with the shape, dtype and device of ``reference``."""
    return Tensor(
        np.zeros_like(reference.data),
        device=reference.device,
        requires_grad=requires_grad,
    )


def ones_like(reference: Tensor, *, requires_grad: bool = False) -> Tensor:
    """Return ones with the shape, dtype and device of ``reference``."""
    return Tensor(
        np.ones_like(reference.data),
        device=reference.device,
        requires_grad=requires_grad,
    )


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    *,
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Return evenly spaced values within a half-open interval."""
    values = np.arange(start) if stop is None else np.arange(start, stop, step)
    return _creation(values, dtype, device, False)


def linspace(
    start: float,
    stop: float,
    num: int = 50,
    *,
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
) -> Tensor:
    """Return ``num`` evenly spaced samples from ``start`` to ``stop``."""
    return _creation(np.linspace(start, stop, num), dtype, device, False)


def eye(n: int, m: int | None = None, *, device: DeviceLike = "cpu") -> Tensor:
    """Return a 2-D identity-like tensor."""
    return _creation(np.eye(n, m), None, device, False)


def _generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def uniform(
    shape: Sequence[int],
    low: float = 0.0,
    high: float = 1.0,
    *,
    seed: int | np.random.Generator | None = None,
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """Sample from a uniform distribution over ``[low, high)``."""
    values = _generator(seed).uniform(low, high, size=tuple(shape))
    return _creation(values, dtype, device, requires_grad)


def normal(
    shape: Sequence[int],
    mean: float = 0.0,
    std: float = 1.0,
    *,
    seed: int | np.random.Generator | None = None,
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """Sample from a normal distribution."""
    values = _generator(seed).normal(mean, std, size=tuple(shape))
    return _creation(values, dtype, device, requires_grad)


def rand(*shape: int, seed: int | None = None, device: DeviceLike = "cpu") -> Tensor:
    """Shorthand for :func:`uniform` over ``[0, 1)``."""
    return uniform(_shape_args(shape), seed=seed, device=device)


def randn(*shape: int, seed: int | None = None, device: DeviceLike = "cpu") -> Tensor:
    """Shorthand for a standard :func:`normal` sample."""
    return normal(_shape_args(shape), seed=seed, device=device)


def one_hot(indices: Any, num_classes: int, *, dtype: Any | None = None) -> Tensor:
    """Convert integer class indices into a one-hot encoded tensor."""
    values = (
        np.asarray(indices.data if isinstance(indices, Tensor) else indices)
        .astype(np.int64)
        .reshape(-1)
    )
    if values.size and (values.min() < 0 or values.max() >= num_classes):
        raise EveryOShapeError(
            f"Class indices must be in [0, {num_classes - 1}] but the data "
            f"contains values in [{values.min()}, {values.max()}]."
        )
    encoded = np.zeros(
        (values.size, num_classes), dtype=resolve_dtype(dtype or DEFAULT_FLOAT_DTYPE)
    )
    encoded[np.arange(values.size), values] = 1.0
    return Tensor(encoded)


def _shape_args(shape: tuple[Any, ...]) -> tuple[int, ...]:
    """Allow both ``zeros(2, 3)`` and ``zeros((2, 3))``."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(int(dim) for dim in shape[0])
    return tuple(int(dim) for dim in shape)


# ----------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------
def add(a: Any, b: Any) -> Tensor:
    """Element-wise addition with broadcasting."""
    left, right = _binary_inputs(a, b, "add")
    device = _result_device((left, right))
    data = _dispatch("add", device, left.data, right.data)

    def backward_fn(gradient: np.ndarray):
        return (
            unbroadcast(gradient, left.shape),
            unbroadcast(gradient, right.shape),
        )

    return _make(data, (left, right), "add", backward_fn)


def subtract(a: Any, b: Any) -> Tensor:
    """Element-wise subtraction with broadcasting."""
    left, right = _binary_inputs(a, b, "subtract")
    data = np.subtract(left.data, right.data)

    def backward_fn(gradient: np.ndarray):
        return (
            unbroadcast(gradient, left.shape),
            unbroadcast(-gradient, right.shape),
        )

    return _make(data, (left, right), "subtract", backward_fn)


def multiply(a: Any, b: Any) -> Tensor:
    """Element-wise multiplication with broadcasting."""
    left, right = _binary_inputs(a, b, "multiply")
    device = _result_device((left, right))
    data = _dispatch("multiply", device, left.data, right.data)

    def backward_fn(gradient: np.ndarray):
        return (
            unbroadcast(gradient * right.data, left.shape),
            unbroadcast(gradient * left.data, right.shape),
        )

    return _make(data, (left, right), "multiply", backward_fn)


def divide(a: Any, b: Any) -> Tensor:
    """Element-wise division with broadcasting."""
    left, right = _binary_inputs(a, b, "divide")
    data = np.divide(left.data, right.data)

    def backward_fn(gradient: np.ndarray):
        return (
            unbroadcast(gradient / right.data, left.shape),
            unbroadcast(-gradient * left.data / (right.data**2), right.shape),
        )

    return _make(data, (left, right), "divide", backward_fn)


def negative(a: Any) -> Tensor:
    """Element-wise negation."""
    value = as_tensor(a)

    def backward_fn(gradient: np.ndarray):
        return (-gradient,)

    return _make(-value.data, (value,), "negative", backward_fn)


def power(a: Any, exponent: float) -> Tensor:
    """Raise a tensor to a scalar power."""
    value = as_tensor(a)
    if isinstance(exponent, Tensor):
        raise EveryOShapeError(
            "power() currently supports a scalar exponent only; got a tensor. "
            "Use exp(log(a) * b) for a tensor-valued exponent."
        )
    exponent = float(exponent)
    data = np.power(value.data, exponent)

    def backward_fn(gradient: np.ndarray):
        return (gradient * exponent * np.power(value.data, exponent - 1.0),)

    return _make(data, (value,), "power", backward_fn)


def exp(a: Any) -> Tensor:
    """Element-wise exponential."""
    value = as_tensor(a)
    data = np.exp(value.data)

    def backward_fn(gradient: np.ndarray):
        return (gradient * data,)

    return _make(data, (value,), "exp", backward_fn)


def log(a: Any) -> Tensor:
    """Element-wise natural logarithm."""
    value = as_tensor(a)
    data = np.log(value.data)

    def backward_fn(gradient: np.ndarray):
        return (gradient / value.data,)

    return _make(data, (value,), "log", backward_fn)


def sqrt(a: Any) -> Tensor:
    """Element-wise square root."""
    value = as_tensor(a)
    data = np.sqrt(value.data)

    def backward_fn(gradient: np.ndarray):
        return (gradient * 0.5 / np.maximum(data, _EPSILON),)

    return _make(data, (value,), "sqrt", backward_fn)


def abs(a: Any) -> Tensor:  # noqa: A001 - mirrors the NumPy name on purpose
    """Element-wise absolute value."""
    value = as_tensor(a)
    data = np.abs(value.data)

    def backward_fn(gradient: np.ndarray):
        return (gradient * np.sign(value.data),)

    return _make(data, (value,), "abs", backward_fn)


def clip(a: Any, low: float, high: float) -> Tensor:
    """Clamp values into ``[low, high]``; gradients vanish outside the range."""
    value = as_tensor(a)
    if low > high:
        raise EveryOShapeError(f"clip() requires low <= high, but got low={low} and high={high}.")
    data = np.clip(value.data, low, high)
    mask = (value.data >= low) & (value.data <= high)

    def backward_fn(gradient: np.ndarray):
        return (gradient * mask,)

    return _make(data, (value,), "clip", backward_fn)


# ----------------------------------------------------------------------
# Linear algebra
# ----------------------------------------------------------------------
def to(
    a: Any,
    device: DeviceLike | None = None,
    dtype: Any | None = None,
) -> Tensor:
    """Place a tensor on ``device`` and/or cast it, keeping the graph intact.

    Moving a tensor is an identity operation on its values, so the gradient
    flows straight through to the source. A cast to a non-floating dtype cannot
    carry gradients and therefore detaches, which is reported by the result
    having ``requires_grad=False`` rather than by a silent dead end.

    Args:
        a: The tensor to move.
        device: Target device, or ``None`` to keep the current one.
        dtype: Target dtype, or ``None`` to keep the current one.

    Returns:
        A tensor on the requested device with the requested dtype.
    """
    value = as_tensor(a)
    target_device = value.device if device is None else resolve_device(device)
    target_dtype = None if dtype is None else resolve_dtype(dtype)
    data = value.data if target_dtype is None else value.data.astype(target_dtype)
    source_dtype = value.data.dtype

    if target_dtype is not None and not is_floating(target_dtype):
        # dtype is passed explicitly: without it the Tensor constructor would
        # re-apply its "integers become float32" default and undo the cast.
        return Tensor(data, dtype=target_dtype, device=target_device)

    def backward_fn(gradient: np.ndarray):
        return (np.asarray(gradient).astype(source_dtype),)

    return _make(data, (value,), "to", backward_fn, device=target_device)


def matmul(a: Any, b: Any) -> Tensor:
    """Matrix multiplication for 1-D and 2-D tensors (with batched 2-D+ support).

    Raises:
        EveryOShapeError: If the inner dimensions do not agree.
    """
    left = as_tensor(a)
    right = as_tensor(b)
    if left.ndim == 0 or right.ndim == 0:
        raise EveryOShapeError(
            "matmul() requires tensors with at least one dimension; use '*' "
            "for scalar multiplication."
        )
    inner_left = left.shape[-1]
    inner_right = right.shape[0] if right.ndim == 1 else right.shape[-2]
    if inner_left != inner_right:
        raise EveryOShapeError.for_matmul(left.shape, right.shape)

    device = _result_device((left, right))
    data = _dispatch("matmul", device, left.data, right.data)

    left_is_vector = left.ndim == 1
    right_is_vector = right.ndim == 1

    def backward_fn(gradient: np.ndarray):
        grad = np.asarray(gradient)
        # Promote 1-D operands to matrices so one formula covers every rank
        # combination, including batched operands. NumPy's matmul does the same
        # promotion in the forward pass, so the shapes always line up.
        left_matrix = left.data[np.newaxis, :] if left_is_vector else left.data
        right_matrix = right.data[:, np.newaxis] if right_is_vector else right.data

        grad_matrix = grad
        if left_is_vector and right_is_vector:
            grad_matrix = grad.reshape(*grad.shape, 1, 1)
        elif left_is_vector:
            # (..., n) -> (..., 1, n)
            grad_matrix = grad[..., np.newaxis, :]
        elif right_is_vector:
            # (..., m) -> (..., m, 1)
            grad_matrix = grad[..., np.newaxis]

        grad_left = grad_matrix @ np.swapaxes(right_matrix, -1, -2)
        grad_right = np.swapaxes(left_matrix, -1, -2) @ grad_matrix

        # Drop the axis that promotion added, then let unbroadcast sum away any
        # batch dimensions the operand did not have.
        if left_is_vector:
            grad_left = grad_left[..., 0, :]
        if right_is_vector:
            grad_right = grad_right[..., 0]

        return (
            unbroadcast(np.asarray(grad_left), left.shape),
            unbroadcast(np.asarray(grad_right), right.shape),
        )

    return _make(data, (left, right), "matmul", backward_fn)


def dot(a: Any, b: Any) -> Tensor:
    """Dot product of two 1-D tensors, or matrix product for higher ranks."""
    left = as_tensor(a)
    right = as_tensor(b)
    if left.ndim == 1 and right.ndim == 1:
        if left.shape != right.shape:
            raise EveryOShapeError(
                f"dot() requires vectors of equal length, got {left.shape} and {right.shape}."
            )
        return sum(multiply(left, right))
    return matmul(left, right)


# ----------------------------------------------------------------------
# Reductions
# ----------------------------------------------------------------------
def sum(  # noqa: A001 - mirrors the NumPy name on purpose
    a: Any,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum elements over ``axis`` (all elements when ``axis`` is ``None``)."""
    value = as_tensor(a)
    axes = _normalise_axis(axis, value.ndim)
    if axes is None and not keepdims and value.device.is_cuda:
        data = _dispatch("sum_all", value.device, value.data)
    else:
        data = np.sum(value.data, axis=axes, keepdims=keepdims)
    shape = value.shape

    def backward_fn(gradient: np.ndarray):
        grad = _expand_for_reduction(np.asarray(gradient), shape, axes, keepdims)
        return (np.array(grad, copy=True),)

    return _make(data, (value,), "sum", backward_fn)


def mean(
    a: Any,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Arithmetic mean over ``axis``."""
    value = as_tensor(a)
    axes = _normalise_axis(axis, value.ndim)
    data = np.mean(value.data, axis=axes, keepdims=keepdims)
    shape = value.shape
    count = value.size if axes is None else int(np.prod([shape[i] for i in axes]))
    count = builtins.max(count, 1)

    def backward_fn(gradient: np.ndarray):
        grad = _expand_for_reduction(np.asarray(gradient), shape, axes, keepdims)
        return (np.array(grad, copy=True) / count,)

    return _make(data, (value,), "mean", backward_fn)


def _extremum(a: Any, axis: int | None, keepdims: bool, largest: bool) -> Tensor:
    value = as_tensor(a)
    axes = _normalise_axis(axis, value.ndim)
    reducer = np.max if largest else np.min
    data = reducer(value.data, axis=axes, keepdims=keepdims)
    shape = value.shape
    expanded = reducer(value.data, axis=axes, keepdims=True)
    mask = (value.data == expanded).astype(value.data.dtype)
    # Ties share the gradient equally, which keeps the operation well defined.
    mask = mask / np.maximum(mask.sum(axis=axes, keepdims=True), 1.0)
    name = "max" if largest else "min"

    def backward_fn(gradient: np.ndarray):
        grad = _expand_for_reduction(np.asarray(gradient), shape, axes, keepdims)
        return (grad * mask,)

    return _make(data, (value,), name, backward_fn)


def max(  # noqa: A001 - mirrors the NumPy name on purpose
    a: Any, axis: int | None = None, keepdims: bool = False
) -> Tensor:
    """Maximum over ``axis``.  Gradient flows to the maximal elements."""
    return _extremum(a, axis, keepdims, largest=True)


def min(  # noqa: A001 - mirrors the NumPy name on purpose
    a: Any, axis: int | None = None, keepdims: bool = False
) -> Tensor:
    """Minimum over ``axis``.  Gradient flows to the minimal elements."""
    return _extremum(a, axis, keepdims, largest=False)


def variance(a: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
    """Population variance, expressed with differentiable primitives."""
    value = as_tensor(a)
    centred = subtract(value, mean(value, axis=axis, keepdims=True))
    return mean(multiply(centred, centred), axis=axis, keepdims=keepdims)


# ----------------------------------------------------------------------
# Shape manipulation
# ----------------------------------------------------------------------
def reshape(a: Any, shape: Sequence[int]) -> Tensor:
    """Return a tensor with the same elements and a new shape."""
    value = as_tensor(a)
    target = tuple(int(dim) for dim in shape)
    try:
        data = value.data.reshape(target)
    except ValueError as exc:
        raise EveryOShapeError(
            f"Cannot reshape a tensor of shape {value.shape} ({value.size} elements) into {target}."
        ) from exc
    original = value.shape

    def backward_fn(gradient: np.ndarray):
        return (np.asarray(gradient).reshape(original),)

    return _make(data, (value,), "reshape", backward_fn)


def transpose(a: Any, axes: Sequence[int] | None = None) -> Tensor:
    """Permute tensor axes; defaults to reversing them."""
    value = as_tensor(a)
    order = None if axes is None else tuple(int(axis) for axis in axes)
    if order is not None and sorted(order) != list(range(value.ndim)):
        raise EveryOShapeError(
            f"transpose() received axes {order}, which is not a permutation of "
            f"the {value.ndim} tensor axes."
        )
    data = np.transpose(value.data, order)
    inverse = None if order is None else np.argsort(order)

    def backward_fn(gradient: np.ndarray):
        grad = np.asarray(gradient)
        return (np.transpose(grad, None if inverse is None else tuple(inverse)),)

    return _make(data, (value,), "transpose", backward_fn)


def flatten(a: Any, start_dim: int = 0) -> Tensor:
    """Collapse all dimensions from ``start_dim`` onwards into one axis."""
    value = as_tensor(a)
    if start_dim < 0:
        start_dim += value.ndim
    if not 0 <= start_dim <= builtins.max(value.ndim - 1, 0):
        raise EveryOShapeError(
            f"start_dim={start_dim} is out of range for a tensor with {value.ndim} dimension(s)."
        )
    head = value.shape[:start_dim]
    return reshape(value, (*head, -1) if value.size else (*head, 0))


def concatenate(tensors: Sequence[Any], axis: int = 0) -> Tensor:
    """Join tensors along an existing axis."""
    values = [as_tensor(item) for item in tensors]
    if not values:
        raise EveryOShapeError("concatenate() requires at least one tensor.")
    try:
        data = np.concatenate([item.data for item in values], axis=axis)
    except ValueError as exc:
        shapes = [item.shape for item in values]
        raise EveryOShapeError(
            f"Cannot concatenate tensors with shapes {shapes} along axis {axis}; "
            "every other dimension must match."
        ) from exc
    sizes = [item.shape[axis] for item in values]
    offsets = np.cumsum([0, *sizes])

    def backward_fn(gradient: np.ndarray):
        grad = np.asarray(gradient)
        pieces = []
        for index in range(len(values)):
            selector: list[Any] = [slice(None)] * grad.ndim
            selector[axis] = slice(int(offsets[index]), int(offsets[index + 1]))
            pieces.append(grad[tuple(selector)])
        return tuple(pieces)

    return _make(data, values, "concatenate", backward_fn)


def stack(tensors: Sequence[Any], axis: int = 0) -> Tensor:
    """Join tensors along a new axis."""
    values = [as_tensor(item) for item in tensors]
    if not values:
        raise EveryOShapeError("stack() requires at least one tensor.")
    expanded = [reshape(item, (*item.shape[:axis], 1, *item.shape[axis:])) for item in values]
    return concatenate(expanded, axis=axis)


def slice_(a: Any, index: Any) -> Tensor:
    """Differentiable indexing (``tensor[...]``)."""
    value = as_tensor(a)
    key = index.data.astype(np.int64) if isinstance(index, Tensor) else index
    try:
        data = value.data[key]
    except IndexError as exc:
        raise EveryOShapeError(
            f"Invalid index {index!r} for a tensor with shape {value.shape}."
        ) from exc
    data = np.asarray(data)
    original = value.shape

    def backward_fn(gradient: np.ndarray):
        grad = np.zeros(original, dtype=value.data.dtype)
        np.add.at(grad, key, np.asarray(gradient))
        return (grad,)

    return _make(data, (value,), "slice", backward_fn)


# ----------------------------------------------------------------------
# Activation numerics
# ----------------------------------------------------------------------
def relu(a: Any) -> Tensor:
    """Rectified linear unit."""
    value = as_tensor(a)
    data = _dispatch("relu", value.device, value.data)
    positive = value.data > 0

    def backward_fn(gradient: np.ndarray):
        return (np.asarray(gradient) * positive,)

    return _make(data, (value,), "relu", backward_fn)


def binary_cross_entropy_with_logits(a: Any, b: Any) -> Tensor:
    """Binary cross entropy computed from logits, with an exact gradient.

    The forward pass uses the stable form ``max(x, 0) - x*y + log1p(exp(-|x|))``.
    The backward pass is written out rather than composed from ``relu`` and
    ``abs``: at ``x == 0`` both of those have a zero subgradient, which would
    give ``-y`` instead of the true derivative ``sigmoid(x) - y = 0.5 - y``.
    Logits of exactly zero are common (zero-initialised final layers, zero
    inputs), so that difference matters.
    """
    logits, targets = _binary_inputs(a, b, "binary_cross_entropy_with_logits")
    x = logits.data
    y = targets.data

    magnitude = np.abs(x)
    data = np.maximum(x, 0) - x * y + np.log1p(np.exp(-magnitude))
    probabilities = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-magnitude)),
        np.exp(-magnitude) / (1.0 + np.exp(-magnitude)),
    ).astype(x.dtype, copy=False)

    def backward_fn(gradient: np.ndarray):
        grad = np.asarray(gradient)
        return (
            unbroadcast(grad * (probabilities - y), logits.shape),
            unbroadcast(-grad * x, targets.shape),
        )

    return _make(data, (logits, targets), "binary_cross_entropy_with_logits", backward_fn)


def sigmoid(a: Any) -> Tensor:
    """Logistic sigmoid, computed in a numerically stable way."""
    value = as_tensor(a)
    x = value.data
    data = np.where(
        x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x)))
    )
    data = data.astype(x.dtype, copy=False)

    def backward_fn(gradient: np.ndarray):
        return (np.asarray(gradient) * data * (1.0 - data),)

    return _make(data, (value,), "sigmoid", backward_fn)


def tanh(a: Any) -> Tensor:
    """Hyperbolic tangent."""
    value = as_tensor(a)
    data = np.tanh(value.data)

    def backward_fn(gradient: np.ndarray):
        return (np.asarray(gradient) * (1.0 - data**2),)

    return _make(data, (value,), "tanh", backward_fn)


def softmax(a: Any, axis: int = -1) -> Tensor:
    """Softmax along ``axis``, shifted by the maximum for stability."""
    value = as_tensor(a)
    shifted = value.data - np.max(value.data, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    data = exponentials / np.sum(exponentials, axis=axis, keepdims=True)

    def backward_fn(gradient: np.ndarray):
        grad = np.asarray(gradient)
        weighted = np.sum(grad * data, axis=axis, keepdims=True)
        return (data * (grad - weighted),)

    return _make(data, (value,), "softmax", backward_fn)


def log_softmax(a: Any, axis: int = -1) -> Tensor:
    """Logarithm of :func:`softmax`, computed without forming softmax first."""
    value = as_tensor(a)
    shifted = value.data - np.max(value.data, axis=axis, keepdims=True)
    log_denominator = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    data = shifted - log_denominator
    probabilities = np.exp(data)

    def backward_fn(gradient: np.ndarray):
        grad = np.asarray(gradient)
        return (grad - probabilities * np.sum(grad, axis=axis, keepdims=True),)

    return _make(data, (value,), "log_softmax", backward_fn)
