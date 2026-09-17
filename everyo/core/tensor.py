"""The EveryO :class:`Tensor`.

A tensor owns a NumPy array, a device, and (optionally) a node in the
autograd graph.  Arithmetic is implemented in :mod:`everyo.core.operations`;
the dunder methods here forward to it so that ``a + b`` and ``eo.add(a, b)``
behave identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np

from everyo.core import autograd
from everyo.core.device import Device, DeviceLike, resolve_device
from everyo.core.dtype import DEFAULT_FLOAT_DTYPE, dtype_name, is_floating, resolve_dtype
from everyo.exceptions import EveryODTypeError, EveryOGradientError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from everyo.core.autograd import Node

__all__ = ["Tensor", "tensor", "as_tensor", "is_tensor"]

_MAX_REPR_ELEMENTS = 1000


class Tensor:
    """An n-dimensional array with optional gradient tracking.

    Args:
        data: Array-like data, another tensor, or a scalar.
        dtype: Optional dtype; defaults to the dtype of ``data`` when it is
            already floating point, otherwise ``float32``.
        device: ``"cpu"``, ``"cuda"``, ``"auto"`` or a :class:`Device`.
        requires_grad: Track gradients for this tensor.
        name: Optional label used in :func:`repr` and debugging.

    Example:
        >>> import everyo as eo
        >>> x = eo.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> x.shape
        (2, 2)
    """

    __slots__ = (
        "_data",
        "_device",
        "_requires_grad",
        "grad",
        "grad_node",
        "retains_grad",
        "name",
    )

    def __init__(
        self,
        data: Any,
        *,
        dtype: Any | None = None,
        device: DeviceLike = "cpu",
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        self._data = _to_array(data, dtype)
        self._device = resolve_device(device)
        self._requires_grad = False
        self.grad: np.ndarray | None = None
        self.grad_node: Node | None = None
        self.retains_grad = False
        self.name = name
        if requires_grad:
            self.requires_grad = True  # goes through the validating setter

    # ------------------------------------------------------------------
    # Construction helpers used by the operations module
    # ------------------------------------------------------------------
    @classmethod
    def _wrap(
        cls,
        data: np.ndarray,
        *,
        device: Device,
        node: Node | None = None,
    ) -> Tensor:
        """Create a tensor for an operation result (internal API)."""
        out = cls.__new__(cls)
        out._data = data
        out._device = device
        out._requires_grad = node is not None
        out.grad = None
        out.grad_node = node
        out.retains_grad = False
        out.name = None
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """The underlying NumPy buffer."""
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = _to_array(value, self._data.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of dimension sizes."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(self._data.size)

    @property
    def dtype(self) -> str:
        """Canonical dtype name, e.g. ``"float32"``."""
        return dtype_name(self._data.dtype)

    @property
    def device(self) -> Device:
        """The device operations on this tensor execute on."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Whether this tensor participates in automatic differentiation."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        value = bool(value)
        if value and not is_floating(self._data.dtype):
            raise EveryOGradientError(
                f"Only floating point tensors can require gradients, but this "
                f"tensor has dtype '{self.dtype}'. Cast it first, for example "
                f"with .astype('float32')."
            )
        self._requires_grad = value

    @property
    def is_leaf(self) -> bool:
        """``True`` when this tensor was not produced by a tracked operation."""
        return self.grad_node is None

    @property
    def T(self) -> Tensor:
        """Transpose of a tensor (reverses the axis order)."""
        from everyo.core import operations as ops

        return ops.transpose(self)

    # ------------------------------------------------------------------
    # Gradient plumbing
    # ------------------------------------------------------------------
    def accumulate_grad(self, gradient: np.ndarray) -> None:
        """Add ``gradient`` into :attr:`grad` (internal autograd API)."""
        gradient = np.asarray(gradient, dtype=self._data.dtype)
        if gradient.shape != self.shape:  # pragma: no cover - internal bug guard
            raise EveryOGradientError(
                f"Gradient shape {gradient.shape} does not match tensor shape {self.shape}."
            )
        self.grad = gradient.copy() if self.grad is None else self.grad + gradient

    def backward(self, gradient: Any | None = None) -> None:
        """Run reverse-mode differentiation from this tensor.

        Args:
            gradient: Seed gradient; optional for scalar tensors.

        Example:
            >>> import everyo as eo
            >>> x = eo.tensor([2.0], requires_grad=True)
            >>> y = x * x
            >>> y.backward()
            >>> x.grad
            array([4.], dtype=float32)
        """
        if gradient is not None and isinstance(gradient, Tensor):
            gradient = gradient.data
        autograd.backward(self, None if gradient is None else np.asarray(gradient))

    def zero_grad(self) -> None:
        """Reset the accumulated gradient to ``None``."""
        self.grad = None

    def retain_grad(self) -> Tensor:
        """Keep :attr:`grad` for a non-leaf tensor (useful when debugging)."""
        self.retains_grad = True
        return self

    def detach(self) -> Tensor:
        """Return a tensor sharing this buffer but detached from the graph."""
        return Tensor._wrap(self._data, device=self._device)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def numpy(self) -> np.ndarray:
        """Return a copy of the data as a NumPy array."""
        return self._data.copy()

    def tolist(self) -> Any:
        """Return the data as nested Python lists."""
        return self._data.tolist()

    def item(self) -> float:
        """Return the single element of a size-1 tensor as a Python scalar."""
        if self.size != 1:
            raise ValueError(
                f"item() expects a tensor with exactly one element, but this "
                f"tensor has {self.size} elements (shape {self.shape})."
            )
        return self._data.reshape(-1)[0].item()

    def astype(self, dtype: Any) -> Tensor:
        """Return this tensor cast to ``dtype``.

        A cast between floating point dtypes keeps the autograd graph intact;
        a cast to an integer or boolean dtype detaches, because such a tensor
        cannot carry gradients.
        """
        from everyo.core import operations as ops

        return ops.to(self, dtype=dtype)

    def to(self, device: DeviceLike = None, *, dtype: Any | None = None) -> Tensor:
        """Return a tensor placed on ``device`` and/or cast to ``dtype``.

        The move is differentiable: gradients flow back to the source tensor,
        so ``(x * 2).to("cpu").sum().backward()`` still fills in ``x.grad``.
        """
        from everyo.core import operations as ops

        return ops.to(self, device=device, dtype=dtype)

    def cpu(self) -> Tensor:
        """Return this tensor placed on the CPU device."""
        return self.to("cpu")

    def cuda(self, *, strict: bool = False) -> Tensor:
        """Return this tensor placed on CUDA, falling back to CPU if needed."""
        return self.to(resolve_device("cuda", strict=strict))

    def clone(self) -> Tensor:
        """Return a deep copy that is detached from the autograd graph."""
        return Tensor(self._data.copy(), device=self._device, requires_grad=self._requires_grad)

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        """NumPy interoperability hook (``np.asarray(tensor)``)."""
        return self._data if dtype is None else self._data.astype(dtype)

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------
    def reshape(self, *shape: int | Sequence[int]) -> Tensor:
        """Return a tensor with the same data and a new shape."""
        from everyo.core import operations as ops

        return ops.reshape(self, _normalise_shape(shape))

    def transpose(self, *axes: int) -> Tensor:
        """Permute the axes of the tensor."""
        from everyo.core import operations as ops

        return ops.transpose(self, axes if axes else None)

    def flatten(self, start_dim: int = 0) -> Tensor:
        """Collapse dimensions from ``start_dim`` onwards into a single axis."""
        from everyo.core import operations as ops

        return ops.flatten(self, start_dim=start_dim)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Sum elements over ``axis``."""
        from everyo.core import operations as ops

        return ops.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Arithmetic mean over ``axis``."""
        from everyo.core import operations as ops

        return ops.mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Maximum over ``axis``."""
        from everyo.core import operations as ops

        return ops.max(self, axis=axis, keepdims=keepdims)

    def min(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Minimum over ``axis``."""
        from everyo.core import operations as ops

        return ops.min(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis: int | None = None) -> np.ndarray:
        """Indices of the maximum values (not differentiable)."""
        return np.argmax(self._data, axis=axis)

    def argmin(self, axis: int | None = None) -> np.ndarray:
        """Indices of the minimum values (not differentiable)."""
        return np.argmin(self._data, axis=axis)

    # ------------------------------------------------------------------
    # Element-wise maths
    # ------------------------------------------------------------------
    def exp(self) -> Tensor:
        """Element-wise exponential."""
        from everyo.core import operations as ops

        return ops.exp(self)

    def log(self) -> Tensor:
        """Element-wise natural logarithm."""
        from everyo.core import operations as ops

        return ops.log(self)

    def sqrt(self) -> Tensor:
        """Element-wise square root."""
        from everyo.core import operations as ops

        return ops.sqrt(self)

    def abs(self) -> Tensor:
        """Element-wise absolute value."""
        from everyo.core import operations as ops

        return ops.abs(self)

    def clip(self, low: float, high: float) -> Tensor:
        """Clamp values into ``[low, high]``."""
        from everyo.core import operations as ops

        return ops.clip(self, low, high)

    def matmul(self, other: Any) -> Tensor:
        """Matrix multiplication."""
        from everyo.core import operations as ops

        return ops.matmul(self, other)

    def dot(self, other: Any) -> Tensor:
        """Dot product (see :func:`everyo.core.operations.dot`)."""
        from everyo.core import operations as ops

        return ops.dot(self, other)

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------
    def __add__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.add(self, other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.subtract(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.subtract(other, self)

    def __mul__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.multiply(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.divide(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.divide(other, self)

    def __pow__(self, exponent: float) -> Tensor:
        from everyo.core import operations as ops

        return ops.power(self, exponent)

    def __neg__(self) -> Tensor:
        from everyo.core import operations as ops

        return ops.negative(self)

    def __matmul__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.matmul(self, other)

    def __rmatmul__(self, other: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.matmul(other, self)

    def __getitem__(self, index: Any) -> Tensor:
        from everyo.core import operations as ops

        return ops.slice_(self, index)

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-dimensional tensor is undefined.")
        return int(self.shape[0])

    def __iter__(self) -> Iterator[Tensor]:
        for index in range(len(self)):
            yield self[index]

    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError(
                "The truth value of a tensor with more than one element is "
                "ambiguous. Use .any() on the NumPy data, or compare .item()."
            )
        return bool(self._data.reshape(-1)[0])

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if self.size > _MAX_REPR_ELEMENTS:
            body = f"<{self.size} elements>"
        else:
            body = np.array2string(self._data, precision=4, suppress_small=True)
            body = body.replace("\n", "\n       ")
        parts = [f"shape={self.shape}", f"dtype={self.dtype}"]
        if self._device.type != "cpu":
            parts.append(f"device={self._device}")
        if self._requires_grad:
            parts.append("requires_grad=True")
        if self.name:
            parts.append(f"name={self.name!r}")
        return f"Tensor({body}, {', '.join(parts)})"

    def __str__(self) -> str:
        return self.__repr__()


def _normalise_shape(shape: tuple[Any, ...]) -> tuple[int, ...]:
    """Accept both ``reshape(2, 3)`` and ``reshape((2, 3))``."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(int(dim) for dim in shape[0])
    return tuple(int(dim) for dim in shape)


def _contiguous(array: np.ndarray, dtype: Any) -> np.ndarray:
    """Return ``array`` as ``dtype``, contiguous, without promoting 0-d to 1-d.

    ``np.ascontiguousarray`` turns a scalar array into shape ``(1,)``, which
    would silently change the rank of scalar tensors, so it is only applied to
    arrays that actually have dimensions.
    """
    converted = np.asarray(array, dtype=dtype)
    if converted.ndim and not converted.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(converted)
    return converted


def _to_array(data: Any, dtype: Any | None) -> np.ndarray:
    """Convert ``data`` into a NumPy array with a dtype EveryO supports."""
    if isinstance(data, Tensor):
        array = data.data
    else:
        try:
            array = np.asarray(data)
        except (ValueError, TypeError) as exc:
            raise EveryODTypeError(
                "Could not build a numeric tensor from the given data "
                f"({exc}). Ragged or non-numeric input is not supported; make "
                "sure every row has the same length and contains numbers."
            ) from exc

    if array.dtype == np.object_:
        raise EveryODTypeError(
            "Could not build a numeric tensor from the given data. Ragged or "
            "non-numeric input is not supported; make sure every row has the "
            "same length and contains numbers."
        )

    if dtype is not None:
        return _contiguous(array, resolve_dtype(dtype))
    if array.dtype == np.bool_:
        return _contiguous(array, np.bool_)
    if np.issubdtype(array.dtype, np.floating):
        # A NumPy array keeps its own precision (so float64 work stays float64),
        # while Python scalars and lists adopt the float32 default.
        keep_precision = isinstance(data, (np.ndarray, np.generic, Tensor))
        target = resolve_dtype(array.dtype) if keep_precision else DEFAULT_FLOAT_DTYPE
        return _contiguous(array, target)
    if np.issubdtype(array.dtype, np.integer):
        # Integers become float32 by default so that tensors are immediately
        # usable in gradient-based code; pass dtype="int64" to keep integers.
        return _contiguous(array, DEFAULT_FLOAT_DTYPE)
    raise EveryODTypeError(
        f"Unsupported input dtype '{array.dtype}'. Supported dtypes are "
        "float32, float64, int32, int64 and bool."
    )


def tensor(
    data: Any,
    *,
    dtype: Any | None = None,
    device: DeviceLike = "cpu",
    requires_grad: bool = False,
    name: str | None = None,
) -> Tensor:
    """Create a :class:`Tensor` (the primary public constructor).

    Example:
        >>> import everyo as eo
        >>> eo.tensor([1.0, 2.0, 3.0]).shape
        (3,)
    """
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, name=name)


def as_tensor(data: Any, *, device: DeviceLike = None, dtype: Any | None = None) -> Tensor:
    """Return ``data`` unchanged when it is already a tensor, else convert it."""
    if isinstance(data, Tensor) and dtype is None:
        if device is None or resolve_device(device) == data.device:
            return data
        return data.to(device)
    return Tensor(data, dtype=dtype, device=device or "cpu")


def is_tensor(obj: Any) -> bool:
    """Return ``True`` when ``obj`` is an EveryO tensor."""
    return isinstance(obj, Tensor)
