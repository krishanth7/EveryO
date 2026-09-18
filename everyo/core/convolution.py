"""Convolution and pooling.

These operations live in their own module rather than in
:mod:`everyo.core.operations` because they need a shared set of windowing
helpers that nothing else uses, and because the file would otherwise be twice
the size of anything else in the package.

**Layout.** Tensors are ``NHWC``: ``(batch, height, width, channels)``, and a
convolution kernel is ``(kernel_height, kernel_width, in_channels,
out_channels)``. That is TensorFlow's native layout, which means the results
here can be compared against ``tf.nn.conv2d`` without transposing anything —
and the test suite does exactly that.

**Method.** The forward pass uses *im2col*: every sliding window is laid out as
a row of a matrix, so the convolution becomes a single matrix multiplication.
That is how real frameworks do it, it reuses the fast path NumPy already has,
and the backward pass falls out as two more matrix multiplications plus a
scatter-add back into the padded image.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from everyo.core.autocast import autocast_like, match_reduced_dtype
from everyo.core.autograd import unbroadcast
from everyo.core.operations import _make
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError

__all__ = ["conv2d", "max_pool2d", "avg_pool2d", "compute_output_size"]

_PADDING_MODES = ("valid", "same")


# ----------------------------------------------------------------------
# Shape helpers
# ----------------------------------------------------------------------
def _pair(value: Any, name: str) -> tuple[int, int]:
    """Normalise ``3`` or ``(3, 3)`` into ``(3, 3)``."""
    if isinstance(value, int):
        pair = (value, value)
    else:
        items = tuple(value)
        if len(items) != 2:
            raise EveryOShapeError(f"{name} must be an int or a pair of ints, got {value!r}.")
        pair = (int(items[0]), int(items[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise EveryOShapeError(f"{name} must be positive, got {pair}.")
    return pair


def _resolve_padding(
    padding: Any, size: int, kernel: int, stride: int, name: str
) -> tuple[int, int]:
    """Return ``(before, after)`` padding for one spatial axis.

    ``"same"`` reproduces TensorFlow's rule exactly: the output keeps
    ``ceil(size / stride)`` positions, and when the total padding is odd the
    extra row or column goes at the end.
    """
    if isinstance(padding, str):
        mode = padding.lower()
        if mode not in _PADDING_MODES:
            raise EveryOShapeError(
                f"Unknown padding {padding!r}. Use 'valid', 'same', an int, or a pair of ints."
            )
        if mode == "valid":
            return (0, 0)
        out = -(-size // stride)  # ceil division
        total = max((out - 1) * stride + kernel - size, 0)
        return (total // 2, total - total // 2)

    amount = _pair_nonneg(padding, name)
    return (amount, amount)


def _pair_nonneg(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise EveryOShapeError(
            f"{name} must be 'valid', 'same' or a non-negative int, got {value!r}."
        )
    if value < 0:
        raise EveryOShapeError(f"{name} must be non-negative, got {value}.")
    return value


def compute_output_size(size: int, kernel: int, stride: int, pad: tuple[int, int]) -> int:
    """Return the output length of one spatial axis."""
    return (size + pad[0] + pad[1] - kernel) // stride + 1


def _check_image(value: Tensor, name: str) -> None:
    if value.ndim != 4:
        raise EveryOShapeError(
            f"{name} expects a 4-D tensor shaped (batch, height, width, channels), "
            f"got shape {value.shape}. Add a channel axis with "
            f"x.reshape(*x.shape, 1) for single-channel images."
        )


def _windows(padded: np.ndarray, kernel: tuple[int, int], stride: tuple[int, int]) -> np.ndarray:
    """Return every sliding window as ``(batch, out_h, out_w, kh, kw, channels)``.

    This is a view, not a copy: ``sliding_window_view`` only rewrites strides.
    """
    view = sliding_window_view(padded, kernel, axis=(1, 2))
    # view: (batch, positions_h, positions_w, channels, kh, kw)
    view = view[:, :: stride[0], :: stride[1]]
    return view.transpose(0, 1, 2, 4, 5, 3)


def _scatter_windows(
    gradient: np.ndarray,
    padded_shape: tuple[int, ...],
    kernel: tuple[int, int],
    stride: tuple[int, int],
) -> np.ndarray:
    """Accumulate per-window gradients back into a padded image.

    The loop runs ``kernel_height * kernel_width`` times — a handful of
    iterations — and each one adds a whole strided slab at once, so the batch
    and channel axes stay vectorised. Overlapping windows accumulate, which is
    exactly the derivative of sharing an input pixel between them.
    """
    out = np.zeros(padded_shape, dtype=gradient.dtype)
    out_h, out_w = gradient.shape[1], gradient.shape[2]
    for i in range(kernel[0]):
        rows = slice(i, i + out_h * stride[0], stride[0])
        for j in range(kernel[1]):
            columns = slice(j, j + out_w * stride[1], stride[1])
            out[:, rows, columns, :] += gradient[:, :, :, i, j, :]
    return out


def _pad_image(data: np.ndarray, pad_h: tuple[int, int], pad_w: tuple[int, int], value: float):
    if pad_h == (0, 0) and pad_w == (0, 0):
        return data
    return np.pad(data, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant", constant_values=value)


def _unpad(data: np.ndarray, pad_h: tuple[int, int], pad_w: tuple[int, int]) -> np.ndarray:
    if pad_h == (0, 0) and pad_w == (0, 0):
        return data
    height = data.shape[1] - pad_h[1]
    width = data.shape[2] - pad_w[1]
    return data[:, pad_h[0] : height, pad_w[0] : width, :]


# ----------------------------------------------------------------------
# Convolution
# ----------------------------------------------------------------------
def conv2d(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    stride: Any = 1,
    padding: Any = "valid",
) -> Tensor:
    """2-D convolution over an ``NHWC`` batch of images.

    Args:
        x: Input of shape ``(batch, height, width, in_channels)``.
        weight: Kernel of shape ``(kh, kw, in_channels, out_channels)``.
        bias: Optional ``(out_channels,)`` bias.
        stride: Int or ``(stride_h, stride_w)``.
        padding: ``"valid"``, ``"same"``, or an int applied to both sides.

    Returns:
        A tensor of shape ``(batch, out_height, out_width, out_channels)``.

    Raises:
        EveryOShapeError: If the ranks disagree, the channel counts do not
            match, or the kernel is larger than the padded input.

    Example:
        >>> import everyo as eo
        >>> images = eo.zeros(2, 8, 8, 1)
        >>> kernel = eo.zeros(3, 3, 1, 4)
        >>> eo.conv2d(images, kernel, padding="same").shape
        (2, 8, 8, 4)
    """
    value = as_tensor(x)
    kernel_tensor = as_tensor(weight)
    _check_image(value, "conv2d")

    if kernel_tensor.ndim != 4:
        raise EveryOShapeError(
            f"conv2d expects a 4-D kernel shaped (kh, kw, in_channels, "
            f"out_channels), got shape {kernel_tensor.shape}."
        )

    batch, height, width, in_channels = value.shape
    kernel_h, kernel_w, kernel_in, out_channels = kernel_tensor.shape
    if kernel_in != in_channels:
        raise EveryOShapeError(
            f"conv2d channel mismatch: the input has {in_channels} channel(s) but "
            f"the kernel expects {kernel_in}. Input shape {value.shape}, kernel "
            f"shape {kernel_tensor.shape}."
        )

    stride_pair = _pair(stride, "stride")
    pad_h = _resolve_padding(padding, height, kernel_h, stride_pair[0], "padding")
    pad_w = _resolve_padding(padding, width, kernel_w, stride_pair[1], "padding")

    padded_h = height + pad_h[0] + pad_h[1]
    padded_w = width + pad_w[0] + pad_w[1]
    if padded_h < kernel_h or padded_w < kernel_w:
        raise EveryOShapeError(
            f"conv2d kernel {kernel_h}x{kernel_w} does not fit in the padded input "
            f"{padded_h}x{padded_w}. Use a smaller kernel, or padding='same'."
        )

    out_h = compute_output_size(height, kernel_h, stride_pair[0], pad_h)
    out_w = compute_output_size(width, kernel_w, stride_pair[1], pad_w)

    # Autocast hook: the im2col matmul is the expensive part of a convolution,
    # so it is the part that runs in reduced precision. autocast_like() adds an
    # explicit cast node whose backward pass restores each operand's dtype, so
    # the kernel and bias keep their float32 gradients. Outside an autocast
    # block these calls return their argument unchanged.
    value = autocast_like(value)
    kernel_tensor = autocast_like(kernel_tensor)

    padded = _pad_image(value.data, pad_h, pad_w, 0.0)
    columns = np.ascontiguousarray(_windows(padded, (kernel_h, kernel_w), stride_pair))
    flat_columns = columns.reshape(batch * out_h * out_w, -1)
    flat_kernel = kernel_tensor.data.reshape(-1, out_channels)

    result = flat_columns @ flat_kernel
    bias_tensor = None if bias is None else autocast_like(as_tensor(bias))
    if bias_tensor is not None:
        if bias_tensor.shape != (out_channels,):
            raise EveryOShapeError(
                f"conv2d bias must have shape ({out_channels},) to match the "
                f"kernel's output channels, got {bias_tensor.shape}."
            )
        result = result + bias_tensor.data
    data = result.reshape(batch, out_h, out_w, out_channels)

    inputs = [value, kernel_tensor] + ([bias_tensor] if bias_tensor is not None else [])
    padded_shape = (batch, padded_h, padded_w, in_channels)

    def backward_fn(gradient: np.ndarray):
        grad = match_reduced_dtype(np.asarray(gradient), data).reshape(
            batch * out_h * out_w, out_channels
        )

        grad_input = (grad @ flat_kernel.T).reshape(
            batch, out_h, out_w, kernel_h, kernel_w, in_channels
        )
        grad_padded = _scatter_windows(grad_input, padded_shape, (kernel_h, kernel_w), stride_pair)
        grads: list[np.ndarray | None] = [
            _unpad(grad_padded, pad_h, pad_w),
            (flat_columns.T @ grad).reshape(kernel_h, kernel_w, in_channels, out_channels),
        ]
        if bias_tensor is not None:
            grads.append(
                unbroadcast(
                    # Accumulate the bias reduction in at least fp32 -- summing
                    # thousands of rows is where 16 bits runs out -- then return
                    # it in the incoming gradient's dtype, as the node contract
                    # requires. The cast node above restores float32 for the
                    # parameter itself.
                    grad.sum(axis=0, dtype=np.result_type(grad.dtype, np.float32)),
                    bias_tensor.shape,
                ).astype(grad.dtype, copy=False)
            )
        return tuple(grads)

    return _make(data, inputs, "conv2d", backward_fn)


# ----------------------------------------------------------------------
# Pooling
# ----------------------------------------------------------------------
def _pool_setup(
    x: Any, pool_size: Any, stride: Any, padding: Any, name: str
) -> tuple[Tensor, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], int, int]:
    value = as_tensor(x)
    _check_image(value, name)

    pool_pair = _pair(pool_size, "pool_size")
    stride_pair = pool_pair if stride is None else _pair(stride, "stride")
    _, height, width, _ = value.shape
    pad_h = _resolve_padding(padding, height, pool_pair[0], stride_pair[0], "padding")
    pad_w = _resolve_padding(padding, width, pool_pair[1], stride_pair[1], "padding")

    if height + pad_h[0] + pad_h[1] < pool_pair[0] or width + pad_w[0] + pad_w[1] < pool_pair[1]:
        raise EveryOShapeError(
            f"{name} window {pool_pair[0]}x{pool_pair[1]} is larger than the padded "
            f"input {height}x{width}."
        )

    out_h = compute_output_size(height, pool_pair[0], stride_pair[0], pad_h)
    out_w = compute_output_size(width, pool_pair[1], stride_pair[1], pad_w)
    return value, pool_pair, stride_pair, pad_h, pad_w, out_h, out_w


def max_pool2d(x: Any, pool_size: Any = 2, *, stride: Any = None, padding: Any = "valid") -> Tensor:
    """Max pooling over an ``NHWC`` batch of images.

    Padding cells are filled with ``-inf`` so they can never win a window,
    which is what TensorFlow does and what keeps ``padding="same"`` meaningful.
    The gradient goes to the single winning element of each window; ties go to
    the first, matching TensorFlow and PyTorch.

    Args:
        x: Input of shape ``(batch, height, width, channels)``.
        pool_size: Int or ``(height, width)`` window.
        stride: Defaults to ``pool_size`` (non-overlapping windows).
        padding: ``"valid"``, ``"same"``, or an int.

    Example:
        >>> import everyo as eo
        >>> eo.max_pool2d(eo.zeros(2, 8, 8, 3), 2).shape
        (2, 4, 4, 3)
    """
    value, pool, stride_pair, pad_h, pad_w, out_h, out_w = _pool_setup(
        x, pool_size, stride, padding, "max_pool2d"
    )
    batch, height, width, channels = value.shape

    padded = _pad_image(value.data, pad_h, pad_w, -np.inf)
    windows = _windows(padded, pool, stride_pair)
    flat = windows.reshape(batch, out_h, out_w, pool[0] * pool[1], channels)
    winners = np.argmax(flat, axis=3)
    data = np.take_along_axis(flat, winners[:, :, :, None, :], axis=3).squeeze(axis=3)

    padded_shape = (batch, height + pad_h[0] + pad_h[1], width + pad_w[0] + pad_w[1], channels)

    def backward_fn(gradient: np.ndarray):
        grad = np.asarray(gradient)
        routed = np.zeros((batch, out_h, out_w, pool[0] * pool[1], channels), dtype=grad.dtype)
        np.put_along_axis(routed, winners[:, :, :, None, :], grad[:, :, :, None, :], axis=3)
        routed = routed.reshape(batch, out_h, out_w, pool[0], pool[1], channels)
        grad_padded = _scatter_windows(routed, padded_shape, pool, stride_pair)
        return (_unpad(grad_padded, pad_h, pad_w),)

    return _make(data, (value,), "max_pool2d", backward_fn)


def avg_pool2d(x: Any, pool_size: Any = 2, *, stride: Any = None, padding: Any = "valid") -> Tensor:
    """Average pooling over an ``NHWC`` batch of images.

    With ``padding="same"`` the average is taken over the *valid* cells only,
    never over the zeros introduced by padding — again matching TensorFlow, so
    an edge window is not silently darkened.

    Args:
        x: Input of shape ``(batch, height, width, channels)``.
        pool_size: Int or ``(height, width)`` window.
        stride: Defaults to ``pool_size``.
        padding: ``"valid"``, ``"same"``, or an int.

    Example:
        >>> import everyo as eo
        >>> pooled = eo.avg_pool2d(eo.ones(1, 4, 4, 1), 2)
        >>> pooled.shape
        (1, 2, 2, 1)
        >>> float(pooled.data[0, 0, 0, 0])
        1.0
    """
    value, pool, stride_pair, pad_h, pad_w, out_h, out_w = _pool_setup(
        x, pool_size, stride, padding, "avg_pool2d"
    )
    batch, height, width, channels = value.shape
    padded_shape = (batch, height + pad_h[0] + pad_h[1], width + pad_w[0] + pad_w[1], channels)

    padded = _pad_image(value.data, pad_h, pad_w, 0.0)
    totals = _windows(padded, pool, stride_pair).sum(axis=(3, 4))

    # How many real (unpadded) cells each window covers.
    mask = _pad_image(np.ones((batch, height, width, 1), dtype=padded.dtype), pad_h, pad_w, 0.0)
    counts = _windows(mask, pool, stride_pair).sum(axis=(3, 4))
    data = totals / counts

    def backward_fn(gradient: np.ndarray):
        share = np.asarray(gradient) / counts
        spread = np.broadcast_to(
            share[:, :, :, None, None, :], (batch, out_h, out_w, pool[0], pool[1], channels)
        )
        grad_padded = _scatter_windows(spread, padded_shape, pool, stride_pair)
        return (_unpad(grad_padded, pad_h, pad_w),)

    return _make(data, (value,), "avg_pool2d", backward_fn)
