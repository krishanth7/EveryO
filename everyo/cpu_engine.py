"""Optional, CPU-only execution primitives for measured NumPy workloads.

The engine does not replace EveryO's autograd operations. These functions
operate on arrays; callers explicitly opt in and retain ownership of their
outputs. No thread pool, compiler, or binary is started on import.
"""

from __future__ import annotations

import ctypes
import os
import platform
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def _float_array(value: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("CPU kernels require float32 or float64 arrays")
    return array


class ArrayPool:
    """Bounded, thread-safe scratch-array reuse by shape/dtype.

    Leased arrays are never shared. ``release`` rejects non-owned arrays,
    duplicate releases and arrays that no longer own their storage. Reused
    arrays are uninitialized; pass ``zero=True`` when old contents matter.
    """

    def __init__(self, *, max_cached_bytes: int = 64 * 1024 * 1024) -> None:
        if not isinstance(max_cached_bytes, int) or max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be a nonnegative integer")
        self.max_cached_bytes = max_cached_bytes
        self._free: dict[tuple[tuple[int, ...], str], list[np.ndarray]] = {}
        self._leased: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self.cached_bytes = 0
        self.reuses = 0

    def acquire(
        self, shape: Sequence[int], dtype: object = np.float64, *, zero: bool = False
    ) -> np.ndarray:
        dimensions = tuple(int(x) for x in shape)
        if any(x < 0 for x in dimensions):
            raise ValueError("shape dimensions must not be negative")
        item_dtype = np.dtype(dtype)
        if item_dtype.hasobject:
            raise TypeError("object arrays cannot be pooled")
        key = (dimensions, item_dtype.str)
        with self._lock:
            available = self._free.get(key)
            if available:
                array = available.pop()
                self.cached_bytes -= array.nbytes
                self.reuses += 1
            else:
                array = np.empty(dimensions, dtype=item_dtype)
            self._leased[id(array)] = array
        if zero:
            array.fill(0)
        return array

    def release(self, array: np.ndarray) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError("release expects a NumPy array")
        with self._lock:
            original = self._leased.pop(id(array), None)
            if original is not array:
                raise ValueError("array is not leased from this pool")
            if array.base is not None or not array.flags.owndata:
                return
            if self.cached_bytes + array.nbytes <= self.max_cached_bytes:
                key = (array.shape, array.dtype.str)
                self._free.setdefault(key, []).append(array)
                self.cached_bytes += array.nbytes

    def clear(self) -> None:
        """Discard only cached buffers; outstanding leases remain valid."""
        with self._lock:
            self._free.clear()
            self.cached_bytes = 0


def zero_copy_view(value: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Return a true storage-sharing reshape; fail if NumPy would copy."""
    array = np.asarray(value)
    requested = tuple(int(n) for n in shape)
    if any(n < -1 for n in requested):
        raise ValueError("invalid reshape dimension")
    reshaped = np.reshape(array, requested)
    if array.size and not np.shares_memory(array, reshaped):
        raise ValueError("requested view requires copying array storage")
    return reshaped


def vectorized_axpy(
    x: np.ndarray, y: np.ndarray, alpha: float, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Calculate ``alpha*x + y`` with strict matching shapes and dtype."""
    left, right = _float_array(x), _float_array(y)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("x and y must have identical shape and dtype")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if out is None:
        out = np.empty_like(left)
    if out.shape != left.shape or out.dtype != left.dtype or not out.flags.writeable:
        raise ValueError("out must be writeable with the matching shape and dtype")
    # An explicit temporary preserves correctness for overlapping inputs.
    result = left * alpha + right
    np.copyto(out, result)
    return out


def parallel_map_chunks(
    array: np.ndarray,
    operation: Callable[[np.ndarray], np.ndarray],
    *,
    workers: int = 2,
    min_chunk_rows: int = 1024,
) -> np.ndarray:
    """Map a pure, shape-preserving NumPy operation across the first axis.

    Each worker receives its own input copy to prevent cross-thread mutation.
    The return is ordered. For small arrays the serial path avoids overhead.
    The callable must not depend on global mutable state.
    """
    data = np.asarray(array)
    if data.ndim == 0:
        raise ValueError("parallel input requires at least one dimension")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not isinstance(min_chunk_rows, int) or min_chunk_rows < 1:
        raise ValueError("min_chunk_rows must be a positive integer")
    if data.shape[0] == 0:
        return data.copy()
    count = min(workers, max(1, data.shape[0] // min_chunk_rows))
    chunks = [chunk.copy() for chunk in np.array_split(data, count, axis=0)]

    def run(chunk: np.ndarray) -> np.ndarray:
        result = np.asarray(operation(chunk))
        if result.shape != chunk.shape:
            raise ValueError("chunk operation must preserve its input shape")
        return result.copy()

    if count == 1:
        outputs = [run(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            outputs = list(executor.map(run, chunks))
    if any(result.dtype != outputs[0].dtype for result in outputs):
        raise TypeError("all chunks must return the same dtype")
    return np.concatenate(outputs, axis=0)


def fused_affine_relu(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Fused dense affine + ReLU inference operation (no autograd graph).

    This uses NumPy's optimized matrix multiplication and applies the
    activation in-place, reducing intermediate allocation. ``out`` must not
    alias any input. Use EveryO layers when gradients are required.
    """
    a, w, b = (_float_array(v) for v in (x, weight, bias))
    if (
        a.ndim != 2
        or w.ndim != 2
        or b.ndim != 1
        or a.shape[1] != w.shape[0]
        or w.shape[1] != b.size
    ):
        raise ValueError("expected x=(batch,in), weight=(in,out), bias=(out,)")
    if a.dtype != w.dtype or w.dtype != b.dtype:
        raise TypeError("all inputs must use the same floating dtype")
    shape = (a.shape[0], w.shape[1])
    if out is None:
        out = np.empty(shape, dtype=a.dtype)
    if out.shape != shape or out.dtype != a.dtype or not out.flags.writeable:
        raise ValueError("invalid output buffer")
    if any(np.may_share_memory(out, source) for source in (a, w, b)):
        raise ValueError("output must not alias input storage")
    np.matmul(a, w, out=out)
    np.add(out, b, out=out)
    np.maximum(out, 0, out=out)
    return out


class OptionalCppAxpy:
    """Explicitly load a separately built C++ shared library; never compile on import.

    The binary must export ``everyo_axpy_f64(size_t,double,const double*,
    const double*,double*)``. Only contiguous float64 arrays are accepted.
    """

    def __init__(self, library: str | os.PathLike[str]) -> None:
        path = Path(library)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._library = ctypes.CDLL(str(path.resolve()))
        fn = self._library.everyo_axpy_f64
        fn.argtypes = [
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        fn.restype = None
        self._function = fn

    def axpy(self, x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        a, b = np.asarray(x), np.asarray(y)
        if a.dtype != np.float64 or b.dtype != np.float64 or a.shape != b.shape:
            raise TypeError("C++ axpy requires equal-shape float64 inputs")
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        left, right = np.ascontiguousarray(a), np.ascontiguousarray(b)
        result = np.empty_like(left)
        ptr = ctypes.POINTER(ctypes.c_double)
        self._function(
            left.size,
            float(alpha),
            left.ctypes.data_as(ptr),
            right.ctypes.data_as(ptr),
            result.ctypes.data_as(ptr),
        )
        return result.reshape(a.shape)


@dataclass(frozen=True)
class CpuProfile:
    operation: str
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    runs: int
    output_bytes: int
    checksum: float
    python: str
    numpy: str
    machine: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_cpu(
    operation: Callable[[], np.ndarray], *, name: str, runs: int = 7, warmups: int = 2
) -> CpuProfile:
    """Profile deterministic CPU output with warmups and correctness checks."""
    if not name or not isinstance(name, str):
        raise ValueError("name must be nonempty")
    if not isinstance(runs, int) or runs < 1 or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("runs must be positive and warmups nonnegative integers")
    durations: list[float] = []
    reference: np.ndarray | None = None
    for index in range(runs + warmups):
        start = time.perf_counter_ns()
        output = np.asarray(operation())
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if output.dtype.kind not in "biuf" or not np.all(np.isfinite(output)):
            raise ValueError("operation must return a finite numeric array")
        if reference is None:
            reference = output.copy()
        elif not np.array_equal(output, reference):
            raise ValueError("operation output changed during profiling")
        if index >= warmups:
            durations.append(elapsed)
    assert reference is not None
    return CpuProfile(
        name,
        statistics.median(durations),
        min(durations),
        max(durations),
        runs,
        reference.nbytes,
        float(np.asarray(reference, dtype=np.float64).sum()),
        platform.python_version(),
        np.__version__,
        platform.machine(),
    )


def check_cpu_regression(
    baseline: CpuProfile, candidate: CpuProfile, *, threshold: float = 1.25
) -> float:
    """Compare like-for-like profiles; return slowdown ratio or raise."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    for field in ("operation", "python", "numpy", "machine", "output_bytes"):
        if getattr(baseline, field) != getattr(candidate, field):
            raise ValueError(f"incompatible profiles: {field} differs")
    if not np.isclose(baseline.checksum, candidate.checksum, rtol=1e-12, atol=1e-12):
        raise ValueError("profile result checksum differs")
    if baseline.median_ms <= 0:
        raise ValueError("baseline timing must be positive")
    slowdown = candidate.median_ms / baseline.median_ms
    if slowdown > threshold:
        raise RuntimeError(f"CPU regression: {slowdown:.3f}x baseline")
    return slowdown
