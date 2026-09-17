"""Benchmark helpers shared by the CLI and the scripts in ``benchmarks/``.

Every number these functions return comes from an actual measured run on the
machine executing them.  Nothing is cached, estimated or hardcoded, and the
result rows record the backend, problem size, device and iteration count so the
measurement can be reproduced.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Callable, Sequence

import numpy as np

from everyo._logging import get_logger

__all__ = [
    "time_callable",
    "environment",
    "available_matmul_backends",
    "run_matmul_benchmark",
    "run_relu_benchmark",
]

_LOGGER = get_logger(__name__)


def environment() -> dict[str, Any]:
    """Describe the machine the benchmark ran on."""
    from everyo.backends import tensorflow_backend
    from everyo.cuda.availability import is_available as cuda_available
    from everyo.version import __version__

    return {
        "everyo_version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
        "tensorflow": (
            tensorflow_backend.get_tensorflow().__version__
            if tensorflow_backend.is_available()
            else None
        ),
        "tensorflow_gpu": tensorflow_backend.gpu_available(),
        "cuda_extension": cuda_available(),
    }


def time_callable(
    fn: Callable[[], Any],
    *,
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, float]:
    """Time ``fn`` and return timing statistics in seconds.

    Warm-up runs are executed and discarded first: the first call pays for lazy
    imports, memory allocation and (on a GPU) kernel compilation, which would
    otherwise dominate the measurement.

    Args:
        fn: Zero-argument callable to measure.
        repeats: Number of timed runs.
        warmup: Number of discarded warm-up runs.

    Returns:
        ``{"seconds", "best", "worst", "repeats"}`` where ``seconds`` is the mean.
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}.")
    for _ in range(max(warmup, 0)):
        fn()

    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)

    return {
        "seconds": float(np.mean(timings)),
        "best": float(np.min(timings)),
        "worst": float(np.max(timings)),
        "repeats": int(repeats),
    }


def available_matmul_backends() -> dict[str, Callable[[np.ndarray, np.ndarray], Any]]:
    """Return the matmul implementations that can run on this machine."""
    from everyo.backends import numpy_backend, tensorflow_backend
    from everyo.cuda.availability import is_available as cuda_available

    backends: dict[str, Callable[[np.ndarray, np.ndarray], Any]] = {
        "numpy-cpu": numpy_backend.matmul,
        "everyo-cpu": _everyo_matmul,
    }
    if tensorflow_backend.is_available():
        backends["tensorflow-gpu" if tensorflow_backend.gpu_available() else "tensorflow-cpu"] = (
            _tensorflow_matmul()
        )
    if cuda_available():
        from everyo.cuda import interface as cuda_interface

        backends["everyo-cuda"] = lambda a, b: cuda_interface.matmul(a, b, strict=True)
    return backends


def _everyo_matmul(a: np.ndarray, b: np.ndarray) -> Any:
    from everyo.core import operations as ops
    from everyo.core.autograd import no_grad
    from everyo.core.tensor import Tensor

    with no_grad():
        return ops.matmul(Tensor(a), Tensor(b))


def _tensorflow_matmul() -> Callable[[np.ndarray, np.ndarray], Any]:
    from everyo.backends import tensorflow_backend

    tf = tensorflow_backend.require()

    def run(a: np.ndarray, b: np.ndarray) -> Any:
        # .numpy() forces the (possibly asynchronous) device computation to
        # complete, so the timing includes the actual work.
        return tf.matmul(tf.convert_to_tensor(a), tf.convert_to_tensor(b)).numpy()

    return run


def run_matmul_benchmark(
    *,
    sizes: Sequence[int] = (128, 256, 512),
    repeats: int = 3,
    warmup: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Benchmark square matrix multiplication on every available backend.

    Returns:
        One row per (backend, size) pair, including measured seconds and the
        derived GFLOP/s for the standard ``2 n^3`` operation count.
    """
    rng = np.random.default_rng(seed)
    backends = available_matmul_backends()
    results: list[dict[str, Any]] = []

    for size in sizes:
        if size <= 0:
            raise ValueError(f"Matrix sizes must be positive, got {size}.")
        left = rng.normal(size=(size, size)).astype(np.float32)
        right = rng.normal(size=(size, size)).astype(np.float32)
        operations = 2.0 * size**3

        for name, fn in backends.items():
            try:
                timing = time_callable(
                    lambda fn=fn, left=left, right=right: fn(left, right),
                    repeats=repeats,
                    warmup=warmup,
                )
            except Exception as exc:  # pragma: no cover - backend specific
                _LOGGER.warning("Backend '%s' failed at size %d: %s", name, size, exc)
                continue
            results.append(
                {
                    "benchmark": "matmul",
                    "backend": name,
                    "size": int(size),
                    "gflops": operations / timing["seconds"] / 1e9,
                    **timing,
                }
            )
    return results


def run_relu_benchmark(
    *,
    sizes: Sequence[int] = (100_000, 1_000_000, 10_000_000),
    repeats: int = 3,
    warmup: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Benchmark the ReLU activation on every available backend."""
    from everyo.backends import numpy_backend, tensorflow_backend
    from everyo.core import operations as ops
    from everyo.core.autograd import no_grad
    from everyo.core.tensor import Tensor
    from everyo.cuda.availability import is_available as cuda_available

    rng = np.random.default_rng(seed)

    def everyo_relu(values: np.ndarray) -> Any:
        with no_grad():
            return ops.relu(Tensor(values))

    backends: dict[str, Callable[[np.ndarray], Any]] = {
        "numpy-cpu": numpy_backend.relu,
        "everyo-cpu": everyo_relu,
    }
    if tensorflow_backend.is_available():
        tf = tensorflow_backend.require()
        label = "tensorflow-gpu" if tensorflow_backend.gpu_available() else "tensorflow-cpu"
        backends[label] = lambda v: tf.nn.relu(tf.convert_to_tensor(v)).numpy()
    if cuda_available():
        from everyo.cuda import interface as cuda_interface

        backends["everyo-cuda"] = lambda v: cuda_interface.relu(v, strict=True)

    results: list[dict[str, Any]] = []
    for size in sizes:
        values = rng.normal(size=int(size)).astype(np.float32)
        for name, fn in backends.items():
            try:
                timing = time_callable(
                    lambda fn=fn, values=values: fn(values),
                    repeats=repeats,
                    warmup=warmup,
                )
            except Exception as exc:  # pragma: no cover - backend specific
                _LOGGER.warning("Backend '%s' failed at size %d: %s", name, size, exc)
                continue
            results.append(
                {
                    "benchmark": "relu",
                    "backend": name,
                    "size": int(size),
                    "elements_per_second": float(size) / timing["seconds"],
                    **timing,
                }
            )
    return results
