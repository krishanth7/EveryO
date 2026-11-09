"""Measured CPU inference performance, with environment and memory metadata."""

from __future__ import annotations

import platform
import time
import tracemalloc
from typing import Any, Callable

import numpy as np


def benchmark_cpu(
    fn: Callable[[], Any], *, repeats: int = 10, warmup: int = 2, samples_per_call: int = 1
) -> dict[str, Any]:
    if repeats <= 0 or warmup < 0 or samples_per_call <= 0:
        raise ValueError("repeats and samples_per_call must be positive; warmup nonnegative")
    for _ in range(warmup):
        fn()
    tracemalloc.start()
    try:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    values = np.asarray(samples)
    return {
        "mean_ms": float(values.mean() * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "samples_per_second": float(samples_per_call / values.mean())
        if values.mean() > 0
        else float("inf"),
        "peak_python_bytes": int(peak),
        "repeats": repeats,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
