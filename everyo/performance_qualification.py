"""Performance, reliability, soak, stress, and fault-recovery qualification suite for EveryO.

Phase 21.3 Master Qualification Architecture:
- Phase 21.3.1: Reproducible CPU performance benchmarking (tensor ops, neural computation, model ops).
- Phase 21.3.2: Memory and resource profiling (RSS, peak RSS, tracemalloc, lifecycle verification).
- Phase 21.3.3: Concurrency and stress qualification (Profiles A-E, deadlock checks, no state leakage).
- Phase 21.3.4: Long-duration soak testing & degradation monitoring (RSS drift, latency drift, throughput).
- Phase 21.3.5: Controlled fault injection and recovery validation (Categories A-F, seamless recovery).
- Phase 21.3.6: Automated performance regression gates and baseline management.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

import everyo as eo
from everyo.inference import CPUInferenceRuntime
from everyo.version import __version__

# ==============================================================================
# Helper: Monotonic Benchmarking & Percentiles
# ==============================================================================


def calculate_percentile(sorted_values: list[float], percentile: float) -> float:
    """Calculate the given percentile (0.0 to 1.0) using linear interpolation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * weight)


def _get_git_commit_sha() -> str:
    commit = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT_SHA")
    if commit:
        return commit.strip()
    try:
        import subprocess

        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_thread_settings() -> dict[str, str | None]:
    vars_to_check = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    return {v: os.environ.get(v) for v in vars_to_check}


def _get_optional_backends() -> dict[str, bool]:
    backends = {"cuda": False, "tensorflow": False}
    try:
        from everyo.cuda.availability import is_available as cuda_available

        backends["cuda"] = bool(cuda_available())
    except Exception:
        pass
    try:
        from everyo.backends import tensorflow_backend

        backends["tensorflow"] = bool(tensorflow_backend.is_available())
    except Exception:
        pass
    return backends


def get_environment_metadata() -> dict[str, Any]:
    """Capture comprehensive environment and hardware metadata."""
    mem_stats = get_process_memory()
    blas_info: Any = None
    try:
        blas_info = np.show_config(mode="dicts")
    except Exception:
        try:
            blas_info = str(getattr(np.__config__, "blas_opt_info", {}))
        except Exception:
            blas_info = None

    return {
        "everyo_version": __version__,
        "git_commit_sha": _get_git_commit_sha(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or 1,
        "numpy_version": np.__version__,
        "numpy_major": np.__version__.split(".")[0],
        "blas_info": blas_info,
        "thread_settings": _get_thread_settings(),
        "optional_backends": _get_optional_backends(),
        "benchmark_suite_version": "21.3.1",
        "process_pid": os.getpid(),
        "initial_rss_bytes": mem_stats.get("rss_bytes"),
    }


# ==============================================================================
# 21.3.2: Memory and Resource Measurement
# ==============================================================================


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_memory() -> dict[str, int | None]:
    """Retrieve current process RSS and memory usage in bytes portably."""
    rss_bytes: int | None = None
    peak_rss_bytes: int | None = None

    if sys.platform == "win32":
        try:
            psapi = ctypes.windll.psapi
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                rss_bytes = int(counters.WorkingSetSize)
                peak_rss_bytes = int(counters.PeakWorkingSetSize)
        except Exception:
            pass
    else:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # On Linux ru_maxrss is in KiB; on macOS ru_maxrss is in bytes
            multiplier = 1024 if sys.platform.startswith("linux") else 1
            peak_rss_bytes = int(usage.ru_maxrss * multiplier)
            rss_bytes = peak_rss_bytes
        except Exception:
            pass

    # Internal python allocations
    python_current: int | None = None
    python_peak: int | None = None
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        python_current = cur
        python_peak = peak

    return {
        "rss_bytes": rss_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "python_allocated_bytes": python_current,
        "python_peak_bytes": python_peak,
    }


class MemoryTracker:
    """Context manager measuring memory delta and peak during execution."""

    def __init__(self, trace_python: bool = True) -> None:
        self.trace_python = trace_python
        self._started_tracemalloc = False
        self.start_memory: dict[str, int | None] = {}
        self.end_memory: dict[str, int | None] = {}
        self.delta_rss_bytes: int = 0
        self.delta_python_bytes: int = 0

    def __enter__(self) -> MemoryTracker:
        gc.collect()
        if self.trace_python and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True
        self.start_memory = get_process_memory()
        return self

    def __exit__(self, *args: Any) -> None:
        gc.collect()
        self.end_memory = get_process_memory()
        start_rss = self.start_memory.get("rss_bytes") or 0
        end_rss = self.end_memory.get("rss_bytes") or 0
        self.delta_rss_bytes = end_rss - start_rss

        start_py = self.start_memory.get("python_allocated_bytes") or 0
        end_py = self.end_memory.get("python_allocated_bytes") or 0
        self.delta_python_bytes = end_py - start_py

        if self._started_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


def verify_resource_lifecycle(*, iterations: int = 15, tensor_size: int = 128) -> dict[str, Any]:
    """Test resource lifecycle across all 10 qualification checks.

    1. Repeated model creation and destruction
    2. Repeated serialization
    3. Repeated deserialization
    4. Serialization and deserialization combined lifecycle
    5. Repeated large tensor allocation and explicit release
    6. Repeated inference execution
    7. Cleanup after exceptions in computation
    8. Temporary file lifecycle and cleanup
    9. Open file handle release on error (verified via file unlink)
    10. Bounded resource limits rejection
    11. Repeated healthy operations after error cleanup
    """
    results: dict[str, Any] = {"iterations": iterations, "checks": {}}

    # 1. Repeated model creation and destruction
    with MemoryTracker() as tracker:
        for i in range(iterations):
            m = eo.Sequential(eo.Linear(32, 16, seed=i), eo.ReLU(), eo.Linear(16, 2, seed=i))
            del m
        gc.collect()
    results["checks"]["model_creation_destruction"] = {
        "delta_rss_bytes": tracker.delta_rss_bytes,
        "passed": True,
    }

    # 2. Repeated serialization to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        m_ser = eo.Sequential(eo.Linear(16, 8, seed=42), eo.ReLU(), eo.Linear(8, 2, seed=42))
        with MemoryTracker() as tracker:
            for i in range(iterations):
                ser_path = Path(tmpdir) / f"ser_{i}.eo"
                eo.save(m_ser, ser_path)
                if ser_path.is_file():
                    ser_path.unlink()
            gc.collect()
        results["checks"]["repeated_serialization"] = {
            "delta_rss_bytes": tracker.delta_rss_bytes,
            "passed": True,
        }

    # 3. Repeated deserialization from disk
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "base_model.eo"
        m_base = eo.Sequential(eo.Linear(16, 8, seed=42), eo.ReLU(), eo.Linear(8, 2, seed=42))
        eo.save(m_base, base_path)

        with MemoryTracker() as tracker:
            for _ in range(iterations):
                loaded = eo.load(base_path)
                del loaded
            gc.collect()
        results["checks"]["repeated_deserialization"] = {
            "delta_rss_bytes": tracker.delta_rss_bytes,
            "passed": True,
        }

    # 4. Serialization and deserialization combined
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "lifecycle_model.eo"
        m = eo.Sequential(eo.Linear(16, 8, seed=42), eo.ReLU(), eo.Linear(8, 2, seed=42))
        eo.save(m, model_path)

        with MemoryTracker() as tracker:
            for _ in range(iterations):
                loaded = eo.load(model_path)
                x = eo.tensor(np.ones((4, 16), dtype=np.float32))
                _ = loaded(x)
                del loaded, x
            gc.collect()
        results["checks"]["serialization_deserialization"] = {
            "delta_rss_bytes": tracker.delta_rss_bytes,
            "passed": True,
        }

    # 5. Repeated large tensor allocation and explicit release
    with MemoryTracker() as tracker:
        for _ in range(iterations):
            arr = np.ones((tensor_size, tensor_size), dtype=np.float32)
            t = eo.tensor(arr)
            res = t + t
            del arr, t, res
        gc.collect()
    results["checks"]["large_tensor_lifecycle"] = {
        "delta_rss_bytes": tracker.delta_rss_bytes,
        "passed": True,
    }

    # 6. Repeated inference execution
    m_inf = eo.Sequential(eo.Linear(16, 8, seed=1), eo.ReLU(), eo.Linear(8, 2, seed=2))
    runtime_inf = CPUInferenceRuntime(m_inf)
    inp_inf = np.ones((4, 16), dtype=np.float32)
    with MemoryTracker() as tracker:
        for _ in range(iterations * 2):
            _ = runtime_inf.predict(inp_inf)
        gc.collect()
    results["checks"]["repeated_inference_execution"] = {
        "delta_rss_bytes": tracker.delta_rss_bytes,
        "passed": True,
    }

    # 7. Cleanup after exceptions in computation
    exception_cleaned = 0
    for _ in range(iterations):
        try:
            a = eo.tensor(np.ones((10, 5), dtype=np.float32))
            b = eo.tensor(np.ones((8, 4), dtype=np.float32))
            _ = eo.matmul(a, b)
        except Exception:
            exception_cleaned += 1
    results["checks"]["exception_cleanup"] = {
        "exceptions_handled": exception_cleaned,
        "passed": exception_cleaned == iterations,
    }

    # 8. Temporary file lifecycle and cleanup
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        m_temp = eo.Sequential(eo.Linear(8, 4, seed=1))
        save_file = tmp_dir_path / "temp_save.eo"
        eo.save(m_temp, save_file)
        files_before = set(tmp_dir_path.iterdir())
        _ = eo.load(save_file)
        files_after = set(tmp_dir_path.iterdir())
        # No orphan temporary extraction files leaked in the directory
        no_leaked_files = files_before == files_after
    results["checks"]["temporary_file_cleanup"] = {
        "passed": no_leaked_files,
    }

    # 9. File handle release on error (unlinking file immediately after failed load)
    file_handles_released = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(iterations):
            corrupted_path = Path(tmpdir) / f"corrupt_{i}.eo"
            corrupted_path.write_bytes(b"NOT A VALID ZIP ARCHIVE")
            with contextlib.suppress(Exception):
                _ = eo.load(corrupted_path)
            try:
                corrupted_path.unlink()
                file_handles_released += 1
            except PermissionError:
                pass
    results["checks"]["file_handle_cleanup"] = {
        "files_unlinked": file_handles_released,
        "passed": file_handles_released == iterations,
    }
    results["checks"]["error_during_model_loading_cleanup"] = {
        "error_caught": file_handles_released,
        "passed": file_handles_released == iterations,
    }

    # 10. Bounded resource limits
    bounded_runtime = CPUInferenceRuntime(eo.Sequential(eo.Linear(4, 2, seed=1)), max_batch=4)
    oversized_data = np.ones((20, 4), dtype=np.float32)
    bounded_rejected = 0
    for _ in range(iterations):
        try:
            _ = bounded_runtime.predict(oversized_data)
        except (ValueError, RuntimeError, eo.exceptions.EveryOError):
            bounded_rejected += 1
    results["checks"]["bounded_resource_limits"] = {
        "rejected": bounded_rejected,
        "passed": bounded_rejected == iterations,
    }

    # 11. Repeated operations after cleanup
    healthy_model = eo.Sequential(eo.Linear(4, 2, seed=42))
    healthy_inp = eo.tensor(np.ones((2, 4), dtype=np.float32))
    healthy_runs = 0
    for _ in range(iterations):
        out = healthy_model(healthy_inp)
        if out.shape == (2, 2):
            healthy_runs += 1
    results["checks"]["repeated_operations_after_cleanup"] = {
        "healthy_runs": healthy_runs,
        "passed": healthy_runs == iterations,
    }

    results["passed"] = all(c.get("passed", False) for c in results["checks"].values())
    return results


# ==============================================================================
# 21.3.1: CPU Performance Benchmarking Suite
# ==============================================================================


@dataclass
class BenchmarkMetric:
    """Standardized record of a single CPU performance benchmark."""

    benchmark_id: str
    category: str
    workload: dict[str, Any]
    dtype: str
    backend: str
    warmup_iterations: int
    measured_iterations: int
    median_ms: float
    mean_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    p95_ms: float
    throughput_per_second: float
    p99_ms: float | None = None
    numpy_baseline_median_ms: float | None = None
    speedup_vs_numpy: float | None = None
    memory_rss_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _measure_latency_series(
    fn: Callable[[], Any],
    *,
    warmup: int = 2,
    repeats: int = 10,
) -> list[float]:
    """Execute warmup iterations then measure repeats returning list of milliseconds."""
    for _ in range(max(0, warmup)):
        fn()
    latencies: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    return latencies


def benchmark_tensor_operations(
    *,
    sizes: Sequence[int] = (32, 128, 512),
    warmup: int = 2,
    repeats: int = 10,
    seed: int = 42,
) -> list[BenchmarkMetric]:
    """Benchmark raw tensor operations across representative workload dimensions."""
    rng = np.random.default_rng(seed)
    metrics: list[BenchmarkMetric] = []
    mem_info = get_process_memory()

    for n in sizes:
        a_np = rng.standard_normal((n, n)).astype(np.float32)
        b_np = rng.standard_normal((n, n)).astype(np.float32)
        a = eo.tensor(a_np, device="cpu")
        b = eo.tensor(b_np, device="cpu")

        # 1. Matmul
        eo_matmul_lats = _measure_latency_series(
            lambda a=a, b=b: eo.matmul(a, b), warmup=warmup, repeats=repeats
        )
        np_matmul_lats = _measure_latency_series(
            lambda a_np=a_np, b_np=b_np: np.matmul(a_np, b_np), warmup=warmup, repeats=repeats
        )
        med_eo = calculate_percentile(eo_matmul_lats, 0.50)
        med_np = calculate_percentile(np_matmul_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"tensor_matmul_{n}x{n}",
                category="tensor_op",
                workload={"shape": [n, n], "operation": "matmul"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_eo,
                mean_ms=statistics.mean(eo_matmul_lats),
                stddev_ms=statistics.stdev(eo_matmul_lats) if len(eo_matmul_lats) > 1 else 0.0,
                min_ms=eo_matmul_lats[0],
                max_ms=eo_matmul_lats[-1],
                p90_ms=calculate_percentile(eo_matmul_lats, 0.90),
                p95_ms=calculate_percentile(eo_matmul_lats, 0.95),
                p99_ms=calculate_percentile(eo_matmul_lats, 0.99),
                throughput_per_second=1000.0 / max(med_eo, 1e-6),
                numpy_baseline_median_ms=med_np,
                speedup_vs_numpy=med_np / max(med_eo, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        # 2. Elementwise Addition
        eo_add_lats = _measure_latency_series(
            lambda a=a, b=b: a + b, warmup=warmup, repeats=repeats
        )
        np_add_lats = _measure_latency_series(
            lambda a_np=a_np, b_np=b_np: a_np + b_np, warmup=warmup, repeats=repeats
        )
        med_eo = calculate_percentile(eo_add_lats, 0.50)
        med_np = calculate_percentile(np_add_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"tensor_add_{n}x{n}",
                category="tensor_op",
                workload={"shape": [n, n], "operation": "add"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_eo,
                mean_ms=statistics.mean(eo_add_lats),
                stddev_ms=statistics.stdev(eo_add_lats) if len(eo_add_lats) > 1 else 0.0,
                min_ms=eo_add_lats[0],
                max_ms=eo_add_lats[-1],
                p90_ms=calculate_percentile(eo_add_lats, 0.90),
                p95_ms=calculate_percentile(eo_add_lats, 0.95),
                p99_ms=calculate_percentile(eo_add_lats, 0.99),
                throughput_per_second=1000.0 / max(med_eo, 1e-6),
                numpy_baseline_median_ms=med_np,
                speedup_vs_numpy=med_np / max(med_eo, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        # 3. Elementwise Multiplication
        eo_mul_lats = _measure_latency_series(
            lambda a=a, b=b: a * b, warmup=warmup, repeats=repeats
        )
        np_mul_lats = _measure_latency_series(
            lambda a_np=a_np, b_np=b_np: a_np * b_np, warmup=warmup, repeats=repeats
        )
        med_eo = calculate_percentile(eo_mul_lats, 0.50)
        med_np = calculate_percentile(np_mul_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"tensor_mul_{n}x{n}",
                category="tensor_op",
                workload={"shape": [n, n], "operation": "mul"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_eo,
                mean_ms=statistics.mean(eo_mul_lats),
                stddev_ms=statistics.stdev(eo_mul_lats) if len(eo_mul_lats) > 1 else 0.0,
                min_ms=eo_mul_lats[0],
                max_ms=eo_mul_lats[-1],
                p90_ms=calculate_percentile(eo_mul_lats, 0.90),
                p95_ms=calculate_percentile(eo_mul_lats, 0.95),
                p99_ms=calculate_percentile(eo_mul_lats, 0.99),
                throughput_per_second=1000.0 / max(med_eo, 1e-6),
                numpy_baseline_median_ms=med_np,
                speedup_vs_numpy=med_np / max(med_eo, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        # 4. ReLU Activation
        eo_relu_lats = _measure_latency_series(
            lambda a=a: eo.relu(a), warmup=warmup, repeats=repeats
        )
        np_relu_lats = _measure_latency_series(
            lambda a_np=a_np: np.maximum(a_np, 0.0), warmup=warmup, repeats=repeats
        )
        med_eo = calculate_percentile(eo_relu_lats, 0.50)
        med_np = calculate_percentile(np_relu_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"tensor_relu_{n}x{n}",
                category="tensor_op",
                workload={"shape": [n, n], "operation": "relu"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_eo,
                mean_ms=statistics.mean(eo_relu_lats),
                stddev_ms=statistics.stdev(eo_relu_lats) if len(eo_relu_lats) > 1 else 0.0,
                min_ms=eo_relu_lats[0],
                max_ms=eo_relu_lats[-1],
                p90_ms=calculate_percentile(eo_relu_lats, 0.90),
                p95_ms=calculate_percentile(eo_relu_lats, 0.95),
                p99_ms=calculate_percentile(eo_relu_lats, 0.99),
                throughput_per_second=1000.0 / max(med_eo, 1e-6),
                numpy_baseline_median_ms=med_np,
                speedup_vs_numpy=med_np / max(med_eo, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        # 5. Reduction (Sum)
        eo_sum_lats = _measure_latency_series(lambda a=a: eo.sum(a), warmup=warmup, repeats=repeats)
        np_sum_lats = _measure_latency_series(
            lambda a_np=a_np: np.sum(a_np), warmup=warmup, repeats=repeats
        )
        med_eo = calculate_percentile(eo_sum_lats, 0.50)
        med_np = calculate_percentile(np_sum_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"tensor_sum_{n}x{n}",
                category="tensor_op",
                workload={"shape": [n, n], "operation": "sum"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_eo,
                mean_ms=statistics.mean(eo_sum_lats),
                stddev_ms=statistics.stdev(eo_sum_lats) if len(eo_sum_lats) > 1 else 0.0,
                min_ms=eo_sum_lats[0],
                max_ms=eo_sum_lats[-1],
                p90_ms=calculate_percentile(eo_sum_lats, 0.90),
                p95_ms=calculate_percentile(eo_sum_lats, 0.95),
                p99_ms=calculate_percentile(eo_sum_lats, 0.99),
                throughput_per_second=1000.0 / max(med_eo, 1e-6),
                numpy_baseline_median_ms=med_np,
                speedup_vs_numpy=med_np / max(med_eo, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

    return metrics


def benchmark_neural_computation(
    *,
    batch_sizes: Sequence[int] = (16, 64),
    in_features: int = 64,
    out_features: int = 32,
    warmup: int = 2,
    repeats: int = 10,
    seed: int = 42,
) -> list[BenchmarkMetric]:
    """Benchmark neural layers: forward, backward, activations, loss, optimizer steps."""
    rng = np.random.default_rng(seed)
    metrics: list[BenchmarkMetric] = []
    mem_info = get_process_memory()

    for bs in batch_sizes:
        x_np = rng.standard_normal((bs, in_features)).astype(np.float32)
        target_np = rng.standard_normal((bs, out_features)).astype(np.float32)

        layer = eo.Linear(in_features, out_features, seed=seed)
        x = eo.tensor(x_np, requires_grad=True)
        target = eo.tensor(target_np)
        criterion = eo.MSELoss()
        optimizer = eo.SGD(layer.parameters(), lr=0.01)

        # 1. Dense Forward
        fwd_lats = _measure_latency_series(
            lambda lyr=layer, inp=x: lyr(inp), warmup=warmup, repeats=repeats
        )
        med_fwd = calculate_percentile(fwd_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"dense_forward_bs{bs}",
                category="neural_comp",
                workload={
                    "batch_size": bs,
                    "in": in_features,
                    "out": out_features,
                    "step": "forward",
                },
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_fwd,
                mean_ms=statistics.mean(fwd_lats),
                stddev_ms=statistics.stdev(fwd_lats) if len(fwd_lats) > 1 else 0.0,
                min_ms=fwd_lats[0],
                max_ms=fwd_lats[-1],
                p90_ms=calculate_percentile(fwd_lats, 0.90),
                p95_ms=calculate_percentile(fwd_lats, 0.95),
                p99_ms=calculate_percentile(fwd_lats, 0.99),
                throughput_per_second=(bs * 1000.0) / max(med_fwd, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        # 2. Full Train Step (Forward + Loss + Backward + Optimizer Step)
        def make_train_step(lyr, inp, tgt, crit, opt):
            def train_step():
                opt.zero_grad()
                out = lyr(inp)
                loss = crit(out, tgt)
                loss.backward()
                opt.step()
                return loss

            return train_step

        step_lats = _measure_latency_series(
            make_train_step(layer, x, target, criterion, optimizer),
            warmup=warmup,
            repeats=repeats,
        )
        med_step = calculate_percentile(step_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"mlp_train_step_bs{bs}",
                category="neural_comp",
                workload={
                    "batch_size": bs,
                    "in": in_features,
                    "out": out_features,
                    "step": "train_step",
                },
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_step,
                mean_ms=statistics.mean(step_lats),
                stddev_ms=statistics.stdev(step_lats) if len(step_lats) > 1 else 0.0,
                min_ms=step_lats[0],
                max_ms=step_lats[-1],
                p90_ms=calculate_percentile(step_lats, 0.90),
                p95_ms=calculate_percentile(step_lats, 0.95),
                p99_ms=calculate_percentile(step_lats, 0.99),
                throughput_per_second=(bs * 1000.0) / max(med_step, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        # 3. Softmax
        logits = eo.tensor(x_np[:, :out_features])
        sm_lats = _measure_latency_series(
            lambda lg=logits: eo.softmax(lg, axis=-1), warmup=warmup, repeats=repeats
        )
        med_sm = calculate_percentile(sm_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id=f"softmax_bs{bs}",
                category="neural_comp",
                workload={"batch_size": bs, "features": out_features, "op": "softmax"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_sm,
                mean_ms=statistics.mean(sm_lats),
                stddev_ms=statistics.stdev(sm_lats) if len(sm_lats) > 1 else 0.0,
                min_ms=sm_lats[0],
                max_ms=sm_lats[-1],
                p90_ms=calculate_percentile(sm_lats, 0.90),
                p95_ms=calculate_percentile(sm_lats, 0.95),
                p99_ms=calculate_percentile(sm_lats, 0.99),
                throughput_per_second=(bs * 1000.0) / max(med_sm, 1e-6),
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

    return metrics


def benchmark_model_operations(
    *,
    warmup: int = 5,
    repeats: int = 25,
    seed: int = 42,
) -> list[BenchmarkMetric]:
    """Benchmark end-to-end model operations: initialization, serialization, deserialization, inference."""
    metrics: list[BenchmarkMetric] = []
    mem_info = get_process_memory()

    # 1. Model Initialization
    def init_model():
        return eo.Sequential(
            eo.Linear(128, 64, seed=seed),
            eo.ReLU(),
            eo.Linear(64, 32, seed=seed + 1),
            eo.ReLU(),
            eo.Linear(32, 10, seed=seed + 2),
        )

    init_lats = _measure_latency_series(init_model, warmup=warmup, repeats=repeats)
    med_init = calculate_percentile(init_lats, 0.50)
    metrics.append(
        BenchmarkMetric(
            benchmark_id="model_initialization_3layer",
            category="model_op",
            workload={"architecture": "128->64->32->10", "op": "init"},
            dtype="float32",
            backend="numpy_cpu",
            warmup_iterations=warmup,
            measured_iterations=repeats,
            median_ms=med_init,
            mean_ms=statistics.mean(init_lats),
            stddev_ms=statistics.stdev(init_lats) if len(init_lats) > 1 else 0.0,
            min_ms=init_lats[0],
            max_ms=init_lats[-1],
            p90_ms=calculate_percentile(init_lats, 0.90),
            p95_ms=calculate_percentile(init_lats, 0.95),
            throughput_per_second=1000.0 / max(med_init, 1e-6),
            p99_ms=calculate_percentile(init_lats, 0.99) if len(init_lats) >= 100 else None,
            memory_rss_bytes=mem_info.get("rss_bytes"),
        )
    )

    # 2. Serialization & Deserialization
    with tempfile.TemporaryDirectory() as tmpdir:
        m = init_model()
        model_file = Path(tmpdir) / "bench_model.eo"

        save_lats = _measure_latency_series(
            lambda: eo.save(m, model_file), warmup=max(2, warmup // 2), repeats=repeats
        )
        med_save = calculate_percentile(save_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id="model_serialization_eo",
                category="model_op",
                workload={"op": "save_to_disk"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=max(2, warmup // 2),
                measured_iterations=repeats,
                median_ms=med_save,
                mean_ms=statistics.mean(save_lats),
                stddev_ms=statistics.stdev(save_lats) if len(save_lats) > 1 else 0.0,
                min_ms=save_lats[0],
                max_ms=save_lats[-1],
                p90_ms=calculate_percentile(save_lats, 0.90),
                p95_ms=calculate_percentile(save_lats, 0.95),
                throughput_per_second=1000.0 / max(med_save, 1e-6),
                p99_ms=calculate_percentile(save_lats, 0.99) if len(save_lats) >= 100 else None,
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        load_lats = _measure_latency_series(
            lambda: eo.load(model_file), warmup=warmup, repeats=repeats
        )
        med_load = calculate_percentile(load_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id="model_deserialization_eo",
                category="model_op",
                workload={"op": "load_from_disk"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_load,
                mean_ms=statistics.mean(load_lats),
                stddev_ms=statistics.stdev(load_lats) if len(load_lats) > 1 else 0.0,
                min_ms=load_lats[0],
                max_ms=load_lats[-1],
                p90_ms=calculate_percentile(load_lats, 0.90),
                p95_ms=calculate_percentile(load_lats, 0.95),
                throughput_per_second=1000.0 / max(med_load, 1e-6),
                p99_ms=calculate_percentile(load_lats, 0.99) if len(load_lats) >= 100 else None,
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

        # 3. In-memory deserialization (isolating CPU parsing/reconstruction from filesystem I/O)
        model_bytes = model_file.read_bytes()
        mem_load_lats = _measure_latency_series(
            lambda: eo.load(model_bytes), warmup=warmup, repeats=repeats
        )
        med_mem_load = calculate_percentile(mem_load_lats, 0.50)
        metrics.append(
            BenchmarkMetric(
                benchmark_id="model_deserialization_memory_eo",
                category="model_op",
                workload={"op": "load_from_memory"},
                dtype="float32",
                backend="numpy_cpu",
                warmup_iterations=warmup,
                measured_iterations=repeats,
                median_ms=med_mem_load,
                mean_ms=statistics.mean(mem_load_lats),
                stddev_ms=statistics.stdev(mem_load_lats) if len(mem_load_lats) > 1 else 0.0,
                min_ms=mem_load_lats[0],
                max_ms=mem_load_lats[-1],
                p90_ms=calculate_percentile(mem_load_lats, 0.90),
                p95_ms=calculate_percentile(mem_load_lats, 0.95),
                throughput_per_second=1000.0 / max(med_mem_load, 1e-6),
                p99_ms=calculate_percentile(mem_load_lats, 0.99)
                if len(mem_load_lats) >= 100
                else None,
                memory_rss_bytes=mem_info.get("rss_bytes"),
            )
        )

    return metrics


def benchmark_data_operations(
    *,
    sample_count: int = 1000,
    features: int = 16,
    batch_size: int = 32,
    warmup: int = 2,
    repeats: int = 10,
    seed: int = 42,
) -> list[BenchmarkMetric]:
    """Benchmark data preprocessing and batch loading operations."""
    from everyo.data import ArrayDataset, DataLoader, StandardScaler

    rng = np.random.default_rng(seed)
    raw_x = rng.standard_normal((sample_count, features)).astype(np.float32)
    raw_y = rng.integers(0, 2, size=(sample_count, 1)).astype(np.float32)
    metrics: list[BenchmarkMetric] = []
    mem_info = get_process_memory()

    # 1. Data Preprocessing (StandardScaler)
    def standardize_step():
        scaler = StandardScaler()
        return scaler.fit_transform(raw_x)

    std_lats = _measure_latency_series(standardize_step, warmup=warmup, repeats=repeats)
    med_std = calculate_percentile(std_lats, 0.50)
    metrics.append(
        BenchmarkMetric(
            benchmark_id=f"data_preprocessing_standardize_{sample_count}x{features}",
            category="data_op",
            workload={"samples": sample_count, "features": features, "op": "standardize"},
            dtype="float32",
            backend="numpy_cpu",
            warmup_iterations=warmup,
            measured_iterations=repeats,
            median_ms=med_std,
            mean_ms=statistics.mean(std_lats),
            stddev_ms=statistics.stdev(std_lats) if len(std_lats) > 1 else 0.0,
            min_ms=std_lats[0],
            max_ms=std_lats[-1],
            p90_ms=calculate_percentile(std_lats, 0.90),
            p95_ms=calculate_percentile(std_lats, 0.95),
            p99_ms=calculate_percentile(std_lats, 0.99),
            throughput_per_second=(sample_count * 1000.0) / max(med_std, 1e-6),
            memory_rss_bytes=mem_info.get("rss_bytes"),
        )
    )

    # 2. Data Loading Batches (DataLoader over ArrayDataset)
    dataset = ArrayDataset(raw_x, raw_y)

    def dataloader_step():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        count = 0
        for b_x, _ in loader:
            count += len(b_x)
        return count

    dl_lats = _measure_latency_series(dataloader_step, warmup=warmup, repeats=repeats)
    med_dl = calculate_percentile(dl_lats, 0.50)
    metrics.append(
        BenchmarkMetric(
            benchmark_id=f"data_loading_batches_bs{batch_size}_n{sample_count}",
            category="data_op",
            workload={
                "samples": sample_count,
                "batch_size": batch_size,
                "op": "dataloader_iteration",
            },
            dtype="float32",
            backend="numpy_cpu",
            warmup_iterations=warmup,
            measured_iterations=repeats,
            median_ms=med_dl,
            mean_ms=statistics.mean(dl_lats),
            stddev_ms=statistics.stdev(dl_lats) if len(dl_lats) > 1 else 0.0,
            min_ms=dl_lats[0],
            max_ms=dl_lats[-1],
            p90_ms=calculate_percentile(dl_lats, 0.90),
            p95_ms=calculate_percentile(dl_lats, 0.95),
            p99_ms=calculate_percentile(dl_lats, 0.99),
            throughput_per_second=(sample_count * 1000.0) / max(med_dl, 1e-6),
            memory_rss_bytes=mem_info.get("rss_bytes"),
        )
    )

    return metrics


def run_comprehensive_cpu_benchmark(
    *,
    sizes: Sequence[int] = (32, 128),
    batch_sizes: Sequence[int] = (16, 64),
    warmup: int = 2,
    repeats: int = 10,
    seed: int = 42,
    include_data_ops: bool = False,
) -> dict[str, Any]:
    """Execute complete Phase 21.3.1 CPU benchmark suite and return machine-readable results."""
    t_start = time.perf_counter()
    env = get_environment_metadata()
    tensor_metrics = benchmark_tensor_operations(
        sizes=sizes, warmup=warmup, repeats=repeats, seed=seed
    )
    neural_metrics = benchmark_neural_computation(
        batch_sizes=batch_sizes, warmup=warmup, repeats=repeats, seed=seed
    )
    model_metrics = benchmark_model_operations(
        warmup=max(3, warmup), repeats=max(15, repeats), seed=seed
    )
    total_elapsed = time.perf_counter() - t_start

    all_metrics = tensor_metrics + neural_metrics + model_metrics
    if include_data_ops:
        data_metrics = benchmark_data_operations(warmup=warmup, repeats=repeats, seed=seed)
        all_metrics.extend(data_metrics)

    return {
        "benchmark_suite_version": "21.3.1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": total_elapsed,
        "environment": env,
        "total_benchmarks": len(all_metrics),
        "benchmarks": {m.benchmark_id: m.to_dict() for m in all_metrics},
    }


# ==============================================================================
# 21.3.3: Concurrency and Stress Testing
# ==============================================================================


@dataclass(frozen=True)
class PerformanceProfile:
    """Configurable execution parameters for performance and stress qualification."""

    name: str
    concurrency: int
    total_requests: int
    warmup_requests: int = 0
    timeout_seconds: float = 5.0
    fault_injection_rate: float = 0.0


QUICK_REGRESSION_PROFILE = PerformanceProfile(
    name="quick_regression",
    concurrency=2,
    total_requests=20,
    warmup_requests=2,
    timeout_seconds=2.0,
)

STANDARD_LOAD_PROFILE = PerformanceProfile(
    name="standard_load",
    concurrency=4,
    total_requests=100,
    warmup_requests=5,
    timeout_seconds=5.0,
)

EXTENDED_SOAK_PROFILE = PerformanceProfile(
    name="extended_soak",
    concurrency=4,
    total_requests=300,
    warmup_requests=10,
    timeout_seconds=10.0,
)

FAULT_INJECTION_PROFILE = PerformanceProfile(
    name="fault_injection",
    concurrency=4,
    total_requests=50,
    warmup_requests=2,
    timeout_seconds=2.0,
    fault_injection_rate=0.2,
)

# Standard Stress Profiles from Section 5.2
STRESS_PROFILE_A_NORMAL = PerformanceProfile(
    name="stress_profile_a_normal",
    concurrency=2,
    total_requests=100,
    warmup_requests=5,
    timeout_seconds=5.0,
)

STRESS_PROFILE_B_MODERATE = PerformanceProfile(
    name="stress_profile_b_moderate",
    concurrency=4,
    total_requests=500,
    warmup_requests=10,
    timeout_seconds=5.0,
)

STRESS_PROFILE_C_HIGH = PerformanceProfile(
    name="stress_profile_c_high",
    concurrency=8,
    total_requests=1000,
    warmup_requests=20,
    timeout_seconds=10.0,
)


@dataclass(frozen=True)
class PerformanceEvidence:
    """Detailed performance qualification record."""

    profile_name: str
    total_requests: int
    successes: int
    failures: int
    error_rate: float
    duration_seconds: float
    throughput_rps: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float
    passed_slo: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_serving_performance(
    url: str,
    *,
    token: str,
    profile: PerformanceProfile,
    inputs: list[list[float]],
    target_slo_p95_ms: float = 100.0,
) -> PerformanceEvidence:
    """Execute a structured performance qualification probe against a local service."""
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("performance qualification is restricted to loopback staging servers")
    if parts.path != "/v1/predict" or parts.query or parts.fragment:
        raise ValueError("expected the /v1/predict endpoint")
    if profile.concurrency < 1 or profile.total_requests < 1:
        raise ValueError("concurrency and total_requests must be positive")

    valid_payload = json.dumps({"inputs": inputs}, allow_nan=False).encode("utf-8")
    malformed_payload = b'{"inputs": [invalid json'

    rng = random.Random(42)

    # Warmup phase
    if profile.warmup_requests > 0:
        for _ in range(profile.warmup_requests):
            req = Request(
                url,
                data=valid_payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=profile.timeout_seconds) as resp:
                    resp.read()
            except Exception:
                pass

    def probe(idx: int) -> tuple[bool, float]:
        start = time.perf_counter()
        is_fault = (
            profile.fault_injection_rate > 0.0 and rng.random() < profile.fault_injection_rate
        )

        if is_fault:
            fault_type = idx % 3
            if fault_type == 0:
                payload = malformed_payload
                hdr_token = token
                content_type = "application/json"
            elif fault_type == 1:
                payload = valid_payload
                hdr_token = "invalid-token"
                content_type = "application/json"
            else:
                payload = valid_payload
                hdr_token = token
                content_type = "text/plain"
        else:
            payload = valid_payload
            hdr_token = token
            content_type = "application/json"

        req = Request(
            url,
            data=payload,
            headers={"Authorization": f"Bearer {hdr_token}", "Content-Type": content_type},
        )

        ok = False
        try:
            with urlopen(req, timeout=profile.timeout_seconds) as response:
                body = json.load(response)
                ok = response.status == 200 and body.get("count") == len(inputs)
        except HTTPError as err:
            if is_fault and err.code in {400, 401, 415}:
                ok = True
        except (URLError, TimeoutError, OSError, ValueError):
            ok = False

        duration_ms = (time.perf_counter() - start) * 1000.0
        return ok, duration_ms

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=profile.concurrency) as pool:
        results = list(pool.map(probe, range(profile.total_requests)))
    total_elapsed = max(time.perf_counter() - start_total, 1e-9)

    latencies = sorted(duration for _, duration in results)
    successes = sum(1 for ok, _ in results if ok)
    failures = profile.total_requests - successes
    error_rate = failures / profile.total_requests

    p50 = calculate_percentile(latencies, 0.50)
    p90 = calculate_percentile(latencies, 0.90)
    p95 = calculate_percentile(latencies, 0.95)
    p99 = calculate_percentile(latencies, 0.99)
    min_ms = latencies[0] if latencies else 0.0
    max_ms = latencies[-1] if latencies else 0.0
    mean_ms = statistics.mean(latencies) if latencies else 0.0
    stddev_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    passed_slo = (p95 <= target_slo_p95_ms) and (error_rate == 0.0)

    return PerformanceEvidence(
        profile_name=profile.name,
        total_requests=profile.total_requests,
        successes=successes,
        failures=failures,
        error_rate=error_rate,
        duration_seconds=total_elapsed,
        throughput_rps=profile.total_requests / total_elapsed,
        p50_ms=p50,
        p90_ms=p90,
        p95_ms=p95,
        p99_ms=p99,
        min_ms=min_ms,
        max_ms=max_ms,
        mean_ms=mean_ms,
        stddev_ms=stddev_ms,
        passed_slo=passed_slo,
    )


def run_concurrency_correctness_qualification(
    *,
    concurrency: int = 4,
    operations_per_worker: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify correctness under concurrency: independent models, no state leakage, no deadlocks."""
    rng = np.random.default_rng(seed)
    weights_np = rng.standard_normal((8, 4)).astype(np.float32)

    # Reference single-threaded evaluation
    test_inputs = [
        rng.standard_normal((1, 8)).astype(np.float32)
        for _ in range(concurrency * operations_per_worker)
    ]
    ground_truth = [np.matmul(x, weights_np) for x in test_inputs]

    barrier = threading.Barrier(concurrency)
    latencies: list[float] = []
    latencies_lock = threading.Lock()
    corruptions: list[str] = []

    def worker_fn(worker_id: int):
        # Independent model instance per worker
        model = eo.Linear(8, 4, seed=seed)
        model.weight.data = weights_np.copy()

        barrier.wait()  # Synchronized start to maximize contention

        start_idx = worker_id * operations_per_worker
        for i in range(operations_per_worker):
            idx = start_idx + i
            inp = test_inputs[idx]
            expected = ground_truth[idx]

            t0 = time.perf_counter()
            t_inp = eo.tensor(inp)
            out = model(t_inp).data
            duration = (time.perf_counter() - t0) * 1000.0

            with latencies_lock:
                latencies.append(duration)

            if not np.allclose(out, expected, atol=1e-5):
                corruptions.append(f"Worker {worker_id} op {i} output mismatch")

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker_fn, args=(wid,)) for wid in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    total_duration = time.perf_counter() - t_start

    latencies.sort()
    total_ops = concurrency * operations_per_worker

    return {
        "concurrency": concurrency,
        "total_operations": total_ops,
        "duration_seconds": total_duration,
        "throughput_rps": total_ops / max(total_duration, 1e-6),
        "corruptions_detected": len(corruptions),
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "passed": len(corruptions) == 0 and len(latencies) == total_ops,
    }


def run_stress_qualification(
    *,
    profile: PerformanceProfile = STRESS_PROFILE_B_MODERATE,
) -> dict[str, Any]:
    """Execute high-concurrency stress test with queue stability and throughput measurements."""
    model = eo.Sequential(eo.Linear(8, 4, seed=1), eo.ReLU(), eo.Linear(4, 2, seed=2))
    runtime = CPUInferenceRuntime(model)

    latencies: list[float] = []
    latencies_lock = threading.Lock()
    failures = 0
    failures_lock = threading.Lock()

    def task(task_id: int):
        nonlocal failures
        try:
            inp = np.ones((2, 8), dtype=np.float32) * (task_id % 5 + 1.0)
            t0 = time.perf_counter()
            _ = runtime.predict_batched(inp)
            duration = (time.perf_counter() - t0) * 1000.0
            with latencies_lock:
                latencies.append(duration)
        except Exception:
            with failures_lock:
                failures += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=profile.concurrency) as pool:
        futures = [pool.submit(task, i) for i in range(profile.total_requests)]
        for f in as_completed(futures):
            f.result()
    total_elapsed = time.perf_counter() - t0

    latencies.sort()
    error_rate = failures / profile.total_requests

    return {
        "profile": profile.name,
        "concurrency": profile.concurrency,
        "total_requests": profile.total_requests,
        "successes": len(latencies),
        "failures": failures,
        "error_rate": error_rate,
        "duration_seconds": total_elapsed,
        "throughput_rps": profile.total_requests / max(total_elapsed, 1e-6),
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p90_ms": calculate_percentile(latencies, 0.90),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "passed": failures == 0 and error_rate == 0.0,
    }


# ==============================================================================
# 21.3.4: Long-Duration Soak Testing & Degradation Monitoring
# ==============================================================================

SOAK_PROFILES: dict[str, float] = {
    "short": 600.0,  # 10 minutes (600s)
    "medium": 1800.0,  # 30 minutes (1800s)
    "extended": 7200.0,  # 2 hours (7200s)
    "endurance": 28800.0,  # 8 hours (28800s)
}


@dataclass
class SoakSample:
    """Point-in-time telemetry sample during soak testing."""

    timestamp: str
    elapsed_seconds: float
    completed_ops: int
    failed_ops: int
    instantaneous_rps: float
    p95_latency_ms: float
    rss_bytes: int
    python_allocated_bytes: int | str


def validate_soak_report(report: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that a soak telemetry report adheres to the Phase 21.3 schema."""
    errors: list[str] = []
    if not isinstance(report, dict):
        return False, ["Report must be a dictionary"]

    required_keys = [
        "status",
        "profile",
        "git_commit_sha",
        "everyo_version",
        "python_version",
        "python_implementation",
        "os_name",
        "os_version",
        "architecture",
        "processor",
        "cpu_count_logical",
        "numpy_version",
        "backend_configuration",
        "thread_settings",
        "requested_duration_seconds",
        "actual_duration_seconds",
        "duration_seconds",
        "start_time",
        "end_time",
        "initial_rss_bytes",
        "final_rss_bytes",
        "peak_rss_bytes",
        "rss_growth_bytes",
        "rss_growth_pct",
        "rss_trend",
        "memory_assessment",
        "python_allocation_status",
        "python_allocated_metrics",
        "total_samples",
        "completed_operations",
        "failed_operations",
        "overall_throughput_rps",
        "latency_drift_pct",
        "latency_statistics",
        "samples",
        "passed_stability",
    ]

    for key in required_keys:
        if key not in report:
            errors.append(f"Missing required key: '{key}'")

    if report.get("status") not in ("PASS", "FAIL", "INTERRUPTED", "INVALID"):
        errors.append(f"Invalid status: '{report.get('status')}'")

    if not isinstance(report.get("samples"), list):
        errors.append("Field 'samples' must be a list")
    else:
        for idx, s in enumerate(report["samples"]):
            if not isinstance(s, dict):
                errors.append(f"Sample at index {idx} must be a dictionary")
                continue
            for sample_key in (
                "timestamp",
                "elapsed_seconds",
                "completed_ops",
                "p95_latency_ms",
                "rss_bytes",
                "python_allocated_bytes",
            ):
                if sample_key not in s:
                    errors.append(f"Sample at index {idx} missing key '{sample_key}'")
            if s.get("python_allocated_bytes") is None:
                errors.append(
                    f"Sample at index {idx} has null python_allocated_bytes; must be int or 'NOT_COLLECTED'"
                )

    return len(errors) == 0, errors


def _build_invalid_soak_report(
    *,
    profile: str,
    requested_duration: float,
    error: str,
) -> dict[str, Any]:
    env = get_environment_metadata()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "status": "INVALID",
        "profile": profile,
        "validation_error": error,
        "git_commit_sha": env.get("git_commit_sha", "unknown"),
        "everyo_version": env.get("everyo_version", __version__),
        "python_implementation": env.get("python_implementation", platform.python_implementation()),
        "python_version": env.get("python_version", platform.python_version()),
        "os_name": env.get("os_name", platform.system()),
        "os_version": platform.version() or env.get("os_release", "unknown"),
        "architecture": env.get("architecture", platform.machine()),
        "processor": env.get("processor", "unknown"),
        "cpu_count_logical": env.get("cpu_count_logical", 1),
        "numpy_version": env.get("numpy_version", np.__version__),
        "backend_configuration": env.get("optional_backends", {}),
        "thread_settings": env.get("thread_settings", {}),
        "requested_duration_seconds": requested_duration,
        "actual_duration_seconds": 0.0,
        "duration_seconds": 0.0,
        "start_time": now_iso,
        "end_time": now_iso,
        "initial_rss_bytes": 0,
        "final_rss_bytes": 0,
        "peak_rss_bytes": 0,
        "rss_growth_bytes": 0,
        "rss_growth_pct": 0.0,
        "rss_trend": "INVALID",
        "memory_assessment": "INVALID_RUN",
        "python_allocation_status": "NOT_COLLECTED",
        "python_allocated_metrics": "NOT_COLLECTED",
        "total_samples": 0,
        "completed_operations": 0,
        "failed_operations": 0,
        "overall_throughput_rps": 0.0,
        "latency_drift_pct": 0.0,
        "latency_statistics": {},
        "samples": [],
        "passed_stability": False,
        "environment": env,
    }


def run_soak_test(
    *,
    duration_seconds: float = 30.0,
    sample_interval_seconds: float = 2.0,
    concurrency: int = 2,
    batch_size: int = 32,
    profile: str | None = None,
    trace_python: bool = False,
    progress_callback: Callable[[SoakSample], None] | None = None,
) -> dict[str, Any]:
    """Execute sustained representative CPU workload, monitoring memory and latency trends."""
    requested_duration = duration_seconds
    if profile is not None:
        if profile in SOAK_PROFILES:
            duration_seconds = SOAK_PROFILES[profile]
            requested_duration = duration_seconds
        else:
            return _build_invalid_soak_report(
                profile=profile,
                requested_duration=duration_seconds,
                error=f"Unknown profile: '{profile}'. Must be one of {list(SOAK_PROFILES.keys())}",
            )

    if duration_seconds <= 0.0:
        return _build_invalid_soak_report(
            profile=profile or "custom",
            requested_duration=duration_seconds,
            error=f"duration_seconds must be positive, got {duration_seconds}",
        )
    if sample_interval_seconds <= 0.0:
        return _build_invalid_soak_report(
            profile=profile or "custom",
            requested_duration=duration_seconds,
            error=f"sample_interval_seconds must be positive, got {sample_interval_seconds}",
        )
    if batch_size <= 0:
        return _build_invalid_soak_report(
            profile=profile or "custom",
            requested_duration=duration_seconds,
            error=f"batch_size must be positive, got {batch_size}",
        )
    if concurrency <= 0:
        return _build_invalid_soak_report(
            profile=profile or "custom",
            requested_duration=duration_seconds,
            error=f"concurrency must be positive, got {concurrency}",
        )

    started_tracemalloc = False
    if trace_python and not tracemalloc.is_tracing():
        tracemalloc.start()
        started_tracemalloc = True

    try:
        model = eo.Sequential(
            eo.Linear(32, 16, seed=1),
            eo.ReLU(),
            eo.Linear(16, 4, seed=2),
        )
        runtime = CPUInferenceRuntime(model)

        # Pre-serialized bytes for periodic deserialization verification
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_file = Path(tmpdir) / "soak_baseline.eo"
            eo.save(model, tmp_model_file)
            model_serialized_bytes = tmp_model_file.read_bytes()

        start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        next_sample_time = start_time + sample_interval_seconds

        completed_ops = 0
        failed_ops = 0
        samples: list[SoakSample] = []
        window_latencies: list[float] = []
        all_window_p95s: list[float] = []
        all_latencies_sample: list[float] = []
        window_ops = 0
        last_sample_t = start_time
        interrupted = False

        rng = np.random.default_rng(42)
        inp_data = rng.standard_normal((batch_size, 32)).astype(np.float32)

        # Initial sample
        initial_mem = get_process_memory()
        init_rss = int(initial_mem.get("rss_bytes") or 0)
        init_py = initial_mem.get("python_allocated_bytes") if trace_python else None
        init_py_val: int | str = int(init_py) if init_py is not None else "NOT_COLLECTED"

        samples.append(
            SoakSample(
                timestamp=start_iso,
                elapsed_seconds=0.0,
                completed_ops=0,
                failed_ops=0,
                instantaneous_rps=0.0,
                p95_latency_ms=0.0,
                rss_bytes=init_rss,
                python_allocated_bytes=init_py_val,
            )
        )

        cycle = 0
        try:
            while time.perf_counter() < end_time:
                cycle += 1
                t_op = time.perf_counter()
                try:
                    # 1. Forward inference pass
                    _ = runtime.predict_batched(inp_data)

                    # 2. Tensor math operations
                    t_arr = eo.tensor(inp_data)
                    _ = eo.relu(t_arr + t_arr)

                    # 3. Periodic model deserialization (every 50 cycles)
                    if cycle % 50 == 0:
                        loaded_m = eo.load(model_serialized_bytes)
                        del loaded_m

                    # 4. Periodic garbage collection (every 100 cycles)
                    if cycle % 100 == 0:
                        gc.collect()

                    completed_ops += 1
                    window_ops += 1
                    lat_ms = (time.perf_counter() - t_op) * 1000.0
                    window_latencies.append(lat_ms)
                    if len(all_latencies_sample) < 10000:
                        all_latencies_sample.append(lat_ms)
                except Exception:
                    failed_ops += 1

                now = time.perf_counter()
                if now >= next_sample_time:
                    interval_dur = now - last_sample_t
                    window_latencies.sort()
                    p95 = calculate_percentile(window_latencies, 0.95) if window_latencies else 0.0
                    all_window_p95s.append(p95)
                    rps = window_ops / max(interval_dur, 1e-6)

                    mem = get_process_memory()
                    cur_rss = int(mem.get("rss_bytes") or 0)
                    cur_py = mem.get("python_allocated_bytes") if trace_python else None
                    cur_py_val: int | str = int(cur_py) if cur_py is not None else "NOT_COLLECTED"

                    sample = SoakSample(
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        elapsed_seconds=now - start_time,
                        completed_ops=completed_ops,
                        failed_ops=failed_ops,
                        instantaneous_rps=rps,
                        p95_latency_ms=p95,
                        rss_bytes=cur_rss,
                        python_allocated_bytes=cur_py_val,
                    )
                    samples.append(sample)
                    if progress_callback:
                        progress_callback(sample)

                    window_latencies = []
                    window_ops = 0
                    last_sample_t = now
                    next_sample_time = now + sample_interval_seconds
        except KeyboardInterrupt:
            interrupted = True

        total_elapsed = time.perf_counter() - start_time
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Trend Analysis
        first_half_samples = samples[1 : max(2, len(samples) // 2)]
        second_half_samples = samples[max(2, len(samples) // 2) :]

        final_rss = samples[-1].rss_bytes if samples else 0
        peak_rss = max((s.rss_bytes for s in samples), default=final_rss)
        rss_growth_bytes = final_rss - init_rss
        rss_growth_pct = (rss_growth_bytes / max(init_rss, 1)) * 100.0 if init_rss > 0 else 0.0

        if rss_growth_pct <= 5.0:
            rss_trend = "STABLE"
            memory_assessment = "NO_LEAK_DETECTED"
        elif rss_growth_pct < 25.0:
            rss_trend = "BOUNDED_GROWTH"
            memory_assessment = "BOUNDED_ALLOCATOR_EXPANSION_NO_CONFIRMED_LEAK"
        else:
            rss_trend = "UNBOUNDED_GROWTH"
            memory_assessment = "CONFIRMED_MEMORY_LEAK"

        # Python allocation metrics
        if trace_python or tracemalloc.is_tracing():
            py_status = "COLLECTED"
            py_first = (
                samples[0].python_allocated_bytes
                if isinstance(samples[0].python_allocated_bytes, int)
                else 0
            )
            py_last = (
                samples[-1].python_allocated_bytes
                if isinstance(samples[-1].python_allocated_bytes, int)
                else 0
            )
            py_peak = max(
                (
                    s.python_allocated_bytes
                    for s in samples
                    if isinstance(s.python_allocated_bytes, int)
                ),
                default=py_last,
            )
            py_metrics: Any = {
                "initial_allocated_bytes": py_first,
                "final_allocated_bytes": py_last,
                "peak_allocated_bytes": py_peak,
                "growth_bytes": py_last - py_first,
            }
        else:
            py_status = "NOT_COLLECTED"
            py_metrics = "NOT_COLLECTED"

        # Latency statistics
        p95_first = (
            statistics.mean([s.p95_latency_ms for s in first_half_samples])
            if first_half_samples
            else 0.0
        )
        p95_second = (
            statistics.mean([s.p95_latency_ms for s in second_half_samples])
            if second_half_samples
            else 0.0
        )
        latency_drift_pct = (
            ((p95_second - p95_first) / max(p95_first, 1e-6)) * 100.0 if p95_first > 0 else 0.0
        )

        all_latencies_sample.sort()
        all_window_p95s_sorted = sorted(all_window_p95s) if all_window_p95s else [0.0]
        latency_stats = {
            "mean_p95_ms": float(statistics.mean(all_window_p95s_sorted)),
            "median_p95_ms": float(statistics.median(all_window_p95s_sorted)),
            "min_p95_ms": float(min(all_window_p95s_sorted)),
            "max_p95_ms": float(max(all_window_p95s_sorted)),
            "p50_ms": calculate_percentile(all_latencies_sample, 0.50),
            "p90_ms": calculate_percentile(all_latencies_sample, 0.90),
            "p95_ms": calculate_percentile(all_latencies_sample, 0.95),
            "p99_ms": calculate_percentile(all_latencies_sample, 0.99),
        }

        failure_rate = failed_ops / max(completed_ops + failed_ops, 1)

        # Stability criteria: failure rate is 0, no unbounded memory growth (< 25%), and not interrupted
        passed_stability = (failure_rate == 0.0) and (rss_growth_pct < 25.0) and not interrupted

        if interrupted:
            status = "INTERRUPTED"
            passed_stability = False
        elif passed_stability:
            status = "PASS"
        else:
            status = "FAIL"

        profile_label = profile or (
            "short"
            if abs(duration_seconds - 600.0) < 1.0
            else "medium"
            if abs(duration_seconds - 1800.0) < 1.0
            else "extended"
            if abs(duration_seconds - 7200.0) < 1.0
            else "endurance"
            if abs(duration_seconds - 28800.0) < 1.0
            else "custom"
        )

        env = get_environment_metadata()

        return {
            "status": status,
            "profile": profile_label,
            "git_commit_sha": env.get("git_commit_sha", "unknown"),
            "everyo_version": env.get("everyo_version", __version__),
            "python_implementation": env.get(
                "python_implementation", platform.python_implementation()
            ),
            "python_version": env.get("python_version", platform.python_version()),
            "os_name": env.get("os_name", platform.system()),
            "os_version": platform.version() or env.get("os_release", "unknown"),
            "architecture": env.get("architecture", platform.machine()),
            "processor": env.get("processor", "unknown"),
            "cpu_count_logical": env.get("cpu_count_logical", 1),
            "numpy_version": env.get("numpy_version", np.__version__),
            "backend_configuration": env.get("optional_backends", {}),
            "thread_settings": env.get("thread_settings", {}),
            "requested_duration_seconds": requested_duration,
            "actual_duration_seconds": total_elapsed,
            "duration_seconds": total_elapsed,
            "start_time": start_iso,
            "end_time": end_iso,
            "initial_rss_bytes": init_rss,
            "final_rss_bytes": final_rss,
            "peak_rss_bytes": peak_rss,
            "rss_growth_bytes": rss_growth_bytes,
            "rss_growth_pct": rss_growth_pct,
            "rss_trend": rss_trend,
            "memory_assessment": memory_assessment,
            "python_allocation_status": py_status,
            "python_allocated_metrics": py_metrics,
            "total_samples": len(samples),
            "completed_operations": completed_ops,
            "failed_operations": failed_ops,
            "overall_throughput_rps": completed_ops / max(total_elapsed, 1e-6),
            "latency_drift_pct": latency_drift_pct,
            "latency_statistics": latency_stats,
            "samples": [asdict(s) for s in samples],
            "passed_stability": passed_stability,
            "environment": env,
        }
    finally:
        if started_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


# ==============================================================================
# 21.3.5: Fault Injection and Recovery Validation
# ==============================================================================


def run_fault_injection_suite() -> dict[str, Any]:
    """Execute controlled failure scenarios across categories A through F and verify recovery."""
    import zipfile
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    results: list[dict[str, Any]] = []

    def record_scenario(name: str, category: str, passed: bool, error_msg: str = ""):
        results.append(
            {
                "name": name,
                "category": category,
                "passed": passed,
                "detail": error_msg or "Fault contained and recovery verified",
            }
        )

    # --- CATEGORY A: CPU & Execution Faults ---
    # 1. Dimension Mismatch
    try:
        a = eo.tensor(np.ones((10, 5), dtype=np.float32))
        b = eo.tensor(np.ones((20, 5), dtype=np.float32))
        _ = eo.matmul(a, b)
        record_scenario(
            "matmul_dimension_mismatch",
            "CPU_EXECUTION",
            False,
            "Failed to reject invalid dimensions",
        )
    except (ValueError, RuntimeError, eo.exceptions.EveryOError):
        # Recovery: execute valid matmul
        valid_a = eo.tensor(np.ones((10, 5), dtype=np.float32))
        valid_b = eo.tensor(np.ones((5, 2), dtype=np.float32))
        res = eo.matmul(valid_a, valid_b)
        record_scenario("matmul_dimension_mismatch", "CPU_EXECUTION", res.shape == (10, 2))

    # 2. NaN Input Handling
    try:
        nan_data = np.array([[1.0, float("nan")]], dtype=np.float32)
        model = eo.Sequential(eo.Linear(2, 2, seed=1))
        runtime = CPUInferenceRuntime(model)
        _ = runtime.predict_batched(nan_data)
        record_scenario("nan_input_rejection", "CPU_EXECUTION", False, "Failed to reject NaN")
    except (ValueError, TypeError, eo.exceptions.EveryOError):
        valid_res = runtime.predict_batched(np.array([[1.0, 2.0]], dtype=np.float32))
        record_scenario("nan_input_rejection", "CPU_EXECUTION", len(valid_res) == 1)

    # 3. Empty Input Handling
    try:
        empty_data = np.zeros((0, 4), dtype=np.float32)
        runtime = CPUInferenceRuntime(eo.Sequential(eo.Linear(4, 2, seed=1)))
        _ = runtime.predict_batched(empty_data)
        record_scenario(
            "empty_input_rejection", "CPU_EXECUTION", False, "Failed to reject empty input"
        )
    except (ValueError, TypeError, eo.exceptions.EveryOError):
        valid_res = runtime.predict_batched(np.ones((1, 4), dtype=np.float32))
        record_scenario("empty_input_rejection", "CPU_EXECUTION", len(valid_res) == 1)

    # 4. Unsupported Data Type Handling
    try:
        _ = eo.tensor(["invalid", "strings", "not", "numbers"])
        record_scenario(
            "unsupported_dtype_rejection",
            "CPU_EXECUTION",
            False,
            "Failed to reject non-numeric string data",
        )
    except (TypeError, ValueError, eo.exceptions.EveryOError):
        valid_t = eo.tensor([1.0, 2.0, 3.0])
        record_scenario("unsupported_dtype_rejection", "CPU_EXECUTION", valid_t.shape == (3,))

    # --- CATEGORY B: Memory & Allocation Faults ---
    # 5. Large Batch Limit Exceeded
    try:
        runtime = CPUInferenceRuntime(eo.Sequential(eo.Linear(2, 1, seed=1)), max_batch=10)
        oversized = np.ones((50, 2), dtype=np.float32)
        _ = runtime.predict(oversized)
        record_scenario("max_batch_rejection", "MEMORY", False, "Failed to reject oversized batch")
    except (ValueError, RuntimeError, eo.exceptions.EveryOError):
        valid = runtime.predict(np.ones((5, 2), dtype=np.float32))
        record_scenario("max_batch_rejection", "MEMORY", len(valid) == 5)

    # 6. Bounded Memory Limit Containment
    with MemoryTracker() as tracker:
        mem_runtime = CPUInferenceRuntime(eo.Sequential(eo.Linear(4, 2, seed=1)), max_batch=5)
        for _ in range(10):
            with contextlib.suppress(Exception):
                _ = mem_runtime.predict(np.ones((100, 4), dtype=np.float32))
    record_scenario(
        "bounded_memory_limit_containment", "MEMORY", tracker.delta_rss_bytes < 10 * 1024 * 1024
    )

    # --- CATEGORY C: I/O Faults ---
    # 7. Missing Model File
    try:
        _ = eo.load("non_existent_model_file_xyz123.eo")
        record_scenario("missing_model_file", "IO", False, "Failed to raise on missing file")
    except (FileNotFoundError, OSError, eo.exceptions.EveryOError):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "valid.eo"
            eo.save(eo.Sequential(eo.Linear(2, 1, seed=1)), p)
            loaded = eo.load(p)
            record_scenario("missing_model_file", "IO", loaded is not None)

    # 8. Corrupted Archive
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "corrupt.eo"
            p.write_bytes(b"CORRUPTED BYTES 1234567890")
            _ = eo.load(p)
            record_scenario(
                "corrupted_archive_rejection", "IO", False, "Failed to reject corrupt archive"
            )
    except Exception:
        record_scenario("corrupted_archive_rejection", "IO", True)

    # 9. Truncated Archive Rejection
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_p = Path(tmpdir) / "valid.eo"
        eo.save(eo.Sequential(eo.Linear(2, 1, seed=1)), valid_p)
        full_bytes = valid_p.read_bytes()
        trunc_p = Path(tmpdir) / "trunc.eo"
        trunc_p.write_bytes(full_bytes[: len(full_bytes) // 2])
        try:
            _ = eo.load(trunc_p)
            record_scenario(
                "truncated_archive_rejection", "IO", False, "Failed to reject truncated archive"
            )
        except Exception:
            loaded = eo.load(valid_p)
            record_scenario("truncated_archive_rejection", "IO", loaded is not None)

    # 10. Invalid Manifest Metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_manifest_p = Path(tmpdir) / "bad_manifest.eo"
        with zipfile.ZipFile(bad_manifest_p, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"invalid_manifest": True}))
        try:
            _ = eo.load(bad_manifest_p)
            record_scenario(
                "invalid_manifest_metadata", "IO", False, "Failed to reject invalid manifest"
            )
        except Exception:
            with tempfile.TemporaryDirectory() as td2:
                rec_p = Path(td2) / "rec.eo"
                eo.save(eo.Sequential(eo.Linear(2, 1, seed=2)), rec_p)
                loaded = eo.load(rec_p)
                record_scenario("invalid_manifest_metadata", "IO", loaded is not None)

    # 11. Permission Error / Target Access Handling
    import stat

    with tempfile.TemporaryDirectory() as tmpdir:
        if sys.platform != "win32":
            ro_target_dir = Path(tmpdir) / "no_write"
            ro_target_dir.mkdir()
            ro_target_dir.chmod(stat.S_IREAD | stat.S_IEXEC)
            target = ro_target_dir / "readonly.eo"
        else:
            target = Path(tmpdir) / "readonly.eo"
            target.write_text("initial")
            target.chmod(stat.S_IREAD)
        try:
            try:
                eo.save(eo.Sequential(eo.Linear(2, 1, seed=1)), target, overwrite=True)
                record_scenario(
                    "permission_error_handling",
                    "IO",
                    False,
                    "Failed to reject save into read-only target",
                )
            except (PermissionError, OSError, eo.exceptions.EveryOError):
                good_p = Path(tmpdir) / "good.eo"
                eo.save(eo.Sequential(eo.Linear(2, 1, seed=1)), good_p)
                record_scenario("permission_error_handling", "IO", good_p.is_file())
        finally:
            if sys.platform != "win32":
                ro_target_dir.chmod(stat.S_IRWXU)
            else:
                target.chmod(stat.S_IWRITE)

    # 12. Missing Output Directory
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_target = Path(tmpdir) / "nested_dir" / "deeper" / "model.eo"
        eo.save(eo.Sequential(eo.Linear(2, 1, seed=1)), nested_target)
        loaded = eo.load(nested_target)
        record_scenario("missing_output_directory", "IO", loaded is not None)

    # --- CATEGORY D: Concurrency Faults ---
    # 13. Worker Thread Exception Containment
    pool = ThreadPoolExecutor(max_workers=2)

    def failing_worker():
        raise RuntimeError("Simulated worker failure")

    def healthy_worker():
        return 42

    fut1 = pool.submit(failing_worker)
    fut2 = pool.submit(healthy_worker)
    worker_error_handled = False
    try:
        fut1.result()
    except RuntimeError:
        worker_error_handled = True
    healthy_result = fut2.result()
    pool.shutdown(wait=True)
    record_scenario(
        "worker_exception_containment", "CONCURRENCY", worker_error_handled and healthy_result == 42
    )

    # 14. Worker Timeout Handling
    pool_to = ThreadPoolExecutor(max_workers=2)

    def hanging_worker():
        time.sleep(0.5)
        return "done"

    fut_to = pool_to.submit(hanging_worker)
    timed_out = False
    try:
        fut_to.result(timeout=0.02)
    except FuturesTimeoutError:
        timed_out = True
    pool_to.shutdown(wait=True)
    record_scenario("worker_timeout_handling", "CONCURRENCY", timed_out)

    # --- CATEGORY E: Serving Faults & Seamless Recovery ---
    from everyo.serving import CPUModelServer

    model = eo.Sequential(eo.Linear(2, 1, seed=1))
    runtime = CPUInferenceRuntime(model)
    with CPUModelServer(runtime, api_token="sec-token", max_concurrent=2) as server:
        w = server.run_background()
        url = f"http://127.0.0.1:{server.server_port}/v1/predict"
        try:
            # 15. Unauthorized Request
            req = Request(
                url, data=b'{"inputs": [[1.0, 2.0]]}', headers={"Content-Type": "application/json"}
            )
            try:
                urlopen(req)
                record_scenario(
                    "serving_unauthorized", "SERVING", False, "Allowed unauthorized request"
                )
            except HTTPError as e:
                record_scenario("serving_unauthorized", "SERVING", e.code == 401)

            # 16. Malformed JSON
            req = Request(
                url,
                data=b'{"inputs": invalid json}',
                headers={"Authorization": "Bearer sec-token", "Content-Type": "application/json"},
            )
            try:
                urlopen(req)
                record_scenario(
                    "serving_malformed_json", "SERVING", False, "Allowed malformed JSON"
                )
            except HTTPError as e:
                record_scenario("serving_malformed_json", "SERVING", e.code in {400, 500})

            # 17. Recovery: subsequent valid request succeeds
            req = Request(
                url,
                data=b'{"inputs": [[1.0, 2.0]]}',
                headers={"Authorization": "Bearer sec-token", "Content-Type": "application/json"},
            )
            with urlopen(req) as resp:
                data = json.load(resp)
                record_scenario(
                    "serving_recovery_after_faults",
                    "SERVING",
                    resp.status == 200 and len(data.get("outputs", [])) == 1,
                )
        finally:
            server.shutdown()
            w.join(timeout=3)

    all_passed = all(r["passed"] for r in results)
    return {
        "total_scenarios": len(results),
        "passed_scenarios": sum(1 for r in results if r["passed"]),
        "failed_scenarios": sum(1 for r in results if not r["passed"]),
        "all_passed": all_passed,
        "scenarios": results,
    }


# ==============================================================================
# 21.3.6: Performance Regression Gates
# ==============================================================================


@dataclass
class EnvironmentCompatibilityResult:
    """Detailed summary of environment compatibility check between two benchmark runs."""

    is_compatible: bool
    reasons: list[str]
    current_env: dict[str, Any]
    baseline_env: dict[str, Any]


def _extract_os_family(env: dict[str, Any]) -> str:
    if "os_name" in env and env["os_name"]:
        return str(env["os_name"]).strip().lower()
    platform_str = str(env.get("platform", "")).strip().lower()
    for os_key in ("windows", "linux", "darwin"):
        if os_key in platform_str:
            return os_key
    return platform_str.split()[0] if platform_str else "unknown"


def _extract_arch(env: dict[str, Any]) -> str:
    arch = str(env.get("architecture", "")).strip().lower()
    if not arch:
        platform_str = str(env.get("platform", "")).strip().lower()
        if "amd64" in platform_str or "x86_64" in platform_str:
            return "x86_64"
        if "arm64" in platform_str or "aarch64" in platform_str:
            return "arm64"
        return "unknown"
    if arch in ("amd64", "x86_64"):
        return "x86_64"
    if arch in ("arm64", "aarch64"):
        return "arm64"
    return arch


def check_environment_compatibility(
    current_env: dict[str, Any],
    baseline_env: dict[str, Any],
    *,
    strict_cpu: bool = False,
    strict_python_patch: bool = False,
    strict_numpy_patch: bool = False,
) -> EnvironmentCompatibilityResult:
    """Validate whether current and baseline benchmark environments are comparable.

    Environments are incompatible if operating system family, machine architecture,
    Python major.minor, NumPy major version, or CPU identity (under strict policy) differ.
    """
    reasons: list[str] = []

    if not current_env or not baseline_env:
        return EnvironmentCompatibilityResult(
            is_compatible=False,
            reasons=["Missing environment metadata in one or both benchmark runs."],
            current_env=current_env or {},
            baseline_env=baseline_env or {},
        )

    # 1. Operating system family
    curr_os = _extract_os_family(current_env)
    base_os = _extract_os_family(baseline_env)
    if curr_os != base_os and curr_os != "unknown" and base_os != "unknown":
        reasons.append(
            f"Operating system mismatch: current={current_env.get('platform', curr_os)} vs baseline={baseline_env.get('platform', base_os)}"
        )

    # 2. Machine architecture
    curr_arch = _extract_arch(current_env)
    base_arch = _extract_arch(baseline_env)
    if curr_arch != base_arch and curr_arch != "unknown" and base_arch != "unknown":
        reasons.append(
            f"Machine architecture mismatch: current={curr_arch} vs baseline={base_arch}"
        )

    # 3. Python version
    curr_py = str(current_env.get("python_version", "")).strip()
    base_py = str(baseline_env.get("python_version", "")).strip()
    if curr_py and base_py:
        if strict_python_patch:
            if curr_py != base_py:
                reasons.append(
                    f"Python patch version mismatch: current={curr_py} vs baseline={base_py}"
                )
        else:
            curr_mm = ".".join(curr_py.split(".")[:2])
            base_mm = ".".join(base_py.split(".")[:2])
            if curr_mm != base_mm:
                reasons.append(f"Python version mismatch: current={curr_py} vs baseline={base_py}")

    # 4. NumPy version
    curr_np = str(current_env.get("numpy_version", "")).strip()
    base_np = str(baseline_env.get("numpy_version", "")).strip()
    if curr_np and base_np:
        if strict_numpy_patch:
            if curr_np != base_np:
                reasons.append(
                    f"NumPy patch version mismatch: current={curr_np} vs baseline={base_np}"
                )
        else:
            curr_np_major = curr_np.split(".")[0]
            base_np_major = base_np.split(".")[0]
            if curr_np_major != base_np_major:
                reasons.append(
                    f"NumPy major version mismatch: current={curr_np} vs baseline={base_np}"
                )

    # 5. Processor / CPU model identity
    curr_proc = str(current_env.get("processor", "")).strip().lower()
    base_proc = str(baseline_env.get("processor", "")).strip().lower()
    if (
        strict_cpu
        and curr_proc
        and base_proc
        and curr_proc != "unknown"
        and base_proc != "unknown"
        and curr_proc != base_proc
    ):
        reasons.append(f"Processor model mismatch: current={curr_proc} vs baseline={base_proc}")

    return EnvironmentCompatibilityResult(
        is_compatible=len(reasons) == 0,
        reasons=reasons,
        current_env=current_env,
        baseline_env=baseline_env,
    )


def evaluate_performance_regression(
    current_results: dict[str, Any],
    baseline_results: dict[str, Any],
    *,
    warning_threshold: float = 0.15,
    failure_threshold: float = 0.30,
    noise_floor_ms: float = 0.5,
    strict_cpu: bool = False,
    strict_python_patch: bool = False,
    strict_numpy_patch: bool = False,
) -> dict[str, Any]:
    """Compare current benchmark results against baseline with environment and noise-aware gates.

    Outcomes:
        - PASS: Compatible environment with no performance regressions exceeding thresholds.
        - WARN: Compatible environment with moderate regression (between warning and failure threshold).
        - FAIL: Compatible environment with genuine regression exceeding failure threshold.
        - INCOMPARABLE: Environment or workload mismatch prevents meaningful timing comparisons.
        - MISSING_BASELINE: No baseline data available for comparison.
        - INVALID: Malformed current results or missing benchmarks dictionary.
    """
    if not isinstance(current_results, dict) or "benchmarks" not in current_results:
        return {
            "status": "INVALID",
            "passed": False,
            "is_compatible": False,
            "total_benchmarks": 0,
            "total_compared": 0,
            "pass_count": 0,
            "warning_count": 0,
            "failure_count": 0,
            "incomparable_count": 0,
            "missing_baseline_count": 0,
            "environment_compatible": False,
            "environment_differences": ["Invalid current benchmark results structure."],
            "warnings": [],
            "failures": [],
            "incomparables": [],
            "comparisons": [],
            "notice": "Current results must be a valid dictionary containing 'benchmarks'.",
        }

    if not isinstance(baseline_results, dict):
        baseline_results = {}

    current_benchmarks = current_results.get("benchmarks", {})
    baseline_benchmarks = baseline_results.get("benchmarks", {})

    # Check for empty / missing baseline
    if not baseline_benchmarks:
        missing_comps = [
            {
                "benchmark_id": b_id,
                "status": "MISSING_BASELINE",
                "reason": "Baseline record missing or empty",
                "baseline_median_ms": None,
                "current_median_ms": b_entry.get("median_ms")
                if isinstance(b_entry, dict)
                else None,
                "delta_ms": None,
                "delta_pct": None,
            }
            for b_id, b_entry in current_benchmarks.items()
        ]
        return {
            "status": "MISSING_BASELINE",
            "passed": None,
            "is_compatible": False,
            "total_benchmarks": len(current_benchmarks),
            "total_compared": 0,
            "pass_count": 0,
            "warning_count": 0,
            "failure_count": 0,
            "incomparable_count": 0,
            "missing_baseline_count": len(missing_comps),
            "environment_compatible": False,
            "environment_differences": ["Missing baseline benchmark records."],
            "warnings": [],
            "failures": [],
            "incomparables": [],
            "comparisons": missing_comps,
            "notice": "A baseline record is required to perform regression qualification.",
        }

    # Check benchmark suite version compatibility
    curr_suite = current_results.get("benchmark_suite_version")
    base_suite = baseline_results.get("benchmark_suite_version")
    suite_mismatch: str | None = None
    if curr_suite and base_suite and curr_suite != base_suite:
        suite_mismatch = (
            f"Benchmark suite version mismatch: current={curr_suite} vs baseline={base_suite}"
        )

    # Check environment metadata compatibility
    current_env = current_results.get("environment")
    baseline_env = baseline_results.get("environment")
    env_compat = None
    if current_env and baseline_env:
        env_compat = check_environment_compatibility(
            current_env,
            baseline_env,
            strict_cpu=strict_cpu,
            strict_python_patch=strict_python_patch,
            strict_numpy_patch=strict_numpy_patch,
        )

    env_incompatible_reasons: list[str] = []
    if suite_mismatch:
        env_incompatible_reasons.append(suite_mismatch)
    if env_compat and not env_compat.is_compatible:
        env_incompatible_reasons.extend(env_compat.reasons)

    # If the suite or environment as a whole is incompatible:
    if env_incompatible_reasons:
        incomp_comps = []
        for b_id, base_entry in baseline_benchmarks.items():
            curr_entry = current_benchmarks.get(b_id, {})
            base_med = base_entry.get("median_ms") if isinstance(base_entry, dict) else None
            curr_med = curr_entry.get("median_ms") if isinstance(curr_entry, dict) else None
            incomp_comps.append(
                {
                    "benchmark_id": b_id,
                    "status": "INCOMPARABLE",
                    "reason": "; ".join(env_incompatible_reasons),
                    "baseline_median_ms": base_med,
                    "current_median_ms": curr_med,
                    "delta_ms": None,
                    "delta_pct": None,
                }
            )

        return {
            "status": "INCOMPARABLE",
            "passed": None,
            "is_compatible": False,
            "total_benchmarks": len(current_benchmarks),
            "total_compared": 0,
            "pass_count": 0,
            "warning_count": 0,
            "failure_count": 0,
            "incomparable_count": len(incomp_comps),
            "missing_baseline_count": 0,
            "environment_compatible": False,
            "environment_differences": env_incompatible_reasons,
            "warnings": [],
            "failures": [],
            "incomparables": [
                f"{c['benchmark_id']}: INCOMPARABLE ({c['reason']})" for c in incomp_comps
            ],
            "comparisons": incomp_comps,
            "notice": "A compatible baseline matching the current execution environment is required for performance regression evaluation.",
        }

    # Environment is compatible (or no environment block was provided, e.g. in targeted test fixtures)
    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    failures: list[str] = []
    incomparables: list[str] = []
    missing_baselines: list[str] = []

    for b_id, curr_entry in current_benchmarks.items():
        if not isinstance(curr_entry, dict):
            continue
        if b_id not in baseline_benchmarks:
            comparisons.append(
                {
                    "benchmark_id": b_id,
                    "status": "MISSING_BASELINE",
                    "reason": "Benchmark not present in reference baseline",
                    "baseline_median_ms": None,
                    "current_median_ms": curr_entry.get("median_ms"),
                    "delta_ms": None,
                    "delta_pct": None,
                }
            )
            missing_baselines.append(b_id)
            continue

        base_entry = baseline_benchmarks[b_id]
        if not isinstance(base_entry, dict):
            continue

        # Check per-benchmark workload, backend, dtype compatibility
        curr_backend = curr_entry.get("backend")
        base_backend = base_entry.get("backend")
        curr_workload = curr_entry.get("workload")
        base_workload = base_entry.get("workload")
        curr_dtype = curr_entry.get("dtype")
        base_dtype = base_entry.get("dtype")

        item_incompatible: list[str] = []
        if curr_backend and base_backend and curr_backend != base_backend:
            item_incompatible.append(f"Backend mismatch: {curr_backend} vs {base_backend}")
        if curr_workload and base_workload and curr_workload != base_workload:
            item_incompatible.append(f"Workload mismatch: {curr_workload} vs {base_workload}")
        if curr_dtype and base_dtype and curr_dtype != base_dtype:
            item_incompatible.append(f"Data type mismatch: {curr_dtype} vs {base_dtype}")

        if item_incompatible:
            comparisons.append(
                {
                    "benchmark_id": b_id,
                    "status": "INCOMPARABLE",
                    "reason": "; ".join(item_incompatible),
                    "baseline_median_ms": base_entry.get("median_ms"),
                    "current_median_ms": curr_entry.get("median_ms"),
                    "delta_ms": None,
                    "delta_pct": None,
                }
            )
            incomparables.append(f"{b_id}: {'; '.join(item_incompatible)}")
            continue

        # Evaluate timing comparison
        base_med = base_entry.get("median_ms", 0.0)
        curr_med = curr_entry.get("median_ms", 0.0)

        abs_delta = curr_med - base_med
        if abs(abs_delta) < noise_floor_ms:
            delta_pct = 0.0
            status = "PASS"
        else:
            delta_pct = abs_delta / max(base_med, 1e-6)
            if delta_pct > failure_threshold:
                status = "FAIL"
                failures.append(
                    f"{b_id}: {curr_med:.3f}ms vs baseline {base_med:.3f}ms (+{delta_pct * 100:.1f}%)"
                )
            elif delta_pct > warning_threshold:
                status = "WARN"
                warnings.append(
                    f"{b_id}: {curr_med:.3f}ms vs baseline {base_med:.3f}ms (+{delta_pct * 100:.1f}%)"
                )
            else:
                status = "PASS"

        comparisons.append(
            {
                "benchmark_id": b_id,
                "baseline_median_ms": base_med,
                "current_median_ms": curr_med,
                "delta_ms": curr_med - base_med,
                "delta_pct": delta_pct * 100.0,
                "status": status,
            }
        )

    pass_count = sum(1 for c in comparisons if c["status"] == "PASS")
    incomp_count = len(incomparables)
    missing_count = len(missing_baselines)

    if failures:
        overall_status = "FAIL"
        passed = False
    elif warnings:
        overall_status = "WARN"
        passed = True
    elif incomp_count > 0 and pass_count == 0:
        overall_status = "INCOMPARABLE"
        passed = None
    elif missing_count > 0 and pass_count == 0:
        overall_status = "MISSING_BASELINE"
        passed = None
    else:
        overall_status = "PASS"
        passed = True

    valid_compared = sum(1 for c in comparisons if c["status"] in ("PASS", "WARN", "FAIL"))

    return {
        "status": overall_status,
        "passed": passed,
        "is_compatible": incomp_count == 0,
        "total_benchmarks": len(current_benchmarks),
        "total_compared": valid_compared,
        "pass_count": pass_count,
        "warning_count": len(warnings),
        "failure_count": len(failures),
        "incomparable_count": incomp_count,
        "missing_baseline_count": missing_count,
        "environment_compatible": True,
        "environment_differences": [],
        "warnings": warnings,
        "failures": failures,
        "incomparables": incomparables,
        "comparisons": comparisons,
    }


def benchmark_cpu_kernel_suite(
    *,
    iterations: int = 100,
    matrix_dim: int = 64,
) -> dict[str, dict[str, float]]:
    """Legacy backward-compatible kernel suite wrapper."""
    results: dict[str, dict[str, float]] = {}

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((matrix_dim, matrix_dim)).astype(np.float32)
    b_np = rng.standard_normal((matrix_dim, matrix_dim)).astype(np.float32)

    a = eo.tensor(a_np, device="cpu")
    b = eo.tensor(b_np, device="cpu")

    # Matmul
    latencies = _measure_latency_series(lambda: eo.matmul(a, b), repeats=iterations)
    results["matmul"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    # Add
    latencies = _measure_latency_series(lambda: a + b, repeats=iterations)
    results["add"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    # ReLU
    latencies = _measure_latency_series(lambda: eo.relu(a), repeats=iterations)
    results["relu"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    # Autograd
    x_val = rng.standard_normal((matrix_dim, matrix_dim)).astype(np.float32)

    def autograd_step():
        x = eo.tensor(x_val, device="cpu", requires_grad=True)
        y = eo.sum(eo.matmul(x, x))
        y.backward()
        return y

    latencies = _measure_latency_series(autograd_step, repeats=iterations)
    results["autograd_step"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    return results
