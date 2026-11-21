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
    "measure_startup_overhead",
    "run_tensor_ops_benchmark",
    "run_forward_backward_benchmark",
    "run_hardware_benchmark_suite",
]

_LOGGER = get_logger(__name__)


def environment() -> dict[str, Any]:
    """Describe the machine the benchmark ran on."""
    from everyo.backends import tensorflow_backend
    from everyo.backends.detection import detect_cpu_info, detect_system_memory
    from everyo.cuda.availability import is_available as cuda_available
    from everyo.version import __version__

    return {
        "everyo_version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or "unknown",
        "cpu_info": detect_cpu_info(),
        "system_memory": detect_system_memory(),
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


def measure_startup_overhead() -> dict[str, float]:
    """Measure device detection and backend resolution overhead in seconds."""
    from everyo.backends.registry import detect_all, get_available_devices

    def probe() -> None:
        detect_all()
        get_available_devices()

    timing = time_callable(probe, repeats=5, warmup=1)
    return {
        "detection_overhead_seconds": timing["seconds"],
        "detection_overhead_ms": timing["seconds"] * 1000.0,
        "best_ms": timing["best"] * 1000.0,
    }


def run_tensor_ops_benchmark(
    *,
    sizes: Sequence[int] = (10_000, 100_000, 1_000_000),
    repeats: int = 3,
    warmup: int = 1,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Benchmark basic tensor elementwise addition across available devices."""
    import everyo as eo

    rng = np.random.default_rng(seed)
    devices = ["cpu"]
    if eo.cuda.is_available():
        devices.append("cuda")

    results: list[dict[str, Any]] = []
    for size in sizes:
        a_np = rng.normal(size=int(size)).astype(np.float32)
        b_np = rng.normal(size=int(size)).astype(np.float32)
        for dev_name in devices:
            a = eo.tensor(a_np, device=dev_name)
            b = eo.tensor(b_np, device=dev_name)

            def op(x=a, y=b) -> Any:
                return x + y

            timing = time_callable(op, repeats=repeats, warmup=warmup)
            results.append(
                {
                    "benchmark": "tensor_add",
                    "device": dev_name,
                    "size": int(size),
                    "elements_per_second": float(size) / timing["seconds"],
                    **timing,
                }
            )
    return results


def run_forward_backward_benchmark(
    *,
    batch_size: int = 64,
    in_features: int = 128,
    out_features: int = 64,
    repeats: int = 3,
    warmup: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Benchmark a forward pass and backward pass on a Linear layer."""
    import everyo as eo

    rng = np.random.default_rng(seed)
    x_np = rng.normal(size=(batch_size, in_features)).astype(np.float32)

    layer = eo.Linear(in_features, out_features, seed=seed)
    x = eo.tensor(x_np, requires_grad=True)

    def forward_pass() -> Any:
        return layer(x)

    def train_step() -> Any:
        layer.zero_grad()
        out = layer(x)
        loss = eo.sum(out)
        loss.backward()
        return loss

    fwd_timing = time_callable(forward_pass, repeats=repeats, warmup=warmup)
    train_timing = time_callable(train_step, repeats=repeats, warmup=warmup)

    return {
        "batch_size": batch_size,
        "in_features": in_features,
        "out_features": out_features,
        "forward_seconds": fwd_timing["seconds"],
        "forward_best": fwd_timing["best"],
        "train_step_seconds": train_timing["seconds"],
        "train_step_best": train_timing["best"],
    }


def run_hardware_benchmark_suite(
    *,
    matmul_sizes: Sequence[int] = (128, 256),
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Execute the full reproducible hardware benchmark suite."""
    return {
        "environment": environment(),
        "startup_overhead": measure_startup_overhead(),
        "matmul": run_matmul_benchmark(sizes=matmul_sizes, repeats=repeats, seed=seed),
        "relu": run_relu_benchmark(sizes=[100_000, 500_000], repeats=repeats, seed=seed),
        "tensor_ops": run_tensor_ops_benchmark(
            sizes=[100_000, 500_000], repeats=repeats, seed=seed
        ),
        "forward_backward": run_forward_backward_benchmark(repeats=repeats, seed=seed),
    }


def parse_args(argv: Sequence[str] | None = None) -> Any:
    """Parse command line arguments for the benchmarking CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="everyo.benchmarking",
        description="EveryO performance, stress, soak and qualification benchmark runner.",
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "full", "memory", "concurrency", "soak", "fault-recovery"],
        default="quick",
        help="Benchmark execution profile (default: quick).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds for soak profile (default: 30.0s).",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare execution results against baseline record.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="benchmarks/baselines/cpu_baseline.json",
        help="Path to reference baseline JSON for regression comparison.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path to write machine-readable JSON results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON directly to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute benchmarking profiles and output structured evidence."""
    import json
    from pathlib import Path

    from everyo.performance_qualification import (
        evaluate_performance_regression,
        run_comprehensive_cpu_benchmark,
        run_concurrency_correctness_qualification,
        run_fault_injection_suite,
        run_soak_test,
        run_stress_qualification,
        verify_resource_lifecycle,
    )

    args = parse_args(argv)
    profile = args.profile.lower().strip()
    result_data: dict[str, Any] = {}
    exit_code = 0

    if profile == "quick":
        result_data = run_comprehensive_cpu_benchmark(
            sizes=(32, 128), batch_sizes=(16, 64), repeats=5, warmup=2
        )
    elif profile == "standard":
        result_data = run_comprehensive_cpu_benchmark(
            sizes=(32, 128, 256), batch_sizes=(16, 64), repeats=10, warmup=3
        )
    elif profile == "full":
        result_data = run_comprehensive_cpu_benchmark(
            sizes=(32, 128, 256, 512), batch_sizes=(16, 64, 128), repeats=15, warmup=3
        )
    elif profile == "memory":
        result_data = verify_resource_lifecycle(iterations=20)
    elif profile == "concurrency":
        c_res = run_concurrency_correctness_qualification(concurrency=4, operations_per_worker=50)
        s_res = run_stress_qualification()
        result_data = {
            "profile": "concurrency",
            "concurrency_correctness": c_res,
            "stress_qualification": s_res,
            "passed": c_res["passed"] and s_res["passed"],
        }
        if not result_data["passed"]:
            exit_code = 1
    elif profile == "soak":
        result_data = run_soak_test(
            duration_seconds=args.duration, sample_interval_seconds=max(1.0, args.duration / 15)
        )
        if not result_data["passed_stability"]:
            exit_code = 1
    elif profile == "fault-recovery":
        result_data = run_fault_injection_suite()
        if not result_data["all_passed"]:
            exit_code = 1

    # Optional baseline regression check
    if args.compare_baseline:
        base_path = Path(args.baseline)
        if base_path.is_file():
            base_data = json.loads(base_path.read_text(encoding="utf-8"))
            comparison = evaluate_performance_regression(result_data, base_data)
            result_data["regression_comparison"] = comparison
            if not comparison["passed"]:
                exit_code = 1
        else:
            print(
                f"Warning: Baseline file {base_path} not found; skipping comparison.",
                file=sys.stderr,
            )

    # Output handling
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
        print(f"Results written to {out_p}")

    if args.json:
        print(json.dumps(result_data, indent=2))
    else:
        print(
            f"\n==================== EveryO Benchmark Report ({profile.upper()}) ===================="
        )
        if "duration_seconds" in result_data:
            print(f"Duration: {result_data['duration_seconds']:.2f}s")
        if "benchmarks" in result_data:
            benchmarks = result_data["benchmarks"]
            print(
                f"\n{'Benchmark ID':<30} {'Median (ms)':<14} {'p95 (ms)':<12} {'Throughput (ops/s)':<20}"
            )
            print("-" * 78)
            for b_id, b_val in benchmarks.items():
                print(
                    f"{b_id:<30} {b_val['median_ms']:<14.4f} {b_val['p95_ms']:<12.4f} {b_val['throughput_per_second']:<20.1f}"
                )
        if "checks" in result_data:
            print("\nLifecycle Checks:")
            for c_name, c_val in result_data["checks"].items():
                status = "PASS" if c_val.get("passed") else "FAIL"
                print(f"  [{status}] {c_name}: delta_rss={c_val.get('delta_rss_bytes', 0)} bytes")
        if "concurrency_correctness" in result_data:
            c = result_data["concurrency_correctness"]
            s = result_data["stress_qualification"]
            print(f"\nConcurrency Correctness: {'PASS' if c['passed'] else 'FAIL'}")
            print(
                f"  Workers: {c['concurrency']}, Total Ops: {c['total_operations']}, RPS: {c['throughput_rps']:.1f}"
            )
            print(
                f"  p50: {c['p50_ms']:.4f}ms, p95: {c['p95_ms']:.4f}ms, p99: {c['p99_ms']:.4f}ms, Corruptions: {c['corruptions_detected']}"
            )
            print(f"\nStress Qualification: {'PASS' if s['passed'] else 'FAIL'}")
            print(
                f"  Requests: {s['total_requests']}, RPS: {s['throughput_rps']:.1f}, Error Rate: {s['error_rate'] * 100:.2f}%"
            )
            print(f"  p50: {s['p50_ms']:.4f}ms, p95: {s['p95_ms']:.4f}ms, p99: {s['p99_ms']:.4f}ms")
        if "scenarios" in result_data:
            print(
                f"\nFault Scenarios ({result_data['passed_scenarios']}/{result_data['total_scenarios']} passed):"
            )
            for s in result_data["scenarios"]:
                status = "PASS" if s["passed"] else "FAIL"
                print(f"  [{status}] ({s['category']}) {s['name']}: {s['detail']}")
        if "passed_stability" in result_data:
            print(f"\nSoak Stability: {'PASS' if result_data['passed_stability'] else 'FAIL'}")
            print(
                f"  Completed Ops: {result_data['completed_operations']}, Overall RPS: {result_data['overall_throughput_rps']:.1f}"
            )
            print(
                f"  RSS Growth: {result_data['rss_growth_pct']:.2f}%, Latency Drift: {result_data['latency_drift_pct']:.2f}%"
            )
        if "regression_comparison" in result_data:
            comp = result_data["regression_comparison"]
            print(
                f"\nRegression Comparison: {'PASS' if comp['passed'] else 'FAIL'} ({comp['warning_count']} warnings, {comp['failure_count']} failures)"
            )
            for f in comp["failures"]:
                print(f"  [FAIL] {f}")
            for w in comp["warnings"]:
                print(f"  [WARN] {w}")
        print("==============================================================================\n")

    return exit_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
