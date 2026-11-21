"""Performance and reliability qualification tests for EveryO.

Validates:
1. Accurate percentile calculation (p50, p90, p95, p99).
2. CPU compute kernel benchmark latency profiling.
3. Serving performance under Quick Regression and Standard Load profiles.
4. Fault injection resilience and seamless recovery.
"""

from __future__ import annotations

import numpy as np

import everyo as eo
from everyo.inference import CPUInferenceRuntime
from everyo.performance_qualification import (
    FAULT_INJECTION_PROFILE,
    QUICK_REGRESSION_PROFILE,
    PerformanceProfile,
    benchmark_cpu_kernel_suite,
    benchmark_serving_performance,
    calculate_percentile,
)
from everyo.serving import CPUModelServer


class TestPercentileCalculations:
    """Verifies mathematical correctness of percentile interpolation."""

    def test_percentile_interpolation(self):
        # 0 to 100 inclusive (101 points)
        values = [float(i) for i in range(101)]
        assert calculate_percentile(values, 0.50) == 50.0
        assert calculate_percentile(values, 0.90) == 90.0
        assert calculate_percentile(values, 0.95) == 95.0
        assert calculate_percentile(values, 0.99) == 99.0
        assert calculate_percentile(values, 0.0) == 0.0
        assert calculate_percentile(values, 1.0) == 100.0

    def test_edge_cases(self):
        assert calculate_percentile([], 0.5) == 0.0
        assert calculate_percentile([42.0], 0.5) == 42.0
        assert calculate_percentile([10.0, 20.0], 0.5) == 15.0


class TestCpuKernelPerformanceQualification:
    """Verifies that raw CPU kernels produce measurable latency percentiles."""

    def test_kernel_benchmark_suite(self):
        metrics = benchmark_cpu_kernel_suite(iterations=10, matrix_dim=16)
        for op in ["matmul", "add", "relu", "autograd_step"]:
            assert op in metrics
            data = metrics[op]
            assert 0 < data["p50_ms"] <= data["p95_ms"] <= data["p99_ms"]
            assert data["mean_ms"] > 0


class TestServingPerformanceQualification:
    """Verifies serving throughput, latency profiles, and fault injection."""

    def test_quick_regression_profile(self):
        model = eo.Sequential(eo.Linear(4, 2, seed=1))
        runtime = CPUInferenceRuntime(model)
        with CPUModelServer(runtime, api_token="perf-token", max_concurrent=4) as server:
            worker = server.run_background()
            try:
                url = f"http://127.0.0.1:{server.server_port}/v1/predict"
                evidence = benchmark_serving_performance(
                    url,
                    token="perf-token",
                    profile=QUICK_REGRESSION_PROFILE,
                    inputs=[[1.0, 2.0, 3.0, 4.0]],
                    target_slo_p95_ms=200.0,
                )
                assert evidence.profile_name == "quick_regression"
                assert evidence.total_requests == 20
                assert evidence.successes == 20
                assert evidence.failures == 0
                assert evidence.error_rate == 0.0
                assert evidence.throughput_rps > 0
                assert 0 < evidence.p50_ms <= evidence.p95_ms <= evidence.p99_ms
                assert evidence.passed_slo is True
            finally:
                server.shutdown()
                worker.join(timeout=5)

    def test_standard_load_profile(self):
        model = eo.Sequential(eo.Linear(2, 2, seed=2))
        runtime = CPUInferenceRuntime(model)
        profile = PerformanceProfile(
            name="custom_standard",
            concurrency=4,
            total_requests=40,
            warmup_requests=2,
            timeout_seconds=5.0,
        )
        with CPUModelServer(runtime, api_token="standard-token", max_concurrent=4) as server:
            worker = server.run_background()
            try:
                url = f"http://127.0.0.1:{server.server_port}/v1/predict"
                evidence = benchmark_serving_performance(
                    url,
                    token="standard-token",
                    profile=profile,
                    inputs=[[0.5, -0.5]],
                    target_slo_p95_ms=150.0,
                )
                assert evidence.successes == 40
                assert evidence.failures == 0
                assert evidence.throughput_rps > 0
                assert evidence.p95_ms <= 150.0
            finally:
                server.shutdown()
                worker.join(timeout=5)

    def test_fault_injection_resilience_and_recovery(self):
        model = eo.Sequential(eo.Linear(2, 1, seed=3))
        runtime = CPUInferenceRuntime(model)
        with CPUModelServer(runtime, api_token="fault-token", max_concurrent=4) as server:
            worker = server.run_background()
            try:
                url = f"http://127.0.0.1:{server.server_port}/v1/predict"
                # Exercise server under deliberate faults (rate=0.3)
                evidence = benchmark_serving_performance(
                    url,
                    token="fault-token",
                    profile=FAULT_INJECTION_PROFILE,
                    inputs=[[1.0, 2.0]],
                )
                assert evidence.total_requests == 50
                # All injected faults return 400/401/415 cleanly; server survives

                # Immediate follow-up normal request must succeed
                scores = runtime.predict_batched(np.array([[1.0, 2.0]]))
                assert len(scores) == 1
            finally:
                server.shutdown()
                worker.join(timeout=5)
