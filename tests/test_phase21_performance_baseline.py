"""Unit tests for Phase 21.3.1 CPU performance baselines and percentiles."""

from __future__ import annotations

import pytest

from everyo.performance_qualification import (
    BenchmarkMetric,
    benchmark_model_operations,
    benchmark_neural_computation,
    benchmark_tensor_operations,
    calculate_percentile,
    get_environment_metadata,
    run_comprehensive_cpu_benchmark,
)


class TestPercentileCalculations:
    """Verifies mathematical correctness of percentile calculations."""

    def test_percentile_ordering(self):
        values = [float(x) for x in range(1, 101)]
        p50 = calculate_percentile(values, 0.50)
        p90 = calculate_percentile(values, 0.90)
        p95 = calculate_percentile(values, 0.95)
        p99 = calculate_percentile(values, 0.99)

        assert p50 == pytest.approx(50.5)
        assert p90 == pytest.approx(90.1)
        assert p95 == pytest.approx(95.05)
        assert p99 == pytest.approx(99.01)
        assert p50 <= p90 <= p95 <= p99

    def test_single_value_and_empty(self):
        assert calculate_percentile([], 0.5) == 0.0
        assert calculate_percentile([12.5], 0.5) == 12.5
        assert calculate_percentile([12.5], 0.99) == 12.5


class TestCpuPerformanceBaselines:
    """Verifies that CPU benchmarks produce reproducible, schema-compliant metrics."""

    def test_environment_metadata(self):
        env = get_environment_metadata()
        assert "everyo_version" in env
        assert "python_version" in env
        assert "cpu_count_logical" in env
        assert env["cpu_count_logical"] >= 1
        assert "numpy_version" in env

    def test_tensor_operations_benchmark(self):
        metrics = benchmark_tensor_operations(sizes=(16, 32), warmup=1, repeats=5, seed=42)
        assert len(metrics) > 0
        for m in metrics:
            assert isinstance(m, BenchmarkMetric)
            assert m.category == "tensor_op"
            assert m.median_ms > 0
            assert m.median_ms <= m.p95_ms <= m.p99_ms
            assert m.min_ms <= m.median_ms <= m.max_ms
            assert m.throughput_per_second > 0
            assert m.numpy_baseline_median_ms is not None

    def test_neural_computation_benchmark(self):
        metrics = benchmark_neural_computation(
            batch_sizes=(8, 16), in_features=16, out_features=8, warmup=1, repeats=5, seed=42
        )
        assert len(metrics) > 0
        op_types = {m.workload.get("step") or m.workload.get("op") for m in metrics}
        assert "forward" in op_types
        assert "train_step" in op_types
        assert "softmax" in op_types

        for m in metrics:
            assert m.category == "neural_comp"
            assert m.median_ms > 0
            assert m.throughput_per_second > 0

    def test_model_operations_benchmark(self):
        metrics = benchmark_model_operations(warmup=1, repeats=3, seed=42)
        assert len(metrics) >= 3
        ids = [m.benchmark_id for m in metrics]
        assert "model_initialization_3layer" in ids
        assert "model_serialization_eo" in ids
        assert "model_deserialization_eo" in ids

    def test_comprehensive_cpu_benchmark_suite(self):
        res = run_comprehensive_cpu_benchmark(
            sizes=(16,), batch_sizes=(8,), warmup=1, repeats=3, seed=42
        )
        assert res["benchmark_suite_version"] == "21.3.1"
        assert res["total_benchmarks"] > 0
        assert "benchmarks" in res
        assert "environment" in res
        assert res["duration_seconds"] > 0
