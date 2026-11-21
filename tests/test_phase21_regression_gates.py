"""Unit tests for Phase 21.3.6 automated regression gates and noise filtering."""

from __future__ import annotations

from everyo.performance_qualification import (
    evaluate_performance_regression,
    run_comprehensive_cpu_benchmark,
)


class TestPerformanceRegressionGates:
    """Verifies noise-aware comparison, baseline compliance, and failure detection."""

    def test_self_comparison_passes(self):
        benchmarks = run_comprehensive_cpu_benchmark(
            sizes=(16,), batch_sizes=(8,), warmup=1, repeats=3, seed=42
        )
        res = evaluate_performance_regression(benchmarks, benchmarks)
        assert res["passed"] is True
        assert res["failure_count"] == 0
        assert res["warning_count"] == 0
        assert res["total_compared"] == benchmarks["total_benchmarks"]

    def test_sub_millisecond_noise_filtering(self):
        # 0.2ms jitter on a sub-millisecond benchmark must be filtered as noise
        base = {
            "benchmarks": {
                "matmul_fast": {"median_ms": 0.10},
            }
        }
        current = {
            "benchmarks": {
                "matmul_fast": {"median_ms": 0.25},  # Delta is 0.15ms (<0.5ms threshold)
            }
        }
        res = evaluate_performance_regression(current, base)
        assert res["passed"] is True
        assert res["failure_count"] == 0
        assert res["comparisons"][0]["status"] == "PASS"

    def test_significant_regression_triggers_failure(self):
        base = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 10.0},
            }
        }
        current = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 15.0},  # +50% regression (>30% failure threshold)
            }
        }
        res = evaluate_performance_regression(current, base, failure_threshold=0.30)
        assert res["passed"] is False
        assert res["failure_count"] == 1
        assert "matmul_heavy" in res["failures"][0]

    def test_moderate_regression_triggers_warning(self):
        base = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 10.0},
            }
        }
        current = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 12.0},  # +20% regression (between 15% and 30%)
            }
        }
        res = evaluate_performance_regression(
            current, base, warning_threshold=0.15, failure_threshold=0.30
        )
        assert res["passed"] is True
        assert res["warning_count"] == 1
        assert "matmul_heavy" in res["warnings"][0]
