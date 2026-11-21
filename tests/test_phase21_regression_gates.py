"""Unit tests for Phase 21.3.6 automated regression gates and noise filtering."""

from __future__ import annotations

import copy

from everyo.performance_qualification import (
    evaluate_performance_regression,
    run_comprehensive_cpu_benchmark,
)


class TestPerformanceRegressionGates:
    """Verifies noise-aware comparison, environment compatibility, baseline compliance, and failure detection."""

    def test_self_comparison_passes(self):
        benchmarks = run_comprehensive_cpu_benchmark(
            sizes=(16,), batch_sizes=(8,), warmup=1, repeats=3, seed=42
        )
        res = evaluate_performance_regression(benchmarks, benchmarks)
        assert res["status"] == "PASS"
        assert res["passed"] is True
        assert res["failure_count"] == 0
        assert res["warning_count"] == 0
        assert res["total_compared"] == benchmarks["total_benchmarks"]

    def test_identical_environments_compare_normally(self):
        env = {
            "os_name": "Linux",
            "platform": "Linux 6.8.0 (x86_64)",
            "architecture": "x86_64",
            "python_version": "3.12.0",
            "numpy_version": "2.1.0",
            "processor": "x86_64",
        }
        base = {
            "environment": env,
            "benchmark_suite_version": "21.3.1",
            "benchmarks": {
                "matmul": {
                    "median_ms": 1.0,
                    "backend": "cpu",
                    "dtype": "float32",
                    "workload": {"n": 32},
                }
            },
        }
        current = {
            "environment": env,
            "benchmark_suite_version": "21.3.1",
            "benchmarks": {
                "matmul": {
                    "median_ms": 1.05,
                    "backend": "cpu",
                    "dtype": "float32",
                    "workload": {"n": 32},
                }
            },
        }
        res = evaluate_performance_regression(current, base)
        assert res["status"] == "PASS"
        assert res["passed"] is True
        assert res["failure_count"] == 0
        assert res["warning_count"] == 0
        assert res["total_compared"] == 1

    def test_different_operating_systems_become_incomparable(self):
        base = {
            "environment": {
                "os_name": "Windows",
                "platform": "Windows 11 (AMD64)",
                "architecture": "AMD64",
                "python_version": "3.15.2",
                "numpy_version": "2.5.5",
            },
            "benchmarks": {"matmul": {"median_ms": 1.0}},
        }
        current = {
            "environment": {
                "os_name": "Linux",
                "platform": "Linux 6.8.0-1018-azure (x86_64)",
                "architecture": "x86_64",
                "python_version": "3.12.16",
                "numpy_version": "2.5.5",
            },
            "benchmarks": {"matmul": {"median_ms": 2.5}},
        }
        res = evaluate_performance_regression(current, base)
        assert res["status"] == "INCOMPARABLE"
        assert res["passed"] is None
        assert res["is_compatible"] is False
        assert res["incomparable_count"] == 1
        assert any("Operating system mismatch" in diff for diff in res["environment_differences"])
        # Incompatible comparisons must NOT calculate a misleading percentage regression
        assert res["comparisons"][0]["delta_pct"] is None
        assert res["comparisons"][0]["status"] == "INCOMPARABLE"
        assert res["failure_count"] == 0

    def test_different_cpu_identities_follow_compatibility_policy(self):
        base = {
            "environment": {
                "platform": "Linux",
                "python_version": "3.12.0",
                "numpy_version": "2.0.0",
                "processor": "intel_xeon_platinum",
            },
            "benchmarks": {"matmul": {"median_ms": 1.0}},
        }
        current = {
            "environment": {
                "platform": "Linux",
                "python_version": "3.12.0",
                "numpy_version": "2.0.0",
                "processor": "amd_epyc_7763",
            },
            "benchmarks": {"matmul": {"median_ms": 1.1}},
        }
        # Under strict CPU policy, different CPU models are INCOMPARABLE
        res_strict = evaluate_performance_regression(current, base, strict_cpu=True)
        assert res_strict["status"] == "INCOMPARABLE"
        assert res_strict["passed"] is None
        assert any(
            "Processor model mismatch" in diff for diff in res_strict["environment_differences"]
        )

        # Under default CPU policy (same OS, same Python, same architecture), timing comparison executes
        res_default = evaluate_performance_regression(current, base, strict_cpu=False)
        assert res_default["status"] == "PASS"
        assert res_default["passed"] is True

    def test_different_python_versions_follow_compatibility_policy(self):
        base = {
            "environment": {
                "platform": "Linux",
                "python_version": "3.15.2",
                "numpy_version": "2.0.0",
            },
            "benchmarks": {"matmul": {"median_ms": 1.0}},
        }
        current = {
            "environment": {
                "platform": "Linux",
                "python_version": "3.12.16",
                "numpy_version": "2.0.0",
            },
            "benchmarks": {"matmul": {"median_ms": 1.8}},
        }
        res = evaluate_performance_regression(current, base)
        assert res["status"] == "INCOMPARABLE"
        assert res["passed"] is None
        assert any("Python version mismatch" in diff for diff in res["environment_differences"])

    def test_different_numpy_versions_follow_compatibility_policy(self):
        base = {
            "environment": {
                "platform": "Linux",
                "python_version": "3.12.0",
                "numpy_version": "2.5.5",
            },
            "benchmarks": {"matmul": {"median_ms": 1.0}},
        }
        current = {
            "environment": {
                "platform": "Linux",
                "python_version": "3.12.0",
                "numpy_version": "1.26.4",
            },
            "benchmarks": {"matmul": {"median_ms": 1.0}},
        }
        res = evaluate_performance_regression(current, base)
        assert res["status"] == "INCOMPARABLE"
        assert res["passed"] is None
        assert any("NumPy major version mismatch" in diff for diff in res["environment_differences"])

    def test_different_benchmark_suite_versions_become_incomparable(self):
        base = {
            "benchmark_suite_version": "21.3.1",
            "benchmarks": {"matmul": {"median_ms": 1.0}},
        }
        current = {
            "benchmark_suite_version": "22.0.0",
            "benchmarks": {"matmul": {"median_ms": 1.0}},
        }
        res = evaluate_performance_regression(current, base)
        assert res["status"] == "INCOMPARABLE"
        assert res["passed"] is None
        assert any(
            "Benchmark suite version mismatch" in diff for diff in res["environment_differences"]
        )

    def test_different_workloads_are_never_compared(self):
        base = {
            "benchmarks": {
                "tensor_op": {"median_ms": 1.0, "workload": {"shape": [32, 32]}},
            }
        }
        current = {
            "benchmarks": {
                "tensor_op": {"median_ms": 10.0, "workload": {"shape": [128, 128]}},
            }
        }
        res = evaluate_performance_regression(current, base)
        assert res["is_compatible"] is False
        assert res["comparisons"][0]["status"] == "INCOMPARABLE"
        assert "Workload mismatch" in res["comparisons"][0]["reason"]
        assert res["comparisons"][0]["delta_pct"] is None
        assert res["failure_count"] == 0

    def test_different_data_types_are_never_compared(self):
        base = {
            "benchmarks": {
                "tensor_op": {"median_ms": 1.0, "dtype": "float32"},
            }
        }
        current = {
            "benchmarks": {
                "tensor_op": {"median_ms": 0.5, "dtype": "float16"},
            }
        }
        res = evaluate_performance_regression(current, base)
        assert res["is_compatible"] is False
        assert res["comparisons"][0]["status"] == "INCOMPARABLE"
        assert "Data type mismatch" in res["comparisons"][0]["reason"]
        assert res["comparisons"][0]["delta_pct"] is None

    def test_missing_baseline_reported_explicitly(self):
        current = {
            "benchmarks": {
                "new_op": {"median_ms": 1.5},
            }
        }
        # Empty baseline
        res_empty = evaluate_performance_regression(current, {})
        assert res_empty["status"] == "MISSING_BASELINE"
        assert res_empty["passed"] is None
        assert res_empty["missing_baseline_count"] == 1

        # Baseline with different benchmark
        base = {"benchmarks": {"old_op": {"median_ms": 1.0}}}
        res_partial = evaluate_performance_regression(current, base)
        assert res_partial["comparisons"][0]["status"] == "MISSING_BASELINE"
        assert res_partial["missing_baseline_count"] == 1

    def test_genuine_compatible_50pct_regression_still_fails(self):
        base = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 10.0},
            }
        }
        current = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 15.0},
            }
        }
        res = evaluate_performance_regression(current, base, failure_threshold=0.30)
        assert res["status"] == "FAIL"
        assert res["passed"] is False
        assert res["failure_count"] == 1
        assert "matmul_heavy" in res["failures"][0]

    def test_compatible_20pct_regression_produces_warning(self):
        base = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 10.0},
            }
        }
        current = {
            "benchmarks": {
                "matmul_heavy": {"median_ms": 12.0},
            }
        }
        res = evaluate_performance_regression(
            current, base, warning_threshold=0.15, failure_threshold=0.30
        )
        assert res["status"] == "WARN"
        assert res["passed"] is True
        assert res["warning_count"] == 1
        assert res["failure_count"] == 0
        assert "matmul_heavy" in res["warnings"][0]

    def test_small_timing_noise_handled_according_to_noise_policy(self):
        # 0.2ms delta on sub-millisecond benchmark (<0.5ms absolute noise floor)
        base = {"benchmarks": {"fast_op": {"median_ms": 0.10}}}
        current = {"benchmarks": {"fast_op": {"median_ms": 0.25}}}
        res = evaluate_performance_regression(current, base)
        assert res["status"] == "PASS"
        assert res["passed"] is True
        assert res["failure_count"] == 0
        assert res["comparisons"][0]["status"] == "PASS"
        assert res["comparisons"][0]["delta_pct"] == 0.0

    def test_incompatible_environments_never_produce_misleading_percentage_regressions(self):
        # Reproduction of CI failure: Windows baseline (0.932ms) vs Ubuntu measurement (1.940ms)
        base = {
            "environment": {
                "os_name": "Windows",
                "platform": "Windows 11 (AMD64)",
                "python_version": "3.15.2",
                "numpy_version": "2.5.5",
                "processor": "Intel64 Family 6 Model 181",
                "cpu_count_logical": 14,
            },
            "benchmarks": {
                "model_deserialization_eo": {"median_ms": 0.932},
            },
        }
        current = {
            "environment": {
                "os_name": "Linux",
                "platform": "Linux 6.8.0-1018-azure (x86_64)",
                "python_version": "3.12.16",
                "numpy_version": "2.5.5",
                "processor": "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
                "cpu_count_logical": 2,
            },
            "benchmarks": {
                "model_deserialization_eo": {"median_ms": 1.940},
            },
        }
        res = evaluate_performance_regression(current, base)
        # Must NOT fail the gate or claim +108.1% regression on incompatible environments
        assert res["status"] == "INCOMPARABLE"
        assert res["passed"] is None
        assert res["failure_count"] == 0
        assert res["comparisons"][0]["delta_pct"] is None
        assert res["comparisons"][0]["delta_ms"] is None
        assert res["comparisons"][0]["status"] == "INCOMPARABLE"

    def test_baseline_refresh_cannot_happen_silently(self):
        base = {
            "environment": {"platform": "Linux", "python_version": "3.12.0"},
            "benchmarks": {"matmul": {"median_ms": 5.0}},
        }
        base_copy = copy.deepcopy(base)
        current = {
            "environment": {"platform": "Linux", "python_version": "3.12.0"},
            "benchmarks": {"matmul": {"median_ms": 10.0}},
        }
        # Evaluation must never mutate or silently update the reference baseline object
        _ = evaluate_performance_regression(current, base)
        assert base == base_copy
        assert base["benchmarks"]["matmul"]["median_ms"] == 5.0
