"""Unit tests for Phase 21.3.4 long-duration soak testing and degradation detection."""

from __future__ import annotations

from everyo.performance_qualification import (
    SOAK_PROFILES,
    assess_memory_stability,
    run_soak_test,
    validate_soak_report,
)


class TestSoakQualification:
    """Verifies that extended continuous execution does not suffer from resource or latency drift."""

    def test_soak_profiles_definitions(self):
        assert "short" in SOAK_PROFILES
        assert "medium" in SOAK_PROFILES
        assert "extended" in SOAK_PROFILES
        assert "endurance" in SOAK_PROFILES
        assert SOAK_PROFILES["short"] == 600.0  # 10 minutes
        assert SOAK_PROFILES["medium"] == 1800.0  # 30 minutes
        assert SOAK_PROFILES["extended"] == 7200.0  # 2 hours
        assert SOAK_PROFILES["endurance"] == 28800.0  # 8 hours

    def test_short_soak_telemetry_collection(self):
        telemetry_samples = []

        def on_sample(s):
            telemetry_samples.append(s)

        res = run_soak_test(
            duration_seconds=3.0,
            sample_interval_seconds=1.0,
            batch_size=16,
            progress_callback=on_sample,
        )

        assert res["duration_seconds"] >= 2.5
        assert res["completed_operations"] > 0
        assert res["failed_operations"] == 0
        assert res["passed_stability"] is True
        assert res["status"] == "PASS"
        assert res["total_samples"] >= 2
        assert len(telemetry_samples) >= 2

        # Check sample metrics structure
        for s in res["samples"]:
            assert "timestamp" in s
            assert "elapsed_seconds" in s
            assert "completed_ops" in s
            assert "p95_latency_ms" in s
            assert "rss_bytes" in s
            assert "python_allocated_bytes" in s
            # Non-collected python allocations must be explicit string, not null
            assert s["python_allocated_bytes"] == "NOT_COLLECTED"

        # Schema validation
        is_valid, errors = validate_soak_report(res)
        assert is_valid is True, f"Schema validation errors: {errors}"
        assert res["python_allocation_status"] == "NOT_COLLECTED"
        assert res["python_allocated_metrics"] == "NOT_COLLECTED"
        assert res["rss_trend"] in ("STABLE", "BOUNDED_GROWTH", "UNBOUNDED_GROWTH", "INCONCLUSIVE")
        assert res["memory_assessment"] in (
            "NO_LEAK_DETECTED",
            "BOUNDED_ALLOCATOR_EXPANSION_NO_CONFIRMED_LEAK",
            "INCONCLUSIVE",
        )
        assert "latency_statistics" in res
        assert "mean_p95_ms" in res["latency_statistics"]

    def test_soak_with_tracemalloc_enabled(self):
        res = run_soak_test(
            duration_seconds=2.0,
            sample_interval_seconds=1.0,
            batch_size=8,
            trace_python=True,
        )
        assert res["status"] == "PASS"
        assert res["python_allocation_status"] == "COLLECTED"
        assert isinstance(res["python_allocated_metrics"], dict)
        assert "initial_allocated_bytes" in res["python_allocated_metrics"]
        assert "growth_bytes" in res["python_allocated_metrics"]
        is_valid, errors = validate_soak_report(res)
        assert is_valid is True, f"Schema validation errors: {errors}"

    def test_invalid_soak_duration(self):
        res = run_soak_test(duration_seconds=0.0)
        assert res["status"] == "INVALID"
        assert res["passed_stability"] is False
        assert "validation_error" in res

        res_neg = run_soak_test(duration_seconds=-5.0)
        assert res_neg["status"] == "INVALID"
        assert res_neg["passed_stability"] is False

    def test_invalid_profile_name(self):
        res = run_soak_test(profile="non_existent_profile_123")
        assert res["status"] == "INVALID"
        assert res["passed_stability"] is False
        assert "validation_error" in res
        assert "Unknown profile" in res["validation_error"]

    def test_invalid_parameters(self):
        res_interval = run_soak_test(duration_seconds=5.0, sample_interval_seconds=0.0)
        assert res_interval["status"] == "INVALID"
        assert res_interval["passed_stability"] is False

        res_batch = run_soak_test(duration_seconds=5.0, batch_size=0)
        assert res_batch["status"] == "INVALID"
        assert res_batch["passed_stability"] is False

        res_concurrency = run_soak_test(duration_seconds=5.0, concurrency=-1)
        assert res_concurrency["status"] == "INVALID"
        assert res_concurrency["passed_stability"] is False

    def test_endurance_profile_execution_entrypoint(self):
        """Dedicated separately executable entrypoint for the 8-hour endurance profile.

        Run explicitly via:
            EVERYO_RUN_ENDURANCE_SOAK=1 pytest tests/test_phase21_soak.py -k test_endurance_profile_execution_entrypoint
        """
        import os

        import pytest

        if os.environ.get("EVERYO_RUN_ENDURANCE_SOAK") != "1":
            pytest.skip(
                "Pass EVERYO_RUN_ENDURANCE_SOAK=1 to execute the full 8-hour endurance soak qualification profile"
            )
        res = run_soak_test(profile="endurance", sample_interval_seconds=60.0)
        assert res["status"] == "PASS"
        assert res["duration_seconds"] >= 28800.0

    def test_assess_memory_stability_classifications(self):
        # 1. Short duration (< 60s) or few samples -> INCONCLUSIVE
        samples_short = [
            {"rss_bytes": 50000000, "python_allocated_bytes": 1000000} for _ in range(4)
        ]
        trend, assessment, is_leak = assess_memory_stability(
            samples_short, total_elapsed=10.0, init_rss=50000000, final_rss=50000000
        )
        assert trend == "INCONCLUSIVE"
        assert assessment == "INCONCLUSIVE"
        assert is_leak is False

        # 2. Stable long run (600s, 0% growth) -> STABLE / NO_LEAK_DETECTED
        samples_stable = [
            {"rss_bytes": 50000000, "python_allocated_bytes": 1000000} for _ in range(10)
        ]
        trend, assessment, is_leak = assess_memory_stability(
            samples_stable, total_elapsed=600.0, init_rss=50000000, final_rss=50000000
        )
        assert trend == "STABLE"
        assert assessment == "NO_LEAK_DETECTED"
        assert is_leak is False

        # 3. Bounded allocator expansion (warmup from 45MB to 55MB, then flat at 55MB)
        samples_bounded = [
            {"rss_bytes": 45000000, "python_allocated_bytes": 1000000},
            {"rss_bytes": 52000000, "python_allocated_bytes": 1000000},
            {"rss_bytes": 55000000, "python_allocated_bytes": 1000000},
            {"rss_bytes": 55000000, "python_allocated_bytes": 1000000},
            {"rss_bytes": 55100000, "python_allocated_bytes": 1000000},
            {"rss_bytes": 55000000, "python_allocated_bytes": 1000000},
            {"rss_bytes": 55100000, "python_allocated_bytes": 1000000},
        ]
        trend, assessment, is_leak = assess_memory_stability(
            samples_bounded, total_elapsed=600.0, init_rss=45000000, final_rss=55100000
        )
        assert trend == "BOUNDED_GROWTH"
        assert assessment == "BOUNDED_ALLOCATOR_EXPANSION_NO_CONFIRMED_LEAK"
        assert is_leak is False

        # 4. Monotonic continuing leak (>50MB growth, second half continuing growth)
        samples_leak = [
            {"rss_bytes": 50000000 + i * 20000000, "python_allocated_bytes": 1000000}
            for i in range(10)
        ]
        trend, assessment, is_leak = assess_memory_stability(
            samples_leak,
            total_elapsed=600.0,
            init_rss=50000000,
            final_rss=50000000 + 9 * 20000000,
        )
        assert trend == "UNBOUNDED_GROWTH"
        assert assessment == "CONFIRMED_MEMORY_LEAK"
        assert is_leak is True
