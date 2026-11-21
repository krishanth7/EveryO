"""Unit tests for Phase 21.3.3 stress profiles and queue stability."""

from __future__ import annotations

import everyo as eo
from everyo.inference import CPUInferenceRuntime
from everyo.performance_qualification import (
    STRESS_PROFILE_A_NORMAL,
    PerformanceProfile,
    benchmark_serving_performance,
    run_stress_qualification,
)
from everyo.serving import CPUModelServer


class TestStressQualification:
    """Verifies framework resilience under high concurrent operation volumes."""

    def test_stress_profile_a(self):
        res = run_stress_qualification(profile=STRESS_PROFILE_A_NORMAL)
        assert res["passed"] is True
        assert res["failures"] == 0
        assert res["error_rate"] == 0.0
        assert res["total_requests"] == 100
        assert res["throughput_rps"] > 0

    def test_stress_profile_b(self):
        profile = PerformanceProfile(
            name="test_stress_b",
            concurrency=4,
            total_requests=150,
            warmup_requests=5,
            timeout_seconds=5.0,
        )
        res = run_stress_qualification(profile=profile)
        assert res["passed"] is True
        assert res["failures"] == 0
        assert res["throughput_rps"] > 0

    def test_serving_concurrency_stress(self):
        model = eo.Sequential(eo.Linear(4, 2, seed=1))
        runtime = CPUInferenceRuntime(model)
        profile = PerformanceProfile(
            name="serving_stress",
            concurrency=4,
            total_requests=40,
            warmup_requests=2,
            timeout_seconds=5.0,
        )
        with CPUModelServer(runtime, api_token="stress-token", max_concurrent=4) as server:
            w = server.run_background()
            try:
                url = f"http://127.0.0.1:{server.server_port}/v1/predict"
                evidence = benchmark_serving_performance(
                    url,
                    token="stress-token",
                    profile=profile,
                    inputs=[[1.0, 2.0, 3.0, 4.0]],
                    target_slo_p95_ms=250.0,
                )
                assert evidence.failures == 0
                assert evidence.error_rate == 0.0
                assert evidence.throughput_rps > 0
                assert evidence.passed_slo is True
            finally:
                server.shutdown()
                w.join(timeout=3)
