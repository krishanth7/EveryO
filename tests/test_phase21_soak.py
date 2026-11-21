"""Unit tests for Phase 21.3.4 long-duration soak testing and degradation detection."""

from __future__ import annotations

from everyo.performance_qualification import run_soak_test


class TestSoakQualification:
    """Verifies that extended continuous execution does not suffer from resource or latency drift."""

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
        assert res["total_samples"] >= 2
        assert len(telemetry_samples) >= 2

        # Check sample metrics structure
        for s in res["samples"]:
            assert "timestamp" in s
            assert "elapsed_seconds" in s
            assert "completed_ops" in s
            assert "p95_latency_ms" in s
            assert "rss_bytes" in s
