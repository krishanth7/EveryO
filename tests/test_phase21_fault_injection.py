"""Unit tests for Phase 21.3.5 fault injection and exception categorization."""

from __future__ import annotations

from everyo.performance_qualification import run_fault_injection_suite


class TestFaultInjection:
    """Verifies that controlled failure conditions across categories A-F are cleanly caught and handled."""

    def test_fault_injection_suite(self):
        res = run_fault_injection_suite()
        assert res["all_passed"] is True
        assert res["total_scenarios"] >= 8
        assert res["failed_scenarios"] == 0

        categories = {s["category"] for s in res["scenarios"]}
        assert "CPU_EXECUTION" in categories
        assert "MEMORY" in categories
        assert "IO" in categories
        assert "CONCURRENCY" in categories
        assert "SERVING" in categories
