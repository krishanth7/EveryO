"""Unit tests for Phase 21.3.3 concurrency correctness and thread safety."""

from __future__ import annotations

import threading

import numpy as np

import everyo as eo
from everyo.inference import CPUInferenceRuntime
from everyo.performance_qualification import run_concurrency_correctness_qualification


class TestConcurrencyCorrectness:
    """Verifies thread-safety, absence of cross-request state leakage, and deterministic outputs."""

    def test_concurrency_correctness_qualification(self):
        res = run_concurrency_correctness_qualification(
            concurrency=4, operations_per_worker=25, seed=42
        )
        assert res["passed"] is True
        assert res["corruptions_detected"] == 0
        assert res["total_operations"] == 100
        assert res["throughput_rps"] > 0
        assert res["p50_ms"] <= res["p95_ms"] <= res["p99_ms"]

    def test_shared_runtime_concurrent_reads(self):
        model = eo.Sequential(eo.Linear(4, 2, seed=1), eo.ReLU())
        runtime = CPUInferenceRuntime(model)

        sample_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        expected = runtime.predict(sample_input)

        errors = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            for _ in range(20):
                out = runtime.predict(sample_input)
                if not np.allclose(out, expected, atol=1e-5):
                    errors.append("Output divergence under concurrent read")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
