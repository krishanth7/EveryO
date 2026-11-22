"""Unit tests for Phase 21.3.3 concurrency correctness and thread safety."""

from __future__ import annotations

import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import everyo as eo
from everyo.inference import CPUInferenceRuntime
from everyo.performance_qualification import run_concurrency_correctness_qualification


def _process_worker(seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = eo.tensor(rng.standard_normal((8, 8)).astype(np.float32))
    b = eo.tensor(rng.standard_normal((8, 8)).astype(np.float32))
    res = eo.matmul(a, b)
    return float(np.sum(res.data))


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

    def test_concurrent_model_creation_and_destruction(self):
        barrier = threading.Barrier(4)
        errors = []

        def worker(wid: int):
            barrier.wait()
            try:
                for i in range(10):
                    m = eo.Sequential(
                        eo.Linear(16, 8, seed=wid * 100 + i),
                        eo.ReLU(),
                        eo.Linear(8, 2, seed=wid * 100 + i + 1),
                    )
                    x = eo.tensor(np.ones((2, 16), dtype=np.float32))
                    out = m(x)
                    assert out.shape == (2, 2)
                    del m, x, out
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0

    def test_concurrent_serialization_deserialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = eo.Sequential(eo.Linear(8, 4, seed=42), eo.ReLU(), eo.Linear(4, 2, seed=43))
            base_file = Path(tmpdir) / "shared.eo"
            eo.save(model, base_file)

            errors = []
            barrier = threading.Barrier(4)

            def worker(wid: int):
                barrier.wait()
                try:
                    for i in range(5):
                        # Concurrent load
                        loaded = eo.load(base_file)
                        x = eo.tensor(np.ones((1, 8), dtype=np.float32))
                        _ = loaded(x)
                        # Concurrent save to worker-specific file
                        w_file = Path(tmpdir) / f"w_{wid}_{i}.eo"
                        eo.save(loaded, w_file)
                        del loaded
                except Exception as e:
                    errors.append(str(e))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

            assert len(errors) == 0

    def test_weights_immutability_under_concurrent_inference(self):
        model = eo.Sequential(eo.Linear(4, 2, seed=99), eo.ReLU())
        init_weights = model[0].weight.data.copy()
        runtime = CPUInferenceRuntime(model)

        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            rng = np.random.default_rng()
            for _ in range(15):
                inp = rng.standard_normal((4, 4)).astype(np.float32)
                _ = runtime.predict(inp)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert np.array_equal(model[0].weight.data, init_weights)

    def test_process_pool_execution(self):
        with ProcessPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_process_worker, [10, 20, 30, 40]))
        assert len(results) == 4
        assert all(isinstance(r, float) for r in results)

    def test_worker_exception_propagation(self):
        from concurrent.futures import ThreadPoolExecutor

        def faulty_worker():
            raise ValueError("Deliberate worker computational error")

        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(faulty_worker)
            try:
                future.result()
                raised = False
            except ValueError as e:
                raised = True
                assert "Deliberate worker computational error" in str(e)
            assert raised is True

    def test_thread_timeout_detection_not_marked_terminated(self):
        import time

        stop_event = threading.Event()

        def hanging_worker():
            while not stop_event.is_set():
                time.sleep(0.01)

        t = threading.Thread(target=hanging_worker)
        t.start()
        # Verify that join with timeout detects active non-terminated thread
        t.join(timeout=0.05)
        is_still_alive = t.is_alive()
        # Cleanly signal worker to terminate
        stop_event.set()
        t.join(timeout=2.0)

        assert is_still_alive is True, "Timed-out thread must not be treated as terminated"
        assert t.is_alive() is False, "Worker must terminate cleanly after event set"

    def test_resource_cleanup_after_concurrency_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = eo.Sequential(eo.Linear(4, 2, seed=1))
            save_path = Path(tmpdir) / "concurrent_fail.eo"
            eo.save(model, save_path)

            corrupted_path = Path(tmpdir) / "corrupt.eo"
            corrupted_path.write_bytes(b"INVALID_ZIP")

            barrier = threading.Barrier(3)
            results = []

            def worker(path_to_load, expect_error):
                barrier.wait()
                try:
                    loaded = eo.load(path_to_load)
                    results.append(("success", loaded is not None))
                except Exception as exc:
                    results.append(("error", type(exc).__name__))

            t1 = threading.Thread(target=worker, args=(corrupted_path, True))
            t2 = threading.Thread(target=worker, args=(save_path, False))
            t3 = threading.Thread(target=worker, args=(save_path, False))

            for t in (t1, t2, t3):
                t.start()
            for t in (t1, t2, t3):
                t.join(timeout=5.0)

            statuses = {r[0] for r in results}
            assert "error" in statuses
            assert "success" in statuses
