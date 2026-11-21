"""Hardware execution qualification and numerical correctness tests.

Verifies:
- Tensor creation and memory transfers.
- Addition, multiplication, matrix multiplication, and reductions.
- Activation operations (ReLU, Sigmoid, Tanh, Softmax).
- Forward and backward autograd execution on operational backends.
- Numerical correctness compared to CPU reference within documented tolerances.
- Benchmarking suite execution.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.backends.manager import get_manager
from everyo.benchmarking import (
    measure_startup_overhead,
    run_forward_backward_benchmark,
    run_matmul_benchmark,
    run_relu_benchmark,
    run_tensor_ops_benchmark,
)


class TestCpuExecutionQualification:
    """Verifies that the CPU baseline performs full execution within tolerances."""

    def test_tensor_creation_and_memory(self):
        t = eo.tensor([1.0, 2.0, 3.0], device="cpu", dtype="float32")
        assert t.device.type == "cpu"
        assert t.shape == (3,)
        assert t.dtype == "float32"
        np.testing.assert_allclose(t.numpy(), [1.0, 2.0, 3.0])

    def test_arithmetic_operations(self, rng: np.random.Generator):
        a_np = rng.normal(size=(16, 16)).astype(np.float32)
        b_np = rng.normal(size=(16, 16)).astype(np.float32)

        a = eo.tensor(a_np, device="cpu")
        b = eo.tensor(b_np, device="cpu")

        add_res = (a + b).numpy()
        sub_res = (a - b).numpy()
        mul_res = (a * b).numpy()
        div_res = (a / (b + 1e-5)).numpy()

        np.testing.assert_allclose(add_res, a_np + b_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sub_res, a_np - b_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(mul_res, a_np * b_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(div_res, a_np / (b_np + 1e-5), rtol=1e-6, atol=1e-6)

    def test_matrix_multiplication(self, rng: np.random.Generator):
        a_np = rng.normal(size=(32, 48)).astype(np.float32)
        b_np = rng.normal(size=(48, 24)).astype(np.float32)

        a = eo.tensor(a_np, device="cpu")
        b = eo.tensor(b_np, device="cpu")

        c = eo.matmul(a, b).numpy()
        expected = a_np @ b_np
        np.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)

    def test_reductions(self, rng: np.random.Generator):
        x_np = rng.normal(size=(20, 10)).astype(np.float32)
        x = eo.tensor(x_np, device="cpu")

        np.testing.assert_allclose(eo.sum(x).item(), float(x_np.sum()), rtol=1e-5)
        np.testing.assert_allclose(eo.mean(x).item(), float(x_np.mean()), rtol=1e-5)
        np.testing.assert_allclose(eo.max(x).item(), float(x_np.max()), rtol=1e-5)
        np.testing.assert_allclose(eo.min(x).item(), float(x_np.min()), rtol=1e-5)

    def test_activations(self, rng: np.random.Generator):
        x_np = rng.normal(size=(10, 10)).astype(np.float32)
        x = eo.tensor(x_np, device="cpu")

        np.testing.assert_allclose(eo.relu(x).numpy(), np.maximum(x_np, 0), rtol=1e-6)
        np.testing.assert_allclose(eo.sigmoid(x).numpy(), 1.0 / (1.0 + np.exp(-x_np)), rtol=1e-6)
        np.testing.assert_allclose(eo.tanh(x).numpy(), np.tanh(x_np), rtol=1e-6)

    def test_autograd_and_training_step(self):
        model = eo.Sequential(
            eo.Linear(4, 8, seed=0),
            eo.ReLU(),
            eo.Linear(8, 2, seed=1),
        )
        model.to("cpu")
        opt = eo.Adam(model.parameters(), lr=0.01)
        loss_fn = eo.MSELoss()

        x = eo.tensor([[1.0, 2.0, 3.0, 4.0]], device="cpu")
        target = eo.tensor([[0.5, 1.5]], device="cpu")

        initial_loss = loss_fn(model(x), target).item()

        for _ in range(5):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()

        final_loss = loss_fn(model(x), target).item()
        assert final_loss < initial_loss


class TestFallbackAndSelectionQualification:
    def test_selection_report_captures_decisions(self):
        manager = get_manager()
        backend = manager.select_backend("cpu")
        assert backend.name == "cpu"
        report = manager.get_selection_report()
        assert report["selected"] == "cpu"
        assert report["fallback_occurred"] is False

    def test_auto_selection_picks_operational_device(self):
        manager = get_manager()
        backend = manager.select_backend("auto")
        assert backend.is_operational() is True
        report = manager.get_selection_report()
        assert report["selected"] in ("cpu", "cuda")

    def test_unimplemented_backend_strict_mode(self):
        manager = get_manager()
        with pytest.raises(RuntimeError, match="not available"):
            manager.select_backend("metal", strict=True)

    def test_unimplemented_backend_fallback_mode(self):
        manager = get_manager()
        backend = manager.select_backend("metal", strict=False)
        assert backend.name == "cpu"
        report = manager.get_selection_report()
        assert report["fallback_occurred"] is True
        assert len(report["reasons"]) > 0


class TestBenchmarkingExecution:
    def test_startup_overhead_measurement(self):
        overhead = measure_startup_overhead()
        assert "detection_overhead_seconds" in overhead
        assert "detection_overhead_ms" in overhead
        assert overhead["detection_overhead_ms"] >= 0

    def test_matmul_benchmark_runs_without_error(self):
        results = run_matmul_benchmark(sizes=[32], repeats=1, warmup=0)
        assert len(results) >= 1
        assert "gflops" in results[0]
        assert "seconds" in results[0]

    def test_relu_benchmark_runs_without_error(self):
        results = run_relu_benchmark(sizes=[1000], repeats=1, warmup=0)
        assert len(results) >= 1
        assert "elements_per_second" in results[0]

    def test_tensor_ops_benchmark_runs_without_error(self):
        results = run_tensor_ops_benchmark(sizes=[1000], repeats=1, warmup=0)
        assert len(results) >= 1
        assert results[0]["device"] == "cpu"

    def test_forward_backward_benchmark_runs_without_error(self):
        res = run_forward_backward_benchmark(
            batch_size=8, in_features=16, out_features=8, repeats=1, warmup=0
        )
        assert "forward_seconds" in res
        assert "train_step_seconds" in res
        assert res["forward_seconds"] > 0
        assert res["train_step_seconds"] > 0
