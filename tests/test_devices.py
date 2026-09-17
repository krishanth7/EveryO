"""Tests for device handling and graceful CUDA fallback.

These tests are the guarantee behind EveryO's CPU-first promise: nothing here
requires a GPU, and every CUDA-specific test skips cleanly without one.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.core.device import Device
from everyo.exceptions import EveryOCudaError, EveryODeviceError

cuda_available = eo.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not cuda_available, reason="No working EveryO CUDA extension on this machine."
)


class TestDeviceObject:
    def test_cpu_is_the_default(self):
        assert eo.device().type == "cpu"

    def test_string_parsing(self):
        assert str(eo.device("cpu")) == "cpu"

    def test_devices_compare_by_value(self):
        assert Device("cpu", 0) == Device("cpu", 0)

    def test_unknown_device_type(self):
        with pytest.raises(EveryODeviceError, match="Unknown device type"):
            eo.device("tpu")

    def test_malformed_device_string(self):
        with pytest.raises(EveryODeviceError, match="Invalid device string"):
            eo.device("cuda:abc")

    def test_cpu_index_must_be_zero(self):
        with pytest.raises(EveryODeviceError, match="index 0"):
            Device("cpu", 1)

    def test_negative_index(self):
        with pytest.raises(EveryODeviceError):
            Device("cuda", -1)

    def test_auto_resolves_to_something_usable(self):
        resolved = eo.device("auto")
        assert resolved.type in ("cpu", "cuda")
        assert resolved.is_cuda == cuda_available


class TestCudaAvailability:
    def test_is_available_returns_a_bool(self):
        assert isinstance(eo.cuda.is_available(), bool)

    def test_probing_never_raises(self):
        eo.cuda.runtime_info()
        eo.cuda.device_count()
        eo.cuda.unavailable_reason()

    def test_runtime_info_shape(self):
        info = eo.cuda.runtime_info()
        assert {"available", "device_count", "devices"}.issubset(info)

    def test_reason_is_given_when_unavailable(self):
        if cuda_available:
            assert eo.cuda.unavailable_reason() is None
        else:
            assert isinstance(eo.cuda.unavailable_reason(), str)
            assert eo.cuda.device_count() == 0


class TestFallback:
    @pytest.mark.skipif(cuda_available, reason="This machine has CUDA.")
    def test_requesting_cuda_falls_back_to_cpu(self):
        assert eo.device("cuda").type == "cpu"

    @pytest.mark.skipif(cuda_available, reason="This machine has CUDA.")
    def test_strict_mode_raises_without_cuda(self):
        with pytest.raises(EveryOCudaError, match="strict"):
            eo.device("cuda", strict=True)

    @pytest.mark.skipif(cuda_available, reason="This machine has CUDA.")
    def test_tensors_requesting_cuda_still_work(self):
        tensor = eo.tensor([1.0, 2.0], device="cuda")
        assert tensor.device.type == "cpu"
        assert eo.sum(tensor).item() == pytest.approx(3.0)

    @pytest.mark.skipif(cuda_available, reason="This machine has CUDA.")
    def test_a_whole_model_trains_after_fallback(self, rng):
        features = rng.normal(size=(40, 3)).astype(np.float32)
        targets = features.sum(axis=1, keepdims=True)
        model = eo.Sequential(eo.Linear(3, 4, seed=0), eo.ReLU(), eo.Linear(4, 1, seed=1))
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.05), eo.MSELoss(), device="cuda"
        )
        history = trainer.fit(
            eo.DataLoader(eo.ArrayDataset(features, targets), batch_size=10),
            epochs=2,
            verbose=False,
        )
        assert history.epochs == 2
        assert trainer.device.type == "cpu"

    def test_cuda_interface_falls_back_transparently(self, rng):
        a = rng.normal(size=(8,)).astype(np.float32)
        b = rng.normal(size=(8,)).astype(np.float32)
        np.testing.assert_allclose(eo.cuda.add(a, b), a + b, rtol=1e-6)
        np.testing.assert_allclose(eo.cuda.relu(a), np.maximum(a, 0), rtol=1e-6)

    @pytest.mark.skipif(cuda_available, reason="This machine has CUDA.")
    def test_strict_kernels_raise_without_cuda(self, rng):
        values = rng.normal(size=(4,)).astype(np.float32)
        with pytest.raises(EveryOCudaError):
            eo.cuda.relu(values, strict=True)


class TestTensorPlacement:
    def test_to_changes_the_device(self):
        assert eo.tensor([1.0]).to("cpu").device.type == "cpu"

    def test_cpu_helper(self):
        assert eo.tensor([1.0]).cpu().device.type == "cpu"

    def test_cuda_helper_never_crashes(self):
        assert eo.tensor([1.0]).cuda().device.type in ("cpu", "cuda")

    def test_requires_grad_survives_a_move(self):
        tensor = eo.tensor([1.0], requires_grad=True)
        assert tensor.to("cpu").requires_grad


@requires_cuda
class TestCudaKernels:
    """Executed only on machines with a compiled extension and a GPU."""

    def test_vector_add_matches_numpy(self, rng):
        a = rng.normal(size=(1024,)).astype(np.float32)
        b = rng.normal(size=(1024,)).astype(np.float32)
        np.testing.assert_allclose(eo.cuda.add(a, b, strict=True), a + b, rtol=1e-5, atol=1e-6)

    def test_multiply_matches_numpy(self, rng):
        a = rng.normal(size=(999,)).astype(np.float32)
        b = rng.normal(size=(999,)).astype(np.float32)
        np.testing.assert_allclose(eo.cuda.multiply(a, b, strict=True), a * b, rtol=1e-5, atol=1e-6)

    def test_relu_matches_numpy(self, rng):
        values = rng.normal(size=(4097,)).astype(np.float32)
        np.testing.assert_allclose(
            eo.cuda.relu(values, strict=True), np.maximum(values, 0), rtol=1e-6
        )

    def test_matmul_matches_numpy(self, rng):
        a = rng.normal(size=(67, 43)).astype(np.float32)
        b = rng.normal(size=(43, 51)).astype(np.float32)
        np.testing.assert_allclose(eo.cuda.matmul(a, b, strict=True), a @ b, rtol=1e-3, atol=1e-4)

    def test_sum_matches_numpy(self, rng):
        values = rng.normal(size=(100_003,)).astype(np.float32)
        assert float(eo.cuda.sum_all(values, strict=True)) == pytest.approx(
            float(values.sum()), rel=1e-4
        )

    def test_tensor_operations_on_cuda_match_cpu(self, rng):
        a = rng.normal(size=(64, 64)).astype(np.float32)
        b = rng.normal(size=(64, 64)).astype(np.float32)
        on_cpu = eo.matmul(eo.tensor(a), eo.tensor(b)).numpy()
        on_gpu = eo.matmul(eo.tensor(a, device="cuda"), eo.tensor(b, device="cuda")).numpy()
        np.testing.assert_allclose(on_gpu, on_cpu, rtol=1e-3, atol=1e-4)
