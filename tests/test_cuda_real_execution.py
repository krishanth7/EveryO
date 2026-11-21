"""Physical NVIDIA CUDA hardware execution tests.

CRITICAL DISTINCTION:
- This file contains REAL hardware integration tests that execute exclusively
  when an actual NVIDIA GPU and the compiled native extension ('everyo._everyo_cuda')
  are present on the system.
- It is NEVER mocked and NEVER claims hardware support when physical hardware is absent.
- Boundary transfer count testing in the absence of a GPU is handled separately
  in 'tests/test_cuda_resident_transfers.py'.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.backends import numpy_backend
from everyo.backends.base import BackendStatus
from everyo.exceptions import EveryOCudaError

_CUDA_AVAILABLE = eo.cuda.is_available()
_SKIP_REASON = "Physical NVIDIA GPU and compiled EveryO CUDA extension required."


@pytest.mark.cuda
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason=_SKIP_REASON)
class TestCudaRealHardwareExecution:
    """Verifies all native CUDA kernels against CPU reference on real hardware."""

    def test_cuda_device_presence_and_properties(self):
        backend = eo.backends.get_backend("cuda")
        assert backend is not None
        assert backend.is_available() is True
        assert backend.status() == BackendStatus.AVAILABLE
        assert backend.is_operational() is True

        devices = backend.detect_devices()
        assert len(devices) >= 1
        props = backend.get_device_properties(0)
        assert "name" in props

    def test_cuda_vector_add_matches_cpu(self):
        rng = np.random.default_rng(42)
        for shape in [(100,), (32, 64), (16, 16, 16)]:
            a = rng.standard_normal(shape).astype(np.float32)
            b = rng.standard_normal(shape).astype(np.float32)
            gpu_res = eo.cuda.add(a, b, strict=True)
            cpu_res = numpy_backend.add(a, b)
            np.testing.assert_allclose(gpu_res, cpu_res, rtol=1e-5, atol=1e-6)

    def test_cuda_vector_multiply_matches_cpu(self):
        rng = np.random.default_rng(43)
        for shape in [(100,), (32, 64), (8, 8, 8)]:
            a = rng.standard_normal(shape).astype(np.float32)
            b = rng.standard_normal(shape).astype(np.float32)
            gpu_res = eo.cuda.multiply(a, b, strict=True)
            cpu_res = numpy_backend.multiply(a, b)
            np.testing.assert_allclose(gpu_res, cpu_res, rtol=1e-5, atol=1e-6)

    def test_cuda_matmul_matches_cpu(self):
        rng = np.random.default_rng(44)
        for m, k, n in [(16, 32, 16), (64, 64, 64), (128, 64, 128)]:
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n)).astype(np.float32)
            gpu_res = eo.cuda.matmul(a, b, strict=True)
            cpu_res = numpy_backend.matmul(a, b)
            np.testing.assert_allclose(gpu_res, cpu_res, rtol=1e-4, atol=1e-5)

    def test_cuda_relu_matches_cpu(self):
        rng = np.random.default_rng(45)
        x = rng.standard_normal((100, 50)).astype(np.float32)
        gpu_res = eo.cuda.relu(x, strict=True)
        cpu_res = numpy_backend.relu(x)
        np.testing.assert_allclose(gpu_res, cpu_res, rtol=1e-5, atol=1e-6)

    def test_cuda_sum_all_matches_cpu(self):
        rng = np.random.default_rng(46)
        x = rng.standard_normal((50, 50)).astype(np.float32)
        gpu_res = eo.cuda.sum_all(x, strict=True)
        cpu_res = numpy_backend.sum_all(x)
        assert float(gpu_res) == pytest.approx(float(cpu_res), rel=1e-4)

    def test_cuda_resident_lifecycle(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        b = np.eye(4, dtype=np.float32)
        da = eo.cuda.to_device(a)
        db = eo.cuda.to_device(b)
        result = ((da @ db) + da).relu()
        assert result.device == "cuda"
        np.testing.assert_allclose(result.numpy(), np.maximum((a @ b) + a, 0), rtol=1e-5)
        assert result.sum() == pytest.approx(float(np.maximum((a @ b) + a, 0).sum()))

    def test_cuda_tensor_autograd(self):
        dev = eo.device("cuda")
        x = eo.tensor([[2.0, 3.0], [4.0, 5.0]], device=dev, requires_grad=True)
        y = eo.sum(x * x)
        y.backward()
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [[4.0, 6.0], [8.0, 10.0]])

    def test_cuda_synchronize_and_release(self):
        backend = eo.backends.get_backend("cuda")
        assert backend is not None
        backend.synchronize(0)
        backend.release(0)

    def test_cuda_strict_dtype_rejection(self):
        a_f64 = np.array([1.0, 2.0], dtype=np.float64)
        b_f64 = np.array([3.0, 4.0], dtype=np.float64)
        with pytest.raises(EveryOCudaError, match="float32 only"):
            eo.cuda.add(a_f64, b_f64, strict=True)

    def test_cuda_mismatched_shape_fallback(self):
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 1), dtype=np.float32)
        # Broadcasting is not implemented in CUDA kernel; in strict=False, falls back to CPU
        res = eo.cuda.add(a, b, strict=False)
        np.testing.assert_array_equal(res, a + b)
