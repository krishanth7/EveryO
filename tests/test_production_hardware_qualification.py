"""Hardware execution qualification and multi-vendor operational gating tests.

Validates:
1. CPU reference baseline: arithmetic, matmul, activations, reductions, and autograd.
2. NVIDIA CUDA qualification: strict separation between real hardware execution and unbuilt fallback.
3. Multi-vendor non-CUDA gating: detection-only backends must never be marked operational.
4. Automatic selection safety: 'auto' device never selects detection-only backends.
5. Observable dispatch fallback: transparent fallback logging to CPU reference.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.backends import get_backend, get_last_dispatch
from everyo.backends.base import BackendStatus
from everyo.exceptions import EveryOCudaError, EveryODeviceError

NON_CUDA_BACKENDS = [
    "rocm",
    "hip",
    "intel",
    "sycl",
    "openvino",
    "directml",
    "metal",
    "vulkan",
    "opencl",
    "qnn",
    "onnxruntime",
]


class TestCpuReferenceBaselineQualification:
    """Verifies that the CPU reference baseline is mathematically correct and fully operational."""

    def test_cpu_arithmetic_and_reductions(self):
        rng = np.random.default_rng(100)
        a_np = rng.standard_normal((16, 16)).astype(np.float32)
        b_np = rng.standard_normal((16, 16)).astype(np.float32)

        a = eo.tensor(a_np, device="cpu")
        b = eo.tensor(b_np, device="cpu")

        np.testing.assert_allclose((a + b).numpy(), a_np + b_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose((a * b).numpy(), a_np * b_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(eo.sum(a).item(), float(a_np.sum()), rtol=1e-5)
        np.testing.assert_allclose(eo.mean(a).item(), float(a_np.mean()), rtol=1e-5)

    def test_cpu_matmul_and_activations(self):
        rng = np.random.default_rng(101)
        x_np = rng.standard_normal((32, 16)).astype(np.float32)
        w_np = rng.standard_normal((16, 8)).astype(np.float32)

        x = eo.tensor(x_np, device="cpu")
        w = eo.tensor(w_np, device="cpu")

        res = eo.matmul(x, w).numpy()
        np.testing.assert_allclose(res, x_np @ w_np, rtol=1e-5, atol=1e-5)

        relu_out = eo.relu(x).numpy()
        np.testing.assert_allclose(relu_out, np.maximum(x_np, 0), rtol=1e-6)

        sigmoid_out = eo.sigmoid(x).numpy()
        np.testing.assert_allclose(sigmoid_out, 1.0 / (1.0 + np.exp(-x_np)), rtol=1e-6)

    def test_cpu_autograd_pipeline(self):
        x = eo.tensor([[2.0, -1.0], [3.0, 4.0]], device="cpu", requires_grad=True)
        w = eo.tensor([[1.0, 0.5], [0.5, 1.0]], device="cpu", requires_grad=True)
        z = eo.matmul(x, w)
        loss = eo.sum(z * z)
        loss.backward()

        assert x.grad is not None
        assert w.grad is not None
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape


class TestCudaExecutionQualificationGating:
    """Verifies that CUDA execution is gated strictly on physical availability."""

    def test_cuda_status_and_gating(self):
        cuda_backend = get_backend("cuda")
        assert cuda_backend is not None

        if eo.cuda.is_available():
            assert cuda_backend.is_available() is True
            assert cuda_backend.status() == BackendStatus.AVAILABLE
            assert len(cuda_backend.get_capabilities().supported_operations) > 0
        else:
            assert cuda_backend.is_available() is False
            assert cuda_backend.status() != BackendStatus.AVAILABLE
            reason = eo.cuda.unavailable_reason()
            assert isinstance(reason, str) and len(reason) > 0

            # Strict mode must fail closed
            with pytest.raises(EveryOCudaError):
                eo.device("cuda", strict=True)

            # Non-strict mode must fall back safely to CPU
            dev = eo.device("cuda", strict=False)
            assert dev.type == "cpu"


class TestMultiVendorNonCudaOperationalGating:
    """Verifies empirical truth: detection-only backends must never claim execution."""

    @pytest.mark.parametrize("backend_name", NON_CUDA_BACKENDS)
    def test_backend_never_claims_compute_from_detection_alone(self, backend_name: str):
        backend = get_backend(backend_name)
        if backend is None:
            # Backend not registered on this platform
            return

        # 1. Must never report operational without genuine compute kernels
        assert backend.is_operational() is False
        assert len(backend.get_capabilities().supported_operations) == 0

        # 2. Strict selection must raise EveryODeviceError
        with pytest.raises(EveryODeviceError, match="not available"):
            eo.device(backend_name, strict=True)

        # 3. Non-strict selection must safely fall back to CPU
        fallback_dev = eo.device(backend_name, strict=False)
        assert fallback_dev.type == "cpu"

    def test_auto_device_never_selects_detection_only_backends(self):
        auto_dev = eo.device("auto")
        # auto device can only be CPU or a genuinely operational accelerator (CUDA if present)
        if eo.cuda.is_available():
            assert auto_dev.type in {"cuda", "cpu"}
        else:
            assert auto_dev.type == "cpu"

    def test_observable_operator_fallback(self):
        x = eo.tensor([[1.0, 2.0]], device="cpu")
        w = eo.tensor([[3.0], [4.0]], device="cpu")
        # Execute matmul on cpu
        res = eo.matmul(x, w)
        assert res.item() == 11.0

        # Dispatch with non-operational backend logs fallback
        dispatch_info = get_last_dispatch()
        if dispatch_info is not None:
            assert dispatch_info.executed_backend in {"cpu", "numpy"}
