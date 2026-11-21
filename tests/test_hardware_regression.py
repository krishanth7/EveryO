"""Regression test suite for EveryO hardware qualification and fallback guarantees.

Verifies:
1. False-positive device detection (detection never conflated with compute capability).
2. Detection-only backend rejection in strict and automatic selection.
3. Unsupported operation fallback with observable logging and DispatchResult.
4. Unsupported dtype fallback and reason reporting.
5. Strict selection errors across all backend specs.
6. Device memory reporting when unsupported.
7. Backend initialization failures handled gracefully.
8. Resource cleanup and synchronization calls.
9. Pure CPU-only installation numerical and architectural integrity.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import numpy as np
import pytest

import everyo as eo
from everyo.backends.base import BackendStatus
from everyo.backends.manager import get_manager
from everyo.backends.operator_backend import (
    CudaOperatorBackend,
    backend_for,
    clear_dispatch_history,
    dispatch,
    get_last_dispatch,
)
from everyo.core.device import Device
from everyo.exceptions import EveryOCudaError, EveryODeviceError


class TestHardwareQualificationRegression:
    """Regression test suite covering the 9 qualification scenarios."""

    # 1. False-positive device detection
    def test_false_positive_device_detection(self):
        """Even if tools like vulkaninfo exist, detection-only backends report NOT_IMPLEMENTED."""
        detection_only = [
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
        for name in detection_only:
            backend = eo.backends.get_backend(name)
            assert backend is not None, f"Backend {name} must be registered."
            assert backend.is_available() is False, f"Backend {name} must not be marked available."
            assert backend.is_operational() is False, f"Backend {name} must not be operational."
            assert backend.status() in (
                BackendStatus.NOT_IMPLEMENTED,
                BackendStatus.UNAVAILABLE,
                BackendStatus.NOT_INSTALLED,
                BackendStatus.RUNTIME_MISSING,
            )

        # Ensure auto device never selects any detection-only backend
        dev = eo.device("auto")
        assert dev.type in ("cpu", "cuda")

    # 2. Detection-only backend rejection
    def test_detection_only_backend_rejection(self):
        """Strict selection of detection-only backends must raise explicit errors."""
        manager = get_manager()
        for name in ("vulkan", "rocm", "directml", "metal", "intel", "opencl", "qnn"):
            # Device strict mode
            with pytest.raises(EveryODeviceError, match=f"Backend '{name}'"):
                eo.device(name, strict=True)

            # Manager strict mode
            with pytest.raises(RuntimeError, match=f"Backend '{name}'"):
                manager.select_backend(name, strict=True)

            # Non-strict mode safely falls back to CPU
            fallback_dev = eo.device(name, strict=False)
            assert fallback_dev.type == "cpu"

            fallback_backend = manager.select_backend(name, strict=False)
            assert fallback_backend.name == "cpu"

    # 3. Unsupported operation fallback
    def test_unsupported_operation_fallback(self):
        """Unsupported operations route safely to CPU and populate observable DispatchResult."""
        clear_dispatch_history()
        dev = Device("vulkan", 0)  # construct device directly

        # Issue an operation
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        res = dispatch("add", dev, a, b)

        assert res.fell_back is True
        assert res.requested_backend == "vulkan"
        assert res.executed_backend == "numpy"
        assert res.fallback_reason is not None
        np.testing.assert_array_equal(res.value, [4.0, 6.0])

        last = get_last_dispatch()
        assert last is not None
        assert last.fell_back is True
        assert last.requested_backend == "vulkan"

    # 4. Unsupported dtype fallback
    def test_unsupported_dtype_fallback(self):
        """Passing non-supported dtype to CUDA or other backend triggers CPU fallback with reason."""
        clear_dispatch_history()
        cuda_backend = CudaOperatorBackend()
        # Float64 is unsupported by CUDA kernels
        x_f64 = np.array([2.0], dtype=np.float64)
        assert not cuda_backend.supports("relu", x_f64)
        reason = cuda_backend.unsupported_reason("relu", x_f64)
        assert "float32 only" in reason or "unavailable" in reason

        # Dispatch float64 on CUDA device
        res = dispatch("relu", Device("cuda", 0), x_f64)
        assert res.executed_backend == "numpy"
        if not eo.cuda.is_available():
            assert res.fell_back is True
        np.testing.assert_array_equal(res.value, [2.0])

    # 5. Strict selection errors
    def test_strict_selection_errors(self):
        """Strict device selection produces clear, actionable error messages."""
        with pytest.raises(EveryODeviceError, match="Invalid device string"):
            eo.device("invalid:device:spec", strict=True)

        with pytest.raises(EveryODeviceError, match="Unknown device type"):
            eo.device("nonexistent_vendor_xyz", strict=True)

        with pytest.raises(EveryODeviceError, match="CPU device only supports index 0"):
            Device("cpu", index=1)

        if not eo.cuda.is_available():
            with pytest.raises(EveryOCudaError, match="CUDA was requested with strict=True"):
                eo.device("cuda", strict=True)

    # 6. Device memory reporting when unsupported
    def test_device_memory_reporting_when_unsupported(self):
        """get_device_memory() returns safely without crashing across all backends."""
        for backend in eo.backends.get_all_backends().values():
            mem = backend.get_device_memory(0)
            assert isinstance(mem, dict)
            # Must either contain byte metrics or an error explanation
            assert "error" in mem or ("total_bytes" in mem and "free_bytes" in mem)
            if "total_bytes" in mem:
                assert mem["total_bytes"] >= 0
                assert mem["free_bytes"] >= 0

        # CPU device memory must be non-negative and self-consistent
        cpu_mem = eo.backends.get_backend("cpu").get_device_memory(0)
        assert cpu_mem["total_bytes"] >= 0
        assert cpu_mem["free_bytes"] >= 0
        assert cpu_mem["used_bytes"] >= 0
        assert cpu_mem["total_bytes"] == cpu_mem["free_bytes"] + cpu_mem["used_bytes"]

        # On primary supported platforms (Windows, Linux, macOS), native detection must report positive RAM
        import platform

        if platform.system() in ("Windows", "Linux", "Darwin"):
            assert cpu_mem["total_bytes"] > 0
            assert cpu_mem["free_bytes"] > 0

    # 7. Backend initialization failures
    def test_backend_initialization_failures_handled_gracefully(self):
        """Device detection failure in underlying subprocess does not crash the backend."""
        vulkan = eo.backends.get_backend("vulkan")
        assert vulkan is not None

        # Simulate subprocess failure during device detection
        with patch("subprocess.run", side_effect=subprocess.SubprocessError("Mocked failure")):
            devices = vulkan.detect_devices()
            assert devices == []

        with patch("subprocess.run", side_effect=FileNotFoundError("Tool missing")):
            devices = vulkan.detect_devices()
            assert devices == []

    # 8. Resource cleanup and synchronization
    def test_resource_cleanup_and_synchronize(self):
        """synchronize() and release() execute safely without errors across all backends."""
        for backend in eo.backends.get_all_backends().values():
            # Must not raise exceptions
            backend.synchronize(0)
            backend.release(0)

    # 9. CPU-only installation integrity
    def test_cpu_only_installation_integrity(self):
        """Full numerical computing pipeline functions identically without any accelerator."""
        # Verification that backend_for('cpu') resolves to NumPy reference
        cpu_op_backend = backend_for("cpu")
        assert cpu_op_backend.name == "numpy"

        # Basic tensors
        x = eo.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w = eo.tensor([[0.5, -0.5], [1.0, 0.0]], requires_grad=True)
        b = eo.tensor([0.1, 0.2], requires_grad=True)

        # Forward operations
        z = eo.matmul(x, w) + b
        a = eo.relu(z)
        loss = eo.mean(a)

        # Backward autograd
        loss.backward()
        assert x.grad is not None
        assert w.grad is not None
        assert b.grad is not None

        # Optimizer step
        opt = eo.optim.SGD([w, b], lr=0.1)
        w_before = w.numpy().copy()
        opt.step()
        assert not np.array_equal(w.numpy(), w_before)

        # Check dispatch confirmed numpy execution
        last = get_last_dispatch()
        assert last is not None
        assert last.executed_backend == "numpy"
