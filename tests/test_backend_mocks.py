"""Mocked vendor hardware detection tests.

NOTE: These tests deliberately use mocks to verify hardware parsing, discovery,
and fallback logic across diverse platform conditions (Linux ROCm, Intel SYCL,
macOS Metal, Windows DirectML).

A mocked test is strictly an architectural verification tool and is NEVER
reported as real hardware execution.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from everyo.backends.base import BackendStatus
from everyo.backends.directml import DirectMlBackend
from everyo.backends.intel import IntelBackend
from everyo.backends.metal import MetalBackend
from everyo.backends.rocm import RocmBackend
from everyo.backends.vulkan import VulkanBackend


class TestMockedRocmDetection:
    """Verifies ROCm discovery logic on simulated Linux ROCm environment."""

    @patch("platform.system", return_value="Linux")
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_rocm_device_parsing_linux(self, mock_run, mock_which, mock_system):
        mock_which.side_effect = lambda cmd: (
            "/opt/rocm/bin/" + cmd if cmd in ("rocm-smi", "hipconfig") else None
        )
        fake_stdout = """
======================= ROCm System Management Interface =======================
================================= Product Info =================================
GPU[0]                  : Card Series: AMD Radeon RX 7900 XTX
GPU[1]                  : Card Series: AMD Radeon PRO W7900
================================================================================
"""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["rocm-smi", "--showproductname"],
            returncode=0,
            stdout=fake_stdout,
            stderr="",
        )

        backend = RocmBackend()
        devices = backend.detect_devices()
        assert len(devices) == 2
        assert "AMD Radeon RX 7900 XTX" in devices[0].name
        assert "AMD Radeon PRO W7900" in devices[1].name
        assert devices[0].vendor == "amd"
        # Crucial: detection works, but compute operations report not implemented
        assert backend.is_available() is False
        assert backend.status() == BackendStatus.NOT_IMPLEMENTED


class TestMockedIntelSyclDetection:
    """Verifies Intel oneAPI SYCL device parsing on simulated system."""

    @patch("shutil.which", return_value="/opt/intel/oneapi/compiler/latest/bin/sycl-ls")
    @patch("subprocess.run")
    def test_sycl_ls_parsing(self, mock_run, mock_which):
        fake_stdout = """
[opencl:gpu:0] Intel(R) OpenCL HD Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
"""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["sycl-ls"],
            returncode=0,
            stdout=fake_stdout,
            stderr="",
        )

        backend = IntelBackend()
        devices = backend.detect_devices()
        assert len(devices) == 2
        assert "Intel(R) Arc(TM) A770 Graphics" in devices[0].name
        assert backend.is_available() is False
        assert backend.status() == BackendStatus.NOT_IMPLEMENTED


class TestMockedMetalDetection:
    """Verifies Apple Metal discovery on simulated macOS environment."""

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    @patch("subprocess.run")
    def test_metal_apple_silicon_detection(self, mock_run, mock_machine, mock_system):
        fake_stdout = """
Graphics/Displays:

    Chipset Model: Apple M3 Max
    Type: GPU
    Bus: Built-In
    Total Number of Cores: 40
    Metal Support: Metal 3
"""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"],
            returncode=0,
            stdout=fake_stdout,
            stderr="",
        )

        backend = MetalBackend()
        devices = backend.detect_devices()
        assert len(devices) >= 1
        assert "Apple M3 Max" in devices[0].name
        assert devices[0].vendor == "apple"
        assert backend.status() == BackendStatus.NOT_IMPLEMENTED


class TestMockedDirectMlDetection:
    """Verifies DirectML discovery when onnxruntime DirectML EP is present."""

    @patch("platform.system", return_value="Windows")
    def test_directml_provider_detected(self, mock_system):
        fake_ort = MagicMock()
        fake_ort.get_available_providers.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
            backend = DirectMlBackend()
            devices = backend.detect_devices()
            assert len(devices) == 1
            assert devices[0].device_type == "directml"
            assert backend.status() == BackendStatus.NOT_IMPLEMENTED


class TestMockedVulkanDetection:
    """Verifies Vulkan summary parsing."""

    @patch("shutil.which", return_value="vulkaninfo")
    @patch("subprocess.run")
    def test_vulkaninfo_summary_parsing(self, mock_run, mock_which):
        fake_stdout = """
Devices:
========
GPU0:
    apiVersion         = 1.3.261
    driverVersion      = 31.0.101.4577
    vendorID           = 0x8086
    deviceID           = 0x56a0
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = Intel(R) Arc(TM) A770 Graphics
"""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["vulkaninfo", "--summary"],
            returncode=0,
            stdout=fake_stdout,
            stderr="",
        )

        backend = VulkanBackend()
        devices = backend.detect_devices()
        assert len(devices) == 1
        assert "Intel(R) Arc(TM) A770 Graphics" in devices[0].name
        assert backend.status() == BackendStatus.NOT_IMPLEMENTED
        assert backend.is_available() is False
