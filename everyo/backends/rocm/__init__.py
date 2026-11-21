"""AMD ROCm hardware backend.

ROCm is AMD's open-source GPU compute platform.  This backend detects
ROCm-capable AMD GPUs and can delegate execution through available
ROCm/HIP runtimes.

.. note::

   ROCm has limited Windows support.  This backend is primarily
   designed for Linux systems.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from typing import Any

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = ["RocmBackend"]

_LOGGER = logging.getLogger(__name__)

_UNAVAILABLE_REASON: str | None = "ROCm availability has not been probed yet."


class RocmBackend(HardwareBackend):
    """AMD ROCm/HIP backend."""

    @property
    def name(self) -> str:
        return "rocm"

    @property
    def display_name(self) -> str:
        return "AMD ROCm"

    @property
    def vendor(self) -> str:
        return "amd"

    def is_available(self) -> bool:
        return self.status() == BackendStatus.AVAILABLE

    def unavailable_reason(self) -> str | None:
        self.status()
        return _UNAVAILABLE_REASON

    def status(self) -> BackendStatus:
        global _UNAVAILABLE_REASON

        # ROCm is Linux-primary.
        if platform.system() not in ("Linux",):
            _UNAVAILABLE_REASON = (
                f"ROCm is not supported on {platform.system()}. ROCm requires Linux."
            )
            return BackendStatus.UNAVAILABLE

        # Check for rocm-smi or hipconfig.
        if not shutil.which("rocm-smi") and not shutil.which("hipconfig"):
            rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
            if not os.path.isdir(rocm_path):
                _UNAVAILABLE_REASON = (
                    "ROCm runtime not found.  Install ROCm from "
                    "https://rocm.docs.amd.com/ or set the ROCM_PATH "
                    "environment variable."
                )
                return BackendStatus.RUNTIME_MISSING

        # Try to detect devices.
        devices = self.detect_devices()
        if not devices:
            _UNAVAILABLE_REASON = (
                "ROCm runtime is installed but no compatible AMD GPU was detected."
            )
            return BackendStatus.NO_DEVICES

        _UNAVAILABLE_REASON = "AMD ROCm hardware detected, but ROCm compute kernels are not yet implemented in EveryO."
        return BackendStatus.NOT_IMPLEMENTED

    def detect_devices(self) -> list[DeviceInfo]:
        if platform.system() != "Linux":
            return []
        devices: list[DeviceInfo] = []
        try:
            import subprocess

            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse output for GPU names.
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("=") and "GPU" in line.upper():
                        devices.append(
                            DeviceInfo(
                                backend="rocm",
                                device_type="rocm",
                                index=len(devices),
                                name=line,
                                vendor="amd",
                            )
                        )
        except Exception as exc:
            _LOGGER.debug("ROCm device detection failed: %s", exc)
        return devices

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        devices = self.detect_devices()
        if index < len(devices):
            return {
                "name": devices[index].name,
                "index": index,
                "vendor": "amd",
            }
        return {"error": f"Device index {index} out of range."}

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_operations=frozenset(),  # No operations implemented yet.
            supported_dtypes=frozenset({"float32"}),
            supports_autograd=False,
            supports_training=False,
            supports_inference=False,
        )

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["note"] = (
            "ROCm backend: device detection implemented; compute operations not yet implemented."
        )
        return base
