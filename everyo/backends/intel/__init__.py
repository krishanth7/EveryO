"""Intel oneAPI / SYCL / Level Zero hardware backend.

Detects Intel GPUs (Arc, integrated), Intel CPUs, and NPUs through
the oneAPI ecosystem.
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

__all__ = ["IntelBackend"]

_LOGGER = logging.getLogger(__name__)


class IntelBackend(HardwareBackend):
    """Intel oneAPI / SYCL backend."""

    @property
    def name(self) -> str:
        return "intel"

    @property
    def display_name(self) -> str:
        return "Intel oneAPI"

    @property
    def vendor(self) -> str:
        return "intel"

    def is_available(self) -> bool:
        return self.status() == BackendStatus.AVAILABLE

    def unavailable_reason(self) -> str | None:
        status = self.status()
        if status == BackendStatus.AVAILABLE:
            return None
        return (
            "Intel oneAPI runtime not detected.  Install from "
            "https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html"
        )

    def status(self) -> BackendStatus:
        # Check for SYCL runtime.
        if shutil.which("sycl-ls"):
            return BackendStatus.AVAILABLE
        # Check for oneAPI environment.
        if os.environ.get("ONEAPI_ROOT") or os.environ.get("CMPLR_ROOT"):
            return BackendStatus.AVAILABLE
        return BackendStatus.RUNTIME_MISSING

    def detect_devices(self) -> list[DeviceInfo]:
        devices: list[DeviceInfo] = []
        try:
            import subprocess

            result = subprocess.run(
                ["sycl-ls"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line and ("gpu" in line.lower() or "cpu" in line.lower()):
                        devices.append(
                            DeviceInfo(
                                backend="intel",
                                device_type="intel",
                                index=len(devices),
                                name=line,
                                vendor="intel",
                            )
                        )
        except Exception as exc:
            _LOGGER.debug("Intel device detection failed: %s", exc)
        return devices

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        devices = self.detect_devices()
        if index < len(devices):
            return {"name": devices[index].name, "index": index}
        return {"error": f"Device index {index} out of range."}

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_operations=frozenset(),
            supported_dtypes=frozenset({"float32"}),
        )

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["note"] = (
            "Intel backend: device detection implemented; compute operations not yet implemented."
        )
        return base
