"""Vulkan Compute hardware backend.

Vulkan is a cross-platform graphics and compute API maintained by Khronos.
When vendor-specific runtimes are unavailable, Vulkan can serve as a
portable GPU compute fallback.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = ["VulkanBackend"]

_LOGGER = logging.getLogger(__name__)


class VulkanBackend(HardwareBackend):
    """Vulkan Compute backend."""

    @property
    def name(self) -> str:
        return "vulkan"

    @property
    def display_name(self) -> str:
        return "Vulkan Compute"

    @property
    def vendor(self) -> str:
        return "khronos"

    def is_available(self) -> bool:
        return shutil.which("vulkaninfo") is not None

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return "Vulkan runtime not detected.  Install GPU drivers with Vulkan support."

    def status(self) -> BackendStatus:
        return BackendStatus.AVAILABLE if self.is_available() else BackendStatus.RUNTIME_MISSING

    def detect_devices(self) -> list[DeviceInfo]:
        if not self.is_available():
            return []
        devices: list[DeviceInfo] = []
        try:
            import subprocess

            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "deviceName" in line:
                        name = line.split("=", 1)[1].strip() if "=" in line else line.strip()
                        devices.append(
                            DeviceInfo(
                                backend="vulkan",
                                device_type="vulkan",
                                index=len(devices),
                                name=name,
                                vendor="unknown",
                            )
                        )
        except Exception as exc:
            _LOGGER.debug("Vulkan device detection failed: %s", exc)
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
