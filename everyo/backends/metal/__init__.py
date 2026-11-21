"""Apple Metal hardware backend.

Metal provides GPU acceleration on macOS and iOS.  This backend detects
Apple Silicon (M-series) GPUs and Metal Performance Shaders (MPS).
"""

from __future__ import annotations

import logging
import platform
from typing import Any

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = ["MetalBackend"]

_LOGGER = logging.getLogger(__name__)


class MetalBackend(HardwareBackend):
    """Apple Metal / MPS backend."""

    @property
    def name(self) -> str:
        return "metal"

    @property
    def display_name(self) -> str:
        return "Apple Metal"

    @property
    def vendor(self) -> str:
        return "apple"

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        if platform.system() != "Darwin":
            return "Metal is only available on macOS."
        return "Apple Metal hardware detected, but Metal compute kernels are not yet implemented in EveryO."

    def status(self) -> BackendStatus:
        if platform.system() != "Darwin":
            return BackendStatus.UNAVAILABLE
        return BackendStatus.NOT_IMPLEMENTED

    def detect_devices(self) -> list[DeviceInfo]:
        if platform.system() != "Darwin":
            return []
        devices: list[DeviceInfo] = []
        try:
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse for chipset/GPU names.
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if "Chipset Model:" in line or "Chip:" in line:
                        name = line.split(":", 1)[1].strip()
                        devices.append(
                            DeviceInfo(
                                backend="metal",
                                device_type="metal",
                                index=len(devices),
                                name=name,
                                vendor="apple",
                            )
                        )
        except Exception as exc:
            _LOGGER.debug("Metal device detection failed: %s", exc)

        # Ensure at least one device on Apple Silicon.
        if not devices and platform.machine() == "arm64":
            devices.append(
                DeviceInfo(
                    backend="metal",
                    device_type="metal",
                    index=0,
                    name="Apple Silicon GPU",
                    vendor="apple",
                )
            )
        return devices

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        devices = self.detect_devices()
        if index < len(devices):
            return {"name": devices[index].name, "index": index}
        return {"error": f"Device index {index} out of range."}

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_operations=frozenset(),
            supported_dtypes=frozenset({"float32", "float16"}),
            supports_inference=True,
        )
