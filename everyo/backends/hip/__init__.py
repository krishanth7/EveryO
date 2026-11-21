"""AMD HIP hardware backend.

HIP (Heterogeneous-Compute Interface for Portability) is AMD's C++ runtime
API that can target both AMD and NVIDIA GPUs.
"""

from __future__ import annotations

import logging
import platform
import shutil
from typing import Any

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = ["HipBackend"]

_LOGGER = logging.getLogger(__name__)


class HipBackend(HardwareBackend):
    """AMD HIP backend."""

    @property
    def name(self) -> str:
        return "hip"

    @property
    def display_name(self) -> str:
        return "AMD HIP"

    @property
    def vendor(self) -> str:
        return "amd"

    def is_available(self) -> bool:
        return self.status() == BackendStatus.AVAILABLE

    def unavailable_reason(self) -> str | None:
        status = self.status()
        if status == BackendStatus.AVAILABLE:
            return None
        return "HIP runtime not detected.  Install ROCm/HIP from https://rocm.docs.amd.com/"

    def status(self) -> BackendStatus:
        if platform.system() != "Linux":
            return BackendStatus.UNAVAILABLE
        if not shutil.which("hipconfig"):
            return BackendStatus.RUNTIME_MISSING
        return BackendStatus.AVAILABLE

    def detect_devices(self) -> list[DeviceInfo]:
        # HIP shares device detection with ROCm.
        from everyo.backends.rocm import RocmBackend

        rocm = RocmBackend()
        return [
            DeviceInfo(
                backend="hip",
                device_type="hip",
                index=d.index,
                name=d.name,
                vendor="amd",
            )
            for d in rocm.detect_devices()
        ]

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
        base["note"] = "HIP backend: device detection implemented; compute operations not yet implemented."
        return base
