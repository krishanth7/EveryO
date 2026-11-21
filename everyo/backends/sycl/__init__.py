"""SYCL cross-vendor backend.

SYCL is a C++ standard for heterogeneous computing that can target
CPUs, GPUs, and accelerators from multiple vendors.
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

__all__ = ["SyclBackend"]

_LOGGER = logging.getLogger(__name__)


class SyclBackend(HardwareBackend):
    """SYCL / DPC++ backend."""

    @property
    def name(self) -> str:
        return "sycl"

    @property
    def display_name(self) -> str:
        return "SYCL / DPC++"

    @property
    def vendor(self) -> str:
        return "khronos"

    def is_available(self) -> bool:
        return shutil.which("sycl-ls") is not None

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return "SYCL runtime not detected.  Install a SYCL implementation such as Intel oneAPI DPC++."

    def status(self) -> BackendStatus:
        return BackendStatus.AVAILABLE if self.is_available() else BackendStatus.RUNTIME_MISSING

    def detect_devices(self) -> list[DeviceInfo]:
        # Shares detection with the Intel backend.
        from everyo.backends.intel import IntelBackend
        intel = IntelBackend()
        return [
            DeviceInfo(
                backend="sycl",
                device_type="sycl",
                index=d.index,
                name=d.name,
                vendor=d.vendor,
            )
            for d in intel.detect_devices()
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
