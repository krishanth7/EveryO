"""OpenCL cross-vendor compute backend.

OpenCL is an open standard for parallel computing on CPUs, GPUs, and
other accelerators.  It serves as a portable fallback when vendor-specific
runtimes are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = ["OpenClBackend"]

_LOGGER = logging.getLogger(__name__)


class OpenClBackend(HardwareBackend):
    """OpenCL backend."""

    @property
    def name(self) -> str:
        return "opencl"

    @property
    def display_name(self) -> str:
        return "OpenCL"

    @property
    def vendor(self) -> str:
        return "khronos"

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        try:
            import pyopencl  # noqa: F401

            return "OpenCL runtime detected via PyOpenCL, but OpenCL compute operations are not yet implemented in EveryO."
        except ImportError:
            return "PyOpenCL is not installed.  Install with: pip install pyopencl"

    def status(self) -> BackendStatus:
        try:
            import pyopencl  # noqa: F401

            return BackendStatus.NOT_IMPLEMENTED
        except ImportError:
            return BackendStatus.NOT_INSTALLED

    def detect_devices(self) -> list[DeviceInfo]:
        if not self.is_available():
            return []
        devices: list[DeviceInfo] = []
        try:
            import pyopencl as cl

            for plat in cl.get_platforms():
                for dev in plat.get_devices():
                    devices.append(
                        DeviceInfo(
                            backend="opencl",
                            device_type="opencl",
                            index=len(devices),
                            name=dev.name.strip(),
                            vendor=dev.vendor.strip(),
                        )
                    )
        except Exception as exc:
            _LOGGER.debug("OpenCL device detection failed: %s", exc)
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
