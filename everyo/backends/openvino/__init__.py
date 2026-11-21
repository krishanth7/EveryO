"""Intel OpenVINO hardware backend.

OpenVINO is Intel's toolkit for optimising and deploying inference
workloads on Intel hardware (CPUs, GPUs, NPUs).
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

__all__ = ["OpenVinoBackend"]

_LOGGER = logging.getLogger(__name__)


class OpenVinoBackend(HardwareBackend):
    """OpenVINO backend for Intel inference acceleration."""

    @property
    def name(self) -> str:
        return "openvino"

    @property
    def display_name(self) -> str:
        return "Intel OpenVINO"

    @property
    def vendor(self) -> str:
        return "intel"

    def is_available(self) -> bool:
        # OpenVINO can be probed if installed, but direct tensor compute operations are not yet implemented.
        return False

    def unavailable_reason(self) -> str | None:
        try:
            import openvino  # noqa: F401

            return "OpenVINO runtime detected, but direct tensor compute operations are not yet implemented in EveryO."
        except ImportError:
            return "OpenVINO is not installed.  Install with: pip install openvino"

    def status(self) -> BackendStatus:
        try:
            import openvino  # noqa: F401

            return BackendStatus.NOT_IMPLEMENTED
        except ImportError:
            return BackendStatus.NOT_INSTALLED

    def detect_devices(self) -> list[DeviceInfo]:
        if not self.is_available():
            return []
        devices: list[DeviceInfo] = []
        try:
            import openvino as ov

            core = ov.Core()
            for dev_name in core.available_devices:
                devices.append(
                    DeviceInfo(
                        backend="openvino",
                        device_type="openvino",
                        index=len(devices),
                        name=dev_name,
                        vendor="intel",
                    )
                )
        except Exception as exc:
            _LOGGER.debug("OpenVINO device detection failed: %s", exc)
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
