"""Microsoft DirectML hardware backend.

DirectML provides GPU acceleration on Windows through DirectX 12.
It supports any DirectX 12-compatible GPU (NVIDIA, AMD, Intel, Qualcomm).
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

__all__ = ["DirectMlBackend"]

_LOGGER = logging.getLogger(__name__)

_UNAVAILABLE_REASON: str | None = None


class DirectMlBackend(HardwareBackend):
    """Windows DirectML backend."""

    @property
    def name(self) -> str:
        return "directml"

    @property
    def display_name(self) -> str:
        return "Microsoft DirectML"

    @property
    def vendor(self) -> str:
        return "microsoft"

    def is_available(self) -> bool:
        return self.status() == BackendStatus.AVAILABLE

    def unavailable_reason(self) -> str | None:
        self.status()
        return _UNAVAILABLE_REASON

    def status(self) -> BackendStatus:
        global _UNAVAILABLE_REASON

        if platform.system() != "Windows":
            _UNAVAILABLE_REASON = "DirectML is only available on Windows."
            return BackendStatus.UNAVAILABLE

        # Check for onnxruntime-directml or directml package.
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                _UNAVAILABLE_REASON = "DirectML runtime detected via ONNX Runtime, but DirectML compute kernels are not yet implemented in EveryO."
                return BackendStatus.NOT_IMPLEMENTED
            _UNAVAILABLE_REASON = (
                "ONNX Runtime is installed but DmlExecutionProvider is not available. "
                "Install onnxruntime-directml: pip install onnxruntime-directml"
            )
            return BackendStatus.RUNTIME_MISSING
        except ImportError:
            pass

        try:
            import directml  # noqa: F401

            _UNAVAILABLE_REASON = "DirectML runtime detected, but DirectML compute kernels are not yet implemented in EveryO."
            return BackendStatus.NOT_IMPLEMENTED
        except ImportError:
            pass

        _UNAVAILABLE_REASON = (
            "DirectML runtime not found.  Install onnxruntime-directml: "
            "pip install onnxruntime-directml"
        )
        return BackendStatus.NOT_INSTALLED

    def detect_devices(self) -> list[DeviceInfo]:
        if platform.system() != "Windows":
            return []
        devices: list[DeviceInfo] = []
        try:
            import onnxruntime as ort

            # DirectML through ONNX Runtime doesn't expose device enumeration
            # directly, but we can confirm it's available.
            if "DmlExecutionProvider" in ort.get_available_providers():
                devices.append(
                    DeviceInfo(
                        backend="directml",
                        device_type="directml",
                        index=0,
                        name="DirectML Device",
                        vendor="microsoft",
                    )
                )
        except Exception as exc:
            _LOGGER.debug("DirectML device detection failed: %s", exc)
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
