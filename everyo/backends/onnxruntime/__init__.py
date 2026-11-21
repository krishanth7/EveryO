"""ONNX Runtime multi-provider backend.

ONNX Runtime can delegate to many execution providers (CPU, CUDA, DirectML,
OpenVINO, CoreML, etc.).  This backend dynamically detects which providers
are installed and available.
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

__all__ = ["OnnxRuntimeBackend"]

_LOGGER = logging.getLogger(__name__)


class OnnxRuntimeBackend(HardwareBackend):
    """ONNX Runtime backend with dynamic execution-provider detection."""

    @property
    def name(self) -> str:
        return "onnxruntime"

    @property
    def display_name(self) -> str:
        return "ONNX Runtime"

    @property
    def vendor(self) -> str:
        return "microsoft"

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        if self.get_available_providers():
            return "ONNX Runtime providers detected, but direct tensor operations are not yet implemented in EveryO."
        return "ONNX Runtime is not installed.  Install with: pip install onnxruntime"

    def status(self) -> BackendStatus:
        try:
            import onnxruntime  # noqa: F401

            return BackendStatus.NOT_IMPLEMENTED
        except ImportError:
            return BackendStatus.NOT_INSTALLED

    def get_available_providers(self) -> list[str]:
        """Return the list of available ONNX Runtime execution providers."""
        if not self.is_available():
            return []
        try:
            import onnxruntime as ort

            return list(ort.get_available_providers())
        except Exception as exc:
            _LOGGER.debug("ONNX Runtime provider detection failed: %s", exc)
            return []

    def detect_devices(self) -> list[DeviceInfo]:
        providers = self.get_available_providers()
        return [
            DeviceInfo(
                backend="onnxruntime",
                device_type="onnxruntime",
                index=i,
                name=provider,
                vendor="microsoft",
            )
            for i, provider in enumerate(providers)
        ]

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        providers = self.get_available_providers()
        if index < len(providers):
            return {"provider": providers[index], "index": index}
        return {"error": f"Provider index {index} out of range."}

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_operations=frozenset(),
            supported_dtypes=frozenset({"float32", "float16", "float64", "int32", "int64"}),
            supports_inference=True,
        )

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["providers"] = self.get_available_providers()
        return base
