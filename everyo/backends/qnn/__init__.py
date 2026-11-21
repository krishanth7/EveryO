"""Qualcomm QNN / AI Engine hardware backend.

QNN (Qualcomm Neural Network) SDK provides NPU acceleration on
supported Snapdragon hardware.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = ["QnnBackend"]

_LOGGER = logging.getLogger(__name__)


class QnnBackend(HardwareBackend):
    """Qualcomm QNN / AI Engine backend."""

    @property
    def name(self) -> str:
        return "qnn"

    @property
    def display_name(self) -> str:
        return "Qualcomm QNN"

    @property
    def vendor(self) -> str:
        return "qualcomm"

    def is_available(self) -> bool:
        return self.status() == BackendStatus.AVAILABLE

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        return (
            "Qualcomm QNN SDK not detected.  Install from "
            "https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk"
        )

    def status(self) -> BackendStatus:
        # Check for QNN SDK environment.
        if os.environ.get("QNN_SDK_ROOT"):
            return BackendStatus.AVAILABLE
        # Check for ONNX Runtime QNN EP.
        try:
            import onnxruntime as ort

            if "QNNExecutionProvider" in ort.get_available_providers():
                return BackendStatus.AVAILABLE
        except ImportError:
            pass
        return BackendStatus.NOT_INSTALLED

    def detect_devices(self) -> list[DeviceInfo]:
        if not self.is_available():
            return []
        return [
            DeviceInfo(
                backend="qnn",
                device_type="qnn",
                index=0,
                name="Qualcomm NPU",
                vendor="qualcomm",
            )
        ]

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        return {"name": "Qualcomm NPU", "index": index}

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_operations=frozenset(),
            supported_dtypes=frozenset({"float32", "float16"}),
            supports_inference=True,
        )
