"""NVIDIA CUDA hardware backend.

Wraps the existing :mod:`everyo.cuda` module as a :class:`HardwareBackend`
without duplicating any logic.  All existing CUDA functionality is preserved.
"""

from __future__ import annotations

from typing import Any

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = ["CudaHardwareBackend"]


class CudaHardwareBackend(HardwareBackend):
    """CUDA backend delegating to :mod:`everyo.cuda`."""

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def display_name(self) -> str:
        return "NVIDIA CUDA"

    @property
    def vendor(self) -> str:
        return "nvidia"

    def is_available(self) -> bool:
        from everyo.cuda.availability import is_available

        return is_available()

    def unavailable_reason(self) -> str | None:
        from everyo.cuda.availability import unavailable_reason

        return unavailable_reason()

    def status(self) -> BackendStatus:
        if self.is_available():
            return BackendStatus.AVAILABLE
        reason = self.unavailable_reason() or ""
        if "not built" in reason.lower() or "not installed" in reason.lower():
            return BackendStatus.NOT_INSTALLED
        if "no cuda" in reason.lower() or "no device" in reason.lower():
            return BackendStatus.NO_DEVICES
        return BackendStatus.UNAVAILABLE

    def detect_devices(self) -> list[DeviceInfo]:
        from everyo.cuda.availability import device_count, runtime_info

        if not self.is_available():
            return []

        info = runtime_info()
        devices = []
        for i in range(device_count()):
            props = info.get("devices", [{}])
            dev_props = props[i] if i < len(props) else {}
            devices.append(
                DeviceInfo(
                    backend="cuda",
                    device_type="cuda",
                    index=i,
                    name=dev_props.get("name", f"CUDA Device {i}"),
                    vendor="nvidia",
                    runtime_version=str(info.get("runtime_version", "")),
                    total_memory_bytes=dev_props.get("total_memory", 0),
                    supported_dtypes=("float32",),
                    extra=dev_props,
                )
            )
        return devices

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        from everyo.cuda.availability import runtime_info

        info = runtime_info()
        devices = info.get("devices", [])
        if index < len(devices):
            return dict(devices[index])
        return {"error": f"Device index {index} out of range."}

    def get_capabilities(self) -> BackendCapabilities:
        from everyo.cuda.interface import KERNEL_NAMES

        return BackendCapabilities(
            supported_operations=frozenset(KERNEL_NAMES),
            supported_dtypes=frozenset({"float32"}),
            supports_autograd=False,
            supports_training=False,
            supports_inference=True,
        )

    def execute(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        from everyo.cuda import interface

        kernel = interface.get_kernel(operation)
        if kernel is None:
            raise NotImplementedError(f"CUDA backend does not implement operation '{operation}'.")
        import numpy as np

        return np.asarray(kernel(*args, strict=True))

    def describe(self) -> dict[str, Any]:
        from everyo.cuda.availability import runtime_info

        base = super().describe()
        base["cuda_info"] = runtime_info()
        return base
