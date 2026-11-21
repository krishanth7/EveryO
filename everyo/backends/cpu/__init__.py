"""CPU backend implemented as a :class:`HardwareBackend`.

The CPU backend is *always* available.  It wraps :mod:`everyo.backends.numpy_backend`
and provides the guaranteed fallback for every operation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.backends import numpy_backend
from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)
from everyo.backends.capabilities import STANDARD_OPERATIONS

__all__ = ["CpuBackend"]


class CpuBackend(HardwareBackend):
    """Always-available CPU backend backed by NumPy."""

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def display_name(self) -> str:
        return "CPU (NumPy)"

    @property
    def vendor(self) -> str:
        return "generic"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def status(self) -> BackendStatus:
        return BackendStatus.AVAILABLE

    def detect_devices(self) -> list[DeviceInfo]:
        from everyo.backends.detection import detect_cpu_info, detect_system_memory

        info = detect_cpu_info()
        mem = detect_system_memory()
        return [
            DeviceInfo(
                backend="cpu",
                device_type="cpu",
                index=0,
                name=info.get("model_name", info.get("processor", "CPU")),
                vendor="generic",
                total_memory_bytes=mem["total_bytes"],
                free_memory_bytes=mem["free_bytes"],
                supported_dtypes=("float32", "float64", "int32", "int64", "bool"),
            )
        ]

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        from everyo.backends.detection import detect_cpu_info

        return detect_cpu_info()

    def get_device_memory(self, index: int = 0) -> dict[str, int]:
        from everyo.backends.detection import detect_system_memory

        return detect_system_memory()

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_operations=frozenset(STANDARD_OPERATIONS),
            supported_dtypes=frozenset({"float32", "float64", "int32", "int64", "bool"}),
            supports_autograd=True,
            supports_training=True,
            supports_inference=True,
        )

    def execute(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(numpy_backend, operation, None)
        if fn is None:
            raise NotImplementedError(f"CPU backend does not implement operation '{operation}'.")
        return np.asarray(fn(*args))

    def allocate(self, shape: tuple[int, ...], dtype: str = "float32") -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def transfer_to_device(self, data: Any, index: int = 0) -> np.ndarray:
        return np.asarray(data)

    def transfer_to_host(self, device_data: Any) -> np.ndarray:
        return np.asarray(device_data)

    def synchronize(self, index: int = 0) -> None:
        pass  # CPU is always synchronous.

    def release(self, device_data: Any) -> None:
        pass  # NumPy arrays are managed by Python's GC.

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["numpy_version"] = np.__version__
        return base
