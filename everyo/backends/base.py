"""Abstract base class for all EveryO hardware backends.

Every backend — CPU, CUDA, ROCm, Intel, DirectML, Metal, Vulkan, OpenCL,
ONNX Runtime, QNN, ARM, and future additions — implements this contract.

The base class deliberately does *not* import NumPy or any vendor SDK at
module level so that importing this module is always cheap and safe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "DeviceType",
    "DeviceInfo",
    "BackendCapabilities",
    "BackendStatus",
    "HardwareBackend",
]


class DeviceType(str, Enum):
    """Known device types in EveryO."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    HIP = "hip"
    INTEL = "intel"
    SYCL = "sycl"
    OPENVINO = "openvino"
    DIRECTML = "directml"
    METAL = "metal"
    VULKAN = "vulkan"
    OPENCL = "opencl"
    QNN = "qnn"
    ONNXRUNTIME = "onnxruntime"
    ARM = "arm"


@dataclass(frozen=True)
class DeviceInfo:
    """Description of a single detected hardware device."""

    backend: str
    device_type: str
    index: int = 0
    name: str = "unknown"
    vendor: str = "unknown"
    driver_version: str = ""
    runtime_version: str = ""
    compute_capability: str = ""
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    supported_dtypes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def device_string(self) -> str:
        """Return the ``'type:index'`` device identifier."""
        return f"{self.device_type}:{self.index}" if self.index else self.device_type


@dataclass(frozen=True)
class BackendCapabilities:
    """Declares which operations a backend supports."""

    supported_operations: frozenset[str] = frozenset()
    supported_dtypes: frozenset[str] = frozenset({"float32"})
    supports_autograd: bool = False
    supports_training: bool = False
    supports_inference: bool = True
    max_tensor_dims: int = 64
    extra: dict[str, Any] = field(default_factory=dict)

    def supports_operation(self, operation: str) -> bool:
        """Check whether a specific operation is supported."""
        return operation in self.supported_operations

    def supports_dtype(self, dtype: str) -> bool:
        """Check whether a specific dtype is supported."""
        return dtype in self.supported_dtypes


class BackendStatus(str, Enum):
    """Current status of a backend."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_INSTALLED = "not_installed"
    NOT_INITIALIZED = "not_initialized"
    INITIALIZATION_FAILED = "initialization_failed"
    RUNTIME_MISSING = "runtime_missing"
    DRIVER_MISSING = "driver_missing"
    NO_DEVICES = "no_devices"
    NOT_IMPLEMENTED = "not_implemented"


class HardwareBackend(ABC):
    """Contract implemented by every hardware acceleration backend.

    A backend wraps a vendor-specific compute runtime (CUDA, ROCm, oneAPI,
    DirectML, Metal, etc.) behind a uniform interface.  The base class
    enforces the contract; concrete subclasses provide the implementation.

    **Thread safety:** A backend instance may be shared across threads; each
    method must be safe to call concurrently or document otherwise.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, lowercase identifier: ``'cpu'``, ``'cuda'``, ``'rocm'``, etc."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in diagnostics."""

    @property
    @abstractmethod
    def vendor(self) -> str:
        """Hardware vendor: ``'nvidia'``, ``'amd'``, ``'intel'``, etc."""

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when this backend can execute operations *right now*.

        This must verify runtime + driver + device, not just check an import.
        """

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """Human-readable explanation when :meth:`is_available` returns False."""

    @abstractmethod
    def status(self) -> BackendStatus:
        """Return the current lifecycle status."""

    # ------------------------------------------------------------------
    # Device enumeration
    # ------------------------------------------------------------------
    @abstractmethod
    def detect_devices(self) -> list[DeviceInfo]:
        """Return every device this backend can see.

        Returns an empty list when the runtime or driver is missing.
        Never raises for missing hardware.
        """

    @abstractmethod
    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        """Return detailed properties for the device at ``index``."""

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Describe which operations, dtypes and features this backend offers."""

    def supports_operation(self, operation: str) -> bool:
        """Convenience: does this backend support ``operation``?"""
        return self.get_capabilities().supports_operation(operation)

    def supports_dtype(self, dtype: str) -> bool:
        """Convenience: does this backend support ``dtype``?"""
        return self.get_capabilities().supports_dtype(dtype)

    # ------------------------------------------------------------------
    # Execution (default stubs — concrete backends override as needed)
    # ------------------------------------------------------------------
    def execute(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a named operation on this backend.

        Raises:
            NotImplementedError: When the operation is not supported.
        """
        raise NotImplementedError(
            f"Backend '{self.name}' does not implement operation '{operation}'."
        )

    def allocate(self, shape: tuple[int, ...], dtype: str = "float32") -> Any:
        """Allocate a device buffer. Returns a backend-specific handle."""
        raise NotImplementedError(f"Backend '{self.name}' does not implement allocate().")

    def transfer_to_device(self, data: Any, index: int = 0) -> Any:
        """Copy host data to this backend's device."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not implement transfer_to_device()."
        )

    def transfer_to_host(self, device_data: Any) -> Any:
        """Copy device data back to host (NumPy array)."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not implement transfer_to_host()."
        )

    def synchronize(self, index: int = 0) -> None:
        """Block until all pending operations on the device complete."""

    def release(self, device_data: Any) -> None:
        """Release a device buffer."""

    def benchmark(self, operation: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run a micro-benchmark of ``operation`` and return timing data."""
        raise NotImplementedError(f"Backend '{self.name}' does not implement benchmark().")

    # ------------------------------------------------------------------
    # Descriptors
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for diagnostics."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "available": self.is_available(),
            "status": self.status().value,
            "reason": self.unavailable_reason(),
            "devices": [
                {
                    "name": d.name,
                    "index": d.index,
                    "vendor": d.vendor,
                    "memory_bytes": d.total_memory_bytes,
                }
                for d in self.detect_devices()
            ],
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} available={self.is_available()}>"
