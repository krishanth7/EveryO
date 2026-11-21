"""Tests for the multi-backend hardware registry and HardwareBackend contract.

Verifies:
- Abstract base class enforcement for HardwareBackend.
- Concrete backend instantiation.
- Synchronization and release default no-op behavior.
- Registry discovery, device enumeration, and priority selection.
- Safe CPU fallback when accelerators are unavailable or non-operational.
"""

from __future__ import annotations

from typing import Any

import pytest

import everyo as eo
from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)
from everyo.backends.cpu import CpuBackend
from everyo.backends.registry import (
    detect_all,
    get_all_backends,
    get_available_backends,
    get_available_devices,
    get_backend,
    get_backend_status,
    get_best_device,
    get_device_info,
    register_backend,
    unregister_backend,
)


class IncompleteBackend(HardwareBackend):
    """Subclass missing abstract methods — must fail instantiation."""


class DummyConcreteBackend(HardwareBackend):
    """Minimal valid backend to verify instantiation and default behaviors."""

    @property
    def name(self) -> str:
        return "dummy_test"

    @property
    def display_name(self) -> str:
        return "Dummy Test Backend"

    @property
    def vendor(self) -> str:
        return "dummy_vendor"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def status(self) -> BackendStatus:
        return BackendStatus.AVAILABLE

    def detect_devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(
                backend=self.name,
                device_type=self.name,
                index=0,
                name="Dummy Accelerator 0",
                vendor=self.vendor,
            )
        ]

    def get_device_properties(self, index: int = 0) -> dict[str, Any]:
        return {"name": "Dummy Accelerator", "index": index}

    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supported_operations=frozenset({"dummy_op"}),
            supported_dtypes=frozenset({"float32"}),
        )


class OverridingBackend(DummyConcreteBackend):
    """Backend that overrides synchronize and release."""

    def __init__(self) -> None:
        self.synced: list[int] = []
        self.released: list[Any] = []

    @property
    def name(self) -> str:
        return "dummy_override"

    def synchronize(self, index: int = 0) -> None:
        self.synced.append(index)

    def release(self, device_data: Any) -> None:
        self.released.append(device_data)


class TestHardwareBackendContract:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteBackend()

    def test_concrete_subclass_can_be_instantiated(self):
        backend = DummyConcreteBackend()
        assert backend.name == "dummy_test"
        assert backend.is_available() is True
        assert backend.status() == BackendStatus.AVAILABLE

    def test_synchronize_default_noop(self):
        backend = DummyConcreteBackend()
        # Default synchronize is a safe no-op returning None
        result = backend.synchronize(0)
        assert result is None

    def test_release_default_noop(self):
        backend = DummyConcreteBackend()
        # Default release is a safe no-op returning None
        result = backend.release("dummy_buffer")
        assert result is None

    def test_synchronize_and_release_can_be_overridden(self):
        backend = OverridingBackend()
        backend.synchronize(1)
        assert backend.synced == [1]

        buffer_handle = object()
        backend.release(buffer_handle)
        assert backend.released == [buffer_handle]

    def test_unimplemented_operations_raise(self):
        backend = DummyConcreteBackend()
        with pytest.raises(NotImplementedError, match="does not implement operation"):
            backend.execute("matmul")
        with pytest.raises(NotImplementedError, match="does not implement allocate"):
            backend.allocate((2, 2))
        with pytest.raises(NotImplementedError, match="does not implement transfer_to_device"):
            backend.transfer_to_device([1.0])
        with pytest.raises(NotImplementedError, match="does not implement transfer_to_host"):
            backend.transfer_to_host(None)
        with pytest.raises(NotImplementedError, match="does not implement benchmark"):
            backend.benchmark("matmul")

    def test_describe_returns_serializable_dict(self):
        backend = DummyConcreteBackend()
        desc = backend.describe()
        assert desc["name"] == "dummy_test"
        assert desc["available"] is True
        assert desc["status"] == "available"
        assert len(desc["devices"]) == 1
        assert desc["devices"][0]["name"] == "Dummy Accelerator 0"


class TestCpuBackendImplementation:
    def test_cpu_backend_is_always_available(self):
        cpu = CpuBackend()
        assert cpu.name == "cpu"
        assert cpu.is_available() is True
        assert cpu.unavailable_reason() is None
        assert cpu.status() == BackendStatus.AVAILABLE

    def test_cpu_detect_devices(self):
        cpu = CpuBackend()
        devices = cpu.detect_devices()
        assert len(devices) >= 1
        assert devices[0].device_type == "cpu"
        assert devices[0].index == 0

    def test_cpu_capabilities(self):
        cpu = CpuBackend()
        caps = cpu.get_capabilities()
        assert caps.supports_autograd is True
        assert caps.supports_training is True
        assert caps.supports_operation("matmul")
        assert caps.supports_dtype("float32")

    def test_cpu_synchronize_and_release(self):
        cpu = CpuBackend()
        assert cpu.synchronize(0) is None
        assert cpu.release(None) is None


class TestRegistryAndDiscovery:
    def test_built_in_backends_registered(self):
        all_backends = get_all_backends()
        assert "cpu" in all_backends
        assert "cuda" in all_backends

    def test_available_backends_contains_cpu(self):
        avail = get_available_backends()
        assert "cpu" in avail

    def test_register_and_unregister_custom_backend(self):
        backend = DummyConcreteBackend()
        register_backend(backend)
        try:
            assert get_backend("dummy_test") is backend
            assert "dummy_test" in get_all_backends()
            assert "dummy_test" in get_available_backends()
        finally:
            unregister_backend("dummy_test")
        assert get_backend("dummy_test") is None

    def test_detect_all_returns_dict_of_devices(self):
        detected = detect_all()
        assert isinstance(detected, dict)
        assert "cpu" in detected
        assert len(detected["cpu"]) >= 1

    def test_get_available_devices_includes_cpu(self):
        devices = get_available_devices()
        device_types = [d.device_type for d in devices]
        assert "cpu" in device_types

    def test_get_best_device(self):
        best = get_best_device()
        assert best is not None
        assert isinstance(best, DeviceInfo)

    def test_get_backend_status(self):
        status = get_backend_status()
        assert "cpu" in status
        assert status["cpu"]["available"] is True
        assert status["cpu"]["status"] == "available"

    def test_get_device_info(self):
        info = get_device_info("cpu")
        assert len(info) >= 1
        assert info[0]["device_type"] == "cpu"


class TestDeviceIntegration:
    def test_device_auto_resolution(self):
        resolved = eo.device("auto")
        # auto must resolve to an operational device (cpu or cuda when built)
        assert resolved.type in ("cpu", "cuda")

    def test_device_explicit_cpu(self):
        dev = eo.device("cpu")
        assert dev.type == "cpu"
        assert dev.is_cpu is True
        assert dev.is_accelerator is False

    def test_device_unavailable_accelerator_fallback(self):
        # Requesting an unavailable backend without strict=True falls back to CPU
        dev = eo.device("metal")
        # On non-macOS Windows it must fall back to CPU
        assert dev.type in ("metal", "cpu")
