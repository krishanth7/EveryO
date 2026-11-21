"""Contract verification for every registered hardware backend in EveryO.

Ensures that all built-in and optional backends strictly implement the
HardwareBackend contract without runtime errors, even when the underlying
hardware or runtime is not present on the host system.
"""

from __future__ import annotations

import pytest

from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)
from everyo.backends.registry import get_all_backends

ALL_REGISTERED_BACKENDS = list(get_all_backends().values())


@pytest.mark.parametrize("backend", ALL_REGISTERED_BACKENDS, ids=lambda b: b.name)
class TestBackendContractCompliance:
    def test_name_and_metadata(self, backend: HardwareBackend):
        assert isinstance(backend.name, str) and len(backend.name) > 0
        assert isinstance(backend.display_name, str) and len(backend.display_name) > 0
        assert isinstance(backend.vendor, str) and len(backend.vendor) > 0

    def test_availability_and_status(self, backend: HardwareBackend):
        is_avail = backend.is_available()
        assert isinstance(is_avail, bool)

        status = backend.status()
        assert isinstance(status, BackendStatus)

        reason = backend.unavailable_reason()
        if not is_avail:
            assert isinstance(reason, str) and len(reason) > 0
        else:
            assert reason is None

    def test_device_detection_never_raises(self, backend: HardwareBackend):
        devices = backend.detect_devices()
        assert isinstance(devices, list)
        for dev in devices:
            assert isinstance(dev, DeviceInfo)
            assert dev.backend == backend.name
            assert isinstance(dev.index, int) and dev.index >= 0
            assert isinstance(dev.name, str)

    def test_device_properties_and_memory(self, backend: HardwareBackend):
        props = backend.get_device_properties(0)
        assert isinstance(props, dict)

        mem = backend.get_device_memory(0)
        assert isinstance(mem, dict)
        assert "total_bytes" in mem
        assert "free_bytes" in mem
        assert "used_bytes" in mem
        assert isinstance(mem["total_bytes"], int)
        assert isinstance(mem["free_bytes"], int)
        assert isinstance(mem["used_bytes"], int)

    def test_capabilities_declaration(self, backend: HardwareBackend):
        caps = backend.get_capabilities()
        assert isinstance(caps, BackendCapabilities)
        assert isinstance(caps.supported_operations, frozenset)
        assert isinstance(caps.supported_dtypes, frozenset)
        assert isinstance(caps.supports_autograd, bool)
        assert isinstance(caps.supports_training, bool)
        assert isinstance(caps.supports_inference, bool)

    def test_synchronization_and_cleanup(self, backend: HardwareBackend):
        # Must be safe to call synchronize and release without raising
        assert backend.synchronize(0) is None
        assert backend.release(None) is None

    def test_describe_contract(self, backend: HardwareBackend):
        desc = backend.describe()
        assert isinstance(desc, dict)
        assert desc["name"] == backend.name
        assert desc["display_name"] == backend.display_name
        assert desc["vendor"] == backend.vendor
        assert isinstance(desc["available"], bool)
        assert isinstance(desc["status"], str)
        assert isinstance(desc["devices"], list)

    def test_is_operational_consistency(self, backend: HardwareBackend):
        is_op = backend.is_operational()
        assert isinstance(is_op, bool)
        if is_op:
            assert backend.is_available() is True
            assert len(backend.get_capabilities().supported_operations) > 0
        else:
            # If not operational, auto device selection must not choose it
            pass
