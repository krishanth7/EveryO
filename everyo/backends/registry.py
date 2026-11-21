"""Backend registry with auto-discovery and priority-based selection.

The registry is the single source of truth for which backends exist and in
what order they should be tried.  Backends register themselves at import time
via :func:`register_backend`; the registry never imports vendor SDKs eagerly.
"""

from __future__ import annotations

import logging
from typing import Any

from everyo.backends.base import (
    BackendStatus,
    DeviceInfo,
    HardwareBackend,
)

__all__ = [
    "register_backend",
    "get_backend",
    "get_all_backends",
    "get_available_backends",
    "detect_all",
    "get_available_devices",
    "get_best_device",
    "get_backend_status",
    "get_device_info",
]

_LOGGER = logging.getLogger(__name__)

# Ordered dict: insertion order = default priority (higher priority first).
_REGISTRY: dict[str, HardwareBackend] = {}

# Backend priority order (higher number = higher priority when scores are equal).
_PRIORITY: dict[str, int] = {
    "cuda": 100,
    "rocm": 90,
    "hip": 85,
    "metal": 80,
    "directml": 75,
    "intel": 70,
    "sycl": 65,
    "openvino": 60,
    "vulkan": 50,
    "opencl": 45,
    "qnn": 40,
    "onnxruntime": 35,
    "arm": 20,
    "cpu": 10,
}


def register_backend(backend: HardwareBackend) -> None:
    """Register a backend instance.  Idempotent."""
    name = backend.name
    if name in _REGISTRY:
        _LOGGER.debug("Backend '%s' re-registered (replaced).", name)
    _REGISTRY[name] = backend
    _LOGGER.debug("Backend '%s' registered.", name)


def unregister_backend(name: str) -> None:
    """Remove a backend.  No-op if absent."""
    _REGISTRY.pop(name, None)


def get_backend(name: str) -> HardwareBackend | None:
    """Return the backend registered under ``name``, or ``None``."""
    return _REGISTRY.get(name)


def get_all_backends() -> dict[str, HardwareBackend]:
    """Return every registered backend (available or not)."""
    return dict(_REGISTRY)


def get_available_backends() -> dict[str, HardwareBackend]:
    """Return only backends whose :meth:`is_available` returns ``True``."""
    return {name: b for name, b in _REGISTRY.items() if b.is_available()}


def detect_all() -> dict[str, list[DeviceInfo]]:
    """Probe every registered backend and return detected devices.

    Returns:
        ``{backend_name: [DeviceInfo, ...]}`` for every backend that
        reports at least one device.  Backends that fail or find nothing
        are included with an empty list.
    """
    result: dict[str, list[DeviceInfo]] = {}
    for name, backend in _REGISTRY.items():
        try:
            result[name] = backend.detect_devices()
        except Exception as exc:
            _LOGGER.debug("Backend '%s' detection failed: %s", name, exc)
            result[name] = []
    return result


def get_available_devices() -> list[DeviceInfo]:
    """Return a flat list of all devices across all available backends."""
    devices: list[DeviceInfo] = []
    for backend in _REGISTRY.values():
        try:
            if backend.is_available():
                devices.extend(backend.detect_devices())
        except Exception as exc:
            _LOGGER.debug("Backend '%s' device enumeration failed: %s", backend.name, exc)
    return devices


def get_best_device() -> DeviceInfo | None:
    """Return the single best available device, or ``None`` if none found.

    Uses priority order, then total memory as a tiebreaker.
    """
    devices = get_available_devices()
    if not devices:
        return None
    return max(
        devices,
        key=lambda d: (
            _PRIORITY.get(d.device_type, 0),
            d.total_memory_bytes,
        ),
    )


def get_backend_status() -> dict[str, dict[str, Any]]:
    """Return a diagnostic summary for every registered backend."""
    result: dict[str, dict[str, Any]] = {}
    for name, backend in _REGISTRY.items():
        try:
            result[name] = {
                "available": backend.is_available(),
                "status": backend.status().value,
                "reason": backend.unavailable_reason(),
                "device_count": len(backend.detect_devices()),
            }
        except Exception as exc:
            result[name] = {
                "available": False,
                "status": BackendStatus.INITIALIZATION_FAILED.value,
                "reason": str(exc),
                "device_count": 0,
            }
    return result


def get_device_info(device_type: str | None = None) -> list[dict[str, Any]]:
    """Return serialisable info about detected devices.

    Args:
        device_type: Optional filter (e.g. ``'cuda'``, ``'cpu'``).
    """
    devices = get_available_devices()
    if device_type:
        devices = [d for d in devices if d.device_type == device_type]
    return [
        {
            "backend": d.backend,
            "device_type": d.device_type,
            "index": d.index,
            "name": d.name,
            "vendor": d.vendor,
            "driver_version": d.driver_version,
            "runtime_version": d.runtime_version,
            "total_memory_bytes": d.total_memory_bytes,
        }
        for d in devices
    ]
