"""Backend selection manager with configurable fallback policy.

The manager decides which backend executes a given operation.  It respects
explicit user choices, verifies compatibility, and falls back through the
priority chain until a working backend is found — or raises.
"""

from __future__ import annotations

import logging
from typing import Any

from everyo.backends.base import HardwareBackend
from everyo.backends.registry import (
    get_available_backends,
    get_backend,
)

__all__ = ["BackendManager", "get_manager"]

_LOGGER = logging.getLogger(__name__)

# Valid user-facing backend selection strings.
VALID_SELECTIONS: tuple[str, ...] = (
    "auto",
    "cpu",
    "cuda",
    "rocm",
    "hip",
    "intel",
    "sycl",
    "openvino",
    "directml",
    "metal",
    "vulkan",
    "opencl",
    "qnn",
    "onnxruntime",
)


class BackendManager:
    """Stateful backend selection with fallback and reporting.

    The manager is a singleton obtained via :func:`get_manager`.
    """

    def __init__(self) -> None:
        self._fallback_log: list[dict[str, Any]] = []

    def select_backend(
        self,
        requested: str = "auto",
        *,
        operation: str | None = None,
        strict: bool = False,
    ) -> HardwareBackend:
        """Pick the best backend for the request.

        Args:
            requested: ``"auto"`` or one of :data:`VALID_SELECTIONS`.
            operation: Optional operation name to verify support.
            strict: When ``True``, raise instead of falling back.

        Returns:
            The chosen :class:`HardwareBackend`.

        Raises:
            RuntimeError: When no backend can serve the request.
        """
        self._fallback_log.clear()

        if requested != "auto":
            backend = get_backend(requested)
            if backend is not None and backend.is_available():
                if operation is None or backend.supports_operation(operation):
                    return backend
                # Backend is available but does not support the operation.
                reason = f"Backend '{requested}' does not support operation '{operation}'."
                self._fallback_log.append({"requested": requested, "reason": reason})
                if strict:
                    raise RuntimeError(reason)
            else:
                reason = (
                    f"Backend '{requested}' is not available"
                    + (f": {backend.unavailable_reason()}" if backend else "")
                    + "."
                )
                self._fallback_log.append({"requested": requested, "reason": reason})
                if strict:
                    raise RuntimeError(reason)
            _LOGGER.info("Backend '%s' unavailable, trying fallback.", requested)

        # Auto-selection: iterate available backends in priority order.
        available = get_available_backends()
        # Sort by priority (the registry module defines the order).
        from everyo.backends.registry import _PRIORITY

        candidates = sorted(
            available.values(),
            key=lambda b: _PRIORITY.get(b.name, 0),
            reverse=True,
        )

        for backend in candidates:
            if operation and not backend.supports_operation(operation):
                self._fallback_log.append(
                    {
                        "skipped": backend.name,
                        "reason": f"Does not support operation '{operation}'.",
                    }
                )
                continue
            _LOGGER.debug("Selected backend '%s'.", backend.name)
            return backend

        # Last resort: CPU should always be in the registry.
        cpu = get_backend("cpu")
        if cpu is not None and cpu.is_available():
            self._fallback_log.append({"fallback": "cpu", "reason": "No accelerator available."})
            return cpu

        raise RuntimeError("No backend is available.  Ensure at least NumPy is installed.")

    @property
    def fallback_log(self) -> list[dict[str, Any]]:
        """Return the fallback decisions from the last :meth:`select_backend` call."""
        return list(self._fallback_log)


_MANAGER: BackendManager | None = None


def get_manager() -> BackendManager:
    """Return the singleton :class:`BackendManager`."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = BackendManager()
    return _MANAGER
