"""Detection of the optional EveryO CUDA extension.

The native extension (``everyo._everyo_cuda``) is built from the sources in the
top-level ``cuda/`` directory and is *never* required.  Everything in this
module answers the single question "can we run CUDA kernels right now?" without
raising, so that CPU-only installations behave normally.
"""

from __future__ import annotations

import functools
import importlib
import os
from types import ModuleType
from typing import Any

from everyo._logging import get_logger

__all__ = [
    "EXTENSION_NAME",
    "get_extension",
    "is_available",
    "device_count",
    "runtime_info",
    "unavailable_reason",
    "reset_cache",
]

_LOGGER = get_logger(__name__)

#: Import path of the compiled pybind11 extension.
EXTENSION_NAME = "everyo._everyo_cuda"

#: Setting this environment variable to "1" disables CUDA even when built.
DISABLE_ENV_VAR = "EVERYO_DISABLE_CUDA"

_UNAVAILABLE_REASON: str | None = "CUDA availability has not been probed yet."


@functools.lru_cache(maxsize=1)
def get_extension() -> ModuleType | None:
    """Import and return the CUDA extension module, or ``None``.

    The result is cached.  Import failures are logged at DEBUG level and never
    propagate: a missing extension is an expected, supported configuration.
    """
    global _UNAVAILABLE_REASON

    if os.environ.get(DISABLE_ENV_VAR, "") == "1":
        _UNAVAILABLE_REASON = f"CUDA disabled by the {DISABLE_ENV_VAR}=1 environment variable."
        return None
    try:
        module = importlib.import_module(EXTENSION_NAME)
    except ImportError as exc:
        _UNAVAILABLE_REASON = f"The native extension '{EXTENSION_NAME}' is not built ({exc})."
        _LOGGER.debug("CUDA extension unavailable: %s", exc)
        return None

    try:
        count = int(module.device_count())
    except Exception as exc:  # pragma: no cover - depends on driver state
        _UNAVAILABLE_REASON = f"The CUDA extension failed to query devices ({exc})."
        _LOGGER.warning("CUDA extension loaded but unusable: %s", exc)
        return None

    if count <= 0:
        _UNAVAILABLE_REASON = "No CUDA-capable device was reported by the driver."
        return None

    _UNAVAILABLE_REASON = None
    return module


def is_available() -> bool:
    """Return ``True`` when CUDA kernels can be executed."""
    return get_extension() is not None


def device_count() -> int:
    """Return the number of usable CUDA devices (``0`` when unavailable)."""
    module = get_extension()
    if module is None:
        return 0
    try:
        return int(module.device_count())
    except Exception:  # pragma: no cover - depends on driver state
        return 0


def unavailable_reason() -> str | None:
    """Explain why CUDA is unavailable, or ``None`` when it is available."""
    get_extension()
    return _UNAVAILABLE_REASON


def runtime_info() -> dict[str, Any]:
    """Return a JSON-serialisable description of the CUDA runtime state."""
    module = get_extension()
    info: dict[str, Any] = {
        "available": module is not None,
        "extension": EXTENSION_NAME,
        "device_count": device_count(),
        "reason": unavailable_reason(),
        "devices": [],
    }
    if module is None:
        return info
    try:
        info["runtime_version"] = module.runtime_version()
        info["devices"] = [module.device_properties(index) for index in range(info["device_count"])]
    except Exception as exc:  # pragma: no cover - depends on driver state
        info["reason"] = f"Failed to read device properties ({exc})."
    return info


def reset_cache() -> None:
    """Clear the cached import result.  Mainly useful in tests."""
    get_extension.cache_clear()
