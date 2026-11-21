"""Computation backends.

* :mod:`everyo.backends.numpy_backend` is the always-available CPU reference.
* :mod:`everyo.backends.tensorflow_backend` is an optional integration used for
  cross-checking results and for hardware-accelerated reference training.

Importing this package never imports TensorFlow.

The new hardware-acceleration registry lives alongside the original API.
Every function that existed before is still here and behaves identically.
"""

from __future__ import annotations

from typing import Any

from everyo.backends import numpy_backend
from everyo.backends.base import (
    BackendCapabilities,
    BackendStatus,
    DeviceInfo,
    DeviceType,
    HardwareBackend,
)
from everyo.backends.manager import (
    BackendManager,
    get_manager,
)
from everyo.backends.operator_backend import (
    CudaOperatorBackend,
    DispatchResult,
    NumpyOperatorBackend,
    OperatorBackend,
    backend_for,
    dispatch,
)
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
)

__all__ = [
    # --- original public API (preserved) ---
    "numpy_backend",
    "available_backends",
    "describe_backends",
    "OperatorBackend",
    "NumpyOperatorBackend",
    "CudaOperatorBackend",
    "DispatchResult",
    "backend_for",
    "dispatch",
    # --- new multi-backend public API ---
    "HardwareBackend",
    "BackendCapabilities",
    "BackendManager",
    "BackendStatus",
    "DeviceInfo",
    "DeviceType",
    "get_manager",
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


def available_backends() -> list[str]:
    """Return the names of backends usable in this process.

    This is the original EveryO API; it now also includes hardware backends
    registered through the new registry.
    """
    names = ["numpy"]
    from everyo.backends import tensorflow_backend

    if tensorflow_backend.is_available():
        names.append("tensorflow")
    from everyo.cuda.availability import is_available as cuda_available

    if cuda_available():
        names.append("cuda")

    # Append any additional hardware backends that are available.
    for backend_name, backend in get_all_backends().items():
        if backend_name not in names and backend_name not in ("cpu",):
            try:
                if backend.is_available():
                    names.append(backend_name)
            except Exception:
                pass
    return names


def describe_backends() -> dict[str, Any]:
    """Return a JSON-serialisable description of every backend."""
    from everyo.backends import tensorflow_backend
    from everyo.cuda.availability import runtime_info

    result: dict[str, Any] = {
        "numpy": numpy_backend.describe(),
        "tensorflow": tensorflow_backend.describe(),
        "cuda": runtime_info(),
    }

    # Include new hardware backends.
    for name, backend in get_all_backends().items():
        if name not in result:
            try:
                result[name] = backend.describe()
            except Exception as exc:
                result[name] = {"name": name, "available": False, "error": str(exc)}
    return result


# ---------------------------------------------------------------------------
# Auto-register built-in backends on first import.
# ---------------------------------------------------------------------------


def _register_builtin_backends() -> None:
    """Import and register all built-in hardware backends."""
    from everyo.backends.cpu import CpuBackend
    from everyo.backends.cuda import CudaHardwareBackend

    register_backend(CpuBackend())
    register_backend(CudaHardwareBackend())

    # Optional backends — import errors are expected and silenced.
    _optional = [
        ("everyo.backends.rocm", "RocmBackend"),
        ("everyo.backends.hip", "HipBackend"),
        ("everyo.backends.intel", "IntelBackend"),
        ("everyo.backends.sycl", "SyclBackend"),
        ("everyo.backends.openvino", "OpenVinoBackend"),
        ("everyo.backends.directml", "DirectMlBackend"),
        ("everyo.backends.metal", "MetalBackend"),
        ("everyo.backends.vulkan", "VulkanBackend"),
        ("everyo.backends.opencl", "OpenClBackend"),
        ("everyo.backends.qnn", "QnnBackend"),
        ("everyo.backends.onnxruntime", "OnnxRuntimeBackend"),
    ]
    import importlib

    for module_path, class_name in _optional:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            register_backend(cls())
        except Exception:
            pass  # Backend not available — expected.


_register_builtin_backends()
