"""Device abstraction.

EveryO stores every tensor buffer in host (CPU) memory as a NumPy array.  The
device attached to a tensor selects *where operations execute*: ``"cpu"`` uses
the NumPy backend, while ``"cuda"`` routes supported kernels through the
compiled CUDA extension when it is present.

Requesting ``"cuda"`` on a machine without a working extension is not an error
by default: the device degrades to CPU and a warning is logged.  Code that
genuinely requires the GPU can opt into strict behaviour with
``resolve_device("cuda", strict=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from everyo._logging import get_logger
from everyo.exceptions import EveryOCudaError, EveryODeviceError

__all__ = ["Device", "device", "resolve_device", "DeviceLike", "VALID_DEVICE_TYPES"]

_LOGGER = get_logger(__name__)

VALID_DEVICE_TYPES = (
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


@dataclass(frozen=True)
class Device:
    """An execution device.

    Attributes:
        type: One of the values in :data:`VALID_DEVICE_TYPES`.
        index: Device ordinal.  Always ``0`` for CPU.
    """

    type: str = "cpu"
    index: int = 0

    def __post_init__(self) -> None:
        if self.type not in VALID_DEVICE_TYPES:
            raise EveryODeviceError(
                f"Unknown device type {self.type!r}. "
                f"Valid device types are: {', '.join(VALID_DEVICE_TYPES)}."
            )
        if not isinstance(self.index, int) or self.index < 0:
            raise EveryODeviceError(
                f"Device index must be a non-negative integer, got {self.index!r}."
            )
        if self.type == "cpu" and self.index != 0:
            raise EveryODeviceError("The CPU device only supports index 0.")

    @property
    def is_cuda(self) -> bool:
        """``True`` when this device executes on the GPU."""
        return self.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """``True`` when this device is the CPU."""
        return self.type == "cpu"

    @property
    def is_accelerator(self) -> bool:
        """``True`` when this device is any non-CPU accelerator."""
        return self.type != "cpu"

    def __str__(self) -> str:
        return self.type if self.type == "cpu" else f"{self.type}:{self.index}"

    def __repr__(self) -> str:
        return f"device('{self}')"


DeviceLike = Union[str, Device, None]

CPU = Device("cpu")


def _parse(spec: str) -> tuple[str, int]:
    """Split ``"cuda:1"`` into ``("cuda", 1)``."""
    text = spec.strip().lower()
    if ":" not in text:
        return text, 0
    name, _, index = text.partition(":")
    try:
        return name, int(index)
    except ValueError as exc:
        raise EveryODeviceError(
            f"Invalid device string {spec!r}. Expected a form like 'cuda:0'."
        ) from exc


def device(spec: DeviceLike = "cpu", *, strict: bool = False) -> Device:
    """Create a :class:`Device`, resolving ``"auto"`` and backend availability.

    Args:
        spec: ``"cpu"``, ``"cuda"``, ``"rocm"``, ``"intel"``, ``"directml"``,
            ``"metal"``, ``"auto"``, a :class:`Device`, or ``None``
            (equivalent to ``"cpu"``).
        strict: When ``True``, requesting an unavailable backend raises
            instead of falling back to the CPU.

    Returns:
        The resolved device.

    Raises:
        EveryODeviceError: If the specification cannot be parsed.
        EveryOCudaError: If ``strict`` is set and CUDA is unavailable.
    """
    if spec is None:
        return CPU
    if isinstance(spec, Device):
        name, index = spec.type, spec.index
    else:
        name, index = _parse(str(spec))

    if name == "gpu":  # friendly alias
        name = "cuda"

    # --- auto selection ---
    if name == "auto":
        return _auto_select_device()

    # --- CUDA-specific backward-compatible path ---
    if name == "cuda":
        from everyo.cuda.availability import is_available as cuda_is_available

        if not cuda_is_available():
            if strict:
                raise EveryOCudaError(
                    "CUDA was requested with strict=True but the EveryO CUDA "
                    "extension is not available. Build it with "
                    "'./scripts/build_cuda.sh' on a machine with the NVIDIA CUDA "
                    "Toolkit, or use device='cpu'."
                )
            _LOGGER.warning(
                "CUDA is not available; falling back to the CPU device. "
                "Run 'everyo doctor' for details."
            )
            return CPU
        return Device(name, index)

    # --- Other accelerator backends ---
    if name in VALID_DEVICE_TYPES and name != "cpu":
        if _is_backend_available(name):
            return Device(name, index)
        if strict:
            raise EveryODeviceError(
                f"Backend '{name}' was requested with strict=True but is not "
                f"available. Run 'everyo doctor' for details."
            )
        _LOGGER.warning(
            "Backend '%s' is not available; falling back to the CPU device. "
            "Run 'everyo doctor' for details.",
            name,
        )
        return CPU

    return Device(name, index)


def _auto_select_device() -> Device:
    """Select the best available device automatically."""
    # First try CUDA (backward compatible).
    from everyo.cuda.availability import is_available as cuda_is_available

    if cuda_is_available():
        return Device("cuda", 0)

    # Then try other operational accelerators through the registry.
    try:
        from everyo.backends.registry import _PRIORITY, get_available_backends

        candidates = sorted(
            get_available_backends().values(),
            key=lambda b: _PRIORITY.get(b.name, 0),
            reverse=True,
        )
        for backend in candidates:
            # Only select backends that are actually operational with compute kernels
            if backend.name != "cpu" and len(backend.get_capabilities().supported_operations) > 0:
                devices = backend.detect_devices()
                if devices:
                    return Device(backend.name, devices[0].index)
                return Device(backend.name, 0)
    except Exception:
        pass

    return CPU


def _is_backend_available(name: str) -> bool:
    """Check if a named backend is operational through the registry."""
    try:
        from everyo.backends.registry import get_backend

        backend = get_backend(name)
        return backend is not None and backend.is_operational()
    except Exception:
        return False


def resolve_device(spec: DeviceLike = "cpu", *, strict: bool = False) -> Device:
    """Alias of :func:`device` used internally for readability."""
    return device(spec, strict=strict)
