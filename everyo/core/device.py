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

VALID_DEVICE_TYPES = ("cpu", "cuda")


@dataclass(frozen=True)
class Device:
    """An execution device.

    Attributes:
        type: Either ``"cpu"`` or ``"cuda"``.
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
    """Create a :class:`Device`, resolving ``"auto"`` and CUDA availability.

    Args:
        spec: ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"auto"``, a :class:`Device`,
            or ``None`` (equivalent to ``"cpu"``).
        strict: When ``True``, requesting CUDA without a usable CUDA extension
            raises instead of falling back to the CPU.

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

    # Imported lazily so that importing everyo.core.device never imports the
    # optional native extension.
    from everyo.cuda.availability import is_available as cuda_is_available

    if name == "auto":
        return Device("cuda", 0) if cuda_is_available() else CPU

    if name == "gpu":  # friendly alias
        name = "cuda"

    if name == "cuda" and not cuda_is_available():
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


def resolve_device(spec: DeviceLike = "cpu", *, strict: bool = False) -> Device:
    """Alias of :func:`device` used internally for readability."""
    return device(spec, strict=strict)
