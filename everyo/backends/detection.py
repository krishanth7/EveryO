"""Hardware detection utilities.

Thin wrappers that call into individual backends and aggregate the results.
Detection never raises for missing hardware or runtimes.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any

from everyo.backends.base import DeviceInfo

__all__ = [
    "detect_cpu_info",
    "detect_all_hardware",
]

_LOGGER = logging.getLogger(__name__)


def detect_cpu_info() -> dict[str, Any]:
    """Return information about the host CPU.

    Uses only the standard library — no optional packages.
    """
    info: dict[str, Any] = {
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "system": platform.system(),
        "python_version": platform.python_version(),
    }

    # Core / thread count
    try:
        info["physical_cores"] = os.cpu_count() or 0
    except Exception:
        info["physical_cores"] = 0
    info["logical_cores"] = os.cpu_count() or 0

    # Instruction set hints (best-effort)
    machine = platform.machine().lower()
    info["is_x86_64"] = machine in ("x86_64", "amd64", "x86")
    info["is_aarch64"] = machine in ("aarch64", "arm64")

    # On Linux, parse /proc/cpuinfo for model name and flags.
    if platform.system() == "Linux":
        info.update(_parse_proc_cpuinfo())
    elif platform.system() == "Windows":
        info.update(_detect_cpu_windows())
    elif platform.system() == "Darwin":
        info.update(_detect_cpu_macos())

    return info


def _parse_proc_cpuinfo() -> dict[str, Any]:
    """Extract CPU model and instruction flags from ``/proc/cpuinfo``."""
    result: dict[str, Any] = {}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    result.setdefault("model_name", line.split(":", 1)[1].strip())
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].strip().split()
                    result["cpu_flags"] = flags
                    result["has_sse"] = "sse" in flags
                    result["has_sse2"] = "sse2" in flags
                    result["has_avx"] = "avx" in flags
                    result["has_avx2"] = "avx2" in flags
                    result["has_avx512"] = any(f.startswith("avx512") for f in flags)
                    result["has_fma"] = "fma" in flags
                    break
    except Exception:
        pass
    return result


def _detect_cpu_windows() -> dict[str, Any]:
    """Read CPU info from the Windows registry or environment."""
    result: dict[str, Any] = {}
    # The PROCESSOR_IDENTIFIER environment variable is always set on Windows.
    proc_id = os.environ.get("PROCESSOR_IDENTIFIER", "")
    if proc_id:
        result["model_name"] = proc_id
    # Number of processors
    num = os.environ.get("NUMBER_OF_PROCESSORS", "")
    if num.isdigit():
        result["logical_cores"] = int(num)
    return result


def _detect_cpu_macos() -> dict[str, Any]:
    """Read CPU info from macOS sysctl."""
    import subprocess

    result: dict[str, Any] = {}
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            timeout=5,
        ).strip()
        result["model_name"] = out
    except Exception:
        pass
    return result


def detect_all_hardware() -> dict[str, Any]:
    """Run detection across CPU and all registered backends.

    Returns a dict with 'cpu' and one key per registered backend.
    """
    from everyo.backends.registry import detect_all

    hardware: dict[str, Any] = {
        "cpu": detect_cpu_info(),
    }
    hardware["backends"] = {
        name: [
            {
                "name": d.name,
                "device_type": d.device_type,
                "index": d.index,
                "vendor": d.vendor,
                "memory_bytes": d.total_memory_bytes,
            }
            for d in devices
        ]
        for name, devices in detect_all().items()
    }
    return hardware
