"""Hardware detection utilities.

Thin wrappers that call into individual backends and aggregate the results.
Detection never raises for missing hardware or runtimes.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
from typing import Any

__all__ = [
    "detect_cpu_info",
    "detect_system_memory",
    "detect_all_hardware",
]

_LOGGER = logging.getLogger(__name__)


def _detect_memory_windows() -> tuple[int, int]:
    """Detect total and available memory on Windows via GlobalMemoryStatusEx."""
    total = 0
    free = 0
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total = int(stat.ullTotalPhys)
            free = int(stat.ullAvailPhys)
    except Exception as exc:
        _LOGGER.debug("Windows GlobalMemoryStatusEx failed: %s", exc)
    return total, free


def _detect_memory_macos() -> tuple[int, int]:
    """Detect total and available physical memory on macOS (Darwin).

    - Total physical memory is obtained from sysctl hw.memsize.
    - Available memory is estimated from vm_stat by extracting the reported
      Mach VM page size (4KB on Intel, 16KB on Apple Silicon ARM64) and summing
      unallocated (free), cached/reclaimable (inactive), and pre-fetched (speculative) pages.

    Returns:
        tuple[int, int]: (total_bytes, free_bytes).
    """
    total = 0
    free = 0

    # 1. Total physical memory via sysctl hw.memsize
    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip().isdigit():
            total = int(res.stdout.strip())
        else:
            # Fallback: sysctl without -n (outputs 'hw.memsize: <bytes>')
            res = subprocess.run(
                ["sysctl", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and ":" in res.stdout:
                val = res.stdout.partition(":")[2].strip()
                if val.isdigit():
                    total = int(val)
    except Exception as exc:
        _LOGGER.debug("macOS sysctl hw.memsize failed: %s", exc)

    if total <= 0:
        try:
            if (
                hasattr(os, "sysconf")
                and "SC_PAGE_SIZE" in os.sysconf_names
                and "SC_PHYS_PAGES" in os.sysconf_names
            ):
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                total = int(page_size * int(os.sysconf("SC_PHYS_PAGES")))
        except Exception:
            pass

    # 2. VM page statistics via vm_stat
    try:
        vm_res = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if vm_res.returncode == 0 and vm_res.stdout:
            page_size = 4096  # default fallback
            m = re.search(r"page size of (\d+) bytes", vm_res.stdout, re.IGNORECASE)
            if m:
                page_size = int(m.group(1))
            elif hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
                page_size = int(os.sysconf("SC_PAGE_SIZE"))

            pages_free = 0
            pages_inactive = 0
            pages_speculative = 0

            for line in vm_res.stdout.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip().lower().strip('"')
                raw_num = val.strip().rstrip(".")
                if not raw_num.isdigit():
                    continue
                num = int(raw_num)
                if key == "pages free":
                    pages_free = num
                elif key == "pages inactive":
                    pages_inactive = num
                elif key == "pages speculative":
                    pages_speculative = num

            avail_pages = pages_free + pages_inactive + pages_speculative
            if avail_pages > 0 and page_size > 0 and total > 0:
                free = min(avail_pages * page_size, total)
            else:
                free = 0
    except Exception as exc:
        _LOGGER.debug("macOS vm_stat parsing failed: %s", exc)

    return total, free


def _detect_memory_linux() -> tuple[int, int]:
    """Detect total and available memory on Linux."""
    total = 0
    free = 0
    if (
        hasattr(os, "sysconf")
        and "SC_PAGE_SIZE" in os.sysconf_names
        and "SC_PHYS_PAGES" in os.sysconf_names
    ):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = int(page_size * int(os.sysconf("SC_PHYS_PAGES")))
        if "SC_AVPHYS_PAGES" in os.sysconf_names:
            free = int(page_size * int(os.sysconf("SC_AVPHYS_PAGES")))

    # Fallback to /proc/meminfo if free or total is unpopulated
    if (free <= 0 or total <= 0) and os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:") and free <= 0:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            free = int(parts[1]) * 1024
                    elif line.startswith("MemTotal:") and total <= 0:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            total = int(parts[1]) * 1024
        except Exception as exc:
            _LOGGER.debug("/proc/meminfo parsing failed: %s", exc)

    return total, free


def detect_system_memory() -> dict[str, int]:
    """Return physical memory statistics (total, free, used) in bytes.

    Platform-specific detection:
    - Windows: GlobalMemoryStatusEx via ctypes.
    - macOS (Darwin): Total physical memory via sysctl hw.memsize,
      available memory estimated via vm_stat (free + inactive + speculative pages)
      using the dynamically reported Mach page size (4KB on Intel, 16KB on Apple Silicon).
    - Linux: os.sysconf with /proc/meminfo fallback.
    - Generic POSIX: os.sysconf (SC_PAGE_SIZE, SC_PHYS_PAGES, SC_AVPHYS_PAGES).
    """
    total = 0
    free = 0
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            total, free = _detect_memory_windows()
        elif sys_name == "Darwin":
            total, free = _detect_memory_macos()
        elif sys_name == "Linux":
            total, free = _detect_memory_linux()
        elif (
            hasattr(os, "sysconf")
            and "SC_PAGE_SIZE" in os.sysconf_names
            and "SC_PHYS_PAGES" in os.sysconf_names
        ):
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total = int(page_size * int(os.sysconf("SC_PHYS_PAGES")))
            if "SC_AVPHYS_PAGES" in os.sysconf_names:
                free = int(page_size * int(os.sysconf("SC_AVPHYS_PAGES")))
    except Exception as exc:
        _LOGGER.debug("detect_system_memory failed: %s", exc)

    total = max(0, total)
    free = max(0, min(free, total) if total > 0 else 0)
    used = max(0, total - free)
    return {"total_bytes": total, "free_bytes": free, "used_bytes": used}


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

    mem = detect_system_memory()
    info["total_memory_bytes"] = mem["total_bytes"]
    info["free_memory_bytes"] = mem["free_bytes"]

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
