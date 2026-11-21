"""Focused unit tests for cross-platform memory detection in EveryO.

Tests:
- macOS ARM64 memory detection with 16KB Mach VM pages.
- macOS Intel memory detection with 4KB Mach VM pages.
- sysctl hw.memsize parsing (-n flag and key: value syntax).
- vm_stat output parsing (free, inactive, speculative pages).
- Dynamic VM page sizes (4KB, 16KB, 64KB).
- Graceful handling of missing vm_stat.
- Graceful handling of failed sysctl.
- Malformed / corrupted command output.
- Zero or unavailable memory handling (no fabricated numbers).
- Invariant guarantees: used = total - free, never negative.
- Windows GlobalMemoryStatusEx cross-platform test double (runs on Linux, macOS, Windows).
- Windows API failure and missing windll handling.
- Linux sysconf and /proc/meminfo regression.
- Unsupported platform handling.
"""

from __future__ import annotations

import subprocess
import types
from unittest.mock import mock_open, patch

from everyo.backends.detection import (
    _detect_memory_linux,
    _detect_memory_macos,
    _detect_memory_windows,
    detect_system_memory,
)


class TestMacosMemoryDetection:
    """Verifies macOS native sysctl and vm_stat parsing across architectures."""

    def test_macos_arm64_memory_detection(self):
        """Simulates macOS on Apple Silicon (M-series, 16KB VM page size)."""
        fake_memsize = "17179869184\n"  # 16 GB
        fake_vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               24580.
Pages active:                            432100.
Pages inactive:                          198200.
Pages speculative:                        12340.
Pages throttled:                              0.
Pages wired down:                        210400.
Pages purgeable:                          34100.
"Translation faults":                123456789.
"""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sysctl":
                return subprocess.CompletedProcess(cmd, 0, stdout=fake_memsize, stderr="")
            if cmd[0] == "vm_stat":
                return subprocess.CompletedProcess(cmd, 0, stdout=fake_vm_stat, stderr="")
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            total, free = _detect_memory_macos()
            # 24580 + 198200 + 12340 = 235120 pages * 16384 bytes = 3852156928 bytes
            expected_free = (24580 + 198200 + 12340) * 16384
            assert total == 17179869184
            assert free == expected_free

            mem = detect_system_memory()
            assert mem["total_bytes"] == 17179869184
            assert mem["free_bytes"] == expected_free
            assert mem["used_bytes"] == 17179869184 - expected_free
            assert mem["used_bytes"] > 0

    def test_macos_intel_memory_detection(self):
        """Simulates macOS on Intel x86_64 (4KB VM page size)."""
        fake_memsize = "34359738368\n"  # 32 GB
        fake_vm_stat = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                              100000.
Pages active:                            500000.
Pages inactive:                          300000.
Pages speculative:                        50000.
Pages wired down:                        200000.
"""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sysctl":
                return subprocess.CompletedProcess(cmd, 0, stdout=fake_memsize, stderr="")
            if cmd[0] == "vm_stat":
                return subprocess.CompletedProcess(cmd, 0, stdout=fake_vm_stat, stderr="")
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            total, free = _detect_memory_macos()
            # 100000 + 300000 + 50000 = 450000 pages * 4096 = 1843200000 bytes
            expected_free = 450000 * 4096
            assert total == 34359738368
            assert free == expected_free

            mem = detect_system_memory()
            assert mem["total_bytes"] == 34359738368
            assert mem["free_bytes"] == expected_free
            assert mem["used_bytes"] == 34359738368 - expected_free

    def test_macos_sysctl_colon_syntax_fallback(self):
        """Verifies parsing of 'hw.memsize: <bytes>' when -n flag is omitted or ignored."""
        fake_colon_output = "hw.memsize: 8589934592\n"  # 8 GB

        def fake_run(cmd, **kwargs):
            if cmd == ["sysctl", "-n", "hw.memsize"]:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")
            if cmd == ["sysctl", "hw.memsize"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=fake_colon_output, stderr="")
            if cmd[0] == "vm_stat":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 1000.\n",
                    stderr="",
                )
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            total, free = _detect_memory_macos()
            assert total == 8589934592
            assert free == 1000 * 4096

    def test_macos_dynamic_page_size_64k(self):
        """Verifies custom VM page size (e.g. 64KB)."""
        fake_vm_stat = """Mach Virtual Memory Statistics: (page size of 65536 bytes)
Pages free:                                 100.
Pages inactive:                             200.
Pages speculative:                           50.
"""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sysctl":
                return subprocess.CompletedProcess(cmd, 0, stdout="1000000000\n", stderr="")
            if cmd[0] == "vm_stat":
                return subprocess.CompletedProcess(cmd, 0, stdout=fake_vm_stat, stderr="")
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            total, free = _detect_memory_macos()
            assert total == 1000000000
            assert free == 350 * 65536

    def test_macos_missing_vm_stat_handles_safely(self):
        """When vm_stat is absent, total is still detected and free defaults honestly to 0."""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sysctl":
                return subprocess.CompletedProcess(cmd, 0, stdout="17179869184\n", stderr="")
            if cmd[0] == "vm_stat":
                raise FileNotFoundError("vm_stat not found")
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            total, free = _detect_memory_macos()
            assert total == 17179869184
            assert free == 0

            mem = detect_system_memory()
            assert mem["total_bytes"] == 17179869184
            assert mem["free_bytes"] == 0
            assert mem["used_bytes"] == 17179869184

    def test_macos_failed_sysctl_handles_safely(self):
        """When sysctl fails completely, returns honest zero without crashing."""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sysctl":
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="permission denied")
            if cmd[0] == "vm_stat":
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.\n",
                    stderr="",
                )
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
            patch("os.sysconf", side_effect=ValueError, create=True),
        ):
            total, free = _detect_memory_macos()
            assert total == 0
            assert free == 0

            mem = detect_system_memory()
            assert mem["total_bytes"] == 0
            assert mem["free_bytes"] == 0
            assert mem["used_bytes"] == 0

    def test_macos_malformed_command_output(self):
        """Corrupted or non-numeric output returns 0 without raising exceptions."""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sysctl":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="garbage_output_not_a_number", stderr=""
                )
            if cmd[0] == "vm_stat":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="Corrupted header\nPages free: abc.\n", stderr=""
                )
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
            patch("os.sysconf", side_effect=ValueError, create=True),
        ):
            total, free = _detect_memory_macos()
            assert total == 0
            assert free == 0

    def test_free_never_exceeds_total(self):
        """If raw page stats sum to more than physical RAM, free is clamped to total."""

        def fake_run(cmd, **kwargs):
            if cmd[0] == "sysctl":
                return subprocess.CompletedProcess(cmd, 0, stdout="1000000\n", stderr="")  # 1MB
            if cmd[0] == "vm_stat":
                # Reports 100MB free
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 25000.\n",
                    stderr="",
                )
            raise FileNotFoundError(cmd[0])

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=fake_run),
        ):
            total, free = _detect_memory_macos()
            assert total == 1000000
            assert free == 1000000

            mem = detect_system_memory()
            assert mem["total_bytes"] == 1000000
            assert mem["free_bytes"] == 1000000
            assert mem["used_bytes"] == 0


class TestLinuxMemoryDetection:
    """Verifies Linux sysconf and /proc/meminfo parsing."""

    def test_linux_sysconf_detection(self):
        """Simulates standard Linux sysconf availability."""
        sysconf_map = {
            "SC_PAGE_SIZE": 4096,
            "SC_PHYS_PAGES": 4194304,  # 16 GB
            "SC_AVPHYS_PAGES": 1048576,  # 4 GB
        }

        with (
            patch("platform.system", return_value="Linux"),
            patch("os.sysconf_names", sysconf_map, create=True),
            patch("os.sysconf", side_effect=lambda name: sysconf_map[name], create=True),
        ):
            total, free = _detect_memory_linux()
            assert total == 17179869184
            assert free == 4294967296

            mem = detect_system_memory()
            assert mem["total_bytes"] == 17179869184
            assert mem["free_bytes"] == 4294967296
            assert mem["used_bytes"] == 17179869184 - 4294967296

    def test_linux_proc_meminfo_fallback(self):
        """When SC_AVPHYS_PAGES is unpopulated (0), falls back to /proc/meminfo."""
        sysconf_map = {
            "SC_PAGE_SIZE": 4096,
            "SC_PHYS_PAGES": 4194304,
            "SC_AVPHYS_PAGES": 0,
        }
        fake_meminfo = """MemTotal:       16384000 kB
MemFree:         2048000 kB
MemAvailable:    4096000 kB
Buffers:          500000 kB
Cached:          2000000 kB
"""

        with (
            patch("platform.system", return_value="Linux"),
            patch("os.sysconf_names", sysconf_map, create=True),
            patch("os.sysconf", side_effect=lambda name: sysconf_map[name], create=True),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=fake_meminfo)),
        ):
            total, free = _detect_memory_linux()
            assert total == 17179869184
            assert free == 4096000 * 1024


class TestWindowsMemoryDetection:
    """Verifies Windows ctypes GlobalMemoryStatusEx detection across all CI OSes."""

    def test_windows_global_memory_status_ex_success(self):
        """Simulates Windows GlobalMemoryStatusEx using a controlled fake Windows API double.

        This test runs portably across Linux, macOS, and Windows.
        """

        def fake_api(byref_stat):
            stat = byref_stat._obj
            stat.ullTotalPhys = 16000000000
            stat.ullAvailPhys = 6000000000
            return 1

        fake_kernel32 = types.SimpleNamespace(GlobalMemoryStatusEx=fake_api)
        fake_windll = types.SimpleNamespace(kernel32=fake_kernel32)

        with (
            patch("platform.system", return_value="Windows"),
            patch("ctypes.windll", fake_windll, create=True),
        ):
            total, free = _detect_memory_windows()
            assert total == 16000000000
            assert free == 6000000000

            mem = detect_system_memory()
            assert mem["total_bytes"] == 16000000000
            assert mem["free_bytes"] == 6000000000
            assert mem["used_bytes"] == 10000000000

    def test_windows_api_failure_returns_zero(self):
        """When GlobalMemoryStatusEx returns 0 (failure), returns honest zeros."""

        def fake_api(byref_stat):
            return 0  # Win32 failure return code

        fake_kernel32 = types.SimpleNamespace(GlobalMemoryStatusEx=fake_api)
        fake_windll = types.SimpleNamespace(kernel32=fake_kernel32)

        with (
            patch("platform.system", return_value="Windows"),
            patch("ctypes.windll", fake_windll, create=True),
        ):
            total, free = _detect_memory_windows()
            assert total == 0
            assert free == 0

            mem = detect_system_memory()
            assert mem["total_bytes"] == 0
            assert mem["free_bytes"] == 0
            assert mem["used_bytes"] == 0

    def test_windows_missing_windll_returns_zero(self):
        """When ctypes.windll is absent or None, returns honest zeros without crashing."""
        with (
            patch("platform.system", return_value="Windows"),
            patch("ctypes.windll", None, create=True),
        ):
            total, free = _detect_memory_windows()
            assert total == 0
            assert free == 0

            mem = detect_system_memory()
            assert mem["total_bytes"] == 0
            assert mem["free_bytes"] == 0
            assert mem["used_bytes"] == 0

    def test_windows_api_exception_handled_safely(self):
        """When GlobalMemoryStatusEx raises an exception, handles safely without crashing."""

        def fake_api(byref_stat):
            raise OSError("Win32 access violation")

        fake_kernel32 = types.SimpleNamespace(GlobalMemoryStatusEx=fake_api)
        fake_windll = types.SimpleNamespace(kernel32=fake_kernel32)

        with (
            patch("platform.system", return_value="Windows"),
            patch("ctypes.windll", fake_windll, create=True),
        ):
            total, free = _detect_memory_windows()
            assert total == 0
            assert free == 0

            mem = detect_system_memory()
            assert mem["total_bytes"] == 0
            assert mem["free_bytes"] == 0
            assert mem["used_bytes"] == 0


class TestUnsupportedPlatformMemoryDetection:
    """Verifies memory detection behavior on non-mainstream or unsupported platforms."""

    def test_unsupported_platform_does_not_crash(self):
        """Unrecognized operating systems return valid schema and non-negative integers."""
        with (
            patch("platform.system", return_value="FreeBSD"),
            patch("os.sysconf", side_effect=ValueError, create=True),
        ):
            mem = detect_system_memory()
            assert isinstance(mem, dict)
            assert mem["total_bytes"] >= 0
            assert mem["free_bytes"] >= 0
            assert mem["used_bytes"] >= 0
            assert mem["total_bytes"] == mem["free_bytes"] + mem["used_bytes"]

    def test_memory_values_non_negative_and_consistent(self):
        """Physical memory detection across all calls enforces non-negative arithmetic consistency."""
        mem = detect_system_memory()
        assert mem["total_bytes"] >= 0
        assert mem["free_bytes"] >= 0
        assert mem["used_bytes"] >= 0
        assert mem["total_bytes"] == mem["free_bytes"] + mem["used_bytes"]
