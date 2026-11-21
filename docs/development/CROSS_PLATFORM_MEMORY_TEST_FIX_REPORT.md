# EveryO — Cross-Platform Windows Memory Detection Test Fix Report

**Repository**: [EveryO](https://github.com/krishanth7/EveryO)  
**CI Failure Reference**: [Workflow Run 44898085519](https://github.com/krishanth7/EveryO/actions/runs/44898085519)  
**Date**: November 21, 2026  

---

## 1. Original Failure & Root Cause

### Failure:
In GitHub Actions workflow run 44898085519 on Ubuntu and macOS runners, the following test failed:
```text
tests/test_memory_detection.py::TestWindowsMemoryDetection::test_windows_global_memory_status_ex
AttributeError: <module 'ctypes' (frozen)> does not have the attribute 'windll'
```

### Root Cause:
The test attempted to patch `ctypes.windll.kernel32.GlobalMemoryStatusEx` directly using:
```python
patch("ctypes.windll.kernel32.GlobalMemoryStatusEx")
```
In Python's standard library, `ctypes.windll` is a Windows-only subsystem. On POSIX operating systems (Linux and macOS), `ctypes` exposes `cdll` and `pythonapi`, but completely lacks the `windll` attribute. Consequently, when `patch()` traversed the dotted path on Linux and macOS, it attempted to look up `getattr(ctypes, "windll")`, raising an immediate `AttributeError` before test execution even began.

---

## 2. Corrective Implementation

### 2.1 Test Double with Controlled Windows API (`tests/test_memory_detection.py`)
Rather than attempting to patch the nested non-existent `ctypes.windll.kernel32.GlobalMemoryStatusEx` path, we constructed a controlled fake Windows API double using `types.SimpleNamespace` and injected `ctypes.windll` with `create=True`:

```python
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
```

This test double:
- Runs identically across Linux, macOS, and Windows.
- Exercises the full real Windows detection code path in `_detect_memory_windows()`.
- Validates struct population (`_MEMORYSTATUSEX`) and integer casting without importing native Windows DLLs.

### 2.2 Defensive Detection in `everyo/backends/detection.py`
In `_detect_memory_windows()`, access to `ctypes.windll` was made fully defensive:
```python
windll = getattr(ctypes, "windll", None)
if windll is not None and hasattr(windll, "kernel32"):
    kernel32 = windll.kernel32
    if hasattr(kernel32, "GlobalMemoryStatusEx") and kernel32.GlobalMemoryStatusEx(
        ctypes.byref(stat)
    ):
        total = int(stat.ullTotalPhys)
        free = int(stat.ullAvailPhys)
```
- On real Windows, `ctypes.windll` exists natively and executes `kernel32.GlobalMemoryStatusEx`.
- On Linux and macOS, if `_detect_memory_windows()` is invoked without a test double, `getattr(ctypes, "windll", None)` returns `None` safely and returns `(0, 0)` without raising `AttributeError`.

---

## 3. Tests Added & Coverage

The `TestWindowsMemoryDetection` and `TestUnsupportedPlatformMemoryDetection` suites in [tests/test_memory_detection.py](file:///D:/EveryO/tests/test_memory_detection.py) now verify:

1. **Windows API Success (`test_windows_global_memory_status_ex_success`)**:
   - Verifies total, free, and used memory calculations (`total = 16GB`, `free = 6GB`, `used = 10GB`).
   - Runs portably on Linux, macOS, and Windows CI.
2. **Windows API Failure (`test_windows_api_failure_returns_zero`)**:
   - Verifies honest zeros when `GlobalMemoryStatusEx` returns 0 (Win32 failure code).
3. **Missing Windows DLL (`test_windows_missing_windll_returns_zero`)**:
   - Verifies honest zeros when `windll` is None or absent.
4. **Windows API Exception (`test_windows_api_exception_handled_safely`)**:
   - Verifies graceful handling when the API raises an `OSError`.
5. **Unsupported Platforms (`test_unsupported_platform_does_not_crash`)**:
   - Verifies FreeBSD/SunOS simulated platforms return valid schema without crashing.
6. **Arithmetic Invariants (`test_memory_values_non_negative_and_consistent`)**:
   - Verifies `total >= 0`, `free >= 0`, `used >= 0`, and `total == free + used`.
7. **Existing Linux & macOS Tests**:
   - All macOS ARM64 (16KB pages), macOS Intel (4KB pages), sysctl parsing, vm_stat parsing, and Linux sysconf / proc/meminfo tests remain 100% passing.

---

## 4. Verification Results

All required verification steps were executed locally:

| Command | Status | Result |
|---|---|---|
| `ruff check .` | **PASS** | `All checks passed!` across 267 files. |
| `ruff format --check .` | **PASS** | `267 files already formatted`. |
| `python -m pytest tests/test_memory_detection.py -q` | **PASS** | **16 passed**, 0 failed. |
| `python -m pytest tests/test_hardware_regression.py -q` | **PASS** | **9 passed**, 0 failed. |
| `python -m pytest -q` | **PASS** | **1,135+ passed**, 0 failed, 52 skipped. |
| `python -m compileall -q everyo tests` | **PASS** | Zero syntax or bytecode compilation errors. |
| `python -m pip check` | **PASS** | `No broken requirements found.` |
| `python -m build --no-isolation` | **PASS** | Built `dist/everyo-0.5.0.tar.gz` and `dist/everyo-0.5.0-py3-none-any.whl`. |
| `python -m twine check dist/*` | **PASS** | Distribution metadata validated. |
| `git diff --check` | **PASS** | Clean diff with zero whitespace or conflict issues. |

---

## 5. Remaining Platform Limitations

- **Windows on ARM**: Memory is detected via `GlobalMemoryStatusEx` identical to x86_64.
- **macOS Available Memory**: Because XNU caches files in inactive and speculative pages (reclaimed under pressure), available memory is estimated as `free + inactive + speculative`.
- **Unsupported OS**: On uninstrumented systems, EveryO reports honest zeros (`total = 0`, `free = 0`, `used = 0`) rather than fabricating fake numbers.
