# CPU Acceleration & Mandatory Baseline

The CPU backend (`device="cpu"`) is EveryO's guaranteed numerical reference baseline. It is always available, requires no GPU or compiler toolchain, and provides numerical ground truth for all EveryO operations.

## Architecture

- **Engine**: Built upon NumPy and vector BLAS libraries (OpenBLAS, MKL, or Apple Accelerate depending on your environment).
- **Precision**: Full support for `float32`, `float64`, `int32`, `int64`, and `bool`.
- **Autograd**: Complete reverse-mode automatic differentiation engine.
- **Threading**: Configurable via `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`.

## Device Detection

```python
import everyo as eo

cpu_devices = eo.backends.detect_all()["cpu"]
for dev in cpu_devices:
    print(f"Device: {dev.name}, Memory: {dev.total_memory_bytes / (1024**3):.1f} GB")
```

## Operation Support

The CPU backend supports 100% of EveryO's standard operations:
- Basic arithmetic: `add`, `subtract`, `multiply`, `divide`, `power`, `negative`, `abs`.
- Linear algebra: `matmul`, `dot`, `transpose`, `reshape`, `flatten`.
- Reductions: `sum`, `mean`, `max`, `min`, `variance`.
- Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`.
- Convolutions: `conv2d`, `max_pool2d`, `avg_pool2d`.
- Normalization: `batch_norm`, `layer_norm`.

## System Memory Detection & Diagnostics

EveryO provides zero-dependency physical memory diagnostics through `eo.backends.detect_system_memory()` and `cpu_backend.get_device_memory()`. The returned structure adheres to the schema:

```python
{
    "total_bytes": int,
    "free_bytes": int,
    "used_bytes": int,
}
```

### Platform-Specific Detection Methods

1. **Microsoft Windows**:
   - Total and available memory are obtained via `GlobalMemoryStatusEx` using the Win32 API through `ctypes`.
   - `ullTotalPhys` yields `total_bytes`; `ullAvailPhys` yields `free_bytes`.
2. **Linux & POSIX**:
   - Total memory is computed using `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")`.
   - Available memory queries `os.sysconf("SC_AVPHYS_PAGES")`.
   - If `SC_AVPHYS_PAGES` is unpopulated, EveryO parses `/proc/meminfo` (`MemAvailable` or `MemFree`).
3. **Apple macOS (Darwin)**:
   - Total physical memory is obtained directly from the kernel via `sysctl -n hw.memsize`.
   - Available memory is estimated via `vm_stat` Mach virtual memory statistics.
   - The VM page size is dynamically extracted from the `vm_stat` header (4,096 bytes on Intel x86_64, 16,384 bytes on Apple Silicon ARM64).
   - Available memory is calculated as `(Pages free + Pages inactive + Pages speculative) * page_size`.

### Why macOS Available Memory is an Estimate

Under Apple's XNU kernel virtual memory architecture:
- **Pages free**: RAM containing no data at all.
- **Pages inactive**: RAM holding cached data that has not been recently accessed. The operating system leaves files cached in inactive RAM for performance, but instantly reclaims these pages whenever an application requests memory allocations.
- **Pages speculative**: RAM pre-fetched by the kernel based on memory access patterns. Like inactive memory, speculative pages are immediately reclaimable under allocation pressure.
- Consequently, reporting solely `Pages free` on macOS severely underestimates usable memory (frequently reporting near-zero even on systems with gigabytes of reclaimable cache). Combining free, inactive, and speculative pages provides an honest, reliable metric of memory available for machine learning workloads.

### Invariants & Unsupported Reporting

- Invariant guarantees: `total_bytes >= 0`, `free_bytes >= 0`, `used_bytes >= 0`, and `used_bytes = total_bytes - free_bytes`.
- Non-fabrication guarantee: When a platform or command environment fails to provide memory metrics, EveryO reports honest zeros (`{"total_bytes": 0, "free_bytes": 0, "used_bytes": 0}`) and never invents or hardcodes arbitrary non-zero numbers.
- Accelerators that do not expose dedicated device allocations report explicit structured errors (`{"error": "..."}`) or zeros rather than masquerading host memory as VRAM.

## Qualification Tier & Implementation Status

- **Qualification Tier**: **Implemented and tested** (Cross-platform).
- **Compute Kernels**: **Implemented** for all operations and standard dtypes (`float32`, `float64`, integers).
- **Fallback Role**: Primary reference baseline. The CPU backend never falls back to any other device. Every other backend falls back to CPU when unsupported operations, dtypes, or shapes are encountered.
