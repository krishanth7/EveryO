# OpenCL Portable Acceleration

OpenCL (Open Computing Language) is an open, royalty-free standard for cross-platform parallel programming across multi-core CPUs, GPUs, DSPs, and FPGAs.

## Platforms

- Supported on Windows, Linux, and legacy macOS installations.
- Hardware: AMD, Intel, NVIDIA, and ARM Mali processors.
- Optional Python binding: `pyopencl` (`pip install pyopencl`).

## Detection

When `pyopencl` is installed, EveryO enumerates all OpenCL platforms and sub-devices:

```python
import everyo as eo

status = eo.backends.get_backend_status()["opencl"]
print(f"OpenCL: {status['status']}")
```

## Qualification Tier & Implementation Status

- **Qualification Tier**: **Device detection only** (Cross-platform).
- **Compute Kernels**: **Not implemented**. OpenCL C compute kernels are tracked for Phase 3.
- **Execution & Fallback Behavior**: Detection-only support does NOT execute compute operations. Requesting `opencl` safely falls back to CPU when `strict=False`, or raises `EveryODeviceError` when `strict=True`. It is never auto-selected by `eo.device("auto")`.
