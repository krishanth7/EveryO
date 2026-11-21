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
