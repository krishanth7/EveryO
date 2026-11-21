# AMD ROCm & HIP Acceleration

EveryO supports AMD Radeon GPUs, Radeon PRO GPUs, and AMD Instinct accelerators through ROCm and the Heterogeneous-Compute Interface for Portability (HIP).

## Platform Compatibility

- **Supported OS**: Linux (Ubuntu 22.04/24.04, RHEL 8/9, SLES 15).
- **Windows**: ROCm is Linux-first. On Windows, AMD GPUs are accelerated via DirectML or Vulkan.
- **Supported Hardware**:
  - AMD Instinct MI100, MI200, MI300 series.
  - AMD Radeon PRO W6800, W7800, W7900.
  - AMD Radeon RX 6800, 6900, 7800, 7900 series.

## Detection

EveryO detects ROCm devices through `rocm-smi` and the `/opt/rocm` filesystem hierarchy:

```python
import everyo as eo

status = eo.backends.get_backend_status()["rocm"]
print(f"ROCm Status: {status['status']}")
```

## Current Status

Device detection is qualified. Compute kernels are tracked for Phase 3 integration via HIP C++ bindings.
