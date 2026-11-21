# Intel oneAPI, SYCL & OpenVINO Acceleration

EveryO targets Intel Arc discrete GPUs, Intel Data Center GPU Flex/Max, Intel Iris Xe integrated graphics, and Intel Core Ultra NPUs.

## Technologies

- **Intel oneAPI**: Unified development environment for cross-architecture compute.
- **SYCL / DPC++**: Khronos standard C++ heterogeneous computing programming model.
- **Intel OpenVINO**: Inference optimization engine targeting Intel CPUs, GPUs, and NPUs.

## Detection

EveryO inspects `sycl-ls`, the Level Zero runtime (`ze_loader`), and the `openvino` Python package:

```python
import everyo as eo

intel_devices = eo.backends.detect_all().get("intel", [])
for dev in intel_devices:
    print(f"Intel Device: {dev.name}")
```

## Qualification Tier & Implementation Status

- **Qualification Tier**: **Device detection only** (Linux, Windows).
- **Compute Kernels**: **Not implemented**. oneAPI/SYCL kernels and OpenVINO inference bridge are planned for Phase 3.
- **Execution & Fallback Behavior**: Detection-only support does NOT execute compute operations. Requesting `intel`, `sycl`, or `openvino` safely falls back to CPU when `strict=False`, or raises `EveryODeviceError` when `strict=True`. It is never auto-selected by `eo.device("auto")`.
