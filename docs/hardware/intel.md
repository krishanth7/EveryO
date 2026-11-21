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

## Qualification Status

Detection and runtime discovery are qualified. Compute kernels and OpenVINO inference bridge are tracked for Phase 3.
