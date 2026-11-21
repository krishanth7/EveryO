# EveryO Hardware Acceleration Guide

EveryO is an open-source, multi-hardware machine learning and numerical computing framework designed to execute on whatever hardware developers already own: CPU, NVIDIA, AMD, Intel, Apple Silicon, Qualcomm Snapdragon, and portable runtimes (DirectML, Vulkan, OpenCL).

## Core Principle

**ONE FRAMEWORK — MULTIPLE BACKENDS — AUTOMATIC HARDWARE DETECTION — RELIABLE FALLBACK.**

EveryO does **NOT** require NVIDIA hardware, CUDA, or TensorFlow to function. It installs and runs on ordinary CPU-only computers with zero specialized drivers.

---

## Hardware Qualification Classification

To ensure complete technical truth, every backend is categorized into one of five rigorous qualification tiers:

1. **Implemented and tested**: Full compute kernel execution verified on real physical hardware with mathematical equivalence to reference implementations.
2. **Implemented but not tested on real hardware**: Compute kernels or transfer abstraction implemented and tested via architectural verification or simulated environments, but physical accelerator hardware was not attached during testing.
3. **Device detection only**: Driver, runtime, device querying, and metadata discovery are implemented and tested, but compute execution kernels are not yet integrated.
4. **Not implemented**: Backends or kernels that are planned but have no implementation in EveryO.
5. **Unsupported**: Combinations of OS, architectures, or datatypes that are explicitly rejected by vendor SDK constraints.

---

## Backend Capability Matrix

| Vendor / Platform | Backend | Hardware Targets | Supported OS | Qualification Status | Compute Capability | Default / Fallback Policy |
|---|---|---|---|---|---|---|
| **Generic / CPU** | `cpu` | x86-64, ARM64 | Linux, Windows, macOS | **Implemented and tested** | Full (Math, LinAlg, Reductions, NN, Autograd) | Primary reference baseline. Never falls back. |
| **NVIDIA** | `cuda` | GeForce, RTX, Tesla, Hopper | Linux, Windows | **Implemented and tested** (when compiled and GPU attached); **Implemented but not tested on real hardware** (transfer mock validation when GPU absent) | C++/CUDA kernels for `add`, `multiply`, `matmul`, `relu`, `sum_all`, `ResidentTensor` (float32 only) | Falls back cleanly to CPU if unbuilt or if non-float32 dtype is passed (`strict=False`). Raises in strict mode. |
| **AMD** | `rocm`, `hip` | Radeon RX, Radeon PRO, Instinct | Linux only | **Device detection only** | **Not implemented** (Pending Phase 3 HIP kernels) | Detection only. Execution safely falls back to CPU. Windows ROCm is **Unsupported**. |
| **Intel** | `intel`, `sycl`, `openvino` | Intel Arc GPUs, Iris Xe, Core Ultra NPUs | Linux, Windows | **Device detection only** | **Not implemented** (Pending Phase 3 SYCL kernels) | Detection only. Execution safely falls back to CPU. |
| **Microsoft** | `directml` | DirectX 12 GPUs (AMD, Intel, NVIDIA, Qualcomm) | Windows 10/11 only | **Device detection only** | **Not implemented** (Pending Phase 3 DirectML graph kernels) | Detection only. Execution safely falls back to CPU. Non-Windows OS is **Unsupported**. |
| **Apple** | `metal` | Apple Silicon (M1–M4), MPS | macOS 12+ only | **Device detection only** | **Not implemented** (Pending Phase 3 Metal shaders) | Detection only. Execution safely falls back to CPU. Non-macOS is **Unsupported**. |
| **Khronos** | `vulkan` | Vulkan 1.2+ compliant GPUs | Linux, Windows, Android | **Device detection only** | **Not implemented** (Pending Phase 3 SPIR-V compute shaders) | Detection only. Execution safely falls back to CPU. Merely having `vulkaninfo` on PATH does NOT enable compute. |
| **Khronos** | `opencl` | OpenCL 2.0+ compute devices | Linux, Windows, macOS | **Device detection only** | **Not implemented** (Pending Phase 3 OpenCL C kernels) | Detection only. Execution safely falls back to CPU. |
| **Qualcomm** | `qnn` | Snapdragon X Elite, Hexagon NPU | Windows on ARM, Linux | **Device detection only** | **Not implemented** (Pending Phase 3 QNN graph dispatch) | Detection only. Execution safely falls back to CPU. |
| **Multi-Provider** | `onnxruntime` | CPU, CUDA, DirectML, CoreML | Cross-Platform | **Device detection only** | **Not implemented** (Pending Phase 3 graph bridge) | Detection only. Execution safely falls back to CPU. |

---

## CPU Fallback Behavior & Selection Rules

### 1. Automatic Selection (`eo.device("auto")`)
`eo.device("auto")` selects the highest-priority accelerator that is **genuinely operational** (`backend.is_operational() and len(supported_operations) > 0`).
- Detection-only backends (`rocm`, `intel`, `directml`, `metal`, `vulkan`, `opencl`, `qnn`, `onnxruntime`) are **never** selected by `eo.device("auto")`.
- If no operational accelerator is available (or CUDA native extension is not built), `eo.device("auto")` safely resolves to `Device("cpu", 0)`.

### 2. Explicit Backend Selection
- **Non-Strict Mode (Default, `strict=False`)**:
  Requesting an unavailable or detection-only device (e.g. `eo.device("vulkan")` or `eo.device("cuda")` when unbuilt) logs a warning and returns `Device("cpu", 0)`.
- **Strict Mode (`strict=True`)**:
  Requesting an unavailable or detection-only device with `strict=True` raises an explicit error:
  - `eo.EveryOCudaError` for CUDA requests.
  - `eo.EveryODeviceError` for all other hardware backends.

### 3. Observable Operator Fallback
If an operation is invoked on a device that does not support the operation or the input dtype/shape, EveryO executes the operation on the CPU reference backend and records an observable `DispatchResult`:
```python
from everyo.backends import get_last_dispatch

# Dispatch an operation
result = eo.add(a, b)

# Inspect fallback decision
dispatch_info = get_last_dispatch()
if dispatch_info and dispatch_info.fell_back:
    print(f"Fell back from {dispatch_info.requested_backend} to {dispatch_info.executed_backend}")
    print(f"Reason: {dispatch_info.fallback_reason}")
```

---

## Unified Developer Workflow

```python
import everyo as eo

# 1. Discover hardware (detection across all vendors)
devices = eo.backends.detect_all()
print("Detected devices:", devices)

# 2. Select the optimal available accelerator automatically
# Resolves to CUDA if built with GPU, otherwise safely resolves to CPU
device = eo.device("auto")
print(f"Executing on: {device}")

# 3. Build neural network
model = eo.Sequential(
    eo.Linear(128, 64),
    eo.ReLU(),
    eo.Linear(64, 10),
)
model.to(device)

# 4. Differentiable execution
x = eo.randn(32, 128, device=device)
output = model(x)
loss = eo.sum(output)
loss.backward()
```

---

## Detailed Hardware Guides

- [CPU Acceleration & Baseline](cpu.md)
- [NVIDIA CUDA Integration](nvidia-cuda.md)
- [AMD ROCm & HIP](amd-rocm.md)
- [Intel oneAPI, SYCL & OpenVINO](intel.md)
- [Microsoft DirectML](directml.md)
- [Apple Metal & MPS](apple-metal.md)
- [Vulkan Compute](vulkan.md)
- [OpenCL Portable Acceleration](opencl.md)
- [Qualcomm QNN & Snapdragon](qualcomm-qnn.md)
