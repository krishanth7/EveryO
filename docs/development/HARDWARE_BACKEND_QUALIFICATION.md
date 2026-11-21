# EveryO Hardware Backend Qualification & Architecture Audit

## 1. Executive Summary

EveryO was originally architected as a CPU-first framework on NumPy with an optional NVIDIA CUDA C++ extension (`everyo._everyo_cuda`). Phase 2 transitions EveryO from a two-backend framework (`cpu`, `cuda`) into a vendor-independent, multi-hardware machine learning and numerical computing system.

The core design principle is:
**ONE FRAMEWORK — MULTIPLE BACKENDS — AUTOMATIC HARDWARE DETECTION — RELIABLE FALLBACK.**

EveryO strictly adheres to empirical truth in hardware acceleration:
- Detection of a hardware driver or SDK is never conflated with operational compute capability.
- Backends report `is_available() == True` only when drivers, runtimes, devices, and compute execution kernels are genuinely operational.
- Unsupported operations trigger safe, documented, numerically verified CPU fallback without corrupting tensor data or autograd graphs.

---

## 2. Complete Architecture Audit

### 2.1 Core Tensor & Execution Architecture
- **Tensor Storage (`everyo/core/tensor.py`)**:
  - `Tensor` wraps an `np.ndarray` buffer in host memory.
  - Every tensor has an immutable `_device: Device`.
  - Properties: `shape`, `dtype`, `ndim`, `size`, `requires_grad`, `grad`, `grad_node`.
  - Methods: `.to(device)`, `.cpu()`, `.cuda(strict=False)`, `.numpy()`.
- **Operator Dispatch (`everyo/backends/operator_backend.py`)**:
  - `dispatch(operation, device, *arrays)` evaluates `backend_for(device)`.
  - Protocol `OperatorBackend`: defines `supports()`, `execute()`, `unsupported_reason()`, `to_host()`, `from_host()`.
  - If a backend does not support an operation or input dtypes/shapes, `dispatch()` logs an observable fallback decision and dispatches to `NumpyOperatorBackend`, returning `DispatchResult(value, requested_backend, executed_backend, fallback_reason)`.
- **Autograd Engine (`everyo/core/autograd.py`)**:
  - Reverse-mode automatic differentiation.
  - Operations register backward closures in autograd `Node`.
  - Backward evaluation operates on host gradients and accumulates via `tensor.accumulate_grad(gradient)`.
- **Neural Network Modules (`everyo/nn/module.py`)**:
  - `Module.to(device)` traverses `_parameters` and child `_modules`, placing parameters on the target `Device`.
- **Optimizers (`everyo/optim/`)**:
  - Parameter updates execute in-place on parameter buffers (`parameter.data`).

### 2.2 Device Abstraction (`everyo/core/device.py`)
- `Device(type: str, index: int = 0)` validates against `VALID_DEVICE_TYPES`:
  `("cpu", "cuda", "rocm", "hip", "intel", "sycl", "openvino", "directml", "metal", "vulkan", "opencl", "qnn", "onnxruntime")`.
- `device(spec, strict=False)`:
  - Resolves `"auto"` by querying the backend registry for the highest-priority operational accelerator.
  - Resolves explicit strings (e.g. `"cuda:0"`, `"rocm"`, `"cpu"`).
  - When an explicit backend is unavailable:
    - If `strict=True`, raises `EveryODeviceError` or `EveryOCudaError`.
    - If `strict=False`, logs a warning and safely returns `Device("cpu", 0)`.

---

## 3. Existing Hardware Implementations vs. Qualification Tiers

EveryO classifies hardware backend capabilities across five explicit tiers:
1. **Implemented and tested**: Full compute kernel execution verified on real physical hardware with mathematical equivalence to reference implementations.
2. **Implemented but not tested on real hardware**: Compute kernels or transfer abstraction implemented and tested via architectural verification or simulated environments, but physical accelerator hardware was not attached during testing.
3. **Device detection only**: Driver, runtime, device querying, and metadata discovery are implemented and tested, but compute execution kernels are not yet integrated.
4. **Not implemented**: Backends or kernels that are planned but have no implementation in EveryO.
5. **Unsupported**: Combinations of OS, architectures, or datatypes that are explicitly rejected by vendor SDK constraints.

| Backend | Driver / Runtime | Compute Implementation | Qualification Tier | Fallback & Execution Policy |
|---|---|---|---|---|
| **CPU** | Python, NumPy, OS CPU APIs | All standard operations (Math, LinAlg, Reductions, NN, Autograd) | **Implemented and tested** | Primary baseline; always operational. |
| **NVIDIA CUDA** | CUDA Toolkit 11.8+, Driver 525+, pybind11 | C++/CUDA kernels for `add`, `multiply`, `matmul`, `relu`, `sum_all`, `ResidentTensor` (float32 only) | **Implemented and tested** (when compiled and GPU attached); **Implemented but not tested on real hardware** (transfer mock validation when GPU absent) | Clean CPU fallback for unbuilt extension or non-float32 dtypes (`strict=False`). Raises in strict mode. Non-float32 CUDA execution is **Unsupported**. |
| **AMD ROCm / HIP** | ROCm 5.x/6.x, `rocm-smi`, `hipconfig` on Linux | None (Pending Phase 3 HIP kernels) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Windows ROCm is **Unsupported**. Never auto-selected. |
| **Intel oneAPI** | oneAPI, DPC++, `sycl-ls`, Level Zero | None (Pending Phase 3 SYCL kernels) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Never auto-selected. |
| **Intel OpenVINO** | `openvino` Python package | None (Pending Phase 3 OpenVINO bridge) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Never auto-selected. |
| **DirectML** | DirectX 12, `onnxruntime-directml` | None (Pending Phase 3 DirectML graph kernels) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Non-Windows OS is **Unsupported**. Never auto-selected. |
| **Apple Metal** | macOS Metal.framework, MPS | None (Pending Phase 3 Metal shaders) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Non-macOS is **Unsupported**. Never auto-selected. |
| **Vulkan Compute** | Vulkan SDK, `vulkaninfo` | None (Pending Phase 3 SPIR-V shaders) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Merely having `vulkaninfo` does NOT enable compute. Never auto-selected. |
| **OpenCL** | OpenCL ICD, `pyopencl` | None (Pending Phase 3 OpenCL C kernels) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Never auto-selected. |
| **Qualcomm QNN** | Qualcomm AI Engine SDK | None (Pending Phase 3 QNN graph dispatch) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Never auto-selected. |
| **ONNX Runtime** | `onnxruntime` package | None (Pending Phase 3 graph bridge) | **Device detection only**; Compute is **Not implemented** | Execution safely falls back to CPU. Never auto-selected. |

---

## 4. Key Limitations and Reliability Risks

1. **Detection False Positives**:
   - Systems often have vendor tools installed (e.g. `vulkaninfo.exe` from GPU display drivers) without a corresponding Python compute binding or compiled shaders.
   - *Mitigation*: `is_available()` returns `True` only when compute operations can be dispatched. Merely discovering a device string sets `status = BackendStatus.NOT_IMPLEMENTED` with an explanatory `unavailable_reason`.
2. **Python 3.15 Compatibility**:
   - The current development environment uses Python 3.15.2. Most third-party GPU wheels (PyTorch ROCm, TensorFlow, OpenVINO) only publish binaries for Python 3.9–3.12/3.13.
   - *Mitigation*: Core EveryO has zero mandatory third-party vendor dependencies. Vendor bindings are isolated in optional try/except import paths.
3. **Silent Device Degradation**:
   - Developers must know when and why an operation ran on CPU instead of an accelerator.
   - *Mitigation*: `DispatchResult` records `fell_back: bool` and `fallback_reason: str`. `BackendManager` maintains an observable fallback log.
4. **Memory Leaks and Resource Cleanup**:
   - Accelerators requiring unmanaged device allocations must implement `release()` and `synchronize()`.
   - Default implementations provide explicit no-ops for CPU/GC-managed buffers, while asynchronous backends override them.
5. **Cross-Platform Memory Detection Differences**:
   - macOS virtual memory management does not expose `SC_AVPHYS_PAGES` via `os.sysconf`.
   - *Mitigation*: EveryO implements platform-tailored memory detection: `sysctl hw.memsize` combined with `vm_stat` (free + inactive + speculative pages) with dynamic Mach page sizes (4KB on Intel, 16KB on Apple Silicon) on macOS, `GlobalMemoryStatusEx` on Windows, and `sysconf` with `/proc/meminfo` on Linux.

---

## 5. Phase 2 Implementation Plan

1. **Refine `HardwareBackend` Contract (`everyo/backends/base.py`)**:
   - Add `is_operational()`, `get_device_memory()`, `supports()`, `unsupported_reason()`.
   - Ensure `HardwareBackend` fully conforms to `OperatorBackend`.
2. **Standardize Backend Statuses**:
   - Ensure detection-only backends report `BackendStatus.NOT_IMPLEMENTED` rather than `AVAILABLE`.
3. **Dynamic Selection in `BackendManager` (`everyo/backends/manager.py`)**:
   - Enforce that `auto` selection considers only operational backends (`len(supported_operations) > 0`).
   - Provide structured selection reporting (`get_selection_report()`).
4. **Benchmarking Suite (`everyo/benchmarking.py`)**:
   - Extend matmul and activation benchmarks across all operational backends with memory and overhead measurements.
5. **Contract & Execution Test Suites (`tests/`)**:
   - Add contract compliance tests across all backends.
   - Add execution validation tests (forward, backward, numerical tolerance).
   - Add mocked detection tests for CI environments without vendor hardware.
6. **Hardware Documentation (`docs/hardware/`)**:
   - Comprehensive guides for CPU, CUDA, ROCm, Intel, DirectML, Metal, Vulkan, OpenCL, QNN.
