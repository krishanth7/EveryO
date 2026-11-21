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

## 3. Existing Hardware Implementations vs. Placeholders

| Backend | Status | Driver / Runtime | Compute Implementation | Qualification Status |
|---|---|---|---|---|
| **CPU** | **Production** | Python, NumPy, OS CPU APIs | All standard operations (Math, LinAlg, Reductions, NN, Autograd) | **Qualified Baseline** |
| **NVIDIA CUDA** | **Production** | CUDA Toolkit 11.8+, Driver 525+, pybind11 | C++/CUDA kernels for `add`, `multiply`, `matmul`, `relu`, `sum_all`, `ResidentTensor` | **Qualified (Optional)** |
| **AMD ROCm / HIP** | *Detection Only* | ROCm 5.x/6.x, `rocm-smi`, `hipconfig` | None (Stub) | *Unqualified (Pending Kernels)* |
| **Intel oneAPI** | *Detection Only* | oneAPI, DPC++, `sycl-ls` | None (Stub) | *Unqualified (Pending Kernels)* |
| **Intel OpenVINO** | *Detection Only* | `openvino` Python package | None (Stub) | *Unqualified (Pending Kernels)* |
| **DirectML** | *Detection Only* | DirectX 12, `onnxruntime-directml` | None (Stub) | *Unqualified (Pending Kernels)* |
| **Apple Metal** | *Detection Only* | macOS Metal.framework, MPS | None (Stub) | *Unqualified (Pending Kernels)* |
| **Vulkan Compute** | *Detection Only* | Vulkan SDK, `vulkaninfo` | None (Stub) | *Unqualified (Pending Kernels)* |
| **OpenCL** | *Detection Only* | OpenCL ICD, `pyopencl` | None (Stub) | *Unqualified (Pending Kernels)* |
| **Qualcomm QNN** | *Detection Only* | Qualcomm AI Engine SDK | None (Stub) | *Unqualified (Pending Kernels)* |
| **ONNX Runtime** | *Detection Only* | `onnxruntime` package | None (Stub) | *Unqualified (Pending Kernels)* |

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
