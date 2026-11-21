# EveryO Hardware Acceleration Guide

EveryO is an open-source, multi-hardware machine learning and numerical computing framework designed to execute on whatever hardware developers already own: CPU, NVIDIA, AMD, Intel, Apple Silicon, Qualcomm Snapdragon, and portable runtimes (DirectML, Vulkan, OpenCL).

## Core Principle

**ONE FRAMEWORK — MULTIPLE BACKENDS — AUTOMATIC HARDWARE DETECTION — RELIABLE FALLBACK.**

EveryO does **NOT** require NVIDIA hardware, CUDA, or TensorFlow to function. It installs and runs on ordinary CPU-only computers with zero specialized drivers.

---

## Supported Platforms & Backends

| Vendor / Platform | Backend | Hardware Targets | OS | Current Status |
|---|---|---|---|---|
| **Generic / CPU** | `cpu` | x86-64 (SSE, AVX, AVX-512), ARM64 (NEON) | Linux, Windows, macOS | **Production Baseline** |
| **NVIDIA** | `cuda` | GeForce, Quadro, RTX, Tesla, Hopper | Linux, Windows | **Production (Optional C++)** |
| **AMD** | `rocm`, `hip` | Radeon RX, Radeon PRO, Instinct accelerators | Linux (ROCm 5+) | *Detection Qualified* |
| **Intel** | `intel`, `sycl`, `openvino` | Intel Arc GPUs, Data Center Max, Iris Xe, Core Ultra NPUs | Linux, Windows | *Detection Qualified* |
| **Microsoft Windows** | `directml` | DirectX 12-capable GPUs (AMD, Intel, NVIDIA, Qualcomm) | Windows 10/11 | *Detection Qualified* |
| **Apple** | `metal` | Apple Silicon (M1, M2, M3, M4), Metal 3 | macOS 12.0+ | *Detection Qualified* |
| **Cross-Vendor** | `vulkan` | Vulkan 1.2+ compliant GPUs | Linux, Windows, Android | *Detection Qualified* |
| **Cross-Vendor** | `opencl` | OpenCL 2.0+ compute devices | Linux, Windows, macOS | *Detection Qualified* |
| **Qualcomm** | `qnn` | Snapdragon X Elite, Snapdragon Mobile NPUs | Windows on ARM, Linux | *Detection Qualified* |
| **Multi-Provider** | `onnxruntime` | CPU, CUDA, DirectML, OpenVINO, CoreML providers | Cross-Platform | *Detection Qualified* |

---

## Unified Developer Experience

EveryO code is written once and runs across all available backends:

```python
import everyo as eo

# 1. Inspect detected devices
print(eo.backends.detect_all())

# 2. Select the optimal available accelerator automatically
device = eo.device("auto")
print(f"Executing on: {device}")

# 3. Build neural network
model = eo.Sequential(
    eo.Linear(128, 64),
    eo.ReLU(),
    eo.Linear(64, 10),
)

# 4. Place model parameters on device
model.to(device)

# 5. Differentiable execution with safe fallback
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
