# NVIDIA CUDA Hardware Acceleration

EveryO includes a high-performance native C++/CUDA extension for NVIDIA GPUs, providing hardware acceleration for linear algebra, elementwise operations, and GPU-resident inference.

## Requirements

- **Operating System**: Linux (Ubuntu 20.04+, RHEL 8+) or Windows 10/11 (64-bit).
- **GPU Architecture**: NVIDIA Pascal, Turing, Ampere, Ada Lovelace, Hopper, Blackwell (Compute Capability 6.0+).
- **Toolkit**: NVIDIA CUDA Toolkit 11.8 or 12.x (`nvcc`).
- **Python**: Python 3.9–3.12 (or self-compiled on 3.15).
- **Build tool**: `pybind11>=2.11` and CMake.

## Building the CUDA Extension

On Linux:
```bash
./scripts/build_cuda.sh
```

On Windows:
```powershell
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --config Release
```

## Supported Kernels

The CUDA extension provides native float32 kernels:
- `vector_add`: 1D elementwise addition.
- `vector_multiply`: 1D elementwise multiplication.
- `matmul`: Shared-memory 2D tiled matrix multiplication.
- `relu`: Elementwise ReLU activation.
- `sum`: Parallel reduction sum.
- `ResidentTensor`: GPU-resident inference tensor avoiding host-device transfer roundtrips.

## Qualification Tier & Implementation Status

- **Qualification Tier**:
  - **Implemented and tested**: When physical NVIDIA GPU is attached and the extension is built, all five C++/CUDA kernels (`vector_add`, `vector_multiply`, `matmul`, `relu`, `sum`) and `ResidentTensor` are tested against CPU reference.
  - **Implemented but not tested on real hardware**: When physical NVIDIA GPU is absent (e.g. CI runners), boundary transfer counts and protocol adherence are verified via mocked extension tests in `tests/test_cuda_resident_transfers.py`.
- **Compute Kernels**: **Implemented** for `float32` arrays only.
- **Unsupported Features**: Non-float32 execution (`float64`, `int32`, `int64`) on CUDA is **Unsupported** by the kernels and safely routes to CPU NumPy unless `strict=True`, in which case it raises `EveryOCudaError`. Mismatched shapes without explicit expansion are also unsupported on CUDA.

## Strict vs. Fallback Mode

```python
import everyo as eo

# Graceful fallback (default): falls back to CPU if CUDA is unavailable
dev = eo.device("cuda", strict=False)

# Strict mode: raises EveryOCudaError if CUDA cannot be used
try:
    dev = eo.device("cuda", strict=True)
except eo.EveryOCudaError as exc:
    print(f"CUDA unavailable: {exc}")
```
