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
