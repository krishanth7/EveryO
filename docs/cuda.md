# CUDA acceleration

CUDA is **optional**. EveryO runs fully on the CPU and every GPU feature
degrades gracefully when the hardware or toolchain is absent.

## Status

| Component | State |
| --- | --- |
| Kernel sources (`cuda/src`) | Available |
| pybind11 bindings | Available |
| CMake build | Available |
| Automatic CPU fallback | Available |
| Kernels used by tensors on a `cuda` device | Available (add, multiply, matmul, relu, sum) |
| GPU-resident tensors (no per-call transfer) | Planned |
| Fused kernels, mixed precision | Planned |

## Checking availability

```python
import everyo as eo

eo.cuda.is_available()        # True only with a built extension AND a device
eo.cuda.device_count()
eo.cuda.unavailable_reason()  # a sentence explaining why, or None
eo.cuda.runtime_info()        # full details, including device properties
```

None of these raise. On a CPU-only machine `everyo doctor` prints the reason.

## Building the extension

Requirements: an NVIDIA GPU, the CUDA Toolkit (nvcc), CMake 3.18+ and
pybind11. CUDA is supported only on NVIDIA's own supported platforms and
toolchains — there is no CUDA path on machines without NVIDIA hardware, and
EveryO does not pretend otherwise.

```bash
pip install -e ".[cuda]"     # installs pybind11
./scripts/build_cuda.sh
```

Or with CMake directly:

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --parallel
```

The build copies `_everyo_cuda*.so` next to the Python package so
`import everyo._everyo_cuda` resolves. To target a specific compute
capability:

```bash
./scripts/build_cuda.sh --arch=86
```

Clean up with `./scripts/build_cuda.sh --clean`.

## Using it

```python
import everyo as eo

x = eo.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
y = eo.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")

z = eo.matmul(x, y)     # runs the tiled CUDA kernel when available
```

`eo.device("auto")` picks CUDA when it is usable and CPU otherwise.

## Fallback rules

| Situation | Behaviour |
| --- | --- |
| `eo.device("cuda")`, no CUDA | Returns the CPU device, logs a warning |
| `eo.device("cuda", strict=True)`, no CUDA | Raises `EveryOCudaError` |
| `eo.tensor(..., device="cuda")`, no CUDA | Tensor is created on the CPU |
| `eo.cuda.relu(x)`, no CUDA | Computes with NumPy |
| `eo.cuda.relu(x, strict=True)`, no CUDA | Raises `EveryOCudaError` |
| `EVERYO_DISABLE_CUDA=1` | CUDA is treated as unavailable even when built |

Nothing crashes because CUDA is missing. The only way to get an error is to ask
for one with `strict=True`.

## The kernels

| Source | Kernel | Notes |
| --- | --- | --- |
| `vector_add.cu` | `VectorAdd`, `VectorMultiply` | One thread per element, bounds-checked |
| `relu.cu` | `Relu` | One thread per element |
| `matmul.cu` | `MatMul` | 16x16 shared-memory tiling |
| `reduction.cu` | `Sum` | Block-level tree reduction, host-side final sum |
| `device.cu` | Device queries | Never raises when no driver is present |

All of them use `EVERYO_CUDA_CHECK`, which turns a CUDA error code into a C++
exception carrying the file, line and failing call. `EVERYO_CUDA_CHECK_KERNEL`
also synchronises, so an asynchronous kernel fault surfaces at the launch site
instead of at some unrelated call later. Device memory is owned by an RAII
`DeviceBuffer`, so it is released even when a later call throws.

## Performance

**Do not assume the GPU is faster.** Each kernel call copies its inputs to the
device and the result back; for small problems that transfer dominates and the
CPU wins. Measure on your own hardware:

```bash
python benchmarks/benchmark_matmul.py --sizes 128 512 1024 2048
```

Removing per-call transfers by keeping tensors resident on the device is the
main planned optimisation, and is listed as planned rather than available.

## Verifying correctness

Kernel output is compared against the NumPy backend in
`tests/test_devices.py::TestCudaKernels`. Those tests skip automatically
without a GPU, and run for real with one:

```bash
pytest tests/test_devices.py -v
```
