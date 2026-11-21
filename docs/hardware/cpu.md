# CPU Acceleration & Mandatory Baseline

The CPU backend (`device="cpu"`) is EveryO's guaranteed numerical reference baseline. It is always available, requires no GPU or compiler toolchain, and provides numerical ground truth for all EveryO operations.

## Architecture

- **Engine**: Built upon NumPy and vector BLAS libraries (OpenBLAS, MKL, or Apple Accelerate depending on your environment).
- **Precision**: Full support for `float32`, `float64`, `int32`, `int64`, and `bool`.
- **Autograd**: Complete reverse-mode automatic differentiation engine.
- **Threading**: Configurable via `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`.

## Device Detection

```python
import everyo as eo

cpu_devices = eo.backends.detect_all()["cpu"]
for dev in cpu_devices:
    print(f"Device: {dev.name}, Memory: {dev.total_memory_bytes / (1024**3):.1f} GB")
```

## Operation Support

The CPU backend supports 100% of EveryO's standard operations:
- Basic arithmetic: `add`, `subtract`, `multiply`, `divide`, `power`, `negative`, `abs`.
- Linear algebra: `matmul`, `dot`, `transpose`, `reshape`, `flatten`.
- Reductions: `sum`, `mean`, `max`, `min`, `variance`.
- Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`.
- Convolutions: `conv2d`, `max_pool2d`, `avg_pool2d`.
- Normalization: `batch_norm`, `layer_norm`.

## Qualification Tier & Implementation Status

- **Qualification Tier**: **Implemented and tested** (Cross-platform).
- **Compute Kernels**: **Implemented** for all operations and standard dtypes (`float32`, `float64`, integers).
- **Fallback Role**: Primary reference baseline. The CPU backend never falls back to any other device. Every other backend falls back to CPU when unsupported operations, dtypes, or shapes are encountered.
