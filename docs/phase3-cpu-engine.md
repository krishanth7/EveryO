# Phase 3 — High-performance CPU engine

This phase adds explicit, opt-in CPU primitives. It does **not** change the
existing automatic differentiation path or promise universal speedups.

| Stage | Implementation | Limitation |
| --- | --- | --- |
| 3.1 | `ArrayPool` caches scratch buffers with a byte cap and lease checks | Only reuse after the caller releases an array; reused storage is uninitialized by default |
| 3.2 | `zero_copy_view` validates actual sharing with NumPy | Noncontiguous reshapes that require a copy raise |
| 3.3 | `parallel_map_chunks` splits by row and preserves result ordering | NumPy work must be pure; threads help only when the underlying kernels release the GIL |
| 3.4 | `vectorized_axpy` applies a shape-checked array kernel | Strict float32/float64 and matching shapes |
| 3.5 | `fused_affine_relu` applies bias and activation in the output of BLAS matmul | Inference only; no graph node for backward |
| 3.6 | `OptionalCppAxpy` invokes a separately compiled C++ library | No compilation on import; user opts in with a trusted local library |
| 3.7 | `profile_cpu` records repeated timings and output identity | Wall clock depends on machine load and BLAS threads |
| 3.8 | `check_cpu_regression` validates like-for-like checksums/environment and slowdown | Set thresholds from repeated controlled baseline runs |

## Usage

```python
import numpy as np
from everyo.cpu_engine import ArrayPool, fused_affine_relu, parallel_map_chunks

pool = ArrayPool(max_cached_bytes=1024 * 1024)
scratch = pool.acquire((32, 32), np.float64, zero=True)
pool.release(scratch)

x = np.ones((2048, 32), dtype=np.float64)
w = np.eye(32, dtype=np.float64)
b = np.zeros(32, dtype=np.float64)
y = fused_affine_relu(x, w, b)
mapped = parallel_map_chunks(y, lambda block: block * 2, workers=2)
```

## Optional C++ acceleration

Linux example (requires a local C++ compiler):

```bash
c++ -O3 -std=c++17 -fPIC -shared cpu_cpp/everyo_axpy.cpp -o cpu_cpp/libeveryo_axpy.so
```

```python
from everyo.cpu_engine import OptionalCppAxpy

native = OptionalCppAxpy("cpu_cpp/libeveryo_axpy.so")
result = native.axpy(x, x, 0.5)  # x must be float64
```

Never load an untrusted shared library. The optional C++ path is not the
default and is not a dependency for CPU-only EveryO. The cross-platform
contract is the pure NumPy implementation. Do not infer that C++ is faster
without benchmarking on your actual data sizes.

## Verification

```bash
python -m pytest -q tests/test_phase3_cpu_engine.py
python -m benchmarks.benchmark_phase3_cpu --size 256 --runs 7
```

Always compare CPU results against an unfused reference before relying on
performance measurements. GPU acceleration remains a separate hardware test.
