# CPU foundation hardening — Phase 1 (1.1–1.7)

This is a CPU-only API addition. It does not claim that the framework is
production-certified or that its CUDA path has been validated on a GPU.

| Subphase | Implemented behavior | Verification |
| --- | --- | --- |
| 1.1 Tensor invariants | `inspect_tensor`, `assert_tensor_valid` for gradient shape and finite-value checks | Shape, dtype and malformed-gradient tests |
| 1.2 Indexing/views | `index_select` supports repeated indices; `Tensor.view` enforces zero-copy reshape | Randomized gradients and storage-sharing tests |
| 1.3 Mutation safety | `safe_assign_` validates replacements; opt-in `mutation_guard` snapshots graph inputs and detects NumPy writes before backward | Mutation, alias and error-path tests |
| 1.4 Numerical stability | Stable differentiable `logsumexp` using shifted exponentials | 12 randomized large-magnitude forward/backward tests |
| 1.5 Diagnostics | Structured, JSON-compatible tensor health report | Non-finite and invalid-gradient tests |
| 1.6 Deterministic tests | Generated shapes, values and broadcasting gradients using 100 fixed seeds | `tests/test_quality.py` |
| 1.7 Benchmarks | `benchmark_cpu` with warmups, seed, timings, output checksum, environment; `compare_benchmarks` | Determinism and regression guard tests |

## API examples

```python
import everyo as eo

x = eo.tensor([[1000.0, 1001.0]], dtype="float64", requires_grad=True)
y = eo.logsumexp(x, axis=1)
y.backward()
print(y.numpy(), x.grad)

print(eo.inspect_tensor(x).to_dict())
eo.assert_tensor_valid(x, finite=True)

selected = eo.index_select(x, axis=1, indices=[1, 1, 0])

with eo.mutation_guard():
    loss = eo.sum(x.view(2) * x.view(2))
loss.backward()  # rejects mutations made after the guarded forward pass
```

### Safe replacement of untracked tensor data

```python
import numpy as np

x = eo.tensor([1.0, 2.0], dtype="float64")
eo.safe_assign_(x, np.array([3.0, 4.0], dtype="float64"))
```

`safe_assign_` is a guarded replacement interface. Direct mutation of
`tensor.data` or a NumPy view can bypass it **unless** `mutation_guard()`
is enabled during forward graph construction. The guard retains full snapshots
and checks them before backward writes gradients. This catches value changes
including writes through aliases but increases memory use; it is intended for
diagnostics and qualification, not fast production training.

### Benchmarking

```shell
python -m benchmarks.benchmark_cpu_quality --size 128 --runs 10 --warmups 3
python -m benchmarks.benchmark_cpu_quality --size 128 --runs 10 --output cpu.json
```

The resulting JSON records Python, NumPy and platform identification, a fixed
RNG seed and output checksum. Compare medians only on controlled, comparable
hardware and repeat runs to account for CPU scheduling variance. A slowdown
threshold is a heuristic, not a proof of a real performance regression.

## Limitations

- `logsumexp` explicitly supports finite floating-point data. It does not
  implement infinities as special limiting cases.
- `index_select` uses the existing differentiable indexing engine and copies
  integer indices to int64; it is not a zero-copy advanced-index operation.
- Arbitrary NumPy-level in-place writes remain possible when `mutation_guard`
  is not enabled. A write followed by restoration of identical values cannot
  be distinguished from no write; that restoration does not change gradients.
- Generated property tests cover selected operations, not every existing
  operator, shape, dtype, or malformed input.

## Acceptance commands

```shell
python -m pytest tests/test_quality.py -q
python -m pytest -q
python -m ruff check everyo tests benchmarks/benchmark_cpu_quality.py
python -m ruff format --check everyo tests benchmarks/benchmark_cpu_quality.py
```
