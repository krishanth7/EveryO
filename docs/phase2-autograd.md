# Phase 2 — CPU derivative and autograd qualification (2.1–2.7)

EveryO's first-order reverse-mode engine produces NumPy gradients. This phase
does **not** add symbolic, graph-tracked, exact second-order autodiff. Higher
derivatives are CPU-only numerical approximations using independent probes.

| Subphase | API or deliverable | Boundaries |
| --- | --- | --- |
| 2.1 Graph integrity | `graph_report(output)` traverses the reachable graph iteratively | Read-only counters, not a graph serializer |
| 2.2 Higher-order derivative pathway | `hvp` computes a finite-difference of first-order autodiff gradients | Approximate, two backward passes |
| 2.3 Jacobian/Hessian | `jacobian`, `hessian`, `jvp`, `hvp` | float64 CPU, finite deterministic functions |
| 2.4 Numerical correctness | Seeded closed-form comparisons, validation of eps/input/output | Finite differences have truncation and roundoff error |
| 2.5 Memory inspection | Graph snapshots and retained-grad byte counters; `clear_intermediate_grads` | Leaves graph edges and leaf gradients intact |
| 2.6 Debugging | `detect_anomaly` rejects non-finite seeds and returned gradients | Thread-local opt-in; not a complete anomaly trace |
| 2.7 Qualification | 40 generated derivative cases plus error/graph tests | No GPU tests or production certification |

## Example

```python
import numpy as np
import everyo as eo

x = eo.tensor([1.0, 2.0], dtype="float64")
fn = lambda t: eo.sum(t * t * t)
H = eo.hessian(fn, x)  # approximately diag([6, 12])
Hv = eo.hvp(fn, x, np.ones(2))
J = eo.jacobian(lambda t: t * t, x)
Jv = eo.jvp(lambda t: t * t, x, np.ones(2))

tracked = eo.tensor([2.0], requires_grad=True)
with eo.detect_anomaly():
    loss = tracked * tracked
    loss.backward()
print(eo.graph_report(loss))
```

## Precision and resource use

- `jacobian`: two function evaluations per input element; result memory is
  output element count × input element count × 8 bytes.
- `hessian`: two first-order backward passes per input element; dense output
  memory grows quadratically with input dimension.
- `jvp`: two forward evaluations; `hvp`: two first-order backward evaluations.
- Default `eps` is `1e-5` for forward differences and `1e-4` for gradients.
  Tune for your problem scale. Nonsmooth functions may not have a Hessian.
- The APIs copy input values, do not mutate the caller, and reject non-finite
  inputs or function outputs. They require deterministic functions for
  meaningful results.
- The anomaly context guards backward only while the context is active. It
  adds validation overhead and cannot identify every source of an earlier NaN.

## Acceptance

```sh
python -m pytest tests/test_phase2_derivatives.py -q
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```
