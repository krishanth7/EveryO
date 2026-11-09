# Numerical verification (CPU only)

EveryO provides `gradcheck` to compare the gradient from reverse-mode
automatic differentiation with a central finite-difference approximation.
It needs only NumPy and a CPU; it does not require TensorFlow or CUDA.

```python
import everyo as eo

x = eo.tensor([0.2, -0.3, 0.8], dtype="float64")
result = eo.gradcheck(lambda a: eo.sum(a * a), [x])
assert result.passed, result
print(result.max_absolute_error)
```

Use it whenever adding an operator, layer, loss or broadcasting rule. The
function must be deterministic, produce one scalar tensor, and differentiate
with respect to every supplied input. Pass **float64 CPU tensors**. For
vector-valued functions, reduce to a scalar explicitly (for example by
computing a weighted sum) to check a particular vector-Jacobian product.

The verifier constructs fresh tensors and does not change caller data or
gradient buffers. It checks **every** input element, requiring two extra
forward evaluations per element. This makes it suitable for small numerical
tests, not full-sized models. Non-smooth points (ReLU at zero, ties in max)
may fail legitimately because a central difference spans multiple branches.

`GradcheckResult` includes `passed`, `checked_elements`,
`failing_elements`, `max_absolute_error` and `max_relative_error`. A pass is
based on the per-element criterion `abs(analytic - numerical) <= atol +
rtol * abs(numerical)` with defaults `atol=1e-5`, `rtol=1e-3` and
`eps=1e-5`. This result verifies local derivatives for the sampled inputs;
it is not a proof of correctness across all shapes, dtypes or hardware.

Run the regression tests:

```shell
python -m pytest tests/test_verification.py -q
```
