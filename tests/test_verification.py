"""Public, hardware-independent gradient-checking contract."""

import numpy as np
import pytest

import everyo as eo


def test_gradcheck_broadcast_and_preserves_inputs():
    x = eo.tensor([[0.2, -0.5], [1.2, 0.3]], dtype="float64", requires_grad=True)
    y = eo.tensor([0.7, -0.4], dtype="float64", requires_grad=True)
    x.grad = np.ones_like(x.data)
    original = x.numpy()
    result = eo.gradcheck(lambda a, b: eo.sum((a * b) ** 2), [x, y])
    assert result.passed
    assert result.checked_elements == 6
    assert result.failing_elements == 0
    np.testing.assert_array_equal(x.data, original)
    np.testing.assert_array_equal(x.grad, np.ones_like(x.data))
    assert y.grad is None


def test_gradcheck_reports_wrong_analytic_derivative():
    def broken(a):
        return eo.sum(a * a.detach())

    result = eo.gradcheck(broken, [eo.tensor([1.0, 2.0], dtype="float64")])
    assert not result.passed
    assert result.failing_elements == 2


@pytest.mark.parametrize("eps,atol,rtol", [(0, 0, 0), (1e-5, -1, 0), (1e-5, 0, np.nan)])
def test_gradcheck_rejects_invalid_tolerances(eps, atol, rtol):
    with pytest.raises(ValueError):
        eo.gradcheck(
            lambda a: eo.sum(a * a),
            [eo.tensor([1.0], dtype="float64")],
            eps=eps,
            atol=atol,
            rtol=rtol,
        )


def test_gradcheck_rejects_nonscalar_and_unused_input():
    x = eo.tensor([1.0, 2.0], dtype="float64")
    with pytest.raises(ValueError, match="scalar"):
        eo.gradcheck(lambda a: a * a, [x])
    with pytest.raises(ValueError, match="gradient"):
        eo.gradcheck(lambda a, b: eo.sum(a * a), [x, x])


def test_gradcheck_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="non-finite"):
        eo.gradcheck(lambda a: eo.sum(a), [eo.tensor([np.inf], dtype="float64")])
