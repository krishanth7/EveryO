"""Phase 2 CPU autograd observability and numerical derivative qualification."""

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOGradientError


def test_graph_report_and_intermediate_cleanup():
    x = eo.tensor([1.0, 2.0], requires_grad=True)
    with eo.mutation_guard():
        y = x * x
        y.retain_grad()
        output = eo.sum(y)
    before = eo.graph_report(output)
    assert before.operation_nodes == 2
    assert before.edges == 3
    assert before.saved_input_bytes > 0
    output.backward()
    assert eo.graph_report(output).retained_gradient_bytes > 0
    assert eo.clear_intermediate_grads(output) == 1
    assert y.grad is None
    np.testing.assert_allclose(x.grad, [2, 4])
    assert eo.clear_intermediate_grads(output) == 0


@pytest.mark.parametrize("seed", range(20))
def test_jacobian_jvp_against_closed_form(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(2, 3))
    direction = rng.normal(size=(2, 3))
    x = eo.tensor(values, dtype="float64", requires_grad=True)
    result = eo.jacobian(lambda item: item * item, x)
    assert result.shape == (2, 3, 2, 3)
    expected = np.diag((2 * values).ravel()).reshape(2, 3, 2, 3)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        eo.jvp(lambda item: item * item, x, direction), 2 * values * direction, atol=1e-8
    )
    assert x.grad is None
    np.testing.assert_array_equal(x.data, values)


@pytest.mark.parametrize("seed", range(20))
def test_hessian_hvp_against_closed_form(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(2, 3))
    direction = rng.normal(size=(2, 3))
    x = eo.tensor(values, dtype="float64", requires_grad=True)

    def scalar(item):
        return eo.sum(item * item * item)

    expected = np.diag((6 * values).ravel())
    np.testing.assert_allclose(eo.hessian(scalar, x).reshape(6, 6), expected, atol=1e-7)
    np.testing.assert_allclose(eo.hvp(scalar, x, direction), 6 * values * direction, atol=1e-7)
    assert x.grad is None


def test_numerical_derivative_input_validation():
    x = eo.tensor([1.0], dtype="float64")
    with pytest.raises(ValueError, match="positive"):
        eo.jacobian(lambda a: a * a, x, eps=0)
    with pytest.raises(ValueError, match="direction"):
        eo.jvp(lambda a: a * a, x, [1, 2])
    with pytest.raises(ValueError, match="scalar"):
        eo.hessian(lambda a: a * a, eo.tensor([1.0, 2.0], dtype="float64"))
    with pytest.raises(TypeError, match="float64"):
        eo.jacobian(lambda a: a, eo.tensor([1.0], dtype="float32"))
    with pytest.raises(ValueError, match="non-finite"):
        eo.jacobian(lambda a: a, eo.tensor([np.nan], dtype="float64"))


def test_anomaly_detection_rejects_nonfinite_seed_and_gradients():
    x = eo.tensor([2.0], requires_grad=True)
    y = x * x
    with eo.detect_anomaly(), pytest.raises(EveryOGradientError, match="seed"):
        y.backward(np.array([np.nan]))
    assert x.grad is None
    with eo.detect_anomaly(), eo.detect_anomaly():
        y.backward()
    np.testing.assert_allclose(x.grad, [4.0])


def test_anomaly_mode_is_optional():
    x = eo.tensor([1.0], requires_grad=True)
    y = x * eo.tensor([np.nan])
    with eo.detect_anomaly(), pytest.raises(EveryOGradientError, match="Non-finite gradient"):
        y.backward()
    assert x.grad is None
