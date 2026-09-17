"""Tests for the automatic differentiation engine.

Analytic gradients are compared against central finite differences, which is
the strongest practical check that a backward pass is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOGradientError
from tests.conftest import numeric_gradient

GRADIENT_RTOL = 1e-4
GRADIENT_ATOL = 1e-6


class TestBasicGradients:
    def test_square_gradient(self):
        """The documented example: d(x^2)/dx = 2x = 4 at x = 2."""
        x = eo.tensor([2.0], requires_grad=True)
        y = x * x
        y.backward()
        np.testing.assert_allclose(x.grad, [4.0])

    def test_addition_gradient(self):
        x = eo.tensor([3.0], requires_grad=True)
        y = eo.tensor([5.0], requires_grad=True)
        (x + y).backward()
        np.testing.assert_allclose(x.grad, [1.0])
        np.testing.assert_allclose(y.grad, [1.0])

    def test_subtraction_gradient(self):
        x = eo.tensor([3.0], requires_grad=True)
        y = eo.tensor([5.0], requires_grad=True)
        (x - y).backward()
        np.testing.assert_allclose(x.grad, [1.0])
        np.testing.assert_allclose(y.grad, [-1.0])

    def test_product_rule(self):
        x = eo.tensor([3.0], requires_grad=True)
        y = eo.tensor([5.0], requires_grad=True)
        (x * y).backward()
        np.testing.assert_allclose(x.grad, [5.0])
        np.testing.assert_allclose(y.grad, [3.0])

    def test_quotient_rule(self):
        x = eo.tensor([6.0], requires_grad=True)
        y = eo.tensor([2.0], requires_grad=True)
        (x / y).backward()
        np.testing.assert_allclose(x.grad, [0.5])
        np.testing.assert_allclose(y.grad, [-1.5])

    def test_chain_rule(self):
        """d/dx exp(3x) = 3 exp(3x)."""
        x = eo.tensor([0.5], requires_grad=True)
        eo.exp(x * 3.0).backward()
        np.testing.assert_allclose(x.grad, [3.0 * np.exp(1.5)], rtol=1e-5)

    def test_gradients_accumulate(self):
        x = eo.tensor([2.0], requires_grad=True)
        (x * 3).backward()
        (x * 5).backward()
        np.testing.assert_allclose(x.grad, [8.0])

    def test_zero_grad_clears_accumulation(self):
        x = eo.tensor([2.0], requires_grad=True)
        (x * 3).backward()
        x.zero_grad()
        assert x.grad is None

    def test_reused_tensor_sums_both_paths(self):
        """y = x + x has dy/dx = 2, exercising a diamond in the graph."""
        x = eo.tensor([4.0], requires_grad=True)
        (x + x).backward()
        np.testing.assert_allclose(x.grad, [2.0])


class TestNumericGradientChecks:
    """Finite-difference checks over the differentiable operation set."""

    @staticmethod
    def _check(build, shape=(3, 4), seed=0):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=shape).astype(np.float64)

        tensor = eo.tensor(values.copy(), requires_grad=True)
        build(tensor).backward()

        expected = numeric_gradient(
            lambda array: float(build(eo.tensor(array.copy())).item()), values.copy()
        )
        np.testing.assert_allclose(tensor.grad, expected, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)

    def test_mean(self):
        self._check(lambda t: eo.mean(t * t))

    def test_sum_over_axis(self):
        self._check(lambda t: eo.sum(eo.sum(t * 2.0, axis=1)))

    def test_relu(self):
        self._check(lambda t: eo.sum(eo.relu(t)))

    def test_sigmoid(self):
        self._check(lambda t: eo.sum(eo.sigmoid(t)))

    def test_tanh(self):
        self._check(lambda t: eo.sum(eo.tanh(t)))

    def test_softmax(self):
        weights = np.random.default_rng(7).normal(size=(3, 4))
        self._check(lambda t: eo.sum(eo.softmax(t, axis=1) * eo.tensor(weights)))

    def test_log_softmax(self):
        weights = np.random.default_rng(8).normal(size=(3, 4))
        self._check(lambda t: eo.sum(eo.log_softmax(t, axis=1) * eo.tensor(weights)))

    def test_exp(self):
        self._check(lambda t: eo.sum(eo.exp(t * 0.5)))

    def test_log(self):
        self._check(lambda t: eo.sum(eo.log(eo.abs(t) + 2.0)))

    def test_sqrt(self):
        self._check(lambda t: eo.sum(eo.sqrt(eo.abs(t) + 2.0)))

    def test_power(self):
        self._check(lambda t: eo.sum(t**3))

    def test_division(self):
        self._check(lambda t: eo.sum(t / (eo.abs(t) + 3.0)))

    def test_matmul(self):
        other = np.random.default_rng(9).normal(size=(4, 2))
        self._check(lambda t: eo.sum(eo.matmul(t, eo.tensor(other))))

    def test_matvec(self):
        other = np.random.default_rng(10).normal(size=(4,))
        self._check(lambda t: eo.sum(eo.matmul(t, eo.tensor(other))))

    def test_transpose(self):
        weights = np.random.default_rng(11).normal(size=(4, 3))
        self._check(lambda t: eo.sum(eo.transpose(t) * eo.tensor(weights)))

    def test_reshape(self):
        weights = np.random.default_rng(12).normal(size=(4, 3))
        self._check(lambda t: eo.sum(eo.reshape(t, (4, 3)) * eo.tensor(weights)))

    def test_broadcast(self):
        weights = np.random.default_rng(13).normal(size=(4,))
        self._check(lambda t: eo.sum(t * eo.tensor(weights)))

    def test_max_reduction(self):
        self._check(lambda t: eo.sum(eo.max(t, axis=1)))

    def test_min_reduction(self):
        self._check(lambda t: eo.sum(eo.min(t, axis=0)))

    def test_slicing(self):
        self._check(lambda t: eo.sum(t[:, 1:3] * 2.0))

    def test_concatenate(self):
        self._check(lambda t: eo.sum(eo.concatenate([t, t * 2.0], axis=0)))

    def test_clip(self):
        self._check(lambda t: eo.sum(eo.clip(t, -0.5, 0.5) * 3.0))

    def test_variance(self):
        self._check(lambda t: eo.variance(t))

    def test_deep_composition(self):
        weights = np.random.default_rng(14).normal(size=(4, 4))
        self._check(lambda t: eo.mean(eo.tanh(eo.matmul(eo.relu(t), eo.tensor(weights))) ** 2))


class TestGradientMode:
    def test_no_grad_disables_tracking(self):
        x = eo.tensor([2.0], requires_grad=True)
        with eo.no_grad():
            y = x * x
        assert not y.requires_grad
        assert y.grad_node is None

    def test_grad_mode_is_restored_after_the_block(self):
        x = eo.tensor([2.0], requires_grad=True)
        with eo.no_grad():
            pass
        assert (x * x).requires_grad

    def test_enable_grad_inside_no_grad(self):
        x = eo.tensor([2.0], requires_grad=True)
        with eo.no_grad(), eo.enable_grad():
            assert (x * x).requires_grad

    def test_no_grad_as_decorator(self):
        @eo.set_grad_enabled(False)
        def compute(value):
            return value * value

        assert not compute(eo.tensor([2.0], requires_grad=True)).requires_grad

    def test_detached_tensors_stop_the_graph(self):
        x = eo.tensor([2.0], requires_grad=True)
        y = (x * 3).detach() * 2
        assert not y.requires_grad


class TestErrors:
    def test_backward_requires_grad(self):
        with pytest.raises(EveryOGradientError, match="does not require gradients"):
            eo.tensor([1.0]).backward()

    def test_non_scalar_backward_needs_a_seed(self):
        x = eo.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(EveryOGradientError, match="explicit gradient"):
            (x * 2).backward()

    def test_non_scalar_backward_with_a_seed_works(self):
        x = eo.tensor([1.0, 2.0], requires_grad=True)
        (x * 2).backward(np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(x.grad, [2.0, 2.0])

    def test_seed_shape_is_validated(self):
        x = eo.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(EveryOGradientError, match="shape"):
            (x * 2).backward(np.ones((3,), dtype=np.float32))


class TestGraphStructure:
    def test_intermediate_tensors_do_not_keep_gradients(self):
        x = eo.tensor([2.0], requires_grad=True)
        intermediate = x * 3
        (intermediate * 2).backward()
        assert intermediate.grad is None
        np.testing.assert_allclose(x.grad, [6.0])

    def test_retain_grad_keeps_the_intermediate_gradient(self):
        x = eo.tensor([2.0], requires_grad=True)
        intermediate = (x * 3).retain_grad()
        (intermediate * 2).backward()
        np.testing.assert_allclose(intermediate.grad, [2.0])

    def test_constant_operands_need_no_gradient(self):
        x = eo.tensor([2.0], requires_grad=True)
        constant = eo.tensor([5.0])
        (x * constant).backward()
        assert constant.grad is None

    def test_deep_graph_does_not_overflow_the_stack(self):
        """The traversal is iterative, so a 2000-deep graph is fine."""
        x = eo.tensor([1.0], requires_grad=True)
        value = x
        for _ in range(2000):
            value = value + 1.0
        value.backward()
        np.testing.assert_allclose(x.grad, [1.0])
