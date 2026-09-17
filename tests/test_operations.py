"""Tests for the numerical behaviour of tensor operations.

Results are checked against NumPy, which is the reference implementation for
the CPU backend.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOShapeError

RTOL = 1e-5
ATOL = 1e-7


@pytest.fixture
def pair(rng):
    a = rng.normal(size=(4, 3)).astype(np.float64)
    b = rng.normal(size=(4, 3)).astype(np.float64)
    return a, b


class TestArithmetic:
    def test_add_matches_numpy(self, pair):
        a, b = pair
        np.testing.assert_allclose(
            eo.add(eo.tensor(a), eo.tensor(b)).numpy(), a + b, rtol=RTOL, atol=ATOL
        )

    def test_subtract_matches_numpy(self, pair):
        a, b = pair
        np.testing.assert_allclose((eo.tensor(a) - eo.tensor(b)).numpy(), a - b, rtol=RTOL)

    def test_multiply_matches_numpy(self, pair):
        a, b = pair
        np.testing.assert_allclose((eo.tensor(a) * eo.tensor(b)).numpy(), a * b, rtol=RTOL)

    def test_divide_matches_numpy(self, pair):
        a, b = pair
        b = b + 2.0  # keep the denominator away from zero
        np.testing.assert_allclose((eo.tensor(a) / eo.tensor(b)).numpy(), a / b, rtol=RTOL)

    def test_scalar_operands(self):
        tensor = eo.tensor([1.0, 2.0])
        np.testing.assert_allclose((tensor + 3).numpy(), [4.0, 5.0])
        np.testing.assert_allclose((3 - tensor).numpy(), [2.0, 1.0])
        np.testing.assert_allclose((2 * tensor).numpy(), [2.0, 4.0])
        np.testing.assert_allclose((tensor**2).numpy(), [1.0, 4.0])
        np.testing.assert_allclose((-tensor).numpy(), [-1.0, -2.0])

    def test_elementwise_maths(self, pair):
        a, _ = pair
        np.testing.assert_allclose(eo.exp(eo.tensor(a)).numpy(), np.exp(a), rtol=RTOL)
        np.testing.assert_allclose(eo.abs(eo.tensor(a)).numpy(), np.abs(a), rtol=RTOL)
        positive = np.abs(a) + 1.0
        np.testing.assert_allclose(eo.log(eo.tensor(positive)).numpy(), np.log(positive), rtol=RTOL)
        np.testing.assert_allclose(
            eo.sqrt(eo.tensor(positive)).numpy(), np.sqrt(positive), rtol=RTOL
        )

    def test_clip(self):
        np.testing.assert_allclose(
            eo.clip(eo.tensor([-2.0, 0.5, 3.0]), 0.0, 1.0).numpy(), [0.0, 0.5, 1.0]
        )

    def test_clip_rejects_inverted_range(self):
        with pytest.raises(EveryOShapeError, match="low <= high"):
            eo.clip(eo.tensor([1.0]), 1.0, 0.0)


class TestBroadcasting:
    def test_row_vector_broadcast(self, rng):
        a = rng.normal(size=(4, 3))
        b = rng.normal(size=(3,))
        np.testing.assert_allclose((eo.tensor(a) + eo.tensor(b)).numpy(), a + b, rtol=RTOL)

    def test_column_vector_broadcast(self, rng):
        a = rng.normal(size=(4, 3))
        b = rng.normal(size=(4, 1))
        np.testing.assert_allclose((eo.tensor(a) * eo.tensor(b)).numpy(), a * b, rtol=RTOL)

    def test_incompatible_shapes_raise(self):
        with pytest.raises(EveryOShapeError, match="broadcast"):
            eo.tensor(np.zeros((4, 3))) + eo.tensor(np.zeros((5, 2)))

    def test_error_message_names_both_shapes(self):
        with pytest.raises(EveryOShapeError) as info:
            eo.tensor(np.zeros((4, 3))) + eo.tensor(np.zeros((5, 2)))
        assert "(4, 3)" in str(info.value)
        assert "(5, 2)" in str(info.value)


class TestLinearAlgebra:
    def test_matmul_matches_numpy(self, rng):
        a = rng.normal(size=(5, 4))
        b = rng.normal(size=(4, 3))
        np.testing.assert_allclose(eo.matmul(eo.tensor(a), eo.tensor(b)).numpy(), a @ b, rtol=RTOL)

    def test_matmul_operator(self, rng):
        a, b = rng.normal(size=(2, 3)), rng.normal(size=(3, 2))
        np.testing.assert_allclose((eo.tensor(a) @ eo.tensor(b)).numpy(), a @ b, rtol=RTOL)

    def test_matrix_vector(self, rng):
        a, b = rng.normal(size=(3, 4)), rng.normal(size=(4,))
        np.testing.assert_allclose(eo.matmul(eo.tensor(a), eo.tensor(b)).numpy(), a @ b, rtol=RTOL)

    def test_dot_of_vectors(self, rng):
        a, b = rng.normal(size=(6,)), rng.normal(size=(6,))
        assert eo.dot(eo.tensor(a), eo.tensor(b)).item() == pytest.approx(float(a @ b), rel=1e-5)

    def test_mismatched_inner_dimensions(self):
        with pytest.raises(EveryOShapeError) as info:
            eo.matmul(eo.zeros(32, 64), eo.zeros(128, 10))
        message = str(info.value)
        assert "(32, 64)" in message and "(128, 10)" in message
        assert "inner dimensions" in message

    def test_scalar_matmul_is_rejected(self):
        with pytest.raises(EveryOShapeError, match="at least one dimension"):
            eo.matmul(eo.tensor(2.0), eo.tensor(3.0))


class TestReductions:
    def test_sum_all(self, pair):
        a, _ = pair
        assert eo.sum(eo.tensor(a)).item() == pytest.approx(float(a.sum()))

    def test_sum_axis(self, pair):
        a, _ = pair
        np.testing.assert_allclose(eo.sum(eo.tensor(a), axis=0).numpy(), a.sum(axis=0), rtol=RTOL)

    def test_sum_keepdims(self, pair):
        a, _ = pair
        assert eo.sum(eo.tensor(a), axis=1, keepdims=True).shape == (4, 1)

    def test_mean(self, pair):
        a, _ = pair
        np.testing.assert_allclose(eo.mean(eo.tensor(a), axis=1).numpy(), a.mean(axis=1), rtol=RTOL)

    def test_max_and_min(self, pair):
        a, _ = pair
        np.testing.assert_allclose(eo.max(eo.tensor(a), axis=0).numpy(), a.max(axis=0), rtol=RTOL)
        np.testing.assert_allclose(eo.min(eo.tensor(a)).numpy(), a.min(), rtol=RTOL)

    def test_variance(self, pair):
        a, _ = pair
        assert eo.variance(eo.tensor(a)).item() == pytest.approx(float(a.var()), rel=1e-5)

    def test_argmax(self):
        assert eo.tensor([[1.0, 7.0], [9.0, 2.0]]).argmax(axis=1).tolist() == [1, 0]

    def test_invalid_axis(self):
        with pytest.raises(EveryOShapeError, match="out of range"):
            eo.sum(eo.zeros(2, 3), axis=5)


class TestShapeManipulation:
    def test_reshape(self, pair):
        a, _ = pair
        assert eo.tensor(a).reshape(3, 4).shape == (3, 4)
        assert eo.tensor(a).reshape((2, 6)).shape == (2, 6)

    def test_reshape_with_wrong_size(self):
        with pytest.raises(EveryOShapeError, match="Cannot reshape"):
            eo.zeros(4, 3).reshape(5, 5)

    def test_transpose(self, pair):
        a, _ = pair
        np.testing.assert_allclose(eo.transpose(eo.tensor(a)).numpy(), a.T)

    def test_transpose_with_axes(self, rng):
        a = rng.normal(size=(2, 3, 4))
        np.testing.assert_allclose(
            eo.tensor(a).transpose(1, 0, 2).numpy(), np.transpose(a, (1, 0, 2))
        )

    def test_transpose_rejects_bad_permutation(self):
        with pytest.raises(EveryOShapeError, match="permutation"):
            eo.zeros(2, 3).transpose(0, 0)

    def test_flatten(self, rng):
        a = rng.normal(size=(5, 2, 3))
        assert eo.flatten(eo.tensor(a), start_dim=1).shape == (5, 6)

    def test_concatenate(self, rng):
        a, b = rng.normal(size=(2, 3)), rng.normal(size=(4, 3))
        assert eo.concatenate([eo.tensor(a), eo.tensor(b)], axis=0).shape == (6, 3)

    def test_concatenate_mismatch(self):
        with pytest.raises(EveryOShapeError, match="Cannot concatenate"):
            eo.concatenate([eo.zeros(2, 3), eo.zeros(2, 4)], axis=0)

    def test_stack(self):
        assert eo.stack([eo.zeros(2, 3), eo.zeros(2, 3)]).shape == (2, 2, 3)


class TestActivationNumerics:
    def test_relu(self):
        np.testing.assert_allclose(eo.relu(eo.tensor([-1.0, 0.0, 2.0])).numpy(), [0.0, 0.0, 2.0])

    def test_sigmoid_matches_reference(self, pair):
        a, _ = pair
        np.testing.assert_allclose(
            eo.sigmoid(eo.tensor(a)).numpy(), 1.0 / (1.0 + np.exp(-a)), rtol=1e-6
        )

    def test_sigmoid_is_stable_for_large_inputs(self):
        values = eo.sigmoid(eo.tensor([-800.0, 800.0])).numpy()
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, [0.0, 1.0], atol=1e-6)

    def test_tanh(self, pair):
        a, _ = pair
        np.testing.assert_allclose(eo.tanh(eo.tensor(a)).numpy(), np.tanh(a), rtol=RTOL)

    def test_softmax_rows_sum_to_one(self, pair):
        a, _ = pair
        probabilities = eo.softmax(eo.tensor(a), axis=1).numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4), rtol=1e-6)

    def test_softmax_is_stable_for_large_inputs(self):
        probabilities = eo.softmax(eo.tensor([[1000.0, 1000.0]])).numpy()
        np.testing.assert_allclose(probabilities, [[0.5, 0.5]], rtol=1e-6)

    def test_log_softmax_matches_log_of_softmax(self, pair):
        a, _ = pair
        np.testing.assert_allclose(
            eo.log_softmax(eo.tensor(a), axis=1).numpy(),
            np.log(eo.softmax(eo.tensor(a), axis=1).numpy()),
            rtol=1e-5,
            atol=1e-6,
        )


class TestCreation:
    def test_zeros_and_ones(self):
        assert eo.zeros(2, 3).numpy().sum() == 0.0
        assert eo.ones((2, 3)).numpy().sum() == pytest.approx(6.0)

    def test_full_and_like(self):
        assert eo.full((2, 2), 7.0).numpy().mean() == pytest.approx(7.0)
        assert eo.zeros_like(eo.ones(3, 3)).shape == (3, 3)

    def test_arange_and_linspace(self):
        np.testing.assert_allclose(eo.arange(0, 5, 1).numpy(), [0, 1, 2, 3, 4])
        assert eo.linspace(0.0, 1.0, 11).shape == (11,)

    def test_eye(self):
        np.testing.assert_allclose(eo.eye(3).numpy(), np.eye(3))

    def test_random_is_reproducible(self):
        first = eo.normal((3, 3), seed=42).numpy()
        second = eo.normal((3, 3), seed=42).numpy()
        np.testing.assert_allclose(first, second)

    def test_uniform_respects_bounds(self):
        values = eo.uniform((500,), low=-1.0, high=1.0, seed=0).numpy()
        assert values.min() >= -1.0 and values.max() < 1.0

    def test_one_hot(self):
        encoded = eo.one_hot([0, 2], 3).numpy()
        np.testing.assert_allclose(encoded, [[1, 0, 0], [0, 0, 1]])

    def test_one_hot_rejects_out_of_range(self):
        with pytest.raises(EveryOShapeError):
            eo.one_hot([5], 3)
