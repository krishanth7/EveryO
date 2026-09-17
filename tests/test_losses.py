"""Tests for loss functions."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOShapeError
from tests.conftest import numeric_gradient


class TestMSELoss:
    def test_known_value(self):
        loss = eo.MSELoss()(eo.tensor([2.0, 4.0]), eo.tensor([1.0, 1.0]))
        assert loss.item() == pytest.approx((1.0 + 9.0) / 2)

    def test_zero_for_perfect_prediction(self):
        assert eo.MSELoss()(eo.tensor([1.0]), eo.tensor([1.0])).item() == pytest.approx(0.0)

    def test_matches_numpy(self, rng):
        prediction = rng.normal(size=(20, 3))
        target = rng.normal(size=(20, 3))
        expected = float(np.mean((prediction - target) ** 2))
        result = eo.MSELoss()(eo.tensor(prediction), eo.tensor(target)).item()
        assert result == pytest.approx(expected, rel=1e-5)

    def test_reduction_sum(self, rng):
        prediction, target = rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
        expected = float(np.sum((prediction - target) ** 2))
        loss = eo.MSELoss(reduction="sum")(eo.tensor(prediction), eo.tensor(target))
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_reduction_none_keeps_shape(self):
        loss = eo.MSELoss(reduction="none")(eo.zeros(4, 2), eo.ones(4, 2))
        assert loss.shape == (4, 2)

    def test_invalid_reduction(self):
        with pytest.raises(ValueError, match="Unknown reduction"):
            eo.MSELoss(reduction="average")

    def test_shape_mismatch(self):
        with pytest.raises(EveryOShapeError, match="same shape"):
            eo.MSELoss()(eo.zeros(4, 2), eo.zeros(4, 3))

    def test_gradient(self):
        prediction = eo.tensor([3.0, 1.0], requires_grad=True)
        eo.MSELoss()(prediction, eo.tensor([1.0, 1.0])).backward()
        np.testing.assert_allclose(prediction.grad, [2.0, 0.0])


class TestBCELoss:
    def test_known_value(self):
        loss = eo.BCELoss()(eo.tensor([0.5]), eo.tensor([1.0]))
        assert loss.item() == pytest.approx(-np.log(0.5), rel=1e-5)

    def test_matches_reference(self, rng):
        probabilities = rng.uniform(0.05, 0.95, size=(30, 1))
        labels = rng.integers(0, 2, size=(30, 1)).astype(np.float64)
        expected = float(
            -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        )
        loss = eo.BCELoss()(eo.tensor(probabilities), eo.tensor(labels))
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_confident_wrong_prediction_stays_finite(self):
        loss = eo.BCELoss()(eo.tensor([0.0]), eo.tensor([1.0]))
        assert np.isfinite(loss.item())
        assert loss.item() > 10.0

    def test_rejects_values_outside_the_unit_interval(self):
        with pytest.raises(EveryOShapeError, match="probabilities"):
            eo.BCELoss()(eo.tensor([1.5]), eo.tensor([1.0]))

    def test_with_logits_matches_bce_after_sigmoid(self, rng):
        logits = rng.normal(size=(20, 1))
        labels = rng.integers(0, 2, size=(20, 1)).astype(np.float64)
        from_logits = eo.BCEWithLogitsLoss()(eo.tensor(logits), eo.tensor(labels)).item()
        via_sigmoid = eo.BCELoss()(eo.sigmoid(eo.tensor(logits)), eo.tensor(labels)).item()
        assert from_logits == pytest.approx(via_sigmoid, rel=1e-5)

    def test_with_logits_is_stable_for_extreme_values(self):
        loss = eo.BCEWithLogitsLoss()(eo.tensor([-500.0, 500.0]), eo.tensor([1.0, 0.0]))
        assert np.isfinite(loss.item())


class TestBCEWithLogitsGradient:
    """The from-logits form must be exactly right at the awkward point, x = 0."""

    def test_gradient_at_zero_logits(self):
        """d/dx BCEWithLogits = sigmoid(x) - y, which is 0.5 - y at x = 0.

        Composing relu() and abs() gives both a zero subgradient there and
        yields -y instead, which doubles the step for positive labels and
        cancels it for negative ones.
        """
        logits = eo.tensor([0.0, 0.0], requires_grad=True)
        eo.BCEWithLogitsLoss(reduction="sum")(logits, eo.tensor([1.0, 0.0])).backward()
        np.testing.assert_allclose(logits.grad, [-0.5, 0.5], rtol=1e-6)

    def test_gradient_is_sigmoid_minus_target(self, rng):
        values = rng.normal(size=(12, 1))
        targets = rng.integers(0, 2, size=(12, 1)).astype(np.float64)

        logits = eo.tensor(values, requires_grad=True)
        eo.BCEWithLogitsLoss(reduction="sum")(logits, eo.tensor(targets)).backward()

        expected = 1.0 / (1.0 + np.exp(-values)) - targets
        np.testing.assert_allclose(logits.grad, expected, rtol=1e-5, atol=1e-7)

    def test_gradient_matches_finite_differences(self, rng):
        values = rng.normal(size=(8, 1))
        targets = rng.integers(0, 2, size=(8, 1)).astype(np.float64)

        logits = eo.tensor(values.copy(), requires_grad=True)
        eo.BCEWithLogitsLoss(reduction="sum")(logits, eo.tensor(targets)).backward()

        expected = numeric_gradient(
            lambda array: eo.BCEWithLogitsLoss(reduction="sum")(
                eo.tensor(array.copy()), eo.tensor(targets)
            ).item(),
            values.copy(),
        )
        np.testing.assert_allclose(logits.grad, expected, rtol=1e-4, atol=1e-6)

    def test_zero_logits_train_away_from_the_starting_point(self):
        """A model whose logits start at zero must still learn both classes."""
        model = eo.Sequential(eo.Linear(2, 1, bias=True, initializer="zeros", seed=0))
        optimizer = eo.SGD(model.parameters(), lr=0.5)
        loss_fn = eo.BCEWithLogitsLoss()

        features = eo.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = eo.tensor([[1.0], [0.0]])
        for _ in range(50):
            optimizer.zero_grad()
            loss_fn(model(features), targets).backward()
            optimizer.step()

        predictions = eo.sigmoid(model(features)).numpy().reshape(-1)
        assert predictions[0] > 0.7 and predictions[1] < 0.3


class TestCrossEntropyLoss:
    def test_confident_correct_prediction_is_near_zero(self):
        logits = eo.tensor([[20.0, 0.0, 0.0]])
        assert eo.CrossEntropyLoss()(logits, [0]).item() == pytest.approx(0.0, abs=1e-6)

    def test_uniform_logits_give_log_num_classes(self):
        logits = eo.tensor([[0.0, 0.0, 0.0, 0.0]])
        assert eo.CrossEntropyLoss()(logits, [2]).item() == pytest.approx(np.log(4), rel=1e-5)

    def test_matches_manual_computation(self, rng):
        logits = rng.normal(size=(8, 5))
        labels = rng.integers(0, 5, size=8)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        expected = float(-np.mean(log_probs[np.arange(8), labels]))
        loss = eo.CrossEntropyLoss()(eo.tensor(logits), labels)
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_accepts_one_hot_targets(self, rng):
        logits = rng.normal(size=(6, 3))
        labels = rng.integers(0, 3, size=6)
        one_hot = eo.one_hot_encode(labels, 3)
        by_index = eo.CrossEntropyLoss()(eo.tensor(logits), labels).item()
        by_one_hot = eo.CrossEntropyLoss()(eo.tensor(logits), eo.tensor(one_hot)).item()
        assert by_index == pytest.approx(by_one_hot, rel=1e-5)

    def test_requires_2d_logits(self):
        with pytest.raises(EveryOShapeError, match=r"\(batch, classes\)"):
            eo.CrossEntropyLoss()(eo.tensor([1.0, 2.0]), [0])

    def test_target_count_must_match(self):
        with pytest.raises(EveryOShapeError, match="targets"):
            eo.CrossEntropyLoss()(eo.zeros(4, 3), [0, 1])

    def test_gradient_is_softmax_minus_one_hot(self, rng):
        logits_values = rng.normal(size=(4, 3))
        labels = np.array([0, 1, 2, 1])
        logits = eo.tensor(logits_values, requires_grad=True)
        eo.CrossEntropyLoss()(logits, labels).backward()

        probabilities = eo.softmax(eo.tensor(logits_values), axis=1).numpy()
        expected = probabilities.copy()
        expected[np.arange(4), labels] -= 1.0
        expected /= 4.0
        np.testing.assert_allclose(logits.grad, expected, rtol=1e-5, atol=1e-7)

    def test_gradient_matches_finite_differences(self, rng):
        labels = np.array([0, 2, 1])
        values = rng.normal(size=(3, 4))
        logits = eo.tensor(values.copy(), requires_grad=True)
        eo.CrossEntropyLoss()(logits, labels).backward()
        expected = numeric_gradient(
            lambda array: eo.CrossEntropyLoss()(eo.tensor(array.copy()), labels).item(),
            values.copy(),
        )
        np.testing.assert_allclose(logits.grad, expected, rtol=1e-4, atol=1e-6)


class TestFunctionalForms:
    def test_functional_matches_class(self, rng):
        prediction, target = rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
        assert eo.nn.mse_loss(eo.tensor(prediction), eo.tensor(target)).item() == pytest.approx(
            eo.MSELoss()(eo.tensor(prediction), eo.tensor(target)).item()
        )


class TestMAELoss:
    def test_known_value(self):
        assert eo.MAELoss()(eo.tensor([3.0, -1.0]), eo.tensor([1.0, 1.0])).item() == pytest.approx(
            2.0
        )
