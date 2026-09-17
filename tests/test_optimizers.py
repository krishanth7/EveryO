"""Tests for optimizers: update rules, hyper-parameters and convergence."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOError


def quadratic_problem(seed: int = 0):
    """A convex least-squares problem with a known optimum."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(200, 4)).astype(np.float32)
    weights = rng.normal(size=(4, 1)).astype(np.float32)
    targets = features @ weights + 0.5
    return features, targets


class TestOptimizerBase:
    def test_empty_parameter_list_is_rejected(self):
        with pytest.raises(EveryOError, match="no parameters"):
            eo.SGD([], lr=0.1)

    def test_non_tensor_parameter_is_rejected(self):
        with pytest.raises(EveryOError, match="EveryO tensor"):
            eo.SGD([np.zeros(3)], lr=0.1)

    def test_non_positive_learning_rate_is_rejected(self):
        model = eo.Sequential(eo.Linear(2, 1, seed=0))
        with pytest.raises(ValueError, match="positive"):
            eo.SGD(model.parameters(), lr=0.0)

    def test_zero_grad_clears_gradients(self):
        model = eo.Sequential(eo.Linear(2, 1, seed=0))
        optimizer = eo.SGD(model.parameters(), lr=0.1)
        eo.sum(model(eo.ones(3, 2))).backward()
        optimizer.zero_grad()
        assert all(p.grad is None for p in model.parameters())

    def test_step_without_gradients_is_a_no_op(self):
        model = eo.Sequential(eo.Linear(2, 1, seed=0))
        before = model[0].weight.numpy()
        eo.SGD(model.parameters(), lr=0.1).step()
        np.testing.assert_allclose(model[0].weight.numpy(), before)


class TestSGD:
    def test_plain_update_rule(self):
        parameter = eo.Parameter(np.array([1.0, 2.0], dtype=np.float32))
        parameter.grad = np.array([0.5, 1.0], dtype=np.float32)
        eo.SGD([parameter], lr=0.1).step()
        np.testing.assert_allclose(parameter.numpy(), [0.95, 1.9], rtol=1e-6)

    def test_momentum_accumulates(self):
        parameter = eo.Parameter(np.array([1.0], dtype=np.float32))
        optimizer = eo.SGD([parameter], lr=0.1, momentum=0.9)

        parameter.grad = np.array([1.0], dtype=np.float32)
        optimizer.step()
        np.testing.assert_allclose(parameter.numpy(), [0.9], rtol=1e-6)

        # velocity = 0.9 * 1.0 + 1.0 = 1.9  =>  0.9 - 0.19 = 0.71
        parameter.grad = np.array([1.0], dtype=np.float32)
        optimizer.step()
        np.testing.assert_allclose(parameter.numpy(), [0.71], rtol=1e-5)

    def test_weight_decay_shrinks_parameters(self):
        parameter = eo.Parameter(np.array([2.0], dtype=np.float32))
        parameter.grad = np.array([0.0], dtype=np.float32)
        eo.SGD([parameter], lr=0.1, weight_decay=0.5).step()
        np.testing.assert_allclose(parameter.numpy(), [1.9], rtol=1e-6)

    def test_invalid_momentum(self):
        parameter = eo.Parameter(np.zeros(1))
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            eo.SGD([parameter], lr=0.1, momentum=1.0)

    def test_nesterov_requires_momentum(self):
        parameter = eo.Parameter(np.zeros(1))
        with pytest.raises(ValueError, match="Nesterov"):
            eo.SGD([parameter], lr=0.1, nesterov=True)

    def test_converges_on_a_convex_problem(self):
        features, targets = quadratic_problem()
        model = eo.Sequential(eo.Linear(4, 1, seed=0))
        optimizer = eo.SGD(model.parameters(), lr=0.05, momentum=0.9)
        loss_fn = eo.MSELoss()

        first = None
        for _ in range(150):
            optimizer.zero_grad()
            loss = loss_fn(model(eo.tensor(features)), eo.tensor(targets))
            loss.backward()
            optimizer.step()
            first = first if first is not None else loss.item()

        assert loss.item() < first * 0.01
        assert loss.item() < 1e-3


class TestAdam:
    def test_first_step_is_approximately_the_learning_rate(self):
        """With bias correction the first Adam step has magnitude ~lr."""
        parameter = eo.Parameter(np.array([1.0], dtype=np.float32))
        parameter.grad = np.array([3.0], dtype=np.float32)
        eo.Adam([parameter], lr=0.1).step()
        assert parameter.item() == pytest.approx(0.9, abs=1e-4)

    def test_step_size_is_scale_invariant(self):
        """Adam normalises by the gradient magnitude, unlike SGD."""
        small = eo.Parameter(np.array([1.0], dtype=np.float32))
        large = eo.Parameter(np.array([1.0], dtype=np.float32))
        small.grad = np.array([1e-4], dtype=np.float32)
        large.grad = np.array([1e4], dtype=np.float32)
        eo.Adam([small], lr=0.1).step()
        eo.Adam([large], lr=0.1).step()
        assert small.item() == pytest.approx(large.item(), abs=1e-4)

    def test_invalid_betas(self):
        parameter = eo.Parameter(np.zeros(1))
        with pytest.raises(ValueError, match="beta1"):
            eo.Adam([parameter], lr=0.1, betas=(1.0, 0.999))

    def test_invalid_eps(self):
        parameter = eo.Parameter(np.zeros(1))
        with pytest.raises(ValueError, match="eps"):
            eo.Adam([parameter], lr=0.1, eps=0.0)

    def test_amsgrad_runs(self):
        features, targets = quadratic_problem()
        model = eo.Sequential(eo.Linear(4, 1, seed=0))
        optimizer = eo.Adam(model.parameters(), lr=0.05, amsgrad=True)
        loss_fn = eo.MSELoss()
        for _ in range(50):
            optimizer.zero_grad()
            loss = loss_fn(model(eo.tensor(features)), eo.tensor(targets))
            loss.backward()
            optimizer.step()
        assert loss.item() < 1.0

    def test_converges_on_a_convex_problem(self):
        features, targets = quadratic_problem()
        model = eo.Sequential(eo.Linear(4, 1, seed=0))
        optimizer = eo.Adam(model.parameters(), lr=0.05)
        loss_fn = eo.MSELoss()

        first = None
        for _ in range(200):
            optimizer.zero_grad()
            loss = loss_fn(model(eo.tensor(features)), eo.tensor(targets))
            loss.backward()
            optimizer.step()
            first = first if first is not None else loss.item()

        assert loss.item() < first * 0.01
        assert loss.item() < 1e-3

    def test_recovers_known_linear_weights(self):
        """After training, the layer should approximate the generating weights."""
        rng = np.random.default_rng(3)
        true_weights = np.array([[1.5], [-2.0], [0.5]], dtype=np.float32)
        features = rng.normal(size=(400, 3)).astype(np.float32)
        targets = features @ true_weights

        model = eo.Sequential(eo.Linear(3, 1, bias=False, seed=0))
        optimizer = eo.Adam(model.parameters(), lr=0.05)
        loss_fn = eo.MSELoss()
        for _ in range(500):
            optimizer.zero_grad()
            loss_fn(model(eo.tensor(features)), eo.tensor(targets)).backward()
            optimizer.step()

        np.testing.assert_allclose(model[0].weight.numpy(), true_weights, atol=1e-2)


class TestOptimizerReprAndConfig:
    def test_config_contains_hyperparameters(self):
        model = eo.Sequential(eo.Linear(2, 1, seed=0))
        config = eo.Adam(model.parameters(), lr=0.01).get_config()
        assert config["lr"] == 0.01
        assert config["betas"] == (0.9, 0.999)

    def test_repr(self):
        model = eo.Sequential(eo.Linear(2, 1, seed=0))
        assert "SGD" in repr(eo.SGD(model.parameters(), lr=0.1))
