"""CPU fault-injection and model-behavior qualification tests."""

import numpy as np
import pytest

from everyo import Linear
from everyo.robustness import (
    batch_consistency_check,
    bounded_prediction,
    feature_dropout,
    gaussian_perturbation,
    repeatability_check,
    robustness_report,
    stress_inputs,
)


def test_seeded_perturbations_and_bounds():
    data = np.arange(12).reshape(6, 2) / 12
    np.testing.assert_array_equal(
        gaussian_perturbation(data, sigma=0.1, seed=2),
        gaussian_perturbation(data, sigma=0.1, seed=2),
    )
    np.testing.assert_array_equal(feature_dropout(data, probability=1), np.zeros_like(data))
    assert set(stress_inputs(data)) == {"unchanged", "zero", "small_noise", "dropout"}
    with pytest.raises(ValueError):
        gaussian_perturbation(data, sigma=-0.1)
    with pytest.raises(ValueError):
        feature_dropout(data, probability=2)


def test_deterministic_model_checks():
    model = Linear(2, 2, seed=3)
    data = np.array([[1.0, 2.0], [4.0, 5.0]])
    assert repeatability_check(model, data)["passed"]
    assert batch_consistency_check(model, data)["passed"]
    assert bounded_prediction(model, data).shape == (2, 2)
    with pytest.raises(ValueError):
        bounded_prediction(lambda x: np.full((len(x), 2), np.nan), data)


def test_model_fails_repeatability_when_stateful():
    rng = np.random.default_rng(5)
    result = repeatability_check(lambda x: rng.normal(size=(len(x), 2)), np.ones((3, 2)))
    assert not result["passed"]


def test_robustness_baseline_and_failure_gates():
    def classifier(x):
        return np.column_stack([1 - x[:, 0], x[:, 0]])

    data = np.array([[0.0, 2.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    report = robustness_report(classifier, data, np.array([0, 1, 0, 1]), sigma=0, dropout=0)
    assert report.passed and report.baseline_accuracy == 1
    failed = robustness_report(
        classifier, data, np.array([0, 1, 0, 1]), sigma=0, dropout=1, max_degradation=0
    )
    assert not failed.passed and failed.relative_degradation > 0
    with pytest.raises(ValueError):
        robustness_report(classifier, data, np.array([0, 1]))
