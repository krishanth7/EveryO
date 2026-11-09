"""CPU monitoring and drift test suite."""

import numpy as np
import pytest

from everyo.monitoring import (
    RollingAccuracy,
    build_reference_profile,
    drift_report,
    mean_shift_scores,
    monitoring_summary,
    population_stability_index,
    prediction_entropy,
)


def test_reference_and_stable_population():
    values = np.random.default_rng(1).normal(size=(1000, 3))
    reference = build_reference_profile(values)
    assert reference.count == 1000 and reference.features == 3
    np.testing.assert_allclose(population_stability_index(reference, values), 0, atol=1e-12)
    np.testing.assert_allclose(mean_shift_scores(reference, values), 0, atol=1e-12)
    assert not drift_report(reference, values)["drift_detected"]


def test_shift_alerts_and_summary():
    values = np.random.default_rng(4).normal(size=(300, 2))
    reference = build_reference_profile(values)
    moved = values + np.array([5.0, 0.0])
    report = monitoring_summary(reference, moved, np.full((300, 2), 0.5))
    assert report["drift_detected"] and 0 in report["alert_features"]
    assert np.isclose(report["prediction_entropy"], np.log(2))
    assert report["samples"] == 300


def test_validation_and_rolling_feedback():
    with pytest.raises(ValueError):
        build_reference_profile([[float("nan")]])
    with pytest.raises(ValueError):
        prediction_entropy([[0.2, 0.2]])
    with pytest.raises(ValueError):
        build_reference_profile(np.ones((2, 2)), bins=1)
    rolling = RollingAccuracy(window=3)
    assert rolling.update([1, 0, 1], [1, 1, 1]) == pytest.approx(2 / 3)
    assert rolling.update([0], [0]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        rolling.update([1, 2], [1])


def test_constant_features_and_out_of_range():
    reference = build_reference_profile(np.ones((20, 2)))
    assert np.isfinite(population_stability_index(reference, np.ones((10, 2)))).all()
    with pytest.raises(ValueError):
        drift_report(reference, np.ones((10, 3)))
