"""Deterministic Phase 1 CPU correctness and diagnostics acceptance tests."""

import json
from dataclasses import replace

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOGradientError, EveryOShapeError


def test_tensor_invariants_and_nonfinite_diagnostics():
    value = eo.tensor([1, np.nan, np.inf, -np.inf], dtype="float64")
    report = eo.inspect_tensor(value)
    assert (
        report.size,
        report.nan_count,
        report.positive_inf_count,
        report.negative_inf_count,
    ) == (4, 1, 1, 1)
    assert json.loads(json.dumps(report.to_dict()))["shape"] == [4]
    with pytest.raises(ValueError, match="non-finite"):
        eo.assert_tensor_valid(value, finite=True)
    value.grad = np.ones((2,))
    with pytest.raises(EveryOGradientError, match="Gradient shape"):
        eo.assert_tensor_valid(value)


def test_safe_mutation_and_storage_isolation():
    value = eo.tensor([1.0, 2.0], dtype="float64")
    data = np.array([3.0, 4.0])
    assert eo.safe_assign_(value, data) is value
    data[0] = 99.0
    np.testing.assert_array_equal(value.data, [3.0, 4.0])
    with pytest.raises(EveryOShapeError, match="Replacement shape"):
        eo.safe_assign_(value, np.ones(3, dtype="float64"))
    with pytest.raises(TypeError, match="Replacement dtype"):
        eo.safe_assign_(value, np.ones(2, dtype="float32"))
    tracked = eo.tensor([1.0], requires_grad=True)
    with pytest.raises(EveryOGradientError, match="autograd"):
        eo.safe_assign_(tracked, np.ones(1, dtype="float32"))


@pytest.mark.parametrize("seed", range(12))
def test_index_select_duplicate_indices_gradient(seed):
    rng = np.random.default_rng(seed)
    array = rng.normal(size=(4, 5)).astype("float64")
    indices = rng.integers(0, 5, size=7)
    x = eo.tensor(array, requires_grad=True)
    y = eo.index_select(x, 1, indices)
    np.testing.assert_array_equal(y.data, array[:, indices])
    eo.sum(y).backward()
    expected = np.zeros_like(array)
    for index in indices:
        expected[:, index] += 1
    np.testing.assert_array_equal(x.grad, expected)


def test_index_select_validates_inputs():
    x = eo.tensor([[1.0, 2.0]])
    with pytest.raises(EveryOShapeError):
        eo.index_select(x, 2, [0])
    with pytest.raises(TypeError, match="integer"):
        eo.index_select(x, 1, [0.1])


@pytest.mark.parametrize("seed", range(12))
def test_logsumexp_is_stable_and_gradient_is_correct(seed):
    rng = np.random.default_rng(seed)
    array = (rng.normal(size=(3, 4)) * 5 + 900).astype("float64")
    x = eo.tensor(array, requires_grad=True)
    y = eo.logsumexp(x, axis=1)
    expected = np.log(np.exp(array - np.max(array, axis=1, keepdims=True)).sum(axis=1)) + np.max(
        array, axis=1
    )
    np.testing.assert_allclose(y.data, expected, rtol=1e-12)
    eo.sum(y).backward()
    shifted = np.exp(array - np.max(array, axis=1, keepdims=True))
    np.testing.assert_allclose(x.grad, shifted / shifted.sum(axis=1, keepdims=True), atol=1e-12)


def test_logsumexp_scalar_and_errors():
    assert eo.logsumexp(eo.tensor(8.0)).item() == pytest.approx(8.0)
    with pytest.raises(ValueError, match="finite"):
        eo.logsumexp(eo.tensor([np.inf]))
    with pytest.raises(EveryOShapeError, match="empty"):
        eo.logsumexp(eo.tensor([], dtype="float64"))
    with pytest.raises(EveryOShapeError, match="unique"):
        eo.logsumexp(eo.tensor([[2.0]]), axis=(0, 0))


def test_benchmark_is_seeded_and_serializable():
    result = eo.benchmark_cpu(
        "matmul",
        lambda rng: rng.normal(size=(8, 8)) @ rng.normal(size=(8, 8)),
        runs=3,
        warmups=1,
        seed=7,
    )
    assert result.runs == 3 and result.warmups == 1
    assert 0 <= result.min_ms <= result.median_ms <= result.max_ms
    assert json.loads(json.dumps(result.to_dict()))["name"] == "matmul"


def test_benchmark_rejects_nondeterminism_and_invalid_configuration():
    counter = iter(range(5))
    with pytest.raises(ValueError, match="changed"):
        eo.benchmark_cpu("unstable", lambda _: np.array([next(counter)]), runs=2, warmups=0)
    with pytest.raises(ValueError, match="runs"):
        eo.benchmark_cpu("bad", lambda _: np.ones(1), runs=0)


def test_benchmark_regression_guard():
    base = eo.benchmark_cpu("tiny", lambda rng: rng.normal(size=4), runs=2, warmups=0)
    assert eo.compare_benchmarks(
        base, replace(base, median_ms=base.median_ms * 1.1)
    ) == pytest.approx(1.1)
    with pytest.raises(RuntimeError, match="regression"):
        eo.compare_benchmarks(base, replace(base, median_ms=base.median_ms * 1.5))
    with pytest.raises(ValueError, match="environment"):
        eo.compare_benchmarks(base, replace(base, platform="another-machine"))


def test_zero_copy_view_and_gradient():
    x = eo.tensor(np.arange(12, dtype="float64").reshape(3, 4), requires_grad=True)
    y = x.view(2, 6)
    assert np.shares_memory(x.data, y.data)
    eo.sum(y * y).backward()
    np.testing.assert_allclose(x.grad, 2 * x.data)


def test_view_rejects_copy_required():
    x = eo.tensor(np.arange(12, dtype="float64").reshape(3, 4))
    transposed = x.T
    with pytest.raises(EveryOShapeError, match="without copying"):
        transposed.view(12)
    assert transposed.reshape(12).shape == (12,)


def test_mutation_guard_detects_external_numpy_write_before_gradient():
    x = eo.tensor([2.0, 3.0], dtype="float64", requires_grad=True)
    with eo.mutation_guard():
        loss = eo.sum(x * x)
    x.data[0] = 7.0
    with pytest.raises(EveryOGradientError, match="changed after forward"):
        loss.backward()
    assert x.grad is None


def test_mutation_guard_detects_reshape_alias_write():
    x = eo.tensor(np.arange(4, dtype="float64"), requires_grad=True)
    with eo.mutation_guard():
        view = x.view(2, 2)
        loss = eo.sum(view * view)
    view.data[0, 0] = 99
    with pytest.raises(EveryOGradientError, match="changed after forward"):
        loss.backward()
    assert x.grad is None


def test_mutation_guard_allows_unchanged_inputs_and_nested_contexts():
    x = eo.tensor([3.0], requires_grad=True)
    with eo.mutation_guard(), eo.mutation_guard():
        y = x * x
    y.backward()
    np.testing.assert_allclose(x.grad, [6.0])


@pytest.mark.parametrize("seed", range(100))
def test_generated_broadcast_and_reduction_properties(seed):
    """Reproducible generated shape/value corpus; report failed seed via pytest."""
    rng = np.random.default_rng(seed)
    rows = int(rng.integers(1, 6))
    cols = int(rng.integers(1, 6))
    x_data = rng.normal(size=(rows, cols)).astype("float64")
    y_data = rng.normal(size=(1, cols)).astype("float64")
    x = eo.tensor(x_data, requires_grad=True)
    y = eo.tensor(y_data, requires_grad=True)
    result = eo.sum((x + y) * (x + y))
    result.backward()
    expected = 2 * (x_data + y_data)
    np.testing.assert_allclose(x.grad, expected, atol=1e-12)
    np.testing.assert_allclose(y.grad, expected.sum(axis=0, keepdims=True), atol=1e-12)
