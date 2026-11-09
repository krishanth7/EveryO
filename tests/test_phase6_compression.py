"""Phase 6 CPU-only model compression and inference acceptance checks."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.compression import (
    FakeQuantLinear,
    benchmark_inference,
    compare_model_outputs,
    distillation_loss,
    dynamic_batch_predict,
    export_onnx_compatible,
    fake_quantize,
    model_size_report,
    prune_linear_channels,
    prune_unstructured,
)


def _network():
    return eo.Sequential(eo.Linear(4, 10, seed=7), eo.ReLU(), eo.Linear(10, 3, seed=8))


def test_unstructured_pruning_exact_zero_count_and_original_untouched():
    model = _network()
    original = model.state_dict()
    compressed = prune_unstructured(model, 0.5)
    assert compressed is not model
    for name, array in original.items():
        np.testing.assert_array_equal(model.state_dict()[name], array)
    for layer in (compressed[0], compressed[2]):
        expected = int(layer.weight.size * 0.5)
        assert int(np.count_nonzero(layer.weight.data == 0)) >= expected
    with pytest.raises(ValueError):
        prune_unstructured(model, 1)


def test_structured_pruning_reduces_parameter_count_and_runs():
    first, second = prune_linear_channels(eo.Linear(4, 10, seed=7), eo.Linear(10, 3, seed=8), 0.4)
    assert first.weight.shape == (4, 6)
    assert second.weight.shape == (6, 3)
    assert first(eo.zeros(5, 4)).shape == (5, 6)
    assert second(first(eo.zeros(5, 4))).shape == (5, 3)
    with pytest.raises(ValueError):
        prune_linear_channels(eo.Linear(3, 4), eo.Linear(5, 2), 0.2)


def test_fake_quant_forward_and_ste_backward():
    x = eo.tensor(np.array([0.13, -0.4, 0.7], dtype=np.float32), requires_grad=True)
    quantized = fake_quantize(x, bits=4)
    assert not np.array_equal(quantized.data, x.data)
    quantized.sum().backward()
    np.testing.assert_array_equal(x.grad, np.ones(3))
    with pytest.raises(ValueError):
        fake_quantize(x, bits=1)
    qat = FakeQuantLinear(eo.Linear(3, 2, seed=2), bits=8)
    out = qat(eo.tensor(np.ones((4, 3), dtype=np.float32)))
    out.sum().backward()
    assert qat.weight.grad is not None
    assert np.all(np.isfinite(qat.weight.grad))


def test_distillation_gradient_and_teacher_detachment():
    student = eo.tensor(np.array([[0.1, 0.3, -0.2], [0.4, 0.2, -0.1]]), requires_grad=True)
    teacher = eo.tensor(np.array([[0.2, 0.1, -0.1], [0.9, -0.2, 0.5]]), requires_grad=True)
    loss = distillation_loss(student, teacher, temperature=2)
    assert loss.item() >= -1e-7
    loss.backward()
    assert np.all(np.isfinite(student.grad))
    assert teacher.grad is None
    with pytest.raises(ValueError):
        distillation_loss(student, teacher, temperature=0)


def test_dynamic_batch_and_output_comparison():
    model = _network()
    data = np.random.default_rng(2).normal(size=(19, 4)).astype(np.float32)
    model.train()
    expected = model(eo.tensor(data)).numpy()
    batched = dynamic_batch_predict(model, data, batch_size=5)
    assert model.training
    np.testing.assert_allclose(batched, expected, atol=1e-6)
    assert compare_model_outputs(expected, batched).within_tolerance
    assert not compare_model_outputs(expected, batched + 10).within_tolerance
    assert model_size_report(model)["state_bytes"] > 0
    speed = benchmark_inference(model, data, warmup=1, repetitions=2, batch_size=5)
    assert speed["min_ms"] > 0
    with pytest.raises(ValueError):
        dynamic_batch_predict(model, data[:0])


def test_quantized_onnx_expansion_if_available(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    assert onnx is not None
    model = eo.quantize_dynamic(_network())
    data = np.random.default_rng(4).normal(size=(3, 4)).astype(np.float32)
    report = export_onnx_compatible(
        model, tmp_path / "compressed.onnx", example=data, atol=1e-5, rtol=1e-5
    )
    assert report["batch_sizes"] == [3, 6]
    assert (tmp_path / "compressed.onnx").is_file()
