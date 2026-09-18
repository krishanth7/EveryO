import numpy as np
import pytest

import everyo as eo
from everyo.quantization import QuantizationError, QuantizedLinear, quantize, quantize_dynamic


def test_per_channel_round_trip_is_close():
    values = np.random.default_rng(0).normal(size=(32, 12)).astype(np.float32)
    packed = quantize(values, axis=1)
    assert packed.values.dtype == np.int8
    np.testing.assert_allclose(packed.dequantize(), values, atol=0.015)


def test_zero_array_is_supported():
    packed = quantize(np.zeros((3, 4), dtype=np.float32), axis=1)
    np.testing.assert_array_equal(packed.dequantize(), 0.0)


def test_non_finite_input_is_rejected():
    with pytest.raises(QuantizationError, match="finite"):
        quantize([1.0, np.nan])


def test_quantized_linear_matches_float_model():
    layer = eo.Linear(16, 7, seed=0).eval()
    inputs = eo.tensor(np.random.default_rng(1).normal(size=(8, 16)).astype(np.float32))
    quantized = QuantizedLinear(layer).eval()
    np.testing.assert_allclose(
        quantized(inputs).numpy(), layer(inputs).numpy(), atol=0.03, rtol=0.03
    )
    assert quantized.qweight.dtype == np.int8
    assert quantized.parameters() == []


def test_quantize_dynamic_replaces_nested_linear_layers_without_mutating_source():
    model = eo.Sequential(eo.Linear(4, 8, seed=0), eo.ReLU(), eo.Linear(8, 2, seed=1))
    converted = quantize_dynamic(model)
    assert isinstance(converted[0], QuantizedLinear)
    assert isinstance(converted[2], QuantizedLinear)
    assert isinstance(model[0], eo.Linear)
    inputs = eo.tensor(np.random.default_rng(2).normal(size=(5, 4)).astype(np.float32))
    np.testing.assert_allclose(
        converted(inputs).numpy(), model.eval()(inputs).numpy(), atol=0.05, rtol=0.05
    )


def test_inplace_conversion():
    model = eo.Sequential(eo.Linear(2, 2, seed=0))
    assert quantize_dynamic(model, inplace=True) is model
    assert isinstance(model[0], QuantizedLinear)
