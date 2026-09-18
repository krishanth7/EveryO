"""Tests for ONNX export.

Every test that claims an export is correct proves it by running the exported
graph through ONNX Runtime and comparing against the EveryO model it came from.
Checking that a file was written, or that ``onnx.checker`` is happy with it,
would only establish that the graph is well-formed -- not that it computes the
same function.

Both packages are optional, so the round-trip tests skip when they are missing
rather than silently passing.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOSerializationError
from everyo.serialization.onnx_export import onnx_available

try:  # onnxruntime is optional and separate from onnx itself.
    import onnxruntime  # noqa: F401

    runtime_available = True
except ImportError:  # pragma: no cover - depends on the environment
    runtime_available = False

requires_onnx = pytest.mark.skipif(not onnx_available(), reason="onnx is not installed.")
requires_runtime = pytest.mark.skipif(
    not (onnx_available() and runtime_available),
    reason="onnx and onnxruntime are both needed to run an exported graph.",
)

# float32 arithmetic reordered by a different runtime; this is rounding, not drift.
TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}


def _round_trip(model, shape, tmp_path, name="model"):
    """Export ``model``, run both it and the export, and return the two outputs."""
    model.eval()
    path = eo.export_onnx(model, tmp_path / f"{name}.onnx", input_shape=(1, *shape[1:]))
    inputs = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    expected = np.asarray(model(eo.tensor(inputs)).data)
    actual = eo.run_onnx(path, inputs)
    return expected, actual


@requires_runtime
class TestRoundTrip:
    def test_mlp(self, tmp_path):
        model = eo.Sequential(eo.Linear(4, 8, seed=0), eo.ReLU(), eo.Linear(8, 3, seed=1))
        expected, actual = _round_trip(model, (6, 4), tmp_path, "mlp")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_mlp_without_bias(self, tmp_path):
        model = eo.Sequential(eo.Linear(4, 3, bias=False, seed=0), eo.Tanh())
        expected, actual = _round_trip(model, (5, 4), tmp_path, "nobias")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    @pytest.mark.parametrize("activation", ["ReLU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax"])
    def test_activations(self, tmp_path, activation):
        model = eo.Sequential(eo.Linear(4, 5, seed=0), getattr(eo, activation)())
        expected, actual = _round_trip(model, (3, 4), tmp_path, activation)
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_cnn_with_valid_padding(self, tmp_path):
        model = eo.Sequential(
            eo.Conv2D(3, 8, 3, padding="valid", seed=0),
            eo.ReLU(),
            eo.MaxPool2D(2),
            eo.Flatten(),
            eo.Linear(8 * 3 * 3, 4, seed=1),
        )
        expected, actual = _round_trip(model, (6, 8, 8, 3), tmp_path, "cnn")
        assert actual.shape == (6, 4)
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_same_padding_and_stride(self, tmp_path):
        # An odd input with stride 2 and "same" padding is where TensorFlow's
        # asymmetric padding rule bites; getting this right is the point of
        # exporting explicit pads instead of auto_pad.
        model = eo.Sequential(
            eo.Conv2D(3, 6, 3, stride=2, padding="same", seed=0),
            eo.Tanh(),
            eo.AvgPool2D(2, padding="same"),
        )
        expected, actual = _round_trip(model, (4, 7, 7, 3), tmp_path, "same")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_average_pooling_ignores_padded_cells(self, tmp_path):
        # EveryO divides by the count of real cells; ONNX must be told not to
        # count the padding, or the border values come out too small.
        model = eo.Sequential(eo.AvgPool2D(2, padding="same"))
        expected, actual = _round_trip(model, (2, 5, 5, 3), tmp_path, "avgpool")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_batch_norm_uses_trained_running_statistics(self, tmp_path):
        model = eo.Sequential(eo.Linear(6, 5, seed=0), eo.BatchNorm1D(5), eo.ReLU())
        # Train the running statistics away from their (0, 1) initialisation,
        # so the export is actually carrying learned numbers.
        model.train()
        rng = np.random.default_rng(1)
        for _ in range(20):
            model(eo.tensor(rng.normal(loc=3.0, scale=2.0, size=(16, 6)).astype(np.float32)))
        batch_norm = model[1]
        assert not np.allclose(batch_norm.running_mean, 0.0)

        expected, actual = _round_trip(model, (4, 6), tmp_path, "bn1d")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_batch_norm_2d(self, tmp_path):
        model = eo.Sequential(
            eo.Conv2D(2, 4, 3, padding="same", seed=0), eo.BatchNorm2D(4), eo.Sigmoid()
        )
        model.train()
        rng = np.random.default_rng(2)
        for _ in range(10):
            model(eo.tensor(rng.normal(size=(8, 5, 5, 2)).astype(np.float32)))
        expected, actual = _round_trip(model, (3, 5, 5, 2), tmp_path, "bn2d")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_layer_norm(self, tmp_path):
        model = eo.Sequential(eo.Linear(6, 8, seed=0), eo.LayerNorm(8))
        expected, actual = _round_trip(model, (4, 6), tmp_path, "ln")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_dropout_exports_as_the_identity(self, tmp_path):
        model = eo.Sequential(eo.Linear(4, 4, seed=0), eo.Dropout(0.5), eo.ReLU())
        expected, actual = _round_trip(model, (3, 4), tmp_path, "dropout")
        np.testing.assert_allclose(actual, expected, **TOLERANCE)

    def test_batch_dimension_is_symbolic(self, tmp_path):
        model = eo.Sequential(eo.Linear(4, 2, seed=0))
        model.eval()
        path = eo.export_onnx(model, tmp_path / "sym.onnx", input_shape=(1, 4))
        for batch in (1, 7, 32):
            inputs = np.ones((batch, 4), dtype=np.float32)
            assert eo.run_onnx(path, inputs).shape == (batch, 2)


@requires_onnx
class TestExportContract:
    def test_training_mode_is_refused(self, tmp_path):
        model = eo.Sequential(eo.Linear(4, 2, seed=0))
        model.train()
        with pytest.raises(EveryOSerializationError, match="training mode"):
            eo.export_onnx(model, tmp_path / "x.onnx", input_shape=(1, 4))

    def test_unsupported_layer_is_named(self, tmp_path):
        model = eo.Sequential(eo.LSTM(4, 4, seed=0))
        model.eval()
        with pytest.raises(EveryOSerializationError, match="LSTM"):
            eo.export_onnx(model, tmp_path / "x.onnx", input_shape=(1, 3, 4))

    def test_untracked_batch_norm_is_refused(self, tmp_path):
        model = eo.Sequential(eo.BatchNorm1D(4, track_running_stats=False))
        model.eval()
        with pytest.raises(EveryOSerializationError, match="track_running_stats"):
            eo.export_onnx(model, tmp_path / "x.onnx", input_shape=(1, 4))

    def test_the_graph_passes_the_onnx_checker(self, tmp_path):
        import onnx

        model = eo.Sequential(
            eo.Conv2D(3, 4, 3, padding="same", seed=0),
            eo.Flatten(),
            eo.Linear(4 * 8 * 8, 2, seed=1),
        )
        model.eval()
        path = eo.export_onnx(model, tmp_path / "checked.onnx", input_shape=(1, 8, 8, 3))
        onnx.checker.check_model(onnx.load(str(path)))

    def test_output_name_is_honoured(self, tmp_path):
        import onnx

        model = eo.Sequential(eo.Linear(4, 2, seed=0))
        model.eval()
        path = eo.export_onnx(
            model, tmp_path / "named.onnx", input_shape=(1, 4), output_name="logits"
        )
        proto = onnx.load(str(path))
        assert [o.name for o in proto.graph.output] == ["logits"]
        assert [i.name for i in proto.graph.input] == ["input"]

    def test_parent_directories_are_created(self, tmp_path):
        model = eo.Sequential(eo.Linear(4, 2, seed=0))
        model.eval()
        path = eo.export_onnx(model, tmp_path / "nested" / "dir" / "m.onnx", input_shape=(1, 4))
        assert path.exists()
