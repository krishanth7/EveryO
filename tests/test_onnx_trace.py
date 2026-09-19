"""Tests for the tracing ONNX exporter.

The central claim is that a traced export *computes the same thing* as the
model it came from, so almost every test here runs the exported graph through
ONNX Runtime and compares. Asserting that a file was written would prove very
little.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.core import autograd
from everyo.exceptions import EveryOSerializationError
from everyo.serialization.onnx_trace import IR_VERSION, _slice_parameters

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def _reference(model, array: np.ndarray) -> np.ndarray:
    """Run the EveryO model the way inference would."""
    with autograd.no_grad():
        output = model(eo.tensor(array))
    if isinstance(output, tuple):
        output = output[0]
    return np.asarray(output.data)


def _roundtrip(model, array: np.ndarray, path, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Export, run through ONNX Runtime, and return ``(exported, expected)``."""
    model.eval()
    eo.export_onnx_traced(model, path, example_input=array, **kwargs)
    return np.asarray(eo.run_onnx(path, array)), _reference(model, array)


# The layers the layer-by-layer exporter refuses. Exporting these is the whole
# point of the tracer, so each one is checked end to end.
RECURRENT_AND_ATTENTION = [
    pytest.param(lambda: eo.RNN(4, 5, seed=0), (2, 3, 4), id="rnn"),
    pytest.param(lambda: eo.LSTM(4, 5, seed=0), (2, 3, 4), id="lstm"),
    pytest.param(lambda: eo.GRU(4, 5, seed=0), (2, 3, 4), id="gru"),
    pytest.param(lambda: eo.MultiHeadAttention(8, 2, seed=0), (2, 3, 8), id="attention"),
    pytest.param(lambda: eo.TransformerEncoderBlock(8, 2, seed=0), (2, 3, 8), id="encoder_block"),
    pytest.param(
        lambda: eo.TransformerEncoder(8, 2, num_layers=2, seed=0), (2, 3, 8), id="encoder"
    ),
]


@pytest.mark.parametrize("build,shape", RECURRENT_AND_ATTENTION)
def test_traced_export_matches_the_model(build, shape, tmp_path):
    """Every layer the other exporter cannot reach round-trips through ONNX."""
    array = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    exported, expected = _roundtrip(build(), array, tmp_path / "model.onnx")

    assert exported.shape == expected.shape
    np.testing.assert_allclose(exported, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("build,shape", RECURRENT_AND_ATTENTION)
def test_the_layer_exporter_refuses_and_names_the_tracer(build, shape, tmp_path):
    """The refusal has to point somewhere, now that somewhere exists."""
    model = build()
    model.eval()
    with pytest.raises(EveryOSerializationError, match="export_onnx_traced"):
        eo.export_onnx(model, tmp_path / "model.onnx", input_shape=shape)


def test_feedforward_models_trace_too(tmp_path):
    """The tracer is not limited to the layers the other exporter refuses."""
    model = eo.Sequential(eo.Linear(4, 6, seed=0), eo.ReLU(), eo.Linear(6, 2, seed=1))
    array = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    exported, expected = _roundtrip(model, array, tmp_path / "model.onnx")
    np.testing.assert_allclose(exported, expected, rtol=1e-5, atol=1e-6)


def test_attention_export_accepts_a_different_batch_size(tmp_path):
    """dynamic_batch is only worth having if a second batch size really works."""
    model = eo.MultiHeadAttention(8, 2, seed=0)
    array = np.random.default_rng(2).standard_normal((2, 3, 8)).astype(np.float32)
    destination = tmp_path / "model.onnx"
    exported, expected = _roundtrip(model, array, destination)
    np.testing.assert_allclose(exported, expected, rtol=1e-4, atol=1e-5)

    graph_input = onnx.load(destination).graph.input[0]
    assert graph_input.type.tensor_type.shape.dim[0].dim_param == "batch"

    larger = np.random.default_rng(3).standard_normal((5, 3, 8)).astype(np.float32)
    np.testing.assert_allclose(
        eo.run_onnx(destination, larger), _reference(model, larger), rtol=1e-4, atol=1e-5
    )


def test_recurrent_export_falls_back_to_a_fixed_batch(tmp_path):
    """The dynamic-batch rewrite is a heuristic, and it does not hold here.

    A recurrent layer reshapes ``(batch, time, features)`` into
    ``(batch * time, features)``. The leading dimension of that reshape is not
    the batch, so rewriting it to -1 would be wrong -- and the exporter's own
    verification catches that and exports a fixed batch instead. A graph that
    is honest about accepting one batch size beats a graph that claims to
    accept any and then miscomputes.
    """
    model = eo.LSTM(4, 5, seed=0)
    array = np.random.default_rng(4).standard_normal((2, 3, 4)).astype(np.float32)
    destination = tmp_path / "model.onnx"
    exported, expected = _roundtrip(model, array, destination)
    np.testing.assert_allclose(exported, expected, rtol=1e-4, atol=1e-5)

    dimension = onnx.load(destination).graph.input[0].type.tensor_type.shape.dim[0]
    assert dimension.dim_value == 2, "expected the traced batch, not a symbolic one"
    assert not dimension.dim_param


def test_the_graph_is_stamped_for_the_oldest_supported_runtime(tmp_path):
    """A newer onnxruntime accepts an IR version an older one rejects.

    That asymmetry means loading the file here cannot catch the problem, so the
    stamp is read out of the proto directly.
    """
    model = eo.LSTM(4, 3, seed=0)
    model.eval()
    destination = tmp_path / "model.onnx"
    eo.export_onnx_traced(model, destination, example_input=np.zeros((1, 2, 4), dtype=np.float32))
    assert onnx.load(destination).ir_version == IR_VERSION <= 9


def test_training_mode_is_refused(tmp_path):
    """Tracing a training-mode model would record dropout sampling."""
    model = eo.TransformerEncoderBlock(8, 2, seed=0)
    model.train()
    with pytest.raises(EveryOSerializationError, match="training mode"):
        eo.export_onnx_traced(
            model,
            tmp_path / "model.onnx",
            example_input=np.zeros((1, 2, 8), dtype=np.float32),
        )


def test_a_model_that_records_nothing_is_reported_clearly():
    """An empty trace is a specific failure and deserves a specific message."""

    class Passthrough(eo.Module):
        def forward(self, x):
            return x

    model = Passthrough()
    model.eval()
    with pytest.raises(EveryOSerializationError, match="recorded no operations"):
        eo.trace_operations(model, np.zeros((1, 3), dtype=np.float32))


def test_trace_operations_reports_the_decomposition():
    """An LSTM is gates and arithmetic once traced -- that is why it exports."""
    model = eo.LSTM(4, 5, seed=0)
    model.eval()
    operations = eo.trace_operations(model, np.zeros((2, 3, 4), dtype=np.float32))

    assert set(operations) <= set(eo.TRACEABLE_OPERATIONS), "traced an untranslatable operation"
    # The gates themselves: three sigmoids and a tanh per timestep.
    assert operations.count("sigmoid") == 9
    assert operations.count("tanh") == 6
    assert "matmul" in operations


def test_a_longer_sequence_traces_a_longer_graph():
    """Tracing flattens the timestep loop, and the docs say so. Pin it."""
    model = eo.LSTM(4, 5, seed=0)
    model.eval()
    short = eo.trace_operations(model, np.zeros((1, 2, 4), dtype=np.float32))
    long = eo.trace_operations(model, np.zeros((1, 6, 4), dtype=np.float32))
    assert len(long) > len(short)
    assert long.count("sigmoid") == 3 * short.count("sigmoid")


class TestSliceTranslation:
    """Indexing is where a trace most easily goes subtly wrong."""

    def test_a_plain_slice_becomes_starts_and_ends(self):
        starts, ends, axes, steps = _slice_parameters((slice(1, 3), slice(None)), (5, 4))
        assert (starts, ends, axes, steps) == ([1, 0], [3, 4], [0, 1], [1, 1])

    def test_an_integer_index_becomes_a_one_element_slice(self):
        starts, ends, axes, steps = _slice_parameters(2, (5, 4))
        assert (starts, ends, axes, steps) == ([2], [3], [0], [1])

    def test_a_negative_index_is_resolved_against_the_shape(self):
        starts, ends, _, _ = _slice_parameters(-1, (5, 4))
        assert (starts, ends) == ([4], [5])

    def test_ellipsis_is_refused_rather_than_guessed(self):
        with pytest.raises(EveryOSerializationError, match="Ellipsis"):
            _slice_parameters((Ellipsis, slice(None)), (5, 4))

    def test_a_reversed_slice_is_refused(self):
        with pytest.raises(EveryOSerializationError, match="forward slices only"):
            _slice_parameters(slice(None, None, -1), (5,))

    def test_too_many_index_entries_is_a_shape_error(self):
        with pytest.raises(Exception, match="more entries than"):
            _slice_parameters((0, 0, 0), (5, 4))


def test_integer_indexing_round_trips_through_onnx(tmp_path):
    """An integer index drops a dimension in NumPy but not in ONNX.

    A recurrent layer indexes a timestep exactly this way, so getting the rank
    back is load-bearing rather than cosmetic.
    """

    class LastStep(eo.Module):
        def forward(self, x):
            return eo.relu(x[:, -1])

    model = LastStep()
    array = np.random.default_rng(5).standard_normal((2, 4, 3)).astype(np.float32)
    exported, expected = _roundtrip(model, array, tmp_path / "model.onnx")
    assert exported.shape == expected.shape == (2, 3)
    np.testing.assert_allclose(exported, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "operation,build",
    [
        ("mean", lambda x: eo.mean(x, axis=1, keepdims=True)),
        ("sum", lambda x: eo.sum(x, axis=1, keepdims=True)),
        ("max", lambda x: eo.max(x, axis=1, keepdims=True)),
        ("min", lambda x: eo.min(x, axis=1, keepdims=True)),
        ("softmax", lambda x: eo.softmax(x, axis=-1)),
        ("log_softmax", lambda x: eo.log_softmax(x, axis=-1)),
        ("transpose", lambda x: eo.transpose(x, (0, 2, 1))),
        ("sqrt", lambda x: eo.sqrt(eo.abs(x))),
        ("power", lambda x: eo.power(x, 2.0)),
        ("clip", lambda x: eo.clip(x, -0.5, 0.5)),
        ("concatenate", lambda x: eo.concatenate([x, x], axis=2)),
    ],
)
def test_each_traced_operation_matches_onnx_runtime(operation, build, tmp_path):
    """ONNX moved several of these between attribute and input across opsets.

    Getting one wrong yields a graph the checker accepts and the runtime
    rejects, so each is exercised against the runtime rather than the checker.
    """

    class Wrapper(eo.Module):
        def forward(self, x):
            return build(x)

    array = np.random.default_rng(6).standard_normal((2, 3, 4)).astype(np.float32)
    exported, expected = _roundtrip(Wrapper(), array, tmp_path / f"{operation}.onnx")
    assert exported.shape == expected.shape
    np.testing.assert_allclose(exported, expected, rtol=1e-5, atol=1e-6)
