"""Export a model to ONNX by tracing it, rather than by walking its layers.

:mod:`everyo.serialization.onnx_export` translates a model one layer at a time.
That is precise and readable, but it can only ever export layers it has been
taught, and it cannot express a layer whose behaviour is a *program* rather
than a formula. An LSTM is the clearest case: there is no single ONNX node that
means "this EveryO LSTM". Its meaning is fifty-odd primitive operations in a
particular order, and that order only exists once the layer has actually run.

So this exporter runs it. EveryO's autograd already builds a dynamic graph --
every operation records the tensors it consumed -- and since :class:`Node` also
records the arguments that were *not* tensors (the axis a softmax reduced over,
the permutation a transpose applied), that graph holds everything needed to
re-express the computation in another language. Tracing here means: run the
model once on an example input, then walk the graph it left behind and emit one
ONNX node per EveryO operation.

What that buys: recurrent layers, attention and the transformer blocks export,
because the tracer never needs to know what an LSTM *is*. It only ever sees
adds, matmuls, sigmoids and slices.

**What tracing costs, stated plainly.** A trace records one path through the
model, so:

- **Control flow is flattened.** A loop over ten timesteps becomes ten copies of
  its body, not an ONNX ``Loop``. The export is correct for the sequence length
  traced and is not a general-length model. Trace a different length, get a
  different graph.
- **Shapes are baked in.** ``dynamic_batch=True`` recovers the batch dimension
  by rewriting it to ``-1`` wherever a traced ``Reshape`` names it, and
  :func:`export_onnx_traced` verifies that rewrite really holds by re-running
  the model at a second batch size before it writes the file. Every other
  dimension is fixed.
- **Only what ran is exported.** A branch the example input did not take is not
  in the graph. This is inherent to tracing, and is why the exported graph is
  checked against the model rather than assumed to match it.

The layer-by-layer exporter remains the better choice where it applies: it
produces ``Conv`` nodes rather than the primitives a convolution decomposes
into, and it has no example-input or shape caveats. Use this one for the layers
it cannot reach.

Example:
    >>> import tempfile, pathlib
    >>> import numpy as np
    >>> import everyo as eo
    >>> model = eo.LSTM(4, 3, seed=0)
    >>> _ = model.eval()
    >>> example = np.zeros((1, 5, 4), dtype=np.float32)
    >>> destination = pathlib.Path(tempfile.mkdtemp()) / "lstm.onnx"
    >>> written = eo.export_onnx_traced(model, destination, example_input=example)
    >>> written.exists()
    True
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from everyo._logging import get_logger
from everyo.core import autograd
from everyo.core.autograd import topological_order
from everyo.core.tensor import Tensor
from everyo.exceptions import EveryOSerializationError, EveryOShapeError
from everyo.nn.module import Module
from everyo.serialization.onnx_export import DEFAULT_OPSET, _require_onnx

__all__ = [
    "MINIMUM_OPSET",
    "TRACEABLE_OPERATIONS",
    "export_onnx_traced",
    "trace_operations",
]

_LOGGER = get_logger(__name__)

#: The IR version the exporter stamps on every file it writes.
#:
#: Without this the stamp is whatever the installed ``onnx`` package defaults
#: to, which rises with each release, and ONNX Runtime refuses to load a model
#: whose IR version is newer than it understands ("Unsupported model IR
#: version"). ``pyproject.toml`` allows ``onnxruntime>=1.17``, which supports
#: IR 9, so the export targets that floor rather than the build machine.
IR_VERSION = 9

#: EveryO operations with a direct one-node ONNX equivalent and no arguments.
_DIRECT: dict[str, str] = {
    "abs": "Abs",
    "add": "Add",
    "divide": "Div",
    "dot": "MatMul",
    "exp": "Exp",
    "log": "Log",
    "matmul": "MatMul",
    "multiply": "Mul",
    "negative": "Neg",
    "relu": "Relu",
    "sigmoid": "Sigmoid",
    "sqrt": "Sqrt",
    "subtract": "Sub",
    "tanh": "Tanh",
}

#: Reductions other than ``sum``, and their ONNX operator names.
#:
#: ONNX moved ``axes`` from attribute to input at a *different version for each
#: of these operators*: ``ReduceSum`` at opset 13, but ``ReduceMean``,
#: ``ReduceMax`` and ``ReduceMin`` not until opset 18. Between 13 and 17 the two
#: groups therefore disagree, and emitting the wrong form produces a graph that
#: ``onnx.checker`` rejects outright. :func:`_emit` picks the form from the
#: requested opset rather than assuming the default.
_OTHER_REDUCTIONS = {"mean": "ReduceMean", "max": "ReduceMax", "min": "ReduceMin"}

#: ``axes`` became an input to ReduceSum here.
_REDUCE_SUM_AXES_INPUT_OPSET = 13

#: ``axes`` became an input to the other reductions here.
_REDUCE_AXES_INPUT_OPSET = 18

#: The oldest opset this exporter will emit.
#:
#: Not an arbitrary floor. Before opset 13, ONNX's ``Softmax`` and
#: ``LogSoftmax`` *coerce their input to 2-D* and normalise over the flattened
#: trailing dimensions, which is not what EveryO's softmax computes on a 3-D or
#: 4-D tensor. Emitting a ``Softmax`` node against those schemas would produce a
#: graph that loads and runs and quietly returns different numbers -- the worst
#: possible outcome -- so the exporter refuses instead.
MINIMUM_OPSET = 13

#: Every operation the tracer can translate.
TRACEABLE_OPERATIONS: tuple[str, ...] = tuple(
    sorted(
        {
            *_DIRECT,
            *_OTHER_REDUCTIONS,
            "clip",
            "concatenate",
            "log_softmax",
            "power",
            "reshape",
            "slice",
            "softmax",
            "sum",
            "transpose",
        }
    )
)


class _TracedGraph:
    """Accumulates ONNX nodes and initializers while walking a traced graph."""

    def __init__(self, helper: Any, numpy_helper: Any) -> None:
        self._helper = helper
        self._numpy_helper = numpy_helper
        self.nodes: list[Any] = []
        self.initializers: list[Any] = []
        self._counts: dict[str, int] = {}

    def name(self, prefix: str) -> str:
        index = self._counts.get(prefix, 0)
        self._counts[prefix] = index + 1
        return f"{prefix}_{index}"

    def constant(self, array: np.ndarray, prefix: str, dtype: Any = np.float32) -> str:
        name = self.name(prefix)
        tensor = np.ascontiguousarray(array, dtype=dtype)
        self.initializers.append(self._numpy_helper.from_array(tensor, name))
        return name

    def node(self, op_type: str, inputs: list[str], prefix: str, **attributes: Any) -> str:
        output = self.name(prefix)
        self.nodes.append(
            self._helper.make_node(
                op_type, inputs, [output], name=self.name(f"{prefix}_node"), **attributes
            )
        )
        return output


def _as_axes_list(axes: Any, rank: int) -> list[int]:
    """Normalise a recorded ``axes`` attribute to a concrete list.

    ``None`` means "every axis" in EveryO, which ONNX spells out explicitly.
    """
    if axes is None:
        return list(range(rank))
    if isinstance(axes, int):
        return [int(axes)]
    return [int(axis) for axis in axes]


def _slice_parameters(
    key: Any, in_shape: tuple[int, ...]
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Turn a NumPy index into ONNX ``Slice`` starts/ends/axes/steps.

    Integer entries are emitted as a one-element slice; the dimension they drop
    in NumPy is restored by the caller's reshape to the traced output shape,
    which keeps this function from having to track rank changes itself.
    """
    entries = key if isinstance(key, tuple) else (key,)
    if any(entry is Ellipsis or entry is None for entry in entries):
        raise EveryOSerializationError(
            "The tracing exporter cannot translate an index containing Ellipsis "
            "or None (np.newaxis). Index with explicit slices instead."
        )
    if len(entries) > len(in_shape):
        raise EveryOShapeError(
            f"Index {key!r} has more entries than the tensor's {len(in_shape)} dimension(s)."
        )

    starts: list[int] = []
    ends: list[int] = []
    axes: list[int] = []
    steps: list[int] = []
    for axis, entry in enumerate(entries):
        extent = in_shape[axis]
        if isinstance(entry, slice):
            start, stop, step = entry.indices(extent)
            if step <= 0:
                raise EveryOSerializationError(
                    "The tracing exporter supports forward slices only; "
                    f"axis {axis} was indexed with step {step}."
                )
        elif isinstance(entry, (int, np.integer)):
            start = int(entry) + (extent if int(entry) < 0 else 0)
            stop, step = start + 1, 1
        else:
            raise EveryOSerializationError(
                f"The tracing exporter cannot translate index entry {entry!r} on "
                f"axis {axis}. Supported entries are integers and slices."
            )
        starts.append(start)
        ends.append(stop)
        axes.append(axis)
        steps.append(step)
    return starts, ends, axes, steps


def _emit(
    operation: str,
    attributes: dict[str, Any],
    inputs: list[str],
    parents: tuple[Tensor, ...],
    out_shape: tuple[int, ...],
    graph: _TracedGraph,
    opset: int,
) -> str:
    """Emit the ONNX node(s) for one traced operation and return its output."""
    if operation in _DIRECT:
        return graph.node(_DIRECT[operation], inputs, operation)

    if operation in ("softmax", "log_softmax"):
        op = "Softmax" if operation == "softmax" else "LogSoftmax"
        return graph.node(op, inputs, operation, axis=int(attributes["axis"]))

    if operation == "transpose":
        axes = attributes.get("axes")
        # EveryO's default (None) reverses the axes, exactly like NumPy.
        perm = list(reversed(range(len(parents[0].shape)))) if axes is None else list(axes)
        return graph.node("Transpose", inputs, "transpose", perm=[int(a) for a in perm])

    if operation == "concatenate":
        return graph.node("Concat", inputs, "concat", axis=int(attributes["axis"]))

    if operation == "reshape":
        shape = graph.constant(np.asarray(attributes["shape"], dtype=np.int64), "shape", np.int64)
        return graph.node("Reshape", [inputs[0], shape], "reshape")

    if operation == "power":
        exponent = graph.constant(np.asarray(attributes["exponent"], dtype=np.float32), "exponent")
        return graph.node("Pow", [inputs[0], exponent], "pow")

    if operation == "clip":
        low = graph.constant(np.asarray(attributes["low"], dtype=np.float32), "clip_low")
        high = graph.constant(np.asarray(attributes["high"], dtype=np.float32), "clip_high")
        return graph.node("Clip", [inputs[0], low, high], "clip")

    if operation == "sum" or operation in _OTHER_REDUCTIONS:
        op_type = "ReduceSum" if operation == "sum" else _OTHER_REDUCTIONS[operation]
        threshold = _REDUCE_SUM_AXES_INPUT_OPSET if operation == "sum" else _REDUCE_AXES_INPUT_OPSET
        axes = _as_axes_list(attributes.get("axes"), len(parents[0].shape))
        keepdims = 1 if attributes.get("keepdims") else 0

        if opset >= threshold:
            axes_input = graph.constant(np.asarray(axes, dtype=np.int64), "reduce_axes", np.int64)
            return graph.node(
                op_type, [inputs[0], axes_input], f"reduce_{operation}", keepdims=keepdims
            )
        return graph.node(op_type, [inputs[0]], f"reduce_{operation}", axes=axes, keepdims=keepdims)

    if operation == "slice":
        starts, ends, axes, steps = _slice_parameters(attributes["key"], parents[0].shape)
        sliced = graph.node(
            "Slice",
            [
                inputs[0],
                graph.constant(np.asarray(starts, dtype=np.int64), "starts", np.int64),
                graph.constant(np.asarray(ends, dtype=np.int64), "ends", np.int64),
                graph.constant(np.asarray(axes, dtype=np.int64), "slice_axes", np.int64),
                graph.constant(np.asarray(steps, dtype=np.int64), "steps", np.int64),
            ],
            "slice",
        )
        # An integer index drops its axis in NumPy but not in ONNX. Rather than
        # track that here, restore the rank the trace actually produced.
        sliced_rank = len(parents[0].shape)
        if sliced_rank != len(out_shape):
            shape = graph.constant(np.asarray(out_shape, dtype=np.int64), "shape", np.int64)
            sliced = graph.node("Reshape", [sliced, shape], "slice_squeeze")
        return sliced

    raise EveryOSerializationError(
        f"The tracing exporter cannot translate the operation {operation!r}. "
        f"It knows: {', '.join(TRACEABLE_OPERATIONS)}."
    )


def _run_trace(model: Module, example: np.ndarray) -> tuple[Tensor, Tensor]:
    """Run ``model`` once and return ``(input_tensor, output_tensor)``.

    The input is marked ``requires_grad`` because EveryO only records graph
    nodes for operations that touch something differentiable. Without it the
    forward pass runs and leaves no trace at all.
    """
    source = Tensor(np.ascontiguousarray(example, dtype=np.float32), requires_grad=True)
    with autograd.enable_grad():
        output = model(source)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, Tensor):
        raise EveryOSerializationError(
            f"Tracing expects the model to return a Tensor, but it returned "
            f"{type(output).__name__}."
        )
    if output.grad_node is None:
        raise EveryOSerializationError(
            "Tracing the model recorded no operations. This happens when the "
            "forward pass runs under no_grad(), or when the model returns its "
            "input unchanged."
        )
    return source, output


def trace_operations(model: Module, example_input: np.ndarray) -> list[str]:
    """Return the operation names one forward pass records, in execution order.

    Useful on its own for seeing what a layer decomposes into before exporting
    it, and used by the tests to pin the decomposition.

    Example:
        >>> import numpy as np
        >>> import everyo as eo
        >>> model = eo.RNN(2, 2, seed=0)
        >>> _ = model.eval()
        >>> ops = eo.trace_operations(model, np.zeros((1, 3, 2), dtype=np.float32))
        >>> "tanh" in ops and "matmul" in ops
        True
    """
    _, output = _run_trace(model, example_input)
    return [
        tensor.grad_node.operation
        for tensor in reversed(topological_order(output))
        if tensor.grad_node is not None
    ]


def _rewrite_batch_dimension(graph: _TracedGraph, numpy_helper: Any, batch: int) -> None:
    """Replace the traced batch size with ``-1`` in every Reshape target.

    A traced ``Reshape`` names concrete sizes, which pins the graph to the batch
    it was traced with. Rewriting a leading dimension equal to the traced batch
    to ``-1`` lets ONNX infer it. Only the leading dimension is touched, and
    only when no other ``-1`` is already present, so a reshape that folds the
    batch into another axis is left alone rather than silently mangled.

    This is a heuristic, which is why :func:`export_onnx_traced` re-runs the
    model at a different batch size and compares before trusting it.
    """
    reshape_inputs = {node.input[1] for node in graph.nodes if node.op_type == "Reshape"}
    for index, initializer in enumerate(graph.initializers):
        if initializer.name not in reshape_inputs:
            continue
        values = numpy_helper.to_array(initializer)
        if values.size == 0 or int(values[0]) != batch or -1 in values.tolist():
            continue
        rewritten = values.copy()
        rewritten[0] = -1
        graph.initializers[index] = numpy_helper.from_array(rewritten, initializer.name)


def export_onnx_traced(
    model: Module,
    path: str | Path,
    *,
    example_input: np.ndarray,
    input_name: str = "input",
    output_name: str = "output",
    opset: int = DEFAULT_OPSET,
    model_name: str = "everyo_traced_model",
    dynamic_batch: bool = True,
) -> Path:
    """Trace ``model`` on ``example_input`` and write the result to ``path``.

    Args:
        model: The model to export, in eval mode.
        path: Destination file.
        example_input: A real input array. Its shape decides the exported
            graph's shapes, and its values decide which path through the model
            is recorded, so it should be representative rather than empty.
        input_name: Name of the graph input.
        output_name: Name of the graph output.
        opset: ONNX opset version to target. Must be at least
            :data:`MINIMUM_OPSET`; reductions are emitted against the schema
            that version actually defines, since ONNX moved ``axes`` from
            attribute to input at a different version for each of them.
        model_name: Graph name recorded in the file.
        dynamic_batch: Rewrite the batch dimension to be symbolic, and verify
            that rewrite against a second batch size before writing. When the
            check fails the export falls back to the traced batch size, which
            is always correct, and logs that it did so.

    Returns:
        The path that was written.

    Raises:
        EveryOSerializationError: If ``onnx`` is missing, the model is in
            training mode, the trace is empty, or it contains an operation the
            exporter cannot translate.
    """
    onnx, TensorProto, helper, numpy_helper = _require_onnx()

    if getattr(model, "training", False):
        raise EveryOSerializationError(
            "The model is in training mode. Tracing would record training-time "
            "behaviour (dropout sampling, batch norm on batch statistics), so "
            "call model.eval() first and the exported graph will match what you "
            "get back from the model."
        )

    if opset < MINIMUM_OPSET:
        raise EveryOSerializationError(
            f"The tracing exporter targets opset {MINIMUM_OPSET} or newer, but "
            f"{opset} was requested. Before opset {MINIMUM_OPSET}, ONNX's "
            "Softmax and LogSoftmax coerce their input to 2-D and normalise "
            "over the flattened trailing dimensions, which is not what EveryO "
            "computes. The resulting graph would load, run, and return "
            "different numbers, so it is refused rather than written."
        )

    example = np.ascontiguousarray(example_input, dtype=np.float32)
    if example.ndim == 0:
        raise EveryOShapeError("example_input must have at least one dimension.")

    source, output = _run_trace(model, example)
    graph = _TracedGraph(helper, numpy_helper)

    # topological_order() lists consumers before producers; emission needs the
    # reverse, so that every operand is named before it is used.
    ordered = list(reversed(topological_order(output)))
    names: dict[int, str] = {id(source): input_name}

    for tensor in ordered:
        node = tensor.grad_node
        if node is None:
            if id(tensor) not in names:
                # A leaf that is not the input is a learned parameter or a
                # constant the model built; either way its value travels with
                # the graph.
                names[id(tensor)] = graph.constant(tensor.data, "const")
            continue
        missing = [parent for parent in node.parents if id(parent) not in names]
        for parent in missing:
            names[id(parent)] = graph.constant(parent.data, "const")
        names[id(tensor)] = _emit(
            node.operation,
            node.attributes,
            [names[id(parent)] for parent in node.parents],
            node.parents,
            tensor.shape,
            graph,
            opset,
        )

    if not graph.nodes:
        raise EveryOSerializationError("Tracing produced an empty graph; nothing to export.")

    graph.nodes[-1].output[0] = output_name

    batch = int(example.shape[0])
    in_dims: list[Any] = list(example.shape)
    out_dims: list[Any] = list(output.shape)

    # A symbolic batch is only meaningful if the *output* still has a batch on
    # axis 0. A model that reduces the batch away (a scalar loss) or moves it
    # (a transpose) does not, and labelling its leading dimension "batch"
    # writes a signature that does not describe the graph. The array-level
    # check below compares values, so it would not catch a wrong label.
    output_carries_batch = bool(output.shape) and int(output.shape[0]) == batch
    dynamic = dynamic_batch and output_carries_batch
    if dynamic_batch and not output_carries_batch:
        _LOGGER.info(
            "The traced output does not have the batch on axis 0 (input batch %d, "
            "output shape %s), so the graph is exported with fixed shapes.",
            batch,
            tuple(output.shape),
        )

    if dynamic:
        _rewrite_batch_dimension(graph, numpy_helper, batch)
        in_dims[0] = "batch"
        out_dims[0] = "batch"

    proto = _build_model(
        onnx,
        TensorProto,
        helper,
        graph,
        model_name,
        input_name,
        output_name,
        in_dims,
        out_dims,
        opset,
    )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(destination))

    if dynamic and not _batch_rewrite_holds(model, example, destination, input_name):
        # The heuristic did not survive a second batch size. A graph pinned to
        # the traced batch is still a correct graph, so fall back rather than
        # ship one that silently misbehaves off the traced shape.
        _LOGGER.warning(
            "dynamic_batch could not be verified for this model; exporting with "
            "the traced batch size of %d instead.",
            batch,
        )
        return export_onnx_traced(
            model,
            destination,
            example_input=example,
            input_name=input_name,
            output_name=output_name,
            opset=opset,
            model_name=model_name,
            dynamic_batch=False,
        )

    _LOGGER.info("Traced %d operation(s) into %s (opset %d).", len(graph.nodes), destination, opset)
    return destination


def _build_model(
    onnx: Any,
    TensorProto: Any,
    helper: Any,
    graph: _TracedGraph,
    model_name: str,
    input_name: str,
    output_name: str,
    in_dims: list[Any],
    out_dims: list[Any],
    opset: int,
) -> Any:
    """Assemble, stamp and check the ONNX model proto."""
    onnx_graph = helper.make_graph(
        graph.nodes,
        model_name,
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, in_dims)],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, out_dims)],
        initializer=graph.initializers,
    )
    proto = helper.make_model(
        onnx_graph,
        producer_name="everyo",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    proto.ir_version = IR_VERSION
    onnx.checker.check_model(proto)
    return proto


#: ONNX Runtime's own default log severity (2 = warning).
#:
#: It exposes a setter but no getter, so the level to restore has to be named
#: rather than read back.
_ORT_DEFAULT_SEVERITY = 2


@contextmanager
def _quiet_onnxruntime() -> Iterator[None]:
    """Silence ONNX Runtime's error log for the duration of the block.

    Nothing this function does is important enough to fail the caller, and the
    caller reads *any* exception from the block as "the rewrite does not hold".
    So every step here is guarded: a problem adjusting a log level must never
    be mistaken for a problem with the exported graph.
    """
    restore = False
    try:
        import onnxruntime

        onnxruntime.set_default_logger_severity(4)  # 4 = fatal only
        restore = True
    except Exception:  # noqa: BLE001 - logging is best-effort
        pass
    try:
        yield
    finally:
        if restore:
            with suppress(Exception):  # logging is best-effort
                onnxruntime.set_default_logger_severity(_ORT_DEFAULT_SEVERITY)


def _batch_rewrite_holds(model: Module, example: np.ndarray, path: Path, input_name: str) -> bool:
    """Return whether the exported graph still matches the model off the traced batch.

    Run at a *different* batch size than the trace, because that is the only
    thing the rewrite changed and therefore the only thing worth checking.
    """
    try:
        from everyo.serialization.onnx_export import run_onnx
    except ImportError:  # pragma: no cover - defensive
        return True

    doubled = np.concatenate([example, example], axis=0)
    # A refusal below is an expected outcome, not a problem to report, so ONNX
    # Runtime's own error logging is turned down for the probe: otherwise a
    # successful fallback still prints a red kernel error that reads like a
    # broken export when it is the check working.
    with _quiet_onnxruntime():
        try:
            exported = run_onnx(path, doubled, input_name=input_name)
        except Exception as error:  # noqa: BLE001 - any runtime refusal means "no"
            _LOGGER.debug("dynamic_batch verification could not run: %s", error)
            return False

    # The model itself can refuse the doubled batch -- a forward pass is free to
    # validate x.shape[0]. That is a failed probe, not an error to propagate:
    # letting it escape would abort the export *after* the provisional dynamic
    # file was written, leaving an unverified graph on disk.
    try:
        with autograd.no_grad():
            expected = model(Tensor(doubled))
    except Exception as error:  # noqa: BLE001 - the model refusing is an answer
        _LOGGER.debug("the model itself rejected the probe batch: %s", error)
        return False

    if isinstance(expected, tuple):
        expected = expected[0]

    # The batch has to have actually doubled. Comparing values alone would
    # accept a model whose output does not scale with the batch.
    if not expected.shape or int(expected.shape[0]) != doubled.shape[0]:
        return False

    return bool(
        np.asarray(exported).shape == tuple(expected.shape)
        and np.allclose(np.asarray(exported), expected.data, rtol=1e-4, atol=1e-5)
    )
