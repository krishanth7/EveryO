"""Export a trained EveryO model to ONNX.

ONNX is how a model leaves EveryO: once a graph is written out, ONNX Runtime,
TensorRT, CoreML and the rest can serve it without EveryO installed at all.
This module walks a model's layers and emits the equivalent ONNX nodes.

Two details are worth knowing before you read the code.

**Layout.** EveryO uses ``NHWC`` for images, matching TensorFlow. ONNX's
``Conv``, ``MaxPool`` and ``AveragePool`` are defined on ``NCHW``. Rather than
pretend the two agree, every convolution and pooling node is wrapped in a pair
of ``Transpose`` nodes, and the kernel is permuted from ``(kh, kw, in, out)`` to
ONNX's ``(out, in, kh, kw)``. The transposes are cheap and ONNX Runtime's graph
optimizer usually cancels adjacent pairs, but they are really there: the
exported graph is faithful, not fast.

**Scope.** This exporter covers the feed-forward layers, listed in
:data:`SUPPORTED_LAYERS`. The recurrent layers, attention and the transformer
blocks are *not* exported — they unroll into long chains of primitives that
would need a tracing exporter rather than a layer-by-layer one, and shipping a
half-correct translation of them would be worse than shipping none. Passing one
raises :class:`~everyo.exceptions.EveryOSerializationError` naming the layer.

Padding is written out as explicit ``pads`` attributes computed by EveryO's own
:func:`~everyo.core.convolution._resolve_padding`, not as ``auto_pad``, so a
``"same"`` convolution exports to exactly the asymmetric padding EveryO used.

See :func:`export_onnx` for the entry point and :func:`run_onnx` for checking
an export against the model it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from everyo._logging import get_logger
from everyo.core.convolution import _pair, _resolve_padding
from everyo.exceptions import EveryOSerializationError, EveryOShapeError
from everyo.nn.module import Module

__all__ = [
    "DEFAULT_OPSET",
    "SUPPORTED_LAYERS",
    "export_onnx",
    "onnx_available",
    "run_onnx",
]

_LOGGER = get_logger(__name__)

#: The opset the exporter targets. 17 is the first with ``LayerNormalization``.
DEFAULT_OPSET = 17

#: Layer class names this exporter can translate.
SUPPORTED_LAYERS = (
    "AvgPool2D",
    "BatchNorm1D",
    "BatchNorm2D",
    "Conv2D",
    "Dropout",
    "Flatten",
    "LayerNorm",
    "Linear",
    "LogSoftmax",
    "MaxPool2D",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
)

# NHWC -> NCHW and back.
_TO_NCHW = [0, 3, 1, 2]
_TO_NHWC = [0, 2, 3, 1]


def onnx_available() -> bool:
    """Return ``True`` when the optional ``onnx`` package is importable."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        return False
    return True


def _require_onnx():
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise EveryOSerializationError(
            "ONNX export needs the optional 'onnx' package. Install it with "
            "'pip install everyo[onnx]' or 'pip install onnx'."
        ) from error
    return onnx, TensorProto, helper, numpy_helper


class _GraphBuilder:
    """Accumulates ONNX nodes and initializers while walking a model."""

    def __init__(self, helper: Any, numpy_helper: Any) -> None:
        self._helper = helper
        self._numpy_helper = numpy_helper
        self.nodes: list[Any] = []
        self.initializers: list[Any] = []
        self._counts: dict[str, int] = {}

    def name(self, prefix: str) -> str:
        """Return a unique name built from ``prefix``."""
        index = self._counts.get(prefix, 0)
        self._counts[prefix] = index + 1
        return f"{prefix}_{index}"

    def constant(self, array: np.ndarray, prefix: str) -> str:
        """Add ``array`` as a graph initializer and return its name."""
        name = self.name(prefix)
        tensor = np.ascontiguousarray(array, dtype=np.float32)
        self.initializers.append(self._numpy_helper.from_array(tensor, name))
        return name

    def node(self, op_type: str, inputs: list[str], prefix: str, **attributes: Any) -> str:
        """Append a single-output node and return the name of that output."""
        output = self.name(prefix)
        self.nodes.append(
            self._helper.make_node(
                op_type, inputs, [output], name=self.name(f"{prefix}_node"), **attributes
            )
        )
        return output


def _spatial_padding(layer: Any, height: int, width: int) -> list[int]:
    """Return ONNX ``pads`` ([top, left, bottom, right]) for a windowed layer."""
    window = _pair(getattr(layer, "kernel_size", None) or layer.pool_size, "window")
    stride = _pair(layer.stride if layer.stride is not None else window, "stride")
    pad_h = _resolve_padding(layer.padding, height, window[0], stride[0], "padding")
    pad_w = _resolve_padding(layer.padding, width, window[1], stride[1], "padding")
    return [pad_h[0], pad_w[0], pad_h[1], pad_w[1]]


def _window_and_stride(layer: Any) -> tuple[list[int], list[int]]:
    window = _pair(getattr(layer, "kernel_size", None) or layer.pool_size, "window")
    stride = _pair(layer.stride if layer.stride is not None else window, "stride")
    return list(window), list(stride)


def _export_layer(
    layer: Module,
    value: str,
    shape: tuple[int, ...],
    builder: _GraphBuilder,
) -> tuple[str, tuple[int, ...]]:
    """Emit the nodes for one layer.

    Args:
        layer: The layer to translate.
        value: Name of the tensor flowing in.
        shape: Static shape of that tensor, tracked so windowed layers can
            resolve ``"same"`` padding exactly as EveryO did.
        builder: Accumulator for nodes and initializers.

    Returns:
        The output tensor name and its shape.
    """
    kind = type(layer).__name__

    if kind == "Sequential":
        for child in layer:
            value, shape = _export_layer(child, value, shape, builder)
        return value, shape

    if kind == "Linear":
        weight = builder.constant(layer.weight.data, "linear_weight")
        value = builder.node("MatMul", [value, weight], "linear")
        if layer.use_bias:
            bias = builder.constant(layer.bias.data, "linear_bias")
            value = builder.node("Add", [value, bias], "linear_bias_add")
        return value, (*shape[:-1], layer.out_features)

    if kind in {"ReLU", "Sigmoid", "Tanh"}:
        return builder.node(kind.replace("ReLU", "Relu"), [value], kind.lower()), shape

    if kind in {"Softmax", "LogSoftmax"}:
        op = "Softmax" if kind == "Softmax" else "LogSoftmax"
        return builder.node(op, [value], op.lower(), axis=layer.axis), shape

    if kind == "Dropout":
        # Inference-time dropout is the identity. The export is of the model in
        # eval mode, which export_onnx() enforces, so emitting nothing is exact.
        return value, shape

    if kind == "Flatten":
        start = layer.start_dim if layer.start_dim >= 0 else len(shape) + layer.start_dim
        flattened = int(np.prod(shape[start:])) if shape[start:] else 1
        return (
            builder.node("Flatten", [value], "flatten", axis=start),
            (*shape[:start], flattened),
        )

    if kind == "Conv2D":
        if len(shape) != 4:
            raise EveryOShapeError(f"Conv2D export expects a 4-D NHWC shape, got {shape}.")
        _, height, width, _ = shape
        window, stride = _window_and_stride(layer)
        pads = _spatial_padding(layer, height, width)

        # (kh, kw, in, out) -> (out, in, kh, kw)
        kernel = builder.constant(np.transpose(layer.weight.data, (3, 2, 0, 1)), "conv_weight")
        inputs = [builder.node("Transpose", [value], "to_nchw", perm=_TO_NCHW), kernel]
        if layer.use_bias:
            inputs.append(builder.constant(layer.bias.data, "conv_bias"))

        convolved = builder.node(
            "Conv", inputs, "conv", kernel_shape=window, strides=stride, pads=pads
        )
        out_h = (height + pads[0] + pads[2] - window[0]) // stride[0] + 1
        out_w = (width + pads[1] + pads[3] - window[1]) // stride[1] + 1
        return (
            builder.node("Transpose", [convolved], "to_nhwc", perm=_TO_NHWC),
            (shape[0], out_h, out_w, layer.out_channels),
        )

    if kind in {"MaxPool2D", "AvgPool2D"}:
        if len(shape) != 4:
            raise EveryOShapeError(f"{kind} export expects a 4-D NHWC shape, got {shape}.")
        _, height, width, channels = shape
        window, stride = _window_and_stride(layer)
        pads = _spatial_padding(layer, height, width)

        transposed = builder.node("Transpose", [value], "to_nchw", perm=_TO_NCHW)
        if kind == "MaxPool2D":
            pooled = builder.node(
                "MaxPool",
                [transposed],
                "maxpool",
                kernel_shape=window,
                strides=stride,
                pads=pads,
            )
        else:
            # EveryO divides each window by the number of *real* cells it
            # covers, so padded cells must not count towards the average.
            pooled = builder.node(
                "AveragePool",
                [transposed],
                "avgpool",
                kernel_shape=window,
                strides=stride,
                pads=pads,
                count_include_pad=0,
            )
        out_h = (height + pads[0] + pads[2] - window[0]) // stride[0] + 1
        out_w = (width + pads[1] + pads[3] - window[1]) // stride[1] + 1
        return (
            builder.node("Transpose", [pooled], "to_nhwc", perm=_TO_NHWC),
            (shape[0], out_h, out_w, channels),
        )

    if kind in {"BatchNorm1D", "BatchNorm2D"}:
        if not layer.track_running_stats:
            raise EveryOSerializationError(
                f"{kind} was built with track_running_stats=False, so it has no "
                "inference-time statistics to export. ONNX has no equivalent of "
                "normalising over whatever batch happens to arrive."
            )
        # Folded into a single affine transform on the channel axis. ONNX's own
        # BatchNormalization is defined on NCHW; folding sidesteps the layout
        # question entirely and is numerically identical.
        scale = 1.0 / np.sqrt(np.asarray(layer.running_var, dtype=np.float32) + layer.eps)
        shift = -np.asarray(layer.running_mean, dtype=np.float32) * scale
        if layer.affine:
            gamma = np.asarray(layer.weight.data, dtype=np.float32)
            beta = np.asarray(layer.bias.data, dtype=np.float32)
            shift = shift * gamma + beta
            scale = scale * gamma
        value = builder.node("Mul", [value, builder.constant(scale, "bn_scale")], "bn_scale_mul")
        return builder.node(
            "Add", [value, builder.constant(shift, "bn_shift")], "bn_shift_add"
        ), shape

    if kind == "LayerNorm":
        rank = len(layer.normalized_shape)
        ones = np.ones(layer.normalized_shape, dtype=np.float32)
        zeros = np.zeros(layer.normalized_shape, dtype=np.float32)
        scale = np.asarray(layer.weight.data, dtype=np.float32) if layer.affine else ones
        bias = np.asarray(layer.bias.data, dtype=np.float32) if layer.affine else zeros
        return (
            builder.node(
                "LayerNormalization",
                [
                    value,
                    builder.constant(scale, "ln_scale"),
                    builder.constant(bias, "ln_bias"),
                ],
                "layernorm",
                axis=-rank,
                epsilon=float(layer.eps),
            ),
            shape,
        )

    raise EveryOSerializationError(
        f"ONNX export does not support {kind}. Supported layers are: "
        f"{', '.join(SUPPORTED_LAYERS)}. Recurrent layers, attention and the "
        "transformer blocks unroll into primitive chains that need a tracing "
        "exporter, which EveryO does not have yet."
    )


def export_onnx(
    model: Module,
    path: str | Path,
    *,
    input_shape: tuple[int, ...],
    input_name: str = "input",
    output_name: str = "output",
    opset: int = DEFAULT_OPSET,
    model_name: str = "everyo_model",
) -> Path:
    """Write ``model`` to ``path`` as an ONNX graph.

    Args:
        model: The model to export. It must be in eval mode, because the
            exported graph has inference semantics for dropout and batch norm.
        path: Destination file.
        input_shape: Static shape of one input batch, in EveryO's own layout
            (``NHWC`` for images). The batch dimension is exported as the
            symbolic dimension ``"batch"``, so the graph accepts any batch size.
        input_name: Name of the graph input.
        output_name: Name of the graph output.
        opset: ONNX opset version to target.
        model_name: Graph name recorded in the file.

    Returns:
        The path that was written.

    Raises:
        EveryOSerializationError: If ``onnx`` is not installed, the model is in
            training mode, or it contains a layer the exporter cannot translate.

    Example:
        >>> import tempfile, pathlib
        >>> import everyo as eo
        >>> model = eo.Sequential(eo.Linear(4, 3, seed=0), eo.ReLU())
        >>> _ = model.eval()
        >>> destination = pathlib.Path(tempfile.mkdtemp()) / "model.onnx"
        >>> written = eo.export_onnx(model, destination, input_shape=(1, 4))
        >>> written.exists()
        True
    """
    onnx, TensorProto, helper, numpy_helper = _require_onnx()

    if getattr(model, "training", False):
        raise EveryOSerializationError(
            "The model is in training mode. ONNX export writes inference "
            "semantics (dropout off, batch norm on running statistics), so call "
            "model.eval() first and the exported graph will match what you get "
            "back from the model."
        )

    shape = tuple(int(dimension) for dimension in input_shape)
    if not shape:
        raise EveryOShapeError("input_shape must have at least one dimension.")

    builder = _GraphBuilder(helper, numpy_helper)
    output, output_shape = _export_layer(model, input_name, shape, builder)

    # Rename the last node's output so the graph output has the requested name.
    if builder.nodes and output != input_name:
        builder.nodes[-1].output[0] = output_name
        output = output_name
    elif output == input_name:
        output = builder.node("Identity", [input_name], "identity")
        builder.nodes[-1].output[0] = output_name
        output = output_name

    def _dims(dimensions: tuple[int, ...]) -> list[Any]:
        return ["batch", *list(dimensions[1:])]

    graph = helper.make_graph(
        builder.nodes,
        model_name,
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, _dims(shape))],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, _dims(output_shape))],
        initializer=builder.initializers,
    )
    proto = helper.make_model(
        graph,
        producer_name="everyo",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(proto)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(destination))
    _LOGGER.info("Exported %d node(s) to %s (opset %d).", len(builder.nodes), destination, opset)
    return destination


def run_onnx(path: str | Path, inputs: np.ndarray, *, input_name: str = "input") -> np.ndarray:
    """Run an exported graph through ONNX Runtime and return its output.

    This exists so an export can be checked against the model it came from
    rather than merely declared correct. It needs the optional ``onnxruntime``
    package.

    Raises:
        EveryOSerializationError: If ``onnxruntime`` is not installed.
    """
    try:
        import onnxruntime
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise EveryOSerializationError(
            "Running an exported graph needs the optional 'onnxruntime' package. "
            "Install it with 'pip install everyo[onnx]' or 'pip install onnxruntime'."
        ) from error

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {input_name: np.ascontiguousarray(inputs, dtype=np.float32)})[0]
