"""CPU-first model compression, distillation and inference qualification tools."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.autograd import no_grad
from everyo.core.tensor import Tensor, as_tensor
from everyo.nn.layers import Linear
from everyo.nn.module import Module, Parameter
from everyo.quantization import QuantizedLinear

__all__ = [
    "FakeQuantLinear",
    "fake_quantize",
    "prune_unstructured",
    "prune_linear_channels",
    "distillation_loss",
    "model_size_report",
    "benchmark_inference",
    "dynamic_batch_predict",
    "compare_model_outputs",
    "export_onnx_compatible",
    "OutputComparison",
]


def _validate_fraction(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError("sparsity must be finite and in [0, 1)")
    return value


def prune_unstructured(model: Module, sparsity: float, *, inplace: bool = False) -> Module:
    """Magnitude-prune floating weight matrices, preserving their shape.

    Produces a zero-valued mask applied at call time, not a permanent pruning
    constraint: continued training may regrow zero weights unless masks are
    reapplied. Unstructured sparsity does not imply a NumPy speedup.
    """
    fraction = _validate_fraction(sparsity)
    result = model if inplace else copy.deepcopy(model)
    for name, parameter in result.named_parameters():
        if not name.endswith("weight") or parameter.ndim < 2:
            continue
        data = parameter.data
        cut = int(data.size * fraction)
        if cut:
            # Stable mergesort makes tied magnitudes deterministic.
            order = np.argsort(np.abs(data).reshape(-1), kind="stable")
            flattened = data.copy().reshape(-1)
            flattened[order[:cut]] = 0
            parameter.data = flattened.reshape(data.shape)
    return result


def prune_linear_channels(first: Linear, second: Linear, fraction: float) -> tuple[Linear, Linear]:
    """Prune output channels from one Linear and matching inputs of the next.

    Only adjacent Linear pairs, or pairs separated by a channel-preserving
    elementwise activation, are valid. Caller must preserve that topology.
    """
    fraction = _validate_fraction(fraction)
    if first.out_features != second.in_features:
        raise ValueError("adjacent Linear dimensions do not match")
    width = first.out_features
    kept = max(1, width - int(width * fraction))
    scores = np.linalg.norm(first.weight.data.astype(np.float64), axis=0)
    selected = np.sort(np.argsort(-scores, kind="stable")[:kept])
    left = Linear(first.in_features, kept, bias=first.use_bias)
    right = Linear(kept, second.out_features, bias=second.use_bias)
    left.weight.data = first.weight.data[:, selected].copy()
    right.weight.data = second.weight.data[selected, :].copy()
    if first.use_bias:
        left.bias.data = first.bias.data[selected].copy()
    if second.use_bias:
        right.bias.data = second.bias.data.copy()
    return left, right


def fake_quantize(value: Any, *, bits: int = 8) -> Tensor:
    """Symmetric fake quantization with identity STE backward (QAT research).

    The forward output is float with quantization noise. Backward passes the
    incoming gradient unchanged, so this is *not* an exact derivative of round.
    """
    if not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError("bits must be an integer in [2, 8]")
    item = as_tensor(value)
    source = item.data
    if not np.all(np.isfinite(source)):
        raise FloatingPointError("cannot fake-quantize non-finite values")
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    limit = (1 << (bits - 1)) - 1
    scale = max(peak / limit, float(np.finfo(np.float32).tiny))
    rounded = (np.clip(np.rint(source.astype(np.float64) / scale), -limit, limit) * scale).astype(
        source.dtype
    )
    return ops._make(rounded, [item], "fake_quant", lambda gradient: (gradient.copy(),))


class FakeQuantLinear(Module):
    """QAT Linear module with full-precision trainable weights and fake int8 operations."""

    def __init__(self, layer: Linear, *, bits: int = 8) -> None:
        super().__init__()
        if not 2 <= bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.bits = bits
        self.use_bias = layer.use_bias
        self.weight = Parameter(layer.weight.data.copy())
        if self.use_bias:
            self.bias = Parameter(layer.bias.data.copy())

    def forward(self, x: Any) -> Tensor:
        output = ops.matmul(
            fake_quantize(x, bits=self.bits), fake_quantize(self.weight, bits=self.bits)
        )
        return ops.add(output, self.bias) if self.use_bias else output


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Any,
    *,
    temperature: float = 2.0,
    hard_loss: Tensor | None = None,
    alpha: float = 0.5,
) -> Tensor:
    """Temperature-scaled teacher/student KL, optionally mixed with hard loss.

    Teacher output is detached, so its parameters are never differentiated.
    Logits must be 2D (batch, classes) with at least two classes.
    """
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError("temperature must be positive and alpha in [0, 1]")
    teacher = as_tensor(teacher_logits)
    if (
        student_logits.ndim != 2
        or student_logits.shape != teacher.shape
        or student_logits.shape[1] < 2
    ):
        raise ValueError("teacher and student logits must have matching (batch, classes) shapes")
    with no_grad():
        probs = ops.softmax(
            Tensor(teacher.data, device=student_logits.device) / temperature, axis=-1
        ).data
    clipped_log = np.log(np.clip(probs, np.finfo(np.float64).tiny, 1))
    student_log = ops.log_softmax(student_logits / temperature, axis=-1)
    kl = ops.mean(
        ops.sum(
            as_tensor(probs * clipped_log, device=student_logits.device)
            - as_tensor(probs, device=student_logits.device) * student_log,
            axis=-1,
        )
    )
    soft = kl * temperature**2
    if hard_loss is None:
        return soft
    return alpha * hard_loss + (1 - alpha) * soft


@dataclass(frozen=True)
class OutputComparison:
    max_absolute_error: float
    max_relative_error: float
    within_tolerance: bool
    compared_elements: int


def compare_model_outputs(
    expected: Any, actual: Any, *, atol: float = 1e-5, rtol: float = 1e-4
) -> OutputComparison:
    """Finite, shape-aware numerical equivalence report."""
    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be nonnegative")
    x = np.asarray(expected.data if isinstance(expected, Tensor) else expected)
    y = np.asarray(actual.data if isinstance(actual, Tensor) else actual)
    if x.shape != y.shape or not x.size:
        raise ValueError("outputs must have identical nonempty shapes")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("outputs must be finite")
    diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
    relative = diff / np.maximum(np.abs(x.astype(np.float64)), np.finfo(np.float64).tiny)
    return OutputComparison(
        float(np.max(diff)),
        float(np.max(relative)),
        bool(np.allclose(x, y, atol=atol, rtol=rtol)),
        int(x.size),
    )


def model_size_report(model: Module) -> dict[str, int]:
    """Parameter/buffer footprint (not Python object or activation memory)."""
    state = model.state_dict()
    return {
        "tensors": len(state),
        "parameters": model.num_parameters(),
        "state_bytes": int(sum(array.nbytes for array in state.values())),
    }


def dynamic_batch_predict(model: Module, data: Any, *, batch_size: int = 64) -> np.ndarray:
    """Run arbitrary batch lengths without retaining autograd graphs."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    array = np.asarray(data)
    if array.ndim < 2 or not len(array):
        raise ValueError("data must contain at least one row and a feature dimension")
    was_training = model.training
    model.eval()
    try:
        with no_grad():
            outputs = [
                model(as_tensor(array[start : start + batch_size])).data.copy()
                for start in range(0, len(array), batch_size)
            ]
        return np.concatenate(outputs, axis=0)
    finally:
        model.train(was_training)


def benchmark_inference(
    model: Module, data: Any, *, repetitions: int = 10, warmup: int = 2, batch_size: int = 64
) -> dict[str, float]:
    """Measure CPU wall time for whole-batch inference, never assert speedups."""
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup nonnegative")
    for _ in range(warmup):
        dynamic_batch_predict(model, data, batch_size=batch_size)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        dynamic_batch_predict(model, data, batch_size=batch_size)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": float(np.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "repetitions": float(repetitions),
    }


def _exportable(model: Module) -> Module:
    """Convert quantized Linear wrappers to float Linear layers for ONNX export."""
    result = copy.deepcopy(model)

    def convert(module: Module) -> Module:
        if isinstance(module, (QuantizedLinear, FakeQuantLinear)):
            layer = Linear(module.in_features, module.out_features, bias=module.use_bias)
            weights = module.weight if isinstance(module, QuantizedLinear) else module.weight.data
            layer.weight.data = np.asarray(weights, dtype=np.float32).copy()
            if module.use_bias:
                bias = module.bias if isinstance(module, QuantizedLinear) else module.bias.data
                layer.bias.data = np.asarray(bias, dtype=np.float32).copy()
            return layer
        for name, child in list(module._modules.items()):  # noqa: SLF001 - controlled internal tree
            setattr(module, name, convert(child))
        return module

    return convert(result).eval()


def export_onnx_compatible(
    model: Module, path: str | Path, *, example: np.ndarray, atol: float = 0.05, rtol: float = 0.05
) -> dict[str, Any]:
    """Export quantized/QAT models via float weights and *verify* two batch sizes.

    Rejects the export when quantization noise exceeds caller tolerances.
    Requires optional ``onnx`` and ``onnxruntime`` packages.
    """
    from everyo.serialization.onnx_export import export_onnx, run_onnx

    array = np.asarray(example, dtype=np.float32)
    if array.ndim < 2 or not len(array):
        raise ValueError("example must be a nonempty batch")
    converted = _exportable(model)
    export_onnx(converted, path, input_shape=array.shape)
    results = []
    for sample in (array, np.concatenate([array, array], axis=0)):
        expected = dynamic_batch_predict(model, sample)
        actual = run_onnx(path, sample)
        check = compare_model_outputs(expected, actual, atol=atol, rtol=rtol)
        if not check.within_tolerance:
            raise ValueError(f"ONNX output drift exceeds tolerance: {check.max_absolute_error}")
        results.append(check)
    return {
        "path": str(path),
        "batch_sizes": [len(array), 2 * len(array)],
        "max_absolute_error": max(check.max_absolute_error for check in results),
    }
