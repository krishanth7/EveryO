"""CPU-first Phase 8 fine-tuning, LoRA, instruction data and evaluation.

These are educational, small-model utilities. They never download a model,
require an accelerator, or silently train frozen base parameters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.autograd import Node, no_grad
from everyo.core.tensor import Tensor, as_tensor
from everyo.language_model import CharTokenizer, TinyDecoderLM
from everyo.nn.layers import Linear
from everyo.nn.losses import CrossEntropyLoss
from everyo.nn.module import Module, Parameter, register_module
from everyo.optim import Adam
from everyo.precision import GradScaler, autocast

__all__ = [
    "LoRALinear",
    "LoRATinyDecoderLM",
    "InstructionBatch",
    "prepare_instruction_batch",
    "masked_next_token_loss",
    "gradient_checkpoint",
    "FineTuneReport",
    "fine_tune_instructions",
    "evaluate_instructions",
]


class LoRALinear(Module):
    """Frozen base linear layer plus trainable low-rank A @ B update.

    The adapter is initialized to produce exactly zero update, preserving the
    original layer's predictions until fine-tuning starts.
    """

    def __init__(self, base: Linear, *, rank: int = 2, alpha: float = 2.0, seed: int = 0):
        super().__init__()
        if not isinstance(base, Linear) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("expected Linear base, positive rank and finite positive alpha")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.scale = self.alpha / self.rank
        self.in_features, self.out_features = base.in_features, base.out_features
        for parameter in base.parameters():
            parameter.requires_grad = False
        rng = np.random.default_rng(seed)
        self.lora_a = Parameter(rng.normal(0, 0.02, (self.in_features, rank)).astype(np.float32))
        self.lora_b = Parameter(np.zeros((rank, self.out_features), dtype=np.float32))

    def forward(self, x: Any) -> Tensor:
        input_tensor = as_tensor(x)
        update = ops.matmul(ops.matmul(input_tensor, self.lora_a), self.lora_b)
        return self.base(input_tensor) + update * self.scale

    def merged_weight(self) -> np.ndarray:
        """Return effective weight without mutating or unfreezing the base."""
        return self.base.weight.data.copy() + self.scale * (self.lora_a.data @ self.lora_b.data)


@register_module
class LoRATinyDecoderLM(TinyDecoderLM):
    """Serializable TinyDecoderLM whose output head is a LoRA adapter."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        max_context: int = 32,
        seed: int = 0,
        rank: int = 2,
        alpha: float = 2.0,
    ) -> None:
        super().__init__(vocab_size, embed_dim, num_heads, num_layers, max_context, seed)
        self.rank, self.alpha = rank, alpha
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.head = LoRALinear(self.head, rank=rank, alpha=alpha, seed=seed + 101)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "rank": self.rank, "alpha": self.alpha}

    def load_base(self, source: TinyDecoderLM) -> None:
        """Copy a compatible base model while leaving adapters zero-initialized."""
        if not isinstance(source, TinyDecoderLM) or (
            self.vocab_size,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.max_context,
        ) != (
            source.vocab_size,
            source.embed_dim,
            source.num_heads,
            source.num_layers,
            source.max_context,
        ):
            raise ValueError("incompatible base language model configuration")
        source_state = source.state_dict()
        for name, parameter in self.named_parameters():
            source_name = name.replace("head.base.", "head.")
            if name.startswith("head.lora_"):
                continue
            if source_name in source_state:
                parameter.data = source_state[source_name].copy()

    def cached_step(self, ids: Any, cache: Any = None) -> tuple[np.ndarray, Any]:
        """Cached decoding via projected adapter head without autograd graph."""
        # The parent uses self.head(...) and works with any Module head.
        return super().cached_step(ids, cache)


@dataclass(frozen=True)
class InstructionBatch:
    """Teacher-forced token inputs and target supervision mask."""

    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray


def prepare_instruction_batch(
    pairs: Sequence[tuple[str, str]],
    tokenizer: CharTokenizer,
    *,
    max_context: int,
    separator: str = "\nAnswer: ",
) -> InstructionBatch:
    """Build padded SFT examples, supervising answer tokens only.

    Tokens not present in the tokenizer vocabulary are rejected, rather than
    quietly converting instructions and answers into ``<unk>``.
    """
    if not pairs or max_context < 2:
        raise ValueError("require nonempty instruction pairs and context >= 2")
    examples: list[tuple[list[int], list[int], list[bool]]] = []
    for prompt, completion in pairs:
        if not isinstance(prompt, str) or not isinstance(completion, str) or not completion:
            raise ValueError("each instruction must have a string prompt and nonempty answer")
        prefix = f"Instruction: {prompt}{separator}"
        encoded_prefix = tokenizer.encode(prefix)
        encoded_answer = tokenizer.encode(completion)
        if 1 in encoded_prefix + encoded_answer:
            raise ValueError("instruction contains characters outside tokenizer vocabulary")
        sequence = encoded_prefix + encoded_answer
        if len(sequence) < 2 or len(sequence) > max_context + 1:
            raise ValueError("instruction example exceeds max_context")
        inputs, targets = sequence[:-1], sequence[1:]
        mask = [position >= len(encoded_prefix) - 1 for position in range(len(targets))]
        examples.append((inputs, targets, mask))
    length = max(len(row[0]) for row in examples)
    x = np.zeros((len(examples), length), dtype=np.int64)
    y = np.zeros_like(x)
    m = np.zeros_like(x, dtype=np.bool_)
    for index, (inputs, targets, mask) in enumerate(examples):
        width = len(inputs)
        x[index, :width], y[index, :width], m[index, :width] = inputs, targets, mask
    return InstructionBatch(x, y, m)


def masked_next_token_loss(logits: Tensor, targets: Any, mask: Any) -> Tensor:
    """Differentiable CE over selected (batch, time) targets only."""
    labels, active = np.asarray(targets), np.asarray(mask)
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or active.shape != labels.shape:
        raise ValueError("expected logits (B,T,V) with matching targets and mask")
    if not np.issubdtype(labels.dtype, np.integer) or active.dtype != np.bool_:
        raise TypeError("targets must be integer IDs and loss_mask must be boolean")
    positions = np.flatnonzero(active.reshape(-1))
    if positions.size == 0:
        raise ValueError("loss mask contains no supervised tokens")
    selected = ops.slice_(ops.reshape(logits, (-1, logits.shape[-1])), positions)
    return CrossEntropyLoss()(selected, labels.reshape(-1)[positions])


def gradient_checkpoint(
    function: Callable[[Tensor], Tensor],
    x: Tensor,
    parameters: Iterable[Parameter],
) -> Tensor:
    """Recompute a pure function on backward, storing only forward output.

    The callable must be deterministic and free of mutable forward side
    effects (no dropout or BatchNorm running-stat updates). Parameter values
    must not be changed between forward and backward.
    """
    if not isinstance(x, Tensor):
        raise TypeError("checkpoint input must be an EveryO Tensor")
    params = tuple(dict.fromkeys(parameters))
    with no_grad():
        result = function(x)
    if not isinstance(result, Tensor):
        raise TypeError("checkpointed function must return Tensor")
    parents: tuple[Tensor, ...] = (x, *params)
    if not any(parent.requires_grad for parent in parents):
        return result
    saved_x, saved_params = x.data.copy(), tuple(p.data.copy() for p in params)

    def backward_fn(upstream: np.ndarray) -> tuple[np.ndarray | None, ...]:
        if any(not np.array_equal(p.data, saved) for p, saved in zip(params, saved_params)):
            raise ValueError("checkpoint parameters mutated before backward")
        recomputed_input = Tensor(saved_x, requires_grad=True)
        prior = [p.grad for p in params]
        try:
            for param in params:
                param.grad = None
            recomputed = function(recomputed_input)
            recomputed.backward(upstream)
            gradients = (recomputed_input.grad, *(p.grad for p in params))
        finally:
            for param, previous in zip(params, prior):
                param.grad = previous
        return tuple(
            None if not parent.requires_grad else grad for parent, grad in zip(parents, gradients)
        )

    return Tensor._wrap(
        result.data.copy(), device=x.device, node=Node("gradient_checkpoint", parents, backward_fn)
    )


@dataclass(frozen=True)
class FineTuneReport:
    initial_loss: float
    final_loss: float
    losses: tuple[float, ...]
    trainable_parameters: int


def evaluate_instructions(model: TinyDecoderLM, batch: InstructionBatch) -> float:
    """Deterministic mean supervised token loss, with no parameter updates."""
    was_training = model.training
    model.eval()
    try:
        with no_grad():
            return float(
                masked_next_token_loss(model(batch.inputs), batch.targets, batch.loss_mask).item()
            )
    finally:
        model.train(was_training)


def fine_tune_instructions(
    model: LoRATinyDecoderLM,
    batch: InstructionBatch,
    *,
    steps: int = 20,
    lr: float = 0.02,
    mixed_precision: bool = False,
    gradient_accumulation: int = 1,
) -> FineTuneReport:
    """Train *only* LoRA parameters on CPU with optional precision controls."""
    if not isinstance(model, LoRATinyDecoderLM) or steps < 1 or gradient_accumulation < 1:
        raise ValueError("require a LoRA model and positive training step counts")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be positive and finite")
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adam(params, lr=lr)
    scaler = GradScaler(enabled=mixed_precision)
    first = evaluate_instructions(model, batch)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        for _ in range(gradient_accumulation):
            with autocast(enabled=mixed_precision):
                loss = masked_next_token_loss(model(batch.inputs), batch.targets, batch.loss_mask)
            scaler.backward(loss / gradient_accumulation)
        scaler.step(optimizer)
        scaler.update()
        history.append(evaluate_instructions(model, batch))
    return FineTuneReport(first, history[-1], tuple(history), sum(p.size for p in params))
