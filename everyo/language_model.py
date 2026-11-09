"""Small, educational, CPU-first autoregressive language models for EveryO.

The model uses EveryO's own autograd for training, and NumPy for no-grad
incremental attention with a real per-layer projected key/value cache.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.autograd import no_grad
from everyo.core.tensor import Tensor
from everyo.nn.activations import ReLU
from everyo.nn.attention import MultiHeadAttention, causal_mask
from everyo.nn.embedding import Embedding
from everyo.nn.layers import Linear
from everyo.nn.losses import CrossEntropyLoss
from everyo.nn.module import Module, register_module
from everyo.nn.normalization import LayerNorm

__all__ = [
    "CharTokenizer",
    "ByteBPETokenizer",
    "DecoderBlock",
    "TinyDecoderLM",
    "next_token_loss",
    "sample_next_token",
    "KVCache",
]


class CharTokenizer:
    """Deterministic character-level tokenizer with PAD=0 and UNK=1."""

    def __init__(self, corpus: str) -> None:
        chars = sorted(set(corpus))
        self.tokens = ["<pad>", "<unk>", *chars]
        self.lookup = {char: index + 2 for index, char in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.lookup.get(char, 1) for char in text]

    def decode(self, ids: Any) -> str:
        result = []
        for token_id in ids:
            index = int(token_id)
            if not 0 <= index < len(self.tokens):
                raise ValueError("token ID out of range")
            if index == 0:
                continue
            result.append("�" if index == 1 else self.tokens[index])
        return "".join(result)


class ByteBPETokenizer:
    """Deterministic UTF-8 byte-pair tokenizer with a reversible byte vocabulary.

    This educational BPE learns merges directly from the supplied corpus. It
    is not compatible with GPT-2/tiktoken vocabularies or regex pretokenizers.
    """

    def __init__(self, corpus: str, *, merges: int = 64) -> None:
        if merges < 0:
            raise ValueError("merges must be nonnegative")
        self.pieces: list[bytes] = [bytes([index]) for index in range(256)]
        self.rules: list[tuple[int, int, int]] = []
        sequence = list(corpus.encode("utf-8"))
        for _ in range(merges):
            counts = Counter(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            if count < 2:
                break
            merged = len(self.pieces)
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])
            self.rules.append((pair[0], pair[1], merged))
            sequence = self._merge(sequence, pair, merged)

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], replacement: int) -> list[int]:
        merged = []
        index = 0
        while index < len(ids):
            if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
                merged.append(replacement)
                index += 2
            else:
                merged.append(ids[index])
                index += 1
        return merged

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    def encode(self, text: str) -> list[int]:
        sequence = list(text.encode("utf-8"))
        for left, right, merged in self.rules:
            sequence = self._merge(sequence, (left, right), merged)
        return sequence

    def decode(self, ids: Any) -> str:
        chunks = []
        for item in ids:
            index = int(item)
            if not 0 <= index < len(self.pieces):
                raise ValueError("token ID out of range")
            chunks.append(self.pieces[index])
        return b"".join(chunks).decode("utf-8")


@dataclass(frozen=True)
class KVCache:
    """Projected attention keys and values, shape (batch, heads, steps, depth)."""

    keys: np.ndarray
    values: np.ndarray

    @property
    def length(self) -> int:
        return int(self.keys.shape[2])


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values, axis=-1, keepdims=True)
    weights = np.exp(centered)
    return weights / np.sum(weights, axis=-1, keepdims=True)


class DecoderBlock(Module):
    """Pre-norm causal attention plus position-wise feedforward network."""

    def __init__(self, embed_dim: int, heads: int, *, expansion: int = 2, seed: int = 0) -> None:
        super().__init__()
        if embed_dim < 1 or heads < 1 or embed_dim % heads or expansion < 1:
            raise ValueError("invalid decoder dimensions")
        self.norm1 = LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, heads, dropout=0.0, seed=seed)
        self.norm2 = LayerNorm(embed_dim)
        self.ff1 = Linear(embed_dim, embed_dim * expansion, seed=seed + 4)
        self.activation = ReLU()
        self.ff2 = Linear(embed_dim * expansion, embed_dim, seed=seed + 5)

    def forward(self, hidden: Tensor) -> Tensor:
        steps = hidden.shape[1]
        attended = self.attention(self.norm1(hidden), mask=causal_mask(steps))
        residual = hidden + attended
        return residual + self.ff2(self.activation(self.ff1(self.norm2(residual))))

    def cached_step(self, hidden: Tensor, previous: KVCache | None) -> tuple[Tensor, KVCache]:
        """Run one new position, projecting only the new K/V (no cache recompute)."""
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise ValueError("cached_step expects (batch, 1, width)")
        attended_input = self.norm1(hidden)
        attention = self.attention
        batch = hidden.shape[0]
        q = attention._split_heads(attention.query_projection(attended_input), batch, 1).data
        k = attention._split_heads(attention.key_projection(attended_input), batch, 1).data
        v = attention._split_heads(attention.value_projection(attended_input), batch, 1).data
        if previous is not None:
            if previous.keys.shape[:2] != k.shape[:2] or previous.keys.shape[3] != k.shape[3]:
                raise ValueError("KV cache dimensions are incompatible with this model/batch")
            k = np.concatenate((previous.keys, k), axis=2)
            v = np.concatenate((previous.values, v), axis=2)
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(attention.head_dim)
        mixed = np.matmul(_softmax(scores), v)
        merged = np.transpose(mixed, (0, 2, 1, 3)).reshape(batch, 1, attention.embed_dim)
        residual = hidden + attention.output_projection(Tensor(merged, device=hidden.device))
        result = residual + self.ff2(self.activation(self.ff1(self.norm2(residual))))
        return result, KVCache(k.copy(), v.copy())


@register_module
class TinyDecoderLM(Module):
    """Decoder-only transformer, trained with teacher-forced next-token loss.

    Lengths must fit ``max_context``; generated tokens beyond it are rejected
    instead of silently changing positional embeddings or truncating caches.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        max_context: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("invalid model vocabulary/attention dimensions")
        if num_layers < 1 or max_context < 2:
            raise ValueError("num_layers must be positive and max_context >= 2")
        self.vocab_size, self.embed_dim, self.num_heads = vocab_size, embed_dim, num_heads
        self.num_layers, self.max_context, self.seed = num_layers, max_context, seed
        self.tokens = Embedding(vocab_size, embed_dim, seed=seed)
        self.positions = Embedding(max_context, embed_dim, seed=seed + 1)
        for index in range(num_layers):
            setattr(
                self,
                f"block_{index}",
                DecoderBlock(embed_dim, num_heads, seed=seed + 10 * (index + 1)),
            )
        self.final_norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size, bias=False, seed=seed + 99)

    def _validate(self, ids: Any) -> np.ndarray:
        tokens = np.asarray(ids)
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("token IDs must be shaped (batch, time), with nonempty axes")
        if not np.issubdtype(tokens.dtype, np.integer):
            raise TypeError("token IDs must be integers")
        if np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
            raise ValueError("token IDs outside vocabulary")
        return tokens.astype(np.int64, copy=False)

    def forward(self, ids: Any) -> Tensor:
        tokens = self._validate(ids)
        steps = tokens.shape[1]
        if steps > self.max_context:
            raise ValueError("sequence exceeds max_context")
        positions = np.broadcast_to(np.arange(steps), tokens.shape)
        hidden = self.tokens(tokens) + self.positions(positions)
        for index in range(self.num_layers):
            hidden = getattr(self, f"block_{index}")(hidden)
        return self.head(self.final_norm(hidden))

    def cached_step(
        self, ids: Any, cache: list[KVCache] | None = None
    ) -> tuple[np.ndarray, list[KVCache]]:
        """One-token forward pass returning last logits and new K/V caches."""
        tokens = self._validate(ids)
        if tokens.shape[1] != 1:
            raise ValueError("cached_step processes exactly one token")
        if cache is not None and len(cache) != self.num_layers:
            raise ValueError("cache count must match number of layers")
        offset = 0 if cache is None else cache[0].length
        if cache is not None and any(item.length != offset for item in cache):
            raise ValueError("cache layer lengths disagree")
        if offset >= self.max_context:
            raise ValueError("context length exhausted")
        with no_grad():
            hidden = self.tokens(tokens) + self.positions(np.full_like(tokens, offset))
            next_cache = []
            for index in range(self.num_layers):
                previous = None if cache is None else cache[index]
                hidden, current = getattr(self, f"block_{index}").cached_step(hidden, previous)
                next_cache.append(current)
            logits = self.head(self.final_norm(hidden)).data[:, -1, :].copy()
        return logits, next_cache

    def generate(
        self,
        prompt: Any,
        *,
        max_new_tokens: int = 8,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        seed: int = 0,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Sample autoregressively with deterministic local RNG and optional KV cache."""
        tokens = self._validate(prompt)
        if max_new_tokens < 0 or tokens.shape[1] + max_new_tokens > self.max_context:
            raise ValueError("requested generation exceeds max_context")
        rng = np.random.default_rng(seed)
        output = tokens.copy()
        if max_new_tokens == 0:
            return output
        was_training = self.training
        self.eval()
        try:
            if use_cache:
                cache = None
                for index in range(output.shape[1]):
                    logits, cache = self.cached_step(output[:, index : index + 1], cache)
            for iteration in range(max_new_tokens):
                if not use_cache:
                    with no_grad():
                        logits = self(output).data[:, -1, :].copy()
                choices = np.array(
                    [
                        sample_next_token(
                            row, rng=rng, temperature=temperature, top_k=top_k, top_p=top_p
                        )
                        for row in logits
                    ],
                    dtype=np.int64,
                )[:, None]
                output = np.concatenate((output, choices), axis=1)
                if use_cache and iteration + 1 < max_new_tokens:
                    logits, cache = self.cached_step(choices, cache)
            return output
        finally:
            self.train(was_training)

    def get_config(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_context": self.max_context,
            "seed": self.seed,
        }


def next_token_loss(logits: Tensor, targets: Any) -> Tensor:
    """Mean teacher-forced next-token cross entropy for (B,T,V) logits."""
    labels = np.asarray(targets)
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be (batch,time,vocab) and labels (batch,time)")
    return CrossEntropyLoss()(ops.reshape(logits, (-1, logits.shape[-1])), labels.reshape(-1))


def sample_next_token(
    logits: Any,
    *,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> int:
    """Apply temperature, top-k and nucleus filtering before sampling."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 2 or not np.all(np.isfinite(scores)):
        raise ValueError("logits must be a finite 1-D vocabulary")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k is not None and (not isinstance(top_k, int) or not 1 <= top_k <= scores.size):
        raise ValueError("top_k must be a valid positive vocabulary count")
    scores = scores / temperature
    # Stable descending order gives deterministic tie-handling.
    order = np.argsort(-scores, kind="stable")
    if top_k is not None:
        order = order[:top_k]
    probs = _softmax(scores[order])
    cumulative = np.cumsum(probs)
    cutoff = int(np.searchsorted(cumulative, top_p, side="left")) + 1
    order, probs = order[:cutoff], probs[:cutoff]
    probs = probs / probs.sum()
    return int(rng.choice(order, p=probs))
