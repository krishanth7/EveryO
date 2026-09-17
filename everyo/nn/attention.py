"""Attention and transformer blocks.

Attention asks, for every position in a sequence, "which other positions
matter to me?" and answers it with a weighted average. The whole mechanism is:

    softmax(Q Kᵀ / √d) V

Everything else — multiple heads, masks, residual connections, the
feed-forward block — is packaging around that one line. Because EveryO's
matmul, softmax, reshape and transpose are already differentiable and
finite-difference tested, the packaging is composition rather than new calculus.

Sequences are ``(batch, time, features)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.tensor import Tensor, as_tensor
from everyo.exceptions import EveryOShapeError
from everyo.nn.layers import Dropout, Linear
from everyo.nn.module import Module, register_module
from everyo.nn.normalization import LayerNorm

__all__ = [
    "scaled_dot_product_attention",
    "causal_mask",
    "padding_mask",
    "MultiHeadAttention",
    "PositionalEncoding",
    "TransformerEncoderBlock",
    "TransformerEncoder",
]

#: Added to masked-out scores. Large enough to vanish under softmax, finite
#: enough that ``0 * mask`` never produces a NaN the way ``-inf`` would.
MASK_VALUE = -1e9


def causal_mask(size: int) -> np.ndarray:
    """Return an additive mask that hides every future position.

    Shape ``(1, 1, size, size)``, broadcastable over batch and heads. Position
    ``i`` may attend to positions ``0..i`` only, which is what makes a decoder
    able to generate text one token at a time without reading ahead.

    Example:
        >>> from everyo.nn.attention import causal_mask
        >>> causal_mask(3)[0, 0, 0, 1] < 0
        True
    """
    if int(size) <= 0:
        raise EveryOShapeError(f"causal_mask needs a positive size, got {size}.")
    future = np.triu(np.ones((size, size), dtype=np.float32), k=1)
    return (future * MASK_VALUE).reshape(1, 1, size, size)


def padding_mask(lengths: Any, size: int) -> np.ndarray:
    """Return an additive mask that hides padding beyond each sequence length.

    Args:
        lengths: Per-sample true lengths, shape ``(batch,)``.
        size: Padded sequence length.

    Returns:
        A ``(batch, 1, 1, size)`` mask, broadcastable over heads and queries.
    """
    counts = np.asarray(lengths).astype(np.int64).reshape(-1)
    if np.any(counts < 0) or np.any(counts > size):
        raise EveryOShapeError(
            f"padding_mask lengths must lie in [0, {size}], got [{counts.min()}, {counts.max()}]."
        )
    positions = np.arange(size)[None, :]
    keep = positions < counts[:, None]
    return ((~keep).astype(np.float32) * MASK_VALUE).reshape(len(counts), 1, 1, size)


def scaled_dot_product_attention(
    query: Any, key: Any, value: Any, *, mask: Any = None, dropout: Dropout | None = None
) -> tuple[Tensor, Tensor]:
    """Compute ``softmax(Q Kᵀ / √d) V``.

    Args:
        query: ``(..., queries, depth)``.
        key: ``(..., keys, depth)``.
        value: ``(..., keys, value_depth)``.
        mask: Optional additive mask broadcastable to ``(..., queries, keys)``;
            masked positions should hold a large negative number.
        dropout: Optional :class:`~everyo.nn.layers.Dropout` applied to the
            attention weights.

    Returns:
        ``(output, weights)`` — the weighted values and the attention weights
        themselves, which are worth returning because they are the most
        interpretable thing a transformer produces.

    Raises:
        EveryOShapeError: If the depths or sequence lengths do not line up.
    """
    q = as_tensor(query)
    k = as_tensor(key)
    v = as_tensor(value)

    if q.shape[-1] != k.shape[-1]:
        raise EveryOShapeError(
            f"Attention needs matching query and key depths, got {q.shape[-1]} and {k.shape[-1]}."
        )
    if k.shape[-2] != v.shape[-2]:
        raise EveryOShapeError(
            f"Attention needs as many keys as values, got {k.shape[-2]} keys and "
            f"{v.shape[-2]} values."
        )

    depth = q.shape[-1]
    # The 1/sqrt(d) scale keeps the dot products from growing with depth; without
    # it softmax saturates and the gradient disappears.
    scores = ops.divide(ops.matmul(q, ops.transpose(k, _swap_last_two(k.ndim))), math.sqrt(depth))
    if mask is not None:
        scores = ops.add(scores, as_tensor(mask))

    weights = ops.softmax(scores, axis=-1)
    if dropout is not None:
        weights = dropout(weights)
    return ops.matmul(weights, v), weights


def _swap_last_two(ndim: int) -> tuple[int, ...]:
    """Axis permutation that transposes the final two axes."""
    if ndim < 2:
        raise EveryOShapeError("Attention inputs need at least two dimensions.")
    axes = list(range(ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return tuple(axes)


@register_module
class MultiHeadAttention(Module):
    """Multi-head self- or cross-attention.

    Splitting ``embed_dim`` into ``num_heads`` independent subspaces lets one
    head track syntax while another tracks, say, coreference — the same
    parameter budget, several relationships at once.

    Args:
        embed_dim: Model width; must divide evenly by ``num_heads``.
        num_heads: Number of attention heads.
        dropout: Dropout applied to the attention weights.
        bias: Learn biases in the four projections.
        seed: Makes initialisation reproducible.

    Example:
        >>> import everyo as eo
        >>> attention = eo.MultiHeadAttention(32, 4, seed=0)
        >>> attention(eo.zeros(2, 10, 32)).shape
        (2, 10, 32)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if int(embed_dim) <= 0 or int(num_heads) <= 0:
            raise EveryOShapeError(
                f"MultiHeadAttention requires positive sizes, got embed_dim="
                f"{embed_dim}, num_heads={num_heads}."
            )
        if int(embed_dim) % int(num_heads):
            raise EveryOShapeError(
                f"embed_dim={embed_dim} must divide evenly by num_heads="
                f"{num_heads}; each head takes embed_dim // num_heads features."
            )

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout_p = float(dropout)
        self.use_bias = bool(bias)
        self.seed = seed

        def offset(step: int) -> int | None:
            return None if seed is None else seed + step

        self.query_projection = Linear(
            embed_dim, embed_dim, bias=bias, initializer="xavier_uniform", seed=offset(0)
        )
        self.key_projection = Linear(
            embed_dim, embed_dim, bias=bias, initializer="xavier_uniform", seed=offset(1)
        )
        self.value_projection = Linear(
            embed_dim, embed_dim, bias=bias, initializer="xavier_uniform", seed=offset(2)
        )
        self.output_projection = Linear(
            embed_dim, embed_dim, bias=bias, initializer="xavier_uniform", seed=offset(3)
        )
        self.attention_dropout = Dropout(dropout, seed=seed) if dropout else None
        #: Attention weights from the most recent forward pass, for inspection.
        self.last_attention_weights: Tensor | None = None

    def _split_heads(self, value: Tensor, batch: int, steps: int) -> Tensor:
        """``(batch, time, embed)`` -> ``(batch, heads, time, head_dim)``."""
        reshaped = ops.reshape(value, (batch, steps, self.num_heads, self.head_dim))
        return ops.transpose(reshaped, (0, 2, 1, 3))

    def forward(
        self, query: Any, key: Any = None, value: Any = None, *, mask: Any = None
    ) -> Tensor:
        """Attend ``query`` over ``key``/``value``.

        With only ``query`` given this is self-attention. Passing a different
        ``key``/``value`` gives cross-attention.
        """
        q_in = as_tensor(query)
        k_in = q_in if key is None else as_tensor(key)
        v_in = k_in if value is None else as_tensor(value)

        for name, item in (("query", q_in), ("key", k_in), ("value", v_in)):
            if item.ndim != 3:
                raise EveryOShapeError(
                    f"MultiHeadAttention expects {name} shaped (batch, time, "
                    f"embed_dim), got {item.shape}."
                )
            if item.shape[-1] != self.embed_dim:
                raise EveryOShapeError(
                    f"MultiHeadAttention(embed_dim={self.embed_dim}) received a "
                    f"{name} whose last axis is {item.shape[-1]}."
                )

        batch, q_steps, _ = q_in.shape
        k_steps = k_in.shape[1]

        heads_q = self._split_heads(self.query_projection(q_in), batch, q_steps)
        heads_k = self._split_heads(self.key_projection(k_in), batch, k_steps)
        heads_v = self._split_heads(self.value_projection(v_in), batch, k_steps)

        context, weights = scaled_dot_product_attention(
            heads_q, heads_k, heads_v, mask=mask, dropout=self.attention_dropout
        )
        self.last_attention_weights = weights

        merged = ops.reshape(ops.transpose(context, (0, 2, 1, 3)), (batch, q_steps, self.embed_dim))
        return self.output_projection(merged)

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_p,
            "bias": self.use_bias,
        }


@register_module
class PositionalEncoding(Module):
    """Add fixed sinusoidal position information to a sequence.

    Attention is permutation-invariant: on its own it cannot tell "dog bites
    man" from "man bites dog". This layer injects order by adding a fixed
    pattern of sines and cosines, one frequency per dimension pair. It learns
    nothing — the table is a buffer, so it is saved with the model and costs no
    parameters.

    Args:
        embed_dim: Model width.
        max_length: Longest sequence the table covers.
        dropout: Dropout applied after adding the encoding.

    Example:
        >>> import everyo as eo
        >>> eo.PositionalEncoding(32)(eo.zeros(2, 10, 32)).shape
        (2, 10, 32)
    """

    def __init__(self, embed_dim: int, *, max_length: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        if int(embed_dim) <= 0 or int(max_length) <= 0:
            raise EveryOShapeError(
                f"PositionalEncoding requires positive sizes, got embed_dim="
                f"{embed_dim}, max_length={max_length}."
            )
        self.embed_dim = int(embed_dim)
        self.max_length = int(max_length)
        self.dropout_p = float(dropout)
        self.dropout = Dropout(dropout) if dropout else None

        position = np.arange(self.max_length, dtype=np.float32)[:, None]
        pair_index = np.arange(0, self.embed_dim, 2, dtype=np.float32)
        frequency = np.exp(-math.log(10000.0) * pair_index / self.embed_dim)

        table = np.zeros((self.max_length, self.embed_dim), dtype=np.float32)
        table[:, 0::2] = np.sin(position * frequency)
        table[:, 1::2] = np.cos(position * frequency)[:, : self.embed_dim // 2]
        self.register_buffer("encoding", table)

    def forward(self, x: Any) -> Tensor:
        """Add the positional encoding to ``x``."""
        value = as_tensor(x)
        if value.ndim != 3 or value.shape[-1] != self.embed_dim:
            raise EveryOShapeError(
                f"PositionalEncoding(embed_dim={self.embed_dim}) expects "
                f"(batch, time, {self.embed_dim}), got {value.shape}."
            )
        steps = value.shape[1]
        if steps > self.max_length:
            raise EveryOShapeError(
                f"Sequence of length {steps} exceeds max_length={self.max_length}. "
                "Build the layer with a larger max_length."
            )
        out = ops.add(value, Tensor(self.encoding[:steps][None, :, :]))
        return self.dropout(out) if self.dropout is not None else out

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "embed_dim": self.embed_dim,
            "max_length": self.max_length,
            "dropout": self.dropout_p,
        }


@register_module
class TransformerEncoderBlock(Module):
    """One transformer encoder layer: attention, then a feed-forward network.

    Each sublayer is wrapped in a residual connection, so the block starts life
    close to the identity and gradients have a short path back — the reason
    very deep stacks train at all.

    ``norm_first=True`` (the default) normalises *before* each sublayer. That is
    the arrangement modern large models use, because post-norm stacks need a
    learning-rate warm-up to stay stable.

    Args:
        embed_dim: Model width.
        num_heads: Attention heads.
        feedforward_dim: Hidden width of the feed-forward network; four times
            ``embed_dim`` is the usual choice.
        dropout: Applied to attention weights and both sublayer outputs.
        norm_first: Pre-norm (default) or post-norm.
        seed: Makes initialisation reproducible.

    Example:
        >>> import everyo as eo
        >>> block = eo.TransformerEncoderBlock(32, 4, seed=0)
        >>> block(eo.zeros(2, 10, 32)).shape
        (2, 10, 32)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        norm_first: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim or 4 * embed_dim)
        self.dropout_p = float(dropout)
        self.norm_first = bool(norm_first)
        self.seed = seed

        def offset(step: int) -> int | None:
            return None if seed is None else seed + step

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, seed=offset(0))
        self.attention_norm = LayerNorm(embed_dim)
        self.feedforward_norm = LayerNorm(embed_dim)
        self.linear_in = Linear(
            embed_dim, self.feedforward_dim, initializer="he_uniform", seed=offset(10)
        )
        self.linear_out = Linear(
            self.feedforward_dim, embed_dim, initializer="xavier_uniform", seed=offset(11)
        )
        self.attention_dropout = Dropout(dropout, seed=offset(20)) if dropout else None
        self.feedforward_dropout = Dropout(dropout, seed=offset(21)) if dropout else None

    def _maybe_drop(self, value: Tensor, dropout: Dropout | None) -> Tensor:
        return value if dropout is None else dropout(value)

    def _feedforward(self, value: Tensor) -> Tensor:
        return self.linear_out(ops.relu(self.linear_in(value)))

    def forward(self, x: Any, *, mask: Any = None) -> Tensor:
        """Run one encoder block over ``x``."""
        value = as_tensor(x)
        if value.ndim != 3 or value.shape[-1] != self.embed_dim:
            raise EveryOShapeError(
                f"TransformerEncoderBlock(embed_dim={self.embed_dim}) expects "
                f"(batch, time, {self.embed_dim}), got {value.shape}."
            )

        if self.norm_first:
            attended = self.attention(self.attention_norm(value), mask=mask)
            value = ops.add(value, self._maybe_drop(attended, self.attention_dropout))
            projected = self._feedforward(self.feedforward_norm(value))
            return ops.add(value, self._maybe_drop(projected, self.feedforward_dropout))

        attended = self.attention(value, mask=mask)
        value = self.attention_norm(
            ops.add(value, self._maybe_drop(attended, self.attention_dropout))
        )
        projected = self._feedforward(value)
        return self.feedforward_norm(
            ops.add(value, self._maybe_drop(projected, self.feedforward_dropout))
        )

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout_p,
            "norm_first": self.norm_first,
        }


@register_module
class TransformerEncoder(Module):
    """A stack of :class:`TransformerEncoderBlock` layers.

    With ``norm_first`` the stack ends with a final normalization, which is
    what keeps the output scale bounded no matter how many layers are stacked.

    Args:
        embed_dim: Model width.
        num_heads: Attention heads per block.
        num_layers: How many blocks to stack.
        feedforward_dim: Hidden width inside each block.
        dropout: Dropout used throughout.
        norm_first: Pre-norm (default) or post-norm.
        seed: Makes initialisation reproducible.

    Example:
        >>> import everyo as eo
        >>> encoder = eo.TransformerEncoder(32, 4, num_layers=2, seed=0)
        >>> encoder(eo.zeros(2, 10, 32)).shape
        (2, 10, 32)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        num_layers: int = 2,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        norm_first: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise EveryOShapeError(f"num_layers must be positive, got {num_layers}.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.feedforward_dim = int(feedforward_dim or 4 * embed_dim)
        self.dropout_p = float(dropout)
        self.norm_first = bool(norm_first)
        self.seed = seed

        for index in range(self.num_layers):
            setattr(
                self,
                f"block_{index}",
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    feedforward_dim=self.feedforward_dim,
                    dropout=dropout,
                    norm_first=norm_first,
                    seed=None if seed is None else seed + 100 * index,
                ),
            )
        self.final_norm = LayerNorm(embed_dim) if norm_first else None

    @property
    def blocks(self) -> list[TransformerEncoderBlock]:
        """The encoder blocks, in order."""
        return [getattr(self, f"block_{index}") for index in range(self.num_layers)]

    def forward(self, x: Any, *, mask: Any = None) -> Tensor:
        """Run the full stack over ``x``."""
        value = as_tensor(x)
        for block in self.blocks:
            value = block(value, mask=mask)
        return self.final_norm(value) if self.final_norm is not None else value

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout_p,
            "norm_first": self.norm_first,
        }
