"""Embedding lookup."""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.tensor import Tensor
from everyo.exceptions import EveryOShapeError
from everyo.nn.initialization import get_initializer
from everyo.nn.module import Module, Parameter, register_module

__all__ = ["Embedding"]


@register_module
class Embedding(Module):
    """A learned lookup table mapping integer ids to dense vectors.

    The forward pass is an indexing operation, and its gradient is a
    scatter-add: a token appearing three times in a batch accumulates three
    contributions into its row. EveryO's differentiable indexing already does
    exactly that, so the layer is a lookup and nothing more.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim: Width of each vector.
        initializer: Scheme from :mod:`everyo.nn.initialization`.
        seed: Makes the initial table reproducible.

    Example:
        >>> import everyo as eo
        >>> table = eo.Embedding(100, 16, seed=0)
        >>> table([[1, 5, 5]]).shape
        (1, 3, 16)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        initializer: str = "normal",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if int(num_embeddings) <= 0 or int(embedding_dim) <= 0:
            raise EveryOShapeError(
                f"Embedding requires positive sizes, got num_embeddings="
                f"{num_embeddings}, embedding_dim={embedding_dim}."
            )
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.initializer = str(initializer)
        self.seed = seed

        values = get_initializer(initializer)((self.num_embeddings, self.embedding_dim), seed=seed)
        self.weight = Parameter(np.asarray(values, dtype=np.float32), name="weight")

    def forward(self, ids: Any) -> Tensor:
        """Look up ``ids`` in the table.

        Args:
            ids: Integer ids of any shape; the result gains a trailing
                ``embedding_dim`` axis.

        Raises:
            EveryOShapeError: If an id falls outside the vocabulary.
        """
        indices = np.asarray(ids.data if isinstance(ids, Tensor) else ids)
        if indices.size and not np.all(np.equal(np.mod(indices, 1), 0)):
            raise EveryOShapeError(
                "Embedding ids must be whole numbers; got floating point values."
            )
        indices = indices.astype(np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= self.num_embeddings):
            raise EveryOShapeError(
                f"Embedding id out of range: ids must be in "
                f"[0, {self.num_embeddings - 1}], but the batch contains "
                f"[{indices.min()}, {indices.max()}]."
            )
        return ops.slice_(self.weight, indices)

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this layer."""
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "initializer": self.initializer,
        }
