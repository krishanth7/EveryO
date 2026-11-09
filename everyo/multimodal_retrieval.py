"""Phase 17: bounded, normalized CPU vector search for multimodal embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from everyo.validation import validate_array


def l2_normalize_embeddings(values: Any, *, epsilon: float = 1e-12) -> np.ndarray:
    vectors = validate_array(values)
    if vectors.ndim != 2 or epsilon <= 0:
        raise ValueError("embeddings must be 2D and epsilon positive")
    vectors = vectors.astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("zero-norm embeddings cannot be searched")
    return vectors / norms


class CPUEmbeddingIndex:
    """Exact cosine search, bounded and in memory; caller owns ID privacy."""

    def __init__(self, embeddings: Any, ids: list[str], *, max_items: int = 10_000):
        vectors = l2_normalize_embeddings(embeddings)
        if len(vectors) > max_items or len(ids) != len(vectors) or len(set(ids)) != len(ids):
            raise ValueError("index exceeds capacity or contains invalid IDs")
        if any(not isinstance(identifier, str) or not identifier for identifier in ids):
            raise ValueError("IDs must be nonempty strings")
        self._vectors = vectors.copy()
        self._ids = tuple(ids)

    def search(self, queries: Any, *, k: int = 5) -> list[list[tuple[str, float]]]:
        vectors = l2_normalize_embeddings(queries)
        if vectors.shape[1] != self._vectors.shape[1] or not 0 < k <= len(self._ids):
            raise ValueError("query dimension or k invalid")
        scores = vectors @ self._vectors.T
        # mergesort preserves index order for ties, yielding deterministic results.
        ranking = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return [
            [(self._ids[int(i)], float(scores[row, i])) for i in choices]
            for row, choices in enumerate(ranking)
        ]
