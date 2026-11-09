"""Deterministic, CPU-only held-out evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(logits: Any, targets: Any) -> dict[str, float]:
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape (nonempty batch, classes>=2)")
    if labels.shape != (scores.shape[0],) or labels.dtype.kind not in "iu":
        raise ValueError("targets must be a vector of integer class indices")
    if not np.isfinite(scores).all() or np.any(labels < 0) or np.any(labels >= scores.shape[1]):
        raise ValueError("logits and target indices must be finite and valid")
    shifted = scores - scores.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    nll = logsumexp - shifted[np.arange(len(labels)), labels]
    predictions = scores.argmax(axis=1)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "cross_entropy": float(nll.mean()),
        "samples": float(len(labels)),
    }


def language_model_metrics(
    logits: Any, targets: Any, *, mask: Any | None = None
) -> dict[str, float]:
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets)
    if scores.ndim != 3 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("logits must have shape (batch, time, vocab)")
    if labels.shape != scores.shape[:2] or labels.dtype.kind not in "iu":
        raise ValueError("targets must contain integer token indices")
    selected = np.ones(labels.shape, dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != labels.shape or selected.dtype.kind != "b":
        raise ValueError("mask must be boolean with the same shape as targets")
    if not selected.any():
        raise ValueError("at least one token must be evaluated")
    result = classification_metrics(scores[selected], labels[selected])
    return {
        "token_accuracy": result["accuracy"],
        "cross_entropy": result["cross_entropy"],
        "perplexity": float(np.exp(min(result["cross_entropy"], 700))),
        "tokens": result["samples"],
    }


def deterministic_split(
    size: int, *, train: float = 0.7, validation: float = 0.15, seed: int = 0
) -> dict[str, np.ndarray]:
    """Return disjoint integer indices, never silently duplicating examples."""
    if (
        not isinstance(size, int)
        or size < 3
        or not 0 < train < 1
        or not 0 < validation < 1
        or train + validation >= 1
    ):
        raise ValueError("size >= 3 and positive train/validation/test proportions are required")
    order = np.random.default_rng(seed).permutation(size)
    a, b = int(size * train), int(size * (train + validation))
    if a == 0 or b in (a, size):
        raise ValueError("dataset is too small for nonempty train/validation/test splits")
    return {"train": order[:a], "validation": order[a:b], "test": order[b:]}


def assert_no_leakage(splits: dict[str, Any], *, fingerprints: Any | None = None) -> None:
    """Reject repeated IDs across splits and optionally duplicate content."""
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("train, validation, and test splits are required")
    seen: set[int] = set()
    content: set[str] = set()
    for indices in splits.values():
        ids = np.asarray(indices)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0:
            raise ValueError("split indices must be nonempty integer vectors")
        for raw in ids:
            idx = int(raw)
            if idx < 0 or idx in seen:
                raise ValueError("dataset leakage: duplicate or negative sample index")
            seen.add(idx)
            if fingerprints is not None:
                if idx >= len(fingerprints):
                    raise ValueError("fingerprint index outside dataset")
                fp = str(fingerprints[idx])
                if fp in content:
                    raise ValueError("dataset leakage: duplicate content fingerprint")
                content.add(fp)
