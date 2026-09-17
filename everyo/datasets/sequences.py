"""Synthetic sequence tasks, generated locally from a seed.

Each task isolates one ability, so a model that scores well has demonstrably
learned that thing rather than a shortcut:

* :func:`make_recall_task` — the label is the *first* token of a long
  sequence, so the model must carry information across the whole span.
* :func:`make_copy_task` — reproduce the input, which needs position as well
  as content.
* :func:`make_parity_task` — a global property no single position reveals.
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_recall_task", "make_copy_task", "make_parity_task"]


def make_recall_task(
    *,
    n_samples: int = 2000,
    length: int = 20,
    num_classes: int = 4,
    vocab_size: int = 16,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Long-range recall: the label is the first token, the rest is noise.

    The distractor tokens are drawn from a disjoint part of the vocabulary, so
    the only way to score above chance is to remember the opening token all the
    way to the end.

    Args:
        n_samples: Number of sequences.
        length: Tokens per sequence.
        num_classes: Number of possible opening tokens, and of labels.
        vocab_size: Total vocabulary; must exceed ``num_classes``.
        seed: Makes the dataset reproducible.

    Returns:
        ``(sequences, labels)`` of shapes ``(n, length)`` and ``(n,)``.

    Example:
        >>> from everyo.datasets import make_recall_task
        >>> x, y = make_recall_task(n_samples=8, length=5, seed=0)
        >>> x.shape, y.shape
        ((8, 5), (8,))
    """
    if vocab_size <= num_classes:
        raise ValueError(
            f"vocab_size ({vocab_size}) must exceed num_classes ({num_classes}) so "
            "that distractor tokens cannot be confused with the answer."
        )
    if length < 1:
        raise ValueError(f"length must be at least 1, got {length}.")

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n_samples)
    sequences = rng.integers(num_classes, vocab_size, size=(n_samples, length))
    sequences[:, 0] = labels
    return sequences.astype(np.int64), labels.astype(np.int64)


def make_copy_task(
    *, n_samples: int = 2000, length: int = 8, vocab_size: int = 10, seed: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the input sequence: targets are the inputs themselves.

    Trivial for a model that can read position, impossible for a bag of words.

    Returns:
        ``(sequences, targets)``, both ``(n, length)``.
    """
    rng = np.random.default_rng(seed)
    sequences = rng.integers(0, vocab_size, size=(n_samples, length)).astype(np.int64)
    return sequences, sequences.copy()


def make_parity_task(
    *, n_samples: int = 2000, length: int = 12, seed: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Predict whether a binary sequence contains an odd number of ones.

    Every position matters equally and no prefix determines the answer, so the
    model has to aggregate the whole sequence.

    Returns:
        ``(sequences, labels)`` of shapes ``(n, length)`` and ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    sequences = rng.integers(0, 2, size=(n_samples, length)).astype(np.int64)
    return sequences, (sequences.sum(axis=1) % 2).astype(np.int64)
