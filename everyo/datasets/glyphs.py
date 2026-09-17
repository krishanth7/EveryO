"""A small handwritten-digit-style dataset generated entirely offline.

The dataset is built from ten 8x8 bitmap glyphs, one per digit, which are then
augmented with random shifts, stroke-intensity changes and additive noise.  It
is deliberately *not* MNIST: EveryO's test suite and demo must run without
network access, and a locally generated dataset is reproducible from a seed.

The task is easy enough to learn in seconds on a CPU and hard enough that an
untrained model scores near chance, which makes it a useful end-to-end demo.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = ["GLYPHS", "IMAGE_SHAPE", "NUM_CLASSES", "render_digit", "load_digits"]

IMAGE_SHAPE: Final[tuple[int, int]] = (8, 8)
NUM_CLASSES: Final[int] = 10

#: One 8x8 bitmap per digit; '#' marks an inked pixel.
GLYPHS: Final[tuple[tuple[str, ...], ...]] = (
    (  # 0
        "..####..",
        ".##..##.",
        "##....##",
        "##....##",
        "##....##",
        ".##..##.",
        "..####..",
        "........",
    ),
    (  # 1
        "...##...",
        "..###...",
        ".####...",
        "...##...",
        "...##...",
        "...##...",
        ".######.",
        "........",
    ),
    (  # 2
        ".######.",
        "##....##",
        "......##",
        "....###.",
        "..###...",
        ".##.....",
        "########",
        "........",
    ),
    (  # 3
        ".######.",
        "##....##",
        "......##",
        "...####.",
        "......##",
        "##....##",
        ".######.",
        "........",
    ),
    (  # 4
        ".....##.",
        "....###.",
        "...####.",
        "..##.##.",
        "########",
        ".....##.",
        ".....##.",
        "........",
    ),
    (  # 5
        "########",
        "##......",
        "##......",
        "#######.",
        "......##",
        "##....##",
        ".######.",
        "........",
    ),
    (  # 6
        "..####..",
        ".##..##.",
        "##......",
        "#######.",
        "##....##",
        "##....##",
        ".######.",
        "........",
    ),
    (  # 7
        "########",
        "......##",
        ".....##.",
        "....##..",
        "...##...",
        "...##...",
        "...##...",
        "........",
    ),
    (  # 8
        ".######.",
        "##....##",
        "##....##",
        ".######.",
        "##....##",
        "##....##",
        ".######.",
        "........",
    ),
    (  # 9
        ".######.",
        "##....##",
        "##....##",
        ".#######",
        "......##",
        ".##..##.",
        "..####..",
        "........",
    ),
)


def render_digit(digit: int) -> np.ndarray:
    """Return the clean 8x8 bitmap of ``digit`` as a float array in ``[0, 1]``.

    Raises:
        ValueError: If ``digit`` is outside ``0..9``.
    """
    if not 0 <= int(digit) < NUM_CLASSES:
        raise ValueError(f"digit must be in 0..9, got {digit}.")
    rows = GLYPHS[int(digit)]
    return np.array(
        [[1.0 if cell == "#" else 0.0 for cell in row] for row in rows],
        dtype=np.float32,
    )


def _shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate an image by ``(dy, dx)`` pixels, padding with zeros."""
    shifted = np.zeros_like(image)
    height, width = image.shape
    y_src = slice(max(0, -dy), height - max(0, dy))
    x_src = slice(max(0, -dx), width - max(0, dx))
    y_dst = slice(max(0, dy), height - max(0, -dy))
    x_dst = slice(max(0, dx), width - max(0, -dx))
    shifted[y_dst, x_dst] = image[y_src, x_src]
    return shifted


def load_digits(
    *,
    samples_per_class: int = 200,
    noise: float = 0.15,
    max_shift: int = 1,
    flatten: bool = True,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an augmented digit dataset.

    Args:
        samples_per_class: Number of samples generated for each of the 10 digits.
        noise: Standard deviation of the additive Gaussian noise.
        max_shift: Maximum translation in pixels, applied on both axes.
        flatten: Return ``(n, 64)`` vectors instead of ``(n, 8, 8)`` images.
        seed: Seed making the dataset reproducible.

    Returns:
        ``(features, labels)`` with float32 features in ``[0, 1]`` and int64 labels.

    Example:
        >>> from everyo.datasets import load_digits
        >>> x, y = load_digits(samples_per_class=5, seed=0)
        >>> x.shape, y.shape
        ((50, 64), (50,))
    """
    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}.")
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}.")

    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(NUM_CLASSES):
        template = render_digit(digit)
        for _ in range(samples_per_class):
            dy = int(rng.integers(-max_shift, max_shift + 1)) if max_shift else 0
            dx = int(rng.integers(-max_shift, max_shift + 1)) if max_shift else 0
            image = _shift(template, dy, dx)
            image = image * rng.uniform(0.7, 1.0)
            image = image + rng.normal(0.0, noise, size=image.shape)
            images.append(np.clip(image, 0.0, 1.0).astype(np.float32))
            labels.append(digit)

    features = np.stack(images)
    targets = np.asarray(labels, dtype=np.int64)

    order = rng.permutation(len(targets))
    features, targets = features[order], targets[order]
    if flatten:
        features = features.reshape(len(features), -1)
    return features, targets
