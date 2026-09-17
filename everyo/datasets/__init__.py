"""Locally generated datasets.

Nothing in this package downloads data: every dataset is produced from a seed,
so examples, benchmarks and tests run without network access.
"""

from __future__ import annotations

from everyo.datasets.glyphs import (
    GLYPHS,
    IMAGE_SHAPE,
    NUM_CLASSES,
    load_digits,
    render_digit,
)
from everyo.datasets.sequences import make_copy_task, make_parity_task, make_recall_task
from everyo.datasets.synthetic import (
    make_blobs,
    make_moons,
    make_regression,
    make_spirals,
    make_xor,
)

__all__ = [
    "GLYPHS",
    "IMAGE_SHAPE",
    "NUM_CLASSES",
    "load_digits",
    "make_blobs",
    "make_copy_task",
    "make_moons",
    "make_parity_task",
    "make_recall_task",
    "make_regression",
    "make_spirals",
    "make_xor",
    "render_digit",
]
