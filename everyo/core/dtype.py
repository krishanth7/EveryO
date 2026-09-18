"""Data type handling for EveryO tensors.

EveryO deliberately supports a small, well understood set of NumPy dtypes.  The
default floating point type is ``float32`` because it is the common currency of
neural network training: it halves memory traffic relative to ``float64`` while
remaining accurate enough for the operations implemented here.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from everyo.exceptions import EveryODTypeError

__all__ = [
    "DTYPES",
    "float16",
    "DEFAULT_FLOAT_DTYPE",
    "DEFAULT_INT_DTYPE",
    "float32",
    "float64",
    "int32",
    "int64",
    "bool_",
    "resolve_dtype",
    "dtype_name",
    "is_floating",
]

float16: Final = np.float16
float32: Final = np.float32
float64: Final = np.float64
int32: Final = np.int32
int64: Final = np.int64
bool_: Final = np.bool_

#: Mapping of the dtype names EveryO accepts to their NumPy equivalents.
DTYPES: Final[dict[str, Any]] = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}

DEFAULT_FLOAT_DTYPE: Final = np.float32
DEFAULT_INT_DTYPE: Final = np.int64


def resolve_dtype(dtype: Any) -> np.dtype:
    """Normalise ``dtype`` into a supported :class:`numpy.dtype`.

    Args:
        dtype: A dtype name (``"float32"``), a NumPy dtype, or a NumPy scalar type.

    Returns:
        The corresponding :class:`numpy.dtype`.

    Raises:
        EveryODTypeError: If the dtype is unknown or unsupported by EveryO.
    """
    if isinstance(dtype, str):
        key = dtype.lower()
        if key not in DTYPES:
            raise EveryODTypeError(
                f"Unknown dtype {dtype!r}. EveryO supports: {', '.join(sorted(DTYPES))}."
            )
        return np.dtype(DTYPES[key])

    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:  # pragma: no cover - defensive
        raise EveryODTypeError(f"Cannot interpret {dtype!r} as a dtype.") from exc

    if resolved.name not in DTYPES and not (resolved == np.bool_):
        raise EveryODTypeError(
            f"Unsupported dtype {resolved.name!r}. EveryO supports: {', '.join(sorted(DTYPES))}."
        )
    return resolved


def dtype_name(dtype: Any) -> str:
    """Return the canonical EveryO name for ``dtype`` (e.g. ``"float32"``)."""
    resolved = resolve_dtype(dtype)
    return "bool" if resolved == np.bool_ else str(resolved.name)


def is_floating(dtype: Any) -> bool:
    """Return ``True`` when ``dtype`` is a floating point type."""
    return np.issubdtype(resolve_dtype(dtype), np.floating)
