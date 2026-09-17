"""Matplotlib import helper.

Charts must work in headless environments (CI, servers, containers), so the
non-interactive Agg backend is selected when no display is detected.  The
choice is made before ``pyplot`` is imported and never overrides a backend the
user configured themselves.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from everyo.exceptions import EveryOBackendError

__all__ = ["get_pyplot", "is_available", "using_headless_backend"]

_HEADLESS = False


def _headless() -> bool:
    if sys.platform in ("win32", "darwin"):
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_pyplot() -> Any:
    """Import and return ``matplotlib.pyplot``.

    Raises:
        EveryOBackendError: If matplotlib is not installed.
    """
    global _HEADLESS
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - matplotlib is a core dependency
        raise EveryOBackendError(
            "Plotting requires matplotlib. Install it with 'pip install matplotlib'."
        ) from exc

    if "matplotlib.pyplot" not in sys.modules and _headless():
        matplotlib.use("Agg")
        _HEADLESS = True

    import matplotlib.pyplot as plt

    return plt


def is_available() -> bool:
    """Return ``True`` when matplotlib can be imported."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def using_headless_backend() -> bool:
    """Return ``True`` when EveryO selected the non-interactive Agg backend."""
    return _HEADLESS
