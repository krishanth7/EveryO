"""Matplotlib import helper.

Charts must work in headless environments (CI, servers, containers), so EveryO
falls back to the non-interactive Agg backend when the default backend cannot
actually draw. Two checks are needed:

1. On Linux, no ``DISPLAY``/``WAYLAND_DISPLAY`` means there is no display to
   draw on, which is known before importing ``pyplot``.
2. Everywhere else, the only reliable test is to try: an interactive backend
   can be importable yet still fail to build a figure (a Windows runner with a
   broken Tcl installation, for example). Every figure is therefore created
   through :func:`create_figure`, which retries on Agg when the active backend
   raises. That check is per call rather than once per process, because the
   backend can be changed again by anything else sharing the interpreter.

A backend the user configured themselves (via ``MPLBACKEND`` or an explicit
``matplotlib.use(...)``) is respected: matplotlib records that choice before
EveryO ever asks, and step 2 only intervenes when drawing genuinely fails.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

from everyo._logging import get_logger
from everyo.exceptions import EveryOBackendError

__all__ = ["create_figure", "get_pyplot", "is_available", "using_headless_backend"]

_LOGGER = get_logger(__name__)

_HEADLESS = False


def _no_display() -> bool:
    """Return ``True`` when the platform certainly has no display available."""
    if sys.platform in ("win32", "darwin"):
        # Both can have a usable display; failures are caught by the draw test.
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _switch_to_agg(matplotlib: Any, reason: str) -> None:
    """Switch matplotlib to the Agg backend and record that we did."""
    global _HEADLESS

    matplotlib.use("Agg", force=True)
    _HEADLESS = True
    _LOGGER.debug("Using the non-interactive Agg backend: %s", reason)


def get_pyplot() -> Any:
    """Import and return ``matplotlib.pyplot``, usable in headless environments.

    Raises:
        EveryOBackendError: If matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - matplotlib is a core dependency
        raise EveryOBackendError(
            "Plotting requires matplotlib. Install it with 'pip install matplotlib'."
        ) from exc

    if "matplotlib.pyplot" not in sys.modules and _no_display():
        _switch_to_agg(matplotlib, "no DISPLAY or WAYLAND_DISPLAY is set")

    import matplotlib.pyplot as plt

    return plt


def create_figure(factory: Callable[[Any], Any]) -> Any:
    """Build a figure with ``factory``, retrying on Agg if the backend cannot draw.

    Every EveryO chart is created through this helper rather than by calling
    ``plt.subplots`` directly. Checking the backend once at import time is not
    enough: an interactive backend can import cleanly, and anything that
    re-resolves the backend later (another library, a test fixture, a
    ``matplotlib.use`` call elsewhere in the process) can put a broken one back
    in place between two chart calls. Retrying per call means a chart is drawn
    whenever Agg can draw it, which is always.

    Args:
        factory: Callable taking ``pyplot`` and returning whatever
            ``plt.subplots`` returns.

    Returns:
        The factory's result.

    Raises:
        EveryOBackendError: If even the Agg backend cannot create a figure.
    """
    plt = get_pyplot()
    try:
        return factory(plt)
    except Exception as exc:  # noqa: BLE001 - any backend failure counts
        if using_headless_backend():
            raise EveryOBackendError(
                f"Matplotlib could not create a figure with the Agg backend "
                f"({exc}). The matplotlib installation appears to be broken."
            ) from exc

        import matplotlib

        _switch_to_agg(
            matplotlib,
            f"the '{matplotlib.get_backend()}' backend could not create a figure ({exc})",
        )
        try:
            return factory(get_pyplot())
        except Exception as agg_exc:  # pragma: no cover - broken install
            raise EveryOBackendError(
                "Matplotlib could not create a figure with either the default "
                f"backend or Agg ({agg_exc}). The matplotlib installation "
                "appears to be broken."
            ) from agg_exc


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
