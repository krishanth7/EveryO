"""Single source of truth for the EveryO version number."""

from __future__ import annotations

__all__ = ["__version__", "VERSION_INFO"]

VERSION_INFO = (0, 1, 0)
__version__ = ".".join(str(part) for part in VERSION_INFO)
