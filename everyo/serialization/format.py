"""The ``.evo`` model archive format.

An ``.evo`` file is a ZIP archive with two members:

``manifest.json``
    Metadata plus the model architecture as plain JSON: class names and
    constructor arguments, never code.

``parameters.npz``
    An uncompressed NumPy archive holding one array per parameter, saved with
    ``allow_pickle=False``.

Loading rebuilds the model by looking class names up in
:data:`everyo.nn.module.MODULE_REGISTRY`.  Nothing from the file is ever
executed or unpickled, so opening an untrusted archive cannot run code.  The
worst a malformed archive can do is raise
:class:`~everyo.exceptions.EveryOSerializationError`.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "PARAMETERS_NAME",
    "DEFAULT_SUFFIX",
]

#: Bumped whenever the on-disk layout changes incompatibly.
FORMAT_VERSION: Final[int] = 1

MANIFEST_NAME: Final[str] = "manifest.json"
PARAMETERS_NAME: Final[str] = "parameters.npz"
DEFAULT_SUFFIX: Final[str] = ".evo"
