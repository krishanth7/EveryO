"""Writing models to ``.evo`` archives."""

from __future__ import annotations

import datetime as _datetime
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from everyo._logging import get_logger
from everyo.exceptions import EveryOSerializationError
from everyo.nn.module import Module
from everyo.serialization.format import (
    DEFAULT_SUFFIX,
    FORMAT_VERSION,
    MANIFEST_NAME,
    PARAMETERS_NAME,
)
from everyo.version import __version__

__all__ = ["save", "build_manifest"]

_LOGGER = get_logger(__name__)


def build_manifest(model: Module, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe ``model`` as a JSON-serialisable dictionary.

    Args:
        model: The model to describe.
        metadata: Extra keys stored alongside the architecture.

    Raises:
        EveryOSerializationError: If the architecture or metadata is not
            JSON-serialisable.
    """
    state = model.state_dict()
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "everyo_version": __version__,
        "created_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "architecture": {
            "class_name": type(model).__name__,
            "config": model.get_config(),
        },
        "parameters": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in state.items()
        },
        "num_parameters": int(model.num_parameters()),
        "metadata": dict(metadata or {}),
    }
    try:
        json.dumps(manifest)
    except TypeError as exc:
        raise EveryOSerializationError(
            f"The model description is not JSON-serialisable ({exc}). Make sure "
            "get_config() and any metadata only contain numbers, strings, "
            "booleans, lists and dictionaries."
        ) from exc
    return manifest


def save(
    model: Module,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> Path:
    """Save ``model`` to ``path`` as an ``.evo`` archive.

    Args:
        model: The model to save.
        path: Destination path; ``.evo`` is appended when no suffix is given.
        metadata: Extra information stored in the manifest (epoch, metrics, ...).
        overwrite: Allow replacing an existing file.

    Returns:
        The path actually written.

    Raises:
        EveryOSerializationError: If the target exists and ``overwrite`` is
            ``False``, or if the archive cannot be written.

    Example:
        >>> import everyo as eo, tempfile, pathlib
        >>> model = eo.Sequential(eo.Linear(2, 1, seed=0))
        >>> target = pathlib.Path(tempfile.mkdtemp()) / "model.evo"
        >>> eo.save(model, target).name
        'model.evo'
    """
    if not isinstance(model, Module):
        raise EveryOSerializationError(
            f"save() expects an everyo Module, got {type(model).__name__}."
        )

    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(DEFAULT_SUFFIX)
    if target.exists() and not overwrite:
        raise EveryOSerializationError(
            f"{target} already exists. Pass overwrite=True to replace it."
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(model, metadata)
    state = model.state_dict()

    buffer = io.BytesIO()
    np.savez(buffer, **{name: np.asarray(array) for name, array in state.items()})

    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
            archive.writestr(PARAMETERS_NAME, buffer.getvalue())
        temporary.replace(target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise EveryOSerializationError(f"Could not write {target}: {exc}") from exc

    _LOGGER.info("Saved model with %d parameters to %s", manifest["num_parameters"], target)
    return target
