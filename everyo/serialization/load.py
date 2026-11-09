"""Reading models from ``.evo`` archives."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from everyo._logging import get_logger
from everyo.exceptions import EveryOSerializationError
from everyo.nn.module import Module, get_module_class
from everyo.serialization.format import FORMAT_VERSION, MANIFEST_NAME, PARAMETERS_NAME

__all__ = ["load", "load_manifest", "inspect_archive"]

_LOGGER = get_logger(__name__)

# Limits apply to decompressed bytes, not the easily forged compressed size.
# Large deployments may raise these constants deliberately after review.
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PARAMETERS_BYTES = 512 * 1024 * 1024
MAX_PARAMETER_ARRAYS = 10_000


def _bounded_member(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    members = [entry for entry in archive.infolist() if entry.filename == name]
    if len(members) != 1:
        raise EveryOSerializationError(f"Expected exactly one {name} archive entry.")
    if members[0].file_size > limit:
        raise EveryOSerializationError(f"{name} exceeds the decompressed size limit.")
    try:
        with archive.open(members[0]) as source:
            payload = source.read(limit + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise EveryOSerializationError(f"Could not read {name} archive entry.") from exc
    if len(payload) > limit:
        raise EveryOSerializationError(f"{name} exceeds the decompressed size limit.")
    return payload


def _validate_npz(payload: bytes) -> None:
    """Reject nested archive bombs before NumPy expands any arrays."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_PARAMETER_ARRAYS or len({e.filename for e in entries}) != len(
                entries
            ):
                raise EveryOSerializationError("Invalid or duplicate parameter array entries.")
            if any(not e.filename.endswith(".npy") for e in entries):
                raise EveryOSerializationError("Parameter archive contains unexpected entries.")
            if sum(e.file_size for e in entries) > MAX_PARAMETERS_BYTES:
                raise EveryOSerializationError("Expanded parameter arrays exceed the size limit.")
    except zipfile.BadZipFile as exc:
        raise EveryOSerializationError("Parameters are not a valid NumPy archive.") from exc


def _open_archive(path: Path) -> zipfile.ZipFile:
    if not path.is_file():
        raise EveryOSerializationError(f"Model file not found: {path}")
    try:
        return zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise EveryOSerializationError(
            f"{path} is not a valid EveryO model archive ({exc})."
        ) from exc


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read the manifest of an archive without rebuilding the model."""
    archive_path = Path(path)
    with _open_archive(archive_path) as archive:
        if MANIFEST_NAME not in archive.namelist():
            raise EveryOSerializationError(
                f"{archive_path} does not contain {MANIFEST_NAME}; it was not "
                "written by everyo.save()."
            )
        try:
            manifest = json.loads(
                _bounded_member(archive, MANIFEST_NAME, MAX_MANIFEST_BYTES).decode("utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EveryOSerializationError(
                f"The manifest in {archive_path} is corrupt ({exc})."
            ) from exc
    if not isinstance(manifest, dict):
        raise EveryOSerializationError(f"The manifest in {archive_path} must be a JSON object.")
    return manifest


def inspect_archive(path: str | Path) -> dict[str, Any]:
    """Summarise an archive: version, architecture and parameter shapes."""
    manifest = load_manifest(path)
    return {
        "path": str(Path(path)),
        "everyo_version": manifest.get("everyo_version"),
        "format_version": manifest.get("format_version"),
        "created_at": manifest.get("created_at"),
        "class_name": manifest.get("architecture", {}).get("class_name"),
        "num_parameters": manifest.get("num_parameters"),
        "parameters": manifest.get("parameters", {}),
        "metadata": manifest.get("metadata", {}),
    }


def load(path: str | Path, *, model: Module | None = None, strict: bool = True) -> Module:
    """Load a model from an ``.evo`` archive.

    The architecture is rebuilt from registered class names; the file is never
    unpickled and no code from it is executed.

    Args:
        path: Archive written by :func:`everyo.serialization.save.save`.
        model: Optional pre-built model to load the parameters into.  When
            given, the stored architecture is ignored.
        strict: Require the parameter names to match the model exactly.

    Returns:
        The reconstructed model, in evaluation mode.

    Raises:
        EveryOSerializationError: If the archive is missing, corrupt, written by
            an incompatible format version, or refers to an unregistered module.

    Example:
        >>> import everyo as eo, tempfile, pathlib
        >>> original = eo.Sequential(eo.Linear(2, 1, seed=0))
        >>> target = pathlib.Path(tempfile.mkdtemp()) / "m.evo"
        >>> _ = eo.save(original, target)
        >>> type(eo.load(target)).__name__
        'Sequential'
    """
    archive_path = Path(path)
    manifest = load_manifest(archive_path)

    stored_version = int(manifest.get("format_version", 0))
    if stored_version > FORMAT_VERSION:
        raise EveryOSerializationError(
            f"{archive_path} uses model format version {stored_version}, but "
            f"this EveryO release understands up to version {FORMAT_VERSION}. "
            "Upgrade EveryO to load this file."
        )

    with _open_archive(archive_path) as archive:
        if PARAMETERS_NAME not in archive.namelist():
            raise EveryOSerializationError(f"{archive_path} does not contain {PARAMETERS_NAME}.")
        payload = _bounded_member(archive, PARAMETERS_NAME, MAX_PARAMETERS_BYTES)

    _validate_npz(payload)

    try:
        # allow_pickle=False is the guarantee that loading cannot execute code.
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            state = {name: np.asarray(data[name]) for name in data.files}
    except ValueError as exc:
        raise EveryOSerializationError(
            f"Could not read the parameters in {archive_path} ({exc}). The file "
            "may be corrupt, or may contain pickled objects, which EveryO "
            "refuses to load."
        ) from exc

    if model is None:
        architecture = manifest.get("architecture")
        if not isinstance(architecture, dict) or "class_name" not in architecture:
            raise EveryOSerializationError(
                f"{archive_path} has no usable architecture description. Pass "
                "model=... to load the parameters into an existing model."
            )
        model_cls = get_module_class(str(architecture["class_name"]))
        config = architecture.get("config", {})
        if not isinstance(config, dict):
            raise EveryOSerializationError("The stored architecture config must be a JSON object.")
        model = model_cls.from_config(config)

    model.load_state_dict(state, strict=strict)
    model.eval()
    _LOGGER.info("Loaded model from %s", archive_path)
    return model
