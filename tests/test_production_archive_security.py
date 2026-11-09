"""Adversarial CPU tests for model artifact ingestion."""

import io
import json
import zipfile

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOSerializationError
from everyo.serialization import format as model_format
from everyo.serialization import load_manifest


def test_duplicate_manifest_is_rejected(tmp_path):
    path = tmp_path / "duplicate.evo"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(model_format.MANIFEST_NAME, "{}")
        archive.writestr(model_format.MANIFEST_NAME, "{}")
    with pytest.raises(EveryOSerializationError, match="exactly one"):
        load_manifest(path)


def test_manifest_decompressed_limit(tmp_path, monkeypatch):
    import importlib

    loader = importlib.import_module("everyo.serialization.load")
    monkeypatch.setattr(loader, "MAX_MANIFEST_BYTES", 64)
    path = tmp_path / "oversized.evo"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(model_format.MANIFEST_NAME, json.dumps({"padding": "X" * 1000}))
    with pytest.raises(EveryOSerializationError, match="size limit"):
        load_manifest(path)


def test_nested_npz_decompression_limit(tmp_path, monkeypatch):
    import importlib

    loader = importlib.import_module("everyo.serialization.load")
    monkeypatch.setattr(loader, "MAX_PARAMETERS_BYTES", 1024)
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("weights.npy", b"0" * 4096)
    path = tmp_path / "nested.evo"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(model_format.MANIFEST_NAME, json.dumps({"format_version": 1}))
        archive.writestr(model_format.PARAMETERS_NAME, nested.getvalue())
    with pytest.raises(EveryOSerializationError, match="Expanded parameter"):
        eo.load(path)


def test_legitimate_model_still_loads(tmp_path):
    original = eo.Sequential(eo.Linear(2, 1, seed=1))
    path = eo.save(original, tmp_path / "valid.evo")
    restored = eo.load(path)
    values = np.array([[1.0, 2.0]])
    np.testing.assert_allclose(restored(values).data, original(values).data)
