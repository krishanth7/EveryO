"""Tests for release security, checksum manifests, package validation, and version consistency."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import tomllib
except (ModuleNotFoundError, ImportError):
    import tomli as tomllib  # type: ignore[no-redef]

from everyo.artifact_signing import (
    generate_checksum_manifest,
    validate_release_package,
    verify_checksum_manifest,
)
from everyo.governance import validate_release_version
from everyo.version import __version__


def test_version_consistency():
    validate_release_version(__version__)
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproj = tomllib.load(f)
    declared_ver = pyproj["project"]["version"]
    assert __version__ == declared_ver, f"Version mismatch: {__version__} vs {declared_ver}"


def test_version_consistency_with_tomli_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    try:
        import tomli
    except ImportError:
        pytest.skip("tomli not installed")
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproj = tomli.load(f)
    declared_ver = pyproj["project"]["version"]
    assert __version__ == declared_ver, f"Version mismatch: {__version__} vs {declared_ver}"


def test_checksum_manifest_lifecycle(tmp_path: Path):
    file1 = tmp_path / "package-1.0.whl"
    file2 = tmp_path / "package-1.0.tar.gz"

    file1.write_bytes(b"content-of-wheel-package")
    file2.write_bytes(b"content-of-sdist-package")

    manifest_file = tmp_path / "SHA256SUMS.txt"
    manifest = generate_checksum_manifest([file1, file2], output_path=manifest_file)

    assert len(manifest) == 2
    assert "package-1.0.whl" in manifest
    assert "package-1.0.tar.gz" in manifest
    assert manifest_file.exists()

    # Verification must succeed
    assert verify_checksum_manifest(manifest_file, base_dir=tmp_path) is True


def test_checksum_manifest_tamper_detection(tmp_path: Path):
    file1 = tmp_path / "artifact.whl"
    file1.write_bytes(b"genuine-bytes")

    manifest_file = tmp_path / "SHA256SUMS.txt"
    generate_checksum_manifest([file1], output_path=manifest_file)

    # Tamper with the artifact
    file1.write_bytes(b"tampered-bytes")
    assert verify_checksum_manifest(manifest_file, base_dir=tmp_path) is False


def test_checksum_manifest_missing_file_rejected(tmp_path: Path):
    file1 = tmp_path / "artifact.whl"
    file1.write_bytes(b"genuine-bytes")

    manifest_file = tmp_path / "SHA256SUMS.txt"
    generate_checksum_manifest([file1], output_path=manifest_file)

    file1.unlink()
    assert verify_checksum_manifest(manifest_file, base_dir=tmp_path) is False


def test_checksum_manifest_malformed_lines_rejected(tmp_path: Path):
    manifest_file = tmp_path / "MALFORMED.txt"
    manifest_file.write_text("invalid_line_without_space\n", encoding="utf-8")
    assert verify_checksum_manifest(manifest_file, base_dir=tmp_path) is False


def test_validate_release_package(tmp_path: Path):
    wheel_file = tmp_path / "everyo-0.5.0-py3-none-any.whl"
    wheel_file.write_bytes(b"dummy wheel content")

    info = validate_release_package(wheel_file)
    assert info["format"] == "wheel"
    assert info["filename"] == "everyo-0.5.0-py3-none-any.whl"
    assert len(info["sha256"]) == 64

    # Empty file must be rejected
    empty_file = tmp_path / "empty-0.5.0.tar.gz"
    empty_file.touch()
    with pytest.raises(ValueError, match="empty"):
        validate_release_package(empty_file)

    # Unknown extension must be rejected
    bad_ext = tmp_path / "everyo.exe"
    bad_ext.write_bytes(b"content")
    with pytest.raises(ValueError, match="Unrecognized package"):
        validate_release_package(bad_ext)
