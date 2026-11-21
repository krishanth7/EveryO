"""Detached Ed25519 artifact attestations with externally managed signing keys.

This module intentionally never generates, stores, or transmits a private key.
The caller supplies trusted key bytes from an external key-management system.
Also provides SHA-256 checksum manifest generation and verification for release distributions.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Sequence

from everyo.governance import sha256_file


def _crypto() -> Any:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise RuntimeError("Install everyo[signing] to verify release attestations") from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature


def _payload(digest: str, version: str, key_id: str) -> bytes:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("expected lowercase SHA-256 digest")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", key_id):
        raise ValueError("invalid public-key identifier")
    from everyo.governance import validate_release_version

    validate_release_version(version)
    return json.dumps(
        {"artifact_sha256": digest, "key_id": key_id, "version": version},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_artifact(
    artifact: str | Path, *, version: str, key_id: str, private_key_bytes: bytes
) -> dict[str, str]:
    """Sign digest and release context, never the private key itself."""
    private_cls, _, _ = _crypto()
    digest = sha256_file(artifact)
    content = _payload(digest, version, key_id)
    signature = private_cls.from_private_bytes(private_key_bytes).sign(content)
    return {
        "algorithm": "Ed25519",
        "artifact_sha256": digest,
        "version": version,
        "key_id": key_id,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
    }


def verify_signed_artifact(
    artifact: str | Path, attestation: dict[str, str], *, public_key_bytes: bytes
) -> bool:
    """Fail closed for tampering, wrong trusted key, and mismatched release context."""
    _, public_cls, invalid_signature = _crypto()
    if (
        not isinstance(attestation, dict)
        or set(attestation)
        != {"algorithm", "artifact_sha256", "version", "key_id", "signature_base64"}
        or attestation["algorithm"] != "Ed25519"
    ):
        raise ValueError("invalid signed-artifact schema")
    digest = sha256_file(artifact)
    if digest != attestation["artifact_sha256"]:
        return False
    content = _payload(digest, attestation["version"], attestation["key_id"])
    try:
        signature = base64.b64decode(attestation["signature_base64"], validate=True)
        public_cls.from_public_bytes(public_key_bytes).verify(signature, content)
    except (invalid_signature, ValueError):
        return False
    return True


def generate_checksum_manifest(
    artifacts: Sequence[str | Path],
    output_path: str | Path | None = None,
) -> dict[str, str]:
    """Generate SHA-256 checksums for release distributions in GNU sha256sum format."""
    manifest: dict[str, str] = {}
    lines: list[str] = []

    for art in artifacts:
        p = Path(art)
        if not p.is_file():
            raise FileNotFoundError(f"Artifact not found: {p}")
        digest = sha256_file(p)
        name = p.name
        manifest[name] = digest
        lines.append(f"{digest}  {name}\n")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.writelines(sorted(lines))

    return manifest


def verify_checksum_manifest(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> bool:
    """Verify all artifacts listed in a checksum manifest against their calculated SHA-256 digests."""
    m_path = Path(manifest_path)
    if not m_path.is_file():
        return False

    directory = Path(base_dir) if base_dir is not None else m_path.parent

    with m_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return False

    for line in lines:
        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) != 2:
            return False
        expected_digest, filename = parts[0].strip(), parts[1].strip()
        if not re.fullmatch(r"[a-f0-9]{64}", expected_digest.lower()):
            return False

        target_file = directory / filename
        if not target_file.is_file():
            return False

        actual_digest = sha256_file(target_file)
        if actual_digest.lower() != expected_digest.lower():
            return False

    return True


def validate_release_package(path: str | Path) -> dict[str, Any]:
    """Validate release wheel or source distribution packaging convention and integrity."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Release package not found: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"Release package is empty: {p}")

    name = p.name
    is_wheel = name.endswith(".whl")
    is_sdist = name.endswith(".tar.gz") or name.endswith(".zip")
    if not (is_wheel or is_sdist):
        raise ValueError(f"Unrecognized package distribution format: {name}")

    digest = sha256_file(p)
    return {
        "filename": name,
        "format": "wheel" if is_wheel else "sdist",
        "size_bytes": p.stat().st_size,
        "sha256": digest,
    }
