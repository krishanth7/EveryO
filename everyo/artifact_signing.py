"""Detached Ed25519 artifact attestations with externally managed signing keys.

This module intentionally never generates, stores, or transmits a private key.
The caller supplies trusted key bytes from an external key-management system.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

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
