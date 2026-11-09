"""CPU-friendly release provenance, integrity and lightweight qualification gates.

These primitives are local evidence generators, not signatures, SBOM attestation,
regulatory approval or a substitute for independent security review.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: str | Path, *, max_bytes: int = 250_000_000) -> str:
    """Stream-hash an existing file with a size limit."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    source = Path(path)
    if not source.is_file() or source.stat().st_size > max_bytes:
        raise ValueError("file missing or above size limit")
    digest = hashlib.sha256()
    consumed = 0
    with source.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            consumed += len(part)
            if consumed > max_bytes:
                raise ValueError("file exceeds size limit while reading")
            digest.update(part)
    return digest.hexdigest()


def verify_artifact(path: str | Path, expected_sha256: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be lowercase SHA-256 hex")
    import hmac

    return hmac.compare_digest(sha256_file(path), expected_sha256)


def dependency_inventory(names: tuple[str, ...] = ("numpy", "everyo")) -> list[dict[str, str]]:
    """Installed-package inventory; explicitly *not* a complete SBOM."""
    result = []
    for name in sorted(set(names)):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            raise ValueError("invalid dependency name")
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        result.append({"name": name, "version": version})
    return result


def validate_release_version(version: str) -> str:
    """Accept stable semantic versions only; reject tags with shell characters."""
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise ValueError("release version must be MAJOR.MINOR.PATCH")
    return version


def qualification_report(
    metrics: Mapping[str, float],
    gates: Mapping[str, tuple[str, float]],
) -> dict[str, Any]:
    """Evaluate explicit metric gates; missing and nonfinite values fail closed."""
    checks = {}
    for key, (operator, threshold) in gates.items():
        if operator not in {"<=", ">="} or not np.isfinite(threshold):
            raise ValueError("gates must use <= or >= with finite thresholds")
        value = metrics.get(key)
        valid = isinstance(value, (int, float)) and np.isfinite(value)
        passed = bool(valid and (value <= threshold if operator == "<=" else value >= threshold))
        checks[key] = {
            "passed": passed,
            "actual": float(value) if valid else None,
            "operator": operator,
            "threshold": float(threshold),
        }
    if not gates:
        raise ValueError("at least one release gate is required")
    return {"passed": all(row["passed"] for row in checks.values()), "checks": checks}


def redact_metadata(
    record: Mapping[str, Any],
    *,
    blocked: tuple[str, ...] = ("token", "password", "secret", "key", "email"),
) -> dict[str, Any]:
    """Remove recognized sensitive keys recursively, including nested objects."""

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(k): clean(v)
                for k, v in value.items()
                if not any(word in str(k).casefold() for word in blocked)
            }
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    return clean(record)


def build_release_manifest(
    artifact: str | Path,
    *,
    version: str,
    qualification: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_release_version(version)
    if qualification.get("passed") is not True:
        raise ValueError("cannot qualify a release with failing or absent gates")
    path = Path(artifact)
    result = {
        "schema_version": 1,
        "version": version,
        "artifact_name": path.name,
        "artifact_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "qualification": dict(qualification),
        "dependencies": dependency_inventory(),
        "python": sys.version.split()[0],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": redact_metadata(metadata or {}),
    }
    json.dumps(result, allow_nan=False)
    return result


def write_release_manifest(
    manifest: Mapping[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomic JSON writing with restrictive default permissions."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(manifest), indent=2, sort_keys=True, allow_nan=False).encode()
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    temporary = target.with_name(target.name + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def audit_event(action: str, *, artifact_hash: str, result: str) -> dict[str, str]:
    """Structured evidence without personal identifiers or arbitrary payloads."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", action):
        raise ValueError("invalid audit action")
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
        raise ValueError("invalid artifact hash")
    if result not in {"pass", "fail", "review"}:
        raise ValueError("invalid audit result")
    return {
        "action": action,
        "artifact_sha256": artifact_hash,
        "result": result,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
