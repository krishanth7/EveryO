"""Phase 19: fail-closed deployment configuration checks.

Configuration validation is not a substitute for real network security.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeploymentPolicy:
    host: str = "127.0.0.1"
    port: int = 8080
    max_batch: int = 32
    max_request_bytes: int = 1_000_000
    worker_count: int = 1
    tls_terminated: bool = False
    authentication_enabled: bool = False
    audit_enabled: bool = False

    def validate(self, *, environment: str = "development") -> None:
        if environment not in {"development", "staging", "production"}:
            raise ValueError("environment must be development/staging/production")
        if (
            not 1 <= self.port <= 65535
            or not 1 <= self.max_batch <= 1024
            or not 1 <= self.max_request_bytes <= 10_000_000
            or not 1 <= self.worker_count <= 64
        ):
            raise ValueError("deployment limits exceed supported bounds")
        try:
            loopback = ipaddress.ip_address(self.host).is_loopback
        except ValueError as exc:
            raise ValueError("host must be an explicit IP address") from exc
        if not loopback and not (self.authentication_enabled and self.tls_terminated):
            raise ValueError("network-facing configuration requires TLS and authentication")
        if environment == "production" and not (
            self.authentication_enabled and self.tls_terminated and self.audit_enabled
        ):
            raise ValueError("production requires TLS, authentication and audit configuration")


def validate_deployment_artifact(path: str | Path, *, expected_digest: str) -> bool:
    """Verify byte identity of a model file at the deployment boundary."""
    from everyo.governance import verify_artifact

    if Path(path).suffix != ".evo":
        raise ValueError("deployments require .evo model archives")
    if not verify_artifact(path, expected_digest):
        raise ValueError("deployment artifact digest mismatch")
    return True
