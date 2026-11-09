"""Phase 20: auditable CPU release-qualification decision records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from everyo.governance import qualification_report, validate_release_version

REQUIRED_GATES = (
    "unit_tests",
    "integration_tests",
    "heldout_evaluation",
    "security_review",
    "load_test",
    "artifact_integrity",
)


@dataclass(frozen=True)
class ReleaseDecision:
    version: str
    approved: bool
    checks: dict[str, bool]
    missing: tuple[str, ...]


def qualify_release(version: str, evidence: Mapping[str, bool]) -> ReleaseDecision:
    """Fail closed if any mandatory, independently verified evidence is absent."""
    validate_release_version(version)
    if any(not isinstance(v, bool) for v in evidence.values()):
        raise ValueError("each release evidence value must be a boolean")
    checks = {gate: bool(evidence.get(gate, False)) for gate in REQUIRED_GATES}
    missing = tuple(gate for gate in REQUIRED_GATES if not checks[gate])
    return ReleaseDecision(version, not missing, checks, missing)


def qualify_metric_gates(
    metrics: Mapping[str, float], gates: Mapping[str, tuple[str, float]]
) -> dict[str, Any]:
    return qualification_report(metrics, gates)


def release_decision_json(decision: ReleaseDecision) -> str:
    return json.dumps(
        {
            "version": decision.version,
            "approved": decision.approved,
            "checks": decision.checks,
            "missing": list(decision.missing),
        },
        sort_keys=True,
    )
