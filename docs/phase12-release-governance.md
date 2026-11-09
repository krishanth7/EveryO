# Phase 12 — CPU release governance and artifact integrity

Phase 12 adds deterministic artifact integrity checks and fail-closed local
release gates. It does not digitally sign artifacts or replace security audits.

| Milestone | Feature | Verification |
| --- | --- | --- |
| 12.1 | `sha256_file` | Streamed SHA-256 with size bound |
| 12.2 | `verify_artifact` | Reject altered bytes |
| 12.3 | `qualification_report` | Missing/nonfinite metrics fail gates |
| 12.4 | `validate_release_version` | Strict three-part semantic version |
| 12.5 | `dependency_inventory` | Installed versions recorded; not full SBOM |
| 12.6 | `redact_metadata` and `audit_event` | Sanitized structured evidence |
| 12.7 | `build_release_manifest`, `write_release_manifest` | JSON evidence, restrictive temp file, atomic replace |
| 12.8 | Acceptance tests | `tests/test_phase12_governance.py` |

## Local qualification example

```python
from everyo.governance import (
    build_release_manifest,
    qualification_report,
    write_release_manifest,
)

gates = qualification_report(
    {"test_accuracy": 0.91, "p95_ms": 4.2},
    {"test_accuracy": (">=", 0.90), "p95_ms": ("<=", 10.0)},
)
manifest = build_release_manifest("model.evo", version="0.1.0", qualification=gates)
write_release_manifest(manifest, "model.release.json")
```

**Limitations:** SHA-256 establishes *byte identity*, not authenticity: an
attacker able to replace the artifact and manifest can forge both. Use signed
attestations and trusted keys for provenance guarantees. Dependency inventory
is not a transitive SBOM or vulnerability scan. Redaction uses key-name rules,
not exhaustive personal-data detection. Release gates reflect only the
measurements supplied by the caller; independent evaluation is still required.

Phase 12 does **not** create a GitHub Release or publish a package to PyPI.
