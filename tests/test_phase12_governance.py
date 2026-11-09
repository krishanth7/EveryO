"""Release integrity and fail-closed qualification acceptance tests."""

import json

import pytest

from everyo.governance import (
    audit_event,
    build_release_manifest,
    dependency_inventory,
    qualification_report,
    redact_metadata,
    sha256_file,
    validate_release_version,
    verify_artifact,
    write_release_manifest,
)


def test_integrity_and_tampering(tmp_path):
    path = tmp_path / "model.evo"
    path.write_bytes(b"trusted-test-artifact")
    digest = sha256_file(path)
    assert verify_artifact(path, digest)
    path.write_bytes(b"tampered")
    assert not verify_artifact(path, digest)
    with pytest.raises(ValueError):
        sha256_file(path, max_bytes=3)


def test_fail_closed_release_gates():
    passing = qualification_report(
        {"accuracy": 0.9, "latency_ms": 4}, {"accuracy": (">=", 0.8), "latency_ms": ("<=", 10)}
    )
    assert passing["passed"]
    missing = qualification_report({}, {"accuracy": (">=", 0.8)})
    assert not missing["passed"]
    with pytest.raises(ValueError):
        qualification_report({}, {})


def test_manifest_safety_and_atomic_save(tmp_path):
    artifact = tmp_path / "model.evo"
    artifact.write_bytes(b"test")
    gates = qualification_report({"accuracy": 1}, {"accuracy": (">=", 0.9)})
    manifest = build_release_manifest(
        artifact,
        version="0.1.0",
        qualification=gates,
        metadata={"owner": "lab", "api_token": "never-persist"},
    )
    assert "api_token" not in manifest["metadata"]
    assert dependency_inventory(("numpy",))
    output = tmp_path / "release.json"
    write_release_manifest(manifest, output)
    assert json.loads(output.read_text())["sha256"] == sha256_file(artifact)
    with pytest.raises(FileExistsError):
        write_release_manifest(manifest, output)
    with pytest.raises(ValueError):
        build_release_manifest(artifact, version="0.1.0", qualification={"passed": False})


def test_version_redaction_audit():
    assert validate_release_version("2.3.4") == "2.3.4"
    with pytest.raises(ValueError):
        validate_release_version("v2.3.4; rm -rf")
    assert redact_metadata({"profile": {"email": "hidden", "name": "kept"}}) == {
        "profile": {"name": "kept"}
    }
    event = audit_event("release", artifact_hash="a" * 64, result="pass")
    assert event["result"] == "pass"
    with pytest.raises(ValueError):
        audit_event("release", artifact_hash="bad", result="pass")
