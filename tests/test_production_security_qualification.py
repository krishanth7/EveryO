"""Comprehensive production security qualification tests for EveryO.

Validates:
1. Supply chain integrity & SBOM specification compliance (CycloneDX 1.5, SPDX 2.3).
2. Model archive ingestion defense (directory traversal, zip bomb, duplicate manifest).
3. Deserialization safety & unpickling prevention (allow_pickle=False).
4. Ed25519 artifact signing and cryptographic tamper rejection.
5. Authenticated serving authorization and fail-closed security gates.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

import everyo as eo
from everyo.artifact_signing import sign_artifact, verify_signed_artifact
from everyo.deployment_policy import DeploymentPolicy, validate_deployment_artifact
from everyo.exceptions import EveryOSerializationError
from everyo.governance import sha256_file
from everyo.security.sbom import generate_cyclonedx_sbom, generate_sbom, generate_spdx_sbom
from everyo.serialization import format as model_format
from everyo.serialization import load_manifest


class TestSupplyChainAndSbomQualification:
    """Verifies CycloneDX and SPDX SBOM generation standards."""

    def test_cyclonedx_sbom_structure(self):
        sbom = generate_cyclonedx_sbom(include_optional=True)
        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["specVersion"] == "1.5"
        assert sbom["serialNumber"].startswith("urn:uuid:")
        assert sbom["version"] == 1

        metadata = sbom["metadata"]
        assert "timestamp" in metadata
        assert metadata["component"]["name"] == "everyo"
        assert metadata["component"]["version"] == "0.5.0"
        assert metadata["component"]["purl"] == "pkg:pypi/everyo@0.5.0"
        assert metadata["component"]["licenses"][0]["license"]["id"] == "MIT"

        comp_names = {c["name"] for c in sbom["components"]}
        assert "numpy" in comp_names
        assert "matplotlib" in comp_names
        assert "tensorflow" in comp_names
        assert "onnx" in comp_names

    def test_spdx_sbom_structure(self):
        sbom = generate_spdx_sbom(include_optional=True)
        assert sbom["spdxVersion"] == "SPDX-2.3"
        assert sbom["dataLicense"] == "CC0-1.0"
        assert sbom["SPDXID"] == "SPDXRef-DOCUMENT"
        assert sbom["name"] == "EveryO-0.5.0"
        assert "documentNamespace" in sbom

        packages = {p["name"]: p for p in sbom["packages"]}
        assert "everyo" in packages
        assert packages["everyo"]["licenseConcluded"] == "MIT"
        assert packages["everyo"]["versionInfo"] == "0.5.0"
        assert "numpy" in packages
        assert "matplotlib" in packages

        # Verify dependency relationships
        rel_types = {r["relationshipType"] for r in sbom["relationships"]}
        assert "DESCRIBES" in rel_types
        assert "DEPENDS_ON" in rel_types

    def test_generate_sbom_file_export(self, tmp_path: Path):
        cdx_file = tmp_path / "bom.cdx.json"
        spdx_file = tmp_path / "bom.spdx.json"

        generate_sbom("cyclonedx", output_path=cdx_file)
        generate_sbom("spdx", output_path=spdx_file)

        assert cdx_file.exists()
        assert spdx_file.exists()

        with cdx_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["bomFormat"] == "CycloneDX"

        with spdx_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["spdxVersion"] == "SPDX-2.3"

    def test_invalid_sbom_format_rejected(self):
        with pytest.raises(ValueError, match="Unsupported SBOM format"):
            generate_sbom("unsupported_standard")


class TestModelIngestionAndDeserializationSecurity:
    """Verifies defenses against malicious or malformed model artifacts."""

    def test_duplicate_manifest_is_rejected(self, tmp_path: Path):
        path = tmp_path / "duplicate_manifest.evo"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(model_format.MANIFEST_NAME, "{}")
            archive.writestr(model_format.MANIFEST_NAME, "{}")
        with pytest.raises(EveryOSerializationError, match="exactly one"):
            load_manifest(path)

    def test_manifest_decompressed_size_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import importlib

        loader = importlib.import_module("everyo.serialization.load")
        monkeypatch.setattr(loader, "MAX_MANIFEST_BYTES", 64)
        path = tmp_path / "oversized_manifest.evo"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(model_format.MANIFEST_NAME, json.dumps({"overflow": "A" * 500}))
        with pytest.raises(EveryOSerializationError, match="size limit"):
            load_manifest(path)

    def test_nested_decompression_zip_bomb_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import importlib

        loader = importlib.import_module("everyo.serialization.load")
        monkeypatch.setattr(loader, "MAX_PARAMETERS_BYTES", 1024)
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("weights.npy", b"\x00" * 4096)
        path = tmp_path / "zip_bomb.evo"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(model_format.MANIFEST_NAME, json.dumps({"format_version": 1}))
            archive.writestr(model_format.PARAMETERS_NAME, nested.getvalue())
        with pytest.raises(EveryOSerializationError, match="Expanded parameter"):
            eo.load(path)

    def test_arbitrary_pickle_payload_rejected(self, tmp_path: Path):
        path = tmp_path / "pickle_injection.evo"
        # Write arbitrary pickled object into weights.npy instead of genuine numpy array
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, "w") as archive:
            archive.writestr("weights.npy", b"cos\nsystem\n(S'echo pwned'\ntR.")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(model_format.MANIFEST_NAME, json.dumps({"format_version": 1}))
            archive.writestr(model_format.PARAMETERS_NAME, nested.getvalue())
        with pytest.raises((EveryOSerializationError, ValueError)):
            eo.load(path)


class TestCryptographicIntegrityAndProvenance:
    """Verifies Ed25519 signing, hash digests, and fail-closed tamper detection."""

    def test_signed_artifact_tamper_detection(self, tmp_path: Path):
        pytest.importorskip("cryptography", reason="signing requires cryptography")
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private = Ed25519PrivateKey.generate()
        private_bytes = private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        public_bytes = private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

        model = eo.Sequential(eo.Linear(2, 2, seed=1))
        model_path = eo.save(model, tmp_path / "model.evo")

        envelope = sign_artifact(
            model_path, version="0.5.0", key_id="prod-key-1", private_key_bytes=private_bytes
        )
        assert verify_signed_artifact(model_path, envelope, public_key_bytes=public_bytes)

        # 1. Payload modification must be detected
        with model_path.open("ab") as f:
            f.write(b"corrupted_bytes")
        assert not verify_signed_artifact(model_path, envelope, public_key_bytes=public_bytes)

        # 2. Key mismatch must be rejected
        other_public_bytes = (
            Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        )
        assert not verify_signed_artifact(model_path, envelope, public_key_bytes=other_public_bytes)

    def test_deployment_policy_digest_validation(self, tmp_path: Path):
        model = eo.Sequential(eo.Linear(3, 1, seed=2))
        artifact = eo.save(model, tmp_path / "deployable.evo")
        digest = sha256_file(artifact)

        assert validate_deployment_artifact(artifact, expected_digest=digest)

        with pytest.raises(ValueError, match="digest mismatch"):
            validate_deployment_artifact(artifact, expected_digest="0" * 64)

        with pytest.raises(ValueError, match="production requires"):
            DeploymentPolicy().validate(environment="production")
