"""Signed release artifacts and bounded authenticated CPU load probes."""

import pytest

from everyo import Linear, save
from everyo.artifact_signing import sign_artifact, verify_signed_artifact
from everyo.inference import CPUInferenceRuntime
from everyo.load_qualification import benchmark_authenticated_service
from everyo.serving import CPUModelServer


def test_detached_signature_rejects_tampering_and_wrong_keys(tmp_path):
    pytest.importorskip("cryptography", reason="optional everyo[signing] dependency")
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
    model = save(Linear(2, 2, seed=1), tmp_path / "model.evo")
    envelope = sign_artifact(
        model, version="0.2.0", key_id="test-local-key", private_key_bytes=private_bytes
    )
    assert verify_signed_artifact(model, envelope, public_key_bytes=public_bytes)
    assert not verify_signed_artifact(
        model,
        envelope,
        public_key_bytes=Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
    )
    envelope["version"] = "0.2.1"
    assert not verify_signed_artifact(model, envelope, public_key_bytes=public_bytes)
    envelope["version"] = "0.2.0"
    with model.open("ab") as stream:
        stream.write(b"modified")
    assert not verify_signed_artifact(model, envelope, public_key_bytes=public_bytes)


def test_cpu_authenticated_load_and_access_policy():
    runtime = CPUInferenceRuntime(Linear(2, 2, seed=1))
    with CPUModelServer(runtime, api_token="test-token", max_concurrent=8) as server:
        worker = server.run_background()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1/predict"
            evidence = benchmark_authenticated_service(
                url, token="test-token", inputs=[[1.0, 2.0]], requests=40, concurrency=4
            )
            assert evidence.successes == 40 and evidence.failures == 0
            assert 0 < evidence.p50_ms <= evidence.p95_ms <= evidence.p99_ms
            wrong = benchmark_authenticated_service(
                url, token="bad-token", inputs=[[1.0, 2.0]], requests=2
            )
            assert wrong.failures == 2
            with pytest.raises(ValueError, match="local staging"):
                benchmark_authenticated_service(
                    "https://example.com/v1/predict", token="secret", inputs=[[1.0, 2.0]]
                )
        finally:
            server.shutdown()
            worker.join(timeout=5)
