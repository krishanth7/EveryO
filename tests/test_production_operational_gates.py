"""CPU-only operational regression tests, not an external production certification."""

import json
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import pytest

from everyo import Linear, save
from everyo.deployment_policy import DeploymentPolicy, validate_deployment_artifact
from everyo.evaluation import assert_no_leakage, classification_metrics
from everyo.governance import sha256_file
from everyo.inference import CPUInferenceRuntime
from everyo.release_qualification import REQUIRED_GATES, qualify_release
from everyo.serving import CPUModelServer


def _predict(url):
    data = json.dumps({"inputs": [[1.0, 2.0]]}).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=4) as response:
        return response.status, json.load(response)


def test_concurrent_cpu_service_recovers_after_idle_client_timeout():
    runtime = CPUInferenceRuntime(Linear(2, 2, seed=4))
    with CPUModelServer(runtime, request_timeout_seconds=0.2) as server:
        worker = server.run_background()
        try:
            address = ("127.0.0.1", server.server_port)
            with socket.create_connection(address, timeout=2) as slow:
                slow.sendall(
                    b"POST /v1/predict HTTP/1.1\r\nHost: localhost\r\n"
                    b"Content-Type: application/json\r\nContent-Length: 50\r\n\r\n{}"
                )
                slow.settimeout(2)
                reply = slow.recv(4096)
                assert b"408" in reply

            url = f"http://127.0.0.1:{server.server_port}/v1/predict"
            with ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(pool.map(_predict, [url] * 20))
            assert all(code == 200 and result["count"] == 1 for code, result in replies)
            metrics = server.metrics.snapshot()
            assert metrics["requests"] >= 21 and metrics["errors"] >= 1
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_heldout_leakage_rejected_and_metrics_require_valid_labels():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(
            {"train": [0], "validation": [1], "test": [2]},
            fingerprints=["same", "different", "same"],
        )
    with pytest.raises(ValueError):
        classification_metrics([[0.2, 0.8]], [2])
    metrics = classification_metrics([[0.0, 1.0], [2.0, 0.0]], [1, 0])
    assert metrics["accuracy"] == 1.0 and metrics["samples"] == 2.0


def test_artifact_tamper_and_missing_release_evidence_fail_closed(tmp_path):
    artifact = save(Linear(2, 2, seed=1), tmp_path / "model.evo")
    digest = sha256_file(artifact)
    assert validate_deployment_artifact(artifact, expected_digest=digest)
    with artifact.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_deployment_artifact(artifact, expected_digest=digest)

    verdict = qualify_release(
        "0.2.0", {name: True for name in REQUIRED_GATES if name != "load_test"}
    )
    assert not verdict.approved and verdict.missing == ("load_test",)
    with pytest.raises(ValueError, match="production requires"):
        DeploymentPolicy().validate(environment="production")
