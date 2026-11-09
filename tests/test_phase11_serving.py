"""CPU-only service behavior and access-control regression tests."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from everyo import Linear
from everyo.inference import CPUInferenceRuntime
from everyo.serving import CPUModelServer, ServiceMetrics


def call(url, *, inputs=None, token=None, content_type="application/json"):
    headers = {"Content-Type": content_type}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    data = None if inputs is None else json.dumps(inputs).encode()
    request = Request(url, data=data, headers=headers)
    try:
        with urlopen(request, timeout=5) as result:
            return result.status, json.load(result)
    except HTTPError as error:
        return error.code, json.load(error)


def test_service_endpoints_and_security():
    runtime = CPUInferenceRuntime(Linear(2, 2, seed=2), max_batch=2)
    with CPUModelServer(runtime, api_token="test-token", max_body_bytes=1000) as server:
        worker = server.run_background()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            assert call(base + "/health")[0] == 401
            assert call(base + "/health", token="test-token") == (
                200,
                {"status": "ok", "backend": "cpu"},
            )
            assert call(base + "/wrong", token="test-token")[0] == 404
            status, result = call(
                base + "/v1/predict",
                inputs={"inputs": [[1, 2], [3, 4], [5, 6]]},
                token="test-token",
            )
            assert status == 200 and result["count"] == 3 and len(result["outputs"]) == 3
            assert (
                call(
                    base + "/v1/predict", inputs={"inputs": [[float("nan"), 2]]}, token="test-token"
                )[0]
                == 400
            )
            assert call(base + "/v1/predict", inputs={"bad": []}, token="test-token")[0] == 400
            assert call(base + "/v1/predict", inputs={"inputs": [[1, 2]]}, token="wrong")[0] == 401
            assert (
                call(
                    base + "/v1/predict",
                    inputs={"inputs": [[1, 2]]},
                    token="test-token",
                    content_type="text/plain",
                )[0]
                == 415
            )
            status, metrics = call(base + "/metrics", token="test-token")
            assert status == 200 and metrics["requests"] >= 5 and metrics["errors"] >= 3
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_nonlocal_binding_requires_token():
    runtime = CPUInferenceRuntime(Linear(2, 1, seed=2))
    with pytest.raises(ValueError, match="token"):
        CPUModelServer(runtime, host="0.0.0.0")
    with pytest.raises(ValueError):
        CPUModelServer(runtime, max_concurrent=0)


def test_metrics_thread_safe():
    stats = ServiceMetrics()
    stats.record(failed=False, duration=0.1)
    stats.record(failed=True, duration=0.2)
    result = stats.snapshot()
    assert result["requests"] == 2 and result["errors"] == 1
    assert np.isclose(result["total_latency_seconds"], 0.3)
