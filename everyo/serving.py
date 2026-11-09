"""Small CPU inference HTTP service for explicitly trusted local deployments.

No cloud dependency. The default server listens on loopback only. Network-facing
deployments still require a reverse proxy, TLS, monitoring and threat modeling.
"""

from __future__ import annotations

import hmac
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from everyo.inference import CPUInferenceRuntime


class ServiceMetrics:
    """Thread-safe aggregate counters (no user data or request payloads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._errors = 0
        self._latency_seconds = 0.0

    def record(self, *, failed: bool, duration: float) -> None:
        with self._lock:
            self._requests += 1
            self._errors += int(failed)
            self._latency_seconds += max(0.0, duration)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "requests": self._requests,
                "errors": self._errors,
                "total_latency_seconds": self._latency_seconds,
            }


class CPUModelServer(ThreadingHTTPServer):
    """HTTP endpoints: GET /health, GET /metrics, POST /v1/predict."""

    daemon_threads = True

    def __init__(
        self,
        runtime: CPUInferenceRuntime,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        api_token: str | None = None,
        max_body_bytes: int = 1_000_000,
        max_concurrent: int = 4,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(runtime, CPUInferenceRuntime):
            raise TypeError("runtime must be a CPUInferenceRuntime")
        if host not in {"127.0.0.1", "::1", "localhost"} and not api_token:
            raise ValueError("non-loopback serving requires an API token")
        if max_body_bytes <= 0 or max_concurrent <= 0 or not 0 < request_timeout_seconds <= 300:
            raise ValueError("resource limits must be positive")
        self.runtime = runtime
        self.api_token = api_token
        self.max_body_bytes = max_body_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self.slots = threading.BoundedSemaphore(max_concurrent)
        self.metrics = ServiceMetrics()
        super().__init__((host, port), _handler_factory())

    def run_background(self) -> threading.Thread:
        worker = threading.Thread(target=self.serve_forever, name="everyo-cpu-http", daemon=True)
        worker.start()
        return worker


def _handler_factory() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server: CPUModelServer

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(self.server.request_timeout_seconds)

        def log_message(self, format: str, *args: Any) -> None:
            # Never log request paths, auth tokens or raw model inputs.
            return

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            content = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def _authorized(self) -> bool:
            token = self.server.api_token
            if token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, "Bearer " + token)

        def do_GET(self) -> None:
            if not self._authorized():
                self._reply(401, {"error": "unauthorized"})
            elif self.path == "/health":
                self._reply(200, {"status": "ok", "backend": "cpu"})
            elif self.path == "/metrics":
                self._reply(200, self.server.metrics.snapshot())
            else:
                self._reply(404, {"error": "not_found"})

        def do_POST(self) -> None:
            start = time.monotonic()
            failed = True
            try:
                if not self._authorized():
                    self._reply(401, {"error": "unauthorized"})
                elif self.path != "/v1/predict":
                    self._reply(404, {"error": "not_found"})
                elif not self.server.slots.acquire(blocking=False):
                    self._reply(503, {"error": "busy"})
                else:
                    try:
                        failed = not self._predict()
                    finally:
                        self.server.slots.release()
            finally:
                self.server.metrics.record(failed=failed, duration=time.monotonic() - start)

        def _predict(self) -> bool:
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length <= 0 or length > self.server.max_body_bytes:
                    self._reply(413, {"error": "request_size_invalid"})
                    return False
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    self._reply(415, {"error": "expected_json"})
                    return False
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or set(payload) != {"inputs"}:
                    raise ValueError("expected a JSON object containing only inputs")
                scores = self.server.runtime.predict_batched(np.asarray(payload["inputs"]))
                self._reply(200, {"outputs": scores.tolist(), "count": len(scores)})
                return True
            except (socket.timeout, TimeoutError):
                self._reply(408, {"error": "request_timeout"})
            except (ValueError, TypeError, UnicodeDecodeError, OverflowError):
                self._reply(400, {"error": "invalid_input"})
            except Exception:
                # Internal error details must never reach an HTTP client.
                self._reply(500, {"error": "inference_failed"})
            return False

    return Handler
