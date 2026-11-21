"""Deployment readiness and observability qualification tests for EveryO.

Validates:
1. Health and liveness endpoints (/health, /healthz).
2. Readiness endpoints (/ready, /readyz).
3. Telemetry and metrics reporting (/metrics).
4. Security response headers (no-store, nosniff).
5. Reverse proxy (Nginx) configuration directives and rate limiting rules.
6. Containerization specifications (non-root execution, health probes).
7. Fail-closed model deployment backup and rollback procedures.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest

import everyo as eo
from everyo.deployment_policy import validate_deployment_artifact
from everyo.governance import sha256_file
from everyo.inference import CPUInferenceRuntime
from everyo.serving import CPUModelServer


class TestHealthAndObservabilityEndpoints:
    """Verifies standard container and orchestration probes."""

    @pytest.fixture
    def running_server(self):
        model = eo.Sequential(eo.Linear(2, 2, seed=1))
        runtime = CPUInferenceRuntime(model)
        server = CPUModelServer(runtime, api_token="deploy-token")
        worker = server.run_background()
        yield server
        server.shutdown()
        worker.join(timeout=5)

    def _get(self, server: CPUModelServer, path: str, token: str = "deploy-token"):
        url = f"http://127.0.0.1:{server.server_port}{path}"
        req = Request(url, headers={"Authorization": f"Bearer {token}"})
        with urlopen(req, timeout=3) as resp:
            headers = dict(resp.headers)
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, headers, body

    def test_liveness_probes_health_and_healthz(self, running_server: CPUModelServer):
        for path in ["/health", "/healthz"]:
            status, headers, body = self._get(running_server, path)
            assert status == 200
            assert body == {"status": "ok", "backend": "cpu"}
            assert headers.get("Cache-Control") == "no-store"
            assert headers.get("X-Content-Type-Options") == "nosniff"

    def test_readiness_probes_ready_and_readyz(self, running_server: CPUModelServer):
        for path in ["/ready", "/readyz"]:
            status, _, body = self._get(running_server, path)
            assert status == 200
            assert body == {"status": "ready", "backend": "cpu"}

    def test_metrics_endpoint(self, running_server: CPUModelServer):
        status, _, body = self._get(running_server, "/metrics")
        assert status == 200
        assert "requests" in body
        assert "errors" in body
        assert "total_latency_seconds" in body


class TestDeploymentConfigurations:
    """Verifies deployment configuration templates for security best practices."""

    def test_nginx_conf_directives(self):
        conf_path = Path("configs/deployment/nginx.conf")
        assert conf_path.exists()
        content = conf_path.read_text(encoding="utf-8")

        # TLS hardening
        assert "ssl_protocols TLSv1.2 TLSv1.3;" in content
        assert "ssl_prefer_server_ciphers on;" in content

        # Security headers
        assert "Strict-Transport-Security" in content
        assert "X-Content-Type-Options" in content
        assert "X-Frame-Options" in content
        assert "Content-Security-Policy" in content

        # Rate limiting and timeouts
        assert "limit_req_zone" in content
        assert "limit_conn_zone" in content
        assert "client_max_body_size 1m;" in content

        # Endpoints routing
        assert "location ~ ^/(health|healthz)$" in content
        assert "location ~ ^/(ready|readyz)$" in content
        assert "location = /metrics" in content
        assert "location = /v1/predict" in content

    def test_dockerfile_security_directives(self):
        dockerfile_path = Path("configs/deployment/Dockerfile")
        assert dockerfile_path.exists()
        content = dockerfile_path.read_text(encoding="utf-8")

        # Non-root user
        assert "APP_UID=10001" in content
        assert "useradd" in content
        assert "USER ${APP_USER}:${APP_USER}" in content

        # Healthcheck probe
        assert "HEALTHCHECK" in content
        assert "/health" in content


class TestBackupAndRollbackProcedures:
    """Verifies fail-closed deployment rollbacks upon invalid or corrupted model artifacts."""

    def test_model_backup_and_safe_rollback(self, tmp_path: Path):
        # 1. Active Production Model (v1)
        model_v1 = eo.Sequential(eo.Linear(2, 2, seed=10))
        path_v1 = eo.save(model_v1, tmp_path / "model_v1.0.0.evo")
        digest_v1 = sha256_file(path_v1)
        assert validate_deployment_artifact(path_v1, expected_digest=digest_v1)

        # 2. Candidate Model (v2) - simulate candidate preparation
        model_v2 = eo.Sequential(eo.Linear(2, 2, seed=20))
        path_v2 = eo.save(model_v2, tmp_path / "model_v2.0.0.evo")
        digest_v2 = sha256_file(path_v2)

        # Simulate tampering / validation failure on candidate
        with path_v2.open("ab") as stream:
            stream.write(b"corrupted_tail")

        active_path = path_v1
        rollback_occurred = False

        try:
            # Pre-flight deployment check
            validate_deployment_artifact(path_v2, expected_digest=digest_v2)
            active_path = path_v2  # Should not be reached
        except ValueError:
            # Rollback to last known healthy checkpoint
            active_path = path_v1
            rollback_occurred = True

        assert rollback_occurred is True
        assert active_path == path_v1

        # Restored active model must be fully operational
        restored_runtime = CPUInferenceRuntime(eo.load(active_path))
        preds = restored_runtime.predict(np.array([[1.0, 2.0]]))
        assert preds.shape == (1, 2)
