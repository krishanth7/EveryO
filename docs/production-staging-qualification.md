# EveryO CPU staging qualification additions

These checks are reproducible engineering evidence, **not** an independent
penetration test or certification of an externally deployed service.

## Implemented

- Dependency audit: run `python -m pip_audit --local --progress-spinner off`
  in the intended deployment environment, and fail deployment on reported
  vulnerabilities. Version-range requirements are not a pinned supply chain;
  produce a deployment lockfile and audit its resolved environment.
- Static analysis: retain GitHub CodeQL and Ruff workflows. Review findings
  rather than treating their absence as independent security approval.
- Authenticated HTTP probe: `PYTHONPATH=. python -m benchmarks.benchmark_local_http
  --requests 10000 --concurrency 4` records successes, failures, p50, p95,
  p99 and throughput; exits nonzero on request failures. The probe is forcibly
  limited to loopback to prevent accidental load on third-party services.
- Security tests: unauthorized HTTP requests, request timeout/recovery, bounded
  payload handling and artifact tamper detection are in the regression suite.
- Signed artifact: install `everyo[signing]`, then call
  `everyo.artifact_signing.sign_artifact` with an externally supplied 32-byte
  Ed25519 private-key seed. Only save the returned detached attestation.
  Verify with `verify_signed_artifact` and a separately trusted public key;
  verification fails for wrong keys, altered contents and release metadata.

## Production deployment checklist (not yet externally verified)

1. Establish a dedicated staging and production host, gateway, and DNS.
2. Enforce TLS (valid certificate, safe cipher policy) at the trusted proxy;
   keep the Python model server bound to loopback or a private network.
3. Put API tokens in a dedicated secrets manager, rotate and revoke them,
   and restrict rate/concurrency at the gateway. Verify unauthorized traffic.
4. Add structured, non-sensitive logs, centralized metrics, alerts, backup and
   recovery controls. Practice a deployment rollback and incident response.
5. Execute service soak, resource-exhaustion and dependency failure tests with
   actual production model/input shapes. Set and verify acceptance SLOs.
6. Maintain signing keys outside source control and CI logs in KMS/HSM; pin
   trusted public-key identifiers in the deployment verifier. Sign a digest
   and verify it before loading a production artifact.
7. Commission a separate security reviewer to perform penetration testing and
   record vulnerabilities and remediation sign-off.
8. Run sealed, representative held-out evaluation with independently defined
   minimum accuracy/quality gates and sample provenance.

**Release gate remains FAIL until these external checks have actual evidence.**
CPU staging performance is not a latency SLA for other hardware, model sizes,
or remote networks. An Ed25519 signature without managed signing-key custody
and trustworthy public-key distribution is not a complete provenance system.
