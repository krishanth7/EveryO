# EveryO Security & Supply Chain Qualification

## 1. Threat Model & Security Architecture

EveryO operates across training, experimentation, serialization, and serving stages. The security architecture adheres to a strict **zero-trust ingest** and **fail-closed** paradigm.

### 1.1 Trust Boundaries & Assets

| Asset / Boundary | Threat Surface | Primary Risk | Mitigation Control |
|---|---|---|---|
| **Model Artifacts (`.evo`)** | Ingestion of models from external sources | Arbitrary Code Execution (ACE), Zip Bombs, Path Traversal | `allow_pickle=False`, strict zip decompression quotas, duplicate manifest rejection, detached Ed25519 signatures |
| **Inference HTTP API** | HTTP requests from external networks | Denial of Service (DoS), Payload Tampering, Memory Exhaustion | Nginx reverse proxy rate limiting, bounded body size (`1MB`), constant-time token comparison (`hmac.compare_digest`), bounded concurrent worker slots |
| **Telemetry & Logging** | Serving metrics and request processing | Sensitive data / IP leakage | Request bodies, model parameters, and auth tokens are strictly omitted from logs; only aggregate counters retained |
| **Supply Chain & Dependencies** | Upstream PyPI packages and build toolchain | Compromised dependencies, license violations | CycloneDX 1.5 & SPDX 2.3 SBOM generation, automated GitHub CodeQL analysis, minimal runtime dependencies (`numpy`, `matplotlib`) |

---

## 2. Model Ingestion & Deserialization Defenses

EveryO model artifacts (`.evo`) are compressed ZIP archives containing:
1. `manifest.json`: JSON metadata describing architecture, shapes, and layer hyperparameters.
2. `parameters.npz`: Compressed NumPy numeric tensors.

### 2.1 Defense Mechanisms

1. **Pickle Elimination**:
   Standard machine learning serialization formats (such as PyTorch `.pt` or Python `pickle`) are vulnerable to arbitrary code execution during unpickling. EveryO exclusively utilizes `np.load(..., allow_pickle=False)`. Attempts to inject pickled objects are rejected immediately.
2. **Decompression Bomb Protection**:
   Archive readers enforce explicit decompression size ceilings:
   - Manifest size is capped at `MAX_MANIFEST_BYTES` (1 MB).
   - Parameter expansion is bounded to prevent disk and memory exhaustion.
3. **Duplicate Entry Rejection**:
   ZIP archives containing duplicate filenames (a technique used in ZIP spoofing attacks) are rejected during manifest loading.
4. **Directory Traversal Protection**:
   Archive extraction does not unpack files into the host filesystem arbitrarily; components are read directly from in-memory byte streams.

---

## 3. Cryptographic Provenance & Artifact Signing

EveryO includes native Ed25519 digital signature signing and verification (`everyo.artifact_signing`).

### 3.1 Signature Envelope Architecture

When an artifact is signed, a detached signature envelope is generated:

```json
{
  "artifact_name": "model.evo",
  "version": "0.5.0",
  "key_id": "production-key-2026",
  "sha256": "4b92b6...",
  "manifest_digest": "7d3a1...",
  "parameters_digest": "8e2f...",
  "signature": "3045022100..."
}
```

### 3.2 Verification Rules

- **Digest Validation**: The SHA-256 digest of the candidate file is recomputed and compared against `sha256`.
- **Signature Verification**: The signature is verified against the publisher's Ed25519 public key using `cryptography.hazmat.primitives.asymmetric.ed25519`.
- **Metadata Consistency**: Version and key identifiers must strictly match expected deployment policy rules. Any tampering with either the archive bytes or the metadata envelope fails verification.

---

## 4. Software Bill of Materials (SBOM) & Supply Chain

EveryO provides automated SBOM generation compliant with both major global standards:
- **CycloneDX v1.5 JSON**: Designed for vulnerability scanning and automated compliance.
- **SPDX v2.3 JSON**: Designed for intellectual property, licensing, and supply chain provenance.

### 4.1 Dependency Governance & Licensing

EveryO enforces an unencumbered open-source license policy. Every runtime and development dependency is audited for license compatibility:

| Component | Category | License | Purpose |
|---|---|---|---|
| **everyo** | Core Framework | MIT | Neural computing and acceleration framework |
| **numpy** | Runtime Core | BSD-3-Clause | Fundamental tensor mathematics |
| **matplotlib** | Runtime Core | PSF | Non-interactive visualization |
| **tensorflow** | Optional Accelerator | Apache-2.0 | Reference execution backend |
| **onnx** / **onnxruntime** | Optional Exporter | Apache-2.0 / MIT | Neural network graph interoperability |
| **cryptography** | Optional Security | Apache-2.0 / BSD | Asymmetric Ed25519 signing primitives |
| **PyYAML** | Optional Config | MIT | Safe YAML configuration loading |

### 4.2 Automated SBOM Generation

```bash
# Generate CycloneDX 1.5 JSON
python scripts/generate_sbom.py --format cyclonedx --output cyclonedx-sbom.json

# Generate SPDX 2.3 JSON
python scripts/generate_sbom.py --format spdx --output spdx-sbom.json
```

Both outputs include full package URLs (`purl`), version identifiers, declared licenses, and dependency relationships (`DEPENDS_ON`, `OPTIONAL_DEPENDENCY_OF`).

---

## 5. Network Serving Security Hardening

For production deployments, `everyo.serving.CPUModelServer` is deployed behind a hardened reverse proxy (Nginx).

### 5.1 Application Server Hardening (`everyo/serving.py`)

- **Loopback Enforcement**: The internal HTTP server binds by default to `127.0.0.1` and refuses non-loopback binding unless an explicit `api_token` is configured.
- **Constant-Time Comparison**: Token authentication uses `hmac.compare_digest(supplied, "Bearer " + token)` to prevent timing attacks.
- **Concurrency Limiting**: Active request execution is governed by `threading.BoundedSemaphore(max_concurrent)`. When saturated, the server returns HTTP 503 (`{"error":"busy"}`) without thread starvation or unbounded queuing.
- **Zero Information Leakage**: Internal exception stack traces, memory addresses, or Python error strings are never returned to HTTP clients. Responses use standardized, non-leaking error identifiers (`invalid_input`, `unauthorized`, `request_timeout`, `inference_failed`).

### 5.2 Network Gateway Hardening (`configs/deployment/nginx.conf`)

- **TLS 1.2 / 1.3 Termination**: Older protocols (SSLv3, TLS 1.0, TLS 1.1) and insecure ciphers are disabled.
- **Security Headers**: Injects `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and strict `Content-Security-Policy`.
- **Rate & Connection Limiting**: Caps individual clients at 50 requests/sec with burst capacity of 20, and max 10 concurrent TCP connections per IP.
- **Probe Segmentation**: `/health` and `/ready` probes are routed without rate limiting; `/metrics` is restricted to internal monitoring subnets.

---

## 6. Verification & Automated Test Evidence

Automated security qualification tests are implemented in:
- `tests/test_production_security_qualification.py`
- `tests/test_production_archive_security.py`
- `tests/test_production_signing_load.py`

| Test Suite | Evaluated Property | Result |
|---|---|---|
| `TestSupplyChainAndSbomQualification` | CycloneDX 1.5 and SPDX 2.3 schema adherence and metadata | **PASS** |
| `TestModelIngestionAndDeserializationSecurity` | Zip bomb bounds, duplicate manifest rejection, unpickling prevention | **PASS** |
| `TestCryptographicIntegrityAndProvenance` | Ed25519 signing, payload modification detection, key mismatch rejection | **PASS** |
| `TestHealthAndObservabilityEndpoints` | Security headers (`no-store`, `nosniff`), constant-time auth | **PASS** |
