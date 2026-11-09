# Phase 11 — CPU inference serving foundation

Phase 11 introduces **local-only by default**, authenticated, resource-bounded
HTTP inference for numeric array-input EveryO models. No external AI service,
GPU, web framework, or database is required.

| Milestone | Feature | Verification |
| --- | --- | --- |
| 11.1 | `/v1/predict` JSON contract | Numeric arrays in, JSON predictions out |
| 11.2 | Input schema and content-type | Invalid JSON, shapes, type rejected |
| 11.3 | Body and inference limits | `max_body_bytes`, `max_batch`, `max_elements` |
| 11.4 | Optional bearer-token authorization | Unauthorized traffic rejected; token mandatory on nonloopback |
| 11.5 | Concurrency control | Bounded semaphore and 503 busy response |
| 11.6 | Health and metrics | `/health`, `/metrics` and error/latency counters |
| 11.7 | Graceful lifecycle | `run_background()`, `shutdown()`, `server_close()` |
| 11.8 | Integration tests | `tests/test_phase11_serving.py` |

## Minimal local server

```python
from everyo import Linear
from everyo.inference import CPUInferenceRuntime
from everyo.serving import CPUModelServer

runtime = CPUInferenceRuntime(Linear(2, 2, seed=0))
with CPUModelServer(runtime, api_token="a-development-only-token") as server:
    print(f"Local server: http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
```

```bash
curl -X POST http://127.0.0.1:PORT/v1/predict \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer a-development-only-token' \
  -d '{"inputs":[[1,2]]}'
```

**Boundaries:** This is a single-process developer service, not an Internet-ready
deployment. The bearer token is plaintext over HTTP; use TLS at a trusted proxy
for any network deployment. Add request timeouts, quotas, proxy hardening,
service isolation, a persistent monitoring backend, and an independent security
assessment before production exposure. `.evo` loading and user-supplied models
must also be controlled at a separate trusted ingestion boundary.
