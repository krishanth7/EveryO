# Phases 15–20 — CPU-first engineering foundations

These additions extend EveryO without GPU dependencies or changes to the
existing Phase 0–14 APIs. They are **engineering foundations** and are *not*
independent production certification. A release may only be approved after
external qualification provides all mandatory evidence.

| Phase | Scope | Public surface | Acceptance |
| --- | --- | --- | --- |
| 15 — CPU optimization | bounded microbatches, validated concatenation, output parity | `everyo.cpu_optimization` | parity and resource limits |
| 16 — Distributed foundations | reuse existing CPU worker sharding, weighted evaluation | `everyo.distributed_foundations` | deterministic partitions, no sample loss |
| 17 — Multimodal retrieval | exact cosine search over precomputed embeddings | `everyo.multimodal_retrieval` | fixed-score ranking and shape checks |
| 18 — Agent tool integration | allowlisted tool registry, JSON input/output limits, call budget | `everyo.agent_tools` | unknown and malformed calls rejected |
| 19 — Deployment security policy | fail-closed network config and artifact verification | `everyo.deployment_policy` | production security flags required |
| 20 — Qualification gate | explicit evidence for tests, held-out metrics, security, load and integrity | `everyo.release_qualification` | missing evidence rejects release |

## Run the verification suite

```bash
python -m pytest -q tests/test_phase15_20_foundations.py
python -m pytest -q
ruff check .
ruff format --check .
```

## Critical boundaries and unfinished production work

- **Phase 15:** Microbatching bounds work but does not yet prove lower latency
  or memory on representative hardware. Compare benchmark distributions across
  multiple CPUs and real models before selecting a production batch policy.
- **Phase 16:** `evaluate_shards` is sequential reference aggregation, not a
  distributed scheduler or networked worker service. EveryO already has a
  single-host multiprocessing training subsystem. True multi-node deployment
  requires rendezvous, network failure recovery, secrets, and cluster tests.
- **Phase 17:** The index is an exact in-memory search over caller-supplied
  embeddings. It contains no pretrained vision/speech encoder or approximate
  vector-search service, durable storage, access control or tenant isolation.
- **Phase 18:** Trusted developers register Python functions. The registry
  validates tool names, schemas and budgets but does not sandbox the functions
  themselves. Do not register shell, network, filesystem or privileged tools
  without a separately enforced sandbox and explicit authorization.
- **Phase 19:** The policy checks configuration flags. It does **not** enable
  TLS, implement authentication, provide an audit backend or verify an actual
  gateway. Deployments must independently prove all such infrastructure works.
- **Phase 20:** The decision API receives claimed evidence. A caller could lie;
  production acceptance requires independent, signed CI and security evidence.
  Required gates: unit tests, integration tests, held-out evaluation, security
  review, load tests and artifact integrity. Missing gates always fail.

Passing repository CI demonstrates software regression checks, **not** a
production certification, enterprise SLA, or competitive AI performance.
