# EveryO — CPU production qualification evidence (2026-11-09)

**Decision: FAIL / NOT CERTIFIED for unrestricted production deployment.**
The CPU engineering baseline passes local regression checks. That does not
establish deployment security, operational SLOs, external validation, or hardware
coverage. No evidence record should mark an untested gate as passing.

## Reproducible local verification

Environment: Linux x86-64, Python 3.12.14, NumPy 2.5.4, CPU only.

| Gate | Evidence | Decision |
| --- | --- | --- |
| CPU regression/integration | `python -m pytest -q --disable-warnings -ra` exited 0; optional CUDA, TensorFlow and ONNX tests skipped | PASS for executed tests; incomplete optional coverage |
| Source quality | `ruff check .`, `ruff format --check .` and `python -m compileall -q everyo` exited 0 | PASS |
| Archive security | Adversarial tests reject duplicate manifests, oversized decompressed manifests and oversized expanded nested NumPy archives; ordinary round-trip retained | PASS for these threat cases only |
| Security review | Source scan for pickle, process execution and unsafe dynamic evaluation completed; no actual pentest, dependency vulnerability scan or independent security assessment | FAIL / insufficient evidence |
| CPU microbenchmark | `OPENBLAS_NUM_THREADS=1 python -m benchmarks.benchmark_cpu_quality --size 128 --runs 20 --warmups 5`: median 0.5802075 ms, max 1.82618 ms, reproducibility seed 42 | PASS as a microbenchmark; no load/SLO claim |
| Integrated CPU example | `PYTHONPATH=. python examples/cpu_production_pipeline.py` exited 0; 9 held-out toy samples, 0.8889 accuracy, 15 epochs | PASS demo; FAIL representative held-out qualification |
| Sustained authenticated service load | No independently measured concurrent service workload, p95/p99 under failure, or long-running soak test | FAIL / missing |
| Deployability | Reverse proxy TLS, secret rotation, monitoring, alerts, backups, incident response, and recovery were not exercised against a deployed target | FAIL / missing |
| Artifact provenance | Archive size controls added; signed provenance and immutable artifact attestation not established | FAIL / missing |

## Phase-by-phase audit scope and confidence

| Phases | Inspected implementation area | Qualification status |
| --- | --- | --- |
| 0–1 | package/bootstrap, tensor/autograd, neural layers, training, serialization | CPU regression PASS; model artifact ingestion hardened |
| 2–3 | derivatives/gradient checks and CPU engine | CPU regression PASS; performance portability unqualified |
| 4–5 | datasets/experiments and advanced training | CPU regression PASS; production data governance unqualified |
| 6–7 | compression/inference and language-model primitives | CPU regression PASS; representative large-model fidelity unqualified |
| 8–9 | fine-tuning and multimodal foundations | CPU regression PASS; production-scale quality unqualified |
| 10–12 | checkpoint/benchmark, HTTP serving and governance | CPU regression PASS; checkpoint recovery and deployed threat model incomplete |
| 13–14 | monitoring and robustness | CPU regression PASS; real telemetry/alerts and adversarial campaigns incomplete |
| 15–17 | microbatch optimization, worker evaluation and embedding retrieval | CPU regression PASS; cross-machine load and search scale incomplete |
| 18–20 | tool allowlist, deployment policy, release gate | CPU regression PASS; tool functions are trusted (not sandboxed); release evidence remains insufficient |

**Interpretation:** these rows demonstrate implementation and tested CPU behavior,
not an independent line-by-line security certification of every phase. Optional
TensorFlow, ONNX and GPU compatibility must be tested in separate eligible
environments. The `qualify_release` gate must remain unapproved until each
required item is backed by independent evidence.

## Corrective changes from this audit

- `everyo/serialization/load.py`: bound decompressed outer ZIP members;
  reject duplicate manifest/parameter entries and oversized expanded NumPy
  arrays before materializing them with NumPy.
- `tests/test_production_archive_security.py`: exercise malicious archive
  cases and validate normal save/load API compatibility.
- `everyo/serving.py`: set a configurable bounded socket read deadline
  (`request_timeout_seconds`, default 5 seconds) and return HTTP 408 for
  incomplete request bodies that exceed it.
- `tests/test_production_operational_gates.py`: check concurrent CPU requests
  recover after an idle request timeout, reject held-out content leakage,
  detect artifact modification, and reject incomplete release evidence.

These additional tests improve local assurance. A 20-request concurrency
regression and tiny example do not replace sustained production load tests,
an independent security assessor, representative evaluation data, or externally
issued, verifiable release signatures. Those gates remain **FAIL / MISSING**.

## Remaining go-live blockers

1. Independent dependency/SBOM security review and penetration test of the
   actual deployment with its edge gateway and authenticated API.
2. Representative held-out datasets with approved task-specific acceptance
   thresholds, drift checks and independent evaluation sign-off.
3. Load/soak, timeout, concurrency, fault-injection and recovery tests using
   realistic model/input sizes and SLO thresholds.
4. Signed, provenance-checked release artifacts, rollback rehearsal, on-call
   ownership, alert routing and operational runbooks.
5. Optional hardware/interop validation for the declared support matrix.

After publication, record the *new* GitHub main SHA and Actions run URL here
with the separate CI verification. Do not infer CI success from local tests.
