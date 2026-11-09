# Phase 10 — CPU model evaluation and reliability

Phase 10 introduces an offline, NumPy-only qualification toolkit. It deliberately
does not certify EveryO as a production-scale deep learning framework.

| Milestone | API | Acceptance evidence |
| --- | --- | --- |
| 10.1 Evaluation | `classification_metrics`, `language_model_metrics` | Stable accuracy, cross-entropy, perplexity; invalid data rejected |
| 10.2 Dataset splits | `deterministic_split`, `assert_no_leakage` | Seeded disjoint splits and duplicate fingerprint detection |
| 10.3 CPU benchmark | `benchmark_cpu` | Measured latency, sample rate and Python-traced allocation peak |
| 10.4 Tracking | `ExperimentTracker` | Append-only JSONL with finite metrics and run IDs |
| 10.5 Training state | `save_training_checkpoint`, `load_training_checkpoint` in `everyo.checkpointing` | `.evo` weights, optimizer arrays, optional generator state and tokenizer metadata |
| 10.6 Validation | `validate_array`, `validate_predictions` | Reject bad shapes, nonnumeric/nonfinite inputs, oversized arrays |
| 10.7 CPU inference | `CPUInferenceRuntime` | Bounded batch execution with explicit limits |
| 10.8 Qualification | `tests/test_phase10_production.py`, CPU demo | Regression tests and full CI |

## CPU demo

```bash
python -m pytest -q tests/test_phase10_production.py
python examples/cpu_production_pipeline.py
```

## Important limitations

- `benchmark_cpu` measures the executing host. `tracemalloc` reports Python
  allocations, **not** total process RSS or native NumPy allocation peaks.
- `ExperimentTracker` is a local append-only journal, not a concurrent database
  or a secure hosted tracking service.
- Model and training companions must come from a trusted source. Neither this
  feature nor historical `.evo` loading is a hardened untrusted-file service;
  enforce file size and decompression limits at any external upload boundary.
- The optimizer companion is keyed to ordered parameters. Restore into an
  equivalent model and optimizer, with identical parameter ordering and settings.
- The generator state is optional and only covers the supplied NumPy
  `Generator`, not arbitrary global RNGs or third-party backends.
- `CPUInferenceRuntime` is synchronous and expects conventional array-input
  models; multimodal batch models require an application-specific adapter.
- Held-out metrics on toy data are examples, not generalization guarantees.
- This is a feature-level engineering foundation. Production deployment still
  requires independent threat modeling, service-layer access control, monitoring,
  real dataset evaluations, and resource-load testing.

The prior Phase 5 `everyo.advanced_training` training-checkpoint functions
remain unchanged; the richer Phase 10 functions are in `everyo.checkpointing`.
