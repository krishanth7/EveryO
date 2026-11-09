# Phase 5 — Advanced Training Intelligence

Phase 5 adds CPU-first training controls and safe, pickle-free checkpoints.

| Stage | Delivered |
| --- | --- |
| 5.1 | Warmup/cosine and validation-metric plateau learning-rate schedulers |
| 5.2 | AdamW (decoupled decay) and RMSprop (centered/momentum options) |
| 5.3 | Sample-weighted gradient accumulation and global-norm clipping in `Trainer` |
| 5.4 | Seeded discrete random search and deterministic Cartesian grid search |
| 5.5 | Median-based pruning against completed peer trials at a matching step |
| 5.6 | Atomic, pickle-free NumPy checkpoints for model buffers and optimizer moments |
| 5.7 | Non-finite training-loss and gradient checks, optional excessive-norm failure |

```python
import everyo as eo

model = eo.Sequential(eo.Linear(4, 8), eo.ReLU(), eo.Linear(8, 1))
optimizer = eo.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
trainer = eo.Trainer(
    model, optimizer, eo.MSELoss(), accumulation_steps=4, gradient_clip=1.0, detect_anomalies=True
)
scheduler = eo.WarmupCosineScheduler(optimizer, total_steps=100, warmup_steps=10)
# Call scheduler.step() after each optimizer update in your custom loop.
eo.save_training_checkpoint(model, optimizer, "run.npz", epoch=3)
position = eo.load_training_checkpoint(model, optimizer, "run.npz")
print(position["epoch"])
```

For reproducible tuning, enumerate `eo.grid_search(...)` or
`eo.random_search(..., seed=...)` and compare metrics via the Phase 4
`ExperimentRegistry` using the same dataset hash and seed.

## Explicit limitations

- The checkpoint includes model parameter/buffer arrays and optimizer moments,
  not the current dataset iterator, Python/NumPy RNG states, callback state,
  scheduler state, or pending partial accumulation window. Resume at an
  **optimizer-step/epoch boundary** and preserve any external state separately.
- The checkpoint restore validates model names, array shape and dtype,
  optimizer type and tensor layout. It does not reconstruct arbitrary model
  classes or optimizers and must only be loaded from trusted storage.
- Scheduling uses explicit `.step()` calls; the standard `Trainer.fit` does not
  automatically call a scheduler or coordinate pruning/search.
- Tuning functions generate deterministic candidate configurations; they do not
  launch a distributed search service or track trials automatically.
- CPU-only tests verify numerical behavior and a subsequent optimizer step
  after reload. They do not certify bit-for-bit identity of a complete training
  job with arbitrary dropout, data-loader state, or external RNG use.
