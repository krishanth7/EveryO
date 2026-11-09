# Phase 4 — Data and Experiment Engineering

Phase 4 provides CPU-only building blocks for repeatable datasets and local
experiment bookkeeping. It requires no GPU, cloud account, or external database.

| Subphase | Implementation |
| --- | --- |
| 4.1 | `StreamingDataset` creates fresh epoch iterators without materializing datasets. |
| 4.2 | Bounded-buffer deterministic shuffling and RNG-aware transforms with epoch/skip resume. |
| 4.3 | `DatasetCache` stores numeric `.npy` arrays with SHA-256 integrity checks and optional mmap. |
| 4.4 | `ExperimentRegistry` maintains a local transactional SQLite registry. |
| 4.5 | JSON hyperparameter configurations and indexed step/metric logging. |
| 4.6 | `compare`, deterministic run manifests, and persisted epoch/skip progress. |
| 4.7 | Content-addressed, integrity-verified artifact snapshots (including `.evo` files). |

## Example

```python
import numpy as np
from everyo.data_engineering import DatasetCache, StreamingDataset
from everyo.experiments import ExperimentRegistry

cache = DatasetCache("./experiment-data")
data_hash = cache.put("train-v1", np.arange(30))
dataset = StreamingDataset(
    lambda: iter(cache.get("train-v1")),
    seed=7,
    shuffle_buffer=8,
    transform=lambda x, rng: float(x) + rng.normal(0, 0.01),
)
registry = ExperimentRegistry("./runs")
run = registry.start({"learning_rate": 0.01}, dataset_hash=data_hash, seed=7)
for i, batch in enumerate(dataset.batches(5, epoch=0)):
    registry.log_metric(run, i, "mean", float(np.mean(batch)))
    registry.update_progress(run, epoch=0, skip=min((i + 1) * 5, 30))
print(registry.manifest(run))
registry.finish(run)
```

To resume, reopen the same registry, read `get(run_id)["progress"]`,
verify the same dataset checksum and saved configuration, and pass `epoch`
and `skip` to `StreamingDataset`. Save model or optimizer checkpoints with
`add_artifact` and retrieve them with `artifact`.

## Scope and limitations

- A registry stores progress, but **does not automatically restore model weights,
  optimizer state, or external random generators**. Save those explicitly if an
  exact training continuation is required. The dataset's own transform RNG is
  replayed deterministically during skipped items.
- Streaming bounded-buffer shuffling is not a uniform global shuffle. The
  `source` must return a fresh, deterministic iterable each epoch.
- Cache writes are atomic per file, but the two-file data/manifest pair is not
  an atomic transaction. Use separate cache keys per writer and avoid concurrent
  modification of the same key. Verify source provenance separately.
- SQLite is intended for local trusted use; artifacts are immutable by checksum
  convention but not protected against a malicious user with filesystem access.
- All acceptance tests use CPUs and temporary directories. No GPU qualification
  or end-to-end external dataset reproducibility claim is made.
