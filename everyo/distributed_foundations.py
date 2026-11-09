"""Phase 16: deterministic multi-shard CPU evaluation aggregation.

This is single-host sharding; multi-node scheduling is out of scope.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from everyo.distributed import available_workers, shard_indices
from everyo.validation import validate_array


def plan_shards(size: int, *, workers: int, drop_last: bool = False) -> list[np.ndarray]:
    if workers <= 0 or workers > available_workers() * 4:
        raise ValueError("worker count must be positive and bounded")
    if size < workers:
        raise ValueError("each shard must have at least one row")
    shards = [shard_indices(size, rank, workers, drop_last=drop_last) for rank in range(workers)]
    if not drop_last and sorted(int(i) for shard in shards for i in shard) != list(range(size)):
        raise RuntimeError("shard planner dropped or duplicated rows")
    return shards


def aggregate_weighted_metrics(
    summaries: list[dict[str, float]], *, count_key: str = "samples"
) -> dict[str, float]:
    if not summaries or any(set(summary) != set(summaries[0]) for summary in summaries):
        raise ValueError("nonempty summaries with identical metric names required")
    counts = np.asarray([summary[count_key] for summary in summaries], dtype=float)
    if not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("sample counts must be finite and positive")
    result = {count_key: float(counts.sum())}
    for key in summaries[0]:
        if key == count_key:
            continue
        values = np.asarray([summary[key] for summary in summaries], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("metric values must be finite")
        result[key] = float(np.average(values, weights=counts))
    return result


def evaluate_shards(
    inputs: Any, evaluator: Callable[[np.ndarray], dict[str, float]], *, workers: int = 2
) -> dict[str, float]:
    """Deterministic sequential reference for sharded evaluation, not a speedup."""
    data = validate_array(inputs)
    shards = plan_shards(len(data), workers=workers)
    outputs = []
    for indices in shards:
        metrics = evaluator(data[indices])
        if "samples" in metrics and metrics["samples"] != len(indices):
            raise ValueError("evaluator returned incorrect sample count")
        outputs.append({**metrics, "samples": float(len(indices))})
    return aggregate_weighted_metrics(outputs)
