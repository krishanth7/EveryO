"""Data-parallel training on several CPU cores, checked against one process.

Each worker builds the same model from the same seeds, trains on its own slice
of the data, and averages gradients with every other worker before each step.
Because averaging gradients over disjoint shards is the same arithmetic as one
large batch, the result should match single-process training -- and this script
checks that rather than asserting it.

Set OMP_NUM_THREADS=1 before running, or NumPy's BLAS will start a full thread
pool inside every worker and they will fight over the same cores:

    OMP_NUM_THREADS=1 python examples/distributed_training.py
"""

from __future__ import annotations

import time

import numpy as np

import everyo as eo
from everyo.datasets import make_regression
from everyo.distributed import available_workers, average_gradients, shard_indices, spawn

STEPS = 60
LEARNING_RATE = 0.05


def build_model() -> eo.Sequential:
    """Fixed seeds, so every worker starts from bit-identical weights."""
    return eo.Sequential(
        eo.Linear(8, 64, seed=0),
        eo.Tanh(),
        eo.Linear(64, 64, seed=1),
        eo.Tanh(),
        eo.Linear(64, 1, seed=2),
    )


def worker(group, inputs: np.ndarray, targets: np.ndarray) -> dict:
    """Train on this rank's shard, averaging gradients every step.

    Defined at module level so it survives the "spawn" start method, which
    re-imports rather than forking.
    """
    model = build_model()
    optimizer = eo.SGD(model.parameters(), lr=LEARNING_RATE)

    rows = shard_indices(len(inputs), group.rank, group.world_size)
    xs, ys = eo.tensor(inputs[rows]), eo.tensor(targets[rows])

    group.barrier()
    started = time.perf_counter()
    for _ in range(STEPS):
        optimizer.zero_grad()
        residual = model(xs) - ys
        eo.mean(residual * residual).backward()
        average_gradients(model.parameters(), group)
        optimizer.step()
    group.barrier()
    elapsed = time.perf_counter() - started

    with eo.no_grad():
        residual = model(eo.tensor(inputs)) - eo.tensor(targets)
        full_loss = float(eo.mean(residual * residual).item())

    return {
        "rank": group.rank,
        "samples": len(rows),
        "seconds": elapsed,
        "loss_on_full_dataset": full_loss,
        "parameters": [np.asarray(p.data) for p in model.parameters()],
    }


def train_in_one_process(inputs: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
    """The reference: the whole batch, one process, no collectives."""
    model = build_model()
    optimizer = eo.SGD(model.parameters(), lr=LEARNING_RATE)
    xs, ys = eo.tensor(inputs), eo.tensor(targets)
    for _ in range(STEPS):
        optimizer.zero_grad()
        residual = model(xs) - ys
        eo.mean(residual * residual).backward()
        optimizer.step()
    return [np.asarray(p.data) for p in model.parameters()]


def main() -> None:
    inputs, targets = make_regression(n_samples=4096, n_features=8, noise=0.1, seed=0)
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)

    cores = available_workers()
    print(f"This machine reports {cores} usable core(s).")
    reference = train_in_one_process(inputs, targets)

    world_sizes = sorted({1, 2, cores} - {0})
    print()
    print(
        f"{'workers':>8}  {'rows/worker':>12}  {'seconds':>9}  {'max drift':>11}  {'ranks agree':>12}"
    )
    print("-" * 62)

    for world_size in world_sizes:
        results = spawn(worker, world_size, args=(inputs, targets))
        slowest = max(result["seconds"] for result in results)

        drift = max(
            float(np.abs(result["parameters"][index] - reference[index]).max())
            for result in results
            for index in range(len(reference))
        )
        identical = all(
            np.array_equal(result["parameters"][index], results[0]["parameters"][index])
            for result in results
            for index in range(len(reference))
        )
        print(
            f"{world_size:>8}  {results[0]['samples']:>12}  {slowest:>9.2f}  "
            f"{drift:>11.2e}  {str(identical):>12}"
        )

    print()
    print("'max drift' is the largest weight difference from single-process training;")
    print("'ranks agree' says whether every worker ended with bit-identical weights.")
    print()
    print("If the wall times do not improve, check OMP_NUM_THREADS=1 is set.")


if __name__ == "__main__":
    main()
