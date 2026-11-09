"""Offline Phase 10 demo: train, track, save, restore, benchmark and predict."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from everyo import Adam, Linear, tensor
from everyo.benchmarking_phase10 import benchmark_cpu
from everyo.checkpointing import load_training_checkpoint, save_training_checkpoint
from everyo.evaluation import assert_no_leakage, classification_metrics, deterministic_split
from everyo.experiment_tracking import ExperimentTracker
from everyo.inference import CPUInferenceRuntime


def main() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=(60, 2))
    labels = (x[:, 0] + x[:, 1] > 0).astype(int)
    indices = deterministic_split(len(x), seed=2)
    assert_no_leakage(indices)
    model = Linear(2, 2, seed=1)
    optimizer = Adam(model.parameters(), lr=0.03)
    from everyo.nn.losses import CrossEntropyLoss

    loss_fn = CrossEntropyLoss()
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        tracker = ExperimentTracker(folder / "runs.jsonl", run_id="cpu-phase10")
        for epoch in range(15):
            optimizer.zero_grad()
            loss = loss_fn(model(tensor(x[indices["train"]])), labels[indices["train"]])
            loss.backward()
            optimizer.step()
            tracker.log(epoch, {"train_loss": float(loss.item())}, config={"seed": 10})
        checkpoint = folder / "model.evo"
        save_training_checkpoint(model, optimizer, checkpoint, rng=rng)
        load_training_checkpoint(checkpoint, optimizer, model=model, rng=rng)
        runtime = CPUInferenceRuntime(model)
        scores = runtime.predict_batched(x[indices["test"]])
        print("Held-out metrics:", classification_metrics(scores, labels[indices["test"]]))
        print("Benchmark:", benchmark_cpu(lambda: runtime.predict(x[:2]), repeats=3))
        print("Logged epochs:", len(tracker.read()))


if __name__ == "__main__":
    main()
