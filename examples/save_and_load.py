"""Save a trained model and load it back safely.

Run with:
    python examples/save_and_load.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import everyo as eo
from everyo.datasets import make_blobs
from everyo.serialization import inspect_archive


def main() -> None:
    """Train, save, inspect and reload a model."""
    features, labels = make_blobs(n_samples=400, n_features=4, centers=3, seed=0)

    model = eo.Sequential(
        eo.Linear(4, 16, seed=0),
        eo.ReLU(),
        eo.Linear(16, 3, seed=1),
    )
    trainer = eo.Trainer(
        model, eo.Adam(model.parameters(), lr=0.02), eo.CrossEntropyLoss(), metrics=["accuracy"]
    )
    loader = eo.DataLoader(eo.ArrayDataset(features, labels), batch_size=32, shuffle=True, seed=0)
    trainer.fit(loader, epochs=25, verbose=False)

    accuracy = trainer.evaluate(loader)["accuracy"]
    print(f"Trained model accuracy: {accuracy:.4f}")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "classifier.evo"

        print("\n=== Saving ===")
        eo.save(model, path, metadata={"accuracy": accuracy, "dataset": "make_blobs"})
        print(f"  wrote {path.name} ({path.stat().st_size} bytes)")

        print("\n=== Inspecting the archive without loading it ===")
        info = inspect_archive(path)
        print(f"  EveryO version:  {info['everyo_version']}")
        print(f"  architecture:    {info['class_name']}")
        print(f"  parameters:      {info['num_parameters']}")
        print(f"  metadata:        {info['metadata']}")
        print("  parameter shapes:")
        for name, description in info["parameters"].items():
            print(f"    {name:<12} {tuple(description['shape'])}  {description['dtype']}")

        print("\n=== Loading ===")
        restored = eo.load(path)
        print(f"  restored a {type(restored).__name__} with {restored.num_parameters()} parameters")

        before = trainer.predict(features[:10])
        restored_trainer = eo.Trainer(
            restored, eo.Adam(restored.parameters(), lr=0.02), eo.CrossEntropyLoss()
        )
        after = restored_trainer.predict(features[:10])
        print(f"  max prediction difference: {float(np.abs(before - after).max()):.2e}")

    print("\nThe .evo format is a ZIP archive holding JSON metadata and an")
    print("allow_pickle=False .npz of parameters, so loading never executes code.")


if __name__ == "__main__":
    main()
