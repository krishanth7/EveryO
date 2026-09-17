"""Binary classification of the two-moons problem.

Demonstrates that a linear model cannot separate the classes while a small
multi-layer network can.

Run with:
    python examples/binary_classification.py
"""

from __future__ import annotations

from pathlib import Path

import everyo as eo
from everyo.datasets import make_moons

OUTPUT_DIR = Path("artifacts")


def build_mlp() -> eo.Sequential:
    """Two hidden tanh layers, enough for a curved decision boundary."""
    return eo.Sequential(
        eo.Linear(2, 32, seed=0),
        eo.Tanh(),
        eo.Linear(32, 16, seed=1),
        eo.Tanh(),
        eo.Linear(16, 2, seed=2),
    )


def train(model: eo.Sequential, loaders, epochs: int = 60):
    """Train ``model`` and return ``(trainer, history)``."""
    trainer = eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.02),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )
    history = trainer.fit(
        loaders[0],
        epochs=epochs,
        validation_loader=loaders[1],
        verbose=True,
        log_every=15,
    )
    return trainer, history


def main() -> None:
    """Compare a linear model with an MLP on the moons dataset."""
    features, labels = make_moons(n_samples=600, noise=0.12, seed=0)
    x_train, x_test, y_train, y_test = eo.stratified_split(features, labels, test_size=0.2, seed=0)
    loaders = (
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=32, shuffle=True, seed=0),
        eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64),
    )

    print("=== Multi-layer network ===")
    mlp_trainer, history = train(build_mlp(), loaders)
    mlp_accuracy = mlp_trainer.evaluate(loaders[1])["accuracy"]

    print("\n=== Linear model (for comparison) ===")
    linear_trainer, _ = train(eo.Sequential(eo.Linear(2, 2, seed=0)), loaders, epochs=60)
    linear_accuracy = linear_trainer.evaluate(loaders[1])["accuracy"]

    print(f"\nMLP test accuracy:    {mlp_accuracy:.4f}")
    print(f"Linear test accuracy: {linear_accuracy:.4f}")
    print("The moons are not linearly separable, so the MLP should win clearly.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    eo.plot_history(history, save_path=OUTPUT_DIR / "moons_history.png")
    from everyo.visualization import plot_decision_boundary

    plot_decision_boundary(
        mlp_trainer.predict_classes,
        features,
        labels,
        title="MLP decision boundary (two moons)",
        save_path=OUTPUT_DIR / "moons_boundary.png",
    )
    print(f"Charts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
