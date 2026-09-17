"""Multi-class classification on locally generated spirals.

Run with:
    python examples/multiclass_classification.py
"""

from __future__ import annotations

from pathlib import Path

import everyo as eo
from everyo.datasets import make_spirals

OUTPUT_DIR = Path("artifacts")


def main() -> None:
    """Train a three-class classifier and plot its decision regions."""
    features, labels = make_spirals(n_samples=900, classes=3, noise=0.08, seed=0)
    x_train, x_test, y_train, y_test = eo.stratified_split(features, labels, test_size=0.2, seed=0)

    model = eo.Sequential(
        eo.Linear(2, 64, seed=0),
        eo.Tanh(),
        eo.Linear(64, 64, seed=1),
        eo.Tanh(),
        eo.Linear(64, 3, seed=2),
    )
    trainer = eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.01),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )

    history = trainer.fit(
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=32, shuffle=True, seed=0),
        epochs=120,
        validation_loader=eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64),
        callbacks=[eo.EarlyStopping(monitor="val_loss", patience=20)],
        verbose=True,
        log_every=20,
    )

    results = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64))
    print(f"\nTest loss: {results['loss']:.4f}   Test accuracy: {results['accuracy']:.4f}")
    print(f"Trained for {history.epochs} epoch(s).")

    OUTPUT_DIR.mkdir(exist_ok=True)
    eo.plot_history(history, save_path=OUTPUT_DIR / "spirals_history.png")
    eo.plot_confusion_matrix(
        trainer.predict_classes(x_test),
        y_test,
        class_names=["arm 0", "arm 1", "arm 2"],
        save_path=OUTPUT_DIR / "spirals_confusion.png",
    )
    from everyo.visualization import plot_decision_boundary

    plot_decision_boundary(
        trainer.predict_classes,
        features,
        labels,
        title="Spiral decision regions",
        save_path=OUTPUT_DIR / "spirals_boundary.png",
    )
    print(f"Charts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
