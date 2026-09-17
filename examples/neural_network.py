"""The complete EveryO pipeline on the bundled digit dataset.

This is the reference end-to-end demo:

    dataset -> preprocessing -> DataLoader -> network -> training ->
    evaluation -> inference -> visualization -> save/load

The dataset is generated locally from 8x8 bitmap glyphs, so this example needs
no network access.

Run with:
    python examples/neural_network.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import everyo as eo
from everyo.datasets import load_digits

OUTPUT_DIR = Path("artifacts")
MODEL_PATH = OUTPUT_DIR / "digits.evo"


def build_model(input_size: int, num_classes: int) -> eo.Sequential:
    """A small fully connected classifier with dropout regularisation."""
    return eo.Sequential(
        eo.Linear(input_size, 128, seed=0),
        eo.ReLU(),
        eo.Dropout(0.2, seed=0),
        eo.Linear(128, 64, seed=1),
        eo.ReLU(),
        eo.Linear(64, num_classes, seed=2),
    )


def main() -> None:
    """Run the full pipeline and report test accuracy."""
    print("1. Loading the locally generated digit dataset")
    features, labels = load_digits(samples_per_class=300, noise=0.18, seed=0)
    print(
        f"   {features.shape[0]} samples, {features.shape[1]} features, "
        f"{len(np.unique(labels))} classes"
    )

    print("2. Splitting and scaling")
    x_train, x_test, y_train, y_test = eo.stratified_split(features, labels, test_size=0.2, seed=0)
    scaler = eo.StandardScaler().fit(x_train)  # statistics come from training data only
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

    print("3. Building loaders")
    train_loader = eo.DataLoader(
        eo.ArrayDataset(x_train, y_train), batch_size=64, shuffle=True, seed=0
    )
    test_loader = eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128)

    print("4. Building the model")
    model = build_model(features.shape[1], 10)
    print(model.summary())

    print("5. Training")
    trainer = eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.005),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    history = trainer.fit(
        train_loader,
        epochs=40,
        validation_loader=test_loader,
        callbacks=[
            eo.EarlyStopping(monitor="val_loss", patience=8),
            eo.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", mode="max"),
        ],
        verbose=True,
        log_every=5,
    )

    print("\n6. Evaluating")
    results = trainer.evaluate(test_loader)
    print(f"   Test loss: {results['loss']:.4f}   Test accuracy: {results['accuracy']:.4f}")

    print("\n7. Inference on five held-out samples")
    predictions = trainer.predict_classes(x_test[:5])
    print(f"   predicted: {predictions.tolist()}")
    print(f"   actual:    {y_test[:5].tolist()}")

    print("\n8. Charts")
    eo.plot_history(history, save_path=OUTPUT_DIR / "digits_history.png")
    eo.plot_confusion_matrix(
        trainer.predict_classes(x_test),
        y_test,
        title="Digit classification",
        save_path=OUTPUT_DIR / "digits_confusion.png",
    )

    print("\n9. Save and reload")
    path = eo.save(model, MODEL_PATH, metadata={"test_accuracy": results["accuracy"]})
    restored = eo.load(path)
    restored_trainer = eo.Trainer(
        restored,
        eo.Adam(restored.parameters(), lr=0.005),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )
    reloaded_accuracy = restored_trainer.evaluate(test_loader)["accuracy"]
    print(f"   Reloaded model accuracy: {reloaded_accuracy:.4f} (must match the value above)")
    print(f"\nArtifacts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
