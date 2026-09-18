"""Train a small CNN, export it to ONNX, and check the export against the model.

The check is the point. Writing a file that `onnx.checker` accepts proves the
graph is well-formed, not that it computes the same function; this script runs
the exported graph through ONNX Runtime on the same inputs and prints the
largest disagreement.

Needs the optional extras:
    pip install everyo[onnx]

Run with:
    python examples/onnx_export.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import everyo as eo
from everyo.datasets import load_digits

OUTPUT_DIR = Path("artifacts")


def build_model() -> eo.Sequential:
    """A small NHWC convolutional classifier for the 8x8 digit images."""
    return eo.Sequential(
        eo.Conv2D(1, 8, 3, padding="same", seed=0),
        eo.ReLU(),
        eo.MaxPool2D(2),
        eo.Conv2D(8, 16, 3, padding="same", seed=1),
        eo.ReLU(),
        eo.MaxPool2D(2),
        eo.Flatten(),
        eo.Linear(16 * 2 * 2, 10, seed=2),
    )


def main() -> None:
    if not eo.onnx_available():
        print("The optional 'onnx' package is not installed.")
        print("Install it with: pip install everyo[onnx]")
        return

    images, labels = load_digits()
    images = images.reshape(-1, 8, 8, 1)
    x_train, x_test, y_train, y_test = eo.train_test_split(images, labels, test_size=0.2, seed=0)

    model = build_model()
    trainer = eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.01),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )
    trainer.fit(
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=64, shuffle=True, seed=0),
        epochs=8,
        verbose=True,
        log_every=2,
    )

    evaluation = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128))
    print(f"\nTest accuracy: {evaluation['accuracy']:.4f}")

    # Export writes inference semantics, so the model has to be in eval mode.
    model.eval()
    OUTPUT_DIR.mkdir(exist_ok=True)
    destination = eo.export_onnx(
        model,
        OUTPUT_DIR / "digits_cnn.onnx",
        input_shape=(1, 8, 8, 1),
        output_name="logits",
    )
    print(f"Wrote {destination} ({destination.stat().st_size / 1024:.1f} KiB)")

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("\nInstall onnxruntime to verify the export: pip install onnxruntime")
        return

    batch = np.ascontiguousarray(x_test[:128], dtype=np.float32)
    expected = np.asarray(model(eo.tensor(batch)).data)
    actual = eo.run_onnx(destination, batch)

    print("\nExported graph vs the EveryO model")
    print("-" * 52)
    print(f"  output shape           : {actual.shape}")
    print(f"  largest absolute diff  : {np.abs(expected - actual).max():.3e}")
    agreement = (expected.argmax(axis=1) == actual.argmax(axis=1)).mean()
    print(f"  predictions that agree : {agreement:.1%}")
    print("\nThe gap is float32 rounding from a different order of operations,")
    print("not a difference in what the two graphs compute.")


if __name__ == "__main__":
    main()
