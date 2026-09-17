"""Cross-check EveryO against TensorFlow.

TensorFlow is optional. When it is missing this example explains how to install
it and exits cleanly rather than failing.

Run with:
    python examples/tensorflow_backend.py
"""

from __future__ import annotations

import numpy as np

import everyo as eo
from everyo.backends import tensorflow_backend as tfb
from everyo.datasets import make_blobs


def compare_operations(rng: np.random.Generator) -> None:
    """Check a few EveryO kernels against TensorFlow's implementations."""
    print("=== Operation agreement (EveryO vs TensorFlow) ===")
    a = rng.normal(size=(64, 32)).astype(np.float32)
    b = rng.normal(size=(32, 16)).astype(np.float32)

    checks = {
        "matmul": (
            eo.matmul(eo.tensor(a), eo.tensor(b)).numpy(),
            tfb.matmul(a, b),
        ),
        "relu": (
            eo.relu(eo.tensor(a)).numpy(),
            tfb.relu(a),
        ),
    }
    for name, (ours, theirs) in checks.items():
        difference = float(np.abs(ours - theirs).max())
        print(f"  {name:<8} max absolute difference: {difference:.3e}")
        np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6)
    print("  All operations agree within float32 tolerance.")


def compare_training() -> None:
    """Train the same architecture with EveryO and with Keras."""
    print("\n=== Training comparison on the same data ===")
    features, labels = make_blobs(n_samples=600, n_features=4, centers=3, seed=0)

    model = eo.Sequential(
        eo.Linear(4, 32, seed=0),
        eo.ReLU(),
        eo.Linear(32, 3, seed=1),
    )
    trainer = eo.Trainer(
        model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
    )
    history = trainer.fit(
        eo.DataLoader(eo.ArrayDataset(features, labels), batch_size=32, shuffle=True, seed=0),
        epochs=20,
        verbose=False,
    )
    print(
        f"  EveryO      final loss: {history['loss'][-1]:.4f}  "
        f"accuracy: {history['accuracy'][-1]:.4f}"
    )

    reference = tfb.train_reference_model(
        model, features, labels, epochs=20, batch_size=32, learning_rate=0.01
    )
    print(
        f"  TensorFlow  final loss: {reference['loss'][-1]:.4f}  "
        f"accuracy: {reference['accuracy'][-1]:.4f}"
    )
    print("  (Weights are initialised differently, so small differences are expected.)")


def main() -> None:
    """Run the comparison when TensorFlow is available."""
    if not tfb.is_available():
        print("TensorFlow is not available in this environment.")
        print(f"Reason: {tfb.unavailable_reason()}")
        print("\nInstall it with:  pip install 'everyo[tensorflow]'")
        print("EveryO itself continues to work without it.")
        return

    print(f"TensorFlow {tfb.get_tensorflow().__version__} detected.")
    print(f"Devices: {', '.join(tfb.list_devices()) or 'none reported'}")
    print(f"GPU available to TensorFlow: {tfb.gpu_available()}\n")

    compare_operations(np.random.default_rng(0))
    compare_training()


if __name__ == "__main__":
    main()
