"""Train a small convolutional network on the bundled digit dataset.

Convolution is what lets a model exploit the *structure* of an image: nearby
pixels belong together, and a feature worth detecting in one corner is worth
detecting everywhere. This example trains a CNN, compares it against a dense
network of similar size, and then shows what the first layer actually learned.

Run with:
    python examples/convolutional_network.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import everyo as eo
from everyo.datasets import load_digits

OUTPUT_DIR = Path("artifacts")


def build_cnn() -> eo.Sequential:
    """Two convolution blocks, each halving the resolution, then a classifier."""
    return eo.Sequential(
        eo.Conv2D(1, 8, 3, padding="same", seed=0),  # 8x8x1  -> 8x8x8
        eo.ReLU(),
        eo.MaxPool2D(2),  #        -> 4x4x8
        eo.Conv2D(8, 16, 3, padding="same", seed=1),  #        -> 4x4x16
        eo.ReLU(),
        eo.MaxPool2D(2),  #        -> 2x2x16
        eo.Flatten(),  #        -> 64
        eo.Linear(2 * 2 * 16, 10, seed=2),
    )


def build_dense() -> eo.Sequential:
    """A fully connected network of comparable size, for contrast."""
    return eo.Sequential(
        eo.Linear(64, 32, seed=0),
        eo.ReLU(),
        eo.Linear(32, 10, seed=1),
    )


def train(model: eo.Sequential, x_train, y_train, x_test, y_test, *, epochs: int, verbose: bool):
    """Train ``model`` and return ``(trainer, history, test_accuracy)``."""
    trainer = eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.01),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )
    history = trainer.fit(
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=64, shuffle=True, seed=0),
        epochs=epochs,
        validation_loader=eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128),
        verbose=verbose,
        log_every=3,
    )
    accuracy = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128))[
        "accuracy"
    ]
    return trainer, history, accuracy


def plot_filters(model: eo.Sequential, path: Path) -> None:
    """Draw the first convolution layer's learned 3x3 kernels."""
    from everyo.visualization._backend import create_figure

    kernels = model[0].weight.numpy()  # (3, 3, 1, 8)
    count = kernels.shape[-1]
    figure, axes = create_figure(lambda plt: plt.subplots(1, count, figsize=(1.35 * count, 1.9)))
    for index, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(kernels[:, :, 0, index], cmap="RdBu_r")
        ax.set_title(f"filter {index}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    figure.suptitle("What the first convolution layer learned", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")


def plot_feature_maps(model: eo.Sequential, image: np.ndarray, path: Path) -> None:
    """Show one digit and the eight feature maps the first layer produces."""
    from everyo.visualization._backend import create_figure

    with eo.no_grad():
        activated = eo.relu(model[0](eo.tensor(image[None, ...]))).numpy()[0]

    figure, axes = create_figure(lambda plt: plt.subplots(1, 9, figsize=(13, 1.9)))
    axes[0].imshow(image[:, :, 0], cmap="gray_r")
    axes[0].set_title("input", fontsize=9)
    for index in range(activated.shape[-1]):
        axes[index + 1].imshow(activated[:, :, index], cmap="viridis")
        axes[index + 1].set_title(f"map {index}", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    figure.suptitle("The same digit, seen through each learned filter", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    """Train the CNN, compare it with a dense baseline and visualise the filters."""
    print("1. Loading digits as 8x8 images (not flattened vectors)")
    images, labels = load_digits(samples_per_class=200, noise=0.22, seed=0, flatten=False)
    images = images[..., None]  # NHWC: add the single channel axis
    x_train, x_test, y_train, y_test = eo.stratified_split(images, labels, test_size=0.2, seed=0)
    print(f"   train {x_train.shape}   test {x_test.shape}")

    print("\n2. Convolutional network")
    cnn = build_cnn()
    print(cnn.summary())
    cnn_trainer, history, cnn_accuracy = train(
        cnn, x_train, y_train, x_test, y_test, epochs=12, verbose=True
    )

    print("\n3. Dense baseline on the same data, flattened")
    dense = build_dense()
    _, _, dense_accuracy = train(
        dense,
        x_train.reshape(len(x_train), -1),
        y_train,
        x_test.reshape(len(x_test), -1),
        y_test,
        epochs=12,
        verbose=False,
    )

    print("\n4. Result")
    print(f"   CNN   : {cnn_accuracy:.4f} with {cnn.num_parameters():,} parameters")
    print(f"   Dense : {dense_accuracy:.4f} with {dense.num_parameters():,} parameters")
    print("   Convolution shares its weights across the image, so it can afford")
    print("   more feature detectors for the same parameter budget.")

    print("\n5. Charts")
    OUTPUT_DIR.mkdir(exist_ok=True)
    eo.plot_history(history, save_path=OUTPUT_DIR / "cnn_history.png")
    eo.plot_confusion_matrix(
        cnn_trainer.predict_classes(x_test),
        y_test,
        title=f"CNN — {cnn_accuracy:.1%} on held-out digits",
        save_path=OUTPUT_DIR / "cnn_confusion.png",
    )
    plot_filters(cnn, OUTPUT_DIR / "cnn_filters.png")
    plot_feature_maps(cnn, x_test[0], OUTPUT_DIR / "cnn_feature_maps.png")
    print(f"   written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
