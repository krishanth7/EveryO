"""Benchmark a full training run: EveryO against a Keras reference.

The two stacks train the same architecture on the same locally generated data
for the same number of epochs. TensorFlow is optional; when it is missing only
the EveryO timing is reported.

Run with:
    python benchmarks/benchmark_training.py --epochs 10
"""

from __future__ import annotations

import argparse
import time

from _common import add_common_arguments, export, print_environment, print_table

import everyo as eo
from everyo.backends import tensorflow_backend as tfb
from everyo.datasets import load_digits


def build_model(input_size: int, hidden: int, classes: int) -> eo.Sequential:
    """The architecture used by both stacks."""
    return eo.Sequential(
        eo.Linear(input_size, hidden, seed=0),
        eo.ReLU(),
        eo.Linear(hidden, classes, seed=1),
    )


def benchmark_everyo(features, labels, *, epochs: int, batch_size: int, hidden: int) -> dict:
    """Time an EveryO training run and record its final metrics."""
    model = build_model(features.shape[1], hidden, 10)
    trainer = eo.Trainer(
        model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
    )
    loader = eo.DataLoader(
        eo.ArrayDataset(features, labels), batch_size=batch_size, shuffle=True, seed=0
    )

    started = time.perf_counter()
    history = trainer.fit(loader, epochs=epochs, verbose=False)
    elapsed = time.perf_counter() - started

    return {
        "benchmark": "training",
        "backend": "everyo-cpu",
        "epochs": epochs,
        "seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs,
        "final_loss": history["loss"][-1],
        "final_accuracy": history["accuracy"][-1],
    }


def benchmark_tensorflow(
    features, labels, *, epochs: int, batch_size: int, hidden: int
) -> dict | None:
    """Time an equivalent Keras training run, or return ``None`` if unavailable."""
    if not tfb.is_available():
        return None

    model = build_model(features.shape[1], hidden, 10)
    started = time.perf_counter()
    history = tfb.train_reference_model(
        model, features, labels, epochs=epochs, batch_size=batch_size, learning_rate=0.01
    )
    elapsed = time.perf_counter() - started

    return {
        "benchmark": "training",
        "backend": "tensorflow-gpu" if tfb.gpu_available() else "tensorflow-cpu",
        "epochs": epochs,
        "seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs,
        "final_loss": history["loss"][-1],
        "final_accuracy": history.get("accuracy", [float("nan")])[-1],
    }


def main() -> None:
    """Run both training benchmarks and print a comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer width.")
    parser.add_argument(
        "--samples-per-class", type=int, default=300, help="Digit samples generated per class."
    )
    add_common_arguments(parser)
    args = parser.parse_args()

    print_environment()
    features, labels = load_digits(samples_per_class=args.samples_per_class, seed=0)
    print(
        f"Training benchmark: {len(features)} samples, {args.epochs} epochs, "
        f"batch size {args.batch_size}, hidden width {args.hidden}\n"
    )

    results = [
        benchmark_everyo(
            features, labels, epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden
        )
    ]
    reference = benchmark_tensorflow(
        features, labels, epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden
    )
    if reference is not None:
        results.append(reference)
    else:
        print(f"TensorFlow reference skipped: {tfb.unavailable_reason()}\n")

    print_table(
        results,
        ["backend", "epochs", "seconds", "seconds_per_epoch", "final_loss", "final_accuracy"],
    )
    export(results, args.output)

    print(
        "\nNote: EveryO is a readable reference implementation, not a tuned "
        "production runtime. A mature framework being faster here is expected."
    )


if __name__ == "__main__":
    main()
