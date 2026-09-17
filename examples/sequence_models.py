"""Recurrent networks and a transformer on a long-range memory task.

The task is deliberately unfair to shortcuts: the label is the *first* token of
a long sequence and everything after it is noise drawn from a disjoint part of
the vocabulary. A model can only score above chance by carrying information
across the whole span.

Run with:
    python examples/sequence_models.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import everyo as eo
from everyo.datasets import make_recall_task

OUTPUT_DIR = Path("artifacts")
VOCAB, CLASSES, LENGTH, WIDTH = 16, 4, 24, 32


class RecurrentClassifier(eo.Module):
    """Embedding -> recurrent core -> classifier head."""

    def __init__(self, core: eo.Module) -> None:
        super().__init__()
        self.embedding = eo.Embedding(VOCAB, WIDTH, seed=0)
        self.core = core
        self.head = eo.Linear(WIDTH, CLASSES, seed=1)

    def forward(self, ids):
        """Classify a batch of token id sequences."""
        return self.head(self.core(self.embedding(ids)))


class TransformerClassifier(eo.Module):
    """Embedding -> positional encoding -> transformer encoder -> mean -> head."""

    def __init__(self, num_layers: int = 1) -> None:
        super().__init__()
        self.embedding = eo.Embedding(VOCAB, WIDTH, seed=0)
        self.positional = eo.PositionalEncoding(WIDTH)
        self.encoder = eo.TransformerEncoder(
            WIDTH, num_heads=4, num_layers=num_layers, dropout=0.0, seed=1
        )
        self.head = eo.Linear(WIDTH, CLASSES, seed=2)

    def forward(self, ids):
        """Classify a batch of token id sequences."""
        hidden = self.encoder(self.positional(self.embedding(ids)))
        return self.head(eo.mean(hidden, axis=1))


def train(name: str, model: eo.Module, data, *, epochs: int = 12):
    """Train one model and report its held-out accuracy."""
    x_train, x_test, y_train, y_test = data
    trainer = eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.01),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )
    history = trainer.fit(
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=64, shuffle=True, seed=0),
        epochs=epochs,
        verbose=False,
    )
    accuracy = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128))[
        "accuracy"
    ]
    print(f"   {name:<26} accuracy {accuracy:.3f}   params {model.num_parameters():>7,}")
    return trainer, history, accuracy


def plot_attention(model: TransformerClassifier, sequence: np.ndarray, path: Path) -> None:
    """Draw each head's attention weights for a single sequence.

    Row ``i`` shows where position ``i`` sends its attention. Heads typically
    specialise into different, sharply peaked patterns rather than all learning
    the same one — which is the reason for having several.
    """
    from everyo.visualization._backend import create_figure

    with eo.no_grad():
        model(sequence[None, :])
    weights = model.encoder.blocks[0].attention.last_attention_weights.numpy()[0]

    heads = weights.shape[0]
    figure, axes = create_figure(lambda plt: plt.subplots(1, heads, figsize=(3.1 * heads, 3.3)))
    for index, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(weights[index], cmap="viridis", vmin=0)
        ax.set_title(f"head {index}", fontsize=10)
        ax.set_xlabel("attends to")
        if index == 0:
            ax.set_ylabel("position")
    figure.suptitle(
        "Attention weights, one panel per head — each head has learned its own "
        "sharply peaked routing pattern",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    """Compare recurrent and attention-based models on long-range recall."""
    print(f"1. Long-range recall: label = first of {LENGTH} tokens, chance = {1 / CLASSES:.2f}")
    sequences, labels = make_recall_task(
        n_samples=2000, length=LENGTH, num_classes=CLASSES, vocab_size=VOCAB, seed=0
    )
    data = eo.train_test_split(sequences, labels, test_size=0.2, seed=0)
    print(f"   train {data[0].shape}   test {data[1].shape}")

    print("\n2. Recurrent models")
    train("RNN", RecurrentClassifier(eo.RNN(WIDTH, WIDTH, return_sequences=False, seed=2)), data)
    train("GRU", RecurrentClassifier(eo.GRU(WIDTH, WIDTH, return_sequences=False, seed=2)), data)
    train("LSTM", RecurrentClassifier(eo.LSTM(WIDTH, WIDTH, return_sequences=False, seed=2)), data)

    print("\n3. Transformer")
    transformer = TransformerClassifier()
    _, history, _ = train("Transformer (1 block)", transformer, data)

    print("\n4. Why the LSTM can remember")
    length = 40
    probe = eo.tensor(
        np.random.default_rng(0).normal(size=(1, length, 8)).astype(np.float32),
        requires_grad=True,
    )
    eo.sum(eo.LSTM(8, 8, return_sequences=False, seed=0)(probe)).backward()
    lstm_gradient = float(np.abs(probe.grad[:, 0, :]).sum())

    probe2 = eo.tensor(probe.numpy(), requires_grad=True)
    eo.sum(eo.RNN(8, 8, return_sequences=False, seed=0)(probe2)).backward()
    rnn_gradient = float(np.abs(probe2.grad[:, 0, :]).sum())

    print(f"   gradient reaching t=0 across {length} steps:")
    print(f"     RNN  {rnn_gradient:.2e}")
    print(
        f"     LSTM {lstm_gradient:.2e}   ({lstm_gradient / max(rnn_gradient, 1e-30):,.0f}x larger)"
    )
    print("   The LSTM's cell state is updated by addition, so its gradient is not")
    print("   multiplied down at every step the way a plain RNN's is.")

    print("\n5. Charts")
    OUTPUT_DIR.mkdir(exist_ok=True)
    eo.plot_loss(history, save_path=OUTPUT_DIR / "sequence_loss.png")
    plot_attention(transformer, data[1][0], OUTPUT_DIR / "attention_weights.png")
    print(f"   written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
