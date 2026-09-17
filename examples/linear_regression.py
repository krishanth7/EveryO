"""Fit a linear model to a synthetic regression problem.

Run with:
    python examples/linear_regression.py
"""

from __future__ import annotations

from pathlib import Path

import everyo as eo
from everyo.datasets import make_regression

OUTPUT_DIR = Path("artifacts")


def main() -> None:
    """Train a single Linear layer and report the recovered coefficients."""
    features, targets = make_regression(n_samples=400, n_features=1, noise=0.3, bias=2.0, seed=0)
    x_train, x_test, y_train, y_test = eo.train_test_split(features, targets, test_size=0.2, seed=0)

    model = eo.Sequential(eo.Linear(1, 1, seed=0))
    trainer = eo.Trainer(
        model,
        eo.SGD(model.parameters(), lr=0.05, momentum=0.9),
        eo.MSELoss(),
        metrics=["mae", "r2"],
    )

    history = trainer.fit(
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=32, shuffle=True, seed=0),
        epochs=40,
        validation_loader=eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64),
        verbose=True,
        log_every=10,
    )

    results = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64))
    print(
        f"\nTest MSE: {results['loss']:.4f}   MAE: {results['mae']:.4f}   R^2: {results['r2']:.4f}"
    )
    print(f"Learned weight: {float(model[0].weight.numpy()[0, 0]):.4f}")
    print(
        f"Learned bias:   {float(model[0].bias.numpy()[0]):.4f}  (data was generated with bias 2.0)"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    eo.plot_loss(history, save_path=OUTPUT_DIR / "linear_regression_loss.png")
    eo.plot_predictions(
        x_test,
        y_test,
        trainer.predict(x_test),
        title="Linear regression fit",
        save_path=OUTPUT_DIR / "linear_regression_fit.png",
    )
    print(f"\nCharts written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
