"""End-to-end tests covering the full EveryO pipeline.

These exercise the path a real user takes: dataset -> preprocessing -> loader
-> model -> training -> evaluation -> inference -> save/load.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.datasets import load_digits, make_moons, make_regression


@pytest.mark.slow
class TestDigitClassification:
    def test_full_pipeline_learns_the_task(self, tmp_path):
        features, labels = load_digits(samples_per_class=60, seed=0)
        x_train, x_test, y_train, y_test = eo.stratified_split(
            features, labels, test_size=0.25, seed=0
        )

        model = eo.Sequential(
            eo.Linear(64, 64, seed=0),
            eo.ReLU(),
            eo.Linear(64, 10, seed=1),
        )
        trainer = eo.Trainer(
            model,
            eo.Adam(model.parameters(), lr=0.01),
            eo.CrossEntropyLoss(),
            metrics=["accuracy"],
        )
        history = trainer.fit(
            eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=32, shuffle=True, seed=0),
            epochs=20,
            validation_loader=eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64),
            verbose=False,
        )

        results = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64))
        assert results["accuracy"] > 0.85, f"accuracy was only {results['accuracy']:.3f}"
        assert history["loss"][-1] < history["loss"][0] * 0.5

        # Inference on single samples must agree with batched inference.
        batched = trainer.predict(x_test[:8])
        single = np.concatenate([trainer.predict(x_test[i : i + 1]) for i in range(8)])
        np.testing.assert_allclose(batched, single, rtol=1e-5)

        # And the model must survive a save/load round trip.
        path = eo.save(model, tmp_path / "digits.evo", metadata={"accuracy": results["accuracy"]})
        restored = eo.load(path)
        restored_trainer = eo.Trainer(
            restored,
            eo.Adam(restored.parameters(), lr=0.01),
            eo.CrossEntropyLoss(),
            metrics=["accuracy"],
        )
        restored_results = restored_trainer.evaluate(
            eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64)
        )
        assert restored_results["accuracy"] == pytest.approx(results["accuracy"])

    def test_untrained_model_is_near_chance(self):
        features, labels = load_digits(samples_per_class=30, seed=1)
        model = eo.Sequential(eo.Linear(64, 32, seed=0), eo.ReLU(), eo.Linear(32, 10, seed=1))
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        accuracy = trainer.evaluate(
            eo.DataLoader(eo.ArrayDataset(features, labels), batch_size=64)
        )["accuracy"]
        assert accuracy < 0.4, "An untrained model should not classify digits well."


@pytest.mark.slow
class TestNonLinearProblems:
    def test_mlp_solves_moons_and_a_linear_model_does_not(self):
        features, labels = make_moons(n_samples=400, noise=0.1, seed=0)
        loader = eo.DataLoader(
            eo.ArrayDataset(features, labels), batch_size=32, shuffle=True, seed=0
        )

        def train(model, epochs=60):
            trainer = eo.Trainer(
                model,
                eo.Adam(model.parameters(), lr=0.02),
                eo.CrossEntropyLoss(),
                metrics=["accuracy"],
            )
            trainer.fit(loader, epochs=epochs, verbose=False)
            return trainer.evaluate(loader)["accuracy"]

        mlp_accuracy = train(
            eo.Sequential(
                eo.Linear(2, 32, seed=0),
                eo.Tanh(),
                eo.Linear(32, 16, seed=1),
                eo.Tanh(),
                eo.Linear(16, 2, seed=2),
            )
        )
        linear_accuracy = train(eo.Sequential(eo.Linear(2, 2, seed=0)))

        assert mlp_accuracy > 0.95
        assert mlp_accuracy > linear_accuracy


@pytest.mark.slow
class TestRegression:
    def test_recovers_a_linear_relationship(self):
        features, targets = make_regression(n_samples=300, n_features=3, noise=0.05, seed=0)
        x_train, x_test, y_train, y_test = eo.train_test_split(
            features, targets, test_size=0.2, seed=0
        )

        model = eo.Sequential(eo.Linear(3, 1, seed=0))
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.05), eo.MSELoss(), metrics=["r2"]
        )
        trainer.fit(
            eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=32, shuffle=True, seed=0),
            epochs=100,
            verbose=False,
        )
        results = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64))
        assert results["r2"] > 0.95


class TestPreprocessingPipeline:
    def test_scaling_uses_training_statistics_only(self):
        rng = np.random.default_rng(0)
        features = rng.normal(50.0, 20.0, size=(200, 4)).astype(np.float32)
        targets = (features[:, 0] > 50).astype(np.int64)

        x_train, x_test, y_train, y_test = eo.train_test_split(
            features, targets, test_size=0.25, seed=0
        )
        scaler = eo.StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = eo.Sequential(eo.Linear(4, 16, seed=0), eo.ReLU(), eo.Linear(16, 2, seed=1))
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.02), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        trainer.fit(
            eo.DataLoader(
                eo.ArrayDataset(x_train_scaled, y_train), batch_size=16, shuffle=True, seed=0
            ),
            epochs=30,
            verbose=False,
        )
        accuracy = trainer.evaluate(
            eo.DataLoader(eo.ArrayDataset(x_test_scaled, y_test), batch_size=32)
        )["accuracy"]
        assert accuracy > 0.85


class TestDeterminism:
    def test_identical_seeds_give_identical_training(self, classification_data):
        features, labels = classification_data

        def run():
            model = eo.Sequential(eo.Linear(4, 8, seed=0), eo.ReLU(), eo.Linear(8, 3, seed=1))
            trainer = eo.Trainer(model, eo.Adam(model.parameters(), lr=0.02), eo.CrossEntropyLoss())
            history = trainer.fit(
                eo.DataLoader(
                    eo.ArrayDataset(features, labels), batch_size=16, shuffle=True, seed=123
                ),
                epochs=5,
                verbose=False,
            )
            return history["loss"]

        np.testing.assert_allclose(run(), run(), rtol=1e-10)


class TestDatasets:
    def test_digit_dataset_shape_and_range(self):
        features, labels = load_digits(samples_per_class=10, seed=0)
        assert features.shape == (100, 64)
        assert labels.shape == (100,)
        assert features.min() >= 0.0 and features.max() <= 1.0
        assert set(np.unique(labels)) == set(range(10))

    def test_digit_dataset_is_reproducible(self):
        first = load_digits(samples_per_class=5, seed=3)[0]
        second = load_digits(samples_per_class=5, seed=3)[0]
        np.testing.assert_allclose(first, second)

    def test_unflattened_images(self):
        features, _ = load_digits(samples_per_class=4, flatten=False, seed=0)
        assert features.shape == (40, 8, 8)

    def test_synthetic_generators(self):
        from everyo.datasets import make_blobs, make_spirals, make_xor

        for generator in (make_blobs, make_spirals, make_xor):
            features, labels = generator(n_samples=60, seed=0)
            assert len(features) == len(labels)
            assert features.dtype == np.float32
