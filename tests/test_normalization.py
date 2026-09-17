"""Tests for BatchNorm1D, BatchNorm2D and LayerNorm."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.backends import tensorflow_backend as tfb
from everyo.exceptions import EveryOSerializationError, EveryOShapeError
from tests.conftest import numeric_gradient

requires_tensorflow = pytest.mark.skipif(
    not tfb.is_available(), reason="TensorFlow is not installed."
)


class TestLayerNorm:
    def test_normalises_each_sample(self, rng):
        values = rng.normal(5.0, 3.0, size=(4, 10, 16)).astype(np.float32)
        out = eo.LayerNorm(16, affine=False)(eo.tensor(values)).numpy()
        np.testing.assert_allclose(out.mean(axis=-1), np.zeros((4, 10)), atol=1e-5)
        np.testing.assert_allclose(out.std(axis=-1), np.ones((4, 10)), atol=1e-3)

    def test_is_independent_of_other_samples(self, rng):
        """The defining property: batching must not change the result."""
        values = rng.normal(size=(6, 8)).astype(np.float32)
        layer = eo.LayerNorm(8)
        batched = layer(eo.tensor(values)).numpy()
        alone = layer(eo.tensor(values[:1])).numpy()
        np.testing.assert_allclose(batched[:1], alone, rtol=1e-5)

    def test_train_and_eval_agree(self, rng):
        values = eo.tensor(rng.normal(size=(4, 8)).astype(np.float32))
        layer = eo.LayerNorm(8)
        training = layer(values).numpy()
        layer.eval()
        np.testing.assert_allclose(layer(values).numpy(), training, rtol=1e-6)

    def test_affine_parameters_are_learned(self):
        layer = eo.LayerNorm(5)
        assert layer.weight.shape == (5,) and layer.bias.shape == (5,)
        eo.sum(layer(eo.ones(3, 5) * 2.0)).backward()
        assert layer.weight.grad is not None and layer.bias.grad is not None

    def test_without_affine_there_are_no_parameters(self):
        assert eo.LayerNorm(5, affine=False).parameters() == []

    def test_multi_axis_normalisation(self, rng):
        values = rng.normal(size=(3, 4, 5)).astype(np.float32)
        out = eo.LayerNorm((4, 5), affine=False)(eo.tensor(values)).numpy()
        np.testing.assert_allclose(out.reshape(3, -1).mean(axis=1), np.zeros(3), atol=1e-5)

    def test_shape_mismatch_is_explained(self):
        with pytest.raises(EveryOShapeError, match="trailing axes"):
            eo.LayerNorm(8)(eo.zeros(4, 5))

    def test_invalid_shape(self):
        with pytest.raises(EveryOShapeError, match="positive normalized_shape"):
            eo.LayerNorm(0)

    def test_gradient_matches_finite_differences(self, rng):
        layer = eo.LayerNorm(6)
        weights = eo.tensor(rng.normal(size=(3, 4, 6)))

        def loss(t):
            return eo.sum(layer(t) * weights)

        values = rng.normal(size=(3, 4, 6))
        tensor = eo.tensor(values.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), values.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-4, atol=1e-6)

    @requires_tensorflow
    @pytest.mark.tensorflow
    def test_matches_keras(self, rng):
        import tensorflow as tf

        values = rng.normal(size=(3, 7, 16)).astype(np.float32)
        ours = eo.LayerNorm(16)(eo.tensor(values)).numpy()
        theirs = tf.keras.layers.LayerNormalization(epsilon=1e-5)(values).numpy()
        np.testing.assert_allclose(ours, theirs, rtol=1e-4, atol=1e-6)


class TestBatchNorm1D:
    def test_normalises_each_feature_across_the_batch(self, rng):
        values = rng.normal(3.0, 2.0, size=(64, 5)).astype(np.float32)
        out = eo.BatchNorm1D(5, affine=False)(eo.tensor(values)).numpy()
        np.testing.assert_allclose(out.mean(axis=0), np.zeros(5), atol=1e-5)
        np.testing.assert_allclose(out.std(axis=0), np.ones(5), atol=1e-3)

    def test_running_statistics_are_tracked(self, rng):
        layer = eo.BatchNorm1D(4)
        assert float(layer.num_batches_tracked) == 0
        np.testing.assert_allclose(layer.running_mean, np.zeros(4))

        values = eo.tensor(rng.normal(10.0, 1.0, size=(32, 4)).astype(np.float32))
        for _ in range(10):
            layer(values)
        assert float(layer.num_batches_tracked) == 10
        # The running mean should be converging on the batch mean of ~10.
        assert layer.running_mean.mean() > 5.0

    def test_eval_mode_uses_running_statistics(self, rng):
        layer = eo.BatchNorm1D(3)
        values = eo.tensor(rng.normal(size=(16, 3)).astype(np.float32))
        for _ in range(20):
            layer(values)
        layer.eval()

        # In eval mode the result must not depend on the rest of the batch.
        full = layer(values).numpy()
        single = layer(eo.tensor(values.numpy()[:1])).numpy()
        np.testing.assert_allclose(full[:1], single, rtol=1e-5)

    def test_eval_mode_differs_from_training_mode(self, rng):
        layer = eo.BatchNorm1D(3)
        values = eo.tensor(rng.normal(5.0, 2.0, size=(16, 3)).astype(np.float32))
        training = layer(values).numpy()
        layer.eval()
        assert not np.allclose(layer(values).numpy(), training)

    def test_running_stats_can_be_disabled(self, rng):
        layer = eo.BatchNorm1D(3, track_running_stats=False)
        assert not layer.buffers()
        values = eo.tensor(rng.normal(size=(8, 3)).astype(np.float32))
        layer.eval()
        out = layer(values).numpy()
        np.testing.assert_allclose(out.mean(axis=0), np.zeros(3), atol=1e-5)

    def test_buffers_are_not_parameters(self):
        layer = eo.BatchNorm1D(4)
        assert len(layer.parameters()) == 2  # weight and bias only
        assert len(layer.buffers()) == 3  # mean, var, count
        assert layer.num_parameters() == 8

    def test_shape_validation(self):
        with pytest.raises(EveryOShapeError, match="2-D input"):
            eo.BatchNorm1D(4)(eo.zeros(2, 3, 4))
        with pytest.raises(EveryOShapeError, match="last axis"):
            eo.BatchNorm1D(4)(eo.zeros(8, 5))

    def test_invalid_hyperparameters(self):
        with pytest.raises(ValueError, match="eps"):
            eo.BatchNorm1D(4, eps=0.0)
        with pytest.raises(ValueError, match="momentum"):
            eo.BatchNorm1D(4, momentum=1.5)

    def test_gradient_matches_finite_differences(self, rng):
        layer = eo.BatchNorm1D(4)
        weights = eo.tensor(rng.normal(size=(8, 4)))

        def loss(t):
            return eo.sum(layer(t) * weights)

        values = rng.normal(size=(8, 4))
        tensor = eo.tensor(values.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), values.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-4, atol=1e-6)


class TestBatchNorm2D:
    def test_normalises_per_channel(self, rng):
        values = rng.normal(2.0, 3.0, size=(8, 6, 6, 4)).astype(np.float32)
        out = eo.BatchNorm2D(4, affine=False)(eo.tensor(values)).numpy()
        np.testing.assert_allclose(out.mean(axis=(0, 1, 2)), np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(out.std(axis=(0, 1, 2)), np.ones(4), atol=1e-3)

    def test_shape_validation(self):
        with pytest.raises(EveryOShapeError, match="4-D input"):
            eo.BatchNorm2D(3)(eo.zeros(8, 3))

    def test_works_after_a_convolution(self):
        model = eo.Sequential(
            eo.Conv2D(1, 4, 3, padding="same", seed=0), eo.BatchNorm2D(4), eo.ReLU()
        )
        assert model(eo.zeros(2, 8, 8, 1)).shape == (2, 8, 8, 4)

    def test_gradient_matches_finite_differences(self, rng):
        layer = eo.BatchNorm2D(2)
        weights = eo.tensor(rng.normal(size=(3, 4, 4, 2)))

        def loss(t):
            return eo.sum(layer(t) * weights)

        values = rng.normal(size=(3, 4, 4, 2))
        tensor = eo.tensor(values.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), values.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-4, atol=1e-6)


class TestBufferSupport:
    """Buffers are state that is saved but never trained."""

    def test_state_dict_contains_buffers(self):
        layer = eo.BatchNorm1D(3)
        keys = list(layer.state_dict())
        assert "running_mean" in keys and "running_var" in keys
        assert "num_batches_tracked" in keys

    def test_running_statistics_survive_save_and_load(self, tmp_path, rng):
        model = eo.Sequential(eo.Linear(4, 6, seed=0), eo.BatchNorm1D(6), eo.ReLU())
        values = eo.tensor(rng.normal(3.0, 2.0, size=(32, 4)).astype(np.float32))
        for _ in range(5):
            model(values)
        model.eval()
        expected = model(values).numpy()

        restored = eo.load(eo.save(model, tmp_path / "bn.evo"))
        np.testing.assert_allclose(restored[1].running_mean, model[1].running_mean)
        np.testing.assert_allclose(restored[1].running_var, model[1].running_var)
        np.testing.assert_allclose(restored(values).numpy(), expected, rtol=1e-6)

    def test_a_forgotten_buffer_is_reported(self):
        layer = eo.BatchNorm1D(3)
        state = layer.state_dict()
        del state["running_mean"]
        with pytest.raises(EveryOSerializationError, match="Missing keys"):
            layer.load_state_dict(state)

    def test_buffer_shape_is_validated(self):
        layer = eo.BatchNorm1D(3)
        state = layer.state_dict()
        state["running_mean"] = np.zeros(9)
        with pytest.raises(EveryOSerializationError, match="Buffer"):
            layer.load_state_dict(state)

    def test_optimizer_never_sees_buffers(self):
        model = eo.Sequential(eo.Linear(4, 4, seed=0), eo.BatchNorm1D(4))
        optimizer = eo.SGD(model.parameters(), lr=0.1)
        assert len(optimizer.parameters) == 4  # two Linear + two BatchNorm affine


@pytest.mark.slow
class TestNormalizationHelpsTraining:
    def test_batchnorm_trains(self, classification_data):
        features, labels = classification_data
        model = eo.Sequential(
            eo.Linear(4, 16, seed=0), eo.BatchNorm1D(16), eo.ReLU(), eo.Linear(16, 3, seed=1)
        )
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.02), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        history = trainer.fit(
            eo.DataLoader(eo.ArrayDataset(features, labels), batch_size=32, shuffle=True, seed=0),
            epochs=25,
            verbose=False,
        )
        assert history["accuracy"][-1] > 0.85
        assert history["loss"][-1] < history["loss"][0] * 0.5
