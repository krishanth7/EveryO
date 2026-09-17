"""Tests for the optional TensorFlow integration.

Every test in this module skips when TensorFlow is unavailable; the detection
tests themselves run either way, because graceful absence is part of the
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.backends import tensorflow_backend as tfb
from everyo.exceptions import EveryOBackendError

tensorflow_available = tfb.is_available()
requires_tensorflow = pytest.mark.skipif(
    not tensorflow_available, reason="TensorFlow is not installed."
)


class TestDetection:
    """These run with or without TensorFlow installed."""

    def test_is_available_returns_a_bool(self):
        assert isinstance(tfb.is_available(), bool)

    def test_importing_the_module_never_raises(self):
        tfb.describe()
        tfb.list_devices()
        tfb.gpu_available()

    def test_describe_reports_the_state(self):
        description = tfb.describe()
        assert description["name"] == "tensorflow"
        assert description["available"] is tensorflow_available

    def test_reason_is_given_when_unavailable(self):
        if tensorflow_available:
            assert tfb.unavailable_reason() is None
        else:
            assert "TensorFlow" in tfb.unavailable_reason()

    @pytest.mark.skipif(tensorflow_available, reason="TensorFlow is installed.")
    def test_require_raises_a_helpful_error(self):
        with pytest.raises(EveryOBackendError, match="requires TensorFlow"):
            tfb.require()

    @pytest.mark.skipif(tensorflow_available, reason="TensorFlow is installed.")
    def test_core_still_works_without_tensorflow(self):
        model = eo.Sequential(eo.Linear(2, 1, seed=0))
        assert model(eo.zeros(3, 2)).shape == (3, 1)

    def test_available_backends_always_lists_numpy(self):
        from everyo.backends import available_backends

        assert "numpy" in available_backends()


@requires_tensorflow
@pytest.mark.tensorflow
class TestNumericalAgreement:
    """EveryO's own kernels are checked against TensorFlow's."""

    def test_matmul(self, rng):
        a = rng.normal(size=(32, 16)).astype(np.float32)
        b = rng.normal(size=(16, 8)).astype(np.float32)
        np.testing.assert_allclose(
            eo.matmul(eo.tensor(a), eo.tensor(b)).numpy(),
            tfb.matmul(a, b),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_relu(self, rng):
        values = rng.normal(size=(64,)).astype(np.float32)
        np.testing.assert_allclose(eo.relu(eo.tensor(values)).numpy(), tfb.relu(values), rtol=1e-6)

    def test_softmax(self, rng):
        tf = tfb.require()
        values = rng.normal(size=(8, 5)).astype(np.float32)
        np.testing.assert_allclose(
            eo.softmax(eo.tensor(values), axis=1).numpy(),
            np.asarray(tf.nn.softmax(values, axis=1)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_sigmoid(self, rng):
        tf = tfb.require()
        values = rng.normal(size=(50,)).astype(np.float32)
        np.testing.assert_allclose(
            eo.sigmoid(eo.tensor(values)).numpy(),
            np.asarray(tf.nn.sigmoid(values)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_cross_entropy_loss(self, rng):
        tf = tfb.require()
        logits = rng.normal(size=(16, 4)).astype(np.float32)
        labels = rng.integers(0, 4, size=16)
        reference = float(
            np.mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels.astype(np.int32), logits=logits
                ).numpy()
            )
        )
        assert eo.CrossEntropyLoss()(eo.tensor(logits), labels).item() == pytest.approx(
            reference, rel=1e-5
        )

    def test_cross_entropy_gradient(self, rng):
        tf = tfb.require()
        values = rng.normal(size=(8, 3)).astype(np.float32)
        labels = rng.integers(0, 3, size=8)

        logits = eo.tensor(values, requires_grad=True)
        eo.CrossEntropyLoss()(logits, labels).backward()

        tf_logits = tf.Variable(values)
        with tf.GradientTape() as tape:
            tf_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels.astype(np.int32), logits=tf_logits
                )
            )
        tf_gradient = np.asarray(tape.gradient(tf_loss, tf_logits))

        np.testing.assert_allclose(logits.grad, tf_gradient, rtol=1e-4, atol=1e-6)

    def test_mse_loss(self, rng):
        tf = tfb.require()
        prediction = rng.normal(size=(20, 2)).astype(np.float32)
        target = rng.normal(size=(20, 2)).astype(np.float32)
        reference = float(tf.reduce_mean(tf.square(prediction - target)).numpy())
        assert eo.MSELoss()(eo.tensor(prediction), eo.tensor(target)).item() == pytest.approx(
            reference, rel=1e-5
        )


@requires_tensorflow
@pytest.mark.tensorflow
class TestInteroperability:
    def test_to_and_from_tensorflow(self, rng):
        values = rng.normal(size=(4, 3)).astype(np.float32)
        tensor = eo.tensor(values)
        round_trip = tfb.from_tensorflow(tfb.to_tensorflow(tensor))
        np.testing.assert_allclose(round_trip.numpy(), values, rtol=1e-6)

    def test_build_keras_model_mirrors_the_architecture(self):
        model = eo.Sequential(
            eo.Linear(8, 16, seed=0), eo.ReLU(), eo.Dropout(0.2), eo.Linear(16, 3, seed=1)
        )
        keras_model = tfb.build_keras_model(model, input_shape=(8,))
        assert keras_model.output_shape[-1] == 3
        assert keras_model.count_params() == model.num_parameters()

    def test_unsupported_layer_is_reported(self):
        class Custom(eo.Module):
            def forward(self, x):
                return x

        with pytest.raises(EveryOBackendError, match="No Keras equivalent"):
            tfb.build_keras_model(eo.Sequential(Custom()), input_shape=(4,))

    def test_non_sequential_model_is_reported(self):
        with pytest.raises(EveryOBackendError, match="Sequential"):
            tfb.build_keras_model(eo.Linear(2, 2, seed=0), input_shape=(2,))


@requires_tensorflow
@pytest.mark.tensorflow
@pytest.mark.slow
class TestReferenceTraining:
    def test_everyo_matches_the_keras_reference(self, classification_data):
        """Both stacks should reach a comparable loss on the same easy problem."""
        features, labels = classification_data
        model = eo.Sequential(eo.Linear(4, 16, seed=0), eo.ReLU(), eo.Linear(16, 3, seed=1))

        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        everyo_history = trainer.fit(
            eo.DataLoader(eo.ArrayDataset(features, labels), batch_size=32, shuffle=True, seed=0),
            epochs=20,
            verbose=False,
        )
        reference = tfb.train_reference_model(
            model, features, labels, epochs=20, batch_size=32, learning_rate=0.01
        )

        everyo_loss = everyo_history["loss"][-1]
        keras_loss = reference["loss"][-1]
        assert everyo_loss < 2.0 * keras_loss + 0.2
