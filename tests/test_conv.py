"""Tests for convolution and pooling.

Correctness is established three ways: shapes against hand-computed values,
forward results against TensorFlow (which uses the same NHWC layout, so no
transposing is involved), and every gradient against central finite
differences.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.backends import tensorflow_backend as tfb
from everyo.exceptions import EveryOShapeError
from everyo.nn.initialization import compute_fans
from tests.conftest import numeric_gradient

tensorflow_available = tfb.is_available()
requires_tensorflow = pytest.mark.skipif(
    not tensorflow_available, reason="TensorFlow is not installed."
)

GRAD_RTOL, GRAD_ATOL = 1e-5, 1e-7


class TestConvShapes:
    @pytest.mark.parametrize(
        "image,kernel,stride,padding,expected",
        [
            ((2, 7, 7, 3), 3, 1, "valid", (2, 5, 5, 4)),
            ((2, 7, 7, 3), 3, 1, "same", (2, 7, 7, 4)),
            ((2, 8, 8, 3), 3, 2, "same", (2, 4, 4, 4)),
            ((2, 8, 8, 3), 3, 2, "valid", (2, 3, 3, 4)),
            ((1, 6, 9, 3), (2, 3), (2, 1), "valid", (1, 3, 7, 4)),
            ((2, 5, 5, 3), 5, 1, "valid", (2, 1, 1, 4)),
            ((2, 5, 5, 3), 3, 1, 1, (2, 5, 5, 4)),
        ],
    )
    def test_output_shape(self, image, kernel, stride, padding, expected):
        kh, kw = (kernel, kernel) if isinstance(kernel, int) else kernel
        weight = eo.zeros(kh, kw, image[-1], 4)
        out = eo.conv2d(eo.zeros(*image), weight, stride=stride, padding=padding)
        assert out.shape == expected

    def test_same_padding_preserves_size_at_stride_one(self):
        for size in (5, 6, 7, 8):
            for kernel in (1, 3, 5):
                out = eo.conv2d(
                    eo.zeros(1, size, size, 1), eo.zeros(kernel, kernel, 1, 1), padding="same"
                )
                assert out.shape == (1, size, size, 1)

    def test_layer_shapes(self):
        layer = eo.Conv2D(3, 6, 3, padding="same", seed=0)
        assert layer(eo.zeros(4, 10, 10, 3)).shape == (4, 10, 10, 6)
        assert layer.weight.shape == (3, 3, 3, 6)
        assert layer.bias.shape == (6,)
        assert layer.num_parameters() == 3 * 3 * 3 * 6 + 6

    def test_layer_without_bias(self):
        layer = eo.Conv2D(3, 6, 3, bias=False, seed=0)
        assert len(layer.parameters()) == 1


class TestConvNumerics:
    def test_identity_kernel_copies_the_image(self, rng):
        image = rng.normal(size=(2, 5, 5, 1)).astype(np.float32)
        kernel = np.zeros((3, 3, 1, 1), dtype=np.float32)
        kernel[1, 1, 0, 0] = 1.0
        out = eo.conv2d(eo.tensor(image), eo.tensor(kernel), padding="same").numpy()
        np.testing.assert_allclose(out, image, rtol=1e-6)

    def test_sum_kernel_sums_the_window(self):
        image = eo.ones(1, 4, 4, 1)
        kernel = eo.ones(2, 2, 1, 1)
        np.testing.assert_allclose(eo.conv2d(image, kernel).numpy(), np.full((1, 3, 3, 1), 4.0))

    def test_bias_is_added_per_output_channel(self):
        out = eo.conv2d(eo.zeros(1, 4, 4, 1), eo.zeros(3, 3, 1, 2), eo.tensor([1.0, -2.0]))
        np.testing.assert_allclose(out.numpy()[..., 0], np.ones((1, 2, 2)))
        np.testing.assert_allclose(out.numpy()[..., 1], np.full((1, 2, 2), -2.0))

    def test_channels_are_summed(self, rng):
        image = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
        kernel = np.ones((3, 3, 4, 1), dtype=np.float32)
        assert eo.conv2d(eo.tensor(image), eo.tensor(kernel)).item() == pytest.approx(
            float(image.sum()), rel=1e-5
        )


class TestConvGradients:
    @staticmethod
    def _check(shape, kernel_shape, stride, padding, seed=0):
        rng = np.random.default_rng(seed)
        image = rng.normal(size=shape)
        kernel = rng.normal(size=kernel_shape)
        bias = rng.normal(size=(kernel_shape[-1],))

        def loss(i, k, b):
            return eo.sum(eo.conv2d(i, k, b, stride=stride, padding=padding))

        # gradient with respect to each input in turn
        tensor_image = eo.tensor(image.copy(), requires_grad=True)
        loss(tensor_image, eo.tensor(kernel), eo.tensor(bias)).backward()
        expected = numeric_gradient(
            lambda a: loss(eo.tensor(a.copy()), eo.tensor(kernel), eo.tensor(bias)).item(),
            image.copy(),
        )
        np.testing.assert_allclose(tensor_image.grad, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)

        tensor_kernel = eo.tensor(kernel.copy(), requires_grad=True)
        loss(eo.tensor(image), tensor_kernel, eo.tensor(bias)).backward()
        expected = numeric_gradient(
            lambda a: loss(eo.tensor(image), eo.tensor(a.copy()), eo.tensor(bias)).item(),
            kernel.copy(),
        )
        np.testing.assert_allclose(tensor_kernel.grad, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)

        tensor_bias = eo.tensor(bias.copy(), requires_grad=True)
        loss(eo.tensor(image), eo.tensor(kernel), tensor_bias).backward()
        expected = numeric_gradient(
            lambda a: loss(eo.tensor(image), eo.tensor(kernel), eo.tensor(a.copy())).item(),
            bias.copy(),
        )
        np.testing.assert_allclose(tensor_bias.grad, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)

    def test_valid_stride_one(self):
        self._check((2, 5, 5, 2), (3, 3, 2, 3), 1, "valid")

    def test_same_stride_one(self):
        self._check((2, 5, 5, 2), (3, 3, 2, 3), 1, "same")

    def test_valid_stride_two(self):
        self._check((2, 6, 6, 2), (3, 3, 2, 3), 2, "valid")

    def test_same_stride_two(self):
        """Overlapping and clipped windows at once — the fiddliest case."""
        self._check((2, 5, 5, 2), (3, 3, 2, 3), 2, "same")

    def test_rectangular_kernel_and_stride(self):
        self._check((1, 6, 7, 2), (2, 3, 2, 2), (2, 1), "valid")

    def test_single_channel(self):
        self._check((2, 4, 4, 1), (3, 3, 1, 1), 1, "valid")

    def test_gradient_through_a_layer_reaches_both_parameters(self):
        layer = eo.Conv2D(2, 3, 3, padding="same", seed=0)
        eo.sum(layer(eo.ones(2, 5, 5, 2))).backward()
        assert layer.weight.grad is not None and layer.bias.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape
        np.testing.assert_allclose(layer.bias.grad, np.full((3,), 2 * 25.0))


class TestPooling:
    def test_max_pool_picks_the_maximum(self):
        image = eo.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 2, 2, 1))
        assert eo.max_pool2d(image, 2).item() == pytest.approx(4.0)

    def test_avg_pool_averages(self):
        image = eo.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 2, 2, 1))
        assert eo.avg_pool2d(image, 2).item() == pytest.approx(2.5)

    def test_default_stride_is_the_window(self):
        assert eo.max_pool2d(eo.zeros(2, 8, 8, 3), 2).shape == (2, 4, 4, 3)

    def test_overlapping_windows(self):
        assert eo.max_pool2d(eo.zeros(1, 5, 5, 1), 3, stride=1).shape == (1, 3, 3, 1)

    def test_channels_are_pooled_independently(self, rng):
        image = rng.normal(size=(1, 4, 4, 3)).astype(np.float32)
        pooled = eo.max_pool2d(eo.tensor(image), 2).numpy()
        for channel in range(3):
            expected = image[0, :2, :2, channel].max()
            assert pooled[0, 0, 0, channel] == pytest.approx(expected, rel=1e-6)

    def test_avg_pool_same_padding_ignores_the_padding(self):
        """An edge window must average real cells only, not the zeros."""
        image = eo.ones(1, 3, 3, 1)
        pooled = eo.avg_pool2d(image, 2, padding="same").numpy()
        np.testing.assert_allclose(pooled, np.ones((1, 2, 2, 1)))

    def test_max_pool_same_padding_never_returns_padding(self):
        image = eo.tensor(np.full((1, 3, 3, 1), -5.0))
        pooled = eo.max_pool2d(image, 2, padding="same").numpy()
        assert pooled.max() == pytest.approx(-5.0)

    def test_max_pool_gradient_routes_to_the_winner(self):
        image = eo.tensor(
            np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 2, 2, 1), requires_grad=True
        )
        eo.sum(eo.max_pool2d(image, 2)).backward()
        np.testing.assert_allclose(image.grad.reshape(2, 2), [[0.0, 0.0], [0.0, 1.0]])

    def test_max_pool_ties_go_to_one_element(self):
        image = eo.tensor(np.ones((1, 2, 2, 1)), requires_grad=True)
        eo.sum(eo.max_pool2d(image, 2)).backward()
        assert image.grad.sum() == pytest.approx(1.0)

    def test_avg_pool_gradient_is_shared(self):
        image = eo.tensor(np.zeros((1, 2, 2, 1)), requires_grad=True)
        eo.sum(eo.avg_pool2d(image, 2)).backward()
        np.testing.assert_allclose(image.grad.reshape(2, 2), np.full((2, 2), 0.25))

    @pytest.mark.parametrize("pool", ["max", "avg"])
    @pytest.mark.parametrize("padding", ["valid", "same"])
    @pytest.mark.parametrize("stride", [None, 1])
    def test_gradient_matches_finite_differences(self, pool, padding, stride):
        fn = eo.max_pool2d if pool == "max" else eo.avg_pool2d
        rng = np.random.default_rng(0)
        image = rng.normal(size=(2, 5, 5, 2))
        # Random output weights, so the gradient is not uniform and a wrong
        # routing cannot pass by symmetry.
        shape = fn(eo.tensor(image), 2, stride=stride, padding=padding).shape
        weights = eo.tensor(rng.normal(size=shape))

        def loss(t):
            return eo.sum(fn(t, 2, stride=stride, padding=padding) * weights)

        tensor = eo.tensor(image.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), image.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-5, atol=1e-6)


class TestErrors:
    def test_input_must_be_four_dimensional(self):
        with pytest.raises(EveryOShapeError, match="4-D tensor"):
            eo.conv2d(eo.zeros(4, 8, 8), eo.zeros(3, 3, 1, 2))

    def test_kernel_must_be_four_dimensional(self):
        with pytest.raises(EveryOShapeError, match="4-D kernel"):
            eo.conv2d(eo.zeros(1, 8, 8, 1), eo.zeros(3, 3))

    def test_channel_mismatch_is_explained(self):
        with pytest.raises(EveryOShapeError) as info:
            eo.conv2d(eo.zeros(1, 8, 8, 3), eo.zeros(3, 3, 5, 2))
        assert "3 channel(s)" in str(info.value) and "expects 5" in str(info.value)

    def test_layer_channel_mismatch_is_explained(self):
        with pytest.raises(EveryOShapeError, match="must be 3 channel"):
            eo.Conv2D(3, 4, 3, seed=0)(eo.zeros(2, 8, 8, 1))

    def test_kernel_larger_than_input(self):
        with pytest.raises(EveryOShapeError, match="does not fit"):
            eo.conv2d(eo.zeros(1, 3, 3, 1), eo.zeros(5, 5, 1, 1))

    def test_bad_padding_name(self):
        with pytest.raises(EveryOShapeError, match="Unknown padding"):
            eo.conv2d(eo.zeros(1, 5, 5, 1), eo.zeros(3, 3, 1, 1), padding="reflect")

    def test_bias_shape_is_validated(self):
        with pytest.raises(EveryOShapeError, match="bias must have shape"):
            eo.conv2d(eo.zeros(1, 5, 5, 1), eo.zeros(3, 3, 1, 4), eo.zeros(3))

    def test_non_positive_kernel(self):
        with pytest.raises(EveryOShapeError, match="must be positive"):
            eo.Conv2D(1, 2, 0)

    def test_non_positive_channels(self):
        with pytest.raises(EveryOShapeError, match="positive channel counts"):
            eo.Conv2D(0, 2, 3)

    def test_pool_window_larger_than_input(self):
        with pytest.raises(EveryOShapeError, match="larger than the padded"):
            eo.max_pool2d(eo.zeros(1, 2, 2, 1), 4)


class TestInitialization:
    def test_convolution_fans(self):
        assert compute_fans((3, 3, 2, 4)) == (18, 36)

    def test_he_normal_scales_with_the_receptive_field(self):
        from everyo.nn.initialization import he_normal

        values = he_normal((3, 3, 16, 32), seed=0)
        assert float(values.std()) == pytest.approx(np.sqrt(2 / (3 * 3 * 16)), rel=0.05)


class TestSerialization:
    def test_cnn_round_trip(self, tmp_path, rng):
        model = eo.Sequential(
            eo.Conv2D(1, 4, 3, padding="same", seed=0),
            eo.ReLU(),
            eo.MaxPool2D(2),
            eo.Conv2D(4, 8, 3, stride=1, padding="valid", seed=1),
            eo.ReLU(),
            eo.AvgPool2D(2),
            eo.Flatten(),
            eo.Linear(8, 3, seed=2),
        )
        images = eo.tensor(rng.normal(size=(2, 8, 8, 1)).astype(np.float32))
        model.eval()
        expected = model(images).numpy()

        restored = eo.load(eo.save(model, tmp_path / "cnn.evo"))
        np.testing.assert_allclose(restored(images).numpy(), expected, rtol=1e-6)
        assert [type(layer).__name__ for layer in restored] == [
            "Conv2D",
            "ReLU",
            "MaxPool2D",
            "Conv2D",
            "ReLU",
            "AvgPool2D",
            "Flatten",
            "Linear",
        ]
        assert restored[0].padding == "same"
        assert restored[3].kernel_size == (3, 3)


@requires_tensorflow
@pytest.mark.tensorflow
class TestAgainstTensorFlow:
    """EveryO and TensorFlow share the NHWC layout, so results compare directly."""

    @pytest.mark.parametrize(
        "shape,kernel,stride,padding",
        [
            ((2, 7, 7, 3), (3, 3), 1, "VALID"),
            ((2, 7, 7, 3), (3, 3), 1, "SAME"),
            ((3, 8, 8, 1), (5, 5), 2, "SAME"),
            ((1, 6, 9, 2), (2, 3), (2, 1), "VALID"),
            ((2, 5, 5, 2), (3, 3), 2, "SAME"),
        ],
    )
    def test_forward_matches(self, shape, kernel, stride, padding, rng):
        tf = tfb.require()
        image = rng.normal(size=shape).astype(np.float32)
        weight = rng.normal(size=(*kernel, shape[-1], 4)).astype(np.float32)
        bias = rng.normal(size=(4,)).astype(np.float32)
        strides = (stride, stride) if isinstance(stride, int) else stride

        ours = eo.conv2d(
            eo.tensor(image),
            eo.tensor(weight),
            eo.tensor(bias),
            stride=stride,
            padding=padding.lower(),
        ).numpy()
        theirs = (
            tf.nn.conv2d(image, weight, strides=[1, *strides, 1], padding=padding).numpy() + bias
        )
        np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("padding", ["VALID", "SAME"])
    @pytest.mark.parametrize("window,stride", [(2, None), (3, 1), (2, 2), (3, 2)])
    def test_pooling_matches(self, padding, window, stride, rng):
        tf = tfb.require()
        image = rng.normal(size=(2, 7, 7, 3)).astype(np.float32)
        effective = window if stride is None else stride

        np.testing.assert_allclose(
            eo.max_pool2d(eo.tensor(image), window, stride=stride, padding=padding.lower()).numpy(),
            tf.nn.max_pool2d(image, window, effective, padding=padding).numpy(),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            eo.avg_pool2d(eo.tensor(image), window, stride=stride, padding=padding.lower()).numpy(),
            tf.nn.avg_pool2d(image, window, effective, padding=padding).numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_convolution_gradient_matches(self, rng):
        tf = tfb.require()
        image = rng.normal(size=(2, 6, 6, 2)).astype(np.float32)
        weight = rng.normal(size=(3, 3, 2, 3)).astype(np.float32)

        tensor_image = eo.tensor(image, requires_grad=True)
        tensor_weight = eo.tensor(weight, requires_grad=True)
        eo.sum(eo.conv2d(tensor_image, tensor_weight, padding="same")).backward()

        tf_image = tf.Variable(image)
        tf_weight = tf.Variable(weight)
        with tf.GradientTape() as tape:
            out = tf.reduce_sum(tf.nn.conv2d(tf_image, tf_weight, strides=1, padding="SAME"))
        grad_image, grad_weight = tape.gradient(out, [tf_image, tf_weight])

        np.testing.assert_allclose(tensor_image.grad, np.asarray(grad_image), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            tensor_weight.grad, np.asarray(grad_weight), rtol=1e-4, atol=1e-5
        )

    def test_max_pool_gradient_matches(self, rng):
        tf = tfb.require()
        image = rng.normal(size=(2, 6, 6, 2)).astype(np.float32)

        tensor = eo.tensor(image, requires_grad=True)
        eo.sum(eo.max_pool2d(tensor, 2)).backward()

        tf_image = tf.Variable(image)
        with tf.GradientTape() as tape:
            out = tf.reduce_sum(tf.nn.max_pool2d(tf_image, 2, 2, padding="VALID"))
        np.testing.assert_allclose(
            tensor.grad, np.asarray(tape.gradient(out, tf_image)), rtol=1e-5, atol=1e-6
        )

    def test_keras_equivalent_has_the_same_parameter_count(self):
        model = eo.Sequential(
            eo.Conv2D(1, 8, 3, padding="same", seed=0),
            eo.ReLU(),
            eo.MaxPool2D(2),
            eo.Conv2D(8, 16, 3, padding="same", seed=1),
            eo.ReLU(),
            eo.AvgPool2D(2),
            eo.Flatten(),
            eo.Linear(16 * 2 * 2, 10, seed=2),
        )
        keras_model = tfb.build_keras_model(model, input_shape=(8, 8, 1))
        assert keras_model.count_params() == model.num_parameters()


@pytest.mark.slow
class TestTraining:
    def test_cnn_learns_the_digit_task(self):
        from everyo.datasets import load_digits

        images, labels = load_digits(samples_per_class=120, noise=0.22, seed=0, flatten=False)
        images = images[..., None]
        x_train, x_test, y_train, y_test = eo.stratified_split(
            images, labels, test_size=0.2, seed=0
        )

        model = eo.Sequential(
            eo.Conv2D(1, 8, 3, padding="same", seed=0),
            eo.ReLU(),
            eo.MaxPool2D(2),
            eo.Conv2D(8, 16, 3, padding="same", seed=1),
            eo.ReLU(),
            eo.MaxPool2D(2),
            eo.Flatten(),
            eo.Linear(2 * 2 * 16, 10, seed=2),
        )
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        history = trainer.fit(
            eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=64, shuffle=True, seed=0),
            epochs=12,
            verbose=False,
        )
        accuracy = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128))[
            "accuracy"
        ]
        assert accuracy > 0.90, f"CNN only reached {accuracy:.3f}"
        assert history["loss"][-1] < history["loss"][0] * 0.2

    def test_untrained_cnn_is_near_chance(self):
        from everyo.datasets import load_digits

        images, labels = load_digits(samples_per_class=40, seed=1, flatten=False)
        model = eo.Sequential(
            eo.Conv2D(1, 4, 3, padding="same", seed=0),
            eo.ReLU(),
            eo.MaxPool2D(2),
            eo.Flatten(),
            eo.Linear(4 * 4 * 4, 10, seed=1),
        )
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        accuracy = trainer.evaluate(
            eo.DataLoader(eo.ArrayDataset(images[..., None], labels), batch_size=64)
        )["accuracy"]
        assert accuracy < 0.4
