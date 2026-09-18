"""Tests for mixed-precision training.

The bar here is behavioural, not cosmetic. Autocast has to produce genuinely
reduced-precision activations *and* reduced-precision activation gradients,
because the whole point of :class:`~everyo.precision.GradScaler` is to rescue
gradients that would otherwise underflow — and a backward pass that quietly ran
in float32 would have nothing to rescue. So the tests below check the dtypes on
both sides of the graph, then demonstrate the underflow and its repair on
numbers rather than on types.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOError


def _regression_batch(samples: int = 64, features: int = 4):
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(samples, features)).astype(np.float32)
    weights = np.array([[1.0], [-2.0], [0.5], [0.25]], dtype=np.float32)
    return eo.tensor(inputs), eo.tensor(inputs @ weights)


class TestAutocastState:
    def test_disabled_by_default(self):
        assert eo.is_autocast_enabled() is False

    def test_context_manager_sets_and_restores(self):
        with eo.autocast():
            assert eo.is_autocast_enabled() is True
            assert eo.autocast_dtype() == np.float16
        assert eo.is_autocast_enabled() is False

    def test_nesting_restores_the_outer_state(self):
        with eo.autocast():
            with eo.autocast(False):
                assert eo.is_autocast_enabled() is False
            assert eo.is_autocast_enabled() is True
        assert eo.is_autocast_enabled() is False

    def test_reusing_one_instance_nested(self):
        # The same object entered twice must not lose the outer state.
        scope = eo.autocast()
        with scope:
            with scope:
                assert eo.is_autocast_enabled() is True
            assert eo.is_autocast_enabled() is True
        assert eo.is_autocast_enabled() is False

    def test_decorator_form(self):
        @eo.autocast()
        def inside():
            return eo.is_autocast_enabled()

        assert inside() is True
        assert eo.is_autocast_enabled() is False

    def test_state_is_restored_after_an_exception(self):
        with pytest.raises(RuntimeError), eo.autocast():
            raise RuntimeError("boom")
        assert eo.is_autocast_enabled() is False

    def test_integer_dtype_is_rejected(self):
        with pytest.raises(EveryOError, match="floating"):
            eo.autocast(dtype="int32")


class TestAutocastDtypes:
    def test_matmul_output_is_reduced_precision(self):
        left, right = eo.ones(2, 3), eo.ones(3, 4)
        with eo.autocast():
            assert eo.matmul(left, right).dtype == "float16"
        assert eo.matmul(left, right).dtype == "float32"

    def test_conv2d_output_is_reduced_precision(self):
        images, kernel = eo.zeros(2, 5, 5, 3), eo.zeros(3, 3, 3, 4)
        with eo.autocast():
            assert eo.conv2d(images, kernel, padding="same").dtype == "float16"

    def test_linear_bias_does_not_promote_the_activation(self):
        layer = eo.Linear(4, 3, seed=0)
        with eo.autocast():
            assert layer(eo.ones(2, 4)).dtype == "float16"

    def test_parameters_and_their_gradients_stay_float32(self):
        model = eo.Sequential(eo.Linear(8, 4, seed=0), eo.ReLU(), eo.Linear(4, 1, seed=1))
        with eo.autocast():
            output = model(eo.ones(3, 8))
        eo.sum(output).backward()
        for name, parameter in model.named_parameters():
            assert parameter.dtype == "float32", name
            assert parameter.grad.dtype == np.float32, name

    def test_activation_gradients_are_reduced_precision(self):
        # This is the property loss scaling depends on: if the backward pass
        # silently ran in float32, nothing would ever underflow. A float16
        # activation needs no cast node, so the gradient that reaches it is the
        # raw float16 gradient the matmul produced.
        activation = eo.tensor(
            np.ones((2, 4), dtype=np.float16), dtype="float16", requires_grad=True
        )
        with eo.autocast():
            output = eo.matmul(activation, eo.ones(4, 1))
        assert output.dtype == "float16"
        eo.sum(output).backward()
        assert activation.grad.dtype == np.float16

    def test_non_allowlisted_ops_stay_float32(self):
        with eo.autocast():
            assert eo.exp(eo.ones(3)).dtype == "float32"
            assert eo.mean(eo.ones(3)).dtype == "float32"

    def test_disabled_autocast_changes_nothing(self):
        with eo.autocast(False):
            assert eo.matmul(eo.ones(2, 3), eo.ones(3, 2)).dtype == "float32"

    def test_conv2d_values_match_the_float32_result(self):
        rng = np.random.default_rng(0)
        images = eo.tensor(rng.normal(size=(2, 6, 6, 2)).astype(np.float32))
        kernel = eo.tensor(rng.normal(size=(3, 3, 2, 3)).astype(np.float32))
        reference = eo.conv2d(images, kernel, padding="same")
        with eo.autocast():
            reduced = eo.conv2d(images, kernel, padding="same")
        # float16 carries ~3 decimal digits, so this is a loose but real bound.
        np.testing.assert_allclose(
            np.asarray(reduced.data, dtype=np.float32), reference.data, rtol=2e-2, atol=2e-2
        )


class TestGradScalerConfiguration:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"init_scale": 0.0},
            {"growth_factor": 1.0},
            {"backoff_factor": 1.0},
            {"backoff_factor": 0.0},
            {"growth_interval": 0},
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            eo.GradScaler(**kwargs)

    def test_disabled_scaler_is_a_pass_through(self):
        scaler = eo.GradScaler(enabled=False)
        loss = eo.mean(eo.ones(3))
        assert scaler.get_scale() == 1.0
        assert scaler.scale(loss) is loss

    def test_state_round_trips(self):
        scaler = eo.GradScaler(init_scale=1024.0)
        restored = eo.GradScaler()
        restored.load_state_dict(scaler.state_dict())
        assert restored.get_scale() == 1024.0


class TestGradScalerBehaviour:
    def _one_parameter_setup(self):
        layer = eo.Linear(2, 1, seed=0)
        return layer, eo.SGD(layer.parameters(), lr=0.1)

    def test_backward_matches_scale_then_backward(self):
        layer = eo.Linear(2, 1, seed=0)
        scaler = eo.GradScaler(init_scale=64.0)
        with eo.autocast():
            output = layer(eo.ones(3, 2))
        scaler.backward(eo.sum(output))
        through_helper = layer.weight.grad.copy()

        layer.zero_grad()
        with eo.autocast():
            output = layer(eo.ones(3, 2))
        scaler.scale(eo.sum(output)).backward()
        np.testing.assert_array_equal(layer.weight.grad, through_helper)

    def test_backward_suppresses_overflow_warnings(self):
        # An overflow in a scaled float16 backward pass is the signal step()
        # acts on, not a condition the caller needs warned about.
        layer = eo.Linear(2, 1, seed=0)
        scaler = eo.GradScaler(init_scale=2.0**15)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with eo.autocast():
                output = layer(eo.full((3, 2), 1e4))
            scaler.backward(eo.sum(output))

    def test_scaling_multiplies_the_loss(self):
        scaler = eo.GradScaler(init_scale=128.0)
        scaled = scaler.scale(eo.tensor(2.0))
        assert float(scaled.item()) == pytest.approx(256.0)

    def test_gradients_are_unscaled_before_the_step(self):
        layer, optimizer = self._one_parameter_setup()
        scaler = eo.GradScaler(init_scale=256.0)
        with eo.autocast():
            output = layer(eo.ones(4, 2))
        scaler.scale(eo.sum(output)).backward()
        scaler.unscale_(optimizer)
        # Every element of d(sum)/d(weight) is 1 for an all-ones input of 4 rows.
        np.testing.assert_allclose(layer.weight.grad, np.full((2, 1), 4.0), rtol=1e-5)

    def test_overflow_skips_the_step_and_halves_the_scale(self):
        layer, optimizer = self._one_parameter_setup()
        scaler = eo.GradScaler(init_scale=1024.0)
        before = layer.weight.data.copy()

        layer.weight.grad = np.full_like(layer.weight.data, np.inf)
        applied = scaler.step(optimizer)
        scaler.update()

        assert applied is False
        assert scaler.skipped_steps == 1
        assert scaler.get_scale() == 512.0
        np.testing.assert_array_equal(layer.weight.data, before)

    def test_nan_also_skips_the_step(self):
        layer, optimizer = self._one_parameter_setup()
        scaler = eo.GradScaler(init_scale=1024.0)
        layer.weight.grad = np.full_like(layer.weight.data, np.nan)
        assert scaler.step(optimizer) is False

    def test_scale_grows_after_enough_good_steps(self):
        layer, optimizer = self._one_parameter_setup()
        scaler = eo.GradScaler(init_scale=16.0, growth_interval=2)
        for _ in range(2):
            layer.weight.grad = np.ones_like(layer.weight.data)
            assert scaler.step(optimizer) is True
            scaler.update()
        assert scaler.get_scale() == 32.0
        assert scaler.applied_steps == 2

    def test_scale_never_falls_below_one(self):
        layer, optimizer = self._one_parameter_setup()
        scaler = eo.GradScaler(init_scale=1.5, backoff_factor=0.5)
        for _ in range(5):
            layer.weight.grad = np.full_like(layer.weight.data, np.inf)
            scaler.step(optimizer)
            scaler.update()
        assert scaler.get_scale() == 1.0

    def test_explicit_new_scale(self):
        scaler = eo.GradScaler(init_scale=8.0)
        scaler.update(64.0)
        assert scaler.get_scale() == 64.0


class TestUnderflowIsRealAndRepaired:
    """The empirical case for loss scaling, measured rather than asserted."""

    def _tiny_gradients(self, *, use_scaler: bool, loss_weight: float = 1e-7):
        model = eo.Sequential(eo.Linear(4, 32, seed=0), eo.Tanh(), eo.Linear(32, 1, seed=1))
        optimizer = eo.SGD(model.parameters(), lr=1e-12)
        scaler = eo.GradScaler(init_scale=2.0**15, enabled=use_scaler)
        inputs, targets = _regression_batch()

        optimizer.zero_grad()
        with eo.autocast():
            predictions = model(inputs)
        residual = predictions - targets
        loss = eo.mean(residual * residual) * loss_weight
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        return np.concatenate([np.asarray(p.grad).ravel() for p in model.parameters()])

    def test_gradients_underflow_without_scaling(self):
        gradients = self._tiny_gradients(use_scaler=False)
        # Almost everything has flushed to zero: float16's smallest subnormal
        # is about 6e-8 and these gradients are below it.
        assert (gradients == 0).mean() > 0.9

    def test_scaling_recovers_them(self):
        gradients = self._tiny_gradients(use_scaler=True)
        assert not np.any(gradients == 0)
        assert np.all(np.isfinite(gradients))

    def test_training_only_converges_with_the_scaler(self):
        def train(use_scaler: bool, steps: int = 300, loss_weight: float = 1e-7):
            model = eo.Sequential(eo.Linear(4, 32, seed=0), eo.Tanh(), eo.Linear(32, 1, seed=1))
            optimizer = eo.SGD(model.parameters(), lr=0.05 / loss_weight)
            scaler = eo.GradScaler(init_scale=2.0**15, enabled=use_scaler)
            inputs, targets = _regression_batch()
            for _ in range(steps):
                optimizer.zero_grad()
                with eo.autocast():
                    predictions = model(inputs)
                residual = predictions - targets
                loss = eo.mean(residual * residual) * loss_weight
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            with eo.no_grad():
                final = model(inputs) - targets
                return float(eo.mean(final * final).item())

        assert train(use_scaler=True) < 0.5
        assert train(use_scaler=False) > 5.0


class TestMixedPrecisionMatchesFullPrecision:
    def test_a_converged_model_agrees_with_float32_training(self):
        def train(autocast_enabled: bool, steps: int = 400):
            model = eo.Sequential(eo.Linear(4, 32, seed=0), eo.Tanh(), eo.Linear(32, 1, seed=1))
            optimizer = eo.SGD(model.parameters(), lr=0.05)
            scaler = eo.GradScaler(init_scale=2.0**12, enabled=autocast_enabled)
            inputs, targets = _regression_batch()
            for _ in range(steps):
                optimizer.zero_grad()
                with eo.autocast(autocast_enabled):
                    predictions = model(inputs)
                residual = predictions - targets
                loss = eo.mean(residual * residual)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            with eo.no_grad():
                final = model(inputs) - targets
                return float(eo.mean(final * final).item())

        full = train(False)
        mixed = train(True)
        assert full < 0.05
        # Mixed precision should land in the same neighbourhood, not merely
        # "also finish". A wide margin here would hide a real regression.
        assert mixed < 0.1
