"""Tests for modules, layers and the Sequential container."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOError, EveryOShapeError
from everyo.nn.initialization import compute_fans, get_initializer


class TestLinear:
    def test_output_shape(self):
        assert eo.Linear(4, 7, seed=0)(eo.zeros(5, 4)).shape == (5, 7)

    def test_computes_xw_plus_b(self, rng):
        layer = eo.Linear(3, 2, seed=0)
        x = rng.normal(size=(4, 3)).astype(np.float32)
        expected = x @ layer.weight.numpy() + layer.bias.numpy()
        np.testing.assert_allclose(layer(eo.tensor(x)).numpy(), expected, rtol=1e-5)

    def test_without_bias(self):
        layer = eo.Linear(3, 2, bias=False, seed=0)
        assert len(layer.parameters()) == 1
        assert "bias" not in dict(layer.named_parameters())

    def test_parameter_shapes(self):
        layer = eo.Linear(6, 3, seed=0)
        assert layer.weight.shape == (6, 3)
        assert layer.bias.shape == (3,)
        assert layer.num_parameters() == 6 * 3 + 3

    def test_parameters_require_grad(self):
        assert all(p.requires_grad for p in eo.Linear(3, 2, seed=0).parameters())

    def test_seed_makes_initialisation_reproducible(self):
        first = eo.Linear(4, 4, seed=7).weight.numpy()
        second = eo.Linear(4, 4, seed=7).weight.numpy()
        np.testing.assert_allclose(first, second)

    def test_wrong_input_width_is_reported_clearly(self):
        with pytest.raises(EveryOShapeError) as info:
            eo.Linear(4, 2, seed=0)(eo.zeros(3, 9))
        assert "must be 4" in str(info.value)

    def test_non_positive_features_are_rejected(self):
        with pytest.raises(EveryOShapeError):
            eo.Linear(0, 4)

    def test_gradients_reach_both_parameters(self):
        layer = eo.Linear(3, 2, seed=0)
        eo.sum(layer(eo.ones(4, 3))).backward()
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        np.testing.assert_allclose(layer.bias.grad, [4.0, 4.0])

    def test_higher_rank_input(self):
        assert eo.Linear(3, 5, seed=0)(eo.zeros(2, 4, 3)).shape == (2, 4, 5)


class TestFlatten:
    def test_flattens_from_first_dim_by_default(self):
        assert eo.Flatten()(eo.zeros(8, 4, 4)).shape == (8, 16)

    def test_start_dim_is_configurable(self):
        assert eo.Flatten(start_dim=0)(eo.zeros(2, 3)).shape == (6,)

    def test_has_no_parameters(self):
        assert eo.Flatten().parameters() == []


class TestDropout:
    def test_identity_in_eval_mode(self, rng):
        layer = eo.Dropout(0.5, seed=0).eval()
        x = eo.tensor(rng.normal(size=(50, 20)).astype(np.float32))
        np.testing.assert_allclose(layer(x).numpy(), x.numpy())

    def test_drops_units_in_training_mode(self, rng):
        layer = eo.Dropout(0.5, seed=0)
        output = layer(eo.ones(200, 50)).numpy()
        dropped = float(np.mean(output == 0.0))
        assert 0.4 < dropped < 0.6

    def test_expected_value_is_preserved(self, rng):
        layer = eo.Dropout(0.5, seed=0)
        output = layer(eo.ones(500, 100)).numpy()
        assert output.mean() == pytest.approx(1.0, abs=0.05)

    def test_probability_is_validated(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            eo.Dropout(1.0)

    def test_gradient_flows_through_kept_units(self):
        layer = eo.Dropout(0.5, seed=0)
        x = eo.ones(10, 10)
        x.requires_grad = True
        eo.sum(layer(x)).backward()
        assert x.grad is not None


class TestActivationModules:
    @pytest.mark.parametrize(
        "module,reference",
        [
            (eo.ReLU(), lambda v: np.maximum(v, 0)),
            (eo.Sigmoid(), lambda v: 1 / (1 + np.exp(-v))),
            (eo.Tanh(), np.tanh),
        ],
    )
    def test_matches_reference(self, module, reference, rng):
        values = rng.normal(size=(4, 5)).astype(np.float64)
        np.testing.assert_allclose(module(eo.tensor(values)).numpy(), reference(values), rtol=1e-6)

    def test_softmax_axis_is_configurable(self, rng):
        values = eo.tensor(rng.normal(size=(4, 5)))
        np.testing.assert_allclose(
            eo.Softmax(axis=0)(values).numpy().sum(axis=0), np.ones(5), rtol=1e-6
        )


class TestSequential:
    def test_forward_chains_layers(self, small_model):
        assert small_model(eo.zeros(6, 4)).shape == (6, 3)

    def test_parameters_are_discovered_in_order(self, small_model):
        names = [name for name, _ in small_model.named_parameters()]
        assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]

    def test_parameter_count(self, small_model):
        assert small_model.num_parameters() == (4 * 8 + 8) + (8 * 3 + 3)

    def test_rejects_non_modules(self):
        with pytest.raises(TypeError, match="Module instances"):
            eo.Sequential(eo.Linear(2, 2, seed=0), "not a layer")

    def test_indexing_and_iteration(self, small_model):
        assert isinstance(small_model[0], eo.Linear)
        assert len(small_model) == 3
        assert len(list(small_model)) == 3

    def test_append(self):
        model = eo.Sequential(eo.Linear(2, 2, seed=0))
        model.append(eo.ReLU())
        assert len(model) == 2

    def test_train_eval_propagates_to_children(self):
        model = eo.Sequential(eo.Linear(2, 2, seed=0), eo.Dropout(0.5, seed=0))
        model.eval()
        assert all(not layer.training for layer in model)
        model.train()
        assert all(layer.training for layer in model)

    def test_zero_grad_clears_all_parameters(self, small_model):
        eo.sum(small_model(eo.ones(2, 4))).backward()
        small_model.zero_grad()
        assert all(p.grad is None for p in small_model.parameters())

    def test_nested_sequential(self):
        inner = eo.Sequential(eo.Linear(4, 4, seed=0), eo.ReLU())
        model = eo.Sequential(inner, eo.Linear(4, 2, seed=1))
        assert model(eo.zeros(3, 4)).shape == (3, 2)
        assert model.num_parameters() == (4 * 4 + 4) + (4 * 2 + 2)

    def test_summary_lists_layers(self, small_model):
        text = small_model.summary()
        assert "Linear" in text and "Total parameters" in text


class TestModuleBase:
    def test_forward_must_be_implemented(self):
        class Incomplete(eo.Module):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete()(eo.zeros(2))

    def test_missing_super_init_is_reported(self):
        class Broken(eo.Module):
            def __init__(self):  # deliberately missing super().__init__()
                self.weight = eo.Parameter(np.zeros(3))

        with pytest.raises(EveryOError, match="super"):
            Broken()

    def test_state_dict_round_trip(self, small_model):
        state = small_model.state_dict()
        rebuilt = eo.Sequential(eo.Linear(4, 8, seed=9), eo.ReLU(), eo.Linear(8, 3, seed=9))
        rebuilt.load_state_dict(state)
        np.testing.assert_allclose(
            rebuilt(eo.ones(2, 4)).numpy(), small_model(eo.ones(2, 4)).numpy()
        )

    def test_to_device_marks_every_parameter(self, small_model):
        small_model.to("cpu")
        assert all(p.device.type == "cpu" for p in small_model.parameters())


class TestInitialization:
    def test_compute_fans(self):
        assert compute_fans((6, 3)) == (6, 3)
        assert compute_fans((5,)) == (5, 5)

    def test_he_normal_variance(self):
        values = get_initializer("he_normal")((512, 512), seed=0)
        assert float(values.std()) == pytest.approx(np.sqrt(2 / 512), rel=0.05)

    def test_xavier_uniform_bounds(self):
        values = get_initializer("xavier_uniform")((100, 100), seed=0)
        limit = np.sqrt(6 / 200)
        assert values.min() >= -limit and values.max() <= limit

    def test_unknown_initializer(self):
        with pytest.raises(ValueError, match="Unknown initializer"):
            get_initializer("magic")
