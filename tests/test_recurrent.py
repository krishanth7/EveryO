"""Tests for the recurrent layers and the Embedding lookup."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryOShapeError
from tests.conftest import numeric_gradient

LAYERS = [("RNN", eo.RNN, 1), ("LSTM", eo.LSTM, 4), ("GRU", eo.GRU, 3)]


class TestEmbedding:
    def test_lookup_shape(self):
        assert eo.Embedding(100, 16, seed=0)([[1, 5, 5]]).shape == (1, 3, 16)

    def test_rows_are_returned_verbatim(self, rng):
        table = eo.Embedding(10, 4, seed=0)
        weights = table.weight.numpy()
        np.testing.assert_allclose(table([3]).numpy(), weights[3][None, :])

    def test_repeated_ids_accumulate_gradient(self):
        table = eo.Embedding(10, 4, seed=0)
        eo.sum(table([[5, 5, 5]])).backward()
        np.testing.assert_allclose(table.weight.grad[5], np.full(4, 3.0))

    def test_untouched_rows_get_no_gradient(self):
        table = eo.Embedding(10, 4, seed=0)
        eo.sum(table([[2]])).backward()
        assert float(np.abs(table.weight.grad[7]).sum()) == 0.0

    def test_out_of_range_id(self):
        with pytest.raises(EveryOShapeError, match="out of range"):
            eo.Embedding(10, 4, seed=0)([[10]])

    def test_negative_id(self):
        with pytest.raises(EveryOShapeError, match="out of range"):
            eo.Embedding(10, 4, seed=0)([[-1]])

    def test_float_ids_are_rejected(self):
        with pytest.raises(EveryOShapeError, match="whole numbers"):
            eo.Embedding(10, 4, seed=0)(np.array([[1.5]]))

    def test_invalid_size(self):
        with pytest.raises(EveryOShapeError, match="positive sizes"):
            eo.Embedding(0, 4)

    def test_round_trip(self, tmp_path):
        table = eo.Embedding(20, 8, seed=0)
        restored = eo.load(eo.save(eo.Sequential(table), tmp_path / "e.evo"))
        np.testing.assert_allclose(restored[0].weight.numpy(), table.weight.numpy())


class TestRecurrentShapes:
    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_sequence_output(self, name, cls, gates):
        assert cls(8, 16, seed=0)(eo.zeros(4, 12, 8)).shape == (4, 12, 16)

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_last_step_only(self, name, cls, gates):
        layer = cls(8, 16, return_sequences=False, seed=0)
        assert layer(eo.zeros(4, 12, 8)).shape == (4, 16)

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_parameter_count(self, name, cls, gates):
        layer = cls(8, 16, seed=0)
        expected = gates * (8 * 16 + 16 * 16 + 16)
        assert layer.num_parameters() == expected

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_without_bias(self, name, cls, gates):
        layer = cls(8, 16, bias=False, seed=0)
        assert layer.num_parameters() == gates * (8 * 16 + 16 * 16)

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_single_timestep(self, name, cls, gates):
        assert cls(4, 6, seed=0)(eo.zeros(2, 1, 4)).shape == (2, 1, 6)

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_rank_validation(self, name, cls, gates):
        with pytest.raises(EveryOShapeError, match="3-D input"):
            cls(4, 6, seed=0)(eo.zeros(2, 4))

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_feature_validation(self, name, cls, gates):
        with pytest.raises(EveryOShapeError, match="last axis"):
            cls(4, 6, seed=0)(eo.zeros(2, 3, 9))

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_empty_sequence(self, name, cls, gates):
        with pytest.raises(EveryOShapeError, match="length 0"):
            cls(4, 6, seed=0)(eo.zeros(2, 0, 4))


class TestRecurrentBehaviour:
    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_output_depends_on_order(self, name, cls, gates, rng):
        """A recurrent layer must not be a bag of timesteps."""
        layer = cls(3, 4, return_sequences=False, seed=0)
        values = rng.normal(size=(1, 5, 3)).astype(np.float32)
        forward = layer(eo.tensor(values)).numpy()
        reversed_ = layer(eo.tensor(values[:, ::-1, :].copy())).numpy()
        assert not np.allclose(forward, reversed_)

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_gradient_reaches_the_first_timestep(self, name, cls, gates, rng):
        """Backpropagation through time must reach t=0, not stop partway."""
        layer = cls(4, 6, seed=0)
        values = eo.tensor(rng.normal(size=(2, 7, 4)).astype(np.float32), requires_grad=True)
        eo.sum(layer(values)).backward()
        assert values.grad is not None
        assert float(np.abs(values.grad[:, 0, :]).sum()) > 0

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_all_parameters_receive_gradient(self, name, cls, gates):
        layer = cls(4, 6, seed=0)
        eo.sum(layer(eo.ones(2, 5, 4))).backward()
        for parameter_name, parameter in layer.named_parameters():
            assert parameter.grad is not None, parameter_name
            assert float(np.abs(parameter.grad).sum()) > 0, parameter_name

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_gradient_matches_finite_differences(self, name, cls, gates, rng):
        layer = cls(3, 4, seed=0)
        weights = eo.tensor(rng.normal(size=(2, 5, 4)))

        def loss(t):
            return eo.sum(layer(t) * weights)

        values = rng.normal(size=(2, 5, 3))
        tensor = eo.tensor(values.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), values.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-4, atol=1e-6)

    def test_lstm_cell_state_enables_long_memory(self, rng):
        """The LSTM's additive cell path should not vanish as fast as an RNN's."""
        length = 40
        values = eo.tensor(rng.normal(size=(1, length, 4)).astype(np.float32), requires_grad=True)
        eo.sum(eo.LSTM(4, 8, return_sequences=False, seed=0)(values)).backward()
        lstm_first = float(np.abs(values.grad[:, 0, :]).sum())

        values2 = eo.tensor(values.numpy(), requires_grad=True)
        eo.sum(eo.RNN(4, 8, return_sequences=False, seed=0)(values2)).backward()
        rnn_first = float(np.abs(values2.grad[:, 0, :]).sum())

        assert lstm_first > 0
        assert lstm_first > rnn_first

    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_serialization_round_trip(self, name, cls, gates, tmp_path, rng):
        layer = cls(4, 6, seed=0)
        model = eo.Sequential(layer)
        values = eo.tensor(rng.normal(size=(2, 5, 4)).astype(np.float32))
        model.eval()
        expected = model(values).numpy()

        restored = eo.load(eo.save(model, tmp_path / f"{name}.evo"))
        np.testing.assert_allclose(restored(values).numpy(), expected, rtol=1e-6)
        assert type(restored[0]).__name__ == name


@pytest.mark.slow
class TestRecurrentTraining:
    @pytest.mark.parametrize("name,cls,gates", LAYERS)
    def test_learns_long_range_recall(self, name, cls, gates):
        """The label is the first token of the sequence — pure memory."""
        from everyo.datasets import make_recall_task

        sequences, labels = make_recall_task(
            n_samples=1200, length=12, num_classes=4, vocab_size=12, seed=0
        )
        x_train, x_test, y_train, y_test = eo.train_test_split(
            sequences, labels, test_size=0.2, seed=0
        )

        class Model(eo.Module):
            def __init__(self):
                super().__init__()
                self.embedding = eo.Embedding(12, 24, seed=0)
                self.core = cls(24, 24, return_sequences=False, seed=1)
                self.head = eo.Linear(24, 4, seed=2)

            def forward(self, ids):
                return self.head(self.core(self.embedding(ids)))

        model = Model()
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        trainer.fit(
            eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=64, shuffle=True, seed=0),
            epochs=15,
            verbose=False,
        )
        accuracy = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128))[
            "accuracy"
        ]
        assert accuracy > 0.85, f"{name} only reached {accuracy:.3f} (chance is 0.25)"
