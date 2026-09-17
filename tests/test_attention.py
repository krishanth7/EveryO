"""Tests for attention, positional encoding and transformer blocks."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.backends import tensorflow_backend as tfb
from everyo.exceptions import EveryOShapeError
from everyo.nn.attention import MASK_VALUE
from tests.conftest import numeric_gradient

requires_tensorflow = pytest.mark.skipif(
    not tfb.is_available(), reason="TensorFlow is not installed."
)


class TestScaledDotProductAttention:
    def test_output_and_weight_shapes(self, rng):
        q = eo.tensor(rng.normal(size=(2, 5, 8)).astype(np.float32))
        k = eo.tensor(rng.normal(size=(2, 7, 8)).astype(np.float32))
        v = eo.tensor(rng.normal(size=(2, 7, 4)).astype(np.float32))
        out, weights = eo.scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 5, 4)
        assert weights.shape == (2, 5, 7)

    def test_weights_form_a_distribution(self, rng):
        q = eo.tensor(rng.normal(size=(2, 5, 8)).astype(np.float32))
        _, weights = eo.scaled_dot_product_attention(q, q, q)
        values = weights.numpy()
        np.testing.assert_allclose(values.sum(axis=-1), np.ones((2, 5)), rtol=1e-5)
        assert values.min() >= 0.0

    def test_identical_keys_give_uniform_attention(self):
        q = eo.ones(1, 1, 4)
        k = eo.ones(1, 3, 4)
        v = eo.tensor(np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32))
        out, weights = eo.scaled_dot_product_attention(q, k, v)
        np.testing.assert_allclose(weights.numpy(), np.full((1, 1, 3), 1 / 3), rtol=1e-5)
        assert out.item() == pytest.approx(2.0, rel=1e-5)

    def test_attention_selects_the_matching_key(self):
        """A query aligned with one key should retrieve that key's value."""
        q = eo.tensor(np.array([[[10.0, 0.0]]], dtype=np.float32))
        k = eo.tensor(np.array([[[10.0, 0.0], [0.0, 10.0]]], dtype=np.float32))
        v = eo.tensor(np.array([[[1.0], [99.0]]], dtype=np.float32))
        out, weights = eo.scaled_dot_product_attention(q, k, v)
        assert weights.numpy()[0, 0, 0] > 0.99
        assert out.item() == pytest.approx(1.0, abs=0.5)

    def test_depth_mismatch(self):
        with pytest.raises(EveryOShapeError, match="query and key depths"):
            eo.scaled_dot_product_attention(eo.zeros(1, 2, 4), eo.zeros(1, 2, 8), eo.zeros(1, 2, 8))

    def test_key_value_length_mismatch(self):
        with pytest.raises(EveryOShapeError, match="as many keys as values"):
            eo.scaled_dot_product_attention(eo.zeros(1, 2, 4), eo.zeros(1, 3, 4), eo.zeros(1, 5, 4))

    def test_gradient_matches_finite_differences(self, rng):
        weights = eo.tensor(rng.normal(size=(1, 4, 6)))
        constant_k = eo.tensor(rng.normal(size=(1, 4, 6)))
        constant_v = eo.tensor(rng.normal(size=(1, 4, 6)))

        def loss(t):
            out, _ = eo.scaled_dot_product_attention(t, constant_k, constant_v)
            return eo.sum(out * weights)

        values = rng.normal(size=(1, 4, 6))
        tensor = eo.tensor(values.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), values.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-4, atol=1e-6)

    @requires_tensorflow
    @pytest.mark.tensorflow
    def test_matches_a_tensorflow_reference(self, rng):
        import tensorflow as tf

        q = rng.normal(size=(2, 5, 8)).astype(np.float32)
        ours, _ = eo.scaled_dot_product_attention(eo.tensor(q), eo.tensor(q), eo.tensor(q))
        scores = tf.matmul(q, q, transpose_b=True) / np.sqrt(8.0)
        theirs = tf.matmul(tf.nn.softmax(scores, axis=-1), q).numpy()
        np.testing.assert_allclose(ours.numpy(), theirs, rtol=1e-4, atol=1e-6)


class TestMasks:
    def test_causal_mask_shape_and_content(self):
        mask = eo.causal_mask(4)
        assert mask.shape == (1, 1, 4, 4)
        assert mask[0, 0, 0, 0] == 0.0  # may see itself
        assert mask[0, 0, 0, 1] == MASK_VALUE  # may not see the future
        assert mask[0, 0, 3, 0] == 0.0  # may see the past

    def test_causal_mask_blocks_the_future(self, rng):
        attention = eo.MultiHeadAttention(8, 2, seed=0)
        values = eo.tensor(rng.normal(size=(1, 6, 8)).astype(np.float32))
        attention(values, mask=eo.causal_mask(6))
        weights = attention.last_attention_weights.numpy()
        assert float(np.triu(weights[0, 0], k=1).max()) == 0.0

    def test_causal_attention_ignores_later_tokens(self, rng):
        """Changing a later token must not change an earlier position's output."""
        attention = eo.MultiHeadAttention(8, 2, seed=0)
        values = rng.normal(size=(1, 5, 8)).astype(np.float32)
        mask = eo.causal_mask(5)
        first = attention(eo.tensor(values), mask=mask).numpy()

        altered = values.copy()
        altered[0, 4] += 10.0
        second = attention(eo.tensor(altered), mask=mask).numpy()
        np.testing.assert_allclose(first[:, :4], second[:, :4], rtol=1e-5)

    def test_padding_mask_blocks_padding(self):
        attention = eo.MultiHeadAttention(8, 2, seed=0)
        attention(eo.zeros(1, 6, 8), mask=eo.padding_mask([3], 6))
        weights = attention.last_attention_weights.numpy()
        assert float(weights[0, :, :, 3:].max()) == 0.0

    def test_padding_mask_validation(self):
        with pytest.raises(EveryOShapeError, match="lengths must lie"):
            eo.padding_mask([9], 4)

    def test_causal_mask_validation(self):
        with pytest.raises(EveryOShapeError, match="positive size"):
            eo.causal_mask(0)


class TestMultiHeadAttention:
    def test_output_shape_and_parameters(self):
        attention = eo.MultiHeadAttention(32, 4, seed=0)
        assert attention(eo.zeros(2, 10, 32)).shape == (2, 10, 32)
        # four projections, each 32x32 plus a 32 bias
        assert attention.num_parameters() == 4 * (32 * 32 + 32)

    def test_head_dimension(self):
        assert eo.MultiHeadAttention(32, 8, seed=0).head_dim == 4

    def test_heads_must_divide_the_width(self):
        with pytest.raises(EveryOShapeError, match="divide evenly"):
            eo.MultiHeadAttention(30, 4)

    def test_rank_validation(self):
        with pytest.raises(EveryOShapeError, match=r"\(batch, time, embed_dim\)"):
            eo.MultiHeadAttention(8, 2, seed=0)(eo.zeros(4, 8))

    def test_width_validation(self):
        with pytest.raises(EveryOShapeError, match="last axis"):
            eo.MultiHeadAttention(8, 2, seed=0)(eo.zeros(2, 5, 16))

    def test_cross_attention_shape(self):
        attention = eo.MultiHeadAttention(16, 2, seed=0)
        out = attention(eo.zeros(2, 5, 16), eo.zeros(2, 9, 16))
        assert out.shape == (2, 5, 16)
        assert attention.last_attention_weights.shape == (2, 2, 5, 9)

    def test_weights_are_exposed_per_head(self, rng):
        attention = eo.MultiHeadAttention(16, 4, seed=0)
        attention(eo.tensor(rng.normal(size=(3, 7, 16)).astype(np.float32)))
        assert attention.last_attention_weights.shape == (3, 4, 7, 7)

    def test_permuting_the_sequence_permutes_the_output(self, rng):
        """Attention alone is order-agnostic; that is why positions are encoded."""
        attention = eo.MultiHeadAttention(8, 2, seed=0)
        values = rng.normal(size=(1, 4, 8)).astype(np.float32)
        straight = attention(eo.tensor(values)).numpy()
        order = [2, 0, 3, 1]
        shuffled = attention(eo.tensor(values[:, order, :].copy())).numpy()
        np.testing.assert_allclose(shuffled, straight[:, order, :], rtol=1e-4, atol=1e-6)

    def test_gradient_matches_finite_differences(self, rng):
        attention = eo.MultiHeadAttention(8, 2, seed=0)
        weights = eo.tensor(rng.normal(size=(2, 5, 8)))

        def loss(t):
            return eo.sum(attention(t) * weights)

        values = rng.normal(size=(2, 5, 8))
        tensor = eo.tensor(values.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), values.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-4, atol=1e-6)

    def test_all_projections_receive_gradient(self):
        attention = eo.MultiHeadAttention(8, 2, seed=0)
        eo.sum(attention(eo.ones(2, 4, 8))).backward()
        for name, parameter in attention.named_parameters():
            assert parameter.grad is not None, name


class TestPositionalEncoding:
    def test_shape_is_preserved(self):
        assert eo.PositionalEncoding(32)(eo.zeros(2, 10, 32)).shape == (2, 10, 32)

    def test_encoding_is_a_buffer_not_a_parameter(self):
        layer = eo.PositionalEncoding(16)
        assert layer.parameters() == []
        assert len(layer.buffers()) == 1

    def test_positions_differ(self):
        layer = eo.PositionalEncoding(16)
        encoding = layer.encoding
        assert not np.allclose(encoding[0], encoding[1])

    def test_values_are_bounded(self):
        encoding = eo.PositionalEncoding(32, max_length=100).encoding
        assert encoding.max() <= 1.0 and encoding.min() >= -1.0

    def test_it_breaks_permutation_invariance(self, rng):
        """With positions added, a shuffled sequence is genuinely different."""
        encoder = eo.Sequential(eo.PositionalEncoding(8), eo.MultiHeadAttention(8, 2, seed=0))
        values = rng.normal(size=(1, 4, 8)).astype(np.float32)
        order = [2, 0, 3, 1]
        straight = encoder(eo.tensor(values)).numpy()
        shuffled = encoder(eo.tensor(values[:, order, :].copy())).numpy()
        assert not np.allclose(shuffled, straight[:, order, :], rtol=1e-3)

    def test_sequence_longer_than_the_table(self):
        with pytest.raises(EveryOShapeError, match="exceeds max_length"):
            eo.PositionalEncoding(8, max_length=4)(eo.zeros(1, 5, 8))

    def test_encoding_survives_serialization(self, tmp_path):
        model = eo.Sequential(eo.PositionalEncoding(8, max_length=20))
        restored = eo.load(eo.save(model, tmp_path / "pe.evo"))
        np.testing.assert_allclose(restored[0].encoding, model[0].encoding)


class TestTransformerBlocks:
    def test_block_preserves_shape(self):
        assert eo.TransformerEncoderBlock(32, 4, seed=0)(eo.zeros(2, 10, 32)).shape == (2, 10, 32)

    def test_encoder_stack_preserves_shape(self):
        encoder = eo.TransformerEncoder(32, 4, num_layers=3, seed=0)
        assert encoder(eo.zeros(2, 10, 32)).shape == (2, 10, 32)
        assert len(encoder.blocks) == 3

    def test_stacking_multiplies_parameters(self):
        one = eo.TransformerEncoder(32, 4, num_layers=1, seed=0).num_parameters()
        two = eo.TransformerEncoder(32, 4, num_layers=2, seed=0).num_parameters()
        assert two > one

    def test_residual_path_is_the_identity_when_sublayers_output_zero(self, rng):
        """Silence both sublayers and a pre-norm block must pass its input through.

        This is the residual connection stated exactly: output = input +
        attention(...) + feedforward(...). Zeroing the two output projections
        removes both contributions, so anything other than the identity would
        mean the skip connection is missing or misplaced.
        """
        block = eo.TransformerEncoderBlock(16, 2, dropout=0.0, seed=0)
        block.attention.output_projection.weight.data = np.zeros((16, 16), dtype=np.float32)
        block.attention.output_projection.bias.data = np.zeros(16, dtype=np.float32)
        block.linear_out.weight.data = np.zeros((block.feedforward_dim, 16), dtype=np.float32)
        block.linear_out.bias.data = np.zeros(16, dtype=np.float32)

        values = rng.normal(size=(2, 6, 16)).astype(np.float32)
        np.testing.assert_allclose(block(eo.tensor(values)).numpy(), values, rtol=1e-5, atol=1e-6)

    def test_input_information_reaches_the_output(self, rng):
        """The skip connection should keep the output correlated with the input."""
        block = eo.TransformerEncoderBlock(16, 2, dropout=0.0, seed=0)
        values = rng.normal(size=(2, 6, 16)).astype(np.float32)
        out = block(eo.tensor(values)).numpy()
        correlation = float(np.corrcoef(values.ravel(), out.ravel())[0, 1])
        assert correlation > 0.3, f"input barely survives the block (r={correlation:.2f})"

    def test_post_norm_variant(self):
        block = eo.TransformerEncoderBlock(16, 2, norm_first=False, seed=0)
        assert block(eo.zeros(2, 5, 16)).shape == (2, 5, 16)
        assert block.get_config()["norm_first"] is False

    def test_mask_is_forwarded_to_attention(self, rng):
        block = eo.TransformerEncoderBlock(16, 2, dropout=0.0, seed=0)
        block(eo.tensor(rng.normal(size=(1, 5, 16)).astype(np.float32)), mask=eo.causal_mask(5))
        weights = block.attention.last_attention_weights.numpy()
        assert float(np.triu(weights[0, 0], k=1).max()) == 0.0

    def test_dropout_is_disabled_in_eval_mode(self, rng):
        block = eo.TransformerEncoderBlock(16, 2, dropout=0.5, seed=0).eval()
        values = eo.tensor(rng.normal(size=(2, 5, 16)).astype(np.float32))
        np.testing.assert_allclose(block(values).numpy(), block(values).numpy(), rtol=1e-6)

    def test_gradient_matches_finite_differences(self, rng):
        block = eo.TransformerEncoderBlock(8, 2, dropout=0.0, seed=0)
        weights = eo.tensor(rng.normal(size=(2, 5, 8)))

        def loss(t):
            return eo.sum(block(t) * weights)

        values = rng.normal(size=(2, 5, 8))
        tensor = eo.tensor(values.copy(), requires_grad=True)
        loss(tensor).backward()
        expected = numeric_gradient(lambda a: loss(eo.tensor(a.copy())).item(), values.copy())
        np.testing.assert_allclose(tensor.grad, expected, rtol=1e-4, atol=1e-6)

    def test_every_parameter_trains(self):
        block = eo.TransformerEncoderBlock(8, 2, dropout=0.0, seed=0)
        eo.sum(block(eo.ones(2, 4, 8))).backward()
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, name

    def test_serialization_round_trip(self, tmp_path, rng):
        model = eo.Sequential(eo.TransformerEncoder(16, 2, num_layers=2, dropout=0.0, seed=0))
        values = eo.tensor(rng.normal(size=(2, 6, 16)).astype(np.float32))
        model.eval()
        expected = model(values).numpy()

        restored = eo.load(eo.save(model, tmp_path / "t.evo"))
        np.testing.assert_allclose(restored(values).numpy(), expected, rtol=1e-6)
        assert restored[0].num_layers == 2


@pytest.mark.slow
class TestTransformerTraining:
    def test_transformer_learns_long_range_recall(self):
        from everyo.datasets import make_recall_task

        sequences, labels = make_recall_task(
            n_samples=1200, length=16, num_classes=4, vocab_size=12, seed=0
        )
        x_train, x_test, y_train, y_test = eo.train_test_split(
            sequences, labels, test_size=0.2, seed=0
        )

        class Model(eo.Module):
            def __init__(self):
                super().__init__()
                self.embedding = eo.Embedding(12, 32, seed=0)
                self.positional = eo.PositionalEncoding(32)
                self.encoder = eo.TransformerEncoder(32, 4, num_layers=1, dropout=0.0, seed=1)
                self.head = eo.Linear(32, 4, seed=2)

            def forward(self, ids):
                hidden = self.encoder(self.positional(self.embedding(ids)))
                return self.head(eo.mean(hidden, axis=1))

        model = Model()
        trainer = eo.Trainer(
            model, eo.Adam(model.parameters(), lr=0.01), eo.CrossEntropyLoss(), metrics=["accuracy"]
        )
        trainer.fit(
            eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=64, shuffle=True, seed=0),
            epochs=12,
            verbose=False,
        )
        accuracy = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=128))[
            "accuracy"
        ]
        assert accuracy > 0.85, f"transformer only reached {accuracy:.3f} (chance is 0.25)"
