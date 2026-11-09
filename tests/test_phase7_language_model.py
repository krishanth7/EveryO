"""Phase 7: tokenizer, causal decoder, cache, training and checkpoint gates."""

from __future__ import annotations

import numpy as np
import pytest

from everyo.language_model import (
    ByteBPETokenizer,
    CharTokenizer,
    TinyDecoderLM,
    next_token_loss,
    sample_next_token,
)
from everyo.optim import Adam
from everyo.serialization import load, save


def test_character_tokenizer_round_trip_and_unknowns():
    tokenizer = CharTokenizer("banana 🍌")
    assert tokenizer.decode(tokenizer.encode("banana 🍌")) == "banana 🍌"
    assert tokenizer.encode("?") == [1]
    assert tokenizer.decode([0, 1]) == "�"
    with pytest.raises(ValueError):
        tokenizer.decode([10000])


def test_byte_bpe_unicode_deterministic_and_compression():
    corpus = "banana banana banana café こんにちは " * 4
    a = ByteBPETokenizer(corpus, merges=32)
    b = ByteBPETokenizer(corpus, merges=32)
    assert a.rules == b.rules
    assert a.decode(a.encode("café こんにちは banana")) == "café こんにちは banana"
    assert len(a.encode("banana banana")) < len(b"banana banana")
    with pytest.raises(ValueError):
        ByteBPETokenizer("x", merges=-1)


def test_causal_future_isolation_and_shapes():
    model = TinyDecoderLM(20, embed_dim=8, num_heads=2, num_layers=2, seed=7)
    original = np.array([[1, 2, 3, 4]])
    modified = np.array([[1, 2, 15, 16]])
    a = model(original).data
    b = model(modified).data
    np.testing.assert_allclose(a[:, :2], b[:, :2], rtol=1e-5, atol=1e-5)
    assert a.shape == (1, 4, 20)
    with pytest.raises(ValueError):
        model(np.array([[100]]))
    with pytest.raises(TypeError):
        model(np.array([[1.2]]))


def test_incremental_cache_matches_full_causal_forward():
    model = TinyDecoderLM(22, 8, 2, 2, max_context=10, seed=9)
    tokens = np.array([[1, 4, 2, 3, 7], [2, 5, 8, 3, 6]])
    full = model(tokens).data
    cache = None
    for position in range(tokens.shape[1]):
        logits, cache = model.cached_step(tokens[:, position : position + 1], cache)
        np.testing.assert_allclose(logits, full[:, position, :], atol=2e-5, rtol=2e-5)
        assert all(layer.length == position + 1 for layer in cache)
    with pytest.raises(ValueError):
        model.cached_step(tokens[:, :1], cache[:1])


def test_generation_sampling_and_context_validation():
    model = TinyDecoderLM(9, 8, 2, 1, max_context=10, seed=21)
    prompt = np.array([[1, 2, 3]])
    cached = model.generate(prompt, max_new_tokens=5, top_k=1, seed=4)
    uncached = model.generate(prompt, max_new_tokens=5, top_k=1, seed=4, use_cache=False)
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_array_equal(cached, model.generate(prompt, max_new_tokens=5, top_k=1, seed=4))
    assert model.generate(prompt, max_new_tokens=0).shape == (1, 3)
    choice = sample_next_token([1.0, 2.0, 3.0], rng=np.random.default_rng(1), top_k=1)
    assert choice == 2
    for parameters in ({"temperature": 0}, {"top_k": 0}, {"top_p": 1.1}):
        with pytest.raises(ValueError):
            sample_next_token([0.1, 0.2], rng=np.random.default_rng(1), **parameters)
    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=8)


def test_cpu_training_and_evo_checkpoint_roundtrip(tmp_path):
    model = TinyDecoderLM(6, 8, 2, 1, max_context=8, seed=13)
    inputs = np.array([[1, 2, 3, 4], [2, 3, 4, 1]])
    targets = np.array([[2, 3, 4, 1], [3, 4, 1, 2]])
    initial = float(next_token_loss(model(inputs), targets).item())
    optimizer = Adam(model.parameters(), lr=0.02)
    for _ in range(24):
        optimizer.zero_grad()
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final = float(next_token_loss(model(inputs), targets).item())
    assert np.isfinite(final) and final < initial * 0.8, (initial, final)
    path = tmp_path / "tiny.evo"
    save(model, path)
    restored = load(path)
    assert isinstance(restored, TinyDecoderLM)
    np.testing.assert_allclose(restored(inputs).data, model(inputs).data, atol=1e-6)
