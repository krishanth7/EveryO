"""Phase 9 CPU multimodal feature, training and serialization gates."""

import numpy as np
import pytest

from everyo.core.tensor import tensor
from everyo.multimodal import (
    TinyMultimodalNet,
    audio_frame_features,
    contrastive_alignment_loss,
    evaluate_multimodal,
    image_patch_features,
    multimodal_loss,
    prepare_multimodal_batch,
    text_hash_features,
)
from everyo.optim import Adam
from everyo.serialization import load, save


def _batch():
    images = np.zeros((4, 4, 4, 3), dtype=np.float32)
    images[:2] = 1
    waves = np.zeros((4, 32), dtype=np.float32)
    waves[:2, ::2] = 0.8
    return prepare_multimodal_batch(
        images, waves, ["bright", "bright", "dark", "dark"], np.array([1, 1, 0, 0])
    )


def test_image_grayscale_rgb_and_validation():
    gray = np.ones((2, 5, 5, 1), dtype=np.float32)
    features = image_patch_features(gray)
    assert features.shape == (2, 12)
    np.testing.assert_allclose(features, 1)
    with pytest.raises(ValueError):
        image_patch_features(np.ones((1, 4, 4, 3)) * 255)
    with pytest.raises(ValueError):
        image_patch_features(np.ones((4, 4, 3)))


def test_audio_features_and_validation():
    silence = audio_frame_features(np.zeros((3, 16)))
    assert silence.shape == (3, 8)
    np.testing.assert_array_equal(silence, 0)
    with pytest.raises(ValueError):
        audio_frame_features(np.ones((1, 3)))
    with pytest.raises(ValueError):
        audio_frame_features(np.full((1, 16), np.nan))


def test_text_hash_is_deterministic_unicode():
    a = text_hash_features(["hello", "నమస్తే"])
    assert a.shape == (2, 16)
    np.testing.assert_array_equal(a, text_hash_features(["hello", "నమస్తే"]))
    np.testing.assert_allclose(a.sum(axis=1), 1)
    with pytest.raises(ValueError):
        text_hash_features([])


def test_alignment_and_batch_validation():
    batch = _batch()
    assert batch.images.shape == (4, 12)
    assert batch.audio.shape == (4, 8)
    assert batch.text.shape == (4, 16)
    with pytest.raises(ValueError):
        prepare_multimodal_batch(np.ones((2, 4, 4, 3)), np.zeros((1, 16)), ["a", "b"], [0, 1])


def test_contrastive_alignment_gradient():
    a = tensor(np.eye(3, dtype=np.float32), requires_grad=True)
    b = tensor(np.eye(3, dtype=np.float32), requires_grad=True)
    value = contrastive_alignment_loss(a, b)
    assert value.item() < 0.01
    value.backward()
    assert np.all(np.isfinite(a.grad)) and np.all(np.isfinite(b.grad))
    with pytest.raises(ValueError):
        contrastive_alignment_loss(a, b, temperature=0)


def test_cpu_train_and_evo_roundtrip(tmp_path):
    batch = _batch()
    model = TinyMultimodalNet(classes=2, embed_dim=8, seed=12)
    optimizer = Adam(model.parameters(), lr=0.025)
    first = evaluate_multimodal(model, batch)["loss"]
    for _ in range(35):
        optimizer.zero_grad()
        loss = multimodal_loss(model(batch), batch.labels)
        loss.backward()
        optimizer.step()
    result = evaluate_multimodal(model, batch)
    assert result["loss"] < first
    assert result["accuracy"] == 1
    path = tmp_path / "multimodal.evo"
    save(model, path)
    restored = load(path)
    assert isinstance(restored, TinyMultimodalNet)
    np.testing.assert_allclose(restored(batch).data, model(batch).data, atol=1e-6)
