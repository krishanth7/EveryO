"""Phase 9 CPU-native image, audio, text features and trainable fusion.

Feature extractors deliberately use small deterministic descriptors. This is
an educational foundation, not pretrained visual or speech recognition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from everyo.core import operations as ops
from everyo.core.autograd import no_grad
from everyo.core.tensor import Tensor, as_tensor
from everyo.nn.activations import ReLU
from everyo.nn.layers import Linear
from everyo.nn.losses import CrossEntropyLoss
from everyo.nn.module import Module, register_module

__all__ = [
    "image_patch_features",
    "audio_frame_features",
    "text_hash_features",
    "MultimodalBatch",
    "prepare_multimodal_batch",
    "TinyMultimodalNet",
    "contrastive_alignment_loss",
    "multimodal_loss",
    "evaluate_multimodal",
]


def image_patch_features(images: Any) -> np.ndarray:
    """2x2 spatial averages per RGB channel; output (batch,12)."""
    x = np.asarray(images)
    if (
        x.ndim != 4
        or x.shape[0] < 1
        or x.shape[1] < 2
        or x.shape[2] < 2
        or x.shape[3] not in (1, 3)
    ):
        raise ValueError("images must have shape (B,H>=2,W>=2,C=1|3)")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("images must have finite numeric values")
    x = x.astype(np.float32)
    if x.min() < 0 or x.max() > 1:
        raise ValueError("images must be normalized to [0,1]")
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    mid_h, mid_w = x.shape[1] // 2, x.shape[2] // 2
    patches = (
        x[:, :mid_h, :mid_w],
        x[:, :mid_h, mid_w:],
        x[:, mid_h:, :mid_w],
        x[:, mid_h:, mid_w:],
    )
    return np.concatenate([patch.mean(axis=(1, 2)) for patch in patches], axis=1)


def audio_frame_features(waveforms: Any, *, sample_rate: int = 16000) -> np.ndarray:
    """RMS and zero-crossing rate in four frames; output (batch,8)."""
    audio = np.asarray(waveforms)
    if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] < 8 or sample_rate < 1:
        raise ValueError("audio must be (batch, at least 8 samples), with positive sample_rate")
    if not np.issubdtype(audio.dtype, np.number) or not np.isfinite(audio).all():
        raise ValueError("audio must be finite and numeric")
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1:
        raise ValueError("audio must be normalized to [-1,1]")
    frames = np.array_split(audio, 4, axis=1)
    features = []
    for frame in frames:
        rms = np.sqrt(np.mean(frame * frame, axis=1))
        crosses = np.mean((frame[:, 1:] >= 0) != (frame[:, :-1] >= 0), axis=1)
        features.extend((rms, crosses))
    return np.stack(features, axis=1).astype(np.float32)


def text_hash_features(texts: list[str] | tuple[str, ...], *, size: int = 16) -> np.ndarray:
    """Deterministic UTF-8 byte-frequency feature vectors (default width 16)."""
    if not texts or size < 2 or any(not isinstance(text, str) for text in texts):
        raise ValueError("texts must be a nonempty sequence of strings, size >= 2")
    features = np.zeros((len(texts), size), dtype=np.float32)
    for index, text in enumerate(texts):
        payload = text.encode("utf-8")
        for byte in payload:
            features[index, byte % size] += 1
        if payload:
            features[index] /= len(payload)
    return features


@dataclass(frozen=True)
class MultimodalBatch:
    """Aligned image, audio, text descriptors with integer class labels."""

    images: np.ndarray
    audio: np.ndarray
    text: np.ndarray
    labels: np.ndarray


def prepare_multimodal_batch(
    images: Any, audio: Any, texts: list[str], labels: Any
) -> MultimodalBatch:
    image = image_patch_features(images)
    sound = audio_frame_features(audio)
    language = text_hash_features(texts)
    targets = np.asarray(labels)
    if targets.shape != (image.shape[0],) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels must be integer class indices of shape (batch,)")
    if sound.shape[0] != image.shape[0] or language.shape[0] != image.shape[0]:
        raise ValueError("image, audio and text batch sizes must match")
    return MultimodalBatch(image, sound, language, targets.astype(np.int64))


@register_module
class TinyMultimodalNet(Module):
    """Trainable independent modality projections plus late fusion classifier."""

    def __init__(self, classes: int = 2, embed_dim: int = 8, seed: int = 0):
        super().__init__()
        if classes < 2 or embed_dim < 1:
            raise ValueError("classes >= 2 and positive embed_dim are required")
        self.classes, self.embed_dim, self.seed = int(classes), int(embed_dim), int(seed)
        self.image_encoder = Linear(12, embed_dim, seed=seed)
        self.audio_encoder = Linear(8, embed_dim, seed=seed + 1)
        self.text_encoder = Linear(16, embed_dim, seed=seed + 2)
        self.activation = ReLU()
        self.fusion = Linear(embed_dim * 3, embed_dim, seed=seed + 3)
        self.head = Linear(embed_dim, classes, seed=seed + 4)

    def encode(self, batch: MultimodalBatch) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, MultimodalBatch):
            raise TypeError("expected MultimodalBatch")
        expected = ((batch.images, 12), (batch.audio, 8), (batch.text, 16))
        if any(x.ndim != 2 or x.shape[1] != width for x, width in expected):
            raise ValueError("multimodal feature dimensions are invalid")
        return (
            self.activation(self.image_encoder(batch.images)),
            self.activation(self.audio_encoder(batch.audio)),
            self.activation(self.text_encoder(batch.text)),
        )

    def forward(self, batch: MultimodalBatch) -> Tensor:
        vectors = self.encode(batch)
        fused = ops.concatenate(vectors, axis=-1)
        return self.head(self.activation(self.fusion(fused)))

    def get_config(self) -> dict[str, Any]:
        return {"classes": self.classes, "embed_dim": self.embed_dim, "seed": self.seed}


def multimodal_loss(logits: Tensor, labels: Any) -> Tensor:
    targets = np.asarray(labels)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("expected class logits (batch,classes) and labels (batch,)")
    return CrossEntropyLoss()(logits, targets)


def contrastive_alignment_loss(a: Tensor, b: Tensor, *, temperature: float = 0.1) -> Tensor:
    """Symmetric image-text/audio-text InfoNCE for an aligned batch.

    Batch order defines positive pairs, and all other examples are negatives.
    """
    left, right = as_tensor(a), as_tensor(b)
    if left.ndim != 2 or right.shape != left.shape or left.shape[0] < 2:
        raise ValueError("paired embeddings require matching (batch>=2,dim) arrays")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    left_norm = ops.sqrt(ops.sum(left * left, axis=1, keepdims=True) + 1e-8)
    right_norm = ops.sqrt(ops.sum(right * right, axis=1, keepdims=True) + 1e-8)
    scores = ops.matmul(left / left_norm, ops.transpose(right / right_norm)) / temperature
    labels = np.arange(left.shape[0], dtype=np.int64)
    return (
        CrossEntropyLoss()(scores, labels) + CrossEntropyLoss()(ops.transpose(scores), labels)
    ) / 2


def evaluate_multimodal(model: TinyMultimodalNet, batch: MultimodalBatch) -> dict[str, float]:
    """No-grad deterministic loss and accuracy on the supplied aligned batch."""
    previous = model.training
    model.eval()
    try:
        with no_grad():
            logits = model(batch)
            loss = float(multimodal_loss(logits, batch.labels).item())
            accuracy = float(np.mean(np.argmax(logits.data, axis=1) == batch.labels))
            return {"loss": loss, "accuracy": accuracy}
    finally:
        model.train(previous)
