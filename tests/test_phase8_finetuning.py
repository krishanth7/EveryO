"""Phase 8 CPU-only fine-tuning acceptance and regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from everyo.core.tensor import tensor
from everyo.finetuning import (
    LoRALinear,
    LoRATinyDecoderLM,
    evaluate_instructions,
    fine_tune_instructions,
    gradient_checkpoint,
    masked_next_token_loss,
    prepare_instruction_batch,
)
from everyo.language_model import CharTokenizer, TinyDecoderLM
from everyo.nn import Linear
from everyo.serialization import load, save


def _data():
    pairs = [("a", "b"), ("b", "a")]
    tokenizer = CharTokenizer("Instruction: a\nAnswer: bInstruction: b\nAnswer: a")
    return tokenizer, prepare_instruction_batch(pairs, tokenizer, max_context=32)


def test_lora_zero_delta_and_frozen_base():
    base = Linear(4, 3, seed=2)
    before = base(np.ones((2, 4))).data.copy()
    layer = LoRALinear(base, rank=2, alpha=4)
    np.testing.assert_allclose(layer(np.ones((2, 4))).data, before, atol=1e-7)
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert layer.num_parameters(trainable_only=True) == 14
    layer.lora_b.data = np.ones_like(layer.lora_b.data)
    np.testing.assert_allclose(
        layer.merged_weight(), base.weight.data + 2 * (layer.lora_a.data @ layer.lora_b.data)
    )
    with pytest.raises(ValueError):
        LoRALinear(base, rank=0)


def test_instruction_mask_and_missing_characters():
    tokenizer, batch = _data()
    assert batch.inputs.shape == batch.targets.shape == batch.loss_mask.shape
    assert batch.loss_mask.sum() == 2
    assert batch.targets[0, np.flatnonzero(batch.loss_mask[0])[-1]] == tokenizer.encode("b")[0]
    with pytest.raises(ValueError, match="outside tokenizer"):
        prepare_instruction_batch([("not-in-corpus-?", "b")], tokenizer, max_context=64)
    with pytest.raises(ValueError, match="exceeds"):
        prepare_instruction_batch([("a", "b")], tokenizer, max_context=3)


def test_masked_loss_gradients_only_supervised_tokens():
    logits = tensor(np.array([[[2.0, 0.0], [1.0, 1.0]]], dtype=np.float32), requires_grad=True)
    loss = masked_next_token_loss(logits, np.array([[0, 1]]), np.array([[False, True]]))
    loss.backward()
    np.testing.assert_array_equal(logits.grad[0, 0], np.zeros(2))
    assert np.any(logits.grad[0, 1] != 0)
    with pytest.raises(ValueError, match="no supervised"):
        masked_next_token_loss(logits, np.array([[0, 1]]), np.zeros((1, 2), bool))


def test_checkpoint_recomputed_gradients_match_direct():
    direct = Linear(3, 2, seed=4)
    checkpointed = Linear(3, 2, seed=4)
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x1 = tensor(data, requires_grad=True)
    x2 = tensor(data, requires_grad=True)
    direct(x1).backward(np.ones((1, 2), np.float32))
    gradient_checkpoint(checkpointed, x2, checkpointed.parameters()).backward(
        np.ones((1, 2), np.float32)
    )
    np.testing.assert_allclose(x1.grad, x2.grad, atol=1e-6)
    for a, b in zip(direct.parameters(), checkpointed.parameters()):
        np.testing.assert_allclose(a.grad, b.grad, atol=1e-6)


def test_lora_cpu_training_freeze_and_serialization(tmp_path):
    tokenizer, batch = _data()
    model = LoRATinyDecoderLM(
        tokenizer.vocab_size,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        max_context=32,
        rank=2,
        seed=11,
    )
    frozen = {
        name: param.data.copy()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }
    report = fine_tune_instructions(model, batch, steps=25, lr=0.07, gradient_accumulation=2)
    assert report.trainable_parameters == model.num_parameters(trainable_only=True)
    assert np.isfinite(report.final_loss) and report.final_loss < report.initial_loss
    assert report.final_loss == pytest.approx(evaluate_instructions(model, batch))
    for name, parameter in model.named_parameters():
        if name in frozen:
            np.testing.assert_array_equal(parameter.data, frozen[name])
    save(model, tmp_path / "lora.evo")
    restored = load(tmp_path / "lora.evo")
    assert isinstance(restored, LoRATinyDecoderLM)
    np.testing.assert_allclose(restored(batch.inputs).data, model(batch.inputs).data)


def test_copy_base_model_matches_initial_logits():
    original = TinyDecoderLM(17, 8, 2, 1, 32, seed=3)
    adapted = LoRATinyDecoderLM(17, 8, 2, 1, 32, seed=99)
    adapted.load_base(original)
    ids = np.array([[1, 2, 3, 4]])
    np.testing.assert_allclose(original(ids).data, adapted(ids).data, atol=1e-6)
    with pytest.raises(ValueError):
        adapted.load_base(TinyDecoderLM(16, 8, 2))
