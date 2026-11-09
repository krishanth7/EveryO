"""Phase 8: deterministic LoRA instruction fine-tuning, entirely on CPU.

Run: python examples/cpu_instruction_finetune.py --steps 40
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from everyo.finetuning import (
    LoRATinyDecoderLM,
    fine_tune_instructions,
    prepare_instruction_batch,
)
from everyo.language_model import CharTokenizer
from everyo.serialization import load, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", type=Path, help="Optional .evo checkpoint")
    args = parser.parse_args()
    pairs = [("a", "b"), ("b", "a")]
    corpus = "".join(f"Instruction: {prompt}\nAnswer: {answer}" for prompt, answer in pairs)
    tokenizer = CharTokenizer(corpus)
    data = prepare_instruction_batch(pairs, tokenizer, max_context=32)
    model = LoRATinyDecoderLM(
        tokenizer.vocab_size, embed_dim=8, num_heads=2, max_context=32, rank=2, seed=7
    )
    report = fine_tune_instructions(model, data, steps=args.steps, lr=0.06)
    print(f"Trainable parameters: {report.trainable_parameters}")
    print(f"Supervised loss: {report.initial_loss:.4f} -> {report.final_loss:.4f}")

    if args.output is None:
        with TemporaryDirectory(prefix="everyo-sft-") as directory:
            destination = Path(directory) / "adapter.evo"
            save(model, destination)
            restored = load(destination)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(model, args.output)
        restored = load(args.output)
    np.testing.assert_allclose(
        model(data.inputs).data, restored(data.inputs).data, rtol=1e-6, atol=1e-6
    )
    print("LoRA checkpoint round trip: PASS")


if __name__ == "__main__":
    main()
