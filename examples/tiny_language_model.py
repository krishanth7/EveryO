"""Train, sample and safely reload a miniature EveryO language model on CPU.

Run: python examples/tiny_language_model.py --steps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from everyo.language_model import CharTokenizer, TinyDecoderLM, next_token_loss
from everyo.optim import Adam
from everyo.serialization import load, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Optional destination for the .evo model")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    corpus = "the cat sat. the cat ate. " * 8
    tokenizer = CharTokenizer(corpus)
    tokens = np.asarray(tokenizer.encode(corpus), dtype=np.int64)
    width = 12
    starts = np.linspace(0, len(tokens) - width - 2, num=8, dtype=int)
    inputs = np.stack([tokens[start : start + width] for start in starts])
    targets = np.stack([tokens[start + 1 : start + width + 1] for start in starts])
    model = TinyDecoderLM(
        tokenizer.vocab_size, embed_dim=12, num_heads=3, num_layers=1, max_context=24, seed=42
    )
    optimizer = Adam(model.parameters(), lr=0.015)
    first_loss = float(next_token_loss(model(inputs), targets).item())
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    last_loss = float(next_token_loss(model(inputs), targets).item())

    prompt = np.array([tokenizer.encode("the ")])
    completion = model.generate(prompt, max_new_tokens=12, top_k=3, seed=7)
    print(f"CPU training: {first_loss:.4f} -> {last_loss:.4f} loss")
    print(f"Sample: {tokenizer.decode(completion[0])!r}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(model, args.output)
        restored = load(args.output)
    else:
        with TemporaryDirectory(prefix="everyo-lm-") as directory:
            archive = Path(directory) / "tiny.evo"
            save(model, archive)
            restored = load(archive)
    np.testing.assert_allclose(restored(inputs).data, model(inputs).data, atol=1e-6)
    print("Checkpoint round trip: PASS")


if __name__ == "__main__":
    main()
