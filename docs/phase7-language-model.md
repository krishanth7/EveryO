# Phase 7 — CPU-first language model engineering

Phase 7 provides a small, fully trainable decoder-only transformer built from
EveryO's differentiable primitives. It works on NumPy/CPU without a GPU, an
external model provider, or a pretrained checkpoint. This is an educational
framework capability, **not** a competitive pretrained chat model.

| Milestone | Implementation | Acceptance evidence |
| --- | --- | --- |
| 7.1 Character and subword tokenizers | `CharTokenizer`, `ByteBPETokenizer` | Unicode round trip, deterministic BPE merges |
| 7.2 Decoder-only transformer | `TinyDecoderLM`, `DecoderBlock` | Correct `(batch, time, vocab)` logits |
| 7.3 Causal attention | Lower-triangular mask in each block | Changing future tokens does not change past logits |
| 7.4 LM objective | `next_token_loss` | Cross-entropy on next-token targets, differentiable |
| 7.5 Autoregressive generation | `TinyDecoderLM.generate` | Prompt continuation with max-context validation |
| 7.6 KV cache | `KVCache`, `cached_step` | Projected key/value cache matches full causal inference |
| 7.7 Sampling | `sample_next_token` | Temperature, top-k, nucleus/top-p and fixed RNG seed |
| 7.8 Tiny CPU LM | `examples/tiny_language_model.py` | Train, sample, save `.evo`, load, compare logits |

## Run locally

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_phase7_language_model.py
python examples/tiny_language_model.py --steps 30
```

For saved model output: `python examples/tiny_language_model.py --output tiny.evo`.
When no `--output` is given, the example uses an automatically removed temporary
directory. The tokenizer is constructed from the training corpus and is not
embedded in the model archive: persist its corpus/vocabulary separately for
deployments that need decoding after reload.

## Minimal example

```python
import numpy as np
from everyo import CharTokenizer, TinyDecoderLM, next_token_loss

tok = CharTokenizer("hello hello")
ids = np.array([tok.encode("hello")])
model = TinyDecoderLM(tok.vocab_size, embed_dim=8, num_heads=2)
loss = next_token_loss(model(ids[:, :-1]), ids[:, 1:])
loss.backward()
generated = model.generate(ids[:, :2], max_new_tokens=3, top_k=1, seed=0)
print(tok.decode(generated[0]))
```

## Engineering boundaries

- Positions use learned absolute embeddings. Input length plus generated output
  cannot exceed `max_context`; the model fails explicitly when exceeded.
- The KV cache is an inference-only NumPy path. Full forward/backward remains
  in EveryO autograd and uses quadratic causal self-attention.
- The toy BPE is deterministic and reversible on UTF-8 bytes, but is not a
  drop-in GPT-2/tiktoken-compatible tokenizer.
- `CharTokenizer` reserves ID 0 for padding and 1 for unknown characters.
  Padding loss masking and batched variable-length training are not included.
- Checkpoint restoration preserves model weights/configuration, not tokenizer
  metadata, RNG state, or optimizer state.
- Repeatable runs depend on the same CPU numerical environment and settings.
  Passing the test suite is not a claim of production-scale LLM performance.
