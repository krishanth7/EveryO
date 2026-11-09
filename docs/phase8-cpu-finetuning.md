# Phase 8 — CPU-first fine-tuning foundation

Phase 8 extends the Phase 7 educational transformer with a small, reproducible
instruction fine-tuning pipeline. It is **not** a pretrained instruction model
or a production-scale LLM training stack. All acceptance tests use NumPy/CPU.

| Milestone | API / scope | Verified behavior |
| --- | --- | --- |
| 8.1 LoRA | `LoRALinear` | Frozen base, initially zero delta, low-rank update |
| 8.2 Parameter-efficient fine-tuning | `LoRATinyDecoderLM` | Only adapter A/B train; registered `.evo` model |
| 8.3 Supervised instruction tuning | `prepare_instruction_batch`, `masked_next_token_loss` | Answer-only supervision, prompt/padding excluded |
| 8.4 Gradient checkpointing | `gradient_checkpoint` | Recompute deterministic functions in backward; gradient parity |
| 8.5 Mixed precision | Optional `autocast` and `GradScaler` in fine-tuning | Disabled by default on CPU |
| 8.6 Data preparation | `InstructionBatch` | Character vocabulary, truncation/error checks, padding |
| 8.7 Evaluation | `evaluate_instructions`, `FineTuneReport` | Deterministic supervised cross entropy, loss trace |
| 8.8 CPU example | `examples/cpu_instruction_finetune.py` | Train, improve toy loss, save and load checkpoint |

## Run

```bash
python -m pytest -q tests/test_phase8_finetuning.py
python examples/cpu_instruction_finetune.py --steps 40
```

```python
from everyo import CharTokenizer, LoRATinyDecoderLM
from everyo import fine_tune_instructions, prepare_instruction_batch

pairs = [("a", "b"), ("b", "a")]
corpus = "".join(f"Instruction: {q}\nAnswer: {a}" for q, a in pairs)
tokenizer = CharTokenizer(corpus)
batch = prepare_instruction_batch(pairs, tokenizer, max_context=32)
model = LoRATinyDecoderLM(tokenizer.vocab_size, embed_dim=8, num_heads=2, max_context=32)
report = fine_tune_instructions(model, batch, steps=25)
print(report.initial_loss, report.final_loss)
```

## Boundaries and cautions

- LoRA is currently attached to the **output projection head only**. This
  proves the adapter mechanism without claiming full Q/K/V/MLP adaptation.
- For adapting an existing base, call `adapter.load_base(base)` before training;
  it validates dimensions and copies weights but does not change frozen status.
- Gradient checkpointing works on **pure, deterministic single-Tensor** functions
  with explicitly supplied parameters. Avoid dropout, BatchNorm state updates,
  parameter mutation between forward/backward, and other side effects.
- Optional autocast uses float16 operations with float32 parameters; on ordinary
  CPUs it may be slower and provide no performance benefit. Default is float32.
- Instruction dataset preparation rejects unknown characters and oversized
  examples; this pipeline does not silently truncate labels.
- Tokenizer metadata, training dataset, optimizer progress, and RNG state are
  not embedded in the `.evo` archive. Store them separately when needed.
- Evaluation is supervised held-in-data loss only; independent held-out
  evaluation, safety, scalability and deployment gates remain future work.
