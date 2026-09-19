"""Export recurrent and attention models to ONNX by tracing them.

`export_onnx` walks a model layer by layer, which works only for layers it has
been taught. An LSTM is not one of them: there is no single ONNX node that means
"EveryO LSTM", because its meaning is a hundred-odd primitive operations in an
order that only exists once the layer has run.

So this exporter runs it, then exports the graph the run left behind.
"""

import tempfile
from pathlib import Path

import numpy as np

import everyo as eo

destination = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

models = [
    ("RNN", eo.RNN(16, 32, seed=0), (4, 8, 16)),
    ("LSTM", eo.LSTM(16, 32, seed=0), (4, 8, 16)),
    ("GRU", eo.GRU(16, 32, seed=0), (4, 8, 16)),
    ("MultiHeadAttention", eo.MultiHeadAttention(32, 4, seed=0), (4, 8, 32)),
    ("TransformerEncoder", eo.TransformerEncoder(32, 4, num_layers=2, seed=0), (4, 8, 32)),
]

print(f"{'model':<22} {'traced ops':>10} {'max |diff|':>12}")
print("-" * 46)

for name, model, shape in models:
    model.eval()
    example = rng.standard_normal(shape).astype(np.float32)
    path = destination / f"{name}.onnx"

    eo.export_onnx_traced(model, path, example_input=example)

    # The export is compared against the model it came from, not merely
    # declared correct: an exporter that writes a valid file computing the
    # wrong thing is worse than one that fails.
    exported = eo.run_onnx(path, example)
    expected = model(eo.tensor(example)).numpy()
    difference = float(np.abs(exported - expected).max())

    operations = eo.trace_operations(model, example)
    print(f"{name:<22} {len(operations):>10} {difference:>12.2e}")

print()
print("An LSTM, once traced, is just arithmetic:")
counts: dict[str, int] = {}
for operation in eo.trace_operations(
    models[1][1], rng.standard_normal((1, 4, 16)).astype(np.float32)
):
    counts[operation] = counts.get(operation, 0) + 1
for operation, count in sorted(counts.items(), key=lambda pair: -pair[1]):
    print(f"  {operation:<14} {count}")

print()
print("Tracing flattens the timestep loop, so the sequence length is baked in:")
for steps in (2, 4, 8):
    traced = eo.trace_operations(models[1][1], np.zeros((1, steps, 16), dtype=np.float32))
    print(f"  {steps} timesteps -> {len(traced)} operations")
