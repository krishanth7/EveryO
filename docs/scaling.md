# Mixed precision, ONNX export and data-parallel training

Three features that a framework tends to grow once the core is correct: running
in fewer bits, leaving for another runtime, and using more than one core. This
page covers all three, including what each one does *not* do.

---

## Mixed precision

### What autocast changes

Inside an `eo.autocast()` block, exactly two operations change precision:
`matmul` and `conv2d`. Everything else — reductions, exponentials,
normalization statistics, losses — stays in float32, because summing thousands
of values or exponentiating one is where 16 bits runs out of range.

```python
import everyo as eo

model = eo.Sequential(eo.Linear(8, 4, seed=0), eo.ReLU(), eo.Linear(4, 1, seed=1))

with eo.autocast():
    output = model(eo.ones(3, 8))

output.dtype  # 'float16'
model[0].weight.dtype  # 'float32'
```

The mechanism is a cast node, not a flag. When autocast is on, each floating
operand of an allowlisted operation is routed through a differentiable cast; the
forward pass then runs in float16, and the cast's backward pass converts the
incoming gradient back to the operand's own dtype. A float32 parameter therefore
keeps a float32 gradient — the "fp32 master weights" arrangement — while
activations, and the gradients flowing through them, genuinely stay in float16.

That last part matters more than it sounds. If the backward pass quietly ran in
float32, nothing would ever underflow, and loss scaling would be theatre.

`autocast` also works as a decorator and nests:

```python
@eo.autocast()
def forward(model, batch):
    return model(batch)


with eo.autocast():
    with eo.autocast(False):  # a float32 island inside a float16 block
        logits = sensitive_head(features)
```

### Why GradScaler exists

float16's smallest normal value is about `6.1e-5` and its smallest subnormal
about `6e-8`. Real gradients are routinely below both, so they flush to zero and
the weights never move. `GradScaler` multiplies the loss by a large constant
before `backward()`, which shifts the whole gradient distribution into
representable range, and divides it back out before the optimizer steps.

Measured on a small MLP whose loss has been scaled down into that regime:

```
Gradient elements flushed to zero (193 total)
  no GradScaler ........ 192 / 193
  with GradScaler ......   1 / 193

Final MSE after 400 steps
  autocast, no scaler .. 6.2983
  autocast + scaler .... 0.0474      float32 reference: 0.0474
```

The scale is chosen dynamically: it grows after `growth_interval` successful
steps and halves the moment a gradient overflows to infinity — and that step is
skipped rather than applied, because an infinite gradient would destroy the
weights.

```python
scaler = eo.GradScaler()

for batch, targets in loader:
    optimizer.zero_grad()
    with eo.autocast():
        predictions = model(batch)
    loss = criterion(predictions, targets)

    scaler.backward(loss)  # scale, then backward
    applied = scaler.step(optimizer)  # False when the step was skipped
    scaler.update()  # grow or shrink the scale
```

`scaler.backward(loss)` is `scaler.scale(loss).backward()` with NumPy's overflow
warnings suppressed — an overflow there is the signal `step()` acts on, not a
condition worth printing. Use `scaler.scale(loss).backward()` if you want to see
them.

`scaler.state_dict()` and `load_state_dict()` round-trip the scale, so resuming
from a checkpoint does not restart the search.

### The honest caveat

> On a CPU, mixed precision saves **memory, not time**. NumPy has no native
> float16 arithmetic: it upcasts to float32 to compute and casts back, so the
> float16 path is usually *slower* here. The speedup mixed precision is known for
> comes from GPU tensor cores. EveryO implements the numerics so the behaviour is
> right; it does not pretend to make your CPU faster.

---

## ONNX export

```python
model.eval()
eo.export_onnx(model, "model.onnx", input_shape=(1, 8, 8, 1), output_name="logits")
```

The batch dimension is exported as a symbolic dimension, so the graph accepts
any batch size regardless of what `input_shape` said.

### Supported layers

`Sequential`, `Linear`, `Conv2D`, `MaxPool2D`, `AvgPool2D`, `Flatten`,
`Dropout`, `BatchNorm1D`, `BatchNorm2D`, `LayerNorm`, `ReLU`, `Sigmoid`, `Tanh`,
`Softmax`, `LogSoftmax`.

**Not supported**: `RNN`, `LSTM`, `GRU`, `Embedding`, `MultiHeadAttention`,
`PositionalEncoding` and the transformer blocks. These unroll into long chains
of primitives that need a tracing exporter rather than a layer-by-layer one, and
a half-correct translation would be worse than none. Passing one raises
`EveryOSerializationError` naming the layer.

### Two decisions worth knowing

**Layout.** EveryO uses `NHWC`; ONNX's `Conv`, `MaxPool` and `AveragePool` are
defined on `NCHW`. Every convolution and pooling node is therefore wrapped in a
pair of real `Transpose` nodes, and the kernel is permuted from
`(kh, kw, in, out)` to ONNX's `(out, in, kh, kw)`. ONNX Runtime's optimizer
usually cancels adjacent transposes, but the exported graph is faithful first
and fast second.

**Padding.** `"same"` is written out as an explicit `pads` attribute computed by
EveryO's own padding resolver, not as `auto_pad`. TensorFlow's `"same"` rule is
asymmetric on odd inputs, and `auto_pad` would not reproduce it exactly.

Batch normalization is folded into a single scale-and-shift on the channel axis,
which is numerically identical to ONNX's `BatchNormalization` and sidesteps the
layout question entirely. A layer built with `track_running_stats=False` has no
inference-time statistics to export and is refused rather than guessed at.

### Verifying an export

```python
import numpy as np

expected = np.asarray(model(eo.tensor(batch)).data)
actual = eo.run_onnx("model.onnx", batch)
np.abs(expected - actual).max()
```

A trained digit CNN exported and re-run through ONNX Runtime agrees to
`1.1e-05` — float32 rounding from a different order of operations — and every
prediction matches. `everyo`'s own test suite makes this comparison for each
supported layer rather than checking that a file was written.

Both packages are optional:

```bash
pip install everyo[onnx]
```

`eo.onnx_available()` reports whether `onnx` is importable, and `run_onnx`
raises with an install hint when `onnxruntime` is not.

---

## Data-parallel training

Give every worker a copy of the model and a different slice of the batch, then
average the gradients before each optimizer step. Averaging over disjoint shards
is the same arithmetic as one large batch, so the result matches single-process
training.

```python
import everyo as eo
from everyo.distributed import average_gradients, shard_indices, spawn


def worker(group, inputs, targets):
    model = build_model()  # same seeds on every rank
    optimizer = eo.SGD(model.parameters(), lr=0.05)
    rows = shard_indices(len(inputs), group.rank, group.world_size)
    xs, ys = eo.tensor(inputs[rows]), eo.tensor(targets[rows])

    for _ in range(steps):
        optimizer.zero_grad()
        residual = model(xs) - ys
        eo.mean(residual * residual).backward()
        average_gradients(model.parameters(), group)
        optimizer.step()

    return [p.data for p in model.parameters()]


results = spawn(worker, world_size=4, args=(inputs, targets))
```

Measured on a 4-core container, 60 steps of a three-layer MLP over 4096 samples:

```
 workers   rows/worker    seconds    max drift   ranks agree
--------------------------------------------------------------
       1          4096       0.26     0.00e+00          True
       2          2048       0.16     1.79e-07          True
       4          1024       0.14     2.09e-07          True
```

`max drift` is the largest weight difference from single-process training;
`ranks agree` says whether every worker ended bit-identical.

### The pieces

| | |
|---|---|
| `spawn(fn, world_size, args=...)` | Launches the workers, returns their results ordered by rank |
| `ProcessGroup` | Passed to each worker: `rank`, `world_size`, `is_main`, `barrier()`, `all_reduce_mean()`, `broadcast()` |
| `average_gradients(params, group)` | The one line a training loop adds |
| `shard_indices(count, rank, world_size)` | Splits a dataset; `drop_last=True` keeps every rank's batch the same size |
| `available_workers()` | Usable core count, respecting cgroup and affinity limits |

Workers are processes, not threads: NumPy releases the GIL for large array
operations but holds it for the graph bookkeeping around them, so threads would
contend on exactly the part of training that is pure Python.

The collective itself flattens every gradient into one vector, stages it in a
POSIX shared-memory buffer with one slot per rank, and takes two barriers — one
after the write, one after the read, so nobody overwrites a slot a peer is still
reading. Reduction happens in float64 so summing float32 gradients across
workers does not lose a bit to the reduction.

### Two caveats, both load-bearing

> **One machine only.** There is no multi-node support, no TCP rendezvous and no
> NCCL. Workers communicate through shared memory, which means they share
> hardware. Anything else would be untested.

> **Set `OMP_NUM_THREADS=1` before launching Python.** NumPy's BLAS starts a
> thread pool per process. On the same 4-core box, four workers took **1.8s**
> with the variable set and **11.0s** without — slower than not parallelising at
> all, because sixteen threads were fighting over four cores. The variable has to
> be set before NumPy loads its backend, so EveryO warns rather than pretending
> it can fix it for you.

If `spawn` times out, the usual cause is a rank that took a different path
through the loop: every rank must call the same collectives the same number of
times, or the ones that reached the barrier wait forever for the one that did
not.

---

## Running the examples

```bash
python examples/mixed_precision.py
python examples/onnx_export.py
OMP_NUM_THREADS=1 python examples/distributed_training.py
```
