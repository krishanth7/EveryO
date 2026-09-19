# Changelog

All notable changes to EveryO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A tracing ONNX exporter**: `eo.export_onnx_traced`, plus `eo.trace_operations`
  and `eo.TRACEABLE_OPERATIONS`. This closes the last open roadmap item, and
  with it every recurrent and attention model can now reach ONNX.
  - `export_onnx` walks a model layer by layer, so it can only export layers it
    has been taught, and an LSTM is not one: there is no ONNX node meaning
    "EveryO LSTM". Its meaning is ~140 primitives in an order that exists only
    once the layer has run. The tracer runs the model and exports the graph the
    run leaves behind, so it never needs to know what an LSTM *is* — it sees
    adds, matmuls, sigmoids and slices.
  - Verified by re-running each export through ONNX Runtime and comparing
    against EveryO: `RNN` 45 ops / 3.0e-07, `LSTM` 141 / 1.2e-07, `GRU` 165 /
    1.2e-07, `MultiHeadAttention` 21 / 4.8e-07, `TransformerEncoderBlock` 46 /
    1.2e-06, `TransformerEncoder` (2 layers) 101 / 8.3e-07.
  - `dynamic_batch=True` rewrites the batch dimension to be symbolic **and then
    checks that rewrite against a second batch size before writing the file**,
    falling back to the traced batch when it does not hold. Recurrent models
    take that fallback: they fold batch and time into one reshape dimension, so
    the leading dimension is not the batch. A graph honest about accepting one
    batch size beats one that claims to accept any and then miscomputes.
  - Tracing flattens control flow, and that is documented rather than hidden:
    an LSTM traced at 2/4/8 timesteps yields 39/73/141 operations.
  - `everyo.core.autograd.Node` now carries an `attributes` dict recording each
    operation's non-tensor arguments. Backward never reads it; without it the
    axis a softmax reduced over is lost the moment the call returns, and a
    consumer that re-expresses the graph rather than differentiating it cannot
    work. New module, `everyo.serialization.onnx_trace`.
- **Transfer-counting tests for GPU-resident tensors**
  (`tests/test_cuda_resident_transfers.py`). The residency feature's whole point
  — that a chain uploads once instead of once per call — was untested on any
  machine without a GPU, which is every machine in CI. That claim is about *how
  many times the boundary is crossed*, which needs no GPU to check, so a
  stand-in for the native extension now counts every crossing: a 10-operation
  chain crosses 3 times, not 30. This does **not** test the CUDA kernels;
  `tests/test_cuda_resident.py` does, and it still skips without hardware.
- **A cross-process multi-node test**. The TCP all-reduce was exercised between
  threads in one process, which cannot distinguish a working wire protocol from
  ranks sharing memory. It now also runs across two spawned interpreters that
  share nothing but the socket. Still one host — no second machine is available
  in CI, and the README says so.
- `quantize_dynamic`, `quantize`, `profile`, `record_function` and `Profiler` are
  now importable from the top-level `everyo` namespace, like every other
  feature, instead of only from their submodules.

### Fixed

- **Both ONNX exporters now pin the file's IR version** (9, the floor that
  `onnxruntime>=1.17` supports). Previously the stamp was whatever the installed
  `onnx` package defaulted to, which rises with each release; a build machine
  with a recent `onnx` wrote files that a supported `onnxruntime` refuses to
  load with "Unsupported model IR version". A newer runtime accepts exactly what
  an older one rejects, so loading the file on the build machine cannot detect
  this — the test reads the stamp out of the proto instead.
- **CI ran `--doctest-modules` over the new tracing exporter without `onnx`
  installed**, which fails. The base matrix now ignores it exactly as it already
  ignored `onnx_export.py`, and both exporters' doctests run in the ONNX job.

### Changed

- `export_onnx`'s refusal for a recurrent or attention layer now points at
  `export_onnx_traced` instead of stating that EveryO has no tracing exporter.

- **Mixed-precision training**: `eo.autocast` and `eo.GradScaler`. Only
  `matmul` and `conv2d` are cast to float16; reductions, exponentials,
  normalization and losses stay in float32. Autocast works by inserting a
  differentiable cast node, so the backward pass genuinely runs in float16
  while parameters keep float32 gradients — the fp32 master-weight
  arrangement. That matters: with the backward pass in float16, 192 of 193
  gradient elements in a small-loss regime flush to zero without a scaler and
  1 of 193 with one, and the same 400-step run ends at MSE 6.2983 unscaled
  against 0.0474 scaled (float32 reference: 0.0474). `scaler.backward(loss)`
  suppresses the NumPy overflow warnings that a working scaler provokes.
  `float16` is now a supported dtype. **On a CPU this saves memory, not time**:
  NumPy upcasts float16 to compute, so the reduced-precision path is usually
  slower there; the speedup comes from GPU tensor cores.
- **ONNX export**: `eo.export_onnx`, plus `eo.run_onnx` and
  `eo.onnx_available`. Covers `Sequential`, `Linear`, `Conv2D`, `MaxPool2D`,
  `AvgPool2D`, `Flatten`, `Dropout`, `BatchNorm1D`, `BatchNorm2D`, `LayerNorm`
  and the five activations.
  - EveryO is `NHWC` and ONNX `Conv` is `NCHW`, so every convolution and
    pooling node is wrapped in a real pair of `Transpose` nodes and the kernel
    is permuted to `(out, in, kh, kw)`. `"same"` padding is written as explicit
    `pads`, not `auto_pad`, so TensorFlow's asymmetric rule survives.
  - Batch normalization is folded into one scale-and-shift, which is
    numerically identical and layout-free. `track_running_stats=False` is
    refused rather than guessed at.
  - Every supported layer is tested by running the exported graph through ONNX
    Runtime and comparing against the EveryO model, not by checking that a file
    appeared. A trained digit CNN agrees to `1.1e-05`, with 100% of predictions
    matching.
  - Recurrent layers, `Embedding`, attention and the transformer blocks are
    **not** exported; they raise an error naming the layer instead of writing a
    graph that quietly computes something else.
  - New optional extra: `pip install everyo[onnx]`.
- **Single-machine data-parallel training**: `everyo.distributed`, with
  `spawn`, `ProcessGroup`, `average_gradients`, `all_reduce_mean`,
  `shard_indices` and `available_workers`. Workers are processes with a
  shared-memory gradient all-reduce; reduction is in float64 so summing float32
  gradients across ranks loses nothing to the reduction itself.
  - Verified against single-process training: after 60 steps on 4096 samples,
    2 and 4 workers land within `2.1e-07` of the one-process result, and every
    rank ends bit-identical to every other. Wall time on a 4-core container
    went 0.26s → 0.16s → 0.14s for 1, 2 and 4 workers.
  - **One machine only.** No multi-node, no TCP rendezvous, no NCCL.
  - `spawn()` warns when `OMP_NUM_THREADS` and friends are unset: measured on
    the same box, four workers took 1.8s with `OMP_NUM_THREADS=1` and 11.0s
    without — slower than one worker, because sixteen BLAS threads contended
    over four cores.
- New examples: `examples/mixed_precision.py`, `examples/onnx_export.py` and
  `examples/distributed_training.py`, plus `docs/scaling.md` covering all three.
- CI now runs every docstring example (`pytest --doctest-modules everyo`), so
  the documentation cannot drift away from the code it describes.
- **Normalization layers**: `BatchNorm1D`, `BatchNorm2D` and `LayerNorm`.
  Batch normalization keeps running statistics for evaluation mode; layer
  normalization is batch-independent and behaves identically in both modes.
- **Recurrent layers**: `RNN`, `LSTM` and `GRU`, plus an `Embedding` lookup.
  Backpropagation through time is the existing autograd engine walking the
  graph the time loop built, so there is no separate BPTT implementation.
  `LSTM` initialises its forget-gate bias to 1 by default — with a zero bias
  the gate sits at `sigmoid(0) = 0.5`, halving the cell state at every step and
  destroying the long-range memory the cell path exists to provide. Measured
  over 40 timesteps, the fix moved the gradient reaching `t=0` from `9.3e-11`
  to `1.7e-05`.
- **Attention and transformers**: `scaled_dot_product_attention`,
  `MultiHeadAttention`, `PositionalEncoding`, `TransformerEncoderBlock` and
  `TransformerEncoder`, with `causal_mask` and `padding_mask` helpers. Pre-norm
  by default, as modern large models use.
- **Module buffers**: `register_buffer`, `named_buffers` and `buffers`. Buffers
  are state that is saved but never trained — batch normalization's running
  statistics, a positional encoding table — and they travel in `state_dict` and
  through `.evo` archives.
- New sequence datasets (`make_recall_task`, `make_copy_task`,
  `make_parity_task`) and `examples/sequence_models.py`, comparing recurrent
  models with a transformer on long-range recall.
- **Convolution and pooling layers**, the first roadmap item: `Conv2D`,
  `MaxPool2D` and `AvgPool2D`, plus the functional `eo.conv2d`,
  `eo.max_pool2d` and `eo.avg_pool2d`.
  - Tensors use the `NHWC` layout `(batch, height, width, channels)` and
    kernels are `(kh, kw, in_channels, out_channels)` — TensorFlow's layout, so
    results are compared against `tf.nn.conv2d` and `tf.nn.max_pool2d` directly,
    with no transposing. They agree exactly.
  - `padding` accepts `"valid"`, `"same"` or an integer. `"same"` reproduces
    TensorFlow's asymmetric rule; max pooling pads with `-inf` so padding can
    never win a window, and average pooling divides by the count of real cells
    so edge windows are not darkened.
  - Forward passes use im2col, so a convolution is one matrix multiplication;
    the backward pass is two more plus a scatter-add into the padded image.
    Gradients with respect to input, kernel and bias are all verified against
    central finite differences, including strided and `"same"` cases.
  - `compute_fans` now understands 4-D kernels, so He and Xavier initialisation
    scale by `kh * kw * in_channels` rather than a wrong fan-in.
  - The TensorFlow backend maps all three onto their Keras equivalents, and the
    parameter counts match.
  - New example: `examples/convolutional_network.py`, which trains a CNN,
    compares it against a dense baseline of similar size, and visualises the
    learned filters and feature maps.

### Fixed

- Four docstring examples were wrong and had never been executed: `avg_pool2d`
  and `AvgPool2D` called `.item()` on a four-element result, `causal_mask`
  compared against `True` when NumPy returns `np.True_`, and `History.append`
  was shown as returning nothing when it returns the record. All four now run.
- **Moving a tensor no longer severs the autograd graph.** `.to()`, `.cpu()`,
  `.cuda()` and `.astype()` produced a tensor that reported
  `requires_grad=True` but had no edge back to its source, so
  `(x * 2).to("cpu").sum().backward()` silently left `x.grad` unset. Moves are
  now differentiable; a cast to an integer or boolean dtype detaches openly
  (and keeps the requested dtype, which the detach path also used to lose).
- **`BCEWithLogitsLoss` now has the correct derivative at zero.** Composing
  `relu` and `abs` gave both a zero subgradient at `x = 0`, yielding `-target`
  instead of `sigmoid(0) - target = 0.5 - target`, which doubled the step for
  positive labels and cancelled it for negative ones. Zero logits are common
  with zero-initialised output layers, so the loss now has a dedicated backward
  pass.
- **`matmul` differentiates every rank combination it accepts.** A vector times
  a batched matrix succeeded in the forward pass and then raised a core
  dimension error in `backward()`. Gradients are now computed by promoting 1-D
  operands, which covers all vector, matrix and batched combinations.
- **CUDA kernels no longer coerce dtypes.** Every dispatched kernel cast its
  inputs to float32, silently losing precision for float64 tensors and
  corrupting large integers. Non-float32 inputs are now served by the NumPy
  backend, and a `strict=True` call with an unsupported dtype explains why.
- **A summed loss is no longer multiplied by the batch size.** With
  `reduction="sum"` the batch total was weighted by the batch size a second
  time, inflating the recorded loss and making it depend on batching — which
  also affected validation, early stopping and checkpoint selection.
- **Early stopping now says when it restores the best weights.** The restore
  was logged at INFO, which is invisible by default, so metrics measured after
  `fit()` looked inconsistent with the last epoch printed — they describe the
  restored model. `EarlyStopping` prints one explanatory line when it stops;
  pass `verbose=False` to silence it.
- **Charts are drawn even when the active matplotlib backend cannot draw.**
  Every chart is now created through a helper that retries on Agg per call. A
  once-per-process check was not enough: an interactive backend that imports
  cleanly but fails to open a window could be restored mid-session, and later
  charts would still raise.

## [0.1.0] - 2026-09-17

The first release: a working foundation rather than a complete framework.

### Added

**Core**

- `Tensor` with shape, ndim, size, dtype, device and gradient state, plus
  NumPy interoperability and readable representations.
- Dtype handling for `float32`, `float64`, `int32`, `int64` and `bool`, with
  `float32` as the default floating point type.
- `Device` abstraction covering `cpu`, `cuda` and `auto`, with automatic,
  logged fallback to CPU when CUDA is unavailable and an opt-in strict mode.
- Operations: add, subtract, multiply, divide, negate, power, exp, log, sqrt,
  abs, clip, matmul, dot, sum, mean, max, min, variance, reshape, transpose,
  flatten, concatenate, stack, indexing, and creation helpers including seeded
  uniform and normal sampling.
- Reverse-mode automatic differentiation with an iterative graph traversal,
  gradient accumulation, correct unbroadcasting, and `no_grad` /
  `enable_grad` / `set_grad_enabled` modes.

**Neural networks**

- `Module` base class with parameter and sub-module registration, train/eval
  modes, `state_dict`/`load_state_dict`, `get_config` and `summary()`.
- Layers: `Linear`, `Flatten`, `Dropout` (inverted dropout).
- Activations as both functions and modules: ReLU, sigmoid, tanh, softmax,
  log-softmax.
- Losses: MSE, MAE, binary cross entropy (with a numerically stable
  from-logits variant) and multi-class cross entropy.
- Initialisers: Xavier/Glorot and He/Kaiming, uniform and normal.

**Optimizers**

- `SGD` with momentum, Nesterov momentum and weight decay.
- `Adam` with bias correction, weight decay and optional AMSGrad.

**Data**

- `Dataset`, `ArrayDataset`, `TransformDataset`, `Subset` and a numeric CSV
  loader.
- `DataLoader` with batching, seeded shuffling, `drop_last` and a fast
  whole-array path.
- `StandardScaler`, `MinMaxScaler`, normalisation, one-hot encoding.
- `train_test_split`, `random_split`, `stratified_split`.

**Training**

- `Trainer` with `fit`, `evaluate`, `predict` and `predict_classes`, per-epoch
  metrics, validation, elapsed time and optional gradient clipping.
- Callbacks: early stopping with best-weight restoration, model checkpointing,
  progress logging, CSV logging and learning-rate scheduling.
- `History` with metric series, best/last lookup and JSON/CSV export.
- Metrics: accuracy, binary accuracy, MAE, MSE, R², confusion matrix.

**Backends**

- NumPy CPU backend as the reference implementation.
- Optional TensorFlow backend for numerical cross-checks, Keras reference
  models and benchmarks, detected safely and never required.
- Optional CUDA extension: vector add, element-wise multiply, ReLU, tiled
  matrix multiply and a tree reduction, with pybind11 bindings, a CMake build
  and transparent NumPy fallback.

**Everything else**

- Safe, pickle-free `.evo` model format (ZIP + JSON manifest + `allow_pickle=False`
  NumPy archive) with `save`, `load` and `inspect_archive`.
- Matplotlib charts for loss, accuracy, confusion matrices, predictions,
  decision boundaries and benchmarks, all headless-safe.
- `everyo` CLI with `info`, `doctor`, `benchmark`, `test` and `demo`.
- Locally generated datasets: an 8x8 glyph digit dataset plus regression,
  blobs, moons, spirals and XOR generators. No downloads, no API keys.
- Benchmark scripts measuring matmul, ReLU and full training across the
  available backends, with JSON/CSV export.
- 407 tests covering numerics, gradients, layers, losses, optimizers, data,
  training, serialisation, devices, visualization, the CLI and end-to-end
  pipelines.
- Optional JSON/YAML configuration, an exception hierarchy with actionable
  messages, and standard-library logging with no telemetry.

### Notes

- The copyright line in `LICENSE` reads "EveryO contributors". A repository
  owner who wants their own name or organisation there should edit that line.

[Unreleased]: https://github.com/krishanth7/EveryO/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/krishanth7/EveryO/releases/tag/v0.1.0
