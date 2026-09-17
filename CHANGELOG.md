# Changelog

All notable changes to EveryO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
