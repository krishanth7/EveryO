# Changelog

All notable changes to EveryO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

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
- 405 tests covering numerics, gradients, layers, losses, optimizers, data,
  training, serialisation, devices, visualization, the CLI and end-to-end
  pipelines.
- Optional JSON/YAML configuration, an exception hierarchy with actionable
  messages, and standard-library logging with no telemetry.

### Notes

- The copyright line in `LICENSE` reads "EveryO contributors". A repository
  owner who wants their own name or organisation there should edit that line.

[Unreleased]: https://github.com/krishanth7/EveryO/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/krishanth7/EveryO/releases/tag/v0.1.0
