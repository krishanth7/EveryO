# EveryO

**EveryO is an open-source neural computing and experimentation framework for
building, training, evaluating and understanding neural networks across CPU and
GPU environments.**

[![tests](https://github.com/krishanth7/EveryO/actions/workflows/tests.yml/badge.svg)](https://github.com/krishanth7/EveryO/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)

---

## Overview

EveryO implements the machinery of a neural network framework — tensors,
automatic differentiation, layers, optimizers, a training loop — from first
principles on top of NumPy, and then makes that machinery fast where it matters
through optional TensorFlow and CUDA backends.

The CPU path is complete and always available. TensorFlow and CUDA are
accelerators and cross-checks, never requirements: a laptop with no GPU, no
CUDA Toolkit and no TensorFlow runs everything in this repository except the
GPU-specific tests, which skip themselves.

```
BUILD  ->  TRAIN  ->  MEASURE  ->  UNDERSTAND  ->  ACCELERATE
```

## Why EveryO?

Most frameworks are either educational toys that cannot train anything real, or
production systems whose internals are effectively closed to a reader. EveryO
aims at the space between:

* **Readable.** The autograd engine is roughly 250 lines. The whole `core` layer
  can be read in an afternoon, and the docs explain *why* each piece is shaped
  the way it is.
* **Actually correct.** Every gradient is verified against finite differences,
  and operations are compared against NumPy and (when installed) TensorFlow.
  407 tests run in about 12 seconds on CPU (8 of them skip without a GPU).
* **Honest about limits.** Nothing here is a placeholder. Features are marked
  Available, Experimental or Planned, and benchmark numbers come from runs on
  your own machine, never from this README.
* **Self-contained.** No API keys, no accounts, no external AI services, no
  network access in the core. Datasets are generated locally from a seed.

## Features

### Available

| Area | What works |
| --- | --- |
| Tensors | Creation, dtypes, devices, broadcasting, indexing, reshaping, reductions, NumPy interoperability |
| Autograd | Reverse-mode gradients for 25+ operations, iterative graph traversal, `no_grad` mode, finite-difference verified |
| Layers | `Linear`, `Flatten`, `Dropout`, `Sequential`, and a `Module` base class you can subclass |
| Activations | ReLU, sigmoid, tanh, softmax, log-softmax — as functions and as modules |
| Losses | MSE, MAE, binary cross entropy (with a stable from-logits form), cross entropy |
| Optimizers | SGD, SGD with momentum and Nesterov, Adam (with weight decay and AMSGrad) |
| Data | `Dataset`, `ArrayDataset`, `DataLoader`, CSV loading, scalers, train/test and stratified splits |
| Training | `Trainer` with validation, metrics, early stopping, checkpointing, LR scheduling, gradient clipping, history export |
| Serialization | Safe pickle-free `.evo` archives that cannot execute code on load |
| Visualization | Loss, accuracy, confusion matrix, prediction, decision boundary and benchmark charts; headless-safe |
| CLI | `everyo info`, `doctor`, `benchmark`, `test`, `demo` |
| NumPy backend | The reference implementation of every kernel |
| TensorFlow backend | Optional: numerical cross-checks, Keras reference models, benchmarks |
| CUDA backend | Optional: vector add, element-wise multiply, ReLU, tiled matmul, tree reduction — with automatic CPU fallback |

### Experimental

* CUDA kernel dispatch for tensors placed on a `cuda` device. The kernels are
  correct and checked against NumPy, but every call currently copies data to
  the device and back, so the GPU is not necessarily faster. Measure before
  relying on it.

### Planned

Not implemented in v0.1.0 — see the [roadmap](#roadmap).

## Architecture

```
                everyo.cli            everyo.visualization
                     |                          |
        +------------+--------------------------+
        |
   everyo.training  (Trainer, callbacks, history, metrics)
        |
   everyo.optim     (SGD, Adam)          everyo.serialization  (.evo archives)
        |                                        |
   everyo.nn        (Module, layers, activations, losses)
        |
   everyo.core      (Tensor, operations, autograd, device, dtype)
        |
   everyo.backends  (numpy_backend, tensorflow_backend)  everyo.cuda
        |                                                    |
      NumPy                                     optional native extension
```

Each layer depends only on the ones below it. See
[docs/architecture.md](docs/architecture.md) for the reasoning behind the
design decisions.

## Installation

```bash
git clone https://github.com/krishanth7/EveryO.git
cd EveryO

python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -e .
pytest
```

Optional extras:

```bash
pip install -e ".[dev]"          # pytest, ruff, PyYAML
pip install -e ".[tensorflow]"   # the TensorFlow backend (large download)
pip install -e ".[cuda]"         # pybind11, for building the CUDA extension
```

Verify the installation:

```bash
everyo info
everyo doctor
```

`doctor` runs a real gradient check rather than only importing modules, and
explains exactly why any optional component is unavailable.

## Quick start

```python
import everyo as eo

x = eo.tensor([[1.0, 2.0]])
y = eo.tensor([[3.0], [4.0]])

result = eo.matmul(x, y)
print(result)
# Tensor([[11.]], shape=(1, 1), dtype=float32)
```

Gradients:

```python
x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()

print(x.grad)  # [4.]
```

A network, trained:

```python
import everyo as eo
from everyo.datasets import make_blobs

features, labels = make_blobs(n_samples=600, n_features=4, centers=3, seed=0)
x_train, x_test, y_train, y_test = eo.stratified_split(features, labels, test_size=0.2, seed=0)

model = eo.Sequential(
    eo.Linear(4, 16),
    eo.ReLU(),
    eo.Linear(16, 3),
)

trainer = eo.Trainer(
    model=model,
    optimizer=eo.Adam(model.parameters(), lr=0.01),
    loss_fn=eo.CrossEntropyLoss(),
    metrics=["accuracy"],
)

history = trainer.fit(
    eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=32, shuffle=True, seed=0),
    epochs=20,
    validation_loader=eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64),
)

print(trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64)))
```

## Tensor operations

```python
import everyo as eo

x = eo.tensor([[1.0, 2.0], [3.0, 4.0]])

x.shape, x.ndim, x.dtype, x.device, x.size

x + 10  # broadcasting against a scalar
x * x  # element-wise
eo.matmul(x, x)  # matrix product, also x @ x

eo.sum(x)  # 10.0
eo.mean(x, axis=0)  # column means
eo.max(x, axis=1)  # row maxima

x.reshape(4, 1)
x.T
eo.flatten(x)

eo.normal((2, 3), seed=0)  # reproducible random tensors
eo.zeros(2, 3), eo.ones(2, 3), eo.eye(3)
```

Shape errors explain themselves:

```python
>>> eo.matmul(eo.zeros(32, 64), eo.zeros(128, 10))
EveryOShapeError: Cannot multiply matrices with shapes (32, 64) and (128, 10).
Expected the inner dimensions to match: 64 != 128.
```

## Building neural networks

```python
model = eo.Sequential(
    eo.Linear(784, 256),
    eo.ReLU(),
    eo.Dropout(0.2),
    eo.Linear(256, 128),
    eo.ReLU(),
    eo.Linear(128, 10),
)

print(model.summary())
print(model.num_parameters())
```

Or write your own module:

```python
from everyo.nn import Module, register_module


@register_module
class Residual(Module):
    """A linear layer with a skip connection."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
        self.linear = eo.Linear(size, size)

    def forward(self, x):
        return eo.add(x, eo.relu(self.linear(x)))

    def get_config(self):
        return {"size": self.size}
```

Registering the class is what lets a model containing it be saved and loaded.

## Training

```python
trainer = eo.Trainer(
    model=model,
    optimizer=eo.Adam(model.parameters(), lr=0.001),
    loss_fn=eo.CrossEntropyLoss(),
    metrics=["accuracy"],
    gradient_clip=1.0,
)

history = trainer.fit(
    train_loader,
    epochs=50,
    validation_loader=validation_loader,
    callbacks=[
        eo.EarlyStopping(monitor="val_loss", patience=5),
        eo.ModelCheckpoint("artifacts/best.evo", monitor="val_accuracy", mode="max"),
        eo.CSVLogger("artifacts/history.csv"),
    ],
)

results = trainer.evaluate(test_loader)
predictions = trainer.predict_classes(x_test)
```

`history` records loss, validation loss, every configured metric and the
elapsed time per epoch, and exports with `history.to_json(...)` or
`history.to_csv(...)`.

Saving and loading:

```python
eo.save(model, "model.evo", metadata={"accuracy": 0.99})
model = eo.load("model.evo")
```

An `.evo` file is a ZIP archive holding a JSON manifest and a NumPy parameter
archive read with `allow_pickle=False`. Models are rebuilt from registered
class names, so **loading a model file never executes code from it**.

## CUDA acceleration

CUDA is optional and additive:

```python
import everyo as eo

eo.cuda.is_available()  # False on a CPU-only machine — not an error
eo.cuda.unavailable_reason()  # a sentence explaining why
eo.cuda.runtime_info()  # device names, compute capability, memory

x = eo.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")  # falls back to CPU
```

Build the extension on a machine with an NVIDIA GPU and the CUDA Toolkit:

```bash
pip install -e ".[cuda]"
./scripts/build_cuda.sh
```

Kernels implemented in `cuda/src/`: vector addition, element-wise
multiplication, ReLU, tiled matrix multiplication and a tree-reduction sum. All
use bounds checking, CUDA error checking and RAII device memory, and their
output is compared against NumPy in the test suite.

Requesting CUDA without it never crashes; it falls back to the CPU and logs a
warning. Pass `strict=True` when you would rather have an error than a silent
fallback. Details in [docs/cuda.md](docs/cuda.md).

CUDA works only on NVIDIA hardware with NVIDIA's toolchain. EveryO does not
claim otherwise, and the GPU is not automatically faster — per-call host/device
transfers can dominate at small sizes, which is what the benchmarks are for.

## TensorFlow backend

TensorFlow is optional and plays three specific roles: reference numerics for
cross-checking EveryO's own kernels, an equivalent Keras model for comparing
convergence, and a benchmark baseline.

```python
from everyo.backends import tensorflow_backend as tfb

tfb.is_available()
tfb.gpu_available()

keras_model = tfb.build_keras_model(model, input_shape=(64,))
reference = tfb.train_reference_model(model, features, labels, epochs=20)
```

EveryO is **not** a TensorFlow wrapper: nothing in the core imports TensorFlow,
and the framework is fully usable without it.

## Visualization

```python
eo.plot_loss(history, save_path="artifacts/loss.png")
eo.plot_accuracy(history, save_path="artifacts/accuracy.png")
eo.plot_history(history, save_path="artifacts/history.png")
eo.plot_confusion_matrix(predictions, y_test, save_path="artifacts/confusion.png")
eo.plot_benchmark(results, save_path="artifacts/benchmark.png")
```

Charts never require a display: on a headless machine EveryO selects the
non-interactive Agg backend automatically, and every function accepts
`save_path=` to write a PNG.

## Benchmarks

Benchmarks are measured, never quoted:

```bash
python benchmarks/benchmark_matmul.py --sizes 128 256 512 1024
python benchmarks/benchmark_relu.py
python benchmarks/benchmark_training.py --epochs 10

everyo benchmark --sizes 256 512 --output results.json
```

Each script warms up before timing, repeats every measurement, records the
mean, best and worst times, and reports the machine it ran on. Results can be
exported to JSON or CSV and plotted. This README deliberately contains no
performance figures: run the scripts and read your own.

## Examples

```bash
python examples/tensor_basics.py             # tensors, broadcasting, autograd
python examples/linear_regression.py         # a single Linear layer, SGD + momentum
python examples/binary_classification.py     # two moons: MLP vs a linear model
python examples/multiclass_classification.py # spirals, with early stopping
python examples/neural_network.py            # the full pipeline on digits
python examples/save_and_load.py             # .evo archives
python examples/tensorflow_backend.py        # cross-checks (needs TensorFlow)
python examples/cuda_example.py              # CUDA detection and fallback
```

The end-to-end demo is also available from the CLI:

```bash
everyo demo --epochs 15
```

It builds an 8x8 handwritten-digit-style dataset locally from bitmap glyphs
(no download), splits it, trains a two-layer network and reports test accuracy.
See [`examples/`](examples/) for details.

## Project structure

```
EveryO/
├── everyo/
│   ├── core/            tensor, operations, autograd, device, dtype
│   ├── nn/              module, layers, activations, losses, initialization
│   ├── optim/           optimizer, sgd, adam
│   ├── data/            dataset, dataloader, preprocessing, split
│   ├── training/        trainer, callbacks, history, metrics
│   ├── backends/        numpy_backend, tensorflow_backend
│   ├── serialization/   save, load, format
│   ├── visualization/   training, metrics, benchmark charts
│   ├── datasets/        locally generated datasets
│   ├── cuda/            availability detection and kernel interface
│   └── cli/             the everyo command
├── cuda/                .cu kernels, header, pybind11 bindings, CMakeLists
├── tests/               407 tests
├── examples/            runnable scripts
├── benchmarks/          measurement scripts
├── scripts/             setup, test, lint, benchmark, build_cuda, clean
├── configs/             example YAML configuration
└── docs/                architecture, autograd, CUDA, API reference
```

## Development

```bash
./scripts/setup.sh          # virtual environment + dev install
./scripts/test.sh           # the test suite
./scripts/test.sh --fast    # skip slow tests
./scripts/lint.sh           # ruff
./scripts/lint.sh --fix     # ruff with fixes and formatting
./scripts/benchmark.sh      # every benchmark
./scripts/build_cuda.sh     # the optional CUDA extension
./scripts/clean.sh          # caches and build output
```

Windows users can run the equivalent commands directly
(`python -m pytest`, `ruff check .`, `python benchmarks/benchmark_matmul.py`);
the shell scripts target Linux and macOS.

Configuration files are supported but never required:

```python
from everyo import load_config

config = load_config("configs/default.yaml")
```

## Testing

```bash
pytest                                  # everything
pytest -m "not slow"                    # quick pass
pytest --cov=everyo --cov-report=term   # with coverage
pytest tests/test_autograd.py -v        # one module
```

What the suite covers:

* numerical results compared against NumPy, and against TensorFlow when it is
  installed,
* every gradient compared against central finite differences,
* layers, losses, optimizer update rules and convergence on problems with
  known answers,
* data loading, splitting and preprocessing,
* serialization round trips, including refusal of malformed archives,
* CPU fallback when CUDA is absent, and kernel correctness when it is present,
* the CLI, configuration and full end-to-end pipelines.

CUDA tests skip automatically without a GPU; TensorFlow tests skip without
TensorFlow. The core suite needs no network access.

## Roadmap

Planned for future releases — **none of this is implemented today**:

* convolutional and pooling layers,
* batch and layer normalisation,
* recurrent layers, attention and transformer blocks,
* GPU-resident tensors, removing per-call host/device transfers,
* fused and more advanced CUDA kernels,
* mixed-precision training,
* automatic device placement,
* ONNX interoperability,
* model quantization,
* profiling tools and graph optimization,
* distributed training.

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the
development setup, branch workflow, coding conventions, how to add operations
and layers, and the extra care CUDA changes need.

The short version: add tests (gradients need a finite-difference check), keep
`ruff check .` clean, and make sure the core still works with neither CUDA nor
TensorFlow installed.

## Security

EveryO needs no API keys, tokens or accounts, makes no network requests from
the core, and collects no telemetry. Model loading is pickle-free by design.
To report a vulnerability, see [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
