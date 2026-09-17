<div align="center">

<img src="docs/assets/banner.png" alt="EveryO — open-source neural computing framework" width="100%">

<h3>Neural networks, from the math up — not from a wrapper down.</h3>

<p><b>EveryO</b> is a neural computing framework whose autograd engine, layers, optimizers and training loop
are written from scratch on NumPy — then accelerated with optional CUDA kernels and cross-checked against TensorFlow.<br>
Small enough to read in an afternoon. Correct enough to trust.</p>

[![tests](https://github.com/krishanth7/EveryO/actions/workflows/tests.yml/badge.svg)](https://github.com/krishanth7/EveryO/actions/workflows/tests.yml)
[![CUDA build](https://github.com/krishanth7/EveryO/actions/workflows/cuda-build.yml/badge.svg)](https://github.com/krishanth7/EveryO/actions/workflows/cuda-build.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-eb6834)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/krishanth7/EveryO?style=flat&color=eda100)](https://github.com/krishanth7/EveryO/stargazers)

**[Quick start](#-quick-start-60-seconds) · [What it does](#-what-it-actually-does) · [Results](#-results-from-a-real-run) · [Architecture](#-architecture) · [Roadmap](#-roadmap) · [Contribute](#-contributing)**

</div>

---

## Why this exists

Every deep-learning tutorial ends at `model.fit()`. Every production framework starts a million lines below it.
There is almost nothing in between you can actually *read*.

EveryO is that middle. The gradient engine is **251 lines**. The whole core is **~8,000 lines** across 54 focused
modules. You can follow a single number from `loss.backward()` all the way to a CUDA kernel — and every gradient
in it is verified against finite differences **and** against TensorFlow.

```
BUILD  →  TRAIN  →  MEASURE  →  UNDERSTAND  →  ACCELERATE
```

<div align="center">
<img src="docs/assets/terminal.png" alt="everyo doctor and everyo demo running end to end" width="92%">
</div>

---

## ⚡ Quick start (60 seconds)

```bash
git clone https://github.com/krishanth7/EveryO.git
cd EveryO && pip install -e .
everyo demo          # trains a digit classifier end to end
```

```python
import everyo as eo

x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()

print(x.grad)  # [4.]   ← your own autograd engine, not a binding
```

Build and train a network:

```python
model = eo.Sequential(
    eo.Linear(64, 128),
    eo.ReLU(),
    eo.Dropout(0.2),
    eo.Linear(128, 10),
)

trainer = eo.Trainer(
    model, eo.Adam(model.parameters(), lr=0.005), eo.CrossEntropyLoss(), metrics=["accuracy"]
)

history = trainer.fit(train_loader, epochs=25, validation_loader=test_loader)
```

---

## 📊 Results from a real run

Every image below was produced by the code in this repository, on a CPU, in under a second of training.
Reproduce them with `python examples/neural_network.py`.

<div align="center">
<img src="docs/assets/training-curves.png" alt="Training and validation curves over 25 epochs" width="100%">
</div>

**99.3% on 600 held-out digits** — from a 17,226-parameter network that trains in **0.85 s**:

<div align="center">
<img src="docs/assets/confusion-matrix.png" alt="Confusion matrix, 99.3% accuracy on held-out digits" width="62%">
</div>

And the thing a linear model simply cannot do — a curved decision boundary, learned:

<div align="center">
<img src="docs/assets/decision-boundary.png" alt="Linear model at 88.5% versus EveryO MLP at 99.7% on two moons" width="100%">
</div>

### Convolution, and what it learns

A CNN built from `Conv2D` and `MaxPool2D` reaches **99.2% with 1,898 parameters** — beating a dense
network of comparable size (98.5% with 2,410), because convolution shares its weights across the image.
Below: one digit, the eight feature maps the first layer produces, and the 3×3 kernels that produced them.

<div align="center">
<img src="docs/assets/conv-features.png" alt="Learned convolution filters and their feature maps" width="100%">
</div>

```python
model = eo.Sequential(
    eo.Conv2D(1, 8, 3, padding="same"),
    eo.ReLU(),
    eo.MaxPool2D(2),
    eo.Conv2D(8, 16, 3, padding="same"),
    eo.ReLU(),
    eo.MaxPool2D(2),
    eo.Flatten(),
    eo.Linear(2 * 2 * 16, 10),
)
```

Tensors are `NHWC`, the same layout TensorFlow uses — so the forward pass is compared against
`tf.nn.conv2d` **bit for bit** in the test suite, and every gradient against finite differences.

---

## 🎯 What it actually does

| | Available today |
|---|---|
| **Tensors** | dtypes, devices, broadcasting, indexing, reductions, NumPy interop |
| **Autograd** | reverse-mode gradients for 25+ ops, iterative graph walk, `no_grad`, finite-difference verified |
| **Layers** | `Linear`, `Conv2D`, `MaxPool2D`, `AvgPool2D`, `Flatten`, `Dropout`, `Sequential` + a `Module` base you can subclass |
| **Activations** | ReLU, sigmoid, tanh, softmax, log-softmax — as functions *and* modules |
| **Losses** | MSE, MAE, BCE, BCE-with-logits (numerically stable), cross-entropy |
| **Optimizers** | SGD, momentum, Nesterov, weight decay, Adam, AMSGrad |
| **Data** | `Dataset`, `DataLoader`, CSV loading, scalers, stratified splits |
| **Training** | validation, metrics, early stopping, checkpointing, LR scheduling, gradient clipping |
| **Serialization** | `.evo` archives that **cannot execute code on load** |
| **Visualization** | every chart on this page, headless-safe |
| **CLI** | `everyo info · doctor · benchmark · test · demo` |
| **CUDA** | 5 hand-written kernels + pybind11 bindings, with automatic CPU fallback |
| **TensorFlow** | optional cross-checks and a Keras reference model |

> **Experimental** — CUDA dispatch for tensors on a `cuda` device: kernels are correct and checked against NumPy,
> but each call still copies to the device and back, so measure before relying on it.
>
> **Planned** — normalization layers, attention, GPU-resident tensors, mixed precision, ONNX.
> These are *not* implemented; see the [roadmap](#-roadmap).

---

## 🔬 Correctness is the feature

Anyone can write something that *looks* like a framework. Here is why you can trust this one:

```
gradients      vs finite differences ..... every differentiable op, in float64
matmul         vs TensorFlow ............. 0.000e+00
conv2d         vs TensorFlow ............. 0.000e+00   (valid/same, strided, rectangular)
max/avg pool   vs TensorFlow ............. 0.000e+00
relu           vs TensorFlow ............. 0.000e+00
softmax        vs TensorFlow ............. 2.980e-08
cross-entropy  vs TensorFlow ............. exact to 6 decimal places
its GRADIENT   vs TensorFlow ............. 3.725e-09
```

**506 tests** run in under 10 seconds on CPU. CUDA tests skip themselves without a GPU;
TensorFlow tests skip themselves without TensorFlow. The core suite needs no network, no GPU, and no credentials.

```bash
pytest                    # 498 passed, 8 skipped
./scripts/lint.sh         # ruff check + format check
```

---

## 🏗 Architecture

```
          everyo.cli              everyo.visualization
               └──────────┬───────────────┘
                          │
      everyo.training  (Trainer · callbacks · history · metrics)
                          │
      everyo.optim  (SGD · Adam)      everyo.serialization  (.evo)
                          │
      everyo.nn  (Module · layers · activations · losses)
                          │
      everyo.core  (Tensor · operations · autograd · device · dtype)
                          │
      everyo.backends (numpy · tensorflow)      everyo.cuda
                          │                          │
                        NumPy                optional extension
```

Each layer depends only on the ones beneath it. `core` depends on nothing but NumPy.
The reasoning behind each decision is in **[docs/architecture.md](docs/architecture.md)**.

---

## 🚀 CUDA, without the pain

CUDA is **additive, never required**. On a machine without a GPU everything still runs:

```python
eo.cuda.is_available()  # False — and that is not an error
eo.device("cuda")  # → device('cpu'), with a logged warning
eo.device("cuda", strict=True)  # → raises, when you'd rather know
```

Build the kernels where you do have a GPU:

```bash
pip install -e ".[cuda]" && ./scripts/build_cuda.sh
```

Five kernels live in [`cuda/src`](cuda/src): vector add, element-wise multiply, ReLU, a 16×16 tiled matmul and a
tree-reduction sum — all with bounds checking, CUDA error checking and RAII device memory.
Details in **[docs/cuda.md](docs/cuda.md)**.

---

## 📈 Benchmarks

Numbers come from *your* machine, never from this README:

```bash
everyo benchmark --sizes 128 256 512 1024
python benchmarks/benchmark_training.py --epochs 10
```

Each run warms up, repeats, records mean/best/worst and reports the hardware it used.
EveryO is a readable reference implementation — when a tuned runtime beats it, the benchmark will tell you so.

---

## 🗺 Roadmap

Not implemented yet — contributions very welcome on any of these:

- [x] ~~Convolution and pooling layers~~ — **shipped**: `Conv2D`, `MaxPool2D`, `AvgPool2D`
- [ ] Batch / layer normalization
- [ ] Recurrent layers, attention, transformer blocks
- [ ] GPU-resident tensors (removing per-call transfers)
- [ ] Mixed-precision training
- [ ] ONNX interoperability · model quantization · profiling tools
- [ ] Distributed training

---

## 🤝 Contributing

Good first issues are the roadmap items above, and every one of them is self-contained.

```bash
./scripts/setup.sh      # venv + dev install
./scripts/test.sh       # the suite
./scripts/lint.sh --fix # ruff
```

The rules are short: add a test (gradients need a finite-difference check), keep `ruff check` clean, and make
sure the core still works with neither CUDA nor TensorFlow installed. Full guide in
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## 💛 Support this project

If EveryO helped you understand how autograd actually works:

⭐ **[Star the repo](https://github.com/krishanth7/EveryO)** — it is the single most useful thing you can do.
🐛 **[Open an issue](https://github.com/krishanth7/EveryO/issues)** · 🔧 **[Send a PR](CONTRIBUTING.md)** ·
💬 **Tell one other person who is learning this stuff.**

Sponsorship is configured in [`.github/FUNDING.yml`](.github/FUNDING.yml).

---

## 📚 Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | install, first tensor, first network |
| [Architecture](docs/architecture.md) | how it is layered, and why |
| [Autograd](docs/autograd.md) | how the gradient engine works and how it is verified |
| [CUDA](docs/cuda.md) | building, using and benchmarking the kernels |
| [API reference](docs/api-reference.md) | the full public surface |
| [Examples](examples/) | nine runnable scripts |

---

## 🔒 Security

No API keys. No accounts. No telemetry. No network calls from the core.
Model loading is pickle-free by design — an `.evo` archive cannot execute code.
Report vulnerabilities via **[SECURITY.md](SECURITY.md)**.

## 📄 License

[MIT](LICENSE) — use it, fork it, teach with it, ship it.

<div align="center">
<br>
<b>Built to be read.</b><br>
<sub>If this repo taught you something, a ⭐ helps someone else find it.</sub>
</div>
