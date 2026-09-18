# Examples

Every example is a runnable script that uses only the public EveryO API and
locally generated data — no downloads, no API keys, no credentials.

Run them from the repository root:

```bash
python examples/tensor_basics.py
```

| Example | What it shows |
| --- | --- |
| `tensor_basics.py` | Tensor creation, arithmetic, broadcasting, reductions, autograd |
| `linear_regression.py` | A single `Linear` layer trained with SGD + momentum |
| `binary_classification.py` | Two-moons problem; MLP vs a linear model |
| `multiclass_classification.py` | Three-class spirals with early stopping |
| `neural_network.py` | The full pipeline on the bundled digit dataset |
| `convolutional_network.py` | A CNN vs a dense baseline, plus the learned filters |
| `sequence_models.py` | RNN/LSTM/GRU vs a transformer on long-range recall |
| `mixed_precision.py` | `autocast` and `GradScaler`, with the underflow they exist to fix |
| `onnx_export.py` | Exporting a CNN and checking it against ONNX Runtime |
| `distributed_training.py` | Data-parallel training across CPU cores, checked against one process |
| `save_and_load.py` | Saving, inspecting and reloading an `.evo` archive |
| `tensorflow_backend.py` | Cross-checking EveryO against TensorFlow (optional) |
| `cuda_example.py` | CUDA detection, correctness check and CPU fallback |

`onnx_export.py` needs the optional `onnx` extra (`pip install everyo[onnx]`) and reports cleanly
when it is missing. Run `distributed_training.py` with `OMP_NUM_THREADS=1` set, or each worker will
start its own BLAS thread pool and they will contend:

```bash
OMP_NUM_THREADS=1 python examples/distributed_training.py
```

Scripts that produce charts write them to `artifacts/`, which is git-ignored.
`tensorflow_backend.py` and `cuda_example.py` both run on machines without
those optional components: they report what is missing and exit cleanly.
