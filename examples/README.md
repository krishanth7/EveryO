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
| `save_and_load.py` | Saving, inspecting and reloading an `.evo` archive |
| `tensorflow_backend.py` | Cross-checking EveryO against TensorFlow (optional) |
| `cuda_example.py` | CUDA detection, correctness check and CPU fallback |

Scripts that produce charts write them to `artifacts/`, which is git-ignored.
`tensorflow_backend.py` and `cuda_example.py` both run on machines without
those optional components: they report what is missing and exit cleanly.
