# Phase 6 — Model Compression and Inference

Everything in this phase runs on CPUs. ONNX export is an optional integration.

| Stage | Implemented |
| --- | --- |
| 6.1 | Deterministic magnitude-based unstructured pruning and adjacent Linear channel pruning |
| 6.2 | `FakeQuantLinear` for QAT research: fake-quantized float forward and straight-through backward |
| 6.3 | Temperature-scaled KL knowledge distillation, optionally combined with hard-label loss |
| 6.4 | Model parameter/buffer size analysis and repeated CPU inference latency measurement |
| 6.5 | ONNX export bridge for quantized / fake-quantized Linear with multi-batch equivalence gate |
| 6.6 | `dynamic_batch_predict` for arrays of variable batch length and no-grad execution |
| 6.7 | Absolute/relative drift reports and tolerance-based output comparison tests |

```python
import numpy as np
import everyo as eo
from everyo.compression import (
    prune_unstructured,
    model_size_report,
    benchmark_inference,
    dynamic_batch_predict,
    compare_model_outputs,
)

model = eo.Sequential(eo.Linear(4, 8), eo.ReLU(), eo.Linear(8, 2)).eval()
compressed = prune_unstructured(model, 0.5)
x = np.ones((7, 4), dtype=np.float32)
print(model_size_report(compressed))
print(benchmark_inference(compressed, x, repetitions=5))
print(compare_model_outputs(dynamic_batch_predict(model, x), dynamic_batch_predict(compressed, x)))
```

Benchmark on your machine:

```bash
python -m benchmarks.benchmark_phase6_compression --samples 128 --runs 10
```

## Qualification boundaries

- Unstructured pruning retains dense NumPy matrices, so **does not reduce
  memory or execution time** without a sparse backend. The structured helper
  returns *smaller* adjacent Linear layers, but the caller must reconnect them
  correctly (only channel-preserving operations may sit between them).
- Pruning generally changes outputs. Measure accuracy on representative test
  data before accepting a compression ratio. Our test suite establishes
  arithmetic correctness, not a production accuracy threshold.
- Fake quantization uses a straight-through estimator (STE), not the exact
  derivative of rounding; quantization scales are dynamic and not calibrated.
- ONNX conversion restores float matrices from quantized Linear modules. It
  **does not create int8 ONNX operators**. Two batch sizes are verified against
  ONNX Runtime, with explicitly configurable drift tolerances; the bridge
  rejects exports outside those tolerances.
- `model_size_report` measures parameter/buffer storage, not resident process
  memory. Latency measurements are hardware- and workload-dependent, and a
  compression change is not claimed to be faster unless benchmarks show it.
