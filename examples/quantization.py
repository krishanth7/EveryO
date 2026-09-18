"""Compare a float model with its dynamic-int8 inference copy."""

import numpy as np

import everyo as eo
from everyo.quantization import quantize_dynamic

model = eo.Sequential(eo.Linear(32, 64, seed=0), eo.ReLU(), eo.Linear(64, 8, seed=1)).eval()
inputs = eo.tensor(np.random.default_rng(0).normal(size=(16, 32)).astype(np.float32))
quantized = quantize_dynamic(model)

error = np.max(np.abs(model(inputs).numpy() - quantized(inputs).numpy()))
print(f"maximum output error: {error:.6f}")
