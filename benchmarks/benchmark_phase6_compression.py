"""Repeatable size/latency/quality comparison for EveryO Phase 6 on CPU."""

from __future__ import annotations

import argparse
import json

import numpy as np

from everyo.compression import (
    benchmark_inference,
    compare_model_outputs,
    dynamic_batch_predict,
    model_size_report,
    prune_unstructured,
)
from everyo.nn import Linear, ReLU, Sequential
from everyo.quantization import quantize_dynamic


def measure(*, samples: int = 128, runs: int = 10) -> dict:
    if samples < 1 or runs < 1:
        raise ValueError("samples and runs must be positive")
    model = Sequential(Linear(32, 64, seed=11), ReLU(), Linear(64, 8, seed=12)).eval()
    data = np.random.default_rng(0).normal(size=(samples, 32)).astype(np.float32)
    baseline = dynamic_batch_predict(model, data)
    variants = {
        "baseline": model,
        "pruned_50pct": prune_unstructured(model, 0.5),
        "int8": quantize_dynamic(model),
    }
    return {
        name: {
            "size": model_size_report(candidate),
            "latency": benchmark_inference(candidate, data, repetitions=runs),
            "quality": vars(
                compare_model_outputs(baseline, dynamic_batch_predict(candidate, data))
            ),
        }
        for name, candidate in variants.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(measure(samples=args.samples, runs=args.runs), indent=2))


if __name__ == "__main__":
    main()
