"""Reproducible Phase 3 profiling with correctness and no invented speed claims."""

from __future__ import annotations

import argparse
import json

import numpy as np

from everyo.cpu_engine import fused_affine_relu, profile_cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size must be positive")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(args.size, args.size)).astype(np.float64)
    w = rng.normal(size=(args.size, args.size)).astype(np.float64)
    b = rng.normal(size=args.size).astype(np.float64)
    reference = np.maximum(x @ w + b, 0)
    candidate = fused_affine_relu(x, w, b)
    np.testing.assert_allclose(candidate, reference, rtol=1e-12, atol=1e-12)
    baseline = profile_cpu(
        lambda: np.maximum(x @ w + b, 0), name="affine_relu_baseline", runs=args.runs
    )
    fused = profile_cpu(
        lambda: fused_affine_relu(x, w, b), name="affine_relu_fused", runs=args.runs
    )
    print(
        json.dumps(
            {
                "size": args.size,
                "baseline": baseline.to_dict(),
                "fused": fused.to_dict(),
                "median_speed_ratio": baseline.median_ms / fused.median_ms
                if fused.median_ms
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
