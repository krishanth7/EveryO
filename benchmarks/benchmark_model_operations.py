"""CLI benchmark for model initialization, serialization, deserialization and inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import benchmark_model_operations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark EveryO model operations")
    parser.add_argument("--repeats", type=int, default=5, help="Measured repetitions")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    metrics = benchmark_model_operations(warmup=args.warmup, repeats=args.repeats, seed=args.seed)
    payload = {
        "benchmark_group": "model_operations",
        "benchmarks": [m.to_dict() for m in metrics],
    }

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote model operations results to {out_p}")

    print(f"\n{'Model Op ID':<30} {'Median (ms)':<14} {'p95 (ms)':<12} {'Throughput (ops/s)':<20}")
    print("-" * 78)
    for m in metrics:
        print(
            f"{m.benchmark_id:<30} {m.median_ms:<14.4f} {m.p95_ms:<12.4f} {m.throughput_per_second:<20.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
