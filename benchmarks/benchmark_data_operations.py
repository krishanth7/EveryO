"""CLI benchmark for data loading and preprocessing operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import benchmark_data_operations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark EveryO data operations")
    parser.add_argument("--samples", type=int, default=1000, help="Number of dataset samples")
    parser.add_argument("--features", type=int, default=16, help="Feature dimension")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--repeats", type=int, default=10, help="Measured repetitions")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    metrics = benchmark_data_operations(
        sample_count=args.samples,
        features=args.features,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    payload = {
        "benchmark_group": "data_operations",
        "benchmarks": [m.to_dict() for m in metrics],
    }

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote data operations results to {out_p}")

    print(
        f"\n{'Data Op ID':<45} {'Median (ms)':<14} {'p95 (ms)':<12} {'Throughput (samples/s)':<24}"
    )
    print("-" * 97)
    for m in metrics:
        print(
            f"{m.benchmark_id:<45} {m.median_ms:<14.4f} {m.p95_ms:<12.4f} {m.throughput_per_second:<24.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
