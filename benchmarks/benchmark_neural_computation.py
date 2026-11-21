"""CLI benchmark for neural network layers: forward pass, autograd train step, activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import benchmark_neural_computation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark EveryO neural network computation")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[16, 64], help="Batch sizes to evaluate"
    )
    parser.add_argument("--in-features", type=int, default=64, help="Input feature dimension")
    parser.add_argument("--out-features", type=int, default=32, help="Output feature dimension")
    parser.add_argument("--repeats", type=int, default=10, help="Measured repetitions")
    parser.add_argument("--warmup", type=int, default=2, help="Discarded warm-up runs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    metrics = benchmark_neural_computation(
        batch_sizes=args.batch_sizes,
        in_features=args.in_features,
        out_features=args.out_features,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    payload = {
        "benchmark_group": "neural_computation",
        "benchmarks": [m.to_dict() for m in metrics],
    }

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote neural computation results to {out_p}")

    print(
        f"\n{'Neural Op ID':<30} {'Median (ms)':<14} {'p95 (ms)':<12} {'Throughput (samples/s)':<22}"
    )
    print("-" * 80)
    for m in metrics:
        print(
            f"{m.benchmark_id:<30} {m.median_ms:<14.4f} {m.p95_ms:<12.4f} {m.throughput_per_second:<22.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
