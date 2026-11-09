"""Reproducible, dependency-light EveryO CPU quality benchmark.

Run: python -m benchmarks.benchmark_cpu_quality --size 128 --runs 10
"""

from __future__ import annotations

import argparse
import json

import everyo as eo


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the EveryO CPU tensor engine")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if not 1 <= args.size <= 4096:
        parser.error("size must be between 1 and 4096")

    def matmul(rng):
        left = eo.tensor(rng.normal(size=(args.size, args.size)))
        right = eo.tensor(rng.normal(size=(args.size, args.size)))
        return eo.matmul(left, right)

    result = eo.benchmark_cpu(
        "everyo-cpu-matmul", matmul, runs=args.runs, warmups=args.warmups, seed=args.seed
    )
    document = {"benchmark": result.to_dict(), "matrix_size": args.size}
    output = json.dumps(document, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
