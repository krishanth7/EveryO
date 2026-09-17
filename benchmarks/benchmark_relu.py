"""Benchmark the ReLU activation across every available backend.

Run with:
    python benchmarks/benchmark_relu.py
"""

from __future__ import annotations

import argparse

from _common import add_common_arguments, export, plot, print_environment, print_table

from everyo.benchmarking import run_relu_benchmark


def main() -> None:
    """Parse arguments, run the benchmark and report the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100_000, 1_000_000, 10_000_000],
        help="Number of elements per measurement.",
    )
    add_common_arguments(parser)
    args = parser.parse_args()

    print_environment()
    print(f"ReLU activation ({args.repeats} repeats, {args.warmup} warm-up runs)\n")

    results = run_relu_benchmark(sizes=args.sizes, repeats=args.repeats, warmup=args.warmup)
    print_table(results, ["backend", "size", "seconds", "best", "elements_per_second"])
    export(results, args.output)
    plot(results, args.plot, title="ReLU activation", y_key="seconds")


if __name__ == "__main__":
    main()
