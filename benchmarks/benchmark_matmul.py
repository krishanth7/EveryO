"""Benchmark matrix multiplication across every available backend.

All numbers are measured on the machine running the script; nothing is
hardcoded. Backends that are unavailable are simply absent from the results.

Run with:
    python benchmarks/benchmark_matmul.py --sizes 128 256 512
"""

from __future__ import annotations

import argparse

from _common import add_common_arguments, export, plot, print_environment, print_table

from everyo.benchmarking import run_matmul_benchmark


def main() -> None:
    """Parse arguments, run the benchmark and report the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128, 256, 512], help="Square matrix sizes."
    )
    add_common_arguments(parser)
    args = parser.parse_args()

    print_environment()
    print(f"Matrix multiplication ({args.repeats} repeats, {args.warmup} warm-up runs)\n")

    results = run_matmul_benchmark(sizes=args.sizes, repeats=args.repeats, warmup=args.warmup)
    print_table(results, ["backend", "size", "seconds", "best", "gflops"])
    export(results, args.output)
    plot(results, args.plot, title="Matrix multiplication", y_key="seconds")


if __name__ == "__main__":
    main()
