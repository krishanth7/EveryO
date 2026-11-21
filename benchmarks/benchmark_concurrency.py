"""CLI benchmark for concurrent worker execution, thread contention and correctness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import run_concurrency_correctness_qualification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark EveryO concurrent execution correctness"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Number of concurrent worker threads"
    )
    parser.add_argument("--ops-per-worker", type=int, default=50, help="Operations per worker")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    res = run_concurrency_correctness_qualification(
        concurrency=args.concurrency,
        operations_per_worker=args.ops_per_worker,
        seed=args.seed,
    )

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Wrote concurrency benchmark results to {out_p}")

    status = "PASS" if res["passed"] else "FAIL"
    print(f"\n==================== EveryO Concurrency Report ({status}) ====================")
    print(f"Workers: {res['concurrency']}, Total Operations: {res['total_operations']}")
    print(f"Duration: {res['duration_seconds']:.2f}s, Throughput: {res['throughput_rps']:.1f} RPS")
    print(f"p50: {res['p50_ms']:.4f}ms, p95: {res['p95_ms']:.4f}ms, p99: {res['p99_ms']:.4f}ms")
    print(f"Corruptions Detected: {res['corruptions_detected']}")
    print("========================================================================\n")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
