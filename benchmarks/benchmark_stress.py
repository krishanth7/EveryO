"""CLI benchmark for high-throughput stress profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import (
    STRESS_PROFILE_A_NORMAL,
    STRESS_PROFILE_B_MODERATE,
    STRESS_PROFILE_C_HIGH,
    run_stress_qualification,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute stress profiles against EveryO runtime")
    parser.add_argument(
        "--profile",
        choices=["normal", "moderate", "high"],
        default="moderate",
        help="Stress profile intensity level",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    if args.profile == "normal":
        prof = STRESS_PROFILE_A_NORMAL
    elif args.profile == "high":
        prof = STRESS_PROFILE_C_HIGH
    else:
        prof = STRESS_PROFILE_B_MODERATE

    res = run_stress_qualification(profile=prof)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Wrote stress benchmark results to {out_p}")

    status = "PASS" if res["passed"] else "FAIL"
    print(
        f"\n==================== EveryO Stress Report ({prof.name.upper()}: {status}) ===================="
    )
    print(f"Concurrency: {res['concurrency']}, Total Requests: {res['total_requests']}")
    print(f"Duration: {res['duration_seconds']:.2f}s, Throughput: {res['throughput_rps']:.1f} RPS")
    print(
        f"Successes: {res['successes']}, Failures: {res['failures']}, Error Rate: {res['error_rate'] * 100:.2f}%"
    )
    print(
        f"p50: {res['p50_ms']:.4f}ms, p90: {res['p90_ms']:.4f}ms, p95: {res['p95_ms']:.4f}ms, p99: {res['p99_ms']:.4f}ms"
    )
    print(
        "========================================================================================\n"
    )
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
