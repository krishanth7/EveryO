"""CLI benchmark for sustained long-duration soak execution and telemetry degradation monitoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import SOAK_PROFILES, run_soak_test, validate_soak_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute sustained CPU soak test with degradation monitoring"
    )
    parser.add_argument(
        "--profile",
        choices=list(SOAK_PROFILES.keys()),
        default=None,
        help="Named soak profile (short: 600s, medium: 1800s, extended: 7200s, endurance: 28800s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Test duration in seconds (e.g. 600 for 10 min)",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=5.0, help="Seconds between telemetry samples"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Microbatch size for continuous inference"
    )
    parser.add_argument(
        "--trace-python",
        action="store_true",
        default=False,
        help="Enable tracemalloc for detailed Python heap allocation tracking",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    duration = SOAK_PROFILES[args.profile] if args.profile else args.duration
    print(
        f"Starting EveryO Soak Benchmark (Profile: {args.profile or 'custom'}, Duration: {duration:.1f}s, Interval: {args.sample_interval:.1f}s, Trace Python: {args.trace_python})..."
    )

    def progress(sample):
        rss_mb = (sample.rss_bytes or 0) / (1024 * 1024)
        print(
            f"  [{sample.elapsed_seconds:>5.1f}s] completed={sample.completed_ops:<6} rps={sample.instantaneous_rps:>7.1f} p95={sample.p95_latency_ms:>6.3f}ms rss={rss_mb:>6.1f}MB"
        )

    res = run_soak_test(
        duration_seconds=duration,
        sample_interval_seconds=args.sample_interval,
        batch_size=args.batch_size,
        profile=args.profile,
        trace_python=args.trace_python,
        progress_callback=progress,
    )

    # Validate report schema
    is_valid, validation_errors = validate_soak_report(res)
    if not is_valid:
        print(
            f"\nERROR: Soak report failed schema validation: {validation_errors}", file=sys.stderr
        )
        return 1

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nWrote soak telemetry to {out_p}")

    status = "PASS" if res["passed_stability"] else "FAIL"
    print(
        f"\n==================== EveryO Soak Qualification Summary ({status}) ===================="
    )
    print(f"Profile: {res['profile']}, Commit: {res.get('git_commit_sha', 'unknown')[:7]}")
    print(
        f"Completed Operations: {res['completed_operations']}, Overall RPS: {res['overall_throughput_rps']:.1f}"
    )
    print(
        f"Initial RSS: {res['initial_rss_bytes'] / (1024 * 1024):.1f} MB, Final RSS: {res['final_rss_bytes'] / (1024 * 1024):.1f} MB, Peak RSS: {res.get('peak_rss_bytes', 0) / (1024 * 1024):.1f} MB"
    )
    print(
        f"RSS Growth: {res['rss_growth_pct']:.2f}% (Trend: {res.get('rss_trend')}), Latency Drift: {res['latency_drift_pct']:.2f}%"
    )
    print(f"Memory Assessment: {res.get('memory_assessment')}")
    print(f"Python Allocation Tracking: {res.get('python_allocation_status')}")
    print(f"Stability Verdict: {status}")
    print("=====================================================================================\n")
    return 0 if res["passed_stability"] else 1


if __name__ == "__main__":
    sys.exit(main())
