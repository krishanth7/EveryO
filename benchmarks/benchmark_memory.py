"""CLI benchmark for process RSS memory profiling and resource lifecycle qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import get_process_memory, verify_resource_lifecycle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile memory usage and resource lifecycle")
    parser.add_argument("--iterations", type=int, default=20, help="Cycles per lifecycle check")
    parser.add_argument(
        "--tensor-size", type=int, default=256, help="Matrix dimension for tensor allocation checks"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    mem_before = get_process_memory()
    lifecycle = verify_resource_lifecycle(iterations=args.iterations, tensor_size=args.tensor_size)
    mem_after = get_process_memory()

    payload = {
        "benchmark_group": "memory_and_lifecycle",
        "initial_memory": mem_before,
        "final_memory": mem_after,
        "lifecycle_checks": lifecycle,
    }

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote memory benchmark results to {out_p}")

    print("\n==================== EveryO Memory & Lifecycle Report ====================")
    print(
        f"Initial RSS: {mem_before.get('rss_bytes', 0) / (1024 * 1024):.2f} MB, Final RSS: {mem_after.get('rss_bytes', 0) / (1024 * 1024):.2f} MB"
    )
    print("\nLifecycle Verifications:")
    for name, data in lifecycle["checks"].items():
        status = "PASS" if data.get("passed") else "FAIL"
        delta = data.get("delta_rss_bytes", 0)
        print(f"  [{status}] {name:<35} delta_rss: {delta} bytes")
    print(f"\nOverall Verdict: {'PASS' if lifecycle['passed'] else 'FAIL'}")
    print("========================================================================\n")
    return 0 if lifecycle["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
