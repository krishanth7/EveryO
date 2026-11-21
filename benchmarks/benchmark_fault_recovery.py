"""CLI benchmark for controlled failure injection and recovery qualification across categories A-F."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from everyo.performance_qualification import run_fault_injection_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute controlled fault injection and recovery validation"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    res = run_fault_injection_suite()

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Wrote fault injection results to {out_p}")

    status = "PASS" if res["all_passed"] else "FAIL"
    print(f"\n==================== EveryO Fault Recovery Report ({status}) ====================")
    print(f"Scenarios Passed: {res['passed_scenarios']}/{res['total_scenarios']}")
    for s in res["scenarios"]:
        m = "PASS" if s["passed"] else "FAIL"
        print(f"  [{m}] ({s['category']:<14}) {s['name']:<32}: {s['detail']}")
    print("================================================================================\n")
    return 0 if res["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
