#!/usr/bin/env python3
"""CLI utility to generate CycloneDX or SPDX Software Bill of Materials (SBOM) for EveryO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path if not installed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CycloneDX or SPDX SBOM for EveryO.",
    )
    parser.add_argument(
        "--format",
        choices=["cyclonedx", "spdx"],
        default="cyclonedx",
        help="SBOM specification standard (default: cyclonedx).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (prints to stdout if omitted).",
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        help="Exclude optional extras (e.g. tensorflow, onnx).",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from everyo.security.sbom import generate_sbom

        sbom_data = generate_sbom(
            format=args.format,
            output_path=args.output,
            include_optional=not args.no_optional,
        )
        if args.output is None:
            print(json.dumps(sbom_data, indent=2, sort_keys=True))
        else:
            print(f"SBOM ({args.format}) successfully generated to {args.output}")
        return 0
    except Exception as exc:
        print(f"Error generating SBOM: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
