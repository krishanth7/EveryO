"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from everyo.benchmarking import environment

DEFAULT_OUTPUT_DIR = Path("benchmark_results")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments every benchmark script accepts."""
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions per measurement.")
    parser.add_argument("--warmup", type=int, default=2, help="Discarded warm-up runs.")
    parser.add_argument(
        "--output",
        default=None,
        help="Write results to this file (.json or .csv). Defaults to no file.",
    )
    parser.add_argument("--plot", default=None, help="Write a PNG chart to this path.")
    return parser


def print_table(results: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    """Print benchmark rows as a fixed-width table."""
    if not results:
        print("No results: no backend was available.")
        return

    def render(value: Any) -> str:
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    rendered = [[render(row.get(column, "")) for column in columns] for row in results]
    widths = [
        max(len(column), *(len(row[index]) for row in rendered)) + 2
        for index, column in enumerate(columns)
    ]
    header = "".join(column.ljust(width) for column, width in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rendered:
        print("".join(cell.ljust(width) for cell, width in zip(row, widths)))


def export(results: Sequence[dict[str, Any]], path: str | None) -> None:
    """Write results to JSON or CSV, including the machine description."""
    if not path or not results:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.suffix.lower() == ".csv":
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    else:
        payload = {"environment": environment(), "results": list(results)}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} row(s) to {target}")


def plot(results: Sequence[dict[str, Any]], path: str | None, *, title: str, y_key: str) -> None:
    """Write a chart of the results when ``path`` is given."""
    if not path or not results:
        return
    from everyo.visualization import plot_benchmark

    plot_benchmark(results, y_key=y_key, title=title, save_path=path)
    print(f"Wrote chart to {path}")


def print_environment() -> None:
    """Print the machine and library versions the benchmark ran on."""
    print("Environment")
    print("-" * 40)
    for key, value in environment().items():
        print(f"  {key:<18} {value}")
    print()
