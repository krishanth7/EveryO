"""Charts for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from everyo.exceptions import EveryOError
from everyo.visualization._backend import create_figure
from everyo.visualization.training import _finish

__all__ = ["plot_benchmark", "plot_benchmark_bars"]


def _grouped(results: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if "backend" not in row:
            raise EveryOError(f"Each benchmark row needs a 'backend' key; got keys {sorted(row)}.")
        groups.setdefault(str(row["backend"]), []).append(row)
    return groups


def plot_benchmark(
    results: Sequence[dict[str, Any]],
    *,
    x_key: str = "size",
    y_key: str = "seconds",
    title: str = "Benchmark",
    log_scale: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Any:
    """Plot measured execution time against problem size, one line per backend.

    Args:
        results: Rows produced by the scripts in ``benchmarks/``, each with at
            least ``backend``, ``x_key`` and ``y_key``.
        x_key: Row key used for the x axis.
        y_key: Row key used for the y axis.
        log_scale: Use logarithmic axes, which suits timing data.
    """
    if not results:
        raise EveryOError("plot_benchmark() received an empty result list.")

    figure, axes = create_figure(lambda plt: plt.subplots(figsize=figsize))
    for backend, rows in _grouped(results).items():
        ordered = sorted(rows, key=lambda row: row[x_key])
        axes.plot(
            [row[x_key] for row in ordered],
            [row[y_key] for row in ordered],
            marker="o",
            markersize=4,
            label=backend,
        )
    if log_scale:
        axes.set_xscale("log", base=2)
        axes.set_yscale("log")
    axes.set(xlabel=x_key, ylabel=y_key, title=title)
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    return _finish(figure, save_path, show)


def plot_benchmark_bars(
    results: Sequence[dict[str, Any]],
    *,
    value_key: str = "seconds",
    label_key: str = "backend",
    title: str = "Benchmark comparison",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> Any:
    """Draw a bar chart comparing one measurement per backend."""
    if not results:
        raise EveryOError("plot_benchmark_bars() received an empty result list.")

    labels = [str(row[label_key]) for row in results]
    values = [float(row[value_key]) for row in results]

    figure, axes = create_figure(lambda plt: plt.subplots(figsize=figsize))
    positions = np.arange(len(labels))
    bars = axes.bar(positions, values, color="steelblue")
    axes.set_xticks(positions, labels, rotation=20, ha="right")
    axes.set(ylabel=value_key, title=title)
    axes.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _finish(figure, save_path, show)
