"""Matplotlib charts for training runs, evaluation and benchmarks.

Charts never require a GUI: in a headless environment the Agg backend is
selected automatically and figures can be written straight to PNG.
"""

from __future__ import annotations

from everyo.visualization._backend import is_available, using_headless_backend
from everyo.visualization.benchmark import plot_benchmark, plot_benchmark_bars
from everyo.visualization.metrics import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_predictions,
)
from everyo.visualization.training import (
    plot_accuracy,
    plot_history,
    plot_loss,
    plot_metric,
)

__all__ = [
    "is_available",
    "plot_accuracy",
    "plot_benchmark",
    "plot_benchmark_bars",
    "plot_confusion_matrix",
    "plot_decision_boundary",
    "plot_history",
    "plot_loss",
    "plot_metric",
    "plot_predictions",
    "using_headless_backend",
]
