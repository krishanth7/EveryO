"""Charts for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from everyo.training.metrics import confusion_matrix as compute_confusion_matrix
from everyo.visualization._backend import get_pyplot
from everyo.visualization.training import _finish

__all__ = ["plot_confusion_matrix", "plot_predictions", "plot_decision_boundary"]


def plot_confusion_matrix(
    predictions: Any,
    targets: Any,
    *,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (6.0, 5.5),
) -> Any:
    """Draw a confusion matrix as an annotated heat map."""
    plt = get_pyplot()
    matrix = compute_confusion_matrix(predictions, targets).astype(np.float64)
    if normalize:
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)

    labels = list(class_names) if class_names else [str(i) for i in range(len(matrix))]
    figure, axes = plt.subplots(figsize=figsize)
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes.set_yticks(range(len(labels)), labels)
    axes.set(xlabel="predicted", ylabel="true", title=title)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    fmt = "{:.2f}" if normalize else "{:.0f}"
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(
                column,
                row,
                fmt.format(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=9,
            )
    return _finish(figure, save_path, show)


def plot_predictions(
    features: Any,
    targets: Any,
    predictions: Any,
    *,
    title: str = "Predictions vs targets",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> Any:
    """Scatter regression predictions against the true values.

    For 1-D inputs the chart is drawn against the feature; otherwise predictions
    are drawn against targets with the ideal ``y = x`` line.
    """
    plt = get_pyplot()
    x = np.asarray(features)
    y_true = np.asarray(targets).reshape(-1)
    y_pred = np.asarray(predictions).reshape(-1)

    figure, axes = plt.subplots(figsize=figsize)
    if x.ndim == 2 and x.shape[1] == 1:
        order = np.argsort(x.reshape(-1))
        axes.scatter(x.reshape(-1)[order], y_true[order], s=12, alpha=0.6, label="target")
        axes.plot(x.reshape(-1)[order], y_pred[order], color="crimson", label="prediction")
        axes.set(xlabel="feature", ylabel="value")
    else:
        axes.scatter(y_true, y_pred, s=12, alpha=0.6)
        limits = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
        axes.plot(limits, limits, "--", color="grey", label="perfect prediction")
        axes.set(xlabel="target", ylabel="prediction")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    return _finish(figure, save_path, show)


def plot_decision_boundary(
    predict_fn: Any,
    features: Any,
    labels: Any,
    *,
    resolution: int = 200,
    title: str = "Decision boundary",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (6.5, 5.5),
) -> Any:
    """Visualise a 2-D classifier's decision regions.

    Args:
        predict_fn: Callable mapping an ``(n, 2)`` array to class indices.
        features: The ``(n, 2)`` training inputs.
        labels: Integer class labels for the training inputs.
        resolution: Grid resolution per axis.
    """
    plt = get_pyplot()
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(
            f"plot_decision_boundary() needs 2-D features with shape (n, 2), got {x.shape}."
        )
    y = np.asarray(labels).reshape(-1)

    padding = 0.5
    xs = np.linspace(x[:, 0].min() - padding, x[:, 0].max() + padding, resolution)
    ys = np.linspace(x[:, 1].min() - padding, x[:, 1].max() + padding, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    zones = np.asarray(predict_fn(grid)).reshape(grid_x.shape)

    figure, axes = plt.subplots(figsize=figsize)
    axes.contourf(grid_x, grid_y, zones, alpha=0.25, cmap="viridis")
    axes.scatter(x[:, 0], x[:, 1], c=y, s=14, cmap="viridis", edgecolors="k", linewidths=0.3)
    axes.set(xlabel="feature 1", ylabel="feature 2", title=title)
    return _finish(figure, save_path, show)
