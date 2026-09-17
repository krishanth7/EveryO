"""Charts for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from everyo.exceptions import EveryOError
from everyo.training.history import History
from everyo.visualization._backend import create_figure, get_pyplot

__all__ = ["plot_loss", "plot_accuracy", "plot_history", "plot_metric"]

_DEFAULT_FIGSIZE = (8.0, 4.5)


def _finish(figure: Any, save_path: str | Path | None, show: bool) -> Any:
    """Save and/or display a figure, returning it for further customisation."""
    figure.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=150, bbox_inches="tight")
    if show:
        get_pyplot().show()
    return figure


def _as_history(history: History | dict[str, Sequence[float]]) -> dict[str, list[float]]:
    if isinstance(history, History):
        return history.to_dict()
    if isinstance(history, dict):
        return {key: list(values) for key, values in history.items()}
    raise TypeError(
        f"Expected an everyo History or a dict of metric series, got {type(history).__name__}."
    )


def plot_metric(
    history: History | dict[str, Sequence[float]],
    keys: Sequence[str],
    *,
    title: str = "Training metric",
    ylabel: str = "value",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Any:
    """Plot one or more metric series against the epoch number.

    Args:
        history: A :class:`~everyo.training.history.History` or a plain dict.
        keys: Metric names to draw.
        title: Chart title.
        ylabel: Label of the y axis.
        save_path: Optional PNG destination.
        show: Open an interactive window.
        figsize: Figure size in inches.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        EveryOError: If none of ``keys`` is present in the history.
    """
    series = _as_history(history)
    present = [key for key in keys if key in series and series[key]]
    if not present:
        raise EveryOError(
            f"None of the requested metrics {list(keys)} were recorded. "
            f"Recorded metrics: {', '.join(sorted(series)) or 'none'}."
        )

    figure, axes = create_figure(lambda plt: plt.subplots(figsize=figsize))
    for key in present:
        values = series[key]
        axes.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=key)
    axes.set_xlabel("epoch")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    return _finish(figure, save_path, show)


def plot_loss(
    history: History | dict[str, Sequence[float]],
    *,
    title: str = "Training loss",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Any:
    """Plot training (and validation) loss curves.

    Example:
        >>> import everyo as eo
        >>> history = eo.History()
        >>> _ = history.append(epoch=1, loss=1.0)
        >>> _ = history.append(epoch=2, loss=0.5)
        >>> figure = eo.plot_loss(history)
    """
    return plot_metric(
        history,
        ("loss", "val_loss"),
        title=title,
        ylabel="loss",
        save_path=save_path,
        show=show,
        figsize=figsize,
    )


def plot_accuracy(
    history: History | dict[str, Sequence[float]],
    *,
    title: str = "Accuracy",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> Any:
    """Plot training (and validation) accuracy curves."""
    return plot_metric(
        history,
        ("accuracy", "val_accuracy", "binary_accuracy", "val_binary_accuracy"),
        title=title,
        ylabel="accuracy",
        save_path=save_path,
        show=show,
        figsize=figsize,
    )


def plot_history(
    history: History | dict[str, Sequence[float]],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (11.0, 4.5),
) -> Any:
    """Draw loss and accuracy side by side when both are available."""
    series = _as_history(history)
    has_accuracy = any(key in series for key in ("accuracy", "val_accuracy", "binary_accuracy"))
    if not has_accuracy:
        return plot_loss(history, save_path=save_path, show=show)

    figure, (left, right) = create_figure(lambda plt: plt.subplots(1, 2, figsize=figsize))
    for key in ("loss", "val_loss"):
        if key in series:
            left.plot(
                range(1, len(series[key]) + 1), series[key], marker="o", markersize=3, label=key
            )
    left.set(xlabel="epoch", ylabel="loss", title="Loss")
    left.grid(True, alpha=0.3)
    left.legend()

    for key in ("accuracy", "val_accuracy", "binary_accuracy", "val_binary_accuracy"):
        if key in series:
            right.plot(
                range(1, len(series[key]) + 1), series[key], marker="o", markersize=3, label=key
            )
    right.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    right.grid(True, alpha=0.3)
    right.legend()
    return _finish(figure, save_path, show)
