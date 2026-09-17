"""Dataset abstractions.

A dataset answers two questions: how many samples are there
(:meth:`Dataset.__len__`) and what is sample *i* (:meth:`Dataset.__getitem__`).
Anything satisfying that contract works with :class:`~everyo.data.dataloader.DataLoader`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from everyo.core.tensor import Tensor
from everyo.exceptions import EveryOShapeError

__all__ = ["Dataset", "ArrayDataset", "TransformDataset", "Subset", "load_csv_dataset"]


class Dataset:
    """Abstract base class for datasets."""

    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError(f"{type(self).__name__} does not implement __len__().")

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(features, target)`` pair at ``index``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement __getitem__().")

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def _to_numpy(values: Any, name: str) -> np.ndarray:
    array = values.data if isinstance(values, Tensor) else np.asarray(values)
    if array.dtype == np.object_:
        raise EveryOShapeError(
            f"{name} could not be converted to a numeric array. Ragged or "
            "non-numeric input is not supported."
        )
    return array


class ArrayDataset(Dataset):
    """Dataset backed by in-memory NumPy arrays.

    Args:
        features: Array of shape ``(n_samples, ...)``.
        targets: Optional array whose first dimension is ``n_samples``.

    Example:
        >>> import numpy as np, everyo as eo
        >>> dataset = eo.ArrayDataset(np.zeros((10, 3)), np.zeros(10))
        >>> len(dataset)
        10
    """

    def __init__(self, features: Any, targets: Any | None = None) -> None:
        self.features = _to_numpy(features, "features")
        if self.features.ndim == 0:
            raise EveryOShapeError("features must have at least one dimension.")
        if targets is None:
            self.targets: np.ndarray | None = None
        else:
            self.targets = _to_numpy(targets, "targets")
            if len(self.targets) != len(self.features):
                raise EveryOShapeError(
                    f"features and targets must contain the same number of "
                    f"samples, got {len(self.features)} and {len(self.targets)}."
                )

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return int(len(self.features))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Return one sample, or a ``(features, target)`` pair when targets exist."""
        if self.targets is None:
            return self.features[index]
        return self.features[index], self.targets[index]

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """Shape of a single feature sample."""
        return tuple(self.features.shape[1:])

    def __repr__(self) -> str:
        target_shape = None if self.targets is None else self.targets.shape
        return (
            f"ArrayDataset(samples={len(self)}, features={self.features.shape}, "
            f"targets={target_shape})"
        )


class TransformDataset(Dataset):
    """Wrap a dataset and apply callables to its features and/or targets.

    Args:
        dataset: The dataset to wrap.
        transform: Applied to the features of every sample.
        target_transform: Applied to the target of every sample.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        target_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Number of samples in the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        """Return the transformed sample at ``index``."""
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            features, target = sample
            if self.transform is not None:
                features = self.transform(features)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return features, target
        return self.transform(sample) if self.transform else sample


class Subset(Dataset):
    """A view over a subset of another dataset, selected by index."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= len(dataset)):
            raise IndexError(
                f"Subset indices must lie in [0, {len(dataset) - 1}] for a "
                f"dataset with {len(dataset)} samples."
            )

    def __len__(self) -> int:
        """Number of selected samples."""
        return int(self.indices.size)

    def __getitem__(self, index: int) -> Any:
        """Return the underlying sample for the ``index``-th selected item."""
        return self.dataset[int(self.indices[index])]


def load_csv_dataset(
    path: str | Path,
    *,
    target_column: int | str | None = -1,
    delimiter: str = ",",
    has_header: bool | None = None,
    skip_columns: Sequence[int] = (),
) -> ArrayDataset:
    """Load a numeric CSV file into an :class:`ArrayDataset`.

    The reader is deliberately minimal: it expects numeric columns, which keeps
    EveryO free of a pandas dependency.  Non-numeric data should be encoded
    before it reaches this function.

    Args:
        path: Path to the CSV file.
        target_column: Index (or header name) of the target column, or ``None``
            to load features only.
        delimiter: Column separator.
        has_header: Force header detection on or off.  Detected automatically
            when ``None``.
        skip_columns: Column indices to drop (for example an ID column).

    Returns:
        An :class:`ArrayDataset` with float32 features.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        EveryOShapeError: If the file cannot be parsed as a numeric table.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as handle:
        lines = [line for line in (raw.strip() for raw in handle) if line]
    if not lines:
        raise EveryOShapeError(f"The CSV file {csv_path} is empty.")

    header: list[str] | None = None
    first_cells = lines[0].split(delimiter)
    looks_like_header = any(_is_not_number(cell) for cell in first_cells)
    if has_header is True or (has_header is None and looks_like_header):
        header = [cell.strip() for cell in first_cells]
        lines = lines[1:]

    try:
        table = np.array(
            [[float(cell) for cell in line.split(delimiter)] for line in lines],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise EveryOShapeError(
            f"Could not parse {csv_path} as a numeric table ({exc}). Encode "
            "categorical columns before loading, and pass has_header=True if "
            "the first row contains column names."
        ) from exc

    if isinstance(target_column, str):
        if header is None or target_column not in header:
            raise EveryOShapeError(
                f"Target column {target_column!r} was not found in the CSV header."
            )
        target_index: int | None = header.index(target_column)
    else:
        target_index = target_column

    drop = {index % table.shape[1] for index in skip_columns}
    if target_index is None:
        keep = [i for i in range(table.shape[1]) if i not in drop]
        return ArrayDataset(table[:, keep])

    target_index %= table.shape[1]
    drop.add(target_index)
    keep = [i for i in range(table.shape[1]) if i not in drop]
    return ArrayDataset(table[:, keep], table[:, target_index])


def _is_not_number(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return True
    return False
