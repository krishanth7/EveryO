"""Mini-batch iteration over a dataset."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np

from everyo.data.dataset import ArrayDataset, Dataset

__all__ = ["DataLoader", "default_collate"]


def default_collate(samples: Sequence[Any]) -> Any:
    """Stack a list of samples into batched NumPy arrays."""
    first = samples[0]
    if isinstance(first, tuple):
        columns = zip(*samples)
        return tuple(np.stack([np.asarray(item) for item in column]) for column in columns)
    return np.stack([np.asarray(sample) for sample in samples])


class DataLoader:
    """Iterate over a dataset in batches, optionally shuffled.

    Args:
        dataset: Any object implementing the :class:`~everyo.data.dataset.Dataset`
            contract.
        batch_size: Number of samples per batch.
        shuffle: Reshuffle the sample order at the start of every epoch.
        drop_last: Drop the final, smaller batch when the dataset size is not a
            multiple of ``batch_size``.
        seed: Seed for the shuffling generator, for reproducible epochs.

    Example:
        >>> import numpy as np, everyo as eo
        >>> loader = eo.DataLoader(eo.ArrayDataset(np.zeros((10, 2)), np.zeros(10)),
        ...                        batch_size=4)
        >>> len(loader)
        3
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError(
                "DataLoader needs a dataset implementing __len__ and __getitem__. "
                "Wrap plain arrays in eo.ArrayDataset first."
            )
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # Batching whole array slices is much faster than sample-by-sample
        # indexing, so the fast path is used whenever possible.
        self._fast_path = isinstance(dataset, ArrayDataset)

    def __len__(self) -> int:
        """Number of batches produced per epoch."""
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    @property
    def num_samples(self) -> int:
        """Number of samples in the underlying dataset."""
        return len(self.dataset)

    def __iter__(self) -> Iterator[Any]:
        """Yield one batch at a time."""
        total = len(self.dataset)
        order = self._rng.permutation(total) if self.shuffle else np.arange(total)

        for start in range(0, total, self.batch_size):
            indices = order[start : start + self.batch_size]
            if self.drop_last and len(indices) < self.batch_size:
                break
            yield self._build_batch(indices)

    def _build_batch(self, indices: np.ndarray) -> Any:
        if self._fast_path:
            dataset: ArrayDataset = self.dataset  # type: ignore[assignment]
            features = dataset.features[indices]
            if dataset.targets is None:
                return features
            return features, dataset.targets[indices]
        return default_collate([self.dataset[int(index)] for index in indices])

    def __repr__(self) -> str:
        return (
            f"DataLoader(samples={self.num_samples}, batch_size={self.batch_size}, "
            f"batches={len(self)}, shuffle={self.shuffle})"
        )
