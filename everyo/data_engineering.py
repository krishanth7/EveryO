"""Deterministic, streaming CPU dataset utilities (Phase 4)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: str | Path) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StreamingDataset:
    """Re-iterable dataset, optionally deterministically shuffled and augmented.

    The source must be a callable producing a *fresh* iterable for each epoch.
    Shuffle uses a bounded buffer, not a global permutation; examples should
    therefore not be assumed to be uniformly permuted over the whole dataset.
    """

    def __init__(
        self,
        source: Callable[[], Iterable[Any]],
        *,
        seed: int = 0,
        shuffle_buffer: int = 1,
        transform: Callable[[Any, np.random.Generator], Any] | None = None,
    ) -> None:
        if not callable(source):
            raise TypeError("source must be a callable producing an iterable")
        if shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be at least one")
        self.source = source
        self.seed = int(seed)
        self.shuffle_buffer = shuffle_buffer
        self.transform = transform

    def iter_epoch(self, epoch: int = 0, *, skip: int = 0) -> Iterator[Any]:
        """Yield an epoch; ``skip`` allows deterministic mid-epoch resumption."""
        if epoch < 0 or skip < 0:
            raise ValueError("epoch and skip must be non-negative")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        iterator = iter(self.source())

        def ordered() -> Iterator[Any]:
            if self.shuffle_buffer == 1:
                yield from iterator
                return
            buffer = []
            for value in iterator:
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(value)
                    continue
                index = int(rng.integers(len(buffer)))
                yield buffer[index]
                buffer[index] = value
            while buffer:
                index = int(rng.integers(len(buffer)))
                yield buffer.pop(index)

        for index, item in enumerate(ordered()):
            # Even skipped transforms consume the same random sequence, so
            # resumed and uninterrupted streams produce identical values.
            result = self.transform(item, rng) if self.transform is not None else item
            if index >= skip:
                yield result

    def batches(self, batch_size: int, *, epoch: int = 0, skip: int = 0) -> Iterator[list[Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batch: list[Any] = []
        for item in self.iter_epoch(epoch, skip=skip):
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class DatasetCache:
    """Disk-backed, checksum-verified cache of numeric NumPy arrays.

    Cache keys are logical identifiers, never filesystem paths. Writes use a
    temporary file and atomic replace. Never load untrusted object arrays.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("cache name must be a non-empty string")
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    def put(self, name: str, array: np.ndarray) -> str:
        data = np.asarray(array)
        if data.dtype.hasobject:
            raise TypeError("object arrays are not supported")
        key = self._key(name)
        fd, temporary = tempfile.mkstemp(prefix=".everyo-", suffix=".npy", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, data, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            digest = file_sha256(temporary)
            os.replace(temporary, self.root / f"{key}.npy")
            manifest = {
                "name": name,
                "sha256": digest,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            }
            self._atomic_json(self.root / f"{key}.json", manifest)
            return digest
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _atomic_json(destination: Path, data: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".everyo-", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, name: str, *, mmap: bool = False) -> np.ndarray:
        key = self._key(name)
        info = json.loads((self.root / f"{key}.json").read_text(encoding="utf-8"))
        target = self.root / f"{key}.npy"
        if info["name"] != name or file_sha256(target) != info["sha256"]:
            raise ValueError("dataset cache checksum or identity mismatch")
        result = np.load(target, allow_pickle=False, mmap_mode="r" if mmap else None)
        if list(result.shape) != info["shape"] or str(result.dtype) != info["dtype"]:
            raise ValueError("dataset cache metadata mismatch")
        return result
