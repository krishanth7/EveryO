"""Training history: the record of what happened during :meth:`Trainer.fit`."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = ["History"]


@dataclass
class History:
    """Per-epoch metric values collected during training.

    Attributes:
        records: One dictionary per epoch, always containing ``"epoch"`` and
            ``"time"`` alongside the recorded metrics.

    Example:
        >>> history = History()
        >>> history.append(epoch=1, loss=0.5)
        >>> history["loss"]
        [0.5]
    """

    records: list[dict[str, float]] = field(default_factory=list)

    def append(self, **values: Any) -> dict[str, float]:
        """Record one epoch and return the stored row."""
        row = {
            key: (float(value) if isinstance(value, (int, float)) else value)
            for key, value in values.items()
        }
        self.records.append(row)
        return row

    @property
    def epochs(self) -> int:
        """Number of recorded epochs."""
        return len(self.records)

    @property
    def keys(self) -> list[str]:
        """Names of every metric recorded at least once."""
        seen: list[str] = []
        for record in self.records:
            for key in record:
                if key not in seen:
                    seen.append(key)
        return seen

    def __getitem__(self, key: str) -> list[float]:
        """Return the series of values recorded for ``key``."""
        if key not in self.keys:
            raise KeyError(
                f"No metric named {key!r} was recorded. Recorded metrics: "
                f"{', '.join(self.keys) or 'none'}."
            )
        return [record[key] for record in self.records if key in record]

    def get(self, key: str, default: list[float] | None = None) -> list[float] | None:
        """Return a metric series, or ``default`` when it was never recorded."""
        return self[key] if key in self.keys else default

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, float]]:
        return iter(self.records)

    def last(self) -> dict[str, float]:
        """Return the most recent epoch record."""
        if not self.records:
            raise IndexError("The history is empty; nothing has been trained yet.")
        return self.records[-1]

    def best(self, key: str = "val_loss", mode: str = "min") -> dict[str, float]:
        """Return the record with the best value of ``key``."""
        series = [record for record in self.records if key in record]
        if not series:
            raise KeyError(f"No metric named {key!r} was recorded.")
        chooser = min if mode == "min" else max
        return chooser(series, key=lambda record: record[key])

    def to_dict(self) -> dict[str, list[float]]:
        """Return the history as ``{metric: [values...]}``."""
        return {key: self[key] for key in self.keys}

    def to_json(self, path: str | Path) -> Path:
        """Write the history to a JSON file and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        return target

    def to_csv(self, path: str | Path) -> Path:
        """Write the history to a CSV file and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        columns = self.keys
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: record.get(key, "") for key in columns})
        return target

    def __repr__(self) -> str:
        return f"History(epochs={self.epochs}, metrics={self.keys})"
