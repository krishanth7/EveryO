"""Append-only experiment event journal in JSON Lines (no database required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentTracker:
    def __init__(self, path: str | Path, *, run_id: str):
        if not run_id or any(c in run_id for c in "\r\n"):
            raise ValueError("run_id must be a nonempty single-line identifier")
        self.path = Path(path)
        self.run_id = run_id

    def log(
        self, step: int, metrics: dict[str, float], *, config: dict[str, Any] | None = None
    ) -> None:
        if not isinstance(step, int) or step < 0 or not metrics:
            raise ValueError("step must be nonnegative and metrics nonempty")
        import math

        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError("metrics must be finite numbers")
        event = {"run_id": self.run_id, "step": step, "metrics": metrics, "config": config or {}}
        serialized = json.dumps(event, allow_nan=False, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as stream:
            return [row for line in stream if (row := json.loads(line))["run_id"] == self.run_id]
