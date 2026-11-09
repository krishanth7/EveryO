"""Local SQLite experiment tracking, resumability and versioned artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from everyo.data_engineering import file_sha256


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ExperimentRegistry:
    """Durable experiment runs with append-only metrics and content-addressed artifacts.

    SQLite transaction locking supports multiple local processes. This is not
    an authentication service; keep the database in a trusted directory.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        self.database = self.root / "experiments.sqlite3"
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    status TEXT NOT NULL, configuration TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL, seed INTEGER NOT NULL,
                    progress TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    step INTEGER NOT NULL, name TEXT NOT NULL, value REAL NOT NULL,
                    PRIMARY KEY (run_id, step, name)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    name TEXT NOT NULL, sha256 TEXT NOT NULL,
                    PRIMARY KEY (run_id, name)
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def start(self, configuration: dict[str, Any], *, dataset_hash: str, seed: int = 0) -> str:
        if not isinstance(configuration, dict):
            raise TypeError("configuration must be a dictionary")
        if (
            not isinstance(dataset_hash, str)
            or len(dataset_hash) != 64
            or any(char not in "0123456789abcdef" for char in dataset_hash)
        ):
            raise ValueError("dataset_hash must be a lowercase SHA-256 digest")
        run_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    "running",
                    _json(configuration),
                    dataset_hash,
                    int(seed),
                    _json({"epoch": 0, "skip": 0}),
                ),
            )
        return run_id

    def get(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, created_at, status, configuration, dataset_hash, seed, progress "
                "FROM runs WHERE id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return {
            "id": row[0],
            "created_at": row[1],
            "status": row[2],
            "configuration": json.loads(row[3]),
            "dataset_hash": row[4],
            "seed": row[5],
            "progress": json.loads(row[6]),
        }

    def update_progress(self, run_id: str, *, epoch: int, skip: int) -> None:
        if epoch < 0 or skip < 0:
            raise ValueError("epoch and skip must be nonnegative")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET progress=? WHERE id=? AND status='running'",
                (_json({"epoch": epoch, "skip": skip}), run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("run is missing or not running")

    def log_metric(self, run_id: str, step: int, name: str, value: float) -> None:
        if step < 0 or not name or not isinstance(name, str):
            raise ValueError("step must be nonnegative and name non-empty")
        numeric = float(value)
        if not (-float("inf") < numeric < float("inf")):
            raise ValueError("metric must be finite")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO metrics (run_id, step, name, value) "
                "SELECT id, ?, ?, ? FROM runs WHERE id=? AND status='running'",
                (step, name, numeric, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("run is missing or not running")

    def finish(self, run_id: str, *, status: str = "completed") -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid terminal status")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status=? WHERE id=? AND status='running'",
                (status, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("run is missing or not running")

    def metrics(self, run_id: str) -> list[dict[str, Any]]:
        self.get(run_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT step, name, value FROM metrics WHERE run_id=? ORDER BY step, name",
                (run_id,),
            ).fetchall()
        return [{"step": s, "name": n, "value": v} for s, n, v in rows]

    def compare(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Return configurations and the latest value of each metric per run."""
        result = []
        for run_id in run_ids:
            entry = self.get(run_id)
            entry["latest_metrics"] = {}
            for metric in self.metrics(run_id):
                entry["latest_metrics"][metric["name"]] = metric["value"]
            result.append(entry)
        return result

    def add_artifact(self, run_id: str, name: str, source: str | Path) -> str:
        """Snapshot a file by checksum, deduplicating identical artifact bytes."""
        self.get(run_id)
        if not name or not isinstance(name, str) or "/" in name or "\\" in name:
            raise ValueError("artifact name must be a simple filename")
        path = Path(source)
        digest = file_sha256(path)
        destination = self.artifacts / digest
        fd, temporary = tempfile.mkstemp(dir=self.artifacts, prefix=".everyo-")
        try:
            with os.fdopen(fd, "wb") as output, path.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
            if file_sha256(temporary) != digest:
                raise ValueError("artifact source changed during snapshot")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?)", (run_id, name, digest)
            )
        return digest

    def artifact(self, run_id: str, name: str) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT sha256 FROM artifacts WHERE run_id=? AND name=?", (run_id, name)
            ).fetchone()
        if row is None:
            raise KeyError((run_id, name))
        path = self.artifacts / row[0]
        if file_sha256(path) != row[0]:
            raise ValueError("artifact checksum mismatch")
        return path

    def manifest(self, run_id: str) -> dict[str, Any]:
        result = self.get(run_id)
        result["metrics"] = self.metrics(run_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, sha256 FROM artifacts WHERE run_id=? ORDER BY name", (run_id,)
            ).fetchall()
        result["artifacts"] = dict(rows)
        result["manifest_sha256"] = hashlib.sha256(_json(result).encode()).hexdigest()
        return result
