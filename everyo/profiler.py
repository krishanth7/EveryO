"""Low-overhead, opt-in model profiler with Chrome trace export."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

__all__ = ["ProfileEvent", "Profiler", "profile", "record_function"]

_ACTIVE: contextvars.ContextVar[Profiler | None] = contextvars.ContextVar(
    "everyo_active_profiler", default=None
)


@dataclass(frozen=True)
class ProfileEvent:
    """One measured execution interval."""

    name: str
    duration_ns: int
    start_ns: int
    process_id: int
    thread_id: int
    metadata: dict[str, Any]


class Profiler:
    """Collect nested module and user-defined timing events."""

    def __init__(self, *, warmup: int = 0, record_shapes: bool = True) -> None:
        if warmup < 0:
            raise ValueError("warmup must be non-negative.")
        self.warmup = int(warmup)
        self.record_shapes = bool(record_shapes)
        self.events: list[ProfileEvent] = []
        self._seen = 0
        self._token: contextvars.Token[Profiler | None] | None = None

    def __enter__(self) -> Profiler:
        if _ACTIVE.get() is not None:
            raise RuntimeError("EveryO profilers cannot be nested.")
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None

    @contextlib.contextmanager
    def record(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            duration = time.perf_counter_ns() - start
            self._seen += 1
            if self._seen > self.warmup:
                self.events.append(
                    ProfileEvent(
                        name=name,
                        duration_ns=duration,
                        start_ns=start,
                        process_id=os.getpid(),
                        thread_id=0,
                        metadata=metadata or {},
                    )
                )

    def summary(self) -> list[dict[str, Any]]:
        """Aggregate events by name, ordered by total runtime."""
        grouped: dict[str, list[int]] = {}
        for event in self.events:
            grouped.setdefault(event.name, []).append(event.duration_ns)
        rows = [
            {
                "name": name,
                "calls": len(values),
                "total_ms": sum(values) / 1e6,
                "mean_ms": (sum(values) / len(values)) / 1e6,
                "max_ms": max(values) / 1e6,
            }
            for name, values in grouped.items()
        ]
        return sorted(rows, key=lambda row: row["total_ms"], reverse=True)

    def export_json(self, path: str | Path) -> Path:
        """Write raw events to JSON."""
        target = Path(path)
        target.write_text(json.dumps([asdict(event) for event in self.events], indent=2))
        return target

    def export_chrome_trace(self, path: str | Path) -> Path:
        """Write a trace loadable by Chrome/Perfetto."""
        origin = min((event.start_ns for event in self.events), default=0)
        trace = {
            "traceEvents": [
                {
                    "name": event.name,
                    "cat": "everyo",
                    "ph": "X",
                    "ts": (event.start_ns - origin) / 1000,
                    "dur": event.duration_ns / 1000,
                    "pid": event.process_id,
                    "tid": event.thread_id,
                    "args": event.metadata,
                }
                for event in self.events
            ]
        }
        target = Path(path)
        target.write_text(json.dumps(trace))
        return target


def profile(*, warmup: int = 0, record_shapes: bool = True) -> Profiler:
    """Create a profiler context manager."""
    return Profiler(warmup=warmup, record_shapes=record_shapes)


@contextlib.contextmanager
def record_function(name: str, **metadata: Any) -> Iterator[None]:
    """Record a user-defined region when a profiler is active."""
    active = _ACTIVE.get()
    if active is None:
        yield
    else:
        with active.record(name, metadata):
            yield


def _module_scope(name: str, inputs: tuple[Any, ...]) -> contextlib.AbstractContextManager[None]:
    active = _ACTIVE.get()
    if active is None:
        return contextlib.nullcontext()
    metadata: dict[str, Any] = {}
    if active.record_shapes:
        metadata["input_shapes"] = [
            list(value.shape) for value in inputs if hasattr(value, "shape")
        ]
    return active.record(name, metadata)
