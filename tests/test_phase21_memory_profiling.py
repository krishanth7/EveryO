"""Unit tests for Phase 21.3.2 memory measurement and resource lifecycle."""

from __future__ import annotations

import numpy as np

import everyo as eo
from everyo.performance_qualification import (
    MemoryTracker,
    get_process_memory,
    verify_resource_lifecycle,
)


class TestMemoryProfiling:
    """Verifies memory metrics collection and leak detection."""

    def test_get_process_memory(self):
        mem = get_process_memory()
        assert isinstance(mem, dict)
        assert "rss_bytes" in mem
        assert "peak_rss_bytes" in mem
        # On Windows or Linux, RSS should be non-negative integer
        if mem["rss_bytes"] is not None:
            assert mem["rss_bytes"] > 0

    def test_memory_tracker_context_manager(self):
        with MemoryTracker(trace_python=True) as tracker:
            large_arr = np.ones((512, 512), dtype=np.float32)
            t = eo.tensor(large_arr)
            res = t + t
            del large_arr, t, res

        assert tracker.start_memory is not None
        assert tracker.end_memory is not None
        # Verify delta is recorded (int)
        assert isinstance(tracker.delta_rss_bytes, int)

    def test_resource_lifecycle_suite(self):
        lifecycle = verify_resource_lifecycle(iterations=10, tensor_size=64)
        assert lifecycle["passed"] is True
        assert "model_creation_destruction" in lifecycle["checks"]
        assert "serialization_deserialization" in lifecycle["checks"]
        assert "large_tensor_lifecycle" in lifecycle["checks"]
        assert "error_during_model_loading_cleanup" in lifecycle["checks"]
        assert "exception_cleanup" in lifecycle["checks"]
