"""Unit tests for Phase 21.3.5 recovery after fault injection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import everyo as eo
from everyo.inference import CPUInferenceRuntime


class TestRecoveryBehavior:
    """Verifies that subsequent valid operations succeed after injected failures with zero state corruption."""

    def test_recovery_after_dimension_mismatch(self):
        a = eo.tensor(np.ones((4, 2), dtype=np.float32))
        bad_b = eo.tensor(np.ones((5, 3), dtype=np.float32))
        with pytest.raises(eo.exceptions.EveryOShapeError):
            _ = eo.matmul(a, bad_b)

        # Subsequent valid matmul must succeed
        good_b = eo.tensor(np.ones((2, 3), dtype=np.float32))
        res = eo.matmul(a, good_b)
        assert res.shape == (4, 3)

    def test_recovery_after_nan_rejection(self):
        model = eo.Sequential(eo.Linear(3, 2, seed=1))
        runtime = CPUInferenceRuntime(model)

        with pytest.raises(ValueError):
            _ = runtime.predict(np.array([[1.0, float("nan"), 3.0]], dtype=np.float32))

        # Subsequent valid prediction succeeds
        valid = runtime.predict(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        assert valid.shape == (1, 2)

    def test_recovery_after_archive_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "broken.eo"
            bad_path.write_bytes(b"TRUNCATED")
            with pytest.raises(eo.exceptions.EveryOSerializationError):
                _ = eo.load(bad_path)

            # Subsequent load of valid model succeeds
            good_path = Path(tmpdir) / "valid.eo"
            eo.save(eo.Sequential(eo.Linear(2, 1, seed=2)), good_path)
            loaded = eo.load(good_path)
            assert loaded is not None

    def test_model_weight_immutability_on_error(self):
        layer = eo.Linear(4, 2, seed=42)
        initial_weights = layer.weight.data.copy()

        # Incompatible tensor input
        with pytest.raises(eo.exceptions.EveryOShapeError):
            _ = layer(eo.tensor(np.ones((2, 5), dtype=np.float32)))

        # Weights must be unchanged
        assert np.array_equal(layer.weight.data, initial_weights)
