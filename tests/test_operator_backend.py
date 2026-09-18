"""Contract tests for unified NumPy/CUDA operator dispatch."""

from __future__ import annotations

import numpy as np

import everyo as eo
from everyo.backends.operator_backend import (
    CudaOperatorBackend,
    NumpyOperatorBackend,
    backend_for,
    dispatch,
)


def test_device_selects_a_backend_without_import_side_effects():
    assert isinstance(backend_for(eo.device("cpu")), NumpyOperatorBackend)
    assert isinstance(backend_for("cuda:0"), CudaOperatorBackend)


def test_numpy_contract_covers_initial_operator_set():
    backend = NumpyOperatorBackend()
    values = np.array([-1.0, 2.0], dtype=np.float32)
    for operation in ("add", "multiply", "relu", "matmul", "sum_all"):
        assert backend.supports(operation, values)


def test_dispatch_reports_cpu_execution():
    values = np.array([-1.0, 2.0], dtype=np.float32)
    result = dispatch("relu", eo.device("cpu"), values)
    assert result.requested_backend == result.executed_backend == "numpy"
    assert not result.fell_back
    np.testing.assert_array_equal(result.value, [0.0, 2.0])


def test_cuda_backend_explains_unsupported_dtype():
    backend = CudaOperatorBackend()
    values = np.array([1.0], dtype=np.float64)
    assert not backend.supports("relu", values)
    reason = backend.unsupported_reason("relu", values)
    assert "unavailable" in reason or "float32" in reason


def test_transfer_boundary_round_trip_is_explicit():
    backend = NumpyOperatorBackend()
    host = np.array([1.0, 2.0], dtype=np.float32)
    placed = backend.from_host(host)
    restored = backend.to_host(placed)
    np.testing.assert_array_equal(restored, host)
