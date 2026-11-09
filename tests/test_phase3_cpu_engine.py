"""Phase 3 CPU execution acceptance tests; GPU not required."""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys

import numpy as np
import pytest

from everyo.cpu_engine import (
    ArrayPool,
    OptionalCppAxpy,
    check_cpu_regression,
    fused_affine_relu,
    parallel_map_chunks,
    profile_cpu,
    vectorized_axpy,
    zero_copy_view,
)


def test_pool_reuse_and_rejection() -> None:
    pool = ArrayPool(max_cached_bytes=128)
    first = pool.acquire((4,), np.float64, zero=True)
    assert np.array_equal(first, np.zeros(4))
    pool.release(first)
    assert pool.cached_bytes == 32
    second = pool.acquire((4,), np.float64)
    assert second is first
    assert pool.reuses == 1
    with pytest.raises(ValueError):
        pool.release(np.zeros(4))
    pool.release(second)
    with pytest.raises(ValueError):
        pool.release(second)
    pool.clear()
    assert pool.cached_bytes == 0


def test_pool_boundaries() -> None:
    with pytest.raises(ValueError):
        ArrayPool(max_cached_bytes=-1)
    pool = ArrayPool(max_cached_bytes=0)
    array = pool.acquire((2, 3), np.float32)
    pool.release(array)
    assert pool.cached_bytes == 0
    with pytest.raises(ValueError):
        pool.acquire((-1,))
    with pytest.raises(TypeError):
        pool.acquire((2,), object)


def test_zero_copy_view() -> None:
    matrix = np.arange(12, dtype=np.float64).reshape((3, 4))
    view = zero_copy_view(matrix, (2, 6))
    assert np.shares_memory(view, matrix)
    view[0, 0] = 33
    assert matrix[0, 0] == 33
    with pytest.raises(ValueError, match="copying"):
        zero_copy_view(matrix.T, (12,))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vectorized_axpy(dtype: type) -> None:
    x = np.arange(12, dtype=dtype).reshape(3, 4)
    y = np.ones_like(x)
    out = np.empty_like(x)
    assert vectorized_axpy(x, y, 0.5, out=out) is out
    np.testing.assert_allclose(out, 0.5 * x + y)
    assert vectorized_axpy(x, y, -2).shape == (3, 4)
    with pytest.raises(ValueError):
        vectorized_axpy(x, y[:2], 1)
    with pytest.raises(ValueError):
        vectorized_axpy(x, y, np.inf)


def test_parallel_chunks_order_and_errors() -> None:
    data = np.arange(50_000, dtype=np.float64).reshape(10_000, 5)
    result = parallel_map_chunks(data, lambda x: x * x + 3, workers=4, min_chunk_rows=500)
    np.testing.assert_array_equal(result, data * data + 3)
    assert not np.shares_memory(result, data)
    assert parallel_map_chunks(np.empty((0, 3)), lambda x: x).shape == (0, 3)
    with pytest.raises(ValueError):
        parallel_map_chunks(data, lambda x: x.sum(axis=1), workers=2)
    with pytest.raises(ValueError):
        parallel_map_chunks(data, lambda x: x, workers=0)


def test_fused_affine_relu() -> None:
    rng = np.random.default_rng(23)
    data = rng.normal(size=(128, 32)).astype(np.float64)
    weight = rng.normal(size=(32, 17)).astype(np.float64)
    bias = rng.normal(size=17).astype(np.float64)
    expected = np.maximum(data @ weight + bias, 0)
    out = np.empty((128, 17), dtype=np.float64)
    assert fused_affine_relu(data, weight, bias, out=out) is out
    np.testing.assert_allclose(out, expected, atol=1e-12)
    with pytest.raises(ValueError):
        fused_affine_relu(data, weight[:10], bias)
    with pytest.raises(TypeError):
        fused_affine_relu(data.astype(np.float32), weight, bias)


def test_profile_and_regression_guards() -> None:
    report = profile_cpu(
        lambda: np.arange(500, dtype=np.float64) ** 2, name="square", runs=3, warmups=1
    )
    assert report.median_ms >= 0 and report.output_bytes == 4000
    assert check_cpu_regression(report, report) == 1
    with pytest.raises(RuntimeError):
        check_cpu_regression(
            report, dataclasses.replace(report, median_ms=report.median_ms * 5 + 1)
        )
    with pytest.raises(ValueError, match="checksum"):
        check_cpu_regression(report, dataclasses.replace(report, checksum=report.checksum + 100))
    with pytest.raises(ValueError):
        profile_cpu(lambda: np.array([np.nan]), name="invalid")
    with pytest.raises(ValueError):
        profile_cpu(lambda: np.random.default_rng().normal(size=100), name="random", runs=2)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("c++") is None,
    reason="optional Linux C++ toolchain unavailable",
)
def test_optional_cpp_matches_numpy(tmp_path) -> None:
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "cpu_cpp" / "everyo_axpy.cpp"
    library = tmp_path / "libeveryo_axpy.so"
    subprocess.run(
        ["c++", "-O2", "-std=c++17", "-fPIC", "-shared", str(source), "-o", str(library)],
        check=True,
        timeout=30,
    )
    cpp = OptionalCppAxpy(library)
    x = np.arange(200, dtype=np.float64).reshape(10, 20)
    y = np.ones_like(x)
    np.testing.assert_allclose(cpp.axpy(x, y, 0.25), vectorized_axpy(x, y, 0.25))
    with pytest.raises(TypeError):
        cpp.axpy(x.astype(np.float32), y, 1)
