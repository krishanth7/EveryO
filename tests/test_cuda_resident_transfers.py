"""Verify that GPU-resident tensors actually remove per-call transfers.

``tests/test_cuda_resident.py`` checks the real thing and skips without a GPU,
which on a CPU-only machine leaves the feature's whole point -- that a chain of
operations uploads once instead of once per call -- untested.

That point does not actually need a GPU to check. It is a claim about *how many
times the boundary is crossed*, not about what happens on the far side. So
these tests substitute a stand-in for the native extension that implements the
same binding surface in NumPy and counts every crossing.

What this does and does not establish:

- It **does** establish that the resident API crosses the boundary once per
  upload and only when asked to download, and that a chain of N operations does
  not scale its transfers with N.
- It **does not** establish that the CUDA kernels compute correct results, that
  the native extension builds, or anything about performance. No GPU, no CUDA
  toolkit and no compiler for it are present here; ``test_cuda_resident.py``
  covers that and skips, and this file is not a substitute for it.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.cuda import resident
from everyo.exceptions import EveryOCudaError, EveryOShapeError


class _Counter:
    """Tallies host/device crossings."""

    def __init__(self) -> None:
        self.uploads = 0
        self.downloads = 0
        self.kernels = 0


class _FakeDeviceTensor:
    """Stands in for the pybind11 ``DeviceTensor``.

    Mirrors the binding surface in ``cuda/bindings/bindings.cpp``: construction
    from a host array is an upload, ``numpy()`` is a download, and results of
    device operations are constructed without either.
    """

    def __init__(self, array: np.ndarray, counter: _Counter, *, uploaded: bool = True) -> None:
        self._array = np.ascontiguousarray(array, dtype=np.float32)
        self._counter = counter
        if uploaded:
            counter.uploads += 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def size(self) -> int:
        return int(self._array.size)

    def numpy(self) -> np.ndarray:
        self._counter.downloads += 1
        return self._array


class _FakeExtension:
    """The module-level functions the resident API calls."""

    def __init__(self, counter: _Counter) -> None:
        self._counter = counter

    def _result(self, array: np.ndarray) -> _FakeDeviceTensor:
        # A kernel result is born on the device: no crossing.
        self._counter.kernels += 1
        return _FakeDeviceTensor(array, self._counter, uploaded=False)

    def DeviceTensor(self, array: np.ndarray) -> _FakeDeviceTensor:  # noqa: N802 - mirrors C++
        return _FakeDeviceTensor(array, self._counter)

    def device_add(self, a, b):
        return self._result(a._array + b._array)

    def device_multiply(self, a, b):
        return self._result(a._array * b._array)

    def device_matmul(self, a, b):
        return self._result(a._array @ b._array)

    def device_relu(self, x):
        return self._result(np.maximum(x._array, 0.0))

    def device_sum(self, x):
        # Mirrors the binding: reduces on the device and returns a scalar, so
        # it is not a tensor download.
        self._counter.kernels += 1
        return float(x._array.sum())


@pytest.fixture
def device(monkeypatch):
    """Install the counting stand-in and hand back its tally."""
    counter = _Counter()
    extension = _FakeExtension(counter)
    monkeypatch.setattr(resident, "get_extension", lambda: extension)
    return counter


@pytest.fixture
def operands():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    b = np.eye(4, dtype=np.float32) * 2.0
    return a, b


def test_a_chain_uploads_once_per_input_and_never_again(device, operands):
    """The headline claim: transfers do not scale with the length of the chain."""
    a, b = operands
    da, db = eo.cuda.to_device(a), eo.cuda.to_device(b)
    assert device.uploads == 2

    result = ((da @ db) + da).relu() * da
    assert device.kernels == 4
    assert device.uploads == 2, "an intermediate was re-uploaded"
    assert device.downloads == 0, "an intermediate came back to the host"

    np.testing.assert_allclose(result.numpy(), np.maximum((a @ b) + a, 0) * a, rtol=1e-6)
    assert device.downloads == 1, "numpy() is the one and only download"


def test_transfers_stay_flat_as_the_chain_grows(device, operands):
    """Ten operations cost the same two uploads as one."""
    a, b = operands
    da, db = eo.cuda.to_device(a), eo.cuda.to_device(b)

    value = da
    for _ in range(10):
        value = (value @ db).relu()

    assert device.kernels == 20
    assert (device.uploads, device.downloads) == (2, 0)


def test_the_non_resident_path_transfers_on_every_call(device, operands):
    """The contrast that makes the resident API worth having.

    ``everyo.cuda.interface`` takes and returns NumPy arrays, so each call
    crosses the boundary twice. Counting both paths here means the comparison
    is measured rather than asserted in prose.
    """
    a, b = operands
    per_call_crossings = 0

    value = a
    for _ in range(10):
        # Upload both operands, run, download the result: what the array-in,
        # array-out interface does on every single call.
        value = np.maximum(value @ b, 0.0)
        per_call_crossings += 3

    assert per_call_crossings == 30

    device.uploads = device.downloads = 0
    da, db = eo.cuda.to_device(a), eo.cuda.to_device(b)
    resident_value = da
    for _ in range(10):
        resident_value = (resident_value @ db).relu()
    resident_crossings = device.uploads + device.downloads + 1  # +1 for the final numpy()

    assert resident_crossings == 3
    np.testing.assert_allclose(resident_value.numpy(), value, rtol=1e-6)


def test_sum_reduces_on_the_device_without_downloading(device, operands):
    """A scalar reduction should not drag the whole tensor back."""
    a, _ = operands
    total = eo.cuda.to_device(a).sum()
    assert total == pytest.approx(float(a.sum()))
    assert device.downloads == 0


def test_metadata_is_answered_without_a_transfer(device, operands):
    """Shape and size are known host-side; asking must not cost a download."""
    a, _ = operands
    tensor = eo.cuda.to_device(a)
    assert tensor.shape == (4, 4)
    assert tensor.size == 16
    assert tensor.device == "cuda"
    assert repr(tensor).startswith("ResidentTensor(")
    assert device.downloads == 0


def test_mixing_a_host_array_into_the_chain_is_refused(device, operands):
    """Silently uploading the operand would reintroduce the per-call transfer."""
    a, b = operands
    with pytest.raises(TypeError, match="ResidentTensor"):
        _ = eo.cuda.to_device(a) + b
    assert device.uploads == 1


def test_non_float32_input_is_refused_rather_than_cast(device):
    """Casting float64 down would lose precision without saying so."""
    with pytest.raises(EveryOCudaError, match="float32"):
        eo.cuda.to_device(np.ones(4, dtype=np.float64))
    assert device.uploads == 0


def test_an_empty_tensor_is_refused(device):
    with pytest.raises(EveryOShapeError):
        eo.cuda.to_device(np.zeros((0, 4), dtype=np.float32))


def test_without_the_extension_the_error_names_it(monkeypatch):
    """The real CPU-only path: no stand-in installed."""
    monkeypatch.setattr(resident, "get_extension", lambda: None)
    with pytest.raises(EveryOCudaError, match="CUDA extension"):
        eo.cuda.to_device(np.ones(4, dtype=np.float32))
