"""Tests for tensor construction, properties and conversion."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.exceptions import EveryODTypeError, EveryOGradientError, EveryOShapeError


class TestConstruction:
    def test_from_nested_lists(self):
        tensor = eo.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert tensor.shape == (2, 2)
        assert tensor.ndim == 2
        assert tensor.size == 4
        np.testing.assert_allclose(tensor.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    def test_python_floats_default_to_float32(self):
        assert eo.tensor([1.0, 2.0]).dtype == "float32"

    def test_numpy_arrays_keep_their_precision(self):
        assert eo.tensor(np.array([1.0, 2.0], dtype=np.float64)).dtype == "float64"

    def test_integers_become_float32_by_default(self):
        assert eo.tensor([1, 2, 3]).dtype == "float32"

    def test_explicit_dtype_is_respected(self):
        assert eo.tensor([1, 2, 3], dtype="int64").dtype == "int64"

    def test_scalar_tensor(self):
        tensor = eo.tensor(3.5)
        assert tensor.shape == ()
        assert tensor.item() == pytest.approx(3.5)

    def test_ragged_input_is_rejected(self):
        with pytest.raises(EveryODTypeError):
            eo.tensor([[1.0, 2.0], [3.0]])

    def test_unknown_dtype_is_rejected(self):
        with pytest.raises(EveryODTypeError):
            eo.tensor([1.0], dtype="float16")


class TestProperties:
    def test_default_device_is_cpu(self):
        assert eo.tensor([1.0]).device.type == "cpu"

    def test_dtype_name_is_canonical(self):
        assert eo.tensor([1.0], dtype="float64").dtype == "float64"

    def test_is_leaf(self):
        leaf = eo.tensor([1.0], requires_grad=True)
        assert leaf.is_leaf
        assert not (leaf * 2).is_leaf

    def test_transpose_property(self):
        tensor = eo.tensor([[1.0, 2.0, 3.0]])
        assert tensor.T.shape == (3, 1)

    def test_repr_mentions_shape_and_dtype(self):
        text = repr(eo.tensor([1.0, 2.0]))
        assert "shape=(2,)" in text
        assert "dtype=float32" in text

    def test_repr_truncates_large_tensors(self):
        assert "elements>" in repr(eo.zeros(100, 100))


class TestConversion:
    def test_numpy_returns_a_copy(self):
        tensor = eo.tensor([1.0, 2.0])
        array = tensor.numpy()
        array[0] = 99.0
        assert tensor.numpy()[0] == pytest.approx(1.0)

    def test_tolist(self):
        assert eo.tensor([[1.0, 2.0]]).tolist() == [[1.0, 2.0]]

    def test_item_requires_single_element(self):
        with pytest.raises(ValueError, match="exactly one element"):
            eo.tensor([1.0, 2.0]).item()

    def test_astype(self):
        assert eo.tensor([1.5]).astype("int64").numpy()[0] == 1

    def test_numpy_interoperability(self):
        assert np.asarray(eo.tensor([1.0, 2.0])).shape == (2,)

    def test_detach_removes_graph(self):
        source = eo.tensor([2.0], requires_grad=True)
        detached = (source * 3).detach()
        assert not detached.requires_grad
        assert detached.is_leaf

    def test_clone_is_independent(self):
        original = eo.tensor([1.0])
        copy = original.clone()
        copy.data = np.array([5.0], dtype=np.float32)
        assert original.item() == pytest.approx(1.0)


class TestIndexingAndIteration:
    def test_getitem_returns_tensor(self):
        tensor = eo.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert tensor[0].tolist() == [1.0, 2.0]

    def test_slicing(self):
        assert eo.tensor([1.0, 2.0, 3.0, 4.0])[1:3].tolist() == [2.0, 3.0]

    def test_len_and_iteration(self):
        tensor = eo.tensor([[1.0], [2.0], [3.0]])
        assert len(tensor) == 3
        assert [row.item() for row in tensor] == [1.0, 2.0, 3.0]

    def test_len_of_scalar_is_an_error(self):
        with pytest.raises(TypeError):
            len(eo.tensor(1.0))

    def test_bool_of_multi_element_tensor_is_an_error(self):
        with pytest.raises(ValueError, match="ambiguous"):
            bool(eo.tensor([1.0, 2.0]))

    def test_invalid_index_raises_shape_error(self):
        with pytest.raises(EveryOShapeError):
            eo.tensor([1.0, 2.0])[10]


class TestGradientFlag:
    def test_integer_tensors_cannot_require_grad(self):
        with pytest.raises(EveryOGradientError, match="floating point"):
            eo.tensor([1, 2], dtype="int64", requires_grad=True)

    def test_requires_grad_can_be_enabled_later(self):
        tensor = eo.tensor([1.0])
        tensor.requires_grad = True
        assert tensor.requires_grad
