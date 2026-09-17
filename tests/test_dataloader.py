"""Tests for datasets, loaders, preprocessing and splitting."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.data.preprocessing import MinMaxScaler, normalize, shuffle_arrays
from everyo.exceptions import EveryOShapeError


@pytest.fixture
def dataset():
    features = np.arange(40, dtype=np.float32).reshape(20, 2)
    targets = np.arange(20, dtype=np.int64)
    return eo.ArrayDataset(features, targets)


class TestArrayDataset:
    def test_length_and_item(self, dataset):
        assert len(dataset) == 20
        features, target = dataset[3]
        np.testing.assert_allclose(features, [6.0, 7.0])
        assert target == 3

    def test_features_only(self):
        dataset = eo.ArrayDataset(np.zeros((5, 3)))
        assert dataset[0].shape == (3,)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(EveryOShapeError, match="same number of samples"):
            eo.ArrayDataset(np.zeros((5, 2)), np.zeros(4))

    def test_accepts_tensors(self):
        assert len(eo.ArrayDataset(eo.zeros(6, 2), eo.zeros(6))) == 6

    def test_feature_shape(self, dataset):
        assert dataset.feature_shape == (2,)


class TestDataLoader:
    def test_batch_count_without_drop_last(self, dataset):
        assert len(eo.DataLoader(dataset, batch_size=6)) == 4

    def test_batch_count_with_drop_last(self, dataset):
        assert len(eo.DataLoader(dataset, batch_size=6, drop_last=True)) == 3

    def test_batch_shapes(self, dataset):
        batches = list(eo.DataLoader(dataset, batch_size=6))
        assert [batch[0].shape[0] for batch in batches] == [6, 6, 6, 2]

    def test_every_sample_is_visited_exactly_once(self, dataset):
        seen = np.concatenate(
            [batch[1] for batch in eo.DataLoader(dataset, batch_size=7, shuffle=True, seed=0)]
        )
        np.testing.assert_array_equal(np.sort(seen), np.arange(20))

    def test_shuffle_changes_the_order(self, dataset):
        ordered = np.concatenate([b[1] for b in eo.DataLoader(dataset, batch_size=5)])
        shuffled = np.concatenate(
            [b[1] for b in eo.DataLoader(dataset, batch_size=5, shuffle=True, seed=0)]
        )
        assert not np.array_equal(ordered, shuffled)

    def test_shuffle_is_reproducible(self, dataset):
        first = np.concatenate(
            [b[1] for b in eo.DataLoader(dataset, batch_size=5, shuffle=True, seed=42)]
        )
        second = np.concatenate(
            [b[1] for b in eo.DataLoader(dataset, batch_size=5, shuffle=True, seed=42)]
        )
        np.testing.assert_array_equal(first, second)

    def test_epochs_are_reshuffled(self, dataset):
        loader = eo.DataLoader(dataset, batch_size=5, shuffle=True, seed=0)
        first = np.concatenate([b[1] for b in loader])
        second = np.concatenate([b[1] for b in loader])
        assert not np.array_equal(first, second)

    def test_invalid_batch_size(self, dataset):
        with pytest.raises(ValueError, match="batch_size"):
            eo.DataLoader(dataset, batch_size=0)

    def test_rejects_non_dataset(self):
        with pytest.raises(TypeError, match="ArrayDataset"):
            eo.DataLoader(object(), batch_size=2)

    def test_works_with_a_custom_dataset(self):
        class Squares(eo.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, index):
                return np.array([index], dtype=np.float32), np.float32(index**2)

        batches = list(eo.DataLoader(Squares(), batch_size=4))
        assert len(batches) == 3
        assert batches[0][0].shape == (4, 1)

    def test_subset(self, dataset):
        subset = eo.Subset(dataset, [1, 3, 5])
        assert len(subset) == 3
        assert subset[0][1] == 1

    def test_subset_index_validation(self, dataset):
        with pytest.raises(IndexError):
            eo.Subset(dataset, [999])


class TestSplitting:
    def test_train_test_split_sizes(self):
        features = np.arange(100).reshape(100, 1)
        train, test = eo.train_test_split(features, test_size=0.25, seed=0)
        assert len(train) == 75 and len(test) == 25

    def test_split_keeps_arrays_aligned(self):
        features = np.arange(50).reshape(50, 1).astype(np.float32)
        targets = features.reshape(-1) * 2
        x_train, x_test, y_train, y_test = eo.train_test_split(
            features, targets, test_size=0.2, seed=0
        )
        np.testing.assert_allclose(y_train, x_train.reshape(-1) * 2)
        np.testing.assert_allclose(y_test, x_test.reshape(-1) * 2)

    def test_split_is_reproducible(self):
        features = np.arange(30).reshape(30, 1)
        first, _ = eo.train_test_split(features, test_size=0.2, seed=7)
        second, _ = eo.train_test_split(features, test_size=0.2, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_invalid_test_size(self):
        with pytest.raises(ValueError, match="test_size"):
            eo.train_test_split(np.zeros((10, 1)), test_size=1.5)

    def test_mismatched_arrays(self):
        with pytest.raises(EveryOShapeError):
            eo.train_test_split(np.zeros((10, 1)), np.zeros(5), test_size=0.2)

    def test_random_split_covers_the_dataset(self):
        dataset = eo.ArrayDataset(np.zeros((100, 2)), np.zeros(100))
        parts = eo.random_split(dataset, (0.7, 0.2, 0.1), seed=0)
        assert sum(len(part) for part in parts) == 100

    def test_stratified_split_preserves_class_balance(self):
        labels = np.array([0] * 60 + [1] * 40)
        features = np.zeros((100, 2), dtype=np.float32)
        _, _, y_train, y_test = eo.stratified_split(features, labels, test_size=0.2, seed=0)
        assert abs(float(np.mean(y_train)) - 0.4) < 0.05
        assert abs(float(np.mean(y_test)) - 0.4) < 0.1


class TestPreprocessing:
    def test_standard_scaler(self, rng):
        values = rng.normal(5.0, 3.0, size=(200, 4))
        scaled = eo.StandardScaler().fit_transform(values)
        np.testing.assert_allclose(scaled.mean(axis=0), np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(scaled.std(axis=0), np.ones(4), atol=1e-4)

    def test_scaler_uses_training_statistics(self, rng):
        train = rng.normal(0.0, 1.0, size=(100, 2))
        scaler = eo.StandardScaler().fit(train)
        transformed = scaler.transform(train + 10.0)
        assert transformed.mean() > 5.0

    def test_transform_before_fit_is_an_error(self):
        with pytest.raises(EveryOShapeError, match="before fit"):
            eo.StandardScaler().transform(np.zeros((2, 2)))

    def test_inverse_transform_round_trip(self, rng):
        values = rng.normal(size=(50, 3))
        scaler = eo.StandardScaler().fit(values)
        np.testing.assert_allclose(
            scaler.inverse_transform(scaler.transform(values)), values, rtol=1e-4
        )

    def test_min_max_scaler_bounds(self, rng):
        scaled = MinMaxScaler((0.0, 1.0)).fit_transform(rng.normal(size=(100, 3)))
        assert scaled.min() >= -1e-6 and scaled.max() <= 1 + 1e-6

    def test_min_max_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="increasing"):
            MinMaxScaler((1.0, 0.0))

    def test_normalize_gives_unit_rows(self, rng):
        normalized = normalize(rng.normal(size=(20, 5)), axis=1)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(20), rtol=1e-5)

    def test_one_hot_encode(self):
        np.testing.assert_allclose(eo.one_hot_encode([0, 1, 2], 3), np.eye(3, dtype=np.float32))

    def test_one_hot_rejects_out_of_range(self):
        with pytest.raises(EveryOShapeError, match="out of range"):
            eo.one_hot_encode([3], 3)

    def test_shuffle_arrays_stays_aligned(self):
        a = np.arange(20).reshape(20, 1)
        b = np.arange(20) * 3
        shuffled_a, shuffled_b = shuffle_arrays(a, b, seed=0)
        np.testing.assert_allclose(shuffled_a.reshape(-1) * 3, shuffled_b)

    def test_standardize_helper(self, rng):
        standardized = eo.standardize(rng.normal(3.0, 2.0, size=(100, 2)))
        np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-5)


class TestCSVLoading:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b,label\n1,2,0\n3,4,1\n5,6,0\n", encoding="utf-8")
        dataset = eo.load_csv_dataset(path, target_column="label")
        assert len(dataset) == 3
        np.testing.assert_allclose(dataset.features, [[1, 2], [3, 4], [5, 6]])
        np.testing.assert_allclose(dataset.targets, [0, 1, 0])

    def test_headerless_file(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("1,2,0\n3,4,1\n", encoding="utf-8")
        dataset = eo.load_csv_dataset(path, target_column=-1)
        assert dataset.features.shape == (2, 2)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            eo.load_csv_dataset(tmp_path / "nope.csv")

    def test_non_numeric_values_are_reported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,cat\n", encoding="utf-8")
        with pytest.raises(EveryOShapeError, match="numeric table"):
            eo.load_csv_dataset(path)
