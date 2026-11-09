"""Phase 4 CPU-only data and experiment engineering acceptance tests."""

import json

import numpy as np
import pytest

from everyo.data_engineering import DatasetCache, StreamingDataset, file_sha256
from everyo.experiments import ExperimentRegistry


def test_streaming_determinism_resume_and_batches():
    def source():
        return iter(range(35))

    dataset = StreamingDataset(
        source,
        seed=31,
        shuffle_buffer=7,
        transform=lambda item, rng: (item, float(rng.random())),
    )
    first = list(dataset.iter_epoch(2))
    assert first == list(dataset.iter_epoch(2))
    assert first[11:] == list(dataset.iter_epoch(2, skip=11))
    assert first != list(dataset.iter_epoch(3))
    assert sorted(item for item, _ in first) == list(range(35))
    assert sum(len(batch) for batch in dataset.batches(8)) == 35
    with pytest.raises(ValueError):
        list(dataset.iter_epoch(-1))
    with pytest.raises(ValueError):
        list(dataset.batches(0))
    with pytest.raises(ValueError):
        StreamingDataset(source, shuffle_buffer=0)


def test_streaming_empty_and_single_buffer():
    assert list(StreamingDataset(lambda: iter(())).iter_epoch()) == []
    assert list(StreamingDataset(lambda: iter((1, 2, 3))).iter_epoch()) == [1, 2, 3]


def test_verified_cache_and_tampering(tmp_path):
    cache = DatasetCache(tmp_path / "cache")
    matrix = np.arange(18, dtype=np.float32).reshape(3, 6)
    digest = cache.put("a/../b", matrix)
    assert len(digest) == 64
    np.testing.assert_array_equal(cache.get("a/../b"), matrix)
    assert isinstance(cache.get("a/../b", mmap=True), np.memmap)
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 1
    with pytest.raises(TypeError):
        cache.put("bad", np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        cache.put("", matrix)
    target = next((tmp_path / "cache").glob("*.npy"))
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        cache.get("a/../b")


def test_registry_resume_compare_artifacts(tmp_path):
    registry = ExperimentRegistry(tmp_path / "tracking")
    dataset_file = tmp_path / "source.bin"
    dataset_file.write_bytes(b"reproducible dataset")
    dataset_hash = file_sha256(dataset_file)
    a = registry.start({"lr": 0.01}, dataset_hash=dataset_hash, seed=42)
    b = registry.start({"lr": 0.02}, dataset_hash=dataset_hash, seed=42)
    registry.update_progress(a, epoch=3, skip=14)
    registry.log_metric(a, 1, "loss", 0.4)
    registry.log_metric(a, 2, "loss", 0.2)
    registry.log_metric(b, 1, "loss", 0.5)
    assert registry.get(a)["progress"] == {"epoch": 3, "skip": 14}
    assert registry.compare([a, b])[0]["latest_metrics"]["loss"] == 0.2
    artifact_hash = registry.add_artifact(a, "weights.evo", dataset_file)
    assert registry.artifact(a, "weights.evo").read_bytes() == dataset_file.read_bytes()
    manifest = registry.manifest(a)
    assert manifest["artifacts"]["weights.evo"] == artifact_hash
    assert json.loads(json.dumps(manifest))["configuration"] == {"lr": 0.01}
    assert registry.manifest(a) == manifest
    assert ExperimentRegistry(tmp_path / "tracking").get(a)["progress"]["skip"] == 14
    registry.finish(a)
    assert registry.get(a)["status"] == "completed"
    with pytest.raises(ValueError):
        registry.log_metric(a, 3, "loss", 0.1)
    with pytest.raises(ValueError):
        registry.update_progress(a, epoch=4, skip=0)
    with pytest.raises(ValueError):
        registry.finish(a)


def test_registry_validation_integrity(tmp_path):
    registry = ExperimentRegistry(tmp_path / "experiments")
    with pytest.raises(ValueError):
        registry.start({}, dataset_hash="not-a-digest")
    with pytest.raises(TypeError):
        registry.start([], dataset_hash="0" * 64)
    with pytest.raises(KeyError):
        registry.get("missing")
    run = registry.start({}, dataset_hash="0" * 64)
    with pytest.raises(ValueError):
        registry.log_metric(run, 0, "nan", float("nan"))
    with pytest.raises(ValueError):
        registry.update_progress(run, epoch=-1, skip=0)
    with pytest.raises(ValueError):
        registry.add_artifact(run, "../unsafe", tmp_path / "nothing")
    source = tmp_path / "model"
    source.write_bytes(b"model-state")
    registry.add_artifact(run, "model", source)
    registry.artifact(run, "model").write_bytes(b"bad-state")
    with pytest.raises(ValueError, match="checksum"):
        registry.artifact(run, "model")
