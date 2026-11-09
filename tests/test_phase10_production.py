"""Phase 10 CPU reliability acceptance suite."""

import numpy as np
import pytest

from everyo import Adam, Linear, tensor
from everyo.benchmarking_phase10 import benchmark_cpu
from everyo.checkpointing import load_training_checkpoint, save_training_checkpoint
from everyo.evaluation import (
    assert_no_leakage,
    classification_metrics,
    deterministic_split,
    language_model_metrics,
)
from everyo.experiment_tracking import ExperimentTracker
from everyo.inference import CPUInferenceRuntime
from everyo.validation import validate_array


def test_metrics_and_invalid_data():
    report = classification_metrics([[10, 0], [0, 10]], np.array([0, 1]))
    assert report["accuracy"] == 1.0 and report["cross_entropy"] < 0.001
    lm = language_model_metrics(np.array([[[10, 0], [0, 10]]]), np.array([[0, 1]]))
    assert lm["token_accuracy"] == 1 and lm["perplexity"] >= 1
    with pytest.raises(ValueError):
        classification_metrics([[np.nan, 1]], [0])
    with pytest.raises(ValueError):
        language_model_metrics(
            np.ones((1, 2, 3)), np.zeros((1, 2), int), mask=np.zeros((1, 2), bool)
        )


def test_splits_and_leakage():
    splits = deterministic_split(40, seed=23)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(splits.values(), deterministic_split(40, seed=23).values())
    )
    assert_no_leakage(splits)
    splits["test"][0] = splits["train"][0]
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(splits)
    with pytest.raises(ValueError):
        deterministic_split(3, train=0.7, validation=0.15)


def test_benchmark_and_tracker(tmp_path):
    result = benchmark_cpu(lambda: np.eye(4) @ np.eye(4), repeats=2, warmup=0)
    assert result["mean_ms"] >= 0 and result["repeats"] == 2
    tracker = ExperimentTracker(tmp_path / "runs.jsonl", run_id="test")
    tracker.log(0, {"loss": 1.2}, config={"seed": 3})
    assert tracker.read()[0]["metrics"]["loss"] == 1.2
    with pytest.raises(ValueError):
        tracker.log(1, {"bad": float("nan")})


def test_inference_limits():
    runtime = CPUInferenceRuntime(Linear(2, 2, seed=1), max_batch=2)
    inputs = np.ones((5, 2), dtype=float)
    assert runtime.predict_batched(inputs).shape == (5, 2)
    with pytest.raises(ValueError):
        runtime.predict(inputs)
    with pytest.raises(ValueError):
        validate_array([float("inf")])


def test_checkpoint_optimizer_rng_roundtrip(tmp_path):
    model = Linear(2, 1, seed=9)
    opt = Adam(model.parameters(), lr=0.01)
    rng = np.random.default_rng(4)

    def step():
        opt.zero_grad()
        loss = (model(tensor([[1.0, 2.0]])) ** 2).sum()
        loss.backward()
        opt.step()

    step()
    path = tmp_path / "training.evo"
    save_training_checkpoint(model, opt, path, rng=rng, tokenizer={"version": 1})
    expected = rng.random()
    step()
    reference = [p.data.copy() for p in model.parameters()]
    restored_model = Linear(2, 1, seed=9)
    restored_opt = Adam(restored_model.parameters(), lr=0.01)
    restored_rng = np.random.default_rng(22)
    got, tokens = load_training_checkpoint(
        path, restored_opt, model=restored_model, rng=restored_rng
    )
    assert got is restored_model and tokens == {"version": 1}
    assert restored_rng.random() == expected
    restored_opt.zero_grad()
    loss = (restored_model(tensor([[1.0, 2.0]])) ** 2).sum()
    loss.backward()
    restored_opt.step()
    for actual, desired in zip(restored_model.parameters(), reference):
        np.testing.assert_allclose(actual.data, desired, atol=1e-12)


def test_checkpoint_rejects_mismatched_optimizer(tmp_path):
    model = Linear(2, 1, seed=0)
    opt = Adam(model.parameters(), lr=0.01)
    path = tmp_path / "model.evo"
    save_training_checkpoint(model, opt, path)
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(path, Adam(model.parameters(), lr=0.02), model=model)
