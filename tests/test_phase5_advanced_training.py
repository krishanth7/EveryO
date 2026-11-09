"""Phase 5 acceptance: reproducible training, optimizer fidelity and safety."""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo


def test_adamw_decoupled_decay_and_moments():
    p = eo.Parameter(np.array([2.0], dtype=np.float64))
    optim = eo.AdamW([p], lr=0.1, weight_decay=0.1)
    p.grad = np.array([0.0])
    optim.step()
    np.testing.assert_allclose(p.numpy(), [1.98], rtol=1e-12)
    p.grad = np.array([1.0])
    optim.step()
    assert p.numpy()[0] < 1.98
    assert "m" in optim.state_for(p)
    with pytest.raises(ValueError):
        eo.AdamW([p], betas=(1.0, 0.9))


@pytest.mark.parametrize("centered", [False, True])
def test_rmsprop_converges_and_validates(centered):
    p = eo.Parameter(np.array([3.0], dtype=np.float64))
    optim = eo.RMSprop([p], lr=0.025, centered=centered, momentum=0.1)
    for _ in range(60):
        p.grad = 2 * p.data
        optim.step()
    assert abs(p.data[0]) < 2
    with pytest.raises(ValueError):
        eo.RMSprop([p], alpha=1)


def test_schedulers_search_and_pruning():
    p = eo.Parameter(np.array([1.0]))
    optim = eo.SGD([p], lr=0.1)
    scheduler = eo.WarmupCosineScheduler(optim, total_steps=6, warmup_steps=2)
    assert scheduler.step() == pytest.approx(0.05)
    assert scheduler.step() == pytest.approx(0.1)
    scheduler.step()
    state = scheduler.state_dict()
    fresh_optim = eo.SGD([eo.Parameter(np.array([1.0]))], lr=0.1)
    scheduler_copy = eo.WarmupCosineScheduler(fresh_optim, total_steps=6, warmup_steps=2)
    scheduler_copy.load_state_dict(state)
    assert scheduler_copy.state_dict() == state
    for _ in range(3):
        scheduler.step()
    assert optim.lr == pytest.approx(0)
    optim.lr = 0.1
    plateau = eo.ReduceOnPlateauScheduler(optim, patience=1)
    plateau.step(0.5)
    plateau.step(0.5)
    plateau.step(0.5)
    assert optim.lr == pytest.approx(0.05)
    space = {"beta": [1, 2], "alpha": ["a", "b"]}
    assert len(list(eo.grid_search(space))) == 4
    assert list(eo.random_search(space, trials=8, seed=41)) == list(
        eo.random_search(space, trials=8, seed=41)
    )
    pruner = eo.MedianPruner(min_trials=2, warmup_steps=2)
    assert not pruner.should_prune(step=1, value=10, completed_peers=[{1: 1}, {1: 2}])
    assert pruner.should_prune(step=2, value=10, completed_peers=[{2: 1}, {2: 2}])


def test_anomaly_detector_and_clip():
    p = eo.Parameter(np.array([2.0]))
    optim = eo.SGD([p], lr=0.1)
    p.grad = np.array([20.0])
    assert eo.clip_gradients(optim, 2.0) == pytest.approx(20)
    assert p.grad[0] == pytest.approx(2)
    assert eo.TrainingAnomalyDetector().check(optim) == pytest.approx(2)
    with pytest.raises(FloatingPointError):
        eo.TrainingAnomalyDetector(max_grad_norm=1).check(optim)
    p.grad = np.array([np.nan])
    with pytest.raises(FloatingPointError):
        eo.clip_gradients(optim, 2)


@pytest.mark.parametrize("kind", ["AdamW", "RMSprop"])
def test_checkpoint_exact_next_step_and_rejection(tmp_path, kind):
    def new():
        model = eo.Sequential(eo.Linear(3, 2, seed=5))
        optimizer = getattr(eo, kind)(model.parameters(), lr=0.01)
        return model, optimizer

    def one_step(model, optimizer):
        x = eo.tensor(np.array([[1.0, 2.0, -1.0]], dtype=np.float32))
        optimizer.zero_grad()
        loss = (model(x) ** 2).sum()
        loss.backward()
        optimizer.step()

    first, optim1 = new()
    for _ in range(4):
        one_step(first, optim1)
    path = tmp_path / "checkpoint.npz"
    eo.save_training_checkpoint(first, optim1, path, epoch=4, extras={"seed": 5})
    restored, optim2 = new()
    checkpoint = eo.load_training_checkpoint(restored, optim2, path)
    assert checkpoint == {"epoch": 4, "extras": {"seed": 5}}
    one_step(first, optim1)
    one_step(restored, optim2)
    for key, expected in first.state_dict().items():
        np.testing.assert_array_equal(restored.state_dict()[key], expected)
    assert optim1.step_count == optim2.step_count
    mismatched, bad_optimizer = new()
    bad_optimizer = eo.SGD(mismatched.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="optimizer type"):
        eo.load_training_checkpoint(mismatched, bad_optimizer, path)
    if kind == "AdamW":
        incompatible, _ = new()
        differently_configured = eo.AdamW(incompatible.parameters(), weight_decay=0.5)
        with pytest.raises(ValueError, match="hyperparameter"):
            eo.load_training_checkpoint(incompatible, differently_configured, path)


def test_gradient_accumulation_equivalent_large_batch():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(12, 3)).astype(np.float32)
    y = rng.normal(size=(12, 1)).astype(np.float32)

    def train(batch_size, accumulate):
        model = eo.Sequential(eo.Linear(3, 1, seed=19))
        optimizer = eo.SGD(model.parameters(), lr=0.03)
        trainer = eo.Trainer(
            model,
            optimizer,
            eo.MSELoss(),
            accumulation_steps=accumulate,
            detect_anomalies=True,
            gradient_clip=20,
        )
        loader = eo.DataLoader(eo.ArrayDataset(x, y), batch_size=batch_size, shuffle=False)
        trainer.fit(loader, epochs=2, verbose=False)
        return model.state_dict(), optimizer.step_count

    large, big_steps = train(6, 1)
    micro, small_steps = train(2, 3)
    assert big_steps == small_steps == 4
    for name in large:
        np.testing.assert_allclose(micro[name], large[name], rtol=2e-5, atol=2e-6)
