"""Tests for single-machine data-parallel training.

The worker functions live at module level on purpose: under the "spawn" start
method a worker has to be importable by name, and a test that only passed under
"fork" would be a test that only passed on Linux.

The central claim these tests make is the one that matters -- that splitting a
batch across workers and averaging the gradients gives the same answer as
training on the whole batch in one process -- and it is checked by running both
and comparing weights, not by inspecting the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

import everyo as eo
from everyo.distributed import (
    DistributedError,
    available_workers,
    average_gradients,
    shard_indices,
    spawn,
)

STEPS = 40
SAMPLES = 64
TRUE_WEIGHTS = np.array([[1.0], [-2.0], [0.5], [0.25]], dtype=np.float32)


def _dataset():
    inputs = np.random.default_rng(0).normal(size=(SAMPLES, 4)).astype(np.float32)
    return inputs, (inputs @ TRUE_WEIGHTS).astype(np.float32)


def _model():
    return eo.Sequential(eo.Linear(4, 8, seed=0), eo.Tanh(), eo.Linear(8, 1, seed=1))


def _train_locally(inputs, targets, steps=STEPS):
    model = _model()
    optimizer = eo.SGD(model.parameters(), lr=0.1)
    xs, ys = eo.tensor(inputs), eo.tensor(targets)
    for _ in range(steps):
        optimizer.zero_grad()
        residual = model(xs) - ys
        eo.mean(residual * residual).backward()
        optimizer.step()
    return [np.asarray(p.data) for p in model.parameters()]


# --------------------------------------------------------------------------
# Worker functions. Module level so "spawn" can import them.
# --------------------------------------------------------------------------
def train_worker(group, inputs, targets):
    model = _model()
    optimizer = eo.SGD(model.parameters(), lr=0.1)
    rows = shard_indices(len(inputs), group.rank, group.world_size)
    xs, ys = eo.tensor(inputs[rows]), eo.tensor(targets[rows])
    for _ in range(STEPS):
        optimizer.zero_grad()
        residual = model(xs) - ys
        eo.mean(residual * residual).backward()
        average_gradients(model.parameters(), group)
        optimizer.step()
    return [np.asarray(p.data) for p in model.parameters()]


def rank_report_worker(group):
    return (group.rank, group.world_size, group.is_main)


def all_reduce_worker(group):
    # Each rank contributes its own rank number; the mean of 0..n-1 is known.
    values = [np.full((3,), float(group.rank)), np.full((2, 2), float(group.rank) * 10.0)]
    group.all_reduce_mean(values)
    return [value.tolist() for value in values]


def repeated_reduce_worker(group):
    # Diverge, reduce, diverge again -- five rounds. Each round starts from
    # values that differ per rank, so a stale slot would show up immediately.
    value = np.zeros(1)
    for _ in range(5):
        value += float(group.rank)
        group.all_reduce_mean([value])
    return value.tolist()


def dtype_preserving_worker(group):
    value = np.full((4,), float(group.rank), dtype=np.float32)
    group.all_reduce_mean([value])
    return (str(value.dtype), value.tolist())


def broadcast_worker(group):
    value = np.full((3,), float(group.rank + 1), dtype=np.float32)
    group.broadcast([value], src=0)
    return value.tolist()


def mismatched_size_worker(group):
    group.all_reduce_mean([np.zeros(4)])
    # A different element count on the second call must be refused, not
    # silently truncated into the buffer rank 0 already sized.
    group.all_reduce_mean([np.zeros(8)])
    return "unreachable"


def failing_worker(group):
    if group.rank == 0:
        raise ValueError("deliberate failure")
    # The other ranks must not hang waiting for rank 0 at a collective.
    return "fine"


def barrier_worker(group):
    group.barrier()
    return group.rank


# --------------------------------------------------------------------------
class TestSharding:
    def test_even_split(self):
        assert shard_indices(12, 0, 3).tolist() == [0, 1, 2, 3]
        assert shard_indices(12, 2, 3).tolist() == [8, 9, 10, 11]

    def test_drop_last_gives_every_rank_the_same_count(self):
        counts = {len(shard_indices(10, rank, 3)) for rank in range(3)}
        assert counts == {3}

    def test_shards_are_disjoint_and_cover_the_kept_rows(self):
        shards = [shard_indices(10, rank, 3, drop_last=False) for rank in range(3)]
        combined = np.concatenate(shards)
        assert sorted(combined.tolist()) == list(range(10))

    def test_remainder_is_spread_when_not_dropping(self):
        sizes = [len(shard_indices(10, rank, 3, drop_last=False)) for rank in range(3)]
        assert sizes == [4, 3, 3]

    def test_single_worker_gets_everything(self):
        assert shard_indices(7, 0, 1).tolist() == list(range(7))

    def test_rank_out_of_range(self):
        with pytest.raises(DistributedError, match="rank must be"):
            shard_indices(10, 3, 3)

    def test_too_few_samples(self):
        with pytest.raises(DistributedError, match="cannot be split"):
            shard_indices(2, 0, 4)

    def test_available_workers_is_positive(self):
        assert available_workers() >= 1


class TestSpawnContract:
    def test_world_size_must_be_positive(self):
        with pytest.raises(DistributedError, match="at least 1"):
            spawn(rank_report_worker, 0)

    def test_results_come_back_ordered_by_rank(self):
        results = spawn(rank_report_worker, 3)
        assert results == [(0, 3, True), (1, 3, False), (2, 3, False)]

    def test_a_single_worker_still_runs_in_a_process(self):
        assert spawn(rank_report_worker, 1) == [(0, 1, True)]

    def test_a_failing_worker_is_reported_with_its_rank(self):
        with pytest.raises(DistributedError, match="rank 0: ValueError"):
            spawn(failing_worker, 2)

    def test_barrier_releases_every_rank(self):
        assert spawn(barrier_worker, 3) == [0, 1, 2]


class TestCollectives:
    def test_all_reduce_computes_the_mean(self):
        results = spawn(all_reduce_worker, 4)
        # mean of 0, 1, 2, 3
        for first, second in results:
            assert first == [1.5, 1.5, 1.5]
            assert second == [[15.0, 15.0], [15.0, 15.0]]

    def test_every_rank_sees_the_same_result(self):
        results = spawn(all_reduce_worker, 3)
        assert all(result == results[0] for result in results)

    def test_repeated_reductions_reuse_the_buffer(self):
        # Five reductions in a row: if the two barriers were wrong, ranks would
        # read a slot that a peer had already overwritten.
        results = spawn(repeated_reduce_worker, 3)
        assert all(result == results[0] for result in results)

    def test_dtype_is_preserved(self):
        for dtype, values in spawn(dtype_preserving_worker, 2):
            assert dtype == "float32"
            assert values == [0.5, 0.5, 0.5, 0.5]

    def test_single_worker_reduction_is_a_no_op(self):
        (values,) = spawn(all_reduce_worker, 1)
        assert values[0] == [0.0, 0.0, 0.0]

    def test_broadcast_copies_rank_zero(self):
        for values in spawn(broadcast_worker, 3):
            assert values == [1.0, 1.0, 1.0]

    def test_changing_the_reduction_size_is_refused(self):
        with pytest.raises(DistributedError, match="element"):
            spawn(mismatched_size_worker, 2)


class TestDataParallelTraining:
    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_matches_single_process_training(self, world_size):
        inputs, targets = _dataset()
        reference = _train_locally(inputs, targets)
        results = spawn(train_worker, world_size, args=(inputs, targets))

        for rank, parameters in enumerate(results):
            for index, (actual, expected) in enumerate(zip(parameters, reference)):
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=f"rank {rank}, parameter {index}",
                )

    def test_all_ranks_end_bit_identical(self):
        # Averaged gradients applied to identical initial weights must leave
        # every rank holding exactly the same model -- not merely a close one.
        inputs, targets = _dataset()
        results = spawn(train_worker, 4, args=(inputs, targets))
        for parameters in results[1:]:
            for actual, expected in zip(parameters, results[0]):
                np.testing.assert_array_equal(actual, expected)

    def test_training_actually_reduces_the_loss(self):
        inputs, targets = _dataset()
        parameters = spawn(train_worker, 2, args=(inputs, targets))[0]
        model = _model()
        for parameter, trained in zip(model.parameters(), parameters):
            parameter.data[...] = trained
        with eo.no_grad():
            residual = model(eo.tensor(inputs)) - eo.tensor(targets)
            trained_loss = float(eo.mean(residual * residual).item())
        untrained = _model()
        with eo.no_grad():
            residual = untrained(eo.tensor(inputs)) - eo.tensor(targets)
            initial_loss = float(eo.mean(residual * residual).item())
        assert trained_loss < initial_loss / 2
