"""Single-machine data-parallel training.

Data parallelism is the simplest way to use more than one core: give every
worker a copy of the model and a different slice of the batch, then average the
gradients before each optimizer step. Averaging makes the result mathematically
identical to one large batch on one worker -- the same sum of per-sample
gradients, divided by the same number of samples -- so the only thing that
changes is how long it takes.

.. warning::
   **Scope: one machine, multiple processes.** There is no multi-node support
   here, no TCP rendezvous and no NCCL. Workers communicate through POSIX
   shared memory, which means they must share an address space's worth of
   hardware -- one box. Anything claiming otherwise would be untested.

Processes, not threads: NumPy releases the GIL for large array operations but
holds it for the graph bookkeeping around them, so threads would contend on
exactly the part of training that is pure Python. Each worker is a real OS
process with its own interpreter.

The pieces:

* :func:`spawn` launches ``world_size`` workers and collects what they return.
* :class:`ProcessGroup` is handed to each worker; it carries the rank and the
  collective operations.
* :func:`average_gradients` is the one call a training loop needs to add.
* :func:`shard_indices` splits a dataset so no sample is seen twice per epoch.

Example:
    A worker function must be importable by name when the "spawn" start method
    is used, so define it at module level::

        import everyo as eo
        from everyo.distributed import average_gradients, shard_indices, spawn

        def worker(group, inputs, targets):
            model = eo.Sequential(eo.Linear(4, 1, seed=0))
            optimizer = eo.SGD(model.parameters(), lr=0.1)
            rows = shard_indices(len(inputs), group.rank, group.world_size)
            for _ in range(100):
                optimizer.zero_grad()
                residual = model(eo.tensor(inputs[rows])) - eo.tensor(targets[rows])
                eo.mean(residual * residual).backward()
                average_gradients(model.parameters(), group)
                optimizer.step()
            return [p.data for p in model.parameters()]

        results = spawn(worker, world_size=2, args=(inputs, targets))

    Every rank returns identical parameters, because every rank applied the
    same averaged gradient to the same initial weights.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
from multiprocessing import shared_memory
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from everyo._logging import get_logger
from everyo.exceptions import EveryOError

__all__ = [
    "DistributedError",
    "ProcessGroup",
    "all_reduce_mean",
    "average_gradients",
    "available_workers",
    "shard_indices",
    "spawn",
]

_LOGGER = get_logger(__name__)

#: dtype the collective buffer uses. float64 so that summing float32 gradients
#: across workers does not lose a bit to the reduction itself.
_BUFFER_DTYPE = np.float64
_NAME_FIELD = 128


class DistributedError(EveryOError):
    """Raised when a collective operation or a worker process fails."""


#: Environment variables that cap the thread pool inside each BLAS backend.
_THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _warn_about_thread_oversubscription(world_size: int) -> None:
    """Warn when each worker will also start a full BLAS thread pool.

    This is not a theoretical concern. Measured on a 4-core container with the
    same model and data, 30 steps took 5.2s on one worker and 1.8s on four with
    ``OMP_NUM_THREADS=1``; with the variable unset, four workers took 11.0s --
    *slower* than one worker, because each of the four started four BLAS threads
    and sixteen threads then fought over four cores.

    The variables have to be set before NumPy loads its BLAS backend, which has
    already happened by the time this runs, so this warns rather than silently
    setting them and pretending that worked.
    """
    if world_size < 2 or any(os.environ.get(name) for name in _THREAD_LIMIT_VARS):
        return
    _LOGGER.warning(
        "Starting %d workers while NumPy's BLAS is free to use every core. Each "
        "worker will start its own thread pool and they will contend: set "
        "OMP_NUM_THREADS=1 in the environment *before* launching Python to get "
        "the speedup you are asking for.",
        world_size,
    )


def available_workers() -> int:
    """Return a sensible default ``world_size`` for this machine.

    This is the usable CPU count -- respecting a cgroup or affinity limit where
    the OS exposes one, because a container with two cores should not be told it
    has thirty-two.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - not Linux
        return max(1, os.cpu_count() or 1)


def shard_indices(count: int, rank: int, world_size: int, *, drop_last: bool = True) -> np.ndarray:
    """Return the row indices belonging to ``rank``.

    Args:
        count: Number of samples in the dataset.
        rank: This worker's rank.
        world_size: Total number of workers.
        drop_last: When ``True`` (the default) the tail that does not divide
            evenly is dropped, so every rank gets exactly the same number of
            samples. That matters: a rank with fewer samples produces a gradient
            of a different batch size, and averaging those weights the small
            shard too heavily. When ``False`` the remainder is spread one sample
            per rank instead, which is fine for evaluation but biases training.

    Returns:
        A 1-D array of indices into the dataset.

    Raises:
        DistributedError: If ``rank`` is outside ``range(world_size)``, or if
            ``drop_last`` would leave a rank with nothing to do.

    Example:
        >>> from everyo.distributed import shard_indices
        >>> shard_indices(10, 0, 3).tolist()
        [0, 1, 2]
        >>> shard_indices(10, 2, 3).tolist()
        [6, 7, 8]
        >>> shard_indices(10, 0, 3, drop_last=False).tolist()
        [0, 1, 2, 3]
        >>> shard_indices(10, 2, 3, drop_last=False).tolist()
        [7, 8, 9]
    """
    if world_size < 1:
        raise DistributedError(f"world_size must be at least 1, got {world_size}.")
    if not 0 <= rank < world_size:
        raise DistributedError(f"rank must be in range(0, {world_size}), got {rank}.")

    per_rank = count // world_size
    if drop_last:
        if per_rank == 0:
            raise DistributedError(
                f"{count} sample(s) cannot be split across {world_size} worker(s) "
                "with drop_last=True. Use fewer workers or more data."
            )
        return np.arange(rank * per_rank, (rank + 1) * per_rank)

    remainder = count % world_size
    start = rank * per_rank + min(rank, remainder)
    stop = start + per_rank + (1 if rank < remainder else 0)
    return np.arange(start, stop)


class ProcessGroup:
    """One worker's handle on the collective operations.

    Instances are created by :func:`spawn` and passed into the worker function;
    constructing one directly is only useful when driving the processes
    yourself.

    Attributes:
        rank: This worker's index, in ``range(world_size)``.
        world_size: Total number of workers.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        barrier: Any,
        name_slot: Any,
        ready: Any,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._barrier = barrier
        self._name_slot = name_slot
        self._ready = ready
        self._shm: shared_memory.SharedMemory | None = None
        self._view: np.ndarray | None = None

    @property
    def is_main(self) -> bool:
        """``True`` on rank 0, the rank that should log and write checkpoints."""
        return self.rank == 0

    def barrier(self) -> None:
        """Block until every worker in the group has reached this point."""
        self._barrier.wait()

    # ------------------------------------------------------------------
    # Buffer negotiation
    # ------------------------------------------------------------------
    def _attach(self, elements: int) -> np.ndarray:
        """Return the shared ``(world_size, elements)`` staging buffer.

        The buffer is created by rank 0 on the first collective and reused
        afterwards, because the segment's size has to be fixed at creation and
        every step of a training loop reduces the same parameters.
        """
        if self._view is not None:
            if self._view.shape[1] != elements:
                raise DistributedError(
                    f"This process group's collective buffer holds "
                    f"{self._view.shape[1]} element(s) per rank, but {elements} "
                    "were requested. Reduce the same set of parameters on every "
                    "step, and the same set on every rank."
                )
            return self._view

        nbytes = self.world_size * elements * np.dtype(_BUFFER_DTYPE).itemsize
        if self.is_main:
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
            encoded = self._shm.name.encode("utf-8")
            self._name_slot[: len(encoded)] = encoded
            self._name_slot[len(encoded)] = 0
            self._ready.set()
        else:
            if not self._ready.wait(timeout=60.0):
                raise DistributedError(
                    "Timed out waiting for rank 0 to create the collective "
                    "buffer. A worker most likely crashed before its first "
                    "collective."
                )
            raw = bytes(self._name_slot[:])
            self._shm = shared_memory.SharedMemory(name=raw.split(b"\x00", 1)[0].decode("utf-8"))
            # Attaching registers the segment with this process's resource
            # tracker, which then warns about a "leak" when rank 0 -- its only
            # real owner -- unlinks it (CPython issue gh-82300). Only the
            # creator should be tracked.
            with contextlib.suppress(Exception):  # pragma: no cover
                from multiprocessing import resource_tracker

                resource_tracker.unregister(self._shm._name, "shared_memory")

        self._view = np.ndarray(
            (self.world_size, elements), dtype=_BUFFER_DTYPE, buffer=self._shm.buf
        )
        return self._view

    def all_reduce_mean(self, arrays: Sequence[np.ndarray]) -> None:
        """Replace each array with its mean across every worker, in place.

        The arrays are flattened into one contiguous vector so a step costs two
        barriers regardless of how many parameters the model has.

        Raises:
            DistributedError: If the arrays differ in shape between ranks, which
                shows up as a mismatched element count.
        """
        if self.world_size == 1:
            return
        if not arrays:
            return

        sizes = [array.size for array in arrays]
        buffer = self._attach(sum(sizes))

        offset = 0
        for array, size in zip(arrays, sizes):
            buffer[self.rank, offset : offset + size] = array.reshape(-1)
            offset += size

        self.barrier()
        averaged = buffer.mean(axis=0)
        # The read has to finish everywhere before anyone overwrites their slot
        # on the next call, hence the second barrier.
        self.barrier()

        offset = 0
        for array, size in zip(arrays, sizes):
            array[...] = (
                averaged[offset : offset + size]
                .reshape(array.shape)
                .astype(array.dtype, copy=False)
            )
            offset += size

    def broadcast(self, arrays: Sequence[np.ndarray], src: int = 0) -> None:
        """Overwrite every worker's arrays with rank ``src``'s copy, in place.

        Useful for making sure all workers start from identical weights when
        they were not built from the same seed.
        """
        if self.world_size == 1 or not arrays:
            return
        scale = float(self.world_size) if self.rank == src else 0.0
        staged = [np.asarray(array, dtype=_BUFFER_DTYPE) * scale for array in arrays]
        self.all_reduce_mean(staged)
        for array, value in zip(arrays, staged):
            array[...] = value.astype(array.dtype, copy=False)

    def close(self) -> None:
        """Detach from the shared buffer; rank 0 also releases it."""
        self._view = None
        if self._shm is None:
            return
        self._shm.close()
        if self.is_main:
            with contextlib.suppress(FileNotFoundError):  # already reclaimed
                self._shm.unlink()
        self._shm = None

    def __repr__(self) -> str:
        return f"ProcessGroup(rank={self.rank}, world_size={self.world_size})"


def all_reduce_mean(arrays: Sequence[np.ndarray], group: ProcessGroup) -> None:
    """Function form of :meth:`ProcessGroup.all_reduce_mean`."""
    group.all_reduce_mean(arrays)


def average_gradients(parameters: Iterable[Any], group: ProcessGroup) -> None:
    """Average every parameter's gradient across the workers, in place.

    Call this after ``backward()`` and before ``optimizer.step()``. Parameters
    whose gradient is ``None`` -- untouched by this batch -- are skipped, which
    means every rank must skip the same ones; in practice that is automatic,
    since all ranks run the same model.

    Example:
        >>> # in a worker function:
        >>> # loss.backward()
        >>> # average_gradients(model.parameters(), group)
        >>> # optimizer.step()
    """
    gradients = [
        parameter.grad for parameter in parameters if getattr(parameter, "grad", None) is not None
    ]
    group.all_reduce_mean(gradients)


def _bootstrap(
    function: Callable[..., Any],
    rank: int,
    world_size: int,
    barrier: Any,
    name_slot: Any,
    ready: Any,
    results: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Entry point executed inside each worker process."""
    group = ProcessGroup(rank, world_size, barrier, name_slot, ready)
    try:
        results.put((rank, "ok", function(group, *args, **kwargs)))
    except BaseException as error:  # noqa: BLE001 - reported to the parent
        results.put((rank, "error", f"{type(error).__name__}: {error}"))
        raise
    finally:
        # Every rank must reach here before rank 0 unlinks the segment.
        with contextlib.suppress(Exception):  # a peer may already have died
            barrier.wait(timeout=30.0)
        group.close()


def spawn(
    function: Callable[..., Any],
    world_size: int,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    start_method: str | None = None,
    timeout: float = 600.0,
) -> list[Any]:
    """Run ``function`` on ``world_size`` worker processes and collect results.

    Args:
        function: Called as ``function(group, *args, **kwargs)`` in each worker,
            where ``group`` is that worker's :class:`ProcessGroup`. Its return
            value is sent back to the parent, so it must be picklable.
        world_size: Number of worker processes. ``1`` still goes through a
            process, so a distributed script behaves the same way on one core.
        args: Positional arguments passed to every worker after ``group``.
        kwargs: Keyword arguments passed to every worker.
        start_method: ``multiprocessing`` start method. The default is ``"fork"``
            where the platform offers it and ``"spawn"`` otherwise. Under
            ``"spawn"`` the worker function must be importable at module level.
        timeout: Seconds to wait for all workers before giving up.

    Returns:
        A list of the workers' return values, ordered by rank.

    Raises:
        DistributedError: If ``world_size`` is not positive, a worker raises, or
            the workers do not finish within ``timeout``.

    .. important::
       Set ``OMP_NUM_THREADS=1`` in the environment before launching Python.
       NumPy's BLAS starts a thread pool per process, so without it every worker
       competes with every other worker for the same cores. Measured on a
       4-core container: 5.2s for one worker and 1.8s for four with the variable
       set -- and 11.0s for four with it unset, slower than not parallelising at
       all. :func:`spawn` warns when it looks unset.
    """
    if world_size < 1:
        raise DistributedError(f"world_size must be at least 1, got {world_size}.")

    if start_method is None:
        # "fork" is cheapest and lets workers close over anything, but on macOS
        # forking a process that has already started threads is documented as
        # unsafe and does deadlock in practice, so only Linux gets it.
        methods = mp.get_all_start_methods()
        use_fork = sys.platform.startswith("linux") and "fork" in methods
        start_method = "fork" if use_fork else "spawn"
    context = mp.get_context(start_method)

    barrier = context.Barrier(world_size)
    name_slot = context.Array("c", _NAME_FIELD, lock=False)
    ready = context.Event()
    results: Any = context.Queue()

    processes = [
        context.Process(
            target=_bootstrap,
            args=(
                function,
                rank,
                world_size,
                barrier,
                name_slot,
                ready,
                results,
                args,
                kwargs or {},
            ),
            daemon=False,
        )
        for rank in range(world_size)
    ]

    _warn_about_thread_oversubscription(world_size)
    _LOGGER.info("Launching %d worker(s) with the '%s' start method.", world_size, start_method)
    for process in processes:
        process.start()

    collected: dict[int, Any] = {}
    failures: list[str] = []
    try:
        for _ in range(world_size):
            rank, status, payload = results.get(timeout=timeout)
            if status == "ok":
                collected[rank] = payload
            else:
                failures.append(f"rank {rank}: {payload}")
    except Exception as error:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise DistributedError(
            f"Timed out after {timeout:g}s waiting for {world_size} worker(s). "
            "A worker may have deadlocked on a collective -- every rank must "
            "call the same collectives the same number of times."
        ) from error
    finally:
        for process in processes:
            process.join(timeout=timeout)
            if process.is_alive():  # pragma: no cover - defensive
                process.terminate()
                process.join()

    if failures:
        raise DistributedError("Distributed training failed:\n  " + "\n  ".join(failures))
    return [collected[rank] for rank in range(world_size)]
