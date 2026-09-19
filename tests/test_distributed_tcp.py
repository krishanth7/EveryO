import multiprocessing
import threading

import numpy as np
import pytest

from everyo.distributed import DistributedError
from everyo.distributed_tcp import TCPRendezvousServer, init_tcp_process_group


def test_tcp_all_reduce_across_independent_clients():
    server = TCPRendezvousServer("127.0.0.1", 0, 2).start()
    outputs = [None, None]

    def worker(rank):
        group = init_tcp_process_group("127.0.0.1", server.port, rank=rank, world_size=2)
        value = np.array([rank + 1.0, (rank + 1.0) * 2], dtype=np.float32)
        group.all_reduce_mean([value])
        outputs[rank] = value
        group.close()

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    server.close()
    np.testing.assert_allclose(outputs[0], [1.5, 3.0])
    np.testing.assert_allclose(outputs[1], [1.5, 3.0])


def test_invalid_rank_is_rejected_before_network_access():
    with pytest.raises(DistributedError, match="rank"):
        init_tcp_process_group("127.0.0.1", 1, rank=2, world_size=2)


def _worker_process(host, port, rank, world_size, queue):
    """Run one rank in its own interpreter and report what it computed.

    Module-level rather than nested, because the 'spawn' start method pickles
    the target by reference and cannot reach a closure.
    """
    import numpy as np

    from everyo.distributed_tcp import init_tcp_process_group

    try:
        group = init_tcp_process_group(host, port, rank=rank, world_size=world_size)
        value = np.array([rank + 1.0, (rank + 1.0) * 2], dtype=np.float32)
        group.all_reduce_mean([value])
        group.close()
        queue.put((rank, value.tolist()))
    except Exception as error:  # noqa: BLE001 - surfaced through the queue
        queue.put((rank, f"{type(error).__name__}: {error}"))


def test_tcp_all_reduce_across_separate_processes():
    """The collective has to work between genuinely independent interpreters.

    The thread-based test above shares one process, so it cannot distinguish a
    working wire protocol from ranks that happen to see the same memory. Spawned
    processes share no objects at all: everything crossing between them goes
    over the socket, serialised and back. That is the part that has to hold for
    ranks on different machines.

    Still one host. No second machine is available here, so this establishes the
    protocol across process boundaries, not across a physical network.
    """
    context = multiprocessing.get_context("spawn")
    server = TCPRendezvousServer("127.0.0.1", 0, 2).start()
    queue = context.Queue()

    processes = [
        context.Process(target=_worker_process, args=("127.0.0.1", server.port, rank, 2, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    results = {}
    try:
        for _ in processes:
            rank, value = queue.get(timeout=60)
            results[rank] = value
    finally:
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():  # pragma: no cover - only on a hang
                process.terminate()
        server.close()

    assert [process.exitcode for process in processes] == [0, 0]
    for rank in (0, 1):
        assert isinstance(results[rank], list), f"rank {rank} failed: {results[rank]}"
        np.testing.assert_allclose(results[rank], [1.5, 3.0])
