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
