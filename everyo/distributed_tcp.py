"""Multi-node CPU collectives over a trusted TCP network."""

from __future__ import annotations

import json
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from everyo.distributed import DistributedError

__all__ = ["TCPProcessGroup", "TCPRendezvousServer", "init_tcp_process_group"]

_LENGTH = struct.Struct("!Q")


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise DistributedError("A distributed peer disconnected during a collective.")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_packet(connection: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    metadata = json.dumps(header, separators=(",", ":")).encode()
    connection.sendall(
        _LENGTH.pack(len(metadata)) + metadata + _LENGTH.pack(len(payload)) + payload
    )


def _recv_packet(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    metadata_size = _LENGTH.unpack(_read_exact(connection, _LENGTH.size))[0]
    metadata = json.loads(_read_exact(connection, metadata_size))
    payload_size = _LENGTH.unpack(_read_exact(connection, _LENGTH.size))[0]
    return metadata, _read_exact(connection, payload_size)


class TCPRendezvousServer:
    """Rank-zero reduction server used by :class:`TCPProcessGroup`."""

    def __init__(self, host: str, port: int, world_size: int, *, timeout: float = 60.0) -> None:
        if world_size < 1:
            raise DistributedError("world_size must be positive.")
        self.host, self.port, self.world_size = host, int(port), int(world_size)
        self.timeout = float(timeout)
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> TCPRendezvousServer:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(self.world_size)
        listener.settimeout(self.timeout)
        self.port = listener.getsockname()[1]
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="everyo-rendezvous", daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        peers: dict[int, socket.socket] = {}
        try:
            assert self._listener is not None
            while len(peers) < self.world_size:
                connection, _ = self._listener.accept()
                connection.settimeout(self.timeout)
                header, _ = _recv_packet(connection)
                rank = int(header.get("rank", -1))
                if rank in peers or not 0 <= rank < self.world_size:
                    _send_packet(connection, {"error": "invalid or duplicate rank"})
                    connection.close()
                    continue
                peers[rank] = connection
                _send_packet(connection, {"ok": True})

            sequence = 0
            while True:
                packets = []
                for rank in range(self.world_size):
                    header, payload = _recv_packet(peers[rank])
                    if header.get("op") == "close":
                        return
                    if header.get("op") != "all_reduce_mean" or header.get("sequence") != sequence:
                        raise DistributedError("Ranks called collectives in a different order.")
                    packets.append((header, payload))
                count = int(packets[0][0]["count"])
                if any(int(header["count"]) != count for header, _ in packets):
                    raise DistributedError("All ranks must reduce the same number of elements.")
                arrays = [
                    np.frombuffer(payload, dtype="<f8", count=count) for _, payload in packets
                ]
                result = (
                    np.mean(arrays, axis=0, dtype=np.float64).astype("<f8", copy=False).tobytes()
                )
                for connection in peers.values():
                    _send_packet(connection, {"sequence": sequence, "count": count}, result)
                sequence += 1
        except BaseException as error:  # noqa: BLE001
            self._error = error
        finally:
            for connection in peers.values():
                connection.close()
            if self._listener is not None:
                self._listener.close()

    def close(self) -> None:
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout)


@dataclass
class TCPProcessGroup:
    """Process group whose workers may run on different machines."""

    rank: int
    world_size: int
    connection: socket.socket
    _sequence: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def all_reduce_mean(self, arrays: Sequence[np.ndarray]) -> None:
        sizes = [array.size for array in arrays]
        flat = np.concatenate([np.asarray(array, dtype="<f8").reshape(-1) for array in arrays])
        _send_packet(
            self.connection,
            {"op": "all_reduce_mean", "sequence": self._sequence, "count": flat.size},
            flat.tobytes(),
        )
        header, payload = _recv_packet(self.connection)
        if header.get("sequence") != self._sequence:
            raise DistributedError("Received an out-of-order collective response.")
        reduced = np.frombuffer(payload, dtype="<f8")
        offset = 0
        for array, size in zip(arrays, sizes):
            array[...] = reduced[offset : offset + size].reshape(array.shape).astype(array.dtype)
            offset += size
        self._sequence += 1

    def close(self) -> None:
        try:
            _send_packet(self.connection, {"op": "close"})
        finally:
            self.connection.close()


def init_tcp_process_group(
    host: str,
    port: int,
    *,
    rank: int,
    world_size: int,
    timeout: float = 60.0,
) -> TCPProcessGroup:
    """Connect one worker to a rank-zero rendezvous server."""
    if not 0 <= rank < world_size:
        raise DistributedError(f"rank must be in [0, {world_size}), got {rank}.")
    connection = socket.create_connection((host, int(port)), timeout=timeout)
    connection.settimeout(timeout)
    _send_packet(connection, {"rank": rank, "world_size": world_size})
    response, _ = _recv_packet(connection)
    if not response.get("ok"):
        connection.close()
        raise DistributedError(response.get("error", "Rendezvous failed."))
    return TCPProcessGroup(rank, world_size, connection)
