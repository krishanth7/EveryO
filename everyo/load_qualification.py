"""Bounded authenticated HTTP load probe; not a substitute for deployment soak testing."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LoadEvidence:
    requests: int
    successes: int
    failures: int
    elapsed_seconds: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def benchmark_authenticated_service(
    url: str,
    *,
    token: str,
    inputs: list[list[float]],
    requests: int = 100,
    concurrency: int = 4,
    timeout_seconds: float = 5.0,
) -> LoadEvidence:
    """Probe an explicitly local staging service, with finite request budgets."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("load probe is restricted to explicitly local staging servers")
    if parts.path != "/v1/predict" or parts.query or parts.fragment:
        raise ValueError("expected the /v1/predict endpoint")
    if not token or not 1 <= requests <= 100_000 or not 1 <= concurrency <= 32:
        raise ValueError("invalid authentication or load budget")
    if not 0 < timeout_seconds <= 60:
        raise ValueError("timeout must be within 60 seconds")
    payload = json.dumps({"inputs": inputs}, allow_nan=False).encode("utf-8")
    if len(payload) > 1_000_000:
        raise ValueError("probe payload exceeds 1 MB")

    def probe(_: int) -> tuple[bool, float]:
        start = time.perf_counter()
        request = Request(
            url,
            data=payload,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = json.load(response)
                ok = response.status == 200 and body.get("count") == len(inputs)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            ok = False
        return ok, (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(probe, range(requests)))
    elapsed = max(time.perf_counter() - start, 1e-9)
    latencies = sorted(duration for _, duration in results)
    successes = sum(ok for ok, _ in results)
    return LoadEvidence(
        requests=requests,
        successes=successes,
        failures=requests - successes,
        elapsed_seconds=elapsed,
        p50_ms=_percentile(latencies, 0.5),
        p95_ms=_percentile(latencies, 0.95),
        p99_ms=_percentile(latencies, 0.99),
        requests_per_second=requests / elapsed,
    )
