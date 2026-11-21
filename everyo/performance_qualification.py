"""Performance and reliability qualification suite for EveryO.

Provides standardized profiling:
- Quick Regression Profile: Fast smoke benchmark.
- Standard Load Profile: Nominal concurrency and throughput evaluation.
- Extended Soak Profile: Sustained execution to detect degradation or leaks.
- Fault Injection Profile: Evaluates resilience under invalid, timed-out, or malformed requests.

Reports percentiles (p50, p90, p95, p99), throughput (RPS), error rates, and SLO compliance.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class PerformanceProfile:
    """Configurable execution parameters for performance qualification."""

    name: str
    concurrency: int
    total_requests: int
    warmup_requests: int = 0
    timeout_seconds: float = 5.0
    fault_injection_rate: float = 0.0


QUICK_REGRESSION_PROFILE = PerformanceProfile(
    name="quick_regression",
    concurrency=2,
    total_requests=20,
    warmup_requests=2,
    timeout_seconds=2.0,
)

STANDARD_LOAD_PROFILE = PerformanceProfile(
    name="standard_load",
    concurrency=4,
    total_requests=100,
    warmup_requests=5,
    timeout_seconds=5.0,
)

EXTENDED_SOAK_PROFILE = PerformanceProfile(
    name="extended_soak",
    concurrency=4,
    total_requests=300,
    warmup_requests=10,
    timeout_seconds=10.0,
)

FAULT_INJECTION_PROFILE = PerformanceProfile(
    name="fault_injection",
    concurrency=4,
    total_requests=50,
    warmup_requests=2,
    timeout_seconds=2.0,
    fault_injection_rate=0.2,
)


@dataclass(frozen=True)
class PerformanceEvidence:
    """Detailed performance qualification record."""

    profile_name: str
    total_requests: int
    successes: int
    failures: int
    error_rate: float
    duration_seconds: float
    throughput_rps: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float
    passed_slo: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_percentile(sorted_values: list[float], percentile: float) -> float:
    """Calculate the given percentile (0.0 to 1.0) using linear interpolation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * weight


def benchmark_serving_performance(
    url: str,
    *,
    token: str,
    profile: PerformanceProfile,
    inputs: list[list[float]],
    target_slo_p95_ms: float = 100.0,
) -> PerformanceEvidence:
    """Execute a structured performance qualification probe against a local service.

    Parameters
    ----------
    url : str
        Endpoint URL (must target loopback / staging).
    token : str
        Authorization bearer token.
    profile : PerformanceProfile
        Execution parameters.
    inputs : list[list[float]]
        Prediction inputs payload.
    target_slo_p95_ms : float
        Service Level Objective for p95 latency.

    Returns
    -------
    PerformanceEvidence
        Comprehensive metrics including percentiles and SLO verdict.
    """
    parts = urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("performance qualification is restricted to loopback staging servers")
    if parts.path != "/v1/predict" or parts.query or parts.fragment:
        raise ValueError("expected the /v1/predict endpoint")
    if profile.concurrency < 1 or profile.total_requests < 1:
        raise ValueError("concurrency and total_requests must be positive")

    valid_payload = json.dumps({"inputs": inputs}, allow_nan=False).encode("utf-8")
    malformed_payload = b'{"inputs": [invalid json'

    rng = random.Random(42)

    # Warmup phase
    if profile.warmup_requests > 0:
        for _ in range(profile.warmup_requests):
            req = Request(
                url,
                data=valid_payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=profile.timeout_seconds) as resp:
                    resp.read()
            except Exception:
                pass

    def probe(idx: int) -> tuple[bool, float]:
        start = time.perf_counter()
        is_fault = (
            profile.fault_injection_rate > 0.0 and rng.random() < profile.fault_injection_rate
        )

        if is_fault:
            fault_type = idx % 3
            if fault_type == 0:
                payload = malformed_payload
                hdr_token = token
                content_type = "application/json"
            elif fault_type == 1:
                payload = valid_payload
                hdr_token = "invalid-token"
                content_type = "application/json"
            else:
                payload = valid_payload
                hdr_token = token
                content_type = "text/plain"
        else:
            payload = valid_payload
            hdr_token = token
            content_type = "application/json"

        req = Request(
            url,
            data=payload,
            headers={"Authorization": f"Bearer {hdr_token}", "Content-Type": content_type},
        )

        ok = False
        try:
            with urlopen(req, timeout=profile.timeout_seconds) as response:
                body = json.load(response)
                ok = response.status == 200 and body.get("count") == len(inputs)
        except HTTPError as err:
            # Under fault injection, deliberate faults returning 400, 401, 415 demonstrate
            # correct server error handling rather than unhandled server crash.
            if is_fault and err.code in {400, 401, 415}:
                ok = True
        except (URLError, TimeoutError, OSError, ValueError):
            ok = False

        duration_ms = (time.perf_counter() - start) * 1000.0
        return ok, duration_ms

    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=profile.concurrency) as pool:
        results = list(pool.map(probe, range(profile.total_requests)))
    total_elapsed = max(time.perf_counter() - start_total, 1e-9)

    latencies = sorted(duration for _, duration in results)
    successes = sum(1 for ok, _ in results if ok)
    failures = profile.total_requests - successes
    error_rate = failures / profile.total_requests

    p50 = calculate_percentile(latencies, 0.50)
    p90 = calculate_percentile(latencies, 0.90)
    p95 = calculate_percentile(latencies, 0.95)
    p99 = calculate_percentile(latencies, 0.99)
    min_ms = latencies[0] if latencies else 0.0
    max_ms = latencies[-1] if latencies else 0.0
    mean_ms = statistics.mean(latencies) if latencies else 0.0
    stddev_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    passed_slo = (p95 <= target_slo_p95_ms) and (error_rate == 0.0)

    return PerformanceEvidence(
        profile_name=profile.name,
        total_requests=profile.total_requests,
        successes=successes,
        failures=failures,
        error_rate=error_rate,
        duration_seconds=total_elapsed,
        throughput_rps=profile.total_requests / total_elapsed,
        p50_ms=p50,
        p90_ms=p90,
        p95_ms=p95,
        p99_ms=p99,
        min_ms=min_ms,
        max_ms=max_ms,
        mean_ms=mean_ms,
        stddev_ms=stddev_ms,
        passed_slo=passed_slo,
    )


def benchmark_cpu_kernel_suite(
    *,
    iterations: int = 100,
    matrix_dim: int = 64,
) -> dict[str, dict[str, float]]:
    """Benchmark raw CPU kernels (matmul, add, relu, autograd) with latency percentiles."""
    import numpy as np

    import everyo as eo

    results: dict[str, dict[str, float]] = {}

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((matrix_dim, matrix_dim)).astype(np.float32)
    b_np = rng.standard_normal((matrix_dim, matrix_dim)).astype(np.float32)

    a = eo.tensor(a_np, device="cpu")
    b = eo.tensor(b_np, device="cpu")

    # Matmul
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = eo.matmul(a, b)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    results["matmul"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    # Add
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = a + b
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    results["add"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    # ReLU
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = eo.relu(a)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    results["relu"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    # Autograd forward + backward
    x_val = rng.standard_normal((matrix_dim, matrix_dim)).astype(np.float32)
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        x = eo.tensor(x_val, device="cpu", requires_grad=True)
        y = eo.sum(eo.matmul(x, x))
        y.backward()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    results["autograd_step"] = {
        "p50_ms": calculate_percentile(latencies, 0.50),
        "p95_ms": calculate_percentile(latencies, 0.95),
        "p99_ms": calculate_percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
    }

    return results
