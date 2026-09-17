"""Use the optional CUDA backend, with a CPU fallback.

This example runs on any machine. Without a compiled CUDA extension it reports
why CUDA is unavailable and demonstrates the fallback path instead.

Run with:
    python examples/cuda_example.py
"""

from __future__ import annotations

import time

import numpy as np

import everyo as eo


def report_availability() -> bool:
    """Print the CUDA state and return whether kernels can run."""
    print("=== CUDA availability ===")
    info = eo.cuda.runtime_info()
    print(f"  available:    {info['available']}")
    print(f"  device count: {info['device_count']}")
    if not info["available"]:
        print(f"  reason:       {info['reason']}")
        print("\n  To build the extension on a machine with the NVIDIA CUDA Toolkit:")
        print("      ./scripts/build_cuda.sh")
        return False

    print(f"  runtime:      {info.get('runtime_version')}")
    for device in info["devices"]:
        print(
            f"  device {device['index']}: {device['name']} "
            f"(compute {device['compute_capability']}, "
            f"{device['total_memory'] / 1e9:.1f} GB)"
        )
    return True


def demonstrate_fallback() -> None:
    """Show that requesting CUDA never breaks a CPU-only machine."""
    print("\n=== Graceful fallback ===")
    device = eo.device("cuda")  # does not raise; degrades to CPU
    print(f"  eo.device('cuda') resolved to: {device}")

    tensor = eo.tensor([1.0, 2.0, 3.0], device="cuda")
    print(f"  tensor requested on cuda runs on: {tensor.device}")
    print(f"  sum still computes correctly: {eo.sum(tensor).item()}")

    print("\n  Kernels fall back to NumPy automatically:")
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    print(f"    eo.cuda.add(a, b) = {eo.cuda.add(a, b)}")

    print("\n  Strict mode raises instead of silently using the CPU:")
    try:
        eo.cuda.relu(a, strict=True)
    except eo.EveryOCudaError as error:
        print(f"    EveryOCudaError: {str(error)[:90]}...")


def verify_and_benchmark() -> None:
    """Check CUDA results against NumPy and time both paths."""
    print("\n=== Correctness check against NumPy ===")
    rng = np.random.default_rng(0)
    size = 512
    a = rng.normal(size=(size, size)).astype(np.float32)
    b = rng.normal(size=(size, size)).astype(np.float32)

    cpu_result = a @ b
    gpu_result = eo.cuda.matmul(a, b, strict=True)
    difference = float(np.abs(cpu_result - gpu_result).max())
    print(f"  max absolute difference: {difference:.3e}")
    assert difference < 1e-2, "CUDA and NumPy results disagree."

    print("\n=== Timing (includes host/device transfers) ===")
    for label, fn in (
        ("numpy", lambda: a @ b),
        ("cuda ", lambda: eo.cuda.matmul(a, b, strict=True)),
    ):
        fn()  # warm-up
        started = time.perf_counter()
        for _ in range(5):
            fn()
        elapsed = (time.perf_counter() - started) / 5
        print(f"  {label}: {elapsed * 1000:.3f} ms")
    print("\n  Note: at this size the transfer cost can dominate. Never assume the")
    print("  GPU is faster — measure it, which is what benchmarks/ is for.")


def main() -> None:
    """Report availability, then either benchmark CUDA or show the fallback."""
    if report_availability():
        verify_and_benchmark()
    else:
        demonstrate_fallback()


if __name__ == "__main__":
    main()
