"""The ``everyo`` command line interface.

Commands:

``everyo info``
    Print versions and which backends are available.
``everyo doctor``
    Diagnose an installation and suggest fixes.
``everyo benchmark``
    Run a matmul benchmark across the available backends.
``everyo test``
    Run the bundled test suite (requires pytest).
``everyo demo``
    Train the bundled digit classifier end to end.

The CLI collects and transmits nothing.  Every command runs locally.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from everyo._logging import configure_logging
from everyo.version import __version__

__all__ = ["main", "build_parser"]

_OK = "ok"
_MISSING = "not available"


def _numpy_version() -> str:
    import numpy as np

    return np.__version__


def _collect_info() -> dict[str, Any]:
    """Gather the facts shown by ``everyo info``."""
    from everyo.backends import tensorflow_backend
    from everyo.core.device import device
    from everyo.cuda.availability import runtime_info
    from everyo.visualization import _backend as plotting

    cuda_info = runtime_info()
    return {
        "everyo_version": __version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "numpy": _numpy_version(),
        "matplotlib": plotting.is_available(),
        "tensorflow": {
            "available": tensorflow_backend.is_available(),
            "reason": tensorflow_backend.unavailable_reason(),
            "gpu": tensorflow_backend.gpu_available(),
        },
        "cuda": cuda_info,
        "default_device": str(device("auto")),
    }


def _command_info(args: argparse.Namespace) -> int:
    """Print environment and backend information."""
    info = _collect_info()
    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return 0

    tf = info["tensorflow"]
    cuda = info["cuda"]
    lines = [
        "EveryO",
        f"Version:      {info['everyo_version']}",
        f"Python:       {info['python_version']}",
        f"Platform:     {info['platform']}",
        f"NumPy:        {info['numpy']}",
        f"Matplotlib:   {_OK if info['matplotlib'] else _MISSING}",
        f"TensorFlow:   {_OK if tf['available'] else _MISSING}"
        + (" (GPU detected)" if tf["gpu"] else ""),
        f"CUDA:         {_OK if cuda['available'] else _MISSING}"
        + (f" ({cuda['device_count']} device(s))" if cuda["available"] else ""),
        f"Device:       {info['default_device']}",
    ]
    print("\n".join(lines))
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    """Diagnose the installation and report problems."""
    from everyo.backends import tensorflow_backend
    from everyo.cuda.availability import unavailable_reason as cuda_reason
    from everyo.visualization import _backend as plotting

    checks: list[tuple[str, bool, str]] = []

    try:
        import numpy as np

        checks.append(("NumPy", True, f"version {np.__version__}"))
    except ImportError as exc:  # pragma: no cover - numpy is required
        checks.append(("NumPy", False, f"missing: {exc}. Run 'pip install numpy'."))

    checks.append(
        (
            "Matplotlib",
            plotting.is_available(),
            "available"
            if plotting.is_available()
            else "missing. Charts are disabled; run 'pip install matplotlib'.",
        )
    )

    # Core numerics must actually work, not merely import.
    try:
        import everyo as eo

        value = eo.tensor([2.0], requires_grad=True)
        (value * value).backward()
        gradient_ok = abs(float(value.grad[0]) - 4.0) < 1e-6
        checks.append(
            (
                "Autograd",
                gradient_ok,
                "d(x^2)/dx = 4 at x = 2"
                if gradient_ok
                else "gradient check failed; please open an issue with 'everyo info' output.",
            )
        )
    except Exception as exc:  # pragma: no cover - catastrophic install problem
        checks.append(("Autograd", False, f"failed: {exc}"))

    tf_available = tensorflow_backend.is_available()
    checks.append(
        (
            "TensorFlow (optional)",
            True,
            f"available, version {tensorflow_backend.get_tensorflow().__version__}"
            if tf_available
            else f"not available — {tensorflow_backend.unavailable_reason()}",
        )
    )

    from everyo.cuda.availability import is_available as cuda_available

    checks.append(
        (
            "CUDA (optional)",
            True,
            "available" if cuda_available() else f"not available — {cuda_reason()}",
        )
    )

    try:
        import pytest  # noqa: F401

        checks.append(("pytest (optional)", True, "available; run 'everyo test'"))
    except ImportError:
        checks.append(("pytest (optional)", True, "not installed; run 'pip install everyo[dev]'"))

    width = max(len(name) for name, _, _ in checks)
    failures = 0
    print("EveryO installation report")
    print("-" * (width + 40))
    for name, healthy, detail in checks:
        marker = "PASS" if healthy else "FAIL"
        failures += 0 if healthy else 1
        print(f"{marker}  {name.ljust(width)}  {detail}")
    print("-" * (width + 40))

    if failures:
        print(f"\n{failures} required check(s) failed. EveryO may not work correctly.")
        return 1
    print("\nAll required checks passed. Optional components are reported above.")
    return 0


def _command_benchmark(args: argparse.Namespace) -> int:
    """Run the matmul benchmark across available backends."""
    from everyo.benchmarking import run_matmul_benchmark

    results = run_matmul_benchmark(sizes=args.sizes, repeats=args.repeats)
    _report_benchmark(results, args)
    return 0


def _report_benchmark(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        header = f"{'backend':<14}{'size':>8}{'seconds':>14}{'GFLOP/s':>12}"
        print(header)
        print("-" * len(header))
        for row in results:
            print(
                f"{row['backend']:<14}{row['size']:>8}{row['seconds']:>14.6f}{row['gflops']:>12.2f}"
            )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote results to {target}")


def _command_test(args: argparse.Namespace) -> int:
    """Run the bundled test suite."""
    try:
        import pytest
    except ImportError:
        print(
            "pytest is not installed. Install the development extras with:\n"
            "    pip install -e '.[dev]'",
            file=sys.stderr,
        )
        return 1
    target = args.path or str(Path(__file__).resolve().parents[2] / "tests")
    if not Path(target).exists():
        print(
            f"No test directory found at {target}. The test suite ships with the "
            "source repository; clone it from GitHub to run the tests.",
            file=sys.stderr,
        )
        return 1
    return int(pytest.main([target, "-q"]))


def _command_demo(args: argparse.Namespace) -> int:
    """Train the bundled digit classifier end to end."""
    import everyo as eo
    from everyo.datasets import load_digits

    features, labels = load_digits(samples_per_class=args.samples, seed=0)
    x_train, x_test, y_train, y_test = eo.stratified_split(features, labels, test_size=0.2, seed=0)

    model = eo.Sequential(
        eo.Linear(features.shape[1], 64, seed=0),
        eo.ReLU(),
        eo.Linear(64, 10, seed=1),
    )
    trainer = eo.Trainer(
        model,
        eo.Adam(model.parameters(), lr=0.01),
        eo.CrossEntropyLoss(),
        metrics=["accuracy"],
    )
    trainer.fit(
        eo.DataLoader(eo.ArrayDataset(x_train, y_train), batch_size=32, shuffle=True, seed=0),
        epochs=args.epochs,
        validation_loader=eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64),
        verbose=True,
    )
    results = trainer.evaluate(eo.DataLoader(eo.ArrayDataset(x_test, y_test), batch_size=64))
    print(f"\nTest loss: {results['loss']:.4f}  Test accuracy: {results['accuracy']:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``everyo`` command."""
    parser = argparse.ArgumentParser(
        prog="everyo",
        description="EveryO — neural computing and experimentation framework.",
    )
    parser.add_argument("--version", action="version", version=f"EveryO {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Enable EveryO logging at this level.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show versions and available backends.")
    info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    info.set_defaults(handler=_command_info)

    doctor = subparsers.add_parser("doctor", help="Diagnose installation problems.")
    doctor.set_defaults(handler=_command_doctor)

    benchmark = subparsers.add_parser(
        "benchmark", help="Benchmark matrix multiplication across backends."
    )
    benchmark.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Square matrix sizes to measure.",
    )
    benchmark.add_argument(
        "--repeats", type=int, default=3, help="Timed repetitions per measurement."
    )
    benchmark.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    benchmark.add_argument("--output", default=None, help="Write results to this JSON file.")
    benchmark.set_defaults(handler=_command_benchmark)

    test = subparsers.add_parser("test", help="Run the bundled test suite.")
    test.add_argument("path", nargs="?", default=None, help="Test path (default: ./tests).")
    test.set_defaults(handler=_command_test)

    demo = subparsers.add_parser("demo", help="Train the bundled digit classifier.")
    demo.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    demo.add_argument("--samples", type=int, default=200, help="Samples generated per digit class.")
    demo.set_defaults(handler=_command_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``everyo`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.log_level:
        configure_logging(args.log_level)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
