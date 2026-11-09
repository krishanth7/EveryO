"""CPU-only authenticated staging probe; run with PYTHONPATH=. on localhost."""

from __future__ import annotations

import argparse
import json

from everyo import Linear
from everyo.inference import CPUInferenceRuntime
from everyo.load_qualification import benchmark_authenticated_service
from everyo.serving import CPUModelServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=4)
    arguments = parser.parse_args()
    runtime = CPUInferenceRuntime(Linear(2, 2, seed=1))
    with CPUModelServer(runtime, api_token="ephemeral-staging-token", max_concurrent=8) as server:
        worker = server.run_background()
        try:
            result = benchmark_authenticated_service(
                f"http://127.0.0.1:{server.server_port}/v1/predict",
                token="ephemeral-staging-token",
                inputs=[[1.0, 2.0]],
                requests=arguments.requests,
                concurrency=arguments.concurrency,
            )
            print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
            if result.failures:
                raise SystemExit(1)
        finally:
            server.shutdown()
            worker.join(timeout=5)


if __name__ == "__main__":
    main()
