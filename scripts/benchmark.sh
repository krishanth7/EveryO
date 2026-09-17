#!/usr/bin/env bash
# Run every benchmark and write the results to benchmark_results/.
#
# Usage:
#   ./scripts/benchmark.sh
#   ./scripts/benchmark.sh --quick     # smaller problem sizes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results"
MATMUL_SIZES=(128 256 512 1024)
RELU_SIZES=(100000 1000000 10000000)
EPOCHS=10

for argument in "$@"; do
  case "${argument}" in
    --quick)
      MATMUL_SIZES=(64 128 256)
      RELU_SIZES=(100000 1000000)
      EPOCHS=3
      ;;
    -h|--help)
      sed -n '2,6p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "error: unknown option '${argument}'" >&2
      exit 2
      ;;
  esac
done

if ! python -c 'import everyo' >/dev/null 2>&1; then
  echo "error: everyo is not importable in the active environment." >&2
  echo "Install it with: pip install -e ." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

echo "=== Matrix multiplication ==="
python benchmarks/benchmark_matmul.py \
  --sizes "${MATMUL_SIZES[@]}" \
  --output "${OUTPUT_DIR}/matmul-${TIMESTAMP}.json"

echo
echo "=== ReLU ==="
python benchmarks/benchmark_relu.py \
  --sizes "${RELU_SIZES[@]}" \
  --output "${OUTPUT_DIR}/relu-${TIMESTAMP}.json"

echo
echo "=== Training ==="
python benchmarks/benchmark_training.py \
  --epochs "${EPOCHS}" \
  --output "${OUTPUT_DIR}/training-${TIMESTAMP}.json"

echo
echo "Results written to ${OUTPUT_DIR}"
