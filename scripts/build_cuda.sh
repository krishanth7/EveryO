#!/usr/bin/env bash
# Build the optional EveryO CUDA extension.
#
# Requires the NVIDIA CUDA Toolkit (nvcc), CMake 3.18+ and pybind11.
# EveryO works without this step; the extension only adds GPU kernels.
#
# Usage:
#   ./scripts/build_cuda.sh
#   ./scripts/build_cuda.sh --arch 86     # target a specific compute capability
#   ./scripts/build_cuda.sh --clean

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${PROJECT_ROOT}/build/cuda"
ARCH=""

for argument in "$@"; do
  case "${argument}" in
    --clean)
      echo "Removing ${BUILD_DIR}"
      rm -rf "${BUILD_DIR}"
      rm -f "${PROJECT_ROOT}"/everyo/_everyo_cuda*.so
      exit 0
      ;;
    --arch=*) ARCH="${argument#*=}" ;;
    --arch)
      echo "error: use --arch=NN (for example --arch=86)" >&2
      exit 2
      ;;
    -h|--help)
      sed -n '2,11p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "error: unknown option '${argument}'" >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"

missing=0
for tool in cmake nvcc; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: '${tool}' was not found on PATH." >&2
    missing=1
  fi
done
if [[ "${missing}" -eq 1 ]]; then
  echo >&2
  echo "The CUDA extension needs CMake 3.18+ and the NVIDIA CUDA Toolkit." >&2
  echo "EveryO runs on the CPU without it — this step is optional." >&2
  exit 1
fi

if ! python -c 'import pybind11' >/dev/null 2>&1; then
  echo "error: pybind11 is not installed in the active environment." >&2
  echo "Install it with: pip install -e '.[cuda]'" >&2
  exit 1
fi

CMAKE_ARGS=(-S cuda -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release)
if [[ -n "${ARCH}" ]]; then
  CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${ARCH}")
fi

echo "Configuring"
cmake "${CMAKE_ARGS[@]}"

echo "Building"
cmake --build "${BUILD_DIR}" --parallel

echo
echo "Verifying the extension loads"
if python -c 'import everyo; raise SystemExit(0 if everyo.cuda.is_available() else 1)'; then
  python -c 'import everyo, json; print(json.dumps(everyo.cuda.runtime_info(), indent=2, default=str))'
  echo "CUDA extension built successfully."
else
  echo "The extension was built but CUDA is still reported as unavailable." >&2
  python -c 'import everyo; print(everyo.cuda.unavailable_reason())' >&2
  exit 1
fi
