#!/usr/bin/env bash
# Remove build artefacts, caches and generated outputs.
#
# Usage:
#   ./scripts/clean.sh          # caches and build output
#   ./scripts/clean.sh --all    # also remove .venv and generated artefacts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

REMOVE_ALL=0
for argument in "$@"; do
  case "${argument}" in
    --all) REMOVE_ALL=1 ;;
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

echo "Removing Python caches"
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.py[co]' -delete 2>/dev/null || true

echo "Removing test and lint caches"
rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov

echo "Removing build output"
rm -rf build dist ./*.egg-info
rm -f everyo/_everyo_cuda*.so everyo/_everyo_cuda*.pyd

if [[ "${REMOVE_ALL}" -eq 1 ]]; then
  echo "Removing generated artefacts and the virtual environment"
  rm -rf artifacts benchmark_results .venv
fi

echo "Clean."
