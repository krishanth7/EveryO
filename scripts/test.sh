#!/usr/bin/env bash
# Run the EveryO test suite.
#
# Usage:
#   ./scripts/test.sh                 # the whole suite
#   ./scripts/test.sh --fast          # skip tests marked 'slow'
#   ./scripts/test.sh --coverage      # with a coverage report
#   ./scripts/test.sh tests/test_tensor.py -k shape
#
# Any argument that is not one of the flags above is passed straight to pytest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PYTEST_ARGS=()
COVERAGE=0

for argument in "$@"; do
  case "${argument}" in
    --fast) PYTEST_ARGS+=("-m" "not slow") ;;
    --coverage) COVERAGE=1 ;;
    -h|--help)
      sed -n '2,10p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) PYTEST_ARGS+=("${argument}") ;;
  esac
done

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "error: pytest is not installed in the active environment." >&2
  echo "Install the development extras with: pip install -e '.[dev]'" >&2
  exit 1
fi

if [[ "${COVERAGE}" -eq 1 ]]; then
  if ! python -c 'import pytest_cov' >/dev/null 2>&1; then
    echo "error: pytest-cov is not installed." >&2
    echo "Install it with: pip install pytest-cov" >&2
    exit 1
  fi
  PYTEST_ARGS+=("--cov=everyo" "--cov-report=term-missing")
fi

echo "Running: pytest ${PYTEST_ARGS[*]:-}"
exec python -m pytest "${PYTEST_ARGS[@]}"
