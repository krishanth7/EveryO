#!/usr/bin/env bash
# Lint and format-check the project with ruff.
#
# Usage:
#   ./scripts/lint.sh          # report problems
#   ./scripts/lint.sh --fix    # fix what can be fixed automatically

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

if ! command -v ruff >/dev/null 2>&1; then
  echo "error: ruff was not found on PATH." >&2
  echo "Install it with: pip install -e '.[dev]'" >&2
  exit 1
fi

FIX=0
for argument in "$@"; do
  case "${argument}" in
    --fix) FIX=1 ;;
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

if [[ "${FIX}" -eq 1 ]]; then
  ruff check --fix .
  ruff format .
else
  ruff check .
fi

echo "Compiling every module as a final syntax check"
python -m compileall -q everyo
echo "Lint complete."
