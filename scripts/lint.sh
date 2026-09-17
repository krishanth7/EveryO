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

# Invoke ruff through the active interpreter rather than whatever binary is
# first on PATH: the project pins an exact ruff version, and a stray global
# install of a different version formats differently from CI.
RUFF=(python -m ruff)
if ! "${RUFF[@]}" --version >/dev/null 2>&1; then
  if command -v ruff >/dev/null 2>&1; then
    RUFF=(ruff)
    echo "warning: using the ruff on PATH ($(ruff --version)); the pinned" >&2
    echo "         version is installed with: pip install -e '.[dev]'" >&2
  else
    echo "error: ruff is not installed in the active environment." >&2
    echo "Install it with: pip install -e '.[dev]'" >&2
    exit 1
  fi
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
  "${RUFF[@]}" check --fix .
  "${RUFF[@]}" format .
else
  "${RUFF[@]}" check .
  "${RUFF[@]}" format --check .
fi

echo "Compiling every module as a final syntax check"
python -m compileall -q everyo
echo "Lint complete."
