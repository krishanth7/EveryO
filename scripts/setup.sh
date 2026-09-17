#!/usr/bin/env bash
# Create a virtual environment and install EveryO in editable mode.
#
# Usage:
#   ./scripts/setup.sh              # core + development dependencies
#   ./scripts/setup.sh --tensorflow # also install the optional TensorFlow backend
#
# Windows users: see the PowerShell commands in README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON:-python3}"
WITH_TENSORFLOW=0

for argument in "$@"; do
  case "${argument}" in
    --tensorflow) WITH_TENSORFLOW=1 ;;
    -h|--help)
      sed -n '2,9p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "error: unknown option '${argument}'" >&2
      exit 2
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: '${PYTHON_BIN}' was not found on PATH." >&2
  echo "Install Python 3.9 or newer, or set PYTHON=/path/to/python." >&2
  exit 1
fi

VERSION="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Using ${PYTHON_BIN} (Python ${VERSION})"
if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "error: EveryO requires Python 3.9 or newer, found ${VERSION}." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment in ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Reusing the existing virtual environment in ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip"
python -m pip install --upgrade pip --quiet

echo "Installing EveryO with development dependencies"
python -m pip install -e ".[dev]"

if [[ "${WITH_TENSORFLOW}" -eq 1 ]]; then
  echo "Installing the optional TensorFlow backend (this download is large)"
  python -m pip install -e ".[tensorflow]"
fi

echo
echo "Done. Activate the environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo
python -m everyo info
