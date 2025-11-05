#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <work_dir> [config_path]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORK_DIR="$1"
CONFIG_PATH="${2:-${PIPELINE_DIR}/config/pipeline_config.yaml}"
PYTHON_BIN="${PYTHON:-python3}"

export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_openmvs.py" "${WORK_DIR}" --config "${CONFIG_PATH}"
