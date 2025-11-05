#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-/data/gpfs/projects/punim2657/Photogrammetry}"
PIPELINE_DIR="${PIPELINE_DIR_OVERRIDE:-${PROJECT_ROOT}/pipeline}"
CONFIG_PATH="${CONFIG_PATH:-${PIPELINE_DIR}/config/pipeline_config.yaml}"
TARGETS_FILE="${TARGETS_FILE:-}"
PYTHON_BIN="${PYTHON:-python3}"

export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

if [[ -z "${TARGETS_FILE}" ]]; then
  TARGETS_FILE="$("${PYTHON_BIN}" - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
from lib.pipeline_utils import PipelineContext

config_path = sys.argv[1]
ctx = PipelineContext.from_config_path(config_path)
targets_rel = ctx.config.get("targets", {}).get("targets_file", "pipeline/targets.txt")
print(Path(ctx.project_root) / targets_rel)
PY
)"
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. This script must run as part of an array job." >&2
  exit 1
fi

if [[ ! -f "${TARGETS_FILE}" ]]; then
  echo "Targets file not found: ${TARGETS_FILE}" >&2
  exit 1
fi

TREE_PATH="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${TARGETS_FILE}" | tr -d '[:space:]')"
if [[ -z "${TREE_PATH}" ]]; then
  echo "No target found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "=== Slurm Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Tree: ${TREE_PATH}"
echo "Node: $(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
fi

bash "${PIPELINE_DIR}/bin/pipeline_main.sh" "${TREE_PATH}" "${CONFIG_PATH}"
