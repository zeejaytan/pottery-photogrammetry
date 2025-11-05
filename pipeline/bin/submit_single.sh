#!/usr/bin/env bash
#SBATCH --account=punim2657
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=colmap_single

set -euo pipefail

TREE_PATH="${1:-}"
if [[ -z "${TREE_PATH}" ]]; then
  echo "ERROR: Usage: $0 <tree_path>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/data/gpfs/projects/punim2657/Photogrammetry}"
PIPELINE_DIR="${PIPELINE_DIR:-${PROJECT_ROOT}/pipeline}"
CONFIG_PATH="${CONFIG_PATH:-${PIPELINE_DIR}/config/pipeline_config.yaml}"

echo "=== Slurm Single Job ==="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Tree: ${TREE_PATH}"
echo "Node: $(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
fi

bash "${PIPELINE_DIR}/bin/pipeline_main.sh" "${TREE_PATH}" "${CONFIG_PATH}"
