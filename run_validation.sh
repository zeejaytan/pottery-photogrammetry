#!/usr/bin/env bash
#SBATCH --account=punim2657
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=validate_mesh

set -euo pipefail

PROJECT_ROOT="/data/gpfs/projects/punim2657/Photogrammetry"
PIPELINE_DIR="${PROJECT_ROOT}/pipeline"
CONFIG_PATH="${PIPELINE_DIR}/config/pipeline_config.yaml"

cd "${PROJECT_ROOT}"

# Load modules
module load Python/3.10.4

# Set PYTHONPATH
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

# Run validation
python pipeline/bin/split_and_validate.py --config "${CONFIG_PATH}" --tree 16062025
